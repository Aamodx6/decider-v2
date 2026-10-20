# ARCH_V2.md — decider-v2, corrected and shortened

What `ARCH_REVIEW.md` changed relative to `docs/arch.md`, and what I am actually going to build.
Everything here is either measured or explicitly flagged.

Two sentences of context, because they change how the results table should be read:

* The published 74.7 held-out number **is** comparable to what we will measure: my encoder scores the
  published checkpoint at **77.24** on a 5,000-question `default`-config sample and **76.30** on a
  4,000-question `all`-config sample, so the two test distributions are within a point. What is
  *not* comparable is the training recipe (the published run saw more question presentations than one
  epoch of any public config contains — `ARCH_REVIEW.md` §1.9), so the published number is context
  and the baseline replica is the control.
* At n = 4,000 and p ≈ 0.76 the 95% interval is ±1.35 points. A v2-vs-baseline delta below ~3 points
  is noise, whatever the table looks like. I got this wrong myself mid-build: a 1,000-question
  reference draw read as 68.70 and I concluded the `all` config was intrinsically harder than
  `default`, which the 4,000-question measurement disproved (`ARCH_REVIEW.md` §5.4). The real run
  does not rely on the caution at all — it evaluates the **full 80,019-question test split**, where
  the interval is ±0.30 points, and runs two seeds per arm.
* `avbiswas/bev-decision('all')` has **no licence** (`license: unknown`, and the card says so in
  those words). Weights trained on it are research-only until every upstream is audited. That does
  not block the experiments and does not affect any number below.

---

## 0. What v2 is, in one table

| | baseline replica | decider-v2 |
|---|---|---|
| init | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-0.6B` (hard rule 7: **not** bev-decider weights) |
| backbone | first **20** of 28 layers | first **20** of 28 layers |
| final norm | applied after layer 19 | applied after layer 19 |
| LoRA | r=8, α=16, q/k/v/o, layers **8–19** | r=16, α=32, q/k/v/o **+ gate/up/down**, layers **0–19** |
| head | `ChoiceHead` replica (bev) | Perceiver ctx summarizer + 2 position-free option blocks |
| loss | plain CE | CE + label smoothing 0.05 (valid-options only); `score` adds λ·EMD, λ=0.5 |
| option order | dataset order | shuffled every epoch |
| calibration | none in the training recipe | per-type temperature, fitted on a held-out calibration split |
| serialisation | bev | **identical to bev** (see §5) |

Everything else — data, split, steps, batch, seed, tokenizer, truncation limits, collate, eval code —
is the same code path for both arms. The only differences are the four rows marked above plus
calibration, which is applied at evaluation time to both arms.

Measured parameter counts (`decider_v2/_paramcount_out.txt`, real Qwen3 shapes):

| component | baseline | v2 |
|---|---:|---:|
| backbone (20 layers, incl. tied embeddings) | 470,202,368 | 470,202,368 |
| LoRA (trainable) | 983,040 | **7,208,960** |
| head (trainable) | 7,364,608 | **10,779,137** |
| **total** | **478,550,016** | **488,190,465** |
| **trainable** | 8,347,648 | **17,988,097** |

470,202,368 + 7,364,608 = 477,566,976 = the published `bev-decider-0.4B` total, and the LoRA is
already merged there. The replica is structurally exact; "0.4B" is really 0.478B.

---

## Tier 1 — always implemented. These define v2.

### 1.1 Perceiver-style context summarizer (prefix only)

The head has no direct read of the state. arch.md's idea is right and the reason is sharper than
arch.md states: the *only* path from the state to the decision runs through the option vectors, and
those are forced by the isolating mask to be a function of (state, question, that one option). A
separate read of the prefix is a genuinely new channel into the head, not a reshape of an existing
one.

```python
# prefix_h [B, P, 1024]  prefix_pad [B, P]  (True = padding)
c     = proj_ctx(prefix_h)                       # 1024 -> 512
ctx,_ = ctx_attn(ctx_q.expand(B,-1,-1), c, c, key_padding_mask=prefix_pad)   # 8 x 512
```

`n_ctx = 8`, 8 heads, dropout 0. Costs 1,050,880 + 1,052,672 params.

**Slice to the prefix, never the whole sequence.** Passing the option blocks too is exactly
permutation-invariant in real arithmetic and measurably *not* in fp32 — see `ARCH_REVIEW.md` §6.1
(Δ = 1.96e-03 before the fix, 8.94e-08 after). `DeciderV2.forward` slices with `prefix_len`, and
`tests.py` (b2) is the regression.

### 1.2 Two position-free option blocks

```python
class HeadBlockV2(nn.Module):        # d=512, h=8
    n1 LayerNorm ; sa  = MultiheadAttention(512, 8)   # over OPTIONS only, no positions
    n2 LayerNorm ; ca  = MultiheadAttention(512, 8)   # queries = options, keys/values = ctx
    n3 LayerNorm ; ff  = Linear(512,2048) GELU Linear(2048,512)
    forward: x = x + sa(n1(x), n1(x), n1(x), key_padding_mask=opt_pad)
             x = x + ca(n2(x), ctx,      ctx)
             x = x + ff(n3(x))
```

No positional encoding anywhere over the option axis, and `key_padding_mask` is over the *option*
axis only, so the block is permutation-equivariant. Two independent MHA modules per block (not the
fused `in_proj` of arch.md §7, which would share the option and context projections) — 2 × 2 ×
1,052,672 params.

Output: `Linear(512, 1)` per option. arch.md's `out = Sequential(LayerNorm, Linear(d,1))` is kept.

### 1.3 Type embedding, rank embedding, no null option by default

* `type_emb = nn.Embedding(3, 512)` — 1,536 params. `choice`=0, `noul`=1, `score`=2.
* `rank_mlp: Linear(1,512) → GELU → Linear(512,512)` — 263,680 params, applied **only** to `score`.
* The rank is a property of the **level**, carried alongside the option text and permuted with it.
  It is 0 for every `choice` and `noul` option, which is what stops it becoming a positional
  feature. `tests.py` (b2) covers K = 3, 5, 7.
* **Null option: `--use_null 0`.** 23.44% of `choice` questions already contain an abstention-like
  key and 6.03% of all `choice` questions are *answered* with one (`ARCH_REVIEW.md` §4.2). A learned
  null duplicates that, steals probability mass on the 76.6% that have no abstention option, and
  invents a response field the Jev contract does not have. The flag exists (+512 params).

### 1.4 LoRA r=16, α=32, attention + MLP, all 20 kept layers

| | modules | params |
|---|---:|---:|
| baseline r=8, q/k/v/o, layers 8–19 | 48 | 983,040 |
| **v2 r=16, q/k/v/o + gate/up/down, layers 0–19** | **140** | **7,208,960** |

7.3× the trainable backbone parameters. Measured cost on the RTX 4050, same batch, same budget:
**8.6 q/s (baseline) → 6.5 q/s (v2)**, i.e. 1.32× slower, peak VRAM 2.87 → 3.11 GiB. That is the
price of the change and it is the honest number to plan against.

`lora_last_k_layers 0` = frozen backbone (head-only ablation); `-1` = all kept layers.

### 1.5 Loss: smoothed CE (valid options only) + EMD on `score` + per-type weighting

```python
loss = smoothed_cross_entropy(logits, labels, valid, eps)          # eps = 0.05
loss = loss + 0.5 * emd                                            # score rows only
loss = (w[type] * loss).sum() / w[type].sum()                      # w default {0:1, 1:1, 2:1}
```

* **Smoothed CE over valid options only** — mandatory, not optional. torch's version spreads the
  smoothing over all `C` columns including padding, which on a 2-option `noul` row inside a
  17-option batch returns **2.5e+07** instead of 0.71 (`tests.py` (c); 45.35% of questions are
  2-option, so this is the common case, not a corner case). The corrected form is provably
  identical to torch's when no column is padded, and gives padding columns exactly zero gradient.
* **EMD** `Σ_i |cdf_p(i) − cdf_t(i)| / K`, restricted to the `K` real levels. Unit-tested to be
  exactly `0` for a perfect prediction and exactly `1/K` for a one-off error. λ = 0.5.
* **Per-type weighting** is wired (`--type_weights '0:1,1:1,2:1'`) and defaults to uniform. The
  mixture is already close to balanced (44.75 / 41.52 / 13.73), so re-weighting is an ablation, not
  a default. Uniform is the fair choice against a baseline that has no weighting at all.

### 1.6 Option order shuffled every epoch

`DecisionDataset.set_epoch(e)` re-permutes each example's options with
`random.Random((seed, epoch, idx)).shuffle`. Deterministic, resumable, no cost when
`--shuffle_options 0`. It is an invariance-preserving augmentation *and* it is what surfaces an
invariance bug: with shuffling on, the first version of the rank embedding lost 1.96e-03 (§1.3).

### 1.7 Per-type temperature + ECE

`calibrate.py` fits one `T_type` per question type by minimising NLL on the **calibration split**
(21,127 questions, carved from train before any training, stratified by type × domain). `evaluate.py`
reports ECE (10 bins, top-label) before and after, overall and per type, and per-domain accuracy.

Gate: **do not get worse than the reference.** Reference ECE measured: 0.0122 (`default`, n=5,000),
**0.0144** (`all`, n=4,000). arch.md's 0.065 is from sysone-bench, a different distribution, and is
not the number to beat here.

### 1.8 Order-invariance hard gate

fp32, many shuffles, `k` clamped to 3–12, gate `max|Δp| ≤ 1e-5`. Measured:

| model | max Δp |
|---|---:|
| 2-layer random-init v2, `bev` / `special` mode | 5.96e-08 |
| `score` questions, K = 3 / 5 / 7 | 5.96e-08 / 4.47e-08 / 4.47e-08 |
| trained 2-layer v2 on real data, 24 × 8 shuffles | 8.94e-08 |
| **published `bev-decider-0.4B`, 20 layers, 24 × 8** | **3.16e-06 – 3.70e-06** |

The reference itself sits at ~3e-6, not ~1e-8. That is fp32 non-associativity (permuting options
changes GEMM tiling), not a leak: dropping an option from the batch moves every surviving option's
hidden state by ≤7.2e-07. **Expect 1e-6 at 20 layers; a result near 1e-5 is a real finding, a result
near 1e-8 is a bug in the check.**

---

## Tier 2 — implemented behind config flags, run only as ablations

Each is a flag on the same `train.py`. One change at a time, same budget.

| flag | what | why Tier 2 |
|---|---|---|
| `--lora_include_mlp 0 --lora_last_k_layers 12 --lora_r 8` (`--lora_alpha 16`) | `v2_lora_bev`: keep the v2 **head** and every v2 loss/optimisation change, but hand the LoRA back to bev-decider's own shape | **The load-bearing ablation.** Tier 1 changes the LoRA (r=8→16, attn-only→attn+MLP, 12→20 layers) *and* the head *and* the loss, so a v2-vs-baseline delta cannot say which one did the work. This isolates it: if the gain survives, the head and loss carry it and the wide LoRA does not; if it vanishes, the LoRA is doing the work. `lora_alpha` moves 32→16 with `r` so α/r stays 2.0, matching both v2 (32/16) and the baseline (16/8) — changing r alone would silently change the effective LoRA scale. |
| `--num_layers 24` | depth 24 instead of 20 | +62,923,776 params (533,126,144 total). Reference target unknown. Note bev's LoRA indices are absolute, so 28 would change what "layers 8–19" means. |
| `--pool mean` | mean-pool the option instead of last-content-token | The reference's pooling is last-content-token; `opt_end` is only available in `--mode special`. This is the honest version of arch.md's ablation. |
| `--mode special --pool opt_end\|mean` | four new special tokens `<state> <q> <opt> <opt_end>`, embedding resized and the new rows initialised from the embedding mean | `specialize.serialize.py::resize_and_init_embeddings`. Keeping `bev` as the default keeps serialisation identical across arms so the head comparison is not confounded. |
| `--use_derived 1` | deterministic DERIVED pre-pass: dates (regex + `dateparser`), money, units (`pint`), appended to the state before tokenisation | `normalize.py`. Emits only what it finds, and the trainer drops the block 15% of the time (`--derived_dropout`) so the model degrades gracefully when the pre-pass finds nothing. **No synthetic data.** The mixture already contains `numeric_temporal` (19,806 rows, 43 calculation types, labels computed in code) and `counterfactual_15k` (7,500 label-flipping pairs) — generating more would break rule 3 (identical data in both arms). |
| `--use_null 1` | learned null option (+512 params) | Default off; see §1.3. |
| `--resume <ckpt> --lr_lora 5e-5 --lr_head 2.5e-4` | stage 2: hard-negative fine-tune, lr ÷ 4 | The stage-2 entry point is `--resume`, already implemented (optimiser + scheduler + step counter restored from `optim.pt`). Mining the errors from stage-1 train-dev is a data step, not a code step; the flag is the config half. |

All of these launch behind `RUN_ABLATIONS` in notebook cell 14, at seed 0 and the same `MAX_STEPS`
budget as the arms they are compared against. None has been run — see `LOCAL_VERIFICATION.md` §0.

---

## Launch-time gates

Two checks run inside every training container, before the first step, and abort the job on failure
(`train.py --selftest 1 --env_check 1`, both on by default). They exist because the two ways this
experiment silently produces meaningless numbers are both cheap to detect up front.

| gate | what it checks | why |
|---|---|---|
| `ENV_CHECK` | the exact pins in `decider_v2/requirements.txt`, enforced by `decider_v2/environment.py` | the 4-D additive attention mask is returned as-is only from transformers 5.x, so a different transformers is a **correctness** bug, not a warning. The `huggingface_hub==1.33.0` pin is load-bearing: transformers requires `<2.0` and the package is now at 2.2.0, so an unpinned install does not import at all |
| `SELFTEST` | packed-vs-separate equivalence and option-order invariance, on a 2-layer random-init model (~10 s, offline, no data) | catches a broken mask / position-id guarantee in the container rather than after 12k steps |

The container is **PyTorch 2.7.1 / py312**, verified against the SDK's own
`image_uri_config/pytorch.json` (2.5.1 -> py311, 2.6.0 -> py312, 2.7.1 -> py312). Python >= 3.11 is
forced by `numpy 2.3.5` (`requires-python >=3.11`). Cell 1 validates the pairing before submitting
anything. An earlier revision asked for 2.6.0 + py311, which is not a valid pairing.

**Decision rule.** `RESULTS.md` reports the v2 - baseline delta beside the two things that can
hide it, and prints one of three verdicts:

```
(a) the paired-bootstrap 95% CI for the delta excludes 0
(b) |delta| > max(baseline seed spread, v2 seed spread)
(c) |delta| >= 1.0 point
(d) at least 3 seeds per arm

(a) and (b) and (c) and (d)  ->  SEPARABLE
(a) and (b) and (c), 2 seeds  ->  PROVISIONAL
anything else                 ->  NOT SEPARABLE
```

The bootstrap resamples the test questions 10,000 times, pairing each baseline question with the
same-index v2 question, and averages the per-question difference in correctness. It covers **test
sampling only**; (b) covers training-run variance, which is a different quantity. (c) exists because
at n = 80,019 a 0.4-point delta is statistically unambiguous and still not worth acting on.

(d) is why two seeds cannot return `SEPARABLE`. The same configuration, same data and same seeds was
run three times on the development machine and produced seed spreads of **2.52, 0.05 and 0.35
points** (`LOCAL_VERIFICATION.md` §10), with the delta moving by 1.4 points between runs - GPU
training is not bit-deterministic, so even a fixed seed does not give a fixed model. All three had
bootstrap CIs excluding 0, and all three would have been reported as results under the previous
two-seed rule. With *one* seed the spread is 0 by construction, so (b) is vacuous and the verdict is
forced to `NOT SEPARABLE`.

Implemented once in `decider_v2/decide.py` and unit-tested branch by branch in `tests.py` §(e),
including both real measured configurations. The notebook calls that module rather than
re-deriving the rule, so the notebook and the tests cannot disagree about what `SEPARABLE` means.

---


---

## Tier 3 — documented, not built

Each of these is either out of reach with this dataset or is a serving/export concern, and I would
rather say so than ship a stub.

* **Multilingual augmentation.** `language: [en]`, no multilingual data in the mixture. The
  multilingual-intent gate in arch.md §14 is unreachable without it, and the gate should be deleted
  rather than kept as an aspiration.
* **Soft-label distillation.** 8,131 questions carry `label_probs`. It is a real signal, but it is a
  second stage over a *teacher*, and the teacher is not available here.
* **4k context / length curriculum.** p99 state length is 958 tokens and only 7.75% of train rows
  exceed 1,024. A curriculum spends most of the budget on a 1% tail.
* **KV-prefix sharing across questions on one state.** Needs a cache-aware forward; `use_cache=False`
  is structural. Real but separable work.
* **v2-M (Qwen3-1.7B).** Requires re-measuring every budget decision at 3× the size.
* **int8 / ONNX / MLX export and the API server.** Export parity is a separate project; the order
  gate in `tests.py` is the thing that has to survive it.

---

## Tensor shapes and the forward pass

```
input_ids        [B, L]        int64     L = prefix + options + 4 answer tokens, right-padded
position_ids     [B, L]        int64     options all restart at P; answer starts at max(pos)+1
attention_mask   [B, 1, L, L]  float32   additive, 0 / -inf
option_read_idx  [B, C]        int64     C = max options in batch (2..17 measured)
option_mask      [B, C]        bool      True = real option
option_seg       [B, L]        int64     option index per token, -1 elsewhere (mean pooling)
ranks            [B, C]        float32   level rank in [0,1]; 0 for choice/noul
prefix_len       [B]           int64     P per row
answer_end_idx   [B]           int64
task_type        [B]           int64     0 choice / 1 noul / 2 score
labels           [B]           int64
```

```
backbone   Qwen3Model(20 layers, LoRA)      ->  hidden [B, L, 1024]
option_h   hidden[b, option_read_idx]        [B, C, 1024]   (or segment_mean for pool=mean)
answer_h   hidden[b, answer_end_idx]         [B, 1024]
prefix_h   hidden[:, :max(P)]                [B, P, 1024]

head v1 (bev replica, fp32):
  x        cat(task_emb, option_h, answer_h)          [B, C+2, 1024]
  2 x AttentionBlock over [B, C+2, 1024]
  logits   (answer_proj x[:, -1] · choice_proj x[:, 1:-1]) / sqrt(512)   [B, C]

head v2 (fp32):
  x        proj_opt(option_h) + type_emb(t)[:, None]  [B, C, 512]
  x        x + rank_mlp(ranks)[score only]            [B, C, 512]
  ctx      ctx_attn(ctx_q, proj_ctx(prefix_h))         [B, 8, 512]
  2 x HeadBlockV2                                      [B, C, 512]
  logits   out(x).squeeze(-1)                          [B, C],  -inf on padding
```

The mask is built by broadcasting only (`serialize.build_batch`); block ids are 0 = prefix,
1..K = option `i`, K+1 = answer, −1 = padding.

---

## Evaluation plan

One `test` Parquet shard, built once with the fixed `SPLIT_SEED`, uploaded once, and every arm,
every seed and the reference are evaluated on that exact file. A **real run uses the full 80,019
published test questions — never a subsample**; the 4,000-question shard is `SMOKE`-only and cell 3
asserts which one you got. (Two draws of 400 questions from the same split differ by 7.5 accuracy
points, so this is not a formality — `ARCH_REVIEW.md` §5.4.)

| # | what | output |
|---|---|---|
| 1 | reference `avbiswas/bev-decider-0.4B` on our `test` | the number to beat, on our split |
| 2 | reference on a 5,000-question sample of the **`default`** config test | validates my encoder against the published 74.7; the two configs turn out to be within ~1 point, so the published number is a fair target |
| 3 | baseline replica, one job per seed | identical budget, identical data |
| 4 | v2, one job per seed | identical budget, identical data |
| 5 | delta (4) − (3), seed-matched then averaged, per type and per domain | the only honest claim of improvement |
| 6 | calibration before/after, per type | ECE and NLL on the full test split, temperatures from the calib split only |
| 7 | order-invariance gate on the trained v2 weights | must be ≤ 1e-5 |
| 8 | JevBench 231 public tasks and sysone-bench v2.1.0 (1,190 cases = 1,240 evaluation decisions) | **both adapters written** (`decider_v2/bench/`), each graded by the upstream harness's own scorer; see `LOCAL_VERIFICATION.md` §8 for the measured reference rows |
| 9 | Tier-2 ablations, one flag at a time, same budget, seed 0 | separate rows |

Every metric is written to `metrics_<tag>.json` plus a per-example CSV, so any row can be recomputed
from the predictions without re-running the model.

**Explicitly not claimed.** No statement anywhere in this repo that v2 beats bev-decider or
approaches Jev. The only accuracy numbers produced so far are from 100-step smoke runs on a
4,000-question shard, which are a plumbing check and nothing else: baseline **45.80**, v2 **50.63**,
reference **76.30** on the same file. At 100 steps neither arm has converged (v2's fitted
temperatures are 0.81 / 0.56 / 0.74 against the reference's 0.97 / 1.06 / 0.89), and the +4.82-point
gap between two undertrained models on 1/22nd of one epoch is inside the noise floor of §5.4 and
must not be quoted anywhere. The reference row is the only one here with predictive meaning, and it
is the number the real run has to beat.

---

## Budget and cost

Default, matching the **940K budget the published model card claims**:
**14,688 steps × batch 64 = 940,032 question presentations** = 2.42 epochs of the 388,748-question
carved train split. The budget is written in *presentations* (`TARGET_PRESENTATIONS = 940_000`) and
`MAX_STEPS` is derived as `ceil(940,000 / 64)`, because presentations are the unit the card uses and
the unit that stays comparable if the batch size changes.

Why 940K and not the card's other number: the card says both "12,000 steps at batch size 64"
(= 768,000 presentations) and "cross-entropy on about 940K typed questions". Those disagree, and
neither is reachable as distinct data — `bev-decision('all')` train is 422,551 typed questions, of
which 388,748 survive the val/calib holdout (§1.9 flags the published run as CANNOT VERIFY). So
"940K" can only mean question *presentations*, i.e. repeats over the distinct questions, and
matching it is the closest anyone can get to the published recipe.

`--max_steps 14688` is the faithful reading; `--epochs 2` is only a fallback ceiling, since
`train.py` computes `total_steps = max_steps if max_steps > 0 else steps_per_epoch * epochs`. At
batch 32 × 2 that is 6,074 steps/epoch, so 14,688 steps is 2.42 epochs and `max_steps` is what makes
the budget exact.

Measured on the RTX 4050 Laptop (6 GiB, bf16 autocast, gradient checkpointing on,
`decider_v2/_gpu_*_out.txt`) — the **floor**, i.e. the slowest plausible instance:

| | steady-state throughput | peak VRAM | 14,688-step wall time |
|---|---:|---:|---:|
| baseline replica, batch 8 × 2 | 8.6 q/s | 2.87 GiB | 30.4 h |
| v2, batch 8 × 2 | 6.5 q/s | 3.11 GiB | 40.2 h |

This card cannot do the real run (and 6 GiB cannot hold batch 16 × 1,024-token states — micro-batch
8 peaked at 3.22 GiB, micro-batch 16 OOMs or spills to WDDM shared memory at 0.63 q/s). The notebook
measures throughput on **your** SageMaker instance in cell 6 and prints the extrapolated hours and
cost from that, at runtime, with pricing fetched rather than hard-coded. The numbers above are the
floor to sanity-check it against, not the estimate.

**`max_runtime` must exceed the estimate or the job is destroyed.** SageMaker hard-kills a training
job at `max_runtime` and discards the work; at this budget even a 3×-faster A10G puts v2 at ~13 h,
so the 12 h this notebook used to hardcode would have killed a real run. It now defaults to
**`MAX_RUNTIME_HOURS = 72`**, and cell 6b compares the budget against the throughput measured on your
own instance and **raises `SystemExit` rather than launching** if either arm would be killed.

LoRA on all layers including MLPs is 1.32× slower than bev's attention-only adapter. That is the
single biggest cost lever, and `--lora_include_mlp 0 --lora_last_k_layers 12 --lora_r 8` recovers it
if the extra capacity does not pay for itself.