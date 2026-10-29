# ARCH_REVIEW.md — what bev-decider-0.4B actually does, and what that does to arch.md

Every claim below is backed by a file and line in something I read or ran. Sources:

| ref | what it is |
|---|---|
| `ref/bev-train/` | `bev-train` @ `a58c551`, the open training code for bev-decider |
| `ref/bev-decider/` | `bev-decider` @ `main`, the open Apache-2.0 inference package |
| `ref/hf/bev-decider-config.json` | the published checkpoint's own `config.json` |
| `ref/hf/bev-decider-backbone_config.json` | the published checkpoint's 20-layer Qwen3 config |
| `ref/hf/bev-decider-README.md` | the model card (published benchmark numbers) |
| `ref/hf/bev-decision-README.md` | the dataset card |
| `recon/*.txt` | raw output of the recon scripts in `recon/` |
| `decider_v2/_tests_out.txt`, `_tests_gpu_out.txt` | `tests.py` output, CPU and RTX 4050 |
| `decider_v2/_paramcount_out.txt` | `paramcount.py` output |
| `ref/jevbench/`, `ref/sysone-bench/` | cloned benchmark repos (Phase 0.3) |

Numbers I could not verify are marked **CANNOT VERIFY**.

---

## 1. Ground truth of bev-decider, from the code

### 1.1 Backbone truncation, and the final norm

`bev-train/network.py:112-115`

```python
def load_backbone(model_name, num_layers=None):
    overrides = {} if num_layers is None else {"num_hidden_layers": num_layers}
    return Qwen3Model.from_pretrained(model_name, dtype=torch.float32, **overrides)
```

The backbone is `Qwen3Model` (no LM head), `float32`, with `num_hidden_layers` overridden at
`from_pretrained` time. **`Qwen3Model.norm` is a property of `Qwen3Model`, not of the last decoder
layer**, so the final RMSNorm *is* applied to the output of the last kept layer. Layers 20–27 are
never constructed; they cannot contribute anything, so "truncation" here is exact layer removal,
not early exit.

Verified by count (`paramcount.py`, real Qwen3-0.6B shapes):

| | 20 layers | 24 layers |
|---|---:|---:|
| embeddings (`tie_word_embeddings=True`) | 155,582,464 | 155,582,464 |
| decoder | 314,619,904 | 377,543,680 |
| final norm | 1,024 | 1,024 |
| **total** | **470,202,368** | 533,126,144 |

470,202,368 is **exactly** the `BF16` tensor count the HF safetensors index reports for
`bev-decider-0.4B`. The replica is structurally identical.

### 1.2 LoRA

`bev-train/network.py:118-134`, `ref/hf/bev-decider-config.json:24-41`

```python
def lora_target_regex(num_layers, last_k_layers):
    layer_ids = "|".join(str(i) for i in range(num_layers - last_k_layers, num_layers))
    return rf".*layers\.({layer_ids})\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
```

* r = 8, alpha = 16, dropout = 0.0, bias = none
* layers **8–19 of the 20 kept** (the "last 12 kept layers")
* attention projections only — **no MLP projections**
* 48 adapted modules, **983,040 trainable parameters** (printed by `paramcount.py`)
* merged into the released weights, so the published checkpoint has no adapter tensors

### 1.3 Head

`bev-decider/src/bev_decider/model.py:38-65` and `bev-train/network.py:53-84` are identical.

```
ChoiceHead(hidden_dim=1024, new_dim=512, num_layers=2, num_task_types=3)
  task_embedding  Embedding(3, 1024)
  input_norm      LayerNorm(1024)
  layers          2 x AttentionBlock:
                     norm1 LayerNorm(1024)
                     attn  q,k,v: 1024->512 ; o: 512->1024 ; softmax(q.k^T/sqrt(512))
                     norm2 LayerNorm(1024)
                     mlp   Linear(1024,512) GELU Linear(512,1024)
  final_norm      LayerNorm(1024)
  answer_proj     Linear(1024, 512)
  choice_proj     Linear(1024, 512)
  forward: x = [task_token, option_1..option_K, answer_token]   # [B, K+2, 1024]
           logits = (answer_proj(x[:, -1]) . choice_proj(x[:, 1:-1])) / sqrt(512)
           logits[~option_mask] = -inf
```

Parameter count: **7,364,608**. That is **exactly** the `F32` tensor count in the published
checkpoint's safetensors index, and 470,202,368 + 7,364,608 = 477,566,976 = the published total.
`DecisionHeadV1` in `decider_v2/model.py` reproduces this number from code.

The head has **no positional encoding of any kind** over the option axis, and self-attention over
`[task, options..., answer]` is permutation-equivariant, which is where order invariance comes from
on the head side.

### 1.4 How option vectors are pooled — **not** `<opt_end>`, not mean

There are no new special tokens at all. `bev-train/tokenization.py:13-14` uses ordinary text:

```
encode("<option>")  -> [14296, 29]     ['<option', '>']          2 tokens
encode("</option>") -> [522, 2047, 29] ['</', 'option', '>']     3 tokens
```

(measured, `recon/recon_dataset.py` section 6b, on the real Qwen3 tokenizer).

The read index is `bev-train/tokenization.py:136-138`:

```python
# Choice embeddings are read at the last content token of each choice, not at the closing ">" of </option>
choice_end_token_len = len(tokenizer.encode("</option>"))
choice_read_idx = [end_idx - choice_end_token_len for end_idx in span["choices_end_idx"]]
```

i.e. **`end - 3` = the last token of the option's own text**. Identical logic in the inference package
(`bev-decider/src/bev_decider/encode.py:97`). `vocab_size` stays 151936 — the released checkpoint's
`backbone_config.json` confirms it.

So arch.md §5.2's `<opt_end>` / "mean pooling" ablation is a real *alternative*, not a description
of what exists. What exists is **"last content token"**.

### 1.5 Prompt order

`bev-train/tokenization.py:123-125` (identical in `bev-decider/src/bev_decider/encode.py:66-67`):

```python
return f"{TASK_PROMPTS[task_type]}\nQuestion: {instructions}\n\n{truncate(state, max_state_tokens)}\n"
```

**task hint, then question, then state.** arch.md §5.2 puts `<state>` before `<q>`, i.e. the
opposite. The comment in the code explains why: "Task hint and question come before the state, so
every state token is read knowing what is asked." Every state token gets bidirectional-ish causal
context of the question through 20 layers. This is a design decision, not an accident.

### 1.6 Mask and position ids

`bev-train/tokenization.py:45-105`, `dataset.py:96`

* prefix (`question_end_idx + 1` tokens) gets `tril`.
* each option block: `mask[begin:end, :P] = 1` (sees the **whole** prefix) and
  `mask[begin:end, begin:end] = tril` (sees only its own earlier tokens). Nothing else.
* answer tokens (`"The answer is:"`, 4 tokens): `mask[ans:, :ans] = 1` (sees question **and** all
  options) plus `tril` within itself.
* `generate_position_ids`: prefix 0..P-1; **every** option block restarts at P; the answer starts at
  `max(position_ids) + 1`, which is *after the longest option*.
* `collate_fn` starts from `torch.eye(L)` so **padding rows attend only to themselves** (an
  all-`-inf` softmax row would be NaN), then converts 1/0 to `0/-inf` and unsqueezes to `[B,1,L,L]`.

Note `answer_start` depends on `max(position_ids)`, i.e. on option *lengths* only, not their order —
so the answer block's position ids are permutation-stable. Good design; `decider_v2/serialize.py`
keeps it.

### 1.7 `choice` / `noul` / `score`

`bev-train/dataset.py:13-26` (training) and `bev-decider/src/bev_decider/encode.py:27-47` (inference):

| type | training option texts | keys | label |
|---|---|---|---|
| `choice` | `f"{key}: {description}"`, or bare `key` when the description is empty | `list(criteria)` | `keys.index(label)` |
| `score` | `f"{i}: {level}"` — **the level index is written into the option text** | `["0",...,"K-1"]` | `question["label"]` |
| `noul` | `["No", "Yes"]` | `["false","true"]` | `int(label)` |

**Train/serve skew in the reference:** at inference, `noul` with a `criteria` map becomes
`["No: <desc>", "Yes: <desc>"]`; at training it is always `["No","Yes"]`. 17,725 of the 208,677
`noul` questions in `bev-decision('all')` carry `criteria` (`recon/recon_stats_out.txt` §C), so this
is a live inconsistency in the published package, not a hypothetical. `decider_v2/serialize.py`
reproduces the *training* behaviour by default and exposes `noul_use_criteria` for the inference one.

`score` puts the ordinal index in the text precisely because options share position ids. **Any rank
feature must travel with the level, not with the slot** — see §4.3, where I got this wrong first and
the invariance gate caught it.

### 1.8 Loss

`bev-train/train.py:116`

```python
loss = F.cross_entropy(logits, batch["labels"])
```

Plain CE. No label smoothing, no EMD, no per-type weighting, no per-type loss at all. All three
question types are pooled into one softmax-space CE over a variable number of options.

`bev-decider/src/bev_decider/model.py:81-82` and `bev-train/network.py:107-109` both force the head
into **fp32** under `torch.autocast(enabled=False)`, with the comment "bf16 logits are too coarse for
a stable loss".

### 1.9 Optimiser, schedule, budget

`ref/bev-train/configs/full_lora_k12.yaml`

| | value | source |
|---|---|---|
| optimiser | AdamW, weight_decay 0.01 | `train.py:93-96` |
| lr LoRA / head | 1e-4 / 1e-3 | config |
| schedule | cosine, `warmup_ratio` 0.05 | `train.py:100` |
| grad clip | 1.0 | config |
| micro-batch × accum | 32 × 2 = **64 questions/step** | config |
| epochs | 1, `max_train_questions: null` (all) | config |
| `max_state_tokens` / `max_choice_tokens` | 1024 / 64 | config |
| dtype | fp32 weights, bf16 autocast, fp32 head | `inference.py:20-22` |
| attn implementation | not set → transformers default (sdpa) | `network.py` |

Published run (`ref/hf/bev-decider-config.json:21`): `source_checkpoint: run_15-20260928-074927/step-12000`,
model card: "trained for 12,000 steps at batch size 64", "cross-entropy on about 940K typed questions".

**Discrepancy, flagged.** 12,000 × 64 = 768,000 question presentations. The card says ~940K
questions. `bev-decision('all')` train is 422,551 questions (measured). `bev-decision('default')`
train is ~250K questions. So the published run is **not** reproducible from the public recipe: it saw
more question presentations than one epoch of any public config contains. Treat the "baseline
replica" in this project as an *architecture + recipe* replica on our split, not a weight-level
reproduction. **CANNOT VERIFY** the exact data/epochs of the published run.

### 1.10 Checkpoint selection leaks the test split

`bev-train/train.py:57-63`

```python
def validate(network, test_loader, device, logger, step, best_accuracy):
    metrics = evaluate(network, test_loader, device)
    logger.log(step, split="test", **metrics)
    if metrics["accuracy"] > best_accuracy:
        best_accuracy = metrics["accuracy"]
        logger.save_checkpoint(network, name="best", step=step)
```

`checkpoints/best` is selected on **test** accuracy, and `full_lora_k12.yaml` sets
`max_test_questions: 5000` (a fixed prefix of the test split after a seeded shuffle). So the
published checkpoint's "best" is best-on-test. This project replaces that with a validation split
carved from train, per the mission's hard rule 4.

### 1.11 Everything else about the head's blindness

* The head never sees the state except through the option vectors and the answer vector. Those
  option vectors are *contextualised* by the backbone (each option token attends to the whole
  prefix), so "the head only sees pooled option vectors" understates what it has — but the head
  itself has no direct read of the state tokens.
* `noul` is a 2-option choice; there is no separate binary head. arch.md's "deliberate v2 choice"
  is actually what bev already does.

---

## 2. arch.md §18 open questions, answered

**Q1. How exactly does bev-decider pool option vectors and encode `noul`/`score`?**
Pooling: **last content token of the option**, with `</option>` as ordinary text (§1.4). `noul`:
two options `["No","Yes"]` at training, `["No: …","Yes: …"]` at inference when criteria are present
(§1.7). `score`: options are `f"{i}: {level}"`, so the ordinal index lives in the text, and the answer
is `Σ i·p_i` over the softmax (§1.7). No special tokens, no embedding resize.

**Q2. Is `bev-decision` usable for your intended licence?**
**No blanket licence is asserted.** Hub metadata is `license: unknown`, and the card says so
explicitly: "No single blanket license is asserted for this mixed-source release … Consult the linked
upstream sources … before public redistribution or commercial use", naming ProofWriter's
redistribution terms, NVD descriptions authored by third-party CVE reporters, and GoEmotions'
inclusion of Reddit text. Upstream terms it *does* list: MuSiQue/TAT-QA/ContractNLI/MASSIVE/MBPP
CC BY 4.0, CLINC150 CC BY 3.0, 2WikiMultihopQA + CondaQA Apache-2.0, Decider-generated rule cases
Apache-2.0, FinQA + PRM800K MIT, TimeQA BSD-3-Clause (but Wikipedia text CC BY-SA). The card also
says "No non-commercial source was added" — which is a statement about the batches it added, not
about the whole mixture, and it is in tension with "license: unknown" and the redistribution
caveats. `bev-decider-0.4B` itself is **CC-BY-NC-4.0**.
→ **Commercial release of a model trained on this mixture is not cleared.** Treat the weights as
research-only until each upstream is audited individually. This does not block the experiments.

**Q3. Do you need Jev API parity only for the three question types, or also future ones?**
**CANNOT VERIFY** — Jev's API surface is closed and there is no published schema beyond
`choice`/`noul`/`score`. arch.md's own format (§5.1) covers exactly those three, and that is what
`bev-decider/src/bev_decider/serve.py` implements. This project builds only those three and raises on
anything else (`serialize.question_to_options`). One datum worth noting: 23.44% of the `choice`
questions in `all` already contain an abstention-like key (`unknown`, `not_stated`, `none_of_these`,
`none_at_that_time`, `not_determined`, `none`), and that key is the label 6.03% of the time.

**Q4. Target deployment: CPU laptop only, or GPU server too?**
Architecture answers it: the released model is 0.478B with a 1,024/2,048-token state limit, designed
for "a laptop CPU or Apple Silicon" (model card). But my measurements say the *training* constraint is
the binding one and it is a GPU problem: on an RTX 4050 Laptop (6 GiB) the v2 config runs at
**6.5 questions/s** steady state (batch 8 × accum 2, 20 layers, bf16, gradient checkpointing), i.e.
**33 h per 12k-step arm** — see `decider_v2/_gpu_v2_out.txt`. Training is the GPU-server case;
inference is the laptop case. Both are supported by this build.

---

## 3. Verdict on every [ASSUMPTION] and every row of the Gap → change map

### 3.1 The [ASSUMPTION] tags

| arch.md §5.2 | "bev-decider's exact pooling is not documented; ablate `<opt_end>` vs mean-pool" | **WRONG (as a description).** It *is* documented, in `bev-decider/src/bev_decider/encode.py:97` and `bev-train/tokenization.py:136-138`: last content token, with `<option>`/`</option>` as ordinary text. Neither `<opt_end>` nor mean pooling is what the reference does. Both remain legitimate *ablations*; neither is a reproduction of bev. |
| arch.md §5.3 | "bev may implement `noul` differently; this is a deliberate v2 choice" | **CONFIRMED, and the situation is the reverse.** bev *does* implement `noul` as a 2-option choice. The real difference is the criteria text (train `["No","Yes"]` vs serve `["No: …","Yes: …"]`) — a bug-shaped inconsistency, not a design choice. |
| arch.md header | "confirm [assumptions] against `src/bev_decider`" | **There is no `src/bev_decider` in bev-train.** bev-train is flat (`tokenization.py`, `dataset.py`, `network.py`, `train.py`, `logger.py`, `inference.py`, `arch.ipynb`). `src/bev_decider/` is the *separate* `bev-decider` inference repo, which I cloned to `ref/bev-decider`. arch.md conflated two repos. |
| arch.md §2 | "≈0.4B (0.31B layers + 0.16B embeddings + head)" | **CONFIRMED and now exact**: 314,619,904 + 155,582,464 + 7,364,608 (+ 1,024 norm) = 477,566,976. |
| arch.md §2 | "CE on ~940K typed questions, 12k steps, batch 64" | **PARTIALLY CONFIRMED.** 12k steps × batch 64 = 768k presentations ≠ ~940k questions. The recipe is confirmed from `configs/full_lora_k12.yaml`; the data volume is not reproducible from public data. **CANNOT VERIFY.** |
| arch.md §2 | "Trained to 1,024-token states; default limit 2,048" | **CONFIRMED.** `max_state_tokens_train: 1024`, `max_state_tokens: 2048` in the published `config.json`. |
| arch.md §2 | "Weights CC-BY-NC-4.0 (NC training data); code Apache-2.0" | **CONFIRMED.** `bev-decider-0.4B` → `cc-by-nc-4.0`; `bev-decider` → Apache-2.0. Note bev-train itself declares **no licence** (no `LICENSE` file, no `license` key in `pyproject.toml`) — so "code is Apache-2.0" is true of the inference package, not of the training code. |
| arch.md §2 | "Reported … Held-out 74.7, JevBench 65.8/46.8, sysone-bench 70.8, ECE 0.065" | **CONFIRMED as published claims** (`ref/hf/bev-decider-README.md`). I reproduced the held-out number: see §5. |

### 3.2 The Gap → change map (arch.md §3)

| gap | change | verdict |
|---|---|---|
| Head only sees pooled option vectors | Perceiver-style ctx summarizer | **KEEP, and it is better motivated than arch.md realises.** The head has no direct read of the state; the only route is through the option vectors. Implemented in Tier 1. See §6.1 for the fix I had to make (the summarizer must see the *prefix only*). |
| Deeper layers dropped (20/28) | Depth is an ablation (20/24/28) | **KEEP as Tier 2.** Note 24 layers costs 62,923,776 more params (533,126,144 vs 470,202,368 total). 28 layers is untested here: bev's LoRA targets are absolute layer indices, so going to 28 changes what "layers 8-19" means. Flagged, not built. |
| LoRA r=8 layers 8-19, attention only | r=16, all kept layers, + MLP | **KEEP as Tier 1, with a cost warning.** 983,040 → **7,208,960** trainable backbone params (7.3×). Measured cost: 6.5 q/s (v2) vs 8.6 q/s (baseline replica) on the RTX 4050, same batch — a **1.32× slowdown**. |
| Date/arithmetic unreliable | Deterministic pre-pass + synthetic data | **SPLIT.** The pre-pass is implementable (`decider_v2/normalize.py`, Tier 2). The synthetic data is not required and I have not built it: the dataset already contains a `numeric_temporal` config (19,806 rows, "43 calculation types", labels computed in code) — **use that instead of generating anything**. Generating new data would also break the fair-comparison rule (both arms must see the same data). |
| Exceptions overlooked | Counterfactual / minimal-edit pairs + hard-negative stage | **KEEP as Tier 2, and the data already exists**: `counterfactual_15k` (15,000 rows, 7,500 label-flipping pairs). The hard-negative stage (Tier 2, `--resume` + `--lr ÷ 4`) is built as a config, not as a separate script. |
| `score` via CE on expected value | Rank-aware head + CE + EMD | **KEEP as Tier 1, but the rank must travel with the level.** See §4.3. Also: EMD is already unit-tested to be exactly 0 for a perfect prediction and exactly `1/K` for a one-off error (`tests.py` (c)). |
| No "none of the above" | Learned null option | **DEFAULT OFF.** See §4.2 — the dataset already *has* abstention labels, so a learned null is largely redundant. Kept as a Tier-2 flag (`--use_null 1`, +512 params). |
| ECE 0.065 | Label smoothing + per-type temperature | **KEEP, but label smoothing needs a fix to be usable at all.** See §4.1. Measured reference ECE: **0.0122** on the `default` 5,000-question sample and **0.0144** on `all` (n=4,000). The published 0.065 is from sysone-bench, a different distribution. (An earlier draft quoted 0.0381 here; that was the reference ECE on the *1,000-question* draw that §5.4 retracts, not on `all`.) |
| State re-encoded per question | Shared state KV prefix | **DEFER to Tier 3.** It requires a cache-aware forward; `use_cache=False` is baked into the current network. The packed-option design already shares the prefix *within* one question, which is where the cost is. Not built. |
| 1k-token training | Length curriculum 512 → 1k → 2k → 4k | **OUT OF SCOPE.** Measured: only 7.75% of train rows exceed 1,024 state tokens, and the p99 is 958 tokens (3,000-row sample, `recon/recon_dataset.py` §6b). A curriculum would trade real data for a tail that is 1% of the distribution. Tier 3. |
| English-centric | Multilingual augmentation | **OUT OF SCOPE.** The dataset card declares `language: [en]` and the mixture has no multilingual slice; `sysone-bench`'s multilingual intent suite is the only multilingual benchmark and it is a *benchmark*, not training data. Adding it would break rule 3 (identical data in both arms). Tier 3. |
| NC licence | Licence-audited data, init from Qwen3 | **HALF DONE.** Init from Qwen3 is done (hard rule 7). Licence audit: see Q2 — the mixture is **not** cleared for commercial release. This is a distribution constraint, not a modelling one. |

---

## 4. Risks in arch.md, checked

### 4.1 Label smoothing is unusable as written

arch.md §8 and §3 both say "CE, label smoothing 0.05". With the packed multi-option batch, `C` is
the *maximum* option count in the batch and every other column is masked. torch's
`F.cross_entropy(..., label_smoothing=eps)` spreads `eps` over all `C` columns:

```
a batch of 2 rows, C=4, real option count 3, one column at -1e9, eps=0.05
    torch CE  = 2.500e+07
    corrected = 0.710785
```

(measured, `tests.py` (c): "torch's own smoothed CE is unusable with masked columns"). It is not a
rounding issue — 45.35% of all questions are 2-option `noul`, so padding columns are the normal case,
and 2-option rows inside a 17-option batch get a loss of order `1e7` and a gradient that swamps
everything else.

Fix shipped in `decider_v2/losses.py:smoothed_cross_entropy`: renormalise the smoothing term over the
`K` **valid** options only. It is provably identical to torch's version when every column is valid
(also unit-tested), finite on padded batches, and gives padding columns exactly zero gradient
(also unit-tested).

### 4.2 The null option: the data already has abstention, so default it OFF

Measured over all 224,880 `choice` questions in `bev-decision('all')` (`recon/recon_stats_out.txt` §B):

| | count | share of choice questions |
|---|---:|---:|
| questions with ≥1 abstention-like key in `criteria` | 52,709 | **23.44%** |
| of those, the abstention key **is** the label | 13,558 | **6.03%** of all choice questions |

Top keys (times label / times present): `unknown` 5,392/16,228 · `not_stated` 5,320/15,769 ·
`none_at_that_time` 1,076/8,873 · `not_determined` 1,022/8,555 · `none` 746/3,282.

So the dataset is **not** missing abstention supervision — it expresses it as an ordinary option,
and bev's encoder handles it with no extra machinery. A *learned* null slot would (a) duplicate
signal the model already has, (b) steal probability mass from the existing abstention options during
the 76.6% of choice questions that have none, and (c) add a train/serve contract that the Jev API
does not have (what does `probabilities["__null__"]` mean to a caller?).
**Decision: `--use_null 0` by default.** The flag exists and is tested (+512 params when on).

### 4.3 The 4-D mask + shared position ids under `sdpa` — **PROVEN, with a caveat**

`flash_attention_2` genuinely cannot take an arbitrary mask, but that is moot: bev-train never sets
`attn_implementation`, and transformers 5.15 defaults to `sdpa`.

I traced the path. `transformers/masking_utils.py::_preprocess_mask_arguments`:

```python
# If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
    return True, attention_mask, None, None, None, None, None
```

so the `[B,1,L,L]` additive mask is passed straight to the attention kernel.
`integrations.sdpa_attention_forward` forwards it as `attn_mask`, and
`Qwen3Attention.eager_attention_forward` adds it to the scores. Both consume the float
`0/-inf` form.

**Proof, not argument** (`decider_v2/tests.py` (a), real output in `decider_v2/_tests_gpu_out.txt`):

```
      k=3: max|packed - solo| over option read positions = 4.768e-07
      k=5: max|packed - solo| over option read positions = 7.153e-07
      k=8: max|packed - solo| over option read positions = 5.960e-07
  [PASS] packed-vs-separate option hidden states  max delta 7.153e-07 (gate 1e-5)
  [PASS] option isolation (dropping an option does not move the others)  max delta 7.153e-07
  [PASS] attn_implementation=eager head=v1 honours the custom 4-D mask  max delta 5.513e-07
  [PASS] attn_implementation=eager head=v2 honours the custom 4-D mask  max delta 5.513e-07
  [PASS] attn_implementation=sdpa   head=v1 honours the custom 4-D mask  max delta 5.960e-07
  [PASS] attn_implementation=sdpa   head=v2 honours the custom 4-D mask  max delta 5.960e-07
```

"Option isolation" is the stronger statement and the one that matters: delete an option from the
batch and every surviving option's hidden state moves by ≤7.2e-07, so an option cannot see its
neighbours at all.

**The caveat, and a real trap.** transformers 5.15's `sdpa_mask()` returns a **boolean** mask
(`True` = attend), whereas `bev-train` builds a float additive mask. Mixing the two conventions
silently inverts the mask instead of raising. I hit exactly this while writing my own collate and it
produced a plausible-looking, completely wrong model. The collate now converts through an explicit
1/0 matrix (`serialize.build_batch` comment) and `tests.py` (a) fails loudly if it is inverted.

**Order invariance, measured rather than assumed.** fp32, `k` clamped to 3–12:

| model | Δmax | examples × shuffles |
|---|---:|---|
| 2-layer random-init Qwen3, head v2, `bev` mode | 5.96e-08 | 4 × 8 |
| 2-layer random-init, `special` mode, `opt_end` pooling | 5.96e-08 | 4 × 8 |
| same, `score` questions, K = 3 / 5 / 7 | 5.96e-08 / 4.47e-08 / 4.47e-08 | 8 each |
| **trained** 2-layer v2, real data | 8.94e-08 | 24 × 8 |
| **published `bev-decider-0.4B`, 20 layers** | **3.16e-06** (default-config test), **3.70e-06** (`all`) | 24 × 8 |

The reference's own Δ is ~3e-6, i.e. two to three orders of magnitude above the 2-layer numbers but
still inside the model card's "no probability moved by more than 1e-5" claim. This is fp32
non-associativity (permuting options changes the sequence length layout and therefore GEMM
tiling), not a structural leak: the hidden-state isolation test above holds to 7e-7. **The gate
stays at 1e-5, but expect ~1e-6 at 20 layers, not ~1e-8, and do not treat 1e-5 as a bug report.**

### 4.4 The Python-loop mask is replaced by a vectorised collate

arch.md §6 builds the mask with `for l in opt_lens:` inside a function called per example per
batch. With 388,748 train questions × 12k steps that is a real cost, and it makes the mask logic
harder to test than the thing it implements.

`decider_v2/serialize.py::build_batch` builds it with broadcasting only. Block ids are
`0` = prefix, `1..K` = option `i`, `K+1` = answer, `-1` = padding; a query `q` may attend to key
`k` iff

```
(block[q] == block[k] and k <= q)          # within-block causal
or (k < P and block[q] > 0)                # options and answer see the whole prefix
or (k <= q and block[q] == K+1)            # the answer sees every option too
and k < valid_len[b]                        # never attend to padding
```

and padded query rows are then forced to `k == q`. This is ~10 tensor ops on `[B,L,L]` and it is
directly unit-testable against the per-example definition.

### 4.5 Which of arch.md §14's gates are reachable with this dataset

| arch.md §14 gate | target | reachable from `bev-decision('all')` alone? |
|---|---|---|
| Held-out typed decisions (all) | ≥ 77 | **Yes.** This is the headline number, measured on our own split. |
| JevBench all / hard | ≥ 70 / ≥ 52 | **Yes, as a measurement.** The data and eval code are both present (§7) and `decider_v2/bench/run_jevbench.py` drives them, graded by jevbench's own `scoring.score_task` + `summarize.metric`. It is a *benchmark*, not training data, so it costs nothing in rule-3 terms. |
| sysone-bench v2 | ≥ 75 | **Yes, as a measurement** (§7). |
| sst5 | ≥ 40 | **Unreachable.** sst5 is a sysone-bench *suite*, 40 items. 40/40 would be 100%. Nonsense as a gate; I would replace it with the 5-level `score` slice of the mixture (21,258 five-level questions, plus 7,993 three-level and 7,050 seven-level). |
| multilingual intent | ≥ 80 | **Unreachable.** `language: [en]`; the mixture has no multilingual data. Target is meaningless without Tier-3 augmentation. |
| verification domain | ≥ 65 | **Partly.** Domain `Answer and solution verification` = 10,901 rows / 17,371 questions (3.95%). Enough to measure, thin enough that a ±2-point swing is noise at 5k questions. |
| temporal/unit domain | ≥ 70 (with pre-pass) | **Partly.** `Temporal and unit reasoning` (17,274 rows) + `Time-sensitive reading` (8,873) + the whole `numeric_temporal` config. Plenty of data. But the pre-pass is Tier 2 and the reference scores 54.0 here (model card) — the biggest realistic headroom on this dataset. |
| ECE ≤ 0.05 | hard gate | **Yes, and already satisfied.** Measured reference ECE on `all`: **0.0144** (n=4,000; 0.0122 on `default`, n=5,000). arch.md's 0.065 comes from sysone-bench. Make this a regression gate (don't get worse than the reference), not a stretch goal. (0.0381 was the n=1,000 draw retracted in §5.4.) |
| Option-order max Δ ≤ 1e-5 | hard gate | **Yes** — measured 3.70e-06 for the reference on real data. Passes today. |
| own ~1k weakness slice | track separately | **Yes**, but note `numeric_temporal` explicitly warns "28 of 172 calculation-type/rendering combinations are test-only", so a self-made slice risks duplicating them. |

**My honest read:** of the eight gates, three are unreachable with this dataset (sst5 as written,
multilingual intent, and the 4k context claim), one is already satisfied (ECE), and the rest are
measurable. The interesting question is not "can v2 hit 77" but "how much of the gap to Jev on
temporal/unit and verification is closed by head v2 + EMD vs by the DERIVED pre-pass" — which is why
the pre-pass is Tier 2 behind a flag rather than baked in.

---

## 5. Weak spots of bev-decider, measured on the data

Both runs use `decider_v2/evaluate.py --reference bev-decider-0.4B`, i.e. my own encoder,
collate and metrics code against the published weights. That is a pipeline-fidelity check as much as
a model check.

### 5.1 The `default`-config 5,000-question sample — validates my pipeline against the published number

| | overall | choice | noul | score | ECE | NLL |
|---|---:|---:|---:|---:|---:|---:|
| **measured, `default` test, 5,000 sampled (seed 0)** | **77.24** | **74.65** | **87.01** | **64.99** | **0.0122** | 0.5491 |
| published (model card, revision `687e28f`) | 74.7 | 69.8 | 84.6 | 63.7 | — | — |

Same protocol as the card (5,000 questions from the test split), same checkpoint, and my encoder
reproduces the published number within ~2.5 points, in the same direction on every type. I cannot
make it exact because the card's 5,000 are a specific (state, question) draw I do not have; mine are
`random.Random(0).sample` over 46,320 test questions. **The gap is sampling, not implementation** —
which is itself the most important calibration fact for reading any of the numbers below.

### 5.2 The `all` and `default` test splits are within ~1 point of each other

| split | n | overall | choice | noul | score | ECE | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `default` test (published protocol) | 5,000 | 77.24 | 74.65 | 87.01 | 64.99 | 0.0122 | 0.549 |
| published (card, revision `687e28f`) | 5,000 | 74.7 | 69.8 | 84.6 | 63.7 | — | — |
| **`all` test** | **4,000** | **76.30** | **73.88** | **83.53** | **64.87** | **0.0144** | **0.562** |

The two configs land within 0.94 points of each other and both reproduce the published figure. So the
extra batches (`hard_50k`, `numeric_temporal`, `skills`, `counterfactual_15k`) do **not** make the test
split materially harder, and the v2 target on our split really is ~76, comparable to the published
74.7. I initially concluded the opposite from a 1,000-question draw and corrected it — see §5.4.

This is the useful result: **the baseline replica and v2 can be compared against the published
number after all**, because the test distribution barely moved. What still cannot be compared is the
*training* recipe (see §1.9), so the delta that matters remains v2 − baseline replica.

### 5.3 Per-domain accuracy of the reference on `all` (400-question sample, CPU)

Full table in `decider_v2/_refeval_out.txt`. Sorted by the model's own weak spots, from that run:

| domain | n | accuracy |
|---|---:|---:|
| Sensory quality rating | 1 | 0.000 |
| Response preference and quality | 20 | 0.450 |
| Policy and rule reasoning | 14 | 0.500 |
| Reading comprehension | 11 | 0.545 |
| Contract evidence | 3 | 0.667 |
| Software security | 14 | 0.643 |
| Long contracts and legal documents | 11 | 0.636 |
| Time-sensitive reading | 10 | 0.700 |
| Temporal and unit reasoning | 26 | 0.692 |
| Multi-hop reading and evidence | 22 | 0.773 |
| Sentiment, emotion, and moderation | 60 | 0.783 |
| Support and intent routing | 30 | 0.867 |
| Trap and adversarial reading | 11 | 0.909 |

**Read this table with care.** At n = 1–26 per domain the confidence interval is ±0.15 or wider;
several of these cells are single-digit counts. What it is good enough for is *ranking*: the
reference is weak on policy/rule reasoning, response preference, reading comprehension, temporal and
time-sensitive reading, and long contracts — which is a plausible-looking and consistent picture
with the model card's own "known weaknesses" (date/arithmetic, stated exceptions, sst5, English).
It is **not** good enough to set a per-domain target. The notebook's real run evaluates on the full
80,019-question `all` test split, where per-domain counts are 1,300–68,994 and the numbers mean
something.

### 5.4 Small-sample noise is large enough to invent a finding

Four measurements of the *same* checkpoint, on the *same* code, sampled from the same published test
split with `random.Random(0).sample`:

| split | n | accuracy | 95% CI half-width (binomial, p≈0.76) |
|---|---:|---:|---:|
| `all` | 400 | 76.25 | ±4.26 |
| `all` | 1,000 | **68.70** | ±2.69 |
| `all` | 4,000 | **76.30** | ±1.35 |
| `default` | 5,000 | 77.24 | ±1.20 |

The n=1,000 draw is 7.6 points below the n=4,000 draw of the same distribution — about 1.8 binomial
standard errors, entirely ordinary. **An earlier draft of this document concluded from that draw that
the `all` config was intrinsically harder than `default`; it was wrong**, and the n=4,000 measurement
contradicts it (§5.2: the two configs are within 0.94 points). Three lessons, all now structural
rather than advisory:

* the notebook carves one `test` shard with a fixed `SPLIT_SEED`, uploads it once, and evaluates every
  arm, every seed and the reference on that exact file, so arms are always compared on identical bytes;
* **a real run evaluates the FULL 80,019-question published test split, never a shard.** The 4,000
  shard exists only under `SMOKE`, where it keeps validation to minutes; cell 3 asserts which one you
  got and refuses to train on a shard if `SMOKE=False`;
* **no comparison in this repository rests on fewer than 4,000 questions**, and every published
  accuracy is reported with its n.

This is the single most important methodological fact here: at these sample sizes a 5-point
difference between two models is noise, so a claimed v2 improvement below ~3 points should be
treated as unproven no matter how clean the table looks. The real run does not depend on that
caution, because at n=80,019 the interval is ±0.30 points.

---

## 6. Other things the code review found

### 6.1 The Perceiver summarizer must attend to the prefix only

My first cut passed the whole hidden sequence to the context summarizer. It is still *exactly*
permutation-invariant in real arithmetic — 8 learned queries attending to a key **set** are
permutation-invariant — so a reasoning-only argument says it is fine, and the tiny invariance test
passed at 4.5e-08. But measured on trained weights it moved probabilities by **1.96e-03**, because
reordering keys changes the summation order inside the attention softmax.

Two fixes, both shipped: slice the summary to the prefix (`DeciderV2.forward`, `prefix_len`), and
add the regression case to `tests.py` (b2), which specifically shuffles `score` questions.

### 6.2 The `score` rank must travel with the level

First cut computed `rank = slot_index / (K-1)` from the option's position in the batch. Under
shuffling that attaches each level to a *different* rank, which breaks invariance by 1.96e-03 —
same symptom, different cause, and it would have silently degraded every `score` question that
trained on shuffled options (13.7% of the mixture, and 100% of them once per-epoch shuffling is on).

Fixed by carrying `ranks` out of `question_to_options` alongside `options`, permuting them together
in `DecisionDataset.__getitem__`, and passing them through the collate. `tests.py` (b2) covers
K = 3, 5, 7.

### 6.3 `transformers` mask convention trap (see §4.3) and two smaller ones

* `datasets` keys its `.map` cache by **filename**, so reusing one cache path across two different
  source splits silently returns the first split's rows. My first `prepare.py` reported
  `state overlap val->test 97, calib->test 193` when the true overlap is 0 (independently verified:
  0 of 44,856 test rows share a normalized state with train). Fixed with per-split cache names; the
  overlap assertion is now in the manifest.
* `from_pretrained` leaves the model in **eval** mode, and transformers' `GradientCheckpointingLayer`
  only checkpoints when `self.training` is True. Without an explicit `.train()`, gradient
  checkpointing silently does nothing: peak VRAM went 9.22 GiB → 2.84 GiB and throughput 0.63 →
  7.74 questions/s on the RTX 4050 purely from adding `.train()` to the benchmark. `train.py` always
  had it; `bench.py` did not.
* `merge_and_unload()` mutates the live module. An early `save_checkpoint` called it in place, which
  stripped the adapter from the training model on the first save. Now it runs on a `deepcopy`.

### 6.4 The `all` config has more data than one 12k-step epoch consumes

| | count |
|---|---:|
| rows, config `all` | 231,332 train / 44,856 test |
| **typed questions, config `all` train** | **422,551** |
| train after carving val (3%) and calib | 388,748 |
| val | 12,676 |
| calib | 21,127 (the mission's ≥20k, satisfied) |
| typed questions, config `all` test | **80,019** (choice 35,673 / noul 32,496 / score 11,850) |

12k steps × 64 = 768,000 presentations = **1.82 epochs** of the carved train split. With per-epoch
option shuffling on, that means each option ordering is seen roughly twice. `--epochs 2` with
`--max_steps 12000` is the faithful reading of "12k steps"; `--epochs 1 --max_steps 12000` is 0.91 of
an epoch and would be a different experiment. Both are exposed as flags; the notebook uses
`max_steps` so the budget is exact regardless.

### 6.5 Data facts that shape the design

* **0** questions lack a `label`. **0** `choice` questions have empty `criteria`.
* Options per question: 2 → 227,912 (45.35%), 3 → 47,645, 4 → 90,659, 5 → 61,136, 6 → 28,659,
  7 → 12,145, 8 → 21,699, up to 17 (221). `choice` spans 2–17, `noul` is always 2, `score` is
  3 / 4 / 5 / 7 only (7,993 / 32,712 / 21,258 / 7,050). The C = 17 tail means the batched mask's
  second axis really does reach 17, which is why the label-smoothing bug (§4.1) is not a corner case.
* 31,824 distinct `choice` keys; the most common are `option_1` (31,876), `option_2` (31,876),
  `unknown` (16,228), `none_of_these` (16,013), `not_stated` (15,769).
* `noul` labels: False 128,322 / True 80,355 — **61.5% negative**. A `noul`-only model that always
  says "no" gets 61.5%; the reference gets 83.33 on `all` and 87.01 on `default`.
* 8,131 questions carry a `label_probs` soft label (6,993 rows, the `skills` calibration slice).
  **Not used.** arch.md lists soft labels only as a distillation idea; hard CE over `argmax` is what
  bev does and what keeps the arms comparable. Recorded so it is not mistaken for a missed feature.
* State token lengths (3,000-row sample): p50 150, p90 462, p95 625, **p99 958**, max 1,986, mean 212.
  7.75% of train rows exceed 1,024 tokens (11,567-row sample).
* Option token lengths: p50 4, p90 16, p95 22, p99 32, max 221. The 64-token limit truncates
  332 options in an 11,567-row sample.
* Instructions: p99 38 tokens, max 52 — never the bottleneck.
* 28 domains; top five by rows: Sentiment/emotion/moderation 9.64%, Support and intent routing 9.58%,
  Retail 7.20%, Spatial and logical reasoning 7.09%, Financial reporting and banking 6.74%.
* Questions per row: 1 → 153,528, 2 → 72,064, … max 9.

### 6.6 Local environment, recorded

| | CPU env | GPU env |
|---|---|---|
| python | 3.14.6 | 3.14.6 |
| torch | 2.12.1+cpu | **2.11.0+cu128** |
| GPU | none usable (RTX 4050 present but torch was CPU-only) | NVIDIA GeForce RTX 4050 Laptop, **6.0 GiB**, sm_89 |
| transformers / datasets / peft | 5.15.0 / 5.1.0 / 0.21.2 | 5.15.0 / 5.1.0 / 0.21.2 |

The GPU environment is a separate venv (`.venv-gpu`) created for this work so the CPU environment
stays usable. See `LOCAL_VERIFICATION.md` for the throughput measurements.

---

## 7. Phase 0.3: benchmark availability (the mission asks for exactly this)

| benchmark | data | eval code | verdict |
|---|---|---|---|
| **sysone-bench** v2.1.0 | **YES** — `instax-dutta/sysone-bench` @ `main`, `datasets/v2/manifest.v2.1.0.jsonl`, **1,190 cases** (952 evaluation + 238 calibration) = **1,240 evaluation decisions**, with `state`, `questions` (Jev-format, with `criteria` and `labels`), `expected`, `suite_id`, `provenance_id`, `split`. 0.90 MB, MIT licence. | **YES** — `run.py`, `benchmark/{orchestrator,metrics,scoring,canonical,report}.py`, `runners/`. There is a vendor adapter per family (`runners/vendor/*.py`). | **AVAILABLE, adapter written.** `decider_v2/bench/run_sysone_bench.py` drives it and is graded by sysone-bench's own `contracts.validate_answers` + `metrics.summarize_predictions` — the metric is never re-implemented. The contract (`{state, questions}` in, `{qid: answer}` out, `noul` → `{"type","noul"}`, `choice` → `{"type","choice","choice","probabilities"}`, `score` → `{"type","score","score","probabilities"}`) is already what `decider_v2` produces. |
| **JevBench** | **YES** — `fstandhartinger/jevbench`, `datasets/public/{easy,hard,original}.jsonl` = **48 + 111 + 72 = 231 tasks**, MIT. Each line: `{id, state, question{type,instructions,criteria}, labels, expected, family, group, provenance}`. `easy` = the 48 "easy" tier, `original` = the 72 "standard" tier, `hard` = the 111 hard tier — the same 48/72/111 split the model card reports. | **YES** — `jevbench/{tasks,scoring,metrics,composite,composite_v14}.py`, CLI, and adapters (there is already an `adapters/so1_decider.py`, whose shape `decider_v2/bench/run_jevbench.py` follows). | **AVAILABLE, adapter written.** `decider_v2/bench/run_jevbench.py` drives it and is graded by jevbench's own `scoring.score_task` + `summarize.metric`. |
| Hugging Face | `jevbench`, `jevbench`, `instax-dutta/sysone-bench`, `sysone-bench` | — | **NOT FOUND** as datasets; both live on GitHub only. |

**Both adapters are now written and run** (`decider_v2/bench/`, patterned on
`jevbench/adapters/so1_decider.py`: load the model once, one forward pass per question, hand the
distribution to the harness's own scorer). The decode step this section flagged as "new code" was
mechanical once the contracts were read:

* `choice` — `criteria` keys are the label set, so `question_to_options` already yields them in the
  task's own order; the probability map is keyed by the label string.
* `noul` — the model's option keys are `("false", "true")` and JevBench's labels are `("no", "yes")`;
  a two-entry rename is the whole mapping.
* `score` — JevBench labels are `"0".."k-1"` and sysone-bench's `expected` is the integer level, so
  sysone's `score` answer is the **probability-weighted expected level** (its `decision_correct`
  rounds with `floor(x+0.5)`, and MAE/RMSE/Spearman all want a continuous prediction, not an argmax).

Faithfulness check, and the reason to trust these numbers: run through the adapters, the **published**
`bev-decider-0.4B` weights score **64.94** on JevBench (published: 65.8) and **70.52 choice / 81.25
noul** on sysone-bench (published all: 70.8), with 231/231 distributions valid. See
`LOCAL_VERIFICATION.md` §8.2.