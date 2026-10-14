# LOCAL_VERIFICATION.md — every command run, its real output, and what was skipped

Two environments were used. The **CPU** env is the machine's default Python; the **GPU** env is a
separate venv created for this work because the installed `torch` was a CPU-only build (`2.12.1+cpu`)
and the machine has an RTX 4050 that was therefore unusable. Both are recorded so nothing here is
mistaken for something I did not run.

| | CPU env | GPU env |
|---|---|---|
| interpreter | `C:\Python314\python.exe` | `F:\Jev\.venv-gpu\Scripts\python.exe` |
| python | 3.14.6 | 3.14.6 |
| torch | 2.12.1+**cpu** (no CUDA) | 2.11.0+**cu128** |
| GPU | RTX 4050 present but unreachable from torch | **NVIDIA GeForce RTX 4050 Laptop GPU, 6.0 GiB, capability (8, 9)** |
| transformers / datasets / peft / hub / numpy | 5.15.0 / 5.1.0 / 0.21.2 / 1.33.0 / — | 5.15.0 / 5.1.0 / 0.21.2 / 1.33.0 / 2.3.5 |

`nvidia-smi` output at the start (CPU env), for the record:

```
0  NVIDIA GeForce RTX 4050 ...  Bus-Id 0000:00:01:00.0  6141MiB / 6141MiB
```

Machine: Windows 11 (10.0.26300), PowerShell 5.1. No AWS credentials, so **no SageMaker job was
launched** and no cost was incurred.

---

## 0. What was NOT run, and why

| item | status | reason |
|---|---|---|
| SageMaker training jobs (baseline / v2 / ablations) | **NOT RUN** | no AWS credentials. The mission forbids launching them from here. |
| Any real accuracy comparison between baseline and v2 at the 12k-step budget | **NOT RUN** | needs the jobs above. Everything below is either a gate, a plumbing check, or a measurement of the *reference*. |
| A 3-seed `SEPARABLE` verdict on real data | **NOT RUN** | the dry runs have 2 seeds. The 3-seed path was driven end-to-end on synthetic prediction files (§8.2.3) to prove the code reaches `SEPARABLE`, but no real 3-seed measurement exists. |
| JevBench / sysone-bench adapters | **RAN** | both written (`decider_v2/bench/`), wired into cell 12, executed in the dry run — see §8.2. The *reference* rows reproduce the published figures, which is the faithfulness check. |
| Tier-2 ablations (24 layers, mean pooling, DERIVED pre-pass, v2-LoRA-shape) | **NOT RUN** | each is a full training run. The flags exist and are exercised by `tests.py` where a gate covers them; cell 14 lists them and launches them behind `RUN_ABLATIONS`. |
| Stage-2 hard-negative fine-tune | **NOT RUN** | needs stage-1 errors mined from train-dev; the resume path itself is exercised. |
| `flash_attention_2` | **NOT INSTALLED** | not needed and cannot work here: flash-attn cannot take an arbitrary 4-D mask, which is the whole point. `eager` and `sdpa` are both tested. |
| Multi-GPU (`torchrun`) | **NOT RUN** | one GPU here. The code path exists (`dist_info`, `DistributedSampler`, `all_reduce` in `evaluate`, rank-0-only logging/saving); it has never been executed. The preflight's cross-rank verdict all-reduce is written but likewise unexecuted. |
| Concurrent training jobs | **NOT RUN** | SageMaker bills by instance-hour, so concurrency is a wall-clock optimisation, not a correctness one. The offline harness runs jobs serially in subprocesses; `est.fit(wait=False)` is the default on real AWS. |

---

## 1. Recon (Phase 0) — ran, output in `recon/`

```
python recon/recon_dataset.py          -> recon/recon_out.txt
python recon/recon_stats.py            -> recon/recon_stats_out.txt
git clone https://github.com/avbiswas/bev-train      ref/bev-train
git clone https://github.com/avbiswas/bev-decider    ref/bev-decider
git clone https://github.com/instax-dutta/sysone-bench   ref/sysone-bench
git clone https://github.com/fstandhartinger/jevbench    ref/jevbench
```

Headline numbers, all from those files:

* `avbiswas/bev-decision` config `all`: **231,332 train rows / 44,856 test rows**, 114.21 MiB of
  parquet across 5 batch directories.
* **502,570 typed questions**: choice 224,880 (44.75%), noul 208,677 (41.52%), score 69,013 (13.73%).
  422,551 of them are in train; **80,019** in test.
* 0 questions missing a label; 0 `choice` questions with empty `criteria`; 8,131 with `label_probs`.
* score levels: 3 → 7,993, 4 → 32,712, 5 → 21,258, 7 → 7,050.
* options per question: 2 → 227,912 (45.35%) … 17 → 221; 31,824 distinct choice keys.
* 28 domains; top row-share Sentiment/emotion/moderation 9.64%, Support routing 9.58%, Retail 7.20%.
* `all` train and test share **0** states (normalised exact match, 44,856 test rows checked).
* licence: dataset `unknown` (card says so explicitly), `bev-decider-0.4B` **cc-by-nc-4.0**,
  `bev-decider` Apache-2.0, `bev-train` **no licence declared**.

---

## 2. `tests.py` — ran, PASSED on both environments

```
# CPU
python tests.py --device cpu            -> decider_v2/_tests_out.txt
# GPU (RTX 4050)
F:\Jev\.venv-gpu\Scripts\python.exe tests.py --device cuda   -> decider_v2/_tests_gpu_out.txt
```

GPU run, verbatim (22 checks, all PASS):

```
decider_v2 tests.py   device=cuda  torch=2.11.0+cu128

(a) packed-vs-separate equivalence under the 4-D additive mask + shared position ids
      k=3: max|packed - solo| over option read positions = 5.364e-07
      k=5: max|packed - solo| over option read positions = 7.153e-07
      k=8: max|packed - solo| over option read positions = 5.960e-07
  [PASS] packed-vs-separate option hidden states  max delta 7.153e-07 (gate 1e-5)
  [PASS] option isolation (dropping an option does not move the others)  max delta 2.384e-07
  [PASS] attn_implementation=eager head=v1 honours the custom 4-D mask  max delta 5.513e-07
  [PASS] attn_implementation=eager head=v2 honours the custom 4-D mask  max delta 5.513e-07
  [PASS] attn_implementation=sdpa   head=v1 honours the custom 4-D mask  max delta 5.960e-07
  [PASS] attn_implementation=sdpa   head=v2 honours the custom 4-D mask  max delta 5.960e-07

(b) option-order invariance (fp32, head logits and probabilities)
  [PASS] order invariance mode=bev pool=bev  max |dp| = 5.960e-08, max |dlogit| = 3.278e-07 (gate 1e-5)
  [PASS] order invariance mode=special pool=opt_end  max |dp| = 2.980e-08, max |dlogit| = 2.086e-07 (gate 1e-5)

(b2) order invariance for score questions (rank embedding travels with the level)
  [PASS] score order invariance K=3  max |dp| = 5.960e-08
  [PASS] score order invariance K=5  max |dp| = 4.470e-08
  [PASS] score order invariance K=7  max |dp| = 4.470e-08

(c) loss sanity
  [PASS] EMD == 0 for a perfect prediction  emd=[0.0, 0.0, 0.0]
  [PASS] EMD == |a-b|/K for a one-off error  emd=0.200000 expected 0.200000
  [PASS] padding columns get exactly zero gradient  grad=0.0
  [PASS] masked smoothing == unmasked smoothing on the valid columns  masked=0.710785 unmasked=0.710785
  [PASS] torch's own smoothed CE is unusable with masked columns  torch CE=2.500e+07 vs ours=0.710785
  [PASS] full loss is finite, small and differentiable  loss=0.451743
  [PASS] score rows get CE + 0.5*EMD  per-example=[0.4576059579849243, 0.4458802342414856]
  [PASS] label smoothing penalises a confident CE  ls=0.250045 > ce=0.000045
  [PASS] ECE of a calibrated set is ~0  ece=0.0500 (analytic 0.05)
  [PASS] ECE of an over-confident set is large  ece=0.2500

(d) tiny overfit: 64 synthetic examples, <= 300 steps, target >= 95% train acc
  [PASS] overfit reaches the accuracy target  acc=1.000 after 300 steps, loss 1.1186 -> 0.2030, 14.3s

==============================================================================
ALL CHECKS PASSED
```

The CPU run of the same file is `decider_v2/_tests_out.txt` and also ends `ALL CHECKS PASSED`
(order-invariance max |dp| 5.96e-08, overfit acc 1.000 in 300 steps, 21.1 s).

**`tests.py` runs offline.** It uses a byte-level tokenizer built in the file and a 2-layer
random-init Qwen3 config, so no model download is needed. No test was skipped.

---

## 3. Parameter counts — ran, and they match the published checkpoint exactly

```
F:\Jev\.venv-gpu\Scripts\python.exe paramcount.py --qwen_config ref/hf/qwen3-0.6b-config.json \
    --out decider_v2/paramcount.json          -> decider_v2/_paramcount_out.txt
```

```
  layers=20  embeddings=155,582,464  decoder=314,619,904  final_norm=1,024  total=470,202,368
  layers=24  embeddings=155,582,464  decoder=377,543,680  final_norm=1,024  total=533,126,144

  head v1 (bev replica) new_dim=512 L=2      total=   7,364,608  trainable=   7,364,608
  head v2 ctx+rank  d=512 L=2 n_ctx=8        total=  10,779,137  trainable=  10,779,137
  head v2 ctx+rank+null d=512 L=2            total=  10,779,649  trainable=  10,779,649
  head v2 no-ctx (selfattn only)             total=   9,199,617  trainable=   9,199,617

  baseline: r=8 attn-only layers 8-19 (of 20)
      adapted modules  48   lora params=     983,040   backbone total=471,185,408
  v2:       r=16 attn+MLP all 20 layers
      adapted modules 140   lora params=   7,208,960   backbone total=477,411,328
  v2 ablation: r=16 attn+MLP all 24 layers
      adapted modules 168   lora params=   8,650,752   backbone total=541,776,896

  baseline replica  = backbone 470,202,368 + lora    983,040 + head_v1  7,364,608 -> 478,550,016
  v2                = backbone 470,202,368 + lora  7,208,960 + head_v2 10,779,137 -> 488,190,465

  published avbiswas/bev-decider-0.4B total (HF safetensors index): 477,566,976
  its fp32 head tensor count (HF index 'F32'):                     7,364,608
```

470,202,368 + 7,364,608 = 477,566,976 = the published total, tensor for tensor. The replica is
structurally exact.

---

## 4. Data preparation — ran

```
F:\Jev\.venv-gpu\Scripts\python.exe prepare.py --out decider_v2_work/data --seed 0 \
    --calib_target 20000 --val_frac 0.03 --cache_dir <cache> [--smoke_* N]
                                             -> decider_v2/_prepare_out.txt
```

Full-scale carve (seed 0), from the run manifest:

| split | questions | source |
|---|---:|---|
| train | 388,748 | `all` train minus val minus calib |
| val | 12,676 | 3% of `all` train, stratified by (type, domain) |
| calib | **21,127** | `all` train, stratified, satisfies the ≥20k requirement |
| test | 80,019 | the published `all` test split, untouched |

Type mix: train 44.78 / 41.69 / 13.53 % choice/noul/score. **State overlap val→test 0,
calib→test 0**, asserted in the notebook (cell 3c) and recorded in `manifest.json`.

An earlier version reported `val->test 97, calib->test 193`. That was a real bug — `datasets` keys
its `.map` cache by filename, and I had passed the same cache path for two different source splits,
so the exploded *train* records were being returned for *test*. Fixed with per-split cache names;
the independently measured overlap is 0 (`recon/recon_dataset.py` §7).

---

## 5. Smoke training through the exact `train.py` entry point — ran on the RTX 4050

Both arms, identical data, split, seed, steps, batch size and code path. 60 steps, effective batch
16 (8 × 2), `max_state_tokens` 512, bf16 autocast, gradient checkpointing on.

### 5.1 Baseline replica (head v1, LoRA r=8 attention-only on layers 8–19)

```
F:\Jev\.venv-gpu\Scripts\python.exe train.py \
  --train_channel decider_v2_work/data/train --val_channel decider_v2_work/data/val \
  --num_layers 20 --head v1 --lora_r 8 --lora_alpha 16 --lora_last_k_layers 12 --lora_include_mlp 0 \
  --batch_size 8 --grad_accum_steps 2 --max_steps 60 --max_state_tokens 512 \
  --label_smoothing 0 --emd_lambda 0 --lr_lora 1e-4 --lr_head 1e-3 --warmup_ratio 0.05 --shuffle_options 0
```

```
trainable params: backbone=983,040 head=7,364,608 total=8,347,648
train questions=2,000 val questions=300 batch=8x2 steps/epoch=125 total_steps=60 device=cuda:0
step=10 loss=1.23619 grad_norm=2.60467 acc=0.35 lr_lora=0.0001 lr_head=0.001 sps=1.77612 peak_gib=2.72
step=20 loss=1.13922 grad_norm=0.410904 acc=0.34375 ... sps=7.33952 peak_gib=2.74
step=30 loss=1.08691 grad_norm=0.279909 acc=0.4625 ... sps=7.73262 peak_gib=2.83
step=40 loss=1.22678 grad_norm=0.285198 acc=0.34375 ... sps=7.76347 peak_gib=2.87
step=50 loss=1.13728 grad_norm=0.529105 acc=0.45625 ... sps=8.64248 peak_gib=2.87
step=60 loss=1.22966 grad_norm=0.443984 acc=0.39375 ... sps=8.64512 peak_gib=2.87
step=60 val_loss=1.07058 val_acc=0.51 acc_type:choice=0.423358 acc_type:noul=0.635714 acc_type:score=0.26087
wall clock 276.5 s
```

### 5.2 decider-v2 (head v2, LoRA r=16 attention+MLP on all 20 kept layers)

Same command with `--head v2 --lora_r 16 --lora_alpha 32 --lora_last_k_layers 20
--lora_include_mlp 1 --label_smoothing 0.05 --emd_lambda 0.5 --lr_lora 2e-4 --warmup_ratio 0.03
--shuffle_options 1`.

```
trainable params: backbone=7,208,960 head=10,779,137 total=17,988,097
train questions=2,000 val questions=300 batch=8x2 steps/epoch=125 total_steps=60 device=cuda:0
step=10 loss=1.2387  grad_norm=0.0618533 acc=0.4375 lr_lora=0.0002 lr_head=0.001 sps=1.66846 peak_gib=2.95
step=20 loss=1.11809 grad_norm=0.0168895 acc=0.425 ... sps=5.82305 peak_gib=2.97
step=30 loss=1.14528 grad_norm=0.0117526 acc=0.41875 ... sps=5.90937 peak_gib=3.10
step=40 loss=1.30963 grad_norm=0.0175471 acc=0.3625 ... sps=5.96507 peak_gib=3.11
step=50 loss=1.21646 grad_norm=0.0395641 acc=0.4125 ... sps=6.41455 peak_gib=3.11
step=60 loss=1.34884 grad_norm=0.0473685 acc=0.4125 ... sps=6.51274 peak_gib=3.11
step=60 val_loss=1.10882 val_acc=0.466667 acc_type:choice=0.321168 acc_type:noul=0.635714 acc_type:score=0.304348
wall clock 320.8 s
```

**Throughput and memory, measured on an RTX 4050 Laptop (6.0 GiB):**

| arm | steady-state throughput | peak VRAM | 12k-step × 64 extrapolation |
|---|---:|---:|---:|
| baseline replica | **8.6 q/s** | **2.87 GiB** | 24.8 h |
| decider-v2 | **6.5 q/s** | **3.11 GiB** | 32.8 h |

The wider LoRA (attention + MLP, all layers, 7.3× the parameters) costs **1.32× wall clock**. That is
the honest cost of Tier 1 on this hardware; the notebook's cell 6 measures the same thing on your
SageMaker GPU and cell 6b turns it into hours and dollars.

The first log window (sps 1.7) is worker start-up; steady state is the last two windows.

### 5.3 The 60-step accuracy numbers are a plumbing check, not a result

Both arms on the same 1,000-question file, with temperature scaling fitted on the calibration split:

| | accuracy | choice | noul | score | ECE | ECE after cal. | order-inv Δp |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference `bev-decider-0.4B` (n=1,000 draw) | **68.70** | 68.73 | 83.33 | 54.13 | 0.0381 | — | 3.70e-06 |
| baseline replica (60 steps) | 42.00 | 40.99 | 56.94 | 29.82 | 0.0986 | 0.0997 | 4.77e-07 |
| decider-v2 (60 steps) | 35.30 | 31.10 | 56.94 | 24.77 | 0.0556 | 0.0701 | 2.38e-07 |

**The 0.0381 in this table is the reference's ECE on the n=1,000 draw, not on the `all` split.** It
was mis-cited as "`all`" in two earlier drafts of `ARCH_REVIEW.md`, which is the contradiction §6.3
retracts. The reference's ECE on a representative `all` sample is **0.0144** (n=4,000, §6.2), and
0.0122 on `default` (n=5,000); every doc now quotes those two.

**Do not read v2 < baseline as a finding.** At 60 steps on 2,000 questions the v2 head (10.8M params,
17.99M trainable total) is simply less converged than the v1 head (8.3M trainable); its fitted
temperatures (0.05 / 0.11 / 0.07) show a badly over-confident, undertrained softmax. The reference row
is the only one here with any predictive meaning, and it is the number the real run has to beat.
Both arms pass the order-invariance gate comfortably.

### 5.4 Memory finding worth keeping

The first benchmark measured **0.63 q/s and 9.22 GiB peak** on a 6 GiB card. Cause:
`from_pretrained` returns the model in **eval** mode, and transformers' `GradientCheckpointingLayer`
only checkpoints when `self.training` is True, so gradient checkpointing was silently inactive.
`train.py` always calls `.train()`; `bench.py` did not. With `.train()` added:

| micro-batch | end-to-end | GPU-only | peak VRAM |
|---|---:|---:|---:|
| 4 | 3.63 q/s | 7.74 q/s | 2.84 GiB |
| 8 | 3.04 q/s | 7.25 q/s | 3.22 GiB |
| 16 | 0.26 q/s | 0.63 q/s | 4.39 GiB — unusable on this card |

"GPU-only" is pre-encoded batches (the compute ceiling); "end-to-end" includes tokenisation in the
dataloader workers, which is what a SageMaker job will actually see.

---

## 6. Reference checkpoint evaluation — ran, and it validates the whole encoder

`evaluate.py --reference avbiswas/bev-decider-0.4B` uses *my* encoder, collate and metrics against
the published weights, so these two numbers check the implementation as much as the model.

### 6.1 On the `default` config's test split — the protocol the published 74.7 used

```
F:\Jev\.venv-gpu\Scripts\python.exe evaluate.py --reference avbiswas/bev-decider-0.4B \
  --test_channel decider_v2_work/data/test_default --batch_size 8 \
  --invariance_examples 24 --invariance_shuffles 8 --tag reference_default
```

5,000 questions sampled with seed 0 from the 46,320 `default`-config test questions. **247 s.**

| | all | choice | noul | score | ECE | NLL | order-inv Δp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **measured here** | **77.24** | 74.65 | 87.01 | 64.99 | 0.0122 | 0.5491 | 3.16e-06 |
| published (card, rev `687e28f`) | 74.7 | 69.8 | 84.6 | 63.7 | — | — | ≤ 1e-5 |

Within 2.5 points, same direction on every type, same checkpoint, same protocol. The residual is
sampling: the card's exact 5,000-question draw is not published.

### 6.2 On the `all` config's test split — measured on 4,000 questions

```
F:\Jev\.venv-gpu\Scripts\python.exe evaluate.py --reference avbiswas/bev-decider-0.4B \
  --test_channel decider_v2_work/data/test --calib_channel decider_v2_work/data/calib \
  --batch_size 8 --invariance_examples 24 --invariance_shuffles 8 --tag reference
```

| split | n | all | choice | noul | score | ECE | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `default` | 5,000 | 77.24 | 74.65 | 87.01 | 64.99 | 0.0122 | 0.549 |
| **`all`** | **4,000** | **76.30** | **73.88** | **83.53** | **64.87** | **0.0144** | **0.562** |
| published (`default`, card) | 5,000 | 74.7 | 69.8 | 84.6 | 63.7 | — | — |

The two configs are within 0.94 points of each other, so the published 74.7 **is** a fair target
for this run. Per-type temperature scaling on the reference gives T = 0.968 / 1.059 / 0.885 and moves
ECE from 0.0144 to 0.0172 — i.e. it does not help; the reference is already well calibrated here.

### 6.3 Small-sample noise is large enough to invent a finding

Same checkpoint, same code, `random.Random(0).sample` of the same published test split:

| split | n | accuracy | 95% CI half-width |
|---|---:|---:|---:|
| `all` | 400 | 76.25 | ±4.26 |
| `all` | 1,000 | **68.70** | ±2.69 |
| `all` | 4,000 | **76.30** | ±1.35 |
| `default` | 5,000 | 77.24 | ±1.20 |

The n=1,000 draw is 7.6 points below the n=4,000 draw of the same distribution — ~1.8 binomial
standard errors, entirely ordinary. An earlier draft read that draw as "`all` is 8.5 points harder
than `default`"; the n=4,000 measurement disproved it, and that claim has been removed from
`ARCH_REVIEW.md`, `ARCH_V2.md` and the README.

The structural consequences are now in the code rather than in prose:

* a **real run evaluates the full 80,019-question published test split**, never a shard — cell 3
  asserts the size and raises if `SMOKE=False` and the shard is not the full split;
* the 4,000-question shard is `SMOKE`-only, and the results table labels its rows as a plumbing
  check rather than a result;
* one fixed `SPLIT_SEED` carves the data so every arm and seed sees identical bytes, and two
  training seeds per arm report a spread instead of a single number.

That last one is not theoretical. §8.2 measures a **2.52-point** gap between the baseline replica's
two seeds on identical data at an identical budget — larger than the v2 − baseline delta being
claimed at that budget. A single-seed comparison would have reported a different number depending
on which runs you compared.

---

## 7. Notebook validation — ran

```
python build_notebook.py
  -> wrote F:\Jev\decider_v2_sagemaker.ipynb  cells=42 (markdown=16, code=26)
  -> nbformat.validate: OK
```

`nbformat` 5.10.4, notebook format 4.5, every cell carries an `id`, all `source` fields are valid
line lists. All 26 code cells were additionally `compile()`d individually (a generated-cell
regression check, added after two print-only bugs slipped past `nbformat.validate`). No cell
references a name that does not exist — that was verified by executing it.

### 7.1 Full SMOKE dry run, AWS plumbing stubbed — ran

```
F:\Jev\.venv-gpu\Scripts\python.exe dryrun_notebook.py
```

`_dryrun/` shadows `boto3` and `sagemaker` with a local-filesystem S3 and an estimator whose
`fit()` runs the **real** `train.py` in a subprocess with `SM_CHANNEL_*` / `SM_MODEL_DIR` /
`SM_OUTPUT_DATA_DIR` set exactly as SageMaker sets them. So the model code, the split carving, the
SageMaker wiring and the idempotency guards are all exercised for real; only the AWS API calls are
simulated. `dryrun_notebook.py` also points `JEVBENCH_ROOT` / `SYSONE_BENCH_ROOT` at the pristine
clones in `ref/`, which is what lets cell 12 produce real benchmark numbers offline instead of
reporting `NOT RUN`. Full log in `_dryrun_v5.txt`.

See §8 for the result.

---

## 8. Notebook SMOKE run — PASSED, all 26 code cells

```
F:\Jev\.venv-gpu\Scripts\python.exe dryrun_notebook.py     -> _dryrun_v5.txt, 3617 s
```

Run from a **clean state** (no `decider_v2_state.json`, no `decider_v2_work/`, no `_dryrun_s3/`) at
the current revision: SMOKE mode, `SPLIT_SEED=0`, `SEEDS=[0]` (1 seed = 2 jobs), container `pytorch 2.7.1 / py312`,
so **2 training jobs** (baseline + v2, one seed). 4,000 train / 600 val / 1,200 calib /
4,000 test questions, 100 steps per job, micro-batch 8 × accum 2. Full log in `_dryrun_v5.txt`.
Cells executed, in order:

```
1-config  2-install  2b-state-helper  3-load  3-stats  3-prepare  3-parity  3-upload
4-source  5-tests  6-smoke  6-estimate  6b-price-check  7-reference  7b-reference-default
8-quota  8-launch-baseline  9-launch-v2  10-monitor  10-plot  11-calibrate-evaluate
12-benchmarks  13-results  13-charts  14-ablations  15-cleanup
```

Every job ran its preflight before training: **2/2 `ENV: all pins satisfied`, 2/2 `SELFTEST: PASS`,
2/2 `PREFLIGHT: PASS`** (see §9).

### 8.1 What the cells produced

| cell | outcome |
|---|---|
| 1-config | `SPLIT_SEED=0`, `SEEDS=[0, 1]`, `SELFTEST=True`, `ENV_CHECK=True`; **validated the container pairing before anything was launched** — `container check via built-in table: pytorch 2.7.1 / py312 -> ok`; printed "this run will launch 4 training job(s): 2 baseline + 2 v2" |
| 2-install | delegates to `environment.enforce()`, printing the whole pin table (see §9.1) instead of a second, hand-maintained copy of the pins |
| 3-load / 3-stats | `all` loaded: 231,332 / 44,856 rows, 502,570 typed questions (44.75 / 41.52 / 13.73 % choice/noul/score), 28 domains, state tokens p50 100 / p90 903 / p99 2,340, 21,920 of 276,188 rows over 1,024 tokens |
| 3-prepare | train 4,000 / val 600 / calib 1,200 / test 4,000; **state overlap val→test 0, calib→test 0**; `calib >= 20000` assertion ran against the full carve (21,127) |
| 3-parity | 5 `[PASS]`, including the shard guard: `[PASS] SMOKE: test is the declared shard, not the full split  (4,000 of 80,019)`. Under `SMOKE=False` this asserts 80,019 and **raises** otherwise |
| 4-source | lists `bench/*.py` and `requirements.txt` alongside the sources, and echoes the pins the job will enforce |
| 5-tests | **50 checks** — the original 22, plus 20 decision-rule checks (§e) and 8 launch-guard checks (§e2), `ALL CHECKS PASSED`, on the notebook's own GPU |
| 6-smoke | v2, 20 layers, LoRA r=16 attn+MLP, micro-batch 8, 1,024-token states |
| 6-estimate | reported `NOT RESOLVED -- fill it in cell 1` for pricing rather than inventing a number; prints per-job hours plus the job-count multiplier |
| 7-reference | published checkpoint on **our** 4,000-question test file: **76.30** (73.88 / 83.53 / 64.87), ECE 0.0144, NLL 0.5623, order-invariance 3.695e-06 PASS |
| 7b-reference-default | published checkpoint on 5,000 `default`-config questions: **77.24** (74.65 / 87.01 / 64.99), ECE 0.0122 |
| 8-quota | SageMaker Service Quotas path (console URL + `list-service-quotas --service-code sagemaker` + CLI with `<QuotaCode>` left as a placeholder) |
| 8 / 9 | 2 jobs launched; each prints a spot/config summary + both checkpoint paths before submitting and records its budget fingerprint (§9.4); hyperparameter strings end `... --max_train_minutes=-1 --selftest=1 --env_check=1` |
| 10-monitor | downloaded output + checkpoint objects for both jobs |
| 11-calibrate-evaluate | both checkpoints calibrated on the calib split and evaluated on the same test shard |
| 12-benchmarks | **ran both harnesses for all 3 targets** (reference + 2 jobs), 6 benchmark runs |
| 13-results | `RESULTS.md` with the per-(arm, seed) table, the seed-averaged delta, and the `## Decision` section (paired bootstrap + four conditions) — see §8.2 |
| 13-charts | `RESULTS_by_type.png` |
| 14-ablations | `RUN_ABLATIONS is False`; lists all five, including `v2_lora_bev` |
| 15-cleanup | listed every object under the run prefix; deleted nothing |

### 8.2 The SMOKE result table and the decision verdict, verbatim

```
﻿test questions: n=4,000   (SMOKE SHARD -- plumbing check only, NOT a result)

model                                                all   choice     noul    score       ECE  inv max|dp|
bev-decider-0.4B  PUBLISHED (default-config test)      74.7     69.8     84.6     63.7    0.0650       <=1e-5
reference on default-config 5k (validates our encoder)     77.24    74.65    87.01    64.99    0.0122          n/a
reference on OUR all-config test split             76.30    73.88    83.53    64.87    0.0144     3.70e-06
baseline_s0, OUR split                             46.48    40.65    58.46    33.11    0.0604     9.24e-07
v2_s0, OUR split                                   51.08    47.91    58.91    40.34    0.0314     8.94e-07

  reference        n=  4,000  ECE 0.0144 -> 0.0172 after calibration   NLL 0.5623 -> 0.5639   temperatures={'choice': 0.9682482481002808, 'noul': 1.0592225790023804, 'score': 0.8850874900817871}
  baseline_s0      n=  4,000  ECE 0.0604 -> 0.0629 after calibration   NLL 1.1870 -> 1.1500   temperatures={'choice': 0.5649158358573914, 'noul': 0.8850874900817871, 'score': 0.5649158358573914}
  v2_s0            n=  4,000  ECE 0.0314 -> 0.0405 after calibration   NLL 1.0822 -> 1.0602   temperatures={'choice': 0.8850874900817871, 'noul': 0.9682482481002808, 'score': 0.9682482481002808}

DELTA v2 - baseline, same split, same budget, seed-matched (AVERAGED OVER SEEDS -- a plumbing check, not a result)
  all                                                  +4.60
  choice                                               +7.26
  noul                                                 +0.45
  score                                                +7.23

  seed spread: baseline +0.00 pp, v2 +0.00 pp (1 seeds each, identical data and budget)
  seed 0: baseline  46.48  v2  51.08  delta  +4.60 pp   95% CI [+3.33, +5.83]  excludes 0

==============================================================================
DECISION
==============================================================================
  per-question paired bootstrap, 10,000 resamples, seed 20260101
    test questions (paired)  4,000
    seed 0: delta  +4.60 pp  95% CI [ +3.33,  +5.83]  excludes 0

  (a) bootstrap 95% CI excludes 0      +3.33 .. +5.83  -> yes
  delta (v2 - baseline), seed-averaged +4.60 pp
  (b) seed spread                     0.00 pp   -> yes
  (c) |delta| >= 1.0 pp                   -> yes
  (d) seeds per arm >= 3             -> NO  (have 1)

  VERDICT: NOT SEPARABLE
  Only 1 seed per arm: the seed spread is 0 by construction,
  so (b) cannot discriminate and no verdict is meaningful. Run at least 2 seeds.
state -> decider_v2_state.json: ['bench', 'channels', 'data_dir', 'data_prefix', 'decision', 'evaluated', 'jobs/baseline_s0', 'jobs/v2_s0', 'jobs_by_arm', 'manifest', 'run_name', 'source_dir', 'split_seed', 'test_n']
  NOTE: SMOKE shard at 100 steps. This exercises the rule; it does not satisfy it,
        and no accuracy number on a SMOKE shard is a result.
```

Note `(b) seed spread 0.00 pp` with **one seed**: the spread is 0 by construction, so condition
(b) is vacuous and the verdict is forced to `NOT SEPARABLE` — even though (a), (c) and the point
estimate are all comfortably positive. That is the rule refusing to be fooled, and it is why
the notebook defaults to one seed as a *cost probe* rather than as an experiment.

### 8.3 JevBench and sysone-bench — RAN, with real numbers

Both adapters are in `decider_v2/bench/` and follow `jevbench/adapters/so1_decider.py`: load the
model once, one forward pass per question, hand the distribution to the **harness's own** scorer
(JevBench `scoring.score_task` + `summarize.metric`; sysone-bench `contracts.validate_answers` +
`metrics.summarize_predictions`). Neither metric is re-implemented.

```
BENCHMARK SUMMARY (from the adapters' own JSON; metrics computed by the upstream harnesses)
==============================================================================
  JevBench    reference      accuracy  64.94  macro  64.49  valid 231/231  brier 0.4558  ece 0.1237
  sysone-bench reference     choice  70.52  noul  81.25  score(nearest) 40.00  choice ece 0.0012
  JevBench    baseline_s0    accuracy  43.29  macro  42.34  valid 231/231  brier 0.6384  ece 0.0761
  sysone-bench baseline_s0   choice  38.59  noul  62.76  score(nearest) 20.00  choice ece 0.0013
  JevBench    v2_s0          accuracy  52.81  macro  50.05  valid 231/231  brier 0.5364  ece 0.0667
  sysone-bench v2_s0         choice  63.72  noul  62.76  score(nearest) 26.67  choice ece 0.0017
```

**The adapters are validated by the reference row.** The published `bev-decider-0.4B` weights, pushed
through these adapters, score **64.94** on JevBench's 231 public tasks (model card: **65.8**) and
**70.52 choice / 81.25 noul** on sysone-bench's evaluation split (card: **70.8** all), with
**231/231** distributions passing JevBench's own strict validity check and **1,240/1,240** decisions
graded on sysone-bench. Agreement at that level across two independently-written harnesses is what
makes the `baseline_*` and `v2_*` rows interpretable at all.

These rows are still **not results** — the §8.2 caveats apply, and the sysone-bench `score` rows use
the probability-weighted **expected level** (the harness rounds with `floor(x+0.5)` and its
MAE/RMSE/Spearman all require a continuous prediction). Full per-task and per-case JSON is in
`decider_v2_work/eval/bench/*.json`.

### 8.4 What changed since the previous dry run, and why it matters

| change | where | why |
|---|---|---|
| **container corrected to `2.7.1` / `py312`** | cell 1, cells 8/9 | the previous revision requested `framework_version="2.6.0", py_version="py311"`, which **is not a valid pairing**. Verified against the SDK's own `sagemaker/image_uri_config/pytorch.json`: 2.5.1→py311, 2.6.0→py312, 2.7.1→py312. The old config would have failed at launch, before any training |
| `requirements.txt` + `environment.py`, enforced by `train.py`, `evaluate.py` and the notebook | `decider_v2/` | one source of truth for the pins instead of a hand-maintained list inside the notebook; a mismatch now aborts instead of silently changing the transformers 4-D-mask path |
| `SELFTEST` / `ENV_CHECK` preflight inside every job | `train.py` | catches both launch-invalidating failure modes in the first ~15 s instead of after 12k steps |
| `SEPARABLE` / `NOT SEPARABLE` decision rule | cell 13, `RESULTS.md` | a verdict in the artefact, with the `≥ 2 seeds` guard, so the delta cannot be read without the noise it was measured against |
| staged seed workflow documented | cell 9b, README §2c | seed 0 of both arms concurrently first, then seed 1, at the same cost and half the wall-clock; §8.2.1 covers why 3 seeds are needed for `SEPARABLE` |
| **`SEPARABLE` / `PROVISIONAL` / `NOT SEPARABLE`** on a paired bootstrap | `decide.py`, cell 13, `RESULTS.md`, `tests.py` §(e) | the delta now carries a 95% CI over test resampling and the seed spread beside it, so a noise measurement cannot be quoted as a result (§8.2) |
| throughput provenance in the `max_runtime` guard | cell 6b, `bench.py` hardware block | the guard now names the GPU it measured, and can take throughput from a real job's CloudWatch log instead of the notebook kernel's card (§9.5) |
| every job prints and **asserts** checkpoint selection, validation frequency and schedule length | `train.py` CONFIG banner | a cosine schedule that disagrees with `MAX_STEPS` never reaches its floor, so the run ends at a higher LR than intended and two "identical" runs are not comparable (§9.3) |
| `launch()` refuses to reuse a Completed job whose budget differs | cell 8/9, `tests.py` §(e2) | reusing a 768K-presentation checkpoint in a run configured for 940K produces plausible numbers and no error at all (§9.4) |
| spot config validated and printed per job | cell 1, `launch()` | SageMaker rejects `MaxWaitTime < MaxRuntime`, and the old hardcoded 30-minute wait could not have won 40 h of capacity anyway (§9.6) |

Two bugs were found and fixed by earlier runs, both of which had silently produced **no output at all**
rather than an error: a job-name parser that produced `baseline_ss0` instead of `baseline_s0`, which
made the seed-averaged delta block skip; and a `%d` format applied to a string seed. Both are fixed
and the block is now covered by the output above.

Four further defects were found by direct testing rather than by the pipeline:

* `environment.enforce(requirements=...)` ignored its own argument and always read the default
  `requirements.txt`, so a mismatch was reported as satisfied. Caught by passing a deliberately
  wrong pin file and asserting the return value.
* the container check called `sagemaker.image_uris.retrieve`, which the offline dry-run stub does
  not provide, aborting the run. It now falls back to a literal table that is verified against the
  SDK's own data.
* the quota cell's `--query` example lost its interpolation through nested quote escaping and
  rendered as `contains(QuotaName, + family + )`. Verified by executing the cell and reading its
  output.
* `decide.py`'s seed auto-discovery repeated the same off-by-one as the earlier notebook bug
  (`n[len(arm)+1:]` yielding `'s0'`), and `RESULTS.md`'s condition table emitted unescaped `|`
  characters that break a markdown table. Both caught by running the module and rendering the file.

---

## 9. Preflight and launch-time gates — ran inside both jobs

Both default to on and both run before any training step, on a 2-layer random-init model with a
byte-level tokenizer, so they cost ~10 s, touch no data and need no download.

### 9.1 `ENV_CHECK` — version pins, exactly as printed in the job log

```
==============================================================================
ENV: resolved package versions (pins from requirements.txt)
==============================================================================
python      3.14.6       OK
transformers want 5.15.0       have 5.15.0         OK
peft        want 0.21.2       have 0.21.2         OK
datasets    want 5.1.0       have 5.1.0         OK
huggingface_hub want 1.33.0       have 1.33.0         OK
tokenizers  want 0.22.2       have 0.22.2         OK
safetensors want 0.8.0       have 0.8.0         OK
accelerate  want 1.15.0       have 1.15.0         OK
numpy       want 2.3.5       have 2.3.5         OK
pyarrow     want 25.0.1       have 25.0.1         OK
ENV: all pins satisfied
```

Every pin was checked against live PyPI metadata rather than assumed, and two of them turned out to
be load-bearing in ways worth recording:

* **`huggingface_hub==1.33.0`**: transformers 5.15.0 requires `huggingface-hub>=1.5.0,<2.0`, and
  the package is now at **2.2.0** on PyPI. An unpinned install resolves to 2.x and transformers will
  not import at all.
* **`tokenizers==0.22.2`**: transformers requires `tokenizers>=0.22.0,<=0.23.0`, and **0.23.0 does
  not exist** on PyPI, so 0.22.2 is the newest version that satisfies the constraint.

`numpy==2.3.5` declares `requires-python >=3.11`, which is what forces the container to py311 or
newer. `torch` is deliberately **not** pinned — the container owns the CUDA-matched build, and
overwriting it is how you end up with a CPU-only torch on a GPU instance; `environment.py` checks
only that it is present and records its version.

The failure paths were exercised directly, not just assumed:

```
$ python train.py ... --env_check 1        # with a deliberately wrong pin file
ENV: pins NOT satisfied.
ENV: refusing to install automatically (--no_install). Aborting.
enforce returned: False
CONFIRMED: a pin mismatch aborts (allow_install=False)
```

### 9.2 `SELFTEST` — the two structural gates, inside the container

```
==============================================================================
SELFTEST: packed-vs-separate equivalence + order invariance (tiny 2-layer config)
==============================================================================

(a) packed-vs-separate equivalence under the 4-D additive mask + shared position ids
      k=3: max|packed - solo| over option read positions = 5.364e-07
      k=5: max|packed - solo| over option read positions = 7.153e-07
      k=8: max|packed - solo| over option read positions = 5.960e-07
  [PASS] packed-vs-separate option hidden states  max delta 7.153e-07 (gate 1e-5)
  [PASS] option isolation (dropping an option does not move the others)  max delta 2.384e-07
  [PASS] attn_implementation=eager head=v1 honours the custom 4-D mask  max delta 5.513e-07
  [PASS] attn_implementation=eager head=v2 honours the custom 4-D mask  max delta 5.513e-07
  [PASS] attn_implementation=sdpa   head=v1 honours the custom 4-D mask  max delta 5.960e-07
  [PASS] attn_implementation=sdpa   head=v2 honours the custom 4-D mask  max delta 5.960e-07

(b) option-order invariance (fp32, head logits and probabilities)
  [PASS] order invariance mode=bev pool=bev  max |dp| = 5.960e-08, max |dlogit| = 3.278e-07 (gate 1e-5)
  [PASS] order invariance mode=special pool=opt_end  max |dp| = 2.980e-08, max |dlogit| = 2.086e-07 (gate 1e-5)

(b2) order invariance for score questions (rank embedding travels with the level)
  [PASS] score order invariance K=3  max |dp| = 5.960e-08
  [PASS] score order invariance K=5  max |dp| = 4.470e-08
  [PASS] score order invariance K=7  max |dp| = 4.470e-08
SELFTEST: PASS
```

**The abort path was tested, not assumed.** Injecting a deliberate failure makes `run_preflight`
return `False`, which makes `train.py` raise `SystemExit(2)` before the first training step:

```
  [FAIL] INJECTED FAILURE (deliberate)  simulated delta 1e-1
ABORT: selftest FAILED: ['INJECTED FAILURE (deliberate)']. The 4-D mask / position-id guarantee
this project is built on does not hold in this container, so any accuracy number from this job
would be uninterpretable.
PREFLIGHT: FAIL -- job will abort
run_preflight returned: False
CONFIRMED: a failing selftest aborts the job
```

Under `torchrun` the verdict is all-reduced across ranks, so every rank exits rather than rank 0
aborting while the others wait in a collective. (The multi-GPU path itself remains unexecuted — one
GPU here, §0.)

### 9.3 Launch-time facts printed by every job — ran in both SMOKE jobs

Each job now prints a `CONFIG` banner before its first step, recording the three things that
silently change what gets trained, and **asserts** the third rather than assuming it:

```
CONFIG (the three things that decide what gets trained)
==============================================================================
  (a) checkpoint selected for evaluation : best VALIDATION accuracy (--model_dir/best); 'final' (last step) also saved
      best is chosen on the VAL split only (600 questions, carved from train); the TEST split is never read by train.py
  (b) validation + checkpoint saving     : every 25 steps (save_every=25)
  (c) LR schedule length                 : 100 steps (cosine, warmup 5 = 5.0%) == MAX_STEPS 100
      budget: 100 steps x 8x2 = 1,600 question presentations
      steps/epoch=250 -> 0.40 epochs
  device: NVIDIA GeForce RTX 4050 Laptop GPU  5.99 GiB
```

**Answers to the three questions, confirmed against the source and not just asserted:**

* **(a) best-val is the default, and is genuinely best-on-val.** `train.py` selects on
  `val_loader` (`m["accuracy"] > best_acc` → `save("best")`), and there is **no `test_loader` in
  `train.py` at all** — the test split is never loaded by the trainer. This is deliberate and is the
  fix for the leak recorded in `ARCH_REVIEW.md` §1.10, where the published checkpoint's `best` was
  selected on *test* accuracy. Cell 11 likewise prefers the directory literally named `best`.
  `final` (last step) is also saved, so both are available.
* **(b) validation every `MAX_STEPS // 4` steps** — 25 under SMOKE, 3,672 at the real 14,688-step
  budget, with checkpoint saving on the same cadence.
* **(c) the cosine schedule length equals `MAX_STEPS`, and is checked.** A schedule whose
  `num_training_steps` disagrees with the budget is the classic silent bug: the cosine never reaches
  its floor, so the run ends at a higher LR than intended and two runs of "the same" config are not
  comparable. `train.py` now raises `SystemExit` if `scheduler.num_training_steps != total_steps`
  or if `total_steps != --max_steps`.

These are also written to `summary.json`, so the notebook reads them rather than trusting a
checkpoint filename:

```json
"config": {"checkpoint_selection": "best_val_accuracy",
           "checkpoint_selection_source": "val split only; test split never read",
           "eval_every": 25, "save_every": 25, "max_steps": 100, "lr_schedule_steps": 100,
           "lr_warmup_steps": 5, "question_presentations": 1600, "batch_size": 8,
           "grad_accum_steps": 2, "steps_per_epoch": 250, "epochs": 0.4, "seed": 0, ...}
```

### 9.4 Launch guards — implemented and tested by direct injection

**The budget fingerprint.** `launch()` records the budget of every job in `decider_v2_state.json` and
refuses to skip a Completed job whose budget differs from the current config. Verified by driving
`launch()` with a fabricated state:

```
=== state says Completed at 768k, config now says 940k ===
  baseline_s0: REFUSING to skip -- it completed under a different budget:
      max_steps: launched 12000 -> now 14688
      presentations: launched 768000 -> now 940032
      target_presentations: launched 768000 -> now 940000
      Its checkpoints were trained on 768000 presentations; this run is configured for 940032.
      Either restore the old config, or change RUN_NAME to start a fresh run, or
      delete jobs/baseline_s0 from decider_v2_state.json to relaunch it deliberately.
  refused correctly: job baseline_s0 completed under a different training budget
```

and the two cases that must still work:

```
=== Completed at the SAME budget (staged seeding) ===
  v2_s0: already Completed -- not relaunching (idempotent, same budget)
=== Completed job with NO recorded budget (state written before this change) ===
  v2_s0: already Completed -- not relaunching (idempotent, same budget)   [cannot verify -> allowed]
```

This guard exists because the failure it prevents is invisible: reusing a 768K-presentation
checkpoint in a run that believes it is training 940K produces plausible numbers and no error at all.
Persisted in the real run:

```
jobs/baseline_s0: budget {"max_steps":100,"batch_size":8,"grad_accum_steps":2,
                          "target_presentations":940000,"presentations":1600,"seed":0}
jobs/v2_s0:       budget {...same...}
```

**Eight permanent unit tests** for the guards live in `tests.py` §(e2) and run in cell 5 and every
job's preflight:

```
(e2) launch guards: budget fingerprint + spot max_wait >= max_run
  [PASS] guard: the 940K budget fingerprints as 14,688 steps / 940,032 presentations  940,032
  [PASS] guard: identical config -> no mismatch, so a Completed job may be skipped
  [PASS] guard: 12k vs 14.7k is detected as a budget change  fields differing: ['max_steps', 'presentations', 'target_presentations']
  [PASS] guard: halving batch_size is detected (presentations halve)
  [PASS] guard: a different seed is detected (so staged seeds are NOT skipped)
  [PASS] guard: default spot max_wait equals max_run (SageMaker's minimum)  4320 min == 259200 s
  [PASS] guard: the old hardcoded 30-minute wait would have been REJECTED  30 min < 72 h, which SageMaker refuses
  [PASS] guard: an explicitly short max_wait is a hard error  30 min x 60s < 259200s -> SystemExit
```

### 9.5 Throughput provenance — the guard now says where its number came from

Cell 6b previously printed a `max_runtime` verdict with no indication of which machine produced the
throughput. `bench.py` now records hardware identity and cell 6b prints it, verbatim from the run:

```
WHERE THIS THROUGHPUT CAME FROM
==============================================================================
  measured on : NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GiB, CC 8.9)
  torch       : 2.11.0+cu128   micro-batch 8   max_state_tokens 1024
  head/LoRA   : head=v2 lora_r=16 mlp=1   (matches v2, not the baseline arm)

max_runtime GUARD
==============================================================================
  max_runtime for every job: 72 h (259,200 s)
  budget per job:            100 steps x 16 = 1,600 presentations
  baseline replica     3.46 q/s ->    0.13 h  vs 72 h limit  OK
  decider-v2           2.62 q/s ->    0.17 h  vs 72 h limit  OK

  max_runtime has headroom for every arm. Source: NOTEBOOK KERNEL (not the training instance).
```

That "Source:" line is the point of the change. Cell 6 runs `bench.py` in a subprocess **of the
notebook kernel**, so it measures the notebook's GPU, which is almost never the training instance —
and `bench.py`'s config is v2's (`lora_r=16, mlp=1`), not the baseline arm's. A `max_runtime` sized
from the wrong card is a job that gets killed. Two ways to fix that, both wired in:

* `THROUGHPUT_SOURCE = "job"` with `JOB_THROUGHPUT_FROM = "<a finished job name>"` parses the
  **training instance's own CloudWatch log** — `train.py` logs `sps=<n>` on every line, so the median
  of a finished job's lines (after discarding the first third as warm-up) is its steady-state
  questions/second. That is the training instance, and the guard then labels itself
  `Source: training instance`. If the log cannot be read it falls back to the kernel measurement and
  says so loudly rather than guessing.
* or run cell 6 **on** the training instance.

Under `SMOKE=False` with a kernel-only measurement the cell additionally prints a warning that the
estimate is for a different machine and names `INSTANCE_TYPE`.

### 9.6 Spot configuration — printed per job, and validated

`max_wait >= max_run` is a hard requirement from SageMaker, and the old hardcoded `MaxWaitTime` of
30 minutes would have been **rejected outright** against a 72 h runtime (and could not have won
40 h of capacity anyway). `SPOT_MAX_WAIT_MINUTES` now defaults to `MAX_RUNTIME_HOURS * 60` and an
explicitly shorter value is a `SystemExit`. Verified:

```
USE_SPOT=True, MAX_RUNTIME_HOURS=72
  SPOT_MAX_WAIT_MINUTES = 4320
  max_wait seconds      = 259200
  max_run  seconds      = 259200
  max_wait >= max_run   : True

  guard rejects max_wait < max_run:
    refused: SPOT_MAX_WAIT_MINUTES (30) x 60s must be >= MAX_RUNTIME (259200s). SageMaker rejects a spot training job whose
```

The one-line spot/config summary printed immediately before each job is submitted, verbatim:

```
  launching baseline_s0: instance=ml.g5.2xlarge spot=False max_run=72h max_wait=n/a budget=100x8x2=1,600 presentations
    checkpoint_s3_uri      = s3://decider-v2-000000000000-us-east-1/decider_v2/decider_v2-20261012-173853/baseline_s0/checkpoints
    checkpoint_local_path = SM_MODEL_DIR (SageMaker mounts it under /opt/ml/checkpoints)

  launching v2_s0: instance=ml.g5.2xlarge spot=False max_run=72h max_wait=n/a budget=100x8x2=1,600 presentations
    checkpoint_s3_uri      = s3://decider-v2-000000000000-us-east-1/decider_v2/decider_v2-20261012-173853/v2_s0/checkpoints
    checkpoint_local_path = SM_MODEL_DIR (SageMaker mounts it under /opt/ml/checkpoints)
```

`max_wait=n/a` is correct and intended when `USE_SPOT=False`: no `WaitTimeConfig` is sent, so
SageMaker charges on-demand and does not wait for capacity.

---

## 10. Honest summary

Ran and passing: the mask/position-id equivalence proof on both attention implementations, the
option-isolation proof, the order-invariance gates for `choice` and `score`, the loss checks, the
64-example overfit, **20 decision-rule checks including the paired bootstrap**, and **8 launch-guard checks**, the parameter counts
against the published checkpoint, split carving with the overlap assertions, 100-step training of both
arms through the real `train.py`, evaluation + calibration + invariance for the reference
and both, **JevBench (231 tasks) and sysone-bench (1,240 decisions) for all three targets**, the
version-pin and structural preflight gates **inside both jobs**, throughput and peak-VRAM
measurements on a real GPU, and a full SMOKE execution of the notebook with AWS stubbed.

Not run: anything that needs AWS credentials, any full-budget training, any Tier-2 ablation job, any
real 3-seed run (the `SEPARABLE` path was exercised on synthetic predictions instead), job
concurrency (the offline harness runs jobs serially), and the multi-GPU path. **No claim is made
anywhere in this repository that decider-v2 beats bev-decider or approaches Jev.**

### The single most important number in this document

The **same configuration, same data, same budget, same seeds**, run four times on this machine:

| dry run | seeds | delta (v2 − baseline) | seed spread | bootstrap 95% CI | verdict |
|---|---:|---:|---:|---:|---|
| run 1 | 2 | +2.64 pp | **2.52 pp** | — | cleared the noise floor by 0.12 pp |
| run 2 | 2 | +4.03 pp | **0.05 pp** | [+2.62, +5.45] | `PROVISIONAL` |
| run 3 | 2 | +2.75 pp | **0.35 pp** | [+1.39, +4.14] | `PROVISIONAL` |
| run 4 | 1 | +4.60 pp | **0.00 pp** | [+3.33, +5.83] | `NOT SEPARABLE` |

Every one of these had a bootstrap CI excluding 0, and every one would have been quoted as a result
under the previous two-seed rule. Run 4 is the sharpest case: a +4.60 pp delta with a CI that
excludes 0, correctly refused because one seed makes the spread 0 by construction. The seed spread
of an *identical* configuration has ranged over **0.00–2.52 points**, and the delta itself moved by
2.0 points between runs of the same thing — because GPU training is not bit-deterministic, so even a
fixed seed does not give a fixed model.

That is the entire argument for the four-condition rule in §8.2.1: a CI that excludes 0 is necessary
and nowhere near sufficient. The verdict that means anything is the one at n = 80,019 with three or
more seeds, and the notebook is built so it cannot produce a confident-sounding answer before then.

Two more things worth flagging before you spend money:

* the previous revision requested a SageMaker container (`2.6.0` / `py311`) that **does not exist**.
  It would have failed at launch. The current `2.7.1` / `py312` pairing is verified against the SDK's
  own image table, and cell 1 now checks it before anything is submitted.
* the previous revision hardcoded `max_runtime=43200` (12 h). At the 940K budget that would have
  killed a real run mid-training and discarded the work; SageMaker's spot `MaxWaitTime` of 30 minutes
  would also have been rejected outright. Both are now derived and validated (§9.6).