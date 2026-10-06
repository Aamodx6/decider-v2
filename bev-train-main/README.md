# Training JEV networks from Qwen

Train a network that answers choice, yes/no (`noul`), and score questions using a Qwen backbone and a small attention head. Choices share the same starting position IDs, and the attention mask prevents each choice from seeing the others. The head scores the choices without depending on their order.

[Watch the video: Training JEV networks from Qwen](https://youtu.be/sF3CNPbWA8o)

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)

## Start here: `arch.ipynb`

Open [`arch.ipynb`](arch.ipynb) for the architecture walkthrough from the livestream: tokenize the question and choices separately, build their position IDs, and construct the attention mask. The notebook explains the architecture; the Python files implement training and inference.

- [`tokenization.py`](tokenization.py): prompt formatting, choice positions, and attention masks.
- [`dataset.py`](dataset.py): dataset configs, the state-grouped dev split, typed questions, soft labels, and batching.
- [`network.py`](network.py): Qwen backbone, optional LoRA, and the choice-scoring head (`ctx_queries` cross-attention).
- [`train.py`](train.py): training, the flagged loss options, gradient accumulation, dev validation, and checkpoint initialization.
- [`logger.py`](logger.py): run metrics and LoRA/head checkpoints.
- [`inference.py`](inference.py): evaluation (per type and per domain, AUC, ECE) and typed answers through `answer()`.
- [`calibrate.py`](calibrate.py): per-task-type temperature scaling fitted on the dev split.
- [`tests/`](tests/): mask/position, order-invariance, parity, calibration, and smoke tests (pytest).

Dataset: [avbiswas/bev-decision](https://huggingface.co/datasets/avbiswas/bev-decision). Training loads this dataset automatically (the old `bev-decision-150K` id redirects here).

Original livestream: [Architecture walkthrough](https://youtube.com/live/AzxoU7kxjig).

## Setup

Install [uv](https://docs.astral.sh/uv/), then install the locked dependencies:

```bash
uv sync
```

The first run downloads the Qwen3-0.6B weights, tokenizer, and dataset. The code selects CUDA, then Apple MPS, then CPU. Full training configs are intended for a GPU; reduce the batch size for smaller devices.

For a fresh Ubuntu GPU machine, [`cloud_startup_script.sh`](cloud_startup_script.sh) installs the system tools, sets up uv, and checks CUDA availability:

```bash
bash cloud_startup_script.sh
```

## Train

For a small overfitting sanity check, use the 10-question config:

```bash
uv run python train.py configs/smoke.yaml --name smoke
```

This config runs 30 epochs to check whether the model can learn those questions. For full training with the first 20 Qwen layers and LoRA on the last 12 kept layers (8–19):

```bash
uv run python train.py configs/full_lora_k12.yaml --name full-lora-k12
```

That config trains the choice head and LoRA adapters, uses 1,024 state tokens, and accumulates two microbatches of 32 questions per optimizer step. `optim.grad_accum_steps` defaults to 1. Logging, validation, and saving intervals count optimizer steps. An incomplete final accumulation group is skipped.

Other configs include [`full_head_only.yaml`](configs/full_head_only.yaml) for a frozen backbone and [`full_lora.yaml`](configs/full_lora.yaml) for LoRA on the last four of 20 kept layers.

Each run writes its config, metrics, summary, and checkpoints under `runs/<name>-<timestamp>/`. Checkpoints contain the head and, when enabled, LoRA weights; the Qwen base weights are loaded separately. `best` tracks **dev** accuracy, `latest` is overwritten at each save interval, and `final` is saved at the end.

## Data: configs and the dev split

`data.configs` lists any of `default`, `hard_50k`, `numeric_temporal`, `skills`, `counterfactual_15k`, `all`
(the default is `[default]`); several are concatenated. The test split is **never read during training** —
the old code selected `best` on it, which leaked. Instead, `data.dev_fraction` (default 0.05) carves a dev
split out of train, **grouped by normalized state** so a repeated state never straddles train and dev.
Best-checkpoint selection, calibration, and `calibrate.py` all use dev; the test split is only read by
`inference.py` for final reporting. `data.max_dev_questions` caps the dev sample (old configs without it
fall back to `max_test_questions`).

States can be long (up to ~17K chars in `skills`); `data.max_state_tokens` truncates them. The attention
mask is a full 4-D `B x L x L` tensor, so for longer `max_state_tokens` use a smaller `batch_size` with
`optim.grad_accum_steps`.

## decider-v2 flags (all default off)

Every change sits behind a config flag, so the unmodified configs remain the reproducible baseline and
each change can be ablated alone (prompt layout, markers, position ids, and attention masks are untouched —
they are what guarantees option-order invariance).

| Change | Flag | Values (default first) |
| --- | --- | --- |
| Loss | `loss.label_smoothing` | 0.0, 0.05 |
| Loss (score) | `loss.emd_weight` | 0.0, 0.5 (lambda x CDF-EMD) |
| Loss (soft labels) | `loss.use_soft_labels` | false, true (uses `label_probs` from `skills`) |
| Loss (per-type mean) | `loss.balance_types` | false, true (average within type, then across types) |
| Head context | `head.ctx_queries` | 0, 8 learned queries cross-attending to the prefix |
| LoRA scope | `lora.target` | `attn`, `attn_mlp` (+ gate/up/down) |
| LoRA rank | `lora.r` / `lora.alpha` | 8/16, 16/32 |
| Depth | `model.num_layers` | 20, 24, 28 |

Notes: label smoothing is computed over valid options only (padded options have `-inf` logits, which would
make `F.cross_entropy`'s smoothing term infinite). `label_probs` (only in `skills`, ~6.9K questions) is a
dict keyed by criteria keys for `choice`, `{"true": p, "false": p}` for `noul`, and a list for `score`;
`label` is its argmax in every case, and the dicts are reordered to match the option order at load time.

## Calibration

Fit one temperature per task type on dev (torch LBFGS, no new dependencies) and write `calibration.json`
into the checkpoint folder; `load_checkpoint()` picks it up, so `answer()` and `inference.py` report
calibrated probabilities afterwards:

```bash
uv run python calibrate.py runs/<run-id>/checkpoints/best --config configs/full_lora_k12.yaml
```

The file records the temperatures plus ECE (10 bins, top-label) overall, per type, and per domain, and the
NLL before/after.

## Start from saved weights

Pass a checkpoint directory or a run ID (which selects that run's final checkpoint):

```bash
uv run python train.py configs/full_lora_k12.yaml --name continued \
  --resume runs/<run-id>/checkpoints/final
```

Use the same backbone, head architecture, and LoRA configuration as the checkpoint. `--resume` initializes a new training run from the saved weights: the optimizer, learning-rate schedule, and step counter start fresh.

## Evaluate

```bash
uv run python inference.py runs/<run-id>/checkpoints/best \
  --config hard_50k --max_questions 0 --batch_size 16 --num_workers 4
```

Evaluation reports loss, overall accuracy, accuracy **per task type** and **per `domain`**, yes/no AUC when
both label classes are present, and ECE (10 bins, top-label) overall, per type, and per domain. It uses the
checkpoint's state and choice token limits, and applies `calibration.json` when present. `--config` selects
the dataset config to evaluate (`default`, `hard_50k`, `numeric_temporal`, `skills`, `counterfactual_15k`,
`all`; default: the configs recorded in the checkpoint) and `--max_questions 0` means the full split.

## Tests

```bash
uv run pytest                 # everything, including the smoke run (minutes)
uv run pytest -m "not smoke"  # fast suite only
```

The order-invariance test prints its measured maximum per-key probability difference (fp32, random option
permutations, 3 to 12 options); the gate is <= 1e-5. The smoke test runs `configs/smoke.yaml` in a
subprocess and asserts it still overfits the 10 questions.

## Ablation run plan

Baseline = [`full_lora_k12.yaml`](configs/full_lora_k12.yaml) with every flag off. Every ablation changes
exactly one flag, on the same data (`all`), the same dev split (seed 0), and the same step budget.
[`small.yaml`](configs/small.yaml) (2k questions, 1 epoch) is the fast version of the same ladder.

| Run | Command | Flag changed |
| --- | --- | --- |
| baseline | `uv run python train.py configs/full_lora_k12.yaml --name baseline` | none |
| loss | `uv run python train.py configs/full_loss.yaml --name abl-loss` | `loss.label_smoothing=0.05, emd_weight=0.5, balance_types=true` |
| ctx | `uv run python train.py configs/full_ctx.yaml --name abl-ctx` | `head.ctx_queries=8` |
| lora mlp | `uv run python train.py configs/full_lora_mlp.yaml --name abl-lora-mlp` | `lora.target=attn_mlp, r=16, alpha=32` |
| layers 24 | `uv run python train.py configs/full_layers24.yaml --name abl-layers24` | `model.num_layers=24` |
| layers 28 | `uv run python train.py configs/full_layers28.yaml --name abl-layers28` | `model.num_layers=28` |
| soft labels | `uv run python train.py configs/full_softlabels.yaml --name abl-softlabels` | `loss.use_soft_labels=true` |
| stage 2 (optional) | `uv run python train.py configs/full_stage2.yaml --name stage2 --resume runs/<run-id>/checkpoints/final` | `data.configs=[counterfactual_15k, hard_50k]`, lr / 4 |

Each run writes `runs/<name>-<timestamp>/{config.yaml, metrics.jsonl, summary.json, checkpoints/}`.
After training, fit calibration once per run and evaluate the test split of each config:

```bash
uv run python calibrate.py runs/<run-id>/checkpoints/best --config configs/<config>.yaml
uv run python inference.py runs/<run-id>/checkpoints/best --config all --max_questions 0
```

## Results

Numbers below come from actual runs on this machine (RTX 4050, seed 0): the **small ladder** — 2k
training questions, 256-token states, 1 epoch — one run per row, each evaluated the same way:
dev-selected `best` checkpoint, `calibrate.py`, then `inference.py --config all --max_questions 1000`
(the first 1,000 test questions of `all`, a fixed seed-0 sample shared by every run, so the columns
are comparable across rows; the full 80k-question test split is *not* what these numbers measure).

| Run | acc | acc choice | acc noul | acc score | ECE | ECE choice | ECE noul | ECE score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.485 | 0.449 | 0.547 | 0.413 | 0.054 | 0.032 | 0.114 | 0.064 |
| loss | 0.465 | 0.412 | 0.547 | 0.394 | 0.068 | 0.043 | 0.130 | 0.048 |
| ctx (8) | 0.420 | 0.352 | 0.544 | 0.250 | 0.096 | 0.086 | 0.143 | 0.019 |
| lora attn_mlp r=16 | 0.525 | 0.516 | 0.549 | 0.471 | 0.055 | 0.084 | 0.121 | 0.083 |
| layers 24 | 0.462 | 0.426 | 0.542 | 0.317 | 0.086 | 0.108 | 0.104 | 0.053 |
| layers 28 | 0.463 | 0.408 | 0.544 | 0.404 | 0.081 | 0.070 | 0.121 | 0.084 |
| soft labels | 0.474 | 0.428 | 0.547 | 0.404 | 0.051 | 0.049 | 0.106 | 0.080 |

Per domain (same runs and sample, accuracy / ECE per `domain` key of the dataset; the last column
points at the remaining 23 domains in each run's eval output):

| Run | sentiment_emotion_and_moderation | retail_product_and_shopping | spatial_and_logical_reasoning | support_and_intent_routing | (remaining domains ...) |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.65 / 0.09 | 0.65 / 0.14 | 0.44 / 0.09 | 0.52 / 0.13 | 23 more in `runs/ladder-baseline-eval.txt` |
| loss | 0.63 / 0.08 | 0.64 / 0.13 | 0.36 / 0.08 | 0.49 / 0.18 | 23 more in `runs/ladder-loss-eval.txt` |
| ctx (8) | 0.55 / 0.08 | 0.55 / 0.11 | 0.32 / 0.17 | 0.48 / 0.21 | 23 more in `runs/ladder-ctx-eval.txt` |
| lora attn_mlp r=16 | 0.69 / 0.08 | 0.65 / 0.11 | 0.45 / 0.11 | 0.62 / 0.24 | 23 more in `runs/ladder-lora_mlp-eval.txt` |
| layers 24 | 0.62 / 0.09 | 0.59 / 0.11 | 0.43 / 0.07 | 0.49 / 0.18 | 23 more in `runs/ladder-layers24-eval.txt` |
| layers 28 | 0.63 / 0.08 | 0.61 / 0.13 | 0.43 / 0.07 | 0.42 / 0.11 | 23 more in `runs/ladder-layers28-eval.txt` |
| soft labels | 0.64 / 0.09 | 0.65 / 0.10 | 0.42 / 0.08 | 0.51 / 0.10 | 23 more in `runs/ladder-softlabels-eval.txt` |

Read of these numbers: **lora attn_mlp r=16 is the only change that beats the baseline** (+4.0 pts
accuracy, best per-type accuracy in 3 of 3 types, ECE on par at 0.055 vs 0.054). Loss flags and soft
labels are roughly neutral in accuracy (-2.0 / -1.1 pts); soft labels slightly improves ECE (0.051),
the loss flags worsen it (0.068). **ctx (8) and extra depth both hurt at this scale**
(-6.5 and -2.3 pts; layers 24/28 also degrade choice accuracy). One epoch on 2k questions is a smoke-
strength budget, so treat this as a direction check for the small ladder, not a final ranking — the
`full_*.yaml` ladder on the full split is where the ablation conclusions must come from.

## Licenses

- **Dataset** `avbiswas/bev-decision`: `license: unknown` on the Hub — it mixes many upstream sources
  (each with its own terms) and asserts **no blanket license**. Check the linked sources before
  redistribution or commercial use.
- **Backbone** Qwen3 (`Qwen/Qwen3-0.6B`): Apache-2.0.
- **Weights**: initialized from Qwen3 only — **not** from `avbiswas/bev-decider` (whose weights are
  CC-BY-NC-4.0). Code in this repo is Apache-2.0.
