# decider-train

Training, evaluation and calibration code for **decider-v2**, an open-source 0.6B decision model
for Bird's-Eye-View (BEV) autonomous driving and robotics judgment. One Qwen3-0.6B backbone with
LoRA, a small head over per-option states, and a training recipe written down as YAML. The repo is
flat: every module is importable from the repository root (`from network import ...`).

**Status: the code runs. The training runs have not happened.** They need an AWS account, and
nothing here claims decider-v2 beats bev-decider or approaches Jev — no number that would justify
that claim exists yet. Every result cell below is `NOT RUN`.

---

## Architecture

![packing](assets/arch_pack.png)

One question becomes **one** sequence: the state and instruction, then each option, then an answer
slot. Every option's position ids restart at the prefix length, so an option occupies the same
positions it would have if it were alone.

![mask](assets/arch_mask.png)

A `[B, 1, L, L]` additive mask, built from slot ids, keeps each option from attending to any other.
Inside the option block it is the identity: options cannot see each other, and no option can tell
its list position.

![brain and judge](assets/arch_brain.png)

The LoRA'd backbone produces one hidden state per option (`[B, K, H]`); the head scores them.
`head=v1` is the bev-decider replica (per-option MLP, no cross-option attention, what the baseline
arm trains); `head=v2` adds a context stack, a rank feature for `score` questions and an optional
null slot.

The diagrams are placeholders until the PNGs are added to `assets/`. `arch.ipynb` walks through the
same three scenes with real tensors on a 2-layer random-init model — on a CPU, with no downloads.

---

## Layout

```
arch.ipynb            the three scenes above, executed on real tensors (CPU, ~7s)
cloud_startup_script.sh   plain GPU box: uv install -> uv sync -> preflight -> train -> S3 sync
dataset.py            loading, split carving, torch Dataset, length-bucketed sampler, collate
                      + `python dataset.py prepare` (writes the parquet channels)
tokenization.py       special tokens, question -> packed ids, shared position ids, the 4-D mask
network.py            truncated Qwen3 + LoRA, DecisionHeadV1 (replica), DecisionHeadV2, the losses
train.py              one script: single GPU or torchrun, SageMaker channels, config files
logger.py             metrics.jsonl, stdout line format, checkpoint save/load, env pin check
inference.py          load a checkpoint, predict, apply per-type temperatures
evaluate.py           accuracy / ECE / NLL / order-invariance / per-domain, predictions CSV
decide.py             paired bootstrap + the SEPARABLE / PROVISIONAL / NOT SEPARABLE rule
run_ladder.py         multi-seed orchestrator ([0, 1, 2] across arms), automated decide.py, updates RESULTS.md
export_hf.py          Hugging Face packaging (safetensors, model card, config.json, tokenizer)
normalize.py          Tier-2 DERIVED pre-pass (no counterpart in bev-train, kept for ablation)
download_eval.py      entry point for the SageMaker evaluation job (pulls a checkpoint, calls evaluate)
configs/              smoke, both arms, the long option, five Tier-2 ablations
benchmarks/           JevBench and sysone-bench adapters, graded by their own scorers
scripts/              bench.py (throughput/VRAM), paramcount.py (parameter counts)
tests/                test_core.py (the gates) and golden_check.py (old-code parity)
docs/                 ARCH_REVIEW.md, ARCH_V2.md, LOCAL_VERIFICATION.md
```

Imports are flat. There is no `src/` package and no `setup.py`.

---

## Quickstart

```bash
uv sync --frozen --extra dev        # Python 3.12, torch 2.7.1 (the container's version)
python dataset.py prepare --out ./data --seed 0
python tests/test_core.py           # the four structural gates, offline, no download
python train.py --config configs/smoke.yaml
```

`train.py --config` takes precedence `argparse default < yaml < command line`, so a SageMaker
hyperparameter (`--max_steps 2000`) overrides the YAML without editing it. An unknown YAML key
raises rather than being ignored.

The config ladder:

| config | what it is |
|---|---|
| `smoke.yaml` | 100 steps on a small subset — plumbing only, no conclusions |
| `baseline_replica.yaml` | bev-decider's architecture and optimisation, rebuilt as train.py would train it |
| `decider_v2.yaml` | the Tier-1 recipe, same budget as the baseline |
| `decider_v2_long.yaml` | 25–30k steps, batch 128 option |
| `ablations/*.yaml` | 24 layers, mean pooling, DERIVED pre-pass, v2-LoRA-shape, stage 2 |

### Two behaviour fixes relative to the old code

* **Budget.** With `--max_steps` set, the loop keeps iterating epochs until that many steps are
  done; `--epochs` is only a cap when `--max_steps` is unset. The old loop stopped at
  `epochs × steps_per_epoch` (12,150) while the cosine schedule expected 14,688. A run that ends
  off-budget exits non-zero, and `summary.json` records `final_step` and `total_steps`.
* **Checkpoints.** Written to `--model_dir` and mirrored to `--checkpoint_dir`
  (`/opt/ml/checkpoints` when it exists, else `./outputs/checkpoints`) as `latest` / `best` /
  `final`. `--auto_resume 1` resumes step, optimiser and scheduler from `<checkpoint_dir>/latest`.
  `latest` stores the backbone in fp32 because the model trains in fp32 (autocast is CUDA-only) and
  a bf16 copy would start every resumed run a few 1e-4 away from the uninterrupted one; `best` and
  `final` stay bf16 as the evaluation/deployment artifacts, so `latest` is the one checkpoint that
  costs 2x (and the only one `--auto_resume` reads). If a run ends before its first
  `--eval_every`, `best` falls back to the final weights — it is the only state that was evaluated,
  and `evaluate.py` selects `best`.
* **Interruption drill.** `--stop_at_step N` leaves the loop cleanly at step `N` after writing a
  resumable `latest`, and keeps `--max_steps` (and therefore the cosine schedule) unchanged. A
  restarted job then replays the original run exactly; shrinking `--max_steps` instead does not,
  because that also shortens the schedule the first leg was trained on.

---

## SageMaker

The notebook is `sagemaker_train.ipynb` and it is generated from `_nbcells/` by `build_notebook.py`
(repo root) — edit the cell file, not the notebook.

1. **Quota.** A new account usually has **0** training-job quota for GPU instances. The quota cell
   prints the `service-quotas` command and the console path; request the instance type you intend to
   use before launching anything.
2. **Bucket.** One S3 bucket in the same region, for channels and checkpoints. Set `BUCKET` in cell 1.
3. **Notebook.** Upload `sagemaker_train.ipynb` to Studio. It builds its own source pack: cell 1
   copies an explicit allowlist into `build/sagemaker_src/` and that directory is the estimator's
   `source_dir` — never the repository root, which would upload `.git`, notebooks and data.
4. **SMOKE first.** `SMOKE = True` uses `configs/smoke.yaml`, a separate state file and separate run
   names, so a smoke job can never be mistaken for a real one. Run top to bottom; it costs cents.
5. **Cost.** Never hard-coded. The throughput cell measures on the instance you are actually on, the
   budget cell converts that into hours, and the price comes from the AWS Price List API at runtime
   (printed as `NOT RESOLVED` if the lookup fails). The budget cell **refuses to launch** when
   `MAX_RUNTIME` would kill a run that is still training.

Container: `pytorch 2.7.1` / `py312`. `requirements.txt` pins every Python dependency and
**excludes torch** — the image provides a CUDA-matched build and installing over it is how a GPU job
ends up on CPU-only torch.

No AWS credentials were used to build any of this; no job was launched.

---

## Evaluation protocol

* **Full test split, not a sample.** A real run scores all **80,019** published test questions
  (±0.30 points at p≈0.76). SMOKE subsamples and says so.
* **Same bytes for every arm and seed.** Splits are carved once from a fixed `SPLIT_SEED`; only the
  training seed varies. `decide.py` verifies the two arms were scored on an identical question
  sequence instead of assuming it.
* **Paired bootstrap** over questions, 10,000 resamples, 95% CI.
* **Verdict rule:** `|delta| > max(seed spread)` **and** CI excludes 0 **and** `|delta| ≥ 1.0 pp`
  **and** ≥3 seeds → `SEPARABLE`; the same without the third seed → `PROVISIONAL`; anything else →
  `NOT SEPARABLE`. With one seed the spread is 0 by construction, so the verdict is forced to
  `NOT SEPARABLE`. That is the rule working, not failing.
* **Order invariance** is a gate, not a metric: `max |Δp| ≤ 1e-5` in fp32.

## Results

Every cell is `NOT RUN` until real runs exist. Do not fill these in from a SMOKE shard.

| arm | seeds | budget (presentations) | test acc | ECE | 95% CI vs baseline | verdict |
|---|---|---:|---:|---:|---|---|
| `reference` (published checkpoint) | – | – | NOT RUN | NOT RUN | – | – |
| `baseline_replica` | – | 940,032 | NOT RUN | NOT RUN | – | – |
| `decider_v2` | – | 940,032 | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| `decider_v2_long` | – | NOT RUN | NOT RUN | NOT RUN | – | – |

---

## Credits

Architecture, masks and training implementations are based on Bird's-Eye-View decision-making research.
JevBench and sysone-bench belong to their respective authors.

## License

This project is licensed under the Apache 2.0 License. Base model weights derive from [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0).
