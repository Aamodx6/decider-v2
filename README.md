# decider-v2

An open, reproducible attempt to improve on [`avbiswas/bev-decider-0.4B`](https://huggingface.co/avbiswas/bev-decider-0.4B)
and close the gap to TypeSafe Jev, trained on the full
[`avbiswas/bev-decision`](https://huggingface.co/datasets/avbiswas/bev-decision) dataset and evaluated
against the same published benchmarks.

**Status: the code is built, tested and measured. The 12,000-step training runs have not happened
yet** — they need your AWS account. Nothing in this repository claims decider-v2 beats bev-decider or
approaches Jev, because no number that would justify such a claim exists yet.

---

## Read these first

| file | what it is |
|---|---|
| [`ARCH_REVIEW.md`](ARCH_REVIEW.md) | what bev-decider-0.4B **actually** does, read from its source; every `[ASSUMPTION]` and every "gap → change" row of your `arch.md` marked CONFIRMED / WRONG / CANNOT VERIFY; the risks, with the mask-vs-`sdpa` question settled by a test rather than an argument |
| [`ARCH_V2.md`](ARCH_V2.md) | the corrected design, in three tiers, with real parameter counts and the measurement behind every cost claim |
| [`LOCAL_VERIFICATION.md`](LOCAL_VERIFICATION.md) | every command that was run and its real output, plus exactly what was skipped and why |
| [`decider_v2_sagemaker.ipynb`](decider_v2_sagemaker.ipynb) | the one notebook that does everything, in 15 numbered cells |
| `decider_v2/` | the source (12 flat files, torch / transformers / peft / datasets only) |

### The three findings that change how you read everything else

1. **Label smoothing, as written in `arch.md`, is unusable on this task.** With options packed into
   one padded row, torch's `F.cross_entropy(..., label_smoothing=0.05)` spreads the smoothing over the
   padding columns and returns **2.5e+07** on a 3-option row instead of 0.71. 45.35% of questions are
   2-option, so this is the common case. Fixed in `losses.py` by renormalising over the valid
   options; unit-tested to be identical to torch's version when nothing is padded.
2. **The `score` rank embedding must travel with the level, not the slot.** Attaching a rank to a
   list position breaks option-order invariance by 1.96e-03 — which is invisible without shuffling and
   would have quietly degraded every `score` question once per-epoch shuffling is on. Caught by the
   invariance gate; the regression is now a test (`tests.py` (b2)).
3. **At these sample sizes a 5-point difference is noise.** The same reference checkpoint, same code,
   scored 76.25 (n=400), **68.70** (n=1,000) and **76.30** (n=4,000) on the same test split. An
   earlier draft of this README concluded from the n=1,000 draw that the `all` config was harder than
   `default`; the n=4,000 measurement disproved it. So: one fixed test shard, every arm and every
   seed on identical bytes, and no claim resting on fewer than 4,000 questions. A claimed v2
   improvement below ~3 points is unproven. **The real run sidesteps this entirely** — it evaluates
   the full **80,019**-question published test split (±0.30 points) rather than any shard.

For calibration, on 4,000 questions of the `all` test split the reference scores **76.30**
(choice 73.88 / noul 83.53 / score 64.87, ECE 0.0144), and on a 5,000-question `default`-config
sample **77.24** — the published 74.7 on the same protocol. So the two configs are interchangeable
as a test set, and the published number is a fair thing to aim at.

---

## How to run the notebook

### 0. Prerequisites

* A SageMaker Studio space or notebook instance with a GPU (the notebook runs `tests.py` and a
  throughput benchmark in-cell before it launches anything).
* An S3 bucket in the same region, and a SageMaker training-job quota for the instance type. A new
  account usually has **0**, so check cell 8 first: it prints the `service-quotas list` command and
  the console path (<https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotals>
  → Amazon SageMaker AI → the `ml.<family> for training job usage` row → *Request quota increase*).
* `avbiswas/bev-decision` and `avbiswas/bev-decider-0.4B` are public. Set `HF_TOKEN` in the
  environment if you hit a Hub rate limit — not required.
* Optional, for cell 12 only: clones of [JevBench](https://github.com/fstandhartinger/jevbench) and
  [sysone-bench](https://github.com/instax-dutta/sysone-bench), pointed at by `JEVBENCH_ROOT` and
  `SYSONE_BENCH_ROOT`. Without them cell 12 reports `NOT RUN` for that harness instead of a number.
* Put `decider_v2/` next to the notebook, renamed `decider_v2_src/`:

  ```
  your-notebook-dir/
    decider_v2_sagemaker.ipynb
    decider_v2_src/            <- copy of decider_v2/ (the .py files only)
  ```

### 1. Validate it first — SMOKE mode, costs ~nothing

Edit **only** `BUCKET` in cell 1 (or leave it blank and let the notebook derive one) and leave
`SMOKE = True`. Then run all 15 cells top to bottom. It will:

* load `avbiswas/bev-decision('all')` (both splits, no subsampling outside SMOKE),
* print the full statistics block,
* carve train/val/calib/test with the fixed seed and **assert** that val and calib share no state
  with test,
* run `tests.py` and stop the notebook if anything fails,
* measure throughput on the notebook's own GPU and print the extrapolated hours and cost,
* evaluate the published `bev-decider-0.4B` on your split **and** on a `default`-config sample that
  reproduces the published protocol,
* train both arms for 100 steps each on a 4,000-question subset (one seed per arm by default),
* run the JevBench and sysone-bench adapters if their repos are available,
* calibrate, evaluate, and write `RESULTS.md` plus charts.

That is the whole pipeline. If it completes, the real run will too.

> On a 6 GiB card, drop `BATCH_SIZE` to 8 in cell 1 for the dry run: batch 32 × 1,024-token states
> needs about 8.3 GiB and will page on the 4050. The 24 GiB A10G in `ml.g5.2xlarge` is fine.

### 2. The real run

Set in cell 1:

```python
SMOKE         = False
BUCKET        = "your-bucket"
REGION        = "us-east-1"
INSTANCE_TYPE = "ml.g5.2xlarge"      # or ml.g5.12xlarge / ml.p4d.24xlarge

# the budget, expressed the way the model card states it
TARGET_PRESENTATIONS = 940_000        # the card's "about 940K typed questions"
RECIPE_QPS           = 64             # bev-decider's published micro-batch 32 x accum 2
STEPS_FROM_PRESENTATIONS = TARGET_PRESENTATIONS   # -> MAX_STEPS = 14,688
BATCH_SIZE    = 32                   # x GRAD_ACCUM 2 = 64 questions/step

SPLIT_SEED    = 0                    # carves the splits; FIXED, never varies between arms
SEEDS         = [0]                  # one training seed = 2 jobs; see the decision rule below
MAX_RUNTIME_HOURS = 72               # SageMaker KILLS a job that exceeds this; cell 6b checks it
```

`TARGET_PRESENTATIONS` is the unit the model card uses and the one that stays comparable if you
change the batch size. `MAX_STEPS` is derived as `ceil(940,000 / 64) = 14,688`, i.e.
**940,032 question presentations = 2.42 epochs** over the 388,748-question carved train split. Set
`STEPS_FROM_PRESENTATIONS = None` to pin `MAX_STEPS` by hand instead.

**`MAX_RUNTIME_HOURS` is not cosmetic.** SageMaker hard-kills a training job at `max_runtime` and
discards the work. At this budget the measured throughput puts v2 at roughly 13–40 h depending on
instance, so the 12 h this notebook used to hardcode would have terminated a real run
mid-training. Cell 6b compares the budget against the throughput measured on *your* instance and
**refuses to launch** if either arm would be killed before finishing.

`SPLIT_SEED` fixes the data every arm trains and is scored on; `SEEDS` varies only the training
seed. At the defaults that is **2 jobs** (baseline + v2), which is the cheapest way to find out
whether the 940K budget is affordable and whether it converges. It **cannot** produce a verdict —
see the decision rule below. `RUN_BASELINE` / `RUN_V2` / `RUN_ABLATIONS` control which are
launched. Nothing else needs editing.

A real run evaluates the **full 80,019-question published test split** — cell 3 asserts the shard
size and refuses to continue otherwise. Only `SMOKE=True` subsamples the test split.

### 2b. Preflight gates — both on by default

Every job refuses to start training until two checks pass inside the container:

| gate | what it does | why it is not optional |
|---|---|---|
| `ENV_CHECK` | prints the resolved version of every pin in [`decider_v2/requirements.txt`](decider_v2/requirements.txt), `pip install`s them if the container does not already match, and **aborts** if they still do not | this project's correctness depends on transformers 5.x returning the 4-D attention mask as-is. A silently different transformers is a correctness bug, not a warning, and would only show up as a plausible-looking number |
| `SELFTEST` | runs the packed-vs-separate equivalence and option-order-invariance gates on a 2-layer random-init model (~10 s, no download, no data) and **aborts** the job on any failure | catches a broken mask/position-id guarantee inside the container, instead of after 12k steps and a few hundred dollars |

Both are printed at the top of the CloudWatch log, so a failed job says *why* in its first minute.
`--selftest 0 --env_check 0` disables them if you are deliberately re-running something that would
otherwise trip them.

**Container.** `FRAMEWORK_VERSION = "2.7.1"` / `PY_VERSION = "py312"`. Verified against the SDK's own
`image_uri_config/pytorch.json`: 2.5.1 → py311, 2.6.0 → py312, 2.7.1 → py312. Python ≥ 3.11 is
required because numpy 2.3.5 declares `requires-python >=3.11`. Cell 1 validates the pairing before
launching anything. *(An earlier revision asked for 2.6.0 + py311, which is not a valid pairing and
would have failed at launch.)*

### 2c. Running the seeds: seed 0 of both arms first, then seed 1

SageMaker bills by instance-hour, not by wall-clock, so two arms cost the same whether they run at
the same time or one after the other — running them together simply halves how long you wait.
`est.fit(wait=False)` is already the default, so **launching both arms of a seed in cells 8 and 9
back-to-back already runs them concurrently.**

1. Set `SEEDS = [0]` (the default) in cell 1, run cells 8 and 9. `baseline_s0` and `v2_s0` start
   together and finish together. `launch()` is idempotent, so re-running these cells never
   double-launches a job whose state is `Completed`.
2. Set `WAIT = True` in cell 10 and let it poll until both are `Completed`.
3. Run cells 11 and 13. The `## Decision` section will report a delta with a bootstrap CI, but with
   **one seed the verdict is necessarily `NOT SEPARABLE`** — the seed spread is 0 by construction, so
   condition (b) cannot discriminate. This run tells you whether 940K is affordable and whether it
   converges. It does not tell you whether v2 wins.
4. Set `SEEDS = [0, 1]` and re-run cells 8 and 9. Seed 0 is skipped (already `Completed`); only
   `baseline_s1` and `v2_s1` are submitted, concurrently. Re-run cells 10–13. Now there is a real
   spread to test the delta against. The verdict will be **PROVISIONAL** at best — see the decision
   rule below for why two seeds cannot produce `SEPARABLE`.
5. Add a third seed (`SEEDS = [0, 1, 2]`) and re-run cells 8 and 9 again. Only `*_s2` is submitted. This
   is the first point at which a `SEPARABLE` verdict becomes possible. Keep adding seeds while the
   verdict says `NOT SEPARABLE`; more seeds can only make the bar harder to clear, which is the point.

To run arms **sequentially** instead (e.g. to stay inside a strict instance-concurrency quota),
pass `wait=True` to `launch()` in cell 8/9. Same cost, twice the wall-clock. Every arm and seed
trains on the same `SPLIT_SEED` data either way, so staging never changes what is being compared.

### 3. Bigger instance

`ml.g5.2xlarge` is 1× A10G (24 GB) and runs single-GPU. To go wider, set `instance_count=2` or `8`
in cell 8 and add `distribution="pytorch", process_count="per_instance"` to the estimator; `train.py`
already initialises `torch.distributed` from `RANK`/`WORLD_SIZE`, shards with
`DistributedSampler`, all-reduces validation counts and writes logs and checkpoints from rank 0
only. **That path has never been executed** (one GPU here) — read `LOCAL_VERIFICATION.md` §0.

---

## Cost and time

Each job is **14,688 steps × 64 questions = 940,032 question presentations** (2.42 epochs of the
388,748-question carved train split). With the default `SEEDS = [0]` that is **2 jobs**.

Measured on an RTX 4050 Laptop (6.0 GiB) with the real model, batch 8 × 2:

| arm | throughput | peak VRAM | 14,688 steps |
|---|---:|---:|---:|
| baseline replica | 8.6 q/s | 2.87 GiB | 30.4 h |
| decider-v2 | 6.5 q/s | 3.11 GiB | 40.2 h |

That card cannot do the real run, but those figures are the worst case and the reason
`MAX_RUNTIME_HOURS` defaults to 72: **on the 4050 alone v2 would need ~40 h and would be killed by
the old 12 h default.** **Cell 6 measures throughput on your instance and cell 6b converts it into
hours and dollars at runtime**, with the price queried from the AWS Price List API (and, if that
fails, printed as `NOT RESOLVED` rather than guessed). Use those numbers, not these.

Expect roughly 2–4× faster per question on an A10G than on the 4050, so a ballpark of **8–20 h per
job** on `ml.g5.2xlarge` is a reasonable thing to sanity-check the in-cell measurement against — so
**~1 day for the default 2 jobs**, and ~$50–130 — but it is a ballpark, not a quote, and it is not a
number this repository has verified. Adding seeds multiplies it linearly: `SEEDS = [0, 1, 2]` is
6 jobs, roughly $150–390.

---

## Reading the results

`RESULTS.md` and `metrics_<tag>.json` are written per (arm, seed). The rows that matter:

| row | what it means |
|---|---|
| `reference_default` | the published checkpoint on a 5,000-question `default`-config sample. **This is the row that validates the encoder** — it should land near the published 74.7 / 69.8 / 84.6 / 63.7 (it landed at 77.24 / 74.65 / 87.01 / 64.99). If it does not, stop: the pipeline is wrong and nothing else is meaningful. |
| `reference` | the same checkpoint on **your** test file — all **80,019** questions in a real run. Measured on a 4,000-question sample: **76.30**. This is the number v2 has to beat. |
| `baseline_s<seed>` | bev-decider's architecture and optimisation, trained on your split with your budget. |
| `v2_s<seed>` | the same budget, the same data, the same code, with Tier 1 changed. |

**Read the `v2 − baseline` delta, not the `v2 − published` difference.** The published number is
comparable as a *test set* (`all` and `default` are within 1 point) but not as a *control*: it was
produced with a training recipe that is not reproducible from public data (`ARCH_REVIEW.md` §1.9).
It is context, not a baseline.

### The decision rule

`RESULTS.md` ends with a `## Decision` section that tests the delta against the two things that
can hide it, and applies one rule automatically:

```
(a) the paired-bootstrap 95% CI for the delta excludes 0
(b) |delta| > max(baseline seed spread, v2 seed spread)
(c) |delta| >= 1.0 point
(d) at least 3 seeds per arm

(a) and (b) and (c) and (d)  ->  SEPARABLE
(a) and (b) and (c), 2 seeds  ->  PROVISIONAL     (one more seed decides it)
anything else                 ->  NOT SEPARABLE
```

**The bootstrap** resamples the *questions* of the full test split 10,000 times, pairing each
baseline question with the same-index v2 question, and averages the per-question difference in
correctness. Pairing is what makes the interval tight, and `decide.py` *verifies* the two arms were
scored over an identical question sequence rather than assuming it. The CI covers **test sampling
only**; condition (b) is what covers training-run variance, which is a different thing entirely.

**The bar is not a binomial interval.** A binomial CI assumes test sampling is the only source of
randomness. The seed spread is a direct measurement of this experiment's own run-to-run variance on
this exact data, and (b) requires the delta to beat it. Condition (c) exists because at n = 80,019 a
small delta can be statistically unambiguous and still not worth acting on.

**Why three seeds for `SEPARABLE`.** With two seeds the spread is a range between a single pair of
runs, and it was measured at **0.05 pp**, **0.35 pp** and **2.52 pp** for the same configuration in
three separate dry runs. A tight spread on two seeds is not evidence of a tight spread, which is why
that case is labelled `PROVISIONAL` rather than `SEPARABLE`. With *one* seed — the default here — the
spread is 0 **by construction**, so (b) is vacuous and the verdict is forced to `NOT SEPARABLE`. The
latest dry run demonstrates it: a **+4.25 pp** delta whose 95% CI `[+3.02, +5.47]` excludes 0 still
returns `NOT SEPARABLE`, and says why. That is the correct answer, not a bug — add seeds before
believing any delta.

This is not decoration. Measured on the 4,000-question SMOKE shard at 100 steps:

| run | delta | seed spread | verdict |
|---|---:|---:|---|
| run 1 (2 seeds) | +2.64 pp | 2.52 pp | cleared the noise floor by **0.12 pp** |
| run 2 (2 seeds) | +4.03 pp | 0.05 pp | PROVISIONAL (only 2 seeds) |
| run 3 (2 seeds) | +2.75 pp | 0.35 pp | PROVISIONAL (only 2 seeds) |
| run 4 (1 seed) | +4.25 pp | 0.00 pp | **NOT SEPARABLE** — spread is 0 by construction |

Every one of these had bootstrap CIs excluding 0. None is a result, and the artefact says so rather
than leaving the reader to work out which number is load-bearing. The implementation lives in
[`decider_v2/decide.py`](decider_v2/decide.py) and every branch is unit-tested in `tests.py` §(e),
including all of the configurations above.

Two gates are pass/fail and are reported separately from accuracy:

* **order-invariance** — `max|Δp| ≤ 1e-5` in fp32. The reference itself measures 3.16e-06 to 3.70e-06
  at 20 layers, so a result near 1e-5 is a finding and a result near 1e-8 is a bug in the check.
* **ECE after calibration** — do not get worse than the reference (0.0144 measured on `all`, n=4,000;
  0.0122 on `default`, n=5,000).
* **sample size** — a real run scores the full 80,019-question split, where the 95% interval is
  ±0.30 points at p≈0.76, so a delta below ~1 point is still not evidence. Under `SMOKE`
  (n=4,000) the interval is ±1.35 and anything under ~3 points is unproven.

Per-example predictions are written to CSV for every arm, so any row can be recomputed from the
predictions without re-running the model.

---

## Layout

```
decider_v2/
  serialize.py    special tokens, question -> tokens, shared position ids, vectorised 4-D mask
  data.py         dataset, split carving, torch Dataset, length-bucketed sampler, collate
  model.py        truncated Qwen3 + LoRA, DecisionHeadV1 (bev replica) and DecisionHeadV2, checkpoint io
  losses.py       smoothed CE over valid options, EMD, per-type weighting, ECE
  train.py        one script, SageMaker-compatible, single GPU or torchrun; preflight gates
  environment.py  the requirements.txt pins, enforced by train.py / evaluate.py / the notebook
  requirements.txt  exact versions every published number was produced with
  calibrate.py    per-type temperature fitting
  evaluate.py     accuracy / ECE / NLL / order-invariance / per-domain, metrics.json + predictions
  decide.py       paired bootstrap + the SEPARABLE / PROVISIONAL / NOT SEPARABLE rule
  tests.py        the four gates + the decision-rule tests (runs offline, no download)
  normalize.py    DERIVED pre-pass (Tier-2 flag)
  prepare.py      dataset -> Parquet shards + manifest with the parity assertions
  bench.py        throughput / peak-VRAM probe
  paramcount.py   parameter counts for every component
  bench/
    common.py            loads the model once; the (state, question) -> probabilities call
    run_jevbench.py      JevBench adapter, graded by jevbench's own scoring/summarize
    run_sysone_bench.py  sysone-bench adapter, graded by sysone-bench's own contracts/metrics
build_notebook.py generates the notebook from these files, so the sources can never drift
dryrun_notebook.py executes the notebook offline with AWS stubbed (see LOCAL_VERIFICATION.md §7)
ref/              pristine clones: bev-train, bev-decider, sysone-bench, jevbench
recon/            Phase 0 recon scripts and their raw output
```

Regenerate the notebook after editing any source file:

```bash
python build_notebook.py
```

---

## Licensing — read before you publish anything

* `avbiswas/bev-decision` asserts **no blanket licence**. The Hub metadata is `license: unknown` and
  the dataset card says so in those words, naming ProofWriter's redistribution terms, NVD descriptions
  authored by third-party CVE reporters, and GoEmotions' inclusion of Reddit text.
* `avbiswas/bev-decider-0.4B` is **CC-BY-NC-4.0**. Weights trained on this mixture should be assumed
  non-commercial until every upstream is audited individually.
* decider-v2 initialises from `Qwen/Qwen3-0.6B` (Apache-2.0) and never from bev-decider weights, so
  the NC terms are not inherited by construction — but they are inherited by the *data*.

Full detail and the upstream terms the card does list: `ARCH_REVIEW.md` §2, Q2.