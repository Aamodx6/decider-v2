# Training on AWS SageMaker

How to run this codebase (`bev-train-main`) on AWS SageMaker, on the full `bev-decision` dataset
(`data.configs: [all]`, ~420K training questions).

This document is written against the code as of commit `ca27493`. Every claim about code behaviour
below was read out of the source or executed locally. Things that are *estimates* are labelled as
such — read [Sizing the run](#sizing-the-run) before you spend money.

---

## Table of contents

- [What you're actually training](#what-youre-actually-training)
- [Read this first: 4 things that will break on SageMaker](#read-this-first-4-things-that-will-break-on-sagemaker)
- [Prerequisites](#prerequisites)
- [Step 0 — Fix the two code bugs](#step-0--fix-the-two-code-bugs)
- [Step 1 — AWS setup](#step-1--aws-setup)
- [Step 2 — Pre-stage the dataset and tokenizer to S3](#step-2--pre-stage-the-dataset-and-tokenizer-to-s3)
- [Step 3 — Build the ECR image](#step-3--build-the-ecr-image)
- [Step 4 — Write the SageMaker entry script](#step-4--write-the-sagemaker-entry-script)
- [Step 5 — Write the training configs](#step-5--write-the-training-configs)
- [Step 6 — Run the pilot, then size the real run](#step-6--run-the-pilot-then-size-the-real-run)
- [Step 7 — Launch the full run](#step-7--launch-the-full-run)
- [Step 8 — Spot interruption and resume](#step-8--spot-interruption-and-resume)
- [Step 9 — Calibration and evaluation as separate jobs](#step-9--calibration-and-evaluation-as-separate-jobs)
- [Step 10 — The ablation ladder](#step-10--the-ablation-ladder)
- [Sizing the run](#sizing-the-run)
- [Cost](#cost)
- [Monitoring a running job](#monitoring-a-running-job)
- [Troubleshooting](#troubleshooting)
- [Appendix: SageMaker filesystem layout](#appendix-sagemaker-filesystem-layout)

---

## What you're actually training

| | |
|---|---|
| Backbone | `Qwen/Qwen3-0.6B`, truncated to the first 20 of its 28 decoder layers (`network.py:140`) |
| Adaptations | LoRA on the last 12 kept layers (8–19), `r=8`, `alpha=16`, target `attn` |
| Head | `ChoiceHead` (`network.py:61`) — 2 self-attention blocks, `new_dim=512`, bilinear answer×choice scoring |
| Trainable params | LoRA adapters + head only. Base Qwen weights stay frozen |
| Dataset | `avbiswas/bev-decision`, config `all` → 231,332 train rows → **422,551 train questions** |
| Dev split | Carved out of train by a deterministic md5 rule on normalized state (`dataset.py:94`), state-grouped so no state straddles the split. ~1,024 val states ≈ 0.49% ≈ 2K val questions |
| Test split | **Never read during training.** Only `inference.py` touches it |
| Checkpoint contents | LoRA adapter + head only (`logger.py:52`). ~40–60 MB per checkpoint, not the full Qwen weights |
| Optimizer steps (1 epoch) | ~420,000 / 64 questions per step ≈ **6,570 steps** |

### Why the architecture matters for your infrastructure choices

The attention mask is a **full 4-D `B x L x L` float tensor** (`dataset.py:189`), not a sparse or
2-D mask. Consequences:

1. Memory scales **quadratically** in `max_state_tokens`. At `max_state_tokens: 1024` and
   `batch_size: 32` the mask alone is `32 x 1024 x 1024 x 4 bytes = 134 MB` per batch.
2. Compute scales quadratically too, so doubling `max_state_tokens` more than doubles runtime.
   This is the single biggest lever on your bill.

The backbone is loaded in **fp32** (`network.py:143`, `dtype=torch.float32`) and bf16 is applied
via autocast only (`inference.py:22`). That is correct for loss stability but costs memory versus
a bf16 load — budget ~1.8 GB just for the 20-layer fp32 weights.

### Measured throughput (your own numbers)

From `runs/ladder-loss-20261006-215757/metrics.jsonl` — 2,000 questions at
`max_state_tokens: 256`, `batch_size: 8`, on an **RTX 4050 (6 GB laptop)**:

```
2,000 questions in 238 s  =>  8.4 questions/sec  @ 256 state tokens, RTX 4050
```

This is your ground truth for capacity planning. See
[Sizing the run](#sizing-the-run) for turning it into a wall-clock estimate.

---

## Read this first: 4 things that will break on SageMaker

### 1. `uv.lock` pins torch 2.14.0 with CUDA 13 wheels — no AWS image has that driver

`uv.lock` resolves:

```
name = "torch"
version = "2.14.0"
  → nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cuda-toolkit (cu13)
```

`cloud_startup_script.sh:38` says it plainly: *"The locked torch wheel is built for CUDA 13, which
needs NVIDIA driver >= 580."*

No AWS Deep Learning AMI or PyTorch DLC ships a 580+ driver. If you `uv sync` inside a SageMaker
container you get one of:

- a hard install failure (driver too old for the cu13 runtime), or
- **`torch.cuda.is_available() == False` and silent fallback to CPU** — which would turn a 20-hour
  job into a multi-week job that you pay for the whole time. `inference.py:14` `get_device()`
  returns `torch.device("cpu")` with no error.

**Do not run `uv sync` on SageMaker.** Use the PyTorch DLC (which ships a driver matched to its own
torch) and `pip install` only the non-torch dependencies. See [Step 3](#step-3--build-the-ecr-image).

### 2. `instance_count > 1` will silently train N identical models

There is no DDP, no `torch.distributed`, no `sagemaker.distributed` anywhere in the codebase. The
only multi-GPU-related code is `torch.cuda.get_rng_state_all()` (`train.py:145`), which is just
RNG state capture.

SageMaker with `instance_count=2` runs the same `train.py` twice on the same full dataset, each
writing to `/opt/ml/output/data`. You pay 2× and learn nothing.

**Always `instance_count=1`.** To use more GPUs, run more *jobs*.

### 3. Nothing syncs your checkpoints to S3, so Spot interruption loses the whole run

The crash-safe rolling save in `logger.py:75` writes to
`runs/<id>/checkpoints/resume/slot_{a,b}` — **local container disk**.

SageMaker's `OutputDataConfig` sync of `/opt/ml/output/data` happens **only on successful
completion**. On a Spot interruption the container is destroyed and every checkpoint is gone.

This matters because your full run is roughly 20 hours — long enough that Spot interruption is
near-certain without handling. [Step 4](#step-4--write-the-sagemaker-entry-script) patches
`RunLogger.save_resume_state` to `aws s3 sync` after each save.

SageMaker's own `CheckpointConfig` does **not** help here: it only understands
`sagemaker.checkpoint`-managed files, not this repo's custom slot format.

### 4. `num_workers: 6` is wrong for any SageMaker instance, and the tokenizer is your bottleneck

`BEVDataset.__getitem__` (`dataset.py:148`) calls `encode_example` → `tokenize_prompt`, which runs
the HuggingFace tokenizer **per item, per epoch, in the DataLoader workers**. At
`max_state_tokens: 1024` this is substantial CPU work, and it runs while the GPU waits.

Your configs hardcode `num_workers: 6` / `8`. SageMaker GPU instances give you 4 (`g5.2xlarge`) to
192 (`g5.48xlarge`) vCPUs. On `g5.4xlarge` you should use ~12; on `g5.12xlarge`, ~24.

Watch for the opposite failure: `num_workers` above your vCPU count will slow you *down*.

---

## Step 0 — Fix the two code bugs

Both are small, both will cost you time on SageMaker, and the second one is a correctness problem.

### Bug A — `calibrate.py` crashes; it asks for a split that does not exist

`calibrate.py:111` calls `make_loader("dev", ...)`. But neither `make_loader` (`train.py:21`) nor
`load_questions` (`dataset.py:100`) know the name `dev`:

- `train.py:32` only special-cases `"train"` and `"val"`; anything else falls to the `else` branch
  and reads `data["max_test_questions"]`.
- `dataset.py:115` maps `train`/`val` → source split `train`; `dev` is passed through unchanged.

Verified locally against the cached dataset:

```
>>> load_questions('dev', 0, 0, ['default'])
ValueError: Unknown split "dev". Should be one of ['train', 'test'].
```

So `calibrate.py` cannot run at all as committed. (The `calibration.json` files under `runs/` were
produced before the dev→val rename landed, which is why this was never caught — the committed code
path has never been exercised.)

**Fix** — treat `dev` as an alias for `val` in **four** places. This is applied in the repo now.

`train.py:32`
```python
    if split == "train":
        max_questions = data["max_train_questions"]
    elif split in ("val", "dev"):          # "dev" = legacy alias used by calibrate.py
        max_questions = data.get("max_val_questions",
                                 data.get("max_dev_questions", data.get("max_test_questions")))
```

`dataset.py:115`
```python
    source_split = "train" if split in ("train", "val", "dev") else split
```

`dataset.py:121-122`
```python
    if split in ("train", "val", "dev"):
        keep_val = split in ("val", "dev")
```

> **The fourth edit is the one that matters, and it is easy to miss.** The original line reads
> `keep_val = split == "val"`. If you add `"dev"` to the outer tuple but leave that line alone, the
> crash becomes *silent*: `keep_val` is `False` for `dev`, so the filter keeps
> `is_val_state(state) == False` — i.e. the **train** partition. `calibrate.py` would then fit
> temperatures on training data and write a plausible-looking `calibration.json`, and the ECE
> numbers would be optimistic for reasons you could not see from the output. `dev` and `val` must
> select the *same* rows, so both halves of the condition need `"dev"`.

With that, `calibrate.py:108`'s `config["data"]["max_dev_questions"] = args.max_questions` is
correctly picked up by the `max_dev_questions` fallback in `make_loader`, and calibration fits
temperatures on **dev**, which is what you want.

Verify the partition is actually the val one after changing this — do not just check that it runs:

```python
from dataset import load_questions, is_val_state
dev, val = load_questions("dev", 50, 0, ["default"]), load_questions("val", 50, 0, ["default"])
assert len(dev) == len(val)
assert all(is_val_state(r) for r in dev["state"]), "dev is NOT the val partition"
assert not any(is_val_state(r) for r in load_questions("train", 50, 0, ["default"])["state"])
```

Run `uv run pytest -m "not smoke"` after the change.

### Bug B — the val split is loaded and exploded a second time, doubling startup time and RAM

`train.py:262` calls `make_loader("val", ...)` with no `dataset=` argument, so it re-runs
`load_questions("val", ...)` — which reloads the 231K-row `all` config from disk, re-applies the
md5 filter, and re-explodes every row into questions.

That means every job start pays for **two** full load+filter+explode passes over 231,332 rows, and
holds two exploded copies in RAM simultaneously.

For a ~20 minute job that is tolerable. For an ablation ladder of 7 jobs it is 2+ hours of pure
setup. Worth caching the exploded dataset to a local Arrow file in the pilot (see
[Step 6](#step-6--run-the-pilot-then-size-the-real-run)) if you run the ladder.

**Lower priority but worth knowing:** `RunLogger.save_resume_state` (`logger.py:75`) does
`shutil.rmtree(target)` then `os.replace(tmp, target)` on directories. That is fine on Linux
because the target is removed first, but it is the kind of thing that breaks on a different
filesystem. `/opt/ml` is a normal EBS mount, so it will be fine.

---

## Prerequisites

You need:

- An AWS account with an active region and a working default VPC (or explicit subnet + security
  group you control).
- The AWS CLI v2 configured (`aws configure`), and `pip install sagemaker boto3`.
- Docker, only locally, for [Step 3](#step-3--build-the-ecr-image).
- Quota. **This is the step people forget.** New accounts have a default of 0–4 vCPU for `ml.g5.*`:

  ```bash
  aws service-quotas get-service-quota --service-code sagemaker \
    --quota-code L-1216C47 --region $AWS_REGION   # vCPU for training
  ```

  Request an increase to at least 16 vCPU before you try to launch a `g5.4xlarge`.

- A Hugging Face account is **not** required — `avbiswas/bev-decision` and `Qwen/Qwen3-0.6B` are
  both public. But [Step 2](#step-2--pre-stage-the-dataset-and-tokenizer-to-s3) strongly suggests
  setting `HF_TOKEN` anyway, because you are unauthenticated now and rate limits bite.

---

## Step 1 — AWS setup

```bash
export AWS_REGION=us-east-1
export BUCKET=jev-train-$AWS_REGION-$(date +%s)
aws s3 mb s3://$BUCKET
aws s3api put-bucket-versioning --bucket $BUCKET --versioning-configuration Status=Enabled
```

Create the IAM role. **Do not use the `AmazonSageMakerFullAccess` managed policy in production**,
but for a first run it is fine and much faster:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Principal": { "Service": "sagemaker.amazonaws.com" },
      "Action": "sts:AssumeRole" },
    { "Effect": "Allow", "Action": [
        "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"
      ], "Resource": [ "arn:aws:s3:::YOUR_BUCKET", "arn:aws:s3:::YOUR_BUCKET/*" ] },
    { "Effect": "Allow", "Action": [ "CloudWatchLogs:CreateLogGroup",
        "CloudWatchLogs:CreateLogStream", "CloudWatchLogs:PutLogEvents",
        "CloudWatchLogs:DescribeLogStreams" ],
      "Resource": "arn:aws:logs:*:*:*" },
    { "Effect": "Allow", "Action": "ecr:GetAuthorizationToken",
      "Resource": "*" }
  ]
}
```

```bash
aws iam create-role --role-name JevSagemakerRole \
  --assume-role-policy-document file://trust-policy.json
aws iam put-role-policy --role-name JevSagemakerRole --policy-name S3 \
  --policy-document file://inline-policy.json
```

Get the ARN:

```bash
aws iam get-role --role-name JevSagemakerRole --query 'Role.Arn' --output text
```

---

## Step 2 — Pre-stage the dataset and tokenizer to S3

Every training job otherwise downloads Qwen3-0.6B (~1.2 GB) and the whole `all` config (276K rows of
long text) from HuggingFace. Across a 7-run ablation ladder that is ~9 GB of egress from an AWS
region, repeated, for content that never changes.

Run this **once**, on any machine with internet. A throwaway `ml.m5.xlarge` job costs about $0.50
and keeps your laptop out of it.

`scripts/prepare_s3_cache.py` does the download and the upload (boto3, so no AWS CLI needed):

```bash
export HF_HOME=/tmp/hf
pip install boto3 huggingface_hub

python scripts/prepare_s3_cache.py upload --bucket $BUCKET --region $AWS_REGION
python scripts/prepare_s3_cache.py verify --bucket $BUCKET
```

It downloads both repos with `snapshot_download` into `$HF_HOME/hub`, uploads that tree to
`s3://$BUCKET/hf-cache/`, and writes a manifest so `verify` can detect a truncated upload.

**Target `$HF_HOME/hub`, not `$HF_HOME`.** `snapshot_download(cache_dir=...)` creates
`<cache_dir>/models--<repo>`, and HF only resolves a by-name lookup under `$HF_HOME/hub`. Both
`--cache-dir` and `--dest` default to that path for exactly this reason.

> **The dataset card is not optional.** The configs `default`, `hard_50k`, `numeric_temporal`,
> `skills`, `counterfactual_15k` and `all` are defined in the YAML frontmatter of the repo's
> `README.md`, not by directory names — `all` is a virtual config listing all five parquet paths.
> Without the card, `load_dataset("avbiswas/bev-decision", "all")` cannot resolve. Both `upload`
> and `restore` fail loudly if the card is absent rather than letting it break at training time.

Restore it in the training container before `train.py` runs:

```bash
python scripts/prepare_s3_cache.py restore --bucket $BUCKET --dest /cache/hf/hub --region $AWS_REGION
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

**How this works with no code changes:** `datasets` and `transformers` both resolve a repo by
name through `$HF_HOME/hub`. With the tree restored there, `dataset.py:117`
(`load_dataset(DATASET_NAME, configs[0], split=source_split)`) and `tokenization.py:9`
(`AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")`) find the cached copies and never touch the
network. `HF_HUB_OFFLINE=1` turns "the network is unreachable" from a hang into a fast, explicit
error.

Use `/cache/hf/hub`, on the EBS volume — **not** `/opt/ml/output/data`, which is synced to S3 at
job end and you do not want ~2 GB of model cache in your output bucket.

---

## Step 3 — Build the ECR image

Base it on the PyTorch DLC. This is the point of the whole exercise: the DLC ships a driver matched
to its own torch, so you sidestep bug #1 entirely.

```bash
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com

aws ecr create-repository --repository-name jev-train
```

`Dockerfile`:

```dockerfile
# The DLC ships a driver matched to the torch it contains, so we never touch the cu13
# wheels in uv.lock. Only the non-torch deps are installed on top.
ARG BASE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0-cu128-py311-ubuntu22.04
FROM ${BASE}

RUN pip install --no-cache-dir \
      "transformers>=5.17.0" \
      "peft>=0.21.0" \
      "datasets>=5.0.1" \
      "pyyaml>=6.0.3"

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /opt/ml/code
COPY . /opt/ml/code
```

`requirements.txt` — let the SDK own the rest, and record what you actually got:

```
sagemaker>=2.240
```

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/jev-train
TAG=$(date +%Y%m%d-%H%M%S)

docker build -t $REPO:$TAG .
docker tag  $REPO:$TAG $REPO:latest
docker push $REPO:$TAG
docker push $REPO:latest
```

**The one thing to smoke-test early:** your `uv.lock` pairs `transformers>=5.17` with
`torch>=2.14`. You are now running those on torch 2.8. The specific call sites that could break:

| Call site | Risk |
|---|---|
| `network.py:143` `Qwen3Model.from_pretrained(model_name, dtype=...)` | `dtype=` was added in transformers ~4.56. Safe on 5.x |
| `network.py:171` `get_peft_model(...)`, `LoraConfig(target_modules=<regex str>)` | peft 0.21 accepts a regex string. Verify |
| `train.py:113` `set_peft_model_state_dict(...)` -> `.missing_keys` / `.unexpected_keys` | This return shape has changed across peft versions. **Most likely breakage point** |
| `inference.py:14` `torch.backends.mps.is_available()` | Unaffected |

`configs/smoke.yaml` overfits 10 questions and touches every one of these in about 3 minutes. Run it
before anything else (see [Step 6](#step-6--run-the-pilot-then-size-the-real-run)).

---

## Step 4 — Write the SageMaker entry script

`train.py` writes to a hardcoded relative `runs/` (`logger.py:18`, `root="runs"`), so the entry
script must `chdir` into the code directory and patch three things.

`sagemaker_entry.py`:

```python
"""SageMaker entry point for bev-train.

Three patches over train.py:
  1. restore the HF cache from S3 (no network in the training container)
  2. point the run root at /opt/ml/output/data so checkpoints land in the synced output
  3. sync the run dir to S3 after every rolling resume save, so a Spot interruption
     does not lose the run (SageMaker only syncs output/ on successful completion)
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/cache/hf")

CODE_DIR = "/opt/ml/code"
RUN_ROOT = os.environ.get("RUN_ROOT", "/opt/ml/output/data/runs")
S3_SYNC_URI = os.environ.get("S3_SYNC_URI", "")  # e.g. s3://bucket/runs


def restore_hf_cache():
    """Populate HF_HOME/hub from S3 so the container never needs internet."""
    if not S3_SYNC_URI:
        return
    hub = Path(os.environ.get("HF_HOME", "/cache/hf")) / "hub"
    bucket = S3_SYNC_URI.rsplit("/runs", 1)[0]
    hub.mkdir(parents=True, exist_ok=True)
    print(f"[entry] restoring HF cache from {bucket}/hf-cache", flush=True)
    subprocess.run(
        [sys.executable, "scripts/prepare_s3_cache.py", "restore",
         "--bucket", bucket.split("//", 1)[-1].split("/", 1)[0],
         "--dest", str(hub),
         "--region", os.environ.get("AWS_REGION", "us-east-1")],
        check=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def verify_gpu():
    """Fail loudly and early. A silent CPU fallback on a 20h job is very expensive."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not visible to torch. This is the #1 cause of a silent CPU fallback; "
            "check that the DLC driver matches the installed torch."
        )
    print(
        f"[entry] torch {torch.__version__} | CUDA {torch.version.cuda} | "
        f"{torch.cuda.get_device_name(0)} | "
        f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB",
        flush=True,
    )
    print(f"[entry] transformers {__import__('transformers').__version__}", flush=True)
    print(f"[entry] peft {__import__('peft').__version__}", flush=True)
    print(f"[entry] datasets {__import__('datasets').__version__}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sagemaker_full.yaml")
    parser.add_argument("--name", default="sagemaker")
    parser.add_argument("--resume-state", default=None,
                        help="a restored runs/<id> folder to continue (same run dir)")
    parser.add_argument("--resume", default=None,
                        help="checkpoint folder or run id: LoRA + head weights only")
    args = parser.parse_args()

    restore_hf_cache()
    os.chdir(CODE_DIR)
    verify_gpu()

    import yaml

    from logger import RunLogger

    # --- patch: sync the run dir to S3 after every rolling save -------------------
    # _orig writes the slot atomically first, so an interrupted sync can only ever
    # leave S3 one save behind the local disk, never a half-written slot.
    original_save_resume_state = RunLogger.save_resume_state

    def save_resume_state_and_sync(self, network, train_state):
        result = original_save_resume_state(self, network, train_state)
        if S3_SYNC_URI:
            # check=False: never kill a 20-hour training run because S3 was briefly unhappy
            subprocess.run(
                ["aws", "s3", "sync", str(self.run_dir),
                 f"{S3_SYNC_URI}/{self.exp_id}", "--quiet", "--exclude", "checkpoints/latest"],
                check=False,
            )
        return result

    RunLogger.save_resume_state = save_resume_state_and_sync

    from train import train

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # resume-state expects `runs/<id>`; RunLogger(root=...) is the only lever it exposes
    if args.resume_state:
        run_dir = Path(args.resume_state)
        if not run_dir.exists():
            sys.exit(f"[entry] --resume-state path does not exist: {run_dir}")
        if not (run_dir / "checkpoints" / "resume" / "pointer").exists():
            sys.exit(f"[entry] no resume pointer under {run_dir}/checkpoints/resume")
        print(f"[entry] resuming from {run_dir}", flush=True)

    os.makedirs(RUN_ROOT, exist_ok=True)

    # NOTE: train() constructs RunLogger(config, name, device) with root="runs" (relative),
    # so the chdir above is what actually decides where runs land. If you need RUN_ROOT to
    # win, symlink it instead:
    #   (Path(CODE_DIR) / "runs").symlink_to(RUN_ROOT, target_is_directory=True)
    train(config, args.name, resume=args.resume, resume_state=args.resume_state)


if __name__ == "__main__":
    main()
```

### About that `runs` symlink

`RunLogger.__init__` hardcodes `root="runs"` (`logger.py:18`) and `train.py:198` calls it with no
`root=` argument. So there is no environment-variable lever. Two options:

**Option A (simplest, what the script above does).** `os.chdir(CODE_DIR)` and let `runs/` be a real
directory inside the code volume. It works, but it lands in the container writable layer rather than
`/opt/ml/output/data`.

**Option B (recommended).** Add one line to `logger.py` so the root is configurable:

```python
    def __init__(self, config, run_name, device, root=None, run_dir=None):
        self.config = config
        if run_dir is not None:
            ...
        else:
            root = root or os.environ.get("RUNS_ROOT", "runs")
            self.exp_id = f"{run_name}-{datetime.now():%Y%m%d-%H%M%S}"
            self.run_dir = Path(root) / self.exp_id
```

Then in the entry script, drop the `os.makedirs(RUN_ROOT)` / symlink dance and just set
`RUNS_ROOT=/opt/ml/output/data/runs`. This also means you can run the same container locally with
`RUNS_ROOT=./runs` and get identical behaviour — useful when debugging.

### `.sagemakerignore`

`source_dir` gets tarred and uploaded. Do not upload your `runs/` directory — it is full of LoRA
checkpoints from your local ablations.

```gitignore
runs/
.venv/
__pycache__/
*.pyc
*.log
*.ipynb
.git/
.gitignore
uv.lock
```

(`runs/` is already in `.gitignore:13`, but `.sagemakerignore` is a separate mechanism.)

---

## Step 5 — Write the training configs

### `configs/sagemaker_smoke.yaml`

Identical to `configs/smoke.yaml`. Overfits 10 questions in ~3 minutes and exercises every call
site that can break on a version skew. **Always run this first.**

```yaml
seed: 0
model:
  name: Qwen/Qwen3-0.6B
  num_layers: 20
data:
  max_train_questions: 10
  max_test_questions: 0
  max_dev_questions: 50
  max_state_tokens: 512
  max_choice_tokens: 64
  batch_size: 5
  num_workers: 2
lora:
  last_k_layers: 4
  r: 8
  alpha: 16
  dropout: 0.0
head:
  new_dim: 512
  num_layers: 2
optim:
  epochs: 30
  lr_lora: 1.0e-4
  lr_head: 1.0e-3
  weight_decay: 0.01
  warmup_ratio: 0.05
  grad_clip: 1.0
logging:
  log_every: 2
  eval_every: 1000
  save_every: 1000
```

Success looks like `loss` → ~0 and `accuracy` → 1.0 within 30 epochs, plus a
`Saved checkpoint ->` line and a `summary.json`. If `loss` plateaus near 1.4 (uniform over 2-3
options), the model is not learning — check that CUDA is actually being used, not just that it is
available.

### `configs/sagemaker_pilot.yaml`

Start from `full_lora_k12.yaml` and change exactly four things:

```yaml
seed: 0
model:
  name: Qwen/Qwen3-0.6B
  num_layers: 20
data:
  configs: [all]
  max_train_questions: 20000      # was null -> ~420K. This is the pilot cap.
  max_test_questions: 0
  max_dev_questions: 2000
  max_state_tokens: 512           # was 1024. Halves quadratic attention cost.
  max_choice_tokens: 64
  batch_size: 16                  # was 32. Halves the B x L x L mask. 16 x 4 = same effective 64.
  num_workers: 12                 # was 6. g5.4xlarge has 16 vCPU.
lora:
  target: attn
  last_k_layers: 12
  r: 8
  alpha: 16
  dropout: 0.0
head:
  new_dim: 512
  num_layers: 2
  ctx_queries: 0
loss:
  label_smoothing: 0.0
  emd_weight: 0.0
  use_soft_labels: false
  balance_types: false
optim:
  epochs: 1
  lr_lora: 1.0e-4
  lr_head: 1.0e-3
  weight_decay: 0.01
  warmup_ratio: 0.05
  grad_clip: 1.0
  grad_accum_steps: 4             # was 2. 16 x 4 keeps effective batch at 64.
  resume_save_seconds: 300        # rolling save + S3 sync every 5 min, not 30
logging:
  log_every: 20
  eval_every: 250
  save_every: 250
```

Two notes on what is deliberately preserved:

- **Effective batch stays 64.** `batch_size 16 x grad_accum_steps 4` = 64, same as the baseline's
  `32 x 2`. Your LR schedule and comparison against the laptop results both depend on this.
  `train.py:250` will *warn* rather than fail if the effective batch changes on resume, but keeping
  it constant means optimizer positions line up exactly.
- **`resume_save_seconds: 300` instead of the default `resume_save_minutes: 30`.** The save is
  atomic and the last two slots are kept (`logger.py:75`), so this is cheap. On a 20-hour Spot run
  you want to lose at most 5 minutes of work.

### `configs/sagemaker_full.yaml`

Set `max_train_questions: null` to use the whole `all` split, but only after the pilot has told you
the real wall-clock. If the pilot says a full epoch will not fit in your `MaxRuntimeInSeconds`,
set an explicit cap instead — a truncated epoch on 200K questions beats an interrupted one on 420K,
and the cosine schedule is computed from `total_steps` derived from the actual question count
(`train.py:229`), so a capped run is internally consistent.

Also consider `max_state_tokens: 1024` vs `512`. The README notes states up to ~17K chars in the
`skills` config, so 512 truncates meaningfully. The tradeoff:

| `max_state_tokens` | Effect | Cost |
| --- | --- | --- |
| 512 | Truncates long states (esp. `skills`) | Fast; ~2.5-3x cheaper than 1024 |
| 1024 | Fewer truncations | Baseline config; the honest number |

Report which one you used. A per-domain accuracy breakdown (`inference.py` emits
`accuracy_domain_*`) will show you whether truncation is hurting the long-document domains.

---

## Step 6 — Run the pilot, then size the real run

### 6a. Smoke test (3 min, ~$0.40)

```python
import sagemaker
from sagemaker.estimator import Estimator

sm = sagemaker.Session()
est = Estimator(
    image_uri="ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/jev-train:latest",
    source_dir=".",                       # bev-train-main/
    entry_point="sagemaker_entry.py",
    role="arn:aws:iam::ACCOUNT:role/JevSagemakerRole",
    hyperparameters={
        "config": "configs/sagemaker_smoke.yaml",
        "name": "smoke",
    },
    instance_count=1,
    instance_type="ml.g5.4xlarge",
    volume_size=100,
    max_runtime=1800,
    base_job_name="jev-smoke",
    environment={
        "S3_SYNC_URI": f"s3://{BUCKET}/runs",
        "HF_HOME": "/cache/hf",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    },
    sagemaker_session=sm,
)
est.fit()
print(est.wait())
```

Then read the log:

```bash
aws logs get-log-events \
  --log-group-name /aws/sagemaker/us-east-1/SageMaker/jev-smoke \
  --log-stream-name $(aws sagemaker describe-training-job \
      --training-job-name <job> --query 'LatestMonitoringResources' --output text 2>/dev/null || echo "*") \
  --start-from-head | grep -E "entry|loss|accuracy|Saved|Error|Traceback"
```

**The smoke run must pass cleanly before anything else.** It is 3 minutes and it catches the peft /
transformers version-skew problems that would otherwise surface 2 hours into a $120 job.

### 6b. Pilot (measure throughput)

Same script, swap the config. Expected: ~20 min on `g5.4xlarge`.

Extract the throughput from `metrics.jsonl`:

```bash
aws s3 cp s3://$BUCKET/runs/sagemaker-pilot-*/metrics.jsonl - | \
  python -c "
import json, sys
rows = [json.loads(l) for l in sys.stdin if l.strip()]
train = [r for r in rows if r.get('split') == 'train']
t0, t1 = train[0]['time'], train[-1]['time']
step = train[-1]['step'] - train[0]['step']
qps = step * 64 / (t1 - t0)          # 64 = effective batch
print(f'{qps:.2f} questions/sec  ({step} steps in {t1-t0:.0f}s)')
print(f'full 420K epoch: {420000/qps/3600:.1f} hours')
"
```

### 6c. Extrapolate

Use the measured `qps`, not a guess:

```
full_epoch_hours = 420_000 / qps / 3600
```

Then apply the quadratic-attention correction if you want to move between `max_state_tokens`
values. Attention is O(L²) and the feed-forward path is O(L), so:

```
scale(L) ≈ (a * L + b * L²) / (a * L0 + b * L0²)     # a, b unknown; fit from two pilots
```

In practice: **measure at 512, then measure a 5,000-question run at 1024, and interpolate.** Do
not try to predict 1024 from 512 — the quadratic term dominates and the extrapolation is wrong.

Set `max_train_questions` from the result:

```
max_train_questions = int(qps * max_runtime_hours * 3600 * 0.8)     # 0.8 = safety margin
```

The 20% margin covers the first-epoch dataset load (bug B), the initial HF model load, and
validation passes. Clamp to 420,000.

---

## Step 7 — Launch the full run

### On-demand (predictable, expensive)

```python
est = Estimator(
    image_uri=IMAGE,
    source_dir=".",
    entry_point="sagemaker_entry.py",
    role=ROLE,
    hyperparameters={
        "config": "configs/sagemaker_full.yaml",
        "name": "full",
    },
    instance_count=1,                      # MUST be 1. No DDP in this codebase.
    instance_type="ml.g5.4xlarge",
    volume_size=200,                       # g5 hosts have 125GB EBS; dataset + cache + runs
    max_runtime=86400,                     # 24h. SageMaker training max is 28 days.
    base_job_name="jev-full",
    tags=[{"Key": "project", "Value": "jev"}],
    environment=ENV,
    sagemaker_session=sm,
)
est.fit()
```

### Spot (cheaper, needs the resume path)

```python
est = Estimator(
    image_uri=IMAGE,
    source_dir=".",
    entry_point="sagemaker_entry.py",
    role=ROLE,
    hyperparameters={
        "config": "configs/sagemaker_full.yaml",
        "name": "full",
        "resume_state": "/opt/ml/input/resume/runs/<exp_id>",
    },
    instance_count=1,
    instance_type="ml.g5.4xlarge",
    volume_size=200,
    use_spot_instances=True,
    max_runtime=43200,                     # <= MaxRuntimeInSeconds or SageMaker rejects it
    checkpoint_config=sagemaker.checkpoint.CheckpointConfig(
        local_checkpoint_dir="/opt/ml/checkpoint",     # single dir, not a list
        s3_uri=f"s3://{BUCKET}/ckpt",
    ),
    base_job_name="jev-full-spot",
    environment=ENV,
    sagemaker_session=sm,
)
```

Two constraints that bite people:

- **`local_checkpoint_dir` must be a single string, not a list.** Your resume state is one rolling
  slot with a pointer file (`logger.py:86`), so there is nothing to fan out. A list of dirs would
  mean SageMaker is managing a rotation scheme you do not have.
- **`max_runtime` must be ≤ `MaxRuntimeInSeconds`** when using Spot, or the API rejects it.

For Spot, SageMapper will auto-resume the *job* from `CheckpointConfig`, but it cannot restore
*your* optimizer state — that comes from the S3-synced run dir via `--resume-state`. See
[Step 8](#step-8--spot-interruption-and-resume).

### Instance type choice

| Instance | GPU | vCPU | RAM | Verdict |
| --- | --- | ---: | ---: | --- |
| `ml.g5.xlarge` | 1× A10G 24GB | 4 | 16 GiB | Too few vCPU — the tokenizer starves. Avoid |
| `ml.g5.2xlarge` | 1× A10G 24GB | 4 | 32 GiB | Same problem. Avoid |
| **`ml.g5.4xlarge`** | 1× A10G 24GB | 16 | 128 GiB | **Sweet spot.** ~$6.19/hr on-demand |
| `ml.g5.12xlarge` | 1× A10G 24GB | 48 | 384 GiB | Same speed, more room for `num_workers`. Best if you cache the dataset |
| `ml.g5.24xlarge` | 4× A10G | 96 | 384 GiB | **No faster.** 3 of the 4 GPUs sit idle |
| `ml.g5.48xlarge` | 8× A10G | 192 | 768 GiB | Same. Do not pay for these |

Note that `g5.24xlarge` and `g5.48xlarge` have a higher hourly price but the *same* wall clock as
`g5.4xlarge`, because the code is strictly single-GPU. Pick `g5.4xlarge` or `g5.12xlarge` on price.

Other A10G-backed families are also fine and sometimes cheaper on Spot: `ml.g6.xlarge`
(L4 24GB), `ml.g6.2xlarge`, `ml.g5e.xlarge` (A10G, often the cheapest Spot option).

---

## Step 8 — Spot interruption and resume

### What happens on interruption

1. Sagemaker sends SIGTERM, then SIGKILL after 2 minutes.
2. The container and its EBS volume are destroyed.
3. `/opt/ml/output/data` is **never** synced — that only happens on success.
4. Your last `aws s3 sync` from `sagemaker_entry.py` survives at
   `s3://$BUCKET/runs/<exp_id>/`.

The saved state includes LoRA weights, head weights, optimizer state, scheduler state, step
counter, epoch, best val accuracy, and full RNG state (`train.py:155`). `logger.py:75` keeps the
last two slots and flips a pointer atomically, so a save interrupted mid-write can only damage the
slot that is *not* currently pointed to.

### Manual resume

```bash
# 1. What survived?
aws s3 sync s3://$BUCKET/runs/ ./restored-runs/

# 2. Check how far it got
cat ./restored-runs/*/checkpoints/resume/pointer
python -c "
import torch, glob
p = glob.glob('./restored-runs/*/checkpoints/resume/*/train_state.pt')[0]
s = torch.load(p, map_location='cpu', weights_only=False)
print(f\"step {s['step']}, best_val {s['best_val']:.4f}, micro {s['micro']} x accum {s['accum']}\")
"

# 3. Relaunch with resume-state
#    (hyperparameters={"config": ..., "name": "full", "resume_state": "<restored path>"})
```

Or from a notebook / Studio instance, without a job:

```python
est.fit(...)
# after the interruption, re-fit with the resume hyperparameter set
```

### Resuming is exact, and the code enforces it

`train.py:248-253`:

```python
if resumed_state.get("total_questions") != total_questions:
    raise ValueError(f"train set changed since the save: ...")
if resumed_state.get("effective_batch") != effective_batch:
    print(f"note: effective batch changed ...")
set_rng_state(resumed_state["rng"])
```

- The **cosine schedule continues exactly** (`scheduler.load_state_dict`), because
  `total_steps` is recomputed from the same dataset and config.
- The **epoch permutation is reconstructed** from `(seed, epoch)` (`train.py:136`), and the
  already-consumed head of the current epoch is skipped by slicing the index list — skipped examples
  are never re-tokenized (`train.py:281`).
- **RNG state** (torch, python, cuda) is restored, so dropout and any sampling continue the same
  stream.
- If you change `max_train_questions` between the interruption and the resume, it hard-fails. That
  guard is worth keeping — a silent mismatch would corrupt the run.

`--resume` (no `-state`) is different: it loads **only** LoRA + head weights and starts the optimizer
and LR schedule fresh. That is for intentional second-stage runs (see `full_stage2.yaml`), not for
crash recovery.

---

## Step 9 — Calibration and evaluation as separate jobs

Both read a checkpoint and need no GPU training. Run them on `ml.m5.4xlarge` (CPU-only, ~1/6 the
cost) or reuse your GPU box if you are already paying for it.

```python
eval_est = Estimator(
    image_uri=IMAGE,
    source_dir=".",
    entry_point="sagemaker_entry.py",
    role=ROLE,
    hyperparameters={
        "config": "configs/sagemaker_full.yaml",
        "name": "eval",
        "eval_checkpoint": "best",       # your entry script branch
        "eval_max_questions": "0",       # 0 = full test split
    },
    instance_count=1,
    instance_type="ml.m5.4xlarge",
    max_runtime=14400,
    environment=ENV,
    sagemaker_session=sm,
)
```

Add an eval branch to `sagemaker_entry.py`:

```python
    if args.eval_checkpoint:
        from dataset import BEVDataset, collate_fn, load_questions
        from inference import evaluate, get_device
        from logger import load_checkpoint

        ckpt = Path(args.eval_checkpoint)
        if not ckpt.is_absolute():
            ckpt = Path(os.environ.get("CKPT_DIR", "/opt/ml/input/checkpoint")) / ckpt
        device = get_device()
        network, meta = load_checkpoint(ckpt, device)
        questions = load_questions(args.eval_split, args.eval_max_questions or None,
                                   configs=meta.get("data_configs", ["all"]))
        dataset = BEVDataset(questions, meta["max_state_tokens"], meta["max_choice_tokens"])
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=collate_fn,
                                             num_workers=8)
        metrics = evaluate(network, loader, device, meta.get("temperatures"))
        print("EVAL_METRICS " + json.dumps(metrics), flush=True)
        return
```

Pass the checkpoint in as a channel — never bake it into the image:

```python
estimator.fit(inputs={
    "checkpoint": f"s3://{BUCKET}/runs/<exp_id>/checkpoints",
    "resume": "s3://<optional>/",
})
```

### Order matters: calibrate *before* evaluating

```bash
python calibrate.py runs/<exp_id>/checkpoints/best --config configs/sagemaker_full.yaml
python inference.py  runs/<exp_id>/checkpoints/best --config all --max_questions 0
```

`calibrate.py` writes `calibration.json` into the checkpoint folder, `logger.load_checkpoint`
picks it up (`logger.py:135`), and `inference.py` applies the temperatures (`inference.py:85`).
Evaluating without calibrating gives you the uncalibrated ECE numbers — which is a different,
less useful measurement.

**Reminder:** this only works after [Step 0 Bug A](#step-0--fix-the-two-code-bugs) is fixed.

### Full test-split eval cost

The test split is **80,019 questions**. Your `run_full_evals.sh` runs it at
`--batch_size 8 --num_workers 0` sequentially. On `m5.4xlarge` this is slow — budget a few hours.
Raise `--batch_size` and `--num_workers` for the CPU job; `g5.12xlarge`-class CPU/RAM helps too.

Also note `run_full_evals.sh:3`: *"2-way concurrency measured slower (GPU already ~100% utilized)
and concurrent HuggingFace dataset loads raced on the cache."* If you parallelize evals across
jobs, each job gets its own container and its own cache, so the race disappears — but do not run two
evals in one container.

---

## Step 10 — The ablation ladder

The README's small ladder result: **`lora target=attn_mlp, r=16, alpha=32` was the only change that
beat the baseline** (+4.0 pts accuracy, best per-type accuracy in 3 of 3 types, ECE on par). That is
`configs/full_lora_mlp.yaml`.

Each row of the README ladder table is a separate training run. On SageMaker, **run them as
parallel single-instance jobs** rather than sequentially:

```python
from concurrent.futures import ThreadPoolExecutor

RUNS = {
    "baseline":   "configs/full_lora_k12.yaml",
    "loss":       "configs/full_loss.yaml",
    "ctx":        "configs/full_ctx.yaml",
    "lora-mlp":   "configs/full_lora_mlp.yaml",
    "layers24":   "configs/full_layers24.yaml",
    "layers28":   "configs/full_layers28.yaml",
    "softlabels": "configs/full_softlabels.yaml",
    # stage2 depends on baseline's checkpoint -> run after
}

def launch(name, config):
    est = Estimator(..., hyperparameters={"config": config, "name": name},
                    base_job_name=f"jev-{name}", ...)
    est.fit(wait=False)
    return est

with ThreadPoolExecutor(max_workers=4) as pool:   # cap at 4 to stay inside your vCPU quota
    jobs = dict(zip(RUNS, pool.map(lambda kv: launch(*kv), RUNS.items())))
for name, est in jobs.items():
    est.wait(); print(name, est.wait())
```

Caveats:

- **Cap concurrency at your quota.** 7 concurrent `g5.4xlarge` jobs = 112 vCPU. New accounts have
  0–4. Check [Prerequisites](#prerequisites).
- **Every job re-loads and re-explodes the full dataset** (bug B). Seven jobs means seven times that
  cost. Fix bug B or accept it.
- **`full_stage2.yaml` depends on a prior checkpoint** — it must run *after* the baseline finishes,
  with `--resume runs/<baseline_id>/checkpoints/final`. It is not independent.
- `ctx_queries=8` and `num_layers: 28` both *hurt* at small scale. Running them at full scale is
  expensive for a likely-negative result. Consider skipping unless you specifically need the answer.

### Comparing runs

`summary.json` per run, same schema everywhere:

```bash
aws s3 sync s3://$BUCKET/runs/ ./all-runs/
python - <<'PY'
import json, pathlib
rows = []
for p in sorted(pathlib.Path("all-runs").glob("*/summary.json")):
    s = json.loads(p.read_text())
    rows.append((p.parent.name, s.get("final_train_accuracy"),
                 s.get("best_val_accuracy"), s.get("duration_sec", 0) / 3600))
print(f"{'run':<40} {'train_acc':>10} {'val_acc':>9} {'hours':>6}")
for n, tr, va, h in rows:
    print(f"{n:<40} {tr:>10.4f} {va:>9.4f} {h:>6.1f}")
PY
```

For the full test split, extend the pattern in `summarize_full_evals.py` — it already parses the
`{'loss': ...}` dict from each eval log and emits a markdown comparison table.

---

## Sizing the run

**Measured, from your own runs** (`runs/ladder-loss-20261006-215757/metrics.jsonl`):

```
2,000 questions @ max_state_tokens=256, batch_size=8, RTX 4050 → 8.4 q/s
```

At `max_state_tokens: 1024` the quadratic attention term dominates; expect **roughly 3-4× slower**
than the 256-token number, i.e. **~2-3 q/s on an RTX 4050**. An A10G (24 GB) is roughly 2.5-3× an
RTX 4050 for bf16 autocast workloads.

**Treat every number below as an estimate with a ±50% band until the pilot measures it:**

| Instance | `max_state_tokens` | Estimated q/s | 420K epoch |
| --- | ---: | ---: | ---: |
| `g5.4xlarge` | 512 | ~8-12 | **10-15 h** |
| `g5.4xlarge` | 1024 | ~5-8 | **15-24 h** |
| `g5.12xlarge` | 1024 | ~5-8 | 15-24 h |

Formulas:

```
full_epoch_hours = 420_000 / measured_qps / 3600
max_train_questions = int(measured_qps * max_runtime_hours * 3600 * 0.8)
```

### Startup overhead (budget for it)

Before step 1 of training, every job pays:

1. Model + dataset download from S3 (`s3 sync`, a few minutes for ~9 GB).
2. `load_dataset` + the md5 `filter` over 231,332 rows — this is a Python-level
   `ds.filter(lambda row: ...)` (`dataset.py:124`), so it is **single-threaded and slow**.
3. `explode_questions` (`dataset.py:68`) — a batched map that JSON-parses every row.
4. **A second full pass for the val split** (bug B).

Realistically 10–20 minutes. It shows up as a long gap between the first log line and
`step=20`. That gap is not a hang.

---

## Cost

Rates are us-east-1 on-demand list prices and move; Spot fluctuates 40-70% off.

| | On-demand | Spot (~60% off) |
| --- | ---: | ---: |
| `g5.4xlarge`, 15 h at 1024 tokens | ~$93 | ~$37 |
| `g5.4xlarge`, 24 h at 1024 tokens | ~$149 | ~$59 |
| `g5.4xlarge`, 12 h at 512 tokens | ~$74 | ~$30 |
| 7-run ablation ladder (2 h each, parallel) | ~$87 | ~$35 |
| Full test eval on `m5.4xlarge`, 4 h | ~$10 | n/a |
| **Realistic total (1 full run + ladder + evals)** | **~$250-350** | **~$110-150** |

The dominant term is the full run's wall clock, and wall clock is set by `max_state_tokens`. If you
want to halve your bill, drop `max_state_tokens` to 512 first and report that you did.

---

## Monitoring a running job

```bash
# Console output, live
aws logs tail /aws/sagemaker/$AWS_REGION/SageMaker/jev-full

# Download everything at the end
aws s3 sync s3://$BUCKET/runs/ ./results/

# Live training curve (from a synced metrics.jsonl)
python - <<'PY'
import json, pathlib
p = sorted(pathlib.Path("results").glob("*/metrics.jsonl"))[-1]
for line in p.read_text().splitlines():
    r = json.loads(line)
    if r.get("split") == "train":
        print(f"step {r['step']:>6}  t={r['time']:>8.0f}s  "
              f"loss={r.get('loss', float('nan')):.4f}  acc={r.get('accuracy', float('nan')):.4f}")
PY
```

Healthy signals:

- `step` increases roughly linearly with `time`.
- `loss` decreases from ~1.4 toward ~1.1 and then slowly.
- `dev` accuracy appears every `eval_every` steps and is stable or improving.
- `Saved checkpoint -> .../checkpoints/latest` appears every `save_every` steps.
- `aws s3 sync` of the run dir every 5 minutes (check S3 object timestamps).

Failure signals:

- **No progress past step 0 for 20+ min** → startup overhead (see above), not a hang. Check the log
  for the `Map:` progress bars.
- `CUDA out of memory` → lower `batch_size`, raise `grad_accum_steps` to keep effective batch 64.
- `RuntimeError: CUDA not visible to torch` → bug #1. Your image's driver does not match its torch.
- `ValueError: Unknown split "dev"` → bug A. You skipped Step 0.
- `ValueError: train set changed since the save` → correct behaviour. Your config changed between
  the interruption and the resume.

---

## Troubleshooting

**`torch.cuda.is_available()` is False on an A10G.**
The DLC driver does not match the installed torch. Confirm with
`python -c "import torch; print(torch.version.cuda)"` in the container and compare against the
image's `nvidia-smi` driver version. If you `pip install`-ed a torch that expects a newer driver,
that is the cause.

**`Unknown split "dev"`.**
[Step 0, Bug A](#step-0--fix-the-two-code-bugs). Not yet fixed.

**`peft` `set_peft_model_state_dict` has no `missing_keys`.**
Version skew, [Step 3](#step-3--build-the-ecr-image). Either pin `peft` to the version your local
`uv.lock` resolved, or adapt `train.py:113-116` to the new return type.

**Job fails at ~10 min with `Couldn't find a valid CUDA image`.**
No usable Spot capacity. Either switch to on-demand, or raise the Spot max price:

```python
est.fit(spot_config={"max_price": "1.50"})
```

**Job OOMs during the initial validation pass.**
`train.py:265` runs a full validation before step 0 with `network.eval()` but at `batch_size` from
the config. If the training micro-batch fits, this should too — unless `max_dev_questions` is large
enough that a long-state batch is unlucky. Lower `max_val_questions`.

**Everything is 3× slower than the pilot.**
Usually `num_workers` too low for the instance, or CPU throttling. Check
`nvidia-smi` utilization inside the container; if the GPU is not ~100% busy, the dataloader is the
bottleneck. Raise `num_workers` toward `vCPU - 2`.

**Dataset download is very slow.**
You skipped [Step 2](#step-2--pre-stage-the-dataset-and-tokenizer-to-s3) and are pulling ~9 GB from
HuggingFace out of AWS every job. Pre-stage it.

**Output bucket is enormous.**
You put `HF_HOME` under `/opt/ml/output/data`. Move it to `/cache/hf`.

**`AccessDenied` on `s3 sync` from inside the container.**
The job role needs `s3:PutObject`. `AmazonSageMakerFullAccess` includes it; a custom role must list
it explicitly (see [Step 1](#step-1--aws-setup)).

---

## Appendix: SageMaker filesystem layout

| Path | Writable | Lifespan | Contents |
| --- | --- | --- | --- |
| `/opt/ml/code` | no (until you copy) | job | your `source_dir`, extracted |
| `/opt/ml/input/data/<channel>` | no | job | S3 channels |
| `/opt/ml/model` | yes | job | synced to the model channel on success |
| `/opt/ml/output/data` | yes | job | **synced to the output channel on success only** |
| `/opt/ml/output/model` | yes | job | same, for the model channel |
| `/opt/ml/checkpoint` | yes | job | SageMaker-managed, spot-restored |
| `/` and `/cache` | yes | **instance** | the EBS volume, destroyed with the container |

The critical distinction is the last row. `/cache` dies with the instance. `/opt/ml/output/data`
is only uploaded on success. Your rolling resume save must therefore do its own `aws s3 sync`,
which is what `sagemaker_entry.py` patches in.

`instance_type` determines the EBS volume size (100 GB for `g5.4xlarge`, up to 1,950 GB for larger
instances) and it is not configurable in older regions — set `volume_size` explicitly and keep it
above `HF cache (~9 GB) + dataset cache + checkpoints + scratch`.