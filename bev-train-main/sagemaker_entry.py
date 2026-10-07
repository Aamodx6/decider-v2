"""SageMaker entry point.

Four things happen before a single gradient is computed:

1. restore the HuggingFace cache from S3, so the container never needs internet
2. assert CUDA is visible, because a silent CPU fallback costs days not minutes
3. point RUNS_ROOT at /opt/ml/output/data, the only directory SageMaker uploads
4. sync the run dir to S3 after every rolling resume save, because SageMaker only
   uploads output/ on *successful* completion and a Spot interruption destroys the
   container's disk

Then it calls train.train() exactly as the CLI does, so a run is reproducible either way.

    python sagemaker_entry.py --config configs/sagemaker_full.yaml --name full
    python sagemaker_entry.py --config configs/sagemaker_full.yaml --name full \
        --resume-state /opt/ml/input/resume/runs/<exp_id>
    python sagemaker_entry.py --eval-checkpoint /opt/ml/input/checkpoint/checkpoints/best
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CODE_DIR = "/opt/ml/code"
S3_SYNC_URI = os.environ.get("S3_SYNC_URI", "")  # e.g. s3://bucket/runs


def restore_hf_cache():
    """Populate $HF_HOME/hub from S3. Both dataset.py and tokenization.py resolve their
    assets by Hub name, so the same cache layout means zero code changes to run offline."""
    hf_home = Path(os.environ.get("HF_HOME", "/cache/hf"))
    if not S3_SYNC_URI:
        print("[entry] S3_SYNC_URI unset; skipping cache restore", flush=True)
        return
    bucket = S3_SYNC_URI.rsplit("/runs", 1)[0].split("//", 1)[-1].split("/", 1)[0]
    region = os.environ.get("AWS_REGION", "us-east-1")
    print(f"[entry] restoring HF cache from s3://{bucket}/hf-cache", flush=True)
    subprocess.run(
        [sys.executable, "scripts/prepare_s3_cache.py", "restore",
         "--bucket", bucket, "--region", region, "--dest", str(hf_home / "hub")],
        check=True, cwd=CODE_DIR)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def verify_dataset():
    """Cheap preflight that the model and dataset actually resolve, before the 20-minute load.

    The parquet is no longer vendored in-repo, so with HF_HUB_OFFLINE=1 the container depends
    entirely on restore_hf_cache having worked. Without this check a failed restore surfaces
    much later as an opaque "couldn't find the requested files in the cached files".

    The import of `dataset` has to be inside the try: dataset.py imports tokenization, which
    loads the Qwen tokenizer at module import time, so a missing backbone cache blows up here
    before the dataset is ever touched. counterfactual_15k is the smallest config (0.5 MB) and
    still exercises card resolution plus parquet access.
    """
    from datasets import load_dataset

    dataset_name = None
    try:
        from dataset import DATASET_NAME as dataset_name

        probe = load_dataset(dataset_name, "counterfactual_15k", split="train[:1]")
        rows = len(probe)
    except Exception as exc:
        raise SystemExit(
            f"[entry] cannot resolve the model/dataset offline.\n"
            f"  dataset       = {dataset_name}\n"
            f"  backbone      = Qwen/Qwen3-0.6B (tokenizer.py:9)\n"
            f"  HF_HOME       = {os.environ.get('HF_HOME')}\n"
            f"  offline flags = HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} "
            f"HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE')}\n"
            f"  error         = {type(exc).__name__}: {str(exc)[:200]}\n"
            f"Neither the parquet nor the backbone is vendored in this repo, so the cache\n"
            f"restore is required. Run once, from a machine with internet:\n"
            f"  python scripts/prepare_s3_cache.py upload --bucket <your-bucket>\n"
            f"or drop HF_HUB_OFFLINE and HF_DATASETS_OFFLINE from the image ENV and let the\n"
            f"container download from the Hub."
        ) from exc
    print(f"[entry] backbone + dataset resolve: {dataset_name} (probe read {rows} row)", flush=True)


def verify_gpu():
    """A CPU fallback is silent: get_device() just returns torch.device("cpu"). A 20-hour
    job that quietly trains on CPU is the most expensive failure mode here."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not visible to torch. The DLC driver must match the installed torch; "
            "a mismatch falls back to CPU without any other error.")
    print(f"[entry] torch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"{torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB", flush=True)
    for pkg in ("transformers", "peft", "datasets"):
        print(f"[entry] {pkg} {__import__(pkg).__version__}", flush=True)


def install_s3_sync():
    """Wrap RunLogger.save_resume_state so each rolling save is mirrored to S3.
    The original writes the slot atomically first, so an interrupted sync can only ever
    leave S3 one save behind the local disk, never a half-written slot."""
    if not S3_SYNC_URI:
        return
    from logger import RunLogger

    original = RunLogger.save_resume_state

    def save_resume_state_and_sync(self, network, train_state):
        result = original(self, network, train_state)
        # check=False: never kill a long run because S3 was briefly unhappy
        subprocess.run(
            ["aws", "s3", "sync", str(self.run_dir),
             f"{S3_SYNC_URI}/{self.exp_id}", "--quiet"],
            check=False)
        return result

    RunLogger.save_resume_state = save_resume_state_and_sync
    print(f"[entry] resume saves will sync to {S3_SYNC_URI}/<exp_id>", flush=True)


def run_eval(args):
    """Score a checkpoint on the test split. Cheap enough to run on a CPU instance."""
    import torch
    from dataset import BEVDataset, collate_fn, load_questions
    from inference import autocast, evaluate, get_device, to_device
    from logger import load_checkpoint

    ckpt = Path(args.eval_checkpoint)
    if not ckpt.is_absolute():
        ckpt = Path("/opt/ml/input/checkpoint") / ckpt
    device = get_device()
    network, meta = load_checkpoint(ckpt, device)
    questions = load_questions(args.eval_split, args.eval_max_questions or None,
                               configs=[args.eval_config] if args.eval_config
                               else meta.get("data_configs", ["all"]))
    dataset = BEVDataset(questions, meta["max_state_tokens"], meta["max_choice_tokens"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size,
                                         collate_fn=collate_fn,
                                         num_workers=args.eval_num_workers)
    metrics = evaluate(network, loader, device, meta.get("temperatures"))
    print("EVAL_METRICS " + json.dumps(metrics), flush=True)
    Path("/opt/ml/output/data").mkdir(parents=True, exist_ok=True)
    Path("/opt/ml/output/data/eval_metrics.json").write_text(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sagemaker_full.yaml")
    parser.add_argument("--name", default="sagemaker")
    parser.add_argument("--resume", default=None,
                        help="checkpoint folder or run id: LoRA + head weights only, "
                             "optimizer and LR schedule start fresh")
    parser.add_argument("--resume-state", default=None,
                        help="restored runs/<id> to continue: optimizer, scheduler, RNG "
                             "and step counter all resume (exact, not approximate)")
    parser.add_argument("--eval-checkpoint", default=None, help="score a checkpoint instead of training")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--eval-config", default=None, help="dataset config; default the checkpoint's")
    parser.add_argument("--eval-max-questions", type=int, default=0, help="0 = full split")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-num-workers", type=int, default=8)
    args = parser.parse_args()

    os.chdir(CODE_DIR)
    restore_hf_cache()
    verify_gpu()
    verify_dataset()

    if args.eval_checkpoint:
        run_eval(args)
        return

    install_s3_sync()

    import yaml
    from train import train

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.resume_state:
        run_dir = Path(args.resume_state)
        pointer = run_dir / "checkpoints" / "resume" / "pointer"
        if not pointer.exists():
            sys.exit(f"[entry] no resume pointer at {pointer}")
        print(f"[entry] resuming run dir {run_dir}", flush=True)

    train(config, args.name, resume=args.resume, resume_state=args.resume_state)


if __name__ == "__main__":
    main()