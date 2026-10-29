"""SageMaker eval wrapper: runs server-side INSIDE a training container.

SageMaker mounts training-job channels under /opt/ml/input/data/<name>; an evaluate.py job
therefore gets the test/calib shards at fixed paths and needs everything else by argv. A
checkpoint, though, lives in another job's checkpoint prefix (s3://...) - SageMaker does not
spread those across jobs, so this wrapper does the one thing the job cannot do for itself:
download the checkpoint prefix through the container's own execution role, then hand over to
evaluate.py with the model on LOCAL disk.

Every argument is a 1:1 evaluate.py argument except --checkpoint which accepts a raw S3 UL or a
local path. The wrapper's argv therefore stays in lockstep with evaluate.py's argparse (the
notebook's checklist assertion covers the flags) and nothing is re-serialized.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import boto3


def download_s3_dir(uri: str, local: str) -> str:
    """Download a whole "s3://bucket/prefix/" tree; returns the local dir."""
    assert uri.startswith("s3://"), uri
    bucket, _sep, prefix = uri[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/")
    s3 = boto3.client("s3")
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(o["Key"] for o in page.get("Contents", []))
    if not keys:
        raise SystemExit(f"checkpoint prefix empty: {uri}")
    Path(local).mkdir(parents=True, exist_ok=True)
    for key in keys:
        rel = key[len(prefix):].lstrip("/")
        dst = os.path.join(local, rel)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        s3.download_file(bucket, key, dst)
    return local


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="", help="s3:// prefix of a checkpoints dir, or a local path")
    p.add_argument("--reference", default="", help="evaluate.py passthrough (published reference id)")
    p.add_argument("--test_channel", default=os.environ.get("SM_CHANNEL_TEST", "./data/test"))
    p.add_argument("--calib_channel", default=os.environ.get("SM_CHANNEL_CALIB", ""))
    p.add_argument("--out", default=os.environ.get("SM_OUTPUT_DATA_DIR", "./runs/output"))
    p.add_argument("--tag", default="eval")
    p.add_argument("--env_check", type=int, default=0,
                   help="evaluate.py pins answer to requirements.txt; the eval container does not "
                        "need them when torch is the container's")
    p.add_argument("--limit", type=int, default=1600)  # spot-check size on a test split
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--invariance", type=int, default=1)
    p.add_argument("--invariance_examples", type=int, default=32)
    p.add_argument("--invariance_shuffles", type=int, default=32)
    p.add_argument("--predictions_out", default="")
    p.add_argument("--max_state_tokens", type=int, default=1024)
    p.add_argument("--max_choice_tokens", type=int, default=64)
    p.add_argument("--mode", default="bev")
    p.add_argument("--pool", default="bev")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        raise SystemExit(f"download_eval.py: unknown passthrough flags (typo?): {unknown}")
    return args


def main():
    src = parse_args()

    checkpoint = src.checkpoint or ""
    if checkpoint.startswith("s3://"):
        checkpoint = download_s3_dir(checkpoint, "/tmp/checkpoint")
        print(f"[download] checkpoint {src.checkpoint} -> {checkpoint}")
    if not (checkpoint or src.reference):
        raise SystemExit("need --checkpoint (s3:// or local) or --reference")

    cmd = [sys.executable, "evaluate.py",
           "--checkpoint", checkpoint, "--tag", src.tag, "--out", src.out,
           "--test_channel", src.test_channel, "--calib_channel", src.calib_channel,
           "--limit", str(src.limit), "--batch_size", str(src.batch_size),
           "--invariance", str(src.invariance),
           "--invariance_examples", str(src.invariance_examples),
           "--invariance_shuffles", str(src.invariance_shuffles),
           "--max_state_tokens", str(src.max_state_tokens),
           "--max_choice_tokens", str(src.max_choice_tokens),
           "--mode", src.mode, "--pool", src.pool,
           "--env_check", str(src.env_check)]
    if src.reference:
        cmd += ["--reference", src.reference]
    if src.predictions_out:
        cmd += ["--predictions_out", src.predictions_out]
    print("[run]", " ".join(cmd))
    r = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
