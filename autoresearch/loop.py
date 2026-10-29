#!/usr/bin/env python3
"""autoresearch/loop.py -- Autonomous Experiment Orchestrator for decider-v2.

Adapted from Andrej Karpathy's autoresearch paradigm (github.com/karpathy/autoresearch).
Automates the experiment cycle:
  1. Inspects proposed changes (in network.py / train.py / configs)
  2. Runs structural gate checks (order invariance, 4-D mask, loss sanity)
  3. Executes fixed-budget training run (e.g., 20-100 steps or fixed wall-clock)
  4. Evaluates validation metrics (val_loss, val_acc, val_ece, order_invariance_dp)
  5. Ratchets:
       - If val_loss improved AND gates pass -> KEEPS and commits to git
       - If val_loss regressed or gates fail -> DISCARDS and reverts decider-train/ changes
  6. Records experiment outcome in autoresearch/results.tsv

Usage:
  python autoresearch/loop.py --tag v3_contrastive --desc "Evaluate DecisionHeadV3 relative pooling" --config decider-train/configs/autoresearch_v3.yaml --budget_steps 20
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = ROOT / "autoresearch"
RESULTS_TSV = AUTORESEARCH_DIR / "results.tsv"
TRAIN_ROOT = ROOT / "decider-train"
PYTHON = sys.executable


def get_git_head() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT))
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def get_best_val_loss() -> float:
    if not RESULTS_TSV.exists():
        return float("inf")
    best = float("inf")
    with open(RESULTS_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("status") == "keep":
                try:
                    val = float(row.get("val_loss", "inf"))
                    if val < best:
                        best = val
                except ValueError:
                    pass
    return best


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout


def append_result(entry: dict):
    file_exists = RESULTS_TSV.exists()
    fields = [
        "timestamp", "commit", "val_loss", "val_acc", "val_ece",
        "order_invariance_dp", "status", "description"
    ]
    with open(RESULTS_TSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)


def revert_changes():
    print("[RATCHET] Discarding changes in decider-train/ ...")
    run_command(["git", "checkout", "--", "decider-train/"])
    run_command(["git", "clean", "-fd", "decider-train/runs/"])


def commit_changes(tag: str, desc: str, val_loss: float, val_acc: float):
    msg = f"autoresearch({tag}): {desc} [loss: {val_loss:.4f}, acc: {val_acc:.2%}]"
    print(f"[RATCHET] Keeping improvement. Committing: {msg}")
    run_command(["git", "add", "-A"])
    run_command(["git", "commit", "-m", msg])


def main(argv=None):
    p = argparse.ArgumentParser(description="Autoresearch experiment runner for decider-v2")
    p.add_argument("--tag", required=True, help="Short tag for this experiment (e.g. swiglu_head)")
    p.add_argument("--desc", required=True, help="One-line summary of hypothesis and changes")
    p.add_argument("--config", default="decider-train/configs/autoresearch_v2.yaml",
                   help="Config YAML to base experiment on")
    p.add_argument("--budget_steps", type=int, default=20,
                   help="Fixed training step budget for this run (default: 20)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Execution device")
    p.add_argument("--train_channel", default="",
                   help="Custom path to train parquet channel")
    p.add_argument("--val_channel", default="",
                   help="Custom path to val parquet channel")
    p.add_argument("--test_channel", default="",
                   help="Custom path to test parquet channel")
    p.add_argument("--baseline", action="store_true",
                   help="Mark this run as baseline anchor (records keep regardless of prior loss)")
    p.add_argument("--skip_gates", action="store_true",
                   help="Skip preliminary structural test gates (not recommended)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print planned operations without executing")
    args = p.parse_args(argv)

    train_ch = args.train_channel or (
        str(ROOT / "decider_v2_work/golden/data/train") if (ROOT / "decider_v2_work/golden/data/train").exists() else "./data/train"
    )
    val_ch = args.val_channel or (
        str(ROOT / "decider_v2_work/golden/data/val") if (ROOT / "decider_v2_work/golden/data/val").exists() else "./data/val"
    )
    test_ch = args.test_channel or (
        str(ROOT / "decider_v2_work/golden/data/test") if (ROOT / "decider_v2_work/golden/data/test").exists() else "./data/test"
    )

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    head_commit = get_git_head()
    best_loss = get_best_val_loss()

    print("=" * 78)
    print(f"AUTORESEARCH LOOP -- EXPERIMENT: {args.tag}")
    print(f"Description  : {args.desc}")
    print(f"Current HEAD : {head_commit}")
    print(f"Best Val Loss: {best_loss:.4f}")
    print(f"Budget Steps : {args.budget_steps} ({args.device})")
    print(f"Data Channel : {train_ch}")
    print("=" * 78)

    if args.dry_run:
        print("[DRY RUN] Finished without executing.")
        return 0

    # 1. Structural Gate Check
    if not args.skip_gates:
        print("\n[STEP 1/4] Running structural test gates (tests/test_core.py) ...")
        gate_cmd = [PYTHON, "decider-train/tests/test_core.py", "--skip_overfit"]
        rc, out = run_command(gate_cmd)
        if rc != 0:
            print("[FAIL] Structural gates FAILED. Changes violate architectural constraints.")
            print(out[-1000:])
            append_result({
                "timestamp": timestamp,
                "commit": head_commit,
                "val_loss": "N/A",
                "val_acc": "N/A",
                "val_ece": "N/A",
                "order_invariance_dp": "FAILED",
                "status": "gate_failed",
                "description": f"{args.desc} (gates failed)",
            })
            revert_changes()
            return 1
        print("[PASS] Structural gates passed: 4-D mask, order invariance & loss sanity.")

    # 2. Fixed-Budget Training Run
    exp_dir = TRAIN_ROOT / "runs" / f"autoresearch_{args.tag}"
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[STEP 2/4] Executing fixed-budget training ({args.budget_steps} steps) ...")
    train_cmd = [
        PYTHON, "train.py",
        "--config", str(ROOT / args.config),
        "--train_channel", train_ch,
        "--val_channel", val_ch,
        "--test_channel", test_ch,
        "--max_steps", str(args.budget_steps),
        "--epochs", "20",
        "--output_dir", str(exp_dir / "output"),
        "--model_dir", str(exp_dir / "model"),
        "--checkpoint_dir", str(exp_dir / "ckpt"),
        "--log_every", "1",
        "--eval_every", str(args.budget_steps),
        "--save_every", str(args.budget_steps),
        "--num_workers", "0",
    ]

    rc, train_out = run_command(train_cmd, cwd=TRAIN_ROOT)
    if rc != 0:
        print(f"[FAIL] Training crashed with exit code {rc}")
        print(train_out[-1000:])
        append_result({
            "timestamp": timestamp,
            "commit": head_commit,
            "val_loss": "N/A",
            "val_acc": "N/A",
            "val_ece": "N/A",
            "order_invariance_dp": "N/A",
            "status": "crash",
            "description": f"{args.desc} (train crash)",
        })
        revert_changes()
        return 1
    print("[PASS] Training completed successfully.")

    # 3. Validation Evaluation & Order Invariance Check
    print("\n[STEP 3/4] Evaluating validation metrics and order invariance ...")
    val_loss = float("inf")
    val_acc = 0.0
    val_ece = 1.0
    invariance_dp = 0.0

    # Parse metrics from summary or metrics.jsonl
    metrics_path = exp_dir / "output" / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                row = json.loads(line)
                if "val_loss" in row:
                    val_loss = float(row["val_loss"])
                if "val_acc" in row:
                    val_acc = float(row["val_acc"])
                if "val_ece" in row:
                    val_ece = float(row["val_ece"])
            except Exception:
                pass

    # Verify invariance explicitly
    eval_cmd = [
        PYTHON, "evaluate.py",
        "--checkpoint", str(exp_dir / "model" / "latest"),
        "--test_channel", test_ch,
        "--invariance", "1",
        "--invariance_examples", "16",
        "--invariance_shuffles", "16",
        "--out", str(exp_dir / "eval_out"),
    ]
    rc, eval_out = run_command(eval_cmd, cwd=TRAIN_ROOT)
    eval_metrics_path = exp_dir / "eval_out" / "metrics.json"
    if eval_metrics_path.exists():
        try:
            ev_data = json.loads(eval_metrics_path.read_text(encoding="utf-8"))
            if "order_invariance" in ev_data:
                invariance_dp = float(ev_data["order_invariance"].get("max_abs_delta", 0.0))
            if "accuracy" in ev_data:
                val_acc = float(ev_data["accuracy"])
            if "ece" in ev_data:
                val_ece = float(ev_data["ece"])
        except Exception:
            pass
    else:
        dp_match = re.search(r"order invariance: max \|dp\| = ([\d\.e\-\+]+)", eval_out)
        if dp_match:
            invariance_dp = float(dp_match.group(1))

    print(f"Results: val_loss={val_loss:.4f} (best: {best_loss:.4f}), val_acc={val_acc:.2%}, val_ece={val_ece:.4f}, inv_dp={invariance_dp:.2e}")

    # 4. Ratchet Decision
    print("\n[STEP 4/4] Ratchet decision ...")
    improved = (val_loss < best_loss) or args.baseline
    invariance_ok = invariance_dp <= 1e-5

    if improved and invariance_ok:
        status = "keep"
        print(f"[RATCHET KEEP] Improved val_loss: {best_loss:.4f} -> {val_loss:.4f}")
        append_result({
            "timestamp": timestamp,
            "commit": head_commit,
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "val_ece": f"{val_ece:.4f}",
            "order_invariance_dp": f"{invariance_dp:.2e}",
            "status": status,
            "description": args.desc,
        })
        if not args.baseline:
            commit_changes(args.tag, args.desc, val_loss, val_acc)
    else:
        status = "discard"
        reason = "loss did not improve" if not improved else "order invariance violated"
        print(f"[RATCHET DISCARD] {reason} (val_loss: {val_loss:.4f} >= best: {best_loss:.4f})")
        append_result({
            "timestamp": timestamp,
            "commit": head_commit,
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "val_ece": f"{val_ece:.4f}",
            "order_invariance_dp": f"{invariance_dp:.2e}",
            "status": status,
            "description": f"{args.desc} ({reason})",
        })
        revert_changes()

    # Clean up run artifacts to save disk space
    if exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)

    print("=" * 78)
    print(f"Experiment {args.tag} concluded with status: {status.upper()}")
    print("=" * 78)
    return 0 if status == "keep" else 1


if __name__ == "__main__":
    sys.exit(main())
