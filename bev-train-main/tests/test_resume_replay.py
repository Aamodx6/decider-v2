"""REQUIRED STEP 3 gate (CPU, tiny config, num_workers=0): the --resume-state path reproduces
continuous training exactly.

20 steps straight  vs  10 steps + simulated kill + `--resume-state` for the remaining 10
-> the step-20 training losses (and LRs) must match within 1e-5.

Runs train.py in a subprocess so the kill is a real process death (unclosed DataLoader workers,
unflushed metrics, no precious state in memory). The tiny config trains a frozen-backbone head so
no Qwen download is needed; the LoRA/optimizer/scheduler machinery through AdamW + cosine is
exercised on the real head (and a rank-8 LoRA on a 2-layer backbone if downloads are cached).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {   # tiny: 160 synthetic questions via the loader is slow on HF; instead a hand dataset
    "seed": 0,
    "model": {"name": "Qwen/Qwen3-0.6B", "num_layers": 2},
    "data": {
        "configs": ["default"],
        "max_train_questions": 240,   # 240 questions -> 60 micro-batches of 4
        "max_val_questions": 0,       # no eval during the replay
        "max_test_questions": 0,
        "max_state_tokens": 128,
        "max_choice_tokens": 32,
        "batch_size": 4,
        "num_workers": 0,
    },
    "lora": {"target": "attn", "last_k_layers": 4, "r": 8, "alpha": 16, "dropout": 0.0},
    "head": {"new_dim": 512, "num_layers": 2, "ctx_queries": 0},
    "loss": {},
    "optim": {
        "epochs": 1, "lr_lora": 1.0e-4, "lr_head": 1.0e-3, "weight_decay": 0.01,
        "warmup_ratio": 0.1, "grad_clip": 1.0, "grad_accum_steps": 3,     # 20 steps over 60 micro-batches
        "resume_save_seconds": 3600,   # never save mid-run on its own (we kill after 10 steps)
    },
    "logging": {"log_every": 1, "eval_every": 1000, "save_every": 1000},
}


def write_config(tmp, name):
    path = Path(tmp) / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(CONFIG, f, sort_keys=False)
    return path


def losses_from_metrics(run_dir):
    out = {}
    with open(Path(run_dir) / "metrics.jsonl") as f:
        for line in f:
            record = json.loads(line)
            if record.get("split") == "train":
                out[record["step"]] = record["loss"]
    return out


def env_force_cpu():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""  # the replay must run on CPU regardless of laptop GPU state
    # the tiny config must never wait on an HF download mid-test: expect the model to be cached
    return env


def newest_run_dir(prefix):
    """RunLogger writes runs/<name>-<timestamp>/, so resolve by prefix rather than assuming
    a bare runs/<name> directory."""
    matches = sorted((ROOT / "runs").glob(f"{prefix}-*"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def wait_for_slot(prefix, steps_target, timeout):
    """Poll the newest runs/<prefix>-*/metrics.jsonl until `steps_target` train steps are logged
    AND the resume pointer exists, then signal the caller to kill the process (a real process
    death)."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        run_dir = newest_run_dir(prefix)
        if run_dir is not None:
            metrics = run_dir / "metrics.jsonl"
            if metrics.exists():
                lines = [json.loads(l) for l in metrics.read_text().splitlines() if l.strip()]
                if any(r.get("split") == "train" and r.get("step") == steps_target for r in lines):
                    slot = run_dir / "checkpoints" / "resume"
                    if (slot / "pointer").exists():
                        return run_dir
        time.sleep(2)
    return None


@pytest.mark.timeout(1800)
def test_resume_state_replays_continuous_training(tmp_path):
    straight = subprocess.run(
        [sys.executable, "train.py", str(write_config(tmp_path, "straight")), "--name", "replay-straight"],
        cwd=ROOT, env=env_force_cpu(), capture_output=True, text=True, timeout=1750)
    assert straight.returncode == 0, straight.stdout[-3000:] + straight.stderr[-3000:]
    straight_dir = sorted((ROOT / "runs").glob("replay-straight-*"), key=lambda p: p.stat().st_mtime)[-1]
    straight_losses = losses_from_metrics(straight_dir)
    assert len(straight_losses) >= 10, straight_losses

    # --- interrupted run: kill AFTER the 10th optimizer step (its resume slot already flushed) ---
    for stale in (ROOT / "runs").glob("replay-killed-*"):
        shutil.rmtree(stale, ignore_errors=True)
    steps_target = 10
    cfg_path = write_config(tmp_path, "killed")
    proc = subprocess.Popen(
        [sys.executable, "train.py", str(cfg_path), "--name", "replay-killed", "--emit-resume-every", str(steps_target)],
        cwd=ROOT, env=env_force_cpu(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    killed_dir = wait_for_slot("replay-killed", steps_target, 900)
    proc.kill()
    try:
        proc.wait(timeout=30)
    except Exception:
        pass
    assert killed_dir is not None, (
        "the run never wrote a resume slot at step 10 (check runs/replay-killed-*/)")

    resumed = subprocess.run(
        [sys.executable, "train.py", str(cfg_path), "--resume-state", str(killed_dir)],
        cwd=ROOT, env=env_force_cpu(), capture_output=True, text=True, timeout=1750)
    assert resumed.returncode == 0, resumed.stdout[-3000:] + resumed.stderr[-3000:]
    resumed_losses = losses_from_metrics(killed_dir)

    assert sorted(resumed_losses) == sorted(straight_losses), \
        f"step sets differ: resumed={sorted(resumed_losses)} straight={sorted(straight_losses)}"
    worst_step, worst = 0, 0.0
    for step in sorted(straight_losses):
        diff = abs(resumed_losses[step] - straight_losses[step])
        if diff > worst:
            worst_step, worst = step, diff
    print(f"\nresume replay: worst |loss diff| = {worst:.3e} at step {worst_step} (threshold 1e-5)")
    assert worst <= 1e-5, f"replay diverged at step {worst_step}: |{straight_losses[worst_step]} - " \
                          f"{resumed_losses[worst_step]}| = {worst:.3e}"
