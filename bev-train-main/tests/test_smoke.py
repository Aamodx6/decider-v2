"""Smoke: configs/smoke.yaml still overfits 10 questions (runs train.py in a subprocess).

Marked `smoke` (minutes): skip with `uv run pytest -m "not smoke"` for the fast suite.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[1]
NAME = "smoke-pytest"


def test_smoke_overfits_ten_questions():
    subprocess.run([sys.executable, "train.py", "configs/smoke.yaml", "--name", NAME],
                   cwd=ROOT, check=True, timeout=3600)
    runs = sorted((ROOT / "runs").glob(f"{NAME}-*"), key=lambda p: p.stat().st_mtime)
    assert runs, f"no runs/{NAME}-* directory was created"
    summary = json.loads((runs[-1] / "summary.json").read_text())
    print(f"\nsmoke: final_train_loss={summary['final_train_loss']:.4f} "
          f"final_train_accuracy={summary['final_train_accuracy']:.4f}")
    assert summary["final_train_accuracy"] >= 0.9, f"did not overfit 10 questions: {summary}"
    assert summary["final_train_loss"] < 0.5, f"loss did not converge: {summary}"
