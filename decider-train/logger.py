"""Run logging, checkpoint I/O and the environment pin check (old train.py save/log code +
environment.py, merged into one module in the bev-train style).

    logger = Logger(model_dir=..., output_dir=..., checkpoint_dir=..., config=vars(args))    logger.log(step, split="train", loss=0.81, acc=0.71)      # -> metrics.jsonl + one stdout line
    logger.log(step, split="val", _stdout=False, ...)            # -> metrics.jsonl only
    logger.save_checkpoint(network, name, step)               # -> <model_dir>/<name> (+ mirror)
    check_environment()                                       # requirements.txt pins vs installed

The stdout line format is unchanged from train.py's original and is what SageMaker's
metric_definitions regexes parse:

    step=1000 loss=0.8123 grad_norm=0.41 lr_lora=1.9e-04 lr_head=9.5e-04 acc=0.71 sps=42.1
"""

from datetime import datetime
import json
import os
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")

# The container must provide at least this Python, because numpy 2.3.5 declares >=3.11.
MIN_PYTHON = (3, 11)

# torch is deliberately NOT pinned: the SageMaker container owns the CUDA-matched build, and
# overwriting it is how you end up with a CPU-only torch on a GPU instance. It is checked only for
# presence, and its version is recorded in the log.
UNPINNED = ("torch",)


# --------------------------------------------------------------------------- environment pins
def parse_requirements(path=REQUIREMENTS):
    """-> [(name, version)] in file order, skipping comments, blanks and option lines."""
    pins = []
    if not os.path.exists(path):
        return pins
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            # environment markers ("; sys_platform == 'win32'") select platforms; the pin itself
            # is what precedes the ';'.
            line = line.split(";", 1)[0].strip()
            name, sep, version = line.partition("==")
            if sep:
                pins.append((name.strip(), version.strip()))
    return pins


def installed_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _matches(have, want):
    """Exact-match, with a note for local dev builds: torch 2.11.0+cu128 vs a bare 2.11.0 pin."""
    if have == want:
        return True
    return have.split("+")[0] == want


def check(pins=None, verbose=True):
    """Return (ok, rows). `rows` is [(name, want, have, ok)] including torch and Python."""
    pins = pins if pins is not None else parse_requirements()
    rows = []
    ok = True

    py = sys.version_info[:3]
    py_ok = py[:2] >= MIN_PYTHON
    ok &= py_ok
    if verbose:
        print(f"python      {'.'.join(map(str, py)):12s} "
              f"{'OK' if py_ok else 'TOO OLD (need >= %d.%d for numpy 2.3.5)' % MIN_PYTHON}")

    for name in UNPINNED:
        have = installed_version(name)
        rows.append((name, "(container)", have, have is not None))
        if have is None and verbose:
            print(f"{name:11s} MISSING -- the container must provide it (never pip-install torch)")

    for name, want in pins:
        have = installed_version(name)
        good = have is not None and _matches(have, want)
        ok &= good
        rows.append((name, want, have, good))
        if verbose:
            mark = "OK" if good else ("MISSING" if have is None else "MISMATCH")
            print(f"{name:11s} want {want:12s} have {str(have):14s} {mark}")
    return ok, rows


def install_pins(requirements=REQUIREMENTS):
    """pip-install the pinned set into the current interpreter. Returns the subprocess exit code."""
    if not os.path.exists(requirements):
        print(f"env: no {requirements} to install from")
        return 1
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-r", requirements]
    print("env: " + " ".join(cmd))
    return subprocess.call(cmd)


def enforce(requirements=REQUIREMENTS, allow_install=True, prefix="ENV"):
    """Print the resolved versions; on mismatch, optionally install the pins and re-check; then
    return True/False. Never raises, so callers can decide how loudly to fail."""
    print("=" * 78)
    print(f"{prefix}: resolved package versions (pins from {os.path.basename(requirements)})")
    print("=" * 78)
    ok, _rows = check(pins=parse_requirements(requirements))
    if ok:
        print(f"{prefix}: all pins satisfied")
        return True

    print(f"\n{prefix}: pins NOT satisfied.")
    if not allow_install:
        print(f"{prefix}: refusing to install automatically (--no_install). Aborting.")
        return False

    print(f"{prefix}: installing the pinned set into {sys.executable} ...")
    rc = install_pins(requirements)
    if rc != 0:
        print(f"{prefix}: pip install FAILED (exit {rc}). Aborting.")
        return False
    print(f"\n{prefix}: re-checking after install")
    ok, _rows = check(pins=parse_requirements(requirements))
    print(f"{prefix}: {'all pins satisfied' if ok else 'STILL MISMATCHED after install'}")
    return ok


def check_environment():
    """The train.py entry point: enforce() with the module's default policy. Kept as its own name so
    run_preflight reads the same as the old `environment.enforce`."""
    return enforce(prefix="ENV")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true",
                    help="pip-install the pins when they do not match, then re-check")
    ap.add_argument("--no_install", action="store_true",
                    help="never install; just report (used by the test suite)")
    ap.add_argument("--quiet", action="store_true")
    _args = ap.parse_args()
    if _args.quiet:
        ok, _rows = check(pins=parse_requirements(REQUIREMENTS), verbose=False)
        print("env pins OK" if ok else "env pins MISMATCH")
        raise SystemExit(0 if ok else 1)
    ok = enforce(allow_install=_args.install and not _args.no_install)
    raise SystemExit(0 if ok else 1)


# --------------------------------------------------------------------------- Logger
class Logger:
    """metrics.jsonl + checkpoint directories, split out of train.py so the loop stays readable.

    Every run writes metrics.jsonl (one JSON object per line: {step, split?, **metrics}) next to a
    summary.json, and checkpoint folders named `latest` / `best` / `final`.
    """

    def __init__(self, model_dir="./runs/model", output_dir="./runs/output", checkpoint_dir="",
                 config=None, main_rank=True):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.config = dict(config or {})
        self.main_rank = main_rank
        self.start_time = time.time()
        self.step = 0
        self.best_acc = -1.0
        if self.main_rank:
            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "args.json"), "w") as f:
                json.dump(self.config, f, indent=2)
        self._metrics_path = os.path.join(self.output_dir, "metrics.jsonl") if self.main_rank else None

    def log(self, step, split=None, _stdout=True, **metrics):
        """One JSON line to metrics.jsonl + (unless _stdout=False) one regex-friendly line to stdout.
        split=None means a train-step record (the old train.py logged those without a split key)."""
        if not self.main_rank:
            return
        record = {"step": step}
        if split is not None:
            record["split"] = split
        record.update(metrics)
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if _stdout:
            pieces = [f"step={step}"]
            if split is not None and split != "train":
                pieces.append(f"split={split}")
            pieces.extend(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in metrics.items())
            print(" ".join(pieces), flush=True)


    def save_checkpoint(self, network, name, step, meta=None, merge=False, optim=None,
                        scheduler=None, best_acc=None):
        """Write <model_dir>/<name>/ via network.save_checkpoint, plus optim.pt, and mirror both into
        --checkpoint_dir when it differs.

        merge=True folds the LoRA adapter into the backbone. The merge runs on a deepcopy inside
        network.save_checkpoint, so the live training model is never mutated in place.
        """
        if not self.main_rank:
            return
        # meta.json keeps the schema network.load_checkpoint expects: top-level "args" next to the
        # identity fields, exactly like the old train.py's save().
        m = {"args": dict(self.config or {}), "step": step}
        m.update(meta or {})
        m["name"] = name
        m["merge"] = merge
        # 'latest' is the auto-resume source and must round-trip the fp32 training weights; bf16
        # there would put every resumed run ~1e-4 away from the uninterrupted one. 'best'/'final'
        # are evaluation/deployment artifacts and stay bf16.
        lossless = name == "latest"
        m["fp32_backbone"] = lossless
        from network import save_checkpoint as _save
        _save(network, os.path.join(self.model_dir, name), m, merge=merge,
              fp32_backbone=lossless)
        state = {"optim": optim.state_dict() if optim is not None else None,
                 "sched": scheduler.state_dict() if scheduler is not None else None,
                 "step": step, "best_acc": self.best_acc if best_acc is None else best_acc}
        torch.save(state, os.path.join(self.model_dir, name, "optim.pt"))
        if self.checkpoint_dir and os.path.abspath(self.checkpoint_dir) != os.path.abspath(self.model_dir):
            os.makedirs(os.path.join(self.checkpoint_dir, name), exist_ok=True)
            _save(network, os.path.join(self.checkpoint_dir, name), m, merge=merge,
                  fp32_backbone=lossless)
            torch.save(state, os.path.join(self.checkpoint_dir, name, "optim.pt"))

    def write_summary(self, summary):
        if not self.main_rank:
            return
        summary.setdefault("final_step", self.step)
        summary["duration_sec"] = round(time.time() - self.start_time, 1)
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("SUMMARY " + json.dumps(summary), flush=True)
