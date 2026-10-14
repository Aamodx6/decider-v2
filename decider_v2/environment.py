"""environment.py -- the one place that knows which package versions this code is pinned to.

Every number in LOCAL_VERIFICATION.md and ARCH_REVIEW.md was produced with exactly the versions in
`requirements.txt`.  That matters more than usual here because the project's correctness depends on
a transformers behaviour: the 4-D additive attention mask is returned as-is only from the 5.x line
(ARCH_V2.md, and the `attn_implementation` checks in tests.py).  A container that silently resolves
a different transformers could pass a smoke test and produce meaningless numbers at 12k steps.

So both entry points verify and print before doing any work:

    python -m environment            # print the table, exit 1 on mismatch
    python environment.py --install  # additionally pip-install the pins into this interpreter

`parse_requirements` is a deliberately small parser: it reads `requirements.txt` (the same file the
container installs) rather than hard-coding a second copy of the pins, so the two cannot drift.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")

# The container must provide at least this Python, because numpy 2.3.5 declares >=3.11.
MIN_PYTHON = (3, 11)

# torch is deliberately NOT pinned: the SageMaker container owns the CUDA-matched build, and
# overwriting it is how you end up with a CPU-only torch on a GPU instance. It is checked only for
# presence, and its version is recorded in the log.
UNPINNED = ("torch",)


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true",
                    help="pip-install the pins when they do not match, then re-check")
    ap.add_argument("--no_install", action="store_true",
                    help="never install; just report (used by tests.py)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.quiet:
        ok, _rows = check(pins=parse_requirements(REQUIREMENTS), verbose=False)
        print("env pins OK" if ok else "env pins MISMATCH")
        return 0 if ok else 1
    ok = enforce(allow_install=args.install and not args.no_install)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())