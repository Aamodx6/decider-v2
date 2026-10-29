# Cell 8 - run the full test battery BEFORE spending any GPU minutes. The kernel's own python
# runs tests/: they need the pinned stack (torch/transformers/peft/datasets) in THIS kernel
# env, which Studio kernels do not ship by default - pip-install them lazily exactly the way
# train.py's --env_check would (same pins from requirements.txt; torch must already be here).
import importlib
import os
import subprocess
import sys

def _need(module):
    try:
        importlib.import_module(module)
        return None
    except Exception as e:  # noqa: BLE001 - any import trouble counts (partial wheels too)
        return f"{type(e).__name__}: {e}"

missing = [m for m in ("torch", "transformers", "peft", "datasets") if (fail := _need(m))]
if missing:
    print("stack incomplete in this kernel env ->", missing)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "-r", str(REPO / "requirements.txt")],
                       cwd=str(REPO), capture_output=True, text=True)
    print(r.stdout[-2000:] + r.stderr[-2000:])
    still_missing = [m for m in "torch transformers peft datasets".split() if _need(m)]
    if still_missing:
        raise SystemExit(f"pip install did not fix the stack - still missing {still_missing}")

# test_core.py is a script with its own main() (run with --device cpu here: the kernel has no GPU).
# `pytest tests` would instead collect its test_* helpers and fail on their `device` argument.
cmd = [sys.executable, "tests/test_core.py", "--device", "cpu"]
r = subprocess.run(cmd, cwd=str(REPO), env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                   capture_output=True, text=True)
print(r.stdout[-4000:] + r.stderr[-1500:])
if r.returncode != 0:
    raise SystemExit("tests failed - do NOT launch (fix locally first; the GPU run just burns money)")
print("tests: PASS")
