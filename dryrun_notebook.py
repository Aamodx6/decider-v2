"""Execute decider-train/sagemaker_train.ipynb offline with the AWS plumbing stubbed out.

The MODEL code is real: tests/test_core.py, dataset.py prepare, train.py, evaluate.py and the
split-carving all run for real.  Only S3 and the training-job launcher/monitor/metering are
faked (_dryrun/), so this is a genuine end-to-end check of the notebook's control flow, its
SMOKE-mode plumbing and its SageMaker wiring -- with the notebook's own cwd (decider-train/).

    python dryrun_notebook.py                # run every cell
    python dryrun_notebook.py --stop-after N # run only cells 0..N-1 (0-indexed, all kinds)
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
import time

# nbclient captures cell output over ZMQ's iopub socket, which needs add_reader. Windows' default
# Proactor event loop does not implement it, and the symptom is silently EMPTY outputs rather than
# an error, so switch the policy before the kernel starts.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "decider-train")
NB = os.path.join(REPO, "sagemaker_train.ipynb")
DRY = os.path.join(ROOT, "_dryrun")

# The stub is a pip-free fake: importlib.metadata.version("sagemaker") must answer from a real
# Distribution object, so put a synthetic dist-info tree on sys.path for the checker in cell 3.
_SITE = os.path.join(DRY, "_site")
os.makedirs(_SITE, exist_ok=True)
_di = os.path.join(_SITE, "sagemaker-2.247.0.dist-info")
os.makedirs(_di, exist_ok=True)
with open(os.path.join(_di, "METADATA"), "w", encoding="utf-8") as fh:
    fh.write("Metadata-Version: 2.1\nName: sagemaker\nVersion: 2.247.0-dryrun\n")
if os.path.abspath(_SITE) not in [os.path.abspath(p) for p in sys.path]:
    sys.path.insert(0, _SITE)


def _clean():
    """A dry run starts from zero state: wiped runs/state/data and wiped S3 stub tree."""
    for name in ("decider_v2_state_smoke.json", "decider_v2_state.json"):
        p = os.path.join(REPO, name)
        if os.path.exists(p):
            os.remove(p)
    for sub in ("runs", "data_smoke", "data_prepared", "build"):
        p = os.path.join(REPO, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
    s3tree = os.path.join(ROOT, "_dryrun_s3")
    if os.path.isdir(s3tree):
        shutil.rmtree(s3tree)
    logs = os.path.join(DRY, "logs")
    if os.path.isfile(logs):
        os.remove(logs)


def execute_in_process(nb, work):
    """Run the code cells in order in one interpreter, sharing one namespace.

    Same semantics as nbclient minus display handling, and unlike nbclient on Windows it does not
    silently lose cell output to a ZMQ/Proactor event-loop problem.
    """
    os.chdir(work)
    ns = {"__name__": "__main__", "__file__": os.path.join(work, "dryrun_notebook.py")}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source if isinstance(cell.source, str) else "\n".join(cell.source)
        print(f"\n{'#' * 78}\n# CELL {i} ({cell.get('name', '?')}): {src.splitlines()[0]}\n{'#' * 78}",
              flush=True)
        try:
            exec(compile(src, f"cell-{i}", "exec"), ns)
        except SystemExit as e:
            if e.code not in (0, None):
                raise
        except BaseException as e:
            import traceback
            traceback.print_exc()
            raise SystemExit(f"dry run: cell {i} failed ({type(e).__name__}: {e})")
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-after", type=int, default=0,
                    help="run at most the first N cells (counting markdown cells too)")
    ap.add_argument("--keep", action="store_true", help="do not wipe runs/S3 stub state first")
    ap.add_argument("--timeout", type=int, default=60 * 180)
    args = ap.parse_args()

    if DRY not in sys.path:
        sys.path.insert(0, DRY)
    os.environ["PYTHONPATH"] = DRY + os.pathsep + _SITE + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DRYRUN_FAST"] = "1"           # the stub clamps smoke budgets to 2 steps locally
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # benchmark adapters point at the pristine clones in ref/, as a real user would
    for var, sub in (("JEVBENCH_ROOT", "jevbench"), ("SYSONE_BENCH_ROOT", "sysone-bench")):
        path = os.path.join(ROOT, "ref", sub)
        if os.path.isdir(path):
            os.environ[var] = path
            print(f"{var} -> {path}")
        else:
            print(f"{var} NOT SET ({path} missing) -- cell 15 will report NOT RUN for that harness")

    if not args.keep:
        _clean()

    import nbformat
    nb = nbformat.read(NB, as_version=4)
    if args.stop_after:
        nb.cells = nb.cells[:args.stop_after]

    t0 = time.time()
    execute_in_process(nb, REPO)
    print(f"\ndry run finished in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
