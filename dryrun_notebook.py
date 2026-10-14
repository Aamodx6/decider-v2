"""Execute decider_v2_sagemaker.ipynb offline with the AWS plumbing stubbed out.

The MODEL code is real: tests.py, prepare.py, train.py, evaluate.py and the split-carving all run
for real.  Only S3 and the training-job launcher are faked (_dryrun/), so this is a genuine
end-to-end check of the notebook's control flow, its SMOKE-mode plumbing and its SageMaker wiring.

    python dryrun_notebook.py            # run every cell
    python dryrun_notebook.py --stop-after 12
"""
import argparse
import asyncio
import os
import shutil
import sys
import time

# nbclient captures cell output over ZMQ's iopub socket, which needs add_reader.  Windows' default
# Proactor event loop does not implement it, and the symptom is silently EMPTY outputs rather than
# an error, so switch the policy before the kernel starts.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT = os.path.dirname(os.path.abspath(__file__))
DRY = os.path.join(ROOT, "_dryrun")
WORK = os.path.join(ROOT, "decider_v2_work")
NB = os.path.join(ROOT, "decider_v2_sagemaker.ipynb")
OUT = os.path.join(ROOT, "_dryrun_out.ipynb")

# The notebook is executed with CWD = ROOT, exactly as it would be if you opened it from here, so
# it finds ./decider_v2_src and creates ./decider_v2_work.
src_link = os.path.join(ROOT, "decider_v2_src")
if os.path.exists(src_link) and not os.path.isdir(src_link):
    os.remove(src_link)
if os.path.isdir(src_link):
    shutil.rmtree(src_link)
# bench/ must survive the copy: cell 12 runs the JevBench / sysone-bench adapters out of SRC_DIR.
shutil.copytree(os.path.join(ROOT, "decider_v2"), src_link,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_*.txt", "*.json"),
                dirs_exist_ok=True)
os.makedirs(WORK, exist_ok=True)

# Point the benchmark adapters at the pristine clones in ref/, which is what a real user would do
# with JEVBENCH_ROOT / SYSONE_BENCH_ROOT. This is what lets cell 12 produce REAL numbers offline.
for var, sub in (("JEVBENCH_ROOT", "jevbench"), ("SYSONE_BENCH_ROOT", "sysone-bench")):
    path = os.path.join(ROOT, "ref", sub)
    if os.path.isdir(path):
        os.environ[var] = path
        print(f"{var} -> {path}")
    else:
        print(f"{var} NOT SET ({path} missing) -- cell 12 will report NOT RUN for that harness")

# reuse the HF cache that recon already populated so the dry run is offline and fast
cache_src = r"C:\Users\aamod\AppData\Local\Temp\opencode\hf_cache"
cache_dst = os.path.join(WORK, "hf_cache")
os.makedirs(cache_dst, exist_ok=True)
if os.path.isdir(cache_src) and os.path.abspath(cache_src) != os.path.abspath(cache_dst):
    for entry in os.listdir(cache_src):
        s, d = os.path.join(cache_src, entry), os.path.join(cache_dst, entry)
        if os.path.exists(d):
            continue
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    print(f"reused HF cache: {cache_src} -> {cache_dst}")


def _src(cell):
    """nbformat stores source as a string; freshly built notebooks use a list of lines."""
    return cell.source if isinstance(cell.source, str) else "\n".join(cell.source)


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
        src = _src(cell)
        print(f"\n{'#' * 78}\n# CELL {i}: {src.splitlines()[0]}\n{'#' * 78}", flush=True)
        try:
            exec(compile(src, f"cell-{i}", "exec"), ns)
        except SystemExit:
            raise
        except BaseException as e:
            import traceback
            traceback.print_exc()
            raise SystemExit(f"dry run: cell {i} failed ({type(e).__name__}: {e})")
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=60 * 180)
    ap.add_argument("--keep", action="store_true", help="do not clear previous outputs first")
    ap.add_argument("--nbclient", action="store_true")
    args = ap.parse_args()

    os.environ["PYTHONPATH"] = DRY + os.pathsep + os.environ.get("PYTHONPATH", "")
    if DRY not in sys.path:
        sys.path.insert(0, DRY)   # PYTHONPATH is read at interpreter start; this process is already up
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if not args.keep:
        for p in (OUT, os.path.join(WORK, "decider_v2_state.json")):
            if os.path.exists(p):
                os.remove(p)
        for sub in ("data", "eval", "downloads"):
            p = os.path.join(WORK, sub)
            if os.path.isdir(p):
                shutil.rmtree(p)

    import nbformat
    nb = nbformat.read(NB, as_version=4)
    if args.stop_after:
        nb.cells = nb.cells[:args.stop_after]

    t0 = time.time()
    if args.nbclient:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                resources={"metadata": {"path": ROOT}}, allow_errors=False)
        client.execute()
        nbformat.write(nb, OUT)
    else:
        execute_in_process(nb, ROOT)
    print(f"\ndry run finished in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())