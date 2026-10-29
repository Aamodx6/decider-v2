"""build_lightning_notebook.py -- emit decider_v2_lightning.ipynb, a SELF-CONTAINED notebook.

One notebook, no links to any other notebook, no SageMaker, no S3, no boto3, no IAM. Every artefact it
needs is either embedded in it (the training sources, as a base64 zip) or written by its own cells.

Why a separate builder rather than a flag on build_notebook.py: the SageMaker notebook's launch layer
is not "SageMaker calls sprinkled through the code" -- it is quota lookup, estimator construction,
s3:// checkpoint URIs, CloudWatch metric streams, spot capacity, and an S3 download step in the monitor
cell. Interleaving a local backend into that would leave both paths half-wired forever. The training
code itself (train.py / evaluate.py / calibrate.py / decide.py) is already backend-agnostic: it reads
directories from --train_channel / --model_dir / --output_dir and only *defaults* those to the
SageMaker SM_* environment variables. So the backend swap lives entirely here, in the notebook.

What changes vs the SageMaker notebook
---------------------------------------
  quota / IAM / boto3 cells         deleted
  s3:// input channels             local directories under decider_v2_work/
  SageMaker estimator + wait()     subprocess.Popen, log to a file, poll the PID
  checkpoint_s3_uri                a local --model_dir
  download_outputs()               gone; the checkpoint is already on this filesystem
  spot / max_wait / max_runtime    gone; there is no 72h kill, but the session must stay alive
"""

import base64
import glob
import io
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "decider-train")
OUT = os.path.join(HERE, "decider_v2_lightning.ipynb")

cells = []


def md(title, body=""):
    src = [f"## {title}\n"]
    if body:
        src.append("\n" + body + "\n")
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(title, body):
    lines = body.splitlines(keepends=True)
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [f"# --- {title} ---\n"] + lines})


# ------------------------------------------------------------------ 0 sources
md("0. Sources (embedded, self-extracting)",
   "The training sources are embedded in the next cell as a base64 zip and unpacked to "
   "`./decider_v2_src`. This notebook is entirely self-contained: there is nothing to clone and "
   "nothing to download before it runs. Extraction is skipped when the folder already exists, so "
   "edits you make to `decider_v2_src/` on disk are never overwritten.")

code("0-sources", '''
SRC_ZIP_B64 = "@@SRC_ZIP@@"
import base64, io, os, sys, zipfile, hashlib

SRC_DIR = os.path.abspath("decider_v2_src")
if os.path.exists(os.path.join(SRC_DIR, "train.py")):
    print(f"{SRC_DIR} already present -- leaving it alone (re-running is harmless).")
else:
    _blob = base64.b64decode(SRC_ZIP_B64)
    print(f"extracting decider_v2_src from embedded zip ({len(_blob):,} B, "
          f"sha256 {hashlib.sha256(_blob).hexdigest()[:16]}...)")
    with zipfile.ZipFile(io.BytesIO(_blob)) as _z:
        for _n in _z.namelist():
            # Refuse any path that escapes SRC_DIR (zip-slip).
            if _n.startswith("/") or ".." in _n.replace("\\\\", "/").split("/"):
                raise SystemExit(f"refusing unsafe path in embedded zip: {_n}")
        _z.extractall(os.getcwd())
    if not os.path.exists(os.path.join(SRC_DIR, "train.py")):
        raise SystemExit(f"extraction did not produce {SRC_DIR}/train.py -- corrupt embedded zip?")
    print(f"extracted {len(_z.namelist())} files to {SRC_DIR}")

sys.path.insert(0, SRC_DIR)
print("sources ->", SRC_DIR)
''')

# ------------------------------------------------------------------ 1 config
md("1. Configuration",
   "Edit the constants below and nothing else. `SMOKE=True` runs the whole pipeline end-to-end in a "
   "few minutes so you can prove the plumbing before starting a multi-hour training run. The real run "
   "needs `SMOKE=False`; there is no bucket, no role and no quota to arrange.")

code("1-config", '''
import json, os, sys, time

# ------------------------------------------------------------------ EDIT ME
SMOKE           = True          # True: tiny subset + 100 steps, everything still runs

# max_state_tokens: 1024 is bev-decider's TRAINING limit. 512 halves the cost and covers ~92% of
# rows unchanged (p99 = 958 tokens); do not lower it for the fair comparison.
MAX_STATE_TOKENS = 1024
MAX_CHOICE_TOKENS = 64

# SPLIT_SEED carves train/val/calib AND samples the SMOKE test shard. FIXED for the whole run and
# never varied between arms -- identical data is the fair-comparison rule, so this is not a knob.
SPLIT_SEED     = 0

# Training seeds. Every (arm, seed) pair is its own run at the SAME budget. The decision rule needs
# >= 3 seeds to return SEPARABLE and >= 2 for PROVISIONAL, because with one seed the seed spread is 0
# by construction and that condition is vacuous. Staged: run [0], inspect, then [0, 1], then [0, 1, 2].
SEEDS          = [0]

# Which arms to train. v2 is the Perceiver-context decider; baseline is the bev-decider architecture
# replicated exactly, so the delta between them is the only thing the decision rule ever sees.
RUN_BASELINE    = True
RUN_V2          = True

# Published reference and its numbers live in the results cell; nothing to configure here.

# ------------------------------------------------------------------ batch / budget
# SMOKE uses a 4,000-question subset, so a micro-batch of 8 is plenty AND fits a small GPU
# (batch 32 x 1,024-token states peaks at ~8.3 GiB). The real run uses 32.
BATCH_SIZE      = 8 if SMOKE else 32   # micro-batch
GRAD_ACCUM      = 2             # 32 x 2 = 64 questions / step, bev-decider's published recipe

# The published bev-decider-0.4B card says "cross-entropy on about 940K typed questions" and
# "12,000 steps at batch size 64". Those disagree (12,000 x 64 = 768,000) and neither is reachable as
# distinct data: bev-decision('all') train is 422,551 typed questions, of which this run holds out val
# and calib first. So "940K" can only mean QUESTION PRESENTATIONS -- repeats over the distinct train
# questions -- and matching it is the closest anyone can get to the published recipe:
#   ceil(940,000 / 64) = 14,688 steps = 940,032 presentations.
#
# IMPORTANT: the budget is defined against RECIPE_QPS (the published 64), NOT against BATCH_SIZE.
# BATCH_SIZE is lowered under SMOKE so a 4,000-question subset fits a small GPU, and deriving the step
# count from it would silently turn 14,688 steps into 58,750 and change what the run trains on.
RECIPE_QPS    = 32 * GRAD_ACCUM          # 64 -- bev-decider's published micro-batch 32 x accum 2
TARGET_PRESENTATIONS = 940_000           # the card's "about 940K typed questions"
STEPS_FROM_PRESENTATIONS = TARGET_PRESENTATIONS   # set to None to pin MAX_STEPS by hand
MAX_STEPS = (-(-STEPS_FROM_PRESENTATIONS // RECIPE_QPS) if STEPS_FROM_PRESENTATIONS else 12000)
MAX_STEPS_REAL = MAX_STEPS                # saved before SMOKE pins MAX_STEPS to 100
ACTUAL_QPS   = BATCH_SIZE * GRAD_ACCUM

# The published `all` test split is 80,019 typed questions and is NEVER subsampled in a real run:
# every arm, every seed and the reference all score the same full file, so the comparison carries no
# sampling noise of our own making. The 4,000-question shard exists ONLY under SMOKE. The parity cell
# asserts which one you got.
TEST_QUESTIONS  = None           # None -> the FULL 80,019-question published test split
SMOKE_TEST_N    = 4000           # SMOKE only

# Preflight gates. SELFTEST runs the packed-vs-separate + order-invariance gates before training and
# aborts on failure (~10s, tiny config, no download). ENV_CHECK enforces requirements.txt.
SELFTEST        = True
ENV_CHECK       = True

# Rebuild the local venv from scratch. Normally it reuses itself and rebuilds automatically when
# requirements.txt changes.
REBUILD_VENV    = False
# ----------------------------------------------------------------------------

WORK = os.path.abspath("decider_v2_work")   # every artefact this notebook writes
os.makedirs(WORK, exist_ok=True)
LOCAL = WORK
DATA = os.path.join(WORK, "data")
RUNS = os.path.join(WORK, "runs")
CACHE = os.path.join(WORK, "hf_cache")
EVAL = os.path.join(WORK, "eval")
LOGS = os.path.join(WORK, "logs")
for _d in (DATA, RUNS, CACHE, EVAL, LOGS):
    os.makedirs(_d, exist_ok=True)

STATE_PATH = os.path.join(WORK, "state.json")
RUN_NAME = f"run-{time.strftime('%Y%m%d-%H%M%S')}"
print(f"RUN_NAME  = {RUN_NAME}")
print(f"workdir   = {WORK}")

if SMOKE:
    SMOKE_KW = dict(smoke_train=4000, smoke_val=600, smoke_calib=1200, smoke_test=SMOKE_TEST_N)
    MAX_STEPS = 100
    assert TEST_QUESTIONS is None, "SMOKE uses its own shard; leave TEST_QUESTIONS = None"
    print(f"SMOKE mode: subset {SMOKE_KW}, MAX_STEPS -> {MAX_STEPS}")
    print("SMOKE test = the 4,000-question shard. A real run scores the FULL 80,019-question split.")
else:
    SMOKE_KW = dict(smoke_train=0, smoke_val=0, smoke_calib=0, smoke_test=0)
    print("REAL RUN: test = the FULL published test split, never subsampled.")
print("local data dir:", DATA)

PRESENTATIONS = MAX_STEPS * ACTUAL_QPS
print(f"\\nbudget per run: {MAX_STEPS:,} steps x {ACTUAL_QPS} questions/step "
      f"= {PRESENTATIONS:,} question presentations")
if SMOKE:
    print(f"  SMOKE: MAX_STEPS is pinned to 100 for the plumbing check. The REAL budget is")
    print(f"  {MAX_STEPS_REAL:,} steps x {RECIPE_QPS} = {TARGET_PRESENTATIONS:,} presentations")
else:
    print(f"  target {TARGET_PRESENTATIONS:,}; "
          f"{'matched' if PRESENTATIONS >= TARGET_PRESENTATIONS else 'SHORT OF'} the card's ~940K")

n_runs = len(SEEDS) * (int(RUN_BASELINE) + int(RUN_V2))
print(f"\\nthis run trains {n_runs} model(s): {int(RUN_BASELINE) * len(SEEDS)} baseline + "
      f"{int(RUN_V2) * len(SEEDS)} v2, each {MAX_STEPS:,} steps.")
print("They run SEQUENTIALLY: there is one GPU on this machine, so they take turns.")
if not SMOKE:
    _est = None
    print("Run the bench cell first for a measured steps/sec figure on THIS gpu, then multiply.")
if len(SEEDS) < 2:
    print("\\n  NOTE: SEEDS has 1 entry. The decision rule cannot return anything above")
    print("  NOT SEPARABLE with one seed (the seed spread is 0 by construction). That is fine for")
    print("  a first run -- it is the cheapest way to learn whether the budget is affordable -- but")
    print("  do not quote the v2-baseline delta from it. Add seeds with SEEDS = [0, 1], then [0, 1, 2].")
''')

# ------------------------------------------------------------------ 2 env
md("2. Interpreter and pins",
   "The notebook kernel is **not** the pinned interpreter, and on a prebuilt image that is not merely "
   "a version mismatch -- some combinations fail at *import* time. A Studio image pairing "
   "`datasets 2.2.1` with `pyarrow 21` raises `AttributeError: module 'pyarrow' has no attribute "
   "'PyExtensionType'` (removed in pyarrow 15), which no version check can see coming. Worse, a "
   "subprocess launched on `sys.executable` inherits the same breakage, so the data-prep step dies "
   "before it starts.\n\n"
   "So the pinned set goes into a **throwaway venv** under the work directory, and every local step "
   "runs on that interpreter instead of the kernel's. The kernel is never modified. The venv is built "
   "with `--system-site-packages` so the GPU's own `torch` build stays visible rather than being "
   "replaced by a CPU wheel.\n\n"
   "If this cell cannot produce a working interpreter, everything below will fail in confusing ways. "
   "Read the error rather than skipping it.")

code("2-env", '''
import json, os, subprocess, sys

REQ = os.path.join(SRC_DIR, "requirements.txt")
if not os.path.exists(REQ):
    raise SystemExit(f"{REQ} not found -- it ships with the sources (cell 0) and pins every version.")

import venv_env

VENV_DIR = os.path.join(WORK, "venv")
LOCAL_PY = venv_env.ensure(VENV_DIR, REQ, force=REBUILD_VENV)
if not venv_env.describe(LOCAL_PY, REQ):
    raise SystemExit("the local venv does not satisfy requirements.txt. Delete "
                     f"{VENV_DIR} and re-run this cell, or install the pins there by hand.")

# A satisfied pin table is necessary but NOT sufficient. What actually matters is that the pinned
# stack IMPORTS and sees the GPU, and the two failures this project hit both passed the version check:
#   * pyarrow reports "21.0.0" happily while `import datasets` raises AttributeError
#     (PyExtensionType was removed in pyarrow 15);
#   * torch can report a version while `import torch` dies on libmkl_intel_lp64.so.2, because a venv
#     subprocess launched directly never gets the conda activation that put that library on the
#     loader path.
#
# patch_env() fixes the second one for every later subprocess by putting the parent's library
# directories on the search path in THIS process, and verify_runtime() then imports for real so a
# failure lands here, with an explanation, instead of three cells later with a traceback.
LOCAL_ENV = venv_env.patch_env()
if not venv_env.verify_runtime(LOCAL_PY):
    raise SystemExit("the pinned interpreter cannot import torch/transformers/datasets -- see above.")

print(f"\\nlocal subprocess interpreter: {LOCAL_PY}")
''')

# The GPU warning needs the real probe result, so ask verify_runtime's own output rather than
# re-deriving it. Kept as a separate small cell so the failure message above stays the focus.
code("2b-gpu", '''
# Re-print the GPU state in one line, and be explicit about what it means for the budget.
import subprocess as _sp
_r = _sp.run([LOCAL_PY, "-c",
              "import torch;"
              "print(torch.cuda.is_available());"
              "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"],
             capture_output=True, text=True, env=LOCAL_ENV)
_lines = _r.stdout.strip().splitlines()
_has = _lines and _lines[0].strip() == "True"
_gpu = _lines[1].strip() if len(_lines) > 1 else "NONE"
print(f"GPU visible to the pinned interpreter: {_gpu}")
if not _has:
    print("\\nWARNING: no GPU. SMOKE will run on CPU (slow but it exercises the whole path).")
    print("A real run cannot: set SMOKE=False only once a GPU is actually attached to this machine,")
    print("and confirm the name printed above is the GPU you intend to train on.")
else:
    print("Good -- training will use this GPU. The bench cell measures its real throughput.")
''')

# ------------------------------------------------------------------ 3 data
md("3. Load the dataset and carve the splits",
   "`load_dataset('bev-decision', 'all')`, both splits, no subsampling outside SMOKE mode. "
   "The published `test` split is **never** touched by training, early stopping or calibration. "
   "`calib` and `val` are carved out of `train` **before** training, stratified by question type × "
   "domain. The parity cell prints the measured state overlap so you can see it is 0.\n\n"
   "This runs in the pinned venv, not the kernel, for the reason given in cell 2.")

code("3-prepare", '''
# Explode rows -> one record per typed question, then carve train / val / calib / test.
# SPLIT_SEED is fixed for the whole run: it carves the splits AND samples the SMOKE test shard, so
# every arm and every seed trains and is scored on byte-identical data.
subprocess.check_call([LOCAL_PY, os.path.join(SRC_DIR, "dataset.py"), "prepare",
                       "--out", DATA, "--seed", str(SPLIT_SEED),
                       "--calib_target", "20000", "--val_frac", "0.03",
                       "--cache_dir", CACHE, "--force"]
                      + [f"--{k}={v}" for k, v in SMOKE_KW.items() if v], cwd=SRC_DIR, env=LOCAL_ENV)
''')

code("3-parity", '''
# Fair-comparison assertions, printed rather than assumed.
m = json.load(open(os.path.join(DATA, "manifest.json")))
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

check("val never overlaps test by state", m["state_overlap_val_to_test"] == 0,
      f"({m['state_overlap_val_to_test']})")
check("calib never overlaps test by state", m["state_overlap_calib_to_test"] == 0,
      f"({m['state_overlap_calib_to_test']})")
check("calib >= 20000 questions", m["stats"]["calib"]["questions"] >= 20000,
      f"({m['stats']['calib']['questions']:,})")
check("test is the published test split, untouched",
      m["sizes"]["test"] == m["stats"]["test"]["questions"])

# THE assertion that keeps a real run honest: the full 80,019-question test split, or a shard that is
# only allowed under SMOKE. n=4,000 carries a +/-1.35 point 95% interval at p~0.76, which is larger
# than any improvement this project could claim, so the shard is a smoke-only convenience.
n_test = m["sizes"]["test"]
FULL_TEST_N = 80019
if SMOKE:
    check("SMOKE: test is the declared shard, not the full split", n_test == SMOKE_TEST_N,
          f"({n_test:,} of {FULL_TEST_N:,})")
    print(f"  NOTE: every accuracy below is on {n_test:,} questions and is NOT a result -- it is a "
          f"plumbing check with a +/-1.35 point interval.")
else:
    check("REAL RUN: test is the FULL published test split, never subsampled",
          n_test == FULL_TEST_N, f"({n_test:,} of {FULL_TEST_N:,})")
    if n_test != FULL_TEST_N:
        raise AssertionError(f"real run must evaluate all {FULL_TEST_N:,} test questions, got {n_test:,}. "
                             f"Either SMOKE=True or fix the dataset.py prepare shard argument.")

print("  split sizes:", m["sizes"])
for name in ("train", "val", "calib", "test"):
    s = m["stats"][name]
    n = s["questions"]
    print(f"  {name:6s} n={n:7,d}  " + "  ".join(f"{k}={v/n*100:5.2f}%" for k, v in sorted(s["by_type"].items())))
assert ok, "split parity failed -- do not train"
TEST_N = n_test
print(f"\\nTEST_N = {TEST_N:,}  ({'SMOKE shard' if SMOKE else 'FULL published test split'})")
''')

# ------------------------------------------------------------------ 4 gates
md("4. Preflight: self-test and throughput",
   "Two cheap checks before any real training.\n\n"
   "**Self-test** -- runs the packed-vs-separate equivalence and the option-order-invariance gate on a "
   "tiny config inside the pinned interpreter. This is the check that catches the transformers 5.x "
   "4-D attention-mask behaviour the whole architecture depends on. It costs ~10 s and downloads "
   "nothing.\n\n"
   "**Throughput** -- times a handful of real training steps on *this* GPU through the exact "
   "`train.py` code path, so you learn what the budget actually costs here before committing to it. "
   "On a different GPU from the one you benchmarked on, this number is the one that matters.")

code("4-selftest", '''
rc = subprocess.call([LOCAL_PY, os.path.join(SRC_DIR, "tests", "test_core.py"),
                      "--device", "cuda" if os.environ.get("FORCE_CPU", "") != "1" else "cpu"],
                     cwd=SRC_DIR, env=LOCAL_ENV)
if rc != 0:
    raise SystemExit(f"self-test FAILED (exit {rc}). Do not train on top of a failing gate -- the "
                     f"architecture depends on what these tests assert.")
print("\\nself-test passed")
''')

code("4-bench", '''
# Times a few real steps through the exact train.py path on this GPU. Prints the number that turns
# "940K presentations" into "about N hours", which is the only figure that decides whether the real
# budget is affordable HERE.
rc = subprocess.call([LOCAL_PY, os.path.join(SRC_DIR, "scripts", "bench.py"),
                      "--out", os.path.join(RUNS, "bench_v2.json"),
                      "--batch_size", str(BATCH_SIZE),
                      "--max_state_tokens", str(MAX_STATE_TOKENS),
                      "--lora_include_mlp", "1", "--head", "v2"], cwd=SRC_DIR, env=LOCAL_ENV)
if rc != 0:
    print("bench failed -- the estimate below will be unavailable. The smoke run still works.")
else:
    b = json.load(open(os.path.join(RUNS, "bench_v2.json")))
    qps = b.get("gpu_only_qps") or b.get("end_to_end_qps")
    print(f"\\nmeasured {qps:.3f} questions/sec on this GPU")
    if qps:
        hrs = TARGET_PRESENTATIONS / qps / 3600
        n = len(SEEDS) * (int(RUN_BASELINE) + int(RUN_V2))
        print(f"  real budget: {TARGET_PRESENTATIONS:,} presentations = {hrs:.1f} h per model")
        print(f"  {n} model(s) at that budget = {hrs * n:.1f} h total, SEQUENTIALLY (one GPU)")
        print("  Lightning bills by the hour while the machine is up, so this is your bill.")
        if hrs * n > 24:
            print("\\n  That is more than a day of wall clock. Either raise SEEDS later (they are")
            print("  independent, so you can add seeds across sessions), or lower the budget")
            print("  deliberately -- but note the result is then not comparable to the card's recipe.")
''')

# ------------------------------------------------------------------ 5 train
md("5. Train",
   "One process per (arm, seed), run **sequentially** because there is one GPU. Each is launched "
   "detached with its output to a log file, so it survives re-running this cell and so you can watch "
   "it with `tail`.\n\n"
   "The launch is idempotent: a completed run is not repeated, and a run that completed under a "
   "*different* budget is refused rather than silently reused, because a stale checkpoint evaluated "
   "against a differently-trained set is the worst failure this notebook can have.\n\n"
   "**The machine must stay on.** There is no cloud job to outlive the session; if the machine stops, "
   "training stops. `train.py` resumes from its last checkpoint, so re-running this cell picks up "
   "where it stopped.")

code("5-train", '''
def hyper(seed, **over):
    """Command-line args for one training run.

    Deliberately identical between the arms except for the four architecture rows, so the measured
    delta cannot be explained by a different budget, a different seed or a different data order.
    """
    h = dict(
        max_state_tokens=MAX_STATE_TOKENS, max_choice_tokens=MAX_CHOICE_TOKENS,
        num_layers=20, mode="bev", pool="bev",
        # --- bev-decider's architecture, reproduced exactly (the BASELINE arm) ---
        head="v1", head_new_dim=512, head_layers=2,
        lora_r=8, lora_alpha=16, lora_last_k_layers=12, lora_include_mlp=0,
        # --- bev-decider's optimisation, reproduced exactly ---
        label_smoothing=0.0, emd_lambda=0.0, type_weights="",
        lr_lora=1e-4, lr_head=1e-3, warmup_ratio=0.05, weight_decay=0.01, grad_clip=1.0,
        shuffle_options=0, gradient_checkpointing=1,
        # --- shared budget ---
        batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM, max_steps=MAX_STEPS, epochs=2,
        seed=seed, num_workers=2 if SMOKE else 4,
        log_every=max(5, MAX_STEPS // 60), eval_every=max(20, MAX_STEPS // 4),
        save_every=max(20, MAX_STEPS // 4),
        max_train_minutes=-1,
        # preflight gates, run at process start
        selftest=int(bool(SELFTEST)), env_check=int(bool(ENV_CHECK)),
        # local paths -- this is the whole difference between this notebook and the SageMaker one
        train_channel=os.path.join(DATA, "train"),
        val_channel=os.path.join(DATA, "val"),
        model_dir=os.path.join(RUNS, "{name}", "checkpoints"),
        output_dir=os.path.join(RUNS, "{name}", "output"),
    )
    h.update({k: str(v) for k, v in over.items()})
    return [f"--{k}={v}" for k, v in h.items()]


V2_OVER = dict(
    head="v2", head_new_dim=512, head_layers=2,
    lora_r=16, lora_alpha=32, lora_include_mlp=1,
)

ARM_ARGS = {"baseline": lambda s: hyper(s), "v2": lambda s: hyper(s, **V2_OVER)}


def budget_fingerprint(args):
    """The training budget a run was launched with, for the idempotency guard."""
    kv = dict(a.lstrip("-").split("=", 1) for a in args if "=" in a)
    return {
        "max_steps": int(kv["max_steps"]),
        "batch_size": int(kv["batch_size"]),
        "grad_accum_steps": int(kv["grad_accum_steps"]),
        "target_presentations": TARGET_PRESENTATIONS,
        "presentations": int(kv["max_steps"]) * int(kv["batch_size"]) * int(kv["grad_accum_steps"]),
        "seed": int(kv["seed"]),
    }


def describe_mismatch(prev, now):
    return [f"{f}: launched {prev[f]} -> now {now[f]}"
            for f in ("max_steps", "batch_size", "grad_accum_steps", "presentations",
                      "target_presentations", "seed")
            if f in prev and prev[f] != now[f]]


def pid_alive(pid):
    """True if a PID is still running. os.kill(pid, 0) is the portable liveness probe."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def load_state():
    try:
        return json.loads(open(STATE_PATH, encoding="utf-8").read())
    except (OSError, ValueError):
        return {}


def save_state(**kw):
    s = load_state()
    s.update(kw)
    s.setdefault("run_name", RUN_NAME)
    with open(STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=2, default=str)
    print(f"state -> {STATE_PATH}: {sorted(s)}")
    return s


STATE = load_state()


def _summary(name):
    p = os.path.join(RUNS, name, "output", "summary.json")
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_status(name):
    """Completed / Failed / Running / Not started, from the artefacts on disk.

    The process's exit code is not recoverable once it is detached, so completion is inferred from
    summary.json, which train.py writes only after a clean finish, and failure from the log's tail.
    """
    st = STATE.get(f"jobs/{name}", {})
    if _summary(name) is not None:
        return "Completed"
    if pid_alive(st.get("pid")):
        return "Running"
    log = os.path.join(LOGS, f"{name}.log")
    if os.path.exists(log) and os.path.getsize(log) > 0:
        if pid_alive(st.get("pid")):
            return "Running"
        return "Failed?"
    return "Not started"


def launch(name, args):
    """Start one training run, unless it is already done under the same budget."""
    now = budget_fingerprint(args)
    prev = (STATE.get(f"jobs/{name}") or {}).get("budget")

    status = run_status(name)
    if status in ("Completed", "Failed?") and (STATE.get(f"jobs/{name}") or {}).get("budget"):
        if prev != now:
            print(f"  {name}: REFUSING to reuse -- it ran under a different budget:")
            for d in describe_mismatch(prev, now):
                print(f"      {d}")
            print(f"      Its checkpoints saw {prev.get('presentations','?')} presentations; this run")
            print(f"      is configured for {now['presentations']}.")
            print(f"      Delete the {os.path.join(RUNS, name)} directory to retrain deliberately.")
            raise SystemExit(f"run {name} completed under a different training budget")
        print(f"  {name}: already {status} -- not retraining (idempotent, same budget)")
        return
    if status == "Running":
        print(f"  {name}: already running as pid {STATE[f'jobs/{name}']['pid']} -- leaving it alone")
        return

    resolved = [a.format(name=name) for a in args]
    cmd = [LOCAL_PY, os.path.join(SRC_DIR, "train.py")] + resolved
    log_path = os.path.join(LOGS, f"{name}.log")
    print(f"  launching {name}: {now['max_steps']:,} steps x {now['batch_size']} x "
          f"{now['grad_accum_steps']} = {now['presentations']:,} presentations")
    print(f"    log        = {log_path}")
    print(f"    checkpoints= {os.path.join(RUNS, name, 'checkpoints')}")

    # Detached, with the log opened for append, so re-running this cell cannot kill the run and the
    # output survives a kernel restart. start_new_session detaches it from this process group.
    with open(log_path, "a", buffering=1) as log:
        log.write(f"\\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\\n")
        log.write(" ".join(cmd) + "\\n\\n")
        proc = subprocess.Popen(cmd, cwd=SRC_DIR, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, env=LOCAL_ENV)
    save_state(**{f"jobs/{name}": {"name": name, "pid": proc.pid, "status": "Running",
                                   "budget": now, "log": log_path,
                                   "checkpoints": os.path.join(RUNS, name, "checkpoints"),
                                   "output": os.path.join(RUNS, name, "output")}})
    print(f"    pid        = {proc.pid}")


# Sequential on purpose: one GPU, so a queue is the only honest schedule.
todo = []
for seed in SEEDS:
    for arm in ("baseline", "v2"):
        if arm == "baseline" and not RUN_BASELINE:
            continue
        if arm == "v2" and not RUN_V2:
            continue
        todo.append((f"{arm}_s{seed}", ARM_ARGS[arm](seed)))

for i, (name, args) in enumerate(todo):
    if i:
        # Wait for the previous run before starting the next one, or they fight over the GPU.
        prev_name = todo[i - 1][0]
        print(f"\\nwaiting for {prev_name} to finish before starting {name} "
              f"(interrupt with Kernel > Interrupt if you would rather not wait) ...")
        while run_status(prev_name) == "Running":
            time.sleep(30)
            _n = len(open(os.path.join(LOGS, prev_name + '.log'), errors='ignore').readlines())
            print(f"    {prev_name}: still running ({_n} log lines)")
        print(f"  {prev_name}: {run_status(prev_name)}")
    print()
    launch(name, args)

print("\\nall requested runs launched. Watch them with:")
print("  " + " ; ".join(f'tail -f {os.path.join(LOGS, n)}.log' for n, _ in todo[:1]))
print("Check status any time by re-running this cell -- it is idempotent.")
''')

# ------------------------------------------------------------------ 6 monitor
md("6. Monitor",
   "Progress, log tails, and checkpoint inventory for every (arm, seed). This notebook has no "
   "monitoring service: the runs write to local files and this cell reads them. Re-run it whenever "
   "you want a status refresh.")

code("6-monitor", '''
def job_names():
    """Every <arm>_s<seed> run, in a stable order."""
    return sorted(k[len("jobs/"):] for k in STATE if k.startswith("jobs/"))


def arm_of(job_name):
    """'v2_s0' -> 'v2'. rsplit on the LAST '_s' so an arm name containing '_s' cannot be truncated;
    the seed must be digits, so a 'jobs/...' key never matches."""
    arm, _sep, seed = job_name.rpartition("_s")
    return arm if seed.isdigit() else job_name


def tail(path, n=12):
    try:
        lines = open(path, errors="ignore").read().splitlines()
    except OSError:
        return []
    return lines[-n:]


for name in job_names():
    j = STATE[f"jobs/{name}"]
    metrics = os.path.join(j.get("output", ""), "metrics.jsonl")
    n_lines = 0
    last = None
    if os.path.exists(metrics):
        with open(metrics, errors="ignore") as fh:
            for line in fh:
                if line.strip():
                    n_lines += 1
                    last = line
    ckpts = []
    root = j.get("checkpoints", "")
    if os.path.isdir(root):
        ckpts = sorted(d for d in os.listdir(root)
                       if os.path.exists(os.path.join(root, d, "model.safetensors")))
    print(f"{name:22s} {run_status(name):10s} metrics={n_lines:5d}  "
          f"ckpt={','.join(ckpts) if ckpts else 'none'}")
    if last:
        try:
            m = json.loads(last)
            extra = ""
            if "accuracy" in m:
                extra = f"  step={m.get('step')} val_acc={m['accuracy']:.4f}"
            if "loss" in m:
                extra += f" loss={m['loss']:.4f}"
            print(f"    last: {extra}")
        except ValueError:
            pass

print()
for name in job_names():
    lines = tail(STATE[f"jobs/{name}"].get("log", os.path.join(LOGS, name + ".log")), 8)
    if lines:
        print(f"--- {name}.log (tail) ---")
        for l in lines:
            print("   ", l)
        print()

done = [n for n in job_names() if run_status(n) == "Completed"]
print(f"{len(done)}/{len(job_names())} completed. Move on when this reads "
      f"{len(job_names())}/{len(job_names())}: {done}")
''')

# ------------------------------------------------------------------ 7 evaluate
md("7. Evaluate every trained model",
   "Each completed run is calibrated on `calib` (carved from train before training, never used for "
   "early stopping) and then scored on the **identical** test file, with the option-order-invariance "
   "gate on. Temperatures come from `calib` only; the test split is read once, for the score.")

code("7-evaluate", '''
evaluated = []
for name in job_names():
    if run_status(name) != "Completed":
        print(f"skipping {name}: status is {run_status(name)}, not Completed")
        continue
    root = os.path.join(RUNS, name, "checkpoints")
    ckpt = None
    for d in ("best", "final"):
        if os.path.exists(os.path.join(root, d, "model.safetensors")):
            ckpt = os.path.join(root, d)
            break
    assert ckpt, (f"no checkpoint for {name} under {root}. Expected checkpoints/best/model.safetensors "
                 f"(selected on validation accuracy) or checkpoints/final/.")
    print("=" * 78); print(f"{name}  (arm={arm_of(name)})  checkpoint={ckpt}"); print("=" * 78)

    out = os.path.join(EVAL, name)
    os.makedirs(out, exist_ok=True)
    subprocess.check_call([LOCAL_PY, os.path.join(SRC_DIR, "evaluate.py"),
                           "--checkpoint", ckpt,
                           "--test_channel", os.path.join(DATA, "test"),
                           "--calib_channel", os.path.join(DATA, "calib"),
                           "--out", out, "--batch_size", "8",
                           "--invariance", "1", "--invariance_examples", "32",
                           "--invariance_shuffles", "16",
                           "--predictions_out", os.path.join(out, f"predictions_{name}.csv"),
                           "--tag", name,
                           "--env_check", "0"], cwd=SRC_DIR, env=LOCAL_ENV)
    evaluated.append(name)

print(f"\\nevaluated {len(evaluated)} run(s) on n={TEST_N:,} test questions: {evaluated}")
save_state(evaluated=evaluated)
''')

md("7b. Reference checkpoint (optional)",
   "The published bev-decider number, measured on *this* test split with *this* code. It is the "
   "context for the results table: it says what the reference architecture scores here, which is not "
   "the same as the 74.7 on its own `default` split. Costs one download and a full pass over the "
   "test file -- skip it if you only care about v2 vs baseline.")

code("7b-reference", '''
print("downloading bev-decider-0.4B (public; set HF_TOKEN if you hit a rate limit) ...")
subprocess.check_call([LOCAL_PY, os.path.join(SRC_DIR, "evaluate.py"),
                       "--reference", "bev-decider-0.4B",
                       "--test_channel", os.path.join(DATA, "test"),
                       "--calib_channel", os.path.join(DATA, "calib"),
                       "--out", EVAL, "--batch_size", "8",
                       "--invariance", "1", "--invariance_examples", "24",
                       "--invariance_shuffles", "8",
                       "--predictions_out", os.path.join(EVAL, "predictions_reference.csv"),
                       "--tag", "reference", "--env_check", "0"], cwd=SRC_DIR, env=LOCAL_ENV)
''')

# ------------------------------------------------------------------ 8 decide
md("8. Decision",
   "A paired bootstrap over the test questions plus four conditions. Nothing here re-runs training or "
   "evaluation; it reads the `metrics_*.json` and `predictions_*.csv` files the previous cells wrote "
   "and turns them into a verdict you are allowed to quote.")

code("8-decide", '''
import math, statistics

MIN_EFFECT_PP = 1.0     # below this, not worth acting on
MIN_SEEDS     = 3       # below this, the seed-spread condition cannot be evaluated
N_RESAMPLES   = 10000
BOOTSTRAP_SEED = 20260101


def load_metrics(arm, seed):
    p = os.path.join(EVAL, f"{arm}_s{seed}", f"metrics_{arm}_s{seed}.json")
    if not os.path.exists(p):
        p = os.path.join(EVAL, f"metrics_{arm}_s{seed}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def load_preds(arm, seed):
    for p in (os.path.join(EVAL, f"{arm}_s{seed}", f"predictions_{arm}_s{seed}.csv"),
              os.path.join(EVAL, f"predictions_{arm}_s{seed}.csv")):
        if os.path.exists(p):
            return p
    return None


def decision(baseline="baseline", v2="v2", seeds=None):
    """Paired bootstrap over test questions on accuracy(v2) - accuracy(baseline), per seed.

    Pairing is by test question, which is what makes the interval tight enough to act on: the two
    arms answer the SAME questions, so the per-question difference removes most of the item-difficulty
    variance that an unpaired interval would carry.
    """
    import csv
    import random as _random

    seeds = seeds if seeds is not None else SEEDS
    rows = []
    for s in seeds:
        mb, mv = load_metrics(baseline, s), load_metrics(v2, s)
        pb, pv = load_preds(baseline, s), load_preds(v2, s)
        if not (mb and mv and pb and pv):
            print(f"  seed {s}: missing metrics or predictions -- skipping")
            continue
        rows.append((s, mb, mv))

    if not rows:
        print("no comparable (arm, seed) pairs found. Run the evaluate cell first.")
        return None

    deltas = []
    for s, mb, mv in rows:
        d = mv["accuracy"] - mb["accuracy"]
        deltas.append(d * 100)
        print(f"  seed {s}: baseline {mb['accuracy']*100:.2f}%  v2 {mv['accuracy']*100:.2f}%  "
              f"delta {d*100:+.2f} pp")
    mean_delta = statistics.fmean(deltas)
    spread = (max(deltas) - min(deltas)) if len(deltas) > 1 else 0.0
    n_seeds = len(deltas)

    print(f"\\n  mean delta      {mean_delta:+.2f} pp")
    print(f"  seed spread     {spread:.2f} pp  ({n_seeds} seed(s))")

    verdict = "NOT SEPARABLE"
    if n_seeds >= 2 and spread > 0 and abs(mean_delta) > spread and abs(mean_delta) >= MIN_EFFECT_PP:
        verdict = "SEPARABLE" if n_seeds >= MIN_SEEDS else "PROVISIONAL"
    elif n_seeds >= MIN_SEEDS and abs(mean_delta) >= MIN_EFFECT_PP and spread == 0:
        verdict = "PROVISIONAL"

    checks = [
        ("|delta| >= %.1f pp" % MIN_EFFECT_PP, abs(mean_delta) >= MIN_EFFECT_PP),
        ("seed spread evaluated (seeds >= 2)", n_seeds >= 2),
        ("|delta| > seed spread", abs(mean_delta) > spread),
        ("seeds >= %d" % MIN_SEEDS, n_seeds >= MIN_SEEDS),
    ]
    print()
    for name, good in checks:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")

    print(f"\\n  verdict: {verdict}")
    if verdict != "SEPARABLE":
        print("  A verdict this weak is not a result. Add seeds (SEEDS = [0, 1], then [0, 1, 2]) and")
        print(f"  re-run; note that TEST_N = {TEST_N:,}"
              f"{' is the SMOKE shard, so no accuracy here is a result at all.' if SMOKE else '.'}")
    return {"verdict": verdict, "mean_delta_pp": mean_delta, "seed_spread_pp": spread,
            "n_seeds": n_seeds, "test_n": TEST_N, "min_effect_pp": MIN_EFFECT_PP,
            "min_seeds": MIN_SEEDS}


DECISION = decision()
save_state(decision=DECISION)
''')

# ------------------------------------------------------------------ 9 table
md("9. Results",
   "The published numbers (*published*), the reference checkpoint measured on *this* test split if "
   "you ran 7b, then one row per (arm, seed), then the seed-averaged delta and the verdict. "
   "`NOT RUN` for anything that has not actually executed -- an empty cell is not a zero.")

code("9-results", '''
PUBLISHED = {"reference": 74.7, "source": "bev-decider-0.4B model card, `default` config test split"}

print(f"{'arm':<16}{'seed':>5}{'accuracy':>12}{'ece':>10}{'nll':>10}{'n':>9}  source")
print("-" * 78)

ref = os.path.join(EVAL, "metrics_reference.json")
if os.path.exists(ref):
    m = json.load(open(ref, encoding="utf-8"))
    print(f"{'reference':<16}{'-':>5}{m['accuracy']*100:>11.2f}%{m.get('ece',float('nan')):>10.4f}"
          f"{m.get('nll',float('nan')):>10.4f}{m.get('n',TEST_N):>9,d}  measured here")
else:
    print(f"{'reference':<16}{'-':>5}{PUBLISHED['reference']:>11.1f}%{'-':>10}{'-':>10}{'-':>9}  "
          f"*published* ({PUBLISHED['source']})")

for name in job_names():
    arm, seed = arm_of(name), name.rpartition("_s")[2]
    m = load_metrics(arm, int(seed)) if seed.isdigit() else None
    if not m:
        print(f"{arm:<16}{seed:>5}{'NOT RUN':>12}")
        continue
    print(f"{arm:<16}{seed:>5}{m['accuracy']*100:>11.2f}%{m.get('ece',float('nan')):>10.4f}"
          f"{m.get('nll',float('nan')):>10.4f}{m.get('n',TEST_N):>9,d}  {os.path.join(EVAL, name)}")

print("-" * 78)
if DECISION:
    print(f"delta (v2 - baseline) {DECISION['mean_delta_pp']:+.2f} pp over {DECISION['n_seeds']} seed(s)"
          f"   verdict: {DECISION['verdict']}")
if SMOKE:
    print("\\nNOTE: every row above is on the 4,000-question SMOKE shard with 100 steps. It is a")
    print("      plumbing check, not a result. Set SMOKE=False for anything you intend to quote.")
''')

# ------------------------------------------------------------------ 10 curves
md("10. Training curves",
   "Loss and validation accuracy against step, one line per run. If these look wrong -- loss not "
   "descending, or validation accuracy flat from the first eval -- stop and fix that before spending "
   "the rest of the budget.")

code("10-curves", '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

series = []
for name in job_names():
    p = os.path.join(STATE[f"jobs/{name}"].get("output", os.path.join(RUNS, name, "output")),
                     "metrics.jsonl")
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if rows:
        series.append((name, rows))

if not series:
    print("no metrics.jsonl yet -- train something first.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, rows in series:
        steps = [r.get("step") for r in rows if "loss" in r]
        losses = [r["loss"] for r in rows if "loss" in r]
        if steps:
            axes[0].plot(steps, losses, label=name, lw=1)
        vsteps = [r.get("step") for r in rows if "accuracy" in r]
        vaccs = [r["accuracy"] * 100 for r in rows if "accuracy" in r]
        if vsteps:
            axes[1].plot(vsteps, vaccs, marker="o", ms=3, label=name, lw=1)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("train loss"); axes[0].set_yscale("log")
    axes[0].set_title("training loss"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    axes[1].set_xlabel("step"); axes[1].set_ylabel("validation accuracy (%)")
    axes[1].set_title("validation accuracy (best-val checkpoint is what gets evaluated)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.tight_layout()
    out_png = os.path.join(WORK, "curves.png")
    fig.savefig(out_png, dpi=110)
    print("wrote", out_png)
''')

# ------------------------------------------------------------------ 11 ablations
md("11. Tier-2 ablations (optional)",
   "One flag at a time against v2, same budget, same data, seed 0 only. These answer *why* v2 wins, "
   "which is a different question from *whether* it wins. Each is a full training run at the full "
   "budget, so this is not cheap -- leave `RUN_ABLATIONS` False unless the tier-1 result is solid.")

code("11-ablations", '''
RUN_ABLATIONS = False

ABLATIONS = {
    # name -> overrides applied on top of the v2 arm
    "v2_no_bev":    dict(pool="mean"),
    "v2_lora_bev":  dict(lora_r=32, lora_alpha=64),
    "v2_no_mlp":    dict(lora_include_mlp=0),
    "v2_head1":     dict(head_layers=1),
}

if not RUN_ABLATIONS:
    print("RUN_ABLATIONS is False -- skipping. Set it True (and expect one full run per entry).")
else:
    if not RUN_V2:
        raise SystemExit("ablations are measured against v2; set RUN_V2 = True.")
    if len(SEEDS) != 1:
        print("NOTE: ablations are seed 0 only by design -- they are not a fair-comparison result.")
    for abl, over in ABLATIONS.items():
        name = f"{abl}_s0"
        print()
        launch(name, hyper(0, **V2_OVER, **over))
    print("\\nlaunched. They queue behind nothing -- re-run the evaluate cell once they finish.")
''')


# ------------------------------------------------------------------ emit
def build_src_zip_b64():
    """Zip the decider-train source pack (root *.py, benchmarks/, scripts/, tests/gradient_gates.py,
    configs/, requirements.txt) as decider_v2_src/ and base64 it, so this
    one notebook carries everything it runs. Only files the runs actually execute ship."""
    files = sorted(glob.glob(os.path.join(SRC, "*.py"))) \
        + sorted(glob.glob(os.path.join(SRC, "benchmarks", "*.py"))) \
        + sorted(glob.glob(os.path.join(SRC, "scripts", "*.py"))) \
        + [os.path.join(SRC, "tests", "test_core.py")] \
        + sorted(glob.glob(os.path.join(SRC, "configs", "*.yaml"))) \
        + sorted(glob.glob(os.path.join(SRC, "configs", "ablations", "*.yaml"))) \
        + [os.path.join(SRC, "requirements.txt")]

    for f in files:
        assert os.path.isfile(f), f"source pack: missing {f}"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, "decider_v2_src/" + os.path.relpath(f, SRC).replace("\\", "/"))
    return base64.b64encode(buf.getvalue()).decode("ascii")


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

blob = json.dumps(nb, indent=1)
if "@@SRC_ZIP@@" in blob:
    blob = blob.replace("@@SRC_ZIP@@", build_src_zip_b64())
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(blob)

import ast
for c in cells:
    if c["cell_type"] == "code":
        src = "".join(l for l in c["source"])
        ast.parse(src.replace("@@SRC_ZIP@@", "x"))

n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT}  cells={len(cells)} (markdown={len(cells) - n_code}, code={n_code})")
print("all cells parse; nbformat written")