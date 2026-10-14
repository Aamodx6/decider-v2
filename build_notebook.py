"""Generate decider_v2_sagemaker.ipynb.

The notebook is generated rather than hand-written so that the source files are embedded from the
real files on disk (no transcription drift) and the cell list is guaranteed to match the mission's
15 phases.  Run:  python build_notebook.py
"""
import glob
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "decider_v2")
OUT = os.path.join(HERE, "decider_v2_sagemaker.ipynb")

# cells referenced by name from other cells (idempotency helpers, cleanup, results)
STATE = '''
import json, os, time, pathlib

STATE_PATH = pathlib.Path("decider_v2_state.json")


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


STATE = load_state()


def save_state(**kw):
    """Merge into the on-disk state AND the in-memory STATE, so later cells see it immediately."""
    s = load_state()
    s.update(kw)
    STATE.clear()
    STATE.update(s)
    s.setdefault("run_name", RUN_NAME if "RUN_NAME" in dir() else s.get("run_name"))
    STATE_PATH.write_text(json.dumps(s, indent=2, default=str))
    print(f"state -> {STATE_PATH}: {sorted(s)}")
    return STATE
'''

CELLS = []


def md(title, body=""):
    """Emit documentation as a code cell that PRINTS ITSELF when run.

    Two earlier designs both failed. As a real markdown cell, SageMaker's Jupyter executed the prose as
    Python and the reader saw

        Cell In[1], line 3
          The training sources are embedded in the next cell as a base64 zip ...
        SyntaxError: invalid syntax

    As a comment-only code cell, it fixed that but produced the opposite problem: running one printed
    NOTHING AT ALL, and because every cell now looks alike there is no way to tell a note from a step
    until you run it. "the smoke test cell gives no output" is that bug.

    So a note now announces itself. Run it and you get the text back, which both confirms you hit a
    note and tells you what the surrounding step is for. It is still a comment block for the reader,
    but it is not silent.

    The lines are emitted as a list of repr'd strings so that no amount of quotes, backslashes or
    triple-quotes in the prose can break the cell.
    """
    lines = [ln.rstrip() for ln in body.strip("\n").split("\n")]
    bar = "# " + "=" * 74
    out = [bar, f"# {title}", "#", ]
    out += ["# " + ln if ln else "#" for ln in lines]
    out += ["", "# (this is a note -- running it just prints the text above).", ""]
    payload = ",\n    ".join(repr(ln) for ln in lines)
    out += ["_NOTE = [", f"    {payload},", "]", 'print("\\n".join(_NOTE))']
    CELLS.append(("note", "\n".join(out) + "\n", ""))


def code(name, source):
    CELLS.append(("code", f"# --- {name} ---\n", source))


# ---------------------------------------------------------------- 0 sources
# The training sources ship INSIDE this notebook as a base64 zip, extracted to ./decider_v2_src on
# first run, so the notebook is a genuinely single file: upload it to SageMaker Studio and run it.
# This replaces the previous arrangement, where a missing decider_v2/ folder aborted cell 1 with
# "decider_v2_src not found" -- which is exactly what happened when only the .ipynb was uploaded.

md("How this notebook runs",
   "**Training always runs in SageMaker training jobs** on the GPU instance type set in cell 1. This "
   "notebook never trains anything: it carves the data, uploads it as input channels, launches the "
   "jobs and collects the results. Closing the notebook does not stop a running job, and a laptop "
   "with no GPU is a perfectly good place to drive it.\n\n"
   "That matters because it changes what a *no GPU here* does and does not mean:\n\n"
   "- **Does not affect training.** The jobs run on their own GPU instances; `BATCH_SIZE = 32` needs "
   "~8.3 GiB and every instance offered in cell 8 has far more.\n"
   "- **Does affect the self-test and evaluation**, which run here, in this notebook, on CPU. They "
   "are correct but slow — on 4,000 test questions a CPU pass takes tens of minutes rather than "
   "minutes. Start them and let them run.\n"
   "- **Skips the throughput bench**, because a CPU measurement would price a GPU budget against a "
   "card that never trains it. Cell 6 says so and cell 6b then refuses to guess.\n\n"
   "If you want the cost estimate before spending money, restart this notebook instance on a GPU "
   "instance and re-run cell 6.")

md("1. Sources (self-extracting)",
   "The training sources are embedded in the next cell as a base64 zip and unpacked to "
   "`./decider_v2_src`. **This notebook is self-contained** -- there is nothing else to upload or "
   "clone.\n\n"
   "Extraction is keyed on a **content digest**, not on whether the folder exists. That matters "
   "because the folder survives between notebook versions: if you upload an updated notebook over an "
   "old one and the check were merely \"does the folder exist\", the new sources would silently never "
   "be extracted and you would debug the previous version's code. So the cell compares the embedded "
   "zip's sha256 against a stamp in the folder, re-extracts when they differ, and tells you which it "
   "did.\n\n"
   "If you edit `decider_v2_src/` by hand, the next extraction will overwrite your edits. That is "
   "deliberate: the notebook is the source of truth. Edit `decider_v2/` and re-run "
   "`python build_notebook.py` instead.")

code("0-sources", '''
# The notebook's own training code, embedded as a base64 zip by build_notebook.py.
SRC_ZIP_B64 = "@@SRC_ZIP@@"

import base64, io, os, sys, zipfile, hashlib

SRC_DIR = os.path.abspath("decider_v2_src")
STAMP = os.path.join(SRC_DIR, ".notebook_zip.sha256")
# Escape hatch for anyone who really is patching decider_v2_src in place: set this to 1 before
# re-running the cell and the extraction is skipped regardless of the digest.
FORCE_REUSE = False

_blob = base64.b64decode(SRC_ZIP_B64)
_digest = hashlib.sha256(_blob).hexdigest()

# The decision is about CONTENT, not existence. An earlier revision asked only whether train.py was
# present, which meant that uploading a fixed notebook over an older one silently kept running the
# OLD sources -- every fix to venv_env.py, prepare.py or train.py was skipped without a word. That is
# the worst possible failure here: the notebook and the code it claims to run quietly disagree.
_on_disk = None
if os.path.isfile(STAMP):
    try:
        _on_disk = open(STAMP, encoding="utf-8").read().strip()
    except OSError:
        _on_disk = None

_have_sources = os.path.isfile(os.path.join(SRC_DIR, "train.py"))
if FORCE_REUSE:
    _action = "reuse"
    print("FORCE_REUSE=1 -- keeping decider_v2_src as-is, without checking its contents.")
elif _have_sources and _on_disk == _digest:
    _action = "reuse"
    print(f"decider_v2_src already matches this notebook's embedded sources "
          f"(sha256 {_digest[:16]}) -- leaving your files untouched.")
elif _have_sources and _on_disk != _digest:
    _action = "replace"
    print("=" * 78)
    print("STALE SOURCES: decider_v2_src does not match this notebook.")
    print("=" * 78)
    if _on_disk is None:
        print("  The folder has no digest stamp, so it was extracted by an older revision of this")
        print("  notebook that only checked for existence. It is about to be REPLACED.")
    else:
        print(f"  on disk : sha256 {_on_disk[:16]}")
        print(f"  embedded: sha256 {_digest[:16]}")
    print("  Continuing without this would run the OLD code while the notebook showed the NEW")
    print("  code's output, which is how a fix gets 'not working' after being applied.")
    print("  Any hand edits in that folder are about to be overwritten.")
    print("  To keep them instead, set FORCE_REUSE = 1 above and re-run -- but then this notebook")
    print("  is no longer the thing you are testing.")
else:
    _action = "extract"

if _action in ("extract", "replace"):
    print(f"extracting decider_v2_src from embedded zip ({len(_blob):,} B, sha256 {_digest[:16]}) ...")
    with zipfile.ZipFile(io.BytesIO(_blob)) as _z:
        _names = [n for n in _z.namelist() if not n.endswith("/")]
        for _n in _names:
            # Entries are stored as "decider_v2_src/<rel>"; strip that prefix and write relative to
            # SRC_DIR itself, or every file lands in decider_v2_src/decider_v2_src/... .
            _rel = _n[len("decider_v2_src/"):] if _n.startswith("decider_v2_src/") else _n
            _dest = os.path.abspath(os.path.join(SRC_DIR, _rel))
            # Refuse any path that escapes SRC_DIR (zip-slip).
            if not _dest.startswith(SRC_DIR + os.sep):
                raise SystemExit(f"refusing unsafe path in embedded zip: {_n}")
            os.makedirs(os.path.dirname(_dest), exist_ok=True)
            with _z.open(_n) as _src, open(_dest, "wb") as _out:
                _out.write(_src.read())
    print(f"wrote {len(_names)} files to {SRC_DIR}")

if not FORCE_REUSE:
    # Sweep files the notebook does not ship -- on BOTH paths, not just after an extract. A file
    # hand-dropped into the folder survives a digest match otherwise, and since decider_v2_src is
    # first on sys.path such a file can be imported in preference to the real one. That is the same
    # "the folder quietly disagrees with the notebook" hazard the digest check exists to prevent.
    with zipfile.ZipFile(io.BytesIO(_blob)) as _z:
        _want = {(_n[len("decider_v2_src/"):] if _n.startswith("decider_v2_src/") else _n)
                 for _n in _z.namelist() if not _n.endswith("/")}
    for _root, _dirs, _files in os.walk(SRC_DIR):
        for _f in _files:
            _rel = os.path.relpath(os.path.join(_root, _f), SRC_DIR).replace("\\\\", "/")
            if _rel == ".notebook_zip.sha256" or _rel.endswith(".pyc"):
                continue
            if _rel not in _want:
                print(f"  removing stale file not in this notebook: {_rel}")
                try:
                    os.remove(os.path.join(_root, _f))
                except OSError as _e:
                    print(f"    could not remove {_rel}: {_e}")

if not FORCE_REUSE:
    with open(STAMP, "w", encoding="utf-8") as _fh:
        _fh.write(_digest)
    # Dropped AFTER extraction so a stale copy of an already-imported module (this kernel ran an
    # earlier notebook) cannot satisfy the import below.
    for _m in [m for m in list(sys.modules)
               if m in ("environment", "venv_env", "data", "serialize", "train", "evaluate",
                        "decide", "losses", "model", "prepare", "calibrate", "bench")]:
        del sys.modules[_m]

if not os.path.isfile(os.path.join(SRC_DIR, "train.py")):
    raise SystemExit(f"extraction did not produce {SRC_DIR}/train.py -- corrupt embedded zip?")
if SRC_DIR in sys.path:
    sys.path.remove(SRC_DIR)
sys.path.insert(0, SRC_DIR)
import environment
print("requirements pins:", [f"{n}=={v}" for n, v in environment.parse_requirements()])
print("sources ok")
''')


# ---------------------------------------------------------------- 1 config
md("1. Config",
   "Edit the constants below and nothing else. `SMOKE=True` runs the whole pipeline end-to-end on a "
   "tiny subset with 100 steps so you can validate the notebook before spending money. The real run "
   "needs `SMOKE=False` plus a bucket, an instance type and `HF_TOKEN` in the environment "
   "(or already-cached weights).")

code("1-config", '''
import json, os, sys, time, getpass

# ------------------------------------------------------------------ EDIT ME
BUCKET          = ""            # e.g. "my-decider-v2"  (must already exist, region must match)
PREFIX          = "decider_v2"   # S3 key prefix for every artifact this notebook creates
REGION          = ""            # "" -> boto3 session region
INSTANCE_TYPE   = "ml.g5.2xlarge"   # 1x A10G 24 GB.  Bigger options in cell 8.
ROLE_NAME       = ""            # "" -> derived from this notebook instance's own STS identity

# SageMaker container. Verified against sagemaker/image_uri_config/pytorch.json (SDK 2.247.0):
#   2.5.1 -> py311      2.6.0 -> py312      2.7.1 -> py312
# 2.7.1/py312 is the newest combination, and py>=3.11 is REQUIRED because numpy 2.3.5 declares
# requires-python >=3.11. (The previous revision asked for 2.6.0 + py311, which is not a valid
# pairing -- 2.6.0 only exists as py312 -- so the job would have failed at launch.)
FRAMEWORK_VERSION = "2.7.1"
PY_VERSION        = "py312"

SMOKE           = True          # True: tiny subset + 100 steps, everything still runs

# Training always runs as SageMaker training jobs on INSTANCE_TYPE below. This notebook only
# orchestrates: it carves the data, uploads it as input channels, launches the jobs and collects the
# results. It never trains the model itself, so closing the notebook does not stop a running job.

# Micro-batch. The recipe's 32 needs ~8.3 GiB of VRAM, which every instance type offered in cell 8
# provides (ml.g5.2xlarge is 1x A10G 24 GB). Lower it only if you move to a smaller instance, and see
# GRAD_ACCUM below for the rule that must accompany that.
BATCH_SIZE      = 8 if SMOKE else 32   # micro-batch
# 32 x 2 = 64 questions / step, bev-decider's published recipe.
#
# If you ever move to an instance too small for micro-batch 32, you must lower BATCH_SIZE to fit
# VRAM and then RAISE GRAD_ACCUM to keep the same 64 questions per step (8 x 8 = 64). What you must
# NOT do is leave GRAD_ACCUM at 2 -- 8 x 2 = 16 questions/step would silently train on a quarter of
# the data for the same wall clock. The check further down refuses a real run that does not present
# exactly 64, so this cannot happen silently.
GRAD_ACCUM      = 2

# ---- the training budget -------------------------------------------------------------------
# The published bev-decider-0.4B card says "cross-entropy on about 940K typed questions" and
# "12,000 steps at batch size 64". Those two numbers disagree (12,000 x 64 = 768,000), and neither
# is reachable as distinct data: bev-decision('all') train is 422,551 typed questions, of which this
# run carves 388,748 after holding out val and calib (ARCH_REVIEW.md §1.9 flags this as CANNOT
# VERIFY). So "940K" can only mean QUESTION PRESENTATIONS -- repeats over the distinct train
# questions -- and matching it is the closest anyone can get to the published recipe:
#
#   ceil(940,000 / 64) = 14,688 steps = 940,032 presentations = 2.42 epochs over 388,748 questions.
#
# The budget is expressed in PRESENTATIONS, the unit the model card uses and the unit that stays
# comparable if you change the batch size.
#
# IMPORTANT: the budget is denominated in PRESENTATIONS and RECIPE_QPS is the PUBLISHED 64. It is
# deliberately NOT derived from GRAD_ACCUM. The earlier `RECIPE_QPS = 32 * GRAD_ACCUM` looked harmless
# but coupled the budget to the memory layout: on a GPU too small for micro-batch 32 you must set
# BATCH_SIZE=8 and raise GRAD_ACCUM to 8 to keep 64 questions per step, and that formula would then
# read 32 x 8 = 256, cutting MAX_STEPS to 3,672 and training on 235,032 presentations instead of
# 940,032 -- a silent 4x under-train that still reports Completed and still produces a decision
# verdict. The published number is a constant of the comparison, not a function of the hardware.
RECIPE_QPS    = 64                           # bev-decider's published micro-batch 32 x accum 2
TARGET_PRESENTATIONS = 940_000           # the card's "about 940K typed questions"
STEPS_FROM_PRESENTATIONS = TARGET_PRESENTATIONS   # set to None to pin MAX_STEPS by hand
MAX_STEPS = (-(-STEPS_FROM_PRESENTATIONS // RECIPE_QPS) if STEPS_FROM_PRESENTATIONS else 12000)
# Saved before the SMOKE block overwrites MAX_STEPS, so cell 1 can report the real budget.
MAX_STEPS_REAL = MAX_STEPS
# What a step ACTUALLY presents in this run.
ACTUAL_QPS   = BATCH_SIZE * GRAD_ACCUM

# A real run must actually reach the recipe's 64 questions per step. Under SMOKE the subset is tiny
# and the micro-batch is deliberately small, so the mismatch is expected and harmless. In a real run a
# mismatch means the run is NOT the recipe and any number it produces is not comparable -- so refuse
# to continue rather than train something that cannot be quoted.
if not SMOKE and ACTUAL_QPS != RECIPE_QPS:
    raise SystemExit(
        f"real run presents {ACTUAL_QPS} questions/step (BATCH_SIZE {BATCH_SIZE} x GRAD_ACCUM "
        f"{GRAD_ACCUM}) but the recipe is {RECIPE_QPS}. To keep 64 on a smaller GPU, lower "
        f"BATCH_SIZE and RAISE GRAD_ACCUM to compensate -- e.g. BATCH_SIZE=8 with GRAD_ACCUM=8. "
        f"Training at a different questions/step changes what the run is and invalidates the "
        f"comparison against the published recipe.")

# SageMaker hard-kills a training job at max_runtime and the work is lost. At this budget the
# measured throughput puts v2 at ~13-40 h depending on instance, so the 12 h this notebook used to
# hardcode would terminate a real run mid-training and lose it. Cell 6b cross-checks this against
# the throughput measured on your own instance and refuses to launch if the job would be killed.
MAX_RUNTIME_HOURS = 72
MAX_RUNTIME = MAX_RUNTIME_HOURS * 3600

# SPLIT_SEED carves train/val/calib AND samples the SMOKE test shard. It is FIXED for the whole run
# and must never vary between arms: identical data is the fair-comparison rule, so this is not a
# knob to sweep. Only TRAINING seeds vary below.
SPLIT_SEED     = 0
# Training seeds. Every (arm, seed) pair is its own job at the SAME budget, so the v2-vs-baseline
# delta can be reported with a seed spread instead of a single noisy number.
#
# Default [0] = 1 seed = 2 jobs, which is the cheapest way to find out whether the 940K budget is
# affordable and whether the run converges at all. Be aware of what that does and does not buy:
# the decision rule needs >= 3 seeds for SEPARABLE and >= 2 for even PROVISIONAL, because with one
# seed the seed spread is 0 by construction and condition (b) is vacuous. So a 1-seed run cannot
# separate the architectures no matter how large the delta -- add seeds before believing any number.
#
# Staged launch: run [0], inspect, then set [0, 1] and re-run cells 8-9 (seed 0 is skipped as
# Completed), then [0, 1, 2]. `launch()` is idempotent, so re-running only submits the new seeds.
SEEDS          = [0]

# Preflight gates, passed to every job. Both default ON in train.py as well; they are set here
# explicitly so the hyperparameter string recorded in S3 shows exactly what the job will do.
#   SELFTEST     -> runs the packed-vs-separate + order-invariance gates INSIDE the container and
#                   aborts the job on failure (~10s, tiny config, no download, no data).
#   ENV_CHECK    -> verifies decider_v2/requirements.txt pins, pip-installs them if the container
#                   does not already satisfy them, and aborts if they still do not match.
SELFTEST        = True
ENV_CHECK       = True

# Rebuild the local venv from scratch instead of reusing it. Only needed if you suspect the venv is
# corrupt; normally it reuses itself and rebuilds automatically when requirements.txt changes.
REBUILD_VENV    = False

RUN_BASELINE    = True          # baseline replica (bev-decider architecture, our data/split)
RUN_V2          = True          # decider-v2
RUN_ABLATIONS   = False         # Tier-2 one-flag-at-a-time ablations (cost: one job each)
USE_SPOT        = False         # spot for training jobs (needs max_wait; fragile if evicted)
# SageMaker REQUIRES spot MaxWaitTime >= MaxRuntime, and a spot job that waits a long time for
# capacity and is then evicted wastes everything. At this budget a 30-minute wait cannot succeed
# anyway, so the default is tied to the runtime and the notebook refuses an invalid combination.
SPOT_MAX_WAIT_MINUTES = None      # None -> MAX_RUNTIME_HOURS * 60
if SPOT_MAX_WAIT_MINUTES is None:
    SPOT_MAX_WAIT_MINUTES = MAX_RUNTIME_HOURS * 60          # SageMaker requires max_wait >= max_run
elif SPOT_MAX_WAIT_MINUTES * 60 < MAX_RUNTIME:
    raise SystemExit(f"SPOT_MAX_WAIT_MINUTES ({SPOT_MAX_WAIT_MINUTES}) x 60s must be >= "
                     f"MAX_RUNTIME ({MAX_RUNTIME}s). SageMaker rejects a spot training job whose "
                     f"MaxWaitTime is shorter than its MaxRuntime.")

# The published `all` test split is 80,019 typed questions and is NEVER subsampled in the real run:
# every arm, every seed and the reference all score the same full file, so the comparison has no
# sampling noise of our own making. The 4,000-question shard exists ONLY under SMOKE, where it keeps
# a validation pass to a few minutes instead of a few hours. Cell 3 asserts which one you got.
TEST_QUESTIONS  = None           # None -> the FULL 80,019-question published test split
SMOKE_TEST_N    = 4000           # SMOKE only

# max_state_tokens: 1024 is bev-decider's TRAINING limit. 512 halves the cost and covers ~92% of
# rows unchanged (p99 = 958 tokens); do not raise it for the fair comparison.
MAX_STATE_TOKENS = 1024
MAX_CHOICE_TOKENS = 64

# leave these blank and the notebook prints NOT AVAILABLE / asks you, rather than inventing numbers
ON_DEMAND_PRICE_PER_HOUR = None   # e.g. 1.126  (ml.g5.2xlarge, us-east-1)
SPOT_PRICE_PER_HOUR = None                 # e.g. 0.378
# ----------------------------------------------------------------------------

# Re-running cell 1 must not orphan a previous run in S3, so reuse the stored run name if there is
# one.  Delete decider_v2_state.json to start a genuinely new run.
_state_file = os.path.join(os.path.dirname(os.path.abspath(".")), "decider_v2_state.json")
_state_file = os.path.join(os.getcwd(), "decider_v2_state.json")
_prev = json.load(open(_state_file)) if os.path.exists(_state_file) else {}
RUN_NAME = _prev.get("run_name") or f"{PREFIX}-{time.strftime('%Y%m%d-%H%M%S')}"
print(f"RUN_NAME = {RUN_NAME}" + ("   (reused from decider_v2_state.json)" if _prev.get("run_name") else ""))
LOCAL = os.path.abspath("decider_v2_work")      # every artefact the notebook writes lives here
SRC_DIR = os.path.abspath("decider_v2_src")      # the training sources, also used as source_dir
os.makedirs(LOCAL, exist_ok=True)
if not os.path.isdir(SRC_DIR):
    raise SystemExit(f"{SRC_DIR} not found. Run the '0-sources' cell at the top of this notebook "
                     "first -- it embeds the training sources and extracts them here. If you are "
                     "running cells out of order, re-run cell 0 then this one.")
SEEDS = [int(s) for s in SEEDS]
if not SEEDS:
    raise SystemExit("SEEDS is empty; there is nothing to run")
print(json.dumps({k: v for k, v in dict(
    BUCKET=BUCKET, PREFIX=PREFIX, REGION=REGION, INSTANCE_TYPE=INSTANCE_TYPE, SMOKE=SMOKE,
    FRAMEWORK_VERSION=FRAMEWORK_VERSION, PY_VERSION=PY_VERSION,
    MAX_STEPS=MAX_STEPS_REAL, BATCH_SIZE=BATCH_SIZE, GRAD_ACCUM=GRAD_ACCUM,
    TARGET_PRESENTATIONS=TARGET_PRESENTATIONS, RECIPE_QPS=RECIPE_QPS,
    MAX_RUNTIME_HOURS=MAX_RUNTIME_HOURS,
    SPLIT_SEED=SPLIT_SEED, SEEDS=SEEDS, SELFTEST=SELFTEST, ENV_CHECK=ENV_CHECK,
    RUN_BASELINE=RUN_BASELINE, RUN_V2=RUN_V2, RUN_ABLATIONS=RUN_ABLATIONS, USE_SPOT=USE_SPOT,
    RUN_NAME=RUN_NAME).items()}, indent=2))

# The estimator will not start if this (framework_version, py_version) pairing does not exist, so
# check it here before any money is spent. The SDK is the source of truth; the literal table is a
# fallback for environments where the SDK cannot answer offline (and is verified against it in
# LOCAL_VERIFICATION.md).
VALID_PYTORCH = {"2.5.1": ["py311"], "2.6.0": ["py312"], "2.7.1": ["py312"]}
try:
    from sagemaker import image_uris
    image_uris.retrieve(framework="pytorch", region=REGION or "us-east-1",
                        version=FRAMEWORK_VERSION, py_version=PY_VERSION, image_scope="training")
    print(f"container ok (SDK): pytorch {FRAMEWORK_VERSION} / {PY_VERSION}")
except ImportError:
    ok = PY_VERSION in VALID_PYTORCH.get(FRAMEWORK_VERSION, [])
    print(f"container check via built-in table: pytorch {FRAMEWORK_VERSION} / {PY_VERSION} "
          f"-> {'ok' if ok else 'UNKNOWN PAIRING'}")
    if not ok:
        raise SystemExit(f"no SageMaker container for pytorch {FRAMEWORK_VERSION} / {PY_VERSION}. "
                         f"Known pairings: {VALID_PYTORCH}")
except Exception as e:
    raise SystemExit(f"no SageMaker container for pytorch {FRAMEWORK_VERSION} / {PY_VERSION} "
                     f"({type(e).__name__}: {e}). Valid pairings are 2.5.1/py311, 2.6.0/py312, "
                     f"2.7.1/py312.")

n_jobs = len(SEEDS) * (int(RUN_BASELINE) + int(RUN_V2))
print(f"\\nthis run will launch {n_jobs} training job(s): "
      f"{int(RUN_BASELINE) * len(SEEDS)} baseline + {int(RUN_V2) * len(SEEDS)} v2, "
      f"each {MAX_STEPS:,} steps x {ACTUAL_QPS} questions.")

if SMOKE:
    SMOKE_KW = dict(smoke_train=4000, smoke_val=600, smoke_calib=1200, smoke_test=SMOKE_TEST_N)
    MAX_STEPS = 100
    assert TEST_QUESTIONS is None, "SMOKE uses its own shard; leave TEST_QUESTIONS = None"
    print(f"SMOKE mode: subset {SMOKE_KW}, MAX_STEPS -> {MAX_STEPS}")
    print("SMOKE test = the 4,000-question shard. A real run scores the FULL 80,019-question split.")
else:
    SMOKE_KW = dict(smoke_train=0, smoke_val=0, smoke_calib=0, smoke_test=0)
    print("REAL RUN: test = the FULL published test split, never subsampled.")
print("local workdir:", LOCAL)

# What this job actually presents. Printed AFTER the SMOKE block below has settled MAX_STEPS, so
# the numbers describe the job the estimator will really receive rather than the pre-override value.
PRESENTATIONS = MAX_STEPS * ACTUAL_QPS
print(f"\\nbudget per job: {MAX_STEPS:,} steps x {ACTUAL_QPS} questions/step "
      f"= {PRESENTATIONS:,} question presentations")
if SMOKE:
    print(f"  SMOKE: MAX_STEPS is pinned to 100 for the plumbing check. The REAL budget is")
    print(f"  {MAX_STEPS_REAL:,} steps x {RECIPE_QPS} = {TARGET_PRESENTATIONS:,} presentations")
else:
    print(f"  target {TARGET_PRESENTATIONS:,}; "
          f"{'matched' if PRESENTATIONS >= TARGET_PRESENTATIONS else 'SHORT OF'} the card's ~940K")
    _train_n = 388748      # cell 3 prints the real number from the manifest
    print(f"  = {PRESENTATIONS / _train_n:.2f} epochs over the "
          f"{_train_n:,}-question carved train split")
print(f"  max_runtime {MAX_RUNTIME_HOURS} h per job -- SageMaker KILLS a job that exceeds this and")
print("  the work is lost. Cell 6b checks this against your instance's measured throughput.")
if len(SEEDS) < 2:
    print("\\n  NOTE: SEEDS has 1 entry. The decision rule cannot return anything above")
    print("  NOT SEPARABLE with one seed (the seed spread is 0 by construction). That is fine for")
    print("  a first run -- this is the cheapest way to learn whether 940K is affordable -- but")
    print("  do not quote the v2-baseline delta from it. Add seeds with SEEDS = [0, 1], then [0, 1, 2].")

''')

# ---------------------------------------------------------------- 2 install
md("2. Install / imports",
   "Pins live in `decider_v2/requirements.txt` and are enforced by `decider_v2/environment.py`, which "
   "the notebook, `train.py` and `evaluate.py` all call. That matters because the project's "
   "correctness depends on a transformers behaviour (4-D attention masks are returned as-is only "
   "from the 5.x line), so a silently-different transformers is a correctness bug, not a warning. "
   "If the image already has these, this cell is a no-op.")

code("2-install", '''
import json, os, subprocess, sys

# One source of truth for the pins: decider_v2/requirements.txt, enforced by decider_v2/environment.py.
sys.path.insert(0, os.path.abspath("decider_v2_src"))
import environment

REQ = os.path.join(SRC_DIR, "requirements.txt")
if not os.path.exists(REQ):
    raise SystemExit(f"{REQ} not found -- it ships with the sources and pins every version below.")

# REPORT ONLY -- never pip-install into this notebook kernel.
#
# `environment.enforce()` defaults to allow_install=True, which is correct inside a throwaway
# SageMaker TRAINING container (fresh image every job) but catastrophic here: installing
# numpy 2.3.5 / pyarrow 25 into the notebook's live interpreter upgrades the C extensions
# underneath every already-imported module, and the kernel dies with
#   "pyarrow.lib.Array size changed, may indicate binary incompatibility. Expected 80 from
#    C header, got 72"
# while pip simultaneously reports autogluon / sparkmagic / sagemaker-studio / gluonts as
# broken. The notebook kernel is a long-lived Studio instance that other work depends on; the
# pins belong in the job container, which is disposable and gets them via --env_check=1.
ok = environment.enforce(requirements=REQ, allow_install=False, prefix="ENV")
print()
if not ok:
    print("NOTE: this notebook's kernel does not match the pins. That is EXPECTED and harmless --")
    print("the kernel only orchestrates (download, carve, upload, launch). The versions that")
    print("matter are the ones INSIDE each training container, which enforces the pins itself via")
    print("--env_check=1 and ABORTS the job if they are wrong. Continuing.")
    print()
    print("Do NOT run `environment.py --install` here. It upgrades the Studio kernel in place and")
    print("breaks it (see the note above). If you want a matching kernel, start a fresh SageMaker")
    print("notebook instance from a container image that already carries the pins.")

# REPORT ONLY, and every import is guarded. The Studio kernel is not the pinned interpreter and
# can carry combinations that fail AT IMPORT TIME rather than merely version-mismatching -- e.g.
# datasets 2.2.1 against pyarrow 21 dies with
#   AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'
# (PyExtensionType was removed in pyarrow 15). When that happens an unguarded
# `import torch, transformers, datasets, ...` aborts THIS cell before the boto3/SESSION/BUCKET
# lines below it, so a diagnostic block takes down the orchestration the notebook needs. Report
# what imports, name what did not, and carry on: the kernel only orchestrates.
def _try_import(name):
    try:
        return __import__(name)
    except Exception as _e:
        print(f"{name:15s} IMPORT FAILED ({type(_e).__name__}: {_e}) -- harmless here, the job "
              f"container installs and enforces its own pins via --env_check=1.")
        return None

torch = _try_import("torch")
transformers = _try_import("transformers")
datasets = _try_import("datasets")
peft = _try_import("peft")
huggingface_hub = _try_import("huggingface_hub")
numpy = _try_import("numpy")

# `torch` above is bound by the guarded _try_import("torch"), which returns None if it cannot import.
if torch is not None:
    _gpu = ""
    if torch.cuda.is_available():
        _gpu = " " + torch.cuda.get_device_name(0)
    print(f"torch        {torch.__version__}   cuda={torch.cuda.is_available()}{_gpu}")
    if torch.cuda.is_available():
        print(f"gpu memory   {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
        print(f"capability   {torch.cuda.get_device_capability(0)}")
if numpy is not None:
    print(f"numpy        {numpy.__version__}")

import boto3
from botocore.exceptions import ClientError
SESSION = boto3.Session(region_name=REGION or None)

# Resolve the execution role WITHOUT sagemaker.get_execution_role().
#
# That helper was deprecated in SDK v2 and REMOVED in v3, so `from sagemaker import
# get_execution_role` raises ImportError on any current install. The reliable, SDK-independent
# path is the notebook instance's own credentials: boto3's STS caller identity IS the role, and it
# is exactly the role SageMaker will use to run the training jobs. So read it from STS.
def resolve_role():
    if ROLE_NAME:
        print(f"role         {ROLE_NAME}   (from ROLE_NAME)")
        return ROLE_NAME
    ident = SESSION.client("sts").get_caller_identity()
    arn = ident.get("Arn", "")
    acct = ident.get("Account", "")
    # A SageMaker notebook instance assumes a role, so STS returns
    #   arn:aws:sts::<acct>:assumed-role/<RoleName>/<instance-id>
    # Splitting that on ":" puts the assumed-role resource in index 4, NOT the account, so the
    # account must come from GetCallerIdentity's Account field (index 5 would be wrong here).
    # SageMaker's `role` parameter needs the IAM role ARN:
    #   arn:aws:iam::<acct>:role/<RoleName>
    # An IAM user ARN (running outside a notebook) is already usable as-is.
    if ":assumed-role/" in arn:
        role_name = arn.split(":assumed-role/", 1)[1].split("/", 1)[0]
        resolved = f"arn:aws:iam::{acct}:role/{role_name}"
        print(f"role         {resolved}   (derived from this instance's assumed role {role_name!r})")
        return resolved
    print(f"role         {arn}   (not an assumed role; using as-is)")
    return arn

ROLE = resolve_role()
print(f"sagemaker    {getattr(__import__('sagemaker'), '__version__', 'unknown')}")
print(f"caller       {SESSION.client('sts').get_caller_identity().get('Arn')}")

BUCKET = BUCKET or f"decider-v2-{SESSION.client('sts').get_caller_identity()['Account']}-{REGION or SESSION.region_name}"
assert BUCKET, "set BUCKET at the top of this cell"
assert ROLE and ROLE.startswith("arn:aws:iam::"), (
    f"could not resolve an IAM role ARN (got {ROLE!r}). Set ROLE_NAME in cell 1 to the "
    f"arn:aws:iam::<account>:role/<name> form explicitly.")

# The default BUCKET above is a NAME this notebook invented, so on a fresh account it does not exist
# and the upload dies with NoSuchBucket -- after a HeadObject 404 that looks like a missing key rather
# than a missing bucket. Create it instead, and prove it is reachable before any cell depends on it.
#
# Idempotent: if the bucket already exists this is a no-op, and us-east-1 is special-cased because
# CreateBucket rejects an explicit LocationConstraint there (the global endpoint is the only one).
#
# Note the 403-vs-404 distinction below. A bucket that exists but is owned by someone else answers
# AccessDenied, which must NOT be papered over by trying to create it -- that would fail confusingly
# and, worse, could mask a typo'd BUCKET that points at a real bucket you do not own.
def ensure_bucket(name, region):
    s3 = SESSION.client("s3")
    region = region or SESSION.region_name or "us-east-1"
    try:
        s3.head_bucket(Bucket=name)
        print(f"bucket       {name}   (exists)")
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        # 403 means it exists and we cannot use it. 404 means it genuinely is not there.
        if code in ("403", "Forbidden") or status == 403:
            raise SystemExit(
                f"bucket {name!r} exists but this role cannot access it "
                f"({code or status}). Either point BUCKET at a bucket you own, or grant the role "
                f"s3:GetBucketLocation/s3:ListBucket/s3:PutObject on it.")
        # 301 means it exists in a DIFFERENT region. Treating that as "absent" would try to create a
        # bucket whose name is already taken elsewhere, and S3 would answer with a redirect that
        # surfaces much later as a confusing HeadObject failure on every channel.
        if code in ("301", "PermanentRedirect") or status == 301:
            raise SystemExit(
                f"bucket {name!r} exists but not in {region}. SageMaker reads channels from the "
                f"notebook's own region, so the bucket must live there too. Either set REGION in "
                f"cell 1 to the bucket's region, or set BUCKET to a bucket in {region}.")
    print(f"bucket       {name}   (does not exist -- creating it in {region})")
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=name)
        else:
            s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": region})
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise SystemExit(f"could not create bucket {name!r}: {code} -- "
                             f"{e.response.get('Error', {}).get('Message', '')}. "
                             f"Create it manually and re-run this cell, or set BUCKET in cell 1.")
    # Prove it before anything downstream depends on it. Guarded, because create_bucket can legitimately
    # report success (or BucketAlreadyOwnedByYou) while a head_bucket still answers 403 for a role that
    # cannot read bucket metadata -- an unguarded call here would end in a bare traceback instead of the
    # one sentence that tells you which permission is missing.
    try:
        s3.head_bucket(Bucket=name)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        raise SystemExit(
            f"bucket {name!r} was created but head_bucket still fails ({code or status}). The role "
            f"{ROLE} needs s3:ListBucket (and s3:GetBucketLocation) on it. Grant that, or point BUCKET "
            f"at a bucket the role already owns.")
    print(f"bucket       {name}   (created)")
    return True


ensure_bucket(BUCKET, REGION)

# ---------------------------------------------------------------- bucket access
#
# `head_bucket` succeeding proves almost nothing: a role can read bucket metadata via s3:ListBucket
# and still be denied every object operation. That is the default SageMaker-ExecutionRole situation,
# and it produced AccessDenied on the very first upload with a 120-line botocore traceback above it.
#
# So prove the three things this notebook actually does -- put, get, delete -- with a tiny probe
# object. It is cheap, and it turns a three-cells-later traceback into one line here.
#
# If the probe fails, the notebook ATTACHES the missing policy itself when the caller is allowed to
# (iam:PutRolePolicy). Notebook instance roles usually are, since they are admin-adjacent; when they
# are not, the printed policy is exact enough to paste into the console.
def _s3_probe(name):
    s3 = SESSION.client("s3")
    key = f"{PREFIX}/_access_probe.txt"
    body = b"decider-v2 access probe"
    try:
        s3.put_object(Bucket=name, Key=key, Body=body)
        got = s3.get_object(Bucket=name, Key=key)["Body"].read()
        s3.delete_object(Bucket=name, Key=key)
        if got != body:
            return False, "read back different bytes"
        return True, "put/get/delete all succeed"
    except ClientError as e:
        return False, (f"{e.response.get('Error', {}).get('Code', '')} "
                       f"{e.response.get('Error', {}).get('Message', '')}")


def _role_name_from_arn(arn):
    """arn:aws:iam::<acct>:role[/path]/<Name> -> <Name>. iam:PutRolePolicy wants the bare name."""
    return arn.split(":role/", 1)[1].split("/")[-1]


def _attach_s3_policy(name):
    """Attach an inline policy granting this role object access to `name`. Best effort by design."""
    policy_name = "decider-v2-s3-access"
    doc = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "DeciderV2BucketAccess",
            "Effect": "Allow",
            # The full set the notebook and the training jobs need: channels are read, checkpoints
            # and metrics are written, and multipart uploads matter because questions.parquet is
            # large enough to exceed the 100 MB single-part limit for the train shard.
            "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject",
                       "s3:ListBucket", "s3:GetBucketLocation",
                       "s3:AbortMultipartUpload", "s3:ListBucketMultipartUploads"],
            "Resource": [f"arn:aws:s3:::{name}", f"arn:aws:s3:::{name}/*"],
        }],
    }
    iam = SESSION.client("iam")
    try:
        iam.put_role_policy(RoleName=_role_name_from_arn(ROLE), PolicyName=policy_name,
                            PolicyDocument=json.dumps(doc))
        print(f"attached inline policy {policy_name!r} to {_role_name_from_arn(ROLE)}")
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        print(f"could not attach the policy automatically ({code}). Attach it by hand:")
        print(json.dumps({policy_name: doc}, indent=2))
        return False


_ok, _why = _s3_probe(BUCKET)
if not _ok:
    print(f"\\nrole {ROLE} cannot use s3://{BUCKET} yet ({_why}).")
    print("attempting to grant it ...")
    if _attach_s3_policy(BUCKET):
        _ok, _why = _s3_probe(BUCKET)
if not _ok:
    # The role cannot modify itself (no iam:PutRolePolicy), which is the normal case: a SageMaker
    # execution role is deliberately not self-administering. Attaching the policy is therefore a
    # console action, not a notebook one. Before sending the user to the console, try the thing that
    # almost always works without any IAM change at all: use a bucket this role ALREADY has access
    # to. SageMaker creates one per domain/studio and grants the execution role write access, so on a
    # stock setup there is a perfectly good bucket sitting there already.
    print(f"\\ncannot grant the role access from here ({_why}).")
    print("looking for a bucket this role can already write to ...")
    _cands = []
    try:
        _cands = [b["Name"] for b in SESSION.client("s3").list_buckets().get("Buckets", [])]
    except ClientError as e:
        print(f"  list_buckets denied too "
              f"({e.response.get('Error', {}).get('Code', '')}); skipping the search.")
    _found = None
    for _b in _cands[:25]:                      # bounded: some accounts have thousands of buckets
        _o, _w = _s3_probe(_b)
        if _o:
            _found = _b
            break
    if _found:
        print(f"\\nusing s3://{_found} instead -- this role can already read and write it.")
        print("Set BUCKET in cell 1 to that name to make this permanent.")
        BUCKET = _found
        _ok, _why = True, "inherited from an existing grant"
if not _ok:
    _acct = SESSION.client("sts").get_caller_identity()["Account"]
    raise SystemExit(
        f"role {_role_name_from_arn(ROLE)} cannot write to any usable S3 bucket, and cannot grant\\n"
        f"itself access. This is a one-time console action -- the notebook cannot do it for you.\\n\\n"
        f"  1. Open https://console.aws.amazon.com/iam/home#/roles/{_role_name_from_arn(ROLE)}\\n"
        f"  2. Permissions -> Add inline policy -> JSON, and paste this (replace the bucket name if you\\n"
        f"     set BUCKET to something other than the default):\\n"
        + json.dumps({"Version": "2012-10-17", "Statement": [{
            "Sid": "DeciderV2BucketAccess",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject",
                       "s3:ListBucket", "s3:GetBucketLocation",
                       "s3:AbortMultipartUpload", "s3:ListBucketMultipartUploads"],
            "Resource": [f"arn:aws:s3:::{BUCKET}", f"arn:aws:s3:::{BUCKET}/*"]}]}, indent=2)
        + f"\\n\\n  3. Re-run this cell. IAM changes propagate in seconds; if it still fails, wait a\\n"
          f"     minute and retry.\\n\\n"
          f"That one role covers BOTH sides: it is the notebook's role (which uploads the channels)\\n"
          f"and the training job's role (which reads them and writes checkpoints).")
print(f"bucket access {BUCKET}: OK ({_why})")
# ---------------------------------------------------------------- the local interpreter
#
# `sys.executable` is THIS kernel, and on a Studio image it is not the pinned interpreter. Worse than a
# version mismatch, some combinations fail at IMPORT time -- the image this was written against pairs
# datasets 2.2.1 with pyarrow 21, and `from datasets import load_dataset` raises
#   AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'
# (removed in pyarrow 15) which no version check can see coming. So every local subprocess launched
# with sys.executable inherits the breakage and prepare.py cannot even carve the splits.
#
# Installing the pins into the kernel is not the fix (see the note above -- it upgrades numpy/pyarrow
# under every already-imported module and breaks the Studio instance). So build a THROWAWAY venv under
# the work directory and run the local subprocesses there. It sees the instance's site-packages, so
# torch stays the CUDA-matched build the instance provides; only the pinned set is overlaid.
#
# One-time cost, then idempotent: the stamp is keyed on a hash of requirements.txt, so re-running this
# cell costs nothing and editing the pins transparently rebuilds.
import venv_env
VENV_DIR = os.path.join(LOCAL, "venv")
LOCAL_PY = venv_env.ensure(VENV_DIR, REQ, force=REBUILD_VENV)
if not venv_env.describe(LOCAL_PY, REQ):
    raise SystemExit("the local venv does not satisfy requirements.txt. Fix it there, or delete "
                     f"{VENV_DIR} and re-run this cell. The kernel itself is deliberately untouched.")

# A satisfied pin table is necessary but NOT sufficient, and this cell already failed twice in the
# gap between the two. Both failures passed the version check:
#   * pyarrow reports "21.0.0" happily while `import datasets` raises AttributeError
#     (PyExtensionType was removed in pyarrow 15);
#   * on this CONDUA image torch reports a version while `import torch` dies with
#     "OSError: libmkl_intel_lp64.so.2: cannot open shared object file", because the venv subprocess
#     never gets the conda activation that put the prefix's lib directory on the loader path.
#
# patch_env() fixes the second one for every later subprocess by putting the parent's library
# directories on the search path in THIS process, so children inherit it; the explicit `env=LOCAL_ENV`
# on each subprocess below is belt-and-braces for the ones that pass their own env.
LOCAL_ENV = venv_env.patch_env()
if not venv_env.verify_runtime(LOCAL_PY):
    raise SystemExit("the pinned interpreter cannot import torch/transformers/datasets -- see above.")
print(f"local subprocess interpreter: {LOCAL_PY}")
print("every local step below (prepare / tests / bench / calibrate / evaluate) runs on that one.")
''')

code("2b-state-helper", STATE)

# ---------------------------------------------------------------- 3 dataset
md("3. Load the full dataset, carve the splits, upload to S3",
   "`load_dataset('avbiswas/bev-decision', 'all')`, both splits, no subsampling outside SMOKE mode. "
   "The published `test` split is **never** touched by training, early stopping or calibration. "
   "`calib` and `val` are carved out of `train` **before** training, stratified by question type × "
   "domain. The parity check in cell 3d prints the measured state overlap so you can see it is 0.")

code("3-load", '''
# Guarded for the same reason as the 2-install cell: the Studio kernel is not the pinned
# interpreter, and a bad pair (datasets 2.2.1 + pyarrow 21) fails at import, not at version check.
# These two cells are DIAGNOSTIC -- they print the raw row counts and the Phase-0 statistics.
# The actual carving happens in the 3-prepare cell, which runs prepare.py in a subprocess that
# enforces the pins itself, so losing the in-kernel view must not stop the pipeline.
import json, os, sys, collections
try:
    import numpy as np
    from datasets import load_dataset
except Exception as _e:
    print(f"SKIPPED the in-kernel dataset view: {type(_e).__name__}: {_e}")
    print("This cell only prints statistics. 3-prepare carves the splits in a subprocess that")
    print("installs the pins, so the run continues. To see these stats, start a notebook")
    print("instance whose image already carries requirements.txt.")
    ds = None
if ds is not None:
    sys.path.insert(0, os.path.join(os.getcwd(), "decider_v2_src"))
    CACHE = os.path.join(LOCAL, "hf_cache")
    os.makedirs(CACHE, exist_ok=True)
    print("loading avbiswas/bev-decision config='all' (both splits, no subsampling) ...")
    ds = load_dataset("avbiswas/bev-decision", "all", cache_dir=CACHE)
    print(ds)
    print("raw row counts:", {k: len(v) for k, v in ds.items()})
    print("features:", ds["train"].features)
else:
    CACHE = os.path.join(LOCAL, "hf_cache")
    os.makedirs(CACHE, exist_ok=True)
''')

code("3-stats", '''
# Same statistics as Phase 0 of ARCH_REVIEW.md, recomputed here from the loaded object.
import json, collections
if ds is None:
    print("SKIPPED: 3-load could not import datasets in this kernel (see that cell for why).")
    print("These are the Phase-0 dataset statistics; they do not affect training or scoring.")
else:
    from transformers import AutoTokenizer

    type_q, domain_q = collections.Counter(), collections.Counter()
    n_opts, levels, key_counts = collections.Counter(), collections.Counter(), collections.Counter()
    state_chars, instr_chars, opt_chars = [], [], []
    no_label = empty_criteria = label_probs = 0
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    state_toks, opt_toks = [], []
    for split in ds:
        for row in ds[split]:
            domain_q[row["domain"]] += 1
            state_chars.append(len(row["state"]))
            st = len(tok.encode(row["state"]))
            state_toks.append(st)
            for q in json.loads(row["questions_json"]).values():
                type_q[q["type"]] += 1
                instr_chars.append(len(q.get("instructions", "")))
                if "label" not in q:
                    no_label += 1
                if "label_probs" in q:
                    label_probs += 1
                t = q["type"]
                if t == "choice":
                    crit = q["criteria"]
                    n_opts[len(crit)] += 1
                    if not crit:
                        empty_criteria += 1
                    for k, v in crit.items():
                        key_counts[k] += 1
                        opt_chars.append(len(f"{k}: {v}" if v else k))
                        opt_toks.append(len(tok.encode(f"{k}: {v}" if v else k)))
                elif t == "score":
                    levels[len(q["criteria"])] += 1
                    for i, lv in enumerate(q["criteria"]):
                        opt_chars.append(len(f"{i}: {lv}"))
                        opt_toks.append(len(tok.encode(f"{i}: {lv}")))
                else:
                    n_opts[2] += 1
                    opt_chars += [2, 3]
                    opt_toks += [1, 1]

    total = sum(type_q.values())
    print(f"typed questions         {total:,}")
    print(f"by type                 " + "  ".join(f"{k}={v:,} ({v/total*100:.2f}%)"
                                                    for k, v in sorted(type_q.items())))
    print(f"questions with no label {no_label}   choice with empty criteria {empty_criteria}   "
          f"with label_probs {label_probs:,}")
    print(f"score level counts      {dict(sorted(levels.items()))}")
    print(f"options/question        {dict(sorted(n_opts.items()))}")
    print(f"distinct choice keys    {len(key_counts):,}   top: {key_counts.most_common(5)}")


    def pct(a, name):
        a = np.asarray(a)
        print(f"{name:22s} p50={np.percentile(a,50):8.0f} p90={np.percentile(a,90):8.0f} "
              f"p95={np.percentile(a,95):8.0f} p99={np.percentile(a,99):8.0f} "
              f"max={a.max():8d} mean={a.mean():8.0f}")


    pct(state_chars, "state chars"); pct(instr_chars, "instructions chars")
    pct(state_toks, "state tokens"); pct(opt_toks, "option tokens")
    print(f"rows with state > {MAX_STATE_TOKENS} tokens: "
          f"{sum(1 for s in state_toks if s > MAX_STATE_TOKENS):,} / {len(state_toks):,}")
    print(f"domains ({len(domain_q)}):")
    for d, n in domain_q.most_common():
        print(f"   {d:55s} {n:9,d}")
''')

code("3-prepare", '''
# Explode rows -> one record per typed question, then carve train / val / calib / test.
# SPLIT_SEED is fixed for the whole run: it carves the splits AND samples the SMOKE test shard, so
# every arm and every seed trains and is scored on byte-identical data.
import subprocess
subprocess.check_call([LOCAL_PY, os.path.join(SRC_DIR, "prepare.py"),
                       "--out", os.path.join(LOCAL, "data"), "--seed", str(SPLIT_SEED),
                       "--calib_target", "20000", "--val_frac", "0.03",
                       "--cache_dir", CACHE, "--force"]
                      + [f"--{k}={v}" for k, v in SMOKE_KW.items() if v], env=LOCAL_ENV)
print(open(os.path.join(LOCAL, "data", "manifest.json")).read()[:1400])
''')

code("3-parity", '''
# Fair-comparison assertions, printed rather than assumed.
m = json.load(open(os.path.join(LOCAL, "data", "manifest.json")))
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

# THE assertion that keeps a real run honest: the full 80,019-question test split, or a shard that
# is only allowed under SMOKE.  n=4,000 carries a +/-1.35 point 95% interval at p~0.76, which is
# larger than any improvement this project could claim, so the shard is a smoke-only convenience.
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
                             f"Either SMOKE=True or fix the prepare.py shard argument.")

print("  split sizes:", m["sizes"])
for name in ("train", "val", "calib", "test"):
    s = m["stats"][name]
    n = s["questions"]
    print(f"  {name:6s} n={n:7,d}  " + "  ".join(f"{k}={v/n*100:5.2f}%" for k, v in sorted(s["by_type"].items())))
assert ok, "split parity failed -- do not launch training"
TEST_N = n_test
save_state(manifest=m, data_dir=os.path.join(LOCAL, "data"), test_n=TEST_N, split_seed=SPLIT_SEED)
print(f"\\nTEST_N = {TEST_N:,}  ({'SMOKE shard' if SMOKE else 'FULL published test split'})")
''')

code("3-upload", '''
# Re-establish the handful of globals this cell needs. A kernel restart (or running this cell
# out of order) leaves the namespace empty and the bare `SESSION.client(...)` fails with a
# NameError that says nothing about which earlier cell is missing. Everything here is cheap and
# side-effect free, so re-deriving it is always safe; if a value really cannot be derived the
# error names the cell to re-run instead.
import json as _json
if "SESSION" not in globals():
    print("(restoring globals from decider_v2_state.json -- earlier cells were not in this kernel)")
    _sf = os.path.join(os.getcwd(), "decider_v2_state.json")
    if not os.path.exists(_sf):
        raise SystemExit(f"{_sf} not found. Run cells 0-5 first (they define BUCKET/PREFIX/RUN_NAME/LOCAL "
                         f"and the boto3 session).")
    _prev = _json.load(open(_sf))
    PREFIX   = _prev.get("data_prefix", "").split("/")[0] or "decider_v2"
    RUN_NAME = _prev.get("run_name") or os.path.basename(_prev.get("data_prefix", "").rstrip("/").split("/")[0])
    LOCAL    = os.path.join(os.getcwd(), "decider_v2_work")
    import boto3 as _boto3
    REGION   = REGION if "REGION" in globals() else (_prev.get("region") or "")
    SESSION  = _boto3.Session(region_name=REGION or None)
    BUCKET   = BUCKET or (_prev.get("bucket") or f"decider-v2-{SESSION.client('sts').get_caller_identity()['Account']}-{REGION or SESSION.region_name}")
    if "save_state" not in globals():
        def save_state(**kw):
            _prev.update(kw)
            with open(_sf, "w") as fh:
                _json.dump(_prev, fh, indent=2)
            print("state ->", _sf + ":", sorted(_prev))

# Upload the four shards as SageMaker input channels. Idempotent: re-running skips existing keys.
for _split in ("train", "val", "calib", "test"):
    if not os.path.exists(os.path.join(LOCAL, "data", _split, "questions.parquet")):
        raise SystemExit(f"{LOCAL}/data/{_split}/questions.parquet is missing. Re-run the 3-prepare cell "
                         f"that carved the splits before uploading them.")
s3 = SESSION.client("s3")
DATA_PREFIX = f"{PREFIX}/{RUN_NAME}/data"
uploaded = []
for split in ("train", "val", "calib", "test"):
    key = f"{DATA_PREFIX}/{split}/questions.parquet"
    src = os.path.join(LOCAL, "data", split, "questions.parquet")
    # Only a 404 means "not uploaded yet". Swallowing every exception here is what turned a missing
    # BUCKET into a misleading NoSuchBucket on PutObject, three stack frames deep, instead of the
    # one line that names the actual problem.
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        print(f"  exists, skipping {key}")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code not in ("404", "NoSuchKey", "NotFound") and status != 404:
            raise SystemExit(f"cannot read s3://{BUCKET}/{key}: {code or status} -- "
                             f"{e.response.get('Error', {}).get('Message', '')}")
        s3.upload_file(src, BUCKET, key)
        print(f"  uploaded {key}  ({os.path.getsize(src) / 2**20:.1f} MiB)")
    uploaded.append(key)
save_state(data_prefix=DATA_PREFIX, channels={s: f"{DATA_PREFIX}/{s}" for s in
                                              ("train", "val", "calib", "test")})
''')

# ---------------------------------------------------------------- 4 source
md("4. Write the source files",
   "If `decider_v2/` sits next to this notebook it is copied verbatim; otherwise the cell writes the "
   "files from the `decider_v2_src/` tree. Either way `source_dir` is the same directory that gets "
   "tarred and uploaded as the SageMaker source.")

code("4-source", '''
import glob
files = (sorted(glob.glob(os.path.join(SRC_DIR, "*.py")))
         + sorted(glob.glob(os.path.join(SRC_DIR, "bench", "*.py")))
         + sorted(glob.glob(os.path.join(SRC_DIR, "*.txt"))))
for f in files:
    print(f"  {os.path.relpath(f, SRC_DIR):26s} {os.path.getsize(f):>7,d} B")
print("source_dir =", SRC_DIR)
# requirements.txt ships with the sources precisely so the container can enforce the same pins.
import environment
print(f"pins the job will enforce: {[f'{n}=={v}' for n, v in environment.parse_requirements()]}")
save_state(source_dir=SRC_DIR)
''')

# ---------------------------------------------------------------- 5 tests
md("5. Run tests.py -- the notebook stops here if anything fails",
   "(a) packed-vs-separate equivalence under the 4-D additive mask and shared position ids, "
   "(b) option-order invariance including `score`, (c) loss sanity, (d) a 64-example overfit. "
   "Runs on the notebook's own GPU; falls back to CPU and says so.")

code("5-tests", '''
import subprocess, sys, torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"running tests.py on {dev}")
rc = subprocess.call([LOCAL_PY, os.path.join(SRC_DIR, "tests.py"), "--device", dev],
                     cwd=SRC_DIR, env=LOCAL_ENV)
if rc != 0:
    raise SystemExit(f"tests.py FAILED (exit {rc}) -- fix before launching any job")
print("tests.py passed")
''')

# ---------------------------------------------------------------- 6 smoke
md("6. In-notebook smoke test: throughput, and the cost of the real run",
   "Times a handful of real training steps on this notebook's GPU through the exact `train.py` "
   "entry point, then extrapolates hours and cost for `MAX_STEPS`. Pricing is fetched at runtime "
   "from the AWS Price List API when available; if that fails it prints what you must fill in "
   "rather than guessing.")

code("6-smoke", '''
# Times real training steps on THIS notebook's GPU through the exact same model code, then hands
# the numbers to cell 6b.  Reported separately: end-to-end (tokenizer in loop, i.e. what a
# SageMaker job will actually see) and GPU-only (pre-encoded batches, the compute ceiling).
import sys, json, subprocess, torch

# A CPU-only Studio instance is a real and common configuration, and benching on it produces a number
# that is not merely imprecise but actively misleading: it would make the 940K budget look affordable
# by orders of magnitude, when it is priced against a GPU instance this notebook never touches. So say
# so plainly and skip the extrapolation rather than print a fake one.
_HAVE_GPU = torch.cuda.is_available()
if not _HAVE_GPU:
    print("=" * 78)
    print("NO GPU ON THIS NOTEBOOK INSTANCE -- skipping the throughput bench.")
    print("=" * 78)
    print("Cells 6 and 6b measure how fast training runs so the real budget can be costed.")
    print("On a CPU-only instance that number is meaningless: it would make the 940K-presentation")
    print("budget look affordable when it is actually priced against a GPU instance this notebook")
    print("never touches.")
    print()
    print("This does NOT block the run. Training happens in SageMaker training jobs on their own")
    print("GPU instances; this notebook only uploads data and launches them. What you lose is the")
    print("cost estimate in cell 8. To get it, restart the notebook instance on a GPU instance")
    print("(SageMaker Studio -> the instance's settings -> change instance type) and re-run this cell.")
    print()
    print("Continuing: cells 4-selftest and 7 will also run on CPU, which is slower but correct.")
    BENCH = None
    save_state(bench=None, gpu_available=False)
else:
    BENCH_PATH = os.path.join(LOCAL, "bench.json")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Run in a SUBPROCESS on purpose: a CUDA context in this kernel would hold VRAM that the later
    # evaluate.py subprocesses need, and on a small GPU that is the difference between running and
    # CUDA-out-of-memory.
    subprocess.check_call([LOCAL_PY, os.path.join(SRC_DIR, "bench.py"),
                           "--train_channel", os.path.join(LOCAL, "data", "train"),
                           "--num_layers", "20", "--head", "v2", "--lora_r", "16",
                           "--lora_include_mlp", "1", "--batch_size", str(min(BATCH_SIZE, 8)),
                           "--micro_steps", "6", "--max_state_tokens", str(MAX_STATE_TOKENS),
                           "--gradient_checkpointing", "1", "--num_workers", "2",
                           "--out", BENCH_PATH], env=LOCAL_ENV)
    BENCH = json.load(open(BENCH_PATH))
    print("\\nbench result:", json.dumps(BENCH, indent=2))
    save_state(bench=BENCH)
    print(f"peak VRAM measured: {BENCH['peak_vram_gib']:.2f} GiB of "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
    if BENCH["peak_vram_gib"] > 0.9 * torch.cuda.get_device_properties(0).total_memory / 2**30:
        print("  WARNING: this GPU is nearly full at this micro-batch. Reduce --batch_size here,")
        print("  or the training job on a larger instance may not need to -- but do not read this")
print("  instance's VRAM as the constraint on the real run.")
''')

code("6-estimate", '''
# Extrapolate hours + cost for the real run from the throughput measured in cell 6.
# Pricing is queried from the AWS Price List API at runtime; if that fails it says so rather than
# guessing, and you can fill the two price constants in cell 1.
import torch
BENCH = STATE.get("bench")
N_ARMS = len(SEEDS) * (int(RUN_BASELINE) + int(RUN_V2))
QUESTIONS = PRESENTATIONS      # MAX_STEPS x ACTUAL_QPS, computed in cell 1
print(f"questions per job: {QUESTIONS:,}  ({MAX_STEPS:,} steps x {ACTUAL_QPS})")
print(f"jobs to run:       {N_ARMS}  ({len(SEEDS)} seeds x "
      f"{int(RUN_BASELINE) + int(RUN_V2)} arms)")


def price(instance_type, region, spot):
    try:
        import boto3
        c = boto3.client("pricing", region_name="us-east-1")
        f = {"family": instance_type.split(".")[0]}
        if not spot:
            f["locationType"] = "AWS Region"
            f["regionCode"] = (region or SESSION.region_name)
        for page in c.get_paginator("get_products").paginate(
                ServiceCode="AmazonEC2", Filters=[f], MaxResults=100):
            for p in page["PriceList"]:
                pr = json.loads(p)
                for term in pr.get("terms", {}).get("OnDemand", {}).values():
                    for d in term["priceDimensions"].values():
                        if instance_type in d["description"]:
                            usd = float(d["pricePerUnit"]["USD"])
                            return usd * (10 if spot else 1)   # price list lists spot as 10x on-demand
    except Exception as e:
        print(f"  pricing lookup failed: {type(e).__name__}: {e}")
    return None


on_demand = ON_DEMAND_PRICE_PER_HOUR or price(INSTANCE_TYPE, REGION, False)
spot_price = SPOT_PRICE_PER_HOUR or (price(INSTANCE_TYPE, REGION, True) if USE_SPOT else None)
print(f"price per hour  on-demand: {on_demand if on_demand else 'NOT RESOLVED -- fill it in cell 1'}")
print(f"price per hour  spot    : {spot_price if spot_price else 'NOT RESOLVED -- fill it in cell 1'}")

if not BENCH:
    print("\\nTHROUGHPUT_NOT_CAPTURED -- run cell 6 first (it is the previous cell).")
else:
    # v2 carries LoRA on attention AND MLP for all 20 layers; the bev-replica config measured 8.6 q/s
    # against v2's 6.5 q/s on an RTX 4050, i.e. the wider adapter costs 1.32x.  Apply that ratio.
    v2_qps = BENCH["gpu_only_qps"]
    base_qps = v2_qps * 1.32
    e2e_v2 = BENCH["end_to_end_qps"]
    e2e_base = e2e_v2 * 1.32
    print(f"\\nmeasured on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}: "
          f"v2 GPU-only {v2_qps:.2f} q/s, end-to-end {e2e_v2:.2f} q/s, peak {BENCH['peak_vram_gib']:.2f} GiB")
    for arm, gpu, e2e in (("baseline replica", base_qps, e2e_base), ("decider-v2", v2_qps, e2e_v2)):
        for tag, qps in (("compute ceiling", gpu), ("end-to-end", e2e)):
            hours = QUESTIONS / qps / 3600
            for label, pr in (("on-demand", on_demand), ("spot", spot_price)):
                if pr:
                    print(f"  {arm:18s} {tag:15s} {qps:6.2f} q/s  {hours:7.2f} h/job  "
                          f"{label:9s} ${hours * pr:8.2f}/job")
    print(f"\\n{len(SEEDS)} seed(s) per arm -> multiply the per-job figures above by {N_ARMS} for the")
    print("whole run (both arms, every seed, at the same budget).")
    print("\\nThese are THIS notebook's GPU numbers. The SageMaker job will differ; re-run cell 6 on")
    print("the training instance (or set INSTANCE_TYPE to the type you will actually use) before")
    print("trusting the figure.")
''')

code("6b-runtime-check", '''
# The one check that can destroy a long run: SageMaker KILLS a training job that exceeds
# max_runtime and the work is lost. This compares the budget against a throughput measurement and
# refuses to pass silently if the job would be killed before it finishes.
#
# WHERE THE THROUGHPUT CAME FROM matters as much as the number. Cell 6 runs `bench.py` in a
# SUBPROCESS OF THIS NOTEBOOK KERNEL, so it measures THIS notebook's GPU -- which is almost never
# the training instance. A max_runtime sized from the wrong card is a job that gets killed.
#  - source="kernel"  : measured on this notebook. Fine for a spot check, NOT authoritative.
#  - source="job"     : parsed from a real training job's CloudWatch log (see read_job_qps below).
#                       This is the training instance and is what the guard should use.
THROUGHPUT_SOURCE = "kernel"     # "kernel" | "job"
JOB_THROUGHPUT_FROM = ""          # job name to parse, e.g. "baseline_s0" or a raw SageMaker name


def read_job_qps(job_name):
    """Throughput from a REAL training job's CloudWatch log, i.e. the training instance itself.

    train.py logs `sps=<n>` on every log line, so the median of a finished job's lines is its
    steady-state questions/second. Returns None if the log cannot be read, so the caller can fall
    back to the kernel measurement with a loud warning rather than guessing.
    """
    try:
        import boto3
        log_group = f"/aws/sagemaker/TrainingJobs"
        start = time.time() - 14 * 24 * 3600
        ce = boto3.client("logs", region_name=REGION or None)
        streams = ce.describe_log_streams(logGroupName=log_group, logStreamNamePrefix=job_name,
                                         startTime=int(start * 1000), orderBy="LastEventTime",
                                         descending=True)["logStreams"]
        if not streams:
            return None
        events = ce.get_log_events(logGroupName=log_group,
                                   logStreamName=streams[0]["logStreamName"],
                                   startTime=int(start * 1000), limit=20000)["events"]
        sps = []
        for e in events:
            for tok in str(e.get("message", "")).split():
                if tok.startswith("sps="):
                    try:
                        sps.append(float(tok[4:]))
                    except ValueError:
                        pass
        if len(sps) < 3:
            return None
        sps.sort()
        # steady state: drop the first third (warm-up), take the median of the rest
        steady = sps[len(sps) // 3:]
        return {"qps": steady[len(steady) // 2], "n_samples": len(sps),
                "source": "job", "job": job_name}
    except Exception as e:
        print(f"  could not read CloudWatch for {job_name}: {type(e).__name__}: {e}")
        return None


_b = BENCH or {}          # BENCH is None when the notebook has no GPU (cell 6 skipped the bench)
_hw = _b.get("hardware", {})
print("=" * 78)
print("WHERE THIS THROUGHPUT CAME FROM")
print("=" * 78)
if not _b:
    print("  NO MEASUREMENT -- cell 6 found no GPU on this notebook instance and skipped the bench.")
    print("  Everything below that needs a throughput number is therefore unavailable, and this cell")
    print("  will NOT invent one. That is deliberate: a CPU-derived estimate would make the budget")
    print("  look affordable when it is priced against a GPU training instance.")
else:
    print(f"  measured on : {_hw.get('gpu_name', 'unknown')} "
          f"({_hw.get('gpu_total_gib', '?')} GiB, CC {_hw.get('compute_capability', '?')})")
    print(f"  torch       : {_hw.get('torch', '?')}   micro-batch {_b.get('batch_size', '?')}   "
          f"max_state_tokens {_b.get('max_state_tokens', '?')}")
    print(f"  head/LoRA   : head={_b.get('head', '?')} lora_r={_b.get('lora_r', '?')} "
          f"mlp={_b.get('lora_include_mlp', '?')}   (matches v2, not the baseline arm)")

QPS = None
if THROUGHPUT_SOURCE == "job" and JOB_THROUGHPUT_FROM:
    r = read_job_qps(JOB_THROUGHPUT_FROM)
    if r:
        QPS = r["qps"]
        print(f"\\n  USING REAL JOB THROUGHPUT from {r['job']}: {QPS:.2f} q/s "
              f"(median of {r['n_samples']} log lines, warm-up discarded)")
        print("  This is the TRAINING INSTANCE, so the max_runtime estimate below is authoritative.")
    else:
        print("\\n  *** no job throughput available -- falling back to the kernel measurement, which")
        print("  *** is NOT the training instance. Treat the max_runtime estimate as unreliable.")
        QPS = BENCH["end_to_end_qps"] if BENCH else None

if QPS is None and BENCH:
    if not SMOKE:
        print("\\n  WARNING: this throughput is from the NOTEBOOK KERNEL's GPU. The training job runs")
        print(f"  on {INSTANCE_TYPE}, which is a different machine with a different CPU count, storage")
        print("  and network. A max_runtime sized from this card can be far too short.")
        print(f"  Set THROUGHPUT_SOURCE='job' and JOB_THROUGHPUT_FROM='<a finished job name>' once you")
        print("  have one real job, or run cell 6 ON the training instance for an exact number.")
    QPS = BENCH["end_to_end_qps"]

_hours = MAX_RUNTIME / 3600
print("\\n" + "=" * 78)
print("max_runtime GUARD")
print("=" * 78)
print(f"  max_runtime for every job: {MAX_RUNTIME_HOURS} h ({MAX_RUNTIME:,} s)")
print(f"  budget per job:            {MAX_STEPS:,} steps x {ACTUAL_QPS} = {QUESTIONS:,} presentations")
if QPS is None:
    # Reachable two ways: no GPU here so cell 6 skipped the bench, or the bench failed. They need
    # DIFFERENT responses, and "run cell 6 first" is actively wrong on a CPU-only instance, because
    # running it again will never produce a number.
    print("\\n  THROUGHPUT NOT AVAILABLE -- the max_runtime guard cannot be evaluated.")
    if STATE.get("bench") is None and not STATE.get("gpu_available", True):
        print("  Cause: this notebook instance has no GPU, so cell 6 skipped the bench.")
        print("  Running it again will not help. Three ways forward:")
        print(f"    1. Launch anyway -- MAX_RUNTIME_HOURS = {MAX_RUNTIME_HOURS} h is the budgeted")
        print("       maximum for this recipe and the documented estimate is ~13-40 h per job, so")
        print("       it should clear. SageMaker KILLS a job that exceeds it and the run is lost,")
        print("       so raise it if you have any doubt; a longer limit costs nothing until used.")
        print("    2. Launch one SMOKE job, let it finish, then set")
        print("         THROUGHPUT_SOURCE = 'job'")
        print("         JOB_THROUGHPUT_FROM = '<that job name>'")
        print("       and re-run this cell to size max_runtime from the real training instance.")
        print("    3. Restart this notebook on a GPU instance and re-run cell 6.")
    else:
        print("  Cause: cell 6 has no measurement in the state file. Run cell 6 first.")
        print("  Do not launch a real run without knowing the job will fit in max_runtime.")
else:
    _need = []
    for arm, qps in (("baseline replica", QPS * 1.32), ("decider-v2", QPS)):
        h = QUESTIONS / qps / 3600
        flag = "OK" if _hours - h > 0 else "TOO SHORT -- RAISE MAX_RUNTIME_HOURS"
        print(f"  {arm:18s} {qps:6.2f} q/s -> {h:7.2f} h  vs {MAX_RUNTIME_HOURS} h limit  {flag}")
        if _hours - h <= 0:
            _need.append((arm, h))
    if _need:
        print("\\n*** These jobs would be KILLED before finishing. SageMaker discards the run. ***")
        for arm, h in _need:
            print(f"    {arm}: needs >= {h * 1.3:.0f} h -- set MAX_RUNTIME_HOURS = {int(h * 1.3) + 1}")
        print("    Or lower TARGET_PRESENTATIONS. Do not launch until this is resolved.")
        if not SMOKE:
            raise SystemExit("max_runtime is too short for this budget; refusing to launch.")
    else:
        src_note = ("training instance" if THROUGHPUT_SOURCE == "job" and JOB_THROUGHPUT_FROM
                    else "NOTEBOOK KERNEL (not the training instance)")
        print(f"\\n  max_runtime has headroom for every arm. Source: {src_note}.")
        print("  These are END-TO-END estimates including tokenisation; the first minutes of a real")
        print("  job also pay to download the dataset and the Qwen3 weights, so keep some slack.")

print()
print("reference figures (RTX 4050 Laptop, 6 GiB, batch 8 x 2) -- the SLOWEST plausible card:")
print("  baseline replica 8.6 q/s (peak 2.87 GiB)   decider-v2 6.5 q/s (peak 3.11 GiB)")
print("  the wider LoRA (attn+MLP, all layers) costs 1.32x, which is the ratio cell 6b applies")
print("  micro-batch 8 at 1,024-token states peaked at 3.22 GiB; micro-batch 16 did not fit in 6 GiB")
print("  so on a 24 GB A10G (ml.g5.2xlarge) micro-batch 32 should fit, but MEASURE it, do not assume")
''')

# ---------------------------------------------------------------- 7 reference
md("7. Reference eval: the published bev-decider-0.4B on MY test split",
   "Context for the results table: what the reference architecture scores on *our* test file. It is "
   "**not** part of the v2-vs-baseline decision, which reads only the trained arms, so it is the "
   "first thing to skip when the machine is small or slow.\n\n"
   "`FULL_REFERENCE = False` (the default) scores the test shard only. Setting it True adds the "
   "calibration pass and the invariance gate -- better numbers, but on a CPU-only notebook that is "
   "~25,000 forward passes and it has taken the Jupyter app down with OOM before.")

code("7-reference", '''
# Download the published checkpoint (no HF token needed; set HF_TOKEN if you hit a rate limit).
import torch
env = dict(os.environ)
if not env.get("HF_TOKEN") and not env.get("HUGGING_FACE_HUB_TOKEN"):
    print("no HF_TOKEN in the environment -- the model is public, but you may hit a rate limit")

# Why this is off by default. The full reference pass is 4,000 test questions + 21,127 calib
# questions + an invariance sweep, and load_reference_bev_decider peaks around 4 GB while loading
# (970 MB checkpoint + an fp32 copy + a freshly initialised model). On a CPU-only Studio instance
# that combination OOM-killed the Jupyter app outright -- "App is unhealthy because of continuous
# failed health checks" -- taking the notebook with it. The test-only pass is ~6x cheaper and gives
# the headline accuracy, which is all the context column needs.
FULL_REFERENCE = False

_ref_calib = os.path.join(LOCAL, "data", "calib") if FULL_REFERENCE else ""
_ref_inv = "1" if FULL_REFERENCE else "0"
if not FULL_REFERENCE:
    print("FULL_REFERENCE=False -> scoring the test split only (no calib pass, no invariance sweep).")
    print("Set FULL_REFERENCE = True above for temperature/ECE, on a GPU instance if possible.")

subprocess.check_call([LOCAL_PY, os.path.join(SRC_DIR, "evaluate.py"),
                       "--reference", "avbiswas/bev-decider-0.4B",
                       "--test_channel", os.path.join(LOCAL, "data", "test"),
                       "--calib_channel", _ref_calib,
                       "--out", os.path.join(LOCAL, "eval"),
                       "--batch_size", "4" if not torch.cuda.is_available() else "8",
                       "--invariance", _ref_inv, "--invariance_examples", "24",
                       "--invariance_shuffles", "8",
                       "--predictions_out", os.path.join(LOCAL, "eval", "predictions_reference.csv"),
                       "--tag", "reference",
                        # Report only. --env_check 1 would default to allow_install=True and
                        # pip-install the pins into this notebook kernel, upgrading numpy/pyarrow
                        # underneath every already-imported module and breaking the Studio
                        # instance. The pins are enforced inside the job container instead.
                        "--env_check", "0"], cwd=SRC_DIR, env=env)
''')

code("7b-reference-default", '''
# The published 74.7 is on the `default` config's test split. This cell measures both so the results
# table cannot be misread. (A 1,000-question draw once suggested `all` was ~8.5 points harder; the
# 4,000-question measurement disproved that, so the shard here is only a protocol reproduction and
# the headline comparison stays on OUR full test split.)
sys.path.insert(0, SRC_DIR)
try:
    from data import explode_questions
    from datasets import load_dataset
except Exception as _e:
    print(f"SKIPPED: this kernel cannot import datasets ({type(_e).__name__}: {_e}).")
    print("It only reproduces the PUBLISHED 74.7 number on the `default` config. Our own test")
    print("split is unaffected -- see the 7a cell and the results table.")
    explode_questions = load_dataset = None
import random as _random

if load_dataset is not None:

    out_dir = os.path.join(LOCAL, "data", "test_default")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(os.path.join(out_dir, "questions.parquet")):
        d = load_dataset("avbiswas/bev-decision", "default", cache_dir=CACHE)
        t = d["test"].add_column("row_index", list(range(len(d["test"]))))
        t = t.map(explode_questions, batched=True, batch_size=1000, remove_columns=d["test"].column_names,
                  cache_file_name=os.path.join(CACHE, "test_default.map"))
        print(f"default-config test questions: {len(t):,}")
        ids = sorted(_random.Random(SPLIT_SEED).sample(range(len(t)), min(5000, len(t))))
        t.select(ids).to_parquet(os.path.join(out_dir, "questions.parquet"))
    subprocess.check_call([LOCAL_PY, os.path.join(SRC_DIR, "evaluate.py"),
                           "--reference", "avbiswas/bev-decider-0.4B",
                           "--test_channel", out_dir, "--out", os.path.join(LOCAL, "eval"),
                           "--batch_size", "8", "--invariance", "0", "--tag", "reference_default",
                            "--env_check", "0"],
                      cwd=SRC_DIR, env=LOCAL_ENV)
''')

# ---------------------------------------------------------------- 8/9 jobs
md("8. Launch the baseline replica",
   "Same data, split, seed, tokenizer, truncation limits, collate, steps, batch and eval code as v2. "
   "The only differences are the four rows in the ARCH_V2.md comparison table. One job is launched "
   "per seed in `SEEDS`; all of them share `SPLIT_SEED`, so they train on identical data. Check the "
   "instance-type service quota before launching.")

code("8-quota", '''
# SageMaker training jobs are gated by a per-region, per-instance-type quota. There is no single
# quota code that covers every instance family, so the console path below is the reliable route.
from sagemaker import Session as SMSession
sm = SMSession(boto_session=SESSION)
family = INSTANCE_TYPE.split(".")[0]
region = REGION or SESSION.region_name
big = any(k in INSTANCE_TYPE for k in ("p4d", "p5", "p4de"))
need = 8 if big else 1

print(f"Instance type in use: {INSTANCE_TYPE}")
print(f"Region:               {region}")
print()
print("Check your current quota (read-only, safe to run):")
print("  aws service-quotas list-service-quotas --service-code sagemaker --region " + region + " \\\\")
# Built by concatenation so no nested-quote escaping survives into the notebook.
_q = '      --query ' + chr(34) + "Quota[?contains(QuotaName, '" + family + "')]"
_q += ".[QuotaName,Value,QuotaCode]" + chr(34) + " --output table"
print(_q)
print()
print("If that value is 0 (or lower than instance_count), request an increase in the CONSOLE:")
print("  1. Open the Service Quotas console:")
print("       https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas")
print("  2. Sign in to the " + region + " console (the quota is per-region).")
print("  3. Filter Service = 'Amazon SageMaker AI', then search for:")
print("       - 'ML instances for training job usage' / '" + INSTANCE_TYPE + "'")
print("  4. Select the matching row -> 'Request quota increase'.")
print("  5. Request at least " + str(need) + " instance(s); SageMaker approves requests of this")
print("     size automatically, usually within minutes.")
print()
print("Equivalent CLI (get the QuotaCode from the list command above first -- do not guess it):")
print("  aws service-quotas get-service-quota --service-code sagemaker --quota-code <QuotaCode> --region " + region)
print("  aws service-quotas request-service-quota-change --service-code sagemaker \\\\")
print('      --quota-code <QuotaCode> --desired-value ' + str(need) + " --region " + region + " \\\\")
print('      --support-case-description "increase for decider-v2 ' + INSTANCE_TYPE + ' training jobs"')
print()
print("A NEW SageMaker account also starts with an 'ml.<family> for training job usage' quota of 0,")
print("so this step is required for almost every first run, not just when a job fails.")
''')

code("8-launch-baseline", '''
from sagemaker.pytorch import PyTorch

def hyper(seed, **over):
    h = dict(
        max_state_tokens=MAX_STATE_TOKENS, max_choice_tokens=MAX_CHOICE_TOKENS,
        num_layers=20, mode="bev", pool="bev",
        # --- bev-decider's architecture, reproduced exactly ---
        head="v1", head_new_dim=512, head_layers=2,
        lora_r=8, lora_alpha=16, lora_last_k_layers=12, lora_include_mlp=0,
        # --- bev-decider's optimisation, reproduced exactly ---
        label_smoothing=0.0, emd_lambda=0.0, type_weights="",
        lr_lora=1e-4, lr_head=1e-3, warmup_ratio=0.05, weight_decay=0.01, grad_clip=1.0,
        shuffle_options=0, gradient_checkpointing=1,
        # --- shared budget ---
        batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM, max_steps=MAX_STEPS, epochs=2,
        # 8 dataloader workers is right on a SageMaker box; SMOKE uses a 4k-question subset, so 2
        # is plenty there and avoids the Windows/spawn worker crash (exit 0xC0000409) locally.
        seed=seed, num_workers=2 if SMOKE else 8,
        log_every=max(5, MAX_STEPS // 60), eval_every=max(20, MAX_STEPS // 4),
        save_every=max(20, MAX_STEPS // 4),
        max_train_minutes=-1,
        # preflight gates, run inside the container at job start
        selftest=int(bool(SELFTEST)), env_check=int(bool(ENV_CHECK)),
    )
    h.update({k: str(v) for k, v in over.items()})
    return [f"--{k}={v}" for k, v in h.items()]


def budget_fingerprint(args):
    """The training budget a job was launched with, for the idempotency guard.

    Recorded per job in decider_v2_state.json. A Completed job is NOT skipped if the budget it ran
    under differs from the one now configured: silently reusing a 768K-presentation checkpoint in a
    run that believes it is training 940K is the worst possible failure, because everything then
    looks like it worked.
    """
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
    """Human-readable list of the budget fields that differ between two fingerprints."""
    out = []
    for f in ("max_steps", "batch_size", "grad_accum_steps", "presentations",
              "target_presentations", "seed"):
        if f in prev and prev[f] != now[f]:
            out.append(f"{f}: launched {prev[f]} -> now {now[f]}")
    return out


def launch(name, args, channels, wait=False):
    key = f"jobs/{name}"
    now = budget_fingerprint(args)
    prev = (STATE.get(key) or {}).get("budget")
    if STATE.get(key) and STATE[key].get("status") in ("Completed", "Failed", "Stopped"):
        # Refuse to reuse a job that ran a DIFFERENT budget. This is the check that stops a stale
        # checkpoint from silently being evaluated against a differently-trained set.
        if prev is not None and prev != now:
            diff = describe_mismatch(prev, now)
            print(f"  {name}: REFUSING to skip -- it completed under a different budget:")
            for d in diff:
                print(f"      {d}")
            print(f"      Its checkpoints were trained on {prev.get('presentations', '?')} "
                  f"presentations; this run is configured for {now['presentations']}.")
            print(f"      Either restore the old config, or change RUN_NAME to start a fresh run, or")
            print(f"      delete jobs/{name} from decider_v2_state.json to relaunch it deliberately.")
            raise SystemExit(f"job {name} completed under a different training budget")
        print(f"  {name}: already {STATE[key]['status']} -- not relaunching (idempotent, same budget)")
        return STATE[key]["name"]

    ckpt_s3 = f"s3://{BUCKET}/{PREFIX}/{RUN_NAME}/{name}/checkpoints"
    ckpt_local = "SM_MODEL_DIR (SageMaker mounts it under /opt/ml/checkpoints)"
    max_wait = SPOT_MAX_WAIT_MINUTES * 60 if USE_SPOT else None
    print(f"  launching {name}: instance={INSTANCE_TYPE} spot={USE_SPOT} "
          f"max_run={MAX_RUNTIME_HOURS}h max_wait={(f'{max_wait // 60}h' if max_wait else 'n/a')} "
          f"budget={now['max_steps']:,}x{now['batch_size']}x{now['grad_accum_steps']}"
          f"={now['presentations']:,} presentations")
    print(f"    checkpoint_s3_uri      = {ckpt_s3}")
    print(f"    checkpoint_local_path = {ckpt_local}")

    est = PyTorch(
        entry_point="train.py", source_dir=SRC_DIR, role=ROLE,
        framework_version=FRAMEWORK_VERSION, py_version=PY_VERSION,
        instance_type=INSTANCE_TYPE, instance_count=1,
        output_path=f"s3://{BUCKET}/{PREFIX}/{RUN_NAME}/{name}/output",
        checkpoint_s3_uri=ckpt_s3,
        hyperparameters={"sagemaker_args": " ".join(args)},
        input_channels=channels, volume_size=200, keep_alive_period_in_seconds=600,
        use_spot=USE_SPOT, max_runtime=MAX_RUNTIME,
    )
    if USE_SPOT:
        # SageMaker requires MaxWaitTime >= MaxRuntime for spot training; the SDK raises otherwise,
        # and a spot job that waits 40 h for capacity and then gets evicted is worse than on-demand.
        est.configure(WaitTimeConfig={"MaxWaitTime": max_wait})
    est.fit(wait=wait, logs="None")
    print(f"    -> SageMaker job {est._latest_job_name}")
    save_state(**{key: {"name": est._latest_job_name, "status": "InProgress",
                        "output": f"{PREFIX}/{RUN_NAME}/{name}/output",
                        "checkpoints": f"{PREFIX}/{RUN_NAME}/{name}/checkpoints",
                        "budget": now,
                        "instance_type": INSTANCE_TYPE,
                        "use_spot": bool(USE_SPOT),
                        "max_run_seconds": MAX_RUNTIME,
                        "max_wait_seconds": max_wait,
                        "framework_version": FRAMEWORK_VERSION,
                        "py_version": PY_VERSION}})
    return est._latest_job_name


CHANNELS = {k: f"s3://{BUCKET}/{v}" for k, v in STATE["channels"].items()}
JOBS = {}          # arm -> [seed, ...] actually launched, used by cells 10-14
for seed in SEEDS:
    if RUN_BASELINE:
        launch(f"baseline_s{seed}", hyper(seed), CHANNELS)
        JOBS.setdefault("baseline", []).append(seed)
    else:
        print(f"RUN_BASELINE is False -- skipping baseline seed {seed}")
print(f"baseline jobs: {JOBS.get('baseline', [])}  "
      f"(all on SPLIT_SEED={SPLIT_SEED}, so identical data)")
''')

md("9. Launch decider-v2",
   "Identical budget and data. Tier 1 only: Perceiver ctx summarizer, rank embedding, LoRA r=16 on "
   "q/k/v/o **and** MLP for all 20 kept layers, label smoothing 0.05 with the valid-options fix, "
   "CE + 0.5·EMD on `score`, per-epoch option shuffling, lr 2e-4 / 1e-3, 3% warmup.")

md("9b. Running the seeds: seed 0 first, both arms concurrently",
   "SageMaker bills by instance-hour, so the two arms of one seed cost the same whether they run "
   "back-to-back or at the same time — and running them concurrently halves wall-clock. "
   "`est.fit(wait=False)` is already the default in cell 8, so **launching seed 0 of both arms "
   "back-to-back in cells 8 and 9 already runs them concurrently**: cell 8 submits "
   "`baseline_s0` and returns immediately, cell 9 submits `v2_s0` and returns immediately, and "
   "both train side by side. Nothing extra is required.\\n\\n"
   "The recommended sequence, which also gives you an early separability read at half the cost of "
   "the full grid:\\n\\n"
   "1. **Seed 0, both arms, concurrently.** Set `SEEDS = [0]` in cell 1 and run cells 8 and 9. "
   "Two jobs start together and finish together. `launch()` is idempotent, so re-running these "
   "cells never double-launches a job whose state is `Completed`.\\n"
   "2. While they run, set `WAIT = True` in cell 10 and let it poll until both are `Completed`.\\n"
   "3. Run cell 11 to evaluate both checkpoints, then cell 13. The `## Decision` section will say "
   "`NOT EVALUATED` or report a delta against a zero seed spread — with one seed there is no "
   "spread at all, so the verdict is necessarily `NOT SEPARABLE`. Launch the other seeds before "
   "believing anything.\\n"
   "4. **Seed 1, optionally.** Set `SEEDS = [0, 1]` in cell 1 and run cells 8 and 9 again. "
   "Seed 0 is already `Completed` in `decider_v2_state.json` and is skipped; only "
   "`baseline_s1` and `v2_s1` are submitted, concurrently with each other. Then re-run cells "
   "10-13. The verdict will be **PROVISIONAL** at best: two seeds give a spread that is a range "
   "between a single pair of runs, and the same configuration measured 0.05 pp in one dry run and "
   "2.52 pp in another.\\n"
   "5. **Seed 2.** Set `SEEDS = [0, 1, 2]` and re-run cells 8 and 9. Only `*_s2` is submitted. This "
   "is the first point at which `SEPARABLE` is possible, because the decision rule requires three "
   "seeds per arm. Keep adding seeds for as long as the verdict says `NOT SEPARABLE`. The spread "
   "is a max over arms, so more seeds can only make the bar harder to clear - which is the point.\\n\\n"
   "To run the arms **one after another** instead (e.g. to stay inside a strict instance "
   "concurrency quota), set `wait=True` in the `launch()` call in cell 8/9; the second job then "
   "blocks until the first finishes. Same cost, twice the wall-clock.\\n\\n"
   "Whatever the order, every arm and every seed trains on the same `SPLIT_SEED` data, so a "
   "staged launch is exactly equivalent to launching everything at once.\\n\\n"
   "> The offline dry-run harness (`dryrun_notebook.py`) runs each job as a local subprocess and "
   "so runs them **serially**, not concurrently; the job ordering, hyperparameter strings and S3 "
   "layout it exercises are identical either way. Concurrency itself is a SageMaker scheduling "
   "property and has not been executed here (`LOCAL_VERIFICATION.md` §0).")

code("9-launch-v2", '''
V2_OVER = dict(
    head="v2", head_new_dim=512, head_layers=2, head_heads=8, n_ctx=8,
    use_ctx=1, use_rank=1, use_null=0, use_derived=0,
    lora_r=16, lora_alpha=32, lora_last_k_layers=20, lora_include_mlp=1,
    label_smoothing=0.05, emd_lambda=0.5, type_weights="",
    lr_lora=2e-4, lr_head=1e-3, warmup_ratio=0.03, shuffle_options=1,
)
for seed in SEEDS:
    if RUN_V2:
        launch(f"v2_s{seed}", hyper(seed, **V2_OVER), CHANNELS)
        JOBS.setdefault("v2", []).append(seed)
    else:
        print(f"RUN_V2 is False -- skipping v2 seed {seed}")
print(f"v2 jobs: {JOBS.get('v2', [])}")
print("\\nSubmission is non-blocking (est.fit(wait=False)), so with SEEDS=[0] cells 8 and 9 have")
print("started baseline_s0 and v2_s0 CONCURRENTLY and this cell has already returned. Set WAIT=True")
print("in cell 10 to block until they finish, then evaluate; see the cell-9b notes for the")
print("staged seed 0 -> seed 1 workflow.")

# The longer run arch.md §13.2 proposes, as a documented second option rather than the default.
LONG_RUN_OVER = dict(V2_OVER, batch_size=64, grad_accum_steps=2, max_steps=25000)
print("\\nLonger second run (not launched): --batch_size=64 --grad_accum_steps=2 --max_steps=25000")
print("  " + " ".join(hyper(SEEDS[0], **LONG_RUN_OVER)))
print("  NOTE: batch 64 x 1,024-token states needs >= 40 GB of VRAM -- use ml.p4d.24xlarge or")
print("        ml.g5.48xlarge, and re-measure throughput in cell 6 before trusting the estimate.")
save_state(jobs_by_arm=JOBS)
''')

md("10. Monitor",
   "Both jobs write `metrics.jsonl` to `SM_OUTPUT_DATA_DIR`, i.e. `.../output/data/metrics.jsonl` "
   "under the run prefix. The trainer prints `step=... loss=... val_acc=...` on one line per log "
   "interval, which is also what the CloudWatch metric definitions below match.")

code("10-monitor", '''
import time, glob, json

def job_status(name):
    try:
        d = sm.describe_training_job(TrainingJobName=STATE[f"jobs/{name}"]["name"])
        return d["TrainingJobStatus"]
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def _download_prefix(prefix, local):
    bucket = boto3.resource("s3", region_name=REGION or None).Bucket(BUCKET)
    os.makedirs(local, exist_ok=True)
    n = 0
    for obj in bucket.objects.filter(Prefix=prefix):
        tail = obj.key.split("/", 3)[3:]      # strip <PREFIX>/<run>/<job>/<output|checkpoints>/
        target = os.path.join(local, *tail)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        bucket.download_file(obj.key, target)
        n += 1
    print(f"  downloaded {n} objects from s3://{BUCKET}/{prefix}/")
    return glob.glob(os.path.join(local, "**", "*"), recursive=True)


def download_outputs(name):
    """SageMaker writes metrics to output_path and checkpoints to checkpoint_s3_uri, which are
    DIFFERENT S3 prefixes -- both have to be pulled before anything can be evaluated."""
    j = STATE.get(f"jobs/{name}")
    if not j:
        return []
    files = _download_prefix(j["output"], os.path.join(LOCAL, "downloads", name, "output"))
    if j.get("checkpoints"):
        files += _download_prefix(j["checkpoints"],
                                  os.path.join(LOCAL, "downloads", name, "checkpoints"))
    return files


def job_names():
    """Every <arm>_s<seed> job this run launched, in a stable order."""
    return sorted(k[len("jobs/"):] for k in STATE if k.startswith("jobs/"))


def arm_of(job_name):
    """'v2_s0' -> 'v2', 'v2_lora_bev_s0' -> 'v2_lora_bev'. rsplit on the LAST '_s' so an arm name
    containing '_s' cannot be truncated; the seed must be digits, so 'jobs/...' keys never match."""
    arm, _sep, seed = job_name.rpartition("_s")
    return arm if seed.isdigit() else job_name


WAIT = False     # set True to block until every job finishes
todo = job_names()
print(f"monitoring {len(todo)} job(s): {todo}")
while True:
    done = True
    for n in todo:
        st = job_status(n)
        files = download_outputs(n)
        js = [f for f in files if f.endswith("metrics.jsonl")]
        print(f"  {n:16s} {st:12s} metrics_lines={sum(1 for _ in open(js[0])) if js else 0}")
        if st in ("InProgress", "Starting", "Downloading", "Stopping"):
            done = False
    if done or not WAIT:
        break
    time.sleep(120)

for n in todo:
    save_state(**{f"jobs/{n}": {**STATE[f"jobs/{n}"], "status": job_status(n)}})
''')

code("10-plot", '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
# one line per (arm, seed) so a seed effect is visible rather than averaged away
palette = {"baseline": "tab:blue", "v2": "tab:orange"}
for name in job_names():
    arm = arm_of(name)
    colour = palette.get(arm, "tab:green")
    files = [f for f in download_outputs(name) if f.endswith("metrics.jsonl")]
    if not files:
        continue
    recs = [json.loads(l) for l in open(files[0])]
    tr = [r for r in recs if "loss" in r]
    va = [r for r in recs if "val_acc" in r]
    axes[0].plot([r["step"] for r in tr], [r["loss"] for r in tr], color=colour,
                 label=name, alpha=0.85)
    axes[1].plot([r["step"] for r in va], [r["val_acc"] for r in va], "o-", color=colour,
                 label=name, ms=3, alpha=0.85)
axes[0].set(xlabel="step", ylabel="train loss", title="training loss")
axes[1].set(xlabel="step", ylabel="val accuracy", title="validation accuracy (checkpoint selection)")
for a in axes:
    a.grid(alpha=.3); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(LOCAL, "training_curves.png"), dpi=120)
print("wrote", os.path.join(LOCAL, "training_curves.png"))
''')

md("11. Calibrate then evaluate, on the FULL test split",
   "Temperatures are fitted on `calib` (carved from train before training, never used for early "
   "stopping). Evaluation always uses `test`, which nothing in the pipeline has seen -- in a real "
   "run that is all 80,019 questions, never a shard. The order-invariance gate runs on the trained "
   "v2 weights. One row per (arm, seed).")

code("11-calibrate-evaluate", '''
# every (arm, seed) job launched in cells 8/9/14, evaluated on the identical test file
ARM_ARGS = {"baseline": lambda s: hyper(s), "v2": lambda s: hyper(s, **V2_OVER)}
ABL_ARGS = STATE.get("ablation_args", {})

# This is the one step that genuinely needs a GPU and cannot be moved into a training job: it reads
# the downloaded checkpoints and scores them, here. On a CPU-only Studio instance that is correct but
# slow -- tens of minutes per arm on the 4,000-question SMOKE shard, and hours per arm on the full
# 80,019-question real split. Say so up front rather than letting it look hung.
import torch as _t
if not _t.cuda.is_available():
    print("=" * 78)
    print("NO GPU ON THIS NOTEBOOK -- evaluation will run on CPU.")
    print("=" * 78)
    print(f"Each arm scores {TEST_N:,} test questions plus the calib split and an invariance sweep.")
    print("On CPU that is slow but CORRECT -- the numbers are identical, just later. Let it run;")
    print("do not kill it. To speed it up, restart the notebook on a GPU instance (ml.g5.2xlarge or")
    print("ml.p3.2xlarge) and re-run this cell; it skips arms already evaluated.")
    print()

evaluated = []
# Skip arms already scored. On a GPU this is a convenience; on a CPU-only notebook it is the
# difference between minutes and hours, because re-scoring an arm throws away a completed pass.
# Both files must exist: metrics_*.json is written last, and a partial run must not count as done.
_done = set(STATE.get("evaluated", []))
for name in job_names():
    arm = arm_of(name)
    _metrics = os.path.join(LOCAL, "eval", name, f"metrics_{name}.json")
    _preds = os.path.join(LOCAL, "eval", name, f"predictions_{name}.csv")
    if name in _done and os.path.exists(_metrics) and os.path.exists(_preds):
        print(f"skipping {name}: already evaluated (metrics and predictions both present)")
        evaluated.append(name)
        continue
    print("=" * 78); print(name, f"(arm={arm})"); print("=" * 78)
    local = os.path.join(LOCAL, "downloads", name)
    ckpt, fallback = None, None
    for root, _dirs, files in os.walk(local):
        if "meta.json" in files and "model.safetensors" in files:
            fallback = fallback or root
            if os.path.basename(root) == "best":     # selected on the VALIDATION split
                ckpt = root
    ckpt = ckpt or fallback
    assert ckpt, (f"no checkpoint downloaded for {name} (under {local}). Check that "
                  f"checkpoint_s3_uri is set on the estimator and that the job actually saved one.")
    print("using checkpoint:", ckpt)
    out = os.path.join(LOCAL, "eval", name)
    os.makedirs(out, exist_ok=True)
    # The calib split is 21,127 questions. Scoring it for every (arm, seed) on a CPU-only notebook is
    # the single largest cost in the whole evaluation and it is what exhausted the instance before.
    # FULL_CALIBRATION is on by default because the temperatures feed the calibration table, and off
    # on a machine that cannot afford it -- the accuracy column is identical either way, since
    # temperatures never change an argmax.
    FULL_CALIBRATION = True
    _calib = os.path.join(LOCAL, "data", "calib") if FULL_CALIBRATION else ""
    if not FULL_CALIBRATION:
        print("FULL_CALIBRATION=False -- no temperature/ECE for this arm; accuracy is unaffected.")
    subprocess.check_call([LOCAL_PY, os.path.join(SRC_DIR, "evaluate.py"),
                           "--checkpoint", ckpt,
                           "--test_channel", os.path.join(LOCAL, "data", "test"),
                           "--calib_channel", _calib,
                           "--out", out,
                           "--batch_size", "4" if not _t.cuda.is_available() else "8",
                           "--invariance", "1" if FULL_CALIBRATION else "0",
                           "--invariance_examples", "32",
                           "--invariance_shuffles", "16",
                           "--predictions_out", os.path.join(out, f"predictions_{name}.csv"),
                           "--tag", name,
                        "--env_check", "0"], cwd=SRC_DIR, env=LOCAL_ENV)
    evaluated.append(name)
print(f"\\nevaluated {len(evaluated)} (arm, seed) pair(s) on n={TEST_N:,} test questions")
save_state(evaluated=evaluated)
''')

md("12. JevBench and sysone-bench",
   "Both harnesses are driven by adapters in `decider_v2/bench/`, following the shape of "
   "`jevbench/adapters/so1_decider.py`: an in-process adapter that runs one forward pass per "
   "question and hands the resulting distribution to the harness's OWN scorer. Neither metric is "
   "re-implemented here — JevBench grades with `scoring.score_task`/`summarize.metric` and "
   "sysone-bench with `contracts.validate_answers`/`metrics.summarize_predictions`. Point "
   "`JEVBENCH_ROOT` / `SYSONE_BENCH_ROOT` at clones of the two repos; without them this cell says "
   "NOT RUN rather than inventing a number.")

code("12-benchmarks", '''
JEVBENCH_ROOT = os.environ.get("JEVBENCH_ROOT", "")
SYSONE_ROOT   = os.environ.get("SYSONE_BENCH_ROOT", "")
BENCH_DIR = os.path.join(SRC_DIR, "bench")
RUN_BENCHMARKS = True      # False -> skip and say so
BENCH_TARGETS = ["reference"] + evaluated      # measure the reference plus every trained arm

def bench_cmd(script):
    return [LOCAL_PY, os.path.join(BENCH_DIR, script), "--batch_size", "8",
            "--out", os.path.join(LOCAL, "eval", "bench", script.replace("run_", "").replace(".py", "") + ".json")]

print("Availability (verified in Phase 0, adapters in decider_v2/bench/):")
print(f"  JevBench    : {JEVBENCH_ROOT or '(unset)'}  datasets/public/{{easy,hard,original}}.jsonl")
print("                = 48+111+72 = 231 public tasks, MIT, + the jevbench/ eval package")
print(f"  sysone-bench: {SYSONE_ROOT or '(unset)'}  datasets/v2/manifest.v2.1.0.jsonl")
print("                = 1,190 cases (952 evaluation + 238 calibration) = 1,240 evaluation DECISIONS, MIT")

if not RUN_BENCHMARKS:
    print("\\nRUN_BENCHMARKS is False -- skipping both harnesses.")
else:
    missing = [n for n, p in (("JEVBENCH_ROOT", JEVBENCH_ROOT), ("SYSONE_BENCH_ROOT", SYSONE_ROOT))
               if not p or not os.path.isdir(p)]
    bench_results = {}
    for target in BENCH_TARGETS:
        print("\\n" + "#" * 78); print(f"# BENCHMARK TARGET: {target}"); print("#" * 78)
        extra = []
        if target != "reference":
            local = os.path.join(LOCAL, "downloads", target)
            ckpt, fallback = None, None
            for root, _d, files in os.walk(local):
                if "meta.json" in files and "model.safetensors" in files:
                    fallback = fallback or root
                    if os.path.basename(root) == "best":
                        ckpt = root
            ckpt = ckpt or fallback
            if not ckpt:
                print(f"  SKIP {target}: no checkpoint on disk")
                continue
            extra = ["--checkpoint", ckpt]
        else:
            extra = ["--reference", "avbiswas/bev-decider-0.4B"]
        for script, rootvar in (("run_jevbench.py", "JEVBENCH_ROOT"),
                                ("run_sysone_bench.py", "SYSONE_ROOT")):
            rootpath = JEVBENCH_ROOT if script.startswith("run_jev") else SYSONE_ROOT
            if not rootpath or not os.path.isdir(rootpath):
                print(f"\\n  NOT RUN -- {script} needs {rootvar} set to a clone of the repo.")
                continue
            out_json = os.path.join(LOCAL, "eval", "bench", f"{target}_{script[:-3]}.json")
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            print(f"\\n--- {script} on {target} ---")
            rc = subprocess.call([LOCAL_PY, os.path.join(BENCH_DIR, script),
                                  "--batch_size", "8", "--out", out_json,
                                  "--jevbench_root" if script.startswith("run_jev")
                                  else "--sysone_bench_root", rootpath] + extra, env=LOCAL_ENV)
            bench_results[f"{target}:{script}"] = {"rc": rc, "json": out_json}
    print("\\nbenchmark cells finished:", json.dumps(bench_results, indent=2))
    if missing:
        print(f"\\nnot run for want of: {missing} -- set them and re-run this cell.")

    # The adapters print to the subprocess console; summarise their JSON here so the results are
    # in the notebook output rather than only on disk.
    print("\\n" + "=" * 78)
    print("BENCHMARK SUMMARY (from the adapters' own JSON; metrics computed by the upstream harnesses)")
    print("=" * 78)
    for target in BENCH_TARGETS:
        jp = os.path.join(LOCAL, "eval", "bench", f"{target}_run_jevbench.json")
        sp = os.path.join(LOCAL, "eval", "bench", f"{target}_run_sysone_bench.json")
        if os.path.exists(jp):
            s = json.load(open(jp))["summary"]
            e = s.get("ece")
            e = e.get("ece") if isinstance(e, dict) else e
            e_txt = "n/a" if e is None else f"{e:.4f}"
            print(f"  JevBench    {target:14s} accuracy {s['accuracy']*100:6.2f}  "
                  f"macro {s['macro_accuracy']*100:6.2f}  valid {s['n_valid']}/{s['n_planned']}  "
                  f"brier {s['brier_mean']:.4f}  ece {e_txt}")
        if os.path.exists(sp):
            s = json.load(open(sp))["summary"]
            print(f"  sysone-bench {target:13s} choice {s['choice']['accuracy']*100:6.2f}  "
                  f"noul {s['noul']['accuracy']*100:6.2f}  score(nearest) "
                  f"{s['score']['nearest_level_accuracy']*100:6.2f}  "
                  f"choice ece {s['choice']['ece_raw']:.4f}")
    print("\\nPublished reference points: JevBench all 65.8, sysone-bench v2 all 70.8.")
    print("The `reference` row is the published bev-decider-0.4B weights run through these same")
    print("adapters, so agreement there is the check that the adapters are faithful.")
''')

md("13. Results table",
   "The published numbers (labelled *published*), the reference checkpoint measured on *my* test "
   "split, then one row per (arm, seed), then the seed-averaged delta v2 − baseline and the "
   "`## Decision` verdict from `decide.py` (a paired bootstrap over test questions plus four "
   "conditions). `NOT RUN` for anything that has not actually executed.")

code("13-results", '''
PUBLISHED = {
    "source": "avbiswas/bev-decider-0.4B model card, revision 687e28f",
    "held_out_default_5000": {"all": 74.7, "choice": 69.8, "noul": 84.6, "score": 63.7},
    "jevbench": {"all": 65.8, "easy": 97.9, "standard": 73.6, "hard": 46.8},
    "sysone_bench_v2_all": 70.8, "ece": 0.065,
    "order_invariance_max_delta": "<= 1e-5 (fp32, 960 shuffles, k=3-12)",
}


def load(tag):
    """Metrics land in eval/<tag>/ (per-arm cells) or directly in eval/ (the reference cells)."""
    for path in (os.path.join(LOCAL, "eval", tag, f"metrics_{tag}.json"),
                 os.path.join(LOCAL, "eval", f"metrics_{tag}.json")):
        if os.path.exists(path):
            return json.load(open(path))
    return None


def overall(m):
    """Weighted overall accuracy, recomputed from the per-type table. Never assumed."""
    counts, by_type = m["count_by_type"], m["accuracy_by_type"]
    return sum(by_type[k] * counts[k] for k in counts) / sum(counts.values())


def seeds_of(arm):
    # job names are "<arm>_s<seed>"; use the same parser as arm_of() so the two cannot disagree.
    return sorted(n.rpartition("_s")[2] for n in evaluated if arm_of(n) == arm)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def spread(xs):
    xs = list(xs)
    return (max(xs) - min(xs)) if len(xs) > 1 else 0.0


rows = [("bev-decider-0.4B  PUBLISHED (default-config test)", PUBLISHED["held_out_default_5000"], None)]
for tag, label in (("reference_default", "reference on default-config 5k (validates our encoder)"),
                   ("reference", "reference on OUR all-config test split")):
    mm = load(tag)
    rows.append((label, mm["accuracy_by_type"] if mm else None, mm))
for name in evaluated:
    mm = load(name)
    rows.append((f"{name}, OUR split", mm["accuracy_by_type"] if mm else None, mm))

w = "{:<46s} {:>9s} {:>8s} {:>8s} {:>8s} {:>9s} {:>12s}"
print(f"test questions: n={TEST_N:,}"
      + ("   (SMOKE SHARD -- plumbing check only, NOT a result)" if SMOKE
         else "   (FULL published test split)"))
print()
print(w.format("model", "all", "choice", "noul", "score", "ECE", "inv max|dp|"))
for name, by_type, mm in rows:
    if by_type is None:
        print(f"{name:<46s} {'NOT RUN':>9s}")
        continue
    if mm is None:                       # the published numbers are already percentages
        print(w.format(name, f"{by_type['all']:.1f}", f"{by_type['choice']:.1f}",
                       f"{by_type['noul']:.1f}", f"{by_type['score']:.1f}",
                       f"{PUBLISHED['ece']:.4f}", "<=1e-5"))
        continue
    inv = mm.get("order_invariance", {}).get("max_abs_delta")
    print(w.format(name, f"{overall(mm)*100:.2f}",
                   f"{mm['accuracy_by_type']['choice']*100:.2f}",
                   f"{mm['accuracy_by_type']['noul']*100:.2f}",
                   f"{mm['accuracy_by_type']['score']*100:.2f}",
                   f"{mm['ece']:.4f}", f"{inv:.2e}" if inv else "n/a"))

print()
for tag in ["reference"] + evaluated:
    mm = load(tag)
    if not mm:
        continue
    print(f"  {tag:16s} n={mm['n_questions']:7,d}  ECE {mm['ece']:.4f} -> "
          f"{mm.get('ece_after_calibration', float('nan')):.4f} after calibration   "
          f"NLL {mm['nll']:.4f} -> {mm.get('nll_after_calibration', float('nan')):.4f}   "
          f"temperatures={mm.get('temperatures', 'none (no calib channel)')}")


def arm_stats(arm):
    """Seed-averaged overall + per-type accuracy for one arm."""
    ms = [load(f"{arm}_s{s}") for s in seeds_of(arm)]
    ms = [m for m in ms if m]
    if not ms:
        return None
    return {
        "seeds": seeds_of(arm),
        "overall": mean(overall(m) for m in ms),
        "spread": spread(overall(m) for m in ms),
        "by_type": {t: mean(m["accuracy_by_type"][t] for m in ms)
                    for t in ("choice", "noul", "score")},
        "ece": mean(m["ece"] for m in ms),
        "n": ms[0]["n_questions"],
    }


sb, sv = arm_stats("baseline"), arm_stats("v2")
DECISION = None
DECISION_REPORT = None
if sb and sv:
    delta = (sv['overall'] - sb['overall']) * 100
    spread = max(sb['spread'], sv['spread']) * 100
    print(f"\\nDELTA v2 - baseline, same split, same budget, seed-matched"
          f"{' (AVERAGED OVER SEEDS -- a plumbing check, not a result)' if SMOKE else ''}")
    print(f"  {'all':52s} {delta:+.2f}")
    for t in ("choice", "noul", "score"):
        print(f"  {t:52s} {(sv['by_type'][t] - sb['by_type'][t]) * 100:+.2f}")
    print(f"\\n  seed spread: baseline {sb['spread']*100:+.2f} pp, v2 {sv['spread']*100:+.2f} pp "
          f"({len(sb['seeds'])} seeds each, identical data and budget)")

    # The verdict comes from decider_v2/decide.py, which is the single implementation of the rule
    # and is unit-tested in tests.py section (e). Calling it rather than re-deriving it here is the
    # point: the notebook cannot disagree with the tests about what SEPARABLE means.
    sys.path.insert(0, SRC_DIR)
    import decide as _decide
    _res = _decide.analyse(os.path.join(LOCAL, "eval"), "baseline", "v2",
                           seeds=[int(s) for s in sb['seeds']])
    if _res is None:
        print("\\nDECISION: NOT EVALUATED -- per-example prediction CSVs are missing for at least one")
        print("           (arm, seed). Re-run cell 11; it writes them next to metrics_<tag>.json.")
    else:
        DECISION_REPORT = _decide.format_report(_res)
        print()
        print(DECISION_REPORT)
        DECISION = {"verdict": _res["verdict"], "delta_pp": _res["delta_pp"],
                    "seed_spread_pp": _res["seed_spread_pp"],
                    "n_seeds": _res["n_seeds"], "test_n": _res["bootstrap"]["n_questions"],
                    "n_resamples": _res["bootstrap"]["n_resamples"],
                    "bootstrap_seed": _res["bootstrap"]["bootstrap_seed"],
                    "ci_low_pp": _res["bootstrap"]["ci_low_pp"],
                    "ci_high_pp": _res["bootstrap"]["ci_high_pp"],
                    "excludes_zero": _res["bootstrap"]["excludes_zero"],
                    "checks": _res["checks"],
                    "per_seed": [{"seed": d["seed"], "delta_pp": d["delta_pp"],
                                  "ci_low_pp": d["ci_low_pp"], "ci_high_pp": d["ci_high_pp"],
                                  "excludes_zero": d["excludes_zero"]}
                                 for d in _res["per_seed"]],
                    "min_effect_pp": _decide.MIN_EFFECT_PP,
                    "min_seeds": _decide.MIN_SEEDS_FOR_SEPARABLE}
        save_state(decision=DECISION)
    if SMOKE:
        print("  NOTE: SMOKE shard at 100 steps. This exercises the rule; it does not satisfy it,")
        print("        and no accuracy number on a SMOKE shard is a result.")

    # per-domain deltas from the first seed's rows, so the table shape is unchanged
    mb, mv = load(f"baseline_s{sb['seeds'][0]}"), load(f"v2_s{sv['seeds'][0]}")
    print("\\n  per-domain delta (seed " + str(sb["seeds"][0]) + " only, n>=50):")
    for k, cnt in (("accuracy_by_domain", "count_by_domain"),):
        diffs = {d: mv[k][d] - mb[k][d] for d in mb[k]
                 if d in mv[k] and mb[cnt].get(d, 0) >= 50}
        for d, v in sorted(diffs.items(), key=lambda kv: kv[1])[:6]:
            print(f"    {d:48s} {v*100:+.2f}   (n={mb[cnt][d]})")
        print(f"    ({len(diffs)} entries with n>=50; the full table is in metrics_*.json)")
else:
    print("\\nDECISION: NOT EVALUATED -- need at least one evaluated baseline and one v2 run.")

lines = ["# RESULTS", "", f"run: {RUN_NAME}", f"split: {STATE['data_prefix']}",
         f"test questions: {TEST_N:,}"
         + (" (SMOKE SHARD - plumbing check only, NOT a result)" if SMOKE
            else " (full published test split)"),
         f"seeds: {SEEDS}  split_seed: {SPLIT_SEED}", "",
         "| model | all | choice | noul | score | ECE | order-invariance max abs dp |",
         "|---|---|---|---|---|---|---|"]
for name, by_type, mm in rows:
    if by_type is None:
        lines.append(f"| {name} | NOT RUN | | | | | |")
    elif mm is None:
        pub = PUBLISHED["held_out_default_5000"]
        lines.append(f"| {name} (PUBLISHED) | {pub['all']} | {pub['choice']} | {pub['noul']} | "
                     f"{pub['score']} | {PUBLISHED['ece']} | {PUBLISHED['order_invariance_max_delta']} |")
    else:
        inv = mm.get("order_invariance", {}).get("max_abs_delta")
        lines.append(f"| {name} | {overall(mm)*100:.2f} | {mm['accuracy_by_type']['choice']*100:.2f} "
                     f"| {mm['accuracy_by_type']['noul']*100:.2f} | {mm['accuracy_by_type']['score']*100:.2f} "
                     f"| {mm['ece']:.4f} | {inv if inv is not None else 'n/a'} |")
lines += ["", "Published reference points (measured on a different split, not a control):",
          f"- held-out typed decisions, default config: {PUBLISHED['held_out_default_5000']['all']}",
          f"- JevBench all / hard: {PUBLISHED['jevbench']['all']} / {PUBLISHED['jevbench']['hard']}",
          f"- sysone-bench v2 all: {PUBLISHED['sysone_bench_v2_all']}",
          f"- ECE (sysone-bench): {PUBLISHED['ece']}", ""]

# The decision rule, in the artefact itself so it cannot be separated from the numbers it judges.
lines += ["## Decision", ""]
if DECISION is None:
    lines += ["**NOT EVALUATED** -- per-example prediction CSVs were missing for at least one "
              "(arm, seed). Re-run cell 11 to write them.", ""]
else:
    d = DECISION
    lines += [
        "The delta is tested against the two things that can hide it: **which test questions you "
        "scored** (a paired bootstrap over questions, same seed pairing) and **which training run "
        "you happened to get** (the seed spread). A 95% CI that excludes 0 is necessary, not "
        "sufficient.", "",
        f"Paired bootstrap: {d['n_resamples']:,} resamples of the "
        f"{d['test_n']:,} test questions, resampling seed {d['bootstrap_seed']}. Both arms are "
        f"resampled with the SAME indices, so the CI is for the paired difference. It covers test "
        f"sampling only -- condition (b) is what covers training-run variance.", "",
        "| seed | delta (pp) | 95% CI (pp) | CI excludes 0 |",
        "|---:|---:|---:|:--:|",
    ]
    for r in d["per_seed"]:
        lines.append(f"| {r['seed']} | {r['delta_pp']:+.2f} | "
                     f"[{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] | "
                     f"{'yes' if r['excludes_zero'] else 'NO'} |")
    lines += ["",
              f"**Verdict: {d['verdict']}**", "",
              "| # | condition | measured | result |",
              "|---|---|---:|:--:|"]
    order = ["bootstrap CI excludes 0", "|delta| > seed spread",
             f"|delta| >= {d['min_effect_pp']} pp", f"seeds >= {d['min_seeds']}"]
    measured = [
        f"[{d['ci_low_pp']:+.2f}, {d['ci_high_pp']:+.2f}]",
        f"delta {d['delta_pp']:+.2f} vs spread {d['seed_spread_pp']:.2f}",
        f"|{d['delta_pp']:+.2f}| pp",
        f"{d['n_seeds']} per arm",
    ]
    for i, (name, ok) in enumerate(d["checks"]):
        # a literal "|" inside a markdown table cell breaks the table, so escape the ones in
        # condition names like "|delta| > seed spread"
        lines.append(f"| ({chr(97 + i)}) | {name.replace('|', chr(92) + '|')} | "
                     f"{measured[i].replace('|', chr(92) + '|')} | "
                     f"{'PASS' if ok else 'FAIL'} |")
    lines += ["",
              f"Delta (seed-averaged): **{d['delta_pp']:+.2f} pp** on {d['test_n']:,} test "
              f"questions across {d['n_seeds']} seed(s) per arm.", ""]
    if d["verdict"] == "SEPARABLE":
        lines += [f"All four conditions hold. The delta of {d['delta_pp']:+.2f} pp survives test "
                  f"resampling, exceeds the arms' own seed-to-seed variation "
                  f"({d['seed_spread_pp']:.2f} pp), is large enough to act on, and rests on "
                  f"{d['n_seeds']} seeds per arm."]
    elif d["verdict"] == "PROVISIONAL":
        lines += [
            f"Conditions (a)-(c) hold, but `SEPARABLE` requires **{d['min_seeds']} seeds per arm** "
            f"and this run has {d['n_seeds']}. Two seeds give a spread that is a range between a "
            f"single pair of runs; it was measured at **0.05 pp** in one dry run and **2.52 pp** in "
            f"another for the same configuration, so a tight spread on two seeds is not evidence of "
            f"a tight spread. **Add one more seed to `SEEDS` in cell 1 and re-run cells 8-13 before "
            f"quoting this number.**",
        ]
    else:
        failed = [name for name, ok in d["checks"] if not ok]
        lines += [f"Failing condition(s): **{', '.join(failed)}**. "
                  f"**This comparison does not separate the two architectures.** Do not quote the "
                  f"delta as a result; add seeds and re-run cells 8-13."]
    if SMOKE:
        lines += ["", "> Produced under `SMOKE`, on the 4,000-question shard at 100 steps. It "
                  "exercises the rule; it does not satisfy it, and no accuracy number on a SMOKE "
                  "shard is a result."]
    lines += ["",
              "Why a binomial interval is not the bar: it assumes test sampling is the only source "
              "of randomness. The seed spread is a direct measurement of this experiment's own "
              "run-to-run variance on this exact data, and condition (b) requires the delta to "
              "beat it. Condition (c) exists because at n = 80,019 a small delta can be "
              "statistically unambiguous and still not worth acting on.", ""]

open("RESULTS.md", "w").write("\\n".join(lines))
print("\\nwrote RESULTS.md")
''')

code("13-charts", '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bars, labels, colours = [], [], []
palette = {"reference": "#888", "baseline": "#4C72B0", "v2": "#DD8452"}
for tag in ["reference"] + evaluated:
    mm = load(tag)
    if not mm:
        continue
    arm = arm_of(tag)
    for t in ("choice", "noul", "score"):
        bars.append(mm["accuracy_by_type"][t] * 100)
        labels.append(f"{tag}\\n{t}")
        colours.append(palette.get(arm, "#55A868"))
if bars:
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 4))
    ax.bar(range(len(bars)), bars, color=colours)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 100); ax.set_ylabel("accuracy (%)"); ax.grid(axis="y", alpha=.3)
    ax.set_title(f"accuracy on OUR test split n={TEST_N:,} ({RUN_NAME})")
    plt.tight_layout(); plt.savefig("RESULTS_by_type.png", dpi=120)
    print("wrote RESULTS_by_type.png")
''')

md("14. Optional ablations",
   "One flag at a time against v2, same budget, same data, seed 0 only. Each is a separate job; "
   "they are not free.")

code("14-ablations", '''
# `v2_lora_bev` is the load-bearing one: it keeps the v2 HEAD and every v2 loss/optimisation change
# but hands the LoRA back to bev-decider's own shape (r=8, attention only, last 12 layers). If v2's
# gain survives this, the head and the loss are carrying it and the wide LoRA is not; if the gain
# disappears, the LoRA width is doing the work. One flag at a time is the whole point, so
# lora_alpha is moved with r to keep the alpha/r scaling ratio constant at 2.0 (v2 uses 32/16).
ABLATIONS = {
    "v2_lora_bev":    dict(lora_include_mlp=0, lora_last_k_layers=12, lora_r=8, lora_alpha=16),
    "v2_layers24":    dict(num_layers=24, lora_last_k_layers=24),
    "v2_meanpool":    dict(mode="special", pool="mean"),
    "v2_derived":     dict(use_derived=1),
    "v2_stage2":      None,   # handled below: resume from v2 with lr/4
}
ABLATION_SEED = SEEDS[0]     # ablations are single-seed; they are diagnostics, not the headline

if RUN_ABLATIONS:
    save_state(ablation_args={ab: over for ab, over in ABLATIONS.items() if over})
    for ab, over in ABLATIONS.items():
        if over is None:
            continue
        print(f"\\nlaunching {ab}: " + " ".join(f"--{k}={v}" for k, v in over.items()))
        launch(f"{ab}_s{ABLATION_SEED}", hyper(ABLATION_SEED, **{**V2_OVER, **over}), CHANNELS)
    print("\\nRe-run cells 10, 11, 12 and 13 to evaluate the ablation jobs: cell 10 monitors every")
    print("<arm>_s<seed> job, and cell 13 lists every evaluated job, ablations included.")
    print("\\nstage 2 (hard negatives): once v2 finishes, relaunch v2 from its best checkpoint with")
    print("  --resume <path to checkpoints/best> --lr_lora 5e-5 --lr_head 2.5e-4 --max_steps 5000")
    print("Mine the errors from stage-1 train-dev first: the stage-2 data step is NOT automated here.")
else:
    print("RUN_ABLATIONS is False -- skipping. Set it to True at the top of the notebook to run these:")
    for ab, over in ABLATIONS.items():
        if over is not None:
            print(f"  {ab:16s} v2 plus " + " ".join(f"--{k}={v}" for k, v in over.items()))
    print("  v2_stage2        resume v2 with lr/4")
    print(f"\\nEvery ablation launches at seed {ABLATION_SEED} and the same {MAX_STEPS}-step budget.")
''')

md("15. Cleanup and artifact inventory",
   "Lists what this notebook created in S3 and optionally deletes finished jobs. It never deletes "
   "S3 objects -- delete those yourself once you have what you need.")

code("15-cleanup", '''
KEEP_S3 = os.environ.get("KEEP_S3", "1") == "1"
try:
    n, total = 0, 0
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=f"{PREFIX}/{RUN_NAME}/"):
        for o in page.get("Contents", []):
            n += 1
            total += o["Size"]
            print(f"  s3://{BUCKET}/{o['Key']}  {o['Size'] / 2**20:.1f} MiB")
    print(f"{n} objects, {total / 2**20:.1f} MiB under s3://{BUCKET}/{PREFIX}/{RUN_NAME}/")
except Exception as e:
    print("could not list:", type(e).__name__, e)

if os.environ.get("DELETE_JOBS") == "1":
    for k, v in STATE.items():
        if k.startswith("jobs/") and v.get("name"):
            try:
                sm.delete_training_job(TrainingJobName=v["name"])
                print(f"  deleted {v['name']}")
            except Exception as e:
                print(f"  could not delete {v['name']}: {type(e).__name__}")
else:
    print("set DELETE_JOBS=1 to delete the training jobs (they are in a Completed state)")
if not KEEP_S3:
    print("WARNING: KEEP_S3=0 was set, but this cell never deletes objects. Delete them manually.")
print("done")
''')


def build_src_zip_b64():
    """Zip decider_v2/{*.py, bench/*.py, requirements.txt} as decider_v2_src/ and base64 it.

    The notebook carries its own training code so that uploading a single .ipynb to SageMaker is
    sufficient. Only the sources the jobs actually execute ship; the loose *.txt run logs in the
    repo are build artifacts and are deliberately left out.
    """
    import base64, zipfile
    files = sorted(glob.glob(os.path.join(SRC, "*.py"))) \
        + sorted(glob.glob(os.path.join(SRC, "bench", "*.py"))) \
        + [os.path.join(SRC, "requirements.txt")]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, "decider_v2_src/" + os.path.relpath(f, SRC).replace("\\", "/"))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, (kind, first, src) in enumerate(CELLS):
        # `note` cells arrive fully formed from md(); their body is already commented and already
        # contains the print that echoes the text. Only `code` cells still need their banner stitched
        # to their body.
        blob = first if kind == "note" else first + src
        if "@@SRC_ZIP@@" in blob:
            blob = blob.replace("@@SRC_ZIP@@", build_src_zip_b64())
        lines = blob.split("\n")
        while lines and lines[-1] == "":
            lines.pop()
        # nbformat's list form keeps "\n" at the end of every line except the last
        lines = [ln + "\n" for ln in lines[:-1]] + lines[-1:]
        cid = f"cell-{i:02d}"
        # EVERY cell is a code cell. See md() for why there are no markdown cells: SageMaker's
        # Jupyter executed prose as Python twice, and a cell full of comments cannot do that.
        nb["cells"].append({"cell_type": "code", "id": cid, "execution_count": None,
                            "metadata": {}, "outputs": [], "source": lines})
    return nb


if __name__ == "__main__":
    nb = build()
    import nbformat
    nbformat.validate(nbformat.from_dict(nb))

    # Parse every code cell before writing. The cells are built from '''...''' literals in this file,
    # so a stray backslash-n or an unbalanced quote inside one silently becomes a SyntaxError in the
    # NOTEBOOK -- which is where it is discovered, thousands of lines into a debugging session, instead
    # of here. Two broken cells shipped that way; this check is why the third cannot.
    import ast
    broken = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith(("%", "!")):
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            broken.append((i, c.get("id", "?"), f"line {e.lineno}: {e.msg}"))
    if broken:
        for i, cid, msg in broken:
            print(f"SYNTAX ERROR in cell {i} ({cid}) -- {msg}")
        raise SystemExit(f"refusing to write {OUT}: {len(broken)} code cell(s) do not parse")

    # Prose must never end up in a code cell. The builder assembles each cell from a ``## heading``
    # plus a body, so a stray md()/code() mix-up turns documentation into a SyntaxError at RUNTIME,
    # on someone else's machine, as a cell whose first line is a paragraph rather than a comment.
    # Only the FIRST line is checked: a later line may legitimately be a long trailing comment or a
    # string, and testing every line flags ordinary code.
    prose = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        lines = [ln.strip() for ln in "".join(c["source"]).splitlines() if ln.strip()]
        if not lines:
            continue
        first = lines[0]
        # A real code cell starts with a comment, a decorator, an import, a def, or an assignment.
        if (first.startswith("##") or first.startswith("#") or first.startswith(("%", "!"))
                or first.startswith(("from ", "import ", "def ", "class ", "@", '"""', "'''", '"""'))):
            continue
        prose.append((i, c.get("id", "?"), first[:60]))
    if prose:
        for i, cid, t in prose:
            print(f"PROSE IN CODE CELL {i} ({cid}): first line is {t!r}")
        raise SystemExit(f"refusing to write {OUT}: {len(prose)} code cell(s) start with prose")

    # A cell that uses torch.cuda.is_available() must import torch itself. Relying on an earlier cell
    # having done so is exactly how the SESSION NameError happened: after a kernel restart the earlier
    # cell's globals were gone and the failure surfaced three cells later as something unrelated.
    import re as _re
    undef = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "torch.cuda.is_available()" not in src:
            continue
        if _re.search(r"^\s*import\s+torch\b", src, _re.M):
            continue
        # `import sys, json, torch` is also a real import of torch.
        if _re.search(r"^\s*import\s+[^\n]*\btorch\b", src, _re.M):
            continue
        # `torch = _try_import("torch")` is also a real, guarded binding.
        if _re.search(r'^\s*torch\s*=\s*_try_import\("torch"\)', src, _re.M):
            continue
        undef.append((i, c.get("id", "?")))
    if undef:
        for i, cid in undef:
            print(f"UNDEFINED torch in cell {i} ({cid}): uses torch.cuda but never imports torch")
        raise SystemExit(f"refusing to write {OUT}: {len(undef)} cell(s) use torch without importing it")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"wrote {OUT}  cells={len(nb['cells'])} "
          f"(markdown={len(nb['cells']) - n_code}, code={n_code})")
    print("nbformat.validate: OK")
    print(f"all {n_code} code cells parse")