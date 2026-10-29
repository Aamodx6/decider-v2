# Cell 2 - parameters. The kernel runs the whole notebook in SMOKE mode when SMOKE=True.
#
# SMOKE runs `configs/smoke.yaml` through the real train.py in the real container, but with a
# 3-step override instead of the config's 100 steps — everything else (pack shape, channels,
# serializer, checkpoint wiring) is exactly the real path. The real run launches the two arm
# configs at their own budgets. The two modes keep SEPARATE state files, so a smoke launch can
# never collide with a real job (fingerprint guard works within one mode's file only).
import getpass
import json

# ---- mode                                                                       ---------------
SMOKE = True                        # False for the real run
SMOKE_MAX_STEPS = 3                 # the 3-step budget override (smoke.yaml itself says 100)

# ---- infrastructure ---------------------------------------------------------------
BUCKET = "decider-v2-000000000000-us-east-1"
REGION = "us-east-1"
ROLE_ARN = ""                       # empty -> get_execution_role(); a named role otherwise
ARM_TAG = "decider_v2"              # bucket folder; the job-name form strips '_' (SM rule)

# ---- compute -----------------------------------------------------------------------
INSTANCE_TYPE = "ml.g4dn.4xlarge"   # measured: v2 ~33 qps, baseline ~25 qps on this class
INSTANCE_COUNT = 1
USE_SPOT = False                    # default OFF: without --auto_resume an interruption is silent
MAX_RUN_SECONDS = 24 * 3600         # max_run; also the max_wait ceiling when USE_SPOT
KEEP_ALIVE = 0                      # warm-pool minutes between consecutive jobs

# ---- budget governors --------------------------------------------------------------
TARGET_PRESENTATIONS = 940_032      # 14,688 steps x 64 presentations (the recipe budget)
BASELINE_VS_V2_QPS = 1.32           # baseline is 1.32x SLOWER than v2 at identical batch (measured)

# ---- local paths --------------------------------------------------------------------
REPO = Path.cwd()
DATA_DIR = REPO / ("data_smoke" if SMOKE else "data_prepared")
RUNS_LOCAL = REPO / "runs"

# ---- state -------------------------------------------------------------------------- 
STATE_PATH = REPO / ("decider_v2_state_smoke.json" if SMOKE else "decider_v2_state.json")
state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
state.setdefault("jobs", {})
state.setdefault("launches", {})

print(f"SMOKE={SMOKE}  bucket={BUCKET}  state={STATE_PATH.name}  instance={INSTANCE_TYPE}")
print(f"defaults: use_spot={USE_SPOT}  max_run={MAX_RUN_SECONDS}s  target={TARGET_PRESENTATIONS:,} presentations")
