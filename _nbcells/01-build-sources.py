# Cell 1 - the allowlist source pack under build/sagemaker_src/ (git-ignored).
#
# SageMaker unpacks source_dir into the container and runs the entry point with THAT directory
# as cwd, so everything train.py imports, its requirements.txt (the --env_check pin list it
# enforces), the config yaml the launch cells pass and the eval helper have to be inside this
# allowlist - a 25-file pack beats a 400MB tarball taped together with base64 ink. The notebook
# lives inside the repo checkout, so REPO == Path.cwd().
#
# No tests/ directory is shipped: train.py's --selftest carries its own compact gates, so the
# preflight works in a stripped pack without dragging the test suite into the container.
import shutil
from pathlib import Path

REPO = Path.cwd()
if not (REPO / "train.py").exists():
    raise SystemExit(
        f"no train.py under {REPO} - run this notebook with its working directory set to the "
        "decider-train checkout root (the directory that contains train.py)")

BUILD = REPO / "build" / "sagemaker_src"
if BUILD.exists():
    shutil.rmtree(BUILD)

# train.py's import closure + the pins it enforces + every config the launch cells reference +
# the eval helper used downstream by cell 14.
ALLOWLIST = [
    "train.py", "network.py", "tokenization.py", "dataset.py", "logger.py", "normalize.py",
    "evaluate.py", "download_eval.py", "inference.py", "requirements.txt",
    "configs/baseline_replica.yaml", "configs/decider_v2.yaml", "configs/smoke.yaml",
    "configs/decider_v2_long.yaml",
    "configs/ablations/v2_lora_bev.yaml",
    "configs/ablations/v2_layers24.yaml",
    "configs/ablations/v2_meanpool.yaml",
    "configs/ablations/v2_derived.yaml",
    "configs/ablations/v2_stage2.yaml",
]
for rel in ALLOWLIST:
    src = REPO / rel
    if not src.exists():
        raise SystemExit(f"missing from allowlist: {rel} (REPO = {REPO})")
    dst = BUILD / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if dst.stat().st_size == 0:
        raise SystemExit(f"empty allowlist file: {rel}")

n_pack = sum(1 for p in BUILD.rglob("*") if p.is_file())
print(f"pack built: {n_pack} files under {BUILD.relative_to(REPO)}")
for p in sorted(p.relative_to(BUILD).as_posix() for p in BUILD.rglob("*") if p.is_file()):
    print(" ", p)
