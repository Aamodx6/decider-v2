# Cell 6 - carve the four shards with dataset.py prepare. Re-runs are cheap: dataset.py reuses
# an existing carve when the manifest's seed/calib_target match, and caps only bite on a fresh
# write (SMOKE caps: 4000 train / 400 val / 400 calib / 4000 test -> a Git-LFS-sized shard set
# that still has every task/domain present in the mix).
import json
import os
import subprocess

cap_args = []
if SMOKE:
    cap_args = ["--smoke_train", "4000", "--smoke_val", "400",
                "--smoke_calib", "400", "--smoke_test", "4000"]
cmd = [sys.executable, "dataset.py", "prepare", "--out", str(DATA_DIR),
       "--seed", "0", "--calib_target", "20000", "--val_frac", "0.03", *cap_args]
env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false"}
print(" ".join(os.path.basename(c) if isinstance(c, str) else c for c in cmd))
proc = subprocess.run(cmd, cwd=str(REPO), env=env)
if proc.returncode != 0:
    raise SystemExit("dataset.py prepare failed")

manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
# sizes, not stats: stats describes the FULL exploded train split, while the shard train.py
# actually trains on is the carved (and, in SMOKE, capped) one. steps/epoch must mirror the
# parquet the job receives or the fingerprint's arithmetic is off.
N_TRAIN = manifest["sizes"]["train"]
print(f"\nN_TRAIN={N_TRAIN:,} (written train shard; drives every budget's steps/epoch math)")
for k in ("train", "val", "calib", "test"):
    assert (DATA_DIR / k / "questions.parquet").exists(), f"missing shard: {k}"
