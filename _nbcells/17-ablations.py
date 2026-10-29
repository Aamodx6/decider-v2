# Cell 17 - the Tier-2 ablation ladder (five configs, one job each). Every rule from cells 11-12
# applies: config-driven hyperparameters, fingerprint-from-config, describe-driven reuse guard,
# non-blocking launches collected by cell 13.
from pathlib import Path

ABLATIONS = sorted(str(p.relative_to(REPO)) for p in (REPO / "configs" / "ablations").glob("v2_*.yaml"))
assert ABLATIONS, "no configs/ablations/v2_*.yaml found"
print("ablation ladder:", ", ".join(ABLATIONS))
print("note: configs/decider_v2_long.yaml (the 4x-budget option) is NOT in this ladder; "
      "launching it is a deliberate budget decision, never an accidental side effect.")

if SMOKE:
    print("[skip] SMOKE mode: the ladder would queue five GPUs on jobs; launch it in real mode only")
else:
    for cfg in ABLATIONS:
        name = Path(cfg).stem  # e.g. v2_lora_bev
        launch_job(f"abl-{name}", cfg, n_train=N_TRAIN, seed=0, wait=False)
    print("five ablation jobs submitted; monitor them with cell 13")
