# Cell 12 - launch the v2 arm (configs/decider_v2.yaml), parallel to the baseline in real mode.
#
# Same mechanics: fingerprint-from-config guard, checkpoint wiring, kwargs validation.
overrides = {"max_steps": SMOKE_MAX_STEPS} if SMOKE else None
bh = effective_budget("configs/decider_v2.yaml", N_TRAIN, overrides)
print(f"v2 budget: fingerprint={fingerprint(bh)} -> {bh}")
est_v2 = launch_job("v2", "configs/decider_v2.yaml", n_train=N_TRAIN,
                    overrides=overrides, seed=0, wait=SMOKE)
if est_v2 is None:
    print("[reuse] v2: nothing new to watch here - cell 13 polls every recorded InProgress job")
