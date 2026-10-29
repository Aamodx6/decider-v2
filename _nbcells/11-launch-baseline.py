# Cell 11 - launch the baseline arm (configs/baseline_replica.yaml).
#
# In real mode: non-blocking launch (wait=False, logs="None") - cell 13 owns polling, and the
# v2 job goes up in parallel from cell 12. In SMOKE mode: cells 13-14 only prove plumbing, so
# this launch runs with a tiny override budget and blocking wait (the artifact layout is
# exercised immediately). In both modes the launch is fingerprinted by its EFFECTIVE budget -
# the launcher computes it from the SAME yaml the job receives.
overrides = {"max_steps": SMOKE_MAX_STEPS} if SMOKE else None
bh = effective_budget("configs/baseline_replica.yaml", N_TRAIN, overrides)
print(f"baseline budget: fingerprint={fingerprint(bh)} -> {bh}")
est_base = launch_job("baseline", "configs/baseline_replica.yaml", n_train=N_TRAIN,
                      overrides=overrides, seed=0, wait=SMOKE)
if est_base is None:
    print("[reuse] baseline: nothing to watch here - cell 13 loops over recorded jobs (fresh or old)")
