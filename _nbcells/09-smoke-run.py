# Cell 9 - the SMOKE run: configs/smoke.yaml with a 3-step budget override through the REAL
# train.py in the REAL container (pack, channels, serializer, checkpoint wiring are all the
# production path). Finishes in a couple of minutes on a g4dn; the qps tail feeds cell 10.
# In real mode this cell prints a skip - the two arm launches own the GPU time.
overrides = {"max_steps": SMOKE_MAX_STEPS} if SMOKE else None
SMOKE_JOB = None
if not SMOKE:
    print("[skip] SMOKE=False: cells 11-12 launch the real arms; cell 9 is the smoke path only")
else:
    est_smoke = launch_job("smoke", "configs/smoke.yaml", n_train=N_TRAIN,
                           overrides=overrides, seed=0, wait=True)
    if est_smoke is not None:
        SMOKE_JOB = est_smoke.latest_training_job.name
    else:                                # guard_reuse re-used a completed job -> recover its name
        for name, meta in state["jobs"].items():
            if meta.get("arm") == "smoke":
                SMOKE_JOB = name
                break
    if not SMOKE_JOB:
        raise SystemExit("no smoke job available (fresh launch refused and nothing reusable recorded)")
    print(f"SMOKE_JOB={SMOKE_JOB}")
