# Cell 10 - the runtime and price verdict for the REAL budget, built from evidence in this
# order: (1) the smoke job's own sps tail from CloudWatch (filter_log_events), (2) the measured
# g4dn ratio baseline = v2 x 1.32 applied to it, (3) the Price List on-demand rate (NOT RESOLVED
# is a legal outcome - the verdict then rides on 1+2 only). The guard raises BEFORE cells 11-12
# when the real budget cannot finish inside MAX_RUN_SECONDS; in SMOKE mode it prints the verdict
# and continues (a 3-step smoke run cannot hit the guard by construction).
if not SMOKE:
    raise SystemExit("cell 10 is the smoke-derived estimate; it is a no-op in real mode - "
                     "the real launches happen in cells 11-12 without any smoke run")

base_b = effective_budget("configs/baseline_replica.yaml", N_TRAIN)
v2_b = effective_budget("configs/decider_v2.yaml", N_TRAIN)
print(f"real budgets: baseline {base_b['steps']:,} steps ({base_b['presentations']:,} presentations) | "
      f"v2 {v2_b['steps']:,} steps ({v2_b['presentations']:,} presentations)")
print(f"steps/epoch at N_TRAIN={N_TRAIN:,}: baseline {base_b['steps_per_epoch']:,} | "
      f"v2 {v2_b['steps_per_epoch']:,} (--max_steps sets the budget; --epochs does not cap it)")

q = read_job_qps(SMOKE_JOB)
if q is None:
    raise SystemExit(
        f"no sps=... windows found for {SMOKE_JOB}. The smoke job's log stream never reached "
        "CloudWatch (or it logged no training steps) - open the job's log in the SageMaker "
        "console and re-check the launch before continuing.")
qps_v2, qps_first = q["steady"], q["first"][1]
qps_base = qps_v2 / BASELINE_VS_V2_QPS   # baseline is 1.32x SLOWER (identical batch, measured)
print(f"[throughput] {SMOKE_JOB}: first window {qps_first:.2f} sps -> steady {qps_v2:.2f} sps "
      f"({q['samples']} windows; reference points from the g4dn.4xlarge plate: v2 33.3, baseline 25.2)")

h_base = base_b["presentations"] / qps_base / 3600
h_v2 = v2_b["presentations"] / qps_v2 / 3600
price, price_note = price_per_hour(INSTANCE_TYPE)
if price is not None:
    print(f"[price] {price_note or 'AmazonSageMaker'}: {price:.3f} USD/h -> "
          f"baseline {h_base * price * INSTANCE_COUNT:.0f} USD "
          f"+ v2 {h_v2 * price * INSTANCE_COUNT:.0f} USD (spot discount NOT applied)")
else:
    print("[price] unavailable: budget verdict rides on evidence 1+2 only")

total_h = (h_base + h_v2) * INSTANCE_COUNT
verdict_ok = total_h * 3600 <= MAX_RUN_SECONDS
print(f"[estimate] baseline {h_base:.2f} h + v2 {h_v2:.2f} h = {total_h:.2f} h vs window "
      f"{MAX_RUN_SECONDS / 3600:.2f} h -> {'FITS' if verdict_ok else 'DOES NOT FIT'}")
print("[caveat] the qps above comes from a 3-step smoke run on the smoke config - cross-check it "
      "against the g4dn plate (33.3/25.2 sps) before trusting any USD figure.")
if not verdict_ok:
    if SMOKE:
        print("[WARN] SMOKE mode: the guard would raise here on a real budget; continuing "
              "(a smoke run cannot hit this guard by construction)")
    else:
        raise SystemExit(
            f"the real budget does NOT fit the window: {total_h:.2f} h needed vs "
            f"{MAX_RUN_SECONDS / 3600:.2f} h available. Do NOT launch cells 11-12 until this fits; "
            "fix with a smaller budget, a faster instance class, or spot + auto_resume.")
