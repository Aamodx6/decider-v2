# SageMaker training — decider-train

Run these cells top to bottom. `SMOKE` in cell 2 decides everything downstream: SMOKE runs the
smoke config with a 3-step budget and never launches the real arms; the real run launches
`configs/baseline_replica.yaml` and `configs/decider_v2.yaml` at their own budgets. The two
modes keep separate state, budget fingerprints and launch prefixes, so a smoke run can never
collide with a real job.

| # | cell | what it does |
|---|------|--------------|
| 1 | build-sources | assembles the allowlist pack under `build/sagemaker_src/` (git-ignored) — the estimator's `source_dir` |
| 2 | config | region/bucket/instance, `USE_SPOT=False`, per-mode state file paths |
| 3 | install | asserts `2.247 <= sagemaker < 3` in this kernel (the v3 SDK changed this estimator surface) |
| 4 | clients | boto3 clients, role, bucket, S3 prefixes |
| 5 | helpers | budget arithmetic from the same yaml the job receives, fingerprint/idempotency, `launch_job()` |
| 6 | prepare | carves the four shards with `dataset.py prepare` |
| 7 | upload | uploads them; skips a split when S3 already holds the same ContentLength |
| 8 | tests | runs `tests/` in the repo checkout — no GPU spend before the gates pass |
| 9 | smoke | one real `train.py` job, 3 steps, g4dn — proves the pack and the channels work |
| 10 | estimate | reads the smoke job's qps from CloudWatch, Price-List rate, and refuses to launch if the real budget cannot finish inside the run window |
| 11–12 | launches | both arms via `hyperparameters={"config": yaml, "seed": 0}` |
| 13 | monitor | `describe_training_job` until terminal; downloads artifacts; fails if `final_step != max_steps` |
| 14 | evaluate | both arm checkpoints + the untrained reference on the test split via `download_eval.py` + `evaluate.py` |
| 15 | results | `runs/RESULTS.md` + accuracy/pacing charts |
| 16 | ablations | the five Tier-2 ablation configs (five more launch jobs, fingerprinted the same way) |
| 17 | cleanup | stops every dangling `InProgress` job for this notebook (no delete API — stop is terminal) |

State lives in two JSON files next to this notebook (git-ignored): `decider_v2_state_smoke.json`
and `decider_v2_state.json`. Every launch is fingerprinted by its effective budget; relaunching
the same fingerprint is refused while the old job is `InProgress` and only reused when `Completed`.
