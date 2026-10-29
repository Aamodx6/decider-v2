# Cell 14 - evaluate BOTH arm checkpoints and the untrained reference on the test split.
#
# Each eval is a real SageMaker job running download_eval.py (which pulls the checkpoint prefix
# through the container's own creds, then shells out to evaluate.py). The eval hyperparameters
# are all evaluate.py argparse names - env_check=1 so evaluate.py pins the container libraries
# exactly like train.py does (transformers/peft are NOT preinstalled in the framework image).
# Outputs land at S3_EVAL/<job>/output/data/<tag>.json and are pulled to runs/evals/<tag>.json.
REFERENCE_ID = "bev-decider-0.4B"   # published v1 recipe, no fine-tune (the un-trained lamp)

EVAL_FLAGS = ["checkpoint", "reference", "test_channel", "calib_channel", "out", "tag", "env_check",
              "limit", "batch_size", "invariance", "invariance_examples", "invariance_shuffles",
              "predictions_out", "max_state_tokens", "max_choice_tokens", "mode", "pool"]

def launch_eval(tag, checkpoint_uri, cfg=None, limit="1600"):
    """One evaluate.py job; returns its job name (no body-side prediction files needed)."""
    cfg = cfg or f"configs/{'smoke' if SMOKE and tag != 'reference' else 'decider_v2'}.yaml"
    data = (yaml.safe_load((REPO / cfg).read_text(encoding="utf-8")) or {}).get("data", {})
    hyper = {"checkpoint": str(checkpoint_uri or ""), "tag": tag,
             "limit": str(limit), "batch_size": "8",
             "invariance": "1", "invariance_examples": "32", "invariance_shuffles": "32",
             "env_check": "1",
             "mode": str(data.get("mode", "bev")), "pool": str(data.get("pool", "bev")),
             "max_state_tokens": str(data.get("max_state_tokens", 1024)),
             "max_choice_tokens": str(data.get("max_choice_tokens", 64))}
    if tag == "reference":
        # evaluate.py's published-model path: the checkpoint is empty and --reference names the repo
        hyper["reference"] = REFERENCE_ID
    unknown = [k for k in hyper if k not in EVAL_FLAGS]
    if unknown:
        raise SystemExit(f"eval hyperparameters are not evaluate.py argv ({unknown})")

    base = job_prefix(f"eval-{tag}")
    est_kwargs = dict(
        entry_point="download_eval.py", source_dir=str(BUILD), role=ROLE,
        framework_version="2.7.1", py_version="py312",
        instance_type=INSTANCE_TYPE, instance_count=1,
        sagemaker_session=sagemaker_session,
        output_path=S3_EVAL, hyperparameters=hyper,
        # An eval job reads its checkpoint from S3 and writes no checkpoint of its own, so it needs
        # no checkpoint_s3_uri/local_path pair (and `checkpoint_config` is not a v2 kwarg).
        use_spot_instances=False, max_run=2 * 3600, max_wait=None,
        keep_alive_period_in_seconds=0, base_job_name=base)
    check_estimator_kwargs(est_kwargs, label="download_eval")
    est = sagemaker.estimator.PyTorch(**est_kwargs)
    est.fit(inputs={"test": f"{S3_DATA}/test", "calib": f"{S3_DATA}/calib"}, wait=True, logs="None")
    name = est.latest_training_job.name
    state["jobs"][name] = {"arm": f"eval-{tag}", "config": cfg, "hyperparameters": hyper,
                           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "Completed",
                           "base_job_name": base}
    save_state()
    prefix = f"{S3_EVAL}/{name}/output/data/"
    local = RUNS_LOCAL / "evals" / f"{tag}.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    res = [o["Key"] for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=key_prefix(prefix))
           for o in page.get("Contents", []) if o["Key"].endswith(".json")]
    if len(res) != 1:
        print(f"NOT RESOLVED: expected exactly one result json under {prefix}, found {res}")
        return name
    s3.download_file(BUCKET, res[0], str(local))
    d = json.loads(local.read_text(encoding="utf-8"))
    print(f"[eval:{tag}] {', '.join(f'{k}={v}' for k, v in d.items() if not isinstance(v, dict))}")
    return name


# the arm checkpoint URIs: S3_CKP/<base>/<TrainingJobName>/checkpoints/best (the CheckpointConfig
# mirror SM writes during the run; 'best' because checkpoint_selection picked best VALIDATION acc)
def best_uri(job_name):
    meta = state["jobs"][job_name]
    return f"{S3_CKP}/{meta['base_job_name']}/{job_name}/checkpoints/best"

launch_eval("reference", checkpoint_uri="", cfg="configs/decider_v2.yaml")

def completed(arm):
    out = []
    for n in state["launches"].get(arm, []):
        if describe(n)["TrainingJobStatus"] == "Completed":
            out.append(n)
    return out

if SMOKE:
    launch_eval("smoke-arm", checkpoint_uri=best_uri(SMOKE_JOB), cfg="configs/smoke.yaml")
else:
    for arm, cfg in (("baseline", "configs/baseline_replica.yaml"),
                     ("v2", "configs/decider_v2.yaml")):
        pool = completed(arm)
        if not pool:
            print(f"[skip] {arm} eval: no Completed training job (cells 11-13 launch + monitor first)")
            continue
        launch_eval(arm, checkpoint_uri=best_uri(pool[-1]), cfg=cfg)
