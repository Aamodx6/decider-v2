# Cell 5 - budget arithmetic, state helpers, the reuse guard and the launcher. Every launch cell
# below is thin; everything lives here once.
#
# The budget helper re-implements train.py's precedence (argparse < yaml < CLI) for the four
# budget-relevant arguments, so each fingerprint comes from the SAME yaml the job receives:
# change a config -> new fingerprint -> fresh job; unchanged yaml -> the guard decides.
import hashlib
import inspect as _inspect

# BUDGET_DEFAULTS mirrors train.py's argparse defaults verbatim (tests/test_core.py section (h)
# re-checks them against train.py itself, so a drift there fails CI before any launch).
BUDGET_KEYS = ["max_steps", "epochs", "batch_size", "grad_accum_steps"]
BUDGET_DEFAULTS = {"max_steps": -1, "epochs": 1, "batch_size": 32, "grad_accum_steps": 2}

# What SageMaker scrapes out of the job's stdout. One capture group each (SageMaker takes the last
# group as the value); the trailing context keeps "val_loss=" from feeding the training series.
METRIC_DEFINITIONS = [
    {"Name": "train:loss", "Regex": "loss=([0-9.]+) grad_norm="},
    {"Name": "train:acc", "Regex": "acc=([0-9.]+) lr_lora="},
    {"Name": "train:sps", "Regex": "sps=([0-9.]+)"},
    {"Name": "val:loss", "Regex": "val_loss=([0-9.]+)"},
    {"Name": "val:acc", "Regex": "val_acc=([0-9.]+)"},
]


def effective_budget(config_rel, n_train, overrides=None):
    """Budget of a run exactly as train.py builds it.

    train.py's rule (the fixed budget footgun): totals an epoch when --max_steps is unset, and
    honors --max_steps when it is set; with --max_steps set train.py ignores --epochs entirely
    and iterates epochs until total_steps, so steps_from_epochs is informational only.
    Presentations = steps x batch.
    """
    cfg = yaml.safe_load((REPO / config_rel).read_text(encoding="utf-8")) or {}
    optim = cfg.get("optim", {}) or {}
    b = {k: int(optim.get(k, BUDGET_DEFAULTS[k])) for k in BUDGET_KEYS}
    for k, v in (overrides or {}).items():
        if k not in BUDGET_DEFAULTS:
            raise KeyError(f"override {k} is not budget-relevant ({sorted(BUDGET_DEFAULTS)})")
        b[k] = int(v)
    # train.py applies the config's data.limit_train when loading the channel; mirror it or the
    # steps/epoch here is computed on rows the job never sees (smoke.yaml sets it to 2000).
    limit = int((cfg.get("data") or {}).get("limit_train") or 0)
    n_used = min(n_train, limit) if limit > 0 else n_train
    # steps_per_epoch = ceil(ceil(N / micro-batch) / accumulation): identical double ceil
    steps_per_epoch = math.ceil(math.ceil(n_used / b["batch_size"]) / b["grad_accum_steps"])
    b["steps_per_epoch"] = steps_per_epoch
    b["steps_from_epochs"] = steps_per_epoch * b["epochs"]
    b["steps"] = b["max_steps"] if b["max_steps"] > 0 else b["steps_from_epochs"]
    b["presentations"] = b["steps"] * b["batch_size"] * b["grad_accum_steps"]
    return b


def fingerprint(b):
    keys = ["max_steps", "epochs", "batch_size", "grad_accum_steps", "steps",
            "steps_per_epoch", "steps_from_epochs", "presentations"]
    payload = {k: b[k] for k in keys}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def job_prefix(arm):
    # base_job_name must be alphanumeric+hyphen (SageMaker rule): the config's ARM_TAG may carry
    # an underscore, so replace it here (the bucket folder keeps the readable underscore).
    return f"{'smoke-' if SMOKE else ''}decider-{ARM_TAG.replace('_', '-')}-{arm}"


def save_state():
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def record_launch(arm, config_rel, overrides, hyper, job_name, budget):
    state["jobs"][job_name] = {
        "arm": arm, "config": config_rel,
        "hyperparameters": dict(hyper), "overrides": dict(overrides or {}),
        "fingerprint": fingerprint(budget),
        "budget": {k: budget[k] for k in ("steps", "steps_per_epoch", "steps_from_epochs",
                                          "presentations")},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "InProgress",
        "base_job_name": job_prefix(arm),
    }
    state["launches"].setdefault(arm, []).append(job_name)
    save_state()
    print(f"[launch] {arm}: {job_name}")
    print(f"         fingerprint={fingerprint(budget)}  "
          f"budget={budget['steps']:,} steps x {budget['batch_size']}x{budget['grad_accum_steps']}"
          f" = {budget['presentations']:,} presentations")


def guard_reuse(arm, fp):
    """None -> do a fresh launch; else the completed job name to reuse. Never duplicate.

    Lifecycle is checked against AWS (describe_training_job), not the state file: the state
    file only remembers which job belongs to which arm and budget.
    """
    for name, meta in state["jobs"].items():
        if meta.get("arm") != arm:
            continue
        live = sm.describe_training_job(TrainingJobName=name)["TrainingJobStatus"]
        meta["status"] = live
        if live == "InProgress" and meta.get("fingerprint") == fp:
            raise SystemExit(
                f"{arm}: job {name} is already InProgress with the same budget fingerprint {fp}. "
                f"The guard refuses a duplicate. To force a fresh run, stop/delete the job in "
                f"SageMaker and remove its entry from {STATE_PATH.name}.")
        if live == "Completed" and meta.get("fingerprint") == fp:
            print(f"[reuse] {arm}: completed job {name} already has fingerprint {fp}; not re-launching")
            save_state()
            return name
    return None


# ------------------------------------------------------------------ estimator kwargs check
_sig_cache = None

def check_estimator_kwargs(kwargs, label="PyTorch"):
    """Every launched kwarg must be a NAMED parameter of the real SDK constructor.

    PyTorch.__init__ (and the base Estimator) accept **kwargs, so a typo would be swallowed
    silently and the job would run with the DEFAULT for that argument - e.g. no checkpoints,
    no spot cap. Failing here costs nothing; failing after 12k steps costs the run window.
    """
    global _sig_cache
    if _sig_cache is None:
        _sig_cache = _inspect.signature(sagemaker.estimator.PyTorch.__init__)
    named = set(_sig_cache.parameters)
    orphans = [k for k in kwargs if k not in named]
    if orphans:
        raise SystemExit(
            f"{label}: kwargs not in sagemaker.estimator.PyTorch.__init__ (sagemaker "
            f"{sagemaker.__version__}) -> {orphans}. The SDK surface moved (or a typo) - the "
            "**kwargs catch-all would have run the job with the default for these instead of "
            "raising. Check sagemaker release notes and fix the launcher.")
    return True


# ------------------------------------------------------------------ the launcher
def launch_job(arm, config_rel, n_train, overrides=None, seed=0, wait=False):
    """Create the PyTorch estimator, validate its kwargs, record the launch in the state file.

    Fixed vs the old argv serialization:
      * hyperparameters = {"config": yaml, "seed": 0, ...}: one hyperparameter per train.py
        argparse argument. No "sagemaker_args" string (that escape hatch stays in train.py but
        the notebook never uses it - shell-serialized argv is where old silent quirk bugs lived).
      * every estimator kwarg goes through check_estimator_kwargs BEFORE fit().
      * the job name comes from est.latest_training_job.name, never from estimator internals.
    """
    budget = effective_budget(config_rel, n_train, overrides)
    reused = guard_reuse(arm, fingerprint(budget))
    if reused:
        return None

    hyper = {"config": config_rel, "seed": str(seed)}
    for k, v in (overrides or {}).items():
        if k in BUDGET_DEFAULTS:
            hyper[k] = str(v)
    if USE_SPOT and hyper.get("auto_resume") != "1":
        raise SystemExit(
            "USE_SPOT=True without auto_resume=1: a spot interruption silently restarts from "
            "zero without checkpoints + auto-resume, which is exactly the failure the config "
            "was written to avoid. Set hyper['auto_resume']='1' or USE_SPOT=False.")
    if hyper.get("auto_resume") == "1" and not USE_SPOT:
        print("[note] auto_resume=1 requested without USE_SPOT: it is harmless (no interruption "
              "-> nothing to resume), but unusual - double-check the intent.")

    base = job_prefix(arm)
    assert "_" not in base and job_prefix(arm) == base, base
    ckpt_uri = f"{S3_CKP}/{base}"

    est_kwargs = dict(
        entry_point="train.py",
        source_dir=str(BUILD),             # the allowlist pack from cell 1, NOT the repo root
        role=ROLE,
        framework_version="2.7.1", py_version="py312",
        instance_type=INSTANCE_TYPE, instance_count=INSTANCE_COUNT,
        sagemaker_session=sagemaker_session,
        output_path=S3_RUNS,
        hyperparameters=hyper,
        # The v2 estimator takes these two kwargs and builds the API's CheckpointConfig={S3Uri,
        # LocalPath} from them. There is no `checkpoint_config=` kwarg on PyTorch (that is the
        # v3 ModelTrainer / the API field name); **kwargs would swallow it, and
        # check_estimator_kwargs below rejects it instead of launching a job that never syncs.
        checkpoint_s3_uri=ckpt_uri,               # /opt/ml/checkpoints -> S3_CKP/<base>
        checkpoint_local_path="/opt/ml/checkpoints",
        # MetricDefinitions: SageMaker scrapes the job's stdout for these. Without them the
        # console shows no charts and the training job page reports "No metrics". The regexes are
        # anchored on the surrounding tokens because the log line is
        #   step=N loss=... grad_norm=... acc=... lr_lora=... lr_head=... sps=...
        # and a bare "loss=" would also match "val_loss=".
        metric_definitions=METRIC_DEFINITIONS,
        # The model artifacts here are a LoRA-merged checkpoint plus tokenizer, and tarring them
        # adds minutes to every job for no benefit. The container writes them uncompressed.
        disable_output_compression=True,
        use_spot_instances=USE_SPOT,
        max_run=MAX_RUN_SECONDS,
        max_wait=MAX_RUN_SECONDS if USE_SPOT else None,
        keep_alive_period_in_seconds=int(KEEP_ALIVE * 60),
        base_job_name=base,
    )
    check_estimator_kwargs(est_kwargs)

    est = sagemaker.estimator.PyTorch(**est_kwargs)
    est.fit(inputs={"train": f"{S3_DATA}/train", "val": f"{S3_DATA}/val"},
            wait=wait, logs="None")        # no dead first-run eyeballing: the monitor cell owns tailing
    name = est.latest_training_job.name     # raises if the job never started
    record_launch(arm, config_rel, overrides, hyper, name, budget)
    print(f"         ckpt prefix {ckpt_uri}/{name}")
    return est


# ------------------------------------------------------------------ CloudWatch throughput
def read_job_qps(job_name, minutes_back=90):
    """Steady-state `sps` (presentations/second) from the job's CloudWatch log stream.

    filter_log_events (NOT describe: the rate meter should not pay for job descriptions), and
    only the LAST few windows: the first window mixes in worker start-up, steady state is the
    tail (measured: the first window reads ~5x low on cold CUDA compile).
    """
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes_back)
    samples = []
    for page in logs.get_paginator("filter_log_events").paginate(
            logGroupName="/aws/sagemaker/TrainingJobs", logStreamNamePrefix=job_name,
            startTime=int(start.timestamp() * 1000)):
        for ev in page.get("events", []):
            m = re.search(r"\bsps=([0-9]+(?:\.[0-9]+)?)", ev["message"])
            if m:
                samples.append((ev["timestamp"], float(m.group(1))))
    if not samples:
        return None
    tail = [v for _t, v in samples[-5:]]
    return {"samples": len(samples),
            "median": sorted(tail)[len(tail) // 2],
            "steady": max(tail),
            "first": samples[0]}


# ------------------------------------------------------------------ Price List
def price_per_hour(instance_type, region=REGION):
    """USD/h for an AmazonSageMaker instance-hour; None when the API is unreachable/empty.

    Then prints NOT RESOLVED: a pricing failure must never invent a number and must never kill
    the run - the budget verdict in cell 10 falls back to the measured-rate path.
    """
    filters = [{"Type": "TERM_MATCH", "Field": "ServiceCode", "Value": "AmazonSageMaker"},
               {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
               {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
               {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"}]
    try:
        for page in pricing.get_paginator("get_products").paginate(ServiceCode="AmazonSageMaker", Filters=filters):
            for item in page.get("PriceList", []):
                rec = json.loads(item)
                attrs = rec.get("product", {}).get("attributes", {})
                if region != attrs.get("regionCode"):
                    continue                 # get_products honors the filter, but be strict anyway
                demand = rec.get("terms", {}).get("OnDemand", {})
                for offer in demand.values():
                    for dim in offer.get("priceDimensions", {}).values():
                        rate = dim.get("pricePerUnit", {}).get("USD")
                        if rate and dim.get("unit", "").lower().startswith("hour"):
                            return float(rate), dim.get("description", "")
    except Exception as e:  # noqa: BLE001 - pricing is best-effort by design (NO hard dependency)
        print(f"pricing resolution failed ({type(e).__name__}: {e})")
        print("NOT RESOLVED - Price List response unusable for this region; "
              "cell 10 falls back to the measured-rate budget verdict.")
        return None, ""
    print("NOT RESOLVED - no AmazonSageMaker OnDemand offer matched "
          f"(instanceType {instance_type}, regionCode {region}, operatingSystem Linux)")
    return None, ""
