"""Drive a SageMaker training job for the bev-decision ladder.

    python scripts/launch_sagemaker.py --stage smoke
    python scripts/launch_sagemaker.py --stage pilot
    python scripts/launch_sagemaker.py --stage full --config configs/full_lora_all20.yaml
    python scripts/launch_sagemaker.py --stage full --config configs/full_stage1_arch.yaml --spot
    python scripts/launch_sagemaker.py --stage eval --job-name jev-full-spot

Order matters. smoke catches the DLC version skew in ~3 minutes for a few cents; pilot
measures throughput so the full run's MaxRuntimeInSeconds is grounded rather than guessed;
only then do you spend 15-24h per rung.

The ladder itself is one change at a time (arch.md:310):
    configs/full_lora_all20.yaml    LoRA scope: last 12 kept layers -> all 20
    configs/full_stage1_arch.yaml    + arch.md stage-1 recipe (lr 2e-4, warmup 3%, batch 128)
    configs/full_loss_arch.yaml      + arch.md stage-1 losses (smoothing, EMD, type balance)

instance_count is always 1. There is no DDP in this codebase, so N instances would train N
identical models and N times the bill.

Prerequisites:
  - aws CLI configured, credentials valid (`aws sts get-caller-identity`)
  - vCPU quota raised (L-1216C47) to at least 16
  - the HF cache already mirrored: scripts/prepare_s3_cache.py upload --bucket $BUCKET
  - the image built and pushed (see SAGEMAKER.md step 3)
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = os.environ.get("JEV_BUCKET", "")
DEFAULT_ROLE = os.environ.get("JEV_ROLE_ARN", "")
DEFAULT_IMAGE = os.environ.get("JEV_IMAGE", "")

EFFECTIVE_BATCH_HINT = "keep this equal to the config's batch_size * grad_accum_steps"


def estimator(config, args):
    import sagemaker
    from sagemaker.estimator import Estimator

    hyperparams = {"config": config, "name": args.job_name or Path(config).stem}
    if args.resume_state:
        hyperparams["resume_state"] = args.resume_state

    kwargs = dict(
        image_uri=args.image,
        source_dir=str(ROOT),
        entry_point="sagemaker_entry.py",
        role=args.role,
        hyperparameters=hyperparams,
        instance_count=1,
        instance_type=args.instance_type,
        volume_size=args.volume_size,
        output_s3_uri=f"s3://{args.bucket}/runs/{args.job_name or Path(config).stem}",
        base_job_name=args.job_name or f"jev-{Path(config).stem}",
        environment={
            "S3_SYNC_URI": f"s3://{args.bucket}/runs",
            "HF_HOME": "/cache/hf",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "RUNS_ROOT": "/opt/ml/output/data/runs",
            "PYTHONUNBUFFERED": "1",
        },
        sagemaker_session=sagemaker.Session(),
    )

    if args.stage == "full":
        kwargs["max_runtime"] = args.max_runtime
        if args.spot:
            # Spot requires max_runtime <= MaxRuntimeInSeconds; the SDK folds that into
            # the training job request, so pass the same number to both.
            kwargs["use_spot_instances"] = True
            kwargs["max_runtime"] = min(args.max_runtime, args.spot_max_runtime)
            kwargs["checkpoint_config"] = sagemaker.checkpoint.CheckpointConfig(
                # one dir, not a list: the resume state is a single rolling slot plus a
                # pointer file (logger.py:save_resume_state), so there is nothing to fan out
                local_checkpoint_dir="/opt/ml/checkpoint",
                s3_uri=f"s3://{args.bucket}/ckpt/{args.job_name}",
            )
    else:
        kwargs["max_runtime"] = args.max_runtime

    return Estimator(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["smoke", "pilot", "full", "eval"], required=True)
    parser.add_argument("--config", default=None,
                        help="training config (default: the stage's own config)")
    parser.add_argument("--eval-config", default=None,
                        help="dataset config to evaluate, e.g. all")
    parser.add_argument("--eval-job", default=None,
                        help="name of the training job whose checkpoint to score")
    parser.add_argument("--eval-max-questions", type=int, default=0, help="0 = full test split")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--instance-type", default="ml.g5.4xlarge")
    parser.add_argument("--volume-size", type=int, default=200)
    parser.add_argument("--max-runtime", type=int, default=43200, help="seconds")
    parser.add_argument("--spot-max-runtime", type=int, default=43200)
    parser.add_argument("--spot", action="store_true", help="use Spot (requires the S3 resume path)")
    parser.add_argument("--resume-state", default=None,
                        help="restored runs/<id>; set by the resume flow after a Spot interruption")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    for name, value in (("--bucket", args.bucket), ("--role", args.role), ("--image", args.image)):
        if not value:
            sys.exit(f"{name} is required (or set JEV_BUCKET / JEV_ROLE_ARN / JEV_IMAGE)")

    if args.config is None:
        args.config = {"smoke": "configs/sagemaker_smoke.yaml",
                       "pilot": "configs/sagemaker_pilot.yaml",
                       "full": "configs/full_lora_all20.yaml",
                       "eval": "configs/sagemaker_full.yaml"}[args.stage]
    if not (ROOT / args.config).exists():
        sys.exit(f"config not found: {args.config}")

    if args.stage == "eval":
        if not args.eval_job:
            sys.exit("--eval-job is required: the name of the run whose checkpoint to score")
        args.job_name = args.job_name or "jev-eval"
        est = estimator(args.config, args)
        est.fit(inputs={"checkpoint": f"s3://{args.bucket}/runs/{args.eval_job}/"})
        est.wait()
        print(f"eval done -> s3://{args.bucket}/runs/{args.eval_job}/eval_metrics.json", flush=True)
        return

    print(f"stage={args.stage} config={args.config} instance={args.instance_type} "
          f"spot={args.spot} max_runtime={args.max_runtime}s", flush=True)
    print(f"  effective batch reminder: {EFFECTIVE_BATCH_HINT}", flush=True)

    est = estimator(args.config, args)
    est.fit()
    if args.no_wait:
        print(f"submitted {est.job_name}", flush=True)
        return
    est.wait()
    print(f"done: {est.job_name}", flush=True)
    print(f"artifacts -> s3://{args.bucket}/runs/{args.job_name or Path(args.config).stem}/", flush=True)


if __name__ == "__main__":
    main()