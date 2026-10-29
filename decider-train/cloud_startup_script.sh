#!/usr/bin/env bash
# cloud_startup_script.sh -- bootstrap a plain GPU box (RunPod / Lambda / EC2 / a colleague's dev
# server) and run one training arm exactly as the SageMaker container would.
#
# This is the non-SageMaker path. It deliberately mirrors the container contract:
#   * torch comes from the lockfile, NOT from pip-installing over the image's CUDA build;
#   * requirements.txt is not used here at all (it exists to pin the SageMaker job);
#   * the preflight gates (--env_check, --selftest) run exactly as they do in the cloud job.
#
# Usage:
#   S3_BUCKET=s3://my-bucket/decider uv run --project . python train.py ...   # no, use this:
#   S3_BUCKET=s3://my-bucket/decider CONFIG=configs/decider_v2.yaml ./cloud_startup_script.sh
#
# Environment:
#   S3_BUCKET   s3://bucket/prefix for checkpoints and metrics. REQUIRED.
#   CONFIG      configs/*.yaml to train (default: configs/smoke.yaml -- the cheap one).
#   RUN_NAME    name for the output prefix under S3_BUCKET (default: basename of CONFIG + UTC stamp).
#   HF_TOKEN    needed only if the backbone has to be downloaded (Qwen3-0.6B).
#   ARGS        extra train.py flags, e.g. ARGS="--seed 1 --max_steps 2000"
#
# Anything that fails uploads what exists and exits non-zero -- a run that dies at step 9,000 of
# 14,688 must not lose its checkpoints.

set -euo pipefail

S3_BUCKET="${S3_BUCKET:-}"
CONFIG="${CONFIG:-configs/smoke.yaml}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG%.yaml}")-$(date -u +%Y%m%dT%H%M%SZ)}"
ARGS="${ARGS:-}"

if [ -z "$S3_BUCKET" ]; then
    echo "cloud_startup_script.sh: S3_BUCKET is required (e.g. S3_BUCKET=s3://bucket/decider)" >&2
    exit 2
fi

cd "$(dirname "$0")"

LOCAL_OUT="./outputs/${RUN_NAME}"
CKPT_DIR="${LOCAL_OUT}/checkpoints"
mkdir -p "${LOCAL_OUT}" "${CKPT_DIR}"

upload() {
    local status=$?
    echo "==> uploading to ${S3_BUCKET}/${RUN_NAME}/ (exit status ${status})"
    # --only-show-errors keeps a 4-hour run's log readable; failure here must not mask the
    # training status, so the || true is deliberate and the training status is re-raised below.
    aws s3 sync "${LOCAL_OUT}" "${S3_BUCKET}/${RUN_NAME}/" --only-show-errors || true
    exit "${status}"
}
trap upload EXIT

echo "==> 1/4 toolchain"
if ! command -v uv >/dev/null 2>&1; then
    echo "    uv not found; installing to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version
python3 --version || true

echo "==> 2/4 dependencies (frozen lockfile)"
# --frozen: fail if uv.lock disagrees with pyproject.toml rather than silently re-resolving. A
# drifting lockfile is how a 14,688-step run ends up on a torch nobody measured.
uv sync --frozen --extra dev

echo "==> 3/4 preflight (version pins + the 4-D mask / option-order gates)"
# Same gates the SageMaker job runs: it aborts here rather than at step 14,688.
uv run --frozen python train.py --config "${CONFIG}" --env_check 1 --selftest 1 \
    --max_steps 0 --model_dir "${LOCAL_OUT}/preflight" --output_dir "${LOCAL_OUT}/preflight" \
    --checkpoint_dir "${CKPT_DIR}" >/dev/null 2>&1 || {
        echo "    preflight FAILED -- re-running verbosely"
        uv run --frozen python train.py --config "${CONFIG}" --env_check 1 --selftest 1 \
            --max_steps 0 --model_dir "${LOCAL_OUT}/preflight" \
            --output_dir "${LOCAL_OUT}/preflight" --checkpoint_dir "${CKPT_DIR}"
        exit 1
    }
echo "    preflight PASS"

echo "==> 4/4 training ${CONFIG} -> ${LOCAL_OUT}"
# shellcheck disable=SC2086  # ARGS is intentionally word-split
uv run --frozen python train.py \
    --config "${CONFIG}" \
    --model_dir "${LOCAL_OUT}/model" \
    --output_dir "${LOCAL_OUT}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --auto_resume 1 \
    ${ARGS}

echo "==> done: ${LOCAL_OUT}"
