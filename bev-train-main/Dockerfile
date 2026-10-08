# SageMaker training image for bev-decider.
#
# Why not `uv sync`: uv.lock pins torch 2.14 with CUDA 13 wheels, which need NVIDIA
# driver >= 580. No AWS Deep Learning AMI or PyTorch DLC ships that driver, and the failure
# mode is silent — get_device() falls back to CPU and a 20-hour job becomes a multi-week one.
# The DLC ships a driver matched to the torch it contains, so we build on that and install
# only the non-torch dependencies.
#
# Verify the version skew before a long run: configs/sagemaker_smoke.yaml exercises
# Qwen3Model.from_pretrained, get_peft_model, and set_peft_model_state_dict in ~3 minutes.
# train.py:113 reads .missing_keys/.unexpected_keys off set_peft_model_state_dict, which is
# the call most likely to have moved across peft versions.
# Public gallery, not the 763104351884.dkr.ecr.<region> copy: the DLC is published to every
# AWS region there, but the public URI resolves without an ECR authorization token, which
# keeps `docker build` working for anyone without AWS credentials.
#
# Do not go back to the `2.8.0-cu128-py311-ubuntu22.04` layout AWS used before the 2.7/2.8
# transition. AWS retired that scheme; the replacement is
# `<version>-gpu-py<python>-cu<cuda>-ubuntu<os>-ec2-v<n>`, and the old tag now 404s on a
# manifest pull. buildspec.yml pins the same image so a fixed Dockerfile and a CI build do
# not drift apart.
ARG BASE=public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.49
FROM ${BASE}

RUN pip install --no-cache-dir \
      "transformers>=5.17.0" \
      "peft>=0.21.0" \
      "datasets>=5.0.1" \
      "pyyaml>=6.0.3"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# TOKENIZERS_PARALLELISM=false: tokenization.py already sets it; the tokenizer is shared
# with DataLoader worker processes (see that file's module-level os.environ.setdefault).
#
# HF_HOME must end in /hub: snapshot_download lays the cache out as <HF_HOME>/models--<repo>
# and a by-name lookup only resolves under $HF_HOME/hub.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    TOKENIZERS_PARALLELISM=false \
    RUNS_ROOT=/opt/ml/output/data/runs

WORKDIR /opt/ml/code
COPY . /opt/ml/code

# Verify the base image ships a CUDA-enabled torch build. This must NOT assert
# cuda.is_available(): `docker build` has no GPU attached, so that check can only ever fail
# here and would make the image unbuildable. Actual device visibility is asserted at runtime
# by sagemaker_entry.verify_gpu(), where a silent CPU fallback would cost days.
RUN python -c "import torch; assert torch.version.cuda is not None, 'base image has a CPU-only torch build'; print('base image torch', torch.__version__, 'built against CUDA', torch.version.cuda)"