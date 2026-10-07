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
ARG BASE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0-cu128-py311-ubuntu22.04
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

# Fail the build, not the training job, if torch cannot see the GPU.
RUN python -c "import torch; assert torch.cuda.is_available(), 'no CUDA in the base image'"