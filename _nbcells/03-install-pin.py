# Cell 3 - pin the SageMaker SDK. SageMaker 3.x redesigned the Estimator surface (checkpoint
# kwargs moved onto estimator properties, hyperparameter typing changed, describe endpoints
# renamed); this notebook validates against the 2.247 constructor with inspect.signature. Its
# checkpoint kwargs are the v2 `checkpoint_s3_uri` / `checkpoint_local_path` pair - there is no
# `checkpoint_config=` kwarg on PyTorch in any 2.x. Major 3 in this kernel is a hard
# stop so the buggy path is never reached (cells fail HERE instead of three hours in).
import importlib.metadata as im
import json
import math
import os
import re
import subprocess
import sys
import time

def _require(name, lo, hi):
    """Version window check: '2.247.0' -> (2, 247); raises SystemExit with the pip fix."""
    pin = f"{name}>={lo[0]}.{lo[1]},<{hi[0]}"
    try:
        version = tuple(int(x) for x in im.version(name).split(".")[:2])
    except im.PackageNotFoundError:
        raise SystemExit(f"{name} is NOT installed in this kernel - pip install \"{pin}\"") from None
    if not (lo <= version < hi):
        raise SystemExit(f"{name} {version} outside the supported window [{lo}, {hi}) - "
                         f"pip install \"{pin}\"")
    return version

SM2 = _require("sagemaker", lo=(2, 247), hi=(3, 0))

import boto3
import sagemaker
import yaml

print("sagemaker", sagemaker.__version__, "| boto3", boto3.__version__, "| pyyaml", yaml.__version__)
