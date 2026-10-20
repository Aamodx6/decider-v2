"""Inference on a trained (or the published reference) checkpoint: loading + prediction.

Split out of the old evaluate.py so that debate-free serving code (load a checkpoint, encode a
question, apply the per-type temperature fitted on the calibration split) can be imported without
pulling in the whole evaluation battery (ECE, NLL, order-invariance gate, predictions CSV).

Temperature semantics (from calibrate.py, moved to evaluate.py's CLI): one temperature per question
type, fitted by minimising NLL on the calibration split; temperatures live in
metrics_<tag>.json["temperatures"] {"choice": T, "noul": T, "score": T}.
"""

import argparse
import json
import os

import torch

from network import build_model, load_checkpoint, load_reference_bev_decider
from tokenization import TASK_NAMES, TASK_TYPES, Encoder


def apply_per_type_temperature(logits, task_types, temperatures):
    """Scale logits so each task type's probabilities use its fitted temperature.

    logits [N, C] masked (padding columns may hold -inf); task_types [N]; temperatures maps
    task-type ids (or names) to T.  A missing type falls back to T=1.0, exactly as evaluate.py did.
    """
    temps = {TASK_TYPES.get(k, k): v for k, v in temperatures.items()}
    scaler = torch.tensor([float(temps.get(int(t), 1.0)) for t in task_types.tolist()],
                          device=logits.device)[:, None]
    return logits / scaler


def load_target(src, device=None):
    """-> (network, tokenizer, encoder, device, meta) for a `--reference` or `--checkpoint` source.

    `src` is an argparse-like object with the same fields evaluate.py's CLI declares (see
    eval_source_namespace below); the reference path also auto-derives mode/pool/max tokens from
    the published metadata, exactly as checkpoint paths derive them from meta.json.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    src = eval_source_namespace(src)
    if src.reference:
        network, meta, snapshot = load_reference_bev_decider(src.reference, device=device)
        tokenizer = AutoTokenizer.from_pretrained(snapshot)
        encoder = Encoder(tokenizer, meta["resolved_max_state_tokens"],
                          meta["resolved_max_choice_tokens"], mode="bev", pool="bev")
        return network, tokenizer, encoder, device, {"reference": src.reference, **meta}
    args = json.loads(open(os.path.join(src.checkpoint, "meta.json")).read())["args"]
    network = build_model(argparse.Namespace(**args))
    load_checkpoint(network, src.checkpoint)
    network.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args["backbone_name"])
    encoder = Encoder(tokenizer, args["max_state_tokens"], args["max_choice_tokens"],
                      mode=args.get("mode", "bev"), pool=args.get("pool", "bev"))
    return network, tokenizer, encoder, device, {"checkpoint": src.checkpoint, "args": args}


def eval_source_namespace(src):
    """Fill sentinel defaults for the two mutually-exclusive sources when the caller built a bare
    Namespace without them (tests, notebooks)."""
    if getattr(src, "reference", "") in (None, "") and getattr(src, "checkpoint", "") in (None, ""):
        raise ValueError("load_target needs --reference or --checkpoint")
    return src


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(network, batch, device, amp_dtype=torch.float32, temperatures=None, task_type_key="task_type",
            logits_key=None):
    """-> logits [B, C]; when `temperatures` is given, per-type scaled (probability-preserving in the
    argmax, calibration-corrected in NLL/ECE).  batch must already be to_device()'d."""
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
        logits = network(**batch).float()
    if temperatures:
        logits = apply_per_type_temperature(logits, batch[task_type_key], temperatures)
    return logits


def predictions_rows(logits, labels, task_types, domains, k_options):
    """The per-example rows evaluate.py's CSV writer emits, given masked logits."""
    rows = []
    for i in range(len(labels)):
        rows.append([i, TASK_NAMES[int(task_types[i])], domains[i], int(k_options[i]),
                     int(labels[i]), int(logits[i].argmax()), round(float(logits[i, labels[i]]), 6)])
    return rows
