"""Per-type temperature calibration.

One temperature per question type, fitted by minimising NLL on the calibration split (which is
carved out of TRAIN before training and never used for early stopping).  Reports ECE (10 bins,
top-label) before and after.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DecisionDataset, collate, load_split_table
from losses import expected_calibration_error
from model import build_model, load_checkpoint, load_reference_bev_decider
from serialize import TASK_NAMES, TASK_TYPES, Encoder, pad_ranks


def fit_temperature(logits, labels, task_id, lo=0.05, hi=10.0, iters=200):
    """Grid + golden-section on log T for one type (NLL is smooth and 1-D; this is exact enough)."""
    grid = torch.logspace(float(torch.log10(torch.tensor(lo))), float(torch.log10(torch.tensor(hi))), 60)
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        nll = F.cross_entropy(logits / float(T), labels).item()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T, best_nll


@torch.no_grad()
def collect(network, loader, device, pool, amp_dtype):
    per_type = {t: {"logits": [], "labels": []} for t in TASK_TYPES.values()}
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = network(**batch)
        logits = logits.float()
        for i, t in enumerate(batch["task_type"].tolist()):
            per_type[t]["logits"].append(logits[i].cpu())
            per_type[t]["labels"].append(int(batch["labels"][i]))
    return {t: (torch.stack(v["logits"]), torch.tensor(v["labels"])) for t, v in per_type.items()
            if v["logits"]}


def report(per_type, temps):
    rows = {}
    for t, (logits, labels) in per_type.items():
        T = temps.get(t, 1.0)
        scaled = logits / T
        probs = torch.softmax(scaled, dim=-1)
        rows[TASK_NAMES[t]] = {
            "n": int(labels.numel()),
            "temperature": float(T),
            "nll_before": float(F.cross_entropy(logits, labels)),
            "nll_after": float(F.cross_entropy(scaled, labels)),
            "ece_before": float(expected_calibration_error(torch.softmax(logits, -1), labels)),
            "ece_after": float(expected_calibration_error(probs, labels)),
            "accuracy": float((logits.argmax(-1) == labels).float().mean()),
        }
    all_logits = torch.cat([v[0] for v in per_type.values()])
    all_labels = torch.cat([v[1] for v in per_type.values()])
    ts = torch.tensor([temps.get(t, 1.0) for t, row in sorted(per_type.items()) for _ in row[1]])
    rows["overall"] = {
        "n": int(all_labels.numel()),
        "ece_before": float(expected_calibration_error(torch.softmax(all_logits, -1), all_labels)),
        "ece_after": float(expected_calibration_error(torch.softmax(all_logits / ts, -1), all_labels)),
        "nll_before": float(F.cross_entropy(all_logits, all_labels)),
        "nll_after": float(F.cross_entropy(all_logits / ts, all_labels)),
    }
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="", help="dir written by train.py (or 'reference')")
    p.add_argument("--reference", default="", help="'avbiswas/bev-decider-0.4B' to calibrate that")
    p.add_argument("--calib_channel", default=os.environ.get("SM_CHANNEL_CALIB", "./data/calib"))
    p.add_argument("--max_state_tokens", type=int, default=1024)
    p.add_argument("--max_choice_tokens", type=int, default=64)
    p.add_argument("--mode", default="bev")
    p.add_argument("--pool", default="bev")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", default=os.environ.get("SM_OUTPUT_DATA_DIR", "./runs/output"))
    src = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    encoder = Encoder(tokenizer, src.max_state_tokens, src.max_choice_tokens, mode=src.mode, pool=src.pool)
    pad_id = tokenizer.pad_token_id or 0

    if src.reference:
        network, meta, snapshot = load_reference_bev_decider(src.reference, device=device)
        from transformers import AutoTokenizer as AT
        tokenizer = AT.from_pretrained(snapshot)
        encoder = Encoder(tokenizer, meta["resolved_max_state_tokens"], meta["resolved_max_choice_tokens"],
                          mode="bev", pool="bev")
    else:
        args = json.loads(open(os.path.join(src.checkpoint, "meta.json")).read())["args"]
        ns = argparse.Namespace(**args)
        network = build_model(ns)
        load_checkpoint(network, src.checkpoint)
        network.to(device).eval()
    amp = torch.bfloat16

    table = load_split_table(src.calib_channel, src.limit or None)
    ds = DecisionDataset(table, encoder, shuffle_options=False, seed=0)
    loader = DataLoader(ds, batch_size=src.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, pad_id))
    per_type = collect(network, loader, device, src.pool, amp)

    temps = {}
    for t, (logits, labels) in per_type.items():
        T, _ = fit_temperature(logits, labels, t)
        temps[t] = T
    rows = report(per_type, temps)
    os.makedirs(src.out, exist_ok=True)
    out = os.path.join(src.out, "temperatures.json")
    with open(out, "w") as f:
        json.dump({"temperatures": {TASK_NAMES[t]: v for t, v in temps.items()}, "report": rows}, f, indent=2)
    print(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()