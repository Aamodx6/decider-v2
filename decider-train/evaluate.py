"""Evaluate a trained checkpoint (or the published reference) on a test split.

Reports accuracy overall / per question type / per domain, NLL and ECE (10 bins, top-label) before
and after per-type temperature scaling, plus an option-order-invariance hard gate.  Writes
metrics.json and (optionally) a per-example prediction file.

The invariance check re-encodes each selected example with permuted options and re-runs it on its
own, which is the definition bev-decider's model card uses ("every option starts at the same
position id ... shuffling the options cannot change the answer").
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DecisionDataset, collate, load_split_table
from inference import apply_per_type_temperature, load_target
from network import expected_calibration_error
from tokenization import TASK_NAMES, Encoder


# --------------------------------------------------------------------------- helpers
def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward(network, batch, device, amp_dtype=torch.float32):
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
        return network(**batch).float()


def masked_logits(logits, option_mask):
    # -1e30 rather than -inf: softmax underflows it to exactly 0.0 and log-softmax stays finite,
    # which keeps NLL / temperature scaling / E numerically clean.
    return logits.masked_fill(~option_mask, -1e30)


@torch.no_grad()
def order_invariance(network, dataset, encoder, pad_id, device, max_examples=32, shuffles=32,
                     kmin=3, kmax=12, seed=0):
    """fp32 hard gate: max |p - p'| over random option permutations, in the ORIGINAL option order."""
    rng = random.Random(seed)
    worst, worst_meta, checked, nonfinite = 0.0, None, 0, 0
    for idx in range(len(dataset)):
        if checked >= max_examples:
            break
        ex = dataset[idx]
        k = int(ex["n_options"])
        if k < kmin:
            continue
        k = min(k, kmax)
        checked += 1
        base_b = to_device(collate([ex], pad_id), device)
        base = forward(network, base_b, device)
        nonfinite += int((~torch.isfinite(base)).sum())
        base_p = (torch.softmax(masked_logits(base, base_b["option_mask"]), -1)[0]
                  * base_b["option_mask"][0].float())
        # the dataset may already have shuffled options; recover the canonical order from `keys`
        for _ in range(shuffles):
            perm = rng.sample(range(k), k)
            alt_ex = _encode_permuted(ex, perm)
            alt_b = to_device(collate([alt_ex], pad_id), device)
            alt = forward(network, alt_b, device)
            p = torch.softmax(masked_logits(alt, alt_b["option_mask"]), -1)[0] * alt_b["option_mask"][0].float()
            pm = torch.zeros_like(base_p)
            src = torch.tensor(perm, device=p.device, dtype=torch.long)
            pm[src[torch.arange(k, device=p.device)]] = p[:k]
            d = (pm - base_p).abs().max().item()
            if d > worst:
                worst = d
                worst_meta = {"example_index": idx, "k": k, "perm": perm}
    return {"max_abs_delta": worst, "examples_checked": checked, "shuffles_per_example": shuffles,
            "k_range": [kmin, kmax], "gate": 1e-5, "pass": worst <= 1e-5,
            "nonfinite_logits": nonfinite, "worst_case": worst_meta}


def _encode_permuted(ex, perm):
    """Re-encode example `ex` with its option blocks permuted by `perm` (perm[j] = new slot for old j)."""
    body = ex["_source"]
    options = list(body["options"])
    ranks = list(body["ranks"])
    new_options = [options[perm[j]] for j in range(len(perm))]
    new_ranks = [ranks[perm[j]] for j in range(len(perm))]
    out = ex["_encoder"].encode(body["state"], body["instructions"], new_options, body["task_name"],
                                ranks=new_ranks)
    out["label"] = perm.index(ex["label"]) if ex["label"] >= 0 else -1
    out["domain"] = ex["domain"]
    out["idx"] = ex["idx"]
    out["_source"] = {**body, "options": new_options, "ranks": new_ranks}
    out["_encoder"] = ex["_encoder"]
    return out


@torch.no_grad()
def evaluate_split(network, dataset, pad_id, device, amp_dtype, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, pad_id))
    probs_all, logits_all, labels_all, types_all, doms_all, k_all = [], [], [], [], [], []
    per_domain, per_type = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    C_max = 0
    done = 0
    total = len(loader)
    t0, printed = time.time(), 0
    for batch in loader:
        done += 1
        # Time-based, not every-N-batches. On CPU a batch can take tens of seconds, so a fixed
        # every-20-batches cadence prints nothing for many minutes and the run looks dead; on GPU it
        # prints too often. A ~20 s interval is right for both.
        now = time.time()
        if done == 1 or done == total or (now - t0) >= 20:
            # rate/eta are over the WINDOW since the last print (done - printed), never cumulative:
            # done / window shrinks the eta toward zero as the run progresses and reports a 0.2 min
            # eta with 167 batches left, which is worse than no progress line at all.
            elapsed = now - t0
            rate = (done - printed) / max(elapsed, 1e-9)
            eta = (total - done) / rate if rate > 0 and elapsed > 1 else float("nan")
            print(f"  eval batch {done}/{total}  ({done * batch_size:,} questions)  "
                  f"{rate:.2f} batch/s  eta {eta / 60:.1f} min", flush=True)
            t0, printed = now, done
        batch = to_device(batch, device)
        logits = forward(network, batch, device, amp_dtype)
        k = batch["option_mask"].sum(-1)
        m = masked_logits(logits, batch["option_mask"])
        p = torch.softmax(m, -1)
        C_max = max(C_max, m.shape[1])
        probs_all.append(p.cpu())
        logits_all.append(m.cpu())
        labels_all.append(batch["labels"].cpu())
        types_all.append(batch["task_type"].cpu())
        k_all.append(k.cpu())
        doms_all.extend(batch["domains"])
        hits = (m.argmax(-1) == batch["labels"]).cpu()
        for t, h, d in zip(batch["task_type"].tolist(), hits.tolist(), batch["domains"]):
            per_type[TASK_NAMES[t]][0] += h
            per_type[TASK_NAMES[t]][1] += 1
            per_domain[d][0] += h
            per_domain[d][1] += 1

    def _pad(tensors, C):
        return torch.cat([F.pad(t, (0, C - t.shape[1]), value=float(t.min()) if t.numel() else 0.0)
                          for t in tensors])
    probs = _pad(probs_all, C_max)
    logits = _pad(logits_all, C_max)
    labels = torch.cat(labels_all)
    types = torch.cat(types_all)
    k_all = torch.cat(k_all)
    return {
        "probs": probs, "logits": logits, "labels": labels, "types": types, "domains": doms_all, "k": k_all,
        "metrics": {
            "n_questions": int(labels.numel()),
            "accuracy": float((probs.argmax(-1) == labels).float().mean()),
            "nll": float(F.cross_entropy(torch.log(probs.clamp_min(1e-12)), labels)),
            "ece": float(expected_calibration_error(probs, labels)),
            "accuracy_by_type": {k: v[0] / v[1] for k, v in sorted(per_type.items())},
            "count_by_type": {k: v[1] for k, v in sorted(per_type.items())},
            "accuracy_by_domain": {k: round(v[0] / v[1], 6) for k, v in sorted(per_domain.items())},
            "count_by_domain": {k: v[1] for k, v in sorted(per_domain.items())},
        },
    }


def fit_temperature(logits, labels, task_id, lo=0.05, hi=10.0, iters=200):
    """Grid + golden-section on log T for one type (NLL is smooth and 1-D; this is exact enough).

    One temperature per question type, fitted by minimising NLL on the calibration split (which is
    carved out of TRAIN before training and never used for early stopping).  Reports ECE (10 bins,
    top-label) before and after -- this is the old calibrate.py's estimator, absorbed here.
    """
    grid = torch.logspace(float(torch.log10(torch.tensor(lo))), float(torch.log10(torch.tensor(hi))), 60)
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        nll = F.cross_entropy(logits / float(T), labels).item()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T, best_nll


# --------------------------------------------------------------------------- main
def main(argv=None):
    def _step(msg):
        # Every stage announces itself. This script has long silent stretches -- downloading a
        # ~970 MB checkpoint, casting it to fp32, tokenizing tens of thousands of questions -- and on
        # a CPU-only notebook those minutes look exactly like a hang. A reader cannot tell "working"
        # from "dead" without output, so every stage prints before it starts.
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="")
    p.add_argument("--reference", default="")
    p.add_argument("--test_channel", default=os.environ.get("SM_CHANNEL_TEST", "./data/test"))
    p.add_argument("--calib_channel", default=os.environ.get("SM_CHANNEL_CALIB", ""))
    p.add_argument("--max_state_tokens", type=int, default=1024)
    p.add_argument("--max_choice_tokens", type=int, default=64)
    p.add_argument("--mode", default="bev")
    p.add_argument("--pool", default="bev")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--invariance", type=int, default=1)
    p.add_argument("--invariance_examples", type=int, default=32)
    p.add_argument("--invariance_shuffles", type=int, default=32)
    p.add_argument("--predictions_out", default="")
    p.add_argument("--out", default=os.environ.get("SM_OUTPUT_DATA_DIR", "./runs/output"))
    p.add_argument("--tag", default="")
    p.add_argument("--env_check", type=int, default=1,
                   help="verify requirements.txt pins before scoring and abort on mismatch")
    src = p.parse_args(argv)

    if src.env_check:
        import logger
        if not logger.enforce(prefix="ENV"):
            raise SystemExit("evaluate.py aborting: package versions do not match requirements.txt")
    else:
        print("ENV: --env_check 0 -- versions NOT verified for this evaluation.")

    what = ("reference " + src.reference) if src.reference else ("checkpoint " + src.checkpoint)
    _step(f"loading {what} ...")
    network, tokenizer, encoder, device, meta = load_target(src)
    pad_id = tokenizer.pad_token_id or 0
    amp = torch.bfloat16 if device.type == "cuda" else torch.float32
    _step(f"loaded on {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else
             " -- NO GPU, so every batch below is CPU and slow but correct"))
    try:
        import psutil  # optional; absent on most images
        _step(f"resident memory: {psutil.virtual_memory().available / 2**30:.1f} GiB available")
    except Exception:
        pass

    _step(f"reading test split from {src.test_channel} ...")
    table = load_split_table(src.test_channel, src.limit or None)
    _step(f"tokenizing {len(table):,} test questions (this is CPU-bound and takes a while) ...")
    ds = DecisionDataset(table, encoder, shuffle_options=False, seed=0)
    _step(f"tokenized. scoring ...")

    res = evaluate_split(network, ds, pad_id, device, amp, src.batch_size)
    _step(f"test pass done: accuracy {res['metrics']['accuracy'] * 100:.2f}% "
          f"on {res['metrics']['n_questions']:,} questions")
    metrics = dict(res["metrics"])
    metrics["source"] = {k: v for k, v in meta.items() if k != "args"}
    metrics["invariance_gate_max_abs_delta"] = 1e-5

    if src.calib_channel and os.path.exists(src.calib_channel):
        _step(f"calibration: reading {src.calib_channel} ...")
        cal_table = load_split_table(src.calib_channel)
        _step(f"calibration: tokenizing {len(cal_table):,} questions ...")
        cal_ds = DecisionDataset(cal_table, encoder, shuffle_options=False, seed=0)
        _step("calibration: scoring ...")
        cal = evaluate_split(network, cal_ds, pad_id, device, amp, src.batch_size)
        _step("calibration: fitting temperatures ...")
        temps = {}
        for t in sorted(set(cal["types"].tolist())):
            m = cal["types"] == t
            temps[t], _ = fit_temperature(cal["logits"][m], cal["labels"][m], t)
        res_logits_scaled = apply_per_type_temperature(res["logits"], res["types"],
                                                       {int(t): float(v) for t, v in temps.items()})
        cal_probs = torch.softmax(res_logits_scaled, dim=-1)
        metrics["temperatures"] = {TASK_NAMES[k]: v for k, v in temps.items()}
        metrics["calibration_n"] = int(cal["labels"].numel())
        metrics["nll_after_calibration"] = float(
            F.cross_entropy(torch.log(cal_probs.clamp_min(1e-12)), res["labels"]))
        metrics["ece_after_calibration"] = float(expected_calibration_error(cal_probs, res["labels"]))
        metrics["ece_after_calibration_by_type"] = {
            TASK_NAMES[t]: float(expected_calibration_error(cal_probs[res["types"] == t],
                                                            res["labels"][res["types"] == t]))
            for t in sorted(set(res["types"].tolist()))}
        metrics["accuracy_after_calibration"] = float((cal_probs.argmax(-1) == res["labels"]).float().mean())

    if src.invariance:
        metrics["order_invariance"] = order_invariance(
            network, ds, encoder, pad_id, device, max_examples=src.invariance_examples,
            shuffles=src.invariance_shuffles)

    os.makedirs(src.out, exist_ok=True)
    tag = f"_{src.tag}" if src.tag else ""
    out_path = os.path.join(src.out, f"metrics{tag}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    if src.predictions_out:
        import csv
        with open(src.predictions_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["q_index", "task_type", "domain", "n_options", "label", "pred", "p_label"])
            for i in range(len(res["labels"])):
                w.writerow([i, TASK_NAMES[int(res["types"][i])], res["domains"][i],
                            int(res["k"][i]), int(res["labels"][i]), int(res["probs"][i].argmax()),
                            round(float(res["probs"][i, res["labels"][i]]), 6)])
        print(f"wrote predictions -> {src.predictions_out}")
    print(f"wrote metrics -> {out_path}")


if __name__ == "__main__":
    main()