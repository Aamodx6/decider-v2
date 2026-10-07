"""Per-task-type temperature scaling.

Fits one temperature per task type by minimizing NLL on dev logits (torch LBFGS), computes
top-label ECE (10 bins) overall, per type and per domain, and saves calibration.json inside the
checkpoint folder. load_checkpoint() picks it up, so answer() and inference.py report calibrated
probabilities automatically afterwards.

    uv run python calibrate.py runs/<id>/checkpoints/best --config configs/<name>.yaml
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from dataset import TASK_NAMES
from inference import apply_temperature, autocast, get_device, to_device, top_label_ece
from logger import load_checkpoint
from train import make_loader


def fit_temperature(logits, labels, max_iter=50):
    """Minimize cross_entropy(logits / exp(log_t)) over log_t with LBFGS (no new dependencies)."""
    # Padded options carry -inf logits; without this, grad through logits / T hits 0 * -inf = NaN
    logits = logits.masked_fill(~torch.isfinite(logits), -1e4)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp()[0], labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.exp().detach())
    return temperature if math.isfinite(temperature) and temperature > 0 else 1.0


@torch.no_grad()
def collect(network, loader, device):
    network.eval()
    logits, labels, task_types, domains = [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device):
            logits.append(network(**batch).float().cpu())
        labels.append(batch["labels"].cpu())
        task_types.append(batch["task_type"].cpu())
        domains += batch["domain"]
    # option counts differ per batch -> pad logits so they concatenate (-inf = never a valid option)
    max_choices = max(x.shape[1] for x in logits)
    logits = [F.pad(x, (0, max_choices - x.shape[1]), value=float("-inf")) for x in logits]
    return torch.cat(logits), torch.cat(labels), torch.cat(task_types), domains


def calibrate(network, loader, device, ckpt_dir):
    logits, labels, task_types, domains = collect(network, loader, device)
    if not len(labels):
        raise ValueError("empty dev split: nothing to fit temperatures on")

    temperatures = {name: fit_temperature(logits[task_types == t], labels[task_types == t])
                    if (task_types == t).any() else 1.0
                    for t, name in sorted(TASK_NAMES.items())}
    scaled = apply_temperature(logits, task_types, temperatures)
    confidence, prediction = torch.softmax(scaled, dim=-1).max(-1)
    conf, correct = confidence.tolist(), (prediction == labels).tolist()

    def group_ece(indices):
        return top_label_ece([conf[i] for i in indices], [correct[i] for i in indices])

    by_domain = defaultdict(list)
    for i, domain in enumerate(domains):
        by_domain[domain].append(i)
    calibration = {
        "temperatures": temperatures,
        "ece": {
            "overall": top_label_ece(conf, correct),
            "per_type": {name: group_ece((task_types == t).nonzero().squeeze(-1).tolist())
                         for t, name in sorted(TASK_NAMES.items())},
            "per_domain": {domain: group_ece(indices) for domain, indices in sorted(by_domain.items())},
        },
        "n_dev": len(labels),
        "nll_before": float(F.cross_entropy(logits, labels)),
        "nll_after": float(F.cross_entropy(scaled, labels)),
    }
    path = Path(ckpt_dir) / "calibration.json"
    path.write_text(json.dumps(calibration, indent=2))
    print(f"Saved calibration -> {path}")
    return calibration


def main():
    parser = argparse.ArgumentParser(description="fit per-task-type temperatures on the dev split")
    parser.add_argument("checkpoint", help="runs/<exp_id>/checkpoints/<name>")
    parser.add_argument("--config", required=True, help="YAML config the checkpoint was trained with (provides the dev split)")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="cap dev questions (default: the config's max_dev_questions)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.max_questions is not None:
        config["data"]["max_dev_questions"] = args.max_questions
    device = get_device()
    network, _ = load_checkpoint(args.checkpoint, device)
    print(json.dumps(calibrate(network, make_loader("dev", config, device, shuffle=False), device, args.checkpoint),
                     indent=2))


if __name__ == "__main__":
    main()
