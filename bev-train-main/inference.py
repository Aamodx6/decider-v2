import argparse
import re
from collections import defaultdict

import torch
import torch.nn.functional as F

from dataset import TASK_NAMES, TASK_TYPES, BEVDataset, collate_fn, load_questions, question_to_choices
from logger import load_checkpoint
from tokenization import encode_example


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device):
    # bf16 keeps fp32's exponent range, so no loss scaling is needed (unlike fp16)
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ("cuda", "mps"))


def to_device(batch, device):
    # strings (domains) and lists pass through untouched
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def domain_slug(domain):
    return re.sub(r"[^a-z0-9]+", "_", str(domain).lower()).strip("_")


def top_label_ece(confidences, correct, n_bins=10):
    """Expected calibration error: 10 bins over top-label confidence vs accuracy."""
    if not confidences:
        return float("nan")
    conf, hit = torch.tensor(confidences), torch.tensor(correct, dtype=torch.float32)
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (conf >= edges[i]) & (conf <= edges[i + 1]) if i == 0 else (conf > edges[i]) & (conf <= edges[i + 1])
        if in_bin.any():
            ece += in_bin.float().mean() * abs(hit[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def apply_temperature(logits, task_type, temperatures):
    """Per-task-type temperature scaling (fit by calibrate.py, stored in calibration.json)."""
    if not temperatures:
        return logits
    table = torch.tensor([float(temperatures.get(TASK_NAMES[t], 1.0)) for t in range(len(TASK_TYPES))],
                         device=logits.device)
    return logits.float() / table[task_type].unsqueeze(1)


def binary_auc(scores, labels):
    """Probability that a random positive gets a higher score than a random negative (ties count half)."""
    scores, labels = torch.tensor(scores), torch.tensor(labels, dtype=torch.bool)
    positives, negatives = scores[labels], scores[~labels].sort().values
    below = torch.searchsorted(negatives, positives, right=False)
    below_or_tied = torch.searchsorted(negatives, positives, right=True)
    return (below + below_or_tied).sum().item() / 2 / (len(positives) * len(negatives))


@torch.no_grad()
def evaluate(network, dataloader, device, temperatures=None):
    """Accuracy per task type and per domain, noul AUC, and ECE (10 bins, top-label) overall,
    per type and per domain. temperatures (from calibration.json) are applied when given."""
    network.eval()
    total_loss, correct, count = 0.0, defaultdict(int), defaultdict(int)
    noul_p_yes, noul_labels = [], []
    conf, hit = [], []                      # overall top-label calibration
    by_type, by_domain = defaultdict(lambda: ([], [])), defaultdict(lambda: ([], []))

    for batch in dataloader:
        batch = to_device(batch, device)
        with autocast(device):
            logits = network(**batch)
        logits = apply_temperature(logits, batch["task_type"], temperatures)
        total_loss += F.cross_entropy(logits, batch["labels"], reduction="sum").item()

        probs = torch.softmax(logits.float(), dim=-1)
        confidence, prediction = probs.max(-1)
        hits = prediction == batch["labels"]
        conf += confidence.tolist()
        hit += hits.tolist()
        for task_type, domain, c, h in zip(batch["task_type"].tolist(), batch["domain"],
                                           confidence.tolist(), hits.tolist()):
            name = TASK_NAMES[task_type]
            count[name] += 1
            correct[name] += h
            by_type[name][0].append(c)
            by_type[name][1].append(h)
            by_domain[domain][0].append(c)
            by_domain[domain][1].append(h)

        is_noul = batch["task_type"] == TASK_TYPES["noul"]
        noul_p_yes += probs[is_noul][:, 1].tolist()
        noul_labels += batch["labels"][is_noul].tolist()

    n = sum(count.values())
    metrics = {"loss": total_loss / n, "accuracy": sum(correct.values()) / n,
               "ece": top_label_ece(conf, hit)}
    for name in count:
        metrics[f"accuracy_{name}"] = correct[name] / count[name]
        metrics[f"ece_{name}"] = top_label_ece(*by_type[name])
    for domain, (d_conf, d_hit) in by_domain.items():
        slug = domain_slug(domain)
        metrics[f"accuracy_domain_{slug}"] = sum(d_hit) / len(d_hit)
        metrics[f"ece_domain_{slug}"] = top_label_ece(d_conf, d_hit)
    if 0 < sum(noul_labels) < len(noul_labels):
        metrics["auc_noul"] = binary_auc(noul_p_yes, noul_labels)
    network.train()
    return metrics


@torch.no_grad()
def answer(network, meta, state, question, device):
    """Answers one question dict (dataset / TypeSafe format) with a typed answer."""
    network.eval()
    choices, _, _ = question_to_choices(question)
    example = encode_example(state, question["instructions"], choices, question["type"],
                             meta["max_state_tokens"], meta["max_choice_tokens"])
    example.update(task_type=TASK_TYPES[question["type"]], label=None)
    batch = to_device(collate_fn([example]), device)

    with autocast(device):
        logits = apply_temperature(network(**batch)[0].float().unsqueeze(0),
                                   batch["task_type"], meta.get("temperatures"))[0]
        probs = torch.softmax(logits, dim=-1).tolist()

    if question["type"] == "choice":
        keys = list(question["criteria"])
        return {"choice": keys[max(range(len(keys)), key=probs.__getitem__)], "probabilities": dict(zip(keys, probs))}
    if question["type"] == "score":
        return {"score": sum(i * p for i, p in enumerate(probs)), "probabilities": probs}
    return {"noul": probs[1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="runs/<exp_id>/checkpoints/<name>")
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", help="dataset config to evaluate: default | hard_50k | numeric_temporal | "
                                         "skills | counterfactual_15k | all "
                                         "(default: the configs recorded in the checkpoint)")
    parser.add_argument("--max_questions", type=int, default=500, help="0 = no cap (full split)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    network, meta = load_checkpoint(args.checkpoint, device)
    configs = [args.config] if args.config else meta.get("data_configs", ["default"])
    questions = load_questions(args.split, args.max_questions or None, configs=configs)
    # Same truncation as during training, read from the checkpoint
    dataset = BEVDataset(questions, meta["max_state_tokens"], meta["max_choice_tokens"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                         num_workers=args.num_workers)
    # meta["temperatures"] (written by calibrate.py) is applied so reported probs are calibrated
    print(evaluate(network, loader, device, meta.get("temperatures")))


if __name__ == "__main__":
    main()
