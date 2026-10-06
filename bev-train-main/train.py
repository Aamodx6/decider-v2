import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import get_cosine_schedule_with_warmup

from dataset import DEV_FRACTION, TASK_TYPES, BEVDataset, collate_fn, load_questions
from inference import autocast, evaluate, get_device, to_device
from logger import RunLogger
from network import build_network


def make_loader(split, config, device, shuffle):
    data = config["data"]
    if split == "train":
        max_questions = data["max_train_questions"]
    elif split == "dev":
        # dev is carved from train (grouped by state); old configs fall back to their test cap
        max_questions = data.get("max_dev_questions", data.get("max_test_questions"))
    else:
        max_questions = data["max_test_questions"]
    dataset = BEVDataset(
        load_questions(split, max_questions, config["seed"], data.get("configs"),
                       data.get("dev_fraction", DEV_FRACTION)),
        data["max_state_tokens"], data["max_choice_tokens"],
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=data["num_workers"],
        persistent_workers=data["num_workers"] > 0,  # keep workers alive across epochs and validation passes
        pin_memory=device.type == "cuda",
    )


def compute_loss(logits, batch, loss_config):
    """All flags off = plain mean cross-entropy (the baseline path).

    loss.label_smoothing: smoothed CE on hard labels. Padded options have -inf logits, which make
    F.cross_entropy's smoothing term +inf, so the uniform term is computed over valid options only.
    loss.emd_weight: adds lambda * CDF-based EMD to score rows (lambda ~ 0.5).
    loss.use_soft_labels: rows with label_probs (skills config) use soft CE against it instead.
    loss.balance_types: average within each task type first, then across types, so rare types are not drowned out.
    """
    labels, task_type = batch["labels"], batch["task_type"]
    label_smoothing = float(loss_config.get("label_smoothing", 0.0))
    emd_weight = float(loss_config.get("emd_weight", 0.0))
    use_soft_labels = bool(loss_config.get("use_soft_labels", False))
    balance_types = bool(loss_config.get("balance_types", False))

    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(1, labels[:, None]).squeeze(1)
    if label_smoothing:
        uniform_nll = -log_probs.masked_fill(~torch.isfinite(log_probs), 0.0).mean(-1)
        per_row = (1.0 - label_smoothing) * nll + label_smoothing * uniform_nll
    else:
        per_row = nll

    if use_soft_labels:
        soft_mask = batch["label_probs"].sum(-1) > 0
        if soft_mask.any():
            # 0 * -inf would be NaN at padded options, whose soft probability is 0 anyway
            soft_nll = -(batch["label_probs"].float() * log_probs.masked_fill(~torch.isfinite(log_probs), 0.0)).sum(-1)
            per_row = torch.where(soft_mask, soft_nll, per_row)

    if emd_weight:
        score_rows = task_type == TASK_TYPES["score"]
        if score_rows.any():
            K = logits.shape[-1]
            cdf_target = (torch.arange(K, device=logits.device)[None] >= labels[:, None]).float()
            emd = (torch.softmax(logits.float(), dim=-1).cumsum(-1) - cdf_target).abs().mean(-1)
            per_row = per_row + emd_weight * torch.where(score_rows, emd, torch.zeros_like(emd))

    if balance_types:
        return torch.stack([per_row[task_type == t].mean() for t in task_type.unique()]).mean()
    return per_row.mean()


def resolve_checkpoint(resume):
    """--resume takes a checkpoint folder, or an experiment id (runs/<id>/checkpoints/final)."""
    path = Path(resume)
    if not (path / "meta.json").exists():
        path = Path("runs") / resume / "checkpoints" / "final"
    if not (path / "meta.json").exists():
        raise FileNotFoundError(f"no checkpoint at {resume} or {path}")
    return path


def load_weights(network, ckpt_dir):
    """Load LoRA + head weights into a trainable network with the same architecture."""
    has_lora = (ckpt_dir / "lora").exists()
    if has_lora != isinstance(network.backbone, PeftModel):
        raise ValueError(f"{ckpt_dir} {'has' if has_lora else 'has no'} LoRA weights, but the config's lora.last_k_layers disagrees")
    if has_lora:
        weights = load_file(ckpt_dir / "lora" / "adapter_model.safetensors")
        result = set_peft_model_state_dict(network.backbone, weights)
        missing = [k for k in result.missing_keys if "lora_" in k]
        if result.unexpected_keys or missing:
            raise ValueError(f"LoRA weights don't match the config: unexpected={result.unexpected_keys[:3]} missing={missing[:3]}")
    network.head.load_state_dict(torch.load(ckpt_dir / "head.pt", map_location="cpu"))


def validate(network, dev_loader, device, logger, step, best_accuracy):
    metrics = evaluate(network, dev_loader, device)
    logger.log(step, split="dev", **metrics)
    if metrics["accuracy"] > best_accuracy:
        best_accuracy = metrics["accuracy"]
        logger.save_checkpoint(network, name="best", step=step)
    return metrics, best_accuracy


def train(config, run_name, resume=None):
    torch.manual_seed(config["seed"])
    device = get_device()
    resume_dir, resume_meta = None, {}
    if resume:
        resume_dir = resolve_checkpoint(resume)
        resume_meta = json.loads((resume_dir / "meta.json").read_text())
        config = {**config, "resumed_from": str(resume_dir)}
    logger = RunLogger(config, run_name, device)
    model, lora, optim, log = config["model"], config["lora"], config["optim"], config["logging"]

    head_config = {**config["head"], "num_task_types": len(TASK_TYPES)}
    network = build_network(model["name"], model["num_layers"], lora["r"], lora["alpha"], lora["dropout"],
                            lora["last_k_layers"], head_config, lora.get("target", "attn"))
    if resume_dir:
        load_weights(network, resume_dir)
        print(f"Loaded LoRA + head weights from {resume_dir} (step {resume_meta['step']})")
    network = network.to(device)
    count = lambda params: sum(p.numel() for p in params if p.requires_grad)
    print(f"trainable params: backbone={count(network.backbone.parameters()):,} head={count(network.head.parameters()):,}")

    train_loader = make_loader("train", config, device, shuffle=True)
    # Best checkpoint is selected on dev (carved from train, grouped by state); the test split is only
    # read by inference.py for final reporting, never during training (that was a leak)
    dev_cap = config["data"].get("max_dev_questions", config["data"].get("max_test_questions"))
    dev_loader = None
    if dev_cap != 0:
        dev_loader = make_loader("dev", config, device, shuffle=False)

    lora_params = [p for p in network.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": float(optim["lr_lora"])},
        {"params": network.head.parameters(), "lr": float(optim["lr_head"])},
    ], weight_decay=float(optim["weight_decay"]))
    # grad_accum_steps > 1 splits each optimizer step over that many loader batches (batch_size is per micro-batch)
    accum = optim.get("grad_accum_steps", 1)
    total_steps = optim["epochs"] * (len(train_loader) // accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(optim["warmup_ratio"] * total_steps), total_steps)

    step = 0
    best_accuracy = -1.0
    if resume_dir and dev_loader is not None:
        _, best_accuracy = validate(network, dev_loader, device, logger, step, best_accuracy)
    network.train()
    for epoch in range(optim["epochs"]):
        epoch_loss, epoch_correct, epoch_count = 0.0, 0, 0
        optimizer.zero_grad()
        for micro, batch in enumerate(train_loader, 1):
            if micro > (len(train_loader) // accum) * accum:
                break  # drop the incomplete final accumulation group
            batch = to_device(batch, device)
            with autocast(device):
                logits = network(**batch)
            loss = compute_loss(logits, batch, config.get("loss", {}))
            (loss / accum).backward()

            correct = (logits.argmax(-1) == batch["labels"]).sum().item()
            epoch_loss += loss.item() * len(logits)
            epoch_correct += correct
            epoch_count += len(logits)
            if micro % accum:
                continue

            torch.nn.utils.clip_grad_norm_(network.parameters(), optim["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % log["log_every"] == 0:
                lrs = {"lr_head": scheduler.get_last_lr()[1]}
                if lora_params:
                    lrs["lr_lora"] = scheduler.get_last_lr()[0]
                logger.log(step, split="train", epoch=epoch, loss=loss.item(), accuracy=correct / len(logits), **lrs)
            if step % log["save_every"] == 0:
                logger.save_checkpoint(network, name="latest", step=step)
            if dev_loader is not None and step % log["eval_every"] == 0:
                dev_metrics, best_accuracy = validate(network, dev_loader, device, logger, step, best_accuracy)

        logger.log(step, split="train_epoch", epoch=epoch, loss=epoch_loss / epoch_count,
                   accuracy=epoch_correct / epoch_count)

    summary = {"final_train_loss": epoch_loss / epoch_count, "final_train_accuracy": epoch_correct / epoch_count}
    if dev_loader is not None:
        if step % log["eval_every"] != 0:
            dev_metrics, best_accuracy = validate(network, dev_loader, device, logger, step, best_accuracy)
        summary["dev"] = dev_metrics
        summary["best_dev_accuracy"] = best_accuracy

    ckpt_dir = logger.save_checkpoint(network, step=step)
    logger.finish(checkpoint=str(ckpt_dir), **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config, e.g. configs/smoke.yaml")
    parser.add_argument("--name", help="run name, used as runs/<name>-<timestamp> (default: config file name)")
    parser.add_argument("--resume", help="start from a checkpoint's LoRA + head weights: an experiment id "
                                         "(uses its final checkpoint) or a checkpoint folder. Optimizer and LR schedule start fresh.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config, args.name or Path(args.config).stem, args.resume)


if __name__ == "__main__":
    main()
