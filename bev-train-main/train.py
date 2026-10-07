import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import get_cosine_schedule_with_warmup

from dataset import TASK_TYPES, BEVDataset, collate_fn, load_questions
from inference import autocast, evaluate, get_device, to_device
from logger import RunLogger
from network import build_network


def make_loader(split, config, device, shuffle, order=None, dataset=None):
    """DataLoader for one split. `order` (a list of dataset indices) replaces shuffling: it is the
    deterministic per-epoch permutation, sliced past samples_seen when resuming (cheap: indices
    only, the skipped examples are never tokenized). `dataset` passes a pre-built BEVDataset so
    the questions are tokenized/loaded once, not once per epoch."""
    data = config["data"]
    if split == "train":
        max_questions = data["max_train_questions"]
    elif split in ("val", "dev"):
        # Validation is carved out of train (grouped by state); old dev/test caps fall back for it
        max_questions = data.get("max_val_questions",
                                 data.get("max_dev_questions", data.get("max_test_questions")))
    else:
        max_questions = data["max_test_questions"]
    if dataset is None:
        dataset = BEVDataset(
            load_questions(split, max_questions, config["seed"], data.get("configs")),
            data["max_state_tokens"], data["max_choice_tokens"],
        )
    loader_kwargs = dict(
        batch_size=data["batch_size"],
        shuffle=shuffle if order is None else False,
        collate_fn=collate_fn,
        num_workers=data["num_workers"],
        persistent_workers=data["num_workers"] > 0,  # keep workers alive across epochs/validation
        pin_memory=device.type == "cuda",
    )
    if order is not None:
        return torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, order), **loader_kwargs)
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


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


def validate(network, val_loader, device, logger, step, best_accuracy):
    metrics = evaluate(network, val_loader, device)
    logger.log(step, split="val", **metrics)
    if metrics["accuracy"] > best_accuracy:
        best_accuracy = metrics["accuracy"]
        logger.save_checkpoint(network, name="best", step=step)
    return metrics, best_accuracy


# --- deterministic data order ---------------------------------------------------------------
# One shape-stable permutation per epoch, from its own generator: restarts (and re-ordered
# micro-batches under an OOM fallback) can always reconstruct it from (seed, epoch), and
# resuming skips forward by slicing the index list -- skipped examples are never tokenized.
PER_EPOCH_PERMUTATION_STEP = 100003


def epoch_order(total, seed, epoch):
    generator = torch.Generator().manual_seed((seed * PER_EPOCH_PERMUTATION_STEP + epoch) % (2 ** 63))
    return torch.randperm(total, generator=generator).tolist()


def rng_state():
    state = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def collect_train_state(optimizer, scheduler, step, epoch, samples_seen, best_val,
                        micro, accum, total_questions, seed):
    """Everything needed to continue the SAME cosine schedule exactly (single schedule)."""
    return {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "samples_seen": samples_seen,
        "best_val": best_val,
        "micro": micro, "accum": accum, "effective_batch": micro * accum,
        "total_questions": total_questions, "seed": seed,
        "rng": rng_state(),
        "saved_at_step_wall": None,  # filled just before the save (readability of the save cadence)
    }


def train(config, run_name, resume=None, resume_state=None, emit_resume_every=None):
    seed = config["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    device = get_device()
    resume_dir, resume_meta = None, {}
    if resume:
        resume_dir = resolve_checkpoint(resume)
        resume_meta = json.loads((resume_dir / "meta.json").read_text())
        config = {**config, "resumed_from": str(resume_dir)}

    resumed_state, run_dir, start_step, start_best = None, None, 0, -1.0
    if resume_state:
        run_dir = Path(resume_state)
        resume_root = run_dir / "checkpoints" / "resume"
        pointer = resume_root / "pointer"
        if not pointer.exists():
            raise FileNotFoundError(f"--resume-state: no pointer file under {resume_root} (nothing was saved yet?)")
        slot = resume_root / pointer.read_text().strip()
        resumed_state = torch.load(slot / "train_state.pt", map_location="cpu", weights_only=False)
        resume_dir = slot  # LoRA + head weights come from the same slot
        config = {**config, "resumed_state_from": str(run_dir)}
        # --resume-state continues the SAME run dir and appends to the SAME metrics.jsonl
        logger = RunLogger(config, run_name, device, run_dir=run_dir)
        print(f"resuming state from {slot} (step {resumed_state['step']}, best_val {resumed_state['best_val']:.4f})")
    else:
        logger = RunLogger(config, run_name, device)
    model, lora, optim, log = config["model"], config["lora"], config["optim"], config["logging"]

    head_config = {**config["head"], "num_task_types": len(TASK_TYPES)}
    network = build_network(model["name"], model["num_layers"], lora["r"], lora["alpha"], lora["dropout"],
                            lora["last_k_layers"], head_config, lora.get("target", "attn"))
    if resume_dir:
        load_weights(network, resume_dir)
        where = f"weights {resume_dir}"
        if resumed_state:
            where += f" + optimizer/scheduler/RNG (step {resumed_state['step']})"
        else:
            where += f" (weights only; saved at step {resume_meta['step']})"
        print(f"Loaded {where}")
    network = network.to(device)
    count = lambda params: sum(p.numel() for p in params if p.requires_grad)
    print(f"trainable params: backbone={count(network.backbone.parameters()):,} head={count(network.head.parameters()):,}")

    data = config["data"]
    micro = int(data["batch_size"])
    accum = int(optim.get("grad_accum_steps", 1))
    effective_batch = micro * accum  # effective batch (constant under OOM fallback: micro/2 x accum*2)

    # loaded/exploded once, reused by every epoch loader (the questions are cached in RAM)
    epochs = int(optim["epochs"])
    train_dataset = BEVDataset(load_questions("train", data["max_train_questions"], seed,
                                             data.get("configs")),
                               data["max_state_tokens"], data["max_choice_tokens"])
    total_questions = len(train_dataset)
    micro_batches_per_epoch = total_questions // micro
    steps_per_epoch = micro_batches_per_epoch // accum  # incomplete final accumulation group is dropped
    total_steps = epochs * steps_per_epoch

    lora_params = [p for p in network.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": float(optim["lr_lora"])},
        {"params": network.head.parameters(), "lr": float(optim["lr_head"])},
    ], weight_decay=float(optim["weight_decay"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(optim["warmup_ratio"] * total_steps), total_steps)

    step = 0
    start_epoch = 0
    skip_micro = 0
    if resumed_state is not None:
        optimizer.load_state_dict(resumed_state["optimizer"])
        scheduler.load_state_dict(resumed_state["scheduler"])
        step = int(resumed_state["step"])
        start_best = float(resumed_state["best_val"])
        start_epoch = step // steps_per_epoch
        skip_micro = (step % steps_per_epoch) * accum  # micro-batches already consumed in this epoch
        if resumed_state.get("total_questions") != total_questions:
            raise ValueError(f"train set changed since the save: {resumed_state.get('total_questions')} -> {total_questions}")
        if resumed_state.get("effective_batch") != effective_batch:
            print(f"note: effective batch changed {resumed_state.get('effective_batch')} -> {effective_batch} "
                  f"(micro {resumed_state.get('micro')}->{micro}); positions still align by optimizer step")
        set_rng_state(resumed_state["rng"])
        torch.cuda.empty_cache()

    val_cap = data.get("max_val_questions",
                       data.get("max_dev_questions", data.get("max_test_questions")))
    val_loader = None
    if val_cap != 0:
        # Best checkpoint and all in-training evals run on val (carved from train, state-grouped);
        # the test split is only read by inference.py for final reporting, never during training.
        val_loader = make_loader("val", config, device, shuffle=False)

    if val_loader is not None and (step == 0 or resumed_state is not None):
        _, start_best = validate(network, val_loader, device, logger, step, start_best)
    best_accuracy = start_best
    network.train()

    # crash-safe rolling save cadence: resume_save_seconds wins, resume_save_minutes is the config unit
    save_interval_sec = float(optim.get("resume_save_seconds",
                                        optim.get("resume_save_minutes", 30) * 60.0))
    last_save = time.time()  # don't save immediately at start; the clock starts with the run
    epoch_loss = epoch_correct = epoch_count = 0
    val_metrics = {}
    global_micro = start_epoch * micro_batches_per_epoch + skip_micro  # micro-batches consumed overall

    for epoch in range(start_epoch, epochs):
        order = epoch_order(total_questions, seed, epoch)
        if epoch == start_epoch:
            # resume middle-of-epoch: skip the already-consumed head of this epoch's permutation
            order = order[skip_micro * micro:]
            skip_micro = 0
        epoch_loader = make_loader("train", config, device, shuffle=False, order=order, dataset=train_dataset)
        epoch_loss, epoch_correct, epoch_count = 0.0, 0, 0
        optimizer.zero_grad()
        micro_in_epoch = 0
        for batch in epoch_loader:
            if micro_in_epoch >= steps_per_epoch * accum:
                break  # drop the incomplete final accumulation group (deterministic)
            batch = to_device(batch, device)
            with autocast(device):
                logits = network(**batch)
            loss = compute_loss(logits, batch, config.get("loss", {}))
            (loss / accum).backward()

            correct = (logits.argmax(-1) == batch["labels"]).sum().item()
            epoch_loss += loss.item() * len(logits)
            epoch_correct += correct
            epoch_count += len(logits)
            micro_in_epoch += 1
            global_micro += 1
            if micro_in_epoch % accum:
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
                logger.log(step, split="train", epoch=epoch, loss=loss.item(),
                           accuracy=correct / len(logits), **lrs)
            if step % log["save_every"] == 0:
                logger.save_checkpoint(network, name="latest", step=step)
            if val_loader is not None and step % log["eval_every"] == 0:
                val_metrics, best_accuracy = validate(network, val_loader, device, logger, step, best_accuracy)
            # test hook (--emit-resume-every N) saves a slot exactly at step multiples of N;
            # normal runs save when the time cadence fires
            due = (emit_resume_every is not None and step % emit_resume_every == 0) or \
                time.time() - last_save >= save_interval_sec
            if due:
                # crash-safe rolling save into checkpoints/resume (last 2 kept, single cosine schedule)
                state = collect_train_state(optimizer, scheduler, step, epoch, global_micro * micro,
                                            best_accuracy, micro, accum, total_questions, seed)
                logger.save_resume_state(network, state)
                last_save = time.time()
        logger.log(step, split="train_epoch", epoch=epoch, loss=epoch_loss / epoch_count,
                   accuracy=epoch_correct / epoch_count)
        # a crash right at an epoch boundary resumes at the next epoch's first micro-batch
        if micro_in_epoch >= steps_per_epoch * accum and epoch + 1 < epochs:
            state = collect_train_state(optimizer, scheduler, step, epoch + 1, global_micro * micro,
                                        best_accuracy, micro, accum, total_questions, seed)
            logger.save_resume_state(network, state)
            last_save = time.time()

    if val_loader is not None:
        if step % log["eval_every"] != 0:
            val_metrics, best_accuracy = validate(network, val_loader, device, logger, step, best_accuracy)
    elif log.get("eval_every"):
        logger.log(step, split="val", accuracy=math.nan, note="val disabled (max_val_questions=0)")

    summary = {"final_train_loss": epoch_loss / epoch_count, "final_train_accuracy": epoch_correct / epoch_count,
               "total_steps": step}
    if val_loader is not None:
        summary["val"] = val_metrics
        summary["best_val_accuracy"] = best_accuracy
    ckpt_dir = logger.save_checkpoint(network, step=step)
    logger.finish(checkpoint=str(ckpt_dir), **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config, e.g. configs/smoke.yaml")
    parser.add_argument("--name", help="run name, used as runs/<name>-<timestamp> (default: config file name)")
    parser.add_argument("--resume", help="start from a checkpoint's LoRA + head weights: an experiment id "
                                         "(uses its final checkpoint) or a checkpoint folder. Optimizer and LR schedule start fresh.")
    parser.add_argument("--resume-state", help="continue the SAME run dir after a crash: a runs/<id> folder with "
                                               "checkpoints/resume/ slot. Restores optimizer, scheduler, RNG and step "
                                               "and appends to its metrics.jsonl (single cosine schedule).")
    parser.add_argument("--emit-resume-every", type=int, default=None,
                        help="test hook: write a resume slot every N optimizer steps (instead of the time cadence)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config, args.name or Path(args.config).stem, args.resume, args.resume_state,
          emit_resume_every=args.emit_resume_every)


if __name__ == "__main__":
    main()
