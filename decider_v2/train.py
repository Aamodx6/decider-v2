"""One training script.  Runs identically on a laptop CPU (tiny config) and on SageMaker.

SageMaker surface
  channels    SM_CHANNEL_TRAIN / _VAL / _CALIB / _TEST  (directories of .parquet)
  outputs     SM_MODEL_DIR (checkpoints)  /  SM_OUTPUT_DATA_DIR (final model + metrics.jsonl)
  hyperparams passed as CLI flags by the estimator (no config file needed)

Logging line format (regex-friendly for SageMaker metric_definitions):
    step=1000 loss=0.8123 grad_norm=0.41 lr_lora=1.9e-04 lr_head=9.5e-04 acc=0.71 sps=42.1
"""

import argparse
import json
import math
import os
import random
import time
from functools import partial
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from data import (DecisionDataset, LengthBucketBatchSampler, collate, load_split_table,
                  state_token_lengths)
from losses import compute_loss
from model import build_model, count_params, load_checkpoint, save_checkpoint
from serialize import (POOL_BEV, POOL_MEAN, POOL_OPT_END, TASK_NAMES, Encoder, TASK_TYPES,
                       pad_ranks)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train_channel", default=os.environ.get("SM_CHANNEL_TRAIN", "./data/train"))
    p.add_argument("--val_channel", default=os.environ.get("SM_CHANNEL_VAL", "./data/val"))
    p.add_argument("--test_channel", default=os.environ.get("SM_CHANNEL_TEST", "./data/test"))
    p.add_argument("--max_state_tokens", type=int, default=1024)
    p.add_argument("--max_choice_tokens", type=int, default=64)
    p.add_argument("--mode", default="bev", choices=["bev", "special"])
    p.add_argument("--pool", default=POOL_BEV, choices=[POOL_BEV, POOL_OPT_END, POOL_MEAN])
    p.add_argument("--noul_use_criteria", action="store_true")
    p.add_argument("--shuffle_options", type=int, default=1)
    # model
    p.add_argument("--backbone_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--num_layers", type=int, default=20)
    p.add_argument("--random_init", type=int, default=0)
    p.add_argument("--vocab_size", type=int, default=0,
                   help="only for --random_init 1: embedding rows (default = tokenizer vocab)")
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_last_k_layers", type=int, default=-1,
                   help="-1 means all kept layers; 0 = frozen backbone")
    p.add_argument("--lora_include_mlp", type=int, default=1)
    p.add_argument("--head", default="v2", choices=["v1", "v2"])
    p.add_argument("--head_new_dim", type=int, default=512)
    p.add_argument("--head_layers", type=int, default=2)
    p.add_argument("--head_heads", type=int, default=8)
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--n_ctx", type=int, default=8)
    p.add_argument("--use_ctx", type=int, default=1)
    p.add_argument("--use_rank", type=int, default=1)
    p.add_argument("--use_null", type=int, default=0)
    p.add_argument("--use_derived", type=int, default=0)
    p.add_argument("--derived_dropout", type=float, default=0.15)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    # loss
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--emd_lambda", type=float, default=0.5)
    p.add_argument("--type_weights", default="", help="e.g. '0:1.0,1:1.0,2:1.0'")
    # optim
    p.add_argument("--batch_size", type=int, default=32, help="micro-batch")
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr_lora", type=float, default=2e-4)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    # logging / io
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--limit_train", type=int, default=0)
    p.add_argument("--limit_val", type=int, default=0)
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "./runs/model"))
    p.add_argument("--output_dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "./runs/output"))
    p.add_argument("--resume", default="")
    p.add_argument("--max_train_minutes", type=float, default=-1)
    # --- preflight gates. Both default ON: a job that would produce numbers from the wrong library
    # versions, or from a broken mask, should fail in the first minute rather than after 12k steps.
    p.add_argument("--selftest", type=int, default=1,
                   help="run the packed-vs-separate and order-invariance gates (tiny config, "
                        "offline, ~10s) at job start; abort the job if any check fails")
    p.add_argument("--env_check", type=int, default=1,
                   help="verify requirements.txt pins; pip-install them if the container does not "
                        "already satisfy them, and abort the job if they still do not match")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- preflight
def run_preflight(args, device, main_rank):
    """Version pins, then the two structural gates. Returns False if the job must not proceed.

    Both run on rank 0 only (they are tiny and deterministic, so every rank would compute the same
    answer); a failure is broadcast so that under torchrun every rank exits rather than rank 0
    aborting while the others sit in a collective.
    """
    import environment

    ok = True
    if args.env_check:
        ok &= environment.enforce(prefix="ENV")
        if not ok:
            print("ABORT: package versions do not match requirements.txt. Every published number "
                  "was produced with these pins; running anyway would make the results "
                  "uncomparable. Re-run with --env_check 0 to override deliberately.")
    else:
        print("ENV: --env_check 0 -- versions NOT verified for this job.")

    if args.selftest:
        print("\n" + "=" * 78)
        print("SELFTEST: packed-vs-separate equivalence + order invariance (tiny 2-layer config)")
        print("=" * 78)
        try:
            import tests as _tests
            _tests.FAILURES.clear()
            torch.manual_seed(0)
            _tests.test_packed_vs_separate(device)
            _tests.test_order_invariance(device)
            failures = list(_tests.FAILURES)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"SELFTEST: raised {type(e).__name__}: {e}")
            failures = ["selftest raised"]
        if failures:
            ok = False
            print(f"ABORT: selftest FAILED: {failures}. The 4-D mask / position-id guarantee this "
                  f"project is built on does not hold in this container, so any accuracy number "
                  f"from this job would be uninterpretable.")
        else:
            print("SELFTEST: PASS")
    else:
        print("SELFTEST: --selftest 0 -- structural gates NOT run for this job.")

    # broadcast so every rank makes the same decision
    if args.world > 1:
        import torch.distributed as _dist
        t = torch.tensor([0.0 if ok else 1.0], device=device)
        _dist.all_reduce(t, op=_dist.ReduceOp.SUM)
        ok = t.item() == 0.0
    if main_rank:
        print(f"\nPREFLIGHT: {'PASS -- starting training' if ok else 'FAIL -- job will abort'}")
    return ok


# --------------------------------------------------------------------------- distributed
def dist_info():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    return rank, world, local


def is_main(rank):
    return rank == 0


def all_reduce_mean(value, world):
    if world == 1:
        return value
    t = torch.tensor([value], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world).item()


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, world=1, rank=0):
    model.eval()
    correct = defaultdict(int)
    count = defaultdict(int)
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        k = batch["option_mask"].sum(-1)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(**batch)
        loss, _ = compute_loss(logits.float(), batch["labels"], batch["task_type"], k,
                               label_smoothing=0.0, emd_lambda=0.0)
        hits = (logits.float().argmax(-1) == batch["labels"])
        total_loss += loss.item() * batch["labels"].numel()
        n += batch["labels"].numel()
        for name, h, tt in zip(batch["labels"].tolist(), hits.tolist(), batch["task_type"].tolist()):
            count[f"type:{TASK_NAMES[tt]}"] += 1
            correct[f"type:{TASK_NAMES[tt]}"] += h
        for name, h, dom in zip(batch["labels"].tolist(), hits.tolist(), batch["domains"]):
            count[f"domain:{dom}"] += 1
            correct[f"domain:{dom}"] += h
    if world > 1:
        keys = sorted(count)
        agg_c = torch.tensor([correct[k] for k in keys], dtype=torch.float64)
        agg_n = torch.tensor([count[k] for k in keys], dtype=torch.float64)
        dist.all_reduce(agg_c)
        dist.all_reduce(agg_n)
        correct = {k: int(v) for k, v in zip(keys, agg_c.tolist())}
        count = {k: int(v) for k, v in zip(keys, agg_n.tolist())}
        total_loss = all_reduce_mean(total_loss, world)
        n = all_reduce_mean(n, world)
    out = {"loss": total_loss / max(n, 1), "accuracy": sum(correct.values()) / max(sum(count.values()), 1)}
    for k in sorted(count):
        out[f"acc/{k}"] = correct[k] / count[k]
    model.train()
    return out


# --------------------------------------------------------------------------- training
def main(argv=None):
    args = parse_args(argv)
    rank, world, local = dist_info()
    args.world = world
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    main_rank = is_main(rank)
    amp_dtype = torch.bfloat16

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Preflight BEFORE anything expensive: the gates build a 2-layer random-init model and touch no
    # data and no download, so they cost ~10s, and they catch the two failure modes that would
    # otherwise waste a full 12k-step job: wrong library versions, and a broken attention mask.
    if not run_preflight(args, device, main_rank):
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        raise SystemExit(2)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_name)
    if args.mode == "special":
        from serialize import extend_tokenizer, resize_and_init_embeddings

    encoder = Encoder(tokenizer, args.max_state_tokens, args.max_choice_tokens,
                      mode=args.mode, pool=args.pool, noul_use_criteria=args.noul_use_criteria)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    train_tbl = load_split_table(args.train_channel, args.limit_train or None)
    val_tbl = load_split_table(args.val_channel, args.limit_val or None)
    train_set = DecisionDataset(train_tbl.select(range(len(train_tbl))), encoder,
                                shuffle_options=bool(args.shuffle_options), seed=args.seed)
    val_set = DecisionDataset(val_tbl, encoder, shuffle_options=False, seed=args.seed)

    lengths = state_token_lengths(train_tbl, tokenizer, sample=min(5000, len(train_tbl)),
                                  max_state_tokens=args.max_state_tokens,
                                  max_choice_tokens=args.max_choice_tokens)
    lengths += [0] * (len(train_tbl) - len(lengths))
    sampler = LengthBucketBatchSampler(lengths, args.batch_size, shuffle=True, seed=args.seed)
    if world > 1:
        sampler = torch.utils.data.DistributedSampler(train_set, num_replicas=world, rank=rank,
                                                       shuffle=True, seed=args.seed, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=sampler,
                              collate_fn=partial(collate, pad_id=pad_id), num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=partial(collate, pad_id=pad_id), num_workers=0)

    if args.random_init and args.vocab_size <= 0:
        args.vocab_size = max(len(tokenizer), 512)
    if args.lora_last_k_layers < 0:
        args.lora_last_k_layers = args.num_layers
    model = build_model(args)
    if args.mode == "special":
        resize_and_init_embeddings(model.backbone, tokenizer, verbose=main_rank)
    if args.gradient_checkpointing:
        try:
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model.backbone, "enable_input_require_grads"):
                model.backbone.enable_input_require_grads()
        except Exception as e:  # some peft/transformers combos lack it; not fatal
            if main_rank:
                print(f"warn: gradient checkpointing unavailable ({type(e).__name__}: {e})")
    model.to(device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local],
                                                         find_unused_parameters=True)

    backbone_params = [q for q in model.module.backbone.parameters() if q.requires_grad] if world > 1 \
        else [q for q in model.backbone.parameters() if q.requires_grad]
    head_params = list((model.module if world > 1 else model).head.parameters())
    groups = [{"params": head_params, "lr": args.lr_head}]
    if backbone_params:
        groups.insert(0, {"params": backbone_params, "lr": args.lr_lora})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    from transformers import get_cosine_schedule_with_warmup
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---- the three facts that silently change what gets trained, printed before anything else ----
    # (a) which checkpoint cell 11 evaluates, (b) how often validation runs, (c) that the cosine
    # schedule is built for exactly total_steps. A schedule whose num_training_steps disagrees with
    # the budget is the classic silent bug: the cosine never reaches its floor, so the run ends at a
    # higher LR than intended and two runs of "the same" config are not comparable.
    sched_total = int(getattr(scheduler, "num_training_steps", total_steps))
    if sched_total != total_steps:
        raise SystemExit(f"LR schedule length {sched_total} != MAX_STEPS {total_steps}; refusing to "
                         f"train against a schedule that does not match the budget")
    if args.max_steps > 0 and total_steps != args.max_steps:
        raise SystemExit(f"total_steps {total_steps} != --max_steps {args.max_steps}")
    if main_rank:
        print("=" * 78)
        print("CONFIG (the three things that decide what gets trained)")
        print("=" * 78)
        print(f"  (a) checkpoint selected for evaluation : best VALIDATION accuracy "
              f"(--model_dir/best); 'final' (last step) also saved")
        print(f"      best is chosen on the VAL split only ({len(val_tbl):,} questions, carved from "
              f"train); the TEST split is never read by train.py")
        print(f"  (b) validation + checkpoint saving     : every {args.eval_every} steps "
              f"(save_every={args.save_every})")
        print(f"  (c) LR schedule length                 : {sched_total} steps "
              f"(cosine, warmup {warmup_steps} = {args.warmup_ratio:.1%}) == MAX_STEPS {total_steps}")
        print(f"      budget: {total_steps:,} steps x {args.batch_size}x{args.grad_accum_steps} "
              f"= {total_steps * args.batch_size * args.grad_accum_steps:,} question presentations")
        print(f"      steps/epoch={steps_per_epoch:,} -> {total_steps / steps_per_epoch:.2f} epochs")
        print(f"  device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
              f"{'  ' + str(torch.cuda.get_device_properties(0).total_memory / 2**30)[:4] + ' GiB' if torch.cuda.is_available() else ''}")
        print("=" * 78, flush=True)

    tw = {}
    if args.type_weights:
        for part in args.type_weights.split(","):
            k, v = part.split(":")
            tw[int(k)] = float(v)

    step, best_acc, epoch = 0, -1.0, 0
    start_step = 0
    if args.resume:
        ckpt = args.resume
        if os.path.exists(os.path.join(ckpt, "optim.pt")):
            state = torch.load(os.path.join(ckpt, "optim.pt"), map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optim"])
            scheduler.load_state_dict(state["sched"])
            step = state["step"]
            best_acc = state.get("best_acc", -1.0)
            start_step = step
        else:
            load_checkpoint(model.module if world > 1 else model, ckpt)
        if main_rank:
            print(f"resumed from {ckpt} at step {step} best_val_acc={best_acc:.4f}")

    if main_rank:
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"trainable params: backbone={sum(q.numel() for q in backbone_params):,} "
              f"head={sum(q.numel() for q in head_params):,} "
              f"total={sum(q.numel() for q in backbone_params) + sum(q.numel() for q in head_params):,}")
        print(f"train questions={len(train_tbl):,} val questions={len(val_tbl):,} "
              f"batch={args.batch_size}x{args.grad_accum_steps} steps/epoch={steps_per_epoch} "
              f"total_steps={total_steps} device={device}")
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    meta = {"args": vars(args), "step": step, "pool": args.pool, "mode": args.mode,
            "head": args.head, "num_layers": args.num_layers,
            "max_state_tokens": args.max_state_tokens, "max_choice_tokens": args.max_choice_tokens}

    def log(**kw):
        if not main_rank:
            return
        line = " ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items())
        print(line, flush=True)
        with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(kw) + "\n")

    def save(name, merge=False):
        if not main_rank:
            return
        net = model.module if world > 1 else model
        save_checkpoint(net, os.path.join(args.model_dir, name), {**meta, "name": name, "merge": merge},
                        merge=merge)
        torch.save({"optim": optimizer.state_dict(), "sched": scheduler.state_dict(),
                    "step": step, "best_acc": best_acc},
                   os.path.join(args.model_dir, name, "optim.pt"))

    model.train()
    t0 = time.time()
    run_start = t0
    running, running_n, running_correct = 0.0, 0, 0
    done = False
    deadline = None if args.max_train_minutes <= 0 else time.time() + args.max_train_minutes * 60

    for epoch in range(args.epochs):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if hasattr(train_set, "set_epoch"):
            train_set.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(train_loader, 1):
            if micro > steps_per_epoch * args.grad_accum_steps:
                break
            if step >= total_steps:
                done = True
                break
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            k = batch["option_mask"].sum(-1)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(**batch)
            loss, per = compute_loss(logits.float(), batch["labels"], batch["task_type"], k,
                                     label_smoothing=args.label_smoothing,
                                     emd_lambda=args.emd_lambda, type_weights=tw)
            (loss / args.grad_accum_steps).backward()
            running += loss.item() * batch["labels"].numel()
            running_correct += (logits.float().argmax(-1) == batch["labels"]).sum().item()
            running_n += batch["labels"].numel()
            if micro % args.grad_accum_steps:
                continue

            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters() if world == 1 else model.module.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                sps = running_n / max(time.time() - t0, 1e-9)
                peak = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
                log(step=step, loss=running / max(running_n, 1), grad_norm=float(gn),
                    acc=running_correct / max(running_n, 1), lr_lora=args.lr_lora, lr_head=args.lr_head,
                    sps=sps, peak_gib=round(peak, 2))
                running, running_correct, running_n, t0 = 0.0, 0, 0, time.time()
                t0 = time.time()
            if step % args.save_every == 0:
                save("latest")
            if step % args.eval_every == 0:
                m = evaluate(model.module if world > 1 else model, val_loader, device, amp_dtype, world, rank)
                if main_rank:
                    line = f"step={step} val_loss={m['loss']:.6g} val_acc={m['accuracy']:.6g}"
                    types = {k: v for k, v in m.items() if k.startswith("acc/type:")}
                    doms = sorted(((k, v) for k, v in m.items() if k.startswith("acc/domain:")),
                                  key=lambda kv: -kv[1])
                    for k in sorted(types):
                        line += f" {k.replace('/', '_')}={types[k]:.6g}"
                    line += "  worst_domains=" + ",".join(
                        f"{k.split(':', 1)[1][:18]}:{v:.3f}" for k, v in doms[:5])
                    print(line, flush=True)
                    with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
                        f.write(json.dumps({"step": step, "val_loss": m["loss"],
                                            "val_acc": m["accuracy"],
                                            **{k.replace("/", "_"): v for k, v in m.items() if k != "loss"}}) + "\n")
                if m["accuracy"] > best_acc:
                    best_acc = m["accuracy"]
                    save("best")
            if deadline and time.time() > deadline:
                if main_rank:
                    print(f"stopping early at step {step}: max_train_minutes reached", flush=True)
                done = True
                break
        if done:
            break

    if step > start_step or not os.path.exists(os.path.join(args.model_dir, "final")):
        save("final", merge=False)
    if main_rank:
        m = evaluate(model.module if world > 1 else model, val_loader, device, amp_dtype, world, rank)
        summary = {"final_step": step, "best_val_acc": best_acc, "val": m,
                   "duration_sec": round(time.time() - run_start, 1),
                   # Recorded rather than assumed downstream: the notebook reads these instead of
                   # trusting that a checkpoint named "best" means best-on-validation.
                   "config": {
                       "checkpoint_selection": "best_val_accuracy",
                       "checkpoint_selection_source": "val split only; test split never read",
                       "eval_every": args.eval_every,
                       "save_every": args.save_every,
                       "max_steps": total_steps,
                       "lr_schedule_steps": sched_total,
                       "lr_warmup_steps": warmup_steps,
                       "question_presentations": total_steps * args.batch_size * args.grad_accum_steps,
                       "batch_size": args.batch_size,
                       "grad_accum_steps": args.grad_accum_steps,
                       "steps_per_epoch": steps_per_epoch,
                       "epochs": round(total_steps / steps_per_epoch, 4),
                       "seed": args.seed,
                       "head": args.head,
                       "model_dir": args.model_dir,
                       "output_dir": args.output_dir,
                   }}
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("SUMMARY " + json.dumps(summary), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()