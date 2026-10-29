"""Throughput probe: separate data-pipeline cost from GPU compute cost.

Reports end-to-end samples/s (re-encoding every batch, as train.py does) and GPU-only samples/s
(the same pre-encoded batches, no tokenizer), plus peak VRAM.  Run with --device cuda.
"""
import argparse
import os
import sys
import time
from functools import partial

import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset import DecisionDataset, collate, load_split_table
from network import _unwrap, build_model, compute_loss, count_params
from tokenization import Encoder


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train_channel", required=True)
    ap.add_argument("--num_layers", type=int, default=20)
    ap.add_argument("--head", default="v2")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_include_mlp", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--micro_steps", type=int, default=10)
    ap.add_argument("--max_state_tokens", type=int, default=1024)
    ap.add_argument("--gradient_checkpointing", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prebuild", type=int, default=256)
    ap.add_argument("--out", default="", help="write the result dict here as JSON")
    args = ap.parse_args(argv)
    device = torch.device(args.device)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    enc = Encoder(tok, args.max_state_tokens, 64, mode="bev", pool="bev")
    table = load_split_table(args.train_channel)
    ds = DecisionDataset(table, enc, shuffle_options=True, seed=0)

    ns = argparse.Namespace(backbone_name="Qwen/Qwen3-0.6B", num_layers=args.num_layers,
                            random_init=0, attn_implementation="sdpa", lora_r=args.lora_r,
                            lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                            lora_last_k_layers=args.num_layers,
                            lora_include_mlp=bool(args.lora_include_mlp), head=args.head,
                            head_new_dim=512, head_layers=2, head_heads=8, head_dropout=0.0,
                            n_ctx=8, use_ctx=1, use_rank=1, use_null=0, pool="bev")
    torch.manual_seed(0)
    model = build_model(ns)
    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.backbone.enable_input_require_grads()
    model.to(device)
    # from_pretrained leaves the model in eval mode, and transformers' GradientCheckpointingLayer
    # only checkpoints when self.training is True -- without this, every layer's attention matrix
    # stays alive and peak VRAM explodes with sequence length.
    model.train()
    print(f"trainable backbone={count_params(model.backbone):,} head={count_params(model.head):,} "
          f"total={count_params(model.backbone) + count_params(model.head):,}  "
          f"grad_checkpointing={getattr(_unwrap(model.backbone), 'gradient_checkpointing', None)}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)

    def step(batch):
        b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
             for k, v in batch.items()}
        k = b["option_mask"].sum(-1)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**b)
        loss, _ = compute_loss(logits.float(), b["labels"], b["task_type"], k,
                               label_smoothing=0.05, emd_lambda=0.5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        return b["labels"].numel()

    # ---- (1) end to end: DataLoader (tokenizer inside the loop)
    sub = torch.utils.data.Subset(ds, range(min(args.prebuild, len(ds))))
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                        collate_fn=partial(collate, pad_id=tok.pad_token_id or 0),
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)
    it = iter(loader)
    for _ in range(6):                      # warm up workers + cudnn + autocast caches
        try:
            step(next(it))
        except StopIteration:
            it = iter(loader)
            step(next(it))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    for _ in range(args.micro_steps):
        try:
            n += step(next(it))
        except StopIteration:
            it = iter(loader)
            n += step(next(it))
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_to_end = n / (time.time() - t0)

    # ---- (2) GPU only: pre-encode, then time the same batches
    batches = []
    need = args.micro_steps * args.batch_size
    for i in range(args.micro_steps):
        chunk = [ds[(args.prebuild + i * args.batch_size + j) % len(ds)] for j in range(args.batch_size)]
        batches.append(collate(chunk, tok.pad_token_id or 0))
    for b in batches[:2]:
        step(b)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n = 0
    for b in batches:
        n += step(b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_only = n / (time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    lens = [b["input_ids"].shape[1] for b in batches]
    print(f"\ndevice={device} layers={args.num_layers} head={args.head} "
          f"micro_batch={args.batch_size} grad_ckpt={args.gradient_checkpointing} "
          f"max_state_tokens={args.max_state_tokens}")
    print(f"packed sequence length: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.0f}")
    print(f"end-to-end (tokenizer in the loop, num_workers={args.num_workers}): "
          f"{end_to_end:.2f} questions/s")
    print(f"GPU-only (pre-encoded batches):                                     "
          f"{gpu_only:.2f} questions/s")
    print(f"peak VRAM: {peak:.2f} GiB")
    if args.out:
        import json
        # Hardware identity is recorded so the notebook can say WHICH machine the throughput came
        # from. Without this, "3.5 q/s" is unattributable: the notebook kernel's GPU is very
        # unlikely to be the same as the SageMaker training instance, and a max_runtime estimate
        # built on the wrong card is a job that gets killed.
        hw = {"device_type": device.type, "torch": torch.__version__}
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            hw.update({
                "gpu_name": props.name,
                "gpu_total_gib": round(props.total_memory / 2**30, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                "sm_count": props.multi_processor_count,
            })
        else:
            hw.update({"gpu_name": "none (CPU)", "cpu_count": os.cpu_count()})
        with open(args.out, "w") as f:
            json.dump({"end_to_end_qps": end_to_end, "gpu_only_qps": gpu_only,
                       "peak_vram_gib": peak, "batch_size": args.batch_size,
                       "micro_steps": args.micro_steps,
                       "max_state_tokens": args.max_state_tokens, "layers": args.num_layers,
                       "head": args.head, "lora_r": args.lora_r,
                       "lora_include_mlp": args.lora_include_mlp,
                       "mean_packed_len": sum(lens) / len(lens),
                       "max_packed_len": max(lens),
                       "hardware": hw}, f, indent=2)
        print(f"wrote {args.out}")
    return {"end_to_end_qps": end_to_end, "gpu_only_qps": gpu_only, "peak_vram_gib": peak,
            "batch_size": args.batch_size, "micro_steps": args.micro_steps,
            "max_state_tokens": args.max_state_tokens, "layers": args.num_layers,
            "head": args.head, "lora_r": args.lora_r, "lora_include_mlp": args.lora_include_mlp,
            "mean_packed_len": sum(lens) / len(lens), "max_packed_len": max(lens)}


if __name__ == "__main__":
    main()