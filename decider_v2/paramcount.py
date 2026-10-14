"""Print the real parameter counts for every component of the baseline replica and of v2.

Run:  python paramcount.py
Uses random-init Qwen3 configs, so it needs no download and no GPU: only the SHAPES matter.
"""
import argparse
import json

import torch
from transformers import Qwen3Config

from model import (DecisionHeadV1, DecisionHeadV2, attach_lora, build_backbone, count_params,
                   lora_target_regex)


def n(m):
    return sum(p.numel() for p in m.parameters())


def show(name, module, note=""):
    tot = n(module)
    tr = count_params(module)
    print(f"  {name:42s} total={tot:>12,}  trainable={tr:>12,}  {note}")
    return tot, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--qwen_config", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    QWEN = "Qwen/Qwen3-0.6B"
    D = 1024

    def real_config(num_layers):
        cfg = Qwen3Config.from_pretrained(args.qwen_config)
        cfg.num_hidden_layers = num_layers
        if getattr(cfg, "layer_types", None):
            cfg.layer_types = list(cfg.layer_types)[:num_layers]
        return cfg

    print("=" * 92)
    print("Qwen3-0.6B backbone, truncated (random init -> shapes only, no download)")
    print("=" * 92)
    rows = {}
    for L in (20, 24):
        b = build_backbone(QWEN, L, "sdpa", config=real_config(L))
        embed = n(b.embed_tokens)
        body = n(b) - embed
        print(f"  layers={L:2d}  embeddings={embed:,}  decoder={body:,}  final_norm={n(b.norm):,}  "
              f"total={n(b):,}")
        rows[f"backbone_{L}"] = {"embeddings": embed, "decoder": body, "final_norm": n(b.norm),
                                "total": n(b)}
        del b

    print("\n" + "=" * 92)
    print("Heads (d_model=1024)")
    print("=" * 92)
    h1 = DecisionHeadV1(hidden_dim=D, new_dim=512, num_layers=2, num_task_types=3)
    rows["head_v1"] = {"total": n(h1), "trainable": count_params(h1)}
    show("head v1 (bev replica) new_dim=512 L=2", h1)
    for k, v in h1.named_parameters():
        if v.numel() > 100000:
            print(f"      {k:48s} {v.numel():>10,}")

    for name, kw in [
        ("head v2 ctx+rank  d=512 L=2 n_ctx=8", dict(use_ctx=True, use_rank=True, use_null=False)),
        ("head v2 ctx+rank+null d=512 L=2", dict(use_ctx=True, use_rank=True, use_null=True)),
        ("head v2 no-ctx (selfattn only)", dict(use_ctx=False, use_rank=True, use_null=False)),
    ]:
        h2 = DecisionHeadV2(d_model=D, d=512, h=8, num_layers=2, n_ctx=8, **kw)
        rows[f"head_v2_{name}"] = {"total": n(h2), "trainable": count_params(h2)}
        show(name, h2)
    h2 = DecisionHeadV2(d_model=D, d=512, h=8, num_layers=2, n_ctx=8)
    print("      head v2 breakdown:")
    for k, v in h2.named_parameters():
        print(f"        {k:52s} {v.numel():>10,}")

    print("\n" + "=" * 92)
    print("LoRA adapters")
    print("=" * 92)
    for name, L, last_k, r, alpha, mlp in [
        ("baseline: r=8 attn-only layers 8-19 (of 20)", 20, 12, 8, 16, False),
        ("v2:       r=16 attn+MLP all 20 layers", 20, 20, 16, 32, True),
        ("v2 ablation: r=8 attn-only layers 8-19", 20, 12, 8, 16, False),
        ("v2 ablation: r=16 attn+MLP all 24 layers", 24, 24, 16, 32, True),
    ]:
        b = build_backbone(QWEN, L, "sdpa", config=real_config(L))
        pat = lora_target_regex(L, last_k, mlp)
        b = attach_lora(b, L, r, alpha, 0.0, last_k, mlp)
        tr = count_params(b)
        tot = n(b)
        nmod = sum(1 for m, _ in b.named_modules() if m.endswith("lora_A"))
        print(f"  {name}")
        print(f"      regex           {pat}")
        print(f"      adapted modules {nmod:4d}   lora params={tr:>12,}   backbone total={tot:,}")
        rows[name] = {"lora_params": tr, "backbone_total": tot, "adapted_modules": nmod,
                      "regex": pat}
        del b

    print("\n" + "=" * 92)
    print("Full-network totals")
    print("=" * 92)
    bb = rows["backbone_20"]["total"]
    base_lora = rows["baseline: r=8 attn-only layers 8-19 (of 20)"]["lora_params"]
    v2_lora = rows["v2:       r=16 attn+MLP all 20 layers"]["lora_params"]
    print(f"  baseline replica  = backbone {bb:,} + lora {base_lora:,} + head_v1 {rows['head_v1']['total']:,}"
          f"  -> {bb + base_lora + rows['head_v1']['total']:,}")
    print(f"  v2                = backbone {bb:,} + lora {v2_lora:,} + "
          f"head_v2 {rows['head_v2_head v2 ctx+rank  d=512 L=2 n_ctx=8']['total']:,}"
          f"  -> {bb + v2_lora + rows['head_v2_head v2 ctx+rank  d=512 L=2 n_ctx=8']['total']:,}")
    print(f"\n  published avbiswas/bev-decider-0.4B total (HF safetensors index): 477,566,976")
    print(f"  its fp32 head tensor count (HF index 'F32'):                     7,364,608")
    rows["totals"] = {
        "baseline_replica": bb + base_lora + rows["head_v1"]["total"],
        "v2": bb + v2_lora + rows["head_v2_head v2 ctx+rank  d=512 L=2 n_ctx=8"]["total"],
        "published_bev_decider_total": 477566976,
        "published_bev_decider_head": 7364608,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()