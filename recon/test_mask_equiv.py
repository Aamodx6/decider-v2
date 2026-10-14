"""Phase 1 proof: does a custom 4-D additive mask + shared position ids survive HF Qwen3 with sdpa?

Test: hidden state at each option's read index from the PACKED multi-option sequence must equal the
hidden state obtained by running that option ALONE against the same prefix.
"""
import math
import os

import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3Model

torch.manual_seed(0)
torch.set_printoptions(precision=8)

HIDDEN = 64
N_LAYERS = 2
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16


def tiny_config():
    return Qwen3Config(
        vocab_size=256,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )


# ---------------------------------------------------------------- bev-train's mask/pos builder
def build_packed(prefix_ids, opt_ids, vocab=256):
    """prefix_ids: list[int]; opt_ids: list[list[int]]. Returns bev-train-compatible tensors."""
    P = len(prefix_ids)
    spans = []
    ids = list(prefix_ids)
    pos = list(range(P))
    for o in opt_ids:
        spans.append((len(ids), len(ids) + len(o)))
        ids += list(o)
        pos += list(range(P, P + len(o)))
    ans_start = len(ids)
    ans = [7, 8, 9, 10]
    ids += ans
    pos += list(range(max(pos) + 1, max(pos) + 1 + len(ans)))
    L = len(ids)
    mask = torch.zeros(L, L)
    mask[:P, :P] = torch.tril(torch.ones(P, P))
    for b, e in spans:
        mask[b:e, :P] = 1
        mask[b:e, b:e] = torch.tril(torch.ones(e - b, e - b))
    mask[ans_start:, :ans_start] = 1
    mask[ans_start:, ans_start:] = torch.tril(torch.ones(L - ans_start, L - ans_start))
    add = torch.where(mask == 0, -float("inf"), 0.0).unsqueeze(0).unsqueeze(0)
    return {
        "input_ids": torch.tensor(ids),
        "position_ids": torch.tensor(pos),
        "attention_mask": add,
        "read_idx": [e - 1 for _, e in spans],
    }


def collate(exs, pad_id=0):
    B = len(exs)
    L = max(len(e["input_ids"]) for e in exs)
    C = max(len(e["read_idx"]) for e in exs)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    att = torch.eye(L).repeat(B, 1, 1)
    read = torch.zeros((B, C), dtype=torch.long)
    cmask = torch.zeros((B, C), dtype=torch.bool)
    for b, ex in enumerate(exs):
        n = len(ex["input_ids"])
        input_ids[b, :n] = ex["input_ids"]
        position_ids[b, :n] = ex["position_ids"]
        # additive mask (0 allowed / -inf forbidden) -> bev-train's 0/1 convention (1 = may attend)
        att[b, :n, :n] = torch.where(ex["attention_mask"][0, 0] == 0, 1.0, 0.0)
        read[b, : len(ex["read_idx"])] = torch.tensor(ex["read_idx"])
        cmask[b, : len(ex["read_idx"])] = True
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": torch.where(att == 0, -float("inf"), 0.0).unsqueeze(1),
        "read_idx": read,
        "choice_mask": cmask,
    }


def run_case(attn_impl, prefix_ids, opt_ids, tag):
    torch.manual_seed(1234)
    model = Qwen3Model(tiny_config()).to(torch.float32).eval()
    model.set_attn_implementation(attn_impl) if hasattr(model, "set_attn_implementation") else None
    with torch.no_grad():
        packed = build_packed(prefix_ids, opt_ids)
        b = collate([packed])
        try:
            h = model(input_ids=b["input_ids"], position_ids=b["position_ids"],
                      attention_mask=b["attention_mask"], use_cache=False).last_hidden_state
        except Exception as e:
            print(f"  [{tag}/{attn_impl}] FAILED: {type(e).__name__}: {e}")
            return None
        diffs = []
        for i, o in enumerate(opt_ids):
            solo = build_packed(prefix_ids, [o])
            bs = collate([solo])
            hs = model(input_ids=bs["input_ids"], position_ids=bs["position_ids"],
                       attention_mask=bs["attention_mask"], use_cache=False).last_hidden_state
            d = (h[0, packed["read_idx"][i]] - hs[0, solo["read_idx"][0]]).abs().max().item()
            diffs.append(d)
        return max(diffs)


def main():
    print("=" * 78)
    print("PACKED-vs-SEPARATE EQUIVALENCE under a 4-D additive mask + shared position ids")
    print("=" * 78)
    print(f"tiny Qwen3: hidden={HIDDEN} layers={N_LAYERS} heads={HEADS} kv_heads={KV_HEADS} head_dim={HEAD_DIM}")
    print("fp32, eval mode, no grad\n")

    g = torch.Generator().manual_seed(7)
    P = 23
    prefix_ids = torch.randint(11, 200, (P,), generator=g).tolist()
    cases = {
        "k=3 opts [4,4,4]": [[3, 40, 41, 42], [3, 50, 51, 52], [3, 60, 61, 62]],
        "k=5 uneven      ": [[3, 20, 21], [3, 30, 31, 32, 33], [3, 40], [3, 50, 51, 52], [3, 60, 61]],
        "k=8 uneven      ": [[3, 10 + i, 20 + i, 30 + i] for i in range(8)],
    }
    results = {}
    for impl in ["eager", "sdpa"]:
        print(f"--- attn_implementation={impl}")
        for tag, opts in cases.items():
            d = run_case(impl, prefix_ids, opts, tag)
            if d is not None:
                print(f"  [{tag}] k={len(opts)}  max|packed - separate| = {d:.3e}")
                results[(impl, tag)] = d
        print()

    print("=" * 78)
    ok = True
    for (impl, tag), d in results.items():
        gate = d <= 1e-5
        ok &= gate
        print(f"  {impl:6s} {tag}  {d:.3e}  {'PASS' if gate else 'FAIL (gate 1e-5)'}")
    print("=" * 78)
    print("RESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())