# decider-v2 — Architecture (working name)

An improved, open, commercially-usable take on `avbiswas/bev-decider-0.4B`: a small **System One decision model** that reads a state plus typed questions and returns calibrated probabilities in **one forward pass, no generation**. API-compatible with TypeSafe Jev's `/v1/systemone` format.

> Items tagged **[ASSUMPTION]** are not specified in the bev-decider model card/README. Confirm them against `src/bev_decider` before you commit to them.

---

## 1. Goals / Non-goals

**Goals**
- Match or beat bev-decider-0.4B on JevBench, sysone-bench v2 and held-out typed decisions.
- Fix its documented weak spots: date/arithmetic, stated exceptions, answer verification, 5-level sentiment, English-only, 1k-token states.
- Keep exact choice-order invariance, calibration (ECE ≤ 0.05) and laptop-CPU viability.
- Ship with a license-clean data pipeline so the weights can be used commercially.

**Non-goals**
- Free-form text generation, chat, tool calling.
- Beating Jev 1.13 on everything (closed, unknown size).

---

## 2. Reference: what bev-decider-0.4B does

| Aspect | bev-decider-0.4B |
|---|---|
| Backbone | First **20 of 28 layers** of Qwen3-0.6B, LoRA r=8 on `q/k/v/o_proj` of layers 8–19, merged |
| Head | 2-layer self-attention, 512-d, task-type embedding, ~7.4M params, no position info |
| Params | ≈0.4B (0.31B layers + 0.16B embeddings + head) |
| Option handling | All options share the same starting position id; mask lets option tokens see only the question and their own earlier tokens. Options are compared **only** in the head |
| Question types | `choice` (probabilities per key), `noul` (P(yes)), `score` (expected value over ordered levels) |
| Context | Trained to 1,024-token states; default limit 2,048; options ≤ 64 tokens |
| Training | CE on ~940K typed questions, 12k steps, batch 64 |
| Reported | Held-out 74.7 (Jev 78.0), JevBench 65.8 (hard 46.8), sysone-bench 70.8, ECE 0.065 |
| Weak spots | Date/arithmetic, stated exceptions, sst5, answer verification (54.1), temporal/unit (54.0), English-centric |
| License | Weights CC-BY-NC-4.0 (NC training data); code Apache-2.0 |

---

## 3. Gap → change map

| Gap in bev-decider | Change in decider-v2 | Expected effect (hypothesis, validate by ablation) |
|---|---|---|
| Head only sees pooled option vectors | Head cross-attends to a **learned summary of state/question tokens** (Perceiver-style) | Better verification, exception handling |
| Deeper layers dropped (20/28) | Backbone depth is a tunable ablation (20 / 24 / 28) | Reasoning-heavy domains improve |
| LoRA only on layers 8–19, r=8 | LoRA r=16 on all kept layers (+ MLP projections) | More capacity for new data |
| Date/arithmetic unreliable | **Deterministic pre-pass** that appends derived facts to the state + synthetic data targeting it | Large gain on temporal/unit |
| Exceptions overlooked | Counterfactual / minimal-edit **exception pairs** + hard-negative stage | Fewer rule-override misses |
| `score` via CE over expected value | Rank-aware head + **CE + EMD (ordinal) loss** | Better sst5/ordinal, fewer far misses |
| No "none of the above" | Learned **null option** for `choice` | Abstention, safer routing |
| ECE 0.065 | Label smoothing + per-type **temperature scaling** (+ optional isotonic) | ECE ≤ 0.05 |
| State re-encoded per question | **Shared state KV prefix** across all questions on a state | 2–5× throughput on multi-question calls |
| 1k-token training | Length curriculum 512 → 1k → 2k → 4k | Long-state robustness |
| English-centric | Multilingual augmentation (translated intent/routing/NLI) | Closer to Jev on multilingual intent |
| NC license | License-audited data, init from Qwen3 (not from bev weights) | Commercial use possible |

---

## 4. High-level architecture

```
                 ┌──────────────── input ────────────────┐
 state (text/JSON) ──► normalizer (dates, numbers, units) ──► state' = state + [DERIVED] block
 questions {id: {type, instructions, criteria}}
                 └────────────────────────────────────────┘
                                   │
                      per question (state KV prefix cached)
                                   ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ Backbone: Qwen3-0.6B, first N layers (N=20 default), LoRA merged│
 │                                                                 │
 │  [ state' tokens ][ question tokens ]  ← causal, positions 0..P-1│
 │  [ opt_1 ... <opt> ]  ┐                                         │
 │  [ opt_2 ... <opt> ]  ├ all start at position P, mask: prefix   │
 │  [ opt_k ... <opt> ]  ┘ + own earlier tokens only               │
 └───────────────┬───────────────────────────────┬─────────────────┘
                 │ option vectors (k × d_model)   │ prefix hidden states
                 ▼                                ▼
        ┌────────────────────────────────────────────────┐
        │ Context summarizer: 8 learned queries cross-    │
        │ attend prefix hidden states → ctx (8 × 512)     │
        └───────────────────────┬────────────────────────┘
                                ▼
        ┌────────────────────────────────────────────────┐
        │ Decision head (2 blocks, 512-d, 8 heads)        │
        │  per block: option self-attn (NO positions)     │
        │             → cross-attn to ctx → FFN           │
        │  inputs: option vec + task-type emb             │
        │          (+ rank emb for score, + null option)  │
        └───────────────────────┬────────────────────────┘
                                ▼
                  1 logit per option → softmax
                                ▼
        calibration (per-type temperature) → typed answer
```

Order invariance holds because (a) every option starts at the same position id with an isolating mask, and (b) the head has no positional information and is permutation-equivariant over options.

---

## 5. Input format

### 5.1 Request (Jev / bev-decider compatible)

```json
{
  "state": "text or JSON",
  "questions": {
    "intent":  {"type": "choice", "instructions": "...", "criteria": {"refund": "...", "cancel": "..."}},
    "urgent":  {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},
    "anger":   {"type": "score",  "instructions": "...", "criteria": ["calm", "annoyed", "furious"]}
  }
}
```

### 5.2 Serialization into tokens

```
<state>{state'}</state>
<q type="{choice|noul|score}">{instructions}</q>
```
followed by one option block per candidate:
```
<opt>{key}: {description}<opt_end>
```
- `<opt>` / `<opt_end>` / `<q>` / `<state>` are **new special tokens** (add to tokenizer, resize embeddings, init from mean of existing embeddings).
- Option vector = final-layer hidden state at `<opt_end>`. **[ASSUMPTION]** bev-decider's exact pooling is not documented; ablate `<opt_end>` vs mean-pool.
- Option limit: 64 tokens (keep). State limit: 4,096 (v2), question+criteria budget: 512.

### 5.3 Type handling

| Type | Options fed to model | Output |
|---|---|---|
| `choice` | one per key (+ implicit null option) | softmax → `choice`, `probabilities` |
| `noul` | two options: `true` / `false` (criteria text if given) | `noul = P(true)` |
| `score` | one per level, with a **rank embedding** | `score = Σ i·p_i`, `probabilities` |

`noul` as a 2-option choice keeps one code path and makes P(yes) symmetric. **[ASSUMPTION]** bev may implement it differently; this is a deliberate v2 choice.

---

## 6. Attention mask and position ids

```python
import torch

def build_mask_and_pos(P: int, opt_lens: list[int], device="cpu"):
    """
    P         : prefix length (state' + question tokens)
    opt_lens  : token length of each option block (incl. <opt> ... <opt_end>)
    returns   : bool mask [T, T] (True = may attend), position_ids [T], end indices [k]
    """
    T = P + sum(opt_lens)
    mask = torch.zeros(T, T, dtype=torch.bool, device=device)
    pos = torch.arange(T, device=device)

    # prefix: ordinary causal
    mask[:P, :P] = torch.tril(torch.ones(P, P, dtype=torch.bool, device=device))

    start, ends = P, []
    for l in opt_lens:
        mask[start:start + l, :P] = True                                   # sees full prefix
        mask[start:start + l, start:start + l] = torch.tril(
            torch.ones(l, l, dtype=torch.bool, device=device))             # own earlier tokens only
        pos[start:start + l] = P + torch.arange(l, device=device)          # SAME start position
        ends.append(start + l - 1)                                         # <opt_end> index
        start += l
    return mask, pos, torch.tensor(ends, device=device)
```

Notes:
- Pass as a 4-D additive/boolean mask to the Qwen3 layers with explicit `position_ids`; do not rely on the default causal mask.
- For KV-prefix sharing (section 11), compute the prefix KV once, then run only the option tokens with the same `P`-offset positions.
- Run all inference/tests in fp32 for the invariance check; bf16 is fine for production (rounding can only flip near-ties).

---

## 7. Decision head

```python
import torch, torch.nn as nn

class HeadBlock(nn.Module):
    def __init__(self, d=512, h=8):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, h, batch_first=True)   # across options, no positions
        self.ca = nn.MultiheadAttention(d, h, batch_first=True)   # options -> state summary
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, ctx, opt_pad):
        h = self.n1(x)
        x = x + self.sa(h, h, h, key_padding_mask=opt_pad, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.ca(h, ctx, ctx, need_weights=False)[0]
        return x + self.ff(self.n3(x))


class DecisionHead(nn.Module):
    TYPES = {"choice": 0, "noul": 1, "score": 2}

    def __init__(self, d_model=1024, d=512, h=8, layers=2, n_ctx=8):
        super().__init__()
        self.proj_opt = nn.Linear(d_model, d)
        self.proj_ctx = nn.Linear(d_model, d)
        self.type_emb = nn.Embedding(3, d)
        self.rank_mlp = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))  # score only
        self.null_opt = nn.Parameter(torch.zeros(d))                                  # choice only
        self.ctx_q = nn.Parameter(torch.randn(n_ctx, d) * 0.02)
        self.ctx_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.blocks = nn.ModuleList([HeadBlock(d, h) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, opt_h, prefix_h, prefix_pad, opt_pad, type_id, rank=None, use_null=False):
        # opt_h [B,K,Dm]  prefix_h [B,P,Dm]  prefix_pad/opt_pad: True = padding
        B, K, _ = opt_h.shape
        x = self.proj_opt(opt_h) + self.type_emb(type_id)[:, None, :]
        if rank is not None:                                   # ordinal levels: rank in [0,1]
            x = x + self.rank_mlp(rank[..., None])
        if use_null:                                           # learned "none of the above"
            x = torch.cat([self.null_opt.expand(B, 1, -1), x], dim=1)
            opt_pad = torch.cat([opt_pad.new_zeros(B, 1), opt_pad], dim=1)

        c = self.proj_ctx(prefix_h)
        ctx, _ = self.ctx_attn(self.ctx_q.expand(B, -1, -1), c, c,
                               key_padding_mask=prefix_pad, need_weights=False)
        for blk in self.blocks:
            x = blk(x, ctx, opt_pad)
        logits = self.out(x).squeeze(-1)
        return logits.masked_fill(opt_pad, float("-inf"))
```

Why this stays order-invariant: no positional embeddings anywhere in the head; self-attention is permutation-equivariant; the null option is always at a fixed slot and is not a real candidate. The `rank` input for `score` is a **semantic** property of each level (not its list position the model could shuffle), so it does not break invariance.

Approx. head size: ~9–10M params at d=512, 2 blocks.

---

## 8. Outputs and losses

| Type | Probabilities | Loss |
|---|---|---|
| `choice` | softmax over options (+ null) | CE, label smoothing 0.05 |
| `noul` | softmax over {true,false} | CE, label smoothing 0.05 |
| `score` | softmax over levels | CE + λ·EMD, λ = 0.5 |

EMD for ordinal levels:

```python
def emd_loss(probs, target_idx, K):
    cdf_p = probs.cumsum(-1)
    cdf_t = (torch.arange(K, device=probs.device)[None] >= target_idx[:, None]).float()
    return (cdf_p - cdf_t).abs().mean()
```

Total loss per batch: weighted mean over question types (weights balance rare types), plus a `0.1 ×` auxiliary CE if you add the optional abstention slice (questions where the correct answer is "none").

---

## 9. Calibration

1. Hold out a calibration split (not used for early stopping): ≥ 20k questions, stratified by type and domain.
2. Fit one temperature `T_type` per type by minimizing NLL on the calibration split.
3. Optional: isotonic regression on top-label confidence for `noul`.
4. Report ECE (10 bins, top-label) per type and per domain. Gate: overall ECE ≤ 0.05.
5. Store temperatures in `config.json`; apply inside `decide()` so API users get calibrated numbers by default.

---

## 10. Compute-assist pre-pass (date / arithmetic / units)

The model should not be asked to do arithmetic. A deterministic normalizer appends derived facts to the state **before** tokenization:

```
[DERIVED]
order_date = 2026-03-15
valid_through = 2026-03-14
valid_through < order_date: true   (delta_days = 1)
subtotal + tax = 108.40 ; cap = 100.00 ; exceeds_cap: true
unit_normalized: 2.5 km = 2500 m
[/DERIVED]
```

- Extraction: `dateparser`/regex for dates, a currency/quantity parser, `pint` for units. Emit only comparisons actually present in the text or implied by question criteria.
- Train **with** the DERIVED block on (and with a 15% dropout where it is omitted) so the model degrades gracefully when the pre-pass finds nothing.
- Make the pre-pass optional per request (`"normalize": false`) for latency-critical paths.
- Add a synthetic generator for temporal/arithmetic questions (≥ 80k examples) with exact programmatic labels.

---

## 11. Inference path and serving

1. Tokenize state' once; run the backbone on `[state'][question]` once per question, **reusing the state KV cache** across questions on the same state.
2. Run the option blocks as a continuation with the cached prefix; read `<opt_end>` vectors.
3. Head → logits → temperature → typed answers.
4. Batch all questions for a state; pad options, mask padding in the head.
5. Export: fp32 reference, bf16 GPU/MPS, int8 (ONNX or MLX) for CPU/Apple Silicon. Run the invariance and parity tests on every export.

API (Jev-compatible): `POST /v1/systemone` → `{"model", "answers", "latency_ms"}`; `GET /v1/models`. Bind to `127.0.0.1` by default; add optional API-key auth (bev-decider ships with none).

---

## 12. Model variants

| Variant | Backbone | Layers kept | Head | Approx. params | Target |
|---|---|---|---|---|---|
| **v2-S** (default) | Qwen3-0.6B | 20 → ablate 24 / 28 | 512-d, 2 blocks | ~0.4–0.5B | laptop CPU |
| **v2-M** | Qwen3-1.7B | ablate 20 / 28 | 768-d, 2 blocks | ~1–1.5B (verify) | single small GPU |

Start with v2-S at 20 layers for a clean apples-to-apples comparison against bev-decider, then change **one** thing at a time (Section 15 ablations).

---

## 13. Training data and recipe

### 13.1 Data (target ≈ 1.2M typed questions)

| Slice | Share | Notes |
|---|---|---|
| Routing / intent / triage | 20% | Include multilingual translations |
| Rule- and policy-conditioned decisions | 20% | With **stated exceptions** and minimal-edit counterfactual pairs |
| NLI-style reading, multi-hop evidence | 15% | |
| Answer / solution verification | 12% | Programmatic negatives, hard-negative mined |
| Temporal / arithmetic / unit | 10% | Synthetic, exact labels, DERIVED block on |
| Extraction / procedural reasoning | 8% | |
| Sentiment / emotion / ordinal (`score`) | 8% | Include 5-level, 7-level |
| Guardrails / moderation | 5% | |
| Abstention ("none of the above") | 2% | |

Rules:
- Every example is `{state, question, answer, type, source, license}`; keep `DATA_LICENSES.md`. **I haven't verified individual dataset licenses; audit each source.**
- Prefer sources with permissive licenses and LLM-generated data from permissively licensed generators (e.g., Apache-2.0 Qwen3 larger models). Check the generator's terms for any dataset you distill from.
- Dedupe against all eval sets (JevBench, sysone-bench, held-out) with exact + n-gram overlap; log overlap counts.
- Shuffle option order every epoch (augmentation is free given invariance, but it protects against data-ordering leaks).

### 13.2 Stages

| Stage | What | Config |
|---|---|---|
| 0 | Init: Qwen3-0.6B layers 0–19, head random | **Do not** init from bev-decider weights (inherits NC license) |
| 1 | Main training: CE (+EMD for score) | LoRA r=16, α=32 on `q/k/v/o` + MLP of all kept layers; head full-train; LoRA lr 2e-4, head lr 1e-3, cosine, 3% warmup, AdamW wd 0.01, bs 128, bf16, grad-clip 1.0, 25–30k steps; length curriculum 512 → 1k → 2k → 4k |
| 1.5 | (Optional) soft-label distillation | KL to a larger teacher's option distribution, mixed 50/50 with hard CE |
| 2 | Hard-negative / counterfactual fine-tune | Mine errors from stage 1 on train-dev; 5k steps, lr ÷ 4 |
| 3 | Calibration | Temperature per type (Section 9) |
| 4 | Merge LoRA → single `model.safetensors` | Backbone bf16, head fp32 |

---

## 14. Evaluation plan and gates

Baselines are bev-decider-0.4B's published numbers. **Targets are goals, not promises.**

| Metric | bev-decider-0.4B | v2-S target |
|---|---|---|
| Held-out typed decisions (all) | 74.7 | ≥ 77 |
| JevBench (all / hard) | 65.8 / 46.8 | ≥ 70 / ≥ 52 |
| sysone-bench v2 (all) | 70.8 | ≥ 75 |
| sst5 | 30.8 | ≥ 40 |
| multilingual intent | 69.2 | ≥ 80 |
| Verification domain | 54.1 | ≥ 65 |
| Temporal/unit domain | 54.0 | ≥ 70 (with pre-pass) |
| ECE (sysone-bench) | 0.065 | ≤ 0.05 |
| Option-order max Δ (fp32, 960 shuffles, k = 3–12) | ≤ 1e-5 | ≤ 1e-5 (hard gate) |

Add your own **weakness slice** (~1k items: date compare, arithmetic caps, exceptions) and track it separately.

Tests to keep from day one: parity test vs. training checkpoint (within 0.02), order-invariance test, server contract test, export parity test (fp32 vs bf16 vs int8 top-1 agreement ≥ 99%).

### Ablations (one change at a time)
1. Layers 20 vs 24 vs 28.
2. Head with vs without ctx cross-attention.
3. `<opt_end>` pooling vs mean pooling.
4. CE vs CE+EMD on `score`.
5. DERIVED pre-pass on/off.
6. Null option on/off.
7. LoRA r=8 (layers 8–19) vs r=16 (all layers).

---

## 15. Repo layout

```
decider-v2/
├─ src/decider/
│  ├─ model.py          # backbone wrapper, mask/pos builder, DecisionHead
│  ├─ tokenizer.py      # special tokens, serialization
│  ├─ normalize.py      # DERIVED pre-pass (dates, numbers, units)
│  ├─ calibrate.py      # temperature / isotonic
│  ├─ decide.py         # public API: load(), decide()
│  ├─ serve.py          # /v1/systemone, /v1/models
│  └─ cli.py
├─ train/
│  ├─ data/             # build_dataset.py, synth_temporal.py, counterfactuals.py, DATA_LICENSES.md
│  ├─ train.py          # stages 1–2
│  ├─ distill.py        # optional 1.5
│  └─ configs/          # s.yaml, m.yaml
├─ eval/
│  ├─ run_jevbench.py
│  ├─ run_sysone_bench.py
│  ├─ run_heldout.py
│  └─ weakness_slice.jsonl
├─ tests/               # parity, invariance, server, export
├─ export/              # onnx / mlx / int8
├─ ARCH.md
└─ pyproject.toml
```

---

## 16. Milestones

| # | Deliverable | Exit criterion |
|---|---|---|
| M0 | Re-implement bev-decider-style model with Qwen3-0.6B (20 layers), head v1, same data recipe at small scale | Invariance test passes; within a few points of bev numbers on a sample |
| M1 | Data pipeline + license audit + dedupe vs eval sets | `DATA_LICENSES.md` complete, 0 eval leaks |
| M2 | Head v2 (ctx cross-attn, rank emb, null option), losses, calibration | Ablations 2, 4, 6 logged |
| M3 | DERIVED pre-pass + synthetic temporal data | Temporal/unit domain ≥ 70 |
| M4 | Exception pairs + hard-negative stage + length curriculum | Weakness slice and verification gates met |
| M5 | Multilingual augmentation | Multilingual intent ≥ 80 |
| M6 | KV-prefix serving, int8/MLX export, API server | Throughput target met; export parity ≥ 99% |
| M7 | Model card, benchmark table, release | All hard gates green |

---

## 17. Licensing

- Code: Apache-2.0.
- Backbone: Qwen3 is Apache-2.0 (include its license file).
- Weights: license is determined by your training data. If every source is permissive, release under a permissive license; otherwise inherit the strictest source terms.
- Initializing from bev-decider's weights, or training on `avbiswas/bev-decision` without checking its terms, can pull in CC-BY-NC restrictions. Verify before use.

---

## 18. Open questions

1. How exactly does bev-decider pool option vectors and encode `noul`/`score`? (Read `src/bev_decider`.)
2. Is `avbiswas/bev-decision` usable for your intended license? (Check the dataset card.)
3. Do you need Jev API parity only for the three question types, or also future ones?
4. Target deployment: CPU laptop only, or GPU server too? (Decides whether v2-M is worth building.)