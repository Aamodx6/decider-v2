"""Generate decider-train/arch.ipynb (not part of the repo; run once, then validate/execute)."""
import json
import pathlib

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md(r'''
# decider-v2 architecture, walked through on CPU

Three scenes, in the order the data actually moves:

1. **Pack** — one question becomes one sequence: prefix, then the options, then the answer slot.
2. **Mask** — the 5×5 block that keeps each option from seeing the others.
3. **Brain + judge** — the LoRA'd backbone produces per-option states; a head turns them into a
   distribution over options.

Everything here runs on a **2-layer, randomly-initialised Qwen3-shaped backbone** with a byte-level
tokenizer. No weights are downloaded and nothing is trained, so this notebook takes seconds on a CPU
and every number printed is a shape or an invariant, not a result. Real measurements live in
`docs/LOCAL_VERIFICATION.md`.
''')

md(r'''
## Scene 1 — pack

A multiple-choice question is not `K` separate sequences. It is **one** packed sequence:

```
[ state ... question ] [ <opt> option 0 <opt_end> ] [ <opt> option 1 <opt_end> ] ... [ <answer> ]
      prefix                    option block 1               option block 2            answer
```

Two things make that safe to batch:

* every option's **position ids restart at the prefix length**, so option `i` occupies the same
  positions it would have if it were the only option — the model cannot tell how many siblings it has
  from the positions alone;
* the **slot ids** (block ids) label which option each token belongs to, and the mask in scene 2 is
  derived from them.
''')

code(r'''
import sys, os
sys.path.insert(0, os.path.abspath("."))

import torch
from tokenization import Encoder, POOL_BEV, build_batch

class ByteTok:
    """Deterministic byte-level stand-in for the Qwen tokenizer: no download, exact round-trip."""
    vocab_size = 256
    pad_token_id = 0
    name_or_path = "byte-tiny"
    def get_vocab(self):  return {chr(i): i for i in range(256)}
    def encode(self, text): return [b for b in text.encode("utf-8") if b != 0] or [65]
    def decode(self, ids):  return bytes(i for i in ids if i != 0).decode("utf-8", errors="replace")
    def add_special_tokens(self, d): return 0
    def convert_tokens_to_ids(self, t):
        ids = self.encode(t)
        return ids[0] if ids else 65
    def __len__(self): return 256

tok = ByteTok()
enc = Encoder(tok, max_state_tokens=128, max_choice_tokens=24, mode="bev", pool=POOL_BEV)

STATE  = "The customer was charged twice for the same order."
INSTR  = "Which resolution is correct?"
OPTIONS = ["refund the duplicate charge", "ignore it", "escalate to a human", "close the ticket", "ask for a photo"]

ex = enc.encode(STATE, INSTR, OPTIONS, "choice")
print("tokens in the packed sequence :", ex["L"])
print("prefix length                 :", ex["prefix_len"])
print("options                       :", ex["n_options"])
print("answer slot starts at         :", ex["answer_start"])
print()
print(f'{"idx":>4} {"slot":>5} {"pos":>5}  token')
prefix_len, n_opt = ex["prefix_len"], ex["n_options"]
slot = []
for i in range(ex["L"]):
    if i < prefix_len:               s = "prefix"
    elif i >= ex["answer_start"]:    s = "answer"
    else:
        s = next((f"opt{j}" for j, (b, e, _) in enumerate(ex["spans"]) if b <= i < e), "opt_end")
    slot.append(s)
    if i < 4 or i >= prefix_len - 2:
        print(f'{i:>4} {s:>5} {ex["position_ids"][i]:>5}  {ex["input_ids"][i]}')
print("...")
''')

code(r'''
# The invariant that makes the packing legal: every option restarts at the prefix length.
starts = [b for (b, e, _) in ex["spans"]]
first_pos = [ex["position_ids"][b] for b in starts]
print("option start indices :", starts)
print("first position id    :", first_pos)
print("all equal to prefix  :", all(p == prefix_len for p in first_pos))

for j, (b, e, read) in enumerate(ex["spans"]):
    print(f"  option {j}: tokens [{b}, {e})  read offset {read:>2} inside the option "
          f"-> absolute index {b + read}")
print("  (collate() turns each offset into that absolute index: option_read_idx)")

assert all(p == prefix_len for p in first_pos), "an option did not restart at the prefix length"
print("\nOK: K options share one sequence and still occupy identical positions.")
''')

md(r'''
## Scene 2 — the mask

If the options shared a normal causal mask, option 3 could read option 1 and the model could compare
them directly — the answer would depend on list order, which is exactly what we refuse to let it see.

So the mask is built from the slot ids, as a `[B, 1, L, L]` **additive** mask (0 = may attend,
-inf = may not). Restricted to the option block, a 5-option question gives the 5×5 matrix below: the
diagonal only. Each option sees the whole prefix and itself, nothing else.
''')

code(r'''
batch = build_batch([ex], tok.pad_token_id)
print("attention_mask shape :", tuple(batch["attention_mask"].shape), "(B, 1, L, L)")
print("option_read_idx      :", batch["option_read_idx"].tolist())
print("option_mask          :", batch["option_mask"].tolist())

m = batch["attention_mask"][0, 0]
read = batch["option_read_idx"][0]
sub = m[read][:, read]
allowed = (sub == 0)

print("\nmask restricted to the option read positions (rows = query, cols = key):")
print("    " + " ".join(f"o{j:<3}" for j in range(allowed.shape[0])))
for i in range(allowed.shape[0]):
    print(f"  o{i}  " + " ".join(f"{'  . ' if allowed[i, j] else '  X '}" for j in range(allowed.shape[1])))
print("\n  .  = may attend       X = blocked")

assert bool(torch.equal(allowed, torch.eye(allowed.shape[0], dtype=torch.bool))), \
    "option block is not purely diagonal"
print("OK: the option block is the identity matrix -- no option can see another.")
''')

md(r'''
The first token of each option also sees the **prefix** (that is where the state and question live),
which is why the full mask is not purely diagonal — it is diagonal *within* each block, plus every
option and the answer attending to the prefix. That is the "isolating block mask" in the diagram.
''')

md(r'''
## Scene 3 — brain and judge

* **Brain** — the truncated Qwen3 backbone with LoRA adapters. It reads the packed sequence and, for
  each option, we take the hidden state at that option's *read position*: `[B, K, H]`.
* **Judge** — a small head over those `K` states. Two versions exist:
  * `head=v1` is the bev-decider **replica**: a per-option MLP, scored independently, no
    option-to-option attention. Reproducing it is what makes the comparison fair.
  * `head=v2` adds a context stack (each option attends to the others, *after* the mask has kept
    them apart inside the backbone), a rank feature for `score` questions, and optional null slot.

Shapes are printed at every step; the parameter counts are those of this 2-layer toy, not the real
0.6B model.
''')

code(r'''
import argparse
from network import build_model

def tiny(head, pool=POOL_BEV):
    ns = argparse.Namespace(
        backbone_name="tiny", num_layers=2, random_init=1, attn_implementation="sdpa",
        lora_r=4, lora_alpha=8, lora_dropout=0.0, lora_last_k_layers=2, lora_include_mlp=1,
        head=head, head_new_dim=64, head_layers=2, head_heads=4, head_dropout=0.0,
        n_ctx=4, use_ctx=1, use_rank=1, use_null=0, pool=pool)
    torch.manual_seed(0)
    return build_model(ns).eval()

for head in ("v1", "v2"):
    net = tiny(head)
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\n=== head={head} ===")
    print(f"backbone type            : {type(net.backbone).__name__}")
    print(f"trainable parameters     : {n_train:,}")
    with torch.no_grad():
        hidden = net.backbone(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"],
                              position_ids=batch["position_ids"],
                              use_cache=False).last_hidden_state
        print(f"backbone last_hidden     : {tuple(hidden.shape)}   (B, L, H)")
        bidx = torch.arange(hidden.shape[0])
        opt_states = hidden[bidx[:, None], batch["option_read_idx"]]
        print(f"per-option states        : {tuple(opt_states.shape)}   (B, K, H)")
        logits = net(**batch)
        print(f"head logits              : {tuple(logits.shape)}       (B, K)")
        probs = torch.softmax(logits.masked_fill(~batch["option_mask"], -1e9), -1)
        print(f"probabilities            : {probs.tolist()[0]}")
        print(f"they sum to              : {probs.sum(-1).tolist()[0]}")
''')

md(r'''
### The property that must hold: order invariance

Shuffle the five options and the model must return the **same probability for each option**, just in
a different order. This is not a nicety: with per-epoch option shuffling on, a head that quietly
reads list position would learn a position prior and lose accuracy on exactly the questions it was
supposed to answer.

The check below permutes the options, runs the model again, un-permutes the probabilities and prints
the largest change. Anything above fp32 noise (~1e-7) would mean the pipeline leaks position.
''')

code(r'''
import random

def probs_for(net, options):
    e = enc.encode(STATE, INSTR, options, "choice")
    b = build_batch([e], tok.pad_token_id)
    with torch.no_grad():
        lg = net(**b).float().masked_fill(~b["option_mask"], -1e9)
    return torch.softmax(lg, -1)[0]

for head in ("v1", "v2"):
    net = tiny(head)
    base = probs_for(net, OPTIONS)
    worst = 0.0
    for trial in range(5):
        perm = list(range(len(OPTIONS)))
        random.Random(head + str(trial)).shuffle(perm)
        shuffled = [OPTIONS[i] for i in perm]
        got = probs_for(net, shuffled)
        back = torch.zeros_like(base)
        back[torch.tensor(perm)] = got          # un-permute into the original order
        worst = max(worst, (back - base).abs().max().item())
    print(f"head={head}: max |p(shuffled) - p(original)| = {worst:.3e}")
    assert worst < 1e-5, f"order invariance broken for head={head} ({worst:.3e})"
print("\nOK: probabilities follow the option, not its position.")
''')

md(r'''
---

That is the whole architecture: pack once, mask so options cannot see each other, read one state per
option, and score them. The training recipe, the calibration step and the evaluation protocol are in
the repository README and `docs/ARCH_V2.md`.
''')

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"arch-{i:02d}"

out = pathlib.Path("decider-train/arch.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
               newline="\n")
print(f"wrote {out} ({len(nb['cells'])} cells)")
