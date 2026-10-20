"""Compact structural gates for train.py --selftest, for containers where tests/test_core.py is
not shipped (the SageMaker source pack carries no tests/ directory).  Same two guarantees, the
same byte tokenizer, same 1e-5 thresholds, ~4s on CPU:

  g1  packed-vs-separate + option isolation under the [B,1,L,L] additive mask + shared position ids
  g2  option-order invariance (logit and probability, fp32)
"""

import argparse
import os
import random
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_HERE), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset import collate
from network import build_model
from tokenization import POOL_BEV, Encoder

GATE = 1e-5


def tiny_tokenizer():
    """A deterministic byte-level tokenizer: no download, works offline (same as test_core.py)."""
    class ByteTok:
        vocab_size = 256
        pad_token_id = 0
        name_or_path = "byte-tiny"

        def get_vocab(self):
            return {chr(i): i for i in range(256)}

        def encode(self, text):
            return [b for b in text.encode("utf-8") if b != 0] or [65]

        def decode(self, ids):
            return bytes(i for i in ids if i != 0).decode("utf-8", errors="replace")

        def add_special_tokens(self, d):
            added = 0
            for t in d.get("additional_special_tokens", []):
                if t not in self.get_vocab():
                    self._extra = getattr(self, "_extra", {})
                    self._extra[t] = 256 + added
                    added += 1
            return added

        def convert_tokens_to_ids(self, t):
            extra = getattr(self, "_extra", {})
            if t in extra:
                return extra[t]
            ids = self.encode(t)
            return ids[0] if ids else 65

        def __len__(self):
            return 256 + len(getattr(self, "_extra", {}))
    return ByteTok()


def tiny_model(attrs_like_tests=True):
    ns = argparse.Namespace(backbone_name="tiny", num_layers=2, random_init=1,
                            attn_implementation="sdpa", lora_r=4, lora_alpha=8, lora_dropout=0.0,
                            lora_last_k_layers=2, lora_include_mlp=1, head="v2",
                            head_new_dim=64, head_layers=2, head_heads=4, head_dropout=0.0,
                            n_ctx=4, use_ctx=1, use_rank=1, use_null=0, pool=POOL_BEV)
    torch.manual_seed(0)
    return build_model(ns)


@torch.no_grad()
def _option_hidden(net, batch, device):
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    hidden = net.backbone(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                          position_ids=b["position_ids"], use_cache=False).last_hidden_state.float()
    bidx = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[bidx[:, None], b["option_read_idx"]]


def run_gates(device):
    """-> [] on success, else the list of failing gate names."""
    failures = []
    tok = tiny_tokenizer()
    enc = Encoder(tok, max_state_tokens=512, max_choice_tokens=64, mode="bev", pool=POOL_BEV)
    torch.manual_seed(0)
    net = tiny_model().to(device).eval()
    g = torch.Generator().manual_seed(7)
    prefix = " ".join(chr(97 + int(v) % 26) for v in torch.randint(0, 26, (60,), generator=g))

    def print_gate(name, worst, cap=GATE):
        ok = worst <= cap
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  max delta {worst:.3e} (gate {cap:.0e})")
        if not ok:
            failures.append(name)

    # g1: packed-vs-separate
    worst = 0.0
    for k, lens in [(5, [3, 6, 2, 5, 4]), (8, [2 + (i % 4) for i in range(8)])]:
        opts = [" ".join(chr(97 + int(v) % 26) for v in torch.randint(0, 26, (n,), generator=g))
                for n in lens[:k]]
        packed = enc.encode(prefix, "pick one", opts, "choice")
        h = _option_hidden(net, collate([packed], tok.pad_token_id), device)[0]
        for i in range(k):
            solo = enc.encode(prefix, "pick one", [opts[i]], "choice")
            hs = _option_hidden(net, collate([solo], tok.pad_token_id), device)[0, 0]
            worst = max(worst, (h[i] - hs).abs().max().item())
    print_gate("packed-vs-separate option hidden states", worst)

    # option isolation: dropping an option must not move the others
    torch.manual_seed(1)
    net2 = tiny_model().to(device).eval()
    opts = ["alpha beta gamma", "delta epsilon", "zeta eta theta iota"]
    full_ex = enc.encode(prefix, "pick one", opts, "choice")
    hf = _option_hidden(net2, collate([full_ex], tok.pad_token_id), device)[0]
    keep = [0, 2]
    red = enc.encode(prefix, "pick one", [opts[i] for i in keep], "choice")
    hr = _option_hidden(net2, collate([red], tok.pad_token_id), device)[0]
    drift = max((hf[i] - hr[j]).abs().max().item() for j, i in enumerate(keep))
    print_gate("option isolation (dropping an option does not move the others)", drift)

    # g2: order invariance (probabilities), option order shuffles
    torch.manual_seed(3)
    net3 = tiny_model().to(device).eval()
    worst_p = 0.0
    k = 8
    for seed_off in range(4):
        opts = [f"option {i}: " + " ".join(["w"] * (2 + i)) for i in range(k)]
        ex = enc.encode("some state text here", "which one", opts, "choice")
        ex |= {"label": 0, "domain": "d", "idx": 0}
        b = collate([ex], tok.pad_token_id)
        b = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in b.items()}
        with torch.no_grad():
            base = net3(**b).float()
        base = base.masked_fill(~b["option_mask"], -1e9)
        p0 = torch.softmax(base, -1)[0]
        perm = list(range(k))
        random.Random(seed_off).shuffle(perm)
        alt_ex = enc.encode("some state text here", "which one", [opts[i] for i in perm], "choice")
        alt_ex |= {"label": perm.index(0), "domain": "d", "idx": 0}
        ab = collate([alt_ex], tok.pad_token_id)
        ab = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in ab.items()}
        with torch.no_grad():
            alt = net3(**ab).float()
        alt = alt.masked_fill(~ab["option_mask"], -1e9)
        pa = torch.softmax(alt, -1)[0]
        back = torch.zeros_like(p0)
        back[torch.tensor(perm, device=p0.device)] = pa
        worst_p = max(worst_p, (back - p0).abs().max().item())
    print_gate("order invariance mode=bev pool=bev", worst_p)

    return failures
