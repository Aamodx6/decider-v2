"""tests.py -- the four gates.  Run with `python tests.py [--device cpu] [--tiny]`.

(a) packed-vs-separate equivalence   the [B,1,L,L] additive mask + shared position ids reproduce
                                      running each option alone against the same prefix
(b) order invariance                  permuting options does not move any probability (fp32)
(c) loss sanity                       EMD is exactly 0 for a perfect prediction; label smoothing
                                      behaves; padding slots are never selected
(d) tiny overfit                      a 2-layer random-init Qwen3 must reach >=95% train accuracy on
                                      64 examples in <300 steps

`--full` additionally builds the real 20-layer Qwen3-0.6B (needs a download) and runs (a) and (b)
against it; it is skipped by default because it is slow.
"""

import argparse
import csv
import math
import os
import random
import shutil
import sys
import tempfile
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import DecisionDataset, collate
from losses import compute_loss, emd_loss, expected_calibration_error
from model import build_model
from serialize import (POOL_BEV, POOL_OPT_END, TASK_TYPES, Encoder, build_batch, pad_ranks)

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)
    return ok


# --------------------------------------------------------------------------- fixtures
def tiny_tokenizer():
    """A deterministic byte-level tokenizer: no download, works offline.

    vocab 256 = the 256 byte values, so every token id is a real byte and decoding is exact.
    """
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


TINY_CFG = dict(
    backbone_name="tiny", num_layers=2, random_init=1, head="v2", head_new_dim=64, head_layers=2,
    head_heads=4, n_ctx=4, lora_r=4, lora_alpha=8, lora_dropout=0.0, lora_last_k_layers=2,
    lora_include_mlp=1, pool=POOL_BEV, mode="bev", num_layers_=2,
)


def tiny_model(head="v2", pool=POOL_BEV):
    ns = argparse.Namespace(backbone_name="tiny", num_layers=2, random_init=1,
                            attn_implementation="sdpa", lora_r=4, lora_alpha=8, lora_dropout=0.0,
                            lora_last_k_layers=2, lora_include_mlp=1, head=head,
                            head_new_dim=64, head_layers=2, head_heads=4, head_dropout=0.0,
                            n_ctx=4, use_ctx=1, use_rank=1, use_null=0, pool=pool)
    torch.manual_seed(0)
    return build_model(ns)


def with_meta(ex, label=0, domain="synthetic", idx=0):
    ex["label"] = label
    ex["domain"] = domain
    ex["idx"] = idx
    return ex


def synthetic_examples(n, seed=0, encoder=None):
    rng = random.Random(seed)
    words = ["refund", "cancel", "urgent", "invoice", "policy", "yes", "no", "policyholder",
             "claim", "premium", "deadline", "March", "2026", "agent", "policy", "territory"]
    types = ["choice", "noul", "score"]
    out = []
    for i in range(n):
        t = types[i % 3]
        state = " ".join(rng.choice(words) for _ in range(rng.randint(8, 40)))
        instr = f"question {i} about the state"
        if t == "choice":
            k = rng.randint(2, 6)
            keys = [f"opt{j}" for j in range(k)]
            options = [f"{j}: {rng.choice(words)}" for j in range(k)]
            label = rng.randrange(k)
        elif t == "noul":
            options, keys = ["No", "Yes"], ["false", "true"]
            label = rng.randint(0, 1)
        else:
            k = rng.choice([3, 4, 5])
            options = [f"{j}: level{j}" for j in range(k)]
            keys = [str(j) for j in range(k)]
            label = rng.randrange(k)
        out.append(encoder.encode(state, instr, options, t) | {"label": label, "domain": "synthetic",
                                                               "idx": i})
    return out


# --------------------------------------------------------------------------- (a) equivalence
@torch.no_grad()
def option_hidden(net, batch, device):
    """[B, K, H] backbone hidden states at each option's read position."""
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    hidden = net.backbone(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                          position_ids=b["position_ids"], use_cache=False).last_hidden_state.float()
    bidx = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[bidx[:, None], b["option_read_idx"]]


def test_packed_vs_separate(device, full=False):
    print("\n(a) packed-vs-separate equivalence under the 4-D additive mask + shared position ids")
    tok = tiny_tokenizer()
    enc = Encoder(tok, max_state_tokens=512, max_choice_tokens=64, mode="bev", pool=POOL_BEV)
    torch.manual_seed(0)
    net = tiny_model(head="v2").to(device).eval()
    g = torch.Generator().manual_seed(7)
    prefix = " ".join(chr(97 + int(v) % 26) for v in torch.randint(0, 26, (60,), generator=g))
    worst = 0.0
    for k, lens in [(3, [4, 4, 4]), (5, [3, 6, 2, 5, 4]), (8, [2 + (i % 4) for i in range(8)])]:
        opts = [" ".join(chr(97 + int(v) % 26) for v in
                         torch.randint(0, 26, (n,), generator=g)) for n in lens[:k]]
        packed = enc.encode(prefix, "pick one", opts, "choice")
        h = option_hidden(net, collate([packed], tok.pad_token_id), device)[0]
        d_k = 0.0
        for i in range(k):
            solo = enc.encode(prefix, "pick one", [opts[i]], "choice")
            hs = option_hidden(net, collate([solo], tok.pad_token_id), device)[0, 0]
            d_k = max(d_k, (h[i] - hs).abs().max().item())
        worst = max(worst, d_k)
        print(f"      k={k}: max|packed - solo| over option read positions = {d_k:.3e}")
    check("packed-vs-separate option hidden states", worst <= 1e-5, f"max delta {worst:.3e} (gate 1e-5)")

    # the masking guarantee itself: option i's hidden state must NOT change when option j is removed
    torch.manual_seed(1)
    net2 = tiny_model(head="v2").to(device).eval()
    opts = ["alpha beta gamma", "delta epsilon", "zeta eta theta iota"]
    full_ex = enc.encode(prefix, "pick one", opts, "choice")
    hf = option_hidden(net2, collate([full_ex], tok.pad_token_id), device)[0]
    keep = [0, 2]
    red = enc.encode(prefix, "pick one", [opts[i] for i in keep], "choice")
    hr = option_hidden(net2, collate([red], tok.pad_token_id), device)[0]
    drift = max((hf[i] - hr[j]).abs().max().item() for j, i in enumerate(keep))
    check("option isolation (dropping an option does not move the others)", drift <= 1e-5,
          f"max delta {drift:.3e}")

    # same guarantee across sdpa and eager, and against the bev head replica too
    for impl in ("eager", "sdpa"):
        for head in ("v1", "v2"):
            ns = argparse.Namespace(backbone_name="tiny", num_layers=2, random_init=1,
                                    attn_implementation=impl, lora_r=4, lora_alpha=8, lora_dropout=0.0,
                                    lora_last_k_layers=2, lora_include_mlp=1, head=head,
                                    head_new_dim=64, head_layers=2, head_heads=4, head_dropout=0.0,
                                    n_ctx=4, use_ctx=1, use_rank=1, use_null=0, pool=POOL_BEV)
            torch.manual_seed(3)
            m = build_model(ns).to(device).eval()
            hh = option_hidden(m, collate([full_ex], tok.pad_token_id), device)[0]
            ss = option_hidden(m, collate([enc.encode(prefix, "pick one", [opts[1]], "choice")],
                                          tok.pad_token_id), device)[0, 0]
            d = (hh[1] - ss).abs().max().item()
            check(f"attn_implementation={impl} head={head} honours the custom 4-D mask", d <= 1e-5,
                  f"max delta {d:.3e}")


# --------------------------------------------------------------------------- (b) invariance
def masked(logits, option_mask=None):
    return logits.float().masked_fill(~option_mask, -1e30) if option_mask is not None \
        else logits.float()


def test_order_invariance(device, full=False):
    print("\n(b) option-order invariance (fp32, head logits and probabilities)")
    tok = tiny_tokenizer()
    for mode, pool in (("bev", POOL_BEV), ("special", POOL_OPT_END)):
        enc = Encoder(tok, 512, 64, mode=mode, pool=pool)
        net = tiny_model(head="v2", pool=pool).to(device).eval()
        if mode == "special":
            from serialize import resize_and_init_embeddings
            resize_and_init_embeddings(net.backbone, tok, verbose=False)
        worst, worst_p = 0.0, 0.0
        for k in (3, 5, 8, 12):
            opts = [f"option {i}: " + " ".join(["w"] * (2 + i)) for i in range(k)]
            ex = enc.encode("some state text here", "which one", opts, "choice")
            ex |= {"label": 0, "domain": "d", "idx": 0}
            b = collate([ex], tok.pad_token_id)
            b = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in b.items()}
            with torch.no_grad():
                base = net(**b).float()
            base = base.masked_fill(~b["option_mask"], -1e9)
            p0 = torch.softmax(base, -1)[0]
            for _ in range(8):
                perm = list(range(k))
                random.Random(k + _).shuffle(perm)
                alt_ex = enc.encode("some state text here", "which one", [opts[i] for i in perm], "choice")
                alt_ex |= {"label": perm.index(0), "domain": "d", "idx": 0}
                ab = collate([alt_ex], tok.pad_token_id)
                ab = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in ab.items()}
                with torch.no_grad():
                    alt = net(**ab).float()
                alt = alt.masked_fill(~ab["option_mask"], -1e9)
                pa = torch.softmax(alt, -1)[0]
                back = torch.zeros_like(p0)
                back[torch.tensor(perm, device=p0.device)] = pa
                worst_p = max(worst_p, (back - p0).abs().max().item())
                la = torch.zeros_like(base[0])
                la[torch.tensor(perm, device=p0.device)] = alt[0]
                worst = max(worst, (la - base[0]).abs().max().item())
        check(f"order invariance mode={mode} pool={pool}", worst_p <= 1e-5,
              f"max |dp| = {worst_p:.3e}, max |dlogit| = {worst:.3e} (gate 1e-5)")

    # `score` is the case that breaks if the rank embedding is attached to the SLOT instead of the
    # level: the ordinal feature must travel with the level text, not with the list position.
    print("\n(b2) order invariance for score questions (rank embedding travels with the level)")
    tok = tiny_tokenizer()
    enc = Encoder(tok, 512, 64, mode="bev", pool=POOL_BEV)
    net = tiny_model(head="v2").to(device).eval()
    for K in (3, 5, 7):
        levels = [f"{i}: level{i} descriptor" for i in range(K)]
        ranks = [i / (K - 1) for i in range(K)]
        ex = enc.encode("a review", "how extreme is it", levels, "score", ranks=ranks)
        ex.update({"label": 0, "domain": "d", "idx": 0})
        b = collate([ex], tok.pad_token_id)
        b = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in b.items()}
        with torch.no_grad():
            base = masked(net(**b), b["option_mask"])
        p0 = torch.softmax(base, -1)[0]
        d = 0.0
        for _ in range(8):
            perm = list(range(K))
            random.Random(K + _).shuffle(perm)
            ae = enc.encode("a review", "how extreme is it", [levels[i] for i in perm], "score",
                            ranks=[ranks[i] for i in perm])
            ae.update({"label": perm.index(0), "domain": "d", "idx": 0})
            ab = collate([ae], tok.pad_token_id)
            ab = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in ab.items()}
            with torch.no_grad():
                alt = masked(net(**ab), ab["option_mask"])
            pa = torch.softmax(alt, -1)[0]
            back = torch.zeros_like(p0)
            back[torch.tensor(perm, device=p0.device)] = pa
            d = max(d, (back - p0).abs().max().item())
        check(f"score order invariance K={K}", d <= 1e-5, f"max |dp| = {d:.3e}")


# --------------------------------------------------------------------------- (c) losses
def test_losses(device):
    print("\n(c) loss sanity")
    K = 5
    idx = torch.tensor([0, 2, 4])
    perfect = torch.zeros(3, K)
    perfect[torch.arange(3), idx] = 1.0
    v = emd_loss(perfect, idx, K)
    check("EMD == 0 for a perfect prediction", float(v.max()) == 0.0, f"emd={v.tolist()}")
    off = perfect.clone()
    off[0] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
    v = emd_loss(off, idx, K)
    check("EMD == |a-b|/K for a one-off error", abs(float(v[0]) - 1.0 / K) < 1e-6,
          f"emd={float(v[0]):.6f} expected {1.0/K:.6f}")

    # masked padding columns must not distort label smoothing.  With torch's own
    # F.cross_entropy(..., label_smoothing=eps) a -1e9 padding column adds eps/K * 1e9 here.
    from losses import smoothed_cross_entropy
    torch.manual_seed(0)
    mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
    logits = torch.tensor([[2.0, 1.0, 0.0, -1e9], [0.0, 3.0, 1.0, -1e9]], requires_grad=True)
    labels = torch.tensor([0, 1])
    ours = smoothed_cross_entropy(logits, labels, mask, 0.05).sum()
    ours.backward()
    check("padding columns get exactly zero gradient", float(logits.grad[0, 3]) == 0.0,
          f"grad={logits.grad[0, 3].tolist()}")
    full = smoothed_cross_entropy(logits.detach()[:, :3], labels, mask[:, :3], 0.05)
    check("masked smoothing == unmasked smoothing on the valid columns",
          abs(float(ours.detach()) - float(full.sum())) < 1e-6,
          f"masked={float(ours.detach()):.6f} unmasked={float(full.sum()):.6f}")
    naive = F.cross_entropy(logits.detach(), labels, reduction="sum", label_smoothing=0.05)
    check("torch's own smoothed CE is unusable with masked columns", float(naive) > 1e6,
          f"torch CE={float(naive):.3e} vs ours={float(ours.detach()):.6f}")

    tt = torch.tensor([TASK_TYPES["choice"], TASK_TYPES["score"]])
    nopt = torch.tensor([3, 3])
    lg = torch.tensor([[2.0, 1.0, 0.0, -1e9], [0.0, 3.0, 1.0, -1e9]], requires_grad=True)
    loss, per = compute_loss(lg, labels, tt, nopt, label_smoothing=0.05, emd_lambda=0.5)
    loss.backward()
    check("full loss is finite, small and differentiable",
          torch.isfinite(loss).item() and 0 < float(loss.detach()) < 10
          and torch.isfinite(lg.grad).all().item(),
          f"loss={float(loss.detach()):.6f}")
    check("score rows get CE + 0.5*EMD", float(per[1]) > float(
        smoothed_cross_entropy(lg.detach()[1:2], labels[1:2], mask[1:2], 0.05)[0]),
        f"per-example={per.tolist()}")

    ls = F.cross_entropy(torch.tensor([[10.0, 0.0]]), torch.tensor([0]), label_smoothing=0.05)
    no_ls = F.cross_entropy(torch.tensor([[10.0, 0.0]]), torch.tensor([0]))
    check("label smoothing penalises a confident CE", float(ls) > float(no_ls),
          f"ls={float(ls):.6f} > ce={float(no_ls):.6f}")

    probs = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.6, 0.4], [0.6, 0.4]])
    lab = torch.tensor([0, 0, 0, 1])
    ece = expected_calibration_error(probs, lab, num_bins=10)
    # bin (0.5,0.6] holds the two 0.6-confidence samples, one of which is correct:
    # |0.6 - 0.5| * 2/4 = 0.05
    check("ECE of a calibrated set is ~0", abs(float(ece) - 0.05) < 1e-6,
          f"ece={float(ece):.4f} (analytic 0.05)")
    probs2 = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    ece2 = expected_calibration_error(probs2, lab, num_bins=10)
    check("ECE of an over-confident set is large", float(ece2) > 0.2, f"ece={float(ece2):.4f}")


# --------------------------------------------------------------------------- (d) overfit
def test_overfit(device, steps=300, n=64, target=0.95):
    print(f"\n(d) tiny overfit: {n} synthetic examples, <= {steps} steps, target >= {target:.0%} train acc")
    tok = tiny_tokenizer()
    enc = Encoder(tok, 256, 64, mode="bev", pool=POOL_BEV)
    exs = synthetic_examples(n, seed=11, encoder=enc)
    net = tiny_model(head="v2").to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    losses = []
    t0 = time.time()
    step = 0
    bs = 8
    while step < steps:
        for s in range(0, len(exs), bs):
            if step >= steps:
                break
            batch = collate(exs[s:s + bs], tok.pad_token_id)
            batch = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
            k = batch["option_mask"].sum(-1)
            logits = net(**batch).float()
            loss, _ = compute_loss(logits, batch["labels"], batch["task_type"], k,
                                   label_smoothing=0.05, emd_lambda=0.5)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            step += 1
    net.eval()
    correct = 0
    with torch.no_grad():
        for s in range(0, len(exs), bs):
            batch = collate(exs[s:s + bs], tok.pad_token_id)
            batch = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
            k = batch["option_mask"].sum(-1)
            logits = net(**batch).float()
            correct += int((logits.argmax(-1) == batch["labels"]).sum())
    acc = correct / len(exs)
    dt = time.time() - t0
    check("overfit reaches the accuracy target", acc >= target,
          f"acc={acc:.3f} after {step} steps, loss {losses[0]:.4f} -> {losses[-1]:.4f}, {dt:.1f}s")


# --------------------------------------------------------------------------- (e) decision rule
def _fake_csv(path, correct, n_options=2):
    """Minimal predictions CSV in the schema evaluate.py writes."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["q_index", "task_type", "domain", "n_options", "label", "pred", "p_label"])
        for i, c in enumerate(correct):
            label = i % n_options
            pred = label if c else (1 - label)
            w.writerow([i, "choice", "d", n_options, label, pred, 0.5])
    return path


def test_decision_rule(tmpdir=None):
    """Every branch of the SEPARABLE / PROVISIONAL / NOT SEPARABLE rule, plus the bootstrap.

    Includes the two real 100-step dry-run configurations measured on this machine
    (LOCAL_VERIFICATION.md 8.2), because those are the cases the rule exists for: the same
    configuration gave a 2.52 pp seed spread in one run and 0.05 pp in another.
    """
    import decide
    print("\n(e) decision rule: paired bootstrap + the four SEPARABLE conditions")
    tmp = tmpdir or tempfile.mkdtemp(prefix="decide_tests_")

    def mk(name, cb, cv):
        return (_fake_csv(os.path.join(tmp, f"{name}_b.csv"), cb),
                _fake_csv(os.path.join(tmp, f"{name}_v.csv"), cv))

    # ---- bootstrap sanity ----------------------------------------------------------------
    n = 4000
    # a deterministic, clearly-separable case: v2 strictly better on 10% of questions
    rng = random.Random(3)
    cb = [rng.randint(0, 1) for _ in range(n)]
    cv = list(cb)
    for i in range(0, n, 10):        # v2 fixes every 10th question v2-style
        if cv[i] == 0:
            cv[i] = 1
    bcsv, vcsv = mk("boot", cb, cv)
    cbk, cvk = decide.load_pair(bcsv, vcsv)
    boot = decide.paired_bootstrap(cbk, cvk, n_resamples=2000)
    check("bootstrap: reproduces the observed accuracy delta",
          abs(boot["delta_pp"] - 100 * (sum(cvk) - sum(cbk)) / n) < 1e-9,
          f"{boot['delta_pp']:+.4f} pp")
    check("bootstrap: deterministic for a fixed seed",
          decide.paired_bootstrap(cbk, cvk, n_resamples=2000) == boot,
          "same seed -> same CI")
    check("bootstrap: CI brackets the point estimate",
          boot["ci_low_pp"] <= boot["delta_pp"] <= boot["ci_high_pp"],
          f"[{boot['ci_low_pp']:+.2f}, {boot['ci_high_pp']:+.2f}]")
    check("bootstrap: a real effect excludes 0",
          boot["excludes_zero"] and boot["ci_low_pp"] > 0, f"low={boot['ci_low_pp']:+.2f}")

    # identical arms -> zero effect -> CI includes 0
    bcsv0, vcsv0 = mk("null", cb, list(cb))
    cb0, cv0 = decide.load_pair(bcsv0, vcsv0)
    boot0 = decide.paired_bootstrap(cb0, cv0, n_resamples=2000)
    check("bootstrap: identical arms give a null CI containing 0",
          not boot0["excludes_zero"] and abs(boot0["delta_pp"]) < 1e-9,
          f"[{boot0['ci_low_pp']:+.2f}, {boot0['ci_high_pp']:+.2f}]")

    # pairing must be validated, not assumed
    bad = os.path.join(tmp, "mismatch_v.csv")
    with open(bcsv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows[5]["label"] = str((int(rows[5]["label"]) + 1) % 2)     # perturb one label
    with open(bad, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    try:
        decide.load_pair(bcsv, bad)
        check("bootstrap: mismatched question sets are rejected", False, "no error raised")
    except ValueError as e:
        check("bootstrap: mismatched question sets are rejected", "labels differ" in str(e),
              str(e)[:60])

    # ---- the four conditions, one at a time -------------------------------------------------
    def v(delta, spread, excl, seeds):
        return decide.decide(delta, spread, {"excludes_zero": excl}, seeds)[0]

    check("rule: (a) CI includes 0 -> NOT SEPARABLE",
          v(4.0, 0.05, False, 3) == "NOT SEPARABLE")
    check("rule: (b) delta <= seed spread -> NOT SEPARABLE",
          v(1.0, 1.0, True, 3) == "NOT SEPARABLE", "spread == delta is not '>'")
    check("rule: (c) delta < 1.0 pp -> NOT SEPARABLE",
          v(0.9, 0.05, True, 3) == "NOT SEPARABLE")
    check("rule: (c) delta exactly 1.0 pp passes",
          v(1.0, 0.05, True, 3) == "SEPARABLE", "boundary is inclusive")
    check("rule: (d) 2 seeds -> PROVISIONAL",
          v(4.0, 0.05, True, 2) == "PROVISIONAL")
    check("rule: (d) 1 seed -> NOT SEPARABLE (spread is 0 by construction, so (b) is vacuous)",
          v(4.0, 0.05, True, 1) == "NOT SEPARABLE")
    check("rule: all four hold -> SEPARABLE", v(4.0, 0.05, True, 3) == "SEPARABLE")
    check("rule: a large negative delta can be SEPARABLE",
          v(-4.0, 0.05, True, 3) == "SEPARABLE", "|delta|, not delta")

    # ---- the real measured configurations ---------------------------------------------------
    # Run A (LOCAL_VERIFICATION.md 8.4, previous dry-run): baseline spread 2.52, delta 2.64.
    # 2.64 > 2.52 is TRUE, so (b) passes by 0.12 pp -- this is exactly the knife-edge the rule is
    # built around, and it is why the earlier 2-seed rule called it separable on a 0.12 pp margin.
    check("rule: real config A (delta 2.64 vs spread 2.52) -> PROVISIONAL, on a 0.12 pp margin",
          decide.decide(2.64, 2.52, {"excludes_zero": True}, 2)[0] == "PROVISIONAL",
          "passes (b) by 0.12 pp; 2 seeds so not SEPARABLE")
    check("rule: real config A with the same spread and 3 seeds -> SEPARABLE",
          decide.decide(2.64, 2.52, {"excludes_zero": True}, 3)[0] == "SEPARABLE")
    # the same config with the spread LARGER than the delta fails (b) outright
    check("rule: spread 2.80 > delta 2.64 -> NOT SEPARABLE (fails b)",
          decide.decide(2.64, 2.80, {"excludes_zero": True}, 3)[0] == "NOT SEPARABLE")
    # Run B (current dry-run): baseline spread 0.05, delta 4.03, 2 seeds.
    check("rule: real config B (spread 0.05, delta 4.03, 2 seeds) -> PROVISIONAL",
          decide.decide(4.03, 0.05, {"excludes_zero": True}, 2)[0] == "PROVISIONAL")
    check("rule: real config B with a third agreeing seed -> SEPARABLE",
          decide.decide(4.03, 0.05, {"excludes_zero": True}, 3)[0] == "SEPARABLE")

    shutil.rmtree(tmp, ignore_errors=True)


def test_launch_guards(tmpdir=None):
    """The notebook's launch-time guards, tested here so they cannot silently regress.

    Two guards exist because both failure modes are silent. (1) The budget fingerprint: reusing a
    checkpoint trained under a DIFFERENT budget looks like it worked. (2) The spot config: SageMaker
    rejects max_wait < max_run, and a 30-minute wait cannot win 40 hours of capacity anyway.

    These mirror the logic in build_notebook.py's launch(); the notebook is the only place that can
    import the SDK, so the arithmetic is duplicated here rather than imported.
    """
    print("\n(e2) launch guards: budget fingerprint + spot max_wait >= max_run")
    tmp = tmpdir or tempfile.mkdtemp(prefix="launch_tests_")

    # --- budget fingerprint, parsed exactly as launch() parses hyper()'s output ---------------
    def fingerprint(max_steps, batch_size, grad_accum, target, seed):
        args = [f"--max_steps={max_steps}", f"--batch_size={batch_size}",
                f"--grad_accum_steps={grad_accum}", f"--seed={seed}"]
        kv = dict(a.lstrip("-").split("=", 1) for a in args if "=" in a)
        return {"max_steps": int(kv["max_steps"]), "batch_size": int(kv["batch_size"]),
                "grad_accum_steps": int(kv["grad_accum_steps"]),
                "target_presentations": target,
                "presentations": int(kv["max_steps"]) * int(kv["batch_size"]) * int(kv["grad_accum_steps"]),
                "seed": int(kv["seed"])}

    def mismatch(prev, now):
        return [f for f in ("max_steps", "batch_size", "grad_accum_steps", "presentations",
                            "target_presentations", "seed") if prev.get(f) != now.get(f)]

    p940 = fingerprint(14688, 32, 2, 940_000, 0)
    check("guard: the 940K budget fingerprints as 14,688 steps / 940,032 presentations",
          p940["max_steps"] == 14688 and p940["presentations"] == 940032,
          f"{p940['presentations']:,}")
    check("guard: identical config -> no mismatch, so a Completed job may be skipped",
          mismatch(p940, fingerprint(14688, 32, 2, 940_000, 0)) == [])
    check("guard: 12k vs 14.7k is detected as a budget change",
          len(mismatch(p940, fingerprint(12000, 32, 2, 768_000, 0))) >= 2,
          f"fields differing: {mismatch(p940, fingerprint(12000, 32, 2, 768_000, 0))}")
    check("guard: halving batch_size is detected (presentations halve)",
          "batch_size" in mismatch(p940, fingerprint(14688, 16, 2, 940_000, 0)))
    check("guard: a different seed is detected (so staged seeds are NOT skipped)",
          mismatch(p940, fingerprint(14688, 32, 2, 940_000, 1)) == ["seed"])

    # --- spot config: SageMaker requires MaxWaitTime >= MaxRuntime ---------------------------
    MAX_RUNTIME_HOURS = 72
    MAX_RUNTIME = MAX_RUNTIME_HOURS * 3600
    derived_wait_min = MAX_RUNTIME_HOURS * 60
    check("guard: default spot max_wait equals max_run (SageMaker's minimum)",
          derived_wait_min * 60 == MAX_RUNTIME, f"{derived_wait_min} min == {MAX_RUNTIME} s")
    check("guard: the old hardcoded 30-minute wait would have been REJECTED",
          30 * 60 < MAX_RUNTIME, "30 min < 72 h, which SageMaker refuses")
    short = 30
    check("guard: an explicitly short max_wait is a hard error",
          short * 60 < MAX_RUNTIME, f"{short} min x 60s < {MAX_RUNTIME}s -> SystemExit")

    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--skip_overfit", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)

    print("=" * 78)
    print(f"decider_v2 tests.py   device={device}  torch={torch.__version__}")
    print("=" * 78)
    test_packed_vs_separate(device)
    test_order_invariance(device)
    test_losses(device)
    test_decision_rule()
    test_launch_guards()
    if not args.skip_overfit:
        test_overfit(device, steps=args.steps)
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())