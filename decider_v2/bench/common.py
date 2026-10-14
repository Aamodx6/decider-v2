"""Shared plumbing for the JevBench and sysone-bench adapters.

Both harnesses need the same three things, so they live here once:

  1. load the model ONCE (a trained checkpoint, or the published reference), on the best device
     available -- the same `model.load_checkpoint` / `model.load_reference_bev_decider` /
     `serialize.Encoder` path `evaluate.py` uses, so a benchmark number is produced by the same
     code that produces the in-notebook test-split number;
  2. turn one benchmark question into options via `serialize.question_to_options`, which is the
     exact function training used, so the surface form of an option cannot drift between the two;
  3. batch a list of (state, question) through the network and return per-option probabilities.

`DeciderRunner.predict_questions` is the only entry point the two adapters need.  It returns a
`DecisionResult`-shaped dict so the adapters can mirror `jevbench/adapters/base.py`'s contract
(`ok`, `probs`, `probs_source`, `error`, `latency_s`) without importing jevbench, which lets this
module be unit-tested with no harness present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data import collate  # noqa: E402
from model import build_model, load_checkpoint, load_reference_bev_decider  # noqa: E402
from serialize import TASK_TYPES, Encoder, question_to_options  # noqa: E402


def _masked_softmax(logits, option_mask):
    # -1e30 rather than -inf, matching evaluate.py: softmax underflows it to exactly 0.0 and
    # log-softmax stays finite.
    return torch.softmax(logits.masked_fill(~option_mask, -1e30), dim=-1)


class DeciderRunner:
    """Loads a decider_v2 model once and answers (state, question) pairs."""

    def __init__(self, checkpoint=None, reference=None, device=None, batch_size=8,
                 max_state_tokens=1024, max_choice_tokens=64, noul_use_criteria=False):
        if bool(checkpoint) == bool(reference):
            raise ValueError("pass exactly one of checkpoint= or reference=")
        self.checkpoint = checkpoint
        self.reference = reference
        self.batch_size = int(batch_size)
        self.noul_use_criteria = bool(noul_use_criteria)
        self.network = None
        self.pad_id = 0
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ loading
    def load(self):
        if self.network is not None:
            return self
        from transformers import AutoTokenizer
        if self.reference:
            network, meta, snapshot = load_reference_bev_decider(
                self.reference, device=self.device)
            tokenizer = AutoTokenizer.from_pretrained(snapshot)
            encoder = Encoder(tokenizer, meta["resolved_max_state_tokens"],
                              meta["resolved_max_choice_tokens"], mode="bev", pool="bev")
            self.provenance = {"source": "reference", "model": self.reference, **{
                k: v for k, v in meta.items() if k in ("resolved_max_state_tokens",
                                                       "resolved_max_choice_tokens", "revision")}}
        else:
            args = json.load(open(os.path.join(self.checkpoint, "meta.json")))["args"]
            network = build_model(argparse.Namespace(**args))
            load_checkpoint(network, self.checkpoint)
            network.to(self.device).eval()
            tokenizer = AutoTokenizer.from_pretrained(args["backbone_name"])
            encoder = Encoder(tokenizer, args["max_state_tokens"], args["max_choice_tokens"],
                              mode=args.get("mode", "bev"), pool=args.get("pool", "bev"),
                              noul_use_criteria=self.noul_use_criteria)
            self.provenance = {"source": "checkpoint", "checkpoint": self.checkpoint,
                               "backbone_name": args["backbone_name"],
                               "head": args.get("head"), "lora_r": args.get("lora_r"),
                               "num_layers": args.get("num_layers")}
        self.network = network.eval()
        self.pad_id = tokenizer.pad_token_id or 0
        self.encoder = encoder
        return self

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def predict_questions(self, items, progress_every=0):
        """items: list of (state, question) -> list of dicts with keys/options/probs.

        `question` is the harness's own question object; `question_to_options` turns it into the
        exact option strings training used, so a benchmark question is surfaced to the model
        identically to a training row.
        """
        self.load()
        encoded, keysets = [], []
        for state, question in items:
            options, keys, _label, ranks = question_to_options(
                question, noul_use_criteria=self.noul_use_criteria)
            ex = self.encoder.encode(state, question["instructions"], options,
                                     question["type"], ranks=ranks)
            ex["keys"] = keys
            ex["options"] = options
            ex["label"] = -1
            ex["domain"] = ""
            ex["idx"] = len(encoded)
            encoded.append(ex)
            keysets.append(keys)

        amp = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        results = [None] * len(encoded)
        for start in range(0, len(encoded), self.batch_size):
            chunk = encoded[start:start + self.batch_size]
            if progress_every and start and start % progress_every == 0:
                print(f"  {start}/{len(encoded)} decisions", flush=True)
            batch = collate(chunk, self.pad_id)
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(self.device.type, dtype=amp, enabled=amp != torch.float32):
                logits = self.network(**batch).float()
            probs = _masked_softmax(logits, batch["option_mask"]).cpu()
            for i in range(len(chunk)):
                k = int(batch["option_mask"][i].sum())
                results[start + i] = {
                    "keys": keysets[start + i],
                    "options": chunk[i]["options"],
                    "probs": [float(x) for x in probs[i, :k]],
                }
        return results

    def run_one(self, state, question):
        """One decision, timed, in the shape jevbench/adapters/base.py::DecisionResult uses."""
        t0 = time.perf_counter()
        try:
            out = self.predict_questions([(state, question)])[0]
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "latency_s": time.perf_counter() - t0}
        out.update({"ok": True, "error": None, "latency_s": time.perf_counter() - t0,
                    "probs_source": "native", "model": self.reference or self.checkpoint,
                    "task_type": question["type"]})
        return out


__all__ = ["DeciderRunner", "TASK_TYPES", "question_to_options"]