"""One training script.  Runs identically on a laptop CPU (tiny config) and on SageMaker.

New in this layout (bev-train style repo):

  --config <yaml>         a configs/*.yaml file supplies EVERY training argument; precedence is
                          argparse default < yaml < command line, values are cast to the type of
                          the argparse default, and unknown yaml keys raise.  Recipes on the
                          ladder: configs/smoke.yaml, configs/baseline_replica.yaml,
                          configs/decider_v2.yaml, plus configs/ablations/ for the Tier-2 rungs.
  --checkpoint_dir        where `latest` / `best` / `final` are mirrored (default
                          /opt/ml/checkpoints when it exists, else ./outputs/checkpoints)
  --auto_resume 1         if <checkpoint_dir>/latest/optim.pt exists, resume step / optimiser /
                          scheduler and skip the micro-batches already consumed in the current epoch
  budget footgun fixed    with --max_steps set, epochs iterate until total_steps is reached;
                          --epochs is only a cap when --max_steps is unset. summary.json records
                          final_step and total_steps; a run that ends off-budget exits non-zero.

SageMaker surface
  channels    SM_CHANNEL_TRAIN / _VAL / _CALIB / _TEST  (directories of .parquet)
  outputs     SM_MODEL_DIR (checkpoints)  /  SM_OUTPUT_DATA_DIR (final model + metrics.jsonl)
  hyperparams passed as CLI flags by the estimator (a --config selects the recipe file)

Logging line format (regex-friendly for SageMaker metric_definitions):
    step=1000 loss=0.8123 grad_norm=0.41 lr_lora=1.9e-04 lr_head=9.5e-04 acc=0.71 sps=42.1
"""

import argparse
import json
import os
import random
import shlex
import sys
import time
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from dataset import (DecisionDataset, LengthBucketBatchSampler, collate, load_split_table,
                     state_token_lengths)
from logger import Logger, check_environment
from network import build_model, compute_loss, load_checkpoint
from tokenization import POOL_BEV, POOL_MEAN, POOL_OPT_END, TASK_NAMES, Encoder


# --------------------------------------------------------------------------- config files
# yaml section -> (dest mapping).  A leaf key that is missing from the map is UNKNOWN and raises.
def _config_leaf_map():
    return {
        None: {"seed": "seed"},
        "model": {"backbone_name": "backbone_name", "num_layers": "num_layers",
                  "random_init": "random_init", "attn_implementation": "attn_implementation",
                  "vocab_size": "vocab_size"},
        "lora": {"r": "lora_r", "alpha": "lora_alpha", "dropout": "lora_dropout",
                 "last_k_layers": "lora_last_k_layers", "include_mlp": "lora_include_mlp"},
        "head": {"type": "head", "new_dim": "head_new_dim", "layers": "head_layers",
                 "heads": "head_heads", "dropout": "head_dropout", "n_ctx": "n_ctx",
                 "use_ctx": "use_ctx", "use_rank": "use_rank", "use_null": "use_null",
                 "use_derived": "use_derived", "derived_dropout": "derived_dropout"},
        "loss": {"label_smoothing": "label_smoothing", "emd_lambda": "emd_lambda",
                 "type_weights": "type_weights"},
        "data": {"train_channel": "train_channel", "val_channel": "val_channel",
                 "test_channel": "test_channel", "max_state_tokens": "max_state_tokens",
                 "max_choice_tokens": "max_choice_tokens", "mode": "mode", "pool": "pool",
                 "noul_use_criteria": "noul_use_criteria", "shuffle_options": "shuffle_options",
                 "num_workers": "num_workers", "limit_train": "limit_train",
                 "limit_val": "limit_val"},
        "optim": {"batch_size": "batch_size", "grad_accum_steps": "grad_accum_steps",
                  "max_steps": "max_steps", "epochs": "epochs", "lr_lora": "lr_lora",
                  "lr_head": "lr_head", "weight_decay": "weight_decay",
                  "warmup_ratio": "warmup_ratio", "grad_clip": "grad_clip",
                  "gradient_checkpointing": "gradient_checkpointing",
                  "max_train_minutes": "max_train_minutes"},
        "logging": {"log_every": "log_every", "eval_every": "eval_every", "save_every": "save_every",
                    "model_dir": "model_dir", "output_dir": "output_dir",
                    "checkpoint_dir": "checkpoint_dir", "resume": "resume",
                    "auto_resume": "auto_resume", "selftest": "selftest", "env_check": "env_check"},
    }


def load_config(path):
    """-> {argparse dest: value}. Raises on unknown, empty or malformed config files."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"config {path}: expected a mapping of sections, got {type(cfg).__name__}")
    leaf_map = _config_leaf_map()
    out = {}
    top_level = leaf_map.get(None, {})
    for section, body in cfg.items():
        if not isinstance(body, dict) and section in top_level:
            out[top_level[section]] = body      # top-level scalar, e.g. `seed: 0`
            continue
        if section not in leaf_map:
            raise SystemExit(f"config {path}: unknown section {section!r} "
                             f"(valid: {sorted((str(k) for k in leaf_map))})")
        if body is None:
            continue
        if not isinstance(body, dict):
            raise SystemExit(f"config {path}: section {section!r} must be a mapping, "
                             f"got {type(body).__name__}")
        for leaf, val in body.items():
            dest = leaf_map[section].get(leaf)
            if dest is None:
                raise SystemExit(f"config {path}: unknown key {section}.{leaf} "
                                 f"(valid keys: {sorted(leaf_map[section])})")
            out[dest] = val
    if not out:
        raise SystemExit(f"config {path}: file did not define any training argument")
    return out


def _provided_dests(argv):
    """dests explicitly present on the command line (these beat the yaml)."""
    provided = set()
    for tok in argv:
        if tok.startswith("--"):
            provided.add(tok[2:].split("=", 1)[0].replace("-", "_"))
    return provided


def _cast(dest, value, default):
    """Cast a yaml value to the type the argparse default declares."""
    if value is None:
        return default
    if default is True or default is False:          # store_true-style flags
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    # config
    p.add_argument("--config", default="", help="a configs/*.yaml supplying every training argument")
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
    p.add_argument("--epochs", type=int, default=1,
                   help="epoch cap; only a cap when --max_steps is unset (otherwise the budget "
                        "decides the number of epochs)")
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
    p.add_argument("--checkpoint_dir",
                   default=default_checkpoint_dir(),
                   help="checkpoint mirror dir (/opt/ml/checkpoints on SageMaker; "
                        "./outputs/checkpoints locally)")
    p.add_argument("--resume", default="")
    p.add_argument("--auto_resume", type=int, default=0,
                   help="resume from <checkpoint_dir>/latest/optim.pt if it exists "
                        "(step + optimiser + scheduler + the consumed micro-batches)")
    p.add_argument("--max_train_minutes", type=float, default=-1)
    p.add_argument("--stop_at_step", type=int, default=0,
                   help="stop cleanly at this step and write a resumable 'latest' checkpoint, "
                        "while keeping --max_steps and therefore the LR schedule intact (0 = "
                        "off). A preemption drill: the restarted job must replay this run "
                        "exactly, which is not true of a shorter --max_steps, because that "
                        "also shortens the cosine schedule the first leg was trained on.")
    # --- preflight gates. Both default ON: a job that would produce numbers from the wrong library
    # versions, or from a broken mask, should fail in the first minute rather than after 12k steps.
    p.add_argument("--selftest", type=int, default=1,
                   help="run the packed-vs-separate and order-invariance gates (tiny config, "
                        "offline, ~10s) at job start; abort the job if any check fails")
    p.add_argument("--env_check", type=int, default=1,
                   help="verify requirements.txt pins; pip-install them if the container does not "
                        "already satisfy them, and abort the job if they still do not match")
    p.add_argument("--sagemaker_args", default="",
                   help="optional string of serialized CLI args from SageMaker")

    if argv is None:
        argv = sys.argv[1:]

    # if SageMaker embedded a serialized arg string, splice it in front
    _args = p.parse_known_args(argv)[0]
    if getattr(_args, "sagemaker_args", ""):
        embedded = shlex.split(_args.sagemaker_args)
        expanded_argv = []
        skip_next = False
        for a in argv:
            if skip_next:
                skip_next = False
                continue
            if a == "--sagemaker_args":
                skip_next = True
                continue
            if a.startswith("--sagemaker_args="):
                continue
            expanded_argv.append(a)
        argv = embedded + expanded_argv

    config_path = getattr(_args, "config", "")
    args = p.parse_args(argv)
    if config_path:
        overrides = load_config(config_path)
        defaults = {a.dest: a.default for a in p._actions if a.dest !=
                    "help" and a.dest != "sagemaker_args"}
        provided = _provided_dests(argv)
        unknown = []
        for dest, val in overrides.items():
            if dest not in defaults:
                unknown.append(dest)
        if unknown:
            raise SystemExit(f"config {config_path}: keys not accepted by train.py: {unknown}")
        for dest, val in overrides.items():
            if dest not in provided:
                setattr(args, dest, _cast(dest, val, defaults[dest]))
    return args


def default_checkpoint_dir():
    if os.path.isdir("/opt/ml/checkpoints"):
        return "/opt/ml/checkpoints"
    return os.environ.get("SM_CHECKPOINT_DIR", "./outputs/checkpoints")


def _arg_defaults(parser):
    for a in parser._actions:
        yield a.dest, a.default


# --------------------------------------------------------------------------- preflight
SELFTEST_GATE = 1e-5


def _byte_tokenizer():
    """Deterministic byte-level tokenizer: no download, works offline.

    Identical to the fixture in tests/test_core.py, so the two selftest paths measure the same
    thing on the same bytes.
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
            return 0

        def convert_tokens_to_ids(self, t):
            ids = self.encode(t)
            return ids[0] if ids else 65

        def __len__(self):
            return 256
    return ByteTok()


@torch.no_grad()
def _option_hidden(net, batch, device):
    """[B, K, H] backbone hidden states at each option's read position."""
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    hidden = net.backbone(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                          position_ids=b["position_ids"], use_cache=False).last_hidden_state.float()
    bidx = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[bidx[:, None], b["option_read_idx"]]


def _run_compact_gates(device):
    """The two structural gates, self-contained, for a source pack that ships no tests/ directory.

    This is the fallback used inside the SageMaker container (the allowlist ships modules, not the
    test suite).  Same two guarantees as tests/test_core.py, same byte tokenizer, same 1e-5
    thresholds, ~4s on CPU: the packed-vs-separate equivalence (plus option isolation) under the
    [B,1,L,L] additive mask with shared position ids, and option-order invariance.  Returns [] on
    success, a list of failure names otherwise.
    """
    failures = []

    def gate(name, worst, cap=SELFTEST_GATE):
        ok = worst <= cap
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  max delta {worst:.3e} (gate {cap:.0e})")
        if not ok:
            failures.append(name)

    tok = _byte_tokenizer()
    enc = Encoder(tok, max_state_tokens=512, max_choice_tokens=64, mode="bev", pool=POOL_BEV)

    def tiny(seed):
        ns = argparse.Namespace(backbone_name="tiny", num_layers=2, random_init=1,
                                attn_implementation="sdpa", lora_r=4, lora_alpha=8,
                                lora_dropout=0.0, lora_last_k_layers=2, lora_include_mlp=1,
                                head="v2", head_new_dim=64, head_layers=2, head_heads=4,
                                head_dropout=0.0, n_ctx=4, use_ctx=1, use_rank=1, use_null=0,
                                pool=POOL_BEV)
        torch.manual_seed(seed)
        return build_model(ns).to(device).eval()

    net = tiny(0)
    g = torch.Generator().manual_seed(7)
    prefix = " ".join(chr(97 + int(v) % 26) for v in torch.randint(0, 26, (60,), generator=g))

    # gate 1: packed-vs-separate, over the option read positions
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
    gate("packed-vs-separate option hidden states", worst)

    # gate 1b: option isolation -- dropping an option must not move the others
    net2 = tiny(1)
    opts = ["alpha beta gamma", "delta epsilon", "zeta eta theta iota"]
    full_ex = enc.encode(prefix, "pick one", opts, "choice")
    hf = _option_hidden(net2, collate([full_ex], tok.pad_token_id), device)[0]
    keep = [0, 2]
    red = enc.encode(prefix, "pick one", [opts[i] for i in keep], "choice")
    hr = _option_hidden(net2, collate([red], tok.pad_token_id), device)[0]
    drift = max((hf[i] - hr[j]).abs().max().item() for j, i in enumerate(keep))
    gate("option isolation (dropping an option does not move the others)", drift)

    # gate 2: option-order invariance on the probabilities
    net3 = tiny(3)
    worst_p = 0.0
    k = 8
    for seed_off in range(4):
        opts = [f"option {i}: " + " ".join(["w"] * (2 + i)) for i in range(k)]
        ex = enc.encode("some state text here", "which one", opts, "choice")
        ex |= {"label": 0, "domain": "d", "idx": 0}
        b = {kk: (v.to(device) if torch.is_tensor(v) else v)
             for kk, v in collate([ex], tok.pad_token_id).items()}
        with torch.no_grad():
            base = net3(**b).float().masked_fill(~b["option_mask"], -1e9)
        p0 = torch.softmax(base, -1)[0]
        perm = list(range(k))
        random.Random(seed_off).shuffle(perm)
        alt_ex = enc.encode("some state text here", "which one", [opts[i] for i in perm], "choice")
        alt_ex |= {"label": perm.index(0), "domain": "d", "idx": 0}
        ab = {kk: (v.to(device) if torch.is_tensor(v) else v)
              for kk, v in collate([alt_ex], tok.pad_token_id).items()}
        with torch.no_grad():
            alt = net3(**ab).float().masked_fill(~ab["option_mask"], -1e9)
        pa = torch.softmax(alt, -1)[0]
        back = torch.zeros_like(p0)
        back[torch.tensor(perm, device=p0.device)] = pa
        worst_p = max(worst_p, (back - p0).abs().max().item())
    gate("order invariance mode=bev pool=bev", worst_p)

    return failures


def run_selftest(device):
    """The two structural gates on a tiny 2-layer random-init model.

    In a full checkout this runs the complete battery from tests/test_core.py.  In a stripped source
    pack (the SageMaker allowlist ships no tests/ directory) it runs the equivalent gates built into
    this module -- same two guarantees, same 1e-5 thresholds, ~4s on CPU.  Returns [] on success, a
    list of failure names otherwise.
    """
    print("\n" + "=" * 78)
    print("SELFTEST: packed-vs-separate equivalence + order invariance (tiny 2-layer config)")
    print("=" * 78)
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.path.join(here, "tests")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import test_core as _tests  # full battery: 4 gates + the launch-guard checks
    except Exception:  # noqa: BLE001 -- any import trouble (e.g. no tests/ shipped) falls back
        return _run_compact_gates(device)
    torch.manual_seed(0)
    _tests.FAILURES.clear()
    _tests.test_packed_vs_separate(device)
    _tests.test_order_invariance(device)
    return list(_tests.FAILURES)


def run_preflight(args, device, main_rank):
    """Version pins, then the two structural gates. Returns False if the job must not proceed.

    Both run on rank 0 only (they are tiny and deterministic, so every rank would compute the same
    answer); a failure is broadcast so that under torchrun every rank exits rather than rank 0
    aborting while the others sit in a collective.
    """
    ok = True
    if args.env_check:
        ok &= check_environment()
        if not ok:
            print("ABORT: package versions do not match requirements.txt. Every published number "
                  "was produced with these pins; running anyway would make the results "
                  "uncomparable. Re-run with --env_check 0 to override deliberately.")
    else:
        print("ENV: --env_check 0 -- versions NOT verified for this job.")

    if args.selftest:
        failures = run_selftest(device)
        if not failures:
            print("SELFTEST: PASS")
        else:
            ok = False
            print(f"ABORT: selftest FAILED: {failures}. The 4-D mask / position-id guarantee this "
                  f"project is built on does not hold in this container, so any accuracy number "
                  f"from this job would be uninterpretable.")
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
        from tokenization import resize_and_init_embeddings

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
        dev_str = "cpu"
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            dev_str = f"{props.name}  {props.total_memory / 2**30:.2f} GiB"
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
        print(f"      epochs: {'budget-driven' if args.max_steps > 0 else args.epochs} "
              f"({'--epochs is a cap when --max_steps is unset' if args.max_steps <= 0 else '--epochs ignored: epochs iterate until max_steps is reached'})"
              f"  auto_resume={args.auto_resume}")
        print(f"      checkpoints: model_dir={args.model_dir} checkpoint_dir={args.checkpoint_dir or '(same as model_dir)'}")
        print(f"  device: {dev_str}")
        print("=" * 78, flush=True)

    tw = {}
    if args.type_weights:
        for part in args.type_weights.split(","):
            k, v = part.split(":")
            tw[int(k)] = float(v)

    logger = Logger(model_dir=args.model_dir, output_dir=args.output_dir,
                    checkpoint_dir=args.checkpoint_dir, config=vars(args), main_rank=main_rank)

    step, best_acc = 0, -1.0
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
    if args.auto_resume:
        auto = os.path.join(args.checkpoint_dir or args.model_dir, "latest")
        if os.path.exists(os.path.join(auto, "optim.pt")):
            state = torch.load(os.path.join(auto, "optim.pt"), map_location="cpu", weights_only=False)
            if state.get("step", 0) < step:
                raise SystemExit(f"auto_resume: <checkpoint_dir>/latest is at step {state.get('step', 0)}, "
                                 f"BEHIND the explicit --resume point {step}; refusing to go backwards")
            optimizer.load_state_dict(state["optim"])
            scheduler.load_state_dict(state["sched"])
            step = state["step"]
            best_acc = state.get("best_acc", -1.0)
            start_step = step
            load_checkpoint(model.module if world > 1 else model, auto)
            if main_rank:
                print(f"auto-resume: continuing from {auto} at step {step} "
                      f"best_val_acc={best_acc:.4f}")

    if main_rank:
        print(f"trainable params: backbone={sum(q.numel() for q in backbone_params):,} "
              f"head={sum(q.numel() for q in head_params):,} "
              f"total={sum(q.numel() for q in backbone_params) + sum(q.numel() for q in head_params):,}")
        print(f"train questions={len(train_tbl):,} val questions={len(val_tbl):,} "
              f"batch={args.batch_size}x{args.grad_accum_steps} steps/epoch={steps_per_epoch} "
              f"total_steps={total_steps} device={device}")

    meta = {"args": vars(args), "step": step, "pool": args.pool, "mode": args.mode,
            "head": args.head, "num_layers": args.num_layers,
            "max_state_tokens": args.max_state_tokens, "max_choice_tokens": args.max_choice_tokens}

    def save(name, merge=False):
        if not main_rank:
            return
        logger.save_checkpoint(model.module if world > 1 else model, name, step,
                               meta={"pool": args.pool, "mode": args.mode, "head": args.head,
                                     "num_layers": args.num_layers, "step": step},
                               merge=merge, optim=optimizer, scheduler=scheduler, best_acc=best_acc)

    # budget-driven epochs: with --max_steps set, iterate epochs until total_steps; --epochs is then
    # NOT a cap. Only when --max_steps is unset do --epochs bound the loop.
    unbounded_epochs = args.max_steps > 0

    model.train()
    t0 = time.time()
    run_start = t0
    running, running_n, running_correct = 0.0, 0, 0
    done = False
    deadline = None if args.max_train_minutes <= 0 else time.time() + args.max_train_minutes * 60

    epoch = start_step // steps_per_epoch
    epoch_max = 10 ** 9 if unbounded_epochs else max(args.epochs, epoch + args.epochs)
    micros_per_epoch = steps_per_epoch * args.grad_accum_steps
    micros_to_skip = (start_step % steps_per_epoch) * args.grad_accum_steps if args.auto_resume else 0
    optimizer.zero_grad(set_to_none=True)

    while not done and epoch < epoch_max:
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if hasattr(train_set, "set_epoch"):
            train_set.set_epoch(epoch)
        for micro, batch in enumerate(train_loader, 1):
            if micro > micros_per_epoch:
                break
            if step >= total_steps:
                done = True
                break
            if epoch == start_step // steps_per_epoch and micro <= micros_to_skip:
                continue  # auto_resume: skip micro-batches consumed before the resume point
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
                logger.log(step=step, loss=running / max(running_n, 1), grad_norm=float(gn),
                           acc=running_correct / max(running_n, 1), lr_lora=args.lr_lora,
                           lr_head=args.lr_head, sps=sps, peak_gib=round(peak, 2))
                running, running_correct, running_n, t0 = 0.0, 0, 0, time.time()
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
                    logger.log(step=step, split="val", _stdout=False,
                               val_loss=m["loss"], val_acc=m["accuracy"],
                               **{k.replace("/", "_"): v for k, v in m.items() if k != "loss"})
                if m["accuracy"] > best_acc:
                    best_acc = m["accuracy"]
                    save("best")
            if args.stop_at_step > 0 and step >= args.stop_at_step:
                # Deliberate stop at a step boundary: save what auto_resume needs even when
                # stop_at_step is not a multiple of save_every, then leave the loop with
                # total_steps (and the loaded scheduler's horizon) unchanged.
                save("latest")
                if main_rank:
                    print(f"stopping early at step {step}: --stop_at_step reached "
                          f"(budget stays {total_steps})", flush=True)
                done = True
                break
            if deadline and time.time() > deadline:
                if main_rank:
                    print(f"stopping early at step {step}: max_train_minutes reached", flush=True)
                done = True
                break
        if not unbounded_epochs and step >= total_steps:
            done = True
        epoch += 1

    if step > start_step or not os.path.exists(os.path.join(args.model_dir, "final")):
        save("final", merge=False)
    if main_rank:
        m = evaluate(model.module if world > 1 else model, val_loader, device, amp_dtype, world, rank)
        if best_acc < 0:
            # No mid-training evaluation ran (the run is shorter than --eval_every, or it stopped
            # at --stop_at_step before the first one). 'best' would otherwise never exist, and
            # evaluate.py selects 'best'; the only evaluated weights are the final ones.
            best_acc = m["accuracy"]
            save("best")
        summary = {"final_step": step, "total_steps": total_steps, "best_val_acc": best_acc,
                   "val": m,
                   "config": {
                       "checkpoint_selection": "best_val_accuracy",
                       "checkpoint_selection_source": "val split only; test split never read",
                       "eval_every": args.eval_every,
                       "save_every": args.save_every,
                       "max_steps": args.max_steps,
                       "auto_resume": args.auto_resume,
                       "stop_at_step": args.stop_at_step,
                       "lr_schedule_steps": sched_total,
                       "lr_warmup_steps": warmup_steps,
                       "question_presentations": step * args.batch_size * args.grad_accum_steps,
                       "batch_size": args.batch_size,
                       "grad_accum_steps": args.grad_accum_steps,
                       "steps_per_epoch": steps_per_epoch,
                       "epochs": round(step / steps_per_epoch, 4),
                       "seed": args.seed,
                       "head": args.head,
                       "model_dir": args.model_dir,
                       "checkpoint_dir": args.checkpoint_dir,
                       "output_dir": args.output_dir,
                   }}
        logger.write_summary(summary)
        stopped_on_purpose = bool(args.stop_at_step > 0 and step >= args.stop_at_step)
        if args.max_steps > 0 and step != total_steps and not stopped_on_purpose:
            raise SystemExit(
                f"TRAINING BUDGET NOT MET: stopped at step {step} but --max_steps says {total_steps}. "
                f"The run is incomplete (checkpoints 'latest'/'best' exist, 'final' is at step {step}). "
                f"Re-run with --auto_resume 1 to continue from the checkpoint, or raise --epochs; "
                f"summary.json WAS written and records final_step={step} against total_steps={total_steps}."
            )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
