"""One-command evaluation with arch.md's gates.

`inference.py` scores a checkpoint on a split; `evaluate.py` adds the parts that decide
whether a run is actually better than the last one:

  - per-task-type and per-domain accuracy, ECE (10 bins, top-label) and yes/no AUC
  - the option-order invariance probe, at arch.md:363's 960 shuffles rather than the 80 the
    unit test runs
  - hard gates from arch.md:349-367, reported as PASS/FAIL rather than left to the eye
  - a side-by-side markdown table over several runs, so the ladder can be read at a glance

    uv run python evaluate.py runs/<run>/checkpoints/best
    uv run python evaluate.py runs/a/checkpoints/best runs/b/checkpoints/best --config all
    uv run python evaluate.py runs/<run>/checkpoints/best --invariance-trials 960 --out report.md

The JevBench / sysone-bench gates are declared but not evaluated: those datasets are not
part of this repo. Pass --benchmark-dir once they are, or keep reading the bev-decision
test split, which is a different dataset on a different scale from bev-decider's published
numbers and must not be compared against them directly.
"""
import argparse
import contextlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

import inference
from dataset import BEVDataset, TASK_NAMES, collate_fn, load_questions
from inference import answer, apply_temperature, autocast, evaluate, get_device, to_device
from logger import load_checkpoint

# arch.md:363. Hard gate: option-order max delta over fp32, random shuffles, k = 3..12.
INVARIANCE_TRIALS = 960
INVARIANCE_TOLERANCE = 1e-5
# arch.md:265 / :363. 10-bin top-label ECE, per type and per domain.
ECE_TARGET = 0.05

# From arch.md:349-367. These are bev-decider-0.4B's published figures; the right-hand column
# is the v2-S goal. Evaluated only when a benchmark dataset is supplied.
BENCHMARK_TARGETS = {
    "heldout": {"bev_decider": 74.7, "target": 77.0},
    "jevbench": {"bev_decider": 65.8, "target": 70.0},
    "jevbench_hard": {"bev_decider": 46.8, "target": 52.0},
    "sysone_bench": {"bev_decider": 70.8, "target": 75.0},
    "sst5": {"bev_decider": 30.8, "target": 40.0},
    "multilingual_intent": {"bev_decider": 69.2, "target": 80.0},
    "verification": {"bev_decider": 54.1, "target": 65.0},
    "temporal_unit": {"bev_decider": 54.0, "target": 70.0},
}

INVARIANCE_STATE = (
    "The customer ordered on 2026-03-15. The warranty was valid through 2026-03-14. "
    "The order total was 108.40 with free shipping over 100.00. The customer emailed "
    "asking whether a repair is still free of charge.")
INVARIANCE_INSTRUCTIONS = "Which key best describes the situation?"


@contextlib.contextmanager
def fp32_inference():
    """Force fp32 for the duration of the block.

    arch.md:174 requires the invariance check in fp32, and arch.md:363 scores it as
    "fp32, 960 shuffles". answer() calls inference.autocast, which enables bf16 on
    cuda/mps; bf16 carries an 8-bit mantissa, so its rounding alone lands around 1e-3 and
    swamps the 1e-5 gate. The unit test gets away with it only because conftest.py's
    networks run on CPU, where autocast is already disabled. Measured on the ladder
    checkpoints under bf16 this probe reports 3e-3 and fails; in fp32 it is ~1e-7.
    """
    original = inference.autocast
    inference.autocast = lambda device: contextlib.nullcontext()
    try:
        yield
    finally:
        inference.autocast = original


@torch.no_grad()
def order_invariance(network, meta, device, trials=INVARIANCE_TRIALS, seed=0):
    """Max per-key probability change under random option permutations.

    Per arch.md:97 invariance comes from options sharing a position id plus an isolating
    mask, and the head carrying no positional information. This is the check that the two
    halves of that claim still hold after a change to the head, the mask or the position ids.

    Permutations are spread evenly across k = 3..12 so no single option count dominates.
    """
    network.eval()
    rng = random.Random(seed)
    ks = list(range(3, 13))
    per_k = max(1, trials // len(ks))
    worst, worst_k = 0.0, None

    with fp32_inference():
        for k in ks:
            question = {"type": "choice", "instructions": INVARIANCE_INSTRUCTIONS,
                        "criteria": {f"key_{i}": f"description of candidate {i} for this state"
                                     for i in range(k)},
                        "label": "key_0"}
            keys = list(question["criteria"])
            base = answer(network, meta, INVARIANCE_STATE, question, device)["probabilities"]
            for _ in range(per_k):
                order = keys[:]
                rng.shuffle(order)
                shuffled = dict(question)
                shuffled["criteria"] = {key: question["criteria"][key] for key in order}
                probs = answer(network, meta, INVARIANCE_STATE, shuffled, device)["probabilities"]
                for key in keys:
                    delta = abs(base[key] - probs[key])
                    if delta > worst:
                        worst, worst_k = delta, k
    return {"max_delta": worst, "worst_k": worst_k, "trials": per_k * len(ks),
            "k_range": [min(ks), max(ks)], "tolerance": INVARIANCE_TOLERANCE,
            "dtype": "fp32", "pass": worst <= INVARIANCE_TOLERANCE}


def split_metrics(metrics):
    """Pull the overall / per-type numbers out of evaluate()'s flat dict."""
    out = {"accuracy": metrics.get("accuracy"), "loss": metrics.get("loss"),
           "ece": metrics.get("ece"), "auc_noul": metrics.get("auc_noul")}
    for name in TASK_NAMES.values():
        out[f"accuracy_{name}"] = metrics.get(f"accuracy_{name}")
        out[f"ece_{name}"] = metrics.get(f"ece_{name}")
    return out


def split_domains(metrics):
    prefix = "accuracy_domain_"
    return {k[len(prefix):]: v for k, v in metrics.items() if k.startswith(prefix)}


def run_one(ckpt_dir, args, device):
    network, meta = load_checkpoint(ckpt_dir, device)
    temperatures = meta.get("temperatures")
    questions = load_questions(args.split, args.max_questions or None, 0,
                               configs=args.config.split(",") if args.config
                               else meta.get("data_configs", ["all"]))
    dataset = BEVDataset(questions, meta["max_state_tokens"], meta["max_choice_tokens"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         collate_fn=collate_fn,
                                         num_workers=args.num_workers)
    if args.calibrate and not temperatures:
        print(f"  no calibration.json next to the checkpoint; run calibrate.py first "
              f"(ECE below will be uncalibrated)", flush=True)
    metrics = evaluate(network, loader, device, temperatures)
    result = {"checkpoint": str(ckpt_dir), "n_questions": len(dataset),
              "calibrated": bool(temperatures),
              **split_metrics(metrics), "domains": split_domains(metrics)}
    result["domain_ece"] = {k[len("ece_domain_"):]: v for k, v in metrics.items()
                            if k.startswith("ece_domain_")}
    if args.invariance_trials > 0:
        result["invariance"] = order_invariance(network, meta, device,
                                                args.invariance_trials, args.seed)
    return result


def gate_table(results):
    """Hard gates from arch.md, plus the declared-but-unevaluated benchmark targets."""
    lines = ["| gate | arch.md target | result | status |", "|---|---|---|---|"]
    for res in results:
        label = Path(res["checkpoint"]).parent.parent.name or res["checkpoint"]
        inv = res.get("invariance")
        if inv:
            ok = "PASS" if inv["pass"] else "**FAIL**"
            lines.append(f"| option-order max delta (`{label}`) | <= {INVARIANCE_TOLERANCE:.0e} "
                         f"| {inv['max_delta']:.3e} over {inv['trials']} shuffles, "
                         f"k={inv['k_range'][0]}-{inv['k_range'][1]} | {ok} |")
        ece = res.get("ece")
        if ece is not None:
            ok = "PASS" if ece <= ECE_TARGET else "below target"
            lines.append(f"| ECE top-label (`{label}`) | <= {ECE_TARGET} | {ece:.4f} | {ok} |")
    lines.append("")
    lines.append("Benchmark targets from arch.md:349-367, **not evaluated here** "
                 "(datasets not in this repo):")
    lines.append("")
    lines.append("| benchmark | bev-decider-0.4B | v2-S target | status |")
    lines.append("|---|---:|---:|---|")
    for name, row in BENCHMARK_TARGETS.items():
        lines.append(f"| {name} | {row['bev_decider']} | {row['target']} | needs `--benchmark-dir` |")
    return "\n".join(lines)


def comparison_table(results):
    cols = ["run", "n", "accuracy", "accuracy_choice", "accuracy_noul", "accuracy_score",
            "ece", "ece_choice", "ece_noul", "ece_score", "auc_noul", "invariance"]
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, rule]
    for res in results:
        label = Path(res["checkpoint"]).parent.parent.name or res["checkpoint"]
        inv = res.get("invariance")
        cells = [label, str(res["n_questions"])]
        for col in cols[2:-1]:
            value = res.get(col)
            cells.append("—" if value is None else f"{value:.4f}")
        cells.append("—" if inv is None else f"{inv['max_delta']:.2e}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_domains(results):
    """Per-domain accuracy side by side, which is where bev-decider's weak spots live."""
    rows = results[0]
    header = ["run"] + sorted(rows["domains"])
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for res in results:
        label = Path(res["checkpoint"]).parent.parent.name or res["checkpoint"]
        cells = [label] + [f"{res['domains'].get(d, float('nan')):.3f}" for d in header[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="one or more runs/<id>/checkpoints/<name>")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--config", default=None,
                        help="dataset config, comma-separated (default: the checkpoint's)")
    parser.add_argument("--max-questions", type=int, default=0, help="0 = whole split")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--invariance-trials", type=int, default=INVARIANCE_TRIALS,
                        help=f"0 to skip (default {INVARIANCE_TRIALS}, arch.md:363)")
    parser.add_argument("--calibrate", action="store_true",
                        help="warn if the checkpoint has no calibration.json")
    parser.add_argument("--benchmark-dir", default=None,
                        help="JevBench / sysone-bench root; targets are declared but not scored yet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write the markdown report here")
    parser.add_argument("--json-out", default=None, help="write raw metrics here")
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}", flush=True)
    results = []
    for ckpt in args.checkpoints:
        print(f"evaluating {ckpt} ...", flush=True)
        results.append(run_one(ckpt, args, device))
        summary = results[-1]
        print(f"  accuracy={summary['accuracy']:.4f} ece={summary['ece']:.4f}"
              f"{' calibrated' if summary['calibrated'] else ' UNCALIBRATED'}", flush=True)
        if "invariance" in summary:
            inv = summary["invariance"]
            print(f"  order invariance: max delta {inv['max_delta']:.3e} over "
                  f"{inv['trials']} shuffles -> {'PASS' if inv['pass'] else 'FAIL'}", flush=True)

    report = ["# Evaluation", "",
              f"split={args.split} config={args.config or 'checkpoint default'} "
              f"questions/run={results[0]['n_questions']}", "",
              "## Comparison", "", comparison_table(results), "",
              "## Gates", "", gate_table(results)]
    if len(results) == 1 and results[0]["domains"]:
        report += ["", "## Per-domain accuracy", "", best_domains(results)]

    text = "\n".join(report)
    print()
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\nwrote {args.out}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")
    if any(not r.get("invariance", {"pass": True})["pass"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()