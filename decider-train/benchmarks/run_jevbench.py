"""JevBench adapter for decider-train.

Follows the shape of `jevbench/adapters/so1_decider.py`: an in-process adapter that loads a model
once, exposes `run(task)`, and returns a `DecisionResult`-shaped dict.  The difference is the
plumbing direction -- so1_decider drives a third-party library's `Decider`; here WE hold the model
and jevbench drives us, so this file is a self-contained runner rather than a class jevbench
instantiates.  Grading is still entirely jevbench's: `scoring.score_task` decides validity and
correctness and `summarize.summarize` computes the aggregates, per the repo rule that the metric is
never re-implemented.

Mapping (documented in ARCH_REVIEW.md §7):

  choice -> options are the criteria keys in the task's OWN label order; the probability map is
            keyed by the label string, because jevbench scores `argmax(probs) == task.labels[i]`.
  noul   -> labels are ["no", "yes"]; we emit P(false) and P(true) from the model's two options.
  score  -> labels are ["0".."k-1"]; the model's score options are already "i: <level text>", so the
            probability map is keyed by the level index and jevbench's `expected_value` does the
            ordinal arithmetic.

Data: `datasets/public/{easy,hard,original}.jsonl` = 48 + 111 + 72 = 231 tasks, split "public".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from common import DeciderRunner  # noqa: E402

JEVBENCH_ROOT = os.environ.get("JEVBENCH_ROOT", "")
SYSONE_ROOT = os.environ.get("SYSONE_BENCH_ROOT", "")

NOUL_LABELS = ["no", "yes"]


def find_repo(explicit=""):
    """Locate a jevbench checkout: explicit path, then $JEVBENCH_ROOT, then ./ref/jevbench."""
    for cand in (explicit, JEVBENCH_ROOT,
                 os.path.join(os.getcwd(), "ref", "jevbench"),
                 os.path.join(os.path.dirname(_HERE), "..", "ref", "jevbench")):
        if cand and os.path.isdir(os.path.join(cand, "jevbench")):
            return os.path.abspath(cand)
    return None


def load_tasks(root):
    """231 public tasks via jevbench's own loader, so Task.validate() runs on every one."""
    sys.path.insert(0, root)
    from jevbench.tasks import load_jsonl
    tasks = []
    for tier in ("easy", "hard", "original"):
        tasks += load_jsonl(os.path.join(root, "datasets", "public", f"{tier}.jsonl"))
    return tasks


def probs_by_label(task, out):
    """Project the model's per-option probabilities onto jevbench's exact label set.

    `out["keys"]` is `question_to_options`' key list, which is the task's own label order for
    choice and "0".."k-1" for score, so this is a zip with no renaming except for noul, whose
    model keys are ("false", "true") and whose labels are ("no", "yes").
    """
    kind = task.question["type"]
    labels = [str(x) for x in task.labels]
    if kind == "noul":
        # question_to_options gives keys ["false", "true"]; jevbench wants {"no", "yes"}.
        p = dict(zip(out["keys"], out["probs"]))
        return {"no": p.get("false", 0.0), "yes": p.get("true", 0.0)}
    p = dict(zip(out["keys"], out["probs"]))
    missing = [l for l in labels if l not in p]
    if missing:
        raise KeyError(f"{task.id}: model produced no probability for label(s) {missing}")
    return {l: p[l] for l in labels}


def run(runner, tasks, limit=0, progress_every=0):
    from jevbench.scoring import score_task
    from jevbench.summarize import summarize

    if limit:
        tasks = tasks[:limit]

    # Batch the forward passes: one encode+forward per task, but grouped so the GPU sees batches.
    t_all = time.perf_counter()
    items = [(t.state, t.question) for t in tasks]
    print(f"running {len(items)} decisions through decider-train ...", flush=True)
    outs = runner.predict_questions(items, progress_every=progress_every)
    wall = time.perf_counter() - t_all

    records = []
    for task, out in zip(tasks, outs):
        res = {"task_id": task.id, "model": runner.reference or runner.checkpoint,
               "probs_source": "native", "latency_s": wall / max(len(tasks), 1),
               "ok": True, "error": None}
        try:
            probs = probs_by_label(task, out)
        except Exception as e:  # noqa: BLE001
            records.append({**res, "ok": False, "valid": False, "correct": False,
                            "error": f"{type(e).__name__}: {e}"})
            continue
        graded = score_task(probs, task)
        records.append({**res, "probs": graded.get("probs", probs), **graded})

    summary = summarize(tasks, records)
    return summary, records


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--reference", default="")
    ap.add_argument("--jevbench_root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--noul_use_criteria", type=int, default=0,
                    help="1 renders 'No: <desc>'/'Yes: <desc>'; 0 matches the TRAINING surface form")
    ap.add_argument("--progress_every", type=int, default=64)
    args = ap.parse_args(argv)

    root = find_repo(args.jevbench_root)
    if not root:
        print("NOT RUN -- jevbench checkout not found. Set JEVBENCH_ROOT to a clone of "
              "https://github.com/fstandhartinger/jevbench")
        return 2

    tasks = load_tasks(root)
    runner = DeciderRunner(checkpoint=args.checkpoint or None, reference=args.reference or None,
                           batch_size=args.batch_size, noul_use_criteria=bool(args.noul_use_criteria))
    runner.load()
    print(f"model: {runner.provenance}")
    print(f"tasks: {len(tasks)} from {root}")

    summary, records = run(runner, tasks, limit=args.limit, progress_every=args.progress_every)

    print()
    print(f"{'metric':34s} {'value':>12s}")
    # Some aggregates are legitimately None (e.g. price for a self-hosted run), so format through
    # `fmt` rather than assuming a number. `ece` is a dict from metrics.ece_top_label.
    fmt = lambda v: "n/a" if v is None else f"{v:>12}"
    for k in ("n_planned", "n_attempted", "n_scorable", "n_valid", "n_correct", "accuracy",
              "coverage", "schema_validity", "schema_validity_strict", "macro_accuracy",
              "brier_mean", "ordinal_mae", "price_per_1000_decisions_usd"):
        if k in summary:
            print(f"  {k:32s} {fmt(summary[k])}")
    ece = summary.get("ece")
    print(f"  {'ece':32s} "
          f"{fmt(ece.get('ece') if isinstance(ece, dict) else ece)}")
    print()
    print("per-family accuracy:")
    for fam, m in sorted(summary.get("per_family", {}).items()):
        acc = m.get("accuracy")
        print(f"  {fam:22s} n={m.get('n_scorable', 0):4d}  "
              f"{'n/a' if acc is None else f'{acc * 100:6.2f}'}")

    payload = {"benchmark": "jevbench", "repo": root, "tasks": len(tasks),
               "model": runner.provenance, "summary": summary,
               "per_task": [{k: r.get(k) for k in
                             ("task_id", "ok", "valid", "correct", "predicted", "probs", "error")}
                            for r in records]}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())