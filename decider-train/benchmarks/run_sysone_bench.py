"""sysone-bench adapter for decider_v2.

Same shape as `run_jevbench.py` and the same rule: sysone-bench grades sysone-bench.  Its own
`benchmark/contracts.validate_answers` decides whether an answer is legal, and its own
`benchmark/metrics.summarize_predictions` computes every reported number, including the
calibration-split-only temperature fits.  This file decodes cases, runs the model, and projects each
answer onto the exact field set the contract declares:

  choice -> {"type","choice","probabilities","confidence"}
  noul   -> {"type","noul"}                       (P(true))
  score  -> {"type","score","confidence"}         (a level in [0, max_score])

Mapping notes that are easy to get wrong and are therefore asserted rather than assumed:

  * a case's `state` is a JSON OBJECT (e.g. {"message": "..."}), not a string. `serialize.as_text`
    renders it, which is exactly what the training pipeline did for any non-string state.
  * a choice question's `expected` is the label STRING; a noul's is the INT 0/1; a score's is the
    INT level index. The answer contract wants the chosen label, P(true), and a level respectively.
  * the score answer is the probability-weighted EXPECTED LEVEL, because `decision_correct` rounds
    with `score_level` = floor(x + 0.5) and the richer ordinal metrics (MAE, RMSE, Spearman) want a
    continuous prediction. Using the argmax instead would silently change what MAE means.
  * choice `probabilities` must sum to 1 within 1e-6, so the distribution is renormalized over the
    real options only.

Data: `datasets/v2/manifest.v2.1.0.jsonl`, 1,190 cases = 952 evaluation + 238 calibration,
which is 1,240 evaluation DECISIONS (a case holds several questions).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from common import DeciderRunner  # noqa: E402

SYSONE_ROOT = os.environ.get("SYSONE_BENCH_ROOT", "")
MANIFEST_RELPATH = os.path.join("datasets", "v2", "manifest.v2.1.0.jsonl")


def find_repo(explicit=""):
    """Locate a sysone-bench checkout: explicit path, then $SYSONE_BENCH_ROOT, then ./ref/sysone-bench."""
    for cand in (explicit, SYSONE_ROOT,
                 os.path.join(os.getcwd(), "ref", "sysone-bench"),
                 os.path.join(os.path.dirname(_HERE), "..", "ref", "sysone-bench")):
        if cand and os.path.isfile(os.path.join(cand, MANIFEST_RELPATH)):
            return os.path.abspath(cand)
    return None


def load_cases(root):
    sys.path.insert(0, root)
    path = os.path.join(root, MANIFEST_RELPATH)
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def project_answer(question, out, expected):
    """Model output -> the exact answer object the sysone-bench contract validates."""
    kind = question["type"]
    probs = {k: float(p) for k, p in zip(out["keys"], out["probs"])}
    total = sum(probs.values())
    if not total > 0:
        raise ValueError("model produced a zero-sum distribution")
    if kind == "choice":
        labels = [str(x) for x in question["criteria"]]
        missing = [l for l in labels if l not in probs]
        if missing:
            raise KeyError(f"no probability for label(s) {missing}")
        norm = {l: probs[l] / total for l in labels}
        return {"type": "choice", "choice": max(norm, key=norm.get),
                "probabilities": norm, "confidence": max(norm.values())}
    if kind == "noul":
        # model keys are ("false", "true"); the contract wants P(true).
        return {"type": "noul", "noul": probs.get("true", 0.0) / total}
    levels = len(question["criteria"])
    weights = [probs.get(str(i), 0.0) / total for i in range(levels)]
    if sum(weights) <= 0:
        raise ValueError("model produced a zero-sum score distribution")
    value = sum(i * w for i, w in enumerate(weights))
    return {"type": "score", "score": value, "confidence": max(weights)}


def run(runner, cases, limit_cases=0, progress_every=0):
    from benchmark.contracts import validate_answers
    from benchmark.metrics import summarize_predictions

    if limit_cases:
        cases = cases[:limit_cases]

    # Flatten to one row per (case, question) so the model sees uniform batches, then regroup.
    flat = [(case, q) for case in cases for q in case["questions"]]
    print(f"running {len(flat)} decisions from {len(cases)} cases through decider_v2 ...", flush=True)
    outs = runner.predict_questions([(c["state"], q) for c, q in flat],
                                    progress_every=progress_every)

    rows, per_case = [], []
    pos = 0
    for case in cases:
        answers, failures = {}, []
        for q in case["questions"]:
            out = outs[pos]
            pos += 1
            try:
                answers[q["qid"]] = project_answer(q, out, case["expected"][q["qid"]])
            except Exception as e:  # noqa: BLE001
                failures.append(f"{q['qid']}: {type(e).__name__}: {e}")
        qids = [q["qid"] for q in case["questions"]]
        entry = {"case_id": case["case_id"], "suite_id": case["suite_id"], "split": case["split"],
                 "questions": case["questions"], "question_ids": qids,
                 "expected": case["expected"], "answers": answers,
                 "execution_phase": "benchmark", "phase": case["split"],
                 "failures": failures}
        per_case.append(entry)
        if not failures:
            # The contract check is sysone-bench's, and a row that fails it must not be graded.
            validate_answers(case["questions"], answers)
            rows.append(entry)
        else:
            print(f"  case {case['case_id']} skipped: {failures}")

    if not rows:
        raise RuntimeError("no case produced a contract-valid answer set")
    summary = summarize_predictions(rows)
    return summary, per_case


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--reference", default="")
    ap.add_argument("--sysone_bench_root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit_cases", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--progress_every", type=int, default=128)
    args = ap.parse_args(argv)

    root = find_repo(args.sysone_bench_root)
    if not root:
        print("NOT RUN -- sysone-bench checkout not found. Set SYSONE_BENCH_ROOT to a clone of "
              "https://github.com/instax-dutta/sysone-bench")
        return 2

    cases = load_cases(root)
    runner = DeciderRunner(checkpoint=args.checkpoint or None, reference=args.reference or None,
                           batch_size=args.batch_size)
    runner.load()
    print(f"model: {runner.provenance}")
    ev = [c for c in cases if c["split"] == "evaluation"]
    ca = [c for c in cases if c["split"] == "calibration"]
    print(f"cases: {len(cases)} ({len(ev)} evaluation + {len(ca)} calibration) from {root}")
    print(f"decisions: {sum(len(c['questions']) for c in ev)} evaluation + "
          f"{sum(len(c['questions']) for c in ca)} calibration")

    summary, per_case = run(runner, cases, limit_cases=args.limit_cases,
                            progress_every=args.progress_every)

    def g(d, *path, default=None):
        for p in path:
            if d is None or p not in d:
                return default
            d = d[p]
        return d

    print()
    print(f"{'metric':40s} {'value':>12s}")
    for label, path in (("choice accuracy", ("choice", "accuracy")),
                        ("choice macro F1", ("choice", "macro_f1")),
                        ("choice balanced accuracy", ("choice", "balanced_accuracy")),
                        ("choice Brier", ("choice", "brier")),
                        ("choice NLL", ("choice", "nll")),
                        ("choice ECE raw", ("choice", "ece_raw")),
                        ("choice ECE fitted", ("choice", "ece_fitted")),
                        ("noul accuracy", ("noul", "accuracy")),
                        ("noul Brier", ("noul", "brier")),
                        ("noul ECE raw", ("noul", "ece_raw")),
                        ("noul ECE fitted", ("noul", "ece_fitted")),
                        ("score nearest-level accuracy", ("score", "nearest_level_accuracy")),
                        ("score within-one accuracy", ("score", "within_one_accuracy")),
                        ("score MAE", ("score", "mae")),
                        ("score RMSE", ("score", "rmse")),
                        ("score Spearman", ("score", "spearman"))):
        v = g(summary, *path)
        print(f"  {label:38s} {v if v is not None else 'n/a':>12}")

    # Headline: the evaluation-split decision accuracy sysone-bench itself reports.
    n_eval = sum(len(r["questions"]) for r in per_case if r["split"] == "evaluation")
    correct, scored = 0, 0
    from benchmark.metrics import decision_correct
    for entry in per_case:
        if entry["split"] != "evaluation":
            continue
        for q in entry["questions"]:
            if q["qid"] not in entry["answers"]:
                continue
            scored += 1
            correct += bool(decision_correct(q["type"], entry["expected"][q["qid"]],
                                             entry["answers"][q["qid"]]))
    if scored:
        print(f"\n  evaluation decisions scored: {scored}/{n_eval}   "
              f"accuracy: {correct / scored * 100:.2f}")

    per_suite = {}
    for entry in per_case:
        if entry["split"] != "evaluation":
            continue
        slot = per_suite.setdefault(entry["suite_id"], [0, 0])
        for q in entry["questions"]:
            if q["qid"] not in entry["answers"]:
                continue
            slot[1] += 1
            slot[0] += bool(decision_correct(q["type"], entry["expected"][q["qid"]],
                                             entry["answers"][q["qid"]]))
    if per_suite:
        print("\nper-suite evaluation accuracy:")
        for suite, (c, n) in sorted(per_suite.items()):
            print(f"  {suite:22s} n={n:4d}  {c / n * 100:6.2f}")

    payload = {"benchmark": "sysone-bench", "repo": root, "cases": len(cases),
               "model": runner.provenance, "summary": summary,
               "per_case": [{k: e[k] for k in ("case_id", "suite_id", "split", "answers",
                                               "failures")} for e in per_case]}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())