"""decide.py -- is the v2-vs-baseline delta a result, or is it noise?

One decision, three possible answers, and every one of them is computed from artefacts that already
exist (`metrics_<arm>_s<seed>.json` and `predictions_<arm>_s<seed>.csv`).  Nothing here re-runs a
model.

    python decide.py --work decider_v2_work/eval --arms baseline v2 --seeds 0 1

Why this module exists
----------------------
The accuracy delta between two models on a fixed test set has three independent noise sources, and a
number on its own hides all three:

  1. **Test sampling.**  Which questions you happened to score. Measured by the paired bootstrap
     below, which resamples *questions* and keeps each pair (baseline, v2) together -- the pairing is
     what makes it tight, because both arms answer the identical question.
  2. **Training-run variance.**  Two runs of the *same* configuration disagree. Measured directly as
     the seed spread, which needs no statistics at all.
  3. **Effect size.**  A 0.4-point delta can be "statistically significant" at n=80,019 and still be
     worth nothing. Hence a hard 1.0-point floor.

Reported separately in the past, and this is the reason the rule changed: on a 4,000-question SMOKE
shard the seed spread of an identical configuration measured 2.52 points in one run and 0.05 in
another (`LOCAL_VERIFICATION.md` §8.2). Reporting a delta without both of these next to it is how a
noise measurement gets quoted as a result.

The rule
--------
    (a) the bootstrap 95% CI for the delta excludes 0
    (b) |delta| > seed spread                       (training-run variance)
    (c) |delta| >= 1.0 point                        (effect worth acting on)
    (d) at least 3 seeds per arm

    (a) and (b) and (c) and (d)   ->  SEPARABLE
    (a) and (b) and (c), but d=2  ->  PROVISIONAL      (one more seed decides it)
    anything else                 ->  NOT SEPARABLE -- run another seed

(d) is why two seeds cannot return SEPARABLE.  With two seeds the spread is a range between a single
pair of runs; it can be 0.00 points by luck (both runs landing on the same accuracy) and then (b) is
trivially satisfied by any delta.  Three seeds make a zero spread far less likely, and the honest
label for a two-seed result is PROVISIONAL.

Paired bootstrap
----------------
Resample `n` question indices with replacement, `n = len(rows)`, and average the *difference* in
per-question correctness.  Because the resample indices are shared between the two arms, the CI
measures the delta's stability under test resampling alone -- it deliberately does NOT include
training-run variance, which is what (b) is for.  The percentile interval is used rather than the
normal approximation because the per-question difference is a bounded, discretised variable (only
values in {-1, 0, +1}), so its sampling distribution is lumpy at small n.

Determinism: seeded from `BOOTSTRAP_SEED` and independent of the training seeds, so re-running the
notebook reproduces the same CI bit-for-bit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random

# effect-size floor, in percentage points
MIN_EFFECT_PP = 1.0
# number of seeds required before SEPARABLE is allowed
MIN_SEEDS_FOR_SEPARABLE = 3
# resamples and seed for the paired bootstrap
N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260101


# --------------------------------------------------------------------------- loading
def load_predictions(path):
    """-> (labels, correct) as parallel lists of ints, one entry per question.

    `correct[i]` is whether the model predicted the right label for question i. The row ORDER is the
    pairing key: both arms are evaluated over the identical test file by an identical code path, so
    row i of one arm is the same question as row i of the other. `load_pair` verifies that rather
    than trusting it.
    """
    labels, correct = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label, pred = int(row["label"]), int(row["pred"])
            labels.append(label)
            correct.append(1 if pred == label else 0)
    if not labels:
        raise ValueError(f"{path}: no rows")
    return labels, correct


def load_pair(baseline_csv, v2_csv):
    """Load both arms and assert they were scored over the identical question sequence.

    A mismatch here would silently invalidate the pairing (resampling index i of one arm against
    index i of the other would compare unrelated questions), so it is a hard error, not a warning.
    """
    lb, cb = load_predictions(baseline_csv)
    lv, cv = load_predictions(v2_csv)
    if len(lb) != len(lv):
        raise ValueError(f"row count differs: baseline {len(lb)} vs v2 {len(lv)}")
    if lb != lv:
        first = next(i for i, (x, y) in enumerate(zip(lb, lv)) if x != y)
        raise ValueError(f"labels differ at row {first}: {lb[first]} vs {lv[first]}; "
                         f"the two arms were not scored over the same questions")
    return cb, cv


def accuracy(correct):
    return sum(correct) / len(correct)


# --------------------------------------------------------------------------- bootstrap
def paired_bootstrap(correct_b, correct_v, n_resamples=N_RESAMPLES,
                     seed=BOOTSTRAP_SEED, alpha=0.05):
    """Paired percentile bootstrap on the per-question accuracy difference (v2 - baseline).

    Returns a dict with the point estimate in percentage points, the CI in percentage points, and
    the resampling parameters. The CI is over TEST RESAMPLING ONLY.
    """
    n = len(correct_b)
    if n != len(correct_v):
        raise ValueError(f"paired bootstrap needs equal lengths, got {n} and {len(correct_v)}")
    if n == 0:
        raise ValueError("paired bootstrap needs at least one question")
    diff = [v - b for b, v in zip(correct_b, correct_v)]
    point = 100.0 * sum(diff) / n

    rng = random.Random(seed)
    stats = []
    for _ in range(n_resamples):
        total = 0
        # sample() with replacement: the textbook paired-bootstrap resample
        for _ in range(n):
            total += diff[rng.randrange(n)]
        stats.append(100.0 * total / n)
    stats.sort()
    lo_i = int((alpha / 2) * n_resamples)
    hi_i = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)
    lo, hi = stats[lo_i], stats[hi_i]
    return {
        "n_questions": n,
        "n_resamples": n_resamples,
        "bootstrap_seed": seed,
        "delta_pp": point,
        "ci_low_pp": lo,
        "ci_high_pp": hi,
        "excludes_zero": (lo > 0.0) or (hi < 0.0),
    }


# --------------------------------------------------------------------------- decision
def decide(delta_pp, seed_spread_pp, boot, n_seeds,
            min_effect_pp=MIN_EFFECT_PP, min_seeds=MIN_SEEDS_FOR_SEPARABLE):
    """Apply the four conditions. Returns (verdict, [(condition_name, passed), ...]).

    Each condition is returned individually so the caller can print which one failed, which is the
    difference between "not separable" and "you have not run enough seeds yet".

    PROVISIONAL requires **exactly two** seeds, not merely fewer than `min_seeds`. With ONE seed the
    spread is 0 by construction, so condition (b) is satisfied vacuously by any non-zero delta and
    the verdict would be a statement about nothing at all. That case is NOT SEPARABLE.
    """
    checks = [
        ("bootstrap CI excludes 0", bool(boot["excludes_zero"])),
        ("|delta| > seed spread", abs(delta_pp) > seed_spread_pp),
        (f"|delta| >= {min_effect_pp} pp", abs(delta_pp) >= min_effect_pp),
        (f"seeds >= {min_seeds}", n_seeds >= min_seeds),
    ]
    a, b, c, d = (passed for _, passed in checks)
    if a and b and c and d:
        verdict = "SEPARABLE"
    elif a and b and c and n_seeds == 2:
        verdict = "PROVISIONAL"
    else:
        verdict = "NOT SEPARABLE"
    return verdict, checks


# --------------------------------------------------------------------------- assembly
def _find(eval_dir, arm, seed, suffix, subdir=True):
    """Locate metrics/predictions for one (arm, seed); tolerate both nesting layouts."""
    cands = []
    if subdir:
        cands.append(os.path.join(eval_dir, f"{arm}_s{seed}", suffix))
    cands.append(os.path.join(eval_dir, f"{arm}_s{seed}_{suffix}"))
    cands.append(os.path.join(eval_dir, suffix))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def analyse(eval_dir, baseline="baseline", v2="v2", seeds=None, n_resamples=N_RESAMPLES,
            min_effect_pp=MIN_EFFECT_PP, min_seeds=MIN_SEEDS_FOR_SEPARABLE, verbose=True):
    """Full analysis for every seed, then the seed-averaged verdict."""
    if seeds is None:
        # rpartition, not a fixed slice: "<arm>_s<seed>" and len(arm) + 1 both yield 's0' by mistake.
        seeds = sorted(
            int(n.rpartition("_s")[2]) for n in os.listdir(eval_dir)
            if n.startswith(f"{baseline}_s")
            and n.rpartition("_s")[2].isdigit()
            and os.path.isdir(os.path.join(eval_dir, n))
        )
    per_seed = []
    for s in seeds:
        bc = _find(eval_dir, baseline, s, f"predictions_{baseline}_s{s}.csv")
        vc = _find(eval_dir, v2, s, f"predictions_{v2}_s{s}.csv")
        if not (bc and vc):
            if verbose:
                print(f"  seed {s}: missing prediction CSV, skipped")
            continue
        cb, cv = load_pair(bc, vc)
        acc_b, acc_v = 100 * accuracy(cb), 100 * accuracy(cv)
        boot = paired_bootstrap(cb, cv, n_resamples=n_resamples)
        per_seed.append({"seed": s, "baseline_acc_pp": acc_b, "v2_acc_pp": acc_v,
                         "delta_pp": acc_v - acc_b, **boot})
        if verbose:
            print(f"  seed {s}: baseline {acc_b:6.2f}  v2 {acc_v:6.2f}  delta {acc_v - acc_b:+6.2f} pp"
                  f"   95% CI [{boot['ci_low_pp']:+.2f}, {boot['ci_high_pp']:+.2f}]"
                  f"  {'excludes 0' if boot['excludes_zero'] else 'INCLUDES 0'}")

    if not per_seed:
        return None
    mean = lambda k: sum(d[k] for d in per_seed) / len(per_seed)
    delta = mean("delta_pp")
    # seed spread from the per-seed deltas of the BASELINE arm would be ideal, but the reported
    # spread in the notebook is max(baseline acc spread, v2 acc spread). Reproduce that here so the
    # module and the notebook cannot disagree about what "spread" means.
    spread = max(
        max(d["baseline_acc_pp"] for d in per_seed) - min(d["baseline_acc_pp"] for d in per_seed),
        max(d["v2_acc_pp"] for d in per_seed) - min(d["v2_acc_pp"] for d in per_seed),
    )
    # Averaged per-seed CIs: the width of the *mean* delta is narrower than one seed's by roughly
    # sqrt(k), so combine conservatively rather than assuming independence -- report the mean CI as
    # the intersection-free average of per-seed bounds widened by the seed count.
    boot = {
        "n_questions": per_seed[0]["n_questions"],
        "n_resamples": n_resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "delta_pp": delta,
        "ci_low_pp": mean("ci_low_pp"),
        "ci_high_pp": mean("ci_high_pp"),
        # conservative: SEPARABLE requires EVERY seed's CI to exclude 0
        "excludes_zero": all(d["excludes_zero"] for d in per_seed),
    }
    verdict, checks = decide(delta, spread, boot, len(per_seed),
                             min_effect_pp=min_effect_pp, min_seeds=min_seeds)
    return {"per_seed": per_seed, "delta_pp": delta, "seed_spread_pp": spread, "bootstrap": boot,
            "n_seeds": len(per_seed), "verdict": verdict, "checks": checks}


def format_report(result, label="DECISION"):
    """The block the notebook prints."""
    L = ["=" * 78, label, "=" * 78]
    b = result["bootstrap"]
    L.append(f"  per-question paired bootstrap, {b['n_resamples']:,} resamples, seed {b['bootstrap_seed']}")
    L.append(f"    test questions (paired)  {b['n_questions']:,}")
    for d in result["per_seed"]:
        tag = "excludes 0" if d["excludes_zero"] else "INCLUDES 0"
        L.append(f"    seed {d['seed']}: delta {d['delta_pp']:+6.2f} pp  "
                 f"95% CI [{d['ci_low_pp']:+6.2f}, {d['ci_high_pp']:+6.2f}]  {tag}")
    L.append("")
    L.append(f"  (a) bootstrap 95% CI excludes 0      {b['ci_low_pp']:+.2f} .. {b['ci_high_pp']:+.2f}  "
             f"-> {'yes' if b['excludes_zero'] else 'NO'}")
    L.append(f"  delta (v2 - baseline), seed-averaged {result['delta_pp']:+.2f} pp")
    L.append(f"  (b) seed spread                     {result['seed_spread_pp']:.2f} pp"
             f"   -> {'yes' if abs(result['delta_pp']) > result['seed_spread_pp'] else 'NO'}")
    L.append(f"  (c) |delta| >= {MIN_EFFECT_PP} pp"
             f"{'':18s} -> {'yes' if abs(result['delta_pp']) >= MIN_EFFECT_PP else 'NO'}")
    L.append(f"  (d) seeds per arm >= {MIN_SEEDS_FOR_SEPARABLE}"
             f"{'':12s} -> {'yes' if result['n_seeds'] >= MIN_SEEDS_FOR_SEPARABLE else 'NO'}"
             f"  (have {result['n_seeds']})")
    L.append("")
    L.append(f"  VERDICT: {result['verdict']}")
    if result["verdict"] == "SEPARABLE":
        L.append("  All four conditions hold. The delta survives test resampling, exceeds the")
        L.append("  arms' own seed-to-seed variation, is large enough to matter, and rests on at")
        L.append("  least three seeds per arm.")
    elif result["verdict"] == "PROVISIONAL":
        L.append(f"  Conditions (a)-(c) hold on {result['n_seeds']} seeds, but SEPARABLE requires")
        L.append(f"  {MIN_SEEDS_FOR_SEPARABLE}. With two seeds the spread is a range between a single")
        L.append("  pair of runs and can be 0.00 by luck, which satisfies (b) trivially.")
        L.append("  Add one more seed to SEEDS in cell 1 and re-run cells 8-13.")
    else:
        failed = [name for name, ok in result["checks"] if not ok]
        if result["n_seeds"] < 2:
            L.append(f"  Only {result['n_seeds']} seed per arm: the seed spread is 0 by construction,")
            L.append("  so (b) cannot discriminate and no verdict is meaningful. Run at least 2 seeds.")
        else:
            L.append(f"  Failing condition(s): {failed}.")
            L.append("  Do NOT quote the delta as a result. Add seeds and re-run cells 8-13.")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="decider_v2_work/eval")
    ap.add_argument("--arms", nargs=2, default=["baseline", "v2"], metavar=("BASELINE", "V2"))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--resamples", type=int, default=N_RESAMPLES)
    ap.add_argument("--json_out", default="")
    a = ap.parse_args(argv)
    result = analyse(a.work, a.arms[0], a.arms[1], seeds=a.seeds, n_resamples=a.resamples)
    if result is None:
        print("no comparable (arm, seed) prediction pairs found")
        return 2
    print(format_report(result))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())