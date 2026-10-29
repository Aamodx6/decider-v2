"""run_ladder.py -- Multi-seed ladder orchestrator and automated decision pipeline.

Runs both arms (baseline replica and decider-v2) across seeds (e.g. 0, 1, 2),
evaluates checkpoints on the held-out test split, computes the paired bootstrap
and 4 SEPARABLE conditions via decide.py, and updates RESULTS.md.

Usage:
    python run_ladder.py --seeds 0 1 2 --mode full
    python run_ladder.py --seeds 0 1 2 --mode smoke --device cpu
    python run_ladder.py --skip_train --work_dir runs/eval
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import decide


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Multi-seed decider training and evaluation ladder")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="List of random seeds to run (default: 0 1 2)")
    p.add_argument("--arms", nargs=2, default=["baseline", "v2"],
                   metavar=("BASELINE", "V2"),
                   help="The two arms to compare (default: baseline v2)")
    p.add_argument("--baseline_config", default="configs/baseline_replica.yaml",
                   help="Config YAML for baseline arm")
    p.add_argument("--v2_config", default="configs/decider_v2.yaml",
                   help="Config YAML for v2 arm")
    p.add_argument("--mode", choices=["smoke", "full"], default="full",
                   help="Execution mode (smoke uses fast settings, full uses production budget)")
    p.add_argument("--data_dir", default="",
                   help="Base directory containing train/val/test/calib parquets (auto-detected if blank)")
    p.add_argument("--work_dir", default="runs/ladder",
                   help="Output directory for checkpoints, metrics, and prediction CSVs")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training phase and evaluate existing checkpoints in work_dir")
    p.add_argument("--results_md", default="../RESULTS.md",
                   help="Path to markdown results file to update")
    p.add_argument("--device", default="cuda" if sys.platform != "win32" else "cpu",
                   help="Device to pass to training and evaluation (cpu or cuda)")
    p.add_argument("--max_steps", type=int, default=0,
                   help="Override max_steps in configs if > 0")
    p.add_argument("--dry_run", action="store_true",
                   help="Print planned operations without executing")
    return p.parse_args(argv)


def resolve_data_paths(data_dir: str, mode: str) -> dict[str, str]:
    if data_dir and os.path.isdir(data_dir):
        base = Path(data_dir)
    elif mode == "smoke":
        cand = [ROOT / "data_smoke", ROOT.parent / "decider_v2_work" / "golden" / "data", ROOT / "data"]
        base = next((c for c in cand if c.is_dir()), ROOT / "data")
    else:
        cand = [ROOT / "data", ROOT.parent / "data", Path("/opt/ml/input/data")]
        base = next((c for c in cand if c.is_dir()), ROOT / "data")

    return {
        "train": str(base / "train"),
        "val": str(base / "val"),
        "test": str(base / "test"),
        "calib": str(base / "calib"),
    }


def run_training_step(arm: str, config_file: str, seed: int, out_dir: Path,
                      data_paths: dict[str, str], device: str, max_steps: int = 0,
                      dry_run: bool = False) -> Path:
    arm_dir = out_dir / f"{arm}_s{seed}"
    model_dir = arm_dir / "model"
    ckpt_dir = arm_dir / "ckpt"
    output_dir = arm_dir / "output"
    best_ckpt = model_dir / "best"

    if best_ckpt.exists() and (best_ckpt / "meta.json").exists():
        print(f"[TRAIN] Checkpoint already exists for {arm} (seed {seed}) at {best_ckpt}. Skipping train.")
        return best_ckpt

    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--config", str(ROOT / config_file),
        "--seed", str(seed),
        "--train_channel", data_paths["train"],
        "--val_channel", data_paths["val"],
        "--test_channel", data_paths["test"],
        "--model_dir", str(model_dir),
        "--checkpoint_dir", str(ckpt_dir),
        "--output_dir", str(output_dir),
    ]
    if max_steps > 0:
        cmd.extend(["--max_steps", str(max_steps)])
    if device == "cpu":
        cmd.extend(["--attn_implementation", "sdpa"])

    print(f"\n[TRAIN] Launching {arm} seed {seed} ...")
    print(f"        Command: {' '.join(cmd)}")
    if dry_run:
        return best_ckpt

    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Training failed for {arm} seed {seed} with returncode {res.returncode}")

    if not best_ckpt.exists():
        final_ckpt = model_dir / "final"
        if final_ckpt.exists():
            return final_ckpt
        raise FileNotFoundError(f"Neither best nor final checkpoint found at {model_dir}")
    return best_ckpt


def run_eval_step(arm: str, seed: int, ckpt_path: Path, out_dir: Path,
                  data_paths: dict[str, str], device: str, dry_run: bool = False) -> Path:
    pred_csv = out_dir / f"predictions_{arm}_s{seed}.csv"
    if pred_csv.exists() and pred_csv.stat().st_size > 0:
        print(f"[EVAL] Prediction file already exists: {pred_csv}. Skipping eval.")
        return pred_csv

    cmd = [
        sys.executable, str(ROOT / "evaluate.py"),
        "--checkpoint", str(ckpt_path),
        "--test_channel", data_paths["test"],
        "--predictions_out", str(pred_csv),
        "--out", str(out_dir / f"{arm}_s{seed}_eval"),
        "--env_check", "0",
    ]
    if os.path.isdir(data_paths.get("calib", "")):
        cmd.extend(["--calib_channel", data_paths["calib"]])

    print(f"\n[EVAL] Evaluating {arm} seed {seed} ...")
    print(f"       Command: {' '.join(cmd)}")
    if dry_run:
        return pred_csv

    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {arm} seed {seed} with returncode {res.returncode}")
    return pred_csv


def generate_results_markdown(result: dict, baseline_arm: str, v2_arm: str, mode: str) -> str:
    b = result["bootstrap"]
    lines = [
        "# RESULTS",
        "",
        f"mode: {mode.upper()}",
        f"arms: {baseline_arm} vs {v2_arm}",
        f"test questions: {b['n_questions']:,} (paired bootstrap over {b['n_resamples']:,} resamples)",
        f"seeds: {sorted(d['seed'] for d in result['per_seed'])}",
        "",
        "## Per-Seed Summary",
        "",
        f"| Seed | {baseline_arm} Acc (%) | {v2_arm} Acc (%) | Delta (pp) | 95% CI (pp) | Excludes 0 |",
        "|---:|---:|---:|---:|:---:|:---:|",
    ]
    for d in result["per_seed"]:
        ci = f"[{d['ci_low_pp']:+.2f}, {d['ci_high_pp']:+.2f}]"
        ex = "yes" if d["excludes_zero"] else "NO"
        lines.append(
            f"| {d['seed']} | {d['baseline_acc_pp']:.2f} | {d['v2_acc_pp']:.2f} | {d['delta_pp']:+.2f} | {ci} | {ex} |"
        )

    lines.extend([
        "",
        "## Decision Gate",
        "",
        f"**Verdict: {result['verdict']}**",
        "",
        "| # | Condition | Measured | Result |",
        "|---|---|---:|:---:|",
        f"| (a) | Bootstrap 95% CI excludes 0 | [{b['ci_low_pp']:+.2f}, {b['ci_high_pp']:+.2f}] | {'PASS' if b['excludes_zero'] else 'FAIL'} |",
        f"| (b) | |delta| > seed spread | |{result['delta_pp']:+.2f}| vs {result['seed_spread_pp']:.2f} | {'PASS' if abs(result['delta_pp']) > result['seed_spread_pp'] else 'FAIL'} |",
        f"| (c) | |delta| >= 1.0 pp | |{result['delta_pp']:+.2f}| pp | {'PASS' if abs(result['delta_pp']) >= 1.0 else 'FAIL'} |",
        f"| (d) | Seeds >= 3 | {result['n_seeds']} per arm | {'PASS' if result['n_seeds'] >= 3 else 'FAIL'} |",
        "",
        f"Delta (seed-averaged): **{result['delta_pp']:+.2f} pp** across {result['n_seeds']} seed(s) per arm.",
        "",
    ])

    if result["verdict"] == "SEPARABLE":
        lines.append("> **ALL CONDITIONS PASSED.** The architecture separation is statistically confirmed.")
    elif result["verdict"] == "PROVISIONAL":
        lines.append("> **PROVISIONAL RESULT.** Conditions (a)-(c) hold, but condition (d) requires additional seeds.")
    else:
        lines.append("> **NOT SEPARABLE.** Do not quote this delta as a verified finding.")

    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    data_paths = resolve_data_paths(args.data_dir, args.mode)
    arm_configs = {
        args.arms[0]: args.baseline_config,
        args.arms[1]: args.v2_config,
    }

    print("=" * 78)
    print("DECIDER MULTI-SEED LADDER")
    print(f"  Arms:       {args.arms[0]} ({arm_configs[args.arms[0]]}) vs {args.arms[1]} ({arm_configs[args.arms[1]]})")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Mode:       {args.mode}")
    print(f"  Work Dir:   {work_dir}")
    print("=" * 78)

    # Step 1: Train & Eval each (arm, seed)
    for arm in args.arms:
        cfg = arm_configs[arm]
        for seed in args.seeds:
            if not args.skip_train:
                ckpt = run_training_step(
                    arm=arm, config_file=cfg, seed=seed, out_dir=work_dir,
                    data_paths=data_paths, device=args.device,
                    max_steps=args.max_steps, dry_run=args.dry_run
                )
            else:
                ckpt = work_dir / f"{arm}_s{seed}" / "model" / "best"

            if not args.dry_run:
                run_eval_step(
                    arm=arm, seed=seed, ckpt_path=ckpt, out_dir=work_dir,
                    data_paths=data_paths, device=args.device, dry_run=args.dry_run
                )

    if args.dry_run:
        print("\n[DRY RUN] Finished planned steps.")
        return 0

    # Step 2: Statistical Decision analysis via decide.py
    print("\n" + "=" * 78)
    print("COMPUTING STATISTICAL SEPARATION VIA decide.py")
    print("=" * 78)

    result = decide.analyse(str(work_dir), args.arms[0], args.arms[1], seeds=args.seeds)
    if result is None:
        print("[ERROR] Could not compute decision report. Missing prediction CSV pairs.")
        return 1

    report_text = decide.format_report(result)
    print("\n" + report_text)

    # Step 3: Write RESULTS.md
    md_content = generate_results_markdown(result, args.arms[0], args.arms[1], args.mode)
    res_path = Path(args.results_md).resolve()
    with open(res_path, "w", encoding="utf-8") as fh:
        fh.write(md_content)
    print(f"\n[UPDATED] {res_path}")

    # Step 4: Write machine-readable json
    summary_json = work_dir / "ladder_summary.json"
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"[SAVED] {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
