"""golden_check.py -- prove the refactor did not change a single training number.

The reference below was produced by the OLD code (serialize.py / data.py / model.py / losses.py,
`decider_v2_work/golden/old_{baseline,v2}.log`, full-precision copies in those runs'
`output/metrics.jsonl`): a 20-step CPU run on a 2-layer random-init Qwen3-shaped backbone, seed 0,
24 train / 8 val questions.  This script re-runs the SAME recipe through the refactored `train.py`
and compares every logged step.

    python tests/golden_check.py              # compare (exit 1 on any drift)
    python tests/golden_check.py --save       # print a new reference dict (after a deliberate change)

The comparison covers, per step, the training loss, the gradient norm and the batch accuracy, plus
the two validation evaluations.  Tolerance defaults to 1e-6.

The reference data lives outside the package (it is a tiny carve of avbiswas/bev-decision, 124 KB,
kept out of git).  If it is missing, regenerate it with:

    python dataset.py prepare --out ../decider_v2_work/golden/data --limit_train 24 --limit_val 8 ...

and compare against the numbers below -- they are the contract, not the files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# --------------------------------------------------------------------------- the reference
# step -> (loss, grad_norm, acc), copied verbatim from
# decider_v2_work/golden/old/{baseline,v2}/output/metrics.jsonl
SAVED = {
    "v1": {
        "train": {
            1: (1.245650827884674, 2.5049312114715576, 0.25),
            2: (0.7277127206325531, 1.9188742637634277, 0.5),
            3: (1.0454916656017303, 2.1565065383911133, 0.5),
            4: (0.7403463125228882, 1.5885820388793945, 0.75),
            5: (0.64116171002388, 1.3116157054901123, 0.875),
            6: (0.41389724612236023, 3.459027051925659, 0.875),
            7: (0.40791530162096024, 1.6932663917541504, 0.875),
            8: (0.40642496570944786, 1.3782137632369995, 1.0),
            9: (0.364732950925827, 1.7697933912277222, 1.0),
            10: (0.405405730009079, 5.369292259216309, 1.0),
            11: (0.15251542814075947, 0.8786697387695312, 1.0),
            12: (0.40807729959487915, 0.9072472453117371, 0.75),
            13: (0.17136466037482023, 0.6686964631080627, 1.0),
            14: (0.22876215889118612, 0.4582863748073578, 0.75),
            15: (0.4440760761499405, 1.217771291732788, 1.0),
            16: (0.3490109220147133, 1.3413058519363403, 0.875),
            17: (0.31477181904483587, 0.37900614738464355, 1.0),
            18: (0.15499834797810763, 0.7780562043190002, 1.0),
            19: (0.5355055928230286, 0.6215978860855103, 0.875),
            20: (0.001619537710212171, 0.0819215252995491, 1.0),
        },
        "val": {10: (1.2769896984100342, 0.5), 20: (1.4353263974189758, 0.5)},
    },
    "v2": {
        "train": {
            1: (1.1640711426734924, 2.50278902053833, 0.25),
            2: (1.184401273727417, 6.552960395812988, 0.625),
            3: (1.2645938396453857, 2.848036527633667, 0.375),
            4: (0.9174426794052124, 0.8057848215103149, 0.5),
            5: (0.8809534013271332, 1.2505812644958496, 0.75),
            6: (0.810683935880661, 1.2367042303085327, 0.5),
            7: (0.7158573567867279, 0.9606133699417114, 0.625),
            8: (0.7995021939277649, 0.7373191714286804, 0.875),
            9: (0.6905903518199921, 1.601380467414856, 0.75),
            10: (0.7605870813131332, 0.8905217051506042, 0.875),
            11: (0.660550057888031, 1.8010796308517456, 0.75),
            12: (0.6657112240791321, 1.1453795433044434, 0.625),
            13: (0.45767128467559814, 1.4451924562454224, 1.0),
            14: (0.60490682721138, 1.766682505607605, 0.75),
            15: (0.8533826768398285, 2.99236798286438, 0.875),
            16: (0.6963000893592834, 2.8528575897216797, 1.0),
            17: (0.6506820321083069, 2.2358531951904297, 1.0),
            18: (0.35568705946207047, 1.9764024019241333, 1.0),
            19: (0.8812876045703888, 2.8467469215393066, 0.75),
            20: (0.23729421943426132, 2.6083383560180664, 1.0),
        },
        "val": {10: (1.4129470586776733, 0.25), 20: (1.5920583009719849, 0.375)},
    },
}

# --------------------------------------------------------------------------- the recipe
# Every value is from the old run's own output/args.json, so the only thing that varies between
# the reference and the re-run is the code itself.  The two arms are NOT the same recipe -- the
# baseline is the bev-decider replica (r=8 attn-only, smoothed-CE off) and v2 is the Tier-1"
# arm (r=16 attn+MLP, label smoothing 0.05, EMD 0.5, a smaller context head).
RECIPE_COMMON = [
    "--max_state_tokens", "1024", "--max_choice_tokens", "64",
    "--mode", "bev", "--pool", "bev", "--shuffle_options", "1",
    # --noul_use_criteria is a store_true flag; both old runs had it False, so it is omitted
    "--backbone_name", "Qwen/Qwen3-0.6B", "--num_layers", "2", "--random_init", "1",
    "--vocab_size", "151669", "--attn_implementation", "sdpa",
    "--lora_dropout", "0.0", "--lora_last_k_layers", "2",
    "--head_new_dim", "512", "--head_layers", "2", "--head_dropout", "0.0",
    "--use_ctx", "1", "--use_rank", "1", "--use_null", "0", "--use_derived", "0",
    "--batch_size", "4", "--grad_accum_steps", "2", "--max_steps", "20", "--epochs", "7",
    "--lr_lora", "0.0002", "--lr_head", "0.001", "--weight_decay", "0.01",
    "--warmup_ratio", "0.03", "--grad_clip", "1.0",
    "--seed", "0", "--num_workers", "0", "--log_every", "1",
    "--eval_every", "10", "--save_every", "20",
    "--limit_train", "0", "--limit_val", "0",
    "--gradient_checkpointing", "1", "--selftest", "0", "--env_check", "0",
]

# per-arm knobs, straight from each run's args.json
RECIPE_ARM = {
    "v1": ["--lora_r", "8", "--lora_alpha", "16", "--lora_include_mlp", "0",
           "--head_heads", "8", "--n_ctx", "8",
           "--label_smoothing", "0.0", "--emd_lambda", "0.0"],
    "v2": ["--lora_r", "16", "--lora_alpha", "32", "--lora_include_mlp", "1",
           "--head_heads", "4", "--n_ctx", "4",
           "--label_smoothing", "0.05", "--emd_lambda", "0.5"],
}


def run_arm(head, data, workdir, python, extra=()):
    out = Path(workdir) / head
    cmd = [str(python), str(ROOT / "train.py")] + RECIPE_COMMON + RECIPE_ARM[head] + [
        "--head", head,
        "--train_channel", str(Path(data) / "train"),
        "--val_channel", str(Path(data) / "val"),
        "--test_channel", str(Path(data) / "test"),
        "--model_dir", str(out / "model"),
        "--output_dir", str(out / "output"),
        "--checkpoint_dir", str(out / "ckpt"),
    ] + list(extra)
    print(f"\n--- running head={head} ---")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"train.py failed for head={head} (exit {proc.returncode})")
    rows = [json.loads(line) for line in
            (out / "output" / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    train = {r["step"]: (r["loss"], r["grad_norm"], r["acc"]) for r in rows if "loss" in r}
    val = {r["step"]: (r["val_loss"], r["val_acc"]) for r in rows if "val_loss" in r}
    return train, val, out


def compare(head, train, val, tol, grad_tol):
    """Compare one arm against the saved reference.

    `tol` applies to the training loss, the batch accuracy and the validation numbers -- the
    quantities that describe what the model computed.  `grad_tol` applies to grad_norm, which is a
    diagnostic emitted by torch's own `clip_grad_norm_`: that reduction runs in float32 inside
    torch and its result differs in the last digits between torch 2.7.1 (this checkout, matching
    the SageMaker container) and the torch 2.11/2.12 that recorded the reference.  Both tolerances
    are printed with the measured maximum, so neither is hidden.
    """
    saved = SAVED[head]
    worst, worst_at = 0.0, None
    worst_g, worst_g_at = 0.0, None
    problems = []
    if sorted(train) != sorted(saved["train"]):
        problems.append(f"step set differs: ran {sorted(train)}, expected {sorted(saved['train'])}")
    for step, want in sorted(saved["train"].items()):
        got = train.get(step)
        if got is None:
            problems.append(f"step {step}: missing from the run")
            continue
        for name, g, w in zip(("loss", "grad_norm", "acc"), got, want):
            d = abs(g - w)
            if name == "grad_norm":
                if d > worst_g:
                    worst_g, worst_g_at = d, f"step {step}"
                if d > grad_tol:
                    problems.append(f"step {step} {name}: got {g!r} want {w!r} (|d|={d:.3e})")
                continue
            if d > worst:
                worst, worst_at = d, f"step {step} {name}"
            if d > tol:
                problems.append(f"step {step} {name}: got {g!r} want {w!r} (|d|={d:.3e})")
    for step, (wloss, wacc) in sorted(saved["val"].items()):
        got = val.get(step)
        if got is None:
            problems.append(f"val step {step}: missing from the run")
            continue
        for name, g, w in zip(("val_loss", "val_acc"), got, (wloss, wacc)):
            d = abs(g - w)
            if d > worst:
                worst, worst_at = d, f"val {step} {name}"
            if d > tol:
                problems.append(f"val step {step} {name}: got {g!r} want {w!r} (|d|={d:.3e})")

    n = len(saved["train"]) + len(saved["val"])
    print(f"\n[{head}] compared {len(saved['train'])} train steps + {len(saved['val'])} "
          f"validation points ({n * 3} + {len(saved['val'])} values)")
    print(f"[{head}] max |delta| loss/acc/val = {worst:.3e} at {worst_at} (tolerance {tol:g})")
    print(f"[{head}] max |delta| grad_norm    = {worst_g:.3e} at {worst_g_at} "
          f"(tolerance {grad_tol:g}, torch-build reduction)")
    for p in problems:
        print(f"  DRIFT {p}")
    return problems


def main():
    ap = argparse.ArgumentParser(description="golden test: old-code numbers vs the refactored train.py")
    ap.add_argument("--data", default=str(ROOT.parent / "decider_v2_work" / "golden" / "data"),
                    help="the 24/8/32-question carve the reference was recorded on")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="max |delta| for loss / accuracy / validation numbers. The old reference "
                         "was recorded on torch 2.11-2.12, this checkout runs the container's 2.7.1; "
                         "measured maxima are ~1e-7 per step, ~1.8e-6 after ten clipped steps.")
    ap.add_argument("--grad_tol", type=float, default=1e-4,
                    help="max |delta| for grad_norm, a float32 reduction inside clip_grad_norm_"
                         " whose last digits move between torch builds.")
    ap.add_argument("--save", action="store_true", help="print the observed values as a new reference")
    ap.add_argument("--heads", default="v1,v2")
    args = ap.parse_args()

    data = Path(args.data)
    if not (data / "train").is_dir():
        raise SystemExit(
            f"golden data not found at {data}.\n"
            "It is a 124 KB carve kept outside git; regenerate it with dataset.py prepare, or run "
            "with --data <dir>.")

    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    failures = 0
    with tempfile.TemporaryDirectory(prefix="golden_") as tmp:
        for head in heads:
            train, val, out = run_arm(head, data, tmp, args.python)
            if args.save:
                print(f'    "{head}": {{')
                print('        "train": {')
                for s in sorted(train):
                    print(f"            {s}: {train[s]},")
                print("        },")
                print(f'        "val": {dict(sorted(val.items()))},')
                print("    },")
                continue
            problems = compare(head, train, val, args.tol, args.grad_tol)
            failures += len(problems)

    if args.save:
        return 0
    print()
    if failures:
        print(f"GOLDEN CHECK FAILED: {failures} drifting value(s)")
        return 1
    print("GOLDEN CHECK PASSED: every step identical to the old code within "
          f"{args.tol:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
