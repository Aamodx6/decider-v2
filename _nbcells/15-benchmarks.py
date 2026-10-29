# Cell 15 - benchmark spot-checks against the two SVM harnesses (jevbench / sysone-bench), on the
# monitor-downloaded checkpoints. NOT RUN is a legal outcome: each harness needs its own clone
# checkout (JEVBENCH_ROOT / SYSONE_BENCH_ROOT) that the notebook cannot conjure; the SageMaker
# run itself is not benchmark-harness-dependent, so a missing clone never blocks the launchers.
import os
import subprocess

HARNESSES = [("jevbench", "JEVBENCH_ROOT"), ("sysone_bench", "SYSONE_BENCH_ROOT")]
BENCH_LIMIT = 64        # spot-check scale; a full sweep is a separate machine-sized job

def local_ckpt(arm):
    for job in state["launches"].get(arm, []):
        d = RUNS_LOCAL / job / "checkpoints" / "best"
        if d.is_dir() and (d / "meta.json").exists():
            return d
    return None

for harness, root_env in HARNESSES:
    script = REPO / "benchmarks" / f"run_{harness}.py"
    if not script.exists():
        print(f"[NOT RUN] {harness}: benchmarks/run_{harness}.py is not in this checkout")
        continue
    root = os.environ.get(root_env, "")
    if not (root and os.path.isdir(root)):
        print(f"[NOT RUN] {harness}: {root_env} is not set to a clone checkout")
        continue
    for arm in ("baseline", "v2") if not SMOKE else ("smoke",):
        ckpt = local_ckpt(arm)
        if ckpt is None:
            print(f"[skip] {harness}/{arm}: no downloaded checkpoint (run cells 11-13 first)")
            continue
        out = RUNS_LOCAL / f"{harness}-{arm}-limit{BENCH_LIMIT}.json"
        cmd = [sys.executable, str(script), "--checkpoint", str(ckpt),
               "--out", str(out), "--limit", str(BENCH_LIMIT),
               "--batch_size", "8", "--progress_every", "32"]
        if harness == "jevbench":
            cmd += ["--jevbench_root", root]
        if harness == "sysone_bench":
            cmd += ["--sysone_bench_root", root]
        print(f"[bench] {harness}/{arm}: {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=str(REPO), env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        print(f"[bench] {harness}/{arm}: rc={r.returncode}"
              + ("" if r.returncode == 0 else " - see the traceback above; result json absent"))
