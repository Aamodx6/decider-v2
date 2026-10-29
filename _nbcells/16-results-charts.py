# Cell 16 - RESULTS.md (runs/RESULTS.md) + the pacing chart. Both are driven by the
# monitor-downloaded artifacts, so the cell reports an honest table for whatever ran here
# plus .NOT-RUN marks for every missing arm: no cell ever invents a number.
import datetime

rows, evals = [], {}
for job, meta in sorted(state["jobs"].items()):
    if meta["arm"].startswith("eval-"):
        evals.setdefault(meta["arm"].removeprefix("eval-"), meta)
        continue
    s = RUNS_LOCAL / job / "output" / "data" / "summary.json"
    if not s.exists():
        rows.append((meta["arm"], job, "-", "FAILED/partial"))
        continue
    d = json.loads(s.read_text(encoding="utf-8"))
    rows.append((meta["arm"], job, f"{d.get('best_val_acc', float('nan'))*100:.2f}%",
                 f"{d.get('final_step')}/{meta['budget']['steps']}"))

md = [f"## SageMaker run {datetime.date.today().isoformat()}",
      f"instance {INSTANCE_TYPE}" + (" x spot" if USE_SPOT else " (on-demand)"),
      "",
      "| arm | job | best val acc | budget |",
      "|-----|-----|-------------:|--------|"]
for arm, job, acc, b in rows:
    md.append(f"| {arm} | {job} | {acc} | {b} |")
for tag in ("baseline", "v2", "reference"):
    if tag in evals:
        md.append(f"\n### eval / {tag}: *summary json preserved; read it from runs/evals/{tag}.json*")

out = RUNS_LOCAL / "RESULTS"
out.mkdir(exist_ok=True)
(out / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
print(f"\nwrote {out / 'RESULTS.md'}")

# pacing chart: sps per step for every completed arm against one figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[charts] matplotlib unavailable in this kernel - RESULTS.md still written")
    plt = None

if plt:
    fig, ax = plt.subplots(figsize=(10, 5))
    drew = False
    for job, meta in sorted(state["jobs"].items()):
        if meta["arm"].startswith("eval-"):
            continue
        p = RUNS_LOCAL / job / "output" / "data" / "metrics.jsonl"
        if not p.exists():
            continue
        steps, sps = [], []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d.get("step"), int) and d.get("sps"):
                steps.append(d["step"]); sps.append(d["sps"])
        if steps:
            ax.plot(steps, sps, marker="o", markersize=3, label=meta["arm"])
            drew = True
    if drew:
        ax.set_xlabel("step"); ax.set_ylabel("presentations / second (sps)")
        ax.set_title("training pacing by arm")
        ax.legend(); ax.grid(True, alpha=0.3)
        chart = out / "pacing.png"
        fig.savefig(chart, dpi=100, bbox_inches="tight")
        print(f"wrote {chart}")
    else:
        print("[charts] no metrics.jsonl downloaded - nothing to plot")
