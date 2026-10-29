# autoresearch for decider-v2

An autonomous machine learning research harness for **decider-v2**, adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

---

## What is this?

Instead of manually editing code, running training runs, waiting, and checking metrics, `autoresearch` shifts human involvement to "programming the research program":
1. The **Human** defines the objective, boundaries, and prioritized hypothesis backlog (`program.md`, `hypotheses.md`).
2. The **AI Agent** formulates specific hypotheses, edits the architecture/training code in `decider-train/`, verifies test gates, and executes time-budgeted runs.
3. The **Ratchet Loop** automatically keeps winning commits and reverts regressions or broken gates, appending all trials to `results.tsv`.

---

## Directory Layout

```
autoresearch/
├── program.md        # Master instruction prompt for the AI research agent
├── hypotheses.md     # Prioritized queue of modeling and optimization hypotheses
├── loop.py           # Automated experiment runner & git ratchet orchestrator
├── results.tsv       # Append-only experiment ledger tracking every trial
└── README.md         # This overview
```

---

## How to Run an Experiment

### 1. Dry Run (Test the pipeline without executing training)
```bash
python autoresearch/loop.py --tag test_pipeline --desc "Validate autoresearch loop" --dry_run
```

### 2. Run a Fixed-Budget Experiment (Local / CPU or GPU)
```bash
# Propose a code change in decider-train/network.py, then launch:
python autoresearch/loop.py --tag exp01_swiglu --desc "Upgrade decision head MLP to SwiGLU" --budget_steps 100
```

### 3. Check Experiment Ledger
```bash
cat autoresearch/results.tsv
```
