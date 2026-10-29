# autoresearch: Autonomous Research Loop for decider-v2

This document is the master instruction brief for an autonomous AI research agent operating on the **decider-v2** autonomous driving decision model. It adapts the [karpathy/autoresearch](https://github.com/karpathy/autoresearch) paradigm for order-invariant neural decision architectures.

---

## 1. The Core Objective

Your primary objective is to **discover architectural, optimization, and algorithmic improvements** that outperform the baseline `decider-v2` model:
- **Primary Metric**: **Validation Loss (`val_loss`)** — lower is better.
- **Secondary Metric**: **Validation Accuracy (`val_acc`)** — higher is better.
- **Calibration Metric**: **Expected Calibration Error (`val_ece`)** — lower is better.
- **HARD GATING CONSTRAINT**: **Choice-Order Invariance ($\max |\Delta p| \le 10^{-5}$)**. Any modification that allows option order to affect output probabilities fails immediately and MUST be discarded.

All experiments are evaluated on the held-out validation split (`data/val/questions.parquet`).

---

## 2. Boundaries: What You CAN and CANNOT Touch

### In-Scope (The Mutable Sandbox)
You are permitted and encouraged to modify:
1. `decider-train/network.py`:
   - Decision head architecture (`DecisionHeadV2`, `DecisionHeadV3`, relative contrastive pooling, SwiGLU, RMSNorm).
   - Prefix context aggregation (Perceiver queries, spatial vs instruction token split).
   - Backbone LoRA configuration (target modules, rsLoRA, rank, alpha, dropout).
   - Loss functions (temperature scaling, calibrated focal loss, ordinal margin loss).
2. `decider-train/train.py`:
   - Optimizers (AdamW, Lion, Muon, Adafactor).
   - Learning rate schedules, warmup ratios, weight decay, gradient clipping.
   - Batch size and accumulation settings within memory limits.
3. `decider-train/configs/autoresearch.yaml`:
   - Hyperparameter configurations and ablation toggles.

### Out-of-Scope (The Immutable Yardstick)
You MUST NEVER modify:
1. `decider-train/dataset.py` — The data loader and split definitions are the ground truth yardstick.
2. `decider-train/evaluate.py` — The metric computation and benchmark scoring scripts.
3. `decider-train/decide.py` — The paired bootstrap significance tester.
4. `decider-train/tests/test_core.py` — The structural gate tests.

Cheating by weakening the yardstick or altering metrics results in an immediate discard.

---

## 3. Fixed Budget Rule

Every experiment must execute under a **strict, identical compute budget** so that progress represents genuine modeling gains, not just training longer:
- **Local / Smoke Budget**: 200 training steps (or a fixed 5-minute wall-clock cap).
- **GPU Budget**: 1,000 training steps on a single GPU.

Training duration is not a variable.

---

## 4. The Ratchet Loop

For every iteration:
1. **Hypothesize**: Select a concrete hypothesis from `autoresearch/hypotheses.md` or formulate a new one based on recent `results.tsv` findings.
2. **Implement**: Make clean, minimal, self-contained code edits in the mutable sandbox.
3. **Verify Gates**: Run `python tests/test_core.py --skip_overfit`. If structural checks fail, fix or discard immediately.
4. **Train**: Execute the fixed-budget training run:
   ```bash
   python autoresearch/loop.py --tag <tag> --desc "<one-line description>"
   ```
5. **Evaluate**: Parse the validation metrics (`val_loss`, `val_acc`, `val_ece`, `invariance_max_dp`).
6. **Ratchet Decision**:
   - **KEEP**: If `val_loss < best_val_loss` AND `invariance_max_dp <= 1e-5`, commit the change:
     `git commit -am "autoresearch(<tag>): <description> [val_loss: X.XXXX, val_acc: YY.YY%]"`
     Log `status=keep` in `autoresearch/results.tsv`.
   - **DISCARD**: If `val_loss >= best_val_loss` or gates failed:
     Revert changes using `git checkout -- .`
     Log `status=discard` in `autoresearch/results.tsv`.
7. **Repeat**: Proceed immediately to the next experiment.

---

## 5. Simplicity Criterion

All else being equal, **simpler code is better**.
- A 3-line modification that matches performance while reducing parameter count or FLOPs is a KEEP.
- Over-engineered, fragile mechanisms that fail to demonstrate measurable statistical gains are discarded.
- Never add external dependencies. Use standard PyTorch and existing library primitives.

---

## 6. The Never-Stop Directive

Do not pause to ask the user whether to continue. Work systematically through hypotheses, reason about why previous trials succeeded or failed, and keep iterating until stopped.
