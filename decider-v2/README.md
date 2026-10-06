# decider-v2

An improved, open, commercially-usable reimplementation of
[`avbiswas/bev-decider-0.4B`](https://huggingface.co/avbiswas/bev-decider-0.4B):
a small **System One decision model** that reads a state plus typed questions
(`choice` / `noul` / `score`) and returns calibrated probabilities in **one
forward pass, no generation**.

This repository currently contains **Milestone M0**: a faithful, measurable
bev-decider-style baseline. See `ARCH.md` for the full architecture document
and `NOTES.md` for assumptions, environment details and reproduction commands.

> **Status: research baseline (M0).** Weights trained on the
> `avbiswas/bev-decision` dataset are for research only until the dataset's
> mixed-source terms are verified — see `train/data/DATA_LICENSES.md`.

<!-- README is finalized in step 10; install/train/eval commands are listed there. -->
