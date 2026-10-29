# decider-v2

> **0.6B Parameter System One Decision Model for Bird's-Eye-View (BEV) Autonomous Driving**  

---

## Overview

**decider-v2** is an open-source, mathematically order-invariant decision model for autonomous vehicle reasoning and robotic judgment. It processes Bird's-Eye-View (BEV) situational state tokens and candidate action options in a single forward pass without autoregressive decoding.

### Key Capabilities
- **Choice-Order Invariant by Construction**: Uses shared position ID resets and a 4-D block-additive attention mask $\left[B, 1, L, L\right]$ that prevents candidate options from cross-attending. Permuting candidate options produces identical probabilities ($\max |\Delta p| \le 5.96 \times 10^{-8}$).
- **Context-Aware Decision Head (`head=v2`)**: Includes a Perceiver-style context stack ($n_\text{ctx}=8$) across prefix state representations, option-level self-attention, and explicit rank embeddings for ordinal score questions.
- **Statistical Rigor First**: Integrated multi-seed pipeline (`run_ladder.py`) enforcing $\ge 3$ seeds and paired bootstrap resampling over test questions before declaring architectural separability.

---

## Benchmark Highlights

Evaluated on the held-out BEV decision test split (5,000 questions across 2,617 driving scenarios) and public decision suites:

| Benchmark | Jev 1.13 *(Closed API)* | 0.4B Baseline | decider-v2 *(Ours)* | Kev 0.8B | Laya 0.4B |
|:---|:---:|:---:|:---:|:---:|:---:|
| **All Questions (BEV Decision)** | **78.0%** | 74.7% | **+4.60 pp** *(Smoke)* | 56.6% | 50.4% |
| Choice | **77.8%** | 69.8% | *+7.26 pp* | 54.7% | 44.3% |
| Yes/No (`noul`) | **85.8%** | 84.6% | *+0.45 pp* | 65.1% | 66.8% |
| Ordinal Score | 59.3% | **63.7%** | *+7.23 pp* | 40.3% | 26.5% |
| **sysone-bench v2** | **90.7%** | 70.8% | *In Progress* | 76.6% | 68.6% |
| **JevBench (231 tasks)** | **86.6%** | 65.8% | *In Progress* | 66.7% | 58.4% |
| **Order Invariance ($\max \|\Delta p\|$)** | N/A | $\le 10^{-5}$ | $\mathbf{5.96 \times 10^{-8}}$ | N/A | N/A |
| **Calibration ECE** | — | 0.065 | **0.031** | — | — |

---

## Repository Structure

The core training, evaluation, and export code lives in the flat [`decider-train/`](decider-train/) package:

```
decider-train/
├── network.py             # Qwen3-0.6B backbone truncation + LoRA + DecisionHeadV1/V2
├── tokenization.py        # 4-D block attention mask, shared position ids, sequence packing
├── dataset.py             # Dataset loading, parquet split reader, bucketed batch sampler
├── train.py               # Single GPU / torchrun / SageMaker training engine with budget guards
├── evaluate.py            # Test accuracy, ECE calibration, NLL, and order invariance checker
├── decide.py              # Paired bootstrap & 4-condition statistical significance gate
├── run_ladder.py          # Automated multi-seed orchestrator ([0, 1, 2] across arms)
├── export_hf.py           # Standalone Hugging Face release packaging (safetensors + model card)
├── configs/               # YAML training recipes (smoke, baseline_replica, decider_v2, ablations)
├── benchmarks/            # JevBench and sysone-bench adapters and evaluators
├── scripts/               # Throughput bench (bench.py) and parameter counter (paramcount.py)
├── tests/                 # Offline verification gates (test_core.py, test_ladder.py, golden_check.py)
└── docs/                  # Architecture review and local verification records
```

---

## Quickstart

### 1. Environment Setup
```bash
# Python 3.12 with PyTorch 2.7.1
cd decider-train
uv sync --frozen --extra dev
```

### 2. Run Verification Gates (Offline, CPU, No Downloads)
```bash
# Verify the 4 structural gates, config precedence, and train budget fixes
python tests/test_core.py

# Verify byte-for-byte floating point parity against reference logs
python tests/golden_check.py

# Verify the multi-seed ladder orchestrator and export suite
python tests/test_ladder.py
```

### 3. Run Multi-Seed Ladder
```bash
# Dry-run multi-seed pipeline
python run_ladder.py --seeds 0 1 2 --mode smoke --dry_run

# Full multi-seed run on GPU
python run_ladder.py --seeds 0 1 2 --mode full --device cuda
```

### 4. Export for Hugging Face Release
```bash
python export_hf.py --checkpoint runs/ladder/v2_s0/model/best --out hf_release/decider-v2-0.6B
```

---

## License

This project is licensed under the Apache 2.0 License. Base model weights derive from [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0).
