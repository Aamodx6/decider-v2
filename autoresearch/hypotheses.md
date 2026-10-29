# Research Hypotheses Backlog for decider-v2

This document maintains a prioritized queue of architectural and optimization hypotheses for autonomous iteration under `autoresearch`.

---

## Tier 1: Decision Head Architecture (High Impact, Fast Iteration)

### `H1_RELATIVE_OPTION_DIFF` (Priority: HIGH)
- **Hypothesis**: In driving judgment, selecting an action is comparative rather than absolute. Augmenting each option representation $\mathbf{x}_k$ with its difference from the mean option embedding $\mathbf{x}_k - \bar{\mathbf{x}}$ (where $\bar{\mathbf{x}} = \frac{1}{K}\sum_j \mathbf{x}_j$) provides contrastive relative signal while preserving strict mathematical order invariance.
- **Expected Impact**: Faster convergence, sharper separation between close options, -0.05 val_loss.
- **File**: `decider-train/network.py` (`DecisionHeadV2` / `DecisionHeadV3`).

### `H2_SWIGLU_HEAD` (Priority: HIGH)
- **Hypothesis**: Replacing the 2-layer GELU MLP ($\mathbf{W}_2 \text{GELU}(\mathbf{W}_1 \mathbf{x})$) in each head block with a SwiGLU gating mechanism ($\mathbf{W}_\text{down} (\text{SiLU}(\mathbf{W}_\text{gate} \mathbf{x}) \odot \mathbf{W}_\text{up} \mathbf{x})$) increases representational capacity per parameter.
- **Expected Impact**: Higher accuracy on multi-choice questions with negligible compute overhead.
- **File**: `decider-train/network.py` (`HeadBlockV2`).

### `H3_RMSNORM_STABILITY` (Priority: MEDIUM)
- **Hypothesis**: Replacing standard `nn.LayerNorm` with `RMSNorm` removes mean-centering operations, speeding up backward passes and stabilizing gradients for deeper head stacks ($L \ge 4$).
- **Expected Impact**: ~5% faster step throughput, lower gradient variance.
- **File**: `decider-train/network.py`.

---

## Tier 2: Loss Function & Probability Calibration

### `H4_LEARNABLE_TEMPERATURE` (Priority: HIGH)
- **Hypothesis**: Logits from linear projection heads are often poorly scaled, causing probability overconfidence and high ECE. Adding a learnable per-task temperature parameter $\tau_\text{type} \in [0.1, 5.0]$ with an entropy regularization penalty will directly minimize ECE during training.
- **Expected Impact**: Reduces ECE from ~0.065 to $< 0.030$, lower NLL.
- **File**: `decider-train/network.py`, `decider-train/train.py`.

### `H5_ORDINAL_MARGIN_LOSS` (Priority: MEDIUM)
- **Hypothesis**: Standard Earth Mover's Distance treats rank errors linearly. An ordinal margin ranking loss that penalizes wrong-direction rank predictions with quadratic distance $(\text{rank}_i - \text{rank}_j)^2$ will improve accuracy on ordinal `score` questions.
- **Expected Impact**: +3.0 pp on score question accuracy.
- **File**: `decider-train/network.py` (`emd_loss`).

---

## Tier 3: Context Aggregation & Spatial Perception

### `H6_SPATIAL_INTENT_SPLIT` (Priority: MEDIUM)
- **Hypothesis**: Currently, prefix tokens (BEV scene description + instruction) are pooled uniformly with $n_\text{ctx}=8$ learned queries. Splitting the prefix into spatial tokens (first 80%) and instruction tokens (last 20%), and cross-attending the spatial summary against the instruction query, ensures candidate options are scored specifically against the driver's intent.
- **Expected Impact**: Better generalization across unseen scenario domains.
- **File**: `decider-train/network.py` (`context_summary`).

---

## Tier 4: Backbone Adaptation & LoRA Topology

### `H7_RSLORA_MLP_INCLUSION` (Priority: LOW / GPU ONLY)
- **Hypothesis**: Applying LoRA to both self-attention (`q, k, v, o`) and MLP layers (`gate, up, down`) with Rank-Stabilized LoRA scaling ($\gamma = \frac{\alpha}{\sqrt{r}}$) allows deeper domain adaptation from general text to Bird's-Eye-View spatial tokens without catastrophic forgetting.
- **Expected Impact**: -0.08 val_loss on full-scale training.
- **File**: `decider-train/network.py` (`lora_target_regex`).
