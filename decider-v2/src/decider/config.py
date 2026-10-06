"""Configuration dataclasses and YAML loader for decider-v2 M0.

Every tunable value lives here (mirrored by ``configs/m0.yaml``); the rest of
the code contains no magic numbers. The v2 features from ARCH.md (context
cross-attention, null option, rank embedding, EMD loss) exist only as flags
that default to off and raise if enabled — they are NOT implemented in M0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

QUESTION_TYPES: tuple[str, ...] = ("choice", "noul", "score")
# Maps question types to the head's task-type ids; identical to bev-decider's.
TASK_TYPE_IDS: dict[str, int] = {"choice": 0, "noul": 1, "score": 2}
POOLING_MODES: tuple[str, ...] = ("opt_end", "mean")
PRECISIONS: tuple[str, ...] = ("fp32", "bf16")


@dataclass
class ModelConfig:
    """Backbone + decision head. Defaults mirror bev-decider-0.4B (ARCH.md §2)."""

    backbone_name: str = "Qwen/Qwen3-0.6B"
    num_layers: int = 20  # keep the first N of Qwen3-0.6B's 28 layers
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_layers: list[int] = field(default_factory=lambda: list(range(8, 20)))
    lora_targets: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    head_dim: int = 512
    head_heads: int = 8
    head_layers: int = 2
    # ---- v2 features: flags only, off in M0, raise NotImplementedError if enabled ----
    use_ctx_cross_attn: bool = False
    use_null_option: bool = False
    use_rank_emb: bool = False
    # ---- M0 knobs ----
    pooling: str = "opt_end"  # "opt_end" (default) or "mean" (ablation only)
    # [ASSUMPTION] Qwen3Model always applies its final RMSNorm in forward; we keep that
    # behaviour by default and expose the flag so the ablation is a one-line change.
    apply_final_norm: bool = True
    # Custom 4-D masks are passed through by transformers 5.18 for sdpa (verified by
    # tests/test_equivalence.py); flip to "eager" only if a future version breaks it.
    attn_implementation: str = "sdpa"


@dataclass
class DataConfig:
    """Tokenization limits and dataset slicing.

    The dataset has train/test parquet splits and no validation split, so the
    dev set is carved out of train deterministically (see train/data/adapters.py).
    """

    dataset_name: str = "avbiswas/bev-decision"
    dataset_config: str = "default"
    dataset_revision: str | None = None
    # Local clone of the dataset repo, preferred over the Hub when it exists
    # (keeps runs hermetic and offline). Relative to the project root.
    local_data_dir: str | None = "../bev-decision"
    max_state_tokens: int = 1024
    max_option_tokens: int = 64
    max_question_tokens: int = 256
    train_split: str = "train"
    test_split: str = "test"  # held out; never trained on
    train_small_size: int = 50_000  # questions in the fast-iteration subset
    dev_size: int = 2_000  # questions carved out of train for eval-during-training
    subset_seed: int = 0
    max_options: int = 12  # questions with more options than this are skipped
    min_options: int = 2  # and with fewer than this as well


@dataclass
class TrainConfig:
    """Optimization recipe (ARCH.md §13.2 stage 1, M0-faithful)."""

    lr_lora: float = 2e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.03
    batch_size: int = 32  # micro-batch (per forward/backward)
    grad_accum: int = 4  # effective batch = batch_size * grad_accum
    steps: int = 12_000
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    bf16: bool = True
    seed: int = 0
    eval_every: int = 1_000
    ckpt_every: int = 2_000
    # ---- operational extras (documented in NOTES.md; none affect the recipe) ----
    grad_checkpointing: bool = False  # needed to fit 6 GB GPUs at long context
    log_every: int = 20
    num_workers: int = 0
    shuffle_options: bool = True  # per-epoch option-order augmentation
    emd_weight: float = 0.0  # v2 ordinal loss; must stay 0.0 in M0
    smoke_size: int = 64  # --smoke: overfit this many examples ...
    smoke_steps: int = 200  # ... for this many steps


@dataclass
class EvalConfig:
    """Held-out evaluation knobs (eval/run_heldout.py, eval/compare_bev.py)."""

    batch_size: int = 8
    ece_bins: int = 10
    limit: int = 0  # 0 = evaluate the whole split


@dataclass
class Config:
    """Top-level configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run_name: str = "m0-small"
    runs_dir: str = "runs"
    precision: str = "fp32"  # decide() precision: "fp32" | "bf16"

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        """Build a Config from a nested dict, rejecting unknown keys loudly."""
        sections = {
            "model": ModelConfig,
            "data": DataConfig,
            "train": TrainConfig,
            "eval": EvalConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, sub_cls in sections.items():
            sub_raw = raw.get(name, {})
            if not isinstance(sub_raw, dict):
                raise ValueError(f"config section '{name}' must be a mapping")
            kwargs[name] = _build(sub_cls, sub_raw, prefix=name)
        top_fields = {f.name for f in fields(cls) if f.name not in sections}
        unknown = sorted(set(raw) - set(sections) - top_fields)
        if unknown:
            raise ValueError(f"unknown top-level config keys: {unknown}")
        kwargs.update({k: raw[k] for k in raw if k in top_fields})
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config file {path} must contain a mapping")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )

    # ---------------------------------------------------------------- validation
    def validate(self) -> None:
        m, d, t, e = self.model, self.data, self.train, self.eval
        if m.num_layers < 1:
            raise ValueError("model.num_layers must be >= 1")
        if not (0 <= min(m.lora_layers, default=0) and max(m.lora_layers, default=-1) < m.num_layers):
            raise ValueError("model.lora_layers must index into the kept layers [0, num_layers)")
        if m.head_dim % m.head_heads != 0:
            raise ValueError("model.head_dim must be divisible by model.head_heads")
        if m.head_layers < 1:
            raise ValueError("model.head_layers must be >= 1")
        if m.pooling not in POOLING_MODES:
            raise ValueError(f"model.pooling must be one of {POOLING_MODES}")
        if t.emd_weight != 0.0:
            raise NotImplementedError(
                "EMD loss is a v2 feature (ARCH.md §8) and is not implemented in M0; "
                "keep train.emd_weight at 0.0"
            )
        if not (0.0 <= t.warmup_frac < 1.0):
            raise ValueError("train.warmup_frac must be in [0, 1)")
        if t.batch_size < 1 or t.grad_accum < 1:
            raise ValueError("train.batch_size and train.grad_accum must be >= 1")
        if d.max_options < d.min_options:
            raise ValueError("data.max_options must be >= data.min_options")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if e.ece_bins < 2:
            raise ValueError("eval.ece_bins must be >= 2")


def _build(cls: type, raw: dict[str, Any], prefix: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ValueError(f"unknown config keys for '{prefix}': {unknown}")
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        f = known[name]
        if _is_list_field(f) and isinstance(value, (list, tuple)):
            kwargs[name] = list(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _is_list_field(f: Any) -> bool:
    origin = getattr(f.type, "__origin__", None)
    if origin is list:
        return True
    # String annotations (from __future__ import annotations) need a cheap check.
    return isinstance(f.type, str) and f.type.startswith("list")


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    return Config.from_yaml(path)
