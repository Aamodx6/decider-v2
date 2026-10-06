"""Step 1 acceptance: config loading, defaults mirroring, and validation."""

from pathlib import Path

import pytest

from decider.config import Config, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "m0.yaml"


def test_yaml_mirrors_defaults():
    cfg = load_config(CONFIG_PATH)
    assert cfg.model == Config().model
    assert cfg.data == Config().data
    assert cfg.train == Config().train
    assert cfg.eval == Config().eval


def test_default_values_match_spec():
    m = Config().model
    assert m.backbone_name == "Qwen/Qwen3-0.6B"
    assert m.num_layers == 20
    assert (m.lora_r, m.lora_alpha) == (8, 16)
    assert m.lora_layers == list(range(8, 20))
    assert m.lora_targets == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert (m.head_dim, m.head_heads, m.head_layers) == (512, 8, 2)
    assert m.pooling == "opt_end"
    # v2 flags default to off
    assert not m.use_ctx_cross_attn
    assert not m.use_null_option
    assert not m.use_rank_emb

    d = Config().data
    assert (d.max_state_tokens, d.max_option_tokens, d.max_question_tokens) == (1024, 64, 256)

    t = Config().train
    assert t.lr_lora == 2e-4
    assert t.lr_head == 1e-3
    assert (t.batch_size, t.grad_accum, t.steps) == (32, 4, 12_000)
    assert t.emd_weight == 0.0


def test_dict_round_trip():
    cfg = load_config(CONFIG_PATH)
    again = Config.from_dict(cfg.to_dict())
    assert again == cfg


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.from_dict({"model": {"nonexistent_flag": 1}, "train": {}, "data": {}, "eval": {}})


def test_emd_weight_is_v2_only():
    with pytest.raises(NotImplementedError):
        Config.from_dict({"train": {"emd_weight": 0.5}})


def test_head_dim_must_divide_heads():
    with pytest.raises(ValueError, match="divisible"):
        Config.from_dict({"model": {"head_dim": 500, "head_heads": 8}})
