"""Critical correctness gate: joint forward == per-option separate forwards.

Running prefix + all options in one pass with the custom 4-D mask/position ids
must give the same <opt_end> hidden states as running [prefix + that option]
alone with the same position ids. If this fails, the masking is wrong — fix
the masking, never this test.
"""

import pytest
import torch

from decider.backbone import TruncatedQwen3Backbone
from decider.config import ModelConfig
from decider.masking import build_mask_and_pos, pack_single, to_additive_mask
from decider.tokenizer import DeciderTokenizer, load_tokenizer

BACKBONE_NAME = "Qwen/Qwen3-0.6B"
STATE = {
    "order_id": "A-1029",
    "customer": {"name": "Dana", "tier": "gold"},
    "items": [{"sku": "x1", "qty": 2}, {"sku": "y9", "qty": 1}],
    "notes": "Gift wrap requested. Delivery deadline 2026-03-14.",
}
QUESTION = {
    "type": "choice",
    "instructions": "Which fulfilment option should be used for this order?",
    "criteria": {
        "express": "Ship within 24 hours by courier",
        "standard": "Ship within 5 business days",
        "pickup": "Customer collects from the store",
        "backorder": "Wait for restock, then ship standard",
    },
}


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(BACKBONE_NAME)


@pytest.fixture(scope="module")
def encoded(tokenizer):
    data_cfg_limits = (1024, 64, 256)
    enc = DeciderTokenizer(tokenizer, *data_cfg_limits)
    return enc.encode_question(STATE, QUESTION)


def _forward(backbone, input_ids, mask_bool, pos):
    return backbone(input_ids, to_additive_mask(mask_bool.unsqueeze(0)), pos.unsqueeze(0))


def _joint_and_separate(backbone, encoded, device):
    """Return (joint <opt_end> states, separate <opt_end> states, joint prefix states)."""
    packed = pack_single(encoded, device)
    with torch.no_grad():
        h_joint = backbone(
            packed["input_ids"], packed["attention_mask"], packed["position_ids"]
        )[0]
        P = packed["prefix_len"]
        sep_ends = []
        for block in encoded.option_blocks:
            ids_i = torch.tensor([encoded.prefix_ids + block], dtype=torch.long, device=device)
            mask_i, pos_i, _ = build_mask_and_pos(P, [len(block)], device=device)
            h_i = _forward(backbone, ids_i, mask_i, pos_i)[0]
            sep_ends.append(h_i[-1])
    ends = packed["end_indices"]
    return h_joint[ends], torch.stack(sep_ends), h_joint[:P]


@pytest.mark.slow
def test_joint_vs_separate_real_backbone(tokenizer, encoded):
    """The real Qwen3-0.6B (20 layers, LoRA as configured) under sdpa."""
    model_cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = TruncatedQwen3Backbone.from_pretrained(model_cfg, tokenizer=tokenizer, apply_lora=True)
    backbone = backbone.to(device).eval()
    assert len(backbone._base_model().layers) == 20
    assert any("lora_" in n for n, _ in backbone.named_parameters()), "LoRA must be attached"

    joint, separate, prefix = _joint_and_separate(backbone, encoded, device)
    assert joint.shape == separate.shape
    torch.testing.assert_close(joint, separate, atol=1e-4, rtol=0.0)
    # prefix states must not depend on the options at all
    for block in encoded.option_blocks:
        ids_i = torch.tensor([encoded.prefix_ids + block], dtype=torch.long, device=device)
        mask_i, pos_i, _ = build_mask_and_pos(encoded.prefix_len, [len(block)], device=device)
        with torch.no_grad():
            h_i = _forward(backbone, ids_i, mask_i, pos_i)[0]
        torch.testing.assert_close(prefix, h_i[: encoded.prefix_len], atol=1e-4, rtol=0.0)


def test_joint_vs_separate_tiny_config(tokenizer, encoded):
    """Fast CPU check with a random tiny Qwen3 — same mask/pos machinery."""
    from transformers import Qwen3Config

    cfg = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=len(tokenizer),
    )
    model_cfg = ModelConfig(num_layers=4, lora_layers=[2, 3], lora_r=4, lora_alpha=8)
    backbone = TruncatedQwen3Backbone.from_qwen3_config(cfg, model_cfg, apply_lora=True).eval()

    joint, separate, prefix = _joint_and_separate(backbone, encoded, torch.device("cpu"))
    torch.testing.assert_close(joint, separate, atol=1e-5, rtol=0.0)
    for block in encoded.option_blocks:
        ids_i = torch.tensor([encoded.prefix_ids + block], dtype=torch.long)
        mask_i, pos_i, _ = build_mask_and_pos(encoded.prefix_len, [len(block)])
        with torch.no_grad():
            h_i = _forward(backbone, ids_i, mask_i, pos_i)[0]
        torch.testing.assert_close(prefix, h_i[: encoded.prefix_len], atol=1e-5, rtol=0.0)
