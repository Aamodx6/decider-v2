"""Step 5 acceptance: head equivariance, padding behaviour, shapes, v2 flags."""

import pytest
import torch

from decider.config import ModelConfig
from decider.head import DecisionHead


@pytest.fixture(scope="module")
def head():
    torch.manual_seed(0)
    cfg = ModelConfig(head_dim=32, head_heads=4, head_layers=2)
    return DecisionHead(d_model=64, model_cfg=cfg).eval()


def _run(head, K, B=2, seed=0, pad_last=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, K, head.d_model, generator=g)
    pad = torch.zeros(B, K, dtype=torch.bool)
    if pad_last:
        pad[:, K - pad_last:] = True
    types = torch.randint(0, 3, (B,), generator=g)
    with torch.no_grad():
        return head(x, pad, types), x, pad, types


def test_output_shapes_k2_to_12(head):
    for K in range(2, 13):
        logits, _, pad, _ = _run(head, K, pad_last=1 if K > 3 else 0)
        assert logits.shape == (2, K)
        assert torch.isfinite(logits.masked_fill(pad, 0.0)).all()


def test_permutation_equivariance(head):
    torch.manual_seed(1)
    for K, pad_last in [(6, 2), (5, 0), (8, 3)]:
        logits, x, pad, types = _run(head, K, B=3, seed=42, pad_last=pad_last)
        perm = torch.randperm(K)
        with torch.no_grad():
            logits_perm = head(x[:, perm], pad[:, perm], types)
        torch.testing.assert_close(logits[:, perm], logits_perm, atol=1e-5, rtol=0.0)


def test_double_permutation_is_identity(head):
    """Two independent permutations compose; output must still match per-slot."""
    logits, x, pad, types = _run(head, 7, B=2, seed=7, pad_last=2)
    p1 = torch.randperm(7, generator=torch.Generator().manual_seed(11))
    p2 = torch.randperm(7, generator=torch.Generator().manual_seed(12))
    perm = p2[p1]
    with torch.no_grad():
        out = head(x[:, perm], pad[:, perm], types)
    torch.testing.assert_close(logits[:, perm], out, atol=1e-5, rtol=0.0)


def test_padded_options_get_no_probability_mass(head):
    logits, _, pad, _ = _run(head, 8, B=3, seed=3, pad_last=4)
    probs = torch.softmax(logits, dim=-1)
    assert (probs.masked_fill(pad, 0.0) == probs).all(), "padded slots carry probability mass"
    assert torch.isclose(probs.sum(-1), torch.ones(3), atol=1e-6).all()
    assert (probs[~pad] > 0).all()


def test_v2_flags_raise_not_implemented():
    for flag in ("use_ctx_cross_attn", "use_null_option", "use_rank_emb"):
        cfg = ModelConfig(head_dim=32, head_heads=4, **{flag: True})
        with pytest.raises(NotImplementedError, match=flag):
            DecisionHead(d_model=64, model_cfg=cfg)


def test_realistic_dims_forward():
    torch.manual_seed(2)
    cfg = ModelConfig()  # 1024 -> 512, 8 heads, 2 blocks
    head = DecisionHead(d_model=1024, model_cfg=cfg).eval()
    x = torch.randn(2, 10, 1024)
    pad = torch.zeros(2, 10, dtype=torch.bool)
    pad[1, 7:] = True
    types = torch.tensor([0, 2])
    with torch.no_grad():
        logits = head(x, pad, types)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits.masked_fill(pad, 0.0)).all()
    # exact param count as an architecture regression check:
    #   proj_opt 1024*512+512 = 524_800 | type_emb 3*512 = 1_536
    #   per block: in_proj 3*(512*512+512) = 787_968
    #              out_proj 512*512+512    = 262_656
    #              ffn 512*2048+2048 + 2048*512+512 = 2_099_712
    #              norms 4*512 = 2_048            -> block 3_152_384
    #   out: norm 1_024 + linear 513 = 1_537
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params == 524_800 + 1_536 + 2 * 3_152_384 + 1_537 == 6_832_641
