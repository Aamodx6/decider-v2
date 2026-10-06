"""Decision head v1: order-invariant self-attention over option vectors.

Architecture (bev-decider-faithful v1, ARCH.md §7):

    option vectors [B, K, d_model]
      -> proj_opt (d_model -> head_dim)
      + task-type embedding (choice=0, noul=1, score=2) added to every option
      -> head_layers x HeadBlock (pre-LN multi-head self-attention across
         options with key_padding_mask, FFN 4x, NO positional information)
      -> LayerNorm -> Linear -> 1 logit per option
      -> padded option slots forced to -inf

Because the head has no positional information and every block is
permutation-equivariant over the option axis, permuting the options permutes
the logits identically (hard gate: tests/test_invariance.py).

The head always runs in fp32: the model disables autocast around it (same
convention as bev-decider).

v2 features (context cross-attention, null option, rank embedding) are NOT
implemented in M0; their config flags exist and default to off, and enabling
them raises NotImplementedError.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TASK_TYPE_IDS, ModelConfig


def _reject_v2_flags(model_cfg: ModelConfig) -> None:
    enabled = []
    if model_cfg.use_ctx_cross_attn:
        enabled.append("model.use_ctx_cross_attn")
    if model_cfg.use_null_option:
        enabled.append("model.use_null_option")
    if model_cfg.use_rank_emb:
        enabled.append("model.use_rank_emb")
    if enabled:
        raise NotImplementedError(
            "v2 head features are not implemented in M0 (planned for M2, ARCH.md §16): "
            f"{', '.join(enabled)} must stay false in configs/m0.yaml"
        )


class HeadBlock(nn.Module):
    """Pre-LN self-attention over the option axis + token-wise FFN (4x)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        return x + self.ff(self.norm2(x))


class DecisionHead(nn.Module):
    """Scores each option with 1 logit; padded options get -inf."""

    def __init__(self, d_model: int, model_cfg: ModelConfig) -> None:
        super().__init__()
        _reject_v2_flags(model_cfg)
        d, h = model_cfg.head_dim, model_cfg.head_heads
        self.d_model = d_model
        self.head_dim = d
        self.num_task_types = len(TASK_TYPE_IDS)
        self.proj_opt = nn.Linear(d_model, d)
        self.type_emb = nn.Embedding(self.num_task_types, d)
        self.blocks = nn.ModuleList(HeadBlock(d, h) for _ in range(model_cfg.head_layers))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(
        self,
        option_vectors: torch.Tensor,
        option_padding: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        option_vectors : [B, K, d_model] backbone hidden states per option
        option_padding : [B, K] bool, True = padded slot (no real option)
        type_ids       : [B] long task-type ids
        returns        : [B, K] logits; padded slots are -inf
        """
        if option_padding.any():
            real = (~option_padding).sum(dim=1)
            if (real == 0).any():
                raise ValueError("every example needs at least one real option")
        x = self.proj_opt(option_vectors) + self.type_emb(type_ids).unsqueeze(1)
        for block in self.blocks:
            x = block(x, option_padding)
        logits = self.out(x).squeeze(-1)  # [B, K]
        return logits.masked_fill(option_padding, float("-inf"))
