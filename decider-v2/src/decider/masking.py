"""Attention masks and position ids for the order-invariant option layout.

Single-example layout (ARCH.md §6):

    tokens:  [ prefix (P) ][ opt_1 (l1) ][ opt_2 (l2) ]...
    mask:    prefix rows     -> causal within prefix
             option rows     -> full prefix + own earlier tokens only
    pos:     prefix          -> 0..P-1
             every option    -> P..P+len-1 (all options share the same start P)

The batched version pads to [B, T, T] and marks padding with a [B, T] validity
mask. Padding rows attend only to themselves (the diagonal), which keeps every
softmax row non-empty so fully-masked rows cannot produce NaNs downstream;
padding columns are masked out for all real rows, so real tokens never read
padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # avoids a runtime cycle; tokenizer never imports masking
    from .tokenizer import EncodedQuestion

NEG_INF = float("-inf")


def build_mask_and_pos(P: int, opt_lens: list[int], device="cpu"):
    """Single-example mask, position ids and <opt_end> indices.

    P        : prefix length (state + question tokens)
    opt_lens : token length of each option block (incl. <opt> ... <opt_end>)
    returns  : bool mask [T, T] (True = may attend), position ids [T],
               <opt_end> indices [K]
    """
    T = P + sum(opt_lens)
    mask = torch.zeros(T, T, dtype=torch.bool, device=device)
    pos = torch.arange(T, device=device)

    # prefix: ordinary causal
    mask[:P, :P] = torch.tril(torch.ones(P, P, dtype=torch.bool, device=device))

    start, ends = P, []
    for l in opt_lens:
        mask[start : start + l, :P] = True  # sees full prefix
        mask[start : start + l, start : start + l] = torch.tril(
            torch.ones(l, l, dtype=torch.bool, device=device)
        )  # own earlier tokens only
        pos[start : start + l] = P + torch.arange(l, device=device)  # SAME start position
        ends.append(start + l - 1)  # <opt_end> index
        start += l
    return mask, pos, torch.tensor(ends, dtype=torch.long, device=device)


@dataclass
class BatchedMasks:
    """Padded mask/position tensors for a batch of examples."""

    attn_mask: torch.Tensor  # [B, T, T] bool (True = may attend)
    position_ids: torch.Tensor  # [B, T] long; padding rows carry position 0
    valid: torch.Tensor  # [B, T] bool; True = real token, False = padding
    end_indices: torch.Tensor  # [B, Kmax] long; <opt_end> index per option slot
    option_mask: torch.Tensor  # [B, Kmax] bool; True = real option slot


def build_batched_mask_and_pos(
    prefix_lens: list[int],
    opt_lens_per_example: list[list[int]],
    device="cpu",
) -> BatchedMasks:
    """Batched version of :func:`build_mask_and_pos` with padding.

    Every example keeps its internal layout; padding is appended at the end of
    the sequence. Padding rows are allowed to attend only to themselves and
    padding columns are closed to real rows, so the additive conversion in
    :func:`to_additive_mask` never yields a fully-masked softmax row.
    """
    B = len(prefix_lens)
    if B != len(opt_lens_per_example):
        raise ValueError("prefix_lens and opt_lens_per_example must have the same length")
    lengths = [P + sum(lens) for P, lens in zip(prefix_lens, opt_lens_per_example)]
    T = max(lengths)
    Kmax = max((len(lens) for lens in opt_lens_per_example), default=1)
    Kmax = max(Kmax, 1)

    attn_mask = torch.zeros(B, T, T, dtype=torch.bool, device=device)
    position_ids = torch.zeros(B, T, dtype=torch.long, device=device)
    valid = torch.zeros(B, T, dtype=torch.bool, device=device)
    end_indices = torch.zeros(B, Kmax, dtype=torch.long, device=device)
    option_mask = torch.zeros(B, Kmax, dtype=torch.bool, device=device)

    for b, (P, lens) in enumerate(zip(prefix_lens, opt_lens_per_example)):
        m, p, ends = build_mask_and_pos(P, lens, device=device)
        L = lengths[b]
        attn_mask[b, :L, :L] = m
        position_ids[b, :L] = p
        valid[b, :L] = True
        k = len(lens)
        end_indices[b, :k] = ends
        option_mask[b, :k] = True

    # Padding rows attend only to themselves (never all-masked -> no NaNs).
    for b, L in enumerate(lengths):
        if L < T:
            attn_mask[b, L:, L:] = torch.eye(T - L, dtype=torch.bool, device=device)

    return BatchedMasks(
        attn_mask=attn_mask,
        position_ids=position_ids,
        valid=valid,
        end_indices=end_indices,
        option_mask=option_mask,
    )


def pack_single(encoded: "EncodedQuestion", device="cpu") -> dict:
    """Pack one :class:`~decider.tokenizer.EncodedQuestion` into model inputs.

    Returns ``input_ids [1, T]``, ``attention_mask [1, 1, T, T]`` (float
    additive), ``position_ids [1, T]`` and ``end_indices [K]`` (the
    ``<opt_end>`` positions). This is the single source of truth used by the
    model, the collate function and the equivalence tests.
    """
    P = encoded.prefix_len
    option_lens = encoded.option_lens
    mask, pos, ends = build_mask_and_pos(P, option_lens, device=device)
    flat_options = [t for block in encoded.option_blocks for t in block]
    input_ids = torch.tensor(encoded.prefix_ids + flat_options, dtype=torch.long, device=device)
    return {
        "input_ids": input_ids.unsqueeze(0),
        "attention_mask": to_additive_mask(mask.unsqueeze(0)),
        "position_ids": pos.unsqueeze(0),
        "end_indices": ends,
        "prefix_len": P,
        "option_lens": list(option_lens),
    }


def to_additive_mask(attn_mask_bool: torch.Tensor) -> torch.Tensor:
    """[B, T, T] bool mask -> [B, 1, T, T] float additive mask (0.0 / -inf).

    This is the form consumed by the HF Qwen3 attention implementation. Every
    row of the input mask must contain at least one True (guaranteed by
    :func:`build_mask_and_pos` / :func:`build_batched_mask_and_pos`), so the
    softmax over any row has at least one finite entry.
    """
    if attn_mask_bool.ndim != 3:
        raise ValueError(f"expected [B, T, T] bool mask, got shape {tuple(attn_mask_bool.shape)}")
    additive = torch.zeros_like(attn_mask_bool, dtype=torch.float32)
    additive.masked_fill_(~attn_mask_bool, NEG_INF)
    return additive.unsqueeze(1)
