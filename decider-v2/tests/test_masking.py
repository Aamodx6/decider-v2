"""Step 3 acceptance: mask/position construction, option isolation, padding safety."""

import torch

from decider.masking import build_batched_mask_and_pos, build_mask_and_pos, to_additive_mask


def test_hand_checked_small_case():
    P, opt_lens = 3, [2, 3]
    mask, pos, ends = build_mask_and_pos(P, opt_lens)
    T = 8
    assert mask.shape == (T, T) and mask.dtype == torch.bool
    assert pos.shape == (T,) and pos.dtype == torch.long
    assert ends.shape == (2,) and ends.dtype == torch.long
    assert ends.tolist() == [4, 7]

    # prefix is ordinary causal and blind to the options
    assert mask[:P, :P].equal(torch.tril(torch.ones(P, P, dtype=torch.bool)))
    assert not mask[:P, P:].any()

    # option 1 rows 3..4: prefix + own earlier tokens
    assert mask[3, :4].tolist() == [True] * 4 and not mask[3, 4:].any()
    assert mask[4, :5].tolist() == [True] * 5 and not mask[4, 5:].any()

    # option 2 rows 5..7: prefix + own earlier tokens, never option 1's tokens
    assert mask[5, [0, 1, 2, 5]].tolist() == [True] * 4 and not mask[5, [3, 4]].any()
    assert mask[6, [0, 1, 2, 5, 6]].tolist() == [True] * 5 and not mask[6, [3, 4]].any()
    assert mask[7, [0, 1, 2, 5, 6, 7]].tolist() == [True] * 6 and not mask[7, [3, 4]].any()

    # every option starts at position P
    assert pos.tolist() == [0, 1, 2, 3, 4, 3, 4, 5]


def test_no_option_attends_another_option():
    P, opt_lens = 5, [4, 1, 6, 2]
    mask, _, _ = build_mask_and_pos(P, opt_lens)
    bounds, start = [], P
    for l in opt_lens:
        bounds.append((start, start + l))
        start += l
    for i, (s1, e1) in enumerate(bounds):
        for j, (s2, e2) in enumerate(bounds):
            if i == j:
                continue
            assert not mask[s1:e1, s2:e2].any(), f"option {i} attends option {j}"


def test_all_options_share_start_position():
    P, opt_lens = 7, [1, 2, 5, 3]
    _, pos, _ = build_mask_and_pos(P, opt_lens)
    start = P
    for l in opt_lens:
        assert pos[start] == P
        assert (pos[start : start + l] == torch.arange(P, P + l)).all()
        start += l


def test_batched_shapes_and_padding():
    prefix_lens = [3, 4]
    opt_lens = [[2, 3], [2]]
    m = build_batched_mask_and_pos(prefix_lens, opt_lens)
    T, B, Kmax = 8, 2, 2
    assert m.attn_mask.shape == (B, T, T) and m.attn_mask.dtype == torch.bool
    assert m.position_ids.shape == (B, T) and m.position_ids.dtype == torch.long
    assert m.valid.shape == (B, T) and m.valid.dtype == torch.bool
    assert m.end_indices.shape == (B, Kmax) and m.option_mask.shape == (B, Kmax)

    # example 0 matches the single-example construction exactly
    mask0, pos0, ends0 = build_mask_and_pos(3, [2, 3])
    assert m.attn_mask[0, :8, :8].equal(mask0)
    assert m.position_ids[0, :8].equal(pos0)
    assert m.end_indices[0].tolist() == ends0.tolist()
    assert m.option_mask[0].tolist() == [True, True]

    # example 1 (len 6) is padded at rows/cols 6..7; option slot 1 is padding
    assert m.valid[1].tolist() == [True] * 6 + [False, False]
    assert not m.attn_mask[1, :6, 6:].any(), "real rows must not attend padding columns"
    for row in (6, 7):
        assert m.attn_mask[1, row].sum() == 1 and m.attn_mask[1, row, row], "pad rows self-attend"
    assert m.end_indices[1, 0] == 5  # 4 prefix + 2 option tokens - 1
    assert m.option_mask[1].tolist() == [True, False]
    # padding rows carry position 0 and are marked invalid
    assert (m.position_ids[1, 6:] == 0).all()


def test_padding_rows_cannot_produce_nans():
    m = build_batched_mask_and_pos([3, 4], [[2, 3], [2]])
    additive = to_additive_mask(m.attn_mask)
    assert additive.shape == (2, 1, 8, 8) and additive.dtype == torch.float32
    # simulate the attention softmax over the additive mask
    probs = torch.softmax(additive, dim=-1)
    assert torch.isfinite(probs).all()
    # masked entries get exactly zero mass
    masked = ~m.attn_mask.unsqueeze(1)
    assert (probs.masked_fill(masked, 0.0) == probs).all()


def test_single_example_with_no_options():
    mask, pos, ends = build_mask_and_pos(4, [])
    assert mask.shape == (4, 4) and ends.numel() == 0
    assert mask[:4, :4].equal(torch.tril(torch.ones(4, 4, dtype=torch.bool)))
    assert (pos == torch.arange(4)).all()
