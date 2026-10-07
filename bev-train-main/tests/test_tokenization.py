"""Mask and position-id facts on a toy example (no model needed)."""
import torch

from tokenization import encode_example, tokenizer

STATE = "The customer ordered on 2026-03-15. The warranty was valid through 2026-03-14."
INSTRUCTIONS = "Which statement applies to this order?"
CHOICES = [
    "covered: The order is covered by the warranty",
    "expired: The warranty had already expired",
    "unknown: There is not enough information",
    "longer_option: The order is covered because the warranty lasts exactly one full year after purchase",
]

END_TOKEN_LEN = len(tokenizer.encode("</option>"))


def build(task_type="choice", choices=CHOICES):
    return encode_example(STATE, INSTRUCTIONS, choices, task_type, 128, 32)


def option_spans(example):
    """(start, end) token indices of each option block, reconstructed from choice_read_idx."""
    spans = []
    prev_end = example["question_end_idx"]
    for read_idx in example["choice_read_idx"]:
        end = read_idx + END_TOKEN_LEN
        spans.append((prev_end + 1, end))
        prev_end = end
    return spans


def test_options_share_start_position_id():
    example = build()
    pos, mask = example["position_ids"], example["attention_mask"]
    qlen = example["question_end_idx"] + 1
    spans = option_spans(example)
    # first option starts right after the question, and every option's position VALUE is the same
    assert spans[0][0] == qlen
    assert len({pos[start] for start, _ in spans}) == 1
    assert pos[spans[0][0]] == qlen
    # question occupies positions 0..qlen-1
    assert pos[:qlen] == list(range(qlen))


def test_options_cannot_see_each_other_but_see_question_and_themselves():
    example = build()
    mask, qlen = example["attention_mask"], example["question_end_idx"] + 1
    spans = option_spans(example)
    for i, (start_i, end_i) in enumerate(spans):
        # sees the whole question, and its own tokens causally
        assert mask[start_i:end_i + 1, :qlen].all()
        assert torch.equal(mask[start_i:end_i + 1, start_i:end_i + 1],
                           torch.tril(torch.ones(end_i - start_i + 1, end_i - start_i + 1)))
        # sees nothing of any other option (and never the answer)
        for j, (start_j, end_j) in enumerate(spans):
            if i != j:
                assert not mask[start_i:end_i + 1, start_j:end_j + 1].any()


def test_answer_sees_everything_and_starts_after_longest_option():
    example = build()
    pos, mask = example["position_ids"], example["attention_mask"]
    answer_start = option_spans(example)[-1][1] + 1
    assert example["answer_end_idx"] == len(pos) - 1
    # answer tokens attend to every token before them, causally among themselves
    assert mask[answer_start:, :answer_start].all()
    assert torch.equal(mask[answer_start:, answer_start:],
                       torch.tril(torch.ones(len(pos) - answer_start, len(pos) - answer_start)))
    # no option token attends to the answer
    assert not mask[:answer_start, answer_start:].any()
    # answer positions start at max(position_ids) + 1 (right after the longest option)
    assert pos[answer_start] == max(pos[:answer_start]) + 1
