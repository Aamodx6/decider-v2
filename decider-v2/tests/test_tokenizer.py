"""Step 2 acceptance: special tokens, serialization, truncation behaviour."""

import pytest

from decider.tokenizer import (
    DeciderTokenizer,
    SPECIAL_TOKENS,
    add_special_tokens,
    as_text,
    load_tokenizer,
    option_strings_and_keys,
    resize_and_init,
)
from decider.config import DataConfig

BACKBONE = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(BACKBONE)


@pytest.fixture(scope="module")
def enc(tokenizer):
    d = DataConfig()
    return DeciderTokenizer(tokenizer, d.max_state_tokens, d.max_option_tokens, d.max_question_tokens)


def test_special_tokens_map_to_single_ids(tokenizer):
    for token in SPECIAL_TOKENS:
        assert tokenizer.encode(token, add_special_tokens=False) == [tokenizer.convert_tokens_to_ids(token)]
        # and they are recognized as special tokens inside longer text
        assert len(tokenizer.encode(f"x {token} y", add_special_tokens=False)) == 4
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
    assert len(set(ids.values())) == len(SPECIAL_TOKENS), "special tokens must have distinct ids"


def test_state_and_question_block_layout(enc):
    prefix_ids, blocks, keys = enc.serialize("hello world", {"type": "noul", "instructions": "is it true?"})
    ids = enc.ids
    assert prefix_ids[0] == ids["<state>"]
    assert prefix_ids[-1] == ids["</q>"]
    assert prefix_ids.count(ids["<state>"]) == 1
    assert ids["</state>"] in prefix_ids and ids["<q>"] in prefix_ids
    # question open tag = <q> special token + ' type="noul">' plain text
    q_pos = prefix_ids.index(ids["<q>"])
    assert enc.decode(prefix_ids[q_pos:]) == '<q> type="noul">is it true?</q>'
    # noul always yields exactly two options, keys in spec order
    assert len(blocks) == 2 and keys == ["true", "false"]


def test_dict_states_serialize_deterministically(enc):
    q = {"type": "choice", "instructions": "pick", "criteria": {"a": "first", "b": "second"}}
    p1, b1, k1 = enc.serialize({"z": 1, "a": [1, 2], "nested": {"y": 0, "x": 1}}, q)
    p2, b2, k2 = enc.serialize({"a": [1, 2], "nested": {"x": 1, "y": 0}, "z": 1}, q)
    assert p1 == p2 and b1 == b2 and k1 == k2, "key order must not change the token ids"
    # and the state is rendered as canonical compact JSON
    assert as_text({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'


def test_option_truncation_preserves_opt_end(enc):
    long_desc = "word " * 500
    q = {"type": "choice", "instructions": "pick", "criteria": {"a": long_desc}}
    prefix_ids, blocks, keys = enc.serialize("s", q)
    block = blocks[0]
    assert len(block) == 2 + enc.max_option_tokens, "<opt> + 64 content tokens + <opt_end>"
    assert block[0] == enc.ids["<opt>"] and block[-1] == enc.ids["<opt_end>"]
    # short option keeps its text intact
    _, blocks2, _ = enc.serialize("s", {"type": "choice", "instructions": "p", "criteria": {"b": "hi"}})
    assert len(blocks2[0]) == 2 + len(enc._encode("b: hi")) and blocks2[0][-1] == enc.ids["<opt_end>"]


def test_empty_description_option(enc):
    _, blocks, keys = enc.serialize("s", {"type": "choice", "instructions": "p", "criteria": {"a": "", "b": "x"}})
    assert keys == ["a", "b"]
    assert blocks[0] == [enc.ids["<opt>"]] + enc._encode("a") + [enc.ids["<opt_end>"]]


def test_noul_variants(enc):
    # default text
    _, blocks, keys = enc.serialize("s", {"type": "noul", "instructions": "i"})
    assert keys == ["true", "false"] and len(blocks) == 2
    assert blocks[0] == [enc.ids["<opt>"]] + enc._encode("true") + [enc.ids["<opt_end>"]]
    # with criteria descriptions
    _, blocks, keys = enc.serialize(
        "s", {"type": "noul", "instructions": "i", "criteria": {"true": "yes because", "false": "no because"}}
    )
    assert keys == ["true", "false"] and len(blocks) == 2
    assert enc._encode("true: yes because") == blocks[0][1:-1]


def test_score_options_are_zero_indexed_levels(enc):
    q = {"type": "score", "instructions": "rate anger", "criteria": ["calm", "annoyed", "furious"]}
    _, blocks, keys = enc.serialize("s", q)
    assert keys == ["0", "1", "2"]
    assert blocks[1] == [enc.ids["<opt>"]] + enc._encode("1: annoyed") + [enc.ids["<opt_end>"]]


def test_state_and_question_truncation(enc):
    big_state = {f"k{i}": "v" * 50 for i in range(200)}
    q = {"type": "noul", "instructions": "why? " * 300}
    prefix_ids, _, _ = enc.serialize(big_state, q)
    ids = enc.ids
    state_start, state_end = prefix_ids.index(ids["<state>"]), prefix_ids.index(ids["</state>"])
    assert state_end - state_start - 1 == enc.max_state_tokens  # truncated from the end
    q_start, q_end = prefix_ids.index(ids["<q>"]), prefix_ids.index(ids["</q>"])
    assert q_end - q_start - 1 - len(enc._encode(' type="noul">')) == enc.max_question_tokens


def test_unknown_type_raises(enc):
    with pytest.raises(ValueError, match="unknown question type"):
        enc.serialize("s", {"type": "mood", "instructions": "i"})


def test_choice_requires_criteria(enc):
    with pytest.raises(ValueError, match="criteria"):
        enc.serialize("s", {"type": "choice", "instructions": "i", "criteria": {}})


def test_option_strings_and_keys_types():
    assert option_strings_and_keys({"type": "noul", "instructions": ""})[1] == ["true", "false"]


def test_resize_and_init(tokenizer):
    import torch

    from transformers import Qwen3Config, Qwen3Model

    cfg = Qwen3Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=len(tokenizer) - 6, head_dim=8,
    )
    model = Qwen3Model(cfg)
    old_n = model.get_input_embeddings().weight.shape[0]
    resize_and_init(model, tokenizer)
    emb = model.get_input_embeddings().weight
    assert emb.shape[0] == len(tokenizer)
    # new rows equal the mean of the original rows
    mean = model.get_input_embeddings().weight[:old_n].mean(dim=0)
    assert torch.allclose(emb[old_n:], mean.expand(len(tokenizer) - old_n, -1), atol=1e-6)
    # tied output embeddings share storage
    out = model.get_output_embeddings()
    assert out is None or out.weight.shape[0] == len(tokenizer)
