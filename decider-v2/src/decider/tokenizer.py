"""Special tokens and serialization for decider-v2 M0.

Token stream layout (ARCH.md §5.2), tokenized segment-wise so every special
token is exactly one id:

    prefix:  <state> {state} </state> <q>  type="..."> {instructions} </q>
    option:  <opt> {key}: {description} <opt_end>          (choice)
             <opt> true: {description} <opt_end>           (noul, or bare "true"/"false")
             <opt> {i}: {level_description} <opt_end>      (score)

- ``state`` may be a string or any JSON value; non-strings are rendered with
  ``json.dumps(..., ensure_ascii=False, sort_keys=True)`` so dict states
  serialize deterministically.
- The canonical rendering of the question open tag is ``<q type="...">``; since
  the tokenizer special token is ``<q>``, the open tag is tokenized as the
  ``<q>`` id followed by the plain text `` type="...">``. In decoded text the
  special token contributes its own ``>``, so the decoded prefix reads
  ``<state>...</state><q> type="...">{instructions}</q>`` — the same layout,
  with the open tag's ``>`` carried by the special token itself.
- Truncation (token-level, from the end): state -> ``max_state_tokens``,
  instructions -> ``max_question_tokens``, option text -> ``max_option_tokens``
  (the ``<opt>`` / ``<opt_end>`` ids are appended outside the cap, so the final
  ``<opt_end>`` is always preserved).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import TASK_TYPE_IDS, DataConfig

SPECIAL_TOKENS: tuple[str, ...] = ("<state>", "</state>", "<q>", "</q>", "<opt>", "<opt_end>")


def add_special_tokens(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Register the M0 special tokens on a tokenizer in place and return it."""
    added = tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    ids = special_token_ids(tokenizer)
    for name, token_id in ids.items():
        if isinstance(token_id, list):
            raise ValueError(f"special token {name!r} does not map to a single id")
    if added == 0 and len(ids) != len(SPECIAL_TOKENS):
        raise ValueError("special tokens missing from tokenizer")
    return tokenizer


def special_token_ids(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    """Map each special token to its single id (raises if any is missing)."""
    out: dict[str, int] = {}
    for token in SPECIAL_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"special token {token!r} maps to {encoded}, expected a single id")
        out[token] = encoded[0]
    return out


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Guarantee a pad id; Qwen3 base models often leave it unset.

    Padding only ever appears in masked-out positions, so reusing the eos id is
    safe (pad tokens attend only to themselves and are never read).
    """
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an eos token to reuse")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_tokenizer(name: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer and register the M0 special tokens on it."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    add_special_tokens(tokenizer)
    ensure_pad_token(tokenizer)
    return tokenizer


@torch.no_grad()
def resize_and_init(model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase) -> torch.nn.Module:
    """Resize the model's embeddings to fit ``tokenizer`` and mean-init new rows.

    New input-embedding rows are set to the mean of the existing rows. If the
    output (LM head) embedding exists and is untied, it is resized and
    initialized the same way. For the tied Qwen3-0.6B embeddings the output
    matrix shares storage and needs no extra pass.
    """
    in_emb = model.get_input_embeddings()
    old_n = in_emb.weight.shape[0]
    new_n = len(tokenizer)
    if new_n <= old_n:
        return model
    mean_vec = in_emb.weight.detach().float().mean(dim=0)
    model.resize_token_embeddings(new_n)
    in_emb = model.get_input_embeddings()
    in_emb.weight.data[old_n:] = mean_vec.to(in_emb.weight.dtype)
    out_emb = model.get_output_embeddings()
    if out_emb is not None and out_emb.weight.data_ptr() != in_emb.weight.data_ptr():
        out_emb.weight.data[old_n:] = mean_vec.to(out_emb.weight.dtype)
    return model


def as_text(value) -> str:
    """Render a state/instruction/description: strings pass through, JSON values
    are dumped deterministically (sorted keys, no ASCII escaping)."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def option_strings_and_keys(question: dict) -> tuple[list[str], list[str]]:
    """Jev-format question -> (option texts, probability keys), per ARCH.md §5.3.

    Ordering follows the input (criteria dict order / level order) and is
    stable; the model must be invariant to it anyway.
    """
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "choice":
        if not criteria or not isinstance(criteria, dict):
            raise ValueError("a choice question needs criteria: {option key: description}")
        keys = [str(k) for k in criteria]
        options = [f"{k}: {as_text(criteria[k])}" if criteria[k] else k for k in keys]
    elif kind == "score":
        if not criteria or not isinstance(criteria, (list, tuple)):
            raise ValueError("a score question needs criteria: a list of level descriptions")
        keys = [str(i) for i in range(len(criteria))]
        options = [f"{i}: {as_text(level)}" for i, level in enumerate(criteria)]
    elif kind == "noul":
        keys = ["true", "false"]
        if criteria and isinstance(criteria, dict):
            options = [
                f"true: {as_text(criteria['true'])}" if criteria.get("true") else "true",
                f"false: {as_text(criteria['false'])}" if criteria.get("false") else "false",
            ]
        else:
            options = ["true", "false"]
    else:
        raise ValueError(f"unknown question type {kind!r}; expected choice, noul or score")
    return options, keys


@dataclass
class EncodedQuestion:
    """One (state, question) pair turned into token ids."""

    type: str
    type_id: int
    prefix_ids: list[int]
    option_blocks: list[list[int]]  # each block: [<opt>] + content + [<opt_end>]
    keys: list[str]

    @property
    def prefix_len(self) -> int:
        return len(self.prefix_ids)

    @property
    def option_lens(self) -> list[int]:
        return [len(b) for b in self.option_blocks]


class DeciderTokenizer:
    """Wraps a Qwen3 tokenizer with the M0 serialization rules and limits."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_state_tokens: int,
        max_option_tokens: int,
        max_question_tokens: int,
    ) -> None:
        self.tok = tokenizer
        self.max_state_tokens = max_state_tokens
        self.max_option_tokens = max_option_tokens
        self.max_question_tokens = max_question_tokens
        self.ids = special_token_ids(tokenizer)
        ensure_pad_token(tokenizer)

    @classmethod
    def from_config(cls, data_cfg: DataConfig, tokenizer_name: str | None = None) -> "DeciderTokenizer":
        tok = load_tokenizer(tokenizer_name or "Qwen/Qwen3-0.6B")
        return cls(tok, data_cfg.max_state_tokens, data_cfg.max_option_tokens, data_cfg.max_question_tokens)

    # ------------------------------------------------------------------ helpers
    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _truncate_text(self, text: str, max_tokens: int) -> list[int]:
        return self._encode(text)[:max_tokens]

    # --------------------------------------------------------------- serialize
    def serialize(self, state, question: dict) -> tuple[list[int], list[list[int]], list[str]]:
        """(state, question) -> (prefix_ids, option blocks, probability keys).

        This is the ARCH.md §5.2/§5.3 serialization; ``option_blocks`` include
        the ``<opt>`` and ``<opt_end>`` ids so their lengths feed the mask
        builder directly.
        """
        kind = question.get("type")
        if kind not in TASK_TYPE_IDS:
            raise ValueError(f"unknown question type {kind!r}")
        state_text = as_text(state)
        state_ids = self._truncate_text(state_text, self.max_state_tokens)
        instructions = question.get("instructions", "")
        if not isinstance(instructions, str):
            instructions = as_text(instructions)
        instr_ids = self._truncate_text(instructions, self.max_question_tokens)

        ids = self.ids
        prefix_ids = (
            [ids["<state>"]]
            + state_ids
            + [ids["</state>"]]
            + [ids["<q>"]]
            + self._encode(f' type="{kind}">')
            + instr_ids
            + [ids["</q>"]]
        )

        texts, keys = option_strings_and_keys(question)
        option_blocks = [
            [ids["<opt>"]] + self._truncate_text(text, self.max_option_tokens) + [ids["<opt_end>"]]
            for text in texts
        ]
        return prefix_ids, option_blocks, keys

    def encode_question(self, state, question: dict) -> EncodedQuestion:
        prefix_ids, option_blocks, keys = self.serialize(state, question)
        return EncodedQuestion(
            type=question["type"],
            type_id=TASK_TYPE_IDS[question["type"]],
            prefix_ids=prefix_ids,
            option_blocks=option_blocks,
            keys=keys,
        )

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)
