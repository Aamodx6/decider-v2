"""Step 6 acceptance: dataset adapters — parsing, labels, filters, splits.

Unit tests use synthetic parquet fixtures; the two integration tests read the
real local clone (skipped when it is not present) so the adapter is exercised
against the actual bev-decision files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from decider.config import Config
from decider.tokenizer import DeciderTokenizer, load_tokenizer
from train.data.adapters import (
    Example,
    _check_label,
    _dev_split,
    _option_text,
    _parse_questions,
    load_examples,
    load_test,
    load_train_dev,
    subsample_stratified,
    target_index,
    type_counts,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA = PROJECT_ROOT.parent / "bev-decision"


# ----------------------------------------------------------------- fixtures
def _row(state: str, questions: dict, domain: str = "test-domain") -> dict:
    return {
        "state": state,
        "questions_json": json.dumps(questions),
        "domain": domain,
        "question_types": [q["type"] for q in questions.values()],
        "question_count": len(questions),
    }


CHOICE_Q = {
    "type": "choice",
    "instructions": "Pick one",
    "criteria": {"a": "first", "b": "second", "c": "third"},
    "label": "b",
}
NOUL_Q = {"type": "noul", "instructions": "True?", "label": True}
SCORE_Q = {
    "type": "score",
    "instructions": "Rate it",
    "criteria": ["calm", "annoyed", "furious"],
    "label": 2,
}


@pytest.fixture()
def parquet_dir(tmp_path: Path) -> Path:
    train = pd.DataFrame(
        [
            _row("state one", {"q1": CHOICE_Q, "q2": NOUL_Q}),
            _row("state two", {"q3": SCORE_Q}),
            _row("state three", {"q4": CHOICE_Q}),
            _row("state four", {"q5": NOUL_Q, "q6": SCORE_Q}),
        ]
    )
    test = pd.DataFrame([_row("state five", {"q7": CHOICE_Q}), _row("state six", {"q8": NOUL_Q})])
    d = tmp_path / "data"
    d.mkdir()
    train.to_parquet(d / "train.parquet")
    test.to_parquet(d / "test.parquet")
    return tmp_path


@pytest.fixture()
def cfg(parquet_dir: Path) -> Config:
    c = Config()
    c.data.local_data_dir = str(parquet_dir)
    c.data.dev_size = 2
    return c


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer("Qwen/Qwen3-0.6B")


def _tok(cfg: Config, tokenizer) -> DeciderTokenizer:
    return DeciderTokenizer(
        tokenizer, cfg.data.max_state_tokens, cfg.data.max_option_tokens, cfg.data.max_question_tokens
    )


# ------------------------------------------------------------ parsing/labels
def test_parse_questions_returns_file_order():
    qd = {"z": CHOICE_Q, "a": NOUL_Q}
    assert [n for n, _ in _parse_questions({"questions_json": json.dumps(qd)})] == ["z", "a"]


def test_check_label_accepts_well_typed_labels():
    _check_label("choice", "b", CHOICE_Q["criteria"], "q")
    _check_label("noul", False, None, "q")
    _check_label("score", 1, SCORE_Q["criteria"], "q")


def test_check_label_rejects_malformed():
    with pytest.raises(ValueError, match="not a criteria key"):
        _check_label("choice", "zz", CHOICE_Q["criteria"], "q")
    with pytest.raises(ValueError, match="must be str"):
        _check_label("choice", 1, CHOICE_Q["criteria"], "q")
    with pytest.raises(ValueError, match="must be bool"):
        _check_label("noul", 1, None, "q")
    with pytest.raises(ValueError, match="out of range"):
        _check_label("score", 5, SCORE_Q["criteria"], "q")
    with pytest.raises(ValueError, match="unknown question type"):
        _check_label("mood", 1, None, "q")


def test_option_text_picks_longest():
    q = {"type": "choice", "criteria": {"a": "short", "b": "much longer text here"}}
    assert _option_text(q) == "b: much longer text here"
    # "1: annoyed"/"2: furious" tie in length; use an unambiguous longest level
    score_q = {"type": "score", "criteria": ["a", "bbbb", "cc"]}
    assert _option_text(score_q) == "1: bbbb"
    assert _option_text(NOUL_Q) is None  # no criteria -> nothing to check


# ------------------------------------------------------------------ loading
def test_load_examples_expands_rows_to_questions(cfg, tokenizer):
    train, dev = load_train_dev(cfg, tokenizer=_tok(cfg, tokenizer))
    all_train_dev = train + dev
    # 4 train rows: 2 + 1 + 1 + 2 = 6 questions
    assert len(all_train_dev) == 6
    assert {e.split for e in all_train_dev} <= {"train", "dev"}
    # domain/question_name provenance survives
    by_name = {e.question_name: e for e in all_train_dev}
    assert by_name["q1"].domain == "test-domain"
    assert by_name["q1"].label == "b"
    assert by_name["q2"].label is True


def test_train_dev_are_disjoint_and_deterministic(cfg, tokenizer):
    tok = _tok(cfg, tokenizer)
    t1, d1 = load_train_dev(cfg, tokenizer=tok)
    t2, d2 = load_train_dev(cfg, tokenizer=tok)
    assert {(e.state, e.question_name) for e in d1} == {(e.state, e.question_name) for e in d2}
    assert len(d1) == cfg.data.dev_size or len(d1) == cfg.data.dev_size  # cap respected
    train_keys = {(e.state, e.question_name) for e in t1}
    dev_keys = {(e.state, e.question_name) for e in d1}
    assert not train_keys & dev_keys
    assert all(e.split == "dev" for e in d1) and all(e.split == "train" for e in t1)


def test_test_split_never_carves_dev(cfg, tokenizer):
    test = load_test(cfg, tokenizer=_tok(cfg, tokenizer))
    assert len(test) == 2
    assert all(e.split == "test" for e in test)
    assert type_counts(test) == {"choice": 1, "noul": 1, "score": 0}


def test_filter_option_count(cfg, tokenizer, parquet_dir: Path):
    huge = {"type": "choice", "instructions": "i", "criteria": {f"k{i}": "v" for i in range(20)}, "label": "k0"}
    single = {"type": "choice", "instructions": "i", "criteria": {"only": "v"}, "label": "only"}
    df = pd.DataFrame([_row("s", {"bad_big": huge, "bad_single": single, "ok": CHOICE_Q})])
    (parquet_dir / "data" / "train.parquet").unlink()
    df.to_parquet(parquet_dir / "data" / "train.parquet")
    ex = load_examples(cfg, "train", tokenizer=_tok(cfg, tokenizer))
    assert [e.question_name for e in ex] == ["ok"]


def test_filter_long_option_tokens(cfg, tokenizer, parquet_dir: Path):
    long_q = {"type": "choice", "instructions": "i", "criteria": {"a": "word " * 100}, "label": "a"}
    df = pd.DataFrame([_row("s", {"long": long_q, "ok": CHOICE_Q})])
    (parquet_dir / "data" / "train.parquet").unlink()
    df.to_parquet(parquet_dir / "data" / "train.parquet")
    ex = load_examples(cfg, "train", tokenizer=_tok(cfg, tokenizer))
    assert [e.question_name for e in ex] == ["ok"]


def test_state_passes_through_verbatim(parquet_dir: Path, cfg: Config, tokenizer):
    """A JSON-looking state string must survive the adapter byte-for-byte.

    The tokenizer renders string states verbatim (as_text), so re-dumping the
    text through json.dumps would reorder/normalize it — that must not happen
    at the adapter layer.
    """
    raw = '{"b": 1, "a": 2, "note": "  padded  "}'  # not sorted-keys order
    df = pd.DataFrame([_row(raw, {"q": NOUL_Q})])
    (parquet_dir / "data" / "train.parquet").unlink()
    df.to_parquet(parquet_dir / "data" / "train.parquet")
    tok = _tok(cfg, tokenizer)
    examples = load_train_dev(cfg, tokenizer=tok)
    all_ex = [e for e in examples[0] + examples[1] if e.question_name == "q"]
    assert len(all_ex) == 1
    assert all_ex[0].state == raw  # verbatim, not re-dumped
    enc = tok.encode_question(all_ex[0].state, all_ex[0].question)
    # decoded prefix still contains the original byte order
    assert '{"b": 1, "a": 2' in tok.decode(enc.prefix_ids)


def test_examples_feed_the_tokenizer(cfg, tokenizer):
    """End-to-end: an Example's (state, question) must serialize losslessly."""
    tok = _tok(cfg, tokenizer)
    train, _ = load_train_dev(cfg, tokenizer=tok)
    e = next(e for e in train if e.question["type"] == "score")
    enc = tok.encode_question(e.state, e.question)
    assert enc.keys == [str(i) for i in range(len(e.question["criteria"]))]
    assert target_index(e) == int(e.label)
    assert target_index(e) < len(enc.keys)


def test_target_index_matches_criteria_order():
    e = Example(
        state="s", question=CHOICE_Q, label="c", question_name="q", domain="d", split="train"
    )
    assert target_index(e) == 2
    e_noul = Example(state="s", question={**NOUL_Q, "label": False}, label=False, question_name="q", domain="d", split="train")
    assert target_index(e_noul) == 1


# --------------------------------------------------------------- subsampling
def _mk(i: int, t: str) -> Example:
    return Example(state="s", question={"type": t}, label=0, question_name="q", domain="d", split="train")


def test_subsample_stratified_proportional_and_exact():
    exs = [_mk(i, "choice") for i in range(600)] + [_mk(i, "noul") for i in range(300)] + [_mk(i, "score") for i in range(100)]
    out = subsample_stratified(exs, 500, seed=0)
    assert len(out) == 500
    tc = type_counts(out)
    assert tc == {"choice": 300, "noul": 150, "score": 50}


def test_subsample_deterministic_and_interleaved():
    exs = [_mk(i, "choice") for i in range(60)] + [_mk(i, "score") for i in range(40)]
    a = subsample_stratified(exs, 50, seed=0)
    b = subsample_stratified(exs, 50, seed=0)
    assert a == b
    kinds = [e.question["type"] for e in a[:9]]
    assert len(set(kinds)) == 3 or len(set(kinds)) == 2  # interleaved, not type-blocked


def test_subsample_passthrough_when_small():
    exs = [_mk(i, "choice") for i in range(5)]
    assert subsample_stratified(exs, 100, seed=0) == exs


# ------------------------------------------------------------------ dev split
def test_dev_split_hits_target_exactly_and_has_no_head_bias():
    n, size = 50_000, 2_000
    c: Counter = Counter()
    hits = [_dev_split(f"s-{i}", "run", size, c, n_rows=n) for i in range(n)]
    assert sum(hits) == size
    assert sum(hits[: n // 4]) < 0.35 * size, "dev must not concentrate at the file head"


def test_dev_split_reproducible_and_run_name_sensitive():
    c1: Counter = Counter()
    c2: Counter = Counter()
    a = [_dev_split(f"s-{i}", "m0-small", 100, c1, n_rows=10_000) for i in range(10_000)]
    b = [_dev_split(f"s-{i}", "m0-small", 100, c2, n_rows=10_000) for i in range(10_000)]
    assert a == b
    c3: Counter = Counter()
    d = [_dev_split(f"s-{i}", "other-run", 100, c3, n_rows=10_000) for i in range(10_000)]
    assert d != a and sum(d) == 100


# ------------------------------------------------- real-data integration
@pytest.mark.skipif(not LOCAL_DATA.is_dir(), reason="local bev-decision clone not present")
def test_real_train_dev_carveout(tokenizer):
    """dev_size is a ROW budget (repeated states must not straddle train/dev),
    so the dev question count lands at roughly 2x dev_size."""
    cfg = Config()  # local_data_dir=../bev-decision, dev_size=2000 rows
    tok = _tok(cfg, tokenizer)
    train, dev = load_train_dev(cfg, tokenizer=tok)
    assert cfg.data.dev_size <= len(dev) <= 2 * cfg.data.dev_size
    assert len(train) > 100_000  # default config keeps the full mixture
    assert {e.split for e in train} == {"train"} and {e.split for e in dev} == {"dev"}
    # every state appears on exactly one side
    train_states = {e.state for e in train}
    dev_states = {e.state for e in dev}
    assert not train_states & dev_states
    tc = type_counts(train + dev)
    assert set(tc) == {"choice", "noul", "score"} and all(v > 0 for v in tc.values())


@pytest.mark.skipif(not LOCAL_DATA.is_dir(), reason="local bev-decision clone not present")
def test_real_subset_is_stratified_and_tokenizer_compatible(tokenizer):
    cfg = Config()
    tok = _tok(cfg, tokenizer)
    train, dev = load_train_dev(cfg, tokenizer=tok)
    subset = train + dev
    small = subsample_stratified(subset, cfg.data.train_small_size, seed=cfg.data.subset_seed)
    assert len(small) == cfg.data.train_small_size
    tc = type_counts(small)
    # proportional to the full mixture (choice 40.7% / noul 39.9% / score 19.4%)
    share = {t: tc[t] / len(small) for t in tc}
    assert share["choice"] == pytest.approx(0.407, abs=0.02)
    assert share["noul"] == pytest.approx(0.399, abs=0.02)
    assert share["score"] == pytest.approx(0.194, abs=0.02)
    # every kept example must serialize and carry a valid target index
    for e in small[:200]:
        enc = tok.encode_question(e.state, e.question)
        assert len(enc.option_blocks) == len(enc.keys)
        assert 0 <= target_index(e) < len(enc.keys)
