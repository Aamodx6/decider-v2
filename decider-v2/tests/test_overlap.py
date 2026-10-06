"""Step 7 acceptance: state normalization and train<->test overlap report."""

import json
from pathlib import Path

import pandas as pd
import pytest

from decider.config import Config
from train.data.adapters import normalize_state, state_overlap


def _write_splits(tmp_path: Path, train_states: list[str], test_states: list[str]) -> Path:
    question = {"type": "noul", "instructions": "i", "label": True}

    def rows(states: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "state": s,
                    "questions_json": json.dumps({"q": question}),
                    "domain": "d",
                    "question_types": ["noul"],
                    "question_count": 1,
                }
                for s in states
            ]
        )

    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    rows(train_states).to_parquet(d / "train.parquet")
    rows(test_states).to_parquet(d / "test.parquet")
    return tmp_path


def test_normalize_state_rules():
    assert normalize_state("  Hello   World  ") == "hello world"
    assert normalize_state("A\u00a0B") == "a b"  # NBSP -> space via NFKC
    assert normalize_state("Ｆｕｌｌ-width") == "full-width"  # NFKC compatibility fold
    assert normalize_state("Keep punctuation! Yes?") == "keep punctuation! yes?"


@pytest.fixture()
def overlap_cfg(tmp_path: Path) -> Config:
    _write_splits(
        tmp_path,
        train_states=["Same Text", "unique train", "  THIRD   state "],
        test_states=["same text", "unique test"],
    )
    c = Config()
    c.data.local_data_dir = str(tmp_path)
    return c


def test_overlap_counts_and_report(overlap_cfg: Config, capsys):
    r = state_overlap(overlap_cfg)
    assert r["train_states"] == 3
    assert r["test_states"] == 2
    assert r["overlap"] == 1  # "Same Text" vs "same text" collapse together
    out = capsys.readouterr().out
    assert "overlap=1" in out


def test_overlap_zero_when_disjoint(tmp_path: Path):
    _write_splits(tmp_path, train_states=["a"], test_states=["b"])
    c = Config()
    c.data.local_data_dir = str(tmp_path)
    assert state_overlap(c)["overlap"] == 0


def test_no_local_dir_reports_hub_error():
    """The overlap report needs both parquet files; the Hub path is not
    unit-tested, but a missing local dir must fail loudly, not return 0."""
    c = Config()
    c.data.local_data_dir = str(Path("definitely/not/here"))
    with pytest.raises(Exception):
        state_overlap(c)
