"""Dataset → (state, question) adapters.

The bev-decision parquet files store one row per input state:

    state            str   raw state text (or JSON string; rendered verbatim)
    questions_json   str   JSON object: {question_name: {type, instructions, criteria?, label}}
    domain           str   primary domain label (kept for stratification and logging)
    question_types   list  per-row question types (denormalized, not re-derived)
    question_count   int   number of questions in the row

Each row expands into one :class:`Example` per question. `state` is passed
through as a raw string on purpose: bev states are rendered verbatim by the
tokenizer (``as_text`` only normalizes non-string values), so re-dumping the
text would risk changing the bytes the original model saw.

The dataset has train/test parquet splits and no validation split, so the dev
set is carved out of train deterministically: rows are hashed with
``sha1(run_name|state)`` and assigned to dev while the running dev count is
below ``dev_size``. The same state always lands on the same side for a given
``run_name``, and two different ``run_name`` values give independent, seeded
carve-outs (multiple independent dev sets, zero extra data loading).

Filtering (in order): option count outside ``[min_options, max_options]``,
questions whose option texts tokenize over ``max_option_tokens``, and — for
the fast-iteration subset only — everything not picked by the stratified
per-type subsample. A filtered example is dropped, never raised: the dataset
is large and every consumer can survive a smaller one.

Labels are checked here, once, at the adapter boundary — training and eval
code can assume a well-typed label.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from decider.config import Config
from decider.tokenizer import DeciderTokenizer

# Parquet columns consumed by this adapter; anything else is ignored.
REQUIRED_COLUMNS: tuple[str, ...] = ("state", "questions_json", "question_types", "question_count")
OPTIONAL_COLUMNS: tuple[str, ...] = ("domain",)


@dataclass
class Example:
    """One (state, question) pair with its supervision and provenance."""

    state: str
    question: dict  # {type, instructions, criteria?, label}; Jev format
    label: str | bool | int
    question_name: str
    domain: str
    split: str  # "train" | "dev" | "test"
    # token count of the longest option block ("" for noul without criteria);
    # filled by the tokenizer-based filters, 0 if no filter ran
    max_option_tokens_used: int = field(default=0)


def _resolve_parquet_path(cfg: Config, split: str) -> Path:
    """Local clone first (hermetic, offline), then the Hub cache.

    ``local_data_dir`` is relative to the decider-v2 project root (the
    repository default is ``../bev-decision``, i.e. a checkout next to
    decider-v2). Hub loading is attempted only when there is no local copy;
    pass ``local_data_dir: null`` in the config to force it.
    """
    data_cfg = cfg.data
    if data_cfg.local_data_dir:
        root = Path(__file__).resolve().parents[2]  # decider-v2/
        base = Path(data_cfg.local_data_dir)
        path = base if base.is_absolute() else (root / base)
        if path.is_file():
            return path
        if path.is_dir():
            # the dataset repo's own layout is <clone>/data/<split>.parquet
            # (README "configs" section); a flat <dir>/<split>.parquet also works
            for candidate in (path / "data" / f"{split}.parquet", path / f"{split}.parquet"):
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(
                f"no {split}.parquet found under {path} (expected data/{split}.parquet)"
            )
    from datasets import load_dataset

    ds = load_dataset(
        data_cfg.dataset_name,
        data_cfg.dataset_config,
        revision=data_cfg.dataset_revision,
        split=split,
    )
    return ds.data  # type: ignore[return-value]


def _require_columns(df: pd.DataFrame, source: object) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset {source} is missing required columns {missing}")


def _parse_questions(row) -> list[tuple[str, dict]]:
    """questions_json -> [(question_name, question_dict)], file order.

    Accepts a mapping row, a namedtuple row (``itertuples``) or the raw JSON
    string itself. Raises on malformed rows: a truncated JSON object means a
    broken export, and silently training on half a dataset is worse than
    failing loudly.
    """
    if isinstance(row, str):
        raw_json = row
    elif isinstance(row, dict):
        raw_json = row["questions_json"]
    else:
        raw_json = row.questions_json
    raw = json.loads(raw_json)
    if not isinstance(raw, dict):
        raise ValueError(f"questions_json must decode to an object, got {type(raw).__name__}")
    return list(raw.items())


def _check_label(qtype: str, label: object, criteria: object, name: str) -> None:
    """Validate one label against its question type; raise ValueError if malformed."""
    if qtype == "choice":
        if not isinstance(label, str):
            raise ValueError(f"choice label must be str, got {type(label).__name__} in {name!r}")
        if not isinstance(criteria, dict) or label not in criteria:
            raise ValueError(f"choice label {label!r} is not a criteria key in {name!r}")
    elif qtype == "noul":
        if not isinstance(label, bool):
            raise ValueError(f"noul label must be bool, got {type(label).__name__} in {name!r}")
    elif qtype == "score":
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(f"score label must be int, got {type(label).__name__} in {name!r}")
        if not isinstance(criteria, (list, tuple)):
            raise ValueError(f"score criteria must be a list in {name!r}")
        if not 0 <= label < len(criteria):
            raise ValueError(f"score label {label} out of range for {len(criteria)} levels in {name!r}")
    else:
        raise ValueError(f"unknown question type {qtype!r} in {name!r}")


def _option_text(q: dict) -> str | None:
    """Longest option text for the token-budget filter; None if not checkable.

    Checks the single longest candidate, which upper-bounds every block the
    tokenizer would build for this question.
    """
    qtype, criteria = q.get("type"), q.get("criteria")
    if qtype == "choice":
        texts = [f"{k}: {v}" if v else str(k) for k, v in criteria.items()]
    elif qtype == "score":
        texts = [f"{i}: {v}" for i, v in enumerate(criteria)]
    elif qtype == "noul" and isinstance(criteria, dict):
        texts = [f"true: {criteria['true']}" if criteria.get("true") else "true",
                 f"false: {criteria['false']}" if criteria.get("false") else "false"]
    else:
        return None
    return max(texts, key=len) if texts else None


def _dev_split(
    row_state: str, run_name: str, dev_size: int, counter: Counter, n_rows: int
) -> bool:
    """Deterministic per-row dev assignment (online Bernoulli sampling).

    Row ``i`` is taken with probability ``(dev_size - taken) / (n_rows - i)``,
    so the expected total is exactly ``min(dev_size, n_rows)`` and the running
    count never overflows (no early-exit bias toward the head of the file).
    The decision is a function of ``sha1(run_name|state)`` plus the counters,
    so the same state always lands on the same side for a given ``run_name``
    (repeated states in the train parquet share questions' split), and a
    different ``run_name`` yields an independent carve-out.
    """
    counter["seen"] += 1
    if dev_size <= 0 or counter["dev"] >= dev_size:
        return False
    remaining_rows = n_rows - (counter["seen"] - 1)
    need = dev_size - counter["dev"]
    if need >= remaining_rows:  # take every remaining row
        counter["dev"] += 1
        return True
    h = hashlib.sha1(f"{run_name}|{row_state}".encode("utf-8")).digest()
    u = int.from_bytes(h[:4], "big") / 2**32
    if u < need / remaining_rows:
        counter["dev"] += 1
        return True
    return False


def load_examples(cfg: Config, split: str, tokenizer: DeciderTokenizer | None = None) -> list[Example]:
    """Load one parquet split into Examples (train/dev for ``train``, test for ``test``).

    Filtering and the dev carve-out depend on the tokenizer limits and
    ``run_name``, so callers building several example lists from the same
    config should reuse one tokenizer instance (it only caches token ids).
    """
    source = _resolve_parquet_path(cfg, "test" if split == "test" else "train")
    if isinstance(source, Path):
        df = pd.read_parquet(source)
        name = source.name
    else:  # datasets.Dataset (Hub path)
        df = source.to_pandas()
        name = f"hub:{cfg.data.dataset_name}:{source.split}"
    _require_columns(df, name)

    data_cfg = cfg.data
    if tokenizer is None:
        tokenizer = DeciderTokenizer.from_config(data_cfg)
    max_opt = data_cfg.max_option_tokens

    dev_counter: Counter = Counter()
    examples: list[Example] = []
    is_train = split == "train"
    skipped = Counter()

    for row in df.itertuples(index=False):
        row_state = row.state
        domain = getattr(row, "domain", "unknown")
        if is_train and _dev_split(
            row_state, cfg.run_name, data_cfg.dev_size, dev_counter, n_rows=len(df)
        ):
            row_split = "dev"
        else:
            row_split = "train" if is_train else "test"
        for qname, q in _parse_questions(row):
            qtype = q.get("type")
            criteria = q.get("criteria")
            try:
                _check_label(qtype, q.get("label"), criteria, qname)
            except ValueError:
                skipped["bad_label"] += 1
                continue
            text = _option_text(q)
            if text is not None:
                n = len(tokenizer.tok.encode(text, add_special_tokens=False))
                if n > max_opt:
                    skipped["option_tokens"] += 1
                    continue
            n_opts = len(criteria) if isinstance(criteria, (dict, list)) else 2
            if not data_cfg.min_options <= n_opts <= data_cfg.max_options:
                skipped["option_count"] += 1
                continue
            examples.append(
                Example(
                    state=row_state,
                    question=q,
                    label=q["label"],
                    question_name=qname,
                    domain=domain,
                    split=row_split,
                    max_option_tokens_used=n if text is not None else 0,
                )
            )
    if skipped:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
        total = sum(skipped.values())
        print(f"[adapters] {split}: kept {len(examples)} examples, skipped {total} ({detail})")
    return examples


def subsample_stratified(
    examples: list[Example], size: int, seed: int, key=lambda e: e.question["type"]
) -> list[Example]:
    """Deterministic per-type proportional subsample of exactly ``size`` items.

    Quotas are proportional to type frequency, with largest-remainder
    allocation of the leftover slots, filled per type with a seeded shuffle.
    A type with fewer items than its quota comes up short (logged once).
    """
    if size <= 0 or len(examples) <= size:
        return list(examples)
    rng = random.Random(seed)
    by_type: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        by_type[key(e)].append(e)
    for bucket in by_type.values():
        rng.shuffle(bucket)

    quota = {t: len(v) * size / len(examples) for t, v in by_type.items()}
    picked: dict[str, int] = {t: int(q) for t, q in quota.items()}
    leftover = size - sum(picked.values())
    # largest remainder to distribute the leftover slots
    fracs = sorted(quota, key=lambda t: quota[t] - picked[t], reverse=True)
    for t in fracs[:leftover]:
        picked[t] += 1

    picks: dict[str, list[Example]] = {}
    short: list[str] = []
    for t, n in picked.items():
        bucket = by_type[t]
        take = min(n, len(bucket))
        if take < n:
            short.append(f"{t} ({take}/{n})")
        picks[t] = bucket[:take]
    if short:
        print(f"[adapters] subsample: types below quota: {', '.join(short)}")
    # round-robin interleave over types so consecutive items alternate types;
    # the training loop shuffles anyway, this keeps even unshuffled batches mixed
    out: list[Example] = []
    i = 0
    while any(i < len(p) for p in picks.values()):
        for t in sorted(picks):
            if i < len(picks[t]):
                out.append(picks[t][i])
        i += 1
    return out


def load_train_dev(cfg: Config, tokenizer: DeciderTokenizer | None = None) -> tuple[list[Example], list[Example]]:
    """Train + dev in one pass (dev is carved out of the train parquet)."""
    examples = load_examples(cfg, "train", tokenizer=tokenizer)
    train = [e for e in examples if e.split == "train"]
    dev = [e for e in examples if e.split == "dev"]
    return train, dev


def load_test(cfg: Config, tokenizer: DeciderTokenizer | None = None) -> list[Example]:
    return load_examples(cfg, "test", tokenizer=tokenizer)


def target_index(example: Example) -> int:
    """Label -> index into the tokenizer's option order (criteria order).

    The tokenizer emits options in criteria order (ARCH.md §5.3), so this is
    the class index for the CE loss. noul maps True->0 ("true"), False->1.
    """
    q = example.question
    if q["type"] == "choice":
        keys = list(q["criteria"])
        return keys.index(example.label)
    if q["type"] == "noul":
        return 0 if example.label else 1
    return int(example.label)  # score: the label IS the zero-based level index


def normalize_state(state: str) -> str:
    """Normalization for exact-match state dedup and overlap checks.

    Rule (fixed, deterministic): unicode NFKC, lowercase, and every whitespace
    run collapsed to a single space with ends stripped. Nothing else — no
    punctuation folding — so "exact-match on normalized state text" stays a
    conservative, reproducible rule.
    """
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", str(state)).lower().split())


def state_overlap(cfg: Config) -> dict[str, int]:
    """Exact-match overlap between train and test states (normalized).

    Loads only the ``state`` columns of both parquet files and compares the
    distinct normalized states. This is a superset of the model-visible
    overlap (it ignores adapter filtering), so an overlap of 0 guarantees the
    training set cannot leak into the held-out evaluation. The dev carve-out
    lives inside train and cannot change this number.
    """
    train_states = {
        normalize_state(s)
        for s in pd.read_parquet(_resolve_parquet_path(cfg, "train"), columns=["state"])["state"]
    }
    test_states = {
        normalize_state(s)
        for s in pd.read_parquet(_resolve_parquet_path(cfg, "test"), columns=["state"])["state"]
    }
    overlap = train_states & test_states
    report = {
        "train_states": len(train_states),
        "test_states": len(test_states),
        "overlap": len(overlap),
    }
    print(
        f"[adapters] state overlap train<->test (normalized exact match): "
        f"train={report['train_states']} test={report['test_states']} overlap={report['overlap']}"
    )
    if overlap:
        sample = sorted(overlap)[:3]
        for s in sample:
            print(f"[adapters]   overlapping state: {s[:120]!r}")
    return report


def type_counts(examples: list[Example]) -> dict[str, int]:
    c = Counter(e.question["type"] for e in examples)
    return {t: c.get(t, 0) for t in ("choice", "noul", "score")}
