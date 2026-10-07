import hashlib
import json
import random
from pathlib import Path

import torch
from datasets import Features, Sequence, Value, concatenate_datasets, load_dataset

from tokenization import encode_example, tokenizer

# The dataset is tracked in-repo at bev-decision/. `datasets` reads the config definitions
# from that directory's README.md card, so a local path resolves exactly like the Hub id does
# and needs no network. Falls back to the Hub when the directory is absent (partial checkout).
_LOCAL_DATASET = Path(__file__).resolve().parent / "bev-decision"
DATASET_NAME = str(_LOCAL_DATASET) if (_LOCAL_DATASET / "README.md").exists() else "avbiswas/bev-decision"
# data.configs accepts any of these, or "all" for every config at once
DATASET_CONFIGS = ("default", "hard_50k", "numeric_temporal", "skills", "counterfactual_15k", "all")
TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}
TASK_NAMES = {v: k for k, v in TASK_TYPES.items()}

# explode_questions() emits an empty list for rows with no soft labels, which is every row
# outside the `skills` config (~7%). Left to inference, `map` types label_probs from
# whichever batch it happens to see first: a batch with no soft labels yields list<null>,
# and the first batch that does contain them then fails to cast double into null. That
# breaks any load large enough to span several batches over the concatenated `all` config.
# Declaring the schema up front fixes the type once, and keeps the empty list meaningful
# as "no soft label" (collate_fn and compute_loss both test it for truthiness).
EXPLODED_FEATURES = Features({
    "state": Value("string"),
    "task_type": Value("int64"),
    "instructions": Value("string"),
    "choices": Sequence(Value("string")),
    "label": Value("int64"),
    "label_probs": Sequence(Value("float32")),
    "domain": Value("string"),
    "domain_id": Value("int64"),
})

# The dataset has no validation split. Val is carved out of train with a deterministic rule on
# the normalized state text alone: ~1/VAL_STATE_MODULUS (~0.5%) of distinct states, state-grouped
# so a repeated state never straddles train and val. Same rule for every config and seed.
VAL_STATE_MODULUS = 200

# domain string -> long id, assigned on first sight (in-memory contract; names are looked up by
# id where metrics need labels, queue.py builds one for REPORT.md). Ids live in the batch as a
# tensor: strings would break to_device().
_KNOWN_DOMAIN_IDS = {}


def domain_id(domain):
    if domain not in _KNOWN_DOMAIN_IDS:
        _KNOWN_DOMAIN_IDS[domain] = len(_KNOWN_DOMAIN_IDS)
    return _KNOWN_DOMAIN_IDS[domain]


def domain_name_for_id(domain_id_value):
    for name, index in _KNOWN_DOMAIN_IDS.items():
        if index == domain_id_value:
            return name
    raise KeyError(f"no domain name was ever seen for id {domain_id_value}")


def question_to_choices(question):
    """Turns any of the 3 question types into a list of option strings (+ label index, + soft label_probs if present).

    label_probs exists only in the skills config and was verified against real rows to have one fixed
    format per type: choice -> dict keyed by the criteria keys, noul -> {"true": p, "false": p},
    score -> list aligned with criteria. label is its argmax in every case. The dict formats are
    reordered here so the returned label_probs always aligns with the returned choices.
    """
    label_probs = question.get("label_probs")
    if question["type"] == "choice":
        keys = list(question["criteria"])
        choices = [f"{key}: {question['criteria'][key]}" if question["criteria"][key] else key for key in keys]
        label = keys.index(question["label"]) if "label" in question else None
        if label_probs is not None:
            label_probs = [label_probs[key] for key in keys]
    elif question["type"] == "score":
        # Choices share position ids, so the level index is written into the text to keep the ordering
        choices = [f"{i}: {level}" for i, level in enumerate(question["criteria"])]
        label = question.get("label")
        # label_probs is already a list aligned with the criteria list
    else:  # noul -> P(Yes)
        choices = ["No", "Yes"]
        label = int(question["label"]) if "label" in question else None
        if label_probs is not None:
            label_probs = [label_probs["false"], label_probs["true"]]
    return choices, label, label_probs


def explode_questions(rows):
    """One dataset row can hold many questions about the same state -> one output row per question.

    Keeps the domain string (for metric labels / REPORT.md) and adds a numeric `domain_id`
    (registered here, exactly once per name); only the id ever enters the batch (strings would
    break to_device)."""
    out = {"state": [], "task_type": [], "instructions": [], "choices": [], "label": [],
           "label_probs": [], "domain": [], "domain_id": []}
    for state, questions_json, domain in zip(rows["state"], rows["questions_json"], rows["domain"]):
        for question in json.loads(questions_json).values():
            choices, label, label_probs = question_to_choices(question)
            out["state"].append(state)
            out["task_type"].append(TASK_TYPES[question["type"]])
            out["instructions"].append(question["instructions"])
            out["choices"].append(choices)
            out["label"].append(label)
            out["label_probs"].append(label_probs if label_probs is not None else [])
            out["domain"].append(domain)
            out["domain_id"].append(domain_id(domain))
    return out


def normalize_state(state):
    return " ".join(state.split())


def is_val_state(state):
    """Deterministic val carve-out rule (see VAL_STATE_MODULUS), on the normalized state text."""
    inner = hashlib.md5(normalize_state(state).encode("utf-8")).hexdigest()
    return int(hashlib.md5(inner.encode()).hexdigest(), 16) % VAL_STATE_MODULUS == 0


def load_questions(split, max_questions=None, seed=0, configs=None, dev_fraction=None):
    """Loads one split; 'val' is carved out of train (state-grouped), 'test' is only read for
    final reporting (train.py never loads it). dev_fraction is accepted for old configs but
    ignored: the deterministic val rule replaced the seed-random dev carve-out (leak-free).

    Sampling is question-aware: a row holds ~1.83 questions on average (and one row can hold
    several questions for the same state), so a question cap needs ~cap rows shuffled before
    exploding. Rows are shuffled once with a fixed seed, over-drawn 2x + 32, exploded, then cut
    to the cap — the cap is a question count, not a row count (the old code conflated the two).
    """
    configs = list(configs or ["default"])
    unknown = [config for config in configs if config not in DATASET_CONFIGS]
    if unknown:
        raise ValueError(f"unknown data.configs {unknown}, expected any of {list(DATASET_CONFIGS)}")
    # "val" is carved out of train below; "test" is only ever read for final reporting (train.py never loads it)
    source_split = "train" if split in ("train", "val", "dev") else split
    if len(configs) == 1:
        ds = load_dataset(DATASET_NAME, configs[0], split=source_split)
    else:
        ds = concatenate_datasets([load_dataset(DATASET_NAME, config, split=source_split) for config in configs])

    if split in ("train", "val", "dev"):
        keep_val = split in ("val", "dev")
        # Grouped by normalized state so a repeated state never straddles train and val
        ds = ds.filter(lambda row: is_val_state(row["state"]) == keep_val)

    rng = random.Random(seed)
    if max_questions is not None:
        order = list(range(len(ds)))
        rng.shuffle(order)
        # ~1.83 questions per training row (measured): 2x + 32 rows covers the cap after exploding,
        # without materializing questions for rows that get cut
        ds = ds.select(sorted(order[:min(len(ds), int(max_questions * 2) + 32)]))
    ds = ds.map(explode_questions, batched=True, remove_columns=ds.column_names,
                features=EXPLODED_FEATURES)
    if max_questions is not None:
        ds = ds.select(range(min(max_questions, len(ds))))
    return ds


class BEVDataset(torch.utils.data.Dataset):
    def __init__(self, questions, max_state_tokens, max_choice_tokens):
        self.questions = questions
        self.max_state_tokens = max_state_tokens
        self.max_choice_tokens = max_choice_tokens

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        row = self.questions[idx]
        example = encode_example(
            row["state"], row["instructions"], row["choices"], TASK_NAMES[row["task_type"]],
            self.max_state_tokens, self.max_choice_tokens,
        )
        example["task_type"] = row["task_type"]
        example["label"] = row["label"]
        # precomputed id from explode_questions; hand-built rows (tests) register the string instead
        example["domain_id"] = row["domain_id"] if "domain_id" in row else domain_id(row["domain"])
        example["label_probs"] = row.get("label_probs") or []
        return example


def collate_fn(batch):
    B = len(batch)
    L = max(len(ex["input_ids"]) for ex in batch)
    C = max(len(ex["choice_read_idx"]) for ex in batch)

    input_ids = torch.full((B, L), tokenizer.pad_token_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    # Padded query rows attend only to themselves, otherwise their softmax row is all -inf -> NaN
    attention_mask = torch.eye(L).repeat(B, 1, 1)
    choice_read_idx = torch.zeros((B, C), dtype=torch.long)
    choice_mask = torch.zeros((B, C), dtype=torch.bool)
    answer_end_idx = torch.zeros(B, dtype=torch.long)
    prefix_len = torch.zeros(B, dtype=torch.long)  # tokens before the first option (hint + question + state)
    label_probs = torch.zeros((B, C))              # soft labels (skills config); 0 where absent

    for b, ex in enumerate(batch):
        n, c = len(ex["input_ids"]), len(ex["choice_read_idx"])
        input_ids[b, :n] = torch.tensor(ex["input_ids"])
        position_ids[b, :n] = torch.tensor(ex["position_ids"])
        attention_mask[b, :n, :n] = ex["attention_mask"]
        choice_read_idx[b, :c] = torch.tensor(ex["choice_read_idx"])
        choice_mask[b, :c] = True
        answer_end_idx[b] = ex["answer_end_idx"]
        prefix_len[b] = ex["question_end_idx"] + 1
        if ex.get("label_probs"):
            label_probs[b, :len(ex["label_probs"])] = torch.tensor(ex["label_probs"], dtype=torch.float32)

    attention_mask = torch.where(attention_mask == 0, -float("inf"), 0.0).unsqueeze(1)  # B, 1, L, L

    out = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "choice_read_idx": choice_read_idx,
        "choice_mask": choice_mask,
        "answer_end_idx": answer_end_idx,
        "prefix_len": prefix_len,
        "label_probs": label_probs,
        "task_type": torch.tensor([ex["task_type"] for ex in batch], dtype=torch.long),
        # per-domain metrics see ids (strings would break to_device); domain_name_for_id() maps back
        "domain_id": torch.tensor([ex["domain_id"] for ex in batch], dtype=torch.long),
    }
    if all(ex["label"] is not None for ex in batch):
        out["labels"] = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    return out
