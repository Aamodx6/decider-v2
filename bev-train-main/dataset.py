import hashlib
import json
import random

import torch
from datasets import concatenate_datasets, load_dataset

from tokenization import encode_example, tokenizer

DATASET_NAME = "avbiswas/bev-decision"
# data.configs accepts any of these, or "all" for every config at once
DATASET_CONFIGS = ("default", "hard_50k", "numeric_temporal", "skills", "counterfactual_15k", "all")
TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}
TASK_NAMES = {v: k for k, v in TASK_TYPES.items()}
DEV_FRACTION = 0.05  # fraction of distinct states held out of train for dev (the dataset has no validation split)


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
    """One dataset row can hold many questions about the same state -> one output row per question."""
    out = {"state": [], "task_type": [], "instructions": [], "choices": [], "label": [],
           "domain": [], "label_probs": []}
    for state, questions_json, domain in zip(rows["state"], rows["questions_json"], rows["domain"]):
        for question in json.loads(questions_json).values():
            choices, label, label_probs = question_to_choices(question)
            out["state"].append(state)
            out["task_type"].append(TASK_TYPES[question["type"]])
            out["instructions"].append(question["instructions"])
            out["choices"].append(choices)
            out["label"].append(label)
            out["domain"].append(domain)
            out["label_probs"].append(label_probs if label_probs is not None else [])
    return out


def normalize_state(state):
    return " ".join(state.split())


def state_digest(state):
    # Hashed so grouping stays cheap even for the ~17K-char skills states
    return hashlib.sha1(normalize_state(state).encode("utf-8")).hexdigest()


def held_out_states(states, seed, dev_fraction):
    """Deterministic ~dev_fraction of distinct normalized states, for the dev split."""
    digests = sorted({hashlib.sha1(state.encode("utf-8")).hexdigest() for state in states})
    random.Random(seed).shuffle(digests)
    return {digests[i] for i in range(max(1, int(round(dev_fraction * len(digests)))))}


def load_questions(split, max_questions=None, seed=0, configs=None, dev_fraction=DEV_FRACTION):
    configs = list(configs or ["default"])
    unknown = [config for config in configs if config not in DATASET_CONFIGS]
    if unknown:
        raise ValueError(f"unknown data.configs {unknown}, expected any of {list(DATASET_CONFIGS)}")
    # "dev" is carved out of train below; "test" is only ever read for final reporting (train.py never loads it)
    source_split = "train" if split in ("train", "dev") else split
    if len(configs) == 1:
        ds = load_dataset(DATASET_NAME, configs[0], split=source_split)
    else:
        ds = concatenate_datasets([load_dataset(DATASET_NAME, config, split=source_split) for config in configs])

    if split in ("train", "dev") and dev_fraction > 0:
        # Group by normalized state so a repeated state never straddles train and dev
        held_out = held_out_states((normalize_state(s) for s in ds["state"]), seed, dev_fraction)
        keep_dev = split == "dev"
        ds = ds.filter(lambda row: (state_digest(row["state"]) in held_out) == keep_dev)

    ds = ds.shuffle(seed=seed)
    if max_questions is not None:
        # Every row has at least one question, so this many rows is always enough
        ds = ds.select(range(min(max_questions, len(ds))))
    ds = ds.map(explode_questions, batched=True, remove_columns=ds.column_names)
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
        example["domain"] = row.get("domain")
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
    label_probs = torch.zeros((B, C))              # soft labels (skills config);0 where absent

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
        # strings pass through to_device untouched, for per-domain metrics
        "domain": [ex.get("domain") for ex in batch],
    }
    if all(ex["label"] is not None for ex in batch):
        out["labels"] = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    return out
