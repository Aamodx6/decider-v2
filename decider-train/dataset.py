"""Dataset construction, split carving, batch collation -- plus the `dataset.py prepare` CLI.

The heavy work (explode rows -> one record per typed question, carve train/val/calib/test) happens
ONCE (in the notebook or via `python dataset.py prepare ...`) and is written to Parquet.  Everything
else here only loads those Parquet shards, shuffles option order per epoch and collates into the
vectorised packed batches from tokenization.py.

Prepare CLI (moved here verbatim from the old prepare.py; same flags):

    python dataset.py prepare --out ./data_prepared [--seed 0] [--calib_target 20000]
                              [--val_frac 0.03] [--smoke_train N --smoke_val N --smoke_calib N
                                                  --smoke_test N] [--cache_dir PATH] [--force]
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter

import torch
from datasets import load_dataset

from tokenization import TASK_NAMES, TASK_TYPES, build_batch, question_to_options

DATASET_NAME = "bev-decision"
DATASET_CONFIG = "all"


# --------------------------------------------------------------------------- exploding rows
def explode_questions(batch):
    """One dataset row can hold many typed questions about the same state -> one record per question."""
    out = {k: [] for k in ("row_index", "state", "instructions", "options", "keys", "ranks", "label",
                           "task_type", "task_name", "domain", "question_id")}
    for i, (state, questions_json, domain) in enumerate(
            zip(batch["state"], batch["questions_json"], batch["domain"])):
        for qid, q in json.loads(questions_json).items():
            options, keys, label, ranks = question_to_options(q, noul_use_criteria=False)
            out["row_index"].append(batch["row_index"][i])
            out["state"].append(state)
            out["instructions"].append(q.get("instructions", ""))
            out["options"].append(options)
            out["keys"].append(keys)
            out["ranks"].append(ranks)
            out["label"].append(-1 if label is None else int(label))
            out["task_type"].append(TASK_TYPES[q["type"]])
            out["task_name"].append(q["type"])
            out["domain"].append(domain)
            out["question_id"].append(qid)
    return out


RECORD_COLUMNS = ["row_index", "state", "instructions", "options", "keys", "ranks", "label",
                  "task_type", "task_name", "domain", "question_id"]


def build_records(ds, cache_dir=None, name="records"):
    """explode_questions over a whole split -> a Dataset of typed-question records.

    `name` must differ per split: datasets keys its .map cache by the file you pass, so reusing one
    path across two different source splits silently returns the first split's result.
    """
    d = ds.add_column("row_index", list(range(len(ds))))
    cache_file = os.path.join(cache_dir, f"{name}.map") if cache_dir else None
    return d.map(explode_questions, batched=True, batch_size=1000, remove_columns=ds.column_names,
                 cache_file_name=cache_file)


# --------------------------------------------------------------------------- split carving
def stratify_key(task_name, domain):
    return f"{task_name}|{domain}"


def carve_splits(records, seed=0, calib_target=20000, calib_min_frac=0.05,
                 val_frac=0.03, test_source_split="test"):
    """records: Dataset of train-split questions.  Returns dict of split -> index list.

    calib is carved from TRAIN before any training, stratified by (question type, domain), and
    targets calib_target questions; if the data cannot supply that many it falls back to
    calib_min_frac of train (per the mission's ">=20k if the data allows, otherwise 5%").
    val is a separate held-out slice of train used for early stopping / checkpoint selection.
    The published test split is returned untouched under 'test'.
    """
    n = len(records)
    tasks = records["task_name"]
    domains = records["domain"]
    keys = [stratify_key(t, d) for t, d in zip(tasks, domains)]

    rng = random.Random(seed)
    buckets = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)
    for k in buckets:
        rng.shuffle(buckets[k])

    calib_n = max(calib_target, int(n * calib_min_frac)) if calib_target else int(n * calib_min_frac)
    calib_n = min(calib_n, int(n * 0.15))
    calib, rest = [], []
    # proportional allocation with a deterministic largest-remainder pass
    keys_sorted = sorted(buckets)
    total = sum(len(buckets[k]) for k in keys_sorted)
    quotas = {k: int(calib_n * len(buckets[k]) / total) for k in keys_sorted}
    short = calib_n - sum(quotas.values())
    for k in sorted(keys_sorted, key=lambda k: (-(calib_n * len(buckets[k]) / total - quotas[k]), k))[:short]:
        quotas[k] += 1
    for k in keys_sorted:
        take = min(quotas[k], len(buckets[k]))
        calib += buckets[k][:take]
        rest += buckets[k][take:]

    rng.shuffle(rest)
    val_n = int(n * val_frac)
    val, train = rest[:val_n], rest[val_n:]
    return {"train": sorted(train), "val": sorted(val), "calib": sorted(calib)}


# --------------------------------------------------------------------------- torch dataset
class DecisionDataset(torch.utils.data.Dataset):
    """Wraps a Parquet/Arrow dataset of question records.

    set_epoch(e) re-shuffles the option order for every example (an order-invariance-preserving
    augmentation).  The permutation is a pure function of (seed, epoch, row index), so it is
    reproducible and resumes cleanly.
    """

    def __init__(self, table, encoder, shuffle_options=True, seed=0, epoch=0):
        self.records = table
        self.encoder = encoder
        self.shuffle_options = shuffle_options
        self.seed = seed
        self.epoch = epoch
        self.length_cache = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def _row(self, idx):
        r = self.records[idx]
        return (r["state"], r["instructions"], list(r["options"]), r["task_name"],
                int(r["label"]), r["domain"], r["task_type"],
                list(r["ranks"]) if "ranks" in r.keys() else [0.0] * len(r["options"]))

    def __getitem__(self, idx):
        (state, instructions, options, task_name, label, domain, task_type,
         ranks) = self._row(idx)
        if self.shuffle_options:
            rng = random.Random((self.seed, self.epoch, idx).__hash__())
            order = list(range(len(options)))
            rng.shuffle(order)
            label = order.index(label) if label >= 0 else -1
            options = [options[i] for i in order]
            ranks = [ranks[i] for i in order]
        ex = self.encoder.encode(state, instructions, options, task_name, ranks=ranks)
        ex["label"] = label
        ex["domain"] = domain
        ex["idx"] = idx
        # kept so that evaluate.py can re-encode this example with permuted options
        ex["_source"] = {"state": state, "instructions": instructions, "options": options,
                         "task_name": task_name, "ranks": ranks}
        ex["_encoder"] = self.encoder
        return ex


def collate(batch, pad_id):
    out = build_batch([b for b in batch], pad_id)
    out["labels"] = torch.tensor([b.get("label", 0) for b in batch], dtype=torch.long)
    out["domains"] = [b.get("domain", "") for b in batch]
    out["idx"] = torch.tensor([b.get("idx", i) for i, b in enumerate(batch)], dtype=torch.long)
    return out


class LengthBucketBatchSampler(torch.utils.data.Sampler):
    """Groups examples of similar encoded length into batches.

    The additive mask is O(L^2) per example, so padding a 20-token question next to a 1,100-token
    state costs ~3,000x the mask memory for nothing.  Bucketing on the state token count keeps L
    tight.  Shuffling happens at the batch level, so batches stay random.
    """

    def __init__(self, lengths, batch_size, shuffle=True, seed=0, drop_last=False, bucket=64):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        self.bucket = bucket
        self._n = len(lengths)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        n = self._n
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(self._n))
        if self.shuffle:
            rng.shuffle(idx)
        # sort within large windows: keeps randomness, keeps lengths local
        window = self.batch_size * self.bucket
        batches = []
        for s in range(0, len(idx), window):
            chunk = sorted(idx[s:s + window], key=lambda i: self.lengths[i])
            for t in range(0, len(chunk), self.batch_size):
                batches.append(chunk[t:t + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        for b in batches:
            if self.drop_last and len(b) < self.batch_size:
                continue
            yield b


def state_token_lengths(records, tokenizer, sample=None, max_state_tokens=1024, max_choice_tokens=64):
    """Cheap length proxy: token counts of state + instructions + options (plus 8 tokens overhead)."""
    n = len(records) if sample is None else min(sample, len(records))
    lengths = []
    for i in range(n):
        r = records[i]
        total = 8
        total += min(len(tokenizer.encode(r["state"])), max_state_tokens)
        total += min(len(tokenizer.encode(r["instructions"])), 512)
        for o in r["options"]:
            total += min(len(tokenizer.encode(o)), max_choice_tokens) + 3
        lengths.append(total)
    return lengths


# --------------------------------------------------------------------------- loading
def load_split_table(path, limit=None):
    """Load one or many Parquet shards from a directory or file path."""
    from datasets import load_dataset as ld
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet"))
        if not files:
            raise FileNotFoundError(f"no .parquet under {path}")
    else:
        files = [path]
    d = ld("parquet", data_files=files)
    if hasattr(d, "keys"):        # datasets>=5 returns a DatasetDict keyed "train"
        d = d["train"] if "train" in d.keys() else d[list(d.keys())[0]]
    return d.select(range(min(limit, len(d)))) if limit else d


def split_summary(tables):
    rows = {}
    for name, t in tables.items():
        tc = Counter(t["task_name"])
        rows[name] = {"questions": len(t), "task_type": dict(tc)}
    return rows


# --------------------------------------------------------------------------- prepare CLI
def stats(table):
    c = Counter(table["task_name"])
    d = Counter(table["domain"])
    return {"questions": len(table), "by_type": dict(c), "by_domain": dict(d.most_common())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare"], help="'prepare' builds the split shards")
    ap.add_argument("--out", default="./data_prepared")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib_target", type=int, default=20000)
    ap.add_argument("--val_frac", type=float, default=0.03)
    ap.add_argument("--smoke_train", type=int, default=0, help="cap train questions (SMOKE mode)")
    ap.add_argument("--smoke_val", type=int, default=0)
    ap.add_argument("--smoke_calib", type=int, default=0)
    ap.add_argument("--smoke_test", type=int, default=0)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    manifest_path = os.path.join(args.out, "manifest.json")
    if os.path.exists(manifest_path) and not args.force:
        m = json.load(open(manifest_path))
        if m["seed"] == args.seed and m["calib_target"] == args.calib_target:
            print("prepare: shards already exist with the same seed/calib_target -> reusing them")
            print(json.dumps(m["stats"], indent=2))
            return

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    for s in ("train", "val", "calib", "test"):
        os.makedirs(os.path.join(args.out, s), exist_ok=True)

    print(f"loading {DATASET_NAME} config={DATASET_CONFIG} ...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, cache_dir=args.cache_dir)
    print(ds)
    print("raw split sizes:", {k: len(v) for k, v in ds.items()})

    print("exploding rows -> one record per typed question (train split) ...")
    train_records = build_records(ds["train"], cache_dir=args.cache_dir, name="train")
    print(f"train questions: {len(train_records):,}")

    idx = carve_splits(train_records, seed=args.seed, calib_target=args.calib_target,
                       val_frac=args.val_frac)
    print({k: len(v) for k, v in idx.items()})

    caps = {"train": args.smoke_train, "val": args.smoke_val,
            "calib": args.smoke_calib, "test": args.smoke_test}
    sizes = {}
    for name, ids in idx.items():
        cap = caps.get(name, 0)
        if cap and len(ids) > cap:
            ids = random.Random(args.seed).sample(ids, cap)
            ids.sort()
        t = train_records.select(ids)
        t.to_parquet(os.path.join(args.out, name, "questions.parquet"))
        sizes[name] = len(t)
        print(f"  wrote {name}: {len(t):,} questions")

    print("exploding the published test split (untouched) ...")
    test_records = build_records(ds["test"], cache_dir=args.cache_dir, name="test")
    if caps["test"] and len(test_records) > caps["test"]:
        ids = sorted(random.Random(args.seed).sample(range(len(test_records)), caps["test"]))
    else:
        ids = None
    test_records = test_records if ids is None else test_records.select(ids)
    test_records.to_parquet(os.path.join(args.out, "test", "questions.parquet"))
    sizes["test"] = len(test_records)
    print(f"  wrote test: {len(test_records):,} questions")

    # state-overlap audit: the published splits must not share a state, and our val/calib come from
    # train, so we also check val/calib states against the test states.
    def norm(s):
        return " ".join(s.lower().split())

    train_tbl = train_records
    val_ids = idx["val"]
    calib_ids = idx["calib"]
    val_tbl = train_records.select(val_ids)
    calib_tbl = train_records.select(calib_ids)
    test_states = {norm(s) for s in test_records["state"]}
    val_hit = sum(1 for s in val_tbl["state"] if norm(s) in test_states)
    calib_hit = sum(1 for s in calib_tbl["state"] if norm(s) in test_states)
    print(f"state overlap: val->test {val_hit}, calib->test {calib_hit} (must be 0)")

    manifest = {
        "dataset": DATASET_NAME, "config": DATASET_CONFIG, "seed": args.seed,
        "calib_target": args.calib_target, "val_frac": args.val_frac,
        "raw_rows": {k: len(v) for k, v in ds.items()},
        "sizes": sizes, "state_overlap_val_to_test": val_hit,
        "state_overlap_calib_to_test": calib_hit,
        "stats": {k: stats(t) for k, t in
                  {"train": train_records.select(idx["train"][:0]) if False else train_records,
                   "val": val_tbl, "calib": calib_tbl, "test": test_records}.items()},
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "stats"}, indent=2))
    print(f"\ntype mix per split: ")
    for name, t in (("train", train_records), ("val", val_tbl), ("calib", calib_tbl),
                    ("test", test_records)):
        c = Counter(t["task_name"])
        n = sum(c.values())
        print(f"  {name:6s} n={n:7,d} " + "  ".join(f"{k}={v/n*100:5.2f}%" for k, v in sorted(c.items())))
    print(f"\nwrote {manifest_path}")


__all__ = [
    "DATASET_NAME", "DATASET_CONFIG", "RECORD_COLUMNS", "TASK_NAMES", "TASK_TYPES",
    "explode_questions", "build_records", "carve_splits", "DecisionDataset", "collate",
    "LengthBucketBatchSampler", "state_token_lengths", "load_split_table", "split_summary",
]


if __name__ == "__main__":
    main()
