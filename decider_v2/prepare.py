"""prepare.py -- build the train / val / calib / test Parquet shards from avbiswas/bev-decision('all').

Run once (in the notebook) and upload the shards to S3 as SageMaker input channels.  Idempotent:
it writes a manifest.json with the seed and every choice it made, and skips work when the shards
already exist and the manifest matches.
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict

from datasets import load_dataset

from data import DATASET_CONFIG, DATASET_NAME, build_records, carve_splits
from serialize import TASK_TYPES


def stats(table):
    c = Counter(table["task_name"])
    d = Counter(table["domain"])
    return {"questions": len(table), "by_type": dict(c), "by_domain": dict(d.most_common())}


def main():
    ap = argparse.ArgumentParser()
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
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()