"""Phase 0 recon part 2: domain table, abstention keys, truncation rates, label balance, benchmarks."""
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset

CACHE = r"C:\Users\aamod\AppData\Local\Temp\opencode\hf_cache"


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


ds = load_dataset("avbiswas/bev-decision", "all", cache_dir=CACHE)

rule("A. DOMAIN TABLE (complete)")
domain_rows, domain_q = Counter(), Counter()
for split in ds:
    for row in ds[split]:
        domain_rows[row["domain"]] += 1
        for qq in json.loads(row["questions_json"]).values():
            domain_q[row["domain"]] += 1
tot = sum(domain_rows.values())
print(f"{len(domain_rows)} distinct domains; total rows {tot:,}")
for d, n in domain_rows.most_common():
    print(f"  {d:55s} rows={n:8,d} ({n/tot*100:5.2f}%)  questions={domain_q[d]:8,d}")

rule("B. ABSTENTION-LIKE CHOICE KEYS (the 'null option' question)")
ABSTAIN = re.compile(
    r"^(none|none_of|none_at|unknown|unclear|not_stated|not_determined|not_specified|"
    r"cannot|undetermined|insufficient|n/?a|other|insufficient_information|"
    r"none_at_that_time|no_answer|unanswerable|indeterminate)$", re.I)
keys = Counter()
label_keys = Counter()
abstain_key_present = 0
abstain_key_is_label = 0
choice_total = 0
per_key_abstain_rate = defaultdict(lambda: [0, 0])
for split in ds:
    for row in ds[split]:
        for qq in json.loads(row["questions_json"]).values():
            if qq["type"] != "choice":
                continue
            choice_total += 1
            crit = qq["criteria"]
            hit = [k for k in crit if ABSTAIN.match(k)]
            keys.update(crit.keys())
            label_keys[qq["label"]] += 1
            if hit:
                abstain_key_present += 1
                for k in hit:
                    per_key_abstain_rate[k][1] += 1
                    if qq["label"] == k:
                        per_key_abstain_rate[k][0] += 1
                if qq["label"] in hit:
                    abstain_key_is_label += 1
print(f"choice questions                          : {choice_total:,}")
print(f"  with >=1 abstention-like key in criteria : {abstain_key_present:,} ({abstain_key_present/choice_total*100:.2f}%)")
print(f"  where that key IS the label             : {abstain_key_is_label:,} ({abstain_key_is_label/choice_total*100:.2f}%)")
print("\ntop abstention-like keys: key -> (times it is the label / times it appears)")
for k, (lab, app) in sorted(per_key_abstain_rate.items(), key=lambda x: -x[1][1])[:15]:
    print(f"  {k:24s} {lab:6,d} / {app:7,d}")
print("\nmost frequent labels:", dict(label_keys.most_common(15)))

rule("C. NOUL LABEL BALANCE  /  SCORE LABEL + LEVELS")
noul_lab, score_lab, score_lv = Counter(), Counter(), Counter()
noul_label_is_int = 0
score_label_is_int = 0
score_n = noul_n = 0
for split in ds:
    for row in ds[split]:
        for qq in json.loads(row["questions_json"]).values():
            if qq["type"] == "noul":
                noul_n += 1
                noul_lab[qq["label"]] += 1
                noul_label_is_int += isinstance(qq["label"], int)
            elif qq["type"] == "score":
                score_n += 1
                score_lab[(len(qq["criteria"]), qq["label"])] += 1
                score_label_is_int += isinstance(qq["label"], int)
print(f"noul questions {noul_n:,}  labels={dict(noul_lab)}  label-is-int(bool subclass)={noul_label_is_int:,}")
print(f"score questions {score_n:,}  label-is-int={score_label_is_int:,}")
print("score (n_levels, label) -> count, top 20:")
for (n, l), c in score_lab.most_common(20):
    print(f"  K={n} label={l} : {c:,}")
print("\nper-K label distribution:")
byk = defaultdict(Counter)
for (n, l), c in score_lab.items():
    byk[n][l] += c
for n in sorted(byk):
    tot_n = sum(byk[n].values())
    print(f"  K={n} n={tot_n:7,d} : " + ", ".join(f"{l}:{c/tot_n*100:.1f}%" for l, c in sorted(byk[n].items())))

rule("D. TRUNCATION IMPACT at max_choice_tokens=64 / max_state_tokens=1024")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
sub = ds["train"].select(range(0, len(ds["train"]), 20))  # ~11.5k rows, every 20th
opt_trunc = Counter()
state_over = 0
n = 0
for row in sub:
    n += 1
    sl = len(tok.encode(row["state"]))
    if sl > 1024:
        state_over += 1
    for qq in json.loads(row["questions_json"]).values():
        if qq["type"] == "choice":
            texts = [f"{k}: {v}" if v else k for k, v in qq["criteria"].items()]
        elif qq["type"] == "score":
            texts = [f"{i}: {lv}" for i, lv in enumerate(qq["criteria"])]
        else:
            texts = ["No", "Yes"]
        for t in texts:
            l = len(tok.encode(t))
            if l > 64:
                opt_trunc[l] += 1
print(f"sampled {n:,} train rows (every 20th)")
print(f"states > 1024 tokens: {state_over:,} ({state_over/n*100:.2f}%)  -> truncated by bev recipe")
tot_opt = sum(opt_trunc.values())
print(f"options > 64 tokens  : {tot_opt:,} total; lengths {dict(sorted(opt_trunc.items()))}")
# how many QUESTIONS have at least one truncated option
print("(question-level rate computed separately)")

rule("E. QUESTIONS PER ROW")
qc = Counter()
for split in ds:
    for row in ds[split]:
        qc[row["question_count"]] += 1
print(dict(sorted(qc.items())))
print("max", max(qc))

rule("F. SPLIT-LEVEL COUNTS per source batch (config all)")
for split in ds:
    print(f"{split}: {len(ds[split]):,} rows")

rule("G. BENCHMARK REPOS")
import urllib.request
for url in [
    "https://api.github.com/repos/instax-dutta/sysone-bench",
    "https://api.github.com/search/repositories?q=jevbench",
    "https://api.github.com/search/repositories?q=sysone-bench",
    "https://api.github.com/users/instax-dutta/repos?per_page=100",
    "https://api.github.com/repos/avbiswas/bev-decider",
]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "recon"})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.load(r)
        if "items" in body:
            print(f"\n{url}\n  total={body['total_count']}")
            for it in body["items"][:10]:
                print(f"   - {it['full_name']}  {it.get('description')}")
        elif isinstance(body, list):
            print(f"\n{url}\n  {len(body)} repos")
            for it in body[:40]:
                print(f"   - {it['name']}")
        else:
            print(f"\n{url}\n  FOUND: {body.get('full_name')}  license={body.get('license')}  desc={body.get('description')}")
    except Exception as e:
        print(f"\n{url}\n  FAIL {type(e).__name__}: {e}")