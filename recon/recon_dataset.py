"""Phase 0 recon: load bev-decision (config 'all') and print real statistics."""
import json
import os
import sys
from collections import Counter, defaultdict

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

import numpy as np
from datasets import load_dataset

CACHE = r"C:\Users\aamod\AppData\Local\Temp\opencode\hf_cache"
os.makedirs(CACHE, exist_ok=True)


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


rule("0. ENVIRONMENT")
import platform
import torch
import transformers
import datasets as hfds

print("python      ", sys.version.split()[0])
print("platform    ", platform.platform())
print("torch       ", torch.__version__)
print("cuda avail  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu         ", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**20, "MiB")
print("transformers", transformers.__version__)
print("datasets    ", hfds.__version__)

rule("1. LOAD_DATASET('bev-decision', 'all')")
ds = load_dataset("bev-decision", "all", cache_dir=CACHE)
print(ds)

rule("1b. PER-FILE SIZES (config all = union of 5 batch dirs)")
from huggingface_hub import HfApi

api = HfApi()
info = api.repo_info("bev-decision", repo_type="dataset", files_metadata=True)
total = 0
for s in info.siblings:
    if s.rfilename.endswith(".parquet"):
        total += s.size or 0
        print(f"  {s.rfilename:45s} {(s.size or 0)/2**20:9.2f} MiB")
print(f"  {'TOTAL parquet':45s} {total/2**20:9.2f} MiB")

rule("2. COLUMNS / FEATURES / TYPES")
for split, d in ds.items():
    print(f"\n[{split}] rows={len(d):,}")
    print("  features:", d.features)

rule("3. RAW EXAMPLES -- 3 per question type")
# find one row containing each type
seen = Counter()
examples = defaultdict(list)
for split in ds:
    for i in range(len(ds[split])):
        row = ds[split][i]
        q = json.loads(row["questions_json"])
        types = {qq["type"] for qq in q.values()}
        for t in types:
            if seen[t] < 3:
                seen[t] += 1
                examples[t].append((split, i, row))
        if all(seen[t] >= 3 for t in ("choice", "noul", "score")):
            break
    if all(seen[t] >= 3 for t in ("choice", "noul", "score")):
        break

for t in ("choice", "noul", "score"):
    for split, i, row in examples[t]:
        print(f"\n--- type={t}  [{split}[{i}]]  domain={row['domain']!r}  question_types={row['question_types']}  question_count={row['question_count']}")
        print("  state (first 400 chars):", repr(row["state"][:400]))
        q = json.loads(row["questions_json"])
        for k, qq in list(q.items())[:2]:
            print(f"  question[{k!r}] keys={sorted(qq.keys())}")
            print("   ", json.dumps(qq, ensure_ascii=False)[:700])

rule("4. FULL DISTRIBUTIONS (all splits)")
type_q = Counter()
type_rows = Counter()
domain_q = Counter()
domain_rows = Counter()
n_opts = Counter()
n_opts_by_type = defaultdict(Counter)
score_levels = Counter()
choice_keys = Counter()
label_probs_present = 0
label_probs_rows = 0
no_label = 0
criteria_missing = 0
opt_text_present = Counter()
opt_empty = Counter()

state_lens = []
instr_lens = []
opt_lens_all = []
question_type_seq = Counter()

for split in ds:
    d = ds[split]
    for row in d:
        state_lens.append(len(row["state"]))
        domain_rows[row["domain"]] += 1
        q = json.loads(row["questions_json"])
        seen_types = set()
        for k, qq in q.items():
            t = qq["type"]
            type_q[t] += 1
            domain_q[row["domain"]] += 1
            seen_types.add(t)
            question_type_seq[tuple(sorted(seen_types))] += 1
            instr_lens.append(len(qq.get("instructions", "")))
            if "label" not in qq:
                no_label += 1
            if t == "choice":
                crit = qq.get("criteria", {})
                n = len(crit)
                n_opts[n] += 1
                n_opts_by_type[t][n] += 1
                for kk, vv in crit.items():
                    choice_keys[kk] += 1
                    opt_text_present["has_desc" if vv else "empty_desc"] += 1
                    opt_empty[vv == "" or vv is None] += 1
                if not crit:
                    criteria_missing += 1
            elif t == "score":
                levels = qq.get("criteria", [])
                n = len(levels)
                n_opts[n] += 1
                n_opts_by_type[t][n] += 1
                score_levels[n] += 1
                for lv in levels:
                    opt_text_present["level_desc" if lv else "level_empty"] += 1
            else:  # noul
                n_opts[2] += 1
                n_opts_by_type[t][2] += 1
                crit = qq.get("criteria")
                opt_text_present["noul_with_criteria" if crit else "noul_bare"] += 1
            if "label_probs" in qq:
                label_probs_present += 1
        for t in seen_types:
            type_rows[t] += 1
        if any("label_probs" in qq for qq in q.values()):
            label_probs_rows += 1

total_q = sum(type_q.values())
print(f"total questions         : {total_q:,}")
print(f"total rows              : {len(ds['train']) + len(ds['test']):,}")
print(f"questions per type      : {dict(type_q)}")
print(f"shares                   : {{k: round(v/total_q*100,2) for k,v in type_q.items()}}")
print(f"rows containing type     : {dict(type_rows)}")
print(f"questions WITHOUT 'label': {no_label:,}")
print(f"choice rows with 0 criteria: {criteria_missing:,}")
print(f"questions with 'label_probs': {label_probs_present:,} (in {label_probs_rows:,} rows)")
print(f"choice criteria desc present/empty: {dict(opt_text_present)}")
print(f"noul criteria present?   : {dict(opt_text_present)}")
print(f"score level counts       : {dict(sorted(score_levels.items()))}")
print(f"per-row type-set combos  : {dict(question_type_seq.most_common(10))}")

rule("4b. OPTIONS-PER-QUESTION HISTOGRAM (all types / by type)")
allc = Counter(n_opts)
for k in sorted(allc):
    print(f"  {k:3d} options : {allc[k]:9,d}  ({allc[k]/total_q*100:5.2f}%)")
print("\nby type:")
for t, c in n_opts_by_type.items():
    print(f"  {t}: " + ", ".join(f"{k}->{c[k]:,}" for k in sorted(c)))

rule("4c. TOP CHOICE KEYS (top 30)")
print(dict(choice_keys.most_common(30)))
print("distinct choice keys:", len(choice_keys))

rule("5. DOMAIN DISTRIBUTION")
tot_rows = sum(domain_rows.values())
print(f"{'domain':60s} {'rows':>9s} {'share':>7s} {'questions':>10s}")
for d, n in domain_rows.most_common():
    print(f"{d:60s} {n:9,d} {n/tot_rows*100:6.2f}% {domain_q[d]:10,d}")

rule("6. TOKEN-LENGTH PERCENTILES (char lengths; tokenisation printed later)")
def pct(a, name):
    a = np.asarray(a)
    print(f"{name:22s} n={len(a):8,d}  p1={np.percentile(a,1):9.0f} p5={np.percentile(a,5):9.0f} "
          f"p50={np.percentile(a,50):9.0f} p90={np.percentile(a,90):9.0f} p95={np.percentile(a,95):9.0f} "
          f"p99={np.percentile(a,99):9.0f} max={a.max():9,d} mean={a.mean():9.0f}")

pct(state_lens, "state chars")
pct(instr_lens, "instructions chars")

rule("6b. TOKEN-LENGTH PERCENTILES (Qwen3 tokenizer)")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
print("tokenizer:", type(tok).__name__, "vocab_size", tok.vocab_size, "len", len(tok))
print("model_max_length", tok.model_max_length)
print("encode('<option>') =", tok.encode("<option>"), tok.convert_ids_to_tokens(tok.encode("<option>")))
print("encode('</option>')=", tok.encode("</option>"), tok.convert_ids_to_tokens(tok.encode("</option>")))
print("encode('The answer is:') =", tok.encode("The answer is:"))

sub = ds["train"].select(range(min(3000, len(ds["train"]))))
sl, il, ol = [], [], []
for row in sub:
    sl.append(len(tok.encode(row["state"])))
    q = json.loads(row["questions_json"])
    for qq in q.values():
        il.append(len(tok.encode(qq.get("instructions", ""))))
        if qq["type"] == "choice":
            for k, v in qq["criteria"].items():
                ol.append(len(tok.encode(f"{k}: {v}" if v else k)))
        elif qq["type"] == "score":
            for i, lv in enumerate(qq["criteria"]):
                ol.append(len(tok.encode(f"{i}: {lv}")))
        else:
            ol.append(len(tok.encode("No")))
            ol.append(len(tok.encode("Yes")))
pct(sl, "state tokens")
pct(il, "instructions tokens")
pct(ol, "option tokens")

rule("7. STATE OVERLAP train/test (normalized exact match)")
def norm(s):
    return " ".join(s.lower().split())

train_states = set()
for row in ds["train"]:
    train_states.add(norm(row["state"]))
print("distinct normalized train states:", f"{len(train_states):,}")
hit = 0
for row in ds["test"]:
    if norm(row["state"]) in train_states:
        hit += 1
print(f"test rows whose state also appears in train: {hit:,} / {len(ds['test']):,}")

rule("8. LICENSE / TERMS")
print("dataset card front-matter license: 'unknown' (see bev-decision-README.md ref/hf/)")
print("bev-decider-0.4B weights license : cc-by-nc-4.0")
print("bev-train code license           : not declared in the repo (no LICENSE file)")
import os as _os
print("bev-train LICENSE file present   :", _os.path.exists(r"F:\Jev\ref\bev-train\LICENSE"))
print("bev-train pyproject license field:", "license" in open(r"F:\Jev\ref\bev-train\pyproject.toml").read())

rule("9. BENCHMARK AVAILABILITY SEARCH")
print("JevBench     : searching bev-train repo + dataset columns + HF ...")
print("sysone-bench : searching bev-train repo + dataset columns + HF ...")
for name in ["/jevbench", "jevbench", "instax-dutta/sysone-bench", "sysone-bench"]:
    try:
        i = HfApi().repo_info(name, repo_type="dataset")
        print(f"  HF dataset {name}: FOUND")
    except Exception as e:
        print(f"  HF dataset {name}: not found ({type(e).__name__})")
cols = set(ds["train"].column_names)
for probe in ["jevbench", "sysone", "benchmark", "suite", "split_name", "source"]:
    print(f"  dataset column {probe!r} present: {probe in cols}")

rule("10. DONE")
print("wrote recon/recon_out.txt")