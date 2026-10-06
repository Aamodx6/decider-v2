# NOTES — decider-v2 M0

Working notes for Milestone M0: assumptions, environment, reproduction
commands, limitations and results. Filled in as steps land; results are only
ever filled with real numbers from real runs.

## Assumptions

(recorded as they are made; every one is behind a config flag or a documented
default)

- **`state` passes through verbatim** (step 6): bev states are strings and the
  tokenizer renders string states verbatim (`as_text`), so the adapter never
  re-dumps state text through `json.dumps` — that would silently reorder keys
  of JSON-string states. Only non-string values (none exist in the parquet)
  get canonical JSON rendering.
- **`dev_size` counts dataset rows, not questions** (step 6): the carve-out is
  row-level so a repeated state (140,007 distinct states over 150,000 rows)
  never straddles train/dev; each row expands into ~2 questions, so the dev
  question count is ~2× `dev_size` (2,000 rows → ~3.9k questions on the real
  train split).
- **Dev carve-out = online Bernoulli sampling** keyed on
  `sha1(run_name|state)` (step 6): row *i* is taken with probability
  `(need)/(remaining)`, so the count lands exactly on `dev_size` with no head
  bias and no early-exit artifact; a different `run_name` yields an
  independent carve-out.
- **Option-token filter checks the single longest option text** (step 6):
  upper-bounds every option block the tokenizer builds for that question
  (equivalent to checking all, one encode instead of k).
- **Filtered examples are dropped, not raised** (step 6): over-long options
  (4,120 of 250,262 questions on the real train split, 1.6%) and
  out-of-bounds option counts are counted and logged once per split; the
  dataset is large enough that a smaller kept set is harmless.
- **bev-decision has no validation split** (dataset README) — the dev set is
  carved out of train as above; the `test` parquet is held out and never
  touched by training or by the dev carve-out.
- **State normalization for dedup/overlap** (step 7): NFKC + lowercase +
  whitespace collapse, nothing else (no punctuation folding) — conservative
  and reproducible.
- **Eval-leak policy** (step 7): `state_overlap()` reports train<->test
  normalized exact-match overlap but never mutates the training set; removal
  decisions are M1's (dataset audit), made explicitly.
- **Over-long options: train drops, eval truncates** (step 7):
  `load_test(truncate_options=True)` (default) keeps every held-out question
  and truncates option text to `max_option_tokens` (the tokenizer appends
  `<opt_end>` outside the cap, so the readout token always survives);
  training keeps the drop behaviour so the model never learns from clipped
  evidence. On the real test split this keeps **546** questions that were
  previously dropped (46,320 vs 45,774).
- **`max_option_tokens_used` semantics** (step 7): now records the longest
  option's raw token count even when it exceeds the budget (0 for noul
  without criteria text), instead of only counting kept examples.

## Environment

(to be recorded in step 10)

## Reproduction

(to be recorded in step 10)

## Known limitations

(to be recorded in step 10)

## Results

(only real numbers from real runs are recorded here)

### Train<->test state overlap (step 7, normalized exact match)

- train=116,653 distinct normalized states, test=23,352, **overlap=1**:
  `'dumber than a 5th grader ...check my poll'` (an Upworthy headline). The
  dataset README claims no raw state appears in both splits; this one differs
  only up to normalization (case/whitespace), so it is a near-duplicate leak
  of exactly one state (~2 questions). Recorded for the M1 dataset audit;
  not removed from training data.
- Equivalence gate on the real backbone (step 7 re-run): joint vs separate
  max abs diff **8.20e-05** (T=143, K=4, fp32/sdpa, 20 layers + LoRA) —
  within the 1e-4 hard gate.

### Real-data audit (step 6, `data/train.parquet`, 125,614 rows)

- Question types (20k-row sample, expanded): `choice` 23,139 (labels: str,
  2–8 options per criteria map), `noul` 6,837 (labels: bool), `score` 11,416
  (labels: zero-based int < len(criteria)). No extra fields, no missing
  `instructions`, no malformed labels found — every question is well-typed.
- Full-train adapter pass: **kept 246,142 questions, skipped 4,120** (1.6%,
  all `option_tokens` — option text over the 64-token budget). Test split
  passes with the same filters for the held-out eval.
- Adapter test suite: 19 tests (unit fixtures + real-data integration),
  full suite 51 passed.
