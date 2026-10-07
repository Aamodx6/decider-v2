# Dataset counts, verified 2026-10-07 (STEP 0 of the full-all plan)

All numbers below were measured locally via `count_all.py` / `split_probe.py` against
`avbiswas/bev-decision` (HF cache; local parquet at F:\Jev\bev-decision not used directly,
the config `all` files are the same hub files). Explode counts use the repo's
`dataset.explode_questions` (the exact loader used by train/inference).

## What matches the dataset card exactly (rows)
| measure | card | measured | status |
| --- | ---: | ---: | --- |
| `default` train rows | 125,614 | 125,614 | exact |
| `default` test rows | 24,386 | 24,386 | exact |
| `default` total rows | 150,000 | 150,000 | exact |
| `all` train rows | 231,332 | 231,332 | exact |
| `all` test rows | 44,856 | 44,856 | exact |
| `all` total rows | 276,188 | 276,188 | exact |

## Questions (the card's 296,582 is the `default` config, not `all`)
| measure | card | measured |
| --- | ---: | ---: |
| `default` questions | 296,582 (CHOICE 120,842 / NOUL 118,265 / SCORE 57,475) | 296,582 = 250,262 train + 46,320 test (exact, all three type counts consistent at config level) |
| `all` train questions | — | 422,551 (CHOICE 189,207 / NOUL 176,181 / SCORE 57,163) |
| `all` test questions | — | 80,019 (the same cap[less] full-test split the running evals use) |
| `all` total questions | — | 502,570 |

Cross-check: the parquet's own `question_count` column sums to 296,582 (`default`) and
502,570 (`all`) — identical to the repo explode counts, so neither count is inflated by a
loader/double-count bug; the `default` card table's \"{Total} 296,582 questions / 150,000 rows\"
row is simply scoped to the `default` config (which the same card says holds the 150K release),
while the `all` config card only states row counts.

## The earlier \"125k states -> 166k core questions\" reading
- \"125k\" almost certainly referred to the `default` TRAIN ROWS (125,614) or the
  default-train distinct normalized states (116,659; 140,007 is the whole release).
- \"166k core questions\" matches nothing measured here: `default` train has 250,262
  questions (train+test 296,582). It cannot be produced by explode_contacts counting either
  (parquet `question_count` agrees with explode to the question). Estimates suggest it came from
  an earlier local pass over a *filtered* view (e.g. the decider-v2 adapters' over-long-option
  filter) or a truncated/invalid listing. Origin of 166k could not be reproduced; treat it as
  stale/incorrect. **Correct figures for training are: `all` = 231,332 train rows ->
  422,551 train questions; test 44,856 rows -> 80,019 questions.**

## Proposed val carve-out (STEP 1), measured per config
Rule: md5(md5(normalized state)) % 200 == 0, state-grouped, carved from train
(deterministic, same rule for every config).
| config | train rows | val rows | val states |
| --- | ---: | ---: | ---: |
| default | 125,022 | 592 | 568 |
| hard_50k | 41,777 | 223 | 202 |
| numeric_temporal | 16,381 | 96 | 62 |
| skills | 34,465 | 162 | 140 |
| counterfactual_15k | 12,558 | 56 | 56 |
| **all** | **230,203** | **1,129** | **1,024** |

`all` val ≈ 0.49% of states -> ~2.0k val questions at config level; capped to ~2k questions
(eval budget) exactly as planned.
