---
pretty_name: BEV Decision Mix
language:
- en
license: unknown
size_categories:
- 100K<n<1M
task_categories:
- text-classification
tags:
- decision-making
- structured-prediction
- multiple-choice
- boolean-classification
- ordinal-regression
- tool-routing
- source-grounded
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path: data/train.parquet
  - split: test
    path: data/test.parquet
- config_name: hard_50k
  data_files:
  - split: train
    path: hard_50k/train.parquet
  - split: test
    path: hard_50k/test.parquet
- config_name: numeric_temporal
  data_files:
  - split: train
    path: numeric_temporal/train.parquet
  - split: test
    path: numeric_temporal/test.parquet
- config_name: skills
  data_files:
  - split: train
    path: skills/train.parquet
  - split: test
    path: skills/test.parquet
- config_name: counterfactual_15k
  data_files:
  - split: train
    path: counterfactual_15k/train.parquet
  - split: test
    path: counterfactual_15k/test.parquet
- config_name: all
  data_files:
  - split: train
    path:
    - data/train.parquet
    - hard_50k/train.parquet
    - numeric_temporal/train.parquet
    - skills/train.parquet
    - counterfactual_15k/train.parquet
  - split: test
    path:
    - data/test.parquet
    - hard_50k/test.parquet
    - numeric_temporal/test.parquet
    - skills/test.parquet
    - counterfactual_15k/test.parquet
---

# BEV Decision Mix

*Formerly `avbiswas/bev-decision-150K`; the old ID redirects here.*

The `default` config holds **150,000 English-language decision rows** drawn from existing datasets, public
records, and text-based game environments. Each row presents source-grounded
input state and one or more bounded decisions inspired by the JEV `CHOICE`,
`NOUL`, and `SCORE` contracts. The mixture aims to train models that select an
option, judge a proposition, or assign an ordered score without generating a
long free-form answer.

| Split | Rows |
| --- | ---: |
| Train | 125,614 |
| Test | 24,386 |

> **Configs.** The `default` config is the original 150K release,
> byte-for-byte, so `load_dataset("avbiswas/bev-decision")` returns the
> original rows. The same files are also pinned at revision `v1.0-150k`. Four later batches are separate configs:
> `hard_50k`, `numeric_temporal`, `skills`, and `counterfactual_15k`. The `all` config loads all 276,188 rows
> (see [Additional configs](#additional-configs)).

There is **no validation split**. The split counts differ slightly from the
four input batches because 169 rows were reassigned when packaging: 166
Upworthy rows moved from test to train and three rows moved from train to test
to keep repeated input states and related source groups on one side. No rows
were added or dropped. The original four batches remain unchanged. The
release contains 140,007 distinct normalized input states; repeated states
within a split can carry different decision questions.

## Decision formats

Each decision row has a `state` string and a `questions_json` string
containing a `questions` object keyed by meaningful decision names. Question
types are:

| Type | Target | Questions | Rows containing it | Share of all questions |
| --- | --- | ---: | ---: | ---: |
| **CHOICE** | Select a key from a bounded `criteria` map | 120,842 | 90,946 | 40.7% |
| **NOUL** | Judge a yes/no proposition | 118,265 | 63,968 | 39.9% |
| **SCORE** | Select an index on an ordered, described scale | 57,475 | 44,134 | 19.4% |
| **Total** | | **296,582** | **150,000 unique rows** | **100%** |

The number of questions exceeds the number of rows because a state can have
several decision targets. For CHOICE, `criteria` maps stable answer keys to
option descriptions and `label` is one key. For NOUL, `label` is a boolean.
For SCORE, `criteria` is an ordered list of anchored levels and `label` is its
zero-based index. A SCORE label is an ordinal target; it is **not** a
calibrated probability or an unconstrained numeric prediction.

## Domains and skills

The table uses one primary domain per row. It describes the actual released
mixture, not the sizes of the upstream datasets. Each link leads to an
original source or source collection.

| Domain | Rows | Share | Example decisions and upstream sources |
| --- | ---: | ---: | --- |
| Sentiment, emotion, and moderation | 25,156 | 16.8% | Sentiment intensity, emotion, and toxicity from [SST-5](https://huggingface.co/datasets/SetFit/sst5), [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions), and [Civil Comments](https://huggingface.co/datasets/google/civil_comments) |
| Retail, product, and shopping | 19,883 | 13.3% | Purchase/cancellation outcomes, product categories, and review ratings from [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), [Online Shoppers](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), [Product Classification](https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering), and [Recipe Reviews](https://archive.ics.uci.edu/dataset/911/recipe+reviews+and+user+feedback+dataset) |
| Spatial and logical reasoning | 19,579 | 13.1% | Spatial relations and true/false/unknown inference from [SpaRTQA](https://github.com/HLR/SpartQA-baselines), [SpaRP/SpaRTUN](https://huggingface.co/datasets/UKPLab/sparp), and [ProofWriter](https://huggingface.co/datasets/rlhf-and-friends/proofwriter) |
| Support and intent routing | 12,656 | 8.4% | Classify a request and route it to a service area using [CLINC150](https://github.com/clinc/oos-eval), [MASSIVE](https://github.com/alexa/massive), and [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) |
| Tool and workflow decisions | 10,224 | 6.8% | Choose a service/tool and some enum or boolean arguments from [Taskmaster-1](https://github.com/google-research-datasets/Taskmaster/tree/master/TM-1-2019) and [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) |
| Scientific and paper understanding | 9,133 | 6.1% | Citation intent, claim evidence, and scientific yes/no judgments from [SciCite](https://github.com/allenai/scicite), [SciFact](https://github.com/allenai/scifact), and [SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF) |
| Financial reporting and banking | 9,003 | 6.0% | Report-table magnitude and observed term-deposit outcomes from [TAT-QA](https://github.com/NExTplusplus/TAT-QA) and [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| Response preference and quality | 8,693 | 5.8% | Choose a preferred response or an ordered human/reward rating from [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) and [UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) |
| Software engineering and code | 8,186 | 5.5% | Select test transitions or assertion outcomes from [SWE-smith-py](https://huggingface.co/datasets/SWE-bench/SWE-smith-py) and [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) |
| Software security | 7,000 | 4.7% | Source-assigned CVSS severity from [NIST NVD](https://nvd.nist.gov/vuln/data-feeds) descriptions |
| Browser interaction | 4,786 | 3.2% | Pick a demonstrated page element from [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) candidates |
| Engagement and ranking | 4,248 | 2.8% | Compare measured headline click-through rates from the [Upworthy Archive](https://upworthy.natematias.com/about-the-archive.html) |
| Reading comprehension | 3,619 | 2.4% | Answer grounded yes/no reading questions from [BoolQ](https://huggingface.co/datasets/google/boolq) |
| Contract evidence | 2,000 | 1.3% | Supported, contradicted, or unmentioned claims from [ContractNLI](https://github.com/stanfordnlp/contract-nli) |
| Game-state decisions | 2,000 | 1.3% | Immediate action and score decisions from [TextWorldExpress](https://github.com/cognitiveailab/TextWorldExpress), [ScienceWorld](https://github.com/allenai/ScienceWorld), and [OpenSpiel](https://github.com/google-deepmind/open_spiel) |
| Spam detection | 1,399 | 0.9% | SMS screening from the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| Sensory quality rating | 1,300 | 0.9% | Ordered wine quality from [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| Civic and safety operations | 1,135 | 0.8% | Service routing, recall class/remedy, and deadline policy from [NYC 311](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/data), [FDA](https://open.fda.gov/apis/food/enforcement/), [CPSC](https://www.cpsc.gov/Recalls/CPSC-Recalls-Application-Program-Interface-API-Information), and [CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) |

## Construction and provenance

Four independently built batches contribute 50,000; 19,997; 50,000; and
30,003 rows. Most labels come from upstream annotations, recorded outcomes,
test transitions, simulator replays, or declared deterministic policies.
In one 20,000-row slice, GPT-6 Luna rewrote bounded decision questions and
selected source quotes from existing records; source-backed labels were kept.
Other smaller Luna steps named existing response options or produced
date-checkable deadline questions. The model did **not** author new primary
source passages for this release. Some *upstream* corpora are themselves
synthetic, template-generated, or model-generated; this is not a collection
of exclusively human-written prompts.

The published rows include a primary `domain` category. The domain table
above links the upstream sources. Row-level source identifiers, raw source
downloads, and generation scripts are not included in this release.

## Loading the data

`data/train.parquet` and `data/test.parquet` are the Hub-compatible viewer
files. Their five columns are `state`, `questions_json`, `domain`,
`question_types`, and `question_count`. The JSON column contains the flexible
JEV-like question object. Read the decision questions with:

```python
import json
from datasets import load_dataset

ds = load_dataset("avbiswas/bev-decision")
example = ds["train"][0]
state = example["state"]
questions = json.loads(example["questions_json"])
```

The `default` config contains only the two Parquet data files. Parquet is used because
the heterogeneous nested question objects cannot be reliably inferred as one
Arrow JSON schema.

## Additional configs

| Config | Train | Test | Questions (CHOICE / NOUL / SCORE) | Focus |
| --- | ---: | ---: | --- | --- |
| `default` | 125,614 | 24,386 | 120,842 / 118,265 / 57,475 | The original 150K release described above |
| `hard_50k` | 42,000 | 8,000 | 51,321 / 37,724 / 480 | Multi-hop lookup, confusable routing, temporal/numeric, trap cases, unsettled evidence, answer adequacy |
| `numeric_temporal` | 16,477 | 3,329 | 18,004 / 17,204 / 4,158 | Time zones, elapsed time, warranty windows, proration, arithmetic, unit conversions, time-scoped facts |
| `skills` | 34,627 | 6,755 | 27,493 / 30,204 / 4,400 | Rule-in-the-question decisions, calibrated probabilities (soft labels), final-value extraction, long contracts, adversarial content, JSON states, paraphrase twins, judging correct-but-unusual answers |
| `counterfactual_15k` | 12,614 | 2,386 | 7,220 / 5,280 / 2,500 | Label-flipping pairs across routing, rule application, answer adequacy, final-plan extraction, numeric thresholds, and ordered scales |
| `all` | 231,332 | 44,856 | — | All five |

```python
from datasets import load_dataset

hard = load_dataset("avbiswas/bev-decision", "hard_50k")
numeric = load_dataset("avbiswas/bev-decision", "numeric_temporal")
counterfactual = load_dataset("avbiswas/bev-decision", "counterfactual_15k")
everything = load_dataset("avbiswas/bev-decision", "all")
```

All configs share the five-column schema. No state appears in both the train
and test splits across any config. When a later batch repeats a state from an
earlier config (with new questions), it keeps that state's split.

**`hard_50k` (50,000 rows)**

| Family | Rows | Sources |
| --- | ---: | --- |
| Multi-hop | 11,000 | [MuSiQue](https://github.com/StonyBrookNLP/musique), [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop), generated policy rulebooks |
| Temporal/numeric | 9,000 | [FinQA](https://github.com/czyssrs/FinQA), TAT-QA, a time-zone/business-day/unit generator |
| Trap cases | 8,500 | [CondaQA](https://github.com/AbhilashaRavichander/CondaQA), procedural access logs, and quoted/negated/misleading-subject/injected-instruction pairs over CLINC150/BANKING77/MASSIVE requests |
| Answer adequacy | 8,000 | [PRM800K](https://github.com/openai/prm800k) first incorrect step, planted-flaw answers over MuSiQue/FinQA, executed MBPP mutants |
| Short routing | 7,000 | Confusable-neighbor CLINC150/BANKING77/MASSIVE options with definitions, and handler catalogs |
| Unsettled evidence | 6,500 | MuSiQue unanswerable twins, 2Wiki and ContractNLI deleted-evidence twins |

**`numeric_temporal` (19,806 rows)**

| Source | Rows | Decisions |
| --- | ---: | --- |
| [TimeQA](https://github.com/wenhuchen/Time-Sensitive-QA) | 8,873 | Which fact held at a stated time; `none_at_that_time` for unanswerable questions |
| Generator, template renderings | 3,977 | 43 calculation types across short, medium and long templates |
| Generator, model-written renderings | 3,997 | The same calculation types, written as free-form messages |
| FinQA | 2,604 | Exact executed values, near-threshold checks, change-size bands |
| TAT-QA | 355 | Exact values over report tables |

**`skills` (41,382 rows).** Sources: CLINC150/BANKING77/MASSIVE, ContractNLI, and FDA recalls with decisive rules added to the instructions; Civil Comments, GoEmotions, and [Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) annotator shares, plus UCI outcome tables, as soft labels; Taskmaster-2/3 dialogues for final-value extraction; [CUAD](https://www.atticusprojectai.org/cuad) and [MAUD](https://www.atticusprojectai.org/maud) contracts and long generated rulebooks (states up to ~17K characters); [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) inserted as untrusted content; JSON re-layouts; paraphrase twins; and PRM800K/FinQA/MBPP answer judging. 6,993 rows (calibration) carry a `label_probs` field inside the question: the probability of each answer, with `label` equal to its argmax. Filter long rows at load time if your context is short.

**`counterfactual_15k` (15,000 rows).** This separate batch has 7,500 paired decisions, with both sides of each pair in the same split. It combines a new seed of the [Decider rules generator](https://github.com/Mapika/decider), short routing requests from CLINC150/BANKING77/MASSIVE and BEV handler catalogs edited by GPT-6 Luna, and local procedural adequacy, final-plan, threshold, and ordinal cases. Rule and procedural labels are computed in code. Routing edits passed a blind Luna label check, a separate Sol quality screen, and a second blind Sol classification. The public Parquet files omit pair IDs and provenance metadata; those remain in the local JSONL audit. Model agreement is not a guarantee that every routing label is correct. The procedural families retain a synthetic wording style.

**How labels were made.** In `hard_50k` and `numeric_temporal`, labels come from source
annotations, executed programs, time-zone computation (IANA tz database), or
rules evaluated in code. GPT-6 Luna never assigned a label. It wrote text in
9,576 `hard_50k` rows and 3,997 `numeric_temporal` rows; every output passed
deterministic checks and a separate model check, and failures were dropped.

**Caveats.**
- Deletion twins are checked for the absence of the deciding value, not for
  every possible implication.
- Generator rows share a template house style. In `numeric_temporal`, 28 of
  172 calculation-type/rendering combinations are test-only, so the test
  measures transfer to unseen templates.
- TimeQA "wrong period" checks assume its annotated answers are complete.
- The FinQA and TAT-QA dev sets are used for training in `numeric_temporal`.
- A MuSiQue sub-question can appear in both splits of `hard_50k`.
- `counterfactual_15k` was screened against the 231 public JevBench states and the extra validation set for exact and high word-8-gram overlap; this does not rule out every semantic similarity.

**Added sources and licenses.**
- MuSiQue, TAT-QA, ContractNLI, MASSIVE and MBPP: CC BY 4.0
- CLINC150: CC BY 3.0
- 2WikiMultihopQA and CondaQA: Apache-2.0
- Decider-generated rule cases: Apache-2.0
- FinQA and PRM800K: MIT
- TimeQA: BSD-3-Clause, but its passages are Wikipedia text (CC BY-SA), as
  with MuSiQue, 2Wiki and BoolQ

No non-commercial source was added.

## Intended use and limits

This mixture is for training and studying **bounded decision models**. Its
test split is a held-out portion of the mixture, not an independent benchmark
suite. It includes existing benchmark families, so evaluating on those same
public benchmarks requires checking record and source-family overlap first.
Some choice alternatives are sampled or constructed, and answer frequencies
were sometimes rebalanced; neither the test split nor SCORE outputs imply
real-world calibration.

Several labels have narrower meanings than their names may suggest: a bank
subscription is an observed outcome, not a causal recommendation to contact
that customer; CVSS base severity is not asset-specific risk; game scores
describe immediate transitions, not long-run expected return; and
Taskmaster service choices are derived from annotated dialogues, not logs of
executed tools. Some source judgments and LLM-rewritten questions may be
noisy. English filtering was automated and should not be read as a guarantee
that every excerpt is perfect English.

## Attribution and redistribution

**No single blanket license is asserted for this mixed-source release.**
The Hub metadata uses `license: unknown` for that reason. Consult the linked
upstream sources, their papers, and their current terms before public
redistribution or commercial use, and attribute every source used. In
particular, the ProofWriter mirror leaves upstream redistribution rights to
the user to confirm; NVD descriptions may be authored by third-party CVE
reporters; and the GoEmotions release includes Reddit text. The generated
question wording does not erase those upstream obligations.

Please cite [this dataset's Hub page](https://huggingface.co/datasets/avbiswas/bev-decision) **and** the relevant upstream
datasets or papers linked in the domain table above when using a subset.
