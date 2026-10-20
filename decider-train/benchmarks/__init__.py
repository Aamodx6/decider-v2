"""Benchmark adapters for decider_v2.

Both upstream harnesses (JevBench and sysone-bench) grade a model by mapping a distribution over a
task's exact label set.  The model here already produces exactly that, so an adapter is a thin
projection: encode the benchmark's (state, question) with `serialize.Encoder`, run one forward pass
through the same head the training run produced, and hand the resulting per-option probabilities to
the harness's OWN scorer.

The rule these adapters follow, taken from the harnesses' own contracts: never re-implement the
metric.  JevBench's `scoring.score_task` + `summarize.metric` grade JevBench, and sysone-bench's
`contracts.validate_answers` + `metrics.summarize_predictions` grade sysone-bench.  Everything here
is decode/encode plumbing and provenance recording.
"""