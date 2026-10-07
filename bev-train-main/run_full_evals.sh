#!/bin/bash
# Full test-split (80,019 questions, --max_questions 0) eval for every small-ladder run.
# Sequential on purpose: 2-way concurrency measured slower (GPU already ~100% utilized)
# and concurrent HuggingFace dataset loads raced on the cache.
cd "$(dirname "$0")" || exit 1
PROG=runs/full-eval-progress.log

declare -A RUNS=(
  [lora_mlp]=ladder-lora_mlp-20261006-221221
  [baseline]=ladder-baseline-20261006-215105
  [loss]=ladder-loss-20261006-215757
  [softlabels]=ladder-softlabels-20261006-234445
  [layers24]=ladder-layers24-20261006-222936
  [layers28]=ladder-layers28-20261006-231801
  [ctx]=ladder-ctx-20261006-220428
)
ORDER=(lora_mlp baseline loss softlabels layers24 layers28 ctx)

overall=0
for name in "${ORDER[@]}"; do
  run=${RUNS[$name]}
  out=runs/ladder-$name-full-eval.txt
  echo "START $name $(date +%FT%T)" >> "$PROG"
  t0=$(date +%s)
  uv run --no-sync python inference.py "runs/$run/checkpoints/best" \
    --config all --max_questions 0 --batch_size 8 --num_workers 0 > "$out" 2>&1
  rc=$?
  t1=$(date +%s)
  if grep -q "{'loss'" "$out" 2>/dev/null; then ok=OK; else ok=FAIL; overall=1; fi
  echo "DONE $name rc=$rc $ok $((t1-t0))s $(date +%FT%T)" >> "$PROG"
done
echo "ALL_DONE overall=$overall $(date +%FT%T)" >> "$PROG"
exit $overall
