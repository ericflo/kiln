#!/bin/bash
# Drive pi-diff-patch-apply through N iters. Designed to be re-entrant —
# picks up at next iter based on capability.jsonl + START_ITER override.
#
# Sets POD_ID/LEASE_ID/RP in env from /tmp/grpo-pod.env, runs run_iter.sh
# for each iter, commits + pushes capability.jsonl, and B2-backs up.
#
# Usage:
#   bash drive_iters.sh --pod <pod_id> --max-iters 50 [--start-iter N]

set -euo pipefail

MAX_ITERS=50
START_ITER=""
POD_ID=""

while [ $# -gt 0 ]; do
  case "$1" in
    --pod) POD_ID="$2"; shift 2 ;;
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --start-iter) START_ITER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$POD_ID" ]; then echo "--pod required" >&2; exit 1; fi

export POD_ID
export RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Ensure /tmp/grpo-pod.env exists.
if [ ! -f /tmp/grpo-pod.env ]; then
  echo "export POD_ID=${POD_ID}" > /tmp/grpo-pod.env
  echo "export RP=${RP}" >> /tmp/grpo-pod.env
fi

next_iter() {
  if [ -n "$START_ITER" ]; then
    echo "$START_ITER"
    return
  fi
  if [ ! -s "$HERE/capability.jsonl" ]; then
    echo 0; return
  fi
  python3 -c "
import json
n = -1
for line in open('$HERE/capability.jsonl'):
    try:
        e = json.loads(line)
        v = e.get('iter')
        if isinstance(v, int):
            n = max(n, v)
    except Exception:
        pass
print(n + 1)
"
}

# Append-one-row helper. Use after each iter to log to capability.jsonl
# (run_iter.sh doesn't itself log because the JSONL row needs the eval
# composite which is computed AFTER backup).
append_log_row() {
  local n=$1 kind=$2 slug=$3 composite=$4 baseline=$5 delta=$6 status=$7
  python3 -c "
import json, datetime, pathlib
row = {
    'iter': $n,
    'kind': '$kind',
    'slug': '$slug',
    'composite': $composite,
    'baseline_composite': $baseline,
    'delta_vs_baseline': $delta,
    'status': '$status',
    'logged_at': datetime.datetime.utcnow().isoformat() + 'Z',
}
p = pathlib.Path('$HERE/capability.jsonl')
with p.open('a') as f:
    f.write(json.dumps(row) + chr(10))
"
}

# Recipe table — one line per iter, drives 50-iter sweep.
# Format: ITER KIND SLUG EXTRA_FLAGS
recipe_for() {
  local n=$1
  case "$n" in
    0)   echo "baseline baseline-v1 --skip-train --eval-adapter base" ;;
    1)   echo "train h1-default-recipe --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04" ;;
    2)   echo "train h2-strong-signal-filter --num-train-tasks 24 --num-gens 4 --lr 1e-5 --filter-var 0.05" ;;
    3)   echo "train h3-temperature-bump --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --temperature 1.0" ;;
    4)   echo "train h4-more-gens --num-train-tasks 12 --num-gens 8 --lr 1e-5 --filter-var 0.04" ;;
    5)   echo "train h5-lower-lr --num-train-tasks 16 --num-gens 4 --lr 5e-6 --filter-var 0.04" ;;
    6)   echo "train h6-higher-lr --num-train-tasks 16 --num-gens 4 --lr 2e-5 --filter-var 0.04" ;;
    7)   echo "train h7-rank-32 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --rank 32 --alpha 64" ;;
    8)   echo "ablation h8-no-echo --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --no-echo" ;;
    9)   echo "train h9-higher-echo-lambda --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --echo-lambda 0.10" ;;
    10)  echo "train h10-2epochs --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --epochs 2" ;;
    11)  echo "train h15-more-turns --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --max-turns 16" ;;
    12)  echo "train h16-fewer-turns --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --max-turns 8" ;;
    13)  echo "train h21-larger-corpus --num-train-tasks 30 --num-gens 4 --lr 1e-5 --filter-var 0.03" ;;
    14)  echo "train h22-pass-amplify --num-train-tasks 16 --num-gens 6 --lr 1e-5 --filter-var 0.02" ;;
    15)  echo "train h23-seed-2 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --seed 1729" ;;
    16)  echo "train h24-seed-3 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --seed 4242" ;;
    17)  echo "train h25-rank-8 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --rank 8 --alpha 16" ;;
    18)  echo "train h26-3epochs --num-train-tasks 16 --num-gens 4 --lr 5e-6 --filter-var 0.04 --epochs 3" ;;
    19)  echo "train h27-lower-temp --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --temperature 0.5" ;;
    20)  echo "train h28-bigger-batch --num-train-tasks 32 --num-gens 4 --lr 1e-5 --filter-var 0.04" ;;
    21)  echo "train h29-no-policy-loss --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --no-policy-loss" ;;
    22)  echo "train h30-rank-32-lr-2e-5 --num-train-tasks 16 --num-gens 4 --lr 2e-5 --filter-var 0.04 --rank 32 --alpha 64" ;;
    23)  echo "train h31-tiny-filter --num-train-tasks 24 --num-gens 4 --lr 1e-5 --filter-var 0.01" ;;
    24)  echo "train h32-large-filter --num-train-tasks 24 --num-gens 4 --lr 1e-5 --filter-var 0.10" ;;
    25)  echo "train h33-echo-0.02 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --echo-lambda 0.02" ;;
    26)  echo "train h34-echo-0.08 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --echo-lambda 0.08" ;;
    27)  echo "train h35-temp-0.6 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --temperature 0.6" ;;
    28)  echo "train h36-temp-1.2 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --temperature 1.2" ;;
    29)  echo "train h37-12gens --num-train-tasks 8 --num-gens 12 --lr 1e-5 --filter-var 0.02" ;;
    30)  echo "train h38-lr-warmup-up --num-train-tasks 16 --num-gens 4 --lr 7e-6 --filter-var 0.04" ;;
    31)  echo "train h39-lr-warmup-down --num-train-tasks 16 --num-gens 4 --lr 1.5e-5 --filter-var 0.04" ;;
    32)  echo "train h40-best-recipe-fresh-seed-1 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --seed 11111" ;;
    33)  echo "train h41-best-recipe-fresh-seed-2 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --seed 22222" ;;
    34)  echo "train h42-best-recipe-fresh-seed-3 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --seed 33333" ;;
    35)  echo "train h43-rank-64 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --rank 64 --alpha 128" ;;
    36)  echo "train h44-rank-4 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --rank 4 --alpha 8" ;;
    37)  echo "train h45-many-tasks-many-gens --num-train-tasks 24 --num-gens 8 --lr 1e-5 --filter-var 0.04" ;;
    38)  echo "train h46-lower-lr-many-tasks --num-train-tasks 30 --num-gens 4 --lr 5e-6 --filter-var 0.04" ;;
    39)  echo "train h47-2x-2epochs --num-train-tasks 16 --num-gens 4 --lr 5e-6 --filter-var 0.04 --epochs 2" ;;
    40)  echo "train h48-fewer-turns-tight --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --max-turns 6" ;;
    41)  echo "train h49-many-turns-loose --num-train-tasks 12 --num-gens 4 --lr 1e-5 --filter-var 0.04 --max-turns 20" ;;
    42)  echo "train h50-best-large-seed --num-train-tasks 32 --num-gens 4 --lr 1e-5 --filter-var 0.04 --seed 31415" ;;
    43)  echo "train h51-temp-bump-many-gens --num-train-tasks 12 --num-gens 8 --lr 1e-5 --filter-var 0.04 --temperature 1.0" ;;
    44)  echo "train h52-tight-clip-1e-5 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04" ;;
    45)  echo "train h53-bigger-corpus-rank-32 --num-train-tasks 32 --num-gens 4 --lr 1e-5 --filter-var 0.04 --rank 32 --alpha 64" ;;
    46)  echo "train h54-replay-best-1 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --seed 55555" ;;
    47)  echo "train h55-replay-best-2 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --seed 66666" ;;
    48)  echo "train h56-replay-best-3 --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04 --seed 77777" ;;
    49)  echo "train h57-final-mass --num-train-tasks 40 --num-gens 4 --lr 1e-5 --filter-var 0.04" ;;
    *)   echo "train h-default --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.04" ;;
  esac
}

iter=$(next_iter)
end=$((iter + MAX_ITERS))
echo "Driving iters [$iter, $end)"

while [ "$iter" -lt 50 ] && [ "$iter" -lt "$end" ]; do
  RECIPE=$(recipe_for "$iter")
  KIND=$(echo "$RECIPE" | awk '{print $1}')
  SLUG=$(echo "$RECIPE" | awk '{print $2}')
  EXTRA=$(echo "$RECIPE" | cut -d' ' -f3-)

  echo "==============================================="
  echo "  ITER $iter  kind=$KIND  slug=$SLUG"
  echo "  flags: $EXTRA"
  echo "==============================================="

  EVAL_ADAPTER="pi-diff-patch-apply-iter${iter}"
  if [ "$iter" = "0" ]; then EVAL_ADAPTER="base"; fi

  bash "$HERE/run_iter.sh" --iter "$iter" --kind "$KIND" --eval-adapter "$EVAL_ADAPTER" $EXTRA 2>&1 | tail -200

  # Extract composite from the eval summary.
  EVAL_OUT="/tmp/iter${iter}-eval"
  COMPOSITE=$(python3 $RP ssh $POD_ID "python3 -c 'import json; d=json.load(open(\"${EVAL_OUT}/summary.json\")); print(d[\"mean_composite\"])'" 2>/dev/null || echo "null")
  if [ "$iter" = "0" ]; then
    BASELINE=$COMPOSITE
    echo "$BASELINE" > "$HERE/.baseline_composite"
  else
    BASELINE=$(cat "$HERE/.baseline_composite" 2>/dev/null || echo "null")
  fi
  DELTA="null"
  if [ "$COMPOSITE" != "null" ] && [ "$BASELINE" != "null" ]; then
    DELTA=$(python3 -c "print(${COMPOSITE} - ${BASELINE})")
  fi
  STATUS="logged"
  echo "  iter ${iter}: composite=${COMPOSITE} baseline=${BASELINE} delta=${DELTA}"
  append_log_row "$iter" "$KIND" "$SLUG" "$COMPOSITE" "$BASELINE" "$DELTA" "$STATUS"

  # Commit + push.
  cd "$HERE/../../.."
  git add capabilities/agentic-grpo/pi-diff-patch-apply/capability.jsonl || true
  git commit -m "cap[agentic-grpo/pi-diff-patch-apply]: iter ${iter} ${SLUG} (composite=${COMPOSITE}, Δ=${DELTA})" || true
  git pull --rebase origin main || true
  git push origin main || true
  cd - >/dev/null

  iter=$((iter + 1))
done

echo "==============================================="
echo "  DRIVE COMPLETE (next iter would be $iter)"
echo "==============================================="
