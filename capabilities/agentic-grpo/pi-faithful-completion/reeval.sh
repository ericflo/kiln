#!/bin/bash
# Re-eval a specific adapter with the fixed rollout.py.
# Usage: reeval.sh <adapter_name> <iter_number>
set -uo pipefail
ADAPTER="$1"
ITER="$2"

source /tmp/pi-faithful.env

POD_REPO=/workspace/kiln/capabilities/agentic-grpo/pi-faithful-completion
OUT="/tmp/iter${ITER}-reeval"

# Load adapter
python3 $RP ssh $POD_ID "curl -sS -X POST http://localhost:8420/v1/adapters/load -H 'Content-Type: application/json' -d '{\"name\":\"${ADAPTER}\"}' >/dev/null"

# Run eval
python3 $RP bg $POD_ID /tmp/iter${ITER}-reeval.log \
  "cd ${POD_REPO} && rm -rf ${OUT} && python3 rollout.py \
    --tasks datasets/eval.tasks.jsonl \
    --out-dir ${OUT} --mode eval --num-generations 1 \
    --adapter ${ADAPTER} --temperature 0.2 --top-p 0.95 --max-tokens 768 \
    --seed 3141592653 --concurrency 3 --verbose 2>&1"
python3 $RP wait-file $POD_ID ${OUT}/summary.json --timeout 600

python3 $RP ssh $POD_ID "cat ${OUT}/summary.json"
