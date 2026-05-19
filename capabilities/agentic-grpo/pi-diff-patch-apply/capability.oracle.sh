#!/bin/bash
# Blind eval oracle for pi-diff-patch-apply.
#
# Usage:
#   capability.oracle.sh <adapter-name>     # adapter on pod; e.g. ""/"base" = base model
#
# Runs the eval task set (24 held-out tasks) against the named adapter
# via the kiln server on the pod, then prints `SCORE=<mean_composite>`
# on stdout (eval-script-friendly).
#
# Requires: $POD_ID and $RP env (sourced from /tmp/grpo-pod.env on the
# local machine, see drive_iters.sh).

set -euo pipefail
ADAPTER="${1:-}"
if [ -z "$ADAPTER" ] || [ "$ADAPTER" = "base" ]; then
  ADAPTER_ARG=""
  ADAPTER_LABEL="base"
else
  ADAPTER_ARG="$ADAPTER"
  ADAPTER_LABEL="$ADAPTER"
fi

source /tmp/grpo-pod.env

OUT="/tmp/oracle-eval-$(echo "$ADAPTER_LABEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
POD_REPO=/workspace/kiln/capabilities/agentic-grpo/pi-diff-patch-apply

# Set adapter.
if [ -z "$ADAPTER_ARG" ]; then
  python3 $RP ssh $POD_ID 'curl -sS -X POST http://localhost:8420/v1/adapters/unload >/dev/null'
else
  python3 $RP ssh $POD_ID "curl -sS -X POST http://localhost:8420/v1/adapters/load -H 'Content-Type: application/json' -d '{\"name\":\"${ADAPTER_ARG}\"}'"
fi

# Run eval.
python3 $RP bg $POD_ID "/tmp/oracle-${ADAPTER_LABEL}.log" \
  "cd ${POD_REPO} && rm -rf ${OUT} && python3 rollout.py \
    --tasks datasets/eval.tasks.jsonl \
    --out-dir ${OUT} --mode eval --num-generations 1 \
    --adapter current --seed-base 3141592653 --parallel 2 \
    --max-turns 12 --max-wall-clock-s 240 --temperature 0.0 \
    --verbose 2>&1"
python3 $RP wait-file $POD_ID "${OUT}/summary.json" --timeout 7200

# Pull and print.
python3 $RP ssh $POD_ID "cat ${OUT}/summary.json"
SCORE=$(python3 $RP ssh $POD_ID "python3 -c 'import json; d=json.load(open(\"${OUT}/summary.json\")); print(d[\"mean_composite\"])'")
echo "SCORE=${SCORE}"
