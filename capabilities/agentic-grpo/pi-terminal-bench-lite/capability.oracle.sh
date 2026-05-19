#!/usr/bin/env bash
# capability.oracle.sh — Blind eval for pi-terminal-bench-lite.
#
# Usage: ./capability.oracle.sh <adapter_name>
#   <adapter_name> = "" means base model (no adapter loaded)
#
# Runs the 30-task held-out eval set and prints `SCORE=<composite>`
# plus per-sub-score breakdown.
set -euo pipefail

ADAPTER="${1:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$HERE/capability.config.json"
OUT_DIR="${OUT_DIR:-/tmp/pi-tblite-eval-$$}"

mkdir -p "$OUT_DIR"

python3 "$HERE/rollout.py" \
  --tasks "$HERE/datasets/eval.tasks.jsonl" \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR" \
  --mode eval \
  --adapter "$ADAPTER"

# rollout.py writes <OUT_DIR>/eval.json with {mean_composite, sub_scores, ...}
COMPOSITE=$(python3 -c "import json; d=json.load(open('$OUT_DIR/eval.json')); print(d['mean_composite'])")
echo "SCORE=$COMPOSITE"
echo "DETAILS=$OUT_DIR/eval.json"
