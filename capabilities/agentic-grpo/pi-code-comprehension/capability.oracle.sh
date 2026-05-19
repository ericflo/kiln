#!/usr/bin/env bash
# Blind eval oracle for pi-code-comprehension.
# Usage: ./capability.oracle.sh [adapter-name]
set -euo pipefail
ADAPTER="${1:-}"
TASKS="${TASKS:-datasets/eval.tasks.jsonl}"
KILN_URL="${KILN_URL:-http://localhost:8420}"
NUM_GEN="${NUM_GEN:-1}"
MAX_WALL="${MAX_WALL:-180}"
OUT_DIR="${OUT_DIR:-/tmp/pi-code-comp-eval-$$}"

if ! curl -sf "$KILN_URL/v1/models" >/dev/null 2>&1; then
  echo "ORACLE_ERROR: kiln-server not reachable at $KILN_URL" >&2
  exit 2
fi

python3 rollout.py \
  --tasks "$TASKS" \
  --out-dir "$OUT_DIR" \
  --adapter "$ADAPTER" \
  --num-generations "$NUM_GEN" \
  --mode eval \
  --kiln-url "$KILN_URL" \
  --max-wall-clock-s "$MAX_WALL" \
  --concurrency 1 \
  --verbose

python3 - "$OUT_DIR/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"SCORE={s['mean_composite']:.4f}")
print(f"outcome={s['mean_outcome']:.4f}")
print(f"grounding={s['mean_grounding']:.4f}")
print(f"cross_file_caller_recall={s['mean_cross_file_caller_recall']:.4f}")
print(f"invariant_coverage={s['mean_invariant_coverage']:.4f}")
print(f"format_compliance={s['mean_format_compliance']:.4f}")
print(f"N={s['n_rollouts']}")
print(f"mean_wall_clock_s={s['mean_wall_clock_s']:.2f}")
PY
