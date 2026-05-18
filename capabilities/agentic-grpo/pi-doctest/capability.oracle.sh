#!/usr/bin/env bash
# Blind oracle for pi-doctest.
# Runs `rollout.py --mode eval` over the held-out eval task set with the
# given adapter loaded, prints SCORE=<composite> + sub-scores on stdout.
set -euo pipefail
ADAPTER="${1:-}"
TASKS="${TASKS:-datasets/eval.tasks.jsonl}"
KILN_URL="${KILN_URL:-http://localhost:8420}"
NUM_GEN="${NUM_GEN:-1}"        # eval = 1 sample at T=0 for determinism
MAX_WALL="${MAX_WALL:-120}"
OUT_DIR="${OUT_DIR:-/tmp/pi-doctest-eval-$$}"

if ! curl -sf "$KILN_URL/v1/models" > /dev/null 2>&1; then
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
  --parallel 1

# Print SCORE= and sub-scores in the same shape as opd capability.oracle.sh.
python3 - "$OUT_DIR/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"SCORE={s['mean_composite']:.4f}")
print(f"outcome={s['mean_composite']:.4f}")  # v0: outcome == composite
print(f"N={s['n_rollouts']}")
print(f"mean_wall_clock_s={s['mean_wall_clock_s']:.2f}")
PY
