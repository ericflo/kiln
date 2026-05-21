#!/usr/bin/env bash
# capability.oracle.sh — blind eval driver for python-algo.
#
# Wraps `kiln eval-adapter` (KILN_IMPROVEMENT_ISSUES.md #33) — multi-seed,
# paired eval, base-vs-adapter comparison, sigma-vs-lift warning, writes a
# stable eval_summary.json schema.
#
# Usage:
#   ./capability.oracle.sh             # eval base
#   ./capability.oracle.sh my-adapter  # eval one adapter
#   SEEDS=5 ./capability.oracle.sh my-adapter
set -euo pipefail
cd "$(dirname "$0")"

ADAPTER="${1:-}"
TASKS="${TASKS:-datasets/eval.tasks.jsonl}"
KILN_URL="${KILN_URL:-http://localhost:8420}"
SEEDS="${SEEDS:-3}"
ADAPTER_DIR="${ADAPTER_DIR:-/workspace/adapters}"
SCORER="${SCORER:-./rubric.py}"
OUT_FILE="${OUT_FILE:-/tmp/python-algo-eval-${ADAPTER:-base}.json}"
KILN_BIN="${KILN_BIN:-kiln}"

if ! curl -sf "$KILN_URL/v1/health" > /dev/null 2>&1; then
  echo "ORACLE_ERROR: kiln-server not reachable at $KILN_URL" >&2
  exit 2
fi

if [ ! -f "$TASKS" ]; then
  echo "ORACLE_ERROR: $TASKS missing — run build_corpus.py first" >&2
  exit 3
fi

"$KILN_BIN" eval-adapter \
  --url "$KILN_URL" \
  --adapter "${ADAPTER:-}" \
  --adapter-dir "$ADAPTER_DIR" \
  --tasks "$TASKS" \
  --seeds "$SEEDS" \
  --scorer "$SCORER" \
  --output "$OUT_FILE" \
  --thinking off

python3 - "$OUT_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
mc = d.get("mean_composite")
n  = d.get("n_tasks")
print(f"SCORE={mc:.4f}")
print(f"N={n}")
for k, v in (d.get("sub_scores_mean") or {}).items():
    print(f"{k}={v:.4f}")
stdev = d.get("composite_stdev")
if stdev is not None:
    print(f"STDEV={stdev:.4f}")
PY
