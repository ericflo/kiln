#!/usr/bin/env bash
# capability.oracle.sh — blind eval driver for pi-context-aware-edits.
#
# This is a multi-turn Pi capability, so the oracle runs the Pi rollout
# harness in eval mode and normalizes its aggregate summary into the stable
# eval_summary.json shape. It intentionally does not inspect or print
# per-example rollouts.
set -euo pipefail
cd "$(dirname "$0")"

ADAPTER="${1:-}"
TASKS="${TASKS:-datasets/eval.tasks.jsonl}"
CONFIG="${CONFIG:-capability.config.json}"
KILN_URL="${KILN_URL:-http://localhost:8420}"
SEEDS="${SEEDS:-3}"
OUT_FILE="${OUT_FILE:-/tmp/pi-context-aware-edits-eval-${ADAPTER:-base}.json}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-/tmp/pi-context-aware-edits-eval-${ADAPTER:-base}-$$}"
MAX_WALL_CLOCK_S="${MAX_WALL_CLOCK_S:-120}"
LIMIT="${LIMIT:-0}"
BASELINE_JSON="${BASELINE_JSON:-}"

if ! curl -sf "$KILN_URL/v1/health" > /dev/null 2>&1; then
  echo "ORACLE_ERROR: kiln-server not reachable at $KILN_URL" >&2
  exit 2
fi

if [ ! -f "$TASKS" ]; then
  echo "ORACLE_ERROR: $TASKS missing — run build_corpus.py first" >&2
  exit 3
fi

python3 rollout.py \
  --tasks "$TASKS" \
  --out-dir "$EVAL_OUT_ROOT" \
  --config "$CONFIG" \
  --adapter "${ADAPTER:-}" \
  --num-generations "$SEEDS" \
  --mode eval \
  --max-wall-clock-s "$MAX_WALL_CLOCK_S" \
  --limit "$LIMIT"

python3 - "$EVAL_OUT_ROOT" "$OUT_FILE" "$BASELINE_JSON" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
out_file = Path(sys.argv[2])
baseline_json = sys.argv[3]

summary = json.loads((root / "summary.json").read_text())
records = [
    json.loads(line)
    for line in (root / "rollout.jsonl").read_text().splitlines()
    if line.strip()
]

composites = [float((r.get("score") or {}).get("composite", 0.0)) for r in records]
mean = sum(composites) / len(composites) if composites else 0.0
var = sum((x - mean) ** 2 for x in composites) / len(composites) if composites else 0.0

sub_totals = {}
sub_counts = {}
for r in records:
    for k, v in (r.get("score") or {}).items():
        if k == "composite":
            continue
        if isinstance(v, (int, float)):
            sub_totals[k] = sub_totals.get(k, 0.0) + float(v)
            sub_counts[k] = sub_counts.get(k, 0) + 1
sub_scores = {
    k: sub_totals[k] / sub_counts[k]
    for k in sorted(sub_totals)
    if sub_counts[k]
}

baseline_mean = None
if baseline_json:
    try:
        baseline_mean = float(json.loads(Path(baseline_json).read_text()).get("mean_composite"))
    except Exception:
        baseline_mean = None

out = {
    "schema_version": 1,
    "rubric_version": summary.get("rubric_version"),
    "status": "kept-with-caveat",
    "adapter": summary.get("adapter", ""),
    "n_tasks": summary.get("n_tasks"),
    "n_rollouts": summary.get("n_rollouts"),
    "n_generations": summary.get("n_generations"),
    "mean_composite": mean,
    "composite_stdev": math.sqrt(var),
    "composite_delta": (mean - baseline_mean) if baseline_mean is not None else None,
    "sub_scores_mean": sub_scores,
    "rollout_stats": {
        "mean_wall_clock_s": summary.get("mean_wall_clock_s"),
        "rollouts_nonzero": summary.get("rollouts_nonzero"),
        "rollouts_zero": summary.get("rollouts_zero"),
        "mean_tool_calls": summary.get("mean_tool_calls"),
        "mean_thinking_chars": summary.get("mean_thinking_chars"),
        "mean_thinking_blocks": summary.get("mean_thinking_blocks"),
        "mean_thinking_chars_per_tool_call": summary.get("mean_thinking_chars_per_tool_call"),
        "source_dir": str(root),
    },
    "verdict": "baseline" if not summary.get("adapter") else "needs_promotion_check",
}

out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

print(f"SCORE={mean:.4f}")
print(f"N={out['n_tasks']}")
for k, v in sub_scores.items():
    print(f"{k}={v:.4f}")
print(f"STDEV={out['composite_stdev']:.4f}")
if out["composite_delta"] is not None:
    print(f"DELTA={out['composite_delta']:+.4f}")
stats = out["rollout_stats"]
if stats["mean_tool_calls"] is not None:
    print(f"MEAN_TOOL_CALLS={stats['mean_tool_calls']:.2f}")
if stats["mean_thinking_chars"] is not None:
    print(f"MEAN_THINKING_CHARS={stats['mean_thinking_chars']:.1f}")
if stats["mean_thinking_chars_per_tool_call"] is not None:
    print(
        "MEAN_THINKING_CHARS_PER_TOOL_CALL="
        f"{stats['mean_thinking_chars_per_tool_call']:.1f}"
    )
PY
