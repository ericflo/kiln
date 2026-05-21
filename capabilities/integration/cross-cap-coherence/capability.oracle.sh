#!/usr/bin/env bash
# capability.oracle.sh — cross-cap coherence eval.
#
# Round 2 integration track. Eval one or more adapters against a
# held-out slice of every member cap's eval set; report cross-cap
# composite + per-cap deltas + regressions.
#
# Usage:
#   ./capability.oracle.sh                              # base only
#   ./capability.oracle.sh adapter-name                 # one adapter vs base
#   ./capability.oracle.sh adapter1 adapter2 adapter3   # multi-adapter compare
set -euo pipefail
cd "$(dirname "$0")"

CFG="capability.config.json"
KILN_URL="${KILN_URL:-http://localhost:8420}"
ADAPTER_DIR="${ADAPTER_DIR:-/workspace/adapters}"
SEEDS="${SEEDS:-3}"
OUT_FILE="${OUT_FILE:-/tmp/cross-cap-coherence-report.json}"
TASKS="datasets/integration_eval.tasks.jsonl"
KILN_BIN="${KILN_BIN:-kiln}"

# Check kiln server.
if ! curl -sf "$KILN_URL/v1/health" > /dev/null 2>&1; then
  echo "ORACLE_ERROR: kiln-server not reachable at $KILN_URL" >&2
  exit 2
fi

# Build the integration eval set if missing.
if [ ! -f "$TASKS" ]; then
  echo "Building integration eval set…"
  python3 build_corpus.py
fi

ADAPTERS=("$@")
if [ ${#ADAPTERS[@]} -eq 0 ]; then
  ADAPTERS=("")  # base only
fi

# For each adapter (and base if not skipped), run kiln eval-adapter against
# the integration eval set with the cross-cap rubric.
RESULTS_DIR="$(mktemp -d)"

# Always run base first.
echo "=== Evaluating BASE (no adapter) ==="
"$KILN_BIN" eval-adapter \
  --url "$KILN_URL" \
  --adapter "" \
  --adapter-dir "$ADAPTER_DIR" \
  --tasks "$TASKS" \
  --seeds "$SEEDS" \
  --scorer "./rubric.py" \
  --output "$RESULTS_DIR/base.json" \
  --thinking off

for ADAPTER in "${ADAPTERS[@]}"; do
  if [ -z "$ADAPTER" ]; then continue; fi
  echo "=== Evaluating adapter: $ADAPTER ==="
  "$KILN_BIN" eval-adapter \
    --url "$KILN_URL" \
    --adapter "$ADAPTER" \
    --adapter-dir "$ADAPTER_DIR" \
    --tasks "$TASKS" \
    --seeds "$SEEDS" \
    --scorer "./rubric.py" \
    --output "$RESULTS_DIR/${ADAPTER}.json" \
    --thinking off
done

# Aggregate: per-cap composites + cross-cap composite + per-cap deltas.
RESULTS_DIR="$RESULTS_DIR" \
ADAPTERS="${ADAPTERS[*]}" \
OUT_FILE="$OUT_FILE" \
TASKS="$TASKS" \
CFG="$CFG" \
python3 - <<'PY'
import json, os
from pathlib import Path
from collections import defaultdict

results_dir = Path(os.environ["RESULTS_DIR"])
cfg = json.loads(Path(os.environ["CFG"]).read_text())
tasks_path = Path(os.environ["TASKS"])
adapters = [a for a in os.environ["ADAPTERS"].split() if a]
regr_thresh = cfg.get("regression_threshold", -0.02)

# Map task index → member_cap.
member_of = []
with open(tasks_path) as f:
    for line in f:
        if line.strip():
            t = json.loads(line)
            member_of.append(t.get("_member_cap", "_unknown"))

def per_cap_composites(eval_summary):
    """Group per-task scores by _member_cap and average."""
    by_member = defaultdict(list)
    # kiln eval-adapter writes `per_task` with the rubric's score_one output per task.
    per_task = eval_summary.get("per_task") or []
    for i, ts in enumerate(per_task):
        member = member_of[i] if i < len(member_of) else "_unknown"
        # ts is the dict returned by rubric.score_one(). Look for composite or mean_composite.
        comp = ts.get("composite")
        if comp is None:
            comp = ts.get("mean_composite")
        if comp is None:
            continue
        by_member[member].append(float(comp))
    return {m: (sum(v) / len(v) if v else 0.0) for m, v in by_member.items()}

base_summary = json.loads((results_dir / "base.json").read_text())
base_per_cap = per_cap_composites(base_summary)
base_cross = sum(base_per_cap.values()) / max(1, len(base_per_cap))

report = {
    "kiln_commit": base_summary.get("kiln_commit"),
    "tasks_sha256": base_summary.get("tasks_sha256"),
    "regression_threshold": regr_thresh,
    "base": {
        "cross_cap_composite": base_cross,
        "per_cap": base_per_cap,
    },
    "per_adapter": {},
}

for adapter in adapters:
    fp = results_dir / f"{adapter}.json"
    if not fp.exists():
        continue
    s = json.loads(fp.read_text())
    per_cap = per_cap_composites(s)
    cross = sum(per_cap.values()) / max(1, len(per_cap))
    per_cap_delta = {m: per_cap.get(m, 0.0) - base_per_cap.get(m, 0.0) for m in per_cap}
    regressions = [m for m, d in per_cap_delta.items() if d < regr_thresh]
    report["per_adapter"][adapter] = {
        "cross_cap_composite": cross,
        "cross_cap_delta": cross - base_cross,
        "per_cap": per_cap,
        "per_cap_delta": per_cap_delta,
        "regressions": regressions,
    }

with open(os.environ["OUT_FILE"], "w") as f:
    json.dump(report, f, indent=2)

# Print summary.
print()
print(f"=== CROSS-CAP COHERENCE REPORT ===")
print(f"base cross_cap_composite: {base_cross:.4f}")
for adapter, r in report["per_adapter"].items():
    delta = r["cross_cap_delta"]
    sign = "+" if delta >= 0 else ""
    print(f"\n{adapter}: cross_cap={r['cross_cap_composite']:.4f} (delta {sign}{delta:.4f})")
    for m, d in sorted(r["per_cap_delta"].items()):
        marker = "  REGRESSION" if d < regr_thresh else ""
        sign_d = "+" if d >= 0 else ""
        print(f"  {m:30s} delta {sign_d}{d:.4f}{marker}")
    if r["regressions"]:
        print(f"  ** {len(r['regressions'])} regression(s): {r['regressions']}")

print(f"\nWrote: {os.environ['OUT_FILE']}")
PY
