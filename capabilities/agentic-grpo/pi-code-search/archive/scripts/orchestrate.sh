#!/usr/bin/env bash
# Drive a single iter on the pod.
#
# Reads ITER, SLUG, and hyperparams from env; writes
#   capability.jsonl  ← appended one row
#   hypotheses/<slug>.md ← optional verdict
# and tars+uploads the adapter to B2 (if positive vs baseline).
set -euo pipefail

CAP_DIR=/workspace/kiln/capabilities/agentic-grpo/pi-code-search
cd "$CAP_DIR"

: "${ITER:?ITER required}"
: "${SLUG:?SLUG required}"

LOG_DIR="/workspace/iter${ITER}-logs"
mkdir -p "$LOG_DIR"

# Run the actual iter pipeline.
bash run_iter.sh 2>&1 | tee "$LOG_DIR/run_iter.log"

# Extract summary metrics and append to capability.jsonl.
BASE_DIR="${BASE_DIR:-/workspace/pi-code-search-iter${ITER}}"
EVAL_SUMMARY="$BASE_DIR/eval/summary.json"

python3 - "$ITER" "$SLUG" "$EVAL_SUMMARY" "$BASE_DIR/manifest.json" <<'PY'
import json, sys, os, time, subprocess
iter_n = int(sys.argv[1])
slug = sys.argv[2]
summary_path = sys.argv[3]
manifest_path = sys.argv[4]
if not os.path.exists(summary_path):
    print(f"WARN: no eval summary at {summary_path}; skipping log append")
    sys.exit(0)
s = json.load(open(summary_path))
m = json.load(open(manifest_path))
git_sha = subprocess.run(["git", "-C", "/workspace/kiln", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
row = {
    "iter": iter_n,
    "slug": slug,
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "composite": s.get("mean_composite"),
    "outcome": s.get("mean_outcome"),
    "efficiency": s.get("mean_efficiency"),
    "tool_choice": s.get("mean_tool_choice"),
    "grounding": s.get("mean_grounding"),
    "format_compliance": s.get("mean_format"),
    "mean_wall_clock_s": s.get("mean_wall_clock_s"),
    "mean_n_tool_calls": s.get("mean_n_tool_calls"),
    "mean_bytes_consumed": s.get("mean_bytes_consumed"),
    "mean_n_large_reads": s.get("mean_n_large_reads"),
    "n_rollouts": s.get("n_rollouts"),
    "rollouts_outcome_pass": s.get("rollouts_outcome_pass"),
    "rollouts_zero": s.get("rollouts_zero"),
    "manifest": m,
    "git_sha": git_sha,
}
path = "/workspace/kiln/capabilities/agentic-grpo/pi-code-search/capability.jsonl"
with open(path, "a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps(row, indent=2))
PY

# Backup adapter to B2 (always, even on null — cheap and recoverable).
B2_BUCKET="${B2_BUCKET:-clouderic}"
B2_PREFIX="${B2_PREFIX:-kiln/pi-code-search}"
ADAPTER_DIR="$BASE_DIR/adapter"
ADAPTER_NAME="${ADAPTER_NAME:-pi-code-search-iter${ITER}-${SLUG}}"

if [ -d "$ADAPTER_DIR" ]; then
  TARBALL="/workspace/${ADAPTER_NAME}.tgz"
  tar czf "$TARBALL" -C "$BASE_DIR" adapter
  b2 file upload "$B2_BUCKET" "$TARBALL" "$B2_PREFIX/${ADAPTER_NAME}.tgz" 2>&1 | tail -3 || \
    echo "WARN: b2 upload failed; tarball remains at $TARBALL"
fi

echo "[orchestrate] iter $ITER ($SLUG) done"
