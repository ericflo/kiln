#!/usr/bin/env bash
# runpod-validate-substrate-orchestrator.sh — outside-the-pod runner.
#
# Acquires a pod from the Cloud Eric kiln pool, refreshes the repo to
# the requested ref, executes scripts/runpod-substrate-validate.sh in
# the background using the wait-file pattern from the kiln skill
# (`kiln-ssh-polling-deadlock` mandate), and releases the lease.
#
# Usage:
#   bash scripts/runpod-validate-substrate-orchestrator.sh \
#       --ref main \
#       [--gpu-smoke] \
#       [--timeout 1800]
#
# Where:
#   --ref REF       git ref to validate (default: main)
#   --gpu-smoke     also build the --features cuda kiln-bench binary
#   --timeout N     wait-file timeout in seconds (default: 1800)
#
# Returns 0 if validate-substrate exits 0; non-zero on every other
# path, surfacing the underlying error.
#
# Designed to be agent-callable from outside the pod — no SSH-polling
# loops, no `until ssh ... kill -0` patterns ($99.76 incident,
# 2026-04-20).

set -euo pipefail

REF="main"
GPU_SMOKE=0
TIMEOUT=1800
while [ $# -gt 0 ]; do
  case "$1" in
    --ref)        REF="$2"; shift 2 ;;
    --gpu-smoke)  GPU_SMOKE=1; shift ;;
    --timeout)    TIMEOUT="$2"; shift 2 ;;
    -h|--help)    sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Find the RunPod CLI helper distributed with the kiln skill.
RP="${RP:-${CLOUDERIC_SKILL_DIR:-/data/knowledge/skills}/kiln/scripts/runpod_api.py}"
if [ ! -f "$RP" ]; then
  echo "runpod_api.py not found at $RP" >&2
  echo "Set RP=/path/to/runpod_api.py to override." >&2
  exit 3
fi

echo "[1/5] Acquiring kiln pod from pool..."
LEASE_JSON=$(ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000' 2>&1)
if [ -z "$LEASE_JSON" ]; then
  echo "ce kiln-pod-acquire returned empty output" >&2
  exit 4
fi
LEASE_ID=$(echo "$LEASE_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['lease_id'])")
POD_ID=$(echo "$LEASE_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['pod_id'])")
echo "  lease=$LEASE_ID pod=$POD_ID"

# Failure cleanup: release the lease as failed; success path releases below.
trap 'ce kiln-pod-release --lease "$LEASE_ID" --failure-reason "orchestrator-failed" 2>/dev/null || true' ERR INT TERM

echo "[2/5] Refreshing repo to ref=$REF on pod..."
python3 "$RP" exec "$POD_ID" \
  "cd /workspace/kiln && git fetch origin && git reset --hard origin/$REF" \
  > /tmp/pod-fetch.log 2>&1 \
  || { cat /tmp/pod-fetch.log >&2; exit 5; }
tail -1 /tmp/pod-fetch.log

DONE_MARKER="/workspace/kiln/substrate-validate.done"
LOG_PATH="/tmp/substrate-validate-orchestrated.log"
echo "[3/5] Launching substrate validate in background..."
EXTRA=""
[ $GPU_SMOKE -eq 1 ] && EXTRA="--gpu-smoke"
python3 "$RP" bg "$POD_ID" "$LOG_PATH" \
  "source /root/.kiln-build-env 2>/dev/null || true; \
   cd /workspace/kiln && \
   (bash scripts/runpod-substrate-validate.sh $EXTRA && touch '$DONE_MARKER') \
     2>&1 | tee '$LOG_PATH'" \
  > /tmp/orchestrator-bg.log
tail -1 /tmp/orchestrator-bg.log

echo "[4/5] Waiting for completion (timeout=${TIMEOUT}s)..."
python3 "$RP" wait-file "$POD_ID" "$DONE_MARKER" --timeout "$TIMEOUT"

echo "[5/5] Tailing log + releasing lease..."
python3 "$RP" exec "$POD_ID" "tail -30 '$LOG_PATH'" 2>&1 | tail -30
ce kiln-pod-release --lease "$LEASE_ID"
trap - ERR INT TERM
echo "OK — substrate validation completed on pod=$POD_ID."
