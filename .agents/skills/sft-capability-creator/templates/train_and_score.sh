#!/bin/bash
# Train one adapter (async kiln SFT job, polled to completion), then
# call the blind oracle. Print one summary line:
#   ADAPTER=<name> LOSS=<float> ELAPSED=<int> SCORE=<float> N=<int>
#
# Usage:
#   train_and_score.sh <slug> [--lr 1e-4] [--epochs 1] [--lora-rank 4]
#                             [--server URL] [--no-score]
#
# Reads dataset from datasets/<slug>.jsonl, names adapter cap-<slug>.
# Does NOT touch the log — the caller writes the structured entry.

set -euo pipefail

SLUG="$1"; shift
LR=1e-4
EPOCHS=1
RANK=4
SERVER="${KILN_SERVER_URL:-http://localhost:8420}"
DO_SCORE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --lr) LR="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --lora-rank) RANK="$2"; shift 2;;
    --server) SERVER="$2"; shift 2;;
    --no-score) DO_SCORE=0; shift;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done

DS="datasets/$SLUG.jsonl"
ADAPTER="cap-$SLUG"

if [ ! -f "$DS" ]; then
  echo "no dataset at $DS" >&2
  exit 2
fi

# Sanity-check the dataset shape — every line must parse as JSON
# with a non-empty messages array. Refuse to train on a malformed file.
BAD=$(jq -c 'select((.messages|type) != "array" or (.messages|length) == 0)' "$DS" 2>/dev/null | head -3 || true)
if [ -n "$BAD" ]; then
  echo "dataset sanity check failed (sample of bad lines below):" >&2
  echo "$BAD" >&2
  exit 2
fi

# Bonus check: no empty user/assistant content.
EMPTY=$(jq -c 'select(.messages | any(.content == "" or .content == null))' "$DS" 2>/dev/null | head -1 || true)
if [ -n "$EMPTY" ]; then
  echo "dataset sanity warning: at least one example has empty content" >&2
  # Not fatal — sometimes legitimate, but worth flagging.
fi

START=$(date +%s)

# Submit the SFT job. kiln train sft returns the job id on stdout.
SUBMIT_OUT=$(kiln train sft \
  --url "$SERVER" \
  --file "$DS" \
  --adapter "$ADAPTER" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --lora-rank "$RANK" 2>&1)

JOB_ID=$(echo "$SUBMIT_OUT" | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)
if [ -z "$JOB_ID" ]; then
  echo "TRAIN_ERROR: could not parse job_id from kiln train sft output" >&2
  echo "$SUBMIT_OUT" >&2
  exit 3
fi

# Poll until completion.
while :; do
  STATUS_JSON=$(curl -fsS "$SERVER/v1/train/status/$JOB_ID" 2>/dev/null || true)
  STATE=$(echo "$STATUS_JSON" | jq -r '.state // empty')
  case "$STATE" in
    completed) break;;
    failed)
      ERR=$(echo "$STATUS_JSON" | jq -r '.error // "unknown"')
      echo "TRAIN_ERROR: job $JOB_ID failed: $ERR" >&2
      exit 3 ;;
    "")
      echo "TRAIN_ERROR: cannot poll job $JOB_ID" >&2
      exit 3 ;;
    *) sleep 3 ;;
  esac
done

END=$(date +%s)
ELAPSED=$((END - START))

LOSS=$(curl -fsS "$SERVER/v1/train/status/$JOB_ID" 2>/dev/null | jq -r '.current_loss // empty')
LOSS="${LOSS:-0}"

mkdir -p adapters
echo "$ADAPTER" > "adapters/$SLUG.txt"

if [ "$DO_SCORE" -eq 0 ]; then
  echo "ADAPTER=$ADAPTER LOSS=$LOSS ELAPSED=$ELAPSED"
  exit 0
fi

# Now the blind oracle — and ONLY the blind oracle.
if [ ! -x ./capability.oracle.sh ]; then
  echo "ORACLE_MISSING: ./capability.oracle.sh is not executable" >&2
  exit 4
fi

ORACLE_OUT=$(./capability.oracle.sh "$ADAPTER")
SCORE=$(echo "$ORACLE_OUT" | grep -oE 'SCORE=[-0-9.]+' | cut -d= -f2)
N=$(echo "$ORACLE_OUT" | grep -oE 'N=[0-9]+' | cut -d= -f2 || true)
N="${N:-0}"

echo "ADAPTER=$ADAPTER LOSS=$LOSS ELAPSED=$ELAPSED SCORE=$SCORE N=$N"
