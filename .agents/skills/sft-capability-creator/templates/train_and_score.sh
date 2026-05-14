#!/bin/bash
# Train one adapter, then call the blind oracle. Print one summary line:
#   ADAPTER=<name> LOSS=<float> ELAPSED=<int> SCORE=<float> N=<int>
#
# Usage:
#   train_and_score.sh <slug> [--lr 1e-4] [--epochs 1] [--lora-rank 4]
#
# Reads dataset from datasets/<slug>.jsonl, names adapter cap-<slug>.
# Does NOT touch the log — the caller writes the structured entry.

set -euo pipefail

SLUG="$1"; shift
LR=1e-4
EPOCHS=1
RANK=4
SERVER="${KILN_SERVER_URL:-http://localhost:8420}"

while [ $# -gt 0 ]; do
  case "$1" in
    --lr) LR="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --lora-rank) RANK="$2"; shift 2;;
    --server) SERVER="$2"; shift 2;;
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
# with a non-empty messages array.
BAD=$(jq -c 'select((.messages|type) != "array" or (.messages|length) == 0)' "$DS" 2>/dev/null | head -3 || true)
if [ -n "$BAD" ]; then
  echo "dataset sanity check failed (sample of bad lines below):" >&2
  echo "$BAD" >&2
  exit 2
fi

START=$(date +%s)
TRAIN_OUT=$(kiln train sft \
  --file "$DS" \
  --adapter "$ADAPTER" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --lora-rank "$RANK" 2>&1)
END=$(date +%s)
ELAPSED=$((END - START))

# Try to recover the final loss from training output. Best-effort.
LOSS=$(echo "$TRAIN_OUT" | grep -oE 'final_loss[":=][^,}]+' | head -1 | grep -oE '[-0-9.]+' || true)
LOSS="${LOSS:-0}"

echo "$ADAPTER" > "adapters/$SLUG.txt"

# Now the blind oracle — and ONLY the blind oracle.
ORACLE_OUT=$(./capability.oracle.sh "$ADAPTER")
SCORE=$(echo "$ORACLE_OUT" | grep -oE 'SCORE=[-0-9.]+' | cut -d= -f2)
N=$(echo "$ORACLE_OUT" | grep -oE 'N=[0-9]+' | cut -d= -f2 || true)
N="${N:-0}"

echo "ADAPTER=$ADAPTER LOSS=$LOSS ELAPSED=$ELAPSED SCORE=$SCORE N=$N"
