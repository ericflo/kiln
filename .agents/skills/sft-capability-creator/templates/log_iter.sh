#!/bin/bash
# Append one structured JSON line to capability.jsonl.
#
# Usage:
#   log_iter.sh <slug> <status> <score> <n> <hypothesis_file> <dataset_file> <adapter_name> <final_loss> <elapsed_s>
#
# Pulls extra fields from capability.jsonl (prev best to compute delta) and
# from hypotheses/<slug>.md (claim line). Writes a single JSON object on its
# own line.

set -euo pipefail

SLUG="$1"; STATUS="$2"; SCORE="${3:-}"; N="${4:-}"
HYP_FILE="${5-}"; DS_FILE="${6-}"; ADAPTER="${7-}"; LOSS="${8-}"; ELAPSED="${9-}"

# Crash / oracle_error / firewall_breach entries have no score. Mark
# them with a JSON null score and zero delta — they exist only for the
# experimental record and the resume-time crash-recovery scan.
NO_SCORE_STATES="crash|oracle_error|firewall_breach"
if echo "$STATUS" | grep -qE "^($NO_SCORE_STATES)$" || [ -z "$SCORE" ]; then
  SCORE_JSON=null
  SCORE=0     # awk-safe placeholder; the JSON will say null.
else
  SCORE_JSON="$SCORE"
fi

LOG=capability.jsonl
CFG=capability.config.json

DIR=$(jq -r '.direction // "higher"' "$CFG" 2>/dev/null || echo "higher")
if [ -f "$LOG" ]; then
  PREV_BEST=$(jq -rs --arg dir "$DIR" '
    map(select(.score != null and (.status == "kept" or .slug == "baseline" or .iter == 0)))
    | if length == 0 then "null"
      elif $dir == "lower" then (min_by(.score).score|tostring)
      else (max_by(.score).score|tostring) end
  ' "$LOG")
  ITER=$(($(wc -l < "$LOG")))
else
  PREV_BEST=null
  ITER=0
fi

if [ "$PREV_BEST" = "null" ] || [ "$SCORE_JSON" = "null" ]; then
  DELTA=0
else
  DELTA=$(awk -v a="$SCORE" -v b="$PREV_BEST" -v d="$DIR" 'BEGIN {
    if (d == "lower") print b - a; else print a - b;
  }')
fi

CLAIM=""
if [ -n "$HYP_FILE" ] && [ -f "$HYP_FILE" ]; then
  CLAIM=$(awk '/^## Claim/{flag=1; next} /^## /{flag=0} flag' "$HYP_FILE" | sed '/^$/d' | head -1)
fi

DS_SIZE=0
if [ -n "$DS_FILE" ] && [ -f "$DS_FILE" ]; then
  DS_SIZE=$(wc -l < "$DS_FILE")
fi

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

jq -nc \
  --argjson iter "$ITER" \
  --arg slug "$SLUG" \
  --arg ts "$TS" \
  --arg status "$STATUS" \
  --argjson score "$SCORE_JSON" \
  --argjson n "${N:-0}" \
  --argjson delta "$DELTA" \
  --arg hypothesis "$CLAIM" \
  --arg ds_path "$DS_FILE" \
  --argjson ds_size "$DS_SIZE" \
  --arg adapter "$ADAPTER" \
  --argjson loss "${LOSS:-0}" \
  --argjson elapsed "${ELAPSED:-0}" \
  '{
    iter: $iter,
    slug: $slug,
    ts: $ts,
    status: $status,
    score: $score,
    n: $n,
    delta: $delta,
    hypothesis: $hypothesis,
    dataset: { path: $ds_path, size: $ds_size },
    training: { adapter: $adapter, final_loss: $loss, elapsed_s: $elapsed },
    asi: {},
    notes: ""
  }' >> "$LOG"

echo "logged iter=$ITER slug=$SLUG score=$SCORE delta=$DELTA status=$STATUS"
