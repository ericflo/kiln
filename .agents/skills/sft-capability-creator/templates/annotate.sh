#!/bin/bash
# Annotate the most-recent capability.jsonl entry with an ASI field
# and/or notes. Designed to be called right after log_iter.sh.
#
# Usage:
#   annotate.sh --what_worked "..." --what_failed "..." \
#               --next_focus "..." --notes "..." [--key value ...]
#
# Any --key value pair becomes asi.key in the last entry.
# --notes value sets the top-level "notes" field instead.

set -euo pipefail
LOG="${LOG:-capability.jsonl}"
if [ ! -s "$LOG" ]; then
  echo "no entries in $LOG" >&2
  exit 2
fi

declare -A ASI
NOTES=""

while [ $# -gt 0 ]; do
  KEY="$1"; shift
  case "$KEY" in
    --notes)
      NOTES="$1"; shift ;;
    --*)
      NAME="${KEY#--}"
      ASI[$NAME]="${1:-}"; shift ;;
    *)
      echo "unknown arg: $KEY" >&2; exit 2 ;;
  esac
done

# Build a jq filter that merges into the last line. We construct the
# ASI delta as a stand-alone object, then merge it into the existing
# asi object on the line.
JQ_ARGS=(--arg notes "$NOTES")
ASI_DELTA="{}"
for k in "${!ASI[@]}"; do
  JQ_ARGS+=(--arg "v_$k" "${ASI[$k]}")
  ASI_DELTA="(${ASI_DELTA}) + {\"$k\": \$v_$k}"
done

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# Read all lines but the last unchanged. Rewrite the last with merged fields.
LAST=$(tail -1 "$LOG")
head -n -1 "$LOG" > "$TMP"
printf '%s\n' "$LAST" | jq -c "${JQ_ARGS[@]}" \
  ".asi = ((.asi // {}) + ($ASI_DELTA))
   | (if \$notes == \"\" then . else .notes = \$notes end)" >> "$TMP"

mv "$TMP" "$LOG"
echo "annotated entry at line $(wc -l < "$LOG") in $LOG"
