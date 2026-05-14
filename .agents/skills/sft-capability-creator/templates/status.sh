#!/bin/bash
# Print a one-screen status of the current session. Designed to be
# called at the start of every iteration so the agent (or a resuming
# agent) has the full state in front of it.

set -euo pipefail
LOG="${1:-capability.jsonl}"
CFG="${2:-capability.config.json}"

if [ ! -f "$LOG" ]; then
  echo "no log at $LOG"
  exit 0
fi

DIR=$(jq -r '.direction // "higher"' "$CFG" 2>/dev/null || echo "higher")
SUITE=$(jq -r '.eval_suite // "<unset>"' "$CFG" 2>/dev/null || echo "<unset>")
SCORER=$(jq -r '.scorer_field // "accuracy"' "$CFG" 2>/dev/null || echo "accuracy")
NSESS=$(wc -l < "$LOG")

echo "session: suite=$SUITE scorer=$SCORER direction=$DIR"
echo "iterations logged: $NSESS"
echo

# Headline confidence stats.
bash "$(dirname "${BASH_SOURCE[0]}")/confidence.sh" "$LOG" "$CFG" 2>/dev/null || true
echo

# Compact ledger of recent iterations.
echo "recent ledger (last 10):"
printf "%4s  %-30s  %-15s  %7s  %8s  %s\n" iter slug status score delta hypothesis
tail -10 "$LOG" | jq -r '[
    (.iter|tostring),
    (.slug|.[:30]),
    .status,
    (.score|tostring),
    (.delta|tostring),
    (.hypothesis|.[:60])
  ] | @tsv' | awk -F"\t" '{printf "%4s  %-30s  %-15s  %7s  %8s  %s\n", $1,$2,$3,$4,$5,$6}'
echo

# Slugs the agent has used — useful to avoid accidental reuse.
echo "slugs in use:"
jq -r '.slug' "$LOG" | sort -u | head -20 | sed 's/^/  /'
