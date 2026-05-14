#!/bin/bash
# Produce a final-summary report for the session. Prints markdown to
# stdout, intended to be appended to capability.md or pasted to the
# user at stop time. Read-only — does not modify any files.
#
# Usage: summary.sh [capability.jsonl] [capability.config.json]

set -euo pipefail
LOG="${1:-capability.jsonl}"
CFG="${2:-capability.config.json}"

if [ ! -s "$LOG" ]; then
  echo "## Summary"; echo; echo "_No iterations logged yet._"; exit 0
fi

DIR=$(jq -r '.direction // "higher"' "$CFG" 2>/dev/null || echo "higher")
SUITE=$(jq -r '.eval_suite // "<unset>"' "$CFG" 2>/dev/null || echo "<unset>")
FIELD=$(jq -r '.scorer_field // "accuracy"' "$CFG" 2>/dev/null || echo "accuracy")

# Aggregate stats over scored entries.
STATS=$(jq -rs --arg dir "$DIR" '
  map(select(.score != null))
  | (length) as $n
  | (map(select(.slug == "baseline" or .iter == 0)) | .[-1].score) as $base
  | (if $dir == "lower" then min_by(.score) else max_by(.score) end) as $best
  | "N=\($n)|BASE=\($base)|BEST=\($best.score)|BEST_SLUG=\($best.slug)|DELTA=\($best.score - $base)"
' "$LOG")
N=$(echo "$STATS" | awk -F\| '{print $1}' | cut -d= -f2)
BASE=$(echo "$STATS" | awk -F\| '{print $2}' | cut -d= -f2)
BEST=$(echo "$STATS" | awk -F\| '{print $3}' | cut -d= -f2)
BEST_SLUG=$(echo "$STATS" | awk -F\| '{print $4}' | cut -d= -f2)
BEST_DELTA=$(echo "$STATS" | awk -F\| '{print $5}' | cut -d= -f2)

echo "## Summary"
echo
echo "- **Suite**: \`$SUITE\` (scorer: $FIELD, direction: $DIR, $N iterations)"
echo "- **Baseline**: $BASE"
echo "- **Best**: $BEST (slug \`$BEST_SLUG\`, Δ from baseline: $BEST_DELTA)"
echo

echo "### Top 5 kept ablations"
echo
echo "| iter | slug | score | Δ | hypothesis |"
echo "|------|------|------:|---:|------------|"
jq -rs --arg dir "$DIR" '
  map(select(.status == "kept" and .score != null and .slug != "baseline"))
  | (if $dir == "lower" then sort_by(.score) else sort_by(-.score) end)
  | .[0:5][]
  | "| \(.iter) | `\(.slug)` | \(.score) | \(.delta) | \(.hypothesis|.[:80]) |"
' "$LOG"
echo

echo "### Discards with notable scores"
echo "_Discards that came close — useful for the next session._"
echo
echo "| iter | slug | score | Δ | hypothesis |"
echo "|------|------|------:|---:|------------|"
jq -rs --arg dir "$DIR" '
  map(select(.status == "discard" and .score != null))
  | (if $dir == "lower" then sort_by(.score) else sort_by(-.score) end)
  | .[0:3][]
  | "| \(.iter) | `\(.slug)` | \(.score) | \(.delta) | \(.hypothesis|.[:80]) |"
' "$LOG"
echo

echo "### Errors and breaches"
ERR_COUNT=$(jq -rs '[.[] | select(.status == "crash" or .status == "oracle_error" or .status == "firewall_breach")] | length' "$LOG")
if [ "$ERR_COUNT" = "0" ]; then
  echo "_None._"
else
  jq -rs '
    .[] | select(.status == "crash" or .status == "oracle_error" or .status == "firewall_breach")
    | "- iter \(.iter) `\(.slug)`: **\(.status)** — \(.notes|.[:80])"
  ' "$LOG"
fi
echo

echo "### Confidence at finalisation"
bash "$(dirname "${BASH_SOURCE[0]}")/confidence.sh" "$LOG" "$CFG" 2>/dev/null \
  | sed 's/^/    /'
echo
echo "_Generated $(date -u +%FT%TZ)._"
