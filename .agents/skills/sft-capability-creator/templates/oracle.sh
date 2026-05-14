#!/bin/bash
# capability.oracle.sh — blind eval wrapper.
#
# Calls a kiln registered eval suite and prints exactly one line:
#   SCORE=<float> N=<int>
#
# Everything else is suppressed. The skill agent is forbidden from
# reading the underlying JSON (suite contents, per-example responses,
# judgments). This wrapper is the firewall.
#
# Config in ./capability.config.json:
#   { "eval_suite": "<suite-name>",
#     "server": "http://localhost:8420",
#     "scorer_field": "accuracy" | "mean_score" }
#
# Usage: ./capability.oracle.sh <adapter_name>
#   Pass "" for the base model.

set -euo pipefail

CONFIG="${CONFIG_PATH:-capability.config.json}"
if [ ! -f "$CONFIG" ]; then
  echo "ORACLE_ERROR: missing $CONFIG" >&2
  exit 2
fi

SUITE=$(jq -r '.eval_suite // empty' "$CONFIG")
SERVER=$(jq -r '.server // "http://localhost:8420"' "$CONFIG")
FIELD=$(jq -r '.scorer_field // "accuracy"' "$CONFIG")

if [ -z "$SUITE" ] || [ "$SUITE" = "<REPLACE-WITH-SUITE-NAME-OR-PASTE>" ]; then
  echo "ORACLE_ERROR: eval_suite is not configured in $CONFIG" >&2
  exit 2
fi

ADAPTER="${1-}"

# Use the CLI which handles polling + JSON output across server versions.
# We capture stdout (the JSON) but never read fields outside summary.
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
if ! kiln-eval --server "$SERVER" run \
      --suite "$SUITE" --adapter "$ADAPTER" --watch --json > "$TMP" 2>/dev/null; then
  echo "ORACLE_ERROR: kiln-eval invocation failed" >&2
  exit 3
fi

STATE=$(jq -r '.state // empty' "$TMP")
if [ "$STATE" != "completed" ]; then
  REASON=$(jq -r '.error // .message // "unknown"' "$TMP")
  echo "ORACLE_ERROR: eval state=$STATE — $REASON" >&2
  exit 3
fi

SCORE=$(jq -r ".summary.$FIELD // empty" "$TMP")
N=$(jq -r '.summary.num_examples // empty' "$TMP")
if [ -z "$SCORE" ] || [ "$SCORE" = "null" ]; then
  echo "ORACLE_ERROR: missing summary.$FIELD in eval result" >&2
  exit 2
fi

# Print exactly one line. Nothing else.
if [ -n "$N" ] && [ "$N" != "null" ]; then
  echo "SCORE=$SCORE N=$N"
else
  echo "SCORE=$SCORE"
fi
