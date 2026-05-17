#!/bin/bash
# capability.oracle.sh — blind math oracle.
#
# Submits an inline EvalSuite (whose contents are held privately under
# .oracle-build/) to the kiln server, polls until completion, and prints
# exactly one line to stdout:
#     SCORE=<float> N=<int>
#
# On failure: exit non-zero with `ORACLE_ERROR: <reason>` on stderr.
#
# Usage: ./capability.oracle.sh <adapter>
#   <adapter>: kiln adapter name. Empty string ("") means evaluate the
#              base model.
#
# Implementation notes (private):
#   The suite file `.oracle-build/math_suite.json` is read-only state;
#   we never expose its contents on stdout. The main agent's permitted
#   read set treats `.oracle-build/` as off-limits per the experiment's
#   firewall.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SUITE_FILE="${SCRIPT_DIR}/.oracle-build/math_suite.json"
SERVER="${KILN_SERVER_URL:-http://localhost:8420}"

if [ ! -f "$SUITE_FILE" ]; then
  echo "ORACLE_ERROR: missing oracle suite file under .oracle-build/" >&2
  exit 2
fi

# Validate server first; a broken server should fail fast and clearly.
if ! curl -fsS -m 5 "$SERVER/health" >/dev/null 2>&1; then
  echo "ORACLE_ERROR: kiln server not reachable at $SERVER" >&2
  exit 3
fi

ADAPTER="${1-}"

# Build the request body: inline suite + adapter selection.
# `python3` rather than `jq` because we need to embed an object inside an
# object — much safer than shell-string-formatting JSON.
BODY=$(python3 -c "
import json, sys
suite = json.load(open(sys.argv[1]))
body = {'inline_suite': suite, 'adapter': sys.argv[2]}
print(json.dumps(body))
" "$SUITE_FILE" "$ADAPTER")

# Submit. Capture response.
SUBMIT=$(curl -fsS -m 30 -X POST "$SERVER/v1/eval/run" \
  -H "Content-Type: application/json" \
  --data "$BODY" 2>/dev/null) || {
    echo "ORACLE_ERROR: failed to submit eval job" >&2
    exit 3
  }

JOB_ID=$(echo "$SUBMIT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('job_id',''))")
if [ -z "$JOB_ID" ]; then
  echo "ORACLE_ERROR: server did not return a job_id" >&2
  exit 3
fi

# Poll until terminal state. Cap at ~25 minutes wall (oracle is built
# to run well under this) so a stuck server can't hang the loop forever.
DEADLINE=$(( $(date +%s) + 1500 ))
STATE=""
RESULT=""
while [ $(date +%s) -lt $DEADLINE ]; do
  RESULT=$(curl -fsS -m 30 "$SERVER/v1/eval/jobs/$JOB_ID" 2>/dev/null) || {
    sleep 5
    continue
  }
  STATE=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('state',''))")
  case "$STATE" in
    completed|failed|cancelled) break ;;
  esac
  sleep 5
done

if [ "$STATE" != "completed" ]; then
  echo "ORACLE_ERROR: eval job terminal state=${STATE:-timeout}" >&2
  exit 3
fi

# Pull accuracy out of the result. We deliberately read ONLY the headline
# accuracy + num_examples — never per-example outcomes.
read SCORE N <<EOF
$(echo "$RESULT" | python3 -c "
import json, sys
d = json.load(sys.stdin)
runs = d.get('runs', [])
if not runs:
    print('NA NA'); sys.exit(0)
m = runs[0].get('metrics', {}) or {}
acc = m.get('accuracy')
n = m.get('num_examples')
if acc is None or n is None:
    print('NA NA')
else:
    print(f'{acc:.6f} {int(n)}')
")
EOF

if [ "$SCORE" = "NA" ] || [ -z "$SCORE" ]; then
  echo "ORACLE_ERROR: could not read accuracy from result" >&2
  exit 3
fi

# Final line — and only this line — goes to stdout.
echo "SCORE=$SCORE N=$N"
