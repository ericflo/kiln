#!/bin/bash
# capability.anchor.sh — blind general-competence regression watch.
#
# Same contract as capability.oracle.sh, different (non-math) suite.
# Use this to detect stylistic clobber from math SFT.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SUITE_FILE="${SCRIPT_DIR}/.oracle-build/anchor_suite.json"
SERVER="${KILN_SERVER_URL:-http://localhost:8420}"

if [ ! -f "$SUITE_FILE" ]; then
  echo "ORACLE_ERROR: missing anchor suite file under .oracle-build/" >&2
  exit 2
fi

if ! curl -fsS -m 5 "$SERVER/health" >/dev/null 2>&1; then
  echo "ORACLE_ERROR: kiln server not reachable at $SERVER" >&2
  exit 3
fi

ADAPTER="${1-}"

BODY=$(python3 -c "
import json, sys
suite = json.load(open(sys.argv[1]))
body = {'inline_suite': suite, 'adapter': sys.argv[2]}
print(json.dumps(body))
" "$SUITE_FILE" "$ADAPTER")

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

DEADLINE=$(( $(date +%s) + 900 ))
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

echo "SCORE=$SCORE N=$N"
