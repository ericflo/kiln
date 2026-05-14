#!/bin/bash
# capability.oracle.sh — human-in-the-loop variant.
#
# The user runs the eval themselves on whatever adapter the skill
# requests, then types one number back to stdin (or it can be
# supplied as an env var).
#
# Usage:  ./capability.oracle.sh <adapter_name>
# Stdin:  one number per line; first non-empty line is the score.
# Or:     SCORE=0.42 N=25 ./capability.oracle.sh <adapter_name>

set -euo pipefail
ADAPTER="${1-}"
TARGET="${ADAPTER:-<base>}"

# Allow the user to short-circuit by setting SCORE in the environment.
if [ -n "${SCORE-}" ]; then
  echo "SCORE=$SCORE${N:+ N=$N}"
  exit 0
fi

# Otherwise prompt on stderr (skill won't see this), read from stdin.
{
  echo
  echo "==> Blind eval requested for adapter: $TARGET"
  echo "==> Run your evaluator and type the score below (a float)."
  echo "==> Optional: 'SCORE=<float> N=<int>' on one line."
  echo
} >&2

while IFS= read -r LINE; do
  LINE=$(echo "$LINE" | tr -d '[:space:]')
  [ -z "$LINE" ] && continue
  if echo "$LINE" | grep -qE '^SCORE='; then
    echo "$LINE"
    exit 0
  fi
  if echo "$LINE" | grep -qE '^-?[0-9]+(\.[0-9]+)?$'; then
    echo "SCORE=$LINE"
    exit 0
  fi
  echo "(unrecognised; type a float or SCORE=<float>)" >&2
done
echo "ORACLE_ERROR: no score provided on stdin" >&2
exit 2
