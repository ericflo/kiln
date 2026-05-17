#!/bin/bash
# capability.oracle.sh — blind primary oracle for python algorithmic problem-solving.
#
# Usage: ./capability.oracle.sh <adapter_name>
#   Empty string "" means use the base model.
#
# On success: exits 0, prints SCORE=<float> N=<int> on the last line.
# On failure: exits non-zero, prints ORACLE_ERROR: <reason> on stderr.
#
# The main agent must not look inside .oracle-build/. That dir contains the
# eval items, scorer, sandbox, and reference solutions — all of which would
# leak the eval and invalidate the experiment.

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/.oracle-build"
RUNNER="$BUILD/sandbox/eval_runner.py"

if [ ! -f "$RUNNER" ]; then
  echo "ORACLE_ERROR: eval runner missing at $RUNNER" >&2
  exit 2
fi

ADAPTER="${1-}"

mkdir -p "$BUILD/logs"
LOG="$BUILD/logs/oracle_$(date +%s)_$$.log"

# stderr -> log file (full diagnostics); stdout -> oracle stdout.
# On success we print only the final SCORE line; on failure we surface the
# ORACLE_ERROR / RuntimeError line on stderr.
if ! python3 "$RUNNER" "$ADAPTER" 2>"$LOG"; then
  grep -E '^(ORACLE_ERROR|RuntimeError):' "$LOG" >&2 || true
  if ! grep -q '^ORACLE_ERROR' "$LOG"; then
    echo "ORACLE_ERROR: eval runner failed (see oracle log)" >&2
  fi
  exit 3
fi
