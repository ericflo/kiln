#!/bin/bash
# capability.anchor.sh — blind regression-watch oracle for general competence.
#
# Tests something OTHER than Python coding (basic knowledge, instruction-
# following, reading comprehension, arithmetic). Used to detect adapters
# that catastrophically damage general model behaviour.
#
# Usage: ./capability.anchor.sh <adapter_name>
#   Empty string "" means use the base model.
#
# On success: exits 0, prints SCORE=<float> N=<int> on the last line.
# On failure: exits non-zero, prints ORACLE_ERROR: <reason> on stderr.

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/.oracle-build"
RUNNER="$BUILD/sandbox/anchor_runner.py"

if [ ! -f "$RUNNER" ]; then
  echo "ORACLE_ERROR: anchor runner missing at $RUNNER" >&2
  exit 2
fi

ADAPTER="${1-}"

mkdir -p "$BUILD/logs"
LOG="$BUILD/logs/anchor_$(date +%s)_$$.log"

if ! python3 "$RUNNER" "$ADAPTER" 2>"$LOG"; then
  grep -E '^(ORACLE_ERROR|RuntimeError):' "$LOG" >&2 || true
  if ! grep -q '^ORACLE_ERROR' "$LOG"; then
    echo "ORACLE_ERROR: anchor runner failed (see anchor log)" >&2
  fi
  exit 3
fi
