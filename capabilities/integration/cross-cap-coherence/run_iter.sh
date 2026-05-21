#!/usr/bin/env bash
# run_iter.sh — integration eval wrapper.
#
# This cap does NOT train. `run_iter.sh` exists for symmetry with the
# per-cap layout and just delegates to capability.oracle.sh against
# the named adapter(s).
#
# Usage:
#   ./run_iter.sh adapter-name [adapter2] ...
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -eq 0 ]; then
  echo "USAGE: $0 <adapter-name> [adapter-name-2] ..." >&2
  exit 1
fi

OUT_FILE="${OUT_FILE:-/tmp/cross-cap-coherence-iter-$(date +%s).json}" \
  ./capability.oracle.sh "$@"
