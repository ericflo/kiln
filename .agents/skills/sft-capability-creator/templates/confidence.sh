#!/bin/bash
# Compute confidence stats from capability.jsonl.
# Prints, on stdout:
#   N=<int> BASELINE=<float> BEST=<float> BEST_SLUG=<str> BEST_DELTA=<float> MAD=<float> CONFIDENCE=<float>
#
# CONFIDENCE = |BEST_DELTA| / MAD when MAD > 0, else "inf".
# Direction is read from capability.config.json (default "higher").

set -euo pipefail
LOG="${1:-capability.jsonl}"
CFG="${2:-capability.config.json}"

if [ ! -f "$LOG" ]; then
  echo "no log at $LOG" >&2
  exit 2
fi

DIR=$(jq -r '.direction // "higher"' "$CFG" 2>/dev/null || echo "higher")

# Pull all scores (non-null), the baseline (iter 0), and best by direction.
jq -rs --arg dir "$DIR" '
  map(select(.score != null))
  | if length == 0 then "N=0 BASELINE=null BEST=null BEST_SLUG=none BEST_DELTA=0 MAD=0 CONFIDENCE=na"
    else
      (length) as $n
      | ((map(select(.slug == "baseline" or .iter == 0)) | .[0].score) // .[0].score) as $base
      | (if $dir == "lower"
           then min_by(.score)
           else max_by(.score)
         end) as $best
      | ($best.score - $base) as $delta
      | (map(.score) | sort) as $sorted
      | (($sorted | length) as $len | $sorted[($len/2|floor)]) as $median
      | (map(.score - $median | fabs) | sort) as $devs
      | (($devs | length) as $dlen | $devs[($dlen/2|floor)]) as $mad
      | "N=\($n) BASELINE=\($base) BEST=\($best.score) BEST_SLUG=\($best.slug) BEST_DELTA=\($delta) MAD=\($mad) CONFIDENCE=\(if $mad > 0 then ($delta|fabs)/$mad else "inf" end)"
    end
' "$LOG"
