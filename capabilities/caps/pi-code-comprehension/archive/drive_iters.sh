#!/bin/bash
# Drive the pi-code-comprehension GRPO loop for many iters.
#
# Usage:
#   bash drive_iters.sh --pod <pod_id> --start-iter N --max-iters K
#
# Reads $HERE/capability.jsonl to find the next iter (max(iter)+1 if no
# --start-iter). Runs run_iter.sh with a recipe selected from
# capability.md's "Hypotheses to try" section.

set -euo pipefail

MAX_ITERS=50
START_ITER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --pod) POD_ID="$2"; shift 2 ;;
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --start-iter) START_ITER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
if [ -z "${POD_ID:-}" ]; then echo "--pod required" >&2; exit 1; fi

export POD_ID
export RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

next_iter() {
  if [ -n "$START_ITER" ]; then echo "$START_ITER"; return; fi
  if [ ! -s "$HERE/capability.jsonl" ]; then echo "0"; return; fi
  python3 -c "
import json
n = -1
for line in open('$HERE/capability.jsonl'):
    try:
        e = json.loads(line)
        v = e.get('iter')
        if isinstance(v, int):
            n = max(n, v)
    except Exception:
        pass
print(n + 1)
"
}

run_iter_n() {
  local n=$1
  echo "==============================================="
  echo "  ITER $n"
  echo "==============================================="
  case "$n" in
    0)  bash "$HERE/run_iter.sh" --iter 0 --kind baseline --skip-train --eval-adapter base --num-gens 1 ;;
    1)  bash "$HERE/run_iter.sh" --iter 1 --kind train --num-train-tasks 20 --num-gens 4 --lr 1e-5 --filter-var 0.02 ;;
    2)  bash "$HERE/run_iter.sh" --iter 2 --kind train --num-train-tasks 30 --num-gens 4 --lr 1e-5 --filter-var 0.02 ;;
    3)  bash "$HERE/run_iter.sh" --iter 3 --kind train --num-train-tasks 20 --num-gens 4 --lr 1e-5 --filter-var 0.02 --echo-lambda 0.075 ;;
    4)  bash "$HERE/run_iter.sh" --iter 4 --kind train --num-train-tasks 20 --num-gens 4 --lr 5e-6 --filter-var 0.02 ;;
    5)  bash "$HERE/run_iter.sh" --iter 5 --kind train --num-train-tasks 20 --num-gens 4 --lr 1e-5 --filter-var 0.02 --rank 32 --alpha 64 ;;
    6)  bash "$HERE/run_iter.sh" --iter 6 --kind train --num-train-tasks 20 --num-gens 8 --lr 1e-5 --filter-var 0.02 ;;
    7)  bash "$HERE/run_iter.sh" --iter 7 --kind train --num-train-tasks 12 --num-gens 4 --lr 1e-5 --filter-var 0.05 ;;
    8)  bash "$HERE/run_iter.sh" --iter 8 --kind train --num-train-tasks 20 --num-gens 4 --lr 1e-5 --filter-var 0.02 --epochs 2 ;;
    9)  bash "$HERE/run_iter.sh" --iter 9 --kind train --num-train-tasks 20 --num-gens 4 --lr 2e-5 --filter-var 0.02 ;;
    10) bash "$HERE/run_iter.sh" --iter 10 --kind train --num-train-tasks 30 --num-gens 4 --lr 1e-5 --filter-var 0.01 ;;
    *)  bash "$HERE/run_iter.sh" --iter "$n" --kind train --num-train-tasks 20 --num-gens 4 --lr 1e-5 --filter-var 0.02 ;;
  esac

  # Append result + commit + push.
  cd "$HERE"
  python3 record_iter.py --iter "$n" --pod "$POD_ID" || true
  cd /data/projects/kiln-pi-code-comprehension/kiln
  git add capabilities/agentic-grpo/pi-code-comprehension/capability.jsonl \
          capabilities/agentic-grpo/pi-code-comprehension/kiln-polish.jsonl 2>/dev/null || true
  git commit -m "cap[agentic-grpo/pi-code-comprehension]: iter $n result" || true
  git push origin main || true
  cd "$HERE"
}

iter=$(next_iter)
end=$((iter + MAX_ITERS))
while [ "$iter" -lt "$end" ]; do
  run_iter_n "$iter" || echo "iter $iter failed; continuing"
  iter=$((iter + 1))
done

echo "==============================================="
echo "  DRIVE COMPLETE ($iter iters)"
echo "==============================================="
