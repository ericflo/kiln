#!/bin/bash
# Drive the pi-compaction GRPO iter loop relentlessly.
#
# Picks the next-iter recipe based on what's already in capability.jsonl,
# runs it via run_iter.sh, backs up, then loops.
#
# Usage:
#   bash drive_iters.sh --pod tmbzxps5dx0uq2 --max-iters 20
#
# The recipe sequence (defaults — override individual iters by setting
# entries in capability.jsonl before launching):
#   iter 0  baseline (no adapter)
#   iter 1  H1 default GRPO, 20 train tasks × 4 gens
#   iter 2  H12 strong-signal filter (var > 0.05) on iter 1's pool
#   iter 3  H11 num_generations doubled (4 → 8) for sharper advantages
#   iter 4  H1 lower lr (5e-6) on iter 2's filtered pool
#   iter 5+ Multi-seed verification + ablation
#
# After each iter: backup_to_b2 -> commit capability.jsonl -> push.

set -euo pipefail

MAX_ITERS=20
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
export LEASE_ID
export RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

next_iter() {
  if [ -n "$START_ITER" ]; then
    echo "$START_ITER"
    return
  fi
  if [ ! -s "$HERE/capability.jsonl" ]; then
    echo "0"; return
  fi
  python3 -c "
import json
n = 0
for line in open('$HERE/capability.jsonl'):
    try:
        e = json.loads(line)
        v = e.get('iter')
        if isinstance(v, int):
            n = max(n, v + 1)
    except Exception:
        pass
print(n)
"
}

run_iter_n() {
  local n=$1
  echo "==============================================="
  echo "  ITER $n"
  echo "==============================================="
  case "$n" in
    0)  bash "$HERE/run_iter.sh" --iter 0 --kind baseline --skip-train --eval-adapter base ;;
    1)  bash "$HERE/run_iter.sh" --iter 1 --kind train --num-train-tasks 20 --num-gens 4 --lr 1e-5 --filter-var 0.05 ;;
    2)  bash "$HERE/run_iter.sh" --iter 2 --kind train --num-train-tasks 30 --num-gens 4 --lr 1e-5 --filter-var 0.05 ;;
    3)  bash "$HERE/run_iter.sh" --iter 3 --kind train --num-train-tasks 20 --num-gens 8 --lr 1e-5 --filter-var 0.05 ;;
    4)  bash "$HERE/run_iter.sh" --iter 4 --kind train --num-train-tasks 30 --num-gens 4 --lr 5e-6 --filter-var 0.05 ;;
    5)  bash "$HERE/run_iter.sh" --iter 5 --kind train --num-train-tasks 20 --num-gens 4 --lr 1e-5 --filter-var 0.02 ;;
    *)  bash "$HERE/run_iter.sh" --iter "$n" --kind train --num-train-tasks 20 --num-gens 4 --lr 1e-5 --filter-var 0.05 ;;
  esac

  # Commit + push capability.jsonl
  cd /tmp/kiln-eval-work
  git add capabilities/agentic-grpo/pi-compaction/capability.jsonl || true
  git commit -m "iter $n result" || true
  git push origin grpo/agentic-pi-compaction || true
  cd - >/dev/null
}

iter=$(next_iter)
end=$((iter + MAX_ITERS))
while [ "$iter" -lt "$end" ]; do
  run_iter_n "$iter"
  iter=$((iter + 1))
done

echo "==============================================="
echo "  DRIVE COMPLETE ($iter iters)"
echo "==============================================="
