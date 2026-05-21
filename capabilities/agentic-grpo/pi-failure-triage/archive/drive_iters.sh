#!/usr/bin/env bash
# Drive 50 pi-failure-triage iters relentlessly.
#
# Strategy: cache rollouts and reuse them across many iters. Only
# regenerate rollouts on designated "data refresh" iters (and on iter 1
# which has no source to copy from). Each iter is a distinct training
# experiment; data refreshes use the current best adapter for the new
# rollouts.
#
# Per-iter logging: backup_to_b2, _append_iter_log, commit, push.
# Designed for autonomous overnight runs.

set -euo pipefail

START_ITER=0
END_ITER=50
POD_ID=""
RUN_TAG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --pod) POD_ID="$2"; shift 2 ;;
    --start) START_ITER="$2"; shift 2 ;;
    --end) END_ITER="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
if [ -z "$POD_ID" ]; then echo "--pod required" >&2; exit 1; fi
if [ -z "$RUN_TAG" ]; then RUN_TAG="$(date -u +%Y%m%d)-pft-50loop"; fi
export RUN_TAG
export POD_ID
export RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

# Source iter for each iter's rollouts:
#   iter 1: fresh (from base)
#   iters 2-12: reuse iter 1
#   iter 13: fresh (from best of iters 1-12)
#   iters 14-25: reuse iter 13
#   iter 26: fresh (from best of iters 1-25)
#   iters 27-37: reuse iter 26
#   iter 38: fresh
#   iters 39-49: reuse iter 38
rollout_source_for_iter() {
  local n=$1
  if [ "$n" -le 1 ]; then echo ""; return; fi
  if [ "$n" -le 12 ]; then echo "--rollout-source-iter 1"; return; fi
  if [ "$n" -le 25 ]; then echo "--rollout-source-iter 13"; return; fi
  if [ "$n" -le 37 ]; then echo "--rollout-source-iter 26"; return; fi
  echo "--rollout-source-iter 38"
}

# Per-iter recipe (training hyperparams + data filter).
recipe_for_iter() {
  local n=$1
  case "$n" in
    0)   echo "--kind baseline --skip-train --eval-adapter base" ;;
    1)   echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --filter-var 0.02 --max-wall 180" ;;
    2)   echo "--kind train --lr 5e-6 --filter-var 0.02" ;;
    3)   echo "--kind train --lr 2e-5 --filter-var 0.02" ;;
    4)   echo "--kind train --lr 1e-5 --filter-var 0.05" ;;
    5)   echo "--kind train --lr 1e-5 --filter-var 0.0  " ;;
    6)   echo "--kind train --lr 1e-5 --filter-var 0.02 " ;;
    7)   echo "--kind train --lr 1e-5 --filter-var 0.02 --rank 8  --alpha 16" ;;
    8)   echo "--kind train --lr 1e-5 --filter-var 0.02 --rank 32 --alpha 64" ;;
    9)   echo "--kind train --lr 1e-5 --filter-var 0.02 " ;;
    10)  echo "--kind train --lr 1e-5 --filter-var 0.02 " ;;
    11)  echo "--kind train --lr 1e-5 --filter-var 0.02 --echo-lambda 0.10" ;;
    12)  echo "--kind train --lr 1e-5 --filter-var 0.02 --echo-lambda 0.01" ;;
    13)  echo "--kind train --num-train-tasks 12 --num-gens 4 --lr 1e-5 --filter-var 0.02 --train-adapter pi-failure-triage-iter1 --max-wall 180" ;;
    14)  echo "--kind train --lr 5e-6 --filter-var 0.02" ;;
    15)  echo "--kind train --lr 2e-5 --filter-var 0.02" ;;
    16)  echo "--kind train --lr 1e-5 --filter-var 0.10" ;;
    17)  echo "--kind train --lr 1e-5 --filter-var 0.02 --epochs 2" ;;
    18)  echo "--kind train --lr 1e-5 --filter-var 0.02 --epochs 3" ;;
    19)  echo "--kind train --lr 1e-5 --filter-var 0.02 --no-echo" ;;
    20)  echo "--kind train --lr 1e-5 --filter-var 0.02 --echo-lambda 0.10" ;;
    21)  echo "--kind train --lr 1e-5 --filter-var 0.02 --seed 271828" ;;
    22)  echo "--kind train --lr 1e-5 --filter-var 0.02 --seed 1618033" ;;
    23)  echo "--kind train --lr 1e-5 --filter-var 0.02 --rank 16 --alpha 64" ;;
    24)  echo "--kind train --lr 1e-5 --filter-var 0.02 " ;;
    25)  echo "--kind train --lr 1e-5 --filter-var 0.02 " ;;
    26)  echo "--kind train --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.02 --train-adapter pi-failure-triage-iter13 --max-wall 180" ;;
    27)  echo "--kind train --lr 5e-6 --filter-var 0.02" ;;
    28)  echo "--kind train --lr 3e-6 --filter-var 0.01" ;;
    29)  echo "--kind train --lr 7.5e-6 --filter-var 0.02" ;;
    30)  echo "--kind train --lr 1.5e-5 --filter-var 0.02" ;;
    31)  echo "--kind train --lr 1e-5 --filter-var 0.02 " ;;
    32)  echo "--kind train --lr 1e-5 --filter-var 0.02 " ;;
    33)  echo "--kind train --lr 1e-5 --filter-var 0.02 --echo-lambda 0.03" ;;
    34)  echo "--kind train --lr 1e-5 --filter-var 0.02 --echo-lambda 0.07" ;;
    35)  echo "--kind train --lr 1e-5 --filter-var 0.02 --rank 64 --alpha 64" ;;
    36)  echo "--kind train --lr 1e-5 --filter-var 0.02 --rank 8 --alpha 32" ;;
    37)  echo "--kind train --lr 5e-6 --filter-var 0.02 --epochs 2" ;;
    38)  echo "--kind train --num-train-tasks 12 --num-gens 8 --lr 1e-5 --filter-var 0.02 --train-adapter pi-failure-triage-iter26 --max-wall 180" ;;
    39)  echo "--kind train --lr 1e-5 --filter-var 0.05" ;;
    40)  echo "--kind train --lr 5e-6 --filter-var 0.02" ;;
    41)  echo "--kind train --lr 1e-5 --filter-var 0.10" ;;
    42)  echo "--kind train --lr 1e-5 --filter-var 0.02 --rank 32 --alpha 32" ;;
    43)  echo "--kind train --lr 1e-5 --filter-var 0.02 " ;;
    44)  echo "--kind train --lr 1e-5 --filter-var 0.02 " ;;
    45)  echo "--kind train --lr 1e-5 --filter-var 0.02 --grpo-mode phase1_reinforce" ;;
    46)  echo "--kind train --lr 1e-5 --filter-var 0.02 --no-echo --no-policy" ;;
    47)  echo "--kind train --lr 1e-5 --filter-var 0.02 --echo-lambda 0.15" ;;
    48)  echo "--kind train --lr 5e-6 --filter-var 0.02 --epochs 3" ;;
    49)  echo "--kind train --lr 1e-5 --filter-var 0.02 --seed 31415" ;;
    *)   echo "--kind train --lr 1e-5 --filter-var 0.02" ;;
  esac
}

cd "$REPO_ROOT"

for ((n=START_ITER; n<END_ITER; n++)); do
  source=$(rollout_source_for_iter $n)
  recipe="$(recipe_for_iter $n)"
  echo ""
  echo "==========================================="
  echo " PFT ITER $n"
  echo "  rollout: $source"
  echo "  recipe : $recipe"
  echo "==========================================="

  if ! bash "$HERE/run_iter.sh" --iter "$n" $recipe $source 2>&1 | tee "/tmp/pft-iter${n}.drive.log"; then
    echo "iter $n FAILED — moving on" >&2
  fi

  # Pull summaries and append a row to capability.jsonl
  python3 "$HERE/_append_iter_log.py" --iter "$n" --pod "$POD_ID" --recipe "$recipe $source" || true

  # Backup adapter + summaries to B2
  python3 "$HERE/backup_to_b2.py" --iter "$n" --kind iter --pod "$POD_ID" || true

  # Commit + push
  cd "$REPO_ROOT"
  git add capabilities/agentic-grpo/pi-failure-triage/capability.jsonl \
          capabilities/agentic-grpo/pi-failure-triage/hypotheses/ 2>/dev/null || true
  git commit -m "cap[pi-failure-triage/iter$n]: result row" || true
  git pull --rebase origin main 2>&1 | tail -2 || true
  git push origin main 2>&1 | tail -2 || true
done

echo "==========================================="
echo " DRIVE COMPLETE (iters ${START_ITER}..${END_ITER})"
echo "==========================================="
