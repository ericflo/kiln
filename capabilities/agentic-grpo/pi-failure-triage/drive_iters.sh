#!/usr/bin/env bash
# Drive 50 pi-failure-triage iters relentlessly. Picks per-iter recipes that
# vary hyperparameters and data filters. After each iter: backup_to_b2 ->
# append to capability.jsonl -> commit + push.
#
# Designed for autonomous "all-night" runs. Each iter writes its own row.
# Stop early by deleting the lease's environment file at /tmp/grpo-pod.env.
#
# Usage:
#   bash drive_iters.sh --pod <pod_id> --start 0 --end 50

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

# Helper: emit RUN command for one iter. Echoes the eval-summary path so the
# caller can log it.
run_iter() {
  local n="$1"; shift
  echo "==========================================="
  echo " PFT ITER $n"
  echo "==========================================="
  bash "$HERE/run_iter.sh" --iter "$n" "$@" || return $?
}

# Hypothesis dispatch table. Family choices borrow from agentic-grpo skill §4.
# We deliberately use a wide variety so the 50-loop covers axes.
recipe_for_iter() {
  local n=$1
  case "$n" in
    0)   echo "--kind baseline --skip-train --eval-adapter base" ;;
    1)   echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --filter-var 0.02" ;;
    2)   echo "--kind train --num-train-tasks 12 --num-gens 4 --lr 1e-5 --filter-var 0.02" ;;
    3)   echo "--kind train --num-train-tasks 8 --num-gens 8 --lr 1e-5 --filter-var 0.02" ;;
    4)   echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 5e-6 --filter-var 0.02" ;;
    5)   echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 2e-5 --filter-var 0.02" ;;
    6)   echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --filter-var 0.05" ;;
    7)   echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --filter-var 0.10" ;;
    8)   echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --rank 8 --alpha 16" ;;
    9)   echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --rank 32 --alpha 64" ;;
    10)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --kl-coeff 0.05" ;;
    11)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --kl-coeff 0.20" ;;
    12)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --clip-eps 0.10" ;;
    13)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --clip-eps 0.30" ;;
    14)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --echo-lambda 0.01" ;;
    15)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --echo-lambda 0.03" ;;
    16)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --echo-lambda 0.10" ;;
    17)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --no-echo" ;;
    18)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --seed 271828" ;;
    19)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --seed 1618033" ;;
    20)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --seed 1414213" ;;
    21)  echo "--kind train --num-train-tasks 16 --num-gens 4 --lr 1e-5 --filter-var 0.02" ;;
    22)  echo "--kind train --num-train-tasks 20 --num-gens 4 --lr 1e-5 --filter-var 0.02" ;;
    23)  echo "--kind train --num-train-tasks 24 --num-gens 4 --lr 1e-5 --filter-var 0.05" ;;
    24)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --epochs 2" ;;
    25)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --epochs 3" ;;
    26)  echo "--kind train --num-train-tasks 8 --num-gens 12 --lr 1e-5 --filter-var 0.01" ;;
    27)  echo "--kind train --num-train-tasks 8 --num-gens 16 --lr 1e-5 --filter-var 0.01" ;;
    28)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 5e-6 --filter-var 0.01" ;;
    29)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --max-wall 240" ;;
    30)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --max-wall 300" ;;
    31)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --rank 64 --alpha 64" ;;
    32)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --rank 16 --alpha 64" ;;
    33)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --kl-coeff 0.0" ;;
    34)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --adv-mode std_grpo" ;;
    35)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --no-policy --echo-lambda 0.10" ;;
    36)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --no-policy --echo-lambda 0.20" ;;
    37)  echo "--kind train --num-train-tasks 12 --num-gens 8 --lr 1e-5 --filter-var 0.02" ;;
    38)  echo "--kind train --num-train-tasks 12 --num-gens 8 --lr 5e-6 --filter-var 0.05" ;;
    39)  echo "--kind train --num-train-tasks 12 --num-gens 4 --lr 1e-5 --filter-var 0.0 --echo-lambda 0.05" ;;
    40)  echo "--kind train --num-train-tasks 16 --num-gens 4 --lr 1e-5 --epochs 2" ;;
    41)  echo "--kind train --num-train-tasks 16 --num-gens 4 --lr 5e-6 --epochs 2" ;;
    42)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 3e-6 --filter-var 0.02" ;;
    43)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1.5e-5 --filter-var 0.02" ;;
    44)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --train-adapter pi-failure-triage-iter1" ;;
    45)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --train-adapter pi-failure-triage-iter2" ;;
    46)  echo "--kind train --num-train-tasks 16 --num-gens 4 --lr 5e-6 --filter-var 0.01" ;;
    47)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --filter-var 0.02 --clip-eps 0.15" ;;
    48)  echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --filter-var 0.02 --kl-coeff 0.15" ;;
    49)  echo "--kind train --num-train-tasks 8 --num-gens 8 --lr 1e-5 --filter-var 0.05 --echo-lambda 0.03" ;;
    *)   echo "--kind train --num-train-tasks 8 --num-gens 4 --lr 1e-5 --filter-var 0.02" ;;
  esac
}

cd "$REPO_ROOT"

for ((n=START_ITER; n<END_ITER; n++)); do
  recipe="$(recipe_for_iter $n)"
  echo ""
  echo "==========================================="
  echo " PFT ITER $n  recipe: $recipe"
  echo "==========================================="

  if ! bash "$HERE/run_iter.sh" --iter "$n" $recipe 2>&1 | tee "/tmp/pft-iter${n}.drive.log"; then
    echo "iter $n FAILED — moving on" >&2
  fi

  # Pull summaries and append a row to capability.jsonl
  python3 "$HERE/_append_iter_log.py" --iter "$n" --pod "$POD_ID" --recipe "$recipe" || true

  # Backup to B2
  python3 "$HERE/backup_to_b2.py" --iter "$n" --kind iter --pod "$POD_ID" || true

  # Commit + push
  cd "$REPO_ROOT"
  git add capabilities/agentic-grpo/pi-failure-triage/capability.jsonl \
          capabilities/agentic-grpo/pi-failure-triage/hypotheses/ 2>/dev/null || true
  git commit -m "cap[pi-failure-triage/iter$n]: result row" || true
  git push origin main || true
done

echo "==========================================="
echo " DRIVE COMPLETE ($((END_ITER - START_ITER)) iters)"
echo "==========================================="
