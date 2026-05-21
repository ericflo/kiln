#!/usr/bin/env bash
# Run a batch of training iters with cached rollouts.
set -uo pipefail

POD_ID=""
SOURCE_ITER=""
START_ITER=""
END_ITER=""
RUN_TAG="${RUN_TAG:-20260519-pft-50loop}"

while [ $# -gt 0 ]; do
  case "$1" in
    --pod) POD_ID="$2"; shift 2 ;;
    --source-iter) SOURCE_ITER="$2"; shift 2 ;;
    --start) START_ITER="$2"; shift 2 ;;
    --end) END_ITER="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
if [ -z "$POD_ID" ] || [ -z "$START_ITER" ] || [ -z "$END_ITER" ] || [ -z "$SOURCE_ITER" ]; then
  echo "all of --pod --source-iter --start --end required" >&2; exit 1
fi
export POD_ID RUN_TAG
export RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

recipe_for_iter() {
  local n=$1
  case "$n" in
    13)  echo "--lr 5e-6 --filter-var 0.02 --seed 271828" ;;
    14)  echo "--lr 5e-6 --filter-var 0.02 --seed 1618033" ;;
    15)  echo "--lr 5e-6 --filter-var 0.02 --seed 31415" ;;
    16)  echo "--lr 3e-6 --filter-var 0.02" ;;
    17)  echo "--lr 1e-6 --filter-var 0.02" ;;
    18)  echo "--lr 5e-6 --filter-var 0.01" ;;
    19)  echo "--lr 5e-6 --filter-var 0.005" ;;
    20)  echo "--lr 5e-6 --filter-var 0.02 --rank 4 --alpha 8" ;;
    21)  echo "--lr 5e-6 --filter-var 0.02 --rank 8 --alpha 32" ;;
    22)  echo "--lr 5e-6 --filter-var 0.02 --echo-lambda 0.01" ;;
    23)  echo "--lr 5e-6 --filter-var 0.02 --echo-lambda 0.03" ;;
    24)  echo "--lr 5e-6 --filter-var 0.02 --grpo-mode phase1_cispo" ;;
    25)  echo "--lr 5e-6 --filter-var 0.02 --grpo-mode phase1_reinforce" ;;
    26)  echo "--lr 5e-6 --filter-var 0.02 --max-groups 1" ;;
    27)  echo "--lr 7.5e-6 --filter-var 0.02" ;;
    28)  echo "--lr 4e-6 --filter-var 0.02" ;;
    29)  echo "--lr 5e-6 --filter-var 0.02 --rank 8 --alpha 64" ;;
    30)  echo "--lr 5e-6 --filter-var 0.02 --rank 16 --alpha 16" ;;
    *)   echo "--lr 5e-6 --filter-var 0.02" ;;
  esac
}

cd "$REPO_ROOT"

for ((n=START_ITER; n<END_ITER; n++)); do
  recipe="$(recipe_for_iter $n)"
  echo ""
  echo "==========================================="
  echo " PFT ITER $n  recipe: $recipe"
  echo "==========================================="
  if ! bash "$HERE/run_iter.sh" --iter "$n" --kind train $recipe --rollout-source-iter "$SOURCE_ITER" --max-wall 180 2>&1 | tee "/tmp/pft-iter${n}.drive.log"; then
    echo "iter $n FAILED — moving on" >&2
    if grep -q "AttributeError" "/tmp/pft-iter${n}.drive.log" 2>/dev/null; then
      echo "POD HIBERNATED — stopping batch" >&2; exit 99
    fi
  fi

  python3 "$HERE/_append_iter_log.py" --iter "$n" --pod "$POD_ID" --recipe "$recipe (source=$SOURCE_ITER)" || true
  python3 "$HERE/_refresh_in_progress.py" || true
  python3 "$HERE/backup_to_b2.py" --iter "$n" --kind iter --pod "$POD_ID" --run-tag "$RUN_TAG" || true

  cd "$REPO_ROOT"
  git add capabilities/agentic-grpo/pi-failure-triage/capability.jsonl \
          capabilities/agentic-grpo/pi-failure-triage/IN_PROGRESS.md \
          capabilities/agentic-grpo/pi-failure-triage/hypotheses/ 2>/dev/null || true
  git commit -m "cap[pi-failure-triage/iter$n]: result row + IN_PROGRESS refresh" || true
  git pull --rebase origin main 2>&1 | tail -2 || true
  git push origin main 2>&1 | tail -2 || true
done

echo "==========================================="
echo " BATCH DONE iters $START_ITER..$((END_ITER-1))"
echo "==========================================="
