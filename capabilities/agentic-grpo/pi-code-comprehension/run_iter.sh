#!/bin/bash
# Run one complete iter of pi-code-comprehension GRPO on a runpod pod.
#
# Stages:
#   1. (optional) training rollouts from a chosen adapter (or base)
#   2. (optional) filter strong-signal groups (var > FILTER_VAR)
#   3. (optional) GRPO step -> save adapter
#   4. eval the resulting adapter (or specified --eval-adapter)
#   5. backup artifacts to B2 (locally, on Cloud Eric, since B2 creds live there)
#
# Usage:
#   bash run_iter.sh --iter N --kind train|baseline|abl \
#                    --num-train-tasks 20 --num-gens 4 \
#                    --train-adapter "" --eval-adapter "pi-cc-iterN" \
#                    --lr 1e-5 --filter-var 0.02 --max-wall 180 \
#                    --rank 16 --alpha 32 --epochs 1 --seed 3141592653 \
#                    [--skip-train] [--skip-eval] [--echo-lambda 0.05] \
#                    [--no-echo] [--no-policy-loss]
#
# Requires `/tmp/grpo-pod.env` to be sourced with POD_ID, LEASE_ID, RP.
set -euo pipefail

ITER=""
KIND="train"
NUM_TRAIN_TASKS=20
NUM_GENS=4
TRAIN_ADAPTER=""
EVAL_ADAPTER=""
LR="1e-5"
FILTER_VAR="0.02"
SKIP_TRAIN=0
SKIP_EVAL=0
SEED=3141592653
EPOCHS=1
RANK=16
ALPHA=32
ECHO_LAMBDA=""
NO_ECHO=0
NO_POLICY_LOSS=0
MAX_WALL=180
EVAL_TASKS=24

while [ $# -gt 0 ]; do
  case "$1" in
    --iter) ITER="$2"; shift 2 ;;
    --kind) KIND="$2"; shift 2 ;;
    --num-train-tasks) NUM_TRAIN_TASKS="$2"; shift 2 ;;
    --num-gens) NUM_GENS="$2"; shift 2 ;;
    --train-adapter) TRAIN_ADAPTER="$2"; shift 2 ;;
    --eval-adapter) EVAL_ADAPTER="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --filter-var) FILTER_VAR="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --rank) RANK="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --echo-lambda) ECHO_LAMBDA="$2"; shift 2 ;;
    --no-echo) NO_ECHO=1; shift ;;
    --no-policy-loss) NO_POLICY_LOSS=1; shift ;;
    --max-wall) MAX_WALL="$2"; shift 2 ;;
    --eval-tasks) EVAL_TASKS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$ITER" ]; then echo "--iter required" >&2; exit 1; fi
EVAL_ADAPTER="${EVAL_ADAPTER:-pi-cc-iter${ITER}}"

source /tmp/grpo-pod.env

POD_REPO=/workspace/kiln/capabilities/agentic-grpo/pi-code-comprehension
TRAIN_OUT="/tmp/iter${ITER}-rollouts"
EVAL_OUT="/tmp/iter${ITER}-eval"
ADAPTER_OUT="/tmp/iter${ITER}-adapter"
TRAIN_LOG="/tmp/iter${ITER}-train.log"

echo "== iter ${ITER} kind=${KIND} =="

ECHO_FLAGS=()
if [ "$NO_ECHO" = "1" ]; then ECHO_FLAGS+=("--no-echo"); fi
if [ -n "$ECHO_LAMBDA" ]; then ECHO_FLAGS+=("--echo-lambda" "$ECHO_LAMBDA"); fi
if [ "$NO_POLICY_LOSS" = "1" ]; then ECHO_FLAGS+=("--no-policy-loss"); fi

############################################################################
# 1+2+3 — training
############################################################################
if [ "$SKIP_TRAIN" = "0" ]; then
  echo ">>> set train adapter -> '${TRAIN_ADAPTER:-(base)}'"
  if [ -z "$TRAIN_ADAPTER" ] || [ "$TRAIN_ADAPTER" = "base" ]; then
    python3 $RP ssh $POD_ID 'curl -sS -X POST http://localhost:8420/v1/adapters/unload >/dev/null || true'
  else
    python3 $RP ssh $POD_ID "curl -sS -X POST http://localhost:8420/v1/adapters/load -H 'Content-Type: application/json' -d '{\"name\":\"${TRAIN_ADAPTER}\"}'"
  fi

  echo ">>> training rollouts: N=${NUM_TRAIN_TASKS} tasks × ${NUM_GENS} gens"
  python3 $RP bg $POD_ID "$TRAIN_LOG.rollout" \
    "cd ${POD_REPO} && rm -rf ${TRAIN_OUT} && python3 rollout.py \
      --tasks datasets/train.tasks.jsonl --task-limit ${NUM_TRAIN_TASKS} \
      --out-dir ${TRAIN_OUT} --mode train --num-generations ${NUM_GENS} \
      --max-wall-clock-s ${MAX_WALL} --concurrency 1 --verbose --adapter current 2>&1"
  python3 $RP wait-file $POD_ID "${TRAIN_OUT}/summary.json" --timeout 7200

  python3 $RP ssh $POD_ID "cat ${TRAIN_OUT}/summary.json | head -c 1200"

  echo ">>> filter strong-signal groups (var > ${FILTER_VAR})"
  python3 $RP ssh $POD_ID "python3 - <<PYEOF
import json, statistics
inp = '${TRAIN_OUT}/grpo-train.jsonl'
out = '${TRAIN_OUT}/grpo-train-strong.jsonl'
kept = 0; total = 0
with open(out, 'w') as fo:
    for line in open(inp):
        total += 1
        g = json.loads(line)
        rewards = [c.get('reward', 0) for c in g.get('completions', [])]
        if len(rewards) >= 2 and statistics.variance(rewards) > ${FILTER_VAR}:
            fo.write(line); kept += 1
print(f'kept {kept}/{total} strong-signal groups')
PYEOF"

  echo ">>> kill kiln serve, train GRPO step"
  python3 $RP ssh $POD_ID 'pkill -9 -f "kiln serve" 2>/dev/null || true; sleep 3'

  python3 $RP bg $POD_ID "$TRAIN_LOG" \
    "cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b \
      ./target/release/examples/cuda_grpo_ablation \
      --data ${TRAIN_OUT}/grpo-train-strong.jsonl \
      --model /workspace/qwen3.5-4b \
      --output ${ADAPTER_OUT} \
      --adapter pi-cc-iter${ITER} \
      --mode phase1 --rank ${RANK} --alpha ${ALPHA} --lr ${LR} --seed ${SEED} \
      ${ECHO_FLAGS[@]+\"\${ECHO_FLAGS[@]}\"} 2>&1"
  python3 $RP wait-file $POD_ID "${ADAPTER_OUT}/pi-cc-iter${ITER}/adapter_model.safetensors" --timeout 1800

  echo ">>> symlink adapter into kiln model dir"
  python3 $RP ssh $POD_ID "ln -sfn ${ADAPTER_OUT}/pi-cc-iter${ITER} /workspace/qwen3.5-4b/adapters/pi-cc-iter${ITER}"

  echo ">>> restart kiln serve"
  python3 $RP bg $POD_ID /tmp/kiln-serve-iter${ITER}.log \
    'cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b ./target/release/kiln serve 2>&1'
  sleep 25
  python3 $RP ssh $POD_ID "curl -sS http://localhost:8420/v1/adapters | head -c 400"
fi

############################################################################
# 4 — eval
############################################################################
if [ "$SKIP_EVAL" = "0" ]; then
  echo ">>> load eval adapter: ${EVAL_ADAPTER}"
  if [ -z "$EVAL_ADAPTER" ] || [ "$EVAL_ADAPTER" = "base" ]; then
    python3 $RP ssh $POD_ID 'curl -sS -X POST http://localhost:8420/v1/adapters/unload >/dev/null || true'
  else
    python3 $RP ssh $POD_ID "curl -sS -X POST http://localhost:8420/v1/adapters/load -H 'Content-Type: application/json' -d '{\"name\":\"${EVAL_ADAPTER}\"}'"
  fi

  echo ">>> eval rollouts"
  python3 $RP bg $POD_ID "/tmp/iter${ITER}-eval.log" \
    "cd ${POD_REPO} && rm -rf ${EVAL_OUT} && python3 rollout.py \
      --tasks datasets/eval.tasks.jsonl --task-limit ${EVAL_TASKS} \
      --out-dir ${EVAL_OUT} --mode eval --num-generations 1 \
      --max-wall-clock-s ${MAX_WALL} --adapter current --concurrency 1 --verbose 2>&1"
  python3 $RP wait-file $POD_ID "${EVAL_OUT}/summary.json" --timeout 7200

  echo ">>> eval done"
  python3 $RP ssh $POD_ID "cat ${EVAL_OUT}/summary.json"
fi

############################################################################
# 5 — backup to B2 (locally — Cloud Eric has B2 creds)
############################################################################
echo ">>> backing up iter ${ITER} to B2"
python3 ${0%/*}/backup_to_b2.py --iter ${ITER} --kind ${KIND} --pod ${POD_ID} || echo "backup failed (continuing)"

echo "== iter ${ITER} done =="
