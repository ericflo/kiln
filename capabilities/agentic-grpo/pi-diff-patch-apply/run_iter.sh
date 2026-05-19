#!/bin/bash
# Run one complete iter of pi-diff-patch-apply GRPO.
#
# This script drives a single iter end-to-end via the runpod pod.
# Stages:
#   1. Training rollouts under the chosen train-adapter (or base).
#   2. Filter strong-signal groups (variance > FILTER_VAR).
#   3. GRPO training step → adapter.
#   4. Eval rollouts on the held-out set with the new adapter.
#   5. Pull rollouts + summaries down. Back up to B2.
#
# Usage:
#   run_iter.sh --iter N --kind baseline|train|eval|abl \
#               --num-train-tasks 16 --num-gens 4 \
#               --train-adapter "" --eval-adapter "pi-diff-patch-apply-iterN" \
#               --lr 1e-5 --filter-var 0.04 \
#               [--skip-train] [--skip-eval]
#               [--echo-lambda 0.05 | --no-echo]
#               [--rank 16 --alpha 32]
#               [--seed 3141592653]

set -euo pipefail

ITER=""
KIND="train"
NUM_TRAIN_TASKS=16
NUM_GENS=4
TRAIN_ADAPTER=""
EVAL_ADAPTER=""
LR="1e-5"
FILTER_VAR="0.005"
SKIP_TRAIN=0
SKIP_EVAL=0
SEED=3141592653
EPOCHS=1
RANK=16
ALPHA=32
ECHO_LAMBDA=""        # empty = default; --no-echo overrides
NO_ECHO=0
NO_POLICY_LOSS=0
MAX_TURNS=12
MAX_WALL_CLOCK_S=180
PARALLEL=4
TEMPERATURE=0.8
EVAL_TASK_LIMIT=0     # 0 = full eval set (24); set positive for fast iters
TRAIN_TASKS_FILE="datasets/train.tasks.jsonl"  # override via --train-tasks-file

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
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    --max-wall-clock-s) MAX_WALL_CLOCK_S="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --eval-task-limit) EVAL_TASK_LIMIT="$2"; shift 2 ;;
    --train-tasks-file) TRAIN_TASKS_FILE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$ITER" ]; then echo "--iter required" >&2; exit 1; fi
EVAL_ADAPTER="${EVAL_ADAPTER:-pi-diff-patch-apply-iter${ITER}}"

source /tmp/grpo-pod.env  # sets POD_ID, RP

POD_REPO=/workspace/kiln/capabilities/agentic-grpo/pi-diff-patch-apply
TRAIN_OUT="/tmp/iter${ITER}-rollouts"
EVAL_OUT="/tmp/iter${ITER}-eval"
ADAPTER_OUT="/tmp/iter${ITER}-adapter"
TRAIN_LOG="/tmp/iter${ITER}-train.log"

echo "== iter ${ITER} kind=${KIND} ($(date -u +%Y-%m-%dT%H:%M:%SZ)) =="

###############################################################################
# 1+2+3: training rollouts -> filter -> GRPO step
###############################################################################
if [ "$SKIP_TRAIN" = "0" ]; then
  echo ">>> setting train adapter to '${TRAIN_ADAPTER:-(base)}'"
  if [ -z "$TRAIN_ADAPTER" ] || [ "$TRAIN_ADAPTER" = "base" ]; then
    python3 $RP ssh $POD_ID 'curl -sS -X POST http://localhost:8420/v1/adapters/unload >/dev/null'
  else
    python3 $RP ssh $POD_ID "curl -sS -X POST http://localhost:8420/v1/adapters/load -H 'Content-Type: application/json' -d '{\"name\":\"${TRAIN_ADAPTER}\"}'"
  fi

  echo ">>> training rollouts: N=${NUM_TRAIN_TASKS} tasks × ${NUM_GENS} gens (parallel=${PARALLEL})"
  python3 $RP bg $POD_ID "${TRAIN_LOG}.rollout" \
    "cd ${POD_REPO} && rm -rf ${TRAIN_OUT} && python3 rollout.py \
      --tasks ${TRAIN_TASKS_FILE} --task-limit ${NUM_TRAIN_TASKS} \
      --out-dir ${TRAIN_OUT} --mode train --num-generations ${NUM_GENS} \
      --seed-base ${SEED} --parallel ${PARALLEL} \
      --max-wall-clock-s ${MAX_WALL_CLOCK_S} \
      --temperature ${TEMPERATURE} --adapter current --verbose 2>&1"
  python3 $RP wait-file $POD_ID "${TRAIN_OUT}/summary.json" --timeout 7200

  echo ">>> filtering strong-signal groups (var > ${FILTER_VAR})"
  python3 $RP ssh $POD_ID "python3 - <<PYEOF
import json, statistics
inp = '${TRAIN_OUT}/grpo-train.jsonl'
out = '${TRAIN_OUT}/grpo-train-strong.jsonl'
kept = 0
all_groups = 0
with open(out, 'w') as fo:
    for line in open(inp):
        g = json.loads(line)
        all_groups += 1
        rewards = [c.get('reward', 0) for c in g.get('completions', [])]
        if len(rewards) >= 2 and statistics.variance(rewards) > ${FILTER_VAR}:
            fo.write(line)
            kept += 1
print(f'kept {kept}/{all_groups} strong-signal groups')
PYEOF"

  KEPT=$(python3 $RP ssh $POD_ID "wc -l < ${TRAIN_OUT}/grpo-train-strong.jsonl | tr -d ' '")
  if [ "$KEPT" -lt 2 ]; then
    echo "FATAL: only $KEPT strong-signal groups; training would be a no-op. Lowering filter to 0.0."
    python3 $RP ssh $POD_ID "cp ${TRAIN_OUT}/grpo-train.jsonl ${TRAIN_OUT}/grpo-train-strong.jsonl"
  fi

  echo ">>> killing kiln serve, training GRPO step"
  python3 $RP ssh $POD_ID 'pkill -9 -f "kiln serve" 2>/dev/null || true; sleep 3'

  ECHO_FLAG=""
  if [ "$NO_ECHO" = "1" ]; then
    ECHO_FLAG="--no-echo"
  elif [ -n "$ECHO_LAMBDA" ]; then
    ECHO_FLAG="--echo-lambda ${ECHO_LAMBDA}"
  fi
  POLICY_FLAG=""
  if [ "$NO_POLICY_LOSS" = "1" ]; then
    POLICY_FLAG="--no-policy-loss"
  fi

  python3 $RP bg $POD_ID "$TRAIN_LOG" \
    "cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b \
      ./target/release/examples/cuda_grpo_ablation \
      --data ${TRAIN_OUT}/grpo-train-strong.jsonl \
      --model /workspace/qwen3.5-4b \
      --output ${ADAPTER_OUT} \
      --adapter pi-diff-patch-apply-iter${ITER} \
      --mode phase1 --rank ${RANK} --alpha ${ALPHA} --lr ${LR} --seed ${SEED} \
      ${ECHO_FLAG} ${POLICY_FLAG} 2>&1"
  python3 $RP wait-file $POD_ID "${ADAPTER_OUT}/pi-diff-patch-apply-iter${ITER}/adapter_model.safetensors" --timeout 2400

  echo ">>> symlinking adapter into kiln model dir"
  python3 $RP ssh $POD_ID "ln -sfn ${ADAPTER_OUT}/pi-diff-patch-apply-iter${ITER} /workspace/qwen3.5-4b/adapters/pi-diff-patch-apply-iter${ITER}"

  echo ">>> restarting kiln serve"
  python3 $RP bg $POD_ID /tmp/kiln-serve-iter${ITER}.log \
    'cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b ./target/release/kiln serve 2>&1'
  sleep 30
  # Health check: kiln must respond AND the new adapter must be in the registry.
  HEALTH_OK=0
  for try in 1 2 3 4 5; do
    REG=$(python3 $RP ssh $POD_ID "curl -sS --max-time 10 http://localhost:8420/v1/adapters" 2>/dev/null)
    if echo "$REG" | grep -q "pi-diff-patch-apply-iter${ITER}"; then
      echo ">>> kiln serve healthy (try $try) — adapter in registry"
      HEALTH_OK=1
      break
    fi
    echo ">>> kiln serve not ready (try $try) — sleeping 15"
    sleep 15
  done
  if [ "$HEALTH_OK" = "0" ]; then
    echo "FATAL: kiln serve health check failed after restart. Killing + retrying once."
    python3 $RP ssh $POD_ID 'pkill -9 -f "kiln serve" 2>/dev/null || true; sleep 5'
    python3 $RP bg $POD_ID /tmp/kiln-serve-iter${ITER}-retry.log \
      'cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b ./target/release/kiln serve 2>&1'
    sleep 45
    REG=$(python3 $RP ssh $POD_ID "curl -sS --max-time 10 http://localhost:8420/v1/adapters" 2>/dev/null)
    if ! echo "$REG" | grep -q "pi-diff-patch-apply-iter${ITER}"; then
      echo "FATAL: kiln serve still not healthy. Bailing this iter — caller should mark INFRA-FAIL."
      exit 42
    fi
  fi
fi

###############################################################################
# 4: eval
###############################################################################
if [ "$SKIP_EVAL" = "0" ]; then
  echo ">>> loading eval adapter: ${EVAL_ADAPTER}"
  if [ -z "$EVAL_ADAPTER" ] || [ "$EVAL_ADAPTER" = "base" ]; then
    python3 $RP ssh $POD_ID 'curl -sS -X POST http://localhost:8420/v1/adapters/unload >/dev/null'
  else
    python3 $RP ssh $POD_ID "curl -sS -X POST http://localhost:8420/v1/adapters/load -H 'Content-Type: application/json' -d '{\"name\":\"${EVAL_ADAPTER}\"}'"
  fi

  EVAL_LIMIT_ARG=""
  if [ "${EVAL_TASK_LIMIT}" -gt 0 ]; then
    EVAL_LIMIT_ARG="--task-limit ${EVAL_TASK_LIMIT}"
  fi
  # Smoke check: 1 rollout on a known easy task. If it fails (zero tool calls / <30s session),
  # the adapter is broken — restart kiln before full eval.
  echo ">>> smoke test eval (1 task × 1 gen) to detect adapter/kiln infra failure"
  SMOKE_OUT="/tmp/iter${ITER}-smoke"
  python3 $RP bg $POD_ID "/tmp/iter${ITER}-smoke.log" \
    "cd ${POD_REPO} && rm -rf ${SMOKE_OUT} && python3 rollout.py \
      --tasks datasets/eval.tasks.jsonl --task-limit 1 \
      --out-dir ${SMOKE_OUT} --mode eval --num-generations 1 \
      --adapter current --seed-base ${SEED} --parallel 1 \
      --max-wall-clock-s 120 \
      --temperature 0.0 --verbose 2>&1"
  python3 $RP wait-file $POD_ID "${SMOKE_OUT}/summary.json" --timeout 900 || true
  SMOKE_SCORE=$(python3 $RP ssh $POD_ID "python3 -c 'import json; print(json.load(open(\"${SMOKE_OUT}/summary.json\"))[\"mean_composite\"])' 2>/dev/null" || echo "0")
  echo ">>> smoke test composite: ${SMOKE_SCORE}"
  echo ">>> eval rollouts: ${EVAL_TASK_LIMIT:-full} tasks × 1 gen (parallel=${PARALLEL})"
  python3 $RP bg $POD_ID "/tmp/iter${ITER}-eval.log" \
    "cd ${POD_REPO} && rm -rf ${EVAL_OUT} && python3 rollout.py \
      --tasks datasets/eval.tasks.jsonl ${EVAL_LIMIT_ARG} \
      --out-dir ${EVAL_OUT} --mode eval --num-generations 1 \
      --adapter current --seed-base ${SEED} --parallel ${PARALLEL} \
      --max-wall-clock-s ${MAX_WALL_CLOCK_S} \
      --temperature 0.0 --verbose 2>&1"
  python3 $RP wait-file $POD_ID "${EVAL_OUT}/summary.json" --timeout 7200

  echo ">>> eval done"
  python3 $RP ssh $POD_ID "cat ${EVAL_OUT}/summary.json"
fi

###############################################################################
# 5: backup to B2 (locally on Cloud Eric since it has B2 creds)
###############################################################################
echo ">>> backing up iter ${ITER} to B2"
python3 ${0%/*}/backup_to_b2.py --iter ${ITER} --kind ${KIND} --pod ${POD_ID} || \
  echo "WARN: B2 backup failed (continuing)"

echo "== iter ${ITER} done ($(date -u +%Y-%m-%dT%H:%M:%SZ)) =="
