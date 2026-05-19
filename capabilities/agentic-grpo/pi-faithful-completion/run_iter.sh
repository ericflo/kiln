#!/bin/bash
# Run one complete iter of pi-faithful-completion GRPO.
#
# Stages:
#   1. (optional) Generate training rollouts with chosen adapter (or base).
#   2. (optional) Filter to strong-signal groups (var > threshold).
#   3. (optional) Train GRPO -> save adapter.
#   4. Switch the served adapter, eval on the held-out set.
#   5. Pull rollouts + summary down. Compute summary stats.
#
# All knobs are flags so the drive_iters.sh wrapper can compose iters
# without writing one-off scripts each time.
#
# Usage:
#   run_iter.sh --iter N --slug <slug> \
#               [--train-tasks N] [--num-gens N] \
#               [--train-adapter ""]   [--eval-adapter "pi-faithful-iterN"] \
#               [--lr 1e-5] [--rank 16] [--alpha 32] \
#               [--mode phase1] [--echo-lambda 0.05 | --no-echo] [--no-policy-loss] \
#               [--filter-var 0.0]   [--max-groups <int>] \
#               [--base-adapter <name>]  [--seed N] \
#               [--temperature 0.8] [--top-p 0.95] [--max-tokens 768] \
#               [--skip-train] [--skip-eval] \
#               [--system-prompt-file <path>]

set -euo pipefail

ITER=""
SLUG=""
NUM_TRAIN_TASKS=24
NUM_GENS=4
TRAIN_ADAPTER=""        # adapter to use for *training rollouts*
EVAL_ADAPTER=""         # adapter to eval (defaults to pi-faithful-iter<N>)
LR="1e-5"
RANK=16
ALPHA=32
MODE="phase1"
ECHO_LAMBDA=""
NO_ECHO=0
NO_POLICY_LOSS=0
FILTER_VAR="0.0"        # 0.0 = no filter; use small positive value to filter
MAX_GROUPS=""
BASE_ADAPTER=""
SEED=3141592653
TEMP="0.8"
TOP_P="0.95"
MAX_TOKENS="768"
SKIP_TRAIN=0
SKIP_EVAL=0
SYSTEM_PROMPT_FILE=""
TRAIN_TASKS_FILE="datasets/train.tasks.jsonl"
EVAL_TASKS_FILE="datasets/eval.tasks.jsonl"
KL_COEFF=""
CLIP_EPS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --iter) ITER="$2"; shift 2 ;;
    --slug) SLUG="$2"; shift 2 ;;
    --train-tasks) NUM_TRAIN_TASKS="$2"; shift 2 ;;
    --num-gens) NUM_GENS="$2"; shift 2 ;;
    --train-adapter) TRAIN_ADAPTER="$2"; shift 2 ;;
    --eval-adapter) EVAL_ADAPTER="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --rank) RANK="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --echo-lambda) ECHO_LAMBDA="$2"; shift 2 ;;
    --no-echo) NO_ECHO=1; shift ;;
    --no-policy-loss) NO_POLICY_LOSS=1; shift ;;
    --filter-var) FILTER_VAR="$2"; shift 2 ;;
    --max-groups) MAX_GROUPS="$2"; shift 2 ;;
    --base-adapter) BASE_ADAPTER="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --temperature) TEMP="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --system-prompt-file) SYSTEM_PROMPT_FILE="$2"; shift 2 ;;
    --train-tasks-file) TRAIN_TASKS_FILE="$2"; shift 2 ;;
    --eval-tasks-file)  EVAL_TASKS_FILE="$2"; shift 2 ;;
    --kl-coeff) KL_COEFF="$2"; shift 2 ;;
    --clip-eps) CLIP_EPS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$ITER" ]; then echo "--iter required" >&2; exit 1; fi
if [ -z "$SLUG" ]; then SLUG="iter${ITER}"; fi
EVAL_ADAPTER="${EVAL_ADAPTER:-pi-faithful-${SLUG}}"

source /tmp/pi-faithful.env

POD_REPO=/workspace/kiln/capabilities/agentic-grpo/pi-faithful-completion
TRAIN_OUT="/tmp/iter${ITER}-rollouts"
EVAL_OUT="/tmp/iter${ITER}-eval"
ADAPTER_OUT="/tmp/iter${ITER}-adapter"
TRAIN_LOG="/tmp/iter${ITER}-train.log"

echo "== iter ${ITER} slug=${SLUG} train=${SKIP_TRAIN}=skip eval=${SKIP_EVAL}=skip =="

###############################################################################
# 1+2+3: training rollouts -> GRPO step
###############################################################################
if [ "$SKIP_TRAIN" = "0" ]; then
  echo ">>> setting train adapter to '${TRAIN_ADAPTER:-(base)}'"
  if [ -z "$TRAIN_ADAPTER" ] || [ "$TRAIN_ADAPTER" = "base" ]; then
    python3 $RP ssh $POD_ID 'curl -sS -X POST http://localhost:8420/v1/adapters/unload >/dev/null'
  else
    python3 $RP ssh $POD_ID "curl -sS -X POST http://localhost:8420/v1/adapters/load -H 'Content-Type: application/json' -d '{\"name\":\"${TRAIN_ADAPTER}\"}'"
  fi

  EXTRA_ROLLOUT_ARGS=""
  if [ -n "$SYSTEM_PROMPT_FILE" ]; then
    EXTRA_ROLLOUT_ARGS="$EXTRA_ROLLOUT_ARGS --system-prompt-file $SYSTEM_PROMPT_FILE"
  fi

  echo ">>> training rollouts: N=${NUM_TRAIN_TASKS} tasks x ${NUM_GENS} gens"
  python3 $RP bg $POD_ID "$TRAIN_LOG.rollout" \
    "cd ${POD_REPO} && rm -rf ${TRAIN_OUT} && python3 rollout.py \
      --tasks ${TRAIN_TASKS_FILE} --task-limit ${NUM_TRAIN_TASKS} \
      --out-dir ${TRAIN_OUT} --mode train --num-generations ${NUM_GENS} \
      --temperature ${TEMP} --top-p ${TOP_P} --max-tokens ${MAX_TOKENS} \
      --seed ${SEED} --concurrency 3 --verbose $EXTRA_ROLLOUT_ARGS 2>&1"
  python3 $RP wait-file $POD_ID "${TRAIN_OUT}/summary.json" --timeout 1800

  echo ">>> filtering strong-signal groups (var > ${FILTER_VAR})"
  python3 $RP ssh $POD_ID "python3 - <<PYEOF
import json, statistics
inp = '${TRAIN_OUT}/grpo-train.jsonl'
out = '${TRAIN_OUT}/grpo-train-strong.jsonl'
kept = 0
with open(out, 'w') as fo:
    for line in open(inp):
        g = json.loads(line)
        rewards = [c.get('reward', 0) for c in g.get('completions', [])]
        if len(rewards) >= 2 and statistics.variance(rewards) > ${FILTER_VAR}:
            fo.write(line)
            kept += 1
print(f'kept {kept} strong-signal groups')
PYEOF"

  echo ">>> killing kiln serve before training (frees VRAM)"
  python3 $RP ssh $POD_ID 'pkill -9 -f "kiln serve" 2>/dev/null || true; sleep 3'

  TRAIN_ARGS="--data ${TRAIN_OUT}/grpo-train-strong.jsonl \
    --model /workspace/qwen3.5-4b \
    --output ${ADAPTER_OUT} \
    --adapter ${EVAL_ADAPTER} \
    --mode ${MODE} --rank ${RANK} --alpha ${ALPHA} --lr ${LR} --seed ${SEED}"
  if [ "$NO_ECHO" = "1" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --no-echo"
  elif [ -n "$ECHO_LAMBDA" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --echo-lambda $ECHO_LAMBDA"
  fi
  if [ "$NO_POLICY_LOSS" = "1" ]; then TRAIN_ARGS="$TRAIN_ARGS --no-policy-loss"; fi
  if [ -n "$BASE_ADAPTER" ]; then TRAIN_ARGS="$TRAIN_ARGS --base-adapter $BASE_ADAPTER"; fi
  if [ -n "$MAX_GROUPS" ]; then TRAIN_ARGS="$TRAIN_ARGS --max-groups $MAX_GROUPS"; fi

  EXTRA_ENV=""
  if [ -n "$KL_COEFF" ]; then EXTRA_ENV="$EXTRA_ENV KILN_GRPO_KL_COEFF=$KL_COEFF"; fi
  if [ -n "$CLIP_EPS" ]; then EXTRA_ENV="$EXTRA_ENV KILN_GRPO_CLIP_EPSILON=$CLIP_EPS"; fi

  python3 $RP bg $POD_ID "$TRAIN_LOG" \
    "cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 \
     KILN_MODEL_PATH=/workspace/qwen3.5-4b ${EXTRA_ENV} \
     ./target/release/examples/cuda_grpo_ablation ${TRAIN_ARGS} 2>&1"
  python3 $RP wait-file $POD_ID "${ADAPTER_OUT}/${EVAL_ADAPTER}/adapter_model.safetensors" --timeout 1800

  echo ">>> symlinking adapter into kiln model dir"
  python3 $RP ssh $POD_ID "mkdir -p /workspace/qwen3.5-4b/adapters && ln -sfn ${ADAPTER_OUT}/${EVAL_ADAPTER} /workspace/qwen3.5-4b/adapters/${EVAL_ADAPTER}"

  echo ">>> restarting kiln serve"
  python3 $RP bg $POD_ID /tmp/kiln-serve-iter${ITER}.log \
    'cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b ./target/release/kiln serve 2>&1'
  sleep 25
  python3 $RP ssh $POD_ID "curl -sS http://localhost:8420/v1/adapters | head -c 400" || true
fi

###############################################################################
# 4+5: eval
###############################################################################
if [ "$SKIP_EVAL" = "0" ]; then
  echo ">>> loading eval adapter: ${EVAL_ADAPTER}"
  if [ -z "$EVAL_ADAPTER" ] || [ "$EVAL_ADAPTER" = "base" ]; then
    python3 $RP ssh $POD_ID 'curl -sS -X POST http://localhost:8420/v1/adapters/unload >/dev/null'
  else
    python3 $RP ssh $POD_ID "curl -sS -X POST http://localhost:8420/v1/adapters/load -H 'Content-Type: application/json' -d '{\"name\":\"${EVAL_ADAPTER}\"}'" || true
  fi

  echo ">>> eval rollouts on held-out set"
  python3 $RP bg $POD_ID "/tmp/iter${ITER}-eval.log" \
    "cd ${POD_REPO} && rm -rf ${EVAL_OUT} && python3 rollout.py \
      --tasks ${EVAL_TASKS_FILE} \
      --out-dir ${EVAL_OUT} --mode eval --num-generations 1 \
      --adapter current --temperature 0.2 --top-p 0.95 --max-tokens ${MAX_TOKENS} \
      --seed ${SEED} --concurrency 3 --verbose 2>&1"
  python3 $RP wait-file $POD_ID "${EVAL_OUT}/summary.json" --timeout 1800

  echo ">>> eval done"
  python3 $RP ssh $POD_ID "cat ${EVAL_OUT}/summary.json"
fi

echo "== iter ${ITER} done =="
