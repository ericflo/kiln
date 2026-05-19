#!/usr/bin/env bash
# Run one complete pi-failure-triage iter on the pod.
#
# Stages (each skippable):
#   1. (optional) Training rollouts — only if --rollout-source-iter unset.
#   2. (optional) Strong-signal filter (variance threshold).
#   3. (optional) GRPO step → adapter.
#   4. (optional) Eval rollouts.
#
# Drive caches rollouts: subsequent iters can specify
#   --rollout-source-iter <N> to reuse iter N's grpo-train.jsonl
# rather than regenerating rollouts. This is essential for fitting 50
# iters into an overnight window.
#
# Run on the LOCAL cloud-eric box; commands ssh into the pod.

set -euo pipefail

ITER=""
KIND="train"
NUM_TRAIN_TASKS=8
NUM_GENS=4
NUM_EVAL_TASKS=8
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
ADV_MODE="dr_grpo"
KL_COEFF="0.1"
CLIP_EPS="0.20"
ECHO_LAMBDA="0.05"
NO_ECHO=0
NO_POLICY=0
MAX_WALL=180
PARALLEL=1
EXTRA_TRAIN_ARGS=""
ROLLOUT_SOURCE_ITER=""   # if set, reuse rollouts from iter <N> instead of regenerating

while [ $# -gt 0 ]; do
  case "$1" in
    --iter) ITER="$2"; shift 2 ;;
    --kind) KIND="$2"; shift 2 ;;
    --num-train-tasks) NUM_TRAIN_TASKS="$2"; shift 2 ;;
    --num-gens) NUM_GENS="$2"; shift 2 ;;
    --num-eval-tasks) NUM_EVAL_TASKS="$2"; shift 2 ;;
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
    --adv-mode) ADV_MODE="$2"; shift 2 ;;
    --kl-coeff) KL_COEFF="$2"; shift 2 ;;
    --clip-eps) CLIP_EPS="$2"; shift 2 ;;
    --echo-lambda) ECHO_LAMBDA="$2"; shift 2 ;;
    --no-echo) NO_ECHO=1; shift ;;
    --no-policy) NO_POLICY=1; shift ;;
    --max-wall) MAX_WALL="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --extra-train-args) EXTRA_TRAIN_ARGS="$2"; shift 2 ;;
    --rollout-source-iter) ROLLOUT_SOURCE_ITER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$ITER" ]; then echo "--iter required" >&2; exit 1; fi
EVAL_ADAPTER="${EVAL_ADAPTER:-pi-failure-triage-iter${ITER}}"

source /tmp/grpo-pod.env

POD_REPO=/workspace/kiln/capabilities/agentic-grpo/pi-failure-triage
TRAIN_OUT="/tmp/pft-iter${ITER}-rollouts"
EVAL_OUT="/tmp/pft-iter${ITER}-eval"
ADAPTER_OUT="/tmp/pft-iter${ITER}-adapter"
TRAIN_LOG="/tmp/pft-iter${ITER}-train.log"
ROLLOUT_LOG="/tmp/pft-iter${ITER}-rollout.log"
EVAL_LOG="/tmp/pft-iter${ITER}-eval.log"

# If rollouts are reused, point TRAIN_OUT at the source iter's dir.
if [ -n "$ROLLOUT_SOURCE_ITER" ]; then
  TRAIN_OUT="/tmp/pft-iter${ROLLOUT_SOURCE_ITER}-rollouts"
fi

echo "== pft iter ${ITER} kind=${KIND} rollout_source=${ROLLOUT_SOURCE_ITER:-fresh} =="

###############################################################################
# 1+2+3: rollouts (or reuse) → filter → GRPO step
###############################################################################
if [ "$SKIP_TRAIN" = "0" ]; then
  if [ -z "$ROLLOUT_SOURCE_ITER" ]; then
    echo ">>> setting train adapter to '${TRAIN_ADAPTER:-(base)}'"
    if [ -z "$TRAIN_ADAPTER" ] || [ "$TRAIN_ADAPTER" = "base" ]; then
      python3 $RP ssh $POD_ID 'curl -sS -X POST http://localhost:8420/v1/adapters/unload >/dev/null 2>&1 || true'
    else
      python3 $RP ssh $POD_ID "curl -sS -X POST http://localhost:8420/v1/adapters/load -H 'Content-Type: application/json' -d '{\"name\":\"${TRAIN_ADAPTER}\"}'"
    fi

    echo ">>> training rollouts: N=${NUM_TRAIN_TASKS} tasks × ${NUM_GENS} gens"
    python3 $RP bg $POD_ID "$ROLLOUT_LOG" \
      "cd ${POD_REPO} && rm -rf ${TRAIN_OUT} && python3 rollout.py \
        --tasks datasets/train.tasks.jsonl --limit ${NUM_TRAIN_TASKS} \
        --out-dir ${TRAIN_OUT} --mode train --num-generations ${NUM_GENS} \
        --adapter current \
        --max-wall-clock-s ${MAX_WALL} --parallel ${PARALLEL} --verbose 2>&1"
    python3 $RP wait-file $POD_ID "${TRAIN_OUT}/summary.json" --timeout 7200
  else
    echo ">>> reusing rollouts from iter ${ROLLOUT_SOURCE_ITER} → ${TRAIN_OUT}"
  fi

  echo ">>> filtering strong-signal groups (var > ${FILTER_VAR})"
  python3 $RP ssh $POD_ID "python3 - <<PYEOF
import json, statistics
inp = '${TRAIN_OUT}/grpo-train.jsonl'
out = '/tmp/pft-iter${ITER}-grpo-train-strong.jsonl'
kept = 0; total = 0
with open(out, 'w') as fo:
    for line in open(inp):
        total += 1
        g = json.loads(line)
        rewards = [c.get('reward', 0) for c in g.get('completions', [])]
        if len(rewards) >= 2 and statistics.variance(rewards) > ${FILTER_VAR}:
            fo.write(line)
            kept += 1
print(f'kept {kept}/{total} strong-signal groups')
PYEOF"

  echo ">>> killing kiln serve, training GRPO step"
  python3 $RP ssh $POD_ID 'pkill -9 -f "kiln serve" 2>/dev/null || true; sleep 3'

  EXTRA_ECHO=""
  if [ "$NO_ECHO" = "1" ]; then EXTRA_ECHO="--no-echo"; fi
  EXTRA_POL=""
  if [ "$NO_POLICY" = "1" ]; then EXTRA_POL="--no-policy-loss"; fi

  python3 $RP bg $POD_ID "$TRAIN_LOG" \
    "cd /workspace/kiln && source /root/.kiln-build-env 2>/dev/null || true; KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b \
      ./target/release/examples/cuda_grpo_ablation \
      --data /tmp/pft-iter${ITER}-grpo-train-strong.jsonl \
      --model /workspace/qwen3.5-4b \
      --output ${ADAPTER_OUT} \
      --adapter pi-failure-triage-iter${ITER} \
      --mode phase1 --rank ${RANK} --alpha ${ALPHA} --lr ${LR} --seed ${SEED} \
      --advantage-mode ${ADV_MODE} --kl-coeff ${KL_COEFF} --clip-epsilon ${CLIP_EPS} \
      --echo-lambda ${ECHO_LAMBDA} \
      ${EXTRA_ECHO} ${EXTRA_POL} ${EXTRA_TRAIN_ARGS} 2>&1"
  python3 $RP wait-file $POD_ID "${ADAPTER_OUT}/pi-failure-triage-iter${ITER}/adapter_model.safetensors" --timeout 1800

  echo ">>> symlinking adapter into kiln model dir"
  python3 $RP ssh $POD_ID "mkdir -p /workspace/qwen3.5-4b/adapters && ln -sfn ${ADAPTER_OUT}/pi-failure-triage-iter${ITER} /workspace/qwen3.5-4b/adapters/pi-failure-triage-iter${ITER}"

  echo ">>> restarting kiln serve"
  python3 $RP bg $POD_ID /tmp/kiln-serve.log \
    'cd /workspace/kiln && source /root/b2-env; source /root/.kiln-build-env 2>/dev/null || true; KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b ./target/release/kiln serve 2>&1'
  # Wait for /v1/models to come up
  python3 $RP ssh $POD_ID 'for i in $(seq 1 24); do
    if curl -sf http://localhost:8420/v1/models > /dev/null 2>&1; then
      echo "kiln ready"
      break
    fi
    sleep 5
  done'
fi

###############################################################################
# 4+5: eval
###############################################################################
if [ "$SKIP_EVAL" = "0" ]; then
  echo ">>> loading eval adapter: ${EVAL_ADAPTER}"
  if [ -z "$EVAL_ADAPTER" ] || [ "$EVAL_ADAPTER" = "base" ]; then
    python3 $RP ssh $POD_ID 'curl -sS -X POST http://localhost:8420/v1/adapters/unload >/dev/null 2>&1 || true'
  else
    python3 $RP ssh $POD_ID "curl -sS -X POST http://localhost:8420/v1/adapters/load -H 'Content-Type: application/json' -d '{\"name\":\"${EVAL_ADAPTER}\"}'"
  fi

  echo ">>> eval rollouts (${NUM_EVAL_TASKS} tasks × 1 gen)"
  python3 $RP bg $POD_ID "$EVAL_LOG" \
    "cd ${POD_REPO} && rm -rf ${EVAL_OUT} && python3 rollout.py \
      --tasks datasets/eval.tasks.jsonl --limit ${NUM_EVAL_TASKS} \
      --out-dir ${EVAL_OUT} --mode eval --num-generations 1 \
      --adapter current --max-wall-clock-s ${MAX_WALL} --parallel ${PARALLEL} --verbose 2>&1"
  python3 $RP wait-file $POD_ID "${EVAL_OUT}/summary.json" --timeout 7200

  python3 $RP ssh $POD_ID "cat ${EVAL_OUT}/summary.json"
fi

echo "== pft iter ${ITER} done =="
