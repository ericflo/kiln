#!/bin/bash
# Run one complete iter of pi-compaction GRPO.
#
# Stages:
#   1. (optional) Generate training rollouts from a chosen adapter (or base).
#   2. (optional) Filter to strong-signal groups (var > 0.05).
#   3. (optional) Train GRPO -> save adapter.
#   4. Switch the served adapter, eval on the held-out set.
#   5. Pull rollouts + summary down. Compute summary stats.
#   6. Back up to B2 (adapter + rollouts + summary).
#
# Drives via env vars + flags so the autonomous loop can compose iters
# without writing one-off scripts each time.
#
# Usage:
#   run_iter.sh --iter N --kind baseline|train|eval|abl \
#               --num-train-tasks 24 --num-gens 4 \
#               --train-adapter "" --eval-adapter "pi-compaction-iterN" \
#               --lr 1e-5 --filter-var 0.05 \
#               --skip-train (eval-only)

set -euo pipefail

ITER=""
KIND="train"
NUM_TRAIN_TASKS=24
NUM_GENS=4
TRAIN_ADAPTER=""     # adapter to use for *training rollouts*
EVAL_ADAPTER=""      # adapter to eval (defaults to pi-compaction-iter<N>)
LR="1e-5"
FILTER_VAR="0.05"
SKIP_TRAIN=0
SKIP_EVAL=0
SEED=3141592653
EPOCHS=1
RANK=16
ALPHA=32

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
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$ITER" ]; then echo "--iter required" >&2; exit 1; fi
EVAL_ADAPTER="${EVAL_ADAPTER:-pi-compaction-iter${ITER}}"

source /tmp/grpo-pod.env

POD_REPO=/workspace/kiln/capabilities/agentic-grpo/pi-compaction
TRAIN_OUT="/tmp/iter${ITER}-rollouts"
EVAL_OUT="/tmp/iter${ITER}-eval"
ADAPTER_OUT="/tmp/iter${ITER}-adapter"
TRAIN_LOG="/tmp/iter${ITER}-train.log"

echo "== iter ${ITER} kind=${KIND} =="

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

  echo ">>> training rollouts: N=${NUM_TRAIN_TASKS} tasks × ${NUM_GENS} gens"
  python3 $RP bg $POD_ID "$TRAIN_LOG.rollout" \
    "cd ${POD_REPO} && rm -rf ${TRAIN_OUT} && python3 rollout.py \
      --tasks datasets/train.tasks.jsonl --task-limit ${NUM_TRAIN_TASKS} \
      --out-dir ${TRAIN_OUT} --mode train --num-generations ${NUM_GENS} \
      --seed ${SEED} --concurrency 2 --verbose 2>&1"
  python3 $RP wait-file $POD_ID "${TRAIN_OUT}/summary.json" --timeout 3600

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

  echo ">>> killing kiln serve, training GRPO step"
  python3 $RP ssh $POD_ID 'pkill -9 -f "kiln serve" 2>/dev/null || true; sleep 3'

  python3 $RP bg $POD_ID "$TRAIN_LOG" \
    "cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b \
      ./target/release/examples/cuda_grpo_ablation \
      --data ${TRAIN_OUT}/grpo-train-strong.jsonl \
      --model /workspace/qwen3.5-4b \
      --output ${ADAPTER_OUT} \
      --adapter pi-compaction-iter${ITER} \
      --mode phase1 --rank ${RANK} --alpha ${ALPHA} --lr ${LR} --seed ${SEED} 2>&1"
  python3 $RP wait-file $POD_ID "${ADAPTER_OUT}/pi-compaction-iter${ITER}/adapter_model.safetensors" --timeout 1200

  echo ">>> symlinking adapter into kiln model dir"
  python3 $RP ssh $POD_ID "ln -sfn ${ADAPTER_OUT}/pi-compaction-iter${ITER} /workspace/qwen3.5-4b/adapters/pi-compaction-iter${ITER}"

  echo ">>> restarting kiln serve"
  python3 $RP bg $POD_ID /tmp/kiln-serve-iter${ITER}.log \
    'cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b ./target/release/kiln serve 2>&1'
  sleep 20
  python3 $RP ssh $POD_ID "curl -sS http://localhost:8420/v1/adapters | head -c 400"
fi

###############################################################################
# 4+5: eval
###############################################################################
if [ "$SKIP_EVAL" = "0" ]; then
  echo ">>> loading eval adapter: ${EVAL_ADAPTER}"
  if [ -z "$EVAL_ADAPTER" ] || [ "$EVAL_ADAPTER" = "base" ]; then
    python3 $RP ssh $POD_ID 'curl -sS -X POST http://localhost:8420/v1/adapters/unload >/dev/null'
  else
    python3 $RP ssh $POD_ID "curl -sS -X POST http://localhost:8420/v1/adapters/load -H 'Content-Type: application/json' -d '{\"name\":\"${EVAL_ADAPTER}\"}'"
  fi

  echo ">>> eval rollouts (24 tasks × 1 gen)"
  python3 $RP bg $POD_ID "/tmp/iter${ITER}-eval.log" \
    "cd ${POD_REPO} && rm -rf ${EVAL_OUT} && python3 rollout.py \
      --tasks datasets/eval.tasks.jsonl \
      --out-dir ${EVAL_OUT} --mode eval --num-generations 1 \
      --adapter current --seed ${SEED} --concurrency 2 --verbose 2>&1"
  python3 $RP wait-file $POD_ID "${EVAL_OUT}/summary.json" --timeout 3600

  echo ">>> eval done"
  python3 $RP ssh $POD_ID "cat ${EVAL_OUT}/summary.json"
fi

###############################################################################
# 6: backup to B2 (locally on Cloud Eric since it has B2 creds)
###############################################################################
echo ">>> backing up iter ${ITER} to B2"
python3 ${0%/*}/backup_to_b2.py --iter ${ITER} --kind ${KIND} --pod ${POD_ID}

echo "== iter ${ITER} done =="
