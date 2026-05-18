#!/usr/bin/env bash
# Iter 1 — H1 default Phase 1 GRPO recipe over pi-doctest train tasks.
set -euo pipefail
cd /workspace/kiln/capabilities/agentic-grpo/pi-doctest

SLUG="h1-default-recipe"
OUT_ROOT="/tmp/pi-doctest-iter1"
ROLLOUT_DIR="$OUT_ROOT/rollouts"
ADAPTER="pi-doctest-${SLUG}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$ROLLOUT_DIR" "$LOG_DIR"

# 1. Rollout pass — N=4 generations × ~30 training tasks.
python3 rollout.py \
  --tasks datasets/train.tasks.jsonl \
  --out-dir "$ROLLOUT_DIR" \
  --adapter "" \
  --num-generations 4 \
  --mode train \
  --max-wall-clock-s 120 \
  --parallel 1 \
  --limit 30 \
  --verbose \
  2>&1 | tee "$LOG_DIR/rollout.log"

# 2. GRPO step on the rollouts.
KILN_CUDA_ARCHS=80 /workspace/kiln/target/release/examples/cuda_grpo_ablation \
  --data "$ROLLOUT_DIR/grpo-train.jsonl" \
  --model /workspace/qwen3.5-4b \
  --output "$OUT_ROOT/adapter" \
  --adapter "$ADAPTER" \
  --mode phase1 \
  --max-groups 30 \
  --rank 16 --alpha 32 --lr 1e-5 \
  --num-generations 4 \
  --seed 3141592653 \
  2>&1 | tee "$LOG_DIR/train.log"
