#!/usr/bin/env bash
# run_iter1.sh — Phase 1 default recipe for pi-terminal-bench-lite.
# Paired ECHO vs no-ECHO runs at lr=1e-5, rank=16, alpha=32, seed
# 3141592653. Set ECHO_MODE=on (default) or ECHO_MODE=off.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ECHO_MODE="${ECHO_MODE:-on}"
ITER="${ITER:-1}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-/tmp/pi-tblite-iter${ITER}}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/workspace/adapters}"
LOG_DIR="${LOG_DIR:-$ROLLOUT_ROOT/logs}"
MAX_GROUPS="${MAX_GROUPS:-20}"
SEED="${SEED:-3141592653}"

ADAPTER="tblite-iter${ITER}-${ECHO_MODE}"
ECHO_FLAG=""
if [ "$ECHO_MODE" = "off" ]; then
    ECHO_FLAG="--no-echo"
elif [ "$ECHO_MODE" != "on" ]; then
    echo "ECHO_MODE must be 'on' or 'off', got: $ECHO_MODE" >&2
    exit 2
fi

mkdir -p "$ROLLOUT_ROOT" "$LOG_DIR"

echo "=== pi-terminal-bench-lite iter $ITER ($ECHO_MODE) ==="
echo "  adapter: $ADAPTER"
echo "  echo:    $ECHO_FLAG (empty = ECHO on at default λ=0.05)"
echo "  seed:    $SEED"
echo "  log:     $LOG_DIR"
echo

# Phase A — gather rollouts on training set.
echo "[Phase A] Gathering training rollouts..."
python3 "$HERE/rollout.py" \
    --tasks "$HERE/datasets/train.tasks.jsonl" \
    --config "$HERE/capability.config.json" \
    --out-dir "$ROLLOUT_ROOT" \
    --mode train \
    --num-generations 4 \
    --seed "$SEED" 2>&1 | tee "$LOG_DIR/rollout.log"

# Phase B — train via cuda_grpo_ablation.
echo
echo "[Phase B] GRPO training..."
KILN_CUDA_ARCHS="${KILN_CUDA_ARCHS:-80}" \
"/workspace/kiln/target/release/examples/cuda_grpo_ablation" \
    --data "$ROLLOUT_ROOT/grpo-train.jsonl" \
    --model /workspace/qwen3.5-4b \
    --output "$ADAPTER_ROOT/$ADAPTER" \
    --adapter "$ADAPTER" \
    --mode phase1 \
    --max-groups "$MAX_GROUPS" \
    --rank 16 --alpha 32 --lr 1e-5 \
    --seed "$SEED" \
    $ECHO_FLAG \
    2>&1 | tee "$LOG_DIR/train.log"

# Phase C — blind eval on held-out 30 tasks.
echo
echo "[Phase C] Blind eval..."
OUT_DIR="$ROLLOUT_ROOT/eval" \
    "$HERE/capability.oracle.sh" "$ADAPTER" 2>&1 | tee "$LOG_DIR/eval.log"

echo
echo "=== iter $ITER ($ECHO_MODE) complete ==="
echo "  see $LOG_DIR/eval.log for SCORE=<composite>"
