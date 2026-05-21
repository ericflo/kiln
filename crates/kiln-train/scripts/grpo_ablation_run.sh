#!/bin/bash
# GRPO Phase 1+2 ablation runner — wraps cuda_grpo_ablation across all five
# named modes, capturing one log per mode for later analysis.
#
# Expected to run on a kiln pod with the repo checked out at /workspace/kiln
# and the model weights at /workspace/Qwen3.5-4B/.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/kiln}
MODEL_DIR=${MODEL_DIR:-/workspace/Qwen3.5-4B}
DATASET=${DATASET:-$REPO_DIR/capabilities/sft/python-algo/datasets/grpo-humaneval.jsonl}
MAX_GROUPS=${MAX_GROUPS:-20}
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/grpo-ablation-$(date +%Y%m%d-%H%M%S)}
LOG_ROOT=$OUTPUT_ROOT/logs
SEED=${SEED:-6952087183}
LR=${LR:-1e-5}
RANK=${RANK:-8}
ALPHA=${ALPHA:-16}

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT/adapters"

cd "$REPO_DIR"

# Build the example once, then drive all modes.
export KILN_CUDA_ARCHS=${KILN_CUDA_ARCHS:-86}
cargo build --release --features cuda --example cuda_grpo_ablation 2>&1 | tail -3

BIN="$REPO_DIR/target/release/examples/cuda_grpo_ablation"

for MODE in baseline phase1 phase1_gspo phase1_cispo phase1_reinforce; do
  echo "=== running mode=$MODE ==="
  "$BIN" \
    --data "$DATASET" \
    --model "$MODEL_DIR" \
    --output "$OUTPUT_ROOT/adapters/$MODE" \
    --adapter "$MODE" \
    --mode "$MODE" \
    --max-groups "$MAX_GROUPS" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --lr "$LR" \
    --seed "$SEED" \
    > "$LOG_ROOT/$MODE.log" 2>&1 || echo "FAILED: $MODE (see $LOG_ROOT/$MODE.log)"
  echo "=== done mode=$MODE ($(tail -1 "$LOG_ROOT/$MODE.log")) ==="
done

# Tar the logs so we can pull them back in one scp.
tar -czf "$OUTPUT_ROOT/logs.tar.gz" -C "$OUTPUT_ROOT" logs
echo "logs_tar=$OUTPUT_ROOT/logs.tar.gz"
echo "DONE"
