#!/usr/bin/env bash
# Paper §5.5 verifier-free env-only adaptation recipe.
#
# Takes the strongest Phase 2 ECHO checkpoint, runs 100 steps with
# --no-policy-loss + --echo-lambda 0.05 on filtered (clean tool-call)
# rollouts of held-out tasks, and measures pass-rate before/after on
# val100 / ITD / PyTerm. Target deltas (paper §5.5):
#
#   val100  : +3.8 pp
#   ITD     : +5.2 pp
#   PyTerm  : +10.0 pp
#   TBLite  : -3.9 pp (negative control — recipe doesn't generalize)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE_ADAPTER="${BASE_ADAPTER:-echo-tblite-iter5}"
OUTPUT_ADAPTER="${OUTPUT_ADAPTER:-echo-verifier-free-iter1}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-/tmp/pi-script-fixup-vf}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/workspace/adapters}"
LOG_DIR="${LOG_DIR:-$ROLLOUT_ROOT/logs}"
MAX_GROUPS="${MAX_GROUPS:-100}"
SEED="${SEED:-3141592653}"

mkdir -p "$ROLLOUT_ROOT" "$LOG_DIR"

echo "=== pi-script-fixup verifier-free adaptation ==="
echo "  base adapter:   $BASE_ADAPTER"
echo "  output adapter: $OUTPUT_ADAPTER"
echo "  max groups:     $MAX_GROUPS"
echo

# Phase A — baseline eval (BEFORE adaptation) on all four eval sets.
for set in val100 itd pyterm tblite; do
    echo "[Phase A] Baseline eval — $set..."
    OUT_DIR="$ROLLOUT_ROOT/baseline/$set" \
    bash "$HERE/capability.oracle.sh" "$BASE_ADAPTER" "$set" \
        2>&1 | tee "$LOG_DIR/baseline-$set.log"
done

# Phase B — gather PyTerm rollouts with base adapter, filter to clean
# tool-call trajectories.
echo
echo "[Phase B] Gathering clean PyTerm rollouts (base adapter)..."
python3 "$HERE/rollout.py" \
    --tasks "$HERE/datasets/pyterm.tasks.jsonl" \
    --config "$HERE/capability.config.json" \
    --out-dir "$ROLLOUT_ROOT/rollouts" \
    --mode train \
    --num-generations 4 \
    --adapter "$BASE_ADAPTER" \
    --filter-clean-tool-calls \
    --seed "$SEED" 2>&1 | tee "$LOG_DIR/rollout.log"

# Phase C — verifier-free adaptation: train 100 steps with
# --no-policy-loss + --echo-lambda 0.05.
echo
echo "[Phase C] Verifier-free adaptation (100 steps)..."
KILN_CUDA_ARCHS="${KILN_CUDA_ARCHS:-80}" \
"/workspace/kiln/target/release/examples/cuda_grpo_ablation" \
    --data "$ROLLOUT_ROOT/rollouts/grpo-train.jsonl" \
    --model /workspace/qwen3.5-4b \
    --output "$ADAPTER_ROOT/$OUTPUT_ADAPTER" \
    --adapter "$OUTPUT_ADAPTER" \
    --mode phase1 \
    --max-groups "$MAX_GROUPS" \
    --rank 16 --alpha 32 --lr 1e-5 \
    --seed "$SEED" \
    --no-policy-loss \
    --echo-lambda 0.05 \
    2>&1 | tee "$LOG_DIR/train.log"

# Phase D — post-adaptation eval on all four eval sets.
echo
for set in val100 itd pyterm tblite; do
    echo "[Phase D] Post-adaptation eval — $set..."
    OUT_DIR="$ROLLOUT_ROOT/post/$set" \
    bash "$HERE/capability.oracle.sh" "$OUTPUT_ADAPTER" "$set" \
        2>&1 | tee "$LOG_DIR/post-$set.log"
done

# Phase E — diff baseline vs post per set.
echo
echo "=== pass-rate deltas ==="
for set in val100 itd pyterm tblite; do
    BASE=$(python3 -c "import json; d=json.load(open('$ROLLOUT_ROOT/baseline/$set/eval.json')); print(d['mean_composite'])" 2>/dev/null || echo "0.0")
    POST=$(python3 -c "import json; d=json.load(open('$ROLLOUT_ROOT/post/$set/eval.json')); print(d['mean_composite'])" 2>/dev/null || echo "0.0")
    DELTA=$(python3 -c "print(f'{($POST - $BASE) * 100:+.2f}')" 2>/dev/null || echo "?")
    echo "  $set: baseline=$BASE  post=$POST  delta=${DELTA} pp"
done

echo
echo "Done. Paper §5.5 target deltas:"
echo "  val100: +3.8 pp | itd: +5.2 pp | pyterm: +10.0 pp | tblite: -3.9 pp"
