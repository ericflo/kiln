#!/usr/bin/env bash
# Run one OPD experiment end-to-end.
#
# Usage:
#   run_exp.sh NAME [--rank R] [--lr LR] [--top-k K] [--samples-per-prompt N]
#                   [--max-prompts N] [--seed S] [--epochs E]
#
# Steps:
#   1. Stop kiln-server if running.
#   2. cuda_opd_from_fixture trains the LoRA against the teacher fixture.
#   3. Start kiln-server.
#   4. eval_kiln.py runs the rubric against the new adapter.
#   5. Stop kiln-server.
#   6. Append one JSON line to capability.jsonl summarising the result.
#
# Side effects:
#   - Writes adapter to /workspace/kiln/Qwen3.5-4B/adapters/<NAME>/
#   - Writes scores to judgments/<NAME>.json
#   - Appends a line to capability.jsonl
set -euo pipefail

NAME="${1:?usage: run_exp.sh NAME [--rank R] [--lr LR] ...}"
shift
RANK=32
LR=1e-5
TOP_K=32
SAMPLES_PER_PROMPT=1
EPOCHS=1
SEED=4218
MAX_PROMPTS=""
NOTES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rank) RANK="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --samples-per-prompt) SAMPLES_PER_PROMPT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-prompts) MAX_PROMPTS="$2"; shift 2 ;;
    --notes) NOTES="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

WORKDIR="/workspace/kiln/sft-cap.json-schema-adherence"
ADAPTER_DIR="/workspace/kiln/Qwen3.5-4B/adapters"
MODEL_PATH="/workspace/kiln/Qwen3.5-4B"
KILN_BIN="/workspace/kiln/target/release/kiln"
EXAMPLE_BIN="/workspace/kiln/target/release/examples/cuda_opd_from_fixture"
EXP_LOG="$WORKDIR/experiments/$NAME.log"
mkdir -p "$WORKDIR/experiments"

cd "$WORKDIR"

echo "=== exp $NAME ==="
echo "rank=$RANK lr=$LR top_k=$TOP_K samples_per_prompt=$SAMPLES_PER_PROMPT epochs=$EPOCHS seed=$SEED max_prompts=${MAX_PROMPTS:-all}"

# 1. Stop any running kiln-server.
pkill -9 -f "target/release/kiln serve" 2>/dev/null || true
sleep 2

# 2. Train.
T0=$(date +%s)
MAX_ARG=""
if [ -n "$MAX_PROMPTS" ]; then MAX_ARG="--max-prompts $MAX_PROMPTS"; fi
"$EXAMPLE_BIN" \
  --model-path "$MODEL_PATH" \
  --prompts datasets/train.opd.jsonl \
  --teacher-fixture datasets/teacher.fixture.jsonl \
  --output-dir "$ADAPTER_DIR" \
  --adapter-name "$NAME" \
  --top-k "$TOP_K" \
  --rank "$RANK" \
  --lr "$LR" \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --seed "$SEED" \
  $MAX_ARG \
  2>&1 | tee "$EXP_LOG" | tail -8
TRAIN_SECS=$(( $(date +%s) - T0 ))
echo "train: ${TRAIN_SECS}s"

# 3. Start kiln-server in CUDA mode.
nohup "$KILN_BIN" serve --config "$WORKDIR/kiln.toml" \
  > "$WORKDIR/kiln-server.log" 2>&1 &
SERVER_PID=$!
echo "kiln-server pid=$SERVER_PID"
# Wait for server ready by polling.
for i in $(seq 1 30); do
  if curl -s -m 2 http://localhost:8420/v1/models 2>/dev/null | grep -q "qwen3.5-4b-kiln"; then
    break
  fi
  sleep 1
done

# 4. Eval.
T0=$(date +%s)
python3 eval_kiln.py \
  --adapter "$NAME" \
  --out "judgments/$NAME.json" \
  --concurrency 4 \
  2>&1 | tail -1 > "$WORKDIR/experiments/$NAME.eval.json"
EVAL_SECS=$(( $(date +%s) - T0 ))

# 5. Stop server.
pkill -9 -f "target/release/kiln serve" 2>/dev/null || true
sleep 1

# 6. Append capability.jsonl row.
python3 - <<PYEOF
import json, time
agg = json.load(open("judgments/$NAME.json"))
row = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "name": "$NAME",
    "rank": $RANK,
    "lr": "$LR",
    "top_k": $TOP_K,
    "samples_per_prompt": $SAMPLES_PER_PROMPT,
    "epochs": $EPOCHS,
    "seed": $SEED,
    "max_prompts": ${MAX_PROMPTS:-None},
    "train_secs": $TRAIN_SECS,
    "eval_secs": $EVAL_SECS,
    "n_eval": agg["n"],
    "parses": agg["parses"],
    "validates": agg["validates"],
    "is_pure": agg["is_pure"],
    "is_substantive": agg["is_substantive"],
    "composite": agg["composite"],
    "notes": "$NOTES",
}
with open("capability.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"{'$NAME':<26} composite={agg['composite']:.4f} (parses={agg['parses']:.2f} validates={agg['validates']:.2f} is_pure={agg['is_pure']:.2f} is_substantive={agg['is_substantive']:.2f})")
PYEOF
