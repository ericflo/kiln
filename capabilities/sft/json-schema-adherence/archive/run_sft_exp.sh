#!/usr/bin/env bash
# Run one SFT-distillation experiment end-to-end (off-policy distillation on
# teacher-generated assistant turns).
#
# Mirrors run_exp.sh but uses cuda_sft_file (which has gradient checkpointing)
# instead of cuda_opd_from_fixture (which doesn't yet). Same eval + capability.jsonl
# logging.
set -euo pipefail

NAME="${1:?usage: run_sft_exp.sh NAME [--rank R] [--lr LR] [--epochs E] [--max-examples N] [--notes 'free text']}"
shift
RANK=16
LR=1e-4
EPOCHS=2
SEED=4218
MAX_EXAMPLES=""
NOTES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rank) RANK="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-examples) MAX_EXAMPLES="$2"; shift 2 ;;
    --notes) NOTES="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

WORKDIR="/workspace/kiln/sft-cap.json-schema-adherence"
ADAPTER_DIR="/workspace/kiln/Qwen3.5-4B/adapters"
MODEL_PATH="/workspace/kiln/Qwen3.5-4B"
KILN_BIN="/workspace/kiln/target/release/kiln"
SFT_BIN="/workspace/kiln/target/release/examples/cuda_sft_file"
EXP_LOG="$WORKDIR/experiments/$NAME.log"
mkdir -p "$WORKDIR/experiments"

cd "$WORKDIR"
echo "=== exp $NAME ==="
echo "rank=$RANK lr=$LR epochs=$EPOCHS seed=$SEED max_examples=${MAX_EXAMPLES:-all}"

pkill -9 -f "target/release/kiln serve" 2>/dev/null || true
sleep 2

T0=$(date +%s)
MAX_ARG=""
if [ -n "$MAX_EXAMPLES" ]; then MAX_ARG="--max-examples $MAX_EXAMPLES"; fi
"$SFT_BIN" \
  --data datasets/train.opd.jsonl \
  --model-path "$MODEL_PATH" \
  --output-dir "$ADAPTER_DIR" \
  --adapter-name "$NAME" \
  --epochs "$EPOCHS" \
  --rank "$RANK" \
  --lr "$LR" \
  --trainer generic \
  $MAX_ARG \
  2>&1 | tee "$EXP_LOG" | tail -8
TRAIN_SECS=$(( $(date +%s) - T0 ))
echo "train: ${TRAIN_SECS}s"

# cuda_sft_file now accepts --rank --lr (patched in this branch).

nohup "$KILN_BIN" serve --config "$WORKDIR/kiln.toml" \
  > "$WORKDIR/kiln-server.log" 2>&1 &
echo "kiln-server pid=$!"
for i in $(seq 1 40); do
  if curl -s -m 2 http://localhost:8420/v1/models 2>/dev/null | grep -q "qwen3.5-4b"; then
    break
  fi
  sleep 1
done

T0=$(date +%s)
python3 eval_kiln.py \
  --adapter "$NAME" \
  --out "judgments/$NAME.json" \
  --concurrency 4 \
  > "$WORKDIR/experiments/$NAME.eval.json"
EVAL_SECS=$(( $(date +%s) - T0 ))

pkill -9 -f "target/release/kiln serve" 2>/dev/null || true
sleep 1

python3 - <<PYEOF
import json, time
agg = json.load(open("judgments/$NAME.json"))
row = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "name": "$NAME",
    "kind": "sft_distill",
    "rank": $RANK,
    "lr": "$LR",
    "epochs": $EPOCHS,
    "seed": $SEED,
    "max_examples": ${MAX_EXAMPLES:-None},
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
print(f"{'$NAME':<28} composite={agg['composite']:.4f} (parses={agg['parses']:.2f} validates={agg['validates']:.2f} is_pure={agg['is_pure']:.2f} is_substantive={agg['is_substantive']:.2f})")
PYEOF
