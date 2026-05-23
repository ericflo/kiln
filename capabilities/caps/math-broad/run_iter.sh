#!/usr/bin/env bash
# run_iter.sh — full SFT iter recipe for math-broad.
set -euo pipefail
cd "$(dirname "$0")"

SLUG="${1:-h1-default}"
ITER_NUM="${ITER_NUM:-1}"
ADAPTER_NAME="${ADAPTER_NAME:-math-broad-${SLUG}}"
OUT_ROOT="${OUT_ROOT:-/tmp/math-broad-iter-${SLUG}}"
ADAPTER_REGISTRY="${ADAPTER_DIR:-/workspace/adapters}"
LOG_DIR="$OUT_ROOT/logs"
KILN_BIN="${KILN_BIN:-/workspace/kiln/target/release/kiln}"
SFT_BIN="${SFT_BIN:-/workspace/kiln/target/release/examples/cuda_sft_file}"
MODEL_PATH="${MODEL_PATH:-/workspace/Qwen3.5-4B}"
DATA="${DATA:-datasets/train.jsonl}"
RANK="${RANK:-4}"
ALPHA="${ALPHA:-8}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-3141592653}"
DATASET_CAP="${DATASET_CAP:-128}"

mkdir -p "$LOG_DIR"

echo "=== SFT run_iter $SLUG (iter $ITER_NUM) for math-broad ==="

if [ ! -f "$DATA" ]; then
  echo "[0/6] build_corpus…"
  python3 build_corpus.py
fi

# 0b. Rubric sanity gate (round-2 mandatory).
#     Run rubric_sanity.py if it exists. Bypass with KILN_SKIP_RUBRIC_SANITY=1.
if [ -f rubric_sanity.py ] && [ -z "${KILN_SKIP_RUBRIC_SANITY:-}" ]; then
  echo "[0b] rubric_sanity (calibration gate)…"
  python3 rubric_sanity.py 2>&1 | tee "$LOG_DIR/rubric_sanity.log" || {
    echo "rubric_sanity failed; set KILN_SKIP_RUBRIC_SANITY=1 to bypass" >&2
    exit 4
  }
fi

# NOTE on cuda_sft_file CLI (verified 2026-05-23):
# - Flags: --data, --model-path, --output-dir, --adapter-name, --rank, --alpha,
#   --lr, --epochs, --max-examples, --skip-examples, --base-adapter,
#   --allow-adapter-shape-conversion, --allow-high-lora-scale, --trainer.
# - NO --seed, --dry-run, --adapter-smoke-test, --install-adapter-{dir,name}.
# - Output: if basename(--output-dir) == --adapter-name, lands flat at output-dir/.
# - VRAM: kiln serve must be killed first; long seqs (>500 tok) need KILN_CUDA_RECOMPUTE_SFT=1.
echo "[1/4] kill kiln serve to free VRAM (cuda_sft_file shares the GPU)…"
pkill -f "kiln serve" 2>/dev/null || true
sleep 2

echo "[2/4] cuda_sft_file (training)…"
KILN_CUDA_ARCHS="${KILN_CUDA_ARCHS:-86}" \
KILN_CUDA_RECOMPUTE_SFT="${KILN_CUDA_RECOMPUTE_SFT:-1}" \
"$SFT_BIN" \
  --data "$DATA" \
  --model-path "$MODEL_PATH" \
  --output-dir "$ADAPTER_REGISTRY/$ADAPTER_NAME" \
  --adapter-name "$ADAPTER_NAME" \
  --rank "$RANK" --alpha "$ALPHA" --lr "$LR" \
  --epochs "$EPOCHS" \
  --max-examples "$DATASET_CAP" \
  --trainer native \
  2>&1 | tee "$LOG_DIR/train.log"

echo "[3/4] restart kiln serve --eval-mode for adapter verify + eval…"
KILN_MODEL_PATH="$MODEL_PATH" KILN_ADAPTER_DIR="$ADAPTER_REGISTRY" \
  nohup "$KILN_BIN" serve --eval-mode > "$LOG_DIR/kiln-serve.log" 2>&1 &
for i in $(seq 1 60); do
  curl -sf http://localhost:8420/v1/health > /dev/null 2>&1 && break
  sleep 1
done

echo "[4/4] kiln adapter verify…"
"$KILN_BIN" adapter verify "$ADAPTER_NAME" \
  --adapter-dir "$ADAPTER_REGISTRY" \
  --url http://localhost:8420 \
  --json \
  > "$LOG_DIR/verify.json"

echo "[5/6] capability.oracle.sh…"
OUT_FILE="$LOG_DIR/eval.json" SEEDS="${EVAL_SEEDS:-3}" \
  ./capability.oracle.sh "$ADAPTER_NAME" \
  2>&1 | tee "$LOG_DIR/eval.log"

if [ -x "./capability.anchor.sh" ]; then
  echo "[6/6] capability.anchor.sh (regression watch)…"
  ./capability.anchor.sh "$ADAPTER_NAME" \
    2>&1 | tee "$LOG_DIR/anchor.log"
fi

TRAIN_RECEIPT="$ADAPTER_REGISTRY/$ADAPTER_NAME/train_receipt.json" \
EVAL_JSON="$LOG_DIR/eval.json" \
VERIFY_JSON="$LOG_DIR/verify.json" \
ITER_NUM="$ITER_NUM" \
SLUG="$SLUG" \
python3 - <<'PY'
import json, os, time
receipt = json.load(open(os.environ["TRAIN_RECEIPT"]))
eval_sum = json.load(open(os.environ["EVAL_JSON"]))
verify = json.load(open(os.environ["VERIFY_JSON"]))
slug = os.environ["SLUG"]
iter_num = int(os.environ["ITER_NUM"])
row = {
    "iter": iter_num,
    "slug": slug,
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": eval_sum.get("status", "kept-with-caveat"),
    "family": slug.split("-", 1)[0].upper(),
    "composite": eval_sum.get("mean_composite") or eval_sum.get("accuracy"),
    "composite_delta": eval_sum.get("composite_delta"),
    "sub_scores": eval_sum.get("sub_scores_mean"),
    "verdict": eval_sum.get("verdict"),
    "training": {
        "lr": receipt.get("lr"),
        "rank": receipt.get("rank"),
        "alpha": receipt.get("alpha"),
        "epochs": receipt.get("epochs"),
        "n_examples": receipt.get("n_examples"),
        "wall_clock_s": receipt.get("wall_clock_s"),
    },
    "kiln_commit": receipt.get("kiln_commit"),
    "train_receipt": os.environ["TRAIN_RECEIPT"],
    "verify": {"loadable": verify.get("loadable"), "behavioral": verify.get("behavioral")},
    "notes": "",
}
with open("capability.jsonl", "a") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")
print("Appended iter %d %s composite=%s" % (row["iter"], row["slug"], row.get("composite")))
PY

echo "=== SFT iter $SLUG complete ==="
