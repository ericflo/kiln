#!/usr/bin/env bash
# run_iter.sh — full OPD iter recipe for transcript-compaction.
#
# Pipeline:
#   0. Build corpus + teacher fixture if missing.
#   1. cuda_opd_remote --dry-run.
#   2. cuda_opd_remote (training) with --install-adapter-dir/-name + smoke test.
#   3. kiln adapter verify.
#   4. capability.oracle.sh (kiln eval-adapter).
#   5. Append to capability.jsonl.
set -euo pipefail
cd "$(dirname "$0")"

SLUG="${1:-h1-r16-6ep}"
ITER_NUM="${ITER_NUM:-1}"
CFG="capability.config.json"
OUT_ROOT="${OUT_ROOT:-/tmp/transcript-compaction-iter-${SLUG}}"
ADAPTER_NAME="${ADAPTER_NAME:-transcript-compaction-${SLUG}}"
ADAPTER_REGISTRY="${ADAPTER_DIR:-/workspace/adapters}"
LOG_DIR="$OUT_ROOT/logs"
KILN_BIN="${KILN_BIN:-/workspace/kiln/target/release/kiln}"
OPD_BIN="${OPD_BIN:-/workspace/kiln/target/release/examples/cuda_opd_remote}"
MODEL_PATH="${MODEL_PATH:-/workspace/qwen3.5-4b}"
TEACHER_URL="${TEACHER_URL:-http://localhost:8002}"
TEACHER_NAME="${TEACHER_NAME:-qwen3.6-27b-awq}"
PROMPTS="${PROMPTS:-prompts/train.jsonl}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-32}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-6}"
SEED="${SEED:-3141592653}"

mkdir -p "$LOG_DIR"

echo "=== OPD run_iter $SLUG (iter $ITER_NUM) for transcript-compaction ==="

if [ ! -f "$PROMPTS" ]; then
  echo "[0/5] build_corpus…"
  python3 build_corpus.py
fi

echo "[1/5] cuda_opd_remote --dry-run…"
KILN_CUDA_ARCHS="${KILN_CUDA_ARCHS:-86}" "$OPD_BIN" \
  --prompts "$PROMPTS" \
  --model "$MODEL_PATH" \
  --teacher-url "$TEACHER_URL" \
  --teacher-name "$TEACHER_NAME" \
  --output "$OUT_ROOT/adapter" \
  --adapter "$ADAPTER_NAME" \
  --rank "$RANK" --alpha "$ALPHA" --lr "$LR" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --dry-run \
  2>&1 | tee "$LOG_DIR/dry-run.log"

echo "[2/5] cuda_opd_remote (training)…"
KILN_CUDA_ARCHS="${KILN_CUDA_ARCHS:-86}" "$OPD_BIN" \
  --prompts "$PROMPTS" \
  --model "$MODEL_PATH" \
  --teacher-url "$TEACHER_URL" \
  --teacher-name "$TEACHER_NAME" \
  --output "$OUT_ROOT/adapter" \
  --adapter "$ADAPTER_NAME" \
  --rank "$RANK" --alpha "$ALPHA" --lr "$LR" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --adapter-smoke-test \
  --install-adapter-dir "$ADAPTER_REGISTRY" \
  --install-adapter-name "$ADAPTER_NAME" \
  2>&1 | tee "$LOG_DIR/train.log"

echo "[3/5] kiln adapter verify…"
"$KILN_BIN" adapter verify "$ADAPTER_NAME" \
  --adapter-dir "$ADAPTER_REGISTRY" \
  --url http://localhost:8420 \
  --json \
  > "$LOG_DIR/verify.json"

echo "[4/5] capability.oracle.sh…"
OUT_FILE="$LOG_DIR/eval.json" SEEDS="${EVAL_SEEDS:-3}" \
  ./capability.oracle.sh "$ADAPTER_NAME" \
  2>&1 | tee "$LOG_DIR/eval.log"

echo "[5/5] append capability.jsonl…"
TRAIN_RECEIPT="$OUT_ROOT/adapter/train_receipt.json" \
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
    "composite": eval_sum.get("mean_composite"),
    "composite_delta": eval_sum.get("composite_delta"),
    "sub_scores": eval_sum.get("sub_scores_mean"),
    "verdict": eval_sum.get("verdict"),
    "training": {
        "lr": receipt.get("lr"),
        "rank": receipt.get("rank"),
        "alpha": receipt.get("alpha"),
        "epochs": receipt.get("epochs"),
        "samples_per_prompt": receipt.get("samples_per_prompt"),
        "n_prompts": receipt.get("n_prompts"),
        "effective_steps": receipt.get("effective_steps"),
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

echo "=== OPD iter $SLUG complete ==="
