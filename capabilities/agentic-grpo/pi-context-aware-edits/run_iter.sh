#!/usr/bin/env bash
# run_iter.sh — full iter recipe for pi-context-aware-edits (round 2).
#
# Pipeline:
#   0. Build corpus + rubric_sanity gate.
#   1. rollouts via pi
#   2. kiln trajectory inspect
#   3. cuda_grpo_ablation --dry-run
#   4. real training (--filter-var-min --adapter-smoke-test --install-adapter-*)
#   5. kiln adapter verify
#   6. capability.oracle.sh (multi-seed kiln eval-adapter)
#   7. Append iter row from train_receipt.json + eval_summary.json
set -euo pipefail
cd "$(dirname "$0")"

SLUG="${1:-h1-default-recipe}"
ITER_NUM="${ITER_NUM:-1}"
CFG="capability.config.json"
OUT_ROOT="${OUT_ROOT:-/tmp/pi-context-aware-edits-iter-${SLUG}}"
ROLLOUT_DIR="$OUT_ROOT/rollouts"
ADAPTER_NAME="${ADAPTER_NAME:-pi-context-aware-edits-${SLUG}}"
ADAPTER_REGISTRY="${ADAPTER_DIR:-/workspace/adapters}"
LOG_DIR="$OUT_ROOT/logs"
KILN_BIN="${KILN_BIN:-/workspace/kiln/target/release/kiln}"
CUDA_GRPO_BIN="${CUDA_GRPO_BIN:-/workspace/kiln/target/release/examples/cuda_grpo_ablation}"
MODEL_PATH="${MODEL_PATH:-/workspace/qwen3.5-4b}"
TRAIN_LIMIT="${TRAIN_LIMIT:-30}"
NUM_GEN="${NUM_GEN:-4}"
FILTER_VAR_MIN="${FILTER_VAR_MIN:-0.05}"
SEED="${SEED:-3141592653}"
LR="${LR:-1e-5}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-32}"
ECHO_LAMBDA="${ECHO_LAMBDA:-0.05}"
BASE_ADAPTER="${BASE_ADAPTER:-}"

mkdir -p "$ROLLOUT_DIR" "$LOG_DIR"

echo "=== run_iter $SLUG (iter $ITER_NUM) for pi-context-aware-edits ==="

# 0a. Build corpus if missing.
if [ ! -f datasets/train.tasks.jsonl ]; then
  echo "[0a] build_corpus…"
  python3 build_corpus.py
fi

# 0b. Rubric sanity gate (round-2 mandatory — kiln-skill recommendation).
if [ -f rubric_sanity.py ]; then
  echo "[0b] rubric_sanity (calibration gate)…"
  python3 rubric_sanity.py 2>&1 | tee "$LOG_DIR/rubric_sanity.log"
fi

# 1. Gather training rollouts.
echo "[1/7] rollouts (pi)…"
python3 rollout.py \
  --tasks datasets/train.tasks.jsonl \
  --out-dir "$ROLLOUT_DIR" \
  --config "$CFG" \
  --num-generations "$NUM_GEN" \
  --mode train \
  --limit "$TRAIN_LIMIT" \
  2>&1 | tee "$LOG_DIR/rollout.log"

# 2. Trajectory inspector.
echo "[2/7] kiln trajectory inspect…"
"$KILN_BIN" trajectory inspect "$ROLLOUT_DIR/grpo-train.jsonl" --json \
  > "$LOG_DIR/trajectory_inspect.json"

# 3. Dry-run validation.
echo "[3/7] cuda_grpo_ablation --dry-run…"
ECHO_FLAGS="--echo-lambda $ECHO_LAMBDA"
if [ "$ECHO_LAMBDA" = "0" ] || [ "$ECHO_LAMBDA" = "0.0" ]; then
  ECHO_FLAGS="--no-echo"
fi
BASE_FLAGS=""
if [ -n "$BASE_ADAPTER" ]; then
  BASE_FLAGS="--base-adapter $BASE_ADAPTER"
fi

KILN_CUDA_ARCHS="${KILN_CUDA_ARCHS:-86}" "$CUDA_GRPO_BIN" \
  --data "$ROLLOUT_DIR/grpo-train.jsonl" \
  --model "$MODEL_PATH" \
  --output "$OUT_ROOT/adapter" \
  --adapter "$ADAPTER_NAME" \
  --mode phase1 \
  --rank "$RANK" --alpha "$ALPHA" --lr "$LR" \
  --num-generations "$NUM_GEN" \
  --seed "$SEED" \
  --filter-var-min "$FILTER_VAR_MIN" \
  $ECHO_FLAGS $BASE_FLAGS \
  --dry-run \
  2>&1 | tee "$LOG_DIR/dry-run.log"

# 4. Real training.
echo "[4/7] cuda_grpo_ablation (training)…"
KILN_CUDA_ARCHS="${KILN_CUDA_ARCHS:-86}" "$CUDA_GRPO_BIN" \
  --data "$ROLLOUT_DIR/grpo-train.jsonl" \
  --model "$MODEL_PATH" \
  --output "$OUT_ROOT/adapter" \
  --adapter "$ADAPTER_NAME" \
  --mode phase1 \
  --rank "$RANK" --alpha "$ALPHA" --lr "$LR" \
  --num-generations "$NUM_GEN" \
  --seed "$SEED" \
  --filter-var-min "$FILTER_VAR_MIN" \
  $ECHO_FLAGS $BASE_FLAGS \
  --adapter-smoke-test \
  --install-adapter-dir "$ADAPTER_REGISTRY" \
  --install-adapter-name "$ADAPTER_NAME" \
  2>&1 | tee "$LOG_DIR/train.log"

# 5. Verify.
echo "[5/7] kiln adapter verify…"
"$KILN_BIN" adapter verify "$ADAPTER_NAME" \
  --adapter-dir "$ADAPTER_REGISTRY" \
  --url http://localhost:8420 \
  --json \
  > "$LOG_DIR/verify.json"

# 6. Eval.
echo "[6/7] capability.oracle.sh…"
OUT_FILE="$LOG_DIR/eval.json" SEEDS="${EVAL_SEEDS:-3}" \
  ./capability.oracle.sh "$ADAPTER_NAME" \
  2>&1 | tee "$LOG_DIR/eval.log"

# 7. Append row.
echo "[7/7] append capability.jsonl…"
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
    "rubric_version": eval_sum.get("rubric_version"),
    "composite": eval_sum.get("mean_composite"),
    "composite_delta": eval_sum.get("composite_delta"),
    "sub_scores": eval_sum.get("sub_scores_mean"),
    "verdict": eval_sum.get("verdict"),
    "training": {
        "lr": receipt.get("lr"),
        "rank": receipt.get("rank"),
        "alpha": receipt.get("alpha"),
        "seed": receipt.get("seed"),
        "echo_lambda": receipt.get("echo_lambda"),
        "filter_var_min": receipt.get("filter_var_min"),
        "groups_seen": receipt.get("groups_seen"),
        "groups_kept": receipt.get("groups_kept"),
        "wall_clock_s": receipt.get("wall_clock_s"),
        "lora_delta_norm_summary": receipt.get("lora_delta_norm_summary"),
        "echo_metrics": receipt.get("echo_metrics"),
    },
    "kiln_commit": receipt.get("kiln_commit"),
    "adapter_manifest": receipt.get("adapter_manifest_path"),
    "train_receipt": os.environ["TRAIN_RECEIPT"],
    "verify": {"loadable": verify.get("loadable"), "behavioral": verify.get("behavioral")},
    "notes": "",
}
with open("capability.jsonl", "a") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")
print("Appended iter %d %s composite=%s" % (row["iter"], row["slug"], row.get("composite")))
PY

echo "=== iter $SLUG complete ==="
