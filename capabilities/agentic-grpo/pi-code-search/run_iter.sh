#!/usr/bin/env bash
# Run one pi-code-search iteration end-to-end.
#
# Usage:
#   ITER=N SLUG=hX-name [other env...] ./run_iter.sh
#
# Required env:
#   ITER       — integer iter number (used for output dir naming).
#   SLUG       — hypothesis slug, e.g. h1-default-recipe.
#
# Optional env (with sensible defaults):
#   TRAIN_TASKS    — datasets/train.tasks.jsonl
#   TRAIN_LIMIT    — N train tasks to use this iter (0 = all)
#   NUM_GEN        — N rollouts per task (default 4)
#   MAX_WALL       — per-rollout wall-clock cap seconds (default 90)
#   LR             — GRPO lr (default 1e-5)
#   RANK / ALPHA   — LoRA rank/alpha (default 16/32)
#   MAX_GROUPS     — train-step cap (default = number of train tasks)
#   EPOCHS         — passes over the data (default 1)
#   ECHO_LAMBDA    — ECHO weight (default 0.05)
#   FILTER_VAR     — drop groups with variance < FILTER_VAR (default 0.0)
#   FILTER_HEAD_FRAC — drop top-FRAC by mean reward (default 0.0 keeps all)
#   SEED           — training seed (default 3141592653)
#   SHUFFLE_SEED   — rollout shuffle seed (default same as SEED)
#   ADAPTER_NAME   — override adapter name (default pi-code-search-iter<N>-<SLUG>)
#   BASE_DIR       — where to write outputs (default /workspace/pi-code-search-iter${ITER})
#   KILN_MODEL_PATH — path to base model (default /workspace/qwen3.5-4b)
#   KILN_URL       — kiln-server endpoint (default http://localhost:8420)
#   SKIP_TRAIN     — if 1, only rollout+eval (used for baseline iters)
#   SKIP_EVAL      — if 1, skip eval step (used for rollout-only iters)
#   EVAL_ONLY      — if 1, only run eval (skip rollouts and training)
#
# This script is designed to be re-invokable: it will not destroy an
# existing $BASE_DIR but it WILL overwrite intermediate logs.
set -euo pipefail

CAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CAP_DIR"

: "${ITER:?ITER required}"
: "${SLUG:?SLUG required}"

TRAIN_TASKS="${TRAIN_TASKS:-datasets/train.tasks.jsonl}"
EVAL_TASKS="${EVAL_TASKS:-datasets/eval.tasks.jsonl}"
TRAIN_LIMIT="${TRAIN_LIMIT:-30}"
NUM_GEN="${NUM_GEN:-4}"
EVAL_NUM_GEN="${EVAL_NUM_GEN:-1}"
MAX_WALL="${MAX_WALL:-120}"
PARALLEL="${PARALLEL:-2}"
EVAL_PARALLEL="${EVAL_PARALLEL:-2}"
LR="${LR:-1e-5}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-32}"
MAX_GROUPS="${MAX_GROUPS:-}"
EPOCHS="${EPOCHS:-1}"
ECHO_LAMBDA="${ECHO_LAMBDA:-0.05}"
FILTER_VAR="${FILTER_VAR:-0.0}"
FILTER_HEAD_FRAC="${FILTER_HEAD_FRAC:-0.0}"
SEED="${SEED:-3141592653}"
SHUFFLE_SEED="${SHUFFLE_SEED:-$SEED}"
ADAPTER_NAME="${ADAPTER_NAME:-pi-code-search-iter${ITER}-${SLUG}}"
BASE_DIR="${BASE_DIR:-/workspace/pi-code-search-iter${ITER}}"
KILN_MODEL_PATH="${KILN_MODEL_PATH:-/workspace/qwen3.5-4b}"
KILN_URL="${KILN_URL:-http://localhost:8420}"
KILN_BIN="${KILN_BIN:-/workspace/kiln/target/release/kiln}"
GRPO_BIN="${GRPO_BIN:-/workspace/kiln/target/release/examples/cuda_grpo_ablation}"

mkdir -p "$BASE_DIR" "$BASE_DIR/logs"

# Persist hyperparams for reproducibility.
cat > "$BASE_DIR/manifest.json" <<EOF
{
  "iter": $ITER,
  "slug": "$SLUG",
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "train_tasks": "$TRAIN_TASKS",
  "eval_tasks": "$EVAL_TASKS",
  "train_limit": $TRAIN_LIMIT,
  "num_gen": $NUM_GEN,
  "eval_num_gen": $EVAL_NUM_GEN,
  "max_wall": $MAX_WALL,
  "lr": "$LR",
  "rank": $RANK,
  "alpha": $ALPHA,
  "epochs": $EPOCHS,
  "echo_lambda": "$ECHO_LAMBDA",
  "filter_var": "$FILTER_VAR",
  "filter_head_frac": "$FILTER_HEAD_FRAC",
  "seed": $SEED,
  "shuffle_seed": $SHUFFLE_SEED,
  "adapter_name": "$ADAPTER_NAME",
  "base_dir": "$BASE_DIR"
}
EOF
echo "manifest: $BASE_DIR/manifest.json"

# Sanity: kiln-server reachable?
if ! curl -sf "$KILN_URL/v1/models" > /dev/null 2>&1; then
  echo "ERROR: kiln-server not reachable at $KILN_URL" >&2
  exit 2
fi

# ----------------------------------------------------------------------------
# Step 1: Rollouts (train mode produces a grpo-train.jsonl)
# ----------------------------------------------------------------------------
ROLLOUT_DIR="$BASE_DIR/rollouts"
mkdir -p "$ROLLOUT_DIR"

if [ "${EVAL_ONLY:-0}" != "1" ] && [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "[iter $ITER] rollout pass: $TRAIN_LIMIT tasks × $NUM_GEN gens"
  # Use base adapter for rollouts (we're sampling from the current model).
  python3 rollout.py \
    --tasks "$TRAIN_TASKS" \
    --out-dir "$ROLLOUT_DIR" \
    --adapter "" \
    --num-generations "$NUM_GEN" \
    --mode train \
    --kiln-url "$KILN_URL" \
    --max-wall-clock-s "$MAX_WALL" \
    --parallel "$PARALLEL" \
    --limit "$TRAIN_LIMIT" \
    --shuffle-seed "$SHUFFLE_SEED" \
    --verbose \
    2>&1 | tee "$BASE_DIR/logs/rollout.log"
fi

# ----------------------------------------------------------------------------
# Step 2: optional filtering of grpo-train.jsonl (variance + head)
# ----------------------------------------------------------------------------
ORIG_GRPO_JSONL="$ROLLOUT_DIR/grpo-train.jsonl"
GRPO_JSONL="$ORIG_GRPO_JSONL"

if [ "${EVAL_ONLY:-0}" != "1" ] && [ -f "$ORIG_GRPO_JSONL" ]; then
  FILT_JSONL="$ROLLOUT_DIR/grpo-train.filtered.jsonl"
  python3 - "$ORIG_GRPO_JSONL" "$FILT_JSONL" "$FILTER_VAR" "$FILTER_HEAD_FRAC" <<'PY'
import json, sys
src, dst, fv_s, fh_s = sys.argv[1:5]
fv = float(fv_s)
fh = float(fh_s)
groups = []
with open(src) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        groups.append(json.loads(line))

def gvar(g):
    rs = [c["reward"] for c in g["completions"]]
    if not rs: return 0.0
    m = sum(rs)/len(rs)
    return sum((r-m)**2 for r in rs)/len(rs)

def gmean(g):
    rs = [c["reward"] for c in g["completions"]]
    return sum(rs)/len(rs) if rs else 0.0

# Variance filter.
if fv > 0:
    groups = [g for g in groups if gvar(g) >= fv]
# Drop top-FRAC by mean reward (the "trivially easy" tasks).
if fh > 0 and groups:
    sg = sorted(groups, key=gmean, reverse=True)
    drop = int(len(sg)*fh)
    keep = sg[drop:]
    groups = keep

with open(dst, "w") as f:
    for g in groups:
        f.write(json.dumps(g)+"\n")
print(f"filtered: {len(groups)} groups → {dst}", flush=True)
PY
  if [ -s "$FILT_JSONL" ]; then
    GRPO_JSONL="$FILT_JSONL"
  fi
fi

# ----------------------------------------------------------------------------
# Step 3: Training (cuda_grpo_ablation)
# ----------------------------------------------------------------------------
ADAPTER_OUT="$BASE_DIR/adapter"

if [ "${EVAL_ONLY:-0}" != "1" ] && [ "${SKIP_TRAIN:-0}" != "1" ] && [ -s "$GRPO_JSONL" ]; then
  echo "[iter $ITER] training adapter: $ADAPTER_NAME"
  # The kiln-server is holding VRAM. Kill it before training, then restart
  # at the end via the caller's outer loop.
  pkill -9 -f "kiln serve" 2>/dev/null || true
  sleep 2

  N_GROUPS_FILE=$(wc -l < "$GRPO_JSONL")
  EFF_MAX_GROUPS="${MAX_GROUPS:-$N_GROUPS_FILE}"

  ECHO_ARG=()
  if [ "${NO_ECHO:-0}" = "1" ]; then
    ECHO_ARG=(--no-echo)
  else
    ECHO_ARG=(--echo-lambda "$ECHO_LAMBDA")
  fi
  # A6000 / H100 workaround: the `kiln_gdn_gates_bf16` kernel and the
  # fused GDN backend both crash in cuda_grpo_ablation's shared-prefix
  # reference forward. Verified-working minimal set of disable flags
  # (see /workspace/test-train.log — 182s on 3 groups, no kernel error):
  export KILN_BATCHING_ENGINE=0
  export KILN_DISABLE_FUSED_GDN_GATES=1
  export KILN_DISABLE_GDN_KERNEL=1
  set -x
  for ep in $(seq 1 "$EPOCHS"); do
    "$GRPO_BIN" \
      --data "$GRPO_JSONL" \
      --model "$KILN_MODEL_PATH" \
      --output "$ADAPTER_OUT" \
      --adapter "$ADAPTER_NAME" \
      --mode phase1 \
      --max-groups "$EFF_MAX_GROUPS" \
      --rank "$RANK" --alpha "$ALPHA" --lr "$LR" \
      --seed "$SEED" \
      "${ECHO_ARG[@]}" \
      2>&1 | tee -a "$BASE_DIR/logs/train.log"
  done
  set +x

  # Symlink the adapter into kiln-server's adapter dir so /v1/adapters/load
  # finds it. (See kiln-polish.jsonl: adapter dir defaults to model_path/adapters/.)
  KILN_ADAPTERS_DIR="$KILN_MODEL_PATH/adapters"
  mkdir -p "$KILN_ADAPTERS_DIR"
  ln -sfn "$ADAPTER_OUT" "$KILN_ADAPTERS_DIR/$ADAPTER_NAME"
fi

# Restart kiln-server if needed (training killed it). Use the same env
# vars that worked on the new H100/A6000 pod (see notes on kernel
# kill-switch flags above).
#
# IMPORTANT: fully detach stdin/stdout/stderr from the launching shell
# so the parent's `tee`/`tail` pipes close cleanly when run_iter.sh
# exits. Without `</dev/null` the parent shell hangs forever waiting
# for the kiln-serve subprocess's stdout to close.
if ! curl -sf "$KILN_URL/v1/models" >/dev/null 2>&1; then
  echo "[iter $ITER] launching kiln serve in background"
  export KILN_MODEL_PATH
  export KILN_SERVED_MODEL_ID=qwen-3.5-4b-kiln
  export KILN_BATCHING_ENGINE=0
  export KILN_DISABLE_FUSED_GDN_GATES=1
  setsid nohup "$KILN_BIN" serve \
        </dev/null >>"$BASE_DIR/logs/kiln-serve.log" 2>&1 &
  disown $! 2>/dev/null || true
  # Wait up to 180s for kiln-server to become reachable.
  for i in $(seq 1 90); do
    if curl -sf "$KILN_URL/v1/models" >/dev/null 2>&1; then break; fi
    sleep 2
  done
  if ! curl -sf "$KILN_URL/v1/models" >/dev/null 2>&1; then
    echo "ERROR: kiln-server failed to start" >&2
    exit 3
  fi
fi

# ----------------------------------------------------------------------------
# Step 4: Eval (blind oracle on held-out eval set)
# ----------------------------------------------------------------------------
if [ "${SKIP_EVAL:-0}" != "1" ]; then
  EVAL_DIR="$BASE_DIR/eval"
  mkdir -p "$EVAL_DIR"

  echo "[iter $ITER] eval pass with adapter '$ADAPTER_NAME'"
  python3 rollout.py \
    --tasks "$EVAL_TASKS" \
    --out-dir "$EVAL_DIR" \
    --adapter "$ADAPTER_NAME" \
    --num-generations "$EVAL_NUM_GEN" \
    --mode eval \
    --kiln-url "$KILN_URL" \
    --max-wall-clock-s "$MAX_WALL" \
    --parallel "$EVAL_PARALLEL" \
    --shuffle-seed "$SHUFFLE_SEED" \
    --verbose \
    2>&1 | tee "$BASE_DIR/logs/eval.log"
fi

echo "[iter $ITER] done. base_dir=$BASE_DIR"
