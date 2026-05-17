#!/usr/bin/env bash
# Run the OPD sweep on the SHORT-prompt subset (sequences ≤ 400 tokens), which
# is what fits the un-checkpointed cuda_opd_from_fixture binary on a 48GB GPU.
set -euo pipefail
cd /workspace/kiln/sft-cap.json-schema-adherence

# Build short subset.
python3 filter_short.py --max-tokens 400

# Each run uses run_exp.sh which calls cuda_opd_from_fixture.
# Modify run_exp.sh to point at the short data files first.
SHORT_PROMPTS="datasets/train.opd.short.jsonl"
SHORT_FIXTURE="datasets/teacher.fixture.short.jsonl"

run_opd() {
    local NAME="$1"; shift
    local RANK="${1:?}"; shift
    local LR="${1:?}"; shift
    local TOPK="${1:?}"; shift
    local SPP="${1:-1}"
    local SEED="4218"

    local ADAPTER_DIR="/workspace/kiln/Qwen3.5-4B/adapters"
    local MODEL_PATH="/workspace/kiln/Qwen3.5-4B"
    local KILN_BIN="/workspace/kiln/target/release/kiln"
    local EXAMPLE_BIN="/workspace/kiln/target/release/examples/cuda_opd_from_fixture"
    local EXP_LOG="/workspace/kiln/sft-cap.json-schema-adherence/experiments/$NAME.log"
    mkdir -p "$(dirname "$EXP_LOG")"

    echo "=== opd $NAME rank=$RANK lr=$LR top_k=$TOPK spp=$SPP ==="

    pkill -9 -f "target/release/kiln serve" 2>/dev/null || true
    sleep 1

    T0=$(date +%s)
    "$EXAMPLE_BIN" \
      --model-path "$MODEL_PATH" \
      --prompts "$SHORT_PROMPTS" \
      --teacher-fixture "$SHORT_FIXTURE" \
      --output-dir "$ADAPTER_DIR" \
      --adapter-name "$NAME" \
      --top-k "$TOPK" \
      --rank "$RANK" \
      --lr "$LR" \
      --samples-per-prompt "$SPP" \
      --seed "$SEED" \
      2>&1 | tee "$EXP_LOG" | tail -5
    TRAIN_SECS=$(( $(date +%s) - T0 ))
    echo "train: ${TRAIN_SECS}s"

    nohup "$KILN_BIN" serve --config "/workspace/kiln/sft-cap.json-schema-adherence/kiln.toml" \
      > "/workspace/kiln/sft-cap.json-schema-adherence/kiln-server.log" 2>&1 &
    for i in $(seq 1 40); do
      if curl -s -m 2 http://localhost:8420/v1/models 2>/dev/null | grep -q "qwen3.5-4b"; then break; fi
      sleep 1
    done

    T0=$(date +%s)
    python3 eval_kiln.py --adapter "$NAME" --out "judgments/$NAME.json" --concurrency 4 \
      > "/workspace/kiln/sft-cap.json-schema-adherence/experiments/$NAME.eval.json"
    EVAL_SECS=$(( $(date +%s) - T0 ))

    pkill -9 -f "target/release/kiln serve" 2>/dev/null || true
    sleep 1

    python3 - <<PY
import json, time
agg = json.load(open("judgments/$NAME.json"))
row = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "name": "$NAME",
    "kind": "opd",
    "rank": $RANK, "lr": "$LR", "top_k": $TOPK,
    "samples_per_prompt": $SPP, "seed": $SEED,
    "train_secs": $TRAIN_SECS, "eval_secs": $EVAL_SECS,
    "n_eval": agg["n"],
    "parses": agg["parses"], "validates": agg["validates"],
    "is_pure": agg["is_pure"], "is_substantive": agg["is_substantive"],
    "composite": agg["composite"],
    "notes": "OPD on short subset",
}
with open("capability.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"$NAME composite={agg['composite']:.4f}")
PY
}

run_opd opd-v1-r16-lr1e5-spp1 16 1e-5 32 1
run_opd opd-v2-r32-lr1e5-spp1 32 1e-5 32 1
run_opd opd-v3-r16-lr5e5-spp1 16 5e-5 32 1
run_opd opd-v4-r16-lr1e5-spp4 16 1e-5 32 4

echo "=== OPD sweep complete ==="
python3 summarize.py
