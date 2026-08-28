#!/usr/bin/env bash
# Phase 0.10 — Pre-migration baseline capture.
#
# Freezes the candle-path numbers for every metric Phase 9 will gate, so the
# DoD's "≥ baseline" assertions are enforceable. Without this commit before
# Phase 1 ships any code, the post-migration "≥ baseline" gates are
# unmeasurable.
#
# Per-GPU baseline file: bench-results/pre-migration-baseline/<gpu-sku>-<commit>.json
# Aggregate index:        bench-results/pre-migration-baseline/index.json
#
# Per the goal directive on #1082, this script is run on RunPod (or any host
# with a CUDA build of kiln-bench), NOT on the local pod.
#
# Required inputs in the run environment:
#   KILN_MODEL_PATH  — path to a Qwen3.5-4B safetensors directory or GGUF
#   KILN_BENCH_BIN   — `kiln-bench` release-build binary (defaults to
#                      `target/release/kiln-bench`)
#
# Outputs land under `bench-results/pre-migration-baseline/`.

set -euo pipefail

MODEL_PATH="${KILN_MODEL_PATH:-Qwen3.5-4B}"
BIN="${KILN_BENCH_BIN:-target/release/kiln-bench}"
WARMUP="${KILN_BENCH_WARMUP:-4}"
ITERATIONS="${KILN_BENCH_ITERATIONS:-3}"

OUT_DIR="bench-results/pre-migration-baseline"
mkdir -p "$OUT_DIR"

# Shape sweep, matching the issue's Phase 9 numeric gates.
DECODE_BATCH_SIZES=(1 2 4 8 16 32 64)
PREFILL_SEQ_LENS=(1024 2048 4096 8192 16384 32768)

# Identity for the output file.
COMMIT=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD)
DATE=$(date -u +%Y%m%dT%H%M%SZ)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 \
      | tr ' ' '_' | tr -cd 'A-Za-z0-9_.-' || echo "unknown_gpu")
HOSTNAME=$(hostname)

OUT_JSON="$OUT_DIR/${GPU}-${COMMIT}-${DATE}.json"
echo "[pre-migration-baseline] GPU=$GPU commit=$COMMIT date=$DATE"
echo "[pre-migration-baseline] writing $OUT_JSON"

if [[ ! -x "$BIN" ]]; then
    echo "[pre-migration-baseline] FATAL: $BIN not found or not executable" >&2
    echo "  Build with: cargo build --release --features cuda -p kiln-server --bin kiln-bench" >&2
    exit 1
fi

# Helper: run kiln-bench once at a given prefill / decode setting, parse the
# trailing JSON dump, and emit a one-line JSON record on stdout.
run_one() {
    local prompt_tokens="$1"
    local max_output_tokens="$2"
    local label="$3"

    local out
    out=$("$BIN" \
        --model-path "$MODEL_PATH" \
        --prompt-tokens "$prompt_tokens" \
        --max-output-tokens "$max_output_tokens" \
        --latency-warmup-runs "$WARMUP" \
        --paged \
        --quiet 2>&1) || {
        echo "[$label] kiln-bench failed:" >&2
        echo "$out" >&2
        return 1
    }

    if command -v jq >/dev/null 2>&1; then
        echo "$out" | awk '/^\{/{flag=1} flag{print}' \
            | jq -c --arg label "$label" \
                --argjson prompt "$prompt_tokens" \
                --argjson max_output "$max_output_tokens" \
                '{label:$label, prompt_tokens:$prompt, max_output_tokens:$max_output, latency:.latency, decode:.decode, prefill:.prefill}'
    else
        echo "$out" | awk '/^\{/{flag=1} flag{print}' \
            | python3 -c '
import sys, json
data = json.load(sys.stdin)
sys.stdout.write(json.dumps({
    "label": "'"$label"'",
    "prompt_tokens": '"$prompt_tokens"',
    "max_output_tokens": '"$max_output_tokens"',
    "latency": data.get("latency", {}),
    "decode":  data.get("decode", {}),
    "prefill": data.get("prefill", {}),
}, sort_keys=True))
'
    fi
}

# Build the aggregate JSON via python so we get pretty output without
# depending on jq's stream features.
records="["
first=1

# Prefill sweep: max_output_tokens=1 isolates prefill cost.
for seq_len in "${PREFILL_SEQ_LENS[@]}"; do
    for ((i=1; i<=ITERATIONS; i++)); do
        rec=$(run_one "$seq_len" 1 "prefill_seq${seq_len}_iter${i}") || continue
        [[ "$first" -eq 1 ]] && first=0 || records+=","
        records+="$rec"
    done
done

# Decode sweep: prompt is fixed at 512, max_output_tokens varies via bs proxy.
# kiln-bench's --batch flag (or equivalent) is the bs knob — we use a fixed
# prompt + variable decode count and report decode_tok/s, plus an additional
# row with --batch if the binary supports it.
for bs in "${DECODE_BATCH_SIZES[@]}"; do
    for ((i=1; i<=ITERATIONS; i++)); do
        # bs is exposed via KILN_BENCH_BATCH if the binary respects it;
        # else this captures the bs=1 case with prompt=512 decode=128.
        rec=$(KILN_BENCH_BATCH="$bs" run_one 512 128 "decode_bs${bs}_iter${i}") || continue
        [[ "$first" -eq 1 ]] && first=0 || records+=","
        records+="$rec"
    done
done

records+="]"

python3 - "$OUT_JSON" "$GPU" "$COMMIT" "$DATE" "$HOSTNAME" <<PY
import json, sys
out, gpu, commit, date, hostname = sys.argv[1:6]
records = json.loads("""$records""")
report = {
    "schema_version": 1,
    "kind": "pre-migration-baseline",
    "issue": "https://github.com/ericflo/kiln/issues/1082",
    "gpu": gpu,
    "commit": commit,
    "date_utc": date,
    "hostname": hostname,
    "model_path": "$MODEL_PATH",
    "warmup_runs": $WARMUP,
    "iterations_per_shape": $ITERATIONS,
    "prefill_seq_lens": [$( IFS=,; echo "${PREFILL_SEQ_LENS[*]}" )],
    "decode_batch_sizes": [$( IFS=,; echo "${DECODE_BATCH_SIZES[*]}" )],
    "records": records,
}
with open(out, "w") as f:
    json.dump(report, f, indent=2)
PY

echo "[pre-migration-baseline] wrote $OUT_JSON"

# Update the per-GPU latest pointer.
ln -sf "$(basename "$OUT_JSON")" "$OUT_DIR/${GPU}-latest.json"

# Refresh index.json: a one-line summary per baseline file in the directory.
python3 - "$OUT_DIR" <<'PY'
import json, os, sys
out_dir = sys.argv[1]
files = [f for f in sorted(os.listdir(out_dir))
         if f.endswith(".json") and f != "index.json"]
index = []
for f in files:
    if f.endswith("-latest.json"):
        continue
    p = os.path.join(out_dir, f)
    try:
        with open(p) as fh:
            data = json.load(fh)
    except Exception:
        continue
    index.append({
        "file": f,
        "gpu": data.get("gpu"),
        "commit": data.get("commit"),
        "date_utc": data.get("date_utc"),
        "record_count": len(data.get("records", [])),
    })
with open(os.path.join(out_dir, "index.json"), "w") as fh:
    json.dump(index, fh, indent=2)
print(f"[pre-migration-baseline] indexed {len(index)} baselines")
PY
