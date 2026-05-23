#!/usr/bin/env bash
# A/B bench for the fused MLP gate+up prefill GEMM (PR landing the
# `gate_up_proj_t` cache on GpuFfnWeights). Runs kiln-bench at four prefill
# seq_len points with the kill switch on (legacy two-matmul) and off (one
# concatenated GEMM), then prints prefill_time_ms side-by-side so the
# per-layer gate+up savings show up at the model level.
#
# Expected: per-layer gate+up time drops from ~6.3 ms toward the ~2.5 ms
# compute roof. At 32 layers the model-level prefill_time_ms swing should
# track 32 × per-layer-delta scaled by the share of prefill spent in MLP
# gate+up.
#
# Usage:
#   scripts/bench-fused-mlp-gate-up-prefill.sh [--model-path PATH]
#
# Required: a CUDA build of kiln-bench. Build with:
#   PATH=/usr/local/cuda-12.4/bin:$PATH \
#     cargo build --release --features cuda -p kiln-server --bin kiln-bench
set -euo pipefail

MODEL_PATH="${KILN_MODEL_PATH:-Qwen3.5-4B}"
BIN="${KILN_BENCH_BIN:-target/release/kiln-bench}"
SEEDS=("${KILN_BENCH_SEED:-42}")
SEQ_LENS=(1024 2048 4096 8192)
MAX_OUT="${KILN_BENCH_MAX_OUTPUT:-1}"
WARMUP="${KILN_BENCH_WARMUP:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"; shift 2 ;;
    --bin)
      BIN="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,30p' "$0" ; exit 0 ;;
    *)
      echo "unknown arg: $1" >&2 ; exit 1 ;;
  esac
done

if [[ ! -x "$BIN" ]]; then
  echo "kiln-bench not found at $BIN" >&2
  echo "Build it with:" >&2
  echo "  PATH=/usr/local/cuda-12.4/bin:\$PATH cargo build --release --features cuda -p kiln-server --bin kiln-bench" >&2
  exit 1
fi

run_one() {
  local label="$1"
  local seq_len="$2"
  local disable="$3"
  local env_args=()
  if [[ "$disable" == "1" ]]; then
    env_args+=("KILN_DISABLE_FUSED_MLP_GATE_UP_PREFILL=1")
  fi
  # JSON is not emitted by default — parse the human prefill summary line.
  local out
  out=$(env "${env_args[@]}" "$BIN" \
          --model-path "$MODEL_PATH" \
          --prompt-tokens "$seq_len" \
          --max-output-tokens "$MAX_OUT" \
          --latency-only \
          --latency-warmup-runs "$WARMUP" \
          --paged \
          --quiet 2>&1) || {
    echo "[$label seq=$seq_len] kiln-bench failed; output:" >&2
    echo "$out" >&2
    return 1
  }
  # Match "    Prefill: NN.Nms (NN tok/s)" emitted by the latency path.
  local prefill_ms
  prefill_ms=$(echo "$out" | grep -oE 'Prefill: [0-9]+\.[0-9]+ms' | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
  if [[ -z "$prefill_ms" ]]; then
    echo "[$label seq=$seq_len] could not parse prefill_time_ms" >&2
    echo "$out" | tail -20 >&2
    return 1
  fi
  echo "$prefill_ms"
}

printf '%-8s | %-14s | %-14s | %-10s\n' 'seq_len' 'legacy_ms' 'fused_ms' 'speedup'
printf -- '---------+----------------+----------------+-----------\n'
for seq_len in "${SEQ_LENS[@]}"; do
  legacy_ms=$(run_one legacy "$seq_len" 1)
  fused_ms=$(run_one fused  "$seq_len" 0)
  speedup=$(awk -v a="$legacy_ms" -v b="$fused_ms" 'BEGIN{ if (b>0) printf "%.3fx", a/b; else print "n/a" }')
  printf '%-8s | %14s | %14s | %-10s\n' "$seq_len" "$legacy_ms" "$fused_ms" "$speedup"
done
