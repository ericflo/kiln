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
# Warmup matters a lot: cuBLAS picks a per-(M,N,K) algorithm on the first
# call and caches it, so the first measurement is ~2× a steady-state run.
# A single warmup pass leaves several "still warming up" layers in the
# tail; 4 runs lets the GPU clocks settle and exhausts cuBLAS's algorithm
# search before the timed run.
WARMUP="${KILN_BENCH_WARMUP:-4}"
ITERATIONS="${KILN_BENCH_ITERATIONS:-3}"

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
  # Prefer the precise prefill_time_ms from the trailing JSON dump;
  # fall back to the eprintln line if jq isn't available.
  local prefill_ms
  if command -v jq >/dev/null 2>&1; then
    # The bench prints a trailing JSON object containing
    # `.latency.prefill_time_ms`. Strip everything before the first `{`
    # so tracing logs / banners don't break parsing.
    local json
    json=$(echo "$out" | awk '/^\{/{flag=1} flag{print}')
    prefill_ms=$(echo "$json" | jq -r '.latency.prefill_time_ms // empty' 2>/dev/null | head -1)
  fi
  if [[ -z "${prefill_ms:-}" ]]; then
    # eprintln fallback: "    Prefill: NN.Nms" or "    Prefill (paged): NN.Nms"
    prefill_ms=$(echo "$out" | grep -oE 'Prefill( \(paged\))?: [0-9]+\.[0-9]+ms' | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
  fi
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
  # Average ITERATIONS runs to suppress cuBLAS algo-cache noise / GPU
  # clock-up jitter. Each run already does WARMUP passes internally
  # before the timed prefill.
  legacy_sum=0; fused_sum=0
  for ((i=1; i<=ITERATIONS; i++)); do
    legacy_ms=$(run_one legacy "$seq_len" 1)
    fused_ms=$(run_one fused  "$seq_len" 0)
    legacy_sum=$(awk -v a="$legacy_sum" -v b="$legacy_ms" 'BEGIN{ printf "%.6f", a+b }')
    fused_sum=$(awk -v a="$fused_sum" -v b="$fused_ms" 'BEGIN{ printf "%.6f", a+b }')
  done
  legacy_avg=$(awk -v s="$legacy_sum" -v n="$ITERATIONS" 'BEGIN{ printf "%.3f", s/n }')
  fused_avg=$(awk -v s="$fused_sum" -v n="$ITERATIONS" 'BEGIN{ printf "%.3f", s/n }')
  speedup=$(awk -v a="$legacy_avg" -v b="$fused_avg" 'BEGIN{ if (b>0) printf "%.3fx", a/b; else print "n/a" }')
  printf '%-8s | %14s | %14s | %-10s\n' "$seq_len" "$legacy_avg" "$fused_avg" "$speedup"
done
