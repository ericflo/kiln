#!/usr/bin/env bash
set -euo pipefail
cd /workspace/kiln
SLUG="h1-r16-6ep"
DATA="capabilities/opd/tool-call-arg-fidelity/prompts/${SLUG}.jsonl"
OUT_DIR="/tmp/opd-toolcall-${SLUG}"
ADAPTER="toolcall-${SLUG}"
LOG="/tmp/opd-toolcall-iter1.log"
export KILN_STREAMING_PREFILL=1
mkdir -p "$OUT_DIR"
exec ./target/release/examples/cuda_opd_remote \
  --data "$DATA" \
  --model-path /workspace/kiln/Qwen3.5-4B \
  --teacher-url http://localhost:8002 \
  --teacher-model qwen3.6-27b-awq \
  --output-dir "$OUT_DIR" \
  --adapter-name "$ADAPTER" \
  --epochs 6 \
  --rank 16 --alpha 32 --lr 1e-4 \
  --top-k 8 --temperature 1.0 --top-p 0.9 \
  --max-tokens 384 \
  --samples-per-prompt 1 \
  > "$LOG" 2>&1
