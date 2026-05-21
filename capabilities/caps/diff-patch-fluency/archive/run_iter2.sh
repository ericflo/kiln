#!/usr/bin/env bash
# Iter 2 — gentle re-attempt after iter 1's catastrophic collapse.
set -euo pipefail
cd /workspace/kiln
SLUG="h1-r8-lr5e5-2ep-spp2"
DATA="capabilities/opd/diff-patch-fluency/prompts/h1-r16-6ep.jsonl"
OUT_DIR="/tmp/opd-diff-${SLUG}"
ADAPTER="diff-${SLUG}"
LOG="/tmp/opd-diff-iter2.log"
export KILN_STREAMING_PREFILL=1
mkdir -p "$OUT_DIR"
exec ./target/release/examples/cuda_opd_remote \
  --data "$DATA" \
  --model-path /workspace/kiln/Qwen3.5-4B \
  --teacher-url http://localhost:8002 \
  --teacher-model qwen3.6-27b-awq \
  --output-dir "$OUT_DIR" \
  --adapter-name "$ADAPTER" \
  --epochs 2 \
  --rank 8 --alpha 16 --lr 5e-5 \
  --top-k 8 --temperature 1.0 --top-p 0.9 \
  --max-tokens 512 \
  --samples-per-prompt 2 \
  --checkpoint-interval 25 \
  > "$LOG" 2>&1
