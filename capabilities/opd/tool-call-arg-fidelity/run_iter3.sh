#!/usr/bin/env bash
# Iter 3 — H9 asymmetric, same prompts + hyperparams as iter 2, BUT with
# checkpointing now wired in the kiln OPD trainer. Default interval = 25.
# If killed mid-run, the latest checkpoint dir is usable as the adapter.
set -euo pipefail
cd /workspace/kiln
SLUG="h9-asym-ckpt"
DATA="capabilities/opd/tool-call-arg-fidelity/prompts/h9-asym.jsonl"
OUT_DIR="/tmp/opd-toolcall-${SLUG}"
ADAPTER="toolcall-${SLUG}"
LOG="/tmp/opd-toolcall-iter3.log"
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
  --checkpoint-interval 25 \
  > "$LOG" 2>&1
