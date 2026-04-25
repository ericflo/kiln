#!/bin/bash
# Phase C37 — variance re-anchor at C35 Cell D (L=128, N=10).
# Single-cell sweep: chat-template × FP32 argmax × prompt_tokens=512 × decode=128.
# Canonical C35 / C36 config with W4A16 Marlin, MTP draft, paged KV, CUDA graphs.
set -u
cd /workspace/kiln
OUTDIR=/workspace/c37-results-v2
mkdir -p $OUTDIR
MODEL_PATH=/workspace/qwen3.5-4b
for seed in 0 1 2 3 4 5 6 7 8 9; do
  OUT=$OUTDIR/seed-$seed.json
  LOG=$OUTDIR/seed-$seed.log
  echo "=== seed=$seed === $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a $OUTDIR/progress.log
  KILN_W4A16=1 \
  KILN_SPEC_ENABLED=1 \
  KILN_SPEC_METHOD=mtp \
  KILN_MTP_ARGMAX_FP32=1 \
  KILN_CUDA_GRAPHS=true \
    /workspace/kiln/target/release/kiln-bench \
      --model-path "$MODEL_PATH" \
      --paged \
      --chat-template \
      --skip-training \
      --prompt-tokens 512 \
      --max-output-tokens 128 \
      --seed $seed \
      > $OUT 2> $LOG
  rc=$?
  echo "  rc=$rc size=$(stat -c %s $OUT 2>/dev/null || echo 0)" | tee -a $OUTDIR/progress.log
  if [ $rc -ne 0 ]; then
    echo "SEED $seed FAILED — see $LOG"
    tail -20 $LOG | tee -a $OUTDIR/progress.log
  fi
done
touch $OUTDIR/done.sentinel
echo "=== ALL DONE === $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a $OUTDIR/progress.log
