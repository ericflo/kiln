#!/bin/bash
# C36 H14a decode-length sweep: Cell D (chat × FP32) × {128, 256, 512, 1024} × seeds {0,1,2}
set -euo pipefail
cd /workspace/kiln
mkdir -p /tmp/c36

export KILN_SPEC_ENABLED=1
export KILN_SPEC_METHOD=mtp
export KILN_W4A16=1
export KILN_CUDA_GRAPHS=true
export KILN_MTP_ARGMAX_FP32=1

START=$(date +%s)
echo "[$(date '+%H:%M:%S')] C36 bench start" | tee -a /tmp/c36.log

for LEN in 128 256 512 1024; do
  for SEED in 0 1 2; do
    OUT=/tmp/c36/c36_L${LEN}_s${SEED}.json
    echo "[$(date '+%H:%M:%S')] LEN=${LEN} SEED=${SEED} -> ${OUT}" | tee -a /tmp/c36.log
    ./target/release/kiln-bench \
      --model-path /workspace/qwen3.5-4b \
      --paged --chat-template \
      --prompt-tokens 512 --max-output-tokens "${LEN}" \
      --skip-training --seed "${SEED}" \
      > "${OUT}" 2>>/tmp/c36.log
    echo "[$(date '+%H:%M:%S')] LEN=${LEN} SEED=${SEED} done ($(stat -c%s "${OUT}") bytes)" | tee -a /tmp/c36.log
  done
done

END=$(date +%s)
echo "[$(date '+%H:%M:%S')] C36 bench complete in $((END-START))s" | tee -a /tmp/c36.log
touch /tmp/c36.done
