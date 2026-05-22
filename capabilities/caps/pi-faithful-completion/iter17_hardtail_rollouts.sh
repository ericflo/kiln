#!/usr/bin/env bash
# Generate deep rollouts on hard tasks only — 16 generations × ~15 hard tasks.
# Then filter to composite >= 0.5 (lower threshold since these are hard).
set -euo pipefail
cd /workspace/kiln/capabilities/caps/pi-faithful-completion
source /root/.kiln-build-env || true

# kiln serve already up (or we'll start it)
export KILN_MODEL_PATH=/workspace/Qwen3.5-4B
export KILN_ADAPTER_DIR=/workspace/adapters

if ! curl -sf http://localhost:8420/v1/health > /dev/null 2>&1; then
  echo "starting kiln serve..."
  nohup /workspace/kiln/target/release/kiln serve --eval-mode > /workspace/iter17-serve.log 2>&1 &
  for i in $(seq 1 120); do
    if curl -sf http://localhost:8420/v1/health > /dev/null 2>&1; then break; fi
    sleep 2
  done
fi

echo "=== generating hard-tail rollouts ==="
rm -rf /tmp/iter17-rollouts
python3 rollout.py \
  --tasks datasets/hard.tasks.jsonl \
  --out-dir /tmp/iter17-rollouts \
  --mode train \
  --num-generations 16 \
  --kiln-base http://localhost:8420 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-tokens 768 \
  --seed 42 \
  --concurrency 4 \
  --system-prompt-file prompts/h15-strict-system-prompt-system.txt \
  --adapter base 2>&1 | tail -10

echo "=== filter hard rollouts at composite > 0.5 ==="
python3 iter5_prep_sft_data.py \
  --rollouts /tmp/iter17-rollouts/rollouts.jsonl \
  --tasks datasets/train.tasks.jsonl \
  --out datasets/hard.sft.jsonl \
  --threshold 0.5 \
  --input-system-prompt default

wc -l datasets/hard.sft.jsonl
touch /workspace/iter17-rollouts.done
