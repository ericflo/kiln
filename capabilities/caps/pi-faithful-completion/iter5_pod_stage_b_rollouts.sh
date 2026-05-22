#!/usr/bin/env bash
# iter5 stage B: start kiln server (base, no adapter), generate
# strict-prompt rollouts on train.tasks.jsonl, then write the SFT dataset.
set -euo pipefail

cd /workspace/kiln
source /root/.kiln-build-env

# Kill any prior kiln serve
pkill -f 'target/release/kiln' || true
sleep 2

# Make sure /workspace/adapters exists
mkdir -p /workspace/adapters

# Start kiln serve in eval-mode (no adapter loaded => base)
echo "=== starting kiln serve ==="
nohup ./target/release/kiln serve --eval-mode \
  --model-path /workspace/Qwen3.5-4B \
  --adapter-dir /workspace/adapters \
  > /workspace/iter5-kiln-serve.log 2>&1 &
echo "kiln serve pid: $!"

# Wait for /v1/health
echo "=== waiting for /v1/health ==="
for i in $(seq 1 120); do
  if curl -sf http://localhost:8420/v1/health > /dev/null 2>&1; then
    echo "kiln serve up at iter=$i"
    break
  fi
  sleep 2
  if [ "$i" -eq 120 ]; then
    echo "FATAL: kiln serve did not become healthy"
    tail -100 /workspace/iter5-kiln-serve.log || true
    exit 1
  fi
done

# Confirm base model loaded, no adapter
curl -s http://localhost:8420/v1/adapters | python3 -m json.tool || true

# Generate strict-prompt rollouts on train.tasks.jsonl
# - num_generations=4 gives us up to 4*73=292 candidate completions
# - temperature 0.8 for diversity (above threshold filter selects winners)
cd /workspace/kiln/capabilities/caps/pi-faithful-completion
rm -rf /tmp/iter5-rollouts
mkdir -p /tmp/iter5-rollouts

echo "=== generating strict-prompt rollouts ==="
python3 rollout.py \
  --tasks datasets/train.tasks.jsonl \
  --out-dir /tmp/iter5-rollouts \
  --mode train \
  --num-generations 4 \
  --kiln-base http://localhost:8420 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-tokens 768 \
  --seed 3141592653 \
  --concurrency 4 \
  --system-prompt-file prompts/h15-strict-system-prompt-system.txt \
  --adapter base \
  --verbose 2>&1 | tail -100

ls -la /tmp/iter5-rollouts/

# Now filter rollouts and build sft.train.jsonl
echo "=== filtering rollouts → sft.train.jsonl ==="
python3 iter5_prep_sft_data.py \
  --rollouts /tmp/iter5-rollouts/rollouts.jsonl \
  --tasks datasets/train.tasks.jsonl \
  --out datasets/sft.train.jsonl \
  --threshold 0.7 \
  --input-system-prompt default

echo "=== sft.train.jsonl preview ==="
wc -l datasets/sft.train.jsonl
head -1 datasets/sft.train.jsonl | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
for m in d['messages']:
    print(f'--- {m[\"role\"]} ({len(m[\"content\"])} chars) ---')
    print(m['content'][:300])
"

touch /workspace/iter5-stage-b.done
echo "STAGE B COMPLETE"
