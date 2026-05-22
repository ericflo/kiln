#!/usr/bin/env bash
# iter5 stage C: train SFT adapter, eval no-prompt 3-seed paired
set -euo pipefail

cd /workspace/kiln
source /root/.kiln-build-env

ADAPTER_NAME="pi-faithful-iter5-sft-strict"
OUT_ROOT="/tmp/iter5-sft-out"
rm -rf "$OUT_ROOT" && mkdir -p "$OUT_ROOT"

cd /workspace/kiln/capabilities/caps/pi-faithful-completion

# 1. Train SFT
echo "=== cuda_sft_file (SFT training) ==="
KILN_CUDA_ARCHS=86 /workspace/kiln/target/release/cuda_sft_file \
  --data datasets/sft.train.jsonl \
  --model /workspace/Qwen3.5-4B \
  --output "$OUT_ROOT/adapter" \
  --adapter "$ADAPTER_NAME" \
  --rank 4 --alpha 8 --lr 1e-4 --epochs 1 \
  --dataset-cap 128 \
  --seed 3141592653 \
  --adapter-smoke-test \
  --install-adapter-dir /workspace/adapters \
  --install-adapter-name "$ADAPTER_NAME" 2>&1 | tail -50

# 2. Verify adapter loads + has behavioral effect
echo "=== kiln adapter verify ==="
/workspace/kiln/target/release/kiln adapter verify "$ADAPTER_NAME" \
  --adapter-dir /workspace/adapters \
  --url http://localhost:8420 2>&1 | tail -20

# 3. Eval no-prompt 3-seed paired (the goal: lift no-prompt composite)
echo "=== eval-adapter no-prompt 3-seed ==="
/workspace/kiln/target/release/kiln eval-adapter \
  --url http://localhost:8420 \
  --adapter "$ADAPTER_NAME" \
  --adapter-dir /workspace/adapters \
  --tasks datasets/eval.tasks.jsonl \
  --seeds 3 \
  --scorer ./rubric.py \
  --output /tmp/iter5-eval-no-prompt.json \
  --thinking off 2>&1 | tail -40

# 4. Eval no-prompt 3-seed BASE for paired baseline
echo "=== eval-adapter no-prompt 3-seed (BASE) ==="
/workspace/kiln/target/release/kiln eval-adapter \
  --url http://localhost:8420 \
  --adapter base \
  --adapter-dir /workspace/adapters \
  --tasks datasets/eval.tasks.jsonl \
  --seeds 3 \
  --scorer ./rubric.py \
  --output /tmp/iter5-eval-base-no-prompt.json \
  --thinking off 2>&1 | tail -20

# 5. ALSO eval with strict prompt to confirm the prompt+adapter still works
# (if the SFT adapter destroyed the strict-prompt distribution, we know the LoRA
# is destructive and we need to back off rank/lr)
echo "=== eval-adapter WITH strict prompt 3-seed (ceiling reference) ==="
# kiln eval-adapter doesn't support system_prompt override directly — use rollout.py
for seed in 1 2 3; do
  echo "--- seed $seed strict-prompt eval ---"
  python3 rollout.py \
    --tasks datasets/eval.tasks.jsonl \
    --out-dir /tmp/iter5-eval-strict-seed-$seed \
    --mode eval \
    --num-generations 1 \
    --kiln-base http://localhost:8420 \
    --temperature 0.2 --top-p 0.95 --max-tokens 768 \
    --seed $((3141592653 + seed)) \
    --concurrency 4 \
    --system-prompt-file prompts/h15-strict-system-prompt-system.txt \
    --adapter "$ADAPTER_NAME" 2>&1 | tail -10
done

# Compute mean
echo "=== iter5 results summary ==="
python3 - <<'PY'
import json, statistics
from pathlib import Path

print("\n=== ADAPTER no-prompt 3-seed ===")
try:
    es = json.loads(Path("/tmp/iter5-eval-no-prompt.json").read_text())
    print(json.dumps(es, indent=2, default=str)[:1500])
except Exception as e:
    print(f"ERROR loading adapter eval: {e}")

print("\n=== BASE no-prompt 3-seed ===")
try:
    es = json.loads(Path("/tmp/iter5-eval-base-no-prompt.json").read_text())
    print(json.dumps(es, indent=2, default=str)[:1500])
except Exception as e:
    print(f"ERROR loading base eval: {e}")

print("\n=== ADAPTER + strict prompt 3-seed (mean composite) ===")
seeds_means = []
for s in [1, 2, 3]:
    f = Path(f"/tmp/iter5-eval-strict-seed-{s}/summary.json")
    if f.exists():
        d = json.loads(f.read_text())
        seeds_means.append(d.get("mean_composite", 0.0))
        print(f"  seed {s}: {d.get('mean_composite', 0.0):.4f}")
if seeds_means:
    m = statistics.mean(seeds_means)
    sd = statistics.stdev(seeds_means) if len(seeds_means) > 1 else 0.0
    print(f"  mean = {m:.4f}, stdev = {sd:.4f}")
PY

touch /workspace/iter5-stage-c.done
echo "STAGE C COMPLETE"
