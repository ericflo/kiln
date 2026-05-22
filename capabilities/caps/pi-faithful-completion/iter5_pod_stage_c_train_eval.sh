#!/usr/bin/env bash
# iter5 stage C: train SFT adapter, eval no-prompt 3-seed paired (base vs adapter)
set -euo pipefail

cd /workspace/kiln
source /root/.kiln-build-env

ADAPTER_NAME="pi-faithful-iter5-sft-strict"
ADAPTER_OUT="/workspace/adapters/${ADAPTER_NAME}"
mkdir -p /workspace/adapters
rm -rf "$ADAPTER_OUT"

CAPS_DIR=/workspace/kiln/capabilities/caps/pi-faithful-completion
cd "$CAPS_DIR"

# 1. Verify the SFT dataset is there
echo "=== sft.train.jsonl check ==="
ls -la datasets/sft.train.jsonl
wc -l datasets/sft.train.jsonl

# 1b. Kill any running kiln serve to free VRAM for training (we'll restart it after)
echo "=== killing existing kiln serve to free VRAM ==="
pkill -f 'target/release/kiln serve' 2>/dev/null || true
sleep 10
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# 2. Train SFT (cuda_sft_file)
#    Defaults: rank=8 alpha=16 lr=1e-4 epochs=1. Per METHODS.md §3.1 the canonical
#    SFT recipe is rank=4 alpha=8 lr=1e-4 epochs=1 dataset_cap=128 — let's match that.
echo "=== cuda_sft_file (SFT training) ==="
KILN_CUDA_ARCHS=86 /workspace/kiln/target/release/examples/cuda_sft_file \
  --data "$CAPS_DIR/datasets/sft.train.jsonl" \
  --model-path /workspace/Qwen3.5-4B \
  --output-dir "$ADAPTER_OUT" \
  --adapter-name "$ADAPTER_NAME" \
  --rank 4 --alpha 8 --lr 1e-4 \
  --epochs 1 \
  --max-examples 128 2>&1 | tee /tmp/iter5-sft-train.log | tail -80

echo "=== adapter output ==="
ls -la "$ADAPTER_OUT"

# 3. (Re)start kiln serve so it picks up the new adapter
echo "=== restart kiln serve ==="
pkill -f 'target/release/kiln serve' 2>/dev/null || true
sleep 3
export KILN_MODEL_PATH=/workspace/Qwen3.5-4B
export KILN_ADAPTER_DIR=/workspace/adapters
export KILN_DISABLE_FUSED_GDN_GATES=1
nohup /workspace/kiln/target/release/kiln serve --eval-mode \
  > /workspace/iter5-kiln-serve.log 2>&1 &
echo "kiln serve pid: $!"
for i in $(seq 1 120); do
  if curl -sf http://localhost:8420/v1/health > /dev/null 2>&1; then
    echo "kiln serve up at iter=$i"
    break
  fi
  sleep 2
  if [ "$i" -eq 120 ]; then
    echo "FATAL: kiln serve did not become healthy"; tail -100 /workspace/iter5-kiln-serve.log
    exit 1
  fi
done

# Confirm adapter is registered
curl -s http://localhost:8420/v1/adapters | python3 -m json.tool || true

# 4. Eval no-prompt 3-seed adapter
echo "=== ADAPTER no-prompt eval (3 seeds, paired) ==="
for s in 1 2 3; do
  echo "  --- seed $s ---"
  python3 rollout.py \
    --tasks datasets/eval.tasks.jsonl \
    --out-dir /tmp/iter5-eval-adapter-seed-$s \
    --mode eval \
    --num-generations 1 \
    --kiln-base http://localhost:8420 \
    --temperature 0.2 --top-p 0.95 --max-tokens 768 \
    --seed $((100 + s)) \
    --concurrency 4 \
    --adapter "$ADAPTER_NAME" 2>&1 | tail -8
done

# 5. Eval no-prompt 3-seed BASE
echo "=== BASE no-prompt eval (3 seeds, paired) ==="
for s in 1 2 3; do
  echo "  --- seed $s ---"
  python3 rollout.py \
    --tasks datasets/eval.tasks.jsonl \
    --out-dir /tmp/iter5-eval-base-seed-$s \
    --mode eval \
    --num-generations 1 \
    --kiln-base http://localhost:8420 \
    --temperature 0.2 --top-p 0.95 --max-tokens 768 \
    --seed $((100 + s)) \
    --concurrency 4 \
    --adapter base 2>&1 | tail -8
done

# 6. ALSO eval base + strict prompt (ceiling reference, paired)
echo "=== BASE+strict-prompt eval (3 seeds, paired) ==="
for s in 1 2 3; do
  python3 rollout.py \
    --tasks datasets/eval.tasks.jsonl \
    --out-dir /tmp/iter5-eval-strict-seed-$s \
    --mode eval \
    --num-generations 1 \
    --kiln-base http://localhost:8420 \
    --temperature 0.2 --top-p 0.95 --max-tokens 768 \
    --seed $((100 + s)) \
    --concurrency 4 \
    --system-prompt-file prompts/h15-strict-system-prompt-system.txt \
    --adapter base 2>&1 | tail -5
done

# 7. Compute paired summary
echo "=== iter5 results summary ==="
python3 - <<'PY'
import json, statistics
from pathlib import Path

def load(arm, seed):
    p = Path(f"/tmp/iter5-eval-{arm}-seed-{seed}/summary.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())

def mean(field, arm):
    vals = [load(arm, s).get(field) for s in [1,2,3] if load(arm, s)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) < 2:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)

arms = ["adapter", "base", "strict"]
print(f"{'arm':10s} {'mean':>8s} {'stdev':>8s}")
results = {}
for arm in arms:
    m, sd = mean("mean_composite", arm)
    if m is not None:
        print(f"{arm:10s} {m:8.4f} {sd:8.4f}")
        results[arm] = (m, sd)
    else:
        print(f"{arm:10s} MISSING")

# Paired per-seed lift
print()
print("=== ADAPTER vs BASE paired per-seed lift (no-prompt) ===")
lifts = []
for s in [1,2,3]:
    a = load("adapter", s)
    b = load("base", s)
    if a and b:
        delta = a["mean_composite"] - b["mean_composite"]
        lifts.append(delta)
        print(f"  seed {s}: adapter={a['mean_composite']:.4f} base={b['mean_composite']:.4f} lift={delta:+.4f}")
if lifts:
    m = statistics.mean(lifts)
    sd = statistics.stdev(lifts) if len(lifts) > 1 else 0.0
    print(f"  PAIRED LIFT: {m:+.4f} ± {sd:.4f}")
    if sd > 0:
        print(f"  SIGMA: {abs(m)/sd:.1f}σ")

# Show subscores for adapter vs base
print()
print("=== Sub-score means (adapter no-prompt) ===")
a = load("adapter", 1)
if a:
    for k in ['outcome.value_correct', 'honesty.score', 'format_strict.score', 'terseness.score',
              'no_question.score', 'no_soft_punt.score']:
        seeds_means = [load("adapter", s).get("subscore_means",{}).get(k) for s in [1,2,3]]
        seeds_means = [v for v in seeds_means if v is not None]
        if seeds_means:
            print(f"  {k}: {statistics.mean(seeds_means):.4f}")
PY

touch /workspace/iter5-stage-c.done
echo "STAGE C COMPLETE"
