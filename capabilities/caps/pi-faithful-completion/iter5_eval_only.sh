#!/usr/bin/env bash
# iter5 eval only — adapter is already trained, just run 3-arm × 3-seed eval
set -euo pipefail
cd /workspace/kiln/capabilities/caps/pi-faithful-completion
source /root/.kiln-build-env

ADAPTER_NAME="pi-faithful-iter5-sft-strict"

# 1. Restart kiln serve fresh
echo "=== restart kiln serve ==="
export KILN_MODEL_PATH=/workspace/Qwen3.5-4B
export KILN_ADAPTER_DIR=/workspace/adapters
nohup /workspace/kiln/target/release/kiln serve --eval-mode > /workspace/iter5-kiln-serve.log 2>&1 &
echo "pid: $!"
for i in $(seq 1 120); do
  if curl -sf http://localhost:8420/v1/health > /dev/null 2>&1; then
    echo "kiln serve up at iter=$i"; break
  fi
  sleep 2
  if [ "$i" -eq 120 ]; then echo "FATAL: kiln serve down"; tail -50 /workspace/iter5-kiln-serve.log; exit 1; fi
done

echo "=== /v1/adapters ==="
curl -s http://localhost:8420/v1/adapters | python3 -m json.tool || true

# 2. Eval no-prompt 3-seed adapter
echo "=== ADAPTER no-prompt eval ==="
for s in 1 2 3; do
  echo "--- seed $s ---"
  python3 rollout.py --tasks datasets/eval.tasks.jsonl --out-dir /tmp/iter5-eval-adapter-seed-$s \
    --mode eval --num-generations 1 --kiln-base http://localhost:8420 \
    --temperature 0.2 --top-p 0.95 --max-tokens 768 \
    --seed $((100 + s)) --concurrency 4 \
    --adapter "$ADAPTER_NAME" 2>&1 | tail -5
done

# 3. Eval no-prompt 3-seed BASE
echo "=== BASE no-prompt eval ==="
for s in 1 2 3; do
  echo "--- seed $s ---"
  python3 rollout.py --tasks datasets/eval.tasks.jsonl --out-dir /tmp/iter5-eval-base-seed-$s \
    --mode eval --num-generations 1 --kiln-base http://localhost:8420 \
    --temperature 0.2 --top-p 0.95 --max-tokens 768 \
    --seed $((100 + s)) --concurrency 4 \
    --adapter base 2>&1 | tail -5
done

# 4. Eval BASE+strict-prompt 3-seed (ceiling reference)
echo "=== BASE+strict eval ==="
for s in 1 2 3; do
  python3 rollout.py --tasks datasets/eval.tasks.jsonl --out-dir /tmp/iter5-eval-strict-seed-$s \
    --mode eval --num-generations 1 --kiln-base http://localhost:8420 \
    --temperature 0.2 --top-p 0.95 --max-tokens 768 \
    --seed $((100 + s)) --concurrency 4 \
    --system-prompt-file prompts/h15-strict-system-prompt-system.txt \
    --adapter base 2>&1 | tail -3
done

# 5. Summary
echo "=== ITER5 RESULTS ==="
python3 - <<'PY'
import json, statistics
from pathlib import Path

def load(arm, seed):
    p = Path(f"/tmp/iter5-eval-{arm}-seed-{seed}/summary.json")
    return json.loads(p.read_text()) if p.exists() else None

def mean(arm, field="mean_composite"):
    vals = [load(arm, s).get(field) for s in [1,2,3] if load(arm, s)]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2: return (vals[0] if vals else None, 0.0)
    return (statistics.mean(vals), statistics.stdev(vals))

print(f"{'arm':18s} {'mean':>8s} {'stdev':>8s}")
print("-"*40)
for arm, label in [("adapter","ADAPTER (no-prompt)"), ("base","BASE (no-prompt)"), ("strict","BASE + strict-prompt")]:
    m, sd = mean(arm)
    if m is not None:
        print(f"{label:18s} {m:8.4f} {sd:8.4f}")

# Paired lift
print()
print("=== ADAPTER vs BASE paired per-seed lift (the goal metric) ===")
lifts = []
for s in [1,2,3]:
    a = load("adapter", s); b = load("base", s)
    if a and b:
        delta = a["mean_composite"] - b["mean_composite"]
        lifts.append(delta)
        print(f"  seed {s}: adapter={a['mean_composite']:.4f} base={b['mean_composite']:.4f} lift={delta:+.4f}")
if lifts:
    m = statistics.mean(lifts)
    sd = statistics.stdev(lifts) if len(lifts) > 1 else 0.0
    print(f"  PAIRED LIFT: {m:+.4f} ± {sd:.4f}")
    if sd > 0: print(f"  SIGMA: {abs(m)/sd:.1f}σ")

# Sub-scores
print()
print("=== Sub-score means ===")
for arm in ["adapter","base"]:
    print(f"--- {arm} ---")
    for k in ['outcome.value_correct','honesty.score','format_strict.score','terseness.score','no_question.score','no_soft_punt.score']:
        v = [load(arm, s).get("subscore_means",{}).get(k) for s in [1,2,3]]
        v = [x for x in v if x is not None]
        if v: print(f"  {k}: {statistics.mean(v):.4f}")
PY

touch /workspace/iter5-eval.done
echo "EVAL COMPLETE"
