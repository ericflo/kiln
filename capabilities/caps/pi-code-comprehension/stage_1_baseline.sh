#!/usr/bin/env bash
# stage_1_baseline.sh — round-3 baseline eval for pi-code-comprehension.
#
# Three concerns in one orchestrated run:
#   1. Round-3 baseline: 3-seed eval, no adapter, default pi prompt.
#   2. Round-1 winner reproduction: 3-seed eval, iter4 adapter restored from B2.
#   3. Optional: prompted-ceiling diagnostic (skipped here; see stage_2_strict.sh).
#
# Output: /workspace/iter0/{base,iter4}/eval-seed-{1,2,3}/summary.json + a top-level
# stage-1-results.json combining the means and computing paired lift + σ.
#
# Assumes:
#   - kiln binaries built under /workspace/kiln/target/release/
#   - /workspace/Qwen3.5-4B base model present
#   - /workspace/adapters registry dir present
#   - kiln-runpod image with /usr/bin/pi, b2, jq, python3 baked in
set -euo pipefail
cd /workspace/kiln/capabilities/caps/pi-code-comprehension

ROOT="/workspace/iter0"
mkdir -p "$ROOT"

source /root/.kiln-build-env

# 1. Restart kiln serve --eval-mode fresh.
echo "=== restarting kiln serve --eval-mode ==="
pkill -f "kiln serve" 2>/dev/null || true
sleep 2
export KILN_MODEL_PATH=/workspace/Qwen3.5-4B
export KILN_ADAPTER_DIR=/workspace/adapters
nohup /workspace/kiln/target/release/kiln serve --eval-mode > "$ROOT/kiln-serve.log" 2>&1 &
KILN_SERVE_PID=$!
echo "  pid: $KILN_SERVE_PID"
for i in $(seq 1 120); do
  if curl -sf http://localhost:8420/v1/health > /dev/null 2>&1; then
    echo "  kiln serve up after ${i}*2s"; break
  fi
  sleep 2
  if [ "$i" -eq 120 ]; then
    echo "FATAL: kiln serve did not come up"
    tail -50 "$ROOT/kiln-serve.log"
    exit 1
  fi
done
echo "=== /v1/health ==="
curl -s http://localhost:8420/v1/health | python3 -m json.tool || true

# 2. Pi setup (wire pi to talk to local kiln server).
echo "=== kiln pi-setup ==="
/workspace/kiln/target/release/kiln pi-setup --url http://localhost:8420 \
  2>&1 | head -20 || true
# Confirm pi sees Qwen3.5-4B
ls -la ~/.pi/agent/models.json 2>&1 | head -3 || true

# 3. Base no-adapter eval, 3 seeds.
echo "=== BASE 3-seed eval ==="
for s in 1 2 3; do
  outdir="$ROOT/base/seed-$s"
  mkdir -p "$outdir"
  echo "--- seed $s -> $outdir ---"
  KILN_URL=http://localhost:8420 PI_BIN=/usr/bin/pi \
  python3 rollout.py \
    --tasks datasets/eval.tasks.jsonl \
    --out-dir "$outdir" \
    --mode eval \
    --num-generations 1 \
    --kiln-url http://localhost:8420 \
    --max-wall-clock-s 180 \
    --concurrency 4 \
    --seed $((100 + s)) \
    --adapter "" \
    --verbose 2>&1 | tail -25 | tee "$outdir/rollout-tail.log"
  echo "--- seed $s summary ---"
  jq '.mean_composite, .mean_outcome, .mean_grounding, .mean_cross_file_caller_recall, .mean_invariant_coverage, .mean_format_compliance' "$outdir/summary.json" 2>/dev/null || true
done

# 4. Aggregate base means.
echo "=== aggregating base 3-seed paired ==="
python3 - "$ROOT" <<'PY'
import json, sys, statistics
from pathlib import Path
root = Path(sys.argv[1])
arm = "base"
seeds = [1, 2, 3]
summaries = []
for s in seeds:
    p = root / arm / f"seed-{s}" / "summary.json"
    if not p.exists():
        print(f"MISSING {p}"); continue
    summaries.append(json.load(open(p)))
if not summaries:
    print("NO SUMMARIES"); sys.exit(0)
comps = [s["mean_composite"] for s in summaries]
out = {
    "arm": arm,
    "n_seeds": len(comps),
    "seeds": seeds[:len(comps)],
    "composites": comps,
    "mean": statistics.mean(comps),
    "stdev": statistics.stdev(comps) if len(comps) > 1 else 0.0,
    "sub_scores_mean": {
        k: statistics.mean([s.get(f"mean_{k}", 0) for s in summaries])
        for k in ["outcome", "grounding", "cross_file_caller_recall",
                  "invariant_coverage", "format_compliance"]
    },
    "n_tasks": summaries[0].get("n_tasks"),
    "rollouts_nonzero_total": sum(s.get("rollouts_nonzero", 0) for s in summaries),
    "rollouts_zero_total": sum(s.get("rollouts_zero", 0) for s in summaries),
}
out_path = root / f"{arm}-3seed.json"
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY

echo "=== stage 1 baseline complete; results in $ROOT ==="
echo DONE > "$ROOT/stage_1_baseline.done"
