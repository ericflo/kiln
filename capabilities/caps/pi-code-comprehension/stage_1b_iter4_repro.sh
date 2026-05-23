#!/usr/bin/env bash
# stage_1b_iter4_repro.sh — round-1 winner reproduction under round-3 eval-mode.
#
# Restores iter4 adapter from B2, runs 3-seed paired eval, compares to round-1
# reported +12.93pp. Mirrors the pi-faithful-completion playbook: round-1
# winners often regress under stricter round-3 eval-mode.
set -euo pipefail
cd /workspace/kiln/capabilities/caps/pi-code-comprehension
source /root/.kiln-build-env

ROOT="/workspace/iter0"
ITER4_TGZ="/workspace/iter4-adapter.tgz"
ITER4_DIR="/workspace/adapters/pi-code-comprehension-iter4-h4-echo-0075"
ADAPTER_REGISTRY="${ADAPTER_DIR:-/workspace/adapters}"

# 1. Download iter4 adapter from B2.
echo "=== restoring iter4 adapter from B2 ==="
# B2 creds come from /root/.kiln-build-env (sourced above). Use explicit args
# to b2 account authorize so we never prompt interactively.
if [ -n "${B2_APPLICATION_KEY_ID:-}" ] && [ -n "${B2_APPLICATION_KEY:-}" ]; then
  b2 account authorize "$B2_APPLICATION_KEY_ID" "$B2_APPLICATION_KEY" 2>&1 | head -5
else
  echo "WARN: B2_APPLICATION_KEY_ID / KEY unset; trying cached auth"
  b2 account get 2>&1 | head -5 || true
fi
b2 file download b2://clouderic/kiln/pi-code-comprehension/BEST_ADAPTER_iter4/adapter.tgz "$ITER4_TGZ" 2>&1 | head -10

# 2. Extract to registry.
mkdir -p "$ITER4_DIR"
tar -xzf "$ITER4_TGZ" -C "$ITER4_DIR" 2>&1 | head -5
ls "$ITER4_DIR" 2>&1 | head -10

# 3. Verify adapter via kiln (loadable + behavioral).
echo "=== kiln adapter verify ==="
/workspace/kiln/target/release/kiln adapter verify pi-code-comprehension-iter4-h4-echo-0075 \
  --adapter-dir "$ADAPTER_REGISTRY" \
  --url http://localhost:8420 \
  --json 2>&1 | head -30 || echo "verify-failed-continuing"

# 4. Run 3-seed eval with iter4 loaded.
echo "=== ITER4-ADAPTER 3-seed eval ==="
for s in 1 2 3; do
  outdir="$ROOT/iter4/seed-$s"
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
    --adapter pi-code-comprehension-iter4-h4-echo-0075 \
    --verbose 2>&1 | tail -25 | tee "$outdir/rollout-tail.log"
  jq '.mean_composite, .mean_outcome, .mean_grounding' "$outdir/summary.json" 2>/dev/null || true
done

# 5. Aggregate + paired-lift vs base.
echo "=== aggregating iter4 3-seed paired lift vs base ==="
python3 - "$ROOT" <<'PY'
import json, sys, statistics
from pathlib import Path
root = Path(sys.argv[1])
seeds = [1, 2, 3]

def load_arm(arm):
    out = []
    for s in seeds:
        p = root / arm / f"seed-{s}" / "summary.json"
        if p.exists():
            out.append(json.load(open(p)))
    return out

base = load_arm("base")
iter4 = load_arm("iter4")
if not base or not iter4:
    print(f"missing arm; base={len(base)} iter4={len(iter4)}")
    sys.exit(0)
base_comps = [s["mean_composite"] for s in base]
iter4_comps = [s["mean_composite"] for s in iter4]
paired = [a - b for a, b in zip(iter4_comps, base_comps)]
out = {
    "arm": "iter4 vs base",
    "round_1_reported_lift": 0.1293,
    "base_3seed_mean": statistics.mean(base_comps),
    "base_3seed_stdev": statistics.stdev(base_comps) if len(base_comps) > 1 else 0,
    "iter4_3seed_mean": statistics.mean(iter4_comps),
    "iter4_3seed_stdev": statistics.stdev(iter4_comps) if len(iter4_comps) > 1 else 0,
    "paired_lifts": paired,
    "paired_lift_mean": statistics.mean(paired),
    "paired_lift_stdev": statistics.stdev(paired) if len(paired) > 1 else 0,
    "sigma_above_zero": statistics.mean(paired) / max(statistics.stdev(paired) if len(paired) > 1 else 1e-9, 1e-9),
    "reproduces_round_1": statistics.mean(paired) >= 0.10,
}
out_path = root / "iter4-vs-base-paired.json"
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY

echo DONE > "$ROOT/stage_1b_iter4_repro.done"
