#!/usr/bin/env bash
# stage_2_sft_ideal.sh — SFT on synthesized ideal-from-gold completions.
#
# Per pi-faithful-completion iter18 pattern: every train task has perfect
# gold; the "ideal" completion is `<answer>\n{json.dumps(gold)}\n</answer>`.
# SFT teaches the model the OUTPUT shape (exact line citations + implicit
# invariants + all 7 fields). At inference, multi-turn pi still works —
# tool-use comes from base, output shape from SFT.
#
# Targets: grounding (currently 0.48), invariant_coverage (currently 0.37),
# outcome.invariants (currently low). Cross-file recall already at 0.94.
#
# Recipe (conservative):
#   rank=4, alpha=8 (low — start small per METHODS.md §3.1)
#   lr=1e-4, 1 epoch
#   data: 188 train tasks → 188 SFT rows (no filtering)
set -euo pipefail
cd /workspace/kiln/capabilities/caps/pi-code-comprehension
source /root/.kiln-build-env

ITER_SLUG="${ITER_SLUG:-iter1-sft-ideal-r4}"
ADAPTER_NAME="pi-code-comprehension-${ITER_SLUG}"
ADAPTER_REGISTRY="${ADAPTER_DIR:-/workspace/adapters}"
ROOT="/workspace/iter1"
mkdir -p "$ROOT/logs"

# 1. Synthesize ideal SFT data
echo "=== prep_sft_ideal ==="
python3 prep_sft_ideal.py --out datasets/sft.ideal.jsonl --include-system-prompt 2>&1 | tail -5
wc -l datasets/sft.ideal.jsonl

# 2. Dry-run validate
echo "=== cuda_sft_file --dry-run ==="
KILN_CUDA_ARCHS=86 /workspace/kiln/target/release/examples/cuda_sft_file \
  --data datasets/sft.ideal.jsonl \
  --model /workspace/Qwen3.5-4B \
  --output "$ROOT/adapter" \
  --adapter-name "$ADAPTER_NAME" \
  --rank 4 --alpha 8 --lr 1e-4 --epochs 1 \
  --seed 3141592653 \
  --dry-run 2>&1 | tail -20

# 3. Real training
echo "=== cuda_sft_file (training) ==="
KILN_CUDA_ARCHS=86 /workspace/kiln/target/release/examples/cuda_sft_file \
  --data datasets/sft.ideal.jsonl \
  --model /workspace/Qwen3.5-4B \
  --output "$ROOT/adapter" \
  --adapter-name "$ADAPTER_NAME" \
  --rank 4 --alpha 8 --lr 1e-4 --epochs 1 \
  --seed 3141592653 \
  --adapter-smoke-test \
  --install-adapter-dir "$ADAPTER_REGISTRY" \
  --install-adapter-name "$ADAPTER_NAME" \
  2>&1 | tee "$ROOT/logs/train.log" | tail -50

# 4. Verify adapter
echo "=== kiln adapter verify ==="
/workspace/kiln/target/release/kiln adapter verify "$ADAPTER_NAME" \
  --adapter-dir "$ADAPTER_REGISTRY" \
  --url http://localhost:8420 \
  --json 2>&1 | tee "$ROOT/logs/verify.json" | head -20

# 5. 3-seed eval
echo "=== ${ITER_SLUG} 3-seed eval ==="
mkdir -p "$ROOT/eval"
for s in 1 2 3; do
  outdir="$ROOT/eval/seed-$s"
  mkdir -p "$outdir"
  echo "--- seed $s ---"
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
    --adapter "$ADAPTER_NAME" \
    --verbose 2>&1 | tail -10
  jq '.mean_composite, .mean_outcome, .mean_grounding, .mean_invariant_coverage' "$outdir/summary.json" 2>/dev/null || echo "no summary"
done

# 6. Aggregate + paired lift vs base
python3 - "$ROOT" "$ITER_SLUG" <<'PY'
import json, sys, statistics
from pathlib import Path
root = Path(sys.argv[1])
slug = sys.argv[2]
seeds = [1, 2, 3]
def load(arm_path):
    return [json.load(open(p)) for p in [arm_path/f"seed-{s}"/"summary.json" for s in seeds] if p.exists()]
base = load(Path("/workspace/iter0/base"))
this = load(root/"eval")
if not base or not this:
    print(f"missing: base={len(base)} this={len(this)}"); sys.exit(0)
bc = [s["mean_composite"] for s in base]
tc = [s["mean_composite"] for s in this]
paired = [a - b for a, b in zip(tc, bc)]
out = {
    "iter_slug": slug,
    "base_3seed_mean": statistics.mean(bc),
    "this_3seed_mean": statistics.mean(tc),
    "this_3seed_stdev": statistics.stdev(tc) if len(tc) > 1 else 0.0,
    "paired_lifts": paired,
    "paired_lift_mean": statistics.mean(paired),
    "paired_lift_stdev": statistics.stdev(paired) if len(paired) > 1 else 0.0,
    "sigma_above_zero": statistics.mean(paired) / max(statistics.stdev(paired) if len(paired) > 1 else 1e-9, 1e-9),
    "ships": statistics.mean(paired) >= 0.10 and (statistics.mean(paired) / max(statistics.stdev(paired) if len(paired) > 1 else 1e-9, 1e-9)) >= 3.0,
    "sub_scores_mean": {
        k: statistics.mean([s.get(f"mean_{k}", 0) for s in this])
        for k in ["outcome", "grounding", "cross_file_caller_recall", "invariant_coverage", "format_compliance"]
    },
}
out_path = root / f"{slug}-paired.json"
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY

echo "DONE" > "$ROOT/stage_2_sft_ideal.done"
echo "=== stage 2 complete; results in $ROOT/${ITER_SLUG}-paired.json ==="
