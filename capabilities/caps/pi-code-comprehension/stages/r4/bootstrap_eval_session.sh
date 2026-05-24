#!/usr/bin/env bash
# R4 Eval Bootstrap — pull all R4 GRPO arm adapters from B2 and run paired 5-seed eval vs iter4.
# Run on a fresh kiln-runpod A6000 pod where kiln serve is running.
#
# Assumes:
#   - $B2_APPLICATION_KEY_ID and $B2_APPLICATION_KEY are set (or /root/.b2-creds exists)
#   - kiln server listening on localhost:8420
#   - /workspace/kiln/capabilities/caps/pi-code-comprehension/datasets/eval.tasks.jsonl exists

set +e
source /root/.b2-creds 2>/dev/null
b2 account authorize "$B2_APPLICATION_KEY_ID" "$B2_APPLICATION_KEY" >/dev/null 2>&1

CAPDIR=/workspace/kiln/capabilities/caps/pi-code-comprehension
ROOT=/workspace/r4-eval-session
mkdir -p $ROOT/adapters

# 1. Pull all R4 adapters from B2
PARENT=pi-code-comprehension-iter4-h4-echo-0075
ADAPTERS_TO_FETCH=(
  "pi-cc-r4-grpo-chain-iter4-r16a32-v4"
  "pi-cc-r4-grpo-armB-pow2-r16a32"
  "pi-cc-r4-grpo-armD-binarized-r16a32"
  "pi-cc-r4-grpo-armG-iter4strong-r16a32"
  "pi-cc-r4-grpo-armH-ranked-r16a32"
  "pi-cc-r4-grpo-armI-topbot-r16a32"
  "pi-cc-r4-grpo-armJ-base-iter4data-r16a32"
  "pi-cc-r4-grpo-armK-top50merged-r16a32"
)

# Restore iter4 baseline first
if [ ! -f /workspace/adapters/$PARENT/adapter_model.safetensors ]; then
  mkdir -p /workspace/adapters/$PARENT
  b2 file download b2://clouderic/kiln/pi-code-comprehension/BEST_ADAPTER_iter4/adapter.tgz /tmp/iter4.tgz
  tar -xzf /tmp/iter4.tgz -C /workspace/adapters/$PARENT --strip-components=1 2>&1 | tail -3 || \
    tar -xzf /tmp/iter4.tgz -C /workspace/adapters/$PARENT 2>&1 | tail -3
  # The tgz contains pi-cc-iter4/ subdir; flatten if so
  if [ -d /workspace/adapters/$PARENT/pi-cc-iter4 ]; then
    mv /workspace/adapters/$PARENT/pi-cc-iter4/* /workspace/adapters/$PARENT/
    rmdir /workspace/adapters/$PARENT/pi-cc-iter4
  fi
  # Synthesize lineage (B2 strips it)
  cat > /workspace/adapters/$PARENT/lineage.json <<EOF_LINEAGE
{
  "schema_version": 1,
  "adapter_name": "$PARENT",
  "base_model": {"id": "Qwen/Qwen3.5-4B", "config_digest": "sha256:a9362490141004b2336bea8e9d1aa78510664f3072236c80e20ce3b6de623cd1"},
  "kiln_commit": "0.2.19", "created_at": "2026-05-19T17:13:00Z",
  "parent_adapter": null, "replay_hash": "0" 
}
EOF_LINEAGE
fi

# Pull each arm adapter
for ADAPTER in "${ADAPTERS_TO_FETCH[@]}"; do
  if [ -f /workspace/adapters/$ADAPTER/adapter_model.safetensors ]; then
    echo "$ADAPTER already present, skipping"
    continue
  fi
  TGZ="kiln/pi-code-comprehension/r4/${ADAPTER}.tgz"
  echo "fetching $TGZ..."
  if b2 file download b2://clouderic/$TGZ /tmp/${ADAPTER}.tgz 2>&1 | grep -q "Download finished"; then
    mkdir -p /workspace/adapters/$ADAPTER
    tar -xzf /tmp/${ADAPTER}.tgz -C /workspace/adapters/$ADAPTER --strip-components=1 2>&1 || \
      tar -xzf /tmp/${ADAPTER}.tgz -C /workspace/adapters/$ADAPTER
    echo "extracted $ADAPTER"
  else
    echo "  not found in B2 (likely not trained yet in pod-bacd20aa session)"
  fi
done

# 2. Verify each loadable
for ADAPTER in "$PARENT" "${ADAPTERS_TO_FETCH[@]}"; do
  if [ -f /workspace/adapters/$ADAPTER/adapter_model.safetensors ]; then
    echo "=== loading $ADAPTER ==="
    curl -s -X POST http://localhost:8420/v1/adapters/load \
      -H "Content-Type: application/json" \
      -d "{\"name\":\"$ADAPTER\"}" 2>&1 | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('status', d))"
  fi
done

# 3. Eval iter4 baseline (5 seeds)
echo ""
echo "=== eval iter4 baseline (5 seeds) ==="
for s in 1 2 3 4 5; do
  outdir=$ROOT/eval-iter4/seed-$s
  mkdir -p $outdir
  KILN_URL=http://localhost:8420 PI_BIN=/usr/bin/pi \
  python3 $CAPDIR/rollout.py \
    --tasks $CAPDIR/datasets/eval.tasks.jsonl \
    --out-dir $outdir \
    --mode eval --num-generations 1 --kiln-url http://localhost:8420 \
    --max-wall-clock-s 180 --concurrency 4 --seed $((100 + s)) \
    --adapter $PARENT 2>&1 | tail -1
  jq -c "{seed:$s, c:.mean_composite, inv:.mean_invariant_coverage}" $outdir/summary.json 2>/dev/null
done

# 4. Eval each completed arm adapter
for ADAPTER in "${ADAPTERS_TO_FETCH[@]}"; do
  if [ ! -f /workspace/adapters/$ADAPTER/adapter_model.safetensors ]; then continue; fi
  ARM=$(echo $ADAPTER | sed 's/pi-cc-r4-grpo-//; s/-r16a32.*//; s/chain-iter4/v4/')
  echo ""
  echo "=== eval $ARM ($ADAPTER, 5 seeds) ==="
  for s in 1 2 3 4 5; do
    outdir=$ROOT/eval-$ARM/seed-$s
    mkdir -p $outdir
    KILN_URL=http://localhost:8420 PI_BIN=/usr/bin/pi \
    python3 $CAPDIR/rollout.py \
      --tasks $CAPDIR/datasets/eval.tasks.jsonl \
      --out-dir $outdir \
      --mode eval --num-generations 1 --kiln-url http://localhost:8420 \
      --max-wall-clock-s 180 --concurrency 4 --seed $((100 + s)) \
      --adapter $ADAPTER 2>&1 | tail -1
    jq -c "{seed:$s, c:.mean_composite, inv:.mean_invariant_coverage}" $outdir/summary.json 2>/dev/null
  done
done

# 5. Paired analysis
echo ""
echo "=== paired analysis ==="
python3 - <<PY
import json, statistics
from pathlib import Path
R = Path("$ROOT")
def load(arm):
    out = []
    for s in range(1, 6):
        p = R/f"eval-{arm}"/f"seed-{s}"/"summary.json"
        if p.exists():
            out.append(json.load(open(p)))
    return out

iter4 = load("iter4")
if not iter4:
    print("no baseline")
    exit(0)
ic = [s["mean_composite"] for s in iter4]
print(f"iter4 baseline ({len(ic)} seeds): mean={statistics.mean(ic):.4f}")

results = {"iter4": {"per_seed": ic, "mean": statistics.mean(ic),
                     "stdev": statistics.stdev(ic) if len(ic)>1 else 0}}

for arm in ["v4", "armB", "armD", "armG", "armH", "armI", "armJ", "armK"]:
    arm_seeds = load("v4" if arm == "v4" else arm[3:].lower() if arm.startswith("arm") else arm)
    if not arm_seeds:
        continue
    ac = [s["mean_composite"] for s in arm_seeds]
    paired = [a - b for a, b in zip(ac, ic[:len(ac)])]
    pm = statistics.mean(paired)
    ps = statistics.stdev(paired) if len(paired) > 1 else 0
    sigma = pm / max(ps, 1e-9)
    print(f"{arm}: mean={statistics.mean(ac):.4f} paired_lift={pm:+.4f} ({sigma:+.2f}σ) ships={pm >= 0.10 and sigma >= 3.0}")
    results[arm] = {"per_seed": ac, "mean": statistics.mean(ac),
                    "paired_lift_mean": pm, "paired_lift_stdev": ps, "sigma": sigma,
                    "ships": pm >= 0.10 and sigma >= 3.0}

Path("$ROOT/paired-all.json").write_text(json.dumps(results, indent=2))
PY
b2 file upload clouderic $ROOT/paired-all.json kiln/pi-code-comprehension/r4/eval-session-paired-all.json 2>&1 | tail -1

echo "BOOTSTRAP_EVAL_DONE"
