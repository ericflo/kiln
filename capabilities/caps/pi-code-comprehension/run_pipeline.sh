#!/usr/bin/env bash
# run_pipeline.sh — restore iter4 ECHO adapter from B2, 3-seed paired eval vs base.
#
# This is the round-3 ceiling-documented reproducer. iter4 ECHO is the strongest
# training arm tested across rounds 1-3 but sits below the +0.10 / 3σ ship gate.
# See pipeline.md for the full ceiling write-up.
#
# Prerequisites (sourced via lib/pod_bootstrap.sh):
#   - A6000 pod (ghcr.io/ericflo/kiln-runpod:latest or equivalent)
#   - kiln built with --features cuda --bin kiln --bin kiln-bench --examples
#   - kiln serve --eval-mode listening on :8420
#   - pi binary on PATH (Node 22 + @earendil-works/pi-coding-agent)
#   - B2 creds (B2_APPLICATION_KEY_ID + B2_APPLICATION_KEY OR AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
#   - capabilities/lib/pod_bootstrap.sh on PATH
#
# Env overrides:
#   ADAPTER_DIR (default /workspace/adapters)
#   MODEL_PATH  (default /workspace/Qwen3.5-4B)
#   KILN_URL    (default http://localhost:8420)
#   EVAL_SEEDS  (default "1 2 3")
#   ROOT        (default /tmp/pi-code-comprehension-pipeline)

set -euo pipefail

CAPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CAPDIR"

# Source pod bootstrap helpers (B2 creds remap, pi install, kiln build, etc).
LIB="${LIB:-/workspace/kiln/capabilities/lib/pod_bootstrap.sh}"
if [ -f "$LIB" ]; then
  # shellcheck disable=SC1090
  source "$LIB"
  pod_export_b2_creds || true
fi

ADAPTER_DIR="${ADAPTER_DIR:-/workspace/adapters}"
MODEL_PATH="${MODEL_PATH:-/workspace/Qwen3.5-4B}"
KILN_URL="${KILN_URL:-http://localhost:8420}"
EVAL_SEEDS="${EVAL_SEEDS:-1 2 3}"
ROOT="${ROOT:-/tmp/pi-code-comprehension-pipeline}"
ADAPTER_NAME="pi-code-comprehension-iter4-h4-echo-0075"
ADAPTER_PATH="${ADAPTER_DIR}/${ADAPTER_NAME}"
B2_KEY="b2://clouderic/kiln/pi-code-comprehension/BEST_ADAPTER_iter4/adapter.tgz"

mkdir -p "$ROOT/base" "$ROOT/iter4"

# 1. Restore iter4 ECHO adapter from B2 if absent.
if [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]; then
  echo "[1/5] Restoring iter4 ECHO adapter from B2…"
  if [ -n "${B2_APPLICATION_KEY_ID:-}" ] && [ -n "${B2_APPLICATION_KEY:-}" ]; then
    b2 account authorize "$B2_APPLICATION_KEY_ID" "$B2_APPLICATION_KEY" >/dev/null
  fi
  mkdir -p "$ADAPTER_PATH"
  TGZ="${ROOT}/iter4-adapter.tgz"
  rm -f "$TGZ"
  b2 file download "$B2_KEY" "$TGZ"
  tar -xzf "$TGZ" -C "$ADAPTER_PATH"
  ls "$ADAPTER_PATH" | head -6
fi

# 2. Confirm kiln serve is up.
if ! curl -sf "$KILN_URL/v1/health" > /dev/null; then
  echo "[2/5] kiln serve not up at $KILN_URL — start it first, e.g.:"
  echo "  KILN_MODEL_PATH=$MODEL_PATH KILN_ADAPTER_DIR=$ADAPTER_DIR \\"
  echo "    nohup /workspace/kiln/target/release/kiln serve --eval-mode > /workspace/kiln-serve.log 2>&1 &"
  exit 2
fi
echo "[2/5] kiln serve up at $KILN_URL"

# 3. kiln adapter verify (loadable + behavioral).
echo "[3/5] kiln adapter verify $ADAPTER_NAME"
/workspace/kiln/target/release/kiln adapter verify "$ADAPTER_NAME" \
  --adapter-dir "$ADAPTER_DIR" \
  --url "$KILN_URL" \
  --json > "$ROOT/verify.json"
jq '{loadable, behavioral_response_status}' "$ROOT/verify.json" || true

# 4. 3-seed paired eval (base, then iter4).
echo "[4/5] 3-seed paired eval (base then iter4)…"
for arm in base iter4; do
  for s in $EVAL_SEEDS; do
    outdir="$ROOT/$arm/seed-$s"
    mkdir -p "$outdir"
    extra=""
    if [ "$arm" = "iter4" ]; then
      extra="--adapter $ADAPTER_NAME"
    fi
    echo "  --- $arm seed $s ---"
    KILN_URL="$KILN_URL" PI_BIN="${PI_BIN:-/usr/bin/pi}" \
    python3 rollout.py \
      --tasks datasets/eval.tasks.jsonl \
      --out-dir "$outdir" \
      --mode eval \
      --num-generations 1 \
      --kiln-url "$KILN_URL" \
      --max-wall-clock-s 180 \
      --concurrency 4 \
      --seed $((100 + s)) \
      $extra \
      --verbose 2>&1 | tail -8
  done
done

# 5. Aggregate paired lift.
echo "[5/5] aggregating paired lift…"
python3 - "$ROOT" "$EVAL_SEEDS" <<'PY'
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2].split()]
def load_arm(arm):
    return [json.load(open(p)) for p in (root/arm/f"seed-{s}"/"summary.json" for s in seeds) if p.exists()]
base = load_arm("base")
iter4 = load_arm("iter4")
if len(base) != len(seeds) or len(iter4) != len(seeds):
    print(f"missing data; base={len(base)} iter4={len(iter4)} expected={len(seeds)}")
    sys.exit(1)
b = [s["mean_composite"] for s in base]
i = [s["mean_composite"] for s in iter4]
p = [a - bb for a, bb in zip(i, b)]
out = {
    "iter_slug": "iter4-ECHO-pipeline-reproducer",
    "session": "now",
    "base_3seed_mean": statistics.mean(b),
    "iter4_3seed_mean": statistics.mean(i),
    "iter4_3seed_stdev": statistics.stdev(i) if len(i) > 1 else 0.0,
    "paired_lifts": p,
    "paired_lift_mean": statistics.mean(p),
    "paired_lift_stdev": statistics.stdev(p) if len(p) > 1 else 0.0,
    "sigma_above_zero": statistics.mean(p) / max(statistics.stdev(p) if len(p) > 1 else 1e-9, 1e-9),
    "ships": statistics.mean(p) >= 0.10 and (statistics.mean(p) / max(statistics.stdev(p) if len(p) > 1 else 1e-9, 1e-9)) >= 3.0,
}
(root / "pipeline-result.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY

echo "Done. Result: $ROOT/pipeline-result.json"
