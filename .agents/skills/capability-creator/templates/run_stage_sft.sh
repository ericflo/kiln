#!/usr/bin/env bash
# run_stage_sft.sh — reference SFT stage runner.
#
# Usage (from inside capabilities/caps/<cap>/):
#   ./run_stage.sh sft stage-1-bootstrap [--base-adapter <name>]
#
# This is the reference; copy into a cap's run_stage.sh and customise.
#
# Pipeline:
#   1. rubric_sanity (mandatory)
#   2. build_corpus --method sft (if data file missing)
#   3. cuda_sft_file with install-adapter-dir + adapter-smoke-test
#   4. kiln adapter verify
#   5. 3-seed eval via capability.oracle.sh
#   6. capability.anchor.sh (SFT-specific regression watch)
#   7. integration/cross-cap-coherence sibling check
#   8. Append iter row to capability.jsonl
#   9. If kept, write stages/<N>.json and update pipeline.md

set -euo pipefail
cd "$(dirname "$0")"

METHOD="${1:-sft}"
SLUG="${2:?usage: run_stage_sft.sh sft stage-N-slug [--base-adapter <name>]}"
shift 2

BASE_ADAPTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-adapter) BASE_ADAPTER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

CAP="$(basename "$(pwd)")"
CFG="capability.config.json"
OUT_ROOT="/tmp/${CAP}-iter-${SLUG}"
ADAPTER_NAME="${CAP}-${SLUG}"
ADAPTER_REGISTRY="${ADAPTER_DIR:-/workspace/adapters}"
mkdir -p "$OUT_ROOT"

# 1. rubric_sanity
python3 rubric_sanity.py

# 2. build data file if missing
if [[ ! -s datasets/sft.train.jsonl ]]; then
  python3 build_corpus.py --method sft
fi

# 3. cuda_sft_file
EXTRA_FLAGS=()
[[ -n "$BASE_ADAPTER" ]] && EXTRA_FLAGS+=(--base-adapter "$BASE_ADAPTER")

KILN_CUDA_ARCHS=86 cuda_sft_file \
  --data datasets/sft.train.jsonl \
  --model /workspace/Qwen3.5-4B \
  --output "$OUT_ROOT/adapter" \
  --adapter "$ADAPTER_NAME" \
  --rank 4 --alpha 8 --lr 1e-4 --epochs 1 \
  --dataset-cap 128 \
  --seed 3141592653 \
  --adapter-smoke-test \
  --install-adapter-dir "$ADAPTER_REGISTRY" \
  --install-adapter-name "$ADAPTER_NAME" \
  "${EXTRA_FLAGS[@]}"

# 4. kiln adapter verify
kiln adapter verify "$ADAPTER_NAME" \
  --adapter-dir "$ADAPTER_REGISTRY" \
  --url http://localhost:8420

# 5. 3-seed eval
SEEDS=3 ./capability.oracle.sh "$ADAPTER_NAME"

# 6. SFT-specific anchor regression
if [[ -x ./capability.anchor.sh ]]; then
  ./capability.anchor.sh "$ADAPTER_NAME"
fi

# 7. cross-cap regression check
(cd ../../integration/cross-cap-coherence/ && \
  ./capability.oracle.sh "$ADAPTER_NAME")

# 8. Append iter row to capability.jsonl
python3 - "$CAP" "$SLUG" "$ADAPTER_NAME" "$BASE_ADAPTER" "$OUT_ROOT" <<'PY'
import json, sys, os, datetime
from pathlib import Path

cap, slug, adapter, base_adapter, out_root = sys.argv[1:6]
receipt = Path(out_root) / "adapter" / "train_receipt.json"
eval_summary = Path(f"/tmp/{cap}-eval-{adapter}.json")
row = {
    "ts": datetime.datetime.utcnow().isoformat() + "Z",
    "slug": slug,
    "method": "sft",
    "base_adapter": base_adapter or None,
    "output_adapter": adapter,
    "stage": int(slug.split("-")[1]) if slug.startswith("stage-") else None,
}
if receipt.exists():
    row["train_receipt"] = str(receipt)
if eval_summary.exists():
    es = json.loads(eval_summary.read_text())
    row.update({
        "composite": es.get("mean_composite"),
        "sub_scores": es.get("sub_scores_mean"),
        "verdict": es.get("verdict"),
        "status": "kept" if es.get("verdict") == "positive" else "ablation",
    })
with open("capability.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"appended iter row for {slug}")
PY

echo "stage-sft complete: $ADAPTER_NAME"
echo "If kept, promote with: bash \$SKILL/templates/promote_iter_to_stage.sh $SLUG"
