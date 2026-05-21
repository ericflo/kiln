#!/usr/bin/env bash
# run_stage_grpo.sh — reference single-turn GRPO stage runner.
#
# Usage:
#   ./run_stage.sh grpo stage-1-default [--base-adapter <name>]
#
# For multi-turn tool-calling tasks, use run_stage_agentic_grpo.sh instead.
#
# Pipeline:
#   1. rubric_sanity
#   2. build_corpus --method grpo (if data missing)
#   3. cuda_grpo_ablation --dry-run (pre-GPU validation)
#   4. cuda_grpo_ablation real training
#   5. kiln adapter verify
#   6. 3-seed eval
#   7. integration sibling check
#   8. Append iter row

set -euo pipefail
cd "$(dirname "$0")"

METHOD="${1:-grpo}"
SLUG="${2:?usage: run_stage_grpo.sh grpo stage-N-slug [--base-adapter <name>]}"
shift 2

BASE_ADAPTER=""
NO_POLICY_LOSS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-adapter) BASE_ADAPTER="$2"; shift 2 ;;
    --no-policy-loss) NO_POLICY_LOSS="--no-policy-loss"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

CAP="$(basename "$(pwd)")"
OUT_ROOT="/tmp/${CAP}-iter-${SLUG}"
ADAPTER_NAME="${CAP}-${SLUG}"
ADAPTER_REGISTRY="${ADAPTER_DIR:-/workspace/adapters}"
mkdir -p "$OUT_ROOT"

# 1. rubric_sanity
python3 rubric_sanity.py

# 2. build data
if [[ ! -s datasets/grpo.tasks.jsonl ]]; then
  python3 build_corpus.py --method grpo
fi

# 3. dry-run (MANDATORY before GPU)
EXTRA_FLAGS=()
[[ -n "$BASE_ADAPTER" ]] && EXTRA_FLAGS+=(--base-adapter "$BASE_ADAPTER")
[[ -n "$NO_POLICY_LOSS" ]] && EXTRA_FLAGS+=("$NO_POLICY_LOSS")

KILN_CUDA_ARCHS=86 cuda_grpo_ablation \
  --data datasets/grpo.tasks.jsonl \
  --model /workspace/Qwen3.5-4B \
  --output "$OUT_ROOT/adapter" \
  --adapter "$ADAPTER_NAME" \
  --mode phase1 \
  --rank 16 --alpha 32 --lr 1e-5 \
  --kl-coeff 0.1 --clip-epsilon 0.20 \
  --num-generations 4 \
  --seed 3141592653 \
  --filter-var-min 0.05 \
  --on-empty-filter fail \
  --dry-run \
  "${EXTRA_FLAGS[@]}"

# 4. real training
KILN_CUDA_ARCHS=86 cuda_grpo_ablation \
  --data datasets/grpo.tasks.jsonl \
  --model /workspace/Qwen3.5-4B \
  --output "$OUT_ROOT/adapter" \
  --adapter "$ADAPTER_NAME" \
  --mode phase1 \
  --rank 16 --alpha 32 --lr 1e-5 \
  --kl-coeff 0.1 --clip-epsilon 0.20 \
  --num-generations 4 \
  --seed 3141592653 \
  --filter-var-min 0.05 \
  --on-empty-filter fail \
  --adapter-smoke-test \
  --install-adapter-dir "$ADAPTER_REGISTRY" \
  --install-adapter-name "$ADAPTER_NAME" \
  "${EXTRA_FLAGS[@]}"

# 5. verify
kiln adapter verify "$ADAPTER_NAME" \
  --adapter-dir "$ADAPTER_REGISTRY" \
  --url http://localhost:8420

# 6. 3-seed eval
SEEDS=3 ./capability.oracle.sh "$ADAPTER_NAME"

# 7. cross-cap regression
(cd ../../integration/cross-cap-coherence/ && \
  ./capability.oracle.sh "$ADAPTER_NAME")

# 8. append iter row
python3 - "$CAP" "$SLUG" "$ADAPTER_NAME" "$BASE_ADAPTER" "$OUT_ROOT" <<'PY'
import json, sys, datetime
from pathlib import Path

cap, slug, adapter, base_adapter, out_root = sys.argv[1:6]
receipt_p = Path(out_root) / "adapter" / "train_receipt.json"
eval_p = Path(f"/tmp/{cap}-eval-{adapter}.json")
row = {
    "ts": datetime.datetime.utcnow().isoformat() + "Z",
    "slug": slug,
    "method": "grpo",
    "base_adapter": base_adapter or None,
    "output_adapter": adapter,
    "stage": int(slug.split("-")[1]) if slug.startswith("stage-") else None,
}
if receipt_p.exists():
    r = json.loads(receipt_p.read_text())
    row["train_receipt"] = str(receipt_p)
    row["method_specific"] = {"grpo": {
        "groups_seen": r.get("groups_seen"),
        "groups_trained": r.get("groups_trained"),
        "reward_mean": r.get("reward_mean"),
        "reward_stdev": r.get("reward_stdev"),
        "saturation_warning": r.get("reward_saturation_warning"),
    }}
if eval_p.exists():
    es = json.loads(eval_p.read_text())
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

echo "stage-grpo complete: $ADAPTER_NAME"
echo "If kept, promote with: bash \$SKILL/templates/promote_iter_to_stage.sh $SLUG"
