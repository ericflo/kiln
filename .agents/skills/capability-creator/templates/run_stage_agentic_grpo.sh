#!/usr/bin/env bash
# run_stage_agentic_grpo.sh — reference agentic-GRPO stage runner (with ECHO).
#
# Usage:
#   ./run_stage.sh agentic-grpo stage-3-final [--base-adapter <name>] [--no-policy-loss]
#
# Pipeline:
#   1. rubric_sanity
#   2. pi_smoke (if no prior agentic stage shipped — only once per cap)
#   3. rollouts via rollout.py (pi-driven, scored by rubric.py)
#   4. kiln trajectory inspect (validate masks)
#   5. cuda_grpo_ablation --dry-run with ECHO on
#   6. cuda_grpo_ablation real training
#   7. kiln adapter verify
#   8. 3-seed eval
#   9. integration sibling check
#  10. Append iter row

set -euo pipefail
cd "$(dirname "$0")"

METHOD="${1:-agentic-grpo}"
SLUG="${2:?usage: run_stage_agentic_grpo.sh agentic-grpo stage-N-slug [--base-adapter NAME] [--no-policy-loss]}"
shift 2

BASE_ADAPTER=""
NO_POLICY_LOSS=""
ECHO_LAMBDA="${ECHO_LAMBDA:-0.05}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-adapter) BASE_ADAPTER="$2"; shift 2 ;;
    --no-policy-loss) NO_POLICY_LOSS="--no-policy-loss"; shift ;;
    --echo-lambda) ECHO_LAMBDA="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

CAP="$(basename "$(pwd)")"
OUT_ROOT="/tmp/${CAP}-iter-${SLUG}"
ROLLOUT_DIR="$OUT_ROOT/rollouts"
ADAPTER_NAME="${CAP}-${SLUG}"
ADAPTER_REGISTRY="${ADAPTER_DIR:-/workspace/adapters}"
mkdir -p "$ROLLOUT_DIR"

# 1. rubric_sanity
python3 rubric_sanity.py

# 2. pi_smoke (only if pi hasn't been smoke-tested for this cap)
if [[ ! -f .pi-smoke-passed ]]; then
  bash "$(dirname "$(realpath "$0")")/pi_smoke.sh"
  touch .pi-smoke-passed
fi

# 3. gather rollouts
python3 rollout.py \
  --tasks datasets/train.tasks.jsonl \
  --out-dir "$ROLLOUT_DIR" \
  --config capability.config.json \
  --num-generations 4 \
  --mode train \
  --limit 30

# 4. validate trajectories
kiln trajectory inspect "$ROLLOUT_DIR/grpo-train.jsonl" --json \
  > "$OUT_ROOT/trajectory_inspect.json"

# 5. dry-run with ECHO
EXTRA_FLAGS=()
[[ -n "$BASE_ADAPTER" ]] && EXTRA_FLAGS+=(--base-adapter "$BASE_ADAPTER")
[[ -n "$NO_POLICY_LOSS" ]] && EXTRA_FLAGS+=("$NO_POLICY_LOSS")

KILN_CUDA_ARCHS=86 cuda_grpo_ablation \
  --data "$ROLLOUT_DIR/grpo-train.jsonl" \
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
  --echo-lambda "$ECHO_LAMBDA" \
  --echo-env-mask-mode env_only \
  --dry-run \
  "${EXTRA_FLAGS[@]}"

# 6. real training
KILN_CUDA_ARCHS=86 cuda_grpo_ablation \
  --data "$ROLLOUT_DIR/grpo-train.jsonl" \
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
  --echo-lambda "$ECHO_LAMBDA" \
  --echo-env-mask-mode env_only \
  --adapter-smoke-test \
  --install-adapter-dir "$ADAPTER_REGISTRY" \
  --install-adapter-name "$ADAPTER_NAME" \
  "${EXTRA_FLAGS[@]}"

# 7. verify
kiln adapter verify "$ADAPTER_NAME" \
  --adapter-dir "$ADAPTER_REGISTRY" \
  --url http://localhost:8420

# 8. 3-seed eval
SEEDS=3 ./capability.oracle.sh "$ADAPTER_NAME"

# 9. cross-cap regression
(cd ../../integration/cross-cap-coherence/ && \
  ./capability.oracle.sh "$ADAPTER_NAME")

# 10. append iter row
python3 - "$CAP" "$SLUG" "$ADAPTER_NAME" "$BASE_ADAPTER" "$OUT_ROOT" <<'PY'
import json, sys, datetime
from pathlib import Path

cap, slug, adapter, base_adapter, out_root = sys.argv[1:6]
receipt_p = Path(out_root) / "adapter" / "train_receipt.json"
eval_p = Path(f"/tmp/{cap}-eval-{adapter}.json")
row = {
    "ts": datetime.datetime.utcnow().isoformat() + "Z",
    "slug": slug,
    "method": "agentic-grpo",
    "base_adapter": base_adapter or None,
    "output_adapter": adapter,
    "stage": int(slug.split("-")[1]) if slug.startswith("stage-") else None,
}
if receipt_p.exists():
    r = json.loads(receipt_p.read_text())
    row["train_receipt"] = str(receipt_p)
    em = r.get("echo_metrics") or {}
    row["method_specific"] = {"agentic-grpo": {
        "groups_seen": r.get("groups_seen"),
        "groups_trained": r.get("groups_trained"),
        "reward_mean": r.get("reward_mean"),
        "echo_lambda": r.get("echo_lambda"),
        "no_policy_loss": r.get("no_policy_loss"),
        "env_token_ce_initial": em.get("env_token_ce_initial"),
        "env_token_ce_final": em.get("env_token_ce_final"),
        "env_ce_steps_observed": em.get("env_ce_steps_observed"),
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

echo "stage-agentic-grpo complete: $ADAPTER_NAME"
echo "If kept, promote with: bash \$SKILL/templates/promote_iter_to_stage.sh $SLUG"
