#!/usr/bin/env bash
# run_stage_opd.sh — reference OPD stage runner.
#
# Usage:
#   ./run_stage.sh opd stage-2-polish --base-adapter <name>
#
# Pipeline:
#   1. rubric_sanity
#   2. teacher health check (vLLM on :8002 by default)
#   3. build_corpus --method opd (if prompts missing)
#   4. cuda_opd_remote with install-adapter-dir + adapter-smoke-test
#   5. kiln adapter verify
#   6. 3-seed eval
#   7. integration/cross-cap-coherence sibling check
#   8. Append iter row to capability.jsonl

set -euo pipefail
cd "$(dirname "$0")"

METHOD="${1:-opd}"
SLUG="${2:?usage: run_stage_opd.sh opd stage-N-slug --base-adapter <name>}"
shift 2

BASE_ADAPTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-adapter) BASE_ADAPTER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

CAP="$(basename "$(pwd)")"
OUT_ROOT="/tmp/${CAP}-iter-${SLUG}"
ADAPTER_NAME="${CAP}-${SLUG}"
ADAPTER_REGISTRY="${ADAPTER_DIR:-/workspace/adapters}"
TEACHER_URL="${TEACHER_URL:-http://localhost:8002}"
TEACHER_NAME="${TEACHER_NAME:-qwen3.6-27b-awq}"
mkdir -p "$OUT_ROOT"

# 1. rubric_sanity
python3 rubric_sanity.py

# 2. teacher health
if ! curl -sf "$TEACHER_URL/v1/models" > /dev/null; then
  echo "ORACLE_ERROR: teacher not reachable at $TEACHER_URL" >&2
  exit 2
fi

# 3. build data
if [[ ! -s datasets/opd.prompts.jsonl ]]; then
  python3 build_corpus.py --method opd
fi

# 4. cuda_opd_remote
EXTRA_FLAGS=()
[[ -n "$BASE_ADAPTER" ]] && EXTRA_FLAGS+=(--base-adapter "$BASE_ADAPTER")

KILN_CUDA_ARCHS=86 cuda_opd_remote \
  --prompts datasets/opd.prompts.jsonl \
  --model /workspace/Qwen3.5-4B \
  --teacher-url "$TEACHER_URL" \
  --teacher-name "$TEACHER_NAME" \
  --output "$OUT_ROOT/adapter" \
  --adapter "$ADAPTER_NAME" \
  --rank 16 --alpha 32 --lr 1e-4 --epochs 6 --samples-per-prompt 2 \
  --seed 3141592653 \
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
    "method": "opd",
    "base_adapter": base_adapter or None,
    "output_adapter": adapter,
    "stage": int(slug.split("-")[1]) if slug.startswith("stage-") else None,
}
if receipt_p.exists():
    r = json.loads(receipt_p.read_text())
    row["train_receipt"] = str(receipt_p)
    row["method_specific"] = {"opd": {
        "effective_steps": r.get("effective_steps"),
        "teacher_calls_made": r.get("teacher_calls_made"),
        "skip_rate": r.get("skip_rate"),
        "env_ce_delta": (r.get("echo_metrics") or {}).get("env_token_ce_final", 0)
                        - (r.get("echo_metrics") or {}).get("env_token_ce_initial", 0),
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

echo "stage-opd complete: $ADAPTER_NAME"
echo "If kept, promote with: bash \$SKILL/templates/promote_iter_to_stage.sh $SLUG"
