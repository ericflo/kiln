#!/usr/bin/env bash
# promote_iter_to_stage.sh — promote a kept capability.jsonl iter to a stage.
#
# Usage (from inside capabilities/caps/<cap>/):
#   bash $SKILL/templates/promote_iter_to_stage.sh <slug>
#
# Reads the most recent capability.jsonl row matching slug, validates
# promotion criteria (see PIPELINE.md §4.3), writes stages/stage-<N>-<slug>.json,
# and appends to pipeline.md's `stages:` array.

set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

# Actually we want to run from the cap dir, not the skill dir.
# Caller should be in the cap dir.
if [[ ! -f capability.jsonl ]]; then
  echo "error: must be run from inside a cap dir with capability.jsonl" >&2
  exit 2
fi

SLUG="${1:?usage: promote_iter_to_stage.sh <slug>}"

# Find the most recent row matching slug
ROW=$(tac capability.jsonl | grep "\"slug\": \"$SLUG\"" | head -1)

if [[ -z "$ROW" ]]; then
  echo "error: no capability.jsonl row with slug=$SLUG" >&2
  exit 2
fi

# Use stage_manifest.py to write stages/ + update pipeline.md
python3 - "$ROW" "$SLUG" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../lib").resolve()))
from stage_manifest import record_new_stage  # type: ignore

row = json.loads(sys.argv[1])
slug = sys.argv[2]

if row.get("status") != "kept":
    print(f"ERROR: row status={row.get('status')!r}; only 'kept' rows can be promoted", file=sys.stderr)
    sys.exit(1)

stage = row.get("stage")
if stage is None:
    # Infer from slug
    if slug.startswith("stage-"):
        stage = int(slug.split("-")[1])
    else:
        print("ERROR: row has no stage field and slug doesn't start with stage-N-", file=sys.stderr)
        sys.exit(1)

stage_record = {
    "schema_version": 1,
    "stage": stage,
    "slug": slug,
    "method": row.get("method"),
    "hypothesis": row.get("hypothesis", ""),
    "stage_transition_rationale": row.get("stage_transition_rationale", ""),
    "base_adapter": row.get("base_adapter"),
    "output_adapter": row.get("output_adapter"),
    "training_iters": [row],
    "promoted_iter": row.get("iter"),
    "final_composite": row.get("composite"),
    "final_composite_delta": row.get("composite_delta"),
    "final_sub_scores": row.get("sub_scores"),
    "cross_cap_check": row.get("sibling_regression_check"),
    "adapter_manifest": None,
    "train_receipt": row.get("train_receipt"),
    "kiln_commit": row.get("kiln_commit"),
    "ts_promoted": __import__("datetime").datetime.utcnow().isoformat() + "Z",
}

record_new_stage(Path("."), row, stage_record)
print(f"promoted iter {row.get('iter')} to stage {stage} ({slug})")
PY

echo ""
echo "Now edit pipeline.md to add the per-stage prose section."
echo "Then commit pipeline.md + stages/stage-${SLUG##stage-*-}.json"
