#!/usr/bin/env bash
# run_pipeline.sh — re-run all stages from pipeline.md.
#
# Usage:
#   ./run_pipeline.sh [--from-stage N] [--validate-only]
#
# --from-stage N : start at stage N (default 1). Useful after a base refresh
#                  when earlier stages may consolidate.
# --validate-only: don't train; just kiln adapter verify each stage and
#                  re-run the eval against current base. Reports drift.

set -euo pipefail
cd "$(dirname "$0")"

FROM_STAGE=1
VALIDATE_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-stage) FROM_STAGE="$2"; shift 2 ;;
    --validate-only) VALIDATE_ONLY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

CAP="$(basename "$(pwd)")"
ADAPTER_REGISTRY="${ADAPTER_DIR:-/workspace/adapters}"

# Validate pipeline.md ↔ stages/ ↔ capability.jsonl
python3 ../../lib/stage_manifest.py validate . > /tmp/${CAP}-pipeline-validate.json
if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('/tmp/${CAP}-pipeline-validate.json'))['ok'] else 1)"; then
  echo "ERROR: pipeline.md ↔ stages/ ↔ capability.jsonl inconsistent" >&2
  cat /tmp/${CAP}-pipeline-validate.json >&2
  exit 1
fi

# Parse stages from pipeline.md header
STAGES_JSON=$(python3 - <<'PY'
import json, sys
sys.path.insert(0, "../../lib")
from stage_manifest import parse_pipeline_header
header = parse_pipeline_header(__import__("pathlib").Path("pipeline.md"))
print(json.dumps(header.get("stages") or []))
PY
)

# Iterate stages in order
echo "$STAGES_JSON" | python3 -c "
import json, sys, subprocess
stages = json.load(sys.stdin)
from_stage = $FROM_STAGE
validate_only = $VALIDATE_ONLY

prev_adapter = None
for s in stages:
    n = s['n']
    method = s['method']
    slug = s['slug']
    adapter = f\"\$(basename \$(pwd))-{slug}\".replace('\$(basename \$(pwd))', __import__('pathlib').Path.cwd().name)
    if n < from_stage:
        prev_adapter = adapter
        continue
    print(f'\\n=== stage {n}: {method} {slug} ===')
    if validate_only:
        subprocess.run(['kiln', 'adapter', 'verify', adapter, '--adapter-dir', '$ADAPTER_REGISTRY', '--url', 'http://localhost:8420'], check=True)
        subprocess.run(['./capability.oracle.sh', adapter], check=True, env={'SEEDS': '3', **__import__('os').environ})
    else:
        cmd = ['./run_stage.sh', method, slug]
        if prev_adapter:
            cmd += ['--base-adapter', prev_adapter]
        subprocess.run(cmd, check=True)
    prev_adapter = adapter
"

# Final integration check on the chain's last adapter
LAST_ADAPTER=$(python3 -c "
import sys, pathlib
sys.path.insert(0, '../../lib')
from stage_manifest import parse_pipeline_header
header = parse_pipeline_header(pathlib.Path('pipeline.md'))
print(header.get('final_adapter') or '')
")

if [[ -n "$LAST_ADAPTER" ]]; then
  echo ""
  echo "=== final integration check: $LAST_ADAPTER ==="
  (cd ../../integration/cross-cap-coherence/ && \
    ./capability.oracle.sh "$LAST_ADAPTER")
fi

echo ""
echo "pipeline complete: $LAST_ADAPTER"
