#!/bin/bash
# Backfill eval for iters whose train+GRPO succeeded but eval bailed on
# transient SSH/pod glitches. Loads the named adapter, runs the full eval,
# updates the capability.jsonl row, commits + pushes, B2-backs up.
#
# Usage: bash backfill_eval.sh <pod_id> <iter_num>
set -uo pipefail
POD_ID="${1:?usage: backfill_eval.sh <pod_id> <iter_num>}"
ITER="${2:?usage: backfill_eval.sh <pod_id> <iter_num>}"
export POD_ID
export RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "== backfill eval iter $ITER pod=$POD_ID $(date -u +%H:%M:%SZ) =="

# Ensure kiln serve is healthy
if ! python3 $RP ssh $POD_ID 'curl -sS --max-time 5 http://localhost:8420/v1/adapters >/dev/null 2>&1'; then
  echo ">>> kiln serve down — restarting"
  python3 $RP ssh $POD_ID 'pgrep -x kiln | xargs -r kill -9 2>/dev/null || true; sleep 3'
  python3 $RP bg $POD_ID /tmp/kiln-serve-backfill-iter${ITER}.log \
    'cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b ./target/release/kiln serve 2>&1'
  sleep 40
fi

echo ">>> loading adapter pi-diff-patch-apply-iter${ITER}"
python3 $RP ssh $POD_ID "curl -sS -X POST http://localhost:8420/v1/adapters/load -H 'Content-Type: application/json' -d '{\"name\":\"pi-diff-patch-apply-iter${ITER}\"}'"
echo

echo ">>> running eval (24 tasks)"
python3 $RP bg $POD_ID /tmp/iter${ITER}-backfill-eval.log \
  "cd /workspace/kiln/capabilities/agentic-grpo/pi-diff-patch-apply && rm -rf /tmp/iter${ITER}-eval && python3 rollout.py \
    --tasks datasets/eval.tasks.jsonl \
    --out-dir /tmp/iter${ITER}-eval --mode eval --num-generations 1 \
    --adapter current --seed-base 3141592653 --parallel 4 \
    --max-wall-clock-s 180 \
    --temperature 0.0 --verbose 2>&1"
python3 $RP wait-file $POD_ID /tmp/iter${ITER}-eval/summary.json --timeout 3600

echo ">>> pulling summary"
python3 $RP ssh $POD_ID "cat /tmp/iter${ITER}-eval/summary.json" > /tmp/eval-summary-iter${ITER}.json

COMPOSITE=$(python3 -c "import json; print(json.load(open('/tmp/eval-summary-iter${ITER}.json'))['mean_composite'])")
BASELINE=0.9418645833333333
DELTA=$(python3 -c "print($COMPOSITE - $BASELINE)")
echo "iter ${ITER}: composite=$COMPOSITE baseline=$BASELINE delta=$DELTA"

echo ">>> updating capability.jsonl row"
python3 << PYEOF
import json, pathlib, datetime
p = pathlib.Path('$HERE/capability.jsonl')
keep = []
for line in p.read_text().splitlines():
    if not line.strip(): continue
    try: e = json.loads(line)
    except: keep.append(line); continue
    if e.get('iter') == $ITER and e.get('status') in ('EVAL-PENDING', 'FAILED-no-eval'):
        s = json.load(open('/tmp/eval-summary-iter${ITER}.json'))
        e['composite'] = float($COMPOSITE)
        e['delta_vs_baseline'] = float($DELTA)
        e['status'] = 'logged-backfill'
        e['class_means'] = s.get('class_means')
        e['rollouts_passed'] = s.get('rollouts_passed')
        e['n_rollouts'] = s.get('n_rollouts')
        e['backfilled_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
        keep.append(json.dumps(e))
    else:
        keep.append(line)
p.write_text('\n'.join(keep) + '\n')
print(f'updated iter $ITER row')
PYEOF

cd "$HERE/../../.."
git add capabilities/agentic-grpo/pi-diff-patch-apply/capability.jsonl
git pull --rebase origin main 2>&1 | tail -1
git commit -m "cap[agentic-grpo/pi-diff-patch-apply]: iter ${ITER} backfill eval (composite=${COMPOSITE}, Δ=${DELTA})" 2>&1 | tail -1
git push origin main 2>&1 | tail -1
cd - >/dev/null

python3 $HERE/backup_to_b2.py --iter ${ITER} --kind train --pod ${POD_ID} 2>&1 | tail -3 || echo "B2 backup failed"

echo "== backfill iter ${ITER} done =="
