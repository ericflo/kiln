#!/bin/bash
# Drive 50 iters of pi-faithful-completion through the hypothesis loop.
#
# This is the autonomous loop spine. For each iter:
#   1. Run drive_iter.py --iter N → invokes run_iter.sh with the hypothesis args
#   2. Pull the eval summary from the pod
#   3. Append a row to capability.jsonl
#   4. Backup the adapter to B2
#   5. git commit + push
#
# Resumable: skips iters already logged with status=recorded.

set -uo pipefail   # no -e; iter failures are recorded then continue

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

source /tmp/pi-faithful.env

START_ITER="${1:-0}"
END_ITER="${2:-50}"
BASELINE=""

echo "== drive_iters: $START_ITER -> $END_ITER =="

mkdir -p eval-summaries train-summaries

for ITER in $(seq "$START_ITER" "$END_ITER"); do
  # skip if already logged
  if grep -q "\"iter\": $ITER," capability.jsonl 2>/dev/null; then
    echo "[drive_iters] iter $ITER already logged; skipping"
    continue
  fi

  echo "================================================"
  echo "[drive_iters] === iter $ITER ==="
  echo "================================================"

  if [ "$ITER" = "0" ]; then
    # Iter 0 = baseline eval, no training, no adapter
    SLUG="iter0-baseline"
    EVAL_SUMMARY="eval-summaries/iter0-baseline.json"
    bash run_iter.sh --iter 0 --slug iter0-baseline --skip-train --eval-adapter base 2>&1 | tee /tmp/iter0.log
    # Pull eval summary down
    python3 $RP ssh $POD_ID "cat /tmp/iter0-eval/summary.json" > "$EVAL_SUMMARY" 2>/dev/null || echo '{"mean_composite": 0.0}' > "$EVAL_SUMMARY"
    BASELINE=$(python3 -c "import json; print(json.load(open('$EVAL_SUMMARY'))['mean_composite'])")
    python3 log_iter.py --iter 0 --slug iter0-baseline --family baseline \
      --eval-summary "$EVAL_SUMMARY" \
      --verdict "Baseline eval — base model on held-out set"
    git add capability.jsonl eval-summaries/ || true
    git commit -m "cap[agentic-grpo/pi-faithful-completion]: iter 0 baseline ($BASELINE)" 2>&1 | tail -3
    git push origin main 2>&1 | tail -3
    continue
  fi

  # Look up the hypothesis row to get the slug
  SLUG=$(python3 -c "import json; hs=json.load(open('hypotheses.json'));
[print(h['slug']) for h in hs if h['iter']==$ITER]")
  FAMILY=$(python3 -c "import json; hs=json.load(open('hypotheses.json'));
[print(h['family']) for h in hs if h['iter']==$ITER]")
  if [ -z "$SLUG" ]; then
    echo "[drive_iters] no hypothesis for iter $ITER; stopping"
    break
  fi
  EVAL_SUMMARY="eval-summaries/iter${ITER}-${SLUG}.json"
  TRAIN_SUMMARY="train-summaries/iter${ITER}-${SLUG}.json"
  ADAPTER_NAME="pi-faithful-${SLUG}"

  # Run the iter
  python3 drive_iter.py --iter $ITER 2>&1 | tee /tmp/iter${ITER}.log
  RC=$?

  # Pull eval summary regardless of training success
  python3 $RP ssh $POD_ID "cat /tmp/iter${ITER}-eval/summary.json 2>/dev/null || echo '{}'" > "$EVAL_SUMMARY"
  python3 $RP ssh $POD_ID "cat /tmp/iter${ITER}-rollouts/summary.json 2>/dev/null || echo '{}'" > "$TRAIN_SUMMARY"

  # Hyperparams snapshot
  HP=$(python3 -c "import json; hs=json.load(open('hypotheses.json'));
[print(json.dumps(h['args'])) for h in hs if h['iter']==$ITER]")

  if [ -z "$BASELINE" ]; then
    BASELINE=$(python3 -c "
import json
rows=[json.loads(l) for l in open('capability.jsonl') if l.strip()]
print([r['composite'] for r in rows if r.get('iter')==0][0])
" 2>/dev/null || echo "0")
  fi

  VERDICT=""
  if grep -q '"mean_composite"' "$EVAL_SUMMARY"; then
    COMP=$(python3 -c "import json; d=json.load(open('$EVAL_SUMMARY')); print(d.get('mean_composite', 0))")
    VERDICT="iter $ITER ${SLUG}: composite ${COMP}"
  fi
  python3 log_iter.py --iter $ITER --slug "$SLUG" --family "$FAMILY" \
    --train-summary "$TRAIN_SUMMARY" --eval-summary "$EVAL_SUMMARY" \
    --adapter "$ADAPTER_NAME" \
    --verdict "$VERDICT" \
    --hyperparams "$HP" \
    --baseline "$BASELINE"

  # Backup adapter to B2 (non-fatal)
  python3 backup_to_b2.py --iter $ITER --slug "$SLUG" --adapter "$ADAPTER_NAME" --pod "$POD_ID" 2>&1 | tail -10 || echo "[drive_iters] B2 backup failed (non-fatal)"

  # Git commit + push
  git add capability.jsonl eval-summaries/ train-summaries/ prompts/ datasets/ || true
  git commit -m "cap[agentic-grpo/pi-faithful-completion]: iter ${ITER} ${SLUG} (${VERDICT})" 2>&1 | tail -3 || true
  git push origin main 2>&1 | tail -3 || true

done

echo "== drive_iters: done =="
