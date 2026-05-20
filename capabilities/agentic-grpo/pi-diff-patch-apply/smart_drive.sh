#!/bin/bash
# Smart drive: each iter is hand-designed based on prior data, not a fixed table.
#
# Principle: every new hypothesis must respond to what previous iters showed,
# not just sweep hyperparameters. Track which adapter is "best so far" and
# chain from it when warmstart is part of the hypothesis.
#
# Usage: bash smart_drive.sh --pod 9jshui49gl9up2

set -uo pipefail
POD_ID="9jshui49gl9up2"
while [ $# -gt 0 ]; do
  case "$1" in
    --pod) POD_ID="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
export POD_ID
export RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Per-cap env so concurrent caps don't clobber.
echo "export POD_ID=${POD_ID}" > /tmp/grpo-pod-pdp.env
echo "export RP=${RP}" >> /tmp/grpo-pod-pdp.env

# Find the next iter to run (max iter in capability.jsonl + 1)
NEXT_ITER=$(python3 -c "
import json
n = -1
for line in open('$HERE/capability.jsonl'):
    try:
        e = json.loads(line)
        v = e.get('iter')
        if isinstance(v, int):
            n = max(n, v)
    except Exception: pass
print(n + 1)
")
echo "starting from iter $NEXT_ITER"

# Find the best trained adapter so far (max composite, status=logged or NEGATIVE/NEGATIVE-mild)
best_adapter() {
  python3 -c "
import json
best = (None, 0.0)
for line in open('$HERE/capability.jsonl'):
    try:
        e = json.loads(line)
        v = e.get('iter')
        c = e.get('composite')
        if isinstance(v, int) and v >= 1 and isinstance(c, (int, float)) and c > best[1]:
            best = (v, c)
    except Exception: pass
if best[0] is None:
    print('base')
else:
    print(f'pi-diff-patch-apply-iter{best[0]}')
"
}

# Hand-designed hypothesis table. Each row: SLUG | rationale | run_iter.sh-flags
recipe_for() {
  local n=$1
  local BEST=$(best_adapter)
  case "$n" in
    13)
      # Hypothesis: iter 2's recipe (T=1.0, hard-mix, default echo) was the
      # original best (0.9246). Re-run with chain from current best to compound.
      echo "h13-chain-best-hard-mix --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --train-tasks-file datasets/train.hard_mix.tasks.jsonl --train-adapter ${BEST}"
      ;;
    14)
      # Hypothesis: train ONLY on incorrect-class tasks (where base has 24%
      # headroom). All other classes are noise — base already gets 0.998/0.975.
      echo "h14-incorrect-only --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --train-tasks-file datasets/train.incorrect_only.tasks.jsonl --train-adapter ${BEST}"
      ;;
    15)
      # Hypothesis: rank 2 LoRA = minimal perturbation. The catastrophic
      # collapses (iter 5/7) and the verbosity pathology (all iters) both
      # suggest the LoRA is moving the model too far. Rank 2 caps it.
      echo "h15-rank2-chain --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --rank 2 --alpha 4 --train-tasks-file datasets/train.hard_mix.tasks.jsonl --train-adapter ${BEST}"
      ;;
    16)
      # Hypothesis: 3 epochs on the hard-mix subset. Iter 11 showed 2 ep > 1 ep.
      # Test if more compounds (with chain from best).
      echo "h16-3epochs-chain --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --epochs 3 --train-tasks-file datasets/train.hard_mix.tasks.jsonl --train-adapter ${BEST}"
      ;;
    17)
      # Hypothesis: 6 gens × 6 tasks (less tasks, more gens per task) gives
      # higher per-task variance → more strong-signal groups. Closer to iter 2's
      # 6×3 recipe but with double the gens.
      echo "h17-6tasks-6gens --num-train-tasks 6 --num-gens 6 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --train-tasks-file datasets/train.hard_mix.tasks.jsonl --train-adapter ${BEST}"
      ;;
    18)
      # Hypothesis: lower lr (5e-6) + chain. Iter 4 (lr 5e-6 from base) was
      # second-best at 0.9109. Maybe lr 5e-6 from chain is even better.
      echo "h18-lr5e-6-chain --num-train-tasks 8 --num-gens 3 --lr 5e-6 --filter-var 0.04 --temperature 1.0 --train-tasks-file datasets/train.hard_mix.tasks.jsonl --train-adapter ${BEST}"
      ;;
    19)
      # Hypothesis: combine winners — rank 2 + 3 epochs + lr 5e-6 + chain. Small
      # gradient steps over a long horizon from the best baseline.
      echo "h19-rank2-3ep-lr5e6-chain --num-train-tasks 8 --num-gens 3 --lr 5e-6 --filter-var 0.04 --temperature 1.0 --rank 2 --alpha 4 --epochs 3 --train-tasks-file datasets/train.hard_mix.tasks.jsonl --train-adapter ${BEST}"
      ;;
    20)
      # Hypothesis: filter-var 0.0 (no filter, all groups train). The hard-mix
      # subset already biases toward high-variance tasks; further filtering may
      # discard signal-rich groups. Test if removing the filter helps.
      echo "h20-no-filter-chain --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.0 --temperature 1.0 --train-tasks-file datasets/train.hard_mix.tasks.jsonl --train-adapter ${BEST}"
      ;;
    21)
      # Replay iter 2 EXACTLY (no chain): same recipe, same seed. Sanity check
      # that the iter 2 result is reproducible, not a lucky seed.
      echo "h21-iter2-replay --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --seed 3141592653 --train-tasks-file datasets/train.hard_mix.tasks.jsonl"
      ;;
    22)
      # Combine: chain from best + incorrect-only + rank 2 (smallest-perturb
      # gradient on the largest-headroom class).
      echo "h22-incorrect-rank2-chain --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --rank 2 --alpha 4 --train-tasks-file datasets/train.incorrect_only.tasks.jsonl --train-adapter ${BEST}"
      ;;
    *)
      # Default: chain best with default recipe. New ideas to be added.
      echo "h${n}-tbd-chain --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --train-tasks-file datasets/train.hard_mix.tasks.jsonl --train-adapter ${BEST}"
      ;;
  esac
}

# Pre-iter kiln-serve health check.
ensure_kiln_serve() {
  if ! python3 $RP ssh $POD_ID 'curl -sS --max-time 5 http://localhost:8420/v1/adapters >/dev/null 2>&1'; then
    echo "  kiln serve down — killing zombies + restarting"
    python3 $RP ssh $POD_ID 'pgrep -x kiln | xargs -r kill -9 2>/dev/null || true; sleep 3'
    python3 $RP bg $POD_ID /tmp/kiln-serve-smart-iter${1}.log \
      'cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b ./target/release/kiln serve 2>&1'
    sleep 35
  fi
}

# Append capability.jsonl row from the eval summary.
log_row() {
  local n=$1 slug=$2 verdict=$3
  local EVAL_OUT="/tmp/iter${n}-eval"
  python3 $RP ssh $POD_ID "cat ${EVAL_OUT}/summary.json 2>/dev/null" > /tmp/eval-summary-iter${n}.json 2>/dev/null || true
  python3 -c "
import json, datetime, pathlib
try:
    s = json.load(open('/tmp/eval-summary-iter${n}.json'))
except Exception:
    s = {}
composite = s.get('mean_composite')
baseline = 0.9418645833333333
delta = (composite - baseline) if isinstance(composite, (int, float)) else None
status = 'logged' if composite is not None else 'FAILED-no-eval'
row = {
    'iter': ${n},
    'kind': 'smart',
    'slug': '${slug}',
    'composite': composite,
    'baseline_composite': baseline,
    'delta_vs_baseline': delta,
    'status': status,
    'verdict': '${verdict}',
    'sub_scores': s.get('mean_sub_scores') or {},
    'class_means': s.get('class_means') or {},
    'rollouts_passed': s.get('rollouts_passed'),
    'n_rollouts': s.get('n_rollouts'),
    'temperature': 0.0,
    'eval_seed': 3141592653,
    'pod_id': '${POD_ID}',
    'logged_at': datetime.datetime.utcnow().isoformat() + 'Z',
}
pathlib.Path('$HERE/capability.jsonl').open('a').write(json.dumps(row) + chr(10))
print(f'logged iter ${n}: composite={composite} delta={delta}')
"
}

# Main loop.
iter=$NEXT_ITER
while [ "$iter" -lt 50 ]; do
  RECIPE=$(recipe_for "$iter")
  SLUG=$(echo "$RECIPE" | awk '{print $1}')
  FLAGS=$(echo "$RECIPE" | cut -d' ' -f2-)
  BEST=$(best_adapter)

  echo "================================================="
  echo "  ITER $iter  slug=$SLUG  best_so_far=$BEST"
  echo "  flags: $FLAGS"
  echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "================================================="

  ensure_kiln_serve "$iter"

  EVAL_ADAPTER="pi-diff-patch-apply-iter${iter}"
  bash "$HERE/run_iter.sh" --iter "$iter" --kind "train" --eval-adapter "$EVAL_ADAPTER" $FLAGS 2>&1 \
    | tee /tmp/iter${iter}-run.log | tail -150 \
    || echo "WARN: run_iter iter ${iter} exited non-zero — continuing"

  log_row "$iter" "$SLUG" "Smart iter, best-so-far=${BEST}"

  cd "$HERE/../../.."
  git add capabilities/agentic-grpo/pi-diff-patch-apply/capability.jsonl
  git pull --rebase origin main 2>&1 | tail -1
  git commit -m "cap[agentic-grpo/pi-diff-patch-apply]: iter ${iter} ${SLUG} (smart)" 2>&1 | tail -1
  git push origin main 2>&1 | tail -1
  cd - >/dev/null

  python3 $HERE/backup_to_b2.py --iter ${iter} --kind train --pod ${POD_ID} 2>&1 | tail -3 \
    || echo "WARN: B2 backup failed"

  iter=$((iter + 1))
done

echo "================================================="
echo "  SMART DRIVE COMPLETE (next iter $iter)"
echo "================================================="
