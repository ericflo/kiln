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

# Hand-designed hypothesis table. Each row: SLUG | rationale | run_iter.sh-flags.
#
# KEY DATA SO FAR (as of iter 13, updated as iters complete):
#   - iter 2 (T=1.0 hard-mix from base): 0.9246 — best trained iter
#   - iter 12 (rank 8 from base): 0.8882 — smaller rank retains clean class
#   - iter 13 (T=1.0 hard-mix chain-from-iter2): 0.8462 — **chain compound FAILS**
#     → don't chain unless explicitly testing compounding
#   - all iters except iter 2 regress more than 5pp vs baseline 0.9419
#
# DESIGN PRINCIPLES going forward:
#   - DEFAULT to train from base (chain compounding actively hurts per iter 13)
#   - Test one variable at a time, response-to-data hypotheses
#   - Prefer untested axes (incorrect-only corpus, rank 2, no filter, etc.)
recipe_for() {
  local n=$1
  local BEST=$(best_adapter)
  case "$n" in
    13)
      # Already run — chain compound test (failed).
      echo "h13-chain-best-hard-mix --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --train-tasks-file datasets/train.hard_mix.tasks.jsonl --train-adapter ${BEST}"
      ;;
    14)
      # Hypothesis: training ONLY on incorrect-class tasks (24pp headroom) from
      # base. iter 12 retained clean class — maybe targeting only the broken
      # class will let us actually improve it without trading off clean/drift.
      echo "h14-incorrect-only-from-base --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --train-tasks-file datasets/train.incorrect_only.tasks.jsonl"
      ;;
    15)
      # Hypothesis: rank 2 LoRA (alpha 4) from base. iter 12 (rank 8) was best
      # of the 8-task sweep; rank 2 = even smaller perturbation. Should retain
      # more clean/drift performance while still nudging incorrect.
      echo "h15-rank2-hardmix-from-base --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --rank 2 --alpha 4 --train-tasks-file datasets/train.hard_mix.tasks.jsonl"
      ;;
    16)
      # Hypothesis: 3 epochs from base on hard-mix. iter 11 showed 2 ep > 1 ep;
      # extrapolate. Distinct from chain because all 3 epochs are on same data.
      echo "h16-3epochs-from-base --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --epochs 3 --train-tasks-file datasets/train.hard_mix.tasks.jsonl"
      ;;
    17)
      # Hypothesis: 6 tasks × 6 gens — more gens per task → more per-group
      # variance → more strong-signal groups passing the filter. Closer to
      # iter 2's 6×3 but with double gens.
      echo "h17-6tasks-6gens-from-base --num-train-tasks 6 --num-gens 6 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --train-tasks-file datasets/train.hard_mix.tasks.jsonl"
      ;;
    18)
      # Hypothesis: lr 5e-6 from base. iter 4 was 2nd best from base at this lr.
      # Test on hard-mix corpus (iter 4 used the older select_hard_tasks output).
      echo "h18-lr5e-6-from-base --num-train-tasks 8 --num-gens 3 --lr 5e-6 --filter-var 0.04 --temperature 1.0 --train-tasks-file datasets/train.hard_mix.tasks.jsonl"
      ;;
    19)
      # Hypothesis: combined winners — rank 2 + 3 epochs + lr 5e-6 — all from
      # base. Smallest perturbation × longest horizon × lowest lr.
      echo "h19-combined-from-base --num-train-tasks 8 --num-gens 3 --lr 5e-6 --filter-var 0.04 --temperature 1.0 --rank 2 --alpha 4 --epochs 3 --train-tasks-file datasets/train.hard_mix.tasks.jsonl"
      ;;
    20)
      # **PIVOT after iters 13/14/17/18/19 all degraded on hard_mix corpus.**
      # The hard_mix.tasks.jsonl subset (4 drift + 4 incorrect) has the WRONG
      # class distribution vs eval (13c / 6d / 5i = 54/25/21). Training away
      # from eval distribution -> degraded eval. Pivoting iters 20+ to default
      # `train.tasks.jsonl` (50/30/20) which matches eval better.
      #
      # iter 20: default corpus, rank 8 (replicate iter 12's 0.888 best).
      echo "h20-rank8-default-corpus --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --rank 8 --alpha 16"
      ;;
    21)
      # iter 21: default corpus, rank 2 (smallest perturbation on the right mix).
      echo "h21-rank2-default-corpus --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --rank 2 --alpha 4"
      ;;
    22)
      # iter 22: default corpus, T=1.0 (iter 2's temperature). Tests whether
      # iter 2's success was the corpus (matches eval) + T=1.0 (high variance).
      echo "h22-default-T1.0 --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0"
      ;;
    23)
      # iter 23: default corpus, 2 epochs (iter 11 showed 2ep > 1ep at lr 1e-5).
      echo "h23-2epochs-default --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --epochs 2"
      ;;
    24)
      # iter 24: default corpus, lr 5e-6 (iter 4 was 0.911 at this lr).
      echo "h24-lr5e-6-default --num-train-tasks 8 --num-gens 3 --lr 5e-6 --filter-var 0.04 --temperature 1.0"
      ;;
    25)
      # iter 25: --no-policy-loss (ECHO-only). Tests if removing the GRPO
      # advantage helps. Default corpus.
      echo "h25-no-policy-loss-default --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --no-policy-loss"
      ;;
    26)
      # iter 26: rank 1 LoRA. Absolute minimum perturbation, default corpus.
      echo "h26-rank1-default --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --rank 1 --alpha 2"
      ;;
    27)
      # iter 27: combined-best on default — rank 2 + 2 epochs + lr 5e-6.
      echo "h27-combined-default --num-train-tasks 8 --num-gens 3 --lr 5e-6 --filter-var 0.04 --temperature 1.0 --rank 2 --alpha 4 --epochs 2"
      ;;
    28)
      # iter 28: full 16-task default corpus for more diversity.
      echo "h28-full-corpus-16tasks --num-train-tasks 16 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0"
      ;;
    29)
      # iter 29: seed 1729 sweep, default corpus + T=1.0.
      echo "h29-seed-1729-default --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --seed 1729"
      ;;
    30)
      # Hypothesis: seed sweep 2.
      echo "h30-seed-4242 --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --seed 4242 --train-tasks-file datasets/train.hard_mix.tasks.jsonl"
      ;;
    *)
      # Default for iters 31+: design when we get there based on what 14-30 show.
      echo "h${n}-tbd --num-train-tasks 8 --num-gens 3 --lr 1e-5 --filter-var 0.04 --temperature 1.0 --train-tasks-file datasets/train.hard_mix.tasks.jsonl"
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
