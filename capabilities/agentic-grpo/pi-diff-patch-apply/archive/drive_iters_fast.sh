#!/bin/bash
# Faster iter cadence: 8 tasks × 3 gens, eval 12, parallel=4, 180s/rollout.
# Target: ~12-15 min per iter.

# Don't use set -e here. If a single iter fails (e.g., kiln serve died,
# rollouts timed out, GRPO crashed), we want to log the failure and continue
# to the next iter. We discovered 2026-05-19 that `bash run_iter.sh | tail
# -100` under `set -euo pipefail` silently killed the whole loop after one
# iter, because run_iter.sh's set -e + pipe buffering hit a non-zero from
# pkill or similar. Loosening here lets the loop survive transient failures.
set -uo pipefail
MAX_ITERS=50
START_ITER=""
POD_ID=""

while [ $# -gt 0 ]; do
  case "$1" in
    --pod) POD_ID="$2"; shift 2 ;;
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --start-iter) START_ITER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
if [ -z "$POD_ID" ]; then echo "--pod required" >&2; exit 1; fi
export POD_ID
export RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Per-cap env file. /tmp/grpo-pod.env is shared across all GRPO caps and
# gets clobbered when multiple concurrent caps run; use pdp-specific path.
echo "export POD_ID=${POD_ID}" > /tmp/grpo-pod-pdp.env
echo "export RP=${RP}" >> /tmp/grpo-pod-pdp.env

next_iter() {
  if [ -n "$START_ITER" ]; then echo "$START_ITER"; return; fi
  if [ ! -s "$HERE/capability.jsonl" ]; then echo 0; return; fi
  python3 -c "
import json
n = -1
for line in open('$HERE/capability.jsonl'):
    try:
        e = json.loads(line)
        v = e.get('iter')
        if isinstance(v, int):
            n = max(n, v)
    except: pass
print(n + 1)
"
}

append_log_row() {
  local n=$1 kind=$2 slug=$3 composite=$4 baseline=$5 delta=$6 status=$7
  python3 -c "
import json, datetime, pathlib
row = {'iter': $n, 'kind': '$kind', 'slug': '$slug', 'composite': $composite,
       'baseline_composite': $baseline, 'delta_vs_baseline': $delta,
       'status': '$status', 'logged_at': datetime.datetime.utcnow().isoformat() + 'Z'}
pathlib.Path('$HERE/capability.jsonl').open('a').write(json.dumps(row) + chr(10))
"
}

# 8×3 small iters; use eval-task-limit 12 via env override.
recipe_for() {
  local n=$1
  case "$n" in
    1)   echo "train h1-default-recipe -gens 3 -tasks 8 -lr 1e-5 -fv 0.04" ;;
    2)   echo "train h2-strong-filter -gens 3 -tasks 8 -lr 1e-5 -fv 0.06" ;;
    3)   echo "train h3-temp10 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -temp 1.0" ;;
    4)   echo "train h4-6gens -gens 6 -tasks 6 -lr 1e-5 -fv 0.04" ;;
    5)   echo "train h5-lower-lr -gens 3 -tasks 8 -lr 5e-6 -fv 0.04" ;;
    6)   echo "train h6-higher-lr -gens 3 -tasks 8 -lr 2e-5 -fv 0.04" ;;
    7)   echo "train h7-rank32 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -rank 32 -alpha 64" ;;
    8)   echo "ablation h8-no-echo -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -no-echo" ;;
    9)   echo "train h9-echo-0.10 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -echo 0.10" ;;
    10)  echo "train h10-echo-0.02 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -echo 0.02" ;;
    11)  echo "train h11-2epoch -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -epochs 2" ;;
    12)  echo "train h12-rank8 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -rank 8 -alpha 16" ;;
    13)  echo "train h13-rank64 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -rank 64 -alpha 128" ;;
    14)  echo "train h14-12tasks -gens 4 -tasks 12 -lr 1e-5 -fv 0.04" ;;
    15)  echo "train h15-seed-2 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -seed 1729" ;;
    16)  echo "train h16-seed-3 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -seed 4242" ;;
    17)  echo "train h17-no-filter -gens 3 -tasks 8 -lr 1e-5 -fv 0.0" ;;
    18)  echo "train h18-low-temp -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -temp 0.5" ;;
    19)  echo "train h19-high-temp -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -temp 1.2" ;;
    20)  echo "train h20-mid-lr -gens 3 -tasks 8 -lr 1.5e-5 -fv 0.04" ;;
    21)  echo "train h21-3epoch -gens 3 -tasks 8 -lr 5e-6 -fv 0.04 -epochs 3" ;;
    22)  echo "train h22-many-tasks -gens 4 -tasks 16 -lr 1e-5 -fv 0.04" ;;
    23)  echo "train h23-fewer-tasks -gens 6 -tasks 4 -lr 1e-5 -fv 0.02" ;;
    24)  echo "train h24-no-policy -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -nopol" ;;
    25)  echo "train h25-seed-4 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -seed 11111" ;;
    26)  echo "train h26-seed-5 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -seed 22222" ;;
    27)  echo "train h27-seed-6 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -seed 33333" ;;
    28)  echo "train h28-replay-best -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -seed 55555" ;;
    29)  echo "train h29-replay-best-2 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -seed 66666" ;;
    30)  echo "train h30-mega-batch -gens 4 -tasks 16 -lr 1e-5 -fv 0.05" ;;
    31)  echo "train h31-tinyfilter -gens 3 -tasks 8 -lr 1e-5 -fv 0.01" ;;
    32)  echo "train h32-bigfilter -gens 3 -tasks 8 -lr 1e-5 -fv 0.10" ;;
    33)  echo "train h33-rank16-lr2e-5 -gens 3 -tasks 8 -lr 2e-5 -fv 0.04" ;;
    34)  echo "train h34-mid-rank-mid-lr -gens 3 -tasks 8 -lr 1.5e-5 -fv 0.04 -rank 32 -alpha 64" ;;
    35)  echo "train h35-3gens-8tasks-3epoch -gens 3 -tasks 8 -lr 5e-6 -fv 0.04 -epochs 3" ;;
    36)  echo "train h36-warmstart-iter1 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -warmstart 1" ;;
    37)  echo "train h37-warmstart-best -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -warmstart-best 1" ;;
    38)  echo "train h38-temp-0.6 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -temp 0.6" ;;
    39)  echo "train h39-temp-0.9 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -temp 0.9" ;;
    40)  echo "train h40-temp-1.1 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -temp 1.1" ;;
    41)  echo "train h41-8gens -gens 8 -tasks 4 -lr 1e-5 -fv 0.02" ;;
    42)  echo "train h42-12gens -gens 12 -tasks 3 -lr 1e-5 -fv 0.01" ;;
    43)  echo "train h43-final-mass -gens 4 -tasks 20 -lr 1e-5 -fv 0.04" ;;
    44)  echo "train h44-mass-low-lr -gens 4 -tasks 20 -lr 5e-6 -fv 0.04" ;;
    45)  echo "train h45-seed-final-1 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -seed 77777" ;;
    46)  echo "train h46-seed-final-2 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -seed 88888" ;;
    47)  echo "train h47-seed-final-3 -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -seed 99999" ;;
    48)  echo "train h48-mega-rank -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -rank 128 -alpha 256" ;;
    49)  echo "train h49-tiny-rank -gens 3 -tasks 8 -lr 1e-5 -fv 0.04 -rank 4 -alpha 8" ;;
    50)  echo "train h50-replay-mass -gens 4 -tasks 16 -lr 1e-5 -fv 0.04 -seed 31415" ;;
    *)   echo "train h-default -gens 3 -tasks 8 -lr 1e-5 -fv 0.04" ;;
  esac
}

parse_recipe() {
  # Convert shorthand to run_iter.sh args
  local recipe="$1"
  local args=""
  local prev=""
  for tok in $recipe; do
    case "$prev" in
      "-gens") args+=" --num-gens $tok" ;;
      "-tasks") args+=" --num-train-tasks $tok" ;;
      "-lr") args+=" --lr $tok" ;;
      "-fv") args+=" --filter-var $tok" ;;
      "-temp") args+=" --temperature $tok" ;;
      "-rank") args+=" --rank $tok" ;;
      "-alpha") args+=" --alpha $tok" ;;
      "-echo") args+=" --echo-lambda $tok" ;;
      "-epochs") args+=" --epochs $tok" ;;
      "-seed") args+=" --seed $tok" ;;
      "-warmstart") args+=" --train-adapter pi-diff-patch-apply-iter$tok" ;;
      "-warmstart-best") args+=" --train-adapter pi-diff-patch-apply-iter-best-so-far" ;;
    esac
    case "$tok" in
      "-no-echo") args+=" --no-echo" ;;
      "-nopol") args+=" --no-policy-loss" ;;
    esac
    prev="$tok"
  done
  echo "$args"
}

iter=$(next_iter)
end=$((iter + MAX_ITERS))
echo "Driving iters [$iter, $end), max 50"

while [ "$iter" -lt 50 ] && [ "$iter" -lt "$end" ]; do
  RECIPE=$(recipe_for "$iter")
  KIND=$(echo "$RECIPE" | awk '{print $1}')
  SLUG=$(echo "$RECIPE" | awk '{print $2}')
  REST=$(echo "$RECIPE" | cut -d' ' -f3-)
  EXTRA=$(parse_recipe "$REST")

  echo "================================================="
  echo "  ITER $iter  kind=$KIND  slug=$SLUG"
  echo "  flags: $EXTRA"
  echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "================================================="

  EVAL_ADAPTER="pi-diff-patch-apply-iter${iter}"
  if [ "$iter" = "0" ]; then EVAL_ADAPTER="base"; fi

  # Pre-iter kiln-serve health check — if down, restart before failing.
  # Without this, one bad iter that leaves kiln serve dead cascades through
  # every subsequent iter (discovered iter 10 → 49 silent cascade 2026-05-19).
  # ALWAYS pkill first if we're going to restart, because:
  #   - a zombie kiln process holding port 8420 will silently SIGKILL the new one
  #     (discovered iter 10 — concurrent serves issue)
  #   - kiln may be hung but still bound to the port (curl times out but bind fails)
  if ! python3 $RP ssh $POD_ID 'curl -sS --max-time 5 http://localhost:8420/v1/adapters >/dev/null 2>&1'; then
    echo "  kiln serve is down before iter $iter — killing any zombies + restarting"
    python3 $RP ssh $POD_ID 'pgrep -x kiln | xargs -r kill -9 2>/dev/null || true; sleep 3'
    python3 $RP bg $POD_ID /tmp/kiln-serve-iter${iter}-restart.log \
      'cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b ./target/release/kiln serve 2>&1'
    sleep 35
    if ! python3 $RP ssh $POD_ID 'curl -sS --max-time 5 http://localhost:8420/v1/adapters >/dev/null 2>&1'; then
      echo "  WARN: kiln serve still down after restart — iter $iter will likely fail"
    fi
  fi

  # Pipe through tee to /tmp/iter${iter}-run.log so we have full output for
  # post-mortem. tail -200 lets us see more context if the iter dies.
  bash "$HERE/run_iter.sh" --iter "$iter" --kind "$KIND" --eval-adapter "$EVAL_ADAPTER" $EXTRA 2>&1 | tee /tmp/iter${iter}-run.log | tail -200 || echo "WARN: run_iter.sh iter ${iter} exited non-zero — continuing loop"

  EVAL_OUT="/tmp/iter${iter}-eval"
  # Pull full summary down for sub-score logging.
  python3 $RP ssh $POD_ID "cat ${EVAL_OUT}/summary.json 2>/dev/null" > /tmp/eval-summary-iter${iter}.json 2>/dev/null || true
  # COMPOSITE/BASELINE are interpolated into a Python expression below; use
  # 'None' (Python literal) when missing, not 'null'. Otherwise the row-log
  # python -c block hits NameError. Discovered iter 10-49 silent failures.
  COMPOSITE=$(python3 -c "import json; d=json.load(open('/tmp/eval-summary-iter${iter}.json')); v=d.get('mean_composite'); print(v if v is not None else 'None')" 2>/dev/null || echo "None")
  BASELINE=$(cat "$HERE/.baseline_composite" 2>/dev/null || echo "None")
  DELTA="None"
  if [ "$COMPOSITE" != "None" ] && [ "$BASELINE" != "None" ]; then
    DELTA=$(python3 -c "print(${COMPOSITE} - ${BASELINE})")
  fi
  echo "  iter ${iter}: composite=${COMPOSITE} baseline=${BASELINE} delta=${DELTA}"
  # Enriched log row including sub_scores from summary.json.
  python3 -c "
import json, datetime, pathlib
try:
    s = json.load(open('/tmp/eval-summary-iter${iter}.json'))
except Exception:
    s = {}
row = {
    'iter': $iter,
    'kind': '$KIND',
    'slug': '$SLUG',
    'composite': $COMPOSITE,
    'baseline_composite': $BASELINE,
    'delta_vs_baseline': $DELTA,
    'status': 'logged' if $COMPOSITE is not None else 'FAILED-no-eval',
    'sub_scores': s.get('mean_sub_scores') or {},
    'class_means': s.get('class_means') or {},
    'rollouts_passed': s.get('rollouts_passed'),
    'n_rollouts': s.get('n_rollouts'),
    'logged_at': datetime.datetime.utcnow().isoformat() + 'Z',
    'pod_id': '$POD_ID',
}
pathlib.Path('$HERE/capability.jsonl').open('a').write(json.dumps(row) + chr(10))
"

  cd "$HERE/../../.."
  git add capabilities/agentic-grpo/pi-diff-patch-apply/capability.jsonl || true
  git commit -m "cap[agentic-grpo/pi-diff-patch-apply]: iter ${iter} ${SLUG} (composite=${COMPOSITE}, Δ=${DELTA})" 2>&1 | tail -1
  git pull --rebase origin main 2>&1 | tail -1 || true
  git push origin main 2>&1 | tail -1 || true
  cd - >/dev/null

  iter=$((iter + 1))
done

echo "================================================="
echo "  DRIVE COMPLETE (next iter $iter)"
echo "================================================="
