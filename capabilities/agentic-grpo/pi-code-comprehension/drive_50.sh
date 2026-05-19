#!/bin/bash
# Drive the 50-iter pi-code-comprehension GRPO loop using recipes.json.
#
# Usage:
#   bash drive_50.sh --pod <pod_id> [--start-iter N] [--stop-iter N]
#
# Reads recipes.json for the per-iter recipe; supports `train_adapter_from`
# pointing to "best" (looks up best iter in capability.jsonl) or "iter-K".
set -euo pipefail

START_ITER=""
STOP_ITER=50

while [ $# -gt 0 ]; do
  case "$1" in
    --pod) POD_ID="$2"; shift 2 ;;
    --start-iter) START_ITER="$2"; shift 2 ;;
    --stop-iter) STOP_ITER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
if [ -z "${POD_ID:-}" ]; then echo "--pod required" >&2; exit 1; fi

export POD_ID
export RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"

# Sync pod env file
cat > /tmp/grpo-pod.env <<EOF
export POD_ID=$POD_ID
export RP=$RP
EOF

next_iter() {
  if [ -n "$START_ITER" ]; then echo "$START_ITER"; return; fi
  python3 -c "
import json
n = 0
try:
    for line in open('$HERE/capability.jsonl'):
        try:
            e = json.loads(line)
            v = e.get('iter')
            if isinstance(v, int):
                n = max(n, v + 1)
        except: pass
except: pass
print(n)
"
}

best_iter_adapter() {
  python3 -c "
import json
best_iter = None; best_score = -1
try:
    for line in open('$HERE/capability.jsonl'):
        try:
            e = json.loads(line)
            ev = e.get('eval') or {}
            score = ev.get('mean_composite')
            it = e.get('iter')
            if score is not None and isinstance(it, int) and it > 0 and score > best_score:
                best_score = score; best_iter = it
        except: pass
except: pass
print(f'pi-cc-iter{best_iter}' if best_iter is not None else '')
"
}

iter_adapter() {
  echo "pi-cc-iter$1"
}

run_iter_n() {
  local n=$1
  echo "==============================================="
  echo "  ITER $n"
  echo "==============================================="
  # Pull recipe
  local recipe=$(python3 -c "
import json
for r in json.load(open('$HERE/recipes.json')):
    if r.get('iter') == $n:
        print(json.dumps(r)); break
")
  if [ -z "$recipe" ]; then
    echo "no recipe for iter $n; falling back to defaults"
    recipe="{\"slug\":\"h-default\",\"num_train\":16,\"num_gens\":4,\"lr\":\"1e-5\",\"filter_var\":\"0.02\",\"rank\":16,\"alpha\":32,\"kind\":\"train\"}"
  fi
  echo "Recipe: $recipe"

  local slug=$(echo "$recipe" | python3 -c "import sys,json; print(json.load(sys.stdin).get('slug','x'))")
  local num_train=$(echo "$recipe" | python3 -c "import sys,json; print(json.load(sys.stdin).get('num_train',16))")
  local num_gens=$(echo "$recipe" | python3 -c "import sys,json; print(json.load(sys.stdin).get('num_gens',4))")
  local lr=$(echo "$recipe" | python3 -c "import sys,json; print(json.load(sys.stdin).get('lr','1e-5'))")
  local filter_var=$(echo "$recipe" | python3 -c "import sys,json; print(json.load(sys.stdin).get('filter_var','0.02'))")
  local rank=$(echo "$recipe" | python3 -c "import sys,json; print(json.load(sys.stdin).get('rank',16))")
  local alpha=$(echo "$recipe" | python3 -c "import sys,json; print(json.load(sys.stdin).get('alpha',32))")
  local kind=$(echo "$recipe" | python3 -c "import sys,json; print(json.load(sys.stdin).get('kind','train'))")
  local epochs=$(echo "$recipe" | python3 -c "import sys,json; print(json.load(sys.stdin).get('epochs',1))")
  local echo_lambda=$(echo "$recipe" | python3 -c "import sys,json; v=json.load(sys.stdin).get('echo_lambda'); print(v or '')")
  local no_echo=$(echo "$recipe" | python3 -c "import sys,json; print('1' if json.load(sys.stdin).get('no_echo') else '0')")
  local no_policy_loss=$(echo "$recipe" | python3 -c "import sys,json; print('1' if json.load(sys.stdin).get('no_policy_loss') else '0')")
  local skip_train=$(echo "$recipe" | python3 -c "import sys,json; print('1' if json.load(sys.stdin).get('skip_train') else '0')")
  local seed=$(echo "$recipe" | python3 -c "import sys,json; print(json.load(sys.stdin).get('seed',3141592653))")
  local train_adapter_from=$(echo "$recipe" | python3 -c "import sys,json; print(json.load(sys.stdin).get('train_adapter_from',''))")
  local eval_adapter_recipe=$(echo "$recipe" | python3 -c "import sys,json; print(json.load(sys.stdin).get('eval_adapter',''))")

  # Resolve train adapter
  local train_adapter=""
  if [ "$train_adapter_from" = "best" ]; then
    train_adapter=$(best_iter_adapter)
  elif [ -n "$train_adapter_from" ] && [ "$train_adapter_from" != "base" ]; then
    if [[ "$train_adapter_from" =~ ^iter-([0-9]+)$ ]]; then
      train_adapter="pi-cc-iter${BASH_REMATCH[1]}"
    fi
  fi
  echo "Train-adapter: ${train_adapter:-(base)}"

  # Resolve eval adapter — defaults to this iter's adapter
  local eval_adapter="pi-cc-iter$n"
  if [ -n "$eval_adapter_recipe" ]; then
    if [ "$eval_adapter_recipe" = "best" ]; then
      eval_adapter=$(best_iter_adapter)
    elif [[ "$eval_adapter_recipe" =~ ^iter-([0-9]+)$ ]]; then
      eval_adapter="pi-cc-iter${BASH_REMATCH[1]}"
    elif [ "$eval_adapter_recipe" = "base" ]; then
      eval_adapter="base"
    fi
  fi
  echo "Eval-adapter: $eval_adapter"

  local cmd="bash $HERE/run_iter.sh --iter $n --kind $kind --num-train-tasks $num_train --num-gens $num_gens --lr $lr --filter-var $filter_var --rank $rank --alpha $alpha --epochs $epochs --seed $seed --max-wall 120"
  if [ "$skip_train" = "1" ]; then cmd="$cmd --skip-train"; fi
  if [ -n "$train_adapter" ] && [ "$skip_train" != "1" ]; then cmd="$cmd --train-adapter $train_adapter"; fi
  if [ "$eval_adapter" != "" ]; then cmd="$cmd --eval-adapter $eval_adapter"; fi
  if [ "$no_echo" = "1" ]; then cmd="$cmd --no-echo"; fi
  if [ -n "$echo_lambda" ]; then cmd="$cmd --echo-lambda $echo_lambda"; fi
  if [ "$no_policy_loss" = "1" ]; then cmd="$cmd --no-policy-loss"; fi

  echo "+ $cmd"
  eval "$cmd" || echo "iter $n run_iter exited non-zero — continuing"

  # Append result, commit + push
  python3 "$HERE/record_iter.py" --iter "$n" --pod "$POD_ID" --kind "$kind" || true
  cd /data/projects/kiln-pi-code-comprehension/kiln
  git add capabilities/agentic-grpo/pi-code-comprehension/capability.jsonl 2>/dev/null || true
  git commit -m "cap[agentic-grpo/pi-code-comprehension]: iter $n ($slug)" || true
  git push origin main 2>&1 | tail -2 || true
  cd "$HERE"
}

iter=$(next_iter)
echo "Starting at iter $iter; stopping at $STOP_ITER"
while [ "$iter" -le "$STOP_ITER" ]; do
  run_iter_n "$iter" || echo "iter $iter failed; continuing"
  iter=$((iter + 1))
done

echo "==============================================="
echo "  DRIVE COMPLETE"
echo "==============================================="
