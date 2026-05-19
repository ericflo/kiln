#!/bin/bash
# Wait for an iter, then log + backup. Standalone wrapper.
# Usage: process_iter.sh <N>
set -uo pipefail
ITER=$1
source /tmp/pi-faithful.env

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SLUG=$(python3 -c "
import json
hs = json.load(open('hypotheses.json'))
for h in hs:
    if h['iter'] == $ITER:
        print(h['slug']); break
")
FAMILY=$(python3 -c "
import json
hs = json.load(open('hypotheses.json'))
for h in hs:
    if h['iter'] == $ITER:
        print(h['family']); break
")
ARGS=$(python3 -c "
import json
hs = json.load(open('hypotheses.json'))
for h in hs:
    if h['iter'] == $ITER:
        print(json.dumps(h['args'])); break
")

# Wait for eval
python3 $RP wait-file $POD_ID /tmp/iter${ITER}-eval/summary.json --timeout 2400 2>&1 | tail -3 || true

# Fetch eval via clean read
python3 $RP ssh $POD_ID "python3 -c \"print(open('/tmp/iter${ITER}-eval/summary.json').read())\"" > /tmp/iter${ITER}-raw.json 2>&1

# Strip extra data
python3 -c "
import json
data = open('/tmp/iter${ITER}-raw.json').read()
obj, _ = json.JSONDecoder().raw_decode(data)
open('eval-summaries/iter${ITER}-${SLUG}.json','w').write(json.dumps(obj, indent=2))
print('composite:', obj['mean_composite'])
"
rm -f /tmp/iter${ITER}-raw.json

python3 log_iter.py --iter ${ITER} --slug "$SLUG" --family "$FAMILY" \
  --eval-summary eval-summaries/iter${ITER}-${SLUG}.json \
  --adapter "pi-faithful-${SLUG}" \
  --hyperparams "$ARGS" \
  --baseline 0.7237 2>&1 | tail -3

python3 backup_to_b2.py --iter ${ITER} --slug "$SLUG" --adapter "pi-faithful-${SLUG}" --pod $POD_ID 2>&1 | tail -3

# Commit and push
cd /data/projects/kiln-pi-faithful-completion/kiln
git add capabilities/agentic-grpo/pi-faithful-completion/
git commit -m "cap[agentic-grpo/pi-faithful-completion]: iter ${ITER} ${SLUG}" 2>&1 | tail -3
git pull --rebase origin main 2>&1 | tail -3
git push origin main 2>&1 | tail -3
