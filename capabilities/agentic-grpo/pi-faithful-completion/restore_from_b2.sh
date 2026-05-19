#!/bin/bash
# Restore B2-backed adapters into /tmp/iter<N>-adapter/<name>/ on the pod.
#
# Usage: restore_from_b2.sh
#   reads names from B2 and reconstructs the per-iter dirs that drive_iter.py
#   resolves BEST/PREV against.
set -uo pipefail

source /tmp/pi-faithful.env

cd /tmp

# List adapters in B2
B2_DIR="b2://clouderic/capabilities/pi-faithful-completion/adapters"
ADAPTERS=$(b2 ls "$B2_DIR/" | xargs -L1 basename | sed 's/\.tar\.gz$//' | grep -v '^$')

echo "Adapters in B2:"
echo "$ADAPTERS"
echo "---"

# Determine iter number for each by querying capability.jsonl
CAPLOG="/data/projects/kiln-pi-faithful-completion/kiln/capabilities/agentic-grpo/pi-faithful-completion/capability.jsonl"

# Push capability.jsonl to pod
python3 $RP upload $POD_ID "$CAPLOG" /tmp/capability.jsonl 2>&1 | tail -3

for ADAPTER in $ADAPTERS; do
  # Find iter for this adapter
  ITER=$(python3 -c "
import json
for line in open('$CAPLOG'):
    r = json.loads(line)
    if r.get('adapter') == '$ADAPTER':
        print(r['iter'])
        break
")
  if [ -z "$ITER" ]; then
    echo "skip $ADAPTER (no iter found in capability.jsonl)"
    continue
  fi
  REMOTE_PATH="/tmp/iter${ITER}-adapter/${ADAPTER}"
  echo "Restoring $ADAPTER -> $REMOTE_PATH"
  # Download from B2 locally
  rm -f "/tmp/${ADAPTER}.tar.gz"
  b2 file download "$B2_DIR/${ADAPTER}.tar.gz" "/tmp/${ADAPTER}.tar.gz" 2>&1 | tail -3
  # Upload to pod
  python3 $RP upload $POD_ID "/tmp/${ADAPTER}.tar.gz" "/tmp/${ADAPTER}.tar.gz" 2>&1 | tail -3
  # Extract on pod
  python3 $RP ssh $POD_ID "mkdir -p /tmp/iter${ITER}-adapter && tar xzf /tmp/${ADAPTER}.tar.gz -C /tmp/iter${ITER}-adapter && ls /tmp/iter${ITER}-adapter/${ADAPTER}/" 2>&1 | tail -3
  rm -f "/tmp/${ADAPTER}.tar.gz"
done

# Symlink the adapters into the kiln serve adapter dir
python3 $RP ssh $POD_ID 'mkdir -p /workspace/qwen3.5-4b/adapters
for d in /tmp/iter*-adapter/pi-faithful-*; do
  name=$(basename $d)
  ln -sfn $d /workspace/qwen3.5-4b/adapters/$name
done
ls /workspace/qwen3.5-4b/adapters/'
