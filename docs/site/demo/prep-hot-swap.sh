#!/bin/bash
# Pre-stages the two adapters that demo-hot-swap.sh requires:
#   ./adapters/demo/    — the canonical Kiln-as-software adapter
#   ./adapters/formal/  — a more clipped, formal phrasing of the same facts
#
# Run this ONCE on the recording host before opening asciinema. Both jobs
# train on a single A6000 in well under two minutes. The recording itself
# then assumes both adapters already exist on disk and skips the train scene.

set -e

export KILN_MODEL_PATH="${KILN_MODEL_PATH:-./Qwen3.5-4B}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_SFT="${SCRIPT_DIR}/demo-sft.json"
FORMAL_SFT="${SCRIPT_DIR}/demo-sft-formal.json"

./target/release/kiln serve --config kiln.example.toml >/tmp/kiln-prep.log 2>&1 &
SRV_PID=$!
trap 'kill -TERM "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true' EXIT

# Wait for the server to be live.
for i in $(seq 1 240); do
    if curl -sf http://localhost:8420/health -o /dev/null 2>/dev/null; then
        break
    fi
    sleep 0.5
done

submit_and_wait() {
    local fixture="$1"
    local adapter="$2"

    echo ">>> training adapter=${adapter} from ${fixture}"
    curl -s http://localhost:8420/v1/train/sft \
        -H 'Content-Type: application/json' \
        -d @"${fixture}" \
        | jq -c '{job_id, state}'

    for i in $(seq 1 360); do
        state=$(curl -s http://localhost:8420/v1/train/status \
            | python3 -c "import sys,json
data=json.load(sys.stdin)
matches=[j for j in data if j.get('adapter_name')=='${adapter}']
print(matches[-1]['state'] if matches else 'none')")
        if [ "$state" = "completed" ]; then
            echo ">>> ${adapter}: done"
            break
        fi
        if [ "$state" = "failed" ]; then
            echo ">>> ${adapter}: FAILED"
            return 1
        fi
        sleep 0.5
    done
}

submit_and_wait "${DEMO_SFT}"   "demo"
submit_and_wait "${FORMAL_SFT}" "formal"

echo
echo "Both adapters staged. Recording host is ready."
