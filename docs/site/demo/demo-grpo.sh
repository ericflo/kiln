#!/bin/bash
# Kiln GRPO with a custom reward: ship a tiny offline-generated batch of
# scored completions, run a few GRPO steps, hot-swap the resulting LoRA.
# Same pattern as the SFT demo, but the supervision signal is a reward
# function — RL over HTTP, on the same single GPU that is serving traffic.
#
# Run from the kiln repo root via:
#
#   COLUMNS=120 LINES=32 TERM=xterm-256color asciinema rec docs/site/demo/grpo.cast \
#     --title "Kiln: GRPO with a custom reward, online over HTTP" \
#     --idle-time-limit 2 \
#     --command ./docs/site/demo/demo-grpo.sh
#
# Prerequisites:
#   - ./target/release/kiln (--features cuda) + ./Qwen3.5-4B/
#   - kiln.example.toml with `inference_memory_fraction` ~0.4 so the trainer
#     can grab scratch space alongside the inference KV cache.

set -e

export KILN_MODEL_PATH="${KILN_MODEL_PATH:-./Qwen3.5-4B}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_JSON="${SCRIPT_DIR}/demo-grpo.json"

typecmd() {
    local cmd="$1"
    printf '$ '
    sleep 0.4
    local i
    for ((i=0; i<${#cmd}; i++)); do
        printf '%s' "${cmd:i:1}"
        sleep 0.018
    done
    printf '\n'
}

beat() { sleep "$1"; }

# ------------------------------------------------------------------
# Scene 1 — Boot kiln. Banner + spinner + ready.
# ------------------------------------------------------------------
typecmd './target/release/kiln serve --config kiln.example.toml &'

# Server logs redirected to a file so they don't bleed into the asciinema TTY.
./target/release/kiln serve --config kiln.example.toml >/tmp/kiln-grpo.log 2>&1 &
SRV_PID=$!

for i in $(seq 1 180); do
    if curl -sf http://localhost:8420/health -o /dev/null 2>/dev/null; then
        break
    fi
    sleep 0.5
done

beat 1.0

# ------------------------------------------------------------------
# Scene 2 — Peek at the reward batch. Two prompts, four completions each,
# good answers labelled +1.0, off-topic answers labelled -1.0.
# ------------------------------------------------------------------
typecmd 'jq ".groups[0].completions[] | {reward, text: .text[:60]}" docs/site/demo/demo-grpo.json'

jq '.groups[0].completions[] | {reward, text: (.text[:60])}' "${GRPO_JSON}"

beat 2.0

# ------------------------------------------------------------------
# Scene 3 — Submit the GRPO job.
# ------------------------------------------------------------------
typecmd 'curl -s http://localhost:8420/v1/train/grpo \'
typecmd '    -H "Content-Type: application/json" \'
typecmd '    -d @docs/site/demo/demo-grpo.json \'
typecmd '    | jq -c "{job_id, state}"'

curl -s http://localhost:8420/v1/train/grpo \
    -H 'Content-Type: application/json' \
    -d @"${GRPO_JSON}" \
    | jq -c '{job_id, state}'

beat 1.0

# ------------------------------------------------------------------
# Scene 4 — Watch GRPO complete.
# ------------------------------------------------------------------
typecmd 'curl -s http://localhost:8420/v1/train/status | jq -c ".[-1] | {state, adapter_name, current_loss, elapsed_secs}"'

for i in $(seq 1 240); do
    state=$(curl -s http://localhost:8420/v1/train/status \
        | python3 -c 'import sys,json
data=json.load(sys.stdin)
matches=[j for j in data if j.get("adapter_name")=="grpo-live"]
print(matches[-1]["state"] if matches else "none")' 2>/dev/null)
    if [ "$state" = "completed" ] || [ "$state" = "failed" ]; then
        break
    fi
    sleep 0.5
done

curl -s http://localhost:8420/v1/train/status | jq -c '.[-1] | {state, adapter_name, current_loss, elapsed_secs}'

# Silent warmup for the grpo-demo adapter — consumes the empty-think trap on
# its cold first generation so Scene 5's visible answer is clean. Warmup uses
# seed=1 (different from Scene 5's seed=67) so the prefix cache does not
# replay the trap'd completion.
curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"adapter":"grpo-demo","messages":[{"role":"user","content":"In one short sentence, what is the Kiln inference server?"}],"max_tokens":80,"temperature":0.3,"seed":1}' \
    >/dev/null 2>&1

beat 1.5

# ------------------------------------------------------------------
# Scene 5 — Same prompt as the high-reward completions. Adapter routes RL
# preference into the next generation.
# ------------------------------------------------------------------
typecmd 'curl -s http://localhost:8420/v1/chat/completions \'
typecmd '    -H "Content-Type: application/json" \'
typecmd '    -d '\''{"adapter":"grpo-demo","messages":[{"role":"user","content":"In one short sentence, what is the Kiln inference server?"}],"max_tokens":80,"temperature":0.3,"seed":67}'\'' \'
typecmd '    | jq -r ".choices[0].message.content"'

curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"adapter":"grpo-demo","messages":[{"role":"user","content":"In one short sentence, what is the Kiln inference server?"}],"max_tokens":80,"temperature":0.3,"seed":67}' \
    | jq -r '.choices[0].message.content'

beat 2.0

kill -TERM "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
