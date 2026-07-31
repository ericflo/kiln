#!/bin/bash
# Kiln 60-second demo: live LoRA online learning
#
# This is the canonical recording driver for `docs/site/demo/kiln-60s.cast`.
# Run from the kiln repo root via:
#
#   COLUMNS=120 LINES=32 TERM=xterm-256color asciinema rec docs/site/demo/kiln-60s.cast \
#     --title "Kiln 60-second demo: live LoRA online learning" \
#     --idle-time-limit 2 \
#     --command ./docs/site/demo/demo.sh
#
# Prerequisites:
#   - Kiln binary at ./target/release/kiln by default; override KILN_BIN to use
#     an extracted release artifact or another source-built binary.
#   - Model weights at ./Qwen3.5-4B/ (override with KILN_MODEL_PATH if elsewhere).
#   - The chat template disables thinking-mode by default — Qwen3.5-4B will otherwise
#     emit reasoning into a separate `reasoning_content` field and Scene 2/5 read empty.
#     The simplest fix is to replace this Jinja block in the model's chat_template:
#
#       {%- if enable_thinking is defined and enable_thinking is false %}
#           {{- '<think>\n\n</think>\n\n' }}
#       {%- else %}
#           {{- '<think>\n' }}
#       {%- endif %}
#
#     with:
#
#       {{- '<think>\n\n</think>\n\n' }}
#
#     so the rendered prompt always prefills an empty think block.
#   - kiln.example.toml's `inference_memory_fraction` set to ~0.4 so the trainer can
#     allocate scratch space alongside the inference KV cache. The default 0.7 OOMs
#     during the SFT scene on a 48 GB A6000.

set -e

# Default to ./Qwen3.5-4B but allow override.
export KILN_MODEL_PATH="${KILN_MODEL_PATH:-./Qwen3.5-4B}"
KILN_BIN="${KILN_BIN:-./target/release/kiln}"

# Resolve the directory this script lives in so we can find demo-sft.json next to it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_JSON="${SCRIPT_DIR}/demo-sft.json"

# Slow-print a command line as if a human is typing it.
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

# Pause helper.
beat() { sleep "$1"; }

# ------------------------------------------------------------------
# Scene 1 — Cold start
# ------------------------------------------------------------------
typecmd "KILN_MODEL_PATH=${KILN_MODEL_PATH} ${KILN_BIN} serve --config kiln.example.toml &"

# Server logs redirected to a file so they don't bleed into the asciinema TTY.
"${KILN_BIN}" serve --config kiln.example.toml >/tmp/kiln-demo.log 2>&1 &
SRV_PID=$!

# Wait for /health, capped so the recording cannot hang on a broken build.
for i in $(seq 1 180); do
    if curl -sf http://localhost:8420/health -o /dev/null 2>/dev/null; then
        break
    fi
    sleep 0.5
done

# Silent warmup: the patched chat template prefills <think>\n\n</think>\n\n,
# which traps the cold first generation per (adapter, prompt) into 5-9 garbage
# tokens. One throwaway query at temp=0.3 seed=1 consumes the trap; subsequent
# queries with different sampling params (e.g. temp=0.0 greedy or seed=23) come
# out clean. The warmup MUST use sampling params that differ from the visible
# scene so the prefix cache does not just replay the trap'd completion. This
# matches what a user sees on a warm server.
curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"In one short sentence, what is the Kiln inference server?"}],"max_tokens":80,"temperature":0.3,"seed":1}' \
    >/dev/null 2>&1

beat 1.0

# ------------------------------------------------------------------
# Scene 2 — First chat completion (base model, no adapter)
# ------------------------------------------------------------------
typecmd 'curl -s http://localhost:8420/v1/chat/completions \'
typecmd '    -H "Content-Type: application/json" \'
typecmd '    -d '\''{"messages":[{"role":"user","content":"In one short sentence, what is the Kiln inference server?"}],"max_tokens":48,"temperature":0.0}'\'' \'
typecmd '    | jq -r ".choices[0].message.content"'

curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"In one short sentence, what is the Kiln inference server?"}],"max_tokens":48,"temperature":0.0}' \
    | jq -r '.choices[0].message.content'

beat 1.5

# ------------------------------------------------------------------
# Scene 3 — Submit a correction via /v1/train/sft
# ------------------------------------------------------------------
typecmd 'curl -s http://localhost:8420/v1/train/sft \'
typecmd '    -H "Content-Type: application/json" \'
typecmd '    -d @demo-sft.json \'
typecmd '    | jq -c "{job_id, state}"'

curl -s http://localhost:8420/v1/train/sft \
    -H 'Content-Type: application/json' \
    -d @"${SFT_JSON}" \
    | jq -c '{job_id, state}'

beat 1.0

# ------------------------------------------------------------------
# Scene 4 — Watch training complete
# ------------------------------------------------------------------
typecmd 'curl -s http://localhost:8420/v1/train/status | jq -c ".[-1] | {state, adapter_name, current_loss, elapsed_secs}"'

# Poll until the demo job reaches a terminal state. Capped at 240s.
for i in $(seq 1 480); do
    state=$(curl -s http://localhost:8420/v1/train/status \
        | python3 -c 'import sys,json;d=json.load(sys.stdin);demo=[j for j in d if j.get("adapter_name")=="demo-live"];print(demo[-1]["state"] if demo else "none")' 2>/dev/null)
    if [ "$state" = "completed" ] || [ "$state" = "failed" ]; then
        break
    fi
    sleep 0.5
done

curl -s http://localhost:8420/v1/train/status | jq -c '.[-1] | {state, adapter_name, current_loss, elapsed_secs}'

# Silent warmup for the demo adapter — consumes the empty-think trap on its
# cold first generation so Scene 5's visible answer is clean. Warmup uses
# seed=1 (different from Scene 5's seed=23) so the prefix cache does not
# replay the trap'd completion.
curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"adapter":"demo","messages":[{"role":"user","content":"In one short sentence, what is the Kiln inference server?"}],"max_tokens":80,"temperature":0.3,"seed":1}' \
    >/dev/null 2>&1

beat 1.5

# ------------------------------------------------------------------
# Scene 5 — Second chat completion (improved, with adapter)
#
# The "adapter":"demo" field is REQUIRED — kiln does not implicitly route
# requests to the most recently trained adapter, by design.
# ------------------------------------------------------------------
typecmd 'curl -s http://localhost:8420/v1/chat/completions \'
typecmd '    -H "Content-Type: application/json" \'
typecmd '    -d '\''{"adapter":"demo","messages":[{"role":"user","content":"In one short sentence, what is the Kiln inference server?"}],"max_tokens":80,"temperature":0.3,"seed":23}'\'' \'
typecmd '    | jq -r ".choices[0].message.content"'

curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"adapter":"demo","messages":[{"role":"user","content":"In one short sentence, what is the Kiln inference server?"}],"max_tokens":80,"temperature":0.3,"seed":23}' \
    | jq -r '.choices[0].message.content'

beat 1.5

# ------------------------------------------------------------------
# Scene 6 — Adapter list
# ------------------------------------------------------------------
typecmd 'curl -s http://localhost:8420/v1/adapters | jq -c ".available[] | {name, size_bytes, modified_at}"'

curl -s http://localhost:8420/v1/adapters | jq -c '.available[] | {name, size_bytes, modified_at}'

beat 2.0

# Clean shutdown so asciinema's recording terminates promptly.
kill -TERM "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
