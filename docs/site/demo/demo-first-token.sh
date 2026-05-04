#!/bin/bash
# Kiln cold-start to first token: a single-take asciicast that opens with the
# new structured banner, watches the model load, then streams a chat completion
# token-by-token so the viewer sees first-token latency and per-token throughput.
#
# Run from the kiln repo root via:
#
#   COLUMNS=120 LINES=32 TERM=xterm-256color asciinema rec docs/site/demo/first-token.cast \
#     --title "Kiln cold start to first streamed token" \
#     --idle-time-limit 2 \
#     --command ./docs/site/demo/demo-first-token.sh
#
# Prerequisites:
#   - ./target/release/kiln binary (--features cuda)
#   - ./Qwen3.5-4B/ weights
#   - kiln.example.toml at the repo root

set -e

export KILN_MODEL_PATH="${KILN_MODEL_PATH:-./Qwen3.5-4B}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAM_PARSER="${SCRIPT_DIR}/demo-stream-parser.py"

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
# Scene 1 — Boot. The new structured banner + spinner + ready line.
# ------------------------------------------------------------------
typecmd './target/release/kiln serve --config kiln.example.toml &'

# Server logs redirected to a file so they don't bleed into the asciinema TTY.
./target/release/kiln serve --config kiln.example.toml >/tmp/kiln-first-token.log 2>&1 &
SRV_PID=$!

for i in $(seq 1 180); do
    if curl -sf http://localhost:8420/health -o /dev/null 2>/dev/null; then
        break
    fi
    sleep 0.5
done

beat 1.0

# ------------------------------------------------------------------
# Scene 2 — Streaming chat completion. Tokens land one-by-one.
# ------------------------------------------------------------------
typecmd 'curl -sN http://localhost:8420/v1/chat/completions \'
typecmd '    -H "Content-Type: application/json" \'
typecmd '    -d '\''{"messages":[{"role":"user","content":"In two short sentences, why is pure-Rust LLM inference exciting?"}],"max_tokens":96,"temperature":0.3,"seed":13,"stream":true}'\'' \'
typecmd '    | python3 docs/site/demo/demo-stream-parser.py'

curl -sN http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"In two short sentences, why is pure-Rust LLM inference exciting?"}],"max_tokens":96,"temperature":0.3,"seed":13,"stream":true}' \
    | python3 "${STREAM_PARSER}"

beat 2.0

kill -TERM "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
