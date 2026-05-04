#!/bin/bash
# Kiln OpenAI client drop-in: the official `openai` Python SDK, base_url
# pointed at localhost:8420, streaming a chat completion. Same code anyone
# is already running against api.openai.com — the only line that changes
# is `base_url`.
#
# Run from the kiln repo root via:
#
#   COLUMNS=120 LINES=32 TERM=xterm-256color asciinema rec docs/site/demo/openai.cast \
#     --title "Kiln: drop-in OpenAI Python client, just change base_url" \
#     --idle-time-limit 2 \
#     --command ./docs/site/demo/demo-openai.sh
#
# Prerequisites:
#   - ./target/release/kiln (--features cuda) + ./Qwen3.5-4B/
#   - kiln.example.toml at the repo root
#   - Python 3.10+ with `openai` installed: `pip install openai`

set -e

export KILN_MODEL_PATH="${KILN_MODEL_PATH:-./Qwen3.5-4B}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENAI_SCRIPT="${SCRIPT_DIR}/demo-openai.py"

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
# Scene 1 — Boot kiln. Standard new banner.
# ------------------------------------------------------------------
typecmd './target/release/kiln serve --config kiln.example.toml &'

# Server logs redirected to a file so they don't bleed into the asciinema TTY.
./target/release/kiln serve --config kiln.example.toml >/tmp/kiln-openai.log 2>&1 &
SRV_PID=$!

for i in $(seq 1 180); do
    if curl -sf http://localhost:8420/health -o /dev/null 2>/dev/null; then
        break
    fi
    sleep 0.5
done

beat 1.0

# ------------------------------------------------------------------
# Scene 2 — Show the script. Plain OpenAI SDK code, base_url is the only edit.
# ------------------------------------------------------------------
typecmd 'cat docs/site/demo/demo-openai.py'
cat "${OPENAI_SCRIPT}"

beat 2.0

# ------------------------------------------------------------------
# Scene 3 — Run it. Streaming tokens hit the prompt one-by-one.
# ------------------------------------------------------------------
typecmd 'python3 docs/site/demo/demo-openai.py'

python3 "${OPENAI_SCRIPT}"

beat 2.0

kill -TERM "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
