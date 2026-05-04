#!/bin/bash
# Kiln live adapter hot-swap: one server, three answers. Same prompt, no
# adapter / adapter=demo / adapter=formal — the kiln server hot-swaps the
# LoRA per request without reloading the base model.
#
# Run from the kiln repo root via:
#
#   COLUMNS=120 LINES=32 TERM=xterm-256color asciinema rec docs/site/demo/hot-swap.cast \
#     --title "Kiln: hot-swap LoRA adapters per request" \
#     --idle-time-limit 2 \
#     --command ./docs/site/demo/demo-hot-swap.sh
#
# Prerequisites:
#   - ./target/release/kiln (--features cuda) + ./Qwen3.5-4B/
#   - kiln.example.toml at the repo root
#   - Two adapters PRE-STAGED on disk under ./adapters/:
#         ./adapters/demo/      — kiln-as-software answer (from demo-sft.json)
#         ./adapters/formal/    — formal/concise answer (from demo-sft-formal.json)
#     The companion script docs/site/demo/prep-hot-swap.sh trains both
#     before recording so the asciicast itself stays under 60s.

set -e

export KILN_MODEL_PATH="${KILN_MODEL_PATH:-./Qwen3.5-4B}"

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
# Scene 1 — Boot kiln. The new structured banner + ready line.
# ------------------------------------------------------------------
typecmd './target/release/kiln serve --config kiln.example.toml &'

# Server logs redirected to a file so they don't bleed into the asciinema TTY.
./target/release/kiln serve --config kiln.example.toml >/tmp/kiln-hotswap.log 2>&1 &
SRV_PID=$!

for i in $(seq 1 180); do
    if curl -sf http://localhost:8420/health -o /dev/null 2>/dev/null; then
        break
    fi
    sleep 0.5
done

# NOTE: per-adapter warmup happens INLINE before each visible scene below.
# Batching all three warmups here doesn't work — adapter switching between the
# warmup loop and the visible scene re-traps the model on the empty-think
# prefill, so each adapter must be warmed immediately before it is queried.

beat 1.0

# ------------------------------------------------------------------
# Scene 2 — List the adapters already on disk.
# ------------------------------------------------------------------
typecmd 'curl -s http://localhost:8420/v1/adapters | jq -c ".available[] | {name, size_bytes}"'

curl -s http://localhost:8420/v1/adapters | jq -c '.available[] | {name, size_bytes}'

beat 1.5

# ------------------------------------------------------------------
# Scene 3 — Same prompt, no adapter (base model).
# ------------------------------------------------------------------
typecmd '# Same prompt across all three calls. Watch the answer change.'
typecmd 'PROMPT='\''In one sentence, what is the Kiln inference server?'\'

PROMPT='In one sentence, what is the Kiln inference server?'

# Inline warmup: consumes the empty-think trap on this adapter immediately
# before the visible call. Different seed (1) so the prefix cache won't
# replay the trapped tokens into the visible answer (seed 23).
curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":80,\"temperature\":0.3,\"seed\":1}" \
    >/dev/null 2>&1

typecmd 'curl -s http://localhost:8420/v1/chat/completions \'
typecmd '    -H "Content-Type: application/json" \'
typecmd '    -d '\''{"messages":[{"role":"user","content":"'\''"$PROMPT"'\''"}],"max_tokens":80,"temperature":0.3,"seed":23}'\'' \'
typecmd '    | jq -r ".choices[0].message.content"'

curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":80,\"temperature\":0.3,\"seed\":23}" \
    | jq -r '.choices[0].message.content'

beat 1.5

# ------------------------------------------------------------------
# Scene 4 — Same prompt, adapter=demo.
# ------------------------------------------------------------------
# Inline warmup for the demo adapter, seed=1 (different from visible seed 17).
curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"adapter\":\"demo\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":80,\"temperature\":0.3,\"seed\":1}" \
    >/dev/null 2>&1

typecmd 'curl -s http://localhost:8420/v1/chat/completions \'
typecmd '    -H "Content-Type: application/json" \'
typecmd '    -d '\''{"adapter":"demo","messages":[{"role":"user","content":"'\''"$PROMPT"'\''"}],"max_tokens":80,"temperature":0.3,"seed":17}'\'' \'
typecmd '    | jq -r ".choices[0].message.content"'

curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"adapter\":\"demo\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":80,\"temperature\":0.3,\"seed\":17}" \
    | jq -r '.choices[0].message.content'

beat 1.5

# ------------------------------------------------------------------
# Scene 5 — Same prompt, adapter=formal. No restart, no second model load.
# ------------------------------------------------------------------
# Inline warmup for the formal adapter, seed=1 (different from visible seed 29).
curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"adapter\":\"formal\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":80,\"temperature\":0.3,\"seed\":1}" \
    >/dev/null 2>&1

typecmd 'curl -s http://localhost:8420/v1/chat/completions \'
typecmd '    -H "Content-Type: application/json" \'
typecmd '    -d '\''{"adapter":"formal","messages":[{"role":"user","content":"'\''"$PROMPT"'\''"}],"max_tokens":80,"temperature":0.3,"seed":29}'\'' \'
typecmd '    | jq -r ".choices[0].message.content"'

curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"adapter\":\"formal\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":80,\"temperature\":0.3,\"seed\":29}" \
    | jq -r '.choices[0].message.content'

beat 2.0

kill -TERM "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
