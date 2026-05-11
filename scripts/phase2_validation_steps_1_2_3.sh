#!/usr/bin/env bash
#
# Phase 2 hardware validation — Steps 1 and 2 from the runbook
# (`docs/audits/phase2_hardware_validation_runbook_2026-05-11.md`).
#
# Step 1: smallest payload exercising FLCE auto-engagement at
#         active_count >= 16, with all KILN_VULKAN_LINEAR* defaults.
# Step 2: same payload with KILN_VULKAN_LINEAR=1 — first run that
#         exercises the in-op chunking against the SFT path.
# Step 3: same payload with KILN_VULKAN_LINEAR=1 + KILN_VULKAN_SDPA=1
#         — full-attn matmuls go through the new SDPA F32 kernel.
#         At T~256 the per-dispatch SDPA work is ~134 MFLOP × 8
#         layers, well under any safety threshold.
#
# All three steps are bounded (1 example × 1 epoch, T~256, ~30s each)
# and abort if MemAvailable drops below 8 GiB. Step 4 (the original
# T=918 repro) is still operator-driven per the runbook — running it
# autonomously is what crashed the host twice.
#
# Usage:
#   ./scripts/phase2_validation_step1_step2.sh <model-path>
#
# Exit codes:
#   0  — all three steps passed
#   1  — pre-flight check failed (build, memory, or stale processes)
#   2  — Step 1 failed
#   3  — Step 2 failed
#   4  — Step 3 failed

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <model-path>" >&2
    exit 1
fi
MODEL_PATH="$1"
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model path '$MODEL_PATH' does not exist or is not a directory" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$(mktemp -d -t kiln-phase2-validation-XXXXXX)"
echo ">>> Logs in $LOG_DIR" >&2

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
echo ">>> Pre-flight: orphan kiln processes" >&2
if pgrep -f "kiln-server\|target/release/kiln" >/dev/null; then
    echo "ERROR: stale kiln processes — kill them before running" >&2
    pgrep -af "kiln-server\|target/release/kiln" >&2
    exit 1
fi

echo ">>> Pre-flight: memory" >&2
mem_avail_kb="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
mem_avail_gib=$(( mem_avail_kb / 1024 / 1024 ))
if [[ "$mem_avail_gib" -lt 25 ]]; then
    echo "WARNING: MemAvailable ${mem_avail_gib} GiB < 25 GiB — restart the desktop session?" >&2
fi

echo ">>> Pre-flight: build" >&2
cargo build --release -p kiln-server --features vulkan --no-default-features 2>&1 \
    | tee "$LOG_DIR/build.log" >&2

# -----------------------------------------------------------------------------
# Tiny SFT payload
# -----------------------------------------------------------------------------
SFT_FILE="$LOG_DIR/sft-tiny.jsonl"
cat > "$SFT_FILE" <<'EOF'
{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hello! How can I help you today?"}]}
EOF

# -----------------------------------------------------------------------------
# Memory watchdog — sets WATCHDOG_TRIPPED if MemAvailable drops
# below 8 GiB while a training run is in progress.
# -----------------------------------------------------------------------------
WATCHDOG_TRIP_FILE="$LOG_DIR/watchdog-tripped"
start_watchdog() {
    rm -f "$WATCHDOG_TRIP_FILE"
    {
        while true; do
            sleep 5
            kb="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
            if [[ "$kb" -lt $((8 * 1024 * 1024)) ]]; then
                echo "MemAvailable dropped below 8 GiB ($((kb / 1024 / 1024)) GiB); aborting"
                touch "$WATCHDOG_TRIP_FILE"
                # Send SIGTERM to the kiln server — give it a chance
                # to flush logs before SIGKILL.
                pkill -TERM -f "target/release/kiln-server" || true
                sleep 2
                pkill -KILL -f "target/release/kiln-server" || true
                return
            fi
        done
    } >> "$LOG_DIR/watchdog.log" 2>&1 &
    echo $!
}

# -----------------------------------------------------------------------------
# Run a single SFT submission against a freshly-started kiln server.
#
# Args: 1=step-name, 2=env-var-block (multi-line, set -a then unset)
# Returns 0 on pass, non-zero on fail.
# -----------------------------------------------------------------------------
run_step() {
    local step_name="$1"
    local env_block="$2"

    echo ">>> Step '$step_name' starting" >&2

    local server_log="$LOG_DIR/$step_name-server.log"
    local curl_log="$LOG_DIR/$step_name-curl.log"

    # Start the server with the requested env vars.
    (
        set -a
        eval "$env_block"
        set +a
        exec ./target/release/kiln-server --model "$MODEL_PATH"
    ) > "$server_log" 2>&1 &
    local server_pid=$!

    # Wait for the server to come up (max 60s).
    local ready=0
    for _ in $(seq 1 60); do
        if curl -fs http://localhost:8080/v1/models > /dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 1
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "ERROR: server never came up; tail of log:" >&2
        tail -20 "$server_log" >&2
        kill -KILL "$server_pid" 2>/dev/null || true
        return 1
    fi

    local watchdog_pid
    watchdog_pid=$(start_watchdog)

    # Submit the SFT job.
    if ! curl -fsS -X POST http://localhost:8080/v1/training/sft \
        -H 'Content-Type: application/json' \
        -d "{\"file\":\"$SFT_FILE\",\"epochs\":1,\"adapter\":\"phase2-validation-$step_name\"}" \
        > "$curl_log" 2>&1; then
        echo "ERROR: SFT submission failed; tail of curl log:" >&2
        tail -10 "$curl_log" >&2
        kill -TERM "$server_pid" 2>/dev/null || true
        kill -TERM "$watchdog_pid" 2>/dev/null || true
        return 1
    fi

    # Poll training status until done (max 5 minutes).
    local job_id
    job_id="$(jq -r '.id // .job_id // empty' "$curl_log" 2>/dev/null || true)"
    local done_at=$(( $(date +%s) + 300 ))
    while [[ "$(date +%s)" -lt "$done_at" ]]; do
        if [[ -f "$WATCHDOG_TRIP_FILE" ]]; then
            echo "ERROR: watchdog tripped during step '$step_name'" >&2
            kill -TERM "$server_pid" 2>/dev/null || true
            return 1
        fi
        sleep 3
        local status
        status="$(curl -fs http://localhost:8080/v1/training/queue 2>/dev/null \
            | jq -r --arg id "$job_id" '
                (.completed[]? | select(.id == $id) | .state) //
                (.running.id == $id | if . then "running" else empty end)' 2>/dev/null \
            || true)"
        if [[ "$status" == "completed" ]] || [[ "$status" == "succeeded" ]]; then
            break
        fi
        if [[ "$status" == "failed" ]] || [[ "$status" == "errored" ]]; then
            echo "ERROR: training job failed; server log tail:" >&2
            tail -30 "$server_log" >&2
            kill -TERM "$server_pid" 2>/dev/null || true
            kill -TERM "$watchdog_pid" 2>/dev/null || true
            return 1
        fi
    done

    # Shut down server + watchdog.
    kill -TERM "$server_pid" 2>/dev/null || true
    kill -TERM "$watchdog_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true

    if [[ -f "$WATCHDOG_TRIP_FILE" ]]; then
        return 1
    fi
    echo ">>> Step '$step_name' passed" >&2
    return 0
}

# -----------------------------------------------------------------------------
# Step 1: defaults — KILN_VULKAN_LINEAR=0, KILN_VULKAN_SDPA=0,
# FLCE auto-engagement at active_count >= 16.
# -----------------------------------------------------------------------------
if ! run_step "step1-defaults" 'RUST_LOG=info'; then
    echo ">>> Step 1 FAILED. Logs in $LOG_DIR" >&2
    exit 2
fi

# -----------------------------------------------------------------------------
# Step 2: KILN_VULKAN_LINEAR=1 — chunked dispatch on SFT path.
# -----------------------------------------------------------------------------
if ! run_step "step2-vulkan-linear" 'RUST_LOG=info
KILN_VULKAN_LINEAR=1'; then
    echo ">>> Step 2 FAILED. Logs in $LOG_DIR" >&2
    exit 3
fi

# -----------------------------------------------------------------------------
# Step 3: KILN_VULKAN_LINEAR=1 + KILN_VULKAN_SDPA=1 — full-attn
# prefill matmuls now go through the new SDPA F32 kernel.
# -----------------------------------------------------------------------------
if ! run_step "step3-vulkan-sdpa" 'RUST_LOG=info
KILN_VULKAN_LINEAR=1
KILN_VULKAN_SDPA=1'; then
    echo ">>> Step 3 FAILED. Logs in $LOG_DIR" >&2
    exit 4
fi

echo ">>> Steps 1, 2, 3 passed. Logs in $LOG_DIR" >&2
echo ">>> Next: run Step 4 (original /tmp/sft-data.jsonl repro) per runbook." >&2
exit 0
