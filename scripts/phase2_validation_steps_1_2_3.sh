#!/usr/bin/env bash
#
# Bounded Vulkan training smoke retained from the Phase 2 validation work.
#
# The old three-arm environment comparison was removed with the immutable
# Vulkan execution policy. This runs the one supported qualified route with a
# tiny payload and aborts if MemAvailable drops below 8 GiB. The original
# T=918 repro remains operator-driven because it crashed the host twice.
#
# Usage:
#   ./scripts/phase2_validation_steps_1_2_3.sh <model-path>
#
# Exit codes:
#   0  — qualified-policy smoke passed
#   1  — pre-flight check failed (build, memory, or stale processes)
#   2  — training smoke failed

set -euo pipefail

SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --help|-h)
            cat >&2 <<EOF
usage: $0 [--skip-build] <model-path>

  --skip-build  Don't run \`cargo build\` — assume target/release/kiln-server
                is already up to date. Useful for re-runs.
EOF
            exit 0
            ;;
        *)
            MODEL_PATH="$1"
            shift
            ;;
    esac
done

if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "usage: $0 [--skip-build] <model-path>" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model path '$MODEL_PATH' does not exist or is not a directory" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
BASE_URL="${KILN_URL:-http://localhost:8420}"

LOG_DIR="$(mktemp -d -t kiln-phase2-validation-XXXXXX)"
echo ">>> Logs in $LOG_DIR" >&2

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
echo ">>> Pre-flight: required tools" >&2
for tool in cargo curl jq pgrep awk mktemp; do
    if ! command -v "$tool" > /dev/null 2>&1; then
        echo "ERROR: required tool '$tool' not on PATH" >&2
        exit 1
    fi
done

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

if [[ "$SKIP_BUILD" -eq 1 ]]; then
    if [[ ! -x "./target/release/kiln-server" ]]; then
        echo "ERROR: --skip-build set but ./target/release/kiln-server not found" >&2
        exit 1
    fi
    echo ">>> Pre-flight: build (skipped per --skip-build)" >&2
else
    echo ">>> Pre-flight: build" >&2
    PATH="$HOME/.cargo/bin:$PATH" \
        KILN_CARGO_MIN_AVAILABLE_GIB=15 \
        KILN_CARGO_CPU_QUOTA_PERCENT=50 \
        scripts/cargo-bounded.sh build --release -p kiln-server --features vulkan --no-default-features 2>&1 \
        | tee "$LOG_DIR/build.log" >&2
fi

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
        if curl -fs "$BASE_URL/v1/models" > /dev/null 2>&1; then
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
    if ! curl -fsS -X POST "$BASE_URL/v1/training/sft" \
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
        status="$(curl -fs "$BASE_URL/v1/training/queue" 2>/dev/null \
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
# Portable immutable Vulkan route. Environment A/B arms no longer exist.
# -----------------------------------------------------------------------------
if ! run_step "qualified-policy" 'RUST_LOG=info'; then
    echo ">>> Qualified-policy smoke FAILED. Logs in $LOG_DIR" >&2
    exit 2
fi

echo ">>> Qualified-policy smoke passed. Logs in $LOG_DIR" >&2

# Compact summary of the per-step server logs — surfaces the
# acceleration-profile and chunking traces so the operator can confirm
# the right paths engaged without grepping the full server logs.
for step in qualified-policy; do
    echo "" >&2
    echo "--- $step traces ---" >&2
    if [[ -f "$LOG_DIR/$step-server.log" ]]; then
        grep -E "Vulkan training acceleration profile|first chunked dispatch|first sub-chunked dispatch|first call|first dispatch|GPU memory budget" \
            "$LOG_DIR/$step-server.log" | head -20 >&2 || true
    fi
done

echo "" >&2
echo ">>> The original T=918 repro remains a separate operator-driven run." >&2
exit 0
