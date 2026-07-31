#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
oracle_root=${OPENENV_INTEROP_ROOT:-"$repo_root/../miniopenenv"}
cargo_bin=${CARGO_BIN:-cargo}
base_port=${KILN_OPENENV_INTEROP_PORT:-18990}
declare -a server_names=(counter bandit connect4 maze wordle)
declare -a server_pids=()
declare -a log_files=()

if rg --quiet 'MINIOPENENV' "$repo_root/crates"; then
    echo "miniopenenv-specific environment namespaces are forbidden in Kiln crates; use generic OPENENV_INTEROP test controls" >&2
    exit 1
fi
if rg --quiet --ignore-case 'miniopenenv' "$repo_root/crates" -g '**/src/**'; then
    echo "production Kiln Rust sources must remain implementation-neutral and may not branch on the protocol oracle" >&2
    exit 1
fi

cleanup() {
    for pid in "${server_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            wait "$pid" 2>/dev/null || true
        fi
    done
    rm -f "${log_files[@]}"
}
trap cleanup EXIT

for index in "${!server_names[@]}"; do
    name=${server_names[$index]}
    binary="$oracle_root/build/rel/bin/$name"
    port=$((base_port + index))
    log_file=$(mktemp "${TMPDIR:-/tmp}/kiln-miniopenenv-$name.XXXXXX.log")
    log_files+=("$log_file")
    if [[ ! -x "$binary" ]]; then
        make -C "$oracle_root" "build/rel/bin/$name"
    fi
    extra=()
    if [[ "$name" == bandit ]]; then
        extra=(--max-sessions 1)
    fi
    "$binary" --host 127.0.0.1 --port "$port" "${extra[@]}" >"$log_file" 2>&1 &
    server_pids+=("$!")
    for _ in $(seq 1 100); do
        if curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null; then
            break
        fi
        if ! kill -0 "${server_pids[-1]}" 2>/dev/null; then
            sed -n '1,120p' "$log_file" >&2
            exit 1
        fi
        sleep 0.05
    done
    curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null
done

(
    cd "$repo_root"
    KILN_OPENENV_INTEROP_COUNTER_URL="http://127.0.0.1:$base_port" \
    KILN_OPENENV_INTEROP_BANDIT_URL="http://127.0.0.1:$((base_port + 1))" \
    KILN_OPENENV_INTEROP_CONNECT4_URL="http://127.0.0.1:$((base_port + 2))" \
    KILN_OPENENV_INTEROP_MAZE_URL="http://127.0.0.1:$((base_port + 3))" \
    KILN_OPENENV_INTEROP_WORDLE_URL="http://127.0.0.1:$((base_port + 4))" \
        "$cargo_bin" test -p kiln-openenv --test miniopenenv_interop -- --ignored
    KILN_OPENENV_INTEROP_BANDIT_URL="http://127.0.0.1:$((base_port + 1))" \
        "$cargo_bin" test -p kiln-server --no-default-features \
        --test openenv_training_interop -- --ignored --exact \
        collects_submits_verifies_and_replays_a_real_arcade_batch
)
