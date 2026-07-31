#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
oracle_root=${OPENENV_INTEROP_ROOT:-"$repo_root/../miniopenenv"}
cargo_bin=${CARGO_BIN:-cargo}
base_port=${KILN_OPENENV_INTEROP_PORT:-18990}
declare -a exact_text_names=(
    measure algebra change space discrete chance evidence reason
)
declare -a server_names=(
    counter bandit cartpole snake g2048 wordle blackjack maze pong connect4 kuhn
    dyna inducto logic filtro "${exact_text_names[@]}"
)
declare -a server_pids=()
declare -a log_files=()
declare -a arcade_urls=()
declare -a exact_text_urls=()
declare -A server_urls=()
declare -A exact_text_name_set=()
for name in "${exact_text_names[@]}"; do
    exact_text_name_set["$name"]=1
done
forbidden_oracle_namespace='MINI''OPENENV'

if rg --quiet --fixed-strings "$forbidden_oracle_namespace" "$repo_root/crates"; then
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

declare -a server_targets=()
for name in "${server_names[@]}"; do
    server_targets+=("build/rel/bin/$name")
done
# Always ask the oracle's own build graph to refresh the binaries. A locally
# cached executable must never make source-level protocol drift invisible.
make -C "$oracle_root" "${server_targets[@]}"

for index in "${!server_names[@]}"; do
    name=${server_names[$index]}
    binary="$oracle_root/build/rel/bin/$name"
    port=$((base_port + index))
    log_file=$(mktemp "${TMPDIR:-/tmp}/kiln-miniopenenv-$name.XXXXXX.log")
    log_files+=("$log_file")
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
    server_urls["$name"]="http://127.0.0.1:$port"
    if [[ "$name" != counter ]]; then
        arcade_urls+=("http://127.0.0.1:$port")
    fi
    if [[ -n ${exact_text_name_set[$name]:-} ]]; then
        exact_text_urls+=("http://127.0.0.1:$port")
    fi
done

(
    cd "$repo_root"
    "$cargo_bin" test -p kiln-server --no-default-features \
        api::openenv::tests --lib
    "$cargo_bin" test -p kiln-server --no-default-features \
        openenv_cli::tests --lib
    "$cargo_bin" test -p kiln-server --no-default-features \
        openenv_credentials::tests --lib
    KILN_OPENENV_INTEROP_ARCADE_URLS="$(IFS=,; echo "${arcade_urls[*]}")" \
    KILN_OPENENV_INTEROP_EXACT_TEXT_URLS="$(IFS=,; echo "${exact_text_urls[*]}")" \
    KILN_OPENENV_INTEROP_COUNTER_URL="${server_urls[counter]}" \
    KILN_OPENENV_INTEROP_BANDIT_URL="${server_urls[bandit]}" \
    KILN_OPENENV_INTEROP_CONNECT4_URL="${server_urls[connect4]}" \
    KILN_OPENENV_INTEROP_MAZE_URL="${server_urls[maze]}" \
    KILN_OPENENV_INTEROP_WORDLE_URL="${server_urls[wordle]}" \
        "$cargo_bin" test -p kiln-openenv --test miniopenenv_interop -- \
        --ignored --test-threads=1
    KILN_OPENENV_INTEROP_BANDIT_URL="${server_urls[bandit]}" \
        "$cargo_bin" test -p kiln-server --no-default-features \
        --test openenv_training_interop -- --ignored --exact \
        collects_submits_verifies_and_replays_a_real_arcade_batch
    KILN_OPENENV_INTEROP_BANDIT_URL="${server_urls[bandit]}" \
        "$cargo_bin" test -p kiln-server --no-default-features --lib \
        openenv_evaluation::tests::paired_evaluation_drives_a_real_openenv_bandit \
        -- --ignored --exact
)
