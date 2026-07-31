#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
miniopenenv_root=${MINIOPENENV_ROOT:-"$repo_root/../miniopenenv"}
counter_bin="$miniopenenv_root/build/rel/bin/counter"
port=${KILN_MINIOPENENV_PORT:-18990}
cargo_bin=${CARGO_BIN:-cargo}
url="http://127.0.0.1:$port"
log_file=$(mktemp "${TMPDIR:-/tmp}/kiln-miniopenenv.XXXXXX.log")
counter_pid=

cleanup() {
    if [[ -n "$counter_pid" ]] && kill -0 "$counter_pid" 2>/dev/null; then
        kill "$counter_pid"
        wait "$counter_pid" 2>/dev/null || true
    fi
    rm -f "$log_file"
}
trap cleanup EXIT

if [[ ! -x "$counter_bin" ]]; then
    make -C "$miniopenenv_root" build/rel/bin/counter
fi

"$counter_bin" --host 127.0.0.1 --port "$port" >"$log_file" 2>&1 &
counter_pid=$!

for _ in $(seq 1 100); do
    if curl --fail --silent "$url/health" >/dev/null; then
        break
    fi
    if ! kill -0 "$counter_pid" 2>/dev/null; then
        sed -n '1,120p' "$log_file" >&2
        exit 1
    fi
    sleep 0.05
done

curl --fail --silent "$url/health" >/dev/null
(
    cd "$repo_root"
    KILN_MINIOPENENV_URL="$url" \
        "$cargo_bin" test -p kiln-openenv \
        --test miniopenenv_interop -- --ignored --exact drives_a_stateful_miniopenenv_counter_episode
)
