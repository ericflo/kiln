#!/bin/bash
# Kiln benchmark sprint: showcases the structured kiln-bench output
# (cyan section headers, indicatif progress bar, summary table) plus the
# `-v` / `-q` verbosity controls. Recorded on RunPod A6000 at 120x32.
#
# Run from the kiln repo root via:
#
#   COLUMNS=120 LINES=32 TERM=xterm-256color asciinema rec docs/site/demo/bench.cast \
#     --title "kiln-bench: throughput, latency, training, all from one binary" \
#     --idle-time-limit 2 \
#     --command ./docs/site/demo/demo-bench.sh
#
# Prerequisites:
#   - kiln-bench at ./target/release/kiln-bench by default; override
#     KILN_BENCH_BIN to use an extracted release artifact or another
#     source-built binary.
#   - ./Qwen3.5-4B/ weights (override with KILN_MODEL_PATH)
#   - kiln.example.toml exists at the repo root

set -e

export KILN_MODEL_PATH="${KILN_MODEL_PATH:-./Qwen3.5-4B}"
KILN_BENCH_BIN="${KILN_BENCH_BIN:-./target/release/kiln-bench}"

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
# Scene 1 — Defaults: clean output, structured headers, summary table
# ------------------------------------------------------------------
typecmd "${KILN_BENCH_BIN} --model-path ${KILN_MODEL_PATH} --paged --skip-training \\"
typecmd '    --prompt-tokens 256 --max-output-tokens 64'

"${KILN_BENCH_BIN}" --model-path "${KILN_MODEL_PATH}" --paged --skip-training \
    --prompt-tokens 256 --max-output-tokens 64

beat 2.0

# ------------------------------------------------------------------
# Scene 2 — Same run with -v: structured tracing for CI / log aggregators
# ------------------------------------------------------------------
typecmd '# Same flags, with -v: structured tracing for CI + log scrapers.'
typecmd "${KILN_BENCH_BIN} -v --model-path ${KILN_MODEL_PATH} --paged --skip-training \\"
typecmd '    --prompt-tokens 256 --max-output-tokens 32 2>&1 | head -20'

"${KILN_BENCH_BIN}" -v --model-path "${KILN_MODEL_PATH}" --paged --skip-training \
    --prompt-tokens 256 --max-output-tokens 32 2>&1 | head -20

beat 2.0
