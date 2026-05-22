#!/usr/bin/env bash
# kiln-smoke-check — post-build sanity check for fused CUDA kernels.
#
# Background (issue #1066): on a fresh pod, the B2-backed sccache can serve
# a corrupted GDN kernel object that compiles fine but explodes at inference
# time with an error chain like:
#
#   batched-engine prefill forward pass failed
#   ...
#   kiln_gdn_gates_bf16 failed with status 500
#
# The build itself reports success — only the first real inference call
# surfaces the breakage. This script exercises every kernel on the inference
# hot path with a minimal prompt so the same failure mode is caught at
# build time instead of by the first user request.
#
# Usage:
#   kiln-smoke-check                              # uses defaults
#   kiln-smoke-check --bench <path>               # override kiln-bench location
#   kiln-smoke-check --model-path <dir>           # override model dir
#   kiln-smoke-check --max-output-tokens <n>      # default 4
#   kiln-smoke-check --prompt-tokens <n>          # default 32
#   kiln-smoke-check --no-paged                   # skip --paged (non-prod path)
#   kiln-smoke-check --quiet                      # suppress bench stdout on success
#   kiln-smoke-check --timeout <secs>             # wall-clock cap (default 600)
#
# Defaults:
#   bench:        $KILN_REPO_DIR/target/release/kiln-bench, or `which kiln-bench`
#   model:        ${KILN_MODEL_DIR:-/workspace/Qwen3.5-4B}
#
# Exit codes:
#   0   smoke check passed (every kernel on the inference path returned cleanly)
#   1   prerequisite missing (no binary or no model)
#   2   smoke check failed AND output matched a known-bad sccache pattern;
#       recovery instructions printed
#   3   smoke check failed for an unknown reason; raw output printed
#   4   cli misuse
#   124 wall-clock timeout
#
# This script never modifies the cache. The user must opt in to SCCACHE_RECACHE=1
# and a rebuild — automating that would mask real kernel regressions.

set -uo pipefail

KILN_REPO_DIR="${KILN_REPO_DIR:-/workspace/kiln}"
KILN_MODEL_DIR="${KILN_MODEL_DIR:-/workspace/Qwen3.5-4B}"

BENCH=""
MODEL="${KILN_MODEL_DIR}"
MAX_OUTPUT_TOKENS=4
PROMPT_TOKENS=32
PAGED=1
QUIET=0
TIMEOUT_SECS=600

usage() {
    sed -n '2,33p' "$0"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --bench)               BENCH="$2"; shift 2 ;;
        --model-path)          MODEL="$2"; shift 2 ;;
        --max-output-tokens)   MAX_OUTPUT_TOKENS="$2"; shift 2 ;;
        --prompt-tokens)       PROMPT_TOKENS="$2"; shift 2 ;;
        --no-paged)            PAGED=0; shift ;;
        --quiet|-q)            QUIET=1; shift ;;
        --timeout)             TIMEOUT_SECS="$2"; shift 2 ;;
        -h|--help)             usage; exit 0 ;;
        *) echo "kiln-smoke-check: unknown arg: $1" >&2; usage >&2; exit 4 ;;
    esac
done

# Locate kiln-bench
if [ -z "${BENCH}" ]; then
    for candidate in \
        "${KILN_REPO_DIR}/target/release/kiln-bench" \
        "$(command -v kiln-bench 2>/dev/null || true)"
    do
        if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
            BENCH="${candidate}"
            break
        fi
    done
fi

if [ -z "${BENCH}" ] || [ ! -x "${BENCH}" ]; then
    echo "kiln-smoke-check: kiln-bench binary not found" >&2
    echo "  tried: ${KILN_REPO_DIR}/target/release/kiln-bench and PATH" >&2
    echo "  build first: cargo build --release --features cuda --bin kiln-bench" >&2
    echo "  or pass --bench <path>" >&2
    exit 1
fi

if [ ! -f "${MODEL}/config.json" ]; then
    echo "kiln-smoke-check: model not found at ${MODEL}" >&2
    echo "  config.json missing — run kiln-setup to download, or pass --model-path <dir>" >&2
    exit 1
fi

echo "=== kiln-smoke-check ==="
echo "  bench:               ${BENCH}"
echo "  model:               ${MODEL}"
echo "  prompt-tokens:       ${PROMPT_TOKENS}"
echo "  max-output-tokens:   ${MAX_OUTPUT_TOKENS}"
echo "  paged:               $([ ${PAGED} -eq 1 ] && echo yes || echo no)"
echo "  timeout:             ${TIMEOUT_SECS}s"

BENCH_ARGS=(
    --model-path "${MODEL}"
    --skip-training
    --latency-only
    --max-output-tokens "${MAX_OUTPUT_TOKENS}"
    --prompt-tokens "${PROMPT_TOKENS}"
)
[ ${PAGED} -eq 1 ] && BENCH_ARGS+=(--paged)

LOG="$(mktemp -t kiln-smoke-check.XXXXXX 2>/dev/null || mktemp "${TMPDIR:-/tmp}/kiln-smoke-check.XXXXXX")"
trap 'rm -f "${LOG}"' EXIT

# Run the bench. Capture combined stdout+stderr because anyhow context
# strings come on stderr but the JSON summary lands on stdout, and both
# matter for diagnosing what went wrong.
STATUS=0
if command -v timeout >/dev/null 2>&1; then
    # GNU coreutils accepts --kill-after; busybox uses -k. Try both forms.
    if timeout --kill-after=10 "${TIMEOUT_SECS}" true 2>/dev/null; then
        timeout --kill-after=10 "${TIMEOUT_SECS}" "${BENCH}" "${BENCH_ARGS[@]}" >"${LOG}" 2>&1 || STATUS=$?
    elif timeout -k 10 "${TIMEOUT_SECS}" true 2>/dev/null; then
        timeout -k 10 "${TIMEOUT_SECS}" "${BENCH}" "${BENCH_ARGS[@]}" >"${LOG}" 2>&1 || STATUS=$?
    else
        timeout "${TIMEOUT_SECS}" "${BENCH}" "${BENCH_ARGS[@]}" >"${LOG}" 2>&1 || STATUS=$?
    fi
else
    "${BENCH}" "${BENCH_ARGS[@]}" >"${LOG}" 2>&1 || STATUS=$?
fi

if [ ${STATUS} -eq 124 ] || [ ${STATUS} -eq 137 ]; then
    echo "kiln-smoke-check: TIMEOUT after ${TIMEOUT_SECS}s (status ${STATUS})" >&2
    tail -40 "${LOG}" >&2
    exit 124
fi

if [ ${STATUS} -eq 0 ]; then
    # Sanity: the bench can return 0 but still have logged an internal kernel
    # warning. Belt-and-braces grep for the canonical bail message.
    if grep -qE 'failed with status [0-9]+' "${LOG}"; then
        STATUS=99
    fi
fi

if [ ${STATUS} -eq 0 ]; then
    if [ ${QUIET} -eq 0 ]; then
        tail -20 "${LOG}"
    fi
    echo ""
    echo "✓ kiln-smoke-check: PASSED"
    echo "  All kernels on the inference path returned cleanly."
    exit 0
fi

# Failed. Try to classify.
echo ""
echo "✗ kiln-smoke-check: FAILED (bench exit ${STATUS})" >&2
echo "" >&2

# Look for the canonical "<kernel> failed with status <N>" line. If found,
# it's almost certainly the corrupted-cache pattern documented in #1066.
KERNEL_LINE="$(grep -oE 'kiln_[a-z0-9_]+ failed with status [0-9]+' "${LOG}" | head -1 || true)"
GENERIC_LINE="$(grep -oE '[a-z_]+ kernel failed with status [0-9]+' "${LOG}" | head -1 || true)"

if [ -n "${KERNEL_LINE}" ] || [ -n "${GENERIC_LINE}" ]; then
    echo "Detected fused-kernel failure: ${KERNEL_LINE:-${GENERIC_LINE}}" >&2
    echo "" >&2
    echo "This pattern is the signature of issue #1066: the build succeeded but" >&2
    echo "a cached object served from the remote sccache (B2) is corrupted. The" >&2
    echo "kernel source is fine; only the cached compile output is bad." >&2
    echo "" >&2
    echo "Recovery (forces sccache to re-compile + overwrite the bad object):" >&2
    echo "" >&2
    echo "  source /root/.kiln-build-env" >&2
    echo "  SCCACHE_RECACHE=1 cargo build --release --features cuda --bin kiln-bench" >&2
    echo "  kiln-smoke-check" >&2
    echo "" >&2
    echo "If a single kernel crate is implicated, you can scope the rebuild:" >&2
    echo "" >&2
    echo "  SCCACHE_RECACHE=1 cargo build --release --features cuda -p kiln-gdn-kernel" >&2
    echo "" >&2
    echo "Last 30 lines of bench output:" >&2
    echo "" >&2
    tail -30 "${LOG}" >&2
    exit 2
fi

# Unknown failure mode — surface the raw output so the operator can triage.
echo "No known sccache-corruption signature in output. Full tail follows:" >&2
echo "" >&2
tail -60 "${LOG}" >&2
exit 3
