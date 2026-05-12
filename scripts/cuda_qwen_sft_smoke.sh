#!/usr/bin/env bash
#
# CUDA Qwen3.5-4B real-model SFT smoke.
#
# Intended for RunPod A6000 validation. It downloads Qwen/Qwen3.5-4B when the
# local model directory is missing, builds kiln-bench with CUDA, then runs one
# real SFT training step through the default CUDA trainer path.
#
# Usage:
#   scripts/cuda_qwen_sft_smoke.sh
#   scripts/cuda_qwen_sft_smoke.sh --model-path /workspace/qwen3.5-4b --skip-build
#   scripts/cuda_qwen_sft_smoke.sh --model-path /workspace/qwen3.5-4b --skip-build --native-cuda

set -euo pipefail

MODEL_ID="${KILN_MODEL_ID:-Qwen/Qwen3.5-4B}"
MODEL_PATH="${KILN_MODEL_PATH:-/workspace/qwen3.5-4b}"
BENCH_BIN="${KILN_BENCH_BIN:-target/release/kiln-bench}"
CUDA_ARCHS="${KILN_CUDA_ARCHS:-86}"
SKIP_BUILD=0
SKIP_DOWNLOAD=0
NATIVE_CUDA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --bench-bin)
            BENCH_BIN="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=1
            shift
            ;;
        --native-cuda)
            NATIVE_CUDA=1
            shift
            ;;
        --help|-h)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

for tool in cargo; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "missing required tool: $tool" >&2
        exit 1
    fi
done

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
        echo "model is missing and --skip-download was set: $MODEL_PATH" >&2
        exit 1
    fi
    if ! command -v hf >/dev/null 2>&1; then
        echo "model is missing and huggingface-cli 'hf' is not installed" >&2
        exit 1
    fi
    echo "downloading $MODEL_ID into $MODEL_PATH" >&2
    HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
        hf download "$MODEL_ID" --local-dir "$MODEL_PATH"
fi

if [[ "$SKIP_BUILD" -eq 0 || ! -x "$BENCH_BIN" ]]; then
    echo "building $BENCH_BIN with CUDA archs=$CUDA_ARCHS" >&2
    KILN_CUDA_ARCHS="$CUDA_ARCHS" cargo build --release --features cuda --bin kiln-bench
fi

echo "running one-step CUDA SFT smoke against $MODEL_PATH" >&2
if [[ "$NATIVE_CUDA" -eq 1 ]]; then
    echo "enabling KILN_CUDA_NATIVE_TRAINING=1 for native CUDA SFT smoke" >&2
fi
KILN_CUDA_ARCHS="$CUDA_ARCHS" \
KILN_SPEC_METHOD=off \
KILN_USE_FLCE=1 \
KILN_CUDA_NATIVE_TRAINING="$NATIVE_CUDA" \
"$BENCH_BIN" \
    --model-path "$MODEL_PATH" \
    --prompt-tokens 8 \
    --max-output-tokens 1 \
    --training-steps 1 \
    --paged \
    --quiet
