#!/usr/bin/env bash
set -euo pipefail

source /root/.kiln-build-env
export KILN_CUDA_ARCHS=86
export CARGO_PROFILE_DEV_DEBUG=0
export KILN_W4A16=0
export KILN_MTP_ARGMAX_FP32=1
export KILN_SPEC_ENABLED=1
export KILN_SPEC_METHOD=mtp
export KILN_CUDA_GRAPHS=true

cd /workspace/kiln
mkdir -p docs/archive/phase-c/phase-c40f

for seed in $(seq 0 19); do
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --chat-template --skip-training --latency-only \
    --prompt-tokens 512 --max-output-tokens 128 \
    --prompt-subset humaneval \
    --temperature 0.0 \
    --seed "$seed" \
    > "docs/archive/phase-c/phase-c40f/seed-$seed.json" \
    2> "docs/archive/phase-c/phase-c40f/seed-$seed.log"
  echo $? > "docs/archive/phase-c/phase-c40f/seed-$seed.exit"
done
