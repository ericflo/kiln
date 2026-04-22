#!/usr/bin/env bash
set -euo pipefail
source /root/.kiln-build-env
export KILN_CUDA_ARCHS=86 CARGO_PROFILE_DEV_DEBUG=0
cd /workspace/kiln
mkdir -p /workspace/c40f-results
cat > /workspace/c40f-results/common-env.txt <<'EOT'
KILN_W4A16=0
KILN_MTP_ARGMAX_FP32=1
KILN_SPEC_ENABLED=1
KILN_SPEC_METHOD=mtp
KILN_CUDA_GRAPHS=true
EOT
cat > /workspace/c40f-results/command.txt <<'EOT'
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --chat-template --skip-training --latency-only \
  --prompt-tokens 512 --max-output-tokens 128 \
  --prompt-subset humaneval \
  --seed <SEED> --temperature 0.0
EOT
sha256sum ./target/release/kiln-bench > /workspace/c40f-results/kiln-bench.sha256
git rev-parse HEAD > /workspace/c40f-results/git-head.txt
for seed in $(seq 0 19); do
  export KILN_W4A16=0 KILN_MTP_ARGMAX_FP32=1 KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp KILN_CUDA_GRAPHS=true
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --chat-template --skip-training --latency-only \
    --prompt-tokens 512 --max-output-tokens 128 \
    --prompt-subset humaneval \
    --seed "$seed" --temperature 0.0 \
    > "/workspace/c40f-results/seed-${seed}.json" \
    2> "/workspace/c40f-results/seed-${seed}.log"
  printf %sn $? > "/workspace/c40f-results/seed-${seed}.exit"
done
date -u +%Y-%m-%dT%H:%M:%SZ > /workspace/c40f-results/done.txt
