#!/usr/bin/env bash
# iter5 stage A: clone kiln + download model + build cuda binaries
set -euo pipefail

mkdir -p /workspace
cd /workspace

# Setup: clone kiln + download model + configure sccache (only if not already done)
if [ ! -f /root/.kiln-build-env ]; then
  if [ -z "${B2_APPLICATION_KEY_ID:-}" ] || [ -z "${B2_APPLICATION_KEY:-}" ]; then
    echo "FATAL: B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY required for first-time setup"
    exit 1
  fi
  if [ ! -d /workspace/kiln/.git ]; then
    kiln-setup --clone --repo /workspace/kiln
  else
    kiln-setup --repo /workspace/kiln
  fi
else
  echo "build env already set up; skipping kiln-setup"
fi

# Source the build env
source /root/.kiln-build-env

# Pull the iter5 branch (carries iter5_prep_sft_data.py + stage scripts)
cd /workspace/kiln
git fetch --depth=1 origin iter5-sft-strict-rollouts && git checkout -f iter5-sft-strict-rollouts && git reset --hard origin/iter5-sft-strict-rollouts

# Build the cuda release binaries (kiln serve, cuda_sft_file, kiln eval-adapter, etc.)
# SCCACHE_RECACHE=1: force fresh compile + WRITE back to B2 cache. Previous build
# pulled a corrupted gdn_gates.o from B2 cache — kiln_gdn_gates_bf16 crashed on every
# inference. Fresh nvcc compile produces a working kernel; RECACHE=1 also heals B2
# for future pods.
echo "=== building kiln (cuda release, SCCACHE_RECACHE=1) ==="
SCCACHE_RECACHE=1 KILN_CUDA_ARCHS=86 cargo build --release --features cuda 2>&1 | tail -100
echo "exit=$?"
echo "=== building examples (cuda_sft_file etc.) ==="
SCCACHE_RECACHE=1 KILN_CUDA_ARCHS=86 cargo build --release --features cuda --examples 2>&1 | tail -30
echo "exit=$?"

# Verify the binaries we need exist
echo "=== checking binaries ==="
ls -la target/release/kiln target/release/cuda_sft_file 2>&1 || true
for bin in kiln cuda_sft_file cuda_opd_remote cuda_grpo_ablation; do
  if [ -x "target/release/$bin" ]; then
    echo "OK: target/release/$bin"
  else
    echo "MISSING: target/release/$bin"
  fi
done

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader

# Pull the latest capabilities directory state from our work
echo "=== git status of capabilities ==="
cd /workspace/kiln/capabilities/caps/pi-faithful-completion
ls iter5_prep_sft_data.py 2>&1 || echo "MISSING iter5_prep_sft_data.py"

touch /workspace/iter5-stage-a.done
echo "STAGE A COMPLETE"
