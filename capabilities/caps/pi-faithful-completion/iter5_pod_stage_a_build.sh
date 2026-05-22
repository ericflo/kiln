#!/usr/bin/env bash
# iter5 stage A: clone kiln + download model + build cuda binaries
set -euo pipefail

mkdir -p /workspace
cd /workspace

# Setup: clone kiln, download model, configure sccache
if [ ! -d /workspace/kiln/.git ]; then
  kiln-setup --clone --repo /workspace/kiln
else
  echo "kiln already cloned; just running setup"
  kiln-setup --repo /workspace/kiln
fi

# Source the build env
source /root/.kiln-build-env

# Pull the latest kiln main (in case base image is older)
cd /workspace/kiln
git fetch --depth=1 origin main && git reset --hard origin/main

# Build the cuda release binaries (kiln serve, cuda_sft_file, kiln eval-adapter, etc.)
echo "=== building kiln (cuda release) ==="
KILN_CUDA_ARCHS=86 cargo build --release --features cuda 2>&1 | tail -100
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
