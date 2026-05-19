#!/bin/bash
# Bootstrap a fresh kiln-runpod pod for pi-diff-patch-apply iters.
#
# Stages:
#   1. Clone kiln repo (cached) + cargo build release
#   2. Install Node 22 + pi CLI
#   3. Download/verify qwen3.5-4b weights
#   4. Sync this capability's source files to /workspace/kiln
#   5. Start kiln serve in background and verify
#
# Idempotent — re-running on warm pod is fast.
#
# Usage: bootstrap_pod.sh <pod_id>
set -euo pipefail

POD_ID="${1:-}"
if [ -z "$POD_ID" ]; then echo "usage: bootstrap_pod.sh <pod_id>" >&2; exit 1; fi
RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py

echo "== bootstrap pod=$POD_ID ($(date -u +%H:%M:%SZ)) =="

run_pod() {
  python3 "$RP" ssh "$POD_ID" "$1"
}

echo ">>> [1/6] check kiln source"
EXISTS=$(run_pod 'test -d /workspace/kiln/.git && echo yes || echo no' 2>/dev/null | tail -1)
if [ "$EXISTS" != "yes" ]; then
  echo ">>> cloning kiln"
  run_pod 'cd /workspace && git clone --reference /data/repo-cache/ericflo/kiln.git --dissociate https://github.com/ericflo/kiln.git || git clone https://github.com/ericflo/kiln.git'
fi
echo ">>> fetching latest main"
run_pod 'cd /workspace/kiln && git fetch origin && git checkout main && git reset --hard origin/main'

echo ">>> [2/6] build kiln (release)"
# Check if pre-built
BUILT=$(run_pod 'test -x /workspace/kiln/target/release/kiln && echo yes || echo no' 2>/dev/null | tail -1)
if [ "$BUILT" != "yes" ]; then
  echo ">>> compiling kiln + cuda_grpo_ablation example (--features cuda, KILN_CUDA_ARCHS=80)"
  # KILN_CUDA_ARCHS=80 builds compute_80 PTX only. compute_80 is forward-compatible
  # with A6000 (sm_86) and A100 (sm_80) via PTX→SASS JIT. Default 80;89;90 builds
  # 3 archs and is 2-3x slower without runtime benefit on A6000. Override with
  # 80;89;90 if targeting H100 (sm_90) or RTX 40-series (sm_89).
  python3 "$RP" bg "$POD_ID" /tmp/cargo-build.log \
    'cd /workspace/kiln && KILN_CUDA_ARCHS=80 cargo build --release --features cuda --bin kiln 2>&1 && KILN_CUDA_ARCHS=80 cargo build --release --features cuda --example cuda_grpo_ablation 2>&1'
  python3 "$RP" wait-file "$POD_ID" /workspace/kiln/target/release/kiln --timeout 1800
  python3 "$RP" wait-file "$POD_ID" /workspace/kiln/target/release/examples/cuda_grpo_ablation --timeout 1800
fi

echo ">>> [3/6] install Node 22 + pi"
NODE_VER=$(run_pod 'node --version 2>/dev/null || echo none' 2>/dev/null | tail -1)
case "$NODE_VER" in
  v22.*|v23.*|v24.*) echo ">>> Node $NODE_VER OK" ;;
  *)
    echo ">>> installing Node 22"
    run_pod 'curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs'
    ;;
esac
PI_OK=$(run_pod 'command -v pi >/dev/null 2>&1 && echo yes || echo no' 2>/dev/null | tail -1)
if [ "$PI_OK" != "yes" ]; then
  echo ">>> installing pi"
  run_pod '
if [ ! -d /workspace/pi-src ]; then
  git clone --depth=1 https://github.com/earendil-works/pi.git /workspace/pi-src
fi
cd /workspace/pi-src && npm install 2>&1 | tail -3 && npm run build 2>&1 | tail -3 || true
PKG_DIR=$(grep -lr "\"name\": \"pi\"" packages */package.json 2>/dev/null | head -1 | xargs -r dirname)
cd "$PKG_DIR" 2>/dev/null || true
npm link 2>&1 | tail -3 || true
'
fi
run_pod 'pi --version' 2>&1 | tail -2

echo ">>> [4/6] check qwen3.5-4b weights"
WEIGHTS_OK=$(run_pod 'test -f /workspace/qwen3.5-4b/model.safetensors && echo yes || echo no' 2>/dev/null | tail -1)
if [ "$WEIGHTS_OK" != "yes" ]; then
  echo ">>> downloading qwen3.5-4b weights (this is slow)"
  # Use `hf download` (not huggingface-cli, which was deprecated/aliased and
  # silently no-op'd on the A100 image 2026-05-19). The download is ~8GB.
  # Run synchronously and verify a model file exists before continuing.
  run_pod 'mkdir -p /workspace/qwen3.5-4b && cd /workspace/qwen3.5-4b && hf download Qwen/Qwen3.5-4B --local-dir . 2>&1 | tail -5'
  WEIGHTS_OK=$(run_pod 'test -f /workspace/qwen3.5-4b/config.json && ls /workspace/qwen3.5-4b/model.safetensors* 2>/dev/null | wc -l' 2>/dev/null | tail -1)
  if [ "$WEIGHTS_OK" = "0" ] || [ -z "$WEIGHTS_OK" ]; then
    echo "FATAL: qwen3.5-4b download produced no model files"
    exit 41
  fi
fi
echo ">>> ensuring adapters dir exists"
run_pod 'mkdir -p /workspace/qwen3.5-4b/adapters'

echo ">>> install pytest (rollouts need it for the verify step)"
run_pod 'python3 -m pytest --version 2>/dev/null || pip install pytest 2>&1 | tail -2'

echo ">>> run kiln pi-setup (configures pi to use local kiln on :8420)"
# This writes ~/.pi/agent/{models,settings}.json so pi uses provider=kiln-local
# and model=qwen-3.5-4b-kiln. Without this, pi defaults to provider=google and
# bails with "No API key found for the selected model" — wasting an entire iter
# of rollouts. Discovered 2026-05-19 on a fresh pod build.
run_pod '/workspace/kiln/target/release/kiln pi-setup 2>&1 | tail -3'

echo ">>> [5/6] sync capability files"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# rsync via runpod_api scp
python3 "$RP" upload "$POD_ID" "$HERE/rubric.py" /workspace/kiln/capabilities/agentic-grpo/pi-diff-patch-apply/rubric.py
python3 "$RP" upload "$POD_ID" "$HERE/rollout.py" /workspace/kiln/capabilities/agentic-grpo/pi-diff-patch-apply/rollout.py
python3 "$RP" upload "$POD_ID" "$HERE/task_scaffold.py" /workspace/kiln/capabilities/agentic-grpo/pi-diff-patch-apply/task_scaffold.py
python3 "$RP" upload "$POD_ID" "$HERE/build_corpus.py" /workspace/kiln/capabilities/agentic-grpo/pi-diff-patch-apply/build_corpus.py
python3 "$RP" upload "$POD_ID" "$HERE/select_hard_tasks.py" /workspace/kiln/capabilities/agentic-grpo/pi-diff-patch-apply/select_hard_tasks.py
# Build corpus on the pod (idempotent — overwrites datasets/)
echo ">>> building corpus"
run_pod 'cd /workspace/kiln/capabilities/agentic-grpo/pi-diff-patch-apply && mkdir -p datasets && python3 build_corpus.py --train 60 --eval 24 --seed 3141592653 2>&1 | tail -3'

echo ">>> [6/6] start kiln serve"
run_pod 'pgrep -x kiln | xargs -r kill -9 2>/dev/null || true; sleep 2'
python3 "$RP" bg "$POD_ID" /tmp/kiln-serve-bootstrap.log \
  'cd /workspace/kiln && KILN_DISABLE_FUSED_GDN_GATES=1 KILN_BATCHING_ENGINE=0 KILN_MODEL_PATH=/workspace/qwen3.5-4b ./target/release/kiln serve 2>&1'
sleep 30
HEALTH=$(run_pod 'curl -sS --max-time 5 http://localhost:8420/v1/adapters | head -c 100' 2>&1 | tail -1)
echo ">>> kiln serve health: $HEALTH"

echo "== bootstrap done pod=$POD_ID ($(date -u +%H:%M:%SZ)) =="
