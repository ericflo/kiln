#!/usr/bin/env bash
# Teardown teacher, bring kiln serve up, eval the named adapter.
# Usage: ./eval_adapter.sh <adapter_path_or_name>
set -euo pipefail
cd "$(dirname "$0")"

ADAPTER="${1:?adapter required}"

# 1. Kill vLLM teacher (frees ~22GB)
pkill -f "vllm.entrypoints.openai.api_server" || true
# Hunt orphan EngineCore procs
for p in /proc/[0-9]*; do
  comm=$(cat "$p/comm" 2>/dev/null || true)
  case "$comm" in
    *VLLM*|*EngineCor*) kill "$(basename "$p")" 2>/dev/null || true;;
  esac
done

# Wait for GPU memory to drop
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 5000 ]; do
  sleep 2
done
echo "teacher_down_vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"

# 2. Start kiln serve
cd /workspace/kiln
KILN_MODEL_PATH=/workspace/kiln/Qwen3.5-4B nohup kiln serve > /tmp/kiln-serve-iter2.log 2>&1 &
KILN_PID=$!
echo "kiln_serve_pid=$KILN_PID"

# Wait for it to come up
until curl -sf --max-time 2 http://localhost:8420/v1/models > /dev/null 2>&1; do
  if ! kill -0 "$KILN_PID" 2>/dev/null; then
    echo "ERROR: kiln serve died during startup" >&2
    tail -20 /tmp/kiln-serve-iter2.log >&2
    exit 1
  fi
  sleep 2
done
echo "kiln_serve_up_vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"

# 3. Run the oracle
cd /workspace/kiln/opd-cap.transcript-compaction
./capability.oracle.sh "$ADAPTER"
