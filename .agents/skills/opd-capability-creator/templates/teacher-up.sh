#!/usr/bin/env bash
# Bring up the OPD teacher (vLLM) with the flags this skill needs.
#
# Usage:
#   $SKILL/templates/teacher-up.sh <model_path> [served_name] [port] [gpu_mem]
#
# Defaults match capability.config.json:
#   served_name = qwen3.6-27b-awq
#   port        = 8002
#   gpu_mem     = 0.45 (leaves ~28GB for student + training)
#
# CRITICAL: --enable-prefix-caching is on. vLLM 0.17 defaults this OFF
# in some configs, which makes every OPD step pay the full prefill cost
# from scratch (we measured this mid-session: a 6h13m asymmetric run
# had Prefix cache hit rate 0.0% the entire time). With caching on, the
# same teacher prefix reused across epochs hits the cache, saving ~54%
# latency on the OPD-shape prompt_logprobs query.
set -euo pipefail

MODEL_PATH="${1:?model path required}"
SERVED_NAME="${2:-qwen3.6-27b-awq}"
PORT="${3:-8002}"
GPU_MEM="${4:-0.45}"
LOG="${TEACHER_LOG:-/tmp/vllm.log}"

# Hunt orphan EngineCore processes from prior runs.
for p in /proc/[0-9]*; do
  comm=$(cat "$p/comm" 2>/dev/null || true)
  case "$comm" in
    *VLLM*|*EngineCor*) kill "$(basename "$p")" 2>/dev/null || true;;
  esac
done

nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --max-model-len 4096 \
  --max-logprobs 64 \
  --enforce-eager \
  --gpu-memory-utilization "$GPU_MEM" \
  --enable-prefix-caching \
  > "$LOG" 2>&1 &

PID=$!
echo "teacher_pid=$PID log=$LOG"

# Block until ready.
until curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: vLLM died during startup. Tail of $LOG:" >&2
    tail -30 "$LOG" >&2
    exit 1
  fi
  sleep 5
done

# Sanity check: confirm prefix caching is on.
if grep -q "'enable_prefix_caching': True" "$LOG"; then
  echo "prefix_caching=on"
else
  echo "WARNING: --enable-prefix-caching was passed but vLLM may have disabled it. Check $LOG." >&2
fi
echo "teacher_up_port=$PORT served=$SERVED_NAME"
