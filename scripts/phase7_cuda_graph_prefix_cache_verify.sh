#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:8420}
MODEL_PATH=${KILN_MODEL_PATH:-/workspace/qwen3.5-4b}
LOG_FILE=${LOG_FILE:-/tmp/kiln-cuda-graphs-prefix-cache.log}
KILN_BIN=${KILN_BIN:-target/release/kiln}

if [[ ! -x "$KILN_BIN" ]]; then
  echo "missing executable: $KILN_BIN" >&2
  exit 1
fi

rm -f "$LOG_FILE"
KILN_MODEL_PATH="$MODEL_PATH" \
KILN_W4A16=1 \
KILN_CUDA_GRAPHS=true \
KILN_SPEC_ENABLED=0 \
KILN_PREFIX_CACHE_ENABLED=1 \
KILN_PREFIX_CACHE_MAX_BLOCKS=${KILN_PREFIX_CACHE_MAX_BLOCKS:-2048} \
KILN_NUM_BLOCKS=${KILN_NUM_BLOCKS:-4096} \
KILN_KV_CACHE_FP8=${KILN_KV_CACHE_FP8:-1} \
"$KILN_BIN" serve >"$LOG_FILE" 2>&1 &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "kiln exited before becoming healthy; log follows" >&2
    cat "$LOG_FILE" >&2
    exit 1
  fi
  sleep 1
done
curl -fsS "$BASE_URL/health" >/dev/null

request() {
  local content=$1
  local max_tokens=${2:-1}
  CONTENT="$content" MAX_TOKENS="$max_tokens" python3 - <<'PY' | curl -fsS "$BASE_URL/v1/chat/completions" -H 'content-type: application/json' -d @-
import json, os
print(json.dumps({
    "model": "kiln",
    "messages": [{"role": "user", "content": os.environ["CONTENT"]}],
    "temperature": 0.0,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "seed": 1,
}))
PY
}

shared=""
for repeats in $(seq 32 512); do
  candidate=$(python3 - <<PY
print(("Kiln CUDA graph prefix cache shared segment ${repeats}. " * ${repeats}).strip())
PY
)
  response=$(request "$candidate" 1)
  prompt_tokens=$(python3 -c 'import json, sys; print(json.load(sys.stdin)["usage"]["prompt_tokens"])' <<<"$response")
  if (( prompt_tokens % 16 == 0 )); then
    shared=$candidate
    echo "warmed block-aligned prompt: repeats=$repeats prompt_tokens=$prompt_tokens"
    break
  fi
done

if [[ -z "$shared" ]]; then
  echo "failed to find a block-aligned prompt" >&2
  exit 1
fi

suffix=$'<|im_end|>\n<|im_start|>assistant\n Continue with one concise CUDA graph prefix-cache sentence.'
request "${shared}${suffix}" 4 >/tmp/kiln-cuda-graphs-prefix-cache-hit.json
metrics=$(curl -fsS "$BASE_URL/metrics")
echo "$metrics" | grep 'kiln_prefix_cache_lookups_total{result="hit"}'
hits=$(python3 -c 'import re, sys; m = re.search(r"kiln_prefix_cache_lookups_total\{result=\"hit\"\}\s+(\d+)", sys.stdin.read()); print(m.group(1) if m else 0)' <<<"$metrics")
if (( hits <= 0 )); then
  echo "expected prefix-cache hit with CUDA graphs enabled" >&2
  exit 1
fi

if grep -Fq 'bypass prefix-cache' "$LOG_FILE"; then
  echo "unexpected CUDA graph prefix-cache bypass warning" >&2
  exit 1
fi

echo "verified CUDA graphs + prefix cache hit count: $hits"
