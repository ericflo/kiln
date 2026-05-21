#!/usr/bin/env bash
set -o pipefail

RUN="${RUN:-/workspace/kiln-validation/issue40/actual-model}"
DONE="${DONE:-/workspace/kiln-validation/issue40/actual-model.done}"
MODEL="${KILN_MODEL_PATH:-/workspace/Qwen3.5-4B}"
PORT="${ISSUE40_PORT:-19440}"
BASE_URL="http://127.0.0.1:${PORT}"
ADAPTER_DIR="${ADAPTER_DIR:-${RUN}/adapters}"
ADAPTER_NAME="${ADAPTER_NAME:-issue40-latency-smoke}"
CYCLES="${ISSUE40_LATENCY_CYCLES:-8}"

rm -f "$DONE"
mkdir -p "$RUN" "$ADAPTER_DIR"
trap 'status=$?; if [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; echo exit=$status > "$DONE"' EXIT
exec > >(tee "$RUN/driver.log") 2>&1
set -euo pipefail

echo "=== issue40 actual Qwen3.5-4B regressions ==="
date -u
echo "model=$MODEL"
echo "base_url=$BASE_URL"
echo "adapter_dir=$ADAPTER_DIR"
echo "adapter_name=$ADAPTER_NAME"

[ -f /root/.kiln-build-env ] && source /root/.kiln-build-env || true
export KILN_CUDA_ARCHS="${KILN_CUDA_ARCHS:-86}"

if [ -d /workspace/qwen3.5-4b ] \
  && [ ! -L /workspace/qwen3.5-4b ] \
  && [ ! -e "$MODEL" ]
then
  mv /workspace/qwen3.5-4b "$MODEL"
fi
ln -sfn Qwen3.5-4B /workspace/qwen3.5-4b
test -f "$MODEL/tokenizer.json"
test -f "$MODEL/config.json"
find "$MODEL" -maxdepth 1 -name '*.safetensors' | grep -q .

cd /workspace/kiln
echo "=== source ==="
git rev-parse HEAD
git status --short

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "=== build cuda kiln server ==="
  cargo build --locked --release -p kiln-server --bin kiln --features cuda
fi

echo "=== start kiln serve ==="
KILN_MODEL_PATH="$MODEL" \
KILN_SERVED_MODEL_ID="Qwen3.5-4B" \
KILN_ADAPTER_DIR="$ADAPTER_DIR" \
KILN_PORT="$PORT" \
KILN_HOST=127.0.0.1 \
KILN_EVAL_MODE=true \
KILN_DEFAULT_THINKING_ENABLED=false \
KILN_NUM_BLOCKS="${KILN_NUM_BLOCKS:-256}" \
KILN_REQUEST_TIMEOUT_SECS="${KILN_REQUEST_TIMEOUT_SECS:-1200}" \
KILN_GRAD_CHECKPOINT_SEGMENTS="${KILN_GRAD_CHECKPOINT_SEGMENTS:-4}" \
KILN_LOG_FORMAT=json \
./target/release/kiln serve --eval-mode > "$RUN/server.log" 2>&1 &
SERVER_PID=$!
export SERVER_PID BASE_URL RUN ADAPTER_NAME CYCLES
echo "server_pid=$SERVER_PID"

python3 - <<'PY'
import json
import os
import time
import urllib.request

base = os.environ["BASE_URL"]
pid = int(os.environ["SERVER_PID"])
run = os.environ["RUN"]
last = None

def checks_pass(body, name):
    checks = body.get("checks")
    if isinstance(checks, list):
        return any(c.get("name") == name and c.get("pass") for c in checks)
    if isinstance(checks, dict):
        c = checks.get(name)
        return bool(c and c.get("pass"))
    return False

for _ in range(180):
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as resp:
            body = json.loads(resp.read().decode())
        last = body
        if body.get("status") == "ok" and body.get("backend") == "model":
            with open(f"{run}/health.json", "w") as f:
                json.dump(body, f, indent=2)
            break
    except Exception as exc:
        last = repr(exc)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit("server exited before health became ready")
    time.sleep(5)
else:
    raise SystemExit(f"health never became ready: {last}")

assert last["model"].startswith("Qwen3.5-4B "), last["model"]
assert last["eval_mode"] is True, last
assert last["default_thinking_enabled"] is False, last
assert checks_pass(last, "model_loaded"), last
assert checks_pass(last, "scheduler_responsive"), last
assert checks_pass(last, "inference_prewarm_complete"), last
print("health_ok model=" + last["model"])
PY

python3 - <<'PY'
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

base = os.environ["BASE_URL"]
run = Path(os.environ["RUN"])
adapter_name = os.environ["ADAPTER_NAME"]
cycles = int(os.environ["CYCLES"])

def request(method, path, body=None, timeout=1200):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return resp.status, json.loads(raw) if raw else {}, elapsed_ms
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        raise RuntimeError(f"{method} {path} failed {exc.code}: {raw}") from exc

def wait_job(job_id, label):
    deadline = time.time() + 1800
    while time.time() < deadline:
        status, body, _ = request("GET", f"/v1/train/status/{job_id}", timeout=30)
        assert status == 200, body
        (run / f"{label}_status_latest.json").write_text(json.dumps(body, indent=2))
        state = body["state"]
        if state in ("completed", "failed"):
            _, detail, _ = request("GET", f"/v1/train/jobs/{job_id}", timeout=30)
            (run / f"{label}_detail.json").write_text(json.dumps(detail, indent=2))
            if state != "completed":
                raise RuntimeError(f"{label} ended in {state}: {detail}")
            return detail
        time.sleep(5)
    raise RuntimeError(f"{label} did not finish before timeout")

def chat(label, adapter_field=None):
    body = {
        "model": "Qwen3.5-4B",
        "messages": [{"role": "user", "content": f"Reply with two words for {label}."}],
        "temperature": 0.0,
        "max_tokens": 8,
        "seed": 40,
        "include_performance": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if adapter_field is not None:
        body["adapter"] = adapter_field
    status, resp, client_ms = request("POST", "/v1/chat/completions", body, timeout=1200)
    assert status == 200, resp
    return resp, client_ms

thinking_resp, thinking_client_ms = chat("issue forty thinking off")
(run / "thinking_off_chat.json").write_text(json.dumps(thinking_resp, indent=2))
msg = thinking_resp["choices"][0]["message"]
content = msg.get("content") or ""
reasoning = msg.get("reasoning_content")
metadata = thinking_resp["metadata"]
assert thinking_resp["model"] == "Qwen3.5-4B", thinking_resp["model"]
assert content.strip(), thinking_resp
assert not (reasoning or "").strip(), thinking_resp
assert "<think>" not in content.lower(), thinking_resp
assert metadata["thinking_enabled"] is False, metadata
assert metadata["thinking_mode"] == "non_reasoning", metadata
print("thinking_off_content_ok content=" + content.replace("\n", "\\n")[:120])

train_body = {
    "examples": [
        {
            "messages": [
                {"role": "user", "content": "For ISSUE40-LATENCY, answer with one word."},
                {"role": "assistant", "content": "stable"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the latency-regression sentinel word?"},
                {"role": "assistant", "content": "stable"},
            ]
        },
    ],
    "config": {
        "output_name": adapter_name,
        "learning_rate": 5e-4,
        "epochs": 1,
        "lora_rank": 2,
        "lora_alpha": 4.0,
        "seed": 20260521,
        "adapter_smoke_test": True,
        "auto_load": False,
    },
}
_, submit, _ = request("POST", "/v1/train/sft", train_body)
(run / "sft_submit.json").write_text(json.dumps(submit, indent=2))
detail = wait_job(submit["job_id"], "sft")
receipt_path = Path(detail["adapter_path"]) / "train_receipt.json"
receipt = json.loads(receipt_path.read_text())
(run / "sft_train_receipt.json").write_text(json.dumps(receipt, indent=2))
assert receipt["status"] == "success", receipt
assert receipt["adapters"]["output"]["adapter_model_sha256"], receipt["adapters"]
assert receipt.get("adapter_smoke_test", {}).get("passed") is True, receipt.get("adapter_smoke_test")
print("adapter_ready path=" + detail["adapter_path"])

def load_adapter():
    status, body, ms = request("POST", "/v1/adapters/load", {"name": adapter_name}, timeout=1200)
    assert status == 200, body
    assert body["status"] == "loaded", body
    return ms

def unload_adapter():
    status, body, ms = request("POST", "/v1/adapters/unload", {}, timeout=1200)
    assert status == 200, body
    assert body["status"] == "unloaded", body
    return ms

load_adapter()
chat("warm adapter")
unload_adapter()
chat("warm base")

records = []
for cycle in range(cycles):
    started = time.perf_counter()
    load_ms = load_adapter()
    adapter_resp, adapter_client_ms = chat(f"adapter cycle {cycle}")
    unload_ms = unload_adapter()
    base_resp, base_client_ms = chat(f"base cycle {cycle}")
    cycle_ms = (time.perf_counter() - started) * 1000.0
    record = {
        "cycle": cycle,
        "load_ms": load_ms,
        "adapter_client_ms": adapter_client_ms,
        "adapter_server_total_ms": adapter_resp["metadata"]["performance"]["total_latency_ms"],
        "unload_ms": unload_ms,
        "base_client_ms": base_client_ms,
        "base_server_total_ms": base_resp["metadata"]["performance"]["total_latency_ms"],
        "cycle_ms": cycle_ms,
    }
    records.append(record)
    (run / "latency_records_latest.json").write_text(json.dumps(records, indent=2))

_, adapters, _ = request("GET", "/v1/adapters", timeout=30)
(run / "adapters_final.json").write_text(json.dumps(adapters, indent=2))
assert adapters["active_adapter"] is None, adapters
assert adapters["loaded_adapter"] is None, adapters

first = [r["cycle_ms"] for r in records[: max(3, cycles // 2)]]
last = [r["cycle_ms"] for r in records[-max(3, cycles // 2) :]]
first_median = statistics.median(first)
last_median = statistics.median(last)
limit = max(first_median * 10.0, first_median + 5000.0)
assert last_median <= limit, {
    "first_median_ms": first_median,
    "last_median_ms": last_median,
    "limit_ms": limit,
    "records": records,
}

summary = {
    "model": "Qwen3.5-4B",
    "thinking_off_content": content,
    "thinking_off_client_ms": thinking_client_ms,
    "adapter_name": adapter_name,
    "adapter_path": detail["adapter_path"],
    "cycles": cycles,
    "first_cycle_median_ms": first_median,
    "last_cycle_median_ms": last_median,
    "latency_drift_limit_ms": limit,
    "records": records,
}
(run / "summary.json").write_text(json.dumps(summary, indent=2))
print("latency_drift_ok first_median_ms=%.2f last_median_ms=%.2f limit_ms=%.2f" % (first_median, last_median, limit))
PY

echo "ISSUE40_ACTUAL_MODEL_REGRESSIONS_OK"
