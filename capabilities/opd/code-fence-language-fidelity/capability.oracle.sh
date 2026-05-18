#!/usr/bin/env bash
set -euo pipefail
ADAPTER="${1:-}"
EVAL_FILE="datasets/eval.jsonl"
KILN_URL="${KILN_URL:-http://localhost:8420}"

if ! curl -sf "$KILN_URL/v1/models" > /dev/null 2>&1; then
  echo "ORACLE_ERROR: kiln-server not reachable at $KILN_URL" >&2
  exit 2
fi

python3 - "$ADAPTER" "$EVAL_FILE" "$KILN_URL" <<'PY'
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
from rubric import score_response

adapter, eval_file, url = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = []
with open(eval_file) as f:
    for line in f:
        if line.strip():
            prompts.append(json.loads(line))


def request_one(prompt):
    body = {
        "messages": prompt["messages"][:-1],
        "max_tokens": 384,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if adapter and adapter not in ("base", "none", ""):
        body["adapter"] = adapter
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
        resp = d["choices"][0]["message"].get("content") or ""
    except Exception:
        resp = ""
    return {"response": resp, "expected_language": prompt["expected_language"]}


with ThreadPoolExecutor(max_workers=4) as exe:
    futs = [exe.submit(request_one, p) for p in prompts]
    results = [f.result() for f in as_completed(futs)]

sums = {"fence_pair": 0.0, "no_extra_text": 0.0, "language_tag_correct": 0.0,
        "code_parses": 0.0, "composite": 0.0}
n = 0
for r in results:
    s = score_response(**r)
    for k in sums:
        sums[k] += s[k]
    n += 1

print(f"SCORE={sums['composite']/n:.4f}")
for k in ["fence_pair", "no_extra_text", "language_tag_correct", "code_parses"]:
    print(f"{k}={sums[k]/n:.4f}")
print(f"N={n}")
PY
