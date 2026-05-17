#!/usr/bin/env bash
# Blind oracle for faithful-code-summarization.
#
# Argument: adapter name (empty = base).
# Output (stdout):
#   SCORE=<float>          (composite, weighted average)
#   parses=<float>
#   entity_recall=<float>
#   entity_precision=<float>
#   concise=<float>
#   N=<int>
#
# The script reads datasets/eval.jsonl internally. The AGENT never reads
# that file or the per-prompt responses; only the aggregate sub-scores.
set -euo pipefail

ADAPTER="${1:-}"
EVAL_FILE="datasets/eval.jsonl"
KILN_URL="${KILN_URL:-http://localhost:8420}"

if ! curl -sf "$KILN_URL/v1/models" > /dev/null 2>&1; then
  echo "ORACLE_ERROR: kiln-server not reachable at $KILN_URL" >&2
  exit 2
fi

python3 - "$ADAPTER" "$EVAL_FILE" "$KILN_URL" <<'PY'
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, ".")
from rubric import score_response

adapter, eval_file, url = sys.argv[1], sys.argv[2], sys.argv[3]

prompts = []
with open(eval_file) as f:
    for line in f:
        line = line.strip()
        if line:
            prompts.append(json.loads(line))


def request_one(prompt):
    body = {
        "messages": prompt["messages"][:-1],  # drop dummy assistant turn
        "max_tokens": 250,
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
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        response = data["choices"][0]["message"].get("content") or ""
    except Exception:
        response = ""
    return {"code": prompt["code"], "response": response}


with ThreadPoolExecutor(max_workers=4) as exe:
    futures = [exe.submit(request_one, p) for p in prompts]
    results = [f.result() for f in as_completed(futures)]

sums = {"parses": 0.0, "entity_recall": 0.0, "entity_precision": 0.0,
        "concise": 0.0, "composite": 0.0}
n = 0
for r in results:
    s = score_response(r["code"], r["response"])
    for k in sums:
        sums[k] += s[k]
    n += 1

print(f"SCORE={sums['composite']/n:.4f}")
print(f"parses={sums['parses']/n:.4f}")
print(f"entity_recall={sums['entity_recall']/n:.4f}")
print(f"entity_precision={sums['entity_precision']/n:.4f}")
print(f"concise={sums['concise']/n:.4f}")
print(f"N={n}")
PY
