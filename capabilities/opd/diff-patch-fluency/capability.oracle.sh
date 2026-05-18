#!/usr/bin/env bash
# Blind oracle for diff-patch-fluency.
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
        "max_tokens": 512,
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
    return {
        "response": resp,
        "source": prompt["source"],
        "intent_keywords": prompt.get("intent_keywords", []),
        "intent_anti_keywords": prompt.get("intent_anti_keywords", []),
        "expected_line_changes": prompt.get("expected_line_changes", 0),
        "source_path": prompt.get("source_path", "src/target.txt"),
    }


# Sequential — `patch` subprocess in score_response can be slow when
# many run in parallel; eval set is small enough.
results = [request_one(p) for p in prompts]

sums = {"strict_format": 0.0, "applies_cleanly": 0.0, "target_intent_captured": 0.0,
        "minimal_changes": 0.0, "composite": 0.0}
n = 0
for r in results:
    s = score_response(**r)
    for k in sums:
        sums[k] += s[k]
    n += 1

print(f"SCORE={sums['composite']/n:.4f}")
for k in ["strict_format", "applies_cleanly", "target_intent_captured", "minimal_changes"]:
    print(f"{k}={sums[k]/n:.4f}")
print(f"N={n}")
PY
