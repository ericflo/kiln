"""Eval a kiln-served adapter (or base) on the JSON-schema-adherence rubric.

Hits `http://localhost:8420/v1/chat/completions` per prompt in
`datasets/eval.jsonl`, builds a request with the schema-strict instruction,
and aggregates per-prompt scores via `rubric.score_response`.

Usage:
    python3 eval_kiln.py --adapter base --out judgments/baseline.json
    python3 eval_kiln.py --adapter opd-json-v1 --out judgments/opd-v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rubric import score_response

KILN_URL = "http://localhost:8420/v1/chat/completions"
EVAL_FILE = Path(__file__).parent / "datasets" / "eval.jsonl"
JUDGMENTS_DIR = Path(__file__).parent / "judgments"
JUDGMENTS_DIR.mkdir(exist_ok=True)

# A consistent system message that gives both the rubric description and the
# schema. The 27B teacher will generate completions using the *same* system
# message during teacher-fixture pre-compute, so student and teacher see
# identical inputs.
SYSTEM = (
    "You are a strict structured-output assistant. Given a user request and a "
    "JSON Schema, you reply with ONE JSON object that:\n"
    "  1. parses as valid JSON,\n"
    "  2. validates against the schema,\n"
    "  3. has substantive content (no placeholder strings),\n"
    "  4. is the ENTIRE response — no preamble, no postamble, no markdown "
    "fences, no commentary.\n"
)


def build_messages(query: str, schema: dict) -> list[dict]:
    schema_str = json.dumps(schema, indent=2)
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Request: {query}\n\nJSON Schema:\n{schema_str}\n\nReturn only the JSON object.",
        },
    ]


def request_one(prompt: dict, adapter: str | None, max_tokens: int, timeout: int) -> dict:
    body = {
        "messages": build_messages(prompt["query"], prompt["schema"]),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if adapter and adapter not in ("base", "none", ""):
        body["adapter"] = adapter
    req = urllib.request.Request(
        KILN_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        msg = data["choices"][0]["message"]
        # Strict-JSON capability: we only consider `content`. If the model
        # leaked into reasoning_content, that's a rubric failure.
        response = msg.get("content") or ""
        finish = data["choices"][0].get("finish_reason")
    except Exception as e:
        response = ""
        finish = f"error:{e}"
    return {"id": prompt["id"], "response": response, "finish_reason": finish}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="base", help="adapter name or 'base'")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument(
        "--eval-file",
        default=str(EVAL_FILE),
        help="JSONL of {id, query, schema} prompts",
    )
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    prompts = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    print(f"Eval: {len(prompts)} prompts, adapter={args.adapter}", file=sys.stderr)

    t0 = time.time()
    responses: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(request_one, p, args.adapter, args.max_tokens, args.timeout): p
            for p in prompts
        }
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            responses[r["id"]] = r
            done += 1
            if done % 5 == 0 or done == len(prompts):
                print(f"  {done}/{len(prompts)}", file=sys.stderr)

    # Score
    rows = []
    for p in prompts:
        rid = p["id"]
        resp = responses[rid]["response"]
        finish = responses[rid]["finish_reason"]
        s = score_response(resp, p["schema"])
        s["id"] = rid
        s["finish_reason"] = finish
        s["response_chars"] = len(resp)
        rows.append(s)

    def mean(k: str) -> float:
        return sum(r[k] for r in rows) / len(rows) if rows else 0.0

    agg = {
        "adapter": args.adapter,
        "n": len(rows),
        "wall_time_s": round(time.time() - t0, 2),
        "parses": round(mean("parses"), 4),
        "validates": round(mean("validates"), 4),
        "is_pure": round(mean("is_pure"), 4),
        "is_substantive": round(mean("is_substantive"), 4),
        "composite": round(mean("composite"), 4),
        "per_prompt": rows,
        # All responses (truncated to 800 chars each) for post-hoc analysis.
        "responses": {
            r["id"]: responses[r["id"]]["response"][:800]
            for r in rows
        },
    }
    Path(args.out).write_text(json.dumps(agg, indent=2))
    print(json.dumps({k: agg[k] for k in ("adapter","n","parses","validates","is_pure","is_substantive","composite","wall_time_s")}))


if __name__ == "__main__":
    main()
