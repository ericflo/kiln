"""H6 SFT cold-start data generation.

For each of our 24 training prompts, ask the 27B teacher (via vLLM)
to produce a diff. Format the result as SFT JSONL.

Output: datasets/h6-sft.jsonl
Each line: {"messages": [system, user, assistant=teacher's diff]}
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TEACHER_URL = "http://localhost:8002/v1/chat/completions"
TEACHER_MODEL = "qwen3.6-27b-awq"
WORKDIR = Path(__file__).resolve().parent
TRAIN_PATH = WORKDIR / "prompts/h1-r16-6ep.jsonl"
OUT_PATH = WORKDIR / "datasets/h6-sft.jsonl"


def request_teacher_diff(prompt: dict) -> str:
    messages = prompt["messages"][:-1]  # drop dummy assistant
    body = {
        "model": TEACHER_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0,
        # Qwen3.6 is a thinking model — disable the <think> block so the
        # response is just the diff, not reasoning preamble that would
        # poison the SFT target.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        TEACHER_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    raw = d["choices"][0]["message"].get("content") or ""
    # If a leftover <think>...</think> block exists, strip it.
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    return raw.strip()


def main() -> None:
    prompts = []
    with TRAIN_PATH.open() as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    print(f"generating teacher diffs for {len(prompts)} prompts...")

    t0 = time.time()
    results: list[dict | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=4) as exe:
        futs = {exe.submit(request_teacher_diff, p): i for i, p in enumerate(prompts)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                diff = fut.result()
                results[i] = {"id": prompts[i]["id"], "diff": diff}
                print(f"  [{i + 1}/{len(prompts)}] {prompts[i]['id']} ({len(diff)} chars)")
            except Exception as e:
                print(f"  [{i + 1}/{len(prompts)}] ERROR: {e}", file=sys.stderr)
                results[i] = None

    elapsed = time.time() - t0
    print(f"teacher generation: {elapsed:.1f}s")

    written = 0
    with OUT_PATH.open("w") as f:
        for i, prompt in enumerate(prompts):
            if results[i] is None:
                continue
            messages = list(prompt["messages"][:-1])
            messages.append({"role": "assistant", "content": results[i]["diff"]})
            f.write(json.dumps({"messages": messages}) + "\n")
            written += 1
    print(f"wrote {OUT_PATH} ({written} examples)")


if __name__ == "__main__":
    main()
