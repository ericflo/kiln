"""Generate iter12 mixed-prompt SFT dataset.

Each existing example becomes TWO examples:
  1. (default_system, user) → strict_output   [as-is]
  2. (strict_system, user)  → strict_output   [new: reinforces strict-when-asked]

Hypothesis: training on BOTH prompt conditions but with the same strict output
teaches the model that the strict-style output is invariant to the system
prompt — closing the gap between with-strict-prompt (0.82) and no-prompt (~0.77)
behavior.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IN = ROOT / "datasets/sft.train.jsonl"
STRICT = (ROOT / "prompts/h15-strict-system-prompt-system.txt").read_text()
OUT = ROOT / "datasets/sft.mix.jsonl"

n_in = n_out = 0
with IN.open() as f, OUT.open("w") as out:
    for line in f:
        d = json.loads(line)
        n_in += 1
        msgs = d["messages"]
        # Locate the assistant content
        assistant = next((m for m in msgs if m["role"] == "assistant"), None)
        user = next((m for m in msgs if m["role"] == "user"), None)
        if assistant is None or user is None:
            continue

        # As-is (default system prompt)
        out.write(json.dumps(d, ensure_ascii=False) + "\n")
        n_out += 1

        # Strict system prompt variant
        strict_msgs = [
            {"role": "system", "content": STRICT},
            {"role": "user", "content": user["content"]},
            {"role": "assistant", "content": assistant["content"]},
        ]
        out.write(json.dumps({"messages": strict_msgs}, ensure_ascii=False) + "\n")
        n_out += 1

print(f"in={n_in} out={n_out} -> {OUT}")
