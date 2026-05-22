"""iter13 OPD prep: self-distillation prompts.

Build opd.prompts.jsonl from sft.train.jsonl:
  - messages: [default_system, user]  (student sees this)
  - teacher_extra_messages: [strict_system]
    (teacher sees [strict, default, user] — strict prompt primes the
     distribution we want the student to inherit)

Hypothesis: OPD reverse-KL pulls the student's token distribution toward
the teacher's distribution (which is the same model + strict prompt =
0.819 composite). Different signal than SFT — no surface mimicry, just
distribution matching. Should break past the 0.7735 SFT plateau.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IN = ROOT / "datasets/sft.train.jsonl"
STRICT = (ROOT / "prompts/h15-strict-system-prompt-system.txt").read_text()
OUT = ROOT / "datasets/opd.prompts.jsonl"

n = 0
with IN.open() as f, OUT.open("w") as out:
    for line in f:
        d = json.loads(line)
        # Drop the assistant message — OPD doesn't want it (student samples its own rollout)
        msgs = [m for m in d["messages"] if m["role"] != "assistant"]
        prompt = {
            "messages": msgs,
            "teacher_extra_messages": [{"role": "system", "content": STRICT}],
        }
        out.write(json.dumps(prompt, ensure_ascii=False) + "\n")
        n += 1

print(f"wrote {n} OPD prompts -> {OUT}")
