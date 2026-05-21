"""Build pi-doctest train + eval task JSONL from kiln's humaneval pool.

Inputs:
  capabilities/caps/python-algo/datasets/grpo-humaneval-best.jsonl
  (or grpo-humaneval.jsonl, grpo-humaneval-big.jsonl)

For each prompt in the source, we extract:
  - function signature (def line + docstring up to first `>>>`)
  - the doctest examples (already in the docstring)
  - any preceding `from typing import ...` lines

We emit:
  capabilities/caps/pi-doctest/datasets/train.tasks.jsonl
  capabilities/caps/pi-doctest/datasets/eval.tasks.jsonl

Each task line:
  {"task_id": str, "function_signature": str, "imports": str}

Eval is the first ~24 tasks (mirrors the cuda_grpo_behavioral_eval
defaults). Train is the rest.

The script is idempotent. Seed-controlled if you ever want
randomization, but for now we preserve source order.
"""

import json
import re
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent.parent.parent / "capabilities/caps/python-algo/datasets"
DST = Path(__file__).parent / "datasets"


def extract_task(prompt_text: str, task_id: str) -> dict | None:
    # Find the fenced ```python block and pull the def line + docstring.
    m = re.search(r"```python\n(.*?)\n```", prompt_text, re.DOTALL)
    if not m:
        return None
    code = m.group(1)

    # Split out imports (top-level `from ... import` / `import ...` lines)
    # and the def block.
    lines = code.split("\n")
    imports = []
    body_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(("import ", "from ")):
            imports.append(line)
        elif s.startswith("def "):
            body_start = i
            break
    if body_start is None:
        return None

    # Function signature is from `def ...` to the end of the docstring.
    # Find the closing `"""` that terminates the docstring.
    sig_lines = lines[body_start:]
    # Walk to find docstring start (first occurrence of `"""` or `'''`)
    in_doc = False
    doc_quote = None
    sig_end = None
    for j, line in enumerate(sig_lines):
        if not in_doc:
            for q in ('"""', "'''"):
                if q in line:
                    in_doc = True
                    doc_quote = q
                    if line.count(q) >= 2:
                        # Single-line docstring; ends here.
                        sig_end = j
                        in_doc = False
                    break
        else:
            if doc_quote in line:
                sig_end = j
                in_doc = False
                break
    if sig_end is None:
        return None

    sig = "\n".join(sig_lines[: sig_end + 1]) + "\n"
    imports_text = ("\n".join(imports) + "\n") if imports else ""
    return {
        "task_id": task_id,
        "function_signature": sig,
        "imports": imports_text,
    }


def main():
    DST.mkdir(parents=True, exist_ok=True)
    source = SRC / "grpo-humaneval-best.jsonl"
    if not source.exists():
        source = SRC / "grpo-humaneval.jsonl"
    if not source.exists():
        print(f"ERROR: no humaneval source under {SRC}", file=sys.stderr)
        sys.exit(1)

    tasks = []
    seen_sig = set()
    with source.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                grp = json.loads(line)
            except Exception:
                continue
            prompt = (grp.get("messages") or [{}])[0].get("content", "")
            t = extract_task(prompt, task_id=f"task_{i:04d}")
            if t is None:
                continue
            # Dedup by signature.
            if t["function_signature"] in seen_sig:
                continue
            seen_sig.add(t["function_signature"])
            tasks.append(t)

    eval_set = tasks[:24]
    train_set = tasks[24:]

    with (DST / "eval.tasks.jsonl").open("w") as f:
        for t in eval_set:
            f.write(json.dumps(t) + "\n")
    with (DST / "train.tasks.jsonl").open("w") as f:
        for t in train_set:
            f.write(json.dumps(t) + "\n")

    print(f"wrote {len(eval_set)} eval tasks, {len(train_set)} train tasks")
    print(f"  eval:  {DST / 'eval.tasks.jsonl'}")
    print(f"  train: {DST / 'train.tasks.jsonl'}")


if __name__ == "__main__":
    main()
