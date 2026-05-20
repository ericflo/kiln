"""Initialize a pi rollout workdir from a task spec.

A task spec is a dict:
{
  "task_id": str,
  "target_file": "lib/foo.py",            # path INSIDE the workdir snapshot
  "target_symbol": "function_name",
  "gold": { ... structured gold summary ... },
  "files": {                              # path → file content (relative)
      "lib/foo.py": "...",
      "lib/bar.py": "...",
      ...
  },
}

`init_workdir(task, dir)`:
  - copies each entry of `files` into <dir>/<path>
  - writes <dir>/README.md describing the task
  - does NOT include `gold` (the model must not read it)

`pi_prompt(task)`:
  - returns the user prompt pi reads

Empirical pi config used (verified 2026-05-18 in pi-doctest):
  - pi reads its workdir as $CWD
  - tools available: read, edit, write, bash
  - session JSONL ends up under the configured --session-dir
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def init_workdir(task: dict, dir: str) -> None:
    workdir = Path(dir)
    workdir.mkdir(parents=True, exist_ok=True)
    files = task.get("files") or {}
    for rel_path, content in files.items():
        if not isinstance(rel_path, str) or not isinstance(content, str):
            continue
        # Never escape the workdir.
        rel = rel_path.lstrip("/").replace("\\", "/")
        if ".." in rel.split("/"):
            continue
        dest = workdir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)

    target_file = task.get("target_file", "")
    target_symbol = task.get("target_symbol", "")
    readme = (
        "# Code comprehension task\n\n"
        f"Target file:   `{target_file}`\n"
        f"Target symbol: `{target_symbol}`\n\n"
        "Goal: produce a structured JSON summary of the target symbol's\n"
        "inputs, returns, mutations, calls, callers, invariants and side\n"
        "effects, with line-number citations from the source.\n\n"
        "See the user prompt for the exact JSON schema.\n"
    )
    (workdir / "README.md").write_text(readme)


# The pi user prompt. This is the only input the model sees besides
# tool outputs. The exact JSON schema is spelled out so a well-trained
# model can fill it without further hints.
PI_PROMPT_TEMPLATE = """/no_think You are exploring a small code repository in the current working
directory. Your goal is to produce a STRUCTURED JSON SUMMARY of the
target symbol below.

Target file:   {target_file}
Target symbol: {target_symbol}

Steps:
1. Use `read` to read the target file.
2. Use `bash` with `grep -rn {target_symbol} .` to find callers.
3. Read 1-2 callers if helpful.

When you have all the information you need, end your work by emitting a
PLAIN ASSISTANT TEXT MESSAGE (no tool calls) containing this exact form:

<answer>
{{"inputs": [{{"name": "arg1", "type": "str", "source_line": N}}], "returns": [{{"type": "T", "source_line": N}}], "mutates": ["arg:foo"], "calls": [{{"name": "helper", "file": "file.py", "line": N}}], "called_by": [{{"file": "caller.py", "line": N}}], "invariants": ["..."], "side_effects": ["raises X"]}}
</answer>

Rules:
- DO NOT call any tool named `answer` — there is no such tool. Use a
  plain assistant text turn instead.
- DO NOT pass the JSON via `write` or `bash` — emit it as ordinary text.
- Line numbers MUST be cited from real source. Omit if unknown.
- For empty fields use the empty list `[]`.
- `invariants` includes implicit preconds (lock must be held, init must
  have been called first), not just docstring text.
- `called_by` is the cross-file callers — find them with grep.
- ONE `<answer>` block. End the session after it.
"""


def pi_prompt(task: dict) -> str:
    return PI_PROMPT_TEMPLATE.format(
        target_file=task.get("target_file", "<unknown>"),
        target_symbol=task.get("target_symbol", "<unknown>"),
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: task_scaffold.py <task.json> <dir>", file=sys.stderr)
        sys.exit(2)
    task = json.loads(Path(sys.argv[1]).read_text())
    init_workdir(task, sys.argv[2])
    print(pi_prompt(task))
