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
PI_PROMPT_TEMPLATE = """You are exploring a small code repository in the current working
directory. Your goal is to produce a STRUCTURED JSON SUMMARY of the
target symbol below.

Target file:   {target_file}
Target symbol: {target_symbol}

Use the available tools to:
1. `read` the target file and any helpers / callers you need.
2. `bash` (run `grep -rn <symbol> .` etc.) to find cross-file callers.
3. Read 1-2 callers to confirm conventions (optional).

When you have enough information, emit a FINAL assistant turn (no tool
calls) containing exactly one `<answer>` block with a JSON object using
this schema:

<answer>
{{
  "inputs":   [{{"name": "arg1", "type": "str", "source_line": N}}, ...],
  "returns":  [{{"type": "type_expr", "source_line": N}}],
  "mutates":  ["filesystem:/path", "global:STATE", "arg:cache_arg", ...],
  "calls":    [{{"name": "helper", "file": "file.py", "line": N}}, ...],
  "called_by":[{{"file": "caller.py", "line": N}}, ...],
  "invariants":   ["invariant 1", "invariant 2", ...],
  "side_effects": ["raises X on Y", "writes log", ...]
}}
</answer>

Rules:
- Line numbers MUST be cited from the actual source. If unknown, omit the
  field rather than guess.
- For functions with NO mutations, use the empty list [].
- `invariants` should include implicit ones too (e.g. "lock must be
  held", "init() must have been called first"), not just docstring text.
- `called_by` should reflect callers across ALL files, found via grep.
- One JSON object inside ONE `<answer>` block. No other prose afterward.
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
