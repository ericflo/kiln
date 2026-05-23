"""Initialize a single rollout's workdir from a task spec.

A task spec is a dict:
{
  "task_id": str,
  "function_signature": "def foo(x: int) -> int:\n    \"\"\"docstring with doctests\"\"\"\n",
  "imports": "from typing import List, Tuple\n",   # optional
}

`init_workdir(task, dir)` writes:
  <dir>/solution.py     stub function with `raise NotImplementedError`
  <dir>/README.md       one-line task description for pi to read

The pi prompt itself is built by `pi_prompt(task)`.
"""

import json
import os
import sys
from pathlib import Path


def init_workdir(task: dict, dir: str) -> None:
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    imports = task.get("imports", "")
    sig = task["function_signature"]
    stub = f"{imports}\n{sig}    raise NotImplementedError\n"
    (dir / "solution.py").write_text(stub)
    (dir / "README.md").write_text(
        "Replace the function body in solution.py so its doctests pass.\n"
        "Then run `python3 -m doctest -v solution.py` to verify.\n"
    )


def pi_prompt(task: dict) -> str:
    return (
        "In the file `solution.py` there is a stub Python function with a "
        "docstring containing doctest examples. Replace the function body "
        "so the doctests pass.\n\n"
        "Steps:\n"
        "1. Read solution.py to see the function signature and doctests.\n"
        "2. Use the `edit` or `write` tool to replace the function body with "
        "a correct implementation.\n"
        "3. Run `python3 -m doctest -v solution.py` via the `bash` tool.\n"
        "4. If all items pass (no failures), reply with a single word `DONE` "
        "and end the session.\n\n"
        "Constraint: do not modify the function signature or the docstring. "
        "Only the function body should change."
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: task_scaffold.py <task.json> <dir>", file=sys.stderr)
        sys.exit(2)
    task = json.loads(Path(sys.argv[1]).read_text())
    init_workdir(task, sys.argv[2])
    print(f"scaffolded {sys.argv[2]}")
