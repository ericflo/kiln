"""Task scaffold: write a single rollout's workdir from a task spec.

A task spec is a dict:
{
  "task_id": str,
  "description": str,              # one-liner for the agent prompt
  "workspace_files": { rel: str }, # the buggy workspace (visible)
  "visible_test_cmd": [argv...],   # how to run the visible test
  "held_out_test_path": str,       # relative path; written ONLY at oracle time
  "held_out_test_content": str,    # the held-out test content
  "held_out_test_cmd": [argv...],  # how to run held-out (after mount)
  "gold_fix_region": {"file": str, "function": str, "module": str},
  "expected_diff_lines": int,
  "test_timeout_s": int            # optional override
}

`init_workdir(task, dir)` writes the visible workspace into `dir`. The
held-out test is NOT written; it's only materialized by the rubric at
scoring time.

The pi prompt itself is built by `pi_prompt(task)` from the task's
`description` field and a fixed scaffold.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def init_workdir(task: dict, dir: str) -> None:
    """Materialize the visible workspace files into `dir`."""
    root = Path(dir)
    root.mkdir(parents=True, exist_ok=True)
    files = task.get("workspace_files") or {}
    for rel, content in files.items():
        if not isinstance(rel, str) or not isinstance(content, str):
            continue
        out = root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content)


def pi_prompt(task: dict) -> str:
    desc = task.get("description", "There is a bug in this project. Fix it.")
    test_hint = " ".join(task.get("visible_test_cmd", ["pytest", "-x"]))
    return (
        f"This project has a failing test that exposes a bug. {desc}\n\n"
        "Steps:\n"
        "1. Read the source files in this directory to understand the code.\n"
        f"2. Reproduce the failure by running: `{test_hint}`\n"
        "3. Identify the ROOT CAUSE — not just the line that throws. Symptom\n"
        "   fixes (wrapping calls in `try/except`, hardcoded values, or\n"
        "   suppressing the error) will not satisfy the success criteria;\n"
        "   the fix must address the actual underlying behavior so similar\n"
        "   inputs behave correctly.\n"
        "4. Edit the source file (NOT the test file) to fix the bug.\n"
        f"5. Re-run `{test_hint}` to verify the test now passes.\n"
        "6. End by emitting a single line of the form:\n"
        "      Fix: <relative_path>::<function_name>: <one-line root cause>\n"
        "   then exit the session.\n\n"
        "Constraints:\n"
        "- Do NOT modify any file under `tests/`. The test is the spec.\n"
        "- Do NOT use blanket `except Exception:` / `except:`. They suppress\n"
        "  signal.\n"
        "- Do NOT add `# noqa`, `# type: ignore`, or `# pragma: no cover`.\n"
        "- Do NOT change pyproject.toml / requirements.txt / Cargo.toml.\n"
        "- Keep the diff minimal — change only what's necessary to fix the\n"
        "  root cause.\n"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: task_scaffold.py <task.json> <dir>", file=sys.stderr)
        sys.exit(2)
    task = json.loads(Path(sys.argv[1]).read_text())
    init_workdir(task, sys.argv[2])
    print(f"scaffolded {sys.argv[2]}")
