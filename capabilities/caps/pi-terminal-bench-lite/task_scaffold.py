"""Initialize a fresh workdir for one TBLite-style task and emit the pi
prompt that scaffolds the model toward the task.

A task spec (in datasets/train.tasks.jsonl) looks like:

    {
      "task_id": "tblite-0042",
      "category": "data_processing",
      "scaffold_files": {
        "input.csv": "name,age\\nalice,30\\nbob,25\\n",
        "README.md": "Filter rows where age > 26 and write to output.csv"
      },
      "user_prompt": "Read input.csv and write rows with age > 26 to output.csv.",
      "verifier": "diff -q output.csv expected.csv",
      "verifier_timeout_s": 30,
      "expected_files": {
        "expected.csv": "name,age\\nalice,30\\n"
      }
    }

Hidden expected_files are written to the workdir AFTER pi exits so the
verifier can compare. The model sees only the scaffold_files +
user_prompt; the verifier files are off-limits to it.
"""
from __future__ import annotations

import json
from pathlib import Path


def init_workdir(task: dict, workdir: Path) -> None:
    """Materialize scaffold files. Returns nothing; the workdir is
    populated in place."""
    workdir.mkdir(parents=True, exist_ok=True)
    for rel_path, content in (task.get("scaffold_files") or {}).items():
        target = workdir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content)


def post_rollout_setup(task: dict, workdir: Path) -> None:
    """After pi exits, write any hidden expected_files so the verifier
    can compare. Called by rollout.py just before invoking the
    verifier."""
    for rel_path, content in (task.get("expected_files") or {}).items():
        target = workdir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content)


def pi_prompt(task: dict) -> str:
    """Build the user prompt pi sees."""
    base = task.get("user_prompt", "").strip()
    extra = task.get("extra_context", "").strip()
    if extra:
        return f"{base}\n\n{extra}"
    return base
