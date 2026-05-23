"""Initialize a rollout workdir from a task spec, and build the pi prompt.

A task dict (from datasets/*.tasks.jsonl) carries:

    init_files          dict[path -> content]   initial workspace state
    gold_files          dict[path -> content]   reference solution
    patch_text          str                      the unified diff the model sees
    gold_diff_text      str                      the canonical correct diff
    gold_diff_lines     int                      ≈ #lines the gold diff touches
    touched_paths       list[str]                files the patch is meant to modify
    protected_paths     list[str]                files pi should not touch (tests)
    expected_repair     bool                     true iff patch_class != 'clean'
    verify_cmd          str                      e.g. "python3 -m pytest -q tests/"
    verify_timeout_s    int                      verifier timeout

`init_workdir(task, dir)` writes:

    <dir>/<every init_files path>     the initial workspace contents
    <dir>/INCOMING_PATCH               the patch text the model will apply
    <dir>/README.md                    one-line task description

`pi_prompt(task)` returns the prompt sent to pi via `-p`.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import subprocess
from pathlib import Path


PATCH_FILENAME = "INCOMING_PATCH"


def init_workdir(task: dict, dir: str) -> None:
    """Set up a fresh rollout workdir. Idempotent — overwrites if exists.

    Initializes a git repo inside the workdir so `git apply` works (which
    pi will usually try). The model can also choose to use `patch` instead.
    """
    workdir = Path(dir)
    if workdir.exists():
        # Clear contents but keep the directory itself.
        for child in workdir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    workdir.mkdir(parents=True, exist_ok=True)

    # Write init files.
    init_files: dict[str, str] = task.get("init_files") or {}
    for rel, content in init_files.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    # Write the patch file the model will be told to apply.
    patch_text: str = task.get("patch_text") or ""
    (workdir / PATCH_FILENAME).write_text(patch_text)

    # Write README to orient pi.
    expected_repair = task.get("expected_repair", False)
    repair_note = (
        "The patch may not apply cleanly the first time — if `git apply` "
        "rejects a hunk, read the file and fix the diff or apply the change "
        "by hand."
        if expected_repair
        else "The patch should apply cleanly with `git apply`."
    )
    readme = (
        f"# Task {task.get('task_id', '?')}\n\n"
        f"Apply the patch in `{PATCH_FILENAME}` to this workspace, then run "
        f"the tests until they pass.\n\n"
        f"{repair_note}\n\n"
        "Touched paths (per patch headers):\n"
        + "\n".join(f"- {p}" for p in (task.get("touched_paths") or []))
        + "\n\nVerify with:\n\n    "
        + (task.get("verify_cmd") or "python3 -m pytest -q tests/")
        + "\n"
    )
    (workdir / "README.md").write_text(readme)

    # Initialize a git repo so `git apply` always has a working repo state.
    # We init with -q and an initial commit on the buggy state — this gives
    # `git apply --3way` something to anchor against.
    try:
        subprocess.run(
            ["git", "init", "-q", "-b", "main"],
            cwd=str(workdir),
            check=False,
            capture_output=True,
            timeout=15,
        )
        subprocess.run(
            ["git", "config", "user.email", "rollout@kiln.local"],
            cwd=str(workdir), check=False, capture_output=True, timeout=5,
        )
        subprocess.run(
            ["git", "config", "user.name", "kiln-rollout"],
            cwd=str(workdir), check=False, capture_output=True, timeout=5,
        )
        subprocess.run(
            ["git", "add", "-A", "--", ":!INCOMING_PATCH"],
            cwd=str(workdir), check=False, capture_output=True, timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-q", "-m", "initial"],
            cwd=str(workdir), check=False, capture_output=True, timeout=10,
        )
    except Exception:
        # git init failures are non-fatal — pi can still use `patch` or
        # raw file edits via the edit tool.
        pass


def pi_prompt(task: dict) -> str:
    """The user prompt pi sees. Keep it terse and concrete; don't reveal
    the rubric or the gold files.
    """
    expected_repair = task.get("expected_repair", False)
    repair_hint = (
        " The patch may have minor offset issues or a subtle bug; if the first apply "
        "fails or tests fail, read the relevant file and repair the change minimally."
        if expected_repair
        else " The patch should apply cleanly."
    )
    return (
        f"Apply the unified diff in `{PATCH_FILENAME}` to this workspace, then run "
        f"`{task.get('verify_cmd') or 'python3 -m pytest -q tests/'}` until all tests pass.\n\n"
        "Steps:\n"
        "1. Read INCOMING_PATCH to see what change is intended.\n"
        "2. Apply it (try `git apply INCOMING_PATCH` or `patch -p1 < INCOMING_PATCH`).\n"
        "3. Run the tests with the verify command.\n"
        f"4. If a hunk rejects, read the affected source file, find the right anchor, and apply the change minimally.{repair_hint}\n"
        "5. When all tests pass, emit a final message summarizing what you changed "
        "and which files you touched. Do NOT modify the test files."
    )


def build_messages(task: dict) -> list[dict]:
    """The system + user messages pi will see. Use this for the GrpoGroup
    `messages` field in training rollouts."""
    return [
        {
            "role": "system",
            "content": (
                "You are a meticulous coding assistant operating in a sandboxed "
                "workspace. You have access to bash, write, read, and edit tools. "
                "Apply patches minimally — do NOT rewrite functions when a "
                "3-line edit will do. Always verify with the test command before "
                "declaring done."
            ),
        },
        {"role": "user", "content": pi_prompt(task)},
    ]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: task_scaffold.py <task.json|tasks.jsonl#N> <dir>", file=sys.stderr)
        sys.exit(2)
    spec = sys.argv[1]
    out = sys.argv[2]
    if "#" in spec:
        path, idx = spec.split("#")
        idx = int(idx)
        with open(path) as f:
            for i, line in enumerate(f):
                if i == idx:
                    task = json.loads(line)
                    break
            else:
                raise SystemExit(f"no line {idx} in {path}")
    else:
        task = json.loads(Path(spec).read_text())
    init_workdir(task, out)
    print(f"scaffolded {out}")
    print(json.dumps(build_messages(task), indent=2)[:500])
