"""Initialize a single rollout's workdir for one pi-code-search task.

Task spec format:
{
  "task_id": str,
  "question_kind": "define" | "refs",
  "symbol": str,           # the symbol being asked about
  "repo_id": "kiln",       # the repo to search inside
  "gold": [[file, line], ...],
  "target_bytes": int,     # gold-optimal grep output size
  "question": str,         # rendered natural-language question (optional)
}

Per-rollout workdir layout:
  <dir>/
    repo/         # SYMLINK to a shared read-only snapshot of the repo
    QUESTION.md   # the question (the same as what's in the pi prompt)
    HOW_TO.md     # static guidance — read it if you want; gold-set firewall safe

The shared snapshot is at $PI_CODE_SEARCH_REPO (default
/workspace/kiln-snapshot). The snapshot is created once on pod setup
before any rollout; symlinks are per-rollout so concurrent pi sessions
can share the snapshot safely.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_SNAPSHOT = os.environ.get("PI_CODE_SEARCH_REPO", "/workspace/kiln-snapshot")


def _render_question(task: dict) -> str:
    kind = task.get("question_kind", "define")
    symbol = task["symbol"]
    if kind == "define":
        return (
            f"Where is `{symbol}` defined in the codebase under `repo/`?\n"
            f"Answer with the file path (relative to `repo/`) and line "
            f"number in the form `path/to/file.ext:LINE`."
        )
    if kind == "refs":
        return (
            f"Find all references to `{symbol}` in the codebase under "
            f"`repo/`. Answer with one line per reference in the form "
            f"`path/to/file.ext:LINE`."
        )
    raise ValueError(f"unknown question_kind: {kind}")


def init_workdir(task: dict, dir: str) -> None:
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    # Symlink the read-only repo snapshot under repo/
    repo_link = dir / "repo"
    if repo_link.exists() or repo_link.is_symlink():
        try:
            repo_link.unlink()
        except FileNotFoundError:
            pass
    snap = Path(REPO_SNAPSHOT)
    if not snap.exists():
        raise FileNotFoundError(
            f"repo snapshot {snap} not found. Set $PI_CODE_SEARCH_REPO or "
            f"run scripts/setup_repo_snapshot.sh first."
        )
    repo_link.symlink_to(snap)

    question = _render_question(task)
    (dir / "QUESTION.md").write_text(question + "\n")
    (dir / "HOW_TO.md").write_text(
        "You are searching the codebase under `repo/`. Prefer `grep`, "
        "`rg` (ripgrep), `glob`, or `find` over reading whole files. "
        "Reading large files wastes context and is penalized.\n\n"
        "Tools available:\n"
        "- bash: run shell commands (use grep, rg, find, ls, etc.).\n"
        "- read: read a file's contents (avoid on files >2KB).\n"
        "- glob: find files matching a pattern.\n\n"
        "When you are confident, emit a final assistant message containing "
        "your answer in the form `path:line` (relative to `repo/`). "
        "Multiple answers should be on separate lines.\n"
    )


def pi_prompt(task: dict) -> str:
    """The user-facing pi prompt for one rollout. The system message is
    set by pi's own configuration; this is the user turn."""
    question = _render_question(task)
    return (
        "You are searching a code repository under `repo/`.\n\n"
        "PROBLEM:\n"
        f"{question}\n\n"
        "STRATEGY: prefer `grep` / `rg` / `glob` / `find` over reading "
        "whole files. A focused regex search is usually enough. Avoid "
        "calling `read` on files larger than 2KB.\n\n"
        "ANSWER FORMAT:\n"
        "- Strip the leading `repo/` from any path.\n"
        "- One `path:line` per line (path relative to repo root).\n"
        "- No prose, no backticks, no explanations.\n\n"
        "EXAMPLE:\n"
        "  crates/kiln-train/src/trainer.rs:7270\n"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: task_scaffold.py <task.json> <dir>", file=sys.stderr)
        sys.exit(2)
    task = json.loads(Path(sys.argv[1]).read_text())
    init_workdir(task, sys.argv[2])
    print(f"scaffolded {sys.argv[2]} for {task['task_id']}")
