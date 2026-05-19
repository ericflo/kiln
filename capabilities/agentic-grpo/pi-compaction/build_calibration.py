"""Generate calibration good.jsonl / bad.jsonl from the first few train tasks.

Strategy:
  - Pick 5 train tasks deterministically.
  - For each task, generate one "good" candidate (pi-format summary built
    programmatically from the ground-truth facts — guaranteed to score well
    on content + faithfulness + format) and several "bad" variants
    (copy-source / empty-template / wrong-format / hallucinated /
    no-continuation-violation).
  - Calibration writes:
      calibration/good.jsonl  — one (task_id, response) per line
      calibration/bad.jsonl   — one (task_id, response, shortcut) per line

This deliberately uses a *programmatic* good response so it isn't tied
to a specific teacher model. A real well-trained model should still beat
the programmatic good in subjective quality, but the programmatic good is
the floor — if the rubric prefers anything lower, the rubric is broken.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import task_scaffold


def goal_section(gt: dict[str, Any]) -> str:
    goal = (gt.get("first_user_goal") or "").strip()
    if not goal:
        return "## Goal\n(no clear user goal stated)\n"
    # Trim to first paragraph
    first_para = goal.split("\n\n", 1)[0]
    first_para = first_para[:600].strip()
    return f"## Goal\n{first_para}\n"


def constraints_section() -> str:
    return "## Constraints & Preferences\n- (none)\n"


def progress_section(gt: dict[str, Any]) -> str:
    read_only = gt.get("read_only_paths") or []
    modified = gt.get("modified_paths") or []
    errors = gt.get("source_errors") or []

    done_lines = []
    # Include up to 6 read/modify entries to maximise file-block recall in
    # case the file blocks are dropped.
    for p in read_only[:6]:
        done_lines.append(f"- [x] Read `{p}`")
    for p in modified[:6]:
        done_lines.append(f"- [x] Modified `{p}`")
    if not done_lines:
        done_lines.append("- [x] Initial exploration of source files")

    in_progress_lines = []
    idents = gt.get("source_identifiers") or []
    if modified:
        in_progress_lines.append(f"- [ ] Continue editing `{modified[0]}`")
    elif read_only:
        in_progress_lines.append(f"- [ ] Reviewing `{read_only[0]}` to inform the next change")
    elif idents:
        in_progress_lines.append(f"- [ ] Investigating `{idents[0]}` and surrounding code")
    else:
        in_progress_lines.append("- [ ] Continue the task")

    blocked_lines = []
    for err in errors[:3]:
        blocked_lines.append(f"- {err[:200]}")
    if not blocked_lines:
        blocked_lines.append("(none)")

    return (
        "## Progress\n"
        "### Done\n"
        + "\n".join(done_lines) + "\n\n"
        + "### In Progress\n"
        + "\n".join(in_progress_lines) + "\n\n"
        + "### Blocked\n"
        + "\n".join(blocked_lines) + "\n"
    )


def key_decisions_section(gt: dict[str, Any]) -> str:
    idents = gt.get("source_identifiers") or []
    if idents:
        return f"## Key Decisions\n- **Focused on `{idents[0]}` first**: load-bearing identifier in the source.\n"
    return "## Key Decisions\n- (none recorded)\n"


def next_steps_section(gt: dict[str, Any]) -> str:
    paths = gt.get("source_paths") or []
    errors = gt.get("source_errors") or []
    lines = []
    if errors:
        lines.append(f"1. Resolve: {errors[0][:120]}")
    if paths:
        for i, p in enumerate(paths[:3], start=len(lines) + 1):
            lines.append(f"{i}. Review `{p}` and continue the work")
    if not lines:
        lines = ["1. Continue from where the assistant left off."]
    return "## Next Steps\n" + "\n".join(lines) + "\n"


def critical_context_section(gt: dict[str, Any]) -> str:
    idents = gt.get("source_identifiers") or []
    paths = gt.get("source_paths") or []
    errors = gt.get("source_errors") or []
    bullet_pool = []
    # Include MORE paths and identifiers — preserving them is the whole point.
    for p in paths[:12]:
        bullet_pool.append(f"- File: `{p}`")
    for i in idents[:12]:
        bullet_pool.append(f"- Identifier: `{i}`")
    for e in errors[:3]:
        bullet_pool.append(f"- Error: {e[:200]}")
    if not bullet_pool:
        bullet_pool = ["- (none)"]
    return "## Critical Context\n" + "\n".join(bullet_pool) + "\n"


def file_blocks(gt: dict[str, Any]) -> str:
    ro = gt.get("read_only_paths") or []
    mod = gt.get("modified_paths") or []
    out = []
    if ro:
        out.append("<read-files>\n" + "\n".join(ro) + "\n</read-files>")
    if mod:
        out.append("<modified-files>\n" + "\n".join(mod) + "\n</modified-files>")
    return ("\n\n" + "\n\n".join(out)) if out else ""


def make_good(task: dict[str, Any]) -> str:
    gt = task["ground_truth"]
    return (
        goal_section(gt) + "\n"
        + constraints_section() + "\n"
        + progress_section(gt) + "\n"
        + key_decisions_section(gt) + "\n"
        + next_steps_section(gt) + "\n"
        + critical_context_section(gt)
        + file_blocks(gt)
    )


def make_bad_copy_source(task: dict[str, Any]) -> str:
    # No format, just paste source
    return task["source_text"][:4000]


def make_bad_template_only(task: dict[str, Any]) -> str:
    # Right format but every section is "(none)" — no content
    return (
        "## Goal\n(none)\n\n"
        "## Constraints & Preferences\n- (none)\n\n"
        "## Progress\n### Done\n- [x] (none)\n\n### In Progress\n- [ ] (none)\n\n### Blocked\n(none)\n\n"
        "## Key Decisions\n- (none)\n\n"
        "## Next Steps\n1. (none)\n\n"
        "## Critical Context\n- (none)\n"
    )


def make_bad_hallucinated(task: dict[str, Any]) -> str:
    # Right format, fully invented paths/identifiers/errors that aren't in source
    return (
        "## Goal\nFix the broken authentication flow in the loginV2 module.\n\n"
        "## Constraints & Preferences\n- Must keep tokens secret.\n\n"
        "## Progress\n### Done\n- [x] Read `/srv/totally/made-up/auth.go`\n- [x] Patched `LoginHandler.verify_token`\n\n"
        "### In Progress\n- [ ] Update `/srv/totally/made-up/middleware.go`\n\n### Blocked\n- TokenExpiredError on every other request\n\n"
        "## Key Decisions\n- **Switched from JWT to PASETO**: longer-lived sessions.\n\n"
        "## Next Steps\n1. Migrate `LoginHandler.verify_token` to use PASETO.\n2. Audit all callers of `loginV2.authenticate`.\n\n"
        "## Critical Context\n- All routes prefixed with `/api/v2`.\n"
    )


def make_bad_wrong_format(task: dict[str, Any]) -> str:
    # Plain prose summary, no section headings — pi-format gate fails
    return (
        "The user asked the assistant to do something complicated. "
        "The assistant read some files, made some edits, and ran into a few errors. "
        "Some progress was made but the work is not yet complete. "
        "Next time, the assistant should continue where it left off."
    )


def make_bad_continuation(task: dict[str, Any]) -> str:
    # Continues the conversation instead of summarizing — explicit pi rule violation
    return (
        "Sure! I can help with that. Let me start by reading the file you mentioned. "
        "I'd suggest first running `ls -la /tmp` to see what's there. Would you like me to do that?"
    )


SHORTCUTS = [
    ("copy_source", make_bad_copy_source),
    ("template_only", make_bad_template_only),
    ("hallucinated", make_bad_hallucinated),
    ("wrong_format", make_bad_wrong_format),
    ("continuation", make_bad_continuation),
]


def main() -> int:
    random.seed(42)
    out_dir = ROOT / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 5 tasks (skip super-long ones — we want runnable test cases)
    tasks: list[dict[str, Any]] = []
    with (ROOT / "datasets/train.tasks.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            if len(t["source_text"]) < 80_000:
                tasks.append(t)
            if len(tasks) >= 5:
                break
    if len(tasks) < 5:
        # Fall back to taking any 5 (long ones too)
        with (ROOT / "datasets/train.tasks.jsonl").open() as f:
            tasks = [json.loads(line) for i, line in enumerate(f) if i < 5]

    with (out_dir / "good.jsonl").open("w") as f_good, (out_dir / "bad.jsonl").open("w") as f_bad:
        for t in tasks:
            good = make_good(t)
            f_good.write(json.dumps({
                "task_id": t["task_id"],
                "response": good,
                "label": "good",
            }, ensure_ascii=False) + "\n")
            for name, fn in SHORTCUTS:
                bad = fn(t)
                f_bad.write(json.dumps({
                    "task_id": t["task_id"],
                    "response": bad,
                    "label": "bad",
                    "shortcut": name,
                }, ensure_ascii=False) + "\n")

    print(f"wrote {len(tasks)} good cases to {out_dir / 'good.jsonl'}")
    print(f"wrote {len(tasks) * len(SHORTCUTS)} bad cases to {out_dir / 'bad.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
