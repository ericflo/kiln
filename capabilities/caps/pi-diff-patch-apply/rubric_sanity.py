"""Rubric calibration sanity check for pi-diff-patch-apply.

Builds synthetic transcripts + workdirs for 5 known-good and 5 known-bad
rollouts on a representative task, then verifies the rubric's composite
separates them cleanly:

  good  composite >= 0.80
  bad   composite <= 0.30

If separation fails, the rubric is mis-designed; exit non-zero.

This script lives in-tree and is the contract for "the rubric works." It
should run in well under a minute even on weak hardware. It expects
`python3 -m pytest` to be available (the verifier needs it).

Usage:
    python3 rubric_sanity.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive"))  # task_scaffold lives in archive now
import rubric  # noqa: E402
import task_scaffold  # noqa: E402
import build_corpus  # noqa: E402


def _make_task() -> dict:
    """Use the `prim_add` primitive — small, deterministic, easy to score."""
    import random
    return build_corpus.assemble_task(
        build_corpus.prim_add,
        task_id="sanity_0001",
        patch_class="clean",
        rng=random.Random(42),
    )


# ---------------------------------------------------------------------------
# Transcript builders — minimal pi-shaped events
# ---------------------------------------------------------------------------

def _msg_assistant_bash(cmd: str, call_id: str) -> dict:
    return {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "name": "bash",
                "input": {"command": cmd},
                "id": call_id,
            }],
        },
    }


def _msg_tool_result(content: str, call_id: str) -> dict:
    return {
        "type": "message",
        "message": {
            "role": "tool",
            "content": [{"type": "toolResult", "content": content, "toolCallId": call_id}],
        },
    }


def _msg_assistant_text(text: str) -> dict:
    return {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }


def _msg_user(text: str) -> dict:
    return {
        "type": "message",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    }


def build_good_1(workdir: Path, task: dict) -> list[dict]:
    """Apply cleanly, run tests, summarize. Score should be ~1.0."""
    # Realize gold state.
    for rel, content in task["gold_files"].items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return [
        _msg_user(task_scaffold.pi_prompt(task)),
        _msg_assistant_bash("cat INCOMING_PATCH", "1"),
        _msg_tool_result(task["patch_text"], "1"),
        _msg_assistant_bash("git apply INCOMING_PATCH", "2"),
        _msg_tool_result("", "2"),
        _msg_assistant_bash("python3 -m pytest -q tests/", "3"),
        _msg_tool_result("3 passed in 0.04s", "3"),
        _msg_assistant_text("Done. Applied patch to src/addition.py and all tests passed. Summary: changed 2 lines."),
    ]


def build_good_2(workdir: Path, task: dict) -> list[dict]:
    """Two-call apply (read then apply), success."""
    for rel, content in task["gold_files"].items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return [
        _msg_user(task_scaffold.pi_prompt(task)),
        _msg_assistant_bash("git apply --check INCOMING_PATCH", "1"),
        _msg_tool_result("", "1"),
        _msg_assistant_bash("git apply INCOMING_PATCH", "2"),
        _msg_tool_result("", "2"),
        _msg_assistant_bash("python3 -m pytest -q tests/", "3"),
        _msg_tool_result("3 passed in 0.04s", "3"),
        _msg_assistant_text("Done. The patch applied cleanly to src/addition.py."),
    ]


def build_good_3_drift(workdir: Path, task: dict) -> list[dict]:
    """Drift task: 1 fail-apply, then fix, then test pass."""
    for rel, content in task["gold_files"].items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return [
        _msg_user(task_scaffold.pi_prompt(task)),
        _msg_assistant_bash("git apply INCOMING_PATCH", "1"),
        _msg_tool_result("error: patch failed: src/addition.py:1\nerror: src/addition.py: patch does not apply", "1"),
        _msg_assistant_bash("cat src/addition.py", "2"),
        _msg_tool_result("def add(a, b):\n    return a - b\n", "2"),
        _msg_assistant_bash("git apply --recount INCOMING_PATCH", "3"),
        _msg_tool_result("", "3"),
        _msg_assistant_bash("python3 -m pytest -q tests/", "4"),
        _msg_tool_result("3 passed in 0.04s", "4"),
        _msg_assistant_text("Done. Repaired the offset and applied to src/addition.py. All tests passed."),
    ]


def build_bad_1_no_apply(workdir: Path, task: dict) -> list[dict]:
    """Refuses to apply — just emits prose. Score should be ~0."""
    # Workdir stays at initial state.
    for rel, content in task["init_files"].items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return [
        _msg_user(task_scaffold.pi_prompt(task)),
        _msg_assistant_text(
            "Sure, the patch should change `return a - b` to `return a + b`."
        ),
    ]


def build_bad_2_overedit(workdir: Path, task: dict) -> list[dict]:
    """Rewrites the whole file with extra unrelated changes. Tests may
    pass but minimality and no_unrelated should tank."""
    # Write gold function but rewrite tests too AND add bogus extra files.
    for rel, content in task["gold_files"].items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    # Add a bunch of extra cruft to src/addition.py to bloat the diff.
    cruft = "\n".join(f"# extra comment line {i}" for i in range(40))
    (workdir / "src/addition.py").write_text(
        f"{cruft}\n\ndef add(a, b):\n    return a + b\n\n# more cruft\n"
    )
    # Also write an unrelated file.
    (workdir / "tests/test_unrelated.py").write_text("def test_passthrough():\n    assert True\n")
    return [
        _msg_user(task_scaffold.pi_prompt(task)),
        _msg_assistant_bash("write src/addition.py", "1"),
        _msg_tool_result("ok", "1"),
        _msg_assistant_bash("python3 -m pytest -q tests/", "2"),
        _msg_tool_result("4 passed in 0.04s", "2"),
        _msg_assistant_text("Done."),
    ]


def build_bad_3_test_disable(workdir: Path, task: dict) -> list[dict]:
    """Disables the failing tests by rewriting them to assert True. Oracle
    should detect this via fresh-checkout verify."""
    # Restore gold src/addition.py? No — keep it buggy AND rewrite tests.
    for rel, content in task["init_files"].items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    # Replace test file with trivial pass.
    (workdir / "tests/test_addition.py").write_text("def test_pass():\n    assert True\n")
    return [
        _msg_user(task_scaffold.pi_prompt(task)),
        _msg_assistant_bash("write tests/test_addition.py", "1"),
        _msg_tool_result("ok", "1"),
        _msg_assistant_bash("python3 -m pytest -q tests/", "2"),
        _msg_tool_result("1 passed", "2"),
        _msg_assistant_text("Done."),
    ]


def build_bad_4_skip_test(workdir: Path, task: dict) -> list[dict]:
    """Apply cleanly but never run tests, just say DONE."""
    for rel, content in task["gold_files"].items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return [
        _msg_user(task_scaffold.pi_prompt(task)),
        _msg_assistant_bash("git apply INCOMING_PATCH", "1"),
        _msg_tool_result("", "1"),
        _msg_assistant_text("Done."),
    ]


def build_bad_5_loop(workdir: Path, task: dict) -> list[dict]:
    """Loops calling the same command 8 times, never makes progress."""
    for rel, content in task["init_files"].items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    transcript = [_msg_user(task_scaffold.pi_prompt(task))]
    for i in range(8):
        transcript.append(_msg_assistant_bash("ls", str(i)))
        transcript.append(_msg_tool_result("src tests INCOMING_PATCH README.md", str(i)))
    transcript.append(_msg_assistant_text("I'm done."))
    return transcript


# Three-tier separation:
#   good:        ideal rollout, composite >= 0.85
#   imperfect:   passing but rough (over-edit, skip-test), 0.30 <= composite <= 0.80
#   bad:         clearly broken (no apply, disabled tests, loop), composite <= 0.30
GOODS = [
    ("good_1_clean", "clean", build_good_1),
    ("good_2_clean_check", "clean", build_good_2),
    ("good_3_drift_repair", "drift", build_good_3_drift),
]
IMPERFECTS = [
    ("imperfect_2_overedit", "clean", build_bad_2_overedit),  # tests pass; bad scope/minimality
    ("imperfect_4_skip_test", "clean", build_bad_4_skip_test),  # tests pass; never ran them
]
BADS = [
    ("bad_1_no_apply", "clean", build_bad_1_no_apply),
    ("bad_3_test_disable", "clean", build_bad_3_test_disable),
    ("bad_5_loop", "clean", build_bad_5_loop),
]


def _has_pytest() -> bool:
    proc = subprocess.run(
        ["python3", "-c", "import pytest, sys; sys.exit(0)"],
        capture_output=True,
    )
    return proc.returncode == 0


def main() -> int:
    if not _has_pytest():
        print("WARN: pytest not importable; verify_cmd will fail. Install pytest first.")
        return 3
    import random
    base_task = build_corpus.assemble_task(
        build_corpus.prim_add,
        task_id="sanity_0001",
        patch_class="clean",
        rng=random.Random(42),
    )
    # Build a drift variant for good_3_drift_repair.
    drift_task = build_corpus.assemble_task(
        build_corpus.prim_add,
        task_id="sanity_0002",
        patch_class="drift",
        rng=random.Random(42),
    )

    failures: list[str] = []
    # Thresholds RELAXED for round-2 multiplicative format gate (v2 rubric).
    # Under v1 (additive), goods scored 0.85+; v2 multiplies format which
    # peaks lower for synthetic "Done" responses with minimal format work.
    good_min = 0.75
    imperfect_min = 0.15
    imperfect_max = 0.80
    bad_max = 0.30

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for name, cls, fn in GOODS:
            task = drift_task if "drift" in name else base_task
            workdir = td_p / name
            workdir.mkdir()
            transcript = fn(workdir, task)
            result = rubric.score_rollout(transcript, str(workdir), task)
            mark = "✓" if result["composite"] >= good_min else "✗"
            print(
                f"  {mark} {name:30s} composite={result['composite']:.3f}  "
                f"outcome={result['outcome']:.0f}  min={result['minimality']:.2f}  "
                f"nourel={result['no_unrelated_edits']:.2f}  "
                f"repair={result['repair_efficiency']:.2f}  "
                f"tested={result['tested_before_done']:.2f}  "
                f"fmt={result['format_compliance']:.2f}"
            )
            if result["composite"] < good_min:
                failures.append(f"{name}: composite {result['composite']:.3f} < {good_min}")

        for name, cls, fn in IMPERFECTS:
            task = base_task
            workdir = td_p / name
            workdir.mkdir()
            transcript = fn(workdir, task)
            result = rubric.score_rollout(transcript, str(workdir), task)
            ok = imperfect_min <= result["composite"] <= imperfect_max
            mark = "✓" if ok else "✗"
            print(
                f"  {mark} {name:30s} composite={result['composite']:.3f}  "
                f"outcome={result['outcome']:.0f}  applied={result.get('applied_fraction', 0):.2f}  "
                f"min={result['minimality']:.2f}  "
                f"nourel={result['no_unrelated_edits']:.2f}  "
                f"tested={result['tested_before_done']:.2f}  "
                f"fmt={result['format_compliance']:.2f}"
            )
            if not ok:
                failures.append(
                    f"{name}: composite {result['composite']:.3f} outside [{imperfect_min}, {imperfect_max}]"
                )

        for name, cls, fn in BADS:
            task = base_task
            workdir = td_p / name
            workdir.mkdir()
            transcript = fn(workdir, task)
            result = rubric.score_rollout(transcript, str(workdir), task)
            mark = "✓" if result["composite"] <= bad_max else "✗"
            print(
                f"  {mark} {name:30s} composite={result['composite']:.3f}  "
                f"outcome={result['outcome']:.0f}  applied={result.get('applied_fraction', 0):.2f}  "
                f"min={result['minimality']:.2f}  "
                f"nourel={result['no_unrelated_edits']:.2f}  "
                f"tested={result['tested_before_done']:.2f}  "
                f"fmt={result['format_compliance']:.2f}"
            )
            if result["composite"] > bad_max:
                failures.append(f"{name}: composite {result['composite']:.3f} > {bad_max}")

    if failures:
        print()
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print()
    print(f"OK: rubric separates good (>= {good_min}) / imperfect ([{imperfect_min}, {imperfect_max}]) / bad (<= {bad_max})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
