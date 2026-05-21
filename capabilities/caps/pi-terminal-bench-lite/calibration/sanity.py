"""Rubric sanity gate for pi-terminal-bench-lite.

Runs the rubric against 3 hand-crafted "good" sessions (should score
near 1.0) and 3 "bad" sessions (should score near 0.0). Exits 0 if the
gates pass; non-zero otherwise.

The good/bad fixture sessions are constructed inline so the calibration
script is self-contained — easier to maintain than scattered JSONL
fixtures.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import rubric  # noqa: E402

# Use a synthetic verifier that succeeds when a specific marker file
# exists in the workdir.
GOOD_VERIFIER = "test -f output.csv && grep -q 'alice,30' output.csv"
BAD_VERIFIER = "test -f output.csv && grep -q 'alice,30' output.csv"


def _write_session(events: list[dict]) -> str:
    """Write a synthetic pi session JSONL and return the path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return path


def _make_assistant_message(parts: list[dict]) -> dict:
    return {
        "type": "message",
        "message": {"role": "assistant", "content": parts},
    }


def _make_tool_message(content: str, tool_call_id: str = "c1") -> dict:
    return {
        "type": "message",
        "message": {
            "role": "tool",
            "content": [
                {"type": "toolResult", "content": content, "toolCallId": tool_call_id}
            ],
        },
    }


def good_session_one_call(workdir: Path) -> tuple[str, dict]:
    """Ideal: one well-formed tool call that writes the expected file."""
    (workdir / "output.csv").write_text("name,age\nalice,30\n")
    events = [
        _make_assistant_message([
            {"type": "thinking", "thinking": "I'll filter the rows."},
            {
                "type": "toolCall",
                "name": "write",
                "input": {"path": "output.csv", "content": "name,age\nalice,30\n"},
                "id": "c1",
            },
        ]),
        _make_tool_message("ok", tool_call_id="c1"),
        _make_assistant_message([{"type": "text", "text": "Done."}]),
    ]
    return _write_session(events), {
        "task_id": "good-1",
        "verifier": GOOD_VERIFIER,
        "verifier_timeout_s": 5,
    }


def bad_session_no_calls(workdir: Path) -> tuple[str, dict]:
    """Bad: assistant chats but never produces the file."""
    events = [
        _make_assistant_message([{"type": "text", "text": "Hmm, let me think."}]),
        _make_assistant_message([{"type": "text", "text": "I don't know."}]),
    ]
    return _write_session(events), {
        "task_id": "bad-1",
        "verifier": BAD_VERIFIER,
        "verifier_timeout_s": 5,
    }


def main() -> int:
    print("Calibrating pi-terminal-bench-lite rubric...")

    failures = 0
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # Three "good" sessions — verifier passes, well-formed.
        for i in range(3):
            workdir = td_path / f"good-{i}"
            workdir.mkdir()
            session, task = good_session_one_call(workdir)
            r = rubric.score_rollout(session, str(workdir), task)
            label = f"GOOD-{i}"
            if r["composite"] >= 0.85:
                print(f"  PASS {label}: composite={r['composite']:.3f}")
            else:
                failures += 1
                print(f"  FAIL {label}: composite={r['composite']:.3f} (expected ≥0.85)")
                print(f"    sub_scores: {r['sub_scores']}")

        # Three "bad" sessions — verifier fails.
        for i in range(3):
            workdir = td_path / f"bad-{i}"
            workdir.mkdir()
            session, task = bad_session_no_calls(workdir)
            r = rubric.score_rollout(session, str(workdir), task)
            label = f"BAD-{i}"
            if r["composite"] <= 0.05:
                print(f"  PASS {label}: composite={r['composite']:.3f}")
            else:
                failures += 1
                print(f"  FAIL {label}: composite={r['composite']:.3f} (expected ≤0.05)")

    if failures:
        print(f"\nFAILED: {failures} calibration gate(s) failed")
        return 1
    print("\nOK: rubric sanity gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
