"""Tests for pi_trajectory.parse_pi_session.

Run from the repo root:
    cd capabilities/lib && python3 test_pi_trajectory.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pi_trajectory as pt  # noqa: E402


def _write_session(events: list[dict]) -> Path:
    fd, path_str = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    path = Path(path_str)
    with path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return path


def test_empty_session_returns_empty_list():
    p = _write_session([])
    try:
        assert pt.parse_pi_session(p) == []
    finally:
        p.unlink()


def test_skips_non_message_events():
    p = _write_session(
        [
            {"type": "tool_summary"},
            {"type": "session_start"},
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}},
        ]
    )
    try:
        segs = pt.parse_pi_session(p)
        assert len(segs) == 1
        assert segs[0]["role"] == "assistant"
        assert segs[0]["content"] == "hi"
        assert segs[0]["kind"] == "action"
    finally:
        p.unlink()


def test_assistant_text_block():
    p = _write_session(
        [
            {"type": "message", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "the answer is 42"}
            ]}}
        ]
    )
    try:
        segs = pt.parse_pi_session(p)
        assert len(segs) == 1
        assert segs[0]["role"] == "assistant"
        assert segs[0]["kind"] == "action"
        assert segs[0]["content"] == "the answer is 42"
    finally:
        p.unlink()


def test_assistant_thinking_wrapped_in_think_tags():
    p = _write_session(
        [
            {"type": "message", "message": {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "let me reason"},
                {"type": "text", "text": "the answer is 42"},
            ]}}
        ]
    )
    try:
        segs = pt.parse_pi_session(p)
        assert len(segs) == 1
        assert segs[0]["content"] == "<think>let me reason</think>the answer is 42"
    finally:
        p.unlink()


def test_assistant_tool_call_accepts_input_and_arguments_fields():
    for field_name in ("input", "arguments"):
        p = _write_session(
            [
                {"type": "message", "message": {"role": "assistant", "content": [
                    {"type": "toolCall", "name": "read", field_name: {"path": "solution.py"}}
                ]}}
            ]
        )
        try:
            segs = pt.parse_pi_session(p)
            assert len(segs) == 1
            # Verify the rendered XML matches what rollout.py was producing
            # (so chat-template round-trips bit-identically).
            content = segs[0]["content"]
            assert content.startswith("<tool_call>")
            assert content.endswith("</tool_call>")
            assert '"name": "read"' in content
            assert '"arguments"' in content
            assert '"path"' in content
            assert '"solution.py"' in content
        finally:
            p.unlink()


def test_tool_result_becomes_observation():
    p = _write_session(
        [
            {"type": "message", "message": {"role": "assistant", "content": [
                {"type": "toolCall", "name": "read", "input": {"path": "x.py"}, "id": "call_1"}
            ]}},
            {"type": "message", "message": {"role": "tool", "content": [
                {"type": "toolResult", "content": "print(42)\n", "toolCallId": "call_1"}
            ]}}
        ]
    )
    try:
        segs = pt.parse_pi_session(p)
        assert len(segs) == 2
        assert segs[0]["kind"] == "action"
        assert segs[1]["kind"] == "observation"
        assert segs[1]["content"] == "print(42)\n"
        assert segs[1]["tool_call_id"] == "call_1"
    finally:
        p.unlink()


def test_observation_warning_prefix_detected():
    """Harness emits 'WARNINGS:\\n- ...\\n<command_output>...' when a tool
    call fails parsing. The warning prefix length is recorded so the
    masker can advance past it (paper §3.2)."""
    warning_text = (
        "WARNINGS:\n- malformed tool call format\n<command_output>"
        "ls: cannot access\n</command_output>"
    )
    p = _write_session(
        [
            {"type": "message", "message": {"role": "tool", "content": [
                {"type": "toolResult", "content": warning_text}
            ]}}
        ]
    )
    try:
        segs = pt.parse_pi_session(p)
        assert len(segs) == 1
        assert segs[0]["kind"] == "observation"
        # warning_prefix_len should be exactly the byte length of
        # "WARNINGS:\n- malformed tool call format\n"
        expected = len("WARNINGS:\n- malformed tool call format\n")
        assert segs[0]["warning_prefix_len"] == expected
    finally:
        p.unlink()


def test_observation_no_warning_no_prefix_field():
    p = _write_session(
        [
            {"type": "message", "message": {"role": "tool", "content": [
                {"type": "toolResult", "content": "clean stdout"}
            ]}}
        ]
    )
    try:
        segs = pt.parse_pi_session(p)
        assert "warning_prefix_len" not in segs[0]
    finally:
        p.unlink()


def test_system_user_filtered_by_default():
    p = _write_session(
        [
            {"type": "message", "message": {"role": "system", "content": [
                {"type": "text", "text": "you are helpful"}
            ]}},
            {"type": "message", "message": {"role": "user", "content": [
                {"type": "text", "text": "do thing"}
            ]}},
            {"type": "message", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "ok"}
            ]}},
        ]
    )
    try:
        segs = pt.parse_pi_session(p)
        # Default: include_context=False
        assert len(segs) == 1
        assert segs[0]["role"] == "assistant"
        # With include_context=True we get all three
        all_segs = pt.parse_pi_session(p, include_context=True)
        assert len(all_segs) == 3
        assert all_segs[0]["kind"] == "context"
        assert all_segs[1]["kind"] == "context"
        assert all_segs[2]["kind"] == "action"
    finally:
        p.unlink()


def test_full_session_round_trip():
    """End-to-end: a realistic pi session with user -> assistant tool call ->
    tool result -> assistant final."""
    p = _write_session(
        [
            {"type": "message", "message": {"role": "system", "content": [
                {"type": "text", "text": "Python assistant"}
            ]}},
            {"type": "message", "message": {"role": "user", "content": [
                {"type": "text", "text": "Print 42"}
            ]}},
            {"type": "message", "message": {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I should use bash"},
                {"type": "toolCall", "name": "bash",
                 "input": {"cmd": "python3 -c 'print(42)'"}, "id": "c1"}
            ]}},
            {"type": "message", "message": {"role": "tool", "content": [
                {"type": "toolResult", "content": "42\n", "toolCallId": "c1"}
            ]}},
            {"type": "message", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Done — the program printed 42."}
            ]}},
        ]
    )
    try:
        segs = pt.parse_pi_session(p)
        # System / user filtered out by default
        assert len(segs) == 3
        assert [s["kind"] for s in segs] == ["action", "observation", "action"]
        assert "<think>" in segs[0]["content"]
        assert "<tool_call>" in segs[0]["content"]
        assert segs[1]["content"] == "42\n"
        assert segs[1]["tool_call_id"] == "c1"
        assert segs[2]["content"] == "Done — the program printed 42."

        # Round-trip into ScoredRollout shape via build_scored_rollout.
        rollout = pt.build_scored_rollout(p, reward=1.0)
        assert rollout["reward"] == 1.0
        # text is the <TURN_BREAK>-flattened Action segments
        assert "<TURN_BREAK>" in rollout["text"]
        assert rollout["trajectory"] == segs
    finally:
        p.unlink()


def test_malformed_jsonl_lines_skipped():
    fd, path_str = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    p = Path(path_str)
    p.write_text(
        json.dumps({"type": "message", "message": {"role": "assistant",
                    "content": [{"type": "text", "text": "good"}]}}) + "\n"
        "not valid json\n"
        "{\"partial\": \n"
        + json.dumps({"type": "message", "message": {"role": "assistant",
                      "content": [{"type": "text", "text": "also good"}]}}) + "\n"
    )
    try:
        segs = pt.parse_pi_session(p)
        # Bad lines silently skipped; both good lines kept.
        assert len(segs) == 2
        assert segs[0]["content"] == "good"
        assert segs[1]["content"] == "also good"
    finally:
        p.unlink()


def test_flatten_action_text():
    traj = [
        {"role": "assistant", "content": "first", "kind": "action"},
        {"role": "tool", "content": "tool out", "kind": "observation"},
        {"role": "assistant", "content": "second", "kind": "action"},
    ]
    assert pt.flatten_action_text(traj) == "first<TURN_BREAK>second"


def test_flatten_action_text_no_actions():
    traj = [{"role": "tool", "content": "x", "kind": "observation"}]
    assert pt.flatten_action_text(traj) == ""


def test_missing_path_returns_empty():
    bogus = Path("/tmp/this-file-definitely-does-not-exist-echo-test.jsonl")
    assert pt.parse_pi_session(bogus) == []


def test_warning_prefix_fallback_to_double_newline():
    # If <command_output> isn't present, fall back to double-newline split.
    warning_text = "WARNINGS:\n- something bad\n\nactual output"
    p = _write_session(
        [
            {"type": "message", "message": {"role": "tool", "content": [
                {"type": "toolResult", "content": warning_text}
            ]}}
        ]
    )
    try:
        segs = pt.parse_pi_session(p)
        # The double-newline split is at byte position 30 (WARNINGS:\n=
        # 10 + - something bad\n = 17 + \n = 1 -> 28; then +2 to skip the
        # \n\n -> 30. Test that warning_prefix_len is approximately right.
        wpl = segs[0].get("warning_prefix_len")
        assert wpl is not None
        # The fallback should land BEFORE "actual output".
        actual_start = warning_text.find("actual output")
        assert wpl <= actual_start
    finally:
        p.unlink()


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
