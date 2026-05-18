"""Reward function for pi-doctest.

Signature:
  score_rollout(transcript: list[dict], workdir: str, task: dict) -> dict

`transcript` is the pi session JSONL parsed line-by-line (each event a
dict). `workdir` is the path where pi ran. `task` is the task spec.

Returns: {
  'outcome': float in [0, 1],
  'composite': float in [0, 1],
  '_doctest_summary': dict,  # diagnostic — not used in training
}

For v0 the composite is just `outcome`. Iter 2+ will add
tested_before_done, tool_call_efficiency, format_compliance.

Verification: re-runs `python3 -m doctest -v <workdir>/solution.py`
inside a 5s timeout subprocess. The pi session may have already run
it; we re-run for hermeticity (the verifier owns scoring, the model
owns trying).
"""

import json
import re
import subprocess
import sys
from pathlib import Path


def score_rollout(transcript: list, workdir: str, task: dict) -> dict:
    solution = Path(workdir) / "solution.py"
    if not solution.exists():
        return {"outcome": 0.0, "composite": 0.0,
                "_doctest_summary": {"reason": "no solution.py"}}

    # Run doctests in a subprocess. Use python3 -m doctest -v, parse output.
    try:
        proc = subprocess.run(
            ["python3", "-I", "-m", "doctest", "-v", str(solution)],
            capture_output=True, text=True, timeout=10,
            cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return {"outcome": 0.0, "composite": 0.0,
                "_doctest_summary": {"reason": "doctest timed out"}}
    except Exception as e:
        return {"outcome": 0.0, "composite": 0.0,
                "_doctest_summary": {"reason": f"doctest run error: {e}"}}

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # doctest -v output ends with lines like:
    #   N items had no tests:
    #   M items passed all tests:
    #      K tests in <name>
    #   X items had failures:
    #      Y of K tests in <name>
    # Total: Z tests in W items.
    # We compute: passed / total, where total is the number of doctests in
    # the input docstring. If "***Test Failed***" appears or any "FAIL"
    # line is present in stdout, those count against passed.

    # Parse "Trying ..." and "Expecting ..." / "ok" / "FAIL" pairs.
    # Each "Trying:" is one doctest; followed by either "ok" or "***********..."
    tried = stdout.count("Trying:")
    if tried == 0:
        # No doctests in the file. Either the function had no doctests
        # (task malformed) or solution.py is empty/broken.
        return {"outcome": 0.0, "composite": 0.0,
                "_doctest_summary": {"reason": "no doctests run",
                                     "exit_code": proc.returncode,
                                     "stderr_head": stderr[:300]}}
    # Count failures via the "Failed example:" header.
    failed = stdout.count("Failed example:")
    passed = max(0, tried - failed)
    outcome = passed / tried

    return {
        "outcome": outcome,
        "composite": outcome,
        "_doctest_summary": {
            "tried": tried,
            "passed": passed,
            "failed": failed,
            "exit_code": proc.returncode,
        },
    }


# Iter 2+ — sub-scores that quantify agentic behavior. Kept here as a
# reference so the rubric extension is one diff away.

def _count_assistant_turns_with_tool_call(transcript: list, tool_name: str) -> int:
    n = 0
    for ev in transcript:
        msgs = ev.get("messages") or []
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            for tc in (ev.get("tool_calls") or []):
                if tc.get("name") == tool_name:
                    n += 1
    return n


def _tested_before_done(transcript: list) -> float:
    """1.0 iff a successful bash call to doctest appears in the transcript
    BEFORE the final assistant turn."""
    last_bash_at_idx = None
    final_assistant_idx = None
    for i, ev in enumerate(transcript):
        tc = ev.get("tool_calls") or []
        for c in tc:
            if c.get("name") in ("bash", "shell"):
                cmd = json.dumps(c.get("arguments") or c.get("input") or {})
                if "doctest" in cmd:
                    last_bash_at_idx = i
        msgs = ev.get("messages") or []
        for m in msgs:
            if m.get("role") == "assistant":
                final_assistant_idx = i
    if last_bash_at_idx is None or final_assistant_idx is None:
        return 0.0
    return 1.0 if last_bash_at_idx < final_assistant_idx else 0.5


def _tool_call_efficiency(transcript: list, expected: int = 4) -> float:
    """1.0 if num_tool_calls <= expected, decaying to 0.0 at 3× expected."""
    n = sum(len(ev.get("tool_calls") or []) for ev in transcript)
    if n <= expected:
        return 1.0
    if n >= 3 * expected:
        return 0.0
    return max(0.0, 1.0 - (n - expected) / (2 * expected))


def _format_compliance(transcript: list) -> float:
    """Fraction of assistant turns whose tool_calls list (if any) all
    parse as well-formed (have `name` and JSON-serializable arguments).
    """
    total = 0
    ok = 0
    for ev in transcript:
        tcs = ev.get("tool_calls") or []
        for tc in tcs:
            total += 1
            if not isinstance(tc, dict):
                continue
            if not isinstance(tc.get("name"), str):
                continue
            args = tc.get("arguments") if "arguments" in tc else tc.get("input")
            if args is None:
                ok += 1
                continue
            try:
                json.dumps(args)
                ok += 1
            except Exception:
                pass
    if total == 0:
        return 0.0
    return ok / total


# CLI for manual scoring during development.
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: rubric.py <transcript.jsonl> <workdir> <task.json>",
              file=sys.stderr)
        sys.exit(2)
    transcript = []
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                transcript.append(json.loads(line))
            except Exception:
                pass
    task = json.loads(Path(sys.argv[3]).read_text())
    out = score_rollout(transcript, sys.argv[2], task)
    print(json.dumps(out, indent=2))
