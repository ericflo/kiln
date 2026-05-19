"""Rubric for pi-terminal-bench-lite. Mirrors pi-doctest's structure but
with weights tuned for TBLite-shaped tasks where outcome is the
load-bearing signal.

  composite = outcome * (0.70 +
                         0.15 * tool_call_efficiency +
                         0.10 * format_compliance +
                         0.05 * no_loop)

When outcome=0 (the verifier failed) composite is 0 regardless of the
sub-scores. When outcome=1 the sub-scores add ≤0.30 above the floor of
0.70 — so a model that solves the task with one well-formed tool call
gets composite ≈ 1.0; a model that solves it after spamming gets ~0.85;
a model that fails outright gets 0.0.

Tool-call-related sub-scores come from parsing the pi session JSONL via
the shared pi_trajectory.py lib.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import pi_trajectory  # noqa: E402


EXPECTED_TOOL_CALLS = 12  # TBLite tasks typically need 4-10 tool calls; 12 is the
                          # "fully efficient" line — beyond this, tool_call_efficiency
                          # starts losing points.


def score_rollout(transcript_path: str, workdir: str, task: dict) -> dict:
    """Score one rollout. Returns dict with composite + sub_scores."""
    segments = pi_trajectory.parse_pi_session(Path(transcript_path)) if transcript_path else []
    outcome = _outcome(workdir, task)
    sub_scores = {
        "outcome": float(outcome),
        "tool_call_efficiency": _tool_call_efficiency(segments),
        "format_compliance": _format_compliance(segments),
        "no_loop": _no_loop(segments),
    }
    composite = float(outcome) * (
        0.70
        + 0.15 * sub_scores["tool_call_efficiency"]
        + 0.10 * sub_scores["format_compliance"]
        + 0.05 * sub_scores["no_loop"]
    )
    return {
        "composite": composite,
        "outcome": float(outcome),
        "sub_scores": sub_scores,
        "n_segments": len(segments),
    }


def _outcome(workdir: str, task: dict) -> int:
    """Run the verifier and return 1 if it exits 0, else 0."""
    import subprocess

    verifier = task.get("verifier")
    if not verifier:
        return 0
    cwd = Path(workdir)
    if not cwd.exists():
        return 0
    try:
        result = subprocess.run(
            verifier,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            timeout=task.get("verifier_timeout_s", 60),
        )
        return 1 if result.returncode == 0 else 0
    except (subprocess.TimeoutExpired, OSError):
        return 0


def _tool_call_efficiency(segments: list[dict]) -> float:
    n_tool_calls = sum(
        1 for seg in segments
        if seg["kind"] == "action" and "<tool_call>" in seg["content"]
    )
    if n_tool_calls == 0:
        return 1.0  # nothing to be inefficient about
    overshoot = max(0, n_tool_calls - EXPECTED_TOOL_CALLS) / EXPECTED_TOOL_CALLS
    return max(0.0, 1.0 - overshoot)


def _format_compliance(segments: list[dict]) -> float:
    """Fraction of assistant turns where the content parses as either a
    plain final text turn OR a well-formed tool call."""
    action_segments = [s for s in segments if s["kind"] == "action"]
    if not action_segments:
        return 1.0
    well_formed = 0
    for seg in action_segments:
        content = seg["content"]
        if "<tool_call>" in content and "</tool_call>" in content:
            # Try to parse: <tool_call>{"name":...,"arguments":...}</tool_call>
            start = content.find("<tool_call>") + len("<tool_call>")
            end = content.find("</tool_call>")
            if end > start:
                inner = content[start:end].strip()
                try:
                    parsed = json.loads(inner)
                    if "name" in parsed and "arguments" in parsed:
                        well_formed += 1
                        continue
                except json.JSONDecodeError:
                    pass
        elif content.strip() and "<tool_call>" not in content:
            # Plain text final turn
            well_formed += 1
    return well_formed / len(action_segments)


def _no_loop(segments: list[dict]) -> float:
    """Fraction of tool calls that are unique (deduplicated by name+args)."""
    tool_call_hashes: set[str] = set()
    total = 0
    for seg in segments:
        if seg["kind"] != "action":
            continue
        content = seg["content"]
        if "<tool_call>" not in content:
            continue
        start = content.find("<tool_call>") + len("<tool_call>")
        end = content.find("</tool_call>")
        if end > start:
            inner = content[start:end].strip()
            try:
                parsed = json.loads(inner)
                key = f"{parsed.get('name','')}\\x00{json.dumps(parsed.get('arguments',{}), sort_keys=True)}"
                tool_call_hashes.add(hashlib.sha256(key.encode()).hexdigest())
            except json.JSONDecodeError:
                pass
            total += 1
    if total == 0:
        return 1.0
    return len(tool_call_hashes) / total
