"""Composite reward function for pi-shell-hygiene (v0).

Multiplicative-gate composite:

    composite = outcome × format × (
        0.40 · uses_good_pattern
      + 0.30 · avoids_bad_pattern
      + 0.10 · has_timeout
      + 0.20                         # base
    )

Per kiln-skill anti-pattern doc (clouderic). Rubric matches commands
against the task's `good_patterns` and `bad_patterns` lists.
"""
from __future__ import annotations
import re
from typing import Any

RUBRIC_VERSION = "v0"


def _iter_messages(transcript):
    for ev in transcript or []:
        if isinstance(ev, dict) and ev.get("type") == "message":
            msg = ev.get("message")
            if isinstance(msg, dict):
                yield msg


def _tool_calls(msg):
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "toolCall"]


def _final_text(transcript):
    final = ""
    for msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or []
        text = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        if text.strip():
            final = text.strip()
    return final


def _all_commands(transcript) -> str:
    cmds = []
    for msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls(msg):
            args = tc.get("input") or tc.get("arguments") or {}
            if isinstance(args, dict):
                c = args.get("command") or args.get("cmd") or ""
                if isinstance(c, str):
                    cmds.append(c)
    return "\n".join(cmds)


def _outcome(rollout) -> float:
    """Outcome: at least one good pattern AND no bad pattern in any command."""
    task = rollout.get("task") or {}
    good = task.get("good_patterns") or []
    bad = task.get("bad_patterns") or []
    cmds = _all_commands(rollout.get("transcript") or [])
    if not cmds:
        return 0.0
    has_good = any(g in cmds for g in good)
    has_bad = any(b in cmds for b in bad)
    return 1.0 if (has_good and not has_bad) else 0.0


def _format_compliance(rollout) -> float:
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).strip().lower()
    if not final:
        return 0.0
    score = 0.0
    if any(w in final for w in ("background", "wait", "timeout", "launched", "started", "running", "complete")):
        score += 0.5
    if any(w in final for w in ("done", "complete", "started", "launched", "ready")):
        score += 0.5
    return min(1.0, score)


def _uses_good_pattern(rollout) -> float:
    task = rollout.get("task") or {}
    good = task.get("good_patterns") or []
    cmds = _all_commands(rollout.get("transcript") or [])
    if not cmds:
        return 0.0
    matches = sum(1 for g in good if g in cmds)
    return float(min(1.0, matches / max(1, min(2, len(good)))))


def _avoids_bad_pattern(rollout) -> float:
    task = rollout.get("task") or {}
    bad = task.get("bad_patterns") or []
    cmds = _all_commands(rollout.get("transcript") or [])
    matches = sum(1 for b in bad if b in cmds)
    return 1.0 if matches == 0 else 0.0


def _has_timing_control(rollout) -> float:
    """Has the rollout used some explicit timing mechanism — timeout flag,
    a sleep with a substantial value (>=60s), or wait-file.

    A background launch with no timing whatsoever is a hang risk; this
    sub-score penalizes that.
    """
    cmds = _all_commands(rollout.get("transcript") or [])
    if re.search(r"\b(timeout\s+\d|--timeout[ =]\d)", cmds):
        return 1.0
    if "wait-file" in cmds:
        return 1.0
    # Sleep >=60 is intentional rate-limit timing; <60 is polling
    m = re.findall(r"\bsleep\s+(\d+)", cmds)
    if m and any(int(x) >= 60 for x in m):
        return 1.0
    return 0.0


W_GOOD = 0.35
W_AVOID_BAD = 0.30
W_TIMING = 0.20
BASE = 0.15


def score_one(rollout: dict) -> dict[str, Any]:
    outcome = _outcome(rollout)
    fmt = _format_compliance(rollout)
    good = _uses_good_pattern(rollout)
    avoid_bad = _avoids_bad_pattern(rollout)
    timing = _has_timing_control(rollout)
    process = W_GOOD * good + W_AVOID_BAD * avoid_bad + W_TIMING * timing + BASE
    composite = outcome * fmt * process
    return {
        "composite": float(max(0.0, min(1.0, composite))),
        "outcome": float(outcome),
        "format_compliance": float(fmt),
        "uses_good_pattern": float(good),
        "avoids_bad_pattern": float(avoid_bad),
        "has_timing_control": float(timing),
    }


CHEAT_PROBES = [
    "use both good AND bad patterns (mitigation: outcome requires no bad)",
    "no timeout (mitigation: has_timeout sub-score; not gate but reduces composite)",
    "constant 'good pattern' regardless of context (mitigation: task-specific good_patterns)",
]
