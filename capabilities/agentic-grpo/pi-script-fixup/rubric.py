"""Composite reward function for pi-script-fixup (v0).

§5.5 verifier-free: there's no oracle verifier other than re-running
the script. Composite:

    composite = outcome × format × (
        0.50 · read_error_before_edit
      + 0.20 · single_edit
      + 0.30
    )

outcome = 1.0 iff the script runs cleanly after the agent's edits.
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


def _calls(transcript):
    out = []
    for i, msg in enumerate(_iter_messages(transcript)):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls(msg):
            args = tc.get("input") or tc.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            out.append((i, tc.get("name", ""), args))
    return out


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


def _outcome(rollout) -> float:
    v = rollout.get("outcome_passed")
    return 1.0 if v is True else 0.0


def _format_compliance(rollout) -> float:
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).strip().lower()
    if not final:
        return 0.0
    score = 0.0
    if any(w in final for w in ("fixed", "running", "now runs", "works", "done", "patched")):
        score += 0.5
    if any(w in final for w in ("error", "issue", "wrong", "fix", "missing")):
        score += 0.5
    return min(1.0, score)


def _read_error_before_edit(rollout) -> float:
    """Did the agent observe an error tool-result before issuing the fix edit?"""
    transcript = rollout.get("transcript") or []
    calls = _calls(transcript)
    # Find first edit
    first_edit = None
    for i, n, a in calls:
        if n in ("write", "edit", "replace"):
            first_edit = i
            break
    if first_edit is None:
        return 0.0
    # Was there a tool result (error) before that edit?
    msgs = list(_iter_messages(transcript))
    for j, m in enumerate(msgs):
        if j >= first_edit:
            break
        if m.get("role") != "tool":
            continue
        for tr in m.get("content") or []:
            if isinstance(tr, dict) and tr.get("type") == "toolResult":
                content = (tr.get("content") or "").lower()
                if any(w in content for w in ("error", "syntaxerror", "indenterror", "importerror", "modulenotfound", "traceback", "exit=1", "exit_code=1")):
                    return 1.0
    return 0.0


def _single_edit(rollout) -> float:
    """Reward solving in one edit (not flailing with multiple write attempts)."""
    transcript = rollout.get("transcript") or []
    edits = sum(1 for _, n, _ in _calls(transcript) if n in ("write", "edit", "replace"))
    if edits == 1:
        return 1.0
    if edits == 2:
        return 0.5
    if edits == 0:
        return 0.0
    return 0.2


W_READ = 0.50
W_SINGLE = 0.20
BASE = 0.30


def score_one(rollout: dict) -> dict[str, Any]:
    outcome = _outcome(rollout)
    fmt = _format_compliance(rollout)
    read = _read_error_before_edit(rollout)
    single = _single_edit(rollout)
    process = W_READ * read + W_SINGLE * single + BASE
    composite = outcome * fmt * process
    return {
        "composite": float(max(0.0, min(1.0, composite))),
        "outcome": float(outcome),
        "format_compliance": float(fmt),
        "read_error_before_edit": float(read),
        "single_edit": float(single),
    }


CHEAT_PROBES = [
    "edit before observing the error (mitigation: read_error_before_edit=0)",
    "flail with many edits (mitigation: single_edit decays)",
    "claim fixed without running (mitigation: outcome verifies)",
]
