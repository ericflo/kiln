"""Composite reward function for pi-source-mod-workflow (v0, eval-only).

Score by sub-step coverage: of the gold_sub_steps the task lists,
what fraction did the agent's session demonstrate?

    composite = outcome × format × (
        0.70 · sub_step_coverage
      + 0.30
    )

`sub_step_coverage` = fraction of gold_sub_steps whose key tokens
appear in the agent's tool-call history.
"""
from __future__ import annotations
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


def _all_args_text(transcript) -> str:
    out = []
    for msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls(msg):
            args = tc.get("input") or tc.get("arguments") or {}
            if isinstance(args, dict):
                out.append(str(args.get("command") or args.get("cmd") or ""))
                out.append(str(args.get("path") or ""))
                out.append(str(args.get("content") or "")[:200])
    return "\n".join(out)


def _outcome(rollout) -> float:
    """All gold_sub_steps must appear in TOOL-CALL args (or, for PR
    description sub-steps like '## Summary', in the final assistant text).

    Guards against the "echo the steps without doing them" cheat.
    """
    task = rollout.get("task") or {}
    steps = task.get("gold_sub_steps") or []
    if not steps:
        return 0.0
    args_text = _all_args_text(rollout.get("transcript") or []).lower()
    final = _final_text(rollout.get("transcript") or []).lower()
    hits = 0
    for s in steps:
        sl = s.lower()
        if sl in args_text:
            hits += 1
        elif sl in ("## summary", "summary:") and sl in final:
            hits += 1
    return 1.0 if hits == len(steps) else 0.0


def _format_compliance(rollout) -> float:
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).strip().lower()
    if not final:
        return 0.0
    if any(w in final for w in ("done", "complete", "ready", "committed", "branch", "pr")):
        return 1.0
    return 0.5


def _sub_step_coverage(rollout) -> float:
    """Coverage based on tool-call args only — freetext claims don't count.

    Exception: '## Summary' (PR description) is a freetext step, so we
    additionally accept Markdown summary headings from final text.
    """
    task = rollout.get("task") or {}
    steps = task.get("gold_sub_steps") or []
    if not steps:
        return 0.0
    args_text = _all_args_text(rollout.get("transcript") or []).lower()
    final = _final_text(rollout.get("transcript") or []).lower()
    hits = 0
    for s in steps:
        sl = s.lower()
        if sl in args_text:
            hits += 1
        elif sl in ("## summary", "summary:") and sl in final:
            hits += 1
    return hits / len(steps)


W_COV = 0.70
BASE = 0.30


def score_one(rollout: dict) -> dict[str, Any]:
    outcome = _outcome(rollout)
    fmt = _format_compliance(rollout)
    cov = _sub_step_coverage(rollout)
    process = W_COV * cov + BASE
    composite = outcome * fmt * process
    return {
        "composite": float(max(0.0, min(1.0, composite))),
        "outcome": float(outcome),
        "format_compliance": float(fmt),
        "sub_step_coverage": float(cov),
    }


CHEAT_PROBES = [
    "echo gold sub-step strings in text without doing them (mitigation: hits text in args/cmd, not freetext)",
    "skip the test step (mitigation: outcome requires ALL sub-steps)",
]
