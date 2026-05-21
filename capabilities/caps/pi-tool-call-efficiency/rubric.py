"""Composite reward function for pi-tool-call-efficiency (v0, eval-only).

This cap is repurposed for round 2 as a TRANSFER EVAL: measures how
efficiently various adapters use tool calls. Single sub-score is enough
because the cap doesn't train its own adapter; it just reports
distributions across other caps' adapters.

The composite is the per-adapter mean tool-call efficiency:

    efficiency = 1 - clip((n_tool_calls - efficient_cap) / wasteful_band, 0, 1)

where efficient_cap = 4 (an efficient run uses <=4 tool calls) and
wasteful_band = 8 (composite hits 0 at n_tool_calls >= 12).

Outcome is informational here — the cap doesn't train so outcome
doesn't gate. The composite IS the efficiency.
"""
from __future__ import annotations
from typing import Any

RUBRIC_VERSION = "v0-eval-only"

EFFICIENT_CAP = 4
WASTEFUL_BAND = 8  # composite=0 at n_tool_calls = EFFICIENT_CAP + WASTEFUL_BAND


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


def _n_tool_calls(rollout) -> int:
    transcript = rollout.get("transcript") or []
    n = 0
    for msg in _iter_messages(transcript):
        if msg.get("role") == "assistant":
            n += len(_tool_calls(msg))
    return n


def score_one(rollout: dict) -> dict[str, Any]:
    n = _n_tool_calls(rollout)
    if n <= EFFICIENT_CAP:
        eff = 1.0
    elif n >= EFFICIENT_CAP + WASTEFUL_BAND:
        eff = 0.0
    else:
        eff = 1.0 - (n - EFFICIENT_CAP) / WASTEFUL_BAND
    return {
        "composite": float(max(0.0, min(1.0, eff))),
        "tool_call_efficiency": float(eff),
        "n_tool_calls": int(n),
        "bucket": "efficient" if n <= 4 else ("moderate" if n <= 9 else "wasteful"),
    }


CHEAT_PROBES = [
    "0 tool calls + wrong answer (mitigation: outcome is reported separately by caller)",
    "split one call into many small calls (mitigation: composite linearly decays with n)",
]
