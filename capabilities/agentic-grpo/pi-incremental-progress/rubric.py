"""Composite reward function for pi-incremental-progress (v0).

Multiplicative-gate composite per capability.md §Rubric (v0):

    composite = outcome × format_compliance × (
        0.30 · step_progress_observability
      + 0.20 · step_alignment_with_decomposition
      + 0.20 · early_failure_caught
      + 0.15 · no_big_bang
      + 0.15                             # base floor
    )

The rubric depends on `task.gold_decomposition` (ordered list of step
descriptions) and the session transcript. We measure:

- `step_progress_observability`: count of intermediate test/verify calls
  between edits.
- `step_alignment_with_decomposition`: token-overlap between actual
  edit→verify sequence and gold steps.
- `early_failure_caught`: did the agent's NEXT turn address an
  intermediate verify failure?
- `no_big_bang`: first edit doesn't touch >50% of gold-touched files.
"""
from __future__ import annotations
import json
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


def _tool_results(msg):
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "toolResult"]


def _calls_with_idx(transcript):
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


VERIFY_PAT = re.compile(
    r"\b(pytest|python\s*-?m\s*(?:doctest|unittest)|cargo\s+test|"
    r"go\s+test|npm\s+test|tox)\b"
)


def _is_verify(name, args):
    if name not in ("bash", "shell", "exec"):
        return False
    cmd = args.get("command") or args.get("cmd") or ""
    return bool(VERIFY_PAT.search(str(cmd).lower()))


def _is_edit(name, args):
    if name in ("write", "edit", "replace"):
        return True
    if name in ("bash", "shell", "exec"):
        cmd = (args.get("command") or args.get("cmd") or "").lower()
        return bool(re.search(r"\b(sed -i|cat\s+>|tee\s+)\b", cmd))
    return False


def _final_assistant_text(transcript):
    final = ""
    for msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or []
        text_parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                t = b.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
        text = "".join(text_parts).strip()
        if text:
            final = text
    return final


def _outcome(rollout) -> float:
    v = rollout.get("outcome_passed")
    return 1.0 if v is True else 0.0


def _format_compliance(rollout) -> float:
    """Final turn must list the sub-steps taken. Score:
      0.5 — has a numbered or bulleted list of >=2 items
      0.5 — list items reference a tool that actually happened in the session
    """
    final = (rollout.get("format_text") or _final_assistant_text(
        rollout.get("transcript") or []
    )).strip()
    if not final:
        return 0.0
    # Detect a numbered/bulleted list of >=2 items
    items = re.findall(r"(?m)^\s*(?:\d+\.|[-*+])\s+(\S.+)$", final)
    score = 0.5 if len(items) >= 2 else 0.0

    # Check that at least 2 list items overlap with actual session content
    transcript = rollout.get("transcript") or []
    session_keywords = set()
    for _, _, args in _calls_with_idx(transcript):
        cmd = (args.get("command") or args.get("cmd") or args.get("path") or "")
        for tok in re.findall(r"[A-Za-z_][\w.]+", str(cmd)):
            if len(tok) >= 3:
                session_keywords.add(tok.lower())

    matched = 0
    for it in items:
        for tok in re.findall(r"[A-Za-z_][\w.]+", it):
            if tok.lower() in session_keywords and len(tok) >= 3:
                matched += 1
                break
    if matched >= 2:
        score += 0.5
    elif matched == 1:
        score += 0.25
    return min(1.0, score)


def _step_progress_observability(rollout) -> float:
    """Count verifies BETWEEN the first and last edit; >= 2 → 1.0."""
    transcript = rollout.get("transcript") or []
    calls = _calls_with_idx(transcript)
    if not calls:
        return 0.0
    edit_indices = [i for i, n, a in calls if _is_edit(n, a)]
    if len(edit_indices) < 2:
        return 0.0
    first, last = edit_indices[0], edit_indices[-1]
    verifies_between = sum(
        1 for i, n, a in calls if first < i < last and _is_verify(n, a)
    )
    return min(1.0, verifies_between / 2.0)


def _step_alignment_with_decomposition(rollout) -> float:
    """Token-bag overlap between the gold decomposition and the agent's
    edit→verify sequence keywords. Naïve but effective for v0.
    """
    task = rollout.get("task") or {}
    gold = task.get("gold_decomposition") or []
    if not gold:
        return 0.0
    transcript = rollout.get("transcript") or []
    calls = _calls_with_idx(transcript)

    # Build the agent's path: chronological keywords from edit + verify calls.
    agent_tokens = []
    for _, n, a in calls:
        if not (_is_edit(n, a) or _is_verify(n, a)):
            continue
        for v in (a.values() if isinstance(a, dict) else []):
            agent_tokens += re.findall(r"[A-Za-z_][\w.]+", str(v))

    agent_set = {t.lower() for t in agent_tokens if len(t) >= 3}
    if not agent_set:
        return 0.0
    gold_tokens = set()
    for step in gold:
        for t in re.findall(r"[A-Za-z_][\w.]+", step):
            if len(t) >= 3:
                gold_tokens.add(t.lower())
    if not gold_tokens:
        return 0.0
    overlap = len(agent_set & gold_tokens) / len(gold_tokens)
    return float(min(1.0, overlap))


def _early_failure_caught(rollout) -> float:
    """For each intermediate verify call that failed, was the NEXT
    assistant turn topical (mentioned the failure / changed behavior)?

    Heuristic: between an intermediate verify-fail tool-result and the
    next edit, there is at least one assistant text turn OR the next
    edit's content differs substantially from the previous edit.
    """
    transcript = rollout.get("transcript") or []
    msgs = list(_iter_messages(transcript))
    calls = _calls_with_idx(transcript)
    # Walk messages and identify intermediate verify-failures.
    catches = 0
    chances = 0
    for k, msg in enumerate(msgs):
        if msg.get("role") != "tool":
            continue
        # Was this a verify result?
        prev_assistant_was_verify = False
        for j in range(k - 1, -1, -1):
            pm = msgs[j]
            if pm.get("role") == "assistant":
                for tc in _tool_calls(pm):
                    args = tc.get("input") or {}
                    if _is_verify(tc.get("name", ""), args):
                        prev_assistant_was_verify = True
                break
        if not prev_assistant_was_verify:
            continue
        # Was this verify a failure?
        text = ""
        for tr in _tool_results(msg):
            c = tr.get("content")
            if isinstance(c, str):
                text += c
        low = text.lower()
        if "fail" not in low and "error" not in low and "traceback" not in low:
            continue
        chances += 1
        # Is there a subsequent assistant text turn within 2 messages?
        caught = False
        for j in range(k + 1, min(k + 4, len(msgs))):
            nm = msgs[j]
            if nm.get("role") == "assistant":
                content = nm.get("content") or []
                has_text = any(
                    isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
                    for b in content
                )
                if has_text:
                    caught = True
                    break
        if caught:
            catches += 1
    if chances == 0:
        return 1.0  # vacuously OK
    return catches / chances


def _no_big_bang(rollout) -> float:
    """First edit shouldn't touch >50% of gold-touched files.

    We can't see "files touched in one edit" directly from the
    transcript without inspecting workdir state per turn. As a proxy:
    the first edit's `path` argument is one file; if the first edit is
    an `edit`/`write` call we score 1.0 (single file). If the first edit
    is a bash command with `git apply`/multi-file sed → score 0.0.
    """
    transcript = rollout.get("transcript") or []
    calls = _calls_with_idx(transcript)
    for _, n, a in calls:
        if _is_edit(n, a):
            if n in ("write", "edit", "replace"):
                return 1.0  # single-file edit
            cmd = (a.get("command") or a.get("cmd") or "").lower()
            if "git apply" in cmd or "patch -" in cmd:
                return 0.0
            if cmd.count("sed -i") >= 2 or cmd.count(">") >= 2:
                return 0.0
            return 0.8  # bash edit, single file
    return 1.0  # no edit yet — vacuously OK


# Composite
W_PROGRESS = 0.30
W_ALIGN = 0.20
W_CAUGHT = 0.20
W_NO_BIG_BANG = 0.15
BASE = 0.15


def score_one(rollout: dict) -> dict[str, Any]:
    outcome = _outcome(rollout)
    fmt = _format_compliance(rollout)
    progress = _step_progress_observability(rollout)
    align = _step_alignment_with_decomposition(rollout)
    caught = _early_failure_caught(rollout)
    nbb = _no_big_bang(rollout)
    process = (
        W_PROGRESS * progress
        + W_ALIGN * align
        + W_CAUGHT * caught
        + W_NO_BIG_BANG * nbb
        + BASE
    )
    composite = outcome * fmt * process
    return {
        "composite": float(max(0.0, min(1.0, composite))),
        "outcome": float(outcome),
        "format_compliance": float(fmt),
        "step_progress_observability": float(progress),
        "step_alignment_with_decomposition": float(align),
        "early_failure_caught": float(caught),
        "no_big_bang": float(nbb),
    }


CHEAT_PROBES = [
    "spam pytest between random edits (mitigation: alignment requires gold-tokens overlap)",
    "tiny throwaway first edit then everything-in-one (mitigation: alignment punishes empty first)",
    "memorize gold sequence (mitigation: held-out eval pool)",
    "claim N steps in summary without doing them (mitigation: format cross-checks session)",
]
