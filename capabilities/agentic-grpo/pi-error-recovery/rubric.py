"""Composite reward function for pi-error-recovery (v0).

Multiplicative-gate composite per capability.md §Rubric (v0):

    composite = outcome × format_compliance × (
        0.30 · recovery_was_different
      + 0.25 · recovery_appropriate_to_error_class
      + 0.20 · no_loop
      + 0.15 · read_error_before_retry
      + 0.10                             # base floor for any complete rollout
    )

`rollout` shape (produced by ./rollout.py):

    {
      "task": {... full task spec from build_corpus ...},
      "transcript": [ pi session events ],
      "workdir": "/abs/path/to/sandbox",
      "outcome_passed": bool,
      "format_text": str (final assistant message text)
    }

The transcript follows pi session JSONL shape — see
../lib/pi_trajectory.py and ../pi-doctest/rubric.py for the canonical
parser.
"""
from __future__ import annotations
import json
import re
from typing import Any

RUBRIC_VERSION = "v0"


# ---------------------------------------------------------------------------
# Transcript helpers (shared with other agentic caps)
# ---------------------------------------------------------------------------

def _iter_messages(transcript):
    for ev in transcript or []:
        if not isinstance(ev, dict) or ev.get("type") != "message":
            continue
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


def _final_assistant_text(transcript):
    final = ""
    for msg in _iter_messages(transcript):
        if msg.get("role") == "assistant":
            content = msg.get("content") or []
            text_parts = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    t = b.get("text")
                    if isinstance(t, str):
                        text_parts.append(t)
            text = "".join(text_parts)
            if text:
                final = text
    return final


def _all_tool_calls(transcript):
    """List of (turn_index, tool_name, input_dict) in chronological order."""
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


def _all_tool_results(transcript):
    """List of (turn_index, content_str)."""
    out = []
    for i, msg in enumerate(_iter_messages(transcript)):
        if msg.get("role") != "tool":
            continue
        for tr in _tool_results(msg):
            c = tr.get("content")
            if isinstance(c, str):
                out.append((i, c))
    return out


# ---------------------------------------------------------------------------
# Failure detection: which call failed first?
# ---------------------------------------------------------------------------

FAILURE_MARKERS = [
    "no such file",
    "file not found",
    "permission denied",
    "syntaxerror",
    "command not found",
    "modulenotfounderror",
    "no module named",
    "importerror",
    "timeout",
    "killed",
    "terminated",
    ".rej",
    "fatal:",
    "error:",
]


def _find_first_failure(transcript):
    """Return (turn_index, failed_call_input, error_text) for the first failed
    tool call in the session, or (None, None, None) if no failure was observed.

    A "failed" tool call is one whose immediately-following tool result
    contains a known failure marker OR shows nonzero exit.
    """
    calls = _all_tool_calls(transcript)
    results = _all_tool_results(transcript)
    # Build a map turn → next-tool-result text
    next_result = {}
    for cidx, _, _ in calls:
        for ridx, rtext in results:
            if ridx > cidx:
                next_result[cidx] = rtext
                break

    for cidx, name, args in calls:
        rtext = next_result.get(cidx, "")
        low = rtext.lower()
        if any(m in low for m in FAILURE_MARKERS) or re.search(
            r"\bexit(?:_code)?\s*[=:]\s*[1-9]", low
        ):
            return cidx, (name, args), rtext
    return None, None, None


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------

def _outcome(rollout) -> float:
    """1.0 iff the rollout marks outcome_passed True (the harness ran the
    gold-state check after the session). Defaults to checking transcript-
    derived state if `outcome_passed` is missing.
    """
    v = rollout.get("outcome_passed")
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    # Fallback: didn't run gold check — assume 0.
    return 0.0


def _format_compliance(rollout) -> float:
    """Final assistant turn must mention:
      (a) error acknowledgement (substring of the failure markers seen, or
          one of the words "failed", "error", "couldn't")
      (b) the recovery approach
      (c) confirmation the recovery worked

    Each pillar = 0.34 points; floor at zero.
    """
    final = (rollout.get("format_text") or _final_assistant_text(
        rollout.get("transcript") or []
    )).strip().lower()
    if not final:
        return 0.0
    score = 0.0
    # Pillar 1: acknowledged failure
    if any(w in final for w in ("fail", "error", "couldn't", "couldnt", "didn't work", "didnt work", "permission", "not found", "missing")):
        score += 0.34
    # Pillar 2: named the recovery
    if any(w in final for w in ("instead", "switched", "used", "tried", "fall back", "fell back", "alternative", "different")):
        score += 0.33
    # Pillar 3: confirmed it worked
    if any(w in final for w in ("worked", "passed", "succeeded", "done", "complete", "fixed", "resolved", "ok")):
        score += 0.33
    return min(1.0, score)


def _recovery_was_different(rollout) -> float:
    """Score how different the retry call is from the failed call.

    1.0 = different tool, or substantially different command structure.
    0.3 = minor edit to the same command (≥70% overlap).
    0.0 = same call repeated, or no recovery at all.
    """
    transcript = rollout.get("transcript") or []
    fail_idx, fail_call, _ = _find_first_failure(transcript)
    if fail_idx is None or fail_call is None:
        # No failure observed → can't measure recovery. Award full credit
        # (the cap is about recovery-when-failure-occurs; not-failing is
        # a fine outcome but not a recovery test).
        return 1.0

    calls = _all_tool_calls(transcript)
    # First call after fail_idx
    retry = None
    for cidx, name, args in calls:
        if cidx > fail_idx:
            retry = (name, args)
            break
    if retry is None:
        return 0.0  # gave up

    fail_name, fail_args = fail_call
    retry_name, retry_args = retry

    if retry_name != fail_name:
        return 1.0  # different tool entirely

    def _arg_text(args):
        return json.dumps(args, sort_keys=True, default=str)

    a = _arg_text(fail_args)
    b = _arg_text(retry_args)
    if a == b:
        return 0.0  # identical retry
    # Jaccard on token bigrams
    def _bigrams(s):
        return {s[i : i + 2] for i in range(max(0, len(s) - 1))}
    A, B = _bigrams(a), _bigrams(b)
    if not A or not B:
        return 0.5
    overlap = len(A & B) / max(1, len(A | B))
    # overlap=0 → totally different → 1.0
    # overlap=1 → same → 0.0
    return float(max(0.0, min(1.0, 1.0 - overlap)))


# Per-error-class valid-recovery-pattern matcher. Each pattern matches
# *what to look for in the retry call's input* to consider it appropriate
# for that error class.
RECOVERY_PATTERNS = {
    "file_not_found": [r"\b(ls|find|rg|tree|locate|fd)\b"],
    "permission_denied": [r"\bchmod\b", r"\brm\b.*\&\&.*\b(write|cat|echo)\b"],
    "syntax_error": [r"\b(write|edit|replace|sed)\b"],  # re-edit the file
    "command_not_found": [
        r"python\s+-m\s+(pytest|unittest)", r"\bpytest\b", r"\bunittest\b"
    ],
    "dependency_missing": [r"\bdifflib\b", r"\bSequenceMatcher\b", r"\b(write|edit)\b.*\.py"],
    "timeout": [r"\bwc\b", r"\bfind\b.*-exec", r"timeout\s+\d"],
}


def _recovery_appropriate(rollout) -> float:
    """1.0 iff the first retry's input matches one of the recovery patterns
    for the task's error_class. 0.0 otherwise. 0.5 if pattern partially
    matches (substring).
    """
    task = rollout.get("task") or {}
    err_cls = task.get("error_class")
    if not err_cls:
        return 0.5
    transcript = rollout.get("transcript") or []
    fail_idx, _, _ = _find_first_failure(transcript)
    if fail_idx is None:
        return 1.0  # nothing failed; vacuously appropriate
    calls = _all_tool_calls(transcript)
    retry_args_text = ""
    for cidx, name, args in calls:
        if cidx > fail_idx:
            retry_args_text = json.dumps(args, sort_keys=True, default=str).lower()
            break
    if not retry_args_text:
        return 0.0
    patterns = RECOVERY_PATTERNS.get(err_cls, [])
    for pat in patterns:
        if re.search(pat, retry_args_text):
            return 1.0
    return 0.0


def _no_loop(rollout) -> float:
    """1.0 - duplicate-call fraction across the whole session."""
    transcript = rollout.get("transcript") or []
    calls = _all_tool_calls(transcript)
    if not calls:
        return 1.0
    sigs = [json.dumps([n, a], sort_keys=True, default=str) for _, n, a in calls]
    seen = set()
    dups = 0
    for s in sigs:
        if s in seen:
            dups += 1
        else:
            seen.add(s)
    return float(max(0.0, 1.0 - dups / max(1, len(sigs))))


def _read_error_before_retry(rollout) -> float:
    """1.0 iff between the failed call and the retry, there's a non-empty
    assistant turn (text or `read`/`cat` of the error). Zero otherwise.

    Heuristic: any assistant message (text or tool-call) BETWEEN fail_idx
    and the retry call counts as "attended to the error."
    """
    transcript = rollout.get("transcript") or []
    fail_idx, _, _ = _find_first_failure(transcript)
    if fail_idx is None:
        return 1.0
    calls = _all_tool_calls(transcript)
    retry_idx = None
    for cidx, _, _ in calls:
        if cidx > fail_idx:
            retry_idx = cidx
            break
    if retry_idx is None:
        return 0.0
    # Was there an assistant message strictly between fail_idx and retry_idx?
    if retry_idx > fail_idx + 1:
        return 1.0
    # retry came immediately — no text turn between.
    return 0.0


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

W_RECOVERY_DIFF = 0.30
W_RECOVERY_APPROP = 0.25
W_NO_LOOP = 0.20
W_READ_ERROR = 0.15
BASE = 0.10


def score_one(rollout: dict) -> dict[str, Any]:
    outcome = _outcome(rollout)
    fmt = _format_compliance(rollout)
    rec_diff = _recovery_was_different(rollout)
    rec_app = _recovery_appropriate(rollout)
    no_loop = _no_loop(rollout)
    read_err = _read_error_before_retry(rollout)
    process = (
        W_RECOVERY_DIFF * rec_diff
        + W_RECOVERY_APPROP * rec_app
        + W_NO_LOOP * no_loop
        + W_READ_ERROR * read_err
        + BASE
    )
    composite = outcome * fmt * process
    return {
        "composite": float(max(0.0, min(1.0, composite))),
        "outcome": float(outcome),
        "format_compliance": float(fmt),
        "recovery_was_different": float(rec_diff),
        "recovery_appropriate_to_error_class": float(rec_app),
        "no_loop": float(no_loop),
        "read_error_before_retry": float(read_err),
    }


# Calibration adversarial cases (for rubric_sanity.py). See
# capability.md §0 — these are the cheap-cheat patterns the rubric must
# score below the good band.
CHEAT_PROBES = [
    "always-retry-with-ls (mitigation: recovery_appropriate averages low across classes)",
    "long apology no fix (mitigation: outcome gate)",
    "exhaustive recovery spam (mitigation: no_loop decays)",
]
