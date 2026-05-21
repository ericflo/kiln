"""Composite reward function for pi-precondition-check (v0).

Round-2 TRIPLE-MULTIPLICATIVE-GATE composite:

    composite = outcome × format × verified_before_mutation × (
        0.40 · staleness_correctly_detected
      + 0.20 · no_phantom_edit
      + 0.40 · base
    )

`verified_before_mutation` is a multiplicative gate alongside outcome
and format because the *entire point* of this cap is "verify before
you mutate." The mutate-without-read cheat must score near zero;
under an additive composite it scored 0.70 (passing outcome + format
+ default base + staleness "no sentinel emitted" + no_phantom = 0.70),
which would teach the model that skipping reads is acceptable.

For stale tasks (no mutation), verified=1.0 by construction, so the
gate fires only on holds_true tasks that skipped the read — exactly
the failure mode this cap targets.

Round-1 spec called out the always-emit-sentinel cheat as the §0 to
mitigate. v0 implements that *and* the more dangerous
mutate-without-read cheat via this triple-gate.
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


def _next_tool_result_text(transcript, after_idx):
    msgs = list(_iter_messages(transcript))
    for j in range(after_idx + 1, len(msgs)):
        msg = msgs[j]
        if msg.get("role") != "tool":
            continue
        for tr in msg.get("content") or []:
            if isinstance(tr, dict) and tr.get("type") == "toolResult":
                c = tr.get("content")
                if isinstance(c, str):
                    return c
    return ""


def _final_text(transcript):
    final = ""
    for msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or []
        text = "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if text.strip():
            final = text.strip()
    return final


SEARCH_PAT = re.compile(r"\b(grep|rg|ag|find|locate|fd)\b")


def _is_read_op(name, args):
    if name in ("read", "cat", "open"):
        return True
    if name in ("bash", "shell", "exec"):
        cmd = (args.get("command") or args.get("cmd") or "")
        if SEARCH_PAT.search(cmd):
            return True  # grep reads
        return bool(re.search(r"\b(cat|less|head|tail|sed -n)\b", cmd))
    return False


def _is_mutating_op(name, args):
    if name in ("write", "edit", "replace"):
        return True
    if name in ("bash", "shell", "exec"):
        cmd = (args.get("command") or args.get("cmd") or "")
        return bool(re.search(r"\b(sed -i|cat\s*>|tee\s+|\brm\b|\bmv\b|git\s+commit|git\s+apply|patch\s+-)", cmd))
    return False


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------

def _outcome(rollout) -> float:
    v = rollout.get("outcome_passed")
    return 1.0 if v is True else 0.0


def _format_compliance(rollout) -> float:
    """Final assistant turn:
      (0.5) names the claim path or claim symbol
      (0.5) for stale: emits the sentinel; for holds: confirms the edit
    """
    task = rollout.get("task") or {}
    label = task.get("claim_label")
    claim_path = (task.get("claim_path") or "").lower()
    claim_symbol = (task.get("claim_symbol") or "").lower()
    sentinel = (task.get("sentinel") or "").lower()
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).strip().lower()
    if not final:
        return 0.0
    score = 0.0
    if claim_path in final or claim_symbol in final:
        score += 0.5
    if label == "stale":
        if sentinel in final:
            score += 0.5
    else:
        if any(w in final for w in ("done", "complete", "fixed", "edited", "applied", "migrated", "renamed", "changed", "updated")):
            score += 0.5
    return min(1.0, score)


def _verified_before_mutation(rollout) -> float:
    task = rollout.get("task") or {}
    label = task.get("claim_label")
    claim_path = (task.get("claim_path") or "").lower()
    claim_symbol = (task.get("claim_symbol") or "").lower()
    transcript = rollout.get("transcript") or []
    calls = _calls(transcript)

    mutations = [(i, n, a) for i, n, a in calls if _is_mutating_op(n, a)]
    if label == "stale":
        return 1.0 if not mutations else 0.0

    if not mutations:
        return 0.0  # holds_true task with no mutation
    first_mut_idx = mutations[0][0]

    for i, n, a in calls:
        if i >= first_mut_idx:
            break
        if not _is_read_op(n, a):
            continue
        p = (a.get("path") or "").lower()
        cmd = (a.get("command") or a.get("cmd") or "").lower()
        haystack = p + " " + cmd
        if claim_path and claim_path in haystack:
            text = _next_tool_result_text(transcript, i).lower()
            if claim_symbol and claim_symbol in text:
                return 1.0
            return 0.7
    return 0.0


def _staleness_correctly_detected(rollout) -> float:
    task = rollout.get("task") or {}
    label = task.get("claim_label")
    sentinel = (task.get("sentinel") or "").lower()
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).strip().lower()
    has_sentinel = bool(sentinel and sentinel in final)
    if label == "stale":
        return 1.0 if has_sentinel else 0.0
    else:
        return 0.0 if has_sentinel else 1.0


def _no_phantom_edit(rollout) -> float:
    task = rollout.get("task") or {}
    init_files = set((task.get("init_files") or {}).keys())
    transcript = rollout.get("transcript") or []
    for _, n, a in _calls(transcript):
        if not _is_mutating_op(n, a):
            continue
        p = a.get("path")
        if isinstance(p, str) and p not in init_files:
            return 0.0
    return 1.0


W_STALENESS = 0.40
W_NO_PHANTOM = 0.20
BASE = 0.40


def score_one(rollout: dict) -> dict[str, Any]:
    """Composite: outcome × format × verified × (staleness + no_phantom + base).

    `verified_before_mutation` is a multiplicative gate alongside outcome
    and format because the entire point of this cap is "verify before
    you mutate." The mutate-without-read cheat must score near zero.

    For stale tasks (where there's no mutation), verified=1.0 by
    construction, so the gate only fires on holds_true tasks that skipped
    the read.
    """
    outcome = _outcome(rollout)
    fmt = _format_compliance(rollout)
    verified = _verified_before_mutation(rollout)
    staleness = _staleness_correctly_detected(rollout)
    no_phantom = _no_phantom_edit(rollout)
    process = W_STALENESS * staleness + W_NO_PHANTOM * no_phantom + BASE
    composite = outcome * fmt * verified * process
    return {
        "composite": float(max(0.0, min(1.0, composite))),
        "outcome": float(outcome),
        "format_compliance": float(fmt),
        "verified_before_mutation": float(verified),
        "staleness_correctly_detected": float(staleness),
        "no_phantom_edit": float(no_phantom),
    }


CHEAT_PROBES = [
    "always-emit-sentinel (mitigation: staleness=0 on holds_true → composite=0)",
    "mutate-without-read (mitigation: verified_before_mutation=0 on holds_true)",
    "read-wrong-file-then-mutate (mitigation: claim_path overlap check)",
    "skip-edit-claim-already-exists (mitigation: outcome verifies workspace state)",
]
