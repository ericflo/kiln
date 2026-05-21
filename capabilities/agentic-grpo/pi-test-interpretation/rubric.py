"""Composite reward function for pi-test-interpretation (v0).

Multiplicative-gate composite:

    composite = outcome × format × (
        0.30 · ran_multiple_iterations
      + 0.30 · reported_median_not_mean
      + 0.20 · classified_flakes
      + 0.20                         # base
    )

The cap targets bench/test discipline: run >=3 times, report median,
classify flakes vs real failures.
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
    out = []
    for msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls(msg):
            args = tc.get("input") or tc.get("arguments") or {}
            if isinstance(args, dict):
                c = args.get("command") or args.get("cmd") or ""
                if isinstance(c, str):
                    out.append(c)
    return "\n".join(out)


def _outcome(rollout) -> float:
    # Reuse the iteration counter so loop/for/seq idioms count.
    iters = _ran_multiple_iterations(rollout)
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).strip().lower()
    mentions_median_or_flake = bool(re.search(r"\b(median|flake|flaky)\b", final))
    return 1.0 if (iters >= 1.0 and mentions_median_or_flake) else 0.0


def _format_compliance(rollout) -> float:
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).strip().lower()
    if not final:
        return 0.0
    score = 0.0
    if re.search(r"\b(run|iteration|trial|attempt)\s*\d", final):
        score += 0.5
    if any(w in final for w in ("median", "flaky", "flake", "warmup", "discount", "report")):
        score += 0.5
    return min(1.0, score)


def _ran_multiple_iterations(rollout) -> float:
    cmds = _all_commands(rollout.get("transcript") or [])
    n = len(re.findall(r"\b(pytest|cargo\s+test|cargo\s+nextest|bench)\b", cmds))
    # Bash for-loop with 3+ iterations
    if re.search(r"\bfor\s+\w+\s+in\s+1\s+2\s+3", cmds):
        n = max(n, 3)
    if re.search(r"\bfor\s+\w+\s+in\s+\{?1\.\.[35]\}?", cmds):
        n = max(n, 3)
    if re.search(r"\b(seq\s+1\s+[35]|seq\s+[35])\b", cmds):
        n = max(n, 3)
    # Final-message claims of N runs (Run 1: ... Run 2: ... Run 3: ...)
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).lower()
    run_mentions = len(re.findall(r"\brun\s*\d", final))
    if run_mentions >= 3:
        n = max(n, 3)
    if n >= 3:
        return 1.0
    if n >= 2:
        return 0.5
    return 0.0


def _reported_median_not_mean(rollout) -> float:
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).strip().lower()
    if "median" in final:
        return 1.0
    if "mean" in final and "median" not in final:
        return 0.0
    return 0.5


def _classified_flakes(rollout) -> float:
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).strip().lower()
    task = rollout.get("task") or {}
    scenario = task.get("scenario_id", "")
    if "flake" not in scenario and "real_vs_flake" not in scenario:
        return 1.0
    if any(w in final for w in ("flaky", "flake")) and any(w in final for w in ("real", "fail", "broken")):
        return 1.0
    if "flaky" in final or "flake" in final:
        return 0.5
    return 0.0


W_ITER = 0.30
W_MEDIAN = 0.30
W_FLAKE = 0.20
BASE = 0.20


def score_one(rollout: dict) -> dict[str, Any]:
    outcome = _outcome(rollout)
    fmt = _format_compliance(rollout)
    iters = _ran_multiple_iterations(rollout)
    median = _reported_median_not_mean(rollout)
    flake = _classified_flakes(rollout)
    process = W_ITER * iters + W_MEDIAN * median + W_FLAKE * flake + BASE
    composite = outcome * fmt * process
    return {
        "composite": float(max(0.0, min(1.0, composite))),
        "outcome": float(outcome),
        "format_compliance": float(fmt),
        "ran_multiple_iterations": float(iters),
        "reported_median_not_mean": float(median),
        "classified_flakes": float(flake),
    }


CHEAT_PROBES = [
    "report mean (mitigation: reported_median=0 if 'mean' present without 'median')",
    "1 run + claim 'median' (mitigation: ran_multiple_iterations needs >=3)",
    "ignore flakes (mitigation: classified_flakes=0 on flake tasks)",
]
