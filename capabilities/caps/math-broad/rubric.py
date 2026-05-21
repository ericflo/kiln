"""Composite reward function for sft/math-broad (v0).

Compare final-answer substring match against gold_answer (case
insensitive). The cap targets WORD-PROBLEM accuracy.
"""
from __future__ import annotations
import re
from typing import Any

RUBRIC_VERSION = "v0"


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\$,]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def score_one(rollout: dict) -> dict[str, Any]:
    response = rollout.get("response") or rollout.get("completion") or ""
    task = rollout.get("task") or rollout
    gold = (task.get("gold_answer") or task.get("answer") or "").strip()
    if not gold or not response:
        return {"composite": 0.0, "exact_match": 0.0, "substring_match": 0.0}
    response = response.strip()
    gold_n = _normalize(gold)
    resp_n = _normalize(response)
    exact = 1.0 if resp_n == gold_n else 0.0
    substr = 1.0 if gold_n in resp_n else 0.0
    composite = max(exact, substr)
    return {
        "composite": float(composite),
        "exact_match": float(exact),
        "substring_match": float(substr),
        "accuracy": float(composite),
    }


CHEAT_PROBES = [
    "very long response containing the answer somewhere (mitigation: substring match acceptable here)",
    "wrong format (e.g. write '$30' as 'thirty dollars') (mitigation: normalize strips $ but not words)",
]
