"""Wraps text.* to produce a small audit report on a corpus."""

from __future__ import annotations

from typing import List

from text import normalize_whitespace, word_counts, truncate, find_first


def audit_lines(lines: List[str], probe: str = "ERROR") -> dict:
    cleaned = [normalize_whitespace(l) for l in lines]
    counts = word_counts(" ".join(cleaned))
    idx = find_first(probe, cleaned)
    digest = truncate(" | ".join(cleaned), 120)
    return {"counts": counts, "first_match": idx, "digest": digest}
