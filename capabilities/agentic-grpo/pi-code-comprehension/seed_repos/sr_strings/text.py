"""String / text utilities."""

from __future__ import annotations

from typing import Dict, List, Optional


def normalize_whitespace(s: str) -> str:
    """Collapse runs of whitespace to single spaces and strip ends."""
    if not isinstance(s, str):
        raise TypeError("expected str")
    out: List[str] = []
    in_ws = False
    for ch in s:
        if ch.isspace():
            if not in_ws:
                out.append(" ")
                in_ws = True
        else:
            out.append(ch)
            in_ws = False
    return "".join(out).strip()


def word_counts(text: str) -> Dict[str, int]:
    """Return a dict of word→count. Words are case-folded.

    Requires text to be a non-None string.
    """
    if text is None:
        raise ValueError("text must not be None")
    counts: Dict[str, int] = {}
    for w in text.lower().split():
        counts[w] = counts.get(w, 0) + 1
    return counts


def truncate(text: str, max_len: int, suffix: str = "…") -> str:
    """Truncate `text` to at most `max_len` characters; append `suffix` if cut.

    Asserts max_len >= len(suffix).
    """
    assert max_len >= len(suffix), "max_len must be >= len(suffix)"
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def find_first(needle: str, haystacks: List[str]) -> Optional[int]:
    """Return index of first haystack containing needle; None if not found.

    Empty needle matches the first haystack.
    """
    for i, h in enumerate(haystacks):
        if needle in h:
            return i
    return None
