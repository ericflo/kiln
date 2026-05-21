"""Rubric for code-symbol-extraction.

Programmatic. Given a code snippet, ground-truth symbols, and the model's
response, computes 4 sub-scores in [0, 1].

The eval set ships ground-truth symbols pre-computed by build_corpus.py —
the rubric doesn't have to re-extract them at scoring time. This makes the
oracle deterministic and fast.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any


# ---------------------------------------------------------------------------
# parses
# ---------------------------------------------------------------------------

_GIBBERISH_PATTERNS = [
    re.compile(r"(.)\1{15,}"),
    re.compile(r"(\{|\}|\[|\]){8,}"),
]


def score_parses(response: str) -> float:
    s = (response or "").strip()
    if len(s) < 2:
        return 0.0
    for pat in _GIBBERISH_PATTERNS:
        if pat.search(s):
            return 0.0
    printable = sum(1 for c in s if c.isprintable() or c == "\n") / max(len(s), 1)
    if printable < 0.85:
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# format_compliance
# ---------------------------------------------------------------------------
#
# Acceptable output forms:
#   foo
#   foo
#   bar
#   Baz
# OR with optional kind annotation in parens:
#   foo (function)
#   Baz (class)
# Anything else (markdown bullets, prose, code fences, blank lines mixed in)
# loses credit proportionally.

_LINE_OK = re.compile(r"^[a-zA-Z_$][a-zA-Z0-9_$]*\s*(\([a-zA-Z ]+\))?\s*$")
_LINE_BULLET = re.compile(r"^\s*[-*•]\s+")
_LINE_CODE_FENCE = re.compile(r"^\s*```")


def score_format_compliance(response: str) -> float:
    """Fraction of non-empty lines that are clean symbol names."""
    if not response or not response.strip():
        return 0.0
    lines = [ln.strip() for ln in response.strip().splitlines()]
    non_empty = [ln for ln in lines if ln]
    if not non_empty:
        return 0.0
    ok = 0
    for ln in non_empty:
        # Strip optional leading bullet
        cleaned = _LINE_BULLET.sub("", ln).strip()
        if _LINE_CODE_FENCE.match(ln):
            continue  # code fences don't count for or against
        if _LINE_OK.match(cleaned):
            ok += 1
    # Penalty for prose-only lines
    return ok / len(non_empty)


# ---------------------------------------------------------------------------
# Symbol extraction from response
# ---------------------------------------------------------------------------

def extract_listed_symbols(response: str) -> list[str]:
    """Pull the symbol-shaped tokens from the response.

    Accepts: one-per-line, optional `name (kind)` annotation, optional
    leading bullet. Rejects prose-y lines (sentences with verbs etc.).
    """
    if not response:
        return []
    listed: list[str] = []
    for ln in response.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # Strip leading bullet markers
        ln = _LINE_BULLET.sub("", ln).strip()
        # Skip code fences
        if _LINE_CODE_FENCE.match(ln):
            continue
        # Strip optional `(kind)` annotation
        bare = re.sub(r"\s*\([^)]*\)\s*$", "", ln).strip()
        # Strip optional `: type` (e.g. "foo: function")
        bare = re.split(r"[\s:]+", bare)[0]
        # Strip backticks
        bare = bare.strip("`")
        if not bare:
            continue
        if re.match(r"^[a-zA-Z_$][a-zA-Z0-9_$]*$", bare):
            listed.append(bare)
    return listed


# ---------------------------------------------------------------------------
# recall / precision
# ---------------------------------------------------------------------------

def score_recall(ground_truth: list[str], response: str) -> float:
    """Fraction of ground-truth symbols mentioned in response."""
    if not ground_truth:
        return 1.0
    listed = set(extract_listed_symbols(response))
    gt = set(ground_truth)
    overlap = listed & gt
    return len(overlap) / len(gt)


def score_precision(ground_truth: list[str], response: str) -> float:
    """Fraction of listed symbols that are actually ground-truth."""
    listed = extract_listed_symbols(response)
    if not listed:
        return 0.0  # If nothing listed at all, that's a precision failure
    gt = set(ground_truth)
    correct = sum(1 for s in listed if s in gt)
    return correct / len(listed)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

WEIGHTS = {
    "parses": 0.15,
    "format_compliance": 0.15,
    "symbol_recall": 0.35,
    "symbol_precision": 0.35,
}


def score_response(ground_truth: list[str], response: str) -> dict[str, float]:
    s_parses = score_parses(response)
    s_fmt = score_format_compliance(response)
    s_recall = score_recall(ground_truth, response)
    s_prec = score_precision(ground_truth, response)
    composite = (
        WEIGHTS["parses"] * s_parses
        + WEIGHTS["format_compliance"] * s_fmt
        + WEIGHTS["symbol_recall"] * s_recall
        + WEIGHTS["symbol_precision"] * s_prec
    )
    return {
        "parses": s_parses,
        "format_compliance": s_fmt,
        "symbol_recall": s_recall,
        "symbol_precision": s_prec,
        "composite": composite,
    }


def main() -> None:
    """CLI: read JSONL of {ground_truth, response} pairs from stdin."""
    sums = dict.fromkeys(WEIGHTS.keys(), 0.0)
    sums["composite"] = 0.0
    n = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        s = score_response(d.get("ground_truth", []), d.get("response", ""))
        for k in sums:
            sums[k] += s[k]
        n += 1
    if n == 0:
        print("ORACLE_ERROR: no responses scored", file=sys.stderr)
        sys.exit(2)
    print(f"SCORE={sums['composite']/n:.4f}")
    for k in ["parses", "format_compliance", "symbol_recall", "symbol_precision"]:
        print(f"{k}={sums[k]/n:.4f}")
    print(f"N={n}")


if __name__ == "__main__":
    main()
