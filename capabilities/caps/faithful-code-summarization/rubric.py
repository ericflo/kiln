"""Rubric for faithful-code-summarization.

Programmatic. No LLM-as-judge. Given a code snippet and a model's
summary of it, computes 4 sub-scores in [0, 1] and a composite per the
weights in capability.md.

This file is part of the rubric *contract* and is visible to the
experimentalist. The eval SET (which specific snippets are scored) is
blind — the agent must not read `datasets/eval.jsonl`.

Usage as a library:
    from rubric import score_response
    score_response(code, summary) -> dict

Each sub-score is in [0, 1]. The composite combines them per the
weights in capability.md.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Sub-score 1: parses (is the response readable text?)
# ---------------------------------------------------------------------------

_GIBBERISH_PATTERNS = [
    re.compile(r"(.)\1{15,}"),         # 16+ same char in a row
    re.compile(r"(\{|\}|\[|\]){8,}"),  # 8+ brackets in a row (the v6 failure mode)
    re.compile(r"(\\n){10,}"),          # excessive escaped newlines
]


def score_parses(summary: str) -> float:
    """1.0 if summary looks like English-ish prose; 0.0 if gibberish."""
    s = (summary or "").strip()
    if len(s) < 20:
        return 0.0
    if " " not in s:
        return 0.0
    for pat in _GIBBERISH_PATTERNS:
        if pat.search(s):
            return 0.0
    # Heuristic: at least 60% of chars are ASCII letters/spaces/punctuation
    printable_ratio = sum(1 for c in s if c.isprintable()) / max(len(s), 1)
    if printable_ratio < 0.85:
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Entity extraction
#
# Languages we cover: Python, Rust, JavaScript/TypeScript, Go, Java, C/C++.
# Extracts function names, class names, type names. NOT variables (too noisy).
# ---------------------------------------------------------------------------

# Function definitions
_FN_PATTERNS = [
    re.compile(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.MULTILINE),         # python
    re.compile(r"^\s*async\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.MULTILINE),  # python async
    re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[(<]", re.MULTILINE),  # rust
    re.compile(r"^\s*function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(", re.MULTILINE),  # js
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(", re.MULTILINE),  # js export
    re.compile(r"^\s*const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s*)?\(", re.MULTILINE),  # js arrow
    re.compile(r"^\s*func\s+(?:\([^)]+\)\s*)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.MULTILINE),  # go
    re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?[\w<>]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{", re.MULTILINE),  # java/c++ method
]

# Class / type definitions
_TYPE_PATTERNS = [
    re.compile(r"^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.MULTILINE),                   # python/js/java
    re.compile(r"^\s*(?:pub\s+)?struct\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.MULTILINE),       # rust/go
    re.compile(r"^\s*(?:pub\s+)?enum\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.MULTILINE),         # rust/go
    re.compile(r"^\s*(?:pub\s+)?trait\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.MULTILINE),        # rust
    re.compile(r"^\s*type\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.MULTILINE),                    # rust/go/ts
    re.compile(r"^\s*interface\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.MULTILINE),               # ts/java
]


def extract_code_entities(code: str) -> set[str]:
    """Return the set of function/class/type names DEFINED in `code`.

    Used for entity_recall — these are the names we expect a good summary
    to mention.
    """
    names: set[str] = set()
    for pat in _FN_PATTERNS + _TYPE_PATTERNS:
        for m in pat.finditer(code):
            name = m.group(1)
            if name in {"main", "test", "_"} or len(name) < 2:
                continue
            names.add(name)
    return names


def extract_code_universe(code: str) -> set[str]:
    """Return ALL identifier-shaped tokens appearing anywhere in `code`.

    Used for entity_precision — a summary's mention of `get_cache` is
    grounded if `get_cache` literally appears in the code, even as a
    call site rather than a definition. This is the looser, more honest
    test of "did the model invent a name?".
    """
    # Identifiers: snake_case, camelCase, CamelCase, dotted attributes broken at dots
    raw = re.findall(r"[a-zA-Z_$][a-zA-Z0-9_$]*", code)
    return {tok for tok in raw if len(tok) >= 2}


def extract_summary_entities(summary: str, code_entities: set[str]) -> set[str]:
    """Return the set of entity-shaped tokens mentioned in `summary`.

    We use a wide net: any backtick-wrapped or CamelCase or snake_case
    identifier token in the summary is a candidate. Then we intersect
    against the universe of plausible identifiers — anything in
    code_entities, plus standalone CamelCase / snake_case / camelCase
    tokens. Common English words are filtered.
    """
    # Backtick-wrapped: `foo`, `Foo.bar()`, etc. — strongest signal.
    backticked = re.findall(r"`([a-zA-Z_$][a-zA-Z0-9_$.]*)`", summary)
    backticked_names = {t.split(".")[0].split("(")[0] for t in backticked}
    backticked_names = {n for n in backticked_names if len(n) >= 2}

    # Bare identifier-shaped tokens: snake_case (with _) or CamelCase (with capital
    # somewhere in the middle, not at start). These are riskier; only keep ones
    # that aren't English words.
    bare_candidates = set(re.findall(r"\b([a-z][a-z0-9]+_[a-z0-9_]+)\b", summary))  # snake_case
    bare_candidates |= set(re.findall(r"\b([A-Z][a-z]+[A-Z][a-zA-Z]+)\b", summary))  # CamelCase like FooBar
    bare_candidates |= set(re.findall(r"\b([a-z][a-z0-9]*[A-Z][a-zA-Z0-9]+)\b", summary))  # camelCase

    bare_candidates -= _ENGLISH_FALSE_POSITIVES

    return backticked_names | bare_candidates


# A small list of likely false positives in English summaries
_ENGLISH_FALSE_POSITIVES = {
    "JavaScript", "TypeScript", "Python", "Rust", "Java",
    "True", "False", "None", "Null", "Undefined",
    "HTTPRequest", "JSON",  # may appear as language-feature mentions
}


def score_entity_recall(code: str, summary: str) -> float:
    """Of the entities in code, what fraction are named in the summary?

    Returns 1.0 if there are no code entities (vacuously perfect — there's
    nothing to recall).
    """
    code_entities = extract_code_entities(code)
    if not code_entities:
        return 1.0
    summary_entities = extract_summary_entities(summary, code_entities)
    overlap = code_entities & summary_entities
    return len(overlap) / len(code_entities)


def score_entity_precision(code: str, summary: str) -> float:
    """Of the entities the summary names, what fraction actually appear
    anywhere in the code (defined or called)?

    Uses the LOOSER `extract_code_universe` (any identifier appearing in
    code), not just defined names. A summary that says `get_cache` is
    grounded if the code calls `get_cache()` even if `get_cache` is
    defined elsewhere — the model isn't *inventing* the name.

    Returns 1.0 if the summary names nothing specific (vacuously precise);
    entity_recall pulls down a summary that names nothing.
    """
    code_universe = extract_code_universe(code)
    code_entities = extract_code_entities(code)
    summary_entities = extract_summary_entities(summary, code_entities)
    if not summary_entities:
        return 1.0
    overlap = summary_entities & code_universe
    return len(overlap) / len(summary_entities)


def score_concise(summary: str) -> float:
    """Word count in [20, 150] gets 1.0, with a linear penalty outside."""
    words = len(re.findall(r"\S+", summary or ""))
    if 20 <= words <= 150:
        return 1.0
    if words < 20:
        return max(0.0, words / 20.0)
    return max(0.0, 1.0 - (words - 150) / 150.0)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

WEIGHTS = {
    "parses": 0.20,
    "entity_recall": 0.20,
    "entity_precision": 0.40,
    "concise": 0.20,
}


def score_response(code: str, summary: str) -> dict[str, float]:
    """Score one (code, summary) pair. Returns sub-scores + composite."""
    s_parses = score_parses(summary)
    s_recall = score_entity_recall(code, summary)
    s_prec = score_entity_precision(code, summary)
    s_conc = score_concise(summary)
    composite = (
        WEIGHTS["parses"] * s_parses
        + WEIGHTS["entity_recall"] * s_recall
        + WEIGHTS["entity_precision"] * s_prec
        + WEIGHTS["concise"] * s_conc
    )
    return {
        "parses": s_parses,
        "entity_recall": s_recall,
        "entity_precision": s_prec,
        "concise": s_conc,
        "composite": composite,
    }


def main() -> None:
    """CLI: read JSONL of {code, response} pairs from stdin, print aggregate."""
    n = 0
    sums = {"parses": 0.0, "entity_recall": 0.0, "entity_precision": 0.0,
            "concise": 0.0, "composite": 0.0}
    per_prompt = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        code = d.get("code", "")
        summary = d.get("response", "")
        s = score_response(code, summary)
        per_prompt.append({"id": d.get("id"), **s})
        for k in sums:
            sums[k] += s[k]
        n += 1

    if n == 0:
        print("ORACLE_ERROR: no responses scored", file=sys.stderr)
        sys.exit(2)

    print(f"SCORE={sums['composite']/n:.4f}")
    print(f"parses={sums['parses']/n:.4f}")
    print(f"entity_recall={sums['entity_recall']/n:.4f}")
    print(f"entity_precision={sums['entity_precision']/n:.4f}")
    print(f"concise={sums['concise']/n:.4f}")
    print(f"N={n}")


if __name__ == "__main__":
    main()
