"""Rubric for transcript-compaction.

4 programmatic sub-scores in [0, 1]:
  - entity_recall (40%): operational entities (paths, errors, identifiers) preserved
  - no_fabrication (30%): n-gram overlap with original (anti-hallucination)
  - length_band (15%): compacted is 5-25% of original token count
  - decision_retention (15%): imperatives/decisions/error words preserved

Composite = 0.40*entity_recall + 0.30*no_fabrication + 0.15*length_band + 0.15*decision_retention
"""
from __future__ import annotations

import json
import re
import sys
from typing import Iterable


# ---------------------------------------------------------------------------
# Entity extraction from a transcript
#
# Operational entities are anything a downstream agent would need to
# recover work-in-progress:
#   - File paths (slashes, dots, common suffixes)
#   - Function/class names (snake_case, CamelCase, identifier-shaped)
#   - Error strings (lines starting with "Error:", "error[", "panic:", etc)
#   - Command snippets (start with cargo/npm/git/python/kiln/curl/etc)
# ---------------------------------------------------------------------------

_PATH_RE = re.compile(r"\b[\w./-]*/[\w./-]+\b")  # contains a slash
_PATH_WITH_DOT = re.compile(r"\b[\w-]+\.(?:py|rs|js|ts|go|java|c|cpp|h|hpp|md|json|jsonl|toml|yaml|yml|sh|jinja)\b")
_SNAKE_OR_CAMEL = re.compile(r"\b(?:[a-z][a-z0-9]*_[a-z0-9_]+|[A-Z][a-z]+[A-Z][A-Za-z0-9]+|[a-z][a-z0-9]*[A-Z][A-Za-z0-9]+)\b")
_BACKTICKED = re.compile(r"`([^`\n]+)`")
_ERROR_LINE = re.compile(r"^(error|panic|exception|traceback|stderr|fatal|warn|warning|failed)[: ]", re.I | re.M)
_NUMERIC_HEX = re.compile(r"\b(?:0x)?[0-9a-fA-F]{4,}\b")  # error hex, hashes, addrs

_COMMON_WORDS = {
    "The", "This", "That", "These", "Those", "There", "Here", "When", "Where",
    "Then", "Now", "JavaScript", "TypeScript", "Python", "Rust", "Java", "Go",
    "True", "False", "None", "Null",
}


def extract_entities(text: str) -> set[str]:
    if not text:
        return set()
    out: set[str] = set()
    # Paths
    out.update(_PATH_RE.findall(text))
    out.update(_PATH_WITH_DOT.findall(text))
    # Snake/Camel identifiers
    for m in _SNAKE_OR_CAMEL.findall(text):
        if m not in _COMMON_WORDS and len(m) >= 3:
            out.add(m)
    # Backtick-wrapped
    for m in _BACKTICKED.findall(text):
        # Pull the bare token out of `foo.bar()` -> "foo"
        bare = m.split(".")[0].split("(")[0].strip()
        if len(bare) >= 2:
            out.add(bare)
    # Hex/long-numeric (error codes, hashes)
    for m in _NUMERIC_HEX.findall(text):
        out.add(m)
    return out


# ---------------------------------------------------------------------------
# Sub-score: entity_recall
# ---------------------------------------------------------------------------

def score_entity_recall(transcript: str, compaction: str) -> float:
    ents = extract_entities(transcript)
    if not ents:
        return 1.0
    mentioned = extract_entities(compaction)
    overlap = ents & mentioned
    return len(overlap) / len(ents)


# ---------------------------------------------------------------------------
# Sub-score: entity_grounding (anti-hallucination on identifiers)
#
# Of the entities the compaction mentions, what fraction appear anywhere
# in the source transcript? Good compactions paraphrase liberally but
# don't INVENT identifiers — file paths, function names, error strings.
# This catches the worst failure mode (fabricated entities) while
# allowing the natural paraphrase + markdown structure of a real summary.
# Replaces the older "no_fabrication" n-gram-overlap metric, which
# penalized perfectly accurate compactions for paraphrasing.
# ---------------------------------------------------------------------------

def score_entity_grounding(transcript: str, compaction: str) -> float:
    comp_ents = extract_entities(compaction)
    if not comp_ents:
        return 1.0  # vacuously grounded (entity_recall pulls down empty)
    # An entity is grounded if it appears verbatim anywhere in the transcript.
    # Use a relaxed check: case-insensitive substring or matching identifier.
    trans_lower = transcript.lower()
    grounded = 0
    for ent in comp_ents:
        if ent.lower() in trans_lower:
            grounded += 1
    return grounded / len(comp_ents)


# ---------------------------------------------------------------------------
# Sub-score: length_band
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def score_length_band(transcript: str, compaction: str) -> float:
    t = _word_count(transcript)
    c = _word_count(compaction)
    if t == 0 or c == 0:
        return 0.0
    ratio = c / t
    if 0.05 <= ratio <= 0.25:
        return 1.0
    # Linear penalty
    if ratio < 0.05:
        return max(0.0, ratio / 0.05)
    # ratio > 0.25: penalty zero at ratio=0.50
    return max(0.0, 1.0 - (ratio - 0.25) / 0.25)


# ---------------------------------------------------------------------------
# Sub-score: decision_retention
# ---------------------------------------------------------------------------

# Two classes of "decision/state" markers, both worth capturing in a
# compaction:
#   1. Failure/error signals — "error:", "panic:", "OOM", "TypeError"
#   2. Decisions/actions — both verb forms ("decided", "will") and noun
#      forms ("decision:", "next steps:"); plus imperative action verbs
#      ("Revert", "Fix", "Implement") that signal what's being done.
_DECISION_PATTERNS = [
    re.compile(r"\b(error|warning|panic|exception|failed|traceback|stderr|fatal)\b[:.]?", re.I),
    re.compile(r"\b(decid(?:e|ed|es|ing)|decision[s]?)\b", re.I),
    re.compile(r"\b(chose|picked|opted|will|should|must|cannot|wo?n't)\b", re.I),
    re.compile(r"\b(fix(?:ed|ing)?|bug|issue|todo|task[s]?|broken|crash|hang|timeout|oom|leak)\b", re.I),
    re.compile(r"\b(revert(?:ed|ing)?|implement(?:ed|ing|s)?|rename(?:d|ing)?|update[ds]?|"
               r"add(?:ed|ing|s)?|remov(?:e|ed|ing|es)?|delet(?:e|ed|ing|es)?|"
               r"chang(?:e|ed|ing|es)?|modify|modified|edit(?:ed|ing|s)?|"
               r"creat(?:e|ed|ing|es)?|push(?:ed|ing)?|pull(?:ed|ing)?|ran|run)\b", re.I),
    re.compile(r"\b(next step[s]?|action[s]?|plan|status|root cause|blocker[s]?)\b[:.]?", re.I),
    re.compile(r"\?\s*$", re.M),
]


def _count_decisions(text: str) -> int:
    if not text:
        return 0
    return sum(len(pat.findall(text)) for pat in _DECISION_PATTERNS)


def score_decision_retention(transcript: str, compaction: str) -> float:
    """Fraction of source's decision/action markers preserved (with
    sub-linear credit — a compaction at 10-20% length capturing 30% of
    source's markers gets full credit, since per-token density of
    decisions is usually higher in the compaction)."""
    src = _count_decisions(transcript)
    if src == 0:
        return 1.0
    kept = _count_decisions(compaction)
    return min(1.0, kept / max(src * 0.3, 1))


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

WEIGHTS = {
    "entity_recall": 0.40,
    "entity_grounding": 0.30,
    "length_band": 0.15,
    "decision_retention": 0.15,
}


def score_response(transcript: str, compaction: str) -> dict[str, float]:
    s_ent = score_entity_recall(transcript, compaction)
    s_grd = score_entity_grounding(transcript, compaction)
    s_len = score_length_band(transcript, compaction)
    s_dec = score_decision_retention(transcript, compaction)
    composite = (
        WEIGHTS["entity_recall"] * s_ent
        + WEIGHTS["entity_grounding"] * s_grd
        + WEIGHTS["length_band"] * s_len
        + WEIGHTS["decision_retention"] * s_dec
    )
    return {
        "entity_recall": s_ent,
        "entity_grounding": s_grd,
        "length_band": s_len,
        "decision_retention": s_dec,
        "composite": composite,
    }


def main() -> None:
    sums = dict.fromkeys(WEIGHTS.keys(), 0.0)
    sums["composite"] = 0.0
    n = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        d = json.loads(line)
        s = score_response(d.get("transcript", ""), d.get("response", ""))
        for k in sums:
            sums[k] += s[k]
        n += 1
    if n == 0:
        print("ORACLE_ERROR: no responses scored", file=sys.stderr)
        sys.exit(2)
    print(f"SCORE={sums['composite']/n:.4f}")
    for k in ["entity_recall", "entity_grounding", "length_band", "decision_retention"]:
        print(f"{k}={sums[k]/n:.4f}")
    print(f"N={n}")


if __name__ == "__main__":
    main()
