"""Pi-compaction multi-component rubric (v1).

The capability under test: given a serialized pi conversation, produce a
structured summary in pi's EXACT format that preserves the load-bearing
state (goals, files, errors, identifiers, next steps) without
hallucination, while compressing the source.

Design goals for the rubric:
- **Programmatic where possible.** Heading checks, path matches, identifier
  matches, length ratios — all deterministic, fast, GRPO-friendly.
- **Anti-shortcut by construction.** Every "reward presence" sub-score
  has a paired "punish shortcut" sub-score (see capability.md adversarial
  audit table). The cheapest paths to 1.0 without doing the capability
  are all blocked.
- **Calibratable.** `good.jsonl` should score ≥ 0.85 composite; `bad.jsonl`
  ≤ 0.20. Run `rubric_sanity.py` before iter 1.

The composite weighting (10/30/20/10/10/20 = 100) is intentional: content
is the largest single contributor because preserving facts is the central
job, faithfulness comes next because hallucination defeats the purpose,
and the outcome gate (20%) collapses composite if format or content fall
through the floor.

API
---
score_rollout(response_text: str, source_text: str, ground_truth: dict)
    -> dict with keys:
        composite               (float in [0, 1])
        format_*                (sub-scores, dict)
        content_*               (sub-scores, dict)
        faithfulness_*          (sub-scores, dict)
        compression_*           (sub-scores, dict)
        continuability_*        (sub-scores, dict)
        outcome                 (0/1 gate)
        _diagnostics            (raw counts for debugging)

Each sub-score lives in [0, 1]. The composite is a fixed weighted sum
with a gate term.
"""

from __future__ import annotations

import re
from typing import Any


# ============================================================================
# Constants
# ============================================================================

# Pi's EXACT compaction format sections, in order.
PI_FORMAT_SECTIONS = [
    "## Goal",
    "## Constraints & Preferences",
    "## Progress",
    "## Key Decisions",
    "## Next Steps",
    "## Critical Context",
]

# Progress sub-sections (any one is OK as long as the parent ## Progress
# is present).
PROGRESS_SUBSECTIONS = ["### Done", "### In Progress", "### Blocked"]

# Phrases that indicate the model is continuing the conversation rather than
# summarizing it. Pi explicitly forbids this in its system prompt.
CONTINUATION_PHRASES = [
    r"\bI'll\b", r"\bI'd\b", r"\bI can help\b", r"\bI'm happy to\b",
    r"\bLet me\b", r"\bHere's what I\b", r"\bSure[,\s!]\b",
    r"\bGreat question\b", r"\bGood question\b",
    r"\bI hope this helps\b", r"\bAny other questions\b",
    r"\bWould you like me to\b", r"\bDo you want me to\b",
    r"\bShould I\b",
    # Tool-call-style continuations are also bad.
    r"<tool_call>",
    r"```\s*python",
    r"```\s*bash",
    r"```\s*typescript",
]

# Composite weights — kept in one place so they're easy to tune later.
W_FORMAT = 0.10
W_CONTENT = 0.30
W_FAITHFULNESS = 0.20
W_COMPRESSION = 0.10
W_CONTINUABILITY = 0.10
W_OUTCOME = 0.20

# Outcome gate thresholds
GATE_FORMAT_MIN = 0.5
GATE_CONTENT_MIN = 0.30
GATE_FAITHFULNESS_PATHS_MIN = 0.7  # if summary cites paths not in source, gate fails
GATE_FLOOR = 0.10  # if gate fails, composite floors at floor * (f+c+faith)/3


# ============================================================================
# Format scoring
# ============================================================================

def _has_section(text: str, heading: str) -> bool:
    """Case-insensitive, whitespace-tolerant section heading check."""
    pat = re.compile(
        r"(?:^|\n)" + re.escape(heading) + r"(?:[ \t]*$|[ \t]+\S)",
        re.IGNORECASE | re.MULTILINE,
    )
    return bool(pat.search(text))


def _has_progress_subsection(text: str) -> bool:
    """Pi requires ## Progress with Done/InProgress/Blocked underneath.
    At least one subsection must exist."""
    return any(_has_section(text, s) for s in PROGRESS_SUBSECTIONS)


def _continuation_flag(text: str) -> int:
    """Returns 1 if the response continues the conversation (BAD), else 0."""
    for pat in CONTINUATION_PHRASES:
        if re.search(pat, text, re.IGNORECASE):
            return 1
    # Specific failure: the response starts with role-shaped chatter like
    # "Hi there," "Hello," "Sure,"
    first_line = text.strip().split("\n", 1)[0]
    if re.match(r"^(Hi|Hello|Sure|Of course|Absolutely|Great|Thanks)[!,.\s]", first_line, re.IGNORECASE):
        return 1
    return 0


def _sections_in_order(text: str) -> bool:
    """All present sections must appear in pi's specified order."""
    positions = []
    for s in PI_FORMAT_SECTIONS:
        m = re.search(r"(?:^|\n)" + re.escape(s), text, re.IGNORECASE | re.MULTILINE)
        if m:
            positions.append((s, m.start()))
    if not positions:
        return False
    sorted_positions = sorted(positions, key=lambda x: x[1])
    expected_order = [s for s, _ in sorted_positions]
    canonical_order = [s for s in PI_FORMAT_SECTIONS if any(p[0] == s for p in positions)]
    return expected_order == canonical_order


def score_format(response: str) -> dict[str, float]:
    """Format compliance sub-scores."""
    out = {
        "format.has_goal": float(_has_section(response, "## Goal")),
        "format.has_constraints": float(_has_section(response, "## Constraints & Preferences")),
        "format.has_progress": float(
            _has_section(response, "## Progress") and _has_progress_subsection(response)
        ),
        "format.has_key_decisions": float(_has_section(response, "## Key Decisions")),
        "format.has_next_steps": float(_has_section(response, "## Next Steps")),
        "format.has_critical_context": float(_has_section(response, "## Critical Context")),
        "format.order_correct": float(_sections_in_order(response)),
        "format.no_continuation": 1.0 - _continuation_flag(response),
    }
    out["format.score"] = sum(out.values()) / len(out)
    return out


# ============================================================================
# Source-fact extraction (already done for the source by extract_ground_truth.py)
# ============================================================================

# Path regex: matches things like src/foo.py, /tmp/bar.txt, ./baz.json
PATH_RE = re.compile(
    r"(?:(?<![A-Za-z0-9_/])"
    r"(?:[~./]?[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+)"
    r"(?:\.[A-Za-z0-9]{1,8})?)"
)

# Generic identifier regex: `function_name`, `ClassName`, `module.method`
IDENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]{2,}|[a-z][a-z0-9_]{4,}(?:\.[a-z_]+)?)\b")

# Error keyword regex — common Python / shell error fragments
ERROR_PAT = re.compile(
    r"\b(?:Traceback|Error|Exception|FAILED|fatal:|SyntaxError|"
    r"TypeError|ValueError|NameError|AttributeError|KeyError|"
    r"IndexError|ImportError|ModuleNotFoundError|FileNotFoundError|"
    r"OSError|RuntimeError|AssertionError|RecursionError|StopIteration|"
    r"command not found|No such file|Permission denied|exit status [1-9])\b",
    re.IGNORECASE,
)


def extract_paths(text: str) -> set[str]:
    """Extract plausible file-like paths from text. Lower-cased and de-duped."""
    candidates = set()
    for m in PATH_RE.finditer(text):
        path = m.group(0)
        # Skip plain words / URLs / version strings
        if "://" in path:
            continue
        if not ("/" in path or "." in path):
            continue
        if len(path) < 4 or len(path) > 200:
            continue
        # Strip trailing punctuation
        path = path.rstrip(".,;:!?)]}")
        if path:
            candidates.add(path)
    return candidates


def extract_identifiers(text: str) -> set[str]:
    """Extract code-like identifiers from text."""
    out = set()
    for m in IDENT_RE.finditer(text):
        ident = m.group(1)
        # Filter obvious natural-language words
        if ident.lower() in COMMON_WORDS:
            continue
        if len(ident) < 4:
            continue
        out.add(ident)
    return out


def extract_error_fragments(text: str) -> list[str]:
    """Extract lines or fragments that look like errors."""
    out = []
    for line in text.splitlines():
        if ERROR_PAT.search(line):
            stripped = line.strip()
            if stripped and len(stripped) < 400:
                out.append(stripped)
    return out


# Words that look like identifiers but aren't load-bearing.
COMMON_WORDS = {
    "the", "this", "that", "these", "those", "what", "which", "where",
    "when", "while", "with", "without", "should", "would", "could",
    "result", "output", "error", "function", "method", "class", "module",
    "system", "string", "number", "value", "field", "object", "array",
    "request", "response", "session", "context", "conversation", "summary",
    "assistant", "user", "tool", "tools", "model", "models", "prompt",
    "prompts", "argument", "arguments", "parameter", "parameters",
    "return", "returns", "import", "imports", "package", "packages",
    "library", "libraries", "version", "versions",
    "completed", "started", "finished", "process", "running",
    "current", "previous", "following", "above", "below",
    "section", "sections", "summary", "summaries",
}


# ============================================================================
# Content sub-scores
# ============================================================================

def _normalize_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _word_recall(needle_text: str, haystack_text: str, min_word_len: int = 4) -> float:
    """Fraction of distinct content-words from `needle` that appear in `haystack`."""
    needle = _normalize_for_match(needle_text)
    haystack = _normalize_for_match(haystack_text)
    words = {w for w in re.findall(r"[a-z][a-z0-9_]*", needle) if len(w) >= min_word_len and w not in COMMON_WORDS}
    if not words:
        return 1.0  # nothing to recall, trivially full recall
    hit = sum(1 for w in words if w in haystack)
    return hit / max(1, len(words))


def _path_set_recall(needles: set[str], haystack_text: str) -> float:
    if not needles:
        return 1.0
    h_norm = _normalize_for_match(haystack_text)
    hits = 0
    for p in needles:
        # Match path either as-is or just its basename (production summaries often shorten)
        if p.lower() in h_norm:
            hits += 1
            continue
        basename = p.rsplit("/", 1)[-1].lower()
        if basename and len(basename) >= 4 and basename in h_norm:
            hits += 1
    return hits / len(needles)


def _identifier_set_recall(needles: set[str], haystack_text: str) -> float:
    if not needles:
        return 1.0
    h_norm = haystack_text  # case-sensitive on purpose for identifier match
    hits = sum(1 for ident in needles if ident in h_norm)
    return hits / len(needles)


def _error_recall(errors: list[str], haystack_text: str) -> float:
    if not errors:
        return 1.0
    h_norm = _normalize_for_match(haystack_text)
    hits = 0
    for err in errors:
        # Match if any rare identifier from the error appears in summary
        # OR the literal error keyword (Traceback, FileNotFoundError, etc.)
        err_norm = _normalize_for_match(err)
        # First, check if the error type word appears
        type_match = ERROR_PAT.search(err)
        if type_match and type_match.group(0).lower() in h_norm:
            hits += 1
            continue
        # Otherwise, check word overlap (3+ rare content words from the error)
        err_words = {w for w in re.findall(r"[a-z][a-z0-9_]*", err_norm) if len(w) >= 4 and w not in COMMON_WORDS}
        overlap = sum(1 for w in err_words if w in h_norm)
        if overlap >= min(3, len(err_words)):
            hits += 1
    return hits / len(errors)


def _extract_section_body(text: str, heading: str) -> str:
    """Return everything between `heading` and the next ## heading (or end)."""
    pat = re.compile(
        r"(?:^|\n)" + re.escape(heading) + r"\s*?\n+(.*?)(?=\n##\s|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pat.search(text)
    return m.group(1).strip() if m else ""


def _extract_file_block(text: str, tag: str) -> set[str]:
    """Pi appends <read-files>...</read-files> / <modified-files>...</modified-files>."""
    pat = re.compile(rf"<{tag}>\s*\n?(.*?)\n?\s*</{tag}>", re.IGNORECASE | re.DOTALL)
    m = pat.search(text)
    if not m:
        return set()
    return {line.strip() for line in m.group(1).splitlines() if line.strip()}


def score_content(response: str, ground_truth: dict[str, Any]) -> dict[str, float]:
    """Content recall sub-scores. ground_truth carries pre-extracted facts.

    Weighted average: sub-scores whose corresponding GT category is *empty*
    contribute neither to the numerator nor denominator (rather than
    trivially scoring 1.0, which inflates content.score for hallucinated
    responses that happen to fall into an empty-GT bucket).
    """
    first_user_goal = ground_truth.get("first_user_goal", "") or ""
    source_paths = set(ground_truth.get("source_paths") or [])
    modified_paths = set(ground_truth.get("modified_paths") or [])
    read_only_paths = set(ground_truth.get("read_only_paths") or [])
    source_identifiers = set(ground_truth.get("source_identifiers") or [])
    source_errors = list(ground_truth.get("source_errors") or [])

    goal_section = _extract_section_body(response, "## Goal")
    summary_no_file_blocks = re.sub(r"<(?:read|modified)-files>.*?</(?:read|modified)-files>", "", response, flags=re.DOTALL)

    # Goal recall always counts (every task has a first user message).
    goal_recall = _word_recall(first_user_goal, goal_section)

    # File path recall and identifier recall — always count.
    path_recall = _path_set_recall(source_paths, summary_no_file_blocks) if source_paths else None
    ident_recall = _identifier_set_recall(source_identifiers, summary_no_file_blocks) if source_identifiers else None
    err_recall = _error_recall(source_errors, summary_no_file_blocks) if source_errors else None

    read_block = _extract_file_block(response, "read-files")
    modified_block = _extract_file_block(response, "modified-files")

    read_block_recall = (
        len(read_block & read_only_paths) / len(read_only_paths) if read_only_paths else None
    )
    modified_block_recall = (
        len(modified_block & modified_paths) / len(modified_paths) if modified_paths else None
    )

    # Build the weighted average over present sub-scores only.
    weighted: list[tuple[str, float]] = [
        ("content.first_user_goal_recall", goal_recall),
    ]
    for name, val in [
        ("content.file_paths_recall", path_recall),
        ("content.identifier_recall", ident_recall),
        ("content.error_recall", err_recall),
        ("content.read_file_block_correctness", read_block_recall),
        ("content.modified_file_block_correctness", modified_block_recall),
    ]:
        if val is not None:
            weighted.append((name, val))

    if weighted:
        content_score = sum(v for _, v in weighted) / len(weighted)
    else:
        content_score = 0.0

    out: dict[str, float] = {name: val for name, val in weighted}
    # Surface the trivially-1.0 cases explicitly as "N/A" diagnostic-only.
    for name, val, present in [
        ("content.file_paths_recall", path_recall, source_paths),
        ("content.identifier_recall", ident_recall, source_identifiers),
        ("content.error_recall", err_recall, source_errors),
        ("content.read_file_block_correctness", read_block_recall, read_only_paths),
        ("content.modified_file_block_correctness", modified_block_recall, modified_paths),
    ]:
        if name not in out:
            # Empty-GT category: not included in averaged content score, but
            # surface in diagnostics so downstream analysis can see the shape.
            out[f"_diag.{name}_na"] = 1.0
    out["content.score"] = content_score
    return out


# ============================================================================
# Faithfulness (reverse direction) sub-scores
# ============================================================================

def score_faithfulness(response: str, source: str, ground_truth: dict[str, Any]) -> dict[str, float]:
    """Anti-hallucination checks. Every claim in the summary must be in source."""
    source_paths = set(ground_truth.get("source_paths") or [])
    source_identifiers = set(ground_truth.get("source_identifiers") or [])
    source_lower = source.lower()

    summary_no_file_blocks = re.sub(r"<(?:read|modified)-files>.*?</(?:read|modified)-files>", "", response, flags=re.DOTALL)
    summary_paths = extract_paths(summary_no_file_blocks) | _extract_file_block(response, "read-files") | _extract_file_block(response, "modified-files")
    summary_idents = extract_identifiers(summary_no_file_blocks)

    if summary_paths:
        path_supported = 0
        for p in summary_paths:
            if p in source_paths:
                path_supported += 1
                continue
            # Allow basename-only match
            basename = p.rsplit("/", 1)[-1].lower()
            if basename and basename in source_lower:
                path_supported += 1
        path_faithful = path_supported / len(summary_paths)
    else:
        path_faithful = 1.0

    if summary_idents:
        ident_supported = sum(1 for i in summary_idents if i in source or i in source_identifiers)
        ident_faithful = ident_supported / len(summary_idents)
    else:
        ident_faithful = 1.0

    # Invented error fragments: errors in summary that aren't in source.
    summary_errors = extract_error_fragments(summary_no_file_blocks)
    if summary_errors:
        e_supported = 0
        for err in summary_errors:
            # If the error type / key fragment appears in source, count as supported.
            err_norm = _normalize_for_match(err)
            err_words = {w for w in re.findall(r"[a-z][a-z0-9_]*", err_norm) if len(w) >= 4 and w not in COMMON_WORDS}
            if not err_words:
                e_supported += 1
                continue
            overlap = sum(1 for w in err_words if w in source_lower)
            if overlap >= min(3, len(err_words)):
                e_supported += 1
        err_faithful = e_supported / len(summary_errors)
    else:
        err_faithful = 1.0

    # Character n-gram overlap as a soft general-semantics check.
    semantic = _char_ngram_overlap(summary_no_file_blocks, source, n=4)

    out = {
        "faithfulness.file_paths_in_source": path_faithful,
        "faithfulness.identifiers_in_source": ident_faithful,
        "faithfulness.no_invented_errors": err_faithful,
        "faithfulness.semantic_overlap": semantic,
    }
    out["faithfulness.score"] = sum(out.values()) / len(out)
    return out


def _char_ngram_overlap(a: str, b: str, n: int = 4) -> float:
    """Fraction of summary's character n-grams that appear in source."""
    a_norm = _normalize_for_match(a)
    b_norm = _normalize_for_match(b)
    if len(a_norm) < n or len(b_norm) < n:
        return 0.0
    a_ng = {a_norm[i:i+n] for i in range(len(a_norm) - n + 1)}
    b_ng = {b_norm[i:i+n] for i in range(len(b_norm) - n + 1)}
    if not a_ng:
        return 0.0
    return len(a_ng & b_ng) / len(a_ng)


# ============================================================================
# Compression sub-scores
# ============================================================================

def score_compression(response: str, source: str) -> dict[str, float]:
    if not source:
        return {"compression.is_smaller": 1.0, "compression.ratio_band": 1.0, "compression.score": 1.0}
    r = len(response) / max(1, len(source))
    is_smaller = 1.0 if len(response) < len(source) else 0.0
    # Piecewise linear: ideal band 5%-25%, decay outside.
    if 0.05 <= r <= 0.25:
        band = 1.0
    elif r < 0.05:
        # Too short — likely missing content. Linear ramp from 0 at 0% to 1 at 5%.
        band = r / 0.05
    elif r <= 0.50:
        # Linear decay 0.25 → 0.50 maps to 1 → 0
        band = max(0.0, 1.0 - (r - 0.25) / 0.25)
    else:
        band = 0.0
    out = {
        "compression.is_smaller": is_smaller,
        "compression.ratio_band": band,
        "_diag.response_chars": float(len(response)),
        "_diag.source_chars": float(len(source)),
        "_diag.compression_ratio": float(r),
    }
    out["compression.score"] = (is_smaller + band) / 2
    return out


# ============================================================================
# Continuability sub-scores
# ============================================================================

def _parse_next_steps(response: str) -> list[str]:
    """Return list items under ## Next Steps."""
    body = _extract_section_body(response, "## Next Steps")
    if not body:
        return []
    items: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match "1. foo", "- foo", "* foo"
        m = re.match(r"^(?:\d+[.)]\s+|[-*+]\s+)(.*)", line)
        if m:
            items.append(m.group(1).strip())
    return items


def score_continuability(response: str, ground_truth: dict[str, Any]) -> dict[str, float]:
    items = _parse_next_steps(response)
    n = len(items)
    source_paths = ground_truth.get("source_paths") or []
    source_identifiers = ground_truth.get("source_identifiers") or []
    source_errors = ground_truth.get("source_errors") or []
    referable = set(p.lower() for p in source_paths) | set(i.lower() for i in source_identifiers) | set(e.lower() for e in source_errors[:20])

    present = float(n >= 1)
    concrete = 0.0
    if items:
        concrete_n = 0
        for item in items:
            il = item.lower()
            if any(ref and ref in il for ref in referable if len(ref) >= 4):
                concrete_n += 1
        concrete = concrete_n / n

    # 2-5 items is the productive band.
    if 2 <= n <= 5:
        count_band = 1.0
    elif n == 1 or n == 6:
        count_band = 0.5
    elif n == 0:
        count_band = 0.0
    else:
        count_band = max(0.0, 1.0 - (n - 6) * 0.2)

    out = {
        "continuability.next_steps_present": present,
        "continuability.next_steps_concrete": concrete,
        "continuability.count_in_band": count_band,
        "_diag.next_steps_count": float(n),
    }
    out["continuability.score"] = (present + concrete + count_band) / 3
    return out


# ============================================================================
# Composite
# ============================================================================

def score_rollout(response: str, source: str, ground_truth: dict[str, Any]) -> dict[str, Any]:
    """Score a compaction rollout.

    response: the model's compaction output (the structured summary)
    source: the original conversation text (what was fed to the model)
    ground_truth: pre-extracted facts (see extract_ground_truth.py):
        {
          "first_user_goal": str,           # text of first user message
          "source_paths": list[str],        # all unique file paths from source
          "modified_paths": list[str],      # paths in write/edit tool calls
          "read_only_paths": list[str],     # paths in read tool calls only
          "source_identifiers": list[str],  # function/class/method names
          "source_errors": list[str],       # error lines from tool results
        }
    """
    fmt = score_format(response)
    content = score_content(response, ground_truth)
    faith = score_faithfulness(response, source, ground_truth)
    comp = score_compression(response, source)
    cont = score_continuability(response, ground_truth)

    format_score = fmt["format.score"]
    content_score = content["content.score"]
    faith_score = faith["faithfulness.score"]
    comp_score = comp["compression.score"]
    cont_score = cont["continuability.score"]

    # Outcome gate: pass requires
    #   1. format >= 0.5 (legible pi format)
    #   2. content >= 0.30 (actually preserves source facts)
    #   3. faithfulness.file_paths_in_source >= 0.7 (no path hallucination)
    paths_faithful = faith.get("faithfulness.file_paths_in_source", 1.0)
    outcome_pass = (
        format_score >= GATE_FORMAT_MIN
        and content_score >= GATE_CONTENT_MIN
        and paths_faithful >= GATE_FAITHFULNESS_PATHS_MIN
    )
    outcome = 1.0 if outcome_pass else 0.0

    if outcome_pass:
        composite = (
            W_FORMAT * format_score
            + W_CONTENT * content_score
            + W_FAITHFULNESS * faith_score
            + W_COMPRESSION * comp_score
            + W_CONTINUABILITY * cont_score
            + W_OUTCOME * outcome
        )
    else:
        # Floor for failed gate: small fraction of partial signal.
        composite = GATE_FLOOR * (format_score + content_score + faith_score) / 3

    out: dict[str, Any] = {
        "composite": float(composite),
        "outcome": float(outcome),
        "format.score": format_score,
        "content.score": content_score,
        "faithfulness.score": faith_score,
        "compression.score": comp_score,
        "continuability.score": cont_score,
    }
    out.update(fmt)
    out.update(content)
    out.update(faith)
    out.update(comp)
    out.update(cont)
    return out


# ============================================================================
# Top-level convenience for the rollout harness
# ============================================================================

def score_text(response: str, prompt_text: str, task: dict[str, Any]) -> dict[str, Any]:
    """Compatibility shim with the prior pi-doctest harness pattern.

    `task` carries the source conversation text under `source_text` and the
    ground-truth dict under `ground_truth`.
    """
    source = task.get("source_text") or task.get("source") or ""
    ground_truth = task.get("ground_truth") or {}
    return score_rollout(response, source, ground_truth)


if __name__ == "__main__":  # pragma: no cover
    # Tiny self-test.
    sample = """## Goal
Fix the off-by-one error in the doctest for `circular_shift` in solution.py.

## Constraints & Preferences
- Do not modify the docstring.

## Progress
### Done
- [x] Read `/tmp/solution.py` and inspected `circular_shift`.

### In Progress
- [ ] Patching the loop bound.

### Blocked
(none)

## Key Decisions
- **Use slice arithmetic instead of explicit loop**: simpler edit.

## Next Steps
1. Update solution.py with the fixed body.
2. Run `python3 -m doctest -v solution.py`.

## Critical Context
- Function is in `/tmp/solution.py`.
"""
    source = """[User]: please fix the doctest failure in circular_shift
[Assistant tool calls]: read(path="/tmp/solution.py")
[Tool result]: def circular_shift(x, shift): ...
[Assistant tool calls]: bash(cmd="python3 -m doctest -v solution.py")
[Tool result]: AssertionError: doctest failed on circular_shift example 2
"""
    gt = {
        "first_user_goal": "fix the doctest failure in circular_shift",
        "source_paths": ["/tmp/solution.py"],
        "modified_paths": [],
        "read_only_paths": ["/tmp/solution.py"],
        "source_identifiers": ["circular_shift"],
        "source_errors": ["AssertionError: doctest failed on circular_shift example 2"],
    }
    result = score_rollout(sample, source, gt)
    for k, v in sorted(result.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
    print()
    print(f"COMPOSITE: {result['composite']:.4f}")
