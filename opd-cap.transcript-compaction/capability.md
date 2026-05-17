# Capability: transcript-compaction

## Description
The model is shown a multi-turn agent ↔ user transcript (or assistant
"thinking + actions" log) of 500–2000 tokens, and asked to produce a
compact summary that **preserves operational state**: identifiers (file
paths, function names, error strings, command-line snippets), decisions
made, and open questions — so a fresh agent can pick up where the
original left off without re-asking the user.

This is distinct from "code summarization" (capability #1):
- Code summarization optimizes for *comprehension* by an outside reader.
- **Compaction optimizes for *continuation* by the next worker.**

Concrete failure modes the 4B exhibits today:
- Drops file paths and error strings (loses recoverable state)
- Rewrites the user's intent in paraphrase that loses precision
- Adds a "summary" that's pleasant prose but useless to a downstream agent
- Sometimes fabricates content not in the transcript (worst failure)

## Base model
Qwen3.5-4B (kiln serve on http://localhost:8420)

## Teacher
`vllm/qwen3.6-27b-awq` at http://localhost:8002

## Rubric

| Sub-score | Weight | What it measures |
|-----------|--------|-------------------|
| `entity_recall` | 0.40 | Of the operational entities in the transcript (file paths, error tokens, command snippets, identifier-shaped names), what fraction does the compaction mention? Heavy weight — this IS the capability. |
| `entity_grounding` | 0.30 | Fraction of n-grams (n=6) in the compaction that appear in the original transcript. Anti-hallucination. |
| `length_band` | 0.15 | Compacted token count is 5–25% of original (full credit). Linear penalty outside. |
| `decision_retention` | 0.15 | Fraction of imperatives / decisions / error-acknowledgments (`fix:`, `error:`, `decided to`, `do not`, `should`, etc.) preserved. |

Composite = `0.40 × entity_recall + 0.30 × entity_grounding + 0.15 × length_band + 0.15 × decision_retention`.
Direction: higher is better.

This rubric is *deliberately spread* across 4 sub-scores with no one
weight >0.4 — so headroom is well-distributed rather than parked in one
slot. Tests whether the OPD skill can navigate multi-axis tracking.

## Baseline

| Sub-score | Weight | Baseline | Headroom (w×(1−b)) |
|-----------|--------|----------|---------------------|
| entity_recall | 0.40 | 0.6764 | 0.1294 |
| entity_grounding | 0.30 | 0.9606 | 0.0118 |
| length_band | 0.15 | 0.6739 | 0.0489 |
| decision_retention | 0.15 | 0.4540 | 0.0819 |
| **Total movable** | | | **0.2720** |

Composite: **0.7279** (huge headroom — well-distributed across 3 sub-scores).

## Target sub-score

**`entity_recall`** owns 47% of movable headroom (0.1294 of 0.272). The 4B compactions drop ~32% of source-transcript operational entities — file paths, function names, error strings — making the compactions less useful for handoff. Secondary headroom on decision_retention (30%) and length_band (18%).

## Hypothesis log
| iter | slug | family | composite | comp Δ | target Δ | verdict |
|------|------|--------|-----------|--------|----------|---------|
|      |      |        |           |        |          |         |

## Dead ends
(none yet)

## Open questions
(none yet)

## Checkpoints
(every 3rd iter)
