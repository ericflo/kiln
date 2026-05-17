# Capability: faithful-code-summarization

## Description
The model is shown a short code snippet (3–30 lines, any common language) and asked
to write a brief summary of what it does. **Faithful** means: every named entity
the summary mentions (function names, class names, variable names) must actually
appear in the code, and the summary must not invent behavior the code does not
contain. The summary should be concise (under ~150 words) and readable.

Today's 4B baseline gets the gist right but routinely (a) renames functions in
summaries ("the `process` function" when the code defines `handle`), (b) invents
helper functions the code doesn't define, and (c) attributes behavior to the
snippet that lives elsewhere ("then it logs to a file" when no logging is there).

OPD against a 27B teacher should pull the 4B closer to the teacher's grounded
behavior at the states the 4B itself visits.

## Base model
Qwen3.5-4B (kiln serve on http://localhost:8420)

## Teacher
`vllm/qwen3.6-27b-awq` at http://localhost:8002
AWQ-INT4 (Q4), `--max-logprobs 64`, `--gpu-memory-utilization 0.45`,
`--enforce-eager`, `--max-model-len 4096`.

## Rubric

| Sub-score | Weight | What it measures |
|-----------|--------|-------------------|
| `parses` | 0.20 | Response is readable text: ≥20 chars, contains spaces, no malformed control chars, no obvious gibberish. |
| `entity_recall` | 0.20 | Of the named entities in the code (functions, classes, top-level constants), what fraction does the summary mention by name? |
| `entity_precision` | 0.40 | Of the named entities the summary mentions, what fraction actually appears in the code? This is the **anti-hallucination** sub-score. Heavy weight. |
| `concise` | 0.20 | Word count in `[20, 150]` (full credit), with a soft penalty outside that band. |

Composite = `0.20 × parses + 0.20 × entity_recall + 0.40 × entity_precision + 0.20 × concise`.
Direction: higher is better. Composite is in `[0, 1]`.

## Baseline

| Sub-score | Weight | Baseline | Headroom (w×(1−b)) |
|-----------|--------|----------|---------------------|
| parses | 0.20 | 1.0000 | 0.0000 |
| entity_recall | 0.20 | 0.9500 | 0.0100 |
| entity_precision | 0.40 | 1.0000 | 0.0000 |
| concise | 0.20 | 1.0000 | 0.0000 |
| **Total movable** | | | **0.0100** |

**Headroom-gate fired: total movable headroom (0.01) is well under the
0.05 floor.** The skill (§0) halts the session before training. The 4B
already produces accurate, grounded, concise summaries on these 5–15
line snippets; OPD has nothing to teach here.

This capability would be OPD-able with a harder rubric (longer snippets
where the model genuinely drifts, stricter entity match requiring
called-function coverage, or an explicit anti-claim n-gram check). For
this session, we abandon it and move to capability #2.

## Target sub-score
<filled after baseline; `entity_precision` expected given the 4B's known hallucination pattern>

## Hypothesis log
| iter | slug | family | composite | comp Δ | target Δ | verdict |
|------|------|--------|-----------|--------|----------|---------|
|      |      |        |           |        |          |         |

## Dead ends
(none yet)

## Open questions
(none yet)

## Checkpoints
(every 3rd iter, write a brief progress summary here under `### Checkpoint at iter N`)
