# Capability: tool-call-arg-fidelity

## Description
A coding agent that decides to call a tool (file_read, run_bash, web_search,
list_files, etc.) must emit a JSON tool-call object whose arguments are
well-formed and conform to the tool's schema. The 4B today often produces
JSON with subtle defects: a missing required field, a stringified int, an
extra "ignore_errors" key the tool doesn't accept, or a syntax error from
a trailing comma. Downstream the agent harness rejects these calls.

This is structured-output adjacent — same family as code-symbol-extraction
(capability #2, hit 97% skip-rate ceiling). Mitigation: use rollout
max_tokens=384 (well over typical tool-call length of ~60 tokens) so OPD
has loss landscape room per the rollout-sizing finding from capability #3.

Concrete failure modes the 4B exhibits:
- JSON syntax errors (trailing comma, missing brace, single quotes)
- Missing required fields (`file_read` call without `path`)
- Wrong types (`"offset": "10"` instead of `"offset": 10`)
- Hallucinated extras (`"ignore_errors": true` on a tool with no such arg)
- Sometimes wraps the tool call in prose ("I'll call file_read like so: {...}")

## Base model
Qwen3.5-4B (kiln serve on http://localhost:8420)

## Teacher
`vllm/qwen3.6-27b-awq` at http://localhost:8002 (AWQ-INT4, max_logprobs=64)

## Rubric

| Sub-score | Weight | What it measures |
|-----------|--------|-------------------|
| `parses` | 0.10 | Strict JSON.loads succeeds on the response (or first JSON block extracted). Low weight; guards the floor. |
| `required_fields` | 0.40 | Of the tool's required fields, what fraction are present. **Target sub-score.** |
| `type_correctness` | 0.30 | Of the fields the call includes, what fraction have the right type per schema. |
| `no_extra_fields` | 0.20 | 1.0 if no fields outside the schema; linearly penalized. Catches hallucinated args. |

Composite = `0.10 × parses + 0.40 × required_fields + 0.30 × type_correctness + 0.20 × no_extra_fields`.
Direction: higher is better.

Each eval prompt declares the tool schema in the system prompt (so rubric
and model share the contract). The rubric parses the first JSON object,
scoring each sub-score independently — so a response that fails `parses`
can still partially score via best-effort extraction.

## Baseline (filled by headroom.py after iter 0)
| Sub-score | Weight | Baseline | Headroom (w×(1−b)) |
|-----------|--------|----------|---------------------|
|           |        |          |                     |
| **Total** |        |          | **<sum>**           |

## Target sub-score
**`required_fields`** (expected: largest headroom — this is the dominant
failure mode the 4B exhibits). Will confirm via baseline.

## Hypothesis log
| iter | slug | family | composite | comp Δ | target Δ | verdict |
|------|------|--------|-----------|--------|----------|---------|
|      |      |        |           |        |          |         |

## Dead ends
(none yet)

## Open questions
- Does rollout-max_tokens=384 eliminate the 97% skip-rate ceiling that
  plagued capability #2, or is the structured-output issue tighter than
  that?

## Checkpoints
(every 3rd iter)
