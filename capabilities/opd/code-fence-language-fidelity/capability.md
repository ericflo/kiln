# Capability: code-fence-language-fidelity

## Description
When a coding agent is asked to produce a small code snippet, it must
emit the snippet inside a properly-fenced markdown code block with the
CORRECT language tag — `` ```python ``, `` ```rust ``, `` ```javascript ``,
etc. The 4B today often:
- omits the language tag (just `` ``` ``)
- uses the wrong language tag (`` ```python `` for JavaScript code)
- opens a fence and never closes it
- emits multiple fence pairs around what should be one
- adds prose around the code that wasn't asked for

This is a subtle capability — the underlying code is usually fine; the
WRAPPING is what fails. Important for any downstream tool that parses
fenced code blocks (notebooks, docs systems, agent harnesses).

## Base model
Qwen3.5-4B (kiln serve on http://localhost:8420)

## Teacher
`vllm/qwen3.6-27b-awq` at http://localhost:8002 (AWQ-INT4, max_logprobs=64)

## Rubric (designed with adversarial review applied upfront)

| Sub-score | Weight | What it measures |
|-----------|--------|-------------------|
| `fence_pair` | 0.10 | Response contains EXACTLY one well-formed fence pair: an opening `` ``` `` followed eventually by a closing `` ``` ``, both on their own lines, with no orphan fences. |
| `language_tag_correct` | 0.40 | **TARGET.** The opening fence tag matches the actual language of the inner code (verified by language-detection heuristics: keywords, syntax markers). |
| `code_parses` | 0.30 | The inner code is syntactically valid for the declared language (Python: `compile()`, JavaScript: rough Babel-style validation, Rust: presence of valid tokens). |
| `no_extra_text` | 0.20 | Response is the fenced block AND ONLY the fenced block. No prose preamble ("Here's the code:"), no trailing commentary. |

Composite = `0.10·fence_pair + 0.40·language_tag_correct + 0.30·code_parses + 0.20·no_extra_text`.
Direction: higher is better.

### Adversarial design (§0)

**Q: What's the cheapest way to score 1.0 on this rubric without doing the capability?**

1. **Always emit `` ```python ```` with the simplest valid Python (e.g. `pass`).** Passes fence_pair (1.0), passes code_parses (1.0), but if the user asked for JavaScript: fails language_tag_correct (0.0). Net composite ≤ 0.70. *Defended.*

2. **Emit empty fence block: `` ``` ``\n`` `` ``.** Passes fence_pair, but code_parses on empty is dubious — I'll make the rubric require non-trivial code (at least 1 non-whitespace token inside). *Defended by rubric design.*

3. **Emit prose-wrapped fence ("Here's the code: ```python ... ```").** Passes language_tag_correct + code_parses, but fence_pair is OK, but `no_extra_text` fails. Net ≤ 0.80. Same cap-#4 EOS-collapse-style defense.

4. **Emit very long padding around minimal code.** Same as #3 — no_extra_text catches it.

5. **Memorize eval prompts.** Firewall.

Worst remaining cheap-path floor: 0.70. Genuine capability requires hitting the target sub-score (40% weight) which requires language detection on the actual code.

## Baseline (filled after iter 0)
(blank — pending)

## Target sub-score
**`language_tag_correct`** (expected: largest headroom; the 4B's
natural language-tag failure rate is ~13%).

## Hypothesis log
| iter | slug | family | composite | comp Δ | target Δ | verdict |
|------|------|--------|-----------|--------|----------|---------|
| 0 (orig) | baseline | — | 0.8833 | — | — | over-strict rubric, fixed before iter 1 |
| 0 (fixed) | baseline | — | 0.9067 | — | — | corrected for js↔ts, c↔cpp, sh↔bash close-language pairs |
| 1 | h1-r8-lr5e5-2ep | H1 | **0.9300** | +2.33pp | +3.3pp | ✓ confirmed (kept; gentle dosage lifted cleanly) |

## Closeout (iter 1)

**Best adapter: `fence-h1-r8-lr5e5-2ep` at composite 0.93** (+2.33pp,
25% of headroom captured).

This is the positive counterpart to cap #5's high-baseline failure:

- Cap #5 (baseline 0.85, diff/patch): OPD regressed at every setting
- Cap #6 (baseline 0.91, code fence): OPD lifted +2.33pp

The condition that matters is **student rollout consistency**, not
composite value alone. Cap #5's rollouts varied between clean diffs
and malformed prose; OPD locked in the malformed ones. Cap #6's
rollouts are consistent (always a fence + tag, just sometimes wrong
tag); OPD's gradient on a consistent shape lifts cleanly.

Both are now evidence in the OPD skill's §0 — cap #6 is the positive
example validating the 'consistent rollouts' precondition.

Closing at iter 1: the gain is already real, and the +2.33pp/2-min
ROI is much better than chasing more by trying H2 or H9 variants.

## Dead ends
(none yet)

## Open questions
- Are the 4B's language-tag errors random or systematic (always-Python
  bias, etc.)? Will inspect at baseline.

## Wall-time budget per iter
**≤ 2 hours** per the cap #4 lesson. With 24 prompts × 1-2 epochs ×
~5-30s/effective step (depending on asymmetric or not), iters should
land in minutes-to-hours.

## Checkpoints
(every 3rd iter)
