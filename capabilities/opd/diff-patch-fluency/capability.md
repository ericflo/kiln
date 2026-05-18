# Capability: diff-patch-fluency

## Description
A coding agent that decides to modify a file via a unified diff must
emit a patch that `patch -p0` applies cleanly to the target. The 4B
today often produces patches with wrong hunk headers (off-by-N line
numbers), missing context lines, mis-formatted `@@ -A,B +A,B @@`
markers, or extra commentary around the diff that breaks the parser.
Downstream the agent harness rejects the patch.

Concrete failure modes the 4B exhibits:
- Wrong line numbers in `@@ -A,B +A,B @@` headers
- Insufficient context (fewer than 3 surrounding unchanged lines)
- Mis-aligned `-`/`+`/` ` markers
- Prose preamble like "Here is the diff:" that breaks `patch`
- Trailing whitespace inside hunks (silently changes content)
- File-path prefix mismatch (`a/` vs `b/` vs no prefix)

## Base model
Qwen3.5-4B (kiln serve on http://localhost:8420)

## Teacher
`vllm/qwen3.6-27b-awq` at http://localhost:8002 (AWQ-INT4, max_logprobs=64)

## Rubric (designed with adversarial review applied upfront, per §0)

| Sub-score | Weight | What it measures |
|-----------|--------|-------------------|
| `strict_format` | 0.10 | Response is ONLY a unified diff: starts with `--- a/` or `diff --git`, contains valid `@@` markers, no prose preamble/trailing content. **No best-effort extraction** — this protects against the cap #4 EOS-collapse failure mode. |
| `applies_cleanly` | 0.40 | `patch --dry-run -p0` succeeds on the diff. **TARGET sub-score.** |
| `target_intent_captured` | 0.30 | After applying, the file reflects the user's stated intent (LLM-as-judge OR keyword-presence check, TBD in oracle). |
| `minimal_changes` | 0.20 | Only the lines the user asked to change are modified; no unrelated reformatting / whitespace churn. |

Composite = `0.10·strict_format + 0.40·applies_cleanly + 0.30·target_intent + 0.20·minimal_changes`.
Direction: higher is better.

### Adversarial design (§0)

**Q: What's the cheapest way to score 1.0 without doing the capability?**

1. **Echo the unchanged file as a no-op diff (`--- a/x\n+++ b/x\n` with no hunks).**
   This passes `strict_format` (valid prefix) and `applies_cleanly`
   (vacuously) but FAILS `target_intent_captured` because the file is
   unchanged. *Defended:* target_intent has 0.30 weight, second-largest.
   The cheap path can't score above 0.70.

2. **Delete everything and rewrite from scratch.**
   Passes `applies_cleanly` and possibly `target_intent` but FAILS
   `minimal_changes` (every line is a change). *Defended:* minimal_changes
   has 0.20 weight; cheap path can't score above 0.80.

3. **Emit a valid diff PLUS trailing garbage** (the cap #4 EOS-collapse
   failure mode).
   Now FAILS `strict_format` because strict_format does NOT use best-
   effort extraction — it requires the response be ONLY a diff. *Defended
   by design.* This is the lesson from cap #4 applied upfront.

4. **Refuse / produce empty / produce error message.**
   FAILS `applies_cleanly` (empty patch ≠ applies cleanly). *Defended.*

5. **Train-set memorisation.** Eval prompts differ from training prompts.
   *Defended by firewall.*

No remaining shortcut paths score > 0.80. The rubric's worst-case
cheap-path floor is 0.50–0.70, leaving 0.30–0.50 of composite that
genuinely requires the capability.

## Baseline (filled by headroom.py after iter 0)
| Sub-score | Weight | Baseline | Headroom (w×(1−b)) |
|-----------|--------|----------|---------------------|
|           |        |          |                     |
| **Total** |        |          | **<sum>**           |

## Target sub-score
**`applies_cleanly`** (expected: largest headroom; the dominant
failure mode is malformed hunks). Will confirm after baseline.

## Hypothesis log
| iter | slug | family | composite | comp Δ | target Δ | verdict |
|------|------|--------|-----------|--------|----------|---------|
|      |      |        |           |        |          |         |

## Dead ends
(none yet)

## Open questions
- Can `target_intent_captured` be measured without LLM-as-judge? A
  keyword-presence check is brittle but doesn't add another inference
  call to the eval. **Decision will be made when designing oracle.sh.**
- Should asymmetric H9 be tried EARLY this time, dosed at 2 epochs
  rather than 6? The cap #4 lesson says H9 over-shoots at 6 epochs;
  2 might land in the sweet spot.

## Wall-time budget per iter
**≤ 2 hours.** Per the cap #4 lesson: hypothesis-author must
explicitly budget. With 26 prompts × 1 sample × N epochs × ~150s/step
(asymmetric pace), 2 epochs ≈ 2h, 6 epochs ≈ 6h. We target 2h
sessions to keep iteration fast; longer runs only when an iter
budget is explicitly approved.

## Checkpoints
(every 3rd iter)
