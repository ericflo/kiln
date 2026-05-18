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

## Baseline (corrected after rubric fix)
| Sub-score | Weight | Baseline | Headroom (w×(1−b)) |
|-----------|--------|----------|---------------------|
| strict_format | 0.10 | 1.0000 | 0.0000 |
| applies_cleanly | 0.40 | 0.8667 | 0.0533 |
| target_intent_captured | 0.30 | 0.7667 | 0.0700 |
| minimal_changes | 0.20 | 0.8667 | 0.0267 |
| **Total** | | **0.8500** | **0.1500** |

## Target sub-score
**`target_intent_captured`** (47% of movable headroom). The 4B's
diff-emission is mostly correct; the failure mode is producing
applying-but-not-quite-right diffs (wrong line replaced, etc.).

## Hypothesis log
| iter | slug | family | composite | comp Δ | target Δ | verdict |
|------|------|--------|-----------|--------|----------|---------|
| 0 (orig) | baseline | — | 0.1883 | — | — | (wrong — over-strict rubric rejected fenced diffs) |
| 0 (fixed) | baseline | — | **0.8500** | — | — | (corrected; rubric now accepts ```diff fenced output) |
| 1 | h1-r16-2ep | H1 | 0.0983 | -75.2pp | — | ✗ CATASTROPHIC (8 noisy steps destroyed the LoRA) |

## Dead ends
- H1 r16 2 epochs with mis-measured baseline → catastrophic LoRA
  collapse. The training itself isn't the dead-end; the upstream
  eval-design failure that made an aggressive gradient signal seem
  warranted is the dead end.

## Open questions
- With baseline now 0.85 and headroom only 0.15, is OPD worth running
  at all on this capability? The 4B is already quite good at diffs.
  A perfect-ceiling adapter would lift composite by at most 0.15pp.

## Closeout (iter 1)

**Capability retired here. Best kept artifact: the eval-design lesson.**

Two iterations:
- iter 0 baseline (orig rubric): 0.1883 — *wrong* (rubric too strict)
- iter 0 baseline (fixed rubric): 0.8500 — *the real baseline*
- iter 1: 0.0983 — catastrophic LoRA, OPD destroyed a 0.85 model

The valuable artifact is the symmetric of cap #1's mistake:

- **Cap #1** baseline 0.99: rubric too LAX (model already passes; eval
  too easy). Skill §0 warns about this. Prior session abandoned cap
  #1 — wrong call; correct disposition was "harden the rubric and re-
  baseline."
- **Cap #5** baseline 0.19 (initial): rubric too STRICT (model produces
  valid diffs in fenced format; rubric rejects them). Inspecting 5
  responses showed the 4B is actually competent. Correcting the rubric
  jumped baseline to 0.85. The capability is mostly solved; OPD is
  borderline-interesting given the small remaining headroom.

The skill needs §0 to warn about BOTH directions of eval-design failure:
- baseline ≥ 0.95: too lax. Inspect responses; harden rubric.
- baseline < 0.30 on a capability the 4B *should* have some seed-form
  ability at: too strict. Inspect responses; loosen rubric (without
  re-opening Goodhart holes).

Both lessons are now backported to the skill (this session).

## Wall-time budget per iter
**≤ 2 hours.** Per the cap #4 lesson: hypothesis-author must
explicitly budget. With 24 prompts × 1 sample × N epochs × ~150s/step
(asymmetric pace), 2 epochs ≈ 2h, 6 epochs ≈ 6h.

## Checkpoints
(every 3rd iter)
