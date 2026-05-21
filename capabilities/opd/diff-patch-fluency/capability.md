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
| 2 | h1-r8-lr5e5-2ep-spp2 | H1 | 0.5217 (best ckpt) | -32.8pp | — | ✗ falsified (gentle settings avoided catastrophe but still regressed) |

## Dead ends
- H1 r16 2 epochs with mis-measured baseline → catastrophic LoRA
  collapse (-75pp).
- H1 r8 lr5e-5 spp2 with corrected baseline → still regression (-33pp).
  Demonstrates that gentler dosage doesn't fix the underlying problem.
- **The underlying problem: OPD has a high-baseline failure mode when
  student-teacher rollout overlap is variable.** At baseline 0.85,
  some student rollouts match teacher (OPD = no-op); some are
  malformed (OPD locks in malformedness, regression). Settings can't
  fix this asymmetry. Documented in skill §0 ("Where OPD shines").

## Open questions
- Could H6 (off-policy SFT cold-start on teacher rollouts) narrow
  the overlap gap enough for OPD to refine afterward? Out of scope
  this session (requires teacher-rollout generation pipeline).

## Closeout (iter 2)

**Best 'adapter' for this capability: NO ADAPTER.** The base 4B at
composite 0.85 is the ceiling reachable without a different training
paradigm. Two OPD iterations confirmed this:

- iter 1 (r16 lr1e-4): catastrophic 0.10
- iter 2 (r8 lr5e-5 spp2): gentler but still regressed to 0.52

Both confirm the §0 high-baseline failure mode finding: OPD on a
0.85 baseline with variable rollout quality can only regress.

Three rich findings from this capability, more valuable than any
adapter:

1. **Eval-design failure goes BOTH directions.** Cap #1 baseline
   0.99 (rubric too lax); cap #5 baseline 0.19 (rubric too strict).
   Corrected rubric jumped cap #5 baseline from 0.19 → 0.85.
   Skill §0 + Phase 0 step 8 now warn about both directions.

2. **OPD has a high-baseline failure mode** (the deep finding).
   At baseline > 0.80 with variable rollout quality, OPD regresses
   regardless of settings. Skill §0 "Where OPD shines" now reflects
   this with an explicit upper bound (0.80, not 0.95) and a sharper
   condition (consistent rollout quality, not just composite value).

3. **Trust the data — don't retire prematurely.** My first instinct
   was to retire cap #5 after the rubric fix at "baseline 0.85,
   headroom 0.15." The user correctly pointed out this was the cap
   #1 mistake replayed. Iter 2 produced the real finding (high-
   baseline failure mode). The retire-after-rubric-fix would have
   shipped the wrong meta-lesson.

If we ship a cap #5 adapter, it's the base model (no LoRA). The
real value is the findings backported to the skill.

## Wall-time budget per iter
**≤ 2 hours.** Per the cap #4 lesson: hypothesis-author must
explicitly budget. With 24 prompts × 1 sample × N epochs × ~150s/step
(asymmetric pace), 2 epochs ≈ 2h, 6 epochs ≈ 6h.

## Checkpoints
(every 3rd iter)


## Round 2 setup

This cap was normalized to the round-2 layout on 2026-05-21. The previous
iter log and writeups are preserved in [`archive/`](archive/). The
`capability.jsonl` starts empty for the new round.

### Kiln features the new round uses

- `kiln adapter verify` (#4) — adapter loadability + behavioral check.
- `cuda_*` trainer `--install-adapter-dir` / `--install-adapter-name` (#5) —
  atomic install into the registry; no more `output/adapter/` symlink bugs.
- `train_receipt.json` (#8) — the canonical per-run artifact with kiln SHA,
  data hashes, hyperparameters, LoRA delta norms, and ECHO metrics.
- `cuda_grpo_ablation --dry-run` (#9) — pre-GPU validation of data, masks,
  base-adapter shape, and saturated-reward warnings.
- `kiln trajectory inspect` (#10) — Rust-native mask + token-count
  diagnostic; replaces the Python `lib/pi_trajectory.py` for new code.
- ECHO observability in receipt (#12) — env-token CE, action-token count,
  warning-prefix masked-out byte count.
- `kiln serve --eval-mode` (#15) — deterministic, no thinking, no
  per-request adapter drift.
- `--adapter-smoke-test` (#19) — post-train base-vs-adapter logit-delta check.
- `--filter-var-min` (#22) — official strong-signal filtering.
- `kiln eval-adapter --seeds N` (#33) — multi-seed paired-eval driver wrapped
  by `capability.oracle.sh`.
- `adapter_manifest.json` + `kiln adapter restore` (#36) — replaces ad-hoc B2
  backup scripts.

### Workflow

```bash
./capability.oracle.sh                     # baseline (no adapter)
./run_iter.sh h1-default-recipe            # first training iter
./run_iter.sh h2-lower-lr                  # subsequent
```

See [`run_iter.sh`](run_iter.sh) for the full pipeline.
