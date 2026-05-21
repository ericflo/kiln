# Capability: pi-incremental-progress

**Status:** New in round 2. Scaffold.

## Description

When making a non-trivial change, the agent should work in **verified
sub-steps** rather than writing everything then testing once. The 4B's
default behavior:

1. Reads the task.
2. Writes the entire solution in one tool call.
3. Runs tests.
4. If tests fail: starts over or makes large random edits.

The capable behavior:

1. Reads the task; decomposes into 2-4 verifiable sub-steps.
2. Writes step 1; runs the partial test.
3. Sees step 1 works; writes step 2; runs again.
4. Continues until all sub-steps verified.
5. Final assistant turn summarizes the verified path.

The behavioral difference is dramatic on tasks of moderate complexity
(refactor across 3 files, multi-function changes, anything where a
failure mid-flight is informative). Round 1's `pi-faithful-completion`
hinted at this — agents that decompose ship cleaner.

## Base model

Qwen3.5-4B (kiln serve on `http://localhost:8420`).

## Rollout source

Pi sessions. Multi-turn (max 12 turns; this cap inherently needs more
turns than a single-step task).

## Task shape

Each task is a `(workspace, complex_change_request, decomposition,
gold)`:

- **Workspace** — repo with 3-5 files relevant to the change.
- **Complex change request** — natural-language; requires touching ≥2
  files OR ≥2 functions in one file.
- **Decomposition** — ground truth: an ordered list of verifiable
  sub-steps that *would* yield a correct solution. Used by the rubric
  to measure the agent's actual path.
- **Gold** — final-state of the workspace.

Example task:

> Refactor the `lib/cache.py` module: extract the in-memory cache into
> a separate `lib/memory_cache.py`, make `lib/cache.py` import from
> both `memory_cache.py` and the existing `disk_cache.py`, and update
> the tests in `tests/test_cache.py` to mock the new module structure.

Gold decomposition:

1. Create `lib/memory_cache.py` with the extracted class.
2. Verify it imports and the test for the in-memory class passes.
3. Update `lib/cache.py` to import from both backends.
4. Verify both backends import.
5. Update `tests/test_cache.py` mocks.
6. Run full test suite.

A capable agent walks 1→6 with a test invocation between each. The
4B's default is to write all three files in one big edit, then test
and panic when test 3/6 fails.

## Rubric (v0)

Multi-component with multiplicative format gate.

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor (multiplicative) | Final workspace matches gold; all tests pass. | — |
| `format_compliance` | multiplicative gate | Final assistant turn lists the sub-steps taken in order. Each step references a tool call that happened. | Listing fictional steps without evidence. Rubric cross-checks against the session. |
| `step_progress_observability` | 0.30 | Number of *intermediate* test/verify calls (between the first edit and the final edit). Score = `min(1, intermediate_verifies / 2)`. | Spamming `pytest -q` between edits without making meaningful progress — rubric requires the verify to be on different state. |
| `step_alignment_with_decomposition` | 0.20 | The agent's actual sequence of (edit, verify, edit, verify, …) maps onto the gold decomposition. Token-level alignment between the agent's progression and the gold-progression. | Random progression — alignment scores low. |
| `early_failure_caught` | 0.20 | If any intermediate verify fails, does the next assistant turn address that specific failure (not skip past it)? Sub-score for "the agent attended to mid-flight failures." | Always making intermediate verifies pass by cheating with shallow tests — outcome on the held-out final verify catches this. |
| `no_big_bang` | 0.15 | First edit doesn't touch >50% of the gold-touched files. The agent shouldn't write everything in one tool call. | First edit being trivially small but containing nothing useful — alignment sub-score punishes empty first edits. |
| `base` | 0.15 | — | — |

**Composite = `outcome × format_compliance × (0.30·step_progress_observability + 0.20·step_alignment_with_decomposition + 0.20·early_failure_caught + 0.15·no_big_bang + 0.15)`**

## Adversarial design (§0)

**Q: What's the cheapest way to score 1.0 without doing the capability?**

A1: Spam `pytest` between random edits — wins
   `step_progress_observability` cheaply.

   Mitigation: alignment sub-score requires that the verifies are
   *between meaningful state changes*, and that the edit→verify
   pattern progresses toward gold.

A2: Make tiny throwaway first edits to avoid `no_big_bang` penalty,
   then write everything else in one big edit.

   Mitigation: `step_alignment_with_decomposition` measures the
   progression across the whole session, not just the first edit.
   First-edit cheats then second-edit-everything still scores low
   alignment.

A3: Memorize the gold sequence for common refactors. Tasks are drawn
   from a small pool; this is a real risk.

   Mitigation: task corpus is sampled fresh from real OSS refactor
   commits, and the eval pool is held-out and never trained on.

A4: Always emit "I'll do this in 4 steps" in the final summary,
   regardless of actual behavior — wins format_compliance.

   Mitigation: format sub-score cross-checks the claimed steps
   against the session. Mismatch → format_compliance 0.

## Headroom estimate

- Baseline composite: **~0.50** (4B sometimes decomposes, often
  doesn't).
- Headroom: ~0.50.
- Target sub-score: `step_progress_observability` (most movable).

## Hypotheses

- **H1: default GRPO recipe.**
- **H2: extended turn budget.** Default max_turns=8 is too tight for
  this cap; bump to 12. Hypothesis: more headroom for the model to
  show the decomposition behavior.
- **H3: ECHO-heavier λ=0.075.** This cap depends heavily on
  intermediate test results; stronger env-attention should help.
- **H4: chain from pi-faithful-completion best.** Faithful-completion
  already lifted terminal-state discipline; this cap is the "process"
  analog.

## Composition with other caps

- **Upstream:** None — this cap is foundational behavior.
- **Downstream:** Almost all other caps benefit from incremental
  progress as a habit.
- **Integration test:** central member of
  `integration/cross-cap-coherence/` — this is one of the behaviors
  most likely to compose well across caps.

## Round 2 standard workflow

```bash
python3 build_corpus.py             # Mine refactor commits from OSS repos
./capability.oracle.sh              # Baseline (expect ~0.50)
./run_iter.sh h1-default-recipe     # H1
./run_iter.sh h2-extended-turns     # H2 (MAX_TURNS=12)
```

See `../../LAYOUT.md` and `../README.md` for shared infrastructure.
