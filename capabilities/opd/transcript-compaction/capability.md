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
| 0 | baseline | — | 0.8098 | — | — | (re-scored after rubric fix) |
| 1 | h1-r16-6ep | H1 | 0.8415 | +3.17pp | -0.39pp | ? inconclusive — confound: length not entity |
| 2 | h1-r16-6ep-tok256 | H1 | 0.8632 | +5.34pp | +3.46pp | ✓ confirmed — target moved |

## Dead ends
- max_tokens=128 rollout budget: the OPD loss landscape's lowest-cost
  reduction is length compression, not capability uplift. Don't use
  max_tokens narrower than 1.5–2x natural response length.

## Open questions
- Does the H1 epoch curve continue past 6 at max_tokens=256? (would test
  at 12 epochs if we continued iter 3.)
- Could H6 (SFT cold-start on teacher rollouts) reduce the 95% skip rate
  by warming the student toward teacher-compatible rollouts?

## Closeout (iter 2)
Closing at iter 2 with composite 0.8632 (+5.34pp, 28% theoretical
headroom captured). Decision rationale: the broader session goal is
exercising the OPD skill across multiple coding-agent capabilities,
not maximizing this one. Three iterations have:
- produced a working capability LoRA (entity-dense compactions),
- uncovered a kiln finding (rollout max_tokens choice has outsize
  effect on which sub-score moves),
- exercised the rubric-design discipline (rubric was wrong twice; both
  caught and fixed; this fed the new Phase 0 calibration gate),
- exercised the verdict-gate discipline (1 inconclusive, 1 confirmed),
- exercised the failure_mode discipline (capability #2 close-out
  produced the chat-template-render diagnosis).

The marginal value of iter 3 here is lower than starting capability #4
(tool-call argument fidelity). Adapter retained at
`/workspace/kiln/Qwen3.5-4B/adapters/compact-h1-r16-6ep-tok256`.

## Checkpoints
- iter 3 not run; closed at iter 2.


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

## Round 2 improvement plan
Round 1 status: **OPD cap; multiple iters attempted**.

### Round 2 plan

1. **Run after `code-symbol-extraction` canary.**
2. **Pair with `pi-compaction` decision.** Once pi-compaction's
   kiln-bench gates resolve (long-context weight movement), this OPD
   cap and pi-compaction (now OPD per round-2 reshape) are
   complementary — different task framings of the same underlying
   summarization skill. They should share an eval set.
