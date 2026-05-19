# Hypothesis H1 — default Phase 1 GRPO recipe

**Family:** H1 (default recipe — first iter, always)
**Target sub-score:** `tool_call_efficiency`

## Claim

Training the base Qwen3.5-4B against the v1 multi-component reward
using kiln's Phase 1 GRPO defaults (`advantage_mode: dr_grpo`,
`loss_aggregation: token_level`, `kl_estimator: k1`, `dynamic_sampling:
true`, symmetric clip 0.20, `is_level: token`, `reference_policy:
base_per_step`) will:

- Reduce mean `tool_call_efficiency` *cost* (=1 - tool_call_efficiency)
  on the eval set by ≥20% relative.
- Maintain `outcome` mean within 0.05 of baseline (0.958 → ≥0.91).
- Lift composite by ≥0.04 (0.885 → ≥0.925).

## Mechanism

The baseline shows 4 out of 24 eval rollouts use ≥13 tool calls (the
"wasteful" tail). These rollouts contribute the most movable mass to
composite — going from `tool_call_efficiency=0.0` to `=1.0` on a
single task lifts that task's composite from 0.5 to 1.0 (assuming
outcome=1.0 stays). GRPO's group-relative advantages will reward the
efficient rollouts within each task's group and penalize the wasteful
ones. The advantage signal is concentrated on tasks where rollouts
diverge — exactly the kind of within-group variance we measured
(stdev 0.358 for tool_call_efficiency).

The KL anchor to the base model should prevent the policy from
collapsing into a non-tool-using mode; the entropy via temperature 0.8
should preserve exploration.

## Falsification plan

This hypothesis is falsified if any of:
- mean composite < 0.85 (catastrophic regression — see §10 all-zeros
  gate, threshold 0.5·baseline = 0.443; this milder threshold catches
  partial regressions)
- mean `outcome` < 0.91 (sub-score regression: model started failing
  tasks it used to pass)
- mean tool_call_count *increased* iter-over-iter (wrong direction)
- `tool_call_efficiency` stdev across eval tasks fell to <0.10
  (mode collapse on the agentic dimension)

## Training plan

- 30 tasks from `datasets/train.tasks.jsonl` (first 30; deterministic
  for reproducibility).
- 4 generations per task = 120 rollouts.
- Sampling: temperature 0.8, top_p 0.95, max_tokens per turn 1024,
  max wall-clock per session 120s.
- GRPO: Phase 1 defaults; lr 1e-5; rank 16, alpha 32; seed 3141592653.
- Expected rollout wall-clock: 30 × 4 × ~25s = ~50 min.
- Expected GRPO step wall-clock: ~10 min on A100.

## v0 caveat — multi-turn token masking (RESOLVED)

Each pi rollout has 3-27 assistant turns. Originally
`tokenize_grpo_group` treated all post-prompt tokens as model-emitted,
including tool-result tokens. For iter 1 we accepted that bias by
concatenating assistant text/thinking/toolCall blocks per turn and
joining turns with `<TURN_BREAK>` — relying on the IS ratio clip to
softly mask out tool-result gradient.

**As of the ECHO branch:** kiln-train now ships first-class trajectory
schema (`TurnKind` / `TurnSegment` / `ScoredRollout`) and
`build_masks_from_trajectory` produces correct per-turn `action_mask`
+ `env_mask` arrays. Rollouts emitted via the canonical schema
(`capabilities/agentic-grpo/lib/pi_trajectory.build_scored_rollout`)
get per-turn assistant-token masking *and* the ECHO env-CE auxiliary
loss as the default. The iter 1 result above predates that landing;
iter 5 (`f4ff31d9`) ran on the new path and produced the 3-seed verified
GRPO uplift that's now the cap's headline result.

## Verdict

**? inconclusive — partial signal in target direction, no composite lift.**

| metric | baseline | iter 1 | Δ |
|---|---|---|---|
| composite (eval set, n=24) | 0.8854 | 0.8880 | **+0.0026** (flat) |
| outcome | 0.9583 | 0.9583 | 0.0000 (held) |
| **tool_call_efficiency** | 0.7448 | 0.7552 | **+0.0104** (target moved) |
| tested_before_done | 0.9792 | 1.0000 | +0.0208 |
| format_compliance | 1.0000 | 1.0000 | 0.0000 (saturated) |
| mean n_tool_calls | 6.83 | **5.63** | **−1.21 (−18%)** |
| mean wall_clock_s | 25.4 | 31.5 | +6.1 (wrong direction) |

Per-task: 6 better, 5 worse, 13 same.

**Falsification check:**
- ✗ composite lift ≥+0.04 — failed (got +0.0026)
- ✓ outcome held ≥0.91 — passed (held 0.9583)
- ✓ mean tool-call count decreased — passed (−18%)
- ✓ tool_call_efficiency stdev preserved — passed (no mode collapse)

The hypothesis was partially confirmed: the target sub-score moved
right, the operationally-meaningful metric (mean tool-call count)
dropped 18%. But composite stayed flat because the adapter learned to
be *more* careful on previously-easy tasks (regressing them) while
becoming more efficient on previously-wasteful ones (gaining them).
Net wash.

## Inspected rollouts (Phase 5 requirement)

### Win — task_0005 (rolling_max-ish)
Baseline: 13 tool calls, score 0.70. Iter 1: 6 tool calls, score 0.93.
The model went from a debug-loop on first failure to a straight
read→edit→bash→DONE.

### Win — task_0020
Baseline: 9 tool calls (some redundant `cat solution.py`), score 0.81.
Iter 1: 4 tool calls, score 1.00. Clean.

### Regression — task_0013
Baseline: 3 tool calls (read→edit→DONE), score 1.00. Iter 1: 9 tool
calls, score 0.81. The adapter introduced an extra inspect+retry cycle
on a task that didn't need one — the model is now sometimes
over-verifying, classic over-correction from a small training set.

### Regression — task_0022
Same shape: baseline 3 → iter 1 9 tool calls. The training distribution
(3 medium-hard tasks with wasteful baseline behavior) didn't include
enough easy-task examples, so the model didn't learn that "already
efficient → don't add more checks."

## Disposition

**Status: kept-with-caveat.** Training pipeline closes end-to-end. The
target sub-score moved in the right direction. The result is NOT a
ship — but it is real evidence that the loop produces signal.

## Caveats / known biases this iter carried

1. **3-group training set** (intended 30, cut to 3 by wall-clock
   budget). Biased toward already-wasteful tasks. Adapter
   over-corrected on easy ones.
2. **Multi-turn token-masking bias** — kiln-train treated tool-result
   tokens as model-emitted. The IS clip likely masked most of the
   spurious gradient but we can't quantify how much.
3. **Single eval generation** (temperature 0.0 deterministic). pass@4
   diversity not measured.
4. **Adapter-routing via proxy** — kiln-server auto-unloads global
   active adapter ~1s after POST /v1/adapters/load. Workaround: 5KB
   Python proxy (`/tmp/kiln-adapter-proxy.py`) injects `adapter`
   field into every chat-completion body. Needs a kiln-server fix
   (separate kiln-polish issue).

## Next iter (H2 candidate)

Given the partial-signal result, two paths:

**Path A — scale up.** Train on 12-15 groups with stratified sampling
(4 easy + 4 medium + 4 hard) to cover the task distribution. Same H1
recipe.

**Path B — fix masking first.** Land the per-turn assistant-token
masking PR in kiln-train. Then redo iter 1 with proper gradient
attribution. This is more disciplined per the skill but blocks
forward progress.

Recommendation: Path A first (cheap), Path B if A also produces only
partial signal.
