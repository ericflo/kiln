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

## v0 caveat — multi-turn token masking

Each pi rollout has 3-27 assistant turns. Kiln's current
`tokenize_grpo_group` treats all post-prompt tokens as model-emitted,
including tool-result tokens. For iter 1 we accept the bias: we
concatenate assistant text/thinking/toolCall blocks per turn and join
turns with `<TURN_BREAK>`. The IS ratio clip may mask out tool-result
gradient, but we don't rely on it.

If iter 1 results are clean: the bias was tolerable.
If iter 1 results are nonsense: the masking gap is the cause and the
right next step is landing the fix to `tokenize_grpo_group` (see
`kiln-polish-prerequisites.md` #1).

## Verdict

*Filled in Phase 5, after eval lands.*

## Inspected rollouts (Phase 5 requirement)

*Three excerpts from the eval rollouts pasted here after iter 1
completes — to verify the model is doing what we hoped, not gaming
the rubric.*
