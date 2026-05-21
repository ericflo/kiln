# GRPO mode — irreducible lore

Read this when [`capabilities/METHODS.md`](../../../../capabilities/METHODS.md)
routes a stage to **GRPO** (Rule F single-turn, or Rule G ECHO-only on
saturated reward). For agentic (multi-turn) tasks see `agentic-grpo-mode.md`.

`SKILL.md` covers the universal loop; this file covers only what's specific
to GRPO and not in METHODS.md / PIPELINE.md / NEXT_ROUND.md.

## When GRPO is the right tool

Reach for GRPO when **all** of these are true:

1. There exists a programmatic reward function that, given a complete
   response, returns a score in `[0, 1]`.
2. That reward function is **stable** under perturbations of input surface
   form — paraphrases of the same correct answer get similar scores.
3. The base model produces *some* signal — not pure noise — on the
   training distribution. (GRPO needs reward variance within groups;
   if every rollout scores 0.0 you have no advantage signal and dynamic
   sampling drops every group.)
4. The capability has a tractable **rollout token budget**. GRPO trains on
   the model's own outputs, so one training step ≈ `N × max_tokens` tokens
   per prompt. On A6000 with rank 16, ~6000 tokens/group is the practical ceiling.

Reach for **OPD instead** when (1) is hard but a teacher exists. Reach for
**SFT instead** when you can write ~200 ground-truth pairs.

## GRPO ≠ "magic free improvement"

GRPO is policy gradient on top of your reward function. **Everything the
reward function rewards, the model learns.**

- If your reward rewards "matches a regex," the model converges to "shortest
  string matching that regex."
- If your reward rewards a multi-component composite where one component is
  `length_band` with positive weight, the model converges to "the length
  the band centers on, no matter what's inside."

**The reward function is load-bearing.** Adversarial design (§0 in
`capability.md`) is mandatory before writing rubric.py.

## Published GRPO failure modes

All are reward-function failure modes, not method bugs:

- **Length drift** — DAPO §2 (arXiv:2503.14476). Per-sample averaging
  under-penalises long wrong outputs.
- **Mode collapse** — Magistral (arXiv:2506.10910). Symmetric clip plus
  KL anchor against base model collapses exploration.
- **Entropy collapse** — Cui et al. (arXiv:2506.01939). All tokens
  contribute KL, including low-uncertainty ones, killing exploratory tokens.

Kiln's Phase 1 defaults (`DrGrpo` + `TokenLevel` + `dynamic_sampling`)
mitigate length drift and entropy collapse. Mode collapse mitigation is
the KL coefficient (`--kl-coeff 0.1` default) and `--clip-epsilon 0.20`.

## Hyperparameter defaults

- mode=phase1, advantage=dr_grpo, loss=token_level, kl_estimator=k1
- kl_coeff=0.1, clip_epsilon=0.20, dynamic_sampling=true
- lr=1e-5, rank=16, alpha=32, seed=3141592653
- `--filter-var-min 0.05` (strong-signal filter, kiln #22)
- `--adapter-smoke-test` (kiln #19)
- num_generations=4 per prompt, max_tokens=1024 per response

## GRPO-specific failure modes

### All-zero rewards

If the base produces rollouts that all score 0.0, every group has zero
variance and dynamic sampling drops them all. Trainer fires `data_schema_error`
or `zero_action_tokens` with an empty groups_trained count.

**Mitigation:**
- Inspect 5-10 base responses. If they're actually trying but the rubric
  rejects them, the rubric is over-strict (round-1 OPD #5 case).
- If they aren't trying, SFT bootstrap first (METHODS.md Rule C).

### All-1.0 rewards (saturation)

Rewards are saturated; variance is < 0.03; dynamic sampling drops them
all. Trainer fires `reward_saturation_warning`.

**Mitigation:**
- Build `hard_eval.tasks.jsonl` from failures-derived prompts where base
  composite < 0.5; switch eval set.
- Or use `--no-policy-loss` (ECHO-only) — METHODS.md Rule G.

### Length drift

Outputs get longer (or shorter) without improving content. Watch:
`train_receipt.json::action_token_count` over iters. If it climbs faster
than composite, length is the lift mechanism.

**Mitigation:** add a length-penalty sub-score, OR use Phase 1's DrGrpo
loss (already the default; it normalizes per-token not per-sample).

### Mode collapse

Diversity of outputs drops; same response shape on every prompt. Watch:
`train_receipt.json::group_variance_histogram` — if variance concentrates
near 0, the model is producing identical-scored responses.

**Mitigation:** higher KL coefficient (more anchor to base), or lower lr,
or higher rank.

### Entropy collapse

Specific tokens lose entropy entirely. Watch logit distributions on a fixed
prompt across iters. If the policy becomes near-deterministic everywhere,
entropy collapsed.

**Mitigation:** kiln's `TokenLevel` loss aggregation (already default)
limits per-token KL contribution; or reduce clip_epsilon.

## Group statistics watch

For every iter, inspect:

- `groups_seen` (total prompts × num_generations)
- `groups_filtered` (dropped by `--filter-var-min`)
- `groups_trained` (actually contributed to gradient)
- `reward_mean`, `reward_stdev`, `reward_min`, `reward_max`
- `group_variance_histogram`

Red flags:

- `groups_trained` < 50% of `groups_seen` → too many low-variance groups.
  Raise `--filter-var-min` only if you want strong signal; lower it if
  you're filtering away useful samples.
- `reward_stdev < 0.1` on training set → no signal. Either base is
  saturated (use OPD/SFT instead) or the rubric is too coarse.
- All-zero `reward_min` AND all-1.0 `reward_max` → split distribution
  with no middle. Often the rubric is binary-classifying. Useful but
  GRPO can't shape it; consider OPD or rubric redesign.

## Strong-signal filtering (`--filter-var-min`)

Round 1 found this is the single most reproducibly useful knob beyond
the defaults. `--filter-var-min 0.05` keeps only groups with reward
variance above the threshold.

Empty-filter behavior:

- `--on-empty-filter fail` (default): trainer exits non-zero if zero
  groups remain.
- `--on-empty-filter train-all`: ignore the filter when it would empty
  the training set.
- `--on-empty-filter skip`: write a "skipped" receipt and exit zero.

Sidecar JSON records the exact kept/dropped group ids so iter-to-iter
reproduction is exact.

## Receipt fields specific to GRPO

- `groups_seen`, `groups_filtered`, `groups_trained`
- `reward_mean`, `reward_stdev`, `reward_min`, `reward_max`
- `group_variance_histogram`
- `filter_var_min`, `on_empty_filter`
- `kl_coeff`, `clip_epsilon`, `dynamic_sampling`, `is_level`
- `lora_delta_norm_summary`, `grad_norm_min_mean_max`
- `reward_saturation_warning` (boolean)
- `length_drift_warning` (boolean; added by kiln #21)
- No `epochs` field — GRPO is online; no fixed-pass counting.

## Reward design discipline (specific to GRPO)

Pre-flight checklist before any GRPO stage:

1. **§0 adversarial design.** Name ≥3 cheats; design rubric mitigations.
2. **Multiplicative format gate** if format matters: `composite = outcome ×
   format × (process + base)`. Round-1 +12.5pp format gains got trapped in
   additive composites.
3. **Calibration.** Good fixtures score > 0.7; bad < 0.3. Margin > 0.2.
4. **Baseline distribution check.** If > 0.95, harden rubric first.
5. **Reward variance check.** Sample 20 base rollouts; group variance > 0.05?
   If not, GRPO has no signal — use OPD or SFT.

## Stage transitions FROM GRPO

- **GRPO → GRPO warm-best:** single-seed lift was high but stdev was large.
  Chain best adapter as base, re-train fresh seed. If 3-seed mean matches,
  recipe is robust (round-1 pi-doctest pattern).
- **GRPO → OPD:** rare; only if a new teacher becomes available. OPD on top
  of GRPO often undoes the sharpening.
- **GRPO → STOP:** reward_saturation_warning fires AND hard_eval also
  saturated; or composite within σ for 2 consecutive iters.

## Stage transitions TO GRPO

- **none → GRPO:** baseline ∈ [0.6, 0.9], verifier, reward variance > 0.05
  (METHODS.md Rule F).
- **SFT → GRPO:** SFT closed with format ≥ 0.7 AND verifier AND reward
  variance on SFT-adapter rollouts > 0.05.
- **OPD → GRPO:** OPD closed with teacher gap saturated AND hard_eval has
  > 0.05 headroom.

## References

- `caps/pi-doctest/` — agentic-GRPO reference (overlap with agentic-grpo-mode)
- `caps/pi-failure-triage/` — round-1 saturated-reward case; reshaped with
  multiplicative format gate
- `caps/pi-diff-patch-apply/` — round-1 GRPO-harm-vector case; reshaped
- Papers: DAPO arXiv:2503.14476, Magistral arXiv:2506.10910,
  Cui et al. arXiv:2506.01939
