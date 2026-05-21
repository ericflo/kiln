# OPD mode — irreducible lore

Read this when [`capabilities/METHODS.md`](../../../../capabilities/METHODS.md)
routes a stage to **OPD** (Rule E, or as polish on top of a stage-1 SFT).

`SKILL.md` covers the universal loop; this file covers only what's specific
to OPD and not in METHODS.md / PIPELINE.md / NEXT_ROUND.md.

## When OPD is the right tool

- Baseline composite ∈ [0.4, 0.8].
- Teacher available (≥30% stronger composite on this rubric, same family
  14-32B).
- Headroom concentrated in process/format sub-scores (not pure outcome —
  outcome is "knowledge" and OPD doesn't add new knowledge).
- Student already produces in-shape attempts (it can fail at the capability
  but it should at least *try*).

## When OPD is NOT the right tool

- **Baseline > 0.80 with variable rollout quality.** This is the cap #5
  failure mode (next §). OPD will regress, sometimes catastrophically.
- The student doesn't have the basic capability shape (use SFT bootstrap).
- The rubric is already saturated (no headroom to capture).
- The student samples in distributions the teacher considers junk
  (no overlap → no gradient signal).

## The high-baseline failure mode (cap #5, diff-patch-fluency)

**The single most important OPD failure pattern.** Read this twice.

At baseline > 0.80 with variable rollout quality:
- Some student rollouts match the teacher (OPD step ≈ no-op).
- Some student rollouts are the failure-tail's malformed outputs.
- Reverse-KL makes the student MORE confident in WHATEVER it sampled,
  including the malformed rollouts.

The asymmetry kills you: good-rollout steps don't help (already correct),
bad-rollout steps actively regress.

**Round-1 cap #5 evidence:**
- Baseline 0.85
- Iter 1 r16 lr1e-4: catastrophic 0.10
- Iter 2 r8 lr5e-5 spp2: still regressed to 0.52 best-of-checkpoints

Gentler settings cannot fix it. The right tool is:
- **SFT cold-start on teacher rollouts** (METHODS.md §4.3), OR
- **Accept the baseline as the ceiling** and ship base.

`lib/method_router.py` routes baseline > 0.80 caps to SFT-rescue, not OPD.
If you override this routing, you must document the override in
`pipeline.md::stage_transition_rationale` with strong evidence (e.g. a
preliminary test on 8 prompts showing OPD steps are stable).

## Hyperparameter defaults

- rank=16, alpha=32, lr=1e-4, 4-6 epochs, samples_per_prompt=1-2.
- Teacher: vLLM on `:8002`, Q4 of 14-32B model in same family.
- `--adapter-smoke-test` on (kiln #19).

## OPD-specific failure modes

### Skip-rate domination (round-1 code-symbol-extraction)

OPD effectively trains only on tokens where the student's distribution
disagrees enough with the teacher's. EOS tokens often "skip" because both
agree the response should end.

Round-1 `code-symbol-extraction` showed **97% EOS skip rate** — only 3% of
tokens contributed gradient. Trainer reports this now (kiln #37 receipt fields).

**Watch:** `train_receipt.json::effective_steps` and
`train_receipt.json::skip_rate`. If skip_rate > 0.50, OPD has too few
tokens to learn from. Increase teacher temperature, vary prompts more,
or switch to SFT.

### Loss is deceptive

OPD loss is spiky by nature — perfect agreement on some states, large
disagreement on others. Trust the blind eval, never the loss curve.

**Common mistake:** declaring success because loss dropped. Loss dropping
just means the easy tokens got easier; it tells you nothing about the
capability. The blind eval is the only trustworthy signal.

### Teacher hosting

OPD requires a *live teacher server*, unlike SFT (teacher is a file).
Standing it up and cleaning up after it is part of the loop.

```bash
# Typical teacher: 27B Qwen3.6 AWQ Q4 on a separate port
vllm serve qwen3.6-27b-awq --port 8002 --quantization awq --tensor-parallel-size 1 &

# Health-check before training
curl -sf http://localhost:8002/v1/models > /dev/null || exit 1

# Cleanup after training
kill %1
```

`capability.config.json::methods.opd.teacher_url` and `.teacher_name`
record what to start. `run_stage_opd.sh` health-checks before training.

### Sub-score regression

A composite lift from OPD can mask a sub-score regression. Round-1 cap #3
(transcript compaction) +3.17pp composite was actually `length_band`
+29.7pp confounding the target `entity_recall` (which stayed flat).

**Watch:** every sub-score in `eval_summary.json::sub_scores_mean`. If the
target sub-score didn't move but composite did, the lift is a confound.
Re-weight the rubric or accept the regression as a dead-end.

## Receipt fields specific to OPD

- `n_prompts`, `effective_steps` (TRUE step count after EOS-skip filtering)
- `teacher_calls_made`, `teacher_health_at_start`, `teacher_health_at_end`
- `skip_rate` (added by kiln #37)
- `loss_curve_final` is informative but spiky; eval is canonical
- No `groups_*`, no `echo_*`, no `reward_*`

## Anti-shortcut design (specific to OPD)

OPD is particularly prone to length / shape shortcuts because the teacher's
tokens are the gradient target.

- **Length compression:** student learns "produce fewer tokens" because
  fewer tokens = less divergence on average. Pair length-relevant rubric
  components with content-density / entity-presence sub-scores.
- **Output-shape confusion:** student emits teacher's typical shape even on
  prompts where the shape is wrong. A "summary vs source" rubric that scores
  keys but not surrounding structure rewards confusion.
- **Format compliance over content:** student produces correctly-formatted
  nothing-of-substance. Pair format sub-scores with content sub-scores.

The §0 adversarial design in `capability.md` should name these explicitly
and design the rubric to block them.

## Sanity-check the student periodically

After every 2-3 OPD iters, sample 5-10 student responses by hand against
the *plain-English description in capability.md* — not the rubric. If the
responses are formally correct but feel wrong, the rubric is rewarding
the wrong thing. Edit the rubric, re-baseline, continue.

Round-1 cap #3 had two such rubric revisions mid-session. Each fix
improved signal. The error would have been *not* fixing them.

## Stage transitions FROM OPD

- **OPD → GRPO:** teacher gap closes (env_ce delta < -0.3 saturated) AND
  hard_eval has > 0.05 headroom. See METHODS.md §4.2.
- **OPD → STOP:** composite within σ for 2 consecutive iters; sub-scores
  stable; teacher CE saturated.
- **OPD → SFT (rescue):** OPD regressed at high baseline. Sample teacher
  rollouts, SFT on those.

## Stage transitions TO OPD

- **none → OPD:** baseline ∈ [0.4, 0.8], teacher available (METHODS.md Rule E).
- **SFT → OPD:** format stable (≥0.7), process headroom > 0.08, teacher
  available. SFT establishes format prior; OPD polishes process.

## References

- `caps/code-symbol-extraction/` — OPD reference; round-1 EOS-skip bug fixed
- `caps/diff-patch-fluency/` — cap #5 high-baseline failure case
- `caps/transcript-compaction/` — cap #3 sub-score confound case
- `caps/tool-call-arg-fidelity/` — cap #4 (output-shape confusion mitigation)
