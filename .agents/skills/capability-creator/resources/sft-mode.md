# SFT mode — irreducible lore

Read this when [`capabilities/METHODS.md`](../../../../capabilities/METHODS.md)
routes a stage to **SFT** (Rule C, Rule D, or as the recovery path in §4.3
high-baseline OPD rescue).

`SKILL.md` covers the universal loop; this file covers only what's specific
to SFT and not in METHODS.md / PIPELINE.md / NEXT_ROUND.md.

## When SFT is the right tool

- Baseline outcome < 0.3 OR format headroom > 30% of total headroom.
- Curated training data is feasible (8-256 examples, diversity > volume).
- The capability is *teachable through worked examples* — algorithms,
  templates, formats, style.

## When SFT is NOT the right tool

- Baseline is already mid-range (0.4-0.8) and a teacher exists → use OPD.
- Reward is verifiable and rollouts have variance → use GRPO.
- Multi-turn tool-calling → use agentic-GRPO (SFT may bootstrap, but the
  primary trainer is agentic-GRPO).
- You don't have ground-truth pairs and can't write them.

## Hyperparameter defaults

- rank=4, alpha=8, lr=1e-4, 1 epoch, dataset_cap ≤128.
- Start small. Only escalate rank on evidence of underfitting.
- One epoch usually beats two — overtraining is the dominant SFT failure mode.

## SFT-specific failure modes

### Surface-form mimicry

Training drills make the model parrot the eval's surface form (numbers,
formatting, vocabulary). Generalisation fails; you get a brittle adapter
that works only on prompts shaped like training.

**Mitigation:** answer-form discipline (next §) + meta-questions transfer
surprisingly well. Phrase examples 5-10 different ways; don't mass-produce
one phrasing.

### Length compression / inflation

The eval may reward responses in a length range; SFT moves average length
into the window without improving content.

**Mitigation:** check the histogram of trained responses vs base. If only
length changed and not content, the dataset is rewarding length.

### Refusal sneak-through

"I cannot answer" satisfies many ceiling-style evals (no false positives →
no penalty). SFT on examples that include legitimate refusals teaches the
model to refuse more broadly.

**Mitigation:** every refusal-containing example needs a matched
"this is actually answerable; here is the answer" counterpart.

### Eval-driven overfitting

If the eval has a narrow set of recurring prompt shapes, SFT can find a
template that satisfies them without doing the task. Round-1 `math-broad`
hit this initially.

**Mitigation:** anchor regression on a non-target domain (next §). Score
both target eval and anchor; reject the adapter if anchor regresses > 0.02.

## Anchor pattern (SFT-specific)

SFT silently clobbers non-target capabilities through style drift. Every
SFT cap MUST have a `capability.anchor.sh` that runs a regression watch on
a non-target domain (e.g. for `math-broad`, the anchor is a code-generation
eval; for `json-schema-adherence`, the anchor is a free-form Q&A eval).

`run_stage_sft.sh` runs the anchor after the target eval. Anchor regression
> 0.02 fails the stage (the new adapter is broken in the wrong way).

Reference: `caps/math-broad/capability.anchor.sh` for the SFT reference shape.

## Answer-form discipline (the hidden killer of transfer)

Capabilities elicited through SFT generalise much better when the training
data is *varied in surface form* even though it teaches the same skill.

- Train arithmetic on "the algorithm in 30 prose framings" — model
  generalises to numeric problems.
- Train arithmetic on "300 numeric worked examples in one style" — model
  parrots the style.

Verbal/structural framings often beat surface-form drill. Don't drill the
eval's surface form; teach the *algorithm* or *frame*.

Round-1 `math-broad` case: 32 meta + 32 numeric (in that order) at rank 8 /
1 epoch / lr 1e-4 won. 100+ numeric-only examples saturated to overfitting.

## Loss-curve signal (unlike OPD/GRPO)

SFT loss curves are informative. Watch for:

- **Flat from step 1** → data is the problem (too few examples, wrong shape,
  or model already at ceiling on this).
- **Drops sharply then plateaus** → healthy.
- **Drops sharply then climbs** → overfitting; reduce epochs.

`train_receipt.json` carries `loss_curve_final` and per-step samples; check
both before declaring the stage broken.

## Dataset construction

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

One example per line. Keep `datasets/sft.train.jsonl` committed (it's not
a blind-eval source). `build_corpus.py --method sft` produces it from
`datasets/train.tasks.jsonl` + curated outputs (you write the assistant
turn or take it from a teacher rollout).

## Receipt fields specific to SFT

- `epochs`, `dataset_size`, `dataset_cap_applied`
- `loss_curve_final` — final training loss
- `loss_curve_samples` — per-step loss samples
- No `groups_*`, no `echo_*`, no `reward_*` (those are GRPO-specific)

## Stage transitions FROM SFT

- **SFT → OPD:** when format/outcome is stable (≥0.7) and teacher available.
  See METHODS.md §4.2.
- **SFT → GRPO:** when format is stable AND rollouts have reward variance > 0.05.
- **SFT → agentic-GRPO:** when format is stable AND task is multi-turn.
- **SFT → STOP:** if anchor regresses, even if target lifts.

## Stage transitions TO SFT (recovery)

- **OPD → SFT (cap #5 rescue):** baseline > 0.80 with variable rollout
  quality + OPD regressed. Sample teacher rollouts, SFT on those. See
  METHODS.md §4.3.

## References

- `caps/math-broad/` — round-1 reference cap, 32 meta + 32 numeric winner
- `caps/json-schema-adherence/` — JSON shape SFT reference
- `caps/python-algo/` — algorithmic SFT reference
