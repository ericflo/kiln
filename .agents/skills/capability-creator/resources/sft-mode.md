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

## When SFT plateaus on parameter tweaks

If you've run 5+ SFT variants — different ranks, learning rates,
epochs, filter thresholds, chain depths — and they all converge at the
same composite within σ, you've probably hit a **signal plateau**, not
a model-capability ceiling. The training distribution has already told
the model everything it can.

The first move is to check whether you're still on the right method.
METHODS.md §4.5 covers the full picture: a plateau is new evidence,
and re-running §2's decision tree against the plateau's sub-score
profile may fire a different rule than your original choice. Maybe the
right next move isn't more SFT at all — it's GRPO with a redesigned
reward, OPD with different teacher conditioning, or
agentic-GRPO/`--no-policy-loss` if your task has multi-turn structure
you weren't using.

If the tree still points at SFT (often the case when the headroom is
in a sub-score the model already knows how to produce but doesn't do
reliably), one technique that's worked is a **chained alternating
SFT schedule**:

1. Look at the sub-scores at the plateau. Which axes are pinned?
   Which are flexing? A pinned sub-score is usually one the current
   training distribution doesn't carry signal for.
2. Ask: is there a second, qualitatively different distribution that
   would carry signal for the pinned axes? Common pairings — by no
   means exhaustive:
   - rubric-perfect synthesized outputs (drives format precision)
     paired with high-scoring rollouts (drives outcome correctness on
     varied prompt shapes)
   - terse final-answer-only examples paired with worked-through
     derivations
   - canonical idiomatic examples paired with edge-case stress tests
3. If yes: chain SFT stages that **alternate** between the two
   distributions. Each stage stays small (rank 4 / α 8 / lr 1e-5, 1-3
   epochs) so the model never catastrophically forgets the other
   lesson. Stop when an additional swap stops adding lift.

Reach for this when the SFT parameter sweep has clearly plateaued AND
the decision tree still points at SFT. The technique trades simplicity
for one more knob (the schedule); justify the trade by the evidence,
not by habit in either direction. We don't have a frequency estimate
for how often caps need it — early in the round-3 cycle, treat each
cap on its own evidence.

**Synthesizing a second distribution.** If the rubric is programmatic
(format_regex + expected_value), you can often **synthesize**
rubric-perfect examples deterministically from the task scaffold
rather than sampling them — every example scores 1.0 by construction.
See `caps/pi-faithful-completion/iter18_ideal_prep.py` for a working
shape. This is a useful complement to rollout-derived data because the
distributions differ in what kind of noise they expose the model to.

**Reference case study.** `caps/pi-faithful-completion` round-3 hit a
0.77 plateau across 12 single-distribution variants and broke it with
a 6-stage alternating chain that reached 0.808 (capturing 93.4% of the
prompted-lift ceiling). See its `sft_chain_findings.md` for the full
trace. The specific data pairing there (synthesized ideal outputs vs
strict-prompt rollouts) was *one instantiation* — the same diagnostic
applied to a different cap would pick different distributions.

## Practical gotchas

- **Use `--trainer generic`.** The native trainer
  (`cuda_native_sft_train`) is currently ~50× slower than generic on
  Qwen3.5-4B; the generic path (`sft_train` in `trainer.rs`) routes
  through `BackendRuntime` and gets the production-tuned kernels. See
  kiln issue #1063 for the backport tracking the native trainer.
- **Flatten the adapter directory after training.** `cuda_sft_file`
  writes to `<output-dir>/<adapter-name>/...` but kiln serve expects
  `<output-dir>/...` directly. After training, `mv` the inner files up
  and `rmdir` the nested directory or kiln will refuse to load the
  adapter. Tracked in kiln issue #1065.
- **Kill kiln serve before training.** Both the serve process and
  `cuda_sft_file` load the full model — running both at once OOMs the
  A6000. Restart kiln serve after the SFT step completes for eval.

## References

- `caps/math-broad/` — round-1 reference cap, 32 meta + 32 numeric winner
- `caps/json-schema-adherence/` — JSON shape SFT reference
- `caps/python-algo/` — algorithmic SFT reference
- `caps/pi-faithful-completion/sft_chain_findings.md` — round-3
  alternating-chain trace (12 mono-distribution variants plateaued
  before the multi-distribution chain broke through)
