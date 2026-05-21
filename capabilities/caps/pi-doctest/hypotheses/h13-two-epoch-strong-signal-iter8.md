# Hypothesis H13 — Two-epoch GRPO on strong-signal filter (iter 8)

**Family:** H13 (more updates on filtered data, not more data)
**Target sub-score:** `composite`, lift over iter 5's reproducible best (0.8990)

## Claim

Running 2 epochs over iter 5's 11 strong-signal groups (concatenated 2x = 22
effective groups) at the same `lr=1e-5` will lift composite by ≥+0.02 over
iter 5 — because the surviving groups have known non-zero variance and the
policy should benefit from a second pass.

If 2 epochs **regresses** composite, the verdict is "1 epoch is the sweet
spot on filtered data; more updates over-train."

## Mechanism

Iter 3 (40 tasks, lr=1e-5) regressed from iter 5 (11 tasks, lr=1e-5) at
0.8453 vs 0.8990. That said "more groups, same lr = overshoot," not
necessarily "more updates over-train." H13 isolates the question by holding
group composition fixed and only varying the epoch count.

## Results

Run on H100 SXM-80GB, base `pi-doctest-iter5/adapters` (i.e. starting from
the base model, not from iter 5 — fresh GRPO training).

Same env workaround as prior iters
(`KILN_BATCHING_ENGINE=0 KILN_DISABLE_FUSED_GDN_GATES=1`) — H100 quirks
still live in `kiln-polish.jsonl`, not the skill.

### Headline numbers

| metric | base | iter 5 (1ep, 11 strong) | iter 8 (2ep, 11 strong) | Δ vs iter 5 |
|---|---|---|---|---|
| composite | 0.8052 | **0.8990** | **0.7502** | **−0.149** |
| n zeros | 4 | 1 | 5 | +4 |
| rollouts_nonzero | 20 | 23 | 19 | −4 |
| mean wall_clock_s | 31.4 | 19.86 | 56.45 | +36.6 (3× slower) |

### Per-task pattern

Iter 8 lost 4 tasks that iter 5 had recovered or fixed:

- task_0004 (`x_or_y`): iter 5 passed, iter 8 timeout-zero
- task_0007 (`circular_shift`): iter 5 passed, iter 8 timeout-zero
- task_0011 (`closest_integer`): iter 5 passed, iter 8 zero
- task_0015 (`vowels_count`): iter 5 partial, iter 8 timeout-zero
- task_0019 (`fizz_buzz`): iter 5 mixed, iter 8 timeout-zero

The wall-clock 3× blow-up (19s → 56s) is the smoking gun: 2 epochs on the
strong-signal groups trained the model to thrash — the adapter calls more
tools, re-runs more doctests, and hits the 120s wall-clock cap.

### Group statistics (training)

Effective 22 groups (11 strong × 2 epochs). All 22 passed the
`dynamic_sampling` filter (the variance check is per-group, and our 11
groups all had var > 0.05). Loss curve was stable across epoch 1 → epoch 2.

The over-training did not manifest in the loss curve — only in the
behavior on held-out eval. Classic "GRPO over-confidence" pattern.

## Verdict

**✗ falsified.** 2 epochs on strong-signal-filtered data REGRESSED composite
by -0.149 vs iter 5, and -0.055 vs base. 5 of 24 eval tasks went from "pass
or partial" → "zero or timeout."

The hypothesis "more updates on the same strong-signal data lifts composite"
is false at the chosen hyperparameters. The alternative reading — "1 epoch
is the sweet spot on filtered data; more updates over-train" — is consistent
with the data.

## Disposition

**Iter 5 remains the reproducible best.** 1 epoch on strong-signal-filtered
groups at lr=1e-5 is the recipe to ship.

This is an over-training negative result, comparable to iter 3 (40 tasks at
lr=1e-5 → 0.8453, also a regression). The pattern across iters 3, 4, 8:
**any recipe that gives the model more updates than iter 5's 11 strong-only
single-epoch lands below iter 5.** This is consistent with the §0 "training-
set size has a sweet spot" guidance from the skill.

## Iter 9 plan

Per the skill's variance discipline, before claiming "+0.094 composite
uplift" as a kept-ship result, run a second-seed eval on the iter 5 adapter:

- Iter 9: re-load `pi-doctest-iter5` adapter, re-run eval against the same
  24-task held-out set. This isolates eval-rollout variance from training-
  seed variance. Expected band: [0.84, 0.94] if iter 5 is genuinely better
  than base (base measured at 0.8052, base 2nd-seed not yet measured but
  iter 7 establishes ±0.05 single-eval stdev).

- If iter 9 lands in [0.85, 0.92]: iter 5 ships as the final adapter, with
  mean ± stdev reported. The skill's §0 "single-seed GRPO is high-variance"
  guidance is followed.

- If iter 9 lands < 0.83: iter 5 was a lucky sample too. Drop to OPD or
  revisit the rubric (matches iter 7's lesson from iter 2).

## Pod cost

H100 SXM-80GB at $2.99/hr × ~50 min (training + eval) = ~$2.50.
Cumulative session H100 cost going into iter 8: ~$30; total after iter 8:
~$32.50.
