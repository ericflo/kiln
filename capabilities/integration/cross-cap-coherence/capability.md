# Capability: cross-cap-coherence (integration eval)

**Status:** New in round 2. **Eval-only — does not train an adapter.**

## Description

Measure whether an adapter trained on one capability *regresses* other
capabilities, and whether the suite of round-2 adapters *composes* into
a coding agent that's measurably better than base across all measured
dimensions.

This is the missing answer to:

- "Did the round-1 pi-faithful-completion winner accidentally hurt
  pi-code-comprehension?" (round 1 hinted yes; nothing measured it).
- "Can we serve doctest + code-search + faithful-completion adapters
  in any reasonable composition without breakage?"
- "What's the *integrated* coding agent score, not the per-cap score?"

## Mode of operation

For a given adapter name (or comma-separated list), run a small
held-out slice of each member cap's eval set against that adapter and
aggregate. Output: per-cap composite + a single cross-cap composite +
per-cap delta vs base.

## Member capabilities (v0)

The integration suite includes a slice from each:

| Member cap | Eval slice | Why |
| --- | --- | --- |
| `pi-doctest` | 8 tasks | Tool-call efficiency + verify-before-done; pivotal round-1 winner |
| `pi-code-comprehension` | 8 tasks | Read + structured output; biggest round-1 win |
| `pi-code-search` | 8 tasks | Search-then-read habit |
| `pi-faithful-completion` | 8 tasks | Honest termination |
| `pi-error-recovery` | 8 tasks | Error attention |
| `pi-incremental-progress` | 6 tasks | Decomposition discipline |
| `pi-context-aware-edits` | 6 tasks | Style consistency |
| `pi-search-then-read` | 6 tasks | Context efficiency |

Total: 58 tasks. Estimated wall-clock on A6000: ~30 minutes for a full
sweep across base + 1 adapter at 3 seeds.

Each member cap's eval slice is drawn from a *held-out* portion that
was NOT used in that cap's own training or per-cap eval. This avoids
the train-leak class of bugs.

## Composite

```
per_cap_composite[c] = mean(score_one(t) for t in member_eval[c])
cross_cap_composite   = mean(per_cap_composite[c] for c in members)
per_cap_delta[c]      = per_cap_composite[c] - base_per_cap_composite[c]
```

The headline metric is **cross_cap_composite**. The detail view is
**per_cap_delta**, which surfaces regressions clearly: if training X
regressed Y by 0.05, that's a per_cap_delta[Y] of −0.05 even when the
overall cross_cap_composite is positive.

## Rubric (no v0 — wraps per-cap rubrics)

This cap does not have a standalone rubric. It imports each member
cap's `rubric.py` and calls its `score_one(rollout)` function. See
`rubric.py` for the composition logic.

## Reporting shape

`capability.oracle.sh <adapter> [<adapter2>...]` writes a JSON
report:

```json
{
  "adapters": ["pi-doctest-iter5"],
  "base": { "cross_cap_composite": 0.7142, "per_cap": { "pi-doctest": 0.8854, "pi-code-comprehension": 0.6112, ... } },
  "per_adapter": {
    "pi-doctest-iter5": {
      "cross_cap_composite": 0.7234,
      "cross_cap_delta": +0.0092,
      "per_cap": { "pi-doctest": 0.8958, "pi-code-comprehension": 0.6004, ... },
      "per_cap_delta": { "pi-doctest": +0.0104, "pi-code-comprehension": -0.0108, ... },
      "regressions": ["pi-code-comprehension"]
    }
  }
}
```

Any `per_cap_delta` < −0.02 is flagged as a `regression`.

## How to add this to a per-cap workflow

After any round-2 cap's `run_iter.sh` completes successfully, run:

```bash
cd /workspace/kiln/capabilities/integration/cross-cap-coherence
./capability.oracle.sh <adapter-name>
```

The output JSON makes it obvious whether the per-cap win came at the
cost of other capabilities. **A positive per-cap win that ships with
> 2 cross-cap regressions of > 0.02 each is a flag for skill-clobber**
— consider lower lr, shorter training, or chained training with anchor.

## Future directions

### Training a composite adapter

The natural follow-up is: train ONE adapter on a mixture of all
member-cap training corpora, balanced by token count. This would be
a new "phase-3" cap (perhaps `integration/cross-cap-composite/`) that
*trains* using `cuda_grpo_ablation` over a mixed JSONL.

Not in scope for round 2 — cross-cap-coherence first measures whether
composition is even feasible; the composite-adapter training comes
after we see clean per-cap wins that don't regress siblings.

### Adapter merging (future)

If kiln gains adapter-merging support (LoRA weight averaging or
TIES-style merging), the integration suite is exactly the
evaluation harness for that work.

## Round 2 standard workflow

```bash
# Eval one adapter against the full integration suite (3 seeds).
./capability.oracle.sh pi-doctest-iter5

# Eval multiple adapters and compare.
./capability.oracle.sh pi-doctest-iter5 pi-faithful-completion-iter50

# Eval all currently-installed adapters.
./capability.oracle.sh $(ls /workspace/adapters | tr '\n' ',')
```

No training step. `run_iter.sh` is symmetric with other caps but
delegates to `capability.oracle.sh` — kept so cron jobs / drivers can
treat the integration track uniformly.
