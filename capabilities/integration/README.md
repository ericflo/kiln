# `integration/` — cross-cap evaluation track

Round 2 introduces a third top-level capability bucket alongside
`agentic-grpo/`, `opd/`, and `sft/`.

**Purpose.** A per-cap win is necessary but not sufficient. The
question that ultimately matters is: **do these adapters compose into
a coding agent that's better at the integrated task than the base
model?**

Round 1 measured per-cap composite. It did **not** answer:

- Does training `pi-faithful-completion` regress `pi-code-comprehension`?
- Can we serve `pi-doctest-iter5` and `pi-code-search-iter5-h5-replay`
  simultaneously, or do they interfere?
- If we merge / sum / chain these adapters, what comes out?

The integration track measures these.

## Track contents

```
integration/
├── README.md (this file)
└── cross-cap-coherence/
    ├── capability.md        — the contract for the cross-cap eval
    ├── capability.config.json
    ├── capability.jsonl
    ├── rubric.py            — composite of per-cap composites
    ├── build_corpus.py      — samples held-out tasks from each member cap
    ├── capability.oracle.sh — runs each member cap's oracle against one adapter
    ├── run_iter.sh          — evaluates a single adapter against the integration suite
    ├── hypotheses/
    ├── manifest/
    └── README.md
```

## How it differs from the per-cap track

| Per-cap track (round 1+2) | Integration track (round 2) |
| --- | --- |
| Trains an adapter on one cap | Does not train — evaluates only |
| One rubric | Aggregates 5+ per-cap composites |
| Validates a single behavior | Validates **composition** of behaviors |
| Measures uplift vs base on one task type | Measures regression across the whole suite |

## When to use it

After any non-trivial training run in `agentic-grpo/`, `opd/`, or
`sft/`, run `integration/cross-cap-coherence/capability.oracle.sh
<adapter-name>` to check that the new adapter didn't quietly break
something else. Round-2 standard cap workflows include an optional
integration eval step.

## Not a training cap

This is **eval-only.** No `cuda_grpo_ablation`, no `cuda_opd_remote`,
no training rollouts. `run_iter.sh` exists for symmetry but it just
calls `capability.oracle.sh` against the configured adapter.

If you want to *train* a composite adapter, that's a future direction
— see `cross-cap-coherence/capability.md` `## Future directions ##
Training a composite adapter`.
