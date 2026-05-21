# `lib/` — Shared Python helpers (round 3)

Round 3 promotes this directory from `capabilities/agentic-grpo/lib/` to
`capabilities/lib/` because most of it (and all the round-3 additions) are
methodology-agnostic.

## What's here

### Pi-trajectory normalization

- `pi_trajectory.py` — Python parser that maps pi session JSONL into kiln's
  canonical `Trajectory` schema (action / observation / context segments,
  warning-prefix metadata). Compat shim; canonical normalizer is
  `kiln trajectory inspect`.
- `test_pi_trajectory.py` — Python tests for the parser. Round-trip tested
  against the Rust normalizer in CI (see kiln #11 validation evidence).

### Round-3 helpers (new)

- `stage_manifest.py` — Read/write/validate `pipeline.md` header ↔
  `stages/<N>-<slug>.json` files ↔ `capability.jsonl` kept rows. Enforces
  the invariant that every `stages/` file corresponds to exactly one kept
  iter. Used by `run_stage.sh` (to write a new stage) and `run_pipeline.sh`
  (to validate before running).
- `method_router.py` — Apply the [`METHODS.md`](../METHODS.md) decision tree
  given a baseline (or post-stage) eval summary. Inputs: eval_summary.json,
  optional teacher availability, optional reward variance estimate, task
  shape (single-turn vs multi-turn). Outputs: recommended method + the rule
  that fired + the rationale string for `pipeline.md`.
- `headroom.py` — Per-sub-score headroom analysis. Reads
  `eval_summary.json`, prints `H_i = w_i × (1 - s_i)` per sub-score sorted
  descending. Used at baseline and before every stage.
- `cluster_summary.py` — Aggregate `pipeline.md` headers across all caps
  into the cluster manifest expected by [`DISTILLATION.md`](../DISTILLATION.md).
  Runs greedy-compatible-cluster selection given a sibling matrix. Phase G
  prerequisite.

### Agentic-GRPO specifics

- `agentic-grpo-notes.md` — ECHO defaults, pi-rollout shape, and the
  pi-session schema quirks that round 1 encoded into `pi_trajectory.py`.
  Promoted from `capabilities/agentic-grpo/README.md`. Read this when
  any stage uses the `agentic-grpo` method.

## Status of `pi_trajectory.py` after kiln round-2 improvements

This Python parser is a **compat shim**. The canonical pi-session
normalization lives inside kiln (Rust):

- `kiln_train::pi_trajectory` — the Rust-owned normalizer (kiln #11).
- `kiln trajectory inspect <jsonl>` — the CLI surface that exposes it (kiln #10).

New code should call `kiln trajectory inspect --json` to parse and
mask-validate sessions rather than importing `pi_trajectory.py`. The
Python parser remains for two reasons:

1. **Backwards compatibility** with round-1 cap scripts that still import it.
2. **Round-trip validation** — the test suite asserts the Python and Rust
   parsers produce equivalent segment sequences on shared fixtures.

## When to use which

| Use case | Tool |
| --- | --- |
| Live training loop, real-time inspection in Rust | `kiln_train::pi_trajectory` |
| Cap script wants a JSON-shaped trajectory diagnostic | `kiln trajectory inspect --json` |
| Cap script renders pi sessions into ScoredRollout JSONL today | `lib/pi_trajectory.py` |
| Round-trip / regression testing | `lib/test_pi_trajectory.py` |
| Choose methodology at any stage | `lib/method_router.py` |
| Validate stages/pipeline.md/capability.jsonl consistency | `lib/stage_manifest.py` |
| Analyze where headroom lives | `lib/headroom.py` |
| Build cluster manifest for distillation | `lib/cluster_summary.py` |

## Schema reference

See `docs/plans/echo-integration-plan.md` §3.3 for the canonical
trajectory schema (action / observation / context segments + `warning_prefix_len`)
that both pi-trajectory parsers target.
See [`../PIPELINE.md`](../PIPELINE.md) §2.3 and §3.1 for the `stages/`
and `pipeline.md` schemas that `stage_manifest.py` enforces.
See [`../METHODS.md`](../METHODS.md) §2 for the decision tree that
`method_router.py` implements.
