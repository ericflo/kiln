# `lib/` — Shared Python helpers (compat shim for round 2)

## What's here

- `pi_trajectory.py` — Python parser that maps pi session JSONL into kiln's
  canonical `Trajectory` schema (action / observation / context segments,
  warning-prefix metadata).
- `test_pi_trajectory.py` — Python tests for the parser. Round-trip tested
  against the Rust normalizer in CI (see kiln #11 validation evidence).

## Status after kiln round-2 improvements

This Python parser is now a **compat shim**. The canonical pi-session
normalization lives inside kiln (Rust):

- `kiln_train::pi_trajectory` — the Rust-owned normalizer (kiln #11).
- `kiln trajectory inspect <jsonl>` — the CLI surface that exposes it (kiln #10).

New code in this bucket should call `kiln trajectory inspect --json` to
parse and mask-validate sessions rather than importing `lib/pi_trajectory.py`.
The Python parser remains for two reasons:

1. **Backwards compatibility** with round-1 cap scripts that still import it.
2. **Round-trip validation** — the test suite asserts the Python and Rust
   parsers produce equivalent segment sequences on shared fixtures, so the
   Python implementation is the cross-check.

## When to use which

| Use case | Tool |
| --- | --- |
| Live training loop, need real-time inspection in Rust | `kiln_train::pi_trajectory` |
| Cap script wants a JSON-shaped trajectory diagnostic | `kiln trajectory inspect --json` |
| Cap script needs to render pi sessions into ScoredRollout JSONL today | `lib/pi_trajectory.py` (until `kiln rollout` (#34) lands fully) |
| Round-trip / regression testing | `lib/test_pi_trajectory.py` |

## Schema reference

See `docs/plans/echo-integration-plan.md` §3.3 for the canonical trajectory
schema (action / observation / context segments + warning_prefix_len) that
both parsers target.
