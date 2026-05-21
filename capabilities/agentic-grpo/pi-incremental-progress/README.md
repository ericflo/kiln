# pi-incremental-progress

Capability under `capabilities/agentic-grpo/` (round 2 — new cap).

## Read first

1. [`capability.md`](capability.md) — contract: goal, rubric, hypotheses.
2. [`capability.config.json`](capability.config.json) — trainer + rollout defaults.
3. [`../../LAYOUT.md`](../../LAYOUT.md) — uniform layout and kiln CLIs used.
4. [`../README.md`](../README.md) — ECHO defaults and pi-rollout shape.

## Status

**Scaffold.** The capability.md is fully specified; the implementation
files (build_corpus.py, rollout.py, rubric.py) are NotImplementedError
stubs. The next agent picking this up:

1. Fill build_corpus.py per `capability.md ## Task shape`.
2. Fill rubric.py per `capability.md ## Rubric (v0)`.
3. Fill rollout.py per the pi-rollout reference (`../pi-doctest/rollout.py`).
4. Fill rubric_sanity.py + `calibration/{good,bad}.jsonl` per `capability.md ## Adversarial design (§0)`.
5. Run `./capability.oracle.sh` to baseline.
6. Run `./run_iter.sh h1-default-recipe` for the first training iter.

The capability.md `## Adversarial design (§0)` is the calibration spec
— the §0 cheats are the bad-class entries for `calibration/bad.jsonl`.

## History

Brand-new in round 2 — no archive.
