# pi-doctest

Capability under `capabilities/agentic-grpo/`. Round 2 layout
(see [`../../LAYOUT.md`](../../LAYOUT.md)).

## Read first

1. [`capability.md`](capability.md) — the contract: goal, task shape, rubric,
   adversarial design, hypotheses.
2. [`capability.config.json`](capability.config.json) — trainer + rollout defaults.
3. [`../../LAYOUT.md`](../../LAYOUT.md) — uniform layout and which kiln CLIs are used.
4. [`../../agentic-grpo/KILN_IMPROVEMENT_ISSUES.md`](../../agentic-grpo/KILN_IMPROVEMENT_ISSUES.md) — the kiln improvements this layout assumes are landed.

## Quickstart

```bash
# 0. (one time) build corpus
python3 build_corpus.py

# 1. baseline eval (base model only)
./capability.oracle.sh

# 2. first training iter
./run_iter.sh h1-default-recipe

# 3. inspect the new row
tail -1 capability.jsonl | python3 -m json.tool
```

## History

Round 1 artifacts (writeups, old iter log, ad-hoc scripts) live under
[`archive/`](archive/). The next round starts with a fresh
`capability.jsonl`.
