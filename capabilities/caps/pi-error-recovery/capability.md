# Capability: pi-error-recovery

**Status:** New in round 2. Scaffold.

## Description

When a tool call fails — file not found, permission denied, syntax error
in code the agent just wrote, command not found, malformed JSON
argument, timeout — the agent must **read the error**, **diagnose**, and
**try a different approach**. The 4B today exhibits three failure modes:

- **Loops.** Retries the exact same call, sometimes 5-10 times.
- **Gives up.** Emits "I cannot complete this task" without trying any
  alternative.
- **Compounds the error.** Adds new wrong assumptions on top of the
  original failure (e.g. `git apply` fails → it tries `git am` →
  `git am` fails → it tries `patch -p0` → and so on without ever
  reading what the actual error message said).

A capable pi coding agent reads the error text, identifies the failure
category (file path wrong / permission / syntax / dependency missing
/ etc.), and tries a *different* approach informed by the error.

This is a behavior cap with very strong ECHO signal: the tool-result
("observation") tokens *contain the error message*. ECHO trains the
model to actually attend to those tokens before emitting the next
action.

## Base model

Qwen3.5-4B (kiln serve on `http://localhost:8420`).

## Rollout source

Pi sessions. Multi-turn; we want to observe the recovery turn after
the initial failure. Max turns 8.

## Task shape

Each task is a `(workspace, action, error_class)` triple:

- **Workspace** — a small sandbox with deliberate failure modes seeded
  (e.g. read-only file, missing dep, syntax error in a stub).
- **Action** — the prompt asks the agent to do something that requires
  the failing sub-action (e.g. "modify `lib/util.py`" when `lib/util.py`
  is read-only, requiring `chmod +w` first).
- **Error class** — ground truth: the failure category. Used by the
  rubric to verify the agent's recovery is *appropriate to the error*.

Six error classes covered by v0:

1. `file_not_found` — `cat /nonexistent` → recovery: `ls` first
2. `permission_denied` — read-only file → recovery: `chmod` or `sudo` or different file
3. `syntax_error` — model wrote invalid Python → recovery: read the error, fix the syntax
4. `command_not_found` — `pytest` not installed → recovery: `pip install` or use `python -m unittest`
5. `dependency_missing` — `import foo` fails → recovery: install or use stdlib alternative
6. `timeout` — command hangs → recovery: kill, simpler approach

## Rubric (v0)

Multi-component, modelled on pi-doctest §rubric-v1. **Uses the
round-2 multiplicative format gate from the start** (no need to
discover this lesson again).

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor (multiplicative) | Final state of the workspace matches the gold: the originally-requested action completed successfully. | Just emitting "I tried, here's the result" — gold check verifies the workspace was actually modified correctly. |
| `format_compliance` | multiplicative gate | Final assistant turn contains: (a) acknowledgement of the failure, (b) a one-line summary of what went wrong, (c) confirmation that the recovery worked. | Boilerplate "I recovered" without referencing the actual error text. Rubric requires substring overlap with the tool-result error message. |
| `recovery_was_different` | 0.30 | The retry call differs from the failed call in a meaningful way (different tool, different command structure, different file path). Bytes-overlap < 0.7 between failed and recovery calls. | Tiny syntactic tweaks. Score 0.3 for small changes, 1.0 for substantive change. |
| `recovery_appropriate_to_error_class` | 0.25 | The recovery action is in the set of valid responses for the error class. E.g. for `file_not_found`, valid recoveries include `ls` to find the right path; invalid: trying the same path with a different command. | Matching the error class by keyword without making sensible recovery. Rubric checks the recovery *action shape* matches the error class. |
| `no_loop` | 0.20 | Tool-call signatures don't repeat. `1 - duplicates / total`. | Same as round-1 implementations. |
| `read_error_before_retry` | 0.15 | The session shows the model read the tool-result (observation tokens) before emitting the retry. Requires ECHO env-CE to fire. | Skipping straight from one failed call to another without an intervening think/text turn. |
| `base` | 0.10 | Constant floor for any rollout that completed without crashing pi. | — |

**Composite = `outcome × format_compliance × (0.30·recovery_was_different + 0.25·recovery_appropriate_to_error_class + 0.20·no_loop + 0.15·read_error_before_retry + 0.10)`**

Multiplicative format gate from day one because round-1 evidence
showed it's the right shape for behavior caps with measurable
process signal.

## Adversarial design (§0)

**Q: What's the cheapest way to score 1.0 without doing the capability?**

A1: Always retry with `ls` first, regardless of error class. Wins
   `recovery_was_different`, may or may not win
   `recovery_appropriate_to_error_class`.
   
   Mitigation: the error_class-appropriateness rubric is
   *fine-grained per class*, not "any change to recovery." `ls` is the
   right answer for `file_not_found` and wrong for `syntax_error`. Score
   averages across the 6 error classes; constant-`ls` strategy scores
   ~0.17 on this sub-score.

A2: Emit a long apology in the final turn that mentions the error
   message verbatim, then say "I have recovered." Format-compliance
   is gameable by long apologies. Outcome should kill this — if the
   workspace isn't actually fixed, outcome=0 → composite=0.

   Mitigation: outcome gate is sufficient.

A3: Try every possible recovery in sequence (`ls`, `chmod`, `pip
   install`, `kill`) until one works. Wastes turns, may eventually
   pass outcome.

   Mitigation: `no_loop` decays with each duplicate; turns budget
   bounds the rollout to 8 turns; `recovery_was_different` only
   counts the first retry. Spam-strategy scores ~0.4 composite.

A4: Pi never actually fails the first call because the sandbox
   wasn't set up properly. Capability isn't exercised at all.

   Mitigation: `build_corpus.py` verifies each task triggers the
   intended failure on a base-model dry-run before committing the
   task to the eval set.

## Headroom estimate

Pre-train estimate (will be confirmed by iter 0):

- Baseline composite: **~0.40** (4B routinely loops or gives up on
  these errors; expect outcome=0 on majority of tasks)
- Headroom: ~0.60 (genuine behavior gap)
- Target sub-score: `recovery_appropriate_to_error_class` (biggest
  movable mass)

This is a high-headroom cap. The behavior gap is large and the
multiplicative gates make every component matter.

## Hypotheses

- **H1: default Phase 1 GRPO recipe.** Train on a mix of error
  classes. Hypothesis: composite +0.15 over base.
- **H2: ECHO-heavy.** λ=0.10 (paper-suggested 0.05 may underweight env
  attention for this cap). Hypothesis: better `read_error_before_retry`
  signal.
- **H3: strong-signal filter.** `--filter-var-min 0.05` to focus on
  groups where recovery matters most. Hypothesis: cleaner gradient.
- **H4: per-class balance.** Stratified sampling across the 6 error
  classes. Hypothesis: avoids overfitting to the most common one.

## Composition with other caps

- **Upstream:** None — this cap doesn't depend on others.
- **Downstream:** `pi-failure-triage` (which is about *fixing the bug*)
  benefits from error-recovery training because failure-triage trains
  the same env-attention behavior on a different surface.
- **Integration test:** included in `integration/cross-cap-coherence/`.

## Round 2 standard workflow

```bash
python3 build_corpus.py             # Creates train + eval splits across 6 error classes
./capability.oracle.sh              # Baseline
./run_iter.sh h1-default-recipe     # H1
./run_iter.sh h2-echo-heavy         # H2 (ECHO_LAMBDA=0.10)
```

See `../../LAYOUT.md` for the canonical layout.
See `../README.md` for ECHO defaults and pi-rollout shape.
