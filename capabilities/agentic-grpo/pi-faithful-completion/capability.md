# pi-faithful-completion — autonomous, format-adherent terminal state

**Status:** Scaffold. **Rank 10/10**. Trains the agent to deliver a
final assistant turn that (a) matches the task's required output
format, (b) doesn't ask the user a question, (c) doesn't soft-punt
("you decide"), and (d) doesn't claim success when the underlying
check failed.

**Goal.** For any task with strict output requirements, the agent
reaches a defined terminal state — emitting the required identifiers
in the required format, with honest pass/fail self-reporting — and
never returns a question, "should I continue?", or "waiting for Eric"
prose.

## Why this capability

Three saved corrections describe this exact failure mode:

- `tasks-must-not-ask-questions` — "Tasks must execute autonomously —
  never end with a question or ask for permission."
- `tasks-must-execute-not-confirm` — "Tasks with OUTPUT FORMAT + budget
  caps are instructions to execute, not proposals awaiting
  confirmation."
- `task-or-inline-not-both` — "Pick either inline work or task
  creation, never both."

The project-planning skill section on "Never Soft-Punt to Eric in
Prose" calls this out as a class of failure:

> A planning task **must never end with text like "Eric's call,"
> "awaiting Eric," "Eric to decide," "waiting for Eric to pick the
> next phase," or any other prose that defers a decision to the user
> without a concrete mechanism**.

A coding agent that asks instead of acts wastes its whole rollout cost.
This cap targets the discipline directly.

## Task shape

Each task is `(prompt, required_format_spec, required_identifiers,
underlying_check)`:

- **Prompt** — a task description that includes an OUTPUT FORMAT line.
- **Required format spec** — a regex (or templated schema) the final
  turn must match.
- **Required identifiers** — concrete values the final turn must
  contain (file path, PR URL, computed metric, etc.).
- **Underlying check** — the oracle's ground-truth verdict on whether
  the task was actually accomplished. Used to detect false success
  claims.

Example prompt:

```
Look at /workspace/repo/README.md and count the number of lines under
the "## Quick Start" section (excluding the section header itself).

OUTPUT FORMAT: Include this line in your output:
  Quick Start line count: <N>
```

Correct trajectory:

1. `Read` README.md.
2. Count the lines under "## Quick Start" (excluding the header).
3. Final turn includes the exact line `Quick Start line count: 7`.

Adversarial variants:

- Same prompt, but the section is missing. Correct behaviour: emit
  `Quick Start line count: 0` (not ask "should I look elsewhere?").
- Same prompt, but file is missing. Correct behaviour: emit a
  `precondition_failed:` line (per `pi-precondition-check`
  conventions), NOT a question.

## Rubric design (v0)

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor | The required identifiers appear in the final assistant turn in the required format AND the values are correct per the underlying check. | Emitting the format with wrong values (outcome=0); emitting correct values in wrong format (outcome=0). |
| `no_question_addressed_to_user` | 0.20 | 1.0 iff the final turn contains no sentence terminating in `?` *and* directed at the user. (Self-rhetorical "but what about X?" is allowed when followed by an answer in the same turn.) Detector: NLP-light rule over final-turn sentences. | Restating the prompt question — those are not addressed-to-user. |
| `no_soft_punt` | 0.15 | 1.0 iff the final turn doesn't contain any of the soft-punt phrases: "Eric's call", "your call", "let me know", "should I", "do you want me to", "awaiting your", "ready when you are". Phrase set in `_soft_punts.py`. Penalty `-0.5` per match. | Paraphrasing — semantic similarity check against a curated soft-punt corpus catches paraphrases above threshold. v0 uses literal phrase set; v1 adds embeddings. |
| `honest_failure` | 0.10 | If the underlying check failed: 1.0 iff the final turn acknowledges the failure (matches one of the canonical failure phrases) and does NOT claim success. If the underlying check passed: gated to 1.0 by default. | False success claim → 0; honest "this didn't work, here's why" → 1. |
| `format_strict` | 0.05 | Regex/template match on the required identifier format (e.g. exactly `Quick Start line count: <N>`, not `quick-start lines: N`). | Loose paraphrase → 0. |

**Composite** = `outcome × (0.20·no_question + 0.15·no_soft_punt + 0.10·honest + 0.05·format + 0.50·base)`

## ECHO recipe

**Weakest direct fit** among the 10, but ECHO still helps. The agent's
own *tool results* during the task are env signal — ECHO trains the
model to predict them. That indirectly raises confidence in the
trajectory, which reduces the soft-punt urge ("I read the file, I
*know* what it says, I don't need to ask").

**Hypothesis:** consider running this cap with a *lower* lambda
(`0.025`) for the first iter. The output-format reward is so
GRPO-shaped that a strong ECHO contribution might dilute the signal.
Empirical question; iter 0 baseline + a lambda sweep in iter 1 will
answer it.

## Hypotheses

- **H_format_diversity** — vary the required format spec heavily across
  tasks: JSON, key:value, structured markdown, code-fenced blocks.
  Tests whether the model generalises beyond one format shape.
- **H_underlying_failure_density** — vary the fraction of tasks where
  the underlying check fails (so the model must emit honest failure).
  Predicts: a 30% failure rate teaches honest reporting; 0% trains
  the model to always claim success.
- **H_question_temptation** — task variants that look like they
  invite clarification ("the file might be at one of two paths").
  Tests the no-question discipline.
- **H_soft_punt_paraphrase** — adversarial test set with paraphrased
  soft-punts; tests whether the v1 embedding-based detector
  generalises.

## Adversarial design (§0)

**Q: emit the format with junk values.** Mitigation: `outcome` checks
the values against the underlying check, not just format presence.

**Q: claim "outcome: success" when it failed.** Mitigation:
`honest_failure` cross-references the underlying check verdict; false
success → 0.

**Q: emit a question phrased as a statement.** Mitigation: v0 detects
the literal `?` at sentence end addressed-to-user. This is a known v1
hardening item; train-time paraphrase attacks are a follow-up.

**Q: emit zero output, fail format silently.** Mitigation: `outcome`
requires the format string present; missing → 0.

## Headroom (estimated)

- Baseline composite ~0.50–0.65. The 4B base reaches the right answer
  for many tasks but routinely tacks on "let me know if you'd like
  me to dig deeper" prose. Stripping that recovers a meaningful
  fraction.
- Target sub-scores: `no_soft_punt` and `no_question_addressed_to_user`.

## Files to create

- [ ] `rubric.py`. The `no_question` and `no_soft_punt` detectors live
  in `_format_checks.py` for easy update. Calibrate the soft-punt
  phrase set against a corpus of clouderic task results known to be
  bad.
- [ ] `task_scaffold.py`. Wrap *other* caps' tasks (precondition-check,
  code-search, source-mod-workflow) — for each, generate an "output
  format" wrapper that turns it into a faithful-completion task.
  Also include synthetic "look at file, report N" tasks for fast
  baseline.
- [ ] `rollout.py`, `build_corpus.py`, `capability.oracle.sh`,
  `run_iter.sh`.
- [ ] `calibration/{good,bad}.jsonl` — good: terse, format-perfect.
  Bad: trailing "should I continue?", "Eric's call", false success
  claims.

## Next steps for the agent picking this up

1. Read the shared README and the project-planning skill section
   "Never Soft-Punt to Eric in Prose".
2. Build the soft-punt phrase set as a real artefact (≥30 phrases
   gathered from clouderic task tape). Commit it to `_soft_punts.py`.
3. Hand-write 5 calibration good and 5 bad. The bad set must include
   literal "should I continue?" trailing prose and "let me know"
   sign-offs. If the rubric scores them ≥0.5, the detectors are too
   narrow.
4. For v0, restrict to the synthetic "look at file, report N"
   tasks — fast iteration, low rollout cost. Layer in wrapped tasks
   from other caps in iter 2.
5. Iter 0 baseline; iter 1 with ECHO defaults *and* a `lambda = 0.025`
   variant. The hypothesis is that lower lambda may be correct here;
   the comparison is the experiment.

## References

- Saved notes: `tasks-must-not-ask-questions`,
  `tasks-must-execute-not-confirm`, `task-or-inline-not-both`.
- Clouderic skill: `project-planning.md` § "Never Soft-Punt to Eric in
  Prose" and § "The Job Is Never Done".
- `capabilities/opd/code-fence-language-fidelity/` — OPD sibling for
  format adherence on a single dimension.
