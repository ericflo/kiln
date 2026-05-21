# pi-tool-call-efficiency — few, parallel, terminating

**Status:** Scaffold. **Rank 5/10**. The agentic-quality sub-score that
`pi-doctest` v1 measured found this is the single most movable
sub-score on competent base models: group-variance stdev **0.358** on
`tool_call_efficiency` vs 0.100–0.200 on the others.

**Goal.** For any task the agent solves, make the minimum number of
well-targeted tool calls, parallelise the independent ones, and stop
calling tools as soon as the answer is known. No redundant calls, no
infinite probing, no "let me check one more thing."

## Why this capability

pi-doctest §rubric-v1 §iter-0 measured baseline composite **0.8854**
on humaneval-shaped tasks — the 4B model is *competent at the task*
but burns rollouts via tool-call wastefulness:

| tool-call count | rollout count |
| --- | --- |
| 3–4 (efficient) | 14 |
| 5–9 (moderate) | 5 |
| 13–27 (wasteful) | 4 |
| outcome-fail | 1 |

Four of 24 rollouts used 13–27 tool calls when 3–4 would have done.
Across the clouderic corpus, the same pattern dominates planning
tasks: `ce tasks-list` called four times per planning cycle, `Read`
the same file twice, grep with a wider pattern after the narrow one
already hit.

## Task shape

This cap is a **wrapper rubric** that can run on top of *any* other
cap's task set. Recommended: train it on a mixed task source so the
efficiency signal generalises:

- 40% from `pi-precondition-check` tasks.
- 30% from `pi-code-search` tasks.
- 20% from `pi-doctest` tasks (humaneval-style).
- 10% from `pi-diff-patch-apply` tasks.

For each task, the gold annotation includes `target_tool_calls` (the
minimum call count a clean trajectory needs) and `independent_call_set`
(the set of calls that *could have been issued in parallel*).

## Rubric design (v0)

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor | The underlying task's outcome (whatever the source cap defined). | Zero tool calls (no information; outcome usually 0). |
| `efficiency` | 0.30 | `1 - clip((n_calls - target) / span, 0, 1)` with `target = task.target_tool_calls`, `span = 8`. | A single call that ignores the result and guesses — outcome will catch it. |
| `parallelism_bonus` | 0.10 | Among the `independent_call_set` for this task, the fraction issued as a parallel batch (single assistant turn with multiple tool_calls). | Single-turn answer with no tool calls scores 0 on parallelism (no batches at all). |
| `no_redundant_calls` | 0.10 | 1.0 iff no two tool calls in the session have identical `(name, args_json)`. Decays by `0.2 per duplicate`. | Adding a no-op arg (` `) per duplicate — args are normalised before equality. |
| `terminates_cleanly` | 0.05 | 1.0 iff the final assistant turn carries the answer and no tool call — i.e. the model recognised it had enough information. | Tool call followed immediately by a final answer scores 0; the answer must come *after* receiving the result. |

**Composite** = `outcome × (0.30·efficiency + 0.10·parallelism + 0.10·no_redundant + 0.05·terminates + 0.45·base)`

## ECHO recipe

**Moderate fit.** ECHO biases the model toward predicting tool-result
content, which indirectly teaches "don't call this; you already know
what it'll say." But the efficiency signal itself is a meta-property of
the trajectory, scored by the rubric, not directly by ECHO.
**Important:** keep `loss.no_policy_loss = false` — the GRPO term is
what shapes the efficiency signal; ECHO is a co-trainer here.

## Hypotheses

- **H_parallel_bait** — tasks designed to have ≥3 independent reads
  (read three unrelated files to answer one question). Tests whether
  the model batches.
- **H_short_circuit** — tasks where the first call's result already
  determines the answer; subsequent calls add nothing. Tests whether
  the model stops.
- **H_redundancy_seeding** — feed the agent a buggy harness that
  occasionally returns "transient error" on the first call; tests
  whether the model retries (good) vs spirals (bad). v0 keeps the
  harness clean; this is v2+ work.
- **H_cap_specific_targets** — derive `target_tool_calls` per task
  rather than globally. The precondition-check tasks need 1–2 reads;
  search tasks 1–3 greps; doctest tasks 2–4 calls. Mixing without
  per-task targets is noisy.

## Adversarial design (§0)

**Q: zero tool calls, guess.** Mitigation: outcome floor catches it
(can't do the underlying task without information).

**Q: one whole-repo Read, then guess.** Mitigation: paired with
`pi-code-search`-style efficiency on bytes (see that cap's rubric).
v0 of this cap doesn't gate on bytes; v1 should add it.

**Q: batch identical calls in parallel.** Mitigation:
`no_redundant_calls` normalises args; identical calls don't count as
"parallel diversification."

**Q: emit a final answer in the same turn as a tool call.**
Mitigation: `terminates_cleanly` requires the answer turn to follow a
result turn — the model has to see the result before claiming
termination.

## Headroom (estimated)

pi-doctest §iter-0 already measured headroom on this exact sub-score:
**stdev 0.358** on `tool_call_efficiency` — wider than any other
sub-score in that cap. This is the highest-headroom target in the
agentic-grpo bucket.

- Baseline composite expected ~0.55–0.65 (mixed-source corpus).
- Target sub-score: `efficiency` *and* `parallelism_bonus`.

## Files to create

- [ ] `rubric.py`. Re-uses `lib/pi_trajectory.py` heavily — the rubric
  walks the session JSONL and bookkeeps call counts, identity, batch
  positions.
- [ ] `task_scaffold.py` — wrapper around the source caps. Re-loads
  their corpora and annotates each task with `target_tool_calls` and
  `independent_call_set`. **Annotation is the hard part.** Bootstrap by
  running a strong teacher on each task and counting its tool calls,
  then human-edit ~50 tasks to set ground-truth targets.
- [ ] `rollout.py`. Wrap the source cap's rollout but score with the
  composite efficiency rubric.
- [ ] `build_corpus.py`, `capability.oracle.sh`, `run_iter.sh`.
- [ ] `calibration/{good,bad}.jsonl` — good: 3-call solutions with
  parallel reads. Bad: 15-call solutions, repeated identical calls.

**Infra note.** Per-rollout wall-clock budget should be **lower** for
this cap — long sessions usually mean wasteful sessions. 60s cap is
reasonable for doctest-style tasks; 120s for patch-apply tasks. (The
config defaults to 60s.)

## Next steps for the agent picking this up

1. Read shared README and `pi-doctest/capability.md` §rubric-v1.
2. The hard part: ground-truth `target_tool_calls` per task. Don't
   skip this — uniform `target=3` will overweight tasks that need 5
   and underweight tasks that need 2.
3. Start with a single source cap (`pi-precondition-check`) for v0.
   Once the rubric is stable, expand to mixed sources.
4. Calibration: include trajectories from `pi-doctest`'s iter-0
   "wasteful" group (the 13–27-call sessions) as the bad set. If your
   rubric doesn't score them ≤0.5, it's broken.
5. Iter 0 baseline; iter 1 with ECHO defaults.

## References

- `capabilities/caps/pi-doctest/capability.md` — the source of
  the headroom estimate and the v1 rubric pattern.
- `capabilities/caps/tool-call-arg-fidelity/` — OPD sibling (argument
  *correctness*, not call *count*).


## Round 2 setup

This cap was normalized to the round-2 layout on 2026-05-21. The previous
iter log and writeups are preserved in [`archive/`](archive/). The
`capability.jsonl` starts empty for the new round.

### Kiln features the new round uses

- `kiln adapter verify` (#4) — adapter loadability + behavioral check.
- `cuda_*` trainer `--install-adapter-dir` / `--install-adapter-name` (#5) —
  atomic install into the registry; no more `output/adapter/` symlink bugs.
- `train_receipt.json` (#8) — the canonical per-run artifact with kiln SHA,
  data hashes, hyperparameters, LoRA delta norms, and ECHO metrics.
- `cuda_grpo_ablation --dry-run` (#9) — pre-GPU validation of data, masks,
  base-adapter shape, and saturated-reward warnings.
- `kiln trajectory inspect` (#10) — Rust-native mask + token-count
  diagnostic; replaces the Python `lib/pi_trajectory.py` for new code.
- ECHO observability in receipt (#12) — env-token CE, action-token count,
  warning-prefix masked-out byte count.
- `kiln serve --eval-mode` (#15) — deterministic, no thinking, no
  per-request adapter drift.
- `--adapter-smoke-test` (#19) — post-train base-vs-adapter logit-delta check.
- `--filter-var-min` (#22) — official strong-signal filtering.
- `kiln eval-adapter --seeds N` (#33) — multi-seed paired-eval driver wrapped
  by `capability.oracle.sh`.
- `adapter_manifest.json` + `kiln adapter restore` (#36) — replaces ad-hoc B2
  backup scripts.

### Workflow

```bash
./capability.oracle.sh                     # baseline (no adapter)
./run_iter.sh h1-default-recipe            # first training iter
./run_iter.sh h2-lower-lr                  # subsequent
```

See [`run_iter.sh`](run_iter.sh) for the full pipeline.

## Round 2 improvement plan
Round 1 status: **scaffold**.

**Repurposed for round 2: transfer eval, not training cap.** Tool-call
efficiency is already a sub-score in 4 caps' rubrics (pi-doctest,
pi-code-search, pi-code-comprehension, pi-faithful-completion).
Training a *standalone* tool-call-efficiency adapter would either:

- duplicate signal already present in each cap's rubric, or
- train on synthetic prompts that don't transfer to real tasks.

**Round-2 reshape:** make this an **eval-only** cap that measures
tool-call efficiency *across* the trained adapters from other caps.
Its `run_iter.sh` doesn't train; it runs `kiln eval-adapter` against
multiple adapters and reports the tool-call distribution per adapter.

This becomes a *cross-cap behavioral test*: does training
pi-faithful-completion accidentally hurt pi-doctest's tool-call
efficiency? Today we can't measure that without this cap.

### Concrete shape

- `capability.oracle.sh <adapter1> <adapter2> ...` — eval each
  adapter on a fixed mixed task pool, report:
  - mean n_tool_calls per adapter
  - distribution: efficient (≤4) / moderate (5-9) / wasteful (≥10)
  - delta vs base
- `run_iter.sh` is renamed to `run_eval.sh` (no training step).
- Rubric has only the tool_call_efficiency sub-score; everything
  else is irrelevant for this measurement.
