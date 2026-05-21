# pi-test-interpretation — signal vs noise on test & bench output

**Status:** Scaffold. **Rank 7/10**. Trains the agent to read test and
benchmark output correctly: pass/fail/flake classification, median (not
mean) reporting on benchmarks, recognising warmup artifacts, and not
mistaking warnings for errors.

**Goal.** Given a test or benchmark output, the agent classifies each
result, summarises the aggregate accurately, and — for benchmarks —
runs at least 3 iterations and reports the median, not the first run
or the mean.

## Why this capability

Two kiln incidents capture the failure mode:

- **PR #150 false-positive TTFT.** Reported a `20.7×` apparent TTFT
  speedup; turned out to be pure CUDA-graph + JIT warmup. Decode tok/s
  was unchanged. See note `kiln-bench-prefill-warmup-required`.
- **PR #176 closed null at $14.99 burn.** Bench delta was a single-run
  mean within group-variance stdev of the baseline; the underlying
  fusion did nothing measurable.

From the kiln skill's PROFILING.md guidance:

> Bench 3× back-to-back, report median. Run 1 is consistently fastest
> (sccache/graph warmth); it skews means.

This cap trains exactly that habit: never trust a single bench run,
never trust a single mean across 2 runs, distinguish a flake from a
real fail.

## Task shape

Each task is `(test_or_bench_command, raw_output, gold_verdict)`:

- **Command** — pytest / cargo test / cargo nextest / a kiln-bench
  invocation / a microbench script. The agent must run it and read the
  output. For bench tasks the agent is told the budget ("run ≥3
  times").
- **Raw output** — what the command actually prints, including
  warnings, deprecation notices, intermittent flakes.
- **Gold verdict** —
  - Test tasks: `{passed: N, failed: M, flaked: K, skipped: S}` and a
    list of `(test_id, status)` pairs.
  - Bench tasks: `{median_ms: X, p99_ms: Y, runs: [n1, n2, n3, …]}`
    and an interpretation: `improved`, `regressed`, `no_change`,
    `warmup_artifact`.

Example bench prompt:

```
Run `cargo bench --bench decode_paged -- --warm-up-time 1` three
times. Report median tok/s, p99 ITL, and whether the result improves
on a stated baseline (49.76 tok/s, 25.46 ms p99 ITL). Be aware that
run 1 typically benefits from cache + graph warmth.
```

Correct trajectory:

1. Three bench invocations (or a loop that runs three).
2. Per-run extraction of the relevant numbers from output.
3. Compute median across runs, compare against the stated baseline.
4. Final assistant turn: the structured verdict.

## Rubric design (v0)

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor | Verdict matches gold. For tests: pass/fail counts and per-test classification F1. For benches: median within ±5% of gold; verdict label (improved / regressed / no_change / warmup_artifact) matches. | Reporting "all passed" when there are 3 fails → 0. |
| `median_not_mean` | 0.15 | For bench tasks: 1.0 iff the reported number is the per-run median, not the mean and not run 1. Computed by checking which aggregate the reported number equals (within float tolerance) given the recorded per-run values. | Reporting a number that *happens* to equal both mean and median (rare; gold tasks include cases where mean and median differ). |
| `runs_≥3` | 0.10 | For bench tasks: 1.0 iff the agent invoked the bench command ≥3 times in the session. | Calling once with `--repetitions 3` if the bench command supports it — that's fine; the trajectory's tool-call log shows 1 invocation but stdout shows 3 runs, so rubric checks stdout repetition markers, not call count. |
| `warmup_recognition` | 0.10 | For tasks where the first run is a warmup artifact: 1.0 iff the agent's verdict labels it `warmup_artifact` or explicitly drops run 1. | A model that *always* drops run 1 fails the rare task where run 1 is real (e.g. the regression *is* in cold-start). |
| `flake_classification` | 0.10 | For test tasks: F1 on per-test flake labelling. A flake is a test that fails once then passes on rerun; the model is allowed to re-run failing tests. | Marking all fails as flakes → outcome catches it (gold has stable fails). |
| `format_compliance` | 0.05 | Final verdict matches the required schema. | Free-form prose → 0. |

**Composite** = `outcome × (0.15·median + 0.10·runs + 0.10·warmup + 0.10·flake + 0.05·format + 0.50·base)`

The `median_not_mean`, `runs_≥3`, and `warmup_recognition` sub-scores
only apply to bench tasks; for test-only tasks the weight redistributes
to `base`.

## ECHO recipe

**Strong fit.** Test and bench output has highly predictable structure:
pytest summary lines, cargo test status, bench tables. ECHO loss on
these env tokens trains the model's prior over expected formats —
which is exactly the parsing skill we want.

**Hypothesis:** raise `lambda` to `0.075` for bench-heavy task mixes.
Bench output is even more structured than test output; the env signal
is denser.

## Hypotheses

- **H_warmup_only_run1** — bench task families where the first run is
  always the artifact and runs 2–3 are real. Tests warmup recognition.
- **H_real_cold_start** — bench task family where the regression *is*
  in cold-start; runs 2–3 hide it. Tests that the model doesn't
  blanket-drop run 1.
- **H_flake_seeded** — test tasks where 1 of 50 tests is a known flake
  (passes ~50% of the time on rerun). Tests whether the model reruns
  fails before declaring failure.
- **H_warning_vs_error** — outputs with `DeprecationWarning` or
  `RuntimeWarning` in the buffer. Tests warning-vs-error
  classification.

## Adversarial design (§0)

**Q: report run-1 as the answer.** Mitigation: `runs_≥3` checks
invocation count; `median_not_mean` checks the reported number
against per-run values; `warmup_recognition` penalises blind run-1
trust.

**Q: report a constant number that happens to be correct sometimes.**
Mitigation: gold tasks have widely varying ground truth medians.

**Q: declare all tests failed.** Mitigation: outcome F1 on per-test
classification.

**Q: avoid running tests; just guess.** Mitigation: outcome requires
matching ground-truth pass/fail counts — guessing is bounded by the
prior on test outcomes.

## Headroom (estimated)

- Baseline composite ~0.45–0.55. The 4B model reads simple pass/fail
  reliably; struggles with bench median discipline and flake/warmup
  classification.
- Target sub-score: `runs_≥3` and `warmup_recognition`. The model
  *can* call the bench three times if asked; it doesn't *default* to
  it.

## Files to create

- [ ] `rubric.py`. The `median_not_mean` check compares the reported
  number against the set {mean, median, run_1, run_2, run_3} of the
  parsed per-run values; nearest match labels the aggregate. Tasks
  with all three aggregates within float tolerance are excluded from
  the corpus (no signal there).
- [ ] `task_scaffold.py`. Real test/bench corpora: pytest suites from
  small Python projects; `cargo test` and `cargo bench` invocations
  against a Rust toy project; `kiln-bench` invocations against a
  pre-baked checkpoint. Tasks must have *deterministic* gold answers
  per workspace.
- [ ] `rollout.py`, `build_corpus.py`, `capability.oracle.sh`,
  `run_iter.sh`.
- [ ] `calibration/{good,bad}.jsonl` — good: 3-run bench with median
  + warmup recognition. Bad: report run-1 as the answer; mark stable
  fails as flakes.

**Infra note.** Bench tasks have higher wall-clock budget (5+ minutes
per rollout) — config sets `rollout.max_wall_clock_s = 360`.

## Next steps for the agent picking this up

1. Read shared README and the kiln skill section "money-burning
   anti-patterns" (the kiln-bench warmup issue is documented there).
2. Build a small Rust + Python workspace with deterministic
   pass/fail/flake behaviour. The Python side is fastest — `pytest`
   with hand-seeded flake markers.
3. Bench tasks are slower; budget the rollout wall-clock generously.
   Consider faking the bench harness for v0 (use a script that
   produces realistic-looking bench output with hand-set numbers)
   to keep rollout costs low; switch to real `cargo bench` in v2+.
4. Calibration: include the literal PR #176 bench output (or a
   reconstructed version) as a "warmup artifact" gold example.
5. Iter 0 baseline; iter 1 with ECHO defaults.

## References

- `kiln/PROFILING.md` — the canonical source on median-of-3.
- Saved notes: `kiln-bench-prefill-warmup-required`.
- `pi-shell-hygiene/capability.md` — the sibling that trains *running*
  long-running benches correctly; this cap trains *reading* them.


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

**Refined scope for round 2:** this cap is specifically about
*reading noisy/flaky test output correctly*, not about *fixing the
underlying bug*. Bug-fixing belongs to `pi-failure-triage`.

Concrete skills:

- Median-of-3: recognize when a test passes in 2/3 runs and one
  failure was warmup or transient
- Flake classification: distinguish flaky tests from real failures
  by output pattern (timing jitter, race condition signatures,
  network errors)
- Warmup artifact recognition: the first run's TTFT is noise
- Compilation/setup vs test failure: tell apart "build broke" from
  "test broke"

### Highest-leverage improvements

1. **Build a synthetic test-output corpus.** Easier than mining real
   noisy CI logs: generate noise-perturbed pytest output with known
   ground truth (this run *should* be classified as flake / real /
   warmup). Round 2's `kiln rollout` (#34) makes direct HTTP rollouts
   cheap; no pi needed for this cap.
2. **Use direct rollouts, not pi sessions.** This cap doesn't need
   the agentic loop; it's a one-turn classification task. Use
   `kiln rollout` instead of pi for both training and eval — faster,
   cheaper, deterministic.
3. **Chain into pi-failure-triage as input.** The integration track
   should compose: pi-test-interpretation classifies the failure;
   pi-failure-triage fixes the root cause.
