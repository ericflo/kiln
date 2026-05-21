# pi-shell-hygiene — long-running process safety

**Status:** Scaffold. **Rank 6/10**. Trains the agent's shell-execution
discipline: background long jobs, set timeouts, capture logs to files,
detect wedges, clean up on failure — and never write
`until ssh ... kill -0` style polling loops that have cost real money
in the kiln work.

**Goal.** Tasks that involve a long-running command (build, bench,
deploy probe, watcher) complete within budget, leave no orphan
processes, and clean up resources on cancel or error. The agent
prefers the `bg + wait-file --timeout` pattern over blocking SSH
polling.

## Why this capability

Two real incidents, both 2026-04-20:

- Task `c8a185469bcd372de6718b75`: **$13.76** burned in 1h40m polling a
  wedged RunPod sshd with `until ssh $pod "kill -0 $pid"; do sleep 10`.
- Task `dd4948c46c35c942d374f45f`: **$99.76** burned in 3h40m on
  *the second instance of the same pattern that same day*.

The kiln skill calls these out specifically:

> **NEVER write any of these patterns against a RunPod pod:**
>
> ```bash
> until ssh $pod "test -f /tmp/done"; do sleep 5; done    # ❌ BANNED
> while ssh $pod "kill -0 $pid" 2>/dev/null; do sleep 10; done    # ❌ BANNED
> ```

Saved notes: `kiln-ssh-polling-deadlock`, `blocking-ssh-poll-hangs-silently`.

This cap trains the *positive* pattern (bg + wait-file + trap cleanup)
so future agents reach for it by default.

## Task shape

Each task is `(workspace, command_brief, gold_checks)`:

- **Workspace** — a small project under `$SANDBOX/repo/` with a
  long-running command to invoke (configurable sleep, build, or bench
  stand-in).
- **Command brief** — natural-language description of what to run, how
  long the budget is, and what artifact signals completion.
- **Gold checks** —
  - `completed_within_budget`: the artifact appeared before the
    budget expired.
  - `no_orphan_processes`: after the agent exits, no leftover
    background processes own the workspace's PID file.
  - `cleanup_on_failure`: when the command is configured to fail or
    hang, the agent invokes its cleanup path (releases lease,
    terminates pod-stand-in).
  - `no_antipattern_calls`: no `until ssh ...`, no `while ssh ... sleep`,
    no `sleep $LONG &&` patterns.

Example prompt:

```
Build the project in /workspace/myrepo. It takes ~90s. Background it,
wait for /workspace/myrepo/.build-done to appear (max 180s), then
print the contents of /workspace/myrepo/build.log. If the build fails
or the timeout expires, kill the process and exit non-zero.
```

Correct trajectory:

```
bash: cd /workspace/myrepo && trap '...' ERR INT TERM \
       && nohup bash -c 'make all 2>&1 > build.log; touch .build-done' &
bash: pid=$!; echo $pid > .build.pid
bash: timeout 180 bash -c 'until [ -f .build-done ]; do sleep 5; done' ; \
       cat build.log
```

(Or, more idiomatically, the agent uses `ce kiln-pod-acquire`-style
helpers if present in the workspace.)

## Rubric design (v0)

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor | Artifact appeared within budget AND command exit code was correct. | `touch .build-done` directly with no build → log content mismatch → 0. |
| `no_orphan_processes` | 0.20 | After the agent exits, all PIDs from the rollout's process group are dead. Checked by walking `/proc/<pid>` with the rollout group id. | `disown` followed by exit — orphans become children of init; check by PGID matching. |
| `cleanup_on_failure` | 0.15 | For failure-injected tasks: the agent's trap or cleanup logic ran (presence of `cleanup_ran` flag file). | Pre-creating the flag — oracle re-creates the workspace and re-runs from scratch. |
| `no_antipattern_calls` | 0.10 | 1.0 iff no command in the session matches the banned regex set: `^\s*(until\|while)\s+ssh\b.*(kill -0\|test -f\|grep)`, `sleep\s+\d+\s*&&\s*ssh`. Penalty `-0.5` per match. | Putting the antipattern in a heredoc — regex matches inside string bodies too. |
| `format_compliance` | 0.05 | Final assistant turn reports the timing summary: `(command, wall_clock_s, exit_code, artifacts)`. | Free-form prose → 0. |

**Composite** = `outcome × (0.20·no_orphans + 0.15·cleanup + 0.10·no_antipattern + 0.05·format + 0.50·base)`

## ECHO recipe

**Strong fit.** Shell output (build logs, status lines, exit codes) is
rich env signal. Predicting "what will `ps` print after I start this
job?" or "what will `cat build.log` show?" is exactly the skill
distinguishing a careful long-running-process operator from a
brittle one.

## Hypotheses

- **H_wedge_injection** — tasks where the long-running command hangs
  (sleep infinity). Tests whether timeouts fire and cleanup runs.
- **H_cancel_during_wait** — the rollout harness sends SIGTERM to the
  pi process mid-wait. Tests whether the trap fires before exit. (v2+
  — requires harness changes.)
- **H_unknown_artifact** — agent has to discover *which* file signals
  completion (no explicit `wait-for` path given). Tests `inotifywait`
  or polling-with-bound usage.
- **H_resource_release** — task has an associated "lease" file the
  agent must release in cleanup; tests resource-handle hygiene.

## Adversarial design (§0)

**Q: skip background, run in foreground.** Mitigation: if the task's
budget < command duration, the foreground hang exceeds the rollout
wall-clock; outcome 0.

**Q: bg without trap, exit clean if happy path.** Mitigation:
`cleanup_on_failure` is gated to failure-injected tasks; without a
trap, the cleanup_ran flag never appears.

**Q: claim completion without checking the artifact.** Mitigation:
`outcome` checks the artifact contents, not just existence.

**Q: agent shells out to a subprocess and writes the antipattern there.**
Mitigation: regex check walks all bash commands in the session
including heredocs.

## Headroom (estimated)

- Baseline composite ~0.30–0.45. The 4B model can `nohup &` correctly
  but usually misses traps and frequently writes blocking `until ssh`
  patterns when prompted to wait on remote state.
- Target sub-score: `no_antipattern_calls` and `cleanup_on_failure`.

## Files to create

- [ ] `_antipatterns.py` — sibling module holding the banned-pattern
  regex list (so the set is easy to grow without editing rubric.py).
- [ ] `fake_pod/` — local SSH-shim controller. A small script that
  exposes a `ssh fake-pod <cmd>` interface execing into a separate
  PGID and can be made to wedge on demand. Required before SSH-style
  tasks can be trained safely (real SSH would consume pod budget).
- [ ] `rubric.py`. The `no_orphan_processes` check requires running
  the rollout under a fresh process group (`setsid`); record the PGID
  in the rollout meta. Imports the regex set from `_antipatterns.py`.
- [ ] `task_scaffold.py`. Synth tasks vary command duration, failure
  mode (clean / fail / hang), artifact name, and budget. Mix of local
  and "ssh-style" tasks (using the `fake_pod/` shim, not real SSH).
- [ ] `rollout.py`. Each rollout runs in its own PGID; oracle checks
  PGID after pi exits.
- [ ] `build_corpus.py`, `capability.oracle.sh`, `run_iter.sh`.
- [ ] `calibration/{good,bad}.jsonl` — good: bg-trap-wait-cleanup. Bad:
  `until ssh ... kill -0; sleep 10` and friends.

## Next steps for the agent picking this up

1. Read shared README and the kiln skill section
   "money-burning anti-patterns" (it's the canonical source for the
   antipattern regex set).
2. Build the fake-pod controller first — without it you can't safely
   train SSH-shaped tasks. Recommended: a Python script that simulates
   ssh-to-pod by execing a local subprocess in a separate PGID, and
   can be made to "wedge" by sleeping.
3. Calibration: include literal SSH-polling antipatterns in the bad
   set. If your rubric scores them ≥0.5, your antipattern regex set is
   too narrow.
4. Iter 0 baseline; iter 1 with ECHO defaults. Watch for whether the
   model learns the `bg + wait-file` idiom — it's a 2-3 token signature
   in the trajectory.

## References

- Kiln skill: `### 1. SSH polling loops (BANNED — $13.76 + $99.76
  incidents in one day, 2026-04-20)`.
- Saved notes: `kiln-ssh-polling-deadlock`,
  `blocking-ssh-poll-hangs-silently`.
- `deploy/runpod/kiln-setup.sh` — the canonical `bg + wait-file`
  pattern (use it as the gold positive example).


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

Highest-leverage improvements:

1. **Paired examples are non-negotiable.** Eval and calibration must
   include both positive and negative examples of long-running-process
   patterns. Without paired examples the rubric can't measure
   *discrimination*; it only measures whether the model produces some
   pattern.

   Positive (must rate high):
   - `python3 $RP bg <pod> /tmp/log.txt '<cmd>' && python3 $RP wait-file <pod> /tmp/sentinel --timeout 1800`
   - `nohup <long-cmd> > /tmp/out.log 2>&1 &; disown`
   - `trap 'cleanup' ERR INT TERM`

   Negative (must rate low):
   - `until ssh $pod "test -f /tmp/done"; do sleep 5; done`
   - `while ssh $pod "kill -0 $pid" 2>/dev/null; do sleep 10; done`
   - `bare trap ... EXIT` (kills pod on every tool-call shell exit)
   - sleep-then-poll without timeout

2. **Anchor against the kiln-skill anti-pattern doc.** The
   `capabilities/lib/agentic-grpo-notes.md` reference (and the kiln
   skill body in clouderic) enumerate exactly these patterns. Make
   the rubric scoring align with that doc rather than inventing
   new categories.
3. **Cross-domain robustness.** Train on bash; eval on PowerShell /
   fish equivalents. If the recipe generalizes, the behavior is
   real shell-hygiene, not bash-syntax memorization.
