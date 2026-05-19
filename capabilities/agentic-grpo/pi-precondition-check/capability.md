# pi-precondition-check — read-before-write discipline

**Status:** Scaffold. **Rank 1/10** in the agent-capability decomposition
done on 2026-05-19 against the clouderic task corpus (~7,800 tasks).
This is the single largest failure cluster: edits made against stale,
hallucinated, or already-shipped state.

**Goal.** Before any mutating tool call (`write`, `edit`, mutating
`bash` like `git commit` / `rm` / `sed -i`), the agent has performed a
*relevant* read that verified the assumption it is about to act on. If
the assumption fails (file moved, function renamed, bug already fixed,
PR already open), the agent exits cleanly with a "precondition_failed"
verdict rather than producing a phantom edit.

## Why this capability

The clouderic notes index has six saved corrections matching this
pattern. Direct quotes from the notes:

- `verify-before-implementing` — "Read current file state before
  implementing any fix."
- `verify-bugs-before-fixing` — "When a task describes a bug with a
  specific root cause and fix, always verify the bug actually exists
  before [fixing it]."
- `verify-data-shape-before-implementing` — "Task specs … reference
  field names or data structures that don't match the actual data."
- `verify-architecture-claims-in-source` — "Always verify architecture
  claims by reading the specific source file, not inferring from
  adjacent code."
- `verify-app-changes-persisted` — "Always verify /data/apps/ edits
  persisted by inspecting live code, not just task results."
- `source-mod-verify-repo-before-editing` — "Verify clone target repo
  and file paths before editing on source-mod tasks."

The dominant failure mode in clouderic's failed-task tape is some
variant of *"the task said X, the agent did X, X was no longer
true."* This cap targets that head-on.

## Task shape

Each training task is a `(workspace, claim_set, ground_truth)` triple:

- **Workspace** — a small repo snapshot under `$SANDBOX/repo/`.
- **Claim set** — the task prompt asserts ≥1 thing about the workspace:
  a file path, a function name, a current-state property ("`bar()`
  returns `None` when `x == 0`"), or a remaining-work claim ("PR #123 is
  open and unmerged").
- **Ground truth** — for each claim, the dataset records whether it
  *holds in the workspace right now*. The training set is balanced
  ~50/50 between holds-true and stale claims.

Example prompt to pi (stale-claim case):

```
The `compute_hash` helper in `lib/util.py` uses MD5. Migrate it to
SHA-256 by changing the import and the `hashlib.md5(...)` call.
```

In this workspace, `lib/util.py` already uses SHA-256 (the migration
shipped last week). The correct trajectory is:

1. `bash`: `grep -n compute_hash lib/util.py` (or `Read lib/util.py`).
2. Observe: the function already calls `hashlib.sha256(...)`.
3. Emit a final assistant turn explaining the precondition failed and
   exit. No `write`/`edit` call.

Example prompt (holds-true case): same prompt against a workspace where
`lib/util.py` actually uses MD5. The correct trajectory verifies, then
makes the migration.

## Rubric design (v0)

Multi-component, modelled on pi-doctest §rubric-v1.

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor | For holds-true tasks: post-edit assertion holds (gold check). For stale tasks: the agent emits the `precondition_failed` sentinel **and** does not mutate the workspace. | Empty workspace (gold check fails); spurious sentinel on holds-true tasks (post-edit gold fails). |
| `verified_before_mutation` | 0.25 | 1.0 iff every mutating tool call is preceded in the same session by a read whose result content overlaps the claim's referenced byte range (file path, function name, …). Computed by replaying the session and matching reads to mutations by file path. | Reading a tangential file (overlap check requires the claim's path/symbol). |
| `staleness_correctly_detected` | 0.20 | For stale tasks: 1.0 iff the final assistant turn carries the `precondition_failed:{claim_id}` token sequence. For holds-true tasks: 1.0 iff it does *not*. | Always emitting `precondition_failed` (holds-true tasks score 0); never emitting it (stale tasks score 0). |
| `no_phantom_edit` | 0.10 | 1.0 iff no `write`/`edit` referenced a path or symbol that does not exist in the workspace at edit time. | Reading the path first only changes `verified_before_mutation`; an `edit` against a non-existent path still scores 0 here. |
| `format_compliance` | 0.05 | Same shape as pi-doctest: well-formed tool-call XML. | Malformed blocks → 0. |

**Composite** = `outcome × (0.25·verified + 0.20·staleness + 0.10·no_phantom + 0.05·format + 0.40·base)`

Range: `[0, 1]`. `outcome` gates the agentic component so a wrong fix
on a holds-true task scores 0 regardless of how clean the trajectory
was.

## ECHO recipe

**Why ECHO fits here strongly.** The dominant gradient signal is "what
does the file actually say?" — exactly the env-side tokens. When the
model reads a file, the response tokens are env observations the model
should predict; learning to predict them is learning to anchor its
beliefs in what reading would reveal.

Trainer defaults apply (see shared README). Keep `no_policy_loss = false`
— the verifier is informative; GRPO and ECHO co-train here.

**Hypothesis to try in iter 2+:** raise `echo.lambda` to `0.10` for the
first ~50 steps to bias hard toward reading, then anneal back to
`0.05`. The paper §3.3 productive range tops out at `~0.05`, but for a
cap that's specifically about anchoring-in-env, the upper band may be
correct early.

## Hypotheses (for later iters)

- **H_explicit_sentinel** — make the rubric reward the literal token
  sequence `precondition_failed:` only, not a paraphrase. Predicts:
  format compliance separates cleanly; outcome harder to cheat.
- **H_balanced_split** — try the 30/70, 50/50, and 70/30 stale/holds-true
  splits. Predicts: 50/50 dominates; the asymmetry teaches the
  no-default-action habit.
- **H_phantom_path_bait** — task set includes claims that reference
  paths that *do exist* but with one component renamed
  (`lib/utils.py` vs `lib/util.py`). Tests fine-grained path
  verification.
- **H_meta_claim** — claims about open PRs / pending tasks
  (`gh pr list --search "X"` should return nothing). Tests
  verification against systems the agent can't `Read` directly.

## Adversarial design (§0) — cheat-resistance

**Q1: cheapest way to score 1.0 on stale tasks?** Emit
`precondition_failed:claim_0` unconditionally without reading.
Mitigation: `verified_before_mutation` scores 0 if no read happened
before the assistant emits the sentinel — replay the trajectory and
check the read-before-claim ordering. (The "before-mutation" check is
extended to: before any *resolution*, mutating *or* sentinel.)

**Q2: cheapest way to score 1.0 on holds-true tasks?** Always do the
edit; never verify. Mitigation: `verified_before_mutation` scores 0
if no relevant read precedes the edit. Composite gets multiplied down.

**Q3: read everything then guess.** Mitigation: not strictly cheated
against in v0 — over-reading is allowed. v1 may add a
`tool_call_efficiency` term capped at `target=2` reads (read the
asserted file + maybe one neighbor).

**Q4: memorise the stale/holds-true balance of the task set.** A model
that emits sentinel on 50% of tasks uniformly hits the right marginal
but fails on per-task accuracy. Mitigation: `staleness_correctly_detected`
is a per-task accuracy not an aggregate.

## Headroom (estimated, to be measured in iter 0)

The base 4B model has *no* verify-before-mutate prior — it tends to
trust the prompt. Expect:

- Holds-true tasks: baseline composite ~0.55–0.70 (the model writes the
  edit but doesn't verify, scoring 0 on `verified_before_mutation`).
- Stale tasks: baseline composite ~0.05–0.20 (the model makes a phantom
  edit, scoring 0 on `outcome`).
- Target sub-score: `verified_before_mutation` (highest movable mass).
- Composite group-variance stdev: expect 0.25+ — wide.

## Corpus design

`build_corpus.py` should produce ≥100 training tasks and ≥40 held-out
eval tasks across these claim families:

1. **File-exists / file-path** — `path/to/X.py` exists, possibly stale.
2. **Symbol-defined** — function `f` exists at module `m`, possibly stale.
3. **Symbol-content** — function `f` does `Y`, possibly stale.
4. **Test-state** — test `t` currently fails / passes.
5. **External-state** — PR #N is open / merged / closed.

Each family should have ≥20 training tasks balanced stale/holds-true.

## Files to create (TODO for the spawning agent)

- [ ] `rubric.py` — composite reward per the table above. Reuse the
  trajectory-replay helpers from `lib/pi_trajectory.py`. The
  `verified_before_mutation` check requires walking the session JSONL
  in order — see `pi-doctest/rubric.py::tool_call_efficiency` for the
  iterator pattern.
- [ ] `task_scaffold.py` — synth-and-real generator. Reuse repo
  fixtures from `kiln/tests/fixtures/repos/` if available; otherwise
  build a `tmpdir` repo from a parametric template.
- [ ] `rollout.py` — pi runner. Mirror `pi-doctest/rollout.py` but
  expose `claim_id` in the task spec so the rubric can correlate.
- [ ] `build_corpus.py` — produces `datasets/train.tasks.jsonl` and
  `datasets/eval.jsonl` (gitignored).
- [ ] `capability.oracle.sh` — blind eval scoring for a given adapter.
- [ ] `run_iter.sh` — full iter recipe (rollouts → train → eval).
- [ ] `calibration/{good,bad}.jsonl` — 5 hand-written good
  trajectories (read-then-edit / read-then-sentinel) and 5 bad
  (phantom-edit / unverified-edit / spurious-sentinel).

## Next steps for the agent picking this up

1. Read `capabilities/agentic-grpo/README.md` for the shared workflow.
2. Read this whole file.
3. Read `capabilities/agentic-grpo/pi-doctest/capability.md` for the
   most mature reference and `pi-doctest/rubric.py` for the trajectory
   iteration pattern.
4. Build `calibration/{good,bad}.jsonl` *first* (5 examples each).
   Eyeball-score them before writing `rubric.py`.
5. Write `rubric.py`, run `rubric_sanity.py` against calibration. Good
   should dominate bad with clear separation.
6. Build `task_scaffold.py` + `build_corpus.py` for the file-exists and
   symbol-defined claim families only (smallest viable v0). Defer the
   external-state family to iter 2.
7. Run iter 0 baseline rollouts on 24 eval tasks (12 holds-true, 12
   stale). Append the row to `capability.jsonl`.
8. Iter 1: `cuda_grpo_ablation --max-groups 100` with ECHO defaults.
   Append the post-train row.

## References

- `docs/plans/echo-integration-plan.md` §3.1, §3.4 — ECHO defaults.
- `capabilities/agentic-grpo/pi-doctest/capability.md` — v1 rubric pattern.
- Saved clouderic notes: `verify-before-implementing`,
  `verify-bugs-before-fixing`, `verify-data-shape-before-implementing`,
  `verify-architecture-claims-in-source`, `verify-app-changes-persisted`,
  `source-mod-verify-repo-before-editing`.
