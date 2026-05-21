# pi-source-mod-workflow — clone→branch→edit→test→PR end-to-end

**Status:** Scaffold. **Rank 9/10**. Trains end-to-end execution of the
canonical source-modification workflow without skipping required steps.

**Goal.** Given a source-mod task spec, the agent executes the required
N-step workflow in order — clone with auth, branch, edit named files,
run the named tests, stage, commit, push, open PR, file deploy request
(if applicable), and watch post-merge CI — without skipping any anchor
step, without reordering them illegally, and producing the required
output identifiers at the end.

## Why this capability

Straight from the `self-modification.md` skill:

> ⚠️ CRITICAL: Every PR MUST be followed by `ce deploy-request`
> (step 8) and post-merge CI verification (step 9). The #1 agent
> failure mode is skipping deploy requests; the #2 is not verifying
> CI after merge.

The clouderic failed-task tape includes many "PR opened, no deploy
request filed" results. The visible task succeeds (PR exists) but the
contractual workflow is broken — the PR can't ship.

This is shape-of-trajectory training, not isolated skill training. The
model has to *sequence* a known set of steps correctly under real
filesystem / git / gh state.

## Task shape

Each task is `(workspace_spec, edit_spec, gold_step_sequence,
required_identifiers)`:

- **Workspace spec** — repo URL (always a sandbox fork in training,
  never the real ericflo/kiln), branch name to create, auth token in
  env.
- **Edit spec** — files to touch, the concrete edits, the test command
  to run.
- **Gold step sequence** — the canonical step list this task requires.
  Two flavours:
  - "kiln-style" — clone → auth → branch → edit → test → commit →
    push → PR. No deploy request, no CE: prefix.
  - "clouderic-style" — adds → deploy-request → post-merge CI watch.
  - "skip-allowed" — some steps are explicitly optional (e.g. tests
    when the task is doc-only). The gold spec marks them.
- **Required identifiers** — `(PR URL, PR number, deploy_request_id?,
  post_merge_ci_status?)` that must appear in the final assistant turn.

Example prompt (kiln-style):

```
In the ericflo/kiln-sandbox-1234 repo, create a branch
`add-foo-docs`, add a paragraph to README.md under "## Quick start"
explaining that `kiln serve` listens on port 8420, push, and open a
PR with title "docs: clarify default port". This is a doc-only change;
no test command required.
```

Correct trajectory:

1. `bash`: `GITHUB_TOKEN=... git clone ... .`
2. `bash`: configure remote URL with token, set user.email/name.
3. `bash`: `git checkout -b add-foo-docs`.
4. (Edit README.md via `write`.)
5. `bash`: `git add README.md && git commit -m "..."`.
6. `bash`: `git push origin add-foo-docs`.
7. `bash`: `gh pr create --title "docs: clarify default port" --body "..."`.
8. Final assistant turn: PR URL, PR number, "deploy request: not
   applicable (kiln)".

## Rubric design (v0)

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor | All required identifiers are present in the final assistant turn AND the PR actually exists on the sandbox GitHub. | Echoing fake PR URLs — oracle checks GitHub via the API and verifies the URL resolves. |
| `step_sequence_match` | 0.20 | Damerau-Levenshtein-like distance to the gold step sequence (using the step taxonomy: clone, auth, branch, edit, test, commit, push, pr_create, deploy_request, ci_watch). Distance 0 → 1.0; per-step penalty 0.1; capped at 0. | Inserting harmless extra reads is fine (only the canonical steps count); inserting `disable-tests` or `--force-push` is caught by `no_dangerous_op`. |
| `anchor_steps_present` | 0.15 | Required anchors present for the task type: for clouderic-style, `deploy_request` *must* appear; for any style, `pr_create` *must* appear post-`push`. | Reordering deploy_request before push fails — anchors check ordering. |
| `no_dangerous_op` | 0.10 | Penalty `-0.5` per banned op: `push --force` to main, `git reset --hard origin/main` without prior backup, `--no-verify`. | Renaming the flag — git op detection works at the command-token level. |
| `identifier_format` | 0.05 | Required identifiers appear in the canonical `PR: <url>` / `PR number: <N>` / `Deploy request: <id|n/a>` format in the final turn. | Free-form prose containing the URL → identifier_format = 0 unless exact format matches. |

**Composite** = `outcome × (0.20·sequence + 0.15·anchors + 0.10·no_danger + 0.05·format + 0.50·base)`

## ECHO recipe

**Strong fit.** Every git, gh, and shell command produces predictable
env output. Learning to model "what does `git push origin <branch>`
print when the branch tracks correctly?" is exactly the skill that
makes the agent confident enough to chain the next step without
re-checking everything.

## Hypotheses

- **H_failure_injection** — task family where one of the steps fails
  the first time (network blip, push rejected for stale branch).
  Tests whether the agent recovers vs gives up. The right behaviour
  is "pull --rebase, retry push," not "give up" and not "force push."
- **H_stale_state_recovery** — workspace is left in a stale state
  (uncommitted changes from a previous session). Tests whether the
  agent stashes/cleans vs steps on the prior state.
- **H_branch_collision** — branch already exists upstream. Right
  behaviour: rename branch or `--force-with-lease` on the rename.
- **H_doc_only_skip** — for doc-only tasks, the test step is optional.
  Tests whether the model correctly recognises and skips.

## Adversarial design (§0)

**Q: lie about the PR URL.** Mitigation: oracle hits the GitHub API
to verify the URL resolves to a real PR matching the task title.

**Q: skip deploy_request on clouderic-style task.** Mitigation:
`anchor_steps_present` requires it; outcome cross-checks against the
declared task style.

**Q: `git push --force` to fix a state issue.** Mitigation:
`no_dangerous_op` penalises force push (except `--force-with-lease`,
which is allowed and detected at the flag level).

**Q: chain `&&` everything into one bash call.** Mitigation: the step
sequence is parsed from the agent's bash call bodies; a single
`clone && checkout && edit && push && pr_create` chain still counts as
all the steps. **This is allowed.** What we want is correct
*sequencing*, not a particular call structure.

**Q: rely on stale local clone for clouderic source-mod when the task
says `clone`.** Mitigation: the rollout sandbox is fresh; there's no
prior clone to lean on.

## Headroom (estimated)

- Baseline composite ~0.35–0.50. The 4B model can do clone-edit-push
  reliably on simple tasks; loses badly on deploy_request and
  post-merge CI watch (the steps it routinely skips).
- Target sub-score: `anchor_steps_present` (the missing-step issue).

## Files to create

- [ ] **Sandbox provisioning** — *before* writing any other file:
  provision a dedicated GitHub org or account (`ericflo-sandbox/`
  or similar) and a PAT scoped *only* to that namespace. Export it
  as `KILN_SANDBOX_GITHUB_PAT`. **Do NOT run training with the
  production `GITHUB_TOKEN`** — a runaway rollout could open hundreds
  of PRs against real repos. Document the kill switch (revoke PAT)
  prominently in `rollout.py` once written.
- [ ] `_steps.py` — step-taxonomy tokeniser that reads bash bodies and
  emits a stream over `{clone, auth_config, branch_create, edit, test,
  commit, push, pr_create, deploy_request, ci_watch}`. Unit-test it
  exhaustively before depending on it from rubric.py.
- [ ] `rubric.py`. Imports the tokeniser from `_steps.py`.
- [ ] `task_scaffold.py`. Per-rollout sandbox repo lifecycle (create,
  delete after eval); rate-limit-aware (gh API budget).
- [ ] `rollout.py`. Refuses to run if `KILN_SANDBOX_GITHUB_PAT` is
  unset or equals the production token (string equality + scope
  check).
- [ ] `build_corpus.py`, `capability.oracle.sh`, `run_iter.sh`.
- [ ] `calibration/{good,bad}.jsonl` — good: full 8-step kiln-style,
  full 9-step clouderic-style. Bad: skip deploy_request; force push;
  skip post-merge CI.

## Next steps for the agent picking this up

1. Read shared README and `self-modification.md` from the clouderic
   skills tree — that file is the authoritative source for the
   workflow this cap trains.
2. **First task: build the safe sandbox.** Provision an
   `ericflo/kiln-sandbox` org or use a personal account; create a
   PAT scoped *only* to that namespace. Document the kill switch
   (revoke PAT) prominently in `rollout.py`.
3. Build the step-sequence tokeniser early and unit-test it
   exhaustively. It's the heart of the rubric and will eat the most
   debugging time.
4. Use kiln-style tasks for the corpus seed (8 steps, no deploy
   request) — they're cheaper to evaluate. Add clouderic-style tasks
   only after the kiln-style rubric is stable.
5. Iter 0 baseline; iter 1 with ECHO defaults. Watch for whether the
   model starts emitting deploy_request without prompting — that's the
   key signal.

## References

- Clouderic skill: `self-modification.md` — authoritative workflow.
- Clouderic skill: `project-resources.md` — auto-merge handling.
- Saved notes: `task-completion-must-reach-canonical-repo`,
  `source-mod-verify-repo-before-editing`.
- kiln PR conventions (no `CE:` prefix, no deploy-request) —
  `capabilities/lib/agentic-grpo-notes.md` and the kiln skill section
  "PR conventions (kiln ≠ clouderic)".


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
Round 1 status: **scaffold; flagged as integration-test material**.

**Reframe for round 2: this cap is an integration test, not a
training cap.** The full clone → branch → edit → test → push → PR
sequence is too long and multi-turn to get reliable GRPO signal as a
single cap. Three options were considered:

- (a) Split into sub-caps (`pi-clone-branch`, `pi-edit-test`,
  `pi-push-pr`). Three new caps to maintain, partial overlap with
  existing caps.
- (b) Keep as single training cap. Round-1 evidence suggests the
  signal-to-noise ratio is too low.
- (c) **Reframe as integration test under `capabilities/integration/`**.
  Use this cap to *measure* whether an adapter trained on
  pi-context-aware-edits + pi-test-interpretation + pi-faithful-completion
  composes correctly to handle the full workflow.

**Decision: option (c)**. This cap is renamed conceptually to
"source-mod workflow integration test." It still has rollout.py /
rubric.py / capability.oracle.sh because we *evaluate* on the full
workflow, but it does NOT have a meaningful `run_iter.sh` that trains
a new adapter — instead, `run_iter.sh` runs the workflow against the
*latest cross-cap-coherence adapter* and reports pass-rate.

See `capabilities/integration/cross-cap-coherence/` for the integration
adapter; this cap is one of its eval-bucket members.
