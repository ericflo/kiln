# pi-code-search — find without reading

**Status:** Scaffold. **Rank 2/10**. Every non-trivial coding task starts
with a search, and a 4B model that defaults to `Read whole file` burns
context before it gets to the real work.

**Goal.** Given a code-locate question ("where is `X` defined", "which
files reference `Y`", "what handler responds to route `Z`"), the agent
answers correctly with the minimum bytes of tool output consumed —
preferring `grep`/`glob`/`find` over `Read` on large files, and
choosing the cheapest pattern that distinguishes the right hit.

## Why this capability

Two clouderic patterns motivate this:

1. **Read-the-whole-file default.** Failed-task tape shows agents reading
   2,000-line files when a 60-token grep would answer the question.
   Cost & context-window pressure both blow up.
2. **Wrong-grep-pattern → fallback-to-Read spiral.** When a grep returns
   nothing the agent doesn't refine the pattern — it falls back to
   reading the whole tree.

This cap also bridges to the OPD `code-symbol-extraction` capability:
that one trains "given a file, extract symbols"; this one trains "given
a question about the repo, find the right place to look."

## Task shape

Each task is `(repo_snapshot, question, gold_answer)`:

- **Repo snapshot** — small to medium repo under `$SANDBOX/repo/`
  (~50–5000 files). Drawn from kiln itself, candle, real OSS subsets.
- **Question** — one of:
  - `define:<symbol>` — return `file:line` where the symbol is defined.
  - `refs:<symbol>` — return the set of `file:line` references.
  - `route:<http_path>` — return the handler function `file:line`.
  - `import_chain:<symbol>` — return the chain of files importing this.
- **Gold answer** — the exact set or location. Multiple acceptable
  answers (e.g. several call sites) are scored by F1.

Example prompt:

```
Where is `analytic_grpo_tail_loss_grad_pre_final_norm` defined in
this repository? Answer with the file path and line number.
```

Correct trajectory (3 tool calls):

1. `bash`: `grep -rn "fn analytic_grpo_tail_loss_grad_pre_final_norm" --include="*.rs"`
2. Observe: `crates/kiln-train/src/trainer.rs:7270`.
3. Emit final assistant turn with the answer.

## Rubric design (v0)

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor | F1 between predicted and gold answer set. Tokenize as `(file, line)` pairs; allow line off-by-N (N=2 default) for symbol-on-multi-line cases. | Empty answer → 0. Answers without `file:line` shape → 0. |
| `efficiency` | 0.25 | `1 - clip((bytes_consumed - target_bytes)/(span * target_bytes), 0, 1)`. `target_bytes` is the size of the gold grep output. `bytes_consumed` is the sum of tool-result bytes the model saw. `span` default 5. | Read-and-truncate: rubric uses full result body, not what the model emitted. |
| `tool_choice` | 0.15 | Bonus mass for `grep`/`glob`/`find`/`rg`/`ast-grep` over `Read` on files >2KB. Specifically: 1.0 if no >2KB Read occurred; linear penalty per >2KB Read. | One free Read; subsequent ones penalize. |
| `format_compliance` | 0.05 | Final answer matches the `file:line` regex required by the prompt. | Free-form prose → 0. |

**Composite** = `outcome × (0.25·efficiency + 0.15·tool_choice + 0.05·format + 0.55·base)`

## ECHO recipe

**Strong fit.** `grep`/`glob` output is highly structured and
predictable from `(pattern, repo_layout)` — almost the platonic env
signal. ECHO loss on those env tokens trains the model to *predict
what its search would return*, which is exactly the planning skill we
want: "if I grep for X, will I get a hit?"

Defaults: `loss.echo.lambda = 0.05`, `env_only`, `warning_filter=true`.

**Hypothesis worth testing in iter 2:** raise `echo.lambda` to `0.075`
or `0.10`. Search output is unusually informative-per-token (a single
grep line resolves the whole task), so the upper paper band may be
correct here.

## Hypotheses

- **H_pattern_refinement** — task subset where the first grep returns
  noise; agent must refine (add a type filter, anchor `^fn`, etc.).
  Predicts: agents trained with ECHO refine more readily because they
  better predict what each pattern *would* hit.
- **H_glob_vs_grep_choice** — tasks where `glob` (filename) is clearly
  better than `grep` (content). Tests tool-selection, not just usage.
- **H_lexical_vs_ast** — for symbol-defined tasks, does `ast-grep`
  beat ripgrep? If the repo has `ast-grep` available, allow it and
  reward correct AST-pattern syntax.
- **H_context_budget** — cap the visible tool-result bytes per call.
  Forces multi-call refinement; tests recovery from a truncated view.

## Adversarial design (§0)

**Q: cheapest 1.0 without doing the cap?** Guess `crates/kiln-train/src/trainer.rs:1`
for every symbol. Mitigation: gold set spans the repo; uniform guessing
yields F1 near zero.

**Q: read whole codebase in one tool call.** Mitigation: `efficiency`
penalizes bytes consumed; one whole-repo read tanks the score.

**Q: emit grep call, then ignore output, then guess.** Mitigation:
trajectory replay can detect "output not referenced in answer text" —
v1 adds a `grounding` check that the answer's `file:line` matches a
line present in some tool-result body. v0 doesn't enforce this; it's a
known v1 hardening item.

## Headroom (estimated)

The 4B base does use grep when prompted explicitly, but its default is
`Read`. Expect:

- Baseline composite ~0.55. The model gets some answers right; loses on
  efficiency due to over-reading.
- Target sub-score: `efficiency` (highest movable mass).
- Group-variance stdev: 0.20+ on efficiency, 0.15 on outcome.

## Files to create

- [ ] `rubric.py` — composite. `efficiency` requires summing bytes from
  every tool-result body in the session JSONL. The `tool_choice` check
  is a flag walk over the tool-call name+args.
- [ ] `task_scaffold.py` — generate `(repo, question, gold)` triples
  from a labelled symbol corpus. The kiln repo itself is a good source:
  `git grep -n "^pub fn"` gives a labelled `define:` set; `git grep -rn`
  gives `refs:`.
- [ ] `rollout.py` — pi runner; bound `max_tokens_per_turn` aggressively
  (~512) to discourage long answers.
- [ ] `build_corpus.py` — at least 100 training, 40 eval. Stratify by
  question kind and by repo size.
- [ ] `capability.oracle.sh`
- [ ] `run_iter.sh`
- [ ] `calibration/{good,bad}.jsonl` — 5 good (3-call grep solutions), 5
  bad (whole-file Read solutions).

## Next steps for the agent picking this up

1. Read the shared README and `pi-doctest/capability.md`.
2. Decide on the corpus seed. Recommended: use kiln itself as the
   primary repo and generate `define:` / `refs:` tasks from
   `cargo metadata` + `git grep` output. Add candle as a secondary
   corpus to diversify.
3. Write `calibration/` examples *first*. The bad set should include:
   "Read crates/kiln-train/src/trainer.rs entirely then answer." If
   your rubric ranks that anywhere near the good set, it's broken.
4. Iter 0 baseline on ~24 eval questions, balanced across question
   kinds.
5. Iter 1 with ECHO defaults. If group-variance on `efficiency` doesn't
   move, double the `echo.lambda` for iter 2.

## References

- OPD cap `capabilities/opd/code-symbol-extraction/` — sibling skill at
  single-file granularity.
- `docs/plans/echo-integration-plan.md` §3.4 — ECHO defaults.
