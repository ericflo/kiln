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

## Rubric design (v1 — adopted before iter 0 baseline)

The composite combines an ADDITIVE search-quality term with a
MULTIPLICATIVE grounding factor. Grounding is the single most load-
bearing dimension for this capability — without it the model could
score ≥0.85 just by guessing correctly, which is the opposite of what
we want to train.

| Sub-score | Role | What it measures | Cannot be cheated by |
|-----------|------|-------------------|----------------------|
| `outcome` | hard mult floor | F1 over `(file, line)` tuples with ±2 line tolerance. | Empty / wrong-shape answer → 0. |
| `efficiency` | additive 0.50 | `1 - clip((bytes - target)/(span·max(target, 100)), 0, 1)` with `span=5`. Target = optimal grep output size. | Truncate-and-paste: rubric uses raw tool-result body bytes. |
| `tool_choice` | additive 0.30 | `1.0 - 0.20·n_large_reads`, where large = result body ≥2KB AND tool name ∈ {read, view, cat, Read}. | A few small reads OK; many large ones tank it. |
| `format_compliance` | additive 0.20 | Final answer contains ≥1 `file:line` pair matching `[\w/.\-]+\.<ext>:\d+`. | Prose without `path:line` → 0. |
| `grounding` | mult 0.40 floor | Fraction of predicted `(file, line)` pairs whose `file:line` (within ±2 lines, basename or full path) appears in some tool-result body. | Guess-without-search ≤ 0.40. |

**Composite formula:**

  `agentic = 0.50·efficiency + 0.30·tool_choice + 0.20·format_compliance`
  `grounding_factor = 0.40 + 0.60·grounding`   # range [0.40, 1.00]
  `composite = outcome × grounding_factor × agentic`

When `outcome=1` and all sub-scores=1, `composite=1.0`. When the model
guesses correctly without searching, `composite ≤ 0.40`. When the model
returns nothing or the wrong file, `composite=0`.

### Calibration sanity (committed)

`calibration/{good,bad}.jsonl` + `scripts/rubric_sanity.py`. 5 good
sessions (single-grep, multi-grep, refs, narrow-pattern, with thinking)
all score 1.0. 5 bad sessions:
- `bad-read-whole-file`: 0.18 (eff=0, tc=0.80, grd=0)
- `bad-guess-without-search`: 0.40 (grd=0)
- `bad-empty-answer`: 0.0 (outcome=0)
- `bad-wrong-file`: 0.0 (outcome=0)
- `bad-many-large-reads`: 0.20 (eff=0, grd=0)

Separation good→bad = 0.60. Mean lift = 0.85.

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

## Live iter log (running tally — update after each iter)

| Iter | Slug | Recipe | Composite | Δ vs base | Outcome | Eff | Grd | Calls | Verdict |
|------|------|--------|-----------|-----------|---------|-----|-----|-------|---------|
| 0 | baseline-base | (base model, no adapter) | **0.5432** | — | 0.737 | 0.401 | 0.844 | 4.59 | baseline |
| 1 | h1-fast-recipe | TRAIN_LIMIT=10, FILTER_VAR=0.05, MAX_GROUPS=10, lr=1e-5, rank=16 | 0.5461 | +0.003 | 0.755 | 0.455 | 0.906 | 1.81 | flat composite but **-61% tool calls** + **+0.05 eff** |
| 2 | h2-low-filter | TRAIN_LIMIT=12, FILTER_VAR=0.02 | **0.2490** | **-0.294** | 0.294 | 0.590 | 0.344 | 1.59 | **regression**: train loss spiked at end (1.07→0.66→1.06), 22/32 zeros |
| 3 | h3-no-filter | TRAIN_LIMIT=10, no filter, MAX_GROUPS=10 | **0.3604** | **-0.183** | 0.451 | 0.548 | 0.500 | 1.59 | **regression**: untouched-group training corrupted outcome, 17/32 zeros |
| 4 | h4-tight-filter | TRAIN_LIMIT=20, FILTER_VAR=0.08 | **0.2848** | **-0.258** | 0.331 | 0.570 | 0.406 | 1.38 | **regression**: even tight filter regressed; thesis upset |
| 5 | h5-replay-iter1 | EXACT iter 1 config replay (FILTER_VAR=0.05, TRAIN_LIMIT=10) | **0.2324** | **-0.311** | 0.300 | 0.560 | 0.313 | 1.25 | **iter 1 was a fluke** — replay regressed deeper |

**Five trained iters, every one regressed.** Pattern is highly consistent:
- All regressed iters: `mean_wall_clock` 100-115s on eval (vs base 19s, iter 1 outlier 31s). The trained model is **timing out at 120s**.
- All: outcome 0.29-0.45 (vs base 0.74).
- All: very few tool calls (1.25-1.6 vs base 4.6).

**Root cause (inspected transcript of iter 5 / eval_define_0019):**
Trained model:
1. Issued a malformed first grep (`grep -n "X" repo/` — missing `-r`).
2. Got "Is a directory" error.
3. Retried with `grep -rn "X" repo/` and got the right hit `repo/.../trajectory.rs:74`.
4. **Then never emitted a final answer.** Pi timed out at 120s waiting for the final message.

The training pushes the model toward **thinking instead of answering**: it knows how to grep, finds the answer, then loops in thinking tokens without emitting the final `path:line`. Likely cause: the GRPO advantage rewards small `mean_n_tool_calls` (efficiency sub-score) but does not penalize "thought without exit" — the model finds a local optimum where it thinks indefinitely.

**Pipeline-level question I'm now investigating:** even iter 1's small "win" may have been mostly noise — its mean_wall_clock had already crept from 19s → 31s and the +0.05 efficiency could come from the same "fewer tool calls" drift seen in regressed iters.

**Next directions to test:**
- iter 6: `NO_ECHO=1` — disable env-CE, in case the env-token loss is the driver of the over-think drift.
- iter 7: very low learning rate `LR=2e-6` — minimize policy drift.
- iter 8: tighten the rubric so it explicitly rewards "session ended with a clean file:line text" and penalises thinking-without-exit. The current `format_compliance` only checks the final answer; it doesn't punish never reaching a final answer.

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
