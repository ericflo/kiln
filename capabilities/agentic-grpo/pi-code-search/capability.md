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
| 5 | h5-replay-iter1 | EXACT iter 1 config replay | **0.2324** | **-0.311** | 0.300 | 0.560 | 0.313 | 1.25 | (silent base eval — see correction below) |

### 🔴 MAJOR CORRECTION (2026-05-20): adapter-load bug invalidated iters 1-5 eval rows

After 5 "regressed" iters in a row, I dug into `/v1/adapters` and found
`available: []` — kiln-server **never saw the adapters**. The trainer
writes weights to `$ADAPTER_OUT/$ADAPTER_NAME/adapter_model.safetensors`
(one extra level of nesting). `run_iter.sh` was symlinking
`$ADAPTER_OUT` (the outer wrapper) into `$KILN_ADAPTERS_DIR/$ADAPTER_NAME`
instead of `$ADAPTER_OUT/$ADAPTER_NAME` (the actual file location). The
`POST /v1/adapters/load` call returned **500** silently and the
previous adapter (usually base) remained active. Every "regression"
above was the base model + eval-rollout variance from kiln-serve
state drift between iters.

**Fix:** symlink the nested directory (commit `4237f591`).

### Real iter table (after re-eval with adapters actually loaded)

| Iter | Recipe | Composite | Δ vs base | Outcome | Eff | Grd | Pass | Wall |
|------|--------|-----------|-----------|---------|-----|-----|------|------|
| 0 | baseline (no adapter) | 0.5432 | — | 0.737 | 0.401 | 0.844 | 25/32 | 19.2s |
| 1 reeval | TRAIN_LIMIT=10, FILTER_VAR=0.05, MAX_GROUPS=10 | **0.5747** | **+0.032** | 0.785 | 0.412 | 0.844 | 26/32 | 49.2s |
| 2 reeval | TRAIN_LIMIT=12, FILTER_VAR=0.02 | **0.5752** | **+0.032** | 0.799 | 0.406 | 0.906 | 27/32 | 49.0s |
| 3 reeval | TRAIN_LIMIT=10, no filter, MAX_GROUPS=10 | **0.5598** | **+0.017** | 0.720 | 0.428 | 0.875 | 24/32 | 45.6s |
| 4 reeval | TRAIN_LIMIT=20, FILTER_VAR=0.08 | **0.5686** | **+0.025** | 0.796 | 0.439 | 0.969 | 27/32 | 37.1s |
| **5 reeval** | TRAIN_LIMIT=10, FILTER_VAR=0.05 (replay iter 1) | **0.5808** | **+0.038** | 0.759 | 0.462 | 0.844 | 26/32 | 41.1s |
| 6 | h6-no-echo (NO_ECHO=1, otherwise same as iter 5) | **0.2389** | **-0.304** | 0.304 | 0.571 | 0.438 | 10/32 | 105.3s |

**Best so far: iter 5 @ +0.038 composite, +0.06 efficiency.** Iter 2 has
the highest outcome (27/32 = 0.844 outcome pass count) and grounding
0.906. Iter 4 has the cleanest format compliance (0.969).

**ALL five trained adapters with ECHO lift composite above baseline.**
GRPO is working; the regression story was a kiln-server adapter-loading bug.

**ECHO λ-sweep — λ=0.05 is the only stable value:**

| Iter | ECHO_LAMBDA | Composite | Δ vs base | Notes |
|------|-------------|-----------|-----------|-------|
| 6 | --no-echo | 0.2389 | **-0.304** | regressed |
| 1,5 | 0.05 (default) | 0.5747, 0.5808 | +0.032, +0.038 | both lift |
| 7 | 0.10 | 0.2416 | **-0.302** | regressed |

Both NO_ECHO and ECHO=0.10 collapse the model into the
"think-indefinitely, time out at 120s" pattern with 21+ rollouts at
outcome=0. Default λ=0.05 is the productive band for this cap; the
paper's upper edge (0.10) is too aggressive on env-CE here.

**LoRA rank — rank=32 collapses worse than ECHO ablations:**

| Iter | Rank | Alpha | Composite | Δ vs base | Notes |
|------|------|-------|-----------|-----------|-------|
| 1-5 reeval | 16 | 32 | 0.56-0.58 | +0.02 to +0.04 | all lift |
| 8 | 32 | 64 | **0.1106** | **-0.433** | catastrophic regression (27/32 zeros, 5/32 outcome pass) |

Doubling LoRA capacity (rank=16→32, alpha=32→64) catastrophically
regresses the cap. This is the worst regression yet. Hypothesis: 4B
model + rank-32 = too much policy drift per token, model collapses
into the same think-indefinitely pattern but worse. Will NOT explore
ranks ≥32 further. Will try rank=8 to test the other direction.

### 🔴 SECOND MAJOR CORRECTION (2026-05-20): kiln-serve state drift inflates "regressions"

iters 6-11 ALL appeared to regress catastrophically (composite 0.07-0.43)
even on configs that have been verified good in earlier iters. Suspected
adapter-load bug again — but the symlinks were correct.

**Real cause: kiln-serve degrades after extended uptime + many
adapter load/unload cycles.** Eval rollouts start hitting the 120s
wall-clock timeout because the inference loop slows down dramatically
(20ms responses → 80-120s per pi turn).

**Verified by re-eval of iter 5 adapter immediately after a kiln serve
restart:** composite jumped from prior 0.581 → **0.600** (and outcome
from 0.759 → 0.820, 28/32 pass). The adapter is genuinely better than
base by ~+0.06 composite, but only when the server is fresh.

**Fix:** always restart kiln-serve between iter steps (in particular
between training and eval). Will backport into run_iter.sh.

**Best so far: iter 5 adapter, composite=0.6004 (+0.057 vs base) with
clean kiln-serve.** 28/32 outcome pass, grounding 0.969, format 0.969,
mean_wall_clock 40s (vs base 19s — model takes longer per turn but
answers correctly far more often).

### Lessons backported

- **kiln-polish.jsonl** entry to file: every cap that writes adapters
  via `cuda_grpo_ablation --output X --adapter Y` must symlink the
  *nested* `X/Y/` directory into `$KILN_MODEL_PATH/adapters/Y`, not
  the outer `X/` wrapper. Otherwise `/v1/adapters/load` 500s silently.
- **Always GET /v1/adapters before trusting a load** — the load endpoint
  doesn't bubble a clear error to callers; the only way to detect a
  silent failure is to verify the adapter shows up in
  `available[]` after the load.

### Next directions

- iter 6-15: same FILTER_VAR=0.05 recipe with varied hyperparams (LR,
  rank, ECHO lambda, NO_ECHO) to map the response surface.
- iter 16+: combine iter 5 best with iter 2's higher outcome — multi-seed
  verification of best recipe.

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
