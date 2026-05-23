# Capability: pi-doctest

## Description

A coding agent given a Python function spec (signature + docstring with
doctests) must edit a stub file to provide the implementation, run the
doctests, and exit when they pass. Today the base Qwen3.5-4B *can*
emit a correct function body when asked directly (humaneval pass@1
≈ 43% with our Phase 1 adapter, ≈ 39% base, per the most recent kiln
behavioral-eval run). But it has not been observed to use *tools* to
verify its work — it tends to emit and stop.

This capability isolates the agentic part: same task shape as
`grpo/python-doctest-passrate`, but instead of a single assistant
message we measure whether the model uses `pi`'s bash tool to verify
its implementation before declaring done.

Concrete failure modes the 4B exhibits today (to be confirmed on the
pi-smoke):
- Writes a function and exits without running the doctests.
- Runs the doctests in a way that fails (`python solution.py` instead
  of `python -m doctest solution.py -v`).
- Loops: edits, edits again without testing in between.
- Refuses tool use entirely — emits prose explaining the function.

## Base model

Qwen3.5-4B (kiln serve on http://localhost:8420). Pi configured via
`kiln pi-setup` to use `Qwen3.5-4B`.

## Rollout source

Pi sessions, headless (`pi -p "<prompt>" --session-dir <run_dir>`).
Sampling defaults from kiln (temperature=0.8, top_p=0.95). N=4
rollouts per task per training step (raise to 8 once v0 lands).

**Single-turn task design:** the task prompt instructs pi to
emit the entire solution in one assistant turn (one `write` tool
call + at most one `bash` call for verification). Originally this
sidestepped the kiln-train multi-turn assistant-token-masking gap,
but that gap is now closed by the ECHO trajectory schema
(`kiln-train::trajectory_mask`); the single-turn shape is now kept
because it matches the pi-doctest task spec, not because the
trainer requires it. See
[`kiln-polish-prerequisites.md`](kiln-polish-prerequisites.md) §1
(RESOLVED).

## Pi configuration (verified during Phase 0 pi-smoke)

- Pi binary: `/usr/bin/pi` (built from earendil-works/pi `npm link`).
- Model id served by kiln: `Qwen3.5-4B`.
- Session JSONL location: `~/.pi/agent/sessions/<workdir-encoded>/<uuid>.jsonl`
  (per pi v0.75.1 README §Sessions).
- Session end signal: pi exits with code 0 on success; the final
  event has `messages: [{role: "assistant", content: "..."}]` with
  no `tool_calls`.
- Session format: one event per JSONL line. Each event carries at
  least `{id, at, messages: [...]}`. Tool calls appear as
  `tool_calls: [{name, ...}]` on the assistant turn that emits them.
- Turn budget per session: 8 turns (cap at 8 to bound wall-clock).

## Reward function (v1 — multi-component, adopted after iter 0 baseline)

| Sub-score | Weight | What it measures | What it CANNOT be cheated by |
|-----------|--------|-------------------|-----------------------------|
| `outcome` | hard floor | Doctest pass-rate via subprocess `python3 -m doctest -v solution.py` on the final workdir. | Empty `solution.py` (no doctests run → 0.0). |
| `tool_call_efficiency` | 0.30 | `1 - clip((n_tool_calls - 4) / 8, 0, 1)`. 1.0 when ≤4 tool calls; 0.0 at ≥12. | Empty session (no tool calls at all returns 1.0 but `outcome` will catch it). |
| `tested_before_done` | 0.20 | 1.0 iff a `bash` tool call mentioning `doctest` appears before the final assistant turn. | Saying "DONE" without testing → 0.0. |
| `format_compliance` | 0.10 | Fraction of `toolCall` blocks with well-formed `name` + JSON-serializable `arguments`. | Malformed XML blocks → 0.0 per block. |

**Composite = `outcome × (0.30·tool_call_efficiency + 0.20·tested_before_done + 0.10·format_compliance + 0.40)`**

Range: [0, 1]. Outcome multiplies the agentic component so an incorrect
solution gets composite=0 regardless of how clean the agentic process
was. This is the "no-empty-solution-cheating" guard required by §0.

**v0 rubric (single-component `outcome` only) was retired** after iter 0
because baseline composite hit 0.958 — the §0 "rubric too lax" zone.
The 4B base model is genuinely competent at humaneval-style tasks; the
real headroom is in agentic *efficiency*, not correctness. See iter 0
in `capability.jsonl` for the closeout.

### Headroom (v1 rubric, measured iter 0 baseline)

| metric | value |
|--------|-------|
| baseline composite (mean over 24 eval tasks) | **0.8854** |
| headroom remaining | 0.1146 |
| group-variance stdev (composite) | 0.218 |
| group-variance stdev (tool_call_efficiency) | **0.358** (target sub-score) |
| group-variance stdev (outcome) | 0.200 (driven by 1 task fail) |
| group-variance stdev (tested_before_done) | 0.100 |
| group-variance stdev (format_compliance) | 0.000 |
| mean wall-clock per rollout | 25.4 s |
| tool-call count distribution | 14 efficient (3-4 calls), 5 moderate (5-9), 4 wasteful (13-27), 1 outcome-fail |

The cap is in the healthy headroom band. **Target sub-score: `tool_call_efficiency`** — has the most movable mass.

### Local WSL harness note (2026-05-23)

The first round-2 local oracle run is recorded in `capability.jsonl` as
`local-0a/base-normal-serve-reasoning-drift`, but it is **not** a valid
baseline. It used the user-started normal server command while the rollout
harness also passed Pi `--thinking off`. Pi recorded `thinkingLevel=off`, but
for this `openai-completions` provider that flag did not reliably map to
`chat_template_kwargs.enable_thinking=false`, so server-side thinking mode was
ambiguous across requests and the run timed out heavily.

Symptom: the blind eval aggregate fell to composite 0.2205 over 24 tasks × 3
rollouts, with mean wall-clock 105.9s and 54/72 zero rollouts. Server logs
showed `thinking_mode="reasoning"` on the slow requests. A no-thinking
eval-mode smoke confirmed the harness could run quickly, but the target policy
for this cap is **thinking enabled**: thinking usually raises task scores, and
the useful optimization problem is to make that thinking efficient rather than
to disable it.

Current local server policy:
`KILN_MODEL_PATH=./Qwen3.5-4B KILN_DEFAULT_THINKING_ENABLED=true ./target/release/kiln serve`.
Health should show `eval_mode=false`, `default_thinking_enabled=true`. The
rollout harness now leaves `--thinking` unset by default so the server env var
is the source of truth. Pi uses the local model alias
`qwen-3.5-4b-kiln-pi1024`, which keeps thinking enabled but caps each turn at
1024 output tokens instead of the provider's 32768-token ceiling. The prompt
also asks for brief internal reasoning before acting. The v1 composite remains
unchanged, and the rubric now emits aggregate diagnostics for `thinking_chars`,
`thinking_blocks`, and `thinking_chars_per_tool_call` so training/eval reports
can track the score vs. thinking-efficiency tradeoff without changing the blind
score definition.

Next valid measurement: rerun the blind baseline under thinking-on server
defaults before any local H15 training claim.

### Adversarial design (§0)

**Q: What's the cheapest way to score 1.0 without doing the capability?**

A1: Pi reads `solution.py`, sees the docstring, and the model literally
    copies the doctest examples into the function as `if x==1: return
    'foo'; elif x==2: return 'bar'`. This is "memorise the doctests"
    — passes the doctests by case-matching. The seed pool of HumanEval
    is small enough this is a real risk.
    
    Mitigation v0: accept it; humaneval doctests usually have ≥3
    distinct cases and case-matching all of them is *roughly* solving
    the task. At iter 2+ add a `hidden_tests` sub-score that runs
    additional test cases the model can't see.

A2: Pi emits the same canonical function regardless of prompt. The
    model converges to outputting `def foo(x): return x` for every
    task, and the doctests fail uniformly.
    
    Mitigation: this would score 0 on outcome, which is the dominant
    weight. No further mitigation needed for v0.

A3: Pi reads a separate hidden file in the workdir that contains a
    correct implementation.
    
    Mitigation: the workdir scaffold (`task_scaffold.py`) writes only
    `solution.py` + `README.md`. No solutions co-located.

A4: Pi runs the doctests, sees a failure, and continues editing until
    a happy-path implementation passes — but the implementation is
    buggy on edge cases the docstring didn't show.
    
    Mitigation v0: accept it; this is a real bug pattern but it's
    also basically what humans do. Hidden tests at iter 2+ are the
    proper fix.

**Q: What does the within-group reward distribution look like at
baseline?**

A: TBD after Phase 0 step 11 (group-variance baseline).

### Headroom

- baseline composite: TBD
- headroom: TBD
- baseline group variance: TBD

## Pi prompt template (the system + user messages pi sees)

```
[system] You are a Python coding assistant. You have access to bash,
write, read, and edit tools. Solve the user's task. When the task
is complete, emit a final assistant message with no tool calls.

[user] In the file `solution.py` is a stub Python function with a
docstring containing doctest examples. Replace the function body so
the doctests pass. After editing, run `python3 -m doctest -v
solution.py` to verify. If the output shows "items passed all
tests" (and no failures), reply DONE and exit.
```

(The exact prompt text lives in `task_scaffold.py`; this section
documents intent for reviewers.)

## Hypothesis log

| Iter | Slug | Family | Composite | Δ | Status | Notes |
|------|------|--------|-----------|---|--------|-------|
| 0    | baseline-v0-outcome-only | baseline | 0.958 (v0 rubric) | — | infra-fail | v0 outcome-only rubric saturated at >=0.95; retired. |
| 0    | baseline-v1              | baseline | 0.885 (v1 rubric) | — | kept       | Multi-component rubric, in healthy headroom band. Target = tool_call_efficiency (stdev 0.358). |
| 1    | h1-default-recipe-3group-smoke | H1 | 0.888 | +0.003 | kept-with-caveat | 3-group smoke training. Composite flat. Target sub-score `tool_call_efficiency` +0.0104; mean n_tool_calls −18% (6.83→5.63). 6 tasks better, 5 worse, 13 same. Outcome held at 0.958. End-to-end loop closes. |
| 2    | h1-default-recipe-h100-20tasks | H1 | 0.919 | **+0.113** | ✓ kept-ship | 20 train tasks × 4 gens on H100. Outcome 0.83→1.00 (perfect). 4 baseline-failing tasks recovered. Wall-clock −23%. |
| 3    | h1-default-recipe-h100-40tasks | H1 | 0.845 | +0.040 vs base | ablation (-0.073 vs iter 2) | 40 tasks at lr=1e-5 = overtraining. Fixed task_0017 but lost task_0002/_0011. Net regression. |
| 4    | h1-lower-lr-5e-6-40tasks       | H1-lr | 0.873 | +0.068 vs base | ablation (-0.046 vs iter 2) | Lower lr reduced overtraining but didn't recover iter 2's level. The 20-task sweet spot is real. |
| 5    | h1-strong-signal-only-11groups | H1-filter | **0.899** | **+0.094** | ✓ reproducible-best | Filter to strong-signal groups (var>0.05). +9.4pp composite, reproducible across seeds at this scale. Outcome 0.79→0.92 (1 zero remaining). |
| 6    | h1-strong-plus-almost-pass     | H1-mix | 0.859 | +0.054 vs base | ablation (-0.040 vs iter 5) | 11 strong + 4 'almost-pass' — stratification didn't help. |
| 7    | h1-replay-iter2-recipe         | H1 | 0.846 | +0.041 vs base | **ablation-CRITICAL** | Iter 2 recipe replay on fresh rollouts → 0.846 (not 0.919). iter 2's number was sample-variance, not robust. Honest reproducible best = iter 5's 0.899. |
| 8    | h13-2epoch-strong-signal       | H13 | 0.750 | −0.055 vs base | ablation-falsified | 2 epochs on iter 5's 11 strong-signal groups regresses composite −0.149 vs iter 5. 5 zeros (vs iter 5's 1). Mean wall-clock 3× blow-up (19.86s → 56.45s) — the over-training signature. Confirms 1 epoch is the sweet spot on filtered data. |
| 9    | iter5-2nd-seed-eval-variance   | H14 | **0.893** | +0.087 vs base | ✓ kept-verification | 2nd-seed eval of iter 5 adapter (same weights, fresh rollouts) → 0.893. iter 5 mean across 2 seeds = 0.896 ± 0.003. Eval-rollout variance for trained adapter is much smaller than the seed-to-seed variance of the recipe itself. |
| 9b   | base-2nd-seed-eval-variance    | H14 | 0.902 | — (calibration) | ✓ calibration | Base eval-rollout variance is σ≈0.048 — much larger than iter 5's 0.003. Reframes the headline +9.4pp single-seed result to a 2-seed-mean +4.2pp. |
| 10   | h12-fresh-rollouts-training-seed | H14 | **0.896** | +0.091 vs base | ✓ kept-verification-recipe-level | Recipe-level reproducibility test: fresh rollouts on same 12 strong-signal tasks, retrain GRPO, eval. Result 0.8958 — virtually identical to iter 5 family mean. iter 5 family across 3 seeds: 0.8958 ± 0.0032. The H12 recipe is robust at BOTH eval-seed and training-seed levels. |




## Kiln-polish prerequisites

See [`kiln-polish-prerequisites.md`](kiln-polish-prerequisites.md).
§1 (per-turn assistant-token masking) is now **RESOLVED** in kiln-train
via the ECHO trajectory schema (`trajectory.rs` + `trajectory_mask.rs`).
This cap continues to use single-turn rollouts because the task
spec calls for one-shot solutions, but multi-turn agentic GRPO is
now first-class — see `capabilities/caps/pi-terminal-bench-lite/`
for the canonical multi-turn paper-reproduction cap.


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
Round 1 result: **+4.2pp 3-seed verified**, 15× tighter eval variance,
25% faster wall-clock. Strong-signal filter recipe at iter 5/9/10
reproducible at 0.896 ± 0.003 (n=3 seeds).

Highest-leverage improvements:

1. **Add the hidden_tests sub-score** (§0 A1 mitigation, deferred from
   round 1). Each task gets ≥3 visible doctests plus ≥3 hidden test
   cases the model can't see. Composite gains a `hidden_test_passrate`
   sub-score that distinguishes "memorize the doctests" from "actually
   solve the function." Round-1 §0 A1 documented this explicitly as the
   chief cheat path; round-2 fixes it.
2. **Expand eval pool 24 → 50 tasks.** Round-1 base composite_stdev
   across seeds was 0.048 (n=24); doubling the eval set should bring
   it to ~0.034, making smaller deltas distinguishable from noise.
3. **Build a hard-eval pool from round-1 failed tasks.** The task IDs
   the base model failed on (task_0019 circular_shift, etc.) become a
   `datasets/hard_eval.tasks.jsonl` for separate measurement. Lift on
   hard-eval is the cleanest evidence of capability uplift vs.
   lucky-tasks.
4. **Replicate the iter-5 H12 recipe at the larger eval pool** before
   trying new hyperparameter axes. The 3-seed reproducibility result
   should hold; if it doesn't, the larger eval pool exposed a fragility.
