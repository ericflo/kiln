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
`kiln pi-setup` to use `qwen-3.5-4b-kiln`.

## Rollout source

Pi sessions, headless (`pi -p "<prompt>" --session-dir <run_dir>`).
Sampling defaults from kiln (temperature=0.8, top_p=0.95). N=4
rollouts per task per training step (raise to 8 once v0 lands).

**Single-turn constraint for v0:** the task prompt instructs pi to
emit the entire solution in one assistant turn (one `write` tool
call + at most one `bash` call for verification). This sidesteps
the multi-turn assistant-token-masking gap in kiln-train (see
`kiln-polish-prerequisites.md`).

## Pi configuration (verified during Phase 0 pi-smoke)

- Pi binary: `/usr/bin/pi` (built from earendil-works/pi `npm link`).
- Model id served by kiln: `qwen-3.5-4b-kiln`.
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

## Kiln-polish prerequisites

See `kiln-polish-prerequisites.md`. v0 sidesteps the multi-turn
token-masking gap by constraining pi to single-turn task completion.
Multi-turn agentic GRPO requires the masking landing.
