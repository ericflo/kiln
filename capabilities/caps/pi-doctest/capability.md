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

Valid normalized thinking-on smoke measurement:

- Base, `LIMIT=4 SEEDS=1`: composite 0.90625, outcome 1.0,
  tool_call_efficiency 0.6875, mean tool calls 6.0, mean thinking chars 1984.0.
- H15 lean32 adapter: composite 0.7125, delta -0.19375, outcome 0.75,
  tool_call_efficiency 0.84375, mean tool calls 4.75, mean thinking chars
  2234.5.

H15 is rejected at the smoke gate. It moved the intended efficiency metric in
the right direction, but did so by harming outcome and increasing thinking per
tool call. The next local iteration should preserve successful thinking-on
solutions as an anchor instead of training only on lean high-variance failure
loops.

H16 tested that anchor idea mechanically, but was aborted for throughput before
producing an adapter. Filtering by outcome and text length was not enough; the
server-native GRPO path still became impractical on a selected completion with
830 action tokens. The next dataset builder must use `kiln trajectory inspect
--json` to enforce a per-completion action-token cap before submitting a train
job.

H17 enforced that token diagnostic directly with a 650 action-token cap and
kept only outcome-perfect, non-timeout train completions. It produced a clean
but tiny batch, 2 groups / 7 completions / 2675 action tokens, with no
trajectory schema warnings. Training still became too slow on the second group:
the job reached 44% and loss 0.7438, then remained there until 587s elapsed
despite 100% GPU utilization. No H17 adapter was promoted or blind-evaluated.
Another local server-native GRPO attempt needs a much tighter action-token cap
around 350 per completion, otherwise this cap should switch to a gentler
non-GRPO data signal instead of burning time on tiny slow preference groups.

H18 switched to SFT on successful train rollouts. The raw Pi transcripts were
rendered into compact thinking-on workflows with real tool arguments and
normalized `solution.py` paths. The signal was valid, but local SFT still
stalled on examples with full code-writing turns: edit-style examples reached
56% after 689s, and compact write-style examples with a 3100-char cap reached
71% after 580s. Both were stopped without eval.

H19 kept the same write-style SFT anchor but capped examples at 2900 chars.
This trained successfully in 128004 ms with rank 4 / alpha 8 / lr 1e-5 /
1 epoch on 6 examples. Smoke result, `LIMIT=4 SEEDS=1`: composite 0.94375,
delta +0.0375 versus the normalized thinking-on base smoke, outcome 1.0,
tested-before-done 1.0, tool-call efficiency 0.8125, mean tool calls 5.0, and
mean thinking chars 1717.0. This is a smoke-kept adapter only; the next step is
a larger paired blind promotion check before treating it as a stage.

The paired promotion check rejected H19. On `LIMIT=8 SEEDS=1`, base scored
0.8328125 and H19 scored 0.6828125, delta -0.15. H19 produced more zero
rollouts, worse outcome, worse tested-before-done, worse tool-call efficiency,
more tool calls, more thinking chars, and slower wall-clock. The smoke lift was
not stable; H19 is a data-shape/throughput lesson, not a promoted stage.

H20 removed task-solution supervision entirely and trained only bookends:
read `solution.py` first, and stop with `DONE` after a passing doctest output.
It trained successfully, but repeated the false-positive pattern. `LIMIT=4`
smoke scored 0.98125, delta +0.075, with lower wall-clock and fewer tool calls.
The larger `LIMIT=8` paired check scored 0.7265625 versus base 0.8328125,
delta -0.10625, with worse outcome, worse tool-call efficiency, more zero
rollouts, more thinking chars, and slower wall-clock. Tiny SFT pressure is not
stable here; future adapter attempts need a stronger gate before optimism, and
should likely move away from SFT unless a qualitatively different signal is
available.

H21 tested whether prompt wording alone could recover the efficiency headroom
before distillation. Three base-model prompt variants were evaluated on the
same `LIMIT=8 SEEDS=1` blind aggregate size: `lean-tools`, `edit-first`, and
a narrow stop-after-pass extra instruction. None beat the default prompt's
0.8328125 composite. `lean-tools` reduced thinking chars but hurt outcome;
`edit-first` removed zero rollouts but collapsed tool-call efficiency; the
stop-after-pass extra instruction regressed both outcome and efficiency. Do
not distill these prompts.

H22 tested a qualitatively different signal: verifier-free ECHO-only training
with `--no-policy-loss` on the tiny H17 success-anchor train dataset. Dry-run
validated 2 groups / 7 completions / 2584 action tokens / 2201 env tokens and
correctly warned that policy-gradient GRPO would be harmful on the saturated
rewards. The first offline CUDA training command still failed the
local-throughput gate after model load. An explicit retry with
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`, `RUST_LOG=info`, and `--max-groups 1`
confirmed that streamed GRPO checkpointing was active, but it also timed out:
the first completion's backward pass took 270s, the second 178s, and the third
did not complete before the 900s wrapper. ECHO remains plausible as a method,
but this local route needs a smaller env-only microbatch or trainer support
that skips policy/reference work when `--no-policy-loss` is set.

H23 tested that smaller route directly. A one-group / one-completion ECHO
dry-run was rejected as `zero_groups`, because dynamic GRPO filtering still
requires a non-degenerate reward group before ECHO can run. A one-group /
two-completion shape passed dry-run with 896 action tokens, 759 env tokens, and
598 context tokens. Training with explicit 24-segment checkpointing made real
progress but timed out at 900s without an adapter artifact: the first
completion's backward pass took 196s, and the second completion's policy
forward alone took 426s before the run entered backward and timed out.
Gradient checkpointing is active and necessary, but the next useful step is a
trainer change that skips reference/policy work for `--no-policy-loss` ECHO or
adds a dedicated env-token-only ECHO path.

H24 found a no-code throughput workaround: run the one-completion ECHO dataset
under `--mode baseline --no-policy-loss`, which disables dynamic group
rejection and uses per-sample optimizer stepping. The dry-run accepted one
group / one completion with 280 action tokens, 256 env tokens, and 278 context
tokens. Training finished in 129s, installed
`pi-doctest-h24-baseline-one-echo-r4a8`, and `kiln adapter verify` passed with
a nonzero delta proxy. Blind `LIMIT=4 SEEDS=1` smoke rejected it: composite
0.85 versus the normalized base smoke 0.934375, with outcome preserved at 1.0
but tool-call efficiency falling to 0.50, mean tool calls rising to 9.0, and
mean thinking chars rising to 3585.75. Baseline-mode ECHO is now a viable local
training primitive, but pure env-only training on one success trace should not
be scaled; the next data signal must directly favor concise action behavior.

H25 put that into a policy-side test: baseline-mode per-sample GRPO+ECHO on the
same two-completion reward-spread group that H23 used, but with policy loss
enabled. Dry-run passed with 896 action tokens, 759 env tokens, and low but
nonzero reward variance. Training no longer hit the H23 second-forward
group-accumulation stall, but it exposed a stricter sequence-length limit: the
first 1243-token completion's backward pass took 644s, with layers 0-2 alone
taking 342s. The run timed out after the second completion's policy forward
started, before any adapter artifact was saved. Policy-on local GRPO needs
substantially shorter trajectories, likely under 800 sequence tokens and under
300 action tokens, before it can be a practical data experiment on this laptop.

H26 enforced that shorter policy cap by mining the shortest same-task
two-completion pair from the H17 train data. The selected completions were 814
and 801 sequence tokens, with 280 and 299 action tokens. Baseline-mode
GRPO+ECHO trained successfully in 308s and verified with a nonzero adapter
delta. Cheap `LIMIT=4` smoke was positive, composite 0.953125 versus base
0.934375, with tool-call efficiency 0.84375. The larger `LIMIT=8` paired gate
rejected it: H26 scored 0.7796875 versus base 0.8328125. It improved
tool-call efficiency to 0.828125 and reduced mean tool calls to 4.875, but
outcome fell to 0.8125. This is the first completed policy-on local GRPO shape
that moved the targeted efficiency axis, but the tiny saturated reward spread
was not a safe action-side signal.

H27 synthesized a stronger same-task contrast from train-only H17 tasks: each
group paired one concise passing trace with one still-passing but inefficient
repair trace. The first synthetic negatives were too long, so they were trimmed
to one read, one bad edit, one failing doctest, one fix, one passing doctest,
and `DONE`. The final dry-run shape was 2 groups / 4 completions / 838 action
tokens / 764 env tokens, with completion lengths 650-762 sequence tokens and
150-266 action tokens. Explicit `KILN_GRAD_CHECKPOINT_SEGMENTS=24` was active;
training completed in 167s with peak observed VRAM about 15980 MiB and a
verified adapter delta proxy of 0.332205. Blind `LIMIT=4` smoke rejected it:
H27 scored 0.915625 versus base 0.934375. Outcome and tested-before-done
remained 1.0, but tool-call efficiency fell to 0.71875, mean tool calls rose
to 5.75, and mean thinking chars rose to 2102.75. The local throughput shape is
good, but synthetic repair negatives appear to teach extra tool use; do not
run the larger gate.

H28 tested a real chain experiment using the reference-cap lesson that chaining
only makes sense when the second corpus is broader and complementary. It
continued from H26, which moved efficiency but hurt outcome, and trained on five
train-only hard-negative groups: concise verified pass traces versus wrong
no-test terminal guesses. The local shape was good: 10 completions, 936 action
tokens, max 554 sequence tokens, explicit 24-segment checkpointing, 270s train
time, and peak VRAM about 15981 MiB. `LIMIT=4` smoke looked excellent:
composite 0.990625 versus base 0.934375, tool-call efficiency 0.96875, mean
tool calls 3.5, and mean thinking chars 1066.0. The larger `LIMIT=8` paired
gate rejected it hard: composite 0.659375 versus base 0.8328125, outcome
0.6875, tool-call efficiency 0.703125, two zero rollouts, and slower wall
clock. Chaining from a rejected efficiency adapter amplified fragility; future
chains should start from an adapter that clears the larger gate, or test the
broad hard-negative contrast fresh from base before stacking any efficiency
prior. `LIMIT=4` is now only a throughput smoke, not a decision gate.

H29 tried that isolation test by training the same five train-only
hard-negative groups fresh from base, with no H26 adapter in the chain. The
dry-run matched H28's data shape: 10 completions, 936 action tokens, 1177 env
tokens, max 554 sequence tokens, max 150 action tokens per completion, and
reward stdev 0.5. The full run used rank 4 / alpha 8, lr 5e-6, policy loss
enabled, ECHO lambda 0.05, and explicit `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.
It completed four of five groups but hit the 900s guard during group 5
reference forward before writing an adapter. Group 4 showed the cost profile:
176070 ms reference forward, 211237 ms first policy forward, 119011 ms first
backward, and 136273 ms second backward, with observed CUDA memory around
15978 MiB. No blind eval was run. H29 does not falsify the hard-negative data
behaviorally; it falsifies this fresh-from-base five-group policy-on shape as a
practical laptop iteration. The next retry should cut the token/group budget or
precondition with a cheaper no-policy-loss/ECHO pass before policy-on GRPO.

H30 tested the gradient-checkpointing question directly. It retried the same
fresh-from-base hard-negative route on only two groups, but lowered
`KILN_GRAD_CHECKPOINT_SEGMENTS` from 24 to 12. The trainer confirmed 12
segments and it fit in memory, with the first logged CUDA gate around 16376 MiB,
but larger multi-layer checkpoint blocks made recompute worse: one 3-layer
block covering layers 12-15 took 82506 ms, followed by 14172 ms for layers
9-12, 15183 ms for layers 6-9, and 9209 ms for layers 3-6. The run was stopped
during the first completion's backward pass before adapter write. Conclusion:
gradient checkpointing remains necessary, but fewer segments are the wrong
direction for this trainer path. Keep 24 segments for policy-on GRPO and reduce
data/token shape instead.

H31 followed that evidence: keep 24 checkpoint segments, train fresh from base,
and reduce the H29 hard-negative corpus to two groups with `--max-groups 2`.
Dry-run shape was 4 completions, 389 action tokens, 526 env tokens, max 554
sequence tokens, max 150 action tokens per completion, and reward stdev 0.5.
Training completed in 77s observed wall-clock, peak VRAM about 15997 MiB, with
the receipt reporting 70565 ms in backward and 73531 ms total. Verification
passed with 400 nonzero LoRA tensors and delta proxy 0.248275. Blind `LIMIT=4`
smoke rejected it: H31 scored 0.925 versus base 0.934375. Outcome,
tested-before-done, and format stayed at 1.0, but tool-call efficiency fell to
0.75, mean tool calls rose to 5.5, mean thinking chars rose to 2153.25, and
wall-clock rose to 42.35s. The larger gate was skipped. The throughput shape is
now workable, but this hard-negative data still does not improve the target
behavior from base.

H32 returned to natural successful traces instead of failure negatives. It took
the short same-task pair behind H26 and rank-rescaled the tiny original reward
spread `[0.987196, 0.985446]` into `[1.0, 0.0]`, preserving only passing
train-only trajectories. The first two-pair version was rejected before
training because the second pair included a 1243-token / 571-action-token
completion, repeating H25's timeout shape. The one-pair version dry-ran at 579
action tokens, 480 env tokens, max 814 sequence tokens, and max 299 action
tokens per completion. Training completed in 264s observed wall-clock with peak
VRAM about 15997 MiB; verification passed with delta proxy 0.238848. Blind
`LIMIT=4` smoke was exactly flat on the score: composite 0.934375 versus base
0.934375, outcome/tested/format all 1.0, tool-call efficiency 0.78125, mean
tool calls 5.25, mean thinking chars 1979.25, and wall-clock 36.52s. The larger
gate was skipped. Rank-amplified natural preferences are safer than hard
failure negatives, but one tiny-preference group is not enough signal.

H33 tested whether H31's hard-negative failure was partly an ECHO artifact.
It reused H31's exact two-group hard-negative dataset, but trained with
`--no-echo`, keeping policy loss on and all other settings at rank 4 / alpha 8
/ lr 5e-6 / 24 checkpoint segments. Dry-run shape was the same as H31: 4
completions, 389 action tokens, 526 env tokens, max 554 sequence tokens, max
150 action tokens per completion. Training completed in 125s observed
wall-clock with peak VRAM about 15963 MiB; verification passed with delta proxy
0.256303. Blind `LIMIT=4` smoke was strong: composite 0.98125 versus base
0.934375, tool-call efficiency 0.9375, mean tool calls 3.75, mean thinking
chars 1585.0, and wall-clock 28.28s. The larger `LIMIT=8` gate also cleared on
composite, scoring 0.8921875 versus base 0.8328125, with outcome 1.0 versus
base 0.875 and zero zero-rollouts versus base one. The caveat is that
tool-call efficiency fell to 0.640625, mean tool calls rose to 6.75, and
wall-clock rose to 62.65s. H33 is therefore an outcome-reliability win, not an
efficiency win. It should be kept with caveat and either confirmed with another
seed or used as the base for a small efficiency-recovery stage.

H34 tried that small efficiency-recovery stage by chaining from H33 and
reusing H32's natural same-task successful pair, with ECHO still disabled. The
data was intentionally tiny: one group, two completions, 579 action tokens, 480
env tokens, reward stdev 0.5 after rank rescaling, and max 814 sequence tokens.
Training used rank 4 / alpha 8 / lr 5e-6 / policy loss on /
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`, completed in 318s observed wall-clock, and
fit at about 15994 MiB peak VRAM. Verification passed with delta proxy
0.343389. Blind `LIMIT=4` smoke rejected the chain: H34 scored 0.750000 versus
base 0.934375 and H33 smoke 0.981250. It improved base-smoke mean tool calls
to 4.5 and mean thinking chars to 1740.75, but outcome fell to 0.75 and one
rollout was zero. The larger gate was skipped. The lesson is that a tiny
natural-rank efficiency signal can erase H33's reliability benefit when chained
directly; the next H33 follow-up should either confirm H33 with another seed or
use a broader, reliability-preserving data signal rather than one narrow
preference pair.

H35 performed that H33 confirmation before spending more training time. It did
not train a new adapter; it reran `pi-doctest-h33-hardneg-g2-noecho-r4a8` on a
fresh `LIMIT=8 SEEDS=1` blind aggregate, paired against the same base8 summary.
The result did not reproduce H33's larger-gate win: composite fell to 0.721875
versus base8 0.8328125, outcome fell to 0.75, and zero rollouts rose to 2. It
did use fewer tool calls than base (4.625 mean calls, tool-call efficiency
0.84375), but that efficiency was bought by lost correctness. H33 should now be
treated as an unstable candidate rather than a stable kept stage; do not chain
further from it without a stronger multi-seed confirmation.

H36 tested whether H33's instability was caused by too little no-ECHO
hard-negative data. It trained fresh from base on the full five train-only
hard-negative groups from H29, still with `--no-echo`, rank 4 / alpha 8 / lr
5e-6 / policy loss on / `KILN_GRAD_CHECKPOINT_SEGMENTS=24`. The dry-run shape
was 5 groups, 10 completions, 936 action tokens, 1177 env tokens, max 554
sequence tokens, and max 150 action tokens per completion. Training completed
in 439s observed wall-clock and fit at about 16000 MiB peak VRAM; verification
passed with a larger delta proxy of 0.577938. Blind `LIMIT=4` smoke rejected it
hard: H36 scored 0.750000 versus base 0.934375, outcome fell to 0.75,
tested-before-done fell to 0.875, tool-call efficiency fell to 0.75, mean tool
calls rose to 5.75, mean thinking chars rose to 2440.75, and one rollout was
zero. The broader hard-negative/no-ECHO recipe is therefore not a stable
direction; future data should avoid wrong/no-test terminal negatives rather
than simply scaling them.

H37 pivoted from hard negatives to the user's thinking-efficiency tradeoff. It
kept successful train-only workflows intact, then created compressed-thinking
versions of two successful H17 traces and ranked each compressed version above
its original verbose counterpart, with `--no-echo` and policy loss enabled. The
dry-run was valid at 2 groups, 4 completions, 815 action tokens, 1298 env
tokens, max 1010 sequence tokens, and max 325 action tokens per completion.
The full train timed out at 900s before adapter write. Group 1 completed, but
the original verbose counterpart took 393243 ms in backward, with checkpoint
segment 7 alone taking 93361 ms. Group 2 then hit a worse throughput wall:
reference forward alone took 366307 ms for the 1010-token max-sequence group,
and the timeout fired before any adapter artifact was saved. The idea is safer
behaviorally than wrong/no-test negatives, but the next retry must keep every
completion below roughly 850 sequence tokens and below 300 action tokens.

H38 rebuilt H37 as a short-only version: one train-only successful trace from
H17, compressed-thinking variant ranked above the original, with both
completions under the local throughput envelope. Dry-run shape was 1 group, 2
completions, 337 action tokens, 448 env tokens, max 763 sequence tokens, and
max 261 action tokens per completion. Training completed in 102s observed
wall-clock with peak VRAM about 15983 MiB, and verification passed with delta
proxy 0.255293. Blind `LIMIT=4` smoke looked useful: composite 0.962500 versus
base 0.934375, outcome/tested/format all 1.0, tool-call efficiency 0.875, and
mean tool calls 4.25. But the `LIMIT=8` larger gate rejected it: composite
0.731250 versus base 0.8328125, outcome 0.75, tested-before-done 0.9375, two
zero rollouts, mean wall-clock 63.41s, and mean thinking chars 3362.1. The
short compression shape is trainable, but one-task compression overfits smoke
and does not yet stabilize thinking efficiency.

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

## Local continuation notes

### H39: thinking-tier g1 no-ECHO

H39 tested a safer thinking-compression contrast: train-only successful traces
were rewritten into brief-vs-medium thinking tiers while preserving the same
verified workflow and tool payload. The broader three-group/two-group shapes
dry-ran correctly but were rejected as local throughput routes: gradient
checkpointing kept VRAM inside the 16GB envelope, yet lower/mid layer recompute
made the runs too slow, and reducing checkpoint segments to 16 or 8 did not fix
wall-clock.

The bounded one-group variant trained successfully from base as
`pi-doctest-h39-thinking-tier-g1-noecho-r4a8` with rank 4, alpha 8, lr `5e-6`,
no ECHO, and `KILN_GRAD_CHECKPOINT_SEGMENTS=24`. Training took 277518 ms with
314 action tokens, 588 env tokens, 556 context tokens, and about 15982 MiB peak
VRAM. Adapter verify passed.

Blind `LIMIT=4 SEEDS=1` smoke rejected it: composite 0.925 versus base smoke
0.934375, with outcome/tested/format all 1.0 but tool-call efficiency down to
0.75, mean tool calls up to 5.5, mean thinking chars up to 2703, and mean
wall-clock up to 51.08s. The promotion gate was skipped. Lesson: controlled
successful thinking-tier pairs are safer than failure negatives but still too
brittle as one-group GRPO; broader versions need a cheaper method or shorter
workflow-only payloads before this route is worth another promotion check.

### H40: low-dose short thinking compression

H40 reused H38's one-group train-only short-compression pair, but lowered the
update dose to rank 4, alpha 4, lr `1e-6`, no ECHO. This tested whether H38's
promotion failure came from over-strong one-task pressure rather than the data
shape.

The adapter `pi-doctest-h40-think-compress-lowdose-noecho-r4a4-lr1e6` trained
successfully in 63458 ms with 337 action tokens, 448 env tokens, 556 context
tokens, and about 15972 MiB peak VRAM. Adapter verify passed with a much
smaller LoRA update proxy, 0.05226.

Blind `LIMIT=4 SEEDS=1` smoke rejected it: composite 0.9000 versus base smoke
0.934375, outcome 1.0, tested-before-done 0.875, tool-call efficiency 0.75,
mean tool calls 5.25, mean thinking chars 2893.25, and mean wall-clock 46.01s.
Lesson: lowering dose does not rescue the H38 compression pair; future
compression attempts need a changed data distribution, not just smaller
alpha/lr.

### H41: post-pass stop g4 no-ECHO

H41 changed the data distribution from full-trace thinking compression to a
short terminal behavior contrast. Four train-only groups ended after a
successful doctest tool result; the preferred completion briefly acknowledged
the pass and emitted `DONE`, while the rejected completion redundantly reran
doctests. This kept thinking enabled and targeted efficient stopping rather
than disabling reasoning.

The adapter `pi-doctest-h41-postpass-stop-g4-noecho-r4a8` trained successfully
in 202119 ms with 260 action tokens, 0 env tokens, 2970 context tokens, and
about 15965 MiB peak VRAM under `KILN_GRAD_CHECKPOINT_SEGMENTS=24`. Adapter
verify passed with LoRA update proxy 0.78701.

Blind `LIMIT=4 SEEDS=1` smoke was positive: composite 0.971875 versus base
smoke 0.934375, outcome/tested/format all 1.0, tool-call efficiency 0.90625,
mean tool calls 4.0, mean thinking chars 1664.5, and mean wall-clock 33.50s.
Promotion rejected it: `LIMIT=8` composite 0.7859375 versus base 0.8328125,
outcome 0.875, tested-before-done 1.0, tool-call efficiency 0.59375, mean tool
calls 7.25, mean thinking chars 3783.25, mean wall-clock 75.04s, and one zero
rollout. Lesson: post-pass stopping is a useful target, but the adapter update
was too narrow; next attempts should combine terminal stop contrast with
pre-pass workflow coverage or use a lower-impact method.

### H42: low-dose post-pass stop no-ECHO

H42 reused the H41 terminal post-pass-stop corpus but lowered the adapter dose
to rank 4, alpha 4, lr `1e-6`, no ECHO. This tested whether H41's promotion
failure came from too much update magnitude rather than from the terminal-only
data shape.

The adapter `pi-doctest-h42-postpass-stop-lowdose-noecho-r4a4-lr1e6` trained
successfully in 273316 ms with 260 action tokens, 0 env tokens, 2970 context
tokens, and about 15983 MiB peak VRAM under
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. Adapter verify passed with a much smaller
LoRA update proxy, 0.15715 versus H41's 0.78701.

Blind `LIMIT=4 SEEDS=1` smoke again looked positive: composite 0.971875 versus
base smoke 0.934375, outcome/tested/format all 1.0, tool-call efficiency
0.90625, mean tool calls 4.25, mean thinking chars 1670.75, and mean wall-clock
34.38s. Promotion failed badly: `LIMIT=8` composite 0.5875 versus base
0.8328125, outcome 0.625, tested-before-done 0.9375, tool-call efficiency
0.8125, mean tool calls 5.125, mean thinking chars 2815.75, mean wall-clock
72.69s, and three zero rollouts. Lesson: lowering dose does not rescue the
terminal-only post-pass-stop corpus; the next data shape needs pre-pass
workflow coverage, not another dose-only variant.

### H43: balanced workflow low-dose no-ECHO

H43 added pre-pass workflow coverage to the post-pass stop signal: one
start-read contrast, two post-write-test-before-DONE contrasts, and two
post-pass-DONE-before-rerun contrasts, all from train-only data. It used rank
4, alpha 4, lr `1e-6`, no ECHO.

The adapter `pi-doctest-h43-balanced-workflow-lowdose-noecho-r4a4-lr1e6`
trained successfully in 287880 ms with 324 action tokens, 0 env tokens, 4500
context tokens, and about 15970 MiB peak VRAM under
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. Adapter verify passed with LoRA update
proxy 0.11519.

Blind `LIMIT=4 SEEDS=1` smoke was positive but smaller than H41/H42: composite
0.953125 versus base smoke 0.934375, outcome/tested/format all 1.0,
tool-call efficiency 0.84375, mean tool calls 4.5, mean thinking chars 1821.5,
and mean wall-clock 36.97s. Promotion rejected it: `LIMIT=8` composite
0.6484375 versus base 0.8328125, outcome 0.6875, tested-before-done 0.8125,
tool-call efficiency 0.8125, mean tool calls 4.875, mean thinking chars
3284.125, mean wall-clock 80.66s, and two zero rollouts. Lesson: small
tool-choice micro-contrast GRPO continues to create smoke false positives; the
next route should stop adding isolated micro-contrast rows and instead preserve
complete task-solving behavior.

### H44: complete-behavior throughput probe

H44 moved away from micro-contrasts and tried to preserve complete train-only
task-solving behavior. The first branch built a broad SFT corpus from fresh
successful base rollouts: 10 reward-1.0 examples, task-diverse, normalized to
compact read, write, test, DONE steps while preserving real initial files,
final solutions, and doctest outputs. The examples tokenized between 565 and
925 tokens. Native SFT with rank 4 / alpha 4 / lr `1e-6` and
`KILN_CUDA_RECOMPUTE_SFT=1` fit in memory but was too slow: the first step
completed only after roughly twenty minutes at about 15975 MiB VRAM. Generic
SFT on the six most compact examples failed with CUDA OOM at gated deltanet
layer 26. No SFT adapter was saved.

The second branch collected 3 fresh generations on 8 compact-success train
tasks. The collection was highly saturated, mean train composite 0.990625, but
two natural variance groups remained (`task_0028` and `task_0039`) with
reward-1.0 successes and 0.8875 near-misses. Raw complete GRPO dry-ran at 3278
action tokens, 2859 env tokens, and 1780 context tokens, which was too large.
After stripping verbose thinking while preserving tool-call sequence and
original rewards, the 2-group pair dry-run was 1116 action / 1292 env / 1224
context tokens. It trained with no ECHO, rank 4 / alpha 4 / lr `1e-6`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`, reached a mid-run progress point, then hit
the 900s guard before saving an adapter. The smallest one-group `task_0039`
variant dry-ran at 458 action / 798 env / 612 context tokens and also timed out
before adapter save.

Lesson: complete-behavior data is the right direction, but local policy-on
training needs a stricter throughput cap than H44 met. Future attempts should
require sub-300 action tokens per negative completion and sub-800 total
sequence tokens before training, or use a preconditioning route that avoids
full policy backward on complete trajectories. Keep
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`; it is the setting that keeps GRPO inside
the local VRAM envelope, and reducing checkpointing is not the throughput fix.

### H45: post-read edit workflow

H45 kept H44's complete-workflow direction but made it trainable by changing
the action surface. From the first three compact H44 success examples, it kept
the real train-only read context, then trained a post-read contrast that
preferred `edit` -> doctest -> `DONE` over `edit` -> `DONE` without
verification and `edit` -> doctest -> redundant doctest -> `DONE`.

The edit-form dataset dry-ran with 3 groups, 9 completions, 1146 action
tokens, 0 env tokens, 3165 context tokens, and reward stdev 0.408928. Training
used no ECHO, rank 4 / alpha 4 / lr `1e-6`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. It completed in 356.717s observed with
peak VRAM 15941 MiB. Adapter verify passed with 400 nonzero LoRA tensors and
LoRA update proxy 0.079543.

Blind `LIMIT=4 SEEDS=1` smoke rejected H45: paired base composite was 0.9625
with mean wall-clock 23.84s, while H45 scored 0.86875 with mean wall-clock
65.49s. No promotion check was run. Lesson: edit-form compression fixes the
local throughput issue, but this no-test/retest contrast still harms blind
reliability and latency. Next attempts should keep the edit-form token budget
while changing the signal, preferably outcome-preserving pairs that differ only
in efficient thinking/tool use after full verification, or a mixed anchor set
that preserves broad task-solving behavior.

### H46: realistic ECHO workflow

H46 tested whether H45's failure came from its unnatural trajectory shape.
It kept edit-form compression but restored assistant/tool alternation and ECHO:
`edit` -> edit observation -> doctest -> doctest observation -> `DONE` was
preferred over a no-test path and a redundant-retest path. The dataset used the
first two H44 compact successes and dry-ran at 654 action tokens, 756 env
tokens, 2214 context tokens, and reward stdev 0.414327.

Training used rank 4 / alpha 4 / lr `1e-6`, ECHO lambda 0.05, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. It completed in 698.235s observed, with
peak observed VRAM 16007 MiB. ECHO env CE improved from 1.91345 to 1.35367.
Adapter verify passed with 400 nonzero LoRA tensors and LoRA update proxy
0.055388.

Blind `LIMIT=4 SEEDS=1` smoke still rejected it, but the regression was much
smaller than H45: paired base composite was 0.953125 with mean wall-clock
26.24s; H46 scored 0.94375 with mean wall-clock 38.35s. Lesson: realistic
observations plus ECHO substantially reduce the harm, but the no-test/retest
contrast still fails to beat base and adds latency. Next attempts should keep
H46's realistic trajectory structure while removing the `DONE`-without-test
negative, using only verified outcome-preserving efficiency contrasts.

### H47: verified-only realistic ECHO workflow

H47 tested the direct follow-up from H46: remove the unverified no-test
negative entirely and train only a verified outcome-preserving efficiency
contrast. The preferred completion kept the realistic `edit` -> edit
observation -> doctest -> doctest observation -> `DONE` trajectory; the weak
completion also verified but spent an extra redundant doctest rerun before
`DONE`. The dataset reused H46's two train-only groups with the no-test
completion filtered out, preserving rewards 1.0 versus 0.65. Dry-run stats
were 2 groups, 4 completions, 506 action tokens, 587 env tokens, 1646 context
tokens, and reward stdev 0.175.

Training used rank 4 / alpha 4 / lr `1e-6`, ECHO lambda 0.05, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. It completed in 275.301s observed, with
peak observed VRAM 15966 MiB. ECHO env CE improved from 1.96740 to 1.44720.
Adapter verify passed with 400 nonzero LoRA tensors and LoRA update proxy
0.057425.

Blind `LIMIT=4 SEEDS=1` smoke was slightly positive on composite but slower:
paired base scored 0.740625 with mean wall-clock 29.47s; H47 scored 0.75 with
mean wall-clock 37.65s. Because the gain was thin, H47 went to a blind
`LIMIT=8` promotion check. Promotion rejected it: paired base scored 0.8375
with mean wall-clock 51.60s and one zero rollout, while H47 scored 0.66875
with mean wall-clock 76.70s and two zero rollouts. Lesson: removing the
unverified negative avoided H46's immediate smoke regression, but verified-only
ECHO still over-conditioned the model toward slower, less reliable broader
behavior. The next attempt should keep verified-only data but reduce or remove
ECHO, or switch to a non-adapter/prompt-side stop policy rather than another
realistic ECHO workflow update.

### H48: verified-only no-ECHO workflow

H48 reused the exact H47 verified-only dataset but disabled ECHO to isolate
whether H47's env-prediction loss caused the latency and reliability
regression. The dry-run shape was unchanged: 2 groups, 4 completions, 506
action tokens, 587 env tokens, 1646 context tokens, and reward stdev 0.175.

Training used rank 4 / alpha 4 / lr `1e-6`, no ECHO, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. It completed in 112.478s observed, much
faster than H47's 275.301s, with peak observed VRAM 15978 MiB and final loss
0.026854. Adapter verify passed with 400 nonzero LoRA tensors and LoRA update
proxy 0.062328.

Blind `LIMIT=4 SEEDS=1` smoke rejected it decisively: paired base scored
0.98125 with mean wall-clock 19.36s and no zero rollouts, while H48 scored
0.75 with mean wall-clock 38.10s and one zero rollout. No promotion check was
run. Lesson: removing ECHO fixed training throughput but not blind behavior.
The verified-only edit/test/stop micro-contrast is still too narrow as an
adapter update; the next route should either use complete successful behavior
anchors with a smaller policy update, or move efficient stopping into the
runtime/prompt policy rather than another LoRA.

### H49: same-actions concise thinking

H49 changed the target from tool selection to thinking efficiency while
holding behavior fixed. It used H47's successful verified completions only:
the preferred completion kept the concise edit -> doctest -> `DONE` trajectory,
and the rejected completion copied the same tool calls and observations but
expanded each `<think>` block into verbose deliberation. ECHO stayed off. The
dry-run shape was 2 groups, 4 completions, 580 action tokens, 358 env tokens,
1688 context tokens, and reward stdev 0.25.

Training used rank 4 / alpha 4 / lr `1e-6`, no ECHO, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. It completed in 102.232s observed, with
peak observed VRAM 15964 MiB and final loss 0.065274. Adapter verify passed
with 400 nonzero LoRA tensors and LoRA update proxy 0.060037.

Blind `LIMIT=4 SEEDS=1` smoke looked promising on score but not latency:
paired base scored 0.75 with mean wall-clock 16.49s and one zero rollout,
while H49 scored 0.90 with mean wall-clock 48.29s and no zero rollouts. That
earned a `LIMIT=8` promotion check. Promotion rejected it: paired base scored
0.7140625 with mean wall-clock 66.47s and one zero rollout, while H49 scored
0.6671875 with mean wall-clock 63.41s and two zero rollouts. Lesson: same-tool
concise-thinking contrast can move the short smoke score and slightly reduce
larger-sample wall time, but it still increases failure count and loses
composite at the promotion gate. The next adapter data should stop using tiny
two-group synthetic contrasts and instead mix concise thinking with complete,
diverse, train-only successful behavior anchors.

### H50: broad success ECHO-only preconditioning

H50 tested a complementary route after the tiny synthetic contrasts failed:
use broad complete successful train-only trajectories as verifier-free
preconditioning, with policy loss disabled and only ECHO env-token CE active.
The source was H44's complete-success SFT corpus. A first duplicate-pair g4
shape dry-ran at 1804 action tokens, 1702 env tokens, and 2224 context tokens,
but timed out before adapter write after reaching step 12006/25820. The
successful training shape kept one full success trajectory per group and added
a short no-env dummy completion only to satisfy GRPO grouping, still with
`no_policy_loss=true`; this dry-ran at 942 action tokens, 851 env tokens, 2056
context tokens, 4 groups, and 8 completions.

Training used rank 4 / alpha 4 / lr `1e-6`, ECHO lambda 0.025, no policy
loss, and `KILN_GRAD_CHECKPOINT_SEGMENTS=24`. It completed just under the
guardrail in 858.931s observed, with peak observed VRAM 15983 MiB and final
loss 0.046478. ECHO env CE moved from 1.92166 to 2.07419, so the env-only
objective did not improve its own CE on this shape. Adapter verify passed with
400 nonzero LoRA tensors and LoRA update proxy 0.110330.

Blind `LIMIT=4 SEEDS=1` smoke rejected it: paired base scored 0.934375 with
mean wall-clock 30.84s and no zero rollouts, while H50 scored 0.69375 with
mean wall-clock 63.69s and one zero rollout. No promotion check was run.
Lesson: broad success ECHO-only preconditioning is locally trainable only after
aggressive shape reduction, but it is too slow and degrades blind reliability.
The next route should avoid env-only ECHO as a standalone adapter update and
focus on either mixed real success/failure trajectories with policy signal or
non-training runtime constraints for efficient stopping.

### H51: post-failure repair policy signal

H51 moved away from generic success anchoring and tiny stop-only contrasts. It
used two real train-only failure/recovery trajectories from H44's natural
compact GRPO data: the prompt context ended after an actual failed doctest
observation, the preferred completion wrote the corrected file, reran doctest,
and then stopped, while the rejected completion stopped immediately after the
failure. The dry-run shape was 2 groups, 4 completions, 310 action tokens, 210
env tokens, 2995 context tokens, reward stdev 0.5, and two kept medium-variance
groups.

Training used rank 4 / alpha 4 / lr `1e-6`, ECHO lambda 0.025, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. It completed in 816.080s observed, with
peak observed VRAM 15986 MiB and final loss -0.289983. ECHO env CE improved
from 2.70863 to 1.22403. Adapter verify passed with 400 nonzero LoRA tensors
and LoRA update proxy 0.050735.

Blind `LIMIT=4 SEEDS=1` smoke rejected it: paired base scored 0.98125 with
mean wall-clock 16.72s and no zero rollouts, while H51 scored 0.94375 with
mean wall-clock 45.51s and no zero rollouts. No promotion check was run.
Lesson: real post-failure repair signal is more semantically aligned than the
previous stop-only negatives, and it did not introduce zero rollouts, but it
still slowed the policy enough to lose composite. The next adapter attempt
should treat post-failure repair as only one component inside a broader
mixed-policy corpus, or shift efficiency pressure into runtime/prompt
constraints rather than another small LoRA update.

### H54: step-local concise thinking

H52/H53 first tested the next H49-style direction at broader scale: rank
concise thinking above verbose thinking while holding successful tool behavior
fixed across multiple train-only success anchors. Full-trajectory broad
versions were not locally trainable under the 900s guard. H52 used full-file
`write` payloads and timed out before adapter write for both six groups
(`20179/40901`) and three groups (`12440/20179`). H53 converted the same six
anchors to short `edit` payloads, reducing action tokens from 2796 to 1570,
but still timed out before adapter write at `29047/34961`.

H54 kept the same hypothesis but decomposed each successful trajectory into
one-action suffix groups: post-read edit, post-edit doctest, and post-pass
DONE. It used four train-only success anchors, giving 12 groups and 24
completions. Each group ranked a concise assistant action at reward 1.0 above
the same action with verbose thinking at reward 0.75. ECHO stayed off by
design: the contrast holds tool behavior fixed and has no env-token target.
Dry-run shape was 600 action tokens, 0 env tokens, 11714 context tokens, and
reward stdev 0.125.

Training used rank 4 / alpha 4 / lr `5e-7`, no ECHO, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. It completed in 729.045s observed, with
peak observed VRAM 15995 MiB and final loss -0.000651. Adapter verify passed
with 400 nonzero LoRA tensors and LoRA update proxy 0.174229.

Blind `LIMIT=4 SEEDS=1` smoke was positive: paired base scored 0.69375 with
one zero rollout and mean wall-clock 40.34s, while H54 scored 0.8875 with no
zero rollouts and mean wall-clock 40.94s. Because prior concise/stop adapters
produced smoke false positives, H54 went to a blind `LIMIT=8` promotion check.
It cleared that gate: paired base scored 0.84296875 with no zero rollouts and
mean wall-clock 64.68s, while H54 scored 0.903125 with no zero rollouts and
mean wall-clock 67.75s. Lesson: broad thinking-efficiency data becomes locally
trainable when represented as one-action suffix ranking instead of full
trajectory ranking. H54 is kept with a caveat: the gain is meaningful but
slower, so the next iteration should run a fresh confirmation before chaining
from it, or train a lower-dose/rank variant to reduce the 0.174 delta proxy.
