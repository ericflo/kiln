# Agentic GRPO Consolidated Lessons Report

Generated from the local kiln checkout on `main` at `3c22d175`, covering
everything under `capabilities/agentic-grpo/`: capability specs, iter logs,
closeouts, summary JSONs, run READMEs, helper code, and `kiln-polish.jsonl`
notes.

This report separates three evidence levels:

- **Verified results**: measured adapters, eval summaries, and final writeups.
- **Infrastructure discoveries**: bugs or operational constraints found while
  running the loops.
- **Scaffold hypotheses**: capability designs that have not yet been run.

## Executive Summary

Agentic GRPO on Pi trajectories is real, but narrow and highly conditional.
It produced meaningful wins on capabilities with actual behavioral headroom:
`pi-faithful-completion` (+0.0828 composite), `pi-code-comprehension`
(+0.1293), `pi-doctest` (+0.042 mean with large variance reduction), and
`pi-code-search` (+0.024 5-eval mean, +0.057 peak). On near-saturated tasks,
GRPO mostly regressed or only moved small format sub-scores: `pi-diff-patch-
apply` found the base model remained strongest, and `pi-failure-triage` found
only a +0.006 composite lift.

The dominant scientific lesson is: **GRPO needs non-saturated reward signal**.
When base is already around 0.94 to 0.97 composite, the policy gradient often
harms the largest clean class while chasing sparse hardest-class headroom.
When base is around 0.60 to 0.88 and the rubric exposes real process defects,
small LoRA updates can improve behavior.

The dominant engineering lesson is: **the harness matters as much as the
loss**. Multiple early "model regressions" were later traced to adapter-load
mistakes, stale kiln-server state, pi session schema drift, or Qwen thinking
defaults. Every future cap needs adapter verification, fresh-server evals,
multi-seed measurement, and a rubric sanity suite before claiming a result.

The dominant ECHO lesson is: **ECHO gradients are real and useful, but not a
magic default for every shape**. `pi-terminal-bench-lite` proved ECHO's env-CE
gradient reaches LoRA weights and reduces env-token loss. Default lambda=0.05
is generally the safe setting; lambda=0.075 helped code comprehension; 0.10
often hurt. ECHO-only / `--no-policy-loss` is a valuable mode for verifier-free
continuation and saturated tasks where policy gradient is destructive.

## Capability Scoreboard

| Capability | Status | Best measured result | Main lesson |
| --- | --- | --- | --- |
| `pi-terminal-bench-lite` | Infrastructure validation complete, paper-scale eval not run | 5 synthetic GPU iters; ECHO firing 264+ times; LoRA-B ON/OFF diff/value ~= 1.97; verifier-free chaining validated | ECHO and `--no-policy-loss` work end-to-end; `--base-adapter` had to become real weight loading, not lineage-only. |
| `pi-doctest` | Shipped kept adapter `pi-doctest-iter5` | 3-seed iter-5 family 0.8958 +/- 0.003 vs base mean 0.8537; +4.2 pp; 15x tighter eval variance; 25% faster | Strong-signal filtering beats naive more-data/more-epoch scaling; single-seed +11 pp was lucky. |
| `pi-code-search` | Shipped kept adapter `pi-code-search-iter5-h5-replay-iter1` | Peak 0.6004 vs base 0.5432; 5-eval mean 0.5676 +/- 0.030; +0.06 outcome | Real but modest; eval variance is comparable to lift; adapter-load and server-drift bugs initially inverted the story. |
| `pi-code-comprehension` | Kept adapter iter 4 | 0.7405 vs 0.6112 baseline; +0.1293; 4x faster rollouts | ECHO lambda=0.075 improved line grounding; more data at unchanged lr overtrained; Qwen thinking-template change broke later Pi rollouts. |
| `pi-faithful-completion` | 50-iter loop complete, kept iter 50 | 0.8065 vs 0.7237 baseline; +0.0828 | Search combinations, not axes. Temperature 0.6 + light prompt + lr=3e-5 compounded and broke the apparent ceiling. |
| `pi-failure-triage` | 50 slots attempted, 35 valid evals, kept iter 2 | 0.9720 vs 0.9656 baseline; +0.0064, format +12.5 pp | Base 4B was already at ceiling on root-cause fixing; only format moved. Harder bugs or multiplicative format gate needed. |
| `pi-diff-patch-apply` | 25 of 50 attempted, no adapter beats base | Base 0.9419; best trained iter 2 0.9246; best mechanism iter 25 ECHO-only 0.9233, drift 1.000 | On saturated rewards, GRPO policy gradient is the harm vector. Use base, ECHO-only regularization, OPD, or harder evals. |
| `pi-compaction` | Full-length rollout/eval pipeline built; training blocked/no-op | Base full-length eval ~= 0.314; train rollouts ~= 0.57; no adapter improvement | Long-context GRPO is the systems bottleneck; after training became possible, eval output stayed byte-identical, suggesting adapter/load/gradient no-op. |
| `pi-script-fixup` | Scaffold | No local iter rows | Planned paper section 5.5 verifier-free ECHO adaptation on PyTerm/ITD/OOD tasks. |
| `pi-precondition-check` | Scaffold | No iter rows | Highest-ranked future cap: train read-before-write and clean stale-state exit. |
| `pi-tool-call-efficiency` | Scaffold | No iter rows | Wrap other caps to directly train fewer, parallel, terminating tool calls. |
| `pi-shell-hygiene` | Scaffold | No iter rows | Train positive long-running-process patterns and ban SSH polling loops. |
| `pi-test-interpretation` | Scaffold | No iter rows | Train median-of-3, warmup artifact recognition, and flake classification. |
| `pi-source-mod-workflow` | Scaffold | No iter rows | Train full clone -> branch -> edit -> test -> PR workflow sequencing. |

## Shared Architecture Lessons

### ECHO Is The Default Because Agentic Rollouts Need Env Modeling

The shared README is correct: policy-gradient on action tokens alone is
insufficient for multi-turn tool agents. The next action depends on the last
tool result, and tool-result tokens are environment observations. ECHO adds a
small env-token cross-entropy term on those observation tokens so the model
learns to predict its environment, not just its own action text.

`lib/pi_trajectory.py` is the durable shared artifact from this work. It maps
Pi session JSONL into kiln's canonical trajectory schema:

- assistant segments -> `kind: "action"` for policy targets
- tool result segments -> `kind: "observation"` for ECHO env-CE targets
- system/user segments -> `kind: "context"` when included
- warning prefixes -> `warning_prefix_len` so warning boilerplate can be masked

The parser also carries several hard-won Pi schema fixes:

- Pi tool-call arguments are under `input`, not `arguments`.
- Pi 0.75.1 emits role `tool`; Pi 0.75.3 can emit `toolResult`; kiln must
  normalize both to canonical `tool`.
- Pi session events use `message` singular in current observed JSONL, not the
  top-level shape older kiln parsers assumed.

Future caps should import `lib/pi_trajectory.py`; they should not reimplement
Pi rendering in each rollout script.

### ECHO Was Proven Mechanically Before Being Trusted Scientifically

The `pi-terminal-bench-lite` synthetic runs are the clearest proof:

- Iter 1: paired ECHO on/off training from the same seed produced materially
  different LoRA-B weights; diff/value ratio ~= 1.99.
- Iter 2: tracing made ECHO observable; 24/24 completions logged env-CE firing.
- Iter 3: 20 groups produced 80/80 ECHO firing lines and within-task loss drops
  of roughly 9 to 10%.
- Iter 4: `--no-policy-loss` trained cleanly from fresh init; env-CE alone
  drove the same loss-drop shape.
- Iter 5: after `--base-adapter` was fixed to load weights, verifier-free
  continuation from a strong adapter doubled the within-task loss drop to
  roughly 19 to 20%.

These results validate the training plumbing. They do not, by themselves,
validate TerminalBench-style pass-rate gains on real tasks.

## Rubric Lessons

### Outcome-Only Rubrics Saturate Too Easily

`pi-doctest` v0 used only doctest outcome and immediately saturated at 0.958.
That was useless for training the actual agentic skill. The v1 rubric added
tool-call efficiency, tested-before-done, and format compliance, dropping the
baseline to 0.8854 and exposing process headroom.

The same pattern repeated elsewhere:

- `pi-code-search` needed multiplicative grounding so "correct guess without
  search" was capped at 0.40.
- `pi-diff-patch-apply` needed stricter format pillars just to create 1.9 pp
  more headroom.
- `pi-failure-triage` found every root-cause sub-score saturated, leaving only
  `format_compliance` movable.
- `pi-compaction` needed an outcome gate to kill format-template garbage and
  copy-source shortcuts.

Every cap should start with adversarial calibration examples: known-good,
known-bad, and "cheapest way to score without doing the task." If bad examples
score above about 0.4, the rubric is not ready.

### Multiplicative Gates Are Load-Bearing

The successful rubrics generally use `outcome` as a hard floor and then multiply
or gate process quality:

- `pi-code-search`: `outcome * grounding_factor * agentic`.
- `pi-doctest`: `outcome * (efficiency + tested + format + base)`.
- `pi-failure-triage`: `outcome * held_out/root-cause/process sum`.
- `pi-diff-patch-apply`: pass/fail gate with a capped consolation score.

Purely additive rubrics reward partial cheating. Multiplicative gates make the
cap learn the intended capability rather than a proxy.

### Rubrics Can Create Headroom, But Not Real Capability

Strict format weights can make a saturated baseline trainable, but only on the
format dimension. This is useful if format is genuinely important, as in
`pi-failure-triage`, but it does not mean GRPO improved the root capability.
For scientific claims, report which sub-score moved.

## Training Dynamics Lessons

### Strong-Signal Filters Work

`pi-doctest`'s kept recipe filtered rollout groups by reward variance > 0.05.
That removed all-pass/all-fail groups and kept the groups where GRPO had an
actual advantage signal. Iter 5 became the reproducible recipe; iter 10
retrained from fresh rollouts and landed within 0.003 of the iter-5 family mean.

`pi-code-search` also used variance filtering successfully, though its final
closeout found the cap less sensitive to the exact filter threshold once the
adapter-load bug was fixed.

For saturated caps, variance filters often reject everything or keep tiny,
noisy signals. `pi-diff-patch-apply` had to lower filter thresholds to 0.005
and still saw policy-gradient harm.

### More Data And More Epochs Can Overtrain

Several caps found a sweet spot:

- `pi-doctest`: 20 tasks looked huge on one seed, but the reproducible recipe
  was 11 strong-signal groups, one epoch. Forty tasks or two epochs regressed.
- `pi-code-comprehension`: iter 2 used 24 tasks at the same lr as iter 1 and
  overtrained, dropping from 0.707 to 0.589 and slowing rollouts by 4.7x.
- `pi-faithful-completion`: many larger/richer variants were null or negative
  until the correct combination of temperature, prompt, and lr was found.
- `pi-diff-patch-apply`: fewer tasks x more generations overfit the small task
  set and produced the worst recent result.

Do not treat "more groups" as automatically safer. At fixed lr, more groups are
more update budget.

### Single-Seed Claims Are Not Trustworthy

`pi-doctest` is the canonical example. Iter 2's single-seed +11.4 pp looked
like the headline. Replay with fresh rollouts got 0.846, proving the original
was lucky. The honest closeout is +4.2 pp mean with strong variance reduction.

`pi-code-search` also had eval sigma ~= 0.03, comparable to its +0.024 mean
lift. The peak +0.057 is real as a peak, but not the expected deployment lift.

Minimum standard for future claims:

- report baseline variance and adapter variance separately
- re-evaluate the best adapter on multiple eval seeds
- retrain the best recipe with fresh rollouts at least once
- quote the mean, not the max

### Saturated Baselines Prefer ECHO-Only, OPD, Or Harder Eval

On `pi-diff-patch-apply`, base was 0.9419. Full GRPO consistently traded away
clean-class performance to chase noisy incorrect-class headroom. The best
mechanistic trained adapter was iter 25 with `--no-policy-loss`: it still did
not beat base composite, but it improved drift class to 1.000 without the
verbosity pathology of policy-gradient adapters.

On `pi-failure-triage`, base was 0.9656 and all root-cause sub-scores were
already 1.0. The best adapter only moved final-summary format.

For baselines above roughly 0.94, do one of:

- use the base model
- use `--no-policy-loss` as light ECHO regularization
- switch to OPD / teacher distillation
- make the corpus harder
- make the rubric gate the actually important defect

### Prompt Headroom Matters

`pi-faithful-completion` found that a strict discipline prompt made
`no_question` and `no_soft_punt` baseline at 1.0, leaving no gradient there.
A lighter prompt let the model initially expose the defect, then reacquire the
discipline through GRPO. The learned discipline generalized better than the
prompted discipline.

This is subtle: a strong prompt can hide the behavior you are trying to train.

### Rollout Temperature Affects Gradient Quality

`pi-faithful-completion` found temperature 0.6 was the sweet spot:

- temp 0.6 alone: +0.0514
- temp 1.0: negative
- temp 0.5: negative

The interpretation was that lower-but-not-deterministic sampling reduced noisy
within-group advantage variance. Too high was random; too low was too
deterministic.

## Hyperparameter Lessons

### ECHO Lambda

Observed pattern:

- `0.05` is the safe default and was load-bearing in several caps.
- `0.075` helped `pi-code-comprehension`, mainly by improving line grounding.
- `0.10` often hurt (`pi-code-search` early ablations, `pi-diff-patch-apply`,
  `pi-failure-triage`, `pi-faithful-completion`).
- Lower ECHO (`0.025`) hurt `pi-faithful-completion`.
- No ECHO was not universally fatal: `pi-code-search` no-ECHO re-eval lifted,
  and `pi-code-comprehension` no-ECHO still beat baseline. But ECHO often
  improved the best result or the most important sub-score.

Do not assume "more ECHO" is better. Sweep around 0.05 only after the base
recipe is stable.

### Learning Rate

No single lr won everywhere:

- `1e-5` was the default safe point for several caps.
- `3e-5` was essential in `pi-faithful-completion`; at `1e-5`, the LoRA delta
  appeared too small to move BF16 inference logits enough.
- `5e-6` uniquely won the tiny `pi-failure-triage` lift.
- Too-low lr was not always conservative; `pi-code-search` lr=5e-6 regressed.
- `1e-4` or `2e-5` could overshoot badly depending on cap.

Scale lr with effective update budget, data count, and saturation level.

### Rank And Alpha

The default rank=16, alpha=32 is the most reliable baseline.

Discovered constraints:

- rank=32 often did not help and sometimes catastrophically hurt.
- rank=8 could be too small (`pi-faithful-completion`) or less disruptive
  (`pi-diff-patch-apply`).
- alpha/rank ratio > 2 can corrupt behavior; `pi-faithful-completion` rank=16
  alpha=64 collapsed to composite 0.02.
- chain training with mismatched base rank and new rank can silently create a
  broken adapter.

Future scripts should assert base-adapter rank equals requested rank before
training.

### Chaining

Chaining is not automatically compounding.

- `pi-terminal-bench-lite` proved verifier-free chained ECHO training can
  continue reducing env-CE after the base-adapter load fix.
- `pi-faithful-completion` got modest wins from chain-best only when broadening
  the corpus; narrow chained variants eroded gains.
- `pi-diff-patch-apply` chain-best degraded 0.9246 -> 0.8462.
- `pi-code-comprehension` warm-starts often lost progress because reward
  variance collapsed.

Chaining should be treated as a hypothesis, not the default path.

## Infrastructure Discoveries

### Adapter Loading Is Fragile

The most expensive false conclusions came from adapter load semantics.

Hard rules:

1. `cuda_grpo_ablation --output X --adapter Y` writes weights under `X/Y/`.
   Symlink `X/Y/` into `$KILN_MODEL_PATH/adapters/Y`, not outer `X/`.
2. After `POST /v1/adapters/load`, always verify with `GET /v1/adapters`.
3. kiln-server `ensure_adapter` treats a missing `adapter` request field as
   "unload to base" in at least one path. Rollouts that intend to use an
   adapter must explicitly set or preserve it.
4. `--base-adapter` originally recorded lineage only; after the terminal-bench
   work it loads safetensors into `TrainableLoraParams`. Scripts and reports
   before that fix may have assumed chaining when none happened.
5. `--base-adapter` path resolution is easy to misuse; names resolve relative
   to the current output dir, not necessarily prior iter output.
6. Rank mismatch on chain training can silently break the adapter.

### Fresh Server Eval Is Mandatory

`pi-code-search` discovered kiln-server latency creep over long eval sessions:
requests degraded from milliseconds to 80-120 seconds, Pi hit timeouts, and
rollouts scored zero. A fresh server changed several "catastrophic regressions"
into positive re-evals.

Use:

- restart kiln-server before eval
- smoke-test the loaded adapter before full eval
- consider batching evals with periodic restarts
- never trust a load/eval result if wall-clock suddenly jumps

### Qwen Thinking Defaults Can Break Pi

`pi-code-comprehension` session 2 diagnosed an upstream Qwen3.5 chat-template
change: `enable_thinking=true` led to long reasoning traces, empty `content`,
Pi re-prompts, and 180s timeouts. `KILN_DEFAULT_NO_THINK=1` fixed direct
chat-completion latency, but the Pi tool-calling convention had been implicitly
conditioned on the older behavior.

Practical rule:

- For single-turn/direct HTTP rollouts, set `chat_template_kwargs.enable_thinking=false`.
- For Pi-agent rollouts, verify actual tool-call behavior after any template
  change, not just direct curl responses.

### Pi Has Important Operational Gaps

Observed Pi constraints:

- No `--max-turns`; enforce with OS `timeout`.
- Killed sessions leave partial JSONL, which rubrics must handle.
- Fresh Pi defaults to non-kiln provider; run `kiln pi-setup`.
- Session schema changes across Pi versions require robust parsing.

### RunPod And Shell Hygiene Lessons

Repeated operational gotchas:

- Never use `pkill -f "kiln serve"` over SSH; it can kill the wrapper bash
  because the pattern appears in argv. Use `pgrep -x kiln | xargs -r kill -9`.
- Do not run multiple kiln servers against port 8420; healthcheck and kill exact
  process names before restart.
- Use per-cap pod env files, not shared `/tmp/grpo-pod.env`.
- `runpod_api.py bg` plus local `wait-file` can see stale sentinels if the bg
  command starts by deleting the same output dir. Wait on a separate sentinel.
- Pod leases have a 3h TTL with no renewal, causing repeated long-loop churn.
- Do not pipe background drive scripts through `head`; SIGPIPE can kill them.
- `python3 $RP ssh` may append trailing blank lines; parse nonblank output.
- Fresh images may lack `pytest`; install it in bootstrap if rollouts need it.
- Build kiln with `--features cuda`; CPU-mode serve is unusably slow.
- Set `KILN_CUDA_ARCHS` for the actual GPU to reduce build time.
- Use the correct model name/path: Qwen3.5-4B, not Qwen2.5.

### Long-Context GRPO Remains A Frontier

`pi-compaction` is the long-context stress test:

- Full-length eval and rollouts work.
- Long-context training was initially prohibitively slow or OOM.
- Shared-prefix reference forward helped but policy forward/backward remained
  the bottleneck.
- PR #1056 made long-context training tractable by enabling per-layer tile
  reverse across multi-layer checkpoint segments.
- After that, trained adapters still produced byte-identical eval output to
  base, even at lr=1e-3, suggesting adapter-load, effective scale, or gradient
  no-op still needs investigation.

Do not spend 50-iter loops on compaction until that no-op is resolved.

## Capability-Specific Notes

### `pi-terminal-bench-lite`

This cap became the ECHO plumbing receipt rather than a full TerminalBench
reproduction. It caught and fixed:

- missing tracing subscriber in `cuda_grpo_ablation`
- missing uncheckpointed ECHO debug log
- synthetic trajectory shape issues
- `--base-adapter` being lineage-only

The final iter verified base-adapter loading by step-1 loss shift
0.355141 -> 0.316404 and LoRA-B magnitude growth 2.11e-4 -> 4.12e-4.

### `pi-doctest`

The durable result is not the biggest single number, but the recipe:

- v1 multi-component rubric
- strong-signal groups only
- one epoch
- lr=1e-5
- rank=16 alpha=32
- multi-seed verification

The adapter improved reliability and speed more convincingly than raw mean:
23/24 tasks solved consistently, fewer zero rollouts, and lower wall-clock.

### `pi-code-search`

The cap is genuinely trainable, but the gain is modest and noisy. Its most
important contributions were:

- a robust grounded-search rubric
- proof that many recipe variants lift once measured correctly
- adapter symlink fix
- server-drift diagnosis

Closeout supersedes the older live-log interpretation where no-ECHO and
lambda=0.10 looked catastrophic. Fresh-server re-evals showed no-ECHO also
lifted, though the default iter-5 recipe remained the best shipping choice.

### `pi-code-comprehension`

Best result: iter 4, ECHO lambda=0.075, +0.1293 composite.

The remaining bottleneck is invariant coverage. Cross-file recall and format
saturated; line grounding improved; invariants stayed low because AST-derived
gold misses many semantic invariants.

Future work should use curated invariant gold or an LLM judge for invariants,
not just more GRPO over the same auto-extracted corpus.

### `pi-faithful-completion`

This is the best example of why a 50-iter loop can be worth it. Single axes
looked capped; the winning triple combination was discovered late:

- rollout temp 0.6
- light system prompt
- lr=3e-5
- ECHO lambda=0.05

It also produced the clearest LoRA safety lessons: rank mismatch and alpha/rank
ratio > 2 can destroy the adapter.

### `pi-failure-triage`

The rubric is strong: it distinguishes root-cause from symptom fixes with
held-out tests. The corpus is too easy for Qwen3.5-4B: outcome and held-out
pass are already saturated.

Next iteration should not sweep more hyperparameters. It should plant harder,
multi-file bugs and make format compliance more important if final summaries
are the desired behavioral target.

### `pi-diff-patch-apply`

The final summary line is the key: "GRPO's policy gradient is the primary harm
vector on saturated baselines." The ECHO-only iter 25 is mechanistically clean,
but base still wins composite.

This cap is best treated as infrastructure-validated and scientifically
negative for GRPO at 4B scale. Use it to test OPD or harder corpora.

### `pi-compaction`

This cap has high real-world value but is currently blocked by long-context
training and adapter-effect uncertainty. The rubric and full-length eval are
valuable. The next useful step is not another loop; it is a focused debug pass
on why trained adapters produce byte-identical eval responses.

## Scaffold-Only Cap Lessons

These are designs, not results.

### `pi-precondition-check`

Highest-priority future cap. It targets the largest observed failure cluster:
acting on stale or hallucinated state. The core rubric idea is sound:
`outcome * (verified_before_mutation + staleness_detected + no_phantom_edit +
format + base)`.

Key design risk: always emitting `precondition_failed` can cheat stale tasks,
so the sentinel must be task-correct and preceded by a relevant read.

### `pi-tool-call-efficiency`

Best used as a wrapper over existing caps once target tool-call counts are
annotated. The hard part is ground truth: uniform call targets will create
noise. It should start with one source cap, then expand.

### `pi-shell-hygiene`

This should encode the RunPod money-burn lessons into reward:

- no blocking SSH polling loops
- use bg + wait-file with timeouts
- capture logs
- cleanup on failure
- no orphaned processes

A fake-pod SSH shim is required before training safely.

### `pi-test-interpretation`

This cap should train median-of-3 benchmark discipline, flake recognition, and
warning-vs-error classification. It is directly motivated by prior kiln
benchmark false positives.

### `pi-source-mod-workflow`

This cap has external side-effect risk. The first implementation task must be
a sandbox GitHub namespace with a narrowly scoped PAT, not rubric code.

### `pi-script-fixup`

This is the paper section 5.5 verifier-free showcase. It depends on a strong Phase 2
checkpoint and should measure PyTerm/ITD/val100 plus TBLite negative control.
The terminal-bench iter 5 result means the needed `--base-adapter` and
`--no-policy-loss` mechanics now exist.

## Kiln-Itself Improvements

This section is deliberately about changes to kiln, kiln-train, kiln-server,
and the kiln RunPod tooling. It is not more cap-author advice. The agentic-GRPO
bucket surfaced a consistent pattern: the model/loss can work, but we lose time
and scientific confidence because the underlying system makes it too easy to
run the wrong adapter, evaluate against a degraded server, train on a no-op
gradient, or burn pod cycles on brittle orchestration.

### P0: Make Adapter State Explicit, Verifiable, And Hard To Misuse

Adapter lifecycle was the biggest source of false conclusions. `pi-code-search`
initially looked like repeated catastrophic regression because adapters were
symlinked at the wrong directory level and `/v1/adapters/load` failed silently.
`pi-faithful-completion` found that missing `adapter` in chat requests could
unload the active adapter. `pi-terminal-bench-lite` found that `--base-adapter`
was originally lineage-only, not weight-loading. Several chain experiments
collapsed from rank mismatch or unsafe alpha/rank scaling.

Kiln should make adapter state a first-class invariant:

1. **Define request adapter semantics clearly in kiln-server.**
   - Missing `adapter` should mean one explicit policy, documented and tested.
   - Prefer either "keep current server adapter" or "use base", but never
     silently switch in a helper named `ensure_adapter`.
   - Use explicit `adapter: null` or a dedicated `/v1/adapters/unload` call for
     unload semantics.
   - Log adapter transitions with request id, old adapter, new adapter, and
     reason.

2. **Make `/v1/adapters/load` validate the actual safetensors path.**
   - Check that `adapter_model.safetensors` exists at the resolved directory.
   - Check that `adapter_config.json` exists and matches expected model shape.
   - Return structured 4xx errors for wrong nesting, missing files, rank
     mismatch, alpha/rank danger, or incompatible target modules.
   - Do not return "loaded" until weights are actually resident and selected.

3. **Extend `GET /v1/adapters`.**
   It should return:
   - `active_adapter`
   - `loaded_adapters[]`
   - `available_adapters[]`
   - path used for each adapter
   - rank, alpha, target modules, dtype, file size, sha256
   - parent/base adapter metadata if present
   - last load error per attempted adapter

4. **Add a `kiln adapter verify <name>` CLI.**
   It should perform:
   - directory shape validation
   - config/safetensors consistency check
   - rank/alpha safety check
   - load through kiln-server
   - `GET /v1/adapters` confirmation
   - one tiny deterministic completion with and without the adapter
   - a logit-delta check confirming the adapter changes logits

5. **Canonicalize trainer output layout.**
   `cuda_grpo_ablation --output X --adapter Y` writing to `X/Y/` is fine, but
   the tool should print a final machine-readable `adapter_dir` and optionally
   install it into `$KILN_MODEL_PATH/adapters/Y` itself. Cap scripts should not
   need to know nested output conventions.

6. **Reject unsafe chain training before it starts.**
   - If `--base-adapter` rank != `--rank`, fail with a clear error unless a
     deliberate conversion flag is passed.
   - If alpha/rank ratio > 2, warn loudly or require `--allow-high-lora-scale`.
   - If `--base-adapter` path is a name, print the resolved absolute path and
     fail if it does not exist.
   - Include loaded tensor count and shape count in the training receipt.

7. **Add adapter effect tests to CI.**
   A tiny synthetic LoRA should:
   - load successfully
   - appear in `/v1/adapters`
   - alter logits on a fixed prompt
   - survive a chat request that omits unrelated optional fields
   - unload only through explicit unload

### P0: Produce A Receipt For Every Training Run

The cap logs had to reverse-engineer too much from stdout. Every GRPO/SFT run
should emit a single structured receipt JSON next to the adapter, and the
trainer should fail if it cannot write it.

Minimum receipt fields:

- kiln git commit and dirty flag
- model path, tokenizer hash, model config hash
- base adapter name/path/hash, if any
- output adapter name/path/hash
- rank, alpha, alpha/rank, target modules
- lr, epochs, seed, mode, KL coeff, clip epsilon, dynamic sampling mode
- ECHO enabled, lambda, env mask mode, warning filter
- `no_policy_loss` true/false
- number of groups read, filtered, trained
- reward mean, reward stdev, group variance distribution
- action token count and env token count
- ECHO CE initial/final if available
- policy loss, KL, entropy, clip fraction, grad norm trend
- LoRA delta norm by module
- train wall-clock, peak VRAM, GPU type, CUDA arch
- exact data file sha256 and a sample row schema hash

The receipt should include hard assertions:

- fail if zero groups train unless `--allow-zero-groups`
- fail if all action masks are empty
- fail if ECHO is enabled but all env masks are empty
- warn if reward variance is below a configurable floor
- warn if LoRA delta norm is effectively zero
- warn if output adapter logits are byte-identical to base on a smoke prompt

This would have caught or shortened several loops: compaction no-op adapters,
base-adapter lineage-only confusion, no groups after filtering, and ECHO
mistakenly assumed inactive or active.

### P0: Add First-Class Trajectory Inspection

Agentic GRPO depends on masks. A cap author needs to know exactly which tokens
are action targets, which are env targets, and which are context. Right now that
requires reading Python and Rust internals.

Add a `kiln trajectory inspect` tool that accepts a Pi session JSONL or
ScoredRollout JSONL and prints:

- rendered chat messages
- action/env/context segment boundaries
- token counts per segment
- warning-prefix stripping decisions
- mask density
- malformed role normalization warnings
- assistant/tool alternation problems
- first 200 decoded action/env target tokens

Add a `--html` or `--json` output for debugging long trajectories. This should
live in kiln or kiln-train, not in each cap.

This would also be the right place to enforce Pi schema compatibility:

- accept both `tool` and `toolResult`
- accept `message` singular and older variants
- render tool-call arguments from `input`
- normalize to the canonical role set before Qwen chat templates see it

### P0: Stabilize Evaluation Serving

`pi-code-search` showed kiln-server could degrade over an extended eval: request
latency jumped from milliseconds to 80-120 seconds, Pi timed out, and the eval
looked like a bad adapter. Restarting the server changed the result. That makes
scientific iteration shaky.

Kiln-server should expose enough state to detect and prevent this:

1. **Health endpoint with operational metrics.**
   `/v1/health` or `/v1/debug/health` should return:
   - uptime
   - active adapter
   - request count
   - p50/p95/p99 latency over recent window
   - decode tok/s over recent window
   - active CUDA graph count
   - VRAM allocated/reserved
   - prefix cache size
   - loaded adapter count
   - last N errors

2. **Slow-request watchdog.**
   If a request exceeds a configured threshold, log:
   - prompt tokens, max tokens, adapter, batching engine state
   - current model step
   - CUDA graph capture/replay state
   - kernel disable flags
   - whether thinking mode is enabled

3. **Eval-safe server mode.**
   Add `kiln serve --eval-mode` that:
   - disables or bounds caches that grow across requests
   - sets deterministic sampling defaults
   - sets no-thinking by default unless overridden
   - exposes per-request adapter explicitly
   - optionally restarts worker state after N requests

4. **Batch eval helper.**
   Provide `kiln eval-rollouts` or a server-side batch endpoint that evaluates
   a list of prompts/tasks while resetting per-request state in a controlled
   way. Cap scripts currently rebuild this loop repeatedly.

5. **Adapter load/unload leak tests.**
   Create a stress test that loads/unloads adapters 100 times and runs short
   completions, checking latency, VRAM, and correctness. This targets the exact
   state-drift class observed in code search.

### P0: Fix Qwen Thinking Defaults As A Product Surface

Multiple caps were derailed or slowed by Qwen thinking behavior. Direct
single-turn tasks often want `enable_thinking=false`; Pi tool-call loops may
depend on template conventions and need explicit verification.

Kiln should make this impossible to miss:

- Document `chat_template_kwargs.enable_thinking`.
- Add a top-level server config `default_enable_thinking`.
- Make `KILN_DEFAULT_NO_THINK=1` visible in `/v1/health`.
- Log when a request uses thinking mode.
- Add a response field indicating whether content was empty because reasoning
  was emitted separately.
- Add tests for direct chat and Pi-style tool-call prompts with thinking on/off.
- Consider a model-specific default for Qwen3.5-4B that disables thinking for
  OpenAI-compatible tool-agent usage.

### P1: Make Long-Context GRPO A Supported Mode

`pi-compaction` is strategically important but exposed long-context training
as the least mature path. PR #1056 improved memory behavior, but the cap still
ended with byte-identical outputs after training.

Needed kiln-train work:

1. **Long-context benchmark suite.**
   Add a committed synthetic benchmark with 8K, 16K, 32K, and 64K-token
   trajectories. Measure:
   - forward/backward wall-clock
   - peak VRAM
   - token throughput
   - launch count
   - gradient checkpoint setting
   - tile size

2. **Progress logging before long stalls.**
   Long-context training should emit progress before expensive regions:
   - tokenize complete
   - masks built
   - reference forward start/end
   - policy forward start/end
   - backward start/end per segment
   - optimizer step start/end

3. **Adapter-effect smoke eval after training.**
   For compaction-sized inputs, automatically run a short held-out prompt and
   compare output/logits to base. If byte-identical across multiple prompts,
   emit a high-severity warning.

4. **Investigate near-zero gradients.**
   Add debug mode that writes:
   - per-module gradient norms
   - per-module LoRA delta norms
   - policy/ref logprob deltas
   - KL and advantage statistics
   - mask nonzero counts

5. **Make shared-prefix optimizations visible.**
   The logs should say whether the shared-prefix reference forward path fired,
   how many tokens were shared, and how much work was saved.

### P1: Improve `cuda_grpo_ablation` CLI Ergonomics

Several scripts failed because the CLI did not match assumptions.

Concrete changes:

- Add `--print-effective-config`.
- Add `--dry-run` that loads data, filters groups, builds masks, validates
  adapter paths, and exits before GPU training.
- Add `--reward-var-filter` as an official option rather than cap scripts
  hand-filtering JSONL.
- Add `--min-groups` and `--on-empty-groups {fail,train-all,skip}`.
- Make `--no-echo` and `--echo-lambda` mutex errors explain the fix.
- Print all baked defaults: KL coeff, clip epsilon, dynamic sampling, advantage
  mode.
- Either accept explicit `--kl-coeff` and `--clip-epsilon`, or make their
  absence obvious in help text.
- Add `--base-adapter-required` for chain experiments that must fail if the
  base adapter does not load.
- Add `--adapter-install-dir` to install completed adapters into a serve-ready
  directory safely.

### P1: Build A Cap Runner Into Kiln

Every cap recreated the same loop:

1. run rollouts
2. score rollouts
3. filter groups
4. train
5. install adapter
6. restart or healthcheck server
7. eval
8. append JSONL
9. back up artifacts

Kiln should provide a generic runner with cap-specific hooks:

```text
kiln cap run \
  --cap capabilities/agentic-grpo/pi-code-search \
  --iter 5 \
  --recipe recipes.json:h5 \
  --pod-pool \
  --backup b2://...
```

The runner should own:

- server lifecycle
- adapter install/load verification
- health checks
- smoke eval
- summary JSON schema
- receipt capture
- B2 backup manifest
- failure row logging
- no-cascade behavior after failed iter

This would prevent each cap from rediscovering `pkill -f`, stale sentinel,
adapter symlink, and env-file contamination bugs.

### P1: Add Built-In Multi-Seed Evaluation

The report repeatedly shows that single-seed results overclaim. Kiln should
make multi-seed the easy path:

- `kiln eval-adapter --adapter A --seeds 3 --tasks eval.tasks.jsonl`
- outputs mean, stdev, confidence interval, zero-counts, wall-clock
- can compare base vs adapter using paired task seeds
- stores a machine-readable `eval-summary.json`
- fails or warns if sigma is comparable to the claimed lift

The cap runner could require a minimum seed count before marking a result as
`ship`.

### P1: Add Direct HTTP Rollout Mode For Non-Tool Capabilities

Several caps are "agentic input, single-turn output" rather than true Pi tool
loops: compaction, faithful completion, and potentially code comprehension
after the Qwen thinking issue. Pi adds overhead and schema risk when the model
does not need to choose tools.

Kiln should provide a direct rollout harness:

- reads task JSONL
- builds chat-completion request
- sets adapter explicitly per request
- sets thinking defaults
- writes response, logprobs if available, latency, token counts
- calls a Python/Rust scorer
- emits ScoredRollout JSONL for GRPO

This would let caps choose Pi only when tool choice is part of the behavior
being trained.

### P1: Make Performance Problems Actionable

The caps surfaced repeated performance blind spots:

- Pi orchestration dominated short task wall-clock.
- Thinking mode caused 40x slowdowns.
- Long-context training had superlinear cost and OOM cliffs.
- ECHO uncheckpointed path initially did extra softmax work.
- Server state drift caused massive per-request latency.

Kiln should standardize performance counters across serve and train:

- prompt tokens, completion tokens, tok/s
- TTFT and ITL
- per-request wall-clock split: template, prefill, decode, adapter switch
- train split: tokenize, ref forward, policy forward, backward, optimizer
- CUDA kernel launch count where feasible
- peak VRAM and allocator stats
- cache hit/miss counts

Every summary JSON should carry these fields so training can optimize behavior
and speed together.

### P1: Make The Pod Pool Fit Long Experimental Loops

The 3h lease TTL repeatedly interrupted 50-iter loops. That is outside the
model, but inside the kiln workflow.

Improvements:

- Add `--lease-ttl-seconds` to `ce kiln-pod-acquire`.
- Add `ce kiln-pod-renew --lease`.
- Add `ce kiln-pod-current-lease`.
- Preserve `/tmp/preserved-adapters` or sync it before hibernation.
- Make pod hibernation visible to the cap runner before it happens.
- Provide a "long experiment" lease mode with explicit owner, cap name, and
  heartbeat.

### P2: Improve Model/Adapter Safety Checks

Add guardrails that prevent known-bad adapters from being shipped:

- Reject or warn on alpha/rank ratio > 2.
- Reject base-adapter rank mismatch.
- Run a small canary eval after adapter training:
  - empty-output rate
  - average output length
  - logits finite
  - response differs from base
  - no 10x latency regression
- Track adapter verbosity. `pi-diff-patch-apply` showed GRPO policy-gradient
  adapters could double or triple session time by becoming chatty.
- Add "adapter quarantine" status if canaries fail.

### P2: First-Class Artifact Manifests

Every run should leave a complete manifest. Current B2 backups are good, but
cap-specific.

Standard manifest:

- adapter safetensors hash
- adapter config hash
- training data hash
- eval data hash
- train summary hash
- eval summary hash
- kiln commit
- pod id and GPU
- restore instructions
- parent adapter
- exact command line

`kiln adapter restore <manifest>` should download/install/load the adapter and
verify it.

### P2: Better Failure Taxonomy

Several iter logs contain `VOIDED`, `FAILED-no-eval`, `INVALID-pod-state`,
`broken`, and null composite rows. Kiln should standardize failure reasons:

- `train_no_groups`
- `train_crash`
- `adapter_load_failed`
- `server_unhealthy`
- `pod_unreachable`
- `eval_timeout`
- `scorer_error`
- `artifact_backup_failed`
- `schema_error`

That lets a 50-iter runner make correct decisions: retry infra failures, skip
bad recipes, or stop after repeated systemic failures.

### P2: Turn Agentic-GRPO Learnings Into Tests

Every hard-won bug should become a regression test. Suggested tests:

- adapter nested directory load fails clearly
- missing adapter field does not silently unload unless specified
- base-adapter actually changes step-1 loss when loaded
- Pi 0.75.1 and 0.75.3 session roles parse to canonical trajectory
- ECHO enabled with env tokens logs nonzero env count
- `--no-policy-loss` zeroes policy loss but trains ECHO
- rank mismatch fails before training
- alpha/rank > 2 requires override
- `KILN_DEFAULT_NO_THINK=1` changes Qwen chat output as expected
- server survives repeated load/eval/unload cycles without latency creep

### The Highest-Leverage Kiln Roadmap

If prioritizing by impact on future agentic-GRPO work:

1. **Adapter lifecycle hardening**: explicit semantics, verify endpoint, receipt,
   rank/alpha checks.
2. **Training receipt and trajectory inspector**: make masks, gradients, and
   ECHO visible.
3. **Eval server health/stability**: health metrics, eval mode, load/unload
   stress tests.
4. **Generic cap runner**: stop reimplementing brittle orchestration in every
   cap.
5. **Long-context GRPO debug path**: unblock compaction and other real agent
   memory tasks.
6. **Pod lease renewal and artifact preservation**: make 50-iter loops routine
   instead of heroic.

## Recommendations

### For Future Agentic-GRPO Caps

1. Build calibration before rollout. Good examples must score high; bad
   examples must score low; adversarial shortcuts must be capped.
2. Run iter 0 baseline and inspect which sub-score actually has headroom.
3. Avoid saturated caps unless the rubric gates a real, important defect.
4. Use ECHO lambda=0.05 as default; sweep only after a stable recipe exists.
5. Start rank=16 alpha=32; assert rank compatibility when chaining.
6. Prefer strong-signal filtering over more epochs.
7. Measure at least two eval seeds before claiming success.
8. Re-train the best recipe once with fresh rollouts before calling it robust.
9. Report wall-clock and zero-rollout counts as first-class metrics.

### For Harness Hardening

1. Add adapter-load verification to every `run_iter.sh`.
2. Make missing adapter field mean "keep current" or force callers to be
   explicit with a loud error.
3. Add lease renewal or configurable TTL to `ce kiln-pod-acquire`.
4. Add a `ce kiln-pod-current-lease` helper.
5. Make `KILN_DEFAULT_NO_THINK=1` the safe documented default for direct
   chat-completion rollouts.
6. Add a built-in `kiln cap-smoke` that verifies: CUDA build, model path,
   Pi setup, adapter load, direct chat, Pi tool call, and trajectory parse.
7. Add hard preflight checks for banned SSH polling patterns and `pkill -f`.

### Best Existing Templates To Reuse

- Rubric pattern: `pi-code-search/rubric.py`, `pi-doctest/rubric.py`,
  `pi-failure-triage/rubric.py`.
- Trajectory parsing: `lib/pi_trajectory.py`.
- 50-iter process discipline: `pi-faithful-completion`.
- ECHO infrastructure validation: `pi-terminal-bench-lite/runs/iter5-*`.
- Adapter/server gotcha documentation: `pi-code-search/closeout.md` and
  per-cap `kiln-polish.jsonl`.

## Source Map

Primary closeouts and writeups:

- `pi-doctest/closeout.md`
- `pi-code-search/closeout.md`
- `pi-code-search/FINAL_RESULTS.md`
- `pi-code-comprehension/WRITEUP.md`
- `pi-faithful-completion/closeout.md`
- `pi-failure-triage/FINAL_WRITEUP.md`
- `pi-diff-patch-apply/FINAL_WRITEUP.md`
- `pi-diff-patch-apply/PROGRESS_NOTES.md`
- `pi-compaction/capability.md`
- `pi-terminal-bench-lite/runs/*/README.md`

Structured data:

- all `pi-*/capability.jsonl`
- `pi-doctest/iter2-artifacts/*summary.json`
- `pi-compaction/iter-artifacts/*summary.json`
- `pi-faithful-completion/eval-summaries/*.json`
- per-cap `kiln-polish.jsonl`

Shared implementation:

- `README.md`
- `lib/pi_trajectory.py`
- `lib/test_pi_trajectory.py`
