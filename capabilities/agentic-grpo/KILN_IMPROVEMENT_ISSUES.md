# Kiln Improvement Issues For Agentic GRPO Work

This is a concrete backlog of improvements to kiln itself, informed by the
agentic-GRPO capability experiments under `capabilities/agentic-grpo/`.

Scope is intentionally narrow. This list excludes RunPod lifecycle, B2 backup,
GitHub workflow, `ce` pod-pool ergonomics, and cap-specific training recipes.
The tasks below are for kiln-server, kiln-train, kiln-model integration,
examples, CLIs, tests, and kiln-owned docs. If all of these were completed,
future agentic-GRPO work would be much more reliable, measurable, and fast to
iterate.

Each item is written so it can become a GitHub issue or task directly.

## P0: Make Correct Experiments Hard To Misrun

### 1. Define Explicit Adapter Semantics For Chat Completion Requests

**Area:** `crates/kiln-server`

**Problem:** Agentic evals discovered that a chat request omitting the
`adapter` field can be interpreted as "unload active adapter" in at least one
path. That makes evals silently run against base after a successful adapter
load. It invalidates results because the request body shape, not the explicit
adapter operation, controls model state.

**Task:** Define and implement one clear adapter selection policy for
`/v1/chat/completions`.

**Proposed behavior:**

- Missing `adapter`: use current server default adapter, without changing it.
- `adapter: null` or `adapter: ""`: use base for this request only, without
  changing the default.
- `adapter: "<name>"`: use that adapter for this request; fail if it is not
  loaded or available.
- `/v1/adapters/load` and `/v1/adapters/unload`: the only endpoints that mutate
  server default adapter state.

**Acceptance criteria:**

- Unit tests cover missing, null, empty, valid, and invalid `adapter` values.
- Server logs every adapter transition with old adapter, new adapter, request
  id, and reason.
- A loaded adapter remains active across requests that omit `adapter`.
- Documentation states the exact semantics.

### 2. Make Adapter Load Fail Loudly On Wrong Directory Layout

**Area:** `crates/kiln-server`

**Problem:** `cuda_grpo_ablation --output X --adapter Y` writes weights under
`X/Y/`. One cap symlinked `X/` instead of `X/Y/`; `/v1/adapters/load` failed,
the server kept running with the previous adapter, and several evals were
misread as model regressions.

**Task:** Harden `/v1/adapters/load` so it validates the resolved adapter
directory before mutating state.

**Acceptance criteria:**

- Loading an adapter directory without `adapter_model.safetensors` returns a
  structured 4xx error and does not change active adapter state.
- Loading an adapter directory without `adapter_config.json` returns a
  structured 4xx error and does not change active adapter state.
- The error message includes the resolved absolute path and missing files.
- A test covers the common nested-output mistake.
- The server logs load failures at warn level with path and reason.

### 3. Expose Complete Adapter Registry State

**Area:** `crates/kiln-server`

**Problem:** Cap scripts had to infer whether an adapter was loaded by probing
loosely documented endpoints. `GET /v1/adapters` needs enough information to
prove what will actually run.

**Task:** Expand the adapter registry response.

**Response should include:**

- `active_adapter`
- `loaded_adapters[]`
- `available_adapters[]`
- adapter directory path
- `adapter_model.safetensors` sha256
- file size
- rank, alpha, alpha/rank
- target modules
- parent/base adapter metadata if present
- last load error per adapter name, if any

**Acceptance criteria:**

- `GET /v1/adapters` is sufficient for a script to verify a load.
- Invalid adapter directories appear with an error status instead of being
  silently omitted.
- Tests cover active, loaded-but-inactive, available-unloaded, and invalid
  adapters.

### 4. Add `kiln adapter verify`

**Area:** kiln CLI / `crates/kiln-server`

**Problem:** Future cap authors need one command that proves an adapter is
installed, loadable, selected, and behaviorally active.

**Task:** Implement `kiln adapter verify <name-or-path>`.

**Command should:**

- Validate adapter directory shape.
- Validate config and safetensors consistency.
- Load the adapter through kiln-server or an offline loader.
- Confirm registry state after load.
- Run one fixed prompt with base and adapter.
- Compare logits or token probabilities to prove the adapter changes behavior.
- Print a machine-readable JSON receipt.

**Acceptance criteria:**

- Command exits nonzero for wrong nested path, missing file, rank mismatch, or
  no measurable adapter effect.
- Command exits zero for a known-good tiny test adapter.
- Receipt includes hashes, rank, alpha, target modules, and logit-delta summary.

### 5. Canonicalize Trainer Output And Optional Adapter Installation

**Area:** `crates/kiln-train`, `examples/cuda_grpo_ablation.rs`

**Problem:** Every cap script had to know the trainer's nested output
convention. That produced the adapter symlink bug.

**Task:** Make training output self-describing and optionally installable.

**Changes:**

- At successful train end, print `ADAPTER_DIR=<absolute path>` on its own line.
- Write `adapter_receipt.json` containing `adapter_dir`.
- Add `--install-adapter-dir <dir>` to atomically copy or symlink the actual
  adapter directory into a serve-ready adapter registry.
- Add `--install-adapter-name <name>` to override install name.

**Acceptance criteria:**

- Cap scripts no longer need to compute `output/adapter` paths themselves.
- Install operation validates the target before replacing existing symlink.
- Receipt and stdout agree on the adapter path.

### 6. Enforce Base-Adapter Shape Compatibility Before Training

**Area:** `crates/kiln-train`

**Problem:** Chain-training with a base adapter of rank R1 and a new training
rank R2 can silently corrupt behavior. `pi-faithful-completion` saw
rank-mismatch adapters collapse to near-floor composite.

**Task:** Validate `--base-adapter` shape before optimizer setup.

**Acceptance criteria:**

- If `--base-adapter` rank differs from `--rank`, training fails before GPU
  work starts.
- If target module sets differ, training fails before GPU work starts.
- If tensor shapes differ, training fails with the exact offending tensor name.
- A deliberate `--allow-adapter-shape-conversion` flag is required for any
  future conversion behavior.
- Tests cover rank mismatch, missing tensor, extra tensor, and valid match.

### 7. Guard Unsafe LoRA Scaling

**Area:** `crates/kiln-train`

**Problem:** Alpha/rank ratio above 2 caused a corrupted adapter in practice.
The trainer currently allows dangerous settings without warning.

**Task:** Add LoRA scaling validation.

**Acceptance criteria:**

- If `alpha / rank > 2.0`, training fails by default with a clear explanation.
- `--allow-high-lora-scale` can override for deliberate experiments.
- Receipt records `rank`, `alpha`, and `alpha_over_rank`.
- Tests cover accepted and rejected configurations.

### 8. Emit A Structured Training Receipt For Every GRPO/SFT Run

**Area:** `crates/kiln-train`

**Problem:** Several experiments required forensic reconstruction from logs.
There was no single artifact proving what data, masks, loss settings, adapter,
and model commit were actually used.

**Task:** Write `train_receipt.json` next to every adapter.

**Minimum fields:**

- kiln git commit and dirty flag
- model path and model config hash
- tokenizer config hash
- base adapter path/hash, if any
- output adapter path/hash
- training data path and sha256
- rank, alpha, lr, epochs, seed, mode
- KL coeff, clip epsilon, dynamic sampling settings
- ECHO enabled, lambda, env mask mode, warning filter
- `no_policy_loss`
- groups read, groups filtered, groups trained
- reward mean/stdev and group variance histogram
- action token count, env token count, context token count
- train wall-clock, peak VRAM if available
- LoRA delta norm summary by module

**Acceptance criteria:**

- Receipt is always written on successful training.
- Receipt is written with `"status": "failed"` on known validation failures.
- JSON schema is documented and stable.
- Existing cap scripts can parse it without scraping stdout.

### 9. Add GRPO Dry-Run Validation Mode

**Area:** `examples/cuda_grpo_ablation.rs`, `crates/kiln-train`

**Problem:** Many mistakes can be detected before GPU training: empty groups,
all-zero masks, wrong schema, no env tokens with ECHO enabled, zero reward
variance, missing base adapter.

**Task:** Add `--dry-run`.

**Dry run should:**

- Load and validate the JSONL.
- Build trajectory masks.
- Apply group filters.
- Resolve and validate adapters.
- Print effective config.
- Print action/env token counts.
- Print group reward variance summary.
- Exit before model forward/backward.

**Acceptance criteria:**

- Dry run catches malformed trajectory roles.
- Dry run catches empty `action_mask`.
- Dry run catches ECHO enabled with empty `env_mask`.
- Dry run catches zero groups after filtering unless explicitly allowed.
- Dry run writes a receipt.

### 10. Add A Trajectory Inspector CLI

**Area:** `crates/kiln-train` or kiln CLI

**Problem:** Agentic GRPO depends on correct action and env masks. Today,
debugging masks requires reading Python helper code and Rust internals.

**Task:** Implement `kiln trajectory inspect <session-or-rollout.jsonl>`.

**Output should show:**

- rendered messages
- segment role and kind
- token counts by segment
- action token count
- env token count
- warning-prefix stripped bytes
- decoded first N action target tokens
- decoded first N env target tokens
- schema warnings

**Acceptance criteria:**

- Works on Pi 0.75.1-style and 0.75.3-style session JSONL.
- Works on kiln ScoredRollout JSONL.
- Emits JSON with `--json`.
- Fails nonzero when no trainable action tokens are present.

### 11. Move Pi Session Normalization Into Kiln-Owned Code

**Area:** `crates/kiln-train`, `crates/kiln-server` agent trace parser

**Problem:** The shared Python parser learned several Pi schema facts:
`toolResult` role normalization, `message` singular shape, tool args under
`input`. Kiln's Rust trace parser should know these too.

**Task:** Update kiln's agent trace parsing to accept observed Pi session
formats and normalize them to the canonical trajectory schema.

**Acceptance criteria:**

- Rust parser accepts role `tool` and `toolResult`.
- Rust parser accepts content blocks with `toolCall`, `toolResult`, `thinking`,
  and `text`.
- Tool-call args are read from `input` and rendered to canonical arguments.
- Unit tests use fixtures for both Pi 0.75.1 and Pi 0.75.3.
- Python and Rust parsers produce equivalent segment sequences on shared
  fixtures.

### 12. Make ECHO Activity Observable In Every Training Path

**Area:** `crates/kiln-train`

**Problem:** ECHO initially required adapter diffs and custom log greps to prove
it was firing. One path had debug logs, another did not, and tracing was not
initialized in the example.

**Task:** Standardize ECHO metrics and logs across checkpointed and
uncheckpointed paths.

**Acceptance criteria:**

- Every training run reports env token count and action token count.
- If ECHO is enabled and env tokens exist, logs show per-group env CE.
- If ECHO is enabled and env tokens are zero, training warns loudly.
- Receipt includes initial/final env CE when measurable.
- Checkpointed and uncheckpointed paths expose the same metric names.
- Tests cover ECHO on, ECHO off, and no-policy-loss.

## P0: Make Evaluation Trustworthy

### 13. Add Server Health Metrics For Eval Stability

**Area:** `crates/kiln-server`

**Problem:** `pi-code-search` found kiln-server state drift: requests slowed
from milliseconds to 80-120 seconds and evals falsely looked like adapter
regressions.

**Task:** Add `/v1/health` or expand existing health output with eval-relevant
server metrics.

**Fields should include:**

- uptime
- active adapter
- loaded adapter count
- request count
- recent p50/p95/p99 latency
- recent tokens/sec
- recent timeout/error count
- VRAM allocated/reserved if available
- prefix cache size
- CUDA graph/cache state summary if available
- last error summary

**Acceptance criteria:**

- Health response is machine-readable JSON.
- Health response is cheap enough to call between eval rollouts.
- Tests cover at least active adapter and request count.

### 14. Add A Slow-Request Watchdog

**Area:** `crates/kiln-server`

**Problem:** When server requests become unexpectedly slow, cap authors need a
diagnostic record tied to the request.

**Task:** Log structured diagnostics when a chat completion exceeds a threshold.

**Log should include:**

- request id
- adapter
- prompt tokens
- max output tokens
- generated tokens
- elapsed time
- batching engine state
- thinking mode
- CUDA graph state if available
- prefix cache hit/miss
- error or finish reason

**Acceptance criteria:**

- Threshold is configurable.
- Slow requests emit one structured warning.
- Logs do not include full prompt text by default.

### 15. Add Eval Mode To `kiln serve`

**Area:** `crates/kiln-server`

**Problem:** Agentic evals need deterministic, stable, debuggable serving more
than maximum throughput.

**Task:** Add `kiln serve --eval-mode`.

**Eval mode should:**

- expose active adapter in every response metadata block or header
- use deterministic defaults where possible
- disable thinking by default for Qwen tool-agent eval unless overridden
- bound or reset caches that grow across requests
- warn on adapter switching during an eval session
- optionally reset per-request transient state after each completion

**Acceptance criteria:**

- `--eval-mode` is documented.
- Health endpoint reports `eval_mode: true`.
- Direct completion tests work in eval mode.
- Repeated short completions do not show latency creep in a stress test.

### 16. Add Adapter Load/Unload Stress Test

**Area:** `crates/kiln-server` tests or integration tests

**Problem:** Repeated adapter load/unload and eval cycles appear to be a risk
factor for server drift.

**Task:** Add an integration test that repeatedly loads and unloads one or more
small test adapters while running short completions.

**Acceptance criteria:**

- Test runs at least 100 load/eval/unload cycles in a lightweight mode.
- It checks active adapter state every cycle.
- It tracks latency and fails on severe drift.
- It checks memory growth if allocator stats are available.

### 17. Make Qwen Thinking Mode A First-Class Config

**Area:** `crates/kiln-server`, config docs

**Problem:** Qwen thinking defaults caused empty `content`, long reasoning
traces, Pi timeouts, and 40x slowdowns. The control exists through
`chat_template_kwargs.enable_thinking`, but it was not obvious.

**Task:** Promote thinking-mode behavior into kiln config and docs.

**Acceptance criteria:**

- Add a server config/env var for default thinking mode.
- Health endpoint reports the default.
- Each request can still override via `chat_template_kwargs`.
- Response metadata indicates whether thinking was enabled.
- Docs explain behavior for Qwen3.5-4B and tool-agent usage.
- Tests cover content output with thinking on and off.

### 18. Surface Reasoning Content Separately And Safely

**Area:** `crates/kiln-server` OpenAI-compatible API

**Problem:** Some Qwen responses can put reasoning in `reasoning_content` and
leave `content` empty. Pi treated empty content as no response and re-prompted.

**Task:** Make response handling explicit for models/templates that separate
reasoning and final content.

**Acceptance criteria:**

- If reasoning text exists separately, response JSON exposes it in a documented
  field.
- If final content is empty, response metadata says why.
- There is an option to fold reasoning into content for compatibility, default
  off.
- Tests cover empty content with nonempty reasoning.

## P1: Make Training Diagnostics Strong Enough To Debug Science

### 19. Add Adapter-Effect Smoke Test After Training

**Area:** `crates/kiln-train`

**Problem:** `pi-compaction` produced byte-identical eval outputs even after
training. The trainer should detect likely no-op adapters before a full eval.

**Task:** After successful training, optionally run a small adapter-effect
smoke test.

**Smoke test should compare base vs adapter on fixed prompts using:**

- logit delta norm
- generated text difference
- adapter output length
- finite logits check

**Acceptance criteria:**

- Enabled by `--adapter-smoke-test`.
- Receipt records pass/fail and metrics.
- Warns if adapter appears to have no measurable effect.
- Warns if adapter produces empty outputs on canary prompt.

### 20. Report LoRA Delta And Gradient Norms

**Area:** `crates/kiln-train`

**Problem:** Several hypotheses depended on whether training moved weights
enough to affect BF16 inference. We need direct norms rather than guesswork.

**Task:** Record LoRA gradient norms and post-step delta norms by module.

**Acceptance criteria:**

- Receipt includes per-target-module delta norm summary.
- Receipt includes min/mean/max grad norm over training.
- Logs warn if all LoRA deltas are near zero.
- Logs warn if any delta is extreme relative to initialized scale.

### 21. Add Saturated-Reward Diagnostics

**Area:** `crates/kiln-train`

**Problem:** Saturated baselines made GRPO policy gradients harmful. The trainer
can detect low variance and high reward means in the training data.

**Task:** Add diagnostics for saturated or degenerate reward distributions.

**Acceptance criteria:**

- Dry run and training receipt report reward mean, stdev, min, max, and group
  variance histogram.
- Warn if most groups are all-pass or all-fail.
- Warn if reward mean is above a configurable saturation threshold and variance
  is low.
- Suggest `--no-policy-loss` or harder data in the warning text.

### 22. Officialize Strong-Signal Filtering

**Area:** `examples/cuda_grpo_ablation.rs`, `crates/kiln-train`

**Problem:** Strong-signal filtering was repeatedly useful, but each cap
implemented or tuned it outside kiln.

**Task:** Add first-class reward variance filtering.

**CLI options:**

- `--filter-var-min <float>`
- `--filter-var-max <float>` if useful
- `--min-groups <n>`
- `--on-empty-filter {fail,train-all,skip}`

**Acceptance criteria:**

- Filtered group count is printed and written to receipt.
- Exact kept/dropped group ids are written to a sidecar JSON.
- Empty-filter behavior is explicit, not accidental.
- Tests cover each `--on-empty-filter` mode.

### 23. Print Effective Training Config

**Area:** `examples/cuda_grpo_ablation.rs`

**Problem:** Several cap scripts assumed flags existed or misunderstood baked
defaults such as KL coeff, clip epsilon, or advantage mode.

**Task:** Add `--print-effective-config` and print config at train start by
default.

**Acceptance criteria:**

- Output includes all defaults and CLI overrides.
- JSON output mode is available.
- Help text explains that advantage formulation is selected by `--mode`.
- If `--no-echo` and `--echo-lambda` conflict, error text explains the correct
  single-argument pattern.

### 24. Add Standard Failure Reasons To Training Output

**Area:** `crates/kiln-train`

**Problem:** Iter logs used many ad hoc statuses. Kiln should classify common
training and data failures.

**Task:** Standardize failure reasons in receipts and errors.

**Suggested reasons:**

- `data_schema_error`
- `adapter_load_failed`
- `zero_groups`
- `zero_action_tokens`
- `zero_env_tokens`
- `nan_loss`
- `oom`
- `shape_mismatch`
- `unsafe_lora_scale`
- `base_adapter_missing`

**Acceptance criteria:**

- Receipts include `status` and `failure_reason`.
- CLI exits nonzero for all failure reasons.
- Error messages include enough context to fix the issue.

### 25. Add Long-Context GRPO Benchmark Suite

**Area:** `crates/kiln-train`, benchmarks

**Problem:** `pi-compaction` exposed long-context training as a major kiln
bottleneck. We need a stable benchmark to prevent regressions and guide fixes.

**Task:** Add synthetic GRPO benchmarks at 8K, 16K, 32K, and 64K sequence
lengths.

**Metrics:**

- tokenize time
- mask build time
- reference forward time
- policy forward time
- backward time
- optimizer time
- peak VRAM
- kernel launch count if available
- tokens/sec

**Acceptance criteria:**

- Benchmark can run a small CPU/dry version in normal dev environments.
- CUDA benchmark works with configurable model/task size.
- Results are emitted as JSON.
- Docs explain expected use for compaction-like workloads.

### 26. Add Long-Context Progress Logging

**Area:** `crates/kiln-train`

**Problem:** Long-context training could appear hung for many minutes with no
progress lines.

**Task:** Emit progress markers around major long-context phases.

**Markers:**

- data loaded
- tokenize start/end
- mask build start/end
- ref forward start/end
- policy forward start/end
- backward start/end per checkpoint segment
- optimizer start/end

**Acceptance criteria:**

- Logs appear before any phase expected to take more than 30 seconds.
- Logs include token count and segment/tile config.
- Receipt includes per-phase timings.

### 27. Debug Why Long-Context Adapters Can Be Byte-Identical

**Area:** `crates/kiln-train`, possibly `crates/kiln-server`

**Problem:** `pi-compaction` trained adapters that produced byte-identical eval
responses to base, even after long-context training became tractable.

**Task:** Add and use diagnostics to identify whether the no-op is caused by
adapter loading, zero gradients, KL dominance, clipping, masking, or inference
scale.

**Acceptance criteria:**

- A reproducible small long-context fixture exists.
- Training receipt shows nonzero or zero gradient/delta norms.
- Adapter verify shows whether logits change on the fixture.
- If logits change but text is identical, report sampling/determinism reason.
- If logits do not change, file a root-cause note in the doc or test.

### 28. Make Warning-Prefix Masking Testable

**Area:** `crates/kiln-train`

**Problem:** ECHO should not learn harness warning boilerplate. The Python
parser tracks `warning_prefix_len`, but kiln needs explicit tests and metrics.

**Task:** Add tests and receipt metrics for warning-prefix filtering.

**Acceptance criteria:**

- Test trajectory with `WARNINGS:` prefix masks warning tokens out of env CE.
- Receipt reports env tokens before and after warning filter.
- Config can disable warning filter for ablation.
- Logs warn if most env tokens are stripped as warnings.

## P1: Improve Server And API Observability

### 29. Add Per-Request Performance Counters To Chat Responses

**Area:** `crates/kiln-server`

**Problem:** Cap evals need to know whether an adapter is improving behavior or
just making sessions slower and chattier.

**Task:** Add optional performance metadata to chat-completion responses.

**Fields:**

- prompt tokens
- completion tokens
- TTFT
- total latency
- decode tokens/sec
- adapter used
- thinking mode
- finish reason

**Acceptance criteria:**

- Enabled by request flag or server config.
- Does not break OpenAI compatibility by default.
- Eval scripts can parse it from a stable JSON field.

### 30. Add Train-Time And Serve-Time Config Hashes

**Area:** `crates/kiln-server`, `crates/kiln-train`

**Problem:** Results changed after tokenizer/chat-template changes. Runs need
to record which template/config was actually used.

**Task:** Hash relevant model-side config and include it in responses and
receipts.

**Fields:**

- tokenizer config hash
- chat template hash
- model config hash
- kiln env/config hash

**Acceptance criteria:**

- Training receipt includes all hashes.
- Server health includes all hashes.
- Chat response metadata can include all hashes in debug mode.

### 31. Add Model-Specific Defaults Profile For Qwen3.5-4B

**Area:** `crates/kiln-server`, config

**Problem:** Kiln deliberately supports Qwen3.5-4B. It should encode safe
defaults learned from real usage, especially around thinking mode and agentic
tool calls.

**Task:** Add an explicit Qwen3.5-4B defaults profile.

**Acceptance criteria:**

- Profile documents default thinking behavior.
- Profile documents adapter directory expectations.
- Profile documents supported chat template behavior.
- Server logs which profile is active.
- Tests assert profile defaults.

### 32. Add `/v1/debug/model-state`

**Area:** `crates/kiln-server`

**Problem:** During eval, authors need a compact answer to "what model state am
I actually hitting?"

**Task:** Add a debug endpoint for local/trusted use.

**Response should include:**

- model path
- active adapter
- loaded adapters and hashes
- config hashes
- key env flags
- batching engine status
- thinking default
- cache summaries

**Acceptance criteria:**

- Endpoint is disabled or protected in non-debug mode if needed.
- Response contains no prompt/user data.
- Tests cover active adapter and config fields.

## P2: Add Higher-Level Kiln Capabilities For This Workflow

### 33. Add `kiln eval-adapter`

**Area:** kiln CLI

**Problem:** Multi-seed eval was repeatedly needed to avoid overclaiming.
Every cap implemented its own evaluation driver and summary format.

**Task:** Add a generic adapter eval command that runs a task JSONL through a
configurable request/scorer command.

**Command shape:**

```text
kiln eval-adapter --adapter NAME --tasks eval.tasks.jsonl --seeds 3 \
  --request-template request.json --scorer ./score_one.py
```

**Acceptance criteria:**

- Runs base and adapter in paired mode.
- Emits mean, stdev, zero count, and wall-clock stats.
- Writes `eval_summary.json`.
- Warns if sigma is comparable to lift.
- Records adapter and config hashes.

### 34. Add Direct HTTP Rollout Generation Utility

**Area:** kiln CLI or `crates/kiln-server` client tooling

**Problem:** Some useful capabilities do not need the Pi tool loop. Direct
single-turn generation would avoid Pi schema and orchestration overhead.

**Task:** Add a kiln-owned utility for task JSONL -> chat-completion responses
-> scored rollout JSONL.

**Acceptance criteria:**

- Sets adapter explicitly per request.
- Sets thinking mode explicitly.
- Records latency and token counts.
- Supports deterministic seeds if server supports them.
- Emits ScoredRollout-compatible JSONL for trainer input.

### 35. Add Adapter Canary And Quarantine Status

**Area:** `crates/kiln-train`, `crates/kiln-server`

**Problem:** Some trained adapters produced empty outputs or huge latency
regressions. Kiln should mark these as unsafe before users accidentally eval or
serve them.

**Task:** Add optional canary evaluation after training and expose status in
adapter registry.

**Canaries:**

- simple short completion
- simple tool-call-shaped prompt
- output length sanity
- finite logits
- latency sanity
- nonzero logit delta from base

**Acceptance criteria:**

- Failed canaries mark adapter status as `quarantined`.
- `/v1/adapters/load` refuses quarantined adapters unless override flag is set.
- Registry shows canary status and failure reason.

### 36. Add Standard Adapter Manifest And Restore Command

**Area:** kiln CLI, adapter format docs

**Problem:** Every cap wrote its own artifact backup manifest. Kiln should own
adapter provenance regardless of storage backend.

**Task:** Define `adapter_manifest.json` and implement `kiln adapter restore`.

**Manifest fields:**

- adapter name
- safetensors hash
- config hash
- receipt hash
- parent adapter
- model config hash
- kiln commit
- training data hash

**Acceptance criteria:**

- Training writes manifest.
- Server registry reads manifest if present.
- Restore command verifies hashes after copy.
- Manifest schema is documented.

### 37. Add Off-Policy Distillation Training Mode

**Area:** `crates/kiln-train`

**Problem:** Saturated GRPO caps need a better tool than policy gradient over
noisy rewards. Several negative results point toward teacher distillation or
OPD for high-baseline capabilities.

**Task:** Add or formalize an OPD/teacher-distillation path for agentic tasks.

**Acceptance criteria:**

- Accepts teacher response/logprob data in a documented JSONL schema.
- Supports KL or cross-entropy against teacher action tokens.
- Can combine with ECHO on env tokens.
- Receipt distinguishes OPD loss from GRPO loss.
- Includes a small synthetic test.

### 38. Add Reward-Saturation-Aware Training Recommendation

**Area:** `crates/kiln-train`

**Problem:** Cap authors wasted time applying GRPO to near-ceiling baselines.
Kiln can surface when the data distribution is unlikely to benefit.

**Task:** In dry-run and training startup, print a recommendation when reward
distribution is saturated.

**Acceptance criteria:**

- If mean reward > threshold and group variance < threshold, warning says
  policy-gradient may be harmful.
- Warning suggests harder tasks, stronger rubric gates, OPD, or
  `--no-policy-loss`.
- Thresholds are configurable.

### 39. Add End-To-End Agentic-GRPO Plumbing Test

**Area:** tests / examples

**Problem:** The terminal-bench-lite synthetic runs proved the stack manually.
That should become a maintained regression test.

**Task:** Add a small test that builds synthetic action/observation
trajectories, trains with ECHO on/off, and verifies a measurable adapter
difference.

**Acceptance criteria:**

- Test can run in a reduced CPU or tiny-model mode in CI if possible.
- CUDA version can run locally for full validation.
- Verifies ECHO on/off produce different LoRA deltas.
- Verifies `--no-policy-loss` still trains env CE.
- Verifies `--base-adapter` changes step-1 loss after loading.

### 40. Add Regression Tests For Lessons Already Learned

**Area:** tests across server/train

**Problem:** The agentic-GRPO work found repeatable bugs. They should become
tests so they do not recur.

**Task:** Add regression tests for the following cases:

- wrong nested adapter directory fails clearly
- missing adapter request field does not silently unload active adapter
- `--base-adapter` loads weights, not just lineage
- base-adapter rank mismatch fails before training
- alpha/rank > 2 fails or requires override
- Pi `toolResult` role normalizes to `tool`
- ECHO enabled with env tokens records nonzero env count
- no-policy-loss zeroes policy loss while preserving ECHO training
- Qwen thinking off produces normal content on a short prompt
- repeated load/eval/unload does not cause severe latency drift

**Acceptance criteria:**

- Each test links to the bug class in its test name or comment.
- Tests are included in the normal Rust test suite where feasible.
- CUDA-only tests are marked and documented separately.

## Suggested Execution Order

1. Adapter semantics and load validation: issues 1-7.
2. Training receipts, dry-run, and ECHO/mask visibility: issues 8-12.
3. Eval stability and Qwen thinking controls: issues 13-18.
4. Training diagnostics and long-context support: issues 19-28.
5. API observability and config hashes: issues 29-32.
6. Higher-level workflow helpers and regression tests: issues 33-40.

Completing just the first three groups would remove most of the ambiguity that
made the first agentic-GRPO runs hard to interpret. Completing all forty would
turn kiln from "can train agentic adapters" into "can run agentic learning
experiments with trustworthy receipts, safe adapter state, stable evals, and
actionable diagnostics."

