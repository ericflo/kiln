# Kiln Confidence Hardening Goal

**Status:** Temporary working document. Keep it current while executing the
goal, then replace it with a short permanent support/qualification document
when every required item is complete.

**Created:** 2026-07-09

**Starting commit:** `961facd6d9d1452e0336f3bb8d9731b2a74938fd`

## Goal

Make Kiln a small, explainable, empirically qualified single-GPU product whose
correctness, latency, training, and evaluation claims are backed by runtime
evidence on the machines that actually matter.

This goal is complete only when every required checkbox in this document is
checked, every acceptance gate has a compact checked-in receipt, the final
receipts cover one common source tree, and the public documentation says no
more than the evidence supports.

The priority order is:

1. Prevent corruption and unexplained stalls.
2. Correct training and evaluation semantics.
3. Establish repeatable local hardware qualification.
4. Measure and improve performance against honest external baselines.
5. Reduce configuration, scheduler, module, test, and artifact complexity.
6. Finish product/API integration and documentation.

Do not add unrelated features while this goal is active.

## Operating Model

GitHub Actions is not hardware evidence. Expensive backend builds and GPU jobs
must not run automatically on every push. Automatic CI should be a cheap safety
net for formatting, dependency policy, CPU tests, UI smoke tests, and other
fast platform-independent checks. Backend qualification happens locally on
real hardware.

Qualification order:

1. ROCm on the current Strix Halo machine (`gfx1151`).
2. Vulkan on the same Strix Halo machine.
3. CUDA on the RTX 4090 Laptop GPU with 16 GB VRAM.
4. CUDA on the desktop RTX 4090 with 24 GB VRAM.
5. Metal on the M1 MacBook Air.
6. Final cross-platform rerun against one common source-tree hash.

The native Kiln trainer is a bounded single-GPU online LoRA trainer. It is not
intended to replace distributed TRL/Transformers training. HF/TRL must become a
first-class interoperability route and the numerical oracle for the native
path.

## Execution Rules

- [x] Read this entire document before changing code.
- [ ] Update checkboxes and the evidence log in the same commit as the work
  they describe.
- [ ] Work in the order written unless a later item is required to unblock an
  earlier acceptance gate. Record any reordering in the evidence log.
- [ ] Make small, reviewable commits. A backend phase is expected to contain
  many commits and pushes, not one platform-sized commit. Do not accumulate a
  multi-day or multi-feature working tree.
- [ ] The goal runner has standing authorization to commit and push completed
  work directly to `origin/main` throughout this goal.
- [ ] Commit and push at every reasonable green checkpoint, including each
  coherent bug fix, regression test, instrumentation slice, qualification-tool
  addition, benchmark-harness improvement, configuration migration,
  documentation correction, and compact hardware receipt. Do not wait for the
  surrounding phase or machine to be complete.
- [ ] When hardware exposes a defect, normally push at least three checkpoints:
  the focused reproducer/test, the fix with focused validation, and the passing
  qualification receipt. Combine them only when they cannot be made independently
  correct and green.
- [ ] Commit and push before starting a long benchmark/soak, after recording its
  result, before a risky architectural change, before changing task area, and
  before handing work to another machine.
- [ ] Update relevant checkboxes and the evidence log at each checkpoint. Do not
  create artificial WIP commits that knowingly leave `main` broken when a
  coherent green checkpoint is available.
- [ ] Never amend or force-push a published commit. Before starting on another
  machine, fetch and fast-forward to `origin/main` and confirm a clean tree.
- [ ] Do not revert unrelated user changes. Stop and document a real conflict
  rather than using destructive Git commands.
- [ ] A compile-only check, mock, source-string assertion, skipped test, or
  external CI badge does not count as runtime backend evidence.
- [ ] Hardware tests must fail, not silently skip, when run in qualification
  mode and the requested device/backend is unavailable.
- [ ] Do not weaken a tolerance, delete a failing test, ignore an advisory, or
  relabel a feature experimental merely to make a gate pass. Scope can be
  narrowed only when the support matrix and user-facing docs are updated in
  the same commit.
- [ ] Keep raw logs, traces, and profiles out of Git. Check in compact receipts,
  summaries, schemas, and hashes only.
- [ ] Do not rewrite repository history as part of this goal. Any history purge
  requires a separate explicit decision because it disrupts every clone.

## What Counts As Evidence

Create one local qualification entry point and one compact receipt schema.
Prefer a Rust or structured Python implementation over shell text parsing.

Each receipt must contain at least:

- Receipt schema version.
- `source_tree_hash`, computed over source, manifests, lockfiles, shaders,
  kernels, scripts, and runtime configuration, excluding qualification receipts
  and other generated evidence.
- Git commit and whether the worktree was clean.
- Backend, device name, device architecture, VRAM/unified memory, OS, kernel,
  driver, runtime, compiler, and relevant library versions.
- Model ID/path plus base-weight shard hashes, tokenizer hash, template hash,
  and model-config hash.
- Exact command, typed effective runtime configuration, seed, workload ID, and
  workload content hash.
- Start/end times and duration.
- Per-test result, skip count, failure count, and tolerance used.
- TTFT, end-to-end latency, p50/p99/p99.9 inter-token latency, throughput,
  SLO-goodput, peak memory, and unexplained-outlier count where relevant.
- Hashes of raw local logs/profiles without checking those raw files into Git.
- Final verdict and any explicitly unsupported scenarios.

Receipt rules:

- [x] Add a schema and validator for qualification receipts.
- [x] Add a deterministic `source_tree_hash` command and tests proving receipt
  files do not change it while source changes do.
- [ ] Add a local qualification runner that writes raw logs under an ignored
  directory and emits a compact receipt.
- [ ] Add a receipt comparison command that rejects incompatible source trees,
  models, workloads, and environments rather than comparing unlike runs.
- [ ] Add a checked-in workload manifest for correctness, mixed-load serving,
  slow-consumer, cancellation, long-prefill, memory-pressure, and training
  cases. Do not check in generated model output or large traces.
- [ ] Add documentation for running, validating, and committing a receipt on a
  new machine.

Suggested locations:

- `scripts/qualification/` for runners and validators.
- `qualification/schema/` for schemas.
- `qualification/workloads/` for small deterministic workload manifests.
- `qualification/receipts/<backend>/<device>/` for compact receipts.
- `.qualification/` for ignored raw logs, traces, profiles, and temporary data.

## Phase 0: Stop Paying For Weak Signals

### 0.1 Automatic CI cost and scope

- [x] Inventory every automatic workflow, trigger, hosted-runner platform,
  average duration, and what unique defect it can actually catch.
- [x] Remove or change to `workflow_dispatch` automatic CUDA, ROCm, Vulkan,
  Metal, release-matrix, and performance jobs that consume substantial hosted
  minutes without executing on the target hardware.
- [x] Remove the broken scheduled performance workflow or convert it into a
  local qualification command. Do not retain a permanently red nightly badge.
- [x] Keep one inexpensive automatic path for formatting, dependency policy,
  CPU/unit tests, UI smoke, and schema/receipt validation.
- [x] Ensure release packaging can be invoked deliberately after local hardware
  qualification rather than running a full matrix on every ordinary push.
- [x] Record before/after automatic job counts and expected hosted minutes in a
  small checked-in summary.
- [x] Update contributor docs so nobody treats automatic CI as backend
  qualification.

**Acceptance:** An ordinary push starts only the agreed cheap workflows.
Backend and performance workflows require an explicit human invocation. The
local receipt validator runs in cheap CI.

### 0.2 Baseline this machine

- [x] Produce a Strix Halo ROCm environment receipt at the starting source tree.
- [x] Produce a Strix Halo Vulkan environment receipt at the starting source
  tree.
- [ ] Capture a short mixed-load ROCm baseline with all current defaults.
- [ ] Capture controlled A/B baselines with `KILN_KV_AUTOSCALE=0`, with
  `KILN_ROCM_GRAPHS=0`, and with both disabled.
- [ ] Attribute observed gaps to trace events where current instrumentation
  permits; mark the rest unexplained rather than guessing.

## Phase 1: Serving Safety And Predictable Latency

### 1.1 Stable serving profile

- [ ] Add a typed, documented stable serving profile and make it the default
  for supported ROCm operation.
- [ ] In the stable profile, prohibit physical KV resize, pool trim, live graph
  capture, and training GPU ownership changes while requests are active.
- [ ] Add a real off switch for memory-governor automatic reclaim. Changing the
  probe interval is not an off switch.
- [ ] Keep logical admission/eviction available without moving live allocation
  pointers.
- [ ] Require an explicit experimental or maintenance profile for dynamic
  physical memory operations and concurrent training.
- [ ] Expose the effective profile, every resolved setting, and its source
  (default, file, environment, request) in startup diagnostics and health.
- [ ] Make malformed configuration fatal with the variable/field name and
  invalid value. Do not fail open.

**Acceptance:** A stable-profile mixed-load trace contains no physical KV
resize, pool trim, live graph capture, training write lock, or unexplained
device-wide synchronization while a request is active.

### 1.2 Graph-safe, transactional KV resize

- [ ] Add a monotonically increasing KV-pool generation or stable allocation
  identity.
- [ ] Include pool generation/identity in graph replay validation for every
  backend that captures paged-KV pointers.
- [ ] Destroy/invalidate affected graphs before any old pool allocation is
  released.
- [ ] Permit physical resize only at an actor quiescence barrier with no live
  kernel and no active request referencing the pool.
- [ ] Make grow and shrink failure-atomic. Commit BlockManager state only after
  every allocation and copy succeeds.
- [ ] Add fault injection at every allocation, copy, synchronization, and layer
  commit boundary. Prove rollback leaves all layers and logical capacity
  consistent.
- [ ] Add runtime tests for eager and captured decode before/after grow and
  shrink, including a request active when resize is requested.
- [ ] Verify pool-generation mismatch refuses replay instead of launching.

**Acceptance:** No stale graph can replay after resize; injected failures leave
the cache usable; ROCm runtime tests exercise the real device and pass.

### 1.3 Memory governor and synchronization

- [ ] Do not call `hipDeviceSynchronize` from an uncoordinated periodic monitor.
- [ ] Query actual pool reserved/used/free bytes before considering trim.
- [ ] Report actual reclaimed bytes and suppress repeated zero-yield actions.
- [ ] Add hysteresis, cooldown, and exponential backoff to any remaining
  automatic pressure response.
- [ ] Attribute each synchronization, allocation, trim, and resize with reason,
  bytes, wait time, and duration.
- [ ] Add a pressure test where free memory stays below 10 percent without
  producing a two-second periodic inference stall.

### 1.4 Actor scheduling and backpressure

- [ ] Move client delivery/backpressure outside the compute actor. A slow or
  suspended client must not sleep all decode rows.
- [ ] Add deterministic slow-consumer and closed-socket tests.
- [ ] Implement token-budgeted/chunked prefill so a 16K prompt cannot monopolize
  the actor while other rows are decoding.
- [ ] Add fair admission tests mixing short decode, 1K prefill, and 16K prefill.
- [ ] Ensure adapter loading and training cannot acquire an actor-wide GPU lock
  in the stable profile.
- [ ] Define explicit maintenance/drain behavior for operations that require
  exclusive GPU ownership.

### 1.5 ROCm graph behavior

- [ ] Replace per-replay device-wide synchronization with stream/event ordering
  where ROCm correctness permits it.
- [ ] Pre-capture supported graph buckets during an explicit warmup phase, or
  keep graphs disabled in stable mode until this is reliable.
- [ ] Do not perform warm-forward plus failed-capture plus eager-forward on a
  live request without exposing that event and latency.
- [ ] Add eager-versus-graph logit and token parity across sequence buckets,
  block-table changes, prefix reuse, cancellation, and adapter boundaries.
- [ ] Treat graph fallback as a counted event with a bounded reason enum.

### 1.6 Latency observability

- [ ] Measure actor queue, admission, tokenization, prefill, decode, sampling,
  readback, client delivery, GPU-lock wait, graph capture/replay, synchronization,
  resize, trim, adapter, and training phases separately.
- [ ] Associate each emitted token with ready time, delivered time, and any
  blocking phase since the preceding token.
- [ ] Export bounded-cardinality counters/histograms for TTFT, ITL p50/p99/p99.9,
  unexplained gaps, and each stall reason.
- [ ] Show the same attribution in recent-request diagnostics without requiring
  a profiler.
- [ ] Ensure measurement overhead is benchmarked and bounded.

### 1.7 Strix Halo ROCm qualification

- [ ] Run correctness qualification with graphs disabled.
- [ ] Run correctness qualification with pre-captured graphs enabled.
- [ ] Run mixed prompt lengths and concurrency 1, 8, 16, 32, and 64 where the
  supported profile permits them.
- [ ] Exercise prefix reuse, cancellation, slow consumers, adapter load/unload,
  memory pressure, and maintenance-mode resize.
- [ ] Run a 30-minute development soak after each material serving change.
- [ ] Run a final 24-hour mixed-load soak for the ROCm phase.
- [ ] Require every ITL gap above `max(250 ms, 5 * rolling p50 ITL)` to have a
  bounded reason code. The final unexplained count must be zero.
- [ ] Require zero device faults, non-finite logits, token mismatches, leaked
  blocks, cache-generation mismatches, or unexpected process-memory growth.
- [ ] Check in the compact ROCm receipts and push them with this phase checked.

## Phase 2: Thinking Budgets And Product Contract

### 2.1 Shared budget model

- [ ] Move inherit/unlimited/limit semantics, source attribution, effective
  limits, and outcomes into one shared typed model used by API, CLI, eval,
  batch, desktop, and browser code.
- [ ] Remove duplicated `BudgetOverride`, `EvalBudgetOverride`, and CLI-only
  semantic implementations or reduce them to conversions around the shared
  type.
- [ ] Define one canonical schema/reference and generate or validate duplicated
  documentation from it.

### 2.2 Configuration and validation

- [ ] Make invalid thinking-budget environment values fail startup and
  configuration validation.
- [ ] Initialize diagnostics early enough that configuration failures are always
  visible.
- [ ] Document and prevalidate conflicts with `</think>` stop sequences and
  insufficient `max_tokens`.
- [ ] Use strict integer parsing in browser and desktop surfaces. Do not floor
  decimals or convert malformed values into omission.

### 2.3 Transport and provenance parity

- [ ] Preserve final thinking-budget metadata/outcomes when durable request logs
  reassemble SSE streams.
- [ ] Add streaming versus non-streaming durable-log parity tests.
- [ ] Include effective budget and source provenance in batch responses.
- [ ] Make compare mode use the same SSE accumulator and outcome rendering as
  normal chat.
- [ ] Add effective limits and outcome fields to recent requests and bounded
  metrics.

### 2.4 User ergonomics

- [ ] Give token and time budgets independent Inherit/Unlimited/Limit controls.
  Editing one dimension must not disable the inherited other dimension.
- [ ] Fetch and display effective server defaults and preview the effective pair
  before send.
- [ ] Make rollout thinking tri-state or reject inert budget flags when thinking
  is explicitly disabled.
- [ ] Use one shared default server port (`8420`) in server, CLI, desktop Rust,
  desktop JavaScript, tests, and docs.
- [ ] Make desktop settings parsing field-tolerant, versioned, atomic, backed up,
  and visibly erroneous instead of resetting the whole configuration.

**Acceptance:** One conformance suite runs the same budget matrix through chat,
SSE, batch, compare, eval, rollout CLI, request log, recent requests, and
desktop configuration with identical effective semantics and provenance.

## Phase 3: Training Correctness And HF/TRL Interoperability

### 3.1 Correct GRPO policy semantics

- [ ] Separate behavior/old-policy log-probabilities from frozen-reference
  log-probabilities in types, storage, loss functions, metrics, and receipts.
- [ ] Version the rollout schema with exact token IDs, per-token behavior
  log-probabilities, behavior model/adapter content hash, tokenizer/template
  hash, sampling configuration, seed, and generation backend.
- [ ] Reject an off-policy rollout that lacks the information needed for the
  configured importance correction. Do not silently substitute the KL
  reference.
- [ ] Keep KL-reference selection independent from old-policy selection and
  refresh cadence.
- [ ] Add analytic scalar/tensor fixtures where old policy and reference policy
  differ so the historical conflation cannot pass.
- [ ] Compare loss, ratios, clipping, KL, gradients, and one optimizer step
  against a pinned TRL/PyTorch oracle.

### 3.2 Exact resumable checkpoints

- [ ] Define a versioned training-checkpoint manifest.
- [ ] Save and restore adapter parameters, optimizer moments/momentum, scheduler,
  global step, epoch, data cursor/order, all RNG states, reference/EMA state,
  precision policy, and effective configuration.
- [ ] Write checkpoints atomically with an incomplete sentinel and checksum
  validation.
- [ ] Distinguish resumable checkpoints from exportable PEFT adapter snapshots
  in names, API, UI, and docs.
- [ ] Add crash/fault interruption tests.
- [ ] Prove uninterrupted versus stop/resume training produces the same loss
  sequence, optimizer state, and adapter hash inside the declared deterministic
  envelope.

### 3.3 Seeds and provenance

- [ ] Resolve and persist one effective seed at every public SFT, GRPO, OPD,
  distillation, and eval entry point.
- [ ] Record base-weight shard hashes rather than only model ID/config hash.
- [ ] Record backend, device, driver/runtime, build/source hash, tokenizer,
  template, precision, kernels, and effective environment/configuration.
- [ ] Replace claims of exact replay with either a real replay-to-output command
  or precisely scoped integrity-verification language.
- [ ] Add same-environment repeated-run adapter-hash tests.

### 3.4 SFT schema, data policy, and numerics

- [ ] Unify training messages with the core inference/eval message schema,
  including tool calls, names, tool-call IDs, and tool responses.
- [ ] Establish exact Qwen chat-template token and label goldens against HF for
  plain, thinking, tool-call, tool-response, and multi-turn examples.
- [ ] Define assistant-only masking explicitly, including whether role headers
  and terminators are supervised, and match the chosen HF/TRL contract.
- [ ] Use one explicit `fail` or `skip` invalid-row policy across inline and
  streamed ingestion. Receipts must contain stable kept/rejected row hashes.
- [ ] Add ingestion-equivalence tests proving the same corpus trains the same
  rows through every transport.
- [ ] Add microbatch/gradient-accumulation semantics, scheduler/warmup, gradient
  clipping, and a documented precision/optimizer-state contract if the native
  trainer continues to expose general SFT configuration.
- [ ] Otherwise, deliberately narrow the native API to a microtrainer profile
  and route general SFT to HF/TRL. Remove broader claims in the same commit.
- [ ] Compare low-gradient and ordinary N-step AdamW updates with PyTorch,
  including the chosen master-parameter and moment precision.

### 3.5 First-class HF/TRL route

- [ ] Export Kiln datasets, chat-template configuration, adapters, rollout
  provenance, and eval split manifests into documented HF/TRL-compatible forms.
- [ ] Import resulting PEFT adapters with full base/tokenizer/template hash
  validation.
- [ ] Provide pinned reference scripts for SFT and GRPO that reproduce the
  oracle fixtures.
- [ ] Document native Kiln training as bounded single-GPU online LoRA and
  HF/TRL as the path for distributed, broad-method, and highly configurable
  training.
- [ ] Add round-trip tests: Kiln export -> pinned HF/TRL step -> Kiln import ->
  inference/eval.

**Acceptance:** The oracle ladder covers tokenization/masks, scalar losses,
tiny-model logits/log-probs, one-step gradients/updates, 10-step trajectories,
and exact resume. Cross-backend agreement alone is insufficient.

## Phase 4: Evaluation Integrity

### 4.1 Correct multi-sample aggregation

- [ ] Add an explicit suite-level aggregation enum such as pass@k, mean@k, and
  majority@k with precise definitions.
- [ ] Reduce multiple completions to one independent per-example statistic
  before accuracy, confidence interval, tag/tool slice, regression, and
  promotion calculations.
- [ ] Add known-answer n/k fixtures and property tests.
- [ ] Version result schemas and migrate or clearly reject old ambiguous
  results.

### 4.2 Train/eval separation

- [ ] Content-address dataset rows and preserve group/session identity.
- [ ] Implement deterministic group-aware train/validation/holdout splits with
  a persisted split manifest.
- [ ] Reject post-training evaluation whose suite overlaps training rows unless
  explicitly labeled `train-set-eval`.
- [ ] Remove every claim that the same training JSONL automatically becomes a
  held-out suite.
- [ ] Add contamination tests for exact, normalized, and grouped duplicates.

### 4.3 Reproducible evals and promotion

- [ ] Materialize the eval seed before generation and derive deterministic
  per-example/per-completion seeds.
- [ ] Persist raw completion references and model, adapter, tokenizer, template,
  backend, binary/source, generation, and scorer identities.
- [ ] Add a replay command that regenerates byte-identical seeded completions
  inside the declared environment.
- [ ] Gate promotion on per-example paired statistics and a lower confidence
  bound or exact/paired test, with minimum sample size and multi-suite policy.
- [ ] Do not promote on point accuracy alone.
- [ ] Replace mock-labeled-as-e2e coverage with a small real-model runtime path.

## Phase 5: Performance Program On Strix Halo

### 5.1 One honest serving benchmark

- [ ] Build one benchmark driver used unchanged for Kiln and vLLM where vLLM
  supports the hardware/model.
- [ ] Pin model weights, tokenizer/template, input/output lengths, prompts,
  sampling, seeds, request arrival pattern, warmup, concurrency, runtime
  versions, and memory limits.
- [ ] Measure TTFT, end-to-end latency, ITL p50/p99/p99.9, request throughput,
  token throughput, SLO-goodput, peak memory, errors, and output parity.
- [ ] Include concurrency 1, 8, 16, 32, 64, and 128 where memory permits.
- [ ] Separate greedy, API-default sampled, long-prefill, prefix-hit, and mixed
  workloads.
- [ ] Refuse zero-token/error responses in throughput accounting.
- [ ] Emit comparable compact receipts for both engines.

### 5.2 Optimize only measured bottlenecks

- [ ] Use phase attribution to rank total latency/throughput cost before each
  optimization.
- [ ] Implement fused ROCm batched LM head, penalties, top-k/top-p, and sampling
  with one token readback if profiling confirms the current serialization.
- [ ] Autotune supported decode width after fused sampling and graph correctness,
  rather than raising the cap blindly.
- [ ] Add multi-batch graph capture only after eager parity and pool-generation
  safety pass.
- [ ] Re-run correctness, 30-minute soak, and benchmark after each optimization.
- [ ] Record rejected experiments as short summaries, not raw logs.

### 5.3 Honest performance positioning

- [ ] Define the concurrency/latency region Kiln intentionally serves.
- [ ] Make a high-concurrency competitive claim only where Kiln reaches the
  declared target, initially `>= 0.90x` vLLM SLO-goodput with no correctness or
  tail-latency regression on the pinned workload.
- [ ] Where it does not, document vLLM as the recommended high-concurrency
  backend instead of hiding the result.
- [ ] Publish current receipts and remove stale or incomparable benchmark claims.

## Phase 6: Vulkan Qualification On Strix Halo

- [ ] Run the same tokenizer, logit, sampling, cache, cancellation, and eval
  correctness corpus under Vulkan.
- [ ] Run SFT and GRPO oracle fixtures supported by the declared Vulkan scope.
- [ ] Run the same short/mixed/long prompt serving workload and phase metrics.
- [ ] Run a 30-minute development soak after each Vulkan fix.
- [ ] Run a final 8-hour Vulkan mixed-load soak.
- [ ] Require zero silent hardware skips and zero unexplained ITL outliers by the
  ROCm attribution rule.
- [ ] Compare Vulkan outputs to CPU/HF oracles, not merely to another Kiln GPU
  backend.
- [ ] Check in compact Vulkan receipts, update this document, commit, and push.

## Phase 7: CUDA And Metal Handoffs

For each machine, first check out `origin/main`, validate receipts/schema, and
run the environment capture. Commit and push repeatedly while working on that
machine using the checkpoint rules above. Each CUDA or Metal defect, test,
fix, measurement milestone, and receipt should land as soon as it is coherent
and green; the final push on a machine is only the handoff marker, not the sole
platform commit.

### 7.1 RTX 4090 Laptop GPU, 16 GB

- [ ] Capture environment and memory-capability receipt.
- [ ] Run the full CUDA correctness/oracle subset that fits the device.
- [ ] Exercise low-memory admission and explicit failure behavior.
- [ ] Run serving performance at every concurrency that fits without changing
  the workload semantics.
- [ ] Run an 8-hour mixed-load soak.
- [ ] Check in receipts, update this document, commit, and push.

### 7.2 Desktop RTX 4090, 24 GB

- [ ] Capture environment receipt.
- [ ] Run full CUDA correctness, graph, resize, training, and eval qualification.
- [ ] Run the full Kiln/vLLM performance matrix.
- [ ] Run a 24-hour mixed-load soak.
- [ ] Check in receipts, update this document, commit, and push.

### 7.3 M1 MacBook Air, Metal

- [ ] Capture environment and unified-memory receipt.
- [ ] Run the full Metal correctness/oracle subset supported by the machine.
- [ ] Exercise memory pressure and OS coexistence without unbounded allocation
  or unexplained stalls.
- [ ] Run the supported serving performance matrix.
- [ ] Run an 8-hour mixed-load soak.
- [ ] Check in receipts, update this document, commit, and push.

## Phase 8: Reduce The Surface Area

### 8.1 Typed configuration

- [ ] Inventory every direct `KILN_*` read and classify it as public stable,
  experimental/debug, build-time, or test-only.
- [ ] Centralize public runtime configuration into typed structures with one
  parser, validation, defaults, source attribution, and effective-config dump.
- [ ] Prohibit direct runtime environment reads in model kernels, forwarding,
  scheduling, training, eval, UI, and request handlers via a repository check.
- [ ] Put experimental/debug knobs behind one explicit namespace and profile.
- [ ] Delete dead, duplicate, contradictory, and undocumented knobs.
- [ ] Ensure tests use scoped configuration rather than process-global env
  mutation wherever possible; use one global serialization helper where env is
  unavoidable.

### 8.2 One scheduling model

- [ ] Map ownership and production/test use of batching engine, legacy decode
  batcher, scheduler crate, and mock scheduler.
- [ ] Select one production scheduling/admission abstraction.
- [ ] Port tests to the production abstraction or a faithful injected backend.
- [ ] Remove the idle legacy thread and duplicated policy paths.
- [ ] Add state-machine/property tests for admission, cancellation, capacity,
  prefix reuse, fairness, backpressure, and shutdown.

### 8.3 Split oversized modules by behavior

- [ ] Split `forward.rs` into owned attention, linear-attention, FFN, norm,
  cache, backend-dispatch, and training-facing modules without duplicating
  policy.
- [ ] Split `completions.rs` into schema, validation, preparation, streaming,
  finalization, batch, and shared lifecycle modules.
- [ ] Split `trainer.rs` into data/tokenization, SFT, GRPO, checkpointing,
  optimizers, provenance, and oracle-test modules.
- [ ] Split the browser application into maintainable modules using the existing
  build/embedding approach.
- [ ] Establish a reviewed production-file size budget. Exceptions require a
  specific rationale, not generated-code-style growth.

### 8.4 Replace verification theater

- [ ] Inventory source-string and ad hoc Rust-source parsing tests.
- [ ] Replace load-bearing source-string assertions with compile-time types,
  runtime tests, property tests, or structured metadata validation.
- [ ] Ensure hardware qualification reports skipped tests as failures.
- [ ] Rename mock tests so they cannot be mistaken for model/backend e2e tests.
- [ ] Delete redundant tests that add maintenance cost without a distinct
  invariant.

### 8.5 Artifact and documentation policy

- [ ] Stop tracking raw `.log`, SSE, profiler, trace, and large CSV artifacts.
- [ ] Add ignore rules and size checks that prevent recurrence.
- [ ] Keep compact summaries, manifests, hashes, and reproduction commands.
- [ ] Move or remove existing raw audit artifacts from the current tree while
  preserving the useful conclusions in compact documents.
- [ ] Do not rewrite Git history without a separate explicit approval.
- [ ] Generate API/reference documentation from canonical schemas where
  semantics are currently copied across README, QUICKSTART, site HTML, and
  guides.

## Phase 9: Final Common-Tree Qualification

Receipt commits themselves do not change `source_tree_hash`. Any source,
manifest, lockfile, kernel, shader, qualification runner, or runtime-config
change does. If a source-changing fix lands on any later machine, all earlier
final receipts for affected behavior are invalid and must be rerun.

- [ ] Freeze a release-candidate `source_tree_hash` after all source changes.
- [ ] Re-run final ROCm qualification and 24-hour soak on Strix Halo.
- [ ] Re-run final Vulkan qualification and 8-hour soak on Strix Halo.
- [ ] Re-run final CUDA qualification on the 16 GB 4090 laptop.
- [ ] Re-run final CUDA qualification and 24-hour soak on the 24 GB desktop
  4090.
- [ ] Re-run final Metal qualification and 8-hour soak on the M1 MacBook Air.
- [ ] Validate that every final receipt names the same `source_tree_hash` and
  compatible model/workload versions.
- [ ] Run the complete cheap CPU/static CI path locally.
- [ ] Confirm an ordinary push invokes only the intended inexpensive automatic
  workflows.
- [ ] Update the support matrix with exact qualified device/runtime classes and
  explicitly experimental or unsupported combinations.
- [ ] Update benchmark, training, eval, thinking-budget, operations, and
  troubleshooting docs from the final evidence.
- [ ] Remove stale claims and stale benchmark tables.
- [ ] Confirm there are no unchecked required items in this document.
- [ ] Replace this temporary plan with a concise permanent qualification and
  support policy, preserving a link to the final receipts.
- [ ] Commit and push the final documentation/receipt state to `origin/main`.

## Final Release Gates

All gates are required:

- [ ] Zero known stale-pointer, use-after-free, partial-commit, or cache-state
  corruption path in supported serving profiles.
- [ ] Zero unexplained ITL outliers in final hardware soaks.
- [ ] Zero silent hardware-test skips.
- [ ] Eager/graph and backend/oracle correctness within declared tolerances.
- [ ] GRPO old-policy and KL-reference semantics match the pinned oracle.
- [ ] Exact checkpoint/resume equivalence inside the declared envelope.
- [ ] Correct multi-sample eval aggregation and enforced train/eval separation.
- [ ] Thinking-budget semantics and provenance are transport/UI consistent.
- [ ] Performance claims are backed by comparable receipts; vLLM is recommended
  wherever it remains materially better.
- [ ] Native training scope and HF/TRL interoperability are explicit and tested.
- [ ] Automatic GitHub-hosted work is inexpensive and is not represented as
  hardware qualification.
- [ ] Final receipts cover ROCm, Vulkan, both CUDA machines, and Metal on one
  common source tree.
- [ ] Public docs, defaults, health output, and support matrix agree.
- [ ] Worktree and `origin/main` agree at the final pushed commit.

## Evidence Log

Add one row per completed slice. Keep notes short and link to compact receipts
or focused documents. Never paste raw logs here.

| Date | Phase | Source tree | Commit | Backend/device | Evidence | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-09 | Plan created | pending | pending | n/a | this document | pending | Audit converted into executable goal |
| 2026-07-09 | Receipt foundation: source identity | `a76a5e1b664b` | this commit | n/a | `scripts/qualification/source_tree_hash.py` | passed | 5 unit tests; receipts and docs excluded from identity |
| 2026-07-09 | Source identity scope correction | `sha256:ebdb799360a7` | this commit | n/a | source identity unit suite | passed | 7 tests; excludes profiling artifacts and desktop prose; uses repository hash convention |
| 2026-07-09 | Source identity index hardening | `sha256:d56d06727edd` | this commit | portable | source identity unit suite | passed | Rejects unresolved index stages; hashes symlink targets and gitlink object IDs without dereferencing |
| 2026-07-09 | Receipt schema and strict validator | `sha256:74efdd391d91` | this commit | portable | `qualification/schema/receipt-v1.schema.json` | passed | 13 receipt tests plus 7 source identity tests |
| 2026-07-09 | Atomic ROCm/Vulkan environment collector | `sha256:c4f18b397716` | this commit | Strix Halo target | `scripts/qualification/environment.py` | passed | 25 portable tests; real receipts captured after clean collector commit |
| 2026-07-09 | Initial ROCm environment | `sha256:c4f18b397716` | `33f27c5f4dc6` | ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260709t202926z-rocm-strix-halo-environment-v1.json` | passed | ROCm 7.2.4; HIP 7.2.53211; 96 GiB unified VRAM |
| 2026-07-09 | Initial Vulkan environment | `sha256:c4f18b397716` | `1538505b5129` | Vulkan/Strix Halo | `qualification/receipts/vulkan/strix-halo/20260709t203013z-vulkan-strix-halo-environment-v1.json` | passed | Vulkan 1.4.348; RADV Mesa 26.1.3; 96 GiB unified VRAM |
| 2026-07-09 | Manual-only backend compile jobs | `sha256:c4f18b397716` | this commit | GitHub Actions | `.github/workflows/ci.yml` | passed | Metal/Vulkan/CUDA/ROCm changed to explicit dispatch; cancels superseded runs |
| 2026-07-09 | Retire broken automatic perf workflow | `sha256:c4f18b397716` | this commit | GitHub Actions | `.github/workflows/perf-regression-nightly.yml` | passed | Removed nightly/PR triggers; legacy fixture and A6000 paths are explicit manual dispatch only |
| 2026-07-09 | Manual-only release packaging | `sha256:c4f18b397716` | this commit | GitHub Actions | desktop/server/Docker/RunPod workflows | passed | Removed automatic push, tag, PR, and schedule triggers; explicit dispatch remains |
| 2026-07-09 | Consolidated cheap automatic checks | `sha256:c4f18b397716` | this commit | Linux CPU | `.github/workflows/ci.yml` | passed | Folded leaf and dependency-tree guards into default job; added portable receipt contract validation |
| 2026-07-09 | CI cost inventory and operator policy | `sha256:c4f18b397716` | this commit | GitHub Actions/local hardware | `docs/ci-policy.md` | passed | Recorded all prior automatic workflows, measured cost, three-job Rust target, manual dispatch commands, and local evidence ownership |
| 2026-07-09 | Deduplicated release drift check | `sha256:d56d06727edd` | this commit | Linux CPU | `.github/workflows/release-version-drift.yml` | passed | Release workflow now owns version/CLI/link drift only; Pages and UI workflows retain their browser smoke coverage |
| 2026-07-09 | Honest legacy latency status | `sha256:d56d06727edd` | this commit | Metal/Vulkan fixtures | `docs/backend-latency-fixtures.json` | passed | Stale source-bound Metal and Vulkan artifacts are pending rerun; the legacy manifest no longer claims covered hardware evidence |

## Known Starting Defects

These are anchors, not an exhaustive substitute for the checklist:

- ROCm pool trim can synchronize the device every two seconds under pressure:
  `crates/kiln-memory/src/governor.rs`, `crates/kiln-server/src/state.rs`, and
  `crates/kiln-hip/src/lib.rs`.
- Default KV autoscaling can resize/copy live cache capacity between decode
  steps: `crates/kiln-server/src/kv_autoscaler.rs`,
  `crates/kiln-server/src/batching_engine.rs`, and
  `crates/kiln-model/src/paged_kv_cache_kt.rs`.
- ROCm captured graphs omit KV pool generation/identity and are not invalidated
  by physical resize: `crates/kiln-model/src/rocm_graph.rs` and
  `crates/kiln-model/src/generate.rs`.
- A full response channel can sleep the compute actor for up to two seconds:
  `crates/kiln-server/src/batching_engine.rs`.
- GRPO uses one reference tensor for both importance ratio and KL while rollout
  provenance lacks behavior log-probabilities: `crates/kiln-train/src/lib.rs`,
  `crates/kiln-train/src/trajectory.rs`, and
  `crates/kiln-train/src/trainer.rs`.
- Eval `n > 1` aggregates completions rather than examples, and synthesis can
  label training rows as held out: `crates/kiln-eval/src/suite.rs`,
  `crates/kiln-eval/src/result.rs`, `crates/kiln-eval/src/synthesis.rs`, and
  `docs/site/evals.html`.
- Training checkpoints save adapter weights without complete optimizer/cursor/
  RNG/reference state: `crates/kiln-train/src/trainer.rs` and
  `crates/kiln-train/src/opd.rs`.
- Thinking-budget configuration, streaming logs, batch/compare surfaces, recent
  requests, and desktop semantics are not yet one conformance-tested contract:
  `crates/kiln-server/src/config.rs`,
  `crates/kiln-server/src/request_log.rs`,
  `crates/kiln-server/src/api/completions.rs`,
  `crates/kiln-server/src/ui/app.js`, and `desktop/`.
