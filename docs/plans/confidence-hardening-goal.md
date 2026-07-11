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
- [x] Update checkboxes and the evidence log in the same commit as the work
  they describe.
- [x] Work in the order written unless a later item is required to unblock an
  earlier acceptance gate. Record any reordering in the evidence log.
- [x] Make small, reviewable commits. A backend phase is expected to contain
  many commits and pushes, not one platform-sized commit. Do not accumulate a
  multi-day or multi-feature working tree.
- [x] The goal runner has standing authorization to commit and push completed
  work directly to `origin/main` throughout this goal.
- [x] Commit and push at every reasonable green checkpoint, including each
  coherent bug fix, regression test, instrumentation slice, qualification-tool
  addition, benchmark-harness improvement, configuration migration,
  documentation correction, and compact hardware receipt. Do not wait for the
  surrounding phase or machine to be complete.
- [x] When hardware exposes a defect, normally push at least three checkpoints:
  the focused reproducer/test, the fix with focused validation, and the passing
  qualification receipt. Combine them only when they cannot be made independently
  correct and green.
- [x] Commit and push before starting a long benchmark/soak, after recording its
  result, before a risky architectural change, before changing task area, and
  before handing work to another machine.
- [x] Update relevant checkboxes and the evidence log at each checkpoint. Do not
  create artificial WIP commits that knowingly leave `main` broken when a
  coherent green checkpoint is available.
- [x] Never amend or force-push a published commit. Before starting on another
  machine, fetch and fast-forward to `origin/main` and confirm a clean tree.
- [x] Do not revert unrelated user changes. Stop and document a real conflict
  rather than using destructive Git commands.
- [x] A compile-only check, mock, source-string assertion, skipped test, or
  external CI badge does not count as runtime backend evidence.
- [x] Hardware tests must fail, not silently skip, when run in qualification
  mode and the requested device/backend is unavailable.
- [x] Do not weaken a tolerance, delete a failing test, ignore an advisory, or
  relabel a feature experimental merely to make a gate pass. Scope can be
  narrowed only when the support matrix and user-facing docs are updated in
  the same commit.
- [x] Keep raw logs, traces, and profiles out of Git. Check in compact receipts,
  summaries, schemas, and hashes only.
- [x] Do not rewrite repository history as part of this goal. Any history purge
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
- [x] Add a local qualification runner that writes raw logs under an ignored
  directory and emits a compact receipt.
- [x] Add a receipt comparison command that rejects incompatible source trees,
  models, workloads, and environments rather than comparing unlike runs.
- [ ] Add a checked-in workload manifest for correctness, mixed-load serving,
  slow-consumer, cancellation, long-prefill, memory-pressure, and training
  cases. Do not check in generated model output or large traces.
- [x] Add documentation for running, validating, and committing a receipt on a
  new machine.

Suggested locations:

- `scripts/qualification/` for runners and validators.
- `qualification/schema/` for schemas.
- `qualification/workloads/` for small deterministic workload manifests.
- `qualification/receipts/<backend>/<device>/` for compact receipts.
- `benchmarks/receipts/<backend>/<device>/` for detailed serving-benchmark
  receipts validated by their owning driver.
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
- [x] Attribute observed gaps to trace events where current instrumentation
  permits; mark the rest unexplained rather than guessing.

## Phase 1: Serving Safety And Predictable Latency

### 1.1 Stable serving profile

- [x] Add a typed, documented stable serving profile and make it the default
  for supported ROCm operation.
- [x] In the stable profile, prohibit physical KV resize, pool trim, live graph
  capture, and training GPU ownership changes while requests are active.
- [x] Add a real off switch for memory-governor automatic reclaim. Changing the
  probe interval is not an off switch.
- [x] Keep logical admission/eviction available without moving live allocation
  pointers.
- [x] Require an explicit experimental or maintenance profile for dynamic
  physical memory operations and concurrent training.
- [x] Expose the effective profile, every resolved setting, and its source
  (default, file, environment, request) in startup diagnostics and health.
- [ ] Make malformed configuration fatal with the variable/field name and
  invalid value. Do not fail open.
  - Progress: `server.deterministic` / `KILN_DETERMINISTIC` and
    `server.max_decode_batch` / `KILN_MAX_DECODE_BATCH` are now typed, strictly
    validated before tensor initialization, source-attributed, and resolved
    without actor-side environment reads. The repo-wide gate remains open for
    the other permissive legacy overrides inventoried in Phase 8.1.

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
- [x] Attribute backend external-yield synchronization with a bounded boundary
  label, call/failure/slow counts, and total/max duration in health, debug
  state, Prometheus, and the 100 ms slow-sync log.
- [ ] Add a pressure test where free memory stays below 10 percent without
  producing a two-second periodic inference stall.

### 1.4 Actor scheduling and backpressure

- [x] Move client delivery/backpressure outside the compute actor. A slow or
  suspended client must not sleep all decode rows.
- [x] Add deterministic slow-consumer and closed-socket tests.
- [x] Prevent exact non-block-aligned prefix-cache hits from sharing a mutable
  final KV block with decode; prove cached pages remain immutable across reuse.
- [x] Make prefix-cache lookup visibility and accounting failure-atomic: pin
  the source entry before snapshot copies, leave hit counters and LRU unchanged
  on failure, settle the provisional lease while the GPU read permit is held,
  and quarantine rather than recycle partial snapshot destinations.
- [x] Rank exact prefix-cache hits by next-token compatibility before pinning an
  entry, while allowing a shorter compatible strict-prefix fallback.
- [x] Make prefix-cache registration capacity-correct and failure-atomic across
  overlapping blocks, LRU eviction, pinned entries, and invalid block tables.
- [x] Hold each prefix-cache hit lease until decode cleanup and finish-time
  registrations are complete.
- [x] Make eviction, clear, and adapter purge honor active prefix-cache leases.
- [x] Fence late hit and miss completion with global and cache-local per-adapter
  generations so clear or purge cannot resurrect invalidated registrations.
- [x] Synchronize paged streaming, direct paged/interleaved, and batching-actor
  decode at their external-yield boundaries before publishing progress or
  recycling KV ownership; on these paths, latch an observable,
  restart-required quarantine when completion cannot be proven.
- [x] Route the legacy mutable-`BlockManager` paged streaming entry point
  through one settlement-owning reservation. Every post-allocation error,
  receiver-drop, and success exit must settle before recycling pages and retain
  GPU-owned state when settlement cannot be proven.
- [x] Make successful prefix-hit snapshots opaque until an explicit backend
  settlement, and make shared KV reservations retain pages by default unless a
  settlement boundary explicitly releases them.
- [x] Make paged stream workers own cleanup through the terminal event: run the
  sole finalizer before `Done` or `Error`, and retain GPU state plus the movable
  read permit when decode completion cannot be proven.
- [x] Carry a movable GPU read owner from threaded prefill through decode and
  cleanup, and use the same Tokio writer exclusion for portable and CUDA-native
  SFT optimizer steps.
- [x] Emit structured `generation_error` SSE payloads before `[DONE]` when a
  threaded prefill, its blocking task, or an explicit decode worker fails.
- [x] Prevent probe-only streaming completion from failing or overwriting a
  concurrently claimed deterministic-cache owner.
- [x] Make incremental detokenization and unexpected worker-channel termination
  fallible end to end; never turn tokenizer decode failure or a missing
  terminal event into empty text followed by a normal-looking `[DONE]`.
- [x] Propagate cooperative cancellation and the request deadline through
  direct threaded prefill, token delivery, and terminal delivery so timeout and
  disconnect stop the model worker rather than detaching a blocking receiver or
  waiting indefinitely on a full SSE queue.
- [x] Make prompt-logprob token rendering fallible instead of converting a
  tokenizer decode failure into an empty display token.
- [x] Make prompt-logprob numerical rendering fail closed: require the exact
  model vocabulary width, reject non-finite values before JSON serialization,
  and return exactly the requested distinct top-ranked candidates.
- [x] Match vLLM prompt-logprob protocol semantics by including the observed
  prompt token (K or K+1 entries), reporting its full-vocabulary rank, and
  rendering byte-fallback tokens with prompt context while preserving
  shift-stable F32 log-probability precision. Enforce the served-model context
  window and settle or quarantine every scorer-owned accelerator lifetime.
- [x] Make remote-teacher prompt-logprob parsing reject malformed or
  out-of-vocabulary entries instead of dropping them, and validate the
  observed-token/rank contract plus logits-row/target-position alignment before
  caching numeric results.
- [x] Establish an authoritative remote-teacher tokenizer identity handshake
  plus base-model/adapter content revision, then bind accepted/cached logits to
  that identity rather than trusting only the locally configured tokenizer hash.
- [x] Extend the backend quarantine gate to every inference worker, adapter
  mutation, prewarm, training admission, and queued training transition. A
  quarantined backend must reject rather than wait forever on its retained GPU
  read owner.
- [x] Publish one authoritative loaded-adapter tuple containing name and content
  revision atomically with the weight flip. Bind queued requests and all
  deterministic-cache owners to that revision so late completion cannot
  resurrect results invalidated by purge.
- [x] Serialize adapter upload, delete, training rewrite, gate demotion, and
  loaded-weight transitions under the same revision/barrier contract. Reject
  deletion or rewrite of physically loaded content unless the barrier swaps it
  away or reloads it first.
- [x] Implement token-budgeted/chunked prefill so a 16K prompt cannot monopolize
  the actor while other rows are decoding.
- [x] Add fair admission tests mixing short decode, 1K prefill, and 16K prefill.
- [x] Ensure adapter loading and training cannot acquire an actor-wide GPU lock
  in the stable profile.
- [x] Define explicit maintenance/drain behavior for operations that require
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

- [x] Move inherit/unlimited/limit semantics, source attribution, effective
  limits, and outcomes into one shared typed model used by API, CLI, eval,
  batch, desktop, and browser code.
- [x] Remove duplicated `BudgetOverride`, `EvalBudgetOverride`, and CLI-only
  semantic implementations or reduce them to conversions around the shared
  type.
- [x] Define one canonical schema/reference and generate or validate duplicated
  documentation from it.

### 2.2 Configuration and validation

- [x] Make invalid thinking-budget environment values fail startup and
  configuration validation.
- [x] Initialize diagnostics early enough that configuration failures are always
  visible.
- [x] Document and prevalidate conflicts with `</think>` stop sequences and
  insufficient `max_tokens`.
- [x] Use strict integer parsing in browser and desktop surfaces. Do not floor
  decimals or convert malformed values into omission.

### 2.3 Transport and provenance parity

- [x] Preserve final thinking-budget metadata/outcomes when durable request logs
  reassemble SSE streams.
- [x] Add streaming versus non-streaming durable-log parity tests.
- [x] Include effective budget and source provenance in batch responses.
- [x] Make compare mode use the same SSE accumulator and outcome rendering as
  normal chat.
- [x] Add effective limits and outcome fields to recent requests and bounded
  metrics.

### 2.4 User ergonomics

- [x] Give token and time budgets independent Inherit/Unlimited/Limit controls.
  Editing one dimension must not disable the inherited other dimension.
- [x] Fetch and display effective server defaults and preview the effective pair
  before send.
- [x] Make rollout thinking tri-state or reject inert budget flags when thinking
  is explicitly disabled.
- [x] Use one shared default server port (`8420`) in server, CLI, desktop Rust,
  desktop JavaScript, tests, and docs.
- [x] Make desktop settings parsing field-tolerant, versioned, atomic, backed up,
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
- [x] Add analytic scalar/tensor fixtures where old policy and reference policy
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

- [x] Build one benchmark driver used unchanged for Kiln and vLLM where vLLM
  supports the hardware/model.
- [ ] Pin model weights, tokenizer/template, input/output lengths, prompts,
  sampling, seeds, request arrival pattern, warmup, concurrency, runtime
  versions, and memory limits.
- [ ] Measure TTFT, end-to-end latency, ITL p50/p99/p99.9, request throughput,
  token throughput, SLO-goodput, peak memory, errors, and output parity.
- [ ] Include concurrency 1, 8, 16, 32, 64, and 128 where memory permits.
- [ ] Separate greedy, API-default sampled, long-prefill, prefix-hit, and mixed
  workloads.
- [x] Refuse zero-token/error responses in throughput accounting.
- [ ] Emit comparable compact receipts for both engines.

### 5.2 Optimize only measured bottlenecks

- [ ] Use phase attribution to rank total latency/throughput cost before each
  optimization.
- [ ] Implement fused ROCm batched LM head, penalties, top-k/top-p, and sampling
  with one token readback if profiling confirms the current serialization.
- [ ] Replace prompt-logprob full-vocabulary host readback with a device-side
  selected-logprob/rank kernel so transfer and host work scale with O(TK), while
  retaining exact finite-value validation and the bounded projection fallback.
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
| 2026-07-09 | Capability report truth alignment | `sha256:cda13f3f84e5` | this commit | Linux CPU | `docs/backend-capability-report.{json,md}` and `backend_capability_contract` | passed | Generated report exposes fixture-required Phase 8 and pending hardware blockers; generator checks and all 51 focused contract tests pass |
| 2026-07-09 | Exact model content fingerprint | `sha256:85e21e751643` | this commit | portable | `scripts/qualification/model_fingerprint.py` | passed | Loader-aligned shards and serving-template fallback; stable hashes; traversal, symlink, reference, duplicate, and mutation tests |
| 2026-07-09 | Workload contract foundation | `sha256:c49a86951eab` | this commit | portable | `qualification/schema/workload-v1.schema.json` | passed | Exact byte identity, argv-only cases, structured results, closed comparison modes, and initial environment/core-correctness manifests; 14 focused tests |
| 2026-07-09 | Qualification-required backend tests | `sha256:31f9133268ee` | this commit | ROCm/Vulkan | core correctness hardware test binaries | passed | Exact `KILN_QUALIFICATION=1` opt-in converts missing devices and Vulkan initialization failures from developer skips into test failures |
| 2026-07-09 | Strict receipt comparison | `sha256:4fcbf0c90547` | this commit | portable | `scripts/qualification/compare_receipts.py` | passed | Committed policy binds source, model, parameters, config, environment exceptions, and metric semantics; 25 comparator tests and 88 isolated qualification tests |
| 2026-07-09 | Local qualification runner | `sha256:bedb3fc59bee` | this commit | portable + Strix Halo | `scripts/qualification/run.py` | passed | 36 runner tests and 126 qualification tests; strict bounded evidence, config/model/source attestation, exact repetition aggregation, no-network/PID isolation, and process-tree cleanup; real namespace probes preserved ROCm gfx1151 and RADV Vulkan access |
| 2026-07-09 | Shared strict JSON policy | `sha256:429d38681db8` | this commit | portable | `scripts/qualification/strict_json.py` | passed | One exact numeric, duplicate-key, integer-bound, and plain-UTF-8 policy now serves workloads, receipts, comparisons, and command results; 138 qualification tests |
| 2026-07-09 | ROCm core correctness | `sha256:429d38681db8` | `d457a89ce267` | Strix Halo ROCm | `qualification/receipts/rocm/strix-halo/20260709t225523453678z-rocm-strix-halo-core-correctness-v1-9faecc7321-v1.json` | passed | Real gfx1151 device probe, ROCm tensor parity, and ROCm matmul parity passed under loopback-only and PID-isolated execution in 88.4 seconds; receipt and all local artifact hashes revalidated |
| 2026-07-09 | Vulkan core correctness and cross-backend parity | `sha256:429d38681db8` | `2a6df2f83973` | Strix Halo Vulkan | `qualification/receipts/vulkan/strix-halo/20260709t225744609983z-vulkan-strix-halo-core-correctness-v1-0a09f3bcee-v1.json` | passed | Real RADV device probe, tensor parity, and matmul parity passed in 1.30 seconds; declared comparison against the ROCm receipt accepted only the listed backend fields and all three required case metrics were exactly equal |
| 2026-07-09 | Serving A/B configuration observability | `sha256:4ed553507ea1` | `39814f2a` | portable + ROCm | `/health`, `/v1/debug/model-state` | passed | Health now reports CUDA, ROCm, and Metal graph states independently; debug state exposes the ROCm-graph and KV-autoscale launch flags; focused ROCm-feature server tests and formatting passed |
| 2026-07-09 | KV autoscaler runtime observability | `sha256:779d5663643e` | `03cf5015` | portable + ROCm | `/health.decode_runtime.kv_autoscaler` | passed | Health exposes the resolved default/off state and bounded reason without inferring it from ambient environment |
| 2026-07-09 | Safe memory-governor default | `sha256:d4c3f94bdaf3` | `6234e764` | portable + ROCm | memory-governor unit tests and runtime state | passed | Physical reclaim is off by default; on-demand and automatic modes are explicit, strictly parsed, documented, and source-attributed |
| 2026-07-09 | Zero-yield ROCm pool-trim avoidance | `sha256:456146c5693c` | `afd40ee3` | ROCm/gfx1151 | `rocm_trim_pool` hardware tests | passed | Empty pools avoid device synchronization; trim honors the target, propagates HIP failures, and reports measured reserved-byte deltas with timing and reason |
| 2026-07-09 | Qualification variant binding | `sha256:7c60cce84c78` | `8dd88171` | portable | runner and workload unit suites | passed | Runner-owned variant identity is injected into each case, cannot be overridden by a manifest, and is recorded in the run configuration |
| 2026-07-09 | Response backpressure observability | `sha256:29b38d21102e` | `2529652b` | portable | batching-engine snapshots, debug state, and Prometheus metrics | passed | Monotonic counters distinguish channel saturation, actual wait, stall eviction, and closed receivers; exact structured events and 23 focused tests passed |
| 2026-07-09 | ROCm graph execution observability | `sha256:be0bdd7d1bef` | `15ee1f33` | ROCm/gfx1151 | `/health.decode_runtime.rocm_graphs` | passed | Health distinguishes requested, armed, and effective capture state; lifetime capture/replay outcomes and live graph count cover all decode APIs; ROCm-feature server check passed |
| 2026-07-09 | Backpressure health contract | `sha256:613bf86ee1d9` | `5db2ef19` | portable | `/health.decode_runtime.batching_engine` | passed | Health exposes the same saturation, wait, eviction, and closed-receiver counters as debug state and Prometheus; focused serialization test and formatting passed |
| 2026-07-09 | Server-side streaming token timing | `sha256:9a085c4c74aa` | `ee163c2f` | portable + ROCm serving path | opt-in `kiln.token_timing` SSE records | passed | Actor-ready timestamps survive response-channel waits; explicit performance streams expose request-relative ready and handler queue timing; 22 batching tests, focused SSE test, and server check passed |
| 2026-07-09 | Accepted-socket transport control | `sha256:9d68b2504fe4` | `115c1b18` | portable + ROCm serving path | `server.http_send_buffer_bytes`, `/health`, and debug state | passed | Strict bounded `SO_SNDBUF` opt-in is applied and read back on every accepted socket; focused config, API, loopback, portable, and ROCm-feature checks passed |
| 2026-07-09 | Local qualification operator guide | `sha256:9d68b2504fe4` | `f97f276b` | local hardware | `docs/qualification.md` | passed | Clean-tree setup, workload execution, strict validation, comparison, artifact policy, and receipt check-in are documented for new machines |
| 2026-07-09 | Pre-readiness socket-buffer proof | `sha256:9eae13441588` | `689ba61e` | portable + ROCm + Vulkan | listener preflight and accepted-socket regression tests | passed | Platform-normalized efficacy is proven before readiness; requested, kernel-readback, and effective bytes are retained; invalid clamp repro exits without a Ready line |
| 2026-07-09 | ROCm mixed-load qualification contract | `sha256:9ba9a989d350` | `7c3ded29` | Strix Halo ROCm target | `qualification/workloads/serving-mixed-rocm-v1.json` | passed | Four source-bound A/B arms cover batching, long prefill, cancellation, attributed socket pressure, actor-ready timing, graph execution, memory, and fail-closed policy checks; 29 focused and 169 total qualification tests passed |
| 2026-07-09 | ROCm default-arm graph warmup reproducer | `sha256:9ba9a989d350` | `7b49c236` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260710t005326407514z-rocm-strix-halo-serving-mixed-rocm-v1-f2e983b84c-v1.json` | failed | Built-in prewarm captured a graph, but four bounded request warmups produced no successful replay; the harness stopped before measurement instead of accepting unproven graph execution |
| 2026-07-09 | Production ROCm greedy graph routing | `sha256:c62f752dcd12` | `762804f5` | portable + ROCm | `GreedyBatchRoute` regression and backend capability contract | passed | Enabled single-row HIP replay now preempts the generic BF16 greedy path; graphs-off, non-ROCm, sampled, and multi-row routes retain their prior behavior |
| 2026-07-09 | Finished ROCm graph-owner cleanup | `sha256:2d7206487052` | `aafe39a6` | portable + ROCm | graph-owner unit and resource-concurrency contract tests | passed | Every batching-engine completion route releases the row's captured graphs and continuity timeline before fallible finish work, preventing the bounded graph cache from filling with dead request owners |
| 2026-07-09 | Direct ROCm graph-owner isolation | `sha256:5a7d09e0b256` | `ee058502` | portable + ROCm | allocator, owner-isolation, lifecycle-contract, and ROCm-feature server checks | passed | Direct and batched decode share a fail-closed process-wide ID namespace; every direct generation holds a scoped owner lease, all ROCm bs=1 graph APIs require a concrete owner, and recycled KV blocks cannot cross request timelines |
| 2026-07-09 | ROCm graph-owner lifecycle observability | `sha256:c70fe7a869ab` | `b6082716` | portable + ROCm | model counters, health serialization, lifecycle-contract, and ROCm-feature server checks | passed | Health reports live tracked owners plus monotonic owner and graph-release counts; bounded debug events bind owner start, recycled block identity, cleanup, and direct receiver loss to one row ID |
| 2026-07-09 | ROCm default-arm cancellation-reader reproducer | `sha256:c70fe7a869ab` | `b6082716` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260710t013325203346z-rocm-strix-halo-serving-mixed-rocm-v1-f2e983b84c-v1.json` | failed | Graph warmup captured and replayed without graph, resize, reclaim, trim, device-fault, or OOM events; after five seconds without a cancellation-stream byte, Python's buffered `HTTPResponse` became permanently unreadable after its first socket timeout, so the harness failed before it could report mixed-load metrics |
| 2026-07-09 | Socket-readiness qualification stream prototype | `sha256:42578a58f034` | `0a42c98d` | portable | mixed-load stream-reader and cancellation regressions | passed | Socket polling avoided retrying a poisoned timed-out reader and passed 34 focused plus 174 total qualification tests; the following real-device receipt exposed the missing already-buffered-body case |
| 2026-07-09 | ROCm buffered-body readiness reproducer | `sha256:42578a58f034` | `0a42c98d` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260710t024639147403z-rocm-strix-halo-serving-mixed-rocm-v1-f2e983b84c-v1.json` | failed | Server prewarm and native graph capture completed, but the first harness fix polled only kernel socket readiness and therefore starved SSE bytes already buffered by `HTTPResponse` while the keep-alive socket was idle; the bounded warmup deadline rejected the run |
| 2026-07-09 | Buffer-aware qualification stream reads | `sha256:bdaac8b3b632` | `e5587c0a` | portable + loopback HTTP | buffered-body, socket-readiness, deadline, cleanup, and cancellation regressions | passed | Each read drains `HTTPResponse`'s existing buffer before polling the socket and uses only the final request deadline for an actual blocking read; a real chunked keep-alive regression plus 36 focused and 176 total qualification tests passed |
| 2026-07-09 | ROCm batching-actor control-plane starvation | `sha256:bdaac8b3b632` | `e5587c0a` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260710t025524984730z-rocm-strix-halo-serving-mixed-rocm-v1-f2e983b84c-v1.json` | failed | Mixed load drove actor-backed health and metrics requests to 2.0-4.1 second latency before a five-second snapshot timeout aborted measurement; seven normal streams had emitted only 64-70 of 128 tokens after 30 seconds, while logs contained no resize, reclaim, trim, graph-failure, device-fault, or OOM event |
| 2026-07-09 | Nonblocking batching control-plane snapshots | `sha256:384bce9fb516` | `728db9b6` | portable + ROCm | blocked-forward, API, metrics, concurrency-contract, and backend compile checks | passed | Health, metrics, and debug read a timestamped actor-published cache while strong snapshots retain their ordering barrier; in-flight batch state is published before GPU work and `snapshot_age_ms` exposes staleness without rescanning prefixes on the token hot path |
| 2026-07-09 | ROCm mixed-load actor-stall baseline | `sha256:384bce9fb516` | `728db9b6` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260710t031209606216z-rocm-strix-halo-serving-mixed-rocm-v1-f2e983b84c-v1.json` | failed | The complete run measured p99/p99.9 ITL 4.22s/9.16s, p99 TTFT 31.8s, 33 unexplained gaps, and 23.3 tok/s despite 0.13ms p99 response-queue delay; graphs achieved 1 capture plus 45/45 replays with zero failures, KV stayed at 4096 blocks, reclaim stayed off, and the healthy pressure peer made no progress during the actor's attributed 2.012s slow-client wait |
| 2026-07-09 | Typed stream-stall startup policy | `sha256:b37a0b1d7fb9` | `cabea929` | portable + ROCm | config, batching actor, health/debug, and qualification attestation | passed | `server.stream_stall_grace_ms` is strictly parsed in 10..=2000 ms with default/config-file/environment provenance, injected immutably into the actor, and enforced against elapsed wall time; health/debug and the ROCm harness attest the effective value/source. All-target portable and ROCm checks, 23 actor tests, 30 config tests, 17 resource contracts, and 36 qualification tests passed; delivery remains actor-local and is the next checkpoint |
| 2026-07-09 | Off-actor response delivery | `sha256:a913bb0d2084` | `601a284f` | portable + ROCm | fair delivery worker, batching actor, health/debug/Prometheus, and lifecycle contracts | passed | Generation-keyed lanes preserve token/terminal order and continuous wall-clock grace without blocking peer decode or control commands; atomic result cohorts preserve sustained batch width, and cancellation-after-retirement plus concurrent-stop races are covered. All 654 server library tests, 18 resource contracts, 29 actor tests, 23 worker tests, and portable plus ROCm all-target checks passed |
| 2026-07-09 | ROCm off-actor-delivery hardware rerun | `sha256:a913bb0d2084` | `601a284f` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260710t045133806734z-rocm-strix-halo-serving-mixed-rocm-v1-f2e983b84c-v1.json` | failed | The validated clean-tree run removed the prior actor-local 2.012s delivery wait and reduced unexplained ITL gaps from 33 to 14, but the slow stream reached only 63 client-visible tokens and never filled its response channel, so the pressure-peer gate was not exercised. P99 ITL regressed 4.22s to 4.50s, p99 TTFT 31.8s to 38.9s, throughput 23.3 to 19.9 tok/s, and max batch 8 to 7; graphs completed 3/3 measured captures and 30/30 replays with no failures, while KV stayed fixed and reclaim stayed off |
| 2026-07-09 | Admission delivery publication barrier | `sha256:f613e0774f61` | `4456dd86` | portable + ROCm | delivery-worker watermark barrier, delayed eight-row admission regression, lifecycle contracts, and independent concurrency review | passed | The actor now waits for every first-token delivery result produced by one serial admission cohort to be published, then drains the complete FIFO result prefix before selecting decode rows. Healthy acknowledgements therefore become schedulable together while full lanes remain isolated; failed publication or a missing acknowledgement stops the actor within 5s. All 656 server library tests, 18 resource contracts, 29 actor tests, 25 worker tests, formatting, and portable plus ROCm all-target checks passed |
| 2026-07-09 | ROCm admission-barrier hardware rerun | `sha256:f613e0774f61` | `4456dd86` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260710t051840050197z-rocm-strix-halo-serving-mixed-rocm-v1-f2e983b84c-v1.json` | failed | Batch width and throughput recovered from the off-actor-delivery regression to max 8, mean 4.846 rows/forward, and 23.85 tok/s. The slow lane now saturated at sequence 127 and was evicted after 2.009s, but its pressure peer emitted no actor-ready token through that window and had 32.68s TTFT; p99/p99.9 ITL reached 8.28s/10.68s with 16 unexplained gaps. Graphs captured once and replayed 45/45 without failure, KV stayed at 4096 blocks, and reclaim and resize stayed inactive, isolating actor-owned preparation/prefill as the next scheduling target |
| 2026-07-09 | Backend-owned external-yield synchronization | `sha256:26723e522cfa` | `f96e5741` | portable + Strix Halo ROCm/Vulkan; CUDA/Metal contract | backend runtime facet, generated capability report, real device smoke tests, and all-target compile gates | passed | A required backend facet now drains the full CUDA context, ROCm device, Metal companion queue, or VulkanBackend-owned queue before an external scheduler may publish progress or recycle state; synchronization failure is explicitly quarantine-worthy. Metal terminal status and Vulkan sticky device loss fail closed. All 338 model tests and 52 backend contracts passed, the ROCm and Vulkan boundary tests ran on this host, and portable, ROCm, and Vulkan server all-target checks passed; CUDA and Metal hardware execution remain assigned to their target machines |
| 2026-07-09 | Immutable prefix-cache block reuse | `sha256:cc754bd8dc77` | `f4956fc5` | portable + ROCm + Vulkan | model validation, cache selection/registration, legacy-entry compatibility, and real batching integration | passed | Only block-aligned, shape-valid KV tables can cross request boundaries; non-aligned exact entries are neither produced nor retained, manually supplied unsafe hits are rejected in every exact-capable model path, and lookup skips legacy unsafe candidates before ranking so the longest safe strict prefix wins. An identical 81-token retry reproduced the first token, allocated a private suffix block, retained only the five complete cached blocks, and released the suffix on finish. All 339 model and 658 server library tests, focused integration tests, formatting, and portable, ROCm, and Vulkan server all-target checks passed |
| 2026-07-09 | Failure-atomic compatible prefix lookup | `sha256:9d147fe68f9d` | `eb386f41` | portable + ROCm + Vulkan | exact-hit sampling compatibility, strict-prefix fallback, injected snapshot failure, and all-target compile gates | passed | Exact cached logits remain reusable for sampled requests, while cached greedy tokens are eligible only for effectively greedy requests and cannot hide a shorter compatible strict prefix. State snapshots complete before hit counters, LRU order, or active-use pins mutate; an injected Vulkan-storage copy failure proved all accounting remains unchanged. All 11 focused prefix-cache tests and 660 server library tests passed, along with formatting and portable, ROCm, and Vulkan server all-target checks |
| 2026-07-09 | Atomic prefix-cache admission | `sha256:f214c03d9215` | this commit | portable + ROCm + Vulkan | overlap-capacity, pinned-LRU, duplicate-page, refcount, and all-target compile gates | passed | Registration now simulates the complete final refcount union and every required LRU removal before mutating resident state. Intrinsically oversized and duplicate-page tables are rejected, a late pinned-entry blockade cannot leave partial evictions, and successful shared-prefix admission preserves exact refcounts with disjoint retained and released block sets. Six focused adversarial tests and all 662 server library tests passed, along with formatting and portable, ROCm, and Vulkan server all-target checks |
| 2026-07-10 | Leased prefix ownership and generation fencing | `sha256:da76c63f4cc7` | this commit | portable + CPU integration | move-only request/provisional leases, tombstones, generation fences, and real batching integration | passed | Prefix sources are pinned before snapshot copies; hit counters and LRU commit only after success, while partial snapshot failure settles under the GPU permit and quarantines. Clear and purge hide live entries and defer reclamation; cache-local generations reject late registrations. This supersedes the earlier pre-pin snapshot ordering. All 669 server library tests and the real CPU multi-turn integration passed |
| 2026-07-10 | Settled paged streaming and GPU synchronization | `sha256:da76c63f4cc7` | this commit | portable; ROCm/Vulkan compile gates | worker finalizers, movable permit/writer exclusion, health/debug/Prometheus, SSE, reservation, and cache-owner contracts | passed | Paged workers run their sole cleanup before `Done` or `Error`. Decode panic or external-yield sync failure retains unknown GPU state/read ownership and latches quarantine for covered completion APIs; shared KV pages recycle only after settlement. SFT, including CUDA-native dispatch, uses the per-step writer. Bounded sync telemetry is exported, prefill and explicit worker failures emit structured errors, and probe-only completion cannot replace a concurrent cache owner. All 347 model and 669 server library tests, 19 concurrency contracts, formatting, and portable, ROCm, and Vulkan all-target checks passed |
| 2026-07-10 | Fallible streaming text and fail-closed terminals | `sha256:fdba4c099963` | this commit | portable; ROCm/Vulkan compile gates | injected decoder faults, saturated SSE queue, abrupt producer drop, tail salvage, and independent review | passed | Incremental decode and flush errors retain their offsets, roll back the accepted token, and propagate through model and batching streams instead of becoming empty text. Explicit worker failure, missing terminal events, timeout, and an unexpected producer drop emit a structured `generation_error` followed by `[DONE]` after queued deltas drain, even when the event queue is full; recent-request records preserve explicit error/timeout classification. Legacy detokenizer exits settle before page release. All 349 model and 670 server library tests, 19 concurrency contracts, formatting, and portable, ROCm, and Vulkan all-target checks passed; direct cancellation/deadline propagation remains the next checkpoint |
| 2026-07-10 | Settled direct-stream cancellation and deadlines | `sha256:8080e47d551d` | this commit | portable + CPU integration; ROCm/Vulkan compile gates | tiled-prefill/token cancellation, explicit worker settlement, panic ownership, deadline races, saturated terminal queues, and independent review | passed | Direct prefill, token enqueue, worker settlement, and success publication share one remaining absolute deadline. Timeout and disconnect signal cancellation, drain the sole bounded async bridge, and wait for explicit cleanup or intentional quarantine; terminal SSE events cannot be blocked behind the ordinary delta queue. Prefill and prefix-snapshot panics retain GPU/request ownership and quarantine; producer unwind cancels detached work, and racing progress cannot resurrect its gauge. Direct speculative streams temporarily fall back to documented single-token decode. All 354 model and 684 server library tests, the real CPU streaming integration, 19 concurrency and 3 timeout/drain contracts, the desktop default regression, formatting, and portable, ROCm, and Vulkan all-target checks passed; ROCm/Vulkan results here are compile gates, not runtime hardware evidence |
| 2026-07-10 | Settled legacy mutable paged streaming | `sha256:84c3cbb0f641` | this commit | portable + CPU integration; ROCm/Vulkan compile gates | exclusive reservation, error/receiver-drop/panic outcomes, owner retention, source invariant, and public API integration | passed | The earlier latent compatibility item was closed after the active direct-stream path. The synchronous mutable-`BlockManager` API and its shared CUDA-graph compatibility branch now allocate one retain-by-default reservation and converge every post-allocation exit through one backend-settlement epilogue. Success or settled error releases pages; failed or panicked settlement retains pages, linear state, logits, and result while releasing the CUDA coordination mutex. The API is documented as returning an already-populated receiver and deprecated toward threaded settlement. All 359 model tests, four focused reservation/panic tests, the source invariant, and a deterministic real CPU public-API integration passed, along with formatting and portable, ROCm, and Vulkan all-target checks; CUDA graph hardware execution remains assigned to the CUDA machine |
| 2026-07-10 | Fallible prompt-logprob token display | `sha256:25cce8ae2883` | this commit | portable; ROCm/Vulkan compile gates | strict token-ID mapping, injected decoder failure, special-token display, HTTP vocabulary-drift response, and independent protocol review | passed | Prompt-logprob display rendering is fallible through both mock and real response paths. Unknown tokenizer IDs and decoder faults return contextual HTTP 500 `tokenization_error` responses instead of successful empty fields, while known special tokens retain their literal vocabulary text. All 73 runnable core and 686 server library tests passed, along with formatting and portable, ROCm, and Vulkan all-target checks. The audit also narrowed the public claim to a vLLM-shaped subset and recorded numerical integrity, K/K+1 rank semantics, strict remote parsing, and tokenizer-identity handshake as explicit follow-up gates; ROCm/Vulkan results here are compile gates, not runtime hardware evidence |
| 2026-07-10 | Fail-closed prompt-logprob numerics | `sha256:c00979909bae` | this commit | portable + Strix Halo ROCm/Vulkan; CUDA deferred | exact-width and finite-value validation, deterministic K selection, fused log-softmax parity, wave/width sweep, and independent review | passed | Every local forward row, including the unused final row, is validated before ranking, decoding, or JSON; corruption returns `generation_error`, and ties resolve by token ID with exact K cardinality. CUDA/ROCm use one same-dtype output allocation and F32 fused accumulation instead of softmax-then-log or full-vocabulary F32 temporaries. All 981 tensor and 690 server library tests passed, as did portable, ROCm, and Vulkan all-target checks. Real ROCm qualification covered F32/BF16/F16 extremes, special values, multiple rows, wave boundaries, the strided path, widths through 248,320, and legacy softmax parity; real Vulkan covered extreme-range host-fallback parity. CUDA is source-mirrored but was neither compiled nor executed on this host and remains assigned to the CUDA-machine gate |
| 2026-07-10 | vLLM prompt-logprob semantics and settled scoring | `sha256:2bf0182153d4` | this commit | portable + Strix Halo ROCm/Vulkan; CUDA/Metal deferred | vLLM `7614b88ebdd9` protocol cross-check, scorer lifecycle and panic ownership, fused log-softmax, non-destructive Vulkan prewarm, deterministic real endpoints, and independent review | passed | Position zero is null; row `i-1` scores observed token `i`; K=0 returns only the observed token, and K>0 returns K or K+1 entries with full-vocabulary ranks and contextual byte fallback. One absolute deadline now covers admission, runner-owned backend execution, bounded LM-head chunks, rendering, cancellation settlement, and quarantine-worthy panic/synchronization failure. Portable suites passed (77 core with 3 ignored, 359 model, 703 server, 985 tensor, and 12 real-model integrations with 1 ignored), strict ROCm and Vulkan log-softmax hardware tests passed, and portable/ROCm/Vulkan server all-target checks passed. Real ROCm served five successful Qwen3.5-4B requests. The final production-default Vulkan build completed immutable weight prewarm in 62.5s instead of the prior destructive 211.5s path, then served 9/9 raw-ID and text requests with byte-identical repeated K=5, single-row, and text payloads across two restarts; both external-yield boundaries settled 9/9 with zero failures, the backend remained healthy and unquarantined, and no request created a new Vulkan device. The full Vulkan hardware library suite passed 396/396 with eight test threads and all 55 capability contracts passed. CUDA and Metal execution remain assigned to their target machines; the current scorer deliberately uses correctness-first O(TV) host ranking pending the checked device-side top-K gate |
| 2026-07-10 | Strict remote-teacher scoring and cache contract | `sha256:9df21078028f` | this commit | portable + Strix Halo Vulkan; ROCm compile; CUDA/Metal deferred | 326 train and 728 server library tests, 18 API integrations, strict vLLM fixtures, cache/alignment adversarial tests, atomic queue-admission races, real Qwen3.5-4B Vulkan scoring smoke, and two independent blocker reviews | passed | The only admitted remote protocol is vLLM numeric-ID `prompt_logprobs`; URL/auth/body/envelope/model/usage/single-choice/cardinality/canonical-ID/vocabulary/duplicate/rank/order/tie/mass checks fail closed, using source F64 values before F32 conversion. Logits row `p` is read from response target `p+1`, including an appended final-row probe, while OPD action target `q` uses causal logits row `q-1`; every complete batch is validated before cache writes, duplicate/out-of-order requests retain caller order, and v2 cache paths bind exact configured identity without traversal or digest-prefix collisions. Unsupported loss/stability/clipping/cost/provider/source combinations now fail before work; all submit surfaces share structural admission, effective K is reported, and multi-job publication is capacity-atomic against concurrent single submissions. The live smoke queried three rows at K=5 through kiln's strict client against kiln's real Vulkan vLLM-compatible endpoint after 63.4s immutable prewarm; it is not evidence from an actual vLLM process. Authoritative remote tokenizer/base/adapter revision is the next gate; local `model_id` still names the served base rather than loading an independent teacher, cache file replacement is not yet crash-atomic, and CUDA/Metal hardware execution remains deferred |
| 2026-07-10 | Cross-runtime tokenizer vocabulary identity | `sha256:a9c8195a72c2` | this commit | portable | complete numeric vocabulary mapping, added/special tokens, insertion-order invariance, mutation separation, and Python-compatible golden vector | passed | `KilnTokenizer` now hashes every token/ID pair with a versioned, length-framed binary contract rather than treating tokenizer JSON or a caller-supplied label as proof that raw numeric IDs mean the same thing. All 81 runnable core tests passed, including the exact two-token golden digest; this is the first independently published part of the authoritative remote-teacher identity checkpoint, not completion of that gate. |
| 2026-07-10 | Loader-owned base-model content revision | `sha256:d21323218f37` | this commit | portable | exact memory-mapped shard bytes, path/order invariance, byte and multiplicity separation, dense and GPTQ loader propagation | passed | The loader now publishes a domain-separated SHA-256 revision derived from the exact read-only shard mappings it parsed, so later remote-teacher identity cannot substitute a path, alias, mtime, or racy second disk scan for the weights actually loaded. All 362 model library tests passed; runtime identity publication and per-response pinning remain in progress. |
| 2026-07-10 | Canonical teacher identity contract | `sha256:7836d6804d93` | this commit | portable | 12 strict schema, canonical JSON, bounded base64url, digest, field mutation, adapter, duplicate/unknown-key, and malformed-input tests | passed | A dependency-free `TeacherIdentityV1` now binds the served model, exact base and optional adapter content, numeric vocabulary and tokenizer config, context/K/vocab bounds, raw-logprob protocol, implementation, and inference configuration into one canonical fingerprint. Ordinary aliases and stock vLLM fingerprints cannot parse as this identity; network discovery, per-response enforcement, and cache/receipt integration remain in progress. |
| 2026-07-10 | Post-upload model-source mutation guard | `sha256:aef49c5beeef` | this commit | portable | retained mappings/file descriptors, unchanged acceptance, same-length mutation, truncation, and unused-shard mutation tests | passed | The base revision now retains every exact loader mapping and open file until a bounded post-upload verification can recompute the revision. This closes the hash-before-copy race, does not trust paths or mtimes, and detects truncation without touching an invalid mmap. All 366 model tests and the model all-target check passed; the server startup call site lands with identity publication. |
| 2026-07-10 | Immutable vLLM teacher launcher | `sha256:802a02c22492` | this commit | portable; CUDA runtime deferred | 28 focused launcher tests, 204 qualification tests, Rust-compatible base/tokenizer golden hashes, canonical identity round trips, static-adapter revision and rank validation, override rejection, and direct `execve` construction | passed | `scripts/vllm_teacher.py` now computes an exact base/tokenizer/optional-adapter identity before launch, owns every identity-bearing vLLM option, disables dynamic LoRA and resolver plugins, and publishes the canonical fingerprint through vLLM's native custom-fingerprint option. Dry-run and manifest-only modes are dependency-free and redact credentials. vLLM and Transformers are not installed on the Strix Halo host, so an actual vLLM process remains a CUDA-machine handoff rather than claimed local evidence. |
| 2026-07-10 | Authoritative teacher identity and immutable logit provenance | `sha256:a167c72405c5` | this commit | portable; CUDA vLLM runtime deferred | exact workspace CI command, 763 server library tests, 364/365 train tests, 55 backend capability contracts, 7 remote identity integrations, 231 qualification tests, all-target model/server checks, formatting, and diff hygiene | passed | Remote registration performs a bounded two-probe identity handshake and every response, fixture row, cache entry, export, queue binding, receipt, and local adapter load is tied to the exact base/tokenizer/config/runtime/optional-adapter revision. Base models and vLLM model/adapter inputs load from private immutable snapshots; remote logit materialization happens before GPU ownership; legacy identities and cache imports fail closed. Fixed-fixture partial configs now retain their off-policy default and canonical submit-time LoRA validation. The prior CI regression in run `29113435875` was reproduced and repaired along with stale identity fixtures and brittle source-contract markers. No actual vLLM process was available on this host, so CUDA execution remains a handoff gate. |
| 2026-07-10 | Process-wide backend quarantine admission | `sha256:51d43132010a` | this commit | portable + CPU integration; accelerator runtime behavior inherited from the quarantine latch | full workspace tests, retained-owner writer and adapter-barrier races, training admission/queued-transition integrations, 231 qualification tests, all-target server check, formatting, and diff hygiene | passed | The process-lifetime health latch now gates completion/eval workers, prompt scoring, adapter barriers and disk mutations, startup prewarm, every training API and central publication transaction, queued-to-running transitions, memory preparation, job-wide GRPO/OPD/distillation writers, and each SFT step. Writers poll health instead of blocking behind intentionally retained unknown GPU ownership; APIs return 503 `backend_quarantined`, queued jobs become failed with webhook/metric accounting, health/readiness and Prometheus remain restart-required, and no reset endpoint exists. CI run `29120241674` exposed a coarse-filesystem same-length mutation hole in qualification model fingerprinting; every opened input is now content-verified by a second read and all 231 qualification tests pass deterministically. |
| 2026-07-10 | Revision-bound loaded adapters and inference caches | `sha256:12396b785a58` | this commit | portable + CPU integration | full workspace tests, 772 server library tests, 370 model tests, 17 real-model integrations with 1 ignored, 231 qualification tests, all-target server check, formatting, and diff hygiene | passed | The loader derives one canonical config/weights revision and publishes `{name, content_revision}` with the runner flip. Actor queues, direct inference, prefix KV, all deterministic cache keys and owner claims, purge generations, and response headers bind that exact tuple; stale queued work fails before prefill, direct swaps cannot cross request ownership, and late or revoked owners cannot resurrect purged results. A real loader/API test proves the wire revision, published state, and runner source identity are identical. |
| 2026-07-10 | Serialized adapter mutation and revision publication | `sha256:8e4d9d5842f5` | `ec0e4823..95a4c2f5` | portable + CPU/mock integration | 779 server library, 366 train with 1 ignored, 13 composition, 9 upload, 7 merge, 2 registry, and 19 concurrency tests; all-target server check; docs smoke and release checks | passed | Upload, merge, delete, gate demotion, training publication, per-request composition, and live weight transitions use one health-checked mutation contract. Publishers stage hidden outputs and rename atomically; training snapshots its start, compare-and-publishes the target revision, preserves intervening winners, reloads a loaded ungated same-name target inside the drained barrier, and rejects a loaded gated rewrite before GPU work. Delete and upload reject physically loaded content. Composed cache identities include exact source revisions, one guard covers synthesis through protected eviction, and CPU-heavy hashing/merge work stays off async workers. |
| 2026-07-10 | Token-budgeted prefill and fair mixed admission | `sha256:22a66f8622c` | `4103074b` | portable + Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260710t231446356203z-rocm-strix-halo-prefill-scheduling-v1-f397c29fc3-v1.json` | passed | A 32-token actor-cycle budget advances an eight-token short decode while literal 1K and 16K prefills progress round-robin, then cancellation releases the long row. The real-device deterministic nonzero GDN-plus-full-attention proof matched monolithic prefill across six quanta `[17, 17, 17, 17, 12, 1]`, including recurrent and conv state, the block-aligned split snapshot, first and following decode tokens, and KV-block release. All 3 required qualification cases passed with zero assertion failures or skips; 231 qualification-tooling tests and 19/20 runnable real-model integrations also passed. The paired Vulkan arm remains an explicit next-machine-order rerun, not claimed by this receipt. |
| 2026-07-10 | Vulkan bounded-prefill device-mismatch reproducer | `sha256:22a66f8622c` | `323c4e1e` | Strix Halo Vulkan/RADV | `qualification/receipts/vulkan/strix-halo/20260710t231551307388z-vulkan-strix-halo-prefill-scheduling-v1-6899c96516-v1.json` | failed | Device discovery and the literal short-decode/1K/16K actor contract passed, but the real hybrid-model prefill failed closed before measurement because a full-attention elementwise multiply received `vulkan:0 F32 [80,2]` and `cpu F32 [80,2]` inputs. The receipt contains zero skips or unsupported items and is retained as the exact clean-source reproducer for the following backend fix. |
| 2026-07-10 | Paired production-path bounded-prefill qualification | `sha256:c308dc525215` | `d845d402..15a547df` | Strix Halo ROCm/gfx1151 + Vulkan/RADV | `qualification/receipts/rocm/strix-halo/20260710t233006344028z-rocm-strix-halo-prefill-scheduling-v1-f397c29fc3-v1.json`; `qualification/receipts/vulkan/strix-halo/20260710t233042331023z-vulkan-strix-halo-prefill-scheduling-v1-6899c96516-v1.json` | passed | The superseding Vulkan fixture matches serving ownership: deterministic BF16 model tensors and the canonical paged pool remain CPU-backed while the Vulkan runtime seeds its separate resident cache at decode entry; the attention-output gate is enabled as in Qwen3.5. This proved monolithic versus `[17, 17, 17, 17, 12, 1]` resumable prefill, recurrent and conv state, the position-80 snapshot, first token, the following native resident decode token, and block release. The earlier device mismatch was therefore a fixture-contract error, not a production runtime defect, and no speculative RoPE transfer change was retained. Both clean-source receipts passed all three required cases with no skips or assertion failures; `compare_receipts.py` reported compatible cross-backend correctness, no rejected differences, and exact equality for every required metric. |
| 2026-07-10 | Stable runtime mutation guards | `sha256:229d9098f3c0` | this commit | portable | server/model checks, two profile graph-policy tests, and real CPU paged-cache resize rejection | passed | Production startup now resolves graph eligibility once from the immutable serving profile and passes explicit CUDA, ROCm, and Metal options into the runner. Stable and maintenance profiles cannot live-capture graphs; stable construction does not register allocator reclaim hooks, retry allocation through reclaim, start KV autoscaling, or permit the actor's physical resize operation. The compatibility constructor also resolves to stable. The real paged-cache test proves a rejected resize leaves capacity unchanged, and the profile tests prove experimental remains the only live-graph route. This is a portable enforcement checkpoint; adapter/training admission, health provenance, documentation, and the real ROCm stable-profile trace remain open. |
| 2026-07-10 | Stable adapter and training admission | `sha256:c7eb316af477` | this commit | portable + real CPU integration | 786 server library tests and 23/24 real-model integrations | passed | Every real-backend training API and central batch publication now returns `409 serving_profile_conflict` in stable mode, while the queued-to-running transition independently finalizes injected or restored jobs before memory reservation, reclamation, or GPU coordination. Every live adapter flip is guarded inside the swap primitive before disk-to-device loading and the actor barrier; load/unload endpoints reject earlier, while true no-op selections remain available. Three retained-inference-owner tests prove adapter admission, training admission, and queued-worker rejection complete without waiting for or retaining the actor-wide writer. Intentional adapter/training tests now opt into the experimental profile explicitly. |
| 2026-07-10 | Maintenance inference drain enforcement | `sha256:9bd7972295c1` | this commit | portable + real CPU integration | 788 server library tests and 24/25 real-model integrations | passed | Maintenance is a restart-only drain boundary: chat, text, batch, prompt-logprob, and agent-run admission return `503 inference_disabled_by_profile`; inference prewarm is suppressed; all eval publications are rejected before queue/tracking insertion; injected eval work becomes a fully archived failed job before its generator is invoked; and the live generator independently checks before adapter selection and generation. A real-backend test holds the exclusive GPU owner while chat and agent requests return promptly, proving they never wait for an inference read owner. Maintenance still performs a real adapter load and publishes its exact content revision, demonstrating that drained exclusive operations remain available. Health provenance and permanent operator documentation follow in the next checkpoint. |
| 2026-07-10 | Serving-profile diagnostics and operator contract | `sha256:ec21510b0e58` | this commit | portable + real CPU integration | 791 server library tests, 24/25 real-model integrations, all-target server check, formatting, shell syntax, and diff hygiene | passed | One typed resolution report now drives startup diagnostics, `/health`, `/v1/health`, and `/v1/config`, publishing the selected profile, default/file/environment provenance, request-override prohibition, and every effective ownership policy. Maintenance reports a healthy backend separately while readiness returns 503 with `status=maintenance`; stable governor health distinguishes requested reclaim from the profile-forced effective `off`. The permanent policy matrix and restart-only drain runbook are linked from the README, quickstart, training/eval guides, example config, and demo scripts; interactive training demos opt into experimental while inference-only demos retain stable. The stable process still admits logical inference scheduling, cancellation, eviction, and prefix reuse while all physical mutation gates remain closed. |
| 2026-07-10 | Stable ROCm mixed-load qualification contract | `sha256:72bc152f7931` | this commit | portable; Strix Halo ROCm execution follows from the clean commit | 232 qualification-tooling tests, strict workload validation, Python compilation, and diff hygiene | passed | The four historical autoscale/graph A/B arms now pin `experimental`, preserving their actual dynamic-runtime semantics after stable became the default. A distinct `stable` arm requests autoscaling, automatic reclaim, and ROCm graphs but requires health to prove that the profile suppresses every effective mutation path. The receipt contract now records backend external-yield synchronization calls, failures, slow calls, total time, and maximum duration by bounded boundary; stable fails on any failed or >=100 ms sync, resize/reclaim/graph event, changed KV capacity, attributed or unexplained ITL outlier, or missing synchronization evidence. This is a committed pre-benchmark harness checkpoint, not device evidence. |
| 2026-07-10 | Stable ROCm padded-vocabulary startup reproducer | `sha256:72bc152f7931` | `6f7b860474a7` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t002223202949z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | failed | Clean-source stable startup loaded and uploaded Qwen3.5-4B, then rejected its legitimate padded model vocabulary: the tokenizer has 248,070 assigned IDs while the embedding/logit width is 248,320. No workload request ran, all dynamic mutation paths were effectively off, and the private model snapshot was cleaned up. The receipt is retained as the exact reproducer for the following identity-contract fix. |
| 2026-07-10 | Padded model-vocabulary identity contract | `sha256:8b49019be45a` | this commit | portable; ROCm rerun follows from the clean commit | 56 vLLM launcher tests, 233 qualification tests, 82/85 runnable core tests, and 793 server library tests | passed | Local and vLLM teacher identities now record the model embedding/logit width while allowing a smaller tokenizer map for reserved padding rows. Both paths still fail closed when the tokenizer entry count exceeds the model width or any assigned numeric ID falls outside it; the vLLM launcher also resolves nested `text_config.vocab_size` and rejects conflicting declarations. Documentation states the distinction explicitly. This portable fix checkpoint does not substitute for the following Qwen3.5-4B ROCm receipt. |
| 2026-07-10 | Stable ROCm post-startup stall and teardown reproducer | `sha256:8b49019be45a` | `1265cfbf5713` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t003239968059z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | failed | The padded-vocabulary fix reached readiness and completed ten measured requests with zero request failures, resize, reclaim, or graph activity and fixed 4,096-block KV capacity. All 340 external-yield synchronizations succeeded in 3.709 ms total with a 0.369 ms maximum, ruling out that boundary as the observed pause source. The run still produced 37 unexplained ITL gaps (943 ms p99), 24.16 s pressure-peer TTFT, and no peer progress during the attributed slow-consumer window. The server logged clean SIGTERM shutdown, but its worktree-local 9.3 GB snapshot made the run dirty and teardown exceeded the harness's 10 s grace, leaving one 5.0 GB shard; lifecycle cleanup and scheduling attribution are separate follow-up fixes. |
| 2026-07-10 | Run-owned model snapshot teardown contract | `sha256:102b4e658186` | this commit | portable; Strix Halo ROCm teardown rerun follows | 38 focused mixed-load tests, all 234 qualification tests, strict workload validation, and Python compilation | passed | The mixed-load server now places immutable model snapshots below its ignored per-run directory, grants real-model destruction a bounded 60-second SIGTERM window, records exit code/elapsed time/forced-kill state/residual payload paths, and makes a forced or incomplete payload cleanup fail the case. The harness remains final owner of the run directory after child exit, so even a killed child cannot leave multi-gigabyte worktree state. This fixes evidence contamination and residue recovery only; it does not resolve or waive the measured scheduler stalls. |
| 2026-07-10 | Snapshot payload residue attestation | `sha256:9b79f50c8201` | this commit | portable + Strix Halo ROCm teardown observation | 39 focused mixed-load tests, all 235 qualification tests, and Python compilation | passed | A clean hardware rerun exited the server normally in 264 ms, kept Git clean, and removed both 9.3 GB snapshot payload files; only an empty temporary directory remained until the run owner removed its ignored tree. The lifecycle gate now distinguishes that harmless empty control directory from any file, symlink, forced kill, or nonzero exit, all of which still fail. The intermediate duplicate latency receipt was not retained; the next clean commit produces the source-bound scheduler reproducer. |
| 2026-07-10 | Stable ROCm isolated stall and snapshot-residue reproducer | `sha256:9b79f50c8201` | `aa4c2271b6b9` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t004500068632z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | failed | The clean run reproduced 37 unexplained ITL gaps (976 ms p99), 23.45 s pressure-peer TTFT, and no pressure-window peer progress while fixed KV capacity, zero graph/resize/reclaim activity, 0.79 ms p99.9 response-queue delay, and 343 successful external-yield synchronizations totaling 3.489 ms exclude those suspected paths. SIGTERM returned 0 without force in 164 ms and Git stayed clean, but Kiln's silent `TempDir` drop left one 5.3 GB snapshot shard; the run owner removed it. The receipt is the exact common reproducer for explicit snapshot destruction followed by scheduler attribution. |
| 2026-07-10 | Explicit immutable-snapshot destruction | `sha256:e9a27f934dcb` | this commit | portable; Strix Halo ROCm teardown rerun follows | all 371 model library tests, read-only last-owner recovery regression, server all-target check, formatting, and diff hygiene | passed | `ModelSnapshotLease` now explicitly closes its temporary directory at the last source/deferred-MTP owner, emits structured removal timing, and retries three times after restoring private owner deletion permissions. Exhaustion emits an error with the exact residual path instead of silently accepting `TempDir` best-effort failure. Callers use the lease path accessor, and operator documentation describes cleanup recovery. This portable checkpoint requires the following full-size snapshot hardware proof. |
| 2026-07-10 | Explicit snapshot-drop hardware counterexample | `sha256:e9a27f934dcb` | `90880ea18354` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t005129233232z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | failed | The full-size run proved destructor cleanup is not a sufficient server guarantee: SIGTERM returned 0 without force in 164 ms, but no lease cleanup event appeared and one 5.3 GB shard remained for the run owner to remove. The same run reproduced 42 unexplained ITL gaps (949 ms p99, 1.05 s p99.9), 25.44 s p99 TTFT, and no pressure-window peer progress while synchronization totaled only 3.550 ms and mutation paths stayed inactive. Cleanup ownership must move to an explicit drained server-shutdown handle; the destructor remains only a fallback. |
| 2026-07-10 | Drained server snapshot cleanup ownership | `sha256:cc3a299adce5` | this commit | portable; Strix Halo ROCm teardown rerun follows | all 372 model library tests, five focused snapshot tests, server all-target check, formatting, and diff hygiene | passed | The loader now returns an opaque idempotent cleanup handle that does not keep mapped weights alive. Real server startup retains it, and shutdown invokes it only after Axum and the batching engine drain; any residual cleanup error makes shutdown nonzero with the exact path. Lease drop remains fallback coverage for startup failures and library use. The tests prove explicit cleanup while other lease owners remain, repeated cleanup, last-owner cleanup, and read-only recovery. Full-size hardware remains the acceptance gate. |
| 2026-07-10 | Drained snapshot cleanup hardware proof and latency baseline | `sha256:cc3a299adce5` | `6d1e45095174` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t005850623279z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | failed | The explicit drained owner removed the full 9.3 GB snapshot in 225 ms before `server stopped cleanly`; SIGTERM returned 0 without force in 414 ms, snapshot residue was empty, and Git stayed clean. With lifecycle contamination eliminated, the only gate failures were no pressure-window peer progress and 30 unexplained ITL gaps (823 ms p99, 955 ms p99.9); 379 synchronization calls totaled 4.402 ms with a 0.377 ms maximum, response-queue p99.9 was 0.98 ms, and graphs/resize/reclaim remained inactive. This is the clean source-bound scheduler baseline. |
| 2026-07-10 | Idempotent snapshot fallback diagnostics | `sha256:a31d22962ad9` | this commit | portable | five focused snapshot lifecycle tests, formatting, and diff hygiene | passed | After the drained shutdown owner removes a snapshot, the later lease fallback now treats `NotFound` as idempotent success instead of emitting a misleading recovery warning. Cleanup behavior and the hardware latency baseline are unchanged; future shutdown logs contain one authoritative removal event. |
| 2026-07-10 | Batching actor phase attribution | `sha256:8b3683c2f209` | this commit | portable + ROCm compile gate; Strix Halo runtime attribution follows | 34 batching/health/debug tests, Prometheus rendering, 235 qualification tests, portable and ROCm all-target checks, formatting, and diff hygiene | passed | Admission, bounded prefill, and decode forwards now publish cumulative and maximum wall time plus 100 ms slow-phase counters through health, debug state, and Prometheus. Each slow phase emits one bounded structured event with phase/work size; the mixed-load receipt contract records phase deltas and requires event/counter agreement. This passive checkpoint will distinguish the repeated 505-654 ms actor-ready gaps between prefill and decode without logging normal forwards or request content. |
| 2026-07-10 | Stable ROCm actor-phase attribution proof | `sha256:8b3683c2f209` | `d40d02099078` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t011556512375z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | failed | The source-bound run converted all 30 unexplained ITL gaps into causal phase evidence: every outlier overlapped a bounded prefill, sometimes followed by a decode, and zero remained unexplained. Admission totaled 3.34 ms with a 0.64 ms max; 43/43 measured prefills crossed 100 ms, totaled 12.88 s, and maxed at 656.6 ms; decode maxed at 157.0 ms. This rules out admission, response delivery (0.36 ms p99.9 queue delay), external-yield sync (3.98 ms total, 0.36 ms max), resize, reclaim, and graphs. The 512-token prefill remainder is monopolizing the single actor between decode cohorts; the next checkpoint must bound prefill independently below the 250 ms ITL gate. Snapshot teardown remained clean and unforced. |
| 2026-07-10 | Independent prefill-cycle ceiling | `sha256:1a7db6a7a002` | this commit | portable + ROCm compile gate; Strix Halo runtime proof follows | 796 server library tests, 235 qualification tests, portable and ROCm all-target checks, formatting, JSON/Python validation, and diff hygiene | passed | A typed `server.max_prefill_tokens_per_cycle` policy now bounds admission plus resumable prompt work independently of the combined actor-cycle budget after ready decode rows reserve their tokens. The stable default is 64 tokens, chosen from the measured 110-135 ms small-prefill floor and 157 ms decode maximum rather than by weakening the 250 ms ITL gate; strict TOML/environment validation, source provenance, startup diagnostics, health/debug state, Prometheus, qualification attestation, operator docs, and a literal short-decode/1K/16K fairness regression all share the same contract. This is a committed scheduler checkpoint, not hardware acceptance; the next clean-source ROCm receipt must still show zero attributed and unexplained ITL outliers plus pressure-window peer progress. |
| 2026-07-10 | Stable ROCm 64-token prefill-ceiling counterexample | `sha256:1a7db6a7a002` | `2bb1e4d8f198` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t013434402779z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | failed | The clean source-bound run reduced attributed ITL outliers from 30 to 5 and kept unexplained outliers at zero, but a 249.3 ms late-prompt prefill followed by decode as high as 154.8 ms still produced 322-363 ms gaps. P99/p99.9 ITL fell to 460/568 ms, while the added small-forward overhead reduced throughput to 19.30 tok/s. The slow lane reached backpressure only near the end of its 45-second observation window, so no 2-second eviction or three-sided pressure-peer overlap was proven. All 492 external-yield synchronizations succeeded in 4.995 ms total, response-queue p99.9 was 1.18 ms, mutation paths stayed inactive, and unforced shutdown removed all snapshot payloads in 415 ms. The next clean checkpoint halves the prompt ceiling instead of weakening any acceptance gate. |
| 2026-07-10 | Calibrated 32-token stable prefill ceiling | `sha256:0411e68eb442` | this commit | portable + ROCm compile gate; Strix Halo runtime proof follows | 796 server library tests, 235 qualification tests, portable and ROCm all-target checks, formatting, JSON/Python validation, and diff hygiene | passed | The only scheduler change from the preserved 64-token counterexample is a 32-token stable default, propagated through config, examples, operator docs, and every qualification variant/attestation. The override range remains 1..=65,536 and historical receipts remain immutable. This is a committed calibration checkpoint; hardware must still prove that shorter late-position prefills eliminate the five attributed gaps, allow the slow lane to reach a full two-second eviction window, preserve peer decode progress on both sides of that window, and complete the long prompt within the existing deadline. |
| 2026-07-10 | Stable ROCm 32-token prefill-ceiling counterexample | `sha256:0411e68eb442` | `c47e41ec3c23` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t014234610452z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | failed | Halving the fixed token ceiling is not a viable latency policy. At 32 tokens, one attributed 322 ms gap remained, a late prefill still reached 232.5 ms, and even a three-token prefill took 118.7 ms; fixed layer-launch and sequence-position cost dominate token count. Prefill wall time more than doubled to 57.58 s across 385 calls, throughput fell to 9.55 tok/s, and long-prefill TTFT rose to 77.79 s. The slow lane did finally prove one 2.003 s attributed eviction, but the pressure peer failed during chunked prefill with `hipblasLtMatmulAlgoGetHeuristic returned no algos`, producing the second batching error and preventing peer-overlap acceptance. Unexplained outliers remained zero, response-queue p99.9 was 0.30 ms, mutation paths stayed inactive, and shutdown was clean and unforced in 514 ms. Preserve this as evidence against further blind halving; investigate the ROCm matmul failure and replace the fixed-ceiling calibration with cost-aware or mixed-forward scheduling. |
| 2026-07-10 | Fail-open ROCm BLAS heuristic hardening | `sha256:1f91cfa4b62c` | this commit | portable + Strix Halo ROCm/gfx1151 | 29 ROCm-BLAS unit tests, 985 portable tensor tests, 9 real-device matmul parity tests, portable and ROCm server all-target checks, formatting, and diff hygiene | passed | A hipBLASLt cache miss now searches with the handle's full documented 32 MiB workspace budget instead of accidentally constraining first-use shapes to 1 MiB. If the explicit heuristic still returns no candidate for a valid descriptor, the wrapper executes hipBLASLt's supported implicit/default algorithm, leaves the result uncached, and increments a handle fallback counter rather than failing the request. Terminal errors now include exact M/N/K, logical batch/index, dtypes, and layouts. Real parity covers BF16/F32, transposes, batching, bias, and the pressure peer's 32-token attention sequence plus 14-token `[14,256] x [256,238]` tail. That isolated sweep also passed before the fix, so it narrows but does not falsely claim reproduction; the next full mixed-load run remains the intermittent production-path acceptance proof. |
| 2026-07-10 | Restore measured-better 64-token interim default | `sha256:8bfa8d8cdcaa` | this commit | portable; ROCm policy unchanged pending scheduler replacement | 2 focused server config tests, 39 mixed-load contract tests, formatting, and diff hygiene | passed | The disproven 32-token default is rolled back in config, examples, operator docs, and qualification expectations while its failed receipt remains immutable. This does not claim that 64 tokens meets the final latency gate: its preserved receipt still has five attributed gaps. It simply keeps `main` on the materially better measured interim behavior (19.30 versus 9.55 tok/s, no measured request failure, and 44.27 versus 77.79 s long-prefill TTFT) while the next checkpoint replaces token-only calibration with a scheduler policy that accounts for fixed launch and sequence-position cost. |
| 2026-07-10 | Layer-resumable prefill scheduling | `sha256:d1b00df4a314` | this commit | portable + CPU runtime; ROCm/Vulkan compile gates | 799 server, 372 model, and 235 qualification tests; CPU monolithic/resumable parity and retained-state cleanup; portable, ROCm, and Vulkan all-target checks | passed | The actor keeps the measured-better 64-token chunk but now yields after a typed default of four transformer layers, retaining the chunk's hidden, position, rotary, paged-KV, and linear-attention progress so ready decode can run without replaying completed layers. A forced eight-layer/two-layer-quantum regression proves decode interleaving across zero-token yields; CPU parity covers six real layer yields, and cancellation/discard release retained ownership only after backend settlement. Strict TOML/environment validation and provenance, startup logs, health/debug state, Prometheus counters, examples, operator docs, and every mixed-load variant share one contract; qualification now fails if it observes no inter-layer yield. This is a committed scheduler checkpoint, not ROCm latency acceptance; the next clean-source Strix Halo run must prove the existing ITL, peer-progress, error, and teardown gates. |
| 2026-07-10 | Layer-resumable prefill ROCm parity | `sha256:d1b00df4a314` | `745b3446c82d` | Strix Halo ROCm/gfx1151 | strict real-device hybrid prefill parity integration | passed | Qualification mode required a real ROCm device and ran the shared actor path through one GDN and one full-attention layer in one-layer groups. Six retained layer yields across token quanta `[17, 17, 17, 17, 12, 1]` matched monolithic first-token and next-decode output, recurrent state, split snapshot, sequence length, and block cleanup. This proves backend correctness and ownership for the mechanism on this device; it is not the full Qwen3.5-4B mixed-load latency acceptance receipt. |
| 2026-07-10 | Layer-resumable prefill Vulkan parity | `sha256:d1b00df4a314` | `64aa52ba9285` | Radeon 8060S / RADV STRIX_HALO, Mesa 26.1.3 | strict production-runtime hybrid prefill parity integration | passed | Qualification mode required the real Vulkan runtime and exercised the production CPU-backed model/paged-pool path with BF16 weights and attention output gating. Six retained layer yields across token quanta `[17, 17, 17, 17, 12, 1]` matched monolithic first-token and next-decode output, recurrent state, split snapshot, sequence length, and block cleanup. This is Vulkan mechanism and ownership evidence, not a full-model serving throughput receipt. |
| 2026-07-10 | Stable ROCm layer-yield accounting counterexample | `sha256:d1b00df4a314` | `5172d5eb01d4` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t023311359387z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | failed | Layer scheduling eliminated the pause failure: 1,001 inter-layer yields processed 4,576 layers with a 35.6 ms maximum prefill group, 86.5 ms maximum decode, 87/116 ms p99/p99.9 ITL, and zero attributed or unexplained outliers. The run then exposed a scheduler-accounting bug rather than a ROCm math failure: five requests were rejected when a retained 64-token chunk completed in a later cycle whose remaining admission budget was 16-55 tokens, yielding six total batching errors including the intentional stalled-client eviction and invalidating pressure-peer overlap. External synchronization stayed clean at 19.4 ms total/0.097 ms max, mutation paths stayed inactive, and unforced shutdown removed all snapshot payloads in 414 ms. Preserve the selected chunk width across layer yields and defer its continuation until the current cycle can honor that reservation; do not weaken the token-budget invariant. |
| 2026-07-10 | Retained prefill token-width reservation | `sha256:bccc49b9f908` | this commit | portable + Strix Halo ROCm/Vulkan | 800 server, 372 model, and 235 qualification tests; exact-width actor regression; CPU/ROCm/Vulkan resumable parity; portable, ROCm, and Vulkan all-target checks | passed | A layer-resumable chunk now exposes the token width selected when it began. If admission or decode leaves a smaller remainder in a later actor cycle, the scheduler defers that chunk without executing it, shrinking it, replaying completed layers, or weakening the strict post-forward token-budget check. A deterministic 64-token/eight-layer regression proves a 32-token remainder performs no model work or request failure, preserves the reservation, remains schedulable, and completes exactly once when 64 tokens fit. Health, debug state, Prometheus, qualification receipts, examples, and operator docs expose the cumulative deferral count. Real CPU, ROCm, and Vulkan hybrid parity still produced identical monolithic/resumable first and next tokens across six layer yields; cancellation released retained ownership. This is a committed correctness checkpoint; the next clean-source full ROCm run must prove request success and pressure-window overlap while retaining the already-achieved zero-outlier latency result. |
| 2026-07-10 | Stable ROCm retained-width correctness and admission-starvation counterexample | `sha256:bccc49b9f908` | `adb901293b3e` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t024917687277z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | failed | The retained-width fix closed the accounting defect: all ten measured requests succeeded, 128 narrower-cycle deferrals produced no request error, and the sole batching error exactly matched the intentional stalled-client eviction. Layer scheduling retained zero attributed and unexplained ITL outliers across 1,463 yields; max prefill/decode were 36.2/92.9 ms and p99/p99.9 ITL were 96/137 ms. The remaining failure is admission starvation rather than an execution pause: max decode width was only two rows, the dedicated pressure peer did not receive its first token until 24.7 s after dispatch and therefore began only after the attributed 2.007 s pressure window ended, despite later completing successfully. Mean rows per forward were 1.118, throughput was 12.16 tok/s, and the long-prompt TTFT was 67.9 s. All 2,562 external-yield synchronizations succeeded in 33.3 ms total with a 0.396 ms maximum; stable mutation paths stayed inactive; unforced shutdown returned zero in 516 ms and left no snapshot residue. Preserve this receipt, then make ready-prefill admission fair enough for a newly queued peer to reach decode without regressing the proven sub-250 ms inter-token bound. |
| 2026-07-10 | ROCm width-16 admission diagnostic | `sha256:bccc49b9f908` | `adb901293b3e` | Strix Halo ROCm/gfx1151 | exact stable mixed workload with an isolated `KILN_MAX_DECODE_BATCH=16` runtime override | failed | Raising the active/decode cap admitted the pressure peer earlier and reduced its TTFT from 24.7 s to 17.3 s without an ITL regression (89/132 ms p99/p99.9, zero outliers), but it still missed the start of the pressure window by about 0.4 s and max observed decode width remained two. Throughput moved only from 12.16 to 12.59 tok/s. This rules out a blind ROCm cap increase as the complete fix; prefill service order and true multi-row formation remain the measured bottlenecks. The diagnostic result stayed local because the in-memory override was not a source-bound supported configuration; only this compact conclusion is retained. |
| 2026-07-10 | Charge retained prefill tokens at chunk selection | `sha256:14420f772346` | this commit | portable + Strix Halo ROCm/Vulkan | 800 server, 372 model, and 235 qualification tests; exact reservation regression; CPU/ROCm/Vulkan hybrid parity; portable, ROCm, and Vulkan all-target checks | passed | Prefill progress now separates tokens newly scheduled from tokens whose final transformer layer completed. The actor charges a chunk once when selecting it, validates every later completion against the immutable retained width, and yields after the resumed layer group without making that completion compete for a second new-token budget. A deterministic 64-token/eight-layer case resumes under a 32-token new-work remainder with no replay, shrink, deferral, or error; real parity proves each chunk is scheduled and completed exactly once across six yields, and cancellation still releases retained ownership only after settlement. The now-impossible deferral counter and its API, Prometheus, qualification, and documentation surface were removed rather than preserved as dead telemetry. This corrects accounting and simplifies the contract; bounded short-prefill service is still required to resolve the hardware pressure-peer TTFT counterexample. |
| 2026-07-10 | Bounded short-prefill service lane | `sha256:ef09a0e918f2` | this commit | portable; ROCm/Vulkan compile gates | 801 server, 372 model, and 235 qualification tests; deterministic mixed-tail scheduling regression; portable, ROCm, and Vulkan all-target checks | passed | Three of every four eligible prefill dispatches remain strict round-robin. The fourth may select the smallest remaining tail only when it fits within four configured token chunks; if none qualifies, that dispatch also remains round-robin. A five-row regression proves four 1K prompts each retain round-robin progress while a late 128-token prompt receives the bounded opportunities and reaches decode-ready first. Priority selection preserves the independent round-robin cursor across success and failure, so continuous short arrivals cannot consume more than 25% of dispatch capacity. Health, debug state, Prometheus, qualification receipts, examples, and operator docs expose the counter, and the mixed workload now requires the path to execute. This is a committed scheduler checkpoint; the next source-bound ROCm receipt must prove pressure-window peer progress, zero request failures/outliers, and clean teardown before the policy is accepted on hardware. |
| 2026-07-10 | Stable ROCm bounded-prefill acceptance | `sha256:ef09a0e918f2` | `674c150b1fcf` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t032233364936z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | passed | The clean source-bound run exercised 171 bounded short-tail dispatches and the dedicated peer emitted actor-ready tokens before, inside, and after the attributed 2.016 s stalled-client window. All ten measured requests succeeded; the sole batching error exactly matched the intentional eviction. Across 1,344 layer yields, max prefill/decode were 37.0/87.6 ms, p99/p99.9 ITL were 77.6/143.1 ms, and attributed plus unexplained outliers remained zero. The pressure peer TTFT fell from 24.7 s to 16.6 s while the 4,653-token long prompt still completed in 66.8 s. All 2,465 external-yield synchronizations succeeded in 30.3 ms total with a 0.359 ms maximum; stable resize/reclaim/graph paths stayed inactive; response-queue p99.9 was 0.76 ms; unforced shutdown returned zero in 614 ms and left no snapshot residue. This closes the isolated pause/backpressure correctness gate, not the throughput program: max decode width remained two, mean rows per forward were 1.071, and output throughput was only 12.48 tok/s, so true multi-row formation and the vLLM comparison remain explicit follow-up work. |
| 2026-07-10 | Effective decode-width observability | `sha256:77d3085658ce` | this commit | portable; ROCm/Vulkan compile gates | actor snapshot, health/debug JSON, Prometheus, and focused serialization tests | passed | The batching engine now publishes its effective concurrent decode-row ceiling after backend selection and the combined token budget constrain it. Operators and the benchmark can therefore distinguish a configured cap from an observed formation failure instead of interpreting `max_observed_batch_size` in isolation. Focused actor, health, debug, and Prometheus tests passed, as did formatting and portable, ROCm, and Vulkan server all-target checks. The direct `KILN_MAX_DECODE_BATCH` parser is still permissive and untyped; strict startup configuration remains follow-up rather than being hidden by this evidence. |
| 2026-07-10 | Fail-closed shared serving benchmark v2 | `sha256:61197fed3078` | `db68b153dc62` | portable; Kiln/vLLM protocol contract | 246 qualification tests including 11 focused socket, parser, receipt, counter, memory, and failure tests; local vLLM `30f519e94` source audit | passed | The former ad hoc non-streaming nonce loop is now one deterministic OpenAI chat-stream driver for both engines. A real launch barrier, fixed prompt multiset, explicit sampling/template fields, strict SSE/usage/finish validation, fixed-length and uniform-input gates, R7 TTFT/ITL/E2E percentiles, request/token throughput, SLO-goodput, DRM peak memory, optional Kiln phase/width deltas, exact output comparison, strict JSON, clean-source/runtime identity, self-hashing atomic receipts, and overwrite refusal share one contract. The adjacent vLLM source confirms every request extension used by the driver. Documentation marks old nonce tables historical. This is harness acceptance, not hardware performance evidence: real ROCm rows, vLLM execution, concurrency 128, sampled/long/prefix/mixed workloads, and external runtime/binary provenance remain open. The earlier pre-commit evidence draft omitted the then-untracked new test file from the tracked-tree hash; this row records the committed source hash instead. |
| 2026-07-10 | ROCm equal-shape scaling counterexample | `sha256:61197fed3078` | `db68b153dc62` | Strix Halo ROCm/gfx1151 | `benchmarks/receipts/rocm/strix-halo/20260711t040223714292z-rocm-strix-halo-serving-equal-shape-db68b153-v1.json` | passed diagnostic | The clean exact-binary run completed every fixed 146-token prompt and fixed 64-token output with zero request or batching errors at concurrency 1/8/16/32/64. Throughput rose only from 21.93 tok/s at c1 to 34.03 at c8, then plateaued at 32.29/32.13/31.34 tok/s; c64 p99 TTFT reached 123.9 s and SLO-goodput fell to 0.49 tok/s. The effective cap was eight, but observed width topped out at six for c8 and seven thereafter, while mean rows per decode forward stayed 2.40-2.61. At c64 the actor spent 104.6 s in 1,544 decode forwards and 26.0 s in 1,536 layer-bounded prefill forwards; admission itself took only 40 ms. Peak device memory was 39.12 GB, all 64 requests succeeded, and output/receipt hashes are intact. This proves the multi-row ROCm path works but does not sustain its cap; serialized layer-round-robin prefill and decode overlap stagger row readiness, and the short-tail priority tie policy is the next focused A/B before any width increase or kernel work. The receipt also exposed that the driver inherited `OPENAI_API_KEY` even for a local endpoint (`authentication_configured=true`); remove that secret-bearing default before the next official run. |
| 2026-07-10 | Explicit benchmark authentication provenance | `sha256:71a620eda808` | this commit | portable | 247 qualification tests including 12 focused harness tests | passed | The serving driver no longer inherits the generic `OPENAI_API_KEY` secret for arbitrary benchmark targets. Authenticated endpoints must use a mutually exclusive explicit value or named environment source; empty or missing sources fail preflight, while receipts record only `none`, `argument`, or `environment`. A regression proves a populated generic key cannot enter headers, and the CLI receipt round trip proves unauthenticated local runs report `authentication_configured=false`. Documentation prefers the named-environment form to keep credentials out of process listings. |
| 2026-07-10 | Keep equal prefills on round-robin service | `sha256:e6d987d7cf8f` | this commit | portable; ROCm/Vulkan compile gates | 802 server library tests, 247 qualification tests, two focused scheduler regressions, formatting, and portable/ROCm/Vulkan all-target checks | passed | The bounded short-tail lane now admits only work no larger than four chunks and strictly less than half the round-robin row's remaining work. Equal-shape cohorts therefore remain aligned under ordinary round-robin service instead of consuming every fourth priority opportunity and creating an artificial decode-readiness staircase. The existing 128-token-versus-1K mixed-tail test still proves bounded interactive acceleration and long-row progress, while a new five-row equal-work trace proves twenty dispatches stay exact round robin and the priority counter remains zero. This is a source checkpoint for the focused ROCm A/B, not a throughput claim; the clean exact-binary run must improve row formation without regressing the accepted mixed-load latency and pressure-window gates. |
| 2026-07-10 | Separate and validate detailed benchmark receipts | `sha256:1e79f87749f6` | this commit | portable + GitHub Actions contract | 248 qualification tests, every standard qualification receipt, the detailed ROCm receipt, Python compilation, and exact CI command reproduction | passed | CI run `29139245290` correctly rejected the first detailed serving artifact because it had been placed in the compact `receipt-v1` tree. Detailed `kiln.serving-benchmark.v1` artifacts now live under `benchmarks/receipts/`; CI discovers and validates that tree with the owning driver while continuing to validate every compact receipt with the standard validator. The serving validator enforces a closed top-level/run contract, strict JSON, finite counters and latency values, request/gate/verdict consistency, clean passed source, exact workload/run coverage, and the canonical self-hash. The driver also rechecks repository identity after measurement so a mid-run edit cannot produce false clean-source evidence. Documentation distinguishes both receipt families and their check-in paths. |
| 2026-07-10 | ROCm multi-row greedy reproducibility counterexample | `sha256:1e79f87749f6` | `9905f1c64d8e` | Strix Halo ROCm/gfx1151 | `benchmarks/receipts/rocm/strix-halo/20260711t043044767129z-rocm-strix-halo-serving-equal-shape-round-robin-9905f1c6-v1.json` | failed | The exact original workload, prompts, output lengths, sampling, sizes, and baseline receipt proved that keeping equal prefills round-robin improves performance: c8/c16/c32/c64 throughput moved from 34.03/32.29/32.13/31.34 to 37.70/34.89/32.88/32.25 tok/s, mean decode width from 2.56/2.40/2.60/2.61 to 3.45/3.03/2.74/2.77, and p99 TTFT from 6.22/24.37/57.56/123.89 to 3.98/20.14/55.85/119.90 seconds. All 121 requests succeeded with zero batching errors and c1 output matched, but every multi-row output-set hash differed from the baseline, so the fail-closed reference gate rejected the run. Same-binary c8 repeats also changed substantive continuations for identical prompts; disabling prefix caching did not help, while effective decode width one produced 0/8 divergences across two repeats at 21.62 tok/s. This isolates scheduler-dependent BF16 output variation to multi-row decode and blocks treating the throughput win as correctness acceptance. |
| 2026-07-10 | ROCm deterministic serving repeatability | `sha256:d07c3e4d3ab0` | `f4a662b2a362` | Strix Halo ROCm/gfx1151 | `benchmarks/receipts/rocm/strix-halo/20260711t045708958307z-rocm-strix-halo-deterministic-reference-f4a662b2-v1.json`; `benchmarks/receipts/rocm/strix-halo/20260711t045953182327z-rocm-strix-halo-deterministic-restart-f4a662b2-v1.json` | passed | The clean release binary `sha256:55243ebb8202` started with only `KILN_DETERMINISTIC=1`; startup and `/health` reported effective decode width one despite the wider ROCm performance policy. Every measured run reported mean, maximum observed, and effective decode width exactly one with zero request or batching errors. The fixed c1/c8 prompts, 64-token outputs, sampling fields, and workload fingerprint produced identical output-set hashes before and after a fully drained process restart, and the strict reference comparison reported zero mismatches. C1/c8 throughput was 21.65/21.86 tok/s in the reference and 21.86/21.89 tok/s after restart; c8 p99 TTFT was 20.62/20.60 seconds, making the reproducibility-versus-queueing tradeoff explicit rather than presenting this as the performance configuration. Both shutdowns removed their full private snapshots without force or residue. All portable gates and ordinary-push CI run `29140299408`, including UI smoke and both receipt validators, passed; no accelerator CI was used. |
| 2026-07-10 | ROCm mutable-remainder priority regression | `sha256:d07c3e4d3ab0` | `3a9d40b5145f` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t050149904346z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | failed | The full stable mixed workload rejected the equal-prefill half-remainder heuristic. It preserved the pause fix: p99/p99.9 ITL were 80.5/141.6 ms, prefill/decode maxima were 36.2/87.1 ms, and attributed plus unexplained outliers remained zero. All ten requests succeeded, the only batching error was the intentional 2.003-second stalled-client eviction, and synchronization, mutation, shutdown, and snapshot-residue gates were clean. However, bounded short-prefill dispatches fell from the accepted 171 to 114; the 238-token pressure peer's TTFT regressed from 16.6 to 21.1 seconds and it missed the attributed pressure window. Throughput rose only from 12.48 to 12.96 tok/s, which cannot justify the correctness regression. Immutable admission-time prompt work, rather than a mutable remainder ratio, must distinguish equal cohorts from genuinely shorter late arrivals. |
| 2026-07-10 | Admission-work short-prefill classes | `sha256:d7e454b92440` | `f74a3d52b9a3` | portable; ROCm/Vulkan compile gates | 803 server library tests, 248 qualification tests, focused equal/shorter-work scheduler regressions, every receipt validator, formatting, and portable/ROCm/Vulkan all-target checks | passed | Each active prefill now retains its immutable admission-time work class. The first source checkpoint required that class to be strictly smaller than the current round-robin row while still enforcing the four-chunk remaining-work ceiling. Equal 128-token cohorts therefore used zero priority selections across twenty dispatches, while a focused 238-versus-432 case proved the late shorter row was selected when the longer row owned the cursor. A separate lightweight `Qualification contract` workflow now triggers on qualification tooling, workloads, and both receipt trees, closing the path-filter hole that skipped validation for the preceding receipt-only commit without making those commits rebuild the Rust workspace or use accelerator CI. The following hardware receipt shows why comparison against only the cursor was still insufficient. |
| 2026-07-10 | ROCm shorter-cursor priority counterexample | `sha256:d7e454b92440` | `f74a3d52b9a3` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t051221221574z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | failed | Immutable work identity alone did not restore the pressure contract: the run again recorded only 114 priority dispatches, the pressure peer's TTFT regressed further to 22.63 seconds, and it missed the pressure window. The reason is structural: when the genuinely shorter row itself owns the ordinary round-robin cursor, comparing its class to that same row can never mark the bounded opportunity as priority service, so the cursor advances instead of preserving the other 75% lane. P99/p99.9 ITL remained 100.6/143.7 ms with zero attributed or unexplained outliers, all ten requests succeeded, the intentional eviction was the sole batching error, throughput reached 13.43 tok/s, and synchronization, mutation, shutdown, and residue gates stayed clean. Priority eligibility must compare the candidate to the largest eligible active work class, while all-equal cohorts still decline the lane. |
| 2026-07-10 | Largest-class short-prefill eligibility | `sha256:c91b72c227d7` | `bf021bc32533` | portable; ROCm/Vulkan compile gates | 803 server library tests, 248 qualification tests, focused shorter-cursor/equal-cohort regressions, every receipt validator, formatting, and portable/ROCm/Vulkan all-target checks | passed | The bounded lane now asks whether a candidate's immutable admission work is smaller than any eligible active class, rather than comparing only with the ordinary cursor. A 238-token row that already owns that cursor is therefore still marked as the one-in-four priority dispatch beside a 432-token row, preserving the other 75% round-robin cursor for its normal turn. An all-equal 128-token cohort has no larger class and still consumes zero priority opportunities across twenty dispatches. The next exact-source ROCm mixed workload remains the acceptance gate; this row is not hardware evidence. |
| 2026-07-10 | Stable ROCm equal-cohort and mixed-load acceptance | `sha256:c91b72c227d7` | `bf021bc32533` | Strix Halo ROCm/gfx1151 | `qualification/receipts/rocm/strix-halo/20260711t051837168473z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json` | passed | The exact-source run recovered the pressure contract while retaining the equal-cohort policy. It exercised 163 bounded priority dispatches; the 238-token peer reached its first token in 16.59 seconds and emitted actor-ready progress before, inside, and after the attributed 2.002-second stalled-client window. All ten measured requests succeeded, and the intentional eviction was the sole batching error. Across 1,536 prefill forwards, prefill/decode maxima were 36.4/88.1 ms, p99/p99.9 ITL were 79.3/141.0 ms, and attributed plus unexplained outliers remained zero. Throughput was 12.58 tok/s with mean/max decode width 1.085/2, matching the prior mixed-load envelope rather than treating the focused equal-shape improvement as a large-batch claim. All 2,466 external-yield synchronizations succeeded in 31.8 ms total with a 0.38 ms maximum; response-queue p99.9 was 0.29 ms; KV stayed at 4,096 blocks with no graph, resize, or reclaim activity; unforced shutdown returned zero in 364 ms and left no snapshot residue. |
| 2026-07-10 | Typed deterministic and decode-width configuration | `sha256:59185a815a1a` | this commit | portable + ROCm/Vulkan compile + real Strix Halo ROCm process | 808 server, 373 model, 5 focused tensor, 55 capability-contract, and 248 qualification tests; every receipt validator; strict startup failures; real health/config/request smoke | passed | `server.deterministic` and `server.max_decode_batch` are first-class TOML settings with strict environment precedence, primitive ergonomic syntax, source attribution, and bounded validation. Startup fixes the tensor determinism selector exactly once and rejects a conflicting late change. One resolved hard ceiling now bounds actor admission, model decode-buffer allocation, resident-pool warmup, and the fallback rendezvous worker; none rereads the shared decode-width environment variable. Resolution preserves the configured and backend-policy widths while reporting the final authority (`backend_policy`, file, environment, `deterministic`, or `max_batch_tokens`) through startup logs, `/v1/config`, health, and debug. Final ROCm processes rejected `KILN_DETERMINISTIC=sometimes` and `KILN_MAX_DECODE_BATCH=0` with exit 1, variable name, and invalid value. A real 4.206B-parameter ROCm server started from TOML with deterministic=true and configured width 16; startup and both API surfaces reported backend policy 8 and effective width 1 from `deterministic`, a greedy chat request returned the expected `kiln` response with maximum observed width one and zero batching errors, and drained shutdown removed the private snapshot. Documentation now states the narrower proven serving-repeatability contract instead of falsely implying that this selector already makes every accelerator kernel or training path bitwise deterministic. This is a scoped configuration migration, not completion of the repo-wide malformed-configuration or Phase 8.1 gates. |
| 2026-07-10 | Fail-closed default thinking-budget environment | `sha256:4ade1c918acf` | this commit | portable + ROCm/Vulkan compile + real process startup | 810 server library tests; strict grammar/load regressions; portable, ROCm, and Vulkan all-target checks; process exit checks; formatting and diff hygiene | passed | `KILN_DEFAULT_THINKING_BUDGET_TOKENS` and `KILN_DEFAULT_THINKING_BUDGET_MS` now resolve on the fallible startup path instead of warning and retaining a TOML/default value. Their entire documented grammar is a non-negative base-10 integer or case-insensitive `unlimited`; zero still means immediate close. Empty strings, the old undocumented `off`/`none`/`null` aliases, negatives, decimals, unit-suffixed values, overflow, and non-Unicode values fail closed. Real binaries rejected `"64 tokens"`, `"2.5"`, and `"off"` with exit one plus the exact variable and invalid value before model or listener initialization. |
| 2026-07-10 | Pre-validation structured diagnostics | `sha256:e0487aa8e173` | this commit | portable + ROCm/Vulkan compile + real process failures | 811 server library and 7 server binary tests; bootstrap parser regression; portable, ROCm, and Vulkan all-target checks; serve/config process checks; formatting and diff hygiene | passed | The CLI now resolves only string-valued `[logging]` fields plus env/verbosity precedence and installs tracing before any command can load the full configuration. The bootstrap parser remains field-tolerant when another typed field is invalid and falls back safely when TOML syntax or file I/O prevents extraction; authoritative `KilnConfig` parsing is unchanged. Every `KilnConfig::load` error emits one `configuration_load_failed` event containing the selected path and full error chain. Real `serve` and `config` processes emitted valid JSON diagnostics for malformed environment values and a TOML `u16` type failure before exiting one, while retaining the terminal error fallback. |
| 2026-07-10 | Thinking-close compatibility preflight | `sha256:2cc02211435e` | this commit | portable + ROCm/Vulkan compile + Qwen-template HTTP smoke | 812 server library tests; focused overlap/capacity regressions; portable, ROCm, and Vulkan all-target checks; mock HTTP 400; formatting and diff hygiene | passed | Budget setup now reports the exact offending `stop` string when it can match, contain, or overlap any forced `</think>` boundary. A shared preflight validates the effective completion capacity after context-window clamping against the tokenizer-specific close-token count before decode; synthetic coverage proves equal capacity is accepted and insufficient capacity fails with both counts. Operator docs distinguish invalid insufficient capacity from a valid budget larger than the completion cap, where the controller reserves close slots and reports trigger `max_tokens`. A real mock server using the local Qwen tokenizer/template rejected `stop=["</think>\n"]` with HTTP 400 before generation, while the fallback non-reasoning template correctly left the budget inert. |
| 2026-07-10 | Strict browser and desktop budget parsing | `sha256:9f4a57f6e668` | this commit | browser + desktop portable surfaces | four-scenario headless Chromium server smoke; executable desktop UI contract; 812 server library and 11 desktop settings tests; portable/ROCm/Vulkan all-target checks; JavaScript syntax, formatting, and diff checks | passed | Both surfaces accept only base-10 whole token counts within their representable bounds. Decimal seconds are decomposed as strings and combined with integer arithmetic, so up to three fractional digits become exact milliseconds without floating-point rounding. The readers reject exponent, sign, decimal-token, excess-precision, overflow, and native number-input `badInput` states before blank can mean an unused limit. Browser integration proves malformed custom values focus the offending field, show an error, issue no completion request, and cannot overwrite the last valid persisted budget; valid zero tokens plus 1.25 seconds still serialize as `0` and `1250`, while Unlimited remains explicit null. The inexpensive UI workflow now runs the desktop contract whenever its UI changes; no accelerator CI was added. |
| 2026-07-10 | Durable thinking-budget stream parity | `sha256:bd0fb18f7414` | this commit | portable request-log path + ROCm/Vulkan compile gates | 813 server library tests; 8 focused request-log unit tests; 4 on-disk capture integration tests; portable/ROCm/Vulkan all-target checks; formatting and diff hygiene | passed | SSE reassembly now retains the finish chunk's complete metadata object and reconstructs `choices[0].thinking_budget` only when final outcome telemetry is present. A parity matrix compares streamed and non-streaming durable shapes for token-triggered close, natural close, partial forced close, and configured-but-inert budgets across message, finish reason, outcome, usage, and effective limit/source metadata. The middleware integration consumes a real SSE body, flushes the asynchronous logger, and verifies the persisted JSONL row. Streams interrupted before finish metadata remain marked interrupted without a fabricated outcome. Public request-log documentation records the shared mining paths. |
| 2026-07-10 | Batch thinking-budget configuration provenance | `sha256:f5b0d27f2db3` | this commit | portable API + ROCm/Vulkan compile gates | 814 server library tests; focused resolution, cache-rehydration, and HTTP serialization regressions; real Chromium docs smoke; portable/ROCm/Vulkan all-target checks; formatting and diff hygiene | passed | Chat and batch now share one effective-budget resolver. Every batch response path, including zero-token and all deterministic-cache rehydration paths, reports the request-wide resolved pair at `metadata.thinking_budget` with independent `request`, `server_default`, `request_unlimited`, or `unlimited` provenance. Per-completion outcomes remain at `completions[].thinking_budget`, avoiding a false aggregate trigger/close state for heterogeneous rows. Cache hits preserve original outcomes while recomputing provenance from the current request. Public API, quickstart, and site documentation define the split explicitly. |
| 2026-07-10 | Playground compare stream and outcome parity | `sha256:919dea80cf4d` | this commit | browser UI + portable/ROCm/Vulkan compile gates | four-scenario real Chromium UI smoke; normal/compare budget and failure fixtures; 814 server library tests; portable/ROCm/Vulkan all-target checks; JavaScript syntax, formatting, and diff hygiene | passed | Normal chat and both compare sides now use one incremental UTF-8/SSE consumer and one chunk-state accumulator for reasoning, content, TTFT, finish reason, and final thinking-budget metadata. Both views use the same budget/finish outcome formatter; a browser regression proves independent natural-close versus token-cap/truncated results on concurrent sides. Structured `generation_error` events fail only their side and remain visible instead of becoming successful empty output, premature EOF fails closed, and every terminal/error path cancels its reader. |
| 2026-07-10 | Bounded thinking-budget telemetry substrate | `sha256:dad6bcaef5b5` | this commit | portable + ROCm/Vulkan compile gates | 816 server library tests; focused recent-request serialization and Prometheus cardinality/outcome regressions; portable, ROCm, and Vulkan all-target checks; formatting and diff hygiene | passed | Recent requests now have a typed optional budget record that distinguishes unresolved applicability, inert configuration, natural close, forced close, and partial outcomes without fabricating absent state. The central recorder can publish fixed-source counters, a closed outcome counter, and numeric token/time histograms; hostile source text is collapsed to `unknown` and never becomes a label. This is deliberately an infrastructure checkpoint, not Phase 2.3 completion: existing generation, cache, stream, and error paths still need to populate the record, and the dashboard and public documentation still need to expose it. |
| 2026-07-10 | End-to-end thinking-budget request observability | `sha256:43a158aeafc5` | this commit | browser + portable/ROCm/Vulkan compile gates | 819 server library tests; focused effective/inert, cache, streaming-setup failure, metrics, and recent-request serialization regressions; four-scenario real Chromium UI smoke; portable, ROCm, and Vulkan all-target checks; JavaScript syntax, formatting, and diff hygiene | passed | Every chat record path now carries effective token/time limits, independent source provenance, applicability, and the final shared runtime status when one exists. Unary API metadata and recent records use the same captured status; cache hits retain their original outcome; live streams snapshot status on success, timeout, generation error, or client disconnect; setup failures preserve established applicability without inventing token counts. The central recorder publishes one bounded-label outcome and finite-limit histogram observation per physical record, while multi-choice records deliberately use the same first/root outcome convention as response metadata. The request drill renders configuration, provenance, application state, outcome, and measured thinking without adding list-row clutter, and additive old records remain renderable. Public API and operator documentation define field absence, closed label sets, and histogram cardinality safety. |
| 2026-07-10 | Shared Rust thinking-budget configuration model | `sha256:954bf357dd4b` | this commit | portable + ROCm/Vulkan compile gates | 86 core, 215 eval, 819 server, and 4 `kiln-eval` binary tests; focused override/provenance matrices; portable, ROCm, and Vulkan all-target checks; formatting and diff hygiene | passed | `kiln-core` now owns omitted/inherit, explicit unlimited, finite-limit, independent token/time resolution, explicit CLI value, and typed provenance semantics. Chat, batch, eval suite/example/run overrides, rollout/eval CLI flags, and cache policy all consume that implementation. The former server `BudgetOverride`, eval `EvalBudgetOverride`, CLI parser/serde implementation, and eval source-string resolver were removed; the two public names remain compatibility aliases. This is a bounded Phase 2.1 checkpoint, not checkbox completion: runtime outcome records and the browser/desktop schema still need to converge on the canonical model. |
| 2026-07-10 | Canonical thinking-budget terminal outcome | `sha256:40ac51b6873a` | this commit | portable + ROCm/Vulkan compile gates | 87 core, 215 eval, and 819 server library tests; eval all-target check; focused API/cache outcome regressions; portable, ROCm, and Vulkan server all-target checks; formatting and diff hygiene | passed | One `kiln-core` outcome now defines the terminal `triggered`, typed trigger, closure state, measured reasoning tokens, and elapsed milliseconds serialized by chat choices, batch items, eval results, and their durable consumers. It is constructed directly from the shared runtime status, and deserialization fails when `triggered` disagrees with trigger presence. The duplicate server wire struct and eval outcome struct were removed; `EvalThinkingBudgetOutcome` remains a compatibility alias. Phase 2.1 remains open for the effective/applicability record and browser/desktop schema. |
| 2026-07-10 | Typed thinking-budget provenance | `sha256:55818f62335b` | this commit | portable + ROCm/Vulkan compile gates | 88 core, 215 eval, and 819 server library tests; eval all-target check; focused unknown-source/cardinality regressions; portable, ROCm, and Vulkan server all-target checks; formatting and diff hygiene | passed | API metadata, batch metadata, eval results, recent requests, and Prometheus observation now carry the shared provenance enum instead of converting to arbitrary strings between layers. JSON remains the same stable snake-case vocabulary. Deserialization maps unknown legacy/external text to the explicit `unknown` variant, and the metrics label table is enum-backed, so hostile text cannot become a label. Phase 2.1 remains open for the combined effective/applicability record and browser/desktop contract. |
| 2026-07-10 | Canonical effective thinking-budget record | `sha256:26b42c95429f` | this commit | portable + ROCm/Vulkan compile gates | 90 core, 215 eval, and 819 server library tests; eval all-target check; flat/legacy-nested schema and invalid-state regressions; portable, ROCm, and Vulkan server all-target checks; formatting and diff hygiene | passed | Chat metadata and eval results now use one validated `kiln-core` record for effective limits, typed sources, applicability, and the optional canonical outcome. Serialization preserves the established flat chat shape and rejects inconsistent configured/limit, source/limit, applied/configured, or outcome/applied states. Deserialization accepts both that shape and the earlier nested eval-only outcome for backward compatibility; new eval results write the shared flat form documented in the eval guide. The duplicate server and eval record structs were removed behind compatibility aliases. Recent requests retain a small conversion wrapper only for their legitimate unresolved-applicability state. Phase 2.1 remains open for a machine-readable cross-runtime reference and browser/desktop conformance. |
| 2026-07-10 | Versioned thinking-budget contract foundation | `sha256:f44b59dece39` | this commit | portable contract/core | 91 core tests; JSON syntax, formatting, and diff hygiene | passed | A Draft 2020-12 schema now defines request overrides, effective configuration, sources, terminal outcomes, application records, and unresolved recent records. A versioned conformance file covers independent dimensions, zero, inherit, explicit unlimited, request/suite/run/example provenance, flat terminal records, legacy nested eval input, and invalid states. The core suite executes every vector and cross-checks the schema's complete source and trigger vocabularies. A concise human reference links the normative artifacts from the README. This is a foundation checkpoint: Phase 2.1's schema/reference item remains open until browser and desktop runners execute the same vectors. |
| 2026-07-10 | Desktop thinking-budget contract vectors | `sha256:a6631d186c4f` | this commit | desktop portable UI | executable desktop VM smoke plus core v1 contract test; JSON syntax and diff hygiene | passed | The v1 contract now includes server-default vectors for unlimited, token-only, time-only, and dual finite limits. The desktop smoke extracts and runs the production settings parser against every vector, including exact fractional-seconds-to-milliseconds conversion and zero preservation. This closes desktop's cross-runtime portion without adding accelerator CI; browser request vectors remain before the Phase 2.1 schema/reference item can close. |
| 2026-07-10 | Independent Playground thinking-budget controls | `sha256:4c92ea3d97f3` | this commit | real Chromium + portable/ROCm/Vulkan compile gates | five shared request-contract vectors in four-scenario Chromium smoke; 390 px responsive regression; desktop and core contract runners; 819 server library tests; portable, ROCm, and Vulkan all-target checks; JavaScript syntax, formatting, and diff hygiene | passed | Token and time now have separate Inherit, Unlimited, and Limit selectors, so each request dimension independently omits its field, sends explicit null, or sends a validated finite value. Production parsing and request serialization execute every request-scope vector from the same v1 conformance file used by Rust and desktop. Finite fields appear only for their selected dimension, zero and exact fractional milliseconds remain intact, thinking-off preserves but disables the chosen controls, and the 390 px layout has no horizontal overflow. Existing combined `server`/`unlimited`/`custom` local-storage settings migrate dimension by dimension without losing prior values. The shared typed implementation and independent-control checklist items close here; the canonical-documentation item remains open until cheap CI enforces public prose linkage. |
| 2026-07-10 | Canonical thinking-budget documentation guard | `sha256:80e6c38166ba` | this commit | portable docs/contract CI | schema-derived reference vocabulary; required public-document links; real Chromium docs and server UI smoke; desktop and core contract runners; release-version validation; 10 source-identity tests; JSON, JavaScript, YAML, and diff hygiene | passed | The versioned schema and conformance file now drive an executable documentation check for contract version, independent override states, every source, and every trigger. README, Quickstart, eval, and site API explanations must link the canonical reference; contract edits trigger the existing browser/desktop smoke, while prose drift uses the existing lightweight version-drift job rather than launching unrelated GPU work. Pages publishes both JSON artifacts at the schema's declared URL. Qualification source identity now includes `contracts/`, with a regression proving a contract edit changes the hash; prior behavior could leave receipts looking source-identical after a semantic schema change. This closes Phase 2.1. |
| 2026-07-10 | Playground effective thinking-budget preview | `sha256:c183d5a3a8e1` | this commit | real Chromium + portable/ROCm/Vulkan compile gates | four-scenario Chromium UI smoke with config failure/retry and 390 px maximum-value coverage; 819 server library tests; focused config API regression; docs-site smoke; portable, ROCm, and Vulkan all-target checks; visual desktop/mobile review; JavaScript syntax, formatting, and diff hygiene | passed | Playground and Runtime Config now share one cached `GET /v1/config` request. Before send, a quiet status row resolves each independent dimension to its finite value or unlimited state and labels it `server` or `request`; finite edits, explicit unlimited, inheritance, malformed values, and thinking-off state update immediately. Missing or invalid generation defaults remain visibly unavailable rather than being assumed unlimited, with a local retry that recovers after API failure. The toolbar groups each label with its selector, the demo fixture exposes the real config shape, and maximum token/time labels remain overflow-free at 390 px. Public README, Quickstart, canonical contract, and API documentation identify the preview and its source endpoint. This closes the effective-default preview item in Phase 2.4. |
| 2026-07-10 | Fail-closed rollout thinking-budget applicability | `sha256:a2dd2d06dbec` | this commit | portable CLI + ROCm/Vulkan compile gates | five shared request-scope conformance vectors; focused flag/template and no-side-effect regressions; real CLI help/error smoke; 823 server library tests; contract documentation guard; portable, ROCm, and Vulkan all-target checks; formatting and diff hygiene | passed | `kiln rollout-generate` now rejects finite, zero, and explicit-unlimited budget flags whenever thinking is disabled, before reading inputs, creating output, invoking the scorer, or contacting a server. Top-level template budget fields, including JSON null, are rejected after parsing but before output or network activity, and the request renderer enforces the same invariant defensively for internal callers. With `--thinking true`, the rollout renderer executes the canonical request-scope Inherit/Unlimited/Limit matrix and preserves each dimension independently. Long help, option help, the actionable runtime error, and the canonical contract all state the requirement. This closes the rollout applicability item in Phase 2.4. |
| 2026-07-10 | Shared `8420` runtime default | `sha256:19b7d0652709` | this commit | portable desktop/CLI + local ROCm/Vulkan compile gates | versioned cross-surface contract and drift guard; 824 server library tests; 149 desktop tests; 5 eval binary tests; real `kiln config`, `kiln health --help`, and `kiln-eval --help` smoke; desktop VM smoke; portable, ROCm, and Vulkan all-target checks; release/docs guard; shell syntax, formatting, and diff hygiene | passed | A versioned runtime-defaults contract now anchors the server bind host, local client host, and port. Server configuration and every CLI URL default share one Rust definition; desktop Rust and JavaScript each have one guarded default consumed by settings, supervision, and onboarding. Fresh installs now use `127.0.0.1:8420`, while explicitly persisted ports remain user overrides. README, Quickstart, desktop guidance, example configuration, and the Phase 2 validation script agree on `8420`; the validation script's broader hardware workflow was not exercised in this checkpoint. The existing lightweight release-drift job enforces the contract without adding GPU CI. This closes the shared default-port item in Phase 2.4. |
| 2026-07-10 | Durable desktop settings recovery | `sha256:dddae63404a7` | this commit | portable Linux desktop + real Chromium | 159 locked desktop tests including 22 focused settings cases; field-level corruption, invalid JSON, legacy migration, future-schema, backup recovery, semantic validation, private Unix permissions, exact malformed-byte preservation, and injected promotion-failure coverage; Clippy; executable desktop VM smoke; real Chromium docs smoke and 520/390 px settings probes; JavaScript/YAML syntax, release/default/contract guards, formatting, and diff hygiene | passed | `settings.json` now carries schema version 1 and is decoded field by field, so one malformed value preserves every other valid field. Writes are process-serialized, staged beside the destination with mode `0600` on Unix, fsynced, and promoted only after the prior supported document moves to `settings.json.bak`; malformed primary bytes move to `settings.json.invalid`, and an injected promotion failure proves rollback and temporary-file cleanup. Missing or corrupt primary data recovers from backup, while any damage suppresses automatic server launch and OS autolaunch reconciliation until an explicit repair Save. The app logs and notifies, Settings keeps a structured warning visible and Save-enabled for repair, and a newer schema is read-only. Unrelated binary installation cannot clear an unresolved warning. Port and inference-fraction inputs now reject coercion, and the constrained layout has no horizontal overflow. The existing lightweight UI job gains source-contract triggers without adding a hosted Rust build or accelerator work. This closes the final Phase 2.4 checklist item. |
| 2026-07-10 | Distinct GRPO behavior and KL loss inputs | `sha256:e2c48fcb7ff3` | this commit | portable CPU; ROCm/Vulkan feature builds | exact scalar loss and analytic coefficient fixtures with policy, behavior, and frozen-reference log-probabilities all distinct; portable, ROCm, and Vulkan checks; formatting and diff hygiene | passed | PPO token/sequence/CISPO ratios now consume only behavior-policy log-probabilities, while K1/K3 consume only the frozen reference through scalar, tape, host-analytic, and CUDA/ROCm device-coefficient routes. The current one-reference Vulkan fused ABI declines distinct tensors and falls back to the separated generic route. Production still passes its historical reference into both inputs until versioned rollout provenance and independent policy selection land, so the broader semantics and ingestion items remain open. These feature runs used CPU fixtures and are not ROCm/Vulkan hardware evidence; CUDA could not compile locally because `nvcc` is unavailable. |
| 2026-07-10 | Strict rollout provenance v1 foundation | `sha256:e78713870ae2` | this commit | portable | 22 trajectory/provenance compatibility and adversarial tests; train and server all-target checks; formatting and diff hygiene | passed | `ScoredRollout` can now carry a validated `kiln.rollout-provenance.v1` record with exact full input IDs, prompt boundary, ordered sampled/forced action decisions, selected-token behavior log-probabilities, base/adapter content identity, tokenizer/config/template hashes, effective sampling and thinking-budget controls, seed, and generation backend. Unknown fields, unsupported versions, malformed hashes, non-finite or positive log-probabilities, index/token drift, unproven forced tokens, and invalid sampling ranges fail during deserialization. Legacy rollouts remain readable without provenance. This is a type/parser foundation only: the API/CLI do not emit it and the trainer does not consume it yet, so the Phase 3.1 rollout-schema and ingestion items remain open. |
| 2026-07-09 | First reduced-CI measurement | `sha256:cda13f3f84e5` | `3c71cc4002f8` | GitHub Actions | run `29049575526` | passed | Three active jobs completed in 3m52s wall and about 4m36s aggregate time; all GPU backend jobs were skipped |

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
