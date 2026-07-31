# Architecture deep dive

Kiln is one Rust process that serves Qwen3.5-4B, trains LoRA adapters,
evaluates them, and manages the resulting artifacts. This page is for
contributors and operators who need to understand the boundaries between the
HTTP service, scheduler, model runtime, accelerator backends, training system,
and durable state.

If you want to run Kiln rather than study its internals, begin with the
[Quickstart](QUICKSTART.md). For exact request and response shapes, use the
[HTTP API reference](https://ericflo.github.io/kiln/docs/http-api/).

## System at a glance

```text
clients and dashboard
        │
        ▼
HTTP validation, policy, and identity
        │
        ├──────────────► OpenEnv HTTP discovery + stateful WS episodes
        │                         │
        │                         └──► canonical grouped trajectories
        │
        ├──────────────► durable datasets, suites, jobs, and receipts
        │
        ▼
scheduler and accelerator ownership
        │
        ▼
model runner and tensor/autograd substrate
        │
        ▼
CUDA · ROCm · Metal · Vulkan · CPU
```

The same process owns four related workflows:

| Workflow | Primary responsibility | Durable result |
|---|---|---|
| Serving | Validate, schedule, and generate responses | Request logs and lineage, when enabled |
| Training | Run admitted SFT, GRPO, or OPD jobs | Checkpoints and LoRA adapters |
| OpenEnv RL | Discover environments, execute grouped episodes, and aggregate environment rewards | Canonical trajectory JSONL and rollout receipts |
| Evaluation | Run suites, comparisons, judgments, and strict replay | Eval outcomes and replay records |
| Artifact management | Validate, load, merge, export, and remove adapters | Manifests, receipts, and archives |

These workflows share model identity, memory accounting, adapter state, and
accelerator ownership. They do not communicate through Python workers or a
second model server.

## Package boundaries

Kiln is a Cargo workspace. The exact crate list changes as backend work moves,
so the useful boundary is responsibility rather than a fixed crate count:

| Layer | Representative crates | Owns |
|---|---|---|
| Product runtime | `kiln-server`, `kiln-model`, `kiln-scheduler` | HTTP, admission, batching, generation, and process state |
| Environment runtime | `kiln-openenv` | Bounded OpenEnv discovery, typed wire values, stateful WebSocket sessions, and environment identity |
| Learning loop | `kiln-train`, `kiln-eval` | Training, scoring, replay, and promotion evidence |
| Tensor substrate | `kiln-tensor`, `kiln-autograd`, `kiln-param`, `kiln-optim` | Storage, operations, gradients, parameters, and optimizer steps |
| Resource control | `kiln-memory`, `kiln-resource`, `kiln-graph` | Capacity, lifetimes, and replay plans |
| Backend integration | `kiln-blas`, `kiln-hip`, `kiln-rocblas`, `kiln-mps`, `kiln-vulkan-blas` | Accelerator APIs and backend-specific dispatch |
| Kernel libraries | `kiln-flash-attn`, `kiln-gdn-kernel`, `kiln-rmsnorm-kernel`, `kiln-vulkan-kernel`, and peers | Fused or device-specific operations |

Feature flags compile only the backend stack requested by a build. Product
code talks to focused backend traits for attention, linear algebra, sampling,
residency, training loss, optimizer steps, and replay. Backend implementations
may select different kernels, but they must preserve the same public request
contract.

The generated
[backend capability report](docs/backend-capability-report.md) is the
source-tree inventory for current trait coverage and known gaps. It is an
engineering report, not a promise that every route is equally fast or
qualified on every device.

## Startup establishes the runtime contract

Startup is the boundary where mutable inputs become immutable runtime policy:

1. Parse command-line options and merge defaults, `kiln.toml`, and canonical
   `KILN_<SECTION>_<FIELD>` overrides.
2. Validate the complete configuration, including cross-field constraints.
3. Resolve the serving profile and backend-specific execution policies.
4. Select a physical device and verify that its memory probe refers to the
   same accelerator.
5. Inspect device capabilities and install immutable kernel-selection policy.
6. Read and verify the model checkpoint, then upload or bind its weights.
7. Size the memory governor, KV cache, decode buffers, and other resident
   resources.
8. Precompile or prewarm routes required before readiness.
9. Start the batching, training, and evaluation workers.
10. Bind the HTTP server and publish readiness.

Runtime code consumes the resolved objects. It does not repeatedly reinterpret
environment variables in individual layers or kernels. A policy change that
affects memory ownership, kernel selection, graph capture, or serving behavior
requires a restart.

The effective values and their provenance are available from `GET /v1/config`;
readiness and live state are available from `GET /health`.

## Serving path

An OpenAI-compatible chat request passes through these ownership boundaries:

```text
POST /v1/chat/completions
        │
        ▼
schema validation and serving-profile admission
        │
        ▼
chat-template rendering and tokenization
        │
        ▼
model, adapter, sampling, and cache identity
        │
        ▼
batching actor
        │
        ├── prefill prompt chunks
        ├── decode active rows
        └── settle cancellation or completion
        │
        ▼
JSON response or server-sent events
```

The HTTP layer owns wire compatibility, validation, and error responses. The
batching actor owns admission, request progress, cancellation, and terminal
cleanup. `ModelRunner` owns model execution, adapter application, sampling,
backend health, and accelerator-resident state.

Streaming changes response delivery, not model ownership. Generated events
travel through a bounded channel to the async response task. Cancellation,
timeout, disconnect, and panic paths must still establish whether accelerator
work completed before buffers or cache blocks can be reused.

### Continuous batching

The real-model batching actor uses a token budget for each scheduling cycle.
The default combined budget is 512 tokens, but the resolved configuration is
authoritative. Within a cycle, the actor prioritizes:

1. one decode step for each ready row;
2. resumable prompt prefills already in progress;
3. newly admitted prefills that fit the remaining budget.

Long prompts can therefore be split across cycles without preventing ready
decode rows from progressing. Partial prefills rotate rather than allowing one
long prompt to monopolize every cycle.

The actor assembles compatible rows into backend-supported batches. Effective
decode width, prefill quantum, shape constraints, and route support come from
the resolved backend policy. A narrow direct-stream worker remains as a
compatibility path when the actor is unavailable; it is not a second
production scheduler.

### Terminal cleanup is part of request lifetime

A request is not complete merely because its final token was sampled. Before
the actor removes the row, it must release or settle:

- graph or command-replay ownership;
- recurrent Gated DeltaNet state;
- prefix-cache leases;
- private KV blocks;
- response delivery and cancellation state.

Health counters keep a row active through this cleanup. A reported zero-active
boundary therefore means those resources have settled, not only that the row
left a scheduling list.

### Batch completions and thinking budgets

`POST /v1/completions/batch` admits a bounded set of prompts and returns one
aggregated response. Its rows still use the same scheduler, adapter identity,
memory accounting, and cancellation rules as ordinary generation.

Thinking budgets are request-local decode state. Omitted limits inherit server
defaults, JSON `null` means unlimited, and a nonnegative value is an explicit
limit. When a budget closes an open reasoning span, the controller emits a
tokenizer-validated closing sequence through the normal decode path before
answer generation continues. The
[inference schema](https://ericflo.github.io/kiln/docs/inference-schema/)
defines the exact wire fields.

## Model execution

Kiln's current model profile targets Qwen3.5-4B, a 32-layer hybrid model:

- 24 Gated DeltaNet (GDN) linear-attention layers maintain recurrent state;
- 8 grouped-query full-attention layers use a paged KV cache;
- LoRA deltas may be applied to supported projections;
- final normalization, the language-model head, penalties, and sampling
  produce the next token.

This hybrid shape matters operationally. Only the full-attention layers need
the ordinary K/V history, while GDN layers carry fixed-shape recurrent and
convolution state. Prefill chunking must preserve both kinds of state across
cycle boundaries.

### Prefill

Prefill renders the prompt into model state. Depending on the selected backend,
prompt length, and immutable streaming-prefill policy, the model runner may use
one materialized forward or a tiled forward. Tiling is an execution choice,
not a semantic choice: enabled and disabled cache paths must preserve the same
numerical prompt partition where a backend requires that invariant.

The full-attention layers write K/V entries. GDN layers update recurrent and
convolution state. The final prompt position produces the first decode
candidate.

### Decode

Decode advances each active row by one sampled token. The backend may use eager
execution or an eligible replay mechanism:

| Backend | Replay primitive |
|---|---|
| CUDA | CUDA graph |
| ROCm | HIP graph |
| Metal | indirect command buffer |
| Vulkan | resident command batch |

Replay is never assumed from the backend name alone. The serving profile,
request shape, resource identity, capture status, and backend support decide
whether a request is eligible. Failure to prove safe replay falls back only
where the route defines a contained fallback; an unknown completion state
quarantines the backend.

Speculative decoding implementations remain research and qualification
substrate. The serving policy accepts only effective `off`, and startup rejects
an effective non-off configuration before model weights are loaded.

## Memory and resident state

Kiln treats accelerator capacity as a shared budget, not as a collection of
unrelated allocations.

### Physical-device identity

Device selection, the tensor backend, and the operating-system memory probe
must resolve to the same physical accelerator. Startup fails if that identity
cannot be established. A configured memory value is a cap on detected
capacity, not permission to invent capacity that the device does not have.

The memory governor samples live pressure and accounts for model weights, KV
cache, training reservations, replay buffers, allocator pools, and other
tracked categories. Admission uses the same planning inputs that execution
uses, reducing estimate/runtime drift.

### Paged KV cache

The production block manager divides K/V storage into 64-token blocks. Each
request holds a logical block table that maps token positions to physical
slots. Blocks are allocated incrementally and returned only after request
cleanup proves they are no longer in use.

Paging avoids a separate maximum-context allocation for every request and
allows concurrent rows to share one bounded pool. The exact number of blocks
is resolved from the effective memory budget unless the operator supplies a
valid override.

Optional FP8 K/V storage uses half the bytes per element of BF16. That is a
capacity tradeoff, not a universal quality or speed guarantee; it remains
opt-in and must be evaluated for the deployment workload.

### Prefix cache

The production prefix cache reuses complete, block-aligned prompt state. Cache
identity includes the information needed to keep reused state consistent with
the active model, adapter, and generation contract. Retained blocks are
reference-counted and evicted within configured block, entry, and state-byte
budgets.

Requested and effective cache state are separate. In particular, Vulkan prefix
reuse is currently disabled by a source-level correctness quarantine.
`GET /v1/config` reports `vulkan_correctness_quarantine` when configuration
requests the cache but the selected execution route cannot safely use it.
Starting successfully does not override that restriction.

### Recurrent and replay state

GDN state, graph buffers, command batches, and resident activations have
explicit owners and generations. Reusing a resident object requires a matching
request identity, shape, device, allocation generation, and active adapter
where applicable. Adapter transitions invalidate state whose weight pointers
or numerical identity changed.

## Accelerator backends

Every backend has two separate obligations:

1. implement a correct route or decline it explicitly;
2. choose compatible, efficient kernels from the active device's capabilities.

Support predicates are request-aware. A backend can support an operation for
one dtype, layout, or shape and decline another. A declined optimized route may
use a defined reference implementation only when that fallback preserves the
operation's residency and performance contract. Missing hot-path support must
not silently become an unbounded host round trip.

The backend capability report distinguishes native support, constrained
support, portable fallback, and unavailable routes. Hardware qualification
adds evidence for a recorded build, driver, device, and workload; it does not
rewrite product dispatch.

### Vulkan policy is capability-driven

Vulkan runtime selection does **not** branch on a marketing device name,
qualification-machine identity, driver string, vendor ID, or device ID.
Those values may appear in diagnostic or benchmark receipts, but they are not
execution capabilities.

After selecting the physical device, Kiln reads limits and memory properties
such as:

- API version;
- workgroup counts, sizes, and invocation limits;
- shared-memory and push-constant limits;
- descriptor and storage-buffer limits;
- compute subgroup support;
- coherent device-local host-visible memory;
- cached host-visible staging memory.

`VulkanKernelPolicy::from_capabilities` derives compatible routes from those
facts. Individual dispatch sites add workload-dependent checks such as tensor
shape, storage range, and dispatch-grid bounds. The resulting policy is
installed once for the process; conflicting reinstallation fails.

This design has two consequences:

- a capable Vulkan device can select fast routes without appearing in a model
  or vendor allowlist;
- a device that lacks a shader requirement retains only compatible routes and
  explicit fallbacks.

Vulkan's current serving path may execute Vulkan kernels while retaining the
frozen model representation on CPU. The runtime therefore tracks execution
device and model-weight device separately. This is an explicit hybrid design,
not evidence that work silently left Vulkan.

The stable serving profile keeps Vulkan resident prefill disabled until its
release qualification closes. The experimental profile may admit that route,
subject to capability and request checks. This profile gate is about evidence
for a route; it is not a special case for one machine.

### CUDA, ROCm, and Metal

CUDA and ROCm expose backend-specific kernel and graph policies; Metal uses MPS,
Metal shaders, and indirect command buffers where supported. These backends
follow the same boundary as Vulkan: startup resolves policy, runtime queries
support for the actual request, and observability reports intended versus
effective behavior.

Backend-specific optimizations can have narrower eligibility than baseline
inference. Marlin W4A16, FP8 K/V storage, live graph capture, and specialized
GDN or attention routes are examples. Their presence in source does not imply
that stable serving activates them.

## Accelerator ownership and serving profiles

Inference, training, evaluation, adapter transitions, and physical memory
maintenance can touch the same accelerator state. A shared coordination lock
makes that ownership explicit:

- inference takes shared ownership;
- training and physical mutations take exclusive ownership;
- adapter swaps also cross the batching actor's quiescence barrier;
- GPU-backed eval work obeys the same lock rather than running through an
  independent device path.

The immutable serving profile decides which owners are admitted:

| Profile | Inference | Training and live mutation | Intended use |
|---|---|---|---|
| `stable` | Admitted | Rejected | Predictable ordinary serving |
| `experimental` | Admitted | Admitted with writer priority | Controlled development and qualification |
| `maintenance` | Disabled | Admitted after drain | Training, adapter transitions, and memory maintenance |

One process therefore does not mean inference and training execute on the GPU
at the same time. A queued training or eval job may exist while inference is
active, but accelerator work waits for the ownership policy. Stable serving
rejects real-backend training instead of quietly pausing traffic.

If an exclusive owner waits behind accelerator work whose completion becomes
unknown, health checks interrupt the wait. Kiln does not free or reuse
potentially live device resources merely to make the queue move.

## Adapter lifecycle

Adapters have both durable and live identities:

- the adapter directory and manifest identify the saved artifact;
- content hashes identify the exact files;
- the runtime records which revision is physically loaded;
- the server default identifies what adapter-free requests should select.

The live model has one physical adapter state at a time. Loading, unloading, or
switching an adapter pauses new actor admission, lets active rows settle, takes
exclusive ownership, changes the weights, invalidates dependent caches and
replay state, and then resumes admission. Queued requests validate that the
revision they expected is still the revision being served.

Per-request adapter selection uses this same transition machinery; it is not
an invisible second copy of the model. Adapter composition first produces a
content-addressed composed artifact, then loads that artifact through the same
validated path.

Adapter publication is transactional. Temporary output is validated before it
is renamed into the durable adapter directory. Manifests and receipts bind the
artifact to its base model, training inputs, effective configuration, and
provenance. Canary failures can quarantine an adapter independently of backend
health.

See the [adapter manifest](docs/ADAPTER_MANIFEST.md) and
[execution provenance](docs/EXECUTION_PROVENANCE.md) references for the exact
identity contracts.

## OpenEnv rollout path

OpenEnv is the environment-facing half of native reinforcement learning:

```text
kiln openenv inspect / rollout / train
        │
        ├── bounded GET health · metadata · schema · environments · OpenAPI
        │
        ▼
content-addressed environment identity
        │
        ▼
one WS /ws connection per candidate episode
        │
        ├── reset(same seed for every candidate in the group)
        ├── Kiln chat generation → one schema-shaped JSON action
        ├── environment step → observation + tagged reward + done
        └── repeat under explicit step/session/data bounds
        │
        ▼
AgenticGroup JSONL
        ├── Action segments → GRPO policy loss
        ├── Observation segments → ECHO environment loss
        └── OpenEnv identity → scored-payload hash and rollout receipt
        │
        ▼
ordinary /v1/train/grpo admission and training queue
```

`kiln-openenv` owns the protocol boundary and has no model or optimizer
responsibility. `kiln-server::openenv_cli` composes sessions with the existing
chat and training APIs. `kiln-train` owns the optional OpenEnv provenance
inside canonical `ScoredRollout`; this lets the same JSONL travel through
native GRPO without a parallel training representation.

The WebSocket path is load-bearing. OpenEnv's HTTP `/reset` and `/step` routes
construct a fresh environment for each request and cannot carry episode state.
The client is strictly lock-step because OpenEnv has no correlation IDs and
permits no server-initiated application messages.

Every group is assigned exactly one environment and reset seed. Candidates may
run concurrently, but their initial messages must be identical or collection
fails. Step rewards—not reset rewards—sum into episode return. Environment
`done`, Kiln's max-step cutoff, invalid policy JSON, and protocol errors remain
distinct termination states.

The boundary is intentionally bounded: redirects are disabled, discovery
bodies and WebSocket frames have independent byte limits, client messages,
environment count, sessions, groups, candidates, steps, action tokens, and
the inline corpus are capped, and terminal OpenEnv errors latch the session
closed. A pinned miniopenenv server is the live interoperability oracle in CI.

See the [OpenEnv training guide](docs/OPENENV_GUIDE.md) for the operator
workflow and artifact contract.

## Training path

Training submissions follow this path:

```text
HTTP request or dashboard form
        │
        ▼
schema and dataset validation
        │
        ▼
backend-route and memory admission
        │
        ▼
FIFO training queue
        │
        ▼
exclusive accelerator ownership
        │
        ▼
forward → loss → backward → optimizer
        │
        ▼
checkpoint and adapter publication
        │
        ├── optional post-training eval gate
        └── optional explicit or configured activation
```

The training worker executes jobs sequentially. Supported workloads are:

- **SFT**: supervised next-token loss over prepared examples;
- **GRPO**: grouped rollouts, reward scoring, normalized advantages, and policy
  updates;
- **OPD**: on-policy sampling followed by teacher-guided distillation loss.

Admission asks the selected backend for its training tape, loss, precision, and
optimizer routes. A compiled backend is not sufficient: if the exact workload
requires an unavailable route, the request fails before taking accelerator
ownership.

SFT may use gradient checkpointing and sparse boundary replay to trade
additional computation for lower activation memory. Planning identity records
the effective checkpoint and loss route. Exact resume rejects drift in those
inputs instead of continuing with a different memory or numerical contract.

Checkpoints include the state required by their declared resume mode, while
final adapters use the project artifact format. Cancellation is checked at
defined work boundaries and must settle backend work before a job becomes
terminal.

## Evaluation pipeline

The evaluation subsystem stores named datasets and suites under the configured
`eval.eval_dir`. A separate FIFO worker executes registered suites, inline
suites, comparisons, and replay jobs. CPU-only scoring can proceed without
accelerator ownership; generation and model-based judging use the shared model
runtime and its ownership rules.

Each suite combines examples with one or more scorers. Scorers include exact or
structured comparisons, regular-expression and numeric checks, code execution
where explicitly configured, and local model judging. A local judge adapter is
loaded through the same adapter transition boundary as any other adapter.

Strict replay binds the suite, generation settings, seeds, model and adapter
identities, scorer configuration, execution provenance, and raw completion
references. Admission verifies that the recorded environment still matches;
execution rechecks the loaded candidate and judge before generation. The
terminal verdict is `matched`, `mismatch`, or `error`.

A training request can attach a post-eval gate. On successful training, Kiln
queues the adapter and optional baseline evaluations, records their IDs on the
training job, and applies the configured promotion rule to the resulting
evidence. Auto-load on pass is explicit policy, not an implication of job
completion.

The judgment workflow turns locally collected A/B preferences into training
examples for a judge adapter. It does not require a hosted frontier model unless
the operator separately configures a remote teacher.

## Failure and health boundaries

Kiln distinguishes failures by the state they can invalidate:

| Failure class | Typical effect |
|---|---|
| Validation | Reject one request before work |
| Admission | Reject work that lacks capacity, policy, or a required route |
| Request execution | Fail or cancel one request after settling owned state |
| Adapter or artifact validation | Quarantine or reject one artifact |
| Backend completion unknown | Quarantine accelerator execution until restart |
| Physical device loss | Fail readiness and require restart |

The backend health latch preserves the first reason an asynchronous completion
can no longer be proven. Once latched, new accelerator work is rejected and
potentially in-flight resources are retained rather than freed unsafely.
Readiness reports the quarantine and restart requirement.

Fallback is therefore route-specific. Kiln may use a declared eager or
reference route after a contained, fully settled failure. It may not treat an
unknown accelerator state as a recoverable performance miss.

## Observability

The main operator surfaces answer different questions:

| Surface | Answers |
|---|---|
| `GET /health` | Is the process ready, and what runtime state could block work? |
| `GET /v1/config` | What configuration and backend policies are effective, and where did they come from? |
| `GET /metrics` | What are the aggregate request, token, queue, training, and memory counters? |
| `GET /v1/debug/model-state` | What exact model, adapter, cache, and execution identity is loaded? |

The debug model-state endpoint is disabled unless
`server.debug_model_state = true` or eval mode enables it. It is intended for a
trusted diagnostic boundary.

Kiln's HTTP API has no built-in authentication. The default loopback bind is
the safe local boundary. If you expose the server beyond a trusted host, put it
behind an authenticated reverse proxy and appropriate network controls.
Request observability can include sensitive prompt or completion text,
especially the recent-request diagnostics and optional durable request log.

Latency reporting separates queue time, accelerator-lock wait, prefill, decode,
graph capture or replay, response backpressure, training, resizing, trimming,
and adapter transitions where those phases are observable. That separation is
necessary: a slow request is not automatically a slow kernel.

## Configuration and source of truth

Do not copy configuration fields from this architecture page. Use:

- [Configuration guide](docs/CONFIGURATION.md) for precedence, validation, and
  examples;
- [complete configuration reference](https://ericflo.github.io/kiln/docs/configuration-complete/)
  for every supported field;
- [configuration schema](https://ericflo.github.io/kiln/docs/configuration-schema/)
  for machine-readable constraints;
- [HTTP API reference](https://ericflo.github.io/kiln/docs/http-api/) for
  endpoints and wire shapes;
- [observability schema](https://ericflo.github.io/kiln/docs/observability-schema/)
  for health, config, metrics, and diagnostic payloads.

For a claim about current backend coverage, regenerate or inspect
`docs/backend-capability-report.md`. For a performance claim, use
[Benchmarks](BENCHMARKS.md) and retain its exact revision, build, device,
driver, workload, and metric provenance. A benchmark result is evidence about
that recorded run; it is never a device-name dispatch rule.

## Where to continue

- [Architecture overview](docs/public/ARCHITECTURE.md) for the shorter system
  map;
- [GRPO guide](docs/GRPO_GUIDE.md) for the generate–score–train workflow;
- [Evals guide](docs/EVAL_GUIDE.md) for suites, scorers, comparisons, and
  replay;
- [Latency observability](docs/LATENCY_OBSERVABILITY.md) for phase definitions;
- [Troubleshooting](https://ericflo.github.io/kiln/troubleshooting.html) for
  operator-facing diagnosis;
- [Security policy](SECURITY.md) for deployment and vulnerability-reporting
  boundaries.
