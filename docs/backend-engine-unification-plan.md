# Backend And Engine Unification Plan

Date: 2026-06-05

This document maps the current backend and engine implementations across the
workspace and proposes a unification plan for CUDA, ROCm, Metal, and Vulkan. It
is intentionally based on the current source layout, not on the older
architecture docs alone. The important conclusion is that Kiln should unify the
contracts, capability reporting, residency model, and training/graph orchestration
without forcing every backend into the same runtime shape.

## Executive Summary

Kiln already has one shared high-level substrate:

- `kiln-tensor` defines `Tensor`, `Device`, `StorageBackend`, `DeviceOp1/2/3`,
  CPU reference implementations, and feature-gated storage/kernels for CUDA,
  ROCm, Metal, and Vulkan.
- `kiln-model::backend::BackendRuntime` is the broad model-engine surface used
  by inference, decode, paged KV, fused attention, GDN, conv1d, optimizer steps,
  LoRA residency, and several training hooks.
- `kiln-train` has moved SFT, GRPO, OPD, LoRA parameters, and optimizer
  application toward `kiln_tensor::Tensor`, `kiln_autograd::Tape`,
  `kiln_param::Parameter`, and `kiln_optim`.
- CUDA and ROCm share a close shape: vendor runtime storage, vendor BLASLt,
  CUDA/HIP-style kernel crates, and graph capture at the model runner level.
- Metal and Vulkan are not weaker copies of CUDA. Metal is built around Apple
  UMA and MSL/ICB execution. Vulkan is built around explicit device-local
  buffers, SPIR-V kernels, resident decode buffers, and `CommandBatch` submit
  reduction.

The current architecture is powerful but too implicit. Capabilities are spread
across Cargo features, trait overrides, environment gates, dtype/shape checks,
fallback behavior, and test-only assumptions. The unification path should make
those contracts explicit while preserving each backend's performance-critical
implementation.

The target architecture should be:

1. A shared tensor/storage contract for device residency and host transfer.
2. A split backend runtime trait family instead of one all-purpose
   `BackendRuntime`.
3. Typed capability descriptors with shape, dtype, layout, batch, graph-safety,
   and training-residency predicates.
4. A common resident-resource lifecycle that can represent CUDA/ROCm device
   pointers, Metal shared buffers, and Vulkan buffer registries.
5. A common graph/replay abstraction whose implementations remain CUDA Graph,
   HIP Graph, Metal ICB, and Vulkan command batching.
6. A common training contract around kt tape, optimizer dispatch, fused loss
   roots, and per-backend precision policy.

That gives Metal and Vulkan the same architectural standing as CUDA and ROCm
without sacrificing their backend-specific fast paths.

## Source Map

### Feature And Crate Fanout

The root workspace has four GPU feature families: `cuda`, `rocm`, `metal`, and
`vulkan`. The top-level feature edges fan out through:

- `crates/kiln-server/Cargo.toml`, which forwards backend features into
  `kiln-model`, `kiln-tensor`, `kiln-train`, and `kiln-kt-bridge`.
- `crates/kiln-model/Cargo.toml`, which pulls the model runtime backend plus
  backend-specific kernel crates.
- `crates/kiln-train/Cargo.toml`, which forwards training features into
  `kiln-model`, `kiln-flce-kernel`, `kiln-rmsnorm-kernel`,
  `kiln-opd-loss-kernel`, and `kiln-kt-bridge`.
- `crates/kiln-tensor/Cargo.toml`, which pulls storage/runtime/BLAS support:
  `cudarc` and `kiln-blas` for CUDA, `kiln-hip` and `kiln-rocblas` for ROCm,
  `objc2`/`objc2-metal` for Metal, and `kiln-vulkan-kernel` for Vulkan.

The performance crates form two clusters:

- BLAS/request layers:
  - `kiln-blas`: CUDA cublasLt layer plus backend-agnostic matmul request,
    algo cache, and workspace pool types.
  - `kiln-rocblas`: ROCm hipBLASLt analog reusing the backend-agnostic BLAS
    types.
  - `kiln-mps`: Metal BLAS/MPS-facing layer and Apple UMA policy helpers.
  - `kiln-vulkan-blas`: Vulkan BLAS layer extending `kiln-vulkan-kernel`.
- Custom kernels:
  - `kiln-flash-attn`
  - `kiln-gdn-kernel`
  - `kiln-conv1d-kernel`
  - `kiln-rmsnorm-kernel`
  - `kiln-marlin-gemm`
  - `kiln-flce-kernel`
  - `kiln-opd-loss-kernel`

CUDA and ROCm mostly share the custom-kernel crate surface. Metal and Vulkan
mostly implement equivalent behavior through kt-native MSL or SPIR-V paths in
`kiln-tensor`, `kiln-model`, `kiln-graph-metal`, and `kiln-vulkan-kernel`.

### Tensor Substrate

`crates/kiln-tensor/src/device.rs` defines:

- `Device::Cpu`
- `Device::Cuda(usize)`
- `Device::Metal(usize)`
- `Device::Vulkan(usize)`
- `Device::Rocm(usize)`

`crates/kiln-tensor/src/storage.rs` defines the common dynamic storage contract:

- `StorageBackend`
- `Storage = Arc<dyn StorageBackend>`
- `CpuStorage` as the canonical host reference storage

The backend storage implementations are:

- CUDA: `cuda_storage.rs`, `cuda_allocator.rs`, `active_stream.rs`,
  `capture_alloc.rs`, `cuda_matmul.rs`, and `fp8.rs`.
- ROCm: `rocm_storage.rs`, `rocm_allocator.rs`, `active_rocm_stream.rs`,
  `rocm_capture_alloc.rs`, `rocm_matmul.rs`, and `rocm_ops.rs`.
- Metal: `metal_storage.rs`, `metal_allocator.rs`, `metal_rt.rs`,
  `metal_kernels.rs`, `metal_matmul.rs`, and `metal_types.rs`.
- Vulkan: `vulkan_storage.rs`, `vulkan_allocator.rs`, and `vk_shaders.rs`.

`crates/kiln-tensor/src/device_op.rs` is the key correctness layer. Each
`DeviceOp1/2/3` has CPU plus optional backend forwards. CPU is the oracle.
CUDA is intentionally strict: if a CUDA native forward returns `None`, the CPU
forward is attempted on CUDA storage and fails loudly. Metal, Vulkan, and ROCm
currently use correctness-first host fallback for missing native forwards:
device -> CPU -> CPU reference op -> original device. ROCm additionally logs
host fallback when profiling is enabled.

This asymmetry matters. It is useful during bring-up, but it is not a long-term
capability contract. A unified engine should make "native", "host fallback",
"decline", and "must be native" explicit per call site.

### Model Runtime Surface

`crates/kiln-model/src/backend/mod.rs` defines `BackendRuntime`. The trait is
very broad. It currently mixes:

- Backend identity and device reporting.
- Training capability reporting.
- Resident decode pool support.
- Resident activation registration, update, resolve, readback, and eviction.
- FlashAttention prefill and paged decode.
- Paged KV head-major read/write/append.
- GDN forward substitution, recurrent state, chunk prep/scan/full-forward, decode
  fusions, gates, and gated RMSNorm.
- Conv1d prefill/update.
- Optimizer dispatch for SGD and AdamW.
- Linear prefill/decode/lm-head/sample paths.
- LoRA delta and LoRA decode add paths.
- Graph-output variants for decode replay.

That breadth is the reason unification feels harder than it should. It is also
why backend differences are hard to audit: a backend can claim support through a
`supports_*` method, shape-gate inside the implementation, decline with
`Ok(None)`, or be disabled by environment variables.

`for_device_kt` chooses the concrete runtime from `kiln_tensor::Device`. The
current concrete backends are:

- `CpuBackend`: identity only, declines everything else.
- `CudaBackend`: broad fused model and training path.
- `RocmBackend`: close CUDA sibling with HIP runtime and ROCm-specific gaps.
- `MetalBackend`: Apple GPU path with MSL/ICB and UMA storage.
- `VulkanBackend`: explicit SPIR-V/buffer residency path with native decode
  batching.

### Graph And Replay Layer

There are two graph/replay layers today:

1. Generic scaffolding crates:
   - `kiln-graph`
   - `kiln-graph-cuda`
   - `kiln-graph-metal`
   - `kiln-graph-vulkan`
2. Production model-level runners:
   - `crates/kiln-model/src/cuda_graph.rs`
   - `crates/kiln-model/src/rocm_graph.rs`
   - `crates/kiln-model/src/metal_graph.rs`
   - `crates/kiln-model/src/vk_decode_resident.rs`
   - `crates/kiln-vulkan-kernel/src/cmd_batch.rs`

The generic `kiln-graph` crate has the right vocabulary: `CapturedGraph`,
`CaptureSession`, pinned pointer auditing, and allocator mode/freeze concepts.
But it is not yet where the real capture/replay behavior lives.

The real behavior is backend-specific:

- CUDA uses `CudaGraphRunner` around CUDA graph capture/replay for decode.
- ROCm uses `RocmGraphRunner` around HIP graph capture/replay.
- Metal uses `MetalGraphRunner` and `kiln_graph_metal::MetalCapturedGraph` with
  Metal ICB/stable-buffer behavior.
- Vulkan uses command batching and resident decode rather than a CUDA-style graph
  capture equivalent. `CommandBatch` records many compute dispatches into one
  command buffer and submits once.

The unification target should treat "replay a stable decode/training region" as
the common contract, while preserving the backend's execution primitive.

### Training And Tape

`kiln-train` is now centered on the kt stack:

- `kiln_tensor::Tensor`
- `kiln_autograd::Tape`
- `kiln_param::Parameter`
- `kiln_optim`
- `BackendRuntime` optimizer and fused-kernel hooks

The historical CUDA-specific training entry points in `cuda_train.rs` delegate
into the shared trainer. SFT/GRPO/OPD are increasingly kt-native and use
backend-gated fast paths for fused loss roots and optimizer dispatch.

The relevant shims are:

- `sft_tape_shim.rs`: FLCE and SFT loss/tail behavior, with CUDA/ROCm kt paths
  and Vulkan-specific bridges where needed.
- `grpo_tape_shim.rs`: GRPO scalar loss roots and analytic kt backward, with
  CUDA/ROCm fused fast paths and Vulkan-specific loss/grad paths.
- `opd_tape_shim.rs`: OPD top-K/reverse-KL scalar loss, with CUDA/ROCm fused
  kernel paths, Metal/CPU analytic kt paths, and Vulkan-specific active-hidden
  device paths.

Vulkan still has a stronger separate low-level identity than CUDA/ROCm/Metal
because the SPIR-V leaf layer exposes `VkTensor`, explicit buffer operations,
and resident decode/training bridges. The direction is correct: keep the Vulkan
SPIR-V kernels, but make the training orchestration and gradient/optimizer
contracts shared.

## Backend Shapes

### CUDA

CUDA is the most mature vendor-runtime implementation.

Core shape:

- Storage: `CudaStorage` over `cudarc::driver::CudaSlice<u8>` and
  `Arc<CudaContext>`.
- Allocation: `CudaAllocator` and capture-aware arenas for pointer stability.
- Stream model: active CUDA stream thread-local helpers and priority streams.
- BLAS: `kiln-blas` cublasLt matmul and algo/workspace cache.
- Custom kernels: FlashAttention, GDN, conv1d, RMSNorm, Marlin, OPD, FLCE
  surfaces.
- Graphs: real CUDA decode graph runner in `cuda_graph.rs`.
- Fallback stance: strict for `DeviceOp` native misses.

Capability profile:

- Strong BF16/F16/F32 matmul through cublasLt.
- Broad fused GDN, RMSNorm, conv1d, LoRA decode add, and optimizer fast paths.
- FlashAttention prefill and paged decode are first-class.
- CUDA graphs are the decode launch-overhead optimization, although capture is
  shape/cache/metadata-sensitive.
- Training is shared kt tape plus fused CUDA loss/optimizer leaves.

CUDA should remain the reference implementation for high-throughput discrete
vendor-runtime backends.

### ROCm

ROCm is a CUDA-shaped backend, but it is not a literal CUDA clone.

Core shape:

- Storage: `RocmStorage` over `kiln_hip::{RocmContext, RocmSlice}`.
- Allocation: `RocmAllocator`, `RocmCaptureArena`, active ROCm stream helpers.
- BLAS: `kiln-rocblas` hipBLASLt analog to `kiln-blas`.
- Custom kernels: many CUDA/HIP-style kernels are shared-source or hipcc-built;
  FlashAttention and Marlin have correctness-first alternatives where the CUDA
  implementation depends on NVIDIA-specific libraries or inline PTX.
- Graphs: HIP graph runner in `rocm_graph.rs`.
- Fallback stance: correctness-first host fallback for missing `DeviceOp`
  forwards, with profiling hooks.

Capability profile:

- Matmul mirrors CUDA through hipBLASLt for F32/BF16/F16.
- GDN/RMSNorm/conv1d/optimizer paths largely mirror CUDA through HIP-compatible
  kernel crates.
- ROCm has head-major prefill support gated by `KILN_ROCM_HEAD_MAJOR_PREFILL`.
- Some code still carries CUDA names, comments, and telemetry strings.
- At least one support method should be audited: `supports_linear_decode_argmax`
  returns true while the current `linear_decode_argmax` override returns
  `Ok(None)`.

ROCm should be unified with CUDA where the abstraction is genuinely shared:
device pointer extraction, stream/capture lifecycle, BLASLt request descriptors,
and HIP/CUDA-compatible kernel crate wrappers. It should not be forced to claim
native parity for FlashAttention/Marlin internals that are materially different.

### Metal

Metal is an Apple UMA backend, not a discrete-device CUDA emulation.

Core shape:

- Storage: `MetalStorage` over shared `MTLBuffer` handles and a `MetalCompanion`.
- Allocation: `MetalAllocator` using shared Metal buffers.
- Kernel style: kt-native MSL functions, pipeline caches, direct command
  encoders, and some Metal graph/ICB integration.
- BLAS/matmul: BF16-focused custom MSL matmul and `kiln-mps` as the Metal BLAS
  layer direction.
- Graphs: `MetalGraphRunner` and `kiln-graph-metal` ICB/stable-buffer support.
- Fallback stance: correctness-first host fallback for generic missing
  `DeviceOp` forwards. On Apple Silicon, host fallback is still a synchronization
  and CPU-work cost, but not a PCIe copy in the same way as discrete GPUs.

Capability profile:

- Strong resident BF16 model paths via MSL.
- Native Metal attention/paged decode variants, including contiguous and
  graph-output forms.
- Native GDN recurrent/chunk/gates/gated-RMSNorm and conv1d paths.
- Native sampled lm-head paths, including batch mixed greedy/sampling forms.
- AdamW dispatch exists; SGD is not currently overridden in the backend trait
  list.
- The implementation is large and monolithic, especially `backend/metal.rs`,
  which mixes runtime trait implementation, MSL strings, pipeline setup,
  residency, decode helpers, and tests.

Metal should be unified through resource, capability, replay, and training
contracts while keeping MSL kernels and UMA assumptions explicit.

### Vulkan

Vulkan is the explicit-resource backend.

Core shape:

- Storage: `VulkanStorage` around `Arc<VulkanBuffer>` and `Arc<VulkanDevice>`.
- Low-level leaf layer: `kiln-vulkan-kernel`, `VkTensor`, `vk_ops`, SPIR-V
  kernels, buffer upload/readback, and command batching.
- Allocation: `VulkanAllocator` backed by `VulkanStorage::zeros`.
- Decode optimization: resident decode pools and `CommandBatch`, not CUDA-style
  graph capture.
- Fallback stance: correctness-first host fallback for missing generic
  `DeviceOp` forwards.

Capability profile:

- Native F32 and BF16 tensor storage and host round trips are implemented.
- Matmul has Vulkan paths for rank-2/rank-3+ cases, with dtype/shape gates.
- Native GDN, conv1d, attention, lm-head argmax/sample, MLP, QKV decode, and
  optimizer dispatch are implemented through SPIR-V/buffer helpers.
- Resident activation registry uploads and tracks Vulkan buffers keyed by
  `TensorId`.
- Vulkan uses explicit mixed precision policy in training: activations often
  remain F32 while BF16 weights/LoRA are handled through backend-specific paths.
- `lora_delta_resident` explicitly declines because the older Candle Vulkan
  LoRA op was removed; the kt tape path should own that recording.

Vulkan should remain explicit-resource and command-batch-oriented. The unifier
should not try to make it look like cublasLt plus CUDA graphs. It should instead
make `VkTensor`/`VulkanBuffer` residency a first-class implementation of the
same resource and replay contracts used by the other backends.

## Capability Matrix

This table is deliberately architectural. Individual shape gates and environment
flags still need generated capability reporting.

| Capability | CUDA | ROCm | Metal | Vulkan |
|---|---|---|---|---|
| Device storage | `CudaStorage` / `CudaSlice` / `CudaContext` | `RocmStorage` / `RocmSlice` / `RocmContext` | `MetalStorage` / shared `MTLBuffer` | `VulkanStorage` / `VulkanBuffer` |
| Host transfer | H2D/D2H helpers; no generic cross-GPU transfer | H2D/D2H helpers; no generic cross-GPU transfer | Shared-buffer host read/write plus sync | Upload/readback through Vulkan buffers |
| Generic op fallback | Strict/loud on native miss | Host round trip on native miss | Host round trip on native miss | Host round trip on native miss |
| Dense matmul | cublasLt through `kiln-blas` | hipBLASLt through `kiln-rocblas` | BF16 custom MSL and MPS direction | SPIR-V/VkTensor kernels with rank/dtype gates |
| Flash attention | Vendored CUDA FA and paged decode | HIP/composite/ROCm variants; head-major gate | Metal SDPA/paged decode variants | Vulkan SDPA/paged decode variants |
| GDN | Fused CUDA kernel crate | HIP-compatible fused kernel paths | MSL fused paths | SPIR-V fused/resident paths |
| Conv1d | CUDA kernel crate | HIP-compatible kernel path | MSL path | SPIR-V path |
| RMSNorm/gates | CUDA fused kernels | HIP-compatible fused kernels | MSL paths | SPIR-V paths |
| Optimizer dispatch | SGD and AdamW native | SGD and AdamW native | AdamW native, no SGD override observed | SGD and AdamW native |
| LoRA resident delta | Native kt/backend path | Native kt/backend path | Some resident activation paths, no generic linear prefill override observed | Explicitly declined for resident delta; kt tape path preferred |
| Decode replay | CUDA graph runner | HIP graph runner | Metal graph/ICB runner | Resident decode plus `CommandBatch` |
| Training stack | Shared kt tape plus CUDA fused leaves | Shared kt tape plus ROCm fused leaves | Shared kt tape plus Metal analytic/MSL leaves | Shared kt tape with Vulkan-specific loss/buffer leaves |
| Main risk | Capture stability and strict native gaps | Drift from CUDA names plus non-hipifiable kernels | Monolithic implementation and broad shape gates | Parallel explicit-resource leaf layer and residency complexity |

## Major Architectural Differences

### Memory Model

CUDA and ROCm are device-pointer backends. They need stream ownership, pointer
stability during graph capture, explicit H2D/D2H copies, and BLASLt workspace
management.

Metal is a shared-memory backend. It still needs synchronization and layout
discipline, but its fast path is not built around staging bytes across PCIe.
`MetalStorage` can wrap shared `MTLBuffer` objects and pass them directly to MSL
kernels.

Vulkan is explicit device-local buffer management. The runtime must own buffer
lifetimes, descriptor binding, pipeline caches, command buffer recording, and
readback/upload behavior. Its best decode path is about keeping buffers resident
and reducing submits, not capturing a CUDA-like stream graph.

### Fallback Semantics

The fallback rules are not unified:

- CUDA: native miss should be visible because the fallback attempts CPU logic on
  CUDA storage and errors.
- Metal/Vulkan/ROCm: native miss may silently stage to CPU and back, preserving
  correctness but hiding performance cliffs.

This made backend bring-up easier. It now needs a more formal policy:

- Training hot path: native required or explicit error.
- Decode hot path: native required unless an opt-in debug fallback is set.
- Correctness tests and rare ops: host fallback allowed and counted.
- CPU reference: always available for parity.

### Capability Encoding

Capabilities are currently encoded in at least six places:

- Cargo features.
- `BackendRuntime` override presence.
- `supports_*` boolean methods.
- Shape/dtype/layout gates inside implementations.
- Environment variables such as `KILN_DISABLE_*`, `KILN_ROCM_*`,
  `KILN_METAL_*`, and `KILN_VULKAN_*`.
- Test gates and runtime availability checks.

This is brittle. A user or scheduler cannot know from a single source whether
`flash_attn_prefill`, `linear_decode_argmax`, or `dispatch_adamw_step` is
expected to run native for a specific backend, shape, dtype, and mode.

### Graph Layer Split

The `kiln-graph-*` crates are architecturally correct but not authoritative.
Real CUDA/HIP/Metal replay behavior lives in `kiln-model/src/*graph.rs`.
Vulkan replay behavior lives in `kiln-vulkan-kernel::CommandBatch` and
`vk_decode_resident.rs`.

Unification should move the real runner contracts into the graph crates, or
rename the graph crates as scaffolds and introduce a new authoritative replay
layer. Leaving both layers with overlapping names will keep causing confusion.

### Backend File Shapes

CUDA and ROCm are split across runtime files, tensor storage, BLAS crates, and
kernel crates. This is a good shape, although ROCm still carries copy/paste
CUDA naming in places.

Metal has too much in `backend/metal.rs`. It needs module boundaries around:

- runtime trait implementation
- attention
- GDN
- conv1d
- lm-head/sampling
- optimizer
- residency
- graph/ICB
- pipeline/MSL sources

Vulkan is split better between model and kernel crates, but it still has a
parallel explicit-resource world that needs clearer integration points with
`kiln-tensor`, `kiln-autograd`, and `BackendRuntime`.

## Unification Principles

1. Do not define unification as "make Metal and Vulkan look like CUDA".
   CUDA-style graph capture and BLASLt are not the right primitives for every
   backend.

2. Unify semantic contracts first:
   - device storage
   - resident resource identity
   - fallback policy
   - operation capability
   - replay/capture lifecycle
   - training step ownership

3. Keep backend-native execution:
   - CUDA: cublasLt, CUDA graphs, CUDA kernels.
   - ROCm: hipBLASLt, HIP graphs, HIP-compatible kernels and ROCm-native
     substitutes where needed.
   - Metal: MSL, shared buffers, ICB, MPS where it wins.
   - Vulkan: SPIR-V, explicit buffers, descriptor/pipeline caches,
     `CommandBatch`.

4. Replace bool-only capability checks with typed predicates. A capability is
   not just "supports flash attention". It is "supports flash attention for
   dtype BF16, head dim 256, layout X, batch policy Y, graph replay Z, with
   these environment gates".

5. Make performance cliffs observable. Host fallback should increment counters
   and be optionally fatal in hot paths.

6. Preserve CPU as the correctness oracle. The CPU path remains the reference
   for tensor ops and targeted parity tests, not the production fallback for
   every hot model path.

## Proposed Target Architecture

### 1. Backend Capability Descriptors

Add a backend capability layer, for example:

```rust
pub struct BackendCapabilities {
    pub backend: Backend,
    pub storage: StorageCapabilities,
    pub matmul: MatmulCapabilities,
    pub attention: AttentionCapabilities,
    pub gdn: GdnCapabilities,
    pub decode: DecodeCapabilities,
    pub training: TrainingCapabilities,
    pub graph_replay: ReplayCapabilities,
    pub fallback: FallbackPolicy,
}
```

Each sub-capability should expose predicates rather than raw booleans:

```rust
fn supports_matmul(&self, req: &MatmulRequest) -> Support;
fn supports_attention(&self, req: &AttentionRequest) -> Support;
fn supports_replay(&self, req: &ReplayRequest) -> Support;
```

`Support` should distinguish:

- `Native`
- `NativeWithConstraints`
- `HostFallbackAllowed`
- `Declined`
- `Unsupported`
- `DisabledByEnv`
- `RequiresFeature`

This should be available at runtime for diagnostics and scheduler decisions, and
as a generated Markdown/JSON report for audits.

### 2. Split `BackendRuntime`

Keep the existing trait as a compatibility facade initially, but implement it
through focused traits:

- `BackendIdentity`
- `AttentionBackend`
- `PagedKvBackend`
- `GdnBackend`
- `ConvBackend`
- `LinearBackend`
- `SamplingBackend`
- `ResidencyBackend`
- `OptimizerBackend`
- `TrainingLossBackend`
- `ReplayBackend`

Benefits:

- Missing capability becomes local and visible.
- CUDA/ROCm shared implementations can reuse the same adapter where safe.
- Metal/Vulkan can implement replay/residency without inheriting CUDA-specific
  capture vocabulary.
- Tests can target one surface at a time.

Migration path:

1. Add focused traits next to `BackendRuntime`.
2. Implement them for existing backends by delegating to existing methods.
3. Make `BackendRuntime` default methods call the focused traits.
4. Move call sites family by family.
5. Delete or shrink the all-purpose trait once call sites are migrated.

### 3. Unified Resident Resource Lifecycle

Introduce a resident resource abstraction that is explicit about ownership,
layout, dtype, and replay safety:

```rust
pub trait ResidentResource: Send + Sync {
    fn backend(&self) -> Backend;
    fn tensor_id(&self) -> Option<TensorId>;
    fn dtype(&self) -> DType;
    fn shape(&self) -> &[usize];
    fn byte_len(&self) -> usize;
    fn replay_stability(&self) -> ReplayStability;
}
```

Backend-specific implementations can wrap:

- CUDA: `CudaStorage`, borrowed/owned device slices, capture arena allocations.
- ROCm: `RocmStorage`, borrowed/owned HIP slices, ROCm capture arena allocations.
- Metal: shared `MTLBuffer` plus companion/device index.
- Vulkan: `VulkanBuffer`, `VkTensor`, registry entries, descriptor-safe handles.

This should replace ad hoc resident activation registries that vary by backend:
CUDA/ROCm/Metal currently mostly track membership by `TensorId`, while Vulkan
uploads and owns buffer entries keyed by `TensorId`. Both are valid. The
unified contract should expose the semantic lifecycle and leave storage details
backend-specific.

### 4. Common Matmul And Linear Request Surface

The BLAS crates already point in the right direction. The next step is a single
request/capability layer used by all four backends:

```rust
pub struct MatmulRequest {
    pub lhs_shape: Shape,
    pub rhs_shape: Shape,
    pub lhs_dtype: DType,
    pub rhs_dtype: DType,
    pub out_dtype: DType,
    pub accumulation: Accumulation,
    pub layout: MatmulLayout,
    pub batch: BatchPolicy,
    pub epilogue: Option<Epilogue>,
    pub replay_safe: bool,
}
```

Backends then implement the request through their best primitive:

- CUDA: cublasLt and CUDA kernels.
- ROCm: hipBLASLt and ROCm fallbacks.
- Metal: MSL/MPS paths with UMA-aware allocation.
- Vulkan: SPIR-V/VkTensor kernels and command-batch-aware dispatch.

This is the right level of unification. It avoids a fake lowest-common
denominator while letting call sites stop asking backend-specific questions.

### 5. Replay Contract

Create an authoritative replay trait that lives with or replaces the
`kiln-graph-*` scaffolds:

```rust
pub trait ReplayPlan {
    fn backend(&self) -> Backend;
    fn key(&self) -> ReplayKey;
    fn validate_inputs(&self, inputs: &[ResidentResourceRef]) -> Result<()>;
    fn replay(&mut self, inputs: ReplayInputs<'_>) -> Result<ReplayOutputs>;
    fn invalidate_reason(&self, state: &ReplayState) -> Option<InvalidateReason>;
}
```

Backend implementations:

- CUDA: CUDA graph executable plus stable metadata/pointer checks.
- ROCm: HIP graph executable plus the same capture safety concepts.
- Metal: ICB and stable shared-buffer plan.
- Vulkan: `CommandBatch`, secondary command buffers where useful, and resident
  decode buffer plans.

The common layer owns:

- bucketing keys
- invalidation policy
- input/output stability checks
- fallback and retry policy
- counters and logs

The backend layer owns:

- how commands are captured or recorded
- how resources are bound
- how kernels are submitted
- how synchronization works

### 6. Shared Training Contract

The target training architecture should be:

- `kiln-train` owns SFT/GRPO/OPD orchestration.
- `kiln_autograd::Tape` owns graph recording and backward traversal.
- `kiln_param::Parameter` owns master/forward parameter state and epochs.
- `kiln_optim` owns optimizer math contracts.
- Backend traits provide:
  - loss-root fused kernels when available
  - optimizer in-place updates when available
  - resident activation/materialization hooks
  - precision policy
  - graph/replay eligibility

Backend-specific training policies should be explicit:

- CUDA/ROCm: BF16/F16/F32 tensor paths with fused CUDA/HIP leaves.
- Metal: BF16-focused MSL and analytic kt paths where fused leaves are absent.
- Vulkan: explicit F32 activation/mixed BF16 weight policy and VkTensor buffer
  bridges.

The training loop should not know whether the backend is CUDA, ROCm, Metal, or
Vulkan except through capabilities and policy objects.

## Migration Plan

### Phase 0: Audit And Stabilize Capability Reporting

Deliverables:

- Add a generated backend capability report in Markdown and JSON.
- Collect Cargo feature status, trait override presence, support methods,
  shape/dtype gates, environment gates, and fallback policy.
- Add a test that fails when a `supports_*` method returns true but the method
  body always declines for a representative supported request.
- Fix obvious stale naming:
  - CUDA labels in ROCm comments/logs/statics.
  - Tensor docs that still say Vulkan is not implemented where constructors and
    copies now exist.
  - `kiln-graph-*` comments that imply production capture lives there when it
    still lives in model-level files.

This phase should not refactor performance code. It makes the current state
inspectable.

### Phase 1: Introduce Focused Backend Traits

Deliverables:

- Add focused traits for attention, GDN, conv, linear, sampling, residency,
  optimizer, training loss, and replay.
- Implement the traits by delegating to existing backend methods.
- Keep `BackendRuntime` intact as a facade.
- Add tests that each concrete backend's advertised capabilities match the
  focused trait implementations.

This phase gives the codebase a better shape without changing behavior.

### Phase 2: Normalize Fallback Policy

Deliverables:

- Add `FallbackPolicy` per operation family and mode:
  - `CorrectnessAllowed`
  - `WarnAndCount`
  - `ErrorInHotPath`
  - `NativeRequired`
- Count all host fallbacks for Metal, Vulkan, and ROCm.
- Add hot-path guards for decode and training so a missing native kernel cannot
  silently destroy performance.
- Keep CPU parity tests able to opt into host fallback.

This preserves correctness while making performance failures visible.

### Phase 3: Unify Resident Resource Semantics

Deliverables:

- Add `ResidentResource` and `ResidentRegistry` contracts.
- Implement resource wrappers for CUDA, ROCm, Metal, and Vulkan.
- Route existing resident activation APIs through the registry.
- Represent Vulkan's upload-owned registry and Metal/CUDA/ROCm's storage-owned
  membership model through the same lifecycle states.
- Add replay-stability metadata to resident resources.

This phase is the keystone for unified decode, graph replay, and training
residency.

### Phase 4: Unify Matmul And Linear Dispatch

Deliverables:

- Promote a single `MatmulRequest`/`LinearRequest` descriptor.
- Wire cublasLt, hipBLASLt, Metal MSL/MPS, and Vulkan SPIR-V through the same
  request surface.
- Preserve backend-specific algo caches and workgroup/tile heuristics.
- Add dtype/layout/epilogue capability predicates.
- Add parity tests across CPU and every enabled backend for representative
  rank-2, batched, transposed, BF16/F16/F32, and epilogue cases.

This should remove backend-specific linear/matmul branching from model code
without narrowing backend performance.

### Phase 5: Move Replay Into The Authoritative Graph Layer

Deliverables:

- Move or wrap `CudaGraphRunner`, `RocmGraphRunner`, `MetalGraphRunner`, and
  Vulkan `CommandBatch` decode plans behind one `ReplayBackend`.
- Define shared `ReplayKey`, invalidation, capture-safety, and stable-input
  contracts.
- Keep backend implementations native:
  - CUDA graph
  - HIP graph
  - Metal ICB
  - Vulkan command batch/resident plan
- Add replay parity tests that compare eager and replay outputs for small
  stable decode regions.

This makes graph/replay a first-class engine concept rather than four separate
model-side special cases.

### Phase 6: Finish Shared Training Integration

Deliverables:

- Make SFT/GRPO/OPD training choose device behavior only through capability and
  precision policy.
- Route fused loss roots through `TrainingLossBackend`.
- Route optimizer updates through `OptimizerBackend`.
- Make Vulkan's mixed F32/BF16 policy explicit and documented in code.
- Add per-backend one-step training proofs:
  - gradients reach LoRA parameters
  - optimizer updates resident parameters
  - loss decreases on a tiny deterministic fixture where expected
- Keep CUDA/ROCm fused paths and Vulkan/Metal specialized paths intact.

This phase removes the remaining training orchestration differences without
deleting useful backend leaf kernels.

### Phase 7: Decompose Backend Modules

Deliverables:

- Split `backend/metal.rs` by operation family and runtime concern.
- Split `backend/vulkan.rs` around residency, attention, GDN, linear/sampling,
  optimizer, and replay.
- Factor CUDA/ROCm shared code where it is truly identical:
  - support predicates
  - kt bridge helpers
  - BLASLt request conversion
  - resident activation membership
  - optimizer argument validation
- Do not factor code that hides real platform differences:
  - FlashAttention internals
  - Marlin/native quantized GEMM internals
  - graph capture details
  - wave-size/subgroup assumptions

This phase is mostly mechanical once the contracts exist.

### Phase 8: Conformance And Performance Gates

Deliverables:

- A backend conformance suite:
  - storage round trip
  - `DeviceOp` parity
  - matmul/linear parity
  - attention/GDN/conv parity
  - optimizer parity
  - replay parity
  - one-step training proof
- A performance sentinel suite:
  - no unexpected host fallback in decode/training hot paths
  - max submit count or replay count per decode token
  - matmul algorithm/cache hit reporting
  - backend-specific latency thresholds on known hardware fixtures
- A generated capability dashboard checked into docs or build artifacts.

This is what keeps unification from becoming a regression factory.

## Immediate Backlog

1. Generate a capability report from the live tree.
   Start with override presence plus explicit support methods for
   `CudaBackend`, `RocmBackend`, `MetalBackend`, and `VulkanBackend`.

2. Fix ROCm naming drift.
   Rename CUDA-labeled ROCm statics/comments/log messages where they describe
   ROCm behavior. This matters because capability reports and logs will
   otherwise be misleading.

3. Audit `RocmBackend::supports_linear_decode_argmax`.
   If the implementation still returns `Ok(None)` for the advertised path,
   either implement it or make the support predicate return false until native
   support lands.

4. Clarify graph crate authority.
   Either move real runners into `kiln-graph-*` or document the generic crates
   as scaffolding and create a new production replay layer.

5. Add fallback counters to Metal and Vulkan matching ROCm's profiling spirit.
   Host fallback should be measurable across all non-CUDA bring-up paths.

6. Split `BackendRuntime` without changing behavior.
   The first PR can add focused traits and implement them by forwarding to the
   current methods.

7. Decompose `backend/metal.rs`.
   This should follow the focused trait boundaries and avoid changing kernel
   behavior in the same PR.

8. Add hot-path native-required checks.
   Decode and training should fail clearly when a backend silently drops to CPU
   for an operation expected to be resident.

## Expected End State

After the migration, a model/training call site should not ask:

- "Is this CUDA?"
- "Is this Metal?"
- "Does Vulkan need a special path?"
- "Will this method return `Ok(None)`?"

It should ask:

- "Can the active backend execute this request natively?"
- "Is host fallback allowed in this mode?"
- "Is the request replay-safe?"
- "What resident resources does this operation read and write?"
- "What precision policy applies?"

The backend then answers through a typed capability and executes through its
native implementation. CUDA still uses CUDA graphs and cublasLt. ROCm still uses
HIP graphs and hipBLASLt. Metal still uses MSL, shared buffers, and ICB. Vulkan
still uses SPIR-V, explicit buffers, and command batching.

That is the right unification boundary for Kiln: one engine contract, four
native execution models.

