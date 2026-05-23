# `kiln-tensor` substrate quickstart

This is the contributor entry point for the in-house tensor /
autograd / optimizer / graph-capture substrate that
[#1082](https://github.com/ericflo/kiln/issues/1082) introduces.
Read this if you're touching any code under `crates/kiln-tensor/`,
`crates/kiln-blas/`, `crates/kiln-mps/`, `crates/kiln-vulkan-blas/`,
`crates/kiln-param/`, `crates/kiln-autograd/`, `crates/kiln-optim/`,
or `crates/kiln-graph/`.

For the full design, see the [migration substrate section of
ARCHITECTURE.md](../ARCHITECTURE.md#1082-migration-substrate-phase-1-onward)
and the [#1082 issue body](https://github.com/ericflo/kiln/issues/1082).

For the current shipped status, see
[`bench-results/substrate-status.md`](../bench-results/substrate-status.md)
(regenerable via `scripts/audit-substrate-status.sh`).

## The 8 substrate crates at a glance

```
kiln-tensor          Tensor + Storage + Layout + DType + TensorId
                     + ops (11 CPU op families today)
                     + StreamPlanner + Allocator + CpuAllocator
                     + Activation registry + profile counters
                     + safetensors load/save + determinism contract

kiln-blas            CUDA cublasLt path (production AlgoCache +
                     WorkspacePool today; cublasLt MatmulHandle Phase 2.x)
                     Carries candle-core dep behind `probe` feature
                     today; Phase 7 swaps to direct cudarc.

kiln-mps             Metal MPS path (MpsTilePolicy + MpsUmaHint)
                     Carries candle-core dep behind `probe` feature.

kiln-vulkan-blas     Vulkan compute path (VkWorkgroupConfig +
                     VkPipelineCacheKey + VkCooperativeMatrixSupport)
                     **No candle dep** — kiln-vulkan-kernel is already
                     candle-free.

kiln-param           Parameter handle: one logical parameter, one
                     stable TensorId, multiple physical storages
                     (forward + backward + transposed_cache + lora_delta
                     + output heads). AmpPolicy carried per-Parameter.

kiln-autograd        Tape-based reverse-mode autograd over
                     kiln_tensor::Tensor + TensorId. Anti-pattern 16
                     enforcement via Arc<AtomicU64> version handles.

kiln-optim           Fused per-backend optimizer step. AdamW + SGD
                     concrete CPU impls; Lion + Muon scaffolds.
                     MomentLocation + StochasticRoundingPolicy.

kiln-graph           Command-list capture for CUDA / Metal / Vulkan.
                     CapturedGraph trait + CaptureSession (RAII guard)
                     + AllocatorMode (re-exported from kiln-tensor)
                     + dangling-pointer audit (anti-pattern 16's twin).
```

## The dispatch flow

```text
User call site
  └─ kiln_tensor::ops::matmul(&a, &b)
       └─ dispatch2(&MatmulOp, &a, &b)
            └─ match a.device() {
                 Device::Cpu     -> op.cpu_fwd(&a, &b),
                 Device::Cuda(_) -> op.cuda_fwd(&a, &b)
                                    (default Ok(None) ⇒ fall back to cpu_fwd),
                 Device::Metal(_)-> op.metal_fwd(...),
                 Device::Vulkan(_)-> op.vulkan_fwd(...),
               }
            └─ Op may record on the autograd tape (Phase 6a) if
              requires_grad is set on the parent Parameter
            └─ Returns kiln_tensor::Tensor (zero-copy views via Layout)
```

Training:

```text
forward → kiln_autograd::Tape::record per op
       → loss tensor
       → tape.backward(loss_id, seed_grad, accumulator) → GradStore
       → optimizer.step(parameter, grad) per parameter
            └─ reads parameter.amp_policy() for dispatch dtype
            └─ updates parameter.backward_storage() in place
              (in-place mutation calls tensor.bump_version()
              — anti-pattern 16)
       → Tape::clear() (required before next forward)
```

Graph capture:

```text
allocator.warm(dtype, n, count) for every shape the captured graph needs
session = CaptureSession::begin()
allocator.set_mode(AllocatorMode::Frozen)
... per-backend graph capture ops, each session.pin(&tensor) ...
session.finalize()
// On every replay:
session.audit_pinned(&live)  // Err(DanglingPointer) if pinned id dropped
captured.replay()
```

## The 11 CPU op families (Phase 1 reference)

| Op | Path | Migration target |
|---|---|---|
| `embedding(weight, ids)` | `ops::embedding` | `Tensor::index_select` axis 0 |
| `index_select(x, axis, ids)` | `ops::index_select` | `Tensor::index_select` any axis |
| `rms_norm(x, weight, eps)` | `ops::rmsnorm` | candle's `RmsNorm` |
| `add` / `sub` / `mul` / `div` | `ops::elementwise` | `Tensor::{add,sub,mul,div}` |
| `silu` / `sigmoid` | `ops::activation` | `Tensor::{silu, sigmoid}` |
| `softmax_last_dim(x)` | `ops::softmax` | `candle_nn::ops::softmax_last_dim` |
| `matmul(a, b)` | `ops::matmul` | `Tensor::matmul` |
| `argmax_last_dim(x)` | `ops::argmax` | `Tensor::argmax_keepdim(-1)` |
| `cast(x, dtype)` | `ops::cast` | `Tensor::to_dtype` |
| `rope(x, cos, sin, rotary_dim)` | `ops::rope` | `candle_nn::rotary_emb::rope` |
| `l2_norm(x, eps)` | `ops::l2norm` | `Tensor::l2_normalize(-1)` |
| `mul_sigmoid_gate(gate, up)` | `ops::silu_mul` | fused silu*mul from MLP work |
| `sum_all` / `mean_all` / `sum_axis` / `mean_axis` | `ops::reduce` | reductions |
| `masked_fill(x, mask, value)` | `ops::mask` | pre-softmax masking |
| `causal_mask(seq_len)` | `ops::mask` | `Tensor::tril` / `triu` equivalent |

Each op:

- Implements `DeviceOp1`, `DeviceOp2`, or `DeviceOp3`
- Provides `cpu_fwd` (mandatory, canonical reference)
- Has default-`None` `cuda_fwd` / `metal_fwd` / `vulkan_fwd`
- Declares `name()` + `determinism()` for the parity-tolerance + Phase 9 audit
- Returns `bwd() -> Option<Box<dyn BackwardOp>>` (None today; Phase 6b/c
  fills in)

## Adding a new op

1. Pick `DeviceOp1` / `DeviceOp2` / `DeviceOp3` based on arity.
2. Create `crates/kiln-tensor/src/ops/<name>.rs`.
3. Implement the struct + the trait methods.
4. Add a convenience free function (`pub fn my_op(...) -> Result<Tensor>`)
   that wraps `dispatch{1,2,3}`.
5. Wire into `crates/kiln-tensor/src/ops/mod.rs` (module + pub use).
6. Write 6-10 unit tests at minimum:
   - Happy path with exact known arithmetic
   - BF16 path with ULP tolerance (use `1e-2` for elementwise / reductions)
   - Multi-dim shape preservation
   - Each validation error path
   - Op metadata (name + determinism)
7. Add a row to `scripts/audit-substrate-status.sh`'s ROWS table.
8. Update `bench-results/parity-tolerance.csv` with the new
   op × dtype × backend cells if the op introduces a new category.

## Adding a per-backend Allocator impl

1. New file `crates/kiln-tensor/src/cuda_allocator.rs` (or metal /
   vulkan).
2. Feature-gate with `#[cfg(feature = "cuda")]`.
3. Impl `kiln_tensor::Allocator`. Mirror `CpuAllocator`'s structure
   (mode + cache + bytes tracking).
4. Add `cuda_zeros`-style convenience constructors that route
   through the allocator.
5. Tests: gate on `KILN_TENSOR_CUDA_TEST=1` env + a real GPU context;
   silent skip otherwise (the existing CudaStorage tests follow
   this pattern).

## Adding a per-backend CapturedGraph impl

Lives in a new crate `kiln-graph-cuda` / `kiln-graph-metal` /
`kiln-graph-vulkan`. The trait is `kiln_graph::CapturedGraph`. Each
impl wraps:

- CUDA: `cudarc::driver::CudaGraphExec`
- Metal: `MTLIndirectCommandBuffer` + `MTLBinaryArchive` for AOT cache
- Vulkan: secondary command buffer + `cmd_batch.rs` extension

## Anti-patterns: read these before touching the substrate

The full list is in
[#1082](https://github.com/ericflo/kiln/issues/1082). Top-of-mind:

1. **`kiln-tensor` is not a candle wrapper.** Storage is
   `cudarc::CudaSlice` / `metal::Buffer` / `ash::vk::Buffer` directly.
   No `candle_core::Tensor` field anywhere.
2. **Every `contiguous()` is logged.** `Tensor::contiguous()` calls
   `profile::emit_contiguous_copy()` on the materializing branch;
   the bench-gate's "copies per token" metric reads this counter.
4. **The `BackendRuntime` trait is the seam.** `kiln-tensor` slots in
   *below* it. Don't restructure `forward.rs` during the migration.
5. **No "big rewrite" PR.** Every phase is many small mergeable PRs
   (this substrate landed in ~60).
10. **`narrow` / `reshape` / `transpose` / `slice` are zero-copy.**
    Downstream kernels declare stride support in their `supports_*`
    function — don't silently `.contiguous()` to satisfy a kernel.
11. **One Parameter, one TensorId — stable across variants.**
    Forward-quantized + backward-master + transposed cache + LoRA
    delta all live behind one `Parameter` keyed on one
    `kiln_tensor::TensorId`. Verified by
    `crates/kiln-optim/tests/integration.rs::adamw_lora_swap_preserves_optimizer_state`.
13. **NVTX range names are part of the trace contract.** Don't rename
    `kiln/gdn/in_proj` etc. without explicit deliberation; PROFILING.md
    hot-region percentages stay comparable across the migration.
16. **In-place mutation invalidates the tape.** Call
    `tensor.bump_version()` on every in-place mutation; the tape walker
    asserts. Verified by
    `crates/kiln-autograd/tests/end_to_end.rs::backward_anti_pattern_16_detection_end_to_end`.
19. **Specialize to Qwen3.5-4B; generality is a perf cost.** Hot-path
    APIs that take `hidden_dim` / `intermediate_dim` / `vocab_size` as
    runtime parameters when fixed for Qwen3.5-4B are code smell.

## Next-step PR shapes

These are good first contributions:

- **Per-backend `Allocator` impl.** Today's `CpuAllocator` is the
  reference; `CudaAllocator` (cudaMemPool_t), `MetalAllocator`
  (MTLHeap), `VulkanAllocator` (lift `buffer_pool.rs`) are
  straight-line ports.
- **Per-op CUDA forward.** Pick an op family from the table above
  whose CUDA path needs implementing (most do); write a
  `kiln_tensor::ops::<op>::cuda_fwd` impl. Cite the parity-tolerance
  row in your PR.
- **CustomOpN porting.** Each of the 15 sites from Phase 0.2's audit
  (`bench-results/customop-audit.csv`) ports onto the new
  `DeviceOp` shape. Three of the ports already exist as forward-only
  scaffolds; the others need fwd + bwd.
- **Per-backend `CapturedGraph` impl.** Create `kiln-graph-cuda` /
  `-metal` / `-vulkan`. The Phase 5 issue text and
  `crates/kiln-graph/src/captured_graph.rs` trait doc are the
  starting points.
- **kiln-train migration.** Migrate one
  `crates/kiln-train/src/trainer.rs::AdamWMoments` HashMap entry off
  candle's TensorId onto `kiln_tensor::TensorId` + `kiln-optim::AdamW`.
  Behind a feature flag (`KILN_USE_KILN_OPTIM`).

## Where to ask

- Slack / dev channel: tag `@kiln-substrate`
- The `#1082` issue thread for design questions

Happy hacking.
