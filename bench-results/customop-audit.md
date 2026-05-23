# Phase 0.2 — CustomOpN audit

Sources of truth:

- `bench-results/customop-audit.csv` — 15 `impl CustomOpN` blocks

Regenerate: `scripts/audit-customop.py`.

Why this audit
--------------

candle's `CustomOp1` / `CustomOp2` / `CustomOp3` trait is the fifth (and largest unlisted) candle subsystem the migration must replace. Every fused kernel crate — `kiln-flce-kernel`, `kiln-opd-loss-kernel`, `kiln-gdn-kernel`, `kiln-rmsnorm-kernel`, `kiln-conv1d-kernel` — plugs into it for per-backend dispatch plus optional `bwd`. If the kiln-tensor `Op` shape is not defined before Phase 1 lands, the kernels port twice: once for the forward path and again for the backward.

## Per-crate breakdown

| crate | impls | has bwd | only cuda fwd | only metal fwd | all-backend |
|---|---:|---:|---:|---:|---:|
| kiln-flce-kernel | 1 | 1 | 0 | 0 | 1 |
| kiln-gdn-kernel | 2 | 0 | 0 | 0 | 0 |
| kiln-model | 9 | 9 | 0 | 0 | 0 |
| kiln-opd-loss-kernel | 1 | 1 | 0 | 0 | 0 |
| kiln-rmsnorm-kernel | 1 | 1 | 0 | 0 | 0 |
| kiln-train | 1 | 1 | 0 | 0 | 0 |

## All impls

| impl | arity | crate | file:line | bwd? | cpu/cuda/metal fwd | proposed shape |
|---|---:|---|---|:-:|:-:|---|
| `FlceCustomOp` | 1 | kiln-flce-kernel | crates/kiln-flce-kernel/src/phase_b.rs:185 | yes | `cum` | fwd+bwd-closure |
| `GdnGateBetaOp` | 1 | kiln-gdn-kernel | crates/kiln-gdn-kernel/src/lib.rs:1821 | no | `cu-` | closure-only |
| `GdnGateGOp` | 3 | kiln-gdn-kernel | crates/kiln-gdn-kernel/src/lib.rs:1876 | no | `cu-` | closure-only |
| `VulkanLinearOp` | 1 | kiln-model | crates/kiln-model/src/backend/vulkan_linear_op.rs:180 | yes | `c--` | fwd+bwd-closure |
| `VulkanLoraOp` | 3 | kiln-model | crates/kiln-model/src/backend/vulkan_lora_op.rs:67 | yes | `c--` | fwd+bwd-closure |
| `CudaLoraAddF32` | 3 | kiln-model | crates/kiln-model/src/forward.rs:801 | yes | `cu-` | fwd+bwd-closure |
| `CudaLoraLinearBf16` | 3 | kiln-model | crates/kiln-model/src/forward.rs:923 | yes | `cu-` | fwd+bwd-closure |
| `CudaLoraAddBf16` | 3 | kiln-model | crates/kiln-model/src/forward.rs:1134 | yes | `cu-` | fwd+bwd-closure |
| `CudaSigmoidMulTrainingBf16` | 2 | kiln-model | crates/kiln-model/src/forward.rs:1390 | yes | `cu-` | fwd+bwd-closure |
| `CudaFlashAttentionTrainingBf16` | 3 | kiln-model | crates/kiln-model/src/forward.rs:1790 | yes | `cu-` | fwd+bwd-closure |
| `VulkanRmsNormOp` | 1 | kiln-model | crates/kiln-model/src/forward.rs:4484 | yes | `c--` | fwd+bwd-closure |
| `CudaRotaryOneBf16` | 3 | kiln-model | crates/kiln-model/src/forward.rs:5116 | yes | `cu-` | fwd+bwd-closure |
| `OpdLossCustomOp` | 1 | kiln-opd-loss-kernel | crates/kiln-opd-loss-kernel/src/phase_b.rs:316 | yes | `cu-` | fwd+bwd-closure |
| `RmsNormCustomOp` | 2 | kiln-rmsnorm-kernel | crates/kiln-rmsnorm-kernel/src/lib.rs:812 | yes | `cu-` | fwd+bwd-closure |
| `InjectTensorGradient` | 1 | kiln-train | crates/kiln-train/src/trainer.rs:7599 | yes | `cu-` | fwd+bwd-closure |

## Proposed kiln-tensor shape

The kiln-tensor equivalent the kernels can port onto **once**:

```rust
// kiln-tensor crate.
pub trait DeviceOp: Send + Sync {
    /// Arity is fixed at the impl: `DeviceOp1` / `DeviceOp2` / `DeviceOp3`.
    /// Each per-backend method returns Option<...>; None means the op falls
    /// back to the next backend in the device's preference order, matching
    /// today's BackendRuntime contract.
    fn name(&self) -> &'static str;
    fn cpu_fwd  (&self, ...) -> Result<Option<Tensor>>;
    fn cuda_fwd (&self, ...) -> Result<Option<Tensor>>;
    fn metal_fwd(&self, ...) -> Result<Option<Tensor>>;
    fn vulkan_fwd(&self, ...) -> Result<Option<Tensor>>;
    /// Optional backward closure; absence == forward-only kernel.
    /// The boxed closure carries its own captured tensors;
    /// `kiln_autograd::BackwardOp::register` records the closure on
    /// the tape with the source tensor's TensorId.
    fn bwd(&self) -> Option<Box<dyn BackwardOp>>;
}
```

Three migration shapes (column `proposed_kiln_tensor_shape` in the CSV):

- `closure-only` — forward-only ops without `bwd`. Becomes a plain device-method on `kiln_tensor::Tensor`. Migration is mechanical.
- `fwd+bwd-closure` — has `bwd`. Becomes a `kiln_tensor::DeviceOp` plus a `kiln_autograd::BackwardOp` impl. The Vulkan path's `VkBackwardOp` (`vk_ops/`) is the lift template — 34 `impl VkBackwardOp for ...` blocks already follow this shape.
- `static-tape-op` — the existing bwd already routes through `candle_core::backprop::BackpropOp`. The tape op becomes a `kiln_autograd::Op` enum variant; the `bwd` method is its `apply` impl.

The audit is the input to Phase 1's `DeviceOp` API design and to Phase 6a's `kiln-autograd` crate skeleton.
