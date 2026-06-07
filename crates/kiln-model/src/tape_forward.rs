//! Experimental tape-forward path — CP-4 production-caller scaffolding (#1082).
//!
//! This module wires a `KILN_USE_TAPE_FORWARD`-gated branch into
//! `forward.rs` that routes a subset of forward-pass sites through the
//! `kiln_autograd::Tape` substrate (recording onto a thread-local
//! `Tape` instead of building a candle `BackpropOp` graph).
//!
//! # Why this exists
//!
//! The audit in
//! [`docs/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`]
//! documented why a per-call-site flip of `rms_norm` to
//! `fused_rmsnorm_via_kt_tape` cannot land at HEAD: the production
//! caller signature
//! `rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor>`
//! has no `&mut Tape` in scope and no caller transitively up to the
//! training step root has one either.
//!
//! The CP-4 substrate work in `kiln-train` (wave 11, commits
//! `51643ab` + `de647b8`) added the kt-tape primitives the production
//! forward will eventually need:
//!
//! * `tape_step::rms_norm_via_tape`
//! * `tape_step::matmul_via_tape`
//! * `tape_step::silu_via_tape`
//! * `tape_step::cross_entropy_via_tape`
//! * `tape_step::transformer_block_step_via_tape`
//!
//! …all parameterised on `&mut Tape`. To exercise the substrate end-to-end
//! from `kiln-model`'s forward path, this module provides a
//! **thread-local Tape** so existing forward functions can route through
//! tape-aware primitives without rewriting every signature up to the
//! training-step root.
//!
//! # Design
//!
//! 1. [`tape_forward_enabled`] reads `KILN_USE_TAPE_FORWARD` once and
//!    caches the result. Anything other than `0` / `false` / `no` /
//!    empty enables the path.
//! 2. [`with_thread_local_tape`] runs a closure with a fresh `Tape` as
//!    a thread-local. Tape-aware primitives can fetch the active tape
//!    via [`with_active_tape`] and record onto it.
//! 3. `forward::rms_norm` is the production kt-native recorder: it
//!    checks the env flag, borrows kt-tensors via `kiln_kt_bridge`,
//!    runs `fused_rmsnorm_via_kt_tape` with the active tape, copies
//!    the output back to a candle Tensor, and returns. When the gate
//!    is off, the env-tristate returns `None` and the caller falls
//!    through to the existing kt-forward-op shim.
//!
//! # Production-safety
//!
//! This is opt-in only. With `KILN_USE_TAPE_FORWARD` unset (the
//! default), `tape_forward_enabled()` returns `false` and every
//! `try_tape_*` helper short-circuits to `Ok(None)`. The existing
//! `fused_rmsnorm_via_kt_forward_op` path is untouched; the production
//! decode and training loops route through it exactly as before.
//!
//! # What this proves
//!
//! With `KILN_USE_TAPE_FORWARD=1`:
//!
//! * The forward path *can* be made to drive a `Tape`. The tape-
//!   recorded backward node is visible to a subsequent `Tape::backward`
//!   walk.
//! * The output tensor is bit-exact with the kt-forward-op shim
//!   (same kernel FFI call underneath; only the backward-graph
//!   machinery differs).
//! * Inside a `kiln_kt_bridge::tape_bridge::with_tape_scope_emit_to_grad_store`
//!   scope (see `kiln-train::trainer::standard_forward_backward_via_tape_bridge`,
//!   landed `675e0dea`), every adapter registers `(kt_id ↔ candle_id)`
//!   IO mappings via `register_input_mapping` / `register_output_mapping`.
//!   The bridge runs `loss.backward()` candle-side as usual, then walks
//!   the recorded tape with seeds derived from candle's `GradStore`, and
//!   merges the per-kt-input grads back into the same store keyed on the
//!   matched candle TensorIds. Result: callers downstream of the tape-
//!   routed primitives DO see correct grads in candle's `GradStore` even
//!   though the candle walker alone wouldn't traverse the tape op.
//!
//! # Current adapter coverage
//!
//! `try_tape_*_cuda` adapters land for every primitive whose
//! corresponding `kiln_autograd::backwards::*Backward` exists and whose
//! kt-side fused kernel ships a `*_via_kt_tape` entry:
//!
//! * `try_tape_matmul_kt` — `MatmulBackward`
//! * `try_tape_silu_kt` — `SiluBackward`
//! * `try_tape_embedding_kt` — `EmbeddingBackward`
//! * `try_tape_swiglu_kt` — `MulSigmoidGateBackward` (MLP gate path,
//!   ~18% of decode per Phase 6 NVTX profiling)
//!
//! All 5 register IO mappings into the bridge when a scope is active
//! (commits `57f7b678` for rms_norm/matmul/silu, `cf138c9c` for swiglu).
//!
//! # Out of scope (still)
//!
//! * Tape-routing for softmax / log_softmax / cross-entropy (the loss
//!   primitive — would close the kt-tape coverage end-to-end into the
//!   loss). Substrate primitives in `kiln_autograd::backwards::cross_entropy`
//!   exist; the adapter is straightforward but distinct from this PR.
//! * Tape-routing for rotary / layernorm / fused-attn — non-trivial
//!   substrate decisions about which kernels carry their own backward.

#![cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]

use anyhow::{Context, Result};
use kiln_autograd::{
    AddBackward, BackwardOp, ConcatBackward, CrossEntropyKtBackward, EmbeddingBackward,
    LoraDeltaAddBackward, MatmulBackward, MulBackward, MulSigmoidGateBackward, ReshapeBackward,
    RopeSplitHalfBackward, SiluBackward, Tape, TransposeBackward,
};

use crate::backend::BackendRuntime;
use crate::forward::{
    GDN_CHUNK_SIZE, gdn_gated_rms_norm_backward_no_grad, gdn_l2_norm_scale_backward_no_grad,
    gdn_recurrent_backward_no_grad, gdn_recurrent_forward_from_parts,
    sdpa_fallback_backward_no_grad,
};
use crate::lora_loader::LoraProjectionWeights;

// Phase 6a/CP-4 (#1082): the thread-local-tape scope machinery
// (`with_thread_local_tape`, `with_active_tape`, `tape_forward_enabled`)
// originally lived here. Wave-13 (#1082) promoted it into
// `kiln-autograd::tape_scope` so the OPD and FLCE kernel crates (and
// their `kiln-train` callers) can share the same thread-local handle
// without taking a `kiln-model` dependency. We re-export from here for
// back-compat — every existing call site (the parity test, the
// `forward.rs:7178` adapter call) keeps compiling unchanged.
pub use kiln_autograd::{tape_forward_enabled, with_active_tape, with_thread_local_tape};

/// (#1082) True iff a tape-recording scope is currently active on this thread.
///
/// Used by `forward.rs` to skip leaf, non-tape-recording backend kernels (e.g.
/// the Vulkan flash-attn prefill kernel, which also materializes a CPU-host
/// output at the kt<->vk seam) during a tape-authoritative training step, so the
/// forward falls through to the device-resident, tape-recording composite
/// (`try_tape_sdpa_fallback_kt`) instead. No-op outside a tape scope (inference
/// keeps the fast leaf kernel).
pub fn tape_scope_active() -> bool {
    with_active_tape(|_| ()).is_some()
}

/// kt-native SiLU tape recorder (#1082 seam flip) — the kt-native SiLU tape recorder. Takes the kt activation directly and records a
/// `SiluBackward` onto the active tape with **no candle round-trip** (no
/// `kt_logits_to_candle` in, no `kt_tensor_to_candle_cuda_copy` out, no IO-map
/// bookkeeping). Bottoms out in the same `kiln_tensor::ops::silu` +
/// `SiluBackward` as the candle adapter, so forward + backward are bit-identical
/// (guarded by `tape_forward_parity` + the SFT FD test). Chaining is preserved:
/// the returned kt is a recorded tape-node output, so a downstream candle bridge
/// re-registers it via `kt_logits_to_candle`'s `retain_output_for_chaining`, and
/// a downstream kt-native op consumes it directly.
///
/// `Ok(None)` when tape-forward is off or no tape scope is active (decode /
/// inference) — the caller falls through to the kt-native non-tape forward.
pub fn try_tape_silu_kt(x: &kiln_tensor::Tensor) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::silu(x).map_err(|e| anyhow::anyhow!("kt silu: {e}"))?;
        tape.record(&y, &[x], Box::new(SiluBackward { x: x.clone() }));
        Ok(y)
    }) {
        Some(result) => Ok(Some(result?)),
        None => Ok(None),
    }
}

/// kt-native residual-add tape recorder (#1082 seam flip) — kt-only twin of
/// `try_tape_add_cuda`. Records `AddBackward` directly from kt inputs, no candle
/// round-trip. `Ok(None)` on shape mismatch (defer broadcasting to the caller),
/// tape-off, or no active scope.
pub fn try_tape_add_kt(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    if a.dims() != b.dims() {
        return Ok(None);
    }
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::add(a, b).map_err(|e| anyhow::anyhow!("kt add: {e}"))?;
        tape.record(&y, &[a, b], Box::new(AddBackward));
        Ok(y)
    }) {
        Some(result) => Ok(Some(result?)),
        None => Ok(None),
    }
}

pub fn try_tape_concat_kt(
    inputs: &[&kiln_tensor::Tensor],
    axis: usize,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() || inputs.is_empty() || axis >= out.rank() {
        return Ok(None);
    }
    if !matches!(
        out.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }

    let dtype = inputs[0].dtype();
    if dtype != out.dtype() {
        return Ok(None);
    }
    let rank = out.rank();
    let mut axis_total = 0usize;
    let mut input_axis_sizes = Vec::with_capacity(inputs.len());
    let mut input_shapes = Vec::with_capacity(inputs.len());
    for input in inputs {
        if !matches!(
            input.device(),
            kiln_tensor::Device::Cuda(_)
                | kiln_tensor::Device::Metal(_)
                | kiln_tensor::Device::Vulkan(_)
                | kiln_tensor::Device::Rocm(_)
        ) || input.dtype() != dtype
            || input.rank() != rank
        {
            return Ok(None);
        }
        let dims = input.dims();
        if axis >= dims.len() {
            return Ok(None);
        }
        for (dim_idx, (&input_dim, &out_dim)) in dims.iter().zip(out.dims()).enumerate() {
            if dim_idx != axis && input_dim != out_dim {
                return Ok(None);
            }
        }
        axis_total = axis_total.saturating_add(dims[axis]);
        input_axis_sizes.push(dims[axis]);
        input_shapes.push(dims.to_vec());
    }
    if out.dims()[axis] != axis_total {
        return Ok(None);
    }

    let recorded = with_active_tape(|tape: &mut Tape| {
        tape.record(
            out,
            inputs,
            Box::new(ConcatBackward {
                axis,
                input_axis_sizes,
                input_shapes,
                dtype,
            }),
        );
    });
    if recorded.is_none() {
        return Ok(None);
    }
    Ok(Some(out.clone()))
}

/// kt-native split-half RoPE tape recorder (#1082 seam flip) — the kt-native RoPE tape recorder. Records `RopeSplitHalfBackward` directly from kt inputs,
/// no candle round-trip. `Ok(None)` outside the rank-4 envelope, tape-off, or no
/// active scope (caller falls through to the kt-native non-tape RoPE).
pub fn try_tape_rope_kt(
    x: &kiln_tensor::Tensor,
    cos: &kiln_tensor::Tensor,
    sin: &kiln_tensor::Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    if x.rank() != 4 || rotary_dim == 0 || rotary_dim > head_dim {
        return Ok(None);
    }
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::rope_split_half(x, cos, sin, rotary_dim)
            .map_err(|e| anyhow::anyhow!("kt rope_split_half: {e}"))?;
        tape.record(
            &y,
            &[x],
            Box::new(RopeSplitHalfBackward {
                rotary_dim,
                cos: cos.clone(),
                sin: sin.clone(),
            }),
        );
        Ok(y)
    }) {
        Some(result) => Ok(Some(result?)),
        None => Ok(None),
    }
}

/// Device-agnostic RMSNorm backward expressed as a composite of
/// `kiln_tensor::ops` (works on Metal/CUDA storage, unlike the CPU-only
/// `kiln_autograd::RmsNormBackward` whose hand-rolled loop requires
/// `CpuStorage`).
///
/// Uses the **unit-offset** RMSNorm convention `w_eff = 1 + weight` —
/// matching the production Qwen3.5 Metal/CUDA kernels
/// (`scale = 1.0f + weight[col]` in `backend::metal`). This matters: the
/// test models initialise norm weights to ZERO, so the no-offset
/// `x·w/r` form (`kiln_tensor::ops::rms_norm`) would zero the output AND
/// the input gradient — severing every in-block LoRA param from the loss.
///
/// Forward: `r = sqrt(mean_j(x_j²) + eps)`, `y_i = x_i * (1 + w_i) / r`.
/// Backward (per trailing-axis row of size `D`, with `w' = 1 + w`):
/// ```text
/// S    = Σⱼ dy_j * x_j * w'_j
/// dx_k = (dy_k * w'_k) / r  -  x_k * S / (D * r³)
/// dw_k = Σ_rows (dy_k * x_k) / r        (∂w'/∂w = 1)
/// ```
/// All intermediate math runs in F32 then casts back to the input dtype.
#[derive(Debug)]
struct RmsNormKtBackward {
    x: kiln_tensor::Tensor,
    weight: kiln_tensor::Tensor,
    eps: f32,
}

impl BackwardOp for RmsNormKtBackward {
    fn name(&self) -> &'static str {
        "rmsnorm_kt_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn requires_input(&self, idx: usize) -> bool {
        idx == 0 || idx == 1
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        use kiln_tensor::DType;
        use kiln_tensor::ops::{add_scalar, cast, mul, mul_scalar, sqrt, sum_axis};

        let dt = self.x.dtype();
        let shape = self.x.shape().to_vec();
        let last = shape.len() - 1;
        let hidden = shape[last];
        let d_inv = 1.0f32 / hidden as f32;

        let map = |e: kiln_tensor::Error| e;
        let xf = cast(&self.x, DType::F32).map_err(map)?;
        // Unit-offset convention: w' = 1 + weight.
        let wf = add_scalar(&cast(&self.weight, DType::F32).map_err(map)?, 1.0).map_err(map)?; // [hidden]
        let dyf = cast(grad_output, DType::F32).map_err(map)?;

        // r = sqrt(mean(x²)+eps), shape [..., 1] for broadcast.
        let xsq = mul(&xf, &xf).map_err(map)?;
        let mean_sq = mul_scalar(&sum_axis(&xsq, last).map_err(map)?, d_inv).map_err(map)?;
        let mut r_shape = shape.clone();
        r_shape[last] = 1;
        let r = sqrt(&add_scalar(&mean_sq, self.eps).map_err(map)?)
            .map_err(map)?
            .reshape(r_shape.clone())
            .map_err(map)?;
        let inv_r = kiln_tensor::ops::reciprocal(&r).map_err(map)?; // [..., 1]

        // dy * w (broadcast w over rows): [...] * [hidden].
        let dyw = dyf.broadcast_mul(&wf).map_err(map)?;
        // S = Σⱼ dy_j * x_j * w_j, shape [..., 1].
        let s = mul(&dyw, &xf)
            .and_then(|t| sum_axis(&t, last))
            .map_err(map)?
            .reshape(r_shape.clone())
            .map_err(map)?;

        // dx = dyw * inv_r - x * S * inv_r³ * (1/D).
        let term1 = dyw.broadcast_mul(&inv_r).map_err(map)?;
        let inv_r3 = mul(&inv_r, &mul(&inv_r, &inv_r).map_err(map)?).map_err(map)?;
        let s_scaled = mul_scalar(&mul(&s, &inv_r3).map_err(map)?, d_inv).map_err(map)?;
        let term2 = xf.broadcast_mul(&s_scaled).map_err(map)?;
        let dx_f = kiln_tensor::ops::sub(&term1, &term2).map_err(map)?;
        let dx = cast(&dx_f, dt).map_err(map)?;

        // dw = Σ_rows (dy * x) * inv_r, summed over all non-trailing axes -> [hidden].
        let dxw = mul(&dyf, &xf)
            .map_err(map)?
            .broadcast_mul(&inv_r)
            .map_err(map)?;
        let rows: usize = shape[..last].iter().product();
        let dxw_2d = dxw.reshape(vec![rows, hidden]).map_err(map)?;
        let dw_f = sum_axis(&dxw_2d, 0).map_err(map)?; // [hidden]
        let dw = cast(&dw_f, dt).map_err(map)?;

        Ok(vec![Some(dx), Some(dw)])
    }
}

/// kt-native RMSNorm tape recorder (#1082 Metal lane).
///
/// On CUDA, `forward::rms_norm` records the fused `CudaFusedRmsNormBackward`
/// via `kiln_rmsnorm_kernel::fused_rmsnorm_via_kt_tape`. That kernel crate is
/// CUDA-only, so on Metal the production `rms_norm` falls through to a
/// forward-only kernel that does NOT record — severing the tape between the
/// QK-norm and the attention/loss (LoRA grads come back empty).
///
/// This adapter closes that seam with the device-agnostic
/// `kiln_tensor::ops::rms_norm` forward + the device-agnostic
/// [`RmsNormKtBackward`] composite (the `kiln_autograd::RmsNormBackward` loop
/// is CPU-only), recorded against the connected kt `x.id()`. The RMSNorm
/// weight is FROZEN in LoRA training, so only the `dL/dx` edge matters; the
/// recorded node keeps the upstream LoRA-affected activations connected to the
/// loss. Returns `Ok(None)` (caller falls through to the forward-only path)
/// outside a tape scope or out of the BF16/F32 contiguous CUDA/Metal envelope.
pub fn try_tape_rms_norm_kt(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f32,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    if !matches!(
        x.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) || !matches!(
        weight.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }
    // Mixed-precision RMSNorm exceptions are backend precision policy, not a
    // local Vulkan special case. Today the policy allows F32 activations with
    // BF16 norm weights for Vulkan's mixed F32/BF16 training envelope.
    let precision_policy = crate::backend::TrainingPrecisionPolicy::for_device_family(x.device());
    if weight.rank() != 1
        || x.rank() == 0
        || *x.shape().last().unwrap() != weight.shape()[0]
        || !matches!(
            x.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
        )
        || !precision_policy
            .supports_rms_norm_weight_dtype_for_activation(x.dtype(), weight.dtype())
        || !x.is_contiguous()
        || !weight.is_contiguous()
    {
        return Ok(None);
    }
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        // Unit-offset convention (`w' = 1 + weight`) to match the production
        // Qwen3.5 Metal/CUDA RMSNorm kernels. `ops::rms_norm` computes
        // `x·w/r`, so feed it `1 + weight` (cast to the input dtype) to get
        // `x·(1+weight)/r`. The recorded backward applies the same offset.
        let one_plus_w = kiln_tensor::ops::scalar::add_scalar(
            &kiln_tensor::ops::cast(weight, x.dtype())
                .map_err(|e| anyhow::anyhow!("kt rms_norm weight cast: {e}"))?,
            1.0,
        )
        .map_err(|e| anyhow::anyhow!("kt rms_norm 1+weight: {e}"))?;
        let y = kiln_tensor::ops::rms_norm(x, &one_plus_w, eps)
            .map_err(|e| anyhow::anyhow!("kt rms_norm: {e}"))?;
        tape.record(
            &y,
            &[x, weight],
            Box::new(RmsNormKtBackward {
                x: x.clone(),
                weight: weight.clone(),
                eps,
            }),
        );
        Ok(y)
    }) {
        Some(result) => Ok(Some(result?)),
        None => Ok(None),
    }
}

/// kt-native matmul tape recorder (#1082 seam flip) — the kt-native matmul tape recorder. Records `MatmulBackward` directly from kt inputs.
pub fn try_tape_matmul_kt(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::matmul(a, b).map_err(|e| anyhow::anyhow!("kt matmul: {e}"))?;
        let saved_a = maybe_offload_matmul_a_saved_tensor(a)
            .context("try_tape_matmul_kt: save matmul lhs")?;
        tape.record(
            &y,
            &[a, b],
            Box::new(MatmulBackward {
                a: saved_a,
                b: b.clone(),
            }),
        );
        Ok(y)
    }) {
        Some(result) => Ok(Some(result?)),
        None => Ok(None),
    }
}

/// Tape backward for the Vulkan F32-act × BF16-weight mixed-precision matmul
/// (#1443 step1). The frozen base weight is in the transposed `[N, K]` BF16
/// layout; forward computed `out = x @ W.T` (`x` F32 `[rows, K]`). Backward
/// produces only `dx = grad_out @ W` (F32 `[rows, K]`) via the dedicated
/// `vk_matmul_bf16w_bwd` kernel — **there is no `dW`**, the weight is frozen.
///
/// # Why a bespoke `BackwardOp` (not the device-agnostic `MatmulBackward`)
///
/// `kiln_autograd::MatmulBackward` expresses `da`/`db` as
/// `kiln_tensor::ops::matmul`, which requires **equal input dtypes**. With an
/// F32 `grad_out` and a BF16 weight that op cannot run; we route `dx` through
/// the mixed-precision `vk_matmul_bf16w` kernel instead. `input_count() == 1`
/// because only `x` is differentiable (the weight is not recorded as an input).
#[cfg(feature = "vulkan")]
#[derive(Debug)]
pub(crate) struct MatmulBf16wBackward {
    /// Frozen BF16 base weight in transposed `[N, K]` layout. FROZEN — no `dW`.
    pub weight_t: kiln_tensor::Tensor,
}

#[cfg(feature = "vulkan")]
impl BackwardOp for MatmulBf16wBackward {
    fn name(&self) -> &'static str {
        "matmul_bf16w_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn requires_input(&self, _idx: usize) -> bool {
        // dx needs only grad_out + the (saved) weight; the forward `x`
        // activation is not read by the backward.
        false
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        // grad_output may arrive non-contiguous / non-F32 from upstream; the
        // kernel bridge requires contiguous F32. Materialize before dispatch.
        let go = if grad_output.dtype() == kiln_tensor::DType::F32 {
            grad_output.clone()
        } else {
            kiln_tensor::ops::cast(grad_output, kiln_tensor::DType::F32)?
        };
        let go = if go.is_contiguous() {
            go
        } else {
            go.contiguous()?
        };
        let dx = kiln_tensor::vulkan_matmul_bf16w_bwd(&go, &self.weight_t)?;
        Ok(vec![Some(dx)])
    }
}

/// kt-native Vulkan F32-act × BF16-weight matmul tape recorder (#1443 step1).
///
/// Records the mixed-precision base projection `out = x @ W.T` (F32 activation
/// × frozen BF16 weight in transposed `[N, K]` layout) onto the active kt
/// [`Tape`] with a [`MatmulBf16wBackward`] adjoint (`dx` only — the weight is
/// frozen). The forward runs on-device via
/// [`kiln_tensor::vulkan_matmul_bf16w`] (the `vk_matmul_bf16w` kernel bridge),
/// keeping the data GPU-resident.
///
/// This is the recorder the integration step (#1443 step2) routes the base
/// projection through when `x` is F32 and the base weight is BF16 on Vulkan —
/// the case the equal-dtype [`try_tape_lora_linear_kt`] base matmul declines
/// today (its `weight_t.dtype() == x.dtype()` gate). Only `x` is recorded as a
/// tape input (the weight is frozen, no grad edge).
///
/// `Ok(None)` (caller falls through) when tape-forward is off, no tape scope is
/// active, or the inputs are outside the envelope: Vulkan-resident, `x` rank-2
/// F32, `weight_t` rank-2 BF16, matching contraction dim `K`.
#[cfg(feature = "vulkan")]
pub fn try_tape_matmul_bf16w_kt(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    // Vulkan-only: CUDA/Metal carry their own BF16 base-weight paths.
    if !matches!(x.device(), kiln_tensor::Device::Vulkan(_))
        || !matches!(weight_t.device(), kiln_tensor::Device::Vulkan(_))
    {
        return Ok(None);
    }
    if x.dtype() != kiln_tensor::DType::F32 || weight_t.dtype() != kiln_tensor::DType::BF16 {
        return Ok(None);
    }
    if x.rank() != 2 || weight_t.rank() != 2 {
        return Ok(None);
    }
    if x.shape()[1] != weight_t.shape()[1] {
        return Ok(None);
    }
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::vulkan_matmul_bf16w(x, weight_t)
            .map_err(|e| anyhow::anyhow!("kt vulkan_matmul_bf16w: {e}"))?;
        tape.record(
            &y,
            &[x],
            Box::new(MatmulBf16wBackward {
                weight_t: weight_t.clone(),
            }),
        );
        Ok(y)
    }) {
        Some(result) => Ok(Some(result?)),
        None => Ok(None),
    }
}

/// kt-native embedding tape recorder (#1082 seam flip) — the kt-native embedding tape recorder. Records `EmbeddingBackward` directly from kt
/// inputs. `Ok(None)` outside the rank-2-weights envelope.
pub fn try_tape_embedding_kt(
    weights: &kiln_tensor::Tensor,
    token_ids: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    if weights.shape().len() != 2 || token_ids.shape().is_empty() {
        return Ok(None);
    }
    let vocab_size = weights.shape()[0];
    let hidden = weights.shape()[1];
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::embedding(weights, token_ids)
            .map_err(|e| anyhow::anyhow!("kt embedding: {e}"))?;
        tape.record(
            &y,
            &[weights, token_ids],
            Box::new(EmbeddingBackward {
                vocab_size,
                hidden,
                token_ids: token_ids.clone(),
            }),
        );
        Ok(y)
    }) {
        Some(result) => Ok(Some(result?)),
        None => Ok(None),
    }
}

/// kt-native SwiGLU (gate ⊙ sigmoid-gate of up) tape recorder (#1082 seam flip) —
/// the kt-native SwiGLU tape recorder.
pub fn try_tape_swiglu_kt(
    gate: &kiln_tensor::Tensor,
    up: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::mul_sigmoid_gate(gate, up)
            .map_err(|e| anyhow::anyhow!("kt mul_sigmoid_gate: {e}"))?;
        tape.record(
            &y,
            &[gate, up],
            Box::new(MulSigmoidGateBackward {
                gate: gate.clone(),
                up: up.clone(),
            }),
        );
        Ok(y)
    }) {
        Some(result) => Ok(Some(result?)),
        None => Ok(None),
    }
}

/// kt-native elementwise-mul tape recorder (#1082 seam flip) — kt-only twin of
/// `try_tape_mul_cuda`. `Ok(None)` on shape mismatch (defer broadcasting).
pub fn try_tape_mul_kt(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    if a.dims() != b.dims() {
        return Ok(None);
    }
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::mul(a, b).map_err(|e| anyhow::anyhow!("kt mul: {e}"))?;
        tape.record(
            &y,
            &[a, b],
            Box::new(MulBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        );
        Ok(y)
    }) {
        Some(result) => Ok(Some(result?)),
        None => Ok(None),
    }
}

/// kt-native transpose tape recorder (#1082 seam flip) — kt-only twin of
/// `try_tape_transpose_cuda`. Materialises the transposed view contiguous
/// (the `TransposeBackward` adjoint transposes the upstream grad regardless).
pub fn try_tape_transpose_kt(
    x: &kiln_tensor::Tensor,
    axis_a: usize,
    axis_b: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    if !matches!(
        x.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }
    let rank = x.rank();
    if axis_a >= rank || axis_b >= rank {
        return Ok(None);
    }
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = x
            .transpose(axis_a, axis_b)
            .map_err(|e| anyhow::anyhow!("kt transpose: {e}"))?;
        let y = if y.is_contiguous() {
            y
        } else {
            y.contiguous()
                .map_err(|e| anyhow::anyhow!("kt transpose: contiguous: {e}"))?
        };
        tape.record(&y, &[x], Box::new(TransposeBackward { axis_a, axis_b }));
        Ok(y)
    }) {
        Some(result) => Ok(Some(result?)),
        None => Ok(None),
    }
}

/// kt-native reshape tape recorder (#1082 seam flip) — kt-only twin of
/// `try_tape_reshape_cuda`. The adjoint reshapes the upstream grad back to the
/// original input shape. `Ok(None)` on element-count mismatch.
pub fn try_tape_reshape_kt(
    x: &kiln_tensor::Tensor,
    new_shape: Vec<usize>,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    if !matches!(
        x.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }
    let input_shape = x.shape().to_vec();
    let in_elems: usize = input_shape.iter().product();
    let out_elems: usize = new_shape.iter().product();
    if in_elems != out_elems {
        return Ok(None);
    }
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let x_c = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()
                .map_err(|e| anyhow::anyhow!("kt reshape: x.contiguous: {e}"))?
        };
        let out_kt = x_c
            .reshape(new_shape.clone())
            .map_err(|e| anyhow::anyhow!("kt reshape: {e}"))?;
        tape.record(
            &out_kt,
            &[x],
            Box::new(ReshapeBackward {
                input_shape: input_shape.clone(),
            }),
        );
        Ok(out_kt)
    }) {
        Some(result) => Ok(Some(result?)),
        None => Ok(None),
    }
}

/// kt-NATIVE backward for the fused "cross-entropy from full logits" loss node
/// ([`try_tape_cross_entropy_from_logits_kt`]).
///
/// # Why kt-native (vs the candle-copy [`CrossEntropyFromLogitsBackward`])
///
/// The earlier [`CrossEntropyFromLogitsBackward`] saved the FULL `[1, T, V]`
/// logits as a candle `Tensor` because the forward adapter
/// (`try_tape_cross_entropy_from_logits_cuda`) took candle logits — which in
/// the tape-authoritative SFT path meant `cross_entropy_loss` first copied the
/// kt lm_head logits into a candle `[1, T, V]` tensor (≈150 MB+/step for real
/// shapes) purely so the candle adapter could re-borrow them. That copy was
/// pure waste: the saved candle logits are immediately re-bridged to kt inside
/// `apply` for the (already kt-native) analytic backward
/// [`crate::forward::cross_entropy_from_logits_grad_candle`]. (#1082 H6)
///
/// This struct saves the kt logits DIRECTLY (an `Arc` bump on the kt storage —
/// no device copy), plus the host-side `input_ids` / `label_mask`. The backward
/// calls the SAME analytic kt grad function with no candle round-trip.
///
/// # Saved tensors
///
/// `logits` is the FULL forward logits `[1, T, V]` as a kt tensor (a cheap
/// clone — an `Arc` bump on the kt storage), plus the host-side `input_ids` /
/// `label_mask`. The analytic grad recomputes the forward gather + softmax from
/// these (no extra device tensors saved).
///
/// # Gradient
///
/// See [`crate::forward::cross_entropy_from_logits_grad_candle`] (a misnomer —
/// it is kt-native: kt input, kt output): mean reduction (`1/num_active`), the
/// `softmax - one_hot` per-active-row term scaled by the incoming scalar seed,
/// scattered back to the active shifted rows with a trailing zero row for the
/// dropped `lg[T-1]`. Returned as a single `[1, T, V]` kt grad (input count 1).
#[derive(Debug)]
pub(crate) struct CrossEntropyFromLogitsKtBackward {
    /// FULL forward logits `[1, T, V]` (kt clone — an `Arc` bump, no copy).
    logits: kiln_tensor::Tensor,
    input_ids: Vec<u32>,
    label_mask: Vec<bool>,
}

impl BackwardOp for CrossEntropyFromLogitsKtBackward {
    fn name(&self) -> &'static str {
        "cross_entropy_from_logits_kt_backward"
    }
    fn input_count(&self) -> usize {
        // The full logits [1, T, V].
        1
    }
    fn requires_input(&self, _idx: usize) -> bool {
        // The analytic grad recomputes the forward gather from the SAVED kt
        // `logits`; the tape walker need not re-materialise the input activation.
        false
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        // The upstream grad is the scalar dL/dloss seed (typically 1.0). Read the
        // scalar so the analytic grad can fold it into the mean-reduction's
        // per-row gradient (backward is linear in the seed). Pure kt — no candle.
        let grad_scalar = grad_output
            .to_dtype(kiln_tensor::DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| {
                kiln_tensor::Error::Msg(format!(
                    "CrossEntropyFromLogitsKtBackward: grad scalar read: {e}"
                ))
            })?
            .first()
            .copied()
            .ok_or_else(|| {
                kiln_tensor::Error::Msg(
                    "CrossEntropyFromLogitsKtBackward: empty grad_output".to_string(),
                )
            })? as f64;

        // `cross_entropy_from_logits_grad_candle` is kt-native (kt logits in,
        // kt `[1, T, V]` grad out) — its `(softmax - one_hot) * (grad_scalar/A)`
        // is EXACTLY the CE-from-logits gradient. No candle bridge: the kt logits
        // are passed straight in.
        let grad_logits = crate::forward::cross_entropy_from_logits_grad_candle(
            &self.logits,
            &self.input_ids,
            &self.label_mask,
            grad_scalar,
        )
        .map_err(|e| {
            kiln_tensor::Error::Msg(format!(
                "CrossEntropyFromLogitsKtBackward: kt analytic grad: {e}"
            ))
        })?;

        // The analytic grad output is freshly built (cat/unsqueeze → possibly
        // non-contiguous) — materialise contiguous as an owned kt tensor.
        let grad_logits_kt = grad_logits.contiguous().map_err(|e| {
            kiln_tensor::Error::Msg(format!(
                "CrossEntropyFromLogitsKtBackward: grad contiguous: {e}"
            ))
        })?;

        Ok(vec![Some(grad_logits_kt)])
    }
}

/// kt-NATIVE variant of `try_tape_cross_entropy_from_logits_cuda`: roots the
/// WHOLE next-token cross-entropy loss at a SINGLE fused kt `Tape` node taking
/// the FULL `[1, T, V]` kt logits DIRECTLY — with NO `[1, T, V]` kt -> candle
/// copy.
///
/// # Why this exists (#1082 H6)
///
/// The candle-typed sibling `try_tape_cross_entropy_from_logits_cuda` takes
/// candle logits, so in the tape-authoritative SFT path `cross_entropy_loss`
/// first bridged the kt lm_head logits into a full `[1, T, V]` candle tensor
/// (≈150 MB+/step) just so the candle adapter could re-borrow them via
/// `tape_kt_input` — immediately resolving them back to kt. That copy is pure
/// waste. This entry takes the kt logits straight from the trainer (the kt
/// lm_head output), runs the CE forward in kt, and records a kt-native
/// [`CrossEntropyFromLogitsKtBackward`] node against them — only the resulting
/// SCALAR loss crosses back to candle (a tiny bridge for the `candle_core::Tensor`
/// return type the candle-island `cross_entropy_loss` still has).
///
/// # Forward (mirrors `cross_entropy_loss`, kt-native)
///
/// ```text
/// lg        = logits.squeeze(0)            [T, V]
/// shift     = lg.narrow(0, 0, T-1)         [T-1, V]   (predict token[i+1] from logit[i])
/// active    = shift.index_select(active_positions, 0)   [A, V]
/// active32  = active.to_dtype(F32)
/// lse       = active32.log_sum_exp(last)   [A]
/// correct   = active32.flatten[a*V + y_a]  [A]   (FLAT index_select; kt `gather` is CPU-only)
/// loss      = mean_a( lse[a] - correct[a] )         (scalar; matches the candle .mean_all())
/// ```
///
/// where `active_positions = { i in 0..T-1 : label_mask[i+1] }`, `A = num_active`,
/// `y_a = input_ids[active_positions[a] + 1]`. This is numerically identical to
/// `cross_entropy_loss`'s candle baseline (same shift / active-set convention as
/// `crate::trainer::token_log_probs`).
///
/// # Returns
///
/// * `Ok(Some(loss))` — the tape-forward path ran: a DETACHED, lineage-free
///   candle scalar loss (a fresh kt -> candle CUDA copy of the kt scalar loss,
///   numerically identical to the candle baseline) with a
///   [`CrossEntropyFromLogitsKtBackward`] node recorded on the active tape,
///   IO-mapped into the bridge. The loss is detached unconditionally so the
///   tape-authoritative caller's `loss.backward()` is always `{loss: ones}` and
///   the recorded node is the sole backward root.
/// * `Ok(None)` — the gate is off, `logits` isn't a CUDA rank-3 `[1, T, V]`, no
///   tape scope is active, or an empty active set. The caller falls through to
///   the candle loss composite.
/// * `Err(...)` — an unexpected forward or kt -> candle scalar copy-back failure.
pub fn try_tape_cross_entropy_from_logits_kt(
    logits: &kiln_tensor::Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
) -> Result<Option<kiln_tensor::Tensor>> {
    use kiln_tensor::{DType as KtDType, Tensor as KtTensor};

    if !tape_forward_enabled() {
        return Ok(None);
    }

    // Full model logits only: [1, T, V] on CUDA. Defer any other shape/device to
    // the caller's candle composite.
    let dims = logits.dims().to_vec();
    if dims.len() != 3
        || dims[0] != 1
        || dims[1] != input_ids.len()
        || label_mask.len() != input_ids.len()
        || !matches!(
            logits.device(),
            kiln_tensor::Device::Cuda(_)
                | kiln_tensor::Device::Metal(_)
                | kiln_tensor::Device::Vulkan(_)
                | kiln_tensor::Device::Rocm(_)
        )
    {
        return Ok(None);
    }
    let seq_len = input_ids.len();
    if seq_len < 2 {
        return Ok(None);
    }
    let vocab = dims[2];

    // Active next-token positions in the SHIFTED frame (== seq positions in
    // logits[0 .. T-1]): { i : label_mask[i+1] }. Same set/order as
    // `crate::trainer::token_log_probs` builds from `shift_mask = mask[1..]`.
    let active_positions: Vec<usize> = label_mask
        .get(1..)
        .map(|shift_mask| {
            shift_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &m)| if m { Some(i) } else { None })
                .collect()
        })
        .unwrap_or_default();
    if active_positions.is_empty() {
        // No supervised positions — let the caller's composite raise the same
        // error it would have raised, rather than bailing here.
        return Ok(None);
    }
    let num_active = active_positions.len();

    let device = logits.device(); // kt `Device` is returned by value (Copy)

    // kt input — thread the lm_head adapter's output so the tape stays connected
    // (consumer input id == producer output id). The trainer already chains the
    // kt logits id into the bridge before calling, so `tape_bridge::kt_input_for_*`
    // resolves it; fall back to the logits as-is (it IS the kt lm_head output).
    let logits_kt = logits.clone();

    // --- Forward: scalar CE loss, kt-native (no candle [1, T, V] copy). ---
    let shift_logits = logits_kt
        .squeeze(0) // [T, V]
        .and_then(|t| t.narrow(0, 0, seq_len - 1)) // [T-1, V]
        .context("try_tape_cross_entropy_from_logits_kt: shift_logits")?;
    let active_idx_u32: Vec<u32> = active_positions.iter().map(|&i| i as u32).collect();
    let active_idx = KtTensor::from_vec_on(device, active_idx_u32, vec![num_active])
        .map_err(|e| anyhow::anyhow!("try_tape_cross_entropy_from_logits_kt: active_idx: {e}"))?;
    let active_logits_f32 = shift_logits
        .index_select(&active_idx, 0)
        .and_then(|t| t.to_dtype(KtDType::F32)) // [A, V] F32
        .map_err(|e| {
            anyhow::anyhow!("try_tape_cross_entropy_from_logits_kt: active_logits: {e}")
        })?;
    let log_sum_exp = active_logits_f32
        .log_sum_exp(kiln_tensor::D::Minus1) // [A]
        .map_err(|e| anyhow::anyhow!("try_tape_cross_entropy_from_logits_kt: log_sum_exp: {e}"))?;

    // correct_logits[a] = active_logits_f32[a, y_a]. kt `gather` is CPU-only, so
    // select via a FLAT `index_select` (CUDA-capable) at a*vocab + y_a — exactly
    // as `crate::trainer::token_log_probs` does.
    let mut flat_idx: Vec<u32> = Vec::with_capacity(num_active);
    for (a, &p) in active_positions.iter().enumerate() {
        let label = input_ids[p + 1] as usize;
        anyhow::ensure!(
            label < vocab,
            "try_tape_cross_entropy_from_logits_kt: label {label} (pos {p}) >= vocab {vocab}"
        );
        flat_idx.push((a * vocab + label) as u32);
    }
    let flat_indices = KtTensor::from_vec_on(device, flat_idx, vec![num_active])
        .map_err(|e| anyhow::anyhow!("try_tape_cross_entropy_from_logits_kt: flat_idx: {e}"))?;
    let correct_logits = active_logits_f32
        .contiguous()
        .and_then(|t| t.flatten_all()) // [A*V]
        .and_then(|t| t.index_select(&flat_indices, 0)) // [A]
        .map_err(|e| {
            anyhow::anyhow!("try_tape_cross_entropy_from_logits_kt: correct_logits: {e}")
        })?;

    // loss = mean_a( log_sum_exp[a] - correct_logits[a] ).
    let per_token_loss = log_sum_exp
        .sub(&correct_logits)
        .map_err(|e| anyhow::anyhow!("try_tape_cross_entropy_from_logits_kt: per_token: {e}"))?;
    let loss_kt_forward = per_token_loss
        .mean_all()
        .map_err(|e| anyhow::anyhow!("try_tape_cross_entropy_from_logits_kt: mean_all: {e}"))?;

    // Record the fused node: the OUTPUT is the OWNED kt scalar loss (it carries
    // no candle lineage and has independent kt storage, so it does not dangle).
    // The single differentiable input is the CONNECTED kt logits, so the recorded
    // `CrossEntropyFromLogitsKtBackward` node roots `dL/d(logits)` directly at the
    // lm_head kt output — no candle id-mapping dance.
    let loss_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let loss_kt = loss_kt_forward;
        tape.record(
            &loss_kt,
            &[&logits_kt],
            Box::new(CrossEntropyFromLogitsKtBackward {
                logits: logits_kt.clone(),
                input_ids: input_ids.to_vec(),
                label_mask: label_mask.to_vec(),
            }) as Box<dyn BackwardOp>,
        );
        Ok(loss_kt)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };
    let loss_kt = loss_kt
        .context("tape_forward::try_tape_cross_entropy_from_logits_kt: kt-tape forward failed")?;

    // (#1082 DoD-100 keystone) Return the KT scalar loss DIRECTLY — no kt->candle
    // copy, no candle-id output mapping. The kt loss IS the recorded tape root;
    // `with_tape_authoritative_scope_kt` seeds it directly. (The candle-typed
    // `cross_entropy_loss` wrapper bridges this to candle for its value-only
    // callers; the SFT authoritative path takes the kt loss straight.)
    Ok(Some(loss_kt))
}

/// Attempt to run the LoRA delta-and-add through the kt-typed op surface
/// (`kiln_tensor::ops::{matmul, mul_scalar, add}`) and record a fused
/// `LoraDeltaAddBackward` node on the active thread-local tape.
///
/// The forward computes:
/// ```text
/// out = base + scale * (x @ A^T @ B^T)
/// ```
///
/// Returns:
/// * `Ok(Some(out))` — the tape-forward path ran. The returned candle
///   `Tensor` is a copy of the kt-typed output (reshaped back to
///   `base.shape()`); a `LoraDeltaAddBackward { x, a, b, scale }` node was
///   recorded on the active thread-local tape with inputs
///   `[base, x, A, B]` in that order.
/// * `Ok(None)` — gate off (no thread-local tape, `KILN_USE_TAPE_FORWARD`
///   off, `KILN_USE_TAPE_LORA_ADD` off, or device / dtype / shape /
///   contiguity preconditions fail). The caller must fall through to the
///   existing CUDA / Metal / Vulkan / candle dispatch.
/// * `Err(...)` — an unexpected kt forward, kt → candle copy-back, or
///   reshape failure. Propagated so callers see the failure cleanly
///   instead of silently masking it.
///
/// # CP-4 (#1082) context — closes LoRA Var grad coverage
///
/// Without this adapter, the production LoRA delta-add dispatch in
/// `add_lora_delta_to_base` lands in either `cuda_lora_add_training_f32`,
/// `cuda_lora_add_training_bf16`, `backend.lora_decode_add`, or the
/// Phase-4.1 `CustomOp3` path — none of which the kt `Tape` walker sees.
/// Under `KILN_USE_TAPE_AUTHORITATIVE`, the resulting candle `GradStore`
/// has no entries for the LoRA `Var`s (`proj.a`, `proj.b`), so the
/// optimiser step is a no-op for the adapter parameters. With this
/// adapter on, the fused backward emits grads for `proj.a` and `proj.b`
/// in their original `[rank, in_features]` / `[out_features, rank]`
/// shapes, and the IO mapping pairs each kt input id with the Var's
/// candle id so the parity gate sees nonzero matched LoRA grads.
///
/// # Why fused (not 4 chained tape nodes)
///
/// The LoRA delta needs `A^T` and `B^T`; the kt forward uses
/// `matmul_rhs_transposed` so it can consume the original `A` and `B`
/// layouts directly. Fusing the four ops into a single
/// `LoraDeltaAddBackward` keeps the per-input grads in the original Var
/// layouts so the IO mapping is direct:
/// `(a_kt.id(), proj.a.id())`, `(b_kt.id(), proj.b.id())`.
/// See `kiln_autograd::backwards::lora_delta_add` for the math
/// derivation.
#[allow(clippy::too_many_lines)]
/// kt-native LoRA delta-add tape recorder (#1082 seam flip) — kt-only twin of
/// `try_tape_lora_add_cuda`. base + x already kt at the call site, so the
/// composite (base->2d + x->2d reshapes, LoRA delta=(x2d@Aᵀ)@Bᵀ·scale, out=base2d+delta,
/// reshape back) runs in kt ops recording the SAME ReshapeBackward + LoraDeltaAddBackward
/// nodes (base/x reshapes recorded so their grads chain in kt — the candle adapter relied
/// on candle-id mappings + the bridge instead). LoRA-Var grad mapping preserved. (Reached
/// only when the fused linear+LoRA path declines — a rare fallback.)
pub fn try_tape_lora_add_kt(
    base: &kiln_tensor::Tensor,
    x: &kiln_tensor::Tensor,
    proj: &LoraProjectionWeights,
    lora_scale: f32,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() || !tape_lora_add_enabled() {
        return Ok(None);
    }
    if !matches!(
        base.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) || !matches!(
        x.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) || !matches!(
        proj.a.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) || !matches!(
        proj.b.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }
    if base.dtype() != x.dtype() {
        return Ok(None);
    }
    if !matches!(
        base.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    if proj.a.dtype() != base.dtype() || proj.b.dtype() != base.dtype() {
        return Ok(None);
    }
    let base_dims = base.dims().to_vec();
    let x_dims = x.dims().to_vec();
    if base_dims.len() < 2 || x_dims.len() != base_dims.len() {
        return Ok(None);
    }
    if base_dims[..base_dims.len() - 1] != x_dims[..x_dims.len() - 1] {
        return Ok(None);
    }
    let out_features = *base_dims.last().unwrap();
    let in_features = *x_dims.last().unwrap();
    let rows: usize = base_dims[..base_dims.len() - 1].iter().product();
    if rows == 0 {
        return Ok(None);
    }
    let Ok((rank, a_in)) = proj.a.dims2() else {
        return Ok(None);
    };
    let Ok((b_out, b_rank)) = proj.b.dims2() else {
        return Ok(None);
    };
    if a_in != in_features || b_out != out_features || b_rank != rank {
        return Ok(None);
    }
    let a_kt = proj.a.clone();
    let b_kt = proj.b.clone();
    if !a_kt.is_contiguous() || !b_kt.is_contiguous() {
        return Ok(None);
    }
    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let base_c = if base.is_contiguous() {
            base.clone()
        } else {
            base.contiguous()
                .map_err(|e| anyhow::anyhow!("kt base.contiguous: {e}"))?
        };
        let base_2d = base_c
            .reshape(vec![rows, out_features])
            .map_err(|e| anyhow::anyhow!("kt base reshape -> 2d: {e}"))?;
        tape.record(
            &base_2d,
            &[base],
            Box::new(ReshapeBackward {
                input_shape: base.shape().to_vec(),
            }),
        );
        let x_c = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()
                .map_err(|e| anyhow::anyhow!("kt x.contiguous: {e}"))?
        };
        let x_2d = x_c
            .reshape(vec![rows, in_features])
            .map_err(|e| anyhow::anyhow!("kt x reshape -> 2d: {e}"))?;
        tape.record(
            &x_2d,
            &[x],
            Box::new(ReshapeBackward {
                input_shape: x.shape().to_vec(),
            }),
        );
        let h_kt = kiln_tensor::ops::matmul_rhs_transposed(&x_2d, &a_kt)
            .map_err(|e| anyhow::anyhow!("kt matmul_rhs_transposed x@a_t: {e}"))?;
        let d_kt = kiln_tensor::ops::matmul_rhs_transposed(&h_kt, &b_kt)
            .map_err(|e| anyhow::anyhow!("kt matmul_rhs_transposed h@b_t: {e}"))?;
        let delta_kt = kiln_tensor::ops::mul_scalar(&d_kt, lora_scale)
            .map_err(|e| anyhow::anyhow!("kt mul_scalar(scale): {e}"))?;
        let out_2d = kiln_tensor::ops::add(&base_2d, &delta_kt)
            .map_err(|e| anyhow::anyhow!("kt add(base, delta): {e}"))?;
        tape.record(
            &out_2d,
            &[&base_2d, &x_2d, &a_kt, &b_kt],
            Box::new(LoraDeltaAddBackward {
                x: x_2d.clone(),
                a: a_kt.clone(),
                b: b_kt.clone(),
                scale: lora_scale,
            }),
        );
        let out2d_c = if out_2d.is_contiguous() {
            out_2d.clone()
        } else {
            out_2d
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt out2d.contiguous: {e}"))?
        };
        let out = out2d_c
            .reshape(base_dims.clone())
            .map_err(|e| anyhow::anyhow!("kt out reshape -> nd: {e}"))?;
        tape.record(
            &out,
            &[&out_2d],
            Box::new(ReshapeBackward {
                input_shape: vec![rows, out_features],
            }),
        );
        Ok(out)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };
    let out_kt = out_kt.context("tape_forward::try_tape_lora_add_kt: kt-tape forward failed")?;
    kiln_kt_bridge::tape_bridge::register_input_mapping_kt(a_kt.id(), proj.a.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping_kt(b_kt.id(), proj.b.id());
    Ok(Some(out_kt))
}

/// Attempt to run the FULL base projection (and optional fused LoRA delta)
/// through the kt-typed op surface as ONE chained group of `Tape` nodes:
/// `reshape → matmul → [lora_delta_add] → reshape`. Records the backward so
/// `dL/dx` flows through BOTH the frozen base weight AND the LoRA path, and
/// `proj.a` / `proj.b` receive grads.
///
/// The forward computes (matching the existing CUDA dispatch bit-for-bit):
/// ```text
/// out = base + scale * (x @ A^T @ B^T)        // when `lora` is Some
/// out = base = x @ W^T                          // when `lora` is None (lm_head)
/// ```
///
/// # Why a SINGLE fused adapter (base + LoRA together)
///
/// CP-4 (#1082) Increment 2 — the keystone. The tape root
/// (`try_tape_cross_entropy_from_logits_cuda`, Increment 1) connects to the
/// lm_head output, but the lm_head matmul and every q/k/v/o/gate/up/down/GDN
/// projection were unwired, so nothing below the loss got grads
/// (`tape_has_grad=0/50`). In the authoritative path every intermediate is a
/// DETACHED kt-copy (`track_op()==false`), so neither
/// `lm_head_forward_backend_decode_if` nor
/// `linear_with_lora_t_backend_decode_if` reliably hits `linear_prefill_apply`
/// (both gate the autograd-safe branch on `x.track_op()`). This adapter is
/// therefore wired at the TOP of those two functions, before the backend
/// dispatch.
///
/// Folding base + LoRA into one chained group (instead of routing the base
/// matmul and the LoRA delta-add through two separate adapters across a
/// reshape boundary) keeps `x2d` and `base2d` as SHARED kt ids: `base2d` is
/// the matmul node's output (so `dL/dbase2d` flows into the matmul backward),
/// and `x2d` is shared between the matmul node and the LoRA node (so `dL/dx2d`
/// accumulates BOTH the base-weight and LoRA contributions — the correct full
/// `dL/dx`). Splitting them would mint a fresh kt borrow at the reshape and
/// fragment the chain.
///
/// # Node-recording sequence (inside ONE `with_active_tape`)
///
/// 1. `ReshapeBackward { input_shape: x_kt.shape() }` — output `x2d [rows, k]`,
///    input `x_kt`.
/// 2. `MatmulBackward { a: x2d, b: w_kt }` — output `base2d [rows, n]`, inputs
///    `[x2d, w_kt]`.
/// 3. (lora only) `LoraDeltaAddBackward { x: x2d, a, b, scale }` — output
///    `out2d`, inputs `[base2d, x2d, a_kt, b_kt]` (same order as
///    `try_tape_lora_add_cuda`).
/// 4. `ReshapeBackward { input_shape: [rows, n] }` — output `out_kt`, input
///    `out2d` (or `base2d` when `lora` is None).
///
/// # Returns
///
/// * `Ok(Some(out))` — the tape-forward path ran: a candle copy of the kt
///   output, reshaped to `x.dims[..-1] ++ [n]`, with the chained group recorded
///   on the active tape and IO-mapped into the bridge (`x` → `x_kt`, and the
///   LoRA Vars `proj.a`/`proj.b` → `a_kt`/`b_kt` when present).
/// * `Ok(None)` — gate off (no tape, `KILN_USE_TAPE_FORWARD` off,
///   `KILN_USE_TAPE_LORA_ADD` off), or device / dtype / shape / contiguity
///   preconditions fail. The caller falls through to the existing dispatch.
/// * `Err(...)` — an unexpected kt forward or kt → candle copy-back failure.
#[allow(clippy::too_many_lines)]
/// kt-native linear+LoRA tape recorder (#1082 seam flip) — kt-only twin of
/// `try_tape_lora_linear_cuda`. x + base weight are already kt at the call site
/// (the loader produces kt base weights), so no candle bridging: the whole
/// composite (x->2d reshape, base = x2d@W, LoRA delta = (x2d@Aᵀ)@Bᵀ·scale + base,
/// reshape back) runs in kt ops with the SAME tape nodes (ReshapeBackward,
/// MatmulBackward, LoraDeltaAddBackward) the candle adapter records. The LoRA-Var
/// grad mapping (`register_input_mapping_kt`) is preserved so the optimiser sees
/// dA/dB; no candle-id mapping is needed for x/out (they chain in kt directly).
pub fn try_tape_lora_linear_kt(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() || !tape_lora_add_enabled() {
        return Ok(None);
    }
    if !matches!(
        x.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) || !matches!(
        weight_t.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }
    if !matches!(
        x.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    // (#1443 step 2) Mixed-precision base projection on Vulkan: F32 activations ×
    // a frozen BF16 base weight. The base linear runs through the dedicated
    // `vk_matmul_bf16w` kernel (recorded with a `MatmulBf16wBackward` dx-only
    // adjoint) instead of the equal-dtype `MatmulBackward`. Keeping the base
    // weight BF16 is the VRAM win #1443 buys; the LoRA delta + activations stay
    // F32 (so the F32-only Vulkan rmsnorm/softmax kernels are untouched). This
    // branch is Vulkan-only — CUDA/Metal carry their own BF16-base paths and run
    // BF16 activations end-to-end, so they keep the strict `weight_t == x` gate.
    #[cfg(feature = "vulkan")]
    let vk_bf16_base = matches!(x.device(), kiln_tensor::Device::Vulkan(_))
        && x.dtype() == kiln_tensor::DType::F32
        && weight_t.dtype() == kiln_tensor::DType::BF16;
    #[cfg(not(feature = "vulkan"))]
    let vk_bf16_base = false;
    if weight_t.dtype() != x.dtype() && !vk_bf16_base {
        return Ok(None);
    }
    let Ok((wk, n)) = weight_t.dims2() else {
        return Ok(None);
    };
    let x_dims = x.dims().to_vec();
    if x_dims.len() < 2 {
        return Ok(None);
    }
    let k = *x_dims.last().unwrap();
    if k != wk {
        return Ok(None);
    }
    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
    if let Some(proj) = lora {
        if proj.a.dtype() != x.dtype() || proj.b.dtype() != x.dtype() {
            return Ok(None);
        }
        let Ok((rank, a_in)) = proj.a.dims2() else {
            return Ok(None);
        };
        let Ok((b_out, b_rank)) = proj.b.dims2() else {
            return Ok(None);
        };
        if a_in != k || b_out != n || b_rank != rank {
            return Ok(None);
        }
        if !proj.a.is_contiguous() || !proj.b.is_contiguous() {
            return Ok(None);
        }
    }
    let (a_kt, b_kt) = match lora {
        Some(proj) => (Some(proj.a.clone()), Some(proj.b.clone())),
        None => (None, None),
    };
    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let x_c = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()
                .map_err(|e| anyhow::anyhow!("kt x.contiguous: {e}"))?
        };
        let x2d = x_c
            .reshape(vec![rows, k])
            .map_err(|e| anyhow::anyhow!("kt x reshape -> 2d: {e}"))?;
        tape.record(
            &x2d,
            &[x],
            Box::new(ReshapeBackward {
                input_shape: x.shape().to_vec(),
            }),
        );
        // (#1443 step 2) Base projection. On the Vulkan mixed-precision path
        // (`vk_bf16_base`) the frozen base weight is BF16 while `x2d` is F32 — the
        // equal-dtype kt `matmul` can't run, so route the base linear through the
        // dedicated `vk_matmul_bf16w` kernel and record a `MatmulBf16wBackward`
        // (dx only; the weight is frozen, no `dW` edge). Output `base2d` is F32,
        // so the LoRA delta / residual adds below stay F32-vs-F32. Every other
        // backend (CUDA/Metal, and the equal-dtype F32/BF16 Vulkan path) keeps
        // the device-agnostic `MatmulBackward` exactly as before.
        //
        // LAYOUT: the production base `weight_t` reaching this recorder is
        // `[K, N]` = `[in, out]` (the pre-transposed layout `matmul(x2d, weight_t)`
        // consumes for `x @ W`). The `vk_matmul_bf16w` kernel
        // (`vk_matmul_bf16w_fwd_rows` / `_bwd_rows`) instead expects the weight in
        // `[N, K]` = `[out, in]` row-major and computes `x @ W.T`. So transpose
        // `weight_t` `[K,N]` → `[N,K]` and materialize contiguous before dispatch.
        // The transient is BF16 (HALF the F32 size) and the RESIDENT base weight
        // stays BF16 — the #1443 VRAM win is preserved (the only F32 buffers are
        // the activations, which are F32 on Vulkan by design). The same `[N,K]`
        // weight is stored in the backward op so `dx = grad_out @ W` is computed
        // against the matching layout.
        let base2d = if vk_bf16_base {
            #[cfg(feature = "vulkan")]
            {
                let w_nk = weight_t
                    .transpose(0, 1)
                    .map_err(|e| anyhow::anyhow!("kt weight_t.transpose [K,N]->[N,K]: {e}"))?
                    .contiguous()
                    .map_err(|e| anyhow::anyhow!("kt weight_nk.contiguous: {e}"))?;
                let y = kiln_tensor::vulkan_matmul_bf16w(&x2d, &w_nk)
                    .map_err(|e| anyhow::anyhow!("kt vulkan_matmul_bf16w x2d@w: {e}"))?;
                tape.record(
                    &y,
                    &[&x2d],
                    Box::new(MatmulBf16wBackward { weight_t: w_nk }),
                );
                y
            }
            #[cfg(not(feature = "vulkan"))]
            unreachable!("vk_bf16_base is false without the vulkan feature")
        } else {
            let base2d = kiln_tensor::ops::matmul(&x2d, weight_t)
                .map_err(|e| anyhow::anyhow!("kt matmul x2d@w: {e}"))?;
            tape.record(
                &base2d,
                &[&x2d, weight_t],
                Box::new(MatmulBackward {
                    a: maybe_offload_matmul_a_saved_tensor(&x2d)
                        .context("try_tape_lora_linear_kt: save base matmul lhs")?,
                    b: weight_t.clone(),
                }),
            );
            base2d
        };
        let out2d = match (lora, a_kt.as_ref(), b_kt.as_ref()) {
            (Some(_proj), Some(a_kt), Some(b_kt)) => {
                let h_kt = kiln_tensor::ops::matmul_rhs_transposed(&x2d, a_kt)
                    .map_err(|e| anyhow::anyhow!("kt matmul_rhs_transposed x@a_t: {e}"))?;
                let d_kt = kiln_tensor::ops::matmul_rhs_transposed(&h_kt, b_kt)
                    .map_err(|e| anyhow::anyhow!("kt matmul_rhs_transposed h@b_t: {e}"))?;
                let delta_kt = kiln_tensor::ops::mul_scalar(&d_kt, lora_scale)
                    .map_err(|e| anyhow::anyhow!("kt mul_scalar(scale): {e}"))?;
                let out2d = kiln_tensor::ops::add(&base2d, &delta_kt)
                    .map_err(|e| anyhow::anyhow!("kt add(base, delta): {e}"))?;
                tape.record(
                    &out2d,
                    &[&base2d, &x2d, a_kt, b_kt],
                    Box::new(LoraDeltaAddBackward {
                        x: maybe_offload_matmul_a_saved_tensor(&x2d)
                            .context("try_tape_lora_linear_kt: save lora input")?,
                        a: a_kt.clone(),
                        b: b_kt.clone(),
                        scale: lora_scale,
                    }),
                );
                out2d
            }
            _ => base2d,
        };
        let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
        out_shape.push(n);
        let out2d_c = if out2d.is_contiguous() {
            out2d.clone()
        } else {
            out2d
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt out2d.contiguous: {e}"))?
        };
        let out_kt = out2d_c
            .reshape(out_shape)
            .map_err(|e| anyhow::anyhow!("kt out reshape -> nd: {e}"))?;
        tape.record(
            &out_kt,
            &[&out2d],
            Box::new(ReshapeBackward {
                input_shape: vec![rows, n],
            }),
        );
        Ok(out_kt)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };
    let out_kt = out_kt.context("tape_forward::try_tape_lora_linear_kt: kt-tape forward failed")?;
    // LoRA-Var grad mapping (kt-keyed) — ESSENTIAL or dA/dB never reach the optimiser.
    if let (Some(proj), Some(a_kt), Some(b_kt)) = (lora, a_kt.as_ref(), b_kt.as_ref()) {
        kiln_kt_bridge::tape_bridge::register_input_mapping_kt(a_kt.id(), proj.a.id());
        kiln_kt_bridge::tape_bridge::register_input_mapping_kt(b_kt.id(), proj.b.id());
    }
    Ok(Some(out_kt))
}

/// True unless `KILN_USE_TAPE_FLASH_ATTN` is set to a disable value —
/// **DEFAULTS ON** (CP-4 production path).
///
/// Separate gate from `KILN_USE_TAPE_FORWARD` so the flash-attention tape
/// adapter can be opted out independently of the rest of the tape-forward
/// fleet: the attention-block gradient now flows through a kt `Tape` node by
/// default. Set the env to `0`/`false`/`no`/empty to opt out and route through
/// candle's `CudaFlashAttentionTrainingBf16` CustomOp3 (for debugging /
/// comparison). Cached after first read, matching [`tape_lora_add_enabled`].
pub fn tape_flash_attn_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_FLASH_ATTN")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(true)
    })
}

/// Fused tape backward for the vendored FlashAttention-2 forward
/// (`kiln_flash_attn::flash_attn_fwd_kt`).
///
/// # Why this `BackwardOp` lives in `kiln-model`, not `kiln-autograd`
///
/// Every other `BackwardOp` is a device-agnostic composite of
/// `kiln_tensor::ops` and lives in `kiln-autograd`. FlashAttention is the
/// exception: its forward and backward are a single fused CUDA kernel
/// (`kiln_flash_attn::flash_attn_{fwd,bwd}_kt`) with no device-agnostic
/// composite matching the kernel's numerics or memory profile. Since
/// `kiln-autograd` deliberately carries no `kiln-flash-attn` dependency
/// (layering — it stays buildable on every backend), the op that
/// dispatches the kernel lives here in `kiln-model` (which already depends
/// on `kiln-flash-attn`) and implements the `kiln_autograd::BackwardOp`
/// trait. The CPU/composite reference for parity lives in the test.
///
/// # Saved tensors
///
/// `q`, `k`, `v` (the GROUPED GQA inputs as handed to `flash_attn_fwd_kt`),
/// the forward output `out`, and `softmax_lse` — all kt clones (Arc bumps;
/// no allocation). `scale`/`causal` are the forward params;
/// `heads_q`/`heads_kv` drive the GQA gradient collapse.
///
/// # Backward
///
/// `flash_attn_bwd_kt(dout, q, k, v, out, lse, scale, causal)` returns
/// `(dq, dk, dv)` where `dk`/`dv` come back EXPANDED to `heads_q` (the
/// kernel internally broadcasts grouped K/V). When `heads_kv != heads_q`
/// we collapse them to `heads_kv` by reshaping to `[b, sk, heads_kv,
/// groups, hd]` and summing the group axis — mirroring the
/// `CudaFlashAttentionTrainingBf16::bwd` candle path exactly. The collapse
/// runs in F32 (cast → sum → cast back to BF16) so the group reduction
/// doesn't lose precision in BF16.
#[cfg(any(feature = "cuda", feature = "rocm"))]
#[derive(Debug)]
pub(crate) struct FlashAttnBackward {
    q: kiln_tensor::Tensor,
    k: kiln_tensor::Tensor,
    v: kiln_tensor::Tensor,
    out: kiln_tensor::Tensor,
    softmax_lse: kiln_tensor::Tensor,
    scale: f32,
    causal: bool,
    heads_q: usize,
    heads_kv: usize,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl BackwardOp for FlashAttnBackward {
    fn name(&self) -> &'static str {
        "flash_attn_backward"
    }
    fn input_count(&self) -> usize {
        // q, k, v.
        3
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        use kiln_tensor::{DType, bail};

        // FA bwd needs a BF16, compact-contiguous dout shaped like `out`.
        let dout = if grad_output.dtype() == DType::BF16 {
            grad_output.clone()
        } else {
            kiln_tensor::ops::cast(grad_output, DType::BF16)?
        };
        let dout = if dout.is_contiguous() {
            dout
        } else {
            dout.contiguous()?
        };

        #[cfg(feature = "cuda")]
        let trace_timings =
            kiln_core::env_flag::env_flag("KILN_TRACE_FLASH_ATTN_BWD_TIMINGS", false);
        #[cfg(feature = "cuda")]
        let flash_started = if trace_timings {
            if let kiln_tensor::Device::Cuda(i) = self.q.device() {
                kiln_tensor::cuda_synchronize_default_stream(i)?;
            }
            Some(std::time::Instant::now())
        } else {
            None
        };

        let (dq, dk_exp, dv_exp) = kiln_flash_attn::flash_attn_bwd_kt(
            &dout,
            &self.q,
            &self.k,
            &self.v,
            &self.out,
            &self.softmax_lse,
            self.scale,
            self.causal,
        )
        .map_err(|e| {
            kiln_tensor::Error::Msg(format!("FlashAttnBackward: flash_attn_bwd_kt: {e:?}"))
        })?;

        #[cfg(feature = "cuda")]
        if let Some(started) = flash_started {
            if let kiln_tensor::Device::Cuda(i) = self.q.device() {
                kiln_tensor::cuda_synchronize_default_stream(i)?;
                let q_shape = self.q.shape();
                let k_shape = self.k.shape();
                eprintln!(
                    "kiln_flash_attn_bwd_timing phase=ffi batch={} seq_len_q={} seq_len_k={} \
                     heads_q={} heads_kv={} head_dim={} causal={} elapsed_ms={:.3}",
                    q_shape[0],
                    q_shape[1],
                    k_shape[1],
                    self.heads_q,
                    self.heads_kv,
                    q_shape[3],
                    self.causal,
                    started.elapsed().as_secs_f64() * 1000.0,
                );
            }
        }

        #[cfg(feature = "cuda")]
        let collapse_started = if trace_timings && self.heads_kv != self.heads_q {
            if let kiln_tensor::Device::Cuda(i) = self.q.device() {
                kiln_tensor::cuda_synchronize_default_stream(i)?;
            }
            Some(std::time::Instant::now())
        } else {
            None
        };

        // GQA collapse: dk/dv come back expanded to heads_q.
        let (dk, dv) = if self.heads_kv != self.heads_q {
            if self.heads_kv == 0 || self.heads_q % self.heads_kv != 0 {
                bail!(
                    "FlashAttnBackward: invalid GQA heads_q={} heads_kv={}",
                    self.heads_q,
                    self.heads_kv
                );
            }
            let groups = self.heads_q / self.heads_kv;
            let collapse =
                |dexp: &kiln_tensor::Tensor| -> kiln_tensor::Result<kiln_tensor::Tensor> {
                    let s = dexp.shape();
                    if s.len() != 4 {
                        bail!(
                            "FlashAttnBackward: expanded grad must be rank-4 \
                             [b,sk,heads_q,hd], got {s:?}"
                        );
                    }
                    let (b, sk, hq, hd) = (s[0], s[1], s[2], s[3]);
                    if hq != self.heads_q {
                        bail!(
                            "FlashAttnBackward: expanded grad heads {hq} != heads_q {}",
                            self.heads_q
                        );
                    }
                    // Reduce groups in F32 (BF16 group-sum loses precision).
                    let f32g = kiln_tensor::ops::cast(dexp, DType::F32)?;
                    let grouped = f32g.reshape(vec![b, sk, self.heads_kv, groups, hd])?;
                    let grouped = if grouped.is_contiguous() {
                        grouped
                    } else {
                        grouped.contiguous()?
                    };
                    let summed = kiln_tensor::ops::sum_axis(&grouped, 3)?; // [b,sk,heads_kv,hd]
                    kiln_tensor::ops::cast(&summed, DType::BF16)
                };
            (collapse(&dk_exp)?, collapse(&dv_exp)?)
        } else {
            (dk_exp, dv_exp)
        };

        #[cfg(feature = "cuda")]
        if let Some(started) = collapse_started {
            if let kiln_tensor::Device::Cuda(i) = self.q.device() {
                kiln_tensor::cuda_synchronize_default_stream(i)?;
                let k_shape = self.k.shape();
                eprintln!(
                    "kiln_flash_attn_bwd_timing phase=gqa_collapse batch={} seq_len_k={} \
                     heads_q={} heads_kv={} head_dim={} elapsed_ms={:.3}",
                    k_shape[0],
                    k_shape[1],
                    self.heads_q,
                    self.heads_kv,
                    k_shape[3],
                    started.elapsed().as_secs_f64() * 1000.0,
                );
            }
        }

        Ok(vec![Some(dq), Some(dk), Some(dv)])
    }
}

/// Attempt to route the FlashAttention-2 forward through the kt `Tape`
/// instead of candle's `CudaFlashAttentionTrainingBf16` CustomOp3.
///
/// `q` is `[b, sq, heads_q, hd]`; `k`/`v` are the GROUPED `[b, sk,
/// heads_kv, hd]` GQA tensors (the CUDA FA2 wrapper consumes grouped K/V
/// directly). Returns the attention output `[b, sq, heads_q, hd]` (the
/// caller reshapes to `[b, sq, heads_q*hd]` for o_proj). The recorded
/// [`FlashAttnBackward`] node emits GQA-collapsed `dq/dk/dv` so a
/// tape-authoritative backward seeded at the loss reaches the q/k/v
/// projections (and therefore their LoRA `Var`s) — this is the
/// attention-block link the CP-4 tape-authoritative SFT path was missing
/// (flash-attn previously recorded only onto candle's `BackpropOp` graph,
/// leaving the LoRA tape nodes a disconnected island).
///
/// `Ok(None)` (caller falls through to the existing CustomOp3 / fast path)
/// when: the gate is off, no tape scope is active, the inputs leave the
/// BF16/CUDA/contiguous/`head_dim∈{128,256}`/valid-GQA envelope, or a kt
/// borrow fails.
/// kt-native FlashAttention-2 tape recorder (#1082 seam flip) — kt-only twin of
/// `try_tape_flash_attn_cuda`. q/k/v are already kt (recorded upstream outputs),
/// so no candle bridging: runs `flash_attn_fwd_kt` and records the kt-native
/// `FlashAttnBackward` (all kt-tensor fields) directly. The returned kt out lets the
/// downstream reshape stay kt-native too (no kt->candle->kt at the attention seam).
pub fn try_tape_flash_attn_kt(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() || !tape_flash_attn_enabled() {
        return Ok(None);
    }
    // Fused FlashAttention-2 is a CUDA kernel; ROCm (gfx, Phase R.8) records the
    // same kt-native `FlashAttnBackward`, whose backward dispatches the ROCm
    // composite flash-attn fwd/bwd (`flash_attn_{fwd,bwd}_rocm`). On Metal/CPU/
    // Vulkan the kt-native unfused SDPA fallback (`try_tape_sdpa_fallback_kt`)
    // records attention backward instead. (#37) ROCm dq/dk/dv are gradchecked
    // against an independent F32 analytic reference (rocm_flash_attn_bwd_gradcheck).
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        let _ = (q, k, v, num_heads, num_kv_heads, head_dim);
        return Ok(None);
    }
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        if q.dtype() != kiln_tensor::DType::BF16
            || k.dtype() != kiln_tensor::DType::BF16
            || v.dtype() != kiln_tensor::DType::BF16
            || !matches!(
                q.device(),
                kiln_tensor::Device::Cuda(_)
                    | kiln_tensor::Device::Metal(_)
                    | kiln_tensor::Device::Rocm(_)
            )
            || !matches!(
                k.device(),
                kiln_tensor::Device::Cuda(_)
                    | kiln_tensor::Device::Metal(_)
                    | kiln_tensor::Device::Rocm(_)
            )
            || !matches!(
                v.device(),
                kiln_tensor::Device::Cuda(_)
                    | kiln_tensor::Device::Metal(_)
                    | kiln_tensor::Device::Rocm(_)
            )
            || !q.is_contiguous()
            || !k.is_contiguous()
            || !v.is_contiguous()
            || !matches!(head_dim, 128 | 256)
            || num_kv_heads == 0
            || num_heads % num_kv_heads != 0
        {
            return Ok(None);
        }
        let Ok((bq, _sq, hq, dq_)) = q.dims4() else {
            return Ok(None);
        };
        let Ok((bk, sk, hk, dk_)) = k.dims4() else {
            return Ok(None);
        };
        let Ok((bv, sv, hv, dv_)) = v.dims4() else {
            return Ok(None);
        };
        if bq != bk
            || bq != bv
            || sk != sv
            || hq != num_heads
            || hk != num_kv_heads
            || hv != num_kv_heads
            || dq_ != head_dim
            || dk_ != head_dim
            || dv_ != head_dim
        {
            return Ok(None);
        }
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        let causal = true;
        match with_active_tape(|tape: &mut Tape| -> Result<_> {
            let (out_kt, lse_kt) =
                kiln_flash_attn::flash_attn_fwd_kt(q, k, v, softmax_scale, causal)
                    .map_err(|e| anyhow::anyhow!("kt flash_attn_fwd_kt: {e:?}"))?;
            tape.record(
                &out_kt,
                &[q, k, v],
                Box::new(FlashAttnBackward {
                    q: q.clone(),
                    k: k.clone(),
                    v: v.clone(),
                    out: out_kt.clone(),
                    softmax_lse: lse_kt,
                    scale: softmax_scale,
                    causal,
                    heads_q: num_heads,
                    heads_kv: num_kv_heads,
                }),
            );
            Ok(out_kt)
        }) {
            Some(result) => Ok(Some(result?)),
            None => Ok(None),
        }
    }
}

/// True unless `KILN_USE_TAPE_GDN` is set to a disable value — **DEFAULTS ON**
/// (CP-4 production path). Gate for the GDN (linear-attention) recurrence tape
/// adapter, separate from the rest of the fleet. Set the env to
/// `0`/`false`/`no`/empty to opt out (for debugging / comparison). Cached after
/// first read.
pub fn tape_gdn_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_GDN")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(true)
    })
}

/// Offload saved GDN recurrence tensors from the tape node to CPU memory.
///
/// The recorded input ids remain the original device tensors, so gradient
/// chaining is unchanged. Only the values captured by `GdnRecurrentBackward`
/// are moved off the accelerator and uploaded back one node at a time during
/// backward. This trades bandwidth for much lower long-context tape residency.
fn tape_gdn_saved_tensor_offload_enabled(device: &kiln_tensor::Device) -> bool {
    static OVERRIDE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    if let Some(value) = *OVERRIDE.get_or_init(|| {
        std::env::var("KILN_TAPE_GDN_OFFLOAD_SAVED_TENSORS")
            .ok()
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
    }) {
        return value;
    }

    let _ = device;
    false
}

fn maybe_offload_gdn_saved_tensor(
    t: &kiln_tensor::Tensor,
    offload: bool,
) -> kiln_tensor::Result<kiln_tensor::Tensor> {
    if offload && should_offload_saved_tensor(t) {
        t.to_device(kiln_tensor::Device::Cpu)
    } else {
        Ok(t.clone())
    }
}

fn saved_tensor_for_device(
    t: &kiln_tensor::Tensor,
    device: kiln_tensor::Device,
) -> kiln_tensor::Result<kiln_tensor::Tensor> {
    if t.device() == device {
        Ok(t.clone())
    } else {
        t.to_device(device)
    }
}

fn tape_saved_tensor_offload_min_bytes() -> usize {
    static MIN_BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN_BYTES.get_or_init(|| {
        std::env::var("KILN_TAPE_OFFLOAD_MIN_BYTES")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(1 << 20)
    })
}

fn saved_tensor_bytes(t: &kiln_tensor::Tensor) -> usize {
    t.dtype().size_in_bytes().saturating_mul(t.element_count())
}

fn should_offload_saved_tensor(t: &kiln_tensor::Tensor) -> bool {
    !matches!(t.device(), kiln_tensor::Device::Cpu)
        && saved_tensor_bytes(t) >= tape_saved_tensor_offload_min_bytes()
}

fn tape_matmul_a_offload_enabled(device: &kiln_tensor::Device) -> bool {
    static OVERRIDE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    if let Some(value) = *OVERRIDE.get_or_init(|| {
        std::env::var("KILN_TAPE_OFFLOAD_MATMUL_A").ok().map(|v| {
            let v = v.trim().to_lowercase();
            !(v.is_empty() || v == "0" || v == "false" || v == "no")
        })
    }) {
        return value;
    }

    let _ = device;
    false
}

fn maybe_offload_matmul_a_saved_tensor(
    t: &kiln_tensor::Tensor,
) -> kiln_tensor::Result<kiln_tensor::Tensor> {
    if tape_matmul_a_offload_enabled(&t.device()) && should_offload_saved_tensor(t) {
        t.to_device(kiln_tensor::Device::Cpu)
    } else {
        Ok(t.clone())
    }
}

/// Tape backward for the GDN (Gated DeltaNet linear-attention) recurrence.
///
/// # Why a composite wrap (not a kt BackwardOp in kiln-autograd)
///
/// The GDN recurrence backward is a stateful chunk-wise reverse-time
/// algorithm already implemented + CPU-parity-tested as the device-
/// agnostic kt composite [`gdn_recurrent_backward_no_grad`]
/// (`test_gdn_recurrent_backward_no_grad_matches_autograd_cpu`). Rather
/// than re-derive it as kt ops, this `BackwardOp` wraps that proven
/// function: it saves the kt forward inputs and, on backward, runs the
/// existing kt chunk-wise backward directly (#1082 P4-full — no candle
/// bridge on either the saved tensors or the upstream grad). Lives in
/// kiln-model (not kiln-autograd) because it calls `crate::forward` +
/// `crate::backend`, mirroring [`FlashAttnBackward`].
///
/// # Saved tensors / inputs
///
/// `q`/`k`/`v`/`beta`/`g` + `entry_state` (the recurrent state BEFORE the
/// forward mutated it) as kt tensors — the SAME kt ids recorded as this
/// node's `tape.record` inputs, so the upstream forward records kt directly
/// (no per-layer kt->candle save copies); `device` is the kt `Device`
/// (`for_device_kt` reconstructs the backend, candle-free); `chunk_size` is
/// [`GDN_CHUNK_SIZE`]. 5 differentiable inputs `[q, k, v, beta, g]` in the
/// order the adapter records them; `entry_state` is the initial (zero) state
/// at the SFT layer boundary, so the backward's `grad_exit_state` is `None`.
///
/// # Output layout (`head_last_output`)
///
/// The production GDN recurrence dispatch returns the attention output in
/// **head-LAST `[B, T, nv, dv]`** layout on the CUDA prefill /
/// full-chunk paths (`gdn_recurrent_prefill_head_last` /
/// `gdn_chunkwise_recurrence_head_last_full_chunks`) and **head-FIRST
/// `[B, nv, T, dv]`** only on the chunkwise fallback
/// (`gdn_chunkwise_recurrence`). [`gdn_recurrent_backward_no_grad`] always
/// expects a **head-FIRST** grad (its internal `grad_out.narrow(2, …)`
/// indexes the seq axis at dim 2). `head_last_output` records which layout
/// the recorded forward output used so `apply` can transpose a head-LAST
/// upstream grad back to head-FIRST before invoking the backward. The
/// saved `q`/`k`/`v`/`beta`/`g`/`entry_state` are ALWAYS head-first (they
/// are the post-`recur_prep`-transpose recurrence inputs), so the returned
/// `dq`/…/`dg` are head-first and match the head-first input `Var`s
/// regardless of `head_last_output`.
#[derive(Debug)]
pub(crate) struct GdnRecurrentBackward {
    // #1082 P4-full: saved tensors are kt (`kiln_tensor::Tensor`) — the SAME
    // kt ids that flow from the GDN in_proj projections (recorded as the
    // `tape.record` inputs), so the upstream forward no longer bridges
    // kt->candle for these 6 tensors before recording (~7 DtoD copies/GDN
    // layer/step, ×24 GDN layers). `apply` reads them directly; the backward
    // (`gdn_recurrent_backward_no_grad`) is already kt-native.
    q: kiln_tensor::Tensor,
    k: kiln_tensor::Tensor,
    v: kiln_tensor::Tensor,
    beta: kiln_tensor::Tensor,
    g: kiln_tensor::Tensor,
    entry_state: kiln_tensor::Tensor,
    device: kiln_tensor::Device,
    chunk_size: usize,
    /// `true` when the recorded forward output was head-LAST
    /// `[B, T, nv, dv]`; `apply` then transposes the upstream grad to the
    /// head-FIRST `[B, nv, T, dv]` layout the backward requires.
    head_last_output: bool,
}

impl BackwardOp for GdnRecurrentBackward {
    fn name(&self) -> &'static str {
        "gdn_recurrent_backward"
    }
    fn input_count(&self) -> usize {
        // q, k, v, beta, g.
        5
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        // #1082 P4: the upstream grad is ALREADY kt and
        // `gdn_recurrent_backward_no_grad` takes a kt grad — transpose
        // head-last -> head-first IN KT directly, dropping the pointless
        // kt->candle->kt round-trip (one [B,T,nv,dv] DtoD copy per GDN-layer
        // backward, ×24 layers/step). `gdn_recurrent_backward_no_grad` indexes
        // the seq axis at dim 2 (head-FIRST); when the recorded forward output
        // was head-LAST `[B, T, nv, dv]` the upstream grad arrives head-last,
        // so transpose back to `[B, nv, T, dv]`. The saved q/k/v/beta/g/
        // entry_state are already head-first, so the returned grads stay
        // head-first (no transpose on the way out).
        let grad_out_kt = if self.head_last_output {
            grad_output
                .transpose(1, 2)
                .map_err(|e| {
                    kiln_tensor::Error::Msg(format!(
                        "GdnRecurrentBackward: head-last grad transpose: {e}"
                    ))
                })?
                .contiguous()
                .map_err(|e| {
                    kiln_tensor::Error::Msg(format!(
                        "GdnRecurrentBackward: head-last grad contiguous: {e}"
                    ))
                })?
        } else {
            grad_output.contiguous().map_err(|e| {
                kiln_tensor::Error::Msg(format!("GdnRecurrentBackward: grad contiguous: {e}"))
            })?
        };
        // #1082 P4-full: the saved q/k/v/beta/g/entry_state are kt, and
        // `gdn_recurrent_backward_no_grad` is kt-typed (kt inputs, kt-grad
        // outputs) — no candle bridge at all. `for_device_kt` reconstructs the
        // backend straight from the stored kt `Device` (candle-free).
        //
        // (#1082) GDN-on-Vulkan: `gdn_recurrent_backward_no_grad` uses CPU-only kt
        // ops (`cumsum`, `where_cond`) that error on Vulkan storage. Mirror the
        // forward's CPU offload (`gdn_chunkwise_recurrence`): run the analytic
        // backward on CPU, then move the per-input grads back to the activation
        // device so they chain to the (Vulkan) q/k/v projection nodes. CUDA/Metal
        // keep their native device path.
        let on_vulkan = matches!(self.device, kiln_tensor::Device::Vulkan(_));
        let compute_dev = if on_vulkan {
            kiln_tensor::Device::Cpu
        } else {
            self.device
        };
        let backend = crate::backend::for_device_kt(&compute_dev);
        let mv = |t: &kiln_tensor::Tensor| -> kiln_tensor::Result<kiln_tensor::Tensor> {
            if t.device() == compute_dev {
                Ok(t.clone())
            } else {
                t.to_device(compute_dev)
            }
        };
        let (q_c, k_c, v_c, beta_c, g_c, entry_c, grad_c) = (
            mv(&self.q)?,
            mv(&self.k)?,
            mv(&self.v)?,
            mv(&self.beta)?,
            mv(&self.g)?,
            mv(&self.entry_state)?,
            mv(&grad_out_kt)?,
        );
        let grads = gdn_recurrent_backward_no_grad(
            &*backend,
            &q_c,
            &k_c,
            &v_c,
            &beta_c,
            &g_c,
            &entry_c,
            &grad_c,
            None,
            self.chunk_size,
        )
        .map_err(|e| kiln_tensor::Error::Msg(format!("GdnRecurrentBackward: gdn bwd: {e}")))?;
        // grads.* are kt; the backward can return non-contiguous grads
        // (internal transposes/narrows) — materialise contiguous and restore the
        // activation device (the CPU offload above ran on `Device::Cpu`).
        let dev = self.device;
        let to_kt = |t: &kiln_tensor::Tensor| -> kiln_tensor::Result<kiln_tensor::Tensor> {
            let c = t.contiguous().map_err(|e| {
                kiln_tensor::Error::Msg(format!("GdnRecurrentBackward: grad contiguous: {e}"))
            })?;
            if on_vulkan { c.to_device(dev) } else { Ok(c) }
        };
        Ok(vec![
            Some(to_kt(&grads.dq)?),
            Some(to_kt(&grads.dk)?),
            Some(to_kt(&grads.dv)?),
            Some(to_kt(&grads.dbeta)?),
            Some(to_kt(&grads.dg)?),
        ])
    }
}

/// Route the GDN recurrence forward through the kt `Tape` so a
/// tape-authoritative backward reaches the GDN-block q/k/v/beta/g
/// projections (and their LoRA `Var`s) — the linear-attention analogue of
/// `try_tape_flash_attn_cuda`, covering Qwen3.5-4B's 24 GDN layers.
///
/// Runs [`gdn_recurrent_forward_from_parts`] (mutating `recurrent_state`)
/// and records a [`GdnRecurrentBackward`] whose `entry_state` is the
/// snapshot of `recurrent_state` BEFORE the forward. Drop-in for the
/// production recurrence call: `Ok(Some(out))` (with a tape node if a scope
/// is active), or `Ok(None)` (caller runs the recurrence itself) when the
/// gate is off, the inputs aren't CUDA, or a kt borrow fails.
pub fn try_tape_gdn_recurrent_kt(
    backend: &dyn BackendRuntime,
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    recurrent_state: &mut kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() || !tape_gdn_enabled() {
        return Ok(None);
    }
    // #1082 P4-full: q/k/v/beta/g/recurrent_state are kt (this calls the kt
    // production recurrence `gdn_recurrent_forward_from_parts`) and the record
    // adapter (`tape_record_gdn_recurrent_kt`) is now kt-native — no kt->candle
    // bridge on the saved inputs.
    if !matches!(
        q.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }

    // Snapshot the entry state BEFORE the forward mutates it (the backward
    // needs it), and the kt device for backend reconstruction in `apply`.
    let entry_state = recurrent_state.clone();
    let device = q.device();

    // Production recurrence forward (mutates recurrent_state in place).
    // `gdn_recurrent_forward_from_parts` returns the recurrence output in
    // head-FIRST `[B, nv, T, dv]` layout (the short-seq chunkwise path),
    // hence `head_last = false` below.
    let out_kt = gdn_recurrent_forward_from_parts(backend, q, k, v, beta, g, recurrent_state)?;

    // Record the node directly from kt (no-op unless a tape scope is active).
    tape_record_gdn_recurrent_kt(&out_kt, false, q, k, v, beta, g, &entry_state, &device)?;

    // #1082: kt-native — the recorded output node id lives on the kt tape;
    // return the kt output directly (no kt->candle bridge).
    Ok(Some(out_kt))
}

/// kt-native recorder for a [`GdnRecurrentBackward`] node (#1082 P4-full).
///
/// The PRODUCTION GDN forward (`forward.rs:gated_deltanet_forward_decode_if`)
/// calls this DIRECTLY with the kt recurrence output + the kt head-FIRST
/// q/k/v/beta/g recurrence inputs + the kt entry-state snapshot, so it no
/// longer bridges those 6 tensors kt->candle just to record (~7 DtoD copies
/// per GDN layer per step, ×24 GDN layers). The candle entry point
/// (The candle shim `tape_record_gdn_recurrent` that wrapped this was deleted in #1082.)
///
/// # Arguments
///
/// * `out_kt` — the PRODUCTION recurrence-output kt tensor. Its `.id()` must be
///   the SAME id that flows downstream (the next adapter's `tape_kt_input`
///   resolves to it via `forward.rs`'s `kt_logits_to_candle` retain), so the
///   tape stays connected across the recurrence→transpose seam. Layout per
///   `head_last`.
/// * `head_last` — `true` when `out_kt` is head-LAST `[B, T, nv, dv]` (CUDA
///   prefill / full-chunk paths), `false` when head-FIRST `[B, nv, T, dv]`
///   (chunkwise fallback). Stored on the recorded [`GdnRecurrentBackward`] so
///   its `apply` can transpose a head-last grad back to head-first.
/// * `q`/`k`/`v`/`beta`/`g` — the head-FIRST recurrence inputs (post
///   `recur_prep` transpose), the SAME kt ids that flow from the GDN in_proj
///   projections. Recorded as the 5 differentiable inputs AND saved on the
///   `GdnRecurrentBackward` (chaining preserved because the recorded input ids
///   are unchanged — only the *saved* representation flipped candle→kt).
/// * `entry_state` — the recurrent state BEFORE the forward mutated it (kt).
/// * `device` — the kt `Device` (`for_device_kt` reconstructs the backend in
///   `apply`, candle-free).
///
/// Returns `Ok(true)` when a node was recorded (a tape scope was active),
/// `Ok(false)` otherwise (no scope — production output unaffected).
#[allow(clippy::too_many_arguments)]
pub fn tape_record_gdn_recurrent_kt(
    out_kt: &kiln_tensor::Tensor,
    head_last: bool,
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    entry_state: &kiln_tensor::Tensor,
    device: &kiln_tensor::Device,
) -> Result<bool> {
    if !tape_forward_enabled() || !tape_gdn_enabled() {
        return Ok(false);
    }
    if !matches!(
        device,
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(false);
    }

    let recorded = with_active_tape(|tape: &mut Tape| -> Result<()> {
        let offload_saved = tape_gdn_saved_tensor_offload_enabled(device);
        let saved_q = maybe_offload_gdn_saved_tensor(q, offload_saved)
            .context("tape_record_gdn_recurrent_kt: save q")?;
        let saved_k = maybe_offload_gdn_saved_tensor(k, offload_saved)
            .context("tape_record_gdn_recurrent_kt: save k")?;
        let saved_v = maybe_offload_gdn_saved_tensor(v, offload_saved)
            .context("tape_record_gdn_recurrent_kt: save v")?;
        let saved_beta = maybe_offload_gdn_saved_tensor(beta, offload_saved)
            .context("tape_record_gdn_recurrent_kt: save beta")?;
        let saved_g = maybe_offload_gdn_saved_tensor(g, offload_saved)
            .context("tape_record_gdn_recurrent_kt: save g")?;
        let saved_entry_state = maybe_offload_gdn_saved_tensor(entry_state, offload_saved)
            .context("tape_record_gdn_recurrent_kt: save entry_state")?;

        tape.record(
            out_kt,
            &[q, k, v, beta, g],
            Box::new(GdnRecurrentBackward {
                q: saved_q,
                k: saved_k,
                v: saved_v,
                beta: saved_beta,
                g: saved_g,
                entry_state: saved_entry_state,
                device: *device,
                chunk_size: GDN_CHUNK_SIZE,
                head_last_output: head_last,
            }),
        );
        Ok(())
    });
    // `Some(())` => a tape scope was active and the node was recorded.
    match recorded {
        Some(result) => {
            result?;
            Ok(true)
        }
        None => Ok(false),
    }
}

// ===========================================================================
// CP-4 (#1082): GDN surrounding ops — conv1d / L2-qk-norm / gated-RMSNorm.
//
// The GDN recurrence is tape-wired above. For a tape-authoritative backward to
// reach the GDN-block in_proj / out_proj LoRA Vars, EVERY op between the
// projection matmuls and the recurrence must ALSO record onto the kt Tape, or
// the chain fragments. These three adapters cover:
//
//   * `try_tape_causal_conv1d_cuda`     — Step 2 depthwise conv on mixed-QKV.
//   * `try_tape_gdn_l2_norm_scale_cuda` — Step 4/5 L2 qk-norm (q scaled, k not).
//   * `try_tape_gdn_gated_rms_norm_cuda`— Step 8 gated RMSNorm before out_proj.
//
// Plus `try_tape_transpose_cuda` (the head-FIRST→head-LAST chaining-gap fix at
// `forward.rs:gated_deltanet_forward_decode_if`'s `attn_out.transpose(1,2)`).
//
// Each has a narrow `KILN_USE_TAPE_GDN_*` gate (mirroring `tape_gdn_enabled`),
// is a no-op by default, and falls through cleanly so the production forward is
// untouched with the gate off.
// ===========================================================================

/// True unless `KILN_USE_TAPE_GDN_CONV` is set to a disable value —
/// **DEFAULTS ON** (CP-4 production path). Gate for the GDN
/// causal-depthwise-conv1d tape adapter. Set the env to `0`/`false`/`no`/empty
/// to opt out (for debugging / comparison). Cached after first read.
pub fn tape_gdn_conv_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_GDN_CONV")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(true)
    })
}

/// True unless `KILN_USE_TAPE_GDN_QK_NORM` is set to a disable value —
/// **DEFAULTS ON** (CP-4 production path). Gate for the GDN L2-qk-norm tape
/// adapter. Set the env to `0`/`false`/`no`/empty to opt out (for debugging /
/// comparison). Cached after first read.
pub fn tape_gdn_qk_norm_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_GDN_QK_NORM")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(true)
    })
}

/// True unless `KILN_USE_TAPE_GDN_GATED_NORM` is set to a disable value —
/// **DEFAULTS ON** (CP-4 production path). Gate for the GDN gated-RMSNorm tape
/// adapter. Set the env to `0`/`false`/`no`/empty to opt out (for debugging /
/// comparison). Cached after first read.
pub fn tape_gdn_gated_norm_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_GDN_GATED_NORM")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(true)
    })
}

/// Tape backward (w.r.t. input) for the GDN PREFILL causal depthwise conv1d
/// (`forward::causal_conv1d_prefill`, the `[B, C, T]` path the training forward
/// takes when `gdn_forward_only_fastpaths` is OFF), handling the
/// `[B, C, T]` ↔ `[rows, channels]` layout transform.
///
/// # Device-agnostic bwd routing (#1082 Metal GDN)
///
/// The `[rows, channels]` input gradient is computed two ways, both numerically
/// identical (FD-gradient-checked + an explicit-index-loop equality test in
/// `kiln_tensor::ops::causal_conv1d_bwd`):
///
/// - `Device::Cuda(_)`: the proven CUDA bwd-input kernel
///   `causal_depthwise_conv1d_bwd_input_kt` (the SAME kernel the eager
///   `cuda_train.rs` GDN backward uses), unchanged. CUDA path is byte-identical.
/// - `Device::Cpu` / `Device::Metal(_)`: the device-agnostic kt composite
///   `causal_depthwise_conv1d_bwd_input_composite` (pure `kiln_tensor` ops:
///   pad / narrow / broadcast-mul / add — no FFI), which unblocks GDN-layer
///   LoRA training on Metal.
///
/// # Why a dedicated op (not the existing `try_tape_causal_conv1d_cuda`)
///
/// The existing `try_tape_causal_conv1d_cuda` / `CausalConv1dInputBackward`
/// wraps the `[rows, channels]` DECODE/UPDATE kernel
/// (`causal_depthwise_conv1d_kt`) — a DIFFERENT kernel with a different layout
/// contract than the `[B, C, T]` prefill path. Reusing it here would compute
/// the wrong forward. This op wraps the prefill bwd-input on the `[B, C, T]`
/// layout the production prefill forward uses.
///
/// # Saved tensors / inputs
///
/// `weight` (`[channels, kernel]` F32) is the only saved state; one
/// differentiable input (`input`, the `[B, C, T]` conv input). The conv
/// `weight` is a frozen base tensor in LoRA training (not a `Var`), and the
/// conv state is non-differentiable at the SFT layer boundary, so only the
/// input gradient is needed to keep the chain connected back through the
/// in_proj_qkv LoRA `Var`.
///
/// # Layout
///
/// The bwd kernel expects `[rows, channels]` with `rows` the contiguous causal
/// (time) axis. The prefill grad arrives `[B, C, T]`; this transposes to
/// `[B, T, C]`, flattens to `[B*T, C]` (so rows = time within each sequence),
/// runs the kernel, then reverses the layout. Gated to `batch == 1` (the SFT
/// training shape) so flattening never mixes batches across a row boundary;
/// declines (`apply` errors only on a true kernel failure — the adapter's
/// `Ok(None)` envelope guard below keeps `batch>1` off this path entirely).
///
/// Device-agnostic kt-composite for the GDN causal-depthwise-conv1d
/// input-backward — the analytic input-grad derived directly from the CUDA
/// kernel's math (`crates/kiln-rmsnorm-kernel/csrc/causal_conv1d_f32.cu`,
/// `causal_depthwise_conv1d_bwd_input_f32_kernel`), expressed purely in
/// `kiln_tensor` host arithmetic so it runs on CPU / Metal / Vulkan with no
/// FFI, no cudarc, and no candle.
///
/// This is the conv1d analogue of the OPD reverse-KL backward composite
/// (`kiln_opd_loss_kernel::kt_api::opd_top_k_reverse_kl_phase_b_bwd_composite_kt`):
/// the `Cuda(_)` arm keeps the validated/perf FFI kernel; every other device
/// routes here.
///
/// # Forward convention (verified against the CUDA kernel)
///
/// The forward (`causal_depthwise_conv1d_f32_kernel`) with `state_rows = K-1`
/// computes the left-zero-padded causal depthwise conv
///
///   `out[r,c] = Σ_{j=0..K-1} weight[c,j] · x_pad[r+j, c]`
///
/// where `x_pad[p,c] = state[p,c]` for `p < K-1` else `input[p-(K-1), c]`.
///
/// # Input gradient (matches `causal_depthwise_conv1d_bwd_input_f32_kernel`)
///
/// The kernel computes, for `state_rows = K-1`,
///
///   `grad_input[i,c] = Σ_{j: out_row = (K-1)+i-j ∈ [0,rows)} grad_out[out_row,c] · weight[c,j]`.
///
/// Substituting `m = K-1-j` (`out_row = i+m`) gives the equivalent
/// right-zero-padded anti-causal correlation
///
///   `grad_x[s,c] = Σ_{m=0..K-1, (s+m)<rows} weight[c, K-1-m] · grad_out[s+m, c]`,
///
/// which this function evaluates exactly. `grad_out` and `weight` are read to
/// the host (both are small F32 tensors — `[rows, channels]` and
/// `[channels, kernel]`), the correlation is accumulated in F32, and the result
/// is uploaded back to `grad_out`'s device. FD-validated against a central
/// finite difference of the forward in `tests` below.
pub(crate) fn causal_depthwise_conv1d_bwd_input_composite(
    grad_out: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    kernel: usize,
) -> kiln_tensor::Result<kiln_tensor::Tensor> {
    use kiln_tensor::{DType, Error};
    let go_shape = grad_out.shape();
    if go_shape.len() != 2 {
        return Err(Error::Msg(format!(
            "conv1d-bwd-input composite: grad_out must be [rows, channels], got {go_shape:?}"
        )));
    }
    let (rows, channels) = (go_shape[0], go_shape[1]);
    if weight.shape() != [channels, kernel] {
        return Err(Error::Msg(format!(
            "conv1d-bwd-input composite: weight {:?} != [{channels}, {kernel}]",
            weight.shape()
        )));
    }
    if kernel < 2 {
        return Err(Error::Msg(format!(
            "conv1d-bwd-input composite: kernel {kernel} must be >= 2"
        )));
    }
    let dev = grad_out.device();

    // Host-read both operands in F32 (cast then to_vec, which D2H's). Both are
    // small: grad_out is [rows, channels], weight is [channels, kernel].
    let grad_f32 = if grad_out.dtype() == DType::F32 {
        grad_out.contiguous()?
    } else {
        grad_out.to_dtype(DType::F32)?.contiguous()?
    };
    let weight_f32 = if weight.dtype() == DType::F32 {
        weight.contiguous()?
    } else {
        weight.to_dtype(DType::F32)?.contiguous()?
    };
    let go: Vec<f32> = grad_f32.to_vec::<f32>()?; // row-major [rows, channels]
    let w: Vec<f32> = weight_f32.to_vec::<f32>()?; // row-major [channels, kernel]

    // grad_x[s,c] = Σ_{m=0..K-1, (s+m)<rows} weight[c, K-1-m] * grad_out[s+m, c]
    let mut gi = vec![0.0f32; rows * channels];
    for s in 0..rows {
        for c in 0..channels {
            let mut acc = 0.0f32;
            for m in 0..kernel {
                let sr = s + m;
                if sr < rows {
                    // weight[c, K-1-m]
                    acc += w[c * kernel + (kernel - 1 - m)] * go[sr * channels + c];
                }
            }
            gi[s * channels + c] = acc;
        }
    }

    // Build the F32 result on host, then move to grad_out's device.
    let out = kiln_tensor::Tensor::from_vec(gi, vec![rows, channels])?;
    if dev.is_cpu() {
        Ok(out)
    } else {
        out.to_device(dev)
    }
}

/// Device-routed `[rows, channels]` conv1d input-backward used by
/// [`CausalConv1dPrefillInputBackward`]. `Cuda(_)` keeps the validated/perf
/// FFI kernel (`kiln_rmsnorm_kernel::causal_depthwise_conv1d_bwd_input_kt`,
/// only present on `cuda` builds where that optional crate links); every other
/// device (CPU / Metal / Vulkan) routes through the device-agnostic
/// [`causal_depthwise_conv1d_bwd_input_composite`]. Mirrors the OPD kt-tape
/// `Cuda(_)` → FFI / composite-fallthrough dispatch exactly.
fn conv1d_bwd_input_rows_dispatch(
    grad_rows: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    kernel: usize,
) -> kiln_tensor::Result<kiln_tensor::Tensor> {
    #[cfg(feature = "cuda")]
    if matches!(grad_rows.device(), kiln_tensor::Device::Cuda(_)) {
        return kiln_rmsnorm_kernel::causal_depthwise_conv1d_bwd_input_kt(
            grad_rows, weight, kernel,
        )
        .map_err(|e| {
            kiln_tensor::Error::Msg(format!(
                "CausalConv1dPrefillInputBackward: bwd_input (cuda ffi): {e}"
            ))
        });
    }
    // CPU / Metal / Vulkan device-agnostic composite path.
    causal_depthwise_conv1d_bwd_input_composite(grad_rows, weight, kernel)
}

fn causal_conv1d_prefill_linear_from_state_kt(
    input: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    entry_state: &kiln_tensor::Tensor,
    kernel: usize,
) -> kiln_tensor::Result<kiln_tensor::Tensor> {
    let (batch, channels, seq_len) = input.dims3()?;
    if kernel < 2 {
        return Err(kiln_tensor::Error::Msg(format!(
            "causal_conv1d_prefill_linear_from_state_kt: kernel {kernel} must be >= 2"
        )));
    }
    if entry_state.dims() != [batch, channels, kernel - 1] {
        return Err(kiln_tensor::Error::Msg(format!(
            "causal_conv1d_prefill_linear_from_state_kt: entry_state shape {:?} != [{batch}, {channels}, {}]",
            entry_state.dims(),
            kernel - 1
        )));
    }
    let weight_2d = match weight.rank() {
        2 if weight.dims() == [channels, kernel] => weight.clone(),
        3 if weight.dims() == [channels, 1, kernel] => weight.reshape(vec![channels, kernel])?,
        _ => {
            return Err(kiln_tensor::Error::Msg(format!(
                "causal_conv1d_prefill_linear_from_state_kt: weight shape {:?} incompatible with [{channels}, {kernel}]",
                weight.dims()
            )));
        }
    };

    let input_f32 = input.to_dtype(kiln_tensor::DType::F32)?;
    let state_f32 = entry_state.to_dtype(kiln_tensor::DType::F32)?;
    let weight_f32 = weight_2d.to_dtype(kiln_tensor::DType::F32)?;
    let x_padded = kiln_tensor::Tensor::cat(&[&state_f32, &input_f32], 2)?;
    let mut output = kiln_tensor::Tensor::zeros(
        vec![batch, channels, seq_len],
        kiln_tensor::DType::F32,
        input.device(),
    )?;
    for j in 0..kernel {
        let x_slice = x_padded.narrow(2, j, seq_len)?;
        let w_j = weight_f32.narrow(1, j, 1)?.unsqueeze(0)?;
        output = kiln_tensor::ops::add(&output, &x_slice.broadcast_mul(&w_j)?)?;
    }
    Ok(output)
}

fn silu_backward_from_input_kt(
    x: &kiln_tensor::Tensor,
    grad_output: &kiln_tensor::Tensor,
) -> kiln_tensor::Result<kiln_tensor::Tensor> {
    let x_f32 = x.to_dtype(kiln_tensor::DType::F32)?;
    let dy_f32 = grad_output.to_dtype(kiln_tensor::DType::F32)?;
    let sigmoid = kiln_tensor::ops::sigmoid(&x_f32)?;
    let one_minus_sigmoid =
        kiln_tensor::ops::add_scalar(&kiln_tensor::ops::mul_scalar(&sigmoid, -1.0)?, 1.0)?;
    let sigmoid_grad = kiln_tensor::ops::mul(&sigmoid, &one_minus_sigmoid)?;
    let x_sigmoid_grad = kiln_tensor::ops::mul(&x_f32, &sigmoid_grad)?;
    let deriv = kiln_tensor::ops::add(&sigmoid, &x_sigmoid_grad)?;
    kiln_tensor::ops::mul(&dy_f32, &deriv)
}

#[derive(Debug)]
pub(crate) struct CausalConv1dPrefillInputBackward {
    /// Saved F32 conv weight `[channels, kernel]` (#1082: candle->kt; the bwd is
    /// kt-native — CUDA via FFI, every other device via the kt composite).
    weight: kiln_tensor::Tensor,
    batch: usize,
    channels: usize,
    seq_len: usize,
    /// The conv INPUT's dtype. The bwd is F32, so it computes a F32 input-grad;
    /// we cast it back to this dtype before returning so the grad-dtype-follows-
    /// tensor invariant holds at the F32↔BF16 conv boundary (the conv input
    /// `mixed_qkv_ct` is BF16 from in_proj_qkv on a BF16 model, but the conv
    /// computes/outputs F32). Without this cast the F32 grad flows up through the
    /// dtype-preserving conv-in transpose to in_proj_qkv's `MatmulBackward`,
    /// which then runs `matmul(grad_f32, weight_bf16)` → `dtype mismatch
    /// a=f32 b=bf16`. (#1082 CP-4)
    input_dtype: kiln_tensor::DType,
}

impl BackwardOp for CausalConv1dPrefillInputBackward {
    fn name(&self) -> &'static str {
        "gdn_causal_conv1d_prefill_input_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        let compute_device = grad_output.device();
        let weight = saved_tensor_for_device(&self.weight, compute_device)?;
        // #1082 kt-native: grad + the saved weight are kt. Upstream grad is
        // [B, C, T] (conv output layout). The row-block bwd is dispatched by
        // device (CUDA FFI / kt composite) in `conv1d_bwd_input_rows_dispatch`.
        let grad_f32 = if grad_output.dtype() == kiln_tensor::DType::F32 {
            grad_output.clone()
        } else {
            grad_output.to_dtype(kiln_tensor::DType::F32)?
        };
        // [B, C, T] -> [B, T, C] -> [B*T, C] (rows = time, channels = C).
        let rows = self.batch * self.seq_len;
        let grad_rows = grad_f32
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![rows, self.channels]))?;
        // weight is [channels, kernel]; the bwd takes the kernel size.
        let kernel = weight.dims()[1];
        let din_rows = conv1d_bwd_input_rows_dispatch(&grad_rows, &weight, kernel)?;
        // [B*T, C] -> [B, T, C] -> [B, C, T] (back to the conv input layout).
        let din = din_rows
            .reshape(vec![self.batch, self.seq_len, self.channels])
            .and_then(|t| t.transpose(1, 2))
            .and_then(|t| t.contiguous())?;
        // Cast the F32 grad back to the conv input's dtype (grad-dtype-follows-
        // tensor across the F32↔BF16 conv boundary). No-op when F32.
        let din = if din.dtype() == self.input_dtype {
            din
        } else {
            din.to_dtype(self.input_dtype)?
        };
        Ok(vec![Some(din)])
    }
}

#[derive(Debug)]
pub(crate) struct CausalConv1dPrefillSiluInputBackward {
    input: kiln_tensor::Tensor,
    weight: kiln_tensor::Tensor,
    entry_state: kiln_tensor::Tensor,
    batch: usize,
    channels: usize,
    seq_len: usize,
    input_dtype: kiln_tensor::DType,
}

impl BackwardOp for CausalConv1dPrefillSiluInputBackward {
    fn name(&self) -> &'static str {
        "gdn_causal_conv1d_prefill_silu_input_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        let compute_device = grad_output.device();
        let input = saved_tensor_for_device(&self.input, compute_device)?;
        let weight = saved_tensor_for_device(&self.weight, compute_device)?;
        let entry_state = saved_tensor_for_device(&self.entry_state, compute_device)?;
        let pre_silu = causal_conv1d_prefill_linear_from_state_kt(
            &input,
            &weight,
            &entry_state,
            weight.dims()[1],
        )?;
        let grad_conv = silu_backward_from_input_kt(&pre_silu, grad_output)?;
        let rows = self.batch * self.seq_len;
        let grad_rows = grad_conv
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![rows, self.channels]))?;
        let din_rows = conv1d_bwd_input_rows_dispatch(&grad_rows, &weight, weight.dims()[1])?;
        let din = din_rows
            .reshape(vec![self.batch, self.seq_len, self.channels])
            .and_then(|t| t.transpose(1, 2))
            .and_then(|t| t.contiguous())?;
        let din = if din.dtype() == self.input_dtype {
            din
        } else {
            din.to_dtype(self.input_dtype)?
        };
        Ok(vec![Some(din)])
    }
}

/// Route the GDN PREFILL causal depthwise conv1d through the kt `Tape`.
///
/// The production forward already computed `out` (`causal_conv1d_prefill`'s
/// `[B, C, T]` F32 output, BEFORE the SiLU + the transpose-back — those record
/// via `try_tape_silu_cuda` / `try_tape_transpose_cuda`); this records-only (no
/// re-run), borrowing `out` as the recorded node's output. The recorded
/// [`CausalConv1dPrefillInputBackward`] flows the conv output grad back to its
/// `[B, C, T]` input (and thence, via the conv-in transpose, to the in_proj_qkv
/// LoRA `Var`). This is the single largest GDN gap — without it the q/k/v reach
/// `mixed_qkv` but the chain severs before in_proj_qkv.
///
/// Reuses the existing `KILN_USE_TAPE_GDN_CONV` gate (mirroring
/// `try_tape_causal_conv1d_cuda`). `Ok(None)` (caller's production output
/// unchanged) when the gate is off, no tape scope is active, the inputs are on
/// an unsupported device, `batch != 1`, shapes disagree, or a kt borrow fails
/// — NEVER an error.
/// kt-native GDN prefill causal-depthwise-conv1d tape recorder (#1082 seam flip).
/// Record-only: builds the F32 `[channels, kernel]` weight view in kt + records
/// the device-agnostic `CausalConv1dPrefillInputBackward` linking kt `out` back to
/// kt `input`. The recorded backward dispatches the row-block input-grad by device
/// — `Cuda(_)` → the validated FFI kernel
/// (`kiln_rmsnorm_kernel::causal_depthwise_conv1d_bwd_input_kt`), CPU / Metal /
/// Vulkan → the pure-`kiln_tensor` [`causal_depthwise_conv1d_bwd_input_composite`].
/// This un-gates GDN-on-Vulkan training (the last production blocker for the
/// GDN-heavy Qwen3.5-4B Vulkan path); no candle bridge.
pub fn try_tape_causal_conv1d_prefill_kt(
    input: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    kernel: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() || !tape_gdn_conv_enabled() {
        return Ok(None);
    }
    // The bwd is available with any GPU-training backend: CUDA FFI kernel, or the
    // device-agnostic kt composite for Metal/Vulkan. Declines when none is on.
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )))]
    {
        let _ = (input, weight, out, kernel);
        return Ok(None);
    }
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    {
        // The recorded backward is device-agnostic (CUDA FFI / kt composite), so the
        // recorder admits CUDA, Metal, and Vulkan. CPU is excluded here only because
        // the production GDN forward this hooks runs on an accelerator device.
        if !matches!(
            input.device(),
            kiln_tensor::Device::Cuda(_)
                | kiln_tensor::Device::Metal(_)
                | kiln_tensor::Device::Vulkan(_)
                | kiln_tensor::Device::Rocm(_)
        ) || !matches!(
            out.device(),
            kiln_tensor::Device::Cuda(_)
                | kiln_tensor::Device::Metal(_)
                | kiln_tensor::Device::Vulkan(_)
                | kiln_tensor::Device::Rocm(_)
        ) || !matches!(
            weight.device(),
            kiln_tensor::Device::Cuda(_)
                | kiln_tensor::Device::Metal(_)
                | kiln_tensor::Device::Vulkan(_)
                | kiln_tensor::Device::Rocm(_)
        ) {
            return Ok(None);
        }
        if kernel < 2 {
            return Ok(None);
        }
        let (batch, channels, seq_len) = match input.dims3() {
            Ok(d) => d,
            Err(_) => return Ok(None),
        };
        if batch != 1 || out.dims() != [batch, channels, seq_len].as_slice() {
            return Ok(None);
        }
        let weight_2d = match weight.rank() {
            2 => match weight.dims2() {
                Ok((c, k)) if c == channels && k == kernel => weight.clone(),
                _ => return Ok(None),
            },
            3 => match weight.dims3() {
                Ok((c, one, k)) if c == channels && one == 1 && k == kernel => {
                    match weight.reshape(vec![channels, kernel]) {
                        Ok(w) => w,
                        Err(_) => return Ok(None),
                    }
                }
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        let weight_f32 = if weight_2d.dtype() == kiln_tensor::DType::F32 {
            weight_2d
        } else {
            match weight_2d.to_dtype(kiln_tensor::DType::F32) {
                Ok(w) => w,
                Err(_) => return Ok(None),
            }
        };
        let weight_f32 = match weight_f32.contiguous() {
            Ok(w) => w,
            Err(_) => return Ok(None),
        };
        let input_dtype = input.dtype();
        let recorded = with_active_tape(|tape: &mut Tape| -> Result<()> {
            let offload_saved = tape_gdn_saved_tensor_offload_enabled(&input.device());
            tape.record(
                out,
                &[input],
                Box::new(CausalConv1dPrefillInputBackward {
                    weight: maybe_offload_gdn_saved_tensor(&weight_f32, offload_saved)
                        .context("try_tape_causal_conv1d_prefill_kt: save weight")?,
                    batch,
                    channels,
                    seq_len,
                    input_dtype,
                }),
            );
            Ok(())
        });
        match recorded {
            Some(result) => result?,
            None => return Ok(None),
        }
        Ok(Some(out.clone()))
    }
}

pub fn try_tape_causal_conv1d_prefill_silu_kt(
    input: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    entry_state: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    kernel: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() || !tape_gdn_conv_enabled() {
        return Ok(None);
    }
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )))]
    {
        let _ = (input, weight, entry_state, out, kernel);
        return Ok(None);
    }
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    {
        if !matches!(
            input.device(),
            kiln_tensor::Device::Cuda(_)
                | kiln_tensor::Device::Metal(_)
                | kiln_tensor::Device::Vulkan(_)
                | kiln_tensor::Device::Rocm(_)
        ) || !matches!(
            out.device(),
            kiln_tensor::Device::Cuda(_)
                | kiln_tensor::Device::Metal(_)
                | kiln_tensor::Device::Vulkan(_)
                | kiln_tensor::Device::Rocm(_)
        ) || !matches!(
            weight.device(),
            kiln_tensor::Device::Cuda(_)
                | kiln_tensor::Device::Metal(_)
                | kiln_tensor::Device::Vulkan(_)
                | kiln_tensor::Device::Rocm(_)
        ) || !matches!(
            entry_state.device(),
            kiln_tensor::Device::Cuda(_)
                | kiln_tensor::Device::Metal(_)
                | kiln_tensor::Device::Vulkan(_)
                | kiln_tensor::Device::Rocm(_)
        ) {
            return Ok(None);
        }
        if kernel < 2 {
            return Ok(None);
        }
        let (batch, channels, seq_len) = match input.dims3() {
            Ok(d) => d,
            Err(_) => return Ok(None),
        };
        if batch != 1
            || out.dims() != [batch, channels, seq_len].as_slice()
            || entry_state.dims() != [batch, channels, kernel - 1]
        {
            return Ok(None);
        }
        let weight_2d = match weight.rank() {
            2 => match weight.dims2() {
                Ok((c, k)) if c == channels && k == kernel => weight.clone(),
                _ => return Ok(None),
            },
            3 => match weight.dims3() {
                Ok((c, one, k)) if c == channels && one == 1 && k == kernel => {
                    match weight.reshape(vec![channels, kernel]) {
                        Ok(w) => w,
                        Err(_) => return Ok(None),
                    }
                }
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        let weight_f32 = if weight_2d.dtype() == kiln_tensor::DType::F32 {
            weight_2d
        } else {
            match weight_2d.to_dtype(kiln_tensor::DType::F32) {
                Ok(w) => w,
                Err(_) => return Ok(None),
            }
        };
        let weight_f32 = match weight_f32.contiguous() {
            Ok(w) => w,
            Err(_) => return Ok(None),
        };
        let entry_state_f32 = if entry_state.dtype() == kiln_tensor::DType::F32 {
            entry_state.clone()
        } else {
            match entry_state.to_dtype(kiln_tensor::DType::F32) {
                Ok(s) => s,
                Err(_) => return Ok(None),
            }
        };
        let entry_state_f32 = match entry_state_f32.contiguous() {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        let input_dtype = input.dtype();
        let recorded = with_active_tape(|tape: &mut Tape| -> Result<()> {
            let offload_saved = tape_gdn_saved_tensor_offload_enabled(&input.device());
            tape.record(
                out,
                &[input],
                Box::new(CausalConv1dPrefillSiluInputBackward {
                    input: maybe_offload_gdn_saved_tensor(input, offload_saved)
                        .context("try_tape_causal_conv1d_prefill_silu_kt: save input")?,
                    weight: maybe_offload_gdn_saved_tensor(&weight_f32, offload_saved)
                        .context("try_tape_causal_conv1d_prefill_silu_kt: save weight")?,
                    entry_state: maybe_offload_gdn_saved_tensor(&entry_state_f32, offload_saved)
                        .context("try_tape_causal_conv1d_prefill_silu_kt: save entry_state")?,
                    batch,
                    channels,
                    seq_len,
                    input_dtype,
                }),
            );
            Ok(())
        });
        match recorded {
            Some(result) => result?,
            None => return Ok(None),
        }
        Ok(Some(out.clone()))
    }
}

/// Candle-composite tape backward for the GDN L2-qk-norm `y =
/// l2_normalize(x) * scale`. Wraps [`gdn_l2_norm_scale_backward_no_grad`]
/// (analytic adjoint in candle F32), mirroring how [`GdnRecurrentBackward`]
/// wraps `gdn_recurrent_backward_no_grad`.
///
/// One differentiable input (`x`); `scale` is a non-differentiable constant
/// folded into the adjoint (Q uses `1/sqrt(dk)`, K uses `1.0`). `eps` matches
/// `l2_normalize`'s hard-coded `1e-6`.
#[derive(Debug)]
pub(crate) struct GdnL2NormScaleBackward {
    x: kiln_tensor::Tensor, // #1082: candle->kt (the bwd kernel is already kt-native).
    scale: f64,
    eps: f64,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn try_gdn_l2_norm_scale_backward_fused_cuda_rocm(
    x: &kiln_tensor::Tensor,
    scale: f64,
    eps: f64,
    grad_output: &kiln_tensor::Tensor,
) -> kiln_tensor::Result<Option<kiln_tensor::Tensor>> {
    if !matches!(
        x.device(),
        kiln_tensor::Device::Cuda(_) | kiln_tensor::Device::Rocm(_)
    ) || x.dtype() != kiln_tensor::DType::BF16
        || grad_output.dtype() != kiln_tensor::DType::BF16
        || x.dims() != grad_output.dims()
    {
        return Ok(None);
    }

    let x_dims = x.dims().to_vec();
    let Some(&hidden) = x_dims.last() else {
        return Ok(None);
    };
    let rows = x_dims.iter().take(x_dims.len() - 1).product::<usize>();
    if rows == 0 {
        return Ok(None);
    }

    let flatten = |t: &kiln_tensor::Tensor,
                   name: &'static str|
     -> kiln_tensor::Result<kiln_tensor::Tensor> {
        t.contiguous()
            .and_then(|t| t.reshape(vec![rows, hidden]))
            .map_err(|e| {
                kiln_tensor::Error::Msg(format!("GdnL2NormScaleBackward fused {name} flatten: {e}"))
            })
    };

    let x_flat = flatten(x, "x")?;
    let grad_flat = flatten(grad_output, "grad_output")?;
    if !kiln_gdn_kernel::gdn_l2_norm_scale_bwd_supports_kt(&grad_flat, &x_flat) {
        return Ok(None);
    }

    let dx = kiln_gdn_kernel::gdn_l2_norm_scale_bwd_bf16_kt(
        &grad_flat,
        &x_flat,
        scale as f32,
        eps as f32,
    )
    .map_err(|e| kiln_tensor::Error::Msg(format!("GdnL2NormScaleBackward fused bwd: {e}")))?;
    dx.reshape(x_dims).map(Some).map_err(|e| {
        kiln_tensor::Error::Msg(format!("GdnL2NormScaleBackward fused dx reshape: {e}"))
    })
}

impl BackwardOp for GdnL2NormScaleBackward {
    fn name(&self) -> &'static str {
        "gdn_l2_norm_scale_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        let x = saved_tensor_for_device(&self.x, grad_output.device())?;
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        if let Some(dx) =
            try_gdn_l2_norm_scale_backward_fused_cuda_rocm(&x, self.scale, self.eps, grad_output)?
        {
            return Ok(vec![Some(dx)]);
        }

        // #1082 kt-native: `x` is stored kt now (no candle bridge); the bwd kernel
        // is already kt-typed.
        let dx = gdn_l2_norm_scale_backward_no_grad(&x, self.scale, self.eps, grad_output)
            .map_err(|e| kiln_tensor::Error::Msg(format!("GdnL2NormScaleBackward: bwd: {e}")))?;
        // Adjoint can be non-contiguous (broadcast_mul views) — contiguify.
        let dx_kt = dx
            .contiguous()
            .map_err(|e| kiln_tensor::Error::Msg(format!("GdnL2NormScaleBackward: contig: {e}")))?;
        Ok(vec![Some(dx_kt)])
    }
}

/// Route a GDN L2-qk-norm `y = l2_normalize(x) * scale` through the kt `Tape`.
///
/// Used for BOTH halves of `gdn_qk_norm`: Q (`scale = 1/sqrt(dk)`) and K
/// (`scale = 1.0`). The forward runs candle `l2_normalize`-then-scale via the
/// existing production helpers; the recorded [`GdnL2NormScaleBackward`] emits
/// the per-input grad so a tape-authoritative backward reaches the conv / split
/// (and thence the in_proj LoRA Vars).
///
/// `Ok(None)` (caller falls through to the existing `gdn_qk_norm`) when the gate
/// is off, no tape scope is active, the input isn't CUDA, or a kt borrow fails.
/// kt-native L2-norm-scale tape recorder (#1082 seam flip) — kt-only twin of
/// [`try_tape_gdn_l2_norm_scale_cuda`]. Record-only: records the (now kt-native)
/// `GdnL2NormScaleBackward` (stores kt `x`; bwd kernel already kt) linking kt `out`
/// back to kt `x`. No candle round-trip.
pub fn try_tape_gdn_l2_norm_scale_kt(
    x: &kiln_tensor::Tensor,
    scale: f64,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() || !tape_gdn_qk_norm_enabled() {
        return Ok(None);
    }
    if !matches!(
        x.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }
    if x.dims() != out.dims() {
        return Ok(None);
    }
    let recorded = with_active_tape(|tape: &mut Tape| -> Result<()> {
        let offload_saved = tape_gdn_saved_tensor_offload_enabled(&x.device());
        tape.record(
            out,
            &[x],
            Box::new(GdnL2NormScaleBackward {
                x: maybe_offload_gdn_saved_tensor(x, offload_saved)
                    .context("try_tape_gdn_l2_norm_scale_kt: save x")?,
                scale,
                eps: 1e-6,
            }),
        );
        Ok(())
    });
    match recorded {
        Some(result) => result?,
        None => return Ok(None),
    }
    Ok(Some(out.clone()))
}

/// Candle-composite tape backward for the GDN gated RMSNorm `out =
/// rms_norm(x, weight) * silu(z)`. Wraps
/// [`gdn_gated_rms_norm_backward_no_grad`] (analytic adjoint in candle F32),
/// mirroring how [`GdnRecurrentBackward`] wraps `gdn_recurrent_backward_no_grad`.
///
/// Three differentiable inputs `[x, z, weight]` in that order. `x` is the
/// recurrence output (head-LAST `[B, T, nv, dv]`), `z` is the output gate,
/// `weight` is the GDN `norm` `Var`.
#[derive(Debug)]
pub(crate) struct GdnGatedRmsNormBackward {
    // #1082: candle->kt (the bwd kernel gdn_gated_rms_norm_backward_no_grad is kt-native).
    x: kiln_tensor::Tensor,
    z: kiln_tensor::Tensor,
    weight: kiln_tensor::Tensor,
    eps: f64,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn try_gdn_gated_rms_norm_backward_fused_cuda_rocm(
    x: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f64,
    grad_output: &kiln_tensor::Tensor,
) -> kiln_tensor::Result<
    Option<(
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
    )>,
> {
    if !matches!(
        x.device(),
        kiln_tensor::Device::Cuda(_) | kiln_tensor::Device::Rocm(_)
    ) || x.dtype() != kiln_tensor::DType::BF16
        || z.dtype() != kiln_tensor::DType::BF16
        || !matches!(
            weight.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
        )
        || grad_output.dtype() != kiln_tensor::DType::BF16
        || x.dims() != z.dims()
        || x.dims() != grad_output.dims()
    {
        return Ok(None);
    }

    let x_dims = x.dims().to_vec();
    let Some(&hidden) = x_dims.last() else {
        return Ok(None);
    };
    let rows = x_dims.iter().take(x_dims.len() - 1).product::<usize>();
    if rows == 0 {
        return Ok(None);
    }

    let flatten =
        |t: &kiln_tensor::Tensor, name: &'static str| -> kiln_tensor::Result<kiln_tensor::Tensor> {
            t.contiguous()
                .and_then(|t| t.reshape(vec![rows, hidden]))
                .map_err(|e| {
                    kiln_tensor::Error::Msg(format!(
                        "GdnGatedRmsNormBackward fused {name} flatten: {e}"
                    ))
                })
        };

    let x_flat = flatten(x, "x")?;
    let z_flat = flatten(z, "z")?;
    let grad_flat = flatten(grad_output, "grad_output")?;
    if !kiln_gdn_kernel::gdn_gated_rms_norm_bwd_supports_kt(&grad_flat, &x_flat, &z_flat, weight) {
        return Ok(None);
    }

    let grads = match weight.dtype() {
        kiln_tensor::DType::BF16 => kiln_gdn_kernel::gdn_gated_rms_norm_bwd_bf16_kt(
            &grad_flat, &x_flat, &z_flat, weight, eps as f32,
        )
        .map_err(|e| kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward fused bwd: {e}")))?,
        kiln_tensor::DType::F32 => kiln_gdn_kernel::gdn_gated_rms_norm_bwd_bf16_f32_weight_kt(
            &grad_flat, &x_flat, &z_flat, weight, eps as f32,
        )
        .map_err(|e| {
            kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward fused bwd f32-weight: {e}"))
        })?,
        _ => return Ok(None),
    };
    let dx = grads.dx.reshape(x_dims.clone()).map_err(|e| {
        kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward fused dx reshape: {e}"))
    })?;
    let dz = grads.dz.reshape(x_dims).map_err(|e| {
        kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward fused dz reshape: {e}"))
    })?;
    let dw = if grads.dw.dtype() == weight.dtype() {
        grads.dw
    } else {
        grads.dw.to_dtype(weight.dtype()).map_err(|e| {
            kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward fused dw dtype: {e}"))
        })?
    };

    Ok(Some((dx, dz, dw)))
}

impl BackwardOp for GdnGatedRmsNormBackward {
    fn name(&self) -> &'static str {
        "gdn_gated_rms_norm_backward"
    }
    fn input_count(&self) -> usize {
        // x, z, weight.
        3
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        let compute_device = grad_output.device();
        let x = saved_tensor_for_device(&self.x, compute_device)?;
        let z = saved_tensor_for_device(&self.z, compute_device)?;
        let weight = saved_tensor_for_device(&self.weight, compute_device)?;
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        if let Some((dx, dz, dw)) =
            try_gdn_gated_rms_norm_backward_fused_cuda_rocm(&x, &z, &weight, self.eps, grad_output)?
        {
            return Ok(vec![Some(dx), Some(dz), Some(dw)]);
        }

        // #1082 kt-native: x/z/weight are stored kt now (no candle bridge); the bwd
        // kernel is already kt-typed.
        let grads = gdn_gated_rms_norm_backward_no_grad(&x, &z, &weight, self.eps, grad_output)
            .map_err(|e| kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward: bwd: {e}")))?;
        // Adjoints (kt) can be non-contiguous; contiguify.
        let to_kt = |t: &kiln_tensor::Tensor| -> kiln_tensor::Result<kiln_tensor::Tensor> {
            t.contiguous().map_err(|e| {
                kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward: grad contiguous: {e}"))
            })
        };
        Ok(vec![
            Some(to_kt(&grads.dx)?),
            Some(to_kt(&grads.dz)?),
            Some(to_kt(&grads.dw)?),
        ])
    }
}

/// Route the GDN gated RMSNorm `out = rms_norm(x, weight) * silu(z)` through
/// the kt `Tape`.
///
/// The production forward (`gated_rms_norm`) already computed `out`; this
/// records-only (no re-run), borrowing `out` as the node output — like
/// `tape_record_gdn_recurrent`. The recorded [`GdnGatedRmsNormBackward`] emits
/// `dx`/`dz`/`dw` so a tape-authoritative backward reaches the recurrence
/// output (`x`), the gate (`z`), and the `norm` `Var` (`weight`).
///
/// `Ok(None)` (caller falls through) when the gate is off, no tape scope is
/// active, the inputs aren't CUDA, shapes disagree, or a kt borrow fails.
/// kt-native gated-RMSNorm tape recorder (#1082 seam flip) — kt-only twin of
/// `try_tape_gdn_gated_rms_norm_cuda`. Record-only: records the (now kt-native)
/// `GdnGatedRmsNormBackward` (stores kt x/z/weight; bwd kernel already kt) linking
/// kt `out` back to kt x/z/weight. No candle round-trip.
pub fn try_tape_gdn_gated_rms_norm_kt(
    x: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f64,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() || !tape_gdn_gated_norm_enabled() {
        return Ok(None);
    }
    if !matches!(
        x.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }
    if x.dims() != z.dims() || x.dims() != out.dims() {
        return Ok(None);
    }
    if weight.rank() != 1 || *x.dims().last().unwrap() != weight.dims()[0] {
        return Ok(None);
    }
    let recorded = with_active_tape(|tape: &mut Tape| -> Result<()> {
        let offload_saved = tape_gdn_saved_tensor_offload_enabled(&x.device());
        tape.record(
            out,
            &[x, z, weight],
            Box::new(GdnGatedRmsNormBackward {
                x: maybe_offload_gdn_saved_tensor(x, offload_saved)
                    .context("try_tape_gdn_gated_rms_norm_kt: save x")?,
                z: maybe_offload_gdn_saved_tensor(z, offload_saved)
                    .context("try_tape_gdn_gated_rms_norm_kt: save z")?,
                weight: maybe_offload_gdn_saved_tensor(weight, offload_saved)
                    .context("try_tape_gdn_gated_rms_norm_kt: save weight")?,
                eps,
            }),
        );
        Ok(())
    });
    match recorded {
        Some(result) => result?,
        None => return Ok(None),
    }
    Ok(Some(out.clone()))
}

// ===========================================================================
// CP-4 Increment 3 (#1082): GDN forward-chain layout/dtype ops — cast,
// narrow, GQA head-expand.
//
// These three adapters wrap the layout/dtype ops on the GDN PREFILL path
// (`gated_deltanet_forward_decode_if`, training path: `seq_len>1`, BF16,
// `track_op=true` so the fused fast paths are OFF) that sit BETWEEN the
// already-wired GDN nodes (in_proj/out_proj keystone, qk_norm,
// gated_rms_norm, recurrence record, transpose) and otherwise SEVER the
// chain back to the in_proj_qkv/z LoRA `Var`s. Every one mints a fresh
// candle id, so an unwrapped op fragments the tape and the GDN LoRA grads
// never flow.
//
// All three use a CANDLE-COMPOSITE backward (grad kt→candle, compute,
// candle→kt), NOT a kt-native `kiln_autograd` `BackwardOp`, because:
//   * `CastBackward` / `NarrowBackward` in kiln-autograd downcast the grad
//     storage to `CpuStorage` (NarrowBackward) or are otherwise CPU-leaning;
//     a CUDA grad would bail. The candle composite is unambiguously
//     CUDA-safe and value-faithful (cast is a dtype round-trip; narrow's
//     adjoint is a zero-pad; the GQA expand's adjoint is a reshape+sum).
//   * Mirrors the proven `GdnRecurrentBackward` / `GdnL2NormScaleBackward` /
//     `SdpaBackward` candle-composite pattern already in this module.
//
// Gated on `tape_forward_enabled()` + an active tape scope only (pure
// layout/dtype ops with trivial, always-safe adjoints — same contract as
// `try_tape_reshape_cuda` / `try_tape_transpose_cuda`). `Ok(None)` on any
// gate-off / non-CUDA / envelope-miss / kt-borrow failure — NEVER an error.
// ===========================================================================

/// Candle-composite tape backward for a float dtype cast (`to_dtype`). The
/// adjoint casts the upstream grad back to the forward input's dtype — a
/// value-faithful round-trip in the F32↔BF16 space the GDN path uses
/// (`v.to_dtype(input_dtype)` before recurrence; `attn_out.to_dtype(input_dtype)`
/// after gated-RMSNorm). One differentiable input (`x`).
#[derive(Debug)]
pub(crate) struct CastCompositeBackward {
    /// The kt dtype of the forward INPUT — backward casts the grad to it. (#1082:
    /// was `CandleDType`; now kt so the backward is candle-free.)
    source_dtype: kiln_tensor::DType,
}

impl BackwardOp for CastCompositeBackward {
    fn name(&self) -> &'static str {
        "cast_composite_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        // #1082 kt-native: cast the upstream grad back to the forward input's dtype
        // with kt ops — no kt->candle->kt bridge. (cast adjoint = cast-back.)
        let dx = if grad_output.dtype() == self.source_dtype {
            grad_output.clone()
        } else {
            grad_output.to_dtype(self.source_dtype)?
        };
        let dx = dx.contiguous()?;
        Ok(vec![Some(dx)])
    }
}

/// Route a float dtype cast `out = x.to_dtype(target)` through the kt `Tape`.
///
/// The production forward already computed `out` (the caller passes it in);
/// this records-only (no re-run), borrowing `out` as the recorded node's
/// output — like `tape_record_gdn_recurrent`. The recorded
/// [`CastCompositeBackward`] casts the upstream grad back to `x`'s dtype so a
/// tape-authoritative backward stays connected across the cast.
///
/// Used on the GDN path for `v.to_dtype(input_dtype)` (BF16→/F32→BF16 before
/// recurrence) and `attn_out.to_dtype(input_dtype)` (after gated-RMSNorm,
/// before out_proj). `Ok(None)` when the gate is off, no tape scope is active,
/// the inputs aren't CUDA, shapes disagree, or a kt borrow fails.
/// kt-native cast tape recorder (#1082 seam flip) — kt-only twin of
/// `try_tape_cast_cuda`. Record-only: the forward cast is already done by the kt
/// non-tape path; this records the (now kt-native) `CastCompositeBackward` linking
/// the kt `out` back to the kt `x`. No candle round-trip.
pub fn try_tape_cast_kt(
    x: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    if !matches!(
        x.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) || !matches!(
        out.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }
    if x.dims() != out.dims() {
        return Ok(None);
    }
    // No-op cast (same dtype, or out IS x): the chain already flows through x.
    if x.dtype() == out.dtype() || x.id() == out.id() {
        return Ok(None);
    }
    let recorded = with_active_tape(|tape: &mut Tape| {
        tape.record(
            out,
            &[x],
            Box::new(CastCompositeBackward {
                source_dtype: x.dtype(),
            }),
        );
    });
    if recorded.is_none() {
        return Ok(None);
    }
    Ok(Some(out.clone()))
}

/// Candle-composite tape backward for `out = narrow(x, axis, offset, length)`.
/// The adjoint embeds the upstream grad into a zero-filled tensor of the
/// original `x.shape` at `[offset .. offset+length]` along `axis` (the
/// standard "zero-pad" narrow adjoint). One differentiable input (`x`).
///
/// CUDA-safe (unlike `kiln_autograd::NarrowBackward`, which downcasts the grad
/// to `CpuStorage` and bails on a CUDA grad): runs the zero-pad in candle by
/// `slice_assign`-ing the grad into a zeros tensor.
#[derive(Debug)]
pub(crate) struct NarrowCompositeBackward {
    axis: usize,
    offset: usize,
    length: usize,
    /// `x.shape` from the forward (the target shape for `d_x`).
    source_shape: Vec<usize>,
    /// `x.dtype` so the zero-fill matches. (#1082: candle->kt.)
    source_dtype: kiln_tensor::DType,
}

impl BackwardOp for NarrowCompositeBackward {
    fn name(&self) -> &'static str {
        "narrow_composite_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        // #1082 kt-native: cast the grad to the source dtype with kt ops, then
        // the narrow adjoint = zero-pad — place the grad at [offset..offset+length]
        // along `axis`, zeros before/after — built kt-native as
        // `cat([zeros_left, grad, zeros_right], axis)` (kt has no pad_with_zeros).
        // No kt->candle->kt bridge.
        let grad = if grad_output.dtype() == self.source_dtype {
            grad_output.clone()
        } else {
            grad_output.to_dtype(self.source_dtype)?
        };
        let source_axis_len = self.source_shape[self.axis];
        let right = source_axis_len - self.offset - self.length;
        let dev = grad.device();
        let mut left_sh = self.source_shape.clone();
        left_sh[self.axis] = self.offset;
        let mut right_sh = self.source_shape.clone();
        right_sh[self.axis] = right;
        let dx = match (self.offset > 0, right > 0) {
            (true, true) => {
                let lz = kiln_tensor::Tensor::zeros(left_sh, self.source_dtype, &dev)?;
                let rz = kiln_tensor::Tensor::zeros(right_sh, self.source_dtype, &dev)?;
                kiln_tensor::Tensor::cat(&[&lz, &grad, &rz], self.axis)?
            }
            (true, false) => {
                let lz = kiln_tensor::Tensor::zeros(left_sh, self.source_dtype, &dev)?;
                kiln_tensor::Tensor::cat(&[&lz, &grad], self.axis)?
            }
            (false, true) => {
                let rz = kiln_tensor::Tensor::zeros(right_sh, self.source_dtype, &dev)?;
                kiln_tensor::Tensor::cat(&[&grad, &rz], self.axis)?
            }
            (false, false) => grad,
        };
        let dx = dx.contiguous()?;
        Ok(vec![Some(dx)])
    }
}

/// Route `out = narrow(x, axis, offset, length)` through the kt `Tape`.
///
/// The production forward already computed `out` (the caller's
/// `x.narrow(axis, offset, length)`); this records-only (no re-run), borrowing
/// `out` as the recorded node's output. The recorded [`NarrowCompositeBackward`]
/// zero-pads the upstream grad back to `x.shape` so a tape-authoritative
/// backward stays connected across the slice.
///
/// Used on the GDN path for the QKV split (`mixed_qkv.narrow(2, ·, ·)` → q/k/v).
/// `Ok(None)` on any gate-off / non-CUDA / shape-envelope-miss / kt-borrow
/// failure.
/// kt-native narrow tape recorder (#1082 seam flip) — kt-only twin of
/// `try_tape_narrow_cuda`. Record-only: the forward narrow is already done by the
/// kt non-tape path; records the (now kt-native) `NarrowCompositeBackward` linking
/// the kt `out` back to the kt `x`. No candle round-trip.
pub fn try_tape_narrow_kt(
    x: &kiln_tensor::Tensor,
    axis: usize,
    offset: usize,
    length: usize,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    if !matches!(
        x.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) || !matches!(
        out.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }
    let x_dims = x.dims().to_vec();
    if axis >= x_dims.len() || offset + length > x_dims[axis] {
        return Ok(None);
    }
    let out_dims = out.dims();
    if out_dims.len() != x_dims.len() || out_dims[axis] != length {
        return Ok(None);
    }
    for (d, (&od, &xd)) in out_dims.iter().zip(x_dims.iter()).enumerate() {
        if d != axis && od != xd {
            return Ok(None);
        }
    }
    let recorded = with_active_tape(|tape: &mut Tape| {
        tape.record(
            out,
            &[x],
            Box::new(NarrowCompositeBackward {
                axis,
                offset,
                length,
                source_shape: x_dims.clone(),
                source_dtype: x.dtype(),
            }),
        );
    });
    if recorded.is_none() {
        return Ok(None);
    }
    Ok(Some(out.clone()))
}

/// Candle-composite tape backward for the GDN GQA head-expand
/// (`x.unsqueeze(3).expand(...).contiguous().reshape(...)`):
/// `[B, T, nk, dk] → [B, T, nv, dk]` where `nv = nk * gqa_ratio` (each KV head
/// is broadcast across its `gqa_ratio` query heads). The adjoint sums the
/// upstream grad over the broadcast (gqa_ratio) sub-dim back to `nk`:
/// `grad[B,T,nv,dk] → reshape[B,T,nk,gqa_ratio,dk] → sum(dim=3) → [B,T,nk,dk]`.
/// One differentiable input (`x`).
#[derive(Debug)]
pub(crate) struct GqaExpandBackward {
    batch: usize,
    seq_len: usize,
    nk: usize,
    gqa_ratio: usize,
    head_dim: usize,
}

impl BackwardOp for GqaExpandBackward {
    fn name(&self) -> &'static str {
        "gdn_gqa_expand_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        // grad: [B, T, nv, dk] -> [B, T, nk, gqa_ratio, dk] -> sum over gqa_ratio
        // (axis 3) -> [B, T, nk, dk]. (#1082) kt-NATIVE: was a kt->candle->kt
        // round-trip (2 device copies + candle `.sum(3)`); now a single CUDA
        // `sum_axis` (routes to the `cuda_sum_axis` kernel — verified CUDA-native,
        // not a CPU fallback). No candle, no copies. `sum_axis` returns a fresh
        // contiguous reduction so the old explicit `.contiguous()` is unneeded.
        let g = grad_output
            .reshape((
                self.batch,
                self.seq_len,
                self.nk,
                self.gqa_ratio,
                self.head_dim,
            ))
            .map_err(|e| kiln_tensor::Error::Msg(format!("GqaExpandBackward: reshape: {e}")))?;
        let dx = kiln_tensor::ops::sum_axis(&g, 3)
            .map_err(|e| kiln_tensor::Error::Msg(format!("GqaExpandBackward: sum_axis: {e}")))?;
        Ok(vec![Some(dx)])
    }
}

/// Route the GDN GQA head-expand (`x: [B,T,nk,dk] → out: [B,T,nv,dk]`,
/// `nv = nk*gqa_ratio`) through the kt `Tape`.
///
/// The production forward already computed `out` (the
/// `unsqueeze(3).expand(...).contiguous().reshape(...)` chain at the GDN
/// `head_expand` / `head_expand_recur_fallback` sites); this records-only (no
/// re-run), borrowing `out` as the recorded node's output. The recorded
/// [`GqaExpandBackward`] sums the grad over the broadcast head sub-dim so a
/// tape-authoritative backward stays connected from the post-norm q/k back to
/// the pre-expand (post-split) q/k — and thence the in_proj_qkv LoRA `Var`.
///
/// `Ok(None)` on any gate-off / non-CUDA / shape-envelope-miss / kt-borrow
/// failure.
/// kt-native GQA head-expand tape recorder (#1082 seam flip) — kt-only twin of
/// `try_tape_gqa_expand_cuda`. Record-only: the forward expand is already done
/// by the kt non-tape path; this records `GqaExpandBackward` (all-usize fields, no
/// candle types) linking the kt `out` back to the kt `x`. No candle round-trip.
pub fn try_tape_gqa_expand_kt(
    x: &kiln_tensor::Tensor,
    gqa_ratio: usize,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    if !matches!(
        x.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) || !matches!(
        out.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }
    if gqa_ratio <= 1 {
        return Ok(None);
    }
    let (batch, seq_len, nk, head_dim) = match x.dims4() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let nv = nk * gqa_ratio;
    if out.dims() != [batch, seq_len, nv, head_dim].as_slice() {
        return Ok(None);
    }
    let recorded = with_active_tape(|tape: &mut Tape| {
        tape.record(
            out,
            &[x],
            Box::new(GqaExpandBackward {
                batch,
                seq_len,
                nk,
                gqa_ratio,
                head_dim,
            }),
        );
    });
    if recorded.is_none() {
        return Ok(None);
    }
    Ok(Some(out.clone()))
}

// ===========================================================================
// CP-4 (#1082): naive SDPA-fallback attention tape coverage.
//
// `try_tape_flash_attn_cuda` only fires when flash-attention is available
// (`head_dim ∈ {128, 256}`). The GQA full-attention block's NON-flash
// fallback (`forward::gqa_attention_core_prefill`'s naive scaled-dot-product
// path) is the path that runs at every other head_dim — notably the tiny
// synthetic test model's `head_dim = 16`. For a tape-authoritative backward
// to reach the GQA-block q/k/v projection LoRA `Var`s on that path, the
// fallback must ALSO record onto the kt Tape, exactly as the flash path does.
//
// Mirrors `try_tape_gdn_recurrent_kt` / `GdnRecurrentBackward`: a
// candle-composite `BackwardOp` (`SdpaBackward`) wrapping the analytic
// `forward::sdpa_fallback_backward_no_grad`, recorded on the fallback's
// attention output with `[q, k, v]` as inputs. SDPA is stateless, so there is
// no entry-state to snapshot.
// ===========================================================================

/// True unless `KILN_USE_TAPE_SDPA` is set to a disable value — **DEFAULTS ON**
/// (CP-4 production path). Gate for the naive SDPA-fallback attention tape
/// adapter, separate from `KILN_USE_TAPE_FLASH_ATTN` (which covers the flash
/// path) and the rest of the fleet. Set the env to `0`/`false`/`no`/empty to
/// opt out (for debugging / comparison). Cached after first read, mirroring
/// [`tape_gdn_enabled`].
pub fn tape_sdpa_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_SDPA")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(true)
    })
}

/// Candle-composite tape backward for the naive SDPA fallback
/// (`forward::gqa_attention_core_prefill`'s non-flash path). Wraps
/// [`sdpa_fallback_backward_no_grad`] (analytic adjoint in candle F32),
/// mirroring how [`GdnRecurrentBackward`] wraps `gdn_recurrent_backward_no_grad`.
///
/// # Why a candle-composite wrap (not a kt `BackwardOp` in kiln-autograd)
///
/// The SDPA backward is a composite of broadcast / 4D-batched matmuls, a
/// softmax adjoint, a causal mask, and a GQA head collapse. Those aren't
/// cleanly expressible over `kiln_tensor::ops` (no batched `broadcast_matmul`
/// / softmax-adjoint primitive there), and `kiln-autograd` carries no candle
/// dep — so the analytic backward lives as a candle composite in `kiln-model`
/// and this `BackwardOp` bridges grads through it, exactly like the GDN ops.
///
/// # Saved tensors / inputs
///
/// `q`/`k`/`v` are the **pre-attention, head-FIRST** tensors the fallback
/// consumes (`q = [B, nq, T, hd]`, `k`/`v = [B, nkv, T, hd]`, BEFORE the GQA
/// expand) as candle clones; `scale = 1/sqrt(head_dim)`; `causal` selects the
/// strict-upper-triangular mask. 3 differentiable inputs `[q, k, v]` in the
/// order the adapter records them. The returned `dq` keeps `nq` heads;
/// `dk`/`dv` are GQA-collapsed to `nkv` (matching the `k`/`v` `Var` layouts).
#[derive(Debug)]
pub(crate) struct SdpaBackward {
    // #1082: candle->kt (the bwd kernel sdpa_fallback_backward_no_grad is kt-native).
    q: kiln_tensor::Tensor,
    k: kiln_tensor::Tensor,
    v: kiln_tensor::Tensor,
    scale: f64,
    causal: bool,
}

impl BackwardOp for SdpaBackward {
    fn name(&self) -> &'static str {
        "sdpa_fallback_backward"
    }
    fn input_count(&self) -> usize {
        // q, k, v.
        3
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        // #1082 kt-native: q/k/v are stored kt now (no candle bridge); the bwd
        // kernel is already kt-typed.
        let grads = sdpa_fallback_backward_no_grad(
            &self.q,
            &self.k,
            &self.v,
            self.scale,
            self.causal,
            grad_output,
        )
        .map_err(|e| kiln_tensor::Error::Msg(format!("SdpaBackward: sdpa bwd: {e}")))?;
        // Adjoints (kt) can be non-contiguous (broadcast/transpose views);
        // contiguify.
        let to_kt = |t: &kiln_tensor::Tensor| -> kiln_tensor::Result<kiln_tensor::Tensor> {
            t.contiguous()
                .map_err(|e| kiln_tensor::Error::Msg(format!("SdpaBackward: grad contiguous: {e}")))
        };
        Ok(vec![
            Some(to_kt(&grads.dq)?),
            Some(to_kt(&grads.dk)?),
            Some(to_kt(&grads.dv)?),
        ])
    }
}

/// Record an [`SdpaBackward`] node for the naive SDPA-fallback attention
/// output that the PRODUCTION forward has ALREADY computed.
///
/// The forward (`gqa_attention_core_prefill`'s non-flash path) computes the
/// attention output itself (q@kᵀ scaled, causal-masked softmax, @v) and then
/// reshapes it back to `[B, T, hidden]`; this adapter takes that already-
/// computed `out` (head-FIRST `[B, nq, T, hd]`, BEFORE the reshape-back) and
/// records-only (no re-run), borrowing `out` as the recorded node's output —
/// like `tape_record_gdn_recurrent`. The recorded [`SdpaBackward`] emits
/// GQA-collapsed `dq`/`dk`/`dv` so a tape-authoritative backward reaches the
/// q/k/v projections (and their LoRA `Var`s) on the non-flash path — the
/// attention-block link the flash path covers via `try_tape_flash_attn_cuda`.
///
/// # Arguments
///
/// * `q`/`k`/`v` — the **pre-attention head-FIRST** tensors the fallback
///   consumes: `q = [B, nq, T, hd]`, `k`/`v = [B, nkv, T, hd]` (the
///   `prepared.{q,k,v}.transpose(1,2)` layout, BEFORE the GQA expand). They
///   must carry their LoRA lineage from the upstream q/k/v_proj adapters via
///   `tape_kt_input` chaining.
/// * `head_dim` — `scale = 1/sqrt(head_dim)`, matching the forward's score
///   divisor.
/// * `out` — the attention output the forward produced, head-FIRST
///   `[B, nq, T, hd]` (the `attn_weights_softmax.broadcast_matmul(&v)` result
///   BEFORE its `transpose(1,2).reshape(...)`).
///
/// `Ok(None)` (caller's production output unchanged) when the gate is off, no
/// tape scope is active, the inputs aren't CUDA, shapes disagree, or a kt
/// borrow fails.
/// kt-native naive-SDPA fallback tape recorder (#1082 seam flip) — kt-only twin of
/// `try_tape_sdpa_fallback_cuda`. Record-only: records the (now kt-native)
/// `SdpaBackward` (stores kt q/k/v; bwd kernel already kt) linking kt `out` back to
/// kt q/k/v. The returned kt out lets the downstream transpose+reshape stay
/// kt-native (no kt->candle->kt at the SDPA seam).
pub fn try_tape_sdpa_fallback_kt(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    head_dim: usize,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() || !tape_sdpa_enabled() {
        return Ok(None);
    }
    if !matches!(
        q.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) || !matches!(
        k.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) || !matches!(
        v.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) || !matches!(
        out.device(),
        kiln_tensor::Device::Cuda(_)
            | kiln_tensor::Device::Metal(_)
            | kiln_tensor::Device::Vulkan(_)
            | kiln_tensor::Device::Rocm(_)
    ) {
        return Ok(None);
    }
    let (bq, nq, tq, dq_) = match q.dims4() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let (bk, nkv, tk, dk_) = match k.dims4() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let (bv, nvh, tv, dv_) = match v.dims4() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    if bq != bk
        || bq != bv
        || nkv == 0
        || nq % nkv != 0
        || nvh != nkv
        || tq != tk
        || tq != tv
        || dq_ != head_dim
        || dk_ != head_dim
        || dv_ != head_dim
        || out.dims() != [bq, nq, tq, head_dim].as_slice()
    {
        return Ok(None);
    }
    let scale = 1.0f64 / (head_dim as f64).sqrt();
    let causal = true;
    let recorded = with_active_tape(|tape: &mut Tape| {
        tape.record(
            out,
            &[q, k, v],
            Box::new(SdpaBackward {
                q: q.clone(),
                k: k.clone(),
                v: v.clone(),
                scale,
                causal,
            }),
        );
    });
    if recorded.is_none() {
        return Ok(None);
    }
    Ok(Some(out.clone()))
}

/// True unless `KILN_USE_TAPE_LORA_ADD` is set to a disable value —
/// **DEFAULTS ON** (CP-4 production path).
///
/// Separate gate from `KILN_USE_TAPE_FORWARD` so the LoRA add adapter can
/// be opted out independently of the rest of the tape-forward fleet. The
/// LoRA add now records on the tape (analytic kt backward) by default;
/// set the env to `0`/`false`/`no`/empty to opt out and route through the
/// legacy Marlin-fused CustomOp3 backward (for debugging / comparison).
///
/// Cached after first read, matching `tape_forward_enabled()`.
pub fn tape_lora_add_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_LORA_ADD")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(true)
    })
}

// Tape-scope tests live in `kiln-autograd::tape_scope::tests` after
// wave-13 (#1082) promoted the thread-local-tape machinery there. The
// kt-tape adapter tests (`try_tape_{rms_norm,matmul,silu,embedding,swiglu,
// lora_add}_cuda` round-trips) live in the
// `kiln-model/tests/tape_forward_parity.rs` integration test because they
// require the `kiln_kt_bridge` + `kiln_rmsnorm_kernel` cuda surface.

#[cfg(test)]
mod conv1d_bwd_input_composite_tests {
    use super::causal_depthwise_conv1d_bwd_input_composite;
    use kiln_tensor::{DType, Tensor};

    /// Reference forward of the GDN causal depthwise conv1d w.r.t. the input,
    /// with ZERO conv-state (the state contribution is a separate gradient; the
    /// input-grad kernel assumes the input-only left-zero pad). Matches
    /// `causal_depthwise_conv1d_f32_kernel` with `state = 0`:
    ///   out[r,c] = Σ_{j: r+j >= K-1} weight[c,j] * input[r+j-(K-1), c]
    fn ref_forward(
        input: &[f32],
        weight: &[f32],
        rows: usize,
        channels: usize,
        k: usize,
    ) -> Vec<f32> {
        let state_rows = k - 1;
        let mut out = vec![0.0f32; rows * channels];
        for r in 0..rows {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for j in 0..k {
                    let padded = r + j;
                    if padded >= state_rows {
                        let ir = padded - state_rows;
                        if ir < rows {
                            acc += weight[c * k + j] * input[ir * channels + c];
                        }
                    }
                }
                out[r * channels + c] = acc;
            }
        }
        out
    }

    /// A scalar loss `L = Σ_{r,c} grad_out[r,c] * out[r,c]` so that
    /// `dL/d input[i,c] = composite(grad_out, weight)[i,c]` exactly — that is
    /// the very quantity the input-backward returns. Central-difference each
    /// input element against this loss and compare to the composite output.
    #[test]
    fn fd_input_grad_matches_central_difference() {
        let rows = 5usize;
        let channels = 3usize;
        let k = 4usize;

        // Deterministic pseudo-random-but-fixed inputs.
        let input: Vec<f32> = (0..rows * channels)
            .map(|i| ((i as f32) * 0.37 - 0.9).sin() * 0.8)
            .collect();
        let weight: Vec<f32> = (0..channels * k)
            .map(|i| ((i as f32) * 0.71 + 0.2).cos() * 0.5)
            .collect();
        // Upstream grad (the cotangent of `out`); arbitrary fixed values.
        let grad_out: Vec<f32> = (0..rows * channels)
            .map(|i| ((i as f32) * 0.13 + 0.4).cos() * 0.6 + 0.1)
            .collect();

        // Analytic grad via the composite (CPU, deterministic).
        let go_t = Tensor::from_vec(grad_out.clone(), vec![rows, channels]).unwrap();
        let w_t = Tensor::from_vec(weight.clone(), vec![channels, k]).unwrap();
        let gi_t = causal_depthwise_conv1d_bwd_input_composite(&go_t, &w_t, k).unwrap();
        assert_eq!(gi_t.shape(), [rows, channels]);
        assert_eq!(gi_t.dtype(), DType::F32);
        let analytic: Vec<f32> = gi_t.to_vec::<f32>().unwrap();
        assert_eq!(analytic.len(), rows * channels);
        assert!(
            analytic.iter().all(|v| v.is_finite()),
            "analytic grad non-finite"
        );

        // Central-difference dL/d input[i,c] where L = Σ grad_out·out.
        let loss = |inp: &[f32]| -> f32 {
            let out = ref_forward(inp, &weight, rows, channels, k);
            out.iter().zip(grad_out.iter()).map(|(o, g)| o * g).sum()
        };
        let eps = 1e-3f32;
        let mut max_abs_err = 0.0f32;
        for idx in 0..rows * channels {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let fd = (loss(&plus) - loss(&minus)) / (2.0 * eps);
            let err = (fd - analytic[idx]).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
        }
        eprintln!("conv1d bwd-input composite FD max_abs_err = {max_abs_err:.3e}");
        assert!(
            max_abs_err < 1e-2,
            "conv1d bwd-input composite FD max_abs_err {max_abs_err:.3e} exceeds tol 1e-2"
        );
    }

    /// The composite must accept a non-F32 upstream grad (it casts internally)
    /// and return F32 — the dtype the conv-boundary expects before the op casts
    /// back to the input dtype.
    #[test]
    fn accepts_bf16_grad_out_returns_f32() {
        let rows = 3usize;
        let channels = 2usize;
        let k = 3usize;
        let grad_out: Vec<f32> = (0..rows * channels)
            .map(|i| (i as f32) * 0.1 - 0.2)
            .collect();
        let weight: Vec<f32> = (0..channels * k).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let go_t = Tensor::from_vec(grad_out, vec![rows, channels])
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w_t = Tensor::from_vec(weight, vec![channels, k]).unwrap();
        let gi = causal_depthwise_conv1d_bwd_input_composite(&go_t, &w_t, k).unwrap();
        assert_eq!(gi.dtype(), DType::F32);
        assert_eq!(gi.shape(), [rows, channels]);
        assert!(gi.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    }
}
