//! Production tape-forward adapters.
//!
//! Model forward signatures intentionally remain tensor-only. Training opens a
//! thread-local [`Tape`] with [`with_thread_local_tape`], and these adapters use
//! [`with_active_tape`] to record the corresponding backward nodes without
//! threading `&mut Tape` through the model call graph.
//!
//! # Routing contract
//!
//! [`tape_scope_active`] is the sole routing authority. Outside a scope every
//! adapter returns `Ok(None)` and inference retains its backend fast paths.
//! Inside a scope recording is mandatory whenever the operation is within the
//! backend's authoritative tape envelope; process environment cannot disable
//! individual nodes and silently sever a training graph.
//!
//! The adapters cover kt-native tensor transforms, linear and LoRA operations,
//! embedding, RMSNorm, FlashAttention and SDPA, and the GDN recurrence and its
//! surrounding operations. Backend, dtype, shape, and residency checks remain
//! local to each adapter so unsupported envelopes decline explicitly.

#![cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]

use anyhow::{Context, Result};
use kiln_autograd::{
    AddBackward, BackwardOp, ConcatBackward, ContiguousBackward, FrozenRhsMatmulBackward,
    LoraDeltaAddBackward, MatmulBackward, MulBackward, MulSigmoidGateBackward, ReshapeBackward,
    RopeSplitHalfBackward, SiluBackward, Tape, TransposeBackward,
};

use crate::backend::BackendRuntime;
use crate::forward::{
    GDN_CHUNK_SIZE, gdn_gated_rms_norm_frozen_weight_backward_no_grad,
    gdn_l2_norm_scale_backward_no_grad, gdn_recurrent_backward_no_grad,
    gdn_recurrent_forward_from_parts, sdpa_fallback_backward_no_grad,
};
use crate::lora_loader::LoraProjectionWeights;

// The canonical scope machinery lives in `kiln-autograd` so model, kernel, and
// training crates share one thread-local authority. Re-export it here for the
// model forward call sites and integration tests.
pub use kiln_autograd::{tape_scope_active, with_active_tape, with_thread_local_tape};

/// Whether the active authoritative training tape requested expensive anomaly
/// diagnostics. Forward code uses this to localize the first non-finite layer
/// instead of reporting only the downstream loss symptom. Outside a tape
/// scope this is always false, so inference pays no scan or synchronization.
pub(crate) fn tape_detect_anomaly_active() -> bool {
    with_active_tape(|tape: &mut Tape| tape.options().detect_anomaly).unwrap_or(false)
}

fn tape_forward_device_supported(device: kiln_tensor::Device) -> bool {
    // Inference shares these forward helpers but never installs a tape. Check
    // the thread-local scope before consulting backend capability policy: the
    // Vulkan CPU-sentinel lookup used to construct a fresh logical device for
    // every attempted recorder, only to discover `with_active_tape` was empty.
    if !tape_scope_active() {
        return false;
    }
    matches!(
        crate::backend::training_tape_route_for_device_kt(device),
        crate::backend::TrainingTapeRoute::KtTapeAuthoritative
    )
}

/// kt-native SiLU tape recorder (#1082 seam flip). Takes the kt activation
/// directly and records a `SiluBackward` onto the active tape with
/// **no candle round-trip**. Bottoms out in the same
/// `kiln_tensor::ops::silu` + `SiluBackward` as a direct composite, so
/// forward + backward are bit-identical (guarded by `tape_forward_parity` +
/// the SFT FD test).
/// Chaining is preserved: the returned kt is a recorded tape-node output, so
/// a downstream kt-native op consumes it directly.
///
/// `Ok(None)` when no tape scope is active (decode / inference), and the caller
/// falls through to the kt-native non-tape forward.
pub fn try_tape_silu_kt(x: &kiln_tensor::Tensor) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
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

/// kt-native residual-add tape recorder (#1082 seam flip). Records
/// `AddBackward` directly from kt inputs, no candle round-trip. `Ok(None)` on shape mismatch (defer broadcasting to the caller),
/// tape-off, or no active scope.
pub fn try_tape_add_kt(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
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

/// kt-native contiguous tape recorder.
///
/// `Tensor::contiguous()` is value-identical but layout-changing. Calling it
/// inside a tape-authoritative forward without recording would sever the
/// gradient chain for strided views such as full-attention query tiles. This
/// helper materialises the contiguous tensor and records [`ContiguousBackward`]
/// so backward passes the upstream gradient through to the original view op.
pub fn try_tape_contiguous_kt(x: &kiln_tensor::Tensor) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(x.device()) {
        return Ok(None);
    }
    if x.is_contiguous() {
        return Ok(Some(x.clone()));
    }
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = x
            .contiguous()
            .map_err(|e| anyhow::anyhow!("kt contiguous: {e}"))?;
        tape.record(&y, &[x], Box::new(ContiguousBackward));
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
    if !tape_scope_active() || inputs.is_empty() || axis >= out.rank() {
        return Ok(None);
    }
    if !tape_forward_device_supported(out.device()) {
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
        if !tape_forward_device_supported(input.device())
            || input.dtype() != dtype
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
/// no candle round-trip. `Ok(None)` outside the rank-4 envelope or without an
/// active scope (caller falls through to the kt-native non-tape RoPE).
pub fn try_tape_rope_kt(
    x: &kiln_tensor::Tensor,
    cos: &kiln_tensor::Tensor,
    sin: &kiln_tensor::Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
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
struct FrozenWeightRmsNormKtBackward {
    x: kiln_tensor::Tensor,
    weight: kiln_tensor::Tensor,
    eps: f32,
}

impl BackwardOp for FrozenWeightRmsNormKtBackward {
    fn name(&self) -> &'static str {
        "frozen_weight_rmsnorm_kt_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn requires_input(&self, idx: usize) -> bool {
        idx == 0
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

        Ok(vec![Some(dx)])
    }
}

/// Device-agnostic kt-native RMSNorm tape recorder.
///
/// CUDA and ROCm prefer
/// `kiln_rmsnorm_kernel::fused_rmsnorm_frozen_weight_via_kt_tape` when its
/// exact envelope is supported. This composite is the graph-preserving
/// route for every admitted GPU backend when that fused recorder is unavailable.
///
/// This adapter closes that seam with the device-agnostic
/// `kiln_tensor::ops::rms_norm` forward + the device-agnostic
/// [`FrozenWeightRmsNormKtBackward`] composite (the
/// `kiln_autograd::RmsNormBackward` loop
/// is CPU-only), recorded against the connected kt `x.id()`. The RMSNorm
/// weight is FROZEN in LoRA training, so only the `dL/dx` edge matters; the
/// recorded node keeps the upstream LoRA-affected activations connected to the
/// loss. Returns `Ok(None)` outside a tape scope or outside the admitted
/// BF16/F32 contiguous backend envelope. Production `forward::rms_norm` treats
/// `None` as an error while a scope is active; only inference may continue to a
/// forward-only implementation.
pub fn try_tape_rms_norm_kt(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f32,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(x.device()) || !tape_forward_device_supported(weight.device())
    {
        return Ok(None);
    }
    // Mixed-precision RMSNorm exceptions are backend precision policy, not a
    // local Vulkan special case. Today the policy allows F32 activations with
    // BF16 norm weights for Vulkan's mixed F32/BF16 training envelope.
    let precision_policy = crate::backend::training_precision_policy_for_device_kt(x.device());
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
            &[x],
            Box::new(FrozenWeightRmsNormKtBackward {
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
    if !tape_scope_active() {
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
/// `Ok(None)` (caller falls through) when no tape scope is active or the inputs
/// are outside the envelope: Vulkan-resident, `x` rank-2
/// F32, `weight_t` rank-2 BF16, matching contraction dim `K`.
#[cfg(feature = "vulkan")]
pub fn try_tape_matmul_bf16w_kt(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
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

/// Training-time lookup against the frozen token embedding table.
///
/// Adapter training never differentiates token ids or the resident embedding
/// table. The gathered activation is therefore an intentional tape leaf: its
/// consumers record the downstream graph, while this boundary records no node
/// and cannot allocate a full `[vocab, hidden]` embedding gradient.
pub fn try_tape_frozen_embedding_kt(
    weights: &kiln_tensor::Tensor,
    token_ids: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
        return Ok(None);
    }
    if weights.shape().len() != 2 || token_ids.shape().is_empty() {
        return Ok(None);
    }
    let y = kiln_tensor::ops::embedding(weights, token_ids)
        .map_err(|e| anyhow::anyhow!("kt frozen embedding: {e}"))?;
    Ok(Some(y))
}

/// kt-native SwiGLU (gate ⊙ sigmoid-gate of up) tape recorder (#1082 seam flip) —
/// the kt-native SwiGLU tape recorder.
pub fn try_tape_swiglu_kt(
    gate: &kiln_tensor::Tensor,
    up: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
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
    if !tape_scope_active() {
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

#[derive(Debug)]
struct AttnGateSigmoidMulBackward {
    attn_output: kiln_tensor::Tensor,
    sigmoid_gate: kiln_tensor::Tensor,
    attn_dtype: kiln_tensor::DType,
    gate_dtype: kiln_tensor::DType,
}

impl BackwardOp for AttnGateSigmoidMulBackward {
    fn name(&self) -> &'static str {
        "attn_gate_sigmoid_mul_backward"
    }

    fn input_count(&self) -> usize {
        2
    }

    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        use kiln_tensor::DType;
        use kiln_tensor::ops::{add_scalar, cast, mul, mul_scalar};

        if self.attn_output.dims() != self.sigmoid_gate.dims()
            || self.attn_output.dims() != grad_output.dims()
        {
            return Err(kiln_tensor::Error::Msg(format!(
                "AttnGateSigmoidMulBackward: shape mismatch attn={:?} sigmoid={:?} grad={:?}",
                self.attn_output.dims(),
                self.sigmoid_gate.dims(),
                grad_output.dims()
            )));
        }

        let dy = cast(grad_output, DType::F32)?;
        let attn = cast(&self.attn_output, DType::F32)?;
        let sigmoid = cast(&self.sigmoid_gate, DType::F32)?;
        let d_attn = mul(&dy, &sigmoid)?;
        let one_minus_sigmoid = add_scalar(&mul_scalar(&sigmoid, -1.0)?, 1.0)?;
        let sigmoid_grad = mul(&sigmoid, &one_minus_sigmoid)?;
        let d_gate = mul(&mul(&dy, &attn)?, &sigmoid_grad)?;

        Ok(vec![
            Some(cast(&d_attn, self.attn_dtype)?),
            Some(cast(&d_gate, self.gate_dtype)?),
        ])
    }

    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

/// kt-native attention output gate recorder.
///
/// Qwen3.5 full-attention uses `attn_output * sigmoid(gate)`. This is not the
/// SwiGLU `silu(gate) * up` derivative, so it needs its own tape op. Backward
/// is device-composed in F32 and casts gradients back to the source dtypes.
pub fn try_tape_attn_gate_sigmoid_mul_kt(
    attn_output: &kiln_tensor::Tensor,
    gate: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(attn_output.device())
        || !tape_forward_device_supported(gate.device())
        || attn_output.dims() != gate.dims()
    {
        return Ok(None);
    }
    match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let sigmoid_gate =
            kiln_tensor::ops::sigmoid(gate).map_err(|e| anyhow::anyhow!("kt sigmoid: {e}"))?;
        let y = kiln_tensor::ops::mul(attn_output, &sigmoid_gate)
            .map_err(|e| anyhow::anyhow!("kt attention gate mul: {e}"))?;
        tape.record(
            &y,
            &[attn_output, gate],
            Box::new(AttnGateSigmoidMulBackward {
                attn_output: attn_output.clone(),
                sigmoid_gate,
                attn_dtype: attn_output.dtype(),
                gate_dtype: gate.dtype(),
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
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(x.device()) {
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
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(x.device()) {
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
/// # Why kt-native (vs the old candle-copy `CrossEntropyFromLogitsBackward`)
///
/// The earlier `CrossEntropyFromLogitsBackward` saved the FULL `[1, T, V]`
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

/// kt-NATIVE fused next-token cross-entropy loss node (replaces the
/// candle-typed `try_tape_cross_entropy_from_logits_cuda` sibling, deleted in
/// #1082): roots the WHOLE next-token cross-entropy loss at a SINGLE fused kt
/// `Tape` node taking the FULL `[1, T, V]` kt logits DIRECTLY — with NO
/// `[1, T, V]` kt -> candle copy.
///
/// # Why this exists (#1082 H6)
///
/// The candle-typed sibling `try_tape_cross_entropy_from_logits_cuda` took
/// candle logits, so in the tape-authoritative SFT path `cross_entropy_loss`
/// first bridged the kt lm_head logits into a full `[1, T, V]` candle tensor
/// (≈150 MB+/step) just so the candle adapter could re-borrow them via
/// `tape_kt_input` — immediately resolving them back to kt. That copy was
/// pure waste. This entry takes the kt logits straight from the trainer (the
/// kt lm_head output), runs the CE forward in kt, and records a kt-native
/// [`CrossEntropyFromLogitsKtBackward`] node against them — the kt scalar
/// loss is returned directly (no candle bridge).
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
/// loss      = mean_a( lse[a] - correct[a] )         (scalar; the mean-all reduction, as `cross_entropy_loss`)
/// ```
///
/// where `active_positions = { i in 0..T-1 : label_mask[i+1] }`, `A = num_active`,
/// `y_a = input_ids[active_positions[a] + 1]`. This is numerically identical to
/// `cross_entropy_loss`'s baseline (same shift / active-set convention as
/// kiln-train's `token_log_probs`).
///
/// # Returns
///
/// * `Ok(Some(loss))` — the tape-forward path ran: a DETACHED, lineage-free
///   kt scalar loss (numerically identical to the `cross_entropy_loss`
///   baseline) with a [`CrossEntropyFromLogitsKtBackward`] node recorded on
///   the active tape as the backward root. The loss is detached
///   unconditionally so the tape-authoritative caller's `loss.backward()` is
///   always `{loss: ones}` and the recorded node is the sole backward root.
/// * `Ok(None)` — the gate is off, `logits` isn't a CUDA rank-3 `[1, T, V]`, no
///   tape scope is active, or an empty active set. The caller treats this as
///   a hard decline and errors.
/// * `Err(...)` — an unexpected forward failure.
pub fn try_tape_cross_entropy_from_logits_kt(
    logits: &kiln_tensor::Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
) -> Result<Option<kiln_tensor::Tensor>> {
    use kiln_tensor::{DType as KtDType, Tensor as KtTensor};

    if !tape_scope_active() {
        return Ok(None);
    }

    // Full model logits only: [1, T, V] on CUDA. Defer any other shape/device
    // to the caller.
    let dims = logits.dims().to_vec();
    if dims.len() != 3
        || dims[0] != 1
        || dims[1] != input_ids.len()
        || label_mask.len() != input_ids.len()
        || !tape_forward_device_supported(logits.device())
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
    // kiln-train's `token_log_probs` builds from `shift_mask = mask[1..]`.
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

    // kt input — thread the lm_head adapter's output so the tape stays
    // connected (consumer input id == producer output id). The trainer passes
    // the lm_head kt output directly, so the recorded node roots on the live
    // kt logits.
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
    // as kiln-train's `token_log_probs` does.
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

    // Record the fused node: the OUTPUT is the OWNED kt scalar loss (independent
    // kt storage, so it does not dangle). The single differentiable input is
    // the CONNECTED kt logits, so the recorded
    // `CrossEntropyFromLogitsKtBackward` node roots `dL/d(logits)` directly at
    // the lm_head kt output — no id-mapping dance.
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
    // `with_tape_authoritative_scope_kt` seeds it directly (the SFT authoritative
    // path takes the kt loss straight).
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
/// * `Ok(Some(out))` — the tape-forward path ran. The returned kt `Tensor`
///   is the kt-typed output (reshaped back to `base.shape()`); a
///   `LoraDeltaAddBackward { x, a, b, scale }` node was recorded on the
///   active thread-local tape with inputs `[base, x, A, B]` in that order.
/// * `Ok(None)` — no thread-local tape is active, or device / dtype / shape /
///   contiguity preconditions fail. The caller must fall through to the
///   existing CUDA / Metal / Vulkan dispatch.
/// * `Err(...)` — an unexpected kt forward or reshape failure. Propagated so
///   callers see the failure cleanly instead of silently masking it.
///
/// # CP-4 (#1082) context — closes LoRA Var grad coverage
///
/// Without this adapter, the production LoRA delta-add dispatch in
/// `add_lora_delta_to_base` lands in `runtime_lora_decode_add`,
/// `runtime_lora_delta_resident`, or the kt composite fallbacks — none of
/// which the kt `Tape` walker sees. Under tape-authoritative training, the
/// resulting `GradStore` has no entries for the LoRA `Var`s (`proj.a`,
/// `proj.b`), so the optimiser step is a no-op for the adapter parameters.
/// With this adapter on, the fused backward emits grads for `proj.a` and
/// `proj.b` in their original `[rank, in_features]` / `[out_features, rank]`
/// shapes, and the IO mapping pairs each kt input id with the Var's kt id
/// so the parity gate sees nonzero matched LoRA grads.
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
/// kt-native LoRA delta-add tape recorder (#1082 seam flip). base + x already
/// kt at the call site, so the composite (base->2d + x->2d reshapes, LoRA
/// delta=(x2d@Aᵀ)@Bᵀ·scale, out=base2d+delta, reshape back) runs in kt ops
/// recording the SAME ReshapeBackward + LoraDeltaAddBackward
/// nodes (base/x reshapes recorded so their grads chain in kt). LoRA-Var grad
/// mapping preserved. (Reached only when the fused linear+LoRA path declines
/// — a rare fallback.)
pub fn try_tape_lora_add_kt(
    base: &kiln_tensor::Tensor,
    x: &kiln_tensor::Tensor,
    proj: &LoraProjectionWeights,
    lora_scale: f32,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(base.device())
        || !tape_forward_device_supported(x.device())
        || !tape_forward_device_supported(proj.a.device())
        || !tape_forward_device_supported(proj.b.device())
    {
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

/// LoRA delta adjoint for an output-column slice of a larger projection.
///
/// Forward and the inner adjoint operate on `B[start..start+len, :]`, but the
/// tape input is the original full B parameter. The slice gradient is embedded
/// into a zero-filled full-shape tensor before the tape accumulates it, so
/// multiple split projection chunks reconstruct exactly one gradient under the
/// original parameter id.
#[derive(Debug)]
struct LoraDeltaAddOutputSliceBackward {
    inner: LoraDeltaAddBackward,
    output_start: usize,
    full_output_features: usize,
}

impl BackwardOp for LoraDeltaAddOutputSliceBackward {
    fn name(&self) -> &'static str {
        "lora_delta_add_output_slice_backward"
    }

    fn input_count(&self) -> usize {
        4
    }

    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        let mut grads = self.inner.apply(grad_output)?;
        let grad_b_slice = grads.get_mut(3).and_then(Option::take).ok_or_else(|| {
            kiln_tensor::Error::from_str(
                "LoraDeltaAddOutputSliceBackward: inner B gradient missing",
            )
        })?;
        if grad_b_slice.rank() != 2 {
            return Err(kiln_tensor::Error::Msg(format!(
                "LoraDeltaAddOutputSliceBackward: B gradient must be rank-2, got {:?}",
                grad_b_slice.shape()
            )));
        }
        let slice_len = grad_b_slice.shape()[0];
        let rank = grad_b_slice.shape()[1];
        let output_end = self.output_start.checked_add(slice_len).ok_or_else(|| {
            kiln_tensor::Error::from_str(
                "LoraDeltaAddOutputSliceBackward: output slice end overflow",
            )
        })?;
        if output_end > self.full_output_features {
            return Err(kiln_tensor::Error::Msg(format!(
                "LoraDeltaAddOutputSliceBackward: slice [{}, {}) exceeds full B rows {}",
                self.output_start, output_end, self.full_output_features
            )));
        }

        let grad_device = grad_b_slice.device();
        let grad_dtype = grad_b_slice.dtype();
        let mut pieces = Vec::with_capacity(3);
        if self.output_start > 0 {
            pieces.push(kiln_tensor::Tensor::zeros_on(
                grad_device,
                vec![self.output_start, rank],
                grad_dtype,
            )?);
        }
        pieces.push(grad_b_slice);
        if output_end < self.full_output_features {
            pieces.push(kiln_tensor::Tensor::zeros_on(
                grad_device,
                vec![self.full_output_features - output_end, rank],
                grad_dtype,
            )?);
        }
        let grad_b_full = if pieces.len() == 1 {
            pieces.pop().expect("one gradient piece")
        } else {
            kiln_tensor::Tensor::cat(&pieces, 0)?
        };
        grads[3] = Some(grad_b_full);
        Ok(grads)
    }

    fn requires_input(&self, idx: usize) -> bool {
        self.inner.requires_input(idx)
    }
}

/// Run the full base projection and optional fused LoRA delta through the
/// kt-typed op surface as one chained group of `Tape` nodes:
/// `reshape → matmul → [lora_delta_add] → reshape`.
///
/// The forward computes:
/// ```text
/// out = base + scale * (x @ A^T @ B^T) // when `lora` is Some
/// out = base = x @ W^T                 // when `lora` is None
/// ```
///
/// Keeping the base and LoRA paths in one group preserves the shared `x2d` and
/// `base2d` tensor ids. Consequently, `dL/dx2d` accumulates the frozen-base and
/// LoRA contributions, while only the trainable LoRA A/B parameters are
/// registered for exact bridge deposit. The base weight is saved by
/// `FrozenRhsMatmulBackward`; it is not a tape input and receives no gradient.
///
/// The group records these nodes inside one `with_active_tape` call:
///
/// 1. `ReshapeBackward`: `x → x2d [rows, k]`.
/// 2. `FrozenRhsMatmulBackward`: `x2d → base2d [rows, n]`.
/// 3. Optional `LoraDeltaAddBackward`: `base2d → out2d` with shared `x2d`
///    and the original LoRA A/B ids.
/// 4. `ReshapeBackward`: `out2d → out` with the input prefix plus `[n]`.
///
/// Returns `Ok(None)` when there is no active tape or the device, dtype, shape,
/// or contiguity contract is unsupported. Unexpected kt execution or recording
/// failures are returned as errors.
#[allow(clippy::too_many_lines)]
pub fn try_tape_lora_linear_kt(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Option<kiln_tensor::Tensor>> {
    try_tape_lora_linear_impl_kt(x, weight_t, lora, lora_scale, None)
}

/// Tape-routed linear+LoRA for one output slice of a larger projection.
///
/// `weight_t` is already the `[in, slice_out]` frozen base slice. When LoRA is
/// present, the forward narrows the original full B parameter internally while
/// the tape records that original id and zero-embeds this slice's `dB` back into
/// the full parameter shape. This is the authoritative route for split q/gate
/// training projections.
pub fn try_tape_lora_linear_output_slice_kt(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
    output_start: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    try_tape_lora_linear_impl_kt(x, weight_t, lora, lora_scale, Some(output_start))
}

#[allow(clippy::too_many_lines)]
fn try_tape_lora_linear_impl_kt(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
    output_start: Option<usize>,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(x.device())
        || !tape_forward_device_supported(weight_t.device())
    {
        return Ok(None);
    }
    if !matches!(
        x.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    // (#1443 step 2) Mixed-precision base projection: F32 activations × a
    // frozen BF16 base weight are a backend precision-policy exception, not a
    // local device-family check. Today the policy admits this for Vulkan's
    // mixed F32/BF16 envelope; other backends keep the strict equal-dtype gate.
    let precision_policy = crate::backend::training_precision_policy_for_device_kt(x.device());
    let mixed_base_weight = cfg!(feature = "vulkan")
        && precision_policy
            .supports_mixed_base_weight_dtype_for_activation(x.dtype(), weight_t.dtype());
    if weight_t.dtype() != x.dtype() && !mixed_base_weight {
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
        let b_envelope_ok = match output_start {
            Some(start) => start
                .checked_add(n)
                .is_some_and(|output_end| output_end <= b_out),
            None => b_out == n,
        };
        if a_in != k || !b_envelope_ok || b_rank != rank {
            return Ok(None);
        }
        if !proj.a.is_contiguous() || !proj.b.is_contiguous() {
            return Ok(None);
        }
    }
    let (a_kt, b_input_kt, b_forward_kt) = match lora {
        Some(proj) => {
            let b_forward = match output_start {
                Some(start) => proj
                    .b
                    .narrow(0, start, n)
                    .context("split LoRA B narrow")?
                    .contiguous()
                    .context("split LoRA B contiguous")?,
                None => proj.b.clone(),
            };
            (Some(proj.a.clone()), Some(proj.b.clone()), Some(b_forward))
        }
        None => (None, None, None),
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
        // (#1443 step 2) Base projection. On the policy-approved mixed-base path
        // the frozen base weight is BF16 while `x2d` is F32 — the equal-dtype kt
        // `matmul` can't run, so route the base linear through the dedicated
        // Vulkan `vk_matmul_bf16w` leaf and record a `MatmulBf16wBackward` (dx
        // only; the weight is frozen, no `dW` edge). Output `base2d` is F32, so
        // the LoRA delta / residual adds below stay F32-vs-F32. Every equal-dtype
        // path uses the device-agnostic input-only adjoint. Base weights are
        // immutable during adapter training and must never become tape leaves.
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
        let base2d = if mixed_base_weight {
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
            unreachable!("mixed_base_weight is false without the vulkan feature")
        } else {
            let base2d = kiln_tensor::ops::matmul(&x2d, weight_t)
                .map_err(|e| anyhow::anyhow!("kt matmul x2d@w: {e}"))?;
            tape.record(
                &base2d,
                &[&x2d],
                Box::new(FrozenRhsMatmulBackward {
                    b: weight_t.clone(),
                }),
            );
            base2d
        };
        let out2d = match (
            lora,
            a_kt.as_ref(),
            b_input_kt.as_ref(),
            b_forward_kt.as_ref(),
        ) {
            (Some(_proj), Some(a_kt), Some(b_input_kt), Some(b_forward_kt)) => {
                let h_kt = kiln_tensor::ops::matmul_rhs_transposed(&x2d, a_kt)
                    .map_err(|e| anyhow::anyhow!("kt matmul_rhs_transposed x@a_t: {e}"))?;
                let d_kt = kiln_tensor::ops::matmul_rhs_transposed(&h_kt, b_forward_kt)
                    .map_err(|e| anyhow::anyhow!("kt matmul_rhs_transposed h@b_t: {e}"))?;
                let delta_kt = kiln_tensor::ops::mul_scalar(&d_kt, lora_scale)
                    .map_err(|e| anyhow::anyhow!("kt mul_scalar(scale): {e}"))?;
                let out2d = kiln_tensor::ops::add(&base2d, &delta_kt)
                    .map_err(|e| anyhow::anyhow!("kt add(base, delta): {e}"))?;
                let backward = LoraDeltaAddBackward {
                    x: maybe_offload_matmul_a_saved_tensor(&x2d)
                        .context("try_tape_lora_linear_kt: save lora input")?,
                    a: a_kt.clone(),
                    b: b_forward_kt.clone(),
                    scale: lora_scale,
                };
                let backward: Box<dyn BackwardOp> = match output_start {
                    Some(output_start) => Box::new(LoraDeltaAddOutputSliceBackward {
                        inner: backward,
                        output_start,
                        full_output_features: b_input_kt.shape()[0],
                    }),
                    None => Box::new(backward),
                };
                tape.record(&out2d, &[&base2d, &x2d, a_kt, b_input_kt], backward);
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
    if let (Some(proj), Some(a_kt), Some(b_input_kt)) = (lora, a_kt.as_ref(), b_input_kt.as_ref()) {
        kiln_kt_bridge::tape_bridge::register_input_mapping_kt(a_kt.id(), proj.a.id());
        kiln_kt_bridge::tape_bridge::register_input_mapping_kt(b_input_kt.id(), proj.b.id());
    }
    Ok(Some(out_kt))
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
/// `flash_attn_bwd_collapsed_gqa_kt_with_mode(dout, q, k, v, out, lse, scale, causal, mode)`
/// returns `(dq, dk, dv)` where `dk`/`dv` are already collapsed to the grouped
/// K/V head count. The collapse runs in F32 (cast/sum/cast, or the backend's
/// equivalent) so the group reduction doesn't lose precision in BF16.
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
        use kiln_tensor::DType;

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

        let backward_mode = match crate::cuda_training_policy::current_cuda_training_policy()
            .flash_attention_backward_mode
        {
            crate::cuda_training_policy::FlashAttentionBackwardMode::Fast => {
                kiln_flash_attn::FlashAttnBackwardMode::Fast
            }
            crate::cuda_training_policy::FlashAttentionBackwardMode::Deterministic => {
                kiln_flash_attn::FlashAttnBackwardMode::Deterministic
            }
        };
        let (dq, dk, dv) = kiln_flash_attn::flash_attn_bwd_collapsed_gqa_kt_with_mode(
            &dout,
            &self.q,
            &self.k,
            &self.v,
            &self.out,
            &self.softmax_lse,
            self.scale,
            self.causal,
            backward_mode,
        )
        .map_err(|e| {
            kiln_tensor::Error::Msg(format!(
                "FlashAttnBackward: flash_attn_bwd_collapsed_gqa_kt: {e:?}"
            ))
        })?;
        Ok(vec![Some(dq), Some(dk), Some(dv)])
    }
}

/// Attempt to route the FlashAttention-2 forward through the kt `Tape`
/// (the candle `CudaFlashAttentionTrainingBf16` CustomOp3 it replaced was
/// deleted in #1082).
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
/// `Ok(None)` (caller falls through to the portable SDPA-fallback path,
/// which has its own tape recorder) when: the gate is off, no tape scope is
/// active, or the inputs leave the BF16/CUDA/contiguous/
/// `head_dim∈{128,256}`/valid-GQA envelope.
/// kt-native FlashAttention-2 tape recorder (#1082 seam flip). q/k/v are
/// already kt (recorded upstream outputs), so no candle bridging: runs
/// `flash_attn_fwd_kt` and records the kt-native `FlashAttnBackward` (all
/// kt-tensor fields) directly. The returned kt out lets the downstream
/// reshape stay kt-native too (no kt->candle->kt at the attention seam).
pub fn try_tape_flash_attn_kt(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
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
        Ok(None)
    }
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        let fused_device_supported = match q.device() {
            #[cfg(feature = "cuda")]
            kiln_tensor::Device::Cuda(_) => true,
            #[cfg(feature = "rocm")]
            kiln_tensor::Device::Rocm(_) => true,
            _ => false,
        };
        if q.dtype() != kiln_tensor::DType::BF16
            || k.dtype() != kiln_tensor::DType::BF16
            || v.dtype() != kiln_tensor::DType::BF16
            || !fused_device_supported
            || k.device() != q.device()
            || v.device() != q.device()
            || !q.is_contiguous()
            || !k.is_contiguous()
            || !v.is_contiguous()
            || !matches!(head_dim, 128 | 256)
            || num_kv_heads == 0
            || !num_heads.is_multiple_of(num_kv_heads)
        {
            return Ok(None);
        }
        // `sq` is consumed only by the rocm-lane long-sequence policy check below;
        // allow the unused-variable warning on cuda/cpu lanes.
        #[allow(unused_variables)]
        let Ok((bq, sq, hq, dq_)) = q.dims4() else {
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
        #[cfg(feature = "rocm")]
        if matches!(q.device(), kiln_tensor::Device::Rocm(_))
            && (sq.max(sk) > 4096)
            && !crate::rocm_policy::current_rocm_kernel_policy().long_flash_attn
        {
            return Ok(None);
        }
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        let causal = true;
        match with_active_tape(|tape: &mut Tape| -> Result<_> {
            let run_flash = || {
                kiln_flash_attn::flash_attn_fwd_kt(q, k, v, softmax_scale, causal)
                    .map_err(|e| anyhow::anyhow!("kt flash_attn_fwd_kt: {e:?}"))
            };
            #[cfg(feature = "rocm")]
            let (out_kt, lse_kt) = run_flash()?;
            #[cfg(not(feature = "rocm"))]
            let (out_kt, lse_kt) = run_flash()?;
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
                }),
            );
            Ok(out_kt)
        }) {
            Some(result) => Ok(Some(result?)),
            None => Ok(None),
        }
    }
}

/// Offload saved GDN recurrence tensors from the tape node to CPU memory.
///
/// The recorded input ids remain the original device tensors, so gradient
/// chaining is unchanged. Only the values captured by `GdnRecurrentBackward`
/// are moved off the accelerator and uploaded back one node at a time during
/// backward. This trades bandwidth for much lower long-context tape residency.
fn tape_gdn_saved_tensor_offload_enabled(device: &kiln_tensor::Device) -> bool {
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
    1 << 20
}

fn saved_tensor_bytes(t: &kiln_tensor::Tensor) -> usize {
    t.dtype().size_in_bytes().saturating_mul(t.element_count())
}

fn should_offload_saved_tensor(t: &kiln_tensor::Tensor) -> bool {
    !matches!(t.device(), kiln_tensor::Device::Cpu)
        && saved_tensor_bytes(t) >= tape_saved_tensor_offload_min_bytes()
}

fn tape_matmul_a_offload_enabled(device: &kiln_tensor::Device) -> bool {
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

/// Analytic adjoint for `beta = sigmoid(b)` in the GDN gate transform.
///
/// Only `b` is a differentiable tape input. The gate parameters are frozen
/// model weights, so recording them would widen the training gradient
/// contract beyond the configured LoRA leaves.
#[derive(Debug)]
pub(crate) struct GdnBetaGateBackward {
    b: kiln_tensor::Tensor,
    source_dtype: kiln_tensor::DType,
}

impl BackwardOp for GdnBetaGateBackward {
    fn name(&self) -> &'static str {
        "gdn_beta_gate_backward"
    }

    fn input_count(&self) -> usize {
        1
    }

    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        if grad_output.dims() != self.b.dims() {
            return Err(kiln_tensor::Error::Msg(format!(
                "GdnBetaGateBackward: shape mismatch b={:?} grad={:?}",
                self.b.dims(),
                grad_output.dims()
            )));
        }

        let device = grad_output.device();
        let b = saved_tensor_for_device(&self.b, device)?
            .contiguous()?
            .to_dtype(kiln_tensor::DType::F32)?;
        let dy = grad_output
            .contiguous()?
            .to_dtype(kiln_tensor::DType::F32)?;
        let sigmoid = kiln_tensor::ops::sigmoid(&b)?;
        let one_minus_sigmoid =
            kiln_tensor::ops::add_scalar(&kiln_tensor::ops::mul_scalar(&sigmoid, -1.0)?, 1.0)?;
        let derivative = kiln_tensor::ops::mul(&sigmoid, &one_minus_sigmoid)?;
        let db = kiln_tensor::ops::mul(&dy, &derivative)?
            .to_dtype(self.source_dtype)?
            .contiguous()?;
        Ok(vec![Some(db)])
    }
}

/// Analytic adjoint for
/// `g = -exp(a_log) * softplus(a + dt_bias)` in the GDN gate transform.
///
/// `a_log` and `dt_bias` are saved constants. Only `a` is registered as a
/// differentiable input, matching the LoRA-only training contract.
#[derive(Debug)]
pub(crate) struct GdnDecayGateBackward {
    a: kiln_tensor::Tensor,
    a_log: kiln_tensor::Tensor,
    dt_bias: kiln_tensor::Tensor,
    source_dtype: kiln_tensor::DType,
}

impl BackwardOp for GdnDecayGateBackward {
    fn name(&self) -> &'static str {
        "gdn_decay_gate_backward"
    }

    fn input_count(&self) -> usize {
        1
    }

    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        if grad_output.dims() != self.a.dims() {
            return Err(kiln_tensor::Error::Msg(format!(
                "GdnDecayGateBackward: shape mismatch a={:?} grad={:?}",
                self.a.dims(),
                grad_output.dims()
            )));
        }

        let device = grad_output.device();
        let a = saved_tensor_for_device(&self.a, device)?
            .contiguous()?
            .to_dtype(kiln_tensor::DType::F32)?;
        let a_log = saved_tensor_for_device(&self.a_log, device)?
            .contiguous()?
            .to_dtype(kiln_tensor::DType::F32)?;
        let dt_bias = saved_tensor_for_device(&self.dt_bias, device)?
            .contiguous()?
            .to_dtype(kiln_tensor::DType::F32)?;
        let dy = grad_output
            .contiguous()?
            .to_dtype(kiln_tensor::DType::F32)?;

        // d softplus(x) / dx = sigmoid(x).
        let biased_a = a.broadcast_add(&dt_bias)?;
        let softplus_derivative = kiln_tensor::ops::sigmoid(&biased_a)?;
        let neg_decay = kiln_tensor::ops::mul_scalar(&kiln_tensor::ops::exp(&a_log)?, -1.0)?;
        let derivative = softplus_derivative.broadcast_mul(&neg_decay)?;
        let da = kiln_tensor::ops::mul(&dy, &derivative)?
            .to_dtype(self.source_dtype)?
            .contiguous()?;
        Ok(vec![Some(da)])
    }
}

/// Record the two GDN gate outputs that the production fallback already
/// computed.
///
/// Outside an authoritative tape scope this returns `Ok(None)` and inference
/// keeps its existing fused/fallback routing. Inside a scope the complete
/// shape, dtype, device, and backend-admission contract is mandatory: a bad
/// route errors rather than returning an unrecorded forward tensor.
pub fn try_tape_gdn_gates_kt(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
    if !tape_scope_active() {
        return Ok(None);
    }

    let device = a.device();
    anyhow::ensure!(
        tape_forward_device_supported(device),
        "active tape scope does not admit GDN gates on device {device}"
    );
    for (name, tensor) in [
        ("b", b),
        ("a_log", a_log),
        ("dt_bias", dt_bias),
        ("beta", beta),
        ("g", g),
    ] {
        anyhow::ensure!(
            tensor.device() == device,
            "GDN gate {name} device mismatch: expected {device}, got {}",
            tensor.device()
        );
        anyhow::ensure!(
            matches!(
                tensor.dtype(),
                kiln_tensor::DType::F32 | kiln_tensor::DType::BF16 | kiln_tensor::DType::F16
            ),
            "GDN gate {name} must be F32, BF16, or F16, got {}",
            tensor.dtype()
        );
    }
    anyhow::ensure!(
        matches!(
            a.dtype(),
            kiln_tensor::DType::F32 | kiln_tensor::DType::BF16 | kiln_tensor::DType::F16
        ),
        "GDN gate a must be F32, BF16, or F16, got {}",
        a.dtype()
    );
    anyhow::ensure!(
        a.rank() > 0 && a.dims() == b.dims(),
        "GDN gate input shape mismatch: a={:?} b={:?}",
        a.dims(),
        b.dims()
    );
    anyhow::ensure!(
        a.dims() == g.dims() && b.dims() == beta.dims(),
        "GDN gate output shape mismatch: a={:?} b={:?} beta={:?} g={:?}",
        a.dims(),
        b.dims(),
        beta.dims(),
        g.dims()
    );
    anyhow::ensure!(
        a.dtype() == b.dtype() && g.dtype() == a.dtype() && beta.dtype() == b.dtype(),
        "GDN gate activation dtype mismatch: a={} b={} beta={} g={}",
        a.dtype(),
        b.dtype(),
        beta.dtype(),
        g.dtype()
    );
    let channels = *a.dims().last().expect("rank checked above");
    anyhow::ensure!(
        a_log.dims() == [channels] && dt_bias.dims() == [channels],
        "GDN gate parameter shape mismatch: channels={channels} a_log={:?} dt_bias={:?}",
        a_log.dims(),
        dt_bias.dims()
    );

    // Prepare every fallible saved-tensor operation before mutating the tape,
    // so the two-output registration is atomic from the caller's perspective.
    let offload_saved = tape_gdn_saved_tensor_offload_enabled(&device);
    let saved_a = maybe_offload_gdn_saved_tensor(a, offload_saved)
        .context("try_tape_gdn_gates_kt: save a")?;
    let saved_b = maybe_offload_gdn_saved_tensor(b, offload_saved)
        .context("try_tape_gdn_gates_kt: save b")?;
    let saved_a_log = maybe_offload_gdn_saved_tensor(a_log, offload_saved)
        .context("try_tape_gdn_gates_kt: save a_log")?;
    let saved_dt_bias = maybe_offload_gdn_saved_tensor(dt_bias, offload_saved)
        .context("try_tape_gdn_gates_kt: save dt_bias")?;
    let recorded = with_active_tape(|tape: &mut Tape| {
        tape.record(
            beta,
            &[b],
            Box::new(GdnBetaGateBackward {
                b: saved_b,
                source_dtype: b.dtype(),
            }),
        );
        tape.record(
            g,
            &[a],
            Box::new(GdnDecayGateBackward {
                a: saved_a,
                a_log: saved_a_log,
                dt_bias: saved_dt_bias,
                source_dtype: a.dtype(),
            }),
        );
    });
    anyhow::ensure!(
        recorded.is_some(),
        "active tape scope disappeared while recording GDN gates"
    );
    Ok(Some((beta.clone(), g.clone())))
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
    if !tape_scope_active() {
        return Ok(None);
    }
    // #1082 P4-full: q/k/v/beta/g/recurrent_state are kt (this calls the kt
    // production recurrence `gdn_recurrent_forward_from_parts`) and the record
    // adapter (`tape_record_gdn_recurrent_kt`) is now kt-native — no kt->candle
    // bridge on the saved inputs.
    if !tape_forward_device_supported(q.device()) {
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
/// per GDN layer per step, ×24 GDN layers). The candle shim
/// `tape_record_gdn_recurrent` that wrapped this was deleted in #1082.
///
/// # Arguments
///
/// * `out_kt` — the PRODUCTION recurrence-output kt tensor. Its `.id()` must be
///   the SAME id that flows downstream, so the tape stays connected across
///   the recurrence→transpose seam. Layout per `head_last`.
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
    if !tape_scope_active() {
        return Ok(false);
    }
    if !tape_forward_device_supported(*device) {
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
// Each adapter is active exactly when a tape scope is open, and falls through
// cleanly outside training so production inference remains unchanged.
// ===========================================================================

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
/// `Ok(None)` leaves the caller's production output unchanged when no tape
/// scope is active, the inputs are on an unsupported device, `batch != 1`,
/// shapes disagree, or a kt borrow fails.
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
    if !tape_scope_active() {
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
        if !tape_forward_device_supported(input.device())
            || !tape_forward_device_supported(out.device())
            || !tape_forward_device_supported(weight.device())
        {
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
    if !tape_scope_active() {
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
        if !tape_forward_device_supported(input.device())
            || !tape_forward_device_supported(out.device())
            || !tape_forward_device_supported(weight.device())
            || !tape_forward_device_supported(entry_state.device())
        {
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

/// kt-composite tape backward for the GDN L2-qk-norm `y =
/// l2_normalize(x) * scale`. Wraps [`gdn_l2_norm_scale_backward_no_grad`]
/// (analytic adjoint in F32), mirroring how [`GdnRecurrentBackward`]
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
/// (`scale = 1.0`). The forward runs kt `l2_normalize`-then-scale via the
/// existing production helpers; the recorded [`GdnL2NormScaleBackward`] emits
/// the per-input grad so a tape-authoritative backward reaches the conv / split
/// (and thence the in_proj LoRA Vars).
///
/// `Ok(None)` (caller falls through to the existing `gdn_qk_norm`) when the
/// gate is off, no tape scope is active, or the device isn't tape-supported.
/// kt-native L2-norm-scale tape recorder (#1082 seam flip). Record-only:
/// records the (now kt-native)
/// `GdnL2NormScaleBackward` (stores kt `x`; bwd kernel already kt) linking kt `out`
/// back to kt `x`. No candle round-trip.
pub fn try_tape_gdn_l2_norm_scale_kt(
    x: &kiln_tensor::Tensor,
    scale: f64,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(x.device()) {
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

/// kt-composite tape backward for the GDN gated RMSNorm `out =
/// rms_norm(x, weight) * silu(z)`. Wraps
/// [`gdn_gated_rms_norm_frozen_weight_backward_no_grad`] (analytic adjoint in
/// F32), mirroring how [`GdnRecurrentBackward`] wraps
/// `gdn_recurrent_backward_no_grad`.
///
/// Two differentiable inputs `[x, z]` in that order. `x` is the recurrence
/// output (head-LAST `[B, T, nv, dv]`) and `z` is the output gate. The GDN norm
/// weight is frozen adapter-training state and is saved only for `dx`/`dz`.
#[derive(Debug)]
pub(crate) struct GdnGatedRmsNormBackward {
    // #1082: candle->kt (the frozen-weight analytic backward is kt-native).
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
) -> kiln_tensor::Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
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
        kiln_tensor::DType::BF16 => kiln_gdn_kernel::gdn_gated_rms_norm_bwd_bf16_frozen_weight_kt(
            &grad_flat, &x_flat, &z_flat, weight, eps as f32,
        )
        .map_err(|e| {
            kiln_tensor::Error::Msg(format!(
                "GdnGatedRmsNormBackward fused frozen-weight bwd: {e}"
            ))
        })?,
        kiln_tensor::DType::F32 => {
            kiln_gdn_kernel::gdn_gated_rms_norm_bwd_bf16_f32_weight_frozen_kt(
                &grad_flat, &x_flat, &z_flat, weight, eps as f32,
            )
            .map_err(|e| {
                kiln_tensor::Error::Msg(format!(
                    "GdnGatedRmsNormBackward fused frozen f32-weight bwd: {e}"
                ))
            })?
        }
        _ => return Ok(None),
    };
    let dx = grads.dx.reshape(x_dims.clone()).map_err(|e| {
        kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward fused dx reshape: {e}"))
    })?;
    let dz = grads.dz.reshape(x_dims).map_err(|e| {
        kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward fused dz reshape: {e}"))
    })?;
    Ok(Some((dx, dz)))
}

impl BackwardOp for GdnGatedRmsNormBackward {
    fn name(&self) -> &'static str {
        "gdn_gated_rms_norm_backward"
    }
    fn input_count(&self) -> usize {
        // x and z; weight is frozen saved data.
        2
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
        if let Some((dx, dz)) =
            try_gdn_gated_rms_norm_backward_fused_cuda_rocm(&x, &z, &weight, self.eps, grad_output)?
        {
            return Ok(vec![Some(dx), Some(dz)]);
        }

        // #1082 kt-native: x/z/weight are stored kt now (no candle bridge); the bwd
        // kernel is already kt-typed.
        let grads = gdn_gated_rms_norm_frozen_weight_backward_no_grad(
            &x,
            &z,
            &weight,
            self.eps,
            grad_output,
        )
        .map_err(|e| kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward: bwd: {e}")))?;
        // Adjoints (kt) can be non-contiguous; contiguify.
        let to_kt = |t: &kiln_tensor::Tensor| -> kiln_tensor::Result<kiln_tensor::Tensor> {
            t.contiguous().map_err(|e| {
                kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward: grad contiguous: {e}"))
            })
        };
        Ok(vec![Some(to_kt(&grads.dx)?), Some(to_kt(&grads.dz)?)])
    }
}

/// Route the GDN gated RMSNorm `out = rms_norm(x, weight) * silu(z)` through
/// the kt `Tape`.
///
/// The production forward (`gated_rms_norm`) already computed `out`; this
/// records-only (no re-run), borrowing `out` as the node output — like
/// `tape_record_gdn_recurrent_kt`. The recorded [`GdnGatedRmsNormBackward`]
/// emits `dx`/`dz` so a tape-authoritative backward reaches the recurrence
/// output (`x`) and gate (`z`). The norm weight is immutable during adapter
/// training.
///
/// `Ok(None)` (caller falls through) when the gate is off, no tape scope is
/// active, the inputs aren't CUDA, or shapes disagree.
/// kt-native gated-RMSNorm tape recorder (#1082 seam flip). Record-only:
/// records the (now kt-native)
/// `GdnGatedRmsNormBackward` (stores kt x/z/weight; bwd kernel already kt) linking
/// kt `out` back to kt x/z. No candle round-trip.
pub fn try_tape_gdn_gated_rms_norm_kt(
    x: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f64,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(x.device()) {
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
            &[x, z],
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
// chain back to the in_proj_qkv/z LoRA `Var`s. Every one needs its own
// recorded tape node at the seam, so an unwrapped op fragments the tape and
// the GDN LoRA grads never flow.
//
// All three use a kt-composite backward (analytic adjoint on the kt
// substrate), NOT a kt-native `kiln_autograd` `BackwardOp`, because:
//   * `NarrowBackward` in kiln-autograd downcasts the grad storage to
//     `CpuStorage`; a CUDA grad would bail. The kt composite is
//     unambiguously CUDA-safe and value-faithful (cast is a dtype round-trip;
//     narrow's adjoint is a zero-pad; the GQA expand's adjoint is a
//     reshape+sum).
//   * Mirrors the proven `GdnRecurrentBackward` / `GdnL2NormScaleBackward` /
//     `SdpaBackward` kt-composite pattern already in this module.
//
// Gated on an active tape scope only (pure
// layout/dtype ops with trivial, always-safe adjoints — same contract as
// `try_tape_reshape_kt` / `try_tape_transpose_kt`). `Ok(None)` on any
// inactive-scope / non-tape-supported-device / envelope-miss.
// ===========================================================================

/// kt-composite tape backward for a float dtype cast (`to_dtype`). The
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
/// output — like `tape_record_gdn_recurrent_kt`. The recorded
/// [`CastCompositeBackward`] casts the upstream grad back to `x`'s dtype so a
/// tape-authoritative backward stays connected across the cast.
///
/// Used on the GDN path for `v.to_dtype(input_dtype)` (BF16→/F32→BF16 before
/// recurrence) and `attn_out.to_dtype(input_dtype)` (after gated-RMSNorm,
/// before out_proj). `Ok(None)` when the gate is off, no tape scope is
/// active, the inputs aren't CUDA, or shapes disagree.
/// kt-native cast tape recorder (#1082 seam flip). Record-only: the forward
/// cast is already done by the kt
/// non-tape path; this records the (now kt-native) `CastCompositeBackward`
/// linking the kt `out` back to the kt `x`. No candle round-trip.
pub fn try_tape_cast_kt(
    x: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(x.device()) || !tape_forward_device_supported(out.device()) {
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

/// kt-composite tape backward for `out = narrow(x, axis, offset, length)`.
/// The adjoint embeds the upstream grad into a zero-filled tensor of the
/// original `x.shape` at `[offset .. offset+length]` along `axis` (the
/// standard "zero-pad" narrow adjoint). One differentiable input (`x`).
///
/// CUDA-safe (unlike `kiln_autograd::NarrowBackward`, which downcasts the
/// grad to `CpuStorage` and bails on a CUDA grad): runs the zero-pad
/// kt-native as `Tensor::zeros` + `Tensor::cat`.
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
                let lz = kiln_tensor::Tensor::zeros(left_sh, self.source_dtype, dev)?;
                let rz = kiln_tensor::Tensor::zeros(right_sh, self.source_dtype, dev)?;
                kiln_tensor::Tensor::cat(&[lz, grad, rz], self.axis)?
            }
            (true, false) => {
                let lz = kiln_tensor::Tensor::zeros(left_sh, self.source_dtype, dev)?;
                kiln_tensor::Tensor::cat(&[lz, grad], self.axis)?
            }
            (false, true) => {
                let rz = kiln_tensor::Tensor::zeros(right_sh, self.source_dtype, dev)?;
                kiln_tensor::Tensor::cat(&[grad, rz], self.axis)?
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
/// `Ok(None)` on any inactive-scope / non-CUDA / shape-envelope-miss.
/// kt-native narrow tape recorder (#1082 seam flip). Record-only: the forward
/// narrow is already done by the
/// kt non-tape path; records the (now kt-native) `NarrowCompositeBackward`
/// linking the kt `out` back to the kt `x`. No candle round-trip.
pub fn try_tape_narrow_kt(
    x: &kiln_tensor::Tensor,
    axis: usize,
    offset: usize,
    length: usize,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(x.device()) || !tape_forward_device_supported(out.device()) {
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

/// kt-composite tape backward for the GDN GQA head-expand
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
/// `Ok(None)` on any inactive-scope / non-CUDA / shape-envelope-miss.
/// kt-native GQA head-expand tape recorder (#1082 seam flip). Record-only: the
/// forward expand is already done
/// by the kt non-tape path; this records `GqaExpandBackward` (all-usize fields)
/// linking the kt `out` back to the kt `x`. No candle round-trip.
pub fn try_tape_gqa_expand_kt(
    x: &kiln_tensor::Tensor,
    gqa_ratio: usize,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(x.device()) || !tape_forward_device_supported(out.device()) {
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
// kt-composite `BackwardOp` (`SdpaBackward`) wrapping the analytic
// `forward::sdpa_fallback_backward_no_grad`, recorded on the fallback's
// attention output with `[q, k, v]` as inputs. SDPA is stateless, so there is
// no entry-state to snapshot.
// ===========================================================================

/// kt-composite tape backward for the naive SDPA fallback
/// (`forward::gqa_attention_core_prefill`'s non-flash path). Wraps
/// [`sdpa_fallback_backward_no_grad`] (analytic adjoint in F32),
/// mirroring how [`GdnRecurrentBackward`] wraps `gdn_recurrent_backward_no_grad`.
///
/// # Why a composite wrap (not a kt `BackwardOp` in kiln-autograd)
///
/// The SDPA backward is a composite of broadcast / 4D-batched matmuls, a
/// softmax adjoint, a causal mask, and a GQA head collapse. Those aren't
/// cleanly expressible over `kiln_tensor::ops` (no batched `broadcast_matmul`
/// / softmax-adjoint primitive there) — so the analytic backward lives as a
/// kt composite in `kiln-model` and this `BackwardOp` runs it on the grad,
/// exactly like the GDN ops.
///
/// # Saved tensors / inputs
///
/// `q`/`k`/`v` are the **pre-attention, head-FIRST** tensors the fallback
/// consumes (`q = [B, nq, Tq, hd]`, `k`/`v = [B, nkv, Tk, hd]`, BEFORE the GQA
/// expand) as kt clones; `scale = 1/sqrt(head_dim)`; `causal` selects the
/// lower-triangular mask with prefix offset `Tk - Tq`. 3 differentiable inputs
/// `[q, k, v]` in the order the adapter records them. The returned `dq` keeps
/// `nq` heads; `dk`/`dv` are GQA-collapsed to `nkv` (matching the `k`/`v`
/// `Var` layouts).
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
/// like `tape_record_gdn_recurrent_kt`. The recorded [`SdpaBackward`] emits
/// GQA-collapsed `dq`/`dk`/`dv` so a tape-authoritative backward reaches the
/// q/k/v projections (and their LoRA `Var`s) on the non-flash path — the
/// attention-block link the flash path covers via `try_tape_flash_attn_kt`.
///
/// # Arguments
///
/// * `q`/`k`/`v` — the **pre-attention head-FIRST** tensors the fallback
///   consumes: `q = [B, nq, Tq, hd]`, `k`/`v = [B, nkv, Tk, hd]` (the
///   `prepared.{q,k,v}.transpose(1,2)` layout, BEFORE the GQA expand). They
///   must carry their LoRA lineage from the upstream q/k/v_proj adapters via
///   kt tape chaining.
/// * `head_dim` — `scale = 1/sqrt(head_dim)`, matching the forward's score
///   divisor.
/// * `out` — the attention output the forward produced, head-FIRST
///   `[B, nq, Tq, hd]` (the `attn_weights_softmax.broadcast_matmul(&v)` result
///   BEFORE its `transpose(1,2).reshape(...)`).
///
/// `Ok(None)` (caller's production output unchanged) when the gate is off, no
/// tape scope is active, the inputs aren't CUDA, or shapes disagree.
/// kt-native naive-SDPA fallback tape recorder (#1082 seam flip). Record-only:
/// records the (now kt-native)
/// `SdpaBackward` (stores kt q/k/v; bwd kernel already kt) linking kt `out`
/// back to kt q/k/v. The returned kt out lets the downstream transpose+reshape
/// stay kt-native (no kt->candle->kt at the SDPA seam).
pub fn try_tape_sdpa_fallback_kt(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    head_dim: usize,
    out: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_scope_active() {
        return Ok(None);
    }
    if !tape_forward_device_supported(q.device())
        || !tape_forward_device_supported(k.device())
        || !tape_forward_device_supported(v.device())
        || !tape_forward_device_supported(out.device())
    {
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
        || tq > tk
        || tk != tv
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

// Tape-scope tests live in `kiln-autograd::tape_scope::tests` after
// wave-13 (#1082) promoted the thread-local-tape machinery there. The
// kt-tape adapter tests (`try_tape_{rms_norm,matmul,silu,embedding,swiglu,
// lora_add}_cuda` round-trips) live in the
// `kiln-model/tests/tape_forward_parity.rs` integration test because they
// require the `kiln_kt_bridge` + `kiln_rmsnorm_kernel` cuda surface.

#[cfg(test)]
mod lora_output_slice_backward_tests {
    use std::collections::HashMap;

    use super::LoraDeltaAddOutputSliceBackward;
    use kiln_autograd::{BackwardOp, LoraDeltaAddBackward, Tape};
    use kiln_tensor::Tensor;

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_slice(data, shape.to_vec()).unwrap()
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) {
        assert_eq!(actual.shape(), expected.shape());
        for (index, (a, e)) in values(actual).into_iter().zip(values(expected)).enumerate() {
            assert!((a - e).abs() < 1e-5, "entry {index}: {a} != {e}");
        }
    }

    #[test]
    fn output_slice_backward_preserves_inner_grads_and_zero_pads_b() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let a = t(&[0.5, -0.25], &[1, 2]);
        let b = t(&[2.0, -3.0], &[2, 1]);
        let dy = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let inner = LoraDeltaAddBackward {
            x,
            a,
            b,
            scale: 0.5,
        };
        let expected = inner.apply(&dy).unwrap();
        let actual = LoraDeltaAddOutputSliceBackward {
            inner,
            output_start: 1,
            full_output_features: 5,
        }
        .apply(&dy)
        .unwrap();

        for index in 0..3 {
            assert_close(
                actual[index].as_ref().unwrap(),
                expected[index].as_ref().unwrap(),
            );
        }
        assert_eq!(
            values(actual[3].as_ref().unwrap()),
            vec![0.0, 0.75, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn unequal_output_slices_accumulate_under_original_b_id() {
        let x = t(&[1.0, 2.0, -1.0, 0.5], &[2, 2]);
        let a = t(&[0.25, -0.75], &[1, 2]);
        let full_b = t(&[0.5, -1.0, 1.5, 0.25, -0.5], &[5, 1]);
        let b0 = full_b.narrow(0, 0, 2).unwrap().contiguous().unwrap();
        let b1 = full_b.narrow(0, 2, 3).unwrap().contiguous().unwrap();
        let dy0 = t(&[0.2, -0.4, 0.7, 0.1], &[2, 2]);
        let dy1 = t(&[0.3, -0.2, 0.8, -0.5, 0.4, 0.6], &[2, 3]);
        let base0 = Tensor::zeros_cpu(vec![2, 2], kiln_tensor::DType::F32);
        let base1 = Tensor::zeros_cpu(vec![2, 3], kiln_tensor::DType::F32);
        let out0 = Tensor::zeros_cpu(vec![2, 2], kiln_tensor::DType::F32);
        let out1 = Tensor::zeros_cpu(vec![2, 3], kiln_tensor::DType::F32);

        let mut tape = Tape::new();
        tape.record(
            &out0,
            &[&base0, &x, &a, &full_b],
            Box::new(LoraDeltaAddOutputSliceBackward {
                inner: LoraDeltaAddBackward {
                    x: x.clone(),
                    a: a.clone(),
                    b: b0.clone(),
                    scale: 0.5,
                },
                output_start: 0,
                full_output_features: 5,
            }),
        );
        tape.record(
            &out1,
            &[&base1, &x, &a, &full_b],
            Box::new(LoraDeltaAddOutputSliceBackward {
                inner: LoraDeltaAddBackward {
                    x: x.clone(),
                    a: a.clone(),
                    b: b1.clone(),
                    scale: 0.5,
                },
                output_start: 2,
                full_output_features: 5,
            }),
        );
        assert!(
            tape.nodes()
                .iter()
                .all(|node| node.input_ids[3] == full_b.id())
        );
        assert!(
            tape.nodes().iter().all(
                |node| !node.input_ids.contains(&b0.id()) && !node.input_ids.contains(&b1.id())
            )
        );

        let mut seeds = HashMap::new();
        seeds.insert(out0.id(), dy0.clone());
        seeds.insert(out1.id(), dy1.clone());
        let grads = tape
            .backward_with_seeds(seeds, kiln_tensor::ops::add)
            .unwrap();

        let dy_full = Tensor::cat(&[&dy0, &dy1], 1).unwrap();
        let reference = LoraDeltaAddBackward {
            x: x.clone(),
            a: a.clone(),
            b: full_b.clone(),
            scale: 0.5,
        }
        .apply(&dy_full)
        .unwrap();
        assert_close(grads.get(x.id()).unwrap(), reference[1].as_ref().unwrap());
        assert_close(grads.get(a.id()).unwrap(), reference[2].as_ref().unwrap());
        assert_close(
            grads.get(full_b.id()).unwrap(),
            reference[3].as_ref().unwrap(),
        );
    }

    #[test]
    fn output_slice_backward_rejects_out_of_range_target() {
        let one = t(&[1.0], &[1, 1]);
        let op = LoraDeltaAddOutputSliceBackward {
            inner: LoraDeltaAddBackward {
                x: one.clone(),
                a: one.clone(),
                b: one.clone(),
                scale: 1.0,
            },
            output_start: 1,
            full_output_features: 1,
        };
        let error = op.apply(&one).unwrap_err().to_string();
        assert!(error.contains("exceeds full B rows"), "{error}");
    }
}

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

#[cfg(test)]
mod gdn_gate_backward_tests {
    use super::{GdnBetaGateBackward, GdnDecayGateBackward};
    use kiln_autograd::BackwardOp;
    use kiln_tensor::{DType, Tensor};

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn softplus(x: f32) -> f32 {
        x.max(0.0) + (-x.abs()).exp().ln_1p()
    }

    #[test]
    fn beta_gate_backward_matches_analytic_and_finite_difference() {
        let values = vec![-3.0_f32, -0.5, 0.0, 1.5, 4.0, -1.25];
        let upstream = vec![0.7_f32, -1.2, 0.25, 1.1, -0.4, 0.9];
        let b = Tensor::from_vec(values.clone(), vec![2, 3]).unwrap();
        let dy = Tensor::from_vec(upstream.clone(), vec![2, 3]).unwrap();
        let op = GdnBetaGateBackward {
            b,
            source_dtype: DType::F32,
        };
        let actual = op.apply(&dy).unwrap().remove(0).unwrap();
        let actual = actual.to_vec::<f32>().unwrap();

        let loss = |input: &[f32]| -> f32 {
            input
                .iter()
                .zip(&upstream)
                .map(|(&x, &grad)| grad * sigmoid(x))
                .sum()
        };
        let eps = 1e-3_f32;
        for idx in 0..values.len() {
            let sig = sigmoid(values[idx]);
            let analytic = upstream[idx] * sig * (1.0 - sig);
            assert!(
                (actual[idx] - analytic).abs() < 2e-6,
                "beta analytic mismatch at {idx}: actual={} expected={analytic}",
                actual[idx]
            );

            let mut plus = values.clone();
            let mut minus = values.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let finite_difference = (loss(&plus) - loss(&minus)) / (2.0 * eps);
            assert!(
                (actual[idx] - finite_difference).abs() < 3e-4,
                "beta finite-difference mismatch at {idx}: actual={} fd={finite_difference}",
                actual[idx]
            );
        }
    }

    #[test]
    fn decay_gate_backward_matches_analytic_and_finite_difference() {
        let rows = 2usize;
        let channels = 3usize;
        let values = vec![-2.0_f32, -0.25, 1.5, 0.75, 3.0, -1.0];
        let a_log_values = vec![-0.8_f32, 0.2, 1.1];
        let dt_bias_values = vec![0.3_f32, -0.4, 0.15];
        let upstream = vec![0.5_f32, -1.1, 0.3, 1.4, -0.2, 0.8];
        let a = Tensor::from_vec(values.clone(), vec![rows, channels]).unwrap();
        let a_log = Tensor::from_vec(a_log_values.clone(), vec![channels]).unwrap();
        let dt_bias = Tensor::from_vec(dt_bias_values.clone(), vec![channels]).unwrap();
        let dy = Tensor::from_vec(upstream.clone(), vec![rows, channels]).unwrap();
        let op = GdnDecayGateBackward {
            a,
            a_log,
            dt_bias,
            source_dtype: DType::F32,
        };
        let actual = op.apply(&dy).unwrap().remove(0).unwrap();
        let actual = actual.to_vec::<f32>().unwrap();

        let loss = |input: &[f32]| -> f32 {
            input
                .iter()
                .enumerate()
                .map(|(idx, &x)| {
                    let channel = idx % channels;
                    upstream[idx]
                        * -a_log_values[channel].exp()
                        * softplus(x + dt_bias_values[channel])
                })
                .sum()
        };
        let eps = 1e-3_f32;
        for idx in 0..values.len() {
            let channel = idx % channels;
            let analytic = upstream[idx]
                * -a_log_values[channel].exp()
                * sigmoid(values[idx] + dt_bias_values[channel]);
            assert!(
                (actual[idx] - analytic).abs() < 3e-6,
                "decay analytic mismatch at {idx}: actual={} expected={analytic}",
                actual[idx]
            );

            let mut plus = values.clone();
            let mut minus = values.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let finite_difference = (loss(&plus) - loss(&minus)) / (2.0 * eps);
            assert!(
                (actual[idx] - finite_difference).abs() < 8e-4,
                "decay finite-difference mismatch at {idx}: actual={} fd={finite_difference}",
                actual[idx]
            );
        }
    }

    #[test]
    fn gate_backwards_reject_wrong_upstream_shape() {
        let b = Tensor::from_vec(vec![0.0_f32; 6], vec![2, 3]).unwrap();
        let wrong = Tensor::from_vec(vec![1.0_f32; 3], vec![1, 3]).unwrap();
        let op = GdnBetaGateBackward {
            b,
            source_dtype: DType::F32,
        };
        let error = op.apply(&wrong).unwrap_err().to_string();
        assert!(
            error.contains("shape mismatch"),
            "unexpected error: {error}"
        );
    }
}
