//! `kiln_tensor::Tensor`-typed surface for the rmsnorm-kernel CUDA entry
//! points.
//!
//! Phase 7 prep — same pattern as kiln-flash-attn + kiln-conv1d-kernel:
//! kt-typed wrappers bottom out in the same CUDA FFI symbols while the
//! candle-typed surface remains as the fallback/reference during migration.

use kiln_kt_bridge::BridgeError;
use kiln_tensor::{DType as KtDType, Device as KtDevice, Tensor as KtTensor};

use crate::{
    kiln_adamw_step_bf16, kiln_adamw_step_f32, kiln_attn_decode_qkv_split_qk_norm_rope_bf16,
    kiln_causal_depthwise_conv1d_bwd_input_f32, kiln_causal_depthwise_conv1d_bwd_state_f32,
    kiln_causal_depthwise_conv1d_bwd_weight_f32, kiln_causal_depthwise_conv1d_f32,
    kiln_causal_depthwise_conv1d_inplace_f32, kiln_f32_to_bf16, kiln_fused_l2_qk_norm,
    kiln_fused_l2_qk_norm_gqa, kiln_fused_mlp_silu_mul_bf16, kiln_fused_mlp_silu_mul_packed_bf16,
    kiln_fused_rmsnorm, kiln_fused_rmsnorm_bwd, kiln_fused_rotary_one, kiln_fused_rotary_one_bwd,
    kiln_fused_rotary_qk, kiln_fused_sigmoid_mul_bf16, kiln_lora_add_inplace_f32,
    kiln_lora_decode_add_bf16, kiln_lora_decode_hidden_bf16, kiln_muon_step_bf16,
    kiln_muon_step_f32, kiln_sgd_step_bf16, kiln_sgd_step_f32, kiln_silu_inplace_save_sigmoid_f32,
};

#[derive(Debug)]
pub enum RmsNormError {
    Msg(String),
}

impl std::fmt::Display for RmsNormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RmsNormError::Msg(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for RmsNormError {}

impl From<BridgeError> for RmsNormError {
    fn from(e: BridgeError) -> Self {
        RmsNormError::Msg(e.message)
    }
}

// cuda/no-features lanes only: all 5 call sites are inside the `#[cfg(feature = "rocm")]`
// row-tiled wrapper, so this helper is dead everywhere except the rocm lane.
#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
fn kt_error(context: impl Into<String>, err: kiln_tensor::Error) -> RmsNormError {
    RmsNormError::Msg(format!("{}: {err}", context.into()))
}

#[cfg(feature = "rocm")]
fn rocm_fused_rmsnorm_row_tile_rows(x: &KtTensor) -> Result<usize, RmsNormError> {
    let (storage, _) =
        kiln_kt_bridge::rocm_storage_and_byte_offset(x, x.dtype(), "rmsnorm_row_tile_policy")?;
    Ok(storage
        .context()
        .execution_policy()
        .tensor_kernels
        .rmsnorm_row_tile_rows)
}

// Backend-neutral seam (Phase R.7). These bottom out in
// `kiln_kt_bridge::device_*`, which dispatch on the tensor's backend and work
// for BOTH `Device::Cuda` and `Device::Rocm`. The CUDA path is unchanged (the
// neutral dispatchers route `Device::Cuda` tensors to the same cuda helpers).

/// Allocate a fresh zeroed output tensor of `dtype`/`shape` on the SAME device
/// as `source` (replaces `alloc_cuda_tensor` + the
/// storage-then-`cuda_zeros_ctx` pattern).
fn alloc_like(
    source: &KtTensor,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, RmsNormError> {
    Ok(kiln_kt_bridge::alloc_device_tensor_like(
        source, dtype, shape,
    )?)
}

/// Allocate an RMSNorm backward output on the source tensor's actual storage
/// context. This matters for borrowed/external ROCm tensors: allocating by
/// device index would silently move the output to the cached primary context.
fn alloc_rmsnorm_backward_like(
    source: &KtTensor,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, RmsNormError> {
    #[cfg(feature = "rocm")]
    if matches!(source.device(), KtDevice::Rocm(_)) {
        let (storage, _) = kiln_kt_bridge::rocm_storage_and_byte_offset(
            source,
            source.dtype(),
            "rmsnorm_backward_source",
        )?;
        return Ok(kiln_kt_bridge::alloc_rocm_tensor(storage, dtype, shape)?);
    }

    alloc_like(source, dtype, shape)
}

/// Typed GPU stream submission for `t`'s next external FFI call.
fn device_stream_submission(
    t: &KtTensor,
    name: &'static str,
) -> Result<kiln_kt_bridge::DeviceStreamSubmission, RmsNormError> {
    Ok(kiln_kt_bridge::device_stream_submission_of(t, name)?)
}

#[cfg(feature = "rocm")]
fn rocm_owner_stream_identity(
    tensor: &KtTensor,
    name: &'static str,
) -> Result<kiln_tensor::RocmStreamId, RmsNormError> {
    let (storage, _) = kiln_kt_bridge::rocm_storage_and_byte_offset(tensor, tensor.dtype(), name)?;
    Ok(storage.rocm_owner_stream_id())
}

#[cfg(feature = "rocm")]
fn rocm_active_stream_identity(
    tensor: &KtTensor,
    name: &'static str,
) -> Result<kiln_tensor::RocmStreamId, RmsNormError> {
    let (storage, _) = kiln_kt_bridge::rocm_storage_and_byte_offset(tensor, tensor.dtype(), name)?;
    Ok(storage.rocm_stream_id())
}

#[cfg(feature = "rocm")]
fn synchronize_rocm_rmsnorm_backward_inputs(
    launch_stream: kiln_tensor::RocmStreamId,
    output_owner_stream: kiln_tensor::RocmStreamId,
    inputs: &[(&'static str, &KtTensor)],
) -> Result<(), RmsNormError> {
    let capture_active = kiln_tensor::rocm_capture_arena_active();
    for (name, tensor) in inputs {
        let input_owner_stream = rocm_owner_stream_identity(tensor, name)?;
        if capture_active {
            if input_owner_stream != output_owner_stream {
                return Err(RmsNormError::Msg(format!(
                    "kt-rmsnorm bwd: ROCm input {name} belongs to a different storage context during graph capture"
                )));
            }
            continue;
        }
        let input_stream = rocm_active_stream_identity(tensor, name)?;
        if input_stream == launch_stream {
            continue;
        }
        kiln_tensor::rocm_synchronize_tensor_stream(tensor).map_err(|e| {
            RmsNormError::Msg(format!(
                "kt-rmsnorm bwd: synchronize ROCm input {name} before launch: {e}"
            ))
        })?;
    }
    Ok(())
}

/// `fused_rmsnorm` over `kiln_tensor::Tensor` operands.
///
/// `x`: BF16 `[..., hidden]`. `weight`: BF16 `[hidden]`. Returns BF16
/// same shape as `x`. Matches `kiln-model::forward::rms_norm`
/// (Qwen3.5-style, weight centred on 0).
pub fn fused_rmsnorm_kt(
    x: &KtTensor,
    weight: &KtTensor,
    eps: f32,
) -> Result<KtTensor, RmsNormError> {
    let x_shape = x.shape().to_vec();
    let hidden = *x_shape
        .last()
        .ok_or_else(|| RmsNormError::Msg("kt-rmsnorm: x must have rank >= 1".to_string()))?;
    let weight_shape = weight.shape();
    if weight_shape != [hidden] {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm: weight {weight_shape:?} != [{hidden}]"
        )));
    }
    if hidden > 8192 {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm: hidden dim {hidden} > 8192 envelope"
        )));
    }
    let rows: usize = x_shape[..x_shape.len() - 1].iter().product();

    #[cfg(feature = "rocm")]
    if matches!(x.device(), KtDevice::Rocm(_)) {
        let row_tile = rocm_fused_rmsnorm_row_tile_rows(x)?;
        if rows > row_tile {
            return fused_rmsnorm_kt_rocm_row_tiled(
                x, weight, eps, &x_shape, rows, hidden, row_tile,
            );
        }
    }

    // Owner-agnostic input pointers — accepts both Owned and
    // Borrowed kt storage (Phase 7 v2).
    let x_ptr = kiln_kt_bridge::device_input_ptr(x, KtDType::BF16, "x")?;
    let w_ptr = kiln_kt_bridge::device_input_ptr(weight, KtDType::BF16, "weight")?;
    let x_st = x;
    let out = alloc_like(x_st, KtDType::BF16, x_shape.clone())?;
    if rows == 0 {
        return Ok(out);
    }
    let o_ptr = kiln_kt_bridge::device_output_ptr(&out);

    #[cfg(feature = "rocm")]
    if matches!(x.device(), KtDevice::Rocm(_)) {
        kiln_tensor::rocm_synchronize_tensor_stream(x).map_err(|e| {
            RmsNormError::Msg(format!("kt-rmsnorm: synchronize ROCm x before launch: {e}"))
        })?;
        kiln_tensor::rocm_synchronize_tensor_stream(weight).map_err(|e| {
            RmsNormError::Msg(format!(
                "kt-rmsnorm: synchronize ROCm weight before launch: {e}"
            ))
        })?;
    }

    // Run the kernel on the OUTPUT tensor's stream, not the input's. On ROCm
    // each fresh allocation (`alloc_like` -> `rocm_zeros_ctx`) creates a NEW
    // RocmContext with its OWN default stream, and the output's zeroing memset
    // is enqueued (async, unsynchronized) on THAT stream. The readback
    // (`rocm_to_host_copy`) also syncs the output's stream. If the kernel ran on
    // the input's (different) stream, the output-zeroing memset could land AFTER
    // the kernel's writes with no cross-stream ordering, nondeterministically
    // zeroing valid results ("got 0"). Launching on the output's stream
    // serializes memset -> kernel -> readback on one stream. ROCm inputs can be
    // freshly materialized on another stream (not just synchronizing H2D uploads),
    // so synchronize their owning streams before this output-stream launch. CUDA
    // is unaffected: `device_stream_submission` resolves to the shared default
    // stream there.
    let stream_submission = device_stream_submission(&out, "out")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_fused_rmsnorm(
            x_ptr as *const _,
            w_ptr as *const _,
            o_ptr as *mut _,
            rows as i32,
            hidden as i32,
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(out)
}

#[cfg(feature = "rocm")]
fn fused_rmsnorm_kt_rocm_row_tiled(
    x: &KtTensor,
    weight: &KtTensor,
    eps: f32,
    x_shape: &[usize],
    rows: usize,
    hidden: usize,
    row_tile: usize,
) -> Result<KtTensor, RmsNormError> {
    let x2d = x
        .reshape(vec![rows, hidden])
        .map_err(|e| kt_error("kt-rmsnorm rocm row-tiled reshape input", e))?;
    let mut pieces = Vec::with_capacity(rows.div_ceil(row_tile));
    let mut row_start = 0usize;
    while row_start < rows {
        let row_len = (rows - row_start).min(row_tile);
        let x_tile = x2d
            .narrow(0, row_start, row_len)
            .map_err(|e| {
                kt_error(
                    format!(
                        "kt-rmsnorm rocm row-tiled narrow [{row_start}, {})",
                        row_start + row_len
                    ),
                    e,
                )
            })?
            .contiguous()
            .map_err(|e| {
                kt_error(
                    format!(
                        "kt-rmsnorm rocm row-tiled contiguous [{row_start}, {})",
                        row_start + row_len
                    ),
                    e,
                )
            })?;
        let out_tile = fused_rmsnorm_kt(&x_tile, weight, eps).map_err(|e| {
            RmsNormError::Msg(format!(
                "kt-rmsnorm rocm row-tiled forward [{row_start}, {}): {e}",
                row_start + row_len
            ))
        })?;
        if matches!(out_tile.device(), KtDevice::Rocm(_)) {
            kiln_tensor::rocm_synchronize_tensor_stream(&out_tile).map_err(|e| {
                RmsNormError::Msg(format!(
                    "kt-rmsnorm rocm row-tiled synchronize [{row_start}, {}): {e}",
                    row_start + row_len
                ))
            })?;
        }
        pieces.push(out_tile);
        row_start += row_len;
    }
    let piece_refs: Vec<&KtTensor> = pieces.iter().collect();
    let out2d = KtTensor::cat(&piece_refs, 0)
        .map_err(|e| kt_error("kt-rmsnorm rocm row-tiled concat", e))?;
    out2d
        .reshape(x_shape.to_vec())
        .map_err(|e| kt_error("kt-rmsnorm rocm row-tiled reshape output", e))
}

/// `fused_rmsnorm_backward` over `kiln_tensor::Tensor` operands.
///
/// Returns `(grad_x, grad_weight_f32)`. The kernel performs the cross-row
/// reduction directly with F32 atomics, so `grad_weight_f32` is the final
/// `[hidden]` weight gradient rather than a per-row scratch tensor.
///
/// Shapes:
/// - `x`, `weight`, `grad_out`: BF16, matching the forward
/// - `grad_x`: BF16, shape == x
/// - `grad_weight_f32`: F32 `[hidden]`, fully reduced across rows
pub fn fused_rmsnorm_backward_kt(
    x: &KtTensor,
    weight: &KtTensor,
    grad_out: &KtTensor,
    eps: f32,
) -> Result<(KtTensor, KtTensor), RmsNormError> {
    let (grad_x, grad_weight) = fused_rmsnorm_backward_impl(x, weight, grad_out, eps, true)?;
    let grad_weight = grad_weight.ok_or_else(|| {
        RmsNormError::Msg("kt-rmsnorm bwd: internal weight-gradient omission".to_string())
    })?;
    Ok((grad_x, grad_weight))
}

/// Fused RMSNorm backward for a frozen normalization weight.
///
/// Computes and returns only `grad_x`. The GPU launch selects a specialized
/// kernel instantiation that omits the weight-gradient atomic accumulation,
/// and this path does not allocate a weight-gradient tensor.
pub fn fused_rmsnorm_backward_dx_kt(
    x: &KtTensor,
    weight: &KtTensor,
    grad_out: &KtTensor,
    eps: f32,
) -> Result<KtTensor, RmsNormError> {
    let (grad_x, grad_weight) = fused_rmsnorm_backward_impl(x, weight, grad_out, eps, false)?;
    debug_assert!(grad_weight.is_none());
    Ok(grad_x)
}

fn fused_rmsnorm_backward_impl(
    x: &KtTensor,
    weight: &KtTensor,
    grad_out: &KtTensor,
    eps: f32,
    compute_grad_weight: bool,
) -> Result<(KtTensor, Option<KtTensor>), RmsNormError> {
    let x_shape = x.shape().to_vec();
    let hidden = *x_shape
        .last()
        .ok_or_else(|| RmsNormError::Msg("kt-rmsnorm bwd: x must have rank >= 1".to_string()))?;
    if weight.device() != x.device() || grad_out.device() != x.device() {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm bwd: x, weight, and grad_out must share one device, got x={:?}, weight={:?}, grad_out={:?}",
            x.device(),
            weight.device(),
            grad_out.device()
        )));
    }
    if weight.shape() != [hidden] {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm bwd: weight {:?} != [{hidden}]",
            weight.shape()
        )));
    }
    if grad_out.shape() != x.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm bwd: grad_out {:?} != x {x_shape:?}",
            grad_out.shape()
        )));
    }
    let rows: usize = x_shape[..x_shape.len() - 1].iter().product();
    if rows == 0 {
        return Err(RmsNormError::Msg(
            "kt-rmsnorm bwd: rows=0 not supported".to_string(),
        ));
    }

    // Owner-agnostic input pointers (Phase 7 v2).
    let x_ptr = kiln_kt_bridge::device_input_ptr(x, KtDType::BF16, "x")?;
    let w_ptr = kiln_kt_bridge::device_input_ptr(weight, KtDType::BF16, "weight")?;
    let g_ptr = kiln_kt_bridge::device_input_ptr(grad_out, KtDType::BF16, "grad_out")?;
    let x_st = x;

    let grad_x = alloc_rmsnorm_backward_like(x_st, KtDType::BF16, x_shape.clone())?;
    // The kernel atomically reduces every row into one [hidden] F32 buffer.
    // Frozen-weight training passes null instead and allocates no dWeight.
    let grad_weight = if compute_grad_weight {
        Some(alloc_rmsnorm_backward_like(
            x_st,
            KtDType::F32,
            vec![hidden],
        )?)
    } else {
        None
    };

    let gx_ptr = kiln_kt_bridge::device_output_ptr(&grad_x);
    let gw_ptr = grad_weight
        .as_ref()
        .map(kiln_kt_bridge::device_output_ptr)
        .unwrap_or(0);

    // Launch on `grad_x`'s owning stream. ROCm backward outputs preserve x's
    // actual storage context, so both output zeroing memsets, the kernel writes,
    // and later output consumers are ordered on this stream. Inputs owned by a
    // different ROCm stream are handed off below. CUDA continues through its
    // existing allocation and stream path unchanged.
    #[cfg(feature = "rocm")]
    if matches!(x.device(), KtDevice::Rocm(_)) {
        let launch_stream = rocm_active_stream_identity(&grad_x, "grad_x")?;
        let x_owner_stream = rocm_owner_stream_identity(x, "x")?;
        let output_owner_stream = rocm_owner_stream_identity(&grad_x, "grad_x")?;
        if output_owner_stream != x_owner_stream {
            return Err(RmsNormError::Msg(
                "kt-rmsnorm bwd: output allocation did not preserve x's ROCm storage context"
                    .to_string(),
            ));
        }
        if let Some(grad_weight) = &grad_weight {
            let grad_weight_owner_stream = rocm_owner_stream_identity(grad_weight, "grad_weight")?;
            if grad_weight_owner_stream != output_owner_stream {
                return Err(RmsNormError::Msg(
                    "kt-rmsnorm bwd: backward outputs have different ROCm storage contexts"
                        .to_string(),
                ));
            }
        }
        synchronize_rocm_rmsnorm_backward_inputs(
            launch_stream,
            output_owner_stream,
            &[("x", x), ("weight", weight), ("grad_out", grad_out)],
        )?;
    }

    let stream_submission = device_stream_submission(&grad_x, "grad_x")?;
    let raw_stream = stream_submission.raw_stream();
    let status = unsafe {
        kiln_fused_rmsnorm_bwd(
            x_ptr as *const _,
            w_ptr as *const _,
            g_ptr as *const _,
            gx_ptr as *mut _,
            gw_ptr as *mut f32,
            rows as i32,
            hidden as i32,
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm bwd: FFI returned {status}"
        )));
    }
    stream_submission.complete();

    // No post-launch sync is needed outside capture because both outputs own
    // the launch stream. During capture, all in-graph consumers use the active
    // capture stream and the graph runner owns the replay handoff.

    Ok((grad_x, grad_weight))
}

/// `fused_rotary_qk` over `kiln_tensor::Tensor` operands.
///
/// In-place rotary application to Q and K projections. Inputs:
/// - `q`: BF16 `[batch, seq_len, q_heads, head_dim]`
/// - `k`: BF16 `[batch, seq_len, k_heads, head_dim]`
/// - `cos`, `sin`: F32 `[seq_len, rotary_dim / 2]` precomputed tables
///   (the half-dim layout the FFI kernel reads as
///   `cos[t * (rotary_dim/2) + d]`; matches `kiln_rmsnorm_kernel::fused_rotary_qk`
///   and the candle-typed `supports_rotary_qk` predicate).
/// - `rotary_dim`: applied head dim slice; must be ≤ head_dim and even.
///
/// Returns `(q_out, k_out)` BF16 tensors of the same shapes as the
/// inputs.
#[allow(clippy::too_many_arguments)]
pub fn fused_rotary_qk_kt(
    q: &KtTensor,
    k: &KtTensor,
    cos: &KtTensor,
    sin: &KtTensor,
    rotary_dim: usize,
) -> Result<(KtTensor, KtTensor), RmsNormError> {
    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: q must be [B, S, q_heads, head_dim], got {q_shape:?}"
        )));
    }
    let (batch, seq_len, q_heads, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let k_shape = k.shape();
    if k_shape.len() != 4 || (k_shape[0], k_shape[1], k_shape[3]) != (batch, seq_len, head_dim) {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: k {k_shape:?} != [{batch}, {seq_len}, k_heads, {head_dim}]"
        )));
    }
    let k_heads = k_shape[2];
    if rotary_dim > head_dim {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: rotary_dim {rotary_dim} > head_dim {head_dim}"
        )));
    }
    if !rotary_dim.is_multiple_of(2) {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: rotary_dim {rotary_dim} must be even"
        )));
    }
    let half = rotary_dim / 2;
    if cos.shape() != [seq_len, half] {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: cos {:?} != [{seq_len}, {half}]",
            cos.shape()
        )));
    }
    if sin.shape() != [seq_len, half] {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: sin {:?} != [{seq_len}, {half}]",
            sin.shape()
        )));
    }

    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::device_input_ptr(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::device_input_ptr(k, KtDType::BF16, "k")?;
    let cos_ptr = kiln_kt_bridge::device_input_ptr(cos, KtDType::F32, "cos")?;
    let sin_ptr = kiln_kt_bridge::device_input_ptr(sin, KtDType::F32, "sin")?;
    let q_st = q;

    let q_out = alloc_like(q_st, KtDType::BF16, vec![batch, seq_len, q_heads, head_dim])?;
    let k_out = alloc_like(q_st, KtDType::BF16, vec![batch, seq_len, k_heads, head_dim])?;
    let qo_ptr = kiln_kt_bridge::device_output_ptr(&q_out);
    let ko_ptr = kiln_kt_bridge::device_output_ptr(&k_out);

    let stream_submission = device_stream_submission(q_st, "q_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_fused_rotary_qk(
            q_ptr as *const _,
            k_ptr as *const _,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            qo_ptr as *mut _,
            ko_ptr as *mut _,
            batch as i32,
            seq_len as i32,
            q_heads as i32,
            k_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok((q_out, k_out))
}

/// `fused_mlp_silu_mul` over `kiln_tensor::Tensor` operands.
///
/// Element-wise: `out = silu(gate) * up`. Both inputs and output
/// are BF16 of equal element count. Used by the MLP gate||up||silu
/// fused path.
pub fn fused_mlp_silu_mul_kt(gate: &KtTensor, up: &KtTensor) -> Result<KtTensor, RmsNormError> {
    if gate.shape() != up.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-mlp-silu-mul: gate {:?} != up {:?}",
            gate.shape(),
            up.shape()
        )));
    }
    let elems = gate.element_count();
    let shape = gate.shape().to_vec();

    // Owner-agnostic input pointers (Phase 7 v2).
    let g_ptr = kiln_kt_bridge::device_input_ptr(gate, KtDType::BF16, "gate")?;
    let u_ptr = kiln_kt_bridge::device_input_ptr(up, KtDType::BF16, "up")?;
    let g_st = gate;
    let out = alloc_like(g_st, KtDType::BF16, shape)?;
    let o_ptr = kiln_kt_bridge::device_output_ptr(&out);

    let stream_submission = device_stream_submission(g_st, "g_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_fused_mlp_silu_mul_bf16(
            g_ptr as *const _,
            u_ptr as *const _,
            o_ptr as *mut _,
            elems as i64,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-mlp-silu-mul: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(out)
}

/// `sgd_step_f32` over `kiln_tensor::Tensor` operands.
///
/// In-place SGD update: `param -= lr * grad`. F32 only. `param` is
/// mutated in place through the raw device pointer; the caller must
/// hold a unique reference (kt-Tensor borrow-check is at the
/// version-counter layer, anti-pattern 16).
pub fn sgd_step_f32_kt(param: &KtTensor, grad: &KtTensor, lr: f32) -> Result<(), RmsNormError> {
    if param.shape() != grad.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-sgd-step: param {:?} != grad {:?}",
            param.shape(),
            grad.shape()
        )));
    }
    let n = param.element_count() as i64;

    // Owner-agnostic pointers. In-place ops require Owned `param` —
    // the FFI mutates through the pointer. Borrowed inputs would
    // silently mutate the external owner's buffer (UB from kt's
    // perspective). Caller convention: pass Owned for `param`.
    let p_ptr = kiln_kt_bridge::device_input_ptr(param, KtDType::F32, "param")?;
    let g_ptr = kiln_kt_bridge::device_input_ptr(grad, KtDType::F32, "grad")?;
    let p_st = param;

    let stream_submission = device_stream_submission(p_st, "p_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status =
        unsafe { kiln_sgd_step_f32(p_ptr as *mut f32, g_ptr as *const f32, lr, n, raw_stream) };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-sgd-step: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(())
}

/// `adamw_step_f32` over `kiln_tensor::Tensor` operands.
///
/// In-place AdamW update with running moments. F32 only. `param`,
/// `first_moment`, `second_moment` are all mutated in place. The
/// caller passes pre-computed bias-correction terms (matches the
/// candle path's contract).
#[allow(clippy::too_many_arguments)]
pub fn adamw_step_f32_kt(
    param: &KtTensor,
    grad: &KtTensor,
    first_moment: &KtTensor,
    second_moment: &KtTensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32,
    bias_correction2: f32,
) -> Result<(), RmsNormError> {
    if param.shape() != grad.shape()
        || param.shape() != first_moment.shape()
        || param.shape() != second_moment.shape()
    {
        return Err(RmsNormError::Msg(format!(
            "kt-adamw-step: shape mismatch — param {:?}, grad {:?}, m1 {:?}, m2 {:?}",
            param.shape(),
            grad.shape(),
            first_moment.shape(),
            second_moment.shape()
        )));
    }
    let n = param.element_count() as i64;

    // Owner-agnostic pointers. In-place ops require Owned mutable
    // operands (param, first_moment, second_moment).
    let p_ptr = kiln_kt_bridge::device_input_ptr(param, KtDType::F32, "param")?;
    let g_ptr = kiln_kt_bridge::device_input_ptr(grad, KtDType::F32, "grad")?;
    let m1_ptr = kiln_kt_bridge::device_input_ptr(first_moment, KtDType::F32, "first_moment")?;
    let m2_ptr = kiln_kt_bridge::device_input_ptr(second_moment, KtDType::F32, "second_moment")?;
    let p_st = param;

    let stream_submission = device_stream_submission(p_st, "p_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_adamw_step_f32(
            p_ptr as *mut f32,
            g_ptr as *const f32,
            m1_ptr as *mut f32,
            m2_ptr as *mut f32,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
            n,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-adamw-step: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(())
}

/// `(rows, cols)` for a Muon step over `param`. Rank-2 params keep their
/// `[rows, cols]` layout (the only shape the kernel orthogonalizes);
/// every other shape collapses to `(element_count, 1)` so the kernel's
/// non-matrix plain-momentum fallback runs over the flat buffer. The
/// kernel takes `i32` dims (`<<<1, BLK>>>`, one block per matrix), so we
/// reject anything past `i32::MAX`.
fn muon_rows_cols(param: &KtTensor) -> Result<(i32, i32), RmsNormError> {
    let shape = param.shape();
    let (rows, cols) = if shape.len() == 2 {
        (shape[0], shape[1])
    } else {
        (param.element_count(), 1)
    };
    if rows > i32::MAX as usize || cols > i32::MAX as usize {
        return Err(RmsNormError::Msg(format!(
            "kt-muon-step: dims ({rows}, {cols}) exceed i32 kernel envelope"
        )));
    }
    Ok((rows as i32, cols as i32))
}

/// `muon_step_f32` over `kiln_tensor::Tensor` operands.
///
/// In-place fused Muon step: heavy-ball momentum update, then (for
/// rank-2 weights within the kernel's shared-memory bound) Newton-Schulz
/// orthogonalization of the (Nesterov) look-ahead with the RMS-matching
/// `sqrt(max(rows, cols))` scale, then the decoupled-weight-decay descent
/// step. `param` and `momentum` are mutated in place; `grad` is
/// read-only. F32 only. Mirrors `kiln_optim::Muon::step`.
#[allow(clippy::too_many_arguments)]
pub fn muon_step_f32_kt(
    param: &KtTensor,
    grad: &KtTensor,
    momentum: &KtTensor,
    lr: f32,
    momentum_coef: f32,
    nesterov: bool,
    ns_iters: u32,
    weight_decay: f32,
) -> Result<(), RmsNormError> {
    if param.shape() != grad.shape() || param.shape() != momentum.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-muon-step: shape mismatch — param {:?}, grad {:?}, momentum {:?}",
            param.shape(),
            grad.shape(),
            momentum.shape()
        )));
    }
    if param.dtype() != KtDType::F32
        || grad.dtype() != KtDType::F32
        || momentum.dtype() != KtDType::F32
    {
        return Err(RmsNormError::Msg(format!(
            "kt-muon-step: dtype mismatch — param {:?}, grad {:?}, momentum {:?} (want F32)",
            param.dtype(),
            grad.dtype(),
            momentum.dtype()
        )));
    }
    if !param.is_contiguous() || !grad.is_contiguous() || !momentum.is_contiguous() {
        return Err(RmsNormError::Msg(
            "kt-muon-step: param/grad/momentum must be contiguous".to_string(),
        ));
    }
    let (rows, cols) = muon_rows_cols(param)?;

    // In-place op: `param` and `momentum` are mutated through their
    // device storage. Caller convention: pass Owned for both.
    let p_ptr = kiln_kt_bridge::device_input_ptr(param, KtDType::F32, "param")?;
    let g_ptr = kiln_kt_bridge::device_input_ptr(grad, KtDType::F32, "grad")?;
    let m_ptr = kiln_kt_bridge::device_input_ptr(momentum, KtDType::F32, "momentum")?;
    let p_st = param;

    let stream_submission = device_stream_submission(p_st, "p_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_muon_step_f32(
            p_ptr as *mut f32,
            g_ptr as *const f32,
            m_ptr as *mut f32,
            lr,
            momentum_coef,
            if nesterov { 1 } else { 0 },
            ns_iters as i32,
            weight_decay,
            rows,
            cols,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-muon-step: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(())
}

/// `muon_step_bf16` over `kiln_tensor::Tensor` operands.
///
/// BF16-master fused Muon step. `param`, `grad`, `momentum` all BF16;
/// `param` and `momentum` are mutated in place. See [`muon_step_f32_kt`]
/// for the F32-master variant and the full algorithm description.
#[allow(clippy::too_many_arguments)]
pub fn muon_step_bf16_kt(
    param: &KtTensor,
    grad: &KtTensor,
    momentum: &KtTensor,
    lr: f32,
    momentum_coef: f32,
    nesterov: bool,
    ns_iters: u32,
    weight_decay: f32,
) -> Result<(), RmsNormError> {
    if param.shape() != grad.shape() || param.shape() != momentum.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-muon-step-bf16: shape mismatch — param {:?}, grad {:?}, momentum {:?}",
            param.shape(),
            grad.shape(),
            momentum.shape()
        )));
    }
    if param.dtype() != KtDType::BF16
        || grad.dtype() != KtDType::BF16
        || momentum.dtype() != KtDType::BF16
    {
        return Err(RmsNormError::Msg(format!(
            "kt-muon-step-bf16: dtype mismatch — param {:?}, grad {:?}, momentum {:?} (want BF16)",
            param.dtype(),
            grad.dtype(),
            momentum.dtype()
        )));
    }
    if !param.is_contiguous() || !grad.is_contiguous() || !momentum.is_contiguous() {
        return Err(RmsNormError::Msg(
            "kt-muon-step-bf16: param/grad/momentum must be contiguous".to_string(),
        ));
    }
    let (rows, cols) = muon_rows_cols(param)?;

    let p_ptr = kiln_kt_bridge::device_input_ptr(param, KtDType::BF16, "param")?;
    let g_ptr = kiln_kt_bridge::device_input_ptr(grad, KtDType::BF16, "grad")?;
    let m_ptr = kiln_kt_bridge::device_input_ptr(momentum, KtDType::BF16, "momentum")?;
    let p_st = param;

    let stream_submission = device_stream_submission(p_st, "p_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_muon_step_bf16(
            p_ptr as *mut _,
            g_ptr as *const _,
            m_ptr as *mut _,
            lr,
            momentum_coef,
            if nesterov { 1 } else { 0 },
            ns_iters as i32,
            weight_decay,
            rows,
            cols,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-muon-step-bf16: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(())
}

/// `lora_decode_hidden_bf16` over `kiln_tensor::Tensor` operands.
///
/// Computes the LoRA-A projection at decode time: `hidden = x @ A`,
/// where:
/// - `x`: BF16 `[batch, in_dim]`
/// - `a`: BF16 `[rank, in_dim]` (the LoRA-A matrix)
///
/// Returns F32 `[batch, rank]` (the LoRA hidden state, in F32 for
/// downstream numerical accuracy). Used by the multi-LoRA decode
/// path (line 307 of #1082).
pub fn lora_decode_hidden_kt(x: &KtTensor, a: &KtTensor) -> Result<KtTensor, RmsNormError> {
    let x_shape = x.shape();
    if x_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-hidden: x must be [batch, in_dim], got {x_shape:?}"
        )));
    }
    let (batch, in_dim) = (x_shape[0], x_shape[1]);
    let a_shape = a.shape();
    if a_shape.len() != 2 || a_shape[1] != in_dim {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-hidden: a {a_shape:?} != [rank, {in_dim}]"
        )));
    }
    let rank = a_shape[0];

    let x_ptr = kiln_kt_bridge::device_input_ptr(x, KtDType::BF16, "x")?;
    let a_ptr = kiln_kt_bridge::device_input_ptr(a, KtDType::BF16, "a")?;
    let x_st = x;
    let hidden = alloc_like(x_st, KtDType::F32, vec![batch, rank])?;
    let h_ptr = kiln_kt_bridge::device_output_ptr(&hidden);

    let stream_submission = device_stream_submission(x_st, "x_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_lora_decode_hidden_bf16(
            x_ptr as *const _,
            a_ptr as *const _,
            h_ptr as *mut f32,
            batch as i32,
            in_dim as i32,
            rank as i32,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-lora-hidden: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(hidden)
}

/// `lora_decode_add_bf16` over `kiln_tensor::Tensor` operands.
///
/// Adds the LoRA-B contribution to the base projection:
/// `out = base + scale * (hidden @ B)`, where:
/// - `base`: BF16 `[batch, out_dim]` (the base linear projection)
/// - `hidden`: F32 `[batch, rank]` (output of [`lora_decode_hidden_kt`])
/// - `b`: BF16 `[out_dim, rank]` (the LoRA-B matrix)
/// - `scale`: f32 LoRA alpha scale
///
/// Returns BF16 `[batch, out_dim]`.
pub fn lora_decode_add_kt(
    base: &KtTensor,
    hidden: &KtTensor,
    b: &KtTensor,
    scale: f32,
) -> Result<KtTensor, RmsNormError> {
    let base_shape = base.shape();
    if base_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: base must be [batch, out_dim], got {base_shape:?}"
        )));
    }
    let (batch, out_dim) = (base_shape[0], base_shape[1]);
    let h_shape = hidden.shape();
    if h_shape.len() != 2 || h_shape[0] != batch {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: hidden {h_shape:?} != [{batch}, rank]"
        )));
    }
    let rank = h_shape[1];
    let b_shape = b.shape();
    if b_shape != [out_dim, rank] {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: b {b_shape:?} != [{out_dim}, {rank}]"
        )));
    }

    let base_ptr = kiln_kt_bridge::device_input_ptr(base, KtDType::BF16, "base")?;
    let h_ptr = kiln_kt_bridge::device_input_ptr(hidden, KtDType::F32, "hidden")?;
    let b_ptr = kiln_kt_bridge::device_input_ptr(b, KtDType::BF16, "b")?;
    let base_st = base;
    let out = alloc_like(base_st, KtDType::BF16, vec![batch, out_dim])?;
    let o_ptr = kiln_kt_bridge::device_output_ptr(&out);

    let stream_submission = device_stream_submission(base_st, "base_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_lora_decode_add_bf16(
            base_ptr as *const _,
            h_ptr as *const f32,
            b_ptr as *const _,
            o_ptr as *mut _,
            scale,
            batch as i32,
            out_dim as i32,
            rank as i32,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(out)
}

/// Full kt twin of [`crate::lora_decode_add`].
///
/// Accepts the production decode shapes `base=[batch,1,out]`,
/// `x=[batch,1,in]`, `a=[rank,in]`, `b=[out,rank]`, computes the hidden
/// LoRA-A projection and fused LoRA-B add, and returns BF16
/// `[batch,1,out]`.
pub fn lora_decode_add_full_kt(
    base: &KtTensor,
    x: &KtTensor,
    a: &KtTensor,
    b: &KtTensor,
    scale: f32,
) -> Result<KtTensor, RmsNormError> {
    if !supports_lora_decode_add_kt(base, x, a, b) {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-decode-add: unsupported shapes base={:?} x={:?} a={:?} b={:?} dtypes=({:?},{:?},{:?},{:?})",
            base.shape(),
            x.shape(),
            a.shape(),
            b.shape(),
            base.dtype(),
            x.dtype(),
            a.dtype(),
            b.dtype()
        )));
    }
    let batch = base.shape()[0];
    let out_dim = base.shape()[2];
    let in_dim = x.shape()[2];
    let base2 = base
        .reshape(vec![batch, out_dim])
        .map_err(|e| RmsNormError::Msg(format!("kt-lora-decode-add: base reshape: {e}")))?;
    let x2 = x
        .reshape(vec![batch, in_dim])
        .map_err(|e| RmsNormError::Msg(format!("kt-lora-decode-add: x reshape: {e}")))?;
    let hidden = lora_decode_hidden_kt(&x2, a)?;
    let out2 = lora_decode_add_kt(&base2, &hidden, b, scale)?;
    out2.reshape(vec![batch, 1, out_dim])
        .map_err(|e| RmsNormError::Msg(format!("kt-lora-decode-add: out reshape: {e}")))
}

/// `fused_l2_qk_norm` over `kiln_tensor::Tensor` operands.
///
/// L2-normalize each row of `q_in` and `k_in` (shape `[rows, hidden]`,
/// BF16) and scale `q` by `q_scale`. Used by the QK-norm pass in
/// attention pre-projection. Returns `(q_out, k_out)`.
pub fn fused_l2_qk_norm_kt(
    q_in: &KtTensor,
    k_in: &KtTensor,
    q_scale: f32,
    eps: f32,
) -> Result<(KtTensor, KtTensor), RmsNormError> {
    if q_in.shape() != k_in.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-l2-qk-norm: q {:?} != k {:?}",
            q_in.shape(),
            k_in.shape()
        )));
    }
    let q_shape = q_in.shape().to_vec();
    if q_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-l2-qk-norm: q must be [rows, hidden], got {q_shape:?}"
        )));
    }
    let (rows, hidden) = (q_shape[0], q_shape[1]);

    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::device_input_ptr(q_in, KtDType::BF16, "q_in")?;
    let k_ptr = kiln_kt_bridge::device_input_ptr(k_in, KtDType::BF16, "k_in")?;
    let q_st = q_in;
    let q_out = alloc_like(q_st, KtDType::BF16, q_shape.clone())?;
    let k_out = alloc_like(q_st, KtDType::BF16, q_shape)?;
    let qo_ptr = kiln_kt_bridge::device_output_ptr(&q_out);
    let ko_ptr = kiln_kt_bridge::device_output_ptr(&k_out);

    let stream_submission = device_stream_submission(q_st, "q_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_fused_l2_qk_norm(
            q_ptr as *const _,
            k_ptr as *const _,
            qo_ptr as *mut _,
            ko_ptr as *mut _,
            rows as i32,
            hidden as i32,
            q_scale,
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-l2-qk-norm: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok((q_out, k_out))
}

/// GQA variant of `fused_l2_qk_norm`.
///
/// Inputs are unexpanded GDN Q/K tensors `[batch, seq, nk, dk]`; outputs are
/// freshly allocated BF16 tensors `[batch, seq, nv, dk]`, with each normalized
/// input head repeated `nv / nk` times. Semantics match the candle-typed
/// [`crate::fused_l2_qk_norm_gqa`] wrapper.
pub fn fused_l2_qk_norm_gqa_kt(
    q_in: &KtTensor,
    k_in: &KtTensor,
    nv: usize,
    q_scale: f32,
    eps: f32,
) -> Result<(KtTensor, KtTensor), RmsNormError> {
    if !supports_l2_qk_norm_gqa_kt(q_in, k_in, nv) {
        return Err(RmsNormError::Msg(format!(
            "kt-l2-qk-norm-gqa: unsupported shapes q={:?} k={:?} dtypes=({:?},{:?}) nv={nv}",
            q_in.shape(),
            k_in.shape(),
            q_in.dtype(),
            k_in.dtype()
        )));
    }

    let q_shape = q_in.shape();
    let batch = q_shape[0];
    let seq = q_shape[1];
    let nk = q_shape[2];
    let head_dim = q_shape[3];
    let ratio = nv / nk;
    let rows = batch * seq * nk;

    // Owner-agnostic input pointers (Phase 7 v2).
    let q_contig = q_in
        .contiguous()
        .map_err(|e| RmsNormError::Msg(format!("kt-l2-qk-norm-gqa: q contiguous: {e}")))?;
    let k_contig = k_in
        .contiguous()
        .map_err(|e| RmsNormError::Msg(format!("kt-l2-qk-norm-gqa: k contiguous: {e}")))?;
    let q_ptr = kiln_kt_bridge::device_input_ptr(&q_contig, KtDType::BF16, "q_in")?;
    let k_ptr = kiln_kt_bridge::device_input_ptr(&k_contig, KtDType::BF16, "k_in")?;
    let q_st = &q_contig;
    let out_shape = vec![batch, seq, nv, head_dim];
    let q_out = alloc_like(q_st, KtDType::BF16, out_shape.clone())?;
    let k_out = alloc_like(q_st, KtDType::BF16, out_shape)?;
    let qo_ptr = kiln_kt_bridge::device_output_ptr(&q_out);
    let ko_ptr = kiln_kt_bridge::device_output_ptr(&k_out);

    let stream_submission = device_stream_submission(q_st, "q_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_fused_l2_qk_norm_gqa(
            q_ptr as *const _,
            k_ptr as *const _,
            qo_ptr as *mut _,
            ko_ptr as *mut _,
            rows as i32,
            nk as i32,
            ratio as i32,
            head_dim as i32,
            q_scale,
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-l2-qk-norm-gqa: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok((q_out, k_out))
}

/// `fused_rotary_one` over `kiln_tensor::Tensor` operands.
///
/// Single-tensor rotary application (used for Q or K, but not both
/// in one launch — see [`fused_rotary_qk_kt`] for the fused pair).
/// Shape: `[batch, seq_len, heads, head_dim]` BF16; rotary applied
/// to the first `rotary_dim` of head_dim. Returns the rotated tensor.
pub fn fused_rotary_one_kt(
    x: &KtTensor,
    cos: &KtTensor,
    sin: &KtTensor,
    rotary_dim: usize,
) -> Result<KtTensor, RmsNormError> {
    let x_shape = x.shape();
    if x_shape.len() != 4 {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one: x must be [B, S, H, D], got {x_shape:?}"
        )));
    }
    let (batch, seq_len, heads, head_dim) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    if rotary_dim > head_dim {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one: rotary_dim {rotary_dim} > head_dim {head_dim}"
        )));
    }
    if !rotary_dim.is_multiple_of(2) {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one: rotary_dim {rotary_dim} must be even"
        )));
    }
    let half = rotary_dim / 2;
    if cos.shape() != [seq_len, half] || sin.shape() != [seq_len, half] {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one: cos/sin must be [{seq_len}, {half}]"
        )));
    }

    // Owner-agnostic input pointers (Phase 7 v2).
    let x_ptr = kiln_kt_bridge::device_input_ptr(x, KtDType::BF16, "x")?;
    let cos_ptr = kiln_kt_bridge::device_input_ptr(cos, KtDType::F32, "cos")?;
    let sin_ptr = kiln_kt_bridge::device_input_ptr(sin, KtDType::F32, "sin")?;
    let x_st = x;

    let out = alloc_like(x_st, KtDType::BF16, vec![batch, seq_len, heads, head_dim])?;
    let o_ptr = kiln_kt_bridge::device_output_ptr(&out);

    let stream_submission = device_stream_submission(x_st, "x_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_fused_rotary_one(
            x_ptr as *const _,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            o_ptr as *mut _,
            batch as i32,
            seq_len as i32,
            heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(out)
}

/// `fused_sigmoid_mul_bf16` over `kiln_tensor::Tensor` operands.
///
/// Element-wise: `out = sigmoid(gate) * x`. Both BF16, same shape.
/// Used by gated activation paths. Like `fused_mlp_silu_mul` but
/// with sigmoid instead of silu.
pub fn fused_sigmoid_mul_kt(x: &KtTensor, gate: &KtTensor) -> Result<KtTensor, RmsNormError> {
    if x.shape() != gate.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-sigmoid-mul: x {:?} != gate {:?}",
            x.shape(),
            gate.shape()
        )));
    }
    let elems = x.element_count();
    let shape = x.shape().to_vec();

    // Owner-agnostic input device pointers: works for both Owned and
    // Borrowed kt storage (Phase 7 v2 — accepts kt-Tensors built via
    // `kt_tensor_from_candle_cuda_borrow`). This is the migration
    // template for the rest of the kt-API surface.
    let x_ptr = kiln_kt_bridge::device_input_ptr(x, KtDType::BF16, "x")?;
    let g_ptr = kiln_kt_bridge::device_input_ptr(gate, KtDType::BF16, "gate")?;

    // Output is always Owned (alloc_cuda_tensor produces owned storage),
    // so we can reach for the raw pointer the same way.
    let x_st = x;
    let out = alloc_like(x_st, KtDType::BF16, shape)?;
    let o_ptr = kiln_kt_bridge::device_output_ptr(&out);

    let stream_submission = device_stream_submission(x_st, "x_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_fused_sigmoid_mul_bf16(
            x_ptr as *const _,
            g_ptr as *const _,
            o_ptr as *mut _,
            elems as i64,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-sigmoid-mul: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(out)
}

/// `attn_decode_qkv_split_qk_norm_rope_bf16` over `kiln_tensor::Tensor`
/// operands.
///
/// The decode-time mega-fused attention pre-projection kernel.
/// Splits the QKV-packed Q/gate inputs, applies QK-norm (with eps),
/// then applies RoPE — all in one launch. Optionally writes the
/// gate output if `has_gate` is true.
///
/// Shapes (production decode):
/// - `q_raw`: BF16 `[batch, 1, q_heads * head_dim + (q_heads * head_dim if has_gate)]`
/// - `k_raw`: BF16 `[batch, 1, k_heads * head_dim]`
/// - `q_weight`, `k_weight`: BF16 `[head_dim]`
/// - `cos`, `sin`: F32 `[1, rotary_dim / 2]`
///
/// Returns `(q_out, k_out, gate_out)` where:
/// - `q_out`: BF16 `[batch, 1, q_heads, head_dim]`
/// - `k_out`: BF16 `[batch, 1, k_heads, head_dim]`
/// - `gate_out`: Some(BF16 `[batch, 1, q_heads * head_dim]`) if has_gate, else None
#[allow(clippy::too_many_arguments)]
pub fn attn_decode_qkv_split_qk_norm_rope_kt(
    q_raw: &KtTensor,
    k_raw: &KtTensor,
    q_weight: &KtTensor,
    k_weight: &KtTensor,
    cos: &KtTensor,
    sin: &KtTensor,
    q_heads: usize,
    k_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    has_gate: bool,
    eps: f32,
) -> Result<(KtTensor, KtTensor, Option<KtTensor>), RmsNormError> {
    let qr_shape = q_raw.shape();
    if qr_shape.len() != 3 || qr_shape[1] != 1 {
        return Err(RmsNormError::Msg(format!(
            "kt-attn-decode-prep: q_raw must be [B, 1, hidden_q], got {qr_shape:?}"
        )));
    }
    let batch = qr_shape[0];

    // Owner-agnostic input pointers (Phase 7 v2).
    let qr_ptr = kiln_kt_bridge::device_input_ptr(q_raw, KtDType::BF16, "q_raw")?;
    let kr_ptr = kiln_kt_bridge::device_input_ptr(k_raw, KtDType::BF16, "k_raw")?;
    let qw_ptr = kiln_kt_bridge::device_input_ptr(q_weight, KtDType::BF16, "q_weight")?;
    let kw_ptr = kiln_kt_bridge::device_input_ptr(k_weight, KtDType::BF16, "k_weight")?;
    let cos_ptr = kiln_kt_bridge::device_input_ptr(cos, KtDType::F32, "cos")?;
    let sin_ptr = kiln_kt_bridge::device_input_ptr(sin, KtDType::F32, "sin")?;
    let qr_st = q_raw;

    let q_out = alloc_like(qr_st, KtDType::BF16, vec![batch, 1, q_heads, head_dim])?;
    let k_out = alloc_like(qr_st, KtDType::BF16, vec![batch, 1, k_heads, head_dim])?;
    let gate_out = if has_gate {
        Some(alloc_like(
            qr_st,
            KtDType::BF16,
            vec![batch, 1, q_heads * head_dim],
        )?)
    } else {
        None
    };
    let qo_ptr = kiln_kt_bridge::device_output_ptr(&q_out);
    let ko_ptr = kiln_kt_bridge::device_output_ptr(&k_out);
    let go_ptr = gate_out
        .as_ref()
        .map(|go| kiln_kt_bridge::device_output_ptr(go) as *mut _)
        .unwrap_or(core::ptr::null_mut());

    let stream_submission = device_stream_submission(qr_st, "qr_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_attn_decode_qkv_split_qk_norm_rope_bf16(
            qr_ptr as *const _,
            kr_ptr as *const _,
            qw_ptr as *const _,
            kw_ptr as *const _,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            qo_ptr as *mut _,
            ko_ptr as *mut _,
            go_ptr,
            batch as i32,
            q_heads as i32,
            k_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            if has_gate { 1 } else { 0 },
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-attn-decode-prep: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok((q_out, k_out, gate_out))
}

// ============================================================================
// Causal depthwise conv1d family (F32; used by training paths)
// ============================================================================

/// `causal_depthwise_conv1d_f32` over kt operands.
///
/// `input`: F32 `[rows, channels]` (one row per time step).
/// `weight`: F32 `[channels, kernel]`. `state`: F32 `[channels, kernel-1]`.
/// Returns F32 `[rows, channels]`.
pub fn causal_depthwise_conv1d_kt(
    input: &KtTensor,
    weight: &KtTensor,
    state: &KtTensor,
    kernel: usize,
) -> Result<KtTensor, RmsNormError> {
    let input_shape = input.shape();
    if input_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d: input must be [rows, channels], got {input_shape:?}"
        )));
    }
    let (rows, channels) = (input_shape[0], input_shape[1]);
    if weight.shape() != [channels, kernel] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d: weight {:?} != [{channels}, {kernel}]",
            weight.shape()
        )));
    }
    if state.shape() != [channels, kernel - 1] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d: state {:?} != [{channels}, {}]",
            state.shape(),
            kernel - 1
        )));
    }
    // Owner-agnostic input pointers (Phase 7 v2).
    let i_ptr = kiln_kt_bridge::device_input_ptr(input, KtDType::F32, "input")?;
    let w_ptr = kiln_kt_bridge::device_input_ptr(weight, KtDType::F32, "weight")?;
    let s_ptr = kiln_kt_bridge::device_input_ptr(state, KtDType::F32, "state")?;
    let i_st = input;
    let out = alloc_like(i_st, KtDType::F32, vec![rows, channels])?;
    let o_ptr = kiln_kt_bridge::device_output_ptr(&out);

    let stream_submission = device_stream_submission(i_st, "i_st")?;
    let raw_stream = stream_submission.raw_stream();
    let status = unsafe {
        kiln_causal_depthwise_conv1d_f32(
            i_ptr as *const f32,
            w_ptr as *const f32,
            s_ptr as *const f32,
            o_ptr as *mut f32,
            rows as i32,
            channels as i32,
            kernel as i32,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(out)
}

/// In-place variant of [`causal_depthwise_conv1d_kt`]. `input_out`
/// is mutated in place; no allocation, no copy. Returns `()`.
pub fn causal_depthwise_conv1d_inplace_kt(
    input_out: &KtTensor,
    weight: &KtTensor,
    state: &KtTensor,
    kernel: usize,
) -> Result<(), RmsNormError> {
    let shape = input_out.shape();
    if shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-inplace: input_out must be [rows, channels], got {shape:?}"
        )));
    }
    let (rows, channels) = (shape[0], shape[1]);
    if weight.shape() != [channels, kernel] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-inplace: weight {:?} != [{channels}, {kernel}]",
            weight.shape()
        )));
    }
    if state.shape() != [channels, kernel - 1] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-inplace: state {:?} != [{channels}, {}]",
            state.shape(),
            kernel - 1
        )));
    }
    // In-place op — `input_out` is mutated. Caller convention: pass Owned.
    let i_ptr = kiln_kt_bridge::device_input_ptr(input_out, KtDType::F32, "input_out")?;
    let w_ptr = kiln_kt_bridge::device_input_ptr(weight, KtDType::F32, "weight")?;
    let s_ptr = kiln_kt_bridge::device_input_ptr(state, KtDType::F32, "state")?;
    let i_st = input_out;
    let stream_submission = device_stream_submission(i_st, "i_st")?;
    let raw_stream = stream_submission.raw_stream();
    let status = unsafe {
        kiln_causal_depthwise_conv1d_inplace_f32(
            i_ptr as *mut f32,
            w_ptr as *const f32,
            s_ptr as *const f32,
            rows as i32,
            channels as i32,
            kernel as i32,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-inplace: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(())
}

/// Backward through `causal_depthwise_conv1d` w.r.t. the input.
pub fn causal_depthwise_conv1d_bwd_input_kt(
    grad_out: &KtTensor,
    weight: &KtTensor,
    kernel: usize,
) -> Result<KtTensor, RmsNormError> {
    let go_shape = grad_out.shape();
    if go_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-input: grad_out must be [rows, channels], got {go_shape:?}"
        )));
    }
    let (rows, channels) = (go_shape[0], go_shape[1]);
    if weight.shape() != [channels, kernel] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-input: weight {:?} != [{channels}, {kernel}]",
            weight.shape()
        )));
    }
    let g_ptr = kiln_kt_bridge::device_input_ptr(grad_out, KtDType::F32, "grad_out")?;
    let w_ptr = kiln_kt_bridge::device_input_ptr(weight, KtDType::F32, "weight")?;
    let g_st = grad_out;
    let gi = alloc_like(g_st, KtDType::F32, vec![rows, channels])?;
    let gi_ptr = kiln_kt_bridge::device_output_ptr(&gi);
    let stream_submission = device_stream_submission(g_st, "g_st")?;
    let raw_stream = stream_submission.raw_stream();
    let status = unsafe {
        kiln_causal_depthwise_conv1d_bwd_input_f32(
            g_ptr as *const f32,
            w_ptr as *const f32,
            gi_ptr as *mut f32,
            rows as i32,
            channels as i32,
            kernel as i32,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-input: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(gi)
}

/// Backward w.r.t. the weight. Output shape: `[channels, kernel]`.
pub fn causal_depthwise_conv1d_bwd_weight_kt(
    grad_out: &KtTensor,
    input: &KtTensor,
    state: &KtTensor,
    kernel: usize,
) -> Result<KtTensor, RmsNormError> {
    let go_shape = grad_out.shape();
    if go_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-weight: grad_out must be [rows, channels], got {go_shape:?}"
        )));
    }
    let (rows, channels) = (go_shape[0], go_shape[1]);
    if input.shape() != [rows, channels] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-weight: input {:?} != [{rows}, {channels}]",
            input.shape()
        )));
    }
    if state.shape() != [channels, kernel - 1] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-weight: state {:?} != [{channels}, {}]",
            state.shape(),
            kernel - 1
        )));
    }
    let g_ptr = kiln_kt_bridge::device_input_ptr(grad_out, KtDType::F32, "grad_out")?;
    let i_ptr = kiln_kt_bridge::device_input_ptr(input, KtDType::F32, "input")?;
    let s_ptr = kiln_kt_bridge::device_input_ptr(state, KtDType::F32, "state")?;
    let g_st = grad_out;
    let gw = alloc_like(g_st, KtDType::F32, vec![channels, kernel])?;
    let gw_ptr = kiln_kt_bridge::device_output_ptr(&gw);
    let stream_submission = device_stream_submission(g_st, "g_st")?;
    let raw_stream = stream_submission.raw_stream();
    let status = unsafe {
        kiln_causal_depthwise_conv1d_bwd_weight_f32(
            g_ptr as *const f32,
            i_ptr as *const f32,
            s_ptr as *const f32,
            gw_ptr as *mut f32,
            rows as i32,
            channels as i32,
            kernel as i32,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-weight: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(gw)
}

/// Backward w.r.t. the conv state. Output shape: `[channels, kernel-1]`.
pub fn causal_depthwise_conv1d_bwd_state_kt(
    grad_out: &KtTensor,
    weight: &KtTensor,
    kernel: usize,
) -> Result<KtTensor, RmsNormError> {
    let go_shape = grad_out.shape();
    if go_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-state: grad_out must be [rows, channels], got {go_shape:?}"
        )));
    }
    let (rows, channels) = (go_shape[0], go_shape[1]);
    if weight.shape() != [channels, kernel] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-state: weight {:?} != [{channels}, {kernel}]",
            weight.shape()
        )));
    }
    let g_ptr = kiln_kt_bridge::device_input_ptr(grad_out, KtDType::F32, "grad_out")?;
    let w_ptr = kiln_kt_bridge::device_input_ptr(weight, KtDType::F32, "weight")?;
    let g_st = grad_out;
    let gs = alloc_like(g_st, KtDType::F32, vec![channels, kernel - 1])?;
    let gs_ptr = kiln_kt_bridge::device_output_ptr(&gs);
    let stream_submission = device_stream_submission(g_st, "g_st")?;
    let raw_stream = stream_submission.raw_stream();
    let status = unsafe {
        kiln_causal_depthwise_conv1d_bwd_state_f32(
            g_ptr as *const f32,
            w_ptr as *const f32,
            gs_ptr as *mut f32,
            rows as i32,
            channels as i32,
            kernel as i32,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-state: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(gs)
}

/// Suppresses unused-import warnings for the bwd-state extern import
/// in builds that don't reference it directly through the kt-API.
#[allow(dead_code)]
fn _unused_imports_keep() {
    let _ = kiln_causal_depthwise_conv1d_bwd_input_f32;
    let _ = kiln_causal_depthwise_conv1d_bwd_weight_f32;
    let _ = kiln_causal_depthwise_conv1d_bwd_state_f32;
}

/// `kiln_f32_to_bf16` over kt operands.
///
/// Casts an F32 tensor of arbitrary shape to BF16 element-wise.
/// Returns a freshly allocated BF16 tensor with the same shape.
pub fn f32_to_bf16_kt(src: &KtTensor) -> Result<KtTensor, RmsNormError> {
    let shape = src.shape().to_vec();
    let n = src.element_count();
    let s_ptr = kiln_kt_bridge::device_input_ptr(src, KtDType::F32, "src")?;
    let s_st = src;
    let out = alloc_like(s_st, KtDType::BF16, shape)?;
    let o_ptr = kiln_kt_bridge::device_output_ptr(&out);

    let stream_submission = device_stream_submission(s_st, "s_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status =
        unsafe { kiln_f32_to_bf16(s_ptr as *const f32, o_ptr as *mut _, n as i32, raw_stream) };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-f32-to-bf16: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(out)
}

/// `sgd_step_bf16` over `kiln_tensor::Tensor` operands.
///
/// BF16-master SGD step. `param` and `grad` both BF16; updated in
/// place. See [`sgd_step_f32_kt`] for the F32-master variant.
pub fn sgd_step_bf16_kt(param: &KtTensor, grad: &KtTensor, lr: f32) -> Result<(), RmsNormError> {
    if param.shape() != grad.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-sgd-step-bf16: param {:?} != grad {:?}",
            param.shape(),
            grad.shape()
        )));
    }
    let n = param.element_count() as i64;
    let p_ptr = kiln_kt_bridge::device_input_ptr(param, KtDType::BF16, "param")?;
    let g_ptr = kiln_kt_bridge::device_input_ptr(grad, KtDType::BF16, "grad")?;
    let p_st = param;

    let stream_submission = device_stream_submission(p_st, "p_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status =
        unsafe { kiln_sgd_step_bf16(p_ptr as *mut _, g_ptr as *const _, lr, n, raw_stream) };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-sgd-step-bf16: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(())
}

/// `adamw_step_bf16` over `kiln_tensor::Tensor` operands.
///
/// BF16-master AdamW step. All four buffers (param, first_moment,
/// second_moment, grad) are BF16. Mutates `param`, `first_moment`,
/// `second_moment` in place.
#[allow(clippy::too_many_arguments)]
pub fn adamw_step_bf16_kt(
    param: &KtTensor,
    grad: &KtTensor,
    first_moment: &KtTensor,
    second_moment: &KtTensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32,
    bias_correction2: f32,
) -> Result<(), RmsNormError> {
    if param.shape() != grad.shape()
        || param.shape() != first_moment.shape()
        || param.shape() != second_moment.shape()
    {
        return Err(RmsNormError::Msg(format!(
            "kt-adamw-step-bf16: shape mismatch — param {:?}, grad {:?}, m1 {:?}, m2 {:?}",
            param.shape(),
            grad.shape(),
            first_moment.shape(),
            second_moment.shape()
        )));
    }
    let n = param.element_count() as i64;

    let p_ptr = kiln_kt_bridge::device_input_ptr(param, KtDType::BF16, "param")?;
    let g_ptr = kiln_kt_bridge::device_input_ptr(grad, KtDType::BF16, "grad")?;
    let m1_ptr = kiln_kt_bridge::device_input_ptr(first_moment, KtDType::BF16, "first_moment")?;
    let m2_ptr = kiln_kt_bridge::device_input_ptr(second_moment, KtDType::BF16, "second_moment")?;
    let p_st = param;

    let stream_submission = device_stream_submission(p_st, "p_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_adamw_step_bf16(
            p_ptr as *mut _,
            g_ptr as *const _,
            m1_ptr as *mut _,
            m2_ptr as *mut _,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
            n,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-adamw-step-bf16: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(())
}

/// `fused_rotary_one_bwd` over `kiln_tensor::Tensor` operands.
///
/// Backward of [`fused_rotary_one_kt`]. Takes `grad_y` BF16
/// `[B, S, H, D]`, the original `cos`/`sin` F32 tables `[S, R]`, and
/// returns `grad_x` BF16 `[B, S, H, D]` (the gradient w.r.t. the input).
pub fn fused_rotary_one_bwd_kt(
    grad_y: &KtTensor,
    cos: &KtTensor,
    sin: &KtTensor,
    rotary_dim: usize,
) -> Result<KtTensor, RmsNormError> {
    let y_shape = grad_y.shape();
    if y_shape.len() != 4 {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one-bwd: grad_y must be [B, S, H, D], got {y_shape:?}"
        )));
    }
    let (batch, seq_len, heads, head_dim) = (y_shape[0], y_shape[1], y_shape[2], y_shape[3]);
    if rotary_dim > head_dim {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one-bwd: rotary_dim {rotary_dim} > head_dim {head_dim}"
        )));
    }
    if !rotary_dim.is_multiple_of(2) {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one-bwd: rotary_dim {rotary_dim} must be even"
        )));
    }
    let half = rotary_dim / 2;
    if cos.shape() != [seq_len, half] || sin.shape() != [seq_len, half] {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one-bwd: cos/sin must be [{seq_len}, {half}]"
        )));
    }

    let y_ptr = kiln_kt_bridge::device_input_ptr(grad_y, KtDType::BF16, "grad_y")?;
    let cos_ptr = kiln_kt_bridge::device_input_ptr(cos, KtDType::F32, "cos")?;
    let sin_ptr = kiln_kt_bridge::device_input_ptr(sin, KtDType::F32, "sin")?;
    let y_st = grad_y;

    let out = alloc_like(y_st, KtDType::BF16, vec![batch, seq_len, heads, head_dim])?;
    let o_ptr = kiln_kt_bridge::device_output_ptr(&out);

    let stream_submission = device_stream_submission(y_st, "y_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_fused_rotary_one_bwd(
            y_ptr as *const _,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            o_ptr as *mut _,
            batch as i32,
            seq_len as i32,
            heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one-bwd: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(out)
}

/// `fused_mlp_silu_mul_packed_bf16` over kt operands.
///
/// Packed MLP variant: input is `[.., 2*cols]` where the first half is
/// the gate and the second half is the up projection (already
/// matmul-fused). Computes `silu(gate) * up` and returns `[.., cols]`.
pub fn fused_mlp_silu_mul_packed_kt(
    gate_up_packed: &KtTensor,
    cols: usize,
) -> Result<KtTensor, RmsNormError> {
    let dims = gate_up_packed.shape().to_vec();
    if dims.is_empty() || dims[dims.len() - 1] != 2 * cols {
        return Err(RmsNormError::Msg(format!(
            "kt-mlp-packed: gate_up last dim {:?} != 2*cols (cols={cols})",
            dims.last(),
        )));
    }
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let mut out_dims: Vec<usize> = dims[..dims.len() - 1].to_vec();
    out_dims.push(cols);

    let gu_ptr = kiln_kt_bridge::device_input_ptr(gate_up_packed, KtDType::BF16, "gate_up_packed")?;
    let gu_st = gate_up_packed;
    let out = alloc_like(gu_st, KtDType::BF16, out_dims)?;
    if rows == 0 {
        return Ok(out);
    }
    let o_ptr = kiln_kt_bridge::device_output_ptr(&out);

    let stream_submission = device_stream_submission(gu_st, "gu_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_fused_mlp_silu_mul_packed_bf16(
            gu_ptr as *const _,
            o_ptr as *mut _,
            rows as i64,
            cols as i64,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-mlp-packed: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(out)
}

/// `lora_add_inplace_f32` over kt operands.
///
/// In-place LoRA-B fused add. Computes
/// `base += scale * (hidden @ B)` in F32, where:
/// - `base`:   F32 `[rows, out_dim]` — mutated in place
/// - `hidden`: F32 `[rows, rank]`
/// - `b`:      F32 `[out_dim, rank]`
///
/// `scale` is the LoRA alpha/rank scaling factor.
#[allow(clippy::too_many_arguments)]
pub fn lora_add_inplace_f32_kt(
    base: &KtTensor,
    hidden: &KtTensor,
    b: &KtTensor,
    scale: f32,
) -> Result<(), RmsNormError> {
    let base_shape = base.shape();
    if base_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: base must be [rows, out_dim], got {base_shape:?}"
        )));
    }
    let (rows, out_dim) = (base_shape[0], base_shape[1]);
    let h_shape = hidden.shape();
    if h_shape.len() != 2 || h_shape[0] != rows {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: hidden {h_shape:?} != [{rows}, rank]"
        )));
    }
    let rank = h_shape[1];
    if b.shape() != [out_dim, rank] {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: b {:?} != [{out_dim}, {rank}]",
            b.shape()
        )));
    }
    if rows > i32::MAX as usize || out_dim > i32::MAX as usize || rank > i32::MAX as usize {
        return Err(RmsNormError::Msg(
            "kt-lora-add: dimensions exceed i32 kernel envelope".to_string(),
        ));
    }
    if rows == 0 || out_dim == 0 || rank == 0 {
        return Ok(());
    }

    // In-place op: `base` is mutated through its CUDA storage.
    let base_ptr = kiln_kt_bridge::device_input_ptr(base, KtDType::F32, "base")?;
    let h_ptr = kiln_kt_bridge::device_input_ptr(hidden, KtDType::F32, "hidden")?;
    let b_ptr = kiln_kt_bridge::device_input_ptr(b, KtDType::F32, "b")?;
    let base_st = base;

    let stream_submission = device_stream_submission(base_st, "base_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_lora_add_inplace_f32(
            base_ptr as *mut f32,
            h_ptr as *const f32,
            b_ptr as *const f32,
            scale,
            rows as i32,
            out_dim as i32,
            rank as i32,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(())
}

/// Apply SiLU in place to `input_out`, simultaneously writing
/// `sigmoid_out = sigmoid(input)` into a separate buffer (saved for
/// the backward pass). Both buffers are F32 with the same element
/// count.
pub fn silu_inplace_save_sigmoid_f32_kt(
    input_out: &KtTensor,
    sigmoid_out: &KtTensor,
) -> Result<(), RmsNormError> {
    if input_out.shape() != sigmoid_out.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-silu-save: input {:?} != sigmoid {:?}",
            input_out.shape(),
            sigmoid_out.shape(),
        )));
    }
    let elem_count = input_out.element_count();
    if elem_count > i64::MAX as usize {
        return Err(RmsNormError::Msg(
            "kt-silu-save: element count exceeds i64 kernel envelope".to_string(),
        ));
    }
    if elem_count == 0 {
        return Ok(());
    }
    let elems = elem_count as i64;
    // Both buffers are mutated in place through their CUDA storage.
    let i_ptr = kiln_kt_bridge::device_input_ptr(input_out, KtDType::F32, "input_out")?;
    let s_ptr = kiln_kt_bridge::device_input_ptr(sigmoid_out, KtDType::F32, "sigmoid_out")?;
    let i_st = input_out;

    let stream_submission = device_stream_submission(i_st, "i_st")?;
    let raw_stream = stream_submission.raw_stream();

    let status = unsafe {
        kiln_silu_inplace_save_sigmoid_f32(i_ptr as *mut f32, s_ptr as *mut f32, elems, raw_stream)
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(RmsNormError::Msg(format!(
            "kt-silu-save: FFI returned {status}"
        )));
    }
    stream_submission.complete();
    Ok(())
}

// ============================================================================
// kt-typed `supports_*` predicates (#1082 Tier 1).
//
// Mirror the candle-typed `crate::supports_*` predicates from `lib.rs`,
// but inspect `KtTensor` operands directly. Lets forward.rs gate on
// kernel applicability without first bridging to candle (the bridge is
// cheap but does an Arc clone — these predicates allow a pure-kt fast
// path).
//
// Same semantics as the candle twins: returns `true` iff the underlying
// kernel can run on the given operands. Returns `false` (not an Err) for
// any rejection — callers fall through to the candle/composite path.
// ============================================================================

/// True when `t` lives on a GPU backend this crate's fused kernels run on.
/// Phase R.7: accepts `Device::Rocm` (under the `rocm` feature) in addition to
/// `Device::Cuda`, so the `supports_*` predicates gate the kernels in on both
/// backends. Kept named `kt_is_cuda` to avoid churning the many call sites; the
/// CUDA-only behavior is unchanged when `rocm` is off.
fn kt_is_cuda(t: &KtTensor) -> bool {
    match t.device() {
        KtDevice::Cuda(_) => true,
        #[cfg(feature = "rocm")]
        KtDevice::Rocm(_) => true,
        _ => false,
    }
}

/// kt twin of [`crate::supports`].
pub fn supports_rmsnorm_kt(x: &KtTensor, weight: &KtTensor) -> bool {
    kt_is_cuda(x)
        && kt_is_cuda(weight)
        && x.device() == weight.device()
        && x.dtype() == KtDType::BF16
        && weight.dtype() == KtDType::BF16
        && x.is_contiguous()
        && weight.is_contiguous()
        && x.rank() >= 1
        && (1..=8192).contains(&x.shape().last().copied().unwrap_or(0))
        && weight.shape() == [x.shape().last().copied().unwrap_or(0)]
}

/// kt twin of [`crate::supports_mlp_silu_mul`].
pub fn supports_mlp_silu_mul_kt(gate: &KtTensor, up: &KtTensor) -> bool {
    kt_is_cuda(gate)
        && kt_is_cuda(up)
        && gate.dtype() == KtDType::BF16
        && up.dtype() == KtDType::BF16
        && gate.is_contiguous()
        && up.is_contiguous()
        && gate.shape() == up.shape()
        && gate.element_count() <= i64::MAX as usize
}

/// kt twin of [`crate::supports_mlp_silu_mul_packed`].
pub fn supports_mlp_silu_mul_packed_kt(gate_up_packed: &KtTensor, cols: usize) -> bool {
    let dims = gate_up_packed.shape();
    kt_is_cuda(gate_up_packed)
        && gate_up_packed.dtype() == KtDType::BF16
        && gate_up_packed.is_contiguous()
        && !dims.is_empty()
        && dims[dims.len() - 1] == cols * 2
        && cols > 0
        && gate_up_packed.element_count() <= i64::MAX as usize
}

/// kt twin of [`crate::supports_sigmoid_mul`].
pub fn supports_sigmoid_mul_kt(x: &KtTensor, gate: &KtTensor) -> bool {
    kt_is_cuda(x)
        && kt_is_cuda(gate)
        && x.dtype() == KtDType::BF16
        && gate.dtype() == KtDType::BF16
        && x.is_contiguous()
        && gate.is_contiguous()
        && x.shape() == gate.shape()
        && x.element_count() <= i64::MAX as usize
}

/// kt twin of [`crate::supports_rotary_qk`].
///
/// Same shape + dtype rules as the candle predicate: q/k are
/// `[B, S, heads, head_dim]` BF16 on CUDA, contiguous, cos/sin are
/// `[S, rotary_dim/2]` F32 on CUDA, contiguous. `rotary_dim` must be
/// even and ≤ head_dim.
pub fn supports_rotary_qk_kt(
    q: &KtTensor,
    k: &KtTensor,
    cos: &KtTensor,
    sin: &KtTensor,
    head_dim: usize,
    rotary_dim: usize,
) -> bool {
    if !kt_is_cuda(q)
        || !kt_is_cuda(k)
        || !kt_is_cuda(cos)
        || !kt_is_cuda(sin)
        || q.dtype() != KtDType::BF16
        || k.dtype() != KtDType::BF16
        || cos.dtype() != KtDType::F32
        || sin.dtype() != KtDType::F32
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || q.rank() != 4
        || k.rank() != 4
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
    {
        return false;
    }
    let qd = q.shape();
    let kd = k.shape();
    let batch = qd[0];
    let seq_len = qd[1];
    qd[3] == head_dim
        && kd[0] == batch
        && kd[1] == seq_len
        && kd[3] == head_dim
        && cos.shape() == [seq_len, rotary_dim / 2]
        && sin.shape() == [seq_len, rotary_dim / 2]
        && batch <= i32::MAX as usize
        && seq_len <= i32::MAX as usize
        && qd[2] <= i32::MAX as usize
        && kd[2] <= i32::MAX as usize
        && head_dim <= i32::MAX as usize
        && rotary_dim <= i32::MAX as usize
}

/// kt twin of [`crate::supports_attn_decode_qkv_prep`].
///
/// Inspects the same shape/dtype/contig/device invariants as the candle
/// predicate. Used by the decode-time fused QKV prep path.
#[allow(clippy::too_many_arguments)]
pub fn supports_attn_decode_qkv_prep_kt(
    q_raw: &KtTensor,
    k_raw: &KtTensor,
    q_weight: &KtTensor,
    k_weight: &KtTensor,
    cos: &KtTensor,
    sin: &KtTensor,
    q_heads: usize,
    k_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    has_gate: bool,
) -> bool {
    if !kt_is_cuda(q_raw)
        || !kt_is_cuda(k_raw)
        || !kt_is_cuda(q_weight)
        || !kt_is_cuda(k_weight)
        || !kt_is_cuda(cos)
        || !kt_is_cuda(sin)
        || q_raw.dtype() != KtDType::BF16
        || k_raw.dtype() != KtDType::BF16
        || q_weight.dtype() != KtDType::BF16
        || k_weight.dtype() != KtDType::BF16
        || cos.dtype() != KtDType::F32
        || sin.dtype() != KtDType::F32
        || !q_raw.is_contiguous()
        || !k_raw.is_contiguous()
        || !q_weight.is_contiguous()
        || !k_weight.is_contiguous()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || q_raw.rank() != 3
        || k_raw.rank() != 3
        || q_heads == 0
        || k_heads == 0
        || head_dim == 0
        || head_dim > 8192
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
    {
        return false;
    }

    let qd = q_raw.shape();
    let kd = k_raw.shape();
    let batch = qd[0];
    let Some(q_base) = q_heads.checked_mul(head_dim) else {
        return false;
    };
    let Some(q_inner) = (if has_gate {
        q_base.checked_mul(2)
    } else {
        Some(q_base)
    }) else {
        return false;
    };
    let Some(k_inner) = k_heads.checked_mul(head_dim) else {
        return false;
    };
    let Some(total_heads) = q_heads.checked_add(k_heads) else {
        return false;
    };
    let Some(total_rows) = batch.checked_mul(total_heads) else {
        return false;
    };
    qd[1] == 1
        && kd[0] == batch
        && kd[1] == 1
        && qd[2] == q_inner
        && kd[2] == k_inner
        && q_weight.shape() == [head_dim]
        && k_weight.shape() == [head_dim]
        && cos.shape() == [1, rotary_dim / 2]
        && sin.shape() == [1, rotary_dim / 2]
        && batch <= i32::MAX as usize
        && total_rows <= i32::MAX as usize
        && q_heads <= i32::MAX as usize
        && k_heads <= i32::MAX as usize
        && head_dim <= i32::MAX as usize
        && rotary_dim <= i32::MAX as usize
}

/// kt twin of [`crate::supports_l2_qk_norm`].
pub fn supports_l2_qk_norm_kt(q: &KtTensor, k: &KtTensor) -> bool {
    kt_is_cuda(q)
        && kt_is_cuda(k)
        && q.dtype() == KtDType::BF16
        && k.dtype() == KtDType::BF16
        && q.is_contiguous()
        && k.is_contiguous()
        && q.shape() == k.shape()
        && q.rank() >= 1
        && q.shape().last().copied().unwrap_or(0) <= 8192
}

/// kt twin of [`crate::supports_l2_qk_norm_gqa`].
pub fn supports_l2_qk_norm_gqa_kt(q: &KtTensor, k: &KtTensor, nv: usize) -> bool {
    if !kt_is_cuda(q)
        || !kt_is_cuda(k)
        || q.dtype() != KtDType::BF16
        || k.dtype() != KtDType::BF16
        || q.shape() != k.shape()
        || q.rank() != 4
    {
        return false;
    }
    let dims = q.shape();
    let nk = dims[2];
    let dk = dims[3];
    nk > 0 && dk == 128 && nv >= nk && nv.is_multiple_of(nk)
}

/// kt twin of [`crate::supports_lora_decode_add`].
pub fn supports_lora_decode_add_kt(
    base: &KtTensor,
    x: &KtTensor,
    a: &KtTensor,
    b: &KtTensor,
) -> bool {
    if base.rank() != 3 || x.rank() != 3 || a.rank() != 2 || b.rank() != 2 {
        return false;
    }
    let bd = base.shape();
    let xd = x.shape();
    let ad = a.shape();
    let bw = b.shape();
    let (batch, one, out_dim) = (bd[0], bd[1], bd[2]);
    let (x_batch, x_one, in_dim) = (xd[0], xd[1], xd[2]);
    let (rank, a_in_dim) = (ad[0], ad[1]);
    let (b_out_dim, b_rank) = (bw[0], bw[1]);

    kt_is_cuda(base)
        && kt_is_cuda(x)
        && kt_is_cuda(a)
        && kt_is_cuda(b)
        && base.dtype() == KtDType::BF16
        && x.dtype() == KtDType::BF16
        && a.dtype() == KtDType::BF16
        && b.dtype() == KtDType::BF16
        && base.is_contiguous()
        && x.is_contiguous()
        && a.is_contiguous()
        && b.is_contiguous()
        && batch == x_batch
        && one == 1
        && x_one == 1
        && rank == b_rank
        && in_dim == a_in_dim
        && out_dim == b_out_dim
        && batch > 0
        && in_dim > 0
        && out_dim > 0
        && rank > 0
        && rank <= 64
        && batch <= i32::MAX as usize
        && in_dim <= i32::MAX as usize
        && out_dim <= i32::MAX as usize
        && rank <= i32::MAX as usize
}

/// kt twin of [`crate::supports_optimizer_step`]. Inspects a slice of
/// kt tensors against the same per-tensor invariants the candle
/// predicate uses.
pub fn supports_optimizer_step_kt(tensors: &[&KtTensor]) -> bool {
    let Some(first) = tensors.first() else {
        return false;
    };
    kt_is_cuda(first)
        && matches!(first.dtype(), KtDType::F32 | KtDType::BF16)
        && first.is_contiguous()
        && tensors.iter().all(|t| {
            kt_is_cuda(t)
                && t.dtype() == first.dtype()
                && t.element_count() == first.element_count()
                && t.is_contiguous()
        })
}

// Note: the candle-vs-kt regression test modules (kt_rotary_qk_regression,
// kt_l2_qk_norm_gqa_regression, kt_lora_decode_regression,
// kt_optimizer_step_regression, and kt_silu_save_sigmoid_regression) were
// deleted in (#1082) when their candle-typed parity oracles were removed
// from the public surface. The kt-typed entries are still covered by the
// `tests/kt_v2_smoke.rs` integration test, the in-crate CustomOp{1,2,3}
// implementations backing them in `kt_forward_op.rs`, and the production
// dispatch paths in `kiln-model::forward.rs` / `cuda_train.rs`.
