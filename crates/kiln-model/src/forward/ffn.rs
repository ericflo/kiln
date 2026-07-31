use super::*;

/// SwiGLU feed-forward network.
///
/// Computes: down_proj @ (silu(gate_proj @ x) * (up_proj @ x))
///
/// `x`: [batch, seq_len, hidden_size]
/// `mlp`: MLP weight bundle, including optional Marlin W4A16-packed projections.
///
/// Dispatch each projection through the Marlin W4A16 path when the matching
/// `*_marlin` field is `Some`, else the existing BF16 `broadcast_matmul(*_t)`
/// path. LoRA deltas are always added on top so behaviour matches
/// `linear_with_lora_t` in the absence of Marlin weights. Mirrors
/// `q_proj_forward`'s Marlin routing from PR #149.
///
/// Returns: [batch, seq_len, hidden_size]
pub fn swiglu_ffn(
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    swiglu_ffn_impl(None, x, mlp, lora, false)
}

/// SwiGLU gate/up half used by exact training-time split backprop.
pub fn swiglu_ffn_gated_hidden(
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let gate = {
        kiln_nvtx::range!(c"kiln/mlp/gate");
        mlp_proj_forward_decode_if(
            None,
            false,
            x,
            &mlp.gate_proj_t,
            mlp.gate_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.gate_proj.as_ref()),
            lora_scale,
        )?
    };
    let up = {
        kiln_nvtx::range!(c"kiln/mlp/up");
        mlp_proj_forward_decode_if(
            None,
            false,
            x,
            &mlp.up_proj_t,
            mlp.up_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.up_proj.as_ref()),
            lora_scale,
        )?
    };
    synchronize_tensor_ready_for_model_handoff("mlp gated-hidden gate projection", &gate)?;
    synchronize_tensor_ready_for_model_handoff("mlp gated-hidden up projection", &up)?;
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        let hidden = crate::tape_forward::try_tape_swiglu_kt(&gate, &up)
            .context("swiglu_ffn_gated_hidden try_tape_swiglu_kt")?
            .context("active tape scope failed to record SwiGLU")?;
        synchronize_tensor_ready_for_model_handoff("mlp gated-hidden tape silu*mul", &hidden)?;
        return Ok(hidden);
    }
    #[cfg(feature = "metal")]
    {
        if !crate::tape_forward::tape_scope_active()
            && crate::backend::metal::metal_mlp_silu_mul_supports(&gate, &up)
        {
            let hidden = crate::backend::metal::metal_mlp_silu_mul_bf16(&gate, &up)
                .context("metal mlp silu*mul kernel failed")?;
            synchronize_tensor_ready_for_model_handoff("mlp gated-hidden metal silu*mul", &hidden)?;
            return Ok(hidden);
        }
    }
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        if !crate::tape_forward::tape_scope_active()
            && !gpu_fused_mlp_silu_mul_disabled(gate.device())
            && !gate.track_op()
            && !up.track_op()
        {
            if let (Some(gate_kt), Some(up_kt)) =
                (try_borrow_kt_cuda(&gate), try_borrow_kt_cuda(&up))
            {
                if kiln_rmsnorm_kernel::supports_mlp_silu_mul_kt(&gate_kt, &up_kt) {
                    // Phase 7 (#1082): kt-only. Same FFI symbol as the
                    // candle path. Result is kt — return it directly.
                    let out_kt = kiln_rmsnorm_kernel::fused_mlp_silu_mul_kt(&gate_kt, &up_kt)
                        .map_err(|e| anyhow::anyhow!("kt fused_mlp_silu_mul: {e}"))?;
                    synchronize_tensor_ready_for_model_handoff(
                        "mlp gated-hidden fused silu*mul",
                        &out_kt,
                    )?;
                    return Ok(out_kt);
                }
            }
        }
    }
    let gate = cuda_silu(&gate)?;
    synchronize_tensor_ready_for_model_handoff("mlp gated-hidden silu gate", &gate)?;
    let hidden = (gate * up)?;
    synchronize_tensor_ready_for_model_handoff("mlp gated-hidden hidden", &hidden)?;
    Ok(hidden)
}

/// SwiGLU down projection half used by exact training-time split backprop.
pub fn swiglu_ffn_down_from_gated(
    gated: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    synchronize_tensor_ready_for_model_handoff("mlp down-from-gated input", gated)?;
    mlp_proj_forward_decode_if(
        None,
        false,
        gated,
        &mlp.down_proj_t,
        mlp.down_proj_marlin.as_ref(),
        lora_layer.and_then(|l| l.down_proj.as_ref()),
        lora_scale,
    )
}

/// Transformer MLP gate/up half from a post-attention residual state.
pub fn transformer_mlp_gated_hidden(
    hidden: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let normed_post = {
        kiln_nvtx::range!(c"kiln/norm/pre_mlp");
        rms_norm(hidden, &layer.post_attention_layernorm, config.rms_norm_eps)?
    };
    swiglu_ffn_gated_hidden(&normed_post, &layer.mlp, lora)
}

/// Transformer MLP down half from a precomputed SwiGLU gated hidden.
pub fn transformer_mlp_down_from_gated(
    gated: &Tensor,
    layer: &GpuLayerWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    swiglu_ffn_down_from_gated(gated, &layer.mlp, lora)
}

pub(super) fn swiglu_ffn_metal_decode(
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    swiglu_ffn_impl(None, x, mlp, lora, true)
}

pub(super) fn swiglu_ffn_backend_profiled(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
    use_metal_decode_gemv: bool,
) -> Result<Tensor> {
    swiglu_ffn_impl(Some(backend), x, mlp, lora, use_metal_decode_gemv)
}

pub(super) fn swiglu_ffn_impl(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
    use_metal_decode_gemv: bool,
) -> Result<Tensor> {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    let (_, seq_len, _) = x.dims3()?;
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        let chunk_tokens = gpu_training_mlp_chunk_tokens(&x.device());
        if !gpu_training_mlp_chunking_disabled(&x.device())
            && chunk_tokens > 0
            && seq_len > chunk_tokens
            && (x.track_op() || lora.is_some())
            && matches!(x.device(), Device::Cuda(_) | Device::Rocm(_))
        {
            return swiglu_ffn_impl_chunked(
                backend,
                x,
                mlp,
                lora,
                use_metal_decode_gemv,
                chunk_tokens,
            );
        }
    }

    swiglu_ffn_impl_no_chunk(backend, x, mlp, lora, use_metal_decode_gemv)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(super) fn gpu_training_mlp_chunking_disabled(device: &Device) -> bool {
    if matches!(device, Device::Rocm(_)) {
        return !crate::rocm_policy::current_rocm_kernel_policy().training_mlp_chunking;
    }
    false
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(super) fn gpu_training_mlp_chunk_tokens(device: &Device) -> usize {
    match device {
        Device::Rocm(_) => rocm_training_mlp_chunk_tokens(),
        _ => 1024,
    }
}

#[cfg(feature = "rocm")]
pub(super) fn rocm_training_mlp_chunk_tokens() -> usize {
    crate::rocm_policy::current_rocm_kernel_policy().training_mlp_chunk_tokens
}

#[cfg(all(any(feature = "cuda", feature = "rocm"), not(feature = "rocm")))]
pub(super) fn rocm_training_mlp_chunk_tokens() -> usize {
    1024
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(super) fn swiglu_ffn_impl_chunked(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
    use_metal_decode_gemv: bool,
    chunk_tokens: usize,
) -> Result<Tensor> {
    let (_, seq_len, _) = x.dims3()?;
    let mut outputs = Vec::with_capacity(seq_len.div_ceil(chunk_tokens));
    let mut start = 0usize;
    while start < seq_len {
        let len = (seq_len - start).min(chunk_tokens);
        let x_chunk = x.narrow(1, start, len).with_context(|| {
            format!(
                "chunked CUDA training MLP input tile [{start}, {})",
                start + len
            )
        })?;
        let x_chunk = if crate::tape_forward::tape_scope_active() {
            require_active_tape_output(
                crate::tape_forward::try_tape_narrow_kt(x, 1, start, len, &x_chunk).with_context(
                    || {
                        format!(
                            "chunked GPU training MLP input tile [{start}, {}) tape narrow",
                            start + len
                        )
                    },
                )?,
                "chunked GPU training MLP input narrow",
            )?
        } else {
            x_chunk
        };
        let out = swiglu_ffn_impl_no_chunk(backend, &x_chunk, mlp, lora, use_metal_decode_gemv)
            .with_context(|| {
                format!("chunked CUDA training MLP tile [{start}, {})", start + len)
            })?;
        synchronize_tensor_ready_for_model_handoff(
            &format!("chunked GPU training MLP tile [{start}, {})", start + len),
            &out,
        )?;
        outputs.push(out);
        start += len;
    }
    let output_refs: Vec<&Tensor> = outputs.iter().collect();
    // Exact row tiling: the MLP is tokenwise after the post-attention norm, so
    // splitting along seq and concatenating the outputs preserves bitwise math
    // modulo backend GEMM ordering. CUDA can use its kt concat fast path; ROCm
    // falls through to Tensor::cat, which dispatches to rocm_concat.
    #[cfg(feature = "cuda")]
    let out = if let Some(out) = try_kt_cat_dim1(&output_refs)? {
        out
    } else {
        Tensor::cat(&output_refs, 1).context("chunked GPU training MLP cat")?
    };
    #[cfg(not(feature = "cuda"))]
    let out = Tensor::cat(&output_refs, 1).context("chunked GPU training MLP cat")?;
    synchronize_tensor_ready_for_model_handoff("chunked GPU training MLP cat", &out)?;
    if crate::tape_forward::tape_scope_active() {
        return require_active_tape_output(
            crate::tape_forward::try_tape_concat_kt(&output_refs, 1, &out)
                .context("chunked GPU training MLP output tape concat")?,
            "chunked GPU training MLP output concatenation",
        );
    }
    Ok(out)
}

/// Legacy two-matmul split of x against gate_proj_t and up_proj_t with the
/// existing NVTX ranges + sub-op profiles preserved. The CUDA prefill path now
/// prefers the cached `gate_up_proj_t` fused GEMM; this helper is the
/// fallback used when LoRA, Marlin, or backend constraints force the
/// per-projection codepath.
#[allow(clippy::too_many_arguments)]
pub(super) fn swiglu_ffn_split_gate_up(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora_layer: Option<&LoraLayerWeights>,
    lora_scale: f32,
    use_metal_decode_gemv: bool,
) -> Result<(Tensor, Tensor)> {
    // x @ gate_proj_t -> [batch, seq_len, intermediate_size]
    let gate = {
        kiln_nvtx::range!(c"kiln/mlp/gate");
        mlp_proj_forward_decode_if(
            backend,
            use_metal_decode_gemv,
            x,
            &mlp.gate_proj_t,
            mlp.gate_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.gate_proj.as_ref()),
            lora_scale,
        )?
    };
    // x @ up_proj_t -> [batch, seq_len, intermediate_size]
    let up = {
        kiln_nvtx::range!(c"kiln/mlp/up");
        mlp_proj_forward_decode_if(
            backend,
            use_metal_decode_gemv,
            x,
            &mlp.up_proj_t,
            mlp.up_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.up_proj.as_ref()),
            lora_scale,
        )?
    };
    synchronize_tensor_ready_for_model_handoff("mlp gate projection", &gate)?;
    synchronize_tensor_ready_for_model_handoff("mlp up projection", &up)?;
    Ok((gate, up))
}

/// kt-native SwiGLU FFN core (#1082, MLP/FFN region migration —
/// region 2). Computes `down @ (silu(gate @ x) * (up @ x))` entirely
/// over `KtTensor` storage so the gate/up matmul outputs and the
/// silu*mul result never round-trip through candle. This is the
/// consolidated kt-internal computation for the MLP region; the
/// candle↔kt bridging (and all the dispatch-eligibility preconditions)
/// live in the [`try_kt_swiglu_ffn`] wrapper.
///
/// Inputs are all kt 2D views: `x2d` is `[lead, hidden]`, `gate_t` /
/// `up_t` are `[hidden, intermediate]`, `down_t` is
/// `[intermediate, hidden]`. Returns the `[lead, hidden]` FFN output as
/// a `KtTensor`; the wrapper reshapes it back to the input rank.
///
/// Bit-exact to the existing per-op kt path:
/// - each projection is `kiln_tensor::ops::matmul`, so the kt MatmulOp
///   contract dispatches through the active backend's native path;
/// - the SwiGLU activation prefers `fused_mlp_silu_mul_kt` (the exact
///   `kiln_fused_mlp_silu_mul_bf16` FFI symbol the existing fused fast
///   path uses) when the BF16 fused kernel supports the operands, and
///   otherwise falls back to `kiln_tensor::ops::mul_sigmoid_gate`,
///   whose CUDA fast path composes the same
///   `cuda_activation_unary(kind=0)` + `cuda_elementwise_binary(kind=2)`
///   substrate kernels.
///
/// The `kiln/swiglu_ffn_kt` NVTX range is opened by the
/// [`try_kt_swiglu_ffn`] wrapper, so this core does not open its own to
/// avoid a nested duplicate. The inner `kiln/mlp/{gate,up,down}` ranges
/// are preserved so nsys per-stage attribution is unchanged.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(super) fn gpu_matmul_for_swiglu(lhs: &KtTensor, rhs: &KtTensor) -> Result<KtTensor> {
    kiln_tensor::ops::matmul(lhs, rhs)
        .map_err(|e| anyhow::anyhow!("gpu_matmul_for_swiglu: matmul contract: {e}"))
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(super) fn kt_swiglu_ffn_native(
    x2d: &KtTensor,
    gate_t: &KtTensor,
    up_t: &KtTensor,
    down_t: &KtTensor,
) -> Result<KtTensor> {
    let gate = {
        kiln_nvtx::range!(c"kiln/mlp/gate");
        gpu_matmul_for_swiglu(x2d, gate_t)
            .map_err(|e| anyhow::anyhow!("kt_swiglu_ffn_native: gate matmul: {e}"))?
    };
    let up = {
        kiln_nvtx::range!(c"kiln/mlp/up");
        gpu_matmul_for_swiglu(x2d, up_t)
            .map_err(|e| anyhow::anyhow!("kt_swiglu_ffn_native: up matmul: {e}"))?
    };
    synchronize_tensor_ready_for_model_handoff("kt swiglu gate projection", &gate)?;
    synchronize_tensor_ready_for_model_handoff("kt swiglu up projection", &up)?;
    let hidden = {
        kiln_nvtx::range!(c"kiln/mlp/gate_silu_hidden_mul");
        if kiln_rmsnorm_kernel::supports_mlp_silu_mul_kt(&gate, &up) {
            // Same fused BF16 kernel (`kiln_fused_mlp_silu_mul_bf16`) the
            // existing inline fast path uses — bit-exact.
            kiln_rmsnorm_kernel::fused_mlp_silu_mul_kt(&gate, &up)
                .map_err(|e| anyhow::anyhow!("kt_swiglu_ffn_native: fused silu*mul: {e}"))?
        } else {
            // Non-BF16 (F16/F32) operands: the substrate composite
            // (silu via cuda_activation_unary(0), then mul via
            // cuda_elementwise_binary(2)) — identical to the candle
            // fallback's per-op kt route.
            kiln_tensor::ops::mul_sigmoid_gate(&gate, &up)
                .map_err(|e| anyhow::anyhow!("kt_swiglu_ffn_native: mul_sigmoid_gate: {e}"))?
        }
    };
    synchronize_tensor_ready_for_model_handoff("kt swiglu hidden", &hidden)?;
    let out = {
        kiln_nvtx::range!(c"kiln/mlp/down");
        gpu_matmul_for_swiglu(&hidden, down_t)
            .map_err(|e| anyhow::anyhow!("kt_swiglu_ffn_native: down matmul: {e}"))?
    };
    Ok(out)
}

/// Phase 7 default-on (#1082) — consolidated kt-API SwiGLU FFN
/// migration wrapper (region 2). Routes the whole
/// `down @ (silu(gate @ x) * (up @ x))` MLP region through the
/// kt-native [`kt_swiglu_ffn_native`] core, keeping intermediates as
/// `KtTensor` storage (one candle→kt borrow at the input, one kt→candle
/// copy at the output) instead of bridging at every individual op.
///
/// Flattens the leading dims of `x` to a 2D `[lead, hidden]` view
/// before dispatch (mirroring [`matmul_no_broadcast_copy`] /
/// [`try_kt_lm_head`]) and reshapes the result back to the input rank.
///
/// Returns `Ok(None)` — falling through to the existing
/// projection-by-projection candle path — on any of:
/// - `accelerator.kt_api_mode = "disabled"`;
/// - non-CUDA device, or a non-{BF16,F16,F32} / mixed dtype;
/// - non-contiguous `x` or weights, or weight rank ≠ 2, or a K-dim
///   mismatch between `x` and the projections;
/// - autograd-tracked `x` or an active tape recording scope (those must
///   keep flowing through the candle / tape-wired silu+mul + LoRA path
///   so adapter grads are not severed);
/// - any kt borrow / matmul failure (the candle path then runs).
///
/// LoRA and Marlin eligibility is checked by the *caller* (the
/// `!has_mlp_lora && !has_marlin` guard in [`swiglu_ffn_impl_no_chunk`])
/// before this is invoked, because those need the standalone candle
/// projections + delta application.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(super) fn try_kt_swiglu_ffn(x: &Tensor, mlp: &GpuFfnWeights) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    // The tape-wired path (training) must run through the recorded silu+mul so
    // the gate/up projections' grads chain.
    if x.track_op() || crate::tape_forward::tape_scope_active() {
        return Ok(None);
    }
    let x_dims = x.dims();
    if x_dims.len() < 2 {
        return Ok(None);
    }
    let g_dims = mlp.gate_proj_t.dims();
    let u_dims = mlp.up_proj_t.dims();
    let d_dims = mlp.down_proj_t.dims();
    if g_dims.len() != 2 || u_dims.len() != 2 || d_dims.len() != 2 {
        return Ok(None);
    }
    let hidden = x_dims[x_dims.len() - 1];
    let intermediate = g_dims[1];
    // x:[*, hidden] @ gate_t:[hidden, intermediate]; same for up.
    // hidden_act:[*, intermediate] @ down_t:[intermediate, hidden].
    if g_dims[0] != hidden
        || u_dims[0] != hidden
        || u_dims[1] != intermediate
        || d_dims[0] != intermediate
        || d_dims[1] != hidden
    {
        return Ok(None);
    }
    let dtype = x.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || mlp.gate_proj_t.dtype() != dtype
        || mlp.up_proj_t.dtype() != dtype
        || mlp.down_proj_t.dtype() != dtype
        || !cuda_or_rocm_device(x.device())
        || !cuda_or_rocm_device(mlp.gate_proj_t.device())
        || !cuda_or_rocm_device(mlp.up_proj_t.device())
        || !cuda_or_rocm_device(mlp.down_proj_t.device())
        || !x.is_contiguous()
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/swiglu_ffn_kt");

    let lead: usize = x_dims[..x_dims.len() - 1].iter().product();
    let x2d = match x.reshape((lead, hidden)) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    // #1082 forward-flip: `x2d` is already kt; use it directly (no candle
    // borrow). Weight accessors are kt-native.
    let gate_t_kt = match mlp.gate_proj_t_kt() {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let up_t_kt = match mlp.up_proj_t_kt() {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let down_t_kt = match mlp.down_proj_t_kt() {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    // kt-internal computation: gate/up matmuls → silu*mul → down matmul.
    let out_kt = match kt_swiglu_ffn_native(&x2d, &gate_t_kt, &up_t_kt, &down_t_kt) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    // Result is kt — restore input rank and return directly (no copy-out).
    let mut out_shape: Vec<usize> = x_dims[..x_dims.len() - 1].to_vec();
    out_shape.push(hidden);
    let out = out_kt
        .reshape(out_shape)
        .context("try_kt_swiglu_ffn: output reshape")?;
    Ok(Some(out))
}

pub(super) fn swiglu_ffn_impl_no_chunk(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
    use_metal_decode_gemv: bool,
) -> Result<Tensor> {
    let (_, _seq_len, _) = x.dims3()?;
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    let tape_scope_active = crate::tape_forward::tape_scope_active();
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )))]
    let tape_scope_active = false;
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let has_mlp_lora = lora_layer.is_some_and(LoraLayerWeights::has_mlp);
    let has_mlp_gate_up_lora = lora_layer.is_some_and(LoraLayerWeights::has_mlp_gate_up);
    let has_marlin = mlp.gate_proj_marlin.is_some()
        || mlp.up_proj_marlin.is_some()
        || mlp.down_proj_marlin.is_some();
    #[cfg(feature = "rocm")]
    if !tape_scope_active
        && !has_mlp_lora
        && !has_marlin
        && _seq_len == 1
        && x.dtype() == DType::BF16
        && !x.track_op()
        && let (Some(gate_up_w8), Some(down_w8)) =
            (mlp.gate_up_proj_w8.as_ref(), mlp.down_proj_w8.as_ref())
    {
        if gate_up_w8.n % 2 == 0 {
            let g_dim = gate_up_w8.n / 2;
            if crate::rocm_w8_proj::swiglu_bf16_enabled(gate_up_w8) {
                if let Some(hidden) = {
                    kiln_nvtx::range!(c"kiln/mlp/gate_up_swiglu_w8");
                    crate::rocm_w8_proj::swiglu_bf16(x, gate_up_w8)?
                } {
                    synchronize_tensor_ready_for_model_handoff(
                        "mlp gate_up_swiglu_w8 hidden",
                        &hidden,
                    )?;
                    let out = {
                        kiln_nvtx::range!(c"kiln/mlp/down_w8");
                        crate::rocm_w8_proj::matmul_bf16(&hidden, down_w8)?
                    };
                    return Ok(out);
                }
            }
            let gate_up = {
                kiln_nvtx::range!(c"kiln/mlp/gate_up_w8");
                crate::rocm_w8_proj::matmul_bf16(x, gate_up_w8)?
            };
            synchronize_tensor_ready_for_model_handoff("mlp gate_up_w8", &gate_up)?;
            if let Some(gate_up_kt) = try_borrow_kt_cuda(&gate_up)
                .filter(|t| kiln_rmsnorm_kernel::supports_mlp_silu_mul_packed_kt(t, g_dim))
            {
                let hidden = {
                    kiln_nvtx::range!(c"kiln/mlp/gate_silu_hidden_mul_packed");
                    kiln_rmsnorm_kernel::fused_mlp_silu_mul_packed_kt(&gate_up_kt, g_dim)
                        .map_err(|e| anyhow::anyhow!("kt fused_mlp_silu_mul_packed: {e}"))?
                };
                synchronize_tensor_ready_for_model_handoff(
                    "mlp gate_silu_hidden_mul_packed",
                    &hidden,
                )?;
                let out = {
                    kiln_nvtx::range!(c"kiln/mlp/down_w8");
                    crate::rocm_w8_proj::matmul_bf16(&hidden, down_w8)?
                };
                return Ok(out);
            }
        }
    }
    if !tape_scope_active && !has_mlp_lora && !has_marlin {
        if let Some(backend) = backend {
            if let Some(out) = LinearBackend::runtime_mlp_decode(
                backend,
                x,
                &mlp.gate_proj_t,
                &mlp.up_proj_t,
                &mlp.down_proj_t,
            )? {
                return Ok(out);
            }
        }
    }
    if !tape_scope_active && !has_mlp_gate_up_lora && !has_marlin {
        if let Some(backend) = backend {
            if let Some(hidden) = LinearBackend::runtime_mlp_gate_up_decode(
                backend,
                x,
                &mlp.gate_proj_t,
                &mlp.up_proj_t,
            )? {
                synchronize_tensor_ready_for_model_handoff("mlp gate_up_fused hidden", &hidden)?;
                let out = {
                    kiln_nvtx::range!(c"kiln/mlp/down");
                    mlp_proj_forward_decode_if(
                        Some(backend),
                        use_metal_decode_gemv,
                        &hidden,
                        &mlp.down_proj_t,
                        mlp.down_proj_marlin.as_ref(),
                        lora_layer.and_then(|l| l.down_proj.as_ref()),
                        lora_scale,
                    )?
                };
                return Ok(out);
            }
        }
    }
    #[cfg(feature = "metal")]
    if !tape_scope_active {
        if let Some(hidden) = try_metal_mlp_gate_up_hidden(x, mlp, lora_layer)? {
            synchronize_tensor_ready_for_model_handoff("mlp metal gate_up_fused hidden", &hidden)?;
            let out = {
                kiln_nvtx::range!(c"kiln/mlp/down");
                mlp_proj_forward_decode_if(
                    backend,
                    use_metal_decode_gemv,
                    &hidden,
                    &mlp.down_proj_t,
                    mlp.down_proj_marlin.as_ref(),
                    lora_layer.and_then(|l| l.down_proj.as_ref()),
                    lora_scale,
                )?
            };
            return Ok(out);
        }
    }

    // GPU prefill fast path: single [B*T, hidden] @ [hidden, 2*intermediate]
    // GEMM against the cached `gate_up_proj_t`, then either:
    //   (a) feed the packed result straight into the stride-aware fused
    //       silu*mul kernel and jump to the down projection (preferred —
    //       skips the per-half .contiguous() memcpys), or
    //   (b) fall back to splitting into gate/up halves for the legacy
    //       hidden builder when the packed silu*mul isn't available (e.g.
    //       metal builds, dtype mismatch, kill switch off).
    // Replaces two separate `[B*T, hidden] @ [hidden, intermediate]`
    // matmuls. Per the PR goal: per-layer gate+up was ~6.3 ms vs a ~2.5 ms
    // compute roof, because the launch / weight-stream cost was doubled
    // across the pair. Gated identically to the Marlin/LoRA-aware
    // branches above — those callers still need the standalone projections.
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        if !tape_scope_active
            && !has_mlp_gate_up_lora
            && !has_marlin
            && !gpu_fused_mlp_gate_up_prefill_disabled(x.device())
        {
            if let Some(gate_up_proj_t) = mlp.gate_up_proj_t.as_ref() {
                if x.dtype() == DType::BF16
                    && !x.track_op()
                    && cuda_or_rocm_device(x.device())
                    && gate_up_proj_t.dtype() == DType::BF16
                    && !gate_up_proj_t.track_op()
                    && cuda_or_rocm_device(gate_up_proj_t.device())
                    && gate_up_proj_t.is_contiguous()
                {
                    if let (Ok(g_dim), Ok(u_dim)) = (mlp.gate_proj_t.dim(1), mlp.up_proj_t.dim(1)) {
                        let gu_dims = gate_up_proj_t.dims();
                        if gu_dims.len() == 2 && gu_dims[1] == g_dim + u_dim && g_dim == u_dim {
                            let gate_up = {
                                kiln_nvtx::range!(c"kiln/mlp/gate_up_fused_prefill");
                                if let Some(backend) = backend {
                                    if let Some(out) =
                                        runtime_matmul_no_broadcast_copy(backend, x, gate_up_proj_t)
                                            .context("fused MLP gate+up runtime matmul request")?
                                    {
                                        out
                                    } else {
                                        broadcast_matmul_cpu_compatible(x, gate_up_proj_t)
                                            .context("cuda fused MLP gate+up prefill matmul")?
                                    }
                                } else {
                                    broadcast_matmul_cpu_compatible(x, gate_up_proj_t)
                                        .context("cuda fused MLP gate+up prefill matmul")?
                                }
                            };
                            synchronize_tensor_ready_for_model_handoff(
                                "mlp gate_up_fused_prefill",
                                &gate_up,
                            )?;
                            if let Some(gate_up_kt) = try_borrow_kt_cuda(&gate_up).filter(|t| {
                                kiln_rmsnorm_kernel::supports_mlp_silu_mul_packed_kt(t, g_dim)
                            }) {
                                let hidden = {
                                    kiln_nvtx::range!(c"kiln/mlp/gate_silu_hidden_mul_packed");
                                    // #1082: keep the silu*mul output as kt — the
                                    // down-proj path is kt now, so the candle copy-out
                                    // is gone.
                                    kiln_rmsnorm_kernel::fused_mlp_silu_mul_packed_kt(
                                        &gate_up_kt,
                                        g_dim,
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!("kt fused_mlp_silu_mul_packed: {e}")
                                    })?
                                };
                                synchronize_tensor_ready_for_model_handoff(
                                    "mlp gate_silu_hidden_mul_packed",
                                    &hidden,
                                )?;
                                let out = {
                                    kiln_nvtx::range!(c"kiln/mlp/down");
                                    mlp_proj_forward_decode_if(
                                        backend,
                                        use_metal_decode_gemv,
                                        &hidden,
                                        &mlp.down_proj_t,
                                        mlp.down_proj_marlin.as_ref(),
                                        lora_layer.and_then(|l| l.down_proj.as_ref()),
                                        lora_scale,
                                    )?
                                };
                                return Ok(out);
                            }
                        }
                    }
                }
            }
        }
    }
    // Consolidated kt-native SwiGLU FFN (#1082, region 2). When the
    // packed `gate_up_proj_t` prefill fast path above did not fire (no
    // cached fused transpose, or its silu*mul kernel declined) and there
    // is no MLP LoRA / Marlin, route the entire gate/up/down + silu*mul
    // region through the kt substrate keeping intermediates as kt storage
    // (one borrow-in, one copy-out) instead of the per-op candle↔kt
    // round-trips the legacy `swiglu_ffn_split_gate_up` path pays. Returns
    // `None` on any incompatibility (tape scope, tracked input, dtype, …)
    // so the legacy path below runs unchanged. Gated default-on with
    // escape hatch `accelerator.kt_api_mode = "disabled"`.
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        if !has_mlp_lora && !has_marlin {
            if let Some(out) = try_kt_swiglu_ffn(x, mlp)? {
                return Ok(out);
            }
        }
    }
    let (gate, up): (Tensor, Tensor) = swiglu_ffn_split_gate_up(
        backend,
        x,
        mlp,
        lora_layer,
        lora_scale,
        use_metal_decode_gemv,
    )?;
    let hidden = {
        #[cfg(feature = "metal")]
        {
            if tape_scope_active {
                let hidden = crate::tape_forward::try_tape_swiglu_kt(&gate, &up)
                    .context("swiglu_ffn_impl_no_chunk try_tape_swiglu_kt")?
                    .context("active tape scope failed to record SwiGLU")?;
                hidden
            } else if crate::backend::metal::metal_mlp_silu_mul_supports(&gate, &up) {
                let hidden = crate::backend::metal::metal_mlp_silu_mul_bf16(&gate, &up)
                    .context("metal mlp silu*mul kernel failed")?;
                hidden
            } else {
                let gate = cuda_silu(&gate)?;
                synchronize_tensor_ready_for_model_handoff("mlp gate_silu", &gate)?;
                let hidden = (gate * up)?;
                hidden
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            {
                // CP-4 (#1082): the fused MLP silu*mul kernel fuses two ops the kt
                // Tape can't see and gates on !track_op — but the tape-authoritative
                // path's intermediates are detached (track_op==false), so it would
                // fire and leave the gate/up projections' grads islands. Disable it
                // under a tape recording scope so the unfused, tape-wired silu + mul
                // path runs. Default (no tape scope) unchanged.
                let fused_hidden = if !gpu_fused_mlp_silu_mul_disabled(gate.device())
                    && !gate.track_op()
                    && !up.track_op()
                    && !crate::tape_forward::tape_scope_active()
                {
                    if let (Some(gate_kt), Some(up_kt)) =
                        (try_borrow_kt_cuda(&gate), try_borrow_kt_cuda(&up))
                    {
                        if kiln_rmsnorm_kernel::supports_mlp_silu_mul_kt(&gate_kt, &up_kt) {
                            // Phase 7 (#1082): kt-only. Same FFI symbol. Keep the
                            // output as kt — no candle copy-out (down-proj is kt).
                            let hidden =
                                kiln_rmsnorm_kernel::fused_mlp_silu_mul_kt(&gate_kt, &up_kt)
                                    .map_err(|e| anyhow::anyhow!("kt fused_mlp_silu_mul2: {e}"))?;
                            Some(hidden)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(hidden) = fused_hidden {
                    hidden
                } else {
                    // SiLU activation: x * sigmoid(x)
                    // CP-4 Increment 6 (#1082): wire SiLU(gate) onto the kt Tape so
                    // the gate-proj LoRA Vars chain through it. Outside a tape
                    // scope the normal forward remains authoritative.
                    // #1082 seam flip: cuda_silu now records the kt-native
                    // SiluBackward on the active tape internally (no kt->candle->kt
                    // round-trip), so call it directly — the explicit candle-bridge
                    // tape wiring here is subsumed.
                    let gate = cuda_silu(&gate)?;
                    synchronize_tensor_ready_for_model_handoff("mlp gate_silu", &gate)?;
                    // Element-wise multiply
                    // CP-4 Increment 6 (#1082): the production SwiGLU mul. Wire it
                    // so SiLU(gate) and up chain back to the gate/up projections
                    // (down_proj reaches the loss via the FFN residual, but without
                    // this its dL/dx never flows to gate/up — they stayed islands).
                    // #1082 seam flip: kt-native MulBackward recorder — no
                    // kt->candle->kt. Ok(None) outside a scope -> plain mul.
                    let hidden = if tape_scope_active {
                        require_active_tape_output(
                            crate::tape_forward::try_tape_mul_kt(&gate, &up)
                                .context("mlp mul try_tape_mul_kt")?,
                            "SwiGLU hidden multiply",
                        )?
                    } else {
                        (gate * up)?
                    };
                    hidden
                }
            }
            #[cfg(not(any(feature = "cuda", feature = "rocm")))]
            {
                // (#1082) Vulkan training/tape path: record the FUSED SwiGLU
                // `silu(gate) * up` via the device-agnostic `try_tape_swiglu_kt`
                // (`mul_sigmoid_gate` + `MulSigmoidGateBackward`, which emits
                // grads for BOTH gate and up). Without it the unfused
                // `cuda_silu(gate)` (whose tape recorder is cuda/metal-only) + a
                // plain `(gate * up)` would sever the gate_proj/up_proj LoRA
                // grads (they would be islands). No-op (returns None) outside a
                // tape scope — Vulkan inference falls through to the plain
                // composite below.
                #[cfg(feature = "vulkan")]
                if crate::tape_forward::tape_scope_active() {
                    let hidden = crate::tape_forward::try_tape_swiglu_kt(&gate, &up)
                        .context("swiglu_ffn_impl_no_chunk try_tape_swiglu_kt (vulkan)")?
                        .context("active tape scope failed to record Vulkan SwiGLU")?;
                    synchronize_tensor_ready_for_model_handoff("mlp vulkan tape hidden", &hidden)?;
                    return mlp_proj_forward_decode_if(
                        backend,
                        use_metal_decode_gemv,
                        &hidden,
                        &mlp.down_proj_t,
                        mlp.down_proj_marlin.as_ref(),
                        lora_layer.and_then(|l| l.down_proj.as_ref()),
                        lora_scale,
                    );
                }
                // SiLU activation: x * sigmoid(x)
                let gate = cuda_silu(&gate)?;
                synchronize_tensor_ready_for_model_handoff("mlp gate_silu", &gate)?;
                // Element-wise multiply
                let hidden = (gate * up)?;
                hidden
            }
        }
    };
    synchronize_tensor_ready_for_model_handoff("mlp hidden", &hidden)?;
    // hidden @ down_proj_t -> [batch, seq_len, hidden_size]
    let out = {
        kiln_nvtx::range!(c"kiln/mlp/down");
        mlp_proj_forward_decode_if(
            backend,
            use_metal_decode_gemv,
            &hidden,
            &mlp.down_proj_t,
            mlp.down_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.down_proj.as_ref()),
            lora_scale,
        )?
    };
    synchronize_tensor_ready_for_model_handoff("mlp down_proj", &out)?;
    Ok(out)
}

pub(super) fn mlp_proj_forward_decode_if(
    backend: Option<&dyn BackendRuntime>,
    use_metal_decode_gemv: bool,
    x: &Tensor,
    weight_t: &Tensor,
    marlin: Option<&crate::marlin_proj::MarlinPackedProj>,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if crate::tape_forward::tape_scope_active() && marlin.is_some() {
        anyhow::bail!("active tape scope cannot use forward-only Marlin MLP projection");
    }
    #[cfg(feature = "cuda")]
    if !crate::tape_forward::tape_scope_active()
        && let Some(packed) = marlin
    {
        // #1082 H1: kt-native Marlin matmul — no kt->candle->candle->kt
        // round-trip on the per-token activation/result. Runs gate/up/down
        // ×32 layers/token. The kt-native LoRA delta/add chain runs directly
        // on the kt base.
        let base = crate::marlin_proj::matmul_bf16_kt(x, packed)
            .context("mlp_proj_forward: kt-native marlin matmul")?;
        if let Some(proj) = lora {
            if let Some(delta) =
                try_kt_lora_delta(x, proj, lora_scale).context("mlp_proj_forward: kt lora delta")?
            {
                let delta = if delta.dtype() == base.dtype() {
                    delta
                } else {
                    delta
                        .to_dtype(base.dtype())
                        .context("mlp_proj_forward: kt lora delta cast")?
                };
                if let Some(out) =
                    try_kt_lora_add(&base, &delta).context("mlp_proj_forward: kt lora add")?
                {
                    return Ok(out);
                }
                return Ok((base + delta).context("mlp_proj_forward: add lora delta")?);
            }
            // kt-native composite LoRA delta fallback (#1082: compute_lora_delta
            // is kt now — pass kt `x` directly).
            let delta =
                compute_lora_delta(x, proj, lora_scale).context("mlp_proj_forward: lora delta")?;
            return Ok((base + delta).context("mlp_proj_forward: add lora delta")?);
        }
        return Ok(base);
    }
    // Non-CUDA builds never carry Marlin weights; reference the parameter so
    // the signature stays unified without a dead_code warning.
    let _ = marlin;
    linear_with_lora_t_backend_decode_if(
        backend,
        use_metal_decode_gemv,
        x,
        weight_t,
        lora,
        lora_scale,
    )
}
