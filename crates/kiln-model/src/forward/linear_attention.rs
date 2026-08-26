use super::*;

// ---------------------------------------------------------------------------
// Gated DeltaNet (GDN) linear attention primitives
// ---------------------------------------------------------------------------

/// L2 normalize the last dimension: x / sqrt(sum(x^2) + eps).
/// Returns result in F32 regardless of input dtype.
///
/// Phase 7 (#1082): by default, when the (F32-promoted) input is a
/// contiguous CUDA tensor, route through
/// `kiln_tensor::cuda_l2norm_last_axis` via the kt-bridge borrow
/// adapter. Falls through to the portable candle composite when the
/// default-on kt route is disabled or any precondition fails.
pub(super) fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    #[cfg(feature = "vulkan")]
    if matches!(x_f32.device(), Device::Vulkan(_))
        && x_f32.is_contiguous()
        && x_f32.layout().start_offset() == 0
        && x_f32.rank() > 0
        && x_f32
            .shape()
            .last()
            .is_some_and(|&hidden| hidden > 0 && hidden <= 256)
    {
        // Axis-reduction fallback materializes on CPU. GDN feeds this result
        // directly into a Vulkan state matmul, so keep the whole Q/K norm on
        // device through the native trailing-axis kernel. The surrounding
        // gdn_qk_norm recorder owns the backward node for this forward result.
        return kiln_tensor::vulkan_l2norm_last_axis(&x_f32, 1e-6)
            .context("Vulkan GDN L2 normalization");
    }
    #[cfg(feature = "cuda")]
    if crate::kt_api_policy::stable_routes_enabled()
        && matches!(x_f32.device(), Device::Cuda(_))
        && x_f32.is_contiguous()
    {
        if let Some(out) = try_kt_l2_normalize(&x_f32, 1e-6)? {
            return Ok(out);
        }
    }
    // Phase 7 (#1082): when stable KT routes are enabled and the F32 input is a
    // contiguous CUDA tensor, route the `sqr().sum_keepdim(-1)`
    // two-op composite through `kiln_tensor::cuda_sum_squared_last_axis`
    // (single fused kernel) plus a zero-cost `unsqueeze(-1)` to
    // restore the trailing-dim shape. Falls through to the candle
    // composite when any precondition fails so behavior is identical
    // with the gate off.
    let sq_sum = {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_sum_squared_last_dim_keepdim(&x_f32)? {
                out
            } else {
                x_f32.sqr()?.sum_keepdim(LAST_DIM)?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            x_f32.sqr()?.sum_keepdim(LAST_DIM)?
        }
    };
    // Phase 7 (#1082): when stable KT routes are enabled and `sq_sum` is a contiguous
    // CUDA tensor of {F32, BF16, F16}, route the `+ 1e-6` epsilon
    // step through `kiln_tensor::cuda_scalar_op` with kind 0
    // (AddScalar). Falls through to the candle composite when any
    // precondition fails so behavior is identical with the gate
    // off. Mirrors the softplus + sigmoid wirings of try_kt_add_scalar.
    //
    // Phase 7 (#1082): when stable KT routes are enabled and the addend is a
    // contiguous CUDA tensor of a supported dtype, route the
    // `.sqrt()` step through `kiln_tensor::cuda_activation_unary`
    // with kind 14 (Sqrt). Falls through to the candle composite
    // when any precondition fails.
    let sq_sum_eps = {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_add_scalar(&sq_sum, 1e-6)? {
                out
            } else {
                (sq_sum + 1e-6)?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            (sq_sum + 1e-6)?
        }
    };
    let norm = {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_sqrt(&sq_sum_eps)? {
                out
            } else {
                sq_sum_eps.sqrt()?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            sq_sum_eps.sqrt()?
        }
    };
    let normalized = x_f32.broadcast_div(&norm)?;
    Ok(normalized)
}

/// Phase 7 (#1082) — kt-API `sqr().sum_keepdim(-1)` migration
/// helper. Routes a contiguous CUDA candle tensor through
/// `kiln_tensor::cuda_sum_squared_last_axis` (which reduces the
/// trailing axis) and re-applies `unsqueeze(-1)` so the output
/// shape matches `sum_keepdim`. The squaring and reduction are
/// fused into a single kernel by the kt path.
///
/// Returns `Ok(None)` on any incompatibility so the caller falls
/// through to the candle composite. NVTX range
/// `kiln/sum_sq_last_dim_kt` brackets the migrated call so nsys
/// traces separate the path from the baseline composite.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_sum_squared_last_dim_keepdim(x: &Tensor) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    if !matches!(x.device(), Device::Cuda(_))
        || !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
        || !x.is_contiguous()
        || x.rank() == 0
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/sum_sq_last_dim_kt");

    let out_kt = match kiln_tensor::cuda_sum_squared_last_axis(x) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let reduced = out_kt;
    let out = reduced.unsqueeze(reduced.rank()).map_err(|e| {
        anyhow::anyhow!("try_kt_sum_squared_last_dim_keepdim: unsqueeze failed: {e}")
    })?;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API l2-normalize migration helper. Routes
/// a contiguous F32 CUDA candle tensor through
/// `kiln_tensor::cuda_l2norm_last_axis`.
///
/// Returns `Ok(None)` on any incompatibility so the caller falls
/// through to the candle composite. NVTX range
/// `kiln/l2_normalize_kt` brackets the migrated call so nsys traces
/// separate the path from the baseline composite.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_l2_normalize(x_f32: &Tensor, eps: f32) -> Result<Option<Tensor>> {
    kiln_nvtx::range!(c"kiln/l2_normalize_kt");

    let out_kt = match kiln_tensor::cuda_l2norm_last_axis(x_f32, eps) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

pub(super) fn gdn_qk_norm(
    q: &Tensor,
    k: &Tensor,
    input_dtype: DType,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    let (q_out, k_out) = gdn_qk_norm_forward(q, k, input_dtype, scale)?;

    // Phase 6a/CP-4 (#1082): record GdnL2NormScaleBackward nodes for the
    // L2-qk-norm outputs the production op just produced (`q_out =
    // l2_normalize(q) * scale`, `k_out = l2_normalize(k)`). Record-only (no
    // re-run); no-op unless a tape scope is active. The production outputs are
    // untouched.
    // All fast paths feed through here, so the wiring covers every dispatch.
    // #1082 seam flip: kt-native GdnL2NormScaleBackward recorder — no kt->candle->kt.
    // Vulkan included: `try_tape_gdn_l2_norm_scale_kt` + `GdnL2NormScaleBackward`
    // are device-agnostic, and without the qk-norm node the recurrence's dq/dk
    // sever before the qkv_split → in_proj_qkv LoRA `Var` on Vulkan.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        let q_out = require_active_tape_output(
            crate::tape_forward::try_tape_gdn_l2_norm_scale_kt(q, scale, &q_out)
                .context("gdn_qk_norm q try_tape_gdn_l2_norm_scale_kt")?,
            "GDN query L2 normalization",
        )?;
        let k_out = require_active_tape_output(
            crate::tape_forward::try_tape_gdn_l2_norm_scale_kt(k, 1.0, &k_out)
                .context("gdn_qk_norm k try_tape_gdn_l2_norm_scale_kt")?,
            "GDN key L2 normalization",
        )?;
        return Ok((q_out, k_out));
    }

    Ok((q_out, k_out))
}

/// Pure forward of [`gdn_qk_norm`] — `q_out = l2_normalize(q) * scale`,
/// `k_out = l2_normalize(k)`, cast to `input_dtype`. Split out so the tape
/// recording in `gdn_qk_norm` runs once at a single exit (all fast paths
/// feed through here) without re-deriving the per-path dispatch.
pub(super) fn gdn_qk_norm_forward(
    q: &Tensor,
    k: &Tensor,
    input_dtype: DType,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    #[cfg(any(feature = "metal", feature = "cuda"))]
    let fused_forward_only_allowed = !any_kt_tensor_tracks_op(&[q, k]);
    #[cfg(feature = "metal")]
    {
        if fused_forward_only_allowed
            && input_dtype == DType::BF16
            && crate::backend::metal::metal_gdn_qk_norm_supports(q, k)
        {
            return crate::backend::metal::metal_gdn_qk_norm_f32_bf16(q, k, scale as f32, 1e-6)
                .context("metal gdn qk_norm kernel failed");
        }
    }

    #[cfg(feature = "cuda")]
    {
        if crate::cuda_policy::current_cuda_kernel_policy().fused_l2_qk_norm
            && fused_forward_only_allowed
            && input_dtype == DType::BF16
        {
            if let (Some(q_kt), Some(k_kt)) = (try_borrow_kt_cuda(q), try_borrow_kt_cuda(k)) {
                if kiln_rmsnorm_kernel::supports_l2_qk_norm_kt(&q_kt, &k_kt) {
                    // Phase 7 (#1082): kt-only. Same closeout pattern as
                    // conv1d (2ebcfb08), marlin (0841c266), GDN (86c7f134),
                    // flash-attn (9ac211e9). Bit-exact: bottoms out in the
                    // same `kiln_fused_l2_qk_norm` FFI symbol.
                    // #1082: keep the fused L2-QK-norm output as kt — this fn
                    // returns kt and the fallback below is kt, so the candle
                    // copy-out is gone.
                    let (q_out, k_out) =
                        kiln_rmsnorm_kernel::fused_l2_qk_norm_kt(&q_kt, &k_kt, scale as f32, 1e-6)
                            .map_err(|e| anyhow::anyhow!("kt fused_l2_qk_norm: {e}"))?;
                    return Ok((q_out, k_out));
                }
            }
        }
    }

    let q = l2_normalize(q)?; // F32
    let k = l2_normalize(k)?; // F32
    let q = (q * scale)?.to_dtype(input_dtype)?;
    let k = k.to_dtype(input_dtype)?;
    Ok((q, k))
}

/// softplus(x) = ln(1 + exp(x)), numerically stable for all x.
///
/// Uses the identity: softplus(x) = max(x, 0) + ln(1 + exp(-|x|))
/// Since exp(-|x|) ∈ (0, 1], no overflow is possible.
/// This matches PyTorch's F.softplus output (which clamps to linear for x > 20).
pub(super) fn softplus(x: &Tensor) -> Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    // Phase 7 (#1082): when stable KT routes are enabled and `x` + `zeros` are
    // contiguous CUDA tensors of {F32, BF16, F16}, route the
    // pointwise `max(x, 0)` (relu) through
    // `kiln_tensor::cuda_binary_minmax` with kind 1 (Max). Falls
    // through to the candle `.maximum(&zeros)` when any
    // precondition fails so behavior is identical with the gate
    // off.
    let relu_x = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_max_binary(x, &zeros)? {
                Some(out) => out,
                None => x.maximum(&zeros)?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            x.maximum(&zeros)?
        }
    };
    // |x| = relu(x) + relu(-x)
    //
    // Phase 7 (#1082): when stable KT routes are enabled and `x` is a contiguous
    // CUDA tensor of {F32, BF16, F16}, route the `|x|` computation
    // through a single `kiln_tensor::cuda_activation_unary` call
    // with kind 13 (Abs) — one fused kernel replacing the
    // `neg + relu(-x) + add(relu(x), relu(-x))` composite (three
    // candle ops + two intermediate buffers). Falls through to the
    // `relu(x) + relu(-x)` identity when any precondition fails so
    // behavior is identical with the gate off. First production
    // call site for `try_kt_abs`.
    //
    // Phase 7 (#1082): when stable KT routes are enabled and the input is a contiguous
    // CUDA tensor of a supported dtype, route the `.neg()` through
    // `kiln_tensor::cuda_activation_unary` with kind 12 (Neg).
    // Falls through to the candle composite when any precondition
    // fails.
    let abs_x = {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_abs(x)? {
                out
            } else {
                let neg_x = if let Some(out) = try_kt_neg(x)? {
                    out
                } else {
                    x.neg()?
                };
                // Phase 7 (#1082): same kt-API max_binary migration
                // for the `relu(-x)` half of the
                // `|x| = relu(x) + relu(-x)` identity.
                let relu_neg_x = match try_kt_max_binary(&neg_x, &zeros)? {
                    Some(out) => out,
                    None => neg_x.maximum(&zeros)?,
                };
                (relu_x.clone() + relu_neg_x)?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let neg_x = x.neg()?;
            let relu_neg_x = neg_x.maximum(&zeros)?;
            (relu_x.clone() + relu_neg_x)?
        }
    };
    // Phase 7 (#1082): same kt-API neg migration for `abs_x.neg()`.
    let neg_abs = {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_neg(&abs_x)? {
                out
            } else {
                abs_x.neg()?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            abs_x.neg()?
        }
    };
    // log(1 + exp(-|x|)) — always stable since exp(-|x|) ∈ (0, 1]
    //
    // Phase 7 (#1082): when stable KT routes are enabled and `neg_abs` is a
    // contiguous CUDA tensor of {F32, BF16, F16}, route the
    // `.exp()` step of the softplus composite through
    // `kiln_tensor::cuda_activation_unary` with kind 6 (Exp).
    // Falls through to the candle composite when any precondition
    // fails.
    //
    // Phase 7 (#1082): when stable KT routes are enabled and the exp() output is a
    // contiguous CUDA tensor of a supported dtype, route the
    // `+ 1.0` step of the softplus composite through
    // `kiln_tensor::cuda_scalar_op` with kind 0 (AddScalar).
    // Falls through to the candle composite when any precondition
    // fails.
    let exp_neg_abs = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_exp(&neg_abs)? {
                Some(out) => out,
                None => neg_abs.exp()?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            neg_abs.exp()?
        }
    };
    let one_plus = {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_add_scalar(&exp_neg_abs, 1.0)? {
                out
            } else {
                (exp_neg_abs + 1.0)?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            (exp_neg_abs + 1.0)?
        }
    };
    // Phase 7 (#1082): when stable KT routes are enabled and the input is a contiguous
    // CUDA tensor of a supported dtype, route the `.log()` through
    // `kiln_tensor::cuda_activation_unary` with kind 5 (Log = ln(x)).
    // Falls through to the candle composite when any precondition
    // fails so behavior is identical with the gate off.
    let log_term = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_log(&one_plus)? {
                Some(out) => out,
                None => one_plus.log()?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            one_plus.log()?
        }
    };
    Ok((relu_x + log_term)?)
}

pub(super) fn gated_rms_norm(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    // (#1082) GDN-on-Vulkan training: the Vulkan `gdn_gated_rms_norm` backend
    // kernel still crosses host bytes before restoring its result to Vulkan,
    // so it does not preserve an active tape even though inference remains
    // device-resident downstream. CUDA/ROCm/Metal preserve the resident tape
    // path and can use their fused forward before recording the analytic
    // `GdnGatedRmsNormBackward` in the caller.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    let skip_backend_for_active_tape = crate::tape_forward::tape_scope_active()
        && !BackendCapabilityQueries::backend_capabilities(backend)
            .gdn
            .gated_rms_norm_preserves_tape_residency;
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )))]
    let skip_backend_for_active_tape = false;
    if !skip_backend_for_active_tape
        && !any_kt_tensor_tracks_op(&[x, z, weight])
        && GdnBackend::runtime_supports_gdn_gated_rms_norm(backend)
        && let Some(out) = GdnBackend::runtime_gdn_gated_rms_norm(backend, x, z, weight, eps)?
    {
        return Ok(out);
    }
    gated_rms_norm_fallback(x, z, weight, eps)
}

/// Gated RMSNorm: rms_norm(x, weight) * silu(z).
///
/// Applied per-group on the last dimension. Returns F32.
///
/// `x`: [..., dim] — attention output
/// `z`: [..., dim] — output gate (from in_proj_z)
/// `weight`: [dim] — learnable scale
pub(super) fn gated_rms_norm_fallback(
    x: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let x_device = x.device();
    let z = if z.device() != x_device {
        z.to_device(x_device)?
    } else {
        z.clone()
    };
    let weight = if weight.device() != x_device {
        weight.to_device(x_device)?
    } else {
        weight.clone()
    };
    // #1082: x/z arrive as transposed GDN head views (non-contiguous); kt's
    // CastOp requires contiguous (candle's `to_dtype` copied implicitly).
    // `.contiguous()` is an O(1) no-op when already contiguous.
    let x_f32 = x.contiguous()?.to_dtype(DType::F32)?;
    let z_f32 = z.contiguous()?.to_dtype(DType::F32)?;
    let w_f32 = weight.contiguous()?.to_dtype(DType::F32)?;

    // RMS norm on last dimension
    //
    // Phase 7 (#1082): when stable KT routes are enabled and the squared F32 input
    // is a contiguous CUDA tensor, route the `mean_keepdim(-1)`
    // step through `kiln_tensor::cuda_mean_last_axis` plus a
    // zero-cost `unsqueeze(-1)`. Mirrors the rms_norm_fallback
    // wiring (line ~7298). Falls through to the candle
    // `.mean_keepdim()` when any precondition fails so behavior
    // is identical with the gate off.
    let sq = x_f32.sqr()?;
    let variance = {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_mean_last_dim_keepdim(&sq)? {
                out
            } else {
                sq.mean_keepdim(LAST_DIM)?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            sq.mean_keepdim(LAST_DIM)?
        }
    };
    // Phase 7 (#1082): when stable KT routes are enabled and `variance` is a
    // contiguous CUDA tensor, route the `+ eps` step through
    // `kiln_tensor::cuda_scalar_op` with kind 0 (AddScalar).
    // Mirrors the rms_norm_fallback wiring. Falls through to the
    // candle `+ f64` composite when any precondition fails.
    let variance_plus_eps = {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_add_scalar(&variance, eps)? {
                out
            } else {
                (variance + eps)?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            (variance + eps)?
        }
    };
    // Phase 7 (#1082): when stable KT routes are enabled, route the RMSNorm-tail
    // `(variance + eps).sqrt().recip()` composite through
    // `kiln_tensor::cuda_activation_unary` kind 28 (Rsqrt) — a
    // single fused kernel that replaces the two candle calls + the
    // intermediate sqrt buffer. Falls through to the candle
    // composite when any precondition fails.
    let rms_inv = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_rsqrt(&variance_plus_eps)? {
                Some(out) => out,
                None => variance_plus_eps.sqrt()?.recip()?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            variance_plus_eps.sqrt()?.recip()?
        }
    };
    // #1082: `rms_inv` descends from `variance` (mean + `unsqueeze(-1)` view),
    // so it can be non-contiguous; kt elementwise still requires contiguous
    // operands ("stride-aware path not in Phase 1.15"). candle broadcast over
    // strided views implicitly; `.contiguous()` (O(1) no-op when contiguous)
    // restores that. Guard each mul operand defensively.
    let normed = x_f32.broadcast_mul(&rms_inv.contiguous()?)?;
    let normed = normed.contiguous()?.broadcast_mul(&w_f32)?;

    // Output gate: silu(z) = z * sigmoid(z)
    let gate = cuda_silu(&z_f32)?;
    let out = (normed.contiguous()? * gate.contiguous()?)?;
    Ok(out)
}

/// Causal depthwise conv1d for prefill (seq_len > 1).
///
/// `x`: [batch, channels, seq_len]
/// `weight`: [channels, 1, kernel_size]
/// `conv_state`: [batch, channels, kernel_size - 1] — updated to last K-1 inputs
pub(super) fn causal_conv1d_prefill(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    let compute_dtype = causal_conv1d_prefill_compute_dtype(x, weight, conv_state, kernel_size);
    causal_conv1d_prefill_with_dtype(x, weight, conv_state, kernel_size, compute_dtype)
}

pub(super) fn causal_conv1d_prefill_compute_dtype(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
) -> DType {
    if matches!(x.device(), Device::Metal(_))
        && x.dtype() == DType::BF16
        && weight.dtype() == DType::BF16
        && conv_state.dtype() == DType::F32
        && kernel_size == 4
    {
        DType::BF16
    } else {
        DType::F32
    }
}

pub(super) fn causal_conv1d_prefill_with_dtype(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
    compute_dtype: DType,
) -> Result<Tensor> {
    // The portable composite requires every operand to share storage. Backend
    // fast paths may deliberately return a host-resident activation when the
    // workload is too small to amortize a GPU submission, so align the small
    // weight and threaded state with that activation before falling back. The
    // state returns to its entry storage afterward so a temporary prefill
    // crossover cannot evict the following decode from its resident path.
    let x_device = x.device();
    let conv_state_device = conv_state.device();
    let weight = if weight.device() != x_device {
        weight.to_device(x_device)?
    } else {
        weight.clone()
    };
    if conv_state.device() != x_device {
        *conv_state = conv_state.to_device(x_device)?;
    }
    let (_batch, channels, seq_len) = x.dims3()?;
    let x_compute = x.to_dtype(compute_dtype)?;
    let x_state_f32 = if compute_dtype == DType::F32 {
        x_compute.clone()
    } else {
        x.to_dtype(DType::F32)?
    };
    // Squeeze [channels, 1, kernel_size] -> [channels, kernel_size]
    let w_compute = weight
        .to_dtype(compute_dtype)?
        .reshape((channels, kernel_size))?;
    let k_minus_1 = kernel_size - 1;

    // Left-pad with conv_state (previous K-1 inputs, or zeros for fresh state)
    //
    // Phase 7 (#1082): when stable KT routes are enabled and both pieces are
    // contiguous CUDA tensors of a supported dtype, route the
    // time-axis (axis=2) concat through
    // `kiln_tensor::cuda_concat(_, 2)`. Falls through to the candle
    // composite when any precondition fails so behavior is
    // identical with the gate off.
    let conv_state_compute = conv_state.to_dtype(compute_dtype)?;
    let x_padded = {
        let pieces: [&Tensor; 2] = [&conv_state_compute, &x_compute];
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_cat_dim2(&pieces)? {
                out
            } else {
                Tensor::cat(&pieces, 2)?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Tensor::cat(&pieces, 2)?
        }
    };

    // Depthwise conv: output[t] = sum_{j=0}^{K-1} weight[j] * x_padded[t+j]
    let mut output = Tensor::zeros_like(&x_compute)?;
    for j in 0..kernel_size {
        let x_slice = x_padded.narrow(2, j, seq_len)?;
        let w_j = w_compute.narrow(1, j, 1)?.unsqueeze(0)?; // [1, channels, 1]
        output = (output + x_slice.broadcast_mul(&w_j)?)?;
    }

    // Update conv_state to the last K-1 input positions
    if seq_len >= k_minus_1 {
        *conv_state = x_state_f32
            .narrow(2, seq_len - k_minus_1, k_minus_1)?
            .contiguous()?;
    } else {
        // Fewer new tokens than buffer size: shift old state and append new
        let keep = k_minus_1 - seq_len;
        let old_part = conv_state.narrow(2, seq_len, keep)?;
        // Phase 7 (#1082): same kt-API axis-2 cat migration applied
        // to the conv-state shift+append path.
        let shifted = {
            let pieces: [&Tensor; 2] = [&old_part, &x_state_f32];
            #[cfg(feature = "cuda")]
            {
                if let Some(out) = try_kt_cat_dim2(&pieces)? {
                    out
                } else {
                    Tensor::cat(&pieces, 2)?
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                Tensor::cat(&pieces, 2)?
            }
        };
        *conv_state = shifted.contiguous()?;
    }
    if conv_state.device() != conv_state_device {
        *conv_state = conv_state.to_device(conv_state_device)?;
    }

    Ok(output)
}

/// Causal depthwise conv1d for decode (seq_len == 1).
///
/// `x`: [batch, channels, 1]
/// `weight`: [channels, 1, kernel_size]
/// `conv_state`: [batch, channels, kernel_size - 1] — updated
pub(super) fn causal_conv1d_decode(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    let (_batch, channels, _one) = x.dims3()?;
    let x_f32 = x.to_dtype(DType::F32)?;
    let w_f32 = weight
        .to_dtype(DType::F32)?
        .reshape((channels, kernel_size))?;

    // Full window = [conv_state | x] -> [batch, channels, kernel_size]
    //
    // Phase 7 (#1082): when stable KT routes are enabled and both pieces are
    // contiguous CUDA tensors of a supported dtype, route the
    // decode-path time-axis (axis=2) concat through
    // `kiln_tensor::cuda_concat(_, 2)`. Falls through to the candle
    // composite when any precondition fails so behavior is
    // identical with the gate off.
    let conv_state_f32 = conv_state.to_dtype(DType::F32)?;
    let window = {
        let pieces: [&Tensor; 2] = [&conv_state_f32, &x_f32];
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_cat_dim2(&pieces)? {
                out
            } else {
                Tensor::cat(&pieces, 2)?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Tensor::cat(&pieces, 2)?
        }
    };

    // Dot product per channel: sum over kernel dimension.
    //
    // Phase 7 (#1082): when stable KT routes are enabled, the `sum(2)` reduction along
    // the kernel dim is routed through `kiln_tensor::cuda_sum_axis`
    // via the kt-bridge borrow adapter (NVTX range
    // `kiln/sum_axis_kt`). Falls through to candle's `sum(2)` when
    // any precondition fails. This is the GDN single-token conv
    // path's per-channel dot-product, called once per layer per
    // decode step — a less-trafficked but consistent migration site
    // for the sum-axis kernel that mirrors the keepdim variant
    // already used in `cuda_softmax_last_dim`'s fallback composite.
    let w_expanded = w_f32.unsqueeze(0)?; // [1, channels, kernel_size]
    let pre_sum = window.broadcast_mul(&w_expanded)?;
    let output = {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_sum_axis(&pre_sum, 2)? {
                out
            } else {
                pre_sum.sum(2)?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            pre_sum.sum(2)?
        }
    }; // [batch, channels]
    let output = output.unsqueeze(2)?; // [batch, channels, 1]

    // Update conv_state in place: drop oldest, append newest. CUDA graph
    // capture bakes the conv_state device pointer into later decode kernels, so
    // rebinding `conv_state` to a newly allocated tensor during capture leaves
    // replay with a dangling pointer. Keep the caller-owned storage stable.
    let next_state = window.narrow(2, 1, kernel_size - 1)?.contiguous()?;
    // Vulkan's portable adapter path can cross the hybrid-residency boundary
    // between prefill and decode: prefill may leave this state on the CPU while
    // the first LoRA decode projection produces a Vulkan activation. Vulkan
    // decode is not graph-captured, so adopt the newly computed state instead
    // of forcing an invalid in-place write across devices. Same-device CUDA and
    // ROCm paths retain the stable buffer identity required by graph capture.
    #[cfg(feature = "vulkan")]
    if conv_state.device() != next_state.device()
        && matches!(conv_state.device(), Device::Cpu | Device::Vulkan(_))
        && matches!(next_state.device(), Device::Cpu | Device::Vulkan(_))
    {
        *conv_state = next_state;
        return Ok(output);
    }
    conv_state
        .slice_set(&next_state, 0, 0)
        .context("update decode conv_state in place")?;

    Ok(output)
}

// ---------------------------------------------------------------------------
// GDN chunkwise analytical recurrence (Phase 6, approach (b) in the chunkwise
// plan). Replaces the per-token `for t in 0..seq_len` loop inside
// `gated_deltanet_forward` with an unrolled form that processes up to
// `GDN_CHUNK_SIZE` tokens per heavy matmul, dropping the number of GPU kernel
// launches from O(T) to O(T / C) per layer.
// ---------------------------------------------------------------------------

/// Chunk size for the analytical GDN recurrence. C = 64 balances:
///   - intra-chunk [C, dk] × [dk, C] matmuls large enough to saturate tensor
///     cores on A5000/4090-class GPUs for dk = dv = 128,
///   - a small-enough forward-substitution inner loop so the Vec<Tensor> cat
///     churn stays bounded.
pub const GDN_CHUNK_SIZE: usize = 64;
pub(super) const GDN_RECURRENT_PREFILL_MAX_TOKENS: usize = 2048;

/// Build a [n, n] mask on `device` with `dtype`, 1.0 where row > col else 0.0.
/// Used for the strictly lower-triangular `A_strict` mask (i < t, exclusive).
pub(super) fn strict_lower_tri_bool(n: usize, device: &Device) -> Result<Tensor> {
    // #1082: the kt `gt` op requires F32/BF16/F16 (candle allowed u32). Build
    // the index ramp directly in F32 (exact for chunk_size ≤ 64) — NOT u32 then
    // `.to_dtype(F32)`, because the kt U32→F32 cast is unsupported and the
    // fallback hits the CPU branch on a CUDA tensor. F32 `arange` → CUDA via
    // `to_device` is the well-trodden path; the comparison then runs on CUDA.
    let t = Tensor::arange(0f32, n as f32, device)?;
    let cols = t.reshape((1, n))?.broadcast_as((n, n))?;
    let rows = t.reshape((n, 1))?.broadcast_as((n, n))?;
    Ok(kiln_tensor::ops::gt(&rows, &cols)?)
}

#[cfg(test)]
#[allow(dead_code)]
pub(super) fn strict_lower_tri_mask(n: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let values: Vec<f32> = (0..n)
        .flat_map(|row| (0..n).map(move |col| f32::from(row > col)))
        .collect();
    Ok(Tensor::from_vec(values, (n, n))?
        .to_dtype(dtype)?
        .to_device(*device)?)
}

/// Build a [n, n] mask on `device` with `dtype`, 1.0 where row >= col else 0.0.
/// Used for the causal (inclusive) lower-triangular `B_mask` mask (i <= t).
pub(super) fn causal_lower_tri_bool(n: usize, device: &Device) -> Result<Tensor> {
    // #1082: kt `ge` requires F32/BF16/F16 (candle allowed u32). Build the index
    // ramp directly in F32 (exact for chunk_size ≤ 64) — NOT u32 then a cast
    // (the kt U32→F32 cast is unsupported and falls back to the CPU branch on a
    // CUDA tensor). F32 `arange` → CUDA `to_device` runs the comparison on CUDA.
    let t = Tensor::arange(0f32, n as f32, device)?;
    let cols = t.reshape((1, n))?.broadcast_as((n, n))?;
    let rows = t.reshape((n, 1))?.broadcast_as((n, n))?;
    Ok(kiln_tensor::ops::ge(&rows, &cols)?)
}

pub(super) struct GdnBackwardChunkMasks {
    pub(super) strict_bool: Tensor,
    pub(super) causal_bool: Tensor,
    pub(super) strict_mask_f32: Tensor,
    pub(super) causal_mask_f32: Tensor,
    pub(super) last_mask_f32: Tensor,
}

pub(super) fn gdn_backward_chunk_masks(
    batch: usize,
    heads: usize,
    chunk: usize,
    device: &Device,
) -> Result<GdnBackwardChunkMasks> {
    let strict_bool = strict_lower_tri_bool(chunk, device)?
        .reshape((1, 1, chunk, chunk))?
        .broadcast_as((batch, heads, chunk, chunk))?;
    let causal_bool = causal_lower_tri_bool(chunk, device)?
        .reshape((1, 1, chunk, chunk))?
        .broadcast_as((batch, heads, chunk, chunk))?;

    let mask_ones = Tensor::ones((batch, heads, chunk, chunk), DType::F32, device)?;
    let mask_zeros = Tensor::zeros((batch, heads, chunk, chunk), DType::F32, device)?;
    let strict_mask_f32 = strict_bool.where_cond(&mask_ones, &mask_zeros)?;
    let causal_mask_f32 = causal_bool.where_cond(&mask_ones, &mask_zeros)?;

    let idx_f32 = Tensor::arange(0f32, chunk as f32, device)?;
    let target = kiln_tensor::ops::full_like(&idx_f32, (chunk - 1) as f32)?;
    let eq_bool = kiln_tensor::ops::eq(&idx_f32, &target)?
        .reshape((1, 1, chunk))?
        .broadcast_as((batch, heads, chunk))?;
    let last_ones = Tensor::ones((batch, heads, chunk), DType::F32, device)?;
    let last_zeros = Tensor::zeros((batch, heads, chunk), DType::F32, device)?;
    let last_mask_f32 = eq_bool.where_cond(&last_ones, &last_zeros)?;

    Ok(GdnBackwardChunkMasks {
        strict_bool,
        causal_bool,
        strict_mask_f32,
        causal_mask_f32,
        last_mask_f32,
    })
}

#[cfg(test)]
#[allow(dead_code)]
pub(super) fn causal_lower_tri_mask(n: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    Ok(causal_lower_tri_bool(n, device)?.to_dtype(dtype)?)
}

/// Compute the chunk-local W = (I + A_strict)^{-1} (beta * V_prime) by
/// forward substitution. On backends that advertise
/// `supports_gdn_forward_substitution()`, dispatches to the fused kernel
/// when `chunk_size <= 128`. Otherwise it falls back to the per-token
/// reference loop.
pub(super) fn compute_w_chunk(
    backend: &dyn BackendRuntime,
    a_strict: &Tensor, // [B, nv, C, C]
    v_prime: &Tensor,  // [B, nv, C, dv]
    beta_c: &Tensor,   // [B, nv, C]
    c: usize,
) -> Result<Tensor> {
    // The kernel envelope is C <= 128; callers enforce this precondition so
    // we never pay for a backend call we know will decline.
    if c <= 128
        && !any_kt_tensor_tracks_op(&[a_strict, v_prime, beta_c])
        && GdnBackend::runtime_supports_gdn_forward_substitution(backend)
    {
        kiln_nvtx::range!(c"kiln/attn/gdn/chunk");
        // The fused triangular solve is a custom FFI kernel. Make the stream
        // handoff explicit before it reads tensors produced by normal model ops.
        synchronize_tensor_ready_for_model_handoff("gdn forward-substitution a_strict", a_strict)?;
        synchronize_tensor_ready_for_model_handoff("gdn forward-substitution v_prime", v_prime)?;
        synchronize_tensor_ready_for_model_handoff("gdn forward-substitution beta", beta_c)?;
        // #1082 DoD-101/102: `gdn_forward_substitution` is now kt-typed —
        // pass the kt tensors directly, no candle bridge.
        if let Some(out) =
            GdnBackend::runtime_gdn_forward_substitution(backend, a_strict, v_prime, beta_c)?
        {
            // The caller immediately feeds `out` into regular kt matmul /
            // elementwise ops; synchronize the FFI output before that handoff.
            synchronize_tensor_ready_for_model_handoff("gdn forward-substitution w", &out)?;
            return Ok(out);
        }
    }
    compute_w_chunk_fallback(a_strict, v_prime, beta_c, c)
}

/// Reference per-token forward substitution. Kept as the CPU path and as
/// the correctness oracle for the fused CUDA kernel.
pub(super) fn compute_w_chunk_fallback(
    a_strict: &Tensor,
    v_prime: &Tensor,
    beta_c: &Tensor,
    c: usize,
) -> Result<Tensor> {
    let beta_col = beta_c.unsqueeze(3)?; // [B, nv, C, 1]
    let mut w_rows: Vec<Tensor> = Vec::with_capacity(c);
    for t in 0..c {
        let vp_t = v_prime.narrow(2, t, 1)?; // [B, nv, 1, dv]
        let beta_t = beta_col.narrow(2, t, 1)?; // [B, nv, 1, 1]
        let w_t = if t == 0 {
            vp_t.broadcast_mul(&beta_t)?
        } else {
            let a_row = a_strict.narrow(2, t, 1)?.narrow(3, 0, t)?.contiguous()?;
            let w_prev = Tensor::cat(&w_rows, 2)?;
            let sub = a_row.matmul(&w_prev)?; // [B, nv, 1, dv]
            (vp_t - sub)?.broadcast_mul(&beta_t)?
        };
        w_rows.push(w_t);
    }
    Ok(Tensor::cat(&w_rows, 2)?)
}

#[allow(dead_code)]
pub(super) fn compute_chunk_body_reference(
    a_strict: &Tensor,
    b_mask: &Tensor,
    v_prime: &Tensor,
    q_s_scaled: &Tensor,
    beta_c: &Tensor,
    decay_last_col_u: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let c = v_prime.dim(2)?;
    let w = compute_w_chunk_fallback(a_strict, v_prime, beta_c, c)?;
    let intra = b_mask.matmul(&w)?;
    let out_chunk = (q_s_scaled + &intra)?;
    let w_weighted = w.broadcast_mul(decay_last_col_u)?.contiguous()?;
    Ok((out_chunk, w_weighted))
}

/// Specialized single-token GDN recurrence.
///
/// This is the non-CUDA fast path for `seq_len == 1`, avoiding the chunkwise
/// prep work (`KKT`, `QKT`, masks, triangular solve) that is only worthwhile
/// when a chunk contains multiple tokens.
pub(super) fn gdn_single_token_recurrence(
    q: &Tensor,         // [B, nv, 1, dk]
    k: &Tensor,         // [B, nv, 1, dk]
    v: &Tensor,         // [B, nv, 1, dv]
    beta: &Tensor,      // [B, nv, 1]
    g: &Tensor,         // [B, nv, 1]
    state: &mut Tensor, // [B, nv, dk, dv]
) -> Result<Tensor> {
    let dtype = q.dtype();

    // Phase 7 (#1082): cluster migration. The
    // `g.to_dtype(F32).exp().to_dtype(dtype)` gate-decay
    // composite runs every GDN decode token on every layer of
    // the 24 GDN layers. Stable KT policy controls the three-op cluster as
    // one route set. Each helper still falls through cleanly when
    // any single op's preconditions fail (the candle `?`
    // operator preserves identical numerics).
    let p = {
        #[cfg(feature = "cuda")]
        {
            let g_f32 = match try_kt_to_dtype(g, DType::F32)? {
                Some(out) => out,
                None => g.to_dtype(DType::F32)?,
            };
            let g_exp = match try_kt_exp(&g_f32)? {
                Some(out) => out,
                None => g_f32.exp()?,
            };
            match try_kt_to_dtype(&g_exp, dtype)? {
                Some(out) => out,
                None => g_exp.to_dtype(dtype)?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            g.to_dtype(DType::F32)?.exp()?.to_dtype(dtype)?
        }
    };
    let p_u = p.unsqueeze(3)?; // [B, nv, 1, 1]

    let ks_entry = k.matmul(&*state)?; // [B, nv, 1, dv]
    let q_s = q.matmul(&*state)?; // [B, nv, 1, dv]

    let v_prime = (v - ks_entry.broadcast_mul(&p_u)?)?;
    let w = v_prime.broadcast_mul(&beta.unsqueeze(3)?)?; // [B, nv, 1, dv]
    let qk = kiln_tensor::ops::matmul_rhs_transposed(q, k)?; // [B, nv, 1, 1]
    let out = (q_s.broadcast_mul(&p_u)? + qk.matmul(&w)?)?;

    let state_scaled = state.broadcast_mul(&p_u)?;
    let delta_state = kiln_tensor::ops::matmul_lhs_transposed(k, &w)?;
    *state = (state_scaled + delta_state)?;

    Ok(out)
}

/// Analytical chunkwise form of the Gated DeltaNet recurrence.
///
/// The per-token recurrence is
///
/// ```text
///   S_t   = exp(g_t) * S_{t-1}  +  k_t ⊗ delta_t
///   delta_t = beta_t * (v_t - k_t · (exp(g_t) * S_{t-1}))
///   out_t = q_t · S_t
/// ```
///
/// Within a chunk of up to `chunk_size` tokens, let `G[t] = cumsum(g)[t]`.
/// The per-token recurrence unrolls into the closed form (derived from the
/// standard GLA / chunk_gla_fwd identity used in fla-org and RWKV-5):
///
/// 1. Inter-chunk carry
///    ```text
///      V'[t] = v[t] - exp(G[t]) * (k[t] · S_entry)
///    ```
/// 2. Strict intra-chunk decay mask
///    ```text
///      A_strict[t, i] = exp(G[t] - G[i]) * (k[t] · k[i])   for i < t, else 0
///    ```
/// 3. Forward-substitution / triangular solve for W[t]
///    ```text
///      W[t] = beta[t] * ( V'[t] - Σ_{i<t} A_strict[t, i] * W[i] )
///    ```
/// 4. Output
///    ```text
///      B_mask[t, i] = exp(G[t] - G[i]) * (q[t] · k[i])     for i <= t, else 0
///      out[t] = exp(G[t]) * (q[t] · S_entry) + Σ_{i<=t} B_mask[t, i] * W[i]
///    ```
/// 5. State exit
///    ```text
///      S_new = exp(G[C-1]) * S_entry + Σ_i exp(G[C-1] - G[i]) * k[i] ⊗ W[i]
///    ```
///
/// This is numerically equivalent to the per-token loop (modulo rounding in
/// the bf16 hot path) and matches the pre-existing sequential code exactly
/// for chunk_size = 1 (decode path).
///
/// Inputs are already transposed to `[B, nv, T, *]` layout. `state` is
/// mutated in place and must be in the hot-path dtype (bf16 in production,
/// F32 on CPU tests); the caller is responsible for preserving the external
/// F32-state invariant.
///
/// Returns: `[B, nv, T, dv]`.
pub(super) fn gdn_chunkwise_recurrence(
    backend: &dyn BackendRuntime,
    q: &Tensor,         // [B, nv, T, dk]
    k: &Tensor,         // [B, nv, T, dk]
    v: &Tensor,         // [B, nv, T, dv]
    beta: &Tensor,      // [B, nv, T]
    g: &Tensor,         // [B, nv, T]
    state: &mut Tensor, // [B, nv, dk, dv]
    chunk_size: usize,
) -> Result<Tensor> {
    // (#1082) GDN-on-Vulkan training: the chunk prep below uses CPU-only kt ops
    // (`cumsum`, `where_cond`) that error on Vulkan storage rather than host-
    // falling-back. The GPU fast path (`backend.gdn_chunkwise_forward`) is the
    // intended Vulkan route but is skipped under a tape scope (it reads back to
    // CPU and severs the tape — see the caller). So when we reach the unfused
    // scan with Vulkan inputs, run the whole recurrence on CPU and move the
    // result + the in-place-updated `state` back to the input device. The
    // recurrence is wrapped by the caller's analytic `GdnRecurrentBackward`, so
    // only the OUTPUT device matters for tape connectivity — and that is
    // restored here. No-op on CPU/CUDA/Metal (the match leaves them untouched).
    #[cfg(feature = "vulkan")]
    if matches!(q.device(), Device::Vulkan(_)) {
        let dev = q.device();
        let to_cpu = |t: &Tensor| -> Result<Tensor> { Ok(t.to_device(Device::Cpu)?) };
        let q_c = to_cpu(q)?;
        let k_c = to_cpu(k)?;
        let v_c = to_cpu(v)?;
        let beta_c = to_cpu(beta)?;
        let g_c = to_cpu(g)?;
        let mut state_c = to_cpu(state)?;
        let out_c = gdn_chunkwise_recurrence(
            backend,
            &q_c,
            &k_c,
            &v_c,
            &beta_c,
            &g_c,
            &mut state_c,
            chunk_size,
        )?;
        // Move the updated state + output back to the activation device so the
        // external F32-state invariant + the tape chain both stay on Vulkan.
        *state = state_c.to_device(dev)?;
        return Ok(out_c.to_device(dev)?);
    }
    let (batch, heads, seq_len, _) = q.dims4()?;
    let dtype = q.dtype();
    // #1082: kt `.device()` returns a value; keep a reference for mask creation.
    let device_val = q.device();
    let device = &device_val;

    // Single-token decode fast path. The chunkwise machinery (preshape,
    // decay matrix, KKT, forward sub, B_mask) costs more than the per-token
    // recurrence itself when seq_len == 1, which is the cause of the −54%
    // decode regression in PR #80. The backend's `gdn_recurrent_step`
    // kernel collapses the whole recurrence into one block per (B,H). Backend
    // capabilities own dtype-specific eligibility.
    if seq_len == 1 {
        let use_backend_recurrent_step = state.dtype() == dtype
            && !any_kt_tensor_tracks_op(&[q, k, v, beta, g, state])
            && GdnBackend::runtime_supports_gdn_recurrent_step(backend)
            && gdn_recurrent_step_supports_dtype(backend, dtype);
        if use_backend_recurrent_step {
            // The five squeeze+contiguous calls below can copy the single-row
            // inputs before the recurrent forward runs. The dedicated NVTX
            // range lets nsys attribute this separately from the kernel itself.
            let (q1, k1, v1, beta1, g1) = {
                kiln_nvtx::range!(c"kiln/attn/gdn/precopy");
                (
                    q.squeeze(2)?.contiguous()?,
                    k.squeeze(2)?.contiguous()?,
                    v.squeeze(2)?.contiguous()?,
                    beta.squeeze(2)?.contiguous()?,
                    g.squeeze(2)?.contiguous()?,
                )
            };
            let out_opt = {
                kiln_nvtx::range!(c"kiln/attn/gdn/recurrent");
                // #1082 DoD-101/102: `gdn_recurrent_step` is now kt-typed and
                // mutates `state` in place through the kt `&mut` — pass kt
                // tensors directly, no candle bridge / write-back.
                GdnBackend::runtime_gdn_recurrent_step(backend, &q1, &k1, &v1, &beta1, &g1, state)?
            };
            if let Some(out) = out_opt {
                return Ok(out.unsqueeze(2)?);
            }
        }

        let out = gdn_single_token_recurrence(q, k, v, beta, g, state)?;
        return Ok(out);
    }

    // (#1082 Vulkan) Proper GPU-parallel GDN prefill. The chunkwise scan below
    // runs its per-chunk matmuls as raw kt `.matmul()`, which on Vulkan execute
    // on CPU-host tensors — i.e. on the CPU — and dominate prefill (~75%).
    // `backend.gdn_chunkwise_forward` runs the SAME chunkwise scan on the GPU in
    // parallel (`vk_gdn_chunkwise_forward_no_grad`): same gated-delta-rule math,
    // same `[B,nv,T,dv]` output, same in-place `state` replace, so it's a
    // drop-in for the loop below. The trait default returns `Ok(None)` on every
    // other backend (CUDA/Metal/CPU keep their existing paths) and the Vulkan
    // impl declines on any unsupported dtype/config, falling through here.
    // (#1082) GDN-on-Vulkan training: this GPU fast path is a forward-only
    // no-grad scan that reads its result back to `Device::Cpu`
    // (`vk_f32_tensors_to_cpu_tensors_batched_vk`) and replaces `state` with a
    // CPU tensor — severing the Vulkan tape (the gated-norm + out_proj then run
    // on CPU and their LoRA tape recorders, which require the activation on the
    // accelerator, never fire). It gates on `!any_kt_tensor_tracks_op`, but the
    // tape-authoritative path's intermediates are DETACHED (track_op==false), so
    // it would still fire. Skip it whenever a tape recording scope is active so
    // the tape-wired chunkwise loop below runs instead (its kt matmuls keep the
    // output on the activation device via the Vulkan op host-fallback, and the
    // `GdnRecurrentBackward` recorded by the caller flows dq/dk/dv back). Default
    // (inference, no tape scope) behaviour is unchanged.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    let tape_recording_active = crate::tape_forward::tape_scope_active();
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )))]
    let tape_recording_active = false;
    if seq_len > 1
        && !tape_recording_active
        && !any_kt_tensor_tracks_op(&[q, k, v, beta, g, state])
        && let Some(out) =
            GdnBackend::runtime_gdn_chunkwise_forward(backend, q, k, v, beta, g, state, chunk_size)?
    {
        return Ok(out);
    }

    let full_chunks = seq_len / chunk_size;
    let tail = seq_len - full_chunks * chunk_size;

    // Slice full chunks directly. On macOS Metal this avoids the large upfront
    // pre-permute copies that dominated long-prompt GDN recurrence time.
    //
    // On CUDA + BF16, however, the per-chunk `narrow(2, …).contiguous()` chain
    // dominates GDN prefill: each chunk launches 5 strided-copy kernels (one
    // per tensor) plus another for the K^T transpose inside `matmul_prep`, and
    // at full_chunks ≈ 32 × 24 GDN layers that's ~4 800 small kernel launches
    // per prefill. Profiling on RTX 4090 Laptop at seq_len=2042 attributes
    // 22 % of GDN prefill time to `slice_inputs` alone — measured at
    // 114 ms aggregate for the 768 chunk slices. Pre-permuting the five
    // sequence tensors once to chunk-major layout (`[B, num_chunks, nv, C, …]`)
    // turns each per-chunk slice into a stride-free `narrow + squeeze` view:
    // 5 launches per layer instead of 160, with the same byte count moved.
    let pre_permute_chunks = BackendCapabilityQueries::backend_capabilities(backend)
        .gdn
        .chunk_pre_permute_bf16
        .is_native()
        && dtype == DType::BF16
        && full_chunks > 0
        && !any_kt_tensor_tracks_op(&[q, k, v, beta, g])
        && gdn_chunk_pre_permute_policy_enabled(&q.device());

    let pre_permuted: Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Option<Tensor>)> =
        if pre_permute_chunks {
            kiln_nvtx::range!(c"kiln/attn/gdn/chunk_pre_permute");
            let dk = q.dim(3)?;
            let dv = v.dim(3)?;
            let pre_t = full_chunks * chunk_size;
            // View the leading `full_chunks * chunk_size` tokens as
            // `[B, nv, num_chunks, C, …]` then transpose to
            // `[B, num_chunks, nv, C, …]`. The `.contiguous()` pays one
            // memcpy per tensor — the same total bytes as the 32 per-chunk
            // contig copies it replaces, but with 1/32 the launch overhead.
            let q_pre = q
                .narrow(2, 0, pre_t)?
                .reshape((batch, heads, full_chunks, chunk_size, dk))?
                .transpose(1, 2)?
                .contiguous()?;
            let k_pre = k
                .narrow(2, 0, pre_t)?
                .reshape((batch, heads, full_chunks, chunk_size, dk))?
                .transpose(1, 2)?
                .contiguous()?;
            let v_pre = v
                .narrow(2, 0, pre_t)?
                .reshape((batch, heads, full_chunks, chunk_size, dv))?
                .transpose(1, 2)?
                .contiguous()?;
            let beta_pre = beta
                .narrow(2, 0, pre_t)?
                .reshape((batch, heads, full_chunks, chunk_size))?
                .transpose(1, 2)?
                .contiguous()?;
            let g_pre = g
                .narrow(2, 0, pre_t)?
                .reshape((batch, heads, full_chunks, chunk_size))?
                .transpose(1, 2)?
                .contiguous()?;
            // K^T is only needed by the fused full-chunk backend API. Matmul
            // prep uses transposed-GEMM helpers and can consume K in native
            // chunk layout.
            let k_t_pre = if chunk_size == 64
                && GdnBackend::runtime_supports_gdn_full_chunk_forward(backend)
            {
                Some(k_pre.transpose(3, 4)?.contiguous()?)
            } else {
                None
            };
            Some((q_pre, k_pre, v_pre, beta_pre, g_pre, k_t_pre))
        } else {
            None
        };

    let mut out_chunks: Vec<Tensor> = Vec::with_capacity(seq_len.div_ceil(chunk_size));

    for ci in 0..(full_chunks + if tail > 0 { 1 } else { 0 }) {
        let is_tail = ci >= full_chunks;
        let c = if is_tail { tail } else { chunk_size };

        let (q_c, k_c, v_c, beta_c, g_c, k_t_mat_pre) = if is_tail {
            let t_start = full_chunks * chunk_size;
            (
                q.narrow(2, t_start, tail)?.contiguous()?,
                k.narrow(2, t_start, tail)?.contiguous()?,
                v.narrow(2, t_start, tail)?.contiguous()?,
                beta.narrow(2, t_start, tail)?.contiguous()?,
                g.narrow(2, t_start, tail)?.contiguous()?,
                None,
            )
        } else if let Some((q_pre, k_pre, v_pre, beta_pre, g_pre, k_t_pre)) = pre_permuted.as_ref()
        {
            // Slice along the chunk dimension: zero-copy view that's
            // contiguous because num_chunks is the slowest non-batch
            // dimension after the pre-permute.
            let q_c = q_pre.narrow(1, ci, 1)?.squeeze(1)?;
            let k_c = k_pre.narrow(1, ci, 1)?.squeeze(1)?;
            let v_c = v_pre.narrow(1, ci, 1)?.squeeze(1)?;
            let beta_c = beta_pre.narrow(1, ci, 1)?.squeeze(1)?;
            let g_c = g_pre.narrow(1, ci, 1)?.squeeze(1)?;
            let k_t_chunk = k_t_pre
                .as_ref()
                .map(|k_t_pre| k_t_pre.narrow(1, ci, 1)?.squeeze(1))
                .transpose()?;
            (q_c, k_c, v_c, beta_c, g_c, k_t_chunk)
        } else {
            let t_start = ci * chunk_size;
            (
                q.narrow(2, t_start, chunk_size)?.contiguous()?,
                k.narrow(2, t_start, chunk_size)?.contiguous()?,
                v.narrow(2, t_start, chunk_size)?.contiguous()?,
                beta.narrow(2, t_start, chunk_size)?.contiguous()?,
                g.narrow(2, t_start, chunk_size)?.contiguous()?,
                None,
            )
        };
        // Matmuls first — these are well-tuned GEMMs and stay on kt tensors.
        // KKT/QKT and the state update use transposed-GEMM helpers so accelerator
        // backends can avoid materialising K^T. The optional precomputed K^T is
        // retained only for the fused full-chunk backend path, whose API still
        // consumes it directly.
        let ks_entry = k_c.matmul(&*state)?; // [B, nv, C, dv]
        let kkt = kiln_tensor::ops::matmul_rhs_transposed(&k_c, &k_c)?; // [B, nv, C, C]
        let qkt = kiln_tensor::ops::matmul_rhs_transposed(&q_c, &k_c)?; // [B, nv, C, C]
        let q_s = q_c.matmul(&*state)?; // [B, nv, C, dv]
        let full_chunk_out = if !is_tail
            && c == 64
            && GdnBackend::runtime_supports_gdn_full_chunk_forward(backend)
            && dtype == DType::BF16
        {
            if !any_kt_tensor_tracks_op(&[&g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s, &beta_c, state])
            {
                let k_t_mat = match k_t_mat_pre.as_ref() {
                    Some(t) => t.clone(),
                    None => k_c.transpose(2, 3)?.contiguous()?,
                };
                // #1082 DoD-101/102: `gdn_full_chunk_forward` is now kt-typed and
                // mutates `state` in place through the kt `&mut` — pass kt tensors
                // directly, no candle bridge / state write-back.
                GdnBackend::runtime_gdn_full_chunk_forward(
                    backend, &g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s, &beta_c, &k_t_mat, state,
                )?
            } else {
                None
            }
        } else {
            None
        };
        if let Some(out_chunk) = full_chunk_out {
            out_chunks.push(out_chunk);
            continue;
        }

        // Fused prep: cumsum + decay + exp + masked scales + v_prime +
        // q_s_scaled + decay_last_col + p_last in a single CUDA launch.
        // Falls back to the candle-op chain when the backend declines
        // (non-CUDA, non-bf16, envelope violation).
        //
        // Post-conditions on all four paths:
        //   a_strict:         [B, nv, C, C] bf16 — kkt * decay * strict_lower
        //   b_mask:           [B, nv, C, C] bf16 — qkt * decay * causal_lower
        //   v_prime:          [B, nv, C, dv] bf16 — v - ks_entry * p
        //   q_s_scaled:       [B, nv, C, dv] bf16 — q_s * p
        //   decay_last_col_u: [B, nv, C, 1]  bf16 — exp(big_g[C-1] - big_g[i])
        //   p_last_u:         [B, nv, 1, 1]  bf16 — exp(big_g[C-1])
        let (a_strict, b_mask, v_prime, q_s_scaled, decay_last_col_u, p_last_u) = {
            kiln_nvtx::range!(c"kiln/attn/gdn/chunk_prep");
            // #1082 DoD-101/102: `gdn_chunk_prep` is now kt-typed — pass kt
            // tensors directly and consume the 6 kt results without a bridge.
            #[cfg(feature = "cuda")]
            let prep_out = if !any_kt_tensor_tracks_op(&[&g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s])
                && GdnBackend::runtime_supports_gdn_chunk_prep(backend)
                && dtype == DType::BF16
            {
                GdnBackend::runtime_gdn_chunk_prep(
                    backend, &g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s,
                )?
            } else {
                None
            };
            #[cfg(not(feature = "cuda"))]
            let prep_out: Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> = None;
            match prep_out {
                Some((a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last)) => {
                    let decay_last_col_u = decay_last_col.unsqueeze(3)?; // [B,nv,C,1]
                    let p_last_u = p_last.unsqueeze(2)?.unsqueeze(3)?; // [B,nv,1,1]
                    (
                        a_strict.contiguous()?,
                        b_mask.contiguous()?,
                        v_prime,
                        q_s_scaled,
                        decay_last_col_u,
                        p_last_u,
                    )
                }
                None => {
                    // Cumulative decay G[t] = Σ_{s=0..t} g[s].  Done in F32:
                    // exp() of the cumulative sum is the only place bf16
                    // would lose meaningful precision (G can reach -10 or
                    // more across a full 64-token chunk).
                    let g_f32 = g_c.to_dtype(DType::F32)?;
                    let big_g = g_f32.cumsum(LAST_DIM)?; // [B, nv, C], F32

                    // Decay matrix D[t, i] = exp(G[t] - G[i]). Mask before
                    // exp: masked future positions can otherwise overflow to
                    // inf, and inf * 0 is NaN.
                    let big_g_col = big_g.unsqueeze(3)?; // [B, nv, C, 1]
                    let big_g_row = big_g.unsqueeze(2)?; // [B, nv, 1, C]
                    let decay_delta = big_g_col.broadcast_sub(&big_g_row)?;
                    let zero_delta = Tensor::zeros_like(&decay_delta)?;
                    let strict_bool = strict_lower_tri_bool(c, device)?
                        .reshape((1, 1, c, c))?
                        .broadcast_as((batch, heads, c, c))?;
                    let causal_bool = causal_lower_tri_bool(c, device)?
                        .reshape((1, 1, c, c))?
                        .broadcast_as((batch, heads, c, c))?;
                    // Phase 7 (#1082): three clusters of
                    // `.exp()?.to_dtype(dtype)?` in the chunkwise GDN
                    // decay-matrix prep all migrate under the stable KT route
                    // policy. Each leg
                    // (EXP, then TO_DTYPE) falls through to the candle
                    // op when its precondition fails, so behavior is
                    // identical with the gates off. These ops run
                    // per chunkwise prep on every GDN layer; the
                    // bench coverage adds the chunkwise path to the
                    // single-token site already wired (a7864c97).
                    let exp_to_dtype = |t: &Tensor| -> Result<Tensor> {
                        #[cfg(feature = "cuda")]
                        {
                            let exped = match try_kt_exp(t)? {
                                Some(out) => out,
                                None => t.exp()?,
                            };
                            match try_kt_to_dtype(&exped, dtype)? {
                                Some(out) => Ok(out),
                                None => Ok(exped.to_dtype(dtype)?),
                            }
                        }
                        #[cfg(not(feature = "cuda"))]
                        {
                            Ok(t.exp()?.to_dtype(dtype)?)
                        }
                    };
                    let strict_decay =
                        exp_to_dtype(&strict_bool.where_cond(&decay_delta, &zero_delta)?)?;
                    let causal_decay =
                        exp_to_dtype(&causal_bool.where_cond(&decay_delta, &zero_delta)?)?;

                    // p[t] = exp(G[t]).
                    let p = exp_to_dtype(&big_g)?; // [B, nv, C]
                    let p_col = p.unsqueeze(3)?; // [B, nv, C, 1]

                    // #1082: U8→float cast is unsupported on the kt CUDA path
                    // (the unsupported-cast fallback hits the CPU branch on a
                    // CUDA tensor). Build the dtype-typed multiplicative masks via
                    // where_cond (1.0 where set, else 0.0). This forward chunkwise
                    // prep runs when the fused GDN kernels decline (e.g. the F32
                    // tiny model); the BF16 production path uses the fused kernel
                    // and never reaches here.
                    let m_ones = Tensor::ones((batch, heads, c, c), dtype, device)?;
                    let m_zeros = Tensor::zeros((batch, heads, c, c), dtype, device)?;
                    let strict_mask = strict_bool.where_cond(&m_ones, &m_zeros)?;
                    let causal_mask = causal_bool.where_cond(&m_ones, &m_zeros)?;

                    let v_prime = (&v_c - ks_entry.broadcast_mul(&p_col)?)?;
                    let a_strict = kkt
                        .broadcast_mul(&strict_decay)?
                        .broadcast_mul(&strict_mask)?
                        .contiguous()?;
                    let b_mask = qkt
                        .broadcast_mul(&causal_decay)?
                        .broadcast_mul(&causal_mask)?
                        .contiguous()?;
                    let q_s_scaled = q_s.broadcast_mul(&p_col)?;

                    let g_last = big_g.narrow(2, c - 1, 1)?; // [B, nv, 1]
                    // Phase 7 (#1082): wire try_kt_exp + try_kt_to_dtype into
                    // the two remaining `.exp()` sites of the GDN chunkwise
                    // prep — `decay_last_col_u` and `p_last_u`. Mirrors the
                    // strict_decay / causal_decay / p `exp_to_dtype` closure
                    // above so all five chunk-prep exp calls now take the
                    // single-kernel kt-API fast path when stable KT routes are
                    // enabled.
                    let decay_last_col_u = {
                        let g_diff = g_last.broadcast_sub(&big_g)?;
                        exp_to_dtype(&g_diff)?.unsqueeze(3)?
                    }; // [B, nv, C, 1]
                    let p_last_u = exp_to_dtype(&g_last)?.unsqueeze(3)?; // [B,nv,1,1]

                    (
                        a_strict,
                        b_mask,
                        v_prime,
                        q_s_scaled,
                        decay_last_col_u,
                        p_last_u,
                    )
                }
            }
        };
        let decay_last_col = decay_last_col_u.squeeze(3)?;
        let (out_chunk, w_weighted) = {
            kiln_nvtx::range!(c"kiln/attn/gdn/chunk");
            if !any_kt_tensor_tracks_op(&[
                &a_strict,
                &b_mask,
                &v_prime,
                &q_s_scaled,
                &beta_c,
                &decay_last_col,
            ]) && GdnBackend::runtime_supports_gdn_chunk_scan(backend)
                && dtype == DType::BF16
            {
                // #1082 DoD-101/102: `gdn_chunk_scan` is now kt-typed — pass kt
                // tensors directly and consume the kt (out_chunk, w_weighted)
                // pair without a bridge.
                let scan_out = GdnBackend::runtime_gdn_chunk_scan(
                    backend,
                    &a_strict,
                    &b_mask,
                    &v_prime,
                    &q_s_scaled,
                    &beta_c,
                    &decay_last_col,
                )?;
                match scan_out {
                    Some((out_chunk, w_weighted)) => (out_chunk, w_weighted),
                    None => {
                        let w = compute_w_chunk(backend, &a_strict, &v_prime, &beta_c, c)?;
                        let intra = b_mask.matmul(&w)?;
                        let out_chunk = (&q_s_scaled + &intra)?;
                        let w_weighted = w.broadcast_mul(&decay_last_col_u)?.contiguous()?;
                        (out_chunk, w_weighted)
                    }
                }
            } else {
                let w = compute_w_chunk(backend, &a_strict, &v_prime, &beta_c, c)?;
                let intra = b_mask.matmul(&w)?;
                let out_chunk = (&q_s_scaled + &intra)?;
                let w_weighted = w.broadcast_mul(&decay_last_col_u)?.contiguous()?;
                (out_chunk, w_weighted)
            }
        };
        out_chunks.push(out_chunk); // [B, nv, C, dv]

        // State update:
        //   S_new = exp(G[C-1]) * S_entry
        //         + Σ_i exp(G[C-1] - G[i]) * k[i] ⊗ W[i]
        let state_scaled = state.broadcast_mul(&p_last_u)?; // [B, nv, dk, dv]
        let delta_state = kiln_tensor::ops::matmul_lhs_transposed(&k_c, &w_weighted)?; // [B, nv, dk, dv]
        *state = (state_scaled + delta_state)?;
    }

    // Phase 7 (#1082): when stable KT routes are enabled and every chunk-output is
    // a contiguous CUDA tensor of a supported dtype, route the
    // time-axis (axis=2) per-chunk concat through
    // `kiln_tensor::cuda_concat(_, 2)`. Mirrors the conv1d
    // prefill/decode cat_dim2 wirings. Falls through to the
    // candle composite when any precondition fails so behavior
    // is identical with the gate off. This is the
    // gdn_chunkwise_recurrence final assembly step that joins
    // per-chunk outputs into the seq_len-shaped attention
    // output, called once per GDN layer per prefill request.
    let out = {
        let chunk_refs: Vec<&Tensor> = out_chunks.iter().collect();
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_cat_dim2(&chunk_refs)? {
                out
            } else {
                Tensor::cat(&chunk_refs, 2)?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Tensor::cat(&chunk_refs, 2)?
        }
    };
    Ok(out)
}

pub(super) fn gdn_recurrent_prefill_head_last(
    backend: &dyn BackendRuntime,
    q: &Tensor,         // [B, nv, T, dk]
    k: &Tensor,         // [B, nv, T, dk]
    v: &Tensor,         // [B, nv, T, dv]
    beta: &Tensor,      // [B, nv, T]
    g: &Tensor,         // [B, nv, T]
    state: &mut Tensor, // [B, nv, dk, dv]
) -> Result<Option<Tensor>> {
    let (_, _, seq_len, _) = q.dims4()?;
    if seq_len <= 1
        || q.dtype() != DType::BF16
        || state.dtype() != DType::BF16
        || any_kt_tensor_tracks_op(&[q, k, v, beta, g, state])
        || !GdnBackend::runtime_supports_gdn_recurrent_prefill_head_last(backend)
    {
        return Ok(None);
    }
    GdnBackend::runtime_gdn_recurrent_prefill_head_last(backend, q, k, v, beta, g, state)
}

pub(super) fn gdn_recurrent_prefill_native_head_last(
    backend: &dyn BackendRuntime,
    q: &Tensor,         // [B, T, nk, dk]
    k: &Tensor,         // [B, T, nk, dk]
    v: &Tensor,         // [B, T, nv, dv]
    beta: &Tensor,      // [B, T, nv]
    g: &Tensor,         // [B, T, nv]
    state: &mut Tensor, // [B, nv, dk, dv]
) -> Result<Option<Tensor>> {
    let (_, seq_len, _, _) = q.dims4()?;
    if seq_len == 0
        || !matches!(q.dtype(), DType::BF16 | DType::F32)
        || !matches!(state.dtype(), DType::BF16 | DType::F32)
        || any_kt_tensor_tracks_op(&[q, k, v, beta, g, state])
        || !GdnBackend::runtime_supports_gdn_recurrent_prefill_native_head_last(backend)
    {
        return Ok(None);
    }
    GdnBackend::runtime_gdn_recurrent_prefill_native_head_last(backend, q, k, v, beta, g, state)
}

/// Metal BF16 fast path for full 64-token chunks.
///
/// Returns a contiguous head-last `[B, T, nv, dv]` tensor so the caller can feed
/// Metal gated RMSNorm without the `[B,nv,T,dv]` cat + transpose + contiguous
/// copy chain.
pub(super) fn gdn_chunkwise_recurrence_head_last_full_chunks(
    backend: &dyn BackendRuntime,
    q: &Tensor,         // [B, nv, T, dk]
    k: &Tensor,         // [B, nv, T, dk]
    v: &Tensor,         // [B, nv, T, dv]
    beta: &Tensor,      // [B, nv, T]
    g: &Tensor,         // [B, nv, T]
    state: &mut Tensor, // [B, nv, dk, dv]
    chunk_size: usize,
) -> Result<Option<Tensor>> {
    let (batch, heads, seq_len, _) = q.dims4()?;
    let dtype = q.dtype();
    if chunk_size != 64
        || seq_len <= 1
        || seq_len % chunk_size != 0
        || dtype != DType::BF16
        || state.dtype() != DType::BF16
        || any_kt_tensor_tracks_op(&[q, k, v, beta, g, state])
        || !GdnBackend::runtime_supports_gdn_full_chunk_forward_head_last(backend)
    {
        return Ok(None);
    }

    let dv = v.dim(3)?;
    // Full chunks cover the whole sequence and the Metal kernel writes every
    // head-last output element exactly once.
    let out = unsafe { Tensor::empty((batch, seq_len, heads, dv), DType::BF16, q.device())? };

    for ci in 0..(seq_len / chunk_size) {
        let t_start = ci * chunk_size;
        let q_c = q.narrow(2, t_start, chunk_size)?.contiguous()?;
        let k_c = k.narrow(2, t_start, chunk_size)?.contiguous()?;
        let v_c = v.narrow(2, t_start, chunk_size)?.contiguous()?;
        let beta_c = beta.narrow(2, t_start, chunk_size)?.contiguous()?;
        let g_c = g.narrow(2, t_start, chunk_size)?.contiguous()?;

        let k_t_mat = k_c.transpose(2, 3)?.contiguous()?; // [B, nv, dk, C]
        let ks_entry = k_c.matmul(&*state)?; // [B, nv, C, dv]
        let kkt = k_c.matmul(&k_t_mat)?; // [B, nv, C, C]
        let qkt = q_c.matmul(&k_t_mat)?; // [B, nv, C, C]
        let q_s = q_c.matmul(&*state)?; // [B, nv, C, dv]

        if !GdnBackend::runtime_gdn_full_chunk_forward_head_last_into(
            backend, &g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s, &beta_c, &k_t_mat, state, &out,
            t_start, seq_len,
        )? {
            if ci == 0 {
                return Ok(None);
            }
            anyhow::bail!("backend declined GDN head-last full-chunk path mid-sequence");
        }
    }

    Ok(Some(out))
}

/// Gated DeltaNet (GDN) linear attention forward pass.
///
/// Implements the recurrent linear attention mechanism used by 24/32 layers in Qwen3.5-4B.
/// Uses data-dependent gating (alpha/beta) and a delta rule update for the recurrent state.
///
/// `x`: [batch, seq_len, hidden_size]
/// `weights`: linear attention projection weights
/// `config`: model configuration
/// `recurrent_state`: [batch, nv, dk, dv] — mutable recurrent state, updated in place
/// `conv_state`: [batch, conv_dim, kernel_size-1] — mutable conv buffer, updated in place
///
/// Returns: [batch, seq_len, hidden_size]

/// Candle-op reference path for the Step-6 GDN gates. This is the original
/// Phase-6 implementation; it's kept as a fallback for shapes/dtypes outside
/// the fused kernel's envelope and as the algorithmic oracle for parity tests.
///
/// beta = sigmoid(b)                                // bf16
/// g    = -exp(A_log) * softplus(a + dt_bias)       // bf16 (F32 intermediates)
pub(super) fn gated_deltanet_gates_fallback(
    a: &Tensor,
    b: &Tensor,
    weights: &GpuLinearAttentionWeights,
    input_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let gate_device = a.device();
    let a_log = if weights.a_log.device() != gate_device {
        weights.a_log.to_device(gate_device)?
    } else {
        weights.a_log.clone()
    };
    let dt_bias = if weights.dt_bias.device() != gate_device {
        weights.dt_bias.to_device(gate_device)?
    } else {
        weights.dt_bias.clone()
    };
    let beta = cuda_sigmoid(b).context("gdn gates fallback beta cuda_sigmoid")?; // [B, T, nv], bf16
    let a_f32 = a
        .to_dtype(DType::F32)
        .context("gdn gates fallback a to f32")?;
    let a_log_f32 = a_log
        .to_dtype(DType::F32)
        .context("gdn gates fallback a_log to f32")?;
    let dt_bias_f32 = dt_bias
        .to_dtype(DType::F32)
        .context("gdn gates fallback dt_bias to f32")?;
    let g = {
        let a_biased = a_f32
            .broadcast_add(&dt_bias_f32)
            .context("gdn gates fallback broadcast_add dt_bias")?;
        let sp = softplus(&a_biased).context("gdn gates fallback softplus")?;
        // Phase 7 (#1082): when stable KT routes are enabled and `a_log_f32` is a
        // contiguous CUDA tensor of a supported dtype, route the
        // `a_log_f32.exp()` step through
        // `kiln_tensor::cuda_activation_unary` with kind 6 (Exp) via
        // the kt-bridge borrow adapter. Falls through to candle's
        // `.exp()` on any precondition failure.
        //
        // Same fallback path as the `.neg()` migration immediately below.
        // Wiring the exp step in addition to the neg step
        // means the full `-exp(A_log)` decay computation uses the stable KT
        // route set.
        let a_log_exp = {
            #[cfg(feature = "cuda")]
            {
                if let Some(out) =
                    try_kt_exp(&a_log_f32).context("gdn gates fallback try_kt_exp")?
                {
                    out
                } else {
                    a_log_f32.exp().context("gdn gates fallback a_log exp")?
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                a_log_f32.exp().context("gdn gates fallback a_log exp")?
            }
        };
        // Phase 7 (#1082): when stable KT routes are enabled and `a_log_exp` is a
        // contiguous CUDA tensor of a supported dtype, route
        // `a_log_exp.neg()` through `kiln_tensor::cuda_activation_unary`
        // with kind 12 (Neg) via the kt-bridge borrow adapter. Falls
        // through to candle's `.neg()` on any precondition failure.
        //
        // This is the GDN gates fallback path. Wiring here exercises the
        // kt-API on a less-
        // trafficked code path that mirrors the fused kernel's
        // -exp(A_log) decay computation in candle composites, so kt-API
        // parity coverage extends to the parity baseline itself.
        let neg_decay = {
            #[cfg(feature = "cuda")]
            {
                if let Some(out) =
                    try_kt_neg(&a_log_exp).context("gdn gates fallback try_kt_neg")?
                {
                    out
                } else {
                    a_log_exp.neg().context("gdn gates fallback a_log neg")?
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                a_log_exp.neg().context("gdn gates fallback a_log neg")?
            }
        }; // -exp(A_log)
        sp.broadcast_mul(&neg_decay)
            .context("gdn gates fallback broadcast_mul neg_decay")?
    }
    .to_dtype(input_dtype)
    .context("gdn gates fallback output to input dtype")?;

    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        return require_active_tape_output(
            crate::tape_forward::try_tape_gdn_gates_kt(
                a,
                b,
                &weights.a_log,
                &weights.dt_bias,
                &beta,
                &g,
            )
            .context("gdn gates fallback try_tape_gdn_gates_kt")?,
            "GDN gate transforms",
        );
    }
    Ok((beta, g))
}

pub fn gated_deltanet_forward(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
    recurrent_state: &mut Tensor,
    conv_state: &mut Tensor,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    gated_deltanet_forward_decode_if(
        backend,
        x,
        weights,
        config,
        recurrent_state,
        conv_state,
        true,
        false,
        true,
        true,
        lora,
    )
}

/// GDN attention subblock through its residual add, excluding the following
/// MLP. Used by exact training-time split backprop to keep the recurrent GDN
/// graph separate from the MLP graph while preserving full-context state.
pub fn gdn_attention_residual_block(
    backend: &dyn BackendRuntime,
    hidden: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    recurrent_state: &mut Tensor,
    conv_state: &mut Tensor,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let GpuAttentionWeights::Linear(lin_weights) = &layer.attention else {
        anyhow::bail!("gdn_attention_residual_block called on a non-GDN layer");
    };
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_attn");
        rms_norm(hidden, &layer.input_layernorm, config.rms_norm_eps)?
    };
    let attn_out = gated_deltanet_forward(
        backend,
        &normed,
        lin_weights,
        config,
        recurrent_state,
        conv_state,
        lora,
    )?;
    (hidden + &attn_out).map_err(Into::into)
}

/// Pre-attention RMSNorm for a GDN layer.
pub fn gdn_attention_input_norm(
    hidden: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    rms_norm(hidden, &layer.input_layernorm, config.rms_norm_eps)
}

/// Input-projection outputs for the GDN attention subblock.
pub struct GdnInputProjectionParts {
    pub mixed_qkv: Tensor,
    pub z: Tensor,
    pub a: Tensor,
    pub b: Tensor,
}

pub fn gdn_attention_in_projections(
    backend: &dyn BackendRuntime,
    normed: &Tensor,
    weights: &GpuLinearAttentionWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<GdnInputProjectionParts> {
    let (lora_layer, lora_scale) = match lora {
        Some((layer, scale)) => (Some(layer), scale),
        None => (None, 0.0),
    };
    let mixed_qkv = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        normed,
        &weights.in_proj_qkv_t,
        lora_layer.and_then(|layer| layer.in_proj_qkv.as_ref()),
        lora_scale,
    )?;
    let z = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        normed,
        &weights.in_proj_z_t,
        lora_layer.and_then(|layer| layer.in_proj_z.as_ref()),
        lora_scale,
    )?;
    let a = gdn_in_proj_matmul(backend, normed, &weights.in_proj_a_t)?;
    let b = gdn_in_proj_matmul(backend, normed, &weights.in_proj_b_t)?;
    Ok(GdnInputProjectionParts { mixed_qkv, z, a, b })
}

/// Q/K/V tensors in `[B, nv, T, *]` layout after causal conv, GQA expansion,
/// and Q/K L2 normalization.
pub struct GdnQkvParts {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

pub fn gdn_qkv_from_mixed_training(
    _backend: &dyn BackendRuntime,
    mixed_qkv: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
    conv_state: &mut Tensor,
) -> Result<GdnQkvParts> {
    let (batch, seq_len, _) = mixed_qkv.dims3()?;
    let input_dtype = mixed_qkv.dtype();
    let nk = config.linear_num_key_heads;
    let dk = config.linear_key_head_dim;
    let nv = config.linear_num_value_heads;
    let dv = config.linear_value_head_dim;
    let qk_dim = config.linear_qk_dim();
    let v_dim = config.linear_v_dim();
    let kernel_size = config.linear_conv_kernel_dim;
    let gqa_ratio = nv / nk;
    let scale = 1.0 / (dk as f64).sqrt();

    let mixed_qkv_ct = mixed_qkv.transpose(1, 2)?.contiguous()?;
    let post_silu = if seq_len > 1 {
        let y = causal_conv1d_prefill(&mixed_qkv_ct, &weights.conv1d, conv_state, kernel_size)?;
        cuda_silu(&y)?
    } else {
        let y = causal_conv1d_decode(&mixed_qkv_ct, &weights.conv1d, conv_state, kernel_size)?;
        cuda_silu(&y.to_dtype(DType::F32)?)?
    };
    let mixed_qkv = post_silu.transpose(1, 2)?;
    let q = mixed_qkv
        .narrow(2, 0, qk_dim)?
        .reshape((batch, seq_len, nk, dk))?;
    let k = mixed_qkv
        .narrow(2, qk_dim, qk_dim)?
        .reshape((batch, seq_len, nk, dk))?;
    let v = mixed_qkv
        .narrow(2, 2 * qk_dim, v_dim)?
        .reshape((batch, seq_len, nv, dv))?
        .to_dtype(input_dtype)?;
    let (q, k) = if gqa_ratio > 1 {
        let q = q
            .unsqueeze(3)?
            .expand([batch, seq_len, nk, gqa_ratio, dk])?
            .contiguous()?
            .reshape((batch, seq_len, nv, dk))?;
        let k = k
            .unsqueeze(3)?
            .expand([batch, seq_len, nk, gqa_ratio, dk])?
            .contiguous()?
            .reshape((batch, seq_len, nv, dk))?;
        (q, k)
    } else {
        (q.contiguous()?, k.contiguous()?)
    };
    let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
    Ok(GdnQkvParts {
        q: q.transpose(1, 2)?,
        k: k.transpose(1, 2)?,
        v: v.transpose(1, 2)?,
    })
}

pub fn gdn_recurrent_forward_from_parts(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    recurrent_state: &mut Tensor,
) -> Result<Tensor> {
    let input_dtype = q.dtype();
    let state_external_dtype = recurrent_state.dtype();
    let recurrence_dtype = if state_external_dtype == DType::F32 {
        DType::F32
    } else {
        input_dtype
    };
    if state_external_dtype != recurrence_dtype {
        let normalized_state = recurrent_state.to_dtype(recurrence_dtype)?;
        replace_gdn_recurrent_state_handle(
            backend,
            recurrent_state,
            normalized_state,
            "normalize recurrent dtype",
        )?;
    }
    let q = if q.dtype() == recurrence_dtype {
        q.clone()
    } else {
        q.to_dtype(recurrence_dtype)?
    };
    let k = if k.dtype() == recurrence_dtype {
        k.clone()
    } else {
        k.to_dtype(recurrence_dtype)?
    };
    let v = if v.dtype() == recurrence_dtype {
        v.clone()
    } else {
        v.to_dtype(recurrence_dtype)?
    };
    let beta = if beta.dtype() == recurrence_dtype {
        beta.clone()
    } else {
        beta.to_dtype(recurrence_dtype)?
    };
    let g = if g.dtype() == recurrence_dtype {
        g.clone()
    } else {
        g.to_dtype(recurrence_dtype)?
    };

    let (out, head_last) = if let Some(attn_out) =
        gdn_recurrent_prefill_head_last(backend, &q, &k, &v, &beta, &g, recurrent_state)?
    {
        (attn_out, true)
    } else {
        match gdn_chunkwise_recurrence_head_last_full_chunks(
            backend,
            &q,
            &k,
            &v,
            &beta,
            &g,
            recurrent_state,
            GDN_CHUNK_SIZE,
        )? {
            Some(attn_out) => (attn_out, true),
            None => (
                gdn_chunkwise_recurrence(
                    backend,
                    &q,
                    &k,
                    &v,
                    &beta,
                    &g,
                    recurrent_state,
                    GDN_CHUNK_SIZE,
                )?,
                false,
            ),
        }
    };

    if state_external_dtype != recurrence_dtype {
        let external_state = recurrent_state.to_dtype(state_external_dtype)?;
        replace_gdn_recurrent_state_handle(
            backend,
            recurrent_state,
            external_state,
            "restore external recurrent dtype",
        )?;
    }

    if head_last {
        Ok(out.transpose(1, 2)?)
    } else {
        Ok(out)
    }
}

pub fn gdn_gated_norm_from_recurrent(
    backend: &dyn BackendRuntime,
    recurrent_out_head_major: &Tensor,
    z: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    let (batch, heads, seq_len, dv) = recurrent_out_head_major.dims4()?;
    let attn_out = recurrent_out_head_major.transpose(1, 2)?;
    let z = z.reshape((batch, seq_len, heads, dv))?;
    Ok(
        gated_rms_norm(backend, &attn_out, &z, &weights.norm, config.rms_norm_eps)?
            .reshape((batch, seq_len, heads * dv))?
            .to_dtype(z.dtype())?,
    )
}

pub fn gdn_out_proj_from_gated_norm(
    backend: &dyn BackendRuntime,
    normed: &Tensor,
    weights: &GpuLinearAttentionWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (lora_layer, lora_scale) = match lora {
        Some((layer, scale)) => (Some(layer), scale),
        None => (None, 0.0),
    };
    mlp_proj_forward_decode_if(
        Some(backend),
        false,
        normed,
        &weights.out_proj_t,
        weights.out_proj_marlin.as_ref(),
        lora_layer.and_then(|layer| layer.gdn_out_proj.as_ref()),
        lora_scale,
    )
}
