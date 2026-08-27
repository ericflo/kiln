use super::*;

// ---------------------------------------------------------------------------
// Forward pass primitives
// ---------------------------------------------------------------------------

/// Look up token embeddings from the embedding table.
///
/// `token_ids`: 1-D slice of token IDs.
/// `embed_weights`: [vocab_size, hidden_size] embedding matrix.
///
/// Returns: [seq_len, hidden_size] tensor.
///
/// Phase 7 (#1082): when stable KT routes are enabled and inputs are contiguous CUDA
/// tensors of a supported dtype, route the dim-0 `index_select`
/// through `kiln_tensor::cuda_index_select_dim0`. Falls through to the
/// kt `index_select` op when any precondition fails.
pub fn embedding_lookup(token_ids: &[u32], embed_weights: &Tensor) -> Result<Tensor> {
    // #1082: kt 1-D index tensor on the weights' device (no candle `Tensor::new`).
    let index = Tensor::from_vec_on(
        embed_weights.device(),
        token_ids.to_vec(),
        vec![token_ids.len()],
    )?;
    // When a thread-local `Tape` scope is active,
    // route the lookup through the kt op-registry as an intentional frozen
    // leaf. Adapter training differentiates neither token ids nor the resident
    // embedding table, so recording an EmbeddingBackward would allocate a full
    // `[vocab, hidden]` gradient that no optimizer can consume. Downstream tape
    // nodes still consume the gathered activation id normally.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        let out = require_active_tape_output(
            crate::tape_forward::try_tape_frozen_embedding_kt(embed_weights, &index)
                .context("embedding_lookup try_tape_frozen_embedding_kt")?,
            "token embedding lookup",
        )?;
        return promote_cpu_activation(out);
    }
    #[cfg(feature = "cuda")]
    if let Some(out) = try_kt_embedding_lookup(embed_weights, &index)? {
        return promote_cpu_activation(out);
    }
    let out = embed_weights.index_select(&index, 0)?;
    promote_cpu_activation(out)
}

pub(super) fn embedding_lookup_with_index(
    index: &Tensor,
    embed_weights: &Tensor,
) -> Result<Tensor> {
    // Mirror `embedding_lookup` for the index-already-built path used by
    // batched decode and prefill. Decode has no scope and reaches the backend
    // fast path; training establishes the gathered activation as a frozen leaf
    // on every GPU backend.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        let out = require_active_tape_output(
            crate::tape_forward::try_tape_frozen_embedding_kt(embed_weights, index)
                .context("embedding_lookup_with_index try_tape_frozen_embedding_kt")?,
            "indexed token embedding lookup",
        )?;
        return promote_cpu_activation(out);
    }
    #[cfg(feature = "cuda")]
    if let Some(out) = try_kt_embedding_lookup(embed_weights, index)? {
        return promote_cpu_activation(out);
    }
    promote_cpu_activation(embed_weights.index_select(index, 0)?)
}

/// Phase 7 (#1082) — **kt-native** embedding-lookup core.
///
/// Operates entirely on `KtTensor` storage: given the token-embedding
/// table and the U32 index already borrowed as kt tensors, performs the
/// dim-0 gather via `kiln_tensor::cuda_index_select_dim0` and returns
/// the gathered rows as a `KtTensor` (no candle in the signature). This
/// is the consolidated kt-internal computation for the embedding region;
/// the hidden-state and weight-side seams
/// ([`try_kt_embedding_lookup`] and the
/// [`GpuWeights::embed_tokens_kt`] accessor) are thin identity shims now
/// that both sides are kt.
///
/// Bit-exact memcpy gather (no arithmetic, no reordering) — identical
/// byte output to the `index_select` op. The `kiln/embedding_kt`
/// NVTX range is opened by the calling wrappers
/// ([`try_kt_embedding_lookup`] and [`try_kt_embedding_lookup_from_weights`])
/// so it brackets the full migrated computation (borrow-in + gather +
/// copy-out), matching the pre-refactor trace shape; this core does not
/// open its own range to avoid a nested duplicate.
#[cfg(feature = "cuda")]
pub(super) fn kt_embedding_lookup_native(
    embed_weights_kt: &KtTensor,
    index_kt: &KtTensor,
) -> Result<KtTensor> {
    kiln_tensor::cuda_index_select_dim0(embed_weights_kt, index_kt)
        .map_err(|e| anyhow::anyhow!("kt_embedding_lookup_native: index_select failed: {e}"))
}

/// Phase 7 (#1082) — kt-API embedding-lookup seam wrapper.
/// Takes the kt `embed_weights` + `index` (identity alias — both sides kt),
/// runs the kt-native gather ([`kt_embedding_lookup_native`]), and returns
/// the kt `Tensor` output so the public
/// [`embedding_lookup`] / [`embedding_lookup_with_index`] signatures stay
/// kt-typed. The seams are: borrow-in (weight +
/// index) and the identity copy-out (gathered rows).
///
/// Returns `Ok(None)` on any incompatibility (gate off, non-CUDA,
/// non-contiguous, unsupported dtype, indices not U32) so the
/// caller falls through to the kt `index_select` op.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_embedding_lookup(
    embed_weights: &Tensor,
    index: &Tensor,
) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    if !matches!(embed_weights.device(), Device::Cuda(_))
        || !matches!(index.device(), Device::Cuda(_))
        || !embed_weights.is_contiguous()
        || !index.is_contiguous()
        || index.dtype() != DType::U32
    {
        return Ok(None);
    }
    if !matches!(embed_weights.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/embedding_kt");

    // kt seam (weight + index borrow-in; identity alias now).
    // kt-internal computation.
    let out_kt = match kt_embedding_lookup_native(embed_weights, index) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    // kt seam (gathered rows, identity copy-out).
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — weight-aware kt-native embedding lookup at the
/// embedding→layer0 seam. Borrows the token-embedding table through
/// [`GpuWeights::embed_tokens_kt`] (the weight-side accessor)
/// and the U32 index (identity alias, both sides kt), runs the kt-native gather
/// ([`kt_embedding_lookup_native`]), and returns the gathered rows as kt so
/// the kt transformer layers consume an
/// unchanged kt `Tensor`.
///
/// This is the consolidated kt-internal embedding path used by the
/// production `embedding_lookup_from_weights*` callers (the non-stub
/// `[vocab, hidden]` layout). Returns `Ok(None)` on any
/// incompatibility (gate off, tape scope active, non-CUDA,
/// non-contiguous, unsupported dtype, indices not U32) so the caller
/// falls through to the kt `embedding_lookup*` path. The tape path
/// is intentionally NOT handled here — when a tape scope is active the
/// caller's `embedding_lookup*` entry establishes the gathered activation as
/// an intentional frozen tape leaf; we defer to it by returning `Ok(None)` so
/// the training ownership contract remains explicit.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_embedding_lookup_from_weights(
    index: &Tensor,
    weights: &GpuWeights,
) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    // Defer to the recording entry whenever a tape scope is active. Outside a
    // scope inference retains this weight-aware fast path.
    if crate::tape_forward::tape_scope_active() {
        return Ok(None);
    }
    // #1082 forward-flip: `weights.embed_tokens` is now a kt field; `index`
    // is kt too. Gate with kt Device/DType.
    if !matches!(weights.embed_tokens.device(), Device::Cuda(_))
        || !matches!(index.device(), Device::Cuda(_))
        || !weights.embed_tokens.is_contiguous()
        || !index.is_contiguous()
        || index.dtype() != DType::U32
        || !matches!(
            weights.embed_tokens.dtype(),
            DType::F32 | DType::BF16 | DType::F16
        )
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/embedding_kt");

    // Weight side via the kt accessor; index is already kt (no borrow).
    let w_kt = match weights.embed_tokens_kt() {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    // kt-internal computation; return the kt result directly (no copy-back).
    let out_kt = match kt_embedding_lookup_native(&w_kt, index) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    promote_cpu_activation(out_kt).map(Some)
}

pub(super) fn raw_embedding_lookup_from_weights(
    token_ids: &[u32],
    weights: &GpuWeights,
) -> Result<Tensor> {
    let t_dims = weights.embed_tokens_t.dims();
    if t_dims.len() == 2 {
        let expected_embed_dims = [t_dims[1], t_dims[0]];
        if weights.embed_tokens.dims() != expected_embed_dims.as_slice() {
            return embedding_lookup_from_transposed(token_ids, &weights.embed_tokens_t);
        }
    }
    // Non-stub `[vocab, hidden]` layout: route the gather through the
    // weight-aware kt-native path (using `embed_tokens_kt()`) when
    // eligible; fall through to the kt `embedding_lookup` otherwise.
    #[cfg(feature = "cuda")]
    {
        let index = Tensor::from_vec_on(
            weights.embed_tokens.device(),
            token_ids.to_vec(),
            vec![token_ids.len()],
        )?;
        if let Some(out) = try_kt_embedding_lookup_from_weights(&index, weights)? {
            return Ok(out);
        }
    }
    embedding_lookup(token_ids, &weights.embed_tokens)
}

/// (#1443 step 3) Mixed-precision activation cast after embedding lookup.
/// Backend precision policy owns whether an embedding output dtype should be
/// promoted before transformer layers. Today this preserves Vulkan's F32
/// activation / BF16 base-weight envelope while leaving equal-dtype CUDA/Metal
/// paths untouched.
pub(super) fn cast_embedding_output_to_policy_activation(hidden: Tensor) -> Result<Tensor> {
    let precision_policy = crate::backend::training_precision_policy_for_device_kt(hidden.device());
    let target_dtype = if cfg!(feature = "vulkan") {
        precision_policy.activation_dtype_for_embedding_output(hidden.dtype())
    } else {
        hidden.dtype()
    };
    if target_dtype != hidden.dtype() {
        return Ok(hidden.to_dtype(target_dtype)?);
    }
    Ok(hidden)
}

/// Weight-aware embedding lookup at the model activation boundary.
///
/// Keep the precision-policy cast here rather than at forward call sites so
/// paged prefill, batched decode, graph-stable decode, and MTP cannot drift.
pub(super) fn embedding_lookup_from_weights(
    token_ids: &[u32],
    weights: &GpuWeights,
) -> Result<Tensor> {
    cast_embedding_output_to_policy_activation(raw_embedding_lookup_from_weights(
        token_ids, weights,
    )?)
}

pub(super) fn raw_embedding_lookup_from_weights_with_index(
    index: &Tensor,
    weights: &GpuWeights,
) -> Result<Tensor> {
    let t_dims = weights.embed_tokens_t.dims();
    if t_dims.len() == 2 {
        let expected_embed_dims = [t_dims[1], t_dims[0]];
        if weights.embed_tokens.dims() != expected_embed_dims.as_slice() {
            return embedding_lookup_from_transposed_index(index, &weights.embed_tokens_t);
        }
    }

    // Non-stub `[vocab, hidden]` layout: route the gather through the
    // weight-aware kt-native path (using `embed_tokens_kt()`) when
    // eligible; fall through to the kt `embedding_lookup_with_index`.
    #[cfg(feature = "cuda")]
    if let Some(out) = try_kt_embedding_lookup_from_weights(index, weights)? {
        return Ok(out);
    }

    embedding_lookup_with_index(index, &weights.embed_tokens)
}

pub(super) fn embedding_lookup_from_weights_with_index(
    index: &Tensor,
    weights: &GpuWeights,
) -> Result<Tensor> {
    cast_embedding_output_to_policy_activation(raw_embedding_lookup_from_weights_with_index(
        index, weights,
    )?)
}

pub(super) fn embedding_lookup_from_transposed(
    token_ids: &[u32],
    embed_tokens_t: &Tensor,
) -> Result<Tensor> {
    let index = Tensor::from_vec_on(
        embed_tokens_t.device(),
        token_ids.to_vec(),
        vec![token_ids.len()],
    )?;
    embedding_lookup_from_transposed_index(&index, embed_tokens_t)
}

pub(super) fn embedding_lookup_from_transposed_index(
    index: &Tensor,
    embed_tokens_t: &Tensor,
) -> Result<Tensor> {
    let gathered = embed_tokens_t.index_select(index, 1)?;
    promote_cpu_activation(gathered.t()?.contiguous()?)
}

/// RMSNorm: x * weight / sqrt(mean(x^2) + eps).
///
/// `x`: [..., hidden_size]
/// `weight`: `[hidden_size`] (learnable scale)
/// `eps`: small constant for numerical stability (1e-6 for Qwen3.5-4B)
///
/// Returns: same shape as `x`.
///
/// An active tape scope is the training-routing authority on every GPU
/// backend. It records either the fused kt RMSNorm (CUDA/ROCm, when supported)
/// or the portable kt composite and fails closed if neither recorder accepts
/// the tensors. Forward-only backend kernels and their legacy diagnostic
/// switches are considered only when no tape scope is active.
///
/// CPU tensors and GPU tensors outside a native inference-kernel envelope use
/// `rms_norm_fallback`.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        if kiln_rmsnorm_kernel::supports_rmsnorm_kt(x, weight) {
            let out = crate::tape_forward::with_active_tape(|tape| {
                kiln_rmsnorm_kernel::fused_rmsnorm_frozen_weight_via_kt_tape(
                    x, weight, eps as f32, tape,
                )
            })
            .context("active tape scope vanished during RMSNorm")?
            .map_err(|e| anyhow::anyhow!("fused_rmsnorm_frozen_weight_via_kt_tape: {e}"))
            .context("RMSNorm frozen-weight fused kt-tape forward failed")?;
            return Ok(out);
        }

        let out = crate::tape_forward::try_tape_rms_norm_kt(x, weight, eps as f32)
            .context("RMSNorm composite tape forward failed")?
            .context("active tape scope could not record RMSNorm")?;
        return Ok(out);
    }

    #[cfg(feature = "cuda")]
    {
        let cuda_policy = crate::cuda_policy::current_cuda_kernel_policy();
        if cuda_policy.fused_rmsnorm && cuda_policy.rmsnorm_backward {
            if !x.track_op() && !weight.track_op() {
                if let (Some(x_kt), Some(w_kt)) =
                    (try_borrow_kt_cuda(x), try_borrow_kt_cuda(weight))
                {
                    if kiln_rmsnorm_kernel::supports_rmsnorm_kt(&x_kt, &w_kt) {
                        // Phase 7 (#1082): kt-only. Bit-exact: bottoms out in
                        // the same `kiln_fused_rmsnorm` FFI symbol. Result is kt
                        // — return it directly (no candle round-trip).
                        let out_kt =
                            kiln_rmsnorm_kernel::fused_rmsnorm_kt(&x_kt, &w_kt, eps as f32)
                                .map_err(|e| anyhow::anyhow!("kt fused_rmsnorm: {e}"))?;
                        return Ok(out_kt);
                    }
                }
            }
            // #1082 DoD-100 (step 3a): the candle `fused_rmsnorm_via_kt_forward_op`
            // fallback arm was deleted here. It was reached only as a last resort
            // With no tape scope, envelope misses fall through to the portable
            // forward below. Active training returned through a recorder before
            // consulting these inference kill switches.
        }
    }
    #[cfg(feature = "metal")]
    {
        if !x.track_op()
            && !weight.track_op()
            && crate::backend::metal::metal_rms_norm_supports(x, weight)
        {
            return crate::backend::metal::metal_rms_norm_bf16(x, weight, eps as f32)
                .context("metal rms_norm kernel failed");
        }
    }
    // Vulkan inference path: leaf-fast forward kernel. Skipped for
    // autograd-tracked tensors because the leaf would drop the
    // gradient — the autograd-safe CustomOp1 wrapper below handles
    // those instead (when its separate opt-in is set).
    #[cfg(feature = "vulkan")]
    if vulkan_rmsnorm_forward_inference_enabled()
        && crate::backend::vulkan_active()
        && matches!(x.device(), Device::Cpu)
        && matches!(weight.device(), Device::Cpu)
        && weight.is_contiguous()
        && !weight.track_op()
        && !x.track_op()
        && let Some(out) = try_vulkan_rmsnorm_forward(x, weight, eps as f32)?
    {
        return Ok(out);
    }
    // Training returned through the authoritative tape recorder above.
    // Resident Vulkan RMSNorm. The raw `qwen_rmsnorm_forward`
    // shader passes its narrow kernel parity test, but the kt wrapper path has
    // produced non-finite rows on real model tensors under lavapipe/RADV soak
    // setups. Keep correctness as the default by falling through to
    // `rms_norm_fallback`; the portable policy keeps the separate bridged
    // and generic-tensor leaf disabled while this resident route remains on.
    #[cfg(feature = "vulkan")]
    if crate::backend::vulkan_active()
        && matches!(x.device(), Device::Vulkan(_))
        && !x.track_op()
        && !weight.track_op()
    {
        let in_dtype = x.dtype();
        let x32 = x.contiguous()?.to_dtype(DType::F32)?;
        let w32 = weight
            .to_device(x.device())?
            .contiguous()?
            .to_dtype(DType::F32)?;
        // `vulkan_rmsnorm_last_axis` is the kt substrate op for standard
        // RMSNorm (`x * w / rms`) and adapts that to the Qwen shader by
        // staging `w - 1`. The model-level RMSNorm contract is Qwen's
        // unit-offset form, so pass the effective scale (`1 + weight`) into
        // the substrate wrapper to preserve the same math as
        // `rms_norm_fallback` and `try_tape_rms_norm_kt`.
        let w_plus_one = (w32.ones_like()? + w32)?;
        let out32 = kiln_tensor::vulkan_rmsnorm_last_axis(&x32, &w_plus_one, eps as f32)?;
        return if in_dtype == DType::F32 {
            Ok(out32)
        } else {
            Ok(out32.to_dtype(in_dtype)?)
        };
    }
    // ROCm inference path: route through the native fused RMSNorm kernel
    // (`kiln_rmsnorm_kernel::fused_rmsnorm_kt`, R.7) — same FFI shape as the
    // CUDA kt-only branch above. Without this ROCm fell through to
    // `rms_norm_fallback`, whose `ones_like(weight)` builds a [hidden] CPU
    // tensor and `to_device`s it on EVERY call: ~2 H2D syncs per layer
    // (input + post-attn norm), which was the dominant remaining per-token
    // host round-trip (`[2560]` was ~60% of all H2D in the decode profile).
    // Forward-only (gated on !track_op); training uses the kt-tape branch above.
    #[cfg(feature = "rocm")]
    if !x.track_op()
        && !weight.track_op()
        && matches!(x.device(), Device::Rocm(_))
        && rocm_fused_rmsnorm_allowed_for_tensor(x)
        && x.is_contiguous()
        && weight.is_contiguous()
        && kiln_rmsnorm_kernel::supports_rmsnorm_kt(x, weight)
    {
        return kiln_rmsnorm_kernel::fused_rmsnorm_kt(x, weight, eps as f32)
            .map_err(|e| anyhow::anyhow!("rocm kt fused_rmsnorm: {e}"))
            .context("rms_norm rocm kernel failed");
    }
    rms_norm_fallback(x, weight, eps)
}

#[cfg(feature = "rocm")]
pub(super) fn rocm_fused_rmsnorm_allowed_for_tensor(x: &Tensor) -> bool {
    let _ = x;
    // The ROCm kt RMSNorm path row-tiles long inputs internally
    // (`fused_rmsnorm_kt_rocm_row_tiled`), so the immutable profile needs no
    // shape-dependent override for long rows.
    crate::rocm_policy::current_rocm_kernel_policy().fused_rmsnorm
}

/// The unsafe CPU-bridged Vulkan RMSNorm leaf is disabled by device-neutral policy.
#[cfg(feature = "vulkan")]
pub(super) fn vulkan_rmsnorm_forward_inference_enabled() -> bool {
    kiln_vulkan_kernel::kernels::vulkan_kernel_policy().bridged_rmsnorm_forward_enabled
}

#[cfg(feature = "vulkan")]
pub(super) fn vulkan_resident_decode_enabled() -> bool {
    kiln_vulkan_kernel::kernels::vulkan_kernel_policy().resident_decode_enabled
}

/// File-private kt⇔bytes helpers — migrated inline from
/// `kiln_vulkan_kernel::kernels::{extract_tensor_bytes, create_tensor_from_data}`
/// as part of issue #1082 (drop candle from kiln-vulkan-kernel).
///
/// #1082 vulkan candle removal: the vulkan feature no longer pulls in
/// `candle-core`, so these helpers operate directly on the kt
/// `kiln_tensor::Tensor` (which is CPU-resident at the kt<->vk seam — see
/// `loader_kt_device`). kt exposes the same `flatten_all`/`to_dtype`/
/// `to_vec1`/`from_vec`/`reshape` surface, so the byte layout is identical;
/// this is a pure type-substrate swap with no behavior change.
#[cfg(feature = "vulkan")]
#[inline]
pub(super) fn vk_tensor_to_f32_bytes_with_shape(tensor: &Tensor) -> Result<(Vec<u8>, Vec<usize>)> {
    let shape: Vec<usize> = tensor.dims().to_vec();
    let flat = tensor.flatten_all().context("failed to flatten tensor")?;
    let f32_data = flat
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .context("failed to extract f32 data")?;
    Ok((bytemuck::cast_slice(&f32_data).to_vec(), shape))
}

#[cfg(feature = "vulkan")]
#[inline]
pub(super) fn vk_tensor_from_f32_bytes(
    data: &[u8],
    shape: &[usize],
    dtype: DType,
) -> Result<Tensor> {
    let f32_data: &[f32] = bytemuck::cast_slice(data);
    // kt `from_vec` is CPU-resident and takes no device arg.
    let tensor = Tensor::from_vec(f32_data.to_vec(), f32_data.len())?.reshape(shape)?;
    if dtype == DType::BF16 {
        Ok(tensor.to_dtype(DType::BF16)?)
    } else {
        Ok(tensor)
    }
}

/// Inference-only Vulkan RMSNorm dispatch. Promotes inputs to F32,
/// dispatches the kernel, casts result back to the input dtype.
/// Returns `Ok(None)` when preconditions don't fit (caller falls back).
#[cfg(feature = "vulkan")]
pub(super) fn try_vulkan_rmsnorm_forward(
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Option<Tensor>> {
    let Some(vk_device) = vulkan_device_handle() else {
        return Ok(None);
    };
    let in_dtype = x.dtype();
    let x_f32 = if in_dtype == DType::F32 {
        x.contiguous()?
    } else {
        x.to_dtype(DType::F32)?.contiguous()?
    };
    let w_f32 = if weight.dtype() == DType::F32 {
        weight.clone()
    } else {
        weight.to_dtype(DType::F32)?
    };
    let x_dims = x_f32.shape().to_vec();
    let hidden = *x_dims
        .last()
        .ok_or_else(|| anyhow::anyhow!("rmsnorm: x has no dims"))?;
    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
    // #1082 vulkan candle removal: the byte helpers are kt-typed now (the
    // vulkan feature no longer links candle). The kt tensors are already
    // CPU-resident at the kt<->vk seam, so extraction runs directly on them
    // with no candle round-trip.
    let x_bytes = vk_tensor_to_f32_bytes_with_shape(&x_f32)?.0;
    let w_bytes = vk_tensor_to_f32_bytes_with_shape(&w_f32)?.0;
    let out_bytes = kiln_vulkan_kernel::kernels::dispatch_qwen_rmsnorm_forward_bytes(
        vk_device.as_ref(),
        &x_bytes,
        &w_bytes,
        rows,
        hidden,
        eps,
    )?;
    let out_f32 = vk_tensor_from_f32_bytes(&out_bytes, &x_dims, DType::F32)?;
    let out = if out_f32.dtype() == in_dtype {
        out_f32
    } else {
        out_f32.to_dtype(in_dtype)?
    };
    Ok(Some(out))
}

/// Process-cached Vulkan device handle. The first call constructs +
/// caches the handle; subsequent calls just clone the Arc.
#[cfg(feature = "vulkan")]
pub(super) fn vulkan_device_handle() -> Option<std::sync::Arc<kiln_vulkan_kernel::VulkanDevice>> {
    static VK_DEVICE: std::sync::OnceLock<
        Option<std::sync::Arc<kiln_vulkan_kernel::VulkanDevice>>,
    > = std::sync::OnceLock::new();
    VK_DEVICE
        .get_or_init(|| {
            kiln_vulkan_kernel::VulkanDevice::new()
                .ok()
                .map(std::sync::Arc::new)
        })
        .clone()
}

// (#1082) Deleted the candle-CustomOp1 `try_vulkan_rmsnorm_autograd` + its
//   inner `VulkanRmsNormOp`: Vulkan is inference-only; training autograd is
//   CUDA-BF16 / kt-tape only now.

/// Candle-op reference RMSNorm. Kept as the CPU path and as the correctness
/// oracle for the fused CUDA kernel. Matches HF semantics exactly:
/// `out = (1 + w) * x * rsqrt(mean(x^2) + eps)` with F32 reduction and epilogue.
pub fn rms_norm_fallback(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    // Phase 7 (#1082): when stable KT routes are enabled and `x` is a contiguous CUDA
    // tensor in the {F32, BF16, F16} triangle, route the BF16→F32
    // promotion at the RMSNorm fallback entry through
    // `kiln_tensor::cuda_cast`. Falls through to the kt
    // `.to_dtype()` op when any precondition fails so behavior is
    // identical with the gate off.
    let x_f32 = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_to_dtype(x, DType::F32)? {
                Some(out) => out,
                None => x.to_dtype(DType::F32)?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            x.to_dtype(DType::F32)?
        }
    };
    // Phase 7 (#1082): when stable KT routes are enabled and the squared F32 input is
    // a contiguous CUDA tensor, route the `mean_keepdim(-1)` step
    // through `kiln_tensor::cuda_mean_last_axis` plus a zero-cost
    // `unsqueeze(-1)` to restore the trailing-dim shape. Falls
    // through to the kt composite when any precondition fails
    // so behavior is identical with the gate off.
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
    // `kiln_tensor::cuda_scalar_op` with kind 0 (AddScalar) — a
    // single-kernel dispatch instead of the kt composite.
    // Mirrors the l2_normalize and softplus add-scalar wirings.
    // Falls through to the kt `+ f64` composite when any
    // precondition fails so behavior is identical with the gate
    // off.
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
    // Phase 7 (#1082): when stable KT routes are enabled and `variance + eps` is a
    // contiguous CUDA F32 tensor, route the
    // `.sqrt()?.recip()?` composite (the RMSNorm tail) through
    // `kiln_tensor::cuda_activation_unary` kind 28 (Rsqrt) — a
    // single fused kernel that replaces the two kt calls + the
    // intermediate sqrt buffer. Falls through to the kt
    // composite when any precondition fails so behavior is
    // identical with the gate off.
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
    let normed = x_f32.broadcast_mul(&rms_inv)?;
    // Qwen3.5 RMSNorm stores weights centered around 0 and applies as (1 + w) * x_normed.
    // Keep everything in F32 for precision (matches HF: `output * (1.0 + self.weight.float())`),
    // then cast back to input dtype at the end.
    // Vulkan's hybrid residency keeps some decode activations on the kt CPU
    // sentinel while small frozen weights may be materialized as Vulkan
    // tensors. The composite fallback must align the weight before its final
    // broadcast multiply, just as the projection and residual fallbacks do.
    #[cfg(feature = "vulkan")]
    let weight_aligned;
    #[cfg(feature = "vulkan")]
    let weight = if x_f32.device() != weight.device()
        && matches!(x_f32.device(), Device::Cpu | Device::Vulkan(_))
        && matches!(weight.device(), Device::Cpu | Device::Vulkan(_))
    {
        weight_aligned = weight
            .to_device(x_f32.device())
            .context("align Vulkan/CPU RMSNorm weight to activation device")?;
        &weight_aligned
    } else {
        weight
    };
    let w_f32 = weight.to_dtype(DType::F32)?;
    let w_plus_one = (w_f32.ones_like()? + w_f32)?;
    let out = normed.broadcast_mul(&w_plus_one)?;
    Ok(out.to_dtype(x.dtype())?)
}

/// Phase 7 (#1082) — kt-API `mean_keepdim(-1)` migration helper.
/// Routes a contiguous CUDA kt tensor through
/// `kiln_tensor::cuda_mean_last_axis` (which reduces the trailing
/// axis) and re-applies `unsqueeze(-1)` so the output shape
/// matches `mean_keepdim(-1)`.
///
/// Returns `Ok(None)` on any incompatibility so the caller falls
/// through to the kt composite. NVTX range
/// `kiln/mean_last_dim_kt` brackets the migrated call so nsys
/// traces separate the path from the baseline composite.
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_mean_last_dim_keepdim(x: &Tensor) -> Result<Option<Tensor>> {
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

    kiln_nvtx::range!(c"kiln/mean_last_dim_kt");

    let out_kt = match kiln_tensor::cuda_mean_last_axis(x) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let reduced = out_kt;
    let out = reduced
        .unsqueeze(reduced.rank())
        .map_err(|e| anyhow::anyhow!("try_kt_mean_last_dim_keepdim: unsqueeze failed: {e}"))?;
    Ok(Some(out))
}

/// Apply Rotary Position Embeddings (RoPE) to query and key tensors.
///
/// `q`: [batch, seq_len, num_heads, head_dim]
/// `k`: [batch, seq_len, num_kv_heads, head_dim]
/// `positions`: position index for each token in the sequence (length = seq_len)
/// `head_dim`: dimension of each attention head
/// `rotary_dim`: number of head dimensions to apply rotation to (the rest pass through unchanged).
///   For Qwen3.5-4B: 64 (partial_rotary_factor=0.25, so 0.25 * 256 = 64).
/// `inv_freq`: cached frequency table of shape `[rotary_dim / 2]` (F32 on same device as `q`/`k`).
///   Build once via [`compute_rotary_inv_freq`] and reuse across calls.
///
/// Returns: (rotated_q, rotated_k) with same shapes.
pub fn rotary_embedding(
    q: &Tensor,
    k: &Tensor,
    positions: &[u32],
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let device = q.device();

    // (#1082) Align `inv_freq` to the ACTIVATION device. On Vulkan the loader
    // keeps `inv_freq` CPU-host (`loader_kt_device`), but the training-time q/k
    // are Vulkan-resident, so `pos.broadcast_mul(&inv_freq)` would mix a
    // Vulkan `pos` with a CPU `inv_freq` and the kt DeviceOp2 "mul" errors with
    // "inputs on different devices". The `[rotary_dim/2]` table is tiny; on
    // CUDA/Metal it is already on `device` so `to_device` is a no-op.
    let inv_freq = inv_freq.to_device(device)?;

    // Position tensor
    // #1082: kt has no `Tensor::new(slice, &Device)`; build a 1-D kt tensor on
    // the source device via `from_vec_on`, then unsqueeze to [seq_len, 1].
    let pos_f32: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
    let pos_len = pos_f32.len();
    let pos = Tensor::from_vec_on(device, pos_f32, vec![pos_len])?.unsqueeze(1)?; // [seq_len, 1]

    // Outer product: [seq_len, half_rotary]
    let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    #[cfg(feature = "cuda")]
    let cos = match try_kt_cos(&freqs)? {
        Some(t) => t,
        None => freqs.cos()?,
    };
    #[cfg(not(feature = "cuda"))]
    let cos = freqs.cos()?; // [seq_len, half_rotary]
    #[cfg(feature = "cuda")]
    let sin = match try_kt_sin(&freqs)? {
        Some(t) => t,
        None => freqs.sin()?,
    };
    #[cfg(not(feature = "cuda"))]
    let sin = freqs.sin()?; // [seq_len, half_rotary]

    let rotated_q = apply_rope(q, &cos, &sin, head_dim, rotary_dim)?;
    let rotated_k = apply_rope(k, &cos, &sin, head_dim, rotary_dim)?;

    Ok((rotated_q, rotated_k))
}

/// Same as [`rotary_embedding`] but accepts positions as a pre-allocated GPU tensor
/// instead of a CPU slice. This is critical for CUDA graph compatibility: the tensor's
/// GPU address stays stable across graph replays, and its contents can be updated via
/// `cudaMemcpyAsync` outside the captured graph.
///
/// `positions_tensor`: f32 tensor on device, shape `[seq_len`]
/// `inv_freq`: cached frequency table, shape `[rotary_dim / 2]`, F32 on device.
pub fn rotary_embedding_from_tensor(
    q: &Tensor,
    k: &Tensor,
    positions_tensor: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // (#1082) Align the rotary tables to the ACTIVATION device. On Vulkan the
    // loader keeps `inv_freq` (and the position ramp) CPU-host
    // (`loader_kt_device`), but the training-time q/k activations are
    // Vulkan-resident — so `pos.broadcast_mul(&inv_freq)` (and the downstream
    // `apply_rope` muls) would otherwise mix CPU + Vulkan operands and the kt
    // DeviceOp2 errors with "inputs on different devices". The `[seq_len]` ramp
    // and `[rotary_dim/2]` table are tiny, so this H2D is negligible. On
    // CUDA/Metal both are already on `q.device()` so `to_device` is a no-op.
    let dev = q.device();
    let positions_tensor = positions_tensor.to_device(dev)?;
    let inv_freq = inv_freq.to_device(dev)?;
    // positions_tensor is [seq_len], unsqueeze to [seq_len, 1]
    let pos = positions_tensor.unsqueeze(1)?;

    let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    #[cfg(feature = "cuda")]
    let cos = match try_kt_cos(&freqs)? {
        Some(t) => t,
        None => freqs.cos()?,
    };
    #[cfg(not(feature = "cuda"))]
    let cos = freqs.cos()?;
    #[cfg(feature = "cuda")]
    let sin = match try_kt_sin(&freqs)? {
        Some(t) => t,
        None => freqs.sin()?,
    };
    #[cfg(not(feature = "cuda"))]
    let sin = freqs.sin()?;

    #[cfg(feature = "cuda")]
    {
        if !crate::tape_forward::tape_scope_active() && !cuda_fused_rotary_qk_disabled() {
            if let (Some(q_kt), Some(k_kt), Some(cos_kt), Some(sin_kt)) = (
                try_borrow_kt_cuda(q),
                try_borrow_kt_cuda(k),
                try_borrow_kt_cuda(&cos),
                try_borrow_kt_cuda(&sin),
            ) {
                if kiln_rmsnorm_kernel::supports_rotary_qk_kt(
                    &q_kt, &k_kt, &cos_kt, &sin_kt, head_dim, rotary_dim,
                ) {
                    // Phase 7 (#1082): kt-only. Bit-exact: bottoms out in the
                    // same `kiln_fused_rotary_qk` FFI symbol.
                    // #1082 forward-flip: kt-native — return kt directly.
                    let (rq_kt, rk_kt) = kiln_rmsnorm_kernel::fused_rotary_qk_kt(
                        &q_kt, &k_kt, &cos_kt, &sin_kt, rotary_dim,
                    )
                    .map_err(|e| anyhow::anyhow!("kt fused_rotary_qk: {e}"))?;
                    return Ok((rq_kt, rk_kt));
                }
            }
        }
    }

    // ROCm inference: route through the native fused rotary-QK kernel
    // (`kiln_rmsnorm_kernel::fused_rotary_qk_kt`, R.7 — rocm-capable), mirroring
    // the CUDA branch above. Without this ROCm ran the ~10-op `apply_rope`
    // composite (cast/narrow/contiguous/broadcast-mul ×2 for q AND k) every
    // attention layer — many small kernel launches on the decode hot path. The
    // fused kernel is one launch and bit-exact (same FFI symbol).
    // supports_rotary_qk_kt already gates device(Rocm) + BF16 q/k + F32 cos/sin
    // + contiguous + rank-4 + shape; if any fails it falls to apply_rope.
    #[cfg(feature = "rocm")]
    if !crate::tape_forward::tape_scope_active()
        && !q.track_op()
        && !k.track_op()
        && kiln_rmsnorm_kernel::supports_rotary_qk_kt(q, k, &cos, &sin, head_dim, rotary_dim)
    {
        return kiln_rmsnorm_kernel::fused_rotary_qk_kt(q, k, &cos, &sin, rotary_dim)
            .map_err(|e| anyhow::anyhow!("rocm kt fused_rotary_qk: {e}"))
            .context("rotary_embedding_from_tensor rocm fused kernel");
    }

    let rotated_q = apply_rope(q, &cos, &sin, head_dim, rotary_dim)?;
    let rotated_k = apply_rope(k, &cos, &sin, head_dim, rotary_dim)?;

    Ok((rotated_q, rotated_k))
}

// #34: pub(crate) so the CUDA-graph runner fills its rotary buffers via this
// exact GPU path (kt_cos/kt_sin), matching eager bit-for-bit (BUG2 fix).
pub(crate) fn rotary_tables_from_tensor(
    positions_tensor: &Tensor,
    inv_freq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let pos = positions_tensor.unsqueeze(1)?;
    let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
    #[cfg(feature = "cuda")]
    let cos = match try_kt_cos(&freqs)? {
        Some(t) => t,
        None => freqs.cos()?,
    };
    #[cfg(not(feature = "cuda"))]
    let cos = freqs.cos()?;
    #[cfg(feature = "cuda")]
    let sin = match try_kt_sin(&freqs)? {
        Some(t) => t,
        None => freqs.sin()?,
    };
    #[cfg(not(feature = "cuda"))]
    let sin = freqs.sin()?;
    Ok((cos, sin))
}

pub(super) fn rotary_embedding_from_tables(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        if !crate::tape_forward::tape_scope_active() && !cuda_fused_rotary_qk_disabled() {
            if let (Some(q_kt), Some(k_kt), Some(cos_kt), Some(sin_kt)) = (
                try_borrow_kt_cuda(q),
                try_borrow_kt_cuda(k),
                try_borrow_kt_cuda(cos),
                try_borrow_kt_cuda(sin),
            ) {
                if kiln_rmsnorm_kernel::supports_rotary_qk_kt(
                    &q_kt, &k_kt, &cos_kt, &sin_kt, head_dim, rotary_dim,
                ) {
                    // Phase 7 (#1082): kt-only. Same FFI symbol as the
                    // kt composite path.
                    // #1082 forward-flip: kt-native — return the kt outputs
                    // directly (no candle round-trip).
                    let (rq_kt, rk_kt) = kiln_rmsnorm_kernel::fused_rotary_qk_kt(
                        &q_kt, &k_kt, &cos_kt, &sin_kt, rotary_dim,
                    )
                    .map_err(|e| anyhow::anyhow!("kt fused_rotary_qk2: {e}"))?;
                    return Ok((rq_kt, rk_kt));
                }
            }
        }
    }

    #[cfg(feature = "rocm")]
    if !crate::tape_forward::tape_scope_active()
        && !q.track_op()
        && !k.track_op()
        && kiln_rmsnorm_kernel::supports_rotary_qk_kt(q, k, cos, sin, head_dim, rotary_dim)
    {
        return kiln_rmsnorm_kernel::fused_rotary_qk_kt(q, k, cos, sin, rotary_dim)
            .map_err(|e| anyhow::anyhow!("rocm kt fused_rotary_qk tables: {e}"))
            .context("rotary_embedding_from_tables rocm fused kernel");
    }

    #[cfg(feature = "metal")]
    {
        if !crate::tape_forward::tape_scope_active()
            && crate::backend::metal::metal_rotary_embedding_supports(
                q, k, cos, sin, head_dim, rotary_dim,
            )
        {
            return crate::backend::metal::metal_rotary_embedding_bf16(
                q, k, cos, sin, head_dim, rotary_dim,
            )
            .context("metal rotary embedding kernel failed");
        }
    }
    let rotated_q = apply_rope(q, cos, sin, head_dim, rotary_dim)?;
    let rotated_k = apply_rope(k, cos, sin, head_dim, rotary_dim)?;
    Ok((rotated_q, rotated_k))
}

/// Residual add `a + b` (same shape) — the #1082 migration target for
/// `kiln/residual`. Under an active tape scope
/// this routes through `kiln_tensor::ops::add` and records an
/// `AddBackward` node (CP-4 shadow-tape adapter #7); otherwise it is the
/// plain kt add, bit-identical to the prior `(a + b)?` expression.
pub(super) fn residual_add(a: Tensor, b: Tensor) -> Result<Tensor> {
    // (#1082) Vulkan added: device-agnostic pure-kt AddBackward recorder. The
    // residual add is on the critical path between every attn/MLP subblock and
    // the residual stream; without recording it on Vulkan the tape severs at the
    // residual and the in-block LoRA grads never reach the loss.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    {
        // #1082 seam flip: kt-native AddBackward recorder — no kt->candle->kt
        // round-trip (decode has no scope and falls to the plain add below).
        if crate::tape_forward::tape_scope_active() {
            return require_active_tape_output(
                crate::tape_forward::try_tape_add_kt(&a, &b)
                    .context("residual_add try_tape_add_kt")?,
                "residual add",
            );
        }
    }
    #[cfg(feature = "vulkan")]
    let b = if !crate::tape_forward::tape_scope_active()
        && a.device() != b.device()
        && matches!(a.device(), Device::Cpu | Device::Vulkan(_))
        && matches!(b.device(), Device::Cpu | Device::Vulkan(_))
    {
        b.to_device(a.device())
            .context("align Vulkan/CPU residual branch to residual-stream device")?
    } else {
        b
    };
    Ok((a + b)?)
}

/// Apply the rotation to a single tensor, supporting partial rotary embeddings.
/// `x`: [batch, seq_len, num_heads, head_dim]
/// `cos`, `sin`: [seq_len, half_rotary]
/// `head_dim`: total dimension per head
/// `rotary_dim`: number of dimensions to rotate (must be even). The first `rotary_dim` dims
///   are rotated; the remaining `head_dim - rotary_dim` dims pass through unchanged.
pub(super) fn apply_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Tensor> {
    // (#1082) Vulkan added: `try_tape_rope_kt` is a device-agnostic pure-kt
    // recorder (`rope_split_half` + `RopeSplitHalfBackward`). Without it on
    // Vulkan the q/k RoPE severs the tape and `q_proj`/`k_proj` LoRA grads
    // never reach the loss.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    {
        // Route split-half RoPE through the kt Tape when a scope is active. No-ops
        // (returns None) otherwise — the production fast-path below is
        // untouched in the default configuration.
        // #1082 seam flip: kt-native split-half RoPE recorder — no kt->candle->kt.
        if crate::tape_forward::tape_scope_active() {
            let out = crate::tape_forward::try_tape_rope_kt(x, cos, sin, head_dim, rotary_dim)
                .context("apply_rope try_tape_rope_kt")?
                .context("active tape scope could not record RoPE")?;
            return Ok(out);
        }
        // #1082 forward-flip: `apply_rope` is kt-native now. The candle
        // `CudaRotaryOneBf16` CustomOp3 autograd fast-path only applies to
        // tracked candle tensors; kt tensors are forward-only, so the
        // autograd routing is handled by the CP-4 tape path above. The
        // fused inference RoPE for kt inputs goes through the kt rotary
        // kernel exercised elsewhere; here we fall through to the
        // elementwise composite below.
    }

    let half_rotary = rotary_dim / 2;
    let x_dtype = x.dtype();

    // Work in f32 for precision.
    // #1082: x may be a transposed GQA-head view (non-contiguous); kt CastOp
    // requires contiguous (candle copied implicitly). No-op when contiguous.
    let x = x.contiguous()?.to_dtype(DType::F32)?;

    // Split into rotary portion and passthrough portion.
    // #1082 forward-flip: kt `narrow` takes a `usize` axis (not `D`), so the
    // last axis is resolved explicitly from the tensor rank.
    let x_last = x.rank() - 1;
    let x_rot = x.narrow(x_last, 0, rotary_dim)?; // [..., :rotary_dim]
    let x_pass = if rotary_dim < head_dim {
        Some(x.narrow(x_last, rotary_dim, head_dim - rotary_dim)?) // [..., rotary_dim:]
    } else {
        None
    };

    // Split rotary portion into two halves
    let x_rot_last = x_rot.rank() - 1;
    // #1082: narrowed halves are non-contiguous views; kt elementwise requires
    // contiguous operands (candle handled strided). `.contiguous()` is an O(1)
    // no-op when already contiguous.
    let x1 = x_rot.narrow(x_rot_last, 0, half_rotary)?.contiguous()?; // [..., :half_rotary]
    let x2 = x_rot
        .narrow(x_rot_last, half_rotary, half_rotary)?
        .contiguous()?; // [..., half_rotary:rotary_dim]

    // cos/sin are [seq_len, half_rotary], need to broadcast to [batch, seq_len, num_heads, half_rotary]
    // Reshape to [1, seq_len, 1, half_rotary]. #1082: unsqueeze yields a view →
    // contiguify before the broadcast-mul (kt elementwise needs contiguous).
    let cos = cos
        .to_dtype(DType::F32)?
        .unsqueeze(0)?
        .unsqueeze(2)?
        .contiguous()?;
    let sin = sin
        .to_dtype(DType::F32)?
        .unsqueeze(0)?
        .unsqueeze(2)?
        .contiguous()?;

    // The cos/sin tables are built from the request's CPU `positions`, so on a
    // GPU backend they arrive on CPU while `x` (q/k) is device-resident. Align
    // them to `x`'s device so the rotation multiplies run entirely on-device
    // (a ~KB-scale constant table, shared across the rotation — not activation
    // paging). No-op when already co-located (CPU inference, or CUDA/Metal
    // which build tables on-device).
    let cos = if cos.device() != x.device() {
        cos.to_device(x.device())?
    } else {
        cos
    };
    let sin = if sin.device() != x.device() {
        sin.to_device(x.device())?
    } else {
        sin
    };

    // Standard RoPE rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

    // Concatenate rotated dims + passthrough dims.
    //
    // Phase 7 (#1082): when stable KT routes are enabled and the inputs satisfy
    // the kt-bridge borrow preconditions, route through
    // `kiln_tensor::cuda_concat`. Falls through to the kt
    // `Tensor::cat` composite when any precondition fails.
    let out = match x_pass {
        Some(pass) => {
            let pieces: [&Tensor; 3] = [&r1, &r2, &pass];
            #[cfg(feature = "cuda")]
            {
                if let Some(out) = try_kt_concat_last_dim(&pieces)? {
                    out
                } else {
                    Tensor::cat(&pieces, LAST_DIM)?
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                Tensor::cat(&pieces, LAST_DIM)?
            }
        }
        None => {
            let pieces: [&Tensor; 2] = [&r1, &r2];
            #[cfg(feature = "cuda")]
            {
                if let Some(out) = try_kt_concat_last_dim(&pieces)? {
                    out
                } else {
                    Tensor::cat(&pieces, LAST_DIM)?
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                Tensor::cat(&pieces, LAST_DIM)?
            }
        }
    };
    Ok(out.to_dtype(x_dtype)?)
}

/// Phase 7 (#1082) — kt-API last-dim concat migration helper.
/// Routes contiguous CUDA kt tensors of a supported dtype
/// through `kiln_tensor::cuda_concat` along the trailing axis.
///
/// Returns `Ok(None)` on any incompatibility (empty input, mixed
/// devices/dtypes, non-CUDA, non-contiguous, unsupported dtype,
/// rank-0) so the caller falls through to the kt composite.
/// NVTX range `kiln/concat_last_dim_kt` brackets the migrated call
/// so nsys traces separate the path from the baseline composite.
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_concat_last_dim(pieces: &[&Tensor]) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    if pieces.is_empty() {
        return Ok(None);
    }
    let first = pieces[0];
    let rank = first.rank();
    if rank == 0 {
        return Ok(None);
    }
    let axis = rank - 1;
    let dtype = first.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Ok(None);
    }
    if !matches!(first.device(), Device::Cuda(_)) {
        return Ok(None);
    }
    for t in pieces.iter() {
        if !matches!(t.device(), Device::Cuda(_))
            || t.dtype() != dtype
            || !t.is_contiguous()
            || t.rank() != rank
        {
            return Ok(None);
        }
    }

    kiln_nvtx::range!(c"kiln/concat_last_dim_kt");

    let out_kt = match kiln_tensor::cuda_concat(pieces, axis) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API axis-0 concat migration helper. Routes
/// contiguous CUDA kt tensors of a supported dtype through
/// `kiln_tensor::cuda_concat(_, 0)`.
///
/// Returns `Ok(None)` on any incompatibility (empty input, mixed
/// devices/dtypes, non-CUDA, non-contiguous, unsupported dtype,
/// rank-0, or mismatched non-axis-0 dim) so the caller falls
/// through to the kt composite. NVTX range
/// `kiln/cat_dim0_kt` brackets the migrated call so nsys traces
/// separate the path from the baseline composite.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_cat_dim0(pieces: &[&Tensor]) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    if pieces.is_empty() {
        return Ok(None);
    }
    let first = pieces[0];
    let rank = first.rank();
    if rank == 0 {
        return Ok(None);
    }
    let dtype = first.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Ok(None);
    }
    if !matches!(first.device(), Device::Cuda(_)) {
        return Ok(None);
    }
    for t in pieces.iter() {
        if !matches!(t.device(), Device::Cuda(_))
            || t.dtype() != dtype
            || !t.is_contiguous()
            || t.rank() != rank
        {
            return Ok(None);
        }
    }

    kiln_nvtx::range!(c"kiln/cat_dim0_kt");

    let out_kt = match kiln_tensor::cuda_concat(pieces, 0) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API axis-1 concat migration helper. Routes
/// contiguous CUDA kt tensors of a supported dtype through
/// `kiln_tensor::cuda_concat(_, 1)`.
///
/// Returns `Ok(None)` on any incompatibility (empty input, mixed
/// devices/dtypes, non-CUDA, non-contiguous, unsupported dtype,
/// rank < 2) so the caller falls through to the kt composite.
/// NVTX range `kiln/cat_dim1_kt` brackets the migrated call so
/// nsys traces separate the path from the baseline composite.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_cat_dim1(pieces: &[&Tensor]) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    if pieces.is_empty() {
        return Ok(None);
    }
    let first = pieces[0];
    let rank = first.rank();
    if rank < 2 {
        return Ok(None);
    }
    let dtype = first.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Ok(None);
    }
    if !matches!(first.device(), Device::Cuda(_)) {
        return Ok(None);
    }
    for t in pieces.iter() {
        if !matches!(t.device(), Device::Cuda(_))
            || t.dtype() != dtype
            || !t.is_contiguous()
            || t.rank() != rank
        {
            return Ok(None);
        }
    }

    kiln_nvtx::range!(c"kiln/cat_dim1_kt");

    let out_kt = match kiln_tensor::cuda_concat(pieces, 1) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API axis-2 concat migration helper. Routes
/// contiguous CUDA kt tensors of a supported dtype through
/// `kiln_tensor::cuda_concat(_, 2)`.
///
/// Returns `Ok(None)` on any incompatibility (empty input, mixed
/// devices/dtypes, non-CUDA, non-contiguous, unsupported dtype,
/// rank < 3) so the caller falls through to the kt composite.
/// NVTX range `kiln/cat_dim2_kt` brackets the migrated call so
/// nsys traces separate the path from the baseline composite.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_cat_dim2(pieces: &[&Tensor]) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    if pieces.is_empty() {
        return Ok(None);
    }
    let first = pieces[0];
    let rank = first.rank();
    if rank < 3 {
        return Ok(None);
    }
    let dtype = first.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Ok(None);
    }
    if !matches!(first.device(), Device::Cuda(_)) {
        return Ok(None);
    }
    for t in pieces.iter() {
        if !matches!(t.device(), Device::Cuda(_))
            || t.dtype() != dtype
            || !t.is_contiguous()
            || t.rank() != rank
        {
            return Ok(None);
        }
    }

    kiln_nvtx::range!(c"kiln/cat_dim2_kt");

    let out_kt = match kiln_tensor::cuda_concat(pieces, 2) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API add-scalar migration helper. Routes a
/// contiguous CUDA kt tensor through
/// `kiln_tensor::cuda_scalar_op` with kind tag 0 (AddScalar) for
/// the kt `Tensor + scalar` (`Add<f64>`) shape.
///
/// Returns `Ok(None)` on any incompatibility (non-CUDA, unsupported
/// dtype, non-contiguous, rank-0, or non-finite scalar) so the
/// caller falls through to the kt composite. NVTX range
/// `kiln/add_scalar_kt` brackets the migrated call so nsys traces
/// separate the path from the baseline composite.
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_add_scalar(x: &Tensor, c: f64) -> Result<Option<Tensor>> {
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
    if !c.is_finite() {
        return Ok(None);
    }
    let c_f32 = c as f32;

    kiln_nvtx::range!(c"kiln/add_scalar_kt");

    // kind tag 0 = ScalarKind::AddScalar (matches
    // crates/kiln-tensor/src/ops/scalar.rs).
    let out_kt = match kiln_tensor::cuda_scalar_op(x, 0, c_f32) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API neg migration helper. Routes a
/// contiguous CUDA kt tensor through
/// `kiln_tensor::cuda_activation_unary` with kind tag 12 (Neg).
///
/// Returns `Ok(None)` on any incompatibility (non-CUDA, unsupported
/// dtype, non-contiguous, rank-0) so the caller falls through to
/// the kt `.neg()` op. NVTX range `kiln/neg_kt` brackets the
/// migrated call so nsys traces separate the path from the
/// baseline composite.
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_neg(x: &Tensor) -> Result<Option<Tensor>> {
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

    kiln_nvtx::range!(c"kiln/neg_kt");

    // kind tag 12 = Neg (matches csrc/activation.cu constants and
    // crates/kiln-tensor/src/ops/unary_arith.rs::cuda_kind_tag).
    let out_kt = match kiln_tensor::cuda_activation_unary(x, 12) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API sqrt migration helper. Routes a
/// contiguous CUDA kt tensor through
/// `kiln_tensor::cuda_activation_unary` with kind tag 14 (Sqrt).
///
/// Returns `Ok(None)` on any incompatibility (non-CUDA, unsupported
/// dtype, non-contiguous, rank-0) so the caller falls through to
/// the kt `.sqrt()` op. NVTX range `kiln/sqrt_kt` brackets the
/// migrated call so nsys traces separate the path from the
/// baseline composite.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_sqrt(x: &Tensor) -> Result<Option<Tensor>> {
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

    kiln_nvtx::range!(c"kiln/sqrt_kt");

    // kind tag 14 = Sqrt (matches csrc/activation.cu constants and
    // crates/kiln-tensor/src/ops/unary_arith.rs::cuda_kind_tag).
    let out_kt = match kiln_tensor::cuda_activation_unary(x, 14) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API rsqrt migration helper. Routes a
/// contiguous CUDA kt tensor through
/// `kiln_tensor::cuda_activation_unary` with kind tag 28 (Rsqrt).
/// Computes `1 / sqrt(x)` in a single kernel pass, replacing the
/// `sqrt().recip()` composite which makes two passes through
/// device memory and allocates a transient sqrt-output buffer.
///
/// Returns `Ok(None)` on any incompatibility (non-CUDA, unsupported
/// dtype, non-contiguous, rank-0) so the caller falls through to
/// the kt composite. NVTX range `kiln/rsqrt_kt` brackets the
/// migrated call so nsys traces separate the path from the
/// baseline composite.
///
/// Wired up for completeness; today there are no `Tensor::rsqrt()`
/// API calls in `forward.rs` (candle 0.9 has no rsqrt method). The
/// production RMSNorm-tail `(variance + eps).sqrt().recip()` sites
/// (5+ of them) can be ported to call this helper directly in
/// follow-up commits — each port replaces two kt calls + one
/// allocation with a single fused kernel.
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_rsqrt(x: &Tensor) -> Result<Option<Tensor>> {
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

    kiln_nvtx::range!(c"kiln/rsqrt_kt");

    // kind tag 28 = Rsqrt (matches KIND_RSQRT in
    // csrc/activation.cu — added alongside this helper).
    let out_kt = match kiln_tensor::cuda_activation_unary(x, 28) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API binary `maximum(a, b)` migration helper.
/// Routes a pair of same-shape, same-dtype contiguous CUDA kt
/// tensors through `kiln_tensor::cuda_binary_minmax` with kind
/// tag 1 (Max). NaN propagation matches `f32::max` semantics —
/// the non-NaN operand wins when one side is NaN, which matches
/// the kt `Tensor::maximum` contract.
///
/// Returns `Ok(None)` on any incompatibility (gate off, non-CUDA,
/// unsupported dtype, non-contiguous, dtype mismatch, shape
/// mismatch, rank-0) so the caller falls through to the kt
/// `.maximum(other)` op. NVTX range `kiln/max_binary_kt` brackets the
/// migrated call so nsys traces separate the path from the
/// baseline composite.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_max_binary(a: &Tensor, b: &Tensor) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    if !matches!(a.device(), Device::Cuda(_))
        || !matches!(b.device(), Device::Cuda(_))
        || !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16)
        || a.dtype() != b.dtype()
        || a.shape() != b.shape()
        || !a.is_contiguous()
        || !b.is_contiguous()
        || a.rank() == 0
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/max_binary_kt");

    // kind tag 1 = Max (matches KIND_MAXIMUM in
    // csrc/binary_minmax.cu).
    let out_kt = match kiln_tensor::cuda_binary_minmax(a, b, 1) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API abs migration helper. Routes a
/// contiguous CUDA kt tensor through
/// `kiln_tensor::cuda_activation_unary` with kind tag 13 (Abs).
///
/// Returns `Ok(None)` on any incompatibility (non-CUDA, unsupported
/// dtype, non-contiguous, rank-0) so the caller falls through to
/// the kt `.abs()` op. NVTX range `kiln/abs_kt` brackets the
/// migrated call so nsys traces separate the path from the
/// baseline composite.
///
/// Today this helper has no production call site — the only
/// `.abs()` sites in `forward.rs` are in `#[cfg(test)]` parity
/// helpers. The helper is wired up to match the established
/// Phase 7 scaffold so future kernel refactors that compute `|x|`
/// directly (instead of `relu(x) + relu(-x)`) can plug in
/// trivially.
///
/// First production call site (wired in the same #1082 series):
/// the `softplus` helper's `abs_x = relu(x) + relu(-x)`
/// computation now takes a fused single-kernel `try_kt_abs(x)`
/// fast path when stable KT routes are enabled. softplus runs as the last
/// step of the GDN `b` (forget gate) computation, once per
/// decode step.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_abs(x: &Tensor) -> Result<Option<Tensor>> {
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

    kiln_nvtx::range!(c"kiln/abs_kt");

    // kind tag 13 = Abs (matches csrc/activation.cu constants and
    // crates/kiln-tensor/src/ops/unary_arith.rs::cuda_kind_tag).
    let out_kt = match kiln_tensor::cuda_activation_unary(x, 13) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API sin migration helper. Routes a
/// contiguous CUDA kt tensor through
/// `kiln_tensor::cuda_activation_unary` with kind tag 7 (Sin).
///
/// Returns `Ok(None)` on any incompatibility (non-CUDA, unsupported
/// dtype, non-contiguous, rank-0) so the caller falls through to
/// the kt `.sin()` op. NVTX range `kiln/sin_kt` brackets the
/// migrated call so nsys traces separate the path from the
/// baseline composite.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_sin(x: &Tensor) -> Result<Option<Tensor>> {
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

    kiln_nvtx::range!(c"kiln/sin_kt");

    // kind tag 7 = Sin (matches csrc/activation.cu KIND_SIN).
    let out_kt = match kiln_tensor::cuda_activation_unary(x, 7) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API cos migration helper. Routes a
/// contiguous CUDA kt tensor through
/// `kiln_tensor::cuda_activation_unary` with kind tag 8 (Cos).
///
/// Returns `Ok(None)` on any incompatibility (non-CUDA, unsupported
/// dtype, non-contiguous, rank-0) so the caller falls through to
/// the kt `.cos()` op. NVTX range `kiln/cos_kt` brackets the
/// migrated call so nsys traces separate the path from the
/// baseline composite. Mirrors `try_kt_sin` (commit 728b3917).
#[cfg(feature = "cuda")]
pub(super) fn try_kt_cos(x: &Tensor) -> Result<Option<Tensor>> {
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

    kiln_nvtx::range!(c"kiln/cos_kt");

    // kind tag 8 = Cos (matches csrc/activation.cu KIND_COS).
    let out_kt = match kiln_tensor::cuda_activation_unary(x, 8) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API exp migration helper. Routes a
/// contiguous CUDA kt tensor through
/// `kiln_tensor::cuda_activation_unary` with kind tag 6 (Exp).
///
/// Returns `Ok(None)` on any incompatibility (non-CUDA, unsupported
/// dtype, non-contiguous, rank-0) so the caller falls through to
/// the kt `.exp()` op. NVTX range `kiln/exp_kt` brackets the
/// migrated call so nsys traces separate the path from the
/// baseline composite. Mirrors `try_kt_sin` / `try_kt_cos`
/// (commits 728b3917 / 6c22330f).
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_exp(x: &Tensor) -> Result<Option<Tensor>> {
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

    kiln_nvtx::range!(c"kiln/exp_kt");

    // kind tag 6 = Exp (matches csrc/activation.cu KIND_EXP and
    // crates/kiln-tensor/src/ops/unary_arith.rs::cuda_kind_tag).
    let out_kt = match kiln_tensor::cuda_activation_unary(x, 6) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API recip migration helper. Routes a
/// contiguous CUDA kt tensor through
/// `kiln_tensor::cuda_activation_unary` with kind tag 22 (Recip).
///
/// Returns `Ok(None)` on any incompatibility (non-CUDA, unsupported
/// dtype, non-contiguous, rank-0) so the caller falls through to
/// the kt `.recip()` op. NVTX range `kiln/recip_kt` brackets the
/// migrated call so nsys traces separate the path from the
/// baseline composite. Mirrors `try_kt_sqrt` / `try_kt_neg` /
/// `try_kt_abs`.
///
/// IEEE semantics match the kt CPU reference: `1.0 / 0 = ±inf`
/// and `1.0 / NaN = NaN`. See `csrc/activation.cu` `KIND_RECIP`
/// case (added in commit 7a3e1e77 for the same #1082 series).
///
/// First call-site migration: the `cuda_sigmoid` composite's
/// `(1 + e^-x).recip()` final step. Additional RMSNorm
/// `(variance + eps).sqrt().recip()` sites land in follow-up
/// commits of the same #1082 series.
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_recip(x: &Tensor) -> Result<Option<Tensor>> {
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

    kiln_nvtx::range!(c"kiln/recip_kt");

    // kind tag 22 = Recip (matches csrc/activation.cu KIND_RECIP).
    let out_kt = match kiln_tensor::cuda_activation_unary(x, 22) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API log migration helper. Routes a
/// contiguous CUDA kt tensor through
/// `kiln_tensor::cuda_activation_unary` with kind tag 5 (Log =
/// natural log, `ln(x)`).
///
/// Returns `Ok(None)` on any incompatibility (non-CUDA, unsupported
/// dtype, non-contiguous, rank-0) so the caller falls through to
/// the kt `.log()` op. NVTX range `kiln/log_kt` brackets the
/// migrated call so nsys traces separate the path from the
/// baseline composite. Mirrors `try_kt_exp` (212d5b83) and the
/// other Phase 7 elementwise helpers.
///
/// IEEE semantics match the kt CPU reference: `ln(0) = -inf`,
/// `ln(<0) = NaN`. See `csrc/activation.cu` `KIND_LOG` case.
///
/// First call-site migration: the `softplus` helper's
/// `(1 + exp(-|x|)).log()` step. softplus runs as the last step
/// of the GDN `b` (forget gate) computation, once per decode step.
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_log(x: &Tensor) -> Result<Option<Tensor>> {
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

    kiln_nvtx::range!(c"kiln/log_kt");

    // kind tag 5 = Log (matches csrc/activation.cu KIND_LOG = ln(x)).
    let out_kt = match kiln_tensor::cuda_activation_unary(x, 5) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API dtype-cast migration helper. Routes a
/// contiguous CUDA kt tensor through `kiln_tensor::cuda_cast`
/// for the {F32 ↔ BF16 ↔ F16} triangle.
///
/// Returns `Ok(None)` on any incompatibility (non-CUDA, unsupported
/// dtype, non-contiguous, rank-0, target == source) so the caller
/// falls through to the kt `.to_dtype(target)` op. NVTX range
/// `kiln/to_dtype_kt` brackets the migrated call so nsys traces
/// separate the path from the baseline composite. Mirrors the
/// other Phase 7 elementwise helpers.
///
/// `to_dtype(same)` is a no-op in candle but `cuda_cast` short-
/// circuits to a `.contiguous()`, so we treat `target == src.dtype()`
/// as an early `Ok(None)` and let the caller decide.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_to_dtype(x: &Tensor, target: DType) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    if !matches!(x.device(), Device::Cuda(_)) || !x.is_contiguous() || x.rank() == 0 {
        return Ok(None);
    }
    // Restrict to the {F32, BF16, F16} cast triangle that `cuda_cast`
    // supports; everything else falls through to the kt composite.
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
        || !matches!(target, DType::F32 | DType::BF16 | DType::F16)
    {
        return Ok(None);
    }
    if x.dtype() == target {
        // Same-dtype `.to_dtype` is a no-op in candle (zero-copy view);
        // skip the kt-API path so we don't pay an unnecessary dtod copy.
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/to_dtype_kt");

    let kt_target = match target {
        DType::F32 => kiln_tensor::DType::F32,
        DType::BF16 => kiln_tensor::DType::BF16,
        DType::F16 => kiln_tensor::DType::F16,
        _ => return Ok(None),
    };
    let out_kt = match kiln_tensor::cuda_cast(x, kt_target) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

// (#1082) Deleted the dead candle-CustomOp3 `CudaRotaryOneBf16` rotary island:
//   `cuda_rotary_one_training_bf16_supported`, `rotary_one_bwd_kt_bridge_disabled`,
//   `fused_rotary_one_backward_via_kt_bridge`, `CudaRotaryOneBf16` (+impl) and the
//   `rotary_one_backward` candle fallback. The op was never applied (no apply_op3
//   site); rotary autograd flows through `crate::tape_forward::try_tape_rope_cuda`.
