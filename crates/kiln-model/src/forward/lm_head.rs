use super::*;

// Pre-allocated lm-head output buffer installed by the captured-graph
// runner. When present, [`try_kt_lm_head`] writes the matmul result
// directly into this kt Tensor via [`kiln_tensor::cuda_matmul_into`]
// instead of allocating a fresh per-call output. The buffer's device
// pointer is then stable across captured-graph replays, so the
// downstream `slice_set(&logits, …)` records a memcpy whose source
// address remains valid on every replay.
//
// This is the lm-head twin of `kiln_gdn_kernel::with_decode_gates_recurrent_outputs`
// — the same pre-allocate-outside-capture / install-via-thread-local
// trick that fixed the GDN decode kernel's intra-capture allocation
// hazard (`cuda_graph.rs:1592-1601` documentation). Phase 5 #1082.
#[cfg(feature = "cuda")]
thread_local! {
    static LM_HEAD_OUTPUT_BUFFER: std::cell::RefCell<Option<Tensor>> =
        const { std::cell::RefCell::new(None) };
}

/// Install a pre-allocated lm-head output buffer for the duration of
/// `f`. The buffer must be a CUDA kt Tensor of shape
/// `[batch, 1, vocab]` (or any shape that matches the expected
/// lm-head output shape for the captured forward) and dtype matching
/// `weights.embed_tokens.dtype()`. On exit, the previous slot value
/// is restored.
///
/// See [`LM_HEAD_OUTPUT_BUFFER`] for why this exists.
#[cfg(feature = "cuda")]
pub fn with_lm_head_output_buffer<R>(buf: Tensor, f: impl FnOnce() -> R) -> R {
    let previous = LM_HEAD_OUTPUT_BUFFER.with(|cell| cell.replace(Some(buf)));
    let result = f();
    LM_HEAD_OUTPUT_BUFFER.with(|cell| {
        let _ = cell.replace(previous);
    });
    result
}

/// Attempt to consume the thread-local lm-head output buffer if its
/// shape and dtype match the caller-provided expectations. Returns
/// `Ok(None)` if no buffer is installed, the shape doesn't match, the
/// dtype doesn't match, or the buffer isn't on a CUDA device.
///
/// On success the returned `Tensor` is a kt handle over the
/// pre-allocated buffer (storage Arc is shared; a new
/// `Tensor` handle with the same dims). The caller is responsible
/// for writing the matmul result into that storage via
/// `kiln_tensor::cuda_matmul_into`.
#[cfg(feature = "cuda")]
pub(super) fn try_take_lm_head_output_buffer(
    expected_shape: &[usize],
    expected_dtype: DType,
) -> Option<Tensor> {
    LM_HEAD_OUTPUT_BUFFER.with(|cell| {
        let borrowed = cell.borrow();
        let buf = borrowed.as_ref()?;
        if !matches!(buf.device(), Device::Cuda(_)) {
            return None;
        }
        if buf.dtype() != expected_dtype {
            return None;
        }
        if buf.dims() != expected_shape {
            return None;
        }
        if !buf.is_contiguous() {
            return None;
        }
        Some(buf.clone())
    })
}

pub(super) fn lm_head_forward(x: &Tensor, embed_tokens_t: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    {
        if crate::backend::metal::metal_lm_head_supports(x, embed_tokens_t) {
            return crate::backend::metal::metal_lm_head_bf16(x, embed_tokens_t)
                .context("metal lm_head kernel failed");
        }
        if crate::backend::metal::metal_transposed_coop_gemv_decode_batch_supports(
            x,
            embed_tokens_t,
        ) {
            return crate::backend::metal::metal_transposed_coop_gemv_bf16(x, embed_tokens_t)
                .context("metal batch lm_head GEMV failed");
        }
    }
    #[cfg(feature = "cuda")]
    if let Some(out) = try_kt_lm_head(x, embed_tokens_t)? {
        return Ok(out);
    }
    broadcast_matmul_cpu_compatible(x, embed_tokens_t)
}

/// Phase 7 (#1082) — kt-API LoRA add migration helper. Routes
/// the `base + delta` final accumulator from
/// [`add_lora_delta_to_base`] (and analogous Marlin LoRA call
/// sites) through `kiln_tensor::cuda_elementwise_binary` with
/// kind tag 0 (Add) directly, ahead of the kt composite.
///
/// Returns `Ok(None)` on any incompatibility (gate off, non-CUDA,
/// unsupported dtype, dtype/shape mismatch, non-contiguous,
/// rank-0) so the caller falls through to the kt
/// `Add<Tensor>` composite. NVTX range `kiln/lora_add_kt`
/// brackets the migrated call so nsys traces separate the path
/// from the kt composite baseline.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_lora_add(base: &Tensor, delta: &Tensor) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    if !matches!(base.device(), Device::Cuda(_))
        || !matches!(delta.device(), Device::Cuda(_))
        || !matches!(base.dtype(), DType::F32 | DType::BF16 | DType::F16)
        || base.dtype() != delta.dtype()
        || base.shape() != delta.shape()
        || !base.is_contiguous()
        || !delta.is_contiguous()
        || base.rank() == 0
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/lora_add_kt");

    // kind tag 0 = Add (matches BinaryKind::Add in
    // crates/kiln-tensor/src/ops/elementwise.rs).
    let out_kt = match kiln_tensor::cuda_elementwise_binary(base, delta, 0) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API LoRA delta migration helper. Routes
/// the `(x @ A^T) @ B^T * scale` three-step composite from
/// [`crate::lora_loader::compute_lora_delta`] through
/// `kiln_tensor::ops::matmul_rhs_transposed` +
/// `kiln_tensor::ops::matmul_rhs_transposed` +
/// `kiln_tensor::ops::mul_scalar` directly, ahead of the kt composite.
///
/// Flattens any leading dims to a 2D `[lead, in_features]` view
/// before dispatching to keep the cublasLt entry shape canonical;
/// reshapes the result back to match `x`'s rank. Casts A/B to
/// `x.dtype()` first (matching the [`compute_lora_delta`] policy:
/// cuBLAS BF16-input + FP32-accumulate on tensor cores).
///
/// Returns `Ok(None)` on any incompatibility (gate off, non-CUDA,
/// unsupported dtype, dtype/rank mismatch, non-contiguous,
/// non-finite scale, K-dim mismatch) so the caller falls through
/// to the kt composite. NVTX range `kiln/lora_delta_kt`
/// brackets the migrated call so nsys traces separate the path
/// from the kt composite baseline.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_lora_delta(
    x: &Tensor,
    proj: &LoraProjectionWeights,
    scale: f32,
) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    let x_dims = x.dims();
    if x_dims.len() < 2 {
        return Ok(None);
    }
    // #1082: `x`/`proj.a`/`proj.b` are all kt (LoraProjectionWeights a/b flipped
    // to KtTensor) — check each against the kt `Device`.
    if !matches!(x.device(), Device::Cuda(_))
        || !matches!(proj.a.device(), Device::Cuda(_))
        || !matches!(proj.b.device(), Device::Cuda(_))
    {
        return Ok(None);
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        return Ok(None);
    }
    if !(scale as f64).is_finite() {
        return Ok(None);
    }
    let Ok((rank, in_features)) = proj.a.dims2() else {
        return Ok(None);
    };
    let Ok((out_features, b_rank)) = proj.b.dims2() else {
        return Ok(None);
    };
    if b_rank != rank || x_dims[x_dims.len() - 1] != in_features {
        return Ok(None);
    }
    // #1082: `proj.a`/`proj.b` are kt — cast in kt directly and keep the
    // original layouts for transposed-GEMM dispatch.
    let a = match proj.a.to_dtype(x.dtype()).and_then(|t| t.contiguous()) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let b = match proj.b.to_dtype(x.dtype()).and_then(|t| t.contiguous()) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    if !a.is_contiguous() || !b.is_contiguous() {
        return Ok(None);
    }

    let lead: usize = x_dims[..x_dims.len() - 1].iter().product();
    let x2d = match x.reshape((lead, in_features)).and_then(|t| t.contiguous()) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    if !x2d.is_contiguous() {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/lora_delta_kt");

    // #1082: everything is kt now (x2d, a, b) — keep the whole
    // matmul→matmul→scale chain in kt, no candle round-trips (perf mandate).
    // Step 1: hidden = x @ A^T -> shape [lead, rank]
    let hidden_kt = match kiln_tensor::ops::matmul_rhs_transposed(&x2d, &a) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    if !hidden_kt.is_contiguous() {
        return Ok(None);
    }

    // Step 2: delta_pre = hidden @ B^T -> shape [lead, out_features]
    let delta_pre_kt = match kiln_tensor::ops::matmul_rhs_transposed(&hidden_kt, &b) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    if !delta_pre_kt.is_contiguous() {
        return Ok(None);
    }

    // Step 3: delta = delta_pre * scale.
    let delta_kt = match kiln_tensor::ops::mul_scalar(&delta_pre_kt, scale) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    let mut out_shape: Vec<usize> = x_dims[..x_dims.len() - 1].to_vec();
    out_shape.push(out_features);
    Ok(Some(delta_kt.reshape(out_shape)?))
}

/// Phase 7 (#1082) — **kt-native** LM head matmul core.
///
/// Operates entirely on `KtTensor` storage: given the flattened
/// `[lead, K]` hidden state and the `[K, vocab]` transposed embedding,
/// both already borrowed as kt tensors, performs the final projection
/// via `kiln_tensor::ops::matmul` and returns the `[lead, vocab]`
/// logits as a `KtTensor`. This is the consolidated kt-internal
/// computation for the lm_head region; the seam (and the captured-graph
/// output-buffer fast path) lives in the [`try_kt_lm_head`] wrapper.
///
/// The matmul is bit-exact to the kt composite baseline for the LM head
/// while letting the kt `MatmulOp` contract choose the active backend's
/// native implementation. The `kiln/lm_head_kt` NVTX range is opened by
/// the [`try_kt_lm_head`] wrapper (covering both this core and the
/// captured-graph `cuda_matmul_into` branch), so this core does not open
/// its own range to avoid a nested duplicate.
#[cfg(feature = "cuda")]
pub(super) fn kt_lm_head_native(lhs_kt: &KtTensor, rhs_kt: &KtTensor) -> Result<KtTensor> {
    kiln_tensor::ops::matmul(lhs_kt, rhs_kt)
        .map_err(|e| anyhow::anyhow!("kt_lm_head_native: matmul failed: {e}"))
}

/// Phase 7 (#1082) — kt-API LM head migration helper. Routes the
/// `[B*T, hidden] @ [hidden, vocab] -> [B*T, vocab]` final projection
/// through `kiln_tensor::ops::matmul` directly. The LM head is the
/// single highest-impact production matmul site: the
/// `embed_tokens_t` weight has shape `[hidden=2560, vocab=151_936]`,
/// and at prefill `x` has `[B*T, hidden]` with sequence lengths in
/// the hundreds or thousands.
///
/// Flattens any leading dims to a 2D `[lead, K]` view before
/// dispatching to keep the kt path on the `M-N-K` cublasLt entry
/// shape; reshapes the result back to match the input rank. The
/// the kt seam is the identity borrow-in (hidden + weight);
/// except on the captured-graph fast path, which writes the matmul
/// result directly into a pre-allocated,
/// graph-stable kt output buffer via `cuda_matmul_into`.
///
/// Returns `Ok(None)` on any incompatibility (gate off,
/// non-{BF16,F16,F32}, dtype mismatch, non-contiguous, non-rank-2
/// weight, K-dim mismatch) so the caller falls through to
/// [`broadcast_matmul_cpu_compatible`]. NVTX range `kiln/lm_head_kt`
/// brackets the migrated call so nsys traces separate the path from
/// the kt composite baseline.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_lm_head(x: &Tensor, embed_tokens_t: &Tensor) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    let l_dims = x.dims();
    let r_dims = embed_tokens_t.dims();
    if r_dims.len() != 2 || l_dims.len() < 2 {
        return Ok(None);
    }
    let k = l_dims[l_dims.len() - 1];
    if r_dims[0] != k {
        return Ok(None);
    }
    if !matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32)
        || x.dtype() != embed_tokens_t.dtype()
        || !x.is_contiguous()
        || !embed_tokens_t.is_contiguous()
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/lm_head_kt");

    let out_n = r_dims[1];
    let lead: usize = l_dims[..l_dims.len() - 1].iter().product();
    // #1082: `x` and `embed_tokens_t` are already kt — reshape/borrow directly,
    // no candle boundary on the inputs.
    let x2d = match x.reshape((lead, k)) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let lhs_kt = &x2d;
    let rhs_kt = embed_tokens_t;

    let mut out_shape: Vec<usize> = l_dims[..l_dims.len() - 1].to_vec();
    out_shape.push(out_n);

    // Phase 5 #1082 — if a captured-graph runner installed a
    // pre-allocated lm-head output buffer matching our expected shape
    // and dtype, write the matmul result directly into it instead of
    // allocating a transient kt Tensor. The pre-allocated buffer's
    // device pointer is graph-stable across replays, so the downstream
    // `slice_set(&logits, …)` records a memcpy whose source address
    // remains valid on every replay — fixing the
    // `CUDA_ERROR_ILLEGAL_ADDRESS` fault at
    // `greedy_sample_rows(captured.output_logits)` documented in
    // `bench-results/cuda-graph-status.md` (2026-05-26 entries).
    let installed_output_buffer = try_take_lm_head_output_buffer(&out_shape, x.dtype());
    if let Some(dst) = installed_output_buffer {
        // The thread-local hands us a kt Tensor shaped like `[batch, 1, vocab]`.
        // Reshape it to the 2-D `[lead, out_n]` matmul output shape and write
        // the matmul result directly into its graph-stable storage.
        let dst2d = match dst.reshape((lead, out_n)) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        if kiln_tensor::cuda_matmul_into(lhs_kt, rhs_kt, &dst2d).is_err() {
            return Ok(None);
        }
        // Return the original `[batch, 1, vocab]`-shaped kt wrapper so the
        // caller's slice_set reads from the graph-stable buffer.
        return Ok(Some(dst));
    }

    // kt-internal computation (no captured-graph buffer installed).
    let out_kt = match kt_lm_head_native(lhs_kt, rhs_kt) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    Ok(Some(out_kt.reshape(out_shape)?))
}

pub(super) fn lm_head_forward_backend_decode_if(
    // Consumed only by the cuda/metal/vulkan/rocm tape + backend-decode block;
    // on feature-less builds the allow silences the unused parameter (verified by
    // default-lane probe).
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        )),
        allow(unused_variables)
    )]
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    embed_tokens_t: &Tensor,
) -> Result<Tensor> {
    // CP-4 (#1082) Increment 2: route the lm_head matmul (no LoRA) through the
    // kt `Tape` so `dL/d(logits)` reaches the lm_head input — the tape root
    // (cross_entropy-from-logits) connects to this output. Wired at the TOP
    // because the authoritative path's input is a DETACHED kt-copy
    // (`track_op()==false`), so the autograd-safe `linear_prefill_apply` branch
    // below (gated on `x.track_op()`) is not reliably hit. No-ops otherwise.
    // #1082: the kt-tape adapter (`try_tape_lora_linear_kt`, formerly the
    // candle-typed cross-file seam) is kt-native now; the BackendRuntime trait is
    // kt-typed (item 4) so `linear_prefill_apply`/`linear_decode` take/return kt.
    // (#1082) Vulkan added: `try_tape_lora_linear_kt` is device-agnostic and is
    // the producer that connects the CE loss back through the lm_head into the
    // model — without it on Vulkan the tape root dead-ends at the logits.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    {
        // Only training opens a tape scope. Decode falls straight through to
        // the kt-native `LinearBackend::runtime_linear_decode` without touching
        // the full frozen lm_head weight through a recording adapter.
        // #1082 seam flip: kt-native lm_head linear recorder — no kt->candle->kt.
        // The twin returns the RECORDED kt tape node directly (it IS the lm_head
        // output), so cross_entropy chains to it with no id-remapping.
        // NB: the enclosing cfg below is widened to include vulkan.
        if crate::tape_forward::tape_scope_active() {
            return require_active_tape_output(
                crate::tape_forward::try_tape_lora_linear_kt(x, embed_tokens_t, None, 0.0)
                    .context("lm_head try_tape_lora_linear_kt")?,
                "LM head projection",
            );
        }
        if let Some(backend) = backend {
            // For autograd-tracked input, prefer the autograd-safe Vulkan
            // CustomOp; otherwise the leaf from linear_decode drops the grad.
            if x.track_op()
                && let Some(logits) =
                    LinearBackend::runtime_linear_prefill_apply(backend, x, embed_tokens_t)?
            {
                return Ok(logits);
            }
            if let Some(logits) = LinearBackend::runtime_linear_decode(backend, x, embed_tokens_t)?
            {
                return Ok(logits);
            }
            // Production Vulkan keeps the generic linear-decode family behind
            // an explicit quarantine, but layer-bounded prefill can still
            // arrive here as the separately qualified resident rank-2 mixed
            // matmul: F32 `[rows, hidden]` x BF16 `[hidden, vocab]` -> F32.
            // Give the backend's exact matmul contract a chance before the
            // ordinary equal-dtype tensor fallback. This does not widen the
            // linear-decode policy and other backends simply decline or own
            // their already-advertised request.
            if let Some(logits) = runtime_matmul_no_broadcast_copy(backend, x, embed_tokens_t)? {
                return Ok(logits);
            }
        }
    }
    lm_head_forward(x, embed_tokens_t)
}

pub(super) fn lm_head_argmax_with_backend(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    embed_tokens_t: &Tensor,
) -> Result<u32> {
    #[cfg(feature = "metal")]
    {
        if crate::backend::metal::metal_lm_head_argmax_supports(x, embed_tokens_t) {
            return crate::backend::metal::metal_lm_head_argmax_bf16(x, embed_tokens_t)
                .context("metal lm_head argmax kernel failed");
        }
    }
    // Fused matmul + argmax path: when stable KT routes are enabled, chain
    // `MatmulOp` -> `cuda_argmax_last_axis` directly in kt-storage,
    // skipping the intermediate buffer the
    // unfused composition would allocate between the two stages.
    #[cfg(feature = "cuda")]
    if let Some(token) = try_kt_lm_head_argmax(x, embed_tokens_t)? {
        return Ok(token);
    }
    let logits = if backend.is_some() {
        lm_head_forward_backend_decode_if(backend, x, embed_tokens_t)?
    } else {
        lm_head_forward(x, embed_tokens_t)?
    };
    let logits_1d = logits.flatten_all()?;
    #[cfg(feature = "cuda")]
    if crate::kt_api_policy::stable_routes_enabled()
        && matches!(logits_1d.device(), Device::Cuda(_))
        && matches!(logits_1d.dtype(), DType::F32 | DType::BF16 | DType::F16)
        && logits_1d.is_contiguous()
        && let Some(token) = try_kt_argmax_1d(&logits_1d)?
    {
        return Ok(token);
    }
    // #1082: kt `argmax` returns I64 indices (candle returned U32). Read i64 and
    // narrow to the u32 token id. (The CUDA fast paths above return early; this
    // fallback runs on the no-CUDA / kt-API-off path.)
    let index = logits_1d.argmax(0)?.flatten_all()?;
    let values = crate::execution_phase::profile_accelerator_readback(index.device(), || {
        index.to_vec1::<i64>()
    })?;
    Ok(values[0] as u32)
}

pub(super) fn lm_head_argmax(x: &Tensor, embed_tokens_t: &Tensor) -> Result<u32> {
    lm_head_argmax_with_backend(None, x, embed_tokens_t)
}

/// Phase 7 (#1082) — fused kt-API LM head + argmax migration helper.
/// Routes the `[1, 1, hidden] @ [hidden, vocab] -> argmax` pipeline
/// through `kiln_tensor::ops::matmul` followed directly by
/// `kiln_tensor::cuda_argmax_last_axis` in kt-storage, skipping the
/// intermediate buffer that the unfused
/// `try_kt_lm_head` -> `try_kt_argmax_1d` composition would allocate.
///
/// Only fires when the LM head input flattens to exactly one row
/// (`lead == 1`) — the canonical single-token decode case in
/// [`lm_head_argmax`]. For multi-row inputs (e.g. prefill), the
/// caller continues to chain the unfused `lm_head_forward` +
/// `argmax(0)` after `flatten_all`, which is correct but not
/// applicable to argmax-over-flattened-1D semantics on those shapes.
///
/// Requires stable KT routes and falls through cleanly when they are disabled.
/// NVTX range `kiln/lm_head_argmax_kt` brackets the migrated call.
#[cfg(feature = "cuda")]
pub(super) fn try_kt_lm_head_argmax(x: &Tensor, embed_tokens_t: &Tensor) -> Result<Option<u32>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    let l_dims = x.dims();
    let r_dims = embed_tokens_t.dims();
    if r_dims.len() != 2 || l_dims.len() < 2 {
        return Ok(None);
    }
    let k = l_dims[l_dims.len() - 1];
    if r_dims[0] != k {
        return Ok(None);
    }
    let lead: usize = l_dims[..l_dims.len() - 1].iter().product();
    // Fused path only handles the single-row case so the matmul
    // output's last-axis argmax matches the flattened-1D argmax
    // semantics of the kt composite baseline. Multi-row inputs would
    // need argmax over the full flattened logits, not per-row.
    if lead != 1 {
        return Ok(None);
    }
    if !matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32)
        || x.dtype() != embed_tokens_t.dtype()
        || !x.is_contiguous()
        || !embed_tokens_t.is_contiguous()
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/lm_head_argmax_kt");

    let x2d = match x.reshape((lead, k)) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    // #1082 forward-flip: `x2d` / `embed_tokens_t` are already kt; run the
    // kt matmul + argmax directly and read the scalar back (no candle
    // round-trip).
    let logits_kt = match kiln_tensor::ops::matmul(&x2d, embed_tokens_t) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    // `cuda_argmax_last_axis` reduces the last axis. On the
    // `[1, vocab]` matmul output it yields a rank-1 `[1]` I64 tensor.
    let argmax_kt = match kiln_tensor::cuda_argmax_last_axis(&logits_kt) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let index = argmax_kt.flatten_all()?;
    let token_i64 = crate::execution_phase::profile_accelerator_readback(index.device(), || {
        index.to_vec1::<i64>()
    })?[0];
    Ok(Some(token_i64 as u32))
}

/// Phase 7 (#1082) — kt-API argmax migration helper. Routes a 1-D
/// contiguous CUDA kt tensor through
/// `kiln_tensor::cuda_argmax_last_axis` (which on rank-1 reduces
/// to a single I64 scalar). The result is read back as a kt scalar
/// and then cast to `u32` to match the
/// existing return type.
///
/// Returns `Ok(None)` on any incompatibility so the caller falls
/// through to the kt `argmax` op. NVTX range `kiln/argmax_kt` brackets
/// the migrated call so nsys traces separate the path from the
/// baseline.
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_argmax_1d(x: &Tensor) -> Result<Option<u32>> {
    if !matches!(x.device(), Device::Cuda(_))
        || !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
        || !x.is_contiguous()
        || x.rank() != 1
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/argmax_kt");

    // #1082 forward-flip: `x` is already kt; run the kt argmax directly
    // and read the scalar back without the candle round-trip.
    let out_kt = match kiln_tensor::cuda_argmax_last_axis(x) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    // kt_argmax returns I64; cuda_argmax_last_axis on a rank-1 input
    // yields a rank-0 scalar tensor — flatten to rank-1 [1] and read.
    let index = out_kt.flatten_all()?;
    let token_i64 = crate::execution_phase::profile_accelerator_readback(index.device(), || {
        index.to_vec1::<i64>()
    })?[0];
    Ok(Some(token_i64 as u32))
}

pub(super) fn lm_head_argmax_backend_decode_if(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    embed_tokens_t: &Tensor,
) -> Result<u32> {
    if let Some(backend) = backend
        && let Some(token) =
            SamplingBackend::runtime_linear_decode_argmax(backend, x, embed_tokens_t)?
    {
        return Ok(token);
    }
    lm_head_argmax_with_backend(backend, x, embed_tokens_t)
}

/// Phase 7 (#1082) — kt-API sampler argmax migration helper for
/// the multi-row case used by
/// [`crate::sampling::greedy_sample_rows`]. Routes a 2-D or
/// higher contiguous CUDA kt logits tensor through
/// `kiln_tensor::cuda_argmax_last_axis`, which reduces the last
/// axis and yields an I64 tensor with one fewer rank. The result
/// is read back as kt,
/// flattened, and cast to a `Vec<u32>` to match the existing
/// `greedy_sample_rows` return type.
///
/// Distinct from [`try_kt_argmax_1d`]: that helper targets the
/// post-`flatten_all` rank-1 case inside the fused LM head fast
/// path ([`lm_head_argmax`]) and returns a scalar `u32`. The
/// sampler path operates on per-row argmax over the vocab axis
/// (`[..., vocab_size]` -> `[...]`) so it needs the rank-preserving
/// reduction, not the rank-1 scalar collapse.
///
/// Default-on for compatible CUDA tensors. Returns `Ok(None)` when
/// the input is incompatible (non-CUDA, unsupported dtype,
/// non-contiguous, rank-0) so the caller falls through to the kt
/// `argmax(vocab_dim)` + `flatten_all` + `to_vec1::<u32>()` op chain. NVTX range
/// `kiln/sampling_argmax_rows_kt` brackets the migrated call so
/// nsys traces separate the path from the baseline.
///
/// Wired into [`crate::sampling::greedy_sample_rows`]: when the
/// logits tensor is a contiguous CUDA tensor of a supported dtype,
/// the kt-API path runs and the rest of the kt composite
/// (`argmax(vocab_dim)` + `flatten_all` + `to_vec1::<u32>()`) is
/// bypassed entirely. The fallback remains for CPU/Metal/Vulkan,
/// non-contiguous views, and unsupported dtypes while the public
/// sampler signature is kt-typed.
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_sampling_argmax_rows(logits: &Tensor) -> Result<Option<Vec<u32>>> {
    if !matches!(logits.device(), Device::Cuda(_))
        || !matches!(logits.dtype(), DType::F32 | DType::BF16 | DType::F16)
        || !logits.is_contiguous()
        || logits.rank() == 0
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/sampling_argmax_rows_kt");

    let out_kt = match kiln_tensor::cuda_argmax_last_axis(logits) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    // cuda_argmax_last_axis on a rank-N input yields a rank-(N-1) I64
    // tensor; flatten and cast each element to u32 to match the
    // existing greedy_sample_rows return type.
    let indices = out.flatten_all()?;
    let ids_i64: Vec<i64> =
        crate::execution_phase::profile_accelerator_readback(indices.device(), || {
            indices.to_vec1::<i64>()
        })?;
    Ok(Some(ids_i64.into_iter().map(|v| v as u32).collect()))
}

/// Token-history aggregation for the fused on-device sampling path.
/// Returns `(unique_indices, counts)` sorted by ascending token id so
/// the on-device scatter is deterministic across runs.
pub(super) fn unique_history_counts(history: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut counts: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for &t in history {
        *counts.entry(t).or_default() += 1;
    }
    let mut idx = Vec::with_capacity(counts.len());
    let mut cnt = Vec::with_capacity(counts.len());
    for (k, v) in counts {
        idx.push(k);
        cnt.push(v);
    }
    (idx, cnt)
}

/// Hidden-state → sampled token, fully fused on-device when the backend
/// supports it. Returns `Ok(Some(token))` when the fused path ran;
/// `Ok(None)` when the backend declined (e.g. `top_k > kernel max`),
/// signalling to the caller that the legacy host sampler should run.
///
/// `params` is the full sampling spec (Qwen3.5-shaped). `step_seed` is
/// the per-step PRNG seed (overrides `params.seed` for this token);
/// `history` is the slice of generated tokens so far. Pass `&[]` for
/// the first decode token — penalties become a no-op under OpenAI
/// semantics.
#[allow(clippy::too_many_arguments)]
pub fn lm_head_sample_backend_decode_if(
    backend: Option<&dyn BackendRuntime>,
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    params: &kiln_core::sampling::SamplingParams,
    step_seed: Option<u64>,
    history: &[u32],
) -> Result<Option<u32>> {
    let profiled = lm_head_sample_backend_decode_profiled_if(
        backend, hidden, weights, config, params, step_seed, history,
    )?;
    Ok(profiled.map(|profiled| {
        if let Some(duration) = profiled.readback_duration {
            crate::execution_phase::observe_profiled_readback(duration);
        }
        profiled.value
    }))
}

#[derive(Debug)]
pub(crate) struct ProfiledBackendSample<T> {
    pub value: T,
    pub readback_duration: Option<Duration>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn lm_head_sample_backend_decode_profiled_if(
    backend: Option<&dyn BackendRuntime>,
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    params: &kiln_core::sampling::SamplingParams,
    step_seed: Option<u64>,
    history: &[u32],
) -> Result<Option<ProfiledBackendSample<u32>>> {
    let Some(backend) = backend else {
        return Ok(None);
    };
    let seed = step_seed.unwrap_or_else(|| {
        // PRNG seed for un-seeded requests — derived from nanos +
        // history hash so consecutive un-seeded tokens see distinct
        // entropy without burning a kernel side channel for it.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let h = history.iter().fold(0xCBF29CE484222325u64, |acc, &t| {
            (acc ^ t as u64).wrapping_mul(0x100000001B3)
        });
        nanos.wrapping_add(h)
    });
    let penalty_history: &[u32] = if params.token_penalties_are_no_op() {
        &[]
    } else {
        history
    };
    let (history_indices, history_counts) = unique_history_counts(penalty_history);

    #[cfg(feature = "rocm")]
    {
        use kiln_core::sampling::SamplingParams as SP;

        let temperatures = [params.temperature];
        let top_k = [params.top_k];
        if rocm_w8_batch_sample_supported(backend, weights, &temperatures, &top_k) {
            let history_rows = vec![0u32; history_indices.len()];
            if let Some(profiled) = rocm_w8_lm_head_sample_batch_profiled(
                backend,
                hidden,
                weights,
                config,
                &history_rows,
                &history_indices,
                &history_counts,
                &[params.repetition_penalty],
                &[params.presence_penalty],
                &[params.frequency_penalty],
                &temperatures,
                &top_k,
                &[params.top_p],
                &[params.min_p],
                &[seed],
            )? {
                return profiled
                    .value
                    .into_iter()
                    .next()
                    .map(|value| {
                        Some(ProfiledBackendSample {
                            value,
                            readback_duration: profiled.readback_duration,
                        })
                    })
                    .context("ROCm W8 single-row sampler returned no token");
            }
        }

        if crate::rocm_policy::current_rocm_kernel_policy().w8_sampled_lm_head
            && params.top_k == 0
            && SP::top_p_disables_nucleus_filter(params.top_p)
            && SP::min_p_is_disabled(params.min_p)
            && params.temperature.is_finite()
            && params.temperature > 0.0
            && let Some(lm_head_w8) = weights.lm_head_w8.as_ref()
        {
            let normed = rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)?;
            let dims = normed.dims();
            let lead: usize = dims[..dims.len().saturating_sub(1)].iter().product();
            if lead == 1
                && normed.dtype() == DType::BF16
                && !normed.track_op()
                && matches!(normed.device(), Device::Rocm(_))
            {
                let normed = normed
                    .contiguous()
                    .context("rocm w8 sampled lm_head normed contiguous")?;
                let profiled = crate::rocm_w8_proj::gumbel_sample_bf16_profiled(
                    &normed,
                    lm_head_w8,
                    &history_indices,
                    &history_counts,
                    params.repetition_penalty,
                    params.presence_penalty,
                    params.frequency_penalty,
                    params.temperature,
                    seed,
                )
                .context("rocm w8 sampled lm_head gumbel sample")?;
                return Ok(Some(ProfiledBackendSample {
                    value: profiled.value,
                    readback_duration: Some(profiled.readback_duration),
                }));
            }
        }
    }

    if !SamplingBackend::runtime_supports_linear_decode_sample(backend, params.top_k) {
        return Ok(None);
    }
    let normed = rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)?;
    SamplingBackend::runtime_linear_decode_sample(
        backend,
        &normed,
        &weights.embed_tokens_t,
        &history_indices,
        &history_counts,
        params.repetition_penalty,
        params.presence_penalty,
        params.frequency_penalty,
        params.temperature,
        params.top_k,
        params.top_p,
        params.min_p,
        seed,
    )
    .map(|sample| {
        sample.map(|value| ProfiledBackendSample {
            value,
            readback_duration: None,
        })
    })
}

pub(super) fn lm_head_argmax_rows_with_backend(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    embed_tokens_t: &Tensor,
) -> Result<Vec<u32>> {
    #[cfg(feature = "metal")]
    {
        if crate::backend::metal::metal_lm_head_argmax_rows_supports(x, embed_tokens_t) {
            return crate::backend::metal::metal_lm_head_argmax_rows_bf16(x, embed_tokens_t)
                .context("metal batch lm_head argmax kernel failed");
        }
    }
    let logits = if backend.is_some() {
        lm_head_forward_backend_decode_if(backend, x, embed_tokens_t)?
    } else {
        lm_head_forward(x, embed_tokens_t)?
    };
    // #1082: greedy_sample_rows is kt-native now — pass kt logits directly
    // (the former `kt_logits_to_candle` bridge island was removed; works on any
    // build).
    crate::sampling::greedy_sample_rows(&logits).context("batched greedy row sampling failed")
}

pub(super) fn lm_head_argmax_rows_backend_decode_if(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    embed_tokens_t: &Tensor,
) -> Result<Vec<u32>> {
    if let Some(backend) = backend
        && SamplingBackend::runtime_supports_linear_decode_argmax_batch(backend)
        && let Some(tokens) =
            SamplingBackend::runtime_linear_decode_argmax_batch(backend, x, embed_tokens_t)?
    {
        return Ok(tokens);
    }
    lm_head_argmax_rows_with_backend(backend, x, embed_tokens_t)
}

#[cfg(feature = "rocm")]
pub(super) fn rocm_w8_lm_head_argmax_rows(
    backend: &dyn BackendRuntime,
    normed: &Tensor,
    weights: &GpuWeights,
) -> Result<Option<Vec<u32>>> {
    if !SamplingBackend::runtime_supports_quantized_lm_head_argmax_batch(backend)
        || normed.dtype() != DType::BF16
        || normed.track_op()
    {
        return Ok(None);
    }
    let Some(lm_head_w8) = weights.lm_head_w8.as_ref() else {
        return Ok(None);
    };
    let dims = normed.dims();
    if dims.len() < 2 || dims.last().copied() != Some(lm_head_w8.k) {
        return Ok(None);
    }
    let rows: usize = dims[..dims.len() - 1].iter().product();
    if rows == 0 {
        return Ok(None);
    }
    let normed = normed
        .contiguous()
        .context("rocm w8 batched argmax normed contiguous")?;
    crate::rocm_w8_proj::argmax_batch_bf16(&normed, lm_head_w8)
        .map(Some)
        .context("rocm w8 batched lm_head argmax")
}

#[cfg(not(feature = "rocm"))]
pub(super) fn rocm_w8_lm_head_argmax_rows(
    _backend: &dyn BackendRuntime,
    _normed: &Tensor,
    _weights: &GpuWeights,
) -> Result<Option<Vec<u32>>> {
    Ok(None)
}

pub(crate) fn rocm_w8_batch_sample_supported(
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    temperatures: &[f32],
    top_k: &[u32],
) -> bool {
    #[cfg(feature = "rocm")]
    {
        SamplingBackend::runtime_supports_quantized_lm_head_sample_batch(
            backend,
            top_k,
            temperatures,
        ) && weights.lm_head_w8.is_some()
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (backend, weights, temperatures, top_k);
        false
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rocm_w8_lm_head_sample_batch_profiled(
    backend: &dyn BackendRuntime,
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    history_rows: &[u32],
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalties: &[f32],
    presence_penalties: &[f32],
    frequency_penalties: &[f32],
    temperatures: &[f32],
    top_k: &[u32],
    top_p: &[f32],
    min_p: &[f32],
    seeds: &[u64],
) -> Result<Option<ProfiledBackendSample<Vec<u32>>>> {
    if !rocm_w8_batch_sample_supported(backend, weights, temperatures, top_k) {
        return Ok(None);
    }
    #[cfg(feature = "rocm")]
    {
        let lm_head_w8 = weights
            .lm_head_w8
            .as_ref()
            .expect("support predicate requires packed ROCm LM head");
        let normed = rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)?;
        if normed.dtype() != DType::BF16 || normed.track_op() {
            return Ok(None);
        }
        let dims = normed.dims();
        let rows = dims
            .get(..dims.len().saturating_sub(1))
            .map(|leading| leading.iter().product::<usize>())
            .unwrap_or(0);
        if rows != temperatures.len() || dims.last().copied() != Some(lm_head_w8.k) {
            return Ok(None);
        }
        let normed = normed
            .contiguous()
            .context("rocm w8 batched sample normed contiguous")?;
        crate::rocm_w8_proj::sample_batch_bf16_profiled(
            &normed,
            lm_head_w8,
            history_rows,
            history_indices,
            history_counts,
            repetition_penalties,
            presence_penalties,
            frequency_penalties,
            temperatures,
            top_k,
            top_p,
            min_p,
            seeds,
        )
        .map(|profiled| {
            Some(ProfiledBackendSample {
                value: profiled.value,
                readback_duration: Some(profiled.readback_duration),
            })
        })
        .context("rocm w8 batched lm_head sample")
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (
            hidden,
            config,
            history_rows,
            history_indices,
            history_counts,
            repetition_penalties,
            presence_penalties,
            frequency_penalties,
            top_p,
            min_p,
            seeds,
        );
        Ok(None)
    }
}

pub(super) fn lm_head_weighted_prep_argmax(
    x: &Tensor,
    norm_weight: &Tensor,
    embed_tokens_t: &Tensor,
) -> Result<Option<u32>> {
    if weighted_lm_head_prep_disabled()
        || x.dtype() != DType::BF16
        || norm_weight.dtype() != DType::BF16
        || !matches!(x.device(), Device::Metal(_))
        || !matches!(norm_weight.device(), Device::Metal(_))
    {
        return Ok(None);
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    if batch != 1 || seq_len != 1 || norm_weight.dims() != [hidden] {
        return Ok(None);
    }

    let weighted = x.broadcast_mul(norm_weight)?.contiguous()?;
    Ok(Some(lm_head_argmax(&weighted, embed_tokens_t)?))
}
