//! `kiln_tensor::Tensor`-typed surface alongside the candle-typed
//! OPD top-K reverse-KL API.
//!
//! # Status
//!
//! Phase 7 prep — same migration pattern as kiln-flce-kernel,
//! kiln-flash-attn, kiln-conv1d-kernel, kiln-rmsnorm-kernel,
//! kiln-gdn-kernel, and kiln-marlin-gemm (see #1082 line 322
//! "kernel-crate kt-API ports").
//!
//! Like kiln-flce-kernel, the (#1082) candle-drop moved this crate's
//! candle-typed glue UP into `kiln-train::opd_tape_shim`: the Phase A
//! reference path (`candle_core::Tensor` gather + matmul + log-softmax)
//! and the candle `CustomOp1` kt-forward-op shim that wraps this kt
//! surface for candle autograd. The raw CUDA FFI in `phase_b.rs` is only
//! the inner kernel dispatch and is now pure-kt (it takes device
//! pointers via `kiln-kt-bridge`, no candle storage conversions).
//!
//! The kt-typed surface this module defines is therefore the
//! *migration target*: when Phase B's `forward_inner` /
//! `backward_inner` are eventually re-implemented over
//! `kiln_tensor::Tensor` (the natural next step once
//! `kiln-flce-kernel::kt_api` lands its Phase B kt-rewrite), the
//! kt-typed entry points below are the public surface they will plug
//! into.
//!
//! This module ships:
//!
//! 1. [`OpdLossError`] — kt-typed error, independent of candle /
//!    anyhow.
//! 2. [`opd_top_k_reverse_kl_kt`] — kt-typed scalar-mean forward
//!    entry point. Re-implements the gather + matmul + log-softmax
//!    + reverse-KL reduction over [`kiln_tensor`] ops; mirrors the
//!    Phase A candle reference (the per-position variant
//!    `kiln_train::opd_tape_shim::opd_top_k_reverse_kl_phase_a_per_position`
//!    — the scalar-mean phase_a entry was deleted in #1082 since the
//!    only callers were internal dead code).
//! 3. [`opd_top_k_reverse_kl_per_position_kt`] — kt-typed per-
//!    position forward entry point. Same forward kernel as the
//!    scalar entry point but without the final `mean_all`.
//! 4. [`compute_per_position_metrics_kt`] — kt-typed diagnostics
//!    entry point. Reuses the same forward kernel and additionally
//!    computes student / teacher entropy over the K-support.
//!
//! Forward only — backward is still TBD via the same kiln-autograd
//! hooks the FLCE kt-typed backward is waiting on; the candle
//! kt-shim (`kiln_train::opd_tape_shim::opd_top_k_reverse_kl_per_position_via_kt_forward_op`)
//! continues to own production gradient flow until that lands.
//! Historical: pre-`#1082`-2026-05-28 the candle `CustomOp1`
//! `OpdLossCustomOp` (in `phase_b.rs`) wrapped the backward; that
//! wrapper was deleted once production migrated to the kt-shim. The
//! fused backward FFI symbols `kiln_opd_topk_kl_bwd_{bf16,f32}`
//! survive in the trimmed `phase_b.rs` and are still called from
//! `opd_top_k_reverse_kl_phase_b_bwd_kt` below. (#1082)
//!
//! # Numerical contract
//!
//! Numerically equivalent to the candle Phase A reference up to
//! floating-point associativity in the per-row matmul + log-softmax
//! reductions. The kernel sequence (gather → matmul → renormalise →
//! KL reduce) is identical and the reductions are bounded by K (the
//! teacher's support size), so there's no chunked tail.

use kiln_tensor::{
    ops::{
        exp, index_select, log_softmax_last_dim, matmul, mean_all, mul, neg, sub, sum_axis, to_f32,
    },
    Error as KtError, Tensor as KtTensor,
};

// `DType` is used by both the CUDA fused backward (the dtype gate) and
// the device-agnostic kt-composite backward
// ([`opd_top_k_reverse_kl_phase_b_bwd_composite_kt`], which runs on
// CPU/Metal/CUDA) — so it's unconditional. The test module aliases it
// again locally.
use kiln_tensor::DType as KtDType;

#[cfg(any(feature = "cuda", feature = "rocm"))]
use kiln_kt_bridge::BridgeError;

/// Error type for the kiln-tensor-typed OPD loss surface.
///
/// Mirrors `kiln-flce-kernel::kt_api::FlceError` — kept separate
/// from `anyhow::Error` (the candle-typed surface's error) so Phase 7
/// can delete candle without rewriting any kt-typed call site.
#[derive(Debug)]
pub enum OpdLossError {
    /// Generic message error for shape / dtype validation failures
    /// in the kt-typed entry points.
    Msg(String),
    /// Underlying `kiln_tensor` op error surfaced from the forward
    /// gather + matmul + KL reduction.
    Kt(KtError),
    /// The kt-typed entry point exists but its body has not yet been
    /// implemented. Reserved for future backward / extra entry
    /// points; the production forwards
    /// [`opd_top_k_reverse_kl_kt`] /
    /// [`opd_top_k_reverse_kl_per_position_kt`] /
    /// [`compute_per_position_metrics_kt`] no longer return this
    /// variant.
    NotYetImplemented(&'static str),
}

impl std::fmt::Display for OpdLossError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpdLossError::Msg(m) => f.write_str(m),
            OpdLossError::Kt(e) => write!(f, "kt-opd-loss: {e}"),
            OpdLossError::NotYetImplemented(name) => write!(
                f,
                "kt-opd-loss: {name} is not yet implemented; use the candle-typed entry point",
            ),
        }
    }
}

impl std::error::Error for OpdLossError {}

impl OpdLossError {
    pub fn msg(s: impl Into<String>) -> Self {
        OpdLossError::Msg(s.into())
    }
}

impl From<KtError> for OpdLossError {
    fn from(e: KtError) -> Self {
        OpdLossError::Kt(e)
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl From<BridgeError> for OpdLossError {
    fn from(e: BridgeError) -> Self {
        OpdLossError::Msg(format!("kt-opd-loss bridge: {}", e.message))
    }
}

/// Validate the shape / dtype contract on the kt-typed entry points
/// and return the `(T, H, V, T_active, K)` quintuple for downstream
/// code. Mirrors the candle-typed `crate::validate_inputs` but uses
/// `kiln_tensor::Tensor::shape()` (slice) rather than candle's
/// `Tensor::dims()` (slice), and returns [`OpdLossError`] instead of
/// `anyhow::Error`.
fn validate_inputs_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<(usize, usize, usize, usize, usize), OpdLossError> {
    let hidden_dims = hidden.shape();
    if hidden_dims.len() != 3 {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: hidden must be 3-D [1, T, H]; got {hidden_dims:?}",
        )));
    }
    if hidden_dims[0] != 1 {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: hidden batch dim must be 1 (kiln trainer convention); got {hidden_dims:?}",
        )));
    }
    let seq_len = hidden_dims[1];
    let hidden_size = hidden_dims[2];

    let head_dims = head_t.shape();
    if head_dims.len() != 2 {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: head_t must be 2-D [H, V]; got {head_dims:?}",
        )));
    }
    if head_dims[0] != hidden_size {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: hidden_size mismatch: hidden has H={hidden_size} but head_t has H={}",
            head_dims[0],
        )));
    }
    let vocab_size = head_dims[1];

    if label_mask.len() != seq_len {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: label_mask length {} does not match T {seq_len}",
            label_mask.len(),
        )));
    }
    if top_k == 0 {
        return Err(OpdLossError::msg("kt-opd-loss: top_k must be > 0"));
    }

    let active_count = label_mask.iter().filter(|&&m| m).count();
    let expected_logits = active_count * top_k;
    if teacher_topk_indices.len() != expected_logits {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: teacher_topk_indices length {} != T_active * K = {active_count} * {top_k} = {expected_logits}",
            teacher_topk_indices.len(),
        )));
    }
    if teacher_topk_logprobs.len() != expected_logits {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: teacher_topk_logprobs length {} != T_active * K = {active_count} * {top_k} = {expected_logits}",
            teacher_topk_logprobs.len(),
        )));
    }
    for (i, &idx) in teacher_topk_indices.iter().enumerate() {
        if (idx as usize) >= vocab_size {
            return Err(OpdLossError::msg(format!(
                "kt-opd-loss: teacher_topk_indices[{i}] = {idx} >= vocab_size {vocab_size}",
            )));
        }
    }

    Ok((seq_len, hidden_size, vocab_size, active_count, top_k))
}

/// Build a rank-0 F32 scalar tensor holding 0.0. Used by the
/// scalar-mean entry point when there are no active positions.
fn zero_scalar() -> Result<KtTensor, OpdLossError> {
    KtTensor::from_vec(vec![0.0f32], vec![]).map_err(OpdLossError::Kt)
}

/// Build an empty 1-D F32 tensor of shape `[0]`. Used by the
/// per-position entry point when there are no active positions.
fn empty_per_position() -> Result<KtTensor, OpdLossError> {
    KtTensor::from_vec(Vec::<f32>::new(), vec![0]).map_err(OpdLossError::Kt)
}

/// Forward kernel shared by [`opd_top_k_reverse_kl_kt`],
/// [`opd_top_k_reverse_kl_per_position_kt`], and
/// [`compute_per_position_metrics_kt`].
///
/// Computes the per-position reverse KL over the teacher's K
/// support, returning a 1-D `[T_active]` F32 tensor of KL values.
/// Mirrors the candle reference `per_position_phase_a` one-for-one:
///
///   1. Squeeze hidden batch dim 0; index_select active rows from
///      hidden → `[T_active, H]` F32.
///   2. Gather K columns of `head_t` per active token:
///      `head_t.index_select(1, teacher_topk_indices_flat)` →
///      `[H, T_active * K]`. Reshape to `[H, T_active, K]`, permute
///      to `[T_active, H, K]`, contiguous.
///   3. Batched matmul `[T_active, 1, H] @ [T_active, H, K]` →
///      `[T_active, 1, K]`, squeeze → `[T_active, K]` F32 student
///      logits at the K support.
///   4. `log_softmax_last_dim` on both student logits and the
///      teacher-provided log-probabilities → `log_p_hat`, `log_q_hat`.
///   5. `p_hat = exp(log_p_hat)`, `diff = log_p_hat - log_q_hat`,
///      `KL[t] = sum_k p_hat[t, k] * diff[t, k]`. Returns the 1-D
///      `[T_active]` KL vector.
///
/// Also returns `(log_p_hat, log_q_hat)` so the per-position metrics
/// path can compute entropies without redoing the kernel.
///
/// Caller must short-circuit `active_count == 0`; this function
/// expects at least one active row.
fn per_position_forward_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    active_positions: &[u32],
    top_k: usize,
) -> Result<(KtTensor, KtTensor, KtTensor), OpdLossError> {
    let active_count = active_positions.len();
    debug_assert!(active_count > 0, "caller short-circuits empty");

    // All host-built index / scratch tensors below must live on the
    // same device as `hidden` — kt's `DeviceOp2` ops (`index_select`,
    // `sub`, `mul`, etc.) reject mixed-device inputs. Pre-#1082
    // production callers of `opd_top_k_reverse_kl_per_position_kt`
    // were all CPU; the OPD-trainer migration introduced via
    // `kiln-opd-loss-kernel::kt_forward_op` (commit (#1082)) is the
    // first CUDA caller, which is why this device-awareness fix
    // landed alongside the production-caller migration. (#1082 Metal
    // lane) Metal is the second GPU caller — the OPD FORWARD runs on
    // Metal storage (only the fused CUDA backward stays CUDA-only), so
    // the Metal arm builds the host index/scratch tensor on CPU and
    // moves it to the Metal device via `to_device` (`host_to_metal_copy`).
    use kiln_tensor::Device as KtDevice;
    let dev = hidden.device();
    let upload_u32 = |vals: &[u32], shape: Vec<usize>| -> Result<KtTensor, OpdLossError> {
        match dev {
            KtDevice::Cpu => KtTensor::from_vec(vals.to_vec(), shape).map_err(OpdLossError::Kt),
            #[cfg(feature = "cuda")]
            KtDevice::Cuda(i) => {
                KtTensor::cuda_from_slice(vals, shape, i).map_err(OpdLossError::Kt)
            }
            #[cfg(feature = "metal")]
            KtDevice::Metal(_) => KtTensor::from_vec(vals.to_vec(), shape)
                .and_then(|t| t.to_device(dev))
                .map_err(OpdLossError::Kt),
            // (#1082) Vulkan mirrors Metal: build the host index/scratch tensor
            // on CPU and move it to the Vulkan device via `to_device`
            // (`host_to_vulkan_copy`). Needed so the F32-on-Vulkan OPD forward
            // can materialize the teacher top-K index tensor.
            #[cfg(feature = "vulkan")]
            KtDevice::Vulkan(_) => KtTensor::from_vec(vals.to_vec(), shape)
                .and_then(|t| t.to_device(dev))
                .map_err(OpdLossError::Kt),
            // (Phase R.7) ROCm builds the host index tensor on CPU and uploads
            // via `from_vec_on(Device::Rocm(i), ...)` (-> `host_to_rocm_copy`),
            // so the F32/BF16-on-ROCm OPD forward + backward can materialize the
            // teacher top-K index tensor on device.
            #[cfg(feature = "rocm")]
            KtDevice::Rocm(i) => {
                KtTensor::from_vec_on(KtDevice::Rocm(i), vals.to_vec(), shape)
                    .map_err(OpdLossError::Kt)
            }
            #[allow(unreachable_patterns)]
            other => Err(OpdLossError::msg(format!(
                "kt-opd-loss: unsupported device for index tensor {other}"
            ))),
        }
    };
    let upload_f32 = |vals: &[f32], shape: Vec<usize>| -> Result<KtTensor, OpdLossError> {
        match dev {
            KtDevice::Cpu => KtTensor::from_vec(vals.to_vec(), shape).map_err(OpdLossError::Kt),
            #[cfg(feature = "cuda")]
            KtDevice::Cuda(i) => {
                KtTensor::cuda_from_slice(vals, shape, i).map_err(OpdLossError::Kt)
            }
            #[cfg(feature = "metal")]
            KtDevice::Metal(_) => KtTensor::from_vec(vals.to_vec(), shape)
                .and_then(|t| t.to_device(dev))
                .map_err(OpdLossError::Kt),
            // (#1082) Vulkan mirrors Metal (host CPU build -> to_device).
            #[cfg(feature = "vulkan")]
            KtDevice::Vulkan(_) => KtTensor::from_vec(vals.to_vec(), shape)
                .and_then(|t| t.to_device(dev))
                .map_err(OpdLossError::Kt),
            // (Phase R.7) ROCm mirrors the index path (host CPU build ->
            // `from_vec_on(Device::Rocm(i), ...)`).
            #[cfg(feature = "rocm")]
            KtDevice::Rocm(i) => {
                KtTensor::from_vec_on(KtDevice::Rocm(i), vals.to_vec(), shape)
                    .map_err(OpdLossError::Kt)
            }
            #[allow(unreachable_patterns)]
            other => Err(OpdLossError::msg(format!(
                "kt-opd-loss: unsupported device for q_logprobs {other}"
            ))),
        }
    };

    // Step 1: gather active rows from hidden and cast to F32.
    //
    // hidden is `[1, T, H]`; squeeze batch dim → `[T, H]`. Then
    // index_select(0, active_positions) → `[T_active, H]`.
    let hidden_2d = hidden.squeeze(0)?;
    let active_idx = upload_u32(active_positions, vec![active_count])?;
    let active_hidden = index_select(&hidden_2d, 0, &active_idx)?;
    let active_hidden_f32 = to_f32(&active_hidden)?;
    let head_t_f32 = to_f32(head_t)?;

    // Step 2: gather K columns per active token from `head_t`.
    //
    // teacher_topk_indices is the row-major flat `[T_active * K]`
    // index buffer. `head_t.index_select(1, flat)` yields
    // `[H, T_active * K]`. Reshape to `[H, T_active, K]` and
    // permute to `[T_active, H, K]`.
    let hidden_size = head_t.shape()[0];
    let flat_indices = upload_u32(teacher_topk_indices, vec![active_count * top_k])?;
    let gathered = index_select(&head_t_f32, 1, &flat_indices)?;
    let reshaped = gathered.reshape(vec![hidden_size, active_count, top_k])?;
    let head_gather = reshaped.permute(&[1, 0, 2])?.contiguous()?;

    // Step 3: batched matmul. `active_hidden_f32` is `[T_active, H]`;
    // unsqueeze(1) → `[T_active, 1, H]`. matmul against
    // `[T_active, H, K]` → `[T_active, 1, K]`; squeeze(1) →
    // `[T_active, K]`.
    let lhs = active_hidden_f32.unsqueeze(1)?;
    let s_logits = matmul(&lhs, &head_gather)?.squeeze(1)?;

    // Step 4: renormalise both distributions over the K support.
    //
    // `log_softmax_last_dim` requires contiguous inputs. The matmul
    // output is contiguous, and we build q_logprobs fresh on the
    // same device as `s_logits` (= hidden's device).
    let s_logits = s_logits.contiguous()?;
    let q_logprobs = upload_f32(teacher_topk_logprobs, vec![active_count, top_k])?;
    let log_p_hat = log_softmax_last_dim(&s_logits)?;
    let log_q_hat = log_softmax_last_dim(&q_logprobs)?;

    // Step 5: per-position reverse KL.
    //
    //   p_hat        = exp(log_p_hat)                  [T_active, K]
    //   diff         = log_p_hat - log_q_hat           [T_active, K]
    //   per_token[t] = sum_k p_hat[t, k] * diff[t, k]  [T_active]
    let p_hat = exp(&log_p_hat)?;
    let diff = sub(&log_p_hat, &log_q_hat)?;
    let prod = mul(&p_hat, &diff)?;
    let per_token = sum_axis(&prod, 1)?;

    Ok((per_token, log_p_hat, log_q_hat))
}

/// kt-typed entry point for OPD scalar-mean reverse-KL.
///
/// Re-implements the (now-deleted, #1082) candle Phase A scalar-mean
/// reference over `kiln_tensor` ops. Numerically equivalent up to
/// floating-point associativity in the matmul and the per-row
/// log-softmax / KL reductions.
///
/// # Shape contract (matches the (deleted, #1082) candle-typed
/// `opd_top_k_reverse_kl` dispatch entry; the candle-typed scalar-mean
/// surface that wrapped this via `OpdLossCustomOp` was also deleted in
/// (#1082, 2026-05-28) along with the rest of `phase_b.rs`'s candle
/// surface — production callers run the kt-shim per-position entry
/// `kiln_train::opd_tape_shim::opd_top_k_reverse_kl_per_position_via_kt_forward_op`
/// + `mean_all` instead.)
///
/// - `hidden`: `[1, T, H]` student hidden states.
/// - `head_t`: `[H, V]` transposed LM head (matches kiln's
///   `embed_tokens_t` layout).
/// - `teacher_topk_indices`: `[T_active * K]` row-major flat — the
///   teacher's top-K vocab indices at each active position.
/// - `teacher_topk_logprobs`: `[T_active * K]` row-major flat —
///   matching log-probabilities at those indices (log_softmax over
///   the full teacher vocab).
/// - `label_mask`: `[T]` booleans; the position-`t` logit contributes
///   when `label_mask[t]` is true. The number of active positions
///   must equal `T_active` and the order of active positions
///   left-to-right in `hidden` must match the row order of
///   `teacher_topk_indices`.
/// - `top_k`: K — the teacher's support size.
///
/// # Returns
///
/// A scalar F32 [`KtTensor`] (rank-0 / shape `[]`) holding the mean
/// reverse KL over active positions. Returns a scalar 0.0 tensor if
/// no positions are active.
///
/// # Backward
///
/// Backward is not yet implemented in the kt-typed scalar-mean path
/// — the per-position kt-typed backward
/// [`opd_top_k_reverse_kl_phase_b_bwd_kt`] is wired, but a scalar-
/// mean wrapper around it would need a `mean_all` autograd recorder
/// that we haven't yet implemented in `kiln_autograd`. Production
/// callers go through the candle kt-shim
/// `kiln_train::opd_tape_shim::opd_top_k_reverse_kl_per_position_via_kt_forward_op`
/// (per-position) + `mean_all` instead. (Historical: pre-`#1082`
/// 2026-05-28 the candle `CustomOp1` `OpdLossCustomOp` owned this
/// entry's backward via `apply_op1`; that wrapper was deleted with
/// the rest of `phase_b.rs`'s candle surface.)
pub fn opd_top_k_reverse_kl_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<KtTensor, OpdLossError> {
    let (_, _, _, active_count, _) = validate_inputs_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )?;
    if active_count == 0 {
        return zero_scalar();
    }

    let active_positions: Vec<u32> = label_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();

    let (per_token, _log_p_hat, _log_q_hat) = per_position_forward_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        &active_positions,
        top_k,
    )?;
    let loss = mean_all(&per_token)?;
    Ok(loss)
}

/// kt-typed entry point for OPD per-position reverse-KL.
///
/// Same forward kernel as [`opd_top_k_reverse_kl_kt`] but without
/// the final [`mean_all`] reduction. Used by the GRPO importance-
/// sampling advantage construction (`A_t = -KL_t`, §3.1 step 4 of
/// the OPD grand plan).
///
/// # Returns
///
/// A 1-D F32 [`KtTensor`] of shape `[T_active]` holding the per-
/// position reverse KL. Returns an empty `[0]` F32 tensor if no
/// positions are active.
///
/// # Backward
///
/// See [`opd_top_k_reverse_kl_kt`] — forward-only today.
pub fn opd_top_k_reverse_kl_per_position_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<KtTensor, OpdLossError> {
    let (_, _, _, active_count, _) = validate_inputs_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )?;
    if active_count == 0 {
        return empty_per_position();
    }

    let active_positions: Vec<u32> = label_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();

    let (per_token, _log_p_hat, _log_q_hat) = per_position_forward_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        &active_positions,
        top_k,
    )?;
    Ok(per_token)
}

/// kt-typed parallel of [`crate::PerPositionMetrics`].
///
/// Three parallel `[T_active]` arrays carrying the per-position
/// distribution-alignment diagnostics (§3.8 of the OPD grand plan).
/// Lengths are all equal to the number of active positions.
///
/// Mirrors the candle-typed [`crate::PerPositionMetrics`] struct so
/// the Phase B kt-rewrite has a stable target type to populate.
#[derive(Debug, Clone, Default)]
pub struct PerPositionMetricsKt {
    /// Per-position student entropy over the teacher's K support, in
    /// nats. Higher = student less concentrated.
    pub student_entropy: Vec<f32>,
    /// Per-position teacher entropy over the same K support, in nats.
    pub teacher_entropy: Vec<f32>,
    /// Per-position reverse KL (same value the loss kernel emits).
    pub reverse_kl: Vec<f32>,
}

impl PerPositionMetricsKt {
    /// `[T_active]` of `|H(q) - H(p)|` per position.
    pub fn entropy_gap_vec(&self) -> Vec<f32> {
        self.student_entropy
            .iter()
            .zip(self.teacher_entropy.iter())
            .map(|(p, q)| (q - p).abs())
            .collect()
    }

    /// Mean over active positions of `|H(q) - H(p)|`. The scalar §3.8
    /// diagnostic.
    pub fn mean_entropy_gap(&self) -> f64 {
        if self.student_entropy.is_empty() {
            return 0.0;
        }
        let n = self.student_entropy.len() as f64;
        self.student_entropy
            .iter()
            .zip(self.teacher_entropy.iter())
            .map(|(p, q)| (q - p).abs() as f64)
            .sum::<f64>()
            / n
    }

    /// Mean per-position KL — matches what the trainer already tracks,
    /// but recomputed here so the metrics call doesn't depend on a
    /// separate loss pass.
    pub fn mean_reverse_kl(&self) -> f64 {
        if self.reverse_kl.is_empty() {
            return 0.0;
        }
        let n = self.reverse_kl.len() as f64;
        self.reverse_kl.iter().map(|&v| v as f64).sum::<f64>() / n
    }
}

/// Read a 1-D F32 [`KtTensor`] into a `Vec<f32>`. Used to populate
/// the [`PerPositionMetricsKt`] struct from the kt-typed forward
/// kernel outputs.
fn read_f32_vec(t: &KtTensor) -> Result<Vec<f32>, OpdLossError> {
    use kiln_tensor::CpuStorage;
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| {
            OpdLossError::msg(
                "kt-opd-loss: metrics path requires CpuStorage on the kt-typed forward outputs",
            )
        })?;
    Ok(cpu
        .as_bytes()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// kt-typed entry point for per-position distribution-alignment
/// metrics over the teacher's K support.
///
/// Reuses the same forward kernel as [`opd_top_k_reverse_kl_kt`] and
/// additionally computes the per-position entropies of both the
/// renormalised student and teacher distributions over the K
/// support.
///
///   H(p_hat)[t] = -sum_k p_hat[t, k] * log_p_hat[t, k]
///   H(q_hat)[t] = -sum_k q_hat[t, k] * log_q_hat[t, k]
///
/// where `p_hat`, `q_hat` are the student / teacher distributions
/// renormalised over the K support (see the candle reference
/// [`crate::compute_per_position_metrics`] for the definition).
///
/// # Shape contract (matches the candle-typed
/// [`crate::compute_per_position_metrics`] entry point)
///
/// - `hidden`: `[1, T, H]` student hidden states.
/// - `head_t`: `[H, V]` transposed LM head.
/// - `teacher_topk_indices`: `[T_active * K]` row-major flat.
/// - `teacher_topk_logprobs`: `[T_active * K]` row-major flat.
/// - `label_mask`: `[T]` booleans selecting active positions.
/// - `top_k`: K — the teacher's support size.
///
/// # Returns
///
/// A populated [`PerPositionMetricsKt`] with three `[T_active]` f32
/// vectors — student entropy, teacher entropy, reverse KL. Returns
/// the default (empty) value if no positions are active.
pub fn compute_per_position_metrics_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<PerPositionMetricsKt, OpdLossError> {
    let (_, _, _, active_count, _) = validate_inputs_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )?;
    if active_count == 0 {
        return Ok(PerPositionMetricsKt::default());
    }

    let active_positions: Vec<u32> = label_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();

    let (per_token, log_p_hat, log_q_hat) = per_position_forward_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        &active_positions,
        top_k,
    )?;

    // Entropies: H(p) = -sum_k p * log p, H(q) = -sum_k q * log q.
    //
    // Both log_p_hat / log_q_hat are `[T_active, K]` F32 outputs of
    // log_softmax_last_dim above.
    let p_hat = exp(&log_p_hat)?;
    let q_hat = exp(&log_q_hat)?;
    let p_log_p = mul(&p_hat, &log_p_hat)?;
    let q_log_q = mul(&q_hat, &log_q_hat)?;
    let student_entropy_t = sum_axis(&p_log_p, 1)?; // sum_k p log p — negate below
    let teacher_entropy_t = sum_axis(&q_log_q, 1)?;
    // H(p) = -sum_k p log p = neg(sum_k p log p).
    let student_entropy = neg(&student_entropy_t)?;
    let teacher_entropy = neg(&teacher_entropy_t)?;

    let reverse_kl = per_token;

    // Materialise to Vec<f32> for the metrics struct. The metrics
    // path is a diagnostic / logging path, not part of the loss
    // graph, so reading the tensors back to host is fine.
    //
    // Force contiguous before the host read; the upstream
    // sum_axis / neg / sum_axis chain is already 1-D and contiguous
    // on the CPU path, but call `.contiguous()` defensively in case
    // a backend returns a strided view.
    let student_entropy = student_entropy.contiguous()?;
    let teacher_entropy = teacher_entropy.contiguous()?;
    let reverse_kl = reverse_kl.contiguous()?;
    Ok(PerPositionMetricsKt {
        student_entropy: read_f32_vec(&student_entropy)?,
        teacher_entropy: read_f32_vec(&teacher_entropy)?,
        reverse_kl: read_f32_vec(&reverse_kl)?,
    })
}

/// kt-typed mirror of the (deleted, #1082) candle-typed
/// `phase_b::OpdLossOutput` enum.
///
/// Selects whether the backward consumes a scalar `grad_loss`
/// (ScalarMean — the standard loss-scalar autograd contract) or a
/// per-position `grad_loss` of shape `[T_active]` (PerPosition — used
/// when the trainer hands out a per-token upstream gradient).
///
/// Kept public so call sites on the kt substrate don't have to drag
/// in the candle-typed `phase_b::OpdLossOutput`. The two enums are
/// 1:1 isomorphic; convert between them with a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpdLossOutputKt {
    /// Mean over active positions; `grad_loss` is a 0-D or 1-element
    /// scalar tensor. The kernel multiplies by `grad_loss / T_active`.
    ScalarMean,
    /// Per-position vector of shape `[T_active]`; the kernel multiplies
    /// position-wise.
    PerPosition,
}

/// Device-agnostic kt-composite backward for the OPD top-K reverse-KL
/// loss — the analytic gradient `d_hidden` derived directly from the
/// CUDA kernel's math (`csrc/opd_topk_kl.cu`), expressed purely in
/// [`kiln_tensor`] ops so it runs on CPU / Metal / CUDA with no FFI,
/// no cudarc, and no candle.
///
/// This is the OPD analogue of SFT's pure-kt analytic CE backward
/// (`kiln_model::tape_forward::CrossEntropyFromLogitsKtBackward`): it
/// re-derives the forward state (`s_logits`, `log_p_hat`, `log_q_hat`,
/// `KL_t`) from `(hidden, head_t, teacher_topk_*)` and emits the exact
/// analytic gradient.
///
/// # Gradient (matches `opd_topk_kl_bwd_kernel` exactly)
///
/// The forward is `KL_t = Σ_k p̂[t,k] (log p̂[t,k] − log q̂[t,k])`
/// where `p̂ = softmax(s_logits)` over the K support and
/// `s_logits[t,k] = Σ_h hidden[t,h] · head_t[h, idx[t,k]]`. The
/// reverse-KL gradient w.r.t. the student logits is
///
///   `dKL_t/ds_logits[t,k] = p̂[t,k] · (log p̂[t,k] − log q̂[t,k] − KL_t)`
///
/// scaled by the per-position upstream gradient (`grad_loss/T_active`
/// for `ScalarMean`, `grad_loss[t]` for `PerPosition`). The hidden
/// gradient is the matmul transpose
///
///   `d_hidden[t,h] = Σ_k d_s_logits[t,k] · head_t[h, idx[t,k]]`
///
/// which is the batched matmul `head_gather[t] @ d_s_logits[t]` —
/// `head_gather` being the same `[T_active, H, K]` gather of `head_t`
/// columns the forward builds. Non-active positions get a zero row via
/// `scatter_add` into a `[T, H]` zero buffer along axis 0.
///
/// # Shape contract / returns
///
/// Identical to [`opd_top_k_reverse_kl_phase_b_bwd_kt`]:
/// `d_hidden` of shape `[1, T, H]` in the same dtype as `hidden`.
///
/// # Correctness gate
///
/// Validated against a central finite-difference of the forward loss
/// in the `composite_bwd_finite_difference` unit test below (CPU,
/// deterministic).
pub fn opd_top_k_reverse_kl_phase_b_bwd_composite_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    grad_loss: &KtTensor,
    top_k: usize,
    output_mode: OpdLossOutputKt,
) -> Result<KtTensor, OpdLossError> {
    use kiln_tensor::ops::{broadcast_to, scatter_add, sum_axis};
    use kiln_tensor::Device as KtDevice;

    // -- 1. Validate shapes (same envelope as the FFI backward). --------
    validate_inputs_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )?;

    let dtype = hidden.dtype();
    let seq_len = hidden.shape()[1];
    let hidden_size = hidden.shape()[2];
    let dev = hidden.device();

    let active_positions: Vec<u32> = label_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();
    let active_count = active_positions.len();

    // Same device-aware host-tensor upload helper the forward uses.
    let upload_u32 = |vals: &[u32], shape: Vec<usize>| -> Result<KtTensor, OpdLossError> {
        match dev {
            KtDevice::Cpu => KtTensor::from_vec(vals.to_vec(), shape).map_err(OpdLossError::Kt),
            #[cfg(feature = "cuda")]
            KtDevice::Cuda(i) => {
                KtTensor::cuda_from_slice(vals, shape, i).map_err(OpdLossError::Kt)
            }
            #[cfg(feature = "metal")]
            KtDevice::Metal(_) => KtTensor::from_vec(vals.to_vec(), shape)
                .and_then(|t| t.to_device(dev))
                .map_err(OpdLossError::Kt),
            // (#1082) Vulkan mirrors Metal (host CPU build -> to_device).
            #[cfg(feature = "vulkan")]
            KtDevice::Vulkan(_) => KtTensor::from_vec(vals.to_vec(), shape)
                .and_then(|t| t.to_device(dev))
                .map_err(OpdLossError::Kt),
            // (Phase R.7) ROCm host upload (-> `host_to_rocm_copy`).
            #[cfg(feature = "rocm")]
            KtDevice::Rocm(i) => {
                KtTensor::from_vec_on(KtDevice::Rocm(i), vals.to_vec(), shape)
                    .map_err(OpdLossError::Kt)
            }
            #[allow(unreachable_patterns)]
            other => Err(OpdLossError::msg(format!(
                "kt-opd-loss bwd composite: unsupported device for index tensor {other}"
            ))),
        }
    };
    let upload_f32 = |vals: &[f32], shape: Vec<usize>| -> Result<KtTensor, OpdLossError> {
        match dev {
            KtDevice::Cpu => KtTensor::from_vec(vals.to_vec(), shape).map_err(OpdLossError::Kt),
            #[cfg(feature = "cuda")]
            KtDevice::Cuda(i) => {
                KtTensor::cuda_from_slice(vals, shape, i).map_err(OpdLossError::Kt)
            }
            #[cfg(feature = "metal")]
            KtDevice::Metal(_) => KtTensor::from_vec(vals.to_vec(), shape)
                .and_then(|t| t.to_device(dev))
                .map_err(OpdLossError::Kt),
            // (#1082) Vulkan mirrors Metal (host CPU build -> to_device).
            #[cfg(feature = "vulkan")]
            KtDevice::Vulkan(_) => KtTensor::from_vec(vals.to_vec(), shape)
                .and_then(|t| t.to_device(dev))
                .map_err(OpdLossError::Kt),
            // (Phase R.7) ROCm host upload (-> `host_to_rocm_copy`).
            #[cfg(feature = "rocm")]
            KtDevice::Rocm(i) => {
                KtTensor::from_vec_on(KtDevice::Rocm(i), vals.to_vec(), shape)
                    .map_err(OpdLossError::Kt)
            }
            #[allow(unreachable_patterns)]
            other => Err(OpdLossError::msg(format!(
                "kt-opd-loss bwd composite: unsupported device for f32 tensor {other}"
            ))),
        }
    };

    // Short-circuit: no active rows ⇒ d_hidden is all zeros in the
    // input dtype on the input device.
    if active_count == 0 {
        let zeros = KtTensor::from_vec(vec![0.0f32; seq_len * hidden_size], vec![1, seq_len, hidden_size])
            .map_err(OpdLossError::Kt)?;
        let zeros = zeros.to_device(dev).map_err(OpdLossError::Kt)?;
        let zeros = if dtype == KtDType::F32 {
            zeros
        } else {
            kiln_tensor::ops::cast(&zeros, dtype).map_err(OpdLossError::Kt)?
        };
        return Ok(zeros);
    }

    // -- 2. Per-position upstream gradient `upstream[t]` ----------------
    //
    // ScalarMean : upstream[t] = grad_loss[0] / T_active
    // PerPosition: upstream[t] = grad_loss[t]
    //
    // Read grad_loss to the host (it's a tiny [1] or [T_active] F32
    // tensor) so we can fold the scale uniformly; the result is uploaded
    // back to `dev` as a `[T_active, 1]` column for broadcasting.
    let grad_f32 = to_f32(grad_loss)?.contiguous()?;
    let grad_host = {
        use kiln_tensor::CpuStorage;
        let on_cpu = grad_f32.to_device(KtDevice::Cpu).map_err(OpdLossError::Kt)?;
        let cpu = on_cpu
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| {
                OpdLossError::msg("kt-opd-loss bwd composite: grad_loss host read failed")
            })?;
        cpu.as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<f32>>()
    };
    let upstream: Vec<f32> = match output_mode {
        OpdLossOutputKt::ScalarMean => {
            if grad_host.len() != 1 {
                return Err(OpdLossError::msg(format!(
                    "kt-opd-loss bwd composite: ScalarMean grad_loss must have 1 element, got {}",
                    grad_host.len(),
                )));
            }
            let s = grad_host[0] / (active_count as f32);
            vec![s; active_count]
        }
        OpdLossOutputKt::PerPosition => {
            if grad_host.len() != active_count {
                return Err(OpdLossError::msg(format!(
                    "kt-opd-loss bwd composite: PerPosition grad_loss must have {active_count} elements, got {}",
                    grad_host.len(),
                )));
            }
            grad_host
        }
    };
    // [T_active, 1] column for broadcasting against the [T_active, K]
    // gradient.
    let upstream_col = upload_f32(&upstream, vec![active_count, 1])?;

    // -- 3. Re-run the forward to recover (s_logits, log_p_hat, ---------
    //       log_q_hat, p_hat, KL_t) and the gather `head_gather`.
    //
    //       This mirrors `per_position_forward_kt` exactly; we keep the
    //       intermediate `head_gather` so the d_hidden matmul-transpose
    //       can reuse it.
    let hidden_2d = hidden.squeeze(0)?;
    let active_idx = upload_u32(&active_positions, vec![active_count])?;
    let active_hidden = index_select(&hidden_2d, 0, &active_idx)?;
    let active_hidden_f32 = to_f32(&active_hidden)?;
    let head_t_f32 = to_f32(head_t)?;

    let flat_indices = upload_u32(teacher_topk_indices, vec![active_count * top_k])?;
    let gathered = index_select(&head_t_f32, 1, &flat_indices)?;
    let reshaped = gathered.reshape(vec![hidden_size, active_count, top_k])?;
    let head_gather = reshaped.permute(&[1, 0, 2])?.contiguous()?; // [T_active, H, K]

    let lhs = active_hidden_f32.unsqueeze(1)?; // [T_active, 1, H]
    let s_logits = matmul(&lhs, &head_gather)?.squeeze(1)?.contiguous()?; // [T_active, K]

    let q_logprobs = upload_f32(teacher_topk_logprobs, vec![active_count, top_k])?;
    let log_p_hat = log_softmax_last_dim(&s_logits)?; // [T_active, K]
    let log_q_hat = log_softmax_last_dim(&q_logprobs)?; // [T_active, K]
    let p_hat = exp(&log_p_hat)?; // [T_active, K]
    let diff = sub(&log_p_hat, &log_q_hat)?; // log p̂ − log q̂

    // KL_t = Σ_k p̂ · diff  →  [T_active], broadcast to [T_active, K].
    let kl_t = sum_axis(&mul(&p_hat, &diff)?, 1)?; // [T_active]
    let kl_col = kl_t.reshape(vec![active_count, 1])?; // [T_active, 1]
    let kl_b = broadcast_to(&kl_col, &[active_count, top_k])?; // [T_active, K]

    // -- 4. d_s_logits[t,k] = p̂ · (diff − KL_t) · upstream[t] ----------
    let inner = sub(&diff, &kl_b)?; // [T_active, K]
    let d_s_logits = mul(&p_hat, &inner)?; // [T_active, K]
    let upstream_b = broadcast_to(&upstream_col, &[active_count, top_k])?; // [T_active, K]
    let d_s_logits = mul(&d_s_logits, &upstream_b)?; // [T_active, K]

    // -- 5. d_hidden[t,h] = Σ_k d_s_logits[t,k] · head_gather[t,h,k] ----
    //       = head_gather[t] @ d_s_logits[t] as a batched matmul:
    //       [T_active, H, K] @ [T_active, K, 1] → [T_active, H, 1].
    let d_s_col = d_s_logits.unsqueeze(2)?; // [T_active, K, 1]
    let d_hidden_active = matmul(&head_gather, &d_s_col)?.squeeze(2)?.contiguous()?; // [T_active, H]

    // -- 6. Cast to hidden dtype + scatter active rows into [T, H]. -----
    let d_hidden_active = if dtype == KtDType::F32 {
        d_hidden_active
    } else {
        kiln_tensor::ops::cast(&d_hidden_active, dtype).map_err(OpdLossError::Kt)?
    };
    let d_hidden_2d = scatter_add(&d_hidden_active, 0, &active_idx, seq_len)?; // [T, H]
    let d_hidden_3d = d_hidden_2d.reshape(vec![1, seq_len, hidden_size])?;

    Ok(d_hidden_3d)
}

/// CUDA-only kt-typed backward for the fused OPD top-K reverse-KL
/// kernel.
///
/// Wraps the same FFI symbols
/// (`kiln_opd_topk_kl_bwd_{bf16,f32}`) the candle backward path uses
/// (`crate::phase_b::OpdLossCustomOp::cuda_kernel_backward`), so the
/// two are bit-exact by construction. Substrate-only: this entry
/// point does not migrate the `CustomOp1::bwd` body — a follow-up
/// task wires the bridge through `OpdLossCustomOp::bwd` once this
/// kt-typed substrate is in place.
///
/// # Shape contract
///
/// - `hidden`: `[1, T, H]` student hidden states. BF16 or F32.
/// - `head_t`: `[H, V]` transposed LM head, same dtype as hidden.
/// - `teacher_topk_indices`: `[T_active * K]` row-major flat — the
///   teacher's top-K vocab indices at each active position.
/// - `teacher_topk_logprobs`: `[T_active * K]` row-major flat —
///   matching log-probabilities at those indices (log_softmax over
///   the full teacher vocab).
/// - `label_mask`: `[T]` booleans selecting active positions.
/// - `grad_loss`: depends on `output_mode`:
///   - `ScalarMean`: 0-D or 1-element F32 tensor on CUDA.
///   - `PerPosition`: 1-D `[T_active]` F32 tensor on CUDA.
/// - `top_k`: K, must be in {16, 32} (milestone-5 fast-path set; see
///   `crate::phase_b::cuda_kernel_supports`).
/// - `output_mode`: scalar-mean vs per-position selector.
///
/// # Returns
///
/// `d_hidden` of shape `[1, T, H]` in the same dtype as `hidden`. The
/// active-row gradients are computed by the fused kernel directly into
/// a zero-filled full-sequence output; non-active positions stay zero.
///
/// # Errors
///
/// Returns [`OpdLossError::Msg`] when:
/// - `top_k` / `dtype` are outside `cuda_kernel_supports`.
/// - any input is non-contiguous / non-CUDA / wrong dtype.
/// - `grad_loss` shape disagrees with `output_mode`.
/// - the FFI kernel returns a non-zero status.
// (Phase R.7) The fused backward is now reachable on BOTH CUDA and ROCm:
// `build.rs` compiles `csrc/opd_topk_kl.cu` with hipcc for the `rocm` feature
// (emitting the same `kiln_opd_topk_kl_bwd_*` symbols), and the body routes
// through the backend-neutral kt-bridge seam (`device_input_ptr` /
// `device_output_ptr` / `alloc_device_tensor_like` / `device_stream_raw_of`),
// which dispatches to either the CUDA or ROCm helper by the tensor's backend.
// CUDA behavior is unchanged.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn opd_top_k_reverse_kl_phase_b_bwd_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    grad_loss: &KtTensor,
    top_k: usize,
    output_mode: OpdLossOutputKt,
) -> Result<KtTensor, OpdLossError> {
    use kiln_kt_bridge::{
        alloc_device_tensor_like, device_input_ptr, device_output_ptr, device_stream_raw_of,
    };
    use kiln_tensor::Backend;
    use kiln_tensor::Device as KtDevice;

    // -- 1. Validate shapes / dtype envelope ----------------------------
    let (_, _, _vocab_size_decl, _, _) = validate_inputs_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )?;

    // K + dtype gate (mirrors `phase_b::cuda_kernel_supports`).
    let dtype = hidden.dtype();
    if !matches!(dtype, KtDType::F32 | KtDType::BF16) {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss bwd: unsupported dtype {dtype}; only F32 / BF16 are wired",
        )));
    }
    if top_k != 16 && top_k != 32 {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss bwd: top_k must be in {{16, 32}}; got {top_k}",
        )));
    }
    if head_t.dtype() != dtype {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss bwd: head_t dtype {} != hidden dtype {dtype}",
            head_t.dtype(),
        )));
    }

    let seq_len = hidden.shape()[1];
    let hidden_size = hidden.shape()[2];
    let vocab_size = head_t.shape()[1];

    // -- 2. Resolve device + active rows --------------------------------
    //
    // (Phase R.7) Accept CUDA OR ROCm. `dev` carries the index so the
    // host-side teacher/scatter tensors land on the SAME device as `hidden`
    // via the backend-neutral `from_vec_on` constructor below.
    let dev = hidden.device();
    match dev.backend() {
        #[cfg(feature = "cuda")]
        Backend::Cuda => {}
        #[cfg(feature = "rocm")]
        Backend::Rocm => {}
        other => {
            return Err(OpdLossError::msg(format!(
                "kt-opd-loss bwd: hidden must be on a GPU (CUDA/ROCm), got backend {other:?}",
            )));
        }
    }

    // Backend-neutral host->device upload of a typed slice onto `dev`. CUDA
    // routes through `cuda_from_slice` (host->cuda); ROCm through
    // `from_vec_on(Device::Rocm(i), ...)` (-> `host_to_rocm_copy`). Both keep
    // the result on `hidden`'s device so the FFI kernel's pointers are valid.
    let upload_slice = |vals_u32: Option<&[u32]>,
                        vals_f32: Option<&[f32]>,
                        shape: Vec<usize>|
     -> Result<KtTensor, OpdLossError> {
        match (vals_u32, vals_f32, dev) {
            #[cfg(feature = "cuda")]
            (Some(u), None, KtDevice::Cuda(i)) => {
                KtTensor::cuda_from_slice(u, shape, i).map_err(OpdLossError::Kt)
            }
            #[cfg(feature = "cuda")]
            (None, Some(f), KtDevice::Cuda(i)) => {
                KtTensor::cuda_from_slice(f, shape, i).map_err(OpdLossError::Kt)
            }
            #[cfg(feature = "rocm")]
            (Some(u), None, KtDevice::Rocm(i)) => {
                KtTensor::from_vec_on(KtDevice::Rocm(i), u.to_vec(), shape)
                    .map_err(OpdLossError::Kt)
            }
            #[cfg(feature = "rocm")]
            (None, Some(f), KtDevice::Rocm(i)) => {
                KtTensor::from_vec_on(KtDevice::Rocm(i), f.to_vec(), shape)
                    .map_err(OpdLossError::Kt)
            }
            _ => Err(OpdLossError::msg(
                "kt-opd-loss bwd: unsupported device/dtype for host upload",
            )),
        }
    };

    let active_positions: Vec<u32> = label_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();
    let active_count = active_positions.len();

    // Short-circuit: no active rows ⇒ d_hidden is all zeros in the
    // input dtype on the input device. Skip the kernel entirely; this
    // matches `cuda_kernel_backward`'s early return.
    if active_count == 0 {
        let zeros = alloc_device_tensor_like(hidden, dtype, vec![1, seq_len, hidden_size])?;
        return Ok(zeros);
    }

    // -- 3. Upload host-side teacher tensors + row-map inputs ----------
    //
    // active_indices (U32, [T_active]) — both the gather of hidden rows
    // and the fused kernel's full-output row mapping use the same index
    // buffer.
    let active_indices = upload_slice(
        Some(active_positions.as_slice()),
        None,
        vec![active_count],
    )?;

    // Gather active hidden rows, then take contiguous.
    //
    // (Phase R.7) `index_select` has a native `cuda_fwd` but NO `rocm_fwd`;
    // on ROCm `dispatch2` therefore falls back to `cpu_fwd`, which reads its
    // index + data tensors via `downcast_ref::<CpuStorage>()` and so rejects
    // ROCm-backed storage ("indices storage must be CpuStorage"). The ROCm
    // fallback in `dispatch2` does NOT relocate operands to CPU first (unlike
    // Metal/Vulkan). So on ROCm we route the gather through the host: pull the
    // (tiny) active-row selection on CPU, then upload the result back to the
    // ROCm device for the FFI kernel. CUDA keeps the native device gather
    // (byte-identical to the pre-R.7 path).
    let hidden_2d = hidden.squeeze(0)?;
    let active_hidden = match dev.backend() {
        #[cfg(feature = "rocm")]
        Backend::Rocm => {
            let rocm_idx = match dev {
                KtDevice::Rocm(i) => i,
                _ => 0,
            };
            // hidden_2d is a ROCm view; copy to host, gather on CPU with a
            // CPU index tensor, then upload the [T_active, H] result back.
            let hidden_2d_host = kiln_tensor::rocm_to_host_copy(&hidden_2d)?;
            let active_idx_host =
                KtTensor::from_vec(active_positions.clone(), vec![active_count])?;
            let gathered_host =
                index_select(&hidden_2d_host, 0, &active_idx_host)?.contiguous()?;
            kiln_tensor::host_to_rocm_copy(&gathered_host, rocm_idx)?
        }
        #[allow(unreachable_patterns)]
        _ => index_select(&hidden_2d, 0, &active_indices)?.contiguous()?,
    };

    // Ensure head_t is contiguous; the kernel reads from start_offset 0.
    let head_t_contig = if head_t.is_contiguous() {
        head_t.clone()
    } else {
        head_t.contiguous()?
    };

    // Upload teacher tensors as 2-D for the FFI (`[T_active, K]`).
    let topk_idx_dev = upload_slice(
        Some(teacher_topk_indices),
        None,
        vec![active_count, top_k],
    )?;
    let topk_lp_q_dev = upload_slice(
        None,
        Some(teacher_topk_logprobs),
        vec![active_count, top_k],
    )?;

    // Normalise grad_loss to a 1-D contiguous F32 tensor on device,
    // shape {ScalarMean: [1], PerPosition: [active_count]}.
    let grad_loss_dev: KtTensor;
    let (output_mode_i32, scale_factor) = match output_mode {
        OpdLossOutputKt::ScalarMean => {
            if grad_loss.dtype() != KtDType::F32 {
                return Err(OpdLossError::msg(format!(
                    "kt-opd-loss bwd: ScalarMean grad_loss must be F32, got {}",
                    grad_loss.dtype(),
                )));
            }
            let n: usize = grad_loss.shape().iter().product();
            if n != 1 {
                return Err(OpdLossError::msg(format!(
                    "kt-opd-loss bwd: ScalarMean grad_loss must have 1 element, got shape {:?}",
                    grad_loss.shape(),
                )));
            }
            grad_loss_dev = grad_loss.reshape(vec![1])?.contiguous()?;
            (0_i32, 1.0_f32 / (active_count as f32))
        }
        OpdLossOutputKt::PerPosition => {
            if grad_loss.dtype() != KtDType::F32 {
                return Err(OpdLossError::msg(format!(
                    "kt-opd-loss bwd: PerPosition grad_loss must be F32, got {}",
                    grad_loss.dtype(),
                )));
            }
            let s = grad_loss.shape();
            if s.len() != 1 || s[0] != active_count {
                return Err(OpdLossError::msg(format!(
                    "kt-opd-loss bwd: PerPosition grad_loss must have shape [{active_count}], got {s:?}",
                )));
            }
            grad_loss_dev = if grad_loss.is_contiguous() {
                grad_loss.clone()
            } else {
                grad_loss.contiguous()?
            };
            (1_i32, 1.0_f32)
        }
    };

    // -- 4. Allocate full output buffer [T, H] on device ----------------
    //
    // The fused kernel writes each active row directly at
    // active_indices[t]. Because the allocation is zero-filled, inactive
    // rows already have the correct gradient and no post-kernel scatter is
    // needed.
    let d_hidden_2d = alloc_device_tensor_like(hidden, dtype, vec![seq_len, hidden_size])?;

    // -- 5. Pull device pointers ----------------------------------------
    //
    // active_hidden / head_t_contig are typed in hidden's dtype
    // (F32 or BF16); the K indices are U32; logprobs and grad_loss
    // are F32.
    let h_ptr = device_input_ptr(&active_hidden, dtype, "active_hidden")?;
    let head_ptr = device_input_ptr(&head_t_contig, dtype, "head_t")?;
    let i_ptr = device_input_ptr(&topk_idx_dev, KtDType::U32, "topk_idx")?;
    let l_ptr =
        device_input_ptr(&topk_lp_q_dev, KtDType::F32, "topk_lp_q")?;
    let a_ptr = device_input_ptr(&active_indices, KtDType::U32, "active_indices")?;
    let g_ptr =
        device_input_ptr(&grad_loss_dev, KtDType::F32, "grad_loss")?;
    let d_ptr = device_output_ptr(&d_hidden_2d);
    let raw_stream = device_stream_raw_of(hidden, "stream")?;

    // -- 6. Dispatch the FFI --------------------------------------------
    //
    // Same kernel symbols the candle path uses; bit-exact by
    // construction. The output buffer is freshly zero-allocated via
    // `alloc_device_tensor_like`, so the kernel's writes land in a clean
    // F32/BF16 tile of the expected size.
    let status = unsafe {
        match dtype {
            KtDType::F32 => crate::phase_b::kiln_opd_topk_kl_bwd_f32(
                h_ptr as *const _,
                head_ptr as *const _,
                i_ptr as *const _,
                l_ptr as *const _,
                a_ptr as *const _,
                g_ptr as *const _,
                scale_factor,
                d_ptr as *mut _,
                active_count as i32,
                hidden_size as i32,
                vocab_size as i32,
                top_k as i32,
                output_mode_i32,
                raw_stream,
            ),
            KtDType::BF16 => crate::phase_b::kiln_opd_topk_kl_bwd_bf16(
                h_ptr as *const _,
                head_ptr as *const _,
                i_ptr as *const _,
                l_ptr as *const _,
                a_ptr as *const _,
                g_ptr as *const _,
                scale_factor,
                d_ptr as *mut _,
                active_count as i32,
                hidden_size as i32,
                vocab_size as i32,
                top_k as i32,
                output_mode_i32,
                raw_stream,
            ),
            other => {
                return Err(OpdLossError::msg(format!(
                    "kt-opd-loss bwd: unreachable dtype {other:?}",
                )));
            }
        }
    };
    if status != 0 {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss bwd: kiln_opd_topk_kl_bwd_* returned status {status}",
        )));
    }

    // -- 7. Unsqueeze full `[T, H]` gradient to `[1, T, H]`. ------------
    let d_hidden_3d = d_hidden_2d.reshape(vec![1, seq_len, hidden_size])?;
    Ok(d_hidden_3d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::{DType as KtDType, Tensor as KtTensorCtor};

    fn dummy_hidden(seq_len: usize, hidden_size: usize) -> KtTensor {
        let n = seq_len * hidden_size;
        let data = vec![0.0f32; n];
        KtTensorCtor::from_vec(data, vec![1, seq_len, hidden_size]).expect("alloc hidden")
    }

    fn dummy_head_t(hidden_size: usize, vocab_size: usize) -> KtTensor {
        let n = hidden_size * vocab_size;
        let data = vec![0.0f32; n];
        KtTensorCtor::from_vec(data, vec![hidden_size, vocab_size]).expect("alloc head")
    }

    #[test]
    fn opd_loss_error_displays_message() {
        let e = OpdLossError::msg("test message");
        assert_eq!(format!("{e}"), "test message");
        // std::error::Error impl is reachable.
        let _: &dyn std::error::Error = &e;
    }

    #[test]
    fn opd_loss_error_not_yet_implemented_displays_name() {
        let e = OpdLossError::NotYetImplemented("foo_kt");
        let s = format!("{e}");
        assert!(s.contains("foo_kt"), "got: {s}");
        assert!(s.contains("not yet implemented"), "got: {s}");
        // Avoid an unused-import warning on DType when no other test
        // references it.
        let _: KtDType = KtDType::F32;
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_hidden_rank() {
        // 2-D hidden — must be 3-D.
        let h = KtTensor::from_vec(vec![0.0f32; 16], vec![4, 4]).expect("alloc h");
        let w = dummy_head_t(4, 8);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("hidden must be 3-D"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_hidden_batch_dim() {
        // hidden batch dim must be 1.
        let h = KtTensor::from_vec(vec![0.0f32; 32], vec![2, 4, 4]).expect("alloc h");
        let w = dummy_head_t(4, 8);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("batch dim must be 1"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_head_rank() {
        let h = dummy_hidden(4, 4);
        // 3-D head — must be 2-D.
        let w = KtTensor::from_vec(vec![0.0f32; 32], vec![4, 2, 4]).expect("alloc w");
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("head_t must be 2-D"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_hidden_vs_head_hidden_size() {
        // hidden_size=8 but head_t hidden_size=4 — mismatch.
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(4, 16);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("hidden_size mismatch"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_label_mask_length() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 3]; // wrong length
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("label_mask length"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_top_k_zero() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 0).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("top_k must be > 0"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_indices_length() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        // T_active = 4, K = 2, so expected length = 8; pass 6.
        let idx = vec![0u32; 6];
        let lp = vec![0.0f32; 8];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 2).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("teacher_topk_indices length"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_logprobs_length() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        // T_active = 4, K = 2, so expected length = 8; pass 6 for lp.
        let idx = vec![0u32; 8];
        let lp = vec![0.0f32; 6];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 2).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("teacher_topk_logprobs length"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_index_in_vocab() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        // vocab_size = 16, so index 99 is out of bounds.
        let idx = vec![99u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(
            s.contains(">= vocab_size") || s.contains("99"),
            "got: {s}"
        );
    }

    /// Read a F32 [`KtTensor`] into a Vec for assertions.
    fn read_f32(t: &KtTensor) -> Vec<f32> {
        use kiln_tensor::{CpuStorage, Device as KtDevice};
        let on_cpu = t.to_device(KtDevice::Cpu).expect("move tensor to CPU");
        let cpu = on_cpu
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .expect("CpuStorage");
        cpu.as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_returns_scalar_on_valid_shapes() {
        // K=1 with matching teacher logprob = 0 (i.e. teacher places
        // all mass on the single token). With zero hidden and zero
        // head, the student also has log_p_hat = 0 and KL = 0.
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let out = opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).expect("forward");
        assert_eq!(out.shape(), &[] as &[usize], "scalar output");
        let v = read_f32(&out);
        assert_eq!(v.len(), 1);
        assert!(v[0].abs() < 1e-6, "K=1 KL must be exactly 0, got {}", v[0]);
    }

    #[test]
    fn opd_top_k_reverse_kl_per_position_kt_returns_vector_on_valid_shapes() {
        // K=1 ⇒ both renormalised distributions are degenerate
        // singletons ⇒ KL = 0 at every active position.
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let out = opd_top_k_reverse_kl_per_position_kt(&h, &w, &idx, &lp, &mask, 1)
            .expect("forward");
        assert_eq!(out.shape(), &[4]);
        let v = read_f32(&out);
        assert_eq!(v.len(), 4);
        for (i, &x) in v.iter().enumerate() {
            assert!(x.abs() < 1e-6, "per_token[{i}] should be 0, got {x}");
        }
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_no_active_returns_zero_scalar() {
        // All-masked-out positions: T_active = 0, no teacher data,
        // result is a scalar 0.0.
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx: Vec<u32> = vec![];
        let lp: Vec<f32> = vec![];
        let mask = vec![false; 4];
        let out = opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 4).expect("forward");
        assert_eq!(out.shape(), &[] as &[usize], "scalar output");
        let v = read_f32(&out);
        assert_eq!(v, vec![0.0]);
    }

    #[test]
    fn opd_top_k_reverse_kl_per_position_kt_no_active_returns_empty_1d() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx: Vec<u32> = vec![];
        let lp: Vec<f32> = vec![];
        let mask = vec![false; 4];
        let out = opd_top_k_reverse_kl_per_position_kt(&h, &w, &idx, &lp, &mask, 4)
            .expect("forward");
        assert_eq!(out.shape(), &[0]);
        assert!(read_f32(&out).is_empty());
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_uniform_teacher_uniform_student_kl_zero() {
        // K = 4 with uniform teacher logprobs (-log 4) at every active
        // position, zero hidden + zero head ⇒ student is also uniform
        // over the K support ⇒ KL = 0.
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let top_k = 4usize;
        let active = 3usize;
        // active positions at indices 0, 2, 3
        let mask = vec![true, false, true, true];
        // T_active * K distinct indices into [0, 16)
        let mut idx: Vec<u32> = Vec::with_capacity(active * top_k);
        for r in 0..active {
            for k in 0..top_k {
                idx.push((r * top_k + k) as u32);
            }
        }
        let lp = vec![-(top_k as f32).ln(); active * top_k];
        let out = opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, top_k).expect("forward");
        let v = read_f32(&out);
        assert_eq!(v.len(), 1);
        assert!(v[0].abs() < 1e-5, "uniform KL must be 0, got {}", v[0]);
    }

    #[test]
    fn opd_top_k_reverse_kl_per_position_kt_parity_with_scalar_mean() {
        // Build a small but non-trivial forward and confirm that
        // mean(per_position) == scalar entry point output. This is
        // the simplest parity check that exercises the full
        // gather + matmul + log-softmax + KL path.
        //
        // Use random-ish (deterministic) hidden / head values to
        // avoid the degenerate uniform case.
        let t_seq = 5;
        let h_dim = 4;
        let v_dim = 8;
        let k = 3;
        let mask = vec![true, false, true, true, false];
        let active = mask.iter().filter(|&&m| m).count();

        let hidden_data: Vec<f32> = (0..t_seq * h_dim)
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();
        let hidden = KtTensor::from_vec(hidden_data, vec![1, t_seq, h_dim]).unwrap();

        let head_data: Vec<f32> = (0..h_dim * v_dim)
            .map(|i| ((i as f32) * 0.07).cos() * 0.5)
            .collect();
        let head = KtTensor::from_vec(head_data, vec![h_dim, v_dim]).unwrap();

        let mut idx: Vec<u32> = Vec::with_capacity(active * k);
        for r in 0..active {
            for kk in 0..k {
                idx.push(((r + kk) % v_dim) as u32);
            }
        }
        let lp: Vec<f32> = (0..active * k)
            .map(|i| -1.0 - ((i as f32) * 0.05).cos())
            .collect();

        let scalar = read_f32(
            &opd_top_k_reverse_kl_kt(&hidden, &head, &idx, &lp, &mask, k).unwrap(),
        );
        let per_pos = read_f32(
            &opd_top_k_reverse_kl_per_position_kt(&hidden, &head, &idx, &lp, &mask, k)
                .unwrap(),
        );
        assert_eq!(per_pos.len(), active);
        let mean: f32 = per_pos.iter().sum::<f32>() / (active as f32);
        assert!(
            (scalar[0] - mean).abs() < 1e-5,
            "scalar={} != mean(per_pos)={}",
            scalar[0],
            mean
        );
        // Sanity: KL is non-negative.
        for (i, &kl) in per_pos.iter().enumerate() {
            assert!(kl > -1e-6, "KL must be >= 0; per_pos[{i}] = {kl}");
        }
    }

    #[test]
    fn opd_top_k_reverse_kl_per_position_kt_validates_hidden_rank() {
        // Same hidden rank validation applies to per-position entry.
        let h = KtTensor::from_vec(vec![0.0f32; 16], vec![4, 4]).expect("alloc h");
        let w = dummy_head_t(4, 8);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err = opd_top_k_reverse_kl_per_position_kt(&h, &w, &idx, &lp, &mask, 1)
            .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("hidden must be 3-D"), "got: {s}");
    }

    #[test]
    fn compute_per_position_metrics_kt_validates_hidden_rank() {
        let h = KtTensor::from_vec(vec![0.0f32; 16], vec![4, 4]).expect("alloc h");
        let w = dummy_head_t(4, 8);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            compute_per_position_metrics_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("hidden must be 3-D"), "got: {s}");
    }

    #[test]
    fn compute_per_position_metrics_kt_returns_populated_struct_on_valid_shapes() {
        // K = 1, uniform-degenerate case: entropies are exactly 0
        // (single-token distribution) and reverse KL is 0.
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let m = compute_per_position_metrics_kt(&h, &w, &idx, &lp, &mask, 1)
            .expect("metrics");
        assert_eq!(m.student_entropy.len(), 4);
        assert_eq!(m.teacher_entropy.len(), 4);
        assert_eq!(m.reverse_kl.len(), 4);
        for v in m
            .student_entropy
            .iter()
            .chain(m.teacher_entropy.iter())
            .chain(m.reverse_kl.iter())
        {
            assert!(v.abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn compute_per_position_metrics_kt_uniform_entropy_log_k() {
        // K=4, zero hidden + zero head ⇒ student uniform over K ⇒
        // H(p_hat) = ln(K). Teacher logprobs uniform too ⇒
        // H(q_hat) = ln(K).
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let top_k = 4usize;
        let active = 3usize;
        let mask = vec![true, false, true, true];
        let mut idx: Vec<u32> = Vec::with_capacity(active * top_k);
        for r in 0..active {
            for k in 0..top_k {
                idx.push((r * top_k + k) as u32);
            }
        }
        let lp = vec![-(top_k as f32).ln(); active * top_k];
        let m = compute_per_position_metrics_kt(&h, &w, &idx, &lp, &mask, top_k)
            .expect("metrics");
        let expect_h = (top_k as f32).ln();
        for &s in &m.student_entropy {
            assert!((s - expect_h).abs() < 1e-5, "student H = {s}, want {expect_h}");
        }
        for &t in &m.teacher_entropy {
            assert!((t - expect_h).abs() < 1e-5, "teacher H = {t}, want {expect_h}");
        }
        for &kl in &m.reverse_kl {
            assert!(kl.abs() < 1e-5, "KL = {kl}, want 0");
        }
    }

    #[test]
    fn compute_per_position_metrics_kt_no_active_returns_default() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx: Vec<u32> = vec![];
        let lp: Vec<f32> = vec![];
        let mask = vec![false; 4];
        let m = compute_per_position_metrics_kt(&h, &w, &idx, &lp, &mask, 4)
            .expect("metrics");
        assert!(m.student_entropy.is_empty());
        assert!(m.teacher_entropy.is_empty());
        assert!(m.reverse_kl.is_empty());
    }

    #[test]
    fn per_position_metrics_kt_entropy_gap_vec_smoke() {
        let m = PerPositionMetricsKt {
            student_entropy: vec![1.0, 0.5, 2.0],
            teacher_entropy: vec![1.5, 0.5, 1.0],
            reverse_kl: vec![0.1, 0.0, 0.5],
        };
        let gap = m.entropy_gap_vec();
        assert_eq!(gap, vec![0.5, 0.0, 1.0]);
    }

    #[test]
    fn per_position_metrics_kt_mean_helpers() {
        let m = PerPositionMetricsKt {
            student_entropy: vec![1.0, 0.5, 2.0],
            teacher_entropy: vec![1.5, 0.5, 1.0],
            reverse_kl: vec![0.1, 0.0, 0.5],
        };
        let mean_gap = m.mean_entropy_gap();
        assert!((mean_gap - 0.5).abs() < 1e-6, "got: {mean_gap}");
        let mean_kl = m.mean_reverse_kl();
        assert!((mean_kl - 0.2).abs() < 1e-6, "got: {mean_kl}");
    }

    #[test]
    fn per_position_metrics_kt_empty_means_are_zero() {
        let m = PerPositionMetricsKt::default();
        assert_eq!(m.mean_entropy_gap(), 0.0);
        assert_eq!(m.mean_reverse_kl(), 0.0);
        assert!(m.entropy_gap_vec().is_empty());
    }

    // -----------------------------------------------------------------------
    // Device-agnostic composite backward — MANDATORY correctness gate.
    //
    // Compares the analytic composite gradient
    // (`opd_top_k_reverse_kl_phase_b_bwd_composite_kt`) against a CENTRAL
    // finite-difference of the forward loss, on CPU (deterministic, no GPU).
    // -----------------------------------------------------------------------

    /// Scalar forward loss for the FD check, computed from a flat
    /// `[1, T, H]` hidden buffer. `grad_w` are the per-active-position
    /// weights folded into the loss:
    ///   ScalarMean  : L = (1/T_active) Σ_t KL_t     (grad_w all = 1)
    ///   PerPosition : L = Σ_t grad_w[t] · KL_t
    fn forward_loss_for_fd(
        hidden_flat: &[f32],
        seq_len: usize,
        hidden_size: usize,
        head: &KtTensor,
        idx: &[u32],
        lp: &[f32],
        mask: &[bool],
        top_k: usize,
        grad_w: &[f32],
        scalar_mean: bool,
    ) -> f64 {
        let hidden =
            KtTensor::from_vec(hidden_flat.to_vec(), vec![1, seq_len, hidden_size]).unwrap();
        let active: Vec<u32> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
            .collect();
        let (per_token, _, _) =
            per_position_forward_kt(&hidden, head, idx, lp, &active, top_k).unwrap();
        let kl = read_f32(&per_token);
        if scalar_mean {
            let n = kl.len() as f64;
            kl.iter().map(|&v| v as f64).sum::<f64>() / n
        } else {
            kl.iter()
                .zip(grad_w.iter())
                .map(|(&k, &w)| (k as f64) * (w as f64))
                .sum::<f64>()
        }
    }

    fn run_fd_check(scalar_mean: bool) {
        // Small deterministic fixture. T=5 (3 active), H=6, V=10, K=3.
        let seq_len = 5usize;
        let hidden_size = 6usize;
        let vocab = 10usize;
        let top_k = 3usize;
        let mask = vec![true, false, true, true, false];
        let active_count = mask.iter().filter(|&&m| m).count();

        // Deterministic non-degenerate hidden / head.
        let hidden_flat: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| ((i as f32) * 0.37).sin() * 0.8 + 0.1)
            .collect();
        let head_flat: Vec<f32> = (0..hidden_size * vocab)
            .map(|i| ((i as f32) * 0.21).cos() * 0.6)
            .collect();
        let head = KtTensor::from_vec(head_flat, vec![hidden_size, vocab]).unwrap();

        // Distinct teacher top-K indices per active row + non-uniform
        // teacher logprobs so q̂ ≠ uniform (exercises the log_q term).
        let mut idx: Vec<u32> = Vec::with_capacity(active_count * top_k);
        for r in 0..active_count {
            for k in 0..top_k {
                idx.push(((r * 2 + k) % vocab) as u32);
            }
        }
        let lp: Vec<f32> = (0..active_count * top_k)
            .map(|i| -0.5 - ((i as f32) * 0.17).cos().abs())
            .collect();

        // Upstream weights. ScalarMean: grad_loss scalar 1.0 (so the
        // composite folds 1/T_active internally and L = mean KL).
        // PerPosition: a non-trivial per-position grad vector.
        let grad_w: Vec<f32> = if scalar_mean {
            vec![1.0; active_count]
        } else {
            (0..active_count).map(|t| 0.3 + 0.5 * (t as f32)).collect()
        };
        let (grad_loss, mode) = if scalar_mean {
            (
                KtTensor::from_vec(vec![1.0f32], vec![1]).unwrap(),
                OpdLossOutputKt::ScalarMean,
            )
        } else {
            (
                KtTensor::from_vec(grad_w.clone(), vec![active_count]).unwrap(),
                OpdLossOutputKt::PerPosition,
            )
        };

        let hidden = KtTensor::from_vec(hidden_flat.clone(), vec![1, seq_len, hidden_size]).unwrap();
        let analytic = opd_top_k_reverse_kl_phase_b_bwd_composite_kt(
            &hidden, &head, &idx, &lp, &mask, &grad_loss, top_k, mode,
        )
        .expect("composite backward");
        assert_eq!(analytic.shape(), &[1, seq_len, hidden_size]);
        let g_analytic = read_f32(&analytic);

        // Central finite difference: ∂L/∂hidden[i] ≈ (L(x+h)-L(x-h))/2h.
        let h = 1e-3f32;
        let mut max_abs_err = 0.0f64;
        let mut max_ref = 0.0f64;
        for i in 0..seq_len * hidden_size {
            let mut xp = hidden_flat.clone();
            let mut xm = hidden_flat.clone();
            xp[i] += h;
            xm[i] -= h;
            let lp_loss = forward_loss_for_fd(
                &xp, seq_len, hidden_size, &head, &idx, &lp, &mask, top_k, &grad_w, scalar_mean,
            );
            let lm_loss = forward_loss_for_fd(
                &xm, seq_len, hidden_size, &head, &idx, &lp, &mask, top_k, &grad_w, scalar_mean,
            );
            let fd = (lp_loss - lm_loss) / (2.0 * h as f64);
            let an = g_analytic[i] as f64;
            let err = (fd - an).abs();
            max_abs_err = max_abs_err.max(err);
            max_ref = max_ref.max(fd.abs()).max(an.abs());
        }
        // Relative tolerance: max abs err < 1e-2 * max magnitude (with a
        // small absolute floor for near-zero gradients).
        let tol = 1e-2 * max_ref.max(1e-3);
        assert!(
            max_abs_err < tol,
            "composite backward FD mismatch (scalar_mean={scalar_mean}): \
             max_abs_err={max_abs_err:.3e} tol={tol:.3e} max_ref={max_ref:.3e}"
        );
        eprintln!(
            "[OPD-FD] scalar_mean={scalar_mean}: max_abs_err={max_abs_err:.3e} \
             tol={tol:.3e} max_ref={max_ref:.3e}"
        );
    }

    #[test]
    fn composite_bwd_finite_difference_scalar_mean() {
        run_fd_check(true);
    }

    #[test]
    fn composite_bwd_finite_difference_per_position() {
        run_fd_check(false);
    }

    /// No-active-positions short-circuit must return an all-zero
    /// `[1, T, H]` grad of the right dtype.
    #[test]
    fn composite_bwd_no_active_returns_zeros() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx: Vec<u32> = vec![];
        let lp: Vec<f32> = vec![];
        let mask = vec![false; 4];
        let grad = KtTensor::from_vec(vec![1.0f32], vec![1]).unwrap();
        let out = opd_top_k_reverse_kl_phase_b_bwd_composite_kt(
            &h, &w, &idx, &lp, &mask, &grad, 4, OpdLossOutputKt::ScalarMean,
        )
        .expect("composite backward no-active");
        assert_eq!(out.shape(), &[1, 4, 8]);
        for v in read_f32(&out) {
            assert_eq!(v, 0.0);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_bwd_matches_composite_with_sparse_active_rows() {
        if kiln_tensor::primary_cuda_context(0).is_err() {
            eprintln!("CUDA not available; skipping cuda_bwd_matches_composite");
            return;
        }

        let seq_len = 6usize;
        let hidden_size = 128usize;
        let vocab = 256usize;
        let top_k = 16usize;
        let mask = vec![false, true, false, true, true, false];
        let active_count = mask.iter().filter(|&&m| m).count();

        let hidden_flat: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| ((i as f32) * 0.13).sin() * 0.25 + (((i % 7) as f32) - 3.0) * 0.01)
            .collect();
        let head_flat: Vec<f32> = (0..hidden_size * vocab)
            .map(|i| ((i as f32) * 0.07).cos() * 0.18 + (((i % 11) as f32) - 5.0) * 0.005)
            .collect();

        let mut idx: Vec<u32> = Vec::with_capacity(active_count * top_k);
        for r in 0..active_count {
            let base = r * 37 + 5;
            for k in 0..top_k {
                idx.push(((base + k * 11) % vocab) as u32);
            }
        }
        let lp: Vec<f32> = (0..active_count * top_k)
            .map(|i| -0.2 - ((i as f32) * 0.31).cos().abs())
            .collect();
        let grad_host = vec![0.25f32, -0.5, 0.75];

        let hidden_cpu =
            KtTensor::from_vec(hidden_flat.clone(), vec![1, seq_len, hidden_size]).unwrap();
        let head_cpu = KtTensor::from_vec(head_flat.clone(), vec![hidden_size, vocab]).unwrap();
        let grad_cpu = KtTensor::from_vec(grad_host.clone(), vec![active_count]).unwrap();
        let expected = opd_top_k_reverse_kl_phase_b_bwd_composite_kt(
            &hidden_cpu,
            &head_cpu,
            &idx,
            &lp,
            &mask,
            &grad_cpu,
            top_k,
            OpdLossOutputKt::PerPosition,
        )
        .expect("composite backward");

        let hidden_cuda =
            KtTensor::cuda_from_slice(&hidden_flat, vec![1, seq_len, hidden_size], 0)
                .expect("hidden cuda");
        let head_cuda =
            KtTensor::cuda_from_slice(&head_flat, vec![hidden_size, vocab], 0)
                .expect("head cuda");
        let grad_cuda =
            KtTensor::cuda_from_slice(&grad_host, vec![active_count], 0).expect("grad cuda");
        let got = opd_top_k_reverse_kl_phase_b_bwd_kt(
            &hidden_cuda,
            &head_cuda,
            &idx,
            &lp,
            &mask,
            &grad_cuda,
            top_k,
            OpdLossOutputKt::PerPosition,
        )
        .expect("cuda backward");

        assert_eq!(got.shape(), &[1, seq_len, hidden_size]);
        let expected_v = read_f32(&expected);
        let got_v = read_f32(&got);
        let mut max_abs = 0.0f32;
        for (&a, &b) in expected_v.iter().zip(got_v.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 5e-3,
            "CUDA fused OPD backward mismatch: max_abs={max_abs:.3e}"
        );

        for (t, active) in mask.iter().copied().enumerate() {
            if !active {
                let row = &got_v[t * hidden_size..(t + 1) * hidden_size];
                assert!(
                    row.iter().all(|v| v.abs() < 1e-7),
                    "inactive row {t} should stay zero"
                );
            }
        }
    }
}
