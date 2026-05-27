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
//! Like kiln-flce-kernel, kiln-opd-loss-kernel today's Phase A path
//! runs entirely on `candle_core::Tensor` ops (gather + matmul + log-
//! softmax) and Phase B wraps the analytic backward in a candle
//! [`CustomOp1`]. The raw CUDA FFI in `phase_b.rs` is only the inner
//! kernel dispatch and is still tightly coupled to candle storage
//! conversions (`CudaStorage`, `Storage::Cuda`, `BackpropOp::none`).
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
//!    Phase A candle reference (`crate::opd_top_k_reverse_kl_phase_a`).
//! 3. [`opd_top_k_reverse_kl_per_position_kt`] — kt-typed per-
//!    position forward entry point. Same forward kernel as the
//!    scalar entry point but without the final `mean_all`.
//! 4. [`compute_per_position_metrics_kt`] — kt-typed diagnostics
//!    entry point. Reuses the same forward kernel and additionally
//!    computes student / teacher entropy over the K-support.
//!
//! Forward only — backward is still TBD via the same kiln-autograd
//! hooks the FLCE kt-typed backward is waiting on; the candle
//! [`crate::phase_b::OpdLossCustomOp`] continues to own production
//! gradient flow until that lands. (#1082)
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
        exp, index_select, log_softmax_last_dim, matmul, mean_all, mul, neg, scatter_add, sub,
        sum_axis, to_f32,
    },
    DType as KtDType, Error as KtError, Tensor as KtTensor,
};

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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
    // landed alongside the production-caller migration.
    use kiln_tensor::Device as KtDevice;
    let dev = hidden.device();
    let upload_u32 = |vals: &[u32], shape: Vec<usize>| -> Result<KtTensor, OpdLossError> {
        match dev {
            KtDevice::Cpu => KtTensor::from_vec(vals.to_vec(), shape).map_err(OpdLossError::Kt),
            #[cfg(feature = "cuda")]
            KtDevice::Cuda(i) => {
                KtTensor::cuda_from_slice(vals, shape, i).map_err(OpdLossError::Kt)
            }
            #[cfg(not(feature = "cuda"))]
            other => Err(OpdLossError::msg(format!(
                "kt-opd-loss: unsupported device for index tensor {other}"
            ))),
            #[cfg(feature = "cuda")]
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
            #[cfg(not(feature = "cuda"))]
            other => Err(OpdLossError::msg(format!(
                "kt-opd-loss: unsupported device for q_logprobs {other}"
            ))),
            #[cfg(feature = "cuda")]
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
/// Re-implements the candle Phase A reference
/// (`crate::opd_top_k_reverse_kl_phase_a`) over `kiln_tensor` ops.
/// Numerically equivalent up to floating-point associativity in the
/// matmul and the per-row log-softmax / KL reductions.
///
/// # Shape contract (matches the candle-typed
/// [`crate::opd_top_k_reverse_kl`] entry point)
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
/// Backward is not yet implemented in the kt-typed path — it still
/// lives in [`crate::phase_b::OpdLossCustomOp`] and will be migrated
/// once kt-tensor has the necessary autograd hooks. Until then this
/// entry point is forward-only.
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

/// kt-typed mirror of [`crate::phase_b::OpdLossOutput`].
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
/// active-row gradients are computed by the fused kernel; non-active
/// positions are zero (via `scatter_add` into a zero buffer along
/// axis 0).
///
/// # Errors
///
/// Returns [`OpdLossError::Msg`] when:
/// - `top_k` / `dtype` are outside `cuda_kernel_supports`.
/// - any input is non-contiguous / non-CUDA / wrong dtype.
/// - `grad_loss` shape disagrees with `output_mode`.
/// - the FFI kernel returns a non-zero status.
#[cfg(feature = "cuda")]
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
        alloc_cuda_tensor, cuda_input_device_ptr, cuda_output_device_ptr,
        cuda_storage_and_byte_offset,
    };
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
    let device_index = match hidden.device() {
        KtDevice::Cuda(i) => i,
        other => {
            return Err(OpdLossError::msg(format!(
                "kt-opd-loss bwd: hidden must be on CUDA, got {other}",
            )));
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
        let (h_st, _) = cuda_storage_and_byte_offset(hidden, dtype, "hidden")?;
        let zeros = alloc_cuda_tensor(h_st, dtype, vec![1, seq_len, hidden_size])?;
        return Ok(zeros);
    }

    // -- 3. Upload host-side teacher tensors + scatter inputs -----------
    //
    // active_indices (U32, [T_active]) — both the gather of hidden rows
    // and the scatter of d_hidden rows use the same index buffer.
    let active_indices = KtTensor::cuda_from_slice(
        active_positions.as_slice(),
        vec![active_count],
        device_index,
    )?;

    // Gather active hidden rows on device, then take contiguous.
    let hidden_2d = hidden.squeeze(0)?;
    let active_hidden = index_select(&hidden_2d, 0, &active_indices)?.contiguous()?;

    // Ensure head_t is contiguous; the kernel reads from start_offset 0.
    let head_t_contig = if head_t.is_contiguous() {
        head_t.clone()
    } else {
        head_t.contiguous()?
    };

    // Upload teacher tensors as 2-D for the FFI (`[T_active, K]`).
    let topk_idx_dev = KtTensor::cuda_from_slice(
        teacher_topk_indices,
        vec![active_count, top_k],
        device_index,
    )?;
    let topk_lp_q_dev = KtTensor::cuda_from_slice(
        teacher_topk_logprobs,
        vec![active_count, top_k],
        device_index,
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

    // -- 4. Allocate output buffer [T_active, H] on device --------------
    let (h_st, _) = cuda_storage_and_byte_offset(hidden, dtype, "hidden")?;
    let d_hidden_active =
        alloc_cuda_tensor(h_st, dtype, vec![active_count, hidden_size])?;

    // -- 5. Pull device pointers ----------------------------------------
    //
    // active_hidden / head_t_contig are typed in hidden's dtype
    // (F32 or BF16); the K indices are U32; logprobs and grad_loss
    // are F32.
    let h_ptr = cuda_input_device_ptr(&active_hidden, dtype, "active_hidden")?;
    let head_ptr = cuda_input_device_ptr(&head_t_contig, dtype, "head_t")?;
    let i_ptr = cuda_input_device_ptr(&topk_idx_dev, KtDType::U32, "topk_idx")?;
    let l_ptr =
        cuda_input_device_ptr(&topk_lp_q_dev, KtDType::F32, "topk_lp_q")?;
    let g_ptr =
        cuda_input_device_ptr(&grad_loss_dev, KtDType::F32, "grad_loss")?;
    let d_ptr = cuda_output_device_ptr(&d_hidden_active);
    let raw_stream = h_st.cuda_stream_raw();

    // -- 6. Dispatch the FFI --------------------------------------------
    //
    // Same kernel symbols the candle path uses; bit-exact by
    // construction. The output buffer is freshly zero-allocated via
    // `alloc_cuda_tensor`, so the kernel's writes land in a clean
    // F32/BF16 tile of the expected size.
    let status = unsafe {
        match dtype {
            KtDType::F32 => crate::phase_b::kiln_opd_topk_kl_bwd_f32(
                h_ptr as *const _,
                head_ptr as *const _,
                i_ptr as *const _,
                l_ptr as *const _,
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

    // -- 7. Scatter `[T_active, H]` back into `[T, H]`, unsqueeze to ---
    //       `[1, T, H]`. `scatter_add` on CUDA supports axis=0 + 1-D
    //       U32 indices + F32/BF16 values + contiguous inputs — the
    //       exact envelope we built above.
    let d_hidden_2d = scatter_add(&d_hidden_active, 0, &active_indices, seq_len)?;
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

    /// Read a 1-D or 0-D F32 [`KtTensor`] into a Vec for assertions.
    fn read_f32(t: &KtTensor) -> Vec<f32> {
        use kiln_tensor::CpuStorage;
        let cpu = t
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

}
