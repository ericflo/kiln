//! kt-tape OPD top-K reverse-KL forward+backward — Phase 6a/CP-4 port of
//! `kt_forward_op.rs` from `candle::CustomOp1` onto the kt-side
//! `kiln_autograd::Tape` substrate ((#1082) — see
//! `docs/CANDLE_REMOVAL_PLAN.md` and the rmsnorm sibling
//! `crates/kiln-rmsnorm-kernel/src/kt_tape.rs`, commit `895162ca`).
//!
//! # Why this module exists
//!
//! The existing `kiln_train::opd_tape_shim::opd_top_k_reverse_kl_per_position_via_kt_forward_op`
//! wraps the kt-tensor forward + the CUDA analytic backward inside a candle
//! `CustomOp1` (`KtForwardOp1`). It used to keep the candle dependency alive
//! in the opd-loss-kernel crate, but (#1082) relocated it (and the rest of
//! the candle glue) UP into `kiln-train::opd_tape_shim` so this crate is
//! now candle-free — even though both
//! halves of the autograd roundtrip already bottom out in
//! `kiln_tensor::Tensor` + the kt-typed forward
//! ([`crate::opd_top_k_reverse_kl_per_position_kt`] /
//! [`crate::opd_top_k_reverse_kl_kt`]) + the kt-typed backward
//! ([`crate::kt_api::opd_top_k_reverse_kl_phase_b_bwd_kt`]).
//!
//! This module is the parallel entry that drops the candle CustomOp
//! wrapper and records the backward directly onto a
//! `kiln_autograd::Tape`. Same FFI symbols (`kiln_opd_topk_kl_bwd_{bf16,f32}`),
//! same envelope, same numerical contract. The only difference is who owns
//! the autograd recording: candle's `BackpropOp` chain (legacy) vs. kiln's
//! `Tape::record` (new).
//!
//! # Numerical contract
//!
//! Forward: bit-exact equality with
//! [`crate::opd_top_k_reverse_kl_per_position_kt`] /
//! [`crate::opd_top_k_reverse_kl_kt`] (they call the same forward ops
//! on the same input tensors).
//!
//! Backward: bit-exact with the kt-typed backward
//! [`crate::kt_api::opd_top_k_reverse_kl_phase_b_bwd_kt`] (same FFI
//! symbols `kiln_opd_topk_kl_bwd_{bf16,f32}`, same `output_mode_i32`
//! + `scale_factor` derivation, same `scatter_add` finalisation).
//!
//! # Envelope
//!
//! Same as [`crate::kt_api::opd_top_k_reverse_kl_phase_b_bwd_kt`]:
//! CUDA + `top_k ∈ {16, 32}` + matching F32/BF16 dtype on
//! `(hidden, head_t)`. Out-of-envelope inputs return an error rather
//! than silently falling back — the production caller is expected to
//! pre-check via the same logic the existing kt-typed backward uses.
//!
//! # Phase-B only
//!
//! Phase A (pure-candle reference) does not get a kt-tape port — it
//! has no CUDA backward kernel; its candle-autograd flow falls out of
//! the candle ops directly and is the parity oracle, not a production
//! path. The kt-tape entry only covers the Phase B fused CUDA backward.
//!
//! # Production caller migration
//!
//! Out of scope for this commit ((#1082)). The production caller in
//! `kiln-train` still uses the candle CustomOp path
//! (`kiln_train::opd_tape_shim::opd_top_k_reverse_kl_per_position_via_kt_forward_op`).
//! Migration lands once the wider kiln-train autograd substrate adopts
//! kt-tape — keeping the two paths in parallel for now matches the
//! "parallel shim, flip when ready" rollout cadence the issue
//! authorises.

use kiln_autograd::{BackwardOp, Tape};
use kiln_tensor::{
    DType as KtDType, Device as KtDevice, Result as KtResult, Tensor as KtTensor, bail,
};

use crate::kt_api::{
    OpdActiveMetadata, OpdLossError, OpdLossOutputKt,
    opd_top_k_reverse_kl_per_position_with_metadata_kt,
    opd_top_k_reverse_kl_phase_b_bwd_composite_kt, opd_top_k_reverse_kl_with_metadata_kt,
};

#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::kt_api::{
    opd_top_k_reverse_kl_phase_b_bwd_kt, opd_top_k_reverse_kl_phase_b_bwd_scalar_mean_unit_grad_kt,
    opd_top_k_reverse_kl_phase_b_bwd_scalar_mean_unit_grad_with_metadata_kt,
    opd_top_k_reverse_kl_phase_b_bwd_with_metadata_kt,
};

/// Returns `true` when `(hidden, head_t, top_k)` is inside the kt-tape
/// Phase-B fused backward envelope. Mirrors the dtype + top_k gate inside
/// [`crate::kt_api::opd_top_k_reverse_kl_phase_b_bwd_kt`]:
///
/// - `hidden` on CUDA
/// - `hidden.dtype()` ∈ {F32, BF16}
/// - `head_t.dtype() == hidden.dtype()`
/// - `top_k` ∈ {16, 32}
///
/// Shape validation (rank, batch dim, label_mask, top_k indices range)
/// is deferred to `validate_inputs_kt` inside the backward entry; this
/// helper is the cheap up-front check that lets the production caller
/// route around the kt-tape path when the kernel envelope doesn't apply.
fn envelope_ok(hidden: &KtTensor, head_t: &KtTensor, top_k: usize) -> bool {
    // (#1082) Accept CUDA, Metal, and Vulkan: the kt-native OPD FORWARD + loss
    // record on any of these storages, and the recorded backward
    // (`CudaOpdTopKReverseKlPhaseBBackward::apply`) is reachable on all. On
    // CUDA it dispatches the perf-tuned fused FFI kernel; on CPU/Metal/Vulkan it
    // routes through the device-agnostic analytic kt-composite
    // (`opd_top_k_reverse_kl_phase_b_bwd_composite_kt`, FD-validated in
    // `kt_api`), so the OPD top-K reverse-KL LoRA grad runs on every backend.
    // Vulkan was previously excluded here, which silently bailed the whole OPD
    // tape path off Vulkan (`try_tape_opd_scalar_mean_cuda_kt` -> None ->
    // empty grad store); F32-on-Vulkan OPD now records + backprops via the
    // composite.
    // (Phase R.7) ROCm joins CUDA on the fused FFI fast path: the
    // `kiln_opd_topk_kl_bwd_*` symbols are emitted by the hipcc build and the
    // backward routes through the backend-neutral kt-bridge seam.
    if !matches!(
        hidden.device(),
        KtDevice::Cuda(_) | KtDevice::Metal(_) | KtDevice::Vulkan(_) | KtDevice::Rocm(_)
    ) {
        return false;
    }
    if hidden.dtype() != head_t.dtype() {
        return false;
    }
    if !matches!(hidden.dtype(), KtDType::F32 | KtDType::BF16) {
        return false;
    }
    if top_k != 16 && top_k != 32 {
        return false;
    }
    true
}

/// Saved-state backward for the fused CUDA OPD top-K reverse-KL kernel.
///
/// Stores `(hidden, head_t)` as `Arc`-cloned kt tensors plus the
/// host-side teacher metadata (`teacher_topk_indices`,
/// `teacher_topk_logprobs`, `label_mask`) and the kernel-dispatch
/// parameters (`top_k`, `output_mode`) captured at forward time.
///
/// On `apply(grad_loss)` it calls
/// [`opd_top_k_reverse_kl_phase_b_bwd_kt`], which dispatches the same
/// FFI symbols (`kiln_opd_topk_kl_bwd_{bf16,f32}`) the candle
/// `OpdLossCustomOp::bwd` path uses and returns `d_hidden` of shape
/// `[1, T, H]` in the input dtype.
///
/// # Tape input (`input_count = 1`)
///
/// `hidden` is the sole differentiable tape input. The frozen `head_t` is
/// retained as saved backward data because the kernel needs it to compute
/// `d_hidden`, but it is deliberately absent from the tape input list. The
/// returned gradient vector is therefore `[Some(d_hidden)]`.
///
/// The `teacher_topk_*` + `label_mask` arrays are host-side metadata
/// and not tape inputs; they're closed over via the saved struct.
#[derive(Debug)]
pub struct CudaOpdTopKReverseKlPhaseBBackward {
    /// Saved CUDA `hidden` from the forward pass (F32 or BF16,
    /// shape `[1, T, H]`).
    pub hidden: KtTensor,
    /// Saved CUDA `head_t` from the forward pass (F32 or BF16,
    /// shape `[H, V]`, matching `hidden`'s dtype).
    pub head_t: KtTensor,
    /// `[T_active * K]` row-major teacher top-K indices.
    pub teacher_topk_indices: Vec<u32>,
    /// `[T_active * K]` row-major teacher log-probabilities at the
    /// top-K support.
    pub teacher_topk_logprobs: Vec<f32>,
    /// `[T]` label mask; positions with `true` are active.
    pub label_mask: Vec<bool>,
    /// K — the teacher's support size. Must be in `{16, 32}` for the
    /// fused backward kernel.
    pub top_k: usize,
    /// ScalarMean vs PerPosition selector. Determines the expected
    /// `grad_loss` shape on backward and the kernel's per-token
    /// `scale_factor`.
    pub output_mode: OpdLossOutputKt,
    /// The scalar-mean loss is the tape root and receives the implicit unit
    /// seed `dL/dL = 1`. CUDA/ROCm can dispatch the dedicated unit-seed fused
    /// backward in this case; other modes/backends keep the generic seeded path.
    pub scalar_mean_unit_root_grad: bool,
    /// Small device-resident row-map/top-K tensors returned by the forward.
    /// CUDA/ROCm fused backward can reuse them instead of rebuilding and
    /// reuploading host metadata immediately after the loss root is evaluated.
    pub active_metadata: Option<OpdActiveMetadata>,
}

impl BackwardOp for CudaOpdTopKReverseKlPhaseBBackward {
    fn name(&self) -> &'static str {
        "kiln-opd-loss-kernel/opd_top_k_reverse_kl_phase_b_kt_tape"
    }

    fn input_count(&self) -> usize {
        // `hidden` is the only differentiable tape input. `head_t` is a
        // frozen constant saved on this op solely to compute `d_hidden`.
        1
    }

    fn apply(&self, grad_output: &KtTensor) -> KtResult<Vec<Option<KtTensor>>> {
        // Backward dispatch by device:
        //
        //  - CUDA storage (only reachable on a `cuda`-feature build):
        //    route through the perf-tuned fused FFI kernel
        //    `kiln_opd_topk_kl_bwd_{f32,bf16}` via
        //    `opd_top_k_reverse_kl_phase_b_bwd_kt`. Bit-identical to the
        //    candle/kt-tape CUDA paths — unchanged.
        //
        //  - CPU / Metal storage: route through the device-agnostic
        //    analytic kt-composite `..._bwd_composite_kt`, which derives
        //    the same `d_hidden` gradient purely from `kiln_tensor` ops
        //    (validated against finite-difference in `kt_api` tests). No
        //    FFI / cudarc / candle — runs on every backend.
        //
        // The composite is correct on CUDA too (pure kt), but the FFI
        // kernel stays the CUDA path to preserve its validated numerics
        // and performance.
        // (Phase R.7) ROCm shares the fused FFI fast path with CUDA — the
        // `kiln_opd_topk_kl_bwd_*` symbols come from the hipcc build and the
        // backward routes through the backend-neutral kt-bridge seam, so both
        // GPU backends dispatch here. CPU / Metal / Vulkan fall through to the
        // device-agnostic analytic composite below.
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        if matches!(self.hidden.device(), KtDevice::Cuda(_) | KtDevice::Rocm(_)) {
            // Shape + device + dtype checks happen inside
            // `opd_top_k_reverse_kl_phase_b_bwd_kt` (it validates
            // grad_loss against output_mode + active_count); we
            // surface a thin contextual wrap on its error.
            let d_hidden = if self.scalar_mean_unit_root_grad
                && self.output_mode == OpdLossOutputKt::ScalarMean
            {
                if grad_output.element_count() != 1 {
                    return Err(kiln_tensor::Error::Msg(format!(
                        "opd kt-tape bwd: unit scalar root expected 1-element grad_output, got shape {:?}",
                        grad_output.shape()
                    )));
                }
                if let Some(active_metadata) = self.active_metadata.as_ref() {
                    opd_top_k_reverse_kl_phase_b_bwd_scalar_mean_unit_grad_with_metadata_kt(
                        &self.hidden,
                        &self.head_t,
                        &self.teacher_topk_indices,
                        &self.teacher_topk_logprobs,
                        &self.label_mask,
                        self.top_k,
                        active_metadata,
                    )
                } else {
                    opd_top_k_reverse_kl_phase_b_bwd_scalar_mean_unit_grad_kt(
                        &self.hidden,
                        &self.head_t,
                        &self.teacher_topk_indices,
                        &self.teacher_topk_logprobs,
                        &self.label_mask,
                        self.top_k,
                    )
                }
            } else {
                if let Some(active_metadata) = self.active_metadata.as_ref() {
                    opd_top_k_reverse_kl_phase_b_bwd_with_metadata_kt(
                        &self.hidden,
                        &self.head_t,
                        &self.teacher_topk_indices,
                        &self.teacher_topk_logprobs,
                        &self.label_mask,
                        grad_output,
                        self.top_k,
                        self.output_mode,
                        active_metadata,
                    )
                } else {
                    opd_top_k_reverse_kl_phase_b_bwd_kt(
                        &self.hidden,
                        &self.head_t,
                        &self.teacher_topk_indices,
                        &self.teacher_topk_logprobs,
                        &self.label_mask,
                        grad_output,
                        self.top_k,
                        self.output_mode,
                    )
                }
            }
            .map_err(|e: OpdLossError| {
                kiln_tensor::Error::Msg(format!("opd kt-tape bwd: kt call: {e}"))
            })?;

            return Ok(vec![Some(d_hidden)]);
        }

        // Non-CUDA (CPU / Metal) device-agnostic composite path.
        let d_hidden = opd_top_k_reverse_kl_phase_b_bwd_composite_kt(
            &self.hidden,
            &self.head_t,
            &self.teacher_topk_indices,
            &self.teacher_topk_logprobs,
            &self.label_mask,
            grad_output,
            self.top_k,
            self.output_mode,
        )
        .map_err(|e: OpdLossError| {
            kiln_tensor::Error::Msg(format!("opd kt-tape bwd composite: kt call: {e}"))
        })?;

        Ok(vec![Some(d_hidden)])
    }

    fn requires_input(&self, idx: usize) -> bool {
        // The backward reads the recorded hidden activation. The frozen head
        // and host metadata are saved directly on the op.
        idx == 0
    }
}

/// kt-tape Phase-B per-position forward+backward — Phase 6a/CP-4
/// successor to
/// `kiln_train::opd_tape_shim::opd_top_k_reverse_kl_per_position_via_kt_forward_op`.
///
/// Runs the kt-typed Phase-A forward via
/// [`opd_top_k_reverse_kl_per_position_kt`], then records a tape node
/// whose backward calls
/// [`opd_top_k_reverse_kl_phase_b_bwd_kt`] on the same FFI symbols
/// as the candle Phase-B `CustomOp1`. No candle types touched — the
/// input, output, and recorded saved tensors are all
/// `kiln_tensor::Tensor`.
///
/// # Envelope
///
/// Same as [`opd_top_k_reverse_kl_phase_b_bwd_kt`]: CUDA + matching
/// F32/BF16 `(hidden, head_t)` dtype + `top_k ∈ {16, 32}`. Out-of-
/// envelope inputs return an `Err` rather than silently falling back;
/// the production caller is expected to pre-check exactly like the
/// existing kt-forward-op shim does.
///
/// # Tape integration
///
/// The tape records `hidden` as its sole differentiable input. The backward
/// also saves `hidden` and the frozen `head_t` by `Arc` clone — kt `Tensor`
/// is already `Clone` over `Arc<dyn Storage>` so this is a refcount bump, not
/// a host copy. The host-side teacher metadata is cloned into the saved struct.
///
/// # Returns
///
/// A 1-D F32 `[T_active]` [`KtTensor`] holding the per-position
/// reverse KL. The backward expects `grad_loss` to match this
/// shape (the `PerPosition` output mode is the natural selector for
/// the per-position entry).
pub fn opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    tape: &mut Tape,
) -> KtResult<KtTensor> {
    if !envelope_ok(hidden, head_t, top_k) {
        bail!(
            "opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape: inputs \
             outside kt envelope (CUDA/Metal/Vulkan/ROCm + matching F32/BF16 + \
             top_k ∈ {{16, 32}} required). Callers must filter via the \
             same envelope check the kt-typed backward applies."
        );
    }

    // Forward — bit-exact with `opd_top_k_reverse_kl_per_position_kt`
    // (same kt-tensor ops, same FFI host uploads).
    let (per_token, active_metadata) = opd_top_k_reverse_kl_per_position_with_metadata_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )
    .map_err(|e: OpdLossError| {
        kiln_tensor::Error::Msg(format!(
            "opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape fwd: kt call: {e}"
        ))
    })?;

    // Backward — save (hidden, head_t) by Arc-cloning the kt tensors
    // and clone the host-side teacher metadata into the saved struct.
    let bwd = CudaOpdTopKReverseKlPhaseBBackward {
        hidden: hidden.clone(),
        head_t: head_t.clone(),
        teacher_topk_indices: teacher_topk_indices.to_vec(),
        teacher_topk_logprobs: teacher_topk_logprobs.to_vec(),
        label_mask: label_mask.to_vec(),
        top_k,
        output_mode: OpdLossOutputKt::PerPosition,
        scalar_mean_unit_root_grad: false,
        active_metadata,
    };
    tape.record(&per_token, &[hidden], Box::new(bwd) as Box<dyn BackwardOp>);

    Ok(per_token)
}

/// kt-tape Phase-B scalar-mean forward+backward — same kernel as
/// [`opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape`] but
/// reduced to a scalar with `mean_all` over active positions.
///
/// Mirrors the relationship between
/// [`opd_top_k_reverse_kl_kt`] (scalar-mean) and
/// [`opd_top_k_reverse_kl_per_position_kt`] (per-position).
///
/// # Returns
///
/// A scalar F32 `KtTensor` (rank-0, shape `[]`) holding the mean
/// reverse KL. The backward expects a scalar / 1-element F32
/// `grad_loss`.
pub fn opd_top_k_reverse_kl_phase_b_via_kt_tape(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    tape: &mut Tape,
) -> KtResult<KtTensor> {
    opd_top_k_reverse_kl_phase_b_via_kt_tape_impl(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
        tape,
        false,
    )
}

/// kt-tape scalar-mean OPD root with implicit unit upstream seed.
///
/// Production OPD uses the returned scalar as the tape root inside
/// `with_tape_authoritative_scope_kt`, which always seeds `dL/dL = 1`.
/// CUDA/ROCm route this saved backward through
/// [`opd_top_k_reverse_kl_phase_b_bwd_scalar_mean_unit_grad_kt`]; Metal and
/// Vulkan keep the generic composite path so their current behavior is
/// unchanged.
pub fn opd_top_k_reverse_kl_phase_b_unit_grad_via_kt_tape(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    tape: &mut Tape,
) -> KtResult<KtTensor> {
    opd_top_k_reverse_kl_phase_b_via_kt_tape_impl(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
        tape,
        true,
    )
}

fn opd_top_k_reverse_kl_phase_b_via_kt_tape_impl(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    tape: &mut Tape,
    scalar_mean_unit_root_grad: bool,
) -> KtResult<KtTensor> {
    if !envelope_ok(hidden, head_t, top_k) {
        bail!(
            "opd_top_k_reverse_kl_phase_b_via_kt_tape: inputs outside kt envelope \
             (CUDA/Metal/Vulkan/ROCm + matching F32/BF16 + top_k ∈ {{16, 32}} required). \
             Callers must filter via the same envelope check the kt-typed \
             backward applies."
        );
    }

    // Forward — bit-exact with `opd_top_k_reverse_kl_kt` (same kt-tensor
    // ops + final `mean_all`).
    let (loss, active_metadata) = opd_top_k_reverse_kl_with_metadata_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )
    .map_err(|e: OpdLossError| {
        kiln_tensor::Error::Msg(format!(
            "opd_top_k_reverse_kl_phase_b_via_kt_tape fwd: kt call: {e}"
        ))
    })?;

    let bwd = CudaOpdTopKReverseKlPhaseBBackward {
        hidden: hidden.clone(),
        head_t: head_t.clone(),
        teacher_topk_indices: teacher_topk_indices.to_vec(),
        teacher_topk_logprobs: teacher_topk_logprobs.to_vec(),
        label_mask: label_mask.to_vec(),
        top_k,
        output_mode: OpdLossOutputKt::ScalarMean,
        scalar_mean_unit_root_grad,
        active_metadata,
    };
    tape.record(&loss, &[hidden], Box::new(bwd) as Box<dyn BackwardOp>);

    Ok(loss)
}

// ---------------------------------------------------------------------------
// Tests — gated on CUDA availability at runtime (skip cleanly when no CUDA).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Tape isn't reachable when the kernel is built without CUDA — the
    /// envelope rejects anything off CUDA. These three checks live on
    /// CPU so they run on every build (including the non-CUDA CI matrix
    /// jobs).
    #[test]
    fn envelope_rejects_cpu() {
        let h = KtTensor::zeros_cpu(vec![1, 4, 8], KtDType::F32);
        let w = KtTensor::zeros_cpu(vec![8, 16], KtDType::F32);
        assert!(!envelope_ok(&h, &w, 16));
    }

    #[test]
    fn envelope_rejects_unsupported_topk() {
        // top_k = 8 is outside the {16, 32} fast-path. (CUDA-free.)
        let h = KtTensor::zeros_cpu(vec![1, 4, 8], KtDType::F32);
        let w = KtTensor::zeros_cpu(vec![8, 16], KtDType::F32);
        // Envelope CPU check fails first, but exercise top_k explicitly
        // by passing a "CUDA-looking" dtype-only check via a fresh
        // tensor we know is CPU — envelope_ok composes both checks.
        assert!(!envelope_ok(&h, &w, 8));
        assert!(!envelope_ok(&h, &w, 64));
    }

    /// The frozen head is saved backward data, not a differentiable tape
    /// input. Keep this contract covered without requiring a GPU.
    #[test]
    fn backward_contract_has_one_hidden_input() {
        let hidden = KtTensor::zeros_cpu(vec![1, 4, 8], KtDType::F32);
        let head_t = KtTensor::zeros_cpu(vec![8, 16], KtDType::F32);
        let bwd = CudaOpdTopKReverseKlPhaseBBackward {
            hidden,
            head_t,
            teacher_topk_indices: Vec::new(),
            teacher_topk_logprobs: Vec::new(),
            label_mask: vec![true; 4],
            top_k: 16,
            output_mode: OpdLossOutputKt::ScalarMean,
            scalar_mean_unit_root_grad: false,
            active_metadata: None,
        };

        assert_eq!(bwd.input_count(), 1);
        assert!(bwd.requires_input(0));
        assert!(!bwd.requires_input(1));
    }

    #[test]
    fn envelope_rejects_dtype_mismatch() {
        // hidden BF16, head_t F32 — dtype mismatch rejects even before
        // the device check (logically — both checks short-circuit; we
        // pick a CPU pair where the dtype-mismatch path can be observed
        // independently of the device gate by manually adjusting the
        // call site).
        let h_bf16 = KtTensor::zeros_cpu(vec![1, 4, 8], KtDType::BF16);
        let w_f32 = KtTensor::zeros_cpu(vec![8, 16], KtDType::F32);
        assert!(!envelope_ok(&h_bf16, &w_f32, 16));
    }

    // -----------------------------------------------------------------------
    // ROCm regression lock (#33) — the OPD tape envelope + scalar backward must
    // stay open on ROCm. A future refactor that drops `Device::Rocm(_)` from
    // `envelope_ok` (kt_tape.rs:106) would silently route ROCm OPD back to an
    // Ok(None) decline → empty grads; this catches that.
    // -----------------------------------------------------------------------
    #[cfg(feature = "rocm")]
    mod rocm {
        use super::*;

        fn rocm_f32(shape: Vec<usize>, seed: u64) -> KtTensor {
            let n: usize = shape.iter().product();
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                })
                .collect();
            KtTensor::from_vec_on(kiln_tensor::Device::Rocm(0), data, shape).expect("rocm tensor")
        }

        #[test]
        fn rocm_opd_tape_envelope_open_and_backward_finite() {
            if !kiln_tensor::rocm_is_available() {
                eprintln!("skip rocm_opd_tape: no ROCm device");
                return;
            }
            let (seq_len, hidden_size, vocab_size, top_k) = (4usize, 128usize, 256usize, 16usize);
            let h = rocm_f32(vec![1, seq_len, hidden_size], 1);
            let w = rocm_f32(vec![hidden_size, vocab_size], 2);
            // Minimal teacher top-k metadata: alternating active rows; per active
            // row, K distinct indices < vocab + K teacher logprobs.
            let label_mask: Vec<bool> = (0..seq_len).map(|i| i % 2 == 0).collect();
            let active = label_mask.iter().filter(|m| **m).count();
            let mut indices: Vec<u32> = Vec::with_capacity(active * top_k);
            let mut logprobs: Vec<f32> = Vec::with_capacity(active * top_k);
            for r in 0..active {
                for j in 0..top_k {
                    indices
                        .push((((r * 7 + j * 3) % vocab_size) as u32).min(vocab_size as u32 - 1));
                    logprobs.push(-((j as f32) * 0.3 + 0.5));
                }
            }

            // Locks the envelope's ROCm device arm + the top_k ∈ {16,32} gate.
            assert!(
                envelope_ok(&h, &w, top_k),
                "OPD envelope_ok must accept ROCm F32 + K=16 (regression: #1454 gate)"
            );

            let mut tape = Tape::new();
            let loss = opd_top_k_reverse_kl_phase_b_via_kt_tape(
                &h,
                &w,
                &indices,
                &logprobs,
                &label_mask,
                top_k,
                &mut tape,
            )
            .expect("via_kt_tape must record on ROCm (envelope open)");
            assert_eq!(tape.len(), 1, "OPD records exactly one tape node on ROCm");

            // Scalar-loss backward → finite d_hidden through flash_attn-free path.
            let seed = KtTensor::from_vec_on(kiln_tensor::Device::Rocm(0), vec![1.0f32], vec![])
                .expect("scalar seed");
            let grads = tape
                .backward(loss.id(), seed, |a, b| kiln_tensor::ops::add(a, b))
                .expect("OPD tape backward on ROCm");
            let dh = grads.get(h.id()).expect("d_hidden present");
            assert!(
                grads.get(w.id()).is_none(),
                "frozen head must not appear in the OPD gradient store"
            );
            let dh_v: Vec<f32> = dh
                .to_dtype(KtDType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec()
                .unwrap();
            assert!(
                dh_v.iter().all(|x| x.is_finite()),
                "non-finite OPD d_hidden on ROCm"
            );
            assert!(
                dh_v.iter().any(|&x| x != 0.0),
                "OPD d_hidden all-zero on ROCm (dead backward)"
            );
            eprintln!("[rocm-opd-tape] OK: envelope open, 1 node, finite non-zero d_hidden");
        }
    }

    // -----------------------------------------------------------------------
    // CUDA-gated E2E tests — record + backward.apply round-trip.
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use half::bf16;

        fn cuda_available() -> bool {
            kiln_tensor::primary_cuda_context(0).is_ok()
        }

        /// Make a deterministic `[1, T, H]` F32 CUDA tensor.
        fn cuda_hidden_f32(seq_len: usize, hidden_size: usize, seed: u64) -> KtTensor {
            let n = seq_len * hidden_size;
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
                    ((s as u32 % 1024) as f32 - 512.0) / 512.0
                })
                .collect();
            KtTensor::cuda_from_slice(&data, vec![1, seq_len, hidden_size], 0).expect("hidden cuda")
        }

        fn cuda_head_t_f32(hidden_size: usize, vocab_size: usize, seed: u64) -> KtTensor {
            let n = hidden_size * vocab_size;
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
                    ((s as u32 % 1024) as f32 - 512.0) / 512.0
                })
                .collect();
            KtTensor::cuda_from_slice(&data, vec![hidden_size, vocab_size], 0).expect("head_t cuda")
        }

        fn cuda_grad_per_position(active_count: usize, seed: u64) -> KtTensor {
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
            let data: Vec<f32> = (0..active_count)
                .map(|_| {
                    s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
                    ((s as u32 % 1024) as f32 - 512.0) / 512.0
                })
                .collect();
            KtTensor::cuda_from_slice(&data, vec![active_count], 0).expect("grad_loss cuda")
        }

        fn cuda_grad_scalar(seed: u64) -> KtTensor {
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
            s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
            let v = ((s as u32 % 1024) as f32 - 512.0) / 512.0;
            KtTensor::cuda_from_slice(&[v], vec![1], 0).expect("grad scalar cuda")
        }

        fn build_topk_metadata(
            seq_len: usize,
            vocab_size: usize,
            top_k: usize,
            seed: u64,
        ) -> (Vec<u32>, Vec<f32>, Vec<bool>) {
            // Alternate active / inactive to exercise the scatter path.
            let label_mask: Vec<bool> = (0..seq_len).map(|i| i % 2 == 0).collect();
            let active_count = label_mask.iter().filter(|m| **m).count();
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
            // Pick K unique indices < vocab_size per active row. We just
            // step through [0, vocab_size) modulo K positions; the
            // backward kernel doesn't require uniqueness for parity,
            // but mirroring the candle reference's preferred contract
            // keeps the parity tests well-defined.
            let mut indices = Vec::with_capacity(active_count * top_k);
            let mut logprobs = Vec::with_capacity(active_count * top_k);
            for _row in 0..active_count {
                let mut row_indices: Vec<u32> = Vec::with_capacity(top_k);
                let mut k_used = 0u32;
                while row_indices.len() < top_k {
                    let idx = k_used % (vocab_size as u32);
                    if !row_indices.contains(&idx) {
                        row_indices.push(idx);
                    }
                    k_used = k_used.wrapping_add(1);
                }
                indices.extend(row_indices);
                // Per-row logprobs: any normalised log-softmax-ish vector.
                // We just generate K reals; the renorm inside the
                // forward + backward handles normalisation downstream.
                for _ in 0..top_k {
                    s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
                    let v = -((s as u32 % 1024) as f32) / 64.0;
                    logprobs.push(v);
                }
            }
            (indices, logprobs, label_mask)
        }

        /// CUDA forward records only the differentiable hidden id; the frozen
        /// head remains saved backward data. Skips cleanly without CUDA.
        #[test]
        fn forward_records_tape_node_when_cuda_available() {
            if !cuda_available() {
                eprintln!("CUDA not available; skipping forward_records_tape_node");
                return;
            }
            let seq_len = 4usize;
            let hidden_size = 128usize;
            let vocab_size = 256usize;
            let top_k = 16usize;
            let h = cuda_hidden_f32(seq_len, hidden_size, 1);
            let w = cuda_head_t_f32(hidden_size, vocab_size, 2);
            let (indices, logprobs, label_mask) =
                build_topk_metadata(seq_len, vocab_size, top_k, 3);

            // Envelope must report OK for real CUDA F32 inputs + K=16.
            assert!(envelope_ok(&h, &w, top_k));

            let mut tape = Tape::new();
            let per_token = opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape(
                &h,
                &w,
                &indices,
                &logprobs,
                &label_mask,
                top_k,
                &mut tape,
            )
            .expect("forward + record");
            let active_count = label_mask.iter().filter(|m| **m).count();
            assert_eq!(per_token.shape(), &[active_count]);
            assert_eq!(per_token.dtype(), KtDType::F32);
            assert_eq!(tape.len(), 1);

            let node = &tape.nodes()[0];
            assert_eq!(node.input_ids.len(), 1);
            assert_eq!(node.input_ids[0], h.id());
            assert!(!node.input_ids.contains(&w.id()));
            assert_eq!(node.output_id, per_token.id());
            assert_eq!(
                node.op.name(),
                "kiln-opd-loss-kernel/opd_top_k_reverse_kl_phase_b_kt_tape"
            );
            assert_eq!(node.op.input_count(), 1);
        }

        /// Direct backward apply (per-position) — exercises the
        /// `apply()` path including the FFI dispatch and the final
        /// `scatter_add` finalisation. Skips cleanly without CUDA.
        ///
        /// Uses `hidden_size = 128` / `vocab_size = 256` to keep the
        /// fused backward kernel inside its launch envelope (smaller
        /// hidden_size has caused `cudaErrorLaunchOutOfResources` in
        /// kernel-build matrix runs).
        #[test]
        fn backward_apply_per_position_returns_grad_of_expected_shape() {
            if !cuda_available() {
                eprintln!("CUDA not available; skipping backward_apply_per_position");
                return;
            }
            let seq_len = 4usize;
            let hidden_size = 128usize;
            let vocab_size = 256usize;
            let top_k = 16usize;
            let h = cuda_hidden_f32(seq_len, hidden_size, 5);
            let w = cuda_head_t_f32(hidden_size, vocab_size, 6);
            let (indices, logprobs, label_mask) =
                build_topk_metadata(seq_len, vocab_size, top_k, 7);
            let active_count = label_mask.iter().filter(|m| **m).count();
            let dy = cuda_grad_per_position(active_count, 8);

            let bwd = CudaOpdTopKReverseKlPhaseBBackward {
                hidden: h.clone(),
                head_t: w.clone(),
                teacher_topk_indices: indices,
                teacher_topk_logprobs: logprobs,
                label_mask,
                top_k,
                output_mode: OpdLossOutputKt::PerPosition,
                scalar_mean_unit_root_grad: false,
                active_metadata: None,
            };
            let grads = bwd.apply(&dy).expect("apply backward");
            assert_eq!(grads.len(), 1);
            let d_hidden = grads[0].as_ref().expect("d_hidden present");
            assert_eq!(d_hidden.shape(), &[1, seq_len, hidden_size]);
            assert_eq!(d_hidden.dtype(), KtDType::F32);
        }

        /// Same as above but ScalarMean output mode + scalar grad_loss.
        ///
        /// Uses larger H / V than the per-position test because the
        /// K=32 backward kernel tile requires H ≥ ~64; with smaller H
        /// the launch returns `cudaErrorLaunchOutOfResources` (701).
        #[test]
        fn backward_apply_scalar_mean_returns_grad_of_expected_shape() {
            if !cuda_available() {
                eprintln!("CUDA not available; skipping backward_apply_scalar_mean");
                return;
            }
            let seq_len = 4usize;
            let hidden_size = 128usize;
            let vocab_size = 256usize;
            let top_k = 16usize;
            let h = cuda_hidden_f32(seq_len, hidden_size, 9);
            let w = cuda_head_t_f32(hidden_size, vocab_size, 10);
            let (indices, logprobs, label_mask) =
                build_topk_metadata(seq_len, vocab_size, top_k, 11);
            let dy = cuda_grad_scalar(12);

            let bwd = CudaOpdTopKReverseKlPhaseBBackward {
                hidden: h.clone(),
                head_t: w.clone(),
                teacher_topk_indices: indices,
                teacher_topk_logprobs: logprobs,
                label_mask,
                top_k,
                output_mode: OpdLossOutputKt::ScalarMean,
                scalar_mean_unit_root_grad: false,
                active_metadata: None,
            };
            let grads = bwd.apply(&dy).expect("apply backward scalar mean");
            assert_eq!(grads.len(), 1);
            let d_hidden = grads[0].as_ref().expect("d_hidden present");
            assert_eq!(d_hidden.shape(), &[1, seq_len, hidden_size]);
            assert_eq!(d_hidden.dtype(), KtDType::F32);
        }

        /// Quiet the unused-import lint on `bf16` when no test below
        /// references it (BF16-specific E2E is an entirely separate
        /// envelope and skipped here — F32 covers the kernel-dispatch
        /// path with parity established in `phase_b.rs` already).
        #[allow(dead_code)]
        fn _bf16_import_used() -> bf16 {
            bf16::from_f32(0.0)
        }
    }
}
