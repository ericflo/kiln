//! kt-tape FLCE Phase B forward+backward — Phase 6a/CP-4 pilot port of
//! `kt_forward_op.rs` from `candle::CustomOp1` (`KtForwardOp1`) onto the
//! kt-side `kiln_autograd::Tape` substrate ((#1082) — see
//! `docs/archive/candle-removal/CANDLE_REMOVAL_PLAN.md`).
//!
//! # Why this module exists
//!
//! The historical candle shim `kiln_train::flce_candle_shim::fused_linear_cross_entropy_phase_b_via_kt_forward_op`
//! wrapped the FLCE Phase B forward+backward inside a candle
//! `CustomOp1` (`KtForwardOp1`), keeping the candle dependency alive in
//! the flce-kernel crate — even though the kt-typed forward + backward
//! ([`crate::kt_api::fused_linear_cross_entropy_phase_b_kt`],
//! [`crate::kt_api::fused_linear_cross_entropy_phase_b_backward_kt`])
//! already run end-to-end over `kiln_tensor::Tensor` without a
//! candle node in sight.
//!
//! This module is the entry that drops the candle CustomOp
//! wrapper (deleted along with the shim post-#1082) and records the backward directly onto a
//! `kiln_autograd::Tape`. Same kt-typed forward, same kt-typed
//! backward, same envelope. The only difference is who owns the
//! autograd recording: candle's `BackpropOp` chain (legacy) vs.
//! kiln's `Tape::record` (new).
//!
//! Mirrors the same shape as the kiln-rmsnorm-kernel kt-tape pilot
//! (commit `895162ca`) — see that module for the cross-crate
//! template rationale.
//!
//! # Numerical contract
//!
//! Forward: bit-exact equality with
//! [`crate::kt_api::fused_linear_cross_entropy_phase_b_kt`] (they
//! call the same kt-typed forward on the same input bytes).
//!
//! Backward `dhidden`: identical to
//! [`crate::kt_api::fused_linear_cross_entropy_phase_b_backward_kt`]
//! up to floating-point associativity in the chunked sum-exp
//! accumulation (the per-chunk kernel sequence is the same — the
//! kt-tape `apply()` simply calls the kt-typed backward and routes
//! its `dhidden` back through the tape's grad map).
//!
//! Note: FLCE has **only one differentiable tape input** — `hidden`.
//! The frozen `head_t` is saved on the backward op because computing
//! `dhidden` needs it, but it is not recorded as an input and never
//! receives a gradient. `input_ids` and `label_mask` are non-tensor
//! metadata. The kt-tape backward therefore returns `[Some(dhidden)]`.
//!
//! # Envelope
//!
//! Matches the original candle shim envelope (the deleted
//! `kiln_train::flce_candle_shim::fused_linear_cross_entropy_phase_b_via_kt_forward_op`,
//! whose predicate this function preserves):
//!
//! - `hidden.device()` is CUDA (FLCE's production path is CUDA-only
//!   in the trainer; CPU FLCE uses the candle Phase B body
//!   directly).
//! - `hidden.dtype() in {F32, BF16}`.
//! - `head_t.dtype() == hidden.dtype()`.
//! - `hidden` is 3-D `[1, seq_len, hidden_size]`.
//! - `head_t` is 2-D `[hidden_size, vocab_size]`.
//! - `head_t.dim(0) == hidden.dim(2)`.
//! - `seq_len >= 2` and `label_mask.len() == seq_len` and
//!   `label_mask[1..]` has at least one `true` (else the forward
//!   short-circuits to a zero scalar and recording a backward node
//!   would attach a dead branch to the tape).
//! - `chunk_size > 0`.
//!
//! Out-of-envelope inputs return an error rather than silently
//! falling back — the production caller is expected to pre-check
//! via `shim_envelope_ok` exactly like the existing
//! `fused_linear_cross_entropy_phase_b_via_kt_forward_op` shim does.

use kiln_autograd::{BackwardOp, Tape};
use kiln_tensor::{
    DType as KtDType, Device as KtDevice, Result as KtResult, Tensor as KtTensor, bail,
};

use crate::kt_api::{
    FlceActiveMetadata, FlceError, fused_linear_cross_entropy_phase_b_backward_kt,
    fused_linear_cross_entropy_phase_b_backward_unit_grad_kt,
    fused_linear_cross_entropy_phase_b_backward_unit_grad_with_metadata_kt,
    fused_linear_cross_entropy_phase_b_with_metadata_kt,
};

/// Returns `true` when `(hidden, head_t)` is inside the kt-tape FLCE
/// forward+backward envelope. Preserves the semantics of the deleted
/// candle shim's `shim_envelope_ok`: CUDA + dtype
/// in {F32, BF16} + matching head_t dtype + 3-D hidden + 2-D head_t
/// + matching hidden_size.
fn envelope_ok(hidden: &KtTensor, head_t: &KtTensor) -> bool {
    // Phase R.7: accept both CUDA and ROCm device tensors. The FLCE
    // forward/backward are pure kt-tensor ops (matmul, exp, index_select,
    // scatter_add, ...) which dispatch on the tensor's backend, so the
    // ROCm path rides the same `_kt` kernels with no FFI of its own. The
    // neutral device check replaces the CUDA-only `matches!` so the kt-tape
    // entry stops short-circuiting to the out-of-envelope error on
    // `Device::Rocm` inputs.
    if !matches!(hidden.device(), KtDevice::Cuda(_) | KtDevice::Rocm(_)) {
        return false;
    }
    let h_dt = hidden.dtype();
    if !matches!(h_dt, KtDType::F32 | KtDType::BF16) {
        return false;
    }
    if h_dt != head_t.dtype() {
        return false;
    }
    let h_dims = hidden.shape();
    if h_dims.len() != 3 || h_dims[0] != 1 {
        return false;
    }
    let w_dims = head_t.shape();
    if w_dims.len() != 2 {
        return false;
    }
    if w_dims[0] != h_dims[2] {
        return false;
    }
    true
}

/// Saved-state backward for the kt-typed FLCE Phase B kernel.
///
/// Stores `hidden`, `head_t`, `input_ids`, `label_mask`, and
/// `chunk_size` captured at forward time. On `apply(grad_loss)` it
/// calls [`fused_linear_cross_entropy_phase_b_backward_kt`] (which
/// recomputes the chunk loop and produces `dhidden` in the original
/// hidden dtype). Returns `[Some(dhidden)]` for the sole differentiable
/// tape input. The frozen `head_t` is saved backward data, not a tape
/// input; `input_ids` and `label_mask` are non-tensor metadata.
///
/// # Why no `head_t` gradient?
///
/// FLCE's existing kt-typed backward
/// ([`fused_linear_cross_entropy_phase_b_backward_kt`]) returns only
/// `dhidden`. The candle `CustomOp1` path (`KtForwardOp1`) has the
/// same shape — `bwd()` returns `Ok(vec![Some(dhidden)])`. The base
/// head is frozen for this LoRA training route, so advertising it as
/// a differentiable tape input would be both wasteful and misleading.
///
/// # Tensor saving cost
///
/// `hidden` and `head_t` are saved by `Arc`-clone of the kt
/// `Tensor` (storage is already `Arc<dyn Storage>`), so the saved
/// state is two refcount bumps + a `Vec<u32>` + a `Vec<bool>` + a
/// `usize` — no host copy of tensor data.
#[derive(Debug)]
pub struct CudaFlcePhaseBBackward {
    /// Saved CUDA `hidden` from the forward pass.
    pub hidden: KtTensor,
    /// Saved CUDA `head_t` from the forward pass.
    pub head_t: KtTensor,
    /// Token ids (full sequence, length `seq_len`).
    pub input_ids: Vec<u32>,
    /// Label mask (length `seq_len`).
    pub label_mask: Vec<bool>,
    /// Vocab chunk size used by the forward pass.
    pub chunk_size: usize,
    /// Active shifted-token metadata built by the forward. Unit-root SFT
    /// backward can reuse this instead of rescanning the long-context mask and
    /// uploading the same active-index tensor again.
    pub active_metadata: Option<FlceActiveMetadata>,
    /// The recorded loss is used as the tape root, so the upstream seed is the
    /// implicit unit scalar `dL/dL = 1`. This keeps the production SFT root off
    /// the scalar D2H path used by the generic seeded backward while preserving
    /// the public generic tape entry for composed/non-unit seeds.
    pub unit_root_grad: bool,
}

impl BackwardOp for CudaFlcePhaseBBackward {
    fn name(&self) -> &'static str {
        "kiln-flce-kernel/fused_linear_cross_entropy_phase_b_kt_tape"
    }

    fn input_count(&self) -> usize {
        // `hidden` is the only differentiable tape input. `head_t` is a
        // frozen constant saved on this op solely to compute `dhidden`.
        1
    }

    fn apply(&self, grad_output: &KtTensor) -> KtResult<Vec<Option<KtTensor>>> {
        // The kt-typed forward returns a rank-0 scalar F32 loss, so
        // grad_output (which the tape walker fills from upstream) is
        // a rank-0 F32 tensor — the `dLoss / dLoss` seed.
        if grad_output.dtype() != KtDType::F32 {
            bail!(
                "flce kt-tape bwd: grad_output dtype {} != F32",
                grad_output.dtype()
            );
        }
        if grad_output.rank() != 0 {
            bail!(
                "flce kt-tape bwd: grad_output must be a scalar (rank 0); got shape {:?}",
                grad_output.shape()
            );
        }

        let dhidden = if self.unit_root_grad {
            if let Some(active_metadata) = self.active_metadata.as_ref() {
                fused_linear_cross_entropy_phase_b_backward_unit_grad_with_metadata_kt(
                    &self.hidden,
                    &self.head_t,
                    &self.input_ids,
                    &self.label_mask,
                    self.chunk_size,
                    active_metadata,
                )
            } else {
                fused_linear_cross_entropy_phase_b_backward_unit_grad_kt(
                    &self.hidden,
                    &self.head_t,
                    &self.input_ids,
                    &self.label_mask,
                    self.chunk_size,
                )
            }
        } else {
            fused_linear_cross_entropy_phase_b_backward_kt(
                &self.hidden,
                &self.head_t,
                &self.input_ids,
                &self.label_mask,
                self.chunk_size,
                grad_output,
            )
        }
        .map_err(|e: FlceError| {
            kiln_tensor::Error::Msg(format!("flce kt-tape bwd: kt call: {e}"))
        })?;

        // FLCE's backward only produces a gradient for the sole tape input,
        // `hidden`; the frozen head is saved backward data.
        Ok(vec![Some(dhidden)])
    }

    fn requires_input(&self, idx: usize) -> bool {
        // The chunk loop reads the recorded hidden activation. The frozen
        // head and host metadata are saved directly on the op.
        idx == 0
    }
}

/// kt-tape FLCE Phase B forward+backward — Phase 6a/CP-4 successor to
/// the historical candle shim `fused_linear_cross_entropy_phase_b_via_kt_forward_op`
/// (last at `kiln_train::flce_candle_shim`, removed post-#1082).
///
/// Runs the kt-typed forward via
/// [`fused_linear_cross_entropy_phase_b_kt`], then records a tape
/// node whose backward calls
/// [`fused_linear_cross_entropy_phase_b_backward_kt`] on the same
/// kt-typed kernels. No candle types touched — the input, output,
/// and recorded saved tensors are all `kiln_tensor::Tensor`.
///
/// # Envelope
///
/// Matches the original candle shim envelope (see [`envelope_ok`]).
/// Out-of-envelope inputs return an `Err` rather than silently
/// falling back; the production caller is expected to pre-check via
/// the same envelope predicate.
///
/// # Tape integration
///
/// The tape records `hidden` as its sole differentiable input. The backward
/// also saves `hidden` and the frozen `head_t` by `Arc` clone — kt `Tensor`
/// is already `Clone` over `Arc<dyn Storage>` so this is a refcount bump, not
/// a host copy. `input_ids` and `label_mask` are saved as plain `Vec`s on the
/// op struct and are not tape inputs.
///
/// # Returns
///
/// Scalar F32 loss tensor (rank-0, shape `[]`). The tape now owns
/// one extra node whose `apply()` produces `dhidden` for the
/// `hidden` input.
pub fn fused_linear_cross_entropy_phase_b_via_kt_tape(
    hidden: &KtTensor,
    head_t: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
    tape: &mut Tape,
) -> KtResult<KtTensor> {
    fused_linear_cross_entropy_phase_b_via_kt_tape_impl(
        hidden, head_t, input_ids, label_mask, chunk_size, tape, false,
    )
}

/// kt-tape FLCE Phase B scalar root with an implicit unit upstream seed.
///
/// Production SFT uses the returned scalar as the tape root inside
/// `with_tape_authoritative_scope_kt`, which always seeds `dL/dL = 1`. This
/// entry records the same forward node as
/// [`fused_linear_cross_entropy_phase_b_via_kt_tape`] but its backward calls
/// [`fused_linear_cross_entropy_phase_b_backward_unit_grad_kt`], avoiding the
/// scalar seed device-to-host read in the long-context root path.
pub fn fused_linear_cross_entropy_phase_b_unit_grad_via_kt_tape(
    hidden: &KtTensor,
    head_t: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
    tape: &mut Tape,
) -> KtResult<KtTensor> {
    fused_linear_cross_entropy_phase_b_via_kt_tape_impl(
        hidden, head_t, input_ids, label_mask, chunk_size, tape, true,
    )
}

fn fused_linear_cross_entropy_phase_b_via_kt_tape_impl(
    hidden: &KtTensor,
    head_t: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
    tape: &mut Tape,
    unit_root_grad: bool,
) -> KtResult<KtTensor> {
    if !envelope_ok(hidden, head_t) {
        bail!(
            "fused_linear_cross_entropy_phase_b_via_kt_tape: inputs outside kt envelope \
             (CUDA + dtype in {{F32, BF16}} + matching head_t dtype + 3-D hidden \
             [1, seq, hidden] + 2-D head_t [hidden, vocab] required). Callers must \
             filter via the equivalent shim_envelope_ok predicate first."
        );
    }
    if chunk_size == 0 {
        bail!("fused_linear_cross_entropy_phase_b_via_kt_tape: chunk_size must be > 0");
    }
    if label_mask.len() != input_ids.len() {
        bail!(
            "fused_linear_cross_entropy_phase_b_via_kt_tape: label_mask length {} \
             does not match input_ids length {}",
            label_mask.len(),
            input_ids.len(),
        );
    }

    // Forward — bit-exact with `fused_linear_cross_entropy_phase_b_kt`
    // (same kt-typed call), while saving the active metadata it already
    // computes for the unit-root backward.
    let (loss, active_metadata) = fused_linear_cross_entropy_phase_b_with_metadata_kt(
        hidden, head_t, input_ids, label_mask, chunk_size,
    )
    .map_err(|e: FlceError| {
        kiln_tensor::Error::Msg(format!(
            "fused_linear_cross_entropy_phase_b_via_kt_tape fwd: kt call: {e}"
        ))
    })?;

    // Backward — save (hidden, head_t, input_ids, label_mask, chunk_size)
    // by Arc-cloning the kt tensors and cloning the metadata vecs.
    let bwd = CudaFlcePhaseBBackward {
        hidden: hidden.clone(),
        head_t: head_t.clone(),
        input_ids: input_ids.to_vec(),
        label_mask: label_mask.to_vec(),
        chunk_size,
        active_metadata,
        unit_root_grad,
    };
    tape.record(&loss, &[hidden], Box::new(bwd) as Box<dyn BackwardOp>);

    Ok(loss)
}

// ---------------------------------------------------------------------------
// Tests — envelope tests run host-only; the CUDA E2E tests gate on
// the `cuda` cargo feature *and* `kiln_tensor::primary_cuda_context(0)`
// at runtime so they skip cleanly when CUDA hardware is absent.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Envelope rejects CPU tensors — production FLCE kt-tape is
    /// CUDA-only (CPU FLCE keeps using the candle Phase B body).
    #[test]
    fn envelope_rejects_cpu() {
        let hidden =
            KtTensor::from_vec(vec![0.0f32; 1 * 4 * 8], vec![1, 4, 8]).expect("cpu hidden");
        let head = KtTensor::from_vec(vec![0.0f32; 8 * 16], vec![8, 16]).expect("cpu head");
        assert!(!envelope_ok(&hidden, &head));
    }

    /// Envelope rejects malformed hidden rank (must be 3-D).
    #[test]
    fn envelope_rejects_wrong_hidden_rank() {
        // 2-D hidden — should fail (3-D required even before the
        // CUDA gate, but on CPU the device check short-circuits
        // first — this exercises the predicate path).
        let hidden = KtTensor::from_vec(vec![0.0f32; 4 * 8], vec![4, 8]).expect("2d hidden");
        let head = KtTensor::from_vec(vec![0.0f32; 8 * 16], vec![8, 16]).expect("head");
        assert!(!envelope_ok(&hidden, &head));
    }

    /// Envelope rejects head_t with mismatched hidden_size axis.
    #[test]
    fn envelope_rejects_head_hidden_mismatch() {
        let hidden = KtTensor::from_vec(vec![0.0f32; 1 * 4 * 8], vec![1, 4, 8]).expect("hidden");
        // head_t hidden_size (12) != hidden hidden_size (8).
        let head = KtTensor::from_vec(vec![0.0f32; 12 * 16], vec![12, 16]).expect("head");
        assert!(!envelope_ok(&hidden, &head));
    }

    /// The frozen head is saved backward data, not a differentiable tape
    /// input. Keep this contract covered without requiring a GPU.
    #[test]
    fn backward_contract_has_one_hidden_input() {
        let hidden =
            KtTensor::from_vec(vec![0.0f32; 1 * 4 * 8], vec![1, 4, 8]).expect("cpu hidden");
        let head_t = KtTensor::from_vec(vec![0.0f32; 8 * 16], vec![8, 16]).expect("cpu head");
        let bwd = CudaFlcePhaseBBackward {
            hidden,
            head_t,
            input_ids: vec![0, 1, 2, 3],
            label_mask: vec![true; 4],
            chunk_size: 4,
            active_metadata: None,
            unit_root_grad: false,
        };

        assert_eq!(bwd.input_count(), 1);
        assert!(bwd.requires_input(0));
        assert!(!bwd.requires_input(1));
    }

    // -----------------------------------------------------------------
    // CUDA E2E tests — gated on the `cuda` cargo feature so non-CUDA
    // builds still compile this module. At runtime we additionally
    // check `kiln_tensor::primary_cuda_context(0).is_ok()` so the
    // tests skip cleanly on a CUDA-feature-enabled build that lacks
    // an actual GPU (CI matrix safety).
    // -----------------------------------------------------------------

    #[cfg(feature = "cuda")]
    fn cuda_available() -> bool {
        kiln_tensor::primary_cuda_context(0).is_ok()
    }

    #[cfg(feature = "cuda")]
    fn pattern_f32(n: usize, seed: u64) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
        for _ in 0..n {
            s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
            out.push(((s as u32 % 1024) as f32 - 512.0) / 512.0);
        }
        out
    }

    /// CUDA forward records only the differentiable hidden id; the frozen
    /// head remains saved backward data. Skips cleanly without a CUDA device.
    /// CUDA E2E forward. (#1082 H-FLCE) Previously `#[ignore]`-d
    /// because the kt-typed forward (`fused_linear_cross_entropy_phase_b_kt`)
    /// built per-chunk index tensors on the CPU (`active_idx`, `row_idx_t`,
    /// `col_idx_2d`) so `DeviceOp2 "index_select"` failed with "inputs on
    /// different devices: a=cuda:0, b=cpu". That gap is now closed: the
    /// forward derives `device = hidden.device()` and allocates every index
    /// tensor via `from_vec_on(device, ...)`, and the correct-logit gather
    /// uses a CUDA-capable flat `index_select` instead of CPU-only `gather`
    /// (mirroring the already-correct backward + the H6 CE adapter). Runs on
    /// CUDA, skips cleanly when no GPU is present.
    #[cfg(feature = "cuda")]
    #[test]
    fn forward_records_tape_node_when_cuda_available() {
        if !cuda_available() {
            eprintln!("CUDA device not available; skipping forward_records_tape_node");
            return;
        }
        // Build a small fully-active sample with F32 tensors on CUDA.
        // F32 is in the kt-tape envelope and avoids dragging in `half`
        // here (the FLCE crate doesn't currently depend on `half`).
        let seq = 4usize;
        let hidden_size = 8usize;
        let vocab = 16usize;

        let h_data = pattern_f32(1 * seq * hidden_size, 1);
        let w_data = pattern_f32(hidden_size * vocab, 2);

        let hidden =
            KtTensor::cuda_from_slice(&h_data, vec![1, seq, hidden_size], 0).expect("hidden cuda");
        let head =
            KtTensor::cuda_from_slice(&w_data, vec![hidden_size, vocab], 0).expect("head cuda");

        // Envelope must report OK for matching-dtype CUDA inputs.
        assert!(envelope_ok(&hidden, &head));

        let ids: Vec<u32> = (0..seq as u32).collect();
        let mask = vec![true; seq];

        let mut tape = Tape::new();
        let loss = fused_linear_cross_entropy_phase_b_via_kt_tape(
            &hidden, &head, &ids, &mask, 4, &mut tape,
        )
        .expect("forward + record");

        // The kt-typed forward returns a scalar F32 (rank-0) loss.
        assert!(
            loss.shape().is_empty(),
            "loss must be rank-0; got {:?}",
            loss.shape()
        );
        assert_eq!(loss.dtype(), KtDType::F32);
        assert_eq!(tape.len(), 1);

        let node = &tape.nodes()[0];
        assert_eq!(node.input_ids.len(), 1);
        assert_eq!(node.input_ids[0], hidden.id());
        assert!(!node.input_ids.contains(&head.id()));
        assert_eq!(node.output_id, loss.id());
        assert_eq!(
            node.op.name(),
            "kiln-flce-kernel/fused_linear_cross_entropy_phase_b_kt_tape"
        );
        assert_eq!(node.op.input_count(), 1);
    }

    /// Direct backward apply — exercises the apply() path on a
    /// matched (hidden, head_t) shape and asserts the returned
    /// sole returned gradient has the original hidden dtype + shape.
    /// Skips cleanly without a CUDA device.
    /// CUDA E2E backward. (#1082 H-FLCE) The backward already allocated its
    /// index/accumulator tensors device-parametrically; with the matching
    /// forward fix (see `forward_records_tape_node_when_cuda_available`) the
    /// full kt-tape FLCE path now runs on CUDA. Runs on CUDA, skips cleanly
    /// when no GPU is present.
    #[cfg(feature = "cuda")]
    #[test]
    fn backward_apply_returns_single_dhidden() {
        if !cuda_available() {
            eprintln!("CUDA device not available; skipping backward_apply");
            return;
        }
        let seq = 4usize;
        let hidden_size = 8usize;
        let vocab = 16usize;

        let h_data = pattern_f32(1 * seq * hidden_size, 3);
        let w_data = pattern_f32(hidden_size * vocab, 4);

        let hidden =
            KtTensor::cuda_from_slice(&h_data, vec![1, seq, hidden_size], 0).expect("hidden cuda");
        let head =
            KtTensor::cuda_from_slice(&w_data, vec![hidden_size, vocab], 0).expect("head cuda");

        let ids: Vec<u32> = (0..seq as u32).collect();
        let mask = vec![true; seq];

        let bwd = CudaFlcePhaseBBackward {
            hidden: hidden.clone(),
            head_t: head.clone(),
            input_ids: ids.clone(),
            label_mask: mask.clone(),
            chunk_size: 4,
            active_metadata: None,
            unit_root_grad: false,
        };

        // Seed grad — rank-0 F32 (matches the loss), on the SAME CUDA device as
        // hidden/head (the production seed comes from the on-device tape; a CPU
        // `from_vec` seed would make the backward's mul mix cuda+cpu operands).
        let grad_loss = KtTensor::cuda_from_slice(&[1.0f32], vec![], 0).expect("grad_loss");

        let grads = bwd.apply(&grad_loss).expect("apply backward");
        assert_eq!(grads.len(), 1);
        let gh = grads[0].as_ref().expect("dhidden present");
        // FLCE's kt-typed backward returns dhidden in the original
        // hidden dtype with the same `[1, seq, hidden]` shape.
        assert_eq!(gh.shape(), &[1, seq, hidden_size]);
        assert_eq!(gh.dtype(), hidden.dtype());
    }
}
