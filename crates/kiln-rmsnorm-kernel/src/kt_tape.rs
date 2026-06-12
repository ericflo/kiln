//! kt-tape RMSNorm forward+backward — Phase 6a/CP-4 pilot port of
//! `kt_forward_op.rs` from `candle::CustomOp2` onto the kt-side
//! `kiln_autograd::Tape` substrate ((#1082) — see
//! `docs/CANDLE_REMOVAL_PLAN.md`).
//!
//! # Why this module exists
//!
//! The candle-autograd shim `kiln-model::rmsnorm_candle_shim::fused_rmsnorm_via_kt_forward_op`
//! (relocated out of this crate in (#1082)) wraps the CUDA fused RMSNorm
//! forward + backward inside a candle `CustomOp2` (`KtForwardOp2`). That
//! shim is how the candle-autograd path reaches these kernels — even
//! though both halves of the autograd roundtrip already bottom out in
//! `kiln_tensor::Tensor` + the kt-typed `fused_rmsnorm_kt` /
//! `fused_rmsnorm_backward_kt` entries.
//!
//! This module is the parallel entry that drops the candle CustomOp
//! wrapper and records the backward directly onto a
//! `kiln_autograd::Tape`. Same FFI symbols (`kiln_fused_rmsnorm`,
//! `kiln_fused_rmsnorm_bwd`, `kiln_f32_to_bf16`), same envelope, same
//! numerical contract. The only difference is who owns the autograd
//! recording: candle's `BackpropOp` chain (legacy) vs. kiln's
//! `Tape::record` (new).
//!
//! # Numerical contract
//!
//! Forward: bit-exact equality with [`crate::kt_api::fused_rmsnorm_kt`]
//! (they call the same FFI symbol on the same input bytes).
//!
//! Backward: `grad_x` bit-exact across calls (same FFI symbol
//! `kiln_fused_rmsnorm_bwd`, no cross-row reduction). `grad_w` may
//! differ by up to one BF16 ULP across calls because the kernel's
//! `atomicAdd` cross-row reduction is order-non-deterministic across
//! separate launches.
//!
//! # Envelope
//!
//! Same as the CUDA-side [`crate::supports_rmsnorm_kt`]: CUDA, BF16
//! `x` + `weight`, contiguous, rank >= 1, weight shape == `[hidden]`,
//! `hidden <= 8192`. Out-of-envelope inputs return an error rather
//! than silently falling back — the production caller is expected to
//! pre-check via `supports_rmsnorm_kt` exactly like the existing
//! `fused_rmsnorm_via_kt_forward_op` shim does.

use kiln_autograd::{BackwardOp, Tape};
use kiln_tensor::{
    DType as KtDType, Device as KtDevice, Result as KtResult, Tensor as KtTensor, bail,
};

use crate::kt_api::{RmsNormError, fused_rmsnorm_backward_kt, fused_rmsnorm_kt};

/// True when `t` lives on a GPU backend the fused RMSNorm kernel runs on.
/// Phase R.7: accepts `Device::Rocm` (under the `rocm` feature) as well as
/// `Device::Cuda`; CUDA-only behavior is unchanged when `rocm` is off.
fn is_gpu(t: &KtTensor) -> bool {
    match t.device() {
        KtDevice::Cuda(_) => true,
        #[cfg(feature = "rocm")]
        KtDevice::Rocm(_) => true,
        _ => false,
    }
}

/// Returns `true` when `(x, weight)` is inside the kt-tape
/// forward+backward envelope. Matches [`crate::supports_rmsnorm_kt`]
/// exactly (GPU + BF16 + contiguous + rank >= 1 + weight == [hidden] +
/// hidden <= 8192).
fn envelope_ok(x: &KtTensor, weight: &KtTensor) -> bool {
    if !is_gpu(x) {
        return false;
    }
    if !is_gpu(weight) {
        return false;
    }
    if x.dtype() != KtDType::BF16 || weight.dtype() != KtDType::BF16 {
        return false;
    }
    if !x.is_contiguous() || !weight.is_contiguous() {
        return false;
    }
    if x.rank() < 1 {
        return false;
    }
    let hidden = x.shape().last().copied().unwrap_or(0);
    if hidden == 0 || hidden > 8192 {
        return false;
    }
    if weight.shape() != [hidden] {
        return false;
    }
    true
}

/// Saved-state backward for the fused CUDA RMSNorm kernel.
///
/// Stores `x`, `weight`, and `eps` captured at forward time. On
/// `apply(grad_y)` it calls [`fused_rmsnorm_backward_kt`] (which
/// produces `grad_x: BF16, grad_w_partial: F32 [rows, hidden]`),
/// then casts the first `hidden` F32 slots of the partial buffer
/// down to BF16 to form the final `grad_w`. The cast logic mirrors
/// the existing candle-wrapped backward closure exactly — same FFI
/// symbol (`kiln_f32_to_bf16`), same `hidden`-only cast count.
///
/// # Why F32 [rows, hidden] partial -> BF16 [hidden] cast?
///
/// The kernel writes the cross-row `atomicAdd` accumulation into
/// the *first* `hidden` F32 slots of the partial buffer (the rest
/// of the `rows * hidden` allocation is scratch the kernel doesn't
/// touch). See `csrc/fused_rmsnorm_bwd.cu` lines 12-19/122-123.
/// We cast only those populated `hidden` slots to BF16, producing
/// the final `grad_w` tensor.
#[derive(Debug)]
pub struct CudaFusedRmsNormBackward {
    /// Saved BF16 CUDA `x` from the forward pass.
    pub x: KtTensor,
    /// Saved BF16 CUDA `weight` from the forward pass.
    pub weight: KtTensor,
    /// Epsilon used in the forward pass.
    pub eps: f32,
}

impl BackwardOp for CudaFusedRmsNormBackward {
    fn name(&self) -> &'static str {
        "kiln-rmsnorm-kernel/fused_rmsnorm_kt_tape"
    }

    fn input_count(&self) -> usize {
        2
    }

    fn apply(&self, grad_output: &KtTensor) -> KtResult<Vec<Option<KtTensor>>> {
        // Same shape + dtype + device checks as the candle-side
        // backward closure. We bail with a kt `Error::Msg` rather
        // than candle's `bail!` since the kt-tape callers don't see
        // candle errors.
        if grad_output.shape() != self.x.shape() {
            bail!(
                "rmsnorm kt-tape bwd: grad_output {:?} != x {:?}",
                grad_output.shape(),
                self.x.shape()
            );
        }
        if !is_gpu(grad_output) {
            bail!("rmsnorm kt-tape bwd: grad_output must be on a GPU backend");
        }
        if grad_output.dtype() != KtDType::BF16 {
            bail!(
                "rmsnorm kt-tape bwd: grad_output dtype {} != BF16",
                grad_output.dtype()
            );
        }
        if !grad_output.is_contiguous() {
            bail!("rmsnorm kt-tape bwd: grad_output must be contiguous");
        }

        let hidden = self
            .x
            .shape()
            .last()
            .copied()
            .ok_or_else(|| kiln_tensor::Error::from_str("rmsnorm kt-tape bwd: x rank < 1"))?;

        // Run the CUDA backward kernel: produces (grad_x: BF16, grad_w_partial: F32).
        let (grad_x, grad_w_partial) =
            fused_rmsnorm_backward_kt(&self.x, &self.weight, grad_output, self.eps).map_err(
                |e: RmsNormError| {
                    kiln_tensor::Error::Msg(format!("rmsnorm kt-tape bwd: kt call: {e}"))
                },
            )?;

        // Cast the first `hidden` F32 slots of the partial buffer to
        // BF16 to produce the final `grad_w`. See module docs and the
        // candle backward closure in `kt_forward_op.rs` for why this
        // slice is valid (kernel writes the reduced result into the
        // first `hidden` F32 slots).
        let grad_w = cast_partial_hidden_f32_to_bf16(&grad_w_partial, hidden)?;

        Ok(vec![Some(grad_x), Some(grad_w)])
    }

    fn requires_input(&self, idx: usize) -> bool {
        // Both x and weight are needed by the backward kernel.
        idx == 0 || idx == 1
    }
}

/// Cast the first `hidden` F32 slots of `partial` to BF16, returning a
/// fresh `[hidden]` BF16 CUDA tensor on the same stream as `partial`.
///
/// Mirrors the cast block in the candle backward closure of
/// [`crate::kt_forward_op`]. Lives here so the kt-tape backward
/// doesn't have to pull `kiln_kt_bridge::cuda_*` plumbing into its
/// public surface.
fn cast_partial_hidden_f32_to_bf16(partial: &KtTensor, hidden: usize) -> KtResult<KtTensor> {
    // Backend-neutral seam (Phase R.7): the `device_*` dispatchers route
    // `Device::Cuda` tensors to the same CUDA helpers (behavior-identical) and
    // `Device::Rocm` tensors to the ROCm ones.
    let partial_ptr = kiln_kt_bridge::device_output_ptr(partial);
    let raw_stream = kiln_kt_bridge::device_stream_raw_of(partial, "partial")
        .map_err(|e| kiln_tensor::Error::Msg(format!("rmsnorm kt-tape bwd: stream: {e}")))?;
    let dst = kiln_kt_bridge::alloc_device_tensor_like(partial, KtDType::BF16, vec![hidden])
        .map_err(|e| {
            kiln_tensor::Error::Msg(format!("rmsnorm kt-tape bwd: alloc grad_w BF16: {e}"))
        })?;
    let dst_ptr = kiln_kt_bridge::device_output_ptr(&dst);

    // SAFETY: `partial_ptr` points to a F32 buffer of at least `hidden`
    // populated elements (the kernel writes the reduced result into the
    // first `hidden` F32 slots — see `csrc/fused_rmsnorm_bwd.cu`).
    // `dst_ptr` points to a BF16 buffer of exactly `hidden` elements we
    // just allocated. `raw_stream` is the GPU stream associated with
    // `partial`'s storage.
    let status = unsafe {
        crate::kiln_f32_to_bf16(
            partial_ptr as *const f32,
            dst_ptr as *mut _,
            hidden as i32,
            raw_stream,
        )
    };
    if status != 0 {
        bail!("rmsnorm kt-tape bwd: kiln_f32_to_bf16 failed (status {status})");
    }
    Ok(dst)
}

/// kt-tape fused RMSNorm forward+backward — Phase 6a/CP-4 successor to
/// the candle `KtForwardOp2` shim
/// `kiln-model::rmsnorm_candle_shim::fused_rmsnorm_via_kt_forward_op`.
///
/// Runs the CUDA forward via [`fused_rmsnorm_kt`], then records a tape
/// node whose backward calls [`fused_rmsnorm_backward_kt`] on the same
/// FFI symbols. No candle types touched — the input, output, and
/// recorded saved tensors are all `kiln_tensor::Tensor`.
///
/// # Envelope
///
/// Same as [`crate::supports_rmsnorm_kt`]. Out-of-envelope inputs return
/// an `Err` rather than silently falling back; the production caller
/// is expected to pre-check via `supports_rmsnorm_kt` exactly like the
/// existing `fused_rmsnorm_via_kt_forward_op` shim.
///
/// # Tape integration
///
/// The forward and the backward share `(x, weight)` by `Arc` — kt
/// `Tensor` is already `Clone` over `Arc<dyn Storage>` so the saved
/// state is a refcount bump, not a host copy.
///
/// # Production caller status (2026-05-28)
///
/// The production caller in `kiln_model::forward::rms_norm`
/// (`crates/kiln-model/src/forward.rs:7172`) still routes through
/// `kiln-model::rmsnorm_candle_shim::fused_rmsnorm_via_kt_forward_op`,
/// not this entry. The
/// flip is gated on CP-4 substrate work in `kiln-train` (porting
/// `loss.backward()` / `candle_core::backprop::GradStore` onto
/// `kiln_autograd::Var` / `Tape::backward`). See
/// `docs/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md` for
/// the full audit and the architectural reason a per-call-site flip
/// is not progress until CP-4 lands.
pub fn fused_rmsnorm_via_kt_tape(
    x: &KtTensor,
    weight: &KtTensor,
    eps: f32,
    tape: &mut Tape,
) -> KtResult<KtTensor> {
    if !envelope_ok(x, weight) {
        bail!(
            "fused_rmsnorm_via_kt_tape: inputs outside kt envelope \
             (CUDA + BF16 + contiguous + hidden <= 8192 required). \
             Callers must filter via `supports_rmsnorm_kt(x, weight)` first."
        );
    }

    // Forward — bit-exact with `fused_rmsnorm_kt` (same FFI call).
    let y = fused_rmsnorm_kt(x, weight, eps).map_err(|e: RmsNormError| {
        kiln_tensor::Error::Msg(format!("fused_rmsnorm_via_kt_tape fwd: kt call: {e}"))
    })?;

    // Backward — save (x, weight, eps) by Arc-cloning the kt tensors.
    let bwd = CudaFusedRmsNormBackward {
        x: x.clone(),
        weight: weight.clone(),
        eps,
    };
    tape.record(&y, &[x, weight], Box::new(bwd) as Box<dyn BackwardOp>);

    Ok(y)
}

// ---------------------------------------------------------------------------
// Tests — gated on CUDA availability at runtime (skip cleanly when no CUDA).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // The CUDA-specific test helpers (`cuda_from_slice`, `primary_cuda_context`)
    // only exist under the `cuda` feature; gate them so a `rocm`-only build of
    // the crate's unit tests still compiles. The CPU-only envelope tests below
    // need none of this.
    #[cfg(feature = "cuda")]
    use half::bf16;

    #[cfg(feature = "cuda")]
    fn cuda_available() -> bool {
        kiln_tensor::primary_cuda_context(0).is_ok()
    }

    #[cfg(feature = "cuda")]
    fn pattern_bf16(n: usize, seed: u64) -> Vec<bf16> {
        let mut out = Vec::with_capacity(n);
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
        for _ in 0..n {
            s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
            out.push(bf16::from_f32(((s as u32 % 1024) as f32 - 512.0) / 512.0));
        }
        out
    }

    #[test]
    fn envelope_rejects_cpu() {
        let x = KtTensor::zeros_cpu(vec![1, 16], KtDType::BF16);
        let w = KtTensor::zeros_cpu(vec![16], KtDType::BF16);
        assert!(!envelope_ok(&x, &w));
    }

    #[test]
    fn envelope_rejects_oversized_hidden() {
        // hidden = 16384 > 8192 — should fail the envelope. (CUDA-free.)
        let x = KtTensor::zeros_cpu(vec![1, 16384], KtDType::BF16);
        let w = KtTensor::zeros_cpu(vec![16384], KtDType::BF16);
        assert!(!envelope_ok(&x, &w));
    }

    #[test]
    fn envelope_rejects_weight_mismatch() {
        let x = KtTensor::zeros_cpu(vec![1, 16], KtDType::BF16);
        let w = KtTensor::zeros_cpu(vec![8], KtDType::BF16);
        assert!(!envelope_ok(&x, &w));
    }

    /// CUDA forward records a tape node tagged with the saved (x, w) ids.
    /// Skips cleanly without CUDA.
    #[cfg(feature = "cuda")]
    #[test]
    fn forward_records_tape_node_when_cuda_available() {
        if !cuda_available() {
            eprintln!("CUDA not available; skipping forward_records_tape_node");
            return;
        }

        let rows = 2usize;
        let hidden = 16usize;
        let x = KtTensor::cuda_from_slice(&pattern_bf16(rows * hidden, 1), vec![rows, hidden], 0)
            .expect("x cuda");
        let w =
            KtTensor::cuda_from_slice(&pattern_bf16(hidden, 2), vec![hidden], 0).expect("w cuda");

        // Envelope must report OK for real CUDA BF16 inputs.
        assert!(envelope_ok(&x, &w));

        let mut tape = Tape::new();
        let y = fused_rmsnorm_via_kt_tape(&x, &w, 1e-6, &mut tape).expect("forward + record");
        assert_eq!(y.shape(), &[rows, hidden]);
        assert_eq!(y.dtype(), KtDType::BF16);
        assert_eq!(tape.len(), 1);

        let node = &tape.nodes()[0];
        assert_eq!(node.input_ids.len(), 2);
        assert_eq!(node.input_ids[0], x.id());
        assert_eq!(node.input_ids[1], w.id());
        assert_eq!(node.output_id, y.id());
        assert_eq!(node.op.name(), "kiln-rmsnorm-kernel/fused_rmsnorm_kt_tape");
        assert_eq!(node.op.input_count(), 2);
    }

    /// Direct backward apply — exercises the apply() path including the
    /// F32 -> BF16 cast of the partial buffer. Skips cleanly without CUDA.
    #[cfg(feature = "cuda")]
    #[test]
    fn backward_apply_returns_grads_of_expected_shape() {
        if !cuda_available() {
            eprintln!("CUDA not available; skipping backward_apply");
            return;
        }

        let rows = 2usize;
        let hidden = 16usize;
        let x = KtTensor::cuda_from_slice(&pattern_bf16(rows * hidden, 3), vec![rows, hidden], 0)
            .expect("x cuda");
        let w =
            KtTensor::cuda_from_slice(&pattern_bf16(hidden, 4), vec![hidden], 0).expect("w cuda");
        let dy = KtTensor::cuda_from_slice(&pattern_bf16(rows * hidden, 5), vec![rows, hidden], 0)
            .expect("dy cuda");

        let bwd = CudaFusedRmsNormBackward {
            x: x.clone(),
            weight: w.clone(),
            eps: 1e-6,
        };
        let grads = bwd.apply(&dy).expect("apply backward");
        assert_eq!(grads.len(), 2);
        let gx = grads[0].as_ref().expect("grad_x present");
        let gw = grads[1].as_ref().expect("grad_w present");
        assert_eq!(gx.shape(), &[rows, hidden]);
        assert_eq!(gx.dtype(), KtDType::BF16);
        assert_eq!(gw.shape(), &[hidden]);
        assert_eq!(gw.dtype(), KtDType::BF16);
    }
}
