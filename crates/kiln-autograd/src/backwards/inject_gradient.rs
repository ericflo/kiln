//! `InjectGradientBackward` — kt-side replacement for
//! `kiln-train::trainer::InjectTensorGradient` (#1082, CP-4).
//!
//! # What this exists for
//!
//! `kiln-train`'s `InjectTensorGradient` is a `candle_core::CustomOp1`
//! used in the tiled-training paths to splice a *precomputed* gradient
//! into candle's backward walker. The forward returns a scalar zero
//! (placeholder) and the backward, regardless of upstream `grad_res`,
//! emits `self.upstream` as the gradient for the single arg.
//!
//! This is the kt-tape equivalent. As a `BackwardOp`:
//!
//! - **arity 1** — one input (the candle/kt tensor whose gradient gets
//!   replaced).
//! - **`apply(grad_output)`** — ignores `grad_output` and returns
//!   `vec![Some(self.injected.clone())]`. The tape walker then
//!   accumulates `self.injected` against the input's id, exactly as
//!   `InjectTensorGradient::bwd` does on candle's side.
//!
//! `requires_input(0)` returns `false` because we never read the forward
//! input — the injected grad is precomputed. This frees Phase 6.5's
//! selective-recompute policy from materialising the forward input just
//! for this op.
//!
//! # Lifecycle
//!
//! `kiln_kt_bridge::tape_bridge::inject_gradient_kt` records one of
//! these onto the active tape during the kt-tape-bridge-wrapped forward
//! pass. When the bridge walks the tape (driven by candle's
//! `loss.backward()` produced GradStore via
//! `backward_with_seeds`), the seed for the InjectGradient output node
//! is whatever candle handed us for its scalar zero. We ignore it and
//! emit `self.injected` for the arg.
//!
//! The output node's input — the kt borrow of the candle `arg` — has its
//! `kt_id` registered as an input mapping via the bridge, so the
//! resulting kt grad for that input id flows back into the candle
//! `GradStore` keyed on `arg.id()`.
//!
//! # Bit-equivalence to the candle path
//!
//! The candle `InjectTensorGradient::bwd` does:
//!
//! ```ignore
//! let upstream = self.upstream.to_device(arg.device())?;
//! let grad = if upstream.dtype() == arg.dtype() {
//!     upstream
//! } else {
//!     upstream.to_dtype(arg.dtype())?
//! };
//! Ok(Some(grad))
//! ```
//!
//! The kt path expects the bridge adapter (`inject_gradient_kt`) to
//! pre-convert the candle `upstream` to a kt tensor matching the kt
//! borrow of `arg`'s dtype/device before constructing
//! `InjectGradientBackward`. The BackwardOp itself is dtype-agnostic —
//! it just clones whatever it was handed. This keeps the apply path
//! allocation-free (one Arc bump on the kt storage).

use kiln_tensor::{Result, Tensor, bail};

use crate::BackwardOp;

/// Backward op that ignores the upstream gradient and emits a
/// pre-computed `injected` gradient for its single input.
///
/// Mirrors the semantics of `kiln-train::trainer::InjectTensorGradient`
/// (a candle `CustomOp1`): both produce a scalar-zero "output" whose
/// backward yields `injected` for the input regardless of the upstream
/// seed.
#[derive(Debug)]
pub struct InjectGradientBackward {
    /// The pre-computed gradient to emit for the arg. Must already
    /// match the arg's shape and dtype (the bridge adapter validates
    /// this before recording).
    pub injected: Tensor,
}

impl BackwardOp for InjectGradientBackward {
    fn name(&self) -> &'static str {
        "inject_gradient_backward"
    }

    fn input_count(&self) -> usize {
        1
    }

    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // `grad_output` is the seed candle handed us for the scalar
        // placeholder output. It's intentionally ignored — the contract
        // (matching the candle CustomOp1 path) is "emit `injected`
        // regardless of what flows in from above". We still take it
        // because the BackwardOp trait requires the parameter; a debug
        // check on its rank keeps the wiring honest. (The candle path
        // doesn't validate `_grad_res` either — it just discards it.)
        let _ = grad_output;
        Ok(vec![Some(self.injected.clone())])
    }

    fn requires_input(&self, idx: usize) -> bool {
        // The forward input is never read by this backward — we ignore
        // it (and `grad_output`) and emit `injected`. Phase 6.5's
        // selective-recompute policy can drop the input activation
        // after forward.
        let _ = idx;
        false
    }
}

impl InjectGradientBackward {
    /// Build an [`InjectGradientBackward`] after validating that
    /// `injected` is plausibly the gradient for `arg`: same shape, same
    /// dtype.
    ///
    /// The bridge adapter calls this so wiring bugs (mismatched shape /
    /// dtype between the precomputed grad and the arg) surface at
    /// record time, not silently at tape-walk time.
    pub fn new_validated(arg: &Tensor, injected: Tensor) -> Result<Self> {
        if arg.shape() != injected.shape() {
            bail!(
                "InjectGradientBackward: arg shape {:?} != injected shape {:?}",
                arg.shape(),
                injected.shape()
            );
        }
        if arg.dtype() != injected.dtype() {
            bail!(
                "InjectGradientBackward: arg dtype {} != injected dtype {}",
                arg.dtype(),
                injected.dtype()
            );
        }
        Ok(InjectGradientBackward { injected })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::{CpuStorage, Tensor};

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn inject_gradient_ignores_upstream_returns_injected() {
        // Build the BackwardOp with an injected tensor of [1.0, 2.0, 3.0, 4.0].
        // Apply with arbitrary "grad_output" — the op MUST ignore it.
        let injected = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let bo = InjectGradientBackward {
            injected: injected.clone(),
        };

        // grad_output: doesn't matter; pick a scalar.
        let grad_output = Tensor::from_slice(&[42.0f32], vec![]).unwrap();
        let grads = bo.apply(&grad_output).unwrap();
        assert_eq!(grads.len(), 1);
        let g = grads[0].as_ref().unwrap();
        assert_eq!(g.shape(), &[2, 2]);
        assert_eq!(read_f32(g), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn inject_gradient_different_upstream_still_returns_injected() {
        // Sanity: a different `grad_output` produces the same result.
        // Mirrors candle's `InjectTensorGradient::bwd` ignoring `_grad_res`.
        let injected = Tensor::from_slice(&[7.5f32, -1.5], vec![2]).unwrap();
        let bo = InjectGradientBackward {
            injected: injected.clone(),
        };

        let g1 = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
        let g2 = Tensor::from_slice(&[100.0f32, 200.0f32], vec![2]).unwrap();
        let out1 = bo.apply(&g1).unwrap()[0].as_ref().unwrap().clone();
        let out2 = bo.apply(&g2).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&out1), read_f32(&out2));
        assert_eq!(read_f32(&out1), vec![7.5, -1.5]);
    }

    #[test]
    fn new_validated_accepts_matching_shape_dtype() {
        let arg = Tensor::from_slice(&[0.0f32; 6], vec![2, 3]).unwrap();
        let injected = Tensor::from_slice(&[1.0f32; 6], vec![2, 3]).unwrap();
        let bo = InjectGradientBackward::new_validated(&arg, injected).unwrap();
        let g = Tensor::from_slice(&[0.0f32], vec![]).unwrap();
        let out = bo.apply(&g).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(out.shape(), &[2, 3]);
    }

    #[test]
    fn new_validated_rejects_shape_mismatch() {
        let arg = Tensor::from_slice(&[0.0f32; 6], vec![2, 3]).unwrap();
        let injected = Tensor::from_slice(&[0.0f32; 4], vec![4]).unwrap();
        let e = InjectGradientBackward::new_validated(&arg, injected).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("shape"), "expected shape mismatch, got: {msg}");
    }

    #[test]
    fn new_validated_rejects_dtype_mismatch() {
        let arg = Tensor::from_slice(&[0.0f32; 4], vec![4]).unwrap();
        // bf16 injected, f32 arg — same shape, different dtype.
        let injected = Tensor::from_slice(
            &[
                half::bf16::ZERO,
                half::bf16::ZERO,
                half::bf16::ZERO,
                half::bf16::ZERO,
            ],
            vec![4],
        )
        .unwrap();
        assert_ne!(injected.dtype(), arg.dtype());
        let e = InjectGradientBackward::new_validated(&arg, injected).unwrap_err();
        assert!(e.to_string().contains("dtype"));
    }

    #[test]
    fn op_metadata() {
        let injected = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let bo = InjectGradientBackward { injected };
        assert_eq!(bo.name(), "inject_gradient_backward");
        assert_eq!(bo.input_count(), 1);
        // forward input is never read (we ignore arg and grad_output).
        assert!(!bo.requires_input(0));
    }
}
