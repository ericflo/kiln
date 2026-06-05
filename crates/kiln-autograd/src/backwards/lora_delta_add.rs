//! `LoraDeltaAddBackward` — gradient of the fused LoRA delta-and-add:
//!
//! ```text
//! out = base + scale * (x @ A^T @ B^T)
//! ```
//!
//! # Why a fused backward
//!
//! The straightforward composition is four kt ops (`matmul`, `matmul`,
//! `mul_scalar`, `add`) plus two `transpose`-followed-by-`contiguous`
//! materialisations on the way in. Each transpose changes the kt
//! `TensorId`, so a composed tape would emit input gradients keyed on
//! the transposed views — shape `[in_features, rank]` and
//! `[rank, out_features]` — not on the original `A: [rank, in_features]`
//! and `B: [out_features, rank]` `Var`s the optimiser iterates.
//!
//! The kt `Tape` ↔ candle `GradStore` bridge in
//! `kiln_kt_bridge::tape_bridge` requires `(kt_input_id, candle_input_id)`
//! pairs to share a shape, so composing through transposes would either
//! leak per-transpose intermediate IDs into the optimiser, or force a new
//! `TransposeBackward` substrate + a tape-recorded `transpose` op (a much
//! bigger surface). A single fused node sidesteps the whole detour: it
//! takes `[base, x, A, B]` in their original layouts and emits per-input
//! gradients in those same layouts, so the adapter just maps each
//! Var-side `candle_id` straight onto the kt input id. This mirrors the
//! `MulSigmoidGateBackward` (swiglu) and `RmsNormBackward` precedents —
//! single fused tape node for a fused forward.
//!
//! # Backward math
//!
//! Decomposing into intermediates:
//!
//! ```text
//! h     = x @ A^T          shape [rows, rank]
//! d     = h @ B^T          shape [rows, out_features]
//! out   = base + scale * d shape [rows, out_features]
//! ```
//!
//! With upstream gradient `grad_out` of shape `[rows, out_features]`:
//!
//! ```text
//! grad_base = grad_out
//! grad_d    = scale * grad_out
//! grad_h    = grad_d @ B               shape [rows, rank]
//! grad_B    = grad_d^T @ h             shape [out_features, rank]
//! grad_x    = grad_h @ A               shape [rows, in_features]
//! grad_A    = grad_h^T @ x             shape [rank, in_features]
//! ```
//!
//! The transposes used inside the backward (`grad_d^T`, `grad_h^T`) are
//! zero-copy views materialised via `.contiguous()` before the matmul,
//! matching the existing `MatmulBackward` reference. `h` is recomputed
//! cheaply from saved `x` and `A` rather than carried forward — at the
//! shapes LoRA exercises (`rank` typically 16..64), the extra matmul is
//! a fraction of a percent of the layer cost.
//!
//! # Saved tensors
//!
//! `x`, `A`, `B` are saved as `kt` `Tensor` clones — an `Arc` bump on
//! the underlying storage handle; no allocation. `base` is NOT saved
//! (the add passes its grad through verbatim).
//!
//! # Input order
//!
//! `LoraDeltaAddBackward::apply` returns four gradients in the order
//! `[base, x, A, B]`, matching the order the adapter records inputs to
//! the tape node. The tape walker pairs each `Some(grad)` with the
//! corresponding `input_ids[i]` it captured at record time.

use kiln_tensor::ops::{matmul, mul_scalar};
use kiln_tensor::{bail, Result, Tensor};

use crate::BackwardOp;

/// Fused backward for `out = base + scale * (x @ A^T @ B^T)`.
///
/// 4 inputs in order `[base, x, A, B]`. `base` is the leading
/// pass-through addend; `x`, `A`, `B` are the LoRA-side activations and
/// factor matrices.
///
/// Shapes (validated at apply time, not record time, so a wiring bug
/// surfaces with a readable message instead of silently producing
/// mis-shaped grads):
///
/// * `x`        — `[rows, in_features]`
/// * `A` (`a`)  — `[rank, in_features]`
/// * `B` (`b`)  — `[out_features, rank]`
/// * grad_out   — `[rows, out_features]`
#[derive(Debug)]
pub struct LoraDeltaAddBackward {
    /// Saved `x` from forward: shape `[rows, in_features]`.
    pub x: Tensor,
    /// Saved LoRA `A`: shape `[rank, in_features]`.
    pub a: Tensor,
    /// Saved LoRA `B`: shape `[out_features, rank]`.
    pub b: Tensor,
    /// LoRA scale (non-differentiable scalar).
    pub scale: f32,
}

impl BackwardOp for LoraDeltaAddBackward {
    fn name(&self) -> &'static str {
        "lora_delta_add_backward"
    }
    fn input_count(&self) -> usize {
        // base, x, A, B.
        4
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let target_device = grad_output.device();
        let x = if self.x.device() == target_device {
            self.x.clone()
        } else {
            self.x.to_device(target_device)?
        };
        let a = if self.a.device() == target_device {
            self.a.clone()
        } else {
            self.a.to_device(target_device)?
        };
        let b = if self.b.device() == target_device {
            self.b.clone()
        } else {
            self.b.to_device(target_device)?
        };

        // ---- shape validation ----------------------------------------
        if x.rank() != 2 || a.rank() != 2 || b.rank() != 2 {
            bail!(
                "LoraDeltaAddBackward: x/A/B must all be rank-2; got \
                 x.rank={}, a.rank={}, b.rank={}",
                x.rank(),
                a.rank(),
                b.rank()
            );
        }
        if grad_output.rank() != 2 {
            bail!(
                "LoraDeltaAddBackward: grad_output must be rank-2 \
                 [rows, out_features]; got rank {}",
                grad_output.rank()
            );
        }
        let (rows, in_features) = (x.shape()[0], x.shape()[1]);
        let (rank, a_in) = (a.shape()[0], a.shape()[1]);
        let (out_features, b_rank) = (b.shape()[0], b.shape()[1]);
        if a_in != in_features {
            bail!(
                "LoraDeltaAddBackward: A.in_features {a_in} != x.in_features {in_features}"
            );
        }
        if b_rank != rank {
            bail!(
                "LoraDeltaAddBackward: B.rank {b_rank} != A.rank {rank}"
            );
        }
        if grad_output.shape()[0] != rows || grad_output.shape()[1] != out_features {
            bail!(
                "LoraDeltaAddBackward: grad_output shape {:?} != \
                 [rows={rows}, out_features={out_features}]",
                grad_output.shape()
            );
        }

        // ---- backward composition ------------------------------------
        //
        // Recompute h = x @ A^T.  (A^T = transpose then contiguous; the
        // transpose is zero-copy.)
        let a_t = a.transpose(0, 1)?.contiguous()?; // [in_features, rank]
        let h = matmul(&x, &a_t)?; // [rows, rank]

        // grad_d = scale * grad_out.  Single elementwise pass.
        let g_scaled = mul_scalar(grad_output, self.scale)?; // [rows, out_features]

        // grad_h = grad_d @ B  (B is [out_features, rank]).
        let grad_h = matmul(&g_scaled, &b)?; // [rows, rank]

        // grad_x = grad_h @ A  (A is [rank, in_features]).
        let grad_x = matmul(&grad_h, &a)?; // [rows, in_features]

        // grad_A = grad_h^T @ x  (shape [rank, in_features]).
        let grad_h_t = grad_h.transpose(0, 1)?.contiguous()?; // [rank, rows]
        let grad_a = matmul(&grad_h_t, &x)?; // [rank, in_features]

        // grad_B = grad_d^T @ h  (shape [out_features, rank]).
        let g_scaled_t = g_scaled.transpose(0, 1)?.contiguous()?; // [out_features, rows]
        let grad_b = matmul(&g_scaled_t, &h)?; // [out_features, rank]

        // grad_base = grad_out  (the additive identity passes the
        // upstream grad straight through with no allocation).
        Ok(vec![
            Some(grad_output.clone()),
            Some(grad_x),
            Some(grad_a),
            Some(grad_b),
        ])
    }
    fn requires_input(&self, idx: usize) -> bool {
        // `base` (idx 0) is not read — the add passes its grad through.
        // x, A, B (idx 1..=3) are saved in `self` and read by `apply`.
        idx != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::ops::{add, matmul as kt_matmul, mul_scalar as kt_mul_scalar};
    use kiln_tensor::CpuStorage;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn t2(data: &[f32], shape: (usize, usize)) -> Tensor {
        Tensor::from_slice(data, vec![shape.0, shape.1]).unwrap()
    }

    #[test]
    fn op_metadata() {
        let one = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let bo = LoraDeltaAddBackward {
            x: one.clone(),
            a: one.clone(),
            b: one,
            scale: 1.0,
        };
        assert_eq!(bo.name(), "lora_delta_add_backward");
        assert_eq!(bo.input_count(), 4);
        // base is pass-through, x/A/B are saved.
        assert!(!bo.requires_input(0));
        assert!(bo.requires_input(1));
        assert!(bo.requires_input(2));
        assert!(bo.requires_input(3));
    }

    #[test]
    fn shapes_are_validated() {
        // x [rows=2, in=4], A [rank=3, in=4], B [out=5, rank=3].
        let x = t2(&[0.0; 8], (2, 4));
        let a = t2(&[0.0; 12], (3, 4));
        let b = t2(&[0.0; 15], (5, 3));
        let dy_bad = Tensor::from_slice(&[0.0f32; 6], vec![6]).unwrap(); // wrong rank
        let bo = LoraDeltaAddBackward {
            x: x.clone(),
            a: a.clone(),
            b: b.clone(),
            scale: 1.0,
        };
        let e = bo.apply(&dy_bad).unwrap_err().to_string();
        assert!(
            e.contains("grad_output"),
            "expected grad_output shape error, got: {e}"
        );

        // A.in_features must match x.in_features.
        let a_bad = t2(&[0.0; 9], (3, 3));
        let bo2 = LoraDeltaAddBackward {
            x: x.clone(),
            a: a_bad,
            b: b.clone(),
            scale: 1.0,
        };
        let dy = t2(&[0.0; 10], (2, 5));
        let e2 = bo2.apply(&dy).unwrap_err().to_string();
        assert!(e2.contains("in_features"), "expected in_features error: {e2}");
    }

    /// End-to-end shape and zero-grad-out sanity: with grad_out = 0, all
    /// returned grads must be zero with the right shapes.
    #[test]
    fn zero_grad_in_zero_grad_out_shapes() {
        // Realistic LoRA shapes: rows=4, in=8, rank=2, out=6.
        let rows = 4;
        let in_features = 8;
        let rank = 2;
        let out_features = 6;
        let x = t2(&vec![1.0; rows * in_features], (rows, in_features));
        let a = t2(&vec![2.0; rank * in_features], (rank, in_features));
        let b = t2(&vec![3.0; out_features * rank], (out_features, rank));
        let dy = t2(&vec![0.0; rows * out_features], (rows, out_features));
        let bo = LoraDeltaAddBackward {
            x,
            a,
            b,
            scale: 0.5,
        };
        let grads = bo.apply(&dy).unwrap();
        assert_eq!(grads.len(), 4);
        let grad_base = grads[0].as_ref().unwrap();
        let grad_x = grads[1].as_ref().unwrap();
        let grad_a = grads[2].as_ref().unwrap();
        let grad_b = grads[3].as_ref().unwrap();
        assert_eq!(grad_base.shape(), &[rows, out_features]);
        assert_eq!(grad_x.shape(), &[rows, in_features]);
        assert_eq!(grad_a.shape(), &[rank, in_features]);
        assert_eq!(grad_b.shape(), &[out_features, rank]);
        for v in read_f32(grad_x) {
            assert_eq!(v, 0.0);
        }
        for v in read_f32(grad_a) {
            assert_eq!(v, 0.0);
        }
        for v in read_f32(grad_b) {
            assert_eq!(v, 0.0);
        }
    }

    /// Tiny analytic case: compute the LoRA forward via kt ops, derive
    /// `grad_out = ones` (sum loss), and check `grad_A` / `grad_B`
    /// against a hand-computed reference.
    #[test]
    fn fused_backward_matches_hand_derived() {
        // x [1, 2]: [1, 2]
        // A [1, 2]: [3, 4]            (rank=1, in=2)
        // B [1, 1]: [5]               (out=1, rank=1)
        // h = x @ A^T = 1*3 + 2*4 = 11               shape [1, 1]
        // d = h @ B^T = 11 * 5 = 55                  shape [1, 1]
        // out = base + scale * d                     shape [1, 1]
        let x = t2(&[1.0f32, 2.0], (1, 2));
        let a = t2(&[3.0f32, 4.0], (1, 2));
        let b = t2(&[5.0f32], (1, 1));
        let scale = 0.5_f32;

        // Forward via kt ops, to confirm the apply backward composes
        // against the same maths.
        let a_t = a.transpose(0, 1).unwrap().contiguous().unwrap();
        let h = kt_matmul(&x, &a_t).unwrap();
        let b_t = b.transpose(0, 1).unwrap().contiguous().unwrap();
        let d = kt_matmul(&h, &b_t).unwrap();
        let delta = kt_mul_scalar(&d, scale).unwrap();
        let base = t2(&[7.0f32], (1, 1));
        let out = add(&base, &delta).unwrap();
        let out_v = read_f32(&out);
        // out = 7 + 0.5 * 55 = 34.5
        assert!((out_v[0] - 34.5).abs() < 1e-5);

        // grad_out = ones (sum-loss seed).
        let dy = t2(&[1.0f32], (1, 1));
        let bo = LoraDeltaAddBackward {
            x: x.clone(),
            a: a.clone(),
            b: b.clone(),
            scale,
        };
        let grads = bo.apply(&dy).unwrap();
        let grad_base = grads[0].as_ref().unwrap();
        let grad_x = grads[1].as_ref().unwrap();
        let grad_a = grads[2].as_ref().unwrap();
        let grad_b = grads[3].as_ref().unwrap();

        // Hand-derived references for sum-of-out loss:
        //   grad_d = scale         = 0.5
        //   grad_h = grad_d * B    = 0.5 * 5 = 2.5
        //   grad_x = grad_h * A    = 2.5 * [3, 4] = [7.5, 10]
        //   grad_A = grad_h * x    = 2.5 * [1, 2] = [2.5, 5.0]
        //   grad_B = grad_d * h    = 0.5 * 11 = 5.5
        //   grad_base = 1.0
        assert_eq!(read_f32(grad_base), vec![1.0]);
        let gx = read_f32(grad_x);
        assert!((gx[0] - 7.5).abs() < 1e-5);
        assert!((gx[1] - 10.0).abs() < 1e-5);
        let ga = read_f32(grad_a);
        assert!((ga[0] - 2.5).abs() < 1e-5);
        assert!((ga[1] - 5.0).abs() < 1e-5);
        let gb = read_f32(grad_b);
        assert!((gb[0] - 5.5).abs() < 1e-5);
    }

    /// Finite-difference parity check: vary one entry of A and B,
    /// compare against the analytic backward at moderate shapes.
    #[test]
    fn fused_backward_finite_difference_parity() {
        // Reproducible deterministic fill — keep things tiny so the FD
        // step is reliable in F32.
        let rows = 3;
        let in_features = 4;
        let rank = 2;
        let out_features = 3;
        let scale = 0.75_f32;
        let x_data: Vec<f32> = (0..rows * in_features).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let a_data: Vec<f32> = (0..rank * in_features).map(|i| 0.2 - (i as f32) * 0.05).collect();
        let b_data: Vec<f32> = (0..out_features * rank).map(|i| (i as f32) * 0.04 - 0.1).collect();
        let x = t2(&x_data, (rows, in_features));
        let a = t2(&a_data, (rank, in_features));
        let b = t2(&b_data, (out_features, rank));

        let forward = |x: &Tensor, a: &Tensor, b: &Tensor| -> f32 {
            let a_t = a.transpose(0, 1).unwrap().contiguous().unwrap();
            let h = kt_matmul(x, &a_t).unwrap();
            let b_t = b.transpose(0, 1).unwrap().contiguous().unwrap();
            let d = kt_matmul(&h, &b_t).unwrap();
            let delta = kt_mul_scalar(&d, scale).unwrap();
            // Sum-loss is just the sum of delta (drop base since
            // grad_out = ones and grad_base passes through; we
            // exercise the LoRA-only part of the gradient with the
            // base set to zero).
            read_f32(&delta).iter().sum()
        };

        let dy = t2(&vec![1.0; rows * out_features], (rows, out_features));
        let bo = LoraDeltaAddBackward {
            x: x.clone(),
            a: a.clone(),
            b: b.clone(),
            scale,
        };
        let grads = bo.apply(&dy).unwrap();
        let grad_a = grads[2].as_ref().unwrap();
        let grad_b = grads[3].as_ref().unwrap();
        let ga_v = read_f32(grad_a);
        let gb_v = read_f32(grad_b);

        // Spot-check three entries of grad_A via finite difference.
        let eps = 1e-3_f32;
        let tol = 5e-3_f32;
        for &(ki, ii) in &[(0_usize, 0_usize), (0, 3), (1, 2)] {
            let mut a_plus = a_data.clone();
            let mut a_minus = a_data.clone();
            a_plus[ki * in_features + ii] += eps;
            a_minus[ki * in_features + ii] -= eps;
            let a_p = t2(&a_plus, (rank, in_features));
            let a_m = t2(&a_minus, (rank, in_features));
            let fd = (forward(&x, &a_p, &b) - forward(&x, &a_m, &b)) / (2.0 * eps);
            let analytic = ga_v[ki * in_features + ii];
            assert!(
                (fd - analytic).abs() < tol,
                "grad_A[{ki},{ii}] FD {fd} vs analytic {analytic}"
            );
        }

        // Same for two entries of grad_B.
        for &(oi, ri) in &[(0_usize, 0_usize), (2, 1)] {
            let mut b_plus = b_data.clone();
            let mut b_minus = b_data.clone();
            b_plus[oi * rank + ri] += eps;
            b_minus[oi * rank + ri] -= eps;
            let b_p = t2(&b_plus, (out_features, rank));
            let b_m = t2(&b_minus, (out_features, rank));
            let fd = (forward(&x, &a, &b_p) - forward(&x, &a, &b_m)) / (2.0 * eps);
            let analytic = gb_v[oi * rank + ri];
            assert!(
                (fd - analytic).abs() < tol,
                "grad_B[{oi},{ri}] FD {fd} vs analytic {analytic}"
            );
        }
    }
}
