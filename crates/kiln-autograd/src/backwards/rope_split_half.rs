//! `RopeSplitHalfBackward` — gradient of
//! `kiln_tensor::ops::rope_split_half` (split-half / GPT-NeoX rotary
//! position embedding; kiln's Qwen3.5-4B convention).
//!
//! # Backward
//!
//! RoPE is a unitary rotation in each `(i, i + rotary_dim/2)` lane pair,
//! so the adjoint is rotation by the negated angle — i.e. the *same*
//! forward op with `sin` negated:
//!
//! ```text
//! dx = rope_split_half(dy, cos, -sin, rotary_dim)
//! ```
//!
//! This mirrors the production training kernel
//! (`kiln-model::cuda_train::cuda_rope`, which runs
//! `cuda_rope_apply(grad, cos, sin, rotary_dim, /*inverse=*/true)`).
//!
//! `cos` / `sin` are precomputed schedules — non-differentiable; the op
//! has a single differentiable input (`x`), so `apply` returns one grad.
//!
//! # Device-agnostic, no host round-trip
//!
//! Unlike the CPU-`load_f32` backward ops, this runs entirely through the
//! device-agnostic `rope_split_half` composite, so the gradient is
//! computed on whatever device `grad_output` lives on (CPU / CUDA /
//! Vulkan / Metal) with no copy to host. The saved `cos`/`sin` are moved
//! to `grad_output`'s device only if not already there (they are tiny
//! `[seq, rotary_dim/2]` schedules).

use kiln_tensor::ops::{mul_scalar, rope_split_half};
use kiln_tensor::{Result, Tensor};

use crate::BackwardOp;

#[derive(Debug)]
pub struct RopeSplitHalfBackward {
    /// `rotary_dim` from the forward op.
    pub rotary_dim: usize,
    /// Saved cos schedule `[seq, rotary_dim/2]`.
    pub cos: Tensor,
    /// Saved sin schedule `[seq, rotary_dim/2]`.
    pub sin: Tensor,
}

impl BackwardOp for RopeSplitHalfBackward {
    fn name(&self) -> &'static str {
        "rope_split_half_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let dev = grad_output.device();
        let on_dev = |t: &Tensor| -> Result<Tensor> {
            if t.device() == dev {
                Ok(t.clone())
            } else {
                t.to_device(dev)
            }
        };
        let cos = on_dev(&self.cos)?;
        let sin = on_dev(&self.sin)?;
        // Adjoint of a rotation is rotation by the negated angle.
        let neg_sin = mul_scalar(&sin, -1.0)?;
        let dx = rope_split_half(grad_output, &cos, &neg_sin, self.rotary_dim)?;
        Ok(vec![Some(dx)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false // x is not needed for the backward (analytic rotation).
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::ops::rope_split_half;
    use kiln_tensor::{CpuStorage, Tensor};

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Analytic split-half adjoint reference (independent of the op).
    #[allow(clippy::too_many_arguments)]
    fn ref_dx(
        dy: &[f32],
        cos: &[f32],
        sin: &[f32],
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Vec<f32> {
        let half = rotary_dim / 2;
        let mut dx = dy.to_vec();
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..heads {
                    let row = (((b * seq) + s) * heads + h) * head_dim;
                    let sched = s * half;
                    for i in 0..half {
                        let c = cos[sched + i];
                        let sn = sin[sched + i];
                        let dy0 = dy[row + i];
                        let dy1 = dy[row + half + i];
                        dx[row + i] = dy0 * c + dy1 * sn;
                        dx[row + half + i] = -dy0 * sn + dy1 * c;
                    }
                }
            }
        }
        dx
    }

    #[test]
    fn matches_analytic_reference() {
        let (batch, seq, heads, head_dim, rotary_dim) = (2, 3, 2, 8, 4);
        let half = rotary_dim / 2;
        let n = batch * seq * heads * head_dim;
        let dy: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.25 - 0.5).collect();
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for s in 0..seq {
            for i in 0..half {
                let theta = 0.6 * (s as f32) + 0.2 * (i as f32);
                cos.push(theta.cos());
                sin.push(theta.sin());
            }
        }
        let dyt = Tensor::from_slice(&dy, vec![batch, seq, heads, head_dim]).unwrap();
        let ct = Tensor::from_slice(&cos, vec![seq, half]).unwrap();
        let st = Tensor::from_slice(&sin, vec![seq, half]).unwrap();
        let bo = RopeSplitHalfBackward {
            rotary_dim,
            cos: ct,
            sin: st,
        };
        let grads = bo.apply(&dyt).unwrap();
        assert_eq!(grads.len(), 1);
        let dx = read_f32(grads[0].as_ref().unwrap());
        let want = ref_dx(&dy, &cos, &sin, batch, seq, heads, head_dim, rotary_dim);
        for (i, (g, w)) in dx.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-5, "idx {i}: got {g}, want {w}");
        }
    }

    #[test]
    fn finite_difference() {
        let (batch, seq, heads, head_dim, rotary_dim) = (1, 2, 1, 6, 4);
        let half = rotary_dim / 2;
        let n = batch * seq * heads * head_dim;
        let x: Vec<f32> = (0..n).map(|i| 0.3 * (i as f32) - 0.4).collect();
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for s in 0..seq {
            for i in 0..half {
                let theta = 0.7 * (s as f32 + 1.0) + 0.15 * (i as f32);
                cos.push(theta.cos());
                sin.push(theta.sin());
            }
        }
        let ct = Tensor::from_slice(&cos, vec![seq, half]).unwrap();
        let st = Tensor::from_slice(&sin, vec![seq, half]).unwrap();

        let dy = vec![1.0f32; n];
        let dyt = Tensor::from_slice(&dy, vec![batch, seq, heads, head_dim]).unwrap();
        let bo = RopeSplitHalfBackward {
            rotary_dim,
            cos: ct.clone(),
            sin: st.clone(),
        };
        let dx = read_f32(bo.apply(&dyt).unwrap()[0].as_ref().unwrap());

        let loss = |xv: &[f32]| -> f32 {
            let xt = Tensor::from_slice(xv, vec![batch, seq, heads, head_dim]).unwrap();
            let y = rope_split_half(&xt, &ct, &st, rotary_dim).unwrap();
            read_f32(&y).iter().sum()
        };
        let step = 1e-3;
        for j in 0..n {
            let mut up = x.clone();
            up[j] += step;
            let mut dn = x.clone();
            dn[j] -= step;
            let fd = (loss(&up) - loss(&dn)) / (2.0 * step);
            assert!(
                (dx[j] - fd).abs() < 1e-2,
                "idx {j}: analytic {} vs fd {fd}",
                dx[j]
            );
        }
    }

    #[test]
    fn op_metadata() {
        let ct = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let st = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
        let bo = RopeSplitHalfBackward {
            rotary_dim: 2,
            cos: ct,
            sin: st,
        };
        assert_eq!(bo.name(), "rope_split_half_backward");
        assert_eq!(bo.input_count(), 1);
        assert!(!bo.requires_input(0));
    }
}
