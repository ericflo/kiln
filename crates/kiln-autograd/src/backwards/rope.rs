//! `RopeBackward` — gradient of `rope(x, cos, sin)` (rotary position
//! embedding).
//!
//! Forward (from `kiln_tensor::ops::rope`):
//!
//! ```text
//! y[..., s, 2i]     = x[..., s, 2i] * cos[s, i] - x[..., s, 2i+1] * sin[s, i]
//! y[..., s, 2i+1]   = x[..., s, 2i] * sin[s, i] + x[..., s, 2i+1] * cos[s, i]
//! ```
//!
//! Indices past `rotary_dim` are pass-through (partial-rotary case).
//!
//! # Backward
//!
//! RoPE is a unitary rotation in each `(2i, 2i+1)` pair, so the
//! adjoint operation is rotation by the same angle in the **opposite**
//! direction (i.e. with `sin` negated):
//!
//! ```text
//! dx[..., s, 2i]     =  dy[..., s, 2i] * cos[s, i] + dy[..., s, 2i+1] * sin[s, i]
//! dx[..., s, 2i+1]   = -dy[..., s, 2i] * sin[s, i] + dy[..., s, 2i+1] * cos[s, i]
//! ```
//!
//! Pass-through indices (`head_dim - rotary_dim` trailing positions)
//! receive their incoming gradient unchanged.
//!
//! `cos` and `sin` are precomputed schedules — non-differentiable;
//! their gradients are `None`.

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct RopeBackward {
    /// `rotary_dim` from the forward op (number of pair-rotated
    /// trailing dims).
    pub rotary_dim: usize,
    /// Saved cos schedule `[seq, rotary_dim/2]`.
    pub cos: Tensor,
    /// Saved sin schedule `[seq, rotary_dim/2]`.
    pub sin: Tensor,
}

impl BackwardOp for RopeBackward {
    fn name(&self) -> &'static str {
        "rope_backward"
    }
    fn input_count(&self) -> usize {
        3
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let shape = grad_output.shape();
        if shape.len() < 2 {
            bail!(
                "RopeBackward: grad_output must have rank ≥ 2 (got {:?})",
                shape
            );
        }
        let dtype = grad_output.dtype();
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "RopeBackward: dtype must be F32/BF16/F16, got {dtype}"
            );
        }
        let head_dim = *shape.last().unwrap();
        if self.rotary_dim > head_dim {
            bail!(
                "RopeBackward: rotary_dim ({}) > head_dim ({head_dim})",
                self.rotary_dim
            );
        }
        if self.rotary_dim % 2 != 0 {
            bail!(
                "RopeBackward: rotary_dim must be even, got {}",
                self.rotary_dim
            );
        }
        let pair_count = self.rotary_dim / 2;
        let seq = shape[shape.len() - 2];
        // cos/sin shape [seq, pair_count].
        if self.cos.shape() != [seq, pair_count].as_slice()
            || self.sin.shape() != [seq, pair_count].as_slice()
        {
            bail!(
                "RopeBackward: cos/sin must be shape [{seq}, {pair_count}], got cos={:?} sin={:?}",
                self.cos.shape(),
                self.sin.shape()
            );
        }
        let leading: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);

        let dy = load_f32(grad_output)?;
        let cos = load_f32(&self.cos)?;
        let sin = load_f32(&self.sin)?;
        let mut dx = dy.clone(); // pass-through tail is identity

        for l in 0..leading {
            for s in 0..seq {
                let row_base = ((l * seq) + s) * head_dim;
                let sched_base = s * pair_count;
                for i in 0..pair_count {
                    let c = cos[sched_base + i];
                    let si = sin[sched_base + i];
                    let i0 = row_base + 2 * i;
                    let i1 = row_base + 2 * i + 1;
                    let dy0 = dy[i0];
                    let dy1 = dy[i1];
                    dx[i0] = dy0 * c + dy1 * si;
                    dx[i1] = -dy0 * si + dy1 * c;
                }
            }
        }

        let out = store_f32(dtype, shape, &dx)?;
        Ok(vec![
            Some(out),
            None, // d_cos: non-differentiable
            None, // d_sin: non-differentiable
        ])
    }
    fn requires_input(&self, idx: usize) -> bool {
        match idx {
            0 => false, // x not needed
            1 => true,  // cos saved
            2 => true,  // sin saved
            _ => false,
        }
    }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn load_f32(t: &Tensor) -> Result<Vec<f32>> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("rope_backward: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let dtype = t.dtype();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(match dtype {
            DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            DType::F16 => half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            _ => unreachable!(),
        });
    }
    Ok(out)
}

fn store_f32(dtype: DType, shape: &[usize], data: &[f32]) -> Result<Tensor> {
    let per = dtype.size_in_bytes();
    let mut bytes = vec![0u8; data.len() * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in data.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for (i, &v) in data.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in data.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::ops::rope;

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "len mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "idx {i}: got {x}, want {y} (tol {tol})"
            );
        }
    }

    #[test]
    fn rope_backward_identity_angle() {
        // cos = 1, sin = 0 → rope is the identity. Backward also identity.
        let cos = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let sin = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
        // x [seq=1, head_dim=2]
        let dy = Tensor::from_slice(&[3.0f32, 4.0], vec![1, 2]).unwrap();
        let bo = RopeBackward {
            rotary_dim: 2,
            cos,
            sin,
        };
        let grads = bo.apply(&dy).unwrap();
        let dx = load_f32(grads[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[3.0, 4.0], 1e-6);
        assert!(grads[1].is_none());
        assert!(grads[2].is_none());
    }

    #[test]
    fn rope_backward_quarter_turn() {
        // cos = 0, sin = 1 → forward rotates (x_0, x_1) → (-x_1, x_0).
        // The adjoint (backward) is rotation by -90°, i.e. (y_0, y_1) →
        // (y_1, -y_0). Test: dy = (1, 0) → dx = (0, -1).
        let cos = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
        let sin = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 0.0], vec![1, 2]).unwrap();
        let bo = RopeBackward {
            rotary_dim: 2,
            cos,
            sin,
        };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[0.0, -1.0], 1e-6);
    }

    #[test]
    fn rope_backward_passes_through_non_rotary_indices() {
        // head_dim = 4, rotary_dim = 2. Indices 2, 3 should pass through.
        let cos = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let sin = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let bo = RopeBackward {
            rotary_dim: 2,
            cos,
            sin,
        };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        // cos=1, sin=0 → identity rotation. tail passes through.
        approx(&dx, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn rope_backward_finite_difference() {
        // Pick a non-trivial angle.
        let theta = 0.7_f32;
        let cos_val = theta.cos();
        let sin_val = theta.sin();
        let cos = Tensor::from_slice(&[cos_val], vec![1, 1]).unwrap();
        let sin = Tensor::from_slice(&[sin_val], vec![1, 1]).unwrap();
        let x_data = vec![3.0f32, 5.0];
        let x = Tensor::from_slice(&x_data, vec![1, 2]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 1.0], vec![1, 2]).unwrap();
        let bo = RopeBackward {
            rotary_dim: 2,
            cos: cos.clone(),
            sin: sin.clone(),
        };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();

        // Analytic check from the formula:
        // dx_0 = dy_0*cos + dy_1*sin = cos + sin
        // dx_1 = -dy_0*sin + dy_1*cos = -sin + cos
        let expected = vec![cos_val + sin_val, -sin_val + cos_val];
        approx(&dx, &expected, 1e-6);

        // Also confirm via finite difference of sum(rope(x, cos, sin)).
        let loss = |x_vec: &[f32]| -> f32 {
            let xt = Tensor::from_slice(x_vec, vec![1, 2]).unwrap();
            let y = rope(&xt, &cos, &sin, 2).unwrap();
            load_f32(&y).unwrap().iter().sum()
        };
        let step = 1e-3;
        let mut fd = Vec::with_capacity(2);
        for i in 0..2 {
            let mut up = x_data.clone();
            up[i] += step;
            let mut dn = x_data.clone();
            dn[i] -= step;
            fd.push((loss(&up) - loss(&dn)) / (2.0 * step));
        }
        approx(&dx, &fd, 1e-3);
    }

    #[test]
    fn rope_backward_rejects_bad_rotary_dim() {
        let cos = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let sin = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let bo = RopeBackward {
            rotary_dim: 4,
            cos,
            sin,
        };
        let e = bo.apply(&dy).unwrap_err();
        assert!(e.to_string().contains("rotary_dim"));
    }

    #[test]
    fn op_metadata() {
        let cos = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let sin = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
        let bo = RopeBackward {
            rotary_dim: 2,
            cos,
            sin,
        };
        assert_eq!(bo.name(), "rope_backward");
        assert_eq!(bo.input_count(), 3);
        assert!(!bo.requires_input(0));
        assert!(bo.requires_input(1));
        assert!(bo.requires_input(2));
    }
}
