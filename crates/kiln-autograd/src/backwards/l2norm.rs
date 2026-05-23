//! `L2NormBackward` — gradient of `l2_norm(x, eps)`.
//!
//! Forward (from `kiln_tensor::ops::l2_norm`):
//!
//! ```text
//! norm[r] = sqrt(sum_j x[r, j]^2 + eps)   per trailing-axis row
//! y[r, i] = x[r, i] / norm[r]
//! ```
//!
//! Same algebraic structure as `RmsNormBackward` minus the weight
//! term and the `/D` factor:
//!
//! ```text
//! S[r]     = Σⱼ dy[r, j] * x[r, j]
//! dx[r, k] = dy[r, k] / norm[r]  -  x[r, k] * S[r] / norm[r]^3
//! ```

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct L2NormBackward {
    /// Saved `x` from the forward pass.
    pub x: Tensor,
    /// `eps` used by the forward op.
    pub eps: f32,
}

impl BackwardOp for L2NormBackward {
    fn name(&self) -> &'static str {
        "l2_norm_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let shape = self.x.shape();
        if grad_output.shape() != shape {
            bail!(
                "L2NormBackward: grad_output shape {:?} != saved x shape {:?}",
                grad_output.shape(),
                shape
            );
        }
        if grad_output.dtype() != self.x.dtype() {
            bail!("L2NormBackward: dtype mismatch");
        }
        if !matches!(self.x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "L2NormBackward: dtype must be F32/BF16/F16, got {}",
                self.x.dtype()
            );
        }

        let dtype = self.x.dtype();
        let hidden = *shape.last().unwrap();
        let n_rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let mut dx = vec![0.0f32; x.len()];
        for r in 0..n_rows {
            let base = r * hidden;
            let sq: f32 = x[base..base + hidden].iter().map(|&v| v * v).sum();
            let norm = (sq + self.eps).sqrt();
            let inv_n = 1.0 / norm;
            let inv_n3 = inv_n * inv_n * inv_n;
            let mut s = 0.0f32;
            for j in 0..hidden {
                s += dy[base + j] * x[base + j];
            }
            for k in 0..hidden {
                dx[base + k] = dy[base + k] * inv_n - x[base + k] * s * inv_n3;
            }
        }
        let out = store_f32(dtype, shape, &dx)?;
        Ok(vec![Some(out)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

// ----------------------------------------------------------------------
// Helpers (mirrors of rmsnorm.rs).
// ----------------------------------------------------------------------

fn load_f32(t: &Tensor) -> Result<Vec<f32>> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("l2_norm_backward: storage must be CpuStorage"))?;
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
    use kiln_tensor::ops::l2_norm;

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "idx {i}: got {x}, want {y} (tol {tol})"
            );
        }
    }

    #[test]
    fn l2norm_backward_unit_vector() {
        // x = [3, 4, 0]. norm = 5. y = [0.6, 0.8, 0].
        // dy = [1, 0, 0]. S = 1*3 = 3.
        // dx_0 = 1/5 - 3 * 3 / 125 = 0.2 - 0.072 = 0.128
        // dx_1 = 0 - 4 * 3 / 125 = -0.096
        // dx_2 = 0 - 0 = 0
        let x = Tensor::from_slice(&[3.0f32, 4.0, 0.0], vec![1, 3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 0.0, 0.0], vec![1, 3]).unwrap();
        let bo = L2NormBackward { x, eps: 0.0 };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[0.128, -0.096, 0.0], 1e-4);
    }

    #[test]
    fn l2norm_backward_finite_difference() {
        let x_data = vec![1.0f32, -2.0, 3.0];
        let eps = 1e-6f32;
        let x = Tensor::from_slice(&x_data, vec![1, 3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 3], vec![1, 3]).unwrap();
        let bo = L2NormBackward {
            x: x.clone(),
            eps,
        };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        let loss = |x_vec: &[f32]| -> f32 {
            let xt = Tensor::from_slice(x_vec, vec![1, 3]).unwrap();
            let y = l2_norm(&xt, eps).unwrap();
            load_f32(&y).unwrap().iter().sum()
        };
        let step = 1e-3;
        let mut fd = Vec::with_capacity(3);
        for i in 0..3 {
            let mut up = x_data.clone();
            up[i] += step;
            let mut dn = x_data.clone();
            dn[i] -= step;
            fd.push((loss(&up) - loss(&dn)) / (2.0 * step));
        }
        approx(&dx, &fd, 1e-3);
    }

    #[test]
    fn l2norm_backward_batched_independent() {
        // [2, 3] input — two independent rows.
        let x = Tensor::from_slice(&[3.0f32, 4.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap();
        let bo = L2NormBackward {
            x,
            eps: 0.0,
        };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        // Row 0: same as the unit-vector test → [0.128, -0.096, 0]
        // Row 1: x=[0, 1, 0], norm=1. y=[0, 1, 0]. S = 1*1=1.
        //   dx_0 = 1 - 0*1/1 = 1, dx_1 = 1 - 1*1/1 = 0, dx_2 = 1 - 0 = 1
        approx(&dx[..3], &[0.128, -0.096, 0.0], 1e-4);
        approx(&dx[3..], &[1.0, 0.0, 1.0], 1e-6);
    }

    #[test]
    fn l2norm_backward_shape_mismatch_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let bad = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![1, 3]).unwrap();
        let bo = L2NormBackward { x, eps: 0.0 };
        let e = bo.apply(&bad).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    #[test]
    fn op_metadata() {
        let x = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let bo = L2NormBackward { x, eps: 1e-6 };
        assert_eq!(bo.name(), "l2_norm_backward");
        assert_eq!(bo.input_count(), 1);
        assert!(bo.requires_input(0));
    }
}
