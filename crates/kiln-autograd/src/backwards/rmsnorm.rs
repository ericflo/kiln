//! `RmsNormBackward` — gradient of `rms_norm(x, weight, eps)`.
//!
//! Forward (from `kiln_tensor::ops::rms_norm`):
//!
//! ```text
//! r = sqrt(mean_j(x_j^2) + eps)        per trailing-axis row
//! y_i = x_i * w_i / r                  per element
//! ```
//!
//! # Backward derivation
//!
//! Let `D` = trailing-axis size.
//!
//! ```text
//! ∂r/∂x_k    = x_k / (D * r)
//! ∂y_i/∂x_k  = δ_ik * w_i / r  -  x_i * w_i / r^2 * ∂r/∂x_k
//!            = δ_ik * w_i / r  -  x_i * w_i * x_k / (D * r^3)
//! ∂y_i/∂w_k  = δ_ik * x_i / r
//! ```
//!
//! Per row:
//!
//! ```text
//! S    = Σⱼ dy_j * x_j * w_j
//! dx_k = (dy_k * w_k) / r  -  x_k * S / (D * r^3)
//! ```
//!
//! Weight gradient is summed across all non-trailing axes (the weight
//! is shared across batch + sequence):
//!
//! ```text
//! dw_k = Σ_rows (dy_k * x_k) / r
//! ```
//!
//! # Determinism
//!
//! `ToleranceBounded { atomic-bwd }`. The dw reduction sums F32
//! contributions across batch positions; the order is fixed on CPU
//! but GPU atomicAdd is non-associative for F32.

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct RmsNormBackward {
    /// Saved `x` from the forward pass.
    pub x: Tensor,
    /// Saved `weight` from the forward pass.
    pub weight: Tensor,
    /// Epsilon used in the forward pass.
    pub eps: f32,
}

impl BackwardOp for RmsNormBackward {
    fn name(&self) -> &'static str {
        "rmsnorm_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let x_shape = self.x.shape();
        let g_shape = grad_output.shape();
        if x_shape != g_shape {
            bail!(
                "RmsNormBackward: grad_output shape {:?} != saved x shape {:?}",
                g_shape,
                x_shape
            );
        }
        if self.weight.rank() != 1 {
            bail!("RmsNormBackward: weight must be rank-1");
        }
        let hidden = self.weight.shape()[0];
        if *x_shape.last().unwrap() != hidden {
            bail!(
                "RmsNormBackward: x trailing axis {} != weight len {hidden}",
                x_shape.last().unwrap()
            );
        }
        if !matches!(self.x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "RmsNormBackward: dtype must be F32/BF16/F16, got {}",
                self.x.dtype()
            );
        }
        let dtype = self.x.dtype();
        if grad_output.dtype() != dtype || self.weight.dtype() != dtype {
            bail!(
                "RmsNormBackward: dtype mismatch (x={}, weight={}, grad={})",
                dtype,
                self.weight.dtype(),
                grad_output.dtype()
            );
        }

        // Load everything to F32 for the accumulation.
        let x = load_f32(&self.x)?;
        let w = load_f32(&self.weight)?;
        let dy = load_f32(grad_output)?;

        let n_rows = x.len() / hidden;
        let mut dx = vec![0.0f32; x.len()];
        let mut dw = vec![0.0f32; hidden];
        let eps = self.eps;

        for r in 0..n_rows {
            let base = r * hidden;
            // r = sqrt(mean(x^2) + eps).
            let mean_sq: f32 =
                x[base..base + hidden].iter().map(|&v| v * v).sum::<f32>() / hidden as f32;
            let r_val = (mean_sq + eps).sqrt();
            // S = Σⱼ dy_j * x_j * w_j.
            let mut s = 0.0f32;
            for j in 0..hidden {
                s += dy[base + j] * x[base + j] * w[j];
            }
            let inv_r = 1.0 / r_val;
            let inv_r3_d = inv_r * inv_r * inv_r / hidden as f32;
            for k in 0..hidden {
                dx[base + k] = dy[base + k] * w[k] * inv_r - x[base + k] * s * inv_r3_d;
                dw[k] += dy[base + k] * x[base + k] * inv_r;
            }
        }

        let dx_t = store_f32(dtype, x_shape, &dx)?;
        let dw_t = store_f32(dtype, self.weight.shape(), &dw)?;
        Ok(vec![Some(dx_t), Some(dw_t)])
    }
    fn requires_input(&self, idx: usize) -> bool {
        // Both x and weight are saved; backward needs them.
        idx == 0 || idx == 1
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
        .ok_or_else(|| Error::from_str("rmsnorm_backward: storage must be CpuStorage"))?;
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
    use kiln_tensor::ops::rms_norm;

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "len mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "idx {i}: got {x}, want {y} (tol {tol})"
            );
        }
    }

    fn fd_grad_x<F: Fn(&[f32]) -> f32>(x: &[f32], loss: F, step: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let mut up = x.to_vec();
            up[i] += step;
            let mut dn = x.to_vec();
            dn[i] -= step;
            out.push((loss(&up) - loss(&dn)) / (2.0 * step));
        }
        out
    }

    #[test]
    fn rmsnorm_backward_2d_unit_weight() {
        // x = [3, 4, 0], weight = [1, 1, 1], eps = 0.
        // r = sqrt((9+16+0)/3) = sqrt(25/3) ≈ 2.886751
        // y = x / r ≈ [1.039, 1.386, 0.0]
        // dy = [1, 0, 0].
        // S = dy.x.w sum = 1*3*1 = 3
        // inv_r = 1/2.886751 ≈ 0.346410
        // inv_r3_d = inv_r^3 / 3 ≈ 0.013856
        // dx_0 = 1 * 1 * 0.346410 - 3 * 3 * 0.013856 ≈ 0.346410 - 0.124704 ≈ 0.221706
        // dx_1 = 0 - 4 * 3 * 0.013856 ≈ -0.166267
        // dx_2 = 0 - 0 ≈ 0
        // dw_k = dy_k * x_k * inv_r = [1*3*0.34641, 0, 0] ≈ [1.03923, 0, 0]
        let x = Tensor::from_slice(&[3.0f32, 4.0, 0.0], vec![1, 3]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 0.0, 0.0], vec![1, 3]).unwrap();
        let bo = RmsNormBackward {
            x,
            weight: w,
            eps: 0.0,
        };
        let grads = bo.apply(&dy).unwrap();
        let dx = load_f32(grads[0].as_ref().unwrap()).unwrap();
        let dw = load_f32(grads[1].as_ref().unwrap()).unwrap();
        approx(&dx, &[0.221706, -0.166267, 0.0], 1e-4);
        approx(&dw, &[1.03923, 0.0, 0.0], 1e-4);
    }

    #[test]
    fn rmsnorm_backward_finite_difference_x() {
        let x_data = vec![1.0f32, -2.0, 3.0, 0.5];
        let w_data = vec![0.7f32, 1.3, 0.9, 1.1];
        let eps = 1e-6f32;
        let x = Tensor::from_slice(&x_data, vec![1, 4]).unwrap();
        let w = Tensor::from_slice(&w_data, vec![4]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 4], vec![1, 4]).unwrap();
        let bo = RmsNormBackward {
            x: x.clone(),
            weight: w.clone(),
            eps,
        };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();

        // Compute loss(x) = sum(rms_norm(x, w, eps)). Backward of sum
        // sets dy=ones, which matches our test grad_output. So
        // analytic dx must equal the finite-difference of loss.
        let loss = |x_vec: &[f32]| -> f32 {
            let xt = Tensor::from_slice(x_vec, vec![1, 4]).unwrap();
            let wt = Tensor::from_slice(&w_data, vec![4]).unwrap();
            let y = rms_norm(&xt, &wt, eps).unwrap();
            load_f32(&y).unwrap().iter().sum()
        };
        let fd = fd_grad_x(&x_data, loss, 1e-3);
        approx(&dx, &fd, 1e-2);
    }

    #[test]
    fn rmsnorm_backward_finite_difference_w() {
        let x_data = vec![1.0f32, -2.0, 3.0, 0.5];
        let w_data = vec![0.7f32, 1.3, 0.9, 1.1];
        let eps = 1e-6f32;
        let x = Tensor::from_slice(&x_data, vec![1, 4]).unwrap();
        let w = Tensor::from_slice(&w_data, vec![4]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 4], vec![1, 4]).unwrap();
        let bo = RmsNormBackward {
            x: x.clone(),
            weight: w.clone(),
            eps,
        };
        let dw = load_f32(bo.apply(&dy).unwrap()[1].as_ref().unwrap()).unwrap();

        let loss = |w_vec: &[f32]| -> f32 {
            let xt = Tensor::from_slice(&x_data, vec![1, 4]).unwrap();
            let wt = Tensor::from_slice(w_vec, vec![4]).unwrap();
            let y = rms_norm(&xt, &wt, eps).unwrap();
            load_f32(&y).unwrap().iter().sum()
        };
        let fd = fd_grad_x(&w_data, loss, 1e-3);
        approx(&dw, &fd, 1e-2);
    }

    #[test]
    fn rmsnorm_backward_dw_accumulates_across_batch() {
        // Two batch rows; dw must be the sum across both.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let bo = RmsNormBackward {
            x,
            weight: w,
            eps: 0.0,
        };
        let dw = load_f32(bo.apply(&dy).unwrap()[1].as_ref().unwrap()).unwrap();
        // Row 0: x=[1,2], mean_sq=2.5, r≈1.5811, inv_r≈0.6325.
        //   dw += [1*1*0.6325, 1*2*0.6325] = [0.6325, 1.2649]
        // Row 1: x=[3,4], mean_sq=12.5, r≈3.5355, inv_r≈0.2828.
        //   dw += [1*3*0.2828, 1*4*0.2828] = [0.8485, 1.1314]
        // Total: [1.481, 2.396]
        approx(&dw, &[1.481, 2.396], 5e-3);
    }

    #[test]
    fn rmsnorm_backward_bf16_round_trip() {
        let x_data: Vec<half::bf16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let w_data: Vec<half::bf16> = [1.0f32; 3]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let dy_data: Vec<half::bf16> = [1.0f32, 0.0, 0.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&x_data, vec![1, 3]).unwrap();
        let w = Tensor::from_slice(&w_data, vec![3]).unwrap();
        let dy = Tensor::from_slice(&dy_data, vec![1, 3]).unwrap();
        let bo = RmsNormBackward {
            x,
            weight: w,
            eps: 0.0,
        };
        let g = bo.apply(&dy).unwrap();
        assert_eq!(g[0].as_ref().unwrap().dtype(), DType::BF16);
        assert_eq!(g[1].as_ref().unwrap().dtype(), DType::BF16);
    }

    #[test]
    fn rmsnorm_backward_shape_mismatch_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let bad = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![1, 3]).unwrap();
        let bo = RmsNormBackward {
            x,
            weight: w,
            eps: 0.0,
        };
        let e = bo.apply(&bad).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    #[test]
    fn op_metadata() {
        let x = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let w = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = RmsNormBackward {
            x,
            weight: w,
            eps: 1e-6,
        };
        assert_eq!(bo.name(), "rmsnorm_backward");
        assert_eq!(bo.input_count(), 2);
        assert!(bo.requires_input(0));
        assert!(bo.requires_input(1));
    }
}
