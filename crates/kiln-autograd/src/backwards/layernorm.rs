//! `LayerNormBackward` — gradient of LayerNorm (with weight + bias).
//!
//! Forward (from `kiln_tensor::ops::layer_norm`):
//! ```text
//! μ        = mean(x)
//! σ²       = mean((x - μ)^2)
//! σ_eff    = sqrt(σ² + eps)
//! x̂_i      = (x_i - μ) / σ_eff
//! y_i      = x̂_i * w_i + b_i
//! ```
//!
//! Per-row gradient w.r.t. input (with `D = hidden dim`):
//! ```text
//! S1  = Σⱼ w_j * G_j
//! S2  = Σⱼ x̂_j * w_j * G_j
//! dx_i = (1/σ_eff) * [w_i * G_i - S1/D - x̂_i * S2 / D]
//! ```
//!
//! Weight/bias gradients (summed over all non-trailing axes):
//! ```text
//! dw_k = Σ_rows x̂_k * G_k
//! db_k = Σ_rows G_k
//! ```

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct LayerNormBackward {
    pub x: Tensor,
    pub weight: Tensor,
    pub eps: f32,
}

impl BackwardOp for LayerNormBackward {
    fn name(&self) -> &'static str {
        "layer_norm_backward"
    }
    fn input_count(&self) -> usize {
        3
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let x_shape = self.x.shape().to_vec();
        if grad_output.shape() != x_shape {
            bail!(
                "LayerNormBackward: grad shape {:?} != x shape {x_shape:?}",
                grad_output.shape()
            );
        }
        if self.weight.rank() != 1 {
            bail!("LayerNormBackward: weight must be rank-1");
        }
        let hidden = self.weight.shape()[0];
        if *x_shape.last().unwrap() != hidden {
            bail!(
                "LayerNormBackward: x trailing axis {} != weight len {hidden}",
                x_shape.last().unwrap()
            );
        }
        let dtype = self.x.dtype();
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "LayerNormBackward: dtype must be F32/BF16/F16, got {dtype}"
            );
        }
        if grad_output.dtype() != dtype || self.weight.dtype() != dtype {
            bail!("LayerNormBackward: dtype mismatch");
        }

        let x = load_f32(&self.x)?;
        let w = load_f32(&self.weight)?;
        let g = load_f32(grad_output)?;
        let n_rows = x.len() / hidden;
        let mut dx = vec![0.0f32; x.len()];
        let mut dw = vec![0.0f32; hidden];
        let mut db = vec![0.0f32; hidden];

        for r in 0..n_rows {
            let base = r * hidden;
            // 1. μ, σ_eff, x̂
            let mean: f32 = x[base..base + hidden].iter().sum::<f32>() / hidden as f32;
            let var: f32 = x[base..base + hidden]
                .iter()
                .map(|&v| (v - mean) * (v - mean))
                .sum::<f32>()
                / hidden as f32;
            let sigma = (var + self.eps).sqrt();
            let inv_s = 1.0_f32 / sigma;
            let xhat: Vec<f32> = x[base..base + hidden]
                .iter()
                .map(|&v| (v - mean) * inv_s)
                .collect();
            // 2. S1, S2
            let mut s1 = 0.0_f32;
            let mut s2 = 0.0_f32;
            for j in 0..hidden {
                s1 += w[j] * g[base + j];
                s2 += xhat[j] * w[j] * g[base + j];
            }
            let inv_d = 1.0_f32 / hidden as f32;
            for i in 0..hidden {
                dx[base + i] =
                    inv_s * (w[i] * g[base + i] - s1 * inv_d - xhat[i] * s2 * inv_d);
                dw[i] += xhat[i] * g[base + i];
                db[i] += g[base + i];
            }
        }
        let dx_t = store_f32(dtype, &x_shape, &dx)?;
        let dw_t = store_f32(dtype, self.weight.shape(), &dw)?;
        let db_t = store_f32(dtype, self.weight.shape(), &db)?;
        Ok(vec![Some(dx_t), Some(dw_t), Some(db_t)])
    }
    fn requires_input(&self, idx: usize) -> bool {
        idx == 0 || idx == 1 // x + weight saved
    }
}

fn load_f32(t: &Tensor) -> Result<Vec<f32>> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("layer_norm_backward: storage must be CpuStorage"))?;
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
    use kiln_tensor::ops::layer_norm;

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
    fn layernorm_backward_finite_difference_x() {
        let x_data = vec![1.0f32, -2.0, 3.0, 0.5];
        let w_data = vec![0.7f32, 1.3, 0.9, 1.1];
        let b_data = vec![0.1f32, 0.2, -0.1, 0.3];
        let eps = 1e-6f32;
        let x = Tensor::from_slice(&x_data, vec![1, 4]).unwrap();
        let w = Tensor::from_slice(&w_data, vec![4]).unwrap();
        let b = Tensor::from_slice(&b_data, vec![4]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 4], vec![1, 4]).unwrap();
        let bo = LayerNormBackward {
            x: x.clone(),
            weight: w.clone(),
            eps,
        };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();

        let loss = |x_vec: &[f32]| -> f32 {
            let xt = Tensor::from_slice(x_vec, vec![1, 4]).unwrap();
            let wt = Tensor::from_slice(&w_data, vec![4]).unwrap();
            let bt = Tensor::from_slice(&b_data, vec![4]).unwrap();
            let y = layer_norm(&xt, &wt, &bt, eps).unwrap();
            load_f32(&y).unwrap().iter().sum()
        };
        let step = 1e-3;
        let mut fd = Vec::with_capacity(4);
        for i in 0..4 {
            let mut up = x_data.clone();
            up[i] += step;
            let mut dn = x_data.clone();
            dn[i] -= step;
            fd.push((loss(&up) - loss(&dn)) / (2.0 * step));
        }
        approx(&dx, &fd, 5e-3);
    }

    #[test]
    fn layernorm_backward_dw_finite_difference() {
        let x_data = vec![1.0f32, -2.0, 3.0, 0.5];
        let w_data = vec![0.7f32, 1.3, 0.9, 1.1];
        let b_data = vec![0.1f32, 0.2, -0.1, 0.3];
        let eps = 1e-6f32;
        let x = Tensor::from_slice(&x_data, vec![1, 4]).unwrap();
        let w = Tensor::from_slice(&w_data, vec![4]).unwrap();
        let b = Tensor::from_slice(&b_data, vec![4]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 4], vec![1, 4]).unwrap();
        let bo = LayerNormBackward {
            x: x.clone(),
            weight: w.clone(),
            eps,
        };
        let grads = bo.apply(&dy).unwrap();
        let dw = load_f32(grads[1].as_ref().unwrap()).unwrap();

        let loss = |w_vec: &[f32]| -> f32 {
            let xt = Tensor::from_slice(&x_data, vec![1, 4]).unwrap();
            let wt = Tensor::from_slice(w_vec, vec![4]).unwrap();
            let bt = Tensor::from_slice(&b_data, vec![4]).unwrap();
            let y = layer_norm(&xt, &wt, &bt, eps).unwrap();
            load_f32(&y).unwrap().iter().sum()
        };
        let step = 1e-3;
        let mut fd = Vec::with_capacity(4);
        for i in 0..4 {
            let mut up = w_data.clone();
            up[i] += step;
            let mut dn = w_data.clone();
            dn[i] -= step;
            fd.push((loss(&up) - loss(&dn)) / (2.0 * step));
        }
        approx(&dw, &fd, 5e-3);
    }

    #[test]
    fn layernorm_backward_db_is_summed_grad() {
        // db[k] = Σ_rows G[r, k]. For dy=ones across a [3, 4] batch,
        // db = [3, 3, 3, 3].
        let x = Tensor::from_slice(&vec![1.0f32; 12], vec![3, 4]).unwrap();
        let w = Tensor::from_slice(&[1.0f32; 4], vec![4]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 12], vec![3, 4]).unwrap();
        let bo = LayerNormBackward {
            x,
            weight: w,
            eps: 1e-6,
        };
        let db = load_f32(bo.apply(&dy).unwrap()[2].as_ref().unwrap()).unwrap();
        approx(&db, &[3.0; 4], 1e-5);
    }

    #[test]
    fn layernorm_backward_bf16_round_trip() {
        let xv: Vec<half::bf16> = (0..4)
            .map(|i| half::bf16::from_f32(i as f32))
            .collect();
        let wv: Vec<half::bf16> = (0..4).map(|_| half::bf16::ONE).collect();
        let dyv: Vec<half::bf16> = (0..4).map(|_| half::bf16::ONE).collect();
        let x = Tensor::from_slice(&xv, vec![1, 4]).unwrap();
        let w = Tensor::from_slice(&wv, vec![4]).unwrap();
        let dy = Tensor::from_slice(&dyv, vec![1, 4]).unwrap();
        let bo = LayerNormBackward {
            x,
            weight: w,
            eps: 1e-6,
        };
        let grads = bo.apply(&dy).unwrap();
        for g in &grads {
            assert_eq!(g.as_ref().unwrap().dtype(), DType::BF16);
        }
    }

    #[test]
    fn op_metadata() {
        let x = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let w = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = LayerNormBackward {
            x,
            weight: w,
            eps: 1e-6,
        };
        assert_eq!(bo.name(), "layer_norm_backward");
        assert_eq!(bo.input_count(), 3);
        assert!(bo.requires_input(0));
        assert!(bo.requires_input(1));
        assert!(!bo.requires_input(2)); // bias not needed by backward
    }
}
