//! `MulSigmoidGateBackward` — gradient of SwiGLU's core fused op.
//!
//! Forward (from `kiln_tensor::ops::mul_sigmoid_gate`):
//!
//! ```text
//! out[i] = silu(gate[i]) * up[i]
//!        = gate[i] * sigmoid(gate[i]) * up[i]
//! ```
//!
//! # Backward derivation
//!
//! Let `σ = sigmoid(gate)`. Then `silu(gate) = gate * σ` and:
//!
//! - `∂silu/∂gate = σ + gate * σ * (1 - σ)`
//! - `∂out/∂gate  = up * ∂silu/∂gate`
//! - `∂out/∂up    = silu(gate) = gate * σ`
//!
//! Per element:
//!
//! ```text
//! d_gate = dy * up * (σ + gate * σ * (1 - σ))
//! d_up   = dy * gate * σ
//! ```

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct MulSigmoidGateBackward {
    /// Saved `gate` from the forward pass.
    pub gate: Tensor,
    /// Saved `up` from the forward pass.
    pub up: Tensor,
}

impl BackwardOp for MulSigmoidGateBackward {
    fn name(&self) -> &'static str {
        "mul_sigmoid_gate_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if self.gate.shape() != self.up.shape() {
            bail!(
                "MulSigmoidGateBackward: gate shape {:?} != up shape {:?}",
                self.gate.shape(),
                self.up.shape()
            );
        }
        if self.gate.shape() != grad_output.shape() {
            bail!(
                "MulSigmoidGateBackward: grad_output shape {:?} != gate shape {:?}",
                grad_output.shape(),
                self.gate.shape()
            );
        }
        let dtype = self.gate.dtype();
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "MulSigmoidGateBackward: dtype must be F32/BF16/F16, got {dtype}"
            );
        }
        if self.up.dtype() != dtype || grad_output.dtype() != dtype {
            bail!("MulSigmoidGateBackward: dtype mismatch among saved tensors and grad");
        }

        let gate = load_f32(&self.gate)?;
        let up = load_f32(&self.up)?;
        let dy = load_f32(grad_output)?;

        let n = gate.len();
        let mut d_gate = vec![0.0f32; n];
        let mut d_up = vec![0.0f32; n];
        for i in 0..n {
            let s = 1.0 / (1.0 + (-gate[i]).exp());
            let dsilu_dgate = s + gate[i] * s * (1.0 - s);
            d_gate[i] = dy[i] * up[i] * dsilu_dgate;
            d_up[i] = dy[i] * gate[i] * s;
        }
        let dg = store_f32(dtype, self.gate.shape(), &d_gate)?;
        let du = store_f32(dtype, self.up.shape(), &d_up)?;
        Ok(vec![Some(dg), Some(du)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
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
        .ok_or_else(|| Error::from_str("swiglu_backward: storage must be CpuStorage"))?;
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
    use kiln_tensor::ops::mul_sigmoid_gate;

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
    fn swiglu_backward_at_zero_gate() {
        // gate = 0 → σ = 0.5, silu = 0.
        // d_gate = dy * up * (0.5 + 0) = dy * up * 0.5.
        // d_up   = dy * 0 * 0.5 = 0.
        let gate = Tensor::from_slice(&[0.0f32, 0.0], vec![2]).unwrap();
        let up = Tensor::from_slice(&[2.0f32, 4.0], vec![2]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let bo = MulSigmoidGateBackward {
            gate: gate.clone(),
            up: up.clone(),
        };
        let grads = bo.apply(&dy).unwrap();
        approx(&load_f32(grads[0].as_ref().unwrap()).unwrap(), &[1.0, 2.0], 1e-6);
        approx(&load_f32(grads[1].as_ref().unwrap()).unwrap(), &[0.0, 0.0], 1e-6);
    }

    #[test]
    fn swiglu_backward_finite_difference() {
        let g_data = vec![1.0f32, -0.5, 2.0];
        let u_data = vec![0.7f32, 1.3, -0.4];
        let gate = Tensor::from_slice(&g_data, vec![3]).unwrap();
        let up = Tensor::from_slice(&u_data, vec![3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 3], vec![3]).unwrap();
        let bo = MulSigmoidGateBackward {
            gate: gate.clone(),
            up: up.clone(),
        };
        let grads = bo.apply(&dy).unwrap();
        let dg = load_f32(grads[0].as_ref().unwrap()).unwrap();
        let du = load_f32(grads[1].as_ref().unwrap()).unwrap();

        // loss(gate, up) = sum(mul_sigmoid_gate(gate, up)). dy=ones.
        let loss = |g_vec: &[f32], u_vec: &[f32]| -> f32 {
            let gt = Tensor::from_slice(g_vec, vec![3]).unwrap();
            let ut = Tensor::from_slice(u_vec, vec![3]).unwrap();
            let y = mul_sigmoid_gate(&gt, &ut).unwrap();
            load_f32(&y).unwrap().iter().sum()
        };
        let step = 1e-3;
        let mut fd_g = Vec::with_capacity(3);
        let mut fd_u = Vec::with_capacity(3);
        for i in 0..3 {
            let mut up_g = g_data.clone();
            up_g[i] += step;
            let mut dn_g = g_data.clone();
            dn_g[i] -= step;
            fd_g.push((loss(&up_g, &u_data) - loss(&dn_g, &u_data)) / (2.0 * step));
            let mut up_u = u_data.clone();
            up_u[i] += step;
            let mut dn_u = u_data.clone();
            dn_u[i] -= step;
            fd_u.push((loss(&g_data, &up_u) - loss(&g_data, &dn_u)) / (2.0 * step));
        }
        approx(&dg, &fd_g, 1e-3);
        approx(&du, &fd_u, 1e-3);
    }

    #[test]
    fn swiglu_backward_shape_mismatch_errors() {
        let gate = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let up = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = MulSigmoidGateBackward { gate, up };
        let dy = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = bo.apply(&dy).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    #[test]
    fn op_metadata() {
        let g = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let u = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = MulSigmoidGateBackward { gate: g, up: u };
        assert_eq!(bo.name(), "mul_sigmoid_gate_backward");
        assert_eq!(bo.input_count(), 2);
        assert!(bo.requires_input(0));
        assert!(bo.requires_input(1));
    }
}
