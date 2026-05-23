//! `mul_sigmoid_gate` — fused `silu(gate) * up`. The SwiGLU core.
//!
//! Replaces candle's `silu(gate)?.mul(&up)?` pattern at MLP call sites
//! in `forward.rs`. The fused version eliminates the intermediate
//! activation buffer — the packed `silu_mul` kernel from the MLP
//! fusion work (PRs e44c2c84/2a44953a/da1b0467, cited in the issue
//! as the trigger for #1082) is the GPU shape this CPU reference
//! parity-tests against.
//!
//! # Semantics
//!
//! Both inputs share shape + dtype:
//!
//! ```text
//! out[i] = silu(gate[i]) * up[i]
//!        = gate[i] / (1 + exp(-gate[i])) * up[i]
//! ```
//!
//! F32-promoted compute for BF16/F16 inputs.
//!
//! # Determinism
//!
//! `Constructive`. Pointwise; no reduction.

use crate::{
    bail, dispatch2, BackwardOp, CpuStorage, DType, Determinism, DeviceOp2, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// Fused `silu(gate) * up` op.
#[derive(Debug, Default, Clone, Copy)]
pub struct MulSigmoidGateOp;

impl DeviceOp2 for MulSigmoidGateOp {
    fn name(&self) -> &'static str {
        "mul_sigmoid_gate"
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, gate: &Tensor, up: &Tensor) -> Result<Option<Tensor>> {
        if gate.shape() != up.shape() {
            bail!(
                "MulSigmoidGateOp: shape mismatch {:?} vs {:?}",
                gate.shape(),
                up.shape()
            );
        }
        if gate.dtype() != up.dtype() {
            bail!(
                "MulSigmoidGateOp: dtype mismatch {} vs {}",
                gate.dtype(),
                up.dtype()
            );
        }
        if !matches!(gate.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "MulSigmoidGateOp: dtype must be F32/BF16/F16, got {}",
                gate.dtype()
            );
        }
        if !gate.is_contiguous() || !up.is_contiguous() {
            bail!("MulSigmoidGateOp: inputs must be contiguous");
        }

        let dtype = gate.dtype();
        let g_cpu = downcast_cpu(gate, "gate")?;
        let u_cpu = downcast_cpu(up, "up")?;
        let g_bytes = g_cpu.as_bytes();
        let u_bytes = u_cpu.as_bytes();
        let n = gate.element_count();
        let per = dtype.size_in_bytes();
        let mut out = vec![0u8; n * per];

        match dtype {
            DType::F32 => {
                for i in 0..n {
                    let g = f32::from_le_bytes(g_bytes[i * 4..i * 4 + 4].try_into().unwrap());
                    let u = f32::from_le_bytes(u_bytes[i * 4..i * 4 + 4].try_into().unwrap());
                    let y = silu(g) * u;
                    out[i * 4..i * 4 + 4].copy_from_slice(&y.to_le_bytes());
                }
            }
            DType::BF16 => {
                for i in 0..n {
                    let g = half::bf16::from_le_bytes(g_bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32();
                    let u = half::bf16::from_le_bytes(u_bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32();
                    let y = silu(g) * u;
                    out[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes());
                }
            }
            DType::F16 => {
                for i in 0..n {
                    let g = half::f16::from_le_bytes(g_bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32();
                    let u = half::f16::from_le_bytes(u_bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32();
                    let y = silu(g) * u;
                    out[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::f16::from_f32(y).to_le_bytes());
                }
            }
            _ => unreachable!(),
        }

        let cpu = CpuStorage::from_bytes(dtype, out)?;
        let storage: Storage = Arc::new(cpu);
        let t = Tensor::from_parts(
            storage,
            Layout::contiguous(gate.shape().to_vec()),
            TensorId::next(),
        )?;
        Ok(Some(t))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Dispatch `MulSigmoidGateOp`. `out = silu(gate) * up`.
pub fn mul_sigmoid_gate(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    dispatch2(&MulSigmoidGateOp, gate, up)
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn downcast_cpu<'a>(t: &'a Tensor, label: &str) -> Result<&'a CpuStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("MulSigmoidGateOp: {label} storage must be CpuStorage")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol
    }

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
    }

    #[test]
    fn silu_at_zero_is_zero() {
        // silu(0) = 0; silu(0) * anything = 0.
        let gate = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();
        let up = Tensor::from_slice(&[5.0f32, 10.0, 15.0], vec![3]).unwrap();
        let out = mul_sigmoid_gate(&gate, &up).unwrap();
        for v in read_f32(&out) {
            assert!(approx(v, 0.0, 1e-6));
        }
    }

    #[test]
    fn silu_one_times_up_gives_silu_one_scaled() {
        // silu(1) = 1 / (1 + exp(-1)) ≈ 0.7311
        let gate = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let up = Tensor::from_slice(&[10.0f32], vec![1]).unwrap();
        let out = mul_sigmoid_gate(&gate, &up).unwrap();
        let expected = 10.0 / (1.0 + (-1.0_f32).exp());
        assert!(approx(read_f32(&out)[0], expected, 1e-6));
    }

    #[test]
    fn output_preserves_shape() {
        let gate = Tensor::zeros_cpu(vec![2, 3, 4], DType::F32);
        let up = Tensor::zeros_cpu(vec![2, 3, 4], DType::F32);
        let out = mul_sigmoid_gate(&gate, &up).unwrap();
        assert_eq!(out.shape(), &[2, 3, 4]);
    }

    #[test]
    fn bf16_path_within_tolerance() {
        let gv: Vec<half::bf16> = [1.0f32, -1.0, 0.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let uv: Vec<half::bf16> = [2.0f32, 3.0, 5.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let gate = Tensor::from_slice(&gv, vec![3]).unwrap();
        let up = Tensor::from_slice(&uv, vec![3]).unwrap();
        let out = mul_sigmoid_gate(&gate, &up).unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        let cpu = out.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        let expected = [
            (1.0 / (1.0 + (-1.0_f32).exp())) * 2.0,
            (-1.0 / (1.0 + 1.0_f32.exp())) * 3.0,
            0.0,
        ];
        for (i, &e) in expected.iter().enumerate() {
            let v = half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
            assert!(approx(v, e, 1e-2), "[{i}] {v} vs {e}");
        }
    }

    #[test]
    fn rejects_shape_mismatch() {
        let g = Tensor::zeros_cpu(vec![3], DType::F32);
        let u = Tensor::zeros_cpu(vec![4], DType::F32);
        let e = mul_sigmoid_gate(&g, &u).unwrap_err();
        assert!(e.to_string().contains("shape mismatch"));
    }

    #[test]
    fn rejects_dtype_mismatch() {
        let g = Tensor::zeros_cpu(vec![3], DType::F32);
        let u = Tensor::zeros_cpu(vec![3], DType::BF16);
        let e = mul_sigmoid_gate(&g, &u).unwrap_err();
        assert!(e.to_string().contains("dtype mismatch"));
    }

    #[test]
    fn op_metadata() {
        let op = MulSigmoidGateOp;
        assert_eq!(op.name(), "mul_sigmoid_gate");
        assert!(op.determinism().is_constructive());
        assert!(op.bwd().is_none());
    }
}
