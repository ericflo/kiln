//! Single-input activation ops: `silu`, `sigmoid`.
//!
//! Replaces candle's `Tensor::{silu, sigmoid}` and the candle
//! `Activation::SiLU` enum dispatch.
//!
//! # Semantics
//!
//! Both ops are pointwise:
//!
//! - `sigmoid(x) = 1 / (1 + exp(-x))`
//! - `silu(x) = x * sigmoid(x)` (a.k.a. swish)
//!
//! F32-promoted compute for BF16/F16 inputs.
//!
//! # Determinism
//!
//! `Constructive`. Pointwise; no reduction.

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, DType, Determinism, DeviceOp1, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// Unary pointwise activation kind. Selected by [`ActivationOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryKind {
    /// `f(x) = x / (1 + exp(-x))`
    Silu,
    /// `f(x) = 1 / (1 + exp(-x))`
    Sigmoid,
}

impl UnaryKind {
    pub const fn name(self) -> &'static str {
        match self {
            UnaryKind::Silu => "silu",
            UnaryKind::Sigmoid => "sigmoid",
        }
    }

    fn apply_f32(self, x: f32) -> f32 {
        match self {
            UnaryKind::Silu => x / (1.0 + (-x).exp()),
            UnaryKind::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }
}

/// Activation op handle.
#[derive(Debug, Clone, Copy)]
pub struct ActivationOp {
    kind: UnaryKind,
}

impl ActivationOp {
    pub const fn new(kind: UnaryKind) -> Self {
        ActivationOp { kind }
    }
    pub const fn kind(self) -> UnaryKind {
        self.kind
    }
}

impl DeviceOp1 for ActivationOp {
    fn name(&self) -> &'static str {
        self.kind.name()
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.kind)?;
        let dtype = x.dtype();
        let x_cpu = downcast_cpu(x, "x")?;
        let x_bytes = x_cpu.as_bytes();
        let n = x.element_count();
        let per = dtype.size_in_bytes();
        let mut out = vec![0u8; n * per];
        match dtype {
            DType::F32 => {
                for i in 0..n {
                    let v = f32::from_le_bytes(x_bytes[i * 4..i * 4 + 4].try_into().unwrap());
                    let y = self.kind.apply_f32(v);
                    out[i * 4..i * 4 + 4].copy_from_slice(&y.to_le_bytes());
                }
            }
            DType::BF16 => {
                for i in 0..n {
                    let v = half::bf16::from_le_bytes(x_bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32();
                    let y = self.kind.apply_f32(v);
                    out[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes());
                }
            }
            DType::F16 => {
                for i in 0..n {
                    let v = half::f16::from_le_bytes(x_bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32();
                    let y = self.kind.apply_f32(v);
                    out[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::f16::from_f32(y).to_le_bytes());
                }
            }
            other => bail!("ActivationOp({}): dtype {other} not supported", self.kind.name()),
        }
        let cpu = CpuStorage::from_bytes(dtype, out)?;
        let storage: Storage = Arc::new(cpu);
        let layout = Layout::contiguous(x.shape().to_vec());
        let t = Tensor::from_parts(storage, layout, TensorId::next())?;
        Ok(Some(t))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// `out = x / (1 + exp(-x))` — sigmoid-weighted linear unit.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    dispatch1(&ActivationOp::new(UnaryKind::Silu), x)
}

/// `out = 1 / (1 + exp(-x))`.
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    dispatch1(&ActivationOp::new(UnaryKind::Sigmoid), x)
}

fn validate(x: &Tensor, kind: UnaryKind) -> Result<()> {
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "ActivationOp({}): dtype must be F32/BF16/F16, got {}",
            kind.name(),
            x.dtype()
        );
    }
    if !x.is_contiguous() {
        bail!("ActivationOp({}): input must be contiguous", kind.name());
    }
    Ok(())
}

fn downcast_cpu<'a>(t: &'a Tensor, label: &str) -> Result<&'a CpuStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("ActivationOp: {label} storage must be CpuStorage")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
    }

    fn approx_eq(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol
    }

    #[test]
    fn sigmoid_f32_known_values() {
        let x = Tensor::from_slice(&[-10.0f32, 0.0, 10.0], vec![3]).unwrap();
        let y = sigmoid(&x).unwrap();
        let got = read_f32(&y);
        // sigmoid(-10) ≈ 4.54e-5, sigmoid(0) = 0.5, sigmoid(10) ≈ 0.99995
        assert!(approx_eq(got[0], 4.5398e-5, 1e-6));
        assert!(approx_eq(got[1], 0.5, 1e-6));
        assert!(approx_eq(got[2], 1.0 - 4.5398e-5, 1e-6));
    }

    #[test]
    fn silu_f32_known_values() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, -1.0], vec![3]).unwrap();
        let y = silu(&x).unwrap();
        let got = read_f32(&y);
        // silu(0) = 0; silu(1) = 1 * sigmoid(1) ≈ 0.7311; silu(-1) ≈ -0.2689
        assert!(approx_eq(got[0], 0.0, 1e-6));
        assert!(approx_eq(got[1], 1.0 / (1.0 + (-1.0_f32).exp()), 1e-6));
        assert!(approx_eq(got[2], -1.0 / (1.0 + 1.0_f32.exp()), 1e-6));
    }

    #[test]
    fn sigmoid_bf16_within_tolerance() {
        let xv: Vec<half::bf16> = [0.0f32, 1.0, -1.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&xv, vec![3]).unwrap();
        let y = sigmoid(&x).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        let cpu = y.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        let expected = [0.5_f32, 0.7311, 0.2689];
        for (i, &e) in expected.iter().enumerate() {
            let v = half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
            assert!(approx_eq(v, e, 1e-2), "bf16 sigmoid[{i}]={v}, expected {e}");
        }
    }

    #[test]
    fn silu_preserves_shape_and_dtype() {
        let x = Tensor::zeros_cpu(vec![2, 3, 4], DType::F32);
        let y = silu(&x).unwrap();
        assert_eq!(y.shape(), &[2, 3, 4]);
        assert_eq!(y.dtype(), DType::F32);
    }

    #[test]
    fn rejects_bad_dtype() {
        let x = Tensor::from_slice(&[1u32, 2, 3], vec![3]).unwrap();
        let e = silu(&x).unwrap_err();
        assert!(e.to_string().contains("F32/BF16/F16"));
    }

    #[test]
    fn unary_kind_name_strings() {
        assert_eq!(UnaryKind::Silu.name(), "silu");
        assert_eq!(UnaryKind::Sigmoid.name(), "sigmoid");
    }

    #[test]
    fn op_metadata() {
        let op = ActivationOp::new(UnaryKind::Silu);
        assert_eq!(op.name(), "silu");
        assert!(op.determinism().is_constructive());
        assert!(op.bwd().is_none());
    }
}
