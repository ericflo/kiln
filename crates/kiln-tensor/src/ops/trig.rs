//! `sin`, `cos`, `tan` — element-wise trig primitives.
//!
//! Useful for:
//! - **Positional encoding** (sinusoidal embeddings)
//! - **Periodic features** in regression / sequence models
//! - **Custom kernels** that compose with the substrate
//!
//! F32-promoted compute regardless of input dtype.

use std::sync::Arc;

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, DType, Determinism, DeviceOp1, Error, Layout, Result,
    Storage, Tensor, TensorId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrigKind {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
}

impl TrigKind {
    pub const fn name(self) -> &'static str {
        match self {
            TrigKind::Sin => "sin",
            TrigKind::Cos => "cos",
            TrigKind::Tan => "tan",
            TrigKind::Asin => "asin",
            TrigKind::Acos => "acos",
            TrigKind::Atan => "atan",
        }
    }

    pub fn apply_f32(self, x: f32) -> f32 {
        match self {
            TrigKind::Sin => x.sin(),
            TrigKind::Cos => x.cos(),
            TrigKind::Tan => x.tan(),
            TrigKind::Asin => x.asin(),
            TrigKind::Acos => x.acos(),
            TrigKind::Atan => x.atan(),
        }
    }

    /// CUDA kernel kind tag. Only sin/cos/tan are wired to the shared
    /// activation kernel today (#1082). The inverse trig ops (asin,
    /// acos, atan) stay CPU-only until a follow-up.
    #[cfg(feature = "cuda")]
    fn cuda_kind_tag(self) -> Option<i32> {
        match self {
            TrigKind::Sin => Some(7),
            TrigKind::Cos => Some(8),
            TrigKind::Tan => Some(9),
            TrigKind::Asin | TrigKind::Acos | TrigKind::Atan => None,
        }
    }
}

/// `DeviceOp1` adapter for the trig family. CUDA path routes the
/// forward sin/cos/tan through the shared `cuda_activation_unary`
/// kernel; asin/acos/atan fall through to CPU (#1082).
#[derive(Debug, Clone, Copy)]
struct TrigOp {
    kind: TrigKind,
}

impl DeviceOp1 for TrigOp {
    fn name(&self) -> &'static str {
        self.kind.name()
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.kind.name())?;
        let t = cpu_apply(self.kind, x)?;
        Ok(Some(t))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.kind.name())?;
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        match self.kind.cuda_kind_tag() {
            Some(tag) => Ok(Some(crate::cuda_activation_unary(x, tag)?)),
            None => Ok(None),
        }
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

fn validate(x: &Tensor, name: &str) -> Result<()> {
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("{name}: input must be contiguous");
    }
    Ok(())
}

fn cpu_apply(kind: TrigKind, x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("trig: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = x.element_count();
    let per = dtype.size_in_bytes();
    let mut out = vec![0u8; n * per];
    for i in 0..n {
        let v = match dtype {
            DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            DType::F16 => half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            _ => unreachable!(),
        };
        let y = kind.apply_f32(v);
        match dtype {
            DType::F32 => out[i * 4..i * 4 + 4].copy_from_slice(&y.to_le_bytes()),
            DType::BF16 => out[i * 2..i * 2 + 2]
                .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes()),
            DType::F16 => out[i * 2..i * 2 + 2]
                .copy_from_slice(&half::f16::from_f32(y).to_le_bytes()),
            _ => unreachable!(),
        }
    }
    let cpu_out = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(x.shape().to_vec()), TensorId::next())
}

pub fn sin(x: &Tensor) -> Result<Tensor> {
    dispatch1(&TrigOp { kind: TrigKind::Sin }, x)
}
pub fn cos(x: &Tensor) -> Result<Tensor> {
    dispatch1(&TrigOp { kind: TrigKind::Cos }, x)
}
pub fn tan(x: &Tensor) -> Result<Tensor> {
    dispatch1(&TrigOp { kind: TrigKind::Tan }, x)
}
pub fn asin(x: &Tensor) -> Result<Tensor> {
    dispatch1(&TrigOp { kind: TrigKind::Asin }, x)
}
pub fn acos(x: &Tensor) -> Result<Tensor> {
    dispatch1(&TrigOp { kind: TrigKind::Acos }, x)
}
pub fn atan(x: &Tensor) -> Result<Tensor> {
    dispatch1(&TrigOp { kind: TrigKind::Atan }, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

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
    fn sin_known_values() {
        // sin(0)=0, sin(π/2)=1, sin(π)≈0, sin(-π/2)=-1
        let x = Tensor::from_slice(
            &[
                0.0_f32,
                std::f32::consts::FRAC_PI_2,
                std::f32::consts::PI,
                -std::f32::consts::FRAC_PI_2,
            ],
            vec![4],
        )
        .unwrap();
        let y = read_f32(&sin(&x).unwrap());
        approx(&y, &[0.0, 1.0, 0.0, -1.0], 1e-5);
    }

    #[test]
    fn cos_known_values() {
        // cos(0)=1, cos(π/2)≈0, cos(π)=-1, cos(-π/2)≈0
        let x = Tensor::from_slice(
            &[
                0.0_f32,
                std::f32::consts::FRAC_PI_2,
                std::f32::consts::PI,
                -std::f32::consts::FRAC_PI_2,
            ],
            vec![4],
        )
        .unwrap();
        let y = read_f32(&cos(&x).unwrap());
        approx(&y, &[1.0, 0.0, -1.0, 0.0], 1e-5);
    }

    #[test]
    fn tan_known_values() {
        // tan(0)=0, tan(π/4)=1, tan(-π/4)=-1
        let x = Tensor::from_slice(
            &[0.0_f32, std::f32::consts::FRAC_PI_4, -std::f32::consts::FRAC_PI_4],
            vec![3],
        )
        .unwrap();
        let y = read_f32(&tan(&x).unwrap());
        approx(&y, &[0.0, 1.0, -1.0], 1e-5);
    }

    #[test]
    fn pythagorean_identity_holds() {
        // sin²(x) + cos²(x) = 1.
        let x = Tensor::from_slice(&[0.3_f32, 1.2, -2.5, 4.0], vec![4]).unwrap();
        let s = read_f32(&sin(&x).unwrap());
        let c = read_f32(&cos(&x).unwrap());
        for i in 0..4 {
            let id = s[i] * s[i] + c[i] * c[i];
            assert!((id - 1.0).abs() < 1e-5, "i={i}: id={id}");
        }
    }

    #[test]
    fn bf16_round_trips_through_each() {
        for kind in [TrigKind::Sin, TrigKind::Cos, TrigKind::Tan] {
            let bf: Vec<half::bf16> = [0.0f32, 1.0, -1.0]
                .iter()
                .map(|&v| half::bf16::from_f32(v))
                .collect();
            let x = Tensor::from_slice(&bf, vec![3]).unwrap();
            let y = match kind {
                TrigKind::Sin => sin(&x).unwrap(),
                TrigKind::Cos => cos(&x).unwrap(),
                TrigKind::Tan => tan(&x).unwrap(),
                _ => unreachable!(),
            };
            assert_eq!(y.dtype(), DType::BF16);
        }
    }

    #[test]
    fn kind_names() {
        assert_eq!(TrigKind::Sin.name(), "sin");
        assert_eq!(TrigKind::Cos.name(), "cos");
        assert_eq!(TrigKind::Tan.name(), "tan");
    }
}
