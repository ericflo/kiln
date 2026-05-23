//! `sin`, `cos`, `tan` — element-wise trig primitives.
//!
//! Useful for:
//! - **Positional encoding** (sinusoidal embeddings)
//! - **Periodic features** in regression / sequence models
//! - **Custom kernels** that compose with the substrate
//!
//! F32-promoted compute regardless of input dtype.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrigKind {
    Sin,
    Cos,
    Tan,
}

impl TrigKind {
    pub const fn name(self) -> &'static str {
        match self {
            TrigKind::Sin => "sin",
            TrigKind::Cos => "cos",
            TrigKind::Tan => "tan",
        }
    }

    pub fn apply_f32(self, x: f32) -> f32 {
        match self {
            TrigKind::Sin => x.sin(),
            TrigKind::Cos => x.cos(),
            TrigKind::Tan => x.tan(),
        }
    }
}

fn apply(kind: TrigKind, x: &Tensor) -> Result<Tensor> {
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "{}: dtype must be F32/BF16/F16, got {}",
            kind.name(),
            x.dtype()
        );
    }
    if !x.is_contiguous() {
        bail!("{}: input must be contiguous", kind.name());
    }
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
    apply(TrigKind::Sin, x)
}
pub fn cos(x: &Tensor) -> Result<Tensor> {
    apply(TrigKind::Cos, x)
}
pub fn tan(x: &Tensor) -> Result<Tensor> {
    apply(TrigKind::Tan, x)
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
            let y = apply(kind, &x).unwrap();
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
