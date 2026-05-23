//! `sinh`, `cosh` — hyperbolic primitives.
//!
//! `tanh` is in `activation.rs` because it's used as a nonlinearity.
//! These two complete the hyperbolic family for general math use.
//!
//! ```text
//! sinh(x) = (e^x - e^-x) / 2
//! cosh(x) = (e^x + e^-x) / 2
//! ```

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

fn apply(f: impl Fn(f32) -> f32, x: &Tensor, name: &str) -> Result<Tensor> {
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("{name}: input must be contiguous");
    }
    let dtype = x.dtype();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("hyperbolic: storage must be CpuStorage"))?;
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
        let y = f(v);
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

pub fn sinh(x: &Tensor) -> Result<Tensor> {
    apply(|v| v.sinh(), x, "sinh")
}

pub fn cosh(x: &Tensor) -> Result<Tensor> {
    apply(|v| v.cosh(), x, "cosh")
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

    #[test]
    fn sinh_known_values() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, -1.0], vec![3]).unwrap();
        let y = read_f32(&sinh(&x).unwrap());
        assert!(y[0].abs() < 1e-6);
        assert!((y[1] - 1.1752).abs() < 1e-3);
        assert!((y[2] + 1.1752).abs() < 1e-3);
    }

    #[test]
    fn cosh_known_values() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, -1.0], vec![3]).unwrap();
        let y = read_f32(&cosh(&x).unwrap());
        assert!((y[0] - 1.0).abs() < 1e-6);
        assert!((y[1] - 1.5431).abs() < 1e-3);
        assert!((y[2] - 1.5431).abs() < 1e-3); // cosh is even
    }

    #[test]
    fn hyperbolic_identity() {
        // cosh²(x) - sinh²(x) = 1
        let x = Tensor::from_slice(&[0.3f32, 1.5, -2.7], vec![3]).unwrap();
        let s = read_f32(&sinh(&x).unwrap());
        let c = read_f32(&cosh(&x).unwrap());
        for i in 0..3 {
            let id = c[i] * c[i] - s[i] * s[i];
            assert!((id - 1.0).abs() < 1e-4, "id={id}");
        }
    }

    #[test]
    fn bf16_round_trips() {
        let bf: Vec<half::bf16> = [0.0f32, 1.0].iter().map(|&v| half::bf16::from_f32(v)).collect();
        let x = Tensor::from_slice(&bf, vec![2]).unwrap();
        assert_eq!(sinh(&x).unwrap().dtype(), DType::BF16);
        assert_eq!(cosh(&x).unwrap().dtype(), DType::BF16);
    }
}
