//! `sign`, `floor`, `ceil`, `round`, `trunc`, `reciprocal`.
//!
//! Elementwise primitives. All non-differentiable in the
//! mathematical sense (piecewise constant or sparse-derivative),
//! so no BackwardOp.

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
        .ok_or_else(|| Error::from_str("sign_and_round: storage must be CpuStorage"))?;
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

pub fn sign(x: &Tensor) -> Result<Tensor> {
    apply(
        |v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        },
        x,
        "sign",
    )
}

pub fn floor(x: &Tensor) -> Result<Tensor> {
    apply(|v| v.floor(), x, "floor")
}

pub fn ceil(x: &Tensor) -> Result<Tensor> {
    apply(|v| v.ceil(), x, "ceil")
}

pub fn round(x: &Tensor) -> Result<Tensor> {
    apply(|v| v.round(), x, "round")
}

pub fn trunc(x: &Tensor) -> Result<Tensor> {
    apply(|v| v.trunc(), x, "trunc")
}

pub fn reciprocal(x: &Tensor) -> Result<Tensor> {
    apply(|v| 1.0 / v, x, "reciprocal")
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
    fn sign_three_way() {
        let x = Tensor::from_slice(&[-3.0f32, -0.5, 0.0, 0.5, 3.0], vec![5]).unwrap();
        assert_eq!(read_f32(&sign(&x).unwrap()), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn floor_ceil_round_trunc() {
        let x = Tensor::from_slice(&[1.3f32, -1.3, 2.7, -2.7, 0.5, -0.5], vec![6]).unwrap();
        let f = read_f32(&floor(&x).unwrap());
        let c = read_f32(&ceil(&x).unwrap());
        let r = read_f32(&round(&x).unwrap());
        let t = read_f32(&trunc(&x).unwrap());
        assert_eq!(f, vec![1.0, -2.0, 2.0, -3.0, 0.0, -1.0]);
        assert_eq!(c, vec![2.0, -1.0, 3.0, -2.0, 1.0, 0.0]);
        // Rust's round() rounds half away from zero.
        assert_eq!(r, vec![1.0, -1.0, 3.0, -3.0, 1.0, -1.0]);
        assert_eq!(t, vec![1.0, -1.0, 2.0, -2.0, 0.0, 0.0]);
    }

    #[test]
    fn reciprocal_inverts() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 4.0, -0.5], vec![4]).unwrap();
        let y = read_f32(&reciprocal(&x).unwrap());
        assert!((y[0] - 1.0).abs() < 1e-6);
        assert!((y[1] - 0.5).abs() < 1e-6);
        assert!((y[2] - 0.25).abs() < 1e-6);
        assert!((y[3] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn bf16_round_trips() {
        let bf: Vec<half::bf16> = [1.5f32, -1.5]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![2]).unwrap();
        assert_eq!(floor(&x).unwrap().dtype(), DType::BF16);
        assert_eq!(sign(&x).unwrap().dtype(), DType::BF16);
    }
}
