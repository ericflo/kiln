//! `log2`, `log10`, `log1p`, `exp2`, `expm1` — log/exp variants.
//!
//! Useful for:
//! - **log2 / log10**: information-theoretic quantities (entropy in
//!   bits, ratios in dB)
//! - **log1p**: numerically stable `log(1 + x)` for small `x`
//! - **exp2 / expm1**: inverse counterparts; `expm1` is the
//!   numerically stable `exp(x) - 1`.

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
        .ok_or_else(|| Error::from_str("log_variants: storage must be CpuStorage"))?;
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

pub fn log2(x: &Tensor) -> Result<Tensor> {
    apply(|v| v.log2(), x, "log2")
}
pub fn log10(x: &Tensor) -> Result<Tensor> {
    apply(|v| v.log10(), x, "log10")
}
pub fn log1p(x: &Tensor) -> Result<Tensor> {
    apply(|v| v.ln_1p(), x, "log1p")
}
pub fn exp2(x: &Tensor) -> Result<Tensor> {
    apply(|v| v.exp2(), x, "exp2")
}
pub fn expm1(x: &Tensor) -> Result<Tensor> {
    apply(|v| v.exp_m1(), x, "expm1")
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
    fn log2_powers_of_two() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 4.0, 8.0, 16.0], vec![5]).unwrap();
        let y = read_f32(&log2(&x).unwrap());
        for (i, (g, e)) in y.iter().zip([0.0, 1.0, 2.0, 3.0, 4.0]).enumerate() {
            assert!((g - e).abs() < 1e-5, "idx {i}: got {g}, want {e}");
        }
    }

    #[test]
    fn log10_powers_of_ten() {
        let x = Tensor::from_slice(&[1.0f32, 10.0, 100.0, 1000.0], vec![4]).unwrap();
        let y = read_f32(&log10(&x).unwrap());
        for (i, (g, e)) in y.iter().zip([0.0, 1.0, 2.0, 3.0]).enumerate() {
            assert!((g - e).abs() < 1e-5, "idx {i}: got {g}, want {e}");
        }
    }

    #[test]
    fn log1p_small_values_stable() {
        // log1p(0) = 0; log1p(1) = ln(2); log1p(-0.5) = ln(0.5).
        let x = Tensor::from_slice(&[0.0f32, 1.0, -0.5], vec![3]).unwrap();
        let y = read_f32(&log1p(&x).unwrap());
        assert!(y[0].abs() < 1e-6);
        assert!((y[1] - 2.0f32.ln()).abs() < 1e-5);
        assert!((y[2] - 0.5f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn exp2_powers() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0, -1.0], vec![4]).unwrap();
        let y = read_f32(&exp2(&x).unwrap());
        assert!((y[0] - 1.0).abs() < 1e-6);
        assert!((y[1] - 2.0).abs() < 1e-6);
        assert!((y[2] - 4.0).abs() < 1e-6);
        assert!((y[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn expm1_zero_is_zero() {
        let x = Tensor::from_slice(&[0.0f32, 1.0], vec![2]).unwrap();
        let y = read_f32(&expm1(&x).unwrap());
        assert!(y[0].abs() < 1e-6);
        assert!((y[1] - (std::f32::consts::E - 1.0)).abs() < 1e-5);
    }
}
