//! `clamp(x, min, max)` and `pow(x, p)` — two more elementwise
//! primitives.
//!
//! - `clamp(x, lo, hi)` = `min(max(x, lo), hi)` — clips values to
//!   `[lo, hi]`. Used for gradient clipping, value clamping, and
//!   safe ReLU-style activations.
//! - `pow(x, p)` = `x^p` — raises every element to a fixed scalar
//!   power. Used in `L_p` losses, RMSE, regularizers, and Newton-
//!   Schulz iteration support.
//!
//! Both ops F32-promote BF16/F16 inputs.
//!
//! # Dispatch
//!
//! - CPU path: byte-wise F32-promoted apply.
//! - CUDA path: dispatches through `cuda_clamp_pow` (one per-element
//!   kernel in `csrc/clamp_pow.cu`) when input is CUDA-resident,
//!   contiguous, and F32/BF16/F16. (#1082)

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// Kind tag for the CUDA kernel. Must match `KIND_CLAMP` / `KIND_POW`
/// in `csrc/clamp_pow.cu`.
#[cfg(feature = "cuda")]
const KIND_CLAMP: i32 = 0;
#[cfg(feature = "cuda")]
const KIND_POW: i32 = 1;

pub fn clamp(x: &Tensor, lo: f32, hi: f32) -> Result<Tensor> {
    if lo > hi {
        bail!("clamp: lo ({lo}) > hi ({hi})");
    }
    #[cfg(feature = "cuda")]
    {
        if matches!(x.device(), crate::Device::Cuda(_))
            && matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
            && x.is_contiguous()
        {
            return crate::cuda_clamp_pow(x, KIND_CLAMP, lo, hi);
        }
    }
    apply_unary(x, |v| v.clamp(lo, hi), "clamp")
}

pub fn pow(x: &Tensor, p: f32) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if matches!(x.device(), crate::Device::Cuda(_))
            && matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
            && x.is_contiguous()
        {
            // `b` is ignored by the POW kind; pass 0.0 for clarity.
            return crate::cuda_clamp_pow(x, KIND_POW, p, 0.0);
        }
    }
    apply_unary(x, |v| v.powf(p), "pow")
}

fn apply_unary(x: &Tensor, f: impl Fn(f32) -> f32, name: &str) -> Result<Tensor> {
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
        .ok_or_else(|| Error::Msg(format!("{name}: storage must be CpuStorage")))?;
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
    fn clamp_clips_to_range() {
        let x = Tensor::from_slice(&[-5.0f32, -1.0, 0.0, 1.0, 5.0], vec![5]).unwrap();
        let y = clamp(&x, -1.0, 1.0).unwrap();
        assert_eq!(read_f32(&y), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn clamp_no_op_when_in_range() {
        let x = Tensor::from_slice(&[0.5f32, -0.5], vec![2]).unwrap();
        let y = clamp(&x, -1.0, 1.0).unwrap();
        assert_eq!(read_f32(&y), vec![0.5, -0.5]);
    }

    #[test]
    fn clamp_lo_gt_hi_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = clamp(&x, 1.0, -1.0).unwrap_err();
        assert!(e.to_string().contains("lo"));
    }

    #[test]
    fn pow_square() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, -4.0], vec![4]).unwrap();
        let y = pow(&x, 2.0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn pow_cube() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, -2.0], vec![3]).unwrap();
        let y = pow(&x, 3.0).unwrap();
        approx(&read_f32(&y), &[1.0, 8.0, -8.0], 1e-5);
    }

    #[test]
    fn pow_one_half() {
        let x = Tensor::from_slice(&[1.0f32, 4.0, 9.0], vec![3]).unwrap();
        let y = pow(&x, 0.5).unwrap();
        approx(&read_f32(&y), &[1.0, 2.0, 3.0], 1e-5);
    }

    #[test]
    fn pow_zero_is_one() {
        // x^0 = 1 for all x != 0; x^0 = 1 for x=0 (Rust's powf returns 1).
        let x = Tensor::from_slice(&[2.0f32, -3.0, 0.0], vec![3]).unwrap();
        let y = pow(&x, 0.0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn clamp_bf16_path() {
        let bf: Vec<half::bf16> = [-2.0f32, 0.0, 2.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![3]).unwrap();
        let y = clamp(&x, -1.0, 1.0).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
    }

    #[test]
    fn shape_preserved() {
        let x = Tensor::from_slice(&[1.0f32; 8], vec![2, 4]).unwrap();
        let y = clamp(&x, 0.0, 1.0).unwrap();
        assert_eq!(y.shape(), &[2, 4]);
    }
}
