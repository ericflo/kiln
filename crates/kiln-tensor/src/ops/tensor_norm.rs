//! `tensor_norm` — full-tensor scalar norms (L1, L2, L_inf, L_p).
//!
//! Differs from existing per-axis `vector_norm` (reduces along
//! configured axis) and `clip_grad_norm` (joint norm across a list).
//! These reduce the *entire* input to a single scalar `Tensor`.
//!
//! Used for: distance receipts in eval loops, A/B sampler stats,
//! receipts that print "‖θ - θ_ref‖_2" on a parameter drift check.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

/// Materialize `t` on CPU. If `t` already lives on CPU this is a
/// cheap `Arc` bump; if it lives on CUDA we do a D2H copy. The norm
/// ops are full-tensor scalar reductions used in eval / receipts /
/// diagnostics — not on any inner training hot path — so reading
/// back to host is the obvious correct shape today, and a fused
/// `cuda_*_norm_full` kernel can land later without touching the
/// public API. See `#1082`.
fn to_cpu(t: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if matches!(t.device(), crate::Device::Cuda(_)) {
            return crate::cuda_to_host_copy(t);
        }
    }
    Ok(t.clone())
}

fn read_f32_flat(t: &Tensor) -> Result<Vec<f32>> {
    if !t.is_contiguous() {
        bail!("tensor_norm: input must be contiguous");
    }
    if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("tensor_norm: dtype must be F32/BF16/F16, got {}", t.dtype());
    }
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("tensor_norm: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let mut out = Vec::with_capacity(n);
    match t.dtype() {
        DType::F32 => {
            for i in 0..n {
                out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
            }
        }
        DType::BF16 => {
            for i in 0..n {
                out.push(
                    half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                );
            }
        }
        DType::F16 => {
            for i in 0..n {
                out.push(
                    half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                );
            }
        }
        _ => unreachable!(),
    }
    Ok(out)
}

fn scalar_tensor(dtype: DType, v: f32) -> Result<Tensor> {
    let bytes = match dtype {
        DType::F32 => v.to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f32(v).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f32(v).to_le_bytes().to_vec(),
        _ => unreachable!(),
    };
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(Vec::<usize>::new()), TensorId::next())
}

/// L1 norm: `Σ |x_i|`.
pub fn l1_norm(t: &Tensor) -> Result<Tensor> {
    let host = to_cpu(t)?;
    let v = read_f32_flat(&host)?;
    if v.is_empty() {
        bail!("l1_norm: empty input");
    }
    let s: f32 = v.iter().map(|x| x.abs()).sum();
    scalar_tensor(t.dtype(), s)
}

/// L2 norm: `√(Σ x_i²)`.
pub fn l2_norm_scalar(t: &Tensor) -> Result<Tensor> {
    let host = to_cpu(t)?;
    let v = read_f32_flat(&host)?;
    if v.is_empty() {
        bail!("l2_norm_scalar: empty input");
    }
    let s: f32 = v.iter().map(|x| x * x).sum();
    scalar_tensor(t.dtype(), s.sqrt())
}

/// L_∞ norm: `max |x_i|`.
pub fn linf_norm(t: &Tensor) -> Result<Tensor> {
    let host = to_cpu(t)?;
    let v = read_f32_flat(&host)?;
    if v.is_empty() {
        bail!("linf_norm: empty input");
    }
    let m: f32 = v
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, |acc, x| if x > acc { x } else { acc });
    scalar_tensor(t.dtype(), m)
}

/// L_p norm: `(Σ |x_i|^p)^(1/p)`, `p > 0`.
pub fn lp_norm(t: &Tensor, p: f32) -> Result<Tensor> {
    if !(p > 0.0 && p.is_finite()) {
        bail!("lp_norm: p must be finite and > 0, got {p}");
    }
    let host = to_cpu(t)?;
    let v = read_f32_flat(&host)?;
    if v.is_empty() {
        bail!("lp_norm: empty input");
    }
    let s: f32 = v.iter().map(|x| x.abs().powf(p)).sum();
    scalar_tensor(t.dtype(), s.powf(1.0 / p))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(t: &Tensor) -> f32 {
        let cpu = t.storage();
        let cpu = cpu.as_any().downcast_ref::<CpuStorage>().unwrap();
        f32::from_le_bytes(cpu.as_bytes()[..4].try_into().unwrap())
    }

    #[test]
    fn l1_basic() {
        let x = Tensor::from_slice(&[1.0f32, -2.0, 3.0, -4.0], vec![4]).unwrap();
        assert!((scalar(&l1_norm(&x).unwrap()) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn l2_basic() {
        // sqrt(1 + 4 + 9 + 16) = sqrt(30)
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let v = scalar(&l2_norm_scalar(&x).unwrap());
        assert!((v - 30.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn linf_basic() {
        let x = Tensor::from_slice(&[1.0f32, -7.0, 2.0, -3.0], vec![4]).unwrap();
        assert!((scalar(&linf_norm(&x).unwrap()) - 7.0).abs() < 1e-5);
    }

    #[test]
    fn lp_matches_l2_at_p2() {
        let x = Tensor::from_slice(&[3.0f32, 4.0], vec![2]).unwrap();
        let lp = scalar(&lp_norm(&x, 2.0).unwrap());
        let l2 = scalar(&l2_norm_scalar(&x).unwrap());
        assert!((lp - l2).abs() < 1e-5);
    }

    #[test]
    fn lp_matches_l1_at_p1() {
        let x = Tensor::from_slice(&[-1.0f32, 2.0, -3.0], vec![3]).unwrap();
        let lp = scalar(&lp_norm(&x, 1.0).unwrap());
        let l1 = scalar(&l1_norm(&x).unwrap());
        assert!((lp - l1).abs() < 1e-5);
    }

    #[test]
    fn norms_work_on_2d() {
        // 2D tensor reduces to one scalar (full reduction).
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert!((scalar(&l1_norm(&x).unwrap()) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn empty_input_errors() {
        let x = Tensor::from_slice::<f32>(&[], vec![0]).unwrap();
        let e = l1_norm(&x).unwrap_err();
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn lp_invalid_p_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = lp_norm(&x, 0.0).unwrap_err();
        assert!(e.to_string().contains("> 0"));
    }

    /// CUDA parity: ensure CUDA-resident tensors produce the same
    /// scalar norm as their CPU equivalents. The `to_cpu` helper
    /// transparently D2H-copies on entry so call sites that lift a
    /// CPU tensor onto CUDA via `to_device` keep working.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_l1_l2_linf_lp_parity() {
        let cdev = match candle_core::Device::cuda_if_available(0) {
            Ok(candle_core::Device::Cuda(c)) => c,
            _ => return,
        };
        let cdev = std::sync::Arc::new(cdev);

        let cpu_x = Tensor::from_slice(&[1.0f32, -2.0, 3.0, -4.0], vec![4]).unwrap();
        let cuda_x = cpu_x.to_device(crate::Device::Cuda(0), Some(cdev.clone())).unwrap();

        let l1c = scalar(&l1_norm(&cpu_x).unwrap());
        let l1g = scalar(&l1_norm(&cuda_x).unwrap());
        assert!((l1c - l1g).abs() < 1e-5, "l1 parity: cpu={l1c}, cuda={l1g}");

        let l2c = scalar(&l2_norm_scalar(&cpu_x).unwrap());
        let l2g = scalar(&l2_norm_scalar(&cuda_x).unwrap());
        assert!((l2c - l2g).abs() < 1e-5, "l2 parity: cpu={l2c}, cuda={l2g}");

        let linfc = scalar(&linf_norm(&cpu_x).unwrap());
        let linfg = scalar(&linf_norm(&cuda_x).unwrap());
        assert!(
            (linfc - linfg).abs() < 1e-5,
            "linf parity: cpu={linfc}, cuda={linfg}"
        );

        let lpc = scalar(&lp_norm(&cpu_x, 3.0).unwrap());
        let lpg = scalar(&lp_norm(&cuda_x, 3.0).unwrap());
        assert!((lpc - lpg).abs() < 1e-4, "lp parity: cpu={lpc}, cuda={lpg}");
    }
}
