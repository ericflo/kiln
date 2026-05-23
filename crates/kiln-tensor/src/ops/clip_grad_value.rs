//! `clip_grad_value` — clip every gradient element into `[-c, c]`.
//!
//! Mirrors PyTorch's `torch.nn.utils.clip_grad_value_(parameters,
//! clip_value)`. Companion to `clip_grad_norm`: instead of rescaling
//! by a joint L2-norm, clamps each element independently. Cheaper
//! and bounds-aware per-element; useful when individual gradient
//! spikes (not joint magnitude) are the concern.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

fn clip_one(g: &Tensor, c: f32) -> Result<Tensor> {
    let dtype = g.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        bail!("clip_grad_value: dtype must be F32/BF16/F16, got {dtype}");
    }
    if !g.is_contiguous() {
        bail!("clip_grad_value: input must be contiguous");
    }
    let n = g.element_count();
    let cpu = g
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("clip_grad_value: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let mut out = vec![0u8; bytes.len()];
    let lo = -c;
    let hi = c;
    match dtype {
        DType::F32 => {
            for i in 0..n {
                let v = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                let cv = v.clamp(lo, hi);
                out[i * 4..i * 4 + 4].copy_from_slice(&cv.to_le_bytes());
            }
        }
        DType::BF16 => {
            for i in 0..n {
                let v = half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                    .to_f32();
                let cv = v.clamp(lo, hi);
                out[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(cv).to_le_bytes());
            }
        }
        DType::F16 => {
            for i in 0..n {
                let v = half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                    .to_f32();
                let cv = v.clamp(lo, hi);
                out[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(cv).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu_out = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(g.shape().to_vec()), TensorId::next())
}

/// Clip every element of every gradient into `[-clip_value,
/// clip_value]`. `clip_value` must be > 0.
pub fn clip_grad_value(grads: &[&Tensor], clip_value: f32) -> Result<Vec<Tensor>> {
    if grads.is_empty() {
        bail!("clip_grad_value: at least one gradient required");
    }
    if clip_value <= 0.0 {
        bail!(
            "clip_grad_value: clip_value must be > 0, got {clip_value}"
        );
    }
    let mut out = Vec::with_capacity(grads.len());
    for g in grads {
        out.push(clip_one(g, clip_value)?);
    }
    Ok(out)
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
    fn clip_inside_window_is_identity() {
        let g = Tensor::from_slice(&[0.1f32, -0.2, 0.3], vec![3]).unwrap();
        let out = clip_grad_value(&[&g], 1.0).unwrap();
        assert_eq!(read_f32(&out[0]), vec![0.1, -0.2, 0.3]);
    }

    #[test]
    fn clip_clamps_positive_overflow() {
        let g = Tensor::from_slice(&[2.0f32, 5.0, 100.0], vec![3]).unwrap();
        let out = clip_grad_value(&[&g], 1.0).unwrap();
        assert_eq!(read_f32(&out[0]), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn clip_clamps_negative_overflow() {
        let g = Tensor::from_slice(&[-2.0f32, -5.0, -100.0], vec![3]).unwrap();
        let out = clip_grad_value(&[&g], 1.0).unwrap();
        assert_eq!(read_f32(&out[0]), vec![-1.0, -1.0, -1.0]);
    }

    #[test]
    fn clip_mixed_values() {
        let g = Tensor::from_slice(&[-0.5f32, -2.0, 0.0, 2.0, 0.5], vec![5]).unwrap();
        let out = clip_grad_value(&[&g], 1.0).unwrap();
        assert_eq!(read_f32(&out[0]), vec![-0.5, -1.0, 0.0, 1.0, 0.5]);
    }

    #[test]
    fn clip_multi_grad_list() {
        let g1 = Tensor::from_slice(&[10.0f32], vec![1]).unwrap();
        let g2 = Tensor::from_slice(&[-10.0f32], vec![1]).unwrap();
        let out = clip_grad_value(&[&g1, &g2], 0.5).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(read_f32(&out[0]), vec![0.5]);
        assert_eq!(read_f32(&out[1]), vec![-0.5]);
    }

    #[test]
    fn clip_zero_value_errors() {
        let g = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = clip_grad_value(&[&g], 0.0).unwrap_err();
        assert!(e.to_string().contains("clip_value"));
    }

    #[test]
    fn clip_empty_grads_errors() {
        let e = clip_grad_value(&[], 1.0).unwrap_err();
        assert!(e.to_string().contains("at least one"));
    }

    #[test]
    fn clip_bf16() {
        let g = Tensor::from_slice(
            &[
                half::bf16::from_f32(5.0),
                half::bf16::from_f32(-5.0),
                half::bf16::from_f32(0.5),
            ],
            vec![3],
        )
        .unwrap();
        let out = clip_grad_value(&[&g], 1.0).unwrap();
        let cpu = out[0].storage();
        let cpu = cpu.as_any().downcast_ref::<CpuStorage>().unwrap();
        let vals: Vec<f32> = cpu
            .as_bytes()
            .chunks(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect();
        assert!((vals[0] - 1.0).abs() < 1e-3);
        assert!((vals[1] + 1.0).abs() < 1e-3);
        assert!((vals[2] - 0.5).abs() < 1e-3);
    }
}
