//! `clip_grad_norm` — rescale a list of gradients so their joint
//! L2 norm is at most `max_norm`.
//!
//! Mirrors PyTorch's `torch.nn.utils.clip_grad_norm_`. Used by every
//! production training script to keep gradient updates bounded.
//!
//! Returns the (pre-clip) joint L2 norm + the clipped gradient
//! tensors. Callers should plug the clipped grads into their
//! optimizer step.

use std::sync::Arc;

use crate::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

/// Compute the joint L2 norm of all gradients and rescale them in-
/// place if the norm exceeds `max_norm`. Returns
/// `(total_norm, clipped_grads)`.
///
/// `max_norm` must be > 0. All grads must share dtype (F32/BF16/F16).
/// Empty input is rejected.
pub fn clip_grad_norm(grads: &[&Tensor], max_norm: f32) -> Result<(f32, Vec<Tensor>)> {
    if grads.is_empty() {
        bail!("clip_grad_norm: at least one gradient required");
    }
    if max_norm <= 0.0 {
        bail!("clip_grad_norm: max_norm must be > 0, got {max_norm}");
    }
    let dtype = grads[0].dtype();
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "clip_grad_norm: dtype must be F32/BF16/F16, got {dtype}"
        );
    }
    for (i, g) in grads.iter().enumerate() {
        if g.dtype() != dtype {
            bail!(
                "clip_grad_norm: grad {i} dtype {} != grad 0 dtype {dtype}",
                g.dtype()
            );
        }
        if !g.is_contiguous() {
            bail!("clip_grad_norm: grad {i} must be contiguous");
        }
    }

    // 1. Compute joint L2 norm.
    let mut sq_sum = 0.0_f32;
    for g in grads {
        let bytes = g
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("clip_grad_norm: storage must be CpuStorage"))?
            .as_bytes();
        let n = g.element_count();
        for i in 0..n {
            let v = match dtype {
                DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                DType::BF16 => {
                    half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
                }
                DType::F16 => {
                    half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
                }
                _ => unreachable!(),
            };
            sq_sum += v * v;
        }
    }
    let total_norm = sq_sum.sqrt();

    // 2. Rescale if needed.
    let scale = if total_norm > max_norm {
        max_norm / (total_norm + 1e-6)
    } else {
        1.0
    };

    let mut out = Vec::with_capacity(grads.len());
    for g in grads {
        let n = g.element_count();
        let per = dtype.size_in_bytes();
        let bytes = g
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("clip_grad_norm: storage must be CpuStorage"))?
            .as_bytes();
        let mut out_bytes = vec![0u8; n * per];
        for i in 0..n {
            let v = match dtype {
                DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                DType::BF16 => {
                    half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
                }
                DType::F16 => {
                    half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
                }
                _ => unreachable!(),
            };
            let y = v * scale;
            match dtype {
                DType::F32 => out_bytes[i * 4..i * 4 + 4].copy_from_slice(&y.to_le_bytes()),
                DType::BF16 => out_bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes()),
                DType::F16 => out_bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(y).to_le_bytes()),
                _ => unreachable!(),
            }
        }
        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        out.push(Tensor::from_parts(
            storage,
            Layout::contiguous(g.shape().to_vec()),
            TensorId::next(),
        )?);
    }
    Ok((total_norm, out))
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
    fn clip_grad_below_max_is_identity() {
        // Joint norm = √(9+16) = 5; max_norm = 10 → no clipping.
        let a = Tensor::from_slice(&[3.0f32, 0.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[4.0f32], vec![1]).unwrap();
        let (norm, clipped) = clip_grad_norm(&[&a, &b], 10.0).unwrap();
        assert!((norm - 5.0).abs() < 1e-5);
        assert_eq!(read_f32(&clipped[0]), vec![3.0, 0.0]);
        assert_eq!(read_f32(&clipped[1]), vec![4.0]);
    }

    #[test]
    fn clip_grad_above_max_rescales() {
        // Joint norm = 5; max_norm = 1 → scale = 1/5 (modulo eps).
        let a = Tensor::from_slice(&[3.0f32, 0.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[4.0f32], vec![1]).unwrap();
        let (norm, clipped) = clip_grad_norm(&[&a, &b], 1.0).unwrap();
        assert!((norm - 5.0).abs() < 1e-5);
        let ca = read_f32(&clipped[0]);
        let cb = read_f32(&clipped[1]);
        // Post-clip norm should be ≤ max_norm (1.0).
        let post = (ca.iter().map(|v| v * v).sum::<f32>() + cb.iter().map(|v| v * v).sum::<f32>()).sqrt();
        assert!(post <= 1.001, "post-clip norm {post} > max 1.0");
        assert!((ca[0] - 0.6).abs() < 1e-3); // 3/5
        assert!((cb[0] - 0.8).abs() < 1e-3); // 4/5
    }

    #[test]
    fn clip_grad_empty_errors() {
        let e = clip_grad_norm(&[], 1.0).unwrap_err();
        assert!(e.to_string().contains("at least one"));
    }

    #[test]
    fn clip_grad_max_zero_errors() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = clip_grad_norm(&[&a], 0.0).unwrap_err();
        assert!(e.to_string().contains("max_norm"));
    }

    #[test]
    fn clip_grad_dtype_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bf = vec![half::bf16::from_f32(1.0)];
        let b = Tensor::from_slice(&bf, vec![1]).unwrap();
        let e = clip_grad_norm(&[&a, &b], 1.0).unwrap_err();
        assert!(e.to_string().contains("dtype"));
    }
}
