//! `cosine_similarity` — pairwise cosine similarity along trailing
//! axis.
//!
//! ```text
//! cos_sim(a, b)[i] = (a[i, :] · b[i, :]) / (‖a[i, :]‖ * ‖b[i, :]‖)
//! ```
//!
//! `a, b: [..., D]` → output `[...]` (axis D removed). Used in:
//! - **Contrastive learning** (SimCLR / CLIP-style negatives)
//! - **Retrieval scoring** (k-NN over normalized embeddings)
//! - **Cosine-LR schedules** (single-vector dot via this op)

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// Materialize `t` on CPU. CUDA inputs are D2H-copied via
/// `cuda_to_host_copy`; CPU inputs are cheap `Arc` bumps. Cosine
/// similarity is a per-row reduction commonly used in eval /
/// contrastive-eval / k-NN scoring loops where the inputs already
/// live on the device that produced them (encoder outputs on
/// CUDA, etc.), so the public function must accept either device
/// transparently. See `#1082`.
fn to_cpu(t: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if matches!(t.device(), crate::Device::Cuda(_)) {
            return crate::cuda_to_host_copy(t);
        }
    }
    Ok(t.clone())
}

pub fn cosine_similarity(a: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
    if a.shape() != b.shape() {
        bail!(
            "cosine_similarity: shape mismatch: {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
    }
    if a.dtype() != b.dtype() {
        bail!("cosine_similarity: dtype mismatch");
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "cosine_similarity: dtype must be F32/BF16/F16, got {}",
            a.dtype()
        );
    }
    if a.rank() == 0 {
        bail!("cosine_similarity: input must have rank ≥ 1");
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        bail!("cosine_similarity: inputs must be contiguous");
    }
    // Both inputs must be on the same device — matches DeviceOp2's
    // mismatched-device contract.
    if a.device() != b.device() {
        bail!(
            "cosine_similarity: inputs on different devices: a={}, b={}",
            a.device(),
            b.device()
        );
    }
    let a = to_cpu(a)?;
    let b = to_cpu(b)?;
    let dtype = a.dtype();
    let shape = a.shape().to_vec();
    let last = *shape.last().unwrap();
    let outer: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    let per = dtype.size_in_bytes();
    let a_bytes = a
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("cosine_similarity: a storage must be CpuStorage"))?
        .as_bytes();
    let b_bytes = b
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("cosine_similarity: b storage must be CpuStorage"))?
        .as_bytes();

    let mut out_bytes = vec![0u8; outer * per];
    for r in 0..outer {
        let mut dot = 0.0_f32;
        let mut na = 0.0_f32;
        let mut nb = 0.0_f32;
        for i in 0..last {
            let idx = r * last + i;
            let av = match dtype {
                DType::F32 => f32::from_le_bytes(a_bytes[idx * 4..idx * 4 + 4].try_into().unwrap()),
                DType::BF16 => half::bf16::from_le_bytes(
                    a_bytes[idx * 2..idx * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                DType::F16 => half::f16::from_le_bytes(
                    a_bytes[idx * 2..idx * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                _ => unreachable!(),
            };
            let bv = match dtype {
                DType::F32 => f32::from_le_bytes(b_bytes[idx * 4..idx * 4 + 4].try_into().unwrap()),
                DType::BF16 => half::bf16::from_le_bytes(
                    b_bytes[idx * 2..idx * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                DType::F16 => half::f16::from_le_bytes(
                    b_bytes[idx * 2..idx * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                _ => unreachable!(),
            };
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        let y = dot / ((na.sqrt() * nb.sqrt()).max(eps));
        match dtype {
            DType::F32 => out_bytes[r * 4..r * 4 + 4].copy_from_slice(&y.to_le_bytes()),
            DType::BF16 => out_bytes[r * 2..r * 2 + 2]
                .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes()),
            DType::F16 => out_bytes[r * 2..r * 2 + 2]
                .copy_from_slice(&half::f16::from_f32(y).to_le_bytes()),
            _ => unreachable!(),
        }
    }
    let mut out_shape = shape;
    out_shape.pop();
    let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())
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
    fn cosine_sim_parallel_vectors_is_one() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let b = Tensor::from_slice(&[2.0f32, 4.0, 6.0], vec![1, 3]).unwrap();
        let y = cosine_similarity(&a, &b, 1e-8).unwrap();
        assert_eq!(y.shape(), &[1]);
        let v = read_f32(&y);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_antiparallel_is_neg_one() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::from_slice(&[-1.0f32, -2.0], vec![1, 2]).unwrap();
        let v = read_f32(&cosine_similarity(&a, &b, 1e-8).unwrap());
        assert!((v[0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_orthogonal_is_zero() {
        let a = Tensor::from_slice(&[1.0f32, 0.0], vec![1, 2]).unwrap();
        let b = Tensor::from_slice(&[0.0f32, 1.0], vec![1, 2]).unwrap();
        let v = read_f32(&cosine_similarity(&a, &b, 1e-8).unwrap());
        assert!(v[0].abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_batched_per_row() {
        // 2 rows: row 0 parallel → 1; row 1 antiparallel → -1.
        let a = Tensor::from_slice(&[1.0f32, 2.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[2.0f32, 4.0, -1.0, -2.0], vec![2, 2]).unwrap();
        let v = read_f32(&cosine_similarity(&a, &b, 1e-8).unwrap());
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_zero_vector_eps_clamp() {
        // a = zeros, b = anything; without eps clamp this would be NaN.
        let a = Tensor::from_slice(&[0.0f32, 0.0], vec![1, 2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 1.0], vec![1, 2]).unwrap();
        let v = read_f32(&cosine_similarity(&a, &b, 1e-8).unwrap());
        assert!(v[0].is_finite());
    }

    #[test]
    fn cosine_sim_shape_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = cosine_similarity(&a, &b, 1e-8).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    /// CUDA parity: cosine similarity on CUDA-resident tensors
    /// matches the CPU result (within the BF16/F16 cast tolerance
    /// implicit in the read-back path). Validates that lifting CPU
    /// inputs onto CUDA via `to_device` and running the op keeps
    /// working after the to_cpu insertion.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_cosine_sim_parity() {
        let cdev = match candle_core::Device::cuda_if_available(0) {
            Ok(candle_core::Device::Cuda(c)) => c,
            _ => return,
        };
        let cdev = std::sync::Arc::new(cdev);

        // Two rows: parallel + anti-parallel.
        let a_cpu = Tensor::from_slice(&[1.0f32, 2.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let b_cpu = Tensor::from_slice(&[2.0f32, 4.0, -1.0, -2.0], vec![2, 2]).unwrap();
        let a_cuda = a_cpu
            .to_device(crate::Device::Cuda(0), Some(cdev.clone()))
            .unwrap();
        let b_cuda = b_cpu
            .to_device(crate::Device::Cuda(0), Some(cdev.clone()))
            .unwrap();

        let cpu_out = cosine_similarity(&a_cpu, &b_cpu, 1e-8).unwrap();
        let cuda_out = cosine_similarity(&a_cuda, &b_cuda, 1e-8).unwrap();
        let c = read_f32(&cpu_out);
        let g = read_f32(&cuda_out);
        assert_eq!(c.len(), g.len());
        for (i, (ci, gi)) in c.iter().zip(g.iter()).enumerate() {
            assert!((ci - gi).abs() < 1e-5, "row {i}: cpu={ci}, cuda={gi}");
        }
    }

    /// Mismatched-device inputs should error explicitly rather than
    /// silently downcasting to CpuStorage and producing garbage. This
    /// matches the DeviceOp2 contract at dispatch2(); cosine_similarity
    /// is a free function so it enforces the same rule itself.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_cosine_sim_mixed_devices_errors() {
        let cdev = match candle_core::Device::cuda_if_available(0) {
            Ok(candle_core::Device::Cuda(c)) => c,
            _ => return,
        };
        let cdev = std::sync::Arc::new(cdev);

        let a_cpu = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let b_cuda = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2])
            .unwrap()
            .to_device(crate::Device::Cuda(0), Some(cdev.clone()))
            .unwrap();

        let e = cosine_similarity(&a_cpu, &b_cuda, 1e-8).unwrap_err();
        assert!(
            e.to_string().contains("different devices"),
            "expected mixed-device error, got {e}"
        );
    }
}
