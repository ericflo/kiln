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
}
