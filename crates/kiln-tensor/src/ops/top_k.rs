//! `top_k` — top-k values + indices along the trailing axis.
//!
//! Different from [`crate::ops::logit_processor::TopKProcessor`]
//! (which masks all-but-top-k logits with -inf for sampling).
//! `top_k` returns **both** the top-k values and their indices,
//! shaped `[..., k]`. Used by:
//!
//! - **Beam search** — top-k token ids per decoder step
//! - **Mixture-of-experts** — top-k expert indices for routing
//! - **KNN-style retrieval** — top-k scored references
//!
//! # Semantics
//!
//! For each row along the trailing axis of `x: [..., D]`:
//!
//! 1. Sort the row by value, descending.
//! 2. Output values: the first `k` sorted values, in descending order.
//! 3. Output indices: the original positions of those values, in the
//!    same order.
//!
//! Ties are broken by **lowest index** (the lower-index entry sorts
//! "before" the higher-index entry).
//!
//! # Returned shapes
//!
//! - `values: [..., k]`, same dtype as input
//! - `indices: [..., k]`, dtype `I64`
//!
//! # Determinism
//!
//! `Constructive`. The CPU reference does a `sort_by` with a
//! deterministic tie-breaker; output is bit-identical at the same
//! input dtype.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// Compute the top-`k` values + indices along the trailing axis.
pub fn top_k(x: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    if x.rank() == 0 {
        bail!("top_k: input must have rank ≥ 1");
    }
    let shape = x.shape();
    let last = *shape.last().unwrap();
    if k == 0 {
        bail!("top_k: k must be > 0");
    }
    if k > last {
        bail!(
            "top_k: k={k} > trailing axis size {last} (shape {:?})",
            shape
        );
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("top_k: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("top_k: input must be contiguous");
    }
    let dtype = x.dtype();
    let per = dtype.size_in_bytes();
    let n_rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("top_k: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();

    // Output shapes.
    let mut out_shape = shape.to_vec();
    *out_shape.last_mut().unwrap() = k;

    let mut values_bytes = vec![0u8; n_rows * k * per];
    let mut indices_bytes = vec![0u8; n_rows * k * 8]; // I64 = 8 bytes

    for r in 0..n_rows {
        // Read row as f32.
        let mut row: Vec<(f32, usize)> = Vec::with_capacity(last);
        for i in 0..last {
            let v = match dtype {
                DType::F32 => f32::from_le_bytes(
                    bytes[(r * last + i) * 4..(r * last + i) * 4 + 4].try_into().unwrap(),
                ),
                DType::BF16 => half::bf16::from_le_bytes(
                    bytes[(r * last + i) * 2..(r * last + i) * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                DType::F16 => half::f16::from_le_bytes(
                    bytes[(r * last + i) * 2..(r * last + i) * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                _ => unreachable!(),
            };
            row.push((v, i));
        }
        // Sort descending by value; tie-break by lowest index (stable
        // sort + descending key keeps the lower-index entry first
        // when values are equal).
        row.sort_by(|(va, ia), (vb, ib)| {
            vb.partial_cmp(va).unwrap_or(std::cmp::Ordering::Equal).then_with(|| ia.cmp(ib))
        });
        // Write the first k.
        for j in 0..k {
            let (v, i) = row[j];
            let off_val = (r * k + j) * per;
            match dtype {
                DType::F32 => values_bytes[off_val..off_val + 4].copy_from_slice(&v.to_le_bytes()),
                DType::BF16 => values_bytes[off_val..off_val + 2]
                    .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes()),
                DType::F16 => values_bytes[off_val..off_val + 2]
                    .copy_from_slice(&half::f16::from_f32(v).to_le_bytes()),
                _ => unreachable!(),
            }
            let off_idx = (r * k + j) * 8;
            indices_bytes[off_idx..off_idx + 8].copy_from_slice(&(i as i64).to_le_bytes());
        }
    }

    let v_cpu = CpuStorage::from_bytes(dtype, values_bytes)?;
    let v_storage: Storage = Arc::new(v_cpu);
    let values = Tensor::from_parts(v_storage, Layout::contiguous(out_shape.clone()), TensorId::next())?;

    let i_cpu = CpuStorage::from_bytes(DType::I64, indices_bytes)?;
    let i_storage: Storage = Arc::new(i_cpu);
    let indices = Tensor::from_parts(i_storage, Layout::contiguous(out_shape), TensorId::next())?;

    Ok((values, indices))
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

    fn read_i64(t: &Tensor) -> Vec<i64> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn top_k_basic_descending() {
        // [3, 1, 4, 1, 5] top-3 → values [5, 4, 3] indices [4, 2, 0]
        let x = Tensor::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], vec![5]).unwrap();
        let (v, i) = top_k(&x, 3).unwrap();
        assert_eq!(v.shape(), &[3]);
        assert_eq!(i.shape(), &[3]);
        assert_eq!(i.dtype(), DType::I64);
        assert_eq!(read_f32(&v), vec![5.0, 4.0, 3.0]);
        assert_eq!(read_i64(&i), vec![4, 2, 0]);
    }

    #[test]
    fn top_k_ties_break_to_lower_index() {
        // [5, 5, 3, 1] top-2 → values [5, 5] indices [0, 1]
        let x = Tensor::from_slice(&[5.0f32, 5.0, 3.0, 1.0], vec![4]).unwrap();
        let (v, i) = top_k(&x, 2).unwrap();
        assert_eq!(read_f32(&v), vec![5.0, 5.0]);
        assert_eq!(read_i64(&i), vec![0, 1]);
    }

    #[test]
    fn top_k_rank2_per_row() {
        // [[1, 5, 3, 2], [9, 2, 7, 4]] top-2 along trailing axis
        // Row 0: [5, 3], indices [1, 2]
        // Row 1: [9, 7], indices [0, 2]
        let x = Tensor::from_slice(
            &[1.0f32, 5.0, 3.0, 2.0, 9.0, 2.0, 7.0, 4.0],
            vec![2, 4],
        )
        .unwrap();
        let (v, i) = top_k(&x, 2).unwrap();
        assert_eq!(v.shape(), &[2, 2]);
        assert_eq!(i.shape(), &[2, 2]);
        assert_eq!(read_f32(&v), vec![5.0, 3.0, 9.0, 7.0]);
        assert_eq!(read_i64(&i), vec![1, 2, 0, 2]);
    }

    #[test]
    fn top_k_k_equals_full_size() {
        // Equivalent to a full descending sort.
        let x = Tensor::from_slice(&[2.0f32, 1.0, 3.0], vec![3]).unwrap();
        let (v, i) = top_k(&x, 3).unwrap();
        assert_eq!(read_f32(&v), vec![3.0, 2.0, 1.0]);
        assert_eq!(read_i64(&i), vec![2, 0, 1]);
    }

    #[test]
    fn top_k_k_one_is_argmax() {
        let x = Tensor::from_slice(&[0.5f32, -1.0, 3.7, 0.0], vec![4]).unwrap();
        let (v, i) = top_k(&x, 1).unwrap();
        assert_eq!(read_f32(&v), vec![3.7]);
        assert_eq!(read_i64(&i), vec![2]);
    }

    #[test]
    fn top_k_rank3_per_row() {
        // [B=2, S=2, D=3]. Top-1 along axis 2 should match per-row argmax.
        let x = Tensor::from_slice(
            &[
                1.0f32, 3.0, 2.0, // row 0,0 → max at idx 1
                4.0, 0.0, 5.0, // row 0,1 → max at idx 2
                7.0, 6.0, 8.0, // row 1,0 → max at idx 2
                1.0, 9.0, 2.0, // row 1,1 → max at idx 1
            ],
            vec![2, 2, 3],
        )
        .unwrap();
        let (v, i) = top_k(&x, 1).unwrap();
        assert_eq!(v.shape(), &[2, 2, 1]);
        assert_eq!(read_f32(&v), vec![3.0, 5.0, 8.0, 9.0]);
        assert_eq!(read_i64(&i), vec![1, 2, 2, 1]);
    }

    #[test]
    fn top_k_bf16_round_trip() {
        let bf: Vec<half::bf16> = [1.0f32, 5.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![3]).unwrap();
        let (v, i) = top_k(&x, 2).unwrap();
        assert_eq!(v.dtype(), DType::BF16);
        assert_eq!(i.dtype(), DType::I64);
        assert_eq!(read_i64(&i), vec![1, 2]);
    }

    #[test]
    fn top_k_negative_values_handled() {
        let x = Tensor::from_slice(&[-5.0f32, -1.0, -10.0, -2.0], vec![4]).unwrap();
        let (v, i) = top_k(&x, 2).unwrap();
        // Least-negative first: -1 (idx 1), -2 (idx 3)
        assert_eq!(read_f32(&v), vec![-1.0, -2.0]);
        assert_eq!(read_i64(&i), vec![1, 3]);
    }

    #[test]
    fn top_k_k_zero_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = top_k(&x, 0).unwrap_err();
        assert!(e.to_string().contains("k must be > 0"));
    }

    #[test]
    fn top_k_k_too_large_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = top_k(&x, 5).unwrap_err();
        assert!(e.to_string().contains("> trailing axis size"));
    }

    #[test]
    fn top_k_rejects_rank_0() {
        let x = Tensor::zeros_cpu(vec![], DType::F32);
        let e = top_k(&x, 1).unwrap_err();
        assert!(e.to_string().contains("rank ≥ 1"));
    }

    #[test]
    fn top_k_rejects_bad_dtype() {
        let x = Tensor::from_slice(&[1u32, 2, 3], vec![3]).unwrap();
        let e = top_k(&x, 2).unwrap_err();
        assert!(e.to_string().contains("F32/BF16/F16"));
    }
}
