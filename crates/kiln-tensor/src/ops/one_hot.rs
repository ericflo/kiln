//! `one_hot` — encode integer indices as one-hot vectors.
//!
//! ```text
//! y[..., j] = 1 if indices[...] == j else 0
//! ```
//!
//! Input shape `[D1, …, Dn]` (I64 or U32 indices).
//! Output shape `[D1, …, Dn, depth]` (F32/BF16/F16; defaults to F32).
//!
//! # Use cases
//!
//! - **Cross-entropy with hard labels** — alternative form when the
//!   downstream code expects soft probabilities.
//! - **Mixture-of-experts expert masks** — convert top-k indices to
//!   binary expert selectors.
//! - **Discrete → continuous bridge** for any model component that
//!   wants vector-valued inputs from a token id.
//!
//! Non-differentiable. The output is piecewise constant in the input
//! indices, so no `BackwardOp`.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// Encode `indices` as one-hot vectors along a new trailing axis of
/// size `depth`. Returns a tensor with `dtype` (default F32) and
/// shape `indices.shape ++ [depth]`.
pub fn one_hot(indices: &Tensor, depth: usize, dtype: DType) -> Result<Tensor> {
    if !matches!(indices.dtype(), DType::I64 | DType::U32) {
        bail!(
            "one_hot: indices dtype must be I64/U32, got {}",
            indices.dtype()
        );
    }
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "one_hot: output dtype must be F32/BF16/F16, got {dtype}"
        );
    }
    if depth == 0 {
        bail!("one_hot: depth must be > 0");
    }
    if !indices.is_contiguous() {
        bail!("one_hot: indices must be contiguous");
    }
    let n = indices.element_count();
    let in_cpu = indices
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("one_hot: storage must be CpuStorage"))?;
    let in_bytes = in_cpu.as_bytes();
    let ids: Vec<u64> = match indices.dtype() {
        DType::I64 => (0..n)
            .map(|i| {
                let v = i64::from_le_bytes(in_bytes[i * 8..i * 8 + 8].try_into().unwrap());
                if v < 0 {
                    Err(Error::Msg(format!(
                        "one_hot: negative index {v} at position {i}"
                    )))
                } else {
                    Ok(v as u64)
                }
            })
            .collect::<Result<_>>()?,
        DType::U32 => (0..n)
            .map(|i| {
                u32::from_le_bytes(in_bytes[i * 4..i * 4 + 4].try_into().unwrap()) as u64
            })
            .collect(),
        _ => unreachable!(),
    };

    let per = dtype.size_in_bytes();
    let one_bytes = match dtype {
        DType::F32 => 1.0f32.to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f32(1.0).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f32(1.0).to_le_bytes().to_vec(),
        _ => unreachable!(),
    };
    let mut out = vec![0u8; n * depth * per];
    for (i, &id) in ids.iter().enumerate() {
        if id as usize >= depth {
            bail!(
                "one_hot: index {id} out of range (depth={depth}) at position {i}"
            );
        }
        let off = (i * depth + id as usize) * per;
        out[off..off + per].copy_from_slice(&one_bytes);
    }
    let cpu = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu);
    let mut out_shape = indices.shape().to_vec();
    out_shape.push(depth);
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
    fn one_hot_rank1_simple() {
        // indices = [0, 1, 2]; depth = 3
        // out = [[1,0,0], [0,1,0], [0,0,1]]
        let idx = Tensor::from_slice(&[0i64, 1, 2], vec![3]).unwrap();
        let y = one_hot(&idx, 3, DType::F32).unwrap();
        assert_eq!(y.shape(), &[3, 3]);
        assert_eq!(
            read_f32(&y),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn one_hot_depth_larger_than_index() {
        // indices = [0, 2]; depth = 4 → output has trailing zeros.
        let idx = Tensor::from_slice(&[0i64, 2], vec![2]).unwrap();
        let y = one_hot(&idx, 4, DType::F32).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn one_hot_rank2_indices_preserves_leading_shape() {
        // indices [2, 3]; depth=4. Output shape [2, 3, 4].
        let idx = Tensor::from_slice(&[0i64, 1, 2, 3, 2, 1], vec![2, 3]).unwrap();
        let y = one_hot(&idx, 4, DType::F32).unwrap();
        assert_eq!(y.shape(), &[2, 3, 4]);
    }

    #[test]
    fn one_hot_u32_indices() {
        let idx = Tensor::from_slice(&[1u32, 0], vec![2]).unwrap();
        let y = one_hot(&idx, 3, DType::F32).unwrap();
        assert_eq!(read_f32(&y), vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn one_hot_bf16_output() {
        let idx = Tensor::from_slice(&[0i64, 1], vec![2]).unwrap();
        let y = one_hot(&idx, 2, DType::BF16).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        assert_eq!(y.shape(), &[2, 2]);
    }

    #[test]
    fn one_hot_negative_index_errors() {
        let idx = Tensor::from_slice(&[-1i64], vec![1]).unwrap();
        let e = one_hot(&idx, 3, DType::F32).unwrap_err();
        assert!(e.to_string().contains("negative"));
    }

    #[test]
    fn one_hot_index_out_of_range_errors() {
        let idx = Tensor::from_slice(&[5i64], vec![1]).unwrap();
        let e = one_hot(&idx, 3, DType::F32).unwrap_err();
        assert!(e.to_string().contains("out of range"));
    }

    #[test]
    fn one_hot_depth_zero_errors() {
        let idx = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let e = one_hot(&idx, 0, DType::F32).unwrap_err();
        assert!(e.to_string().contains("depth"));
    }

    #[test]
    fn one_hot_bad_index_dtype_errors() {
        let idx = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = one_hot(&idx, 1, DType::F32).unwrap_err();
        assert!(e.to_string().contains("indices dtype"));
    }
}
