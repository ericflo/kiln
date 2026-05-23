//! `searchsorted` — find insertion positions in a sorted 1-D
//! sequence.
//!
//! Given a *sorted* 1-D `sorted_seq` and a same-dtype `values`
//! tensor, returns an `I64` index tensor (same shape as `values`)
//! such that `sorted_seq[i-1] <= v < sorted_seq[i]` (left side) or
//! `sorted_seq[i-1] < v <= sorted_seq[i]` (right side, the default
//! here matching PyTorch's default).
//!
//! PyTorch parity with `torch.searchsorted(sorted_seq, values,
//! right=False)`.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

fn read_f32_flat(t: &Tensor) -> Result<Vec<f32>> {
    if !t.is_contiguous() {
        bail!("searchsorted: input must be contiguous");
    }
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("searchsorted: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let mut out = Vec::with_capacity(n);
    match t.dtype() {
        DType::F32 => {
            for i in 0..n {
                out.push(f32::from_le_bytes(
                    bytes[i * 4..i * 4 + 4].try_into().unwrap(),
                ));
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
        other => bail!("searchsorted: dtype must be F32/BF16/F16, got {other}"),
    }
    Ok(out)
}

/// Search a sorted 1-D `sorted_seq` for insertion points of each
/// element in `values`. `right`: if true, returns the last valid
/// insertion point; if false, returns the first.
pub fn searchsorted(
    sorted_seq: &Tensor,
    values: &Tensor,
    right: bool,
) -> Result<Tensor> {
    if sorted_seq.rank() != 1 {
        bail!(
            "searchsorted: sorted_seq must be rank-1, got {}",
            sorted_seq.rank()
        );
    }
    if sorted_seq.dtype() != values.dtype() {
        bail!(
            "searchsorted: dtype mismatch — sorted_seq {} vs values {}",
            sorted_seq.dtype(),
            values.dtype()
        );
    }
    let seq = read_f32_flat(sorted_seq)?;
    let vals = read_f32_flat(values)?;
    let n = seq.len();
    let v_shape: Vec<usize> = values.shape().to_vec();

    let mut out_bytes = Vec::with_capacity(vals.len() * 8);
    for &v in &vals {
        // Linear binary search.
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let s = seq[mid];
            let go_right = if right { v >= s } else { v > s };
            if go_right {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let pos = lo as i64;
        out_bytes.extend_from_slice(&pos.to_le_bytes());
    }
    let cpu = CpuStorage::from_bytes(DType::I64, out_bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(v_shape), TensorId::next())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_i64(t: &Tensor) -> Vec<i64> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn searchsorted_left_basic() {
        let seq = Tensor::from_slice(&[1.0f32, 3.0, 5.0, 7.0, 9.0], vec![5]).unwrap();
        let v = Tensor::from_slice(&[0.0f32, 1.0, 4.0, 5.0, 9.0, 10.0], vec![6]).unwrap();
        let y = searchsorted(&seq, &v, false).unwrap();
        // 0 < 1 → 0; 1 == 1 left → 0; 4 < 5 → 2; 5 == 5 left → 2; 9 == 9 left → 4; 10 > 9 → 5
        assert_eq!(read_i64(&y), vec![0, 0, 2, 2, 4, 5]);
    }

    #[test]
    fn searchsorted_right_basic() {
        let seq = Tensor::from_slice(&[1.0f32, 3.0, 5.0, 7.0, 9.0], vec![5]).unwrap();
        let v = Tensor::from_slice(&[0.0f32, 1.0, 4.0, 5.0, 9.0, 10.0], vec![6]).unwrap();
        let y = searchsorted(&seq, &v, true).unwrap();
        // right: 0 → 0; 1 → 1; 4 → 2; 5 → 3; 9 → 5; 10 → 5
        assert_eq!(read_i64(&y), vec![0, 1, 2, 3, 5, 5]);
    }

    #[test]
    fn searchsorted_preserves_shape() {
        let seq = Tensor::from_slice(&[0.0f32, 1.0, 2.0], vec![3]).unwrap();
        let v = Tensor::from_slice(&[0.5f32, 1.5, 0.0, 2.0], vec![2, 2]).unwrap();
        let y = searchsorted(&seq, &v, false).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(read_i64(&y), vec![1, 2, 0, 2]);
    }

    #[test]
    fn searchsorted_empty_values() {
        let seq = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let v = Tensor::from_slice::<f32>(&[], vec![0]).unwrap();
        let y = searchsorted(&seq, &v, false).unwrap();
        assert_eq!(y.shape(), &[0]);
    }

    #[test]
    fn searchsorted_rank_mismatch_errors() {
        let seq = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let v = Tensor::from_slice(&[1.5f32], vec![1]).unwrap();
        let e = searchsorted(&seq, &v, false).unwrap_err();
        assert!(e.to_string().contains("rank-1"));
    }

    #[test]
    fn searchsorted_dtype_mismatch_errors() {
        let seq = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let v = Tensor::from_slice(&[half::bf16::from_f32(1.5)], vec![1]).unwrap();
        let e = searchsorted(&seq, &v, false).unwrap_err();
        assert!(e.to_string().contains("dtype mismatch"));
    }
}
