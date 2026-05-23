//! `sort` and `argsort` — sort the last axis.
//!
//! - `sort(x, descending)` returns `(sorted_values, indices)` where
//!   indices map sorted positions back to the original row.
//! - `argsort(x, descending)` returns just the index tensor (same
//!   shape as `x`, `I64`).
//!
//! Stable per-row. PyTorch parity with `torch.sort(x, dim=-1,
//! descending)` and `torch.argsort(x, dim=-1, descending)`.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

fn read_f32_rows(t: &Tensor) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
    if t.rank() < 1 {
        bail!("sort: input must have rank ≥ 1");
    }
    if !t.is_contiguous() {
        bail!("sort: input must be contiguous");
    }
    let dtype = t.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        bail!("sort: dtype must be F32/BF16/F16, got {dtype}");
    }
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("sort: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let shape: Vec<usize> = t.shape().to_vec();
    let last = *shape.last().unwrap();
    let rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);

    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = Vec::with_capacity(last);
        for c in 0..last {
            let i = r * last + c;
            let v = match dtype {
                DType::F32 => {
                    f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
                }
                DType::BF16 => half::bf16::from_le_bytes(
                    bytes[i * 2..i * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                DType::F16 => half::f16::from_le_bytes(
                    bytes[i * 2..i * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                _ => unreachable!(),
            };
            row.push(v);
        }
        out.push(row);
    }
    Ok((out, shape))
}

fn sort_rows(
    rows: Vec<Vec<f32>>,
    descending: bool,
) -> (Vec<Vec<f32>>, Vec<Vec<i64>>) {
    let mut sorted_vals = Vec::with_capacity(rows.len());
    let mut sorted_idx = Vec::with_capacity(rows.len());
    for row in rows {
        let mut idxs: Vec<usize> = (0..row.len()).collect();
        // Stable sort by value.
        if descending {
            idxs.sort_by(|&a, &b| {
                row[b]
                    .partial_cmp(&row[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            idxs.sort_by(|&a, &b| {
                row[a]
                    .partial_cmp(&row[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let vals: Vec<f32> = idxs.iter().map(|&i| row[i]).collect();
        let i64s: Vec<i64> = idxs.iter().map(|&i| i as i64).collect();
        sorted_vals.push(vals);
        sorted_idx.push(i64s);
    }
    (sorted_vals, sorted_idx)
}

fn build_value_tensor(
    rows: &[Vec<f32>],
    dtype: DType,
    shape: Vec<usize>,
) -> Result<Tensor> {
    let mut bytes = Vec::with_capacity(rows.len() * rows[0].len() * dtype.size_in_bytes());
    for row in rows {
        for &v in row {
            match dtype {
                DType::F32 => bytes.extend_from_slice(&v.to_le_bytes()),
                DType::BF16 => bytes
                    .extend_from_slice(&half::bf16::from_f32(v).to_le_bytes()),
                DType::F16 => bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes()),
                _ => unreachable!(),
            }
        }
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape), TensorId::next())
}

fn build_index_tensor(rows: &[Vec<i64>], shape: Vec<usize>) -> Result<Tensor> {
    let mut bytes = Vec::with_capacity(rows.len() * rows[0].len() * 8);
    for row in rows {
        for &v in row {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let cpu = CpuStorage::from_bytes(DType::I64, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape), TensorId::next())
}

/// Sort along the last axis. Returns `(sorted_values, indices)`.
pub fn sort(x: &Tensor, descending: bool) -> Result<(Tensor, Tensor)> {
    let (rows, shape) = read_f32_rows(x)?;
    let (sv, si) = sort_rows(rows, descending);
    let val_t = build_value_tensor(&sv, x.dtype(), shape.clone())?;
    let idx_t = build_index_tensor(&si, shape)?;
    Ok((val_t, idx_t))
}

/// Argsort along the last axis. Returns just the I64 index tensor.
pub fn argsort(x: &Tensor, descending: bool) -> Result<Tensor> {
    let (rows, shape) = read_f32_rows(x)?;
    let (_sv, si) = sort_rows(rows, descending);
    build_index_tensor(&si, shape)
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
    fn sort_ascending_1d() {
        let x = Tensor::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], vec![5]).unwrap();
        let (v, i) = sort(&x, false).unwrap();
        assert_eq!(read_f32(&v), vec![1.0, 1.0, 3.0, 4.0, 5.0]);
        // Stable: equal values keep original order — 1 at idx 1 then idx 3.
        assert_eq!(read_i64(&i), vec![1, 3, 0, 2, 4]);
    }

    #[test]
    fn sort_descending_1d() {
        let x = Tensor::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], vec![5]).unwrap();
        let (v, i) = sort(&x, true).unwrap();
        assert_eq!(read_f32(&v), vec![5.0, 4.0, 3.0, 1.0, 1.0]);
        assert_eq!(read_i64(&i), vec![4, 2, 0, 1, 3]);
    }

    #[test]
    fn sort_per_row_2d() {
        // [[3, 1, 2], [5, 4, 6]] ascending per row → [[1,2,3],[4,5,6]]
        let x = Tensor::from_slice(&[3.0f32, 1.0, 2.0, 5.0, 4.0, 6.0], vec![2, 3]).unwrap();
        let (v, _i) = sort(&x, false).unwrap();
        assert_eq!(read_f32(&v), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn argsort_ascending() {
        let x = Tensor::from_slice(&[3.0f32, 1.0, 2.0], vec![3]).unwrap();
        let i = argsort(&x, false).unwrap();
        assert_eq!(read_i64(&i), vec![1, 2, 0]);
    }

    #[test]
    fn argsort_descending() {
        let x = Tensor::from_slice(&[3.0f32, 1.0, 2.0], vec![3]).unwrap();
        let i = argsort(&x, true).unwrap();
        assert_eq!(read_i64(&i), vec![0, 2, 1]);
    }

    #[test]
    fn sort_already_sorted_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let (_v, i) = sort(&x, false).unwrap();
        assert_eq!(read_i64(&i), vec![0, 1, 2, 3]);
    }

    #[test]
    fn sort_reversed_yields_reverse_indices() {
        let x = Tensor::from_slice(&[4.0f32, 3.0, 2.0, 1.0], vec![4]).unwrap();
        let (_v, i) = sort(&x, false).unwrap();
        assert_eq!(read_i64(&i), vec![3, 2, 1, 0]);
    }
}
