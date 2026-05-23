//! `unique` — return sorted unique values from a 1-D tensor.
//!
//! For a 1-D `I64` or `U32` tensor, returns:
//! - `values`: sorted unique values, dtype matches input
//! - `counts`: I64 occurrence counts in `values` order
//!
//! PyTorch parity with `torch.unique(x, sorted=True, return_counts=True)`.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

pub fn unique(x: &Tensor) -> Result<(Tensor, Tensor)> {
    if x.rank() != 1 {
        bail!("unique: input must be rank-1, got rank {}", x.rank());
    }
    if !x.is_contiguous() {
        bail!("unique: input must be contiguous");
    }
    let n = x.element_count();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("unique: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();

    // Read input as i64 (largest representable integer type we accept).
    let vals: Vec<i64> = match x.dtype() {
        DType::I64 => (0..n)
            .map(|i| i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()))
            .collect(),
        DType::U32 => (0..n)
            .map(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as i64)
            .collect(),
        other => bail!("unique: input dtype must be I64 or U32, got {other}"),
    };

    // Sort + dedupe with counts.
    let mut sorted = vals.clone();
    sorted.sort();
    let mut values: Vec<i64> = Vec::new();
    let mut counts: Vec<i64> = Vec::new();
    let mut i = 0usize;
    while i < sorted.len() {
        let v = sorted[i];
        let mut c = 1i64;
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == v {
            c += 1;
            j += 1;
        }
        values.push(v);
        counts.push(c);
        i = j;
    }

    // Build output value tensor in the input dtype.
    let val_dtype = x.dtype();
    let per = val_dtype.size_in_bytes();
    let mut val_bytes = Vec::with_capacity(values.len() * per);
    match val_dtype {
        DType::I64 => {
            for &v in &values {
                val_bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        DType::U32 => {
            for &v in &values {
                val_bytes.extend_from_slice(&(v as u32).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let val_cpu = CpuStorage::from_bytes(val_dtype, val_bytes)?;
    let val_storage: Storage = Arc::new(val_cpu);
    let val_t = Tensor::from_parts(
        val_storage,
        Layout::contiguous(vec![values.len()]),
        TensorId::next(),
    )?;

    // Counts always I64.
    let mut count_bytes = Vec::with_capacity(counts.len() * 8);
    for c in counts {
        count_bytes.extend_from_slice(&c.to_le_bytes());
    }
    let count_cpu = CpuStorage::from_bytes(DType::I64, count_bytes)?;
    let count_storage: Storage = Arc::new(count_cpu);
    let count_t = Tensor::from_parts(
        count_storage,
        Layout::contiguous(vec![values.len()]),
        TensorId::next(),
    )?;
    Ok((val_t, count_t))
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

    fn read_u32(t: &Tensor) -> Vec<u32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn unique_basic_i64() {
        let x = Tensor::from_slice(&[3i64, 1, 2, 1, 3, 1], vec![6]).unwrap();
        let (v, c) = unique(&x).unwrap();
        assert_eq!(read_i64(&v), vec![1, 2, 3]);
        assert_eq!(read_i64(&c), vec![3, 1, 2]);
    }

    #[test]
    fn unique_basic_u32() {
        let x = Tensor::from_slice(&[5u32, 5, 1, 0, 1], vec![5]).unwrap();
        let (v, c) = unique(&x).unwrap();
        assert_eq!(read_u32(&v), vec![0, 1, 5]);
        assert_eq!(read_i64(&c), vec![1, 2, 2]);
    }

    #[test]
    fn unique_all_same() {
        let x = Tensor::from_slice(&[7i64, 7, 7, 7], vec![4]).unwrap();
        let (v, c) = unique(&x).unwrap();
        assert_eq!(read_i64(&v), vec![7]);
        assert_eq!(read_i64(&c), vec![4]);
    }

    #[test]
    fn unique_all_distinct() {
        let x = Tensor::from_slice(&[3i64, 1, 4, 1, 5], vec![5]).unwrap();
        let (v, c) = unique(&x).unwrap();
        assert_eq!(read_i64(&v), vec![1, 3, 4, 5]);
        assert_eq!(read_i64(&c), vec![2, 1, 1, 1]);
    }

    #[test]
    fn unique_empty_input() {
        let x = Tensor::from_slice::<i64>(&[], vec![0]).unwrap();
        let (v, c) = unique(&x).unwrap();
        assert_eq!(v.shape(), &[0]);
        assert_eq!(c.shape(), &[0]);
    }

    #[test]
    fn unique_rank_not_one_errors() {
        let x = Tensor::from_slice(&[1i64, 2, 3, 4], vec![2, 2]).unwrap();
        let e = unique(&x).unwrap_err();
        assert!(e.to_string().contains("rank"));
    }

    #[test]
    fn unique_sums_to_input_len() {
        let x = Tensor::from_slice(&[2i64, 9, 9, 0, 2, 9, 1], vec![7]).unwrap();
        let (_v, c) = unique(&x).unwrap();
        let sum: i64 = read_i64(&c).iter().sum();
        assert_eq!(sum, 7);
    }
}
