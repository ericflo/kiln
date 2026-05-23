//! `bincount` — count occurrences of each integer value.
//!
//! For a 1-D `I64` or `U32` tensor of non-negative integers, returns a
//! 1-D `I64` tensor of length `max(input) + 1` (or `min_length` if
//! specified, whichever is larger). The value at index `i` is the
//! number of times `i` appears in the input. PyTorch parity with
//! `torch.bincount(x, weights=None, minlength=0)`.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

pub fn bincount(x: &Tensor, min_length: usize) -> Result<Tensor> {
    if x.rank() != 1 {
        bail!("bincount: input must be rank-1, got rank {}", x.rank());
    }
    if !x.is_contiguous() {
        bail!("bincount: input must be contiguous");
    }
    let n = x.element_count();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("bincount: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();

    // Read input values as i64.
    let vals: Vec<i64> = match x.dtype() {
        DType::I64 => (0..n)
            .map(|i| i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()))
            .collect(),
        DType::U32 => (0..n)
            .map(|i| {
                u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as i64
            })
            .collect(),
        other => bail!("bincount: input dtype must be I64 or U32, got {other}"),
    };

    // Validate non-negative.
    for &v in &vals {
        if v < 0 {
            bail!("bincount: negative value {v} in input");
        }
    }

    // Determine output length.
    let max_val = vals.iter().copied().max().unwrap_or(-1);
    let len = ((max_val + 1).max(0) as usize).max(min_length);

    let mut counts = vec![0i64; len];
    for &v in &vals {
        counts[v as usize] += 1;
    }

    let mut out_bytes = Vec::with_capacity(len * 8);
    for c in counts {
        out_bytes.extend_from_slice(&c.to_le_bytes());
    }
    let cpu_out = CpuStorage::from_bytes(DType::I64, out_bytes)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(vec![len]), TensorId::next())
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
    fn bincount_basic_i64() {
        // x = [0, 1, 1, 3, 2, 1, 3, 3] → counts of 0,1,2,3 = 1, 3, 1, 3
        let x = Tensor::from_slice(&[0i64, 1, 1, 3, 2, 1, 3, 3], vec![8]).unwrap();
        let y = bincount(&x, 0).unwrap();
        assert_eq!(y.shape(), &[4]);
        assert_eq!(read_i64(&y), vec![1, 3, 1, 3]);
    }

    #[test]
    fn bincount_basic_u32() {
        let x = Tensor::from_slice(&[2u32, 2, 0, 5], vec![4]).unwrap();
        let y = bincount(&x, 0).unwrap();
        assert_eq!(y.shape(), &[6]);
        assert_eq!(read_i64(&y), vec![1, 0, 2, 0, 0, 1]);
    }

    #[test]
    fn bincount_min_length_pads() {
        let x = Tensor::from_slice(&[0i64, 1], vec![2]).unwrap();
        let y = bincount(&x, 5).unwrap();
        assert_eq!(y.shape(), &[5]);
        assert_eq!(read_i64(&y), vec![1, 1, 0, 0, 0]);
    }

    #[test]
    fn bincount_min_length_smaller_than_max() {
        // min_length 2 but max is 3 → length stays 4.
        let x = Tensor::from_slice(&[0i64, 3], vec![2]).unwrap();
        let y = bincount(&x, 2).unwrap();
        assert_eq!(y.shape(), &[4]);
        assert_eq!(read_i64(&y), vec![1, 0, 0, 1]);
    }

    #[test]
    fn bincount_empty_input() {
        let x = Tensor::from_slice::<i64>(&[], vec![0]).unwrap();
        let y = bincount(&x, 0).unwrap();
        assert_eq!(y.shape(), &[0]);
    }

    #[test]
    fn bincount_empty_with_min_length() {
        let x = Tensor::from_slice::<i64>(&[], vec![0]).unwrap();
        let y = bincount(&x, 3).unwrap();
        assert_eq!(y.shape(), &[3]);
        assert_eq!(read_i64(&y), vec![0, 0, 0]);
    }

    #[test]
    fn bincount_negative_errors() {
        let x = Tensor::from_slice(&[0i64, -1, 2], vec![3]).unwrap();
        let e = bincount(&x, 0).unwrap_err();
        assert!(e.to_string().contains("negative"));
    }

    #[test]
    fn bincount_rank_not_one_errors() {
        let x = Tensor::from_slice(&[0i64, 1, 2, 3], vec![2, 2]).unwrap();
        let e = bincount(&x, 0).unwrap_err();
        assert!(e.to_string().contains("rank"));
    }
}
