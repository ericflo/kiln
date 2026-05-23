//! `nonzero` — return the coordinates where a tensor is non-zero.
//!
//! For a rank-`R` input of shape `[d0, d1, …, d_{R-1}]`, returns a
//! rank-2 `I64` tensor of shape `[N, R]` where `N` is the number of
//! non-zero elements. Each row is one set of coordinates. PyTorch
//! parity with `torch.nonzero(x)` (the `as_tuple=False` default).

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

fn is_nonzero(dtype: DType, bytes: &[u8], i: usize) -> Result<bool> {
    let per = dtype.size_in_bytes();
    let off = i * per;
    Ok(match dtype {
        DType::F32 => {
            let v = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            v != 0.0
        }
        DType::BF16 => {
            half::bf16::from_le_bytes(bytes[off..off + 2].try_into().unwrap()) != half::bf16::ZERO
        }
        DType::F16 => {
            half::f16::from_le_bytes(bytes[off..off + 2].try_into().unwrap()) != half::f16::ZERO
        }
        DType::U8 => bytes[off] != 0,
        DType::U32 => u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) != 0,
        DType::I64 => i64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()) != 0,
        other => bail!("nonzero: unsupported dtype {other}"),
    })
}

pub fn nonzero(x: &Tensor) -> Result<Tensor> {
    if !x.is_contiguous() {
        bail!("nonzero: input must be contiguous");
    }
    let dtype = x.dtype();
    let shape: Vec<usize> = x.shape().to_vec();
    let rank = shape.len();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("nonzero: storage must be CpuStorage"))?
        .as_bytes()
        .to_vec();

    // Compute element-strides.
    let mut strides = vec![1usize; rank.max(1)];
    for d in (0..rank.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }

    let n_elem: usize = if rank == 0 { 1 } else { shape.iter().product() };
    let mut rows: Vec<i64> = Vec::new();

    // Special case: scalar tensor (rank 0). Return shape [N, 0]
    // where N is 1 if non-zero else 0.
    if rank == 0 {
        if is_nonzero(dtype, &cpu, 0)? {
            // One row of zero coords; flat data is empty.
        }
        let count = if is_nonzero(dtype, &cpu, 0)? { 1 } else { 0 };
        let bytes = vec![0u8; count * 0 * 8]; // zero rank-1 inner = 0 bytes
        let cpu_out = CpuStorage::from_bytes(DType::I64, bytes)?;
        let storage: Storage = Arc::new(cpu_out);
        return Tensor::from_parts(
            storage,
            Layout::contiguous(vec![count, 0]),
            TensorId::next(),
        );
    }

    let mut coord = vec![0usize; rank];
    for i in 0..n_elem {
        if is_nonzero(dtype, &cpu, i)? {
            let mut rem = i;
            for d in 0..rank {
                coord[d] = rem / strides[d];
                rem %= strides[d];
            }
            for &c in &coord {
                rows.push(c as i64);
            }
        }
    }
    let n = rows.len() / rank;
    let mut bytes = Vec::with_capacity(rows.len() * 8);
    for v in rows {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let cpu_out = CpuStorage::from_bytes(DType::I64, bytes)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(
        storage,
        Layout::contiguous(vec![n, rank]),
        TensorId::next(),
    )
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
    fn nonzero_1d_f32() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 2.0, 0.0, 3.0], vec![6]).unwrap();
        let y = nonzero(&x).unwrap();
        assert_eq!(y.shape(), &[3, 1]);
        assert_eq!(read_i64(&y), vec![1, 3, 5]);
    }

    #[test]
    fn nonzero_2d_f32() {
        // [[0,1],[2,0]] → nonzero at (0,1), (1,0)
        let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 0.0], vec![2, 2]).unwrap();
        let y = nonzero(&x).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(read_i64(&y), vec![0, 1, 1, 0]);
    }

    #[test]
    fn nonzero_u8_mask() {
        let mask = Tensor::from_slice(&[1u8, 0, 1, 0, 1], vec![5]).unwrap();
        let y = nonzero(&mask).unwrap();
        assert_eq!(y.shape(), &[3, 1]);
        assert_eq!(read_i64(&y), vec![0, 2, 4]);
    }

    #[test]
    fn nonzero_u32() {
        let x = Tensor::from_slice(&[0u32, 5, 0, 3], vec![4]).unwrap();
        let y = nonzero(&x).unwrap();
        assert_eq!(read_i64(&y), vec![1, 3]);
    }

    #[test]
    fn nonzero_i64() {
        let x = Tensor::from_slice(&[0i64, -1, 2, 0], vec![4]).unwrap();
        let y = nonzero(&x).unwrap();
        assert_eq!(read_i64(&y), vec![1, 2]);
    }

    #[test]
    fn nonzero_all_zeros() {
        let x = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();
        let y = nonzero(&x).unwrap();
        assert_eq!(y.shape(), &[0, 1]);
    }

    #[test]
    fn nonzero_all_nonzero() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = nonzero(&x).unwrap();
        assert_eq!(y.shape(), &[4, 2]);
        assert_eq!(read_i64(&y), vec![0, 0, 0, 1, 1, 0, 1, 1]);
    }

    #[test]
    fn nonzero_3d() {
        // shape [2, 1, 3] with nonzeros at (0,0,1), (1,0,2)
        let x = Tensor::from_slice(&[0.0f32, 5.0, 0.0, 0.0, 0.0, 7.0], vec![2, 1, 3]).unwrap();
        let y = nonzero(&x).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(read_i64(&y), vec![0, 0, 1, 1, 0, 2]);
    }
}
