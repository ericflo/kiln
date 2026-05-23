//! `triu` / `tril` — upper / lower triangular masks and matrix
//! triangulizations.
//!
//! - `triu_mask(n)`: U8 mask `[n, n]` with 1 on/above the diagonal.
//! - `tril_mask(n)`: U8 mask `[n, n]` with 1 on/below the diagonal.
//! - `triu(t)`: zero out the lower triangle of a rank-2 tensor.
//! - `tril(t)`: zero out the upper triangle of a rank-2 tensor.
//!
//! `triu_mask` is the complement of `causal_mask`. Used for upper-
//! triangular attention (encoder-style) and matrix decomposition
//! algorithms (Cholesky etc.).

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

pub fn triu_mask(n: usize) -> Result<Tensor> {
    if n == 0 {
        bail!("triu_mask: n must be > 0");
    }
    let mut bytes = vec![0u8; n * n];
    for i in 0..n {
        for j in 0..n {
            if j >= i {
                bytes[i * n + j] = 1;
            }
        }
    }
    let cpu = CpuStorage::from_bytes(DType::U8, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(vec![n, n]), TensorId::next())
}

pub fn tril_mask(n: usize) -> Result<Tensor> {
    if n == 0 {
        bail!("tril_mask: n must be > 0");
    }
    let mut bytes = vec![0u8; n * n];
    for i in 0..n {
        for j in 0..n {
            if j <= i {
                bytes[i * n + j] = 1;
            }
        }
    }
    let cpu = CpuStorage::from_bytes(DType::U8, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(vec![n, n]), TensorId::next())
}

fn apply_triangle(t: &Tensor, keep_upper: bool, name: &str) -> Result<Tensor> {
    if t.rank() != 2 {
        bail!("{name}: input must be rank-2 [N, N], got {:?}", t.shape());
    }
    let n = t.shape()[0];
    if t.shape()[1] != n {
        bail!(
            "{name}: input must be square, got {:?}",
            t.shape()
        );
    }
    if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", t.dtype());
    }
    if !t.is_contiguous() {
        bail!("{name}: input must be contiguous");
    }
    let dtype = t.dtype();
    let per = dtype.size_in_bytes();
    let bytes = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("triangular: storage must be CpuStorage"))?
        .as_bytes();
    let mut out = vec![0u8; n * n * per];
    for i in 0..n {
        for j in 0..n {
            let keep = if keep_upper { j >= i } else { j <= i };
            if keep {
                let off = (i * n + j) * per;
                out[off..off + per].copy_from_slice(&bytes[off..off + per]);
            }
        }
    }
    let cpu = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(vec![n, n]), TensorId::next())
}

pub fn triu(t: &Tensor) -> Result<Tensor> {
    apply_triangle(t, true, "triu")
}

pub fn tril(t: &Tensor) -> Result<Tensor> {
    apply_triangle(t, false, "tril")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u8(t: &Tensor) -> Vec<u8> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes().to_vec()
    }

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn triu_mask_3x3() {
        let m = triu_mask(3).unwrap();
        // Row 0: keep all; row 1: keep cols 1, 2; row 2: keep col 2.
        assert_eq!(read_u8(&m), vec![1, 1, 1, 0, 1, 1, 0, 0, 1]);
    }

    #[test]
    fn tril_mask_3x3() {
        let m = tril_mask(3).unwrap();
        assert_eq!(read_u8(&m), vec![1, 0, 0, 1, 1, 0, 1, 1, 1]);
    }

    #[test]
    fn triu_zeros_lower_triangle() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]).unwrap();
        let y = triu(&x).unwrap();
        // Upper-triangular form of [[1,2,3],[4,5,6],[7,8,9]] = [[1,2,3],[0,5,6],[0,0,9]]
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn tril_zeros_upper_triangle() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]).unwrap();
        let y = tril(&x).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn triu_tril_complement() {
        // triu + tril = original + diagonal (counted twice).
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let u = read_f32(&triu(&x).unwrap());
        let l = read_f32(&tril(&x).unwrap());
        // u + l - diag = orig
        // diag = [1, 0, 0, 4]; sum = [2, 2, 3, 8]; minus diag = [1, 2, 3, 4]
        let orig: Vec<f32> = u.iter().zip(l.iter()).enumerate().map(|(i, (a, b))| {
            a + b - if i == 0 || i == 3 { x.storage().as_any().downcast_ref::<CpuStorage>().unwrap().as_bytes().chunks(4).nth(i).map(|c| f32::from_le_bytes(c.try_into().unwrap())).unwrap() } else { 0.0 }
        }).collect();
        assert_eq!(orig, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn triu_rank_mismatch_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = triu(&x).unwrap_err();
        assert!(e.to_string().contains("rank-2"));
    }

    #[test]
    fn triu_non_square_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let e = triu(&x).unwrap_err();
        assert!(e.to_string().contains("square"));
    }

    #[test]
    fn triu_mask_zero_size_errors() {
        let e = triu_mask(0).unwrap_err();
        assert!(e.to_string().contains("> 0"));
    }
}
