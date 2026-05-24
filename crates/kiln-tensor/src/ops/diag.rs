//! `diagonal` — extract the main diagonal of a rank-2 tensor.
//! `diag` — build a rank-2 tensor with a vector on the diagonal.
//!
//! # Dispatch
//!
//! - CPU path: per-element byte copy of the diagonal entries.
//! - CUDA path: dedicated kernels `kiln_diagonal_extract_async` and
//!   `kiln_diag_build_async` in `csrc/diag.cu`. For `diag`, the
//!   output is first zero-initialized via `cuda_zeros` then the
//!   diagonal is written in a single pass. F32/BF16/F16, contiguous.
//!   (#1082)

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// `diagonal(t)` returns a rank-1 tensor with the main diagonal of `t`.
/// Input must be square rank-2.
pub fn diagonal(t: &Tensor) -> Result<Tensor> {
    if t.rank() != 2 {
        bail!("diagonal: input must be rank-2, got {:?}", t.shape());
    }
    let n = t.shape()[0];
    if t.shape()[1] != n {
        bail!("diagonal: input must be square, got {:?}", t.shape());
    }
    if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("diagonal: dtype must be F32/BF16/F16, got {}", t.dtype());
    }
    if !t.is_contiguous() {
        bail!("diagonal: input must be contiguous");
    }

    // CUDA fast path. (#1082)
    #[cfg(feature = "cuda")]
    {
        if matches!(t.device(), crate::Device::Cuda(_)) {
            return crate::cuda_diagonal_extract(t);
        }
    }

    let dtype = t.dtype();
    let per = dtype.size_in_bytes();
    let bytes = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("diagonal: storage must be CpuStorage"))?
        .as_bytes();
    let mut out = vec![0u8; n * per];
    for i in 0..n {
        let src = (i * n + i) * per;
        out[i * per..(i + 1) * per].copy_from_slice(&bytes[src..src + per]);
    }
    let cpu = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(vec![n]), TensorId::next())
}

/// `diag(v)` builds a `[n, n]` matrix with `v` on the diagonal and
/// zeros elsewhere. Input is rank-1.
pub fn diag(v: &Tensor) -> Result<Tensor> {
    if v.rank() != 1 {
        bail!("diag: input must be rank-1, got {:?}", v.shape());
    }
    if !matches!(v.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("diag: dtype must be F32/BF16/F16, got {}", v.dtype());
    }
    if !v.is_contiguous() {
        bail!("diag: input must be contiguous");
    }

    // CUDA fast path. (#1082)
    #[cfg(feature = "cuda")]
    {
        if matches!(v.device(), crate::Device::Cuda(_)) {
            return crate::cuda_diag_build(v);
        }
    }

    let dtype = v.dtype();
    let per = dtype.size_in_bytes();
    let n = v.element_count();
    let v_bytes = v
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("diag: storage must be CpuStorage"))?
        .as_bytes();
    let mut out = vec![0u8; n * n * per];
    for i in 0..n {
        let src = i * per;
        let dst = (i * n + i) * per;
        out[dst..dst + per].copy_from_slice(&v_bytes[src..src + per]);
    }
    let cpu = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(vec![n, n]), TensorId::next())
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
    fn diagonal_of_identity_is_ones() {
        use crate::ops::eye;
        let i = eye(4, DType::F32).unwrap();
        let d = read_f32(&diagonal(&i).unwrap());
        assert_eq!(d, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn diagonal_extracts_diagonal() {
        // [[1, 2, 3], [4, 5, 6], [7, 8, 9]] → [1, 5, 9]
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]).unwrap();
        let d = diagonal(&t).unwrap();
        assert_eq!(d.shape(), &[3]);
        assert_eq!(read_f32(&d), vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn diag_builds_diagonal_matrix() {
        let v = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let m = diag(&v).unwrap();
        assert_eq!(m.shape(), &[3, 3]);
        // [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        assert_eq!(read_f32(&m), vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn diagonal_diag_roundtrip() {
        // diag → diagonal recovers the input.
        let v = Tensor::from_slice(&[5.0f32, 7.0, 11.0], vec![3]).unwrap();
        let m = diag(&v).unwrap();
        let v2 = diagonal(&m).unwrap();
        assert_eq!(read_f32(&v2), vec![5.0, 7.0, 11.0]);
    }

    #[test]
    fn diagonal_non_square_errors() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let e = diagonal(&t).unwrap_err();
        assert!(e.to_string().contains("square"));
    }
}
