//! `eye` — identity matrix constructor.
//!
//! `eye(n, dtype)` produces a `[n, n]` tensor with `1.0` on the
//! diagonal and `0.0` elsewhere. Used in regularizers (residual
//! identity terms), initialization (identity-as-init for residual
//! blocks), and unit-test fixtures.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

pub fn eye(n: usize, dtype: DType) -> Result<Tensor> {
    if n == 0 {
        bail!("eye: n must be > 0");
    }
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        bail!("eye: dtype must be F32/BF16/F16, got {dtype}");
    }
    let per = dtype.size_in_bytes();
    let one_bytes = match dtype {
        DType::F32 => 1.0f32.to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f32(1.0).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f32(1.0).to_le_bytes().to_vec(),
        _ => unreachable!(),
    };
    let mut bytes = vec![0u8; n * n * per];
    for i in 0..n {
        let off = (i * n + i) * per;
        bytes[off..off + per].copy_from_slice(&one_bytes);
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
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
    fn eye_3() {
        let i = eye(3, DType::F32).unwrap();
        assert_eq!(i.shape(), &[3, 3]);
        assert_eq!(
            read_f32(&i),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn eye_1() {
        let i = eye(1, DType::F32).unwrap();
        assert_eq!(read_f32(&i), vec![1.0]);
    }

    #[test]
    fn eye_matmul_identity_check() {
        use crate::ops::matmul;
        let i = eye(4, DType::F32).unwrap();
        let x = Tensor::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ],
            vec![4, 4],
        )
        .unwrap();
        let y = matmul(&x, &i).unwrap();
        // x @ I = x.
        assert_eq!(read_f32(&y), read_f32(&x));
    }

    #[test]
    fn eye_bf16() {
        let i = eye(2, DType::BF16).unwrap();
        assert_eq!(i.dtype(), DType::BF16);
    }

    #[test]
    fn eye_zero_errors() {
        let e = eye(0, DType::F32).unwrap_err();
        assert!(e.to_string().contains("> 0"));
    }
}
