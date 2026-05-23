//! `zeros_like`, `ones_like`, `full_like` — constructors that match
//! the shape + dtype of an existing tensor.
//!
//! Convenience helpers; used heavily in autograd backward
//! implementations to build same-shape gradient placeholders.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

pub fn zeros_like(t: &Tensor) -> Result<Tensor> {
    full_like(t, 0.0)
}

pub fn ones_like(t: &Tensor) -> Result<Tensor> {
    full_like(t, 1.0)
}

pub fn full_like(t: &Tensor, value: f32) -> Result<Tensor> {
    if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "full_like: dtype must be F32/BF16/F16, got {}",
            t.dtype()
        );
    }
    let dtype = t.dtype();
    let n = t.element_count();
    let per = dtype.size_in_bytes();
    let mut bytes = vec![0u8; n * per];
    let one_bytes = match dtype {
        DType::F32 => value.to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f32(value).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f32(value).to_le_bytes().to_vec(),
        _ => unreachable!(),
    };
    if value != 0.0 {
        for i in 0..n {
            bytes[i * per..(i + 1) * per].copy_from_slice(&one_bytes);
        }
    }
    let _ = Error::from_str;
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(t.shape().to_vec()), TensorId::next())
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
    fn zeros_like_matches_shape() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let z = zeros_like(&t).unwrap();
        assert_eq!(z.shape(), t.shape());
        assert_eq!(z.dtype(), t.dtype());
        assert_eq!(read_f32(&z), vec![0.0; 4]);
    }

    #[test]
    fn ones_like_matches_shape() {
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let o = ones_like(&t).unwrap();
        assert_eq!(read_f32(&o), vec![1.0, 1.0]);
    }

    #[test]
    fn full_like_arbitrary_value() {
        let t = Tensor::from_slice(&[0.0f32; 3], vec![3]).unwrap();
        let y = full_like(&t, 3.14).unwrap();
        for v in read_f32(&y) {
            assert!((v - 3.14).abs() < 1e-6);
        }
    }

    #[test]
    fn bf16_round_trips() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let t = Tensor::from_slice(&bf, vec![2]).unwrap();
        let z = zeros_like(&t).unwrap();
        let o = ones_like(&t).unwrap();
        assert_eq!(z.dtype(), DType::BF16);
        assert_eq!(o.dtype(), DType::BF16);
    }

    #[test]
    fn rejects_bad_dtype() {
        let t = Tensor::from_slice(&[1u32], vec![1]).unwrap();
        let e = zeros_like(&t).unwrap_err();
        assert!(e.to_string().contains("F32/BF16/F16"));
    }
}
