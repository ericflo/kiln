//! Precision-casting convenience helpers: `to_f32`, `to_bf16`,
//! `to_f16`. Wraps `cast` with a typed shortcut.
//!
//! Useful for AMP mixed-precision pipelines where the same call
//! site needs both F32 and BF16 paths.

use crate::ops::cast;
use crate::{DType, Result, Tensor};

pub fn to_f32(x: &Tensor) -> Result<Tensor> {
    cast(x, DType::F32)
}

pub fn to_bf16(x: &Tensor) -> Result<Tensor> {
    cast(x, DType::BF16)
}

pub fn to_f16(x: &Tensor) -> Result<Tensor> {
    cast(x, DType::F16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuStorage;

    #[test]
    fn to_f32_from_bf16() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0].iter().map(|&v| half::bf16::from_f32(v)).collect();
        let x = Tensor::from_slice(&bf, vec![2]).unwrap();
        let y = to_f32(&x).unwrap();
        assert_eq!(y.dtype(), DType::F32);
        let cpu = y.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let v: Vec<f32> = cpu
            .as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(v, vec![1.0, 2.0]);
    }

    #[test]
    fn to_bf16_from_f32() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let y = to_bf16(&x).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
    }

    #[test]
    fn to_f16_from_f32() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let y = to_f16(&x).unwrap();
        assert_eq!(y.dtype(), DType::F16);
    }

    #[test]
    fn round_trip_bf16() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bf = to_bf16(&x).unwrap();
        let back = to_f32(&bf).unwrap();
        assert_eq!(back.dtype(), DType::F32);
    }
}
