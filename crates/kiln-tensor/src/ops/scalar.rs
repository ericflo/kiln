//! Tensor-scalar elementwise ops: `add_scalar`, `mul_scalar`,
//! `pow_scalar` (alias for `pow`).
//!
//! Different from `add`/`mul` which require two same-shape tensors.
//! These take a tensor + an f32 scalar; output has the same shape.
//!
//! Saves having to broadcast_to a same-shape filled tensor for every
//! call site that just wants `x * 0.5`.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

fn apply(f: impl Fn(f32) -> f32, x: &Tensor, name: &str) -> Result<Tensor> {
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("{name}: input must be contiguous");
    }
    let dtype = x.dtype();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("scalar: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = x.element_count();
    let per = dtype.size_in_bytes();
    let mut out = vec![0u8; n * per];
    for i in 0..n {
        let v = match dtype {
            DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            DType::F16 => half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            _ => unreachable!(),
        };
        let y = f(v);
        match dtype {
            DType::F32 => out[i * 4..i * 4 + 4].copy_from_slice(&y.to_le_bytes()),
            DType::BF16 => out[i * 2..i * 2 + 2]
                .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes()),
            DType::F16 => out[i * 2..i * 2 + 2]
                .copy_from_slice(&half::f16::from_f32(y).to_le_bytes()),
            _ => unreachable!(),
        }
    }
    let cpu_out = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(x.shape().to_vec()), TensorId::next())
}

pub fn add_scalar(x: &Tensor, c: f32) -> Result<Tensor> {
    apply(|v| v + c, x, "add_scalar")
}
pub fn sub_scalar(x: &Tensor, c: f32) -> Result<Tensor> {
    apply(|v| v - c, x, "sub_scalar")
}
pub fn mul_scalar(x: &Tensor, c: f32) -> Result<Tensor> {
    apply(|v| v * c, x, "mul_scalar")
}
pub fn div_scalar(x: &Tensor, c: f32) -> Result<Tensor> {
    if c == 0.0 {
        bail!("div_scalar: divisor must be non-zero");
    }
    apply(|v| v / c, x, "div_scalar")
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
    fn add_scalar_works() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = add_scalar(&x, 10.0).unwrap();
        assert_eq!(read_f32(&y), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn sub_scalar_works() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = sub_scalar(&x, 1.0).unwrap();
        assert_eq!(read_f32(&y), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn mul_scalar_works() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = mul_scalar(&x, 0.5).unwrap();
        assert_eq!(read_f32(&y), vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn div_scalar_works() {
        let x = Tensor::from_slice(&[2.0f32, 4.0, 6.0], vec![3]).unwrap();
        let y = div_scalar(&x, 2.0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn div_by_zero_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = div_scalar(&x, 0.0).unwrap_err();
        assert!(e.to_string().contains("non-zero"));
    }

    #[test]
    fn bf16_round_trips() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0].iter().map(|&v| half::bf16::from_f32(v)).collect();
        let x = Tensor::from_slice(&bf, vec![2]).unwrap();
        assert_eq!(add_scalar(&x, 1.0).unwrap().dtype(), DType::BF16);
    }
}
