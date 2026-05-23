//! `trace` — sum of diagonal elements of a rank-2 square tensor.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

pub fn trace(t: &Tensor) -> Result<Tensor> {
    if t.rank() != 2 {
        bail!("trace: input must be rank-2, got {:?}", t.shape());
    }
    let n = t.shape()[0];
    if t.shape()[1] != n {
        bail!("trace: input must be square, got {:?}", t.shape());
    }
    if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("trace: dtype must be F32/BF16/F16, got {}", t.dtype());
    }
    if !t.is_contiguous() {
        bail!("trace: input must be contiguous");
    }
    let dtype = t.dtype();
    let bytes = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("trace: storage must be CpuStorage"))?
        .as_bytes();
    let mut sum = 0.0_f32;
    for i in 0..n {
        let idx = i * n + i;
        let v = match dtype {
            DType::F32 => f32::from_le_bytes(bytes[idx * 4..idx * 4 + 4].try_into().unwrap()),
            DType::BF16 => half::bf16::from_le_bytes(bytes[idx * 2..idx * 2 + 2].try_into().unwrap())
                .to_f32(),
            DType::F16 => half::f16::from_le_bytes(bytes[idx * 2..idx * 2 + 2].try_into().unwrap())
                .to_f32(),
            _ => unreachable!(),
        };
        sum += v;
    }
    let out_bytes = match dtype {
        DType::F32 => sum.to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f32(sum).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f32(sum).to_le_bytes().to_vec(),
        _ => unreachable!(),
    };
    let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(Vec::<usize>::new()), TensorId::next())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_f32(t: &Tensor) -> f32 {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        f32::from_le_bytes(cpu.as_bytes()[..4].try_into().unwrap())
    }

    #[test]
    fn trace_identity_is_n() {
        use crate::ops::eye;
        for n in [1, 2, 5, 10].iter() {
            let i = eye(*n, DType::F32).unwrap();
            assert!((scalar_f32(&trace(&i).unwrap()) - *n as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn trace_arbitrary() {
        // [[1, 0, 0], [0, 5, 0], [0, 0, 9]] → 15
        let t = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 9.0], vec![3, 3]).unwrap();
        assert!((scalar_f32(&trace(&t).unwrap()) - 15.0).abs() < 1e-5);
    }

    #[test]
    fn trace_non_square_errors() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let e = trace(&t).unwrap_err();
        assert!(e.to_string().contains("square"));
    }
}
