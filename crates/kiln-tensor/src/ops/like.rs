//! `zeros_like`, `ones_like`, `full_like` — constructors that match
//! the shape + dtype of an existing tensor.
//!
//! Convenience helpers; used heavily in autograd backward
//! implementations to build same-shape gradient placeholders.

use std::sync::Arc;

use crate::{CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId, bail};

pub fn zeros_like(t: &Tensor) -> Result<Tensor> {
    full_like(t, 0.0)
}

pub fn ones_like(t: &Tensor) -> Result<Tensor> {
    full_like(t, 1.0)
}

pub fn full_like(t: &Tensor, value: f32) -> Result<Tensor> {
    if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("full_like: dtype must be F32/BF16/F16, got {}", t.dtype());
    }

    // CUDA fast path: build on host, copy to the same CUDA device.
    // For value == 0.0 this could be served by `cuda_zeros` directly,
    // but host_to_cuda_copy handles arbitrary values uniformly and is
    // dominated by the H2D transfer anyway.
    #[cfg(feature = "cuda")]
    if let crate::Device::Cuda(device_index) = t.device() {
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
        let cpu = CpuStorage::from_bytes(dtype, bytes)?;
        let storage: Storage = Arc::new(cpu);
        let cpu_t = Tensor::from_parts(
            storage,
            Layout::contiguous(t.shape().to_vec()),
            TensorId::next(),
        )?;
        // Lift to CUDA on the same device as the source.
        //
        // host_to_cuda_copy_ctx (#1082) derives the candle device
        // internally from device_index, so we don't need to downcast
        // the source storage just to forward .candle_device().clone()
        // into the helper.
        return crate::host_to_cuda_copy_ctx(&cpu_t, device_index);
    }

    // ROCm fast path: build the placeholder ON-DEVICE (no host round-trip).
    // The generic path below stages a CPU tensor and `to_device`-uploads it,
    // which does an `host_to_rocm_copy` (htod). That htod is ILLEGAL inside a
    // HIP-graph capture region and ABORTS it — softplus/relu build a
    // `zeros_like` placeholder per GDN layer *inside* the captured decode
    // forward, so the upload there was the dominant capture blocker (24 GDN
    // layers => 24 aborting htods/step). `rocm_zeros_ctx` is capture-arena-
    // aware: under an active arena it mints a frozen-pointer buffer and
    // memsets it on-device, keeping the region sync- and host-copy-free.
    // For value == 0.0 the zeros buffer is the answer directly; nonzero fills
    // (`ones_like` etc.) add the scalar on-device via the ScalarOp rocm_fwd,
    // staying capture-safe.
    #[cfg(feature = "rocm")]
    if let crate::Device::Rocm(device_index) = t.device() {
        let dtype = t.dtype();
        let n = t.element_count();
        let storage = crate::rocm_zeros_ctx(device_index, dtype, n)?;
        let zeros = Tensor::from_parts(
            storage,
            Layout::contiguous(t.shape().to_vec()),
            TensorId::next(),
        )?;
        if value == 0.0 {
            return Ok(zeros);
        }
        return crate::ops::scalar::add_scalar(&zeros, value);
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
    let cpu_t = Tensor::from_parts(
        storage,
        Layout::contiguous(t.shape().to_vec()),
        TensorId::next(),
    )?;
    // Honor the source device: a same-shape placeholder must live where the
    // source tensor lives, or downstream elementwise ops hit a cross-device
    // mismatch (#1082 — `ones_like` on a Metal weight must be Metal). CUDA is
    // served by the fast path above; CPU is a no-op clone; Metal uploads to a
    // Shared UMA buffer.
    cpu_t.to_device(t.device())
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
