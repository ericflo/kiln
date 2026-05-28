#![cfg(feature = "cuda")]

//! H2D + D2H round-trip — host_to_cuda_copy ∘ cuda_to_host_copy = identity.
//!
//! # #1082 candle-removal status
//!
//! As of wave 13 of the #1082 sweep, `kiln_tensor::host_to_cuda_copy`
//! is fully candle-free — its signature is `(src, device_index)` and
//! it derives the cudarc `CudaContext` internally via
//! `primary_cuda_context`. The CPU-source validation test that used
//! to need a candle-side `CandleTensor` to build a CUDA-resident kt
//! tensor now constructs that tensor via `kiln_tensor`'s own CPU →
//! CUDA path plus the existing kt → CUDA H2D, so no candle imports
//! remain on this test file.

use kiln_tensor::{cuda_to_host_copy, host_to_cuda_copy, primary_cuda_context, CpuStorage, Tensor};

fn cuda_available() -> bool {
    primary_cuda_context(0).is_ok()
}

fn make_cpu_tensor_f32(data: &[f32], shape: Vec<usize>) -> Tensor {
    Tensor::from_slice(data, shape).unwrap()
}

#[test]
fn host_to_cuda_to_host_round_trip_f32() {
    if !cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5 - 4.0).collect();
    let cpu_t = make_cpu_tensor_f32(&data, vec![3, 8]);

    // H2D: CPU kt-Tensor → CUDA kt-Tensor.
    let cuda_t = host_to_cuda_copy(&cpu_t, 0).expect("H2D");
    assert_eq!(cuda_t.shape(), &[3, 8]);
    assert!(matches!(cuda_t.device(), kiln_tensor::Device::Cuda(_)));

    // D2H back.
    let cpu_back = cuda_to_host_copy(&cuda_t).expect("D2H");
    assert_eq!(cpu_back.shape(), &[3, 8]);
    assert!(matches!(cpu_back.device(), kiln_tensor::Device::Cpu));

    let cpu_storage = cpu_back
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    assert_eq!(bytes.len(), 24 * 4);
    for i in 0..24 {
        let v = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(v, data[i], "round-trip byte {i} mismatch");
    }
}

#[test]
fn host_to_cuda_copy_validates_input_device() {
    if !cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    // Build a CUDA-side tensor (not CPU) via the kt H2D path itself,
    // then feed it back into host_to_cuda_copy. The function must
    // reject the non-CPU source.
    let cpu_seed =
        Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).expect("cpu seed");
    let kt_cuda = host_to_cuda_copy(&cpu_seed, 0).expect("seed H2D");
    assert!(matches!(kt_cuda.device(), kiln_tensor::Device::Cuda(_)));

    let result = host_to_cuda_copy(&kt_cuda, 0);
    assert!(result.is_err(), "expected CPU-source-required error");
}
