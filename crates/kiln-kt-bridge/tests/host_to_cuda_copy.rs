#![cfg(feature = "cuda")]

//! H2D + D2H round-trip — host_to_cuda_copy ∘ cuda_to_host_copy = identity.
//!
//! # TODO(#1082): candle-removal STOP — by design
//!
//! This test cannot drop its `candle_core::{Device, Tensor}` imports
//! today. It is testing kt-bridge surfaces that are candle-typed by
//! definition:
//!
//! - `kiln_tensor::host_to_cuda_copy(&kt_tensor, Arc<CudaDevice>, usize)`
//!   takes a `candle_core::CudaDevice` (`cuda_storage.rs:2442-2446`).
//!   The test must build `CandleDevice::new_cuda(0)` to feed it.
//! - The `host_to_cuda_copy_validates_input_device` test verifies the
//!   CPU-source-required error path by constructing a CUDA-side
//!   `CandleTensor::from_vec` and bridging it via
//!   `kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow` — that
//!   borrow helper exists specifically to bridge candle→kt and is
//!   not removable until candle is.
//!
//! Until the substrate-level `Arc<CudaDevice>` dep on
//! `host_to_cuda_copy` is broken (Tier 4-5 substrate work tracked
//! in `docs/CANDLE_REMOVAL_PLAN.md` lines 579-585), the candle
//! imports here stay.
//!
//! Precedent for STOP-doc-only commits in this sweep: 6d3fc88d
//! (kiln-opd-loss-kernel), 9a95adc2 (kiln-flce-kernel), acd00bb4
//! (kiln-rmsnorm-kernel phase10_microbench).

use std::sync::Arc;

use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_to_host_copy, host_to_cuda_copy, CpuStorage, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn make_cpu_tensor_f32(data: &[f32], shape: Vec<usize>) -> Tensor {
    Tensor::from_slice(data, shape).unwrap()
}

#[test]
fn host_to_cuda_to_host_round_trip_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let candle_cuda = match dev {
        CandleDevice::Cuda(c) => Arc::new(c),
        _ => unreachable!(),
    };

    let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5 - 4.0).collect();
    let cpu_t = make_cpu_tensor_f32(&data, vec![3, 8]);

    // H2D: CPU kt-Tensor → CUDA kt-Tensor.
    let cuda_t = host_to_cuda_copy(&cpu_t, Arc::clone(&candle_cuda), 0).expect("H2D");
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
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let candle_cuda = match dev.clone() {
        CandleDevice::Cuda(c) => Arc::new(c),
        _ => unreachable!(),
    };

    // Build a CUDA-side tensor (not CPU). host_to_cuda_copy must error.
    let cd = CandleTensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &dev).unwrap();
    let kt_cuda = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    let result = host_to_cuda_copy(&kt_cuda, candle_cuda, 0);
    assert!(result.is_err(), "expected CPU-source-required error");
}
