#![cfg(feature = "cuda")]

//! `Tensor::to_device` — unified device-transfer API surface (issue #1082).
//!
//! The method wraps the existing `host_to_cuda_copy_ctx` /
//! `cuda_to_host_copy` helpers behind a single uniform call site. These
//! tests exercise:
//!
//!   1. Same-device clone (Cpu→Cpu): cheap, no copy.
//!   2. CPU → CUDA(0).
//!   3. CUDA(0) → CPU.
//!   4. Round-trip CPU → CUDA → CPU preserves bytes exactly.
//!
//! **Candle-free as of #1082**: `Tensor::to_device` no longer takes an
//! `Option<Arc<CudaDevice>>` — the cudarc context is derived internally.
//! This test file no longer needs the `candle_core::Device` import that
//! the old contract required.

use std::sync::Arc;

use kiln_tensor::{CpuStorage, Device as KtDevice, Tensor};

fn try_cuda_available() -> bool {
    kiln_tensor::primary_cuda_device(0).is_ok()
}

fn cpu_f32_tensor(data: &[f32], shape: Vec<usize>) -> Tensor {
    Tensor::from_slice(data, shape).unwrap()
}

#[test]
fn to_device_same_device_clones_storage() {
    // CPU→CPU must be a same-device fast path (Arc bump, no copy).
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu_t = cpu_f32_tensor(&data, vec![3, 4]);

    let same = cpu_t.to_device(KtDevice::Cpu).expect("cpu→cpu");

    assert_eq!(same.shape(), &[3, 4]);
    assert_eq!(same.device(), KtDevice::Cpu);
    // Same-device move shares the underlying Arc<dyn StorageBackend>.
    assert!(
        Arc::ptr_eq(cpu_t.storage(), same.storage()),
        "same-device to_device must not copy storage"
    );
}

#[test]
fn to_device_cpu_to_cuda_uploads() {
    if !try_cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    let data: Vec<f32> = (0..20).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let cpu_t = cpu_f32_tensor(&data, vec![4, 5]);

    let cuda_t = cpu_t.to_device(KtDevice::Cuda(0)).expect("cpu→cuda");

    assert_eq!(cuda_t.shape(), &[4, 5]);
    assert!(matches!(cuda_t.device(), KtDevice::Cuda(_)));
}

#[test]
fn to_device_cuda_to_cpu_downloads() {
    if !try_cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    let data: Vec<f32> = (0..16).map(|i| (i as f32) - 8.0).collect();
    let cpu_t = cpu_f32_tensor(&data, vec![4, 4]);

    let cuda_t = cpu_t.to_device(KtDevice::Cuda(0)).expect("cpu→cuda");

    let back = cuda_t.to_device(KtDevice::Cpu).expect("cuda→cpu");

    assert_eq!(back.shape(), &[4, 4]);
    assert_eq!(back.device(), KtDevice::Cpu);
}

#[test]
fn to_device_round_trip_preserves_bytes() {
    if !try_cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5 - 4.0).collect();
    let cpu_t = cpu_f32_tensor(&data, vec![3, 8]);

    let cuda_t = cpu_t.to_device(KtDevice::Cuda(0)).expect("cpu→cuda");

    let cpu_back = cuda_t.to_device(KtDevice::Cpu).expect("cuda→cpu");

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
