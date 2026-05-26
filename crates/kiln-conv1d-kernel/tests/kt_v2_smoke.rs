//! Smoke test: the kt-API conv1d-kernel entry accepts Borrowed
//! kt-Tensors (Phase 7 v2 borrow-compat).
//!
//! Validates that the migration from .slice().slice(off..).device_ptr()
//! to kiln_kt_bridge::cuda_input_device_ptr / cuda_output_device_ptr
//! preserves correctness AND enables the zero-copy candle->kt path.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_conv1d_kernel::{
    causal_conv1d_prefill_kt, causal_conv1d_update_kt, supports_kt, supports_prefill_kt,
};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        out.push(((s as u32 % 1024) as f32 - 512.0) / 512.0);
    }
    out
}

/// causal_conv1d_update_kt accepts Borrowed kt-Tensors (zero-copy from
/// candle). Smoke-tests that the migration doesn't panic on the
/// Borrowed path.
#[test]
fn causal_conv1d_update_kt_accepts_borrowed() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let batch = 1usize;
    let channels = 8usize;
    let kernel_size = 4usize;

    // x [B, C, 1] BF16
    let x_cd = CandleTensor::from_vec(pattern(batch * channels, 1), (batch, channels, 1), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    // weight [C, K] BF16
    let w_cd = CandleTensor::from_vec(pattern(channels * kernel_size, 2), (channels, kernel_size), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    // conv_state [B, C, K-1] F32
    let cs_cd = CandleTensor::from_vec(
        pattern(batch * channels * (kernel_size - 1), 3),
        (batch, channels, kernel_size - 1),
        &dev,
    )
    .unwrap()
    .to_dtype(CandleDType::F32)
    .unwrap();

    // Use the BORROW adapter — zero-copy from candle to kt.
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&w_cd).unwrap();
    let cs_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cs_cd).unwrap();

    // The migration's critical correctness property: the call must
    // not panic on CudaStorage::slice() (which the old impl
    // called, and which panics on Borrowed storage).
    let out = causal_conv1d_update_kt(&x_kt, &w_kt, &cs_kt, kernel_size)
        .expect("causal_conv1d_update_kt on borrowed inputs");

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    assert_eq!(out.shape(), &[batch, channels, 1]);
}

/// causal_conv1d_prefill_kt accepts Borrowed kt-Tensors (zero-copy from
/// candle). Mirrors the update smoke at the prefill envelope (seq_len > 1).
/// Forward-looking coverage for the eventual migration of the in-lib
/// `parity_qwen35_prefill_shape` test off candle.
#[test]
fn causal_conv1d_prefill_kt_accepts_borrowed() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let batch = 1usize;
    let channels = 8usize;
    let seq_len = 8usize;
    let kernel_size = 4usize;

    let x_cd = CandleTensor::from_vec(
        pattern(batch * channels * seq_len, 1),
        (batch, channels, seq_len),
        &dev,
    )
    .unwrap()
    .to_dtype(CandleDType::BF16)
    .unwrap();
    let w_cd = CandleTensor::from_vec(pattern(channels * kernel_size, 2), (channels, kernel_size), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let cs_cd = CandleTensor::from_vec(
        pattern(batch * channels * (kernel_size - 1), 3),
        (batch, channels, kernel_size - 1),
        &dev,
    )
    .unwrap()
    .to_dtype(CandleDType::F32)
    .unwrap();

    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&w_cd).unwrap();
    let cs_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cs_cd).unwrap();

    let out = causal_conv1d_prefill_kt(&x_kt, &w_kt, &cs_kt, kernel_size)
        .expect("causal_conv1d_prefill_kt on borrowed inputs");

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    assert_eq!(out.shape(), &[batch, channels, seq_len]);
}

/// Predicate sanity: supports_kt + supports_prefill_kt accept the
/// Qwen3.5-4B envelope and reject wrong widths / wrong seq_len.
/// Kt-typed twin of the in-lib `supports_accepts_qwen_shape` and
/// `supports_rejects_wrong_width` tests at lib.rs — forward-looking
/// coverage for the eventual migration of the in-lib predicate tests
/// off candle.
#[test]
fn supports_kt_accepts_qwen_envelope_and_rejects_wrong_width() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let batch = 1usize;
    let channels = 8usize;
    let kernel_size = 4usize;

    let x_dec = CandleTensor::from_vec(vec![0f32; batch * channels], (batch, channels, 1), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let w = CandleTensor::from_vec(vec![0f32; channels * kernel_size], (channels, kernel_size), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let cs = CandleTensor::from_vec(
        vec![0f32; batch * channels * (kernel_size - 1)],
        (batch, channels, kernel_size - 1),
        &dev,
    )
    .unwrap()
    .to_dtype(CandleDType::F32)
    .unwrap();

    let x_dec_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_dec).unwrap();
    let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&w).unwrap();
    let cs_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cs).unwrap();

    assert!(supports_kt(&x_dec_kt, &w_kt, &cs_kt, kernel_size));
    // Wrong kernel_size rejected.
    assert!(!supports_kt(&x_dec_kt, &w_kt, &cs_kt, 3));

    let seq_len = 8usize;
    let x_pre = CandleTensor::from_vec(
        vec![0f32; batch * channels * seq_len],
        (batch, channels, seq_len),
        &dev,
    )
    .unwrap()
    .to_dtype(CandleDType::BF16)
    .unwrap();
    let x_pre_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_pre).unwrap();

    assert!(supports_prefill_kt(&x_pre_kt, &w_kt, &cs_kt, kernel_size));
    // Prefill rejects T=1 (that's the update envelope).
    assert!(!supports_prefill_kt(&x_dec_kt, &w_kt, &cs_kt, kernel_size));
}
