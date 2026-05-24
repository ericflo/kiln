//! Smoke test: the kt-API rmsnorm-kernel entry accepts Borrowed
//! kt-Tensors (Phase 7 v2 borrow-compat).
//!
//! Validates that the migration from .slice().slice(off..).device_ptr()
//! to kiln_kt_bridge::cuda_input_device_ptr / cuda_output_device_ptr
//! preserves correctness AND enables the zero-copy candle->kt path.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_rmsnorm_kernel::fused_rmsnorm_kt;

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

/// fused_rmsnorm_kt accepts Borrowed kt-Tensors (zero-copy from
/// candle). Smoke-tests that the migration doesn't panic on the
/// Borrowed path.
#[test]
fn fused_rmsnorm_kt_accepts_borrowed() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let rows = 4usize;
    let hidden = 64usize;

    // fused_rmsnorm_kt: x [rows, hidden] BF16, weight [hidden] BF16,
    // returns [rows, hidden] BF16.
    let x_cd = CandleTensor::from_vec(pattern(rows * hidden, 1), (rows, hidden), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let w_cd = CandleTensor::from_vec(pattern(hidden, 2), (hidden,), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();

    // Use the BORROW adapter — zero-copy from candle to kt.
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&w_cd).unwrap();

    // The migration's critical correctness property: the call must
    // not panic on CudaStorage::slice() (which the old impl
    // called, and which panics on Borrowed storage).
    let out = fused_rmsnorm_kt(&x_kt, &w_kt, 1e-6)
        .expect("fused_rmsnorm_kt on borrowed inputs");

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    assert_eq!(out.shape(), &[rows, hidden]);
}
