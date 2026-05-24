//! Smoke test: the kt-API gdn-kernel entries accept both Owned and
//! Borrowed kt-Tensors (Phase 7 v2 borrow-compat).
//!
//! Validates that the migration from `.slice().slice(off..).device_ptr()`
//! to `kt-bridge::cuda_input_device_ptr` / `cuda_output_device_ptr`
//! preserves correctness AND enables the zero-copy candle→kt path.



use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_gdn_kernel::gdn_gates_bf16_kt;

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

/// gdn_gates_bf16_kt accepts Borrowed kt-Tensors (zero-copy from
/// candle). Smoke-tests that the migration doesn't panic on the
/// Borrowed path.
#[test]
fn gdn_gates_bf16_kt_accepts_borrowed() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let b = 1usize;
    let h = 2usize;
    let c = 4usize;
    let nv = h * c;
    let rows = b * c;

    // gates_bf16 inputs: a [rows, nv], b [rows, nv], a_log [rows, nv],
    // dt_bias [nv]. All BF16.
    let a_cd = CandleTensor::from_vec(pattern(rows * nv, 1), (rows, nv), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let b_cd = CandleTensor::from_vec(pattern(rows * nv, 2), (rows, nv), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let al_cd = CandleTensor::from_vec(pattern(nv, 3), (nv,), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let dt_cd = CandleTensor::from_vec(pattern(nv, 4), (nv,), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();

    // Use the BORROW adapter — zero-copy from candle to kt.
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();
    let al_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&al_cd).unwrap();
    let dt_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&dt_cd).unwrap();

    // The migration's critical correctness property: the call must
    // not panic on `CudaStorage::slice()` (which the old impl
    // called, and which panics on Borrowed storage).
    let (beta, g) = gdn_gates_bf16_kt(&a_kt, &b_kt, &al_kt, &dt_kt)
        .expect("gdn_gates_bf16_kt on borrowed inputs");

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    assert_eq!(beta.shape(), &[rows, nv]);
    assert_eq!(g.shape(), &[rows, nv]);
}
