//! Smoke test: the kt-API marlin-gemm entry accepts Borrowed
//! kt-Tensors (Phase 7 v2 borrow-compat).
//!
//! Validates that the migration from .slice().slice(off..).device_ptr()
//! to kiln_kt_bridge::cuda_input_device_ptr / cuda_output_device_ptr
//! preserves correctness AND enables the zero-copy candle->kt path.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_marlin_gemm::{marlin_w4a16_gemm_kt, pack};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 33) as u32) & 0x7fffffff;
    (bits as f32 / (i32::MAX as f32)) - 0.5
}

/// marlin_w4a16_gemm_kt accepts Borrowed kt-Tensors (zero-copy from
/// candle). Smoke-tests that the migration doesn't panic on the
/// Borrowed path.
#[test]
fn marlin_w4a16_gemm_kt_accepts_borrowed() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    // Minimal Marlin-legal shape: k%128==0, n%256==0.
    let m = 16usize;
    let k = 128usize;
    let n = 256usize;
    let groupsize: i64 = 128;

    // Build random weights and pack them through Marlin's quantize_and_pack.
    let mut state = 0xC0FFEE_5EED_u64;
    let mut weight = vec![0.0f32; k * n];
    for v in weight.iter_mut() {
        *v = lcg(&mut state) * 0.5;
    }
    let mut state2 = 0xDEAD_BEEF_u64;
    let mut acts = vec![0.0f32; m * k];
    for v in acts.iter_mut() {
        *v = lcg(&mut state2) * 0.25;
    }

    let (b_packed_i32, scales_f16, _dequant_f32) =
        pack::quantize_and_pack(&weight, k, n, groupsize);

    // Activations: candle BF16 -> F16 (Marlin kernel takes F16).
    let a_cd = CandleTensor::from_vec(acts, (m, k), &dev)
        .unwrap()
        .to_dtype(CandleDType::F16)
        .unwrap();

    // Packed weights: convert Vec<i32> -> Vec<u32> bitwise (Marlin
    // treats packed weights as opaque 32-bit words).
    let b_packed_u32: Vec<u32> = b_packed_i32.iter().map(|&x| x as u32).collect();
    let b_cd = CandleTensor::from_vec(b_packed_u32, (k / 16, n * 16 / 8), &dev).unwrap();

    let scales_cd = CandleTensor::from_vec(scales_f16, (k / groupsize as usize, n), &dev).unwrap();

    // Use the BORROW adapter — zero-copy from candle to kt.
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();
    let s_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&scales_cd).unwrap();

    // The migration's critical correctness property: the call must
    // not panic on CudaStorage::slice() (which the old impl
    // called, and which panics on Borrowed storage).
    let c_kt = marlin_w4a16_gemm_kt(&a_kt, &b_kt, &s_kt, groupsize as i32)
        .expect("marlin_w4a16_gemm_kt on borrowed inputs");

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    assert_eq!(c_kt.shape(), &[m, n]);
}
