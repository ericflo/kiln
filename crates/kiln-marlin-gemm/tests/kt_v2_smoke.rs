//! Candle-free smoke test for the kt-API Marlin W4A16 GEMM entry.
//!
//! Constructs CUDA tensors directly via `Tensor::cuda_from_slice`
//! (substrate helper #1082 / `a5da6152`) — no `candle_core` import
//! required. Exercises `marlin_w4a16_gemm_kt` on a minimal
//! Marlin-legal shape (k%128==0, n%256==0).
//!
//! The legacy "BORROW adapter" smoke moved to
//! `crates/kiln-kt-bridge/tests/` where the adapter actually lives.
//! This file tests the kt API in isolation.
//!
//! CUDA-only: it builds tensors through `Tensor::cuda_from_slice` /
//! `primary_cuda_context`, which exist only under the `cuda` feature. The ROCm
//! lane is covered by `tests/rocm_marlin_parity.rs`.
#![cfg(feature = "cuda")]

use half::f16;

use kiln_marlin_gemm::{marlin_w4a16_gemm_kt, pack};
use kiln_tensor::{DType, Tensor};

fn cuda_available() -> bool {
    kiln_tensor::primary_cuda_context(0).is_ok()
}

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 33) as u32) & 0x7fffffff;
    (bits as f32 / (i32::MAX as f32)) - 0.5
}

/// `marlin_w4a16_gemm_kt` dispatches on candle-free kt CUDA inputs.
#[test]
fn marlin_w4a16_gemm_kt_dispatches_on_minimal_shape() {
    if !cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    // Minimal Marlin-legal shape: k%128==0, n%256==0.
    let m = 16usize;
    let k = 128usize;
    let n = 256usize;
    let groupsize: i64 = 128;

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

    // Pack returns host-side Vec<i32> (b_packed) + Vec<f16> (scales).
    let (b_packed_i32, scales_f16, _dequant_f32) =
        pack::quantize_and_pack(&weight, k, n, groupsize);

    // Activations: F16 (Marlin kernel takes F16). Convert host-side
    // from f32 to f16 to avoid going through candle's to_dtype.
    let acts_f16: Vec<f16> = acts.iter().map(|&v| f16::from_f32(v)).collect();
    let a_kt = Tensor::cuda_from_slice(&acts_f16, vec![m, k], 0).expect("a");

    // Packed weights are i32 → u32 bitwise (Marlin treats packed as
    // opaque 32-bit words). The bridge's I32→U32 mapping
    // (`668b0847`) lets the kernel consume the U32 surface.
    let b_packed_u32: Vec<u32> = b_packed_i32.iter().map(|&x| x as u32).collect();
    let b_kt = Tensor::cuda_from_slice(&b_packed_u32, vec![k / 16, n * 16 / 8], 0).expect("b");

    let s_kt = Tensor::cuda_from_slice(&scales_f16, vec![k / groupsize as usize, n], 0).expect("s");

    let c_kt =
        marlin_w4a16_gemm_kt(&a_kt, &b_kt, &s_kt, groupsize as i32).expect("marlin_w4a16_gemm_kt");

    assert_eq!(c_kt.shape(), &[m, n]);
    assert_eq!(c_kt.dtype(), DType::F16);
}
