//! Candle-free smoke test for the kt-API rmsnorm-kernel `fused_rmsnorm_kt`
//! entry. Constructs CUDA tensors directly via the `Tensor::cuda_from_slice`
//! substrate helper (#1082 / `a5da6152`) — no `candle_core` import
//! required.
//!
//! The legacy "BORROW adapter" smoke (zero-copy candle→kt round-trip)
//! moved to `crates/kiln-kt-bridge/tests/` where the adapter actually
//! lives. This file tests the kt API in isolation, mirroring the
//! `kiln-conv1d-kernel` / `kiln-marlin-gemm` Tier-1 precedent.

use half::bf16;

use kiln_rmsnorm_kernel::fused_rmsnorm_kt;
use kiln_tensor::Tensor;

fn cuda_available() -> bool {
    kiln_tensor::primary_cuda_device(0).is_ok()
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

fn pattern_bf16(n: usize, seed: u64) -> Vec<bf16> {
    pattern(n, seed).into_iter().map(bf16::from_f32).collect()
}

/// `fused_rmsnorm_kt` dispatches successfully on candle-free
/// kt CUDA inputs.
#[test]
fn fused_rmsnorm_kt_dispatches_on_kt_inputs() {
    if !cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    let rows = 4usize;
    let hidden = 64usize;

    // fused_rmsnorm_kt: x [rows, hidden] BF16, weight [hidden] BF16,
    // returns [rows, hidden] BF16.
    let x = Tensor::cuda_from_slice(
        &pattern_bf16(rows * hidden, 1),
        vec![rows, hidden],
        0,
    )
    .expect("x");
    let w = Tensor::cuda_from_slice(
        &pattern_bf16(hidden, 2),
        vec![hidden],
        0,
    )
    .expect("w");

    let out = fused_rmsnorm_kt(&x, &w, 1e-6)
        .expect("fused_rmsnorm_kt on kt inputs");

    assert_eq!(out.shape(), &[rows, hidden]);
}
