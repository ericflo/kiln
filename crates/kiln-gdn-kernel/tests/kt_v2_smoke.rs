//! Candle-free smoke test for the kt-API gdn-kernel `gdn_gates_bf16_kt`
//! entry. Constructs CUDA tensors directly via the `Tensor::cuda_from_slice`
//! substrate helper (#1082 / `a5da6152`) — no `candle_core` import
//! required.
//!
//! The legacy "BORROW adapter" smoke (zero-copy candle→kt round-trip)
//! moved to `crates/kiln-kt-bridge/tests/` where the adapter actually
//! lives. This file tests the kt API in isolation, mirroring the
//! `kiln-conv1d-kernel` / `kiln-marlin-gemm` Tier-1 precedent.

use half::bf16;

use kiln_gdn_kernel::gdn_gates_bf16_kt;
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

/// `gdn_gates_bf16_kt` dispatches successfully on candle-free
/// kt CUDA inputs.
#[test]
fn gdn_gates_bf16_kt_dispatches_on_kt_inputs() {
    if !cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    let b = 1usize;
    let h = 2usize;
    let c = 4usize;
    let nv = h * c;
    let rows = b * c;

    // gates_bf16 inputs: a [rows, nv], b [rows, nv], a_log [nv],
    // dt_bias [nv]. All BF16.
    let a = Tensor::cuda_from_slice(
        &pattern_bf16(rows * nv, 1),
        vec![rows, nv],
        0,
    )
    .expect("a");
    let b_in = Tensor::cuda_from_slice(
        &pattern_bf16(rows * nv, 2),
        vec![rows, nv],
        0,
    )
    .expect("b");
    let a_log = Tensor::cuda_from_slice(
        &pattern_bf16(nv, 3),
        vec![nv],
        0,
    )
    .expect("a_log");
    let dt_bias = Tensor::cuda_from_slice(
        &pattern_bf16(nv, 4),
        vec![nv],
        0,
    )
    .expect("dt_bias");

    let (beta, g) = gdn_gates_bf16_kt(&a, &b_in, &a_log, &dt_bias)
        .expect("gdn_gates_bf16_kt on kt inputs");

    assert_eq!(beta.shape(), &[rows, nv]);
    assert_eq!(g.shape(), &[rows, nv]);
}
