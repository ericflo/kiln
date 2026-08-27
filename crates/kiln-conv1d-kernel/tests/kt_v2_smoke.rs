//! Candle-free smoke tests for the kt-API conv1d kernel entries.
//!
//! Constructs CUDA tensors directly via the `Tensor::cuda_from_slice`
//! substrate helper (#1082 / `a5da6152`) — no `candle_core` import
//! required. Exercises the `causal_conv1d_update_kt`,
//! `causal_conv1d_prefill_kt`, `supports_kt`, and `supports_prefill_kt`
//! entry points on the Qwen3.5-4B decode/prefill envelopes.
//!
//! This file tests the kt API in isolation.
//!
//! CUDA-only: `Tensor::cuda_from_slice` / `primary_cuda_context` are the
//! cuda-storage substrate helpers and don't exist on the ROCm build. The
//! backend-neutral ROCm parity coverage lives in `rocm_conv1d_parity.rs`
//! (gated on `feature = "rocm"`), which constructs inputs via
//! `Tensor::from_vec_on(Device::Rocm(0), ...)`.
#![cfg(feature = "cuda")]

use half::bf16;

use kiln_conv1d_kernel::{
    causal_conv1d_prefill_kt, causal_conv1d_update_kt, supports_kt, supports_prefill_kt,
};
use kiln_tensor::Tensor;

fn cuda_available() -> bool {
    kiln_tensor::primary_cuda_context(0).is_ok()
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

/// `causal_conv1d_update_kt` dispatches successfully on candle-free
/// kt CUDA inputs at the Qwen3.5-4B decode envelope.
#[test]
fn causal_conv1d_update_kt_dispatches_on_qwen_decode_shape() {
    if !cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    let batch = 1usize;
    let channels = 8usize;
    let kernel_size = 4usize;

    let x = Tensor::cuda_from_slice(
        &pattern_bf16(batch * channels, 1),
        vec![batch, channels, 1],
        0,
    )
    .expect("x");
    let w = Tensor::cuda_from_slice(
        &pattern_bf16(channels * kernel_size, 2),
        vec![channels, kernel_size],
        0,
    )
    .expect("w");
    let cs = Tensor::cuda_from_slice(
        &pattern(batch * channels * (kernel_size - 1), 3),
        vec![batch, channels, kernel_size - 1],
        0,
    )
    .expect("cs");

    let out = causal_conv1d_update_kt(&x, &w, &cs, kernel_size).expect("causal_conv1d_update_kt");
    assert_eq!(out.shape(), &[batch, channels, 1]);
}

/// `causal_conv1d_prefill_kt` dispatches on the Qwen3.5 prefill
/// envelope (B=1, C=8, T=8, K=4) — candle-free inputs.
#[test]
fn causal_conv1d_prefill_kt_dispatches_on_qwen_prefill_shape() {
    if !cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    let batch = 1usize;
    let channels = 8usize;
    let seq_len = 8usize;
    let kernel_size = 4usize;

    let x = Tensor::cuda_from_slice(
        &pattern_bf16(batch * channels * seq_len, 1),
        vec![batch, channels, seq_len],
        0,
    )
    .expect("x");
    let w = Tensor::cuda_from_slice(
        &pattern_bf16(channels * kernel_size, 2),
        vec![channels, kernel_size],
        0,
    )
    .expect("w");
    let cs = Tensor::cuda_from_slice(
        &pattern(batch * channels * (kernel_size - 1), 3),
        vec![batch, channels, kernel_size - 1],
        0,
    )
    .expect("cs");

    let out = causal_conv1d_prefill_kt(&x, &w, &cs, kernel_size).expect("causal_conv1d_prefill_kt");
    assert_eq!(out.shape(), &[batch, channels, seq_len]);
}

/// `supports_kt` / `supports_prefill_kt` predicates accept the
/// Qwen3.5-4B envelope and reject wrong widths / wrong seq_len —
/// candle-free.
#[test]
fn supports_kt_accepts_qwen_envelope_and_rejects_wrong_width() {
    if !cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    let batch = 1usize;
    let channels = 8usize;
    let kernel_size = 4usize;

    let x_dec = Tensor::cuda_from_slice(
        &vec![bf16::from_f32(0.0); batch * channels],
        vec![batch, channels, 1],
        0,
    )
    .expect("x_dec");
    let w = Tensor::cuda_from_slice(
        &vec![bf16::from_f32(0.0); channels * kernel_size],
        vec![channels, kernel_size],
        0,
    )
    .expect("w");
    let cs = Tensor::cuda_from_slice(
        &vec![0f32; batch * channels * (kernel_size - 1)],
        vec![batch, channels, kernel_size - 1],
        0,
    )
    .expect("cs");

    assert!(supports_kt(&x_dec, &w, &cs, kernel_size));
    // Wrong kernel_size rejected.
    assert!(!supports_kt(&x_dec, &w, &cs, 3));

    let seq_len = 8usize;
    let x_pre = Tensor::cuda_from_slice(
        &vec![bf16::from_f32(0.0); batch * channels * seq_len],
        vec![batch, channels, seq_len],
        0,
    )
    .expect("x_pre");

    assert!(supports_prefill_kt(&x_pre, &w, &cs, kernel_size));
    // Prefill rejects T=1.
    assert!(!supports_prefill_kt(&x_dec, &w, &cs, kernel_size));
}
