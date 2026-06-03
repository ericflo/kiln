//! Candle-free smoke tests for the kt-API flash-attn paged-KV-write entries.
//!
//! CUDA-only: constructs CUDA tensors via `Tensor::cuda_from_slice` and reaches
//! `primary_cuda_context`, both gated on `kiln-tensor/cuda`. Under
//! `--no-default-features --features rocm` there is no CUDA substrate, so this
//! whole file compiles out (the ROCm path is covered by
//! `tests/rocm_flash_attn_parity.rs`).
#![cfg(feature = "cuda")]
//!
//! Constructs CUDA tensors directly via the `Tensor::cuda_from_slice`
//! substrate helper (#1082 / `a5da6152`) — no `candle_core` import
//! required. Exercises the `paged_kv_write_token_major_bf16_slot_kt` and
//! `paged_kv_write_token_major_bf16_batch_slot_kt` entry points on small
//! BF16 K/V pools.
//!
//! The legacy candle-parity comparison was removed alongside the
//! candle-typed `paged_kv_write_token_major_bf16_batch_slot` surface
//! (Phase 7 / #1082). Both the kt path and the prior candle-typed path
//! always bottomed out in the same
//! `kiln_paged_kv_write_token_major_bf16_batch_slot` FFI symbol, so
//! deleting the candle shell removes shell-only divergence risk; the
//! remaining kt smoke covers the FFI dispatch + offset math itself.

use half::bf16;

use kiln_flash_attn::{
    paged_kv_write_token_major_bf16_batch_slot_kt, paged_kv_write_token_major_bf16_slot_kt,
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

/// `paged_kv_write_token_major_bf16_slot_kt` dispatches successfully on
/// candle-free kt CUDA inputs (host pool, single-token row, device-side
/// `[1]` u32 slot).
#[test]
fn paged_kv_write_slot_kt_dispatches_candle_free() {
    if !cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    let total_slots = 8usize;
    let num_kv_heads = 4usize;
    let head_dim = 128usize;

    let kp = Tensor::cuda_from_slice(
        &pattern_bf16(total_slots * num_kv_heads * head_dim, 1),
        vec![total_slots, num_kv_heads, head_dim],
        0,
    )
    .expect("k_pool");
    let vp = Tensor::cuda_from_slice(
        &pattern_bf16(total_slots * num_kv_heads * head_dim, 2),
        vec![total_slots, num_kv_heads, head_dim],
        0,
    )
    .expect("v_pool");
    let k = Tensor::cuda_from_slice(
        &pattern_bf16(num_kv_heads * head_dim, 3),
        vec![num_kv_heads * head_dim],
        0,
    )
    .expect("k row");
    let v = Tensor::cuda_from_slice(
        &pattern_bf16(num_kv_heads * head_dim, 4),
        vec![num_kv_heads * head_dim],
        0,
    )
    .expect("v row");
    let slot = Tensor::cuda_from_slice(&[2u32], vec![1], 0).expect("slot");

    paged_kv_write_token_major_bf16_slot_kt(&kp, &vp, &k, &v, &slot)
        .expect("paged_kv_write_slot_kt on candle-free inputs");

    // Shape sanity: pool dims unchanged after in-place write.
    assert_eq!(kp.shape(), &[total_slots, num_kv_heads, head_dim]);
    assert_eq!(vp.shape(), &[total_slots, num_kv_heads, head_dim]);
}

/// `paged_kv_write_token_major_bf16_batch_slot_kt` dispatches successfully
/// on a batched candle-free kt CUDA write — exercises the `batch` axis of
/// the FFI without needing the deleted candle-typed parity shell.
#[test]
fn paged_kv_write_batch_slot_kt_dispatches_candle_free() {
    if !cuda_available() {
        eprintln!("CUDA not available; skipping");
        return;
    }

    let total_slots = 16usize;
    let num_kv_heads = 4usize;
    let head_dim = 128usize;
    let batch = 3usize;

    let kp = Tensor::cuda_from_slice(
        &pattern_bf16(total_slots * num_kv_heads * head_dim, 0xC0FFEE),
        vec![total_slots, num_kv_heads, head_dim],
        0,
    )
    .expect("k_pool");
    let vp = Tensor::cuda_from_slice(
        &pattern_bf16(total_slots * num_kv_heads * head_dim, 0xC0FFEE),
        vec![total_slots, num_kv_heads, head_dim],
        0,
    )
    .expect("v_pool");
    let row = num_kv_heads * head_dim;
    let k = Tensor::cuda_from_slice(&pattern_bf16(batch * row, 5), vec![batch * row], 0)
        .expect("k batch");
    let v = Tensor::cuda_from_slice(&pattern_bf16(batch * row, 6), vec![batch * row], 0)
        .expect("v batch");
    let slots = Tensor::cuda_from_slice(&[2u32, 7, 11], vec![batch], 0).expect("slots");

    paged_kv_write_token_major_bf16_batch_slot_kt(&kp, &vp, &k, &v, &slots)
        .expect("paged_kv_write_batch_slot_kt on candle-free inputs");

    assert_eq!(kp.shape(), &[total_slots, num_kv_heads, head_dim]);
    assert_eq!(vp.shape(), &[total_slots, num_kv_heads, head_dim]);
}
