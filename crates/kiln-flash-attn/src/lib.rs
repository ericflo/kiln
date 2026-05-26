//! Vendored flash-attention-2 CUDA kernels with forward AND backward pass.
//!
//! This crate provides forward, backward, paged-decode, dyn-seqlen paged-decode,
//! and paged-KV-write entry points that operate on [`kiln_tensor::Tensor`]
//! operands, backed by vendored flash-attention CUDA kernels compiled via a
//! thin C-ABI wrapper (no PyTorch, no candle dependency on the public surface).
//!
//! Only bf16, head_dim=128,256, causal=true instantiations are compiled to
//! minimize build time.
//!
//! # API
//!
//! Phase 7 (#1082) — the crate now exposes only the kt-typed surface
//! (`kt_api::*_kt`). The previous candle-typed `flash_attn_*` /
//! `paged_kv_write_*` parallel API had zero production callers after the
//! `kiln-model` CUDA paths migrated to the kt wrappers, and have been removed
//! alongside their in-lib parity scaffolds. The kt smoke tests at
//! `tests/kt_v2_smoke.rs` exercise the kt API on real CUDA hardware via the
//! candle-free [`kiln_tensor::Tensor::cuda_from_slice`] substrate helper.

/// kiln-tensor-typed kt-API surface. All callers route through this
/// module; the crate no longer exposes a candle-typed parallel API.
mod kt_api;
pub use kt_api::{
    flash_attn_bwd_kt, flash_attn_fwd_kt, flash_attn_paged_decode_dyn_seqlen_kt,
    flash_attn_paged_decode_dyn_seqlen_kt_with_graph_outputs, flash_attn_paged_decode_kt,
    paged_kv_write_token_major_bf16_batch_slot_kt, paged_kv_write_token_major_bf16_kt,
    paged_kv_write_token_major_bf16_slot_kt, FlashAttnError,
};

// FFI declarations matching flash_api_c.h. Re-exported as `pub(crate)` so the
// kt-typed shells in `kt_api.rs` can dispatch them.
unsafe extern "C" {
    pub(crate) fn kiln_flash_attn_fwd(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        softmax_lse_out: *mut core::ffi::c_void,
        batch_size: i32,
        seqlen_q: i32,
        seqlen_k: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_flash_attn_fwd_paged_decode(
        q: *const core::ffi::c_void,
        k_pool: *const core::ffi::c_void,
        v_pool: *const core::ffi::c_void,
        block_table: *const i32,
        out: *mut core::ffi::c_void,
        softmax_lse_out: *mut core::ffi::c_void,
        batch_size: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        max_seqlen_k: i32,
        max_blocks_per_seq: i32,
        page_block_size: i32,
        softmax_scale: f32,
        is_causal: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_flash_attn_fwd_paged_decode_dyn_seqlen(
        q: *const core::ffi::c_void,
        k_pool: *const core::ffi::c_void,
        v_pool: *const core::ffi::c_void,
        block_table: *const i32,
        seqused_k: *const i32,
        out: *mut core::ffi::c_void,
        softmax_lse_out: *mut core::ffi::c_void,
        batch_size: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        max_seqlen_k: i32,
        max_blocks_per_seq: i32,
        page_block_size: i32,
        softmax_scale: f32,
        is_causal: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_paged_kv_write_token_major_bf16_slot(
        k_pool: *mut core::ffi::c_void,
        v_pool: *mut core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        slot: *const u32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_paged_kv_write_token_major_bf16(
        k_pool: *mut core::ffi::c_void,
        v_pool: *mut core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        slot: u32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_paged_kv_write_token_major_bf16_batch_slot(
        k_pool: *mut core::ffi::c_void,
        v_pool: *mut core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        slots: *const u32,
        batch: i32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_flash_attn_bwd(
        dout: *const core::ffi::c_void,
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        out: *const core::ffi::c_void,
        softmax_lse: *const core::ffi::c_void,
        dq: *mut core::ffi::c_void,
        dk: *mut core::ffi::c_void,
        dv: *mut core::ffi::c_void,
        softmax_d_out: *mut core::ffi::c_void,
        dq_accum: *mut core::ffi::c_void,
        batch_size: i32,
        seqlen_q: i32,
        seqlen_k: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        deterministic: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

pub(crate) fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}
