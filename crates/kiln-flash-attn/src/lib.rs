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
mod score_policy;
pub use score_policy::{
    DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB, MAX_FULL_ATTENTION_SCORE_BUDGET_MIB,
    MIN_FULL_ATTENTION_SCORE_BUDGET_MIB, install_full_attention_score_budget_mib,
    validate_full_attention_score_budget_mib, with_test_score_geometry,
};

/// ROCm SDPA. The common bf16/head_dim=128 training shape uses a native exact
/// HIP forward/backward kernel; less common shapes fall back to the fully
/// on-device composite built from parity-tested `kiln_tensor` ROCm primitives.
/// CUDA operands are untouched (this module is cfg-gated).
#[cfg(feature = "rocm")]
mod rocm_sdpa;
#[cfg(feature = "rocm")]
pub use rocm_sdpa::with_rocm_online_fwd_disabled;
#[cfg(not(feature = "rocm"))]
pub fn with_rocm_online_fwd_disabled<T>(f: impl FnOnce() -> T) -> T {
    f()
}
pub use kt_api::{
    FlashAttnError, flash_attn_bwd_collapsed_gqa_kt, flash_attn_bwd_kt,
    flash_attn_fwd_head_major_kt, flash_attn_fwd_kt, flash_attn_fwd_no_lse_kt,
    flash_attn_paged_decode_dyn_seqlen_kt,
    flash_attn_paged_decode_dyn_seqlen_kt_with_graph_outputs, flash_attn_paged_decode_kt,
    paged_kv_write_token_major_bf16_batch_slot_kt, paged_kv_write_token_major_bf16_kt,
    paged_kv_write_token_major_bf16_slot_kt,
};

// FFI declarations matching flash_api_c.h. Re-exported as `pub(crate)` so the
// kt-typed shells in `kt_api.rs` can dispatch them. CUDA-only: these symbols are
// provided by the vendored flash-attention `.cu` kernels compiled by `build.rs`
// when the `cuda` feature is on. Under `--features rocm` no `.cu` is compiled
// (no CUTLASS on ROCm) and the `*_kt` shells dispatch into the on-device
// `rocm_sdpa` composite instead, so these declarations are gated out.
#[cfg(feature = "cuda")]
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

#[cfg(feature = "rocm")]
unsafe extern "C" {
    pub(crate) fn kiln_rocm_flash_attn_fwd_ck_bf16(
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

    pub(crate) fn kiln_rocm_flash_attn_fwd_bf16(
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

    pub(crate) fn kiln_rocm_flash_attn_fwd_abs_tile_bf16(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        softmax_lse_out: *mut core::ffi::c_void,
        batch_size: i32,
        seqlen_q_tile: i32,
        seqlen_k: i32,
        seqlen_q_total: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        q_start: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_attn_fwd_abs_tile_base_into_bf16(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        softmax_lse_out: *mut core::ffi::c_void,
        batch_size: i32,
        seqlen_q_tile: i32,
        seqlen_k: i32,
        seqlen_q_total: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        q_start: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_attn_fwd_stream_update_bf16(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        row_m: *mut core::ffi::c_void,
        row_l: *mut core::ffi::c_void,
        acc: *mut core::ffi::c_void,
        batch_size: i32,
        seqlen_q_tile: i32,
        seqlen_k: i32,
        seqlen_q_total: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        q_start: i32,
        key_start: i32,
        key_len: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_attn_fwd_stream_finalize_bf16(
        row_m: *const core::ffi::c_void,
        row_l: *const core::ffi::c_void,
        acc: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        softmax_lse_out: *mut core::ffi::c_void,
        batch_size: i32,
        seqlen_q_tile: i32,
        num_heads: i32,
        head_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_attn_bwd_bf16(
        dout: *const core::ffi::c_void,
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        out: *const core::ffi::c_void,
        softmax_lse: *const core::ffi::c_void,
        dq: *mut core::ffi::c_void,
        dk: *mut core::ffi::c_void,
        dv: *mut core::ffi::c_void,
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

    pub(crate) fn kiln_rocm_flash_attn_bwd_collapsed_gqa_bf16(
        dout: *const core::ffi::c_void,
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        out: *const core::ffi::c_void,
        softmax_lse: *const core::ffi::c_void,
        dq: *mut core::ffi::c_void,
        dk: *mut core::ffi::c_void,
        dv: *mut core::ffi::c_void,
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

    pub(crate) fn kiln_rocm_flash_rowsum_sub_last_axis_f32(
        a: *const core::ffi::c_void,
        rowsum: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        bh: i32,
        sq: i32,
        sk: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_collapse_gqa_bf16(
        expanded: *const core::ffi::c_void,
        collapsed: *mut core::ffi::c_void,
        batch_size: i32,
        seqlen: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_causal_mask_fill_offset_f32(
        scores: *mut core::ffi::c_void,
        bh: i32,
        sq: i32,
        sk: i32,
        q_start: i32,
        causal_offset: i32,
        fill: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_scale_mask_f32(
        scores: *mut core::ffi::c_void,
        bh: i32,
        sq: i32,
        sk: i32,
        q_start: i32,
        k_start: i32,
        causal_offset: i32,
        scale: f32,
        fill: f32,
        is_causal: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_exp_mask_f32(
        scores: *mut core::ffi::c_void,
        row_max: *const core::ffi::c_void,
        bh: i32,
        sq: i32,
        sk: i32,
        q_start: i32,
        k_start: i32,
        causal_offset: i32,
        is_causal: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_prob_from_lse_f32(
        scores: *mut core::ffi::c_void,
        lse: *const core::ffi::c_void,
        bh: i32,
        sq: i32,
        sk: i32,
        total_q: i32,
        q_start: i32,
        k_start: i32,
        causal_offset: i32,
        is_causal: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_softmax_bwd_f32(
        dp: *mut core::ffi::c_void,
        p: *const core::ffi::c_void,
        d_rows: *const core::ffi::c_void,
        bh: i32,
        sq: i32,
        sk: i32,
        total_q: i32,
        q_start: i32,
        k_start: i32,
        causal_offset: i32,
        scale: f32,
        is_causal: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_accum_axis1_f32(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        bh: i32,
        total_s: i32,
        tile_s: i32,
        head_dim: i32,
        s_start: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_online_update_state_f32(
        row_m: *mut core::ffi::c_void,
        row_l: *mut core::ffi::c_void,
        block_m: *const core::ffi::c_void,
        block_l: *const core::ffi::c_void,
        alpha: *mut core::ffi::c_void,
        beta: *mut core::ffi::c_void,
        rows: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_online_update_acc_f32(
        acc: *mut core::ffi::c_void,
        block_out: *const core::ffi::c_void,
        alpha: *const core::ffi::c_void,
        beta: *const core::ffi::c_void,
        rows: i32,
        head_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_rocm_flash_online_finalize_f32(
        acc: *const core::ffi::c_void,
        row_m: *const core::ffi::c_void,
        row_l: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        lse: *mut core::ffi::c_void,
        rows: i32,
        head_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

#[cfg(feature = "cuda")]
pub(crate) fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}
