// Kiln fused L2-norm(Q) + scale(Q) + L2-norm(K) C-ABI wrapper.
//
// Replaces the candle op chain used by `kiln-model::forward::l2_normalize`
// at the GDN `kiln/gdn/qk_norm` step (forward.rs ~line 1402-1411):
//
//   q = to_dtype(F32) → sqr → sum_keepdim → +1e-6 → sqrt → broadcast_div
//   q = q * (1/sqrt(dk)) → to_dtype(bf16)
//   k = to_dtype(F32) → sqr → sum_keepdim → +1e-6 → sqrt → broadcast_div
//   k = k.to_dtype(bf16)
//
// That is ~11 CUDA kernel launches on tiny per-row tensors during decode
// (Qwen3.5-4B GDN decode shape: rows = batch * seq_len * Nv = 1*1*16,
// hidden = dk = 128). Per PROFILING.md (post-PR #166), `:kiln/gdn/qk_norm`
// is 14.9% of decode wallclock — the largest *unfused* GDN region.
//
// This kernel collapses the entire chain into one launch:
//
//   inv_q   = q_scale * rsqrt(sum(q^2) + eps)
//   inv_k   =          rsqrt(sum(k^2) + eps)
//   q_out   = bf16(q * inv_q)
//   k_out   = bf16(k * inv_k)
//
// One block per row processes both Q and K (same shape, contiguous, bf16).
// Two warp-shuffle reductions per block — one for Q sum-of-squares, one for
// K — share the same shared-memory scratch and broadcast `inv_q` / `inv_k`
// to all threads before the bf16 epilogue.
//
// Pointer layout (all bf16, all CUDA, all contiguous):
//   q_in, k_in, q_out, k_out : [rows, hidden]
//
// Scope — forward-only, decode-first, minimal:
//   - bf16 activations; F32 reductions and accumulators.
//   - `hidden` <= 8192 (Qwen3.5-4B uses 128 here).
//   - Any number of rows; blocks tile over rows, threads tile within a row.
//   - `eps` passed as F32 — kiln uses 1e-6.
//   - `q_scale` premultiplied into `inv_q` (kiln uses 1/sqrt(dk)).
//
// Not in scope:
//   - Backward pass (decode/inference is forward-only).
//   - Non-bf16 dtypes, non-contiguous layouts.
//   - Mismatched Q/K shape (caller validates).

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int kiln_l2_qk_norm_status_t;

// Fused L2-norm(Q) + scale(Q) + L2-norm(K) — single CUDA launch, bf16 in/out.
//
//   q_out[r, j] = bf16( q_in[r, j] * q_scale * rsqrt(sum_j(q_in[r,:]^2) + eps) )
//   k_out[r, j] = bf16( k_in[r, j]           * rsqrt(sum_j(k_in[r,:]^2) + eps) )
//
// All pointers are bf16, contiguous, CUDA device memory.
//
// Returns 0 on success, 2 if `hidden` exceeds the kernel envelope (caller
// should fall back), 1 on a CUDA launch error.
kiln_l2_qk_norm_status_t kiln_fused_l2_qk_norm(
    const void *q_in,
    const void *k_in,
    void *q_out,
    void *k_out,
    int rows,
    int hidden,
    float q_scale,
    float eps,
    void *stream
);

// Fused GQA head-expand + L2-norm(Q) + scale(Q) + L2-norm(K).
//
// Input layout:  q_in, k_in   [batch, seq, nk, hidden] bf16 contiguous.
// Output layout: q_out, k_out [batch, seq, nv, hidden] bf16 contiguous,
// where `nv = nk * ratio` and each normalized input head is repeated `ratio`
// times in the output head axis.
//
// Scope is intentionally narrow for Qwen3.5 GDN: bf16, forward-only,
// hidden == 128, nv a positive multiple of nk.
kiln_l2_qk_norm_status_t kiln_fused_l2_qk_norm_gqa(
    const void *q_in,
    const void *k_in,
    void *q_out,
    void *k_out,
    int rows,
    int nk,
    int ratio,
    int hidden,
    float q_scale,
    float eps,
    void *stream
);

#ifdef __cplusplus
}
#endif
