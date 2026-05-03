// Kiln flash-attention C-ABI wrapper — no PyTorch dependency.
// Exposes flash-attention forward and backward passes as plain C functions
// operating on raw CUDA device pointers.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return code: 0 = success, non-zero = error.
typedef int kiln_flash_status_t;

// Flash Attention Forward Pass
//
// All pointer arguments must be CUDA device pointers.
// Tensors are in [batch, seqlen, num_heads, head_dim] layout (row-major).
//
// softmax_lse_out: output buffer of shape [batch, num_heads, seqlen_q], float32.
//                  Must be pre-allocated by the caller.
kiln_flash_status_t kiln_flash_attn_fwd(
    // Input tensors (device pointers, bf16)
    const void *q,           // [batch, seqlen_q, num_heads, head_dim]
    const void *k,           // [batch, seqlen_k, num_heads_k, head_dim]
    const void *v,           // [batch, seqlen_k, num_heads_k, head_dim]
    // Output tensor (device pointer, bf16)
    void *out,               // [batch, seqlen_q, num_heads, head_dim]
    // Softmax log-sum-exp output (device pointer, float32)
    void *softmax_lse_out,   // [batch, num_heads, seqlen_q]
    // Dimensions
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    int head_dim,
    // Parameters
    float softmax_scale,
    int is_causal,
    // CUDA stream (pass 0/nullptr for default stream)
    void *stream
);

// Flash Attention Forward Pass — Paged Decode (single-query GQA)
//
// All pointer arguments must be CUDA device pointers.
// Specialized for the decode step: query length = 1, K/V are gathered from a
// paged pool indexed by `block_table`.
//
// Layouts:
//   q       : [batch, 1, num_heads, head_dim] bf16
//   k_pool  : [total_slots, num_heads_k, head_dim] bf16
//   v_pool  : [total_slots, num_heads_k, head_dim] bf16
//   block_table : [batch, max_blocks_per_seq] int32 (host-allocated, device-resident)
//   out     : [batch, 1, num_heads, head_dim] bf16
//   softmax_lse_out : [batch, num_heads, 1] float32
//
// The kernel reads `kBlockN` (= 128 for hdim128) consecutive K/V tokens per chunk
// using a single block_table entry, so `page_block_size` must be >= 128 and
// physical pages within a chunk must be contiguous. Callers using a smaller
// logical page size (e.g. 16) must construct an equivalent block_table that
// indexes 128-token-aligned super-blocks.
//
// max_seqlen_k is the current K/V length (number of cached tokens incl. the
// freshly-written current step).
kiln_flash_status_t kiln_flash_attn_fwd_paged_decode(
    const void *q,
    const void *k_pool,
    const void *v_pool,
    const int  *block_table,
    void *out,
    void *softmax_lse_out,
    int batch_size,
    int num_heads,
    int num_heads_k,
    int head_dim,
    int max_seqlen_k,
    int max_blocks_per_seq,
    int page_block_size,
    float softmax_scale,
    int is_causal,
    void *stream
);

kiln_flash_status_t kiln_flash_attn_fwd_paged_decode_dyn_seqlen(
    const void *q,
    const void *k_pool,
    const void *v_pool,
    const int  *block_table,
    const int  *seqused_k,
    void *out,
    void *softmax_lse_out,
    int batch_size,
    int num_heads,
    int num_heads_k,
    int head_dim,
    int max_seqlen_k,
    int max_blocks_per_seq,
    int page_block_size,
    float softmax_scale,
    int is_causal,
    void *stream
);

kiln_flash_status_t kiln_paged_kv_write_token_major_bf16_slot(
    void *k_pool,
    void *v_pool,
    const void *k,
    const void *v,
    const unsigned int *slot,
    int num_kv_heads,
    int head_dim,
    void *stream
);

// Flash Attention Backward Pass
//
// All pointer arguments must be CUDA device pointers.
// Computes gradients dq, dk, dv given dout (gradient of the output).
//
// softmax_lse: the log-sum-exp from the forward pass (float32).
// softmax_d_out: scratch buffer of shape [batch, num_heads, seqlen_q_rounded], float32.
//                seqlen_q_rounded = round_up(seqlen_q, 128).
// dq_accum: scratch buffer of shape [batch, seqlen_q_rounded, num_heads, head_dim_rounded], float32.
//           head_dim_rounded = round_up(head_dim, 32).
kiln_flash_status_t kiln_flash_attn_bwd(
    // Gradient of output (device pointer, bf16)
    const void *dout,        // [batch, seqlen_q, num_heads, head_dim]
    // Forward pass inputs (device pointers, bf16)
    const void *q,           // [batch, seqlen_q, num_heads, head_dim]
    const void *k,           // [batch, seqlen_k, num_heads_k, head_dim]
    const void *v,           // [batch, seqlen_k, num_heads_k, head_dim]
    // Forward pass output (device pointer, bf16)
    const void *out,         // [batch, seqlen_q, num_heads, head_dim]
    // Softmax LSE from forward pass (device pointer, float32)
    const void *softmax_lse, // [batch, num_heads, seqlen_q]
    // Gradient outputs (device pointers, bf16)
    void *dq,                // [batch, seqlen_q, num_heads, head_dim]
    void *dk,                // [batch, seqlen_k, num_heads_k, head_dim]
    void *dv,                // [batch, seqlen_k, num_heads_k, head_dim]
    // Scratch buffers (device pointers, float32) — must be pre-allocated
    void *softmax_d_out,     // [batch, num_heads, seqlen_q_rounded]
    void *dq_accum,          // [batch, seqlen_q_rounded, num_heads, head_dim_rounded]
    // Dimensions
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    int head_dim,
    // Parameters
    float softmax_scale,
    int is_causal,
    int deterministic,
    // CUDA stream
    void *stream
);

#ifdef __cplusplus
}
#endif
