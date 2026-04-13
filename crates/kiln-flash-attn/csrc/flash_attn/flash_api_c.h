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
