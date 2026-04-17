// Kiln Gated DeltaNet (GDN) C-ABI wrapper.
//
// Exposes a fused forward-substitution kernel that replaces the per-token
// inner loop in the chunkwise analytical recurrence used by GDN linear
// attention.
//
// Algebraic recurrence inside one chunk of length C (kiln currently uses
// C=64, dv=128, bf16 activations):
//
//   W[t, :] = beta[t] * ( V_prime[t, :] - sum_{i<t} A_strict[t, i] * W[i, :] )
//
// where A_strict is the strictly lower-triangular decay-weighted KKT matrix
// computed in the surrounding Rust code, V_prime = V - exp(G[t]) * (K @ S)
// is the chunk-local "deltanet" residual, and beta is the per-token forget
// scalar.
//
// All pointers are CUDA device pointers in row-major layout. Inputs and
// the W output are bf16; B*H is treated as a single batched leading axis.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int kiln_gdn_status_t;

// Fused forward-substitution kernel.
//
// All device pointers must point to contiguous memory.
//
//   a_strict : bf16, shape [batch_heads, chunk_size, chunk_size]
//   v_prime  : bf16, shape [batch_heads, chunk_size, dv]
//   beta     : bf16, shape [batch_heads, chunk_size]
//   w_out    : bf16, shape [batch_heads, chunk_size, dv]  (overwritten)
//
// chunk_size must be a multiple of 8 and <= 128 in the current
// implementation. dv must equal blockDim.x at launch (the wrapper picks
// blockDim.x = dv if dv <= 1024).
kiln_gdn_status_t kiln_gdn_forward_substitution(
    const void *a_strict,
    const void *v_prime,
    const void *beta,
    void *w_out,
    int batch_heads,
    int chunk_size,
    int dv,
    void *stream
);

#ifdef __cplusplus
}
#endif
