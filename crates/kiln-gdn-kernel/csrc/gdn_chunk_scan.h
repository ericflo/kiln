// Kiln Gated DeltaNet (GDN) chunk-body C-ABI wrapper.
//
// Completes the hottest remaining prefill chunk work after `gdn_chunk_prep`:
// forward substitution for W[t], intra-chunk output accumulation, and the
// per-row decay_last weighting used by the final state GEMM.
//
// Inputs are the materialized outputs of chunk-prep plus beta:
//
//   W[t, :]         = beta[t] * (V'[t, :] - sum_{i<t} A[t, i] * W[i, :])
//   intra[t, :]     = sum_{i<=t} B[t, i] * W[i, :]
//   out[t, :]       = q_s_scaled[t, :] + intra[t, :]
//   w_weighted[t,:] = W[t, :] * decay_last_col[t]
//
// All pointers are CUDA device pointers in row-major layout. This kernel is
// intentionally narrow to the current kiln/Qwen3.5-4B prefill envelope:
// chunk_size <= 64 and dv <= 128.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int kiln_gdn_chunk_scan_status_t;

kiln_gdn_chunk_scan_status_t kiln_gdn_chunk_scan(
    const void *a_strict,        // [B*H, C, C] bf16
    const void *b_mask,          // [B*H, C, C] bf16
    const void *v_prime,         // [B*H, C, dv] bf16
    const void *q_s_scaled,      // [B*H, C, dv] bf16
    const void *beta,            // [B*H, C] bf16
    const void *decay_last_col,  // [B*H, C] bf16
    void *out_chunk,             // [B*H, C, dv] bf16
    void *w_weighted,            // [B*H, C, dv] bf16
    int batch_heads,
    int chunk_size,
    int dv,
    void *stream
);

#ifdef __cplusplus
}
#endif
