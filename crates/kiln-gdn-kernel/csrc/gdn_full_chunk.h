// Kiln Gated DeltaNet (GDN) fused full-chunk prefill C-ABI wrapper.
//
// Narrow CUDA-only path for kiln's canonical prefill envelope:
//   - bf16
//   - full chunks only
//   - chunk_size = 64
//   - dk = 128
//   - dv = 128
//
// This kernel consumes the existing cuBLAS-side chunk matmul products and
// completes the rest of the chunk-local prefill work in one CUDA launch:
//   1. chunk-prep (cumsum / decay / v_prime / q_s scaling)
//   2. forward substitution for W
//   3. intra-chunk output accumulation
//   4. state update

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int kiln_gdn_full_chunk_status_t;

kiln_gdn_full_chunk_status_t kiln_gdn_full_chunk_prefill(
    const void *g,         // [B*H, 64] bf16
    const void *v,         // [B*H, 64, 128] bf16
    const void *kkt,       // [B*H, 64, 64] bf16
    const void *qkt,       // [B*H, 64, 64] bf16
    const void *ks_entry,  // [B*H, 64, 128] bf16
    const void *q_s,       // [B*H, 64, 128] bf16
    const void *beta,      // [B*H, 64] bf16
    const void *k_t,       // [B*H, 128, 64] bf16
    void *state,           // [B*H, 128, 128] bf16, updated in place
    void *out_chunk,       // [B*H, 64, 128] bf16
    int batch_heads,
    int chunk_size,
    int dk,
    int dv,
    void *stream
);

#ifdef __cplusplus
}
#endif
