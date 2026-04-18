// Kiln Gated DeltaNet (GDN) chunk-prep C-ABI wrapper.
//
// Fuses the elementwise portion of the chunkwise analytical GDN recurrence
// that surrounds the three cuBLAS matmuls (KKT, QKT, ks_entry, q_s). The
// candle-op reference in `kiln-model::forward::gdn_chunkwise_recurrence`
// spends 7+ launches per chunk to build these tensors; this kernel folds
// them into a single block-per-(batch, head) launch.
//
// Inside one chunk of length C (kiln currently uses C=64, dv=128, bf16):
//
//   big_g[t]        = sum_{s<=t} g[s]                                  (F32)
//   p[t]            = exp(big_g[t])
//   decay[t, i]     = exp(big_g[t] - big_g[i])
//   a_strict[t, i]  = kkt[t, i] * decay[t, i]       if i < t  else 0
//   b_mask[t, i]    = qkt[t, i] * decay[t, i]       if i <= t else 0
//   v_prime[t, d]   = v[t, d] - ks_entry[t, d] * p[t]
//   q_s_scaled[t,d] = q_s[t, d] * p[t]
//   decay_last[i]   = exp(big_g[C-1] - big_g[i])
//   p_last          = exp(big_g[C-1])
//
// All pointers are CUDA device pointers in row-major layout. All tensors
// are bf16 except big_g which is computed in F32 on-chip and never
// materialised to global memory. B*H is treated as a single batched
// leading axis.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int kiln_gdn_chunk_prep_status_t;

// Fused chunk-prep kernel.
//
// Inputs  (bf16, CUDA, contiguous):
//   g         : [B*H, C]
//   v         : [B*H, C, dv]
//   kkt       : [B*H, C, C]
//   qkt       : [B*H, C, C]
//   ks_entry  : [B*H, C, dv]
//   q_s       : [B*H, C, dv]
//
// Outputs (bf16, CUDA, contiguous, pre-allocated):
//   a_strict       : [B*H, C, C]
//   b_mask         : [B*H, C, C]
//   v_prime        : [B*H, C, dv]
//   q_s_scaled     : [B*H, C, dv]
//   decay_last_col : [B*H, C]
//   p_last         : [B*H]
//
// Constraints: C <= 128, dv <= 1024.
kiln_gdn_chunk_prep_status_t kiln_gdn_chunk_prep(
    const void *g,
    const void *v,
    const void *kkt,
    const void *qkt,
    const void *ks_entry,
    const void *q_s,
    void *a_strict,
    void *b_mask,
    void *v_prime,
    void *q_s_scaled,
    void *decay_last_col,
    void *p_last,
    int batch_heads,
    int chunk_size,
    int dv,
    void *stream
);

#ifdef __cplusplus
}
#endif
