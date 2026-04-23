// Kiln Gated DeltaNet (GDN) fused full-chunk prefill C-ABI wrapper.
//
// This is the next minimal CUDA step past the split `gdn_chunk_prep` +
// `gdn_chunk_scan` path. It keeps the four GEMMs on cuBLAS (KKT, QKT,
// ks_entry, q_s) but collapses the remaining full-chunk prefill work into one
// supported CUDA entrypoint:
//
//   1. chunk-prep algebra on-chip (big_g / p / decay / masks)
//   2. chunk-local forward substitution + intra-chunk output accumulation
//   3. final recurrent-state update
//
// This entrypoint is intentionally narrow to kiln's prompt-heavy CUDA hot
// path: full chunks only (`chunk_size == 64`), bf16, CUDA, Qwen3.5-4B GDN
// envelope (`dk <= 128`, `dv <= 128`).

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int kiln_gdn_full_chunk_forward_status_t;

kiln_gdn_full_chunk_forward_status_t kiln_gdn_full_chunk_forward(
    const void *g,         // [B*H, C] bf16
    const void *v,         // [B*H, C, dv] bf16
    const void *kkt,       // [B*H, C, C] bf16
    const void *qkt,       // [B*H, C, C] bf16
    const void *ks_entry,  // [B*H, C, dv] bf16
    const void *q_s,       // [B*H, C, dv] bf16
    const void *beta,      // [B*H, C] bf16
    const void *k_t,       // [B*H, dk, C] bf16
    void *state,           // [B*H, dk, dv] bf16, mutated in place
    void *out_chunk,       // [B*H, C, dv] bf16
    int batch_heads,
    int chunk_size,
    int dk,
    int dv,
    void *stream
);

#ifdef __cplusplus
}
#endif
