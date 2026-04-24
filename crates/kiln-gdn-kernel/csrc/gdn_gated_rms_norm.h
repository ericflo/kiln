#ifndef KILN_GDN_GATED_RMS_NORM_H
#define KILN_GDN_GATED_RMS_NORM_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Fused GDN gated RMSNorm kernel (bf16 activations, bf16 weight).
//
// Reads `x`, `z` of shape [rows, hidden] and `weight` of shape [hidden],
// all bf16. Writes bf16:
//
//   out = rms_norm(x, weight, eps) * silu(z)
//
// Intermediates are F32 inside the kernel, matching kiln-model's portable
// fallback before the caller casts the result back to the model dtype.
//
// Return codes:
//   0 — success
//   1 — CUDA launch error (see cudaGetLastError)
//   2 — envelope violation (hidden != 128)
int32_t kiln_gdn_gated_rms_norm_bf16(
    const void* x,       // [rows, hidden] bf16
    const void* z,       // [rows, hidden] bf16
    const void* weight,  // [hidden] bf16
    void* out,           // [rows, hidden] bf16
    int32_t rows,
    int32_t hidden,
    float eps,
    void* stream_raw     // cudaStream_t (raw)
);

#ifdef __cplusplus
}
#endif

#endif  // KILN_GDN_GATED_RMS_NORM_H
