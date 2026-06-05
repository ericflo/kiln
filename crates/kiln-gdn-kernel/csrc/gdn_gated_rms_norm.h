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

// Fused GDN gated RMSNorm kernel (bf16 activations, f32 weight).
//
// Same algorithm and shape envelope as `kiln_gdn_gated_rms_norm_bf16`, but
// reads the learned RMSNorm scale as F32. This matches the production
// Qwen3.5 GDN weight dtype while still writing BF16 activations.
int32_t kiln_gdn_gated_rms_norm_wf32_bf16(
    const void* x,       // [rows, hidden] bf16
    const void* z,       // [rows, hidden] bf16
    const void* weight,  // [hidden] f32
    void* out,           // [rows, hidden] bf16
    int32_t rows,
    int32_t hidden,
    float eps,
    void* stream_raw     // cudaStream_t (raw)
);

// Backward for the fused GDN gated RMSNorm kernel.
//
// Reads bf16 `grad_out`, `x`, `z`, and `weight`. Writes bf16 `d_x`/`d_z` and
// F32 `d_weight` accumulated across rows:
//
//   out = rms_norm(x, weight, eps) * silu(z)
//
// Return codes match the forward entry point.
int32_t kiln_gdn_gated_rms_norm_bwd_bf16(
    const void* grad_out, // [rows, hidden] bf16
    const void* x,        // [rows, hidden] bf16
    const void* z,        // [rows, hidden] bf16
    const void* weight,   // [hidden] bf16
    void* d_x,            // [rows, hidden] bf16
    void* d_z,            // [rows, hidden] bf16
    void* d_weight,       // [hidden] f32
    int32_t rows,
    int32_t hidden,
    float eps,
    void* stream_raw      // cudaStream_t (raw)
);

// Backward for the F32-weight fused GDN gated RMSNorm kernel.
//
// Reads bf16 `grad_out`, `x`, `z`, and f32 `weight`. Writes bf16 `d_x`/`d_z`
// and F32 `d_weight` accumulated across rows.
int32_t kiln_gdn_gated_rms_norm_bwd_wf32_bf16(
    const void* grad_out, // [rows, hidden] bf16
    const void* x,        // [rows, hidden] bf16
    const void* z,        // [rows, hidden] bf16
    const void* weight,   // [hidden] f32
    void* d_x,            // [rows, hidden] bf16
    void* d_z,            // [rows, hidden] bf16
    void* d_weight,       // [hidden] f32
    int32_t rows,
    int32_t hidden,
    float eps,
    void* stream_raw      // cudaStream_t (raw)
);

// Backward for the GDN q/k L2 norm scale operation:
//
//   out = scale * x / sqrt(sum(x^2) + eps)
//
// Reads bf16 `grad_out` and `x`, writes bf16 `d_x`.
// Return codes match the forward entry point.
int32_t kiln_gdn_l2_norm_scale_bwd_bf16(
    const void* grad_out, // [rows, hidden] bf16
    const void* x,        // [rows, hidden] bf16
    void* d_x,            // [rows, hidden] bf16
    int32_t rows,
    int32_t hidden,
    float scale,
    float eps,
    void* stream_raw      // cudaStream_t (raw)
);

#ifdef __cplusplus
}
#endif

#endif  // KILN_GDN_GATED_RMS_NORM_H
