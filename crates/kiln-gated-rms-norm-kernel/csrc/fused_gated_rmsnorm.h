// Kiln fused gated RMSNorm C-ABI wrapper.
//
// Replaces the 10-op candle sequence used by `kiln-model::forward::gated_rms_norm`
// (called from `gated_deltanet_forward` under NVTX range `kiln/gdn/gated_norm`):
//
//   to_dtype(x→F32) → to_dtype(z→F32) → to_dtype(w→F32) →
//   sqr → mean_keepdim → + eps → sqrt → recip →
//   broadcast_mul(x, rms_inv) → broadcast_mul(*, w) →
//   silu(z) = z * sigmoid(z) →
//   mul(normed, gate)
//
// with a single CUDA kernel that performs the full Qwen3.5 GDN gated-norm
// epilogue:
//
//   rms_inv = rsqrt(mean(x^2) + eps)            (F32 reduction)
//   out     = bf16(w * x * rms_inv * silu(z))   (F32 epilogue, bf16 store)
//
// Per-row F32 accumulators, per-block reduction via shared memory, bf16 in/out.
// This matches FLA's `fused_norm_gate` Triton kernel algorithmically, reimplemented
// in raw CUDA C so kiln does not add a Triton runtime dependency.
//
// Note on the RMSNorm weight convention: unlike the transformer-block RMSNorm
// which uses Qwen3.5's `(1 + w)` form (weights centred on 0), the GDN gated
// RMSNorm uses the standard `w * x * rms_inv` form (weights centred on 1).
// This matches the HuggingFace / FLA reference implementations.
//
// Pointer layout (all bf16, all CUDA, all contiguous):
//   x      : [rows, hidden]
//   z      : [rows, hidden]
//   weight : [hidden]
//   out    : [rows, hidden]
//
// Scope — forward-only, decode-first, minimal:
//   - bf16 activations; F32 reduction + F32 epilogue accumulator.
//   - `hidden` <= 8192. Qwen3.5-4B uses 128 here (linear_value_head_dim).
//   - Any number of rows; blocks tile over rows, threads tile within a row.
//   - `eps` passed as F32 — kiln uses 1e-6 for Qwen3.5.
//
// Not in scope:
//   - Backward pass (decode/inference is forward-only).
//   - Non-bf16 dtypes, non-contiguous layouts.
//   - Separate x/z/out dtypes (all must be bf16).

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int kiln_gated_rmsnorm_status_t;

// Fused GDN gated RMSNorm: out = w * x * rsqrt(mean(x^2) + eps) * silu(z).
// All pointers are bf16, contiguous, CUDA device memory.
//
// Returns 0 on success, non-zero on launch/config failure (2 = out-of-envelope).
kiln_gated_rmsnorm_status_t kiln_fused_gated_rmsnorm(
    const void *x,
    const void *z,
    const void *weight,
    void *out,
    int rows,
    int hidden,
    float eps,
    void *stream
);

#ifdef __cplusplus
}
#endif
