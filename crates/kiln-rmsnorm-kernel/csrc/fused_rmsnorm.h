// Kiln fused RMSNorm C-ABI wrapper.
//
// Replaces the ~11-kernel candle op sequence used by `kiln-model::forward::rms_norm`:
//
//   to_dtype(F32) → sqr → mean_keepdim → + eps → sqrt → recip →
//   broadcast_mul → to_dtype(F32 weight) → ones_like + w →
//   broadcast_mul → to_dtype(bf16)
//
// with a single CUDA kernel that performs the full Qwen3.5-style RMSNorm:
//
//   rms_inv = rsqrt(mean(x^2) + eps)
//   out     = bf16((1 + w) * x * rms_inv)
//
// Per-row F32 accumulators, per-block reduction via shared memory, bf16 in/out.
//
// Pointer layout (all bf16, all CUDA, all contiguous):
//   x      : [rows, hidden]
//   weight : [hidden]
//   out    : [rows, hidden]
//
// Scope — forward-only, decode-first, minimal:
//   - bf16 activations; F32 reduction + F32 epilogue accumulator.
//   - `hidden` <= 8192. Qwen3.5-4B uses 2560 (hidden_size).
//   - Any number of rows; blocks tile over rows, threads tile within a row.
//   - `eps` passed as F32 — kiln uses 1e-6 for Qwen3.5.
//   - Qwen3.5-style `(1 + w) * x * rms_inv`; kiln stores weight centered on 0.
//
// Not in scope:
//   - Backward pass (decode/inference is forward-only).
//   - Fused GEMM prologue (single fused norm is the minimum-viable ship; the
//     ~11 launches it collapses already dominate the norm/GEMM boundary cost).
//   - Non-bf16 dtypes, non-contiguous layouts.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int kiln_rmsnorm_status_t;

// Fused Qwen3.5-style RMSNorm: out = (1 + weight) * x * rsqrt(mean(x^2) + eps).
// All pointers are bf16, contiguous, CUDA device memory.
//
// Returns 0 on success, non-zero on launch/config failure.
kiln_rmsnorm_status_t kiln_fused_rmsnorm(
    const void *x,
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
