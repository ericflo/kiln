// Kiln fused RMSNorm backward C-ABI wrapper.
//
// Companion to `kiln_fused_rmsnorm` (forward). Computes the gradients of
// Qwen3.5-style RMSNorm `out = (1 + w) * x * rms_inv` with `rms_inv =
// rsqrt(mean(x^2) + eps)` directly from the saved inputs `x`, `weight`, and
// the upstream gradient `grad_out`. No intermediate values from the forward
// pass are required — `rms_inv` is recomputed inside the kernel.
//
// Math (per row `i`, with hidden axis `H`):
//
//   rms_inv_i = rsqrt(sum_j x[i,j]^2 / H + eps)
//   c_i       = (1/H) * rms_inv_i^2 * sum_j ((1 + w[j]) * x[i,j] * grad_out[i,j])
//   grad_x[i,j] = rms_inv_i * ((1 + w[j]) * grad_out[i,j] - x[i,j] * c_i)
//   grad_w[j]   = sum_i x[i,j] * rms_inv_i * grad_out[i,j]
//
// Pointer layout (all bf16 except `grad_w_partial_f32`, all CUDA, contiguous):
//   x                : [rows, hidden]
//   weight           : [hidden]
//   grad_out         : [rows, hidden]
//   grad_x           : [rows, hidden]
//   grad_w_partial_f32 : [hidden] f32 — caller MUST zero this buffer before
//                        the call. Cross-row reduction uses atomicAdd in f32.
//                        Caller is responsible for the optional final cast
//                        to bf16.
//
// Scope:
//   - bf16 activations + bf16 weights + bf16 grad_out + bf16 grad_x.
//   - Per-row F32 reductions; F32 atomicAdd for the cross-row grad_w sum.
//   - `hidden` <= 8192. Qwen3.5-4B uses 2560.
//   - `eps` passed as F32 — kiln uses 1e-6.
//
// Not in scope:
//   - Non-bf16 inputs.
//   - Non-contiguous strides.
//   - Activation recompute fused with the prior projection's GEMM
//     (dispatch-amortization analysis lives in PROFILING.md; the gain
//     this kernel targets is *saved-tensor reduction* during training,
//     which is on the long-context OOM critical path independent of CUDA
//     graph dispatch overhead).

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int kiln_rmsnorm_bwd_status_t;

// Fused Qwen3.5-style RMSNorm backward.
//
//   grad_x = rms_inv * ((1 + w) * grad_out - x * c)
//   grad_w[j] = sum_i (x[i,j] * rms_inv_i * grad_out[i,j])
//
// `grad_w_partial_f32` MUST be a zero-initialized [hidden] f32 buffer; the
// caller is responsible for casting it back to bf16 if needed.
//
// Returns 0 on success, non-zero on launch/config failure.
kiln_rmsnorm_bwd_status_t kiln_fused_rmsnorm_bwd(
    const void *x,
    const void *weight,
    const void *grad_out,
    void *grad_x,
    float *grad_w_partial_f32,
    int rows,
    int hidden,
    float eps,
    void *stream
);

// Cast an [hidden] f32 buffer to bf16 in place (out-of-place: writes to
// `dst`). Used to finalize `grad_w_partial_f32` after the bwd kernel.
//
// Returns 0 on success, non-zero on launch/config failure.
kiln_rmsnorm_bwd_status_t kiln_f32_to_bf16(
    const float *src,
    void *dst,
    int n,
    void *stream
);

#ifdef __cplusplus
}
#endif
