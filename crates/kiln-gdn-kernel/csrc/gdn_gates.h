#ifndef KILN_GDN_GATES_H
#define KILN_GDN_GATES_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Fused GDN gates kernel (bf16 activations, bf16 weights).
//
// Reads `a`, `b` of shape [rows, nv] (rows = B * T flattened) and the
// per-head `A_log`, `dt_bias` of shape [nv], all bf16. Writes:
//
//   beta_out = sigmoid(b)                                [rows, nv] bf16
//   g_out    = -exp(A_log) * softplus(a + dt_bias)       [rows, nv] bf16
//
// Intermediates are in F32 inside the kernel (softplus / exp / sigmoid
// are done in F32 for stability, matching the candle F32 reference path
// in kiln-model::forward::gated_deltanet_forward Step 6).
//
// Return codes:
//   0 — success
//   1 — CUDA launch error (see cudaGetLastError)
//   2 — envelope violation (nv > 256)
int32_t kiln_gdn_gates_bf16(
    const void* a,        // [rows, nv] bf16
    const void* b,        // [rows, nv] bf16
    const void* A_log,    // [nv] bf16
    const void* dt_bias,  // [nv] bf16
    void* beta_out,       // [rows, nv] bf16
    void* g_out,          // [rows, nv] bf16
    int32_t rows,
    int32_t nv,
    void* stream_raw      // cudaStream_t (raw)
);

// Same fused gates kernel for the production decode envelope where the
// per-head gate parameters are loaded as F32 tensors.
int32_t kiln_gdn_gates_bf16_f32_params(
    const void* a,        // [rows, nv] bf16
    const void* b,        // [rows, nv] bf16
    const void* A_log,    // [nv] f32
    const void* dt_bias,  // [nv] f32
    void* beta_out,       // [rows, nv] bf16
    void* g_out,          // [rows, nv] bf16
    int32_t rows,
    int32_t nv,
    void* stream_raw      // cudaStream_t (raw)
);

// Same fused gates kernel for the production decode envelope where A_log is
// F32 and dt_bias remains BF16.
int32_t kiln_gdn_gates_bf16_f32_bf16_params(
    const void* a,        // [rows, nv] bf16
    const void* b,        // [rows, nv] bf16
    const void* A_log,    // [nv] f32
    const void* dt_bias,  // [nv] bf16
    void* beta_out,       // [rows, nv] bf16
    void* g_out,          // [rows, nv] bf16
    int32_t rows,
    int32_t nv,
    void* stream_raw      // cudaStream_t (raw)
);

int32_t kiln_gdn_gate_beta_bf16(
    const void* b,
    void* beta_out,
    int32_t rows,
    int32_t nv,
    void* stream_raw
);

int32_t kiln_gdn_gate_g_bf16(
    const void* a,
    const void* A_log,
    const void* dt_bias,
    void* g_out,
    int32_t rows,
    int32_t nv,
    void* stream_raw
);

#ifdef __cplusplus
}
#endif

#endif  // KILN_GDN_GATES_H
