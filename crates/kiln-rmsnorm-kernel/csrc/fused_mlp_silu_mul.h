// Fused MLP SiLU(gate) * up CUDA kernel for Qwen3.5 SwiGLU.
//
// Collapses the decode/prefill elementwise middle of the MLP from two Candle
// op chains:
//
//   gate_silu = gate / (1 + exp(-gate))
//   hidden    = gate_silu * up
//
// into one bf16 CUDA launch.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t kiln_fused_mlp_silu_mul_bf16(
    const void *gate,
    const void *up,
    void *out,
    int64_t elems,
    void *stream);

#ifdef __cplusplus
}
#endif
