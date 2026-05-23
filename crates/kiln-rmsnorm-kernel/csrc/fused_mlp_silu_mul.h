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

// Packed variant: input is a contiguous `[rows, 2*cols]` BF16 tensor laid
// out as `[gate_row_0 | up_row_0 | gate_row_1 | up_row_1 | ...]`. Reads
// `gate_packed[r*2*cols + c]` and `gate_packed[r*2*cols + cols + c]` for
// each output element `out[r*cols + c]`. Lets the MLP prefill path consume
// the output of a single `[B*T, hidden] @ [hidden, 2*intermediate]` GEMM
// without a `.contiguous()` copy of each half.
int32_t kiln_fused_mlp_silu_mul_packed_bf16(
    const void *gate_up_packed,
    void *out,
    int64_t rows,
    int64_t cols,
    void *stream);

#ifdef __cplusplus
}
#endif
