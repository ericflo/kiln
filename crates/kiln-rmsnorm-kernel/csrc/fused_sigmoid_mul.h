#ifndef KILN_FUSED_SIGMOID_MUL_H
#define KILN_FUSED_SIGMOID_MUL_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Fused bf16 `x * sigmoid(gate)`.
int32_t kiln_fused_sigmoid_mul_bf16(
    const void* x,
    const void* gate,
    void* out,
    int64_t elems,
    void* stream
);

#ifdef __cplusplus
}
#endif

#endif  // KILN_FUSED_SIGMOID_MUL_H
