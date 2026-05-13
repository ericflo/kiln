// Fused RoPE(Q,K) CUDA kernel for Qwen3.5 decode/paged attention.
//
// Applies contiguous-half rotary embedding to bf16 Q/K tensors using prebuilt
// f32 cos/sin tables. The first rotary_dim dimensions are interpreted as
// [x1, x2] halves:
//
//   out1 = x1 * cos - x2 * sin
//   out2 = x1 * sin + x2 * cos
//
// Remaining head dimensions are copied through.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t kiln_fused_rotary_qk(
    const void *q,
    const void *k,
    const float *cos,
    const float *sin,
    void *q_out,
    void *k_out,
    int32_t batch,
    int32_t seq_len,
    int32_t q_heads,
    int32_t k_heads,
    int32_t head_dim,
    int32_t rotary_dim,
    void *stream);

int32_t kiln_fused_rotary_one(
    const void *x,
    const float *cos,
    const float *sin,
    void *out,
    int32_t batch,
    int32_t seq_len,
    int32_t heads,
    int32_t head_dim,
    int32_t rotary_dim,
    void *stream);

int32_t kiln_fused_rotary_one_bwd(
    const void *grad_y,
    const float *cos,
    const float *sin,
    void *grad_x,
    int32_t batch,
    int32_t seq_len,
    int32_t heads,
    int32_t head_dim,
    int32_t rotary_dim,
    void *stream);

#ifdef __cplusplus
}
#endif
