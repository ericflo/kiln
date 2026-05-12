// Fused LoRA delta/add CUDA helpers for single-token decode.
//
// Computes:
//   hidden[row, r] = dot(x[row], A[r])
//   out[row, j] = base[row, j] + scale * dot(hidden[row], B[j])
//
// Shapes are row-major:
//   base:   [batch, out_dim] bf16
//   x:      [batch, in_dim] bf16
//   A:      [rank, in_dim] bf16
//   B:      [out_dim, rank] bf16
//   hidden: [batch, rank] f32 scratch
//   out:    [batch, out_dim] bf16

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t kiln_lora_decode_hidden_bf16(
    const void *x,
    const void *a,
    float *hidden,
    int32_t batch,
    int32_t in_dim,
    int32_t rank,
    void *stream);

int32_t kiln_lora_decode_add_bf16(
    const void *base,
    const float *hidden,
    const void *b,
    void *out,
    float scale,
    int32_t batch,
    int32_t out_dim,
    int32_t rank,
    void *stream);

int32_t kiln_lora_add_inplace_f32(
    float *base,
    const float *hidden,
    const float *b,
    float scale,
    int32_t rows,
    int32_t out_dim,
    int32_t rank,
    void *stream);

#ifdef __cplusplus
}
#endif
