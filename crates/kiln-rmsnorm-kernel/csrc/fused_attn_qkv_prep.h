// Fused Qwen3.5 decode Q/K prep for full attention.
//
// Input tensors are post-projection bf16 rows for a single decode token:
//   q_raw : [batch, 1, q_heads * head_dim * (has_gate ? 2 : 1)]
//   k_raw : [batch, 1, k_heads * head_dim]
//
// The kernel emits:
//   q_out   : [batch, 1, q_heads, head_dim], RMSNorm + RoPE applied
//   k_out   : [batch, 1, k_heads, head_dim], RMSNorm + RoPE applied
//   gate_out: [batch, 1, q_heads * head_dim], raw gate half copied when present
//
// The Q/K normalization matches kiln's Qwen3.5 RMSNorm convention:
//   norm = bf16((1 + weight[d]) * x[d] * rsqrt(mean(x^2) + eps))
// RoPE is then applied to that bf16-rounded norm value, matching the existing
// RMSNorm-then-fused-RoPE dispatch sequence.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t kiln_attn_decode_qkv_split_qk_norm_rope_bf16(
    const void *q_raw,
    const void *k_raw,
    const void *q_weight,
    const void *k_weight,
    const float *cos,
    const float *sin,
    void *q_out,
    void *k_out,
    void *gate_out,
    int32_t batch,
    int32_t q_heads,
    int32_t k_heads,
    int32_t head_dim,
    int32_t rotary_dim,
    int32_t has_gate,
    float eps,
    void *stream);

#ifdef __cplusplus
}
#endif
