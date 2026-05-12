// F32 causal depthwise Conv1d helpers for CUDA-native training.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t kiln_causal_depthwise_conv1d_f32(
    const float *input,
    const float *weight,
    const float *state,
    float *out,
    int32_t rows,
    int32_t channels,
    int32_t kernel,
    void *stream);

int32_t kiln_causal_depthwise_conv1d_inplace_f32(
    float *input_out,
    const float *weight,
    const float *state,
    int32_t rows,
    int32_t channels,
    int32_t kernel,
    void *stream);

int32_t kiln_causal_depthwise_conv1d_bwd_input_f32(
    const float *grad_out,
    const float *weight,
    float *grad_input,
    int32_t rows,
    int32_t channels,
    int32_t kernel,
    void *stream);

int32_t kiln_causal_depthwise_conv1d_bwd_weight_f32(
    const float *grad_out,
    const float *input,
    const float *state,
    float *grad_weight,
    int32_t rows,
    int32_t channels,
    int32_t kernel,
    void *stream);

int32_t kiln_causal_depthwise_conv1d_bwd_state_f32(
    const float *grad_out,
    const float *weight,
    float *grad_state,
    int32_t rows,
    int32_t channels,
    int32_t kernel,
    void *stream);

int32_t kiln_silu_inplace_save_sigmoid_f32(
    float *input_out,
    float *sigmoid_out,
    int64_t elems,
    void *stream);

#ifdef __cplusplus
}
#endif
