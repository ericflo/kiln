#include "causal_conv1d_f32.h"

#include <cuda_runtime.h>
#include <climits>

namespace {

constexpr int kThreads = 256;

__global__ void causal_depthwise_conv1d_f32_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ state,
    float *__restrict__ out,
    int64_t elems,
    int rows,
    int channels,
    int kernel
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) return;

    const int row = static_cast<int>(idx / channels);
    const int channel = static_cast<int>(idx - static_cast<int64_t>(row) * channels);
    const int state_rows = kernel - 1;

    float acc = 0.0f;
    for (int j = 0; j < kernel; ++j) {
        const int padded_row = row + j;
        float x = 0.0f;
        if (padded_row < state_rows) {
            x = state[padded_row * channels + channel];
        } else {
            x = input[(padded_row - state_rows) * channels + channel];
        }
        acc += x * weight[channel * kernel + j];
    }
    out[idx] = acc;
}

__global__ void causal_depthwise_conv1d_inplace_f32_kernel(
    float *__restrict__ input_out,
    const float *__restrict__ weight,
    const float *__restrict__ state,
    int rows,
    int channels,
    int kernel
) {
    const int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) return;

    const int state_rows = kernel - 1;
    for (int row = rows - 1; row >= 0; --row) {
        float acc = 0.0f;
        for (int j = 0; j < kernel; ++j) {
            const int padded_row = row + j;
            float x = 0.0f;
            if (padded_row < state_rows) {
                x = state[padded_row * channels + channel];
            } else {
                x = input_out[(padded_row - state_rows) * channels + channel];
            }
            acc += x * weight[channel * kernel + j];
        }
        input_out[row * channels + channel] = acc;
    }
}

__global__ void causal_depthwise_conv1d_bwd_input_f32_kernel(
    const float *__restrict__ grad_out,
    const float *__restrict__ weight,
    float *__restrict__ grad_input,
    int64_t elems,
    int rows,
    int channels,
    int kernel
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) return;

    const int input_row = static_cast<int>(idx / channels);
    const int channel = static_cast<int>(idx - static_cast<int64_t>(input_row) * channels);
    const int state_rows = kernel - 1;

    float acc = 0.0f;
    for (int j = 0; j < kernel; ++j) {
        const int out_row = state_rows + input_row - j;
        if (out_row >= 0 && out_row < rows) {
            acc += grad_out[out_row * channels + channel] * weight[channel * kernel + j];
        }
    }
    grad_input[idx] = acc;
}

__global__ void causal_depthwise_conv1d_bwd_weight_f32_kernel(
    const float *__restrict__ grad_out,
    const float *__restrict__ input,
    const float *__restrict__ state,
    float *__restrict__ grad_weight,
    int64_t elems,
    int rows,
    int channels,
    int kernel
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) return;

    const int channel = static_cast<int>(idx / kernel);
    const int j = static_cast<int>(idx - static_cast<int64_t>(channel) * kernel);
    const int state_rows = kernel - 1;

    float acc = 0.0f;
    for (int row = 0; row < rows; ++row) {
        const int padded_row = row + j;
        float x = 0.0f;
        if (padded_row < state_rows) {
            x = state[padded_row * channels + channel];
        } else {
            x = input[(padded_row - state_rows) * channels + channel];
        }
        acc += grad_out[row * channels + channel] * x;
    }
    grad_weight[idx] = acc;
}

__global__ void causal_depthwise_conv1d_bwd_state_f32_kernel(
    const float *__restrict__ grad_out,
    const float *__restrict__ weight,
    float *__restrict__ grad_state,
    int64_t elems,
    int rows,
    int channels,
    int kernel
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) return;

    const int state_row = static_cast<int>(idx / channels);
    const int channel = static_cast<int>(idx - static_cast<int64_t>(state_row) * channels);

    float acc = 0.0f;
    for (int j = 0; j < kernel; ++j) {
        const int out_row = state_row - j;
        if (out_row >= 0 && out_row < rows) {
            acc += grad_out[out_row * channels + channel] * weight[channel * kernel + j];
        }
    }
    grad_state[idx] = acc;
}

__global__ void silu_inplace_save_sigmoid_f32_kernel(
    float *__restrict__ input_out,
    float *__restrict__ sigmoid_out,
    int64_t elems
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) return;
    const float x = input_out[idx];
    const float sigmoid = 1.0f / (1.0f + expf(-x));
    sigmoid_out[idx] = sigmoid;
    input_out[idx] = x * sigmoid;
}

}  // namespace

extern "C" int32_t kiln_causal_depthwise_conv1d_f32(
    const float *input,
    const float *weight,
    const float *state,
    float *out,
    int32_t rows,
    int32_t channels,
    int32_t kernel,
    void *stream
) {
    if (rows <= 0 || channels <= 0 || kernel <= 1) return -1;
    const int64_t elems = static_cast<int64_t>(rows) * static_cast<int64_t>(channels);
    const int64_t blocks64 = (elems + kThreads - 1) / kThreads;
    if (blocks64 > static_cast<int64_t>(INT_MAX)) return -2;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    causal_depthwise_conv1d_f32_kernel<<<static_cast<int>(blocks64), kThreads, 0, s>>>(
        input,
        weight,
        state,
        out,
        elems,
        static_cast<int>(rows),
        static_cast<int>(channels),
        static_cast<int>(kernel));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

extern "C" int32_t kiln_causal_depthwise_conv1d_inplace_f32(
    float *input_out,
    const float *weight,
    const float *state,
    int32_t rows,
    int32_t channels,
    int32_t kernel,
    void *stream
) {
    if (rows <= 0 || channels <= 0 || kernel <= 1) return -1;
    const int64_t blocks64 = (static_cast<int64_t>(channels) + kThreads - 1) / kThreads;
    if (blocks64 > static_cast<int64_t>(INT_MAX)) return -2;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    causal_depthwise_conv1d_inplace_f32_kernel<<<static_cast<int>(blocks64), kThreads, 0, s>>>(
        input_out,
        weight,
        state,
        static_cast<int>(rows),
        static_cast<int>(channels),
        static_cast<int>(kernel));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

extern "C" int32_t kiln_causal_depthwise_conv1d_bwd_weight_f32(
    const float *grad_out,
    const float *input,
    const float *state,
    float *grad_weight,
    int32_t rows,
    int32_t channels,
    int32_t kernel,
    void *stream
) {
    if (rows <= 0 || channels <= 0 || kernel <= 1) return -1;
    const int64_t elems = static_cast<int64_t>(channels) * static_cast<int64_t>(kernel);
    const int64_t blocks64 = (elems + kThreads - 1) / kThreads;
    if (blocks64 > static_cast<int64_t>(INT_MAX)) return -2;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    causal_depthwise_conv1d_bwd_weight_f32_kernel<<<static_cast<int>(blocks64), kThreads, 0, s>>>(
        grad_out,
        input,
        state,
        grad_weight,
        elems,
        static_cast<int>(rows),
        static_cast<int>(channels),
        static_cast<int>(kernel));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

extern "C" int32_t kiln_silu_inplace_save_sigmoid_f32(
    float *input_out,
    float *sigmoid_out,
    int64_t elems,
    void *stream
) {
    if (elems < 0) return -1;
    if (elems == 0) return 0;
    const int64_t blocks64 = (elems + kThreads - 1) / kThreads;
    if (blocks64 > static_cast<int64_t>(INT_MAX)) return -2;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    silu_inplace_save_sigmoid_f32_kernel<<<static_cast<int>(blocks64), kThreads, 0, s>>>(
        input_out,
        sigmoid_out,
        elems);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

extern "C" int32_t kiln_causal_depthwise_conv1d_bwd_state_f32(
    const float *grad_out,
    const float *weight,
    float *grad_state,
    int32_t rows,
    int32_t channels,
    int32_t kernel,
    void *stream
) {
    if (rows <= 0 || channels <= 0 || kernel <= 1) return -1;
    const int64_t elems = static_cast<int64_t>(kernel - 1) * static_cast<int64_t>(channels);
    const int64_t blocks64 = (elems + kThreads - 1) / kThreads;
    if (blocks64 > static_cast<int64_t>(INT_MAX)) return -2;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    causal_depthwise_conv1d_bwd_state_f32_kernel<<<static_cast<int>(blocks64), kThreads, 0, s>>>(
        grad_out,
        weight,
        grad_state,
        elems,
        static_cast<int>(rows),
        static_cast<int>(channels),
        static_cast<int>(kernel));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

extern "C" int32_t kiln_causal_depthwise_conv1d_bwd_input_f32(
    const float *grad_out,
    const float *weight,
    float *grad_input,
    int32_t rows,
    int32_t channels,
    int32_t kernel,
    void *stream
) {
    if (rows <= 0 || channels <= 0 || kernel <= 1) return -1;
    const int64_t elems = static_cast<int64_t>(rows) * static_cast<int64_t>(channels);
    const int64_t blocks64 = (elems + kThreads - 1) / kThreads;
    if (blocks64 > static_cast<int64_t>(INT_MAX)) return -2;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    causal_depthwise_conv1d_bwd_input_f32_kernel<<<static_cast<int>(blocks64), kThreads, 0, s>>>(
        grad_out,
        weight,
        grad_input,
        elems,
        static_cast<int>(rows),
        static_cast<int>(channels),
        static_cast<int>(kernel));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}
