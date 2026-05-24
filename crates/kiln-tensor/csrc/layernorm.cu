// Phase 4 substrate kernel: LayerNorm over the last axis of a tensor.
//
// Computes per-row layer normalization: subtract mean, divide by
// sqrt(var + eps), then apply per-element scale + bias.
//
// # Numerical recipe
//
// For each row of `n_cols` elements:
//   1. mean[row] = (1/n_cols) * Σ_c x[row, c]
//   2. var[row]  = (1/n_cols) * Σ_c (x[row, c] - mean[row])^2
//   3. inv_std[row] = 1 / sqrt(var[row] + eps)
//   4. out[row, c] = (x[row, c] - mean[row]) * inv_std[row] * weight[c] + bias[c]
//
// All accumulation happens in F32 regardless of input dtype (kt
// "F32 accumulation always" convention). Output is cast back to the
// input dtype. Weight + bias are rank-1 `[D]` tensors broadcast over
// rows; they share `x`'s dtype.
//
// Variance is computed in a single pass via the "sum + sum_of_squares"
// trick (E[X^2] - E[X]^2), which is numerically acceptable for the
// row sizes seen in practice (≤16384). For very large rows or rows
// with large mean magnitudes, Welford's algorithm would be more
// numerically stable — left as a future optimization.
//
// # Determinism
//
// The reduction tree is fixed (warp-shuffle + shared-memory across
// warps), so for a given input the output is bit-identical across
// runs. Mirrors `softmax.cu`, `reduce_last_axis.cu`, and `rmsnorm.cu`.
//
// # Launch shape
//
// One block per row. Each block uses `n_cols` threads up to a max of
// 1024; rows with more cols use a strided per-thread accumulation.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int MAX_THREADS = 1024;

template <typename T>
__device__ inline float to_f32(T v);
template <>
__device__ inline float to_f32<float>(float v) { return v; }
template <>
__device__ inline float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <>
__device__ inline float to_f32<__half>(__half v) { return __half2float(v); }

template <typename T>
__device__ inline T from_f32(float v);
template <>
__device__ inline float from_f32<float>(float v) { return v; }
template <>
__device__ inline __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>
__device__ inline __half from_f32<__half>(float v) { return __float2half(v); }

template <typename T>
__global__ void layernorm_last_axis_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ out,
    int64_t n_cols,
    float eps) {
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    int blk = blockDim.x;

    const T* row_in = x + row * n_cols;
    T* row_out = out + row * n_cols;

    // ----- Pass 1: per-row sum and sum-of-squares (F32 accumulator). -----
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int64_t c = tid; c < n_cols; c += blk) {
        float v = to_f32<T>(row_in[c]);
        local_sum += v;
        local_sum_sq += v * v;
    }

    // Warp-level reductions (sum + sum_sq).
    __shared__ float shared_sum[32];
    __shared__ float shared_sum_sq[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        local_sum_sq += __shfl_xor_sync(0xFFFFFFFF, local_sum_sq, offset);
    }
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) {
        shared_sum[warp_id] = local_sum;
        shared_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    // Cross-warp reduction in warp 0.
    __shared__ float mean;
    __shared__ float inv_std;
    if (warp_id == 0) {
        int warp_count = (blk + 31) / 32;
        float s = lane < warp_count ? shared_sum[lane] : 0.0f;
        float ss = lane < warp_count ? shared_sum_sq[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            s += __shfl_xor_sync(0xFFFFFFFF, s, offset);
            ss += __shfl_xor_sync(0xFFFFFFFF, ss, offset);
        }
        if (lane == 0) {
            float inv_n = 1.0f / static_cast<float>(n_cols);
            float m = s * inv_n;
            // var = E[X^2] - E[X]^2; clamp to >= 0 in case of round-off
            // noise (theoretically nonneg but FP arithmetic can dip slightly).
            float var = ss * inv_n - m * m;
            if (var < 0.0f) var = 0.0f;
            float denom = var + eps;
            mean = m;
            inv_std = (denom > 0.0f) ? rsqrtf(denom) : 0.0f;
        }
    }
    __syncthreads();

    // ----- Pass 2: per-element (x - mean) * inv_std * weight + bias. -----
    for (int64_t c = tid; c < n_cols; c += blk) {
        float v = to_f32<T>(row_in[c]);
        float w = to_f32<T>(weight[c]);
        float b = to_f32<T>(bias[c]);
        float y = (v - mean) * inv_std * w + b;
        row_out[c] = from_f32<T>(y);
    }
}

}  // anonymous namespace

extern "C" int kiln_layernorm_last_axis_async(
    const void* x,
    const void* weight,
    const void* bias,
    void* out,
    int64_t n_rows,
    int64_t n_cols,
    float eps,
    int32_t dtype_tag,  // 0=F32, 1=BF16, 2=F16
    void* stream_raw) {
    if (n_rows == 0 || n_cols == 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

    int threads = MAX_THREADS;
    while (threads > n_cols && threads > 32) {
        threads /= 2;
    }
    if (threads < 32) threads = 32;
    dim3 grid((unsigned int)n_rows);
    dim3 block(threads);

    switch (dtype_tag) {
        case 0:
            layernorm_last_axis_kernel<float><<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(x),
                reinterpret_cast<const float*>(weight),
                reinterpret_cast<const float*>(bias),
                reinterpret_cast<float*>(out),
                n_cols,
                eps);
            break;
        case 1:
            layernorm_last_axis_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(weight),
                reinterpret_cast<const __nv_bfloat16*>(bias),
                reinterpret_cast<__nv_bfloat16*>(out),
                n_cols,
                eps);
            break;
        case 2:
            layernorm_last_axis_kernel<__half><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x),
                reinterpret_cast<const __half*>(weight),
                reinterpret_cast<const __half*>(bias),
                reinterpret_cast<__half*>(out),
                n_cols,
                eps);
            break;
        default:
            return -2;
    }
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}
