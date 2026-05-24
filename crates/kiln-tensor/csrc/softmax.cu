// Phase 4 substrate kernel: softmax over the last axis of a tensor.
//
// Replaces the candle path `Tensor::softmax(dim=-1)` for the kt
// CUDA backend. Operates in-place over the last axis; per-row
// exp + normalize.
//
// # Numerical recipe
//
// For each row of `n_cols` elements:
//   1. max = max_over_cols(x)
//   2. exp_x[c] = exp(x[c] - max)
//   3. sum = sum(exp_x)
//   4. out[c] = exp_x[c] / sum
//
// All arithmetic in F32 regardless of input dtype (matches the kt
// "F32 accumulation always" convention). Output is cast back to the
// input dtype.
//
// # Determinism
//
// The reduction order is fixed (warp-level + block-level), so for a
// given input the output is bit-identical across runs.
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
__global__ void softmax_last_axis_kernel(
    const T* __restrict__ x,
    T* __restrict__ out,
    int64_t n_cols) {
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    int blk = blockDim.x;

    const T* row_in = x + row * n_cols;
    T* row_out = out + row * n_cols;

    // ----- Pass 1: row max -----
    float local_max = -INFINITY;
    for (int64_t c = tid; c < n_cols; c += blk) {
        float v = to_f32<T>(row_in[c]);
        if (v > local_max) local_max = v;
    }

    __shared__ float shared_max[32];
    // Warp-level reduction.
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = lane < (blk + 31) / 32 ? shared_max[lane] : -INFINITY;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_xor_sync(0xFFFFFFFF, v, offset);
            if (other > v) v = other;
        }
        if (lane == 0) shared_max[0] = v;
    }
    __syncthreads();
    float row_max = shared_max[0];

    // ----- Pass 2: exp(x - max), accumulate sum -----
    float local_sum = 0.0f;
    for (int64_t c = tid; c < n_cols; c += blk) {
        float v = to_f32<T>(row_in[c]);
        float e = expf(v - row_max);
        // Stash into output (still F32 logically, but we store cast
        // back to dtype later — write to dtype now is wrong because
        // we still need the F32 value for division).
        // Cache the F32 exp in the output buffer temporarily — only
        // for F32 dtype. For BF16/F16 we need to recompute exp.
        local_sum += e;
        // For F32 IO, write the partial exp.
        if constexpr (sizeof(T) == 4) {
            row_out[c] = from_f32<T>(e);
        }
    }

    __shared__ float shared_sum[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float v = lane < (blk + 31) / 32 ? shared_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        }
        if (lane == 0) shared_sum[0] = v;
    }
    __syncthreads();
    float row_sum = shared_sum[0];
    float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;

    // ----- Pass 3: divide and store -----
    for (int64_t c = tid; c < n_cols; c += blk) {
        if constexpr (sizeof(T) == 4) {
            // F32 path: read the partial exp we stashed.
            float e = to_f32<T>(row_out[c]);
            row_out[c] = from_f32<T>(e * inv_sum);
        } else {
            // BF16/F16: recompute exp (cheap on tensor cores; cheaper
            // than allocating an F32 scratch buffer).
            float v = to_f32<T>(row_in[c]);
            float e = expf(v - row_max);
            row_out[c] = from_f32<T>(e * inv_sum);
        }
    }
}

}  // anonymous namespace

extern "C" int kiln_softmax_last_axis_async(
    const void* x,
    void* out,
    int64_t n_rows,
    int64_t n_cols,
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
            softmax_last_axis_kernel<float><<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(x),
                reinterpret_cast<float*>(out),
                n_cols);
            break;
        case 1:
            softmax_last_axis_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<__nv_bfloat16*>(out),
                n_cols);
            break;
        case 2:
            softmax_last_axis_kernel<__half><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x),
                reinterpret_cast<__half*>(out),
                n_cols);
            break;
        default:
            return -2;
    }
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}
