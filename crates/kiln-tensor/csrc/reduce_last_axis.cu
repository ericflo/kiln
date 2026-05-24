// Phase 4 substrate kernel: per-row sum-of-squares reduction over the
// last axis of a tensor.
//
// Outputs `[..rows]` (one rank less than input). Used by the kt
// L2-norm path as the reduction half of `x / sqrt(sum(x^2) + eps)`.
//
// # Numerical recipe
//
// For each row of `n_cols` elements:
//   sum_sq[row] = sum_c x[row, c]^2
//
// All accumulation happens in F32 regardless of input dtype (kt
// "F32 accumulation always" convention). Output is F32 (`sum_sq` is
// always a higher-precision reduction result; downstream callers
// can cast back to a target dtype if needed).
//
// # Determinism
//
// The reduction tree is fixed (warp-shuffle + shared-memory across
// warps), so for a given input the output is bit-identical across
// runs.
//
// # Launch shape
//
// One block per row. Each block uses `n_cols` threads up to a max of
// 1024; rows with more cols use a strided per-thread accumulation.
// Mirrors the launch shape of `softmax.cu`.

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
__global__ void sum_squared_last_axis_kernel(
    const T* __restrict__ x,
    float* __restrict__ out,
    int64_t n_cols) {
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    int blk = blockDim.x;

    const T* row_in = x + row * n_cols;

    // Per-thread strided accumulation of squares (F32).
    float local_sum = 0.0f;
    for (int64_t c = tid; c < n_cols; c += blk) {
        float v = to_f32<T>(row_in[c]);
        local_sum += v * v;
    }

    // Warp-level reduction (sum).
    __shared__ float shared_sum[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    // Cross-warp reduction in warp 0.
    if (warp_id == 0) {
        float v = lane < (blk + 31) / 32 ? shared_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        }
        if (lane == 0) {
            out[row] = v;
        }
    }
}

}  // anonymous namespace

extern "C" int kiln_sum_squared_last_axis_async(
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
            sum_squared_last_axis_kernel<float><<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(x),
                reinterpret_cast<float*>(out),
                n_cols);
            break;
        case 1:
            sum_squared_last_axis_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<float*>(out),
                n_cols);
            break;
        case 2:
            sum_squared_last_axis_kernel<__half><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x),
                reinterpret_cast<float*>(out),
                n_cols);
            break;
        default:
            return -2;
    }
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}

// ----------------------------------------------------------------------
// L2-norm apply kernel: `out[row, c] = x[row, c] / sqrt(sum_sq[row] + eps)`
//
// Composes with `kiln_sum_squared_last_axis_async` above. Avoids the
// need to broadcast / cast a per-row F32 scalar through the generic
// elementwise pipeline.
//
// Determinism: each (row, c) is independent; no inter-thread state.

template <typename T>
__global__ void l2norm_apply_kernel(
    const T* __restrict__ x,
    const float* __restrict__ sum_sq,
    T* __restrict__ out,
    int64_t n_cols,
    float eps) {
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    int blk = blockDim.x;

    // Compute the per-row inverse-sqrt scalar once per block; broadcast
    // via shared memory so every thread reads the same value.
    __shared__ float inv_norm;
    if (tid == 0) {
        float s = sum_sq[row] + eps;
        inv_norm = (s > 0.0f) ? rsqrtf(s) : 0.0f;
    }
    __syncthreads();

    const T* row_in = x + row * n_cols;
    T* row_out = out + row * n_cols;

    for (int64_t c = tid; c < n_cols; c += blk) {
        float v = to_f32<T>(row_in[c]);
        float scaled = v * inv_norm;
        if constexpr (sizeof(T) == 4) {
            row_out[c] = static_cast<T>(scaled);
        } else if constexpr (sizeof(T) == 2) {
            // BF16 / F16 path. Use the specialization shape that
            // matches softmax.cu's casts.
            if constexpr (__is_same(T, __nv_bfloat16)) {
                row_out[c] = __float2bfloat16(scaled);
            } else {
                row_out[c] = __float2half(scaled);
            }
        }
    }
}

extern "C" int kiln_l2norm_apply_async(
    const void* x,
    const void* sum_sq_f32,
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

    const float* sum_sq = reinterpret_cast<const float*>(sum_sq_f32);

    switch (dtype_tag) {
        case 0:
            l2norm_apply_kernel<float><<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(x),
                sum_sq,
                reinterpret_cast<float*>(out),
                n_cols,
                eps);
            break;
        case 1:
            l2norm_apply_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x),
                sum_sq,
                reinterpret_cast<__nv_bfloat16*>(out),
                n_cols,
                eps);
            break;
        case 2:
            l2norm_apply_kernel<__half><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x),
                sum_sq,
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
