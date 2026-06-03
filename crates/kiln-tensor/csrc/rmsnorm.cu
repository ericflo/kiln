// Phase 4 substrate kernel: RMSNorm over the last axis of a tensor.
//
// Computes per-row root-mean-square normalization with an optional
// per-element scale (`weight`) applied as a broadcast over the
// trailing axis.
//
// # Numerical recipe
//
// For each row of `n_cols` elements:
//   1. mean_sq[row] = (1/n_cols) * Σ_c x[row, c]^2
//   2. inv_rms[row] = 1 / sqrt(mean_sq[row] + eps)
//   3. out[row, c] = x[row, c] * inv_rms[row] * weight[c]
//
// All accumulation happens in F32 regardless of input dtype (kt
// "F32 accumulation always" convention). Output is cast back to the
// input dtype. Weight is broadcast over rows; it is a rank-1 `[D]`
// tensor and may have the same dtype as `x` (F32 / BF16 / F16).
//
// # Determinism
//
// The reduction tree is fixed (warp-shuffle + shared-memory across
// warps), so for a given input the output is bit-identical across
// runs. Mirrors `softmax.cu` and `reduce_last_axis.cu` patterns.
//
// # Launch shape
//
// One block per row. Each block uses `n_cols` threads up to a max of
// 1024; rows with more cols use a strided per-thread accumulation.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

// Wave-size shim (Phase R.5). The two-level reduction below MUST NOT use a
// hardcoded `tid/32` warp_id + 32-lane `__shfl_xor_sync`: on AMD wave64 that
// makes "warp 0" only half a wavefront and the cross-warp shuffle at offset 16
// over a 64-lane wave touches inactive/self lanes — corrupt sums on RDNA wave64
// (verified on real gfx1151). The fix is the wave-size-AGNOSTIC shared-memory
// block reduction kiln_block_reduce_sum (no cross-lane ops), with the launcher
// choosing a power-of-two blockDim in [64,1024] so every wavefront is full.
// On nvcc this is behavior-identical to the old shared-mem tree (warpSize=32).
#include "kt_gpu_compat.cuh"

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
__global__ void rmsnorm_last_axis_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int64_t n_cols,
    float eps) {
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    int blk = blockDim.x;

    const T* row_in = x + row * n_cols;
    T* row_out = out + row * n_cols;

    // ----- Pass 1: per-row sum of squares (F32 accumulator). -----
    float local_sum = 0.0f;
    for (int64_t c = tid; c < n_cols; c += blk) {
        float v = to_f32<T>(row_in[c]);
        local_sum += v * v;
    }

    // Wave-size-agnostic block reduction via shared memory (no cross-lane ops
    // — correct on AMD wave32/wave64 and NVIDIA). The full sum is returned to
    // every thread, so each computes inv_rms locally (no shared broadcast).
    __shared__ float smem[MAX_THREADS];
    float total_sum = kiln_block_reduce_sum(local_sum, smem);

    // mean_sq = sum / n_cols; rsqrtf for inv_rms; guard degenerate cases.
    float mean_sq = total_sum / static_cast<float>(n_cols);
    float denom = mean_sq + eps;
    float inv_rms = (denom > 0.0f) ? rsqrtf(denom) : 0.0f;

    // ----- Pass 2: per-element scale + multiply by weight. -----
    for (int64_t c = tid; c < n_cols; c += blk) {
        float v = to_f32<T>(row_in[c]);
        float w = to_f32<T>(weight[c]);
        float y = v * inv_rms * w;
        row_out[c] = from_f32<T>(y);
    }
}

}  // anonymous namespace

extern "C" int kiln_rmsnorm_last_axis_async(
    const void* x,
    const void* weight,
    void* out,
    int64_t n_rows,
    int64_t n_cols,
    float eps,
    int32_t dtype_tag,  // 0=F32, 1=BF16, 2=F16
    void* stream_raw) {
    if (n_rows == 0 || n_cols == 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

    // Block size: smallest power of two that covers n_cols, clamped to
    // [64, MAX_THREADS]. Min 64 (not 32) so every wavefront is FULLY populated
    // on AMD wave64 (a half-filled wave breaks shared-mem-tree assumptions and
    // the strided accumulation must cover every lane). Powers of two >= 64 are
    // multiples of both 32 and 64; kiln_block_reduce_sum requires a power of
    // two. Threads past n_cols contribute the reduction identity (0) via the
    // strided accumulation loop.
    int threads = 64;
    while (threads < n_cols && threads < MAX_THREADS) {
        threads *= 2;
    }
    dim3 grid((unsigned int)n_rows);
    dim3 block(threads);

    switch (dtype_tag) {
        case 0:
            rmsnorm_last_axis_kernel<float><<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(x),
                reinterpret_cast<const float*>(weight),
                reinterpret_cast<float*>(out),
                n_cols,
                eps);
            break;
        case 1:
            rmsnorm_last_axis_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(weight),
                reinterpret_cast<__nv_bfloat16*>(out),
                n_cols,
                eps);
            break;
        case 2:
            rmsnorm_last_axis_kernel<__half><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x),
                reinterpret_cast<const __half*>(weight),
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
