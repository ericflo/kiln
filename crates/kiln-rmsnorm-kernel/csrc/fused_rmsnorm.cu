// Kiln fused RMSNorm kernel — single-pass, bf16 in/out, F32 reduction.
//
// Algorithm (Qwen3.5-style RMSNorm, matches `rms_norm` in kiln-model/forward.rs):
//
//   sum_sq = sum_j(x[row, j]^2)            (F32 across the `hidden` axis)
//   rms_inv = rsqrt(sum_sq / hidden + eps)
//   out[row, j] = bf16((1 + w[j]) * x[row, j] * rms_inv)
//
// Launch: one block per row, blockDim.x = 256. Each thread strides over the
// hidden axis with stride == blockDim.x. Intra-block reduction of `sum_sq`
// uses a two-stage warp + shared-memory reduction (32 warps max). The
// row-wide `rms_inv` is broadcast via __shared__ memory; no second kernel
// launch required.
//
// Kept intentionally simple — no vectorised bf16 loads, no cp.async, no
// tensor cores — because the goal is to collapse the ~11 candle kernel
// launches into a single launch, and the arithmetic here is memory-bound.
// Any row width up to 8192 is supported; Qwen3.5-4B uses hidden=2560.

#include "fused_rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxWarps = kThreadsPerBlock / 32;  // 8

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float *smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        smem[warp] = v;
    }
    __syncthreads();

    // First warp reduces the per-warp partial sums.
    if (warp == 0) {
        int num_warps = blockDim.x >> 5;
        float w = (lane < num_warps) ? smem[lane] : 0.0f;
        w = warp_reduce_sum(w);
        if (lane == 0) {
            smem[0] = w;
        }
    }
    __syncthreads();
    return smem[0];
}

__global__ void fused_rmsnorm_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    int hidden,
    float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16 *x_row = x + static_cast<size_t>(row) * hidden;
    __nv_bfloat16 *out_row = out + static_cast<size_t>(row) * hidden;

    __shared__ float smem[kMaxWarps];

    // Pass 1: per-thread partial sum of x^2 in F32.
    float local_sum_sq = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float xj = __bfloat162float(x_row[j]);
        local_sum_sq += xj * xj;
    }

    float total_sum_sq = block_reduce_sum(local_sum_sq, smem);

    // `smem[0]` now holds the row-wide sum_sq. Derive rms_inv once per block
    // and broadcast through shared memory.
    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) {
        float mean_sq = total_sum_sq / static_cast<float>(hidden);
        s_rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    float rms_inv = s_rms_inv;

    // Pass 2: apply Qwen3.5-style `(1 + w) * x * rms_inv`, cast back to bf16.
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float xj = __bfloat162float(x_row[j]);
        float wj = __bfloat162float(weight[j]);
        float y = (1.0f + wj) * xj * rms_inv;
        out_row[j] = __float2bfloat16(y);
    }
}

}  // namespace

extern "C" kiln_rmsnorm_status_t kiln_fused_rmsnorm(
    const void *x,
    const void *weight,
    void *out,
    int rows,
    int hidden,
    float eps,
    void *stream
) {
    if (rows <= 0 || hidden <= 0) {
        // No-op; nothing to compute. Treat as success so callers on zero-row
        // inputs (e.g. prefill chunks of length 0) don't need to special-case.
        return 0;
    }
    if (hidden > 8192) {
        return 2;  // out-of-envelope; caller should fall back.
    }

    dim3 grid(rows);
    dim3 block(kThreadsPerBlock);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    fused_rmsnorm_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(x),
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        reinterpret_cast<__nv_bfloat16 *>(out),
        hidden,
        eps
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
