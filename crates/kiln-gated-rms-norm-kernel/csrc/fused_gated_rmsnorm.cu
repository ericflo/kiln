// Kiln fused gated RMSNorm kernel — single-pass, bf16 in/out, F32 reduction.
//
// Algorithm (matches `gated_rms_norm` in kiln-model/forward.rs and FLA's
// `fused_norm_gate` Triton kernel):
//
//   sum_sq      = sum_j(x[row, j]^2)            (F32 across the `hidden` axis)
//   rms_inv     = rsqrt(sum_sq / hidden + eps)
//   out[row, j] = bf16(w[j] * x[row, j] * rms_inv * silu(z[row, j]))
//
// where silu(z) = z * sigmoid(z) = z / (1 + exp(-z)). All intermediate
// arithmetic is in F32; only the loads (x/z/w) and the final store (out)
// touch bf16.
//
// Launch: one block per row, blockDim.x = 256 (capped at hidden rounded up
// to 32 when hidden < 256). Each thread strides over the hidden axis with
// stride == blockDim.x. Intra-block reduction of `sum_sq` uses the standard
// warp-shuffle + shared-memory reduction. The row-wide `rms_inv` is broadcast
// via __shared__ memory; no second kernel launch required.
//
// Kept intentionally simple — no vectorised bf16 loads, no cp.async, no
// tensor cores — because the goal is to collapse the 10 candle kernel
// launches into a single launch, and the arithmetic here is memory-bound.
// Any row width up to 8192 is supported; Qwen3.5-4B uses hidden=128
// (linear_value_head_dim) for this norm.

#include "fused_gated_rmsnorm.h"

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

__device__ __forceinline__ float silu(float z) {
    // silu(z) = z * sigmoid(z) = z / (1 + exp(-z)). Stable for any finite z
    // since exp(-z) is finite when z >= 0 and sigmoid(z) = 1 - sigmoid(-z)
    // is trivially bounded for z < 0. `expf` handles the large-|z| tails.
    return z / (1.0f + __expf(-z));
}

__global__ void fused_gated_rmsnorm_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ z,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    int hidden,
    float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16 *x_row = x + static_cast<size_t>(row) * hidden;
    const __nv_bfloat16 *z_row = z + static_cast<size_t>(row) * hidden;
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

    // Pass 2: apply `w * x * rms_inv * silu(z)`, cast back to bf16.
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float xj = __bfloat162float(x_row[j]);
        float zj = __bfloat162float(z_row[j]);
        float wj = __bfloat162float(weight[j]);
        float gate = silu(zj);
        float y = wj * xj * rms_inv * gate;
        out_row[j] = __float2bfloat16(y);
    }
}

// Round `x` up to the next multiple of 32, clamped to [32, kThreadsPerBlock].
// The kernel assumes blockDim.x is a multiple of 32 (warp-aligned reduction).
__host__ int choose_threads_per_block(int hidden) {
    if (hidden <= 0) return 32;
    int t = (hidden + 31) / 32 * 32;  // round up to multiple of 32
    if (t < 32) t = 32;
    if (t > kThreadsPerBlock) t = kThreadsPerBlock;
    return t;
}

}  // namespace

extern "C" kiln_gated_rmsnorm_status_t kiln_fused_gated_rmsnorm(
    const void *x,
    const void *z,
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
    dim3 block(choose_threads_per_block(hidden));

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    fused_gated_rmsnorm_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(x),
        reinterpret_cast<const __nv_bfloat16 *>(z),
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
