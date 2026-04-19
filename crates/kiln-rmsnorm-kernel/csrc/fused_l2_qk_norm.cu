// Kiln fused L2-norm(Q) + scale(Q) + L2-norm(K) kernel — single launch,
// bf16 in/out, F32 reductions.
//
// Algorithm (matches `kiln-model::forward::l2_normalize` + the
// `kiln/gdn/qk_norm` block in forward.rs:
//
//   inv_q = q_scale * rsqrt(sum_j(q[r,:]^2) + eps)        (F32)
//   inv_k =          rsqrt(sum_j(k[r,:]^2) + eps)         (F32)
//   q_out[r, j] = bf16(q[r, j] * inv_q)
//   k_out[r, j] = bf16(k[r, j] * inv_k)
//
// Launch: one block per row, blockDim.x = 256. Each thread strides over the
// hidden axis (`dk`) with stride == blockDim.x and accumulates per-thread
// sum-of-squares for both Q and K in F32. Two warp-shuffle reductions share
// the same shared-memory scratch (one barrier between them); the resulting
// `inv_q` and `inv_k` are broadcast to all threads via shared memory before
// the bf16 epilogue.
//
// Why fuse Q and K together: Q and K have identical layout and identical
// sequencing — the candle path already runs both back-to-back, and at decode
// shape (rows = 16, hidden = 128) the per-block work is tiny enough that the
// dominant cost is launch overhead and HBM round-trips, not compute. Folding
// both into one launch halves the launch count and reuses the same loaded
// row data twice (once for sum-of-squares, once for the apply pass).
//
// Kept intentionally simple — no vectorised bf16 loads, no cp.async, no
// tensor cores — for the same reason as fused_rmsnorm.cu: the goal is to
// collapse ~11 candle launches into one, and the arithmetic is memory-bound.
// Any row width up to 8192 is supported; Qwen3.5-4B uses dk = 128 here.

#include "fused_l2_qk_norm.h"

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

__global__ void fused_l2_qk_norm_kernel(
    const __nv_bfloat16 *__restrict__ q_in,
    const __nv_bfloat16 *__restrict__ k_in,
    __nv_bfloat16 *__restrict__ q_out,
    __nv_bfloat16 *__restrict__ k_out,
    int hidden,
    float q_scale,
    float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16 *q_row = q_in + static_cast<size_t>(row) * hidden;
    const __nv_bfloat16 *k_row = k_in + static_cast<size_t>(row) * hidden;
    __nv_bfloat16 *q_out_row = q_out + static_cast<size_t>(row) * hidden;
    __nv_bfloat16 *k_out_row = k_out + static_cast<size_t>(row) * hidden;

    __shared__ float smem[kMaxWarps];
    __shared__ float s_inv_q;
    __shared__ float s_inv_k;

    // Pass 1: per-thread partial sums of squares for Q and K (F32 accumulators).
    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float qj = __bfloat162float(q_row[j]);
        float kj = __bfloat162float(k_row[j]);
        q_sum_sq += qj * qj;
        k_sum_sq += kj * kj;
    }

    // Reduce Q sum-of-squares; broadcast `inv_q` (already premultiplied by
    // `q_scale`) to every thread.
    float q_total = block_reduce_sum(q_sum_sq, smem);
    if (threadIdx.x == 0) {
        s_inv_q = q_scale * rsqrtf(q_total + eps);
    }
    __syncthreads();

    // Reduce K sum-of-squares; broadcast `inv_k` to every thread.
    // `block_reduce_sum` overwrites `smem[*]` but `s_inv_q` lives in a
    // separate shared-memory slot and survives the second reduction.
    float k_total = block_reduce_sum(k_sum_sq, smem);
    if (threadIdx.x == 0) {
        s_inv_k = rsqrtf(k_total + eps);
    }
    __syncthreads();

    float inv_q = s_inv_q;
    float inv_k = s_inv_k;

    // Pass 2: apply the scale + cast back to bf16 in one fused epilogue.
    // Re-reading q_row / k_row is cheap (the L1/L2 caches still hold them
    // from pass 1) and lets us avoid a second tensor allocation.
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float qj = __bfloat162float(q_row[j]) * inv_q;
        float kj = __bfloat162float(k_row[j]) * inv_k;
        q_out_row[j] = __float2bfloat16(qj);
        k_out_row[j] = __float2bfloat16(kj);
    }
}

}  // namespace

extern "C" kiln_l2_qk_norm_status_t kiln_fused_l2_qk_norm(
    const void *q_in,
    const void *k_in,
    void *q_out,
    void *k_out,
    int rows,
    int hidden,
    float q_scale,
    float eps,
    void *stream
) {
    if (rows <= 0 || hidden <= 0) {
        // No-op; nothing to compute. Treat as success so callers on zero-row
        // inputs (e.g. empty prefill chunks) don't need to special-case.
        return 0;
    }
    if (hidden > 8192) {
        return 2;  // out-of-envelope; caller should fall back.
    }

    dim3 grid(rows);
    dim3 block(kThreadsPerBlock);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    fused_l2_qk_norm_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(q_in),
        reinterpret_cast<const __nv_bfloat16 *>(k_in),
        reinterpret_cast<__nv_bfloat16 *>(q_out),
        reinterpret_cast<__nv_bfloat16 *>(k_out),
        hidden,
        q_scale,
        eps
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
