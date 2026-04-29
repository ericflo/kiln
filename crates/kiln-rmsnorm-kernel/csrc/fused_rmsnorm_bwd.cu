// Kiln fused RMSNorm backward kernel — single-pass per row, F32 reductions,
// F32 atomicAdd for the cross-row grad_w sum.
//
// Algorithm (matches the math in fused_rmsnorm_bwd.h):
//
//   Pass 1: sum_x2 = sum_j x[i,j]^2          (F32 across hidden)
//   rms_inv = rsqrt(sum_x2 / H + eps)
//
//   Pass 2: sum_xgw = sum_j ((1 + w[j]) * x[i,j] * grad_out[i,j])
//   c = (1/H) * rms_inv^2 * sum_xgw
//
//   Pass 3 (write-out + grad_w accumulate):
//     grad_x[i,j] = rms_inv * ((1 + w[j]) * grad_out[i,j] - x[i,j] * c)
//     atomicAdd(&grad_w_partial_f32[j], x[i,j] * rms_inv * grad_out[i,j])
//
// Launch: one block per row, 256 threads/block. Each thread strides over
// the hidden axis with stride == blockDim.x. Two-stage warp + smem reduction
// for the per-row sums. Cross-row grad_w accumulation uses atomicAdd into a
// caller-provided F32 buffer; the caller zeros the buffer before launch.
//
// `hidden` <= 8192 (matches the forward kernel envelope).

#include "fused_rmsnorm_bwd.h"

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

__global__ void fused_rmsnorm_bwd_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ grad_out,
    __nv_bfloat16 *__restrict__ grad_x,
    float *__restrict__ grad_w_partial_f32,
    int hidden,
    float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16 *x_row = x + static_cast<size_t>(row) * hidden;
    const __nv_bfloat16 *g_row = grad_out + static_cast<size_t>(row) * hidden;
    __nv_bfloat16 *dx_row = grad_x + static_cast<size_t>(row) * hidden;

    __shared__ float smem[kMaxWarps];

    // Pass 1: per-thread partial sum of x^2 in F32.
    float local_sum_sq = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float xj = __bfloat162float(x_row[j]);
        local_sum_sq += xj * xj;
    }
    float total_sum_sq = block_reduce_sum(local_sum_sq, smem);

    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) {
        float mean_sq = total_sum_sq / static_cast<float>(hidden);
        s_rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    float rms_inv = s_rms_inv;

    // Pass 2: sum_xgw = sum_j ((1 + w_j) * x_ij * g_ij).
    float local_sum_xgw = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float xj = __bfloat162float(x_row[j]);
        float wj = __bfloat162float(weight[j]);
        float gj = __bfloat162float(g_row[j]);
        local_sum_xgw += (1.0f + wj) * xj * gj;
    }
    float total_sum_xgw = block_reduce_sum(local_sum_xgw, smem);

    __shared__ float s_c;
    if (threadIdx.x == 0) {
        s_c = total_sum_xgw / static_cast<float>(hidden) * rms_inv * rms_inv;
    }
    __syncthreads();
    float c = s_c;

    // Pass 3: grad_x = rms_inv * ((1 + w) * grad_out - x * c) and
    // atomic-add the per-row contribution to grad_w[j] into the F32 buffer.
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float xj = __bfloat162float(x_row[j]);
        float wj = __bfloat162float(weight[j]);
        float gj = __bfloat162float(g_row[j]);
        float dx = rms_inv * ((1.0f + wj) * gj - xj * c);
        dx_row[j] = __float2bfloat16(dx);

        // grad_w[j] += x_ij * rms_inv_i * g_ij  (cross-row reduction)
        float dw_contrib = xj * rms_inv * gj;
        atomicAdd(&grad_w_partial_f32[j], dw_contrib);
    }
}

__global__ void f32_to_bf16_kernel(
    const float *__restrict__ src,
    __nv_bfloat16 *__restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

}  // namespace

extern "C" kiln_rmsnorm_bwd_status_t kiln_fused_rmsnorm_bwd(
    const void *x,
    const void *weight,
    const void *grad_out,
    void *grad_x,
    float *grad_w_partial_f32,
    int rows,
    int hidden,
    float eps,
    void *stream
) {
    if (rows <= 0 || hidden <= 0) {
        return 0;
    }
    if (hidden > 8192) {
        return 2;
    }

    dim3 grid(rows);
    dim3 block(kThreadsPerBlock);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    fused_rmsnorm_bwd_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(x),
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        reinterpret_cast<const __nv_bfloat16 *>(grad_out),
        reinterpret_cast<__nv_bfloat16 *>(grad_x),
        grad_w_partial_f32,
        hidden,
        eps
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

extern "C" kiln_rmsnorm_bwd_status_t kiln_f32_to_bf16(
    const float *src,
    void *dst,
    int n,
    void *stream
) {
    if (n <= 0) {
        return 0;
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    f32_to_bf16_kernel<<<blocks, threads, 0, s>>>(
        src,
        reinterpret_cast<__nv_bfloat16 *>(dst),
        n
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
