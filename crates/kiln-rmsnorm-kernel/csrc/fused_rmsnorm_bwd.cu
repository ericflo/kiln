// Kiln fused RMSNorm backward kernel — single-pass per row, F32 reductions,
// and an optional F32 atomicAdd for the cross-row grad_w sum.
//
// Algorithm (matches the math in fused_rmsnorm_bwd.h):
//
//   Pass 1: sum_x2 = sum_j x[i,j]^2          (F32 across hidden)
//   rms_inv = rsqrt(sum_x2 / H + eps)
//
//   Pass 2: sum_xgw = sum_j ((1 + w[j]) * x[i,j] * grad_out[i,j])
//   c = (1/H) * rms_inv^2 * sum_xgw
//
//   Pass 3 (write-out + optional grad_w accumulation):
//     grad_x[i,j] = rms_inv * ((1 + w[j]) * grad_out[i,j] - x[i,j] * c)
//     atomicAdd(&grad_weight_f32[j], x[i,j] * rms_inv * grad_out[i,j])
//
// Launch: one block per row, 256 threads/block. Each thread strides over
// the hidden axis with stride == blockDim.x. Two-stage warp + smem reduction
// for the per-row sums. When requested, cross-row grad_w accumulation uses
// atomicAdd into a caller-provided F32 buffer that is zeroed before launch.
//
// `hidden` <= 8192 (matches the forward kernel envelope).

#include "fused_rmsnorm_bwd.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Wave-size shim (Phase R.7). See fused_rmsnorm.cu: the per-row F32 sums are
// reduced via the wave-agnostic shared-memory kiln_block_reduce_sum so wave64
// (AMD CDNA / RDNA-wave64) is correct. The cross-row grad_w atomicAdd is on F32
// (hipify-clean) and unchanged. blockDim.x = 256 (power of two >= 64).
#include "kt_gpu_compat.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;

template <bool ComputeGradWeight>
__global__ void fused_rmsnorm_bwd_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ grad_out,
    __nv_bfloat16 *__restrict__ grad_x,
    float *__restrict__ grad_weight_f32,
    int hidden,
    float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16 *x_row = x + static_cast<size_t>(row) * hidden;
    const __nv_bfloat16 *g_row = grad_out + static_cast<size_t>(row) * hidden;
    __nv_bfloat16 *dx_row = grad_x + static_cast<size_t>(row) * hidden;

    __shared__ float smem[kThreadsPerBlock];

    // Pass 1: per-thread partial sum of x^2 in F32.
    float local_sum_sq = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float xj = __bfloat162float(x_row[j]);
        local_sum_sq += xj * xj;
    }
    float total_sum_sq = kiln_block_reduce_sum(local_sum_sq, smem);

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
    float total_sum_xgw = kiln_block_reduce_sum(local_sum_xgw, smem);

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

        if constexpr (ComputeGradWeight) {
            // grad_w[j] += x_ij * rms_inv_i * g_ij  (cross-row reduction)
            float dw_contrib = xj * rms_inv * gj;
            atomicAdd(&grad_weight_f32[j], dw_contrib);
        }
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
    float *grad_weight_f32,
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

    if (grad_weight_f32 != nullptr) {
        fused_rmsnorm_bwd_kernel<true><<<grid, block, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            reinterpret_cast<const __nv_bfloat16 *>(grad_out),
            reinterpret_cast<__nv_bfloat16 *>(grad_x),
            grad_weight_f32,
            hidden,
            eps
        );
    } else {
        fused_rmsnorm_bwd_kernel<false><<<grid, block, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16 *>(x),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            reinterpret_cast<const __nv_bfloat16 *>(grad_out),
            reinterpret_cast<__nv_bfloat16 *>(grad_x),
            nullptr,
            hidden,
            eps
        );
    }

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
