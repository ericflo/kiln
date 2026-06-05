// ROCm/CUDA-compatible W8A16 GEMV for single-token decode projections.
//
// Computes C[m, n] = A[m, k] @ dequant(W_q[n, k])^T where W_q is signed
// int8 stored in a U8 tensor and each output row has one F32 scale.
// A and C are BF16. This is intentionally a bandwidth-cutting decode kernel,
// not a general GEMM replacement.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

#include "kt_gpu_compat.cuh"

namespace {

constexpr int BLOCK = 64;
constexpr int ARGMAX_BLOCK = 256;

__global__ void kiln_w8a16_gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_q_u8,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    int64_t m,
    int64_t n,
    int64_t k) {
    int64_t col = static_cast<int64_t>(blockIdx.x);
    int64_t row = static_cast<int64_t>(blockIdx.y);
    int tid = threadIdx.x;
    if (row >= m || col >= n) return;

    const __nv_bfloat16* x_row = x + row * k;
    const int8_t* w_row = reinterpret_cast<const int8_t*>(w_q_u8 + col * k);

    float local = 0.0f;
    for (int64_t c = tid; c < k; c += BLOCK) {
        float xv = __bfloat162float(x_row[c]);
        float wv = static_cast<float>(w_row[c]);
        local += xv * wv;
    }

    __shared__ float smem[BLOCK];
    float sum = kiln_block_reduce_sum(local, smem);
    if (tid == 0) {
        out[row * n + col] = __float2bfloat16(sum * scales[col]);
    }
}

__global__ void kiln_w8a16_gemv_argmax_scores_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_q_u8,
    const float* __restrict__ scales,
    float* __restrict__ scores,
    int64_t n,
    int64_t k) {
    int64_t col = static_cast<int64_t>(blockIdx.x);
    int tid = threadIdx.x;
    if (col >= n) return;

    const int8_t* w_row = reinterpret_cast<const int8_t*>(w_q_u8 + col * k);

    float local = 0.0f;
    for (int64_t c = tid; c < k; c += BLOCK) {
        float xv = __bfloat162float(x[c]);
        float wv = static_cast<float>(w_row[c]);
        local += xv * wv;
    }

    __shared__ float smem[BLOCK];
    float sum = kiln_block_reduce_sum(local, smem);
    if (tid == 0) {
        scores[col] = sum * scales[col];
    }
}

__global__ void kiln_w8a16_gemv_argmax_reduce_kernel(
    const float* __restrict__ scores,
    int64_t* __restrict__ out_idx,
    int64_t n) {
    int tid = threadIdx.x;
    float best_val = -INFINITY;
    int64_t best_idx = 0;
    for (int64_t i = tid; i < n; i += ARGMAX_BLOCK) {
        float v = scores[i];
        if (v > best_val || (v == best_val && i < best_idx)) {
            best_val = v;
            best_idx = i;
        }
    }

    __shared__ float vals[ARGMAX_BLOCK];
    __shared__ int64_t idxs[ARGMAX_BLOCK];
    vals[tid] = best_val;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = ARGMAX_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_val = vals[tid + stride];
            int64_t other_idx = idxs[tid + stride];
            if (other_val > vals[tid] || (other_val == vals[tid] && other_idx < idxs[tid])) {
                vals[tid] = other_val;
                idxs[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[0] = idxs[0];
    }
}

}  // namespace

extern "C" int kiln_w8a16_gemv_bf16_async(
    const void* x,
    const void* w_q,
    const void* scales,
    void* out,
    int64_t m,
    int64_t n,
    int64_t k,
    void* stream_raw) {
    if (m < 0 || n < 0 || k < 0) return 1;
    if (m == 0 || n == 0 || k == 0) return 0;
    if (n > static_cast<int64_t>(2147483647) || m > static_cast<int64_t>(65535)) {
        return 2;
    }

    dim3 grid(static_cast<unsigned int>(n), static_cast<unsigned int>(m), 1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    kiln_w8a16_gemv_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const uint8_t*>(w_q),
        static_cast<const float*>(scales),
        static_cast<__nv_bfloat16*>(out),
        m,
        n,
        k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_w8a16_gemv_argmax_bf16_async(
    const void* x,
    const void* w_q,
    const void* scales,
    void* scores,
    void* out_idx,
    int64_t n,
    int64_t k,
    void* stream_raw) {
    if (n < 0 || k < 0) return 1;
    if (n == 0 || k == 0) return 2;
    if (n > static_cast<int64_t>(2147483647)) return 3;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    kiln_w8a16_gemv_argmax_scores_kernel<<<static_cast<unsigned int>(n), BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const uint8_t*>(w_q),
        static_cast<const float*>(scales),
        static_cast<float*>(scores),
        n,
        k);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);

    kiln_w8a16_gemv_argmax_reduce_kernel<<<1, ARGMAX_BLOCK, 0, stream>>>(
        static_cast<const float*>(scores),
        static_cast<int64_t*>(out_idx),
        n);
    err = cudaGetLastError();
    if (err != cudaSuccess) return 2000 + static_cast<int>(err);
    return 0;
}
