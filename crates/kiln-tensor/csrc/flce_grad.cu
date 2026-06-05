// CUDA FLCE chunk-gradient helper.
//
// Converts a materialized logits chunk [num_active, chunk_len] in-place into:
//
//     grad_logits = (softmax(logits) - one_hot(label)) * scale
//
// using the per-row global log-sum-exp state saved by the FLCE forward:
//
//     softmax(row, col) = exp(logits(row, col) - global_max(row))
//                         / global_sumexp(row)
//
// This removes several generic kt tensor ops from the long-context FLCE
// backward path (broadcast, sub, exp, div, scalar mul, sparse one-hot
// scatter) while still letting the existing cublasLt matmul compute
// grad_logits @ W_chunk.T.

#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

namespace {

__global__ void kiln_flce_grad_logits_chunk_f32_kernel(
    float* __restrict__ logits,
    const uint32_t* __restrict__ labels,
    const float* __restrict__ global_max,
    const float* __restrict__ global_sumexp,
    int64_t num_active,
    int64_t chunk_len,
    int64_t chunk_start,
    float scale) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = num_active * chunk_len;
    if (linear >= total) return;

    int64_t col = linear % chunk_len;
    int64_t row = linear / chunk_len;

    float v = expf(logits[linear] - global_max[row]) / global_sumexp[row];
    if (static_cast<int64_t>(labels[row]) == chunk_start + col) {
        v -= 1.0f;
    }
    logits[linear] = v * scale;
}

} // namespace

extern "C" int kiln_flce_grad_logits_chunk_f32_async(
    void* logits,
    const void* labels,
    const void* global_max,
    const void* global_sumexp,
    int64_t num_active,
    int64_t chunk_len,
    int64_t chunk_start,
    float scale,
    cudaStream_t stream) {
    if (num_active < 0 || chunk_len < 0 || chunk_start < 0) return 1;
    int64_t total = num_active * chunk_len;
    if (total == 0) return 0;
    if (chunk_len != 0 && total / chunk_len != num_active) return 2;

    int64_t blocks_i64 = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > static_cast<int64_t>(2147483647)) return 3;
    int blocks = static_cast<int>(blocks_i64);

    kiln_flce_grad_logits_chunk_f32_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<float*>(logits),
        static_cast<const uint32_t*>(labels),
        static_cast<const float*>(global_max),
        static_cast<const float*>(global_sumexp),
        num_active,
        chunk_len,
        chunk_start,
        scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
