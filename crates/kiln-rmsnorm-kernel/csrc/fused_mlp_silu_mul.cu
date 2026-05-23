#include "fused_mlp_silu_mul.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <climits>

namespace {

constexpr int kThreadsPerBlock = 256;

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void fused_mlp_silu_mul_bf16_kernel(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16 *__restrict__ out,
    int64_t elems
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) return;

    const float g = __bfloat162float(gate[idx]);
    const float u = __bfloat162float(up[idx]);
    out[idx] = __float2bfloat16(silu(g) * u);
}

}  // namespace

extern "C" int32_t kiln_fused_mlp_silu_mul_bf16(
    const void *gate,
    const void *up,
    void *out,
    int64_t elems,
    void *stream
) {
    if (elems < 0) return -1;
    if (elems == 0) return 0;

    const int64_t blocks64 = (elems + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blocks64 > static_cast<int64_t>(INT_MAX)) return -2;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    fused_mlp_silu_mul_bf16_kernel<<<static_cast<int>(blocks64), kThreadsPerBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(gate),
        reinterpret_cast<const __nv_bfloat16 *>(up),
        reinterpret_cast<__nv_bfloat16 *>(out),
        elems);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

namespace {

// Read gate and up from the two halves of the same row of a packed
// [rows, 2*cols] BF16 tensor. The two loads are 4-byte apart (2 BF16) within
// the same cacheline once cols*2 stays L1-resident, so the per-thread
// memory cost is one cacheline read for both operands instead of two
// separate stream loads.
__global__ void fused_mlp_silu_mul_packed_bf16_kernel(
    const __nv_bfloat16 *__restrict__ gate_up_packed,
    __nv_bfloat16 *__restrict__ out,
    int64_t rows,
    int64_t cols
) {
    int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (out_idx >= total) return;
    int64_t row = out_idx / cols;
    int64_t col = out_idx - row * cols;
    int64_t row_base = row * (cols * 2);
    const float g = __bfloat162float(gate_up_packed[row_base + col]);
    const float u = __bfloat162float(gate_up_packed[row_base + cols + col]);
    out[out_idx] = __float2bfloat16(silu(g) * u);
}

}  // namespace

extern "C" int32_t kiln_fused_mlp_silu_mul_packed_bf16(
    const void *gate_up_packed,
    void *out,
    int64_t rows,
    int64_t cols,
    void *stream
) {
    if (rows < 0 || cols < 0) return -1;
    if (rows == 0 || cols == 0) return 0;

    const int64_t elems = rows * cols;
    const int64_t blocks64 = (elems + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blocks64 > static_cast<int64_t>(INT_MAX)) return -2;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    fused_mlp_silu_mul_packed_bf16_kernel<<<static_cast<int>(blocks64), kThreadsPerBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(gate_up_packed),
        reinterpret_cast<__nv_bfloat16 *>(out),
        rows,
        cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}
