// Kiln CUDA-side dtype cast kernel.
//
// Covers the same float↔float matrix as the CPU CastOp:
//   F32 ↔ BF16 ↔ F16
// Integer round-trips (U32 ↔ I64) are CPU-only — the few call sites
// that need them have host data anyway.
//
// One thread per element. F32 staging in registers for all
// conversions (matches the kt-Tensor numerical-reference contract).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BLOCK_SIZE 256

// (from, to) tag pairs — match the dispatch in cast.rs.
#define CAST_F32_TO_BF16  0
#define CAST_F32_TO_F16   1
#define CAST_BF16_TO_F32  2
#define CAST_BF16_TO_F16  3
#define CAST_F16_TO_F32   4
#define CAST_F16_TO_BF16  5

namespace {

__global__ void kiln_cast_f32_to_bf16(const float* __restrict__ src,
                                      __nv_bfloat16* __restrict__ dst,
                                      int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2bfloat16(src[idx]);
}

__global__ void kiln_cast_f32_to_f16(const float* __restrict__ src,
                                     __half* __restrict__ dst,
                                     int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2half(src[idx]);
}

__global__ void kiln_cast_bf16_to_f32(const __nv_bfloat16* __restrict__ src,
                                      float* __restrict__ dst,
                                      int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __bfloat162float(src[idx]);
}

__global__ void kiln_cast_bf16_to_f16(const __nv_bfloat16* __restrict__ src,
                                      __half* __restrict__ dst,
                                      int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2half(__bfloat162float(src[idx]));
}

__global__ void kiln_cast_f16_to_f32(const __half* __restrict__ src,
                                     float* __restrict__ dst,
                                     int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __half2float(src[idx]);
}

__global__ void kiln_cast_f16_to_bf16(const __half* __restrict__ src,
                                      __nv_bfloat16* __restrict__ dst,
                                      int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2bfloat16(__half2float(src[idx]));
}

} // namespace

extern "C" int kiln_cast_async(const void* src,
                               void* dst,
                               int64_t n_elements,
                               int cast_tag,
                               cudaStream_t stream) {
    if (n_elements < 0) return 1;
    if (cast_tag < 0 || cast_tag > 5) return 2;
    if (n_elements == 0) return 0;

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    switch (cast_tag) {
        case CAST_F32_TO_BF16:
            kiln_cast_f32_to_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(src),
                static_cast<__nv_bfloat16*>(dst), n_elements);
            break;
        case CAST_F32_TO_F16:
            kiln_cast_f32_to_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(src),
                static_cast<__half*>(dst), n_elements);
            break;
        case CAST_BF16_TO_F32:
            kiln_cast_bf16_to_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(src),
                static_cast<float*>(dst), n_elements);
            break;
        case CAST_BF16_TO_F16:
            kiln_cast_bf16_to_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(src),
                static_cast<__half*>(dst), n_elements);
            break;
        case CAST_F16_TO_F32:
            kiln_cast_f16_to_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __half*>(src),
                static_cast<float*>(dst), n_elements);
            break;
        case CAST_F16_TO_BF16:
            kiln_cast_f16_to_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __half*>(src),
                static_cast<__nv_bfloat16*>(dst), n_elements);
            break;
        default:
            return 4;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
