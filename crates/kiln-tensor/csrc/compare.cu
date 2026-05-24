// Kiln CUDA-side element-wise comparison kernel.
//
// Covers eq / ne / lt / le / gt / ge over F32 + BF16 + F16. Same
// shape on both operands; output is a U8 mask (1 / 0 per element)
// suitable for masked_fill / where_select. Same per-element-thread
// shape as elementwise.cu.
//
// (#1082)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BLOCK_SIZE 256

// Op kinds — must match CmpKind in compare.rs.
#define KIND_EQ 0
#define KIND_NE 1
#define KIND_LT 2
#define KIND_LE 3
#define KIND_GT 4
#define KIND_GE 5

// Dtype tags
#define DTYPE_F32  0
#define DTYPE_BF16 1
#define DTYPE_F16  2

namespace {

__device__ __forceinline__ bool cmp(int kind, float a, float b) {
    switch (kind) {
        case KIND_EQ: return a == b;
        case KIND_NE: return a != b;
        case KIND_LT: return a <  b;
        case KIND_LE: return a <= b;
        case KIND_GT: return a >  b;
        case KIND_GE: return a >= b;
        default:      return false;
    }
}

__global__ void kiln_compare_f32(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 uint8_t* __restrict__ out,
                                 int64_t n,
                                 int kind) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = cmp(kind, a[idx], b[idx]) ? 1 : 0;
}

__global__ void kiln_compare_bf16(const __nv_bfloat16* __restrict__ a,
                                  const __nv_bfloat16* __restrict__ b,
                                  uint8_t* __restrict__ out,
                                  int64_t n,
                                  int kind) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float av = __bfloat162float(a[idx]);
    float bv = __bfloat162float(b[idx]);
    out[idx] = cmp(kind, av, bv) ? 1 : 0;
}

__global__ void kiln_compare_f16(const __half* __restrict__ a,
                                 const __half* __restrict__ b,
                                 uint8_t* __restrict__ out,
                                 int64_t n,
                                 int kind) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float av = __half2float(a[idx]);
    float bv = __half2float(b[idx]);
    out[idx] = cmp(kind, av, bv) ? 1 : 0;
}

} // namespace

extern "C" int kiln_compare_async(const void* a,
                                  const void* b,
                                  void* out,
                                  int64_t n_elements,
                                  int kind,
                                  int dtype,
                                  cudaStream_t stream) {
    if (n_elements < 0) return 1;
    if (kind < 0 || kind > 5) return 2;
    if (n_elements == 0) return 0;

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    switch (dtype) {
        case DTYPE_F32:
            kiln_compare_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(a),
                static_cast<const float*>(b),
                static_cast<uint8_t*>(out),
                n_elements,
                kind);
            break;
        case DTYPE_BF16:
            kiln_compare_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(a),
                static_cast<const __nv_bfloat16*>(b),
                static_cast<uint8_t*>(out),
                n_elements,
                kind);
            break;
        case DTYPE_F16:
            kiln_compare_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __half*>(a),
                static_cast<const __half*>(b),
                static_cast<uint8_t*>(out),
                n_elements,
                kind);
            break;
        default:
            return 4;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
