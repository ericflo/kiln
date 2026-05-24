// Kiln CUDA-side tensor-scalar elementwise kernel.
//
// Covers `add_scalar`, `sub_scalar`, `mul_scalar`, `div_scalar`
// (Kind tags 0..3) and four convenience variants (4..7) over
// F32 + BF16 + F16. The scalar is passed as f32 from the host;
// BF16/F16 inputs are promoted to F32, the op runs in F32, then
// narrowed back to the storage dtype (kiln's standard
// numerical-reference promotion).
//
// Used by `ScalarOp::cuda_fwd` in `kiln-tensor/src/ops/scalar.rs`
// once the Rust wrapper in `cuda_storage.rs` (`cuda_scalar_op`)
// dispatches through this kernel.
//
// # Algorithm
//
// One thread per element. The scalar `c` is a compile-time f32
// argument applied to every element. Op selection is a runtime
// `switch` (the dispatch overhead is hidden behind the FP math).
//
// # Layout
//
// Input is *contiguous*. Stride-aware variants are a follow-up;
// the call site can `.contiguous()` first.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BLOCK_SIZE 256

// Op kinds — match `ScalarKind` in scalar.rs.
#define KIND_ADD_SCALAR          0
#define KIND_SUB_SCALAR          1
#define KIND_MUL_SCALAR          2
#define KIND_DIV_SCALAR          3
#define KIND_SCALAR_MINUS_TENSOR 4
#define KIND_SCALAR_DIV_TENSOR   5
#define KIND_MAX_WITH_SCALAR     6
#define KIND_MIN_WITH_SCALAR     7

// Dtype tags — match the F32/BF16/F16 subset used elsewhere.
#define DTYPE_F32  0
#define DTYPE_BF16 1
#define DTYPE_F16  2

namespace {

__device__ __forceinline__ float apply_scalar(int kind, float x, float c) {
    switch (kind) {
        case KIND_ADD_SCALAR:          return x + c;
        case KIND_SUB_SCALAR:          return x - c;
        case KIND_MUL_SCALAR:          return x * c;
        case KIND_DIV_SCALAR:          return x / c;
        case KIND_SCALAR_MINUS_TENSOR: return c - x;
        case KIND_SCALAR_DIV_TENSOR:   return c / x;
        case KIND_MAX_WITH_SCALAR:     return fmaxf(x, c);
        case KIND_MIN_WITH_SCALAR:     return fminf(x, c);
        default:                       return 0.0f;
    }
}

__global__ void kiln_scalar_op_f32(const float* __restrict__ x,
                                   float* __restrict__ out,
                                   int64_t n,
                                   int kind,
                                   float c) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = apply_scalar(kind, x[idx], c);
}

__global__ void kiln_scalar_op_bf16(const __nv_bfloat16* __restrict__ x,
                                    __nv_bfloat16* __restrict__ out,
                                    int64_t n,
                                    int kind,
                                    float c) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float xv = __bfloat162float(x[idx]);
    float yv = apply_scalar(kind, xv, c);
    out[idx] = __float2bfloat16(yv);
}

__global__ void kiln_scalar_op_f16(const __half* __restrict__ x,
                                   __half* __restrict__ out,
                                   int64_t n,
                                   int kind,
                                   float c) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float xv = __half2float(x[idx]);
    float yv = apply_scalar(kind, xv, c);
    out[idx] = __float2half(yv);
}

} // namespace

extern "C" int kiln_scalar_op_async(const void* x,
                                    void* out,
                                    int64_t n_elements,
                                    int kind,
                                    int dtype,
                                    float c,
                                    cudaStream_t stream) {
    if (n_elements < 0) return 1;
    if (kind < 0 || kind > 7) return 2;
    if (n_elements == 0) return 0;

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    switch (dtype) {
        case DTYPE_F32:
            kiln_scalar_op_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(x),
                static_cast<float*>(out),
                n_elements,
                kind,
                c);
            break;
        case DTYPE_BF16:
            kiln_scalar_op_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x),
                static_cast<__nv_bfloat16*>(out),
                n_elements,
                kind,
                c);
            break;
        case DTYPE_F16:
            kiln_scalar_op_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __half*>(x),
                static_cast<__half*>(out),
                n_elements,
                kind,
                c);
            break;
        default:
            return 4;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
