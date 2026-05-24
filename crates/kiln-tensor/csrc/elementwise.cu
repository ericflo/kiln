// Kiln CUDA-side element-wise binary kernel.
//
// Covers add / sub / mul / div over F32 + BF16 + F16. Same shape +
// dtype on both operands; output has matching shape + dtype.
//
// Used by Phase 4 layer-glue (residual add, etc.) once
// `ElementwiseOp::cuda_fwd` routes through this kernel — see the
// dispatch in `kiln-tensor/src/ops/elementwise.rs`.
//
// # Algorithm
//
// One thread per element. BF16/F16 are loaded, promoted to F32, the
// op computed in F32, then narrowed back to the storage dtype (kiln's
// numerical-reference contract — Phase 1 line "F32-promotion-on-CPU
// dance" applied uniformly here).
//
// # Layout
//
// Inputs are *contiguous*. Stride-aware element-wise is a follow-up;
// for now the dispatch site can call `.contiguous()` to materialize
// any non-contiguous input (the CUDA contiguous kernel from PR #1374).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BLOCK_SIZE 256

// Op kinds — match `BinaryKind` in elementwise.rs
#define KIND_ADD 0
#define KIND_SUB 1
#define KIND_MUL 2
#define KIND_DIV 3

// Dtype tags — match `DType` indices for the supported subset.
#define DTYPE_F32  0
#define DTYPE_BF16 1
#define DTYPE_F16  2

namespace {

__device__ __forceinline__ float apply(int kind, float a, float b) {
    switch (kind) {
        case KIND_ADD: return a + b;
        case KIND_SUB: return a - b;
        case KIND_MUL: return a * b;
        case KIND_DIV: return a / b;
        default:       return 0.0f;
    }
}

__global__ void kiln_elementwise_binary_f32(const float* __restrict__ a,
                                            const float* __restrict__ b,
                                            float* __restrict__ out,
                                            int64_t n,
                                            int kind) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = apply(kind, a[idx], b[idx]);
}

__global__ void kiln_elementwise_binary_bf16(const __nv_bfloat16* __restrict__ a,
                                             const __nv_bfloat16* __restrict__ b,
                                             __nv_bfloat16* __restrict__ out,
                                             int64_t n,
                                             int kind) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float av = __bfloat162float(a[idx]);
    float bv = __bfloat162float(b[idx]);
    float cv = apply(kind, av, bv);
    out[idx] = __float2bfloat16(cv);
}

__global__ void kiln_elementwise_binary_f16(const __half* __restrict__ a,
                                            const __half* __restrict__ b,
                                            __half* __restrict__ out,
                                            int64_t n,
                                            int kind) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float av = __half2float(a[idx]);
    float bv = __half2float(b[idx]);
    float cv = apply(kind, av, bv);
    out[idx] = __float2half(cv);
}

} // namespace

extern "C" int kiln_elementwise_binary_async(const void* a,
                                             const void* b,
                                             void* out,
                                             int64_t n_elements,
                                             int kind,
                                             int dtype,
                                             cudaStream_t stream) {
    if (n_elements < 0) return 1;
    if (kind < 0 || kind > 3) return 2;
    if (n_elements == 0) return 0;

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    switch (dtype) {
        case DTYPE_F32:
            kiln_elementwise_binary_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(a),
                static_cast<const float*>(b),
                static_cast<float*>(out),
                n_elements,
                kind);
            break;
        case DTYPE_BF16:
            kiln_elementwise_binary_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(a),
                static_cast<const __nv_bfloat16*>(b),
                static_cast<__nv_bfloat16*>(out),
                n_elements,
                kind);
            break;
        case DTYPE_F16:
            kiln_elementwise_binary_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __half*>(a),
                static_cast<const __half*>(b),
                static_cast<__half*>(out),
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
