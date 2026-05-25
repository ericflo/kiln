// Kiln CUDA-side element-wise binary minimum / maximum (#1082).
//
// Mirrors the CPU reference in `crates/kiln-tensor/src/ops/binary_minmax.rs`.
// Two same-shape, same-dtype tensors -> elementwise min(a, b) or max(a, b).
// Both over F32 + BF16 + F16 contiguous tensors. PyTorch / numpy semantics.
//
// Distinct from `reduce_arbitrary_axis_minmax_kernel` (a reduction over one
// axis): this is pointwise, no reduction tree.
//
// One thread per element; no shared memory. BF16 / F16 are loaded, promoted
// to F32, compared in F32, then narrowed back to the storage dtype (same
// numerical-reference contract as elementwise.cu / clamp_pow.cu / activation.cu).
//
// NaN propagation: fminf / fmaxf propagate the non-NaN operand when one
// side is NaN — matches Rust's `f32::min` / `f32::max` and the CPU reference's
// `x.min(y)` / `x.max(y)` calls. (PyTorch's `minimum` / `maximum` propagate NaN
// in the IEEE 754 quiet-NaN sense; kt's contract is the Rust semantics.)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BLOCK_SIZE 256

// Op kinds.
#define KIND_MINIMUM 0
#define KIND_MAXIMUM 1

// Dtype tags.
#define DTYPE_F32  0
#define DTYPE_BF16 1
#define DTYPE_F16  2

namespace {

__device__ __forceinline__ float apply(int kind, float a, float b) {
    switch (kind) {
        case KIND_MINIMUM: return fminf(a, b);
        case KIND_MAXIMUM: return fmaxf(a, b);
        default:           return 0.0f;
    }
}

__global__ void kiln_binary_minmax_f32(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ out,
                                       int64_t n,
                                       int kind) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = apply(kind, a[idx], b[idx]);
}

__global__ void kiln_binary_minmax_bf16(const __nv_bfloat16* __restrict__ a,
                                        const __nv_bfloat16* __restrict__ b,
                                        __nv_bfloat16* __restrict__ out,
                                        int64_t n,
                                        int kind) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float av = __bfloat162float(a[idx]);
    float bv = __bfloat162float(b[idx]);
    out[idx] = __float2bfloat16(apply(kind, av, bv));
}

__global__ void kiln_binary_minmax_f16(const __half* __restrict__ a,
                                       const __half* __restrict__ b,
                                       __half* __restrict__ out,
                                       int64_t n,
                                       int kind) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float av = __half2float(a[idx]);
    float bv = __half2float(b[idx]);
    out[idx] = __float2half(apply(kind, av, bv));
}

} // namespace

extern "C" int kiln_binary_minmax_async(const void* a,
                                        const void* b,
                                        void* out,
                                        int64_t n_elements,
                                        int kind,
                                        int dtype,
                                        cudaStream_t stream) {
    if (n_elements < 0) return 1;
    if (kind < 0 || kind > 1) return 2;
    if (n_elements == 0) return 0;

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    switch (dtype) {
        case DTYPE_F32:
            kiln_binary_minmax_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(a),
                static_cast<const float*>(b),
                static_cast<float*>(out),
                n_elements,
                kind);
            break;
        case DTYPE_BF16:
            kiln_binary_minmax_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(a),
                static_cast<const __nv_bfloat16*>(b),
                static_cast<__nv_bfloat16*>(out),
                n_elements,
                kind);
            break;
        case DTYPE_F16:
            kiln_binary_minmax_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
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
