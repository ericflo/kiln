// Kiln CUDA-side masked-fill kernel.
//
// `out[i] = (mask[i] != 0) ? fill_value : x[i]` over F32 + BF16 + F16.
// Shape-matched contiguous inputs; `mask` is U8, same n_elements as
// `x`. `fill_value` arrives as F32 and is cast to the storage dtype on
// store (same F32-promotion-on-CPU dance kt uses everywhere — see
// elementwise.cu).
//
// Used by `MaskedFillOp::cuda_fwd` in `kiln-tensor/src/ops/mask.rs` —
// replaces the candle `Tensor::where_cond` call site at pre-softmax
// attention masks ("fill -inf above the diagonal").
//
// # Layout
//
// One thread per element. Inputs are *contiguous*; stride-aware
// masked-fill is a follow-up. Dispatch site can call `.contiguous()`
// first when needed (matches the existing elementwise.cu / softmax.cu
// contract).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BLOCK_SIZE 256

// Dtype tags — match `DType` indices for the supported subset
// (same as elementwise.cu / activation.cu).
#define DTYPE_F32  0
#define DTYPE_BF16 1
#define DTYPE_F16  2

namespace {

__global__ void kiln_masked_fill_f32(const float* __restrict__ x,
                                     const uint8_t* __restrict__ mask,
                                     float* __restrict__ out,
                                     int64_t n,
                                     float fill_value) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (mask[idx] != 0) ? fill_value : x[idx];
}

__global__ void kiln_masked_fill_bf16(const __nv_bfloat16* __restrict__ x,
                                      const uint8_t* __restrict__ mask,
                                      __nv_bfloat16* __restrict__ out,
                                      int64_t n,
                                      float fill_value) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (mask[idx] != 0) {
        out[idx] = __float2bfloat16(fill_value);
    } else {
        out[idx] = x[idx];
    }
}

__global__ void kiln_masked_fill_f16(const __half* __restrict__ x,
                                     const uint8_t* __restrict__ mask,
                                     __half* __restrict__ out,
                                     int64_t n,
                                     float fill_value) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (mask[idx] != 0) {
        out[idx] = __float2half(fill_value);
    } else {
        out[idx] = x[idx];
    }
}

} // namespace

extern "C" int kiln_masked_fill_u8_async(const void* x,
                                         const void* mask,
                                         void* out,
                                         int64_t n_elements,
                                         float fill_value,
                                         int dtype,
                                         cudaStream_t stream) {
    if (n_elements < 0) return 1;
    if (n_elements == 0) return 0;

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    const uint8_t* mask_u8 = static_cast<const uint8_t*>(mask);

    switch (dtype) {
        case DTYPE_F32:
            kiln_masked_fill_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(x),
                mask_u8,
                static_cast<float*>(out),
                n_elements,
                fill_value);
            break;
        case DTYPE_BF16:
            kiln_masked_fill_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x),
                mask_u8,
                static_cast<__nv_bfloat16*>(out),
                n_elements,
                fill_value);
            break;
        case DTYPE_F16:
            kiln_masked_fill_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __half*>(x),
                mask_u8,
                static_cast<__half*>(out),
                n_elements,
                fill_value);
            break;
        default:
            return 4;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
