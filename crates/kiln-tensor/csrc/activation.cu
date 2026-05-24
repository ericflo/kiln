// Kiln CUDA-side unary activation kernel.
//
// Covers silu / sigmoid / gelu / tanh / relu over F32 + BF16 + F16.
// In-place math in F32 (kiln's numerical-reference convention),
// narrowed back to the storage dtype on store.
//
// Used by Phase 4 layer-glue when no fused kernel is available
// (the gated MLP path has its own fused silu*mul; standalone silu /
// sigmoid / gelu hit this kernel).
//
// # Algorithm
//
// One thread per element. BF16/F16 inputs are loaded, promoted to
// F32, the activation computed in F32, narrowed back. F32 inputs
// run directly.
//
// # Layout
//
// Inputs are *contiguous*. Stride-aware path is a follow-up; the
// dispatch site can call `.contiguous()` (CUDA contiguous landed
// in PR #1374).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <math_constants.h>

#define BLOCK_SIZE 256

// Op kinds — match `UnaryKind` in activation.rs
#define KIND_SILU    0
#define KIND_SIGMOID 1
#define KIND_GELU    2
#define KIND_TANH    3
#define KIND_RELU    4

// Dtype tags
#define DTYPE_F32  0
#define DTYPE_BF16 1
#define DTYPE_F16  2

namespace {

__device__ __forceinline__ float apply_unary(int kind, float x) {
    switch (kind) {
        case KIND_SILU: {
            // x / (1 + exp(-x))
            return x / (1.0f + __expf(-x));
        }
        case KIND_SIGMOID: {
            return 1.0f / (1.0f + __expf(-x));
        }
        case KIND_GELU: {
            // tanh-approximation matching the kt-tensor CPU reference:
            //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const float kSqrt2OverPi = 0.7978845608028654f;
            float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.0f + tanhf(inner));
        }
        case KIND_TANH: {
            return tanhf(x);
        }
        case KIND_RELU: {
            return fmaxf(0.0f, x);
        }
        default:
            return 0.0f;
    }
}

__global__ void kiln_activation_f32(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    int64_t n,
                                    int kind) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = apply_unary(kind, x[idx]);
}

__global__ void kiln_activation_bf16(const __nv_bfloat16* __restrict__ x,
                                     __nv_bfloat16* __restrict__ out,
                                     int64_t n,
                                     int kind) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float xv = __bfloat162float(x[idx]);
    out[idx] = __float2bfloat16(apply_unary(kind, xv));
}

__global__ void kiln_activation_f16(const __half* __restrict__ x,
                                    __half* __restrict__ out,
                                    int64_t n,
                                    int kind) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float xv = __half2float(x[idx]);
    out[idx] = __float2half(apply_unary(kind, xv));
}

} // namespace

extern "C" int kiln_activation_unary_async(const void* x,
                                           void* out,
                                           int64_t n_elements,
                                           int kind,
                                           int dtype,
                                           cudaStream_t stream) {
    if (n_elements < 0) return 1;
    if (kind < 0 || kind > 4) return 2;
    if (n_elements == 0) return 0;

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    switch (dtype) {
        case DTYPE_F32:
            kiln_activation_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(x),
                static_cast<float*>(out),
                n_elements,
                kind);
            break;
        case DTYPE_BF16:
            kiln_activation_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x),
                static_cast<__nv_bfloat16*>(out),
                n_elements,
                kind);
            break;
        case DTYPE_F16:
            kiln_activation_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __half*>(x),
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
