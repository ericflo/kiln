// Kiln CUDA-side clamp / pow unary kernel.
//
// Two scalar-parameterized unary ops:
//   clamp(x, lo, hi) = min(max(x, lo), hi)
//   pow(x, p)        = x^p
//
// Both over F32 + BF16 + F16 contiguous tensors. Mirrors the CPU
// reference in `crates/kiln-tensor/src/ops/clamp_pow.rs`: math is
// promoted to F32, then narrowed back to storage dtype on store. This
// matches the kt-tensor numerical-reference contract uniformly applied
// across all unary kernels (Phase 4 layer-glue convention).
//
// One thread per element; no shared memory. Stride-aware path is
// follow-up; the dispatch site materializes any non-contiguous input
// via `.contiguous()` first.
//
// (#1082)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BLOCK_SIZE 256

// Op kinds — match the dispatch in `ops::clamp_pow`.
#define KIND_CLAMP 0
#define KIND_POW   1

// Dtype tags
#define DTYPE_F32  0
#define DTYPE_BF16 1
#define DTYPE_F16  2

namespace {

__device__ __forceinline__ float apply_pow(float x, float p) {
    // Match Rust's `f32::powf`: for integer exponents support
    // negative bases (e.g. `(-4.0).powf(2.0) == 16.0`). The CUDA
    // device intrinsic `__powf` (used when `--use_fast_math` is on)
    // is internally `exp2f(p * log2f(x))` which returns NaN for
    // negative bases — so detect small integer exponents and do
    // repeated multiplication with sign correction.
    int p_int = static_cast<int>(p);
    if (p == static_cast<float>(p_int)) {
        // Integer exponent path.
        if (p_int == 0) return 1.0f;
        int e = p_int < 0 ? -p_int : p_int;
        // Repeated multiplication (handles small/moderate exponents
        // with bounded latency; typical use is integer powers like
        // 2, 3, 0.5, 1.5).
        float r = 1.0f;
        float base = x;
        for (int i = 0; i < e; ++i) {
            r *= base;
        }
        return p_int < 0 ? (1.0f / r) : r;
    }
    // Non-integer exponent: powf. Undefined for negative bases per
    // Rust + IEEE-754 (returns NaN, which matches kt-tensor CPU).
    return powf(x, p);
}

__device__ __forceinline__ float apply(int kind, float x, float a, float b) {
    switch (kind) {
        case KIND_CLAMP: {
            // clamp(x, lo=a, hi=b) = min(max(x, lo), hi).
            // fminf/fmaxf propagate NaN per IEEE-754 (matches Rust's
            // f32::clamp NaN behavior).
            return fminf(fmaxf(x, a), b);
        }
        case KIND_POW: {
            return apply_pow(x, a);
        }
        default:
            return 0.0f;
    }
}

__global__ void kiln_clamp_pow_f32(const float* __restrict__ x,
                                   float* __restrict__ out,
                                   int64_t n,
                                   int kind,
                                   float a,
                                   float b) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = apply(kind, x[idx], a, b);
}

__global__ void kiln_clamp_pow_bf16(const __nv_bfloat16* __restrict__ x,
                                    __nv_bfloat16* __restrict__ out,
                                    int64_t n,
                                    int kind,
                                    float a,
                                    float b) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float xv = __bfloat162float(x[idx]);
    out[idx] = __float2bfloat16(apply(kind, xv, a, b));
}

__global__ void kiln_clamp_pow_f16(const __half* __restrict__ x,
                                   __half* __restrict__ out,
                                   int64_t n,
                                   int kind,
                                   float a,
                                   float b) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float xv = __half2float(x[idx]);
    out[idx] = __float2half(apply(kind, xv, a, b));
}

} // namespace

extern "C" int kiln_clamp_pow_async(const void* x,
                                    void* out,
                                    int64_t n_elements,
                                    int kind,
                                    float a,
                                    float b,
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
            kiln_clamp_pow_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(x),
                static_cast<float*>(out),
                n_elements,
                kind,
                a,
                b);
            break;
        case DTYPE_BF16:
            kiln_clamp_pow_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x),
                static_cast<__nv_bfloat16*>(out),
                n_elements,
                kind,
                a,
                b);
            break;
        case DTYPE_F16:
            kiln_clamp_pow_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __half*>(x),
                static_cast<__half*>(out),
                n_elements,
                kind,
                a,
                b);
            break;
        default:
            return 4;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
