// Kiln CUDA-side FP8 (E4M3FN) quantize / dequantize — #1082.
//
// E4M3FN format: 1 sign + 4 exponent + 3 mantissa, bias = 7, range
// [-448, 448], no NaN/Inf (all-ones exponent + all-ones mantissa is
// the max-normal 448, not NaN). Used by Phase 7 PagedKvCacheKt to
// halve KV-cache memory.
//
// Two ops:
//   quantize:   src (F32/BF16/F16) -> dst (U8 with E4M3 bit pattern)
//   dequantize: src (U8 E4M3)       -> dst (F32/BF16/F16)
//
// Both apply a uniform per-tensor scale (set by the caller via the
// kt-API Rust wrapper). Scale=1.0 is the "direct" mode used by the
// FP8 KV-cache writers (different writes carry different value
// ranges; per-tensor scaling isn't practical for cache slots).
//
// CPU reference: `crates/kiln-model/src/fp8.rs` — same E4M3FN bit
// layout, same scale semantics, same clamp-on-overflow behavior.
// Parity is exercised in `crates/kiln-kt-bridge/tests/cuda_fp8_parity.rs`.
//
// One thread per element; no shared memory. BF16/F16 inputs are
// promoted to F32, scaled, converted to E4M3 bits via a portable
// bitwise path (we do NOT use `<cuda_fp8.h>` to keep the kernel
// buildable on older toolchains and to match the CPU reference
// byte-for-byte). Dequant goes through the inverse path.
//
// NaN / Inf inputs are clamped to ±448 to match the CPU reference
// (which clamps via `min(E4M3_MAX, abs(val))`).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

#define BLOCK_SIZE 256

// Dtype tags — match `DType` for the supported numeric subset.
#define DTYPE_F32  0
#define DTYPE_BF16 1
#define DTYPE_F16  2

// Maximum representable absolute value in E4M3FN.
#define E4M3_MAX 448.0f

namespace {

// ---------------------------------------------------------------------
// Bitwise E4M3FN encode / decode. Mirrors `f32_to_e4m3` / `e4m3_to_f32`
// in crates/kiln-model/src/fp8.rs exactly.
// ---------------------------------------------------------------------

__device__ __forceinline__ uint8_t f32_to_e4m3(float val) {
    // Zero / -0 -> +0
    if (val == 0.0f || val == -0.0f) return 0;

    // NaN / Inf -> saturate (E4M3FN has no NaN/Inf encoding).
    if (isnan(val) || isinf(val)) {
        return (val < 0.0f) ? (uint8_t)0xFF : (uint8_t)0x7F;
    }

    uint8_t sign = (val < 0.0f) ? 1u : 0u;
    float abs_val = fabsf(val);

    // Clamp to representable range.
    if (abs_val > E4M3_MAX) abs_val = E4M3_MAX;

    // Subnormal threshold: smallest normal in E4M3FN is 2^-6.
    // (bias=7, smallest normal exponent = 1 - 7 = -6)
    const float min_normal = 1.0f / 64.0f; // 2^-6 = 0.015625

    if (abs_val < min_normal) {
        // Subnormal: value = 2^-6 * (mantissa/8)
        // => mantissa = round(value * 8 / 2^-6) = round(value * 512)
        uint8_t mantissa = (uint8_t)__float2int_rn(abs_val / min_normal * 8.0f);
        if (mantissa > 7) mantissa = 7;
        return (uint8_t)((sign << 7) | mantissa);
    }

    // Normal values — extract f32 fields.
    uint32_t bits = __float_as_uint(abs_val);
    int32_t f32_exp = (int32_t)((bits >> 23) & 0xFFu) - 127; // unbiased
    uint32_t f32_mantissa = bits & 0x7FFFFFu;                // 23 bits

    // E4M3 exponent: biased with bias=7, range [1, 15] for normal.
    int32_t e4m3_exp_unbiased = f32_exp;
    if (e4m3_exp_unbiased < -6) e4m3_exp_unbiased = -6;
    if (e4m3_exp_unbiased > 8)  e4m3_exp_unbiased = 8;
    uint32_t biased_exp = (uint32_t)(e4m3_exp_unbiased + 7);

    // Round 23-bit mantissa to 3 bits (round-to-nearest-even-ish via
    // half-add-and-shift, matching the CPU reference's
    // `(f32_mantissa + (1<<19)) >> 20`).
    uint32_t mantissa_3bit = (f32_mantissa + (1u << 19)) >> 20;

    if (mantissa_3bit >= 8) {
        // Mantissa overflow from rounding — bump exponent.
        biased_exp += 1;
        if (biased_exp > 15) {
            // Saturate to max normal (448).
            return (uint8_t)((sign << 7) | 0x7F);
        }
        return (uint8_t)((sign << 7) | (biased_exp << 3));
    }

    if (biased_exp > 15 || (biased_exp == 15 && mantissa_3bit > 7)) {
        return (uint8_t)((sign << 7) | 0x7F);
    }

    return (uint8_t)((sign << 7) | (biased_exp << 3) | mantissa_3bit);
}

__device__ __forceinline__ float e4m3_to_f32(uint8_t bits) {
    uint32_t sign = (uint32_t)(bits >> 7) & 1u;
    uint32_t exp = (uint32_t)(bits >> 3) & 0xFu;
    uint32_t mantissa = (uint32_t)bits & 0x7u;

    float abs_val;
    if (exp == 0) {
        // Subnormal: value = 2^-6 * (mantissa/8)
        abs_val = (1.0f / 64.0f) * ((float)mantissa / 8.0f);
    } else {
        // Normal: value = 2^(exp - 7) * (1 + mantissa/8)
        // Even exp=15 + mantissa=7 is a normal value (448), not NaN.
        int e = (int)exp - 7;
        // ldexpf is the textbook formulation; for exp in [1,15] this
        // is one ldexpf per element.
        abs_val = ldexpf(1.0f + (float)mantissa / 8.0f, e);
    }
    return (sign == 1u) ? -abs_val : abs_val;
}

// ---------------------------------------------------------------------
// Quantize: src dtype -> U8 (E4M3 bit pattern).
// scaled_src = src / scale, then encoded to E4M3.
// ---------------------------------------------------------------------

__global__ void kiln_fp8_quantize_f32(const float* __restrict__ src,
                                      uint8_t* __restrict__ dst,
                                      int64_t n,
                                      float scale_recip) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = src[idx] * scale_recip;
    dst[idx] = f32_to_e4m3(v);
}

__global__ void kiln_fp8_quantize_bf16(const __nv_bfloat16* __restrict__ src,
                                       uint8_t* __restrict__ dst,
                                       int64_t n,
                                       float scale_recip) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __bfloat162float(src[idx]) * scale_recip;
    dst[idx] = f32_to_e4m3(v);
}

__global__ void kiln_fp8_quantize_f16(const __half* __restrict__ src,
                                      uint8_t* __restrict__ dst,
                                      int64_t n,
                                      float scale_recip) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __half2float(src[idx]) * scale_recip;
    dst[idx] = f32_to_e4m3(v);
}

// ---------------------------------------------------------------------
// Dequantize: U8 (E4M3 bit pattern) -> dst dtype.
// dst = e4m3_to_f32(src) * scale
// ---------------------------------------------------------------------

__global__ void kiln_fp8_dequantize_f32(const uint8_t* __restrict__ src,
                                        float* __restrict__ dst,
                                        int64_t n,
                                        float scale) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = e4m3_to_f32(src[idx]) * scale;
}

__global__ void kiln_fp8_dequantize_bf16(const uint8_t* __restrict__ src,
                                         __nv_bfloat16* __restrict__ dst,
                                         int64_t n,
                                         float scale) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2bfloat16(e4m3_to_f32(src[idx]) * scale);
}

__global__ void kiln_fp8_dequantize_f16(const uint8_t* __restrict__ src,
                                        __half* __restrict__ dst,
                                        int64_t n,
                                        float scale) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2half(e4m3_to_f32(src[idx]) * scale);
}

} // namespace

// ---------------------------------------------------------------------
// FFI entry points.
// ---------------------------------------------------------------------

extern "C" int kiln_fp8_quantize_async(const void* src,
                                       void* dst,
                                       int64_t n_elements,
                                       float scale,
                                       int src_dtype,
                                       cudaStream_t stream) {
    if (n_elements < 0) return 1;
    if (n_elements == 0) return 0;
    if (scale == 0.0f) return 2;

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = (int)blocks_i64;

    // We multiply by 1/scale inside the kernel — pre-compute it once
    // host-side so each thread does a fast mul instead of a div.
    float scale_recip = 1.0f / scale;

    switch (src_dtype) {
        case DTYPE_F32:
            kiln_fp8_quantize_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                (const float*)src, (uint8_t*)dst, n_elements, scale_recip);
            break;
        case DTYPE_BF16:
            kiln_fp8_quantize_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                (const __nv_bfloat16*)src, (uint8_t*)dst, n_elements, scale_recip);
            break;
        case DTYPE_F16:
            kiln_fp8_quantize_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                (const __half*)src, (uint8_t*)dst, n_elements, scale_recip);
            break;
        default:
            return 4;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int)err;
    return 0;
}

extern "C" int kiln_fp8_dequantize_async(const void* src,
                                         void* dst,
                                         int64_t n_elements,
                                         float scale,
                                         int dst_dtype,
                                         cudaStream_t stream) {
    if (n_elements < 0) return 1;
    if (n_elements == 0) return 0;

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = (int)blocks_i64;

    switch (dst_dtype) {
        case DTYPE_F32:
            kiln_fp8_dequantize_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                (const uint8_t*)src, (float*)dst, n_elements, scale);
            break;
        case DTYPE_BF16:
            kiln_fp8_dequantize_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                (const uint8_t*)src, (__nv_bfloat16*)dst, n_elements, scale);
            break;
        case DTYPE_F16:
            kiln_fp8_dequantize_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                (const uint8_t*)src, (__half*)dst, n_elements, scale);
            break;
        default:
            return 4;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int)err;
    return 0;
}
