// Kiln CUDA-side linear interpolation (lerp) — #1082.
//
//   out = a + weight * (b - a)
//
// Element-wise, both tensors must share shape + dtype. Weight is a
// scalar f32 (sent into the kernel as a uniform — same launch shape
// as the other elementwise/scalar-op kernels). PyTorch parity with
// `torch.lerp(a, b, weight)`.
//
// Mirrors the CPU reference in `crates/kiln-tensor/src/ops/lerp.rs`.
// Used by EMA weight averaging, Lion/Muon momentum updates, and DPO
// reference-policy mixing.
//
// One thread per element; no shared memory. BF16/F16 are loaded,
// promoted to F32, the fused-multiply-add computed in F32, then
// narrowed back to the storage dtype (same numerical-reference
// convention as elementwise.cu / activation.cu / clamp_pow.cu).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BLOCK_SIZE 256

// Dtype tags — match `DType` indices for the supported subset.
#define DTYPE_F32  0
#define DTYPE_BF16 1
#define DTYPE_F16  2

namespace {

__device__ __forceinline__ float apply_lerp(float a, float b, float w) {
    // `fmaf(w, b - a, a)` would give us a single fused multiply-add,
    // but the explicit (a + w * (b - a)) form matches the CPU
    // reference's evaluation order one-for-one and avoids surprising
    // ULP drift in the parity tests. The compiler will still fuse it
    // when `--use_fast_math` is on (set by build.rs).
    return a + w * (b - a);
}

__global__ void kiln_lerp_f32(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ out,
                              int64_t n,
                              float weight) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = apply_lerp(a[idx], b[idx], weight);
}

__global__ void kiln_lerp_bf16(const __nv_bfloat16* __restrict__ a,
                               const __nv_bfloat16* __restrict__ b,
                               __nv_bfloat16* __restrict__ out,
                               int64_t n,
                               float weight) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float av = __bfloat162float(a[idx]);
    float bv = __bfloat162float(b[idx]);
    out[idx] = __float2bfloat16(apply_lerp(av, bv, weight));
}

__global__ void kiln_lerp_f16(const __half* __restrict__ a,
                              const __half* __restrict__ b,
                              __half* __restrict__ out,
                              int64_t n,
                              float weight) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float av = __half2float(a[idx]);
    float bv = __half2float(b[idx]);
    out[idx] = __float2half(apply_lerp(av, bv, weight));
}

} // namespace

extern "C" int kiln_lerp_async(const void* a,
                               const void* b,
                               void* out,
                               int64_t n_elements,
                               float weight,
                               int dtype,
                               cudaStream_t stream) {
    if (n_elements < 0) return 1;
    if (n_elements == 0) return 0;

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    switch (dtype) {
        case DTYPE_F32:
            kiln_lerp_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(a),
                static_cast<const float*>(b),
                static_cast<float*>(out),
                n_elements,
                weight);
            break;
        case DTYPE_BF16:
            kiln_lerp_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(a),
                static_cast<const __nv_bfloat16*>(b),
                static_cast<__nv_bfloat16*>(out),
                n_elements,
                weight);
            break;
        case DTYPE_F16:
            kiln_lerp_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __half*>(a),
                static_cast<const __half*>(b),
                static_cast<__half*>(out),
                n_elements,
                weight);
            break;
        default:
            return 4;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
