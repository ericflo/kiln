// Kiln CUDA-side dropout kernel (inverted dropout, training-time).
//
// `out[i] = (rand_i >= p) ? x[i] / (1 - p) : 0` with the survival mask
// emitted to `mask[i] ∈ {0, 1}`. Shape-matched contiguous inputs;
// supports F32 + BF16 + F16. The per-element random value comes from
// a splitmix64-style hash of `(seed, i)` — fully deterministic given
// `seed`, and trivially parallelizable since each element samples
// independently (unlike the CPU op's sequential splitmix64 chain).
//
// Used by `dropout()` in `kiln-tensor/src/ops/dropout.rs`. The CPU
// path keeps its sequential RNG (different mask byte stream); the
// CUDA path is checked at the distribution / scale / mask-shape
// level via `crates/kiln-kt-bridge/tests/cuda_dropout_parity.rs`,
// plus determinism (same seed -> same output).

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

// splitmix64-style hash. Returns a uniform [0, 1) f32 from the top
// 24 bits — same bit-extraction shape as the CPU op's `next_u01`
// helper, but seeded per-element so we don't need a sequential chain.
__device__ __forceinline__ float dropout_u01(uint64_t seed, int64_t idx) {
    // Mix the global seed with the element index to get a unique
    // per-element state. Use a couple of large odd multipliers so
    // distinct (seed, idx) pairs land in different splitmix orbits.
    uint64_t z = seed
        + static_cast<uint64_t>(idx) * 0x9E3779B97F4A7C15ULL
        + 0xDEADBEEFCAFEBABEULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    uint32_t bits = static_cast<uint32_t>(z >> 40);
    return static_cast<float>(bits) / static_cast<float>(1u << 24);
}

__global__ void kiln_dropout_f32(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 uint8_t* __restrict__ mask,
                                 int64_t n,
                                 float p,
                                 float inv_keep,
                                 uint64_t seed) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float r = dropout_u01(seed, idx);
    bool keep = (r >= p);
    mask[idx] = keep ? 1u : 0u;
    float xv = x[idx];
    y[idx] = keep ? (xv * inv_keep) : 0.0f;
}

__global__ void kiln_dropout_bf16(const __nv_bfloat16* __restrict__ x,
                                  __nv_bfloat16* __restrict__ y,
                                  uint8_t* __restrict__ mask,
                                  int64_t n,
                                  float p,
                                  float inv_keep,
                                  uint64_t seed) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float r = dropout_u01(seed, idx);
    bool keep = (r >= p);
    mask[idx] = keep ? 1u : 0u;
    float xv = __bfloat162float(x[idx]);
    float yv = keep ? (xv * inv_keep) : 0.0f;
    y[idx] = __float2bfloat16(yv);
}

__global__ void kiln_dropout_f16(const __half* __restrict__ x,
                                 __half* __restrict__ y,
                                 uint8_t* __restrict__ mask,
                                 int64_t n,
                                 float p,
                                 float inv_keep,
                                 uint64_t seed) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float r = dropout_u01(seed, idx);
    bool keep = (r >= p);
    mask[idx] = keep ? 1u : 0u;
    float xv = __half2float(x[idx]);
    float yv = keep ? (xv * inv_keep) : 0.0f;
    y[idx] = __float2half(yv);
}

} // namespace

extern "C" int kiln_dropout_async(const void* x,
                                  void* y,
                                  void* mask,
                                  int64_t n_elements,
                                  float p,
                                  float inv_keep,
                                  uint64_t seed,
                                  int dtype,
                                  cudaStream_t stream) {
    if (n_elements < 0) return 1;
    if (n_elements == 0) return 0;

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    uint8_t* mask_u8 = static_cast<uint8_t*>(mask);

    switch (dtype) {
        case DTYPE_F32:
            kiln_dropout_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(x),
                static_cast<float*>(y),
                mask_u8,
                n_elements,
                p,
                inv_keep,
                seed);
            break;
        case DTYPE_BF16:
            kiln_dropout_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x),
                static_cast<__nv_bfloat16*>(y),
                mask_u8,
                n_elements,
                p,
                inv_keep,
                seed);
            break;
        case DTYPE_F16:
            kiln_dropout_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __half*>(x),
                static_cast<__half*>(y),
                mask_u8,
                n_elements,
                p,
                inv_keep,
                seed);
            break;
        default:
            return 4;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
