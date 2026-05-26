// Phase 9 substrate kernel: "any non-finite?" tensor-wide reduction.
//
// Used by `Tensor::all_finite()` (kt-tensor side of the
// `kiln_autograd::anomaly_detection_enabled()` / `anomaly_panic()`
// pair, #1082 Phase 9) to scan CUDA-resident gradient tensors without
// the D2H bridge through `cuda_to_host_copy`.
//
// # Contract
//
// Given a contiguous, `start_offset == 0` device buffer of `n_elements`
// of dtype `dtype_tag`, atomically write `1` into `*out_flag` if any
// element is non-finite (NaN, +Inf, or -Inf). Otherwise leave it at
// its initial value (caller pre-zeros).
//
// Caller is responsible for:
//   * allocating + zeroing the 4-byte `out_flag` device buffer
//     (`cuda_zeros` over `DType::U32` of len 1 is fine).
//   * doing a single D2H of those 4 bytes after the launch returns.
//
// # Early-exit
//
// Once any thread has set the flag, subsequent threads in the same
// block test the flag first and skip their remaining strided range.
// This is *advisory* — the flag is read non-atomically — but cuts
// the wasted work on the "found a NaN early" case dramatically.
// The launch still has to drain in-flight blocks; that's fine, the
// goal is correctness with low average cost, not strict early-out.
//
// # Determinism
//
// The output is a single bit (`0` or `1`); the order in which blocks
// atomicOr their findings does not change the final value. Bit-exact
// across runs.
//
// # Dtype coverage
//
// | dtype     | check                                           |
// |-----------|--------------------------------------------------|
// | F32       | `!isfinite(x)`                                  |
// | BF16/F16  | promote to f32, `!isfinite(x)`                  |
// | F8E4M3FN  | `(byte & 0x7F) == 0x7F` (saturate sentinel,     |
// |           | matches `scalar_at_is_finite` CPU walker)       |
// | F8E5M2    | `((byte >> 2) & 0x1F) == 0x1F` (exp all-ones)   |
//
// Packed dtypes (Int4Packed, Fp4Packed) and integer dtypes are
// handled by the early-return in `Tensor::all_finite` and never
// reach this kernel.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int BLOCK_SIZE = 256;

// Dtype tags. Must agree with the Rust-side `cuda_is_finite` switch.
constexpr int DTYPE_F32     = 0;
constexpr int DTYPE_BF16    = 1;
constexpr int DTYPE_F16     = 2;
constexpr int DTYPE_F8E4M3  = 3;
constexpr int DTYPE_F8E5M2  = 4;

__device__ __forceinline__ bool elem_non_finite_f32(float v) {
    // `isfinite` returns false for NaN and +/-Inf. We want the opposite.
    return !isfinite(v);
}

template <typename T>
__device__ __forceinline__ bool load_and_check(const T* p, int64_t i);

template <>
__device__ __forceinline__ bool load_and_check<float>(const float* p, int64_t i) {
    return elem_non_finite_f32(p[i]);
}

template <>
__device__ __forceinline__ bool load_and_check<__nv_bfloat16>(const __nv_bfloat16* p, int64_t i) {
    return elem_non_finite_f32(__bfloat162float(p[i]));
}

template <>
__device__ __forceinline__ bool load_and_check<__half>(const __half* p, int64_t i) {
    return elem_non_finite_f32(__half2float(p[i]));
}

template <typename T>
__global__ void any_non_finite_kernel(
    const T* __restrict__ x,
    uint32_t* __restrict__ out_flag,
    int64_t n_elements) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < n_elements; i += stride) {
        // Advisory early-exit: every thread peeks at the flag once
        // per strided step before doing the load. Non-atomic on
        // purpose — atomic load would serialize the L2 line.
        if (*out_flag != 0u) {
            return;
        }
        if (load_and_check<T>(x, i)) {
            atomicOr(out_flag, 1u);
            return;
        }
    }
}

// FP8 paths walk the raw byte stream. E4M3FN saturates (no Inf;
// 0x7F / 0xFF are the saturate-to-max sentinels we treat as
// non-finite to match the CPU walker). E5M2 has real Inf/NaN with
// exp == 0b11111.

__global__ void any_non_finite_e4m3_kernel(
    const uint8_t* __restrict__ x,
    uint32_t* __restrict__ out_flag,
    int64_t n_elements) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < n_elements; i += stride) {
        if (*out_flag != 0u) {
            return;
        }
        uint8_t b = x[i];
        if ((b & 0x7Fu) == 0x7Fu) {
            atomicOr(out_flag, 1u);
            return;
        }
    }
}

__global__ void any_non_finite_e5m2_kernel(
    const uint8_t* __restrict__ x,
    uint32_t* __restrict__ out_flag,
    int64_t n_elements) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < n_elements; i += stride) {
        if (*out_flag != 0u) {
            return;
        }
        uint8_t b = x[i];
        if (((b >> 2) & 0x1Fu) == 0x1Fu) {
            atomicOr(out_flag, 1u);
            return;
        }
    }
}

}  // anonymous namespace

extern "C" int kiln_is_finite_storage_async(
    const void* x,
    void* out_flag,
    int64_t n_elements,
    int32_t dtype_tag,
    void* stream_raw) {
    if (n_elements <= 0) {
        return 0;
    }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

    // Cap the grid at a reasonable value so each block does real work.
    // The advisory early-exit makes a huge grid wasteful when a NaN
    // shows up early.
    int64_t blocks_full = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks = (int)((blocks_full > 4096) ? 4096 : blocks_full);
    if (blocks < 1) blocks = 1;

    dim3 grid((unsigned int)blocks);
    dim3 block(BLOCK_SIZE);

    switch (dtype_tag) {
        case DTYPE_F32:
            any_non_finite_kernel<float><<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(x),
                reinterpret_cast<uint32_t*>(out_flag),
                n_elements);
            break;
        case DTYPE_BF16:
            any_non_finite_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<uint32_t*>(out_flag),
                n_elements);
            break;
        case DTYPE_F16:
            any_non_finite_kernel<__half><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x),
                reinterpret_cast<uint32_t*>(out_flag),
                n_elements);
            break;
        case DTYPE_F8E4M3:
            any_non_finite_e4m3_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(x),
                reinterpret_cast<uint32_t*>(out_flag),
                n_elements);
            break;
        case DTYPE_F8E5M2:
            any_non_finite_e5m2_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(x),
                reinterpret_cast<uint32_t*>(out_flag),
                n_elements);
            break;
        default:
            return -2;
    }
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}
