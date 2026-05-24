// Kiln CUDA-side rotary position embedding (RoPE) kernel.
//
// Semantics (matches `RopeOp::cpu_fwd` in `ops/rope.rs`):
//
// Given:
//   x: [..., seq, head_dim], dtype F32/BF16/F16
//   cos: [seq, rotary_dim/2], dtype F32/BF16/F16
//   sin: [seq, rotary_dim/2], dtype F32/BF16/F16
//
// For each leading row l in 0..leading, each seq position s in 0..seq,
// each pair i in 0..(rotary_dim/2):
//
//   x_new[..., s, 2i]   = x[..., s, 2i]   * cos[s, i] - x[..., s, 2i+1] * sin[s, i]
//   x_new[..., s, 2i+1] = x[..., s, 2i]   * sin[s, i] + x[..., s, 2i+1] * cos[s, i]
//
// Indices beyond rotary_dim within head_dim are passed through unchanged.
// This supports the partial-rotary case (Qwen3.5-4B: rotary_dim=64 of head_dim=256).
//
// # Layout
//
// Inputs are contiguous. Total elements = leading * seq * head_dim.
// One thread per (l, s, pair_i) for the rotated portion; pass-through tail
// is written by a separate small kernel (or handled by initial copy).
//
// We compute the leading-row * seq * pair_count grid; pass-through tail is
// handled with a cudaMemcpyAsync of the full input first, then the kernel
// overwrites the rotated region.

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

// Promote any of our 3 dtypes to f32 from a raw byte pointer + element index.
__device__ __forceinline__ float load_promoted(const void* p, int64_t idx, int dtype) {
    switch (dtype) {
        case DTYPE_F32:
            return reinterpret_cast<const float*>(p)[idx];
        case DTYPE_BF16:
            return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[idx]);
        case DTYPE_F16:
            return __half2float(reinterpret_cast<const __half*>(p)[idx]);
        default:
            return 0.0f;
    }
}

__device__ __forceinline__ void store_narrowed(void* p, int64_t idx, int dtype, float v) {
    switch (dtype) {
        case DTYPE_F32:
            reinterpret_cast<float*>(p)[idx] = v;
            break;
        case DTYPE_BF16:
            reinterpret_cast<__nv_bfloat16*>(p)[idx] = __float2bfloat16(v);
            break;
        case DTYPE_F16:
            reinterpret_cast<__half*>(p)[idx] = __float2half(v);
            break;
        default:
            break;
    }
}

// One thread per (l, s, d) covering EVERY element of x.
// Total threads = leading * seq * head_dim.
//
// For d < rotary_dim (the rotated region): write rotated value using cos/sin.
// For d >= rotary_dim (the pass-through tail): copy input value through.
//
// The kernel always reads each output position once and writes it once;
// pair members (2i, 2i+1) within the rotated region both read both inputs
// of the pair before writing.
//
// x_dtype: tag for x (in and out share dtype).
// cs_dtype: tag for cos / sin (they share dtype).
__global__ void kiln_rope_kernel(const void* __restrict__ x_in,
                                 void* __restrict__ x_out,
                                 const void* __restrict__ cos,
                                 const void* __restrict__ sin,
                                 int64_t leading,
                                 int64_t seq,
                                 int64_t head_dim,
                                 int64_t pair_count,
                                 int x_dtype,
                                 int cs_dtype) {
    int64_t global_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = leading * seq * head_dim;
    if (global_idx >= total) return;

    // Decompose: global_idx = ((l * seq) + s) * head_dim + d
    int64_t d = global_idx % head_dim;
    int64_t ls = global_idx / head_dim;
    int64_t s = ls % seq;
    int64_t l = ls / seq;

    int64_t rotary_dim = pair_count * 2;
    int64_t base = ((l * seq) + s) * head_dim;

    if (d >= rotary_dim) {
        // Pass-through tail.
        float v = load_promoted(x_in, base + d, x_dtype);
        store_narrowed(x_out, base + d, x_dtype, v);
        return;
    }

    int64_t i = d / 2;       // pair index
    int64_t even = base + 2 * i;
    int64_t odd = base + 2 * i + 1;
    int64_t cs_idx = s * pair_count + i;

    float a = load_promoted(x_in, even, x_dtype);
    float b = load_promoted(x_in, odd, x_dtype);
    float c = load_promoted(cos, cs_idx, cs_dtype);
    float si = load_promoted(sin, cs_idx, cs_dtype);

    if (d == 2 * i) {
        // Even slot.
        float new_a = a * c - b * si;
        store_narrowed(x_out, even, x_dtype, new_a);
    } else {
        // Odd slot (d == 2 * i + 1).
        float new_b = a * si + b * c;
        store_narrowed(x_out, odd, x_dtype, new_b);
    }
}

} // namespace

// Entry point: launches the rope kernel on `stream`. Writes the full
// `leading * seq * head_dim` output: rotated values for the first
// `rotary_dim` of each row's head_dim, and a pass-through copy for the
// tail.
//
// Returns 0 on success, non-zero on launch error.
extern "C" int kiln_rope_async(const void* x_in,
                               void* x_out,
                               const void* cos,
                               const void* sin,
                               int64_t leading,
                               int64_t seq,
                               int64_t head_dim,
                               int64_t pair_count,
                               int x_dtype,
                               int cs_dtype,
                               cudaStream_t stream) {
    if (leading <= 0 || seq <= 0 || head_dim <= 0 || pair_count <= 0) return 1;
    if (pair_count * 2 > head_dim) return 2;
    if (x_dtype < 0 || x_dtype > 2) return 3;
    if (cs_dtype < 0 || cs_dtype > 2) return 4;

    int64_t total = leading * seq * head_dim;
    if (total == 0) return 0;

    int64_t blocks_i64 = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 5;
    int blocks = static_cast<int>(blocks_i64);

    kiln_rope_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        x_in, x_out, cos, sin,
        leading, seq, head_dim, pair_count,
        x_dtype, cs_dtype);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
