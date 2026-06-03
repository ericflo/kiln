// Phase 4 substrate kernel: per-row argmax over the trailing axis of a
// tensor.
//
// Replaces the candle path `Tensor::argmax_keepdim(-1)` for the kt
// CUDA backend. Single-pass reduction that tracks both the max value
// and its index; output dtype is I64 (matches the CPU reference at
// `crates/kiln-tensor/src/ops/argmax.rs`).
//
// # Numerical recipe
//
// For each row of `n_cols` elements:
//   (best_val, best_idx) = (-inf, 0)
//   for c in 0..n_cols:
//       if x[c] > best_val:
//           best_val = x[c]
//           best_idx = c
//   out[row] = best_idx
//
// Ties are broken by **lowest index** — same convention as
// `slice.iter().enumerate().max_by(...)` in standard Rust and
// `candle_core::Tensor::argmax`. The strict `>` comparison preserves
// this: equal values never displace the current best, so the
// lowest-seen index wins.
//
// All comparison happens in F32 regardless of input dtype (matches
// the kt "F32 accumulation always" convention). Output is I64.
//
// # Determinism
//
// The reduction order is fixed (warp-shuffle + shared-memory across
// warps), and ties resolve to the smaller index, so the output is
// bit-identical across runs for a given input.
//
// # Launch shape
//
// One block per row. Each block uses `n_cols` threads up to a max of
// 1024; rows with more cols use a strided per-thread scan. Mirrors
// the launch shape of `softmax.cu` and `reduce_last_axis.cu`.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

// Wave-size shim (Phase R.5). The original two-level reduction used
// `__shfl_xor_sync(0xFFFFFFFF, ..., 16)` + a hardcoded `tid/32` warp_id, which
// is BROKEN on AMD wave64: a 32-bit full-mask drops lanes 32-63, an offset of
// 16 only butterflies within a half-wave, and "warp 0" (`tid/32==0`) is just
// half a wavefront — so the cross-warp shuffle syncs inactive lanes 32-63 (an
// HSA hardware exception, verified on real gfx1151 wave64). argmax reduces a
// PAIR (val, idx) with a lowest-index tie-break, which `kiln_block_reduce_*`
// (single-value) cannot express, so we hand-roll a wave-size-AGNOSTIC paired
// shared-memory tree reduction: no cross-lane ops, correct on NVIDIA + AMD
// wave32 + AMD wave64. The launcher picks a power-of-two blockDim in
// [64, MAX_THREADS] so the tree reduction is exact.
#include "kt_gpu_compat.cuh"

namespace {

constexpr int MAX_THREADS = 1024;

template <typename T>
__device__ inline float to_f32(T v);
template <>
__device__ inline float to_f32<float>(float v) { return v; }
template <>
__device__ inline float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <>
__device__ inline float to_f32<__half>(__half v) { return __half2float(v); }

// Wave-size-AGNOSTIC block-level argmax reduction over a (val, idx) pair via
// shared memory — no cross-lane shuffles, so it is correct on NVIDIA, AMD
// wave32, AND AMD wave64. `blockDim.x` MUST be a power of two (the launcher
// guarantees this). The pair reduces with a strict `>` on val plus a
// lower-index tie-break, matching the CPU reference's lowest-index convention.
// Returns the winning index in thread 0 (other threads' return value is
// unspecified). A trailing __syncthreads() makes the smem buffers reusable.
__device__ inline int64_t block_argmax_reduce(
    float val, int64_t idx, float* smem_val, int64_t* smem_idx) {
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    smem_val[tid] = val;
    smem_idx[tid] = idx;
    __syncthreads();
    for (int s = blk >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = smem_val[tid + s];
            int64_t oi = smem_idx[tid + s];
            float cv = smem_val[tid];
            int64_t ci = smem_idx[tid];
            // Accept the other pair if its value is strictly greater, or
            // equal with a lower index (lowest-index tie-break).
            if (ov > cv || (ov == cv && oi < ci)) {
                smem_val[tid] = ov;
                smem_idx[tid] = oi;
            }
        }
        __syncthreads();
    }
    int64_t result = smem_idx[0];
    __syncthreads();
    return result;
}

template <typename T>
__global__ void argmax_last_axis_kernel(
    const T* __restrict__ x,
    int64_t* __restrict__ out,
    int64_t n_cols) {
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    int blk = blockDim.x;

    const T* row_in = x + row * n_cols;

    // Per-thread strided scan over the row. Tie-break: keep the
    // lower index when values are equal (strict `>` on update).
    float local_val = -INFINITY;
    int64_t local_idx = 0;
    for (int64_t c = tid; c < n_cols; c += blk) {
        float v = to_f32<T>(row_in[c]);
        if (v > local_val) {
            local_val = v;
            local_idx = c;
        }
        // Equal: keep local_idx (lower-c-first scan would not help
        // because c is strictly increasing per thread, so this is
        // already lowest-index for this thread).
    }

    // Wave-size-agnostic block reduction over the (val, idx) pair.
    __shared__ float smem_val[MAX_THREADS];
    __shared__ int64_t smem_idx[MAX_THREADS];
    int64_t best_idx = block_argmax_reduce(local_val, local_idx, smem_val, smem_idx);

    if (tid == 0) {
        out[row] = best_idx;
    }
}

}  // anonymous namespace

extern "C" int kiln_argmax_last_axis_async(
    const void* x,
    void* out,
    int64_t n_rows,
    int64_t n_cols,
    int32_t dtype_tag,  // 0=F32, 1=BF16, 2=F16
    void* stream_raw) {
    if (n_rows == 0 || n_cols == 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

    // Block size: smallest power of two that covers n_cols, clamped to
    // [64, MAX_THREADS]. Min 64 (not 32) so every wavefront is FULLY populated
    // on AMD wave64, and a power of two so the shared-memory tree reduction in
    // block_argmax_reduce is exact. Threads past n_cols contribute the
    // reduction identity (-INFINITY, idx 0) via the strided scan loop.
    int threads = 64;
    while (threads < n_cols && threads < MAX_THREADS) {
        threads *= 2;
    }
    dim3 grid((unsigned int)n_rows);
    dim3 block(threads);

    int64_t* out_ptr = reinterpret_cast<int64_t*>(out);

    switch (dtype_tag) {
        case 0:
            argmax_last_axis_kernel<float><<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(x),
                out_ptr,
                n_cols);
            break;
        case 1:
            argmax_last_axis_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x),
                out_ptr,
                n_cols);
            break;
        case 2:
            argmax_last_axis_kernel<__half><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x),
                out_ptr,
                n_cols);
            break;
        default:
            return -2;
    }
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}
