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

// Warp-level argmax reduction. Pairs (val, idx) reduce together;
// strict `>` on val preserves the lowest-index tie-break.
__device__ inline void warp_argmax_reduce(float& val, int64_t& idx) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        int64_t other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, offset);
        // Strict `>` keeps current pair on ties. To break ties to
        // the lowest index, also accept the other pair when values
        // are equal AND its index is lower.
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
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

    // Warp-level reduction.
    warp_argmax_reduce(local_val, local_idx);

    __shared__ float shared_val[32];
    __shared__ int64_t shared_idx[32];
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) {
        shared_val[warp_id] = local_val;
        shared_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // Cross-warp reduction in warp 0.
    if (warp_id == 0) {
        int n_warps = (blk + 31) / 32;
        float v = lane < n_warps ? shared_val[lane] : -INFINITY;
        int64_t i = lane < n_warps ? shared_idx[lane] : 0;
        warp_argmax_reduce(v, i);
        if (lane == 0) {
            out[row] = i;
        }
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

    int threads = MAX_THREADS;
    while (threads > n_cols && threads > 32) {
        threads /= 2;
    }
    if (threads < 32) threads = 32;
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
