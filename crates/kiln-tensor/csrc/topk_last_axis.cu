// #1082 perf-fix H9: on-device top-k over the trailing axis of a tensor.
//
// Replaces the pre-flip candle path `Tensor::sort_last_dim(desc=true)`
// followed by a slice of the top `k` `(value, index)` pairs. The whole
// point of this kernel is to keep the full `[V]` logits row resident on
// the device and transfer ONLY the `k` selected `(value, index)` pairs
// back to host — for Qwen3.5-4B decode that is `k=20` floats + `k=20`
// i64 indices (~240 bytes) instead of a `V=248320` f32 DtoH (~970 KB)
// EVERY decoded token.
//
// # Numerical recipe (per row)
//
// Iterative selection: k passes, each pass finds the largest element
// that ranks strictly below the previously-selected element under the
// total order
//
//     a > b   iff   val(a) > val(b)
//                   OR (val(a) == val(b) AND idx(a) < idx(b))
//
// i.e. descending value, ties broken by LOWEST index first. This matches
// the host fallback `topk_via_host_sort` in
// `crates/kiln-model/src/sampling.rs`, whose min-heap tie-break is
// `o.1.cmp(&self.1)` (lower index wins on equal value) and whose full
// sort is descending by value. Selecting the strictly-next-ranked
// element each pass means an element is never picked twice even when
// several columns share the same value.
//
//   selected = (+inf, -1)   // sentinel "above everything"
//   for p in 0..k:
//       best = (-inf, +inf)
//       for c in 0..n_cols:
//           cand = (x[c], c)
//           if rank(cand) < rank(selected)        // strictly below prev
//              and rank(cand) > rank(best):       // best so far this pass
//               best = cand
//       out_vals[p]    = best.val
//       out_indices[p] = best.idx
//       selected = best
//
// All comparison happens in F32 regardless of input dtype (matches the
// kt "F32 accumulation always" convention). `-inf`/`NaN` columns sort to
// the bottom: NaN never compares `>` anything, so it is only ever picked
// once the finite candidates are exhausted, and `topk_via_host_sort`
// likewise pushes non-finite weights out via its later
// `w.is_finite() && w > 0` softmax filter.
//
// # Output
//
// `out_vals`    : [n_rows, k] F32   (selected values, descending)
// `out_indices` : [n_rows, k] I64   (selected column indices)
//
// `k` is assumed `<= n_cols`; the Rust dispatcher clamps it. If a row has
// fewer than `k` finite-or-otherwise distinct ranks, the trailing slots
// are filled with the smallest remaining element / its index (the loop
// always assigns SOME column once `c` is scanned), so the output is fully
// defined.
//
// # Determinism
//
// One block per row; the per-pass argmax reduction order is fixed
// (warp-shuffle + shared-memory across warps) and ties resolve to the
// smaller index, so the output is bit-identical across runs for a given
// input. k is tiny (≤1024) so the k-pass outer loop is cheap.
//
// # Launch shape
//
// One block per row. Each block uses up to MAX_THREADS threads; rows
// wider than the block use a strided per-thread scan. Mirrors the launch
// shape of `argmax_last_axis.cu` and `softmax.cu`.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

// Wave-size-portable shuffle (`kiln_shfl_xor`): on HIP it uses the native
// `__shfl_xor(v, offset, warpSize)` (explicit width — correct on wave32/64 and,
// unlike the bare `__shfl_xor_sync(0xFFFFFFFF,...)`, it both compiles under
// ROCm 7.x's 64-bit-mask static_assert and reduces the right lanes); on CUDA it
// is the masked `_sync` form. The reduction below caps the xor offset at 16, so
// it never crosses 32 lanes via shuffle (it crosses 32-lane subgroups through
// shared memory), which is exactly the regime `kiln_shfl_xor` is safe in on
// RDNA wave64.
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

// Returns true iff candidate (cv, ci) ranks strictly ABOVE the current
// best (bv, bi) under: descending value, ties broken by LOWER index.
// NaN handling: any comparison with NaN is false, so a NaN candidate
// never displaces a finite best and a finite candidate never displaces a
// NaN best via the value test — but the explicit `cv == bv` tie path is
// also false for NaN, so NaNs only win when `bv` itself is -inf-sentinel
// territory (i.e. nothing finite remains), matching argmax semantics.
__device__ inline bool rank_gt(float cv, int64_t ci, float bv, int64_t bi) {
    return cv > bv || (cv == bv && ci < bi);
}

// Returns true iff (cv, ci) ranks strictly BELOW the selected frontier
// (sv, si) — i.e. it is a legal pick for the *next* slot.
__device__ inline bool rank_lt(float cv, int64_t ci, float sv, int64_t si) {
    return cv < sv || (cv == sv && ci > si);
}

// Warp-level argmax reduction over (val, idx) pairs under rank_gt.
__device__ inline void warp_rank_reduce(float& val, int64_t& idx) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = kiln_shfl_xor(val, offset);
        int64_t other_idx = kiln_shfl_xor(idx, offset);
        if (rank_gt(other_val, other_idx, val, idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}

template <typename T>
__global__ void topk_last_axis_kernel(
    const T* __restrict__ x,
    float* __restrict__ out_vals,
    int64_t* __restrict__ out_indices,
    int64_t n_cols,
    int k) {
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    int blk = blockDim.x;

    const T* row_in = x + row * n_cols;
    float* row_vals = out_vals + row * k;
    int64_t* row_idx = out_indices + row * k;

    __shared__ float shared_val[32];
    __shared__ int64_t shared_idx[32];
    // Frontier = previously selected (value, index). Starts "above
    // everything" so the first pass picks the global max.
    __shared__ float sel_val;
    __shared__ int64_t sel_idx;
    if (tid == 0) {
        sel_val = INFINITY;
        sel_idx = -1;
    }
    __syncthreads();

    int warp_id = tid / 32;
    int lane = tid & 31;
    int n_warps = (blk + 31) / 32;

    for (int p = 0; p < k; ++p) {
        float cur_sel_val = sel_val;
        int64_t cur_sel_idx = sel_idx;

        // Per-thread strided scan: best candidate strictly below the
        // frontier under rank order.
        float local_val = -INFINITY;
        int64_t local_idx = (int64_t)n_cols;  // sentinel "worst index"
        for (int64_t c = tid; c < n_cols; c += blk) {
            float v = to_f32<T>(row_in[c]);
            if (rank_lt(v, c, cur_sel_val, cur_sel_idx) &&
                rank_gt(v, c, local_val, local_idx)) {
                local_val = v;
                local_idx = c;
            }
        }

        // Warp-level reduction.
        warp_rank_reduce(local_val, local_idx);
        if (lane == 0) {
            shared_val[warp_id] = local_val;
            shared_idx[warp_id] = local_idx;
        }
        __syncthreads();

        // Cross-warp reduction in warp 0, then publish the pick.
        if (warp_id == 0) {
            float v = lane < n_warps ? shared_val[lane] : -INFINITY;
            int64_t i = lane < n_warps ? shared_idx[lane] : (int64_t)n_cols;
            warp_rank_reduce(v, i);
            if (lane == 0) {
                row_vals[p] = v;
                row_idx[p] = i;
                sel_val = v;
                sel_idx = i;
            }
        }
        __syncthreads();
    }
}

}  // anonymous namespace

extern "C" int kiln_topk_last_axis_async(
    const void* x,
    void* out_vals,      // float[n_rows * k]
    void* out_indices,   // int64_t[n_rows * k]
    int64_t n_rows,
    int64_t n_cols,
    int32_t k,
    int32_t dtype_tag,   // 0=F32, 1=BF16, 2=F16
    void* stream_raw) {
    if (n_rows == 0 || n_cols == 0 || k <= 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

    int threads = MAX_THREADS;
    while (threads > n_cols && threads > 32) {
        threads /= 2;
    }
    if (threads < 32) threads = 32;
    dim3 grid((unsigned int)n_rows);
    dim3 block(threads);

    float* vals_ptr = reinterpret_cast<float*>(out_vals);
    int64_t* idx_ptr = reinterpret_cast<int64_t*>(out_indices);

    switch (dtype_tag) {
        case 0:
            topk_last_axis_kernel<float><<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(x), vals_ptr, idx_ptr, n_cols, (int)k);
            break;
        case 1:
            topk_last_axis_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x), vals_ptr, idx_ptr, n_cols, (int)k);
            break;
        case 2:
            topk_last_axis_kernel<__half><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x), vals_ptr, idx_ptr, n_cols, (int)k);
            break;
        default:
            return -2;
    }
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}
