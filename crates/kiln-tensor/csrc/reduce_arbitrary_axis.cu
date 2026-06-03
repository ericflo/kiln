// Issue #1082: sum/mean reduction over an arbitrary axis.
//
// Generalises `reduce_last_axis.cu` to support reducing over any
// single axis of a contiguous row-major tensor. The CPU reference
// in `crates/kiln-tensor/src/ops/reduce.rs::reduce_axis` walks the
// 3-loop structure:
//
//   for o in 0..outer:
//     for i in 0..inner:
//       acc = 0
//       for r in 0..axis_dim:
//         acc += x[(o * axis_dim + r) * inner + i]
//       out[o * inner + i] = acc * divisor
//
// where `outer = product(shape[..axis])`, `inner = product(shape[axis+1..])`.
//
// When `inner == 1` this reduces to the last-axis case; otherwise the
// elements being reduced for a single output sit at stride `inner`,
// not stride 1.
//
// # Launch shape
//
// One block per output element (`outer * inner` total). Block uses
// up to `MAX_THREADS` threads with each thread accumulating a strided
// slice of the reduced dimension; warp-shuffle + shared-memory tree
// reduction within the block. Output is one element per block.
//
// # Determinism
//
// Fixed reduction tree (warp + cross-warp); same input → bit-identical
// output. Same convention as `reduce_last_axis.cu`.
//
// # F32 accumulation
//
// All accumulation in F32 regardless of input dtype, then cast back
// to T on the final store. Matches the CPU reference and the kt
// "F32 accumulation always" rule from rmsnorm/softmax/sum_squared.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

// Wave-size shim (Phase R.5). The two-level reductions below MUST NOT use the
// hardcoded `tid/32` warp / `__shfl_xor_sync(0xFFFFFFFF, ..)` butterfly: on AMD
// wave64 the offset-32 shuffle self-references (lanes 32-63 never participate)
// and the half-filled "warp 0" syncs inactive lanes — a hardware exception
// (HSA 0x1016), verified on real wave64. Routing the cross-block reductions
// through kiln_block_reduce_sum/max (shared-memory tree, no cross-lane ops) is
// the correct, wave32/64-portable fix; on nvcc it is behaviour-identical.
#include "kt_gpu_compat.cuh"

namespace {

constexpr int MAX_THREADS = 1024;

// Smallest power of two in [64, MAX_THREADS] that covers `axis_dim`. Min 64
// (not 32) so every wavefront is FULLY populated on AMD wave64; powers of two
// >= 64 are multiples of both 32 and 64. kiln_block_reduce_* require a
// power-of-two blockDim, which this guarantees.
inline int reduce_block_threads(int64_t axis_dim) {
    int threads = 64;
    while ((int64_t)threads < axis_dim && threads < MAX_THREADS) {
        threads *= 2;
    }
    return threads;
}

template <typename T>
__device__ inline float to_f32(T v);
template <>
__device__ inline float to_f32<float>(float v) { return v; }
template <>
__device__ inline float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <>
__device__ inline float to_f32<__half>(__half v) { return __half2float(v); }

template <typename T>
__device__ inline T cast_from_f32(float v);
template <>
__device__ inline float cast_from_f32<float>(float v) { return v; }
template <>
__device__ inline __nv_bfloat16 cast_from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>
__device__ inline __half cast_from_f32<__half>(float v) { return __float2half(v); }

// One block per (outer, inner) pair, mapped via grid.y = outer,
// grid.x = inner. Each block walks `axis_dim` elements at stride `inner`.
template <typename T>
__global__ void reduce_arbitrary_axis_sum_kernel(
    const T* __restrict__ x,
    T* __restrict__ out,
    int64_t outer,
    int64_t axis_dim,
    int64_t inner,
    float divisor) {
    int64_t inner_idx = blockIdx.x;
    int64_t outer_idx = blockIdx.y;
    if (inner_idx >= inner || outer_idx >= outer) return;

    int tid = threadIdx.x;
    int blk = blockDim.x;

    // Base offset: (outer_idx * axis_dim + 0) * inner + inner_idx
    // Stride between successive reduced elements: `inner`.
    int64_t base = outer_idx * axis_dim * inner + inner_idx;

    float local_sum = 0.0f;
    for (int64_t r = tid; r < axis_dim; r += blk) {
        local_sum += to_f32<T>(x[base + r * inner]);
    }

    // Wave-size-agnostic block reduction via shared memory (no cross-lane ops
    // — correct on AMD wave32/wave64 and NVIDIA). blockDim is a power of two
    // in [64, MAX_THREADS] (see reduce_block_threads).
    __shared__ float smem[MAX_THREADS];
    float v = kiln_block_reduce_sum(local_sum, smem);
    if (tid == 0) {
        out[outer_idx * inner + inner_idx] = cast_from_f32<T>(v * divisor);
    }
}

}  // anonymous namespace

extern "C" int kiln_sum_arbitrary_axis_async(
    const void* x,
    void* out,
    int64_t outer,
    int64_t axis_dim,
    int64_t inner,
    float divisor,
    int32_t dtype_tag,  // 0=F32, 1=BF16, 2=F16
    void* stream_raw) {
    if (outer == 0 || axis_dim == 0 || inner == 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

    // Power-of-two block in [64, MAX_THREADS] so every wavefront is full on
    // wave64 and kiln_block_reduce_* (power-of-two requirement) is satisfied.
    int threads = reduce_block_threads(axis_dim);

    // grid.x = inner, grid.y = outer. Bounded by CUDA limits (2^31 - 1
    // each dimension; for kt's typical shapes this is fine).
    dim3 grid((unsigned int)inner, (unsigned int)outer);
    dim3 block(threads);

    switch (dtype_tag) {
        case 0:
            reduce_arbitrary_axis_sum_kernel<float><<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(x),
                reinterpret_cast<float*>(out),
                outer, axis_dim, inner, divisor);
            break;
        case 1:
            reduce_arbitrary_axis_sum_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<__nv_bfloat16*>(out),
                outer, axis_dim, inner, divisor);
            break;
        case 2:
            reduce_arbitrary_axis_sum_kernel<__half><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x),
                reinterpret_cast<__half*>(out),
                outer, axis_dim, inner, divisor);
            break;
        default:
            return -2;
    }
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}

// ----------------------------------------------------------------------
// bool_reduce (all/any) over an arbitrary axis, U8 mask in/out.
//
// `kind == 0` -> ALL (AND), `kind == 1` -> ANY (OR).
//
// Each block produces one output element using a fixed-tree warp +
// shared-memory reduction. Within a thread we walk a strided slice
// of the axis (stride = inner).

namespace {

template <int Kind>
__global__ void bool_reduce_arbitrary_axis_kernel(
    const unsigned char* __restrict__ mask,
    unsigned char* __restrict__ out,
    int64_t outer,
    int64_t axis_dim,
    int64_t inner) {
    int64_t inner_idx = blockIdx.x;
    int64_t outer_idx = blockIdx.y;
    if (inner_idx >= inner || outer_idx >= outer) return;

    int tid = threadIdx.x;
    int blk = blockDim.x;

    int64_t base = outer_idx * axis_dim * inner + inner_idx;

    // Map AND/OR over 0/1 values onto a single wave-size-agnostic block-MAX
    // (kiln_block_reduce_max; no cross-lane ops). The boolean reductions are:
    //   ANY (OR)  -> max(local)        == 1  iff any element is true
    //   ALL (AND) -> min(local)        == 1  iff every element is true
    // and min(local) = -max(-local), so we feed `-local` for the ALL case.
    // Threads past axis_dim contribute the reduction identity:
    //   ANY: 0   (won't raise the max)
    //   ALL: 1   (won't lower the min)  -> fed as -1 to the max.
    float local;
    if constexpr (Kind == 0) {
        local = -1.0f;  // ALL identity 1, negated for the max-of-negatives
    } else {
        local = 0.0f;   // ANY identity 0
    }
    for (int64_t r = tid; r < axis_dim; r += blk) {
        int b = mask[base + r * inner] != 0 ? 1 : 0;
        if constexpr (Kind == 0) {
            float nv = (float)(-b);
            local = nv > local ? nv : local;  // max of -b == -min(b)
        } else {
            float v = (float)b;
            local = v > local ? v : local;     // max of b
        }
    }

    __shared__ float smem[MAX_THREADS];
    float r = kiln_block_reduce_max(local, smem);
    if (tid == 0) {
        unsigned char res;
        if constexpr (Kind == 0) {
            // ALL: min(b) == -r; true iff min == 1.
            res = (-r >= 0.5f) ? 1 : 0;
        } else {
            // ANY: max(b) == r; true iff max == 1.
            res = (r >= 0.5f) ? 1 : 0;
        }
        out[outer_idx * inner + inner_idx] = res;
    }
}

}  // anonymous namespace

extern "C" int kiln_bool_reduce_arbitrary_axis_async(
    const void* mask,
    void* out,
    int64_t outer,
    int64_t axis_dim,
    int64_t inner,
    int32_t kind,  // 0=ALL, 1=ANY
    void* stream_raw) {
    if (outer == 0 || axis_dim == 0 || inner == 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

    // Power-of-two block in [64, MAX_THREADS] so every wavefront is full on
    // wave64 and kiln_block_reduce_* (power-of-two requirement) is satisfied.
    int threads = reduce_block_threads(axis_dim);

    dim3 grid((unsigned int)inner, (unsigned int)outer);
    dim3 block(threads);

    const unsigned char* x_ptr = reinterpret_cast<const unsigned char*>(mask);
    unsigned char* o_ptr = reinterpret_cast<unsigned char*>(out);

    if (kind == 0) {
        bool_reduce_arbitrary_axis_kernel<0><<<grid, block, 0, stream>>>(
            x_ptr, o_ptr, outer, axis_dim, inner);
    } else if (kind == 1) {
        bool_reduce_arbitrary_axis_kernel<1><<<grid, block, 0, stream>>>(
            x_ptr, o_ptr, outer, axis_dim, inner);
    } else {
        return -2;
    }
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}

// ----------------------------------------------------------------------
// minmax reduction over an arbitrary axis (issue #1082).
//
// `kind == 0` -> MIN, `kind == 1` -> MAX. F32 accumulation throughout
// (same numerical-reference convention as the sum/mean path); cast
// back to T on the final store.
//
// The reduction tree is fixed (warp-shuffle + cross-warp via shared
// memory), so bit-identical determinism is preserved for any given
// (shape, axis, dtype) tuple. Identities are +INF for MIN and -INF
// for MAX so empty thread tiles produce no contamination.

namespace {

template <typename T, int Kind>
__global__ void reduce_arbitrary_axis_minmax_kernel(
    const T* __restrict__ x,
    T* __restrict__ out,
    int64_t outer,
    int64_t axis_dim,
    int64_t inner) {
    int64_t inner_idx = blockIdx.x;
    int64_t outer_idx = blockIdx.y;
    if (inner_idx >= inner || outer_idx >= outer) return;

    int tid = threadIdx.x;
    int blk = blockDim.x;

    int64_t base = outer_idx * axis_dim * inner + inner_idx;

    // Identities: +INF for MIN, -INF for MAX.
    float local;
    if constexpr (Kind == 0) {
        local = INFINITY;
    } else {
        local = -INFINITY;
    }
    for (int64_t r = tid; r < axis_dim; r += blk) {
        float v = to_f32<T>(x[base + r * inner]);
        if constexpr (Kind == 0) {
            // fminf propagates non-NaN over NaN — matches Rust's
            // `f32::min` which returns the non-NaN operand when one
            // side is NaN. CPU reference walks `cur.min(v)`.
            local = fminf(local, v);
        } else {
            local = fmaxf(local, v);
        }
    }

    // Wave-size-agnostic block reduction via shared memory. kiln_block_reduce_max
    // computes the block MAX; MIN is derived as -max(-local). Threads past
    // axis_dim already hold the correct identity (+INF for MIN -> -INF when
    // negated; -INF for MAX), so they never contaminate the result.
    __shared__ float smem[MAX_THREADS];
    float v;
    if constexpr (Kind == 0) {
        v = -kiln_block_reduce_max(-local, smem);  // MIN = -max(-x)
    } else {
        v = kiln_block_reduce_max(local, smem);    // MAX
    }
    if (tid == 0) {
        out[outer_idx * inner + inner_idx] = cast_from_f32<T>(v);
    }
}

}  // anonymous namespace

extern "C" int kiln_minmax_arbitrary_axis_async(
    const void* x,
    void* out,
    int64_t outer,
    int64_t axis_dim,
    int64_t inner,
    int32_t kind,        // 0=MIN, 1=MAX
    int32_t dtype_tag,   // 0=F32, 1=BF16, 2=F16
    void* stream_raw) {
    if (outer == 0 || axis_dim == 0 || inner == 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

    // Power-of-two block in [64, MAX_THREADS] so every wavefront is full on
    // wave64 and kiln_block_reduce_* (power-of-two requirement) is satisfied.
    int threads = reduce_block_threads(axis_dim);

    dim3 grid((unsigned int)inner, (unsigned int)outer);
    dim3 block(threads);

    if (kind == 0) {
        switch (dtype_tag) {
            case 0:
                reduce_arbitrary_axis_minmax_kernel<float, 0><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(x),
                    reinterpret_cast<float*>(out),
                    outer, axis_dim, inner);
                break;
            case 1:
                reduce_arbitrary_axis_minmax_kernel<__nv_bfloat16, 0><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(x),
                    reinterpret_cast<__nv_bfloat16*>(out),
                    outer, axis_dim, inner);
                break;
            case 2:
                reduce_arbitrary_axis_minmax_kernel<__half, 0><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(x),
                    reinterpret_cast<__half*>(out),
                    outer, axis_dim, inner);
                break;
            default:
                return -2;
        }
    } else if (kind == 1) {
        switch (dtype_tag) {
            case 0:
                reduce_arbitrary_axis_minmax_kernel<float, 1><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(x),
                    reinterpret_cast<float*>(out),
                    outer, axis_dim, inner);
                break;
            case 1:
                reduce_arbitrary_axis_minmax_kernel<__nv_bfloat16, 1><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(x),
                    reinterpret_cast<__nv_bfloat16*>(out),
                    outer, axis_dim, inner);
                break;
            case 2:
                reduce_arbitrary_axis_minmax_kernel<__half, 1><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(x),
                    reinterpret_cast<__half*>(out),
                    outer, axis_dim, inner);
                break;
            default:
                return -2;
        }
    } else {
        return -3;
    }

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}
