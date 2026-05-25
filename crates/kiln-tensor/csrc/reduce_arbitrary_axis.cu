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

    // Warp reduction
    __shared__ float shared_sum[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    // Cross-warp reduction in warp 0
    if (warp_id == 0) {
        float v = lane < (blk + 31) / 32 ? shared_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        }
        if (lane == 0) {
            out[outer_idx * inner + inner_idx] = cast_from_f32<T>(v * divisor);
        }
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

    int threads = MAX_THREADS;
    while ((int64_t)threads > axis_dim && threads > 32) {
        threads /= 2;
    }
    if (threads < 32) threads = 32;

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

    // Use 0/1 integers in the reduction; ALL uses AND (init=1), ANY
    // uses OR (init=0).
    int local;
    if constexpr (Kind == 0) {
        local = 1;  // ALL: identity = 1
    } else {
        local = 0;  // ANY: identity = 0
    }

    for (int64_t r = tid; r < axis_dim; r += blk) {
        int v = mask[base + r * inner] != 0 ? 1 : 0;
        if constexpr (Kind == 0) {
            local &= v;
        } else {
            local |= v;
        }
    }

    __shared__ int shared_v[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        int other = __shfl_xor_sync(0xFFFFFFFF, local, offset);
        if constexpr (Kind == 0) {
            local &= other;
        } else {
            local |= other;
        }
    }
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) shared_v[warp_id] = local;
    __syncthreads();

    if (warp_id == 0) {
        int v;
        if (lane < (blk + 31) / 32) {
            v = shared_v[lane];
        } else {
            v = (Kind == 0) ? 1 : 0;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            int other = __shfl_xor_sync(0xFFFFFFFF, v, offset);
            if constexpr (Kind == 0) {
                v &= other;
            } else {
                v |= other;
            }
        }
        if (lane == 0) {
            out[outer_idx * inner + inner_idx] = v ? 1 : 0;
        }
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

    int threads = MAX_THREADS;
    while ((int64_t)threads > axis_dim && threads > 32) {
        threads /= 2;
    }
    if (threads < 32) threads = 32;

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

    __shared__ float shared_v[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local, offset);
        if constexpr (Kind == 0) {
            local = fminf(local, other);
        } else {
            local = fmaxf(local, other);
        }
    }
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) shared_v[warp_id] = local;
    __syncthreads();

    if (warp_id == 0) {
        float v;
        if (lane < (blk + 31) / 32) {
            v = shared_v[lane];
        } else {
            v = (Kind == 0) ? INFINITY : -INFINITY;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_xor_sync(0xFFFFFFFF, v, offset);
            if constexpr (Kind == 0) {
                v = fminf(v, other);
            } else {
                v = fmaxf(v, other);
            }
        }
        if (lane == 0) {
            out[outer_idx * inner + inner_idx] = cast_from_f32<T>(v);
        }
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

    int threads = MAX_THREADS;
    while ((int64_t)threads > axis_dim && threads > 32) {
        threads /= 2;
    }
    if (threads < 32) threads = 32;

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
