// Phase 4 substrate kernel: scan (cumsum / cumprod) over the last axis.
//
// Implements parallel prefix scan along the trailing axis of a contiguous
// tensor:
//   y[..., i] = REDUCE_{j <= i} op(x[..., j])
// where `op` is + (cumsum) or * (cumprod). Same shape as input.
//
// # Algorithm
//
// Three-phase contiguous-chunk scan, single block per row.
//   - Phase 1 (per-thread sequential prefix): each thread reads a
//     contiguous chunk of `chunk` consecutive columns, accumulates an
//     F32 prefix locally, and writes per-element inclusive prefixes
//     back to its chunk in the output buffer. Its final chunk total
//     goes into shared memory at smem[tid].
//   - Phase 2 (Hillis-Steele inclusive scan over per-thread totals):
//     log2(blockDim.x) steps, in shared memory.
//   - Phase 3 (per-thread patch): each thread (tid > 0) takes the
//     scan's exclusive prefix smem[tid-1] and combines it (in F32) into
//     each element of its chunk in the output buffer.
//
// # Numerical recipe
//
// F32 accumulation regardless of input dtype (matches the kt "F32
// accumulation always" convention). Output cast back to input dtype on
// store. cumsum identity = 0, cumprod identity = 1.
//
// # Determinism
//
// The Hillis-Steele scan has a fixed associativity tree per
// (n_cols, block size). For a given input the output is bit-identical
// across runs.
//
// # Launch shape
//
// One block per row. Block size scales with n_cols up to MAX_THREADS;
// each thread owns a contiguous chunk of size ceil(n_cols / blockDim.x).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int MAX_THREADS = 1024;

// Scan ops. 0 = sum, 1 = prod.
enum ScanKind { SCAN_SUM = 0, SCAN_PROD = 1 };

template <typename T>
__device__ inline float to_f32(T v);
template <>
__device__ inline float to_f32<float>(float v) { return v; }
template <>
__device__ inline float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <>
__device__ inline float to_f32<__half>(__half v) { return __half2float(v); }

template <typename T>
__device__ inline T from_f32(float v);
template <>
__device__ inline float from_f32<float>(float v) { return v; }
template <>
__device__ inline __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>
__device__ inline __half from_f32<__half>(float v) { return __float2half(v); }

__device__ inline float scan_combine(float a, float b, int kind) {
    return kind == SCAN_SUM ? (a + b) : (a * b);
}

__device__ inline float scan_identity(int kind) {
    return kind == SCAN_SUM ? 0.0f : 1.0f;
}

// Three-phase contiguous-chunk scan along one row.
//
// Each thread owns a *contiguous* slice of columns:
//   start = tid * chunk
//   end   = min(start + chunk, n_cols)
// where chunk = ceil(n_cols / blockDim.x). Threads with start >= n_cols
// contribute identity to the per-thread totals.
template <typename T>
__global__ void scan_last_axis_kernel(
    const T* __restrict__ x,
    T* __restrict__ out,
    int64_t n_cols,
    int kind) {
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    int blk = blockDim.x;

    const T* row_in = x + row * n_cols;
    T* row_out = out + row * n_cols;

    extern __shared__ float smem[];  // size = blk floats (per-thread totals).

    float identity = scan_identity(kind);

    // Compute this thread's chunk bounds.
    // chunk = ceil(n_cols / blk).
    int64_t chunk = (n_cols + blk - 1) / blk;
    int64_t start = (int64_t)tid * chunk;
    int64_t end = start + chunk;
    if (end > n_cols) end = n_cols;

    // Phase 1: sequential prefix over this thread's contiguous slice.
    float local_total = identity;
    if (start < n_cols) {
        float acc = identity;
        for (int64_t c = start; c < end; c++) {
            float v = to_f32<T>(row_in[c]);
            acc = scan_combine(acc, v, kind);
            // Stash the partial inclusive prefix (relative to this
            // thread's chunk start) cast back to dtype. We'll patch on
            // phase 3.
            row_out[c] = from_f32<T>(acc);
        }
        local_total = acc;
    }

    // Phase 2: Hillis-Steele inclusive scan over per-thread totals.
    smem[tid] = local_total;
    __syncthreads();

    for (int offset = 1; offset < blk; offset *= 2) {
        float v_self = smem[tid];
        float v_left = (tid >= offset) ? smem[tid - offset] : identity;
        __syncthreads();
        if (tid >= offset) {
            smem[tid] = scan_combine(v_left, v_self, kind);
        }
        __syncthreads();
    }

    // Exclusive prefix = inclusive of (tid-1), or identity for tid==0.
    float exclusive = (tid == 0) ? identity : smem[tid - 1];
    __syncthreads();

    // Phase 3: patch each element of this thread's chunk with the
    // exclusive prefix. For tid==0 the stashed values are already
    // correct.
    if (tid != 0 && start < n_cols) {
        for (int64_t c = start; c < end; c++) {
            float partial = to_f32<T>(row_out[c]);
            float patched = scan_combine(exclusive, partial, kind);
            row_out[c] = from_f32<T>(patched);
        }
    }
}

}  // anonymous namespace

extern "C" int kiln_scan_last_axis_async(
    const void* x,
    void* out,
    int64_t n_rows,
    int64_t n_cols,
    int32_t dtype_tag,  // 0=F32, 1=BF16, 2=F16
    int32_t kind,       // 0=sum, 1=prod
    void* stream_raw) {
    if (n_rows == 0 || n_cols == 0) return 0;
    if (kind != SCAN_SUM && kind != SCAN_PROD) return -3;

    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

    // Block size: power of 2, in [32, MAX_THREADS], chosen so each
    // thread owns at least one element when possible.
    int threads = MAX_THREADS;
    while (threads > n_cols && threads > 32) {
        threads /= 2;
    }
    if (threads < 32) threads = 32;

    dim3 grid((unsigned int)n_rows);
    dim3 block(threads);
    size_t smem_bytes = sizeof(float) * threads;

    switch (dtype_tag) {
        case 0:
            scan_last_axis_kernel<float><<<grid, block, smem_bytes, stream>>>(
                reinterpret_cast<const float*>(x),
                reinterpret_cast<float*>(out),
                n_cols,
                kind);
            break;
        case 1:
            scan_last_axis_kernel<__nv_bfloat16><<<grid, block, smem_bytes, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<__nv_bfloat16*>(out),
                n_cols,
                kind);
            break;
        case 2:
            scan_last_axis_kernel<__half><<<grid, block, smem_bytes, stream>>>(
                reinterpret_cast<const __half*>(x),
                reinterpret_cast<__half*>(out),
                n_cols,
                kind);
            break;
        default:
            return -2;
    }
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}
