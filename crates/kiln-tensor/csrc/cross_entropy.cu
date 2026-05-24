// Phase 4 substrate kernel: cross-entropy loss over per-row softmax.
//
// Replaces the candle path `cross_entropy(logits, targets)` for the
// kt CUDA backend. Mirrors the CPU reference at
// `crates/kiln-tensor/src/ops/cross_entropy.rs` and the Phase 6b
// FLCE backward target.
//
// # Numerical recipe (matches CPU exactly)
//
// For each row `b` of logits `[B, V]`:
//   1. m = max_v logits[b, v]
//   2. log_sum_exp = m + log(sum_v exp(logits[b, v] - m))
//   3. row_loss = log_sum_exp - logits[b, target[b]]
// loss = mean_b row_loss (F32 final divide-by-batch)
//
// All accumulation in F32. Output is a scalar (rank-0) at the input
// logits dtype (cast back on store).
//
// # Determinism
//
// Per-row max + log-sum-exp use a fixed-tree warp + cross-warp
// reduction (same shape as softmax.cu / reduce_last_axis.cu).
// The final batch-mean is a single fixed-order sum + divide, again
// in F32. Output is bit-identical across runs at the same input
// dtype.
//
// # Launch shape
//
// Two-kernel chain:
//   1. `cross_entropy_row_kernel<T>` — one block per row. Threads in
//      block reduce max and log-sum-exp. Writes one F32 per row to a
//      scratch buffer.
//   2. `cross_entropy_finalize_kernel<T>` — one block, n_rows threads.
//      Sums the per-row losses, divides by n_rows, casts to T,
//      writes to the scalar output.

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
__device__ inline T from_f32(float v);
template <>
__device__ inline float from_f32<float>(float v) { return v; }
template <>
__device__ inline __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>
__device__ inline __half from_f32<__half>(float v) { return __float2half(v); }

// Per-row log-sum-exp minus the target logit. Output is F32, one
// element per row.
//
// We parameterize the target index type so callers can pass I64 or
// U32 targets without an extra copy.
template <typename T, typename IDX>
__global__ void cross_entropy_row_kernel(
    const T* __restrict__ logits,
    const IDX* __restrict__ targets,
    float* __restrict__ row_loss,
    int* __restrict__ row_err,  // 0 = ok, 1 = target out of range, 2 = all -inf
    int64_t n_cols) {
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    int blk = blockDim.x;

    const T* row_in = logits + row * n_cols;

    // ----- Pass 1: row max -----
    float local_max = -INFINITY;
    for (int64_t c = tid; c < n_cols; c += blk) {
        float v = to_f32<T>(row_in[c]);
        if (v > local_max) local_max = v;
    }

    __shared__ float shared_max[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = lane < (blk + 31) / 32 ? shared_max[lane] : -INFINITY;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_xor_sync(0xFFFFFFFF, v, offset);
            if (other > v) v = other;
        }
        if (lane == 0) shared_max[0] = v;
    }
    __syncthreads();
    float row_max = shared_max[0];

    // Signal undefined loss if all logits are -inf.
    if (!isfinite(row_max)) {
        if (tid == 0) {
            atomicExch(row_err, 2);
            row_loss[row] = 0.0f;  // benign placeholder
        }
        return;
    }

    // ----- Pass 2: sum exp(x - max) -----
    float local_sum = 0.0f;
    for (int64_t c = tid; c < n_cols; c += blk) {
        float v = to_f32<T>(row_in[c]);
        local_sum += __expf(v - row_max);
    }

    __shared__ float shared_sum[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float v = lane < (blk + 31) / 32 ? shared_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        }
        if (lane == 0) shared_sum[0] = v;
    }
    __syncthreads();

    if (tid == 0) {
        float row_sum = shared_sum[0];
        float log_sum_exp = row_max + __logf(row_sum);

        // Look up the target index and validate it.
        // For I64 targets, reject negatives (CPU does so too).
        long long t = (long long)targets[row];
        if (t < 0 || t >= (long long)n_cols) {
            atomicExch(row_err, 1);
            row_loss[row] = 0.0f;  // benign placeholder
            return;
        }
        float target_logit = to_f32<T>(row_in[t]);
        row_loss[row] = log_sum_exp - target_logit;
    }
}

// Sum per-row losses and divide by n_rows. One block, up to
// MAX_THREADS threads. Output cast back to T.
template <typename T>
__global__ void cross_entropy_finalize_kernel(
    const float* __restrict__ row_loss,
    T* __restrict__ out,
    int64_t n_rows) {
    int tid = threadIdx.x;
    int blk = blockDim.x;

    float local_sum = 0.0f;
    for (int64_t r = tid; r < n_rows; r += blk) {
        local_sum += row_loss[r];
    }

    __shared__ float shared_sum[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float v = lane < (blk + 31) / 32 ? shared_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        }
        if (lane == 0) {
            float mean = v / (float)n_rows;
            out[0] = from_f32<T>(mean);
        }
    }
}

}  // anonymous namespace

// FFI entrypoint.
//
// Parameters
// ----------
//   logits      device ptr, shape [n_rows, n_cols], dtype = `dtype_tag`
//   targets     device ptr, shape [n_rows], dtype = `targets_tag`
//   row_loss_f32 device scratch ptr, shape [n_rows], F32 (caller-allocated)
//   row_err_i32 device scratch ptr, single int32, must be zeroed
//               by the caller. Set to 1 by the kernel on out-of-range
//               target, or 2 on all -inf row.
//   out         device ptr, scalar output, dtype = `dtype_tag`
//   n_rows      batch size
//   n_cols      vocab size
//   dtype_tag   0=F32, 1=BF16, 2=F16  (logits + out)
//   targets_tag 0=I64, 1=U32
//   stream_raw  cudaStream_t cast to void*
//
// Returns
// -------
//   0 on launch success, non-zero on dispatch error. The kernel
//   reports row-level errors via `row_err_i32`; the caller is
//   expected to read it back after sync and surface a Result error.
extern "C" int kiln_cross_entropy_loss_async(
    const void* logits,
    const void* targets,
    void* row_loss_f32,
    void* row_err_i32,
    void* out,
    int64_t n_rows,
    int64_t n_cols,
    int32_t dtype_tag,
    int32_t targets_tag,
    void* stream_raw) {
    if (n_rows == 0 || n_cols == 0) return -3;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

    int threads = MAX_THREADS;
    while (threads > n_cols && threads > 32) {
        threads /= 2;
    }
    if (threads < 32) threads = 32;
    dim3 grid((unsigned int)n_rows);
    dim3 block(threads);

    float* row_loss = reinterpret_cast<float*>(row_loss_f32);
    int* row_err = reinterpret_cast<int*>(row_err_i32);

#define LAUNCH_ROW(T, IDX)                                                                \
    cross_entropy_row_kernel<T, IDX><<<grid, block, 0, stream>>>(                         \
        reinterpret_cast<const T*>(logits),                                               \
        reinterpret_cast<const IDX*>(targets),                                            \
        row_loss,                                                                         \
        row_err,                                                                          \
        n_cols)

    switch (dtype_tag) {
        case 0:  // F32
            if (targets_tag == 0) { LAUNCH_ROW(float, int64_t); }
            else if (targets_tag == 1) { LAUNCH_ROW(float, uint32_t); }
            else return -2;
            break;
        case 1:  // BF16
            if (targets_tag == 0) { LAUNCH_ROW(__nv_bfloat16, int64_t); }
            else if (targets_tag == 1) { LAUNCH_ROW(__nv_bfloat16, uint32_t); }
            else return -2;
            break;
        case 2:  // F16
            if (targets_tag == 0) { LAUNCH_ROW(__half, int64_t); }
            else if (targets_tag == 1) { LAUNCH_ROW(__half, uint32_t); }
            else return -2;
            break;
        default:
            return -2;
    }
#undef LAUNCH_ROW

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    // Finalize: sum + divide-by-N + cast back. One block, up to
    // MAX_THREADS threads (clamped to n_rows).
    int fin_threads = MAX_THREADS;
    while (fin_threads > n_rows && fin_threads > 32) {
        fin_threads /= 2;
    }
    if (fin_threads < 32) fin_threads = 32;
    dim3 fin_grid(1);
    dim3 fin_block(fin_threads);

    switch (dtype_tag) {
        case 0:
            cross_entropy_finalize_kernel<float><<<fin_grid, fin_block, 0, stream>>>(
                row_loss, reinterpret_cast<float*>(out), n_rows);
            break;
        case 1:
            cross_entropy_finalize_kernel<__nv_bfloat16><<<fin_grid, fin_block, 0, stream>>>(
                row_loss, reinterpret_cast<__nv_bfloat16*>(out), n_rows);
            break;
        case 2:
            cross_entropy_finalize_kernel<__half><<<fin_grid, fin_block, 0, stream>>>(
                row_loss, reinterpret_cast<__half*>(out), n_rows);
            break;
        default:
            return -2;
    }
    err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}
