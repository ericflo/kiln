// Fused OPD top-K reverse-KL CUDA backward kernel.
//
// See `docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md`
// §9.2 for the design. One launch writes active-row d_hidden entries into
// a zero-filled full-sequence `[T, H]` output for the per-token reverse KL
// between the student's distribution (gathered from `hidden @ head_t`) and
// the teacher's top-K distribution, both renormalised over the K support.
//
// # Memory & arithmetic intensity
//
// For T_active tokens × K=32 support × H=2560 hidden:
//   * Hidden traffic: T_active × H × 2 B (bf16) = ~20 MiB at T=4096.
//   * Head_t traffic: T_active × K × H × 2 B = ~670 MiB scattered reads
//     (the K columns of head_t are not contiguous; we read by index).
//   * Output traffic: T_active × H × 2 B = ~20 MiB.
//
// This kernel is *bandwidth-bound* — the scattered head_t reads dominate.
// On A6000 (768 GB/s HBM), the lower bound is ~1.8 ms for T=4096.
// cuBLAS can't match because we don't have a dense [T, K] gather of
// head_t built upfront; constructing it would cost an extra T*K*H read
// before the matmul. The fused kernel reads each head_t element once.
//
// # Block / thread layout
//
// One CUDA block per active token. Block size = K × WARP_SIZE so each
// of the K logits is computed by exactly one warp via a cooperative
// dot product across H. The block-level reductions (max for log_softmax
// + sum-exp + KL accumulation) run via shfl_xor over K-many warps,
// driven by warp 0.
//
// Supports K in {16, 32}. K=32 is the §6 default and the fast path.
// Other K values fall back to the Phase A candle path on the Rust side.
//
// # Numerics
//
// All accumulations are f32. bf16 hidden / head_t are loaded and cast
// to float on the fly. d_hidden is written in the input dtype. Parity
// vs the candle CPU reference is enforced at ≤1e-4 abs at f32 and ≤5e-2
// at bf16 by the kt-bwd parity tests in `kt_api.rs` / `kt_tape.rs`.
//
// # History
//
// (#1082, 2026-05-28) The fused forward kernel
// `opd_topk_kl_fwd_kernel` + `kiln_opd_topk_kl_fwd_{bf16,f32}` entries
// and the distribution-alignment metrics kernel
// `opd_topk_metrics_kernel` + `kiln_opd_topk_metrics_{bf16,f32}` entries
// were removed. Production now runs the forward as a kt-typed
// composite (`crate::kt_api::per_position_forward_kt`) on CUDA storage,
// and the metrics path had no live caller. See
// `docs/archive/candle-removal/opd-loss-kernel-candle-removal-stop-2026-05-28.md` for the
// audit; the deletion drops 1 fused FWD kernel + 1 metrics kernel
// (~290 LOC) while keeping the kt-shim and kt-tape paths bit-identical
// on the backward via the surviving `kiln_opd_topk_kl_bwd_*` symbols.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

// Wave-size shim (Phase R.7). The two cross-lane reductions below (the
// per-logit dot product and the length-K max/sum that feeds log_softmax)
// historically used 32-lane `__shfl_xor_sync(0xffffffff, ...)` butterflies.
// On AMD wave64 that is BROKEN: HIP's `__shfl_*_sync` static_asserts a
// 64-bit mask (a 32-bit `0xffffffff` is rejected because it would silently
// drop lanes 32-63), and even the native shuffle mangles cross-32-lane
// offsets in wave64 mode (verified on gfx1151). So both reductions are
// re-expressed as wave-agnostic shared-memory block reductions
// (`kiln_block_reduce_sum/max`) over a power-of-two blockDim >= 64 — correct
// on NVIDIA, AMD wave32, AND AMD wave64, and numerically equivalent to the
// old warp-shuffle path up to F32 reduction-order associativity (covered by
// the parity tolerance). The CUDA path takes the identical code.
#include "kt_gpu_compat.cuh"

namespace {

constexpr int kWarpSize = 32;

// Upper bound on threads per block: K_max (32) * 32 = 1024, the Ampere /
// CDNA max. The block-reduce scratch is sized to this.
constexpr int kMaxThreads = 1024;

// Reduce a length-K float array `src` (held in shared memory) to a single
// scalar via the supplied operator, broadcasting the result via `bcast[0]`.
// `K` must be <= 64. Wave-size-AGNOSTIC: every thread of the block loads
// `(tid < K) ? src[tid] : identity` and the reduction runs through the
// shared-memory tree `kiln_block_reduce_*` (no cross-lane shuffles), so it
// is correct on NVIDIA + AMD wave32 + AMD wave64. `scratch` must hold >=
// blockDim.x floats and blockDim.x must be a power of two (the launcher
// guarantees both).
//
// Caller guarantees `__syncthreads()` was called before invoking us (so
// `src` writes by other threads are visible). `kiln_block_reduce_*` ends
// with a barrier, and we `__syncthreads()` after writing `bcast` so callers
// see `bcast[0]`.
enum class ReduceOp { Max, Sum };

template <int K, ReduceOp Op>
__device__ __forceinline__ void block_reduce_k(
    const float* __restrict__ src, float* bcast, float* scratch
) {
    const int tid = threadIdx.x;
    const float identity = (Op == ReduceOp::Max) ? -INFINITY : 0.0f;
    const float v = (tid < K) ? src[tid] : identity;
    float r;
    if (Op == ReduceOp::Max) {
        r = kiln_block_reduce_max(v, scratch);
    } else {
        r = kiln_block_reduce_sum(v, scratch);
    }
    if (tid == 0) {
        bcast[0] = r;
    }
    __syncthreads();
}

// =====================================================================
// Backward kernel: emit d_hidden[t, h] for the OPD top-K reverse-KL loss.
// =====================================================================
//
// The forward computed per-position KL_t = sum_k p_hat[k] * (log_p_hat[k] -
// log_q_hat[k]) where p_hat and log_p_hat are derived from s_logits =
// hidden[t] @ head_t[:, idx[t, k]]. With log_q_hat depending only on the
// (constant) teacher logprobs, the gradient through s_logits is:
//
//   d L / d s_logits[t, j] = p_hat[t, j] * (log_p_hat[t, j] - log_q_hat[t, j] - KL_t) * upstream[t]
//
// where upstream[t] depends on the output mode:
//   ScalarMean   : upstream[t] = grad_loss / T_active
//                  (or 1.0 / T_active when grad_loss == nullptr)
//   PerPosition  : upstream[t] = grad_loss[t]
//
// Then d_hidden[active_pos[t], h] = sum_k d_s_logits[t, k] *
// head_t[h, idx[t, k]].
//
// Implementation: same block layout as forward (one block per token, K
// warps × 32 threads). We reproduce forward steps 1–3 to recover p_hat,
// log_p_hat, log_q_hat, KL_t in shared memory; one warp lane writes
// d_s_logits[k]; then ALL blockDim.x threads cooperatively stride over h
// to compute d_hidden[active_pos[t], h]. The per-h work is K mults +
// K adds + an indexed read from head_t — small and well-balanced across
// threads.
//
// OutputMode: 0 = ScalarMean, 1 = PerPosition.

template <typename T, int K, int OutputMode>
__global__ void opd_topk_kl_bwd_kernel(
    const T* __restrict__ hidden,
    const T* __restrict__ head_t,
    const uint32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_lp_q,
    const uint32_t* __restrict__ active_pos,
    const float* __restrict__ grad_loss,    // scalar/null (mode 0) or [T_active] (mode 1)
    float scale_factor,                      // (1/T_active) for ScalarMean, 1.0 for PerPosition
    T* __restrict__ d_hidden,                // zero-filled [T, H]
    int hidden_size,
    int vocab_size,
    int t_active
) {
    const int t = blockIdx.x;
    if (t >= t_active) return;

    const int tid = threadIdx.x;
    const int blk = blockDim.x;

    __shared__ float s_logits[64];
    __shared__ float s_lp_q[64];
    __shared__ float exp_p[64];
    __shared__ float exp_q[64];
    __shared__ float d_s_logits[64];
    __shared__ float bcast[1];
    // Wave-agnostic block-reduce scratch — one float per thread (the launcher
    // caps blockDim at kMaxThreads and guarantees it is a power of two).
    __shared__ float scratch[kMaxThreads];

    // ---- recompute forward state to recover p_hat, log_p_hat, log_q_hat, KL_t ----
    //
    // Per-logit dot product s_logits[k] = sum_h hidden[t,h] * head_t[h, idx[t,k]].
    // Wave-size fix (R.7): instead of one warp per logit with an intra-warp
    // shuffle (broken on AMD wave64), ALL blockDim.x threads cooperate on one
    // logit at a time, striding over h and reducing through the shared-memory
    // block reduction. K block-reductions total — wave32/64-correct on AMD and
    // NVIDIA, and behavior-identical on CUDA up to F32 reduction-order.
    if (tid < K) {
        s_lp_q[tid] = topk_lp_q[t * K + tid];
    }
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        const uint32_t col = topk_idx[t * K + k];
        float partial = 0.0f;
        for (int h = tid; h < hidden_size; h += blk) {
            const float a = static_cast<float>(hidden[t * hidden_size + h]);
            const float b = static_cast<float>(head_t[h * vocab_size + col]);
            partial += a * b;
        }
        const float dot = kiln_block_reduce_sum(partial, scratch);
        if (tid == 0) {
            s_logits[k] = dot;
        }
        __syncthreads();
    }

    block_reduce_k<K, ReduceOp::Max>(s_logits, bcast, scratch);
    const float m_p = bcast[0];
    if (tid < K) {
        exp_p[tid] = expf(s_logits[tid] - m_p);
    }
    __syncthreads();
    block_reduce_k<K, ReduceOp::Sum>(exp_p, bcast, scratch);
    const float z_p = bcast[0];
    const float log_z_p = logf(z_p);

    block_reduce_k<K, ReduceOp::Max>(s_lp_q, bcast, scratch);
    const float m_q = bcast[0];
    if (tid < K) {
        exp_q[tid] = expf(s_lp_q[tid] - m_q);
    }
    __syncthreads();
    block_reduce_k<K, ReduceOp::Sum>(exp_q, bcast, scratch);
    const float z_q = bcast[0];
    const float log_z_q = logf(z_q);

    // KL_t — needed below in the d_s_logits formula.
    __shared__ float kl_partial[64];
    if (tid < K) {
        const float log_p = (s_logits[tid] - m_p) - log_z_p;
        const float log_q = (s_lp_q[tid] - m_q) - log_z_q;
        const float p_hat = exp_p[tid] / z_p;
        kl_partial[tid] = p_hat * (log_p - log_q);
    }
    __syncthreads();
    block_reduce_k<K, ReduceOp::Sum>(kl_partial, bcast, scratch);
    const float kl_t = bcast[0];

    // Decide the per-position upstream gradient.
    float upstream;
    if (OutputMode == 0) {
        upstream = (grad_loss == nullptr ? 1.0f : grad_loss[0]) * scale_factor;
    } else {
        upstream = grad_loss[t] * scale_factor;
    }

    // ---- d_s_logits[k] = p_hat[k] * (log_p_hat[k] - log_q_hat[k] - KL_t) * upstream ----
    if (tid < K) {
        const float log_p = (s_logits[tid] - m_p) - log_z_p;
        const float log_q = (s_lp_q[tid] - m_q) - log_z_q;
        const float p_hat = exp_p[tid] / z_p;
        d_s_logits[tid] = p_hat * (log_p - log_q - kl_t) * upstream;
    }
    __syncthreads();

    // ---- d_hidden[active_pos[t], h] = sum_k d_s_logits[k] * head_t[h, idx[t, k]] ----
    // All blockDim.x threads stride over h. For K=32 the inner loop is 32
    // gather-mul-adds with a known-small K; the compiler can unroll.
    const uint32_t out_t = active_pos[t];
    const int stride = blockDim.x;
    for (int h = tid; h < hidden_size; h += stride) {
        float acc = 0.0f;
        #pragma unroll 16
        for (int k = 0; k < K; ++k) {
            const uint32_t col = topk_idx[t * K + k];
            const float head_v = static_cast<float>(head_t[h * vocab_size + col]);
            acc += d_s_logits[k] * head_v;
        }
        d_hidden[out_t * hidden_size + h] = static_cast<T>(acc);
    }
}

// Dispatch on K + OutputMode + dtype for the backward.
template <typename T>
static int launch_bwd_for_k(
    const void* hidden,
    const void* head_t,
    const void* topk_idx,
    const void* topk_lp_q,
    const void* active_pos,
    const void* grad_loss,
    float scale_factor,
    void* d_hidden,
    int hidden_size,
    int vocab_size,
    int t_active,
    int top_k,
    int output_mode,
    cudaStream_t stream
) {
    const T* hidden_t = reinterpret_cast<const T*>(hidden);
    const T* head_t_t = reinterpret_cast<const T*>(head_t);
    const uint32_t* idx = reinterpret_cast<const uint32_t*>(topk_idx);
    const float* lp_q = reinterpret_cast<const float*>(topk_lp_q);
    const uint32_t* active = reinterpret_cast<const uint32_t*>(active_pos);
    const float* gl = reinterpret_cast<const float*>(grad_loss);
    T* dh = reinterpret_cast<T*>(d_hidden);

    dim3 grid(t_active);
    if (top_k == 16) {
        dim3 block(16 * kWarpSize);
        if (output_mode == 0) {
            opd_topk_kl_bwd_kernel<T, 16, 0><<<grid, block, 0, stream>>>(
                hidden_t, head_t_t, idx, lp_q, active, gl, scale_factor, dh,
                hidden_size, vocab_size, t_active);
        } else {
            opd_topk_kl_bwd_kernel<T, 16, 1><<<grid, block, 0, stream>>>(
                hidden_t, head_t_t, idx, lp_q, active, gl, scale_factor, dh,
                hidden_size, vocab_size, t_active);
        }
    } else if (top_k == 32) {
        dim3 block(32 * kWarpSize);
        if (output_mode == 0) {
            opd_topk_kl_bwd_kernel<T, 32, 0><<<grid, block, 0, stream>>>(
                hidden_t, head_t_t, idx, lp_q, active, gl, scale_factor, dh,
                hidden_size, vocab_size, t_active);
        } else {
            opd_topk_kl_bwd_kernel<T, 32, 1><<<grid, block, 0, stream>>>(
                hidden_t, head_t_t, idx, lp_q, active, gl, scale_factor, dh,
                hidden_size, vocab_size, t_active);
        }
    } else {
        return -3;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int>(err);
#if KILN_IS_HIP
    // (Phase R.7) Make the ROCm backward SYNCHRONOUS before returning to Rust.
    //
    // Unlike CUDA's cached primary context, kiln's `primary_rocm_context()`
    // mints a FRESH RocmContext — and thus a fresh default stream — for every
    // device tensor (hidden, head_t, the freshly-allocated d_hidden output,
    // ...). So this kernel launches on `hidden`'s stream while the Rust
    // read-back (`rocm_to_host_copy(d_hidden)`) drains a DIFFERENT stream; with
    // no cross-stream ordering the host read races the kernel and returns the
    // zero-initialised output buffer ("got 0"). A device-wide sync here is the
    // only point that guarantees the kernel has completed regardless of which
    // per-tensor stream it ran on, so the caller's read is correct. Also
    // surfaces any synchronous launch fault that the post-launch
    // cudaGetLastError() above cannot see. HIP-only — CUDA stays fully async
    // and byte-identical.
    hipError_t serr = hipDeviceSynchronize();
    if (serr != hipSuccess) return 1000 + static_cast<int>(serr);
#endif
    return 0;
}

}  // namespace

// =====================================================================
// Backward C-ABI entry points.
// =====================================================================
//
// `output_mode` is 0 for ScalarMean (grad_loss is a single scalar f32,
// or nullptr for an implicit unit seed)
// and 1 for PerPosition (grad_loss is a length-t_active f32 array).
// `scale_factor` is (1 / t_active) for ScalarMean and 1.0 for
// PerPosition — caller computes this once, kernel just multiplies.

extern "C" int32_t kiln_opd_topk_kl_bwd_bf16(
    const void* hidden,
    const void* head_t,
    const void* topk_indices,
    const void* topk_lp_q,
    const void* active_pos,
    const void* grad_loss,
    float scale_factor,
    void* d_hidden,
    int32_t t_active,
    int32_t hidden_size,
    int32_t vocab_size,
    int32_t top_k,
    int32_t output_mode,
    void* stream
) {
    if (t_active <= 0) return 0;
    if (hidden_size <= 0 || vocab_size <= 0 || top_k <= 0) return -1;
    if (output_mode != 0 && output_mode != 1) return -2;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    return launch_bwd_for_k<__nv_bfloat16>(
        hidden, head_t, topk_indices, topk_lp_q, active_pos, grad_loss, scale_factor,
        d_hidden, hidden_size, vocab_size, t_active, top_k, output_mode, s);
}

extern "C" int32_t kiln_opd_topk_kl_bwd_f32(
    const void* hidden,
    const void* head_t,
    const void* topk_indices,
    const void* topk_lp_q,
    const void* active_pos,
    const void* grad_loss,
    float scale_factor,
    void* d_hidden,
    int32_t t_active,
    int32_t hidden_size,
    int32_t vocab_size,
    int32_t top_k,
    int32_t output_mode,
    void* stream
) {
    if (t_active <= 0) return 0;
    if (hidden_size <= 0 || vocab_size <= 0 || top_k <= 0) return -1;
    if (output_mode != 0 && output_mode != 1) return -2;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    return launch_bwd_for_k<float>(
        hidden, head_t, topk_indices, topk_lp_q, active_pos, grad_loss, scale_factor,
        d_hidden, hidden_size, vocab_size, t_active, top_k, output_mode, s);
}

// =====================================================================
// Distribution-alignment metrics kernel — REMOVED (#1082, 2026-05-28).
// =====================================================================
//
// `opd_topk_metrics_kernel` + `launch_metrics_for_k` + the C-ABI entries
// `kiln_opd_topk_metrics_{bf16,f32}` were deleted in (#1082) — they were
// the device-side path for `crate::phase_b::compute_per_position_metrics`,
// which had no external callers (the audit at
// `docs/archive/candle-removal/opd-loss-kernel-candle-removal-stop-2026-05-28.md` records the
// scan). The kt-typed sibling `compute_per_position_metrics_kt` covers
// the same diagnostic via a CUDA composite (`per_position_forward_kt`)
// when (and if) a caller wires it up.
