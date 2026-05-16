// Fused OPD top-K reverse-KL CUDA kernel.
//
// See `docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md`
// §9.2 for the design. One launch produces per-token reverse KL between
// the student's distribution (gathered from `hidden @ head_t`) and the
// teacher's top-K distribution, both renormalised over the K support.
//
// # Memory & arithmetic intensity
//
// For T_active tokens × K=32 support × H=2560 hidden:
//   * Hidden traffic: T_active × H × 2 B (bf16) = ~20 MiB at T=4096.
//   * Head_t traffic: T_active × K × H × 2 B = ~670 MiB scattered reads
//     (the K columns of head_t are not contiguous; we read by index).
//   * Output traffic: T_active × 4 B = 16 KiB.
//   * FLOPs: T_active × K × H × 2 = ~671 MFLOPS at T=4096.
//
// This kernel is *bandwidth-bound* — the scattered head_t reads dominate.
// On A6000 (768 GB/s HBM), the lower bound is ~880 µs for T=4096.
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
// to float on the fly. The KL itself produces f32 output. Parity vs
// the candle CPU reference is enforced at ≤1e-4 abs at f32 and ≤5e-2 at
// bf16 by the parity tests in `kiln-opd-loss-kernel/src/tests.rs`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

namespace {

constexpr int kWarpSize = 32;

// Reduce a length-K float array `src` in shared memory to a single
// scalar via the supplied operator, broadcast via `bcast[0]`. `K` must
// be ≤ 64. Only lane 0..K-1 of warp 0 participate.
//
// Caller guarantees `__syncthreads()` was called before invoking us
// (so `src` writes by other warps are visible). We `__syncthreads()`
// at the end so callers see `bcast[0]`.
enum class ReduceOp { Max, Sum };

template <int K, ReduceOp Op>
__device__ __forceinline__ void block_reduce_k(
    const float* __restrict__ src, float* bcast
) {
    if (threadIdx.x < kWarpSize) {
        const float identity = (Op == ReduceOp::Max) ? -INFINITY : 0.0f;
        float v = (threadIdx.x < K) ? src[threadIdx.x] : identity;
        if constexpr (K > kWarpSize) {
            float v2 = (threadIdx.x + kWarpSize < K)
                ? src[threadIdx.x + kWarpSize]
                : identity;
            v = (Op == ReduceOp::Max) ? fmaxf(v, v2) : (v + v2);
        }
        for (int o = kWarpSize / 2; o > 0; o >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, v, o);
            v = (Op == ReduceOp::Max) ? fmaxf(v, other) : (v + other);
        }
        if (threadIdx.x == 0) {
            bcast[0] = v;
        }
    }
    __syncthreads();
}

template <typename T, int K>
__global__ void opd_topk_kl_fwd_kernel(
    const T* __restrict__ hidden,           // [T_active, H]
    const T* __restrict__ head_t,           // [H, V]
    const uint32_t* __restrict__ topk_idx,  // [T_active, K]
    const float* __restrict__ topk_lp_q,    // [T_active, K]
    float* __restrict__ kl_out,             // [T_active]
    int hidden_size,
    int vocab_size,
    int t_active
) {
    const int t = blockIdx.x;
    if (t >= t_active) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);

    // Shared state used across the block.
    __shared__ float s_logits[64];   // length K; one student logit per warp
    __shared__ float s_lp_q[64];     // length K; teacher logprob per warp
    __shared__ float exp_p[64];      // exp(s_logits - m_p), reused as p_hat
    __shared__ float exp_q[64];      // exp(s_lp_q - m_q)
    __shared__ float bcast[1];

    // ---- step 1: compute s_logits[warp_id] via warp-cooperative dot product ----
    if (warp_id < K) {
        if (lane == 0) {
            // One per-warp memo of the teacher logprob.
            s_lp_q[warp_id] = topk_lp_q[t * K + warp_id];
        }
        const uint32_t col = topk_idx[t * K + warp_id];
        float acc = 0.0f;
        for (int h = lane; h < hidden_size; h += kWarpSize) {
            const float a = static_cast<float>(hidden[t * hidden_size + h]);
            const float b = static_cast<float>(head_t[h * vocab_size + col]);
            acc += a * b;
        }
        // Warp reduce.
        for (int o = kWarpSize / 2; o > 0; o >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, o);
        }
        if (lane == 0) {
            s_logits[warp_id] = acc;
        }
    }
    __syncthreads();

    // ---- step 2: log_softmax of s_logits over K → log_p_hat ----
    block_reduce_k<K, ReduceOp::Max>(s_logits, bcast);
    const float m_p = bcast[0];
    if (warp_id < K && lane == 0) {
        exp_p[warp_id] = expf(s_logits[warp_id] - m_p);
    }
    __syncthreads();
    block_reduce_k<K, ReduceOp::Sum>(exp_p, bcast);
    const float z_p = bcast[0];
    const float log_z_p = logf(z_p);

    // ---- step 3: log_softmax of s_lp_q over K → log_q_hat ----
    block_reduce_k<K, ReduceOp::Max>(s_lp_q, bcast);
    const float m_q = bcast[0];
    if (warp_id < K && lane == 0) {
        exp_q[warp_id] = expf(s_lp_q[warp_id] - m_q);
    }
    __syncthreads();
    block_reduce_k<K, ReduceOp::Sum>(exp_q, bcast);
    const float z_q = bcast[0];
    const float log_z_q = logf(z_q);

    // ---- step 4: KL contribution per warp slot, block-sum ----
    __shared__ float kl_partial[64];
    if (warp_id < K && lane == 0) {
        const float log_p = (s_logits[warp_id] - m_p) - log_z_p;
        const float log_q = (s_lp_q[warp_id] - m_q) - log_z_q;
        const float p_hat = exp_p[warp_id] / z_p;
        kl_partial[warp_id] = p_hat * (log_p - log_q);
    }
    __syncthreads();
    block_reduce_k<K, ReduceOp::Sum>(kl_partial, bcast);
    if (threadIdx.x == 0) {
        kl_out[t] = bcast[0];
    }
}

// Dispatch instantiation by K. Returns 0 on success, non-zero on error.
template <typename T>
static int launch_for_k(
    const void* hidden,
    const void* head_t,
    const void* topk_idx,
    const void* topk_lp_q,
    void* kl_out,
    int hidden_size,
    int vocab_size,
    int t_active,
    int top_k,
    cudaStream_t stream
) {
    const T* hidden_t = reinterpret_cast<const T*>(hidden);
    const T* head_t_t = reinterpret_cast<const T*>(head_t);
    const uint32_t* idx = reinterpret_cast<const uint32_t*>(topk_idx);
    const float* lp_q = reinterpret_cast<const float*>(topk_lp_q);
    float* out = reinterpret_cast<float*>(kl_out);

    dim3 grid(t_active);
    if (top_k == 16) {
        dim3 block(16 * kWarpSize);  // 512 threads
        opd_topk_kl_fwd_kernel<T, 16><<<grid, block, 0, stream>>>(
            hidden_t, head_t_t, idx, lp_q, out, hidden_size, vocab_size, t_active);
    } else if (top_k == 32) {
        dim3 block(32 * kWarpSize);  // 1024 threads — max on Ampere
        opd_topk_kl_fwd_kernel<T, 32><<<grid, block, 0, stream>>>(
            hidden_t, head_t_t, idx, lp_q, out, hidden_size, vocab_size, t_active);
    } else {
        return -3;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int>(err);
    return 0;
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
//   PerPosition  : upstream[t] = grad_loss[t]
//
// Then d_hidden[t, h] = sum_k d_s_logits[t, k] * head_t[h, idx[t, k]].
//
// Implementation: same block layout as forward (one block per token, K
// warps × 32 threads). We reproduce forward steps 1–3 to recover p_hat,
// log_p_hat, log_q_hat, KL_t in shared memory; one warp lane writes
// d_s_logits[k]; then ALL blockDim.x threads cooperatively stride over h
// to compute d_hidden[t, h]. The per-h work is K mults + K adds + an
// indexed read from head_t — small and well-balanced across threads.
//
// OutputMode: 0 = ScalarMean, 1 = PerPosition.

template <typename T, int K, int OutputMode>
__global__ void opd_topk_kl_bwd_kernel(
    const T* __restrict__ hidden,
    const T* __restrict__ head_t,
    const uint32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_lp_q,
    const float* __restrict__ grad_loss,    // scalar (mode 0) or [T_active] (mode 1)
    float scale_factor,                      // (1/T_active) for ScalarMean, 1.0 for PerPosition
    T* __restrict__ d_hidden,                // [T_active, H]
    int hidden_size,
    int vocab_size,
    int t_active
) {
    const int t = blockIdx.x;
    if (t >= t_active) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);

    __shared__ float s_logits[64];
    __shared__ float s_lp_q[64];
    __shared__ float exp_p[64];
    __shared__ float exp_q[64];
    __shared__ float d_s_logits[64];
    __shared__ float bcast[1];

    // ---- recompute forward state to recover p_hat, log_p_hat, log_q_hat, KL_t ----
    if (warp_id < K) {
        if (lane == 0) {
            s_lp_q[warp_id] = topk_lp_q[t * K + warp_id];
        }
        const uint32_t col = topk_idx[t * K + warp_id];
        float acc = 0.0f;
        for (int h = lane; h < hidden_size; h += kWarpSize) {
            const float a = static_cast<float>(hidden[t * hidden_size + h]);
            const float b = static_cast<float>(head_t[h * vocab_size + col]);
            acc += a * b;
        }
        for (int o = kWarpSize / 2; o > 0; o >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, o);
        }
        if (lane == 0) {
            s_logits[warp_id] = acc;
        }
    }
    __syncthreads();

    block_reduce_k<K, ReduceOp::Max>(s_logits, bcast);
    const float m_p = bcast[0];
    if (warp_id < K && lane == 0) {
        exp_p[warp_id] = expf(s_logits[warp_id] - m_p);
    }
    __syncthreads();
    block_reduce_k<K, ReduceOp::Sum>(exp_p, bcast);
    const float z_p = bcast[0];
    const float log_z_p = logf(z_p);

    block_reduce_k<K, ReduceOp::Max>(s_lp_q, bcast);
    const float m_q = bcast[0];
    if (warp_id < K && lane == 0) {
        exp_q[warp_id] = expf(s_lp_q[warp_id] - m_q);
    }
    __syncthreads();
    block_reduce_k<K, ReduceOp::Sum>(exp_q, bcast);
    const float z_q = bcast[0];
    const float log_z_q = logf(z_q);

    // KL_t — needed below in the d_s_logits formula.
    __shared__ float kl_partial[64];
    if (warp_id < K && lane == 0) {
        const float log_p = (s_logits[warp_id] - m_p) - log_z_p;
        const float log_q = (s_lp_q[warp_id] - m_q) - log_z_q;
        const float p_hat = exp_p[warp_id] / z_p;
        kl_partial[warp_id] = p_hat * (log_p - log_q);
    }
    __syncthreads();
    block_reduce_k<K, ReduceOp::Sum>(kl_partial, bcast);
    const float kl_t = bcast[0];

    // Decide the per-position upstream gradient.
    float upstream;
    if (OutputMode == 0) {
        upstream = grad_loss[0] * scale_factor;
    } else {
        upstream = grad_loss[t] * scale_factor;
    }

    // ---- d_s_logits[k] = p_hat[k] * (log_p_hat[k] - log_q_hat[k] - KL_t) * upstream ----
    if (warp_id < K && lane == 0) {
        const float log_p = (s_logits[warp_id] - m_p) - log_z_p;
        const float log_q = (s_lp_q[warp_id] - m_q) - log_z_q;
        const float p_hat = exp_p[warp_id] / z_p;
        d_s_logits[warp_id] = p_hat * (log_p - log_q - kl_t) * upstream;
    }
    __syncthreads();

    // ---- d_hidden[t, h] = sum_k d_s_logits[k] * head_t[h, idx[t, k]] ----
    // All blockDim.x threads stride over h. For K=32 the inner loop is 32
    // gather-mul-adds with a known-small K; the compiler can unroll.
    const int stride = blockDim.x;
    for (int h = tid; h < hidden_size; h += stride) {
        float acc = 0.0f;
        #pragma unroll 16
        for (int k = 0; k < K; ++k) {
            const uint32_t col = topk_idx[t * K + k];
            const float head_v = static_cast<float>(head_t[h * vocab_size + col]);
            acc += d_s_logits[k] * head_v;
        }
        d_hidden[t * hidden_size + h] = static_cast<T>(acc);
    }
}

// Dispatch on K + OutputMode + dtype for the backward.
template <typename T>
static int launch_bwd_for_k(
    const void* hidden,
    const void* head_t,
    const void* topk_idx,
    const void* topk_lp_q,
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
    const float* gl = reinterpret_cast<const float*>(grad_loss);
    T* dh = reinterpret_cast<T*>(d_hidden);

    dim3 grid(t_active);
    if (top_k == 16) {
        dim3 block(16 * kWarpSize);
        if (output_mode == 0) {
            opd_topk_kl_bwd_kernel<T, 16, 0><<<grid, block, 0, stream>>>(
                hidden_t, head_t_t, idx, lp_q, gl, scale_factor, dh,
                hidden_size, vocab_size, t_active);
        } else {
            opd_topk_kl_bwd_kernel<T, 16, 1><<<grid, block, 0, stream>>>(
                hidden_t, head_t_t, idx, lp_q, gl, scale_factor, dh,
                hidden_size, vocab_size, t_active);
        }
    } else if (top_k == 32) {
        dim3 block(32 * kWarpSize);
        if (output_mode == 0) {
            opd_topk_kl_bwd_kernel<T, 32, 0><<<grid, block, 0, stream>>>(
                hidden_t, head_t_t, idx, lp_q, gl, scale_factor, dh,
                hidden_size, vocab_size, t_active);
        } else {
            opd_topk_kl_bwd_kernel<T, 32, 1><<<grid, block, 0, stream>>>(
                hidden_t, head_t_t, idx, lp_q, gl, scale_factor, dh,
                hidden_size, vocab_size, t_active);
        }
    } else {
        return -3;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int>(err);
    return 0;
}

}  // namespace

extern "C" int32_t kiln_opd_topk_kl_fwd_bf16(
    const void* hidden,
    const void* head_t,
    const void* topk_indices,
    const void* topk_lp_q,
    void* kl_out,
    int32_t t_active,
    int32_t hidden_size,
    int32_t vocab_size,
    int32_t top_k,
    void* stream
) {
    if (t_active <= 0) return 0;
    if (hidden_size <= 0 || vocab_size <= 0 || top_k <= 0) return -1;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    return launch_for_k<__nv_bfloat16>(
        hidden, head_t, topk_indices, topk_lp_q, kl_out,
        hidden_size, vocab_size, t_active, top_k, s);
}

extern "C" int32_t kiln_opd_topk_kl_fwd_f32(
    const void* hidden,
    const void* head_t,
    const void* topk_indices,
    const void* topk_lp_q,
    void* kl_out,
    int32_t t_active,
    int32_t hidden_size,
    int32_t vocab_size,
    int32_t top_k,
    void* stream
) {
    if (t_active <= 0) return 0;
    if (hidden_size <= 0 || vocab_size <= 0 || top_k <= 0) return -1;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    return launch_for_k<float>(
        hidden, head_t, topk_indices, topk_lp_q, kl_out,
        hidden_size, vocab_size, t_active, top_k, s);
}

// =====================================================================
// Backward C-ABI entry points.
// =====================================================================
//
// `output_mode` is 0 for ScalarMean (grad_loss is a single scalar f32)
// and 1 for PerPosition (grad_loss is a length-t_active f32 array).
// `scale_factor` is (1 / t_active) for ScalarMean and 1.0 for
// PerPosition — caller computes this once, kernel just multiplies.

extern "C" int32_t kiln_opd_topk_kl_bwd_bf16(
    const void* hidden,
    const void* head_t,
    const void* topk_indices,
    const void* topk_lp_q,
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
        hidden, head_t, topk_indices, topk_lp_q, grad_loss, scale_factor,
        d_hidden, hidden_size, vocab_size, t_active, top_k, output_mode, s);
}

extern "C" int32_t kiln_opd_topk_kl_bwd_f32(
    const void* hidden,
    const void* head_t,
    const void* topk_indices,
    const void* topk_lp_q,
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
        hidden, head_t, topk_indices, topk_lp_q, grad_loss, scale_factor,
        d_hidden, hidden_size, vocab_size, t_active, top_k, output_mode, s);
}

// =====================================================================
// Distribution-alignment metrics kernel.
// =====================================================================
//
// §3.8 of the grand plan wants per-position visibility into:
//   - entropy_gap = |H(q_hat) - H(p_hat)|
//   - overlap_token_advantage (Li et al. 2026 eq 7)
//   - per-position reverse KL (already provided by the forward kernel)
//
// We compute three of the four kernel-side here (overlap_ratio needs
// the student's full top-K which we DON'T have — that's computed by a
// separate probe). For each active token writes:
//   metrics_out[3*t + 0] = H(p_hat)           — student entropy over K support
//   metrics_out[3*t + 1] = H(q_hat)           — teacher entropy over K support
//   metrics_out[3*t + 2] = E_p[log p_hat - log q_hat]   (the KL itself)
//
// The trainer derives entropy_gap = |out[1] - out[0]|. Computing it on
// the device side keeps it cheap; the trainer drops it into the
// OpdDiagnosticSnapshot at validation cadence.

template <typename T, int K>
__global__ void opd_topk_metrics_kernel(
    const T* __restrict__ hidden,
    const T* __restrict__ head_t,
    const uint32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_lp_q,
    float* __restrict__ metrics_out,   // [T_active, 3]
    int hidden_size,
    int vocab_size,
    int t_active
) {
    const int t = blockIdx.x;
    if (t >= t_active) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);

    __shared__ float s_logits[64];
    __shared__ float s_lp_q[64];
    __shared__ float exp_p[64];
    __shared__ float exp_q[64];
    __shared__ float bcast[1];

    // Forward recompute (same as the loss kernel; the metrics kernel
    // is intended to run at validation cadence, not every train step).
    if (warp_id < K) {
        if (lane == 0) {
            s_lp_q[warp_id] = topk_lp_q[t * K + warp_id];
        }
        const uint32_t col = topk_idx[t * K + warp_id];
        float acc = 0.0f;
        for (int h = lane; h < hidden_size; h += kWarpSize) {
            const float a = static_cast<float>(hidden[t * hidden_size + h]);
            const float b = static_cast<float>(head_t[h * vocab_size + col]);
            acc += a * b;
        }
        for (int o = kWarpSize / 2; o > 0; o >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, o);
        }
        if (lane == 0) {
            s_logits[warp_id] = acc;
        }
    }
    __syncthreads();

    block_reduce_k<K, ReduceOp::Max>(s_logits, bcast);
    const float m_p = bcast[0];
    if (warp_id < K && lane == 0) {
        exp_p[warp_id] = expf(s_logits[warp_id] - m_p);
    }
    __syncthreads();
    block_reduce_k<K, ReduceOp::Sum>(exp_p, bcast);
    const float z_p = bcast[0];
    const float log_z_p = logf(z_p);

    block_reduce_k<K, ReduceOp::Max>(s_lp_q, bcast);
    const float m_q = bcast[0];
    if (warp_id < K && lane == 0) {
        exp_q[warp_id] = expf(s_lp_q[warp_id] - m_q);
    }
    __syncthreads();
    block_reduce_k<K, ReduceOp::Sum>(exp_q, bcast);
    const float z_q = bcast[0];
    const float log_z_q = logf(z_q);

    // Compute entropy(p_hat), entropy(q_hat), KL(p||q) all over K support.
    //   H(p) = -sum_k p_hat[k] * log_p_hat[k]
    //   H(q) = -sum_k q_hat[k] * log_q_hat[k]
    //   KL   = sum_k p_hat[k] * (log_p_hat[k] - log_q_hat[k])
    __shared__ float Hp_partial[64];
    __shared__ float Hq_partial[64];
    __shared__ float KL_partial[64];

    if (warp_id < K && lane == 0) {
        const float log_p = (s_logits[warp_id] - m_p) - log_z_p;
        const float log_q = (s_lp_q[warp_id] - m_q) - log_z_q;
        const float p_hat = exp_p[warp_id] / z_p;
        const float q_hat = exp_q[warp_id] / z_q;
        Hp_partial[warp_id] = -p_hat * log_p;
        Hq_partial[warp_id] = -q_hat * log_q;
        KL_partial[warp_id] = p_hat * (log_p - log_q);
    }
    __syncthreads();

    block_reduce_k<K, ReduceOp::Sum>(Hp_partial, bcast);
    const float H_p = bcast[0];
    block_reduce_k<K, ReduceOp::Sum>(Hq_partial, bcast);
    const float H_q = bcast[0];
    block_reduce_k<K, ReduceOp::Sum>(KL_partial, bcast);
    const float KL_t = bcast[0];

    if (threadIdx.x == 0) {
        metrics_out[3 * t + 0] = H_p;
        metrics_out[3 * t + 1] = H_q;
        metrics_out[3 * t + 2] = KL_t;
    }
}

template <typename T>
static int launch_metrics_for_k(
    const void* hidden,
    const void* head_t,
    const void* topk_idx,
    const void* topk_lp_q,
    void* metrics_out,
    int hidden_size,
    int vocab_size,
    int t_active,
    int top_k,
    cudaStream_t stream
) {
    const T* hidden_t = reinterpret_cast<const T*>(hidden);
    const T* head_t_t = reinterpret_cast<const T*>(head_t);
    const uint32_t* idx = reinterpret_cast<const uint32_t*>(topk_idx);
    const float* lp_q = reinterpret_cast<const float*>(topk_lp_q);
    float* out = reinterpret_cast<float*>(metrics_out);
    dim3 grid(t_active);
    if (top_k == 16) {
        dim3 block(16 * kWarpSize);
        opd_topk_metrics_kernel<T, 16><<<grid, block, 0, stream>>>(
            hidden_t, head_t_t, idx, lp_q, out, hidden_size, vocab_size, t_active);
    } else if (top_k == 32) {
        dim3 block(32 * kWarpSize);
        opd_topk_metrics_kernel<T, 32><<<grid, block, 0, stream>>>(
            hidden_t, head_t_t, idx, lp_q, out, hidden_size, vocab_size, t_active);
    } else {
        return -3;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int>(err);
    return 0;
}

extern "C" int32_t kiln_opd_topk_metrics_bf16(
    const void* hidden,
    const void* head_t,
    const void* topk_indices,
    const void* topk_lp_q,
    void* metrics_out,
    int32_t t_active,
    int32_t hidden_size,
    int32_t vocab_size,
    int32_t top_k,
    void* stream
) {
    if (t_active <= 0) return 0;
    if (hidden_size <= 0 || vocab_size <= 0 || top_k <= 0) return -1;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    return launch_metrics_for_k<__nv_bfloat16>(
        hidden, head_t, topk_indices, topk_lp_q, metrics_out,
        hidden_size, vocab_size, t_active, top_k, s);
}

extern "C" int32_t kiln_opd_topk_metrics_f32(
    const void* hidden,
    const void* head_t,
    const void* topk_indices,
    const void* topk_lp_q,
    void* metrics_out,
    int32_t t_active,
    int32_t hidden_size,
    int32_t vocab_size,
    int32_t top_k,
    void* stream
) {
    if (t_active <= 0) return 0;
    if (hidden_size <= 0 || vocab_size <= 0 || top_k <= 0) return -1;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    return launch_metrics_for_k<float>(
        hidden, head_t, topk_indices, topk_lp_q, metrics_out,
        hidden_size, vocab_size, t_active, top_k, s);
}
