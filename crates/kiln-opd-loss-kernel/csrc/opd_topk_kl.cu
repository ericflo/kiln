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
