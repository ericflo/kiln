// Kiln GDN recurrent (single-token) forward kernel.
//
// One CUDA block per (batch, head). One thread per dv column (so blockDim.x
// = dv). Each thread:
//   1. Loads its column S[:, my_d] from global state into per-thread
//      registers/local memory as F32 (dk floats).
//   2. Cooperatively pulls k_t, q_t into shared memory; thread 0 loads the
//      scalar decay = exp(g_t) and beta_t.
//   3. __syncthreads() once.
//   4. Phase A: per-thread loop over dk computes
//        decayed[i] = decay * s_local[i]
//        v_pred    += k_smem[i] * decayed[i]
//      (v_pred contributes to one element of the dv-wide v_pred vector,
//       the element this thread owns).
//   5. Computes delta = beta * (v_t[my_d] - v_pred).
//   6. Phase B: per-thread loop over dk computes
//        new_s = decayed[i] + k_smem[i] * delta
//        s_local[i] = new_s             (still in registers)
//        out_acc += q_smem[i] * new_s
//      and writes new_s back to global state.
//   7. Writes bf16(out_acc) to out[my_d].
//
// kiln configures dk = dv = 128, so the per-thread state column is 128 F32
// values = 512 bytes ≈ 128 32-bit registers — comfortably under sm_86's 255
// reg/thread cap. Shared memory is tiny: 2 * dk * 4 (q, k as F32) + a few
// scalars ≈ 1 KiB. The win over the candle-op path is eliminating the
// chunkwise machinery (preshape, decay matrix, KKT, forward sub, B_mask,
// matmul into state) when there is only a single token to advance.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "recurrent_gdn_fwd.h"

namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

// MAX_DK caps the per-thread register array size. kiln uses dk = 128.
template <int MAX_DK>
__global__ void recurrent_gdn_fwd_kernel(
    const __nv_bfloat16 *__restrict__ q,     // [B*H, dk]
    const __nv_bfloat16 *__restrict__ k,     // [B*H, dk]
    const __nv_bfloat16 *__restrict__ v,     // [B*H, dv]
    const __nv_bfloat16 *__restrict__ beta,  // [B*H]
    const __nv_bfloat16 *__restrict__ g,     // [B*H]
    __nv_bfloat16 *__restrict__ state,       // [B*H, dk, dv]
    __nv_bfloat16 *__restrict__ out,         // [B*H, dv]
    int dk,
    int dv
) {
    const int bh  = blockIdx.x;
    const int tid = threadIdx.x;

    if (tid >= dv) return;

    const __nv_bfloat16 *q_base    = q     + (size_t)bh * dk;
    const __nv_bfloat16 *k_base    = k     + (size_t)bh * dk;
    const __nv_bfloat16 *v_base    = v     + (size_t)bh * dv;
    __nv_bfloat16       *s_base    = state + (size_t)bh * dk * dv;
    __nv_bfloat16       *out_base  = out   + (size_t)bh * dv;

    extern __shared__ float smem[];
    float *q_smem = smem;            // dk F32 floats
    float *k_smem = smem + dk;       // dk F32 floats
    float *scalars = smem + 2 * dk;  // [decay, beta]

    // Cooperative load of q_t / k_t into shared memory as F32.
    for (int i = tid; i < dk; i += blockDim.x) {
        q_smem[i] = bf16_to_f32(q_base[i]);
        k_smem[i] = bf16_to_f32(k_base[i]);
    }

    // Thread 0 loads the per-(batch,head) scalars.
    if (tid == 0) {
        float g_val   = bf16_to_f32(g[bh]);
        float b_val   = bf16_to_f32(beta[bh]);
        scalars[0]    = expf(g_val);   // decay
        scalars[1]    = b_val;          // beta
    }
    __syncthreads();

    const float decay  = scalars[0];
    const float beta_t = scalars[1];

    // Per-thread state column [dk] in registers (or local memory for large dk).
    float s_local[MAX_DK];
    #pragma unroll
    for (int i = 0; i < MAX_DK; ++i) {
        if (i < dk) {
            s_local[i] = bf16_to_f32(s_base[(size_t)i * dv + tid]);
        }
    }

    // Phase A: compute decayed state and v_pred for this column.
    //   decayed[i] = decay * s_local[i]
    //   v_pred    += k_smem[i] * decayed[i]
    float v_pred = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAX_DK; ++i) {
        if (i < dk) {
            float d = decay * s_local[i];
            s_local[i] = d;  // overwrite with decayed value
            v_pred += k_smem[i] * d;
        }
    }

    float v_t   = bf16_to_f32(v_base[tid]);
    float delta = beta_t * (v_t - v_pred);

    // Phase B: update state and accumulate output.
    //   new_s = decayed[i] + k_smem[i] * delta
    //   s_local[i] = new_s
    //   out_acc += q_smem[i] * new_s
    float out_acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAX_DK; ++i) {
        if (i < dk) {
            float new_s = s_local[i] + k_smem[i] * delta;
            s_local[i] = new_s;
            out_acc += q_smem[i] * new_s;
            s_base[(size_t)i * dv + tid] = f32_to_bf16(new_s);
        }
    }

    out_base[tid] = f32_to_bf16(out_acc);
}

// --------------------------------------------------------------------------
// Fused variant: qk_norm + recurrent step + gated_norm, one block per
// (batch, head). Semantics match the forward.rs candle reference pipeline:
//
//   Step 5 (qk_norm):
//     inv_q  = q_scale * rsqrt(sum_j(q_raw^2) + l2_eps)     (F32, per row)
//     inv_k  =            rsqrt(sum_j(k_raw^2) + l2_eps)    (F32, per row)
//     q_norm = q_raw * inv_q                                (F32, dk)
//     k_norm = k_raw * inv_k                                (F32, dk)
//
//   Step 7 (recurrent_step): same math as the non-fused kernel, using
//     q_norm / k_norm in place of q / k.
//
//   Step 8 (gated_rms_norm):
//     rms_inv = rsqrt(mean_j(attn_out^2) + rms_eps)         (F32, per row)
//     silu(z) = z * sigmoid(z) = z / (1 + exp(-z))          (F32)
//     out     = attn_out * rms_inv * gamma * silu(z)        (F32 -> bf16)
//
// Shared memory layout (F32):
//   q_smem  [dk]        — normalized Q (reused for raw load pre-norm)
//   k_smem  [dk]        — normalized K
//   reduce  [kMaxWarps] — warp partial sums for block_reduce_sum
//   bcast   [5]         — decay, beta_t, inv_q_scaled, inv_k, rms_inv
//
// kMaxWarps is fixed at 32 (CUDA's cap on threads/block is 1024 = 32 warps);
// kiln runs dv=128 so only the first ceil(dv/32) entries are used. Total
// smem for dk=dv=128: (128 + 128 + 32 + 5) * 4 = 1172 bytes. Well under
// sm_86's 48 KiB/block default.
//
// Register pressure: the per-thread state column `s_local[MAX_DK]` is the
// dominant consumer (128 F32 = 128 reg for dk=128), unchanged from the
// non-fused kernel. The extra L2/RMS reductions use O(1) scalars.
//
// Barrier discipline: block_reduce_sum has two __syncthreads() inside it,
// and we use one more before Phase 3 to guarantee every thread sees the
// normalized q_smem/k_smem. blockDim.x is always equal to dv (one thread
// per output element), so no thread exits early and every barrier is safe.

constexpr int kMaxWarps = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float *reduce_smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        reduce_smem[warp] = v;
    }
    __syncthreads();

    if (warp == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        float w = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
        w = warp_reduce_sum(w);
        if (lane == 0) {
            reduce_smem[0] = w;
        }
    }
    __syncthreads();
    return reduce_smem[0];
}

__device__ __forceinline__ float silu_f32(float x) {
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x)); stable for all x
    // because __expf of a negative input is in (0, 1].
    return x / (1.0f + __expf(-x));
}

template <int MAX_DK>
__global__ void recurrent_gdn_fwd_fused_norm_kernel(
    const __nv_bfloat16 *__restrict__ q_raw,   // [B*H, dk]
    const __nv_bfloat16 *__restrict__ k_raw,   // [B*H, dk]
    const __nv_bfloat16 *__restrict__ v,       // [B*H, dv]
    const __nv_bfloat16 *__restrict__ beta,    // [B*H]
    const __nv_bfloat16 *__restrict__ g,       // [B*H]
    const __nv_bfloat16 *__restrict__ z,       // [B*H, dv]
    const __nv_bfloat16 *__restrict__ gamma,   // [dv]
    __nv_bfloat16 *__restrict__ state,         // [B*H, dk, dv]
    __nv_bfloat16 *__restrict__ out,           // [B*H, dv]
    int dk,
    int dv,
    float q_scale,
    float l2_eps,
    float rms_eps
) {
    const int bh  = blockIdx.x;
    const int tid = threadIdx.x;

    if (tid >= dv) return;  // blockDim.x == dv, so in practice no-op.

    const __nv_bfloat16 *q_base   = q_raw + (size_t)bh * dk;
    const __nv_bfloat16 *k_base   = k_raw + (size_t)bh * dk;
    const __nv_bfloat16 *v_base   = v     + (size_t)bh * dv;
    const __nv_bfloat16 *z_base   = z     + (size_t)bh * dv;
    __nv_bfloat16       *s_base   = state + (size_t)bh * dk * dv;
    __nv_bfloat16       *out_base = out   + (size_t)bh * dv;

    extern __shared__ float smem[];
    float *q_smem  = smem;                        // dk F32
    float *k_smem  = smem + dk;                   // dk F32
    float *reduce  = smem + 2 * dk;               // kMaxWarps F32
    float *bcast   = smem + 2 * dk + kMaxWarps;   // 5 F32: decay, beta, inv_q, inv_k, rms_inv

    // ---------------- Phase 0: cooperative load of raw q/k ----------------
    for (int i = tid; i < dk; i += blockDim.x) {
        q_smem[i] = bf16_to_f32(q_base[i]);
        k_smem[i] = bf16_to_f32(k_base[i]);
    }

    // Thread 0 preloads scalars: decay and beta. No __syncthreads() yet — the
    // block_reduce_sum calls below each have their own barriers.
    if (tid == 0) {
        float g_val = bf16_to_f32(g[bh]);
        float b_val = bf16_to_f32(beta[bh]);
        bcast[0] = expf(g_val);  // decay
        bcast[1] = b_val;        // beta
    }
    __syncthreads();  // q_smem/k_smem visible before the reductions below.

    // ---------------- Phase 1: L2 norm reductions over q and k ----------------
    float q_sumsq = 0.0f;
    float k_sumsq = 0.0f;
    for (int i = tid; i < dk; i += blockDim.x) {
        float qi = q_smem[i];
        float ki = k_smem[i];
        q_sumsq += qi * qi;
        k_sumsq += ki * ki;
    }
    float q_total = block_reduce_sum(q_sumsq, reduce);
    float k_total = block_reduce_sum(k_sumsq, reduce);

    if (tid == 0) {
        bcast[2] = q_scale * rsqrtf(q_total + l2_eps);  // inv_q (already * q_scale)
        bcast[3] = rsqrtf(k_total + l2_eps);            // inv_k
    }
    __syncthreads();

    const float inv_q = bcast[2];
    const float inv_k = bcast[3];

    // ---------------- Phase 2: apply norms in-place in shared memory ----------------
    for (int i = tid; i < dk; i += blockDim.x) {
        q_smem[i] *= inv_q;
        k_smem[i] *= inv_k;
    }
    __syncthreads();  // state-column loop below reads q_smem/k_smem for all i.

    const float decay  = bcast[0];
    const float beta_t = bcast[1];

    // ---------------- Phase 3: recurrent step (identical to non-fused kernel) ----------------
    float s_local[MAX_DK];
    #pragma unroll
    for (int i = 0; i < MAX_DK; ++i) {
        if (i < dk) {
            s_local[i] = bf16_to_f32(s_base[(size_t)i * dv + tid]);
        }
    }

    // Phase 3A: decayed state + v_pred
    float v_pred = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAX_DK; ++i) {
        if (i < dk) {
            float d = decay * s_local[i];
            s_local[i] = d;
            v_pred += k_smem[i] * d;
        }
    }

    float v_t   = bf16_to_f32(v_base[tid]);
    float delta = beta_t * (v_t - v_pred);

    // Phase 3B: state update + out_acc
    float out_acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAX_DK; ++i) {
        if (i < dk) {
            float new_s = s_local[i] + k_smem[i] * delta;
            s_local[i] = new_s;
            out_acc += q_smem[i] * new_s;
            s_base[(size_t)i * dv + tid] = f32_to_bf16(new_s);
        }
    }

    // ---------------- Phase 4: RMSNorm over dv of out_acc ----------------
    float rms_partial = out_acc * out_acc;
    float rms_total   = block_reduce_sum(rms_partial, reduce);

    if (tid == 0) {
        float mean_sq = rms_total / static_cast<float>(dv);
        bcast[4] = rsqrtf(mean_sq + rms_eps);
    }
    __syncthreads();

    const float rms_inv = bcast[4];

    // ---------------- Phase 5: apply gamma, silu(z), final bf16 write ----------------
    float gamma_t = bf16_to_f32(gamma[tid]);
    float z_t     = bf16_to_f32(z_base[tid]);
    float gate    = silu_f32(z_t);
    float y       = out_acc * rms_inv * gamma_t * gate;

    out_base[tid] = f32_to_bf16(y);
}

} // namespace

extern "C" kiln_gdn_recurrent_status_t kiln_gdn_recurrent_forward(
    const void *q,
    const void *k,
    const void *v,
    const void *beta,
    const void *g,
    void *state,
    void *out,
    int batch_heads,
    int dk,
    int dv,
    void *stream
) {
    if (batch_heads <= 0 || dk <= 0 || dv <= 0) {
        return -1;
    }
    if (dv > 1024) {
        // blockDim.x must equal dv and is capped by CUDA at 1024.
        return -2;
    }

    int threads = dv;
    int blocks  = batch_heads;

    // Shared mem: q (dk F32) + k (dk F32) + 2 scalars.
    size_t smem_bytes = (size_t)(2 * dk + 2) * sizeof(float);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    if (dk <= 128) {
        recurrent_gdn_fwd_kernel<128><<<blocks, threads, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            reinterpret_cast<const __nv_bfloat16 *>(beta),
            reinterpret_cast<const __nv_bfloat16 *>(g),
            reinterpret_cast<__nv_bfloat16 *>(state),
            reinterpret_cast<__nv_bfloat16 *>(out),
            dk,
            dv
        );
    } else if (dk <= 256) {
        recurrent_gdn_fwd_kernel<256><<<blocks, threads, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            reinterpret_cast<const __nv_bfloat16 *>(beta),
            reinterpret_cast<const __nv_bfloat16 *>(g),
            reinterpret_cast<__nv_bfloat16 *>(state),
            reinterpret_cast<__nv_bfloat16 *>(out),
            dk,
            dv
        );
    } else {
        return -3;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return (int)err;
    }
    return 0;
}

extern "C" kiln_gdn_recurrent_status_t kiln_gdn_recurrent_forward_fused_norm(
    const void *q_raw,
    const void *k_raw,
    const void *v,
    const void *beta,
    const void *g,
    const void *z,
    const void *gamma,
    void *state,
    void *out,
    int batch_heads,
    int dk,
    int dv,
    float q_scale,
    float l2_eps,
    float rms_eps,
    void *stream
) {
    if (batch_heads <= 0 || dk <= 0 || dv <= 0) {
        return -1;
    }
    if (dv > 1024) {
        // blockDim.x must equal dv (one thread per output element).
        return -2;
    }

    int threads = dv;
    int blocks  = batch_heads;

    // Shared mem: q (dk F32) + k (dk F32) + kMaxWarps reduce slots + 5 bcast scalars.
    size_t smem_bytes = (size_t)(2 * dk + kMaxWarps + 5) * sizeof(float);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    if (dk <= 128) {
        recurrent_gdn_fwd_fused_norm_kernel<128><<<blocks, threads, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16 *>(q_raw),
            reinterpret_cast<const __nv_bfloat16 *>(k_raw),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            reinterpret_cast<const __nv_bfloat16 *>(beta),
            reinterpret_cast<const __nv_bfloat16 *>(g),
            reinterpret_cast<const __nv_bfloat16 *>(z),
            reinterpret_cast<const __nv_bfloat16 *>(gamma),
            reinterpret_cast<__nv_bfloat16 *>(state),
            reinterpret_cast<__nv_bfloat16 *>(out),
            dk,
            dv,
            q_scale,
            l2_eps,
            rms_eps
        );
    } else if (dk <= 256) {
        recurrent_gdn_fwd_fused_norm_kernel<256><<<blocks, threads, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16 *>(q_raw),
            reinterpret_cast<const __nv_bfloat16 *>(k_raw),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            reinterpret_cast<const __nv_bfloat16 *>(beta),
            reinterpret_cast<const __nv_bfloat16 *>(g),
            reinterpret_cast<const __nv_bfloat16 *>(z),
            reinterpret_cast<const __nv_bfloat16 *>(gamma),
            reinterpret_cast<__nv_bfloat16 *>(state),
            reinterpret_cast<__nv_bfloat16 *>(out),
            dk,
            dv,
            q_scale,
            l2_eps,
            rms_eps
        );
    } else {
        return -3;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return (int)err;
    }
    return 0;
}
