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
