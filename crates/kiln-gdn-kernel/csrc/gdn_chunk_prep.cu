// Kiln GDN chunk-prep fused kernel.
//
// One CUDA block per (batch, head). blockDim.x is picked at launch time
// (1-D block, typically dv so v_prime / q_s_scaled are a single pass per
// thread). Each block:
//
//   1. Loads g[bh, 0..C] into shared memory as F32 and runs a serial
//      cumulative sum in thread 0 (C <= 128 so the scan is a handful of
//      adds).
//   2. Stores p[i] = exp(big_g[i]) and decay_last[i] = exp(big_g[C-1]
//      - big_g[i]) into shared memory / global output.
//   3. Strided-parallel across C*C: writes a_strict = kkt * decay *
//      1[i<t] and b_mask = qkt * decay * 1[i<=t]. decay is recomputed
//      per (t, i) from the shared-memory cumsum so no [C,C] scratch is
//      materialised.
//   4. Strided-parallel across C*dv: writes v_prime = v - ks_entry * p[t]
//      and q_s_scaled = q_s * p[t].
//
// Shared memory budget (C<=128): 2 * 128 * 4 = 1 KiB for the big_g / p
// arrays. Well under the 96 KiB sm_86 carveout.
//
// This replaces 7+ candle op launches per chunk inside
// `gdn_chunkwise_recurrence`:
//   - F32 cumsum
//   - F32 broadcast_sub / exp
//   - cast F32 -> bf16 for decay
//   - exp(big_g) -> p
//   - unsqueeze + broadcast_sub + exp for decay_last_col
//   - broadcast_mul for v - p*ks_entry
//   - broadcast_mul + broadcast_mul for decay * strict_mask and
//     decay * causal_mask
//   - broadcast_mul for kkt * ... and qkt * ...
//   - broadcast_mul for q_s * p

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "gdn_chunk_prep.h"

namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

// MAX_C caps the per-block shared-memory cumsum array. kiln uses C=64.
template <int MAX_C>
__global__ void gdn_chunk_prep_kernel(
    const __nv_bfloat16 *__restrict__ g,          // [B*H, C]
    const __nv_bfloat16 *__restrict__ v,          // [B*H, C, dv]
    const __nv_bfloat16 *__restrict__ kkt,        // [B*H, C, C]
    const __nv_bfloat16 *__restrict__ qkt,        // [B*H, C, C]
    const __nv_bfloat16 *__restrict__ ks_entry,   // [B*H, C, dv]
    const __nv_bfloat16 *__restrict__ q_s,        // [B*H, C, dv]
    __nv_bfloat16 *__restrict__ a_strict,         // [B*H, C, C]
    __nv_bfloat16 *__restrict__ b_mask,           // [B*H, C, C]
    __nv_bfloat16 *__restrict__ v_prime,          // [B*H, C, dv]
    __nv_bfloat16 *__restrict__ q_s_scaled,       // [B*H, C, dv]
    __nv_bfloat16 *__restrict__ decay_last_col,   // [B*H, C]
    __nv_bfloat16 *__restrict__ p_last_out,       // [B*H]
    int C,
    int dv
) {
    const int bh  = blockIdx.x;
    const int tid = threadIdx.x;
    const int bd  = blockDim.x;

    // Shared memory: big_g[C] (F32) + p[C] (F32).
    __shared__ float s_big_g[MAX_C];
    __shared__ float s_p[MAX_C];

    const __nv_bfloat16 *g_base        = g         + (size_t)bh * C;
    const __nv_bfloat16 *v_base        = v         + (size_t)bh * C * dv;
    const __nv_bfloat16 *kkt_base      = kkt       + (size_t)bh * C * C;
    const __nv_bfloat16 *qkt_base      = qkt       + (size_t)bh * C * C;
    const __nv_bfloat16 *ks_base       = ks_entry  + (size_t)bh * C * dv;
    const __nv_bfloat16 *qs_base       = q_s       + (size_t)bh * C * dv;
    __nv_bfloat16       *a_base        = a_strict  + (size_t)bh * C * C;
    __nv_bfloat16       *b_base        = b_mask    + (size_t)bh * C * C;
    __nv_bfloat16       *vp_base       = v_prime   + (size_t)bh * C * dv;
    __nv_bfloat16       *qss_base      = q_s_scaled+ (size_t)bh * C * dv;
    __nv_bfloat16       *dlast_base    = decay_last_col + (size_t)bh * C;

    // 1) Load g -> s_big_g (as F32). Ignore MAX_C slots past C.
    for (int i = tid; i < C; i += bd) {
        s_big_g[i] = bf16_to_f32(g_base[i]);
    }
    __syncthreads();

    // 2) Serial cumsum in thread 0. C <= 128 so this is at most ~128
    //    adds — negligible vs. the 7 global-memory passes this kernel
    //    replaces.
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < C; ++i) {
            acc += s_big_g[i];
            s_big_g[i] = acc;
        }
    }
    __syncthreads();

    const float g_last = s_big_g[C - 1];

    // 3) p[i] = exp(big_g[i]); decay_last_col[i] = exp(g_last - big_g[i]).
    for (int i = tid; i < C; i += bd) {
        float pi = __expf(s_big_g[i]);
        s_p[i]   = pi;
        float dl = __expf(g_last - s_big_g[i]);
        dlast_base[i] = f32_to_bf16(dl);
    }
    if (tid == 0) {
        p_last_out[bh] = f32_to_bf16(__expf(g_last));
    }
    __syncthreads();

    // 4) Fused decay * mask for a_strict and b_mask.
    //      a_strict[t, i] = (i <  t) ? kkt[t,i] * exp(big_g[t]-big_g[i]) : 0
    //      b_mask  [t, i] = (i <= t) ? qkt[t,i] * exp(big_g[t]-big_g[i]) : 0
    const int cc = C * C;
    for (int idx = tid; idx < cc; idx += bd) {
        const int t = idx / C;
        const int i = idx % C;
        const float decay_ti = __expf(s_big_g[t] - s_big_g[i]);
        const float kkt_val  = bf16_to_f32(kkt_base[idx]);
        const float qkt_val  = bf16_to_f32(qkt_base[idx]);
        const float a_val    = (i <  t) ? kkt_val * decay_ti : 0.0f;
        const float b_val    = (i <= t) ? qkt_val * decay_ti : 0.0f;
        a_base[idx] = f32_to_bf16(a_val);
        b_base[idx] = f32_to_bf16(b_val);
    }

    // 5) v_prime[t,d] = v[t,d] - ks_entry[t,d] * p[t]
    //    q_s_scaled[t,d] = q_s[t,d] * p[t]
    const int cdv = C * dv;
    for (int idx = tid; idx < cdv; idx += bd) {
        const int t = idx / dv;
        const float p_t  = s_p[t];
        const float v_val  = bf16_to_f32(v_base[idx]);
        const float ks_val = bf16_to_f32(ks_base[idx]);
        const float qs_val = bf16_to_f32(qs_base[idx]);
        vp_base[idx]  = f32_to_bf16(v_val - ks_val * p_t);
        qss_base[idx] = f32_to_bf16(qs_val * p_t);
    }
}

} // namespace

extern "C" kiln_gdn_chunk_prep_status_t kiln_gdn_chunk_prep(
    const void *g,
    const void *v,
    const void *kkt,
    const void *qkt,
    const void *ks_entry,
    const void *q_s,
    void *a_strict,
    void *b_mask,
    void *v_prime,
    void *q_s_scaled,
    void *decay_last_col,
    void *p_last,
    int batch_heads,
    int chunk_size,
    int dv,
    void *stream
) {
    if (batch_heads <= 0 || chunk_size <= 0 || dv <= 0) {
        return -1;
    }
    if (chunk_size > 128) {
        // MAX_C cap. kiln uses C=64.
        return -2;
    }
    if (dv > 1024) {
        return -3;
    }

    // Pick threads to cover dv in a single pass where possible; clamp to
    // CUDA's 1024 max and minimum 32 so warp scheduling stays sane.
    int threads = dv;
    if (threads > 1024) threads = 1024;
    if (threads < 32)   threads = 32;

    int blocks = batch_heads;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    if (chunk_size <= 64) {
        gdn_chunk_prep_kernel<64><<<blocks, threads, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16 *>(g),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            reinterpret_cast<const __nv_bfloat16 *>(kkt),
            reinterpret_cast<const __nv_bfloat16 *>(qkt),
            reinterpret_cast<const __nv_bfloat16 *>(ks_entry),
            reinterpret_cast<const __nv_bfloat16 *>(q_s),
            reinterpret_cast<__nv_bfloat16 *>(a_strict),
            reinterpret_cast<__nv_bfloat16 *>(b_mask),
            reinterpret_cast<__nv_bfloat16 *>(v_prime),
            reinterpret_cast<__nv_bfloat16 *>(q_s_scaled),
            reinterpret_cast<__nv_bfloat16 *>(decay_last_col),
            reinterpret_cast<__nv_bfloat16 *>(p_last),
            chunk_size,
            dv
        );
    } else {
        gdn_chunk_prep_kernel<128><<<blocks, threads, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16 *>(g),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            reinterpret_cast<const __nv_bfloat16 *>(kkt),
            reinterpret_cast<const __nv_bfloat16 *>(qkt),
            reinterpret_cast<const __nv_bfloat16 *>(ks_entry),
            reinterpret_cast<const __nv_bfloat16 *>(q_s),
            reinterpret_cast<__nv_bfloat16 *>(a_strict),
            reinterpret_cast<__nv_bfloat16 *>(b_mask),
            reinterpret_cast<__nv_bfloat16 *>(v_prime),
            reinterpret_cast<__nv_bfloat16 *>(q_s_scaled),
            reinterpret_cast<__nv_bfloat16 *>(decay_last_col),
            reinterpret_cast<__nv_bfloat16 *>(p_last),
            chunk_size,
            dv
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return (int)err;
    }
    return 0;
}
