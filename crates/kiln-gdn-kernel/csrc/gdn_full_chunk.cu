// Kiln GDN fused full-chunk prefill kernel.
//
// This is the next minimal CUDA-owned step after `gdn_chunk_prep` and
// `gdn_chunk_scan`: for the exact kiln full-chunk envelope
// (chunk_size=64, dk=dv=128, bf16) it finishes chunk prep, the chunk-local
// recurrence, and the state update in one launch per (batch, head).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "gdn_full_chunk.h"

namespace {

constexpr int KILN_GDN_FULL_C = 64;
constexpr int KILN_GDN_FULL_D = 128;

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

__global__ void gdn_full_chunk_prefill_kernel(
    const __nv_bfloat16 *__restrict__ g,
    const __nv_bfloat16 *__restrict__ v,
    const __nv_bfloat16 *__restrict__ kkt,
    const __nv_bfloat16 *__restrict__ qkt,
    const __nv_bfloat16 *__restrict__ ks_entry,
    const __nv_bfloat16 *__restrict__ q_s,
    const __nv_bfloat16 *__restrict__ beta,
    const __nv_bfloat16 *__restrict__ k_t,
    __nv_bfloat16 *__restrict__ state,
    __nv_bfloat16 *__restrict__ out_chunk
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= KILN_GDN_FULL_D) return;

    __shared__ float s_big_g[KILN_GDN_FULL_C];
    __shared__ float s_p[KILN_GDN_FULL_C];
    __shared__ __nv_bfloat16 s_a[KILN_GDN_FULL_C * KILN_GDN_FULL_C];
    __shared__ __nv_bfloat16 s_b[KILN_GDN_FULL_C * KILN_GDN_FULL_C];
    __shared__ __nv_bfloat16 s_w[KILN_GDN_FULL_C * KILN_GDN_FULL_D];
    __shared__ __nv_bfloat16 s_k_t[KILN_GDN_FULL_D * KILN_GDN_FULL_C];

    const __nv_bfloat16 *g_base = g + (size_t)bh * KILN_GDN_FULL_C;
    const __nv_bfloat16 *v_base = v + (size_t)bh * KILN_GDN_FULL_C * KILN_GDN_FULL_D;
    const __nv_bfloat16 *kkt_base = kkt + (size_t)bh * KILN_GDN_FULL_C * KILN_GDN_FULL_C;
    const __nv_bfloat16 *qkt_base = qkt + (size_t)bh * KILN_GDN_FULL_C * KILN_GDN_FULL_C;
    const __nv_bfloat16 *ks_base = ks_entry + (size_t)bh * KILN_GDN_FULL_C * KILN_GDN_FULL_D;
    const __nv_bfloat16 *qs_base = q_s + (size_t)bh * KILN_GDN_FULL_C * KILN_GDN_FULL_D;
    const __nv_bfloat16 *beta_base = beta + (size_t)bh * KILN_GDN_FULL_C;
    const __nv_bfloat16 *kt_base = k_t + (size_t)bh * KILN_GDN_FULL_D * KILN_GDN_FULL_C;
    __nv_bfloat16 *state_base = state + (size_t)bh * KILN_GDN_FULL_D * KILN_GDN_FULL_D;
    __nv_bfloat16 *out_base = out_chunk + (size_t)bh * KILN_GDN_FULL_C * KILN_GDN_FULL_D;

    for (int i = tid; i < KILN_GDN_FULL_C; i += blockDim.x) {
        s_big_g[i] = bf16_to_f32(g_base[i]);
    }
    for (int i = tid; i < KILN_GDN_FULL_D * KILN_GDN_FULL_C; i += blockDim.x) {
        s_k_t[i] = kt_base[i];
    }
    __syncthreads();

    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < KILN_GDN_FULL_C; ++i) {
            acc += s_big_g[i];
            s_big_g[i] = acc;
            s_p[i] = __expf(acc);
        }
    }
    __syncthreads();

    const float g_last = s_big_g[KILN_GDN_FULL_C - 1];
    const float p_last = s_p[KILN_GDN_FULL_C - 1];

    for (int idx = tid; idx < KILN_GDN_FULL_C * KILN_GDN_FULL_C; idx += blockDim.x) {
        const int t = idx / KILN_GDN_FULL_C;
        const int i = idx % KILN_GDN_FULL_C;
        const float decay = __expf(s_big_g[t] - s_big_g[i]);
        const float kkt_val = bf16_to_f32(kkt_base[idx]);
        const float qkt_val = bf16_to_f32(qkt_base[idx]);
        s_a[idx] = f32_to_bf16(i < t ? kkt_val * decay : 0.0f);
        s_b[idx] = f32_to_bf16(i <= t ? qkt_val * decay : 0.0f);
    }
    __syncthreads();

    for (int t = 0; t < KILN_GDN_FULL_C; ++t) {
        const float beta_t = bf16_to_f32(beta_base[t]);
        const float p_t = s_p[t];
        const float decay_last_t = __expf(g_last - s_big_g[t]);

        const __nv_bfloat16 *a_row = s_a + (size_t)t * KILN_GDN_FULL_C;
        const __nv_bfloat16 *b_row = s_b + (size_t)t * KILN_GDN_FULL_C;
        const int row_off = t * KILN_GDN_FULL_D + tid;

        float vp_val = bf16_to_f32(v_base[row_off]) - bf16_to_f32(ks_base[row_off]) * p_t;
        float qss_val = bf16_to_f32(qs_base[row_off]) * p_t;

        float acc_a = 0.0f;
        #pragma unroll
        for (int i = 0; i < KILN_GDN_FULL_C; ++i) {
            if (i < t) {
                acc_a += bf16_to_f32(a_row[i]) * bf16_to_f32(s_w[(size_t)i * KILN_GDN_FULL_D + tid]);
            }
        }

        const float w_val = beta_t * (vp_val - acc_a);
        s_w[(size_t)t * KILN_GDN_FULL_D + tid] = f32_to_bf16(w_val);
        __syncthreads();

        float acc_b = 0.0f;
        #pragma unroll
        for (int i = 0; i < KILN_GDN_FULL_C; ++i) {
            if (i <= t) {
                acc_b += bf16_to_f32(b_row[i]) * bf16_to_f32(s_w[(size_t)i * KILN_GDN_FULL_D + tid]);
            }
        }

        out_base[row_off] = f32_to_bf16(qss_val + acc_b);
        s_w[(size_t)t * KILN_GDN_FULL_D + tid] = f32_to_bf16(w_val * decay_last_t);
        __syncthreads();
    }

    for (int dk_idx = 0; dk_idx < KILN_GDN_FULL_D; ++dk_idx) {
        float acc = 0.0f;
        #pragma unroll
        for (int t = 0; t < KILN_GDN_FULL_C; ++t) {
            acc += bf16_to_f32(s_k_t[(size_t)dk_idx * KILN_GDN_FULL_C + t]) *
                   bf16_to_f32(s_w[(size_t)t * KILN_GDN_FULL_D + tid]);
        }
        const size_t state_idx = (size_t)dk_idx * KILN_GDN_FULL_D + tid;
        const float state_old = bf16_to_f32(state_base[state_idx]);
        state_base[state_idx] = f32_to_bf16(state_old * p_last + acc);
    }
}

} // namespace

extern "C" kiln_gdn_full_chunk_status_t kiln_gdn_full_chunk_prefill(
    const void *g,
    const void *v,
    const void *kkt,
    const void *qkt,
    const void *ks_entry,
    const void *q_s,
    const void *beta,
    const void *k_t,
    void *state,
    void *out_chunk,
    int batch_heads,
    int chunk_size,
    int dk,
    int dv,
    void *stream
) {
    if (batch_heads <= 0) return -1;
    if (chunk_size != KILN_GDN_FULL_C || dk != KILN_GDN_FULL_D || dv != KILN_GDN_FULL_D) {
        return -2;
    }

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    gdn_full_chunk_prefill_kernel<<<batch_heads, KILN_GDN_FULL_D, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(g),
        reinterpret_cast<const __nv_bfloat16 *>(v),
        reinterpret_cast<const __nv_bfloat16 *>(kkt),
        reinterpret_cast<const __nv_bfloat16 *>(qkt),
        reinterpret_cast<const __nv_bfloat16 *>(ks_entry),
        reinterpret_cast<const __nv_bfloat16 *>(q_s),
        reinterpret_cast<const __nv_bfloat16 *>(beta),
        reinterpret_cast<const __nv_bfloat16 *>(k_t),
        reinterpret_cast<__nv_bfloat16 *>(state),
        reinterpret_cast<__nv_bfloat16 *>(out_chunk)
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    return 0;
}
