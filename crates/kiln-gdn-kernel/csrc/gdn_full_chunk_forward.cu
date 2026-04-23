// Kiln GDN fused full-chunk prefill kernel.
//
// Scope is intentionally tight:
// - CUDA only
// - bf16 only
// - full 64-token prefill chunks only
// - Qwen3.5-4B GDN envelope (dk <= 128, dv <= 128)
//
// The four GEMMs surrounding the chunk path stay on cuBLAS in Rust
// (`kkt`, `qkt`, `ks_entry`, `q_s`). This kernel owns the remaining hot
// chunk-local orchestration:
//   1. big_g / p / decay / strict+causal masking
//   2. forward substitution for W
//   3. intra-chunk output accumulation
//   4. recurrent state update

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "gdn_full_chunk_forward.h"

namespace {

constexpr int kChunkSize = 64;
constexpr int kMaxDv = 128;

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

__global__ void gdn_full_chunk_forward_kernel(
    const __nv_bfloat16 *__restrict__ g,
    const __nv_bfloat16 *__restrict__ v,
    const __nv_bfloat16 *__restrict__ kkt,
    const __nv_bfloat16 *__restrict__ qkt,
    const __nv_bfloat16 *__restrict__ ks_entry,
    const __nv_bfloat16 *__restrict__ q_s,
    const __nv_bfloat16 *__restrict__ beta,
    const __nv_bfloat16 *__restrict__ k_t,
    __nv_bfloat16 *__restrict__ state,
    __nv_bfloat16 *__restrict__ out_chunk,
    int dk,
    int dv
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= dv) {
        return;
    }

    extern __shared__ unsigned char smem_raw[];
    unsigned char *smem = smem_raw;
    __nv_bfloat16 *s_a = reinterpret_cast<__nv_bfloat16 *>(smem);    // [C, C]
    smem += (size_t)kChunkSize * kChunkSize * sizeof(__nv_bfloat16);
    __nv_bfloat16 *s_b = reinterpret_cast<__nv_bfloat16 *>(smem);    // [C, C]
    smem += (size_t)kChunkSize * kChunkSize * sizeof(__nv_bfloat16);
    uintptr_t smem_aligned = (reinterpret_cast<uintptr_t>(smem) + alignof(float) - 1) &
        ~(uintptr_t)(alignof(float) - 1);
    float *s_w = reinterpret_cast<float *>(smem_aligned);            // [C, dv]

    __shared__ float s_big_g[kChunkSize];
    __shared__ float s_p[kChunkSize];
    __shared__ float s_decay_last[kChunkSize];
    __shared__ float s_p_last;

    const __nv_bfloat16 *g_base = g + (size_t)bh * kChunkSize;
    const __nv_bfloat16 *v_base = v + (size_t)bh * kChunkSize * dv;
    const __nv_bfloat16 *kkt_base = kkt + (size_t)bh * kChunkSize * kChunkSize;
    const __nv_bfloat16 *qkt_base = qkt + (size_t)bh * kChunkSize * kChunkSize;
    const __nv_bfloat16 *ks_base = ks_entry + (size_t)bh * kChunkSize * dv;
    const __nv_bfloat16 *qs_base = q_s + (size_t)bh * kChunkSize * dv;
    const __nv_bfloat16 *beta_base = beta + (size_t)bh * kChunkSize;
    const __nv_bfloat16 *kt_base = k_t + (size_t)bh * dk * kChunkSize;
    __nv_bfloat16 *state_base = state + (size_t)bh * dk * dv;
    __nv_bfloat16 *out_base = out_chunk + (size_t)bh * kChunkSize * dv;

    for (int i = tid; i < kChunkSize; i += blockDim.x) {
        s_big_g[i] = bf16_to_f32(g_base[i]);
    }
    __syncthreads();

    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < kChunkSize; ++i) {
            acc += s_big_g[i];
            s_big_g[i] = acc;
        }
        s_p_last = __expf(s_big_g[kChunkSize - 1]);
    }
    __syncthreads();

    for (int i = tid; i < kChunkSize; i += blockDim.x) {
        const float p = __expf(s_big_g[i]);
        s_p[i] = p;
        s_decay_last[i] = __expf(s_big_g[kChunkSize - 1] - s_big_g[i]);
    }

    const int total_cc = kChunkSize * kChunkSize;
    for (int idx = tid; idx < total_cc; idx += blockDim.x) {
        const int t = idx / kChunkSize;
        const int i = idx % kChunkSize;
        const float decay = __expf(s_big_g[t] - s_big_g[i]);
        const float a_val = (i < t) ? bf16_to_f32(kkt_base[idx]) * decay : 0.0f;
        const float b_val = (i <= t) ? bf16_to_f32(qkt_base[idx]) * decay : 0.0f;
        s_a[idx] = f32_to_bf16(a_val);
        s_b[idx] = f32_to_bf16(b_val);
    }
    __syncthreads();

    for (int t = 0; t < kChunkSize; ++t) {
        const float beta_t = bf16_to_f32(beta_base[t]);
        const float p_t = s_p[t];

        const __nv_bfloat16 *a_row = s_a + (size_t)t * kChunkSize;
        const __nv_bfloat16 *b_row = s_b + (size_t)t * kChunkSize;
        float *w_row = s_w + (size_t)t * dv;

        float acc_a = 0.0f;
        #pragma unroll
        for (int i = 0; i < kChunkSize; ++i) {
            if (i < t) {
                acc_a += bf16_to_f32(a_row[i]) * s_w[(size_t)i * dv + tid];
            }
        }

        const size_t td = (size_t)t * dv + tid;
        const float vp = bf16_to_f32(v_base[td]) - bf16_to_f32(ks_base[td]) * p_t;
        const float w_val = beta_t * (vp - acc_a);
        w_row[tid] = w_val;
        __syncthreads();

        float acc_b = 0.0f;
        #pragma unroll
        for (int i = 0; i < kChunkSize; ++i) {
            if (i <= t) {
                acc_b += bf16_to_f32(b_row[i]) * s_w[(size_t)i * dv + tid];
            }
        }

        const float qss = bf16_to_f32(qs_base[td]) * p_t;
        const float out_val = qss + acc_b;
        out_base[td] = f32_to_bf16(out_val);
        __syncthreads();
    }

    const float p_last = s_p_last;
    for (int k_idx = 0; k_idx < dk; ++k_idx) {
        float delta = 0.0f;
        #pragma unroll
        for (int t = 0; t < kChunkSize; ++t) {
            const float kt = bf16_to_f32(kt_base[(size_t)k_idx * kChunkSize + t]);
            const float w = s_w[(size_t)t * dv + tid];
            const float decay_last = s_decay_last[t];
            delta += kt * (w * decay_last);
        }
        const size_t state_idx = (size_t)k_idx * dv + tid;
        const float prev = bf16_to_f32(state_base[state_idx]);
        state_base[state_idx] = f32_to_bf16(prev * p_last + delta);
    }
}

} // namespace

extern "C" kiln_gdn_full_chunk_forward_status_t kiln_gdn_full_chunk_forward(
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
    if (batch_heads <= 0 || chunk_size != kChunkSize || dk <= 0 || dv <= 0) {
        return -1;
    }
    if (dk > 128 || dv > kMaxDv) {
        return -2;
    }

    const int blocks = batch_heads;
    const int threads = 128;
    const size_t smem_bytes =
        (size_t)kChunkSize * kChunkSize * 2 * sizeof(__nv_bfloat16) +
        (size_t)kChunkSize * dv * sizeof(float) +
        alignof(float);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    gdn_full_chunk_forward_kernel<<<blocks, threads, smem_bytes, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(g),
        reinterpret_cast<const __nv_bfloat16 *>(v),
        reinterpret_cast<const __nv_bfloat16 *>(kkt),
        reinterpret_cast<const __nv_bfloat16 *>(qkt),
        reinterpret_cast<const __nv_bfloat16 *>(ks_entry),
        reinterpret_cast<const __nv_bfloat16 *>(q_s),
        reinterpret_cast<const __nv_bfloat16 *>(beta),
        reinterpret_cast<const __nv_bfloat16 *>(k_t),
        reinterpret_cast<__nv_bfloat16 *>(state),
        reinterpret_cast<__nv_bfloat16 *>(out_chunk),
        dk,
        dv
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return (int)err;
    }
    return 0;
}
