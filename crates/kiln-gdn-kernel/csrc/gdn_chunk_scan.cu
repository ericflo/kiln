// Kiln GDN fused chunk-body kernel.
//
// One CUDA block per (batch, head). The kernel consumes the chunk-prep
// outputs already materialized by `gdn_chunk_prep` and finishes the
// chunk-local recurrence:
//   1. forward-substitution for W[t]
//   2. intra = B_mask @ W
//   3. out = q_s_scaled + intra
//   4. w_weighted = W * decay_last_col
//
// The final state update still uses cuBLAS for K^T @ w_weighted.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "gdn_chunk_scan.h"

namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

template <int MAX_C>
__global__ void gdn_chunk_scan_kernel(
    const __nv_bfloat16 *__restrict__ a_strict,
    const __nv_bfloat16 *__restrict__ b_mask,
    const __nv_bfloat16 *__restrict__ v_prime,
    const __nv_bfloat16 *__restrict__ q_s_scaled,
    const __nv_bfloat16 *__restrict__ beta,
    const __nv_bfloat16 *__restrict__ decay_last_col,
    __nv_bfloat16 *__restrict__ out_chunk,
    __nv_bfloat16 *__restrict__ w_weighted,
    int chunk_size,
    int dv
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= dv) return;

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16 *s_a = smem;
    __nv_bfloat16 *s_b = s_a + (size_t)chunk_size * chunk_size;
    __nv_bfloat16 *s_w = s_b + (size_t)chunk_size * chunk_size;

    const __nv_bfloat16 *a_base = a_strict + (size_t)bh * chunk_size * chunk_size;
    const __nv_bfloat16 *b_base = b_mask + (size_t)bh * chunk_size * chunk_size;
    const __nv_bfloat16 *vp_base = v_prime + (size_t)bh * chunk_size * dv;
    const __nv_bfloat16 *qss_base = q_s_scaled + (size_t)bh * chunk_size * dv;
    const __nv_bfloat16 *beta_base = beta + (size_t)bh * chunk_size;
    const __nv_bfloat16 *dlast_base = decay_last_col + (size_t)bh * chunk_size;
    __nv_bfloat16 *out_base = out_chunk + (size_t)bh * chunk_size * dv;
    __nv_bfloat16 *ww_base = w_weighted + (size_t)bh * chunk_size * dv;

    const int total_cc = chunk_size * chunk_size;
    for (int i = tid; i < total_cc; i += blockDim.x) {
        s_a[i] = a_base[i];
        s_b[i] = b_base[i];
    }
    __syncthreads();

    for (int t = 0; t < chunk_size; ++t) {
        const float beta_t = bf16_to_f32(beta_base[t]);
        const float decay_last_t = bf16_to_f32(dlast_base[t]);
        const __nv_bfloat16 *a_row = s_a + (size_t)t * chunk_size;
        const __nv_bfloat16 *b_row = s_b + (size_t)t * chunk_size;
        const __nv_bfloat16 *vp_row = vp_base + (size_t)t * dv;
        const __nv_bfloat16 *qss_row = qss_base + (size_t)t * dv;
        __nv_bfloat16 *sw_row = s_w + (size_t)t * dv;
        __nv_bfloat16 *out_row = out_base + (size_t)t * dv;
        __nv_bfloat16 *ww_row = ww_base + (size_t)t * dv;

        float acc_a = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_C; ++i) {
            if (i < t) {
                acc_a += bf16_to_f32(a_row[i]) * bf16_to_f32(s_w[(size_t)i * dv + tid]);
            }
        }

        const float vp_val = bf16_to_f32(vp_row[tid]);
        const float w_val = beta_t * (vp_val - acc_a);
        const __nv_bfloat16 w_bf = f32_to_bf16(w_val);
        sw_row[tid] = w_bf;
        __syncthreads();

        float acc_b = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_C; ++i) {
            if (i <= t) {
                acc_b += bf16_to_f32(b_row[i]) * bf16_to_f32(s_w[(size_t)i * dv + tid]);
            }
        }

        const float out_val = bf16_to_f32(qss_row[tid]) + acc_b;
        out_row[tid] = f32_to_bf16(out_val);
        ww_row[tid] = f32_to_bf16(w_val * decay_last_t);
        __syncthreads();
    }
}

} // namespace

extern "C" kiln_gdn_chunk_scan_status_t kiln_gdn_chunk_scan(
    const void *a_strict,
    const void *b_mask,
    const void *v_prime,
    const void *q_s_scaled,
    const void *beta,
    const void *decay_last_col,
    void *out_chunk,
    void *w_weighted,
    int batch_heads,
    int chunk_size,
    int dv,
    void *stream
) {
    if (batch_heads <= 0 || chunk_size <= 0 || dv <= 0) return -1;
    if (chunk_size > 64 || dv > 128) return -2;

    const int threads = 128;
    const int blocks = batch_heads;
    const size_t smem_bytes =
        ((size_t)chunk_size * chunk_size * 2 + (size_t)chunk_size * dv) *
        sizeof(__nv_bfloat16);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    gdn_chunk_scan_kernel<64><<<blocks, threads, smem_bytes, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(a_strict),
        reinterpret_cast<const __nv_bfloat16 *>(b_mask),
        reinterpret_cast<const __nv_bfloat16 *>(v_prime),
        reinterpret_cast<const __nv_bfloat16 *>(q_s_scaled),
        reinterpret_cast<const __nv_bfloat16 *>(beta),
        reinterpret_cast<const __nv_bfloat16 *>(decay_last_col),
        reinterpret_cast<__nv_bfloat16 *>(out_chunk),
        reinterpret_cast<__nv_bfloat16 *>(w_weighted),
        chunk_size,
        dv
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    return 0;
}
