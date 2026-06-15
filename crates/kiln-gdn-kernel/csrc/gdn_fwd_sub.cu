// Kiln GDN fused forward-substitution kernel.
//
// One CUDA block per (batch, head) slot. Each block:
//   1. Loads A_strict ([C, C]) into shared memory.
//   2. Walks t = 0..C-1 sequentially. For each t, every thread cooperates
//      across the dv axis to compute the row sum and writes W[t, :] back
//      to shared memory (also out to global memory).
//
// Per-block resource budget (kiln config: C=64, dv=128, bf16):
//   shared mem = (C*C + C*dv) * 2 = (4096 + 8192) * 2 = 24 KiB
//   threads    = dv = 128 (one element per thread; 4 warps)
//
// Sized to fit comfortably under the A6000 (sm_86) per-block shared
// memory cap. We deliberately cap chunk_size <= 128 / dv <= 1024 so the
// shared allocation stays bounded.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "gdn_fwd_sub.h"

namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

// Generic forward-substitution kernel. `dv_per_thread` lets us keep the
// kernel correct even if dv != blockDim.x; in production blockDim.x = dv
// so every thread owns exactly one column.
__global__ void gdn_fwd_sub_kernel(
    const __nv_bfloat16 *__restrict__ a_strict,  // [B*H, C, C]
    const __nv_bfloat16 *__restrict__ v_prime,   // [B*H, C, dv]
    const __nv_bfloat16 *__restrict__ beta,      // [B*H, C]
    __nv_bfloat16 *__restrict__ w_out,           // [B*H, C, dv]
    int chunk_size,
    int dv
) {
    const int bh  = blockIdx.x;
    const int tid = threadIdx.x;
    const int blk = blockDim.x;

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16 *sA = smem;                                  // C*C
    __nv_bfloat16 *sW = smem + (size_t)chunk_size * chunk_size; // C*dv

    const __nv_bfloat16 *a_base    = a_strict + (size_t)bh * chunk_size * chunk_size;
    const __nv_bfloat16 *v_base    = v_prime  + (size_t)bh * chunk_size * dv;
    const __nv_bfloat16 *beta_base = beta     + (size_t)bh * chunk_size;
    __nv_bfloat16       *w_base    = w_out    + (size_t)bh * chunk_size * dv;

    // 1) Load A_strict into shared memory.
    const int total_a = chunk_size * chunk_size;
    for (int i = tid; i < total_a; i += blk) {
        sA[i] = a_base[i];
    }
    __syncthreads();

    // 2) Sequential rows; parallel across dv columns.
    const int dv_per_thread = (dv + blk - 1) / blk;

    for (int t = 0; t < chunk_size; ++t) {
        const float beta_t = bf16_to_f32(beta_base[t]);
        const __nv_bfloat16 *a_row = sA + (size_t)t * chunk_size;
        const __nv_bfloat16 *vp_row = v_base + (size_t)t * dv;
        __nv_bfloat16 *sw_row = sW + (size_t)t * dv;
        __nv_bfloat16 *gw_row = w_base + (size_t)t * dv;

        for (int j = 0; j < dv_per_thread; ++j) {
            const int d = tid + j * blk;
            if (d >= dv) break;

            // sum_{i<t} A_strict[t, i] * W[i, d]
            float acc = 0.0f;
            #pragma unroll 4
            for (int i = 0; i < t; ++i) {
                float a = bf16_to_f32(a_row[i]);
                float w = bf16_to_f32(sW[(size_t)i * dv + d]);
                acc += a * w;
            }

            float vp_val = bf16_to_f32(vp_row[d]);
            float w_val  = beta_t * (vp_val - acc);

            __nv_bfloat16 w_bf = f32_to_bf16(w_val);
            sw_row[d] = w_bf;
            gw_row[d] = w_bf;
        }
        __syncthreads();
    }
}

__global__ void gdn_fwd_sub_f32_kernel(
    const float *__restrict__ a_strict,  // [B*H, C, C]
    const float *__restrict__ v_prime,   // [B*H, C, dv]
    const float *__restrict__ beta,      // [B*H, C]
    float *__restrict__ w_out,           // [B*H, C, dv]
    int chunk_size,
    int dv
) {
    constexpr int DV_TILE = 64;
    const int bh = blockIdx.x;
    const int dv_tile = blockIdx.y;
    const int tid = threadIdx.x;
    const int blk = blockDim.x;

    const int dv_start = dv_tile * DV_TILE;
    if (dv_start >= dv) {
        return;
    }
    const int dv_count = min(DV_TILE, dv - dv_start);

    extern __shared__ float smem_f32[];
    float *sA = smem_f32;                                         // C*C
    float *sW = smem_f32 + (size_t)chunk_size * chunk_size;       // C*DV_TILE

    const float *a_base = a_strict + (size_t)bh * chunk_size * chunk_size;
    const float *v_base = v_prime + (size_t)bh * chunk_size * dv;
    const float *beta_base = beta + (size_t)bh * chunk_size;
    float *w_base = w_out + (size_t)bh * chunk_size * dv;

    const int total_a = chunk_size * chunk_size;
    for (int i = tid; i < total_a; i += blk) {
        sA[i] = a_base[i];
    }
    __syncthreads();

    for (int t = 0; t < chunk_size; ++t) {
        if (tid < dv_count) {
            float acc = 0.0f;
            const float *a_row = sA + (size_t)t * chunk_size;
            for (int i = 0; i < t; ++i) {
                acc += a_row[i] * sW[(size_t)i * DV_TILE + tid];
            }
            const int d = dv_start + tid;
            const float w_val = beta_base[t] * (v_base[(size_t)t * dv + d] - acc);
            sW[(size_t)t * DV_TILE + tid] = w_val;
            w_base[(size_t)t * dv + d] = w_val;
        }
        __syncthreads();
    }
}

__global__ void gdn_solve_tri_transpose_f32_kernel(
    const float *__restrict__ a_strict,  // [B*H, C, C]
    const float *__restrict__ beta,      // [B*H, C]
    const float *__restrict__ dw,        // [B*H, C, dv]
    float *__restrict__ dr_out,          // [B*H, C, dv]
    int chunk_size,
    int dv
) {
    constexpr int DV_TILE = 64;
    const int bh = blockIdx.x;
    const int dv_tile = blockIdx.y;
    const int tid = threadIdx.x;
    const int blk = blockDim.x;

    const int dv_start = dv_tile * DV_TILE;
    if (dv_start >= dv) {
        return;
    }
    const int dv_count = min(DV_TILE, dv - dv_start);

    extern __shared__ float smem_f32[];
    float *sA = smem_f32;                                         // C*C
    float *sR = smem_f32 + (size_t)chunk_size * chunk_size;       // C*DV_TILE

    const float *a_base = a_strict + (size_t)bh * chunk_size * chunk_size;
    const float *beta_base = beta + (size_t)bh * chunk_size;
    const float *dw_base = dw + (size_t)bh * chunk_size * dv;
    float *dr_base = dr_out + (size_t)bh * chunk_size * dv;

    const int total_a = chunk_size * chunk_size;
    for (int i = tid; i < total_a; i += blk) {
        sA[i] = a_base[i];
    }
    __syncthreads();

    for (int t = chunk_size - 1; t >= 0; --t) {
        if (tid < dv_count) {
            float acc = 0.0f;
            for (int i = t + 1; i < chunk_size; ++i) {
                acc += beta_base[i]
                    * sA[(size_t)i * chunk_size + t]
                    * sR[(size_t)i * DV_TILE + tid];
            }
            const int d = dv_start + tid;
            const float dr_val = dw_base[(size_t)t * dv + d] - acc;
            sR[(size_t)t * DV_TILE + tid] = dr_val;
            dr_base[(size_t)t * dv + d] = dr_val;
        }
        __syncthreads();
    }
}

} // namespace

extern "C" kiln_gdn_status_t kiln_gdn_forward_substitution(
    const void *a_strict,
    const void *v_prime,
    const void *beta,
    void *w_out,
    int batch_heads,
    int chunk_size,
    int dv,
    void *stream
) {
    if (batch_heads <= 0 || chunk_size <= 0 || dv <= 0) {
        return -1;
    }
    if (chunk_size > 128) {
        // Shared mem grows as C*(C + dv) bf16; cap to keep within sm_80+
        // budget. kiln currently uses C=64.
        return -2;
    }

    int threads = dv;
    if (threads > 1024) threads = 1024;
    if (threads < 32)   threads = 32;

    int blocks = batch_heads;

    size_t smem_bytes =
        ((size_t)chunk_size * chunk_size + (size_t)chunk_size * dv)
        * sizeof(__nv_bfloat16);

    if (smem_bytes > 96 * 1024) {
        // Opt-in to the larger dynamic shared mem carveout. On sm_86 this
        // can reach 100 KB; we expect to stay well under that.
        cudaFuncSetAttribute(
            gdn_fwd_sub_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes
        );
    }

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    gdn_fwd_sub_kernel<<<blocks, threads, smem_bytes, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(a_strict),
        reinterpret_cast<const __nv_bfloat16 *>(v_prime),
        reinterpret_cast<const __nv_bfloat16 *>(beta),
        reinterpret_cast<__nv_bfloat16 *>(w_out),
        chunk_size,
        dv
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return (int)err;
    }
    return 0;
}

extern "C" kiln_gdn_status_t kiln_gdn_forward_substitution_f32(
    const void *a_strict,
    const void *v_prime,
    const void *beta,
    void *w_out,
    int batch_heads,
    int chunk_size,
    int dv,
    void *stream
) {
    if (batch_heads <= 0 || chunk_size <= 0 || dv <= 0) {
        return -1;
    }
    if (chunk_size > 128) {
        return -2;
    }

    constexpr int DV_TILE = 64;
    dim3 blocks(batch_heads, (dv + DV_TILE - 1) / DV_TILE);
    int threads = DV_TILE;
    size_t smem_bytes =
        ((size_t)chunk_size * chunk_size + (size_t)chunk_size * DV_TILE)
        * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            gdn_fwd_sub_f32_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes
        );
    }

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    gdn_fwd_sub_f32_kernel<<<blocks, threads, smem_bytes, s>>>(
        reinterpret_cast<const float *>(a_strict),
        reinterpret_cast<const float *>(v_prime),
        reinterpret_cast<const float *>(beta),
        reinterpret_cast<float *>(w_out),
        chunk_size,
        dv
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return (int)err;
    }
    return 0;
}

extern "C" kiln_gdn_status_t kiln_gdn_solve_tri_transpose_f32(
    const void *a_strict,
    const void *beta,
    const void *dw,
    void *dr_out,
    int batch_heads,
    int chunk_size,
    int dv,
    void *stream
) {
    if (batch_heads <= 0 || chunk_size <= 0 || dv <= 0) {
        return -1;
    }
    if (chunk_size > 128) {
        return -2;
    }

    constexpr int DV_TILE = 64;
    dim3 blocks(batch_heads, (dv + DV_TILE - 1) / DV_TILE);
    int threads = DV_TILE;
    size_t smem_bytes =
        ((size_t)chunk_size * chunk_size + (size_t)chunk_size * DV_TILE)
        * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            gdn_solve_tri_transpose_f32_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes
        );
    }

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    gdn_solve_tri_transpose_f32_kernel<<<blocks, threads, smem_bytes, s>>>(
        reinterpret_cast<const float *>(a_strict),
        reinterpret_cast<const float *>(beta),
        reinterpret_cast<const float *>(dw),
        reinterpret_cast<float *>(dr_out),
        chunk_size,
        dv
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return (int)err;
    }
    return 0;
}
