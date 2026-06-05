// Paged-decode metadata kernels (Phase R.9 prerequisite): compute the paged
// gather index and the per-batch tail mask ON-DEVICE from device-resident
// block_table / seqused_k buffers, so the ROCm paged-decode attention path no
// longer stages this metadata to the host (a D2H + H2D round-trip per attention
// layer that both costs sync latency and prevents HIP graph capture).
//
// Both kernels are pure element-wise index math: one thread per output element,
// no cross-lane reductions, so there is no wave32/wave64 hazard (unlike the
// reduction kernels that needed the shared-memory block-reduce fix in R.5).
//
// Compiled by both nvcc (CUDA) and hipcc (ROCm) — `<cuda_runtime.h>` and
// `cudaStream_t` hipify cleanly. The same compiled object services both
// backends, matching the rest of csrc/.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#include "kt_gpu_compat.cuh"

#define BLOCK_SIZE 256

namespace {

// out_idx[bi*seqlen_k + t] = block_table[bi*max_blocks_per_seq + t/page]*page
//                            + t%page
// (`page` == page_block_size). Mirrors the host loop in
// `rocm_sdpa.rs::paged_gather` exactly.
__global__ void kiln_paged_gather_index_kernel(const uint32_t* __restrict__ block_table,
                                               uint32_t* __restrict__ out_idx,
                                               int64_t b,
                                               int64_t seqlen_k,
                                               int64_t max_blocks_per_seq,
                                               int64_t page_block_size) {
    int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = b * seqlen_k;
    if (gid >= total) return;

    int64_t bi = gid / seqlen_k;
    int64_t t = gid % seqlen_k;
    int64_t blk = t / page_block_size;
    int64_t within = t % page_block_size;
    uint32_t phys_block = block_table[bi * max_blocks_per_seq + blk];
    out_idx[gid] = static_cast<uint32_t>(phys_block) * static_cast<uint32_t>(page_block_size)
                   + static_cast<uint32_t>(within);
}

// out_mask[(bi*h + hi)*sk + j] = (j >= seqused_k[bi]) ? 1 : 0   (u8)
// Flattened [b*h, 1, sk] (sq == 1 for decode). Mirrors the host tail-mask loop
// in `rocm_sdpa.rs::sdpa_forward_dyn_tail`.
__global__ void kiln_paged_tail_mask_kernel(const uint32_t* __restrict__ seqused_k,
                                            uint8_t* __restrict__ out_mask,
                                            int64_t b,
                                            int64_t h,
                                            int64_t sk) {
    int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = b * h * sk;
    if (gid >= total) return;

    int64_t bi = gid / (h * sk);
    int64_t rem = gid % (h * sk);
    int64_t j = rem % sk;
    uint32_t used = seqused_k[bi];
    out_mask[gid] = (static_cast<uint32_t>(j) >= used) ? static_cast<uint8_t>(1)
                                                       : static_cast<uint8_t>(0);
}

// Directly gather token-major paged KV rows from
//   pool[pool_rows, hk, d]
// into
//   out[b, seqlen_k, hk, d]
// using block_table[b, max_blocks_per_seq]. One thread copies one logical
// element (1/2/4 bytes depending on dtype). Invalid physical rows are zeroed,
// matching index_select_dim0's out-of-range behavior without requiring a
// separate zero-fill.
__global__ void kiln_paged_gather_rows_kernel(const uint8_t* __restrict__ pool,
                                              const uint32_t* __restrict__ block_table,
                                              uint8_t* __restrict__ out,
                                              int64_t b,
                                              int64_t seqlen_k,
                                              int64_t max_blocks_per_seq,
                                              int64_t page_block_size,
                                              int64_t hk,
                                              int64_t d,
                                              int64_t pool_rows,
                                              int64_t elem_bytes) {
    int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t inner = hk * d;
    int64_t total = b * seqlen_k * inner;
    if (gid >= total) return;

    int64_t out_row = gid / inner;
    int64_t inner_idx = gid % inner;
    int64_t bi = out_row / seqlen_k;
    int64_t t = out_row % seqlen_k;
    int64_t blk = t / page_block_size;
    int64_t within = t % page_block_size;

    bool valid = blk < max_blocks_per_seq;
    int64_t phys_row = 0;
    if (valid) {
        uint32_t phys_block = block_table[bi * max_blocks_per_seq + blk];
        phys_row = static_cast<int64_t>(phys_block) * page_block_size + within;
        valid = phys_row >= 0 && phys_row < pool_rows;
    }

    uint8_t* dst = out + gid * elem_bytes;
    if (valid) {
        const uint8_t* src = pool + (phys_row * inner + inner_idx) * elem_bytes;
        for (int64_t i = 0; i < elem_bytes; ++i) {
            dst[i] = src[i];
        }
    } else {
        for (int64_t i = 0; i < elem_bytes; ++i) {
            dst[i] = 0;
        }
    }
}

// Repeat GQA KV heads:
//   src[b, sk, hk, d] -> out[b, sk, h, d]
// where each kv head is repeated `group = h / hk` times. This replaces the
// generic broadcast materialization path for ROCm SDPA decode; that generic
// path flattens the output and calls index_select_dim0 with millions of
// indices at long context.
__global__ void kiln_gqa_repeat_heads_kernel(const uint8_t* __restrict__ src,
                                             uint8_t* __restrict__ out,
                                             int64_t b,
                                             int64_t sk,
                                             int64_t hk,
                                             int64_t h,
                                             int64_t d,
                                             int64_t elem_bytes) {
    int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = b * sk * h * d;
    if (gid >= total) return;

    int64_t d_i = gid % d;
    int64_t tmp = gid / d;
    int64_t h_i = tmp % h;
    tmp /= h;
    int64_t sk_i = tmp % sk;
    int64_t b_i = tmp / sk;

    int64_t group = h / hk;
    int64_t hk_i = h_i / group;
    int64_t src_elem = ((b_i * sk + sk_i) * hk + hk_i) * d + d_i;

    const uint8_t* src_ptr = src + src_elem * elem_bytes;
    uint8_t* dst_ptr = out + gid * elem_bytes;
    for (int64_t i = 0; i < elem_bytes; ++i) {
        dst_ptr[i] = src_ptr[i];
    }
}

// Direct BF16 paged decode attention for sq=1:
//   q[b, 1, h, d], k/v pool[pool_rows, hk, d] -> out[b, 1, h, d].
//
// One block owns one (batch, query-head) row. It streams the block-table-backed
// KV pool directly and uses an online softmax accumulator:
//   new_m = max(m, score)
//   acc = acc * exp(m - new_m) + exp(score - new_m) * v
//   l = l * exp(m - new_m) + exp(score - new_m)
//
// This avoids materializing gathered K/V, repeated GQA heads, scores, softmax,
// and PV intermediates for long-context decode.
__global__ void kiln_paged_attn_decode_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ seqused_k,
    __nv_bfloat16* __restrict__ out,
    int64_t b,
    int64_t h,
    int64_t hk,
    int64_t d,
    int64_t max_seqlen_k,
    int64_t max_blocks_per_seq,
    int64_t page_block_size,
    int64_t pool_rows,
    float scale) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bi = row / static_cast<int>(h);
    const int qh = row - bi * static_cast<int>(h);
    const int group = static_cast<int>(h / hk);
    const int kvh = qh / group;

    int64_t used = max_seqlen_k;
    if (seqused_k != nullptr) {
        used = static_cast<int64_t>(seqused_k[bi]);
        if (used > max_seqlen_k) used = max_seqlen_k;
    }
    if (used < 0) used = 0;

    const __nv_bfloat16* q_row =
        q + ((static_cast<int64_t>(bi) * h + qh) * d);
    __nv_bfloat16* out_row =
        out + ((static_cast<int64_t>(bi) * h + qh) * d);

    __shared__ float smem[BLOCK_SIZE];

    if (used == 0) {
        for (int64_t di = tid; di < d; di += blockDim.x) {
            out_row[di] = __float2bfloat16(0.0f);
        }
        return;
    }

    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    float accum = 0.0f;
    const bool owns_dim = static_cast<int64_t>(tid) < d;

    for (int64_t t = 0; t < used; ++t) {
        const int64_t blk = t / page_block_size;
        const int64_t within = t - blk * page_block_size;
        float local_dot = 0.0f;
        bool valid = blk < max_blocks_per_seq;
        int64_t phys_row = 0;
        if (valid) {
            const uint32_t phys_block =
                block_table[static_cast<int64_t>(bi) * max_blocks_per_seq + blk];
            phys_row = static_cast<int64_t>(phys_block) * page_block_size + within;
            valid = phys_row >= 0 && phys_row < pool_rows;
        }
        if (valid) {
            const __nv_bfloat16* k_row =
                k_pool + (phys_row * hk + kvh) * d;
            for (int64_t di = tid; di < d; di += blockDim.x) {
                local_dot += __bfloat162float(q_row[di]) * __bfloat162float(k_row[di]);
            }
        }
        const float dot = kiln_block_reduce_sum(local_dot, smem);
        if (valid) {
            const __nv_bfloat16* v_row =
                v_pool + (phys_row * hk + kvh) * d;
            const float score = dot * scale;
            const float new_m = score > m ? score : m;
            const float alpha = expf(m - new_m);
            const float beta = expf(score - new_m);
            if (owns_dim) {
                accum = accum * alpha + beta * __bfloat162float(v_row[tid]);
            }
            l = l * alpha + beta;
            m = new_m;
        }
    }

    if (owns_dim) {
        out_row[tid] = __float2bfloat16(l > 0.0f ? accum / l : 0.0f);
    }
}

} // namespace

extern "C" int kiln_paged_gather_index_async(const void* block_table_u32,
                                             void* out_idx_u32,
                                             int64_t b,
                                             int64_t seqlen_k,
                                             int64_t max_blocks_per_seq,
                                             int64_t page_block_size,
                                             cudaStream_t stream) {
    if (b < 0 || seqlen_k < 0 || max_blocks_per_seq < 0 || page_block_size <= 0) return 1;
    int64_t total = b * seqlen_k;
    if (total == 0) return 0;
    int64_t blocks_i64 = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 2;
    int blocks = static_cast<int>(blocks_i64);

    kiln_paged_gather_index_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<const uint32_t*>(block_table_u32),
        static_cast<uint32_t*>(out_idx_u32),
        b, seqlen_k, max_blocks_per_seq, page_block_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 100 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_gqa_repeat_heads_async(const void* src,
                                           void* out,
                                           int64_t b,
                                           int64_t sk,
                                           int64_t hk,
                                           int64_t h,
                                           int64_t d,
                                           int64_t elem_bytes,
                                           cudaStream_t stream) {
    if (b < 0 || sk < 0 || hk <= 0 || h <= 0 || d < 0) return 1;
    if (h % hk != 0) return 2;
    if (!(elem_bytes == 1 || elem_bytes == 2 || elem_bytes == 4 || elem_bytes == 8)) return 3;
    int64_t total = b * sk * h * d;
    if (total == 0) return 0;
    int64_t blocks_i64 = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 4;
    int blocks = static_cast<int>(blocks_i64);

    kiln_gqa_repeat_heads_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<const uint8_t*>(src),
        static_cast<uint8_t*>(out),
        b, sk, hk, h, d, elem_bytes);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 100 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_paged_gather_rows_async(const void* pool,
                                            const void* block_table_u32,
                                            void* out,
                                            int64_t b,
                                            int64_t seqlen_k,
                                            int64_t max_blocks_per_seq,
                                            int64_t page_block_size,
                                            int64_t hk,
                                            int64_t d,
                                            int64_t pool_rows,
                                            int64_t elem_bytes,
                                            cudaStream_t stream) {
    if (b < 0 || seqlen_k < 0 || max_blocks_per_seq < 0 || page_block_size <= 0
        || hk < 0 || d < 0 || pool_rows < 0) {
        return 1;
    }
    if (!(elem_bytes == 1 || elem_bytes == 2 || elem_bytes == 4 || elem_bytes == 8)) return 2;
    int64_t total = b * seqlen_k * hk * d;
    if (total == 0) return 0;
    int64_t blocks_i64 = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    kiln_paged_gather_rows_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<const uint8_t*>(pool),
        static_cast<const uint32_t*>(block_table_u32),
        static_cast<uint8_t*>(out),
        b, seqlen_k, max_blocks_per_seq, page_block_size, hk, d, pool_rows, elem_bytes);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 100 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_paged_tail_mask_async(const void* seqused_k_u32,
                                          void* out_mask_u8,
                                          int64_t b,
                                          int64_t h,
                                          int64_t sk,
                                          cudaStream_t stream) {
    if (b < 0 || h < 0 || sk < 0) return 1;
    int64_t total = b * h * sk;
    if (total == 0) return 0;
    int64_t blocks_i64 = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 2;
    int blocks = static_cast<int>(blocks_i64);

    kiln_paged_tail_mask_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<const uint32_t*>(seqused_k_u32),
        static_cast<uint8_t*>(out_mask_u8),
        b, h, sk);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 100 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_paged_attn_decode_bf16_async(const void* q_bf16,
                                                 const void* k_pool_bf16,
                                                 const void* v_pool_bf16,
                                                 const void* block_table_u32,
                                                 const void* seqused_k_u32,
                                                 void* out_bf16,
                                                 int64_t b,
                                                 int64_t h,
                                                 int64_t hk,
                                                 int64_t d,
                                                 int64_t max_seqlen_k,
                                                 int64_t max_blocks_per_seq,
                                                 int64_t page_block_size,
                                                 int64_t pool_rows,
                                                 float scale,
                                                 cudaStream_t stream) {
    if (b < 0 || h <= 0 || hk <= 0 || d <= 0 || max_seqlen_k < 0
        || max_blocks_per_seq < 0 || page_block_size <= 0 || pool_rows < 0) {
        return 1;
    }
    if (h % hk != 0) return 2;
    if (d > BLOCK_SIZE) return 3;
    if (max_seqlen_k > max_blocks_per_seq * page_block_size) return 4;
    int64_t total_rows = b * h;
    if (total_rows == 0) return 0;
    if (total_rows > (int64_t)2147483647) return 5;

    kiln_paged_attn_decode_bf16_kernel<<<static_cast<int>(total_rows), BLOCK_SIZE, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q_bf16),
        static_cast<const __nv_bfloat16*>(k_pool_bf16),
        static_cast<const __nv_bfloat16*>(v_pool_bf16),
        static_cast<const uint32_t*>(block_table_u32),
        static_cast<const uint32_t*>(seqused_k_u32),
        static_cast<__nv_bfloat16*>(out_bf16),
        b, h, hk, d, max_seqlen_k, max_blocks_per_seq, page_block_size, pool_rows, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 100 + static_cast<int>(err);
    return 0;
}
