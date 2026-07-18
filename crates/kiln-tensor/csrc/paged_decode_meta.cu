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

// Head-major variant:
//   src[b, hk, sk, d] -> out[b, h, sk, d]
// with each kv head repeated `group = h / hk` times. Used by ROCm streaming
// paged prefill, where Q and cached K/V are already head-major.
__global__ void kiln_gqa_repeat_heads_head_major_kernel(const uint8_t* __restrict__ src,
                                                        uint8_t* __restrict__ out,
                                                        int64_t b,
                                                        int64_t sk,
                                                        int64_t hk,
                                                        int64_t h,
                                                        int64_t d,
                                                        int64_t elem_bytes) {
    int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = b * h * sk * d;
    if (gid >= total) return;

    int64_t d_i = gid % d;
    int64_t tmp = gid / d;
    int64_t sk_i = tmp % sk;
    tmp /= sk;
    int64_t h_i = tmp % h;
    int64_t b_i = tmp / h;

    int64_t group = h / hk;
    int64_t hk_i = h_i / group;
    int64_t src_elem = ((b_i * hk + hk_i) * sk + sk_i) * d + d_i;

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

__global__ void kiln_paged_attn_decode_bf16_split_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ seqused_k,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int64_t b,
    int64_t h,
    int64_t hk,
    int64_t d,
    int64_t max_seqlen_k,
    int64_t max_blocks_per_seq,
    int64_t page_block_size,
    int64_t pool_rows,
    int64_t split_count,
    float scale) {
    const int partial = blockIdx.x;
    const int tid = threadIdx.x;
    const int row = partial / static_cast<int>(split_count);
    const int split = partial - row * static_cast<int>(split_count);
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

    const int64_t chunk = (used + split_count - 1) / split_count;
    const int64_t start = static_cast<int64_t>(split) * chunk;
    int64_t end = start + chunk;
    if (end > used) end = used;

    const __nv_bfloat16* q_row =
        q + ((static_cast<int64_t>(bi) * h + qh) * d);
    float* acc_row =
        partial_acc + (static_cast<int64_t>(partial) * d);

    __shared__ float smem[BLOCK_SIZE];

    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    float accum = 0.0f;
    const bool owns_dim = static_cast<int64_t>(tid) < d;

    for (int64_t t = start; t < end; ++t) {
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

    if (tid == 0) {
        partial_m[partial] = m;
        partial_l[partial] = l;
    }
    if (owns_dim) {
        acc_row[tid] = accum;
    }
}

__global__ void kiln_paged_attn_decode_bf16_split_reduce_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    __nv_bfloat16* __restrict__ out,
    int64_t h,
    int64_t d,
    int64_t split_count) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __nv_bfloat16* out_row = out + (static_cast<int64_t>(row) * d);

    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    float accum = 0.0f;
    const bool owns_dim = static_cast<int64_t>(tid) < d;

    for (int64_t split = 0; split < split_count; ++split) {
        const int64_t partial = static_cast<int64_t>(row) * split_count + split;
        const float m_s = partial_m[partial];
        const float l_s = partial_l[partial];
        if (l_s <= 0.0f) continue;

        const float new_m = m_s > m ? m_s : m;
        const float alpha = expf(m - new_m);
        const float beta = expf(m_s - new_m);
        if (owns_dim) {
            const float acc_s = partial_acc[partial * d + tid];
            accum = accum * alpha + beta * acc_s;
        }
        l = l * alpha + beta * l_s;
        m = new_m;
    }

    if (owns_dim) {
        out_row[tid] = __float2bfloat16(l > 0.0f ? accum / l : 0.0f);
    }
}

// GQA-specialized paged decode for sq=1 and h/hk <= 4. One block owns one
// (batch, kv-head) group and computes all query heads sharing that K/V stream.
// Qwen3.5-4B has h=16, hk=4, group=4, so this cuts long-context K/V reads for
// attention by roughly 4x compared with the one-block-per-query-head kernel.
__global__ void kiln_paged_attn_decode_bf16_gqa4_kernel(
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
    const int group_row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bi = group_row / static_cast<int>(hk);
    const int kvh = group_row - bi * static_cast<int>(hk);
    const int group = static_cast<int>(h / hk);
    const int qh_base = kvh * group;

    int64_t used = max_seqlen_k;
    if (seqused_k != nullptr) {
        used = static_cast<int64_t>(seqused_k[bi]);
        if (used > max_seqlen_k) used = max_seqlen_k;
    }
    if (used < 0) used = 0;

    const bool owns_dim = static_cast<int64_t>(tid) < d;
    __shared__ float smem[BLOCK_SIZE];

    float m[4] = {
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
    };
    float l[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (used == 0) {
        if (owns_dim) {
            for (int g = 0; g < group; ++g) {
                __nv_bfloat16* out_row =
                    out + ((static_cast<int64_t>(bi) * h + qh_base + g) * d);
                out_row[tid] = __float2bfloat16(0.0f);
            }
        }
        return;
    }

    for (int64_t t = 0; t < used; ++t) {
        const int64_t blk = t / page_block_size;
        const int64_t within = t - blk * page_block_size;
        bool valid = blk < max_blocks_per_seq;
        int64_t phys_row = 0;
        if (valid) {
            const uint32_t phys_block =
                block_table[static_cast<int64_t>(bi) * max_blocks_per_seq + blk];
            phys_row = static_cast<int64_t>(phys_block) * page_block_size + within;
            valid = phys_row >= 0 && phys_row < pool_rows;
        }

        float local_dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float v_val = 0.0f;
        if (valid && owns_dim) {
            const __nv_bfloat16* k_row =
                k_pool + (phys_row * hk + kvh) * d;
            const float k_val = __bfloat162float(k_row[tid]);
            v_val = __bfloat162float((v_pool + (phys_row * hk + kvh) * d)[tid]);
            for (int g = 0; g < group; ++g) {
                const __nv_bfloat16* q_row =
                    q + ((static_cast<int64_t>(bi) * h + qh_base + g) * d);
                local_dot[g] = __bfloat162float(q_row[tid]) * k_val;
            }
        }

        float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int g = 0; g < group; ++g) {
            dot[g] = kiln_block_reduce_sum(local_dot[g], smem);
        }

        if (valid) {
            for (int g = 0; g < group; ++g) {
                const float score = dot[g] * scale;
                const float new_m = score > m[g] ? score : m[g];
                const float alpha = expf(m[g] - new_m);
                const float beta = expf(score - new_m);
                if (owns_dim) {
                    accum[g] = accum[g] * alpha + beta * v_val;
                }
                l[g] = l[g] * alpha + beta;
                m[g] = new_m;
            }
        }
    }

    if (owns_dim) {
        for (int g = 0; g < group; ++g) {
            __nv_bfloat16* out_row =
                out + ((static_cast<int64_t>(bi) * h + qh_base + g) * d);
            out_row[tid] = __float2bfloat16(l[g] > 0.0f ? accum[g] / l[g] : 0.0f);
        }
    }
}

__global__ void kiln_paged_attn_decode_bf16_gqa4_split_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ seqused_k,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int64_t b,
    int64_t h,
    int64_t hk,
    int64_t d,
    int64_t max_seqlen_k,
    int64_t max_blocks_per_seq,
    int64_t page_block_size,
    int64_t pool_rows,
    int64_t split_count,
    float scale) {
    const int group_partial = blockIdx.x;
    const int tid = threadIdx.x;
    const int split = group_partial % static_cast<int>(split_count);
    const int group_row = group_partial / static_cast<int>(split_count);
    const int bi = group_row / static_cast<int>(hk);
    const int kvh = group_row - bi * static_cast<int>(hk);
    const int group = static_cast<int>(h / hk);
    const int qh_base = kvh * group;

    int64_t used = max_seqlen_k;
    if (seqused_k != nullptr) {
        used = static_cast<int64_t>(seqused_k[bi]);
        if (used > max_seqlen_k) used = max_seqlen_k;
    }
    if (used < 0) used = 0;

    const int64_t chunk = (used + split_count - 1) / split_count;
    const int64_t start = static_cast<int64_t>(split) * chunk;
    int64_t end = start + chunk;
    if (end > used) end = used;

    const bool owns_dim = static_cast<int64_t>(tid) < d;
    __shared__ float smem[BLOCK_SIZE];

    float m[4] = {
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
        -3.4028234663852886e38f,
    };
    float l[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int64_t t = start; t < end; ++t) {
        const int64_t blk = t / page_block_size;
        const int64_t within = t - blk * page_block_size;
        bool valid = blk < max_blocks_per_seq;
        int64_t phys_row = 0;
        if (valid) {
            const uint32_t phys_block =
                block_table[static_cast<int64_t>(bi) * max_blocks_per_seq + blk];
            phys_row = static_cast<int64_t>(phys_block) * page_block_size + within;
            valid = phys_row >= 0 && phys_row < pool_rows;
        }

        float local_dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float v_val = 0.0f;
        if (valid && owns_dim) {
            const __nv_bfloat16* k_row =
                k_pool + (phys_row * hk + kvh) * d;
            const float k_val = __bfloat162float(k_row[tid]);
            v_val = __bfloat162float((v_pool + (phys_row * hk + kvh) * d)[tid]);
            for (int g = 0; g < group; ++g) {
                const __nv_bfloat16* q_row =
                    q + ((static_cast<int64_t>(bi) * h + qh_base + g) * d);
                local_dot[g] = __bfloat162float(q_row[tid]) * k_val;
            }
        }

        float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int g = 0; g < group; ++g) {
            dot[g] = kiln_block_reduce_sum(local_dot[g], smem);
        }

        if (valid) {
            for (int g = 0; g < group; ++g) {
                const float score = dot[g] * scale;
                const float new_m = score > m[g] ? score : m[g];
                const float alpha = expf(m[g] - new_m);
                const float beta = expf(score - new_m);
                if (owns_dim) {
                    accum[g] = accum[g] * alpha + beta * v_val;
                }
                l[g] = l[g] * alpha + beta;
                m[g] = new_m;
            }
        }
    }

    for (int g = 0; g < group; ++g) {
        const int row = bi * static_cast<int>(h) + qh_base + g;
        const int64_t partial = static_cast<int64_t>(row) * split_count + split;
        if (tid == 0) {
            partial_m[partial] = m[g];
            partial_l[partial] = l[g];
        }
        if (owns_dim) {
            partial_acc[partial * d + tid] = accum[g];
        }
    }
}

__global__ void kiln_paged_attn_decode_bf16_gqa4_d128_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ seqused_k,
    __nv_bfloat16* __restrict__ out,
    int64_t h,
    int64_t hk,
    int64_t max_seqlen_k,
    int64_t max_blocks_per_seq,
    int64_t page_block_size,
    int64_t pool_rows,
    float scale) {
    constexpr int D = 128;
    constexpr int GROUP = 4;
    const int tid = threadIdx.x;
    const int g = tid / D;
    const int di = tid - g * D;
    const int group_row = blockIdx.x;
    const int bi = group_row / static_cast<int>(hk);
    const int kvh = group_row - bi * static_cast<int>(hk);
    const int qh = kvh * GROUP + g;

    int64_t used = max_seqlen_k;
    if (seqused_k != nullptr) {
        used = static_cast<int64_t>(seqused_k[bi]);
        if (used > max_seqlen_k) used = max_seqlen_k;
    }
    if (used < 0) used = 0;

    __shared__ float k_s[D];
    __shared__ float v_s[D];
    __shared__ float dot_s[GROUP * D];

    const __nv_bfloat16* q_row =
        q + ((static_cast<int64_t>(bi) * h + qh) * D);
    __nv_bfloat16* out_row =
        out + ((static_cast<int64_t>(bi) * h + qh) * D);
    const float q_val = __bfloat162float(q_row[di]);

    if (used == 0) {
        out_row[di] = __float2bfloat16(0.0f);
        return;
    }

    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    float accum = 0.0f;

    for (int64_t t = 0; t < used; ++t) {
        const int64_t blk = t / page_block_size;
        const int64_t within = t - blk * page_block_size;
        bool valid = blk < max_blocks_per_seq;
        int64_t phys_row = 0;
        if (valid) {
            const uint32_t phys_block =
                block_table[static_cast<int64_t>(bi) * max_blocks_per_seq + blk];
            phys_row = static_cast<int64_t>(phys_block) * page_block_size + within;
            valid = phys_row >= 0 && phys_row < pool_rows;
        }

        if (g == 0) {
            if (valid) {
                const __nv_bfloat16* k_row =
                    k_pool + (phys_row * hk + kvh) * D;
                const __nv_bfloat16* v_row =
                    v_pool + (phys_row * hk + kvh) * D;
                k_s[di] = __bfloat162float(k_row[di]);
                v_s[di] = __bfloat162float(v_row[di]);
            } else {
                k_s[di] = 0.0f;
                v_s[di] = 0.0f;
            }
        }
        __syncthreads();

        dot_s[tid] = q_val * k_s[di];
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (di < stride) {
                dot_s[g * D + di] += dot_s[g * D + di + stride];
            }
            __syncthreads();
        }
        const float dot = dot_s[g * D];

        if (valid) {
            const float score = dot * scale;
            const float new_m = score > m ? score : m;
            const float alpha = expf(m - new_m);
            const float beta = expf(score - new_m);
            accum = accum * alpha + beta * v_s[di];
            l = l * alpha + beta;
            m = new_m;
        }
        __syncthreads();
    }

    out_row[di] = __float2bfloat16(l > 0.0f ? accum / l : 0.0f);
}

__global__ void kiln_paged_attn_decode_bf16_gqa4_d128_split_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ seqused_k,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int64_t h,
    int64_t hk,
    int64_t max_seqlen_k,
    int64_t max_blocks_per_seq,
    int64_t page_block_size,
    int64_t pool_rows,
    int64_t split_count,
    float scale) {
    constexpr int D = 128;
    constexpr int GROUP = 4;
    const int tid = threadIdx.x;
    const int g = tid / D;
    const int di = tid - g * D;
    const int group_partial = blockIdx.x;
    const int split = group_partial % static_cast<int>(split_count);
    const int group_row = group_partial / static_cast<int>(split_count);
    const int bi = group_row / static_cast<int>(hk);
    const int kvh = group_row - bi * static_cast<int>(hk);
    const int qh = kvh * GROUP + g;

    int64_t used = max_seqlen_k;
    if (seqused_k != nullptr) {
        used = static_cast<int64_t>(seqused_k[bi]);
        if (used > max_seqlen_k) used = max_seqlen_k;
    }
    if (used < 0) used = 0;

    const int64_t chunk = (used + split_count - 1) / split_count;
    const int64_t start = static_cast<int64_t>(split) * chunk;
    int64_t end = start + chunk;
    if (end > used) end = used;

    __shared__ float k_s[D];
    __shared__ float v_s[D];
    __shared__ float dot_s[GROUP * D];

    const __nv_bfloat16* q_row =
        q + ((static_cast<int64_t>(bi) * h + qh) * D);
    const float q_val = __bfloat162float(q_row[di]);

    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    float accum = 0.0f;

    for (int64_t t = start; t < end; ++t) {
        const int64_t blk = t / page_block_size;
        const int64_t within = t - blk * page_block_size;
        bool valid = blk < max_blocks_per_seq;
        int64_t phys_row = 0;
        if (valid) {
            const uint32_t phys_block =
                block_table[static_cast<int64_t>(bi) * max_blocks_per_seq + blk];
            phys_row = static_cast<int64_t>(phys_block) * page_block_size + within;
            valid = phys_row >= 0 && phys_row < pool_rows;
        }

        if (g == 0) {
            if (valid) {
                const __nv_bfloat16* k_row =
                    k_pool + (phys_row * hk + kvh) * D;
                const __nv_bfloat16* v_row =
                    v_pool + (phys_row * hk + kvh) * D;
                k_s[di] = __bfloat162float(k_row[di]);
                v_s[di] = __bfloat162float(v_row[di]);
            } else {
                k_s[di] = 0.0f;
                v_s[di] = 0.0f;
            }
        }
        __syncthreads();

        dot_s[tid] = q_val * k_s[di];
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (di < stride) {
                dot_s[g * D + di] += dot_s[g * D + di + stride];
            }
            __syncthreads();
        }
        const float dot = dot_s[g * D];

        if (valid) {
            const float score = dot * scale;
            const float new_m = score > m ? score : m;
            const float alpha = expf(m - new_m);
            const float beta = expf(score - new_m);
            accum = accum * alpha + beta * v_s[di];
            l = l * alpha + beta;
            m = new_m;
        }
        __syncthreads();
    }

    const int row = bi * static_cast<int>(h) + qh;
    const int64_t partial = static_cast<int64_t>(row) * split_count + split;
    if (di == 0) {
        partial_m[partial] = m;
        partial_l[partial] = l;
    }
    partial_acc[partial * D + di] = accum;
}

__global__ void kiln_paged_attn_decode_bf16_gqa4_d256_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ seqused_k,
    __nv_bfloat16* __restrict__ out,
    int64_t h,
    int64_t hk,
    int64_t max_seqlen_k,
    int64_t max_blocks_per_seq,
    int64_t page_block_size,
    int64_t pool_rows,
    float scale) {
    constexpr int D = 256;
    constexpr int GROUP = 4;
    const int tid = threadIdx.x;
    const int g = tid / D;
    const int di = tid - g * D;
    const int group_row = blockIdx.x;
    const int bi = group_row / static_cast<int>(hk);
    const int kvh = group_row - bi * static_cast<int>(hk);
    const int qh = kvh * GROUP + g;

    int64_t used = max_seqlen_k;
    if (seqused_k != nullptr) {
        used = static_cast<int64_t>(seqused_k[bi]);
        if (used > max_seqlen_k) used = max_seqlen_k;
    }
    if (used < 0) used = 0;

    __shared__ float k_s[D];
    __shared__ float v_s[D];
    __shared__ float dot_s[GROUP * D];

    const __nv_bfloat16* q_row =
        q + ((static_cast<int64_t>(bi) * h + qh) * D);
    __nv_bfloat16* out_row =
        out + ((static_cast<int64_t>(bi) * h + qh) * D);
    const float q_val = __bfloat162float(q_row[di]);

    if (used == 0) {
        out_row[di] = __float2bfloat16(0.0f);
        return;
    }

    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    float accum = 0.0f;

    for (int64_t t = 0; t < used; ++t) {
        const int64_t blk = t / page_block_size;
        const int64_t within = t - blk * page_block_size;
        bool valid = blk < max_blocks_per_seq;
        int64_t phys_row = 0;
        if (valid) {
            const uint32_t phys_block =
                block_table[static_cast<int64_t>(bi) * max_blocks_per_seq + blk];
            phys_row = static_cast<int64_t>(phys_block) * page_block_size + within;
            valid = phys_row >= 0 && phys_row < pool_rows;
        }

        if (g == 0) {
            if (valid) {
                const __nv_bfloat16* k_row =
                    k_pool + (phys_row * hk + kvh) * D;
                const __nv_bfloat16* v_row =
                    v_pool + (phys_row * hk + kvh) * D;
                k_s[di] = __bfloat162float(k_row[di]);
                v_s[di] = __bfloat162float(v_row[di]);
            } else {
                k_s[di] = 0.0f;
                v_s[di] = 0.0f;
            }
        }
        __syncthreads();

        dot_s[tid] = q_val * k_s[di];
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (di < stride) {
                dot_s[g * D + di] += dot_s[g * D + di + stride];
            }
            __syncthreads();
        }
        const float dot = dot_s[g * D];

        if (valid) {
            const float score = dot * scale;
            const float new_m = score > m ? score : m;
            const float alpha = expf(m - new_m);
            const float beta = expf(score - new_m);
            accum = accum * alpha + beta * v_s[di];
            l = l * alpha + beta;
            m = new_m;
        }
        __syncthreads();
    }

    out_row[di] = __float2bfloat16(l > 0.0f ? accum / l : 0.0f);
}

__global__ void kiln_paged_attn_decode_bf16_gqa4_d256_split_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ seqused_k,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int64_t h,
    int64_t hk,
    int64_t max_seqlen_k,
    int64_t max_blocks_per_seq,
    int64_t page_block_size,
    int64_t pool_rows,
    int64_t split_count,
    float scale) {
    constexpr int D = 256;
    constexpr int GROUP = 4;
    const int di = threadIdx.x;
    const int group_partial = blockIdx.x;
    const int split = group_partial % static_cast<int>(split_count);
    const int group_row = group_partial / static_cast<int>(split_count);
    const int bi = group_row / static_cast<int>(hk);
    const int kvh = group_row - bi * static_cast<int>(hk);
    const int qh0 = kvh * GROUP;

    int64_t used = max_seqlen_k;
    if (seqused_k != nullptr) {
        used = static_cast<int64_t>(seqused_k[bi]);
        if (used > max_seqlen_k) used = max_seqlen_k;
    }
    if (used < 0) used = 0;

    const int64_t chunk = (used + split_count - 1) / split_count;
    const int64_t start = static_cast<int64_t>(split) * chunk;
    int64_t end = start + chunk;
    if (end > used) end = used;

    __shared__ float dot_s[GROUP * D];

    const int64_t q_base = (static_cast<int64_t>(bi) * h + qh0) * D + di;
    const float q0 = __bfloat162float(q[q_base]);
    const float q1 = __bfloat162float(q[q_base + D]);
    const float q2 = __bfloat162float(q[q_base + 2 * D]);
    const float q3 = __bfloat162float(q[q_base + 3 * D]);

    float m0 = -3.4028234663852886e38f;
    float m1 = -3.4028234663852886e38f;
    float m2 = -3.4028234663852886e38f;
    float m3 = -3.4028234663852886e38f;
    float l0 = 0.0f;
    float l1 = 0.0f;
    float l2 = 0.0f;
    float l3 = 0.0f;
    float accum0 = 0.0f;
    float accum1 = 0.0f;
    float accum2 = 0.0f;
    float accum3 = 0.0f;

    for (int64_t t = start; t < end; ++t) {
        const int64_t blk = t / page_block_size;
        const int64_t within = t - blk * page_block_size;
        bool valid = blk < max_blocks_per_seq;
        int64_t phys_row = 0;
        if (valid) {
            const uint32_t phys_block =
                block_table[static_cast<int64_t>(bi) * max_blocks_per_seq + blk];
            phys_row = static_cast<int64_t>(phys_block) * page_block_size + within;
            valid = phys_row >= 0 && phys_row < pool_rows;
        }

        float k_val = 0.0f;
        float v_val = 0.0f;
        if (valid) {
            const int64_t kv_base = (phys_row * hk + kvh) * D + di;
            k_val = __bfloat162float(k_pool[kv_base]);
            v_val = __bfloat162float(v_pool[kv_base]);
        }

        dot_s[di] = q0 * k_val;
        dot_s[D + di] = q1 * k_val;
        dot_s[2 * D + di] = q2 * k_val;
        dot_s[3 * D + di] = q3 * k_val;
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (di < stride) {
                dot_s[di] += dot_s[di + stride];
                dot_s[D + di] += dot_s[D + di + stride];
                dot_s[2 * D + di] += dot_s[2 * D + di + stride];
                dot_s[3 * D + di] += dot_s[3 * D + di + stride];
            }
            __syncthreads();
        }

        if (valid) {
            const float score0 = dot_s[0] * scale;
            const float new_m0 = score0 > m0 ? score0 : m0;
            const float alpha0 = expf(m0 - new_m0);
            const float beta0 = expf(score0 - new_m0);
            accum0 = accum0 * alpha0 + beta0 * v_val;
            l0 = l0 * alpha0 + beta0;
            m0 = new_m0;

            const float score1 = dot_s[D] * scale;
            const float new_m1 = score1 > m1 ? score1 : m1;
            const float alpha1 = expf(m1 - new_m1);
            const float beta1 = expf(score1 - new_m1);
            accum1 = accum1 * alpha1 + beta1 * v_val;
            l1 = l1 * alpha1 + beta1;
            m1 = new_m1;

            const float score2 = dot_s[2 * D] * scale;
            const float new_m2 = score2 > m2 ? score2 : m2;
            const float alpha2 = expf(m2 - new_m2);
            const float beta2 = expf(score2 - new_m2);
            accum2 = accum2 * alpha2 + beta2 * v_val;
            l2 = l2 * alpha2 + beta2;
            m2 = new_m2;

            const float score3 = dot_s[3 * D] * scale;
            const float new_m3 = score3 > m3 ? score3 : m3;
            const float alpha3 = expf(m3 - new_m3);
            const float beta3 = expf(score3 - new_m3);
            accum3 = accum3 * alpha3 + beta3 * v_val;
            l3 = l3 * alpha3 + beta3;
            m3 = new_m3;
        }
        __syncthreads();
    }

    const int row0 = bi * static_cast<int>(h) + qh0;
    const int64_t partial0 = static_cast<int64_t>(row0) * split_count + split;
    const int64_t partial1 = partial0 + split_count;
    const int64_t partial2 = partial1 + split_count;
    const int64_t partial3 = partial2 + split_count;
    if (di == 0) {
        partial_m[partial0] = m0;
        partial_l[partial0] = l0;
        partial_m[partial1] = m1;
        partial_l[partial1] = l1;
        partial_m[partial2] = m2;
        partial_l[partial2] = l2;
        partial_m[partial3] = m3;
        partial_l[partial3] = l3;
    }
    partial_acc[partial0 * D + di] = accum0;
    partial_acc[partial1 * D + di] = accum1;
    partial_acc[partial2 * D + di] = accum2;
    partial_acc[partial3 * D + di] = accum3;
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

extern "C" int kiln_gqa_repeat_heads_head_major_async(const void* src,
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
    int64_t total = b * h * sk * d;
    if (total == 0) return 0;
    int64_t blocks_i64 = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 4;
    int blocks = static_cast<int>(blocks_i64);

    kiln_gqa_repeat_heads_head_major_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
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

extern "C" int kiln_paged_attn_decode_bf16_split_async(const void* q_bf16,
                                                       const void* k_pool_bf16,
                                                       const void* v_pool_bf16,
                                                       const void* block_table_u32,
                                                       const void* seqused_k_u32,
                                                       void* out_bf16,
                                                       void* partial_m_f32,
                                                       void* partial_l_f32,
                                                       void* partial_acc_f32,
                                                       int64_t b,
                                                       int64_t h,
                                                       int64_t hk,
                                                       int64_t d,
                                                       int64_t max_seqlen_k,
                                                       int64_t max_blocks_per_seq,
                                                       int64_t page_block_size,
                                                       int64_t pool_rows,
                                                       int64_t split_count,
                                                       float scale,
                                                       cudaStream_t stream) {
    if (b < 0 || h <= 0 || hk <= 0 || d <= 0 || max_seqlen_k < 0
        || max_blocks_per_seq < 0 || page_block_size <= 0 || pool_rows < 0
        || split_count <= 1) {
        return 1;
    }
    if (h % hk != 0) return 2;
    if (d > BLOCK_SIZE) return 3;
    if (max_seqlen_k > max_blocks_per_seq * page_block_size) return 4;
    int64_t total_rows = b * h;
    if (total_rows == 0) return 0;
    if (total_rows > (int64_t)2147483647) return 5;
    int64_t total_partials = total_rows * split_count;
    if (total_partials > (int64_t)2147483647) return 6;

    kiln_paged_attn_decode_bf16_split_kernel<<<static_cast<int>(total_partials), BLOCK_SIZE, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q_bf16),
        static_cast<const __nv_bfloat16*>(k_pool_bf16),
        static_cast<const __nv_bfloat16*>(v_pool_bf16),
        static_cast<const uint32_t*>(block_table_u32),
        static_cast<const uint32_t*>(seqused_k_u32),
        static_cast<float*>(partial_m_f32),
        static_cast<float*>(partial_l_f32),
        static_cast<float*>(partial_acc_f32),
        b, h, hk, d, max_seqlen_k, max_blocks_per_seq, page_block_size,
        pool_rows, split_count, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 100 + static_cast<int>(err);

    kiln_paged_attn_decode_bf16_split_reduce_kernel<<<static_cast<int>(total_rows), BLOCK_SIZE, 0, stream>>>(
        static_cast<const float*>(partial_m_f32),
        static_cast<const float*>(partial_l_f32),
        static_cast<const float*>(partial_acc_f32),
        static_cast<__nv_bfloat16*>(out_bf16),
        h, d, split_count);

    err = cudaGetLastError();
    if (err != cudaSuccess) return 200 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_paged_attn_decode_bf16_gqa4_async(const void* q_bf16,
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
                                                      int64_t parallel_head_dim,
                                                      cudaStream_t stream) {
    if (b < 0 || h <= 0 || hk <= 0 || d <= 0 || max_seqlen_k < 0
        || max_blocks_per_seq < 0 || page_block_size <= 0 || pool_rows < 0) {
        return 1;
    }
    if (h % hk != 0) return 2;
    int64_t group = h / hk;
    if (group < 2 || group > 4) return 3;
    if (d > BLOCK_SIZE) return 4;
    if (max_seqlen_k > max_blocks_per_seq * page_block_size) return 5;
    int64_t total_group_rows = b * hk;
    if (total_group_rows == 0) return 0;
    if (total_group_rows > (int64_t)2147483647) return 6;
    if (parallel_head_dim != 0 && parallel_head_dim != 128 && parallel_head_dim != 256) return 7;

    const bool use_d256_parallel =
        parallel_head_dim == 256 && group == 4 && d == 256;
    const bool use_d128_parallel =
        parallel_head_dim == 128 && group == 4 && d == 128;
    if (use_d256_parallel) {
        kiln_paged_attn_decode_bf16_gqa4_d256_kernel<<<static_cast<int>(total_group_rows), 1024, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_bf16),
            static_cast<const __nv_bfloat16*>(k_pool_bf16),
            static_cast<const __nv_bfloat16*>(v_pool_bf16),
            static_cast<const uint32_t*>(block_table_u32),
            static_cast<const uint32_t*>(seqused_k_u32),
            static_cast<__nv_bfloat16*>(out_bf16),
            h, hk, max_seqlen_k, max_blocks_per_seq, page_block_size, pool_rows, scale);
    } else if (use_d128_parallel) {
        kiln_paged_attn_decode_bf16_gqa4_d128_kernel<<<static_cast<int>(total_group_rows), 512, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_bf16),
            static_cast<const __nv_bfloat16*>(k_pool_bf16),
            static_cast<const __nv_bfloat16*>(v_pool_bf16),
            static_cast<const uint32_t*>(block_table_u32),
            static_cast<const uint32_t*>(seqused_k_u32),
            static_cast<__nv_bfloat16*>(out_bf16),
            h, hk, max_seqlen_k, max_blocks_per_seq, page_block_size, pool_rows, scale);
    } else {
        kiln_paged_attn_decode_bf16_gqa4_kernel<<<static_cast<int>(total_group_rows), BLOCK_SIZE, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_bf16),
            static_cast<const __nv_bfloat16*>(k_pool_bf16),
            static_cast<const __nv_bfloat16*>(v_pool_bf16),
            static_cast<const uint32_t*>(block_table_u32),
            static_cast<const uint32_t*>(seqused_k_u32),
            static_cast<__nv_bfloat16*>(out_bf16),
            b, h, hk, d, max_seqlen_k, max_blocks_per_seq, page_block_size, pool_rows, scale);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 100 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_paged_attn_decode_bf16_gqa4_split_async(const void* q_bf16,
                                                            const void* k_pool_bf16,
                                                            const void* v_pool_bf16,
                                                            const void* block_table_u32,
                                                            const void* seqused_k_u32,
                                                            void* out_bf16,
                                                            void* partial_m_f32,
                                                            void* partial_l_f32,
                                                            void* partial_acc_f32,
                                                            int64_t b,
                                                            int64_t h,
                                                            int64_t hk,
                                                            int64_t d,
                                                            int64_t max_seqlen_k,
                                                            int64_t max_blocks_per_seq,
                                                            int64_t page_block_size,
                                                            int64_t pool_rows,
                                                            int64_t split_count,
                                                            float scale,
                                                            int64_t parallel_head_dim,
                                                            cudaStream_t stream) {
    if (b < 0 || h <= 0 || hk <= 0 || d <= 0 || max_seqlen_k < 0
        || max_blocks_per_seq < 0 || page_block_size <= 0 || pool_rows < 0
        || split_count <= 1) {
        return 1;
    }
    if (h % hk != 0) return 2;
    int64_t group = h / hk;
    if (group < 2 || group > 4) return 3;
    if (d > BLOCK_SIZE) return 4;
    if (max_seqlen_k > max_blocks_per_seq * page_block_size) return 5;
    int64_t total_rows = b * h;
    if (total_rows == 0) return 0;
    if (total_rows > (int64_t)2147483647) return 6;
    int64_t total_group_partials = b * hk * split_count;
    if (total_group_partials > (int64_t)2147483647) return 7;
    if (parallel_head_dim != 0 && parallel_head_dim != 128 && parallel_head_dim != 256) return 8;

    const bool use_d256_parallel =
        parallel_head_dim == 256 && group == 4 && d == 256;
    const bool use_d128_parallel =
        parallel_head_dim == 128 && group == 4 && d == 128;
    if (use_d256_parallel) {
        kiln_paged_attn_decode_bf16_gqa4_d256_split_kernel<<<static_cast<int>(total_group_partials), BLOCK_SIZE, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_bf16),
            static_cast<const __nv_bfloat16*>(k_pool_bf16),
            static_cast<const __nv_bfloat16*>(v_pool_bf16),
            static_cast<const uint32_t*>(block_table_u32),
            static_cast<const uint32_t*>(seqused_k_u32),
            static_cast<float*>(partial_m_f32),
            static_cast<float*>(partial_l_f32),
            static_cast<float*>(partial_acc_f32),
            h, hk, max_seqlen_k, max_blocks_per_seq, page_block_size,
            pool_rows, split_count, scale);
    } else if (use_d128_parallel) {
        kiln_paged_attn_decode_bf16_gqa4_d128_split_kernel<<<static_cast<int>(total_group_partials), 512, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_bf16),
            static_cast<const __nv_bfloat16*>(k_pool_bf16),
            static_cast<const __nv_bfloat16*>(v_pool_bf16),
            static_cast<const uint32_t*>(block_table_u32),
            static_cast<const uint32_t*>(seqused_k_u32),
            static_cast<float*>(partial_m_f32),
            static_cast<float*>(partial_l_f32),
            static_cast<float*>(partial_acc_f32),
            h, hk, max_seqlen_k, max_blocks_per_seq, page_block_size,
            pool_rows, split_count, scale);
    } else {
        kiln_paged_attn_decode_bf16_gqa4_split_kernel<<<static_cast<int>(total_group_partials), BLOCK_SIZE, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_bf16),
            static_cast<const __nv_bfloat16*>(k_pool_bf16),
            static_cast<const __nv_bfloat16*>(v_pool_bf16),
            static_cast<const uint32_t*>(block_table_u32),
            static_cast<const uint32_t*>(seqused_k_u32),
            static_cast<float*>(partial_m_f32),
            static_cast<float*>(partial_l_f32),
            static_cast<float*>(partial_acc_f32),
            b, h, hk, d, max_seqlen_k, max_blocks_per_seq, page_block_size,
            pool_rows, split_count, scale);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 100 + static_cast<int>(err);

    kiln_paged_attn_decode_bf16_split_reduce_kernel<<<static_cast<int>(total_rows), BLOCK_SIZE, 0, stream>>>(
        static_cast<const float*>(partial_m_f32),
        static_cast<const float*>(partial_l_f32),
        static_cast<const float*>(partial_acc_f32),
        static_cast<__nv_bfloat16*>(out_bf16),
        h, d, split_count);

    err = cudaGetLastError();
    if (err != cudaSuccess) return 200 + static_cast<int>(err);
    return 0;
}
