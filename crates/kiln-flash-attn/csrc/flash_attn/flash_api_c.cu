/******************************************************************************
 * Kiln flash-attention C-ABI wrapper.
 *
 * Thin layer that replaces the PyTorch binding in flash_api.cpp.
 * Accepts raw CUDA device pointers + dimensions, populates the kernel param
 * structs, and calls the template-instantiated forward/backward kernels.
 *
 * Only bf16 / head_dim=128 / causal instantiations are compiled.
 ******************************************************************************/

#include "flash_api_c.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "src/namespace_config.h"
#include "src/flash.h"
#include "src/static_switch.h"

// We compile bf16 / causal=true for hdim128 and hdim256 (Qwen3.5-4B uses hdim256).
#include <cutlass/numeric_types.h>

namespace FLASH_NAMESPACE {
    // Forward declarations matching the .cu instantiation files.
    // hdim128
    template<> void run_mha_fwd_<cutlass::bfloat16_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream);
    template<> void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream);
    template<> void run_mha_bwd_<cutlass::bfloat16_t, 128, true>(Flash_bwd_params &params, cudaStream_t stream);
    // hdim256
    template<> void run_mha_fwd_<cutlass::bfloat16_t, 256, true>(Flash_fwd_params &params, cudaStream_t stream);
    template<> void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, true>(Flash_fwd_params &params, cudaStream_t stream);
    template<> void run_mha_bwd_<cutlass::bfloat16_t, 256, true>(Flash_bwd_params &params, cudaStream_t stream);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline int round_up(int x, int m) {
    return (x + m - 1) / m * m;
}

// Populate the forward params struct from raw pointers + dimensions.
static void set_params_fwd(
    FLASH_NAMESPACE::Flash_fwd_params &params,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads, int num_heads_k, int head_dim,
    const void *q, const void *k, const void *v,
    void *out, void *softmax_lse,
    float softmax_scale, bool is_causal)
{
    memset(&params, 0, sizeof(params));

    params.is_bf16 = true;

    // Pointers
    params.q_ptr = const_cast<void *>(q);
    params.k_ptr = const_cast<void *>(k);
    params.v_ptr = const_cast<void *>(v);
    params.o_ptr = out;

    // Strides for [batch, seqlen, num_heads, head_dim] layout (row-major)
    params.q_row_stride  = num_heads * head_dim;
    params.k_row_stride  = num_heads_k * head_dim;
    params.v_row_stride  = num_heads_k * head_dim;
    params.q_head_stride = head_dim;
    params.k_head_stride = head_dim;
    params.v_head_stride = head_dim;

    params.q_batch_stride = seqlen_q * num_heads * head_dim;
    params.k_batch_stride = seqlen_k * num_heads_k * head_dim;
    params.v_batch_stride = seqlen_k * num_heads_k * head_dim;

    params.o_row_stride  = num_heads * head_dim;
    params.o_head_stride = head_dim;
    params.o_batch_stride = seqlen_q * num_heads * head_dim;

    // Softmax LSE: [batch, num_heads, seqlen_q]
    params.softmax_lse_ptr = softmax_lse;

    // Dimensions
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = head_dim;
    params.d_rounded = round_up(head_dim, 32);
    params.seqlen_q_rounded = round_up(seqlen_q, 128);
    params.seqlen_k_rounded = round_up(seqlen_k, 128);

    // Scale
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);

    // No dropout
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;

    // Causal
    params.is_causal = is_causal;
    params.window_size_left = -1;   // no left window limit
    params.window_size_right = is_causal ? 0 : -1;

    // Unused pointers
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_k = nullptr;
    params.p_ptr = nullptr;
    params.softmax_lseaccum_ptr = nullptr;
    params.oaccum_ptr = nullptr;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.cache_batch_idx = nullptr;
    params.block_table = nullptr;
    params.blockmask = nullptr;
    params.alibi_slopes_ptr = nullptr;
    params.rng_state = nullptr;

    params.is_seqlens_k_cumulative = true;
    params.is_rotary_interleaved = false;
    params.num_splits = 1; // no split-KV for now
    params.softcap = 0.0f;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;
}

// ---------------------------------------------------------------------------
// C-ABI entry points
// ---------------------------------------------------------------------------

extern "C" kiln_flash_status_t kiln_flash_attn_fwd(
    const void *q, const void *k, const void *v,
    void *out, void *softmax_lse_out,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads, int num_heads_k, int head_dim,
    float softmax_scale, int is_causal,
    void *stream)
{
    if (head_dim != 128 && head_dim != 256) {
        fprintf(stderr, "kiln_flash_attn_fwd: only head_dim=128,256 supported, got %d\n", head_dim);
        return -1;
    }
    if (num_heads % num_heads_k != 0) {
        fprintf(stderr, "kiln_flash_attn_fwd: num_heads (%d) must be divisible by num_heads_k (%d)\n",
                num_heads, num_heads_k);
        return -2;
    }

    FLASH_NAMESPACE::Flash_fwd_params params;
    set_params_fwd(params,
                   batch_size, seqlen_q, seqlen_k,
                   num_heads, num_heads_k, head_dim,
                   q, k, v, out, softmax_lse_out,
                   softmax_scale, is_causal != 0);

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    if (head_dim == 128) {
        FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, 128, true>(params, cuda_stream);
    } else {
        FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, 256, true>(params, cuda_stream);
    }

    return 0;
}

extern "C" kiln_flash_status_t kiln_flash_attn_fwd_paged_decode(
    const void *q,
    const void *k_pool,
    const void *v_pool,
    const int  *block_table,
    void *out,
    void *softmax_lse_out,
    int batch_size,
    int num_heads,
    int num_heads_k,
    int head_dim,
    int max_seqlen_k,
    int max_blocks_per_seq,
    int page_block_size,
    float softmax_scale,
    int is_causal,
    void *stream)
{
    if (head_dim != 128 && head_dim != 256) {
        fprintf(stderr, "kiln_flash_attn_fwd_paged_decode: only head_dim=128,256 supported, got %d\n", head_dim);
        return -1;
    }
    if (num_heads % num_heads_k != 0) {
        fprintf(stderr, "kiln_flash_attn_fwd_paged_decode: num_heads (%d) must be divisible by num_heads_k (%d)\n",
                num_heads, num_heads_k);
        return -2;
    }
    // The splitkv hdim128/hdim256 instantiations use kBlockN = 128. Within a
    // single 128-token chunk, the kernel reads a contiguous run of physical
    // tokens using a single block_table entry; therefore physical pages within
    // each 128-token chunk must be contiguous in the pool. The kernel handles
    // page_block_size <= kBlockN by computing per-chunk block_table indices,
    // but kBlockN must be evenly divisible by page_block_size.
    constexpr int kBlockN = 128;
    if (page_block_size <= 0 || (kBlockN % page_block_size) != 0) {
        fprintf(stderr, "kiln_flash_attn_fwd_paged_decode: page_block_size (%d) must divide kBlockN (%d)\n",
                page_block_size, kBlockN);
        return -3;
    }

    FLASH_NAMESPACE::Flash_fwd_params params;
    memset(&params, 0, sizeof(params));

    params.is_bf16 = true;

    // Pointers
    params.q_ptr = const_cast<void *>(q);
    params.k_ptr = const_cast<void *>(k_pool);
    params.v_ptr = const_cast<void *>(v_pool);
    params.o_ptr = out;
    params.softmax_lse_ptr = softmax_lse_out;

    // Q layout: [batch, 1, num_heads, head_dim]
    params.q_row_stride   = num_heads * head_dim;
    params.q_head_stride  = head_dim;
    params.q_batch_stride = num_heads * head_dim;  // seqlen_q = 1

    // K/V pool layout: each logical block is [page_block_size, num_heads_k, head_dim]
    // The kernel computes the base of a block via:
    //   row_offset_k = block_table[idx] * params.k_batch_stride + offset_in_block * params.k_row_stride + (kv_head) * params.k_head_stride
    // So k_batch_stride is the stride between adjacent logical pages.
    params.k_row_stride   = num_heads_k * head_dim;
    params.v_row_stride   = num_heads_k * head_dim;
    params.k_head_stride  = head_dim;
    params.v_head_stride  = head_dim;
    params.k_batch_stride = (int64_t)page_block_size * num_heads_k * head_dim;
    params.v_batch_stride = (int64_t)page_block_size * num_heads_k * head_dim;

    // Output: [batch, 1, num_heads, head_dim]
    params.o_row_stride   = num_heads * head_dim;
    params.o_head_stride  = head_dim;
    params.o_batch_stride = num_heads * head_dim;

    // Dimensions
    params.b      = batch_size;
    params.h      = num_heads;
    params.h_k    = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = 1;
    params.seqlen_k = max_seqlen_k;
    params.d        = head_dim;
    params.d_rounded = round_up(head_dim, 32);
    params.seqlen_q_rounded = round_up(1, 128);
    params.seqlen_k_rounded = round_up(max_seqlen_k, 128);

    // Scale
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);

    // Dropout disabled
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;

    // Causal mask
    params.is_causal = is_causal != 0;
    params.window_size_left = -1;
    params.window_size_right = (is_causal != 0) ? 0 : -1;

    // Paged KV
    params.block_table = const_cast<int *>(block_table);
    params.block_table_batch_stride = max_blocks_per_seq;
    params.page_block_size = page_block_size;

    // Unused
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_k = nullptr;
    params.p_ptr = nullptr;
    params.softmax_lseaccum_ptr = nullptr;
    params.oaccum_ptr = nullptr;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.cache_batch_idx = nullptr;
    params.blockmask = nullptr;
    params.alibi_slopes_ptr = nullptr;
    params.rng_state = nullptr;

    params.is_seqlens_k_cumulative = true;
    params.is_rotary_interleaved = false;
    params.num_splits = 1;  // no split-KV; avoids needing accum scratch
    params.softcap = 0.0f;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    if (head_dim == 128) {
        FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, true>(params, cuda_stream);
    } else {
        FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, true>(params, cuda_stream);
    }

    return 0;
}

extern "C" kiln_flash_status_t kiln_flash_attn_fwd_paged_decode_dyn_seqlen(
    const void *q,
    const void *k_pool,
    const void *v_pool,
    const int  *block_table,
    const int  *seqused_k,
    void *out,
    void *softmax_lse_out,
    int batch_size,
    int num_heads,
    int num_heads_k,
    int head_dim,
    int max_seqlen_k,
    int max_blocks_per_seq,
    int page_block_size,
    float softmax_scale,
    int is_causal,
    void *stream)
{
    if (head_dim != 128 && head_dim != 256) {
        fprintf(stderr, "kiln_flash_attn_fwd_paged_decode_dyn_seqlen: only head_dim=128,256 supported, got %d\n", head_dim);
        return -1;
    }
    if (num_heads % num_heads_k != 0) {
        fprintf(stderr, "kiln_flash_attn_fwd_paged_decode_dyn_seqlen: num_heads (%d) must be divisible by num_heads_k (%d)\n",
                num_heads, num_heads_k);
        return -2;
    }
    constexpr int kBlockN = 128;
    if (page_block_size <= 0 || (kBlockN % page_block_size) != 0) {
        fprintf(stderr, "kiln_flash_attn_fwd_paged_decode_dyn_seqlen: page_block_size (%d) must divide kBlockN (%d)\n",
                page_block_size, kBlockN);
        return -3;
    }
    if (seqused_k == nullptr) {
        fprintf(stderr, "kiln_flash_attn_fwd_paged_decode_dyn_seqlen: seqused_k must be non-null\n");
        return -4;
    }

    FLASH_NAMESPACE::Flash_fwd_params params;
    memset(&params, 0, sizeof(params));

    params.is_bf16 = true;
    params.q_ptr = const_cast<void *>(q);
    params.k_ptr = const_cast<void *>(k_pool);
    params.v_ptr = const_cast<void *>(v_pool);
    params.o_ptr = out;
    params.softmax_lse_ptr = softmax_lse_out;

    params.q_row_stride   = num_heads * head_dim;
    params.q_head_stride  = head_dim;
    params.q_batch_stride = num_heads * head_dim;

    params.k_row_stride   = num_heads_k * head_dim;
    params.v_row_stride   = num_heads_k * head_dim;
    params.k_head_stride  = head_dim;
    params.v_head_stride  = head_dim;
    params.k_batch_stride = (int64_t)page_block_size * num_heads_k * head_dim;
    params.v_batch_stride = (int64_t)page_block_size * num_heads_k * head_dim;

    params.o_row_stride   = num_heads * head_dim;
    params.o_head_stride  = head_dim;
    params.o_batch_stride = num_heads * head_dim;

    params.b      = batch_size;
    params.h      = num_heads;
    params.h_k    = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = 1;
    params.seqlen_k = max_seqlen_k;
    params.d        = head_dim;
    params.d_rounded = round_up(head_dim, 32);
    params.seqlen_q_rounded = round_up(1, 128);
    params.seqlen_k_rounded = round_up(max_seqlen_k, 128);

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.is_causal = is_causal != 0;
    params.window_size_left = -1;
    params.window_size_right = (is_causal != 0) ? 0 : -1;

    params.block_table = const_cast<int *>(block_table);
    params.block_table_batch_stride = max_blocks_per_seq;
    params.page_block_size = page_block_size;
    params.seqused_k = const_cast<int *>(seqused_k);

    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.leftpad_k = nullptr;
    params.p_ptr = nullptr;
    params.softmax_lseaccum_ptr = nullptr;
    params.oaccum_ptr = nullptr;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.cache_batch_idx = nullptr;
    params.blockmask = nullptr;
    params.alibi_slopes_ptr = nullptr;
    params.rng_state = nullptr;
    params.is_seqlens_k_cumulative = true;
    params.is_rotary_interleaved = false;
    params.num_splits = 1;
    params.softcap = 0.0f;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    if (head_dim == 128) {
        FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, true>(params, cuda_stream);
    } else {
        FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, true>(params, cuda_stream);
    }

    return 0;
}

__global__ void kiln_paged_kv_write_token_major_bf16_slot_kernel(
    __nv_bfloat16 *k_pool,
    __nv_bfloat16 *v_pool,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    const unsigned int *slot,
    int num_kv_heads,
    int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_kv_heads * head_dim;
    if (idx >= total) return;
    int dst = int(*slot) * total + idx;
    k_pool[dst] = k[idx];
    v_pool[dst] = v[idx];
}

extern "C" kiln_flash_status_t kiln_paged_kv_write_token_major_bf16_slot(
    void *k_pool,
    void *v_pool,
    const void *k,
    const void *v,
    const unsigned int *slot,
    int num_kv_heads,
    int head_dim,
    void *stream)
{
    if (num_kv_heads <= 0 || head_dim <= 0) return -1;
    int total = num_kv_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    kiln_paged_kv_write_token_major_bf16_slot_kernel<<<blocks, threads, 0, cuda_stream>>>(
        static_cast<__nv_bfloat16 *>(k_pool),
        static_cast<__nv_bfloat16 *>(v_pool),
        static_cast<const __nv_bfloat16 *>(k),
        static_cast<const __nv_bfloat16 *>(v),
        slot,
        num_kv_heads,
        head_dim);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -2;
}

extern "C" kiln_flash_status_t kiln_flash_attn_bwd(
    const void *dout,
    const void *q, const void *k, const void *v,
    const void *out, const void *softmax_lse,
    void *dq, void *dk, void *dv,
    void *softmax_d_out, void *dq_accum,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads, int num_heads_k, int head_dim,
    float softmax_scale, int is_causal, int deterministic,
    void *stream)
{
    if (head_dim != 128 && head_dim != 256) {
        fprintf(stderr, "kiln_flash_attn_bwd: only head_dim=128,256 supported, got %d\n", head_dim);
        return -1;
    }
    if (num_heads % num_heads_k != 0) {
        fprintf(stderr, "kiln_flash_attn_bwd: num_heads (%d) must be divisible by num_heads_k (%d)\n",
                num_heads, num_heads_k);
        return -2;
    }

    // Start with forward params
    FLASH_NAMESPACE::Flash_bwd_params params;
    set_params_fwd(params,
                   batch_size, seqlen_q, seqlen_k,
                   num_heads, num_heads_k, head_dim,
                   q, k, v,
                   const_cast<void *>(out),
                   const_cast<void *>(softmax_lse),
                   softmax_scale, is_causal != 0);

    int seqlen_q_rounded = round_up(seqlen_q, 128);
    int head_dim_rounded = round_up(head_dim, 32);

    // dO pointers and strides (same layout as Q)
    params.do_ptr = const_cast<void *>(dout);
    params.do_row_stride = num_heads * head_dim;
    params.do_head_stride = head_dim;
    params.do_batch_stride = seqlen_q * num_heads * head_dim;

    // dQ pointers and strides
    params.dq_ptr = dq;
    params.dq_row_stride = num_heads * head_dim;
    params.dq_head_stride = head_dim;
    params.dq_batch_stride = seqlen_q * num_heads * head_dim;

    // dK pointers and strides
    params.dk_ptr = dk;
    params.dk_row_stride = num_heads * head_dim;
    params.dk_head_stride = head_dim;
    params.dk_batch_stride = seqlen_k * num_heads * head_dim;

    // dV pointers and strides
    params.dv_ptr = dv;
    params.dv_row_stride = num_heads * head_dim;
    params.dv_head_stride = head_dim;
    params.dv_batch_stride = seqlen_k * num_heads * head_dim;

    // Softmax d (gradient scratch): [batch, num_heads, seqlen_q_rounded]
    params.dsoftmax_sum = softmax_d_out;

    // dQ accumulation buffer: [batch, seqlen_q_rounded, num_heads, head_dim_rounded], float32
    params.dq_accum_ptr = dq_accum;
    params.dk_accum_ptr = nullptr;
    params.dv_accum_ptr = nullptr;

    params.deterministic = (deterministic != 0);
    params.dq_accum_split_stride = seqlen_q_rounded * num_heads * head_dim_rounded;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    if (head_dim == 128) {
        FLASH_NAMESPACE::run_mha_bwd_<cutlass::bfloat16_t, 128, true>(params, cuda_stream);
    } else {
        FLASH_NAMESPACE::run_mha_bwd_<cutlass::bfloat16_t, 256, true>(params, cuda_stream);
    }

    return 0;
}
