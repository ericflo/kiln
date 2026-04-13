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
