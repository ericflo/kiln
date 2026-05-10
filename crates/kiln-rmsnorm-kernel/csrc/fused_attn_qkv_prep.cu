#include "fused_attn_qkv_prep.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxWarps = kThreadsPerBlock / 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float *smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        smem[warp] = v;
    }
    __syncthreads();

    if (warp == 0) {
        const int num_warps = blockDim.x >> 5;
        float w = (lane < num_warps) ? smem[lane] : 0.0f;
        w = warp_reduce_sum(w);
        if (lane == 0) {
            smem[0] = w;
        }
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ __nv_bfloat16 qwen_rmsnorm_value_bf16(
    const __nv_bfloat16 *__restrict__ row,
    const __nv_bfloat16 *__restrict__ weight,
    int d,
    float rms_inv
) {
    const float x = __bfloat162float(row[d]);
    const float w = __bfloat162float(weight[d]);
    return __float2bfloat16((1.0f + w) * x * rms_inv);
}

__global__ void attn_decode_qkv_split_qk_norm_rope_kernel(
    const __nv_bfloat16 *__restrict__ q_raw,
    const __nv_bfloat16 *__restrict__ k_raw,
    const __nv_bfloat16 *__restrict__ q_weight,
    const __nv_bfloat16 *__restrict__ k_weight,
    const float *__restrict__ cos,
    const float *__restrict__ sin,
    __nv_bfloat16 *__restrict__ q_out,
    __nv_bfloat16 *__restrict__ k_out,
    __nv_bfloat16 *__restrict__ gate_out,
    int batch,
    int q_heads,
    int k_heads,
    int head_dim,
    int rotary_dim,
    int has_gate,
    float eps
) {
    const int row = blockIdx.x;
    const int q_rows = batch * q_heads;
    const bool is_q = row < q_rows;
    const int local_row = is_q ? row : row - q_rows;
    const int heads = is_q ? q_heads : k_heads;
    const int b = local_row / heads;
    const int h = local_row - b * heads;
    const int q_stride = head_dim * (has_gate ? 2 : 1);

    const __nv_bfloat16 *raw_row = is_q
        ? q_raw + (static_cast<size_t>(b) * q_heads + h) * q_stride
        : k_raw + (static_cast<size_t>(b) * k_heads + h) * head_dim;
    const __nv_bfloat16 *weight = is_q ? q_weight : k_weight;
    __nv_bfloat16 *out_row = is_q
        ? q_out + (static_cast<size_t>(b) * q_heads + h) * head_dim
        : k_out + (static_cast<size_t>(b) * k_heads + h) * head_dim;

    __shared__ float smem[kMaxWarps];

    float local_sum_sq = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float x = __bfloat162float(raw_row[d]);
        local_sum_sq += x * x;
    }
    const float total_sum_sq = block_reduce_sum(local_sum_sq, smem);

    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) {
        s_rms_inv = rsqrtf(total_sum_sq / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();
    const float rms_inv = s_rms_inv;

    const int half = rotary_dim / 2;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float y;
        if (d < rotary_dim) {
            const bool first_half = d < half;
            const int pair_d = first_half ? d + half : d - half;
            const int table_d = first_half ? d : pair_d;
            const float c = cos[table_d];
            const float s = sin[table_d];
            const float x = __bfloat162float(qwen_rmsnorm_value_bf16(raw_row, weight, d, rms_inv));
            const float pair =
                __bfloat162float(qwen_rmsnorm_value_bf16(raw_row, weight, pair_d, rms_inv));
            y = first_half ? (x * c - pair * s) : (pair * s + x * c);
        } else {
            y = __bfloat162float(qwen_rmsnorm_value_bf16(raw_row, weight, d, rms_inv));
        }
        out_row[d] = __float2bfloat16(y);
    }

    if (is_q && has_gate && gate_out != nullptr) {
        const __nv_bfloat16 *gate_row = raw_row + head_dim;
        __nv_bfloat16 *gate_dst = gate_out + (static_cast<size_t>(b) * q_heads + h) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            gate_dst[d] = gate_row[d];
        }
    }
}

}  // namespace

extern "C" int32_t kiln_attn_decode_qkv_split_qk_norm_rope_bf16(
    const void *q_raw,
    const void *k_raw,
    const void *q_weight,
    const void *k_weight,
    const float *cos,
    const float *sin,
    void *q_out,
    void *k_out,
    void *gate_out,
    int32_t batch,
    int32_t q_heads,
    int32_t k_heads,
    int32_t head_dim,
    int32_t rotary_dim,
    int32_t has_gate,
    float eps,
    void *stream
) {
    if (batch <= 0 || q_heads <= 0 || k_heads <= 0 || head_dim <= 0) return -1;
    if (rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 1)) return -2;
    if (head_dim > 8192) return -3;
    if (has_gate != 0 && has_gate != 1) return -4;
    if (has_gate && gate_out == nullptr) return -5;

    const int rows = batch * (q_heads + k_heads);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    attn_decode_qkv_split_qk_norm_rope_kernel<<<rows, kThreadsPerBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(q_raw),
        reinterpret_cast<const __nv_bfloat16 *>(k_raw),
        reinterpret_cast<const __nv_bfloat16 *>(q_weight),
        reinterpret_cast<const __nv_bfloat16 *>(k_weight),
        cos,
        sin,
        reinterpret_cast<__nv_bfloat16 *>(q_out),
        reinterpret_cast<__nv_bfloat16 *>(k_out),
        reinterpret_cast<__nv_bfloat16 *>(gate_out),
        batch,
        q_heads,
        k_heads,
        head_dim,
        rotary_dim,
        has_gate,
        eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}
