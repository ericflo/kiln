#include "fused_rotary_qk.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void fused_rotary_qk_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const float *__restrict__ cos,
    const float *__restrict__ sin,
    __nv_bfloat16 *__restrict__ q_out,
    __nv_bfloat16 *__restrict__ k_out,
    size_t q_elems,
    size_t k_elems,
    int seq_len,
    int q_heads,
    int k_heads,
    int head_dim,
    int rotary_dim
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = q_elems + k_elems;
    if (idx >= total) return;

    const bool is_q = idx < q_elems;
    const size_t local = is_q ? idx : idx - q_elems;
    const int heads = is_q ? q_heads : k_heads;
    const __nv_bfloat16 *in = is_q ? q : k;
    __nv_bfloat16 *out = is_q ? q_out : k_out;

    const int d = static_cast<int>(local % head_dim);
    const size_t row = local / head_dim;
    const int t = static_cast<int>((row / heads) % seq_len);

    float y;
    if (d < rotary_dim) {
        const int half = rotary_dim / 2;
        const bool first_half = d < half;
        const int pair_d = first_half ? d + half : d - half;
        const int table_d = first_half ? d : pair_d;
        const float c = cos[static_cast<size_t>(t) * half + table_d];
        const float s = sin[static_cast<size_t>(t) * half + table_d];
        const size_t base = row * head_dim;
        const float x = __bfloat162float(in[base + d]);
        const float pair = __bfloat162float(in[base + pair_d]);
        y = first_half ? (x * c - pair * s) : (pair * s + x * c);
    } else {
        y = __bfloat162float(in[local]);
    }

    out[local] = __float2bfloat16(y);
}

__global__ void fused_rotary_one_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ cos,
    const float *__restrict__ sin,
    __nv_bfloat16 *__restrict__ out,
    size_t elems,
    int seq_len,
    int heads,
    int head_dim,
    int rotary_dim
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) return;

    const int d = static_cast<int>(idx % head_dim);
    const size_t row = idx / head_dim;
    const int t = static_cast<int>((row / heads) % seq_len);

    float y;
    if (d < rotary_dim) {
        const int half = rotary_dim / 2;
        const bool first_half = d < half;
        const int pair_d = first_half ? d + half : d - half;
        const int table_d = first_half ? d : pair_d;
        const float c = cos[static_cast<size_t>(t) * half + table_d];
        const float s = sin[static_cast<size_t>(t) * half + table_d];
        const size_t base = row * head_dim;
        const float xv = __bfloat162float(x[base + d]);
        const float pair = __bfloat162float(x[base + pair_d]);
        y = first_half ? (xv * c - pair * s) : (pair * s + xv * c);
    } else {
        y = __bfloat162float(x[idx]);
    }

    out[idx] = __float2bfloat16(y);
}

__global__ void fused_rotary_one_bwd_kernel(
    const __nv_bfloat16 *__restrict__ grad_y,
    const float *__restrict__ cos,
    const float *__restrict__ sin,
    __nv_bfloat16 *__restrict__ grad_x,
    size_t elems,
    int seq_len,
    int heads,
    int head_dim,
    int rotary_dim
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) return;

    const int d = static_cast<int>(idx % head_dim);
    const size_t row = idx / head_dim;
    const int t = static_cast<int>((row / heads) % seq_len);

    float y;
    if (d < rotary_dim) {
        const int half = rotary_dim / 2;
        const bool first_half = d < half;
        const int pair_d = first_half ? d + half : d - half;
        const int table_d = first_half ? d : pair_d;
        const float c = cos[static_cast<size_t>(t) * half + table_d];
        const float s = sin[static_cast<size_t>(t) * half + table_d];
        const size_t base = row * head_dim;
        const float gv = __bfloat162float(grad_y[base + d]);
        const float pair = __bfloat162float(grad_y[base + pair_d]);
        y = first_half ? (gv * c + pair * s) : (gv * c - pair * s);
    } else {
        y = __bfloat162float(grad_y[idx]);
    }

    grad_x[idx] = __float2bfloat16(y);
}

}  // namespace

extern "C" int32_t kiln_fused_rotary_qk(
    const void *q,
    const void *k,
    const float *cos,
    const float *sin,
    void *q_out,
    void *k_out,
    int32_t batch,
    int32_t seq_len,
    int32_t q_heads,
    int32_t k_heads,
    int32_t head_dim,
    int32_t rotary_dim,
    void *stream
) {
    if (batch <= 0 || seq_len <= 0 || q_heads <= 0 || k_heads <= 0) return -1;
    if (head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 1)) return -2;

    const size_t q_elems = static_cast<size_t>(batch) * seq_len * q_heads * head_dim;
    const size_t k_elems = static_cast<size_t>(batch) * seq_len * k_heads * head_dim;
    const size_t total = q_elems + k_elems;
    const int blocks = static_cast<int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    fused_rotary_qk_kernel<<<blocks, kThreadsPerBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(q),
        reinterpret_cast<const __nv_bfloat16 *>(k),
        cos,
        sin,
        reinterpret_cast<__nv_bfloat16 *>(q_out),
        reinterpret_cast<__nv_bfloat16 *>(k_out),
        q_elems,
        k_elems,
        seq_len,
        q_heads,
        k_heads,
        head_dim,
        rotary_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

extern "C" int32_t kiln_fused_rotary_one(
    const void *x,
    const float *cos,
    const float *sin,
    void *out,
    int32_t batch,
    int32_t seq_len,
    int32_t heads,
    int32_t head_dim,
    int32_t rotary_dim,
    void *stream
) {
    if (batch <= 0 || seq_len <= 0 || heads <= 0) return -1;
    if (head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 1)) return -2;

    const size_t elems = static_cast<size_t>(batch) * seq_len * heads * head_dim;
    const int blocks = static_cast<int>((elems + kThreadsPerBlock - 1) / kThreadsPerBlock);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    fused_rotary_one_kernel<<<blocks, kThreadsPerBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(x),
        cos,
        sin,
        reinterpret_cast<__nv_bfloat16 *>(out),
        elems,
        seq_len,
        heads,
        head_dim,
        rotary_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

extern "C" int32_t kiln_fused_rotary_one_bwd(
    const void *grad_y,
    const float *cos,
    const float *sin,
    void *grad_x,
    int32_t batch,
    int32_t seq_len,
    int32_t heads,
    int32_t head_dim,
    int32_t rotary_dim,
    void *stream
) {
    if (batch <= 0 || seq_len <= 0 || heads <= 0) return -1;
    if (head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 1)) return -2;

    const size_t elems = static_cast<size_t>(batch) * seq_len * heads * head_dim;
    const int blocks = static_cast<int>((elems + kThreadsPerBlock - 1) / kThreadsPerBlock);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    fused_rotary_one_bwd_kernel<<<blocks, kThreadsPerBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(grad_y),
        cos,
        sin,
        reinterpret_cast<__nv_bfloat16 *>(grad_x),
        elems,
        seq_len,
        heads,
        head_dim,
        rotary_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}
