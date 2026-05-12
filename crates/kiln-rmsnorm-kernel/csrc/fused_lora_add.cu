#include "fused_lora_add.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <climits>

namespace {

constexpr int kDotThreads = 256;
constexpr int kAddThreads = 256;

__global__ void lora_decode_hidden_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ a,
    float *__restrict__ hidden,
    int in_dim,
    int rank
) {
    const int row = blockIdx.x;
    const int r = blockIdx.y;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const float xv = __bfloat162float(x[row * in_dim + i]);
        const float av = __bfloat162float(a[r * in_dim + i]);
        acc += xv * av;
    }

    __shared__ float smem[kDotThreads];
    smem[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        hidden[row * rank + r] = smem[0];
    }
}

__global__ void lora_decode_add_kernel(
    const __nv_bfloat16 *__restrict__ base,
    const float *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ out,
    float scale,
    int64_t elems,
    int out_dim,
    int rank
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) return;

    const int row = static_cast<int>(idx / out_dim);
    const int j = static_cast<int>(idx - static_cast<int64_t>(row) * out_dim);
    float delta = 0.0f;
    for (int r = 0; r < rank; ++r) {
        delta += hidden[row * rank + r] * __bfloat162float(b[j * rank + r]);
    }
    const float base_v = __bfloat162float(base[idx]);
    out[idx] = __float2bfloat16(base_v + scale * delta);
}

__global__ void lora_add_inplace_f32_kernel(
    float *__restrict__ base,
    const float *__restrict__ hidden,
    const float *__restrict__ b,
    float scale,
    int64_t elems,
    int out_dim,
    int rank
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) return;

    const int row = static_cast<int>(idx / out_dim);
    const int j = static_cast<int>(idx - static_cast<int64_t>(row) * out_dim);
    float delta = 0.0f;
    for (int r = 0; r < rank; ++r) {
        delta += hidden[row * rank + r] * b[j * rank + r];
    }
    base[idx] += scale * delta;
}

}  // namespace

extern "C" int32_t kiln_lora_decode_hidden_bf16(
    const void *x,
    const void *a,
    float *hidden,
    int32_t batch,
    int32_t in_dim,
    int32_t rank,
    void *stream
) {
    if (batch <= 0 || in_dim <= 0 || rank <= 0) return -1;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(static_cast<unsigned int>(batch), static_cast<unsigned int>(rank), 1);
    lora_decode_hidden_kernel<<<grid, kDotThreads, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(x),
        reinterpret_cast<const __nv_bfloat16 *>(a),
        hidden,
        static_cast<int>(in_dim),
        static_cast<int>(rank));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

extern "C" int32_t kiln_lora_add_inplace_f32(
    float *base,
    const float *hidden,
    const float *b,
    float scale,
    int32_t rows,
    int32_t out_dim,
    int32_t rank,
    void *stream
) {
    if (rows <= 0 || out_dim <= 0 || rank <= 0) return -1;
    const int64_t elems = static_cast<int64_t>(rows) * static_cast<int64_t>(out_dim);
    const int64_t blocks64 = (elems + kAddThreads - 1) / kAddThreads;
    if (blocks64 > static_cast<int64_t>(INT_MAX)) return -2;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    lora_add_inplace_f32_kernel<<<static_cast<int>(blocks64), kAddThreads, 0, s>>>(
        base,
        hidden,
        b,
        scale,
        elems,
        static_cast<int>(out_dim),
        static_cast<int>(rank));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}

extern "C" int32_t kiln_lora_decode_add_bf16(
    const void *base,
    const float *hidden,
    const void *b,
    void *out,
    float scale,
    int32_t batch,
    int32_t out_dim,
    int32_t rank,
    void *stream
) {
    if (batch <= 0 || out_dim <= 0 || rank <= 0) return -1;
    const int64_t elems = static_cast<int64_t>(batch) * static_cast<int64_t>(out_dim);
    const int64_t blocks64 = (elems + kAddThreads - 1) / kAddThreads;
    if (blocks64 > static_cast<int64_t>(INT_MAX)) return -2;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    lora_decode_add_kernel<<<static_cast<int>(blocks64), kAddThreads, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(base),
        hidden,
        reinterpret_cast<const __nv_bfloat16 *>(b),
        reinterpret_cast<__nv_bfloat16 *>(out),
        scale,
        elems,
        static_cast<int>(out_dim),
        static_cast<int>(rank));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}
