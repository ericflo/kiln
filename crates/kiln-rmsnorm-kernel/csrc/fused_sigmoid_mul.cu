#include "fused_sigmoid_mul.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <climits>

namespace {

constexpr int kThreadsPerBlock = 256;

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_sigmoid_mul_bf16_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    __nv_bfloat16 *__restrict__ out,
    int64_t elems
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elems) return;

    const float xv = __bfloat162float(x[idx]);
    const float gv = __bfloat162float(gate[idx]);
    out[idx] = __float2bfloat16(xv * sigmoid(gv));
}

}  // namespace

extern "C" int32_t kiln_fused_sigmoid_mul_bf16(
    const void *x,
    const void *gate,
    void *out,
    int64_t elems,
    void *stream
) {
    if (elems < 0) return -1;
    if (elems == 0) return 0;

    const int64_t blocks64 = (elems + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blocks64 > static_cast<int64_t>(INT_MAX)) return -2;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    fused_sigmoid_mul_bf16_kernel<<<static_cast<int>(blocks64), kThreadsPerBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(x),
        reinterpret_cast<const __nv_bfloat16 *>(gate),
        reinterpret_cast<__nv_bfloat16 *>(out),
        elems);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int32_t>(err);
    return 0;
}
