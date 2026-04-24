// kiln-gdn-kernel: fused GDN gated RMSNorm kernel
//
// Collapses the `kiln/gdn/gated_norm` bf16 decode/prefill body from the
// portable candle chain:
//
//   x_f32 -> rms_norm(x, weight, eps) -> silu(z_f32) -> mul -> bf16
//
// into one CUDA launch. Scope is intentionally narrow for Qwen3.5 GDN:
// bf16 inputs/weight, hidden=128, contiguous row-major last dimension.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#include "gdn_gated_rms_norm.h"

namespace {

constexpr int kHidden = 128;

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// One CUDA block computes one contiguous row of length 128.
__global__ void gdn_gated_rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int rows,
    float eps
) {
    __shared__ float scratch[kHidden];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    const int base = row * kHidden;
    const float x_val = __bfloat162float(x[base + tid]);
    scratch[tid] = x_val * x_val;
    __syncthreads();

    for (int stride = kHidden / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    const float rms_inv = rsqrtf((scratch[0] / static_cast<float>(kHidden)) + eps);
    const float z_val = __bfloat162float(z[base + tid]);
    const float w_val = __bfloat162float(weight[tid]);
    const float out_val = x_val * rms_inv * w_val * silu(z_val);
    out[base + tid] = __float2bfloat16(out_val);
}

}  // namespace

extern "C" int32_t kiln_gdn_gated_rms_norm_bf16(
    const void* x,
    const void* z,
    const void* weight,
    void* out,
    int32_t rows,
    int32_t hidden,
    float eps,
    void* stream_raw
) {
    if (rows <= 0) return 0;
    if (hidden != kHidden) return 2;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    dim3 grid(rows);
    dim3 block(kHidden);

    gdn_gated_rms_norm_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(z),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out),
        rows,
        eps
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
