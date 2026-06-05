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

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
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

// One CUDA block computes dx/dz for one row of length 128. d_weight is
// accumulated in F32 with one atomic per hidden element per row.
__global__ void gdn_gated_rms_norm_bwd_bf16_kernel(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ d_x,
    __nv_bfloat16* __restrict__ d_z,
    float* __restrict__ d_weight,
    int rows,
    float eps
) {
    __shared__ float scratch[kHidden];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    const int base = row * kHidden;
    const float x_val = __bfloat162float(x[base + tid]);
    const float z_val = __bfloat162float(z[base + tid]);
    const float w_val = __bfloat162float(weight[tid]);
    const float dout = __bfloat162float(grad_out[base + tid]);

    scratch[tid] = x_val * x_val;
    __syncthreads();

    for (int stride = kHidden / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    const float rms_inv = rsqrtf((scratch[0] / static_cast<float>(kHidden)) + eps);
    const float sig = sigmoid(z_val);
    const float gate = z_val * sig;
    const float d_normed = dout * gate;

    scratch[tid] = d_normed * x_val * w_val;
    __syncthreads();

    for (int stride = kHidden / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    const float s = scratch[0];
    const float rms_inv3 = rms_inv * rms_inv * rms_inv;
    const float normed = x_val * w_val * rms_inv;
    const float silu_grad = sig * (1.0f + z_val * (1.0f - sig));

    const float dx = d_normed * w_val * rms_inv
        - x_val * s * (rms_inv3 / static_cast<float>(kHidden));
    const float dz = dout * normed * silu_grad;
    const float dw = d_normed * x_val * rms_inv;

    d_x[base + tid] = __float2bfloat16(dx);
    d_z[base + tid] = __float2bfloat16(dz);
    atomicAdd(&d_weight[tid], dw);
}

// One CUDA block computes the backward of y = scale * x / sqrt(sum(x^2) + eps)
// for one contiguous row of length 128.
__global__ void gdn_l2_norm_scale_bwd_bf16_kernel(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ d_x,
    int rows,
    float scale,
    float eps
) {
    __shared__ float scratch[kHidden];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    const int base = row * kHidden;
    const float x_val = __bfloat162float(x[base + tid]);
    const float dy = __bfloat162float(grad_out[base + tid]);

    scratch[tid] = x_val * x_val;
    __syncthreads();

    for (int stride = kHidden / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    const float inv_n = rsqrtf(scratch[0] + eps);

    scratch[tid] = dy * x_val;
    __syncthreads();

    for (int stride = kHidden / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    const float s = scratch[0];
    const float inv_n3 = inv_n * inv_n * inv_n;
    const float dx = scale * (dy * inv_n - x_val * s * inv_n3);
    d_x[base + tid] = __float2bfloat16(dx);
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

extern "C" int32_t kiln_gdn_gated_rms_norm_bwd_bf16(
    const void* grad_out,
    const void* x,
    const void* z,
    const void* weight,
    void* d_x,
    void* d_z,
    void* d_weight,
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

    gdn_gated_rms_norm_bwd_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_out),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(z),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(d_x),
        reinterpret_cast<__nv_bfloat16*>(d_z),
        reinterpret_cast<float*>(d_weight),
        rows,
        eps
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

extern "C" int32_t kiln_gdn_l2_norm_scale_bwd_bf16(
    const void* grad_out,
    const void* x,
    void* d_x,
    int32_t rows,
    int32_t hidden,
    float scale,
    float eps,
    void* stream_raw
) {
    if (rows <= 0) return 0;
    if (hidden != kHidden) return 2;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    dim3 grid(rows);
    dim3 block(kHidden);

    gdn_l2_norm_scale_bwd_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_out),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(d_x),
        rows,
        scale,
        eps
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
