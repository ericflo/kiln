#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

template <typename T>
__device__ inline float kiln_to_float(T x);

template <>
__device__ inline float kiln_to_float<float>(float x) {
    return x;
}

template <>
__device__ inline float kiln_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ inline T kiln_from_float(float x);

template <>
__device__ inline float kiln_from_float<float>(float x) {
    return x;
}

template <>
__device__ inline __nv_bfloat16 kiln_from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <typename T>
__global__ void sgd_step_kernel(T* param, const T* grad, float lr, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float p = kiln_to_float<T>(param[i]);
    float g = kiln_to_float<T>(grad[i]);
    param[i] = kiln_from_float<T>(p - lr * g);
}

template <typename T>
__global__ void adamw_step_kernel(
    T* param,
    const T* grad,
    T* m,
    T* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float p0 = kiln_to_float<T>(param[i]);
    float g = kiln_to_float<T>(grad[i]);
    float m0 = kiln_to_float<T>(m[i]);
    float v0 = kiln_to_float<T>(v[i]);

    float p_after_wd = p0 * (1.0f - lr * weight_decay);
    float m_new = beta1 * m0 + (1.0f - beta1) * g;
    float v_new = beta2 * v0 + (1.0f - beta2) * g * g;
    float m_hat = m_new / bias_correction1;
    float v_hat = v_new / bias_correction2;
    float update = m_hat / (sqrtf(v_hat) + eps);

    param[i] = kiln_from_float<T>(p_after_wd - lr * update);
    m[i] = kiln_from_float<T>(m_new);
    v[i] = kiln_from_float<T>(v_new);
}

extern "C" int kiln_sgd_step_f32(
    float* param,
    const float* grad,
    float lr,
    int64_t n,
    cudaStream_t stream) {
    if (!param || !grad || n < 0) {
        return 1;
    }
    if (n == 0) {
        return 0;
    }
    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);
    sgd_step_kernel<float><<<blocks, threads, 0, stream>>>(param, grad, lr, n);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int kiln_sgd_step_bf16(
    __nv_bfloat16* param,
    const __nv_bfloat16* grad,
    float lr,
    int64_t n,
    cudaStream_t stream) {
    if (!param || !grad || n < 0) {
        return 1;
    }
    if (n == 0) {
        return 0;
    }
    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);
    sgd_step_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(param, grad, lr, n);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int kiln_adamw_step_f32(
    float* param,
    const float* grad,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    int64_t n,
    cudaStream_t stream) {
    if (!param || !grad || !m || !v || n < 0 || bias_correction1 <= 0.0f || bias_correction2 <= 0.0f) {
        return 1;
    }
    if (n == 0) {
        return 0;
    }
    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);
    adamw_step_kernel<float><<<blocks, threads, 0, stream>>>(
        param, grad, m, v, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, n);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int kiln_adamw_step_bf16(
    __nv_bfloat16* param,
    const __nv_bfloat16* grad,
    __nv_bfloat16* m,
    __nv_bfloat16* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    int64_t n,
    cudaStream_t stream) {
    if (!param || !grad || !m || !v || n < 0 || bias_correction1 <= 0.0f || bias_correction2 <= 0.0f) {
        return 1;
    }
    if (n == 0) {
        return 0;
    }
    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);
    adamw_step_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        param, grad, m, v, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, n);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
