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

// =====================================================================
// Fused Muon optimizer step (momentum-orthogonalized SGD).
//
// One threadblock per matrix (gridDim.x == 1) so __syncthreads() is a
// full barrier across the whole matrix. Updates `param` and `momentum`
// in place; `grad` is read-only. Mirrors the CPU oracle
// `kiln-optim::lion_muon::{Muon::step, newton_schulz}` exactly.
//
// K_MAX bounds the orthogonalization gram dim k = min(rows, cols):
// four k*k float scratch buffers fit in static shared memory
// (4 * 48 * 48 * 4 == 36 KB < 48 KB). Larger ranks (or non-matrix
// params) fall back to plain (Nesterov) momentum SGD inside the kernel.
// =====================================================================

#define KILN_MUON_K_MAX 48
#define KILN_MUON_BLK 256

template <typename T>
__global__ void muon_step_kernel(
    T* param,
    const T* grad,
    T* momentum,
    float lr,
    float mom,
    int nesterov,
    int ns_iters,
    float wd,
    int rows,
    int cols) {
    const int tid = threadIdx.x;
    const int64_t n = (int64_t)rows * (int64_t)cols;

    const int k = rows < cols ? rows : cols;
    const bool transpose = rows > cols;
    const int maxdim = rows > cols ? rows : cols;
    const float scale = sqrtf((float)maxdim);
    const bool do_ortho = (rows >= 2 && cols >= 2 && k <= KILN_MUON_K_MAX);

    __shared__ float A[KILN_MUON_K_MAX * KILN_MUON_K_MAX];
    __shared__ float M[KILN_MUON_K_MAX * KILN_MUON_K_MAX];
    __shared__ float P[KILN_MUON_K_MAX * KILN_MUON_K_MAX];
    __shared__ float Tm[KILN_MUON_K_MAX * KILN_MUON_K_MAX];
    __shared__ float red[KILN_MUON_BLK];

    // STEP 1 — heavy-ball momentum update, in place:
    //   momentum = mom * momentum + grad
    for (int64_t idx = tid; idx < n; idx += KILN_MUON_BLK) {
        float mnew = mom * kiln_to_float<T>(momentum[idx]) + kiln_to_float<T>(grad[idx]);
        momentum[idx] = kiln_from_float<T>(mnew);
    }
    __syncthreads();

    // bval(idx): the (Nesterov) look-ahead direction. `momentum` already
    // holds the post-update heavy-ball state; `grad` stays read-only.
    // STEP 1b — non-matrix / oversized: plain (Nesterov) momentum SGD.
    if (!do_ortho) {
        for (int64_t idx = tid; idx < n; idx += KILN_MUON_BLK) {
            float b = nesterov
                          ? (kiln_to_float<T>(grad[idx]) + mom * kiln_to_float<T>(momentum[idx]))
                          : kiln_to_float<T>(momentum[idx]);
            float p = kiln_to_float<T>(param[idx]);
            p = p * (1.0f - lr * wd) - lr * b;
            param[idx] = kiln_from_float<T>(p);
        }
        return;
    }

    // STEP 2 — Frobenius norm of b: frob2 = sum_idx bval(idx)^2.
    float partial = 0.0f;
    for (int64_t idx = tid; idx < n; idx += KILN_MUON_BLK) {
        float b = nesterov
                      ? (kiln_to_float<T>(grad[idx]) + mom * kiln_to_float<T>(momentum[idx]))
                      : kiln_to_float<T>(momentum[idx]);
        partial += b * b;
    }
    red[tid] = partial;
    __syncthreads();
    for (int s = KILN_MUON_BLK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        __syncthreads();
    }
    float frob2 = red[0];
    __syncthreads();

    if (frob2 == 0.0f) {
        // Only decoupled weight decay acts (update direction is zero).
        for (int64_t idx = tid; idx < n; idx += KILN_MUON_BLK) {
            float p = kiln_to_float<T>(param[idx]);
            param[idx] = kiln_from_float<T>(p * (1.0f - lr * wd));
        }
        return;
    }
    float inv_frob = 1.0f / sqrtf(frob2);
    float inv_frob2 = 1.0f / frob2;

    // STEP 3 — gram A (k x k), normalized by inv_frob2.
    for (int pair = tid; pair < k * k; pair += KILN_MUON_BLK) {
        int i = pair / k;
        int j = pair % k;
        float s = 0.0f;
        if (!transpose) {
            // A = B Bt  (rows-space gram, k == rows).
            for (int c = 0; c < cols; ++c) {
                float bi = nesterov
                               ? (kiln_to_float<T>(grad[i * cols + c]) +
                                  mom * kiln_to_float<T>(momentum[i * cols + c]))
                               : kiln_to_float<T>(momentum[i * cols + c]);
                float bj = nesterov
                               ? (kiln_to_float<T>(grad[j * cols + c]) +
                                  mom * kiln_to_float<T>(momentum[j * cols + c]))
                               : kiln_to_float<T>(momentum[j * cols + c]);
                s += bi * bj;
            }
        } else {
            // A = Bt B  (cols-space gram, k == cols).
            for (int r = 0; r < rows; ++r) {
                float bi = nesterov
                               ? (kiln_to_float<T>(grad[r * cols + i]) +
                                  mom * kiln_to_float<T>(momentum[r * cols + i]))
                               : kiln_to_float<T>(momentum[r * cols + i]);
                float bj = nesterov
                               ? (kiln_to_float<T>(grad[r * cols + j]) +
                                  mom * kiln_to_float<T>(momentum[r * cols + j]))
                               : kiln_to_float<T>(momentum[r * cols + j]);
                s += bi * bj;
            }
        }
        A[i * k + j] = s * inv_frob2;
    }
    __syncthreads();

    // STEP 4 — P-accumulator Newton-Schulz. P starts = I_k.
    const float ca = 3.4445f;
    const float cb = -4.7750f;
    const float cc = 2.0315f;
    for (int idx = tid; idx < k * k; idx += KILN_MUON_BLK) {
        P[idx] = (idx / k == idx % k) ? 1.0f : 0.0f;
    }
    __syncthreads();

    for (int iter = 0; iter < ns_iters; ++iter) {
        // Tm = A @ A.
        for (int pair = tid; pair < k * k; pair += KILN_MUON_BLK) {
            int i = pair / k;
            int j = pair % k;
            float s = 0.0f;
            for (int t = 0; t < k; ++t) {
                s += A[i * k + t] * A[t * k + j];
            }
            Tm[pair] = s;
        }
        __syncthreads();
        // M = a*I + b*A + c*Tm.
        for (int pair = tid; pair < k * k; pair += KILN_MUON_BLK) {
            int i = pair / k;
            int j = pair % k;
            M[pair] = ca * ((i == j) ? 1.0f : 0.0f) + cb * A[pair] + cc * Tm[pair];
        }
        __syncthreads();
        // P <- (!transpose ? M @ P : P @ M).
        for (int pair = tid; pair < k * k; pair += KILN_MUON_BLK) {
            int i = pair / k;
            int j = pair % k;
            float s = 0.0f;
            if (!transpose) {
                for (int t = 0; t < k; ++t) {
                    s += M[i * k + t] * P[t * k + j];
                }
            } else {
                for (int t = 0; t < k; ++t) {
                    s += P[i * k + t] * M[t * k + j];
                }
            }
            Tm[pair] = s;
        }
        __syncthreads();
        for (int idx = tid; idx < k * k; idx += KILN_MUON_BLK) {
            P[idx] = Tm[idx];
        }
        __syncthreads();
        // A <- M @ A @ M  (M symmetric): Tm = M@A ; A = Tm@M.
        for (int pair = tid; pair < k * k; pair += KILN_MUON_BLK) {
            int i = pair / k;
            int j = pair % k;
            float s = 0.0f;
            for (int t = 0; t < k; ++t) {
                s += M[i * k + t] * A[t * k + j];
            }
            Tm[pair] = s;
        }
        __syncthreads();
        for (int pair = tid; pair < k * k; pair += KILN_MUON_BLK) {
            int i = pair / k;
            int j = pair % k;
            float s = 0.0f;
            for (int t = 0; t < k; ++t) {
                s += Tm[i * k + t] * M[t * k + j];
            }
            A[pair] = s;
        }
        __syncthreads();
    }

    // STEP 5 — apply: O = (P@B or B@P) * inv_frob, RMS-scaled; descent
    // with decoupled weight decay.
    for (int64_t idx = tid; idx < n; idx += KILN_MUON_BLK) {
        int r = (int)(idx / cols);
        int c = (int)(idx % cols);
        float o = 0.0f;
        if (!transpose) {
            // P is rows x rows (k == rows): O[r,c] = Σ_t P[r,t] * B[t,c].
            for (int t = 0; t < k; ++t) {
                float bt = nesterov
                               ? (kiln_to_float<T>(grad[t * cols + c]) +
                                  mom * kiln_to_float<T>(momentum[t * cols + c]))
                               : kiln_to_float<T>(momentum[t * cols + c]);
                o += P[r * k + t] * bt;
            }
        } else {
            // P is cols x cols (k == cols): O[r,c] = Σ_t B[r,t] * P[t,c].
            for (int t = 0; t < k; ++t) {
                float bt = nesterov
                               ? (kiln_to_float<T>(grad[r * cols + t]) +
                                  mom * kiln_to_float<T>(momentum[r * cols + t]))
                               : kiln_to_float<T>(momentum[r * cols + t]);
                o += bt * P[t * k + c];
            }
        }
        o *= scale * inv_frob;
        float p = kiln_to_float<T>(param[idx]);
        p = p * (1.0f - lr * wd) - lr * o;
        param[idx] = kiln_from_float<T>(p);
    }
}

extern "C" int kiln_muon_step_f32(
    float* param,
    const float* grad,
    float* m,
    float lr,
    float mom,
    int nesterov,
    int ns_iters,
    float wd,
    int rows,
    int cols,
    cudaStream_t stream) {
    if (!param || !grad || !m || rows < 0 || cols < 0) {
        return 1;
    }
    if (rows == 0 || cols == 0) {
        return 0;
    }
    muon_step_kernel<float><<<1, KILN_MUON_BLK, 0, stream>>>(
        param, grad, m, lr, mom, nesterov, ns_iters, wd, rows, cols);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int kiln_muon_step_bf16(
    __nv_bfloat16* param,
    const __nv_bfloat16* grad,
    __nv_bfloat16* m,
    float lr,
    float mom,
    int nesterov,
    int ns_iters,
    float wd,
    int rows,
    int cols,
    cudaStream_t stream) {
    if (!param || !grad || !m || rows < 0 || cols < 0) {
        return 1;
    }
    if (rows == 0 || cols == 0) {
        return 0;
    }
    muon_step_kernel<__nv_bfloat16><<<1, KILN_MUON_BLK, 0, stream>>>(
        param, grad, m, lr, mom, nesterov, ns_iters, wd, rows, cols);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
