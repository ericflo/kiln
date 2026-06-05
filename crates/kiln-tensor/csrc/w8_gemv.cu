// ROCm/CUDA-compatible W8A16 GEMV for single-token decode projections.
//
// Computes C[m, n] = A[m, k] @ dequant(W_q[n, k])^T where W_q is signed
// int8 stored in a U8 tensor and each output row has one F32 scale.
// A and C are BF16. This is intentionally a bandwidth-cutting decode kernel,
// not a general GEMM replacement.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

#include "kt_gpu_compat.cuh"

namespace {

constexpr int BLOCK = 64;
constexpr int ARGMAX_BLOCK = 256;
constexpr int A8_QUANT_BLOCK = 256;
constexpr int A8_GEMV_BLOCK = 64;

__device__ __forceinline__ int32_t kiln_pack4_i8(const int8_t* p) {
    uint32_t u = static_cast<uint8_t>(p[0]);
    u |= static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 8;
    u |= static_cast<uint32_t>(static_cast<uint8_t>(p[2])) << 16;
    u |= static_cast<uint32_t>(static_cast<uint8_t>(p[3])) << 24;
    return static_cast<int32_t>(u);
}

__device__ __forceinline__ int kiln_sdot4_i8(int32_t a, int32_t b, int c) {
#if KILN_IS_HIP && (defined(__gfx90a__) || defined(__gfx942__))
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    int acc = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int av = static_cast<int8_t>((static_cast<uint32_t>(a) >> (8 * i)) & 0xffu);
        int bv = static_cast<int8_t>((static_cast<uint32_t>(b) >> (8 * i)) & 0xffu);
        acc += av * bv;
    }
    return acc;
#endif
}

__global__ void kiln_w8a8_quantize_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ x_q_u8,
    float* __restrict__ x_scales,
    int64_t m,
    int64_t k) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    int tid = threadIdx.x;
    if (row >= m) return;

    const __nv_bfloat16* x_row = x + row * k;
    uint8_t* q_row_u8 = x_q_u8 + row * k;

    float local_max = 0.0f;
    for (int64_t c = tid; c < k; c += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__bfloat162float(x_row[c])));
    }

    __shared__ float smem[A8_QUANT_BLOCK];
    float max_abs = kiln_block_reduce_max(local_max, smem);
    float scale = max_abs <= 1.0e-12f ? 1.0f : max_abs / 127.0f;
    float inv_scale = 1.0f / scale;
    if (tid == 0) {
        x_scales[row] = scale;
    }

    int8_t* q_row = reinterpret_cast<int8_t*>(q_row_u8);
    for (int64_t c = tid; c < k; c += blockDim.x) {
        float scaled = __bfloat162float(x_row[c]) * inv_scale;
        int q = __float2int_rn(scaled);
        q = q > 127 ? 127 : q;
        q = q < -127 ? -127 : q;
        q_row[c] = static_cast<int8_t>(q);
    }
}

__global__ void kiln_w8a8_gemv_bf16_kernel(
    const uint8_t* __restrict__ x_q_u8,
    const uint8_t* __restrict__ w_q_u8,
    const float* __restrict__ x_scales,
    const float* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ out,
    int64_t m,
    int64_t n,
    int64_t k) {
    int64_t col = static_cast<int64_t>(blockIdx.x);
    int64_t row = static_cast<int64_t>(blockIdx.y);
    int tid = threadIdx.x;
    if (row >= m || col >= n) return;

    const int8_t* x_row = reinterpret_cast<const int8_t*>(x_q_u8 + row * k);
    const int8_t* w_row = reinterpret_cast<const int8_t*>(w_q_u8 + col * k);

    int local = 0;
    int64_t k4 = (k / 4) * 4;
    for (int64_t c = static_cast<int64_t>(tid) * 4; c < k4; c += static_cast<int64_t>(blockDim.x) * 4) {
        int32_t xv = kiln_pack4_i8(x_row + c);
        int32_t wv = kiln_pack4_i8(w_row + c);
        local = kiln_sdot4_i8(xv, wv, local);
    }
    for (int64_t c = k4 + tid; c < k; c += blockDim.x) {
        local += static_cast<int>(x_row[c]) * static_cast<int>(w_row[c]);
    }

    __shared__ int smem[A8_GEMV_BLOCK];
    int sum = kiln_block_reduce_sum(local, smem);
    if (tid == 0) {
        float scale = x_scales[row] * w_scales[col];
        out[row * n + col] = __float2bfloat16(static_cast<float>(sum) * scale);
    }
}

__global__ void kiln_w8a8_swiglu_bf16_kernel(
    const uint8_t* __restrict__ x_q_u8,
    const uint8_t* __restrict__ w_q_u8,
    const float* __restrict__ x_scales,
    const float* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ out,
    int64_t m,
    int64_t gate_up_n,
    int64_t k) {
    int64_t col = static_cast<int64_t>(blockIdx.x);
    int64_t row = static_cast<int64_t>(blockIdx.y);
    int tid = threadIdx.x;
    int64_t g = gate_up_n / 2;
    if (row >= m || col >= g) return;

    const int8_t* x_row = reinterpret_cast<const int8_t*>(x_q_u8 + row * k);
    const int8_t* gate_w = reinterpret_cast<const int8_t*>(w_q_u8 + col * k);
    const int8_t* up_w = reinterpret_cast<const int8_t*>(w_q_u8 + (col + g) * k);

    int gate_local = 0;
    int up_local = 0;
    int64_t k4 = (k / 4) * 4;
    for (int64_t c = static_cast<int64_t>(tid) * 4; c < k4; c += static_cast<int64_t>(blockDim.x) * 4) {
        int32_t xv = kiln_pack4_i8(x_row + c);
        int32_t gate_wv = kiln_pack4_i8(gate_w + c);
        int32_t up_wv = kiln_pack4_i8(up_w + c);
        gate_local = kiln_sdot4_i8(xv, gate_wv, gate_local);
        up_local = kiln_sdot4_i8(xv, up_wv, up_local);
    }
    for (int64_t c = k4 + tid; c < k; c += blockDim.x) {
        int xv = static_cast<int>(x_row[c]);
        gate_local += xv * static_cast<int>(gate_w[c]);
        up_local += xv * static_cast<int>(up_w[c]);
    }

    __shared__ int smem[A8_GEMV_BLOCK];
    int gate_sum = kiln_block_reduce_sum(gate_local, smem);
    int up_sum = kiln_block_reduce_sum(up_local, smem);
    if (tid == 0) {
        float x_scale = x_scales[row];
        float gate = static_cast<float>(gate_sum) * x_scale * w_scales[col];
        float up = static_cast<float>(up_sum) * x_scale * w_scales[col + g];
        float silu = gate / (1.0f + expf(-gate));
        out[row * g + col] = __float2bfloat16(silu * up);
    }
}

__global__ void kiln_w8a16_gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_q_u8,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    int64_t m,
    int64_t n,
    int64_t k) {
    int64_t col = static_cast<int64_t>(blockIdx.x);
    int64_t row = static_cast<int64_t>(blockIdx.y);
    int tid = threadIdx.x;
    if (row >= m || col >= n) return;

    const __nv_bfloat16* x_row = x + row * k;
    const int8_t* w_row = reinterpret_cast<const int8_t*>(w_q_u8 + col * k);

    float local = 0.0f;
    for (int64_t c = tid; c < k; c += BLOCK) {
        float xv = __bfloat162float(x_row[c]);
        float wv = static_cast<float>(w_row[c]);
        local += xv * wv;
    }

    __shared__ float smem[BLOCK];
    float sum = kiln_block_reduce_sum(local, smem);
    if (tid == 0) {
        out[row * n + col] = __float2bfloat16(sum * scales[col]);
    }
}

__global__ void kiln_w8a16_swiglu_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_q_u8,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    int64_t m,
    int64_t gate_up_n,
    int64_t k) {
    int64_t col = static_cast<int64_t>(blockIdx.x);
    int64_t row = static_cast<int64_t>(blockIdx.y);
    int tid = threadIdx.x;
    int64_t g = gate_up_n / 2;
    if (row >= m || col >= g) return;

    const __nv_bfloat16* x_row = x + row * k;
    const int8_t* gate_w = reinterpret_cast<const int8_t*>(w_q_u8 + col * k);
    const int8_t* up_w = reinterpret_cast<const int8_t*>(w_q_u8 + (col + g) * k);

    float gate_local = 0.0f;
    float up_local = 0.0f;
    for (int64_t c = tid; c < k; c += BLOCK) {
        float xv = __bfloat162float(x_row[c]);
        gate_local += xv * static_cast<float>(gate_w[c]);
        up_local += xv * static_cast<float>(up_w[c]);
    }

    __shared__ float smem[BLOCK];
    float gate = kiln_block_reduce_sum(gate_local, smem) * scales[col];
    float up = kiln_block_reduce_sum(up_local, smem) * scales[col + g];
    if (tid == 0) {
        float silu = gate / (1.0f + expf(-gate));
        out[row * g + col] = __float2bfloat16(silu * up);
    }
}

__global__ void kiln_w8a16_gemv_argmax_scores_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_q_u8,
    const float* __restrict__ scales,
    float* __restrict__ scores,
    int64_t n,
    int64_t k) {
    int64_t col = static_cast<int64_t>(blockIdx.x);
    int tid = threadIdx.x;
    if (col >= n) return;

    const int8_t* w_row = reinterpret_cast<const int8_t*>(w_q_u8 + col * k);

    float local = 0.0f;
    for (int64_t c = tid; c < k; c += BLOCK) {
        float xv = __bfloat162float(x[c]);
        float wv = static_cast<float>(w_row[c]);
        local += xv * wv;
    }

    __shared__ float smem[BLOCK];
    float sum = kiln_block_reduce_sum(local, smem);
    if (tid == 0) {
        scores[col] = sum * scales[col];
    }
}

__global__ void kiln_w8a16_gemv_argmax_reduce_kernel(
    const float* __restrict__ scores,
    int64_t* __restrict__ out_idx,
    int64_t n) {
    int tid = threadIdx.x;
    float best_val = -INFINITY;
    int64_t best_idx = 0;
    for (int64_t i = tid; i < n; i += ARGMAX_BLOCK) {
        float v = scores[i];
        if (v > best_val || (v == best_val && i < best_idx)) {
            best_val = v;
            best_idx = i;
        }
    }

    __shared__ float vals[ARGMAX_BLOCK];
    __shared__ int64_t idxs[ARGMAX_BLOCK];
    vals[tid] = best_val;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = ARGMAX_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_val = vals[tid + stride];
            int64_t other_idx = idxs[tid + stride];
            if (other_val > vals[tid] || (other_val == vals[tid] && other_idx < idxs[tid])) {
                vals[tid] = other_val;
                idxs[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[0] = idxs[0];
    }
}

__device__ __forceinline__ uint64_t kiln_splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float kiln_uniform01(uint64_t seed, int64_t idx) {
    uint64_t x = kiln_splitmix64(seed ^ (static_cast<uint64_t>(idx) * 0xd6e8feb86659fd93ull));
    float u = (static_cast<float>((x >> 40) & 0x00ffffffu) + 0.5f) * 5.960464477539063e-8f;
    u = fminf(fmaxf(u, 1.0e-7f), 1.0f - 1.0e-7f);
    return u;
}

__global__ void kiln_w8a16_gemv_gumbel_scores_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_q_u8,
    const float* __restrict__ scales,
    const uint32_t* __restrict__ history_indices,
    const uint32_t* __restrict__ history_counts,
    float* __restrict__ scores,
    int64_t n,
    int64_t k,
    int64_t history_len,
    float repetition_penalty,
    float presence_penalty,
    float frequency_penalty,
    float inv_temperature,
    uint64_t seed) {
    int64_t col = static_cast<int64_t>(blockIdx.x);
    int tid = threadIdx.x;
    if (col >= n) return;

    const int8_t* w_row = reinterpret_cast<const int8_t*>(w_q_u8 + col * k);

    float local = 0.0f;
    for (int64_t c = tid; c < k; c += BLOCK) {
        float xv = __bfloat162float(x[c]);
        float wv = static_cast<float>(w_row[c]);
        local += xv * wv;
    }

    __shared__ float smem[BLOCK];
    float score = kiln_block_reduce_sum(local, smem) * scales[col];
    if (tid == 0) {
        for (int64_t i = 0; i < history_len; ++i) {
            if (static_cast<int64_t>(history_indices[i]) == col) {
                if (repetition_penalty != 1.0f) {
                    score = score > 0.0f ? score / repetition_penalty : score * repetition_penalty;
                }
                score -= presence_penalty;
                score -= frequency_penalty * static_cast<float>(history_counts[i]);
                break;
            }
        }
        score *= inv_temperature;
        const float u = kiln_uniform01(seed, col);
        scores[col] = score - logf(-logf(u));
    }
}

}  // namespace

extern "C" int kiln_w8a16_gemv_bf16_async(
    const void* x,
    const void* w_q,
    const void* scales,
    void* out,
    int64_t m,
    int64_t n,
    int64_t k,
    void* stream_raw) {
    if (m < 0 || n < 0 || k < 0) return 1;
    if (m == 0 || n == 0 || k == 0) return 0;
    if (n > static_cast<int64_t>(2147483647) || m > static_cast<int64_t>(65535)) {
        return 2;
    }

    dim3 grid(static_cast<unsigned int>(n), static_cast<unsigned int>(m), 1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    kiln_w8a16_gemv_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const uint8_t*>(w_q),
        static_cast<const float*>(scales),
        static_cast<__nv_bfloat16*>(out),
        m,
        n,
        k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_w8a16_swiglu_bf16_async(
    const void* x,
    const void* w_q,
    const void* scales,
    void* out,
    int64_t m,
    int64_t gate_up_n,
    int64_t k,
    void* stream_raw) {
    if (m < 0 || gate_up_n < 0 || k < 0) return 1;
    if (m == 0 || gate_up_n == 0 || k == 0) return 0;
    if ((gate_up_n & 1) != 0) return 3;
    int64_t g = gate_up_n / 2;
    if (g > static_cast<int64_t>(2147483647) || m > static_cast<int64_t>(65535)) {
        return 2;
    }

    dim3 grid(static_cast<unsigned int>(g), static_cast<unsigned int>(m), 1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    kiln_w8a16_swiglu_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const uint8_t*>(w_q),
        static_cast<const float*>(scales),
        static_cast<__nv_bfloat16*>(out),
        m,
        gate_up_n,
        k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_w8a8_quantize_bf16_async(
    const void* x,
    void* x_q,
    void* x_scales,
    int64_t m,
    int64_t k,
    void* stream_raw) {
    if (m < 0 || k < 0) return 1;
    if (m == 0 || k == 0) return 0;
    if (m > static_cast<int64_t>(2147483647)) return 2;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    kiln_w8a8_quantize_bf16_kernel<<<static_cast<unsigned int>(m), A8_QUANT_BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<uint8_t*>(x_q),
        static_cast<float*>(x_scales),
        m,
        k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_w8a8_gemv_bf16_async(
    const void* x_q,
    const void* w_q,
    const void* x_scales,
    const void* w_scales,
    void* out,
    int64_t m,
    int64_t n,
    int64_t k,
    void* stream_raw) {
    if (m < 0 || n < 0 || k < 0) return 1;
    if (m == 0 || n == 0 || k == 0) return 0;
    if (n > static_cast<int64_t>(2147483647) || m > static_cast<int64_t>(65535)) {
        return 2;
    }

    dim3 grid(static_cast<unsigned int>(n), static_cast<unsigned int>(m), 1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    kiln_w8a8_gemv_bf16_kernel<<<grid, A8_GEMV_BLOCK, 0, stream>>>(
        static_cast<const uint8_t*>(x_q),
        static_cast<const uint8_t*>(w_q),
        static_cast<const float*>(x_scales),
        static_cast<const float*>(w_scales),
        static_cast<__nv_bfloat16*>(out),
        m,
        n,
        k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_w8a8_swiglu_bf16_async(
    const void* x_q,
    const void* w_q,
    const void* x_scales,
    const void* w_scales,
    void* out,
    int64_t m,
    int64_t gate_up_n,
    int64_t k,
    void* stream_raw) {
    if (m < 0 || gate_up_n < 0 || k < 0) return 1;
    if (m == 0 || gate_up_n == 0 || k == 0) return 0;
    if ((gate_up_n & 1) != 0) return 3;
    int64_t g = gate_up_n / 2;
    if (g > static_cast<int64_t>(2147483647) || m > static_cast<int64_t>(65535)) {
        return 2;
    }

    dim3 grid(static_cast<unsigned int>(g), static_cast<unsigned int>(m), 1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    kiln_w8a8_swiglu_bf16_kernel<<<grid, A8_GEMV_BLOCK, 0, stream>>>(
        static_cast<const uint8_t*>(x_q),
        static_cast<const uint8_t*>(w_q),
        static_cast<const float*>(x_scales),
        static_cast<const float*>(w_scales),
        static_cast<__nv_bfloat16*>(out),
        m,
        gate_up_n,
        k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_w8a16_gemv_argmax_bf16_async(
    const void* x,
    const void* w_q,
    const void* scales,
    void* scores,
    void* out_idx,
    int64_t n,
    int64_t k,
    void* stream_raw) {
    if (n < 0 || k < 0) return 1;
    if (n == 0 || k == 0) return 2;
    if (n > static_cast<int64_t>(2147483647)) return 3;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    kiln_w8a16_gemv_argmax_scores_kernel<<<static_cast<unsigned int>(n), BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const uint8_t*>(w_q),
        static_cast<const float*>(scales),
        static_cast<float*>(scores),
        n,
        k);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);

    kiln_w8a16_gemv_argmax_reduce_kernel<<<1, ARGMAX_BLOCK, 0, stream>>>(
        static_cast<const float*>(scores),
        static_cast<int64_t*>(out_idx),
        n);
    err = cudaGetLastError();
    if (err != cudaSuccess) return 2000 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_w8a16_gemv_gumbel_sample_bf16_async(
    const void* x,
    const void* w_q,
    const void* scales,
    const void* history_indices,
    const void* history_counts,
    void* scores,
    void* out_idx,
    int64_t n,
    int64_t k,
    int64_t history_len,
    float repetition_penalty,
    float presence_penalty,
    float frequency_penalty,
    float inv_temperature,
    uint64_t seed,
    void* stream_raw) {
    if (n < 0 || k < 0 || history_len < 0) return 1;
    if (n == 0 || k == 0) return 2;
    if (n > static_cast<int64_t>(2147483647)) return 3;
    if (!(inv_temperature > 0.0f) || inv_temperature != inv_temperature
        || inv_temperature > 3.4028234663852886e38f) return 4;
    if (!(repetition_penalty > 0.0f) || repetition_penalty != repetition_penalty
        || repetition_penalty > 3.4028234663852886e38f) return 5;
    if (history_len > 0 && (history_indices == nullptr || history_counts == nullptr)) return 6;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    kiln_w8a16_gemv_gumbel_scores_kernel<<<static_cast<unsigned int>(n), BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const uint8_t*>(w_q),
        static_cast<const float*>(scales),
        static_cast<const uint32_t*>(history_indices),
        static_cast<const uint32_t*>(history_counts),
        static_cast<float*>(scores),
        n,
        k,
        history_len,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        inv_temperature,
        seed);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);

    kiln_w8a16_gemv_argmax_reduce_kernel<<<1, ARGMAX_BLOCK, 0, stream>>>(
        static_cast<const float*>(scores),
        static_cast<int64_t*>(out_idx),
        n);
    err = cudaGetLastError();
    if (err != cudaSuccess) return 2000 + static_cast<int>(err);
    return 0;
}
