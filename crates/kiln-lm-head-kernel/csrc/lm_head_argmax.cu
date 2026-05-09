#include "lm_head_argmax.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mutex>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBatch = 32;

__device__ __forceinline__ bool better_pair(float score, unsigned int index, float best_score, unsigned int best_index) {
    return score > best_score || (score == best_score && index < best_index);
}

__global__ void lm_head_argmax_reduce_kernel(
    const float *__restrict__ scores,
    unsigned int *__restrict__ tokens,
    int vocab
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float best_score = -FLT_MAX;
    unsigned int best_index = 0;

    for (int vocab_idx = tid; vocab_idx < vocab; vocab_idx += blockDim.x) {
        const float score = scores[row * vocab + vocab_idx];
        const unsigned int index = static_cast<unsigned int>(vocab_idx);
        if (better_pair(score, index, best_score, best_index)) {
            best_score = score;
            best_index = index;
        }
    }

    __shared__ float shared_scores[kThreads];
    __shared__ unsigned int shared_indices[kThreads];
    shared_scores[tid] = best_score;
    shared_indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float score = shared_scores[tid + stride];
            const unsigned int index = shared_indices[tid + stride];
            if (better_pair(score, index, shared_scores[tid], shared_indices[tid])) {
                shared_scores[tid] = score;
                shared_indices[tid] = index;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        tokens[row] = shared_indices[0];
    }
}


int get_cublas_handle(cublasHandle_t *out) {
    static std::mutex mutex;
    static cublasHandle_t handle = nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    if (handle == nullptr) {
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            return 10000 + static_cast<int>(st);
        }
    }
    *out = handle;
    return 0;
}

} // namespace

extern "C" int kiln_lm_head_argmax_bf16_batch(
    const void *x,
    const void *weight_t,
    float *scores,
    unsigned int *tokens,
    int batch,
    int hidden,
    int vocab,
    void *stream
) {
    if (x == nullptr || weight_t == nullptr || scores == nullptr || tokens == nullptr) {
        return 1;
    }
    if (batch <= 0 || batch > kMaxBatch || hidden <= 0 || vocab <= 0) {
        return 2;
    }

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cublasHandle_t handle = nullptr;
    int handle_status = get_cublas_handle(&handle);
    if (handle_status != 0) {
        return handle_status;
    }
    cublasStatus_t st = cublasSetStream(handle, cuda_stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        return 10100 + static_cast<int>(st);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Candle tensors are row-major. Interpreting them as column-major gives:
    //   weight_t row-major [hidden, vocab] == column-major [vocab, hidden]
    //   x        row-major [batch, hidden] == column-major [hidden, batch]
    //   scores   row-major [batch, vocab]  == column-major [vocab, batch]
    // Therefore C_col[vocab,batch] = W_col[vocab,hidden] * X_col[hidden,batch].
    st = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        vocab,
        batch,
        hidden,
        &alpha,
        weight_t,
        CUDA_R_16BF,
        vocab,
        x,
        CUDA_R_16BF,
        hidden,
        &beta,
        scores,
        CUDA_R_32F,
        vocab,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (st != CUBLAS_STATUS_SUCCESS) {
        return 10200 + static_cast<int>(st);
    }

    lm_head_argmax_reduce_kernel<<<batch, kThreads, 0, cuda_stream>>>(scores, tokens, vocab);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 20000 + static_cast<int>(err);
    }
    return 0;
}
