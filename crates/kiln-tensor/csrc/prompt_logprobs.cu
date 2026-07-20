// Compact prompt-logprob selection for CUDA and ROCm.
//
// One block owns one contiguous vocabulary row. It validates every input
// logit, computes the same F32 stable-normalization recipe as softmax.cu,
// validates every derived log-probability, computes the observed token's full
// rank, and returns only the observed row statistics plus K ranked candidates.
// Device-to-host transfer and host work are therefore O(TK), not O(TV).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "kt_gpu_compat.cuh"

namespace {

constexpr int MAX_THREADS = 1024;
constexpr int64_t MAX_GRID_X = 2147483647LL;

template <typename T>
__device__ inline float to_f32(T v);
template <>
__device__ inline float to_f32<float>(float v) { return v; }
template <>
__device__ inline float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <>
__device__ inline float to_f32<__half>(__half v) { return __half2float(v); }

__device__ inline bool rank_better(float candidate_value,
                                   int64_t candidate_index,
                                   float incumbent_value,
                                   int64_t incumbent_index) {
    return candidate_value > incumbent_value
        || (candidate_value == incumbent_value && candidate_index < incumbent_index);
}

__device__ inline bool below_frontier(float value,
                                      int64_t index,
                                      float frontier_value,
                                      int64_t frontier_index) {
    return value < frontier_value
        || (value == frontier_value && index > frontier_index);
}

__device__ inline int64_t block_reduce_min_i64(int64_t value, int64_t* smem) {
    int tid = threadIdx.x;
    int blk = blockDim.x;
    smem[tid] = value;
    __syncthreads();
    for (int stride = blk >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && smem[tid + stride] < smem[tid]) {
            smem[tid] = smem[tid + stride];
        }
        __syncthreads();
    }
    int64_t result = smem[0];
    __syncthreads();
    return result;
}

__device__ inline void block_reduce_rank(float& value,
                                         int64_t& index,
                                         float* value_smem,
                                         int64_t* index_smem) {
    int tid = threadIdx.x;
    int blk = blockDim.x;
    value_smem[tid] = value;
    index_smem[tid] = index;
    __syncthreads();
    for (int stride = blk >> 1; stride > 0; stride >>= 1) {
        if (tid < stride
            && rank_better(value_smem[tid + stride],
                           index_smem[tid + stride],
                           value_smem[tid],
                           index_smem[tid])) {
            value_smem[tid] = value_smem[tid + stride];
            index_smem[tid] = index_smem[tid + stride];
        }
        __syncthreads();
    }
    value = value_smem[0];
    index = index_smem[0];
    __syncthreads();
}

template <typename T>
__global__ void prompt_logprobs_kernel(
    const T* __restrict__ x,
    const int64_t* __restrict__ observed_ids,
    float* __restrict__ out_row_max,
    float* __restrict__ out_log_sum,
    float* __restrict__ out_observed_logit,
    int64_t* __restrict__ out_observed_rank,
    float* __restrict__ out_top_logits,
    int64_t* __restrict__ out_top_indices,
    uint32_t* __restrict__ out_invalid_kind,
    int64_t* __restrict__ out_invalid_column,
    float* __restrict__ out_invalid_value,
    int64_t n_cols,
    int top_k) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    int tid = threadIdx.x;
    int blk = blockDim.x;
    const T* row_in = x + row * n_cols;

    __shared__ float float_smem[MAX_THREADS];
    __shared__ int64_t i64_smem[MAX_THREADS];
    __shared__ float frontier_value;
    __shared__ int64_t frontier_index;

    int64_t local_invalid = n_cols;
    for (int64_t col = tid; col < n_cols; col += blk) {
        if (!isfinite(to_f32<T>(row_in[col]))) {
            local_invalid = col;
            break;
        }
    }
    int64_t invalid_column = block_reduce_min_i64(local_invalid, i64_smem);
    if (invalid_column < n_cols) {
        if (tid == 0) {
            out_invalid_kind[row] = 1;
            out_invalid_column[row] = invalid_column;
            out_invalid_value[row] = to_f32<T>(row_in[invalid_column]);
            out_row_max[row] = NAN;
            out_log_sum[row] = NAN;
            out_observed_logit[row] = NAN;
            out_observed_rank[row] = 0;
        }
        for (int rank = tid; rank < top_k; rank += blk) {
            int64_t flat = row * static_cast<int64_t>(top_k) + rank;
            out_top_logits[flat] = NAN;
            out_top_indices[flat] = -1;
        }
        return;
    }

    float local_max = -INFINITY;
    for (int64_t col = tid; col < n_cols; col += blk) {
        float value = to_f32<T>(row_in[col]);
        if (value > local_max) local_max = value;
    }
    float row_max = kiln_block_reduce_max(local_max, float_smem);

    float local_sum = 0.0f;
    for (int64_t col = tid; col < n_cols; col += blk) {
        local_sum += expf(to_f32<T>(row_in[col]) - row_max);
    }
    float row_sum = kiln_block_reduce_sum(local_sum, float_smem);
    float log_sum = logf(row_sum);

    local_invalid = n_cols;
    float local_invalid_value = 0.0f;
    for (int64_t col = tid; col < n_cols; col += blk) {
        float logprob = (to_f32<T>(row_in[col]) - row_max) - log_sum;
        if (!isfinite(logprob)) {
            local_invalid = col;
            local_invalid_value = logprob;
            break;
        }
    }
    invalid_column = block_reduce_min_i64(local_invalid, i64_smem);
    if (invalid_column < n_cols) {
        if (local_invalid != invalid_column) local_invalid_value = 0.0f;
        float invalid_value = local_invalid == invalid_column ? local_invalid_value : 0.0f;
        // Exactly one thread owns the minimum column because columns are
        // partitioned by the strided scan. Publish its derived value.
        float published_value = kiln_block_reduce_sum(invalid_value, float_smem);
        if (tid == 0) {
            out_invalid_kind[row] = 2;
            out_invalid_column[row] = invalid_column;
            out_invalid_value[row] = published_value;
            out_row_max[row] = row_max;
            out_log_sum[row] = log_sum;
            out_observed_logit[row] = NAN;
            out_observed_rank[row] = 0;
        }
        for (int rank = tid; rank < top_k; rank += blk) {
            int64_t flat = row * static_cast<int64_t>(top_k) + rank;
            out_top_logits[flat] = NAN;
            out_top_indices[flat] = -1;
        }
        return;
    }

    int64_t observed_id = observed_ids[row];
    float observed_logit = to_f32<T>(row_in[observed_id]);
    int64_t local_rank = 0;
    for (int64_t col = tid; col < n_cols; col += blk) {
        local_rank += static_cast<int64_t>(to_f32<T>(row_in[col]) >= observed_logit);
    }
    int64_t observed_rank = kiln_block_reduce_sum(local_rank, i64_smem);

    if (tid == 0) {
        out_invalid_kind[row] = 0;
        out_invalid_column[row] = -1;
        out_invalid_value[row] = 0.0f;
        out_row_max[row] = row_max;
        out_log_sum[row] = log_sum;
        out_observed_logit[row] = observed_logit;
        out_observed_rank[row] = observed_rank;
        frontier_value = INFINITY;
        frontier_index = -1;
    }
    __syncthreads();

    for (int rank = 0; rank < top_k; ++rank) {
        float local_value = -INFINITY;
        int64_t local_index = n_cols;
        float current_frontier_value = frontier_value;
        int64_t current_frontier_index = frontier_index;
        for (int64_t col = tid; col < n_cols; col += blk) {
            float value = to_f32<T>(row_in[col]);
            if (below_frontier(value, col, current_frontier_value, current_frontier_index)
                && rank_better(value, col, local_value, local_index)) {
                local_value = value;
                local_index = col;
            }
        }
        block_reduce_rank(local_value, local_index, float_smem, i64_smem);
        if (tid == 0) {
            int64_t flat = row * static_cast<int64_t>(top_k) + rank;
            out_top_logits[flat] = local_value;
            out_top_indices[flat] = local_index;
            frontier_value = local_value;
            frontier_index = local_index;
        }
        __syncthreads();
    }
}

}  // namespace

extern "C" int kiln_prompt_logprobs_async(
    const void* x,
    const int64_t* observed_ids,
    void* out_row_max,
    void* out_log_sum,
    void* out_observed_logit,
    void* out_observed_rank,
    void* out_top_logits,
    void* out_top_indices,
    void* out_invalid_kind,
    void* out_invalid_column,
    void* out_invalid_value,
    int64_t n_rows,
    int64_t n_cols,
    int32_t top_k,
    int32_t dtype_tag,
    void* stream_raw) {
    if (n_rows < 0 || n_cols <= 0 || n_rows > MAX_GRID_X
        || top_k < 0 || static_cast<int64_t>(top_k) > n_cols) return -3;
    if (n_rows == 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

    int threads = 64;
    while (threads < n_cols && threads < MAX_THREADS) threads *= 2;
    dim3 grid(static_cast<unsigned int>(n_rows));
    dim3 block(threads);

#define LAUNCH_PROMPT_LOGPROBS(T) \
    prompt_logprobs_kernel<T><<<grid, block, 0, stream>>>( \
        reinterpret_cast<const T*>(x), observed_ids, \
        reinterpret_cast<float*>(out_row_max), \
        reinterpret_cast<float*>(out_log_sum), \
        reinterpret_cast<float*>(out_observed_logit), \
        reinterpret_cast<int64_t*>(out_observed_rank), \
        reinterpret_cast<float*>(out_top_logits), \
        reinterpret_cast<int64_t*>(out_top_indices), \
        reinterpret_cast<uint32_t*>(out_invalid_kind), \
        reinterpret_cast<int64_t*>(out_invalid_column), \
        reinterpret_cast<float*>(out_invalid_value), \
        n_cols, static_cast<int>(top_k))

    switch (dtype_tag) {
        case 0: LAUNCH_PROMPT_LOGPROBS(float); break;
        case 1: LAUNCH_PROMPT_LOGPROBS(__nv_bfloat16); break;
        case 2: LAUNCH_PROMPT_LOGPROBS(__half); break;
        default: return -2;
    }
#undef LAUNCH_PROMPT_LOGPROBS

    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : -1;
}
