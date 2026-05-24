// Kiln CUDA-side index_select(src, axis=0, indices) — gather along the
// outer axis of a CUDA tensor, producing a freshly-allocated output.
//
// Unblocks the non-contiguous-slot-run path of
// `PagedKvCacheKt::read` (#1082 line 110 / 167 / 324) and the
// generic gather pattern used by RoPE table lookups, embedding
// lookups, etc.
//
// # Algorithm
//
// One CUDA block per output row, threads within a block cooperatively
// copy the row bytes. For `src` of shape `[N, ...inner]` and U32
// `indices` of shape `[K]`, output is `[K, ...inner]` with
// `out[i] = src[indices[i]]`.
//
//   row_bytes = product(inner_dims) * bytes_per_elem
//   grid      = K blocks
//   block     = 256 threads
//   per thread loop: copy `b in [tid, row_bytes) step blockDim.x`
//
// `axis=0` only — kt-tensor's index_select is exposed as a free
// function with this signature; higher-axis index_select can be
// expressed via narrow/permute composition.
//
// # Bounds
//
// Out-of-range indices skip the copy (output row stays zero from
// `cudaMalloc + memset`). The caller is responsible for valid index
// values; the kernel does NOT abort the launch on a bad index.

#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

namespace {

__global__ void kiln_index_select_dim0_kernel(const uint8_t* __restrict__ src,
                                              uint8_t* __restrict__ dst,
                                              const uint32_t* __restrict__ indices,
                                              int64_t row_bytes,
                                              int64_t n_indices,
                                              int64_t src_n_rows) {
    int64_t out_row = static_cast<int64_t>(blockIdx.x);
    if (out_row >= n_indices) return;
    uint32_t src_row = indices[out_row];
    if (static_cast<int64_t>(src_row) >= src_n_rows) return;

    int64_t src_off = static_cast<int64_t>(src_row) * row_bytes;
    int64_t dst_off = out_row * row_bytes;

    int64_t tid = threadIdx.x;
    int64_t stride = blockDim.x;
    for (int64_t b = tid; b < row_bytes; b += stride) {
        dst[dst_off + b] = src[src_off + b];
    }
}

} // namespace

extern "C" int kiln_index_select_dim0_async(const void* src,
                                            void* dst,
                                            const void* indices_u32,
                                            int64_t row_bytes,
                                            int64_t n_indices,
                                            int64_t src_n_rows,
                                            cudaStream_t stream) {
    if (row_bytes < 0 || n_indices < 0 || src_n_rows < 0) return 1;
    if (n_indices == 0 || row_bytes == 0) return 0;

    int64_t blocks_i64 = n_indices;
    if (blocks_i64 > (int64_t)2147483647) return 2;
    int blocks = static_cast<int>(blocks_i64);

    kiln_index_select_dim0_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<const uint8_t*>(src),
        static_cast<uint8_t*>(dst),
        static_cast<const uint32_t*>(indices_u32),
        row_bytes,
        n_indices,
        src_n_rows);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
