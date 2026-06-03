// Kiln index_copy_dim0(dst, indices, src) — scatter-COPY along axis 0: write row
// `i` of `src` into row `indices[i]` of `dst`, IN PLACE. The inverse of
// `index_select_dim0` (gather).
//
//   dst[indices[i] * row_bytes + b] = src[i * row_bytes + b]
//
// Overwrite semantics (NOT add): the caller guarantees each index names a
// distinct `dst` row (the paged-KV slot write has exactly one index), so no
// atomics are needed.
//
// # Why this exists (R.9 HIP-graph decode capture)
//
// The on-device paged-KV slot write needs to scatter the current token's K/V
// row into `pool[*slot]` where `slot` is a DEVICE u32 buffer refreshed per
// graph replay. `hipMemcpyDtoDAsync` can only target a host-computed pointer,
// so the previous ROCm slot write read `slot` back to the host — which forces a
// sync and is not recordable into a HIP graph. This kernel reads the slot index
// on-device, so the K/V write records cleanly into the captured graph.
//
// Compiled by hipcc (ROCm) and nvcc (CUDA) from the same source; the CUDA
// headers map to HIP under hipcc. One block per `src` row; threads in a block
// cooperatively copy the `row_bytes` bytes. Out-of-range indices skip the write.

#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

namespace {

__global__ void kiln_index_copy_dim0_kernel(const uint8_t* __restrict__ src,
                                            uint8_t* __restrict__ dst,
                                            const uint32_t* __restrict__ indices,
                                            int64_t row_bytes,
                                            int64_t n_indices,
                                            int64_t dst_n_rows) {
    int64_t in_row = static_cast<int64_t>(blockIdx.x);
    if (in_row >= n_indices) return;
    uint32_t dst_row = indices[in_row];
    if (static_cast<int64_t>(dst_row) >= dst_n_rows) return;

    int64_t src_off = in_row * row_bytes;
    int64_t dst_off = static_cast<int64_t>(dst_row) * row_bytes;

    int64_t tid = threadIdx.x;
    int64_t stride = blockDim.x;
    for (int64_t b = tid; b < row_bytes; b += stride) {
        dst[dst_off + b] = src[src_off + b];
    }
}

} // namespace

// Parameters:
//   - `src` / `dst` / `indices_u32`: device pointers (byte-offset-adjusted by
//     the Rust caller for any start_offset).
//   - `row_bytes`:  inner-slice byte count (`row_inner * bytes_per_elem`).
//   - `n_indices`:  number of `src` rows to scatter (== indices length).
//   - `dst_n_rows`: extent of `dst` along axis 0 (for the bounds check).
//   - `stream`:     stream to launch on (the capture stream during HIP-graph
//                   capture, so the write records into the graph).
//
// Returns 0 on success, 1 on negative dimension, 2 on grid overflow
// (`n_indices` > INT32_MAX), or `1000 + cudaError` if the launch fails.
extern "C" int kiln_index_copy_dim0_async(const void* src,
                                          void* dst,
                                          const void* indices_u32,
                                          int64_t row_bytes,
                                          int64_t n_indices,
                                          int64_t dst_n_rows,
                                          cudaStream_t stream) {
    if (row_bytes < 0 || n_indices < 0 || dst_n_rows < 0) return 1;
    if (n_indices == 0 || row_bytes == 0) return 0;
    if (n_indices > (int64_t)2147483647) return 2;

    int blocks = static_cast<int>(n_indices);
    kiln_index_copy_dim0_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<const uint8_t*>(src),
        static_cast<uint8_t*>(dst),
        static_cast<const uint32_t*>(indices_u32),
        row_bytes,
        n_indices,
        dst_n_rows);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
