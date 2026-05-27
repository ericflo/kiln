// Kiln CUDA-side index_select(src, axis, indices) — gather along an
// arbitrary axis of a CUDA tensor, producing a freshly-allocated
// output.
//
// Unblocks the non-contiguous-slot-run path of
// `PagedKvCacheKt::read` (#1082 line 110 / 167 / 324), the OPD per-
// position kt forward (`kt_forward_op.rs`, which gathers K columns
// along axis 1 of `head_t`), and the generic gather pattern used by
// RoPE table lookups, embedding lookups, etc.
//
// # Algorithm — axis-N gather
//
// Given:
//   - `src` of shape `[d0, ..., d_{axis-1}, src_dim, d_{axis+1}, ..., d_{n-1}]`
//   - U32 `indices` of shape `[k0, ..., k_{m-1}]`
//   - output of shape `[d0, ..., d_{axis-1}, k0, ..., k_{m-1}, d_{axis+1}, ..., d_{n-1}]`
//
// Define:
//   - `left_size  = d0 * d1 * ... * d_{axis-1}`           (outer dims)
//   - `right_size = d_{axis+1} * ... * d_{n-1}`            (inner dims)
//   - `src_dim    = d_{axis}`                              (axis extent in src)
//   - `ids_dim    = product of indices.shape`              (axis extent in out)
//   - `row_bytes  = right_size * bytes_per_elem`
//
// The output has `left_size * ids_dim * row_bytes` bytes. One CUDA
// block per `(left, ids)` pair, threads within a block cooperatively
// copy the `row_bytes` bytes of the right-side slice.
//
//   grid  = dim3(ids_dim, left_size)
//   block = 256 threads
//   per thread loop: copy `b in [tid, row_bytes) step blockDim.x`
//
// The `axis == 0` case reduces to the original behavior:
// `left_size = 1`, grid degenerates to `(ids_dim, 1)` and the row
// copy is identical to the legacy dim0 fast path.
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

// Axis-0 fast path (preserved). One block per `out_row`. `row_bytes` is
// the full inner-slice byte count (i.e. `right_size * bpe` in the
// generalized terminology).
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

// Generic axis-N gather. One block per `(left_i, id_i)` pair, threads
// cooperatively copy the `right_bytes` bytes of one right-side slice.
//
// Output element at `(left_i, id_i, right_byte)` comes from
// `src[(left_i * src_dim + indices[id_i]) * right_bytes + right_byte]`.
__global__ void kiln_index_select_axis_n_kernel(const uint8_t* __restrict__ src,
                                                uint8_t* __restrict__ dst,
                                                const uint32_t* __restrict__ indices,
                                                int64_t right_bytes,
                                                int64_t ids_dim,
                                                int64_t src_dim,
                                                int64_t left_size) {
    // Grid layout: blockIdx.x = id_i (along ids_dim), blockIdx.y = left_i.
    int64_t id_i = static_cast<int64_t>(blockIdx.x);
    int64_t left_i = static_cast<int64_t>(blockIdx.y);
    if (id_i >= ids_dim || left_i >= left_size) return;

    uint32_t id_val = indices[id_i];
    if (static_cast<int64_t>(id_val) >= src_dim) return;

    int64_t src_off = (left_i * src_dim + static_cast<int64_t>(id_val)) * right_bytes;
    int64_t dst_off = (left_i * ids_dim + id_i) * right_bytes;

    int64_t tid = threadIdx.x;
    int64_t stride = blockDim.x;
    for (int64_t b = tid; b < right_bytes; b += stride) {
        dst[dst_off + b] = src[src_off + b];
    }
}

} // namespace

// Legacy entry preserved for callers that go through the dim0 fast
// path directly (`crate::cuda_index_select_dim0`). Callers that need
// arbitrary-axis gather should use `kiln_index_select_axis_n_async`
// below.
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

// Generic axis-N gather. Parameters:
//   - `src` / `dst` / `indices_u32`: device pointers (already byte-
//     offset-adjusted by the Rust caller).
//   - `right_bytes`: `right_size * bytes_per_elem` (bytes per inner
//     slice). Must be >= 0; 0 short-circuits to success.
//   - `ids_dim`:    flattened index count (product of indices.shape).
//   - `src_dim`:    extent of `src` along the gather axis.
//   - `left_size`:  product of `src.shape[..axis]`.
//   - `stream`:     CUDA stream to launch on.
//
// Returns 0 on success, 1 on negative dimension, 2 on grid overflow
// (`ids_dim` or `left_size` > INT32_MAX), or `1000 + cudaError` if
// the kernel launch fails.
extern "C" int kiln_index_select_axis_n_async(const void* src,
                                              void* dst,
                                              const void* indices_u32,
                                              int64_t right_bytes,
                                              int64_t ids_dim,
                                              int64_t src_dim,
                                              int64_t left_size,
                                              cudaStream_t stream) {
    if (right_bytes < 0 || ids_dim < 0 || src_dim < 0 || left_size < 0) return 1;
    if (ids_dim == 0 || left_size == 0 || right_bytes == 0) return 0;

    // Grid: (ids_dim, left_size, 1). Each block copies one right-side
    // slice of `right_bytes` bytes.
    if (ids_dim > (int64_t)2147483647) return 2;
    if (left_size > (int64_t)2147483647) return 2;

    dim3 grid(static_cast<unsigned int>(ids_dim),
              static_cast<unsigned int>(left_size),
              1);
    dim3 block(BLOCK_SIZE, 1, 1);

    kiln_index_select_axis_n_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(src),
        static_cast<uint8_t*>(dst),
        static_cast<const uint32_t*>(indices_u32),
        right_bytes,
        ids_dim,
        src_dim,
        left_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
