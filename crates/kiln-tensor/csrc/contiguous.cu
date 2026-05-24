// Kiln CUDA-side `Tensor::contiguous()` — stride-aware copy of a non-
// contiguous CUDA storage into a freshly-allocated contiguous output.
//
// Unblocks the rest of PagedKvCacheKt + the Phase 4 layer-glue rebound
// (RMSNorm/embedding/residual/LM head) by replacing the CPU-only
// `Tensor::contiguous()` impl in `kiln-tensor::tensor.rs:358` for CUDA
// storage. See #1082 Phase 1 "Pinned-host staging pool per device" and
// the surrounding substrate items.
//
// # Algorithm
//
// One CUDA thread per logical output element. Each thread:
//   1. Computes its flat output index `idx` from `blockIdx * blockDim
//      + threadIdx`.
//   2. Unflattens `idx` to a multi-dim coordinate `(c0, c1, ..., c_{R-1})`
//      via repeated modulo against the shape (highest dim varies
//      fastest — matches the row-major iteration order kt-tensor's
//      `Layout::contiguous` produces).
//   3. Computes the physical source byte offset:
//        `src_byte_off = sum(c_d * strides_e[d]) * bytes_per_elem`
//      where `strides_e` is the source's element-stride array.
//   4. Computes the physical destination byte offset:
//        `dst_byte_off = idx * bytes_per_elem`
//   5. Copies `bytes_per_elem` bytes from `src + src_byte_off` to
//      `dst + dst_byte_off`. Byte-wise so the kernel is dtype-agnostic
//      (BF16 / F32 / U8 / etc. all use the same path).
//
// # Limits
//
// - `MAX_RANK = 8`. Matches kt-tensor's documented max rank.
// - `bytes_per_elem ≤ 8`. Covers F32 (4), BF16/F16 (2), U8 (1), I64 (8).
//
// # Launch
//
// `grid = ceil_div(n_elements, BLOCK_SIZE)`, `block = BLOCK_SIZE`.
// `BLOCK_SIZE = 256` — standard, good occupancy across SM 80/86/89/90.
//
// # Return code
//
// `kiln_contiguous_copy_async` returns 0 on success, non-zero on any
// CUDA error (last-error captured before launch).

#include <cuda_runtime.h>
#include <cstdint>

#define MAX_RANK 8
#define BLOCK_SIZE 256

namespace {

// Per-call descriptor passed by value to the kernel. Up to 8D, so
// (8 + 8) * 8 + 16 = 144 bytes — well within the CUDA per-kernel
// parameter limit (4 KB).
struct ContiguousDesc {
    int64_t shape[MAX_RANK];
    int64_t strides_e[MAX_RANK]; // element strides
    int32_t rank;
    int32_t bytes_per_elem;
    int64_t n_elements;
};

__global__ void kiln_contiguous_copy_kernel(const uint8_t* __restrict__ src,
                                            uint8_t* __restrict__ dst,
                                            ContiguousDesc desc) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= desc.n_elements) return;

    // Unflatten idx to (c0, c1, ..., c_{rank-1}) and accumulate src
    // element offset using the source's element-strides.
    int64_t src_off_e = 0;
    int64_t remaining = idx;
    // Iterate from the last (fastest-varying) dim to the first.
    for (int d = desc.rank - 1; d >= 0; --d) {
        int64_t sz = desc.shape[d];
        int64_t pos = remaining % sz;
        remaining /= sz;
        src_off_e += pos * desc.strides_e[d];
    }

    int64_t bpe = desc.bytes_per_elem;
    int64_t src_byte = src_off_e * bpe;
    int64_t dst_byte = idx * bpe;

    // Byte-wise copy. The compiler unrolls for the common BPE values
    // (1/2/4/8) since `bpe` is uniform across the warp.
    for (int b = 0; b < bpe; ++b) {
        dst[dst_byte + b] = src[src_byte + b];
    }
}

} // namespace

extern "C" int kiln_contiguous_copy_async(const void* src,
                                          void* dst,
                                          const int64_t* shape,
                                          const int64_t* strides_e,
                                          int32_t rank,
                                          int32_t bytes_per_elem,
                                          int64_t n_elements,
                                          cudaStream_t stream) {
    if (rank < 0 || rank > MAX_RANK) return 1;
    if (bytes_per_elem < 1 || bytes_per_elem > 8) return 2;
    if (n_elements < 0) return 3;
    if (n_elements == 0) return 0;

    ContiguousDesc desc;
    desc.rank = rank;
    desc.bytes_per_elem = bytes_per_elem;
    desc.n_elements = n_elements;
    for (int d = 0; d < MAX_RANK; ++d) {
        desc.shape[d] = (d < rank) ? shape[d] : 1;
        desc.strides_e[d] = (d < rank) ? strides_e[d] : 0;
    }

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 4; // CUDA grid x limit
    int blocks = static_cast<int>(blocks_i64);

    kiln_contiguous_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<const uint8_t*>(src),
        static_cast<uint8_t*>(dst),
        desc);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
