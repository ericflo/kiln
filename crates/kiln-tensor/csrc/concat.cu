// Kiln CUDA-side `concat` — join N tensors along a given axis.
//
// Bytewise per-input copy into the destination buffer at the right
// (outer, axis_offset, inner) location. The CPU reference in
// `crates/kiln-tensor/src/ops/concat.rs` performs the exact same
// outer-slab loop with `copy_from_slice`; this kernel issues the same
// copies on-device.
//
// # Algorithm
//
// For each input i:
//   - outer loop o over [0, outer):
//     - copy `t_axis[i] * inner * bpe` bytes from
//       `src[i] + o * t_axis[i] * inner * bpe`
//       to
//       `dst + (o * axis_total + axis_offset_so_far) * inner * bpe`
//
// We unfold this into a single kernel launch per call: one CUDA
// thread per output byte. Each thread computes which (o, a, i_inner)
// it's responsible for in the destination layout, looks up which
// input owns the axis position, and copies the matching source byte.
//
// One element per thread, byte-wise (matches `contiguous.cu`).
//
// # Limits
//
// - `MAX_INPUTS = 32`. Enough for QKV concat (3) and other typical
//   uses; larger N would be unusual.
// - `bytes_per_elem <= 8` — F32 (4), BF16/F16 (2), U8 (1), I64 (8).
//
// # Return code
//
// 0 on success, non-zero on validation or CUDA error.

#include <cuda_runtime.h>
#include <cstdint>

#define MAX_INPUTS 32
#define BLOCK_SIZE 256

namespace {

struct ConcatDesc {
    // Per-input axis lengths.
    int64_t t_axis[MAX_INPUTS];
    // Per-input cumulative axis offset (prefix sum of t_axis).
    int64_t axis_offset[MAX_INPUTS];
    // Per-input source base pointer.
    const uint8_t* src[MAX_INPUTS];
    int32_t n_inputs;
    int64_t outer;
    int64_t axis_total;
    int64_t inner_bytes; // inner * bpe
    int64_t n_dst_elements; // outer * axis_total
};

struct RowConcatDesc {
    // Per-input axis lengths and cumulative axis offsets.
    int64_t t_axis[MAX_INPUTS];
    int64_t axis_offset[MAX_INPUTS];
    const uint8_t* src[MAX_INPUTS];
    int32_t n_inputs;
    int64_t axis_total;
    int64_t inner_bytes;
};

__global__ void kiln_concat_kernel(uint8_t* __restrict__ dst,
                                   ConcatDesc desc) {
    // One thread per (outer, axis_total) position; the thread copies a
    // full `inner_bytes` chunk.
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= desc.n_dst_elements) return;

    int64_t o = idx / desc.axis_total;
    int64_t a = idx % desc.axis_total;

    // Find which input owns axis position `a`. Linear scan — N is small
    // (<= MAX_INPUTS) so this is fine; branchless via cumulative
    // axis_offset.
    int input_i = 0;
    for (int i = 1; i < desc.n_inputs; ++i) {
        if (a >= desc.axis_offset[i]) {
            input_i = i;
        } else {
            break;
        }
    }
    int64_t local_a = a - desc.axis_offset[input_i];

    const uint8_t* src_base = desc.src[input_i];
    int64_t src_off = (o * desc.t_axis[input_i] + local_a) * desc.inner_bytes;
    int64_t dst_off = (o * desc.axis_total + a) * desc.inner_bytes;

    // Byte-wise copy. `inner_bytes` can be large (e.g. hidden_dim *
    // bpe); the compiler will unroll for small bpe but for larger
    // inner blocks we do a serial byte copy. Could be improved with
    // wider loads later if profiling shows this in the hot path.
    for (int64_t b = 0; b < desc.inner_bytes; ++b) {
        dst[dst_off + b] = src_base[src_off + b];
    }
}

__device__ __forceinline__ int row_concat_input_for_row(const RowConcatDesc& desc,
                                                        int64_t row) {
    int input_i = 0;
    for (int i = 1; i < desc.n_inputs; ++i) {
        if (row >= desc.axis_offset[i]) {
            input_i = i;
        } else {
            break;
        }
    }
    return input_i;
}

__global__ void kiln_concat_axis0_vec16_kernel(uint4* __restrict__ dst,
                                               RowConcatDesc desc,
                                               int64_t inner_vecs,
                                               int64_t total_vecs) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_vecs) return;

    int64_t row = idx / inner_vecs;
    int64_t vec_in_row = idx - row * inner_vecs;
    int input_i = row_concat_input_for_row(desc, row);
    int64_t local_row = row - desc.axis_offset[input_i];

    const uint4* src = reinterpret_cast<const uint4*>(desc.src[input_i]);
    dst[row * inner_vecs + vec_in_row] = src[local_row * inner_vecs + vec_in_row];
}

__global__ void kiln_concat_axis0_byte_kernel(uint8_t* __restrict__ dst,
                                              RowConcatDesc desc,
                                              int64_t total_bytes) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_bytes) return;

    int64_t row = idx / desc.inner_bytes;
    int64_t byte_in_row = idx - row * desc.inner_bytes;
    int input_i = row_concat_input_for_row(desc, row);
    int64_t local_row = row - desc.axis_offset[input_i];

    const uint8_t* src = desc.src[input_i];
    dst[row * desc.inner_bytes + byte_in_row] =
        src[local_row * desc.inner_bytes + byte_in_row];
}

} // namespace

extern "C" int kiln_concat_async(void* dst,
                                 const void* const* src_ptrs,
                                 const int64_t* t_axis_lens,
                                 int32_t n_inputs,
                                 int64_t outer,
                                 int64_t inner_bytes,
                                 cudaStream_t stream) {
    if (n_inputs < 1 || n_inputs > MAX_INPUTS) return 1;
    if (outer < 0) return 2;
    if (inner_bytes < 0) return 3;
    if (outer == 0 || inner_bytes == 0) return 0;

    ConcatDesc desc;
    desc.n_inputs = n_inputs;
    desc.outer = outer;
    desc.inner_bytes = inner_bytes;

    int64_t axis_total = 0;
    for (int i = 0; i < n_inputs; ++i) {
        desc.t_axis[i] = t_axis_lens[i];
        desc.axis_offset[i] = axis_total;
        desc.src[i] = static_cast<const uint8_t*>(src_ptrs[i]);
        axis_total += t_axis_lens[i];
    }
    for (int i = n_inputs; i < MAX_INPUTS; ++i) {
        desc.t_axis[i] = 0;
        desc.axis_offset[i] = axis_total;
        desc.src[i] = nullptr;
    }
    desc.axis_total = axis_total;
    desc.n_dst_elements = outer * axis_total;

    if (desc.n_dst_elements == 0) return 0;

    int64_t blocks_i64 = (desc.n_dst_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 4;
    int blocks = static_cast<int>(blocks_i64);

    kiln_concat_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<uint8_t*>(dst),
        desc);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_concat_axis0_contiguous_async(void* dst,
                                                  const void* const* src_ptrs,
                                                  const int64_t* t_axis_lens,
                                                  int32_t n_inputs,
                                                  int64_t inner_bytes,
                                                  cudaStream_t stream) {
    if (n_inputs < 1 || n_inputs > MAX_INPUTS) return 1;
    if (inner_bytes < 0) return 2;
    if (inner_bytes == 0) return 0;

    RowConcatDesc desc;
    desc.n_inputs = n_inputs;
    desc.inner_bytes = inner_bytes;

    int64_t axis_total = 0;
    bool vec16_aligned = ((reinterpret_cast<uintptr_t>(dst) & 0x0f) == 0) &&
                         ((inner_bytes & 0x0f) == 0);
    for (int i = 0; i < n_inputs; ++i) {
        if (t_axis_lens[i] < 0) return 3;
        desc.t_axis[i] = t_axis_lens[i];
        desc.axis_offset[i] = axis_total;
        desc.src[i] = static_cast<const uint8_t*>(src_ptrs[i]);
        vec16_aligned =
            vec16_aligned &&
            ((reinterpret_cast<uintptr_t>(desc.src[i]) & 0x0f) == 0);
        axis_total += t_axis_lens[i];
    }
    for (int i = n_inputs; i < MAX_INPUTS; ++i) {
        desc.t_axis[i] = 0;
        desc.axis_offset[i] = axis_total;
        desc.src[i] = nullptr;
    }
    desc.axis_total = axis_total;

    if (axis_total == 0) return 0;

    int64_t work_items = axis_total * inner_bytes;
    if (vec16_aligned) {
        work_items /= 16;
    }
    int64_t blocks_i64 = (work_items + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 4;
    int blocks = static_cast<int>(blocks_i64);

    if (vec16_aligned) {
        int64_t inner_vecs = inner_bytes / 16;
        kiln_concat_axis0_vec16_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<uint4*>(dst),
            desc,
            inner_vecs,
            work_items);
    } else {
        kiln_concat_axis0_byte_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<uint8_t*>(dst),
            desc,
            work_items);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
