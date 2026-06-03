// Paged-decode metadata kernels (Phase R.9 prerequisite): compute the paged
// gather index and the per-batch tail mask ON-DEVICE from device-resident
// block_table / seqused_k buffers, so the ROCm paged-decode attention path no
// longer stages this metadata to the host (a D2H + H2D round-trip per attention
// layer that both costs sync latency and prevents HIP graph capture).
//
// Both kernels are pure element-wise index math: one thread per output element,
// no cross-lane reductions, so there is no wave32/wave64 hazard (unlike the
// reduction kernels that needed the shared-memory block-reduce fix in R.5).
//
// Compiled by both nvcc (CUDA) and hipcc (ROCm) — `<cuda_runtime.h>` and
// `cudaStream_t` hipify cleanly. The same compiled object services both
// backends, matching the rest of csrc/.

#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

namespace {

// out_idx[bi*seqlen_k + t] = block_table[bi*max_blocks_per_seq + t/page]*page
//                            + t%page
// (`page` == page_block_size). Mirrors the host loop in
// `rocm_sdpa.rs::paged_gather` exactly.
__global__ void kiln_paged_gather_index_kernel(const uint32_t* __restrict__ block_table,
                                               uint32_t* __restrict__ out_idx,
                                               int64_t b,
                                               int64_t seqlen_k,
                                               int64_t max_blocks_per_seq,
                                               int64_t page_block_size) {
    int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = b * seqlen_k;
    if (gid >= total) return;

    int64_t bi = gid / seqlen_k;
    int64_t t = gid % seqlen_k;
    int64_t blk = t / page_block_size;
    int64_t within = t % page_block_size;
    uint32_t phys_block = block_table[bi * max_blocks_per_seq + blk];
    out_idx[gid] = static_cast<uint32_t>(phys_block) * static_cast<uint32_t>(page_block_size)
                   + static_cast<uint32_t>(within);
}

// out_mask[(bi*h + hi)*sk + j] = (j >= seqused_k[bi]) ? 1 : 0   (u8)
// Flattened [b*h, 1, sk] (sq == 1 for decode). Mirrors the host tail-mask loop
// in `rocm_sdpa.rs::sdpa_forward_dyn_tail`.
__global__ void kiln_paged_tail_mask_kernel(const uint32_t* __restrict__ seqused_k,
                                            uint8_t* __restrict__ out_mask,
                                            int64_t b,
                                            int64_t h,
                                            int64_t sk) {
    int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = b * h * sk;
    if (gid >= total) return;

    int64_t bi = gid / (h * sk);
    int64_t rem = gid % (h * sk);
    int64_t j = rem % sk;
    uint32_t used = seqused_k[bi];
    out_mask[gid] = (static_cast<uint32_t>(j) >= used) ? static_cast<uint8_t>(1)
                                                       : static_cast<uint8_t>(0);
}

} // namespace

extern "C" int kiln_paged_gather_index_async(const void* block_table_u32,
                                             void* out_idx_u32,
                                             int64_t b,
                                             int64_t seqlen_k,
                                             int64_t max_blocks_per_seq,
                                             int64_t page_block_size,
                                             cudaStream_t stream) {
    if (b < 0 || seqlen_k < 0 || max_blocks_per_seq < 0 || page_block_size <= 0) return 1;
    int64_t total = b * seqlen_k;
    if (total == 0) return 0;
    int64_t blocks_i64 = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 2;
    int blocks = static_cast<int>(blocks_i64);

    kiln_paged_gather_index_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<const uint32_t*>(block_table_u32),
        static_cast<uint32_t*>(out_idx_u32),
        b, seqlen_k, max_blocks_per_seq, page_block_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 100 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_paged_tail_mask_async(const void* seqused_k_u32,
                                          void* out_mask_u8,
                                          int64_t b,
                                          int64_t h,
                                          int64_t sk,
                                          cudaStream_t stream) {
    if (b < 0 || h < 0 || sk < 0) return 1;
    int64_t total = b * h * sk;
    if (total == 0) return 0;
    int64_t blocks_i64 = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 2;
    int blocks = static_cast<int>(blocks_i64);

    kiln_paged_tail_mask_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<const uint32_t*>(seqused_k_u32),
        static_cast<uint8_t*>(out_mask_u8),
        b, h, sk);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 100 + static_cast<int>(err);
    return 0;
}
