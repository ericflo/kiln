// Kiln CUDA-side scatter_add(out, indices, updates) — inverse of
// index_select(src, axis=0, indices). Used by embedding-backward and
// generic gather-grad accumulation patterns under #1082.
//
// # Algorithm
//
// For `updates` of shape `[n_indices, ...inner]` and `out` of shape
// `[target_dim, ...inner]` (with `out` pre-zeroed), the op computes:
//
//     out[indices[i], j] += updates[i, j]    for i in [0, n_indices), j in inner
//
// Implemented as a flat 1-thread-per-(idx, col) launch with
// `atomicAdd` on each output cell. Two index positions colliding on
// the same target row both contribute; the addition order across
// threads is non-deterministic, which is the documented "atomic-bwd"
// tolerance band on the Rust side (`ScatterAddOp::determinism()`).
//
// dtype dispatch: F32 (native atomicAdd) and BF16 (atomicAdd<__nv_bfloat16>
// available on SM≥80 / Ampere+ — our RunPod target is SM_86).
//
//   row_inner   = product(inner_dims)  (= cols per index row)
//   total_pairs = n_indices * row_inner
//   grid        = ceil(total_pairs / BLOCK_SIZE)
//   block       = 256 threads
//
// `axis=0` only. Higher-axis scatter_add can be expressed at the
// Rust dispatch layer via permute → scatter → permute_back, or by
// falling through to the CPU reference.
//
// # Bounds
//
// Out-of-range indices skip their contribution (no scatter). Caller
// is responsible for valid index values; the kernel does NOT abort
// the launch on a bad index.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define BLOCK_SIZE 256

namespace {

// ---------------------------------------------------------------------------
// BF16 atomic-add (ROCm port, R.5b).
//
// HIP exposes a native `atomicAdd(float*)` but NO native
// `atomicAdd(__hip_bfloat16*)` (ROCm 7.x). The CUDA kernel's `__CUDA_ARCH__ >=
// 800` guard is FALSE under hipcc (hipcc never defines `__CUDA_ARCH__`), so the
// stock `#else` branch is a non-atomic read-modify-write — silently WRONG under
// the index collisions that are the norm for embedding-backward.
//
// This helper restores atomic correctness on AMD by CAS-looping on the aligned
// 32-bit word that contains the target bf16 (bf16 is 2 bytes, so it occupies
// either the low or high half of the enclosing dword). The accumulation is done
// in F32 (matching the kt "F32 accumulation always" convention), then narrowed
// back to bf16 on store. `__bfloat16_as_ushort` / `__ushort_as_bfloat16` are the
// portable bit-reinterpret intrinsics (same names on CUDA and HIP).
//
// Only compiled on the HIP device pass; the nvcc build keeps its native
// `atomicAdd(__nv_bfloat16*)` path below byte-for-byte unchanged.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
__device__ __forceinline__ void kiln_atomic_add_bf16(__nv_bfloat16* addr, float val) {
    uintptr_t a = reinterpret_cast<uintptr_t>(addr);
    uint32_t* base = reinterpret_cast<uint32_t*>(a & ~uintptr_t(0x3));
    bool is_high = (a & 0x2) != 0;  // bf16 lives in the high 16 bits of the dword
    uint32_t old = *base;
    uint32_t assumed;
    do {
        assumed = old;
        unsigned short cur_bits = is_high ? static_cast<unsigned short>(assumed >> 16)
                                          : static_cast<unsigned short>(assumed & 0xFFFFu);
        float sum = __bfloat162float(__ushort_as_bfloat16(cur_bits)) + val;
        unsigned short res_bits = __bfloat16_as_ushort(__float2bfloat16(sum));
        uint32_t newval = is_high
            ? ((assumed & 0x0000FFFFu) | (static_cast<uint32_t>(res_bits) << 16))
            : ((assumed & 0xFFFF0000u) | static_cast<uint32_t>(res_bits));
        old = atomicCAS(base, assumed, newval);
    } while (assumed != old);
}
#endif

__global__ void kiln_scatter_add_dim0_kernel_f32(const float* __restrict__ updates,
                                                 float* __restrict__ out,
                                                 const uint32_t* __restrict__ indices,
                                                 int64_t n_indices,
                                                 int64_t row_inner,
                                                 int64_t target_dim) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n_indices * row_inner;
    if (tid >= total) return;

    int64_t idx_pos = tid / row_inner;
    int64_t col = tid - idx_pos * row_inner;

    uint32_t target = indices[idx_pos];
    if (static_cast<int64_t>(target) >= target_dim) return;

    int64_t src_off = idx_pos * row_inner + col;
    int64_t dst_off = static_cast<int64_t>(target) * row_inner + col;

    atomicAdd(&out[dst_off], updates[src_off]);
}

__global__ void kiln_scatter_add_dim0_kernel_bf16(const __nv_bfloat16* __restrict__ updates,
                                                  __nv_bfloat16* __restrict__ out,
                                                  const uint32_t* __restrict__ indices,
                                                  int64_t n_indices,
                                                  int64_t row_inner,
                                                  int64_t target_dim) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n_indices * row_inner;
    if (tid >= total) return;

    int64_t idx_pos = tid / row_inner;
    int64_t col = tid - idx_pos * row_inner;

    uint32_t target = indices[idx_pos];
    if (static_cast<int64_t>(target) >= target_dim) return;

    int64_t src_off = idx_pos * row_inner + col;
    int64_t dst_off = static_cast<int64_t>(target) * row_inner + col;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    // ROCm (R.5b): HIP has no native bf16 atomicAdd, so route through the
    // CAS-on-dword helper above — atomic-correct under index collisions, F32
    // accumulation, matches the CPU/CUDA scatter-add tolerance band.
    kiln_atomic_add_bf16(&out[dst_off], __bfloat162float(updates[src_off]));
#elif __CUDA_ARCH__ >= 800
    atomicAdd(&out[dst_off], updates[src_off]);
#else
    // Fallback for pre-Ampere: read-modify-write via F32. Not
    // atomic-correct under contention, but our build only targets
    // SM≥80 so this branch is unreachable in practice. Kept compileable
    // for the rare developer build that includes SM<80.
    float v = __bfloat162float(out[dst_off]) + __bfloat162float(updates[src_off]);
    out[dst_off] = __float2bfloat16(v);
#endif
}

} // namespace

// dtype_tag: 0 = F32, 1 = BF16.
extern "C" int kiln_scatter_add_dim0_async(const void* updates,
                                           void* out,
                                           const void* indices_u32,
                                           int64_t n_indices,
                                           int64_t row_inner,
                                           int64_t target_dim,
                                           int32_t dtype_tag,
                                           cudaStream_t stream) {
    if (n_indices < 0 || row_inner < 0 || target_dim < 0) return 1;
    if (n_indices == 0 || row_inner == 0 || target_dim == 0) return 0;

    int64_t total = n_indices * row_inner;
    int64_t blocks_i64 = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 2;
    int blocks = static_cast<int>(blocks_i64);

    switch (dtype_tag) {
        case 0:
            kiln_scatter_add_dim0_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(updates),
                static_cast<float*>(out),
                static_cast<const uint32_t*>(indices_u32),
                n_indices,
                row_inner,
                target_dim);
            break;
        case 1:
            kiln_scatter_add_dim0_kernel_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(updates),
                static_cast<__nv_bfloat16*>(out),
                static_cast<const uint32_t*>(indices_u32),
                n_indices,
                row_inner,
                target_dim);
            break;
        default:
            return 3;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
