// kt_gpu_compat.cuh — shared GPU portability shim, injected via `-include` on
// BOTH the nvcc (CUDA) and hipcc (ROCm) builds.
//
// Its job is the wave-size abstraction that is the #1 correctness hazard of the
// ROCm port: NVIDIA warps are 32 lanes; AMD CDNA wavefronts are 64. Kiln's
// reductions historically hardcode 32 (`__shfl_xor_sync(0xFFFFFFFF, v, 16)`,
// `tid & 31`, `tid / 32`, `(blk + 31) / 32`), which silently corrupts on wave64.
//
// Phase R.5 migrates each reduction kernel to the `kiln_warp_reduce_*` helpers
// and `KILN_WARP` / `KILN_MAX_WARP` below so the math is wave-size correct on
// both backends from one source. Including this header changes nothing until a
// kernel opts in, so it is safe to inject everywhere.
#pragma once

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
// Compile-time upper bound on lanes per wavefront, for `__shared__` sizing.
// AMD wavefronts are at most 64 lanes (CDNA always 64; RDNA wave32/wave64).
#ifndef KILN_MAX_WARP
#define KILN_MAX_WARP 64
#endif
#define KILN_IS_HIP 1
#else
#ifndef KILN_MAX_WARP
#define KILN_MAX_WARP 32
#endif
#define KILN_IS_HIP 0
#endif

// `KILN_WARP` — the runtime lane count of the executing wavefront/warp. On both
// nvcc and hipcc `warpSize` is the built-in device constant (32 on NVIDIA, 32 or
// 64 on AMD depending on wave mode), so reductions that loop `KILN_WARP/2 .. 1`
// are correct regardless of backend. Use `KILN_MAX_WARP` for shared sizing.
#define KILN_WARP (warpSize)

// Full-wavefront mask for the `*_sync` shuffle intrinsics. HIP's
// `__shfl_*_sync` (ROCm 7.x) static_asserts that the mask is a 64-bit integer —
// a 32-bit `0xFFFFFFFF` is rejected precisely because it would silently drop the
// upper 32 lanes of a wave64. So the mask must be 64-bit on HIP and 32-bit on
// CUDA. This is the crux of the wave-size port: kernels MUST route their
// shuffles through KILN_FULL_MASK / kiln_warp_reduce_* rather than a bare
// `0xFFFFFFFF` literal.
#ifndef KILN_FULL_MASK
#if KILN_IS_HIP
#define KILN_FULL_MASK (~0ull)
#else
#define KILN_FULL_MASK (~0u)
#endif
#endif

// Butterfly xor-shuffle across one lane. CRITICAL wave64 detail: HIP's
// `__shfl_xor_sync` compat shim defaults its `width` argument to 32, so an
// offset of 32 on a 64-lane wave is out of range and SELF-references (a single
// 1.0 reduces to 2.0, lanes 32-63 never participate). HIP's NATIVE
// `__shfl_xor(v, off, warpSize)` takes an explicit width and is correct on
// wave32/64; CUDA keeps the masked `_sync` form. Verified on real gfx1151
// wave64.
template <typename T>
__device__ __forceinline__ T kiln_shfl_xor(T v, int offset) {
#if KILN_IS_HIP
    return __shfl_xor(v, offset, warpSize);
#else
    return __shfl_xor_sync(KILN_FULL_MASK, v, offset);
#endif
}

// Intra-wavefront xor-butterfly reductions. CORRECT on NVIDIA and AMD wave32,
// but NOT SAFE on AMD wave64 (RDNA wave64 mangles cross-32-lane shuffles; see
// kiln_shfl_xor). For reductions that must be correct across the whole AMD
// fleet, use kiln_block_reduce_* below. These remain for single-warp / wave32 /
// CUDA-only use sites. Result is valid in every lane (butterfly).
template <typename T>
__device__ __forceinline__ T kiln_warp_reduce_sum(T v) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        v += kiln_shfl_xor(v, offset);
    }
    return v;
}

template <typename T>
__device__ __forceinline__ T kiln_warp_reduce_max(T v) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        T other = kiln_shfl_xor(v, offset);
        v = other > v ? other : v;
    }
    return v;
}

// Wave-size-AGNOSTIC block reductions via shared memory — no cross-lane ops, so
// they are correct on NVIDIA, AMD wave32, AND AMD wave64 (CDNA + RDNA-wave64).
// This matters because RDNA's wave64 mode does NOT perform cross-32-lane
// shuffles correctly (verified on gfx1151: __shfl_xor/__shfl_down at offset 32
// self-reference), so warp-shuffle reductions are unsafe across the AMD fleet.
// `smem` must hold >= blockDim.x elements; blockDim.x MUST be a power of two
// (kiln's reduction launchers guarantee this). Result is returned to every
// thread; a trailing __syncthreads() makes `smem` immediately reusable.
template <typename T>
__device__ __forceinline__ T kiln_block_reduce_sum(T val, T* smem) {
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blk >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    T result = smem[0];
    __syncthreads();
    return result;
}

template <typename T>
__device__ __forceinline__ T kiln_block_reduce_max(T val, T* smem) {
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blk >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            T other = smem[tid + s];
            if (other > smem[tid]) smem[tid] = other;
        }
        __syncthreads();
    }
    T result = smem[0];
    __syncthreads();
    return result;
}
