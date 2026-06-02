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

// Portable full-wavefront reductions. The loop bound is the *runtime* warpSize,
// so a single call reduces all 32 OR 64 lanes correctly — replacing the
// hardcoded `for (offset = 16; ...)` idiom. Result is valid in lane 0 (and,
// because xor-shuffle is a butterfly, in every lane).
template <typename T>
__device__ __forceinline__ T kiln_warp_reduce_sum(T v) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(KILN_FULL_MASK, v, offset);
    }
    return v;
}

template <typename T>
__device__ __forceinline__ T kiln_warp_reduce_max(T v) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        T other = __shfl_xor_sync(KILN_FULL_MASK, v, offset);
        v = other > v ? other : v;
    }
    return v;
}
