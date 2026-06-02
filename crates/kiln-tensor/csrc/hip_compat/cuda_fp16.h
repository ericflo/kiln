// CUDA -> HIP compatibility shim: `cuda_fp16.h`.
//
// HIP's `hip/hip_fp16.h` provides `__half` / `__half2` and the
// `__half2float` / `__float2half` intrinsics under the same names CUDA uses,
// so this shim is just a redirect.
#pragma once

#include <hip/hip_fp16.h>
