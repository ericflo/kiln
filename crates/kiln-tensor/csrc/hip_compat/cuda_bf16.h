// CUDA -> HIP compatibility shim: `cuda_bf16.h`.
//
// HIP mirrors CUDA's bf16 conversion intrinsics by name
// (`__bfloat162float`, `__float2bfloat16`, ...), so only the storage TYPE
// names differ: CUDA `__nv_bfloat16` / `__nv_bfloat162` -> HIP
// `__hip_bfloat16` / `__hip_bfloat162`.
#pragma once

#include <hip/hip_bf16.h>

#define __nv_bfloat16 __hip_bfloat16
#define __nv_bfloat162 __hip_bfloat162
#define nv_bfloat16 __hip_bfloat16
