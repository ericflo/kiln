// CUDA -> HIP compatibility shim: `cuda.h` (crate-local, kiln-gdn-kernel).
//
// Several GDN `.cu` sources carry a vestigial `#include <cuda.h>` (the CUDA
// driver API) even though they only use the runtime API. NVIDIA ships `cuda.h`;
// HIP does not, so under hipcc the include must resolve to this shim. It is
// placed on the hipcc include path via build.rs's `build_rocm()` arm and is
// therefore HIP-only — on the nvcc (CUDA) build this directory is NOT on the
// include path, so the real NVIDIA `cuda.h` is used and this file is inert. The
// shared kiln-tensor `hip_compat/` is intentionally left untouched.
#pragma once

#include <cuda_runtime.h>

// `gdn_fwd_sub.cu` opts into the larger dynamic-shared-memory carveout via
// `cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
// bytes)`. CUDA's `cudaFuncSetAttribute` accepts a typed device-function
// pointer; HIP's `hipFuncSetAttribute` takes `const void*` (and is
// `[[nodiscard]]`). A HIP-only templated forwarding shim lets the call site
// compile byte-unchanged while keeping the CUDA path identical.
#ifndef KILN_GDN_CUDA_FUNC_ATTR_SHIM
#define KILN_GDN_CUDA_FUNC_ATTR_SHIM 1
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
template <typename FnPtr>
static inline void kiln_gdn_cudaFuncSetAttribute(FnPtr fn, hipFuncAttribute attr, int value) {
    (void)hipFuncSetAttribute(reinterpret_cast<const void*>(fn), attr, value);
}
#define cudaFuncSetAttribute kiln_gdn_cudaFuncSetAttribute
#endif
