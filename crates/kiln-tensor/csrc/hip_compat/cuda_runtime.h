// CUDA -> HIP compatibility shim: `cuda_runtime.h`.
//
// Placed on the hipcc include path (`-I csrc/hip_compat`) so kiln's `.cu`
// kernels compile byte-unchanged under hipcc — their `#include <cuda_runtime.h>`
// resolves here instead of to NVIDIA's header. Only the small runtime surface
// the kernels actually use is mapped (token-for-token, the hipify convention).
//
// On the CUDA build (nvcc) this directory is NOT on the include path, so the
// real CUDA header is used and this file is inert.
#pragma once

#include <hip/hip_runtime.h>

// Stream + error handle types.
#define cudaStream_t hipStream_t
#define cudaError_t hipError_t
#define cudaError hipError_t

// Status codes / last-error.
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetErrorString hipGetErrorString

// The few memory calls a couple of kernels make directly.
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStreamSynchronize hipStreamSynchronize
