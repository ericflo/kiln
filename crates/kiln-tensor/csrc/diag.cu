// Kiln CUDA-side diagonal extract / diag-matrix-construct kernels.
//
//   diagonal(t) : t[n, n] -> out[n] where out[i] = t[i*n + i]
//   diag(v)     : v[n]    -> out[n, n] zeros except out[i*n + i] = v[i]
//
// F32 / BF16 / F16. The diag-construct kernel is launched on a
// pre-zeroed `out`; it only writes the n diagonal positions.
//
// (#1082)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BLOCK_SIZE 256

#define DTYPE_F32  0
#define DTYPE_BF16 1
#define DTYPE_F16  2

namespace {

template <typename T>
__global__ void kiln_diagonal_extract(const T* __restrict__ x,
                                      T* __restrict__ out,
                                      int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i * n + i];
}

template <typename T>
__global__ void kiln_diag_build(const T* __restrict__ v,
                                T* __restrict__ out,
                                int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i * n + i] = v[i];
}

} // namespace

extern "C" int kiln_diagonal_extract_async(const void* x,
                                           void* out,
                                           int64_t n,
                                           int dtype,
                                           cudaStream_t stream) {
    if (n <= 0) return n == 0 ? 0 : 1;
    int64_t blocks_i64 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    switch (dtype) {
        case DTYPE_F32:
            kiln_diagonal_extract<float><<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(x),
                static_cast<float*>(out),
                n);
            break;
        case DTYPE_BF16:
            kiln_diagonal_extract<__nv_bfloat16><<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x),
                static_cast<__nv_bfloat16*>(out),
                n);
            break;
        case DTYPE_F16:
            kiln_diagonal_extract<__half><<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __half*>(x),
                static_cast<__half*>(out),
                n);
            break;
        default:
            return 4;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}

extern "C" int kiln_diag_build_async(const void* v,
                                     void* out,
                                     int64_t n,
                                     int dtype,
                                     cudaStream_t stream) {
    if (n <= 0) return n == 0 ? 0 : 1;
    int64_t blocks_i64 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    switch (dtype) {
        case DTYPE_F32:
            kiln_diag_build<float><<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const float*>(v),
                static_cast<float*>(out),
                n);
            break;
        case DTYPE_BF16:
            kiln_diag_build<__nv_bfloat16><<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(v),
                static_cast<__nv_bfloat16*>(out),
                n);
            break;
        case DTYPE_F16:
            kiln_diag_build<__half><<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const __half*>(v),
                static_cast<__half*>(out),
                n);
            break;
        default:
            return 4;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
