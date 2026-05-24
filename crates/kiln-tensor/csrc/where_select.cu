// Kiln CUDA-side element-wise ternary mask-based select.
//
// out[i] = mask[i] != 0 ? t[i] : f[i]
//
// `mask` is U8; `t` and `f` share dtype (F32 / BF16 / F16). Output
// has the same shape + dtype as `t`/`f`. Contiguous-only path.
//
// (#1082)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BLOCK_SIZE 256

// Dtype tags — must match the wrapper in cuda_storage.rs.
#define DTYPE_F32  0
#define DTYPE_BF16 1
#define DTYPE_F16  2

namespace {

__global__ void kiln_where_select_f32(const uint8_t* __restrict__ mask,
                                      const float* __restrict__ t,
                                      const float* __restrict__ f,
                                      float* __restrict__ out,
                                      int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = mask[idx] != 0 ? t[idx] : f[idx];
}

__global__ void kiln_where_select_bf16(const uint8_t* __restrict__ mask,
                                       const __nv_bfloat16* __restrict__ t,
                                       const __nv_bfloat16* __restrict__ f,
                                       __nv_bfloat16* __restrict__ out,
                                       int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = mask[idx] != 0 ? t[idx] : f[idx];
}

__global__ void kiln_where_select_f16(const uint8_t* __restrict__ mask,
                                      const __half* __restrict__ t,
                                      const __half* __restrict__ f,
                                      __half* __restrict__ out,
                                      int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = mask[idx] != 0 ? t[idx] : f[idx];
}

} // namespace

extern "C" int kiln_where_select_async(const void* mask,
                                       const void* t,
                                       const void* f,
                                       void* out,
                                       int64_t n_elements,
                                       int dtype,
                                       cudaStream_t stream) {
    if (n_elements < 0) return 1;
    if (n_elements == 0) return 0;

    int64_t blocks_i64 = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_i64 > (int64_t)2147483647) return 3;
    int blocks = static_cast<int>(blocks_i64);

    switch (dtype) {
        case DTYPE_F32:
            kiln_where_select_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const uint8_t*>(mask),
                static_cast<const float*>(t),
                static_cast<const float*>(f),
                static_cast<float*>(out),
                n_elements);
            break;
        case DTYPE_BF16:
            kiln_where_select_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const uint8_t*>(mask),
                static_cast<const __nv_bfloat16*>(t),
                static_cast<const __nv_bfloat16*>(f),
                static_cast<__nv_bfloat16*>(out),
                n_elements);
            break;
        case DTYPE_F16:
            kiln_where_select_f16<<<blocks, BLOCK_SIZE, 0, stream>>>(
                static_cast<const uint8_t*>(mask),
                static_cast<const __half*>(t),
                static_cast<const __half*>(f),
                static_cast<__half*>(out),
                n_elements);
            break;
        default:
            return 4;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
