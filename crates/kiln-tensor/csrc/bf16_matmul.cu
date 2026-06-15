// ROCm BF16 matmul fallback for unstable long-row hipBLASLt shapes.
//
// Computes C[M, N] = A[M, K] @ B[K, N] for row-major contiguous BF16
// operands, accumulating in FP32 and storing BF16. This is intentionally small
// and shape-generic; production uses it only for split q/gate training slices
// where ROCm 7.2 hipBLASLt has returned zeros/NaNs on gfx115x.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 32;

__global__ void kiln_bf16_matmul_bf16_out_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ c,
    int m,
    int n,
    int k) {
    __shared__ float as[BM][BK];
    __shared__ float bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int row = blockIdx.y * BM + ty;
    const int col = blockIdx.x * BN + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < k; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += BM * BN) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const int global_r = blockIdx.y * BM + r;
            const int global_k = k0 + kk;
            as[r][kk] = (global_r < m && global_k < k)
                ? __bfloat162float(a[global_r * k + global_k])
                : 0.0f;
        }
        for (int idx = tid; idx < BK * BN; idx += BM * BN) {
            const int kk = idx / BN;
            const int c_col = idx - kk * BN;
            const int global_k = k0 + kk;
            const int global_c = blockIdx.x * BN + c_col;
            bs[kk][c_col] = (global_k < k && global_c < n)
                ? __bfloat162float(b[global_k * n + global_c])
                : 0.0f;
        }
        __syncthreads();

        if (row < m && col < n) {
#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                acc += as[ty][kk] * bs[kk][tx];
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = __float2bfloat16(acc);
    }
}

} // namespace

extern "C" int kiln_rocm_bf16_matmul_bf16_out_async(
    const void* a,
    const void* b,
    void* c,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
    if (!a || !b || !c) return 1;
    if (m < 0 || n < 0 || k < 0) return 2;
    if (m == 0 || n == 0 || k == 0) return 0;
    if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) return 3;

    dim3 block(BN, BM, 1);
    dim3 grid(
        static_cast<unsigned int>((n + BN - 1) / BN),
        static_cast<unsigned int>((m + BM - 1) / BM),
        1);

    kiln_bf16_matmul_bf16_out_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(a),
        static_cast<const __nv_bfloat16*>(b),
        static_cast<__nv_bfloat16*>(c),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + static_cast<int>(err);
    return 0;
}
