// kiln-gdn-kernel: fused GDN gates kernel
//
// Collapses the `kiln/gdn/gates` NVTX region — identified in PR #156's
// profile as 18.2% of A6000 BF16+Marlin decode wallclock — into a single
// CUDA launch. Replaces the ~8-op candle chain:
//
//   beta     = sigmoid(b)                              // [B, T, nv] bf16
//   a_f32    = a.to_dtype(F32)
//   a_log32  = A_log.to_dtype(F32)
//   bias32   = dt_bias.to_dtype(F32)
//   a_biased = a_f32 + bias32                          // broadcast_add
//   sp       = softplus(a_biased)                      // log(1+exp(x))
//   decay    = -exp(a_log32)                           // [nv] F32
//   g        = bf16(sp * decay)                        // broadcast_mul
//
// with one kernel that reads `a`, `b` in bf16 plus `A_log`, `dt_bias` in
// bf16/F32 and writes `beta`, `g` in bf16 directly. Algorithm mirrors
// `naive_gdn_gate` in fla-org/flash-linear-attention
// (fla/ops/gated_delta_rule/gate.py) — sigmoid + softplus + exp +
// elementwise mul — ported from Triton to raw CUDA C.
//
// Envelope:
//   - bf16 activations + bf16 weights (kiln loads A_log / dt_bias in bf16)
//   - All F32 intermediates inside the kernel for numerical stability
//   - Shape [B, T, nv]; nv <= 256
//
// Parity oracle: the candle-op chain above in
// kiln_model::forward::gated_deltanet_forward (Step 6).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#include "gdn_gates.h"

namespace {

// softplus(x) = log(1 + exp(x)) with the standard numerically-stable form
// used by PyTorch / FLA: if x > threshold, return x (avoids exp overflow);
// else return log1p(exp(x)).
__device__ __forceinline__ float stable_softplus(float x) {
    // Matches torch.nn.functional.softplus default threshold (20.0).
    if (x > 20.0f) {
        return x;
    }
    return log1pf(expf(x));
}

// sigmoid(x) = 1 / (1 + exp(-x)).
__device__ __forceinline__ float stable_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float param_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float param_to_float(float x) {
    return x;
}


__global__ void gdn_gate_beta_bf16_kernel(
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ beta_out,
    int32_t nv
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= nv) return;
    const int idx = row * nv + tid;
    beta_out[idx] = __float2bfloat16(stable_sigmoid(__bfloat162float(b[idx])));
}

__global__ void gdn_gate_g_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ g_out,
    int32_t nv
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= nv) return;
    const int idx = row * nv + tid;
    const float a_biased = __bfloat162float(a[idx]) + __bfloat162float(dt_bias[tid]);
    g_out[idx] = __float2bfloat16(stable_softplus(a_biased) * -expf(__bfloat162float(A_log[tid])));
}

// One block per (B, T) row; each thread handles one head (nv dimension).
// Loads A_log, dt_bias once per thread (tiny [nv] tensors).
template <typename ALogT, typename DtBiasT>
__global__ void gdn_gates_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,        // [B, T, nv] bf16
    const __nv_bfloat16* __restrict__ b,        // [B, T, nv] bf16
    const ALogT* __restrict__ A_log,            // [nv] bf16/f32
    const DtBiasT* __restrict__ dt_bias,        // [nv] bf16/f32
    __nv_bfloat16* __restrict__ beta_out,       // [B, T, nv] bf16
    __nv_bfloat16* __restrict__ g_out,          // [B, T, nv] bf16
    int32_t nv
) {
    const int row = blockIdx.x;                    // linearised (B, T) row
    const int tid = threadIdx.x;
    if (tid >= nv) return;

    const int idx = row * nv + tid;

    const float a_val  = __bfloat162float(a[idx]);
    const float b_val  = __bfloat162float(b[idx]);
    const float A_log_val = param_to_float(A_log[tid]);
    const float dt_bias_val = param_to_float(dt_bias[tid]);

    // beta = sigmoid(b)
    const float beta_f = stable_sigmoid(b_val);

    // g = -exp(A_log) * softplus(a + dt_bias)
    const float a_biased = a_val + dt_bias_val;
    const float sp = stable_softplus(a_biased);
    const float neg_decay = -expf(A_log_val);
    const float g_f = sp * neg_decay;

    beta_out[idx] = __float2bfloat16(beta_f);
    g_out[idx]    = __float2bfloat16(g_f);
}

}  // namespace

template <typename ALogT, typename DtBiasT>
int32_t launch_gdn_gates_bf16(
    const void* a,
    const void* b,
    const void* A_log,
    const void* dt_bias,
    void* beta_out,
    void* g_out,
    int32_t rows,
    int32_t nv,
    void* stream_raw
) {
    if (rows <= 0 || nv <= 0) return 0;
    if (nv > 256) return 2; // envelope guard

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);

    // Round nv up to the next warp-friendly size but cap at 256.
    int threads = nv;
    if (threads < 32) {
        threads = 32;
    } else if (threads & 31) {
        threads = (threads + 31) & ~31;
    }

    dim3 grid(rows);
    dim3 block(threads);

    gdn_gates_bf16_kernel<ALogT, DtBiasT><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<const ALogT*>(A_log),
        reinterpret_cast<const DtBiasT*>(dt_bias),
        reinterpret_cast<__nv_bfloat16*>(beta_out),
        reinterpret_cast<__nv_bfloat16*>(g_out),
        nv
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return static_cast<int32_t>(err);
    }
    return 0;
}

extern "C" int32_t kiln_gdn_gates_bf16(
    const void* a,
    const void* b,
    const void* A_log,
    const void* dt_bias,
    void* beta_out,
    void* g_out,
    int32_t rows,
    int32_t nv,
    void* stream_raw
) {
    return launch_gdn_gates_bf16<__nv_bfloat16, __nv_bfloat16>(
        a, b, A_log, dt_bias, beta_out, g_out, rows, nv, stream_raw);
}

extern "C" int32_t kiln_gdn_gates_bf16_f32_params(
    const void* a,
    const void* b,
    const void* A_log,
    const void* dt_bias,
    void* beta_out,
    void* g_out,
    int32_t rows,
    int32_t nv,
    void* stream_raw
) {
    return launch_gdn_gates_bf16<float, float>(
        a, b, A_log, dt_bias, beta_out, g_out, rows, nv, stream_raw);
}

extern "C" int32_t kiln_gdn_gates_bf16_f32_bf16_params(
    const void* a,
    const void* b,
    const void* A_log,
    const void* dt_bias,
    void* beta_out,
    void* g_out,
    int32_t rows,
    int32_t nv,
    void* stream_raw
) {
    return launch_gdn_gates_bf16<float, __nv_bfloat16>(
        a, b, A_log, dt_bias, beta_out, g_out, rows, nv, stream_raw);
}


extern "C" int32_t kiln_gdn_gate_beta_bf16(
    const void* b,
    void* beta_out,
    int32_t rows,
    int32_t nv,
    void* stream_raw
) {
    if (rows <= 0 || nv <= 0) return 0;
    if (nv > 256) return 2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    int threads = nv < 32 ? 32 : ((nv + 31) & ~31);
    cudaGetLastError();
    gdn_gate_beta_bf16_kernel<<<dim3(rows), dim3(threads), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<__nv_bfloat16*>(beta_out),
        nv);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : static_cast<int32_t>(err);
}

extern "C" int32_t kiln_gdn_gate_g_bf16(
    const void* a,
    const void* A_log,
    const void* dt_bias,
    void* g_out,
    int32_t rows,
    int32_t nv,
    void* stream_raw
) {
    if (rows <= 0 || nv <= 0) return 0;
    if (nv > 256) return 2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    int threads = nv < 32 ? 32 : ((nv + 31) & ~31);
    cudaGetLastError();
    gdn_gate_g_bf16_kernel<<<dim3(rows), dim3(threads), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(A_log),
        reinterpret_cast<const __nv_bfloat16*>(dt_bias),
        reinterpret_cast<__nv_bfloat16*>(g_out),
        nv);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : static_cast<int32_t>(err);
}
