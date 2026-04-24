// Kiln causal_conv1d_update kernel — vendored from Dao-AILab/causal-conv1d
// (mamba-ssm), narrowed to decode-only, bf16 activations/weights, F32 state,
// kernel_size == 4, silu-fused.
//
// # Provenance
//
// Algorithm modelled after
// <https://github.com/Dao-AILab/causal-conv1d> `csrc/causal_conv1d_update.cu`
// (commit HEAD as of 2026-04-18), kIsCircularBuffer = false branch. The
// upstream kernel handles arbitrary dtypes, arbitrary kernel widths (2/3/4),
// seqlen > 1, optional bias, optional circular buffer, and an optional
// conv_state_indices gather — kiln only needs the single-token,
// kernel_width=4, bf16, no-bias, linear-buffer path, so this is a reduced
// specialisation rather than a full port. License: Apache 2.0 (upstream);
// kiln retains the same licence for the vendored code.
//
// # Layout
//
// One thread per (batch, channel). Launch:
//   grid  = dim3(B, (C + threads - 1) / threads)
//   block = dim3(threads)
//   threads = 64 (matches upstream).
//
// Per thread:
//   1. Load K-1 floats from conv_state (already F32 in kiln).
//   2. Load 1 bf16 from x, cast to F32.
//   3. Load K bf16 weights for this channel, cast to F32.
//   4. Dot-product into an F32 accumulator.
//   5. Write the new K-1 state (drop oldest, append newest x).
//   6. Optionally apply SiLU: out = z / (1 + exp(-z)).
//   7. Store F32 to out.
//
// Because K is fixed at compile time (4) the window + weight loops are fully
// unrolled and the whole kernel runs out of registers — no shared memory, no
// atomics, no sync.
//
// # Scope
//
// - Forward only (mamba-ssm ships a separate backward; we don't need one for
//   inference).
// - bf16 activations and bf16 weights. F32 state. F32 output.
// - kernel_width == 4 (Qwen3.5 GDN). Other widths return status != 0.
// - No bias, no per-batch conv_state_indices, no circular buffer.

#include "causal_conv1d_update.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreadsPerBlock = 64;
constexpr int kWidth = 4;  // specialisation — Qwen3.5 GDN

template <bool kSilu>
__global__ __launch_bounds__(kThreadsPerBlock) void kiln_conv1d_update_k4_kernel(
    const __nv_bfloat16 *__restrict__ x,        // [B, C, 1]
    const __nv_bfloat16 *__restrict__ weight,   // [C, K]
    float *__restrict__ conv_state,             // [B, C, K-1]
    float *__restrict__ out,                    // [B, C, 1]
    int batch,
    int channels) {
    const int b = blockIdx.x;
    const int c = blockIdx.y * kThreadsPerBlock + threadIdx.x;
    if (c >= channels) return;

    const size_t state_row = (static_cast<size_t>(b) * channels + c) * (kWidth - 1);
    const size_t xy_row = static_cast<size_t>(b) * channels + c;

    // Load K-1 previous inputs from state (F32).
    float window[kWidth];
#pragma unroll
    for (int i = 0; i < kWidth - 1; ++i) {
        window[i] = conv_state[state_row + i];
    }

    // Load current token from x (bf16 -> f32). x is [B, C, 1] so index is `xy_row`.
    window[kWidth - 1] = __bfloat162float(x[xy_row]);

    // Load K weights for this channel (bf16 -> f32). weight is [C, K].
    float w[kWidth];
#pragma unroll
    for (int i = 0; i < kWidth; ++i) {
        w[i] = __bfloat162float(weight[static_cast<size_t>(c) * kWidth + i]);
    }

    // Dot product.
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < kWidth; ++i) {
        acc += window[i] * w[i];
    }

    // Update state: drop oldest, append newest x. Writes K-1 floats.
#pragma unroll
    for (int i = 0; i < kWidth - 1; ++i) {
        conv_state[state_row + i] = window[i + 1];
    }

    // SiLU: z / (1 + exp(-z)). Matches cuda_silu() called in forward.rs.
    if constexpr (kSilu) {
        acc = acc / (1.0f + __expf(-acc));
    }

    out[xy_row] = acc;
}

template <bool kSilu>
__global__ __launch_bounds__(kThreadsPerBlock) void kiln_conv1d_prefill_k4_kernel(
    const __nv_bfloat16 *__restrict__ x,        // [B, C, T]
    const __nv_bfloat16 *__restrict__ weight,   // [C, K]
    float *__restrict__ conv_state,             // [B, C, K-1]
    float *__restrict__ out,                    // [B, C, T]
    int batch,
    int channels,
    int seq_len) {
    const int bc = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_channels = batch * channels;
    if (bc >= total_channels) return;

    const int c = bc % channels;
    const size_t x_base = static_cast<size_t>(bc) * seq_len;
    const size_t state_base = static_cast<size_t>(bc) * (kWidth - 1);
    const size_t weight_base = static_cast<size_t>(c) * kWidth;

    __shared__ float entry_state[kWidth - 1];
    if (tid < kWidth - 1) {
        entry_state[tid] = conv_state[state_base + tid];
    }
    __syncthreads();

    float w[kWidth];
#pragma unroll
    for (int i = 0; i < kWidth; ++i) {
        w[i] = __bfloat162float(weight[weight_base + i]);
    }

    for (int t = tid; t < seq_len; t += blockDim.x) {
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < kWidth; ++j) {
            const int padded_idx = t + j;
            float v;
            if (padded_idx < kWidth - 1) {
                v = entry_state[padded_idx];
            } else {
                v = __bfloat162float(x[x_base + padded_idx - (kWidth - 1)]);
            }
            acc += v * w[j];
        }
        if constexpr (kSilu) {
            acc = acc / (1.0f + __expf(-acc));
        }
        out[x_base + t] = acc;
    }
    __syncthreads();

    if (tid == 0) {
        if (seq_len >= kWidth - 1) {
#pragma unroll
            for (int i = 0; i < kWidth - 1; ++i) {
                conv_state[state_base + i] =
                    __bfloat162float(x[x_base + seq_len - (kWidth - 1) + i]);
            }
        } else if (seq_len == 2) {
            conv_state[state_base + 0] = entry_state[2];
            conv_state[state_base + 1] = __bfloat162float(x[x_base + 0]);
            conv_state[state_base + 2] = __bfloat162float(x[x_base + 1]);
        } else if (seq_len == 1) {
            conv_state[state_base + 0] = entry_state[1];
            conv_state[state_base + 1] = entry_state[2];
            conv_state[state_base + 2] = __bfloat162float(x[x_base]);
        }
    }
}

}  // namespace

extern "C" kiln_conv1d_status_t kiln_causal_conv1d_update_bf16_f32(
    const void *x,
    const void *weight,
    void *conv_state,
    void *out,
    int batch,
    int channels,
    int kernel_width,
    int silu,
    void *stream) {
    if (kernel_width != kWidth) {
        return 1;  // unsupported width (only K=4 compiled)
    }
    if (batch <= 0 || channels <= 0) {
        return 2;
    }

    dim3 grid(batch, (channels + kThreadsPerBlock - 1) / kThreadsPerBlock);
    dim3 block(kThreadsPerBlock);
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    const auto *x_bf = reinterpret_cast<const __nv_bfloat16 *>(x);
    const auto *w_bf = reinterpret_cast<const __nv_bfloat16 *>(weight);
    auto *state_f = reinterpret_cast<float *>(conv_state);
    auto *out_f = reinterpret_cast<float *>(out);

    if (silu != 0) {
        kiln_conv1d_update_k4_kernel<true><<<grid, block, 0, cu_stream>>>(
            x_bf, w_bf, state_f, out_f, batch, channels);
    } else {
        kiln_conv1d_update_k4_kernel<false><<<grid, block, 0, cu_stream>>>(
            x_bf, w_bf, state_f, out_f, batch, channels);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 3;
    }
    return 0;
}

extern "C" kiln_conv1d_status_t kiln_causal_conv1d_prefill_bf16_f32(
    const void *x,
    const void *weight,
    void *conv_state,
    void *out,
    int batch,
    int channels,
    int seq_len,
    int kernel_width,
    int silu,
    void *stream) {
    if (kernel_width != kWidth) {
        return 1;  // unsupported width (only K=4 compiled)
    }
    if (batch <= 0 || channels <= 0 || seq_len <= 1) {
        return 2;
    }

    const int threads = seq_len <= 32 ? 32 : (seq_len <= 64 ? 64 : (seq_len <= 128 ? 128 : 256));
    dim3 grid(batch * channels);
    dim3 block(threads);
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    const auto *x_bf = reinterpret_cast<const __nv_bfloat16 *>(x);
    const auto *w_bf = reinterpret_cast<const __nv_bfloat16 *>(weight);
    auto *state_f = reinterpret_cast<float *>(conv_state);
    auto *out_f = reinterpret_cast<float *>(out);

    if (silu != 0) {
        kiln_conv1d_prefill_k4_kernel<true><<<grid, block, 0, cu_stream>>>(
            x_bf, w_bf, state_f, out_f, batch, channels, seq_len);
    } else {
        kiln_conv1d_prefill_k4_kernel<false><<<grid, block, 0, cu_stream>>>(
            x_bf, w_bf, state_f, out_f, batch, channels, seq_len);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 3;
    }
    return 0;
}
