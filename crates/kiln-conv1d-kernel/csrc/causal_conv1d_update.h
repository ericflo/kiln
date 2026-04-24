// Kiln vendored causal_conv1d_update C-ABI wrapper.
//
// Decode-only (seqlen == 1) single-step variant of the mamba-ssm /
// Dao-AILab/causal-conv1d `causal_conv1d_update_kernel`, narrowed to kiln's
// Qwen3.5-4B GDN envelope:
//   - bf16 activations / bf16 weights.
//   - F32 conv_state (matches kiln's state init in `LinearAttentionCache`).
//   - kernel_size == 4, stride == 1, dilation == 1.
//   - silu fused inline (matches the `cuda_silu` call site in forward.rs).
//
// Replaces this candle op chain inside `kiln/gdn/conv` (12.2% of decode
// wall-clock per PROFILING.md post-#158):
//
//   x_f32 = to_dtype(F32)(x)
//   w_f32 = to_dtype(F32)(weight).reshape((C, K))
//   state_f32 = to_dtype(F32)(conv_state)
//   window = cat(state_f32, x_f32, dim=2)
//   out = sum(window * w_f32.unsqueeze(0), dim=2).unsqueeze(2)
//   conv_state = window.narrow(2, 1, K-1).contiguous()
//   out = silu(out)  // caller applies separately in F32
//
// …with a single CUDA launch: one thread per (batch, channel), K==4
// registers for the window, F32 accumulator, F32 state, F32 silu epilogue.
//
// Pointer layout (all contiguous CUDA):
//   x          : bf16, [B, C, 1]
//   weight     : bf16, [C, 1, K]        (we treat as [C, K] — weight_width_stride = 1)
//   conv_state : f32,  [B, C, K-1]       — updated in place
//   out        : f32,  [B, C, 1]

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int kiln_conv1d_status_t;

// Single-step causal depthwise conv1d update with fused SiLU.
//
// All pointers are CUDA device memory, contiguous.
//   x          : bf16, [B, C, 1]
//   weight     : bf16, [C, K]  (flat; caller's [C, 1, K] view is already contiguous)
//   conv_state : f32,  [B, C, K-1] — read and written
//   out        : f32,  [B, C, 1]
// Shape:
//   B    = batch
//   C    = channels
//   K    = kernel width (must be 4 for the current specialisation)
//   silu = 1 to fuse SiLU, 0 to leave the raw accumulator (parity-test mode)
//
// Returns 0 on success, non-zero on launch/config failure.
kiln_conv1d_status_t kiln_causal_conv1d_update_bf16_f32(
    const void *x,
    const void *weight,
    void *conv_state,
    void *out,
    int batch,
    int channels,
    int kernel_width,
    int silu,
    void *stream
);

// Multi-token causal depthwise conv1d prefill with fused SiLU.
//
// All pointers are CUDA device memory, contiguous.
//   x          : bf16, [B, C, T], T > 1
//   weight     : bf16, [C, K]  (flat; caller's [C, 1, K] view is already contiguous)
//   conv_state : f32,  [B, C, K-1] — read as the entry state, then written
//   out        : f32,  [B, C, T]
// Shape:
//   B    = batch
//   C    = channels
//   T    = sequence length (must be > 1)
//   K    = kernel width (must be 4 for the current specialisation)
//   silu = 1 to fuse SiLU, 0 to leave the raw accumulator (parity-test mode)
//
// Returns 0 on success, non-zero on launch/config failure.
kiln_conv1d_status_t kiln_causal_conv1d_prefill_bf16_f32(
    const void *x,
    const void *weight,
    void *conv_state,
    void *out,
    int batch,
    int channels,
    int seq_len,
    int kernel_width,
    int silu,
    void *stream
);

#ifdef __cplusplus
}
#endif
