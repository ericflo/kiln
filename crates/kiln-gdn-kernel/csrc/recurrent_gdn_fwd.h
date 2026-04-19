// Kiln Gated DeltaNet (GDN) recurrent (single-token) forward C-ABI wrapper.
//
// This kernel handles the seq_len == 1 / chunk_size == 1 decode path that
// the chunkwise forward (`gdn_fwd_sub.cu`) leaves on the table because it
// pays per-chunk setup cost regardless of token count.
//
// Per-token GDN recurrence:
//
//   decay   = exp(g_t)                                   (scalar per (B,H))
//   v_pred  = k_t · (decay * S_{t-1})                    (dv-vector)
//   delta_t = beta_t * (v_t - v_pred)                    (dv-vector)
//   S_t     = decay * S_{t-1} + k_t ⊗ delta_t            ([dk, dv])
//   out_t   = q_t · S_t                                  (dv-vector)
//
// One CUDA block per (batch, head). One thread per dv column. Each thread
// owns a single column of the [dk, dv] state in registers (loaded once per
// call), so the inner per-thread loop is dk-long with no cross-thread
// synchronisation beyond the initial load of q_t / k_t into shared memory.
//
// Pointer layout (all bf16, all CUDA, all contiguous):
//   q     : [B*H, dk]
//   k     : [B*H, dk]
//   v     : [B*H, dv]
//   beta  : [B*H]
//   g     : [B*H]
//   state : [B*H, dk, dv]   (read-modify-write in place)
//   out   : [B*H, dv]

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int kiln_gdn_recurrent_status_t;

kiln_gdn_recurrent_status_t kiln_gdn_recurrent_forward(
    const void *q,
    const void *k,
    const void *v,
    const void *beta,
    const void *g,
    void *state,
    void *out,
    int batch_heads,
    int dk,
    int dv,
    void *stream
);

// Fused variant: absorbs qk_norm (L2-norm+scale on Q, L2-norm on K) and
// gated_norm (RMSNorm + silu-gated weight mul on attn_out) into the same
// per-(batch,head) block as the recurrent step, keeping normalized Q/K and
// the pre-norm attn_out in on-chip memory. Eliminates the ~5 bf16↔F32 HBM
// round-trips that the candle / standalone-kernel path pays at decode.
//
// Extra pointer arguments (all bf16, all CUDA, all contiguous):
//   q_raw    : [B*H, dk]   — pre-L2 Q (replaces q of the non-fused entry)
//   k_raw    : [B*H, dk]   — pre-L2 K
//   z        : [B*H, dv]   — output gate (pre silu)
//   gamma    : [dv]        — RMSNorm learnable scale (shared across B*H)
//
// Extra scalar arguments:
//   q_scale  : f32 — applied to Q after L2 norm (typically 1/sqrt(dk))
//   l2_eps   : f32 — eps inside sqrt for L2 norm (typically 1e-6)
//   rms_eps  : f32 — eps inside sqrt for RMSNorm (typically 1e-6)
//
// The `out` tensor receives the final gated-rms-normed value in bf16;
// callers do not need to run qk_norm or gated_norm separately. `state` is
// updated in place with the post-recurrence F32-stored-as-bf16 state.
kiln_gdn_recurrent_status_t kiln_gdn_recurrent_forward_fused_norm(
    const void *q_raw,
    const void *k_raw,
    const void *v,
    const void *beta,
    const void *g,
    const void *z,
    const void *gamma,
    void *state,
    void *out,
    int batch_heads,
    int dk,
    int dv,
    float q_scale,
    float l2_eps,
    float rms_eps,
    void *stream
);

#ifdef __cplusplus
}
#endif
