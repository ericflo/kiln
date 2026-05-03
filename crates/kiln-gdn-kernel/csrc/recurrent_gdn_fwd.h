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

kiln_gdn_recurrent_status_t kiln_gdn_decode_gates_recurrent_vf32_bf16(
    const void *q,
    const void *k,
    const void *v,
    const void *a,
    const void *b,
    const void *a_log,
    const void *dt_bias,
    void *state,
    const void *z,
    const void *weight,
    void *out,
    int batch,
    int q_heads,
    int value_heads,
    int dk,
    int dv,
    float eps,
    void *stream
);

kiln_gdn_recurrent_status_t kiln_gdn_decode_gates_recurrent_bf16(
    const void *q,
    const void *k,
    const void *v,
    const void *a,
    const void *b,
    const void *a_log,
    const void *dt_bias,
    void *state,
    const void *z,
    const void *weight,
    void *out,
    int batch,
    int q_heads,
    int value_heads,
    int dk,
    int dv,
    float eps,
    void *stream
);

#ifdef __cplusplus
}
#endif
