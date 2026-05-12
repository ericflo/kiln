# Vulkan-native Gated DeltaNet (GDN) — design + math

## Shipped state (2026-05-11)

The math + phasing in this document has been substantially implemented:

| Phase | Component | Status | Files |
|-------|-----------|--------|-------|
| G1 | VkLinearAttentionState | ✅ | `vk_ops/gdn_state.rs` |
| G1 | conv1d forward (existing kernel wrapper) | ✅ | `vk_ops/conv1d.rs` |
| G1 | conv1d backward (new shader) | ✅ | `csrc/shaders/vk_causal_conv1d_bwd.comp` |
| G1 | gates forward | ✅ | `vk_ops/gdn_gates.rs` |
| G1 | gates backward (new shader) | ✅ | `csrc/shaders/vk_gdn_gates_bwd.comp` |
| G1 | gated RMSNorm forward | ✅ | `vk_ops/gdn_gated_rms_norm.rs` |
| G1 | gated RMSNorm backward (CPU) | ✅ | same file |
| G2 | chunk_prep wrapper | ✅ | `vk_ops/gdn_chunk_prep.rs` |
| G2 | solve_tri (CPU fallback — see note below) | ✅ | `vk_ops/solve_tri.rs` |
| G2 | chunkwise forward composition | ✅ | `vk_ops/gdn_chunkwise.rs` |
| G3 | solve_tri_transpose (CPU) | ✅ | `vk_ops/gdn_chunk_bwd.rs` |
| G3 | chunk_scan_bwd (CPU) | ✅ | same file |
| G3 | state_exit_bwd (CPU) | ✅ | same file |
| G4 | reverse_cumsum (new shader) | ✅ | `csrc/shaders/vk_reverse_cumsum.comp` |
| G4 | chunk_prep_bwd (CPU) | ✅ | `vk_ops/gdn_chunk_bwd.rs` |
| G5 | cross-chunk autograd composition | ✅ | `vk_ops/gdn_chunkwise.rs::vk_gdn_chunkwise` |
| G6 | full GDN layer + dispatch | ✅ | `kiln-model/src/vk_forward.rs::vk_gdn_layer_forward` |
| G7 | trainer integration + state plumbing | ✅ | `vk_train.rs::vk_native_sft_train` |

**Note on solve_tri**: The existing inference shader `solve_tri.comp`
requests 192 KB of shared memory which exceeds typical per-workgroup
limits and SIGFPEs at pipeline creation on Strix Halo. The vk-native
chunkwise forward currently uses a CPU forward-substitution fallback
for the W solve (sizes are bounded — ~16 KB per layer per chunk). A
correctly-sized GLSL replacement is a follow-up.

Per-piece parity tests live in:
- `tests/vk_gdn_foundation_parity.rs` (4 forward tests)
- `tests/vk_gdn_backward_parity.rs` (6 backward tests, finite-diff
  validated for the analytic backward pieces)
- `tests/vk_gdn_chunkwise_parity.rs` (6 tests, includes T=128 vs CPU
  per-token reference within 7.45e-8)

End-to-end Qwen3.5-4B run on `/tmp/sft.jsonl` is the next validation
gate (no code changes needed — just runtime).


Qwen3.5-4B is a hybrid: 24 of its 32 layers are Gated DeltaNet (linear
recurrent attention with a chunkwise-parallel forward), 8 are full GQA.
For vk-native end-to-end training of this model the GDN layers must
produce VkTensor outputs and propagate gradients to their inputs.

The existing inference stack has ~12 specialized GDN compute shaders
(see `crates/kiln-vulkan-kernel/csrc/shaders/gdn_*.comp`) but none of
them have a backward kernel — the existing trainer relies on candle's
autograd through the candle-native `gdn_chunkwise_recurrence`
implementation in `crates/kiln-model/src/forward.rs:4679`. That path
materializes per-chunk intermediates as candle CPU tensors, which is
exactly the storage class vk-native eliminates. So we need a
vk-native equivalent for both forward and backward.

This document is the spec + math reference.

## High-level architecture of one GDN layer

```text
x_in   →  RMSNorm   →  in_proj_qkv → conv1d → SiLU → split → q, k, v
                    →  in_proj_ab  → split → α, β  (gates)
                    →  in_proj_z   → z (output gate)
       →  chunkwise_recurrence(q, k, v, α, β)  →  o
       →  gated_rmsnorm(o, z)                  →  o_normed
       →  out_proj                             →  x_out
```

`chunkwise_recurrence` is the algorithmic core. The rest is in-projection
matmuls (already covered by `vk_matmul_bf16w` + LoRA composition),
RMSNorm (`vk_rmsnorm`), conv1d (new), SiLU (`vk_silu`), and elementwise
ops.

## Per-token recurrence

Using the gated DeltaNet formulation from
`gdn_single_token_recurrence` (`forward.rs:4604`):

```
S_0 = state_in                          # [B, nv, dk, dv]
For t = 1..T:
    p_t       = exp(g_t)                # scalar gate per (B, nv, t)
    ks_t      = k_t · S_{t-1}           # [B, nv, dv]   (matrix-vector)
    q_s_t     = q_t · S_{t-1}           # [B, nv, dv]
    v'_t      = v_t - p_t * ks_t        # [B, nv, dv]
    w_t       = β_t * v'_t              # [B, nv, dv]
    out_t     = p_t * q_s_t + (q_t · k_t.T) · w_t       # [B, nv, dv]
            (the second term is a scalar — q_t·k_t — times w_t)
    S_t       = p_t * S_{t-1} + k_t.T · w_t             # outer product
```

Shapes: q, k ∈ [dk]; v, w ∈ [dv]; S ∈ [dk, dv]; per-(B, nv, t).

## Chunkwise parallel form (T ≥ 1)

Within a chunk of C tokens (typically C=64), let `G[t] = Σ_{i≤t} g[i]`
(cumulative gate). Then:

```
P[t]            = exp(G[t])                                   # [C]
A_strict[t, i]  = exp(G[t] - G[i]) * (k_t · k_i)   for i < t  # [C, C] strict-lower
                                       ; 0          for i ≥ t

V'[t]           = v[t] - exp(G[t]) * (k_t · S_in)             # [C, dv]
B_mask[t, i]    = exp(G[t] - G[i]) * (q_t · k_i)   for i ≤ t  # [C, C] causal
                                       ; 0          for i > t

W[t]            = β[t] * (V'[t] - Σ_{i<t} A_strict[t, i] * W[i])
                = (I + diag(β) * A_strict)^{-1} · diag(β) · V'  # forward sub

out[t]          = exp(G[t]) * (q_t · S_in)  +  Σ_{i≤t} B_mask[t, i] * W[i]

S_exit          = exp(G[C-1]) * S_in  +  Σ_i exp(G[C-1] - G[i]) * k[i] ⊗ W[i]
```

This is numerically equivalent to the per-token loop. Stages (existing
inference kernels):

| Stage | Existing kernel | Bindings | Math |
|-------|-----------------|----------|------|
| 1: cumsum + decay matrix + V' + q_s_scaled + last column | `gdn_chunk_prep.comp` | g, v, kkt, qkt, ks_entry, q_s → 6 outputs | the formulas above |
| 2: forward sub (W = ...) | `solve_tri.comp` | A_strict, V_prime_beta_scaled → W | triangular solve |
| 3: output | `gdn_chunk_scan.comp` | b_mask, W, q_s_scaled → out_chunk | mat-mat |
| 4: state exit | inline in inference | decay_last_col, k.T, W → S_new | `S_in*p_last + k.T @ (decay*W)` |

There is also a **fused full-chunk forward** kernel (`gdn_full_chunk_forward.comp`)
that does all four stages in one launch when C=64, BF16.

## Backward derivation

Let `L` be the loss and `dL/dout[t]` the upstream gradient for each
chunk output token. The dependencies are:

```
out[t]    ← q_s_scaled[t]      ← q[t], S_in, P[t]
          ← b_mask[t, ≤t] · W
b_mask[t,i] ← q[t], k[i], G[t], G[i]
W[t]      ← β[t], V'[t], A_strict[t, <t], W[<t]   (recursive)
V'[t]     ← v[t], k[t], P[t], S_in
A_strict[t,i] ← k[t], k[i], G[t], G[i]
S_exit    ← S_in, P[C-1], k, W
```

### Backward pass per chunk (working backward in time)

**Stage 4 (state exit) backward.** Given `dS_exit` from the next chunk:

```
dS_in   += P[C-1] * dS_exit                                # accumulates into state-bwd
dW[i]   += k[i] · (decay_last_col[i] * dS_exit).T          # each W gets a contribution
dk[i]   += W[i].T · (decay_last_col[i] * dS_exit)          # symmetric
dG[C-1] += <S_in * P[C-1], dS_exit> + Σ_i decay_last_col[i] * <k[i] ⊗ W[i], dS_exit>
dG[i]   -= decay_last_col[i] * <k[i] ⊗ W[i], dS_exit>      # each i contributes
```

**Stage 3 (output) backward.** Given `dout[t]`:

```
dq_s_scaled[t] = dout[t]
dW            +=  b_mask.T @ dout                          # accumulated into dW
db_mask        =  dout @ W.T                                # per (t, i)
```

**Stage 2 (forward-sub for W) backward.** Given accumulated `dW`:

This is the trickiest piece. We have `W = (I + diag(β)·A_strict)^{-1} · diag(β) · V'`,
so:

Let `M = I + diag(β)·A_strict` (lower triangular with unit diagonal in
the (i, i)·β scaled formulation), `r = diag(β)·V'`. Then `W = M^{-1} r`.

The backward is the standard triangular-solve adjoint:

```
dr = M^{-T} · dW                                  # solve M.T · dr = dW (upper-tri solve)
dM = - dr · W.T                                   # rank-1 outer product accumulator
dV'    = diag(β) · dr
dβ     = Σ_t dr[t] · V'[t] - Σ_t Σ_{i<t} dM[t,i] · A_strict[t,i] / β[t]
        ... (collapses; β enters M as diag(β)·A_strict)
dA_strict = - diag(β) · dr · W.T (above) restricted to strict-lower
```

The two triangular solves (forward for W, backward (transpose-system)
for dr) use the same kernel surface — `solve_tri` for the forward,
`solve_tri_transpose` (NEW) for the backward.

**Stage 1 (chunk_prep) backward.** Given `dV'`, `dB_mask`, `dA_strict`,
`dq_s_scaled`, `dG[C-1]`, `dG[i]`:

```
# V' = v - P · ks_entry, where ks_entry = k · S_in
dv     = dV'
dk     -= P · S_in.T · dV'                       # via ks_entry
dS_in  -= P · k.T · dV'                          # via ks_entry (chunk-summed)
dG     -= ks_entry · dV' summed over dv          # via P = exp(G)

# B_mask[t,i] = exp(G[t] - G[i]) * (q_t · k_i)    for i ≤ t
dq[t]  += Σ_{i≤t} dB_mask[t,i] * exp(G[t]-G[i]) * k[i]
dk[i]  += Σ_{t≥i} dB_mask[t,i] * exp(G[t]-G[i]) * q[t]
dG[t]  += Σ_{i≤t} dB_mask[t,i] * exp(G[t]-G[i]) * (q_t · k_i)
dG[i]  -= same

# A_strict[t,i] = exp(G[t] - G[i]) * (k_t · k_i)  for i < t
dk[t]  += Σ_{i<t} dA_strict[t,i] * exp(G[t]-G[i]) * k[i]
dk[i]  += Σ_{t>i} dA_strict[t,i] * exp(G[t]-G[i]) * k[t]
dG[t]  += Σ_{i<t} dA_strict[t,i] * exp(G[t]-G[i]) * (k_t · k_i)
dG[i]  -= same

# q_s_scaled = q_s * P, q_s = q · S_in
dq_s   = dq_s_scaled · P
dq    += dq_s · S_in.T
dS_in += q.T · dq_s         # accumulated across chunk
dG    += q_s · dq_s_scaled  # via P

# G = cumsum(g)  →  dg[t] = Σ_{s≥t} dG[s]   (reverse cumsum)
dg = reverse_cumsum(dG)
```

### Cross-chunk state propagation

The forward propagates `S_exit` of chunk `c` as `S_in` for chunk `c+1`.
Backward: `dS_in` for chunk `c+1` becomes `dS_exit` for chunk `c`.
Process chunks in reverse order so that each chunk's `dS_exit` is
available when we run its backward.

For the very last chunk, `dS_exit` is whatever the caller passes (or
zero if the state isn't used downstream — typical for training where
we only train on the loss from the chunk outputs).

For the very first chunk, `dS_in` is the gradient w.r.t. the
incoming state (typically zero in training since `S_in` for the first
chunk is the model's initial recurrent state, which is not trained).

## Kernel inventory

### Already exist (forward, used by inference)

| Shader | Does |
|--------|------|
| `gdn_chunk_prep.comp` | cumsum, exp, decay matrices, V', q_s_scaled, last column |
| `solve_tri.comp` | triangular forward-sub for W |
| `gdn_chunk_scan.comp` | output computation from b_mask + W + q_s_scaled |
| `gdn_full_chunk_forward.comp` | fused all four stages (C=64 BF16 fast path) |
| `gdn_recurrent_step_parallel.comp` | single-token decode |
| `causal_conv1d.comp` | causal 1D convolution (forward) |
| `gdn_in_proj_decode_*.comp` | in-projection matmuls (decode-only variants) |
| `gdn_gates.comp` | β/g gate computation |
| `gdn_gated_rms_norm.comp` | output RMSNorm with gate |
| `gdn_decode_gates_recurrent_rmsnorm.comp` | fused decode pipeline |

### Need new (backward + vk-native wrappers)

| Shader | Does |
|--------|------|
| `vk_gdn_chunk_prep_bwd.comp` | reverse cumsum + propagate dG, dV', dB_mask, dA_strict to dq, dk, dv, dg, dS_in |
| `vk_solve_tri_transpose.comp` | upper-tri solve for `M.T · dr = dW` (backward of forward-sub) |
| `vk_gdn_chunk_scan_bwd.comp` | propagate dout to dW, db_mask, dq_s_scaled |
| `vk_gdn_state_exit_bwd.comp` | propagate dS_exit to dS_in, dW, dk, dG |
| `vk_causal_conv1d_bwd.comp` | conv1d backward (gradient w.r.t. input + weights) |
| `vk_gdn_gates_bwd.comp` | gate backward (sigmoid + chained ops) |
| `vk_gdn_gated_rms_norm_bwd.comp` | gated RMSNorm backward |

Plus Rust autograd wrappers per kernel.

## Implementation phasing

This is genuinely 1-2 weeks of focused work. The plan:

**Phase G1 (foundation)**:
- Math derivation doc (this file).
- VkLinearAttentionState type + upload of GDN base weights into VkTensors.
- Wrappers around existing forward kernels via `dispatch_simple`.
- vk-native conv1d forward + backward.
- vk-native gates + gated_rms_norm forward + backward.

**Phase G2 (chunk forward)**:
- vk-native equivalent of `gdn_chunkwise_recurrence` composing existing kernels in VkTensor space.
- Parity test: forward output matches candle reference at small T, dk, dv.
- Hot-path fusion via existing `gdn_full_chunk_forward.comp` for C=64.

**Phase G3 (chunk backward — tractable pieces)**:
- vk_solve_tri_transpose kernel + parity vs CPU dense linsolve.
- vk_gdn_chunk_scan_bwd + parity.
- vk_gdn_state_exit_bwd + parity.

**Phase G4 (chunk backward — chunk_prep)**:
- vk_gdn_chunk_prep_bwd: the most complex backward, propagates through
  cumsum, exp, masked outer products. Needs careful derivation per
  the math above and per-piece parity tests.
- Reverse-cumsum kernel.

**Phase G5 (cross-chunk integration)**:
- Process chunks in reverse during backward, threading dS_exit/dS_in.
- Single-token recurrence backward (degenerate case of chunk
  backward with C=1).

**Phase G6 (end-to-end)**:
- Full GDN layer forward + backward as a VkBackwardOp composition.
- Parity test: gradient of `loss = mean(out · arbitrary)` matches
  candle reference for small synthetic shapes.
- Wire into vk_transformer_layer dispatch (Full vs Linear).

**Phase G7 (Qwen3.5-4B end-to-end)**:
- Real-weights upload of GDN params into VkLinearAttentionWeights.
- Run /tmp/sft.jsonl training with all 24 GDN + 8 FullAttn layers
  vk-native. Validate loss decreases, adapter loads, inference works.

## Why this is well-trodden literature

The math:
- The **delta rule** comes from Schlag et al. (2021), "Linear
  Transformers Are Secretly Fast Weight Programmers."
- The **gated** variant + chunkwise parallel form comes from Yang
  et al. (2024), "Parallelizing Linear Transformers with the Delta Rule"
  and "Gated Delta Networks."
- The **chunk_gla / fla-org** fast-attention codebase has reference
  implementations of both forward and backward in Triton; we are
  porting that math to GLSL/SPIR-V.
- The triangular-solve-with-its-adjoint pattern is standard in linear
  algebra autograd (jax.scipy.linalg.solve_triangular has it).

The Vulkan port is the only mildly novel choice; the kernels are
vanilla compute primitives.
