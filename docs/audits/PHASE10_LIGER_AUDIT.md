# Phase 10 — Liger Kernel Port Audit

**Status:** Doc-only audit. $0 RunPod spend. No code changes.
**Scope:** Filter the 6 candidate Liger-Kernel ports against kiln's actual training hot path on Qwen3.5-4B (V=248320, hidden=2560, intermediate=9216, layers=32, head_dim=256, partial_rotary=0.25).

## TL;DR Recommendation

**Port FLCE (Fused Linear Cross-Entropy) first.** Math ceiling: ~8 GB peak VRAM saved at training seq_len 16384, ~32 GB at seq_len 65536. Effort: ~500–700 LOC, single new crate (`kiln-flce-kernel`), pattern after `kiln-flash-attn` (PR #33) for forward + backward via candle `CustomOp1`.

Kill list (do not port):
- **Fused RMSNorm** — duplicates PR #133 `kiln-rmsnorm-kernel`.
- **Fused LayerNorm** — Qwen3.5-4B has no LayerNorm layers.
- **Fused RoPE** — applies in 8/32 layers with partial_rotary=0.25; <5% of training step.
- **FleCE** — subsumed by FLCE; FLCE already chunks along the vocab dim.

Defer:
- **Fused SwiGLU** — gradient-checkpointing (PR #36) recomputes the SwiGLU intermediate during backward, so the ~74 MB/token intermediate is not the persistent peak. Re-evaluate after FLCE lands and re-profile.

## Kiln Training Hot-Path Audit

### 1. Final logits + cross-entropy (`crates/kiln-train/src/trainer.rs:919`)

`cross_entropy_loss(logits, …)` receives the **full** `[1, T, V]` bf16 logits tensor produced by `model_forward_head` (`crates/kiln-model/src/forward.rs:2751`) via `hidden.broadcast_matmul(&weights.embed_tokens_t)`. The function then `narrow`s and `index_select`s active positions before promoting to F32, but the full tensor is already materialized and **retained for backward** (it is the input to the autograd `log_sum_exp` node).

Peak bf16 logits memory at seq_len `T`:

| T | bf16 logits | bf16 + F32 cast retained for backward |
| --- | --- | --- |
| 2048 | 1.02 GB | ~3.0 GB |
| 8192 | 4.07 GB | ~12.2 GB |
| 16384 | 8.14 GB | ~24.4 GB |
| 65536 | 32.55 GB | ~97.6 GB (OOM on A6000 well before this) |

Per token: `V × 2 = 248,320 × 2 = 497 KB` bf16, ~1.49 MB once the F32 cast and one gradient buffer are in scope. With `KILN_NO_GRAD_CHECKPOINT=1` the per-layer activations dwarf this; with checkpointing on (default), the cross-entropy logits become the single largest persistent tensor in the training step.

The same pattern repeats for `token_log_probs` at `trainer.rs:787` (GRPO log-probability evaluation), so a fused linear-CE port benefits both SFT and GRPO.

### 2. RMSNorm (`crates/kiln-model/src/forward.rs:670`)

`rms_norm()` already dispatches to `kiln_rmsnorm_kernel::fused_rmsnorm` on CUDA + bf16 (PR #133). However, the fused path is **forward-only**: `kiln-rmsnorm-kernel/src/lib.rs:148` returns a fresh `Tensor::zeros(...)` allocation outside the candle autograd tape, with the explicit out-of-scope note "backward pass, fused GEMM prologue, non-bf16 dtypes" at line 48. Training therefore goes through `rms_norm_fallback` (~11 candle ops, all autograd-safe). A Liger-style fused RMSNorm port would either:

- duplicate the existing forward kernel, or
- add a backward kernel + a candle `CustomOp1` wrapper to make both differentiable.

The latter is non-trivial (parity tests against the candle fallback, gradient correctness on bf16 inputs, a new kill-switch). With CUDA graphs on, the per-step dispatch savings are amortized away — see `kiln-cuda-graphs-amortize-fusion`. Skip unless a re-profile at long-context training shows the RMSNorm fallback as a top-3 region.

### 3. RoPE (`crates/kiln-model/src/forward.rs:711, apply_rope:772`)

Partial rotary with `partial_rotary_factor=0.25` (rotary_dim=64 of head_dim=256), applied **only in the 8 full-attn layers** (24/32 layers are GDN linear-attn that bypass rotary). Per training step: 8 × short rotary slice in F32. Static estimate: <5% of training time, ceiling for any Liger-style fused RoPE port is ~1.02× end-to-end. Below the 1.05× floor.

### 4. SwiGLU (`crates/kiln-model/src/forward.rs:822`, `swiglu_ffn`)

Computes `down_proj @ (silu(gate) * up)` with `intermediate_size=9216`. Peak intermediate: `9216 × 2 = 18 KB/token` bf16 (silu output) + `18 KB/token` (up) + `18 KB/token` (gate) ≈ **54 KB/token** in flight, of which only the `silu(gate) * up` product is retained for backward by candle.

Gradient checkpointing (PR #36, `model_forward_segment` path used by training) recomputes layer activations during backward, so the SwiGLU intermediate is **not persistent** across the step — it lives only for one forward sub-segment at a time. Memory ceiling for a fused SwiGLU port: ~50–100 MB peak shaving at T=16384, vs. ~8 GB for FLCE. Math ceiling does not justify the porting cost while gradient checkpointing is on by default.

### 5. CrossEntropy without linear fusion (vanilla CE)

A plain fused softmax-CE (without the lm_head matmul fused in) gives a smaller win: it avoids one full-V softmax materialization but still requires the `[1, T, V]` logits tensor as input. **FLCE strictly dominates** because it never materializes that tensor in the first place — it streams chunks of `hidden @ weight.T` directly into the loss accumulator. Pick FLCE.

### 6. LayerNorm

Qwen3.5-4B uses RMSNorm everywhere. There are zero LayerNorm layers in `crates/kiln-model/src/forward.rs`. **Not applicable.**

## Per-Candidate Decision Matrix

| # | Candidate | Decision | Math ceiling | Notes |
| --- | --- | --- | --- | --- |
| 1 | **FLCE (Fused Linear CE)** | **GREEN — port first** | ~8 GB at T=16384, ~32 GB at T=65536 | Single largest peak-VRAM win on Qwen3.5-4B's V=248320 vocab. Benefits both SFT (`cross_entropy_loss`) and GRPO (`token_log_probs`). |
| 2 | Fused RMSNorm | KILL | n/a | Duplicates PR #133 forward; backward port requires new CustomOp + parity work; CUDA graphs amortize launch savings. |
| 3 | Fused RoPE | KILL | <1.02× end-to-end | Partial rotary, 8/32 layers. Below 1.05× floor. |
| 4 | Fused SwiGLU | DEFER | ~50–100 MB at T=16384 | Gradient checkpointing already amortizes the intermediate. Re-visit after FLCE lands and re-profile. |
| 5 | Fused LayerNorm | KILL | n/a | Architecture has no LayerNorm. |
| 6 | FleCE (chunked CE) | SUBSUMED | n/a | FLCE already chunks along vocab dim; FleCE adds nothing on top. |

## Top Recommendation: Port FLCE as `kiln-flce-kernel`

**Pattern:** new workspace crate `crates/kiln-flce-kernel/`, mirror the layout of `kiln-flash-attn` (PR #33) which exposes both `flash_attn_fwd` and `flash_attn_bwd` and is wired into candle autograd via `CustomOp1`.

**Inputs:** `hidden_states: [1, T, hidden]` bf16, `lm_head_weight: [V, hidden]` bf16, `target_ids: [T]` u32, `label_mask: [T]` bool, `chunk_size: usize` (e.g. 8192 or 16384 tokens of vocab).

**Algorithm (forward):** for each chunk of vocab `[v_start, v_end)`:
1. Compute `chunk_logits = hidden @ weight[v_start:v_end].T` in bf16 directly into a `[T, chunk]` scratch buffer.
2. Track running max and `sum exp(logits - max)` in F32 for stable log-sum-exp.
3. If the target token falls in this chunk, accumulate `-logits[t, target[t] - v_start]` into the per-token loss in F32.
4. Discard the chunk_logits scratch after each step — it never persists.

**Algorithm (backward):** identical chunked pass, but compute `softmax_chunk - one_hot(target_in_chunk)` and accumulate `grad_hidden += softmax_chunk @ weight[v_start:v_end]` and `grad_weight[v_start:v_end] += softmax_chunk.T @ hidden`. F32 accumulation; bf16 outputs.

**Files (estimate):**

| File | LOC est. | Purpose |
| --- | --- | --- |
| `crates/kiln-flce-kernel/Cargo.toml` | 30 | new crate manifest, gated on `cuda` feature |
| `crates/kiln-flce-kernel/src/lib.rs` | 300 | candle `CustomOp1` wrapper, fwd/bwd dispatch, parity fallback |
| `crates/kiln-flce-kernel/src/kernels/flce_fwd.cu` | 200 | chunked forward CUDA kernel (bf16 in, F32 accum) |
| `crates/kiln-flce-kernel/src/kernels/flce_bwd.cu` | 200 | chunked backward CUDA kernel |
| `crates/kiln-train/src/trainer.rs` | ~30 changed | route `cross_entropy_loss` and `token_log_probs` through FLCE behind `KILN_DISABLE_FLCE` kill-switch |
| `crates/kiln-train/Cargo.toml` | 1 | dep on `kiln-flce-kernel` |

Total: ~700 LOC + ~30 LOC of trainer changes. Single new crate, two trainer call-sites updated, one kill-switch added (precedent: PR #92, #133, #158, #166).

**Kill-switch:** `KILN_DISABLE_FLCE=1` falls back to current `cross_entropy_loss` path for parity testing.

## Next-Step Gating Signal

Before opening the FLCE port PR, run a short SFT bench on A6000 at three seq_lens to confirm the static memory math survives gradient checkpointing in practice:

| Run | Settings | What to measure |
| --- | --- | --- |
| A | T=2048, batch=1, KILN_W4A16=1 | nvidia-smi peak VRAM, decode tok/s baseline |
| B | T=8192, batch=1 | peak VRAM (expect ~+12 GB over A) |
| C | T=16384, batch=1 | peak VRAM (expect ~+24 GB over A) |

**Greenlight FLCE port if:** at T=16384, the logits tensor + its F32 cast + retained backward grad together account for ≥30% of peak VRAM (cross-check via nsys `cuMemAlloc` attribution to `model_forward_head` vs. `cross_entropy_loss`).

**$ ceiling for the bench preflight:** $20 (~1 pod-hour on a single A6000, on-demand, no spot — see `runpod-always-on-demand`).
**$ ceiling for the FLCE port PR (bench + parity + 3× median bench):** $40–60 (~2 pod-hours). Use the `ce kiln-pod-acquire` / `ce kiln-pod-release` pool.

## What This Audit Explicitly Does NOT Recommend

- **Do not port any Liger kernel speculatively** — every candidate other than FLCE is killed, deferred, or subsumed by existing kiln work.
- **Do not vendor Liger's source** — port only the kernel ideas (chunked vocab pass, F32 accumulation, fused linear+softmax+CE) into a kiln-native crate. Liger is Apache-2 but the dependency surface (Triton, torch.autograd) does not compose with candle.
- **Do not port FLCE without first running the gating bench** — Phase 6 history shows multiple null-result fusion PRs ($14.99 burn on PR #176, $13.76 incident on bench supervision) when math ceilings were not pre-verified.

## References

- Liger-Kernel: https://github.com/linkedin/Liger-Kernel
- FLCE paper-equivalent description: Liger-Kernel `src/liger_kernel/ops/fused_linear_cross_entropy.py`
- Kiln cross-entropy: `crates/kiln-train/src/trainer.rs:919`
- Kiln lm_head: `crates/kiln-model/src/forward.rs:2751`
- Existing kernel-with-backward template: `crates/kiln-flash-attn/src/lib.rs` (PR #33)
- Existing forward-only kernel template: `crates/kiln-rmsnorm-kernel/src/lib.rs` (PR #133)
- Gradient checkpointing precedent: PR #36 (`KILN_NO_GRAD_CHECKPOINT`)
- Kill-switch precedent: PRs #92, #133, #158, #166
