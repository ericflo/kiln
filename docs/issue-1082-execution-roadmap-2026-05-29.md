# Issue #1082 — verified execution roadmap (2026-05-29)

> **Historical snapshot, not current operating guidance.** This document records
> migration state from May 2026. The `KILN_USE_TAPE_*` and
> `KILN_USE_TAPE_AUTHORITATIVE` switches mentioned below were removed without
> aliases or replacement fields. Current GPU training uses an internal tape
> scope as its sole routing authority. See [Configuration](./CONFIGURATION.md)
> and [Native SFT Profile](./NATIVE_SFT_PROFILE.md) for current behavior.

Output of an 8-track parallel map (Sonnet) + adversarial verify (Opus) +
synthesis (Opus) workflow. Supersedes the prioritization in the
2026-05-27 tier roadmap. **The single most important correction below is
#1 in "Biggest risks": most of what gets called "candle removal" is
actually candle-*bridging* tape work and drops zero candle deps.**

## State (landed on main, H100-validated, all gated default-off)
- CP-1: `kiln-tensor` `cuda` feature is candle-free (`e67efa4d`).
- GQA flash tape coverage: `FlashAttnBackward` + `try_tape_flash_attn_cuda`
  + `try_tape_reshape_cuda`, wired into `gqa_attention_core_prefill` +
  `gqa_attention_pre_o` (`a39ac387`).
- GDN recurrence tape op: `GdnRecurrentBackward` + `try_tape_gdn_recurrent_cuda`
  + structural parity test — **NOT yet wired into the production forward**
  (`5df53d2e`).

## Next 3 increments
1. **[pod-cuda] Wire `try_tape_gdn_recurrent_cuda` into the production GDN
   dispatch — surfacing the real output layout.** Highest leverage: first
   link of the CP-4 tape chain; CP-4-default-on gates everything downstream.
   **CORRECTNESS BUG the verifier caught (would ship silently):** the
   adapter returns `Result<Option<Tensor>>` with no layout flag, but
   `gdn_recurrent_forward_from_parts` returns head-LAST `[B,T,nv,dv]` on the
   CUDA prefill/full-chunk paths and head-FIRST `[B,nv,T,dv]` only on the
   chunkwise fallback — naive `(attn_out,false,false)` double-transposes and
   corrupts GDN output, and the existing parity test (head-first only) won't
   catch it. Fix: (1) adapter returns `Result<Option<(Tensor,bool)>>`
   surfacing `head_last`; (2) insert the guard AFTER the head-first
   transpose block (~forward.rs:16037), not at 16019; (3)
   `recurrent_result = (attn_out, head_last, false)` and **fall through**
   (early-return bypasses gated_rms_norm/reshape/out_proj); (4) first assert
   training reaches the else block (not short-circuited by
   `native_recurrent_result`). Build: `cargo clean -p kiln-gdn-kernel &&
   SCCACHE_RECACHE=1`. Add a head-LAST-producing CUDA test (seq_len>1).
2. **[pod-cuda] Harden the CP-4 parity gate.**
   `tape_authoritative_grads_match_candle_baseline` reports `compared=0` and
   only `eprintln`s it — not a gate. Add `set_var("KILN_USE_TAPE_LORA_ADD","1")`
   in the existing `unsafe` block (trainer.rs:13617-13619; the LoRA-add
   adapter is wired at forward.rs:2466) + `assert!(compared>0)`. Then the
   100-step loss+Adam-moment parity test (DoD gate). Run via `cargo nextest
   -p kiln-train --features cuda` (OnceLock sub-gates need per-process
   isolation). Default-on flip = trainer.rs:10601 `unwrap_or(false)->true`,
   held until the gate passes. **This is parity-gate work, NOT a candle drop.**
3. **[ci-vulkan, PARALLEL] `kiln-model/backend/vulkan.rs` `candle_core::TensorId
   -> kiln_tensor_id::TensorId`.** Only substantive surface-reduction needing
   zero pod time + zero CP-4 dependency. Use the backend/cuda.rs:22 `kt_id()`
   helper pattern; wrap ALL `.id()`/insert/get sites (verifier: the raw list
   was incomplete — incl. 1823/1843/1855/1859/2374/3667/3852/4722 + nested
   `replace()` 3139/3148). **Honest: reduces named-type surface, does NOT
   drop candle (kiln-model candle dep is unconditional).**

## Parallelism
- **SERIAL on the single CUDA pod** (one at a time, release): GDN wiring →
  GDN surrounding adapters (gated_rms_norm/conv1d/l2-qk-norm) → SDPA-fallback
  → parity-gate hardening + 100-step → default-on flip.
- **PARALLEL via CI** (no GPU): CP-6 vulkan TensorId swap; CP-2 metal sdpa
  (ONLY after authoring the missing candle→kt Metal-buffer bridge); the
  missing kt kernel siblings for the final Cargo drops (rmsnorm
  `matmul_f32_bf16w_kt`/`_bwd_lhs_kt`/`lora_add_bf16_kt`, flce
  `fused_linear_cross_entropy_dispatch_kt`); flce kt-index cross-device gap.
- **HARD-GATED on CP-4 default-on**: kernel-crate Cargo drops
  (rmsnorm/opd/flce), kiln-model Tier-3 (model_forward kt-return, GpuWeights
  kt, ~403 cuda copy/borrow sites — the gating work), kiln-train cd_types
  facade (~16k trainer.rs sites), Tier-4 kt-bridge delete, Tier-5 vendor delete.

## Biggest risks (verified)
1. **"candle removal" mislabeling.** candle-core is UNCONDITIONAL in
   kiln-model + kiln-train + the 3 kernels. The vendor-delete gating crate is
   **kiln-model**, not kiln-kt-bridge (~6 production consumers remain). Early
   CP-4 tracks are tape-chain/parity work, not candle drops.
2. **GDN wiring layout bug** (see increment 1) — #1 correctness risk.
3. **Edition 2024**: every `std::env::set_var` needs `unsafe{}`.
4. **OnceLock sub-gate caching**: `KILN_USE_TAPE_{LORA_ADD,FLASH_ATTN,GDN}`
   are cached; tests MUST run under `cargo nextest` (per-process) or the
   first read wins and parity tests silently no-op.
5. **sccache CUDA dlink stale cache**: `cargo clean -p kiln-gdn-kernel &&
   SCCACHE_RECACHE=1` before every fresh GDN-touching pod build.
6. **CI-parallel substrate gaps**: CP-2 metal assumes a candle→kt Metal
   bridge that doesn't exist; SDPA-fallback's kt borrow is CUDA-only
   (validation is pod-cuda, not ci); the final kernel-drop track assumes
   kt-typed siblings that don't exist. Build substrate before scheduling.
7. **kt substrate not thread-safe** under concurrent GPU ops (Phase-1
   stream-pool gap) — gate GPU tests structurally; single-threaded prod is OK.

## Full ordered plan
See the workflow result (18-step `ordered_full_plan`) — CUDA-serial CP-4
chain first (gates everything), CI-parallel surface-reduction + substrate
concurrently, then the gated candle drops (kernels → Tier-3 kiln-model →
kiln-train → kiln-server → kt-bridge → vendor delete) in dependency order.

For #1082.
