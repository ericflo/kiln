# Candle removal status — 2026-05-28 (pm snapshot)

Quick-reference dashboard so a fresh agent can read current state without
scanning 110 KB of issue #1082 body + 7 STOP docs. Companion to
[`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md) — the plan is the
authoritative narrative; this doc is a one-screen snapshot.

Last refresh: `014d1b52` (post wave-15 + opd-loss retry from wave-16).

## Per-crate production candle-core state

| Crate | `[dependencies]` candle? | Blocker |
|---|---|---|
| kiln-autograd | No | — |
| kiln-blas | No | — |
| kiln-core | No | — |
| kiln-eval | No | — |
| kiln-flash-attn | No | — |
| kiln-flce-kernel | **Yes** | CP-4 substrate (`Tape::backward` → kt-typed Var/LoRA params); see [`kiln-flce-kernel-candle-drop-stop-2026-05-28.md`](./kiln-flce-kernel-candle-drop-stop-2026-05-28.md) |
| kiln-gdn-kernel | No | — |
| kiln-kt-bridge | **Yes (by-design)** | Tier-5 deletion target |
| kiln-marlin-gemm | No | — |
| kiln-model | **Yes** | metal_types chokepoint + CP-4 |
| kiln-mps | No | — |
| kiln-nvtx | No | — |
| kiln-opd-loss-kernel | **Yes** | CP-4 — see [`opd-loss-kernel-candle-removal-stop-2026-05-28.md`](./opd-loss-kernel-candle-removal-stop-2026-05-28.md) |
| kiln-rmsnorm-kernel | **Yes** | CP-4 — see [`kiln-rmsnorm-kernel-candle-drop-stop-2026-05-28.md`](./kiln-rmsnorm-kernel-candle-drop-stop-2026-05-28.md) |
| kiln-scheduler | No | — |
| kiln-server | dev-only ✅ | — |
| kiln-tensor | **Yes** | metal_types re-exports `candle_metal_kernels::*` ([`metal-cargo-toml-candle-drop-stop-2026-05-28.md`](./metal-cargo-toml-candle-drop-stop-2026-05-28.md)) |
| kiln-train | **Yes** (candle-nn ✅ dev-deps now) | One production ref: `impl candle_core::CustomOp1 for InjectTensorGradient` in trainer.rs. CP-4 caller flip 🛑 **blocked on substrate** — see [`inject-grad-flip-blocked-2026-05-28.md`](./inject-grad-flip-blocked-2026-05-28.md) |
| kiln-vulkan-kernel | dev-only ✅ | — |

**Production candle-core deps remaining: 7 crates.** (kiln-train was 8th
until `94ce67da` moved candle-nn to dev-deps and confirmed candle-core
sits behind a single `CustomOp1` impl.)

## Critical-path summary

All 5 candle-blocked kernel/train crates fail-back to one architectural
checkpoint:

**CP-4 — Tape::backward → kt-typed production training step.**

Bridge primitives are in place:

- `kiln_autograd::Tape::backward` + `backward_with_seeds` (`1680bebf`)
- `kiln_autograd::tape_scope::with_thread_local_tape` (b2702ce0, public-mod
  fix `3a1f8d02`)
- `kiln_kt_bridge::tape_bridge::with_tape_scope_emit_to_grad_store`
  (`bf248d4d` + import fix `82512751`) — opens both a tape scope and an
  IO-mapping scope so kt grads accumulate into a candle GradStore via the
  candle-typed adapter return values

`KILN_USE_TAPE_FORWARD` adapters land kt-tape recording inside production
forward (`rms_norm`, `matmul`, `silu`, `embedding`, **`swiglu`** — all 5
register IO mappings into the tape bridge via `cf138c9c` + `57f7b678`).

`InjectGradientBackward` kt-tape substrate (`9b2eda8e`) provides the
candle-free replacement for `kiln-train::trainer::InjectTensorGradient`:

- `kiln_autograd::backwards::inject_gradient::InjectGradientBackward` —
  BackwardOp that emits a precomputed `injected` tensor regardless of
  `grad_output`; 6 unit tests cover the contract.
- `kiln_kt_bridge::tape_bridge::inject_gradient_kt(arg, upstream)` —
  candle-typed adapter that records the kt op + registers IO mapping +
  returns a candle scalar zero (matches the existing `apply_op1`
  callsite contract).

What's left for CP-4 closeout (`docs/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`):

1. ✅ Wrap `trainer::sft_train` step root in `with_tape_scope_emit_to_grad_store` — **landed in `675e0dea`** (gated on `KILN_USE_TAPE_FORWARD=1`).
2. 🛑 **BLOCKED on substrate** — see [`docs/inject-grad-flip-blocked-2026-05-28.md`](./inject-grad-flip-blocked-2026-05-28.md).
   - There are **six** call sites, not two: `trainer.rs:8068, 8220, 8366,
     8376, 8446, 8454` (all inside
     `full_attention_single_layer_tiled_mlp_reverse`).
   - Substrate `9b2eda8e` is bit-equivalent to the candle path **only
     when arg IS a Var** (parity test exercises arg=Var; production
     sites pass arg=intermediate). The shim's `bwd` returns
     `zeros_like(arg)`; candle's backward walks those zeros through
     upstream candle ops to the actual leaf Vars BEFORE the bridge's
     post-hoc `insert_or_add_by_raw` runs. The walk result is wrong
     for every production query (`grads.get(pre_o_tile_var)`,
     `grads.get(seg_input_var)`, etc.).
   - Unblocking path (recommended): move the candle `CustomOp1` impl
     from `kiln-train::trainer::InjectTensorGradient` to
     `kiln-kt-bridge::tape_bridge` (Option 2 in the STOP doc), keeping
     the bwd contract intact and adding kt-tape recording as a side
     channel.
3. ⏳ Delete `InjectTensorGradient` from trainer.rs once no callers remain.
4. ⏳ Repeat for the GRPO loop (`trainer.rs:13586`).

## Two parallel architectural pieces (independent of CP-4)

**metal_types chokepoint swap** — `crates/kiln-tensor/src/metal_types.rs`
currently `pub use candle_metal_kernels::metal::{...}` for `ComputePipeline`,
`Library`, `Buffer`, `BufferOffset`, plus `candle_core::metal_backend::{...}`
for `MetalDevice` / `DeviceId` / `Storage`. The `Raw*` objc2-metal
parallel aliases land alongside them (`56bdaffd`), but production code
still imports the candle-typed names. Swap is mechanical-but-large
(`crates/kiln-model/src/backend/metal.rs` is the dominant consumer; 5→1
consolidated import line as of `80235181`).

Substrate-add commits since the swap plan shipped:

- `buffer_o_kt` kt-typed sibling of `buffer_o` (`e82c3017`) — Step 1.
- `metal_sdpa_last_axis` kt-native fused SDPA op (`0e50ee14`) — Step 2,
  the single substrate gap the plan called out. Mirrors the existing
  `metal_softmax_last_axis` pattern.

After these two, every remaining step is a pure caller migration in
`kiln-model::backend::metal` (232 `buffer_o` sites + 15 `sdpa` sites +
the chokepoint re-export flips).

**kiln-kt-bridge deletion (Tier-5 endgame)** — by-design candle user.
Owns the `KtForwardOp{1,2,3}` candle `CustomOp1` shim that lets kt
kernels splice into candle autograd. Lives only as long as CP-4 hasn't
fully replaced candle autograd. Delete after kiln-train's
`InjectTensorGradient` and the kernel-crate `kt_forward_op.rs` shims are
all gone.

## How to use this dashboard

- Before queuing a "drop candle from crate X" task, check the row above.
  If the blocker column says CP-4, write a STOP doc and queue against
  the CP-4 substrate work instead.
- When CP-4 production wiring lands, sweep this table: any blocker that
  says "CP-4" should flip green within one or two PRs.
- metal_types swap is independent and can move in parallel; it doesn't
  need CP-4.
