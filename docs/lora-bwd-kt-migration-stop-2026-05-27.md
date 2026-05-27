# LoRA `CustomOp::bwd` kt-bridge migration — STOP (re-audit 2026-05-27)

## TL;DR

Re-audit on resume task (2026-05-27, branch `main` at `095f1c74`)
confirms the prior STOP recorded by `cca536cb` ("docs: STOP on three
LoRA CustomOp::bwd kt-bridge migrations (#1082)", same day, earlier
branch `ce/lora-bwd-kt-bridge-1082`). Re-audit was not a copy/paste of
the prior decision — the resume task framed the situation as "these
bwd bodies are candle composites of `matmul` + `scale`/`mul_scalar`,
NOT single FFI kernel calls. So they CAN be migrated using the same
in-place rewrite pattern that was used for `RmsNormCustomOp::bwd`"
and explicitly asked the agent to verify or override the prior STOP.

The verification finds: **the prior STOP is technically correct**, and
the resume task's premise conflates two distinct migration patterns
that live in the codebase today:

| Pattern | Used by | Mechanics | Default | Kill-switch shape |
| --- | --- | --- | --- | --- |
| **kt-bridge migration** | `RmsNormCustomOp::bwd` (`341da876`), `CudaRotaryOneBf16::bwd` (`d99a15a3`), `OpdLossCustomOp::bwd` (`0c1be227`), `FlceCustomOp::bwd` (`ab2da23f`) | Replace ONE FFI kernel call with the same FFI symbol via `kt_tensor_from_candle_cuda_borrow` → `*_kt` wrapper → `kt_tensor_to_candle_cuda_copy` | **ON** | `KILN_DISABLE_*_BWD_KT_BRIDGE=1` falls back to the candle FFI body |
| **Phase-7 per-step migration** | `try_kt_mul_scalar`, `try_kt_cat_dim0`, `try_kt_matmul` (gated at `matmul_no_broadcast_copy`), `try_kt_sum_axis`, … | Route ONE generic candle op step through `kiln_tensor::cuda_*` | **OFF** | `KILN_USE_KT_API_*=1` opts in |

The bridge pattern's bit-exactness is by **construction** — same FFI
symbol => same kernel => same bits. That property doesn't exist for
multi-step composites where the wrapping is the only thing that
changes; the underlying ops (cublasLt matmul, scale, dtype cast) are
the same kernels regardless of the Rust wrapper around them.

## What the resume task proposed

> "replace the candle ops with `kt_matmul` + `kt_mul_scalar`
> (or `kt_scale`) via the kt-bridge borrow/copy helpers"
> "the underlying CUDA matmul + scale FFI symbols are the same —
> just different glue"
> "Add kill-switch env vars … `KILN_DISABLE_LORA_*_BWD_KT_BRIDGE`"

The kill-switch shape (`KILN_DISABLE_*` = default ON) is the
*bridge* pattern. But the migration mechanics (hand-composing
generic ops via `try_kt_*` helpers) is the *Phase-7* pattern. Those
two patterns don't mix:

1. **Hand-composing generic kt ops is NOT bit-exact by
   construction.** Candle's `Tensor::matmul` and `kiln_tensor::cuda_matmul`
   both dispatch through cublasLt, but through *different*
   `CublasLtMatmulHandle` instances with *different* algo caches.
   Cublas may pick different heuristics on first call. Empirically
   the difference is small (≤1 BF16 ULP at our magnitudes) but the
   parity claim is no longer "bit-exact by FFI-symbol identity" —
   it's "close enough by tolerance".
2. **Defaulting an opt-out kill-switch ON for a non-bit-exact
   migration is a reversibility regression.** If training drifts a
   ULP on day 5, you can't reproduce the candle baseline without
   setting the kill switch — which is fine, but it's the *opt-in*
   shape (`KILN_USE_KT_API_*=1` is "ask for the new behaviour"),
   not the *opt-out* shape (`KILN_DISABLE_*=1` is "I want the old
   behaviour back").

## What the per-step Phase-7 gates already do

Greppable evidence in `crates/kiln-model/src/forward.rs` as of
`main@095f1c74`:

- `try_kt_mul_scalar(&grad_hidden_pre, self.scale as f64)` —
  `CudaLoraLinearBf16::bwd` line 3081
- `try_kt_mul_scalar(&grad_b_tile_pre, self.scale as f64)` —
  `CudaLoraLinearBf16::bwd` line 3122
- `try_kt_mul_scalar(&grad_hidden_tile_pre, self.scale as f64)` —
  `CudaLoraAddBf16::bwd` line 3284
- `try_kt_mul_scalar(&grad_b_tile_pre, self.scale as f64)` —
  `CudaLoraAddBf16::bwd` line 3303
- `try_kt_cat_dim0(&pieces)` — `CudaLoraAddBf16::bwd` line 3338

These wire each scale and axis-0 concat step into the Phase-7
opt-in env gate `KILN_USE_KT_API_MUL_SCALAR` / `…_CAT_DIM0` /
`…_ALL`. Per `git log --oneline -- crates/kiln-model/src/forward.rs`,
several Phase-7 gates have already had their defaults flipped ON
(`fec54f8b` flipped `KILN_USE_KT_API_LORA_ADD` ON, `57fbf5e0`
flipped the four `CAT_*` defaults ON, etc.), so each of those
in-place migrations is already at the closest-to-status-quo for
that op.

## The one remaining gap (and why it's still not a bridge job)

The `.matmul()` calls inside the three LoRA bwd bodies are *not*
yet routed through `try_kt_matmul`:

- `CudaLoraAddF32::bwd` line 2887: `grad_delta.matmul(&b_f32)?`
- `CudaLoraAddF32::bwd` line 2888: `…matmul(&hidden_f32)?`
- `CudaLoraLinearBf16::bwd` line 3069: `grad_y_tile_bf16.matmul(&weight_t_t)?`
- `CudaLoraLinearBf16::bwd` line 3077: `grad_y_tile_f32.matmul(&b_f32)?`
- `CudaLoraLinearBf16::bwd` line 3096: `grad_hidden.matmul(&a_f32)?`
- `CudaLoraLinearBf16::bwd` line 3113: `x_tile.matmul(&a_t_bf16)?`
- `CudaLoraLinearBf16::bwd` line 3118: `grad_y_tile_f32.t()?.matmul(&hidden)?`
- `CudaLoraLinearBf16::bwd` line 3142: `grad_hidden.t()?.matmul(&x_tile_f32)?`
- `CudaLoraAddBf16::bwd` line 3280: `grad_y_tile_f32.matmul(&b_f32)?`
- `CudaLoraAddBf16::bwd` line 3299: `grad_y_tile_f32.t()?.matmul(&hidden_tile_f32)?`

A real next step, distinct from the "bridge" framing, is to wrap
each of these in a `try_kt_matmul` fallthrough exactly the way the
adjacent `try_kt_mul_scalar` calls already do. That's a meaningful
Phase-7 expansion. But its kill switch should be:

- `KILN_USE_KT_API_MATMUL=1` (already exists — see
  `cuda_use_kt_api_matmul()` at `forward.rs:1424`), or
- A new sibling like `KILN_USE_KT_API_LORA_BWD_MATMUL=1` if a
  separate gate is desired for staged rollout.

NOT `KILN_DISABLE_LORA_*_BWD_KT_BRIDGE`. The latter implies a
default-on, single-FFI-symbol, bit-exact bridge — none of which is
true here.

This Phase-7 matmul expansion is also out of scope for the resume
task as stated (the resume task framed itself as a bridge
migration; if it had been framed as a Phase-7 matmul expansion the
kill-switch direction would have been the opposite, and the
parity test shape would have been `KILN_USE_KT_API_*=1` vs default
off, not `KILN_DISABLE_*=1` vs default on). Treating it as a
follow-up rather than a re-spec of this task keeps the planning
loop honest about which kind of work is being shipped.

## Revisit conditions

The bridge migration becomes mechanical (and worth doing) when
**either** of these lands:

1. **Brand-new fused backward FFI kernels** in
   `kiln-rmsnorm-kernel/csrc/`, exposing:
   - `kiln_lora_add_bwd_f32` — fused `grad_delta = grad_y * scale;
     grad_hidden = grad_delta @ b; grad_b = grad_delta^T @ hidden`
     in one launch
   - `kiln_lora_linear_bwd_bf16` — fused tile-looped
     `grad_x = grad_y @ W^T + (grad_y @ B * s) @ A; grad_a = (grad_y @ B * s)^T @ X; grad_b = grad_y^T @ (X @ A^T) * s`
     in one launch (or one launch per tile)
   - `kiln_lora_add_bwd_bf16` — fused tile-looped
     `grad_hidden_i = grad_y_i @ B * s; grad_b += grad_y_i^T @ hidden_i * s; grad_hidden = cat(...)`
     in one launch (or one launch per tile)

   With those, the migration template becomes purely mechanical
   (borrow 3 inputs, call fused `*_bwd_kt`, copy back 2 or 3
   gradients, add `KILN_DISABLE_LORA_*_BWD_KT_BRIDGE` env opt-out,
   and a parity test per op).

2. **kt-side autograd `Tape` + `BackwardOp` adoption in production**
   (the "(b)" path in `CANDLE_REMOVAL_PLAN.md` §"kt-autograd
   readiness"). At that point the whole training loop swaps from
   `loss.backward()` to `tape.backward(loss_id, ...)`, candle
   `CustomOp{1,2,3}` recording sites are replaced, and the LoRA
   ops just become `BackwardOp` impls — no bridge needed.

## What changed vs the 2026-05-27 STOP

Nothing structurally — the two FFI-grep results are identical:

```
$ grep -rn "kiln_lora.*bwd\|lora.*_bwd_kt\|lora.*_backward_kt" crates/
crates/kiln-model/src/forward.rs:2522:fn cuda_lora_bwd_tile_rows() -> usize {
crates/kiln-model/src/forward.rs:3046:        let tile_rows = cuda_lora_bwd_tile_rows().min(rows.max(1));
crates/kiln-model/src/forward.rs:3262:        let tile_rows = cuda_lora_bwd_tile_rows().min(rows.max(1));
crates/kiln-model/src/forward.rs:3587:        let tile_rows = cuda_lora_bwd_tile_rows().min(rows.max(1));
```

All four matches are the local Rust `cuda_lora_bwd_tile_rows` helper
that controls the tile-loop chunk size — none are FFI symbol names.

The additional value of this audit:

- **Explicit articulation** of why the kill-switch direction
  matters: default-on opt-out vs default-off opt-in encodes
  whether the migration is bit-exact-by-FFI-identity (bridge) or
  approximate-via-different-handles (Phase-7).
- **Names the concrete next-step migration** that the prior STOP
  alluded to but didn't enumerate: route the eight remaining
  `.matmul()` calls in the three LoRA bwd bodies through
  `try_kt_matmul`, gated on `KILN_USE_KT_API_MATMUL` (already
  exists) or a sibling gate. That work is in scope for a future
  Phase-7 task framed correctly; it is NOT in scope for the
  bridge-migration task as framed.

## Pod usage

No pod acquired. The re-audit was static / grep-based — the
question being answered ("does the kt-bridge FFI substrate exist for
LoRA bwd, and was the prior STOP technically correct?") doesn't
require a compile-and-test loop. Skipping the pod saves the ~$1–2 +
10–15 min cold-cache cost that the bridge-migration task would have
otherwise burned to land at the same conclusion.

For #1082.
