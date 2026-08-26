# Review notes — PR1/PR2 build + adversarial review (#1082, branch `feat/vk-tape-harmonization`)

Two reviewers gated the implemented PRs. Both PR1 and PR2 **build green across all
backends and ran on a real Vulkan GPU**. Summary of verdicts and the corrections
applied to this branch.

## Verdicts
- **Build-regression + parity gate → `ship`.** `cargo check` passes for
  `kiln-tensor`/`kiln-train` under both `--features vulkan` and default (no
  candle/CUDA/Metal path broken). On-device tests ran against a live Vulkan GPU:
  PR2 `from_vec_on` F32 round-trip (0.03s) and PR1 resident AdamW (0.02s) both
  pass. **Parity = numerically identical, not similar:** exactly one F32 AdamW
  kernel exists (`dispatch_adamw_step_f32`); the new `dispatch_adamw_step_buffers`
  is a pure 12-arg pass-through; hyperparameters + 1-indexed step are preserved.
- **Adversarial diff review → `fix-then-ship`.** PR2's host↔device copy pair is a
  faithful, line-for-line-correct mirror of the Metal reference (CPU-backed +
  contiguous enforced, exact logical packed byte-length recorded, `Arc<VulkanBuffer>`
  Drop-once sound, flipped tests assert real byte-equality). Findings below.

## Findings and resolution

### [major] op-coverage ownership gap (PR3/PR5 plan gap) — **CORRECTED IN SPECS**
The PR5 §0 thesis "backward just works once the forward records" is **incomplete for
Vulkan**. Backward composites (e.g. `RmsNormKtBackward::apply`, `tape_forward.rs:257`)
call `sum_axis` (reduce.rs), `{mul,add,sub}_scalar` (scalar.rs), and
`log_softmax_last_dim` — **none have a `vulkan_fwd`**, and `dispatch1/dispatch2`
(`device_op.rs:170-192`) **host-fallback only for Metal**. On Vulkan an `Ok(None)`
op falls to `cpu_fwd` on Vulkan storage and **hard-errors**. Metal's backward "just
worked" *partly because of* a Metal-only host fallback Vulkan deliberately lacks.
No PR in the series owns these op-ports; PR3 is matmul-only.

**Resolution (recommended):**
- **(B) first — mirror Metal, unblock immediately:** add the Metal-style Vulkan
  host-fallback in `dispatch1/dispatch2` so `Ok(None)` ops bounce
  D2H→`cpu_fwd`→H2D. Correct-but-slow; makes the PR5 thesis literally true.
- **(A) then — perf:** port the hot ops (`sum_axis`, scalar, `log_softmax_last_dim`,
  likely `scatter_add`/`broadcast`) to `vulkan_fwd`.
- **Ownership:** either expand **PR3** scope or insert a dedicated **PR3.5 — Vulkan
  backward op-coverage** between PR3 and PR4. Correction banners added to
  `PR3-spec.md` and `PR5-spec.md`.

### [minor] PR1 dropped defensive assert — **FIXED (commit `0f0bde95`)**
`vk_optimizer_step_from_grads` had dropped the `state.n_elements == param.num_elements()`
guard the inline AdamW sites carried; restored inside the AdamW arm so stale state
errors on the host instead of indexing OOB in the SPIR-V kernel.

### [minor] PR5 device-gate count 29 → 31 — **FIXED**
Actual `tape_forward.rs` count is 31 (matches PR6 §1). Corrected in `PR5-spec.md`.

### [nit] PR1 commit subject overstates "retire VkAdamWBook" — acknowledged
`VkAdamWBook`/`VkAdamWState` are intentionally **retained** (honestly documented at
`vk_train.rs:633`) to own the on-device m/v buffers until PR2's kt-Tensor-on-Vulkan
storage lets moments move into `OptimizerState.moments`. Diff + doc are honest; only
the subject line is loose. No code impact; left as-is (branch unpushed).

### [nit] PR6 BF16-vs-F32 gate — well-handled in spec, surface in done-when
`base_dtype_supports_tape` (`trainer.rs:7237`) returns true only for a BF16 base, but
Vulkan trains F32 by design. OPD's composite accepts F32 (clean); SFT/GRPO on Vulkan
bail at the dtype check. PR6 already stages this (ship OPD-on-F32-Vulkan; SFT/GRPO
smokes `#[ignore]` pending the dtype decision) — just ensure PR6's "done-when" states
the partial lane coverage rather than implying all three route.

## Other carried caveats (from PR2 self-report, not blockers)
- Round-trip byte-identity tests are compiled and asserted but only **execute** under
  `KILN_TENSOR_VULKAN_TEST` + a present device (CI-safe self-skip).
- `primary_vulkan_device(idx)` keys the cache by ordinal, but `VulkanDevice::new()`
  does its own best-GPU/env-driven physical-device selection (no explicit ordinal
  arg today) — all ordinals currently resolve to the same device. Fine for
  single-GPU; revisit for multi-GPU.
