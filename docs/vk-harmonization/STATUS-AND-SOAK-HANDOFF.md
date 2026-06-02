# Vulkan kt-tape harmonization — status & human-soak handoff (#1082)

Branch `feat/vk-tape-harmonization` (worktree `kiln-vk-harmonize`, **not pushed**).
Everything below was implemented + bounded-validated autonomously; **PR6/PR7 and
the rmsnorm-backward frontier are explicitly held for human GPU soak** (host has
hard-crashed on long GPU runs — bounded/observable validation only).

## What landed (all green, real-GPU-validated on RADV Strix Halo unless noted)

| Commit | Unit | Proof |
| --- | --- | --- |
| `056d7e03` | **PR1** AdamW optimizer seam | numerically identical (same `adamw_step_f32` shader); resident round-trip on GPU |
| `0f0bde95` | PR1 fixup | restored dropped state-size guard |
| `b94feeac` | **PR2** `Device::Vulkan` first-class kt storage | host↔device round-trip byte-exact |
| `9739c715` | **PR3a** Vulkan host-fallback in `dispatch1/2/3` | `sum_axis` on Vulkan bit-exact vs CPU (was a hard-error before) |
| `88573e1e` | **PR3b** zero-copy bridge + `MatmulOp::vulkan_fwd` | matmul `max_abs_err 2.98e-8`; batched correctly declines to fallback |
| `c2acf06d` | **PR3c** hot-op ports (`scale`, `sum/mean_all`) | `max_abs_err 0.0`; `sum_axis`/scalar-bias/`log_softmax` honestly left on fallback (no vk kernel) |
| `99d2d2a8` | **PR4a** generic `VkBwdAdapter` + `family_ported` | cargo R1 feature-unification cleared (no cycle) |
| `c70a1d5f` | **PR4b** adapter grad validation | exact-vs-direct `0.0` (4 families); FD `5–6e-5` |
| `d4e473ab` | **PR4c** validation hardening | dropped unvalidated `matmul_bf16w`; rope/softmax FD oracles `1.3e-5`/`3.1e-5`; zero-copy/dtype/device-guard tests; 12/12 |
| `3af2eb8b` | **PR5a** gate cascade | `tape_forward` + `tape_bridge` module gates + 25 recorder gates widened to admit Vulkan; cuda/metal additive-only |
| `4e4b0ae7` | **PR5b** record-and-backprop proof | `add` recorder on Vulkan: `tape.len()=1`, fwd parity `0.0`, backward `0.0` vs CPU |
| `<this>`  | **PR5c** handoff + test-reason fix | rms_norm gated-recorder forward-record re-verified PASS standalone |

**Net so far:** the entire shared substrate (kt `Tensor` on Vulkan → `kiln_autograd::Tape`
→ device-agnostic backward composites / `VkBwdAdapter` for fused kernels → `kiln_optim` AdamW)
is wired and proven on hardware. No fork code was deleted yet (PR7).

## The PR5b review finding — status

The adversarial reviewer's `[major]`: PR5b's *running* proof used only `try_tape_add_kt`
(a device-agnostic "covered-for-free" recorder), not a PR5a-gated recorder.

**Partially closed (forward direction):** `vk_tape_rms_norm_records_on_vulkan` — a genuinely
PR5a-gated recorder (it returned `Ok(None)` and silently dropped its node before PR5a) — was
re-run standalone on Strix Halo and **passed**: `try_tape_rms_norm_kt` records exactly 1 node on
`Device::Vulkan(0)`. So the gate works for a real gated recorder, not just `add`.

**Open (backward direction) → human soak:** that test does not yet drive `Tape::backward` on
rmsnorm, and the implementing agent observed a **RADV GPUVM write fault + context loss** on the
native rmsnorm path under back-to-back GPU load. Recovering from a context loss needs a human at
the console, so this is the soak frontier, not an autonomous task.

## The frontier to chase during soak (in order)

1. **rmsnorm backward + parity on Vulkan.** Un-`#[ignore]` `vk_tape_rms_norm_records_on_vulkan`,
   extend it to seed `dL/dL` and run `Tape::backward`, compare grads to a `Device::Cpu` oracle
   (tol 1e-3). If the GPUVM fault reproduces, it is almost certainly a buffer-size/binding bug in
   `RmsNormOp::vulkan_fwd` when driven through the kt bridge (suspect: the `weight [hidden]` tensor
   binding, or a workgroup-dispatch sizing using the pool-padded buffer size instead of the logical
   `n·dtype_size`). The PR3b bridge already records the *logical* byte_len; verify the rmsnorm
   shader's descriptor/dispatch agrees. Run **once**, observe, recover if it hangs.
2. **Repeat for the other gated recorders** PR5a widened: `lora_add`/`lora_linear`,
   `gdn_recurrent`, `sdpa_fallback` (the Vulkan attention backward path), `cross_entropy_from_logits`,
   `cast`/`narrow`/`gqa_expand`. Each: record → backward → CPU parity, tiny + single-shot.
3. **R2 op-coverage check at `tape.backward()` time.** Some backward composites call ops without a
   native `vulkan_fwd` (`sum_axis`, `add_scalar`/`sub_scalar`, `log_softmax_last_dim`). PR3a's
   host-fallback should carry them correctly-but-slowly; the soak confirms no op *hard-errors*.
   Anything that does → port it (`vulkan_fwd`) or confirm it routes through `dispatch1/2/3`.

## PR6 (orchestration flip) — human-soak-gated, plan ready in `PR6-spec.md`

Once the gated recorders pass backward+parity: widen the tape-authoritative + grad-checkpointing
gates (`trainer.rs` ~2478/4869/5001/5990/7217, `opd.rs` ~2724/2948) and the server gate
(`KILN_VK_NATIVE_TRAINING`, `backend/vulkan.rs:1739`) to route Vulkan SFT/GRPO/OPD through
`trainer.rs`/`opd.rs`. **Watch (PR6-spec R1):** `base_dtype_supports_tape` (`trainer.rs:7237`) is
BF16-only but Vulkan trains F32 — **OPD routes clean on F32; SFT/GRPO stay gated** behind a dtype
decision. Ship OPD-on-Vulkan first; keep `KILN_VK_NATIVE_TRAINING` as an opt-OUT for one release.
This is the first point that needs **multi-step training** → full soak, not a bounded test.

## PR7 (delete the fork) — after PR6 soak signs off

Delete `vk_train.rs` (6109), `vk_forward.rs` (1762; its sole consumer is `vk_train.rs`), collapse
`vk_tensor.rs` to the leaf-carrier half, remove `save_vk_lora_adapter` (use the shared safetensors
path), remove the `KILN_VK_NATIVE_TRAINING` opt-out. **Keep all SPIR-V in `vk_ops/`** — it is the
Vulkan leaf-kernel layer. Fix the stale `kt_tape.rs`/`opd_candle_shim.rs` "Metal OPD backward bails"
comments. Net ≈ **−10k LOC**. `matmul_bf16w` re-enters `family_ported` once a BF16 FD test lands
(the bridge already passes logical byte_len; it's a test, not a code change).

## Build / run reference

```
CARGO_TARGET_DIR=<main>/target  cargo check -p kiln-model --features vulkan
KILN_TENSOR_VULKAN_TEST=1 ... cargo test -p kiln-model --features vulkan --test vk_tape_record_proof -- --nocapture --test-threads=1
```
Default (non-GPU) and `--features vulkan` builds of `kiln-tensor`/`kiln-model`/`kiln-train` are all
green at branch HEAD. `--features cuda`/`metal` could not be compile-verified on this Linux host
(no nvcc / no Metal); every cuda/metal gate edit is purely additive (`cuda,metal → cuda,metal,vulkan`),
so a compile-confirm on a CUDA box / Apple Silicon is the honest closing check before merge.
