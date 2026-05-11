# Kiln env-var reference — Phase 2 training acceleration

**Date:** 2026-05-11
**Scope:** every env var introduced or modified by the Phase 0–5 work in this branch, plus the relevant pre-existing knobs the runbook references.

This is a single-page operator reference. Authoritative sources are the docstrings on the corresponding helper functions; this doc just collects them in one place so you don't have to grep for each one.

## Memory budget

| Var | Type | Default | Effect |
|-----|------|---------|--------|
| `KILN_GPU_MEMORY_GB` | f64 (GB) | unset | Override auto-detected GPU memory. Highest priority among VRAM detection sources. |
| `KILN_TRAINING_MEMORY_RESERVE_GB` | f64 (GB) | `max(6, MemTotal / 4)` | On unified-memory hosts, GB held back from `MemTotal` before declaring the training budget. Lower it to free more headroom for training; raise it on workstations running heavier desktop workloads alongside. |
| `KILN_TRAINING_MEMORY_GB` | f64 (GB) | unset | Override the trainer-side budget directly (post-correction). Almost always you want `KILN_GPU_MEMORY_GB` instead. |
| `KILN_GRAD_CHECKPOINT_SEGMENTS` | usize | auto from VRAM | Number of segments for gradient checkpointing. More segments = less per-segment activation memory but more recompute. Auto-default is in `kiln-core::vram::recommended_checkpoint_segments`. |
| `KILN_NO_GRAD_CHECKPOINT` | `1`/`true` | unset | Disable gradient checkpointing entirely (keeps full activations resident). |

## Vulkan training acceleration

| Var | Tristate | Default | Effect |
|-----|----------|---------|--------|
| `KILN_VULKAN_LINEAR` | `1`/`0` | **on** | Routes training-time projection forward + backward through `VulkanLinearOp` with chunked dispatch. Default-on after the chunking + FLCE auto-engagement + LoRA CustomOp3 wiring made the entire training stack correct + safe by construction. Set to `0` to opt out for parity comparisons. |
| `KILN_VULKAN_LINEAR_MAX_GFLOP` | f64 (GFLOP) | 20 | Per-submit FLOP ceiling for `VulkanLinearOp`. Above this, the BF16-packed path chunks along output dim (forward) or batch dim (backward); the F32 path bails to CPU `broadcast_matmul`. Set to 0 to disable the guard (NOT recommended on unified APUs). |
| `KILN_VULKAN_SDPA` | `1`/`0` | **on** | Wires `flash_attn_prefill` to the `sdpa_prefill_f32` Vulkan kernel. Default-on now that the kernel is parity-tested at multiple shapes including Qwen3.5-4B head_dim=128. Set to `0` to opt out. |
| `KILN_VULKAN_FLCE` | `1`/`0`/auto | **auto** (engages at `active_count ≥ 16`) | Forces the Vulkan FLCE provider on or off, or lets the auto-heuristic decide. Auto threshold lowered from `active_count × num_chunks ≥ 50_000` after the host hangs (the unfused lm_head path it competes against is now itself catastrophic, not just slow). |
| `KILN_VULKAN_RMSNORM` | `1`/`0` | **on** | Inference-path RMSNorm Vulkan kernel. Default-on since v0.2.14. |
| `KILN_VULKAN_RMSNORM_TRAINING` | `1`/`0`/auto | **auto** (engages at `row_count ≥ 1024`) | Training-path Vulkan RMSNorm autograd. Below the threshold the per-call dispatch overhead exceeds the kernel's compute savings vs the candle CPU `broadcast_mul` chain. |
| `KILN_USE_FLCE` | `1`/`0` | **on** since v0.2.13 | Use Fused Linear Cross-Entropy for SFT loss. Set to 0 to use the unfused path (mainly for debugging). |

## Vulkan inference acceleration (pre-existing)

These are not modified by Phase 0–5 but are common knobs the runbook references.

| Var | Default | Effect |
|-----|---------|--------|
| `KILN_DISABLE_GDN_KERNEL` | unset (= enabled) | Disable all Vulkan GDN kernels (linear-attention layers fall back to candle). |
| `KILN_DISABLE_VULKAN_GDN_PREFILL_IN_PROJ` | unset | Disable the fused GDN in_proj prefill kernel. |
| `KILN_DISABLE_FUSED_GDN_GATES` | unset | Disable fused gates kernel. |
| `KILN_DISABLE_FUSED_GDN_GATED_RMS_NORM` | unset | Disable fused gated-RMSNorm kernel. |
| `KILN_DISABLE_VULKAN_GDN_DECODE_FUSED_RESIDENT_STATE` | unset | Disable the GDN decode-time resident-state fast path. |
| `KILN_DISABLE_VULKAN_LINEAR_DECODE` | unset | Disable the leaf-fast linear-decode kernel (decode-time inference). |
| `KILN_DISABLE_VULKAN_BF16_PACKED_LINEAR_WEIGHTS` | unset | Force F32 weight buffers instead of bf16-packed (debug only — much higher memory). |

## Telemetry

These don't change behaviour, just what gets logged.

- "GPU memory budget" log includes `vram_source` since this branch — `linux-drm-sysfs-unified` indicates the corrected unified-memory budget.
- "Vulkan training acceleration profile" startup log on Vulkan-enabled hosts shows the on/off state of every `KILN_VULKAN_*` training flag.
- `VulkanLinearOp::cpu_fwd first chunked dispatch` and `::bwd first chunked dispatch` traces fire once per process, surfacing the chunk count / per-chunk GFLOP that the guard chose.
- `VulkanBackend::register_resident_activation first call` trace fires once per process, confirming the Phase 3.1 lifecycle is engaging during checkpointed training.
- HTTP 413 rejection messages from `/v1/training/sft` and `/v1/training/grpo` include a `vram_source=...` clause when the corrected budget came from the unified-memory path — makes `KILN_TRAINING_MEMORY_RESERVE_GB` the obvious next knob.

## Quick-reference: defaults already enable the full Vulkan training stack

```sh
# Default `kiln serve` runs:
#   - KILN_VULKAN_LINEAR=on (chunked projections + lora_delta_resident
#     via VulkanLoraOp CustomOp3 with autograd backward)
#   - KILN_VULKAN_SDPA=on (full-attn matmuls via sdpa_prefill_f32)
#   - KILN_VULKAN_FLCE=auto (engages at active_count >= 16)
#   - KILN_VULKAN_RMSNORM=on (inference RMSNorm)
#   - KILN_VULKAN_RMSNORM_TRAINING=auto (engages at row_count >= 1024)
#   - dispatch_sgd_step on resident LoRA Vars (BF16 + F32 kernels)
# No env vars needed. Tune `KILN_VULKAN_LINEAR_MAX_GFLOP` if a
# specific dispatch shape needs more aggressive chunking.
```

## Quick-reference: disabling everything for fall-back

```sh
export KILN_VULKAN_LINEAR=0
export KILN_VULKAN_SDPA=0
export KILN_VULKAN_FLCE=0           # also disables auto-engagement
export KILN_VULKAN_RMSNORM_TRAINING=0
# Inference-path Vulkan kernels keep running — they were stable
# before the Phase 0-5 work and are not affected by this branch.
```
