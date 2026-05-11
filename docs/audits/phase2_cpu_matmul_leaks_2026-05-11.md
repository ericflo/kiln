# Phase 2.1 — CPU-fallback matmul leak audit (Strix Halo)

**Date:** 2026-05-11
**Plan:** `/home/ericflo/.claude/plans/please-redicat-all-of-reactive-hartmanis.md` § 2.1
**Goal:** identify every `broadcast_matmul`-style CPU compute leak in
`crates/kiln-model/src/forward.rs` that fires during a Vulkan training
forward pass on a Strix Halo APU (Qwen3.5-4B, T~918), so the routing
plan can be prioritised by FLOP impact.

## Method

`grep -n "broadcast_matmul\|broadcast_matmul_cpu_compatible"` against
`crates/kiln-model/src/forward.rs`, then walked each call site to
classify whether it (a) already routes through Vulkan via
`linear_with_lora_t_backend_decode_if`/`gdn_in_proj_matmul`/FLCE, or
(b) is a remaining CPU leak. Shapes computed for the original
`/tmp/sft-data.jsonl` repro (T=918, batch=1, hidden=2560,
num_heads=16, head_dim=128, vocab=152064, 24 GDN + 8 full-attn
layers, 32 MLP layers).

## Already routed (no action needed)

| Call site | What | Per-forward FLOP | Vulkan path |
| --- | --- | --- | --- |
| `forward.rs:643` | GDN combined QKV in_proj prefill | 38 GFLOP × 24 layers = 912 GFLOP | `gdn_in_proj_matmul` → `linear_prefill_apply` (chunked) |
| `forward.rs:2206` | GDN in_proj_z, in_proj_a, in_proj_b | ~50 GFLOP × 24 = 1200 GFLOP | `gdn_in_proj_matmul` → `linear_prefill_apply` (chunked) |
| `forward.rs:4154` | lm_head non-FLCE bridge | 715 GFLOP (when fired) | FLCE provider engaged at active_count ≥ 16 → `linear_prefill_apply_offset` (chunk_size=4096) |
| `forward.rs:6498/6576/6674/6751/6987/7191/7601/8090/8365/8401/8582/8742` | q/k/v/o/gate/up/down projection sites | 12-24 GFLOP each, ~3000 GFLOP total | `linear_with_lora_t_backend_decode_if` → `linear_prefill_apply` (chunked, 1-2 chunks each at the new 20 GFLOP ceiling) |
| `forward.rs:9363/9530` | lm_head projection sites | 715 GFLOP | `lm_head_forward_backend_decode_if` → `linear_prefill_apply` (chunked into ~38 submits) when training (`x.track_op()`); FLCE also covers SFT loss path |

## Remaining leaks (Phase 2 work)

| Call site | What | Per-forward FLOP | Status | Replacement plan |
| --- | --- | --- | --- | --- |
| `forward.rs:6728` | Full-attn prefill `Q @ K^T` (per-batch-per-head) | 3.5 GFLOP × 8 layers = 28 GFLOP | **CPU** | New SDPA Vulkan kernel(s); `flash_attn.comp` placeholder is buggy and `supports_flash_attn_prefill` returns `false` |
| `forward.rs:6736` | Full-attn prefill `softmax @ V` (per-batch-per-head) | 3.5 GFLOP × 8 = 28 GFLOP | **CPU** | Same SDPA kernel as above |
| `forward.rs:8484` | GQA grouped scores at decode | small (decode q_len=1) | **CPU** | Decode path; lower priority |
| `forward.rs:8531` | GQA grouped weighted-sum at decode | small | **CPU** | Decode path; lower priority |
| `forward.rs:8646` | Alternate prefill `Q @ K^T` (paged path) | similar to 6728 | **CPU** | Same SDPA kernel |
| `forward.rs:8686` | Alternate prefill `softmax @ V` (paged path) | similar to 6736 | **CPU** | Same SDPA kernel |
| `forward.rs:5404` | CUDA-only GDN combined A/B in_proj fastpath | only fires on CUDA | N/A | Not on the Vulkan path; ignore |
| `forward.rs:10548-50` | MTP `concat @ fc_t` final projection | small (MTP head only) | **CPU** | Phase 2 stretch; not on the SFT loss path |

## Prioritised by FLOP impact

1. **Full-attention SDPA prefill** (lines 6728/6736 and 8646/8686): ~56 GFLOP per forward. Currently the dominant CPU compute leak in the Vulkan training path. Needs a new Vulkan kernel — the existing `flash_attn.comp` is a placeholder with broken sharedmem indexing and missing scratch/LSE/causal-mask buffers, and `supports_flash_attn_prefill()` returns `false` in `vulkan.rs:577`. Substantial work but the only remaining "block" of ~50+ GFLOP not yet on GPU.

2. **MTP head matmul** (lines 10548-50): only fires when MTP is enabled; not part of the SFT loss path. Defer to post-Phase 2.

3. **Decode-time GQA reshape matmuls** (lines 8484/8531): only matter at inference decode. Lower priority for the training-OOM problem this plan targets.

## Total FLOP picture (T=918 forward, KILN_VULKAN_LINEAR=1)

| Component | FLOP | On GPU? |
| --- | --- | --- |
| GDN in_proj (24 layers) | ~2100 GFLOP | yes |
| MLP gate/up/down (32 layers) | ~1500 GFLOP | yes |
| Full-attn q/k/v/o proj (8 layers) | ~400 GFLOP | yes |
| **Full-attn SDPA inner (8 layers)** | **~56 GFLOP** | **NO — Phase 2 work** |
| lm_head + cross-entropy | ~715 GFLOP | yes (FLCE chunked) |
| **Total** | **~4770 GFLOP** | **~99% on GPU** |

Routing the remaining ~56 GFLOP of SDPA inner matmuls would close the
last meaningful CPU leak in the Vulkan training forward path. The wall
impact is bounded (~2 s saved per training step at 25 TFLOPS), but it
removes the last CPU-bound serialization point that could starve the
GPU between projection batches.

## Action items

- [ ] Replace `flash_attn.comp` with a correct fused-attention shader
      (online softmax, causal mask, F32 inputs, F32 output).
- [ ] Wire `flash_attn_prefill_vulkan` (`vulkan.rs:504`) and flip
      `supports_flash_attn_prefill()` to `true` for the Strix Halo /
      head_dim=128 / GQA path.
- [ ] Per-kernel parity test against CPU baseline at small T (T=8,
      H=2, dh=16) and an integration test at training T (T=918) once
      hardware-load-validation is greenlit by the user.
