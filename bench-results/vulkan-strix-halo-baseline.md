# Vulkan decode baseline — AMD Radeon 8060S (RADV STRIX_HALO)

First post-candle-drop (#1082) Vulkan inference baseline, measured on the
Strix Halo APU dev box. This is the regression gate for the DoD line
"**Vulkan decode bs=1 ≥ current Vulkan baseline**".

## Environment

- GPU: AMD Radeon 8060S Graphics (RADV STRIX_HALO), integrated, unified memory
- Driver: Mesa 26.1.1 (radv), Vulkan 1.4
- Host: ~30 GiB unified RAM (shared CPU/GPU)
- Model: Qwen3.5-4B, BF16 weights, head_dim=256
- Build: `cargo build --release --no-default-features --features vulkan`
- Server: `KILN_NUM_BLOCKS=2048 KILN_MODEL_PATH=./Qwen3.5-4B kiln serve`
  (small KV cache so the 18 GiB of CPU-host kt weights + vk buffers + KV fit;
  the default KV cache is sized for an 80 GiB datacenter card)
- Date: 2026-05-31

## Numbers (bs=1, greedy, native single-submit resident decode)

| Metric | Value | Notes |
|---|---|---|
| Decode forward / token | **69.8 ms** | `last_forward_ms`; ~14.3 tok/s steady-state |
| Decode tok/s (steady) | **~14.3 tok/s** | one CommandBatch per token (native resident path) |
| First-token cost | ~1.4 s (one-time) | SPIR-V pipeline compile on first decode of a fresh process |
| Prefill (~14-tok prompt) | ~3.2 s | generic per-op path — NOT yet on the resident/single-submit path |
| Model load | ~20–31 s | CPU-host weights + lazy first-forward vk upload |
| Memory (load → decode) | ~19 GiB, **flat** | no per-token growth, no OOM (kt-keyed weight caches) |

Pre-fix baseline (for reference — the bug state this session resolved):
decode was **~1588 ms/token (~0.5 tok/s)** through the generic
`model_forward_paged_batched_decode_hidden` path, and the process OOM-killed
after a few tokens due to per-token weight re-upload into an unbounded
candle-`TensorId`-keyed cache. The Vulkan path could not even load the model
before the loader-regression fix.

## Correctness (parity)

- `kiln-vulkan-kernel` parity suite: **127/127 pass** on hardware (incl. the
  new `vk_sdpa_prefill_kernel_parity` at head_dim 64/128/256).
- End-to-end generation validated coherent:
  - greedy, short: `"Hi"` → `"Hello!"` (natural EOS)
  - sampling (temp 0.7, freq-penalty 0.5), 36-tok prompt: → `"The clear daytime
    sky is blue"` (natural EOS)
  - greedy long-output repetition is ordinary greedy degeneracy (no
    repetition penalty), not a forward bug.

## Prefill profile (the next lever) — `KILN_PROFILE_PAGED_LAYERS=1`, 48-token prompt

Per-layer wall time (one prefill, all 32 layers):

| layer kind | per-layer | count | subtotal |
|---|---|---|---|
| GDN (linear attention) | **~400 ms** | 24 | ~9.6 s |
| Full attention | ~246 ms | 8 | ~2.0 s |
| **Total prefill (48 tok)** | | | **~12.8 s** |

The **GDN linear-attention prefill dominates** (~75% of prefill).

**Root cause (pinpointed):** `gdn_chunkwise_recurrence` (`forward.rs` ~12338)
runs its per-chunk matmuls as **raw kt `.matmul()`** (`k_c.matmul(&state)`,
`k_c.matmul(&k_t)`, `q_c.matmul(&k_t)`, `q_c.matmul(&state)`, …). On Vulkan the
activations/state are CPU-host kt tensors, so **these matmuls execute on the
CPU** — the GDN prefill recurrence never touches the GPU. (Decode is fine: it
uses the Vulkan `backend.gdn_recurrent_step` kernel. The BF16-gated
`recurrent_unexpanded_qk` native-prefill fast path also doesn't engage, since
Vulkan activations are F32.)

**Update — GPU-parallel chunkwise prefill wired in (commit `kiln-model/vulkan:
run GDN prefill chunkwise scan on the GPU`):** `gdn_chunkwise_recurrence` now
dispatches `vk_gdn_chunkwise_forward_no_grad` (the GPU-parallel forward chunkwise
kernel that already existed for training) via a new `BackendRuntime::
gdn_chunkwise_forward`. GDN prefill layer **400 → 246 ms**, total prefill
**12.8 → 9.1 s (~29%)**. Validated correct (GPU and CPU chunkwise produce
identical greedy output). CUDA/Metal untouched (trait default Ok(None)).

**Single-submit fusion target (the proper "max out the hardware" step):**
246 ms/layer is **dispatch-bound** — `vk_gdn_chunkwise_forward_no_grad` issues
~20-30 submit+wait per layer (`chunk_forward_no_grad` does 6-8
`vk_matmul_batched_no_grad`, each self-submits+reads-back, + `state_update`),
instead of chaining the whole layer's scan into ONE `CommandBatch`.

Implementation plan (confirmed feasible — `CommandBatch::record_shader` +
`record_copy_buffer` + one `submit_and_wait` already host this pattern in
`vk_decode_resident::record_gdn_block_into`): write/finish a
`record_gdn_chunkwise_prefill_block_into` that pre-allocates the chunk
intermediates (q_s, ks, kkt, qkt, b_mask, w_weighted, p_last, k_t, per-chunk
out, state) as device-local buffers and `record_shader`s the matmul / forward-
sub / state-update dispatches in dependency order across all chunks (the
cross-chunk state dep is just the next chunk reading the prior chunk's updated
state buffer — no readback). One submit per layer instead of ~20-30. Then route
`gdn_chunkwise_recurrence` to it. This is correctness-critical (a wrong binding
corrupts all output) → implement deliberately with incremental parity vs the
current GPU chunkwise (already validated identical to CPU), not rushed.

**Update — single-submit recorder landed (first implementation slice):**
`record_gdn_chunkwise_prefill_block_into` now records the same per-chunk
narrow/matmul/prep/solve/update/scatter sequence into one `CommandBatch`, owns
all transient buffers until submit completion, and exposes
`vk_gdn_chunkwise_forward_no_grad_single_submit`. The Vulkan backend tries this
path first for GDN prefill and can fall back to the existing Vulkan chunkwise
path if the recorder rejects a shape/config. Direct parity test coverage is
candle-free and passes against a CPU recurrence oracle on a multi-chunk shape:
output max abs err `1.117587e-8`, state max abs err `5.960464e-8`
(`vk_gdn_chunkwise_single_submit_matches_cpu_multichunk`). Full
`vk_gdn_chunkwise_parity` suite: **7/7 pass** on Vulkan hardware. Real
end-to-end prefill timing still needs a kt/Vulkan-only bench path; do not count
the existing release bench binary as proof for this item because it still links
the old candle-facing app stack.

**Update — kt/Vulkan-only microbench for the single-submit path:** added
`crates/kiln-vulkan-kernel/examples/gdn_chunkwise_prefill_microbench.rs`, which
uploads raw F32 inputs directly into `VkTensor`s and compares the previous
per-dispatch Vulkan chunkwise path with the new single-submit path. The
benchmark excludes input upload and includes output/intermediate allocation,
command recording, queue submits, and GPU waits. `cargo tree -p
kiln-vulkan-kernel --edges normal,build -i candle-core` prints nothing, so this
measurement does not depend on the app-layer tensor stack.

| shape | legacy per-dispatch | single-submit | speedup | correctness |
|---|---:|---:|---:|---|
| B=1, H=32, T=48, DK=128, DV=128, C=64 | 0.655 ms | 0.251 ms | **2.61x** | out/state max abs err 0 |
| B=1, H=32, T=128, DK=128, DV=128, C=64 | 1.531 ms | 0.771 ms | **1.98x** | out/state max abs err 0 |

Other proper work-packages for full saturation (per "max out the hardware in
every config"):
- **True multi-row batched resident decode** (bs>1 / continuous-batched
  routing is wired; remaining work is end-to-end perf validation and saturation
  tuning).
  **Update — first native resident batch primitive landed:** added
  `paged_kv_write_slots`, a Vulkan dispatch that copies `[batch,
  num_kv_heads * head_dim]` projected K/V rows into per-row resolved KV-cache
  slots in one submit, with a bounds-aware slot guard. This removes one of the
  b1-only assumptions in the resident decode block recorder; full bs>1 decode
  orchestration still needs the batched row path through QKV/split/RoPE/MLP and
  the resident paged-attention call.
  **Update — second batch primitive:** added `qkv_gate_split_batched`, which
  splits the existing batched full-attention QKV projection output into
  `[batch, q]`, `[batch, gate]`, `[batch, k]`, and `[batch, v]` buffers on GPU.
  This closes another one-row gap between the batched projection shaders and
  a full multi-row resident full-attention block recorder.
  **Update — GDN split primitives:** added batched GDN in-proj and mixed-QKV
  split kernels, so the existing batched GDN projection/conv kernels can feed
  the recurrent step without rowwise host-side slicing.
  **Update — fused batched MLP down+residual:** added
  `linear_decode_batched_bf16w_add_residual`, closing the single-row fused
  residual-add dependency used by both resident full-attention and GDN blocks.
  **Update — fused batched add+RMSNorm:** added `add_qwen_rmsnorm_batched`,
  closing the other single-row fused residual dependency in the resident block
  recorder while preserving one dispatch per residual+norm pair.
  **Update — resident batch route wired:** greedy continuous-batched Vulkan
  decode now defaults to the multi-row path instead of the server-side rowwise
  loop. `model_forward_paged_decode_contiguous_batch_greedy_with_ids` routes
  stable row-ID batches through the resident transformer-stack + argmax
  `CommandBatch`; row prompt K/V is seeded once per full-attention layer and
  then resident per-token slot writes remain authoritative. Mixed existing/new
  GDN rows seed missing kt-keyed resident recurrent + conv state before batch
  assembly. No-ID callers decline the resident route and use the portable path
  to avoid unsafe cache reuse.
  **Update — Vulkan-only resident token microbench:** moved the decode
  microbench to the normal `kiln-vulkan-kernel` binary
  `vulkan_decode_microbench`, with direct host-slice weight uploads and
  targeted sweep controls via `KILN_VK_MICROBENCH_BATCHES`,
  `KILN_VK_MICROBENCH_WARMUP`, `KILN_VK_MICROBENCH_TIMED`, and
  `KILN_VK_MICROBENCH_REPEATS`. On RADV STRIX_HALO,
  `full_token_resident_batched` (32 layers, synthetic contiguous K/V pool,
  one submit per token) measured after routing the large-batch QKV, out-proj,
  gate/up, and down paths through row-reuse shader variants:

  | batch | per token | rows/s |
  |---:|---:|---:|
  | 1 | 57.3 ms | 17 |
  | 4 | 80.4 ms | 50 |
  | 8 | 139.6 ms | 57 |
  | 16 | 164.9 ms | 97 |
  | 32 | 278.1 ms | 115 |
  | 64 | 508.7 ms | 126 |

  The resident full-attention synthetic benchmark now matches the production
  gated-Q dataflow: Q projection width is doubled for the attention output
  gate, then `qkv_gate_split_batched` and fused Q/K norm run before RoPE; the
  post-attention residual+norm and MLP down+residual use the same fused kernels
  as the production batched recorder. Earlier synthetic numbers skipped the
  split/norm work and overstated the full-attention-only path. With the
  corrected benchmark, throughput scales to about 126 rows/s at batch 64 on
  this APU; the next tuning target is deeper tiling/cache reuse inside the
  large projection/MLP shaders rather than host submit count.
  A follow-up direct-output full-attention QKV+gate projection now writes
  q/gate/k/v directly and removes the separate batched split dispatch from the
  production resident recorder plus these microbench paths. On this APU it is
  a command/bandwidth cleanup rather than a measurable high-batch throughput
  win by itself. The older combined-QKV dispatcher still waits until batch 2
  before using its rows4 shader because same-session standalone A/B showed
  batch 1 regressing from 403 us to 471 us, while batch 2 improved from 671 us
  to 529 us, batch 4 from 1.23 ms to 0.67 ms, and batch 8 from 2.32 ms to
  1.13 ms.
  **Update — mixed Qwen3.5 resident token microbench:** added
  `full_token_resident_mixed_batched`, which records the real Qwen3.5-4B layer
  mix into one `CommandBatch`: 8 full-attention layers at indices
  `3,7,11,...,31` plus 24 GDN layers. This is still a Vulkan-only synthetic
  benchmark with direct host-slice weight uploads and shared scratch buffers,
  but it is the best current signal for decode saturation at the actual
  layer mix. On RADV STRIX_HALO with `KILN_VK_MICROBENCH_BATCHES=1,8,32,64`,
  warmup=2, timed=5, repeats=2, plus a short batch-128 probe:

  | batch | per token | rows/s |
  |---:|---:|---:|
  | 1 | 62.0 ms | 16 |
  | 8 | 115.5 ms | 69 |
  | 32 | 229.6 ms | 139 |
  | 64 | 458.9 ms | 139 |
  | 128 | 924.3 ms | 138 |

  The mixed stack is now the right benchmark for resident decode tuning:
  full-attention-only synthetic decode overstates throughput because the 24
  GDN layers dominate the real model. Batch 128 also confirmed the rows8 MLP
  shader path was still slower than rows4 on STRIX_HALO, so the default rows8
  crossover moved from batch 128 to batch 256. The table above was rerun after
  fixing the full-attention subpath to include the gated-Q split and fused Q/K
  norm used by the production resident recorder, then rerun after fusing the
  paired GDN Q/K L2 expansion dispatches. A follow-up source audit found the
  standalone MLP path had the 256-row rows8 cutoff, but resident MLP dispatch
  selection still had older batch-64 rows8 checks; resident decode now shares
  the same 256-row cutoff so production and benchmark routing match. A short
  patched rerun with warmup=1, timed=3, repeats=2 measured batch 64 at
  462.1 ms / 139 rows/s and batch 128 at 934.6 ms / 137 rows/s. The linear
  bf16 rows4 disable knob is now honored under both the canonical
  `KILN_DISABLE_VULKAN_LINEAR_DECODE_BF16W_ROWS4` name and the older
  command-batch planner name, so production and microbench A/Bs use the same
  routing control.
  **Update — mixed resident paged token microbench:** added
  `full_token_resident_mixed_paged`, which keeps the real 8 full-attention +
  24 GDN layer mix but changes each full-attention layer to use
  `paged_kv_write_slots` plus split-K
  `paged_attn_decode_batch_paged_splitk`/reduce over real per-row block
  tables. This is the closest Vulkan-only synthetic benchmark to production
  continuous-batched decode at the default 256-token history window. On RADV
  STRIX_HALO with `KILN_VK_MICROBENCH_BATCHES=1,8,32,64`, warmup=2, timed=5,
  repeats=2, plus a batch-128 probe:

  | batch | per token | rows/s |
  |---:|---:|---:|
  | 1 | 59.2 ms | 17 |
  | 2 | 60.8 ms | 33 |
  | 4 | 68.6 ms | 58 |
  | 8 | 104.3 ms | 77 |
  | 32 | 240.4 ms | 133 |
  | 64 | 459.0 ms | 139 |
  | 128 | 921.0 ms | 139 |

  The mixed-paged result is effectively tied with the contiguous mixed
  benchmark at batch 64/128, so the paged slot write + block-table attention
  path is not the dominant saturation limit at this window; the remaining
  high-batch ceiling is still in the projection/MLP/GDN-heavy parts of the
  recorded token. Same-session A/Bs then lowered the full-attention QKV+gate
  rows4 threshold to batch 1: mixed paged batch 1 moved from 61.2 ms / 16
  rows/s to 59.2 ms / 17 rows/s, batch 2 moved from 65.6 ms / 31 rows/s to
  60.8 ms / 33 rows/s, batch 4 moved from 71.5 ms / 56 rows/s to 68.6 ms /
  58 rows/s, and batch 8 moved from 114.6 ms / 70 rows/s to 104.3 ms / 77
  rows/s.
  **Update — long-context mixed-paged sweep:** the mixed-paged benchmark now
  exposes `KILN_VK_PAGED_HISTORY` and `KILN_VK_PAGED_BLOCK_SIZE`, sizes
  `blocks_per_seq` from the requested decode position, and allocates resident
  paged K/V storage only for the 8 full-attention layers in the real Qwen3.5-4B
  layer mix. This keeps the Vulkan-only benchmark representative while making
  longer history windows measurable on the APU. With block size 16, warmup=1,
  timed=3, repeats=2:

  | history | batch | per token | rows/s |
  |---:|---:|---:|---:|
  | 256 | 8 | 113.2 ms | 71 |
  | 256 | 64 | 460.2 ms | 139 |
  | 1024 | 8 | 117.2 ms | 68 |
  | 1024 | 32 | 247.0 ms | 130 |
  | 2048 | 8 | 121.9 ms | 66 |

  The 1024/2048-token runs verify the split-K paged-attention block-table path
  at longer contexts without changing the conclusion: at practical resident
  batch sizes, the dominant mixed-stack cost is still the projection/MLP/GDN
  shader work, not paged K/V slot writes or block-table attention.
  **Update — adaptive batched split-K chunks:** short same-session sweeps show
  the batched paged-attention default should keep 4 chunks at the 16-block
  256-token window but use 2 chunks once `max_blocks_per_seq >= 64`. At
  history 256, batch 8 measured 106.8 ms with 2 chunks vs. 106.1 ms with 4,
  and batch 64 measured 464.2 ms with 2 vs. 462.5 ms with 4. At history 1024,
  batch 32 measured 255.4 ms with 2 chunks vs. 260.7 ms with 4. At history
  2048, batch 8 measured 123.5 ms with 1 chunk, 112.1 ms with 2, 113.1 ms
  with 4, and 115.4 ms with 8. Patched no-override smoke checks measured
  history 2048 / batch 8 at 113.0 ms / 71 rows/s and history 256 / batch 64
  at 460.3 ms / 139 rows/s.
  **Update — batched argmax readback cleanup:** the native batched resident
  decode submit path now lets the final LM-head argmax reduce shader write its
  `batch * 4` byte token-id output directly into the persistent host-visible
  staging buffer. This removes the tiny device-local token buffer and the
  recorded copy command from each batched decode submit.
  **Update — no-ID batched resident route:** callers without stable row IDs now
  still use the same native batched resident decode stack. The row-ID path keeps
  its per-row seed cache; the no-ID path conservatively re-seeds each active
  full-attention row from the paged cache on every call and clears the row-ID
  seed cache before doing so, avoiding unsafe cache reuse while eliminating the
  previous forced portable/rowwise route.
  **Update — Vulkan live decode batch default:** the live greedy decode batcher
  now defaults Vulkan to a 16-row max batch instead of 8. Focused resident
  mixed-paged decode microbench on RADV STRIX_HALO, history 256, warmup=1,
  timed=4, repeats=3 measured batch 8 at 105.1 ms / 76 rows/s and batch 16 at
  142.7 ms / 112 rows/s. A second pass measured batch 16 at 148.4 ms /
  108 rows/s and batch 32 at 241.8 ms / 132 rows/s; keep the default at 16 as
  the conservative memory/throughput step, with env overrides still available
  for 32/64-row sweeps.
  **Update — GDN resident recorder uses row-reuse in-proj:** the batched GDN
  `CommandBatch` recorder now selects the same pair QKV/Z plus rows2/rows4
  BF16 in-proj shaders as the standalone dispatcher. Focused
  `vulkan_decode_microbench gdn_in_proj` uses real Qwen3.5-4B GDN dimensions
  (`qkv_dim=8192`, `z_dim=4096`, `a_dim=b_dim=32`). On RADV STRIX_HALO with
  `KILN_VK_MICROBENCH_BATCHES=8,32,64`, warmup=2, timed=5, repeats=2:

  | batch | basic path rows/s | row-reuse rows/s |
  |---:|---:|---:|
  | 8 | 3,626 | 6,520 |
  | 32 | 3,749 | 13,823 |
  | 64 | 3,383 | 14,198 |

  That removes the remaining basic in-proj shader choice from the GDN
  resident batch recorder, which matters because GDN layers are the majority of
  the Qwen3.5-4B stack.
  **Update — whole GDN resident block microbench:** added
  `vulkan_decode_microbench gdn_block_resident_batched`, a Vulkan-only
  one-submit benchmark for a full Qwen3.5-4B GDN block plus MLP. It records 10
  shaders into one `CommandBatch`: RMSNorm, row-reuse GDN in-proj, fused
  conv/split/state advance, fused Q/K L2 expansion, recurrent gate/RMSNorm, GDN
  out-proj, residual, post norm, MLP gate/up, and fused down+residual. On RADV
  STRIX_HALO with `KILN_VK_MICROBENCH_BATCHES=1,4,8,32,64`, warmup=2, timed=5,
  repeats=2:

  | batch | per block | rows/s |
  |---:|---:|---:|
  | 1 | 1.86 ms | 536 |
  | 4 | 2.19 ms | 1,828 |
  | 8 | 3.28 ms | 2,436 |
  | 32 | 6.76 ms | 4,733 |
  | 64 | 13.90 ms | 4,605 |

  This gives a realistic GDN-side decode saturation baseline. A follow-up
  planner change keeps batch 64 on the rows4 MLP shader path by default because
  rows8 was slower on STRIX_HALO at this size. That improved the GDN block from
  16.80 ms / 3,809 rows/s to 13.90 ms / 4,605 rows/s at batch 64 after also
  reducing the state-advance dispatch from one workgroup per row-channel to
  one workgroup per 256 row-channels, then fusing GDN split, conv, state
  advance, QKV split, and the paired Q/K L2 expansion. After correcting the
  synthetic full-attention subpath, the mixed full-token resident batch-64
  case now measures 458.9 ms / 139 rows/s. A follow-up MLP planner sweep keeps
  the bf16 down+residual rows4 path disabled at batch 8, but enables it from
  batch 16: on STRIX_HALO the GDN block batch-16 path moved from 4.73 ms to
  4.19 ms, and the mixed paged token batch-16 path moved from 150.1 ms to
  146.6 ms in a same-session A/B.
- **Vulkan paged-attention decode kernel**: the kernel crate already had
  `paged_attn_decode_batch_paged.comp`; Vulkan now advertises
  `supports_flash_attn_paged_decode` and wires the single-query paged-decode
  trait call through the GPU-side block-table walker from kt tensors, so the
  long-context non-contiguous decode path no longer has to fall straight to the
  manual GQA attention path. Remaining saturation work is the multi-row
  resident decode orchestration below, not the existence of the paged attention
  kernel.
  **Update — multi-row resident paged microbench:** `full_token_resident_paged`
  now uses `paged_kv_write_slots` plus split-K
  `paged_attn_decode_batch_paged_splitk` and reduce over real per-row block
  tables. The default split-K policy uses 32 chunks for batch 1 and 4 chunks
  for multi-row resident batches, with
  `KILN_VK_PAGED_ATTN_SPLITK_CHUNKS` still available for forced sweeps. With
  `KILN_VK_MICROBENCH_BATCHES=1,4,64`, warmup=2, timed=5, repeats=2 on RADV
  STRIX_HALO:

  | batch | per token | rows/s |
  |---:|---:|---:|
  | 1 | 49.3 ms | 20 |
  | 4 | 75.0 ms | 53 |
  | 64 | 503.2 ms | 127 |

  These numbers also use the corrected gated-Q full-attention dataflow and the
  direct-output full-attention QKV+gate projection. A batch-64 split-K sweep
  after the correction measured 505.5 ms at 1 chunk,
  500.7 ms at 2 chunks, 497.3 ms at 4 chunks, 501.5 ms at 8 chunks, and
  503.8 ms at 16 chunks, so the existing default of 4 chunks remains the best
  setting on this machine. A batch-1 sweep at history 2048 measured 65.5 ms
  at 4 chunks, 62.2 ms at 8 chunks, 60.9 ms at 16 chunks, 58.1 ms at 32
  chunks, and 60.5 ms at 64 chunks, so the single-row default moves to 32
  chunks. History 1024 and 256 also showed a small/no-regression edge for 32
  chunks, including 49.3 ms vs 50.0 ms on the paged-only batch-1 history-256
  path.

## Other follow-ups (perf headroom, not regressions)

1. **First-token shader compile** (~1.4 s) could be prewarmed at load.
2. Complete the remaining shared-stack dependency cleanup audit for any
   non-Vulkan decode islands; the Vulkan-specific decode weight path is now
   duplicate-copy-free.
