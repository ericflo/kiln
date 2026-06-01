# Vulkan decode baseline — AMD Radeon 8060S (RADV STRIX_HALO)

First post-legacy-stack-drop (#1082) Vulkan inference baseline, measured on the
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
| Prefill (~14-tok prompt) | ~3.2 s | historical server baseline; current app-level paged latency smoke below measured 6.6 s for the bench's 10-token prompt |
| Model load | ~20–31 s | CPU-host weights + lazy first-forward vk upload |
| Memory (load → decode) | ~19 GiB, **flat** | no per-token growth, no OOM (kt-keyed weight caches) |

Current app-level paged latency smoke after the resident/paged routing and
fallback-gate changes:

```bash
KILN_NUM_BLOCKS=2048 KILN_BENCH_LOG_ITL=1 KILN_VK_RESIDENT_DECODE_TIMING=1 \
  ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only \
  --paged --prompt-tokens 8 --max-output-tokens 6 --latency-warmup-runs 1 \
  --skip-training
```

This rebuilt release binary reported `backend: vulkan`, model load 19.85 s,
10 prompt tokens, prefill 6596.4 ms, first decode step 1061.5 ms, p50 decode
ITL 68.8 ms, mean ITL 234.0 ms across 7 generated tokens, and 4.27 decode
tok/s. The p50 steady decode matches the native resident single-submit baseline;
the first decode step remains the visible first-use pipeline cost in this
app-level path.

Pre-fix baseline (for reference — the bug state this session resolved):
decode was **~1588 ms/token (~0.5 tok/s)** through the generic
`model_forward_paged_batched_decode_hidden` path, and the process OOM-killed
after a few tokens due to per-token weight re-upload into an unbounded
legacy tensor-id keyed cache. The Vulkan path could not even load the model
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
path first for GDN prefill and now treats recorder failure as visible by
default; the older per-dispatch Vulkan chunkwise path is debug opt-in via
`KILN_VULKAN_GDN_CHUNKWISE_FALLBACK=1`. Direct parity test coverage is
legacy-stack-free and passes against a CPU recurrence oracle on a multi-chunk shape:
output max abs err `1.117587e-8`, state max abs err `5.960464e-8`
(`vk_gdn_chunkwise_single_submit_matches_cpu_multichunk`). Full
`vk_gdn_chunkwise_parity` suite: **7/7 pass** on Vulkan hardware. Real
end-to-end prefill timing still needs a kt/Vulkan-only bench path; do not count
the existing release bench binary as proof for this item because it still links
the old app-facing stack.

**Update — kt/Vulkan-only microbench for the single-submit path:** added
`crates/kiln-vulkan-kernel/examples/gdn_chunkwise_prefill_microbench.rs`, which
uploads raw F32 inputs directly into `VkTensor`s and compares the previous
per-dispatch Vulkan chunkwise path with the new single-submit path. The
benchmark excludes input upload and includes output/intermediate allocation,
command recording, queue submits, and GPU waits. A kernel-crate dependency audit
shows no app-layer tensor-stack dependency, so this measurement does not depend
on the app-layer tensor stack.

| shape | legacy per-dispatch | single-submit | speedup | correctness |
|---|---:|---:|---:|---|
| B=1, H=32, T=48, DK=128, DV=128, C=64 | 0.655 ms | 0.251 ms | **2.61x** | out/state max abs err 0 |
| B=1, H=32, T=128, DK=128, DV=128, C=64 | 1.531 ms | 0.771 ms | **1.98x** | out/state max abs err 0 |
| B=1, H=32, T=512, DK=128, DV=128, C=64 | 7.070 ms | 3.041 ms | **2.32x** | out/state max abs err 0 |

Current smoke on the same Vulkan-only example with T=128, warmup=1, iters=3,
repeats=2 measured legacy per-dispatch 1.715 ms vs. single-submit 0.756 ms
(**2.27x**) with output/state max abs err 0.

The expanded Vulkan pipeline prewarm now also fills the path-keyed
`CommandBatch::record_shader` cache for the chunkwise-prefill recorder's narrow,
scatter, batched matmul, transpose, solve, broadcast, and elementwise stages, so
the first recorded GDN prefill no longer lazily creates those pipelines.
The recorder also skips conservative compute barriers between each chunk's
independent q/k/v/beta/g narrow-copy dispatches; the first dependent matmul or
transpose still emits the visibility barrier for all prior slice writes. A
rerun of the T=512 shape after that change measured legacy per-dispatch 6.591 ms
vs. single-submit 2.941 ms (**2.24x**) with output/state max abs err 0.

The single-submit route is now mandatory by default even when
`KILN_DISABLE_VULKAN_GDN_CHUNKWISE_SINGLE_SUBMIT` is set: that debug override
must be paired with `KILN_VULKAN_GDN_CHUNKWISE_FALLBACK=1` before the older
per-dispatch chunkwise implementation can run. This keeps production prefill
from silently losing the one-CommandBatch path. A post-gate Vulkan-only smoke
on the T=128 shape measured legacy per-dispatch 2.325 ms vs. single-submit
0.852 ms (**2.73x**) with output/state max abs err 0.

**Update — app-facing GDN prefill input upload is batched:** the Vulkan backend
now borrows contiguous CPU F32 kt storage bytes for q/k/v/beta/g/state and
uploads all six GDN chunkwise prefill inputs through one staging buffer and one
Vulkan queue submission before recording the single-submit scan. Unsupported
layouts still use the previous flatten/upload path. Focused helper coverage
checks exact F32 byte borrowing and non-F32 rejection under
`cargo test -p kiln-model --lib --no-default-features --features vulkan
cpu_contiguous_f32_tensor_upload -- --nocapture`; the Vulkan round-trip test
`gdn_chunkwise_batched_input_upload_round_trips_on_vulkan` verifies the
batched-upload path itself when a Vulkan device is present.
The same boundary now reads the scan output and updated recurrent state back as
raw F32 bytes and rebuilds CPU kt tensors with `Tensor::from_raw_bytes_on`,
avoiding the previous `to_vec_f32` typed decode and `Tensor::from_vec`
rehydration. `vk_f32_tensor_to_cpu_tensor_rebuilds_from_raw_bytes` covers that
readback helper on Vulkan hardware.
The output/state pair now also shares one readback staging buffer and one queue
submission through `VulkanBuffer::read_back_batch`; kernel-crate coverage
(`read_back_batch_matches_individual_reads`) checks the generic primitive, and
`vk_f32_tensors_to_cpu_tensors_batched_rebuilds_from_raw_bytes` covers the
app-facing F32 tensor reconstruction path.

**Update — full-attention prefill wrapper stays kt-native:** the Vulkan
`flash_attn_prefill` implementation now extracts Q/K/V F32 bytes directly from
kt tensors and reconstructs the SDPA output as a kt tensor. This removes the
deprecated tensor bridge from the full-attention prefill wrapper while keeping
the same `sdpa_prefill_f32` Vulkan kernel route and shape gates. Focused check:
`cargo check -p kiln-model --no-default-features --features vulkan` passes with
the repo's existing warning backlog.

**Update — generic BF16 linear rows8 threshold is runtime-tunable:** the rows8
selector used by attention out-proj, GDN out-proj, resident BF16 linear, and
batched lm-head argmax/sample now honors
`KILN_VULKAN_LINEAR_BF16_ROWS8_MIN_BATCH`, defaulting to 64; the existing rows8
disable envs still win. Same-session mixed-paged batch64 microbench on Strix
Halo kept the default: generic BF16 linear rows8 enabled measured 450.6 ms/iter
(142 rows/s), while forcing it off measured 459.1 ms/iter (139 rows/s).
Disabling full-attention QKV rows8 was also slower at 452.3 ms/iter. Treat this
as a hardware-portability tuning hook, not a default flip.
The same selector family now also honors
`KILN_VULKAN_LINEAR_BF16_ROWS4_MIN_BATCH`, defaulting to 16. This covers the
rows4 crossover for attention out-proj, GDN out-proj, direct resident BF16
linear, and batched lm-head argmax/sample while keeping the Strix Halo default
unchanged.

Other proper work-packages for full saturation (per "max out the hardware in
every config"):
- **True multi-row batched resident decode** (bs>1 / continuous-batched
  routing is wired and live-instrumented; remaining work is saturation tuning,
  not route reachability).
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
  assembly. No-ID callers now use the same resident route; they conservatively
  re-seed active full-attention rows and clear the row-ID seed cache before
  doing so to avoid unsafe cache reuse.
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
  recorded token. An earlier same-session A/B temporarily lowered the
  full-attention QKV+gate rows4 threshold to batch 1: mixed paged batch 1 moved
  from 61.2 ms / 16 rows/s to 59.2 ms / 17 rows/s, batch 2 moved from 65.6 ms
  / 31 rows/s to 60.8 ms / 33 rows/s, batch 4 moved from 71.5 ms / 56 rows/s
  to 68.6 ms / 58 rows/s, and batch 8 moved from 114.6 ms / 70 rows/s to
  104.3 ms / 77 rows/s. A later direct block-level A/B below moved the
  production and microbench cutoff back to batch 2 because rows4 is slower for
  the batch-1 direct-output projection on this APU.
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
  **Update — GDN paired bf16 packed-word reuse:** the paired QKV/Z GDN in-proj
  bf16 shaders now unpack both adjacent bf16 columns from one loaded `uint`
  instead of reloading the same packed word for the second column. Direct
  `gdn_in_proj` on RADV STRIX_HALO with warmup=5, timed=20, repeats=5 moved
  batch 8 from 1093.9 us to 929.6 us, batch 16 from 1277.9 us to 1155.2 us,
  batch 32 from 2319.2 us to 2123.6 us, and batch 64 from 4520.6 us to
  4113.6 us. A production-shaped `full_token_resident_mixed_paged` check with
  history 256, warmup=1, timed=4, repeats=3 measured batch 8 at 95.6 ms /
  84 rows/s, batch 32 at 225.3 ms / 142 rows/s, and batch 64 at 453.2 ms /
  141 rows/s.
  **Update — adaptive batched split-K chunks:** short same-session sweeps show
  the batched paged-attention default should keep 4 chunks at the 16-block
  256-token window, use 2 chunks once `max_blocks_per_seq >= 64` for smaller
  batches, and keep 4 chunks for saturated batch-64+ long-context decode. At
  history 256, batch 8 measured 106.8 ms with 2 chunks vs. 106.1 ms with 4,
  and batch 64 measured 464.2 ms with 2 vs. 462.5 ms with 4. At history 1024,
  batch 32 measured 255.4 ms with 2 chunks vs. 260.7 ms with 4. At history
  2048, batch 8 measured 123.5 ms with 1 chunk, 112.1 ms with 2, 113.1 ms
  with 4, and 115.4 ms with 8. Patched no-override smoke checks measured
  history 2048 / batch 8 at 113.0 ms / 71 rows/s and history 256 / batch 64
  at 460.3 ms / 139 rows/s. A later history-1024 batch-64 repeat measured
  488.1 ms with 2 chunks vs. 486.5 ms with 4, so saturated long-context
  batches now stay on 4 chunks while smaller long-context batches keep 2.
  **Update — long-context split-K keeps 4 chunks from batch 16:** after the
  reduce pass stopped spending barrier levels on empty chunk lanes, the
  long-context crossover moved lower. At history 2048, warmup=2, timed=3,
  repeats=5, the old default 2-chunk policy measured batch 16 at 179.2 ms /
  89 rows/s and batch 32 at 274.1 ms / 117 rows/s. Forcing 4 chunks measured
  batch 16 at 154.7 ms / 103 rows/s and batch 32 at 263.5 ms / 121 rows/s;
  the patched no-override selector measured 157.7 ms / 101 rows/s and
  253.7 ms / 126 rows/s, respectively. Batch 8 remains on the smaller
  long-context 2-chunk policy because 4 chunks was effectively tied/slightly
  slower in the same sweep.
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
  **Update — Vulkan rowwise retry disabled by default:** the live decode worker
  now treats a failed Vulkan multi-row batch as a visible error instead of
  silently retrying one row at a time. That prevents hidden throughput collapse
  back to serialized decode while still allowing explicit debug opt-in with
  `KILN_VULKAN_DECODE_BATCH_ROWWISE_RETRY=1`.
  **Update — Vulkan generic batch fallback disabled by default:** the lower
  multi-row greedy helper now also errors if the native resident argmax route
  declines on Vulkan, instead of entering the generic hidden path that can
  rowwise full-attention work internally. Debug A/B runs can opt into that path
  with `KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK=1`. The serving batched-step
  wrapper uses the same gate, so it no longer swallows that error and continues
  into per-row decode by default.
  **Update — single-row decode fallback also visible by default:** the same
  native-required policy now covers bs=1 Vulkan decode entry points
  (`model_forward_paged`, `_last_token`, and greedy `_last_token_greedy`).
  If the resident token/argmax or resident logits path declines for an
  otherwise native-eligible decode step, the call now errors unless
  `KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK=1` is set for explicit A/B
  debugging. A short same-binary scheduler-width smoke also confirmed the
  existing default should stay at 64 rows: history 256, warmup=1, timed=2,
  repeats=1 measured batch 64 at 461.4 ms / 139 rows/s and batch 128 at
  936.8 ms / 137 rows/s.
  **Update — batched resident KV seeding:** when a resident batch contains
  multiple rows that still need prompt K/V copied into the Vulkan cache, the
  native argmax route now seeds the union of those rows' physical blocks once
  per full-attention layer instead of calling the seeding helper one row at a
  time. Stable row-ID batches still mark each row as seeded after the shared
  upload; no-ID multi-row batches remain conservatively re-seeded each call,
  but with one union upload per layer.
  **Update — hidden-path rowwise fallback disabled by default on Vulkan:** the
  older `model_forward_paged_batched_decode_hidden` path now treats a declined
  batched full-attention layer as a visible error on Vulkan instead of entering
  its rowwise full-attention fallback. The same
  `KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK=1` debug opt-in re-enables the
  fallback for A/B runs.
  **Update — resident hidden-output submit:** the native batched resident stack
  now has a hidden-output submit sibling to the final-argmax submit. It records
  the same full-attention/GDN stack, copies the final `[batch, hidden]` buffer
  into host-visible staging inside the same command batch, and
  `model_forward_paged_decode_contiguous_batch_hidden` tries it before the
  generic hidden path. This gives non-argmax callers a resident stack route and
  keeps hidden-path declines visible by default on Vulkan.
  **Update — multi-row sampling reaches resident hidden decode:** live Vulkan
  batches with mixed sampling params now snapshot per-row sampling context,
  assemble the same row-ID-keyed batched GDN state used by greedy decode, run
  `model_forward_paged_decode_contiguous_batch_hidden_with_ids`, scatter the
  updated resident state back to rows, and then sample from the returned hidden
  batch. Greedy still uses the token-only resident argmax route; non-greedy
  multi-row serving no longer has to enter the generic hidden stack first.
  **Update — Vulkan live decode batch default:** the live greedy decode batcher
  now defaults Vulkan to a 64-row max batch instead of 32; the real-model
  batching-engine actor applies the same backend-aware default when
  `KILN_MAX_DECODE_BATCH` is unset, and the live worker now also treats
  `KILN_MAX_DECODE_BATCH` as the shared width override unless its
  worker-specific `KILN_DECODE_BATCH_MAX` is set. Focused resident mixed-paged
  decode microbench on RADV STRIX_HALO, history 256, warmup=1, timed=4,
  repeats=3 measured batch 8 at 105.1 ms / 76 rows/s and batch 16 at
  142.7 ms / 112 rows/s. A second pass measured batch 16 at 148.4 ms /
  108 rows/s and batch 32 at 241.8 ms / 132 rows/s; a current batch-32 smoke
  measured 239.3 ms / 134 rows/s, and a current batch-64 smoke measured
  461.7 ms / 139 rows/s. The default now favors the highest-throughput
  saturated batch width, with env overrides still available for lower-latency
  or smaller-memory 16/32-row runs.
  **Update — live four-row batch smoke:** after rebuilding the current `kiln`
  release binary with `--no-default-features --features vulkan`, server startup
  with `KILN_NUM_BLOCKS=2048` and prefix cache disabled logged
  `max_decode_batch=64` for the batching actor and `max_batch=64` for the live
  greedy decode batcher. A `/v1/completions/batch` request with 4 distinct
  prompts, `temperature=0`, `top_p=1`, and `max_tokens=4` returned
  4 completions / 16 generated tokens in 26.068 s wall time. The server log
  reported a `Vulkan-resident decode pool ready` event with `num_slots=4`, so
  this smoke reached the live multi-row resident route; it is a short
  validation fixture, not a saturation benchmark.
  **Update — live sixteen-row warmed batch fixture:** a longer serving fixture
  on the same release binary used 16 distinct `/v1/completions/batch` prompts,
  `temperature=0`, `top_p=1`, `max_tokens=8`, prefix cache disabled, and
  default Vulkan batch ceilings. Startup logged `max_decode_batch=64`,
  `max_batch=64`, and prefix-cache `max_blocks=0`. The first request returned
  16 completions / 100 generated tokens in 86.971 s and constructed the
  resident pool mid-request (`Vulkan-resident decode pool ready`,
  `num_slots=4`). A warmed repeat with new distinct prompts returned
  16 completions / 96 generated tokens in 62.884 s; the server reported the
  batch endpoint at `duration_ms=62881.663398000004` with batching enabled for
  each completion. This is live-route evidence for the 64-wide default and a
  better long fixture than the four-row smoke, while the synthetic mixed-paged
  microbench remains the main saturation signal.
  **Update — kt-native Vulkan decode prewarm cleanup:** Vulkan decode-weight
  prewarm now fills the same kt TensorId-keyed F32/BF16-packed buffer caches
  used by resident decode, and the post-upload BF16 weight-stubbing path
  re-keys the packed cache plus any F32 shadow cache entry when it replaces
  local transposed weights with shape-preserving stubs. This removes the
  forbidden legacy-stack bridge from the Vulkan prewarm/stub cleanup path and
  keeps the duplicate-copy memory win aligned with the resident decode cache
  that production actually reads. Focused check: `cargo check -p kiln-model
  --no-default-features --features vulkan` passes with the repo's existing
  warning backlog.
  **Update — resident K/V seeding is named kt-native:** the Vulkan resident
  K/V seed helpers now advertise the actual dataflow: they read block ranges
  from `PagedKvCacheKt` pool tensors and upload those kt-derived F32 bytes into
  `VkPagedKvCache`. The old helper names implied a deprecated pool bridge even
  though the implementation was already kt-native, so the production call sites
  now use `seed_vk_kv_cache_layer_blocks_from_kt`. Focused check:
  `cargo check -p kiln-model --no-default-features --features vulkan` passes
  with the repo's existing warning backlog.
  **Update — resident pool allocated at startup:** server initialization now
  resolves the backend-aware batching actor width and live decode-batcher width,
  then eagerly calls the resident-pool feasibility/allocation path with their
  maximum before returning the real `AppState`. This moves the 4-slot pool
  allocation out of the first live request. Updated release boot on RADV
  STRIX_HALO with `KILN_NUM_BLOCKS=2048` and prefix cache disabled logged
  `Vulkan-resident decode pool ready` followed by
  `Vulkan resident decode pool startup allocation max_batch=64 ready=true`
  before `max_decode_batch=64` and live `max_batch=64` startup logs.
  **Update — default Vulkan decode-weight prewarm restored:** the server still
  skips synthetic token-generation prewarm when Vulkan-native training is
  enabled, but now starts the kt-native decode-weight prewarm in the background
  instead of marking prewarm complete immediately. Updated release boot on RADV
  STRIX_HALO logged `starting Vulkan decode weight prewarm`, uploaded
  `weights=96 f32_cache_mb=8640 bf16_packed_weights=249
  bf16_packed_cache_mb=8020 elapsed_ms=12232`, stubbed `248` local
  pre-transposed BF16 weight caches, then completed the background task in
  `elapsed_ms=26973`. That makes the kt-native cache/memory-residency cleanup
  active on the default Vulkan serving path without running a synthetic decode
  request at startup.
  A same-binary post-prewarm `/v1/completions/batch` fixture with 16 distinct
  prompts, `temperature=0`, `top_p=1`, `max_tokens=8`, prefix cache disabled,
  resident pool already allocated, and decode weights already prewarmed
  returned 16 completions / 120 generated tokens in 80.763 s wall time. The
  server reported `duration_ms=80758.438312` with batching enabled for each
  completion and no resident-pool allocation during the request. This validates
  the default live route after startup prewarm; because this prompt set
  generated more tokens than the prior 96-token warmed fixture, treat it as
  route evidence rather than a strict throughput A/B.
  **Update — live batch-width telemetry:** the batching actor now records
  persistent decode width counters in its snapshot, health/debug JSON, and
  Prometheus output. A rebuilt release server on RADV STRIX_HALO with
  `KILN_NUM_BLOCKS=2048`, prefix cache disabled, default Vulkan batch ceilings,
  startup resident-pool allocation `max_batch=64 ready=true`, and decode-weight
  prewarm complete at `elapsed_ms=23698` returned a 16-distinct-prompt
  `/v1/completions/batch` fixture in 58.408 s. Response usage was
  `prompt_tokens=325`, `completion_tokens=77`, `total_tokens=402`. An immediate
  `/metrics` scrape showed `kiln_batching_engine_max_observed_batch 16`,
  `kiln_batching_engine_decode_forwards_total 6`,
  `kiln_batching_engine_batched_decode_forwards_total 5`,
  `kiln_batching_engine_decode_rows_total 61`,
  `kiln_batching_engine_prefill_tokens_total 325`, and
  `kiln_batching_engine_errors_total 0`. `last_batch_size` was 1 because the
  final tail row finished after the multi-row steps; the max/counter metrics
  are the durable live proof that the default route issued true multi-row
  decode work.
  **Update — first-token timing breakdown:** the batching finish path now
  carries model prefill/decode durations through `BatchedGenerationOutput`,
  records them in recent-request rows, exposes them in opt-in chat performance
  metadata, and includes them in slow-request logs. A rebuilt release server on
  RADV STRIX_HALO with startup resident-pool allocation `max_batch=64
  ready=true` and decode-weight prewarm complete at `elapsed_ms=23921` ran an
  8-distinct-prompt `/v1/completions/batch` fixture after prewarm. The request
  returned 8 completions / 39 completion tokens in 29.975 s. Immediate
  `/v1/stats/recent-requests` rows showed shared `ttft_ms=28931`, per-row
  `model_prefill_ms` from 3568 to 3692 ms, and `model_decode_ms` from 715 to
  1035 ms; `/metrics` showed prefill histogram count 8 / sum 28.896 s and
  decode histogram count 8 / sum 6.703 s. The batch-width counters reported
  `kiln_batching_engine_max_observed_batch 8`,
  `kiln_batching_engine_batched_decode_forwards_total 4`,
  `kiln_batching_engine_decode_rows_total 31`, and
  `kiln_batching_engine_errors_total 0`. This makes the current first-token
  bottleneck concrete: the model kernels are warm and multi-row decode is
  active, but the first token waits for the actor to prefill/admit the full
  group before issuing the first batched decode step.
  **Update — bounded prefill admission before decode:** the batching actor now
  caps successful queued prefill admissions per scheduler cycle with
  `KILN_BATCH_PREFILL_ADMISSION_QUANTUM` (default 4 on non-Vulkan backends,
  clamped to the effective max decode batch). This keeps the Vulkan
  `KILN_MAX_DECODE_BATCH` default at 64 for saturation. A rebuilt release
  server on RADV STRIX_HALO with `KILN_NUM_BLOCKS=2048`, startup resident-pool
  allocation
  `max_batch=64 ready=true`, decode-weight prewarm complete at
  `elapsed_ms=26249`, and `KILN_BATCH_PREFILL_ADMISSION_QUANTUM=4` ran an
  8-distinct-prompt `/v1/completions/batch` fixture with `max_tokens=2`. The
  request returned HTTP 200 in 59.504 s with `prompt_tokens=192`,
  `completion_tokens=16`, and `total_tokens=208`. Recent-request rows showed
  the first four rows at `ttft_ms=43033` and the second four at
  `ttft_ms=59274`; per-row `model_prefill_ms` ranged 3940 to 5559 ms and
  `model_decode_ms` was 220 to 239 ms. The immediate `/metrics` scrape showed
  `kiln_batching_engine_prefill_admission_quantum 4`,
  `kiln_batching_engine_prefill_admission_cycles_total 2`,
  `kiln_batching_engine_max_observed_batch 8`,
  `kiln_batching_engine_decode_forwards_total 3`,
  `kiln_batching_engine_batched_decode_forwards_total 3`,
  `kiln_batching_engine_decode_rows_total 16`,
  `kiln_batching_engine_prefill_tokens_total 192`, and
  `kiln_batching_engine_errors_total 0`. This preserves the observed 8-row live
  decode width on the short fixture while replacing the prior full-group cold
  admission with two bounded prefill rounds.
  **Update — prefill first tokens emit during admission:** newly admitted real
  rows now flush their prefill-produced first token immediately instead of
  waiting for the actor to finish the whole prefill admission quantum and enter
  the next decode batch. Later model decode steps still use the same batched
  resident route, and first-token emission now shares the same EOS/stop/error
  handler as normal decode output. Focused checks: the Vulkan-profile
  `batching_engine::tests::` unit subset and
  `cargo check -p kiln-server --bins --no-default-features --features vulkan`
  pass with the repo's existing warning backlog.
  **Update — active rows yield prefill admission back to decode:** the batching
  actor now uses the full prefill admission quantum only when no row is ready
  for model decode. Once resident decode rows are ready, each scheduler pass
  admits at most one additional cold row before yielding back to the batched
  decode step. This keeps the 64-row Vulkan ceiling and the cold-start quantum,
  but prevents active rows from waiting behind another full prefill quantum.
  Focused checks: the Vulkan-profile `batching_engine::tests::` unit subset
  now covers the active-row cap and passes; `cargo check -p kiln-server --bins
  --no-default-features --features vulkan` also passes with the existing
  warning backlog.
  **Update — Vulkan admission quantum follows decode width:** server startup
  now passes the selected backend into the batching actor so Vulkan defaults
  `KILN_BATCH_PREFILL_ADMISSION_QUANTUM` to the effective max decode batch
  when the env override is unset. With the stock Vulkan decode width of 64, a
  cold burst can be admitted up to the resident batch width before the first
  model-decode step instead of being limited to four rows and then growing one
  row per decode loop. Non-Vulkan backends keep the smaller default, and any
  explicit `KILN_BATCH_PREFILL_ADMISSION_QUANTUM` value still wins and remains
  clamped to the active decode width. Focused checks:
  `cargo test -p kiln-server --features vulkan --lib
  prefill_admission_quantum_default_and_override -- --nocapture`,
  `cargo test -p kiln-server --features vulkan --lib
  enqueue_batches_forward_shape_and_routes_responses -- --nocapture`, and
  `cargo check -p kiln-server --features vulkan --lib` pass with the existing
  warning backlog. A package-wide `kiln-server` test compile is still blocked
  by an unrelated `real_model_integration` match-exhaustiveness error.
  **Update — resident parity test restored for Vulkan profiles:** the
  `vk_resident_decode_parity` integration test now builds against the current kt
  device/cache surface instead of stale test-only APIs. Without
  `KILN_RESIDENT_DECODE_PARITY_MODEL` it still skips at runtime, but both the
  specific integration test and the broader filtered Vulkan test command now
  compile the parity target successfully. This restores a usable correctness
  gate for the resident decode path when the local Qwen3.5-4B model env is set.
  **Update — bs=1 greedy token-only route:** `model_forward_paged_last_token_greedy`
  now tries the resident transformer-stack + final argmax path before the older
  resident logits fallback. For callers without stable row IDs, bs=1 uses the
  existing start-position session tracker instead of the multi-row no-ID
  conservative re-seed-every-call policy, so prompt K/V is seeded once per
  single-row session. Direct Vulkan-only `full_token_resident_mixed_paged`
  smoke with history 256, batch 1, warmup=1, timed=4, repeats=3 measured
  59.0 ms / 17 rows/s. This validates the kernel route reached by serving.
  The app-level paged latency smoke at the top of this note measured p50 decode
  ITL 68.8 ms after one warmup pass, with first-use pipeline cost still visible
  in the first decode step.
  **Update — single-row on-device sampler is one-submit end-to-end:** the
  Vulkan `linear_decode_sample` helper now records hidden upload, optional
  token-history uploads, LM-head projection, optional token-penalty scatter,
  fused top-k/top-p sampling, and the 4-byte token readback copy into one
  `CommandBatch`. The sampled token is then read from host-visible staging, so
  the non-greedy bs=1 route no longer pays separate queue submissions for
  upload, LM head, penalties, sampler, and token readback. Focused Vulkan coverage:
  `cargo test -p kiln-vulkan-kernel --test linear_decode_sample -- --nocapture`
  covers both the no-penalty top-1 path and the optional penalty dispatch.
  The helper now also packs the hidden row and optional token-history arrays
  into one host-visible upload staging buffer and records offsetted copies from
  that buffer, avoiding separate map/allocation setup for each small upload.
  Those upload copies now share one transfer-to-compute barrier before LM-head
  dispatch instead of one barrier per staged segment.
  Command batches that end with an in-batch readback copy now also skip the
  redundant submit-time shader tail barrier; the copy already records the
  shader-to-transfer and transfer-to-host barriers needed for the mapped token.
  **Update — legacy bridge removed from the old bs=1 resident block helper:**
  the older single-row resident full-attention fallback now extracts the input
  activation and RoPE tables directly from kt tensors, uploads those f32 slices
  to Vulkan, and reconstructs the readback as a kt tensor. The default serving
  route remains the native whole-token resident stack above; this cleanup keeps
  the older helper aligned with the kt/Vulkan-only constraint instead of
  crossing the deprecated tensor bridge.
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
  one-submit benchmark for a full Qwen3.5-4B GDN block plus MLP. It records 9
  shaders into one `CommandBatch`: RMSNorm, row-reuse GDN in-proj, fused
  conv/split/state advance, fused Q/K L2 expansion, recurrent gate/RMSNorm, GDN
  out-proj, fused residual+post norm, MLP gate/up, and fused down+residual. On RADV
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
  146.6 ms in a same-session A/B. The post-GDN residual add and post-attention
  RMSNorm are now fused through the same batched add+RMSNorm shader used by the
  full-attention path, removing one recorded dispatch per GDN layer. Current
  smokes measured the 9-dispatch GDN block at batch 32: 6.79 ms / 4,712 rows/s
  and batch 64: 14.05 ms / 4,554 rows/s; the full mixed-paged token measured
  batch 32: 236.4 ms / 135 rows/s and batch 64: 463.1 ms / 138 rows/s.
  A follow-up row-grouping A/B showed the GDN in-proj rows4 shader starts too
  early at batch 8 on STRIX_HALO: standalone in-proj moved from 1.23 ms to
  1.07 ms when using rows2, and the full GDN block moved from 3.30 ms to
  3.11 ms. Rows4 still wins from batch 16 upward, so the rows4 crossover is
  now batch 16. Patched mixed-paged token smokes at history 256, warmup=1,
  timed=3, repeats=2 measured batch 8: 101.6 ms / 79 rows/s, batch 16:
  139.1 ms / 115 rows/s, batch 32: 245.8 ms / 130 rows/s, and batch 64:
  455.9 ms / 140 rows/s.
  **Update — bf16 output projection rows4 at batch 16:** the ordinary
  batched bf16 linear-output planner now joins the residual/down planner in
  selecting rows4 from batch 16. Direct `linear_decode` on the Q-out/GDN-out
  shape moved batch 16 from 570.5 us to 541.9 us in a same-session A/B, while
  batch 32/64 stayed flat. In the full mixed resident-paged token route at
  history 256, warmup=1, timed=3, repeats=3, the same-session baseline
  measured batch 16 at 141.1 ms / 113 rows/s, batch 32 at 241.0 ms /
  133 rows/s, and batch 64 at 453.2 ms / 141 rows/s; patched rerun measured
  batch 16 at 138.2 ms / 116 rows/s, batch 32 at 240.5 ms / 133 rows/s, and
  batch 64 at 451.8 ms / 142 rows/s.
  A follow-up recorder cleanup skips the redundant memory barrier between the
  independent Q RoPE and K RoPE dispatches in full-attention blocks. Focused
  history-256 mixed-paged smokes with warmup=1, timed=4, repeats=3 measured
  batch 8: 101.0 ms / 79 rows/s, batch 32: 234.9 ms / 136 rows/s, and batch
  64: 461.5 ms / 139 rows/s; the effect is small because full-attention is only
  8 of the 32 Qwen3.5 layers, but it removes 8 conservative barriers per token.
  A direct full-attention block A/B also showed the fused QKV+gate rows4
  projection should keep the same batch-2 cutoff as the older combined-QKV
  path: with rows4 at batch 1 the block measured 1.847 ms, while forcing the
  regular row path measured 1.771 ms; batch 2 still favored rows4 at 1.861 ms
  vs. 2.012 ms.
  A current mixed-paged smoke after the later barrier cleanups and batch-2
  direct-output QKV+gate cutoff measured history 256, warmup=1, timed=3,
  repeats=2 at batch 8: 101.8 ms / 79 rows/s, batch 32: 236.5 ms / 135 rows/s,
  and batch 64: 463.6 ms / 138 rows/s.
  **Update — GDN Q/K L2 expansion no longer recomputes replicated rows:** the
  fused Q/K L2 expansion shader now launches one workgroup per input key head
  row and writes all GQA-replicated output rows from that single reduction.
  The previous dispatch launched one workgroup per expanded value-head row,
  so Qwen3.5's GQA ratio 2 recomputed the same norm twice. Focused
  `gdn_block_resident_batched` checks with warmup=2, timed=5, repeats=2
  measured batch 8 at 2.966 ms / 2697 rows/s, batch 16 at 4.020 ms /
  3980 rows/s, batch 32 at 6.616 ms / 4837 rows/s, and batch 64 at
  13.706 ms / 4669 rows/s. A production-shaped mixed-paged token smoke at
  history 256, warmup=1, timed=4, repeats=3 measured batch 8 at 96.6 ms /
  83 rows/s, batch 32 at 226.2 ms / 141 rows/s, and batch 64 at 452.9 ms /
  141 rows/s.
  **Update — GDN recurrent output avoids extra Q read:** the fused
  gates/recurrent/RMSNorm shader now computes `q dot new_state` as
  `q dot (decay * old_state) + delta * (q dot k)`, leaving the state-write loop
  to update recurrent state without also rereading Q and accumulating the
  output. Focused correctness check
  `cargo test -p kiln-vulkan-kernel --test gdn_parity
  gdn_decode_gates_recurrent_rmsnorm -- --nocapture` passes. Current
  `gdn_block_resident_batched` with warmup=2, timed=5, repeats=3 measured
  batch 8 at 2.968 ms / 2696 rows/s, batch 16 at 4.013 ms / 3987 rows/s,
  batch 32 at 6.610 ms / 4841 rows/s, and batch 64 at 13.634 ms /
  4694 rows/s. A production-shaped `full_token_resident_mixed_paged` check
  with history 256, warmup=2, timed=4, repeats=3 measured batch 32 at
  233.8 ms / 137 rows/s and batch 64 at 451.9 ms / 142 rows/s.
  **Update — resident scratch pool drops dead MLP outputs:** the single-row
  full-attention and GDN resident recorders no longer acquire unused
  `nfa_mlp_out` / `ngd_mlp_out` scratch buffers. The fused down+residual
  dispatches write final block outputs directly, so these pool slots were
  never bound. Focused check: `cargo check -p kiln-model
  --no-default-features --features vulkan` passes and the two stale resident
  scratch warnings are gone.
  **Update — native decode stale helper cleanup:** the native bs=1 resident
  orchestrator no longer computes an unused full-attention-layer count and no
  longer carries the unused device-local u32 upload helper from before block
  tables moved to persistent host-visible scratch. The same Vulkan-profile
  `cargo check` now leaves no warnings from `vk_decode_resident.rs` or the
  native decode helper body.
  **Update — resident RoPE setup avoids temporary tensor tables:** the native
  resident single-row route plus the batched argmax/hidden routes now build
  per-row RoPE cos/sin tables directly into host f32 slices from
  `start_positions`/`start_pos` and `rope_theta`, then upload those slices with
  the rest of the resident step metadata. This removes the prior temporary
  tensor table build plus flatten/readback step before the one-submit Vulkan
  transformer stack. Focused check:
  `test_vulkan_resident_host_rope_tables_match_tensor_tables` passes under
  `cargo test -p kiln-model --lib --no-default-features --features vulkan`,
  and `cargo check -p kiln-model --no-default-features --features vulkan`
  passes with the existing warning backlog.
  **Update — resident embedding setup bypasses tensor gather:** the same
  resident single-row and batched decode entry points now gather token
  embeddings directly from CPU-resident F32/BF16/F16 embedding tables into the
  f32 upload rows consumed by the Vulkan resident stack. On the Vulkan weight
  layout this reads the transposed table and avoids building a temporary
  gather/transpose tensor only to flatten it back to host memory before upload.
  Focused checks: the Vulkan-profile `test_vulkan_resident_` unit subset
  covers the RoPE and BF16 transposed embedding helpers and passes; `cargo
  check -p kiln-model --no-default-features --features vulkan` also passes
  with the existing warning backlog.
- **Vulkan paged-attention decode kernel**: the kernel crate already had
  `paged_attn_decode_batch_paged.comp`; Vulkan now advertises
  `supports_flash_attn_paged_decode` and wires the single-query paged-decode
  trait call through the GPU-side block-table walker from kt tensors, so the
  long-context non-contiguous decode path no longer has to fall straight to the
  manual GQA attention path. Remaining saturation work is the multi-row
  resident decode orchestration below, not the existence of the paged attention
  kernel.
  **Update — generic single-row Vulkan route skips compact-probe:** when the
  fused paged-decode helper is reached, Vulkan now goes straight to the
  block-table paged kernel instead of first probing the contiguous/prefill-style
  branch that can build compact K/V views before declining. Resident decode
  remains the preferred serving route; this keeps the generic long-context
  fallback aligned with the paged-kernel objective.
  **Update — generic Vulkan paged decode uses split-K:** the raw paged-attention
  wrapper now exposes the same split-K scan + reduce shaders used by resident
  decode, with the shared chunk policy centralized in the kernel crate. Both
  generic paged decode entry points route through that wrapper and the dynamic
  per-row-length path now extracts kt bytes directly instead of compacting K/V
  through the forbidden legacy stack before dispatch. Added
  `paged_attn_splitk_check`, a kernel-crate-only Vulkan probe comparing the
  split-K wrapper with the non-split paged wrapper on a non-contiguous
  block table and reporting median wrapper timings. Default small-shape release
  run measured max abs diff `2.980232e-8`; batch 1, 2048-token history
  measured non-split 5.241 ms vs. split-K 4.013 ms (1.31x) with max abs diff
  `1.713634e-7`. Batch 8, 1024-token history measured non-split 16.109 ms vs.
  split-K 16.574 ms (0.97x), confirming that the shared chunk policy's main
  generic-path win is the long single-row occupancy case while saturated
  multi-row resident decode remains the throughput path. The generic
  app-facing helper now follows that measurement: split-K is default for
  single-row generic paged decode, while multi-row generic decode stays on the
  non-split paged wrapper unless `KILN_VK_PAGED_ATTN_SPLITK_CHUNKS` explicitly
  forces a split-K sweep.
  **Update — generic multi-row paged decode adopts the shared split-K
  selector:** after the reduce pass started skipping empty chunk-sum barrier
  levels, the older multi-row generic exclusion is no longer correct. The
  generic helper now uses `paged_attn_decode_splitk_chunks` for all batch
  sizes, so it follows the same long-context policy as resident decode while
  still honoring `KILN_VK_PAGED_ATTN_SPLITK_CHUNKS`. Current
  `paged_attn_splitk_check` probes measured batch 8 / history 1024 at
  non-split 15.406 ms vs. split-K 15.018 ms, batch 8 / history 2048 at
  37.952 ms vs. 36.719 ms, and batch 16 / history 2048 at 72.457 ms vs.
  58.201 ms, all with max abs diff below `2.5e-7`.
  **Update — generic Vulkan paged decode accepts non-contiguous block tables:**
  the shared single-row paged helper no longer applies the physical
  intra-chunk contiguity precheck before calling Vulkan's block-table gather
  kernel. That precheck belongs to backends whose paged kernel assumes compact
  physical pages; Vulkan now reaches `flash_attn_paged_decode` for arbitrary
  valid block tables instead of declining into the materialized attention path.
  Current kernel-crate smoke on the non-contiguous default shape measured
  max abs diff `2.980232e-8`, non-split `0.210 ms`, split-K `0.225 ms`
  (`0.93x`) on RADV STRIX_HALO.
  **Update — paged-attention declines are visible by default on Vulkan:** once
  a Vulkan decode step is eligible for the native paged-attention path, a
  kernel decline now errors instead of silently materializing K/V and running
  fallback attention. The existing
  `KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK=1` debug opt-in re-enables the
  fallback for A/B runs.
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

  Current smoke after tightening the Vulkan decline gates used
  `KILN_VK_MICROBENCH_BATCHES=1,8,32,64`, warmup=1, timed=3,
  repeats=2. The selector env was intentionally scoped to resident paged
  decode, but this binary ran the full decode sweep; only the resident paged
  rows are recorded here. On RADV STRIX_HALO:

  | case | batch | per token | rows/s |
  |---|---:|---:|---:|
  | real layer mix, paged KV + split-K paged attention | 1 | 57.0 ms | 18 |
  | real layer mix, paged KV + split-K paged attention | 8 | 99.2 ms | 81 |
  | real layer mix, paged KV + split-K paged attention | 32 | 246.8 ms | 130 |
  | real layer mix, paged KV + split-K paged attention | 64 | 463.2 ms | 138 |
  | synthetic all full-attention, paged KV + split-K paged attention | 1 | 49.9 ms | 20 |
  | synthetic all full-attention, paged KV + split-K paged attention | 8 | 104.2 ms | 77 |
  | synthetic all full-attention, paged KV + split-K paged attention | 32 | 271.0 ms | 118 |
  | synthetic all full-attention, paged KV + split-K paged attention | 64 | 509.0 ms | 126 |

  The microbench selector now also honors `KILN_VK_MICROBENCH_ONLY`, so
  future resident/paged checks can avoid heating the GPU with sibling sweeps.
  A focused selector smoke with batch 1, warmup=1, timed=1, repeats=1 ran only
  `full_token_resident_paged` and measured 49.4 ms / 20 rows/s.
  **Update — split-K reduce scale reuse:** the split-K paged-attention reduce
  shader now computes each chunk's `exp(chunk_max - combined_max)` scale once
  per workgroup, stores it in shared memory, and reuses it across all output
  lanes. The selector still caps explicit split-K chunk sweeps to the 256-lane
  workgroup size. On RADV STRIX_HALO, the focused
  `paged_attn_splitk_check` probe at batch 1, 32 query heads, 8 K/V heads,
  head_dim 128, 2048-token history, chunks 32, warmup=1, iters=3, repeats=2
  measured max abs diff `1.713634e-7`, non-split 4.783 ms, split-K 3.037 ms
  (**1.57x**). A resident full-token paged smoke running only
  `full_token_resident_paged` at history 2048, batch 1, warmup=1, timed=2,
  repeats=2 measured 54.620 ms / 18 rows/s for the 32-layer paged route.
  **Update — split-K reduce skips empty barrier levels:** the split-K reduce
  shader now starts its scalar `chunk_sum` reduction at the first power-of-two
  stride needed for `num_chunks` instead of always starting at 128. The common
  resident cases therefore reduce 4 chunks with strides 2/1 and single-row
  32-chunk generic decode with 16/8/4/2/1, avoiding empty shared-memory
  barriers. A same-session batch-1 `paged_attn_splitk_check` A/B at 2048-token
  history, 32 query heads, 8 K/V heads, head_dim 128, chunks 32, warmup=2,
  iters=5, repeats=5 moved split-K from 3.702 ms to 2.804 ms with max abs diff
  `1.713634e-7`. The production-shaped attention probe at batch 8, 16 query
  heads, 4 K/V heads, head_dim 256, 256-token history, chunks 4 measured
  non-split 2.894 ms vs. split-K 2.815 ms with max abs diff `3.352761e-8`.
  A full mixed resident-paged token smoke at history 256, warmup=1, timed=4,
  repeats=3 measured batch 8 at 96.7 ms / 83 rows/s, batch 32 at 240.5 ms /
  133 rows/s, and batch 64 at 453.5 ms / 141 rows/s; the full token remains
  dominated by projection/GDN work, but the paged-attention reduce no longer
  spends barriers on lanes that cannot contain chunk sums.
  **Update — shared multi-output readbacks use one staging buffer:** the
  single-submit helper paths that produce multiple host results now pack their
  device-to-host copies into one host-visible staging buffer instead of one
  staging allocation/map per output. This covers the cached GDN gates helper and
  the two-dispatch helper used by causal-conv prefill/update and split-K paged
  attention wrapper checks. The same helpers also pack their multiple
  host-to-device inputs into one upload staging buffer while preserving the same
  recorded copy offsets before dispatch, and that packing now maps the staging
  allocation once instead of first building an intermediate packed host vector.
  Focused coverage:
  `gdn_gates_cached_bytes_matches_cpu_reference` and
  `causal_conv1d_prefill_matches_stateful_cpu_reference` pass on Vulkan hardware.

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
  **Update — batched Vulkan sampler path:** multi-row non-greedy resident
  decode now has a backend hook for final-normed hidden rows that records
  batched LM-head projection, row-specific token penalties, per-row top-k /
  top-p / min-p sampling, and token readback in one Vulkan command batch. The
  path reads back only one `u32` token per row when every active row is within
  the sampler kernel's top-k limit, with existing logits fallback preserved for
  unsupported sampler settings.
  **Update — resident decode now chains into batched sampling:** the Vulkan
  multi-row non-greedy branch now tries a resident decode-to-sampler helper
  before the hidden-row fallback. The new helper records the batched
  transformer stack, final RMSNorm, BF16 LM head, optional row-specific token
  penalties, top-k/top-p/min-p sampling, and token-id output in one
  `CommandBatch`, so the hot path no longer has to read final hidden rows back
  just to upload them into the sampler path again. Unsupported sampler settings
  still decline into the existing hidden/logits fallback.
  **Update — resident split-K paged attention has one recorder:** the Vulkan
  kernel crate now exposes a resident split-K paged-attention recorder/wrapper,
  and the model resident decode stack routes both single-row and multi-row
  full-attention blocks through it. The resident KV-cache roundtrip test now
  checks the split-K resident wrapper against the same CPU softmax reference as
  the non-split resident path, keeping the long-context decode kernel wiring
  covered at the kernel boundary used by the serving stack.
  **Update — resident sample errors do not replay the step:** the multi-row
  non-greedy branch now only falls back to the hidden-row sampler when the
  resident sample path cleanly declines before native command recording.
  Once the resident path has selected native execution, an error returns
  immediately instead of replaying the same decode step through the hidden
  fallback after KV/GDN state may have been updated on device.
  **Update — single-row stochastic decode uses resident sampler first:** the
  row-count-1 non-greedy Vulkan serving path now tries the same resident
  decode-to-sampler helper as continuous batches, so supported top-k/top-p/min-p
  settings read back one sampled token instead of materializing full logits for
  host-side sampling. Unsupported sampler settings still fall back to the older
  resident-logits route.
  **Update — single-row sample skips batch-state cat/scatter:** when that
  row-count-1 resident sampler path has GDN layers, it now passes the existing
  per-row `LinearAttentionState` directly into the resident stack instead of
  assembling a synthetic batch with `from_batch_rows`, then scattering it back
  after sampling. The resident buffers are already keyed by the row tensors, so
  this removes the extra CPU-side cat/scatter cycle from stochastic bs=1 decode.
  **Update — single-row hidden fallback skips batch-state cat/scatter:** the
  unsupported-sampler hidden fallback now mirrors the resident sampler path for
  row-count 1. It passes the row's `LinearAttentionState` directly into paged
  decode instead of building a synthetic batch and scattering it back. This
  keeps uncommon sampler fallback settings from reintroducing the per-step
  24-layer state cat/scatter cost on bs=1.
  **Update — resident token embedding skips an independent barrier:** the
  token-id resident paths now record RoPE table generation followed by the
  embedding gather with `record_shader_no_previous_barrier` for the embedding
  dispatch. Those two writes are independent, and the following transformer
  stack dispatch still emits the visibility barrier before reading both
  buffers, removing one conservative compute-to-compute barrier per token
  batch across argmax, sampling, and hidden-output routes.
  **Update — greedy LM-head argmax reuses BF16 weights across four rows:** the
  batched BF16 argmax block stage now has a rows4 shader for larger batches.
  It computes the same per-row/per-vocab-block score and index buffers as the
  existing one-row shader, so the reduce stage is unchanged, but each packed
  weight load is shared across up to four adjacent decode rows. The resident
  greedy path selects this rows4 block stage at the same batch threshold as the
  sampled LM-head projection. Focused Vulkan coverage:
  `batched_bf16_argmax_rows4_matches_cpu_with_tail_rows` passes with a
  17-row tail batch and partial final vocab block.
  **Update — wide batched BF16 projections use rows8 at saturation:** the
  resident batched linear BF16 helper and both LM-head token paths now select
  rows8 kernels once a batch reaches the 64-wide serving saturation point. The
  sampled path reuses the existing rows8 projection shader, while greedy argmax
  adds a rows8 block-score shader that preserves the existing reduce stage.
  Focused Vulkan coverage now includes 65-row tail batches for greedy argmax
  and top-1 sampling.
  **Update — non-recorded resident linear helper matches rows8 routing:** the
  direct resident BF16 linear helper now selects the same rows8 projection
  shader as the generic bytes helper and recorded model stack at batch 64. A
  focused `full_step_resident` batch-64 microbench measured 11.98 ms by
  default vs 12.29 ms with rows8 disabled, and resident parity now covers a
  65-row tail batch through the helper path.
  **Update — full-attention QKV/gate projection uses rows8 at saturation:** the
  resident full-attention block now selects a rows8 packed-BF16 QKV/gate split
  projection for 64-wide continuous batches. This shares each Q/K/V weight load
  across up to eight active rows before writing directly into the resident Q,
  gate, K, and V buffers, while keeping the rows4 path for smaller batches.
  The Vulkan decode microbench planner uses the same threshold, and
  `direct_full_attn_qkv_gate_split_rows8_matches_cpu` covers a tail-row batch.
  The rows8 crossover is now runtime-tunable with
  `KILN_VULKAN_FULL_ATTN_QKV_BF16_ROWS8_MIN_BATCH`, defaulting to the same
  batch-64 Strix Halo setting. This mirrors the MLP, generic linear, and GDN
  in-proj threshold knobs while preserving the current measured default.
  The rows4 crossover is now tunable too via
  `KILN_VULKAN_FULL_ATTN_QKV_BF16_ROWS4_MIN_BATCH`, defaulting to batch 2
  because direct block A/B on this APU showed batch 1 slower on rows4 while
  batch 2+ favored it.
  **Update — GDN input projection rows8 is opt-in pending better hardware
  data:** the resident GDN block has a rows8 packed-BF16 input-projection
  shader that keeps QKV/Z column pairing and shares each packed weight load
  across up to eight active rows before writing the packed projection layout
  consumed by the existing conv/split stage. On Strix Halo, rows4 remains
  faster at batch 64, 128, and 256, so the default planner stays on rows4 and
  rows8 requires `KILN_ENABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_OCTET=1` for further
  experiments. Focused `gdn_in_proj` microbench numbers on this APU were:
  rows4/default 4.56 ms vs rows8/opt-in 5.53 ms at batch 64, 9.25 ms vs
  11.02 ms at batch 128, and 21.17 ms vs 23.16 ms at batch 256. The
  paired-column shaders now also load the second column from
  its actual packed word, which keeps odd-width projection coverage correct.
  Focused Vulkan coverage:
  `gdn_in_proj_rows8_matches_cpu_with_tail_rows_and_odd_pairs`.
  The GDN in-proj rows4/rows8 cutoffs are now runtime-tunable with
  `KILN_VULKAN_GDN_IN_PROJ_ROWS4_MIN_BATCH` and
  `KILN_VULKAN_GDN_IN_PROJ_ROWS8_MIN_BATCH`, defaulting to the Strix Halo
  values 16 and 64. Rows8 remains opt-in through
  `KILN_ENABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_OCTET`, but the model recorder,
  kernel dispatcher, direct resident helper, and decode microbench now share
  the same thresholds. Direct resident parity now covers a 65-row tail batch,
  and the rows8 opt-in validation uses the same test with the octet env set.
  **Update — MLP rows8 crossover is runtime-tunable, defaulting to batch 256:**
  the model recorder, kernel helper, resident helper, and decode microbench now
  all read `KILN_VULKAN_MLP_BF16_ROWS8_MIN_BATCH` before choosing the full-BF16
  MLP rows8 shaders. A steadier same-session mixed-paged token check kept the
  Strix Halo default at 256: batch 64 measured 465.0 ms with a 64-row crossover
  vs 446.2 ms with the 256-row crossover. The override remains available for
  devices where the measured crossover differs. The MLP rows4 crossovers are
  now tunable as well: `KILN_VULKAN_MLP_BF16_GATE_UP_ROWS4_MIN_BATCH` defaults
  to 8, `KILN_VULKAN_MLP_BF16_DOWN_ROWS4_MIN_BATCH` defaults to 16, and
  `KILN_VULKAN_MLP_F32_DOWN_ROWS4_MIN_BATCH` defaults to 8, preserving the
  measured Strix Halo choices while letting other Vulkan devices sweep them.

## Other follow-ups (perf headroom, not regressions)

1. **Batch-admission/TTFT policy live validation** remains the next serving
   latency target: first-token emission and active-row admission yielding are
   now implemented, but the warmed multi-row fixture should be rerun to measure
   how much wall-clock TTFT moved under real request timing while preserving
   the 64-wide saturation path.
2. Complete the remaining shared-stack dependency cleanup audit for any
   non-Vulkan decode islands; the Vulkan-specific decode weight path is now
   duplicate-copy-free.
