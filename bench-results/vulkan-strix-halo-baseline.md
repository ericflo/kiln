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
- **True multi-row batched resident decode** (bs>1 / continuous-batched is
  currently rowwise-serialized through the fast bs=1 path).
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
- **Vulkan paged-attention decode kernel**: the kernel crate already had
  `paged_attn_decode_batch_paged.comp`; Vulkan now advertises
  `supports_flash_attn_paged_decode` and wires the single-query paged-decode
  trait call through the GPU-side block-table walker from kt tensors, so the
  long-context non-contiguous decode path no longer has to fall straight to the
  manual GQA attention path. Remaining saturation work is the multi-row
  resident decode orchestration below, not the existence of the paged attention
  kernel.

## Other follow-ups (perf headroom, not regressions)

1. **First-token shader compile** (~1.4 s) could be prewarmed at load.
2. The Vulkan build still links candle via `kiln-model`'s shared
   `candle-core`/`candle-nn` deps (the same islands the CUDA DoD-100 work is
   removing); the Vulkan-specific decode weight path is now candle-copy-free.
