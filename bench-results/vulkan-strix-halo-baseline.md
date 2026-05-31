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

The **GDN linear-attention prefill dominates** (~75% of prefill). The GDN
recurrent/chunkwise scan is the hot path: the BF16-gated `recurrent_unexpanded_qk`
native prefill fast path does NOT engage on Vulkan (Vulkan activations are F32,
the gate requires BF16 input), so GDN prefill takes the chunkwise/portable scan.
Speeding this up (a Vulkan-native chunkwise GDN prefill, or an F32 path for the
native recurrent prefill) is a focused kernel/algorithm work-package — the single
biggest remaining Vulkan perf win. NOT attempted yet: it touches the working GDN
forward and warrants careful parity validation, not an end-of-session change.

## Other follow-ups (perf headroom, not regressions)

1. **First-token shader compile** (~1.4 s) could be prewarmed at load.
2. The Vulkan build still links candle via `kiln-model`'s shared
   `candle-core`/`candle-nn` deps (the same islands the CUDA DoD-100 work is
   removing); the Vulkan-specific decode weight path is now candle-copy-free.
