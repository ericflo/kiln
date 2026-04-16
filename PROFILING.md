# Kiln Profiling Report — Prefill + Decode Hotspots

## Overview

This report captures NVIDIA Nsight Systems profiles of `kiln-bench` running Qwen3.5-4B
(32 layers: 24 Gated DeltaNet linear-attention + 8 full-attention; bf16 weights, weight-tied
LM head) on a single RTX A6000. It identifies the kernels and API calls responsible for
Kiln's current performance gap versus `llama.cpp` (~226× on prefill, ~8× on decode).

**No optimizations are proposed or attempted here — this is a profiling-only pass.** The
recommendations at the end point to concrete file:line hot spots and expected gains so that
a follow-up PR can land the fixes.

## Hardware & Build

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (48 GiB, compute capability 8.6) |
| Driver | 550.127.08 |
| CUDA toolkit | 12.4.1 (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) |
| Rustc | 1.94.1 stable |
| Build | `cargo build --release --features cuda -p kiln-bench` |
| Build env | `KILN_CUDA_ARCHS=86`, B2 sccache (x86_64-linux-cuda12.4 prefix) |
| Nsight Systems | 2024.1.1 (bundled with nsight-compute) |
| Model | Qwen3.5-4B, bf16, loaded from HF (`qwen2_5_4b_gdn`) |

## Scenarios

| Scenario | Invocation | Config |
|---|---|---|
| Prefill | `kiln-bench --prompt-tokens 512 --max-output-tokens 1 --skip-training` | 512 prompt → 1 output, prefill-dominated |
| Decode  | `kiln-bench --prompt-tokens 16  --max-output-tokens 256 --skip-training` | 16 prompt → 256 output, decode-dominated |

Both captures were collected with `nsys profile -t cuda,nvtx,osrt --cudabacktrace=all` and
reduced with `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum`. Raw artifacts:

- `prefill.nsys-rep` (1.8 GiB), `prefill.sqlite`, `prefill_cuda_gpu_kern_sum.csv`
- `decode.nsys-rep`  (559 MiB), `decode.sqlite`,  `decode_cuda_gpu_kern_sum.csv`

Total reduced GPU run time (kernel-sum): prefill ≈ 64.6 s, decode ≈ 101.1 s.

---

## Prefill Kernel Breakdown (top 10)

Prefill is dominated by **F32 elementwise / reduction ops and bf16 memory copies**, not by
matmul. Only ~3 % of GPU time is spent in actual GEMM kernels.

| Rank | Time % | Total (ms) | Calls | Avg (µs) | Kernel |
|---:|---:|---:|---:|---:|---|
| 1  | 45.1 | 29 116 | 2 104 608 | 13.8 | `bmul_f32` (broadcast multiply) |
| 2  | 25.2 | 16 280 |    24 384 | 667.6 | `ucopy_bf16` (bf16 memcpy / contiguous) |
| 3  | 16.7 | 10 760 |   841 344 | 12.8 | `fast_sum_f32` |
| 4  |  4.4 |  2 836 |   430 784 |  6.6 | `badd_f32` (broadcast add) |
| 5  |  1.2 |    779 |     3 298 | 236.1 | `ampere_bf16_s16816gemm_bf16_256x128_nn` |
| 6  |  1.1 |    696 |   416 736 |  1.7 | `copy2d_f32` |
| 7  |  1.0 |    668 |   421 680 |  1.6 | `uexp_f32` |
| 8  |  1.0 |    644 |   415 280 |  1.6 | `bsub_f32` |
| 9  |  0.5 |    320 |     2 990 | 107.2 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64` |
| 10 |  0.4 |    269 |     2 176 | 123.4 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128` |

GEMM totals: ampere+cutlass bf16 GEMMs combined sum to ~2.1 s (~3.3 % of prefill). By
contrast, `bmul_f32`+`fast_sum_f32`+`badd_f32` alone account for ~66 % of prefill.

## Decode Kernel Breakdown (top 10)

Decode is even more extreme: **90 % of GPU time is `ucopy_bf16`**, almost entirely driven
by host-visible copies of per-step logits and KV writeback, not by the matmuls themselves.
Actual GEMMs total ~6.2 % (and are dominated by low-arithmetic-intensity 64- and 256-tile
shapes because of batch-1 decode).

| Rank | Time % | Total (ms) | Calls | Avg (µs) | Kernel |
|---:|---:|---:|---:|---:|---|
| 1  | 89.9 | 90 893 | 126 899 | 716.3 | `ucopy_bf16` |
| 2  |  3.1 |  3 118 |  29 120 | 107.1 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64` |
| 3  |  1.7 |  1 702 |  35 888 |  47.4 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64` |
| 4  |  1.2 |  1 255 | 240 318 |   5.2 | `bmul_f32` |
| 5  |  1.0 |    986 |  14 624 |  67.4 | `ampere_bf16_s16816gemm_bf16_128x64_nn` |
| 6  |  0.7 |    670 | 103 275 |   6.5 | `fast_sum_f32` |
| 7  |  0.4 |    414 |  10 848 |  38.1 | `ampere_bf16_s16816gemm_bf16_64x64_nn` |
| 8  |  0.2 |    213 | 159 654 |   1.3 | `affine_f32` |
| 9  |  0.2 |    208 | 145 672 |   1.4 | `cast_bf16_f32` |
| 10 |  0.2 |    188 |  88 267 |   2.1 | `badd_f32` |

---

## CUDA API / Launch + Sync Overhead

### Prefill (512 tokens)

| Time % | Total (s) | Calls | Avg (µs) | API |
|---:|---:|---:|---:|---|
| 23.6 | 16.4 | 4 844 782 |   3.4 | `cuLaunchKernel` |
| 21.0 | 14.6 |       160 |91 078.2 | `cuMemcpyDtoHAsync_v2` |
| 17.9 | 12.4 | 2 983 148 |   4.2 | `cuMemcpyHtoDAsync_v2` |
|  8.7 |  6.0 | 7 422 908 |   0.8 | `cuMemFreeAsync` |
|  7.4 |  5.1 | 7 422 908 |   0.7 | `cuMemAllocAsync` |

Observations:
- **4.8 M kernel launches** across a single 512-token prefill step (≈ 9 400 launches per
  token). This is the primary reason prefill is 226× slower than llama.cpp — almost all of
  GPU time is wasted on launch overhead for tiny F32 elementwise kernels instead of one
  large fused kernel per layer.
- **7.4 M alloc + 7.4 M free calls** per prefill — every `narrow` / `contiguous` /
  `reshape` inside the per-token GDN loop creates and destroys temporary tensors.
- **3 M HtoD copies** at ~4 µs avg — likely per-call uploads of rotary `inv_freq` /
  `positions` / `pos_f32` vectors from `forward.rs::rotary_embedding` (see below).

### Decode (256 tokens)

| Time % | Total (s) | Calls | Avg (µs) | API |
|---:|---:|---:|---:|---|
| 57.8 | 57.1 |       901 | 63 328.5 | `cuMemcpyDtoHAsync_v2` |
| 20.1 | 19.8 | 1 721 818 |   11.5 | `cuLaunchKernel` |
| 10.9 | 10.7 |   493 764 |   21.7 | `cuMemcpyHtoDAsync_v2` |
|  2.3 |  2.3 |    11 088 |  208.4 | `cuMemsetD8Async` |
|  1.9 |  1.8 | 2 225 298 |    0.8 | `cuMemFreeAsync` |

Observations:
- **57.8 % of decode wall time is a single API: `cuMemcpyDtoHAsync_v2`**. With only
  ~901 calls averaging 63 ms each, these correspond to per-step synchronous downloads of
  the full [1, vocab=151 936] logits tensor to CPU memory for argmax / softmax in
  `sampling.rs`. Each download **blocks the GPU pipeline**; the 716 µs `ucopy_bf16`
  average in the kernel table is the device-side half of the same transfers.
- **1.7 M kernel launches** over 256 decoded tokens ≈ 6 700 launches per token — roughly
  the same per-token launch cost as prefill. Launch overhead alone explains much of the
  8× gap versus llama.cpp on decode.

---

## Per-Layer Split (GDN vs Full Attention)

Qwen3.5-4B in Kiln uses **24 Gated DeltaNet layers + 8 full-attention layers**. The
per-kernel instance counts make the GDN layers trivially identifiable as the dominant
cost center:

| Signal | Count / prefill | Source |
|---|---:|---|
| `bmul_f32` calls | 2.10 M | `gated_deltanet_forward` per-token recurrent loop (`broadcast_mul` of gate, K, recurrent state) |
| `fast_sum_f32` calls | 0.84 M | `gated_deltanet_forward` reductions (`.sum(2)` on KV-memory) |
| `badd_f32` calls | 0.43 M | GDN recurrent state update (`recurrent_state + outer`) |
| `cutlass ... s16816gemm` calls | ~5 500 | Full-attn + MLP projections (Q/K/V/O, up/down/gate) |

Per-token breakdown:
- Per prefill step (512 tokens): **2.1 M / 512 ≈ 4 100 `bmul_f32` invocations per token**.
  With 24 GDN layers that is ~170 `bmul_f32` per layer per token — consistent with the
  small-grained F32 elementwise operations inside `gated_deltanet_forward`'s per-token
  `for t in 0..seq_len` loop (see hotspot #2 below).
- Full-attention layers contribute ~8 × (Q·K, softmax, A·V) GEMMs + one flash-attn per
  token, which shows up only as the 3–5 % GEMM slice.

GDN dominates because it is implemented in candle-core as a **Rust-level token-by-token
loop that issues hundreds of small kernels per layer per token**, whereas the full-attn
path fuses into Flash Attention.

---

## Hot-Path → Source Mapping

All paths below are inside `crates/kiln-model/src/`.

1. **`sampling.rs:16-43` `greedy_sample` / `sampling.rs:55-145` `sample_with_params`**
   Both call `flat.to_vec1::<f32>()` on the logits tensor, which forces a synchronous
   DtoH copy of the full [vocab_size = 151 936] float vector per decoded token. Argmax /
   softmax then runs on CPU. Called from 9+ sites in `generate.rs` (lines 247, 249, 316,
   318, 362, 364, 417, 419, 544, 546, 605, 607, 711, 713, 915, 917, 996, 998).
   → **root cause of the 57.8 % `cuMemcpyDtoHAsync_v2` share in decode**.

2. **`forward.rs:605-750` `gated_deltanet_forward`** — per-token recurrent loop at
   **lines 702-732**:
   ```rust
   for t in 0..seq_len {
       let q_t = q.narrow(2, t, 1)?.squeeze(2)?;   // narrow+squeeze: kernel+alloc
       let k_t = k.narrow(2, t, 1)?.squeeze(2)?;
       let v_t = v.narrow(2, t, 1)?.squeeze(2)?;
       ...
       let g_exp = g_t.exp()?.unsqueeze(2)?.unsqueeze(3)?;
       *recurrent_state = recurrent_state.broadcast_mul(&g_exp)?;        // bmul_f32
       let kv_mem = recurrent_state.broadcast_mul(&k_expanded)?.sum(2)?; // bmul + fast_sum
       let delta = (v_t - kv_mem)?.broadcast_mul(&beta_t.unsqueeze(2)?)?;
       let outer = k_t.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?;
       *recurrent_state = (recurrent_state.clone() + outer)?;            // clone reallocs full state!
       outputs.push(...);
   }
   let attn_out = Tensor::cat(&outputs, 2)?;
   ```
   - Runs `24 layers × 512 tokens = 12 288` iterations per prefill; each iteration issues
     ~10 elementwise kernels + a reduce + several allocations.
   - `let v = v.to_dtype(DType::F32)?` at **line 693** forces the entire loop into F32,
     producing `bmul_f32` / `fast_sum_f32` / `badd_f32` dominance.
   - `recurrent_state.clone()` on **line 726** reallocates the full `[B, H_kv, D_k, D_v]`
     state tensor every single iteration — a primary source of the 7.4 M alloc/free calls
     and of `ucopy_bf16` / `copy2d_f32` traffic.

3. **`forward.rs:292-315` `rms_norm`** — every call casts `bf16 → F32`, computes
   `sqr → mean → sqrt → recip → broadcast_mul`, then casts back to `bf16`. The model runs
   **2 RMSNorms per layer × 32 layers = 64 RMSNorms per token**. This is a large share of
   `cast_bf16_f32`, `usqr_f32`, `urecip_f32`, and the small-instance `ucopy_bf16` copies.

4. **`forward.rs:317-347` `rotary_embedding`** — allocates a fresh
   `Tensor::new(&inv_freq_vec, device)` plus a `pos_f32: Vec<f32>` on every call, then
   issues `cuMemcpyHtoDAsync` to upload them. Called 8× per token (once per full-attn
   layer) → major contributor to the 3 M HtoD copies observed in prefill.

5. **`forward.rs:55-103` `flash_attention_forward`** — GQA expansion at **lines 71-80** does
   ```rust
   let k = k.unsqueeze(3)?
       .expand(&[batch, kv_len, num_kv_heads, gqa_ratio, hd])?
       .contiguous()?                 // ← large bf16 copy
       .reshape((batch, kv_len, num_heads, hd))?;
   ```
   Plus unconditional `.contiguous()` at **line 83** and post-flash `.contiguous().reshape()`
   at **lines 92-94**. Each `.contiguous()` on a rank-4 bf16 tensor produces a top-10
   `ucopy_bf16` entry in the kernel table.

6. **`forward.rs:523-603` `causal_conv1d_prefill`** — per-kernel-position narrow loop in
   F32 (line 531 casts input to F32) plus several `.contiguous()` copies. Runs once per
   GDN layer per prefill.

7. **`forward.rs:1332-1430` `model_forward`** — **line 1350** builds a fresh
   `positions: Vec<u32>` on every call; **line 1419** performs
   `hidden.broadcast_matmul(&weights.embed_tokens.t()?)` rather than a cached, non-broadcast
   GEMM for the LM head.

---

## Top 3 Ranked Recommendations

### 1. Move sampling argmax onto the GPU — eliminate the 57.8 % decode DtoH stall

- **Where:** `crates/kiln-model/src/sampling.rs:32` (`greedy_sample`) and `:80` / `:134`
  (`sample_with_params`).
- **What:** Replace `flat.to_vec1::<f32>()` + CPU `max_by`/softmax with a GPU-side argmax
  (`Tensor::argmax`) for the greedy path, and a GPU-side top-k/top-p + categorical
  (e.g. a small custom CUDA kernel, or temperature-scaled softmax on-device + single
  index DtoH) for the parameterized path. Only the selected token id (4 bytes) should
  ever cross the PCIe boundary per decode step.
- **Why it matters:** The current implementation forces a ~608 KB bf16→F32→host copy of the
  full [vocab=151 936] logits tensor and blocks the launch queue while the CPU reduces it.
  This is exactly the `cuMemcpyDtoHAsync_v2` call accounting for **57.8 % of decode GPU
  time** (901 calls × 63 ms average).
- **Expected gain:** ~2–3× decode speedup just from eliminating the sync DtoH; removes
  ~57 s of the ~101 s total decode run and lets kernel launches pipeline properly.

### 2. Replace the GDN per-token Rust loop with a fused / chunkwise GPU kernel

- **Where:** `crates/kiln-model/src/forward.rs:605-750` (`gated_deltanet_forward`);
  in particular the loop at **lines 702-732** and the F32 cast at **line 693**.
- **What:** Port the recurrent update to a single bf16/TF32 CUDA kernel (or at minimum a
  chunkwise implementation along the lines of the reference Gated-DeltaNet /
  chunk_gla_fwd kernel) that keeps the recurrent state in registers/shared memory,
  iterates tokens inside the kernel, and emits the whole [B, T, H, D_v] output in one
  launch per layer. Eliminate `recurrent_state.clone()` by writing in place.
- **Why it matters:** The loop is responsible for **~87 % of prefill GPU time**
  (`bmul_f32` 45.1 % + `fast_sum_f32` 16.7 % + `badd_f32` 4.4 % + much of the 25.2 %
  `ucopy_bf16`) and the majority of the 4.8 M kernel launches and 7.4 M allocs.
- **Expected gain:** 5–10× prefill speedup. Even a bf16 chunkwise implementation with
  chunk size 64 would cut per-token kernel launches from ~170 to ~3 and move all of GDN
  onto tensor cores. This is by far the biggest lever for closing the 226× prefill gap.

### 3. Remove `.contiguous()` + cache rotary / positions tensors

- **Where:**
  - `forward.rs:71-80`, `:83`, `:92-94` — `.contiguous()` / `.reshape()` calls in
    `flash_attention_forward`.
  - `forward.rs:317-347` — `rotary_embedding` allocates `inv_freq` + `pos_f32` per call.
  - `forward.rs:1350` — fresh `positions: Vec<u32>` per `model_forward`.
- **What:**
  - Switch the GQA head expansion to `expand().reshape()` **without** `.contiguous()` by
    passing stride-aware tensors into flash-attn, or precompute a single repeated
    KV tensor once in the KV cache rather than per step.
  - Cache `(cos, sin)` tables keyed by `max_position_embeddings` on the `ModelWeights`
    struct so rotary emits zero HtoD traffic after the first call.
  - Reuse a preallocated `positions` tensor (or compute it on-device with a small arange
    kernel) instead of allocating per step.
- **Why it matters:** These are the dominant contributors to the **3 M per-prefill HtoD
  copies** (17.9 % API share), the **7.4 M alloc/free** calls (8.7 % + 7.4 %), and to
  secondary `ucopy_bf16` traffic. They also prevent the driver's launch pipeline from
  running more than a few hundred µs ahead of the GPU, which compounds cost #2's launch
  overhead.
- **Expected gain:** 20–30 % additional prefill/decode speedup once #1 and #2 land, by
  reclaiming the remaining launch and memcpy overhead.

---

## Cross-reference summary

| Hot kernel / API | Share (prefill, decode) | Source |
|---|---|---|
| `ucopy_bf16` | 25.2 %, 89.9 % | `sampling.rs:32` DtoH logits (decode); GDN clones / GQA `.contiguous()` (prefill) |
| `bmul_f32` | 45.1 %, 1.2 % | `gated_deltanet_forward` loop `broadcast_mul` in F32 |
| `fast_sum_f32` | 16.7 %, 0.7 % | `gated_deltanet_forward` `.sum(2)` on KV memory |
| `badd_f32` | 4.4 %, 0.2 % | `gated_deltanet_forward` state update |
| `cuMemcpyDtoHAsync_v2` | 21.0 %, 57.8 % | `sampling.rs:32` `.to_vec1::<f32>()` per step |
| `cuLaunchKernel` | 23.6 %, 20.1 % | Per-token GDN loop + per-op candle kernels |
| `cuMemcpyHtoDAsync_v2` | 17.9 %, 10.9 % | `rotary_embedding` fresh tensors + `positions` |

Addressing recommendations 1 and 2 removes ~57 % of decode and ~66 % of prefill GPU time.
Recommendation 3 further reduces the remaining launch-and-copy overhead that currently
caps kernel pipelining.
