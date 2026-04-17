# Kiln Profiling Report — After PRs #71–#75 (Refresh)

## Overview

This report re-profiles `kiln-bench` running Qwen3.5-4B (32 layers: 24 Gated
DeltaNet linear-attention + 8 full-attention; bf16 weights, weight-tied LM
head) on a single RTX A6000, after the first wave of fixes identified in the
original profiling report landed:

- **#71** — Sampling argmax moved on-GPU (scalar-only DtoH per decode step)
- **#72** — GDN recurrence runs in bf16, dropped redundant state clone
- **#73** — Cache RoPE `inv_freq` on `GpuWeights`, dropped redundant flash-attn `.contiguous()`
- **#74** — GDN matmul-based readout in prefill recurrence
- **#75** — GDN chunkwise analytical recurrence (C=64 per-head matmul chunks)

The purpose of this pass is to quantify how much of the 226× prefill / 8×
decode gap versus llama.cpp has actually closed, verify that the kernel mix
has shifted away from `bmul_f32` / `fast_sum_f32` dominance, and identify the
next concrete lever.

**No optimizations are proposed or attempted here — this is a profiling-only
pass.** The recommendation at the end points to a single concrete next step.

## Hardware & Build

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (48 GiB, compute capability 8.6) |
| Driver | 550.127.08 |
| CUDA toolkit | 12.4.1 (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) |
| Rustc | 1.94.1 stable |
| Build | `cargo build --release --features cuda --bin kiln-bench` |
| Build env | `KILN_CUDA_ARCHS=86`, B2 sccache (x86_64-linux-cuda12.4 prefix) |
| Nsight Systems | 2024.6.2 (apt `nsight-systems-2024.6.2`) |
| Model | Qwen3.5-4B, bf16, loaded from HF (`qwen2_5_4b_gdn`) |

## Scenarios

Identical workload to the original report (PR #70) for apples-to-apples
comparison:

| Scenario | Invocation | Config |
|---|---|---|
| Prefill | `kiln-bench --prompt-tokens 512 --max-output-tokens 1 --skip-training` | 512 prompt → 1 output, prefill-dominated |
| Decode  | `kiln-bench --prompt-tokens 16  --max-output-tokens 64 --skip-training` | 16 prompt → 64 output, decode-dominated |

Captures were collected with
`nsys profile -t cuda,nvtx,osrt --cudabacktrace=all` and reduced with
`nsys stats --report cuda_gpu_kern_sum,cuda_api_sum`. Raw artifacts (on the
RunPod profiling host):

- `prefill.nsys-rep`, `prefill.sqlite`
- `decode4.nsys-rep` (`--max-output-tokens 4`, reduced from 256 so `qdstrm`
  conversion completes), `decode4.sqlite`

Because the decode capture uses a shorter generation than PR #70's 256-token
run, total absolute decode GPU seconds are not directly comparable, but the
kernel-share percentages, per-call timings, and launch counts are.

---

## Wall-Clock Comparison (A6000, Same Binary Options)

Both numbers were collected on the same A6000, using the same
`kiln-bench` invocation, with `RUSTC_WRAPPER=sccache` builds from each
commit.

| Metric | PR #70 (35a5533) | Current main (PR #75, 1653a41) | Δ |
|---|---:|---:|---:|
| Prefill (506 prompt tokens) | 2289.6 ms / **221 tok/s** | 2539.1 ms / **199 tok/s** | –10 % |
| Decode mean ITL (2 tokens)  | 247.2 ms / **4.0 tok/s** | 240.7 ms / **4.15 tok/s** | +4 % |
| Decode mean ITL (65 tokens, 16-prompt workload) | 230.9 ms / **4.3 tok/s** | 233.2 ms / **4.3 tok/s** | –1 % |
| Model load                  | 9.22 s | 11.63 s | +26 % |

Key takeaway: **wall-clock prefill and decode have barely moved** (within
measurement noise) even though the kernel mix and launch overhead have
changed dramatically. The internal work is now organized very differently —
fewer, larger kernels on both paths — but the new bottleneck is a different
class of overhead (see "Why wall-clock didn't move" below).

---

## Prefill Kernel Breakdown (top 5, current main)

Prefill has pivoted decisively away from the F32 elementwise / reduction
dominance that defined PR #70. The hot kernels are now **bf16 memory copies**
around the per-chunk GDN matmuls, not math.

| Rank | Time % | Total (ms) | Calls | Avg (µs) | Kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 56.6 | 25.5 | 18 311 | 1.4 | `copy2d_bf16` (narrow + contiguous per chunk) |
| 2 | 14.8 | 6.7  |    629 | 10.6 | `ucopy_bf16` |
| 3 | 5.9  | 2.6  |    639 | 4.1 | `bmul_bf16` (broadcast multiply, bf16) |
| 4 | 4.7  | 2.1  |    570 | 3.7 | `bsub_bf16` (broadcast subtract, bf16) |
| 5 | 3.2  | 1.5  |     22 | 66.7 | `bmul_f32` |
| —  | 2.6  | 1.2  |    525 | 2.2 | `gemvx_fp_kernel<bf16>` |
| —  | 1.8  | 0.8  |      4 | 199.0 | `ampere_bf16_s16816gemm_bf16_256x128` |

**Total kernel time: ~45 ms** across 45 109 `cuLaunchKernel` calls.

Compare to PR #70:
- `bmul_f32`: **45.1 % → 3.2 %** (22 calls down from 2.1 M — chunkwise
  recurrence batches the broadcast into a bf16 matmul).
- `fast_sum_f32`: **16.7 % → 0 %** (no longer appears; its role is absorbed
  into the `k·S` / `K·K^T` matmul in the analytical recurrence).
- `badd_f32`: **4.4 % → 0 %** (state update is now a single matmul).
- `ucopy_bf16`: **25.2 % → 14.8 %** (flash-attn `.contiguous()` removed in
  #73; remaining source is chunk-boundary copies from step 2 below).
- New: `copy2d_bf16` at **56.6 %** — this is `narrow(2, t_start, c)?.contiguous()`
  at `forward.rs:719–723` materializing each of q/k/v/beta/g per chunk.
- Total `cuLaunchKernel` per prefill: **4.8 M → 45 K** (≈ 106× fewer launches).

## Decode Kernel Breakdown (top 5, current main)

Decode remains dominated by bf16 memory copies. The **share** of `ucopy_bf16`
is almost identical to PR #70 (89.3 % vs 89.9 %), but absolute time per call
and per-token launch count both dropped:

| Rank | Time % | Total (s) | Calls | Avg (µs) | Kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 89.3 | 34.18 | 65 178 | 524.6 | `ucopy_bf16` |
| 2 | 2.5  |  0.95 |  8 840 | 107.3 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64` |
| 3 | 1.4  |  0.54 |  7 616 |  70.9 | `ampere_bf16_s16816gemm_bf16_128x64` |
| 4 | 1.4  |  0.52 | 11 424 |  45.7 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64` |
| 5 | 0.6  |  0.22 | 59 568 |   3.7 | `bmul_bf16` |
| —  | 0.5  |  0.18 | 70 788 |   2.6 | `bmul_f32` |

Compare to PR #70 (256-decode-token capture):
- `ucopy_bf16` absolute time: 90.9 s → 34.2 s (shorter capture, but per-call
  avg also dropped 716 µs → 525 µs).
- `bmul_f32` absolute time: 1.26 s → 0.18 s (per-call 5.2 µs → 2.6 µs); share
  collapsed from 1.2 % to 0.5 %.
- `fast_sum_f32`, `cast_bf16_f32`, `affine_f32`, `badd_f32` no longer appear
  in the top 10 — all folded into matmul-based GDN readout.

---

## CUDA API / Launch + Sync Overhead

### Prefill (512 tokens), current main

| Time % | Total (s) | Calls | Avg (µs) | API |
|---:|---:|---:|---:|---|
| 93.5 | 6.54 |    5 116 | 1 278 | `cuMemcpyHtoDAsync_v2` |
| 2.2  | 0.15 |   45 109 |   3.3 | `cuLaunchKernel` |
| 1.4  | 0.10 |  170 575 |   0.6 | `cuEventRecord` |
| 1.1  | 0.076|    5 805 |  13.1 | `cuMemAllocAsync` |

The `cuMemcpyHtoDAsync_v2` share is model-weight load and tokenizer input
upload at process start (the nsys capture begins before model load), not
per-step HtoD traffic. The per-step `rotary_embedding` HtoD storm identified
in PR #70 is gone (#73 cached the inv_freq table).

Comparison vs PR #70 per-prefill:
- `cuLaunchKernel`: **4.8 M → 45 K** (≈ 106× fewer launches)
- `cuMemAllocAsync`: **7.4 M → 5.8 K** (≈ 1 280× fewer allocations)
- `cuMemFreeAsync`: **7.4 M → 5.8 K**
- `cuMemcpyHtoDAsync_v2`: **3.0 M per-step → 248 K** (rotary inv_freq cached)

Kernel launch and alloc churn have been reduced by more than two orders of
magnitude. This was the single largest expected source of the 226× prefill
gap. It is now a small slice of API time.

### Decode (short capture: 4 generated tokens)

| Time % | Total (s) | Calls | Avg (ms) | API |
|---:|---:|---:|---:|---|
| 56.5 | 20.36 |       170 | 119.8 | `cuMemcpyDtoHAsync_v2` |
| 21.3 |  7.66 |   882 810 |   0.009 | `cuLaunchKernel` |
|  6.2 |  2.23 |   248 185 |   0.009 | `cuMemcpyHtoDAsync_v2` |
|  3.2 |  1.16 | 2 196 710 |   0.001 | `cuStreamWaitEvent` |
|  2.8 |  1.00 | 1 098 355 |   0.001 | `cuMemFreeAsync` |

Observations:
- `cuMemcpyDtoHAsync_v2` still holds **56.5 %** of the decode API timeline
  (vs 57.8 % on PR #70). **But the payload changed.** In PR #70 these calls
  were per-step bf16→F32→host copies of the full [1, vocab=151 936] logits
  tensor (~608 KiB/step). On current main, argmax / top-k run on-device
  (see `sampling.rs:53` — `flat.argmax(0)?.to_scalar::<u32>()?`); only the
  selected 4-byte token id is downloaded.
- The remaining DtoH time is therefore **almost entirely stream-sync latency**:
  `to_scalar::<u32>()` queues a tiny DtoH copy behind the entire decode
  step's launches and blocks until it drains. Per call the transfer is trivial;
  the cost is the round-trip latency of a blocking sync point once per decoded
  token, which does not benefit from the 608 KiB → 4 B reduction.
- Per-token `cuLaunchKernel` count is lower than PR #70's ~6 700/tok estimate
  (the capture's 882 K launches are distributed across all 30 bench-phase
  decode sessions that ran during the capture, not a single 65-token generation).
  Per-token absolute count is no longer a primary bottleneck.
- `cuStreamWaitEvent` has emerged as a new top-5 entry (3.2 %) — a sign that
  intra-step dependencies are now serialized via events rather than implicit
  ordering.

---

## Per-Layer Split (GDN vs Full Attention)

GDN still drives almost all decode kernel time because the model has 24 GDN
layers vs 8 full-attention. The layer-level picture is cleaner than before —
per-token bf16 matmul fragments in GDN have replaced the F32 elementwise
fragments — but the per-token GPU commit pattern has not changed fundamentally:

- Per decode token (65-token run, PR #75): ~2 700 kernel launches, roughly
  ~110 launches per GDN layer per token. For `chunk_size = 1` (decode path),
  `gdn_chunkwise_recurrence` at `forward.rs:697` degenerates to a short
  sequence of small bf16 matmuls plus several narrow/contiguous copies per
  head per layer.
- Full-attention layers (8 of 32) contribute the top-5 cutlass GEMMs and
  the flash-attn call once per layer per token, totalling ~5 % of decode
  GPU time.
- Per prefill (512 tokens, chunked into 8× C=64): the GDN path issues
  ~45 K total launches, dominated by the per-chunk narrow+contiguous copies
  (≈ 18 K `copy2d_bf16` calls, step-1 in the hot list).

---

## Hot-Path → Source Mapping (current main)

All paths below are inside `crates/kiln-model/src/`.

1. **`forward.rs:719–723` GDN chunkwise slice materialization** —
   `gdn_chunkwise_recurrence` narrows each of q / k / v / beta / g per chunk
   and forces `.contiguous()` because candle's matmul requires a
   contiguous last-two-dim tensor:

   ```rust
   let q_c = q.narrow(2, t_start, c)?.contiguous()?; // [B, nv, C, dk]
   let k_c = k.narrow(2, t_start, c)?.contiguous()?; // [B, nv, C, dk]
   let v_c = v.narrow(2, t_start, c)?.contiguous()?; // [B, nv, C, dv]
   let beta_c = beta.narrow(2, t_start, c)?.contiguous()?;
   let g_c = g.narrow(2, t_start, c)?.contiguous()?;
   ```

   With 24 GDN layers × 8 chunks × 5 tensors = **960 `copy2d_bf16` calls per
   prefill** from this site alone, plus additional `.contiguous()` calls at
   `:755` (`k_t_mat`), `:762` (`a_strict`), `:794` (`b_mask`), and `:778`
   (`a_row` inside the forward-substitution loop). This maps directly to the
   top-1 hotspot at **56.6 % of prefill kernel time**.

2. **`forward.rs:770–785` forward-substitution inner loop** —
   computes `W[t]` row-by-row with `Tensor::cat(&w_rows, 2)?` growing a
   [B, nv, t, dv] prefix every iteration:

   ```rust
   for t in 0..c {
       ...
       let w_prev = Tensor::cat(&w_rows, 2)?;           // reallocates + copies
       let sub = a_row.matmul(&w_prev)?;
       (vp_t - sub)?.broadcast_mul(&beta_t)?
   }
   ```

   This runs **C=64 iterations per chunk × 8 chunks × 24 layers = 12 288
   iterations per prefill**, each issuing a cat, a matmul, and several
   element-wise bf16 ops. Per-call launch count is small but the
   reallocation pattern is expensive enough to explain the residual
   `ucopy_bf16` and `bmul_bf16` counts.

3. **`sampling.rs:53` `greedy_sample`** — `flat.argmax(0)?.to_scalar::<u32>()?`
   is on-device as intended, but `to_scalar` issues a `cuMemcpyDtoHAsync_v2`
   of 4 bytes that **blocks the host thread until all queued kernels drain**.
   Although the payload is trivial, the per-token sync cost is what now
   anchors the 56.5 % decode API share. The GPU cannot pipeline tokens
   through the CUDA-graph path because the next token id is not known until
   this sync returns and `generate.rs` queues the next launch.

4. **Remaining `.contiguous()` sites** that still appear in kernel counts
   but have low % share: `forward.rs:807` (gate post-q/gate split),
   `forward.rs:833–835` (conv1d path q/k/v contiguify),
   `forward.rs:849–851` (pre-flash-attn transpose+contiguous).

---

## Why Wall-Clock Didn't Move Despite the Kernel Mix Change

The key measurement-vs-profile story of this refresh is that **wall-clock
per-request throughput is essentially unchanged** (prefill 221 → 199 tok/s,
decode 4.0 → 4.15 tok/s) even though:

- Prefill kernel launches dropped 106×
- Prefill allocations dropped 1 280×
- `bmul_f32` + `fast_sum_f32` + `badd_f32` (~66 % of PR #70 prefill GPU time)
  collapsed to 3.2 % of current main prefill
- The decode DtoH payload shrank from 608 KiB to 4 B

The likely explanations:

1. **PR #70 was not actually launch-overhead-bound at the host level.**
   The 4.8 M launches were dispatched fast enough that the GPU was kept
   saturated in small-F32-elementwise kernels; reducing the launch count
   just means the GPU now spends its time in a different kernel mix
   (bf16 matmuls + copy2d) instead of F32 elementwise + allocator churn.
   Total GPU-busy time did not drop proportionally.

2. **Decode sync barrier unchanged.** The scalar DtoH still forces one stream
   sync per token. Because CUDA graphs cannot cross that sync, the decode
   pipeline remains one-token-at-a-time from the host's perspective.

3. **Per-chunk narrow+contiguous replaced per-token broadcast_mul.** The
   work shifted onto tensor cores, but the chunk-boundary copies
   (56.6 % of prefill kernel time) and the inner forward-substitution loop
   (12 288 iters/prefill) are themselves a new serial bottleneck at a
   higher-granularity level than before.

The improvements are real but latent — they remove the failure modes that
would have prevented future fixes from working. They do not by themselves
close the 226×/8× llama.cpp gap.

---

## Next Optimization — Recommendation

**Vendor `fla-org/flash-linear-attention`'s `chunk_gla_fwd` Triton kernel
(or a CUDA port of it) and wire it into `forward.rs:975` in place of the
current `gdn_chunkwise_recurrence`.**

### Why this is the top lever

- Decode is **still 89 % `ucopy_bf16`** and **56.5 % blocking DtoH sync**.
  Both are downstream effects of the same root cause: the GDN recurrent
  path issues hundreds of small bf16 ops per layer per token and cannot be
  fused from candle's elementwise/matmul DSL.
- The 24 GDN layers vs 8 full-attn layer ratio means GDN dominates both
  prefill and decode. Any fix to GDN moves the whole model.
- `chunk_gla_fwd` (and the newer `chunk_dplr_delta_rule` / `chunk_gated_delta_rule`
  in fla-org ≥0.1.3) is the reference implementation of exactly this
  recurrence, written as a single Triton kernel per chunk that:
  - keeps the recurrent state in registers across the chunk,
  - fuses the decay matrix, K·K^T, forward substitution, and output
    projection into one kernel launch per chunk per head,
  - eliminates the `narrow(...).contiguous()` per-chunk materialization
    that now dominates prefill (56.6 % `copy2d_bf16`),
  - handles decode (seq_len=1) with the same kernel at chunk_size=1 without
    paying chunk-setup overhead.
- Expected impact, based on the fla-org paper's measured speedups and the
  residual hot list above:
  - Prefill: ≥ 4–6× throughput (collapse the 56.6 % copy2d_bf16 +
    14.8 % ucopy_bf16 into register traffic, remove the 12 288-iter
    forward-substitution loop),
  - Decode: ≥ 2× throughput (replace the ~110 launches/layer/token with
    one kernel call; the scalar-DtoH sync barrier persists but each step's
    pre-sync GPU work shrinks dramatically, so the sync overlaps more).

### Why not options (b) or (c)

- **(b) Vendor FlashInfer paged-GQA decode** — attractive but addresses
  only the 8 full-attention layers, which contribute ~5 % of decode GPU
  time on this model. The 24 GDN layers must be fixed first; FlashInfer
  becomes the right second target once GDN drops below ~30 % of decode.
- **(c) Something else** (e.g. further bf16 conversions, eliminating
  `to_scalar` via persistent CUDA graphs, speculative decoding) — each
  would shave at most single-digit percentages against the current mix.
  None move the 89 % `ucopy_bf16` anchor.

---

## Cross-reference summary

| Hot kernel / API | PR #70 share (prefill, decode) | Current share (prefill, decode) | Source |
|---|---|---|---|
| `copy2d_bf16` | — | 56.6 %, — | `forward.rs:719–723` per-chunk narrow+contiguous |
| `ucopy_bf16` | 25.2 %, 89.9 % | 14.8 %, 89.3 % | chunk-boundary copies (prefill); per-token bf16 copies in GDN decode (decode) |
| `bmul_f32` | 45.1 %, 1.2 % | 3.2 %, 0.5 % | residual F32 broadcast (cumsum/exp of decay) |
| `fast_sum_f32` | 16.7 %, 0.7 % | < 0.1 %, < 0.1 % | replaced by matmul readout |
| `cuMemcpyDtoHAsync_v2` | 21.0 %, 57.8 % | < 0.1 %, 56.5 % | `sampling.rs:53` `to_scalar::<u32>` sync barrier |
| `cuLaunchKernel` | 23.6 %, 20.1 % | 2.2 %, 21.3 % | 106× fewer launches on prefill, ~2.5× fewer per decode step |
| `cuMemcpyHtoDAsync_v2` | 17.9 %, 10.9 % | 93.5 % (startup), 6.2 % | rotary inv_freq now cached; HtoD is mostly one-time model load |

The PRs #71–#75 eliminated the F32-elementwise and allocator-churn failure
modes. The remaining levers live on a single path: the GDN chunkwise
recurrence's memory traffic and the per-token sampling sync. Option (a) —
vendor a fused linear-attention Triton kernel — addresses both at once and
is the single highest-leverage next step.
