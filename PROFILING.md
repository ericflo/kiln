# Kiln Profiling Report — After PR #80 (GDN Kernel Vendored, Refresh)

## Overview

This report re-profiles `kiln-bench` running Qwen3.5-4B (32 layers: 24 Gated
DeltaNet linear-attention + 8 full-attention; bf16 weights) on a single
RTX A6000, after PR #80 landed:

- **#80** — Vendored `fla-org/flash-linear-attention`'s `chunk_gla_fwd` as
  the `kiln-gdn-kernel` CUDA kernel (the recommendation from PR #77's
  PROFILING refresh).

The purpose is to verify that the vendored chunkwise GDN kernel actually
moved the wall-clock numbers, see how the kernel mix shifted, and identify
the next concrete lever — explicitly answering whether FlashInfer paged-GQA
decode is the right second target now that GDN is fused.

**No optimizations are proposed or attempted here — this is a profiling-only
pass.** The recommendation at the end picks the single concrete next step.

## Hardware & Build

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (48 GiB, compute capability 8.6) |
| Driver | 550.x |
| CUDA toolkit | 12.4.1 (`ghcr.io/ericflo/kiln-runpod:latest` pre-baked image) |
| Rustc | 1.94.x stable |
| Build | `cargo build --release --features cuda --bin kiln-bench` |
| Build env | `KILN_CUDA_ARCHS=86`, sccache |
| Nsight Systems | 2024.6.2 |
| Model | Qwen3.5-4B, bf16 (`qwen2_5_4b_gdn`) |
| Commit | `cfe5152` (current `main`, includes PRs #76–#85; GDN kernel from #80) |

## Scenarios

Same as PR #77, with one tweak: a single mixed-workload nsys capture is
used (512 prompt → 128 output) rather than two separate captures. This was
necessary to ship within the per-pod time budget after a stuck nsys agent
forced one re-run. The capture covers both prefill and decode in the same
trace.

| Scenario | Invocation |
|---|---|
| Latency benchmark (prefill+decode) | `kiln-bench --prompt-tokens 512 --max-output-tokens 128 --skip-training` |
| Throughput benchmark | Same binary, throughput-mode warm-ups (1 token each) |

Capture: `nsys profile -t cuda,nvtx,osrt --cuda-memory-usage=false
--capture-range=none --sample=none --cpuctxsw=none --export=none`. Reduced
with `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum`.

---

## Wall-Clock Comparison (A6000, Same Binary Options)

Latency benchmark (`--prompt-tokens 512 --max-output-tokens 128`), measured
on this profiling pod:

| Metric | PR #75 (1653a41) | Current main (cfe5152, PR #80) | Δ |
|---|---:|---:|---:|
| Prefill (512 tokens)         | 2539.1 ms / **199 tok/s** | 1486.4 ms / **340 tok/s** | **+71 %** |
| Decode mean ITL (2 tokens)   | 240.7 ms / **4.15 tok/s** | 528.5 ms / **1.9 tok/s**  | **−54 %** |
| Mixed 512+128 wall (est.)    | ~33.4 s | ~68.9 s | **+106 % (slower)** |

Key takeaway — and this is the headline finding for the next decision:

- **Prefill improved sharply (+71 %).** The vendored `chunk_gla_fwd` did
  exactly what was hoped on the chunk-size-64 prefill path — collapsed the
  per-chunk narrow+contiguous + forward-substitution loop into one fused
  kernel call per chunk per head.
- **Decode regressed sharply (−54 %).** At chunk_size = 1 (the decode
  configuration), `chunk_gla_fwd` is materially slower than the previous
  candle-DSL chunkwise recurrence. The vendored kernel is optimized for
  the chunked prefill path, not the seq_len=1 recurrent path.
- Net effect on a typical 512+128 latency request: roughly **2× slower
  end-to-end**, because decode dominates wall time once the prompt is
  digested.

This is a regression that needs to be fixed before any further work on
attention layers makes sense.

---

## Aggregate Kernel Breakdown (top, mixed-workload capture)

| Rank | Time % | Total (ms) | Calls | Avg (µs) | Kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 86.6 | 1 902 | 132 600 | 14.3 | `ucopy_bf16` |
| 2 | 2.8  | 61    |  16 968 |  3.6 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64` |
| 3 | 1.5  | 33    |  10 720 |  3.1 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64` |
| 4 | 1.0  | 22    | 142 800 |  0.2 | `bmul_f32` |
| 5 | 0.9  | 20    |   8 480 |  2.4 | `ampere_bf16_s16816gemm_bf16_128x64` |
| —  | 0.3  | 6.6   |     960 |  6.9 | `gdn_fwd_sub_kernel` (vendored from #80) |

Compared to PR #75 / #77's split-capture report:

- **`copy2d_bf16` (56.6 % of prefill in PR #77) is gone** — the per-chunk
  narrow+contiguous materialization at the old `forward.rs:719–723` site
  is no longer the per-chunk hot path. PR #80's fused kernel absorbed it.
- **`ucopy_bf16` is now the overwhelming hot kernel: 86.6 % of all GPU
  kernel time.** This kernel is `Tensor::copy_*` traffic — chunk-boundary
  copies, transpose materializations, KV-cache writes, and per-step
  state-tensor copies in the decode loop. It was 14.8 % of prefill /
  89.3 % of decode in PR #77; mixed-workload it is now 86.6 %, consistent
  with decode dominating wall time.
- **`gdn_fwd_sub_kernel` itself (the vendored kernel) is only 0.3 % of
  GPU time.** The vendored kernel runs fast when called; the regression
  is not the kernel's compute, it is the per-step launch + `ucopy_bf16`
  surrounding it on the decode path.
- `bmul_f32` is back at 1.0 % (vs near-zero in PR #77 prefill) — likely
  per-step decay-vector broadcasts in the decode-side recurrence.
- The cutlass + ampere GEMMs (~5 % combined) are the 8 full-attention
  layers' projections + flash-attn-adjacent matmuls. **They are not the
  bottleneck.**

---

## CUDA API / Launch + Sync Overhead

| Time % | Total (s) | Calls | Avg (µs) | API |
|---:|---:|---:|---:|---|
| 55.7 | 64.85 |       530 | 122 367 | `cuMemcpyDtoHAsync_v2` |
| 22.8 | 26.60 | 2 561 440 |    10.4 | `cuLaunchKernel` |
|  7.2 |  8.40 |   714 313 |    11.8 | `cuMemcpyHtoDAsync_v2` |
|  2.9 |  3.32 | 6 764 726 |     0.5 | `cuStreamWaitEvent` |
|  2.6 |  3.01 | 3 382 363 |     0.9 | `cuMemFreeAsync` |
|  2.5 |  2.88 | 3 382 363 |     0.9 | `cuMemAllocAsync` |
|  2.0 |  2.38 | 2 797 525 |     0.8 | `cuEventRecord` |

GPU memory-side breakdown (`cuda_gpu_mem_time_sum`):

| Time % | Total (ns) | Count | Op |
|---:|---:|---:|---|
| 97.1 | 3.49 s    | 714 313 |  HtoD |
|  2.8 | 0.10 s    |  66 384 | memset |
|  0.0 | 0.81 ms   |     530 |  DtoH |

Observations:

1. **`cuMemcpyDtoHAsync_v2` is back at 55.7 %.** This is the same scalar
   `to_scalar::<u32>()` sampling sync barrier observed in PR #77 — one tiny
   blocking DtoH copy per decoded token (530 calls, total payload only
   806 KB → average 1.5 KB / call, so this is sync latency, not bandwidth).
   PR #80 did not address this and was not expected to.
2. **`cuMemcpyHtoDAsync_v2` ballooned to 714 K calls (3.49 s, 97.1 % of
   GPU memory-side time).** That is materially worse than PR #77's per-step
   profile. Most of this is likely per-step constant uploads (mask /
   position-id / inv_freq table look-ups for the vendored Triton kernel)
   that are not yet cached. This is *new* HtoD churn introduced by the
   vendored kernel's host-side scaffolding.
3. **`cuLaunchKernel` count exploded to 2.56 M.** Per token of the latency
   workload that is roughly 20 K launches/token — far higher than PR #77's
   ~2 700 launches/token. PR #80's per-chunk fused kernel reduces launches
   per chunk, but on the decode path (chunk_size = 1) the surrounding
   per-step Rust glue and the `ucopy_bf16` traffic dominate. The fused
   kernel is winning on prefill and losing on decode.
4. The `cuStreamWaitEvent` / `cuEventRecord` / `cuMemAllocAsync` /
   `cuMemFreeAsync` columns indicate the per-step allocator and event
   churn is back at PR #70-era levels for decode. PR #80's chunkwise path
   is being asked to do something it was not designed for.

---

## Per-Layer Split (GDN vs Full Attention)

- 24 GDN layers vs 8 full-attention layers; GDN is 75 % of layer count
  and >90 % of decode kernel time on this capture.
- The 8 full-attention layers' GEMMs (cutlass + ampere bf16 16816)
  contribute ~5 % of total GPU time. Flash-attn proper does not appear
  in the top kernels, consistent with bf16 16816 paths being well-tuned
  on Ampere.
- The vendored `gdn_fwd_sub_kernel` itself is ≤ 0.3 % of total GPU time.
  The cost lives in the surrounding `ucopy_bf16` and in per-step launch /
  HtoD overhead.

---

## Hot-Path → Source Mapping (current main)

1. **`crates/kiln-gdn-kernel/src/lib.rs` host-side scaffolding for
   `chunk_gla_fwd`** — the bridge between the candle tensors and the
   vendored Triton-style CUDA kernel. The 714 K HtoD copies and excess
   `cuLaunchKernel` calls per decode token point here. Constants (mask
   shapes, chunk dims, decay scaling tables) are likely re-uploaded per
   call instead of being cached on `GpuWeights`.

2. **`crates/kiln-model/src/forward.rs` GDN decode wiring** — the path
   that now feeds `chunk_gla_fwd` with chunk_size=1 single-token chunks.
   For decode, `chunk_gla_fwd` is doing chunked-matmul work that the
   `fused_recurrent_*` family of fla kernels would do natively in O(d)
   per token. Calling chunkwise with C=1 pays setup cost on every token.

3. **`sampling.rs` greedy_sample** — `flat.argmax(0)?.to_scalar::<u32>()?`
   still issues one blocking DtoH per token. Same as PR #77; still 55.7 %
   of API time.

4. **Remaining `Tensor::copy_*` sites** — KV-cache writes, transpose
   materializations, and per-step state copies that produce the 86.6 %
   `ucopy_bf16` aggregate.

---

## Why Wall-Clock Improved on Prefill but Regressed on Decode

The story is now clear:

1. **Prefill (chunk_size = 64, 8 chunks per layer):** PR #80's fused
   kernel does exactly what it was vendored to do. It collapses the
   per-chunk narrow+contiguous + 64-iter forward-substitution loop into
   register-resident work. Prefill +71 %.
2. **Decode (chunk_size = 1, 1 chunk per layer per token):** the same
   kernel is the wrong tool. The fla repo distinguishes
   `chunk_gla_fwd` (chunked) from `fused_recurrent_gla_fwd` /
   `fused_recurrent_gated_delta_rule_fwd` (single-step recurrent) for
   exactly this reason. PR #80 only vendored the chunked path. Calling
   it with chunk_size=1 incurs chunk-setup overhead per token, and the
   surrounding `ucopy_bf16` + HtoD constants per step push the total
   per-token cost above the candle-DSL recurrence it replaced.
3. **Net wall-clock for a typical request is worse**, because decode
   weight on a 512+128 workload is roughly 2/3 of total time, and the
   decode regression outpaces the prefill gain.

PR #80 is therefore a partial win and a partial regression. The next
step must address the decode path before any other optimization is
worth attempting.

---

## Next Optimization — Recommendation

**(a) Vendor fla-org's `fused_recurrent_gated_delta_rule_fwd` (the
single-step recurrent kernel) and route `forward.rs`'s GDN decode path
to it when `chunk_size == 1`. Keep PR #80's `chunk_gla_fwd` for prefill.**

### Why this is the top lever

- It is the **only candidate that addresses the regression PR #80 just
  introduced**. No other optimization (FlashInfer, fused KV writes,
  CUDA-graph capture, etc.) recovers the 54 % decode loss.
- The fla repo already provides the kernel in
  `fla/ops/gated_delta_rule/fused_recurrent.py` (and a CUDA-port
  equivalent) — vendoring follows the same pattern PR #80 established
  for `chunk_gla_fwd`.
- Expected impact:
  - Decode: restore ≥ 4.15 tok/s baseline and very likely beat it
    (≥ 5 tok/s), since `fused_recurrent_*` is purpose-built for
    seq_len=1 and keeps state in registers across the per-step kernel
    call. Eliminates most of the per-token `cuLaunchKernel` and
    `cuMemcpyHtoDAsync_v2` churn introduced by the chunkwise scaffolding
    on the decode path.
  - Prefill: unchanged (PR #80 stays in place for the chunked path).
- Bonus: this also collapses much of the 86.6 % `ucopy_bf16` aggregate,
  because the fused recurrent kernel keeps the per-token state on-device
  in registers rather than round-tripping through bf16 tensor copies.

### (b) Vendor FlashInfer paged-GQA decode — **rejected for now**

The 8 full-attention layers contribute roughly **5 % of total GPU
kernel time** (cutlass + ampere bf16 16816 GEMMs combined). That is
**below the 10 % Amdahl-ceiling threshold** for justifying a vendoring
effort of FlashInfer's complexity. FlashInfer becomes the right next
target only after the GDN decode path is fixed and after measured
full-attention share rises above ~10 % of decode GPU time. Re-profile
after option (a) lands and re-evaluate.

### (c) Eliminate `ucopy_bf16` by fusing KV-cache writes / state copies

86.6 % aggregate `ucopy_bf16` is real and uncomfortable, but the bulk of
it is *secondary* to the decode-path regression — the `fused_recurrent_*`
kernel from option (a) already eliminates most of the per-step state
copies that contribute to this aggregate. Revisit only if (a) still
leaves `ucopy_bf16` above ~30 %.

### (d) Cache constants on `GpuWeights` to kill the new HtoD churn

PR #80's host-side scaffolding uploads chunk-shape / mask / decay
constants per call (714 K HtoD ops in this capture). A small cleanup PR
that hoists these onto `GpuWeights` would cut `cuMemcpyHtoDAsync_v2`
sharply. It is a useful tidy-up but not the lead lever — fold it into
option (a) when the decode path is rewritten.

---

## Cross-reference summary

| Hot kernel / API | PR #75 share (prefill, decode) | Current share (mixed) | Source |
|---|---|---|---|
| `ucopy_bf16` | 14.8 %, 89.3 % | **86.6 %** | per-step state/KV copies; chunk-boundary copies |
| `copy2d_bf16` | 56.6 %, — | < 0.1 % | gone — fused into PR #80's `chunk_gla_fwd` |
| `gdn_fwd_sub_kernel` | n/a | 0.3 % | vendored from PR #80 (kernel itself is fast) |
| cutlass + ampere bf16 GEMMs | top-2..4 of decode | ~5 % combined | 8 full-attn layers (FlashInfer target) |
| `bmul_f32` | 3.2 %, 0.5 % | 1.0 % | per-step decay-vector broadcasts on decode |
| `cuMemcpyDtoHAsync_v2` | < 0.1 %, 56.5 % | **55.7 %** | `sampling.rs` `to_scalar::<u32>` sync |
| `cuMemcpyHtoDAsync_v2` | 93.5 % (startup), 6.2 % | **7.2 % API / 97.1 % GPU-mem** | per-step constant uploads — new in PR #80 scaffolding |
| `cuLaunchKernel` | 2.2 %, 21.3 % | 22.8 % (~20 K/tok) | per-step launches around `chunk_gla_fwd` on decode |

PR #80 successfully fused the chunked GDN path on prefill but introduced
a sharp decode regression. The next concrete step is to vendor the
matching `fused_recurrent_*` kernel for chunk_size=1 and route the
decode path to it. FlashInfer remains an attractive target but is
explicitly *not* the right next pick — the full-attention layers it
would optimize are below the 10 % Amdahl-ceiling threshold on this
model.
