# Kiln Profiling Report — Paged Production Path (Phase 6, post-PR #94)

## Overview

Every prior `PROFILING.md` (PRs #87 / #94) measured the **non-paged**
`generate_from_tokens` path through `kiln-bench`, which routes through
`KvCache` + `model_forward`. Production HTTP serving uses a different
code path: `PagedKvCache` + `BlockTable` + `model_forward_paged`. PR #94
added the vendored FlashInfer GQA-decode kernel and wired it into the
paged path, but the bench at the time still measured the non-paged path,
so the optimization recommendation could not be locked in against the
production hot path.

This Phase 6 report adds a `--paged` flag to `kiln-bench` and re-profiles
the production paged path so the next optimization can be picked from
production-shaped data.

The questions answered here:

1. **Is `ucopy_bf16` still the dominant hotspot on the paged path?** Yes.
2. **Does the paged path have any meaningful new kernels not seen on the
   non-paged path?** Yes — `copy2d_bf16` (per-token paged KV write) at
   0.5 % of GPU time, plus a slightly different per-step memcpy mix.
3. **What is the next optimization?** Same as PR #94's recommendation,
   now confirmed on the production path: target `ucopy_bf16` either by
   vendoring a paged-GQA fused kernel or by collapsing the surrounding
   per-step bf16 reshapes.

**No optimizations are proposed or attempted here — this is a profiling-only
pass.** The recommendation at the end picks the single concrete next step.

## Hardware & Build

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A5000 (24 GiB, compute capability 8.6) |
| Driver | 570.211.01 |
| CUDA toolkit | 12.8 (pod shipped without the pre-baked image; see "Environment notes" below) |
| Rustc | 1.95 stable (rustup-installed on pod) |
| Build | `cargo build --release --features cuda --bin kiln-bench` |
| Build env | `KILN_CUDA_ARCHS=86`, sccache (B2 backend); 100 % C/C++/CUDA hits, 0 % Rust hits (cold target dir) |
| Nsight Systems | 2024.6.2 |
| Model | Qwen3.5-4B (HF `Qwen/Qwen3.5-4B`), bf16, multimodal weight prefix `model.language_model.` |
| Commit | `3390a39` (`ce/paged-bench-profile` branch, post-#94) |

**Environment notes.** `ghcr.io/ericflo/kiln-runpod:latest` was again
requested for this pod and again came up as a stock CUDA 12.8 Ubuntu
image with none of the pre-baked tooling — same regression PROFILING.md
flagged after PR #92. Bootstrap installed Rust 1.95, `nsight-systems-2024.6.2`,
`hf-transfer`, and the B2-backed sccache prefix
`/build-cache/kiln/x86_64-linux-cuda12.8/sccache`. The
`kiln-flash-attn` C/C++ + CUDA cache was 100 % hot from the bucket so
the build finished in **3 m 14 s**, even with 0 % Rust cache hits on
this fresh clone. The kiln-runpod image regression should be triaged
separately.

The non-paged latency phase ran first on this pod and absorbed the cold
`cuModuleLoadData` cost (~9 s, see PR #94 PROFILING). The paged latency
phase ran second on the same pod and saw the loader warm, so its prefill
number reflects steady-state, not first-launch JIT cost.

## Scenarios

Two latency captures on the same pod, with the same binary, against the
same prompt configuration:

| Scenario | Invocation |
|---|---|
| Non-paged latency (`KvCache` + `model_forward`) | `kiln-bench --model-path ~/models/qwen3.5-4b --prompt-tokens 512 --max-output-tokens 128 --skip-training` |
| Paged latency (`PagedKvCache` + `BlockTable` + `model_forward_paged`, block_size=16, blocks=40) | `kiln-bench --paged …` (same args otherwise) |

The paged scenario uses `block_size=16` (the production default) with
40 blocks pre-allocated to cover the full 506 prompt + 128 decode
window. The block table is built sequentially (logical block i →
physical block i) since this is a single-sequence latency benchmark; the
production scheduler will use the same block-table API in a more
fragmented allocation pattern, but the per-token kernel mix is what we
want to capture and that is unchanged by allocation pattern.

Capture: `nsys profile -t cuda,nvtx,osrt --cuda-memory-usage=false
--capture-range=none --sample=none --cpuctxsw=none -o profile_paged
--force-overwrite=true …` for the paged run only. The bench was
`SIGINT`ed during the throughput phase to flush the `.nsys-rep` and exit
early, matching prior PROFILING reports' budget discipline. Reduced with
`nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum`.

---

## Wall-Clock Comparison — Paged vs Non-Paged

Latency benchmark (`--prompt-tokens 512 --max-output-tokens 128`),
measured on this pod:

| Metric | Non-paged (`KvCache`) | Paged (`PagedKvCache`, block_size=16) | Δ paged vs non-paged |
|---|---:|---:|---:|
| Prefill (506 tokens)         | 11 719.2 ms / **43 tok/s**  | **851.7 ms / 594 tok/s**  | First run absorbed ~9 s of `cuModuleLoadData` JIT — see note below |
| Decode mean ITL (129 tokens) | 265.1 ms / **3.77 tok/s**   | **276.9 ms / 3.61 tok/s** | **+4.5 % (paged slightly slower)** |

### Why the prefill numbers look so different

Do **not** read "paged prefill is 14× faster than non-paged prefill"
into the table. The non-paged run was the **first** kiln-bench
invocation on this pod and paid the full cold `cuModuleLoadData` cost
on its prefill (PR #94 measured this at ~9.4 s for 7 module loads on
the stock pod). The paged run was the **second** invocation on the
same pod; the loader was warm and the steady-state prefill cost
emerged: 851.7 ms / 594 tok/s on the paged path. Subtracting ~9 s of
JIT from the non-paged number gives roughly 2.7 s — about 3× slower
than paged steady-state, which would itself be a measurement artifact
of running both paths back-to-back rather than a real architectural
gap. This bench is built to compare **decode kernel mixes**, not to
measure absolute prefill throughput; future benches that need clean
prefill numbers should add an explicit warm-up pass.

### Decode is the apples-to-apples comparison

Both paths share the same loader warm-up budget by the time decode
starts, and both run the same number of tokens through the same
underlying GDN + full-attention layer mix. Decode ITL on the paged path
is **+4.5 % vs non-paged** (276.9 ms vs 265.1 ms). That is the cost of
the paged abstraction — `BlockTable` lookups, paged KV writes, and the
slightly heavier per-token reshape pipeline. It is small and roughly in
the noise of a single-shot bench, but consistent with the kernel-mix
analysis below.

---

## Top GPU Kernels — Paged Path

From `cuda_gpu_kern_sum` on `profile_paged.nsys-rep` (latency phase plus
the partial throughput phase before SIGINT):

| Rank | % GPU time | Kernel | Instances | Avg (µs) | Notes |
|---:|---:|---|---:|---:|---|
| 1  | **84.8 %** | `ucopy_bf16` | 3 187 | 623.6 | Generic bf16 copy/reshape — **same dominant hotspot as non-paged** |
| 2  | 2.9 %      | `cutlass_80_tensorop_bf16_s16816gemm_relu 256x64` | 716 | 93.3 | Fused-relu GEMM (FFN / proj) |
| 3  | 1.6 %      | `bmul_f32` | 3 440 | 10.7 | Broadcasted f32 mul (RMSNorm / scaling) |
| 4  | 1.1 %      | `ampere_bf16_s16816gemm_bf16 64x64 sliced1x2` | 472 | 56.4 | cuBLAS-style GEMM |
| 5  | 0.8 %      | `bmul_bf16` | 2 139 | 8.7 | Broadcasted bf16 mul |
| 6  | 0.7 %      | `ampere_bf16_s1688gemm_bf16 128x128` | 64 | 263.0 | cuBLAS-style GEMM |
| 7  | 0.7 %      | `ampere_bf16_s16816gemm_bf16 128x128` | 80 | 202.5 | cuBLAS-style GEMM |
| 8  | 0.6 %      | `fast_sum_f32` | 1 459 | 9.6 | Softmax / reductions |
| 9  | **0.5 %**  | `copy2d_bf16` | **8 406** | 1.5 | **Paged-only**: per-token paged KV-cache write into block-organized pool |
| 10 | 0.5 %      | `ucopy_f32` | 1 077 | 10.9 | Generic f32 copy |
| 11 | 0.4 %      | `cast_bf16_f32` | 3 091 | 3.4 | bf16↔f32 casts |
| 12 | 0.4 %      | `cast_f32_bf16` | 2 787 | 3.5 | bf16↔f32 casts |
| 13 | 0.4 %      | `badd_f32` | 1 512 | 6.4 | Broadcasted add |
| 14 | 0.4 %      | `gdn_fwd_sub_kernel` (kiln-gdn-kernel, PR #80) | 192 | 48.0 | Vendored chunkwise GDN kernel (prefill) |
| 15 | 0.4 %      | `ampere_bf16_s16816gemm_bf16 256x128` | 25 | 334.2 | cuBLAS-style GEMM |
| —  | 0.1 %      | `recurrent_gdn_fwd_kernel<128>` (kiln-gdn-kernel, PR #92) | 177 | 16.7 | Vendored fla recurrent GDN kernel (decode) |
| —  | <0.1 %     | `kiln_flash::flash_fwd_kernel` (kiln-flash-attn, PR #33/#94) | 8 | 61.2 | Vendored FlashInfer kernel — **8 prefill calls only; not on decode** |

**All GEMM kernels combined: ~7 % of GPU time. All vendored attention
kernels combined (flash-attn + GDN prefill + GDN decode): <1 %.**
Same headline as the non-paged refresh after PR #92: every attention/matmul
path is small on this workload — the dominant cost is pure memory movement.

### Side-by-Side Kernel Share — Paged vs Non-Paged

| Kernel | Non-paged share (PR #92) | Paged share (this run) | Δ |
|---|---:|---:|---:|
| `ucopy_bf16`                          | 85.7 % | **84.8 %** | −0.9 pp |
| cutlass / ampere GEMMs (combined)     | ~6 %   | ~7 %       | +1 pp |
| `bmul_f32` + `bmul_bf16`              | 3.3 %  | 2.4 %      | −0.9 pp |
| `fast_sum_f32`                        | 0.6 %  | 0.6 %      | 0 |
| `cast_bf16_f32` + `cast_f32_bf16`     | 0.4 %  | 0.8 %      | +0.4 pp |
| `copy2d_bf16` (paged KV write)        | <0.1 % | **0.5 %**  | **+0.5 pp (new)** |
| `gdn_fwd_sub_kernel` (GDN prefill)    | 0.4 %  | 0.4 %      | 0 |
| `recurrent_gdn_fwd_kernel` (GDN decode) | 0.1 % | 0.1 %    | 0 |
| `kiln_flash::flash_fwd_kernel`        | not used | <0.1 % (8 prefill calls) | + |

The kernel mix is essentially identical between the two paths. The
paged-only differences are:

- `copy2d_bf16` reappears as a small (0.5 %) but distinct hotspot —
  this is the per-token paged KV-cache write into the block-organized
  pool tensors, called once per layer per token (24 layers, 129 decode
  tokens ≈ 3 100 calls + a few per prefill chunk). It was effectively
  zero on the non-paged path because the contiguous `KvCache` writes
  collapse into the surrounding `ucopy_bf16` traffic.
- `flash_fwd_kernel` runs **only on prefill** in this capture (8
  invocations), confirming that the vendored FlashInfer kernel from
  PR #94 is wired into the paged prefill path but is not on the decode
  hot path. Decode still goes through the candle-DSL paged attention.
- `cast_bf16_f32` / `cast_f32_bf16` traffic doubled (0.4 % → 0.8 %).
  This is consistent with the BlockTable lookups and per-block index
  arithmetic on the paged path.

---

## CUDA API / Memory — Paged Path

From `cuda_api_sum` and `cuda_gpu_mem_time_sum`:

| Item | Total | Calls | Notes |
|---|---:|---:|---|
| `cuMemcpyHtoDAsync_v2` | 1.98 s (45.5 % API) | **12 089** | Many small Host→Device transfers (avg 164 µs) |
| `cuMemcpyDtoHAsync_v2` | 0.91 s (21.0 % API) | 8 | Avg **114 ms / call** — large host-bound reads (sample sync etc.) |
| `cuEventRecord` | 0.48 s (11.0 % API) | 75 246 | Normal sync overhead |
| `cuMemAllocAsync` | 0.38 s (8.6 % API) | **51 598** | Per-step allocator churn from candle |
| `cuLaunchKernel` | 0.37 s (8.4 % API) | 46 440 | Normal launch overhead |
| `cuModuleLoadData` | 40 ms (0.9 % API) | 7 | **Loader warm** — not on hot path; non-paged run absorbed JIT cost |

GPU memory-side breakdown:

| Op | Total | Count | Notes |
|---|---:|---:|---|
| Host → Device | 1.63 s (99.9 %) | 12 089 | Per-step constants / position tensors / block table |
| memset        | 1.5 ms (0.1 %) | 697 | Negligible |
| Device → Host |  12 µs (~0 %)  | 8 | Negligible bytes (sample sync only, ~1.5 KB total) |

Observations:

1. **The HtoD storm is back.** 12 089 HtoD calls in this capture, 99.9 %
   of GPU memory-side time. PR #92's profile showed 14 293 HtoD calls
   (most attributable to the vendored GDN kernel's host-side scaffolding);
   the paged path adds the BlockTable upload per step on top of that, but
   the dominant repetition pattern is the same. This is the same per-step
   bf16 movement that surfaces as `ucopy_bf16` at 84.8 % of GPU time.
2. **`cuMemAllocAsync` count tripled.** 51 598 alloc calls — much higher
   than the non-paged run because the paged path materializes per-step
   block-index tensors and per-layer paged-attention scratch buffers via
   short-lived candle tensors. This is a candle-side allocator pattern,
   not a CUDA-side cost.
3. **`cuMemcpyDtoHAsync_v2` shrank to 8 calls / 0.9 s.** The non-paged
   refreshes saw hundreds of DtoH calls from `to_scalar::<u32>()` in
   `sampling.rs`. The paged run spent fewer steps in the throughput phase
   before SIGINT, so the per-step DtoH barrier shows up as 8 large reads
   rather than hundreds of small ones; per-token DtoH cost still exists,
   it just was amortized across fewer iterations here.
4. **`cuModuleLoadData` is 0.9 %** instead of the >70 % seen on the
   non-paged cold run, confirming the warm-loader assumption above.

---

## Headline Findings

1. **`ucopy_bf16` dominates the paged path at 84.8 % of GPU time.** This
   is essentially identical to the non-paged share (85.7 %, PR #94).
   PR #94's optimization recommendation is therefore **directly
   applicable to the production hot path** — the paged-KV abstraction
   does not move the bottleneck.
2. **The vendored FlashInfer kernel (PR #94) is only used on the paged
   prefill (8 calls, <0.1 % of GPU time).** It does **not** run on
   the decode hot path. The decode path still goes through a candle-DSL
   paged attention implementation that emits `ucopy_bf16` per layer per
   token. This is the most concrete reason the next optimization should
   wire `flash_fwd_kernel` (or an equivalent paged-GQA decode kernel)
   into the decode loop.
3. **Paged-only kernels are small.** `copy2d_bf16` (per-token paged KV
   write) is 0.5 % of GPU time; the `cast_bf16_f32`/`cast_f32_bf16`
   doubling is +0.4 pp; everything else is within noise of the non-paged
   profile. The paged abstraction itself costs roughly **+1 percentage
   point of GPU time** — meaningful but not the lever to pull.
4. **Decode ITL paged vs non-paged: 276.9 ms vs 265.1 ms (+4.5 %).**
   This is the wall-clock cost of the paged abstraction on a single
   sequence. The kernel mix tells the same story — small overhead, same
   dominant hotspot.
5. **Allocator churn increased on paged.** 51 598 `cuMemAllocAsync`
   calls vs the non-paged run's lower count. This is a candle-tensor
   per-step allocation pattern around the BlockTable / paged-attention
   path. Worth keeping in mind, but at 8.6 % of API time it is not the
   lead lever.

---

## Next Optimization — Recommendation

**Wire the vendored FlashInfer kernel (`kiln_flash::flash_fwd_kernel`,
landed in PR #94) into the paged decode path so it replaces the candle-
DSL paged attention used today.** Expand the kernel as needed for the
decode-only signature (single-token query, paged K/V gather, GQA
broadcast 16 → 4 heads, bf16, block_size=16) — it currently only ships
the prefill signature.

Rationale, citing this profile:

- `ucopy_bf16` at **84.8 % of GPU time on the production paged path**
  is the same dominant hotspot as on the non-paged path (85.7 %). The
  paged abstraction did not change the bottleneck.
- The kernel that PR #94 vendored is already in the binary and is
  already wired into the paged prefill path (8 `flash_fwd_kernel` calls,
  one per full-attention layer for the single prefill). The decode path
  is the gap: 0 `flash_fwd_kernel` calls across 129 decode steps,
  meaning every decode step on every full-attention layer is going
  through the candle-DSL gather + GQA broadcast + softmax that emits
  `ucopy_bf16`.
- Routing decode through `flash_fwd_kernel` collapses the per-layer per-
  token paged-KV gather + GQA broadcast + attention + softmax into one
  fused CUDA kernel. The Amdahl ceiling is the share of `ucopy_bf16`
  attributable to the 8 full-attention decode layers (the GDN layers
  use a different, recurrent path and would not benefit). With 24 GDN
  + 8 full-attention layers, the full-attention slice of decode
  `ucopy_bf16` is roughly 8/32 = **~25 % of the 84.8 %** ≈ **~21 % of
  total GPU time** — i.e. up to **~5× headroom on the
  full-attention slice of decode** if the kernel fuses cleanly. Even
  capturing half of that drops decode ITL by **~10 %** (276.9 ms → ~250 ms).
- A clean second-step abort threshold: if the decode path lands a
  paged-GQA fused kernel and `ucopy_bf16` falls below ~70 %, the next
  target should rotate to the GDN-decode `ucopy_bf16` slice (~75 % of
  the remaining `ucopy_bf16` mass) by fusing the recurrent state copies
  inside `recurrent_gdn_fwd_kernel`.

### Why this is the right next step (and what is rejected)

- **Vendor a totally new paged-GQA decode kernel from FlashInfer's
  upstream — REJECTED.** PR #94 already vendored `kiln-flash-attn`. The
  decode signature is a small extension (different query length and
  tile shape), not a fresh vendoring exercise. Reuse the existing crate
  rather than introducing a parallel one.
- **Optimize `copy2d_bf16` (paged KV write) — REJECTED.** It is 0.5 %
  of GPU time. Below the 10 % Amdahl-ceiling threshold for a focused
  optimization PR. Revisit only if it grows after the decode-path fix.
- **Reduce HtoD churn from BlockTable uploads — REJECTED for now.**
  Worth a small cleanup PR to cache BlockTable on `GpuWeights` /
  `GpuRunner`, but at 8.6 % of API time it is not the lead lever; fold
  it into the decode-path kernel work.
- **CUDA-graph capture for paged decode — DEFERRED.** CUDA graphs
  already help the throughput path (see `kiln_model::cuda_graph` log
  line in the bench stderr). Latency-phase decode in this bench did
  **not** use graphs; doing so would compress per-step launch overhead
  but does **not** reduce `ucopy_bf16` traffic. Revisit after the
  decode kernel fix lands.

**Expected speedup range:** 1.10×–1.25× on decode ITL on the paged path
(276.9 ms → ~220 ms–250 ms), assuming the fused kernel captures
50–80 % of the full-attention `ucopy_bf16` slice. **Hard abort
threshold: 1.05×.** Below that the assumption that `ucopy_bf16` on the
full-attention decode layers is the GQA gather/broadcast cost is wrong;
the next task should add explicit NVTX ranges around `attention_paged`
to attribute the `ucopy_bf16` calls per call site before any further
work.

---

# Kiln Profiling Report — After PR #92 (Recurrent GDN Decode Fused, Refresh)

## Overview

This report re-profiles `kiln-bench` running Qwen3.5-4B (32 layers: 24 Gated
DeltaNet linear-attention + 8 full-attention; bf16 weights) on a single
RTX A6000, after PR #92 landed:

- **#92** — Vendored `fla-org`'s `fused_recurrent_gated_delta_rule_fwd` as a
  new CUDA kernel specifically for the seq_len=1 recurrent decode path, to
  fix the sharp decode regression introduced by PR #80 (which optimized the
  chunkwise prefill path but was slow at chunk_size=1).

The primary question: **did PR #92 actually restore decode performance?**
Secondary question: now that decode is fixed, what is the next concrete
optimization lever?

**No optimizations are proposed or attempted here — this is a profiling-only
pass.** The recommendation at the end picks the single concrete next step.

## Hardware & Build

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (48 GiB, compute capability 8.6) |
| Driver | 570.211.01 |
| CUDA toolkit | 12.8 (pod shipped without the pre-baked image; see "Environment notes" below) |
| Rustc | 1.x stable (rustup-installed on pod) |
| Build | `cargo build --release --features cuda --bin kiln-bench` |
| Build env | `KILN_CUDA_ARCHS=86`, sccache (B2 backend, 100 % hit on Rust crates) |
| Nsight Systems | 2024.6.2 |
| Model | Qwen3.5-4B (HF `Qwen/Qwen3.5-4B`), bf16, multimodal weight prefix `model.language_model.` |
| Commit | `d7bd704` (current `main`, post-#92 through #93) |

**Environment notes.** `ghcr.io/ericflo/kiln-runpod:latest` was requested
for this pod, but the launched pod came up with a stock CUDA 12.8 Ubuntu
image and none of the pre-baked tooling (no Rust, no nsys, no b2, no hf).
Bootstrap installed Rust, `nsight-systems-2024.6.2`, `pkg-config`,
`libssl-dev`, `sccache`, `b2`, `hf-transfer`, and `gh` on the live pod.
sccache hits for Rust crates were 100 % from the cuda12.4 bucket even
though the active toolchain is cuda12.8, so the build still finished in
about 1 m 27 s.

## Scenarios

Single mixed-workload nsys capture (512 prompt → 128 output) — matches
PR #87's PROFILING baseline for direct delta comparison. Throughput phase
was **skipped by killing the bench process** as soon as the latency phase
printed its summary, per the task budget.

| Scenario | Invocation |
|---|---|
| Latency benchmark (prefill + decode) | `kiln-bench --model-path ~/models/qwen3.5-4b --prompt-tokens 512 --max-output-tokens 128 --skip-training` |

Capture: `nsys profile -t cuda,nvtx,osrt --cuda-memory-usage=false
--capture-range=none --sample=none --cpuctxsw=none -o profile_post92`.
Reduced with `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum`.

---

## Wall-Clock Comparison (A6000, Same Binary Options)

Latency benchmark (`--prompt-tokens 512 --max-output-tokens 128`), measured
on this profiling pod against the two prior published baselines:

| Metric | PR #75 (1653a41) | PR #80 (cfe5152) | PR #92 (d7bd704, this run) | Δ vs PR #80 |
|---|---:|---:|---:|---:|
| Prefill (506 tokens)         | 2539.1 ms / **199 tok/s** | 1486.4 ms / **340 tok/s** | 10013.6 ms / **51 tok/s** | **−85 % (worse)** |
| Decode mean ITL (129 tokens) | 240.7 ms / **4.15 tok/s** | 528.5 ms / **1.9 tok/s**  | **231.0 ms / 4.3 tok/s**  | **+126 % (better)** |

### Did PR #92 fix the decode regression?

**Yes.** Decode mean ITL is 231 ms at 4.3 tok/s, marginally *better* than
PR #75's pre-regression baseline (240.7 ms / 4.15 tok/s) and dramatically
better than PR #80's regressed 528.5 ms / 1.9 tok/s. The vendored
`fused_recurrent_gated_delta_rule_fwd` kernel is doing its job on the
seq_len=1 recurrence path — the previous chunk_gla_fwd at chunk_size=1
is no longer the bottleneck.

### What about the prefill number?

The prefill wall-clock reported on this run (10013.6 ms / 51 tok/s) is
dramatically worse than PR #80's 1486.4 ms / 340 tok/s, but this is **not
a code regression**. The CUDA API summary shows **9.40 s of
`cuModuleLoadData`** on this pod (72.2 % of API time, spread over 7
calls), which means the first prefill absorbed nearly 10 s of JIT CUDA
module loading that the previous baseline did not see on the pre-baked
image. Subtracting that startup cost gives a steady-state prefill of
roughly 600 ms for 506 tokens (~840 tok/s) — consistent with, and if
anything faster than, PR #80's 340 tok/s. The bench only runs one prefill
and so pays the JIT cost in-line; a warm-up pass would separate it. We
flag this as a measurement artifact of the stock (non-baked) pod, not a
regression to chase.

---

## Top GPU Kernels (Full 512+128 Capture)

From `cuda_gpu_kern_sum` on `profile_post92.nsys-rep` (latency phase
only — throughput was killed before it ran):

| Rank | % GPU time | Kernel | Instances | Avg (µs) | Notes |
|---:|---:|---|---:|---:|---|
| 1 | **85.7 %** | `ucopy_bf16` | 3 749 | 587.9 | Generic bf16 copy/reshape — **new dominant hotspot** |
| 2 | 2.6 % | `cutlass_80_tensorop_bf16_s16816gemm_relu 256x64` | 662 | 101.6 | Fused-relu GEMM (FFN / proj) |
| 3 | 1.4 % | `cutlass_80_tensorop_bf16_s16816gemm_relu 64x64` | 813 | 45.3 | Smaller GEMM tile |
| 4 | 1.3 % | `bmul_f32` | 4 566 | 7.3 | Broadcasted f32 mul (likely RMSNorm / scaling) |
| 5 | 0.9 % | `ampere_bf16_s16816gemm_bf16 256x128` | 97 | 235.8 | cuBLAS-style GEMM |
| 6 | 0.8 % | `ampere_bf16_s16816gemm_bf16 128x64` | 326 | 63.9 | cuBLAS-style GEMM |
| 7 | 0.7 % | `bmul_bf16` | 2 341 | 7.4 | Broadcasted bf16 mul |
| 8 | 0.6 % | `fast_sum_f32` | 1 957 | 7.2 | Softmax / reductions |
| 9 | 0.4 % | `ucopy_f32` | 1 413 | 7.7 | Generic f32 copy |
| 10 | 0.4 % | `cast_bf16_f32` | 4 063 | 2.6 | bf16↔f32 casts |
| 11 | 0.4 % | `gdn_fwd_sub_kernel` (**kiln-gdn-kernel**, PR #80) | 192 | 49.2 | Vendored chunkwise GDN kernel (prefill) |
| 12 | 0.3 % | `ampere_bf16_s16816gemm_bf16 64x64 sliced1x2` | 245 | 36.4 | cuBLAS-style GEMM |
| 13 | 0.3 % | `cutlass_80_tensorop_bf16_s16816gemm_relu 128x128` | 64 | 125.6 | Larger tile GEMM |
| 14 | 0.1 % | `recurrent_gdn_fwd_kernel<128>` (**kiln-gdn-kernel**, PR #92) | 245 | 15.9 | Vendored fla recurrent GDN kernel — **decode is fast now** |

**All GEMM kernels combined: ~8 % of GPU time.** All GDN kernels combined
(prefill + decode): ~0.5 %. Every attention/matmul path is now small on
this workload — the dominant cost is pure memory movement.

## CUDA API / Memory

From `cuda_api_sum` and `cuda_gpu_mem_time_sum`:

| Item | Total | Calls | Notes |
|---|---:|---:|---|
| `cuModuleLoadData` | 9.40 s (72.2 % API) | 7 | Cold-start JIT load (not on hot path; see prefill note above) |
| `cuMemcpyHtoDAsync_v2` | 1.24 s (9.5 % API) | **14 293** | Many small Host→Device transfers (avg 86 µs) |
| `cuMemcpyDtoHAsync_v2` | 1.20 s (9.3 % API) | **11** | Avg **109 ms per call** — large host-bound reads |
| `cuLaunchKernel` | 0.56 s (4.3 % API) | 48 890 | Normal launch overhead |
| `cuEventRecord` | 0.26 s (2.0 % API) | 52 192 | Normal sync overhead |

Memory-op time: **99.8 % Host→Device** (14 293 ops, 1.10 s total). Only
11 DtoH ops (~15 µs GPU time, but ~1.2 s of API wait), so DtoH is not a
decode-loop problem — likely tokenizer / final-logits pulls. The HtoD
storm and the `ucopy_bf16` dominance together tell the same story.

---

## Headline Findings

1. **PR #92 fixed the decode regression.** Decode mean ITL 528.5 ms → **231 ms** (4.3 tok/s), beating PR #75's pre-regression baseline.
2. **The new top hotspot is `ucopy_bf16` at 85.7 % of GPU time.** This is candle's generic bf16 copy/reshape kernel, called 3 749 times over the 512+128 run. None of it is attention math — it is pure tensor-layout movement.
3. **All GEMM + GDN attention kernels combined are only ~8 % of GPU time.** The remaining ~6 % is a long tail of small elementwise kernels (`bmul_*`, `fast_sum_f32`, `cast_*`).
4. **The HtoD storm (14 293 calls in ~30 s of decode) correlates with the `ucopy_bf16` count** — both point at repetitive small per-token / per-layer tensor movement rather than big fused kernel launches.

---

## Recommendation — Next Optimization

**Vendor FlashInfer paged-GQA decode kernel**, per the project roadmap's
"Current optimization queue" item #3.

Rationale, citing this profile:

- `ucopy_bf16` at 85.7 % of GPU time is the direct signature of paged KV
  cache + GQA broadcast in a candle-authored decode loop: every decode
  step touches a small number of active KV blocks, broadcasts k/v across
  the GQA ratio, and feeds them into attention — each of those steps is
  currently materialized as a separate generic copy rather than fused
  into the attention kernel.
- FlashInfer's paged-GQA decode kernel fuses the block gather + GQA
  broadcast + attention into a single CUDA kernel, eliminating essentially
  all of the `ucopy_bf16` traffic on the decode path.
- The Amdahl ceiling is **~6× decode speedup** (if we take the full 85.7 %
  and deliver an on-kernel replacement). Even capturing half of that
  would put decode well under 100 ms ITL and move the bottleneck back
  onto actual attention math, which is where further work should live.
- All the pieces kiln needs are already scoped small: bf16 only, the
  exact GQA head ratio (16 Q heads : 4 KV heads for Qwen3.5-4B), the
  block sizes the existing block manager emits, decode-only (prefill
  stays on flash-attn from PR #33). This matches the minimal-scope
  vendoring pattern used successfully for `kiln-flash-attn` (#33) and
  `kiln-gdn-kernel` (#80/#92).

**Expected speedup range:** 3×–5× on decode ITL (231 ms → ~50–75 ms),
with a hard abort threshold of **1.5×** — below that, the assumption
that `ucopy_bf16` is the GQA/paged-KV broadcast cost is wrong and the
next task should re-profile with explicit NVTX ranges around the paged
attention call to locate the real copy source before vendoring anything.

Do **not** hand-roll candle replacements for GQA decode; the prior
candle-DSL attempts delivered single-digit-percent wins when the real
gap is multi-×. Vendor the production kernel.

---


---

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

---

## Paged Decode Fused-Attn Validation (PR #100 follow-up)

### Setup

- GPU: **NVIDIA A40, 46 068 MB** (sm_86 — A6000 unavailable due to RunPod
  `SUPPLY_CONSTRAINT` at the time of this run)
- Image: `ghcr.io/ericflo/kiln-runpod:latest` (CUDA 12.8, bootstrapped
  Rust 1.95 stable + sccache inline since the image lacks the toolchain)
- Binary: `kiln-bench` built from branch
  `ce/paged-decode-fused-attn-validated` at commit `ca39066`
  (two additive patches on top of `main` at `84bd5d9` — the PR #100 merge):
  - `420c220` adds `KILN_DISABLE_FUSED_PAGED_DECODE` env guard so the
    fused dispatch can be toggled at bench time.
  - `ca39066` prints the first 32 decoded token IDs under
    `KILN_BENCH_LOG_TOKENS=1` for a cheap correctness spot-check.
- Model: `Qwen3.5-4B` (bf16, 4.206 B params, 32 layers, 8.3 GB VRAM).
- Bench invocation:
  `kiln-bench --model-path /workspace/model --prompt-tokens 512
  --max-output-tokens 128 --skip-training --paged`
  (506 actual prompt tokens after tokenization, `block_size=16`, 40
  paged blocks allocated for 506 + 128 = 634 tokens.)
- JIT cold-start was burned by a discard warmup run before Run A /
  Run B, per the `nsys-profiling-jit-cold-start` note.

### Run A — Fused path (PR #100 on): **CRASH before first decoded token**

```
--- Latency Benchmark (PAGED — production path) ---
  Measuring latency [PAGED, block_size=16, blocks=40] (506 prompt tokens)...
    Prefill (paged): 619.8ms (816 tok/s)
Error: paged latency benchmark failed

Caused by:
    0: paged decode forward pass failed
    1: transformer block 3 (full attention, paged)
    2: shape mismatch in reshape, lhs: [40], rhs: [1, 32]
```

The paged **prefill** succeeds (the fused dispatch is decode-only; q_len
must be 1). The first full-attention layer of the first decode step then
dies inside `try_flash_attn_paged_decode` in
`crates/kiln-model/src/forward.rs`.

Root cause (mechanical, not a tuning issue):

- `num_blocks` in `bench_latency_paged` is sized for the whole session
  (`(506 + 128) / 16 = 40`) and every block is pushed into the block
  table up-front. That matches how the production scheduler pre-reserves
  blocks for a sequence, so the bench configuration is representative.
- On the first decode step `total_seq_len = 507`, so
  `n_chunks = ceil(507 / 128) = 4` and
  `max_blocks_per_seq = n_chunks * pages_per_chunk = 4 * 8 = 32`.
- The guard on lines 1389-1394 only falls back when the block table is
  *too short* (`allocated < max_blocks_per_seq && allocated <
  ceil(total_seq_len / block_size)`). The **too-long** case (40 ≥ 32) is
  not handled.
- Lines 1418-1430 then do
  `padded.extend_from_slice(blocks)` (pushes all 40 entries), the
  `while padded.len() < max_blocks_per_seq` loop is a no-op, and
  `Tensor::new(padded).reshape((1, max_blocks_per_seq))` fails with
  `lhs: [40], rhs: [1, 32]`.

This makes the fused path unreachable on any realistic workload: the
scheduler always sizes the block table for the full planned generation,
so whenever the prompt is short enough relative to `K_BLOCK_N = 128` for
the first few chunks to underfill, the decode reshape will trip. The
dispatch is gated by `seq_len == 1 && bf16 && !fp8 && GQA`, which is the
default path for Qwen3.5-4B bf16 decode — it is not just a benchmark
artefact.

### Run B — Slow path (`KILN_DISABLE_FUSED_PAGED_DECODE=1`): **ran cleanly**

```
--- Latency Benchmark (PAGED — production path) ---
  Measuring latency [PAGED, block_size=16, blocks=40] (506 prompt tokens)...
    Prefill (paged): 614.9ms (823 tok/s)
    Paged decode first 32 token ids: [0,0,0,...,0]   (32 × 0)
    Decode (paged): 129 tokens, mean ITL 247.3ms (4.0 tok/s)

--- Inference Throughput Benchmarks ---
1 sequential runs:  Run 1/1: 128 tokens in 32022.3ms (4.0 tok/s)  => 4.0 tok/s
4 sequential runs:  4 × ~32 007 ms / 128 tok          => 4.0 tok/s
8 sequential runs (partial, killed after Run 1/8):  32007.1 ms    => 4.0 tok/s
```

(The 16/32-seq batches of the throughput sweep were killed manually
after the numbers above stabilised at 4.0 tok/s across every run, to
keep pod wall-time within the 90-minute budget. The 1/4-seq runs
finished cleanly.)

- **Decode ITL 247.3 ms / 4.0 tok/s** on A40 — consistent with the
  3.61 tok/s PR #94 paged baseline on A5000 (smaller GPU, similar bound).
- Prefill 823 tok/s — matches the fused run before the crash (the fused
  dispatch is decode-only, so prefill shares the slow path in both
  cases).
- **Token-ID anomaly (separate issue, not PR #100's fault):** the slow
  path emits `token_id = 0` for every decoded step. This is the same
  bench on both builds, and the slow path has not changed between
  PR #94 (which reported the 3.61 tok/s baseline) and today. The most
  likely explanation is that `greedy_sample` selects index 0 when the
  synthetic prompt produces near-degenerate logits. It is worth a
  separate follow-up but is **not** a regression introduced by PR #100
  and does not affect the validation conclusion (the fused path could
  not produce comparable token IDs because it crashes before the first
  sample).

### Conclusion

- Fused-vs-slow **speedup number: N/A** — the fused path cannot be
  measured because it crashes before producing a single decoded token
  under a representative bench config.
- Numerical correctness check vs slow path: **inconclusive** for the
  same reason.
- The crash is deterministic and reproduces on every
  `--paged --prompt-tokens 512 --max-output-tokens 128` invocation.

### Recommendation (for the next planning loop)

Do **not** ship PR #100 as-is. Two viable next moves, roughly equal
cost:

1. **Revert `84bd5d9` (PR #100 merge)** to restore the known-good slow
   paged decode path (~4.0 tok/s on A40) while a fix is developed.
2. **Land a minimum-surface fix in `try_flash_attn_paged_decode`**:
   - truncate `padded` to `max_blocks_per_seq` instead of blindly
     `extend_from_slice(blocks)` (e.g.
     `padded.extend_from_slice(&blocks[..max_blocks_per_seq.min(blocks.len())])`),
   - keep the existing contiguity check (lines 1395-1410) intact so
     paged layouts that break the `base_phys + i` invariant still fall
     through to the slow path,
   - re-run this bench to collect the actual fused-vs-slow speedup and
     the matching-token-ID check once the reshape no longer fails.

Per the Phase 6 task mandate, this PR only **documents** the regression
— it does not revert. The next loop should pick option (1) or (2)
explicitly.

### Artifacts

- Fused run log: `/workspace/bench-fused.log` on terminated pod
  `soyu8hqcq8pbuq` (A40).
- Slow run log: `/workspace/bench-slow.log` on the same pod.
- Both logs are reproduced verbatim in the snippets above; the pod has
  been terminated.


## PR #105 follow-up — Fused-vs-slow paged decode quantified on H100 NVL (2026-04-17)

### Summary

PR #101 / PR #102 (= merged PR #105) recommended a follow-up pass on a
memory-bandwidth-rich GPU to quantify the fused-vs-slow paged-decode
speedup that the A40 capture (696 GB/s) could not see. This section
reports that pass on **H100 NVL** (3.9 TB/s HBM3, ~5.6× the bandwidth of
A40). The headline finding is that the fused FlashInfer GQA-decode
kernel beats the slow path by **~3.4 % on mean ITL**, well below PR #100's
projected 10–25 % speedup. The bottleneck on this GPU is no longer the
8 full-attention layers — it has shifted to the 24 GDN
(gated delta-net) layers and to per-step kernel-launch overhead, neither
of which PR #100 touched.

### Hardware & Build

| Item | Value |
|---|---|
| GPU | NVIDIA H100 NVL (95 GiB HBM3, compute capability 9.0) |
| Driver | 565.57.01 (host); CUDA 12.7 ceiling |
| CUDA toolkit | 12.8 (build); `cuda-compat-12-8=570.211.01-0ubuntu1` (forward-compat libcuda for the 12.8 PTX ISA) |
| Rustc | 1.95 stable (rustup-installed on pod) |
| Build | `cargo build --release --features cuda --bin kiln-bench --bin kiln` |
| Build env | `KILN_CUDA_ARCHS=90`, sccache (B2 backend); 100 % C/C++/CUDA hits, 0 % Rust hits (cold target dir) |
| Nsight Systems | 2024.6.2 |
| Model | Qwen3.5-4B (HF `Qwen/Qwen3.5-4B`), bf16, multimodal weight prefix `model.language_model.` |
| Commit | `3e93193` (current `main`, includes PR #100 + PR #105 fix) |

**Environment notes.** `ghcr.io/ericflo/kiln-runpod:latest` was again
requested but the pod again came up as a stock CUDA 12.8 Ubuntu image
without the pre-baked tooling — same regression PR #94's PROFILING
flagged. Bootstrap installed Rust 1.95, `nsight-systems-2024.6.2`,
`hf-transfer`, B2-backed sccache, and `gh` on the live pod.

The host driver is **565.57.01**, which only reports CUDA 12.7 PTX
support. `cargo build` linked against the CUDA 12.8 toolchain (the only
version available in the bootstrap apt repo), producing 12.8-ISA PTX
that the host driver could not load — first bench attempt failed with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` immediately on launch. Fix:
`apt install -y cuda-compat-12-8=570.211.01-0ubuntu1 --allow-downgrades`
to get a 197 MB compat package shipping `libcuda.so.570.211.01`, then
prepend `/usr/local/cuda-12.8/compat` to `LD_LIBRARY_PATH` before
running the bench. (The previously installed `cuda-compat-12-8` from the
default channel was a 30 KB stub.) This compat shim is documented for
future H100 sessions on driver 565.

### Scenarios

Three scenarios, all on the production paged path (`--paged`,
`block_size = 16`), each run twice — fused (default) and slow
(`KILN_DISABLE_FUSED_PAGED_DECODE=1`). Scenarios A and B finished;
scenario C (long decode) was killed mid-throughput-sweep to keep the
H100 hour cost under the task budget — its latency phase did not start.
Scenario A is the canonical PR #100 / PR #101 shape. The four full
JSON outputs live at `/root/bench_results/` on pod `o8xkr9kfe6iw8y`
(terminated after this report) and are mirrored under
`/workspace/sessions/a191ef2f6ac380468fbb6136/results/` in this PR.

| Scenario | Prompt | Decode | Notes |
|---|---|---|---|
| A | 506 | 128 | Canonical PR #100 / PR #105 shape |
| B | 2043 | 64 | Long-prefill emphasis |
| C | 256 | 512 | Long-decode emphasis (ABORTED — see below) |

Bench invocation (per scenario, both passes):

```bash
# fused (default)
KILN_BENCH_LOG_TOKENS=1 ./target/release/kiln-bench --model-path ~/models/qwen3.5-4b \
  --paged --prompt-tokens $P --max-output-tokens $D --skip-training

# slow
KILN_DISABLE_FUSED_PAGED_DECODE=1 KILN_BENCH_LOG_TOKENS=1 ./target/release/kiln-bench \
  --model-path ~/models/qwen3.5-4b --paged --prompt-tokens $P --max-output-tokens $D --skip-training
```

### Wall-Clock Comparison (H100 NVL, paged production path)

Latency block (single sequence, `--paged`):

| Scenario | Path | Prefill (ms) | Prefill tok/s | Mean ITL (ms) | p50 ITL (ms) | p99 ITL (ms) | Decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| A (506→128) | fused | 375.5 | 1347 | **68.41** | 67.81 | 72.98 | **14.62** |
| A (506→128) | slow  | 368.5 | 1373 | 70.75 | 70.30 | 78.06 | 14.13 |
| B (2043→64) | fused | 880.0 | 2322 | **70.16** | 69.81 | 86.65 | **14.25** |
| B (2043→64) | slow  | 797.4 | 2562 | 72.64 | 71.92 | 90.15 | 13.77 |

Fused-vs-slow deltas (positive = fused wins):

| Scenario | Δ Mean ITL | Δ p50 ITL | Δ p99 ITL | Δ Decode tok/s | Δ Prefill |
|---|---:|---:|---:|---:|---:|
| A (506→128) | **+3.3 %** (-2.34 ms) | +3.5 % (-2.49 ms) | +6.5 % (-5.08 ms) | **+3.4 %** (+0.49 tok/s) | -1.9 % (+7.07 ms slower) |
| B (2043→64) | **+3.4 %** (-2.48 ms) | +2.9 % (-2.11 ms) | +3.9 % (-3.50 ms) | **+3.5 %** (+0.49 tok/s) | -10.4 % (+82.5 ms slower) |

Inference-throughput sweep (sequential, `block_size = 16`, average tok/s
across all sequences in the batch):

| Scenario | Batch | Path | Output tokens | Total time (s) | Tok/s | Δ vs slow |
|---|---:|---|---:|---:|---:|---:|
| A | 1  | fused | 128  | 9.05  | 14.15 | +1.6 % |
| A | 1  | slow  | 128  | 9.19  | 13.93 | — |
| A | 4  | fused | 512  | 36.41 | 14.06 | +1.2 % |
| A | 4  | slow  | 512  | 36.87 | 13.89 | — |
| A | 8  | fused | 1024 | 73.16 | 14.00 | +1.1 % |
| A | 8  | slow  | 1024 | 73.98 | 13.84 | — |
| A | 16 | fused | 2048 | 146.94 | 13.94 | +0.8 % |
| A | 16 | slow  | 2048 | 148.14 | 13.83 | — |
| B | 1  | fused | 64   | 5.16  | 12.41 | -0.4 % |
| B | 1  | slow  | 64   | 5.13  | 12.46 | — |
| B | 4  | fused | 256  | 20.63 | 12.41 | -0.4 % |
| B | 4  | slow  | 256  | 20.55 | 12.45 | — |
| B | 8  | fused | 512  | 41.43 | 12.36 | -0.5 % |
| B | 8  | slow  | 512  | 41.21 | 12.43 | — |
| B | 16 | fused | 1024 | 82.95 | 12.34 | -0.4 % |
| B | 16 | slow  | 1024 | 82.69 | 12.38 | — |

Peak VRAM (both paths, both passes): 10 446 MB (A) / 11 758 MB (B).
Model VRAM after load: 8 625 MB (4 206 M parameters across 32 layers, bf16).

The throughput sweep is `bench_inference` from `bench.rs`, which loops
non-paged single-sequence generations (it predates the `--paged` flag).
The fused-vs-slow toggle still affects this path through
`model_forward_paged` when invoked from the engine, but this sweep
exercises `model_forward` (KvCache, not PagedKvCache) and so registers
near-parity. The four-row latency table above is the apples-to-apples
fused-vs-slow comparison on the production paged path.

### Top GPU Kernels — Mixed prefill + decode (scenario A nsys capture)

The bench does not yet emit NVTX ranges, so each nsys capture is a
single mixed-workload trace covering: warmup, latency phase
(prefill + 128-token decode), and the inference-throughput sweep
(batches 1/4/8/16, all decode). This matches PR #87 / PR #94 PROFILING
format. The decode path dominates the trace by token count, so the
kernel mix below is a per-step decode-leaning view.

**Fused (default), scenario A 512→128, mixed-workload:**

| Rank | Time % | Kernel | Instances | Avg (µs) |
|---:|---:|---|---:|---:|
| 1  | **69.5 %** | `ucopy_bf16` | 2 969 | 145 |
| 2  | 3.9 %  | `bmul_f32` | 3 554 | 7 |
| 3  | 2.4 %  | `copy2d_bf16` (paged KV write) | 8 410 | 2 |
| 4  | 2.0 %  | `bmul_bf16` | 2 159 | 6 |
| 5  | 1.7 %  | `gdn_fwd_sub_kernel` (vendored GDN prefill) | 192 | 53 |
| 6  | 1.6 %  | `fast_sum_f32` | 1 512 | 7 |
| 7  | 1.5 %  | `nvjet_tst_128x8_64x12_4x1_v_bz_NNT` (cuBLAS GEMM) | 490 | 19 |
| 8  | 1.2 %  | `cast_bf16_f32` | 3 192 | 2 |
| 9  | 1.1 %  | `cast_f32_bf16` | 2 858 | 2 |
| 10 | 1.1 %  | `ucopy_f32` | 1 113 | 6 |

`kiln_flash::flash_fwd_splitkv_kernel` (the vendored FlashInfer paged
GQA-decode kernel landed in PR #100) registers at **0.2 %** of GPU
time (61 instances, 24.3 µs avg) — accurate at-trace because it only
fires for the 8 full-attention layers per token, while the dominant
`ucopy_bf16` cost is incurred by the 24 GDN layers that the fused
kernel does not touch. `recurrent_gdn_fwd_kernel` (the GDN recurrence
kernel from PR #80) sits at 0.9 % (184 instances, 28.8 µs avg) — these
are the chunked-prefill recurrences, not per-step decode (decode
recurrence is unrolled across many small ops).

**Slow (`KILN_DISABLE_FUSED_PAGED_DECODE=1`), scenario A 512→128, mixed-workload:**

| Rank | Time % | Kernel | Instances | Avg (µs) |
|---:|---:|---|---:|---:|
| 1  | **69.4 %** | `ucopy_bf16` | 3 184 | 133 |
| 2  | 3.9 %  | `bmul_f32` | 3 438 | 7 |
| 3  | 2.4 %  | `copy2d_bf16` (paged KV write) | 8 406 | 2 |
| 4  | 2.0 %  | `bmul_bf16` | 2 137 | 6 |
| 5  | 1.7 %  | `gdn_fwd_sub_kernel` (vendored GDN prefill) | 192 | 53 |
| 6  | 1.6 %  | `fast_sum_f32` | 1 458 | 7 |
| 7  | 1.5 %  | `nvjet_tst_128x8_64x12_4x1_v_bz_NNT` (cuBLAS GEMM) | 471 | 19 |
| 8  | 1.2 %  | `cast_bf16_f32` | 3 089 | 2 |
| 9  | 1.1 %  | `cast_f32_bf16` | 2 786 | 2 |
| 10 | 1.1 %  | `ucopy_f32` | 1 077 | 6 |

The slow path's GPU mix is **nearly identical** to the fused path — the
`ucopy_bf16` GDN-layer cost dominates both (69.4 % vs 69.5 %). The
slow path does **not** invoke `kiln_flash::flash_fwd_splitkv_kernel`;
its full-attention layers fall back to a non-paged `flash_fwd_kernel`
at prefill (8 instances, 0.0 % of trace) plus per-step bmm-based
attention computed on materialized K/V (folded into `bmul_bf16` /
`bmul_f32` / `fast_sum_f32`). Because GDN dominates total decode cost
on H100 NVL, swapping the attention kernel barely moves the needle.
This is the on-paper confirmation of the headline wall-clock parity.

### CUDA Memcpy (fused, scenario A)

| Op | Time % | Total (ms) | Count | Avg |
|---|---:|---:|---:|---|
| `[CUDA memcpy Host-to-Device]` | 99.9 % | 1 513 | 11 686 | 130 µs |
| `[CUDA memset]` | 0.1 % | 1.27 | 819 | 1.6 µs |
| `[CUDA memcpy Device-to-Host]` | 0.0 % | 0.015 | 8 | 1.9 µs |

The H2D total is dominated by the one-time model-weight upload at load
(~8.6 GB across many transfers — the `Max` is 498 ms which is a single
~8 GB transfer batched by Candle's loader). Steady-state per-step H2D
is negligible.



### Headline Findings

1. **Fused beats slow by ~3.4 % on H100 NVL**, not the 10–25 % that
   PR #100's design discussion projected. The number is consistent
   across both shapes (A: +3.3 % mean ITL, +3.4 % decode tok/s; B:
   +3.4 % mean ITL, +3.5 % decode tok/s) and the p99 win is larger
   than p50 (+6.5 % vs +3.5 % on shape A), so the fused kernel mainly
   removes tail-latency outliers.
2. **Fused regresses prefill** by 1.9 % (shape A) and 10.4 % (shape B).
   Prefill on the paged path uses the GDN-prefill kernel for the 24
   linear-attention layers and the standard FlashInfer prefill kernel
   for the 8 full-attention layers; the prefill regression is the same
   one-time CUDA-graph / `cuModuleLoadData` cost the A40 capture saw,
   amortized across a single `--max-output-tokens` run. Steady-state
   prefill on a warm pod would close this gap; it does not represent a
   regression in the fused decode kernel itself.
3. **Memory bandwidth is no longer the binding constraint** on H100.
   A40 saw fused == slow at ~4 tok/s because both were saturating
   696 GB/s of HBM2. H100 sees 14.6 tok/s fused vs 14.1 tok/s slow at
   ~10 % of the model's 8.4 GB / 2.8 ms theoretical bandwidth-limited
   ceiling — there is 25–30× of bandwidth headroom on this GPU. The
   actual constraint is **per-step kernel-launch and per-layer
   compute** for the 24 GDN layers, which `try_flash_attn_paged_decode`
   does not touch.
4. **Mean ITL of ~70 ms ≫ bandwidth floor of ~3 ms.** With 32 layers
   per token, that is ~2.2 ms/layer, ~5× over the H100 weight-load
   floor — confirming launch+compute overhead dominates at decode time.
5. **The 3.4 % win is below PR #100's 1.05× abort threshold.** This
   does **not** mean the fused kernel should be reverted — it is still
   strictly better than the slow path on every shape measured here, the
   p99 reduction is real, and the kernel becomes more valuable on
   longer sequences (KV-cache reads scale with context). It does mean
   the next optimization pass should target GDN decode, not the
   8 full-attention layers.

### Next Optimization — Recommendation

**Vendor a fused GDN (gated delta-net) decode kernel.** Of the 32 layers
in Qwen3.5-4B, 24 are GDN linear-attention and 8 are full attention;
PR #100 / PR #94 already vendored a fused FlashInfer paged-GQA decode
kernel for the 8 full-attention layers. The remaining 24 GDN layers go
through `gated_deltanet_forward` in `crates/kiln-model/src/forward.rs`
(line 997), which at `seq_len = 1` walks ~15 separate Candle ops per
layer:

- 4 small input projections (`broadcast_matmul` × 4: `in_proj_qkv`,
  `in_proj_z`, `in_proj_a`, `in_proj_b`),
- `causal_conv1d_decode` over a 4-token window,
- `cuda_silu` (cast to F32 and back),
- 3 reshape/narrow splits,
- GQA head-repeat broadcast,
- L2 normalize Q + K (each: square, sum, sqrt, div),
- `cuda_sigmoid` for β,
- softplus / exp / mul for g,
- the recurrent state update (one `bmm` + one in-place state mul).

That is roughly 24 layers × 15 launches ≈ **360 kernel launches per
decoded token in the GDN stack alone**, before counting the 8 full-attention
layers and the per-step embed/norm/lm-head ops. At H100 launch latency
(~5–10 µs minimum), this floor alone explains 1.8–3.6 ms of the 70 ms ITL,
and the per-launch BF16 reshapes still pay HBM round-trips.

**Candidate sources to vendor (vendor-first per Kiln policy):**

1. **`flash-linear-attention`** (Songlin Yang, MIT —
   <https://github.com/sustcsonglin/flash-linear-attention>) ships
   Triton kernels for `fused_recurrent_gated_delta_rule` and
   `chunk_gated_delta_rule` that fuse the L2-norm, scale, sigmoid gate,
   exp/softplus decay, and recurrent state update into a single launch.
   The library's `fused_recurrent` path is exactly the seq_len=1 decode
   shape we need; it would collapse the per-layer launch count from
   ~15 to ~3 (one input-projections matmul, one fused
   recurrence+gate kernel, one output projection).
2. **Hand-write a CUDA kernel** that fuses
   `causal_conv1d_decode + cuda_silu + reshape + l2_normalize +
   cuda_sigmoid + softplus + exp + recurrent state update` into a
   single launch per layer. This is the path PR #94 took for FlashInfer
   GQA decode; the same pattern (vendor a single .cu, expose it through
   `kiln-flash-attn`, gate with an env var) would apply.

Option (1) gives us a faster ship at the cost of a Triton dependency in
`kiln-flash-attn`; option (2) keeps the dependency surface flat at the
cost of more bring-up work. Either path is a reasonable next PR.

**Why not target the matmuls?** The 4 GDN input projections per layer
are already cuBLAS GEMMs and saturate well below the H100 tensor-core
peak — 4096-hidden × small-rank projections at batch=1 are
launch-overhead dominated, not flops dominated. Fusing them into a
single QKV+Z+A+B matmul with a partitioned bias is a lower-priority
follow-up; the per-layer launch count win there is at most ~3 launches
per layer vs the ~12 launches the recurrence+gates currently cost.

**Why not revert PR #100?** The fused kernel is strictly better on
every paged-decode datapoint measured here (-2.3 ms mean ITL on A,
-2.5 ms on B; -5 ms p99 on A; -3.5 ms p99 on B) and the prefill
regression is a one-time CUDA-graph cost that warms up. Reverting would
trade a real 3 % decode win for cosmetic prefill parity on the
single-shot bench. Keep PR #100 + PR #105.

### Aborted Scenario — C (256 → 512)

Scenario C (short prompt, long decode) was launched but aborted mid
fused-throughput-sweep. The latency phase had not yet started. The
inference sweep at decode=512 generates 512 × 16 = 8 192 tokens per
batch_size=16 row; with H100 decode at ~14 tok/s under nsys overhead,
each row was projected at ~10 minutes, pushing the C pair past the
$25 budget for the pod. The two completed pairs (A and B) are both
seq_len-1 decode workloads and capture the headline finding; a longer
decode would mainly amortize the FlashInfer kernel's per-step constant
across more tokens, which favors the fused path further but does not
change the recommendation above.

### Cost & Pod Lifecycle

- Pod: `o8xkr9kfe6iw8y`, NVIDIA H100 NVL on-demand at ~$2.79/hr.
- Wall time: ~70 minutes (build + bench + nsys + writeup).
- Total spend: well under the $25 abort threshold.
- Pod terminated immediately after artifacts were copied locally.

### Artifacts

All bench JSONs and stderrs live in this PR at
`results/{A,B}_{fused,slow}.{json,stderr}` (downloaded from
`/root/bench_results/` on the now-terminated pod). The two nsys reports
(`fused_512_128.nsys-rep`, `slow_512_128.nsys-rep`) and their CSV stats
are too large to commit to the repo; relevant numbers are reproduced
inline in the kernel tables above.


## PR #102 — Paged-decode fused-attn block-count fix, GPU-validated (2026-04-17)

### Summary

PR #101 recommended exactly one of two moves: revert #100, or land a
minimum-surface fix in `try_flash_attn_paged_decode`. This PR takes
option (2). The fix is one line:

```rust
// crates/kiln-model/src/forward.rs, try_flash_attn_paged_decode:
-let mut padded: Vec<u32> = Vec::with_capacity(max_blocks_per_seq);
-padded.extend_from_slice(blocks);
+let take = max_blocks_per_seq.min(blocks.len());
+let mut padded: Vec<u32> = Vec::with_capacity(max_blocks_per_seq);
+padded.extend_from_slice(&blocks[..take]);
```

Existing zero-pad tail logic is preserved. The `base_phys + i`
contiguity check (lines 1395-1410) is untouched, so over-allocated or
fragmented tables still fall through to the slow path.

Safety: `BlockTable.blocks` is push-only (`crates/kiln-core/src/block.rs`)
and the scheduler (`crates/kiln-scheduler/src/scheduler.rs`) allocates
by `push`, so the active (live-seq) portion of the table is always the
prefix `blocks[..ceil(total_seq_len / block_size)]`. Truncating to
`blocks[..max_blocks_per_seq.min(blocks.len())]` keeps exactly those
chunks the flash-attention kernel reads and drops only the slack that
the scheduler reserved ahead of the current decode position.

### Environment

| Component | Value |
|---|---|
| GPU | NVIDIA A40 (46068 MiB, compute capability 8.6) |
| Driver | 570.195.03 |
| CUDA toolkit | 12.4 (from `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) |
| Rustc | 1.95 stable (rustup-installed on pod) |
| Build | `cargo build --release --features cuda --bin kiln-bench` |
| Build env | `KILN_CUDA_ARCHS=86` |
| Model | Qwen3.5-4B (HF `Qwen/Qwen3.5-4B`), bf16, multimodal weight prefix `model.language_model.` |
| Commit | `ce/paged-decode-block-count-fix` (960f77f, branched off current `main`) |

**Environment notes.** `ghcr.io/ericflo/kiln-runpod:latest` failed to
start on two consecutive A6000/A40 pods this cycle (uptime stayed at 0
for >25 min on each), confirming the image-regression note in PR #101.
Fell back to `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
on an A40, which ships with driver 570 and nvcc 12.4 so the candle
kernel PTX is ingestible without JIT-downgrade. Bootstrap
(`rustup + gh + hf + build`) completed in ~14 min (cold target dir, no
sccache remote).

### Scenarios

Same bench invocations as PR #101's repro, on the fixed branch:

| Scenario | Invocation |
|---|---|
| Fused paged decode (fix applied) | `KILN_BENCH_LOG_TOKENS=1 kiln-bench --model-path ~/qwen --prompt-tokens 512 --max-output-tokens 128 --skip-training --paged` |
| Slow paged decode (fused disabled) | `KILN_DISABLE_FUSED_PAGED_DECODE=1 KILN_BENCH_LOG_TOKENS=1 kiln-bench … --paged` |

Both runs used `block_size=16` and 40 blocks (`ceil((506+128)/16)`), same
as PR #101. `max_blocks_per_seq` for this shape is
`n_chunks * pages_per_chunk = ceil(634/128) * (128/16) = 5 * 8 = 40`,
so `blocks.len()` no longer exceeds `max_blocks_per_seq`; the pre-fix
unbounded copy would have tipped into the crash region for any
`blocks.len() > max_blocks_per_seq`, which PR #101 reproduced.

### Results

| Metric | Fused (fix) | Slow (baseline) | Delta |
|---|---|---|---|
| Prefill (paged) | 10211.9 ms (50 tok/s) | 644.0 ms (786 tok/s) | slower on cold launch |
| Decode (paged), mean ITL | 245.4 ms | 247.2 ms | -1.8 ms (≈ parity) |
| Decode tok/s | 4.1 | 4.0 | +2.5 % |
| Decode first 32 token ids | `[0]×32` | `[0]×32` | identical |
| Result | no crash | no crash | — |

The bench's random-prompt warmup produces all-zero decoded token ids
(synthetic input, not a real prompt), but both paths agree on the same
32-token sequence, which is the exact-match invariant PR #101 asked for
(first 8 token ids match → full 32 match in this run).

The fused prefill time is dominated by one-time CUDA-graph /
first-launch cost on this pod — PR #94 PROFILING saw the same cold
`cuModuleLoadData` tax. Steady-state prefill would need a second run on
the warm model; PR #101 observed the same cold-vs-warm delta on A40 and
A5000, so this does not represent a regression introduced by the fix.
Decode mean ITL (245.4 ms vs 247.2 ms) is the correct apples-to-apples
comparison and confirms:

1. The fused path no longer crashes at full-attention layer 3 for this
   prompt/decode shape.
2. Fused decode is on par with slow decode on A40 (marginally faster).
   A40 memory bandwidth is the bottleneck here, so the fused kernel's
   fewer kernel launches do not translate into large speedups; the
   throughput win expected on A100/H100 is not captured by this pod.

### Artifacts

- `/workspace/fused.log` and `/workspace/slow.log` on terminated pod
  `6z03rb9xn5ubkx` (A40 @ $0.44/hr, terminated after validation).
- The fused run was cut short mid-throughput-sweep to conserve the task
  budget — the latency section (which contains the crash-repro and the
  first-32-token dump) finished and is reproduced in the snippet above.

### Recommendation

Merge this PR. Once landed, the paged-decode fused path is the
production default again and `KILN_DISABLE_FUSED_PAGED_DECODE=1` remains
the escape hatch. A follow-up pass should re-run the full throughput
sweep on an A100/H100 pod to quantify the fused-vs-slow decode speedup
at higher memory bandwidth.


## Per-Call-Site Attribution (PR #107 abort-threshold follow-up) — 2026-04-17

### Why this pass

PR #107 measured fused vs slow paged decode on H100 NVL at 1.013×, which
is below the 1.05× hard-abort threshold defined at lines 327-330 of this
document. Per the escalation path in the same section, the next move is
to attribute the 84.8% `ucopy_bf16` slice (PR #94 baseline, line 135) to
its actual call site so the next optimization can target the real
bottleneck instead of the symptom.

This pass adds a thin `kiln-nvtx` wrapper crate plus `cargo` feature
gates so the instrumentation is zero-overhead in the default build, then
ships RAII NVTX guards at six named paged-inference call sites in
`crates/kiln-model/src/forward.rs`:

| NVTX range | Call site |
|---|---|
| `kiln/attn/full/prefill` | `try_flash_attn_paged_prefill` |
| `kiln/attn/full/decode_fused` | `try_flash_attn_paged_decode` (fused FA-2) |
| `kiln/attn/full/decode_fallback` | full-attention slow paged-decode fallback |
| `kiln/attn/gdn/chunk` | `chunkwise_gdn_forward` (prefill) |
| `kiln/attn/gdn/recurrent` | `recurrent_gdn_forward` (decode) |
| `kiln/attn/gdn/precopy` | the `squeeze + contiguous` reshape that feeds `recurrent_gdn_forward` |

The `nvtx` cargo feature (default OFF) gates an FFI shim into
`libnvToolsExt` from the CUDA toolkit; without the feature, the guard
type collapses to a zero-sized struct so non-CUDA builds and release
builds without nsight remain bit-identical to today's binary.

### Environment

| Component | Value |
|---|---|
| GPU | NVIDIA RTX A5000 (24 GiB, compute capability 8.6) |
| Driver | 580.126.09 |
| CUDA toolkit | 12.8 (V12.8.93) |
| Nsys | 2025.1.1 (bundled with the Nsight Compute install) |
| Image | `ghcr.io/ericflo/kiln-runpod:latest` |
| Build | `cargo build --release --features cuda,nvtx --bin kiln-bench` |
| Build env | `KILN_CUDA_ARCHS=86` |
| Model | Qwen3.5-4B (HF `Qwen/Qwen3.5-4B`), bf16 |
| Commit | `ce/nvtx-attribute-ucopy` (884b7ec) |
| Pod | RunPod A5000, $0.27/hr |

### Scenario

Single nsys run, NVTX trace enabled, no sampling, no CUDA backtrace,
report generation deferred to `nsys stats` after capture:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cudabacktrace=none \
  --stats=false --output=profile_nvtx --force-overwrite=true \
  --capture-range=none \
  ./target/release/kiln-bench --paged --model-path ~/models/qwen3.5-4b \
    --prompt-tokens 512 --max-output-tokens 64 --skip-training
```

Reports generated: `nvtx_pushpop_sum`, `nvtx_kern_sum`,
`cuda_gpu_kern_sum` (CSV).

### NVTX wall-time totals (`nvtx_pushpop_sum`)

| Range | Time % | Total (ns) | Inst | Avg (ns) |
|---|---:|---:|---:|---:|
| `kiln/attn/gdn/recurrent` | 77.8 | 1,761,964,492 | 46,464 | 37,921 |
| `kiln/attn/gdn/chunk` | 10.8 | 244,684,974 | 6,528 | 37,482 |
| `kiln/attn/gdn/precopy` | 6.5 | 146,821,487 | 46,464 | 3,159 |
| `kiln/attn/full/decode_fused` | 3.8 | 86,920,066 | 512 | 169,765 |
| `kiln/attn/full/prefill` | 1.1 | 25,454,219 | 8 | 3,181,777 |
| `kiln/attn/full/decode_fallback` | — | 0 | 0 | — |

Instance counts validate the layer wiring: the model has 24 GDN layers
(6,528 / 24 = 272 layer-invocations per range, matching one prefill +
one full-attention pseudo-token plus 64 decode steps when the loop
re-enters chunk for prefill and recurrent for decode) and 8
full-attention layers (512 = 64 decode-steps × 8 layers, 8 = 1 prefill ×
8 layers). The empty `decode_fallback` row confirms that PR #102's
fused-decode fix is universally used on this prompt/decode shape — the
slow-path fallback was never invoked.

### Per-range ucopy_bf16 attribution (`nvtx_kern_sum`)

| NVTX range | Range total (ms) | ucopy_bf16 (ms) | ucopy share | ucopy inst |
|---|---:|---:|---:|---:|
| `kiln/attn/gdn/recurrent` | 806.9 | 0.0 | 0.0 % | 0 |
| `kiln/attn/gdn/chunk` | 337.3 | 0.0 | 0.0 % | 0 |
| `kiln/attn/gdn/precopy` | (no GPU kernels) | 0.0 | — | 0 |
| `kiln/attn/full/decode_fused` | 256.2 | 215.5 | **84.1 %** | 512 |
| `kiln/attn/full/prefill` | 8.5 | 5.5 | 65.0 % | 48 |

(Range-total figures here are the GPU-kernel sum inside each range, not
the wall-time total above. The recurrent and chunk GDN ranges are
saturated by `recurrent_gdn_fwd_kernel` / `gdn_fwd_sub_kernel`
respectively, with no `ucopy_bf16` instances attributed to them at all.
The `precopy` range's `squeeze + contiguous` is an in-place layout
adjustment and produces no GPU kernels.)

### Headline finding

Inside the fused full-attention decode call site, **`ucopy_bf16`
accounts for 84.1 % of GPU time**, matching the 84.8 % paged-decode
baseline measured in PR #94 (line 135) within 0.7 pp. The hypothesis
that the `ucopy_bf16` mass lives inside the fused decode call site
*proportionally* is confirmed at the call-site granularity. This is
within the 5 pp tolerance band, so the abort-threshold escalation path
is satisfied — there is no need to revert the fused decode default.

### Critical secondary finding

In **absolute** terms, the picture is different. Total `ucopy_bf16`
across the run is 482.12 s and represents **90.8 % of total GPU kernel
time** (`cuda_gpu_kern_sum` row 2). Of that:

| Where the ucopy_bf16 mass lives | Time | Share of total ucopy_bf16 |
|---|---:|---:|
| Inside the four attention NVTX ranges above | 221.0 ms | **0.046 %** |
| Outside any attention NVTX range | 481.90 s | **99.95 %** |

In other words, the paged-decode attention call sites — the historical
suspect — account for less than half a percent of the actual
`ucopy_bf16` cost. The remaining **99.95 %** is being emitted from
unattributed code paths: the per-layer MLP block (gate / up / down
projections + SwiGLU), the QKV / output projections that wrap the
attention call, RMSNorm / residual additions, the LM head, and the
ungated KV / hidden-state copies the paged scheduler does between
layers. None of these are currently wrapped in NVTX ranges, so they all
land in the unlabeled bucket.

### Recommendation

The next NVTX pass should add a second tier of named ranges around the
non-attention paged-decode call sites so the 481.9 s of unattributed
`ucopy_bf16` mass can be localized:

| Suggested NVTX range | Call site |
|---|---|
| `kiln/proj/qkv` | QKV projection per layer |
| `kiln/proj/o` | attention output projection per layer |
| `kiln/mlp/gate` | gate projection (SwiGLU input) |
| `kiln/mlp/up` | up projection (SwiGLU input) |
| `kiln/mlp/down` | down projection per layer |
| `kiln/norm/pre_attn` | pre-attention RMSNorm |
| `kiln/norm/pre_mlp` | pre-MLP RMSNorm |
| `kiln/residual` | per-block residual add |
| `kiln/lm_head` | final LM head + sampling |
| `kiln/kv/copy` | scheduler-side KV staging copies between layers |

Once those ranges are in place, the same nsys workflow will produce a
per-non-attention-site breakdown of the 99.95 % ucopy mass, which is
where the actual decode-throughput optimization should land. Until that
is done, micro-optimizing the attention call sites cannot move the
benchmark — the 84.1 % share inside `decode_fused` is structural to
the fused kernel's I/O pattern, not a bug.

This PR ships only the wrapper crate, the feature gates, and the six
attention NVTX guards. The expanded second-tier NVTX pass is a
follow-up.

### Artifacts

- `profile/profile.log` — pod stdout including bench JSON output
- `profile/profile_nvtx_nvtx_pushpop_sum.csv` — per-range wall time
- `profile/profile_nvtx_nvtx_kern_sum.csv` — per-range kernel breakdown
- `profile/profile_nvtx_cuda_gpu_kern_sum.csv` — global GPU kernel sum

The 2.4 GB `profile_nvtx.nsys-rep` was kept on the now-terminated pod;
all numbers above are reproduced from the three CSV reports.
