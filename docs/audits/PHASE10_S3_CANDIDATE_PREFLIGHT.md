# Phase 10 §3 — Next kernel-fusion candidate preflight (post-#647)

Date: 2026-04-29
Status: **NO KERNEL — PIVOT.** Post-PR-#647 SFT-step re-profile at T=8192
shows none of the remaining Phase 10 priority kernels (RoPE, SwiGLU/GeGLU,
Layer Norm, FleCE) clear the math-ceiling floor for §3 implementation
on either A6000 (sm_86, 48 GiB) or A100 80GB (sm_80). The kernel-fusion
track for SFT T=8192 is exhausted; Phase 10 §3 should pivot to non-kernel
work (Phase 9 release prep, T=16384 evidence-gathering for FleCE) until
new evidence reopens a candidate.

Hardware: NVIDIA A100-SXM4-80GB (sm_80, driver 565.57.01), CUDA 12.4.
A6000/A40 capacity-blocked at task start (warm pool pods
`yzf1plx0m6z092` and `18x6how2blkqon` both returned `SUPPLY_CONSTRAINT`
on resume); A100 80GB SXM4 was the next allowed fallback per the task
description (NOT L40S, NOT RTX 6000 Ada). All measurements report on
A100; the A6000 cross-check is documented in §A6000 implications below
and remains the next planning-cycle item if any candidate looks worth
testing further.
Branch: `audit/phase10-section3-preflight-post-647`
Commit at task start: `3504a06` (PR #648 — FLCE Phase B A6000 closure
deferred).

## TL;DR

* **Three back-to-back nsys profiles on A100 SXM4-80GB; medians are
  bit-stable (per-run total kernel time within 0.04% of median).** SFT
  step time: 30.03s / 30.98s / 30.98s; final loss bit-identical
  (0.8115726113319397) across all three; peak VRAM 70,053-70,081 MiB.
  KILN_USE_FLCE=1, KILN_W4A16 unset (W4A16 + training currently fails
  with "expected rank-2 q_proj_t" on the trainer's Marlin unpack path —
  unrelated, but noted), RMSNorm fused CustomOp2 path active (the A100
  has 80 GiB total VRAM, comfortably above the 47 GiB gate threshold
  set by PR #644).
* **Top NVTX regions are GDN-dominated, as expected.** `:kiln/gdn/in_proj`
  19.6% (mostly the matmul that feeds the chunked-prefill GDN cell),
  `:kiln/mlp/gate` 11.8%, `:kiln/gdn/qk_norm` 10.8%, `:kiln/gdn/gates`
  10.4%, `:kiln/mlp/up` 8.9%, `:kiln/mlp/down` 7.9%. GDN aggregate
  ~50%; MLP trio ~28.6%; full-attn projections (`proj/qkv` + `proj/o`)
  ~7.2%.
* **Top kernel hotspot is FP32 SGEMM, not the fused-kernel candidates
  on the Phase 10 list.** `ampere_sgemm_64x64_nn` at 31.07% of kernel
  time (720 invocations, 9.1 ms each — these are LoRA-adapter
  forward/backward in FP32, not lm_head: shape signature matches the
  rank-8 LoRA-down `(T=8192, H=2560) @ (H, R=8) → (T, R)` and
  LoRA-up `(T, R) @ (R, H) → (T, H)` patterns). Combined with
  `ampere_sgemm_128x64_nn` at 11.02% (704 calls), FP32 LoRA matmuls
  account for **~42% of kernel time at SFT T=8192**. The kiln-vendored
  `gdn_full_chunk_forward_kernel` is at 22.13% (10752 invocations) —
  the Phase 6 post-PR-#502 audit (2026-04-24) declared this kernel
  structural-only with no bounded micro-port remaining.
* **Per-candidate math-ceiling verdicts at SFT T=8192:**
  * ❌ **Fused RoPE** — lives inside `:kiln/proj/qkv` (5.2% of step).
    A 2× RoPE-only speedup buys 0.026 step-fraction → 1.027× overall.
    Below the 1.05× CUDA-graph-amortized floor.
  * ❌ **Fused SwiGLU/GeGLU activation** — the elementwise `silu(gate) *
    up` pieces are scattered across `bmul_bf16` (1.55%), `affine_f32`
    (1.19%), and a sliver of `bmul_f32` (2.33%); attribution is also
    diluted because the matmuls in `:kiln/mlp/gate / mlp/up / mlp/down`
    are the actual cost (~28.6% combined) and Liger's SwiGLU fusion does
    NOT eliminate those — it only fuses the post-gate activation
    elementwise. Generous accounting (5% combined, 2× speedup) gives
    1.025× overall. Below floor.
  * ❌ **Fused Layer Norm** — N/A. Qwen3.5-4B uses RMSNorm only.
    PR #644 already shipped the fused-RMSNorm CustomOp2 path active
    on A100 in this profile (see `kiln rmsnorm gate fused_path=ON` in
    the bench log). No LayerNorm operator to fuse.
  * ⚠️ **FleCE (chunked-vocab cross-entropy)** — kiln's FLCE Phase B
    (PR #647) already chunks along the **time** axis to break the
    autograd-retention chain. FleCE adds chunking along the **vocab**
    axis (V=248,320) on top. At T=8192 on A100 80GB peak is 70/80 GiB
    (no constraint), so FleCE is null on this hardware. A6000 A100
    parity is unknown for this profile; on A6000 48 GiB at T=8192,
    Phase B closes peak below 49 GiB ceiling already (PR #647 + #648
    A40 evidence + the open A6000 closure on capacity), so FleCE is
    null at T=8192 there too. **FleCE only buys new ground at
    T≥16384 on A6000**, which has not been measured. Conditional on
    that evidence — see §"Conditional reopen for FleCE" below.

* **The actual top hotspot at SFT T=8192 (FP32 SGEMM ~42%) is NOT a
  kernel-fusion candidate from the Phase 10 priority list.** Replacing
  it would require either (a) running LoRA in BF16 with FP32 accumulate
  (a numerical-stability change to the LoRA training contract, not a
  kernel fusion), or (b) fusing LoRA into the base GEMM (a major
  scheduler/memory-layout change in `kiln-train`'s autograd path).
  Neither matches the §3 scope. Recording it here for future planning.

## Bench protocol

* **Pod**: A100-SXM4-80GB SXM, fresh pool spawn, lease
  `pod-07296529e763818cd1e37c9e`, runpod id `6c8mntz491y8ld`,
  $1.49/hr.
* **Pre-baked image**: `ghcr.io/ericflo/kiln-runpod:latest` (CUDA 12.4
  toolkit, sccache 0.9.1 to B2 `clouderic/build-cache/kiln/x86_64-linux-cuda12.4/sccache`).
* **Build**: `KILN_CUDA_ARCHS=80 cargo build --release --features
  cuda,nvtx --example phase10_rmsnorm_bench -p kiln-server`. One
  Cargo.toml fix needed on the pod: `kiln-server`'s `cuda` feature did
  not propagate to `kiln-train/cuda`, so `kiln-flce-kernel` was building
  without the `cuda` feature. Local-only patch on the pod (NOT in this
  PR's diff): `cuda = ["kiln-model/cuda", "kiln-train/cuda"]` in
  `crates/kiln-server/Cargo.toml`. Pre-existing
  `phase10_flce_phase_b_t8192_only.py` script applied to make the bench
  run only the T=8192 RMSNorm-on cell. Build time: 2m 01s cold (with B2
  sccache), 15s incremental for the Cargo.toml fix.
* **Bench runner**: existing `crates/kiln-server/examples/phase10_rmsnorm_bench.rs`,
  patched to single-cell T=8192-RMSNorm-on via the upstream
  `phase10_flce_phase_b_t8192_only.py` script. NVTX ranges emitted
  from `crates/kiln-model/src/forward.rs` (the kiln-train path itself
  has no NVTX ranges, so backward attribution is via raw kernel names).
* **nsys**: 2024.5.1 (downloaded fresh from
  `developer.download.nvidia.com`; the pre-baked image ships 2023.4.4
  which fails to import the .qdstrm output of long SFT steps with
  "EventCollection::CheckOrder: Wrong event order has been detected" —
  see the kiln agent note `kiln-nsight-profiling-gotchas`). Captured
  3× back-to-back with `-t cuda,nvtx --capture-range=none --sample=none
  --cpuctxsw=none`; each output ~64 MB.
* **Stats**: `nsys stats --report nvtx_sum,cuda_kern_exec_sum --format
  csv` per run. Per-run kernel-time totals: 21.113s / 21.118s / 21.121s
  (Δ 0.04% of median — the runs are profiler-grade reproducible). The
  ~9s gap between kernel time (21.1s) and SFT-step wall time (30.0s) is
  CPU-side autograd graph construction + 4-segment grad-checkpoint
  recompute orchestration, which has no NVTX coverage today.

## Median NVTX (3 runs, A100 80GB SXM4)

| Range                              |    % | Median ns         | Inst |
|-----------------------------------|-----:|------------------:|-----:|
| `:kiln/gdn/in_proj`                | 19.6 |   148,848,039     |   84 |
| `:kiln/mlp/gate`                   | 11.8 |    90,075,835     |  112 |
| `:kiln/gdn/qk_norm`                | 10.8 |    81,527,461     |   84 |
| `:kiln/gdn/gates`                  | 10.4 |    77,934,441     |   84 |
| `:kiln/mlp/up`                     |  8.9 |    68,318,156     |  112 |
| `:kiln/mlp/down`                   |  7.9 |    59,632,641     |  112 |
| `:kiln/norm/pre_attn`              |  6.0 |    45,963,293     |  112 |
| `:kiln/gdn/gated_norm`             |  5.2 |    39,464,361     |   84 |
| `:kiln/proj/qkv`                   |  5.2 |    38,923,323     |   28 |
| `:kiln/gdn/qkv_split`              |  3.0 |    22,465,216     |   84 |
| `:kiln/gdn/conv`                   |  2.2 |    16,671,217     |   84 |
| `:kiln/proj/o`                     |  2.0 |    15,343,928     |   28 |
| `:kiln/gdn/head_expand`            |  1.9 |    14,176,496     |   84 |
| `:kiln/residual`                   |  1.6 |    11,797,805     |  224 |
| `:kiln/gdn/conv/prefill_update`    |  1.4 |    10,455,536     |   84 |
| `:kiln/gdn/conv/layout`            |  0.7 |     5,543,309     |   84 |
| `:kiln/gdn/out_proj`               |  0.5 |     3,602,797     |   84 |
| `:kiln/norm/pre_mlp`               |  0.4 |     3,228,178     |  112 |
| `:kiln/gdn/recur_prep`             |  0.2 |     1,274,609     |   84 |
| `:kiln/gdn/post_transpose`         |  0.2 |     1,154,361     |   84 |
| `:kiln/final_rmsnorm`              |  0.0 |       128,275     |    4 |

Aggregates:
* **GDN family**: ~50.6% (in_proj 19.6 + qk_norm 10.8 + gates 10.4 +
  gated_norm 5.2 + qkv_split 3.0 + conv 2.2 + head_expand 1.9 +
  conv/prefill_update 1.4 + conv/layout 0.7 + out_proj 0.5 +
  recur_prep 0.2 + post_transpose 0.2). On the post-#481 plain-decode
  baseline (PROFILING.md 2026-04-24, A40), GDN aggregate was 49.2% — so
  GDN dominance is preserved across decode-vs-SFT and across
  A40-vs-A100, with `gdn/in_proj` notably higher in this SFT trace
  (19.6%) than in the decode trace (~7.4%) because SFT runs the full
  chunked-prefill path through `gdn_full_chunk_forward_kernel` and
  amortizes more time inside the in-proj matmul.
* **MLP trio**: 28.6% (gate 11.8 + up 8.9 + down 7.9).
* **Full-attn projections**: 7.2% (qkv 5.2 + o 2.0).
* **Norm overhead**: 6.4% (pre_attn 6.0 + pre_mlp 0.4) — note this is
  the pre-norm CALL site, not the kernel; `:kiln/norm/pre_attn` median
  is 28.5 µs/instance and 401 µs avg (the avg is dragged up by 84
  outlier instances of ~40 ms each that align with grad-checkpoint
  recompute boundaries — a known grad-checkpoint timing artifact, not
  a hot kernel).

## Median CUDA kernel exec (3 runs, A100 80GB SXM4)

Total kernel exec time per run: 21.113s / 21.118s / 21.121s. Step
wall-clock: 30.03s / 30.98s / 30.98s. The non-kernel ~9s is CPU-side
autograd construction + recompute orchestration.

| Kernel                                                              | Count | KAvg ns      |     %  |
|---------------------------------------------------------------------|------:|-------------:|-------:|
| `ampere_sgemm_64x64_nn`                                             |   720 |   9,113,701  | 31.07  |
| `gdn_full_chunk_forward_kernel` (kiln-gdn-kernel)                   | 10,752|     434,683  | 22.13  |
| `ampere_sgemm_128x64_nn`                                            |   704 |   3,305,653  | 11.02  |
| `badd_bf16`                                                         | 1,828 |     431,224  |  3.73  |
| `ucopy_bf16`                                                        | 64,957|      10,742  |  3.30  |
| `bsub_f32`                                                          | 1,612 |     416,553  |  3.18  |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_nn`       |   588 |   1,021,057  |  2.84  |
| `bmul_f32`                                                          | 1,496 |     329,106  |  2.33  |
| `ucopy_f32`                                                         | 2,460 |     158,082  |  1.84  |
| `bdiv_f32`                                                          |   432 |     882,056  |  1.80  |
| `fast_sum_f32`                                                      |   968 |     339,703  |  1.56  |
| `bmul_bf16`                                                         |   436 |     750,391  |  1.55  |
| `badd_f32`                                                          | 2,708 |     116,557  |  1.49  |
| `affine_f32`                                                        | 1,600 |     157,628  |  1.19  |
| `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x...`      | 43,008|       5,458  |  1.11  |
| `cast_f32_bf16`                                                     | 1,136 |     200,933  |  1.08  |
| `cast_bf16_f32`                                                     | 1,072 |     206,335  |  1.05  |
| `uexp_f32`                                                          | 1,464 |     114,931  |  0.80  |
| `fast_max_f32`                                                      |   488 |     330,346  |  0.76  |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nt`       |   144 |   1,118,479  |  0.76  |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn`       |   112 |   1,349,516  |  0.72  |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn`       |    64 |   1,348,623  |  0.41  |
| `kiln_flash::flash_fwd_kernel<...>`                                 |    28 |   2,962,117  |  0.39  |

* **`ampere_sgemm_64x64_nn` + `ampere_sgemm_128x64_nn` = 42.09%** —
  cuBLAS FP32 SGEMM kernels. The shape signatures (64x64 and 128x64
  tiles, FP32 input/accumulate) match the rank-8 LoRA-down/-up
  matmuls in `kiln-train`'s autograd path. Phase B's CustomOp does
  NOT change LoRA's storage dtype; LoRA params remain FP32 for
  numerical stability of the small-rank matmul accumulation.
* **`gdn_full_chunk_forward_kernel` = 22.13%** — kiln's vendored
  mamba-ssm causal chunk kernel (PR #80). The Phase 6 post-#502
  audit (2026-04-24, `gdn-vllm-audit.md`) declared this
  structural-only after the vLLM Triton diff. No bounded micro-port
  remains.
* **Elementwise `bf16/f32` zoo (`badd`, `bsub`, `bmul`, `bdiv`,
  `uexp`, `fast_sum`, `fast_max`, `affine`, `cast_*`, `ucopy_*`)
  combined ≈ 24.6%** — these are scattered across many sites
  (autograd-induced cast-and-copy, softmax/CE inner loops, residual
  bias adds, layer-norm scaling). PR #117/#128/#130/#222/#502/etc
  have already addressed the highest-volume sites; the remaining
  spread is sub-floor per-site.

## Per-candidate math-ceiling verdicts

The math-ceiling rule (kiln skill §"Mandatory preflight"):
**region% × (1 − 1/expected_speedup) ≥ 5% (1.05× under graphs floor) or
≥ 10% (1.10× under graphs floor for kernel fusion)**, expressed as the
overall step-time speedup if the candidate kernel hits its ceiling.
NOTE: under CUDA graphs, dispatch-amortization kills wins ≤1.03×
(see agent note `cuda-graph-dispatch-amortization-kills-small-fusions`).
The inference-mode default is 1.10×; for SFT-step backward we use the
1.05× floor since CUDA graphs are NOT being captured around the
backward path today (the current bench shows 720 cuLaunchKernel calls
for the dominant kernel, which would be ~1 cuGraphLaunch under graph
capture).

| Candidate              | Region(s) attribution               | Step % | Generous ceiling | Overall speedup | Floor       | Verdict |
|------------------------|-------------------------------------|-------:|-----------------:|----------------:|-------------|--------:|
| Fused RoPE             | inside `:kiln/proj/qkv`             |   5.2% | 2.0× RoPE-only   |        1.027×   | 1.05× floor | ❌ DEFER |
| Fused SwiGLU/GeGLU     | `bmul_bf16`+`affine_f32`+slice of `bmul_f32` (post-gate elementwise only) | 5.0% | 2.0× activation-only | 1.025× | 1.05× floor | ❌ DEFER |
| Fused Layer Norm       | N/A — Qwen3.5 uses RMSNorm only    |   0.0% | N/A              |        N/A      | N/A         | ❌ N/A   |
| FleCE (chunked vocab)  | Peak-VRAM only at T≥16384 on A6000  |   0.0% step time at T=8192 | N/A on A100, conditional on A6000 T=16384 evidence | 1.000× at T=8192 | 1.05× floor on speed; peak-VRAM saving is the actual benefit | ⚠️ CONDITIONAL |

### Why RoPE fails

`:kiln/proj/qkv` is 5.2% of step time and includes the q/k/v matmuls
**plus** the in-proj RoPE rotation. The matmul cost dominates the RoPE
cost inside that range (in the post-#481 decode trace, the qkv proj
range was 7.4% on A40 and the matmul was the bulk; RoPE itself, when
profiled separately, has been observed as <1% of step time at
T=8192). Even an aggressive 4× RoPE-only speedup over a 1% RoPE budget
is 0.0075 step-fraction → **1.0075× overall**. Sub-1.05×.

### Why SwiGLU fails

Liger-Kernel's SwiGLU fusion (`liger_swiglu_*`) fuses **only** the
post-gate elementwise `silu(gate) * up`, NOT the three MLP matmuls
(`mlp/gate`, `mlp/up`, `mlp/down`). Mapping to kiln kernels:
* `bmul_bf16` (1.55%) — the `gate * up` half of SwiGLU.
* `affine_f32` (1.19%) — sigmoid/silu approximation. Note the FP32 dtype
  here is consistent with kiln keeping the activation in F32 for
  numerical stability inside the activation; Liger fuses this into
  BF16 with FP32-internal-only.
* Slice of `bmul_f32` (2.33%) — partly residual scaling, partly post-
  layer-norm scaling, partly SwiGLU. Generous attribution: ~1% to
  SwiGLU.

Sum: ~3.7% step-time budget. 2× speedup buys 0.0185 step-fraction →
**1.019× overall**. Sub-1.05×.

The 28.6% MLP trio is the actual MLP cost; ~25% of that is the matmul
itself which Liger does NOT fuse.

### Why Layer Norm is N/A

Qwen3.5-4B uses RMSNorm only. PR #644 already shipped the fused-RMSNorm
CustomOp2 path with VRAM-gated activation (≥47 GiB total VRAM); the
A100 in this profile has 80 GiB, so the gate is ON
(`fused_path=ON` in the bench log). There is no LayerNorm operator
in this model architecture.

### Why FleCE is conditional

Liger's FleCE is FLCE-with-vocab-chunking: it chunks the lm_head
matmul along the V (vocab) axis on top of the T (time) axis chunking
that kiln Phase B already does. This buys **peak VRAM** savings (the
[T, V_chunk] per-chunk logits is smaller than [T, V] — for V=248,320
and T=8192 chunked into 8 V-chunks, peak chunk logits drops from 16 GiB
F32 to 2 GiB F32). It does NOT meaningfully change kernel-exec time
at the lm_head matmul level (the same FLOPs cross the SMs; just split
across more launches; CUDA graph dispatch amortizes the launch cost).

* **At T=8192 on A100 80GB**: peak 70 GiB / 80 GiB ceiling = no
  pressure. FleCE is null here (peak-VRAM saving is unused VRAM).
* **At T=8192 on A6000 48GiB**: PR #647 + #648 (A40 closure +
  pending A6000 closure) suggests Phase B already brings peak below
  49 GiB ceiling. FleCE is null here too.
* **At T=16384 on A6000 48GiB**: Peak with Phase B alone is
  predicted to be ~47-50 GiB (linear scaling of the autograd-saved
  intermediates with T, modulo the 4-segment grad checkpoint). This
  is right at the OOM threshold. FleCE could unblock T=16384 by
  dropping per-V-chunk logits cost from 32 GiB F32 (T=16384 × V) to
  4 GiB F32 (8 chunks). **But this has not been measured.**

The **conditional reopen** for FleCE is to first run a T=16384
SFT-step on A6000 (or A40, since the post-fused-RMSNorm path is
arch-equivalent for VRAM purposes) WITHOUT FleCE and observe the
OOM cell. If T=16384 OOMs even with Phase B, FleCE has bounded
peak-VRAM-only ROI. If T=16384 closes cleanly, FleCE has zero ROI.
Both outcomes are cheap to gather (~15 min A6000 lease, no kernel
work).

## A6000 implications

The profile was captured on A100 80GB SXM4, not A6000 48GB. The
top-level NVTX/kernel ratios are expected to be **directionally
identical** on A6000:
* GDN dominance (~50%) is structural (kernel mix is the same).
* MLP trio (~28.6%) is structural.
* Fused RMSNorm CustomOp2 path is ON above 47 GiB total VRAM. A6000
  total VRAM is 49,140 MiB, so the gate is borderline — this profile
  cannot tell us whether the A6000 is on the unfused or fused path
  for the bench's specific run. (The post-#644 closure profile from
  PR #645 confirms A6000 takes the fused path.)
* FP32 SGEMM (LoRA, ~42%) is structural — A6000 has tensor cores at
  the same FP32 rate as A100 and would show similar absolute
  attribution for FP32 matmuls.
* `gdn_full_chunk_forward_kernel` (22%) is structural.

What MIGHT shift on A6000:
* Step wall-clock will be ~1.5-2.0× longer (A100 has ~3× the SM
  count and 2× the HBM bandwidth of A6000). The 30s A100 step
  becomes ~45-60s on A6000.
* Peak VRAM in this bench was 70 GiB / 80 GiB on A100; peak on
  A6000 is gated to 47 GiB / 49 GiB by the fused RMSNorm path
  (which the A100 profile took as well; the gate is global VRAM,
  not used VRAM).

The audit recommendation does NOT change for A6000 — the
math-ceiling math is invariant under the wall-clock rescale.

## Verdict

**No kernel — pivot.**

* **❌ DEFER (sub-floor)**: Fused RoPE, Fused SwiGLU/GeGLU.
* **❌ N/A**: Fused Layer Norm.
* **⚠️ CONDITIONAL**: FleCE — gate on a T=16384 A6000 SFT-step OOM
  measurement first. Cheap (~15 min lease), no kernel work, produces
  a definitive go/no-go for FleCE in one cycle.
* **⏸️ PIVOT**: Phase 9 release prep (security audit, license review,
  reproducible builds, CI/CD pipeline, Docker GHCR image, landing
  page, v0.1.0 cut). The Phase 9 task list in the project description
  is concrete and unrelated to kernel work; doc-only / CI-only tasks
  there are cheap and ship the v0.1.0 release.

The actual top kernel hotspot at SFT T=8192 (FP32 LoRA SGEMM, ~42%)
is NOT a Phase 10 candidate. Two non-Phase-10 follow-ups
(LoRA-in-BF16-with-FP32-accum, or LoRA-fused-into-base-GEMM) are
recorded for future planning but not recommended as §3 work — both
are major scheduler/autograd changes.

## Reproduction

```bash
# Pod (A6000 first; A100 80GB SXM4 if A6000/A40 capacity-blocked, NOT L40S):
ce kiln-pod-acquire --gpu-type "NVIDIA RTX A6000"   # or NVIDIA A100-SXM4-80GB

# On the pod — kiln-server cuda feature must propagate to kiln-train (one-line
# Cargo.toml fix; see Bench protocol above; not in this PR's diff):
sed -i 's|^cuda = \["kiln-model/cuda"\]$|cuda = ["kiln-model/cuda", "kiln-train/cuda"]|' \
  /workspace/kiln/crates/kiln-server/Cargo.toml

cd /workspace/kiln
python3 scripts/phase10_flce_phase_b_t8192_only.py    # T=8192-only patch

# Build (sm_80 for A100, sm_86 for A6000/A40):
source /root/.kiln-build-env
KILN_CUDA_ARCHS=80 cargo build --release --features cuda,nvtx \
  --example phase10_rmsnorm_bench -p kiln-server

# nsys 2024.5.1 (the pre-baked image's 2023.4.4 fails to import the
# .qdstrm output — see kiln-nsight-profiling-gotchas):
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2024.5.1_2024.5.1.113-1_amd64.deb \
    -O /tmp/nsys-2024.5.1.deb
dpkg -i /tmp/nsys-2024.5.1.deb 2>/dev/null || true
NSYS=/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys

# 3× back-to-back nsys captures, take median:
mkdir -p /tmp/profiles/post-647
for i in 1 2 3; do
  KILN_USE_FLCE=1 $NSYS profile -o /tmp/profiles/post-647/run${i} \
    -t cuda,nvtx --force-overwrite=true --capture-range=none \
    --sample=none --cpuctxsw=none \
    ./target/release/examples/phase10_rmsnorm_bench \
    --model-path /workspace/qwen3.5-4b
done

# Stats:
for i in 1 2 3; do
  $NSYS stats --report nvtx_sum,cuda_kern_exec_sum --format csv \
    --output /tmp/profiles/post-647/run${i} \
    /tmp/profiles/post-647/run${i}.nsys-rep
done
```

GPU spend on this audit: ~$1.20 (47 min A100-SXM4-80GB at $1.49/hr).
Wall-clock: 47 min including pool acquire, build, sanity, 3× profile
capture, stats generation, and CSV pull.

---

## Addendum 2026-04-29 — FleCE T=16384 OOM probe (closure)

Date: 2026-04-29
Status: **§3 closes for FleCE — zero ROI even though T=16384 OOMs.**
Hardware: NVIDIA A40 48GB (sm_86, 46,068 MiB total VRAM, driver 570.195.03),
CUDA 12.4. A6000 capacity-blocked at task start (warm pool pod
`18x6how2blkqon` and `1qnfctm4m9qwoe` both returned `SUPPLY_CONSTRAINT` on
resume); A40 is the audit's named arch-equivalent fallback for the 48 GiB
ceiling probe (also sm_86, ~3 GiB tighter ceiling).
Branch: `audit/phase10-section3-flce-t16384-probe`. Built from `05463c9`
(PR #649 merge).

### What we ran

A single-cell SFT bench at T=16384 on Qwen3.5-4B + rank-8 LoRA + gradient
checkpointing (auto-configured to 8 segments at the detected
`vram_gb=48.305`), with `KILN_USE_FLCE=1` (Phase B chunked-vocab CustomOp1
forward — kiln's current best, the audit's "Phase B alone, no future
FleCE Phase C" configuration), `KILN_W4A16` unset (W4A16+training fails
with the rank-1/rank-2 q_proj_t mismatch — see agent note
`kiln-w4a16-sft-trainer-rank-bug`), RMSNorm CustomOp ON in the bench
(`rmsnorm_on=true`). Two arms:

1. **Gate OFF (default A40 path)**: `KILN_FORCE_RMSNORM_KERNEL` unset;
   the PR #644 VRAM gate (≥47 GiB total) leaves A40 below the threshold
   (46,068 < 48,128 MiB) so `fused_path="OFF"` and the trainer falls
   through to `rms_norm_fallback`.
2. **Gate ON (A6000 simulation)**: `KILN_FORCE_RMSNORM_KERNEL=1` forces
   `fused_path="ON"` to mirror what A6000 would dispatch (A6000 49,140 MiB
   ≥ 48,128 MiB threshold, so its gate is naturally ON).

Bench harness: `crates/kiln-server/examples/phase10_rmsnorm_bench.rs`
patched to a single T=16384 RMSNorm-on cell via the new helper
`scripts/phase10_flce_phase_b_t16384_only.py`. Build: `KILN_CUDA_ARCHS=86
cargo build --release --features cuda,nvtx --example phase10_rmsnorm_bench
-p kiln-server` (the `nvtx` feature is set but unused by this probe;
left on for parity with PR #649). Cargo.toml fix shipped in this PR (see
"Cargo.toml propagation fix" below).

### Results

| arm                 | gate path | peak (MiB) | delta (MiB) | step (s) | status | poller window |
|---------------------|-----------|-----------:|------------:|---------:|:------:|--------------:|
| 1 — A40 default     | fused=OFF |     45,447 |      29,104 |     9.44 | OOM    | full step     |
| 2 — A40 force-fused | fused=ON  |     45,095 |      28,752 |     0.89 | OOM    | partial (\*)  |

(\*) The 50 ms nvidia-smi poller likely missed the actual peak in arm 2:
fused=ON failed at 0.89 s, before the poller could capture the largest
allocation. The reported 45,095 MiB is a **lower** bound; the true peak is
≥ that and ≤ A40 ceiling 46,068 MiB. Arm 1 (fused=OFF) ran for 9.44 s and
saw multiple poller samples, so its 45,447 MiB peak is more reliable.

Baseline VRAM (post-load) is 16,343 MiB in both arms. T_actual=16,380
(target 16,384, lost 4 to tokenizer round-down).

### Mechanism — why FleCE Phase C is not the fix

Both arms OOM with delta ≈ 29 GiB above post-load baseline. That delta is
**not** the lm_head logits — Phase B's chunked-vocab CustomOp1 already
prevents `[T_active, V]` materialization (per
`crates/kiln-flce-kernel/src/lib.rs` lib.rs and `phase_b.rs` module
docs; `[T=16380, V=248320, F32]` would itself be 15.5 GiB if
materialized, and it is not).

The remaining ~29 GiB is GDN/MLP saved activations across the 8
gradient-checkpoint segments. This matches the existing finding from
agent note `kiln-flce-phase-a-validation-2026-04-29` ("GDN-side
activations dominate at long T") and the Phase 10 §1 closure work
(`docs/audits/PHASE10_GDN_TRAINING_STREAMING.md`,
`PHASE10_GDN_TRAINING_STREAMING_IMPL.md`, PR #635/#637).

FleCE Phase C — Liger-style vocab-axis chunking on top of Phase B's
existing vocab-axis chunking — addresses the **head**, not GDN. Even if
it shaved another 1-2 GiB off the head intermediates (which Phase B
already keeps to 256 MiB / chunk × 3 retained tensors = ~768 MiB at
T=16384, V_chunk=4096), it would not move the needle from 45 GiB → ≤43
GiB needed to clear the A40 ceiling, much less the A6000 ceiling once
GDN dominates. The path forward for unblocking T=16384 SFT is GDN
training-time streaming (PR #635/#636/#637 + the §1 layer-pair tiled
work and any post-#637 follow-ups), not a head-side fusion.

### A6000 implications

Both A40 arms OOM at peak 45.0–45.4 GiB. A6000 has 49,140 MiB total —
+3.0 GiB more headroom than A40. The fused-path arm on A40 saved ~352
MiB vs the fallback arm (45,447 → 45,095 MiB), so fused saving alone is
< 1 GiB. With +3 GiB extra ceiling and ≤ 1 GiB more saving, A6000 fused
peak at T=16384 would land at ~45 GiB / 49 GiB ceiling — likely **closes**
on A6000, but with 4 GiB of slack at most.

This is consistent with the audit's pre-probe prediction (~47–50 GiB peak
on A6000 at T=16384 with 4 segments; the actual run used 8 segments
which roughly halves the saved-tensor footprint per segment — explaining
why the observed peak is well below the upper bound). It does **not**
change the FleCE verdict: even if A6000 closes T=16384 at 45 GiB peak,
FleCE saves head intermediates which are already <1 GiB at T=16384 with
Phase B's chunking. There is no A6000 scenario where FleCE Phase C
unblocks new context length on Qwen3.5-4B + rank-8 LoRA SFT.

### Verdict

* **§3 closes for FleCE.** Both audit branches resolve to "no kernel":
  * "If T=16384 OOMs even with Phase B alone …" → the OOM happens, but
    its mechanism (GDN/MLP saved activations) is not what FleCE Phase C
    addresses. FleCE Phase C provides bounded peak-VRAM-only ROI on the
    *head*, and head intermediates are already small under Phase B.
  * "If T=16384 closes cleanly …" → consistent with A6000 prediction
    (~45 GiB peak, fits in 49 GiB), and even there the extra V-axis
    chunking on already-chunked head intermediates is a no-op.
* The path forward for unblocking long-context SFT is GDN training-time
  streaming + per-tile forward+backward inside `checkpointed_forward_backward`
  (Phase 10 §1 follow-ups, PR #635 / #636 / #637 already in flight). FleCE
  is off the §3 candidate list.
* The kernel-fusion track for SFT remains exhausted post-PR-#647. The
  planning loop should pull from Phase 9 release prep (security audit,
  license review, reproducible builds, CI/CD pipeline, GHCR image, landing
  page, v0.1.0 cut) until new evidence reopens a candidate.

### Cargo.toml propagation fix (shipped in this PR)

Both PR #647 and PR #649 needed the same local-only patch on the pod
(see PR #649 §"Bench protocol", `kiln-flce-kernel/Cargo.toml` comment,
and agent note `kiln-server-cuda-doesnt-propagate-to-flce`):

```diff
 # crates/kiln-server/Cargo.toml
 [features]
-cuda = ["kiln-model/cuda"]
+cuda = ["kiln-model/cuda", "kiln-train/cuda"]
```

Without this, `cargo build -p kiln-server --features cuda` activates
`kiln-model/cuda` but leaves `kiln-train/cuda` (and therefore
`kiln-flce-kernel/cuda`) off, and any `KILN_USE_FLCE=1` path on GPU
fails at runtime with `"flce phase b cuda_fwd: kiln-flce-kernel built
without cuda feature"`. PR #647 and PR #649 each reapplied this patch on
the pod by hand. This PR lands the one-line fix so the next agent does
not pay the same discovery cost. Verified post-patch on the same A40 pod
by re-running both probe arms above.

### Reproduction

```bash
ce kiln-pod-acquire --gpu-type "NVIDIA RTX A6000"   # or "NVIDIA A40" (arch-equivalent)

# On the pod (kiln-setup --clone first if /workspace/kiln does not exist):
cd /workspace/kiln
git checkout audit/phase10-section3-flce-t16384-probe   # or main once merged
source /root/.kiln-build-env
KILN_CUDA_ARCHS=86 cargo build --release --features cuda,nvtx \
  --example phase10_rmsnorm_bench -p kiln-server

# Apply T=16384 single-cell patch:
python3 scripts/phase10_flce_phase_b_t16384_only.py

# Arm 1 — default (fused=OFF on A40, fused=ON on A6000):
./target/release/examples/phase10_rmsnorm_bench --model-path /workspace/qwen3.5-4b

# Arm 2 — force fused-path ON (simulates A6000 on A40):
KILN_FORCE_RMSNORM_KERNEL=1 ./target/release/examples/phase10_rmsnorm_bench --model-path /workspace/qwen3.5-4b
```

GPU spend on this audit: ~$0.20 A40 (≈25 min lease at $0.44/hr — most
spent waiting for capacity, not running probes; both probes themselves
ran in <30 s combined including model load).

Raw logs:
* `docs/flce_phase_b_t16384_oom_probe_a40_raw_2026-04-29.log` (arm 1, fused=OFF)
* `docs/flce_phase_b_t16384_oom_probe_a40_fused_raw_2026-04-29.log` (arm 2, fused=ON)
