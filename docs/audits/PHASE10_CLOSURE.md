# Phase 10 (Liger Kernel Integration) — closure consolidation

Date: 2026-04-29
Status: **CLOSED — frontier exhausted as of PR #650.** All three Phase 10
chapters (§1 RMSNorm fusion, §1.5 FLCE Phase B chunked-vocab CustomOp1,
§2 Mode B trace, §3 next-kernel candidate audit + FleCE T=16384 OOM probe)
have landed. The remaining Phase 10 priority kernels (RoPE, SwiGLU/GeGLU,
Layer Norm, FleCE Phase C vocab-axis chunking) all clear `no_kernel_pivot`
under the math-ceiling rule on A100-SXM4-80GB SFT-step T=8192 profiles
(PR #649) and on A40 T=16384 OOM probes (PR #650). The actual long-context
SFT bottleneck is GDN/MLP saved activations — not the head, not the
elementwise post-gate activation, not the attention rotary. No further
Liger kernel ports are planned without a fresh re-profile that surfaces
a candidate ≥1.05× ceiling.

This doc is the single state-of-play pointer for Phase 10. Each chapter
below cites the canonical PR(s) and audit doc; the math-ceiling table
reproduces #649's verdicts; the pivot section names three ranked
post-Phase-10 directions for the planning loop to choose from.

## TL;DR

* **§1 RMSNorm fusion** ✓ landed (#638 kernel, #644 VRAM gate, #645
  A6000 closure status). Bit-exact loss between fused and candle-op
  fallback; A40 regression mitigated; T=2048 SFT closes on A6000.
  T=4096/T=8192 still OOM on A6000 — but the bottleneck is **not**
  RMSNorm, it is GDN/MLP saved activations + FLCE Phase A's autograd
  retention (see §2).
* **§1.5 FLCE Phase B** ✓ landed (#647 implementation + A40 closure,
  #648 A6000 closure deferred on capacity). Manual-backward CustomOp1
  for `fused_linear_cross_entropy` recomputes per-vocab-chunk
  intermediates instead of holding them live for autograd; closes
  T=8192 SFT on A40 at peak 33.2 GiB (vs OOM pre-#647). A6000
  prediction: peak 33–40 GiB / 49 GiB ceiling, finite loss.
* **§2 Mode B trace** ✓ landed (#646). Per-allocation `LD_PRELOAD`
  trace pinpointed the failing single allocation as FLCE Phase A's
  `[num_active=8180, chunk_len=4096]` f32 triplet of `logits_chunk`
  + `shifted` + `shifted.exp()` (3× 127.81 MiB per vocab chunk × 61
  vocab chunks = ~23 GiB held live for autograd). This was the
  evidence base for Phase B (#647).
* **§3 next-kernel candidate audit** ✓ landed (#649 preflight verdict
  `no_kernel_pivot`, #650 FleCE T=16384 OOM probe closure). Median
  NVTX on A100-SXM4-80GB at T=8192 SFT: GDN aggregate ~50.6%, MLP
  trio 28.6%, full-attn projections 7.2%. **Top kernel hotspot is
  FP32 SGEMM at ~42%** — kiln-train rank-8 LoRA-down/-up matmuls,
  NOT a Phase 10 candidate. The remaining priority kernels (RoPE,
  SwiGLU, Layer Norm, FleCE Phase C) all sub-floor under the
  math-ceiling rule.

## Chapter ledger

### §1 — Fused RMSNorm (Liger-style, custom CUDA backward)

| PR | Title | Verdict |
|---:|-------|---------|
| #638 | `phase10: Liger-style RMSNorm with custom CUDA backward (kernel-fusion pass §1)` | Kernel landed; bit-exact with candle-op fallback; -9.5 GiB peak savings on A6000 at T=8192 measurable. |
| #644 | `phase10: gate fused RMSNorm CustomOp2 on VRAM ≥47 GiB (audit Path 1)` | A40 regression mitigated by gating fused path on detected VRAM ≥48,128 MiB; A6000 routes to fused, A40 routes to candle-op fallback. |
| #645 | `audits: phase10 §1 closure status (post-#644 A6000 e2e SFT)` | **PARTIAL.** T=2048 SFT fits on A6000 (peak 42 GiB seg=8); T=4096/T=8192 still OOM at the ceiling. RMSNorm work itself is correct and complete — the remaining gap is a different bottleneck class. |

Canonical doc: [`PHASE10_S1_CLOSURE_STATUS.md`](PHASE10_S1_CLOSURE_STATUS.md).

### §1.5 — FLCE Phase B (chunked-vocab CustomOp1)

| PR | Title | Verdict |
|---:|-------|---------|
| #647 | `phase10: FLCE Phase B — manual-backward CustomOp closes T=8192 SFT on A6000` | Phase B CustomOp1 lands; T=8192 closes on A40 at peak 33.2 GiB / step 109.57s / final loss 0.8339 (vs Phase A OOM). |
| #648 | `audits: phase10 §1.5 — FLCE Phase B A6000 closure (deferred, capacity)` | A6000 closure cell deferred on `SUPPLY_CONSTRAINT`. A40 numbers stand as load-bearing correctness evidence; A6000 fused-RMSNorm + Phase B composition predicted to close at peak 33–40 GiB. |

Canonical doc: [`PHASE10_FLCE_PHASE_B_A6000_CLOSURE.md`](PHASE10_FLCE_PHASE_B_A6000_CLOSURE.md).

### §2 — Mode B per-allocation trace

| PR | Title | Verdict |
|---:|-------|---------|
| #646 | `audits: phase10 §2 prep — Mode B per-allocation trace on A6000` | **DIAGNOSED.** Failing single allocation is FLCE Phase A's `[num_active=8180, chunk_len=4096]` f32 = 127.81 MiB; OOM fires on the 93rd such request (61 vocab chunks × 3 retained tensors = ~23 GiB live). Right next slice is FLCE Phase B (which #647 then shipped). |

Canonical doc: [`PHASE10_MODE_B_TRACE.md`](PHASE10_MODE_B_TRACE.md).

### §3 — Next-kernel candidate audit + FleCE T=16384 OOM probe

| PR | Title | Verdict |
|---:|-------|---------|
| #649 | `audits: phase10 §3 preflight — post-#647 SFT-step re-profile + next-kernel candidate audit` | **`no_kernel_pivot`.** Three nsys profiles on A100-SXM4-80GB; medians bit-stable. None of the remaining Phase 10 priority kernels clears the math-ceiling floor (table below). Conditional reopen for FleCE gated on a T=16384 A6000 OOM measurement. |
| #650 | `audits: phase10 §3 — FleCE T=16384 A6000 OOM probe (closure)` | **§3 closes for FleCE — zero ROI even though T=16384 OOMs.** Both arms (default A40 path / fused-path simulation) OOM at peak ~45 GiB / 46 GiB ceiling; the ~29 GiB delta above post-load baseline is GDN/MLP saved activations across 8 grad-checkpoint segments, **not** lm_head logits. FleCE Phase C (vocab-axis chunking on the head) does not address the GDN-dominated bottleneck. |

Canonical doc: [`PHASE10_S3_CANDIDATE_PREFLIGHT.md`](PHASE10_S3_CANDIDATE_PREFLIGHT.md) (preflight in
the body, FleCE T=16384 closure in the addendum).

## Math-ceiling verdicts (reproduced from PR #649)

The math-ceiling rule (kiln skill §"Mandatory preflight"):
**region% × (1 − 1/expected_speedup) ≥ 5%** (1.05× floor under CUDA
graphs) **or ≥ 10%** (1.10× under graph-amortized fusion). Under CUDA
graphs, dispatch-amortization kills wins ≤1.03× (see agent note
`cuda-graph-dispatch-amortization-kills-small-fusions`). For SFT-step
backward we use the 1.05× floor since CUDA graphs are NOT being captured
around the backward path today.

| Candidate              | Region(s) attribution                                              | Step % | Generous ceiling | Overall speedup | Verdict |
|------------------------|--------------------------------------------------------------------|-------:|-----------------:|----------------:|--------:|
| Fused RoPE             | inside `:kiln/proj/qkv`                                            |   5.2% | 2.0× RoPE-only   |        1.027×   | ❌ DEFER |
| Fused SwiGLU/GeGLU     | `bmul_bf16` + `affine_f32` + slice of `bmul_f32` (post-gate elementwise only; matmuls NOT fused by Liger) | 5.0%   | 2.0× activation-only | 1.025× | ❌ DEFER |
| Fused Layer Norm       | N/A — Qwen3.5 uses RMSNorm only (PR #644 already shipped)          |   0.0% | N/A              |        N/A      | ❌ N/A   |
| FleCE (chunked vocab)  | Peak-VRAM only at T≥16384 on A6000; T=16384 OOM is GDN/MLP, not head (#650) | 0.0% step time at T=8192 | Head intermediates already <1 GiB under Phase B | 1.000× speed; zero peak-VRAM ROI per #650 | ❌ CLOSED (was ⚠️ CONDITIONAL pre-#650) |

Top kernel hotspot at SFT T=8192 is **FP32 SGEMM ~42%** (`ampere_sgemm_64x64_nn`
31.07% + `ampere_sgemm_128x64_nn` 11.02%) — kiln-train rank-8 LoRA-down/-up
matmuls in FP32 for numerical stability. **Not a Phase 10 candidate**:
replacing it requires either a LoRA precision change (BF16 storage with
FP32 accumulate — numerical-stability question, not a kernel fusion) or
fusing LoRA into the base GEMM (a major scheduler/autograd rewrite in
`kiln-train`). Both are recorded for future planning, not §3 work.

## Pivot recommendations (ranked)

The post-Phase-10 frontier has three high-leverage directions. They are
**recommendations, not commitments** — the planning loop chooses based on
operator priority. Each is sized so a single planning cycle can scope it
into one focused implementation task.

### 1. Phase 9 v0.1.0 release prep (highest leverage, lowest risk)

**Why:** Phases 1–7 ✅ shipped; Phase 8 mostly ✅ shipped (multi-GPU TP +
model-agnostic mode are explicitly post-v0.1). The only blockers between
current main and a public v0.1.0 cut are the Phase 9 release-prep items:
security audit (security-audit-v0.1 closed in v0.2.7 release; remaining
items are docs/CI), license review (cargo-deny shipped; THIRD_PARTY_LICENSES
shipped), reproducible builds (Docker `--locked` shipped, Sigstore build
provenance shipped in v0.2.8), CI/CD pipeline (GHCR auto-publish shipped),
release binaries (Linux/macOS/Windows shipped), GHCR Docker image
(shipped), landing page scaffolded (#624), v0.1.0 release semver cut.
Most of the heavy lifting is already done — what remains is an audit pass
to confirm nothing is missing, then cut v0.1.0.

**Where the work would land:** `docs/audits/v0.1.0-release-readiness.md`
(new) → audit + checklist; `CHANGELOG.md` v0.1.0 entry; tag + GitHub
Release. Doc-only / CI-only; **no GPU pod required**.

### 2. GDN training-time streaming follow-ups (active in-flight work)

**Why:** The actual mechanism behind T=16384 SFT OOM — even with FLCE
Phase B + fused RMSNorm — is GDN/MLP saved activations across 8
grad-checkpoint segments (#650). The right next slice for unblocking
long-context SFT is GDN training-time streaming + per-tile forward
+ backward inside `checkpointed_forward_backward`, NOT a head-side
fusion. PR #635 (training-time streaming GDN prefill in
`model_forward_segment`), PR #636 (time-axis tile inside
`checkpointed_forward_backward`), PR #637 (layer-pair time-axis tiled
forward+backward) are all in flight. Per the CHANGELOG entry on #635,
the work currently stands at AMBER on A6000: dispatch is reachable
(CPU parity passes; T=2048 STREAMING-ON loss bit-exact with
STREAMING-unset), but T=8192 SFT still OOMs because autograd recompute
saves all per-tile activations simultaneously. The next slice is the
audit's remediation §2 — per-tile forward+backward inside the
checkpoint shell — which has not yet landed.

**Where the work would land:** `crates/kiln-model/src/forward.rs`
(streaming branch), `crates/kiln-train/src/trainer.rs` (per-tile
forward+backward inside the checkpoint shell). New audit doc:
`docs/audits/PHASE10_GDN_TRAINING_PER_TILE_FWDBWD.md`. Requires GPU
validation on A6000 at T=8192/T=16384.

### 3. LoRA precision study (FP32 → BF16 with FP32 accumulate)

**Why:** PR #649 surfaced FP32 SGEMM at ~42% of SFT-step kernel time
on A100 — `ampere_sgemm_64x64_nn` (31.07%) + `ampere_sgemm_128x64_nn`
(11.02%), both shape-matched to rank-8 LoRA-down/-up matmuls in
`kiln-train`'s autograd path. kiln currently keeps LoRA params in FP32
for numerical stability of the small-rank accumulation. The question
worth scoping: does BF16 storage with FP32 accumulate (the standard
mixed-precision pattern for tensor-core GEMMs) preserve loss-curve
parity for rank-8 LoRA on Qwen3.5-4B SFT? If yes, the ~42% FP32 SGEMM
hotspot collapses into BF16 tensor-core paths (`ampere_bf16_s16816gemm_*`
already at 4–5% combined in the same profile), buying ~30–40% step-time
reduction on Ampere. If no — i.e. loss curves diverge — the result
itself is a recorded design decision and rules the lever out.

**Where the work would land:** A short A/B bench (BF16-LoRA vs
FP32-LoRA on identical training data; compare loss curves over ~100
SFT steps + final loss + final-adapter inference parity). New audit
doc: `docs/audits/PHASE10_LORA_PRECISION_STUDY.md`. Requires GPU on
A6000/A100. Note: this is **not a Liger kernel port** and not in the
Phase 10 priority list; it is post-Liger ground worth scoping because
it is the highest-leverage non-kernel lever on the §3 profile.

## Explicit non-goals

* **No further Liger kernel ports** without a fresh re-profile that
  surfaces a candidate ≥1.05× ceiling. The post-#647 SFT-step profile
  in #649 is the canonical reference; the A40 T=16384 probe in #650
  is the canonical long-context reference. Any future Liger candidate
  task must cite a newer profile.
* **No "FleCE Phase C" / vocab-axis chunking on top of Phase B's
  existing vocab-axis chunking.** PR #650 closed this — head
  intermediates are already <1 GiB at T=16384 under Phase B; the
  bottleneck is GDN/MLP, not the head.
* **No retry of fused RoPE or fused SwiGLU/GeGLU at SFT T=8192 on
  Ampere.** Math-ceiling math is invariant under wall-clock rescale
  between A100 and A6000.

## References

* PR #638 — Liger-style RMSNorm with custom CUDA backward
* PR #644 — fused RMSNorm CustomOp2 VRAM gate (A40 regression remediation)
* PR #645 — Phase 10 §1 closure status (post-#644 A6000 e2e SFT)
* PR #646 — Phase 10 §2 prep: Mode B per-allocation trace on A6000
* PR #647 — FLCE Phase B + A40 closure
* PR #648 — Phase 10 §1.5 A6000 closure deferred (capacity)
* PR #649 — Phase 10 §3 preflight: post-#647 SFT-step re-profile + next-kernel candidate audit
* PR #650 — Phase 10 §3: FleCE T=16384 A6000 OOM probe (closure)
* PR #635 / #636 / #637 — GDN training-time streaming follow-ups (in flight)
* `PROFILING.md` §"Phase 10 §3 post-#647 SFT-step re-profile" — canonical
  reference profile for any future Liger candidate task
* Agent notes: `kiln-flce-phase-b-closure-2026-04-29`,
  `kiln-flce-phase-a-validation-2026-04-29`,
  `kiln-flce-is-prerequisite-not-optimization`,
  `kiln-flce-t16384-probe-result-2026-04-29`,
  `kiln-phase10-liger-audit-findings`,
  `cuda-graph-dispatch-amortization-kills-small-fusions`
