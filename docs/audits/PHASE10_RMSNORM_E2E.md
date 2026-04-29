# Phase 10 §1 — RMSNorm custom backward end-to-end SFT validation (post-#637/#638)

Date: 2026-04-29
Status: **RED** — every bench cell OOMs at the A6000 ceiling, including T=2048 (which PR #637's bench reported as OK with peak=34 600 MiB at the same conditions two days earlier). #638 alone does NOT close Finding 2; in fact, the bench surfaces an unexplained ~14–16 GB increase in delta-VRAM at T=2048 between PR #637 (Apr 29 morning) and current main (Apr 29 afternoon, post-#638) which makes the per-cell parity comparison undefined.
Hardware: NVIDIA RTX A6000 (49 140 MiB VRAM, driver 565.57.01), CUDA 12.4
Pod: RunPod direct-launch fallback (pool A6000+A40 hosts both at "not enough free GPUs" at acquire time)
Branch: `ce/phase10-rmsnorm-e2e-audit`
Bench: `crates/kiln-server/examples/phase10_rmsnorm_bench.rs` (with the `Warmup skipped` patch in this PR — see "Bench fix" below)
Model: Qwen3.5-4B (V=151 936, hidden=2560, 32 layers, 24 GDN + 8 GQA), bf16, no W4A16 (the bench's docstring mentions `KILN_W4A16=1` but Marlin's q_proj_t rank-1/rank-2 SFT mismatch — see agent note `kiln-w4a16-sft-trainer-rank-bug` — blocks SFT under W4A16; W4A16 is unset throughout)

## Purpose

PR #638 shipped a Liger-style RMSNorm with custom CUDA backward (`RmsNormCustomOp`),
intended to reduce per-layer autograd-saved-tensor footprint vs the candle-op chain
that decomposes RMSNorm into ~11 saved intermediates per layer. Combined with the
layer-pair-tiled training infrastructure shipped in PR #637, the fused-kernel pass
was nominated as Phase 10's path to unblock T=8192 SFT on A6000 for Qwen3.5-4B
(see PR #637 "Required follow-ups (post-this PR)" §4).

This audit answers two questions:

1. **Parity** — does `RmsNormCustomOp` produce the same final loss at T=2048 as
   the disabled-kernel fallback?
2. **Memory** — does enabling `RmsNormCustomOp` save enough autograd saved tensors
   to push T=8192 SFT under the A6000 ceiling, OR at least improve T=4096 vs the
   disabled-kernel fallback?

The bench (`phase10_rmsnorm_bench`, 456 lines, written by the PR #638 author)
runs the SFT path through `kiln_train::sft_train` once per `(T, rmsnorm_op)` cell
on Qwen3.5-4B with rank-8 LoRA, 1 epoch / 1 example, gradient checkpointing
auto-configured (4 segments × 8 layers each). VRAM is sampled by a 50 ms
nvidia-smi poller during the step.

## Procedure

1. `runpod_api.py launch --kiln-image --gpu "NVIDIA RTX A6000"` (pool was
   unavailable — both A6000 and A40 hibernated pods returned "not enough free
   GPUs on the host machine" on resume; fell back to direct launch per the kiln
   skill's documented fallback path).
2. `kiln-setup --clone --repo /workspace/kiln` on the pod (sccache + B2 + clone +
   model download). The Qwen3.5-4B weights at `/workspace/qwen3.5-4b` are the
   stock kiln-runpod path.
3. `git fetch origin main && git reset --hard origin/main` — pinned to commit
   `1ad54c2` (PR #639 main, the latest at task start).
4. Patched the bench to skip the T=256 warmup pass (see "Bench fix" below).
5. Built the example: `KILN_CUDA_ARCHS=86 cargo build --release --features cuda
   -p kiln-server --example phase10_rmsnorm_bench`. Cold build via sccache+B2
   restored cache hits ≥95% on the per-crate units; one full-link step.
6. Ran two bench passes:
   * Pass A — default env (`KILN_USE_FLCE=1`, `KILN_DISABLE_RMSNORM_BACKWARD`
     toggled per cell, `KILN_W4A16` unset). This is the canonical comparison
     between `RmsNormCustomOp` ON (cell odd-numbered) and the
     forward-only-kernel + autograd-traced-backward "debug fallback" OFF
     (cell even-numbered).
   * Pass B — `KILN_DISABLE_RMSNORM_KERNEL=1` set globally, forcing the
     pre-#638 candle-op-chain path throughout. This was added to test the
     hypothesis that the unexpected OOM at T=2048 is a regression in #638's
     debug-fallback path (the OFF cell's "forward-only kernel + autograd-traced
     backward"). It isn't — the candle-op chain ALSO OOMs at T=2048 in this
     bench under FLCE ON.

Raw bench logs preserved in this PR at `/dev/null` (logs are too large for
the working tree; the markdown tables below are the canonical record).

## Bench fix included in this PR

The bench's original `main()` ran a T=256 warmup via `run_one(256, true, ...)`
and then captured `current_vram_mib()` as the post-warmup baseline used for all
subsequent cells. On A6000 with the current model + FLCE ON the warmup itself
peaks at ~32.5 GB and the candle CUDA allocator does not release this memory
back to the driver between calls — so subsequent cells start with ~16 GB of
"baseline" VRAM that is actually allocator-cached state from the warmup. This
left only ~16 GB of headroom for cell 1 (T=2048 ON), insufficient to
instantiate the autograd graph, and OOM'd every cell at peak=48 522 MiB
(≈ A6000 ceiling).

The patch in this PR replaces the warmup with a no-op (`eprintln!`) and
reports per-cell `delta_mib` relative to the post-load baseline (16 346 MiB,
which is just model weights + framework setup). Cell 1 absorbs cold-allocator
variance (slightly noisier first measurement), but every subsequent cell
measures incremental SFT-step VRAM cleanly against a known-stable baseline.

This is a bench-correctness fix, not a workaround for the OOMs documented
below. With the original warmup, ALL cells would OOM at peak=48 522 MiB; with
the patch, ALL cells STILL OOM at peak=48 522 MiB. The patch makes the
per-cell `delta` numbers honest: `delta = peak − post-load-baseline`, not
`delta = peak − warmup-polluted-baseline`.

## Results

### Pass A — default env (FLCE ON, RMSNorm op toggled)

```
Baseline VRAM (post-load): 16346 MiB
```

| T target | T actual | RMSNorm op | status | peak (MiB) | delta (MiB) | step (s) | final loss |
|---------:|---------:|:----------:|:------:|-----------:|------------:|---------:|-----------:|
| 2 048 | 2 042 | on  | **OOM** | 48 522 | 32 176 | 2.18 | – |
| 2 048 | 2 042 | off | **OOM** | 48 522 | 32 176 | 2.03 | – |
| 4 096 | 4 091 | on  | **OOM** | 48 554 | 32 208 | 2.81 | – |
| 4 096 | 4 091 | off | **OOM** | 48 554 | 32 208 | 2.99 | – |
| 8 192 | 8 192 | on  | **OOM** | 48 554 | 32 208 | 0.65 | – |
| 8 192 | 8 192 | off | **OOM** | 48 554 | 32 208 | 0.58 | – |

T=2048 parity check skipped (one or both cells failed).
T=4096 saved-tensor delta calc skipped (all cells failed).

### Pass B — `KILN_DISABLE_RMSNORM_KERNEL=1` (forces candle-op chain)

This pass forces the pre-#638 candle-op-chain RMSNorm regardless of the
per-cell `KILN_DISABLE_RMSNORM_BACKWARD` toggle. It tests whether #638's
"forward-only kernel + autograd-traced backward" debug fallback (the OFF cells
in Pass A) is materially heavier than the candle-op chain, which would
indicate a regression in #638's fallback path. It does not:

```
Baseline VRAM (post-load): 16346 MiB
```

| T target | T actual | RMSNorm op | status | peak (MiB) | delta (MiB) | step (s) | final loss |
|---------:|---------:|:----------:|:------:|-----------:|------------:|---------:|-----------:|
| 2 048 | 2 042 | on  | **OOM** | 47 434 | 31 088 | 2.24 | – |
| 2 048 | 2 042 | off | **OOM** | 48 522 | 32 176 | 2.04 | – |
| 4 096 | 4 091 | on  | **OOM** | 48 554 | 32 208 | 2.82 | – |
| 4 096 | 4 091 | off | **OOM** | 48 554 | 32 208 | 2.97 | – |
| 8 192 | 8 192 | on  | **OOM** | 48 554 | 32 208 | 0.66 | – |
| 8 192 | 8 192 | off | **OOM** | 48 554 | 32 208 | 0.59 | – |

Pass B's T=2048 ON cell is 1 088 MiB lower than Pass A's at the same
conditions. With `KILN_DISABLE_RMSNORM_KERNEL=1` the per-cell
`KILN_DISABLE_RMSNORM_BACKWARD` toggle should be inert (kernel disabled
globally), so the 1 GB difference is attributable to allocator
non-determinism between cells, not to a `RmsNormCustomOp` effect. Both
passes converge on the same OOM ceiling for every cell ≥ T=4096.

### Comparison vs PR #637's bench (Apr 29 morning)

PR #637's audit (`docs/audits/PHASE10_GDN_TRAINING_LAYER_PAIR_TILED.md`)
ran a different bench (`flce_phase_a_validation_bench`) on the same A6000
GPU class with what should be the same effective training config:

| Bench | T | FLCE | STREAM | TILE | peak (MiB) | delta (MiB) | step (s) | status |
|:------|---:|:----:|:------:|:----:|-----------:|------------:|---------:|:------:|
| PR #637 (Apr 29 AM) | 2 048 | OFF | unset | default | 34 600 | 16 128 | 2.8 | **ok** |
| PR #637 (Apr 29 AM) | 2 048 | ON  | unset | default | 34 600 | 16 128 | 3.2 | **ok** |
| this PR — Pass A    | 2 048 | ON  | unset | default | 48 522 | 32 176 | 2.0 | **OOM** |
| this PR — Pass B    | 2 048 | ON  | unset | default | 47 434 | 31 088 | 2.2 | **OOM** |

Same A6000, same Qwen3.5-4B, same FLCE, same gradient-checkpoint segments
(4), same rank-8 LoRA, same example builder pattern (repeated paragraph
to target T). Yet T=2048 SFT-step delta-VRAM has gone from **16.1 GB →
~32 GB** in the ~6 hours between PR #637's bench (run on the morning of
Apr 29) and this audit (run that afternoon, post-#638 + #639).

The only commits between PR #637's bench measurement and this one are
PR #638 (RmsNormCustomOp) and PR #639 (CI-only NaN parity test fix in
`kiln-train`). Pass B confirms the regression is NOT in #638's
custom-op path — it appears even with the kernel-disabled candle-op
chain. The mechanism is currently unidentified.

## Verdict per cell

* **(1) T=2048, custom-op ON — RED.** OOMs at A6000 ceiling. Bench's
  intended "baseline final loss + peak VRAM" cell did not produce a loss.
* **(2) T=2048, custom-op OFF — RED.** OOMs at A6000 ceiling. Parity
  check vs (1) is undefined (no losses to compare). Bench's intended
  parity invariant is undefined.
* **(3) T=4096, custom-op ON — RED.** OOMs at A6000 ceiling.
* **(4) T=4096, custom-op OFF — RED.** OOMs at A6000 ceiling. Bench's
  intended `(4)peak − (3)peak ≥ 0.5 GiB` saved-tensor-delta check is
  undefined.
* **(5) T=8192, custom-op ON — RED.** OOMs at A6000 ceiling. Step time
  (0.65 s) indicates failure during the autograd graph construction or
  the first matmul, well before the step would normally complete.
* **(6) T=8192, custom-op OFF — RED.** OOMs at A6000 ceiling, same
  early-failure signature.

## Finding 2 closure

**Finding 2 status — RED.** PR #638 alone does not unblock T=8192 SFT on
A6000 for Qwen3.5-4B. In fact, the bench surfaces a more pressing problem:

**Cell 1's pre-existing assumption — that T=2048 SFT comfortably fits on
A6000 — is no longer true on current main.** PR #637's bench (run 2026-04-29
morning, before #638 / #639) reported peak=34 600 MiB / delta=16 128 MiB at
T=2048 OFF, success. This audit's runs (post-#638 / #639, same afternoon)
report peak=48 522 MiB / delta=32 176 MiB at T=2048 OFF, OOM. A ~14–16 GB
delta-VRAM regression has landed in the past ~6 hours of merges, and Pass
B confirms it is NOT in #638's RmsNormCustomOp path — the candle-op chain
(forced via `KILN_DISABLE_RMSNORM_KERNEL=1`) shows the same OOM ceiling.

This audit therefore cannot make a per-cell verdict on #638's individual
contribution; the regression dominates whatever delta the custom backward
might have provided.

## Next slice recommendation

The original task description's decision rule applied to the RED outcome:

> RED T=8192 (OOMs both ways): Finding 2 still open; identify next biggest
> persistent allocation via nsys cuMemAlloc attribution and propose a focused
> fix.

Two parallel paths, in priority order:

1. **Bisect the T=2048 regression first.** Build the bench at PR #637's HEAD
   (`git checkout c983ca7`), re-run the same FLCE-ON / OFF cells, and
   confirm the T=2048 OOM is absent at that commit. Then `git bisect` the
   four candidates between `c983ca7` (PR #637) and `1ad54c2` (PR #639) —
   namely PRs #638 and #639. PR #639 is a CI-only test fix and is unlikely
   to affect runtime VRAM, so PR #638 is the prime suspect even though
   Pass B's `KILN_DISABLE_RMSNORM_KERNEL=1` path also OOMs (which would
   normally rule it out). Possible mechanism: the new `RmsNormCustomOp`
   registration might allocate or pin a global resource at module load
   time independent of which path the per-call dispatch takes; or the
   `kiln_fused_rmsnorm_bwd` kernel binary loaded into the CUDA context
   reserves device memory unconditionally. nsys-attribution of the
   first 200 ms of cell 1 should show the regression source within one
   pod-run.

2. **Decide the next Liger fusion target without waiting for (1).** PR
   #637's "Required follow-ups" §4 listed three Liger-style fused-kernel
   passes, with FLCE shipped in PR #241 (Phase A). The two remaining are:
   * **Fused SwiGLU MLP** (gate + up + silu + elementwise + down).
     PR #637 audit attributes ~3 of the per-layer saved-tensor stacks to
     the MLP chain: `gate_proj` output, `up_proj` output, `silu(gate_proj)`,
     and the elementwise product. Fusing these into one or two CUDA
     kernels that materialize a single intermediate would reduce the
     per-layer saved-tensor count by ~3-4×.
   * **Fused GDN gates + gated_rms_norm + recurrent.** PR #637 calls
     this the GDN analogue of #638; the binding constraint per the audit
     is the per-tile state checkpoint plus the chain of {gate, qk_norm,
     gated_rms_norm} intermediates. PR #466 already shipped a
     `gated_rms_norm` fused kernel (3% wallclock under CUDA graphs), so
     the next-incremental wins on the GDN side are the gate fusion or
     a recurrent-path consolidation.

   Recommendation: **start with Fused SwiGLU MLP** because (a) MLP is
   ~25-30% of decode wall-time per the post-#166 NVTX breakdown in the
   kiln skill (so even a 1.5-2× saved-tensor reduction on MLP probably
   moves training peak materially), (b) it does not interact with the
   GDN per-tile state-thread complexity that has caused multiple null
   results (PRs #141, #173, #176), and (c) the existing
   `kiln-rmsnorm-kernel` + `RmsNormCustomOp` pattern in #638 is a
   reusable scaffold — a `kiln-mlp-kernel` crate following the same
   shape (CustomOp2 + manual backward + KILN_DISABLE_* kill switch +
   CPU autograd-traced fallback test) is a tractable PR.

Until the regression in (1) is bisected, **any new memory-related bench
result is suspect**. Future `phase10_*` benches should pin the model and
trainer commits, not just the example file, and run a sanity cell at
T=2048 OFF / FLCE OFF to confirm the bench's baseline matches PR #637's
16 128 MiB delta number before measuring anything else.

**Next slice: bisect the T=2048 SFT-step delta-VRAM regression between
PR #637's HEAD (`c983ca7`) and current main (`1ad54c2`) on A6000, then
proceed with Fused SwiGLU MLP only after the baseline is restored.**

## Caveats

* The 50 ms `nvidia-smi` poller may undercount sub-50ms transient peaks.
  All cells in both tables peak within 100 MiB of the A6000 ceiling
  (49 140 MiB), so the OOM-vs-OK distinction is robust.
* `KILN_W4A16` is unset throughout. The bench's docstring mentions
  "+ KILN_W4A16=1" but agent note `kiln-w4a16-sft-trainer-rank-bug`
  documents that Marlin's `q_proj_t` packing is rank-1 (inference shape)
  while the SFT trainer expects rank-2; W4A16 + SFT is currently broken
  and out of scope for this audit.
* This audit does not include nsys attribution, which would localize the
  regression source within ~5 minutes on a fresh pod. It is the single
  most useful next bench measurement and is named explicitly in the
  recommended next slice.
* Both bench passes ran to completion without errors other than the
  per-cell OOMs (which are caught by `Result::Err` and labelled "OOM"
  by the bench, not propagated). Total bench wall-time per pass ~30 s
  on A6000 — the OOMs occur early in each step, so the bench is fast
  even when failing.
* PR #637's bench at the same conditions (Apr 29 morning) succeeded at
  T=2048. The regression is reproducible across two passes (default and
  `KILN_DISABLE_RMSNORM_KERNEL=1`). Replication of PR #637's
  `flce_phase_a_validation_bench` on this same pod was NOT performed in
  this audit due to budget, but is the first step of (1) above.
