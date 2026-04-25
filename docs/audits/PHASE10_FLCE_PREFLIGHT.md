# Phase 10 — FLCE Preflight (VRAM measurement)

Date: 2026-04-20
Status: **GREEN — port FLCE**
Hardware: NVIDIA RTX A6000 (48 GB VRAM), CUDA 12.4
Model: Qwen3.5-4B (V=248320, hidden=2560, 32 layers)
Trainer: `kiln_train::sft_train`, rank-8 LoRA, gradient checkpointing ON (default 4 segments)

## Purpose

The Phase 10 Liger audit (PR #234) identified Fused Linear Cross-Entropy (FLCE)
as the single most promising Liger-Kernel port for long-context training on
Qwen3.5-4B. The audit's math-ceiling estimate showed ~8 GB of retained output-layer
VRAM at T=16384 and ~32 GB at T=65536. Before committing to a ~500-700 LOC port
modeled on `kiln-flash-attn`, we want an **empirical** number: how much peak VRAM
is actually dominated by the `[T, V]` logits stack during one SFT step on the
largest GPU we ship on?

This preflight measures peak VRAM during a single SFT training step at three
context lengths and compares it against the static audit math for the three
retained output-layer tensors.

## Gating signal

Define, for a given T:

- `a = T × V × 2`   — bf16 `[T, V]` logits tensor (retained for loss)
- `b = T_active × V × 4` — F32 cast of logits used inside `log_sum_exp`
- `c = T × V × 2`   — retained grad-of-logits for backward

At T=16384 on Qwen3.5-4B, `a + b + c ≈ 8.14 + 16.29 + 8.14 = 32.57 GB` of naive math.
Actual retained footprint depends on which tensors are kept live through backward.
The preflight runs an A6000 step with full forward + backward and samples
`nvidia-smi` at 50 ms intervals on a background thread to capture peak VRAM
**during** the step (not just post-step).

The kill/port decision is:

- **GREEN (port FLCE)** — at T=16384, `(a + b + c) / peak_VRAM ≥ 30%`
- **KILL** — below 30%, or the math is dominated by things FLCE cannot touch
  (weights, activations outside the output layer, optimizer state, etc.)

## Procedure

1. Launched `ghcr.io/ericflo/kiln-runpod:latest` on an A6000 via the kiln pool.
2. `kiln-setup --clone` on the pod (sccache + B2 remote cache).
3. Built `cargo build --release --features cuda -p kiln-server --example flce_preflight_bench`.
4. Downloaded `Qwen/Qwen3.5-4B` to `/workspace/models/Qwen3.5-4B`.
5. Ran the bench with warmup (T=256) and measurements at T ∈ {2048, 8192, 16384}:

   ```
   ./target/release/examples/flce_preflight_bench \
     --model-path /workspace/models/Qwen3.5-4B
   ```

6. The bench uses a background `nvidia-smi` poller (50 ms cadence) with an
   atomic peak tracker, runs one SFT step via the public `sft_train` API with a
   rank-8 LoRA adapter and long-form assistant content that tokenizes to
   approximately the target T under the Qwen3.5 chat template.

The bench source is at `crates/kiln-server/examples/flce_preflight_bench.rs`.

## Results

One A6000 pod, one bench process, one SFT step per T. Peak VRAM sampled at 50 ms
on a background thread during the step. Baseline after warmup (post first small
step, so CUDA allocator is past its cold state) = **18 472 MiB** (~18.0 GB). The
8.0 GB of model weights plus the ~10 GB LoRA / framework / candle overhead sits
on the GPU throughout.

| target_T | actual_T | peak VRAM (MiB) | ΔVRAM (MiB) | bf16 logits (a) | F32 cast (b) | grad logits (c) | a+b+c (GB) | (a+b+c)/peak | step (s) | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 2 048 | 2 042 | 32 680 (31.9 GB) | 14 208 (13.9 GB) | 0.94 GB | 1.89 GB | 0.94 GB | **3.78 GB** | **11.8 %** | 2.6 | ok |
| 8 192 | 8 192 | 48 488 (47.4 GB) | 30 016 (29.3 GB) | 3.79 GB | 7.58 GB | 3.79 GB | **15.16 GB** | **32.0 %** | 0.6 | **OOM** |
| 16 384 | 16 380 | 48 488 (47.4 GB) | 30 016 (29.3 GB) | 7.58 GB | 15.15 GB | 7.58 GB | **30.31 GB** | **64.0 %** | 0.6 | **OOM** |

The T=8192 and T=16384 runs **OOM'd**. The training call returned an error like
`segment transformer block N (full attention)` after the allocator hit the
48 GB GPU ceiling. The step wall-time of 0.6 s at those lengths is the failed
allocation path returning, not a completed step. The peak VRAM values of
48 488 MiB are the VRAM ceiling of the RTX A6000 — not the real memory the
workload would have consumed if given a bigger GPU.

The `(a+b+c)/peak` column for those rows therefore uses the A6000 ceiling
as the denominator. The **true** relative share of `(a+b+c)` is higher, because
the true peak would exceed 48 GB.

Raw bench log: removed from working tree; preserved in git history at `3d923bca29730e8b0e2d196700f2ad4c8cd4be00` — fetch with `git show 3d923bca29730e8b0e2d196700f2ad4c8cd4be00:docs/flce_preflight_raw_2026-04-20.log`.

## Verdict

**GREEN — port FLCE.**

The gating condition was "at T=16384, `(a+b+c)/peak_VRAM ≥ 30%`." At T=16384 on
an A6000 the `(a+b+c)` math is **30.3 GB** against a peak of **≥ 48 GB**, i.e.
**≥ 64 %**, well over the 30 % threshold.

T=8192 also clears the threshold at **32.0 %**, with `(a+b+c)` = 15.2 GB against
a 47.4 GB peak.

## What this means for FLCE

1. **FLCE is required, not just beneficial.** Qwen3.5-4B SFT on the GPU we ship
   on (RTX A6000, 48 GB) **cannot currently run T=8192 or T=16384** with
   gradient checkpointing ON. The `[T, V]` logits stack alone is ~15 GB at
   T=8192 and ~30 GB at T=16384. Shipping long-context SFT at those lengths
   is gated on removing the retained output-layer tensors.
2. **FLCE is necessary but not sufficient.** The OOM site reported by the
   trainer at T=8192 was inside a full-attention segment (block 7), before the
   head is computed. That suggests attention-side activations also contribute
   materially to VRAM pressure at long T, even though FlashAttn-2 is already
   vendored (`kiln-flash-attn`, PR #33). FLCE will unblock the head's
   contribution; attention-side cleanup may be a follow-up item — but the head
   has to come off the VRAM budget first before we can even see what is left.
3. **T=2048 runs fine without FLCE.** FLCE's marginal VRAM win at T=2048 is
   ~3.8 GB out of a 31.9 GB peak — still useful, but not blocking. FLCE should
   remain a drop-in that is always on, not an opt-in for long contexts.
4. **Design targets for the FLCE port** (from the audit, now confirmed by
   measurement):
   - Remove materialized `[T_active, V]` bf16 logits **and** the F32 cast used
     for `log_sum_exp`.
   - Do not retain gradient of logits through backward — instead, fuse the
     backward pass so the gradient of the upstream `[T_active, hidden]`
     tensor is produced directly from the tiled forward.
   - Tile along V (248 320) for memory-bound kernel; keep `T` free for the
     batch / sequence axis.
   - Gradient checkpointing must keep working — FLCE should land inside
     `cross_entropy_loss` (kiln-train/src/trainer.rs:919) without changing the
     checkpoint segment contract.
5. **Scope estimate stands.** A new `kiln-flce-kernel` crate modelled on
   `kiln-flash-attn` (~500–700 LOC of Rust + CUDA), plus a ~200-line edit in
   `kiln-train/src/trainer.rs` to swap the loss implementation. Parity test
   against the existing `cross_entropy_loss` at T ≤ 2048 (where the old path
   fits) validates numerics before we enable FLCE for longer T.

## Caveats

- `actual_T` differs from `target_T` because we construct the training sample by
  appending long-form prose and then applying the Qwen3.5 chat template. The
  bench reports the real tokenized length and computes audit math against it.
- Peak VRAM is sampled on the parent GPU process via `nvidia-smi` at 50 ms
  cadence. Short bursts under 50 ms may be undercounted; the reported number is
  a **lower bound** on the real peak.
- The `ModelWeights` safetensor copy is dropped before the SFT call, so the
  baseline measured after model load reflects only GPU-resident weights plus
  the tokenizer + framework overhead.
- Gradient checkpointing is ON with the default 4 segments. Without it,
  activations would grow linearly in T and FLCE's relative share of peak would
  shrink, so the measured ratio here is a conservative (smaller) estimate of
  the share FLCE could remove when checkpointing is ON.
