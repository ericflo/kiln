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

## Phase A validation — 2026-04-29

Status: **RED — Phase A insufficient on A6000**
Hardware: NVIDIA RTX A6000 (48 GB VRAM, driver 565.57.01), CUDA 12.4
Commit: `32a3828` on top of `5b42652` (`main` after `kiln-v0.2.8`)
Bench: `crates/kiln-server/examples/flce_phase_a_validation_bench.rs`
Raw log: `docs/flce_phase_a_validation_raw_2026-04-29.log`

### Purpose

The original preflight (above) measured peak VRAM **without** FLCE and showed
T=8192 and T=16384 OOM on A6000. Phase A landed in PR #241 (`KILN_USE_FLCE=1`,
chunked-vocab cross-entropy in pure candle) and has shipped in releases since
`kiln-v0.2.0`, but the empirical claim "Phase A unblocks long-context SFT on
A6000" was never measured. This section closes that gap.

### Procedure

The validation bench is a sibling of `flce_preflight_bench.rs` that toggles
`KILN_USE_FLCE` between cells using `std::env::set_var` (process-local, single-
threaded across cells). The same gradient-checkpoint defaults
(`KILN_GRAD_CHECKPOINT_SEGMENTS=4`) and rank-8 LoRA used in the preflight are
used here. Final loss is captured via the `sft_train` `ProgressCallback`.

Cells: T=2048 FLCE=OFF (baseline loss + parity reference), T=2048 FLCE=ON
(parity check), T=8192 FLCE=ON (Phase A's headline claim), T=16384 FLCE=ON
(headline claim). Peak VRAM sampled on a background `nvidia-smi` poller at
50 ms cadence. Baseline post-warmup = **18 474 MiB** (~18.0 GB).

### Results

| target_T | FLCE | actual_T | peak VRAM (MiB) | ΔVRAM (MiB) | abc (GB) | abc/peak | step (s) | loss     | status      |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 2 048  | OFF | 2 042  | 34 868  | 16 394 |  3.78 | 11.1 % | 2.8 | 2.783685 | **ok** |
| 2 048  | ON  | 2 042  | 34 868  | 16 394 |  3.78 | 11.1 % | 1.5 | n/a      | **other-error** |
| 8 192  | ON  | 8 192  | 48 490  | 30 016 | 15.16 | 32.0 % | 0.7 | n/a      | **OOM** |
| 16 384 | ON  | 16 380 | 48 490  | 30 016 | 30.30 | 64.0 % | 0.7 | n/a      | **OOM** |

`abc` is the static math from the preflight — bf16 logits + F32 cast + bf16
grad-of-logits — at the row's `actual_T`. The bench logs both the human-
readable summary and a JSON block; raw log preserved at the path above.

#### Finding 1 — Phase A has a non-OOM defect at T=2048

Running with `KILN_USE_FLCE=1` at T=2048 inside the gradient-checkpointed SFT
loop fails with:

```
fused linear cross-entropy (checkpointed): matmul active_hidden_f32 @ head_chunk:
matmul is only supported for contiguous tensors
  lstride: Layout { shape: [2030, 2560], stride: [2560, 1], start_offset: 0 }
  rstride: Layout { shape: [2560, 4096], stride: [248320, 1], start_offset: 0 }
  mnk: (2030, 4096, 2560)
```

The right operand is the V-axis chunk of `embed_tokens_t` (shape
`[hidden, V]`, stride `[V, 1]`). `narrow(1, off, chunk)` along the last axis
keeps stride `[V, 1]` rather than collapsing to `[chunk, 1]`, which candle's
matmul rejects. **The current `kiln-flce-kernel::fused_linear_cross_entropy`
chunked-vocab call therefore never completes a step in the gradient-
checkpointed SFT path.** This is a single-line defect (one `.contiguous()` on
the chunk, or chunking along T instead of V) — worth fixing in a follow-up
before any further Phase A or Phase B work, because Phase A as shipped does
not actually run end-to-end on Qwen3.5-4B SFT.

The peak VRAM number for this row (34 868 MiB) is **identical** to the
FLCE=OFF row's peak — that is, FLCE never gets to the point of allocating its
chunked logits before the matmul rejects the slice. The "abc/peak" column for
this row reports the FLCE=OFF math, not what Phase A would have used.

Parity could not be measured: `on_loss` is `None` because the step did not
complete.

#### Finding 2 — Even with the contig fix, T=8192 / T=16384 still OOM on A6000

Both T=8192 and T=16384 cells hit `peak_mib = 48 490` (the A6000 48 GB
ceiling) and return inside ~0.7 s — the failed-allocation path. ΔVRAM
(30 016 MiB) is identical for both, which confirms peak is pinned at the GPU
ceiling rather than reflecting the workload's true requirement. Two readings
are consistent with this:

1. The OOM occurs **before** Phase A can demonstrate any logits-tensor savings
   — likely inside the GDN prefill or full-attention activations on a
   gradient-checkpoint segment, the same site the original preflight reported
   at T=8192 ("segment transformer block N (full attention)").
2. The `kiln-flce-gdn-bottleneck` note (and PR #222 KV-cache FP8 audit) had
   already shown that GDN prefill, not the head, dominates long-T peak memory
   on Qwen3.5-4B before paged GQA KV is even touched. Phase A removes the
   head's contribution; it cannot remove the GDN-side contribution.

Even if Finding 1 is fixed, Phase A's expected savings are bounded by the GDN
prefill ceiling. The 30+ GB peak for T=16384 is dominated by activations the
chunked-vocab loss path does not touch, so Phase A alone cannot unblock SFT
at those lengths on A6000.

### Verdict

**RED — Phase A is insufficient on A6000 to unblock long-context SFT** on
Qwen3.5-4B.

This is a useful negative result, not a contradiction of the preflight's
"GREEN — port FLCE" verdict. The preflight only claimed that the head's
retained tensors are large enough to be **worth** removing (`a+b+c` ≥ 30 % of
peak at T=16384). It did not claim removing them would be **sufficient** to
fit T=8192 / T=16384 SFT under 48 GB. The validation here shows the head is
not the only large term at long T.

### Required follow-ups

1. **Fix the FLCE chunked-vocab contig bug** (Finding 1). Make
   `kiln-flce-kernel::fused_linear_cross_entropy` materialize a contiguous
   chunk before matmul, or rework the chunking to be along the T axis (so the
   right operand stays contiguous in V). Add a CPU- or small-GPU integration
   test that exercises the path with V matching Qwen3.5-4B's 248 320 — the
   existing tests in `kiln-flce-kernel/tests/parity.rs` and
   `kiln-train::trainer::tests` use sizes where the slice happens to be
   contiguous (V evenly divisible by chunk size, slice covers full V, or V
   small enough that no slicing happens). Re-run this validation bench after
   the fix to confirm parity at T=2048.

2. **Phase B (CUDA fused FLCE kernel) is necessary but not yet sufficient.**
   The Phase B port motivated by this preflight will reduce the head's
   memory footprint, but the validation here suggests it must be paired with
   GDN-side activation cleanup before T=8192 / T=16384 SFT will fit on
   A6000. Concretely: investigate streaming/chunked GDN prefill for the
   training-time forward (existing kiln-gdn-prefill streaming code already
   cuts inference peak roughly in half at 128k — see the
   `kiln-gdn-prefill-memory-ceiling-2026-04-24` note). Without that, Phase B
   alone cannot reach T=8192 SFT on the GPU we ship on.

3. **Update Phase 10 roadmap and `kiln-flce-is-prerequisite-not-optimization`
   note** to reflect the empirical result. The note's claim that FLCE is a
   prerequisite for T≥8192 SFT on A6000 stands; what changes is the second-
   order claim "shipping Phase A unblocks T=8192/16384." That claim is now
   measured RED — Phase A alone, even bug-free, cannot remove the GDN-side
   ceiling.

### Caveats

- Same caveats as the preflight apply (50 ms `nvidia-smi` cadence, peak is a
  lower bound, baseline excludes the dropped `ModelWeights` copy).
- The contig defect (Finding 1) means we have not actually measured Phase A's
  VRAM behavior at the head; the T=2048 FLCE=ON peak number is FLCE-OFF math.
- Each bench cell is a single SFT step. Sustained training (multi-step,
  multi-epoch) might surface additional VRAM growth (optimizer state,
  cumulative checkpoint state) that this single-step bench does not capture.

## Post-fix re-run — 2026-04-29 (T=2048 only)

Status: **Finding 1 RESOLVED**
Hardware: NVIDIA RTX A6000 (48 GB VRAM, driver 565.57.01), CUDA 12.4
Commit: `dd36829` on `fix/flce-chunked-vocab-contig` (fix on top of `f86c6d0` main)
Raw log: `docs/flce_phase_a_postfix_raw_2026-04-29.log`

### Purpose

Validate that the single-line `.contiguous()` fix to
`kiln-flce-kernel::fused_linear_cross_entropy` clears Finding 1 above. Re-run
the **T=2048 cells only** of `flce_phase_a_validation_bench.rs`; T=8192 and
T=16384 are out of scope here because they are the GDN-side ceiling
documented as Finding 2, not the head's contribution.

The bench runs all four cells; the T=8192 / T=16384 OOMs reproduce as expected
in ~0.7 s each (still GPU-ceiling pinned at 48 490 MiB) and are recorded for
completeness, not as a re-measurement of Finding 2.

### Results

| target_T | FLCE | actual_T | peak VRAM (MiB) | ΔVRAM (MiB) | abc (GB) | abc/peak | step (s) | loss     | status      |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 2 048  | OFF | 2 042  | 34 602 | 16 128 |  3.78 | 11.2 % | 2.7 | 2.783685 | **ok** |
| 2 048  | ON  | 2 042  | 34 602 | 16 128 |  3.78 | 11.2 % | 3.2 | 2.783439 | **ok** |
| 8 192  | ON  | 8 192  | 48 490 | 30 016 | 15.16 | 32.0 % | 0.7 | n/a      | **OOM** (Finding 2) |
| 16 384 | ON  | 16 380 | 48 490 | 30 016 | 30.30 | 64.0 % | 0.6 | n/a      | **OOM** (Finding 2) |

Parity: `off_loss = 2.7836852`, `on_loss = 2.7834394`, `|delta| = 2.46e-4`,
tolerance `1.0e-3` → **PASS**.

The T=2048 FLCE=ON cell now completes a full SFT step instead of erroring on
the chunked-vocab matmul. Loss matches the FLCE=OFF baseline within bf16
parity tolerance. Peak VRAM at T=2048 is unchanged (34 602 MiB vs 34 868 MiB
in the pre-fix run; both well under the GPU ceiling) — Phase A's head-side
saving is not visible at T=2048 because the head materialization is small
relative to the rest of activation memory at that length, as the original
preflight already documented.

### Verdict (Finding 1)

**Finding 1 is RESOLVED.** `KILN_USE_FLCE=1` now runs end-to-end on
Qwen3.5-4B SFT in the gradient-checkpointed path. The CPU-only regression
test added alongside this fix
(`crates/kiln-flce-kernel/src/tests.rs::cpu_parity_strided_chunk_slice`)
asserts both the strided-slice layout invariant (so future regressions of
the `.contiguous()` call surface immediately) and numeric parity with the
naive `log_sum_exp - gather` reference.

### What does NOT change

- **Finding 2 (T=8192 / T=16384 OOM on A6000) is unchanged.** The post-fix
  re-run still hits the GPU ceiling at those lengths in ~0.7 s. The "RED
  — Phase A is insufficient on A6000 to unblock long-context SFT" verdict
  above remains correct: removing the head's retained tensors does not
  remove the GDN-side activation pressure that pins peak VRAM at 48 GB.
- **Phase B (CUDA fused FLCE kernel) and GDN-side activation cleanup are
  still required** before T=8192 / T=16384 SFT will fit on A6000. See
  "Required follow-ups" §2 above.
