# Phase 10 — GDN training-time streaming-prefill audit (Finding 2 follow-up)

Date: 2026-04-29
Status: **RED — `KILN_STREAMING_PREFILL` has no effect on the SFT path; streaming GDN prefill is unreachable from training-time forward**
Hardware: NVIDIA RTX A6000 (48 GB VRAM, driver 565.57.01), CUDA 12.4
Commit: `15e3829` on `ce/phase10-gdn-streaming-audit`, off `306d26a` main (post-PR #633 FLCE Finding 1 fix)
Bench: `crates/kiln-server/examples/flce_phase_a_validation_bench.rs`
Raw log: `docs/flce_phase_a_streaming_raw_2026-04-29.log`

## Purpose

The Phase A validation (`docs/audits/PHASE10_FLCE_PREFLIGHT.md`, "Phase A
validation — 2026-04-29") closed with two findings. Finding 1 (a
`matmul-not-contiguous` defect in chunked-vocab FLCE) was fixed in PR #633.
Finding 2 — that even with the head's retained tensors removed, T=8192 and
T=16384 SFT still OOM on A6000 because per-segment GDN prefill activations
dominate peak memory — is the remaining blocker for long-context SFT and
required follow-up §2 in that doc.

This audit asks one specific question: **is the existing
`KILN_STREAMING_PREFILL` machinery (Phase 7, PRs #228/#231/#509) reachable
from the SFT training path?** If yes, setting `KILN_STREAMING_PREFILL=1`
during a training step at T≥8192 should change peak VRAM measurably and the
remediation is a small wiring change. If no, training-time GDN streaming
requires net-new implementation work and a different scope estimate applies.

## Hypothesis

Streaming GDN prefill is implemented in
`crates/kiln-model/src/forward.rs` as `model_forward_paged_streaming{,_with,
_last_token_with_last_hidden{,_with}}`, with env-driven dispatch through
`streaming_prefill_enabled_for(device, seq_len)` and tile size from
`streaming_tile_tokens_for(device)`. Both env hooks read the process
environment on every prefill call. The inference paths invoke streaming
when long prompts are detected; the training path may or may not.

The training path goes through `kiln_train::trainer::sft_train` →
`checkpointed_forward_backward` (or `standard_forward_backward`) →
`model_forward_segment` (per gradient-checkpoint segment) and either
`model_forward_head` + `cross_entropy_loss` or `model_forward_no_head` +
`fused_linear_cross_entropy`.

If `model_forward_segment` (and friends) do not call any of the
`model_forward_paged_streaming*` entry points, training-time forward will
ignore `KILN_STREAMING_PREFILL` regardless of how it is set.

## Source-level audit

Verified locally against `306d26a` main (the post-#633 fix). Greps and
reads of the files listed below.

### 1. Is streaming GDN prefill currently reachable from `sft_train`?

**No.** All streaming dispatch in `forward.rs` lives in
`model_forward_paged_streaming*` and the inner `model_forward_paged_inner`
(`crates/kiln-model/src/forward.rs:7014-7222`). The only call sites in
production code paths are:

- `model_forward_paged_streaming` (line 7014) — `pub fn` exposed for the
  inference dispatcher
- `model_forward_paged_streaming_last_token_with_last_hidden` (line 7045) —
  used by the MTP self-spec prefill
- The `_with` variants are explicit-parameter shims used by the same call
  sites and by tests (lines 10840, 10944, 11039, 11211, 11318)

Searching for any of these names from the `kiln-train` and `kiln-server`
crates:

```
$ grep -rn 'model_forward_paged_streaming\|streaming_prefill_enabled_for\
|streaming_tile_tokens' crates/kiln-train/ crates/kiln-server/src/
(zero matches)
```

The training-path forward functions `sft_train` actually invokes are
`model_forward_embed`, `model_forward_segment`, `model_forward_head`,
`model_forward_no_head`, and `model_forward_final_norm`. None of them
reference `streaming_prefill_enabled_for`, `streaming_tile_tokens`,
`model_forward_paged_streaming`, or any `KILN_STREAMING_*` env helper.

### 2. Minimum surface area to wire streaming into training

The training-time GDN prefill happens inside `model_forward_segment`
(`forward.rs:5699`). For every layer in the segment range, that function
materializes the full `[1, T, hidden]` activation tensor at the layer's
output. There is no chunking along T.

Wiring streaming into training would require either:

1. **Tile within `model_forward_segment`.** Add a streaming branch that
   slices `hidden` along T into tiles of `streaming_tile_tokens_for(device)`,
   threads the linear-attention recurrent state across tiles within the
   segment, and stitches the per-tile outputs back into a `[1, T, hidden]`
   tensor for the segment's output. This is bit-exact with the monolithic
   path *for inference* because `LinearAttentionState` provides the O(1)
   hand-off, but for training we additionally need the per-tile activations
   to participate in the autograd graph correctly. The candle backward pass
   currently computes per-segment gradients from the segment's full-T input
   and full-T output; intermediate per-tile activations get recomputed when
   the segment is recomputed during checkpointing, which is fine, but the
   per-tile re-stitch must preserve gradient flow (effectively, the tile
   stitch is a `cat` along the T axis with grad).

2. **Tile at the segment boundary.** Have
   `checkpointed_forward_backward` segment along **both** axes — layer
   axis (existing) and time axis (new). The trainer would split each
   `[start_layer..end_layer)` segment into M time tiles, run forward per
   tile threading `LinearAttentionState`, and only retain the last-tile
   activations at the segment boundary for backward.

Option 1 is the smaller change but adds complexity inside the shared
`model_forward_segment` (which is also called by inference code paths via
`model_forward_no_head`). Option 2 keeps the tiling inside the trainer but
duplicates more of the forward loop.

Either option is **net-new implementation work** of comparable scope to a
small kernel-vendor task. Neither is a one-line wiring change.

### 3. Does `KILN_GRAD_CHECKPOINT_SEGMENTS` segment along the layer axis or the time axis?

**Layer axis only.** `compute_segment_boundaries` (`trainer.rs:1181-1193`)
partitions `[0..num_layers)` into N roughly-equal segments. Default at 32
layers / 4 segments is `[(0,8), (8,16), (16,24), (24,32)]`. The full T
dimension is processed in one shot per segment via
`model_forward_segment(start, end)`. There is no time-axis tiling in the
training loop today.

### 4. Would streaming compose with gradient checkpointing?

Gradient checkpointing already uses `model_forward_segment` per layer-segment
and recomputes the segment from a detached boundary state. At T=8192 with 4
segments, each segment processes 8 layers' worth of `[1, 8192, 2560]`
activations through GDN intermediates. The GDN prefill activations
(`causal_conv1d` F32 promotion ~2 GiB at T=8192, `l2_normalize` F32 buffers
~1 GiB at T=8192, GQA head expansion intermediates) all materialize at the
full `[1, 8192, ·]` shape per layer in the segment. Tiling along T inside
the segment **would** reduce these intermediates to per-tile shapes,
multiplicatively with gradient checkpointing (i.e., layer-axis ÷ N_seg AND
time-axis ÷ M_tile).

So composition is fine in principle. The implementation gap is the
training-side dispatch (no streaming branch in `model_forward_segment`),
not the algebra.

## Empirical check

Augmented `flce_phase_a_validation_bench.rs` to toggle
`KILN_STREAMING_PREFILL` and `KILN_STREAMING_TILE_TOKENS` per cell using
the same `std::env::set_var` pattern the existing `KILN_USE_FLCE` cell
toggle uses. Added four cells; reordered so the T=2048 STREAMING ON
control runs **before** the T=8192/T=16384 OOM cells (the original
ordering's T=2048 STREAMING ON peak was contaminated by post-OOM CUDA
allocator residue from the preceding OOM cells).

Ran on RunPod A6000 (`pod-d76547b152be6d45b2e56452`, image
`ghcr.io/ericflo/kiln-runpod:latest`) with `KILN_W4A16` and
`KILN_KV_CACHE_FP8` unset (matches Phase A baseline conditions),
gradient checkpointing on (default 4 segments, auto-detected from
51.5 GiB VRAM), rank-8 LoRA, single SFT step per cell. Peak VRAM
sampled at 50 ms cadence on a background `nvidia-smi` poller.
Baseline post-warmup = **18 474 MiB** (~18.0 GB).

### Results

| target_T | actual_T | FLCE | STREAM | TILE     | peak VRAM (MiB) | ΔVRAM (MiB) | step (s) | loss        | status |
|---:|---:|:---:|:---:|:---:|---:|---:|---:|---:|:---|
| 2 048 | 2 042 | OFF | unset | default | 34 602 | 16 128 | 2.7 | 2.78368521 | ok |
| 2 048 | 2 042 | ON  | unset | default | 34 602 | 16 128 | 3.2 | 2.78343940 | ok |
| 2 048 | 2 042 | ON  | **ON**    | default | **34 602** | **16 128** | 3.2 | **2.78343940** | **ok** |
| 8 192 | 8 192 | ON  | unset | default | 48 490 | 30 016 | 0.7 | n/a         | OOM |
| 16 384 | 16 380 | ON | unset | default | 48 490 | 30 016 | 0.7 | n/a         | OOM |
| 8 192 | 8 192 | ON  | **ON**    | default | 48 490 | 30 016 | 0.7 | n/a         | OOM |
| 8 192 | 8 192 | ON  | **ON**    | 4096    | 48 490 | 30 016 | 0.7 | n/a         | OOM |
| 8 192 | 8 192 | ON  | **ON**    | 2048    | 48 490 | 30 016 | 0.7 | n/a         | OOM |

Three observations make the source-level audit's "streaming is unreachable
from training" hypothesis empirically hard to deny:

1. **T=2048 control delta is exactly 0 MiB.** The T=2048 FLCE=ON
   STREAMING=ON cell's peak VRAM is **34 602 MiB**, identical to the
   T=2048 FLCE=ON STREAMING=unset cell's **34 602 MiB**. The bench
   reports `T=2048 peak delta (STREAMING ON vs unset): 0 MiB`. If the
   streaming branch were reached and tiled GDN prefill within the SFT
   forward, peak VRAM would change. It does not.
2. **Final loss matches bit-for-bit.** Both the STREAMING=unset and
   STREAMING=ON T=2048 cells produce final loss
   `2.7834393978118896` (full f32 precision printed by the bench). If
   the streaming branch were reached, even a bit-exact streaming path
   would walk slightly different ops (per-tile RMSNorm, per-tile
   convolution, per-tile recurrent state hand-off) and observable bf16
   output noise would diverge. It does not.
3. **T=8192 OOMs identically across all tile sizes.** The most
   discriminative test of whether streaming is reachable is the
   tile-size sweep at T=8192: tile=8192 (default, no actual tiling),
   tile=4096 (would split T into 2 chunks), tile=2048 (would split T
   into 4 chunks). Smaller tiles slash the per-tile GDN intermediate
   memory linearly. If streaming were reachable, tile=2048 should
   reduce peak VRAM by ≈3× relative to tile=8192. Instead, all three
   cells land on the **same 48 490 MiB** ceiling, the **same 30 016 MiB**
   delta, and the **same 0.7 s** failed-allocation step time. They are
   indistinguishable.

The source-level finding (streaming dispatch lives only in
`model_forward_paged_streaming*`, none of which are called by the SFT
path) is the parsimonious explanation: the env flag is parsed without
effect.

## Verdict

**RED — streaming GDN prefill is not reachable from the SFT training
path. `KILN_STREAMING_PREFILL` and `KILN_STREAMING_TILE_TOKENS` have no
measurable effect on training-time peak VRAM at T=2048, and do not
unblock T=8192/T=16384 SFT on A6000.**

This is a useful negative result. It rules out the "minimum-effort
wiring" remediation for Finding 2 and frames the next move as
implementation work, not configuration work.

## Required follow-ups (post-audit)

The remediation menu, in increasing scope:

1. **Tile within `model_forward_segment` (smallest change that could
   work).** Add a streaming branch that splits the segment's T input
   into tiles of `streaming_tile_tokens_for(device)` and threads
   `LinearAttentionState` per tile. Reuse the inference-side tile-loop
   logic in `model_forward_paged_streaming_with` as the design
   reference; the segment variant has no paged KV (`KvCache=None` for
   GDN layers; full-attn layers in training carry no KV cache either).
   Add a CPU parity test against the monolithic `model_forward_segment`
   at small T and a GPU memory-reduction test at T=8192. Estimated
   scope: ~150-300 LOC in `forward.rs`, ~50-100 LOC in `trainer.rs`
   if the `streaming_tile_tokens_for` env is honored at the segment
   level (not the trainer level).

2. **Time-axis tile inside `checkpointed_forward_backward`.** Add a
   per-segment time-tile loop that splits each layer-segment into M
   time tiles, runs forward+backward per tile, and accumulates LoRA
   gradients across tiles within a segment (the existing layer-axis
   accumulation pattern generalizes). This keeps the change inside
   the trainer but duplicates more of the forward dispatch. Estimated
   scope: ~200-400 LOC in `trainer.rs`.

3. **Both, on top of a CUDA fused FLCE kernel (Phase B).** Phase B's
   head-side memory savings stack with GDN-side time-axis tiling. The
   combination is what closes Finding 2. Phase B alone cannot
   (already-shipped Phase A, even bug-free, leaves the GDN-side
   activation pressure intact — that's exactly what the post-fix
   re-run on 2026-04-29 already showed).

Recommended next slice: option (1) as a focused PR — it is the
smallest change that produces a measurable T=8192 SFT peak VRAM
reduction on A6000, and it stacks correctly with Phase B when the
fused CUDA FLCE kernel lands.

The Phase 10 roadmap claim "shipping streaming GDN prefill at
training time will unblock T=8192 SFT on A6000" stays plausible but
**unverified**; this audit narrows it to "shipping a training-time
streaming dispatch (option 1 above) can unblock T=8192 SFT on A6000,
contingent on actual GDN-intermediate memory reduction at the
training-side tile boundary." That contingent claim is what the
follow-up implementation PR should validate end-to-end.

## Caveats

- The bench measures peak VRAM via `nvidia-smi` polling at 50 ms; sub-
  50 ms VRAM transients may be undercounted. The reported peak is a
  lower bound. This is the same caveat the Phase A preflight and
  validation use, so the cross-cell comparison is consistent.
- The bench uses a single SFT step per cell. Multi-step or multi-epoch
  training might surface optimizer-state or cumulative-checkpoint
  growth that this single-step measurement does not capture. The
  audit's hypothesis (streaming flag is parsed without effect) is
  unaffected by single-step vs multi-step.
- The T=2048 FLCE=ON STREAMING=ON cell was specifically reordered
  ahead of the OOM cells to avoid post-OOM CUDA allocator residue
  contaminating its peak. The first run of this bench (with the
  control cell after the OOM cells) reported a spurious 13 888 MiB
  delta in that cell because the allocator had not released after the
  T=8192 OOM. The reordered run (in the table above) shows the true
  delta of 0 MiB. The raw log captures both cell orderings'
  outcomes for the T=8192 OOM cells; only the reordered run has a
  trustworthy T=2048 STREAMING ON peak.
- Audit scoped to GPU-side production training (CUDA path on A6000).
  Metal/MLX training is out of scope. The Metal streaming threshold
  default (2048 tokens, `STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD`)
  would similarly not affect Metal SFT for the same source-level
  reason: trainer never calls `model_forward_paged_streaming*`.
- The empirical check did not measure a non-OOM fallback at T=8192
  (e.g., dropping rank, reducing checkpoint segments) because Finding
  2 is specifically about reaching the T=8192 SFT cell with current
  defaults. Deferred to the follow-up implementation PR.
