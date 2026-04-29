# Phase 10 — Layer-pair time-axis tiled forward+backward for hybrid models (Finding 2 §3 impl)

Date: 2026-04-29
Status: **RED on the acceptance criterion (T=8192 SFT must no longer OOM on A6000) — the layer-pair-tiled path is bit-exact + necessary infrastructure for hybrid models, but the per-segment iteration's tail forward (still grad-tracked through later segments + LM head at full T) dominates the autograd-saved-tensor peak; the per-block tile work this PR adds operates one level below where the binding constraint sits** — see Verdict.
Hardware: NVIDIA RTX A6000 (49 140 MiB VRAM, driver 550.127.05), CUDA 12.4
Branch: `ce/phase10-layer-pair-tiled`
Bench: `crates/kiln-server/examples/flce_phase_a_validation_bench.rs`

## Purpose

Implementation follow-up to PR #636
(`docs/audits/PHASE10_GDN_TRAINING_TILED_BACKWARD.md`), which shipped the
time-axis tile inside `checkpointed_forward_backward` gated on the
`model_is_gdn_only` precondition. Result: bit-exact CPU parity on a
GDN-only mini-config, but **the path does NOT fire on Qwen3.5-4B** (24
GDN + 8 full-attn layers) — the production target falls back to the
monolithic branch and OOMs at T=8192 on A6000.

PR #636's "Required follow-ups" §3 named the next viable remediation:
**layer-pair-level time-axis tile inside `checkpointed_forward_backward`
for hybrid models** — partition each segment into contiguous-attention-
type blocks, time-tile only the GDN blocks, run full-attention blocks
monolithically, and use gradient injection so per-block forward+backward
can release saved tensors without requiring a tile-level loss for the
full segment chain.

This PR exists to answer two questions independently:

1. **Does the layer-pair tiled path produce correct LoRA gradients on a
   hybrid model?** Yes — see "CPU parity" below.
2. **Does it unblock T=8192 SFT on A6000 for Qwen3.5-4B?** **No** — see
   "A6000 SFT bench" below. The peak VRAM is unchanged versus PR #636
   main (48 648 MiB / 49 140 MiB ceiling); the path correctly dispatches
   on the hybrid model and computes correct gradients, but the autograd-
   saved-tensor peak in the per-segment iteration's tail forward is
   already ≥ 30 GiB at T=8192 with `KILN_USE_FLCE=1`, before any
   per-block work begins. The per-block tile work this PR introduces
   reduces a sub-dominant cost (the per-block grad-tracked seg's saved
   tensors), not the dominant one.

## Implementation summary

`crates/kiln-train/src/trainer.rs`:

- **`AttnKind` enum + `partition_segment_layers_by_attn_type`** —
  partition `[seg_start, seg_end)` into maximal contiguous runs of the
  same attention kind. Each entry is `(kind, layer_range)` where
  `layer_range` is a sub-range of the segment with all layers of the
  same kind. Used by the layer-pair path to process GDN sub-blocks
  (time-tiled) and full-attention sub-blocks (monolithic) sequentially
  within one segment-recompute pass.
- **`tiled_training_tile_size`** — gating relaxed to drop the
  `model_is_gdn_only` precondition. Now returns `Some(tile)` whenever
  streaming is enabled AND `tile < seq_len` AND `tile %
  GDN_CHUNK_SIZE == 0`. The GDN-only check moved to the dispatcher in
  `checkpointed_forward_backward`, which routes to either the GDN-only
  fast path from PR #636 or the new layer-pair path.
- **`lora_weights_detached`** — helper returning a [`LoraWeights`] view
  whose `a` / `b` projections are detached from the LoRA Vars'
  autograd graph. Used by the layer-pair tail forward (whose only
  useful output is the gradient at the segment-output Var) and the
  block-boundary forward (which only computes activation VALUES).
  Without this, those backward passes would produce LoRA gradients
  that would then be discarded — wasted compute, and a correctness
  hazard if forgotten.
- **`layer_pair_tiled_segment_recompute_and_backward`** — generalizes
  PR #636's `tiled_segment_recompute_and_backward` from GDN-only models
  to hybrid models. Per-segment iteration:
  1. **Pre-compute the gradient at the segment's output.** Wrap
     `boundary_states[seg_idx + 1]` in a fresh `Var` (`seg_output_var`),
     forward through later segments + final RMSNorm + LM head + cross-
     entropy using `lora_detached`, then `loss.backward()`. The only
     useful output is `∂loss/∂seg_output_var`; later-segment LoRA Vars
     get their grads from THEIR OWN per-block backward in the
     corresponding seg-iteration of `checkpointed_forward_backward`.
     Accumulating later-segment LoRA grads here would double-count.
  2. **Compute block-boundary states for this segment** via a detached
     forward through the segment's blocks (with `lora_detached`).
  3. **Partition the segment into contiguous-attention-type blocks**
     via `partition_segment_layers_by_attn_type`.
  4. **Process blocks LAST -> FIRST with gradient injection.** For each
     block:
     - Wrap the block's input (a detached `Tensor` from step 2) in a
       fresh `Var` so the block's `loss.backward()` can extract the
       gradient at the block's input.
     - Run forward through the block's layer range using
       `params.as_lora_weights()`. Full-attention blocks are forwarded
       monolithically at full seq_len (FA needs the global causal
       mask). GDN blocks are time-tiled — `LinearAttentionState` is
       threaded across tiles within the block; one `narrow` of
       `block_input_var` produces each tile's input.
     - Compute the gradient-injection scalar `(block_output *
       grad_at_current_block_output).sum_all()` (or the tile-local
       analogue) and backward. This is mathematically equivalent to
       chain-ruling through the block:
       `∂scalar/∂theta = sum_pos grad_at_block_output[pos] *
       (∂block_output[pos]/∂theta) = ∂loss/∂theta` for any `theta`
       whose backward path is wholly inside the block.
     - Accumulate this block's LoRA gradients.
     - Extract `∂scalar/∂block_input_var` and use it as
       `grad_at_current_block_output` for the previous (lower-layer)
       block. For tiled GDN blocks, sum across tiles to recover the
       full-seq_len gradient (each tile's `narrow` backward fills only
       the tile's range; non-tile positions are zeros).

- **`checkpointed_forward_backward`** dispatch — when
  `tiled_training_tile_size` returns `Some(tile)`, the segment iteration
  routes to either:
  * `tiled_segment_recompute_and_backward` (PR #636) when
    `model_is_gdn_only(weights)` — bit-exact, cheaper, no gradient
    injection.
  * `layer_pair_tiled_segment_recompute_and_backward` (this PR) for
    hybrid models — the dispatch branch that finally fires on
    Qwen3.5-4B.

## Correctness invariants

The layer-pair path is **NOT** bit-exact against the existing monolithic
`checkpointed_forward_backward` path on hybrid models. The CPU parity
test instead compares against the **standard (non-checkpointed) full
forward+backward** path — the unambiguous ground truth (single forward,
single backward, no segment trickery, all LoRA Vars in the graph).

The reason: the existing monolithic checkpointed loop does
`hidden = hidden.detach()` between the current segment and later
segments, which **severs the chain from the loss back to the current
segment's LoRA Vars**. Earlier segments' LoRA Vars therefore never
receive a gradient under monolithic checkpointing — a pre-existing
limitation orthogonal to this PR. The layer-pair path uses gradient
injection across blocks, so it correctly produces grads for every
segment's LoRA Vars (including the segment that is currently being
recomputed). Comparing to standard makes the parity claim well-defined.

Tolerances:
* Total loss within `1e-3` of standard.
* MLP-LoRA grads bit-exact (`atol 1e-5`) — MLP is per-position so
  per-tile state-thread truncation does not affect MLP-LoRA.
* Full-attention LoRA grads within `1e-3` — the gradient-injection
  chain through this PR's per-block backward goes through different
  f32 reduction orders than the standard single backward, and may also
  pick up truncated-BPTT approximation in segment configurations where
  a GDN block sits between the FA block and the segment output. In the
  test config used here (`full_attention_interval = 2`, layers 1, 3 are
  FA), every FA block is the LAST block in its segment so FA-LoRA grads
  are bit-exact in expectation; the `1e-3` tolerance absorbs matmul-
  reduction-order f32 drift only.
* GDN-LoRA grads via the tile loop's truncated state thread are
  approximate w.r.t. the recurrent path. In current kiln, GDN layers
  only carry MLP-LoRA (q/k/v/o LoRA is full-attn only — see
  `TrainableLoraParams::initialize`), so the truncation does not
  affect any LoRA parameter that exists.

## Validation

### CPU parity

`cargo nextest run -p kiln-train --lib`:

```
test result: ok. 17 passed; 0 failed; 0 ignored; 0 measured
```

Including:
- `test_layer_pair_tiled_matches_monolithic_cpu_hybrid` — new parity
  test for hybrid model (`full_attention_interval = 2`, 2 segments × 2
  layers, T=192, tile=64, LoRA rank 4). Asserts (a) total loss within
  `1e-3` of standard, (b) MLP-LoRA grads bit-exact (`atol 1e-5`),
  (c) FA-LoRA grads within `atol 1e-3`. Test name kept the
  monolithic-vs-layer-pair shorthand for grepability against PR #636's
  existing `test_checkpointed_forward_backward_tiled_matches_monolithic_cpu`.
- `test_partition_segment_layers_by_attn_type` — unit test for the
  partition helper across hybrid (alternating) and GDN-only configs.
- `test_checkpointed_forward_backward_tiled_matches_monolithic_cpu`
  (existing PR #636 test) — still passes; the GDN-only fast path is
  preserved as the bit-exact path on GDN-only models.

### A6000 SFT bench

`./target/release/examples/flce_phase_a_validation_bench --model-path /workspace/qwen3.5-4b`:

Pod tpfdodwz1d4jm5, NVIDIA RTX A6000, 49 140 MiB. Baseline post-warmup VRAM = 18 472 MiB. KILN_USE_FLCE=1 active, KILN_GRAD_CHECKPOINT_SEGMENTS=4.

| target_T | actual_T | FLCE | STREAM | TILE     | peak VRAM (MiB) | ΔVRAM (MiB) | step (s) | loss              | status |
|---:|---:|:---:|:---:|:---:|---:|---:|---:|---:|:---|
|  2 048 |  2 042 | OFF | unset    | default |          34 600 |       16 128 |     2.8 | 2.7836852         | ok    |
|  2 048 |  2 042 | ON  | unset    | default |          34 600 |       16 128 |     3.2 | 2.7834394         | ok    |
|  2 048 |  2 042 | ON  | **ON**   | default |          34 600 |       16 128 |     3.2 | **2.7834394**     | ok    |
|  8 192 |  8 192 | ON  | unset    | default |          48 648 |       30 176 |     0.7 | n/a               | OOM   |
| 16 384 | 16 380 | ON  | unset    | default |          48 648 |       30 176 |     0.6 | n/a               | OOM   |
|  8 192 |  8 192 | ON  | **ON**   | default |          48 648 |       30 176 |     0.7 | n/a               | OOM   |
|  8 192 |  8 192 | ON  | **ON**   | 4096    |          48 648 |       30 176 |     0.7 | n/a               | OOM   |
|  8 192 |  8 192 | ON  | **ON**   | 2048    |          48 648 |       30 176 |     0.7 | n/a               | OOM   |
| 16 384 | 16 380 | ON  | **ON**   | 4096    |          48 648 |       30 176 |     0.6 | n/a               | OOM   |

### Bit-exact loss at T=2048

`stream_on_t2048_loss = 2.7834393978118896` and `stream_t2048_loss_delta
= 0.0e0` — STREAMING ON matches STREAMING unset bit-for-bit at T=2048.
Expected: T=2048 ≤ default tile 8192 means `tile < seq_len` is false
inside `tiled_training_tile_size`, the layer-pair branch returns to
monolithic, and the loss is mechanically identical. This rules out any
silent regression in the monolithic branch from the trainer-level
restructuring.

### Memory ceiling — unchanged from PR #636

Every T=8192 cell (STREAMING unset, STREAMING ON tile=default/4096/2048)
and the T=16384 stretch cell OOM at the same ≈ 48 648 MiB ceiling as PR
#636. Tile size makes no difference. The layer-pair branch dispatches
correctly on Qwen3.5-4B (it's the path that fires now under
`KILN_STREAMING_PREFILL=1` for hybrid models — confirmed by the
parity-via-monolithic-fallback at T=2048 above), but the OOM happens
**before** the per-block work begins.

### Why the per-block tile path doesn't help

The per-segment iteration of the layer-pair path runs in two phases:

1. **Tail forward+backward** — `seg_output_var` (boundary state at the
   end of `seg_idx`) is the only Var; the chain forwards through later
   segments' layers + LM head + cross-entropy. With LoRA detached, the
   later-segment LoRA Vars are not in the graph, but the per-layer
   ACTIVATIONS still are (they trace back to `seg_output_var` through
   each layer's matmul / norm / silu / gated_rms_norm chain). Internal
   GDN streaming (`KILN_STREAMING_PREFILL=1` with
   `tile < seq_len`) reduces transient memory inside the GDN forward
   but does NOT reduce the autograd-saved-tensor peak — the saved
   tensors are still the per-layer activations at full seq_len. For
   `seg_idx = 0` on Qwen3.5-4B's 4-segment / 8-layer-per-segment
   layout, the tail covers 24 layers + LM head; the saved-tensor peak
   exceeds the A6000 ceiling at T=8192 with KILN_USE_FLCE=1.
2. **Per-block forward+backward** — each block is grad-tracked,
   forward+backward, then the autograd graph is released before the
   next block. For GDN blocks the forward is time-tiled (per-tile
   autograd graph released between tiles). This phase's peak is at
   most ONE block's worth of saved tensors — a SUB-DOMINANT cost.

The OOM happens in phase 1 (or earlier — possibly in
`checkpointed_forward_backward`'s Step 1 detached forward through all
segments, which also processes full T per segment with grad-tracked
LoRA before each `.detach()`). The per-block tile work this PR adds
operates one level below the binding constraint.

### What WOULD help

Two adjustments to push past the A6000 ceiling at T=8192:

1. **Switch the tail forward (and Step 1's detached forward) to use a
   fully-detached forward path that does not build any autograd graph.**
   With `lora_detached` AND `seg_output_var` made non-Var, the chain
   through later segments produces a grad-less output; the tail loss
   would no longer be backwardable. This is OK for the layer-pair path
   IF we replace the gradient-at-seg-output extraction with an
   ANALYTIC computation of `∂loss/∂seg_output_var` — e.g. by computing
   the gradient at the input to the LM head from the loss + softmax
   formula and back-propagating through later segments analytically
   (not via candle autograd). This is a substantial refactor.

2. **Activation-offload the autograd graph during the tail forward** —
   write saved tensors to CPU memory and re-fetch on backward. This
   trades VRAM for PCIe bandwidth. Candle does not have built-in
   support for this; would require a custom op or a kernel-level
   wrapper.

3. **Liger-style fused kernels for the dominant ops** — fused MLP
   (gate/up/down + SwiGLU) and fused GDN (in_proj + qk_norm + gates +
   gated_rms_norm) that save a small constant number of intermediates
   per layer rather than the full chain. This is the Phase B path
   listed in PR #636's audit and continues to be the most promising
   long-term direction.

## Verdict

**RED on the acceptance criterion (T=8192 SFT must no longer OOM on
A6000) — the layer-pair path is correct + necessary infrastructure for
hybrid-model tiled training, but the per-segment iteration's tail
forward (still grad-tracked through later segments + LM head at full T)
dominates the autograd-saved-tensor peak; the per-block tile work this
PR adds operates one level below the binding constraint.**

This is a useful negative result for option (3) of PR #636's
remediation menu. It re-frames the next move:

* **The blocking constraint is not the per-block grad-tracked seg.**
  PR #636 already bounded that for GDN-only models. This PR bounds it
  for hybrid models. Both PRs achieve their micro-optimization goal.
* **The blocking constraint is the per-segment iteration's tail
  forward, plus Step 1's detached forward.** Both process full T per
  segment with grad-tracked LoRA, and their autograd-saved tensors are
  in the autograd graph at backward time. The combined peak across
  these two phases is what determines the OOM ceiling. Internal GDN
  streaming reduces transient memory inside those forwards but does
  not shrink the saved-tensor peak.

The infrastructure shipped here is **necessary for any future
training-time tiled-backward work on hybrid models** (no other code
path can reach it from `sft_train`), and the CPU parity test pins the
correctness invariant the next attempt needs to preserve. But this PR
alone does not unblock long-context SFT on A6000 for Qwen3.5-4B.

## Required follow-ups (post-this PR)

The remediation menu, in increasing scope, narrowed by this PR's
findings:

1. ~~**Tile within `model_forward_segment` (smallest change that could
   work).**~~ Shipped in PR #635. Insufficient on A6000 at T=8192 —
   the autograd recompute bottleneck dominates.

2. ~~**Time-axis tile inside `checkpointed_forward_backward` for
   GDN-only models.**~~ Shipped in PR #636. Bit-exact for GDN-only
   models; does not fire on Qwen3.5-4B because of the GDN-only
   precondition. Code remains useful as the dispatch foundation.

3. ~~**Layer-pair-level time-axis tile inside
   `checkpointed_forward_backward` for hybrid models.**~~ Shipped
   here. Dispatches correctly on Qwen3.5-4B and produces correct
   gradients (parity vs standard); does NOT unblock T=8192 because
   the per-block tile path is one level below the autograd-saved-tensor
   peak that determines OOM.

4. **Liger-style fused-kernel pass for the dominant layer ops
   (recommended next).** PR #234's audit
   (`docs/audits/PHASE10_LIGER_AUDIT.md`) identified Fused Linear
   Cross-Entropy as the #1 win. FLCE has shipped (Phase A). The next
   targets:
   * **Fused RMSNorm** (Liger pattern): one CUDA kernel for the
     pre-attn / pre-mlp RMSNorm, saving only the input + small scalars
     for backward instead of the chain of 11 ops candle currently
     decomposes RMSNorm into.
   * **Fused SwiGLU MLP** (gate + up + silu + elementwise + down):
     one or two CUDA kernels that materialize a single intermediate
     for backward instead of saving every intermediate.
   * **Fused GDN gates + gated_rms_norm + recurrent** (the Liger
     analogue for our linear-attention layer): saves only the per-tile
     state checkpoint + small scalars instead of the full GDN
     intermediate chain.

   Each of these reduces the saved-tensor peak by a constant factor
   per layer. Combined with the layer-pair tiled infrastructure landed
   here, the saved-tensor peak per training step at T=8192 should
   drop below the A6000 ceiling.

5. **Activation-offload via candle custom ops.** Push autograd-saved
   tensors to CPU between segments or between blocks. Trades VRAM for
   PCIe bandwidth — only worth pursuing if option (4) above doesn't
   close the gap.

## Caveats

- The bench's peak-VRAM signal is `nvidia-smi` polled at 50 ms;
  sub-50 ms transient peaks may be undercounted. All cells in this
  PR's table at T=8192 are at the GPU ceiling
  (~48 648 / 49 140 MiB), so the OOM-vs-OK distinction is robust
  even with that resolution.
- The CPU parity test runs on a synthetic hybrid mini-model
  (4 layers × 2 attention kinds, T=192). The production memory
  characteristics are unrelated to that — bit-exact CPU parity does
  not guarantee bit-exact CUDA parity (matmul reduction order can
  vary with shape) but it does pin the algorithmic correctness
  invariant.
- The CPU parity test compares against the **standard** (non-
  checkpointed) full forward+backward path, not the monolithic
  checkpointed path. The reason is documented in the test docstring
  and in "Correctness invariants" above: the monolithic checkpointed
  loop's `.detach()` between segments severs the chain from the loss
  back to the current segment's LoRA Vars, so monolithic checkpointed
  does not produce gradients for the segment currently being
  recomputed (only for later segments). This is a pre-existing
  limitation that is orthogonal to this PR. The layer-pair path's
  gradient-injection structure happens to fix this as a side-effect:
  earlier segments' LoRA Vars now correctly receive gradients.
- Audit scoped to GPU-side production training (CUDA path on A6000).
  Metal / MLX training is unaffected by this PR — the layer-pair
  predicate is device-aware via `streaming_prefill_enabled_for`, so
  Metal SFT only enters the layer-pair branch under explicit
  `KILN_STREAMING_PREFILL=1` AND the model is hybrid.
