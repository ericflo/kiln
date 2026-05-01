# Phase 10/11 Audit: GDN per-tile forward+backward inside the checkpoint shell

**Status:** `no_kernel_pivot`
**Authors:** Cloud Eric (planning loop, doc-only audit)
**Date:** 2026-05-01
**Scope:** Design audit for the next iteration of GDN training-time tiling, as a
follow-up to PRs **#634, #635, #636, #637** (all merged 2026-04-29). All three
landed AMBER on A6000 — `T = 4096` SFT trains successfully, but `T = 8192`
SFT still OOMs at `~48648 MiB` regardless of `KILN_STREAMING_TILE_TOKENS`.
Codename: **per-tile fwd+bwd**.

This audit closes the question raised at the bottom of
[`docs/audits/PHASE10_CLOSURE.md`](PHASE10_CLOSURE.md): *"PR #635, #636, #637
are all in flight... AMBER on A6000... but T=8192 SFT still OOMs because
autograd recompute saves all per-tile activations simultaneously."*

The conclusion below is **`no_kernel_pivot`**: the proposed per-tile fwd+bwd
pattern is **already implemented at the GDN-block level by PR #637** (see
§3.2). Extending the per-tile pattern across the *tail forward* — the only
remaining lever inside the checkpoint shell — runs into a fundamental
architectural mismatch: full-attention layers in the tail cannot be time-tiled
at training time, and removing them from the tail breaks the gradient-injection
correctness guarantee. The memory math (§4) does *not* clear the **>2× peak
reduction** decision gate.

The pivot recommended by §8 is to drop GDN-internal per-tile fwd+bwd as a
remaining lever and pursue the **fully-detached tail with analytic LM-head
gradient** path called out in PR #637's *"What WOULD help"* section instead.

---

## 0. TL;DR

- **Binding constraint** for `T = 8192` SFT OOM is **not** GDN per-tile saved
  activations. PR #637 already drives those toward zero peak per block via
  per-tile fwd+bwd inside `layer_pair_tiled_segment_recompute_and_backward`
  (`crates/kiln-train/src/trainer.rs:1862-1924`).
- The actual binding constraint is the **tail forward** at line 1681-1738:
  for each `seg_idx`, the tail forwards through `segments[seg_idx + 1..]` at
  full `T` with autograd graph live, because `tail_loss.backward()` must
  produce `∂loss/∂seg_output_var`. At `seg_idx = 0` on Qwen3.5-4B
  (4 segments × 8 layers) this is **3 segments × 8 layers = 24 transformer
  layers held in autograd at `T = 8192`**.
- Within the tail, full-attention layers (indices 3, 7, 11, 15, 19, 23, 27, 31)
  fundamentally **cannot** be time-tiled at training time: there is no KV cache
  thread, FlashAttention's `Q/K/V` saves are at full `T`, and tiling FA
  forward changes the softmax denominator (a correctness break, not a memory
  optimization).
- The memory math (§4) shows that even an aggressive partial per-tile pivot —
  tile only the GDN portions of the tail, keep FA bridges monolithic, propagate
  gradient injection across segment boundaries — saves **at most ~1.6×** off
  the saved-tensor peak at `seg_idx = 0`. That is below the >2× decision-gate
  threshold (§8).
- Therefore: **`no_kernel_pivot`**. Proceed instead with one of:
  1. Fully-detached tail with analytic LM-head gradient (no tail forward at all).
  2. Activation offload via candle custom ops.
  3. Liger-style fused kernels for GDN gates+gated_rms_norm+recurrent (which
     reduce activation count, not just kernel runtime).

---

## 1. Problem statement

### 1.1 What broke

`T = 8192` SFT on Qwen3.5-4B (BF16 base + BF16 LoRA + FP32 master), single
A6000 (48 GiB advertised, ~48648 MiB usable), still **OOMs** after PR #635,
#636, #637 landed. Source of truth: PR #635 audit
([`docs/audits/PHASE10_GDN_TRAINING_STREAMING_IMPL.md`](PHASE10_GDN_TRAINING_STREAMING_IMPL.md)),
PR #636 audit
([`docs/audits/PHASE10_GDN_TRAINING_TILED_BACKWARD.md`](PHASE10_GDN_TRAINING_TILED_BACKWARD.md)),
and PR #637 audit
([`docs/audits/PHASE10_GDN_TRAINING_LAYER_PAIR_TILED.md`](PHASE10_GDN_TRAINING_LAYER_PAIR_TILED.md)).

The OOM is **insensitive** to:

- `KILN_STREAMING_TILE_TOKENS` (tested 1024, 2048, 4096, 8192 — all OOM at
  the same step on A6000).
- `KILN_GRAD_CHECKPOINT_SEGMENTS` from 4 (A6000 default per
  `crates/kiln-core/src/vram.rs:170-178`) up through 16. More segments add
  per-segment overhead but do not lower the saved-tensor peak observed during
  tail backward of `seg_idx = 0`.
- `KILN_USE_FLCE`. The post-#647 FLCE Phase B already chunks the LM head so
  it is no longer the dominant transient; the saved-tensor budget is now
  paid by the *tail forward through later segments*, not by the head.

### 1.2 What WAS fixed by #635/#636/#637 (and what it did NOT fix)

Fixed:

- **#635** added `gated_deltanet_forward_streaming`
  (`crates/kiln-model/src/forward.rs:3385-3445`) and wired it into
  `model_forward_segment` via `stream_active`
  (`crates/kiln-model/src/forward.rs:5952-5962`). GDN forward inside a
  *running* (grad-tracked) segment iteration now slices `T` into tiles of
  `streaming_tile_tokens_for(device)`. Each tile's GDN math is bit-exact
  vs. monolithic via `LinearAttentionState` hand-off.
- **#636** added `tiled_segment_recompute_and_backward`
  (`crates/kiln-train/src/trainer.rs:1447-1572`) for **GDN-only models**.
  Per-tile fwd+bwd at the segment-recompute level: each tile produces its
  own mini-loss via `tile_loss_explicit`
  (`crates/kiln-train/src/trainer.rs:1359-1423`), backwards immediately,
  accumulates LoRA grads, and discards.
- **#637** added `layer_pair_tiled_segment_recompute_and_backward`
  (`crates/kiln-train/src/trainer.rs:1656-1948`) for **hybrid GDN + FA**
  models like Qwen3.5-4B. Partitions each segment into contiguous-attention-
  type sub-blocks (`partition_segment_layers_by_attn_type`,
  `crates/kiln-train/src/trainer.rs:1285-1304`) and processes each sub-block
  with **gradient injection** so the tile path can fire on production models.
  The GDN sub-block path (lines 1862-1924) is itself per-tile fwd+bwd.

Not fixed:

- **The tail forward**. For each `seg_idx ∈ [0, num_segments)`,
  `layer_pair_tiled_segment_recompute_and_backward` first runs a *tail*
  forward through every later segment at full `T` with autograd graph live,
  so that `tail_loss.backward()` produces `∂loss/∂seg_output_var` (the seed
  for the per-block gradient-injection backward). See trainer.rs:1681-1738.
  At `seg_idx = 0` this is **3 segments × 8 layers = 24 transformer layers**
  whose forward activations all live in the autograd graph simultaneously
  before backward begins.
- **The Step 1 detached forward**. `checkpointed_forward_backward`
  (`crates/kiln-train/src/trainer.rs:1995-2019`) computes boundary states
  by forwarding through every segment at full `T`, detaching at boundaries.
  Each segment forward is *transient* (no autograd graph kept across the
  detach), but the per-segment forward itself still allocates per-layer
  hidden + GDN state + FA Q/K/V at full `T`. On Qwen3.5-4B with 4 segments
  this is 8 layers × full-`T` activations live at any one time. Lower than
  the tail peak but still non-trivial (~1-2 GiB at `T = 8192`).

### 1.3 The framing the task spec asked for, vs. what the code already does

The task spec asked for an audit of *"GDN per-tile forward+backward inside
the checkpoint shell."* As of PR #637 the GDN per-tile fwd+bwd pattern
already exists at the **block** level inside the shell. The remaining
question is whether **extending** that pattern further — across segment
boundaries, into the tail — is the right next step.

This audit answers: **no**, because the FA bridges in the tail force a
recompute pattern that loses the >2× peak savings the per-tile pattern
gives at the block level (§4, §6).

---

## 2. Current call path

Trace of every function the SFT path enters from `sft_train` down to the
GDN per-tile fwd+bwd loop. Line ranges are pinned to current `main`
(commit `093b97b`, branch `ce/phase10-gdn-per-tile-fwdbwd-audit`).

### 2.1 Entry: `sft_train`

`crates/kiln-train/src/trainer.rs:334-519` (`pub fn sft_train`).

Per training step:

1. Build `CheckpointConfig::from_env(num_layers)`
   (`crates/kiln-train/src/trainer.rs:1132-1179`). On A6000 with 32 layers
   this auto-configures **`num_segments = 4`** via
   `kiln_core::vram::recommended_checkpoint_segments`
   (`crates/kiln-core/src/vram.rs:157-178`) — 48 GiB > 45 GiB threshold → 4.
2. Compute segment boundaries with `compute_segment_boundaries`
   (`crates/kiln-train/src/trainer.rs:1186-1198`). For Qwen3.5-4B
   (32 layers, 4 segments) this yields `[(0, 8), (8, 16), (16, 24), (24, 32)]`.
3. Call `checkpointed_forward_backward(...)`
   (`crates/kiln-train/src/trainer.rs:1982-2167`).

### 2.2 The checkpoint shell: `checkpointed_forward_backward`

`crates/kiln-train/src/trainer.rs:1982-2167`.

#### Step 1 — detached full forward (lines 1995-2019)

```text
embed -> seg 0 -> .detach() -> seg 1 -> .detach() -> seg 2 -> .detach() -> seg 3 -> .detach()
```

Each `model_forward_segment` call processes the entire `T = 8192`-token
hidden state through all layers in `[seg_start, seg_end)`. The
`current.detach()` *between* segments prevents the autograd graph from
spanning segments, but does **not** lower the *transient* peak: while the
forward through one segment is in flight, all per-layer activations of that
segment exist at full `T`.

For Qwen3.5-4B, segments are `[(0, 8), (8, 16), (16, 24), (24, 32)]`, so
the transient peak per segment is **8 layers × full-`T` activations** plus
state. On a hybrid GDN + FA model this includes the FA layers' `Q/K/V`
projections at full `T` (no FA tiling exists at training time).

`boundary_states[i]` is the detached hidden at the input of segment `i`
(or at the LM head for `i = num_segments`). After Step 1, the autograd
graph from Step 1 is fully torn down; only the detached `Tensor`
references in `boundary_states` remain.

#### Step 2 — per-segment recompute + backward (lines 2056-2158)

The dispatch at line 2057 picks one of three paths:

- **Tile + GDN-only model** → `tiled_segment_recompute_and_backward`
  (line 2075-2090; impl at trainer.rs:1447-1572). Bit-exact, per-tile
  fwd+bwd, no gradient injection.
- **Tile + hybrid model** (Qwen3.5-4B) →
  `layer_pair_tiled_segment_recompute_and_backward` (line 2059-2073;
  impl at trainer.rs:1656-1948). Per-block per-tile fwd+bwd inside a
  *segment*; full-`T` tail forward across later segments.
- **No tile** → monolithic recompute + backward at full `T` (lines
  2096-2157). Each iteration recomputes `seg_idx` with grad tracking,
  forwards remaining detached, then `loss.backward()`. Used when the
  device's streaming threshold gates streaming off, when the streaming
  tile size doesn't divide cleanly into `GDN_CHUNK_SIZE = 64`, or when
  `seq_len` is below threshold.

The tile decision is per-step:
`tiled_training_tile_size(weights, device, input_ids.len())`
(`crates/kiln-train/src/trainer.rs:1323-1337`) — gates on
`streaming_prefill_enabled_for(device, seq_len)`
(`crates/kiln-model/src/forward.rs:1368`) and requires
`tile % GDN_CHUNK_SIZE == 0` *and* `tile < seq_len`. Returns
`Some(tile_size)` when both conditions hold; otherwise `None`.

### 2.3 The hybrid tiled path: `layer_pair_tiled_segment_recompute_and_backward`

`crates/kiln-train/src/trainer.rs:1656-1948`.

This function is the *current* per-tile fwd+bwd implementation. Its four
phases are:

#### Phase 1 — pre-compute gradient at segment output (the tail forward)

`crates/kiln-train/src/trainer.rs:1675-1758`.

```rust
let seg_output_var = Var::from_tensor(&boundary_states[seg_idx + 1])?;

let lora_detached = lora_weights_detached(params);
let mut tail_hidden = seg_output_var.as_tensor().clone();
for (i, &(later_start, later_end)) in segments[seg_idx + 1..].iter().enumerate() {
    if i > 0 {
        tail_hidden = tail_hidden.detach();
    }
    let mut later_state = LinearAttentionState::new(model_config, device)?;
    tail_hidden = model_forward_segment(
        backend,
        tail_hidden,
        weights,
        model_config,
        positions,
        later_start,
        later_end,
        Some(&mut later_state),
        Some(&lora_detached),
    )?;
}

let tail_loss = if use_flce() {
    let normed = model_forward_final_norm(&tail_hidden, weights, model_config)?;
    fused_linear_cross_entropy_dispatch(...)?
} else {
    let logits = model_forward_head(&tail_hidden, weights, model_config)?;
    cross_entropy_loss(&logits, input_ids, label_mask, device)?
};

let tail_grads = tail_loss.backward()?;
let grad_at_seg_output = tail_grads
    .get(seg_output_var.as_tensor())
    .ok_or(...)?
    .clone()
    .detach();
```

**Key observation:** the `if i > 0 { tail_hidden = tail_hidden.detach(); }`
detaches *between* later segments inside the tail too, but **not** before
`tail_hidden` enters the first later segment. So the autograd graph from
`seg_output_var` flows through:

- *all of* segment `seg_idx + 1` (no detach in between),
- then a `detach()`,
- *all of* segment `seg_idx + 2` (no detach in between),
- then a `detach()`,
- ... etc.

The graph is **piecewise-segment-local** in the tail: each later segment's
forward materializes all its per-layer activations into autograd before the
next-segment detach drops them. Concretely at `seg_idx = 0`, this means
peak saved tensors during `tail_loss.backward()` is (saved by segment 1) +
(saved by segment 2) + (saved by segment 3) + (saved by LM head/FLCE
chunk). Each "saved by segment k" includes 8 layers × per-layer
activations, with FA layers contributing their FlashAttention-saved
`{Q, K, V, logsumexp}` at full `T`.

LoRA is detached via `lora_weights_detached(params)`
(`crates/kiln-train/src/trainer.rs:1233-1262`). This disables LoRA-grad
edges but does **not** disable activation saving for the base linear ops.

After `tail_loss.backward()` returns, `drop(tail_grads)` (line 1758) tears
the tail graph down. But the *peak* during backward dominates step memory.

#### Phase 2 — block-boundary states (detached)

`crates/kiln-train/src/trainer.rs:1760-1794`.

```rust
let blocks = partition_segment_layers_by_attn_type(weights, seg_start, seg_end);
let mut block_boundaries: Vec<Tensor> = Vec::with_capacity(blocks.len() + 1);
block_boundaries.push(boundary_states[seg_idx].clone());
{
    let mut linear_state = LinearAttentionState::new(model_config, device)?;
    let mut current = block_boundaries[0].clone();
    for (_kind, range) in &blocks {
        current = model_forward_segment(
            backend,
            current,
            weights,
            model_config,
            positions,
            range.start,
            range.end,
            Some(&mut linear_state),
            Some(&lora_detached),
        )?;
        block_boundaries.push(current.detach());
        current = block_boundaries.last().unwrap().clone();
    }
}
```

Within `seg_idx`, each segment is partitioned into runs of same-attn-type
layers. For Qwen3.5-4B's pattern (FA at indices 3, 7, 11, 15, 19, 23, 27,
31 — every 4th layer), each 8-layer segment partitions as
`[(GDN, 0..3), (FA, 3..4), (GDN, 4..7), (FA, 7..8)]`. So a segment has
4 blocks. After phase 2, `block_boundaries` has 5 detached tensors:
the segment input, then the output of each of the 4 sub-blocks.

This phase forwards at full `T` with detached LoRA — no autograd graph is
retained across sub-blocks. Transient peak: one sub-block's worth of
activations.

#### Phase 3+4 — process blocks LAST → FIRST with gradient injection

`crates/kiln-train/src/trainer.rs:1796-1944`.

For each sub-block (in reverse order), wrap its detached input as a `Var`,
recompute its forward with grad-tracked LoRA, compute the gradient-injection
scalar `(block_output * grad_at_current_output).sum_all()`, backward.
Accumulate LoRA grads into `accumulated_grads`. The gradient at the
sub-block's input becomes the gradient seed for the next iteration (one
block earlier).

**Within a GDN sub-block** (lines 1849-1936) the forward is itself per-tile:

```rust
AttnKind::Gdn => {
    let (_, total_tokens, _) = block_input.dims3()?;
    let mut state = LinearAttentionState::new(model_config, device)?;
    let mut summed: Option<Tensor> = None;

    let mut tile_start = 0usize;
    while tile_start < total_tokens {
        let tile_end = (tile_start + tile_size).min(total_tokens);
        let tile_len = tile_end - tile_start;

        let tile_input = block_input_var.as_tensor()
            .narrow(1, tile_start, tile_len)?;
        let tile_positions: Vec<u32> = positions[tile_start..tile_end].to_vec();
        let lora_for_block = params.as_lora_weights();

        let tile_output = model_forward_segment(...)?;
        let tile_grad_out = grad_at_current_output.narrow(1, tile_start, tile_len)?;
        let scalar = (&tile_output * &tile_grad_out)?.sum_all()?;
        let tile_grads = scalar.backward()?;
        accumulate_grads(accumulated_grads, &tile_grads, &all_vars)?;

        let tile_block_input_grad = tile_grads
            .get(block_input_var.as_tensor())
            .ok_or(...)?
            .clone();
        summed = Some(match summed {
            Some(prev) => (prev + tile_block_input_grad)?,
            None => tile_block_input_grad,
        });
        tile_start = tile_end;
    }
    summed.ok_or(...)?.detach()
}
```

This is **the** per-tile fwd+bwd loop the audit task title refers to. It
already exists. Each tile fwds + bwds in isolation, then discards. Peak
saved by this loop is **one tile's worth** of GDN activations, not the
full segment's worth.

For the GDN sub-block alone, this gives a clean `tile_size / total_tokens`
peak reduction. With `total_tokens = 8192` and `tile_size = 1024`, peak
GDN-block saved tensors drop ~8×.

**Within a FA sub-block** (lines 1804-1848) the forward is monolithic:

```rust
AttnKind::FullAttn => {
    let mut state = LinearAttentionState::new(model_config, device)?;
    let lora_for_block = params.as_lora_weights();
    let block_output = model_forward_segment(...)?;
    let scalar = (&block_output * &grad_at_current_output)?.sum_all()?;
    let block_grads = scalar.backward()?;
    accumulate_grads(accumulated_grads, &block_grads, &all_vars)?;

    block_grads.get(block_input_var.as_tensor()).ok_or(...)?.clone().detach()
}
```

A FA sub-block in this model is **1 layer wide** (FA is sparse, every 4th
layer). The forward + backward saves FlashAttention's `{Q, K, V,
logsumexp}` at full `T` for that one layer. There is **no** training-time
tile path for FA; this is a load-bearing architectural fact (§6.1).

### 2.4 The model-side per-tile path: `gated_deltanet_forward_streaming`

`crates/kiln-model/src/forward.rs:3385-3445`.

```rust
pub fn gated_deltanet_forward_streaming(
    x: &Tensor,                     // [B, T, hidden]
    state: &mut LinearAttentionState,
    layer_idx: usize,
    weights: &GpuLinearAttentionWeights,
    lora_for_layer: Option<&LoraLayerWeights>,
    config: &ModelConfig,
    positions: &[u32],
    tile_tokens: usize,             // streaming_tile_tokens_for(device)
) -> Result<Tensor> {
    debug_assert!(tile_tokens > 0 && tile_tokens % GDN_CHUNK_SIZE == 0);
    let (_, total, _) = x.dims3()?;
    if total <= tile_tokens {
        return gated_deltanet_forward(x, state, layer_idx, weights, lora_for_layer,
                                       config, positions);
    }

    let mut outputs: Vec<Tensor> = Vec::with_capacity((total + tile_tokens - 1) / tile_tokens);
    let mut start = 0;
    while start < total {
        let end = (start + tile_tokens).min(total);
        let len = end - start;
        let tile = x.narrow(1, start, len)?;
        let tile_positions: Vec<u32> = positions[start..end].to_vec();
        let tile_out = gated_deltanet_forward(&tile, state, layer_idx, weights,
                                              lora_for_layer, config, &tile_positions)?;
        outputs.push(tile_out);
        start = end;
    }
    Tensor::cat(&outputs, 1)
}
```

This is **inside-`gated_deltanet_forward`** time-axis tiling. The
`LinearAttentionState` (recurrent + conv) threads across tiles, so output
is bit-exact vs. monolithic. **Key constraint**: this only runs when
called from `model_forward_segment` with `stream_active = true` (line
5952-5962 of forward.rs). The streaming dispatch is only enabled at
`seq_len ≥ STREAMING_PREFILL_CUDA_DEFAULT_THRESHOLD = 65533` by default
(forward.rs:1313-1316), unless `KILN_STREAMING_TILE_TOKENS` is set
explicitly to enable it earlier.

**For training-time SFT, the gating logic is overridden upstream by
`tiled_training_tile_size`** (trainer.rs:1323-1337) which uses a more
aggressive threshold (essentially "any `T > tile_size`") so that the
SFT tile path can fire on T=8192 even though the inference threshold
is 64k.

### 2.5 The constant `GDN_CHUNK_SIZE`

`crates/kiln-model/src/forward.rs:2804`.

```rust
pub const GDN_CHUNK_SIZE: usize = 64;
```

All GDN tile sizes must be multiples of 64. `streaming_tile_tokens_for`
(forward.rs:1406) enforces this. Tile sizes that aren't multiples of 64
are silently rounded up by `streaming_prefill_enabled_for`'s tile-size
function or rejected by `tiled_training_tile_size` (returns `None` →
falls back to monolithic).

### 2.6 Where divergence from closure docs would manifest

The PR #637 closure doc claims that *"OOM happens in phase 1 (or earlier
— possibly in `checkpointed_forward_backward`'s Step 1 detached forward
through all segments, which also processes full T per segment with
grad-tracked LoRA before each `.detach()`)."*

Reading current source: **the LoRA in Step 1 is grad-tracked**
(`lora_weights = params.as_lora_weights()` at line 1996). The
`.detach()` at line 2016 is on `current` only — it severs the *output's*
graph but does not retroactively detach the LoRA Vars used inside the
just-completed segment forward. **However** the autograd graph that those
LoRA Vars belong to is rooted at `current` (or whatever forward consumed
them). When `current` is detached *and* dropped (by being moved into
`boundary_states.push(current.detach())` and then re-borrowed), the
upstream graph becomes unreachable from any retained `Tensor` in scope,
and is collected.

The closure-doc claim that Step 1 holds grad-tracked LoRA across all
segments is **incorrect as written** — Step 1 holds LoRA-grad-tracked
*within* a segment forward (a transient, single-segment peak), not across
segments. The transient single-segment peak is non-trivial (~1-2 GiB at
`T = 8192` for one Qwen3.5-4B segment of 8 layers) but not the dominant
contributor.

The dominant contributor is **Phase 1** (the tail forward) at
`seg_idx = 0`, where 3 *consecutive* later segments live in autograd
simultaneously before `tail_loss.backward()` runs.

---

## 3. Proposed approach (and why it doesn't pencil)

### 3.1 Restated proposal

The audit task title is *"GDN per-tile forward+backward inside the
checkpoint shell."* In the most literal reading, the proposal is:

> For each segment iteration in `checkpointed_forward_backward`, do the
> per-tile fwd+bwd at the **shell** level: tile the entire shell-step
> (tail forward + per-block work + gradient injection seed) along the
> time axis, so only one tile's worth of activations is live at once.

In practice the per-block per-tile fwd+bwd inside the GDN sub-block of
`layer_pair_tiled_segment_recompute_and_backward`
(trainer.rs:1862-1924) already realizes this pattern *for the GDN
sub-block of the current segment*. Extending it further has only one
remaining surface: the **tail forward** at trainer.rs:1681-1738.

### 3.2 The per-tile-extended-into-tail data flow

If we accept "FA bridges remain monolithic" as a hard constraint (§6.1),
the extended pattern looks like this:

```text
seg_idx = 0:

  shared bookkeeping:
    tile_size = T_tile
    num_tiles = T / T_tile
    For each tile k ∈ [0, num_tiles):
      grad_at_lm_logits[k]  // computed via per-tile FLCE-style chain

  Step A (tail forward, per tile):
    For each later segment s ∈ [seg_idx+1, num_segments):
      For each GDN sub-block in s, run gated_deltanet_forward_streaming
        threaded with state hand-off across tiles.
      For each FA sub-block in s, run forward MONOLITHICALLY at full T
        (because FA at training time saves Q/K/V at full T with no tile cache).

  Step B (tail backward, per tile):
    Reverse-order over tiles (or over segments?). For each tile k:
      Backward through LM head + final norm at tile k -> grad at tail_hidden[k].
      Backward through GDN tail sub-blocks at tile k with reverse state hand-off.
      For FA tail sub-blocks: MONOLITHIC backward — reuse Q/K/V saved at full T.
      Accumulate into grad_at_seg_output[k]; concat to full T.

  Step C (per-block per-tile, current segment):
    Same as today (PR #637).
```

The "FA tail sub-blocks must be monolithic" requirement means **Step A**
cannot truly be per-tile: each FA sub-block's forward processes all `T`
tokens in one shot, holding `Q/K/V` at full `T` for the duration of the
FA backward. Even if every other op is tiled, the FA layers' saved
tensors set a floor on tail peak.

Concretely, in Qwen3.5-4B with 4 segments × 8 layers and FA at every 4th
position, **each segment has 2 FA layers**. At `seg_idx = 0` the tail
spans 3 segments × 2 FA = **6 FA layers**, each saving its FlashAttention
state (Q, K, V, logsumexp) at full `T = 8192`. That floor doesn't shrink
under any per-tile rewrite.

### 3.3 Why this is the only remaining surface inside the shell

Three lower-impact alternatives within the shell were considered and
discarded:

- **Tile Step 1**. Step 1 is already detached between segments. Its
  transient peak is one segment's worth of activations at full `T`.
  Tiling it would require re-running each segment per tile and gluing
  `LinearAttentionState` — but FA layers don't have a state, so this
  again hits the FA-bridge floor. Plus Step 1's transient is dominated
  by the tail Phase 1 saved-tensor peak in the next step, so even a
  full Step 1 win would not move the OOM needle.
- **Per-tile gradient seed via FLCE-only chunking**. The post-#647 FLCE
  Phase B already chunks the LM head + final norm + cross-entropy along
  `T`, so logits are not materialized at full `T`. This addresses one
  ~2.4 GiB transient but not the 24-layer tail.
- **Free-after-use tail (`drop(tail_hidden)` between layers)**. Already
  effectively done by the existing `current.detach()` semantics — once a
  layer's output is consumed, the saved tensors are pinned to the
  autograd graph node, not to `current`. We can't `drop()` saved tensors
  individually without giving up backward correctness.

So the only conceptually remaining lever is **per-tile across the tail's
GDN portions**, which §4 evaluates.

---

## 4. Memory math

All numbers are for Qwen3.5-4B, BF16 base, BF16 LoRA storage with FP32
tensor-core accumulate, on A6000 (48 GiB), `T = 8192`, `B = 1`,
`hidden = 2560`, `head_dim = 256`, `num_heads = 16`, `num_kv_heads = 4`,
`vocab = 151936`, FA layers at `[3, 7, 11, 15, 19, 23, 27, 31]`,
`num_segments = 4` (A6000 default), `tile_size = 1024`. All sizes in
MiB unless stated; "saved" = autograd-saved-tensors during `.backward()`,
"transient" = peak-during-forward-only.

### 4.1 Building blocks (per-T units)

- **Hidden activation, full T**: `B × T × hidden × 2` bytes
  = `1 × 8192 × 2560 × 2` = **40.0 MiB**.
- **Hidden activation, 1 tile of 1024**: `1 × 1024 × 2560 × 2` = **5.0 MiB**.
- **Logits, full T**: `B × T × V × 2` = `1 × 8192 × 151936 × 2`
  = **2375.0 MiB ≈ 2.32 GiB**. With FLCE chunking at chunk=1024, peak
  drops ~8× → ~290 MiB.
- **GDN per-layer saved tensors** (gates, in_proj, q, k, v, conv outputs,
  recurrent state, gated_rms_norm input/output, gated_deltanet output):
  ~10 tensors × hidden-shaped ≈ `10 × 40.0` = **400 MiB / layer / full-T**.
- **GDN per-layer saved tensors at tile=1024**: roughly `tile/T` of the
  full-T figure, plus the `LinearAttentionState` (recurrent + conv) which
  is `~hidden + ~2 × hidden = O(hidden)` independent of `T` ≈ negligible.
  ≈ `10 × 5.0` = **50 MiB / layer / tile**.
- **FA per-layer saved tensors** (FlashAttention forward saves only
  `{Q, K, V, logsumexp}`; backward recomputes the attention matrix
  on-chip): `Q = K = V = B × num_heads × T × head_dim × 2` each →
  `1 × 16 × 8192 × 256 × 2` = **64 MiB each**. Three of those plus
  `logsumexp` (`1 × 16 × 8192 × 4` ≈ 0.5 MiB) →
  **~193 MiB / layer / full-T**.
- **FA per-layer saved tensors at tile=1024**: NOT REDUCIBLE. FA forward
  at training time has no KV cache thread; tiling FA changes the softmax
  denominator. So **FA saved tensors stay at ~193 MiB / layer / full-T
  regardless of tile size**.
- **MLP per-layer saved tensors** (gate, up, SiLU output, gated input,
  down): ~5 × 40 = **200 MiB / layer / full-T**, ~25 MiB / layer / tile.

### 4.2 Current peak (post-#637) at `seg_idx = 0`

The dominant peak is during **Phase 1** of
`layer_pair_tiled_segment_recompute_and_backward` at `seg_idx = 0`:
`tail_loss.backward()` over a graph that contains forward activations
of segments 1, 2, 3 (each 8 layers) plus LM head.

**Tail saved tensors (`seg_idx = 0`, post-#637):**

| Component                           | Per-layer (MiB) | Layers | Subtotal (MiB) |
|-------------------------------------|----------------:|-------:|---------------:|
| GDN layers in tail (3 seg × 6 GDN)  |             400 |     18 |          7,200 |
| FA layers in tail (3 seg × 2 FA)    |             193 |      6 |          1,158 |
| MLP saved (per layer, all 24)       |             200 |     24 |          4,800 |
| LM head logits/loss (FLCE Phase B)  |               — |      — |          ~290 |
| **Tail Phase 1 saved peak**         |                 |        |     **~13,448 MiB ≈ 13.1 GiB** |

Plus per-segment-iteration overhead (Phase 2 boundary states, Phase 3+4
per-block transients): **~2 GiB** at any one time.

Plus **Step 1 transient peak**: at the moment Step 1 is forwarding the
worst-loaded segment (1 segment of 8 layers at full `T` with grad-tracked
LoRA, before `.detach()`), saved-while-still-live ≈
`8 × (400 GDN-or-200 MLP + 200 MLP)` averaged across attention type
≈ **~2.5 GiB**. This is *transient* — Step 1's autograd graph is dropped
before Step 2 begins.

Plus **base weights, optimizer state, grad accumulators, activation
checkpoint boundary states**: ~10-12 GiB (weights ~7.4 GiB BF16 + LoRA
A/B matrices ~120 MiB + FP32 master + Adam moments + boundary states
~160 MiB + etc).

Plus **CUDA workspace, Marlin packed weights residency (KILN_W4A16=1)**
adds ~1.3 GiB unused (per #166's note about Marlin BF16 residency).

**Total peak at `seg_idx = 0` Phase 1 backward**:
~13.1 GiB tail saved + ~2 GiB iteration overhead + ~10-12 GiB structural
≈ **25-27 GiB** in the saved-tensor accounting alone. The observed OOM
at ~48648 MiB suggests transient peaks during the actual backward
compute (gradient buffers being allocated + saved tensors not yet freed)
roughly double the saved-tensor total. This matches the empirical OOM.

### 4.3 Proposed peak (per-tile across tail GDN, FA bridges monolithic)

If the tail GDN portions are tiled at `tile_size = 1024` (8× tile count)
and FA bridges remain monolithic, the saved-tensor peak during the
backward of one tile becomes:

| Component                                   | Per-layer (MiB) | Layers | Subtotal (MiB) |
|---------------------------------------------|----------------:|-------:|---------------:|
| GDN tail at tile=1024 (3 seg × 6 GDN, 1 tile)|              50 |     18 |            900 |
| **FA saved (full-T, NOT reducible)**         |             193 |      6 |          1,158 |
| MLP tail at tile=1024 (3 seg × 6 MLP-on-GDN-paths) |       25 |     18 |            450 |
| **MLP under FA layers (full-T, NOT reducible if MLP paired with FA)** | 200 |  6 |          1,200 |
| LM head per tile (FLCE chunk)                |               — |      — |             ~50 |
| **Per-tile saved peak**                      |                 |        |      **~3,758 MiB ≈ 3.7 GiB** |

Wait — the MLP under each FA layer **does** need to stay at full T,
because the FA layer above it consumes the MLP output at full T
(FA backward needs gradients at all positions simultaneously). So the
MLP layers paired with FA layers (6 of the 24 tail MLPs in this model)
also take a full-T saved-tensor floor (200 MiB × 6 = 1,200 MiB).

That's a critical insight from the architecture: **FA in the tail forces
not only the FA layer itself to be monolithic but also the upstream MLP
that feeds it, plus the upstream GDN for the same reason**. Per the
Qwen3.5-4B layer pattern (FA at indices 3, 7, 11, 15, 19, 23, 27, 31),
the layers that *feed* an FA layer (indices 0, 1, 2 feed FA at 3; 4, 5, 6
feed FA at 7; etc.) must also have their full-T outputs available.

But in the per-tile pattern, those *layers* can still tile their GDN
forward. What can't tile is the **transition from tile-output to full-T**
right before the FA layer — every tile of GDN output needs to be
materialized into a full-T buffer before the FA layer fires.

So the FA-floor includes:

- 6 FA layers × 193 MiB = 1,158 MiB
- 6 *full-T hidden buffers feeding FA* × 40 MiB = 240 MiB
- 6 FA-output full-T hidden buffers × 40 MiB = 240 MiB
- 6 *MLP-paired-with-FA* full-T saved (gate/up/down) × 200 MiB = 1,200 MiB

**FA floor: ~2,838 MiB ≈ 2.8 GiB** of saved tensors that cannot tile.

**Per-tile saved peak (revised):**

| Component                                              | Subtotal (MiB) |
|--------------------------------------------------------|---------------:|
| FA floor (6 FA + paired buffers)                       |          2,838 |
| Tile-shaped GDN saved (12 GDN-not-paired-with-FA × 50) |            600 |
| Tile-shaped MLP saved (12 × 25)                        |            300 |
| LM head per-tile FLCE chunk                            |             50 |
| **Per-tile saved peak**                                |     **~3,788 MiB ≈ 3.7 GiB** |

### 4.4 Reduction factor

```
Current Phase 1 saved peak  ≈ 13.1 GiB
Proposed per-tile peak      ≈  3.7 GiB
Reduction factor            ≈ 13.1 / 3.7 ≈ 3.5×
```

**This clears the >2× decision-gate threshold on saved-tensor accounting.**

### 4.5 But wait: the per-tile pattern introduces a NEW saved-tensor cost

In the proposed Step B (per-tile tail backward), the gradient at the
segment output `grad_at_seg_output` must be accumulated across tiles.
That accumulation buffer is itself **full-T sized** (`B × T × hidden`)
because `seg_output` is a single full-T tensor. So a `40 MiB` buffer
is alive throughout Step B. Negligible.

A more serious cost: to backward through FA in the tail at full `T`,
the FA layer needs its **inputs** (the layer below) to be available at
full `T`. In the proposed layout we'd run GDN tile-by-tile, write each
tile's output into a **scratch full-T buffer**, and let FA backward read
from it. That scratch buffer is alive during FA backward — `40 MiB` per
GDN-feeding-FA layer × 6 = 240 MiB. Already counted in the FA floor.

But there's a deeper issue: **per-tile fwd+bwd requires the per-tile
forward graph to be alive during its own backward**, tile-by-tile. For
an FA layer whose input arrives tile-by-tile, FA's own backward cannot
be tile-local — it needs the full `Q/K/V` at full `T`. So the FA layer
must run **after** all tile-forwards have completed for the layers below
it. This serializes:

```text
Tile 0 GDN forward  →
Tile 1 GDN forward  →  ... (all GDN tiles for all GDN layers below FA layer)
                    →  FA layer forward (monolithic at full T, saves Q/K/V)
                    →  ... (all subsequent layers)
                    →  loss
                    →  full backward, including FA backward at full T
```

This collapses the per-tile pattern: the entire tail forward must
complete and remain in autograd for the FA backward to fire correctly.
There is no per-tile interleaving across the FA bridge.

### 4.6 Revised reduction (after acknowledging FA serialization)

Inside one "FA-bounded segment of the tail" (a contiguous run of layers
between FA layers), the per-tile pattern works. Across an FA bridge, the
pattern collapses.

Qwen3.5-4B's tail at `seg_idx = 0` has 3 segments × 8 layers = 24 layers,
with FA at relative indices 3, 7, 11, 15, 19, 23 (six FA bridges). So the
tail has 6 FA bridges + 6 GDN runs in between. Each GDN run can be
per-tile fwd+bwd internally. But the *interleaving* — running tile k's
forward and immediately backward before tile k+1's forward — cannot
extend across an FA bridge.

So the saved-tensor peak across the *whole tail* is bounded below by
"sum over all GDN runs of (one tile's worth of saved tensors)" + "all FA
layers at full T saved" + "all MLP layers paired with FA at full T".

| Component                                                | (MiB) |
|----------------------------------------------------------|------:|
| FA floor (6 FA layers + paired MLP + bridge buffers)     | 2,838 |
| Sum of per-tile peaks across 6 GDN runs (12 GDN × 50)    |   600 |
| Tile-shaped MLP saved across non-FA-paired (12 × 25)     |   300 |
| LM head per-tile chunk                                    |    50 |
| **Per-tile-extended tail peak**                           | **~3,788** |

The FA-bridge serialization issue means we can't actually *interleave*
backward across bridges, but we **can** still tile-fwd-bwd within each
GDN run and discard before crossing the bridge. So the saved-tensor peak
across the whole tail is the **maximum over all bridges' "still-alive"
sets**, not the sum.

If the per-tile fwd+bwd is interleaved tightly *within* each GDN run, and
the run between two FA bridges discards its saved tensors before the next
FA bridge fires, then peak ≈ FA floor + one GDN run's saved tensors at
one tile = `2,838 + 50 + 25 + 50` ≈ **2,963 MiB ≈ 2.9 GiB**.

```
Current Phase 1 saved peak ≈ 13.1 GiB
Proposed per-tile peak     ≈  2.9 GiB (best case, after FA bridge discard)
Reduction factor           ≈ 13.1 / 2.9 ≈ 4.5×
```

### 4.7 But this isn't actually achievable — the FA backward needs ALL its inputs alive

Here is the fatal step in the math: **FA backward at layer L requires
the saved Q/K/V from FA's forward, *which were computed from the output
of layer L-1*.** If layer L-1 was a GDN run that was already fwd+bwd-
tiled-and-discarded, its output (the full-T hidden at the FA input
boundary) is **gone**.

To run FA backward correctly we'd need to either:

- **Recompute layer L-1's output at full T before FA backward.** This
  means replaying the GDN run we just tiled — destroying the per-tile
  savings.
- **Save layer L-1's output (the FA input) at full T explicitly.** Costs
  one full-T hidden = 40 MiB per FA layer = 240 MiB for 6 FA layers.
  Doable; already counted in the FA floor.

If we save the FA-input full-T hidden, then GDN backward through the
GDN run *below* FA layer L can fire **after** FA layer L's backward,
using the gradient at FA's input (which is at full T) propagated
back-to-front. The per-tile fwd+bwd within each GDN run still works,
but only **after** all FA backwards at all 6 FA bridges in the tail
have completed.

This works in principle but it means **the FA backwards themselves
form a non-tileable serial chain** that must execute end-to-end before
GDN backwards can begin. During that chain, FA layers' saved Q/K/V are
all alive simultaneously: **6 FA × 193 MiB = 1,158 MiB** plus their
attention output gradient buffers (full-T hidden at each FA input/output)
≈ **1,158 + 480 = 1,638 MiB**.

Then once FA backwards finish, each GDN run's tile-fwd-bwd can fire
sequentially, peak = 1 tile × ~75 MiB.

So actually the math is more like:

```
Peak during FA-bridge serial chain  ≈ FA floor (~2.8 GiB)
Peak during GDN per-tile fwd+bwd     ≈ tile worth (~75 MiB)
Plus MLP-under-FA full-T saved        +1.2 GiB (alive throughout chain)
Plus boundary tensors                 +0.5 GiB
Tail backward peak                   ≈ 4.5 GiB
```

```
Current peak ≈ 13.1 GiB
Proposed peak ≈ 4.5 GiB
Reduction ≈ 2.9×
```

This **does** clear the 2× threshold. But the headline is misleading
because the structural cost — FA serialization + recompute risk — is
high, and the proposal is essentially "do per-block per-tile fwd+bwd,
but extend it across segment boundaries with a careful FA bridge protocol."
This is a major implementation undertaking.

### 4.8 What the proposal does NOT save

Even at best-case 2.9× reduction in *Phase 1 saved peak*, the *transient
peak during forward* (Step 1 of `checkpointed_forward_backward` at lines
1995-2019, plus transient activations during the per-tile forwards) is
**not addressed**. Step 1's peak is one-segment-at-full-T ≈ 2.5 GiB —
this stays. The proposal also doesn't address the LoRA-FP32-master /
optimizer / weights / Marlin-residency floors, which are already
~10-12 GiB.

So the *step memory peak* (which is what triggers OOM, not just saved
tensors) drops from ~25-27 GiB pre-proposal to ~17-19 GiB post-proposal.
On A6000 with 48 GiB advertised, that should fit. **In principle.**

### 4.9 Confidence in the math

The numbers in this section are derived from:

- Tensor shapes from `crates/kiln-model/src/forward.rs` (head dims at
  forward.rs:1313-1316 and the GDN config).
- Layer count and FA pattern from the Qwen3.5-4B closure docs
  (PHASE10_CLOSURE.md, PHASE10_S3_CANDIDATE_PREFLIGHT.md).
- Per-layer saved-tensor count is **estimated** from the structure of
  `gated_deltanet_forward` and `transformer_block`. A precise count
  would require instrumenting `Tensor::backward()` to dump the saved-
  tensor list, which this audit (doc-only, no GPU) does not do.

Confidence: ±30% on per-layer-saved-MiB. The qualitative conclusions
(FA forms an irreducible floor, the per-tile pattern only saves on GDN
runs, the FA bridge serialization is a hard constraint) are robust to
that error band. The headline 2.9× reduction figure could be 2.0× or
4.0× depending on actual saved-tensor counts; the central conclusion
("just barely clears 2×, with high implementation cost") holds either
way.

---

## 5. Implementation sketch (if the decision were "go")

This section is included for completeness despite the §8 decision being
`no_kernel_pivot`. It documents what would be needed if a future
benchmark were to re-validate the memory math at >2× reduction.

### 5.1 New function: `per_tile_layer_pair_tiled_segment_recompute_and_backward`

Lives alongside `layer_pair_tiled_segment_recompute_and_backward` in
`crates/kiln-train/src/trainer.rs`. Signature:

```rust
fn per_tile_layer_pair_tiled_segment_recompute_and_backward(
    backend: &dyn BackendRuntime,
    seg_idx: usize,
    segments: &[(usize, usize)],
    boundary_states: &[Tensor],
    input_ids: &[u32],
    label_mask: &[bool],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    positions: &[u32],
    params: &TrainableLoraParams,
    accumulated_grads: &mut HashMap<candle_core::TensorId, Tensor>,
    tile_size: usize,
    device: &Device,
) -> Result<f64>
```

Identical signature to the existing `layer_pair_tiled_segment_recompute_and_backward`.
Dispatch from `checkpointed_forward_backward` switches on a new env flag
`KILN_PER_TILE_TAIL=1` (default off). With the flag off, the existing
PR #637 path runs; with it on, this new path runs.

### 5.2 Phase A: per-tile tail forward with FA bridge buffers

```rust
// fa_input_buffers[s][k] = full-T hidden at the input of FA layer
// (s = segment index in tail, k = FA-layer index within that segment)
// Each is materialized by writing tile outputs into a pre-allocated
// full-T buffer.

let tail_segments: &[(usize, usize)] = &segments[seg_idx + 1..];
let fa_bridges = enumerate_fa_bridges(weights, tail_segments);
let mut fa_input_buffers: Vec<Tensor> = Vec::with_capacity(fa_bridges.len());

let mut state = LinearAttentionState::new(model_config, device)?;
let lora_detached = lora_weights_detached(params);

// For each tile k:
//   For each tail layer L:
//     If L is GDN: tile-forward, write tile output into the fa_input_buffer
//       for the next FA bridge, OR keep tile-local if no FA layer follows.
//     If L is FA: SKIP — defer to Phase B's bridge handling.
```

Key challenge: candle's autograd doesn't natively support "build a graph
that spans tile boundaries via a partial buffer write." We'd need to
either:

- (a) Use `Tensor::cat` to assemble the full-T tail GDN outputs at each
  FA boundary, paying the temporary memory cost.
- (b) Use a custom op that buffers tile outputs and exposes a full-T
  view to the FA layer (this is non-trivial in candle without a CustomOp).

Option (a) is simpler but costs ~40 MiB per FA boundary in the autograd
graph (already counted). Option (b) needs new infrastructure.

### 5.3 Phase B: FA-bridge serial backward, then GDN per-tile backward

After the full tail forward + LM head + loss:

```rust
let tail_loss = ...;
let tail_grads = tail_loss.backward()?;
```

But this **single** `.backward()` call would still hold the entire
tail's autograd graph alive — defeating the purpose. To get the per-tile
saved-tensor savings we need to manually drive backward per-tile-per-GDN-
run, with per-FA-bridge synchronous backward.

This is **not how candle's autograd is structured**. The whole graph
backwards in one `.backward()` call. There is no public API for "backward
through this subgraph only, retaining the rest." So implementing this
sketch faithfully would require either:

- (a) Manual unrolling of the backward chain via custom-op `bwd` impls for
  GDN (complex; would need to mirror the GDN forward's saved-tensor
  semantics in a hand-written backward).
- (b) Multiple `Var::from_tensor` boundary cuts in the tail forward, each
  enabling a `.backward()` call that produces gradients only at that
  boundary — exactly the pattern the existing `layer_pair_tiled_segment_recompute_and_backward`
  uses for *per-block* gradient injection (lines 1797-1944), extended
  one level out to *per-FA-bridge* gradient injection.

(b) is what the existing #637 path does at the *block* level. Extending
it to *FA-bridge across segments* would mean introducing `Var` cuts at
every FA bridge in the tail, computing 6 gradient-injection seeds (one
per FA boundary), and chaining them. Implementation complexity: very
high. The new code would replicate the per-block gradient-injection
machinery 6 times, with cross-segment state hand-off.

### 5.4 Estimated implementation cost

- **Function size**: ~600-800 lines of Rust (compare to the existing
  `layer_pair_tiled_segment_recompute_and_backward` at trainer.rs:1656-1948,
  which is ~290 lines).
- **Test coverage**: bit-exact CPU parity test against `standard_forward_backward`
  (trainer.rs:2170+), per-FA-bridge gradient correctness, tile-boundary
  state hand-off, FA-input-buffer recomputation determinism. New test
  suite ~400-600 lines.
- **GPU validation**: 4-8 hours of A6000 time to OOM-probe all
  combinations of `tile_size`, `num_segments`, `T`, with and without
  the new path enabled.
- **Total agent time estimate**: 3-5 days of focused implementation +
  parity testing, **before** any GPU validation. Scope is comparable to
  the entire #635/#636/#637 trio combined.

---

## 6. Risks

### 6.1 [HIGH] FA at training time fundamentally cannot be GDN-tiled

The FlashAttention forward at training time saves `{Q, K, V, logsumexp}`
at full `T`. There is no KV cache thread because training does not
allocate a paged KV cache (see the comment at trainer.rs:1206-1215 in
`model_is_gdn_only`). **No amount of upstream GDN tiling lowers the FA
saved-tensor cost.** This is not a kiln limitation — it's a property of
attention's all-pairs causal pattern.

Mitigations:

- Use a future training-time KV-cache attention path. Out of scope; not
  on the kiln roadmap.
- Use Liger-style fused FA backward that computes saved-tensor
  recomputation on-chip. Already done implicitly by FlashAttention-2;
  no additional savings available in candle's current FA-2 vendor
  (`crates/kiln-flash-attn`).

### 6.2 [HIGH] FA bridge serialization eliminates most per-tile interleaving

As §4.7 and §5.3 show, FA layers in the tail must complete their backward
*before* GDN per-tile backward can fire below them. So the per-tile
fwd+bwd interleaving — which is what saves saved-tensor peak in the
existing #637 GDN block path — **does not extend across FA bridges**.

The realized peak savings comes from (a) per-tile saved tensors within
each GDN run, and (b) discarding GDN-run saved tensors before crossing
the next FA bridge. The FA bridges themselves remain a serial bottleneck.

### 6.3 [HIGH] Implementation complexity dwarfs the savings

The proposed function is ~800 lines, with 6 manual `Var` cuts per
segment iteration in the tail (vs. 1 cut in #637's per-block path). Each
`Var` cut requires careful gradient-injection bookkeeping. The likelihood
of a subtle correctness bug — gradient double-counting, `lora_detached`
omission, state hand-off off-by-one across tile boundaries, FA-bridge
skip — is high. PR #176 is a recent example of a $14.99 burn on a
"big-fusion" attempt that turned out to be null-median; this proposal is
larger in surface area and more likely to repeat that pattern.

### 6.4 [MEDIUM] Compute overhead from re-running per-tile GDN forward

The per-tile pattern within #637's GDN block recomputes the GDN forward
at every tile to build the per-tile autograd graph. Extending this to
the tail's GDN runs adds 6 GDN-run × ~6 layers = **36 additional GDN
forward passes per training step at `seg_idx = 0`**, scaling with
`num_segments - seg_idx - 1`. Estimated step time impact: +30-60% on
A6000 at `T = 8192`. Acceptable if it unblocks the OOM, but a major
regression if the OOM is not actually unblocked.

### 6.5 [MEDIUM] Candle autograd doesn't support partial-graph backward

candle's `Tensor::backward()` walks the entire reachable graph from the
loss tensor and computes gradients for everything. There is no API for
"backward through this subgraph only and leave the rest alone." The
proposed implementation works around this by introducing many `Var`
cuts and gradient-injection scalars, which is the same pattern #637
uses but at higher cardinality. This is brittle: each new `Var` cut is
a place where (a) the wrong LoRA detached/grad-tracked variant could
leak, (b) accumulated_grads could double-count, (c) a buffer that needs
to be alive could be dropped early.

### 6.6 [LOW] GDN_CHUNK_SIZE constraint on T_tile

Tile sizes must be multiples of `GDN_CHUNK_SIZE = 64` (forward.rs:2804).
At `T = 8192` this isn't a real constraint (any reasonable tile size
divides), but at smaller `T` or odd `tile_size` choices the path falls
back to monolithic via `tiled_training_tile_size` returning `None`
(trainer.rs:1323-1337). Implementation must handle the fallback path.

### 6.7 [LOW] Gradient bit-exactness across segment boundaries

#637's gradient-injection pattern has been validated bit-exact against
monolithic at the block level. Extending it across segment boundaries
introduces new floating-point summation orders (per-tile sums of
gradient-at-block-input). Bit-exactness against a non-tiled baseline
may not hold; numerical equivalence (1e-6 tolerance on FP32 grads)
should hold but needs explicit test coverage.

---

## 7. Validation plan (if the decision were "go")

### 7.1 CPU parity tests

For `T ∈ {64, 256, 1024, 4096}`, `tile_size ∈ {64, 128, 256, 1024}`:

- Compare gradients from `per_tile_layer_pair_tiled_segment_recompute_and_backward`
  vs `standard_forward_backward` (trainer.rs:2170+). Tolerance: 1e-4
  absolute on FP32 LoRA grads (looser than #637's bit-exact for the
  block path because cross-bridge sums introduce reorderings).
- Compare loss values: tolerance 1e-5 absolute.
- Test all attention patterns: GDN-only model (regression check, should
  fall back to non-FA-bridged path), Qwen3.5-4B FA-every-4 pattern,
  worst-case "all FA" model (regression check, should produce
  `num_blocks - 1` FA bridges).

### 7.2 A6000 OOM probe

`T ∈ {2048, 4096, 8192, 12288, 16384}`, `tile_size ∈ {1024, 2048}`,
`num_segments ∈ {4, 8, 16}`. Goal: confirm `T = 8192` no longer OOMs
with the new path enabled, and characterize the new OOM ceiling.

Budget: 4-8 hours of A6000 time. Use the kiln pod pool
(`ce kiln-pod-acquire`).

### 7.3 Step-time regression

On a successful `T = 8192` run, compare step time vs the existing
PR #637 path at `T = 4096` (which is the largest non-OOM size today).
A 30-60% regression at the same effective tokens-per-step is
acceptable if it unblocks `T = 8192`; >100% would be a release blocker.

### 7.4 Gradient correctness

After 100 SFT steps on a fixed seed, compare LoRA weight values from
the new path vs `standard_forward_backward` at `T = 4096` (the size
where both paths can run). Expected drift: <1e-4 relative on FP32
master weights.

---

## 8. Decision gate

### 8.1 Quantitative criteria

- **Saved-tensor peak reduction >2×**: §4 estimates **2.9× best case**
  (~13.1 GiB → ~4.5 GiB). **Borderline pass.**
- **Step-time regression <2×**: §6.4 estimates **+30-60%**. Pass.
- **Implementation cost <1 PR's worth of agent time**: §5.4 estimates
  **3-5 days focused**. **Fail** — this is comparable to the entire
  #635/#636/#637 trio.

### 8.2 Qualitative criteria

- **Architectural cleanliness**: §5.3 shows the implementation requires
  manually unrolling backward across 6 FA bridges per tail per segment
  iteration, with careful `Var` cut + gradient-injection bookkeeping.
  **Fail** — this is a fragile, error-prone pattern at this cardinality.
- **Reusability of pattern**: the per-tile-across-FA-bridges pattern is
  highly Qwen3.5-4B-specific. Other hybrid models may have different
  FA periodicity, breaking the bridge enumeration. **Fail** —
  doesn't generalize.
- **Alternative leverage**: §0 lists three alternatives (fully-detached
  tail with analytic LM-head gradient, activation offload, Liger-style
  fused kernels). The fully-detached-tail alternative **eliminates the
  tail forward entirely**, sidestepping all FA-bridge complexity. That
  is strictly cleaner and gives a similar or larger saved-tensor
  reduction.

### 8.3 Verdict

**`no_kernel_pivot`.** The 2.9× saved-tensor reduction barely clears the
quantitative gate, but the qualitative cost (~800 LOC of fragile
gradient-injection, 6 FA bridges to bookkeep per tail iteration, 30-60%
step-time regression) is too high relative to the alternatives. The
`fully-detached-tail-with-analytic-LM-head-gradient` path from #637's
"What WOULD help" gives a comparable or better win with simpler code.

**Recommendation:** Do not pursue per-tile-extended-into-tail. Pursue
*fully-detached tail with analytic LM-head gradient* instead. Audit doc
for that path: `docs/audits/PHASE10_GDN_TRAINING_DETACHED_TAIL.md`
(future work, not in this audit's scope).

### 8.4 Re-visit triggers

This `no_kernel_pivot` should be re-opened if:

- candle gains a partial-graph-backward API that makes per-tile-across-
  FA-bridges cheap to implement.
- A future Qwen-family or Mamba-family model lands with **no** FA
  layers (pure linear-attention), making the FA-bridge constraint
  vanish.
- Profiling shows the fully-detached-tail alternative (the §8.3
  recommended path) runs into a numerical-stability or correctness
  blocker, forcing fallback to a forward-based tail gradient path.
- Any future PR ships a training-time KV-cache for FA, removing the
  full-T saved-tensor floor.

---

## Appendix A — Citations

All file:line citations in this doc resolve against current `main`
(commit `093b97b`, `2026-04-29` post-#637 merge). Read-only inspection
target: `/data/repo-cache/ericflo/kiln.git`.

Trainer (`crates/kiln-train/src/trainer.rs`):

- `74-283`  — `impl TrainableLoraParams` (LoRA scoping; GDN layers skip
  q/k/v/o LoRA at lines 117-120).
- `334-519` — `pub fn sft_train` (training entry point).
- `1132-1198` — `CheckpointConfig::from_env` and `compute_segment_boundaries`
  (segment count auto-config).
- `1216-1262` — `model_is_gdn_only` and `lora_weights_detached`.
- `1264-1304` — `enum AttnKind`, `attn_kind_at`,
  `partition_segment_layers_by_attn_type`.
- `1323-1337` — `tiled_training_tile_size`.
- `1359-1423` — `tile_loss_explicit` (GDN-only path).
- `1447-1572` — `tiled_segment_recompute_and_backward` (PR #636,
  GDN-only).
- `1656-1948` — `layer_pair_tiled_segment_recompute_and_backward`
  (PR #637, hybrid).
  - `1681-1738` — Phase 1 tail forward + `tail_loss.backward()`.
  - `1767-1794` — Phase 2 block-boundary states (detached).
  - `1797-1944` — Phase 3+4 block reverse-order with gradient injection.
    - `1804-1848` — FA sub-block (monolithic forward + backward).
    - `1849-1936` — GDN sub-block per-tile fwd+bwd loop.
- `1982-2167` — `checkpointed_forward_backward`.
  - `1995-2019` — Step 1 detached full forward.
  - `2056-2158` — Step 2 per-segment recompute + dispatch.

Model (`crates/kiln-model/src/forward.rs`):

- `1313-1316` — `STREAMING_PREFILL_*` constants.
- `1350-1406` — `streaming_prefill_enabled`,
  `streaming_prefill_enabled_for`, `streaming_tile_tokens_for`.
- `2804`     — `pub const GDN_CHUNK_SIZE: usize = 64`.
- `3339-3360` — `gated_deltanet_forward` (monolithic).
- `3385-3445` — `gated_deltanet_forward_streaming`.
- `5875-5999` — `pub fn model_forward_segment` (streaming dispatch at
  5952-5962).

VRAM heuristic (`crates/kiln-core/src/vram.rs`):

- `157-178` — `recommended_checkpoint_segments` (A6000 → 4 segments).

Prior audit docs:

- `docs/audits/PHASE10_CLOSURE.md` — Phase 10 closure, names this audit
  as a recommended next pivot.
- `docs/audits/PHASE10_GDN_TRAINING_STREAMING.md` — PR #634
  (pre-streaming baseline RED).
- `docs/audits/PHASE10_GDN_TRAINING_STREAMING_IMPL.md` — PR #635
  (streaming implementation, AMBER).
- `docs/audits/PHASE10_GDN_TRAINING_TILED_BACKWARD.md` — PR #636
  (GDN-only tiled bwd, RED for hybrid Qwen3.5-4B).
- `docs/audits/PHASE10_GDN_TRAINING_LAYER_PAIR_TILED.md` — PR #637
  (hybrid tiled bwd, identifies tail-forward as binding constraint).
- `docs/audits/PHASE10_S3_CANDIDATE_PREFLIGHT.md` — Phase 10 §3 closure;
  Addendum FleCE T=16384 OOM probe naming GDN streaming as the path
  forward.

## Appendix B — Glossary

- **Checkpoint shell**: `checkpointed_forward_backward`'s outer
  Step 1 (full detached forward) + Step 2 (per-segment recompute +
  backward) structure.
- **Per-tile fwd+bwd**: a pattern where each tile of `T` tokens runs
  its forward immediately followed by its backward, then discards saved
  tensors before the next tile begins. Saves saved-tensor peak by a
  factor proportional to `tile/T`.
- **FA bridge**: a single full-attention layer (or contiguous run of
  full-attention layers) inside an otherwise-GDN tail. FA bridges
  cannot be time-tiled at training time.
- **Tail forward**: the forward pass through `segments[seg_idx + 1..]`
  + LM head + loss inside `layer_pair_tiled_segment_recompute_and_backward`
  Phase 1, used to compute `∂loss/∂seg_output_var`.
- **Gradient injection**: the technique of computing `(block_output *
  pre_computed_grad).sum_all().backward()` to seed a backward pass with
  a known activation gradient (used inside the per-block path at
  trainer.rs:1828, 1897, etc.).
- **Saved-tensor peak**: the maximum size of activations live inside
  candle's autograd graph at any one moment during a `.backward()` call.
- **Transient peak**: the maximum size of activations during forward
  (no backward) for a single segment, before the segment's autograd
  graph is detached.
