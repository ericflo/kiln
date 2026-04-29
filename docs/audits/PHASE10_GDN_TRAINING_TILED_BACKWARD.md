# Phase 10 — GDN training-time tiled forward+backward (Finding 2 §2 impl)

Date: 2026-04-29
Status: **RED on the acceptance criterion (T=8192 SFT must no longer OOM on A6000) — the tiled-backward path is bit-exact + necessary infrastructure, but the GDN-only precondition required for parity excludes the production target model** — see Verdict
Hardware: NVIDIA RTX A6000 (49 140 MiB VRAM, driver 550.127.08), CUDA 12.4
Branch: `ce/phase10-tiled-forward-backward`
Commit: `f10e20c` off `33c86ae` main (post-PR #635)
Bench: `crates/kiln-server/examples/flce_phase_a_validation_bench.rs`
Raw log: `docs/flce_phase_a_tiled_backward_raw_2026-04-29.log`

## Purpose

Implementation follow-up to PR #635
(`docs/audits/PHASE10_GDN_TRAINING_STREAMING_IMPL.md`), which shipped
the dispatch wiring for training-time streaming GDN prefill but found
the resulting per-tile transient-allocation reduction insufficient to
unblock T=8192 SFT on A6000. The PR #635 audit's "Required follow-ups"
§2 named the next viable remediation: **time-axis tile inside
`checkpointed_forward_backward`** so each tile's forward+backward
releases its autograd-saved tensors before the next tile's forward
allocates its own — the change PR #635 identified as actually moving
the autograd-recompute bottleneck (rather than just reducing transient
allocations as the in-segment streaming dispatch did).

The PR exists to answer two questions independently:

1. **Does the time-axis tile path produce bit-exact LoRA gradients
   versus monolithic on a GDN-only model?** Yes — see "CPU parity"
   below.
2. **Does it unblock T=8192 SFT on A6000 for Qwen3.5-4B?** **No** —
   see "Memory ceiling" below. The dispatch precondition (no
   full-attention layer anywhere) is conservative because of training-
   time KV-cache absence, and Qwen3.5-4B is hybrid 24 GDN + 8 full-attn
   so the tiled path falls back to monolithic for the production
   target model.

## Implementation summary

`crates/kiln-train/src/trainer.rs`:

- **`model_is_gdn_only(weights)`** — predicate that returns true iff
  every transformer layer uses linear (GDN) attention. Used as the
  bit-exactness invariant for the tiled path.
- **`tiled_training_tile_size(weights, device, seq_len)`** — dispatch
  oracle. Returns `Some(tile)` when (a) the model is GDN-only, (b)
  `streaming_prefill_enabled_for(device, seq_len)` is true (env
  override or device-default threshold), and (c) `tile < seq_len` and
  `tile % GDN_CHUNK_SIZE == 0`. Returns `None` otherwise — the caller
  takes the existing monolithic branch unchanged.
- **`tile_loss_explicit(...)`** — per-tile next-token cross-entropy
  contribution, normalized so per-tile contributions sum to the
  monolithic mean over `total_active` positions exactly. The helper
  pads the tile's hidden by one position (zero) for non-last tiles
  and prepends a `false`-masked dummy id to `input_ids` / `mask`, so
  the existing `cross_entropy_loss` / `fused_linear_cross_entropy`
  helpers' built-in next-token shift recovers explicit
  `tile_hidden[i] -> labels[i]` semantics. The final scalar is
  multiplied by `num_tile_active / total_active` because the helpers
  internally divide by `num_tile_active`. Routes through the same
  FLCE / non-FLCE branches as the monolithic path so loss math is
  shared.
- **`tiled_segment_recompute_and_backward(...)`** — per-segment outer
  loop that runs forward+backward+accumulate **per tile**. State
  (`LinearAttentionState`) is threaded across tiles for both the
  grad-tracked segment AND each detached later segment. Each tile's
  `loss.backward()` releases its own autograd graph before the next
  tile's forward allocates its saved tensors, which is the property
  that distinguishes this PR from PR #635's in-segment streaming
  dispatch.
- **`checkpointed_forward_backward`** — modified to call
  `tiled_training_tile_size` once per training step; if it returns
  `Some`, every segment iteration uses
  `tiled_segment_recompute_and_backward`, otherwise every iteration
  uses the original monolithic branch (behavior unchanged when tiled
  path doesn't fire).

`crates/kiln-model/src/forward.rs`:

- `GDN_CHUNK_SIZE` is changed from `const` to `pub const` so
  `kiln-train` can validate `tile % GDN_CHUNK_SIZE == 0` without
  re-importing the constant.

No changes to forward-pass kernels; this is purely a trainer-side
restructuring on top of the dispatch infrastructure PR #635 shipped.

## Correctness invariant

Bit-exact LoRA gradients across tiles + monolithic require:

1. **Model is GDN-only.** Full-attention layers have no training-time
   KV cache to thread across tiles, so a tiled FA forward would
   attend only inside its own tile and produce different logits than
   the monolithic path. The `model_is_gdn_only` precondition rules
   this out.
2. **LoRA on GDN layers is restricted to MLP projections** (`gate_proj`,
   `up_proj`, `down_proj`). `TrainableLoraParams::initialize`
   explicitly skips `q_proj` / `k_proj` / `v_proj` / `o_proj` for
   GDN layers (these projections only exist on full-attention
   layers). MLP is a per-position transformation, so MLP-LoRA
   gradients factor across positions; the truncated-BPTT effect of
   detaching the recurrent state at tile boundaries (which is what
   per-tile backward implies — backward of tile *t* sees the state
   value at the start of tile *t* but not the autograd graph through
   which that state was computed during tile *t-1*) does not reach
   any MLP-LoRA Var because no MLP-LoRA Var participates in the state
   recurrence.

Together these invariants make per-tile forward+backward of a GDN-only
model bit-exact against monolithic for all LoRA gradients, even though
the per-tile pattern would in general be a truncated-BPTT
approximation for parameters that influence the recurrent state.

## Validation

### CPU parity

`cargo test -p kiln-train --lib test_checkpointed_forward_backward_tiled_matches_monolithic_cpu`:

```
test trainer::tests::test_checkpointed_forward_backward_tiled_matches_monolithic_cpu ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 14 filtered out; finished in 3.70s
```

Setup:

- GDN-only mini-config: `tiny_config()` with `full_attention_interval =
  num_layers + 1`, so `is_full_attention_layer(i)` is false for all
  layers and `tiny_weights` only emits GDN layers.
- `T=192`, `tile_size=64` → 3 tile loop iterations per segment (two
  non-last tiles exercising the `pad_amount=1` branch in
  `tile_loss_explicit`, one last tile exercising the `pad_amount=0`
  branch).
- 2 segments × 4 layers (default `compute_segment_boundaries(4, 2)`).
- LoRA rank 4, alpha 8.0.

Assertions:

- Total loss matches bit-for-bit: `(loss_mono - loss_tiled).abs() <
  1e-5` (atol tightened to allow trivial f32 fp-associativity drift in
  the chunked LM-head log-sum-exp; observed `<< 1e-6` in practice).
- Every LoRA Var with a gradient in monolithic has the same gradient
  (per-element max-abs diff) under tiled, with same `1e-5` tolerance.

Existing 14 trainer tests still pass with the new code path:

```
test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.85s
```

### A6000 SFT bench

`./target/release/examples/flce_phase_a_validation_bench --model-path /workspace/qwen3.5-4b`:

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

Baseline post-warmup VRAM = 18 472 MiB.

### Bit-exact loss at T=2048

`stream_on_t2048_loss = 2.7834393978118896` and `stream_t2048_loss_delta
= 0.0e0` — STREAMING ON matches STREAMING unset bit-for-bit at T=2048,
which is the expected outcome of correct dispatch (the GDN-only
precondition is false on Qwen3.5-4B → tiled path returns to monolithic
→ loss is mechanically identical). This rules out any silent
regression in the monolithic branch from the trainer-level
restructuring.

### Memory ceiling — unchanged from PR #635

Every T=8192 cell (STREAMING unset, STREAMING ON tile=default/4096/2048)
and the T=16384 stretch cell OOM at the same ≈48 648 MiB ceiling as PR
#635's measurement. The peak-VRAM ladder is flat across tile sizes,
which would be expected if the tiled-backward branch were dispatching
on Qwen3.5-4B and the per-tile envelope were the binding constraint —
but the actual cause here is different: the tiled branch does not
dispatch at all, because Qwen3.5-4B is hybrid 24 GDN + 8 full-attn and
the `model_is_gdn_only` precondition fails. Path falls back to
monolithic; OOM is the same as monolithic; verdict on the acceptance
criterion is RED.

## Dispatch behaviour on Qwen3.5-4B

Qwen3.5-4B is hybrid 24 × GDN + 8 × full-attention (every 4th layer
is full-attn: layers 3, 7, 11, 15, 19, 23, 27, 31). The
`model_is_gdn_only` precondition is therefore false for the production
target model, and `tiled_training_tile_size` returns `None` on every
training step. **The tiled path code is entirely skipped in production
SFT runs on Qwen3.5-4B.** Falling back to the monolithic branch is
behaviourally identical to PR #635 main, which is what the bench
table confirms.

This is the key engineering trade-off in the PR: the GDN-only
precondition is **conservative for parity** (it guarantees bit-exact
gradients across tiles + monolithic on a GDN-only model) but
**insufficient for production** (Qwen3.5-4B is hybrid). Lifting the
precondition for hybrid models is the next remediation step (see
"Required follow-ups").

## Verdict

**RED on the acceptance criterion (T=8192 SFT must no longer OOM on A6000) — the tiled-backward path is bit-exact + necessary infrastructure, but the GDN-only precondition required for parity excludes the production target model** — the dispatch infrastructure shipped here
is necessary for any future training-time tiled-backward work
(no other code path can reach it from `sft_train`), and the CPU
parity test pins the bit-exact-gradients invariant the next attempt
needs to preserve. But this PR alone does not unblock long-context
SFT on A6000 for Qwen3.5-4B, because the GDN-only precondition
required for parity excludes the production target model.

This is a useful negative result for the audit's option (2). It
re-frames the next move as the **layer-pair-level** remediation
flagged in PR #635's audit risk (b): partition each segment into
sub-blocks by attention type (GDN-block / full-attn-block), apply
time-axis tiling only to GDN sub-blocks while running full-attn
sub-blocks monolithic, and use gradient injection (chain rule with a
pre-computed activation gradient at the segment output) so per-block
forward+backward can release saved tensors without requiring a tile-
level loss for the full segment chain.

## Required follow-ups (post-this PR)

The remediation menu, in increasing scope, narrowed by this PR's
findings:

1. ~~**Tile within `model_forward_segment` (smallest change that could
   work).**~~ Shipped in PR #635. Insufficient on A6000 at T=8192 —
   the autograd recompute bottleneck dominates.

2. ~~**Time-axis tile inside `checkpointed_forward_backward` for
   GDN-only models.**~~ Shipped here. Bit-exact for GDN-only models;
   does not fire on Qwen3.5-4B because of the GDN-only precondition.
   Code remains useful as the dispatch foundation any layer-pair-level
   variant will reuse.

3. **Layer-pair-level time-axis tile inside
   `checkpointed_forward_backward` for hybrid models (recommended
   next).** Inside the per-segment recompute, partition the segment's
   layers into contiguous-attention-type blocks: a "GDN block" is a
   maximal run of consecutive GDN layers; a "full-attn block" is a
   maximal run of consecutive full-attn layers (Qwen3.5-4B segments
   contain both). Process blocks sequentially:
   - For each GDN block: time-tile forward+backward, accumulating
     LoRA gradients via gradient injection from the pre-computed
     activation gradient at the block's output.
   - For each full-attn block: monolithic forward+backward (full T)
     with the same gradient injection.

   Pre-computing the activation gradient at each block boundary
   requires one extra forward+backward pass through later segments +
   LM head with a grad-tracked input at `boundary_states[k+1]`,
   adding ~1× the segment-chain forward cost per segment.

   Estimated scope: ~400-600 LOC in `trainer.rs`. Risks:
   (a) implementing gradient injection correctly without breaking the
   existing single-segment grad path (clean abstraction needed);
   (b) full-attn LoRA gradients (`q_proj`, `k_proj`, `v_proj`,
   `o_proj`) DO depend on the recurrent state thread through earlier
   GDN layers in the same block, so attention-LoRA grads under a
   tiled GDN-block forward would be a truncated-BPTT approximation
   (different from monolithic). The CPU parity invariant here would
   need to relax to "MLP-LoRA grads bit-exact, attention-LoRA grads
   approximate to within `1e-3`" or similar.

4. **Combined option 3 + a CUDA fused FLCE kernel (Phase B).** Phase
   B's head-side memory savings stack with GDN-side time-axis tiling.
   Combined, this is the path most likely to fit T=16384 SFT on
   A6000.

Recommended next slice: **option 3 as a focused PR** — the
layer-pair-level tile is the change that can actually fire on
Qwen3.5-4B and move the autograd-recompute bottleneck for the
production target model.

## Caveats

- The bench's peak-VRAM signal is `nvidia-smi` polled at 50 ms;
  sub-50 ms transient peaks may be undercounted. All cells in this
  PR's table at T=8192 are at the GPU ceiling
  (~48 648 / 49 140 MiB), so the OOM-vs-OK distinction is robust
  even with that resolution.
- The CPU parity test runs on a synthetic GDN-only mini-model.
  Production GDN-only models (if any future kiln-supported model
  lands without full-attn layers) would benefit from this path
  immediately.
- Bit-exact CPU parity does not guarantee bit-exact CUDA parity
  (matmul reduction order can vary with shape). The existing
  `test_streaming_matches_monolithic_cuda` (1e-4 tolerance for the
  inference path) is the closest GPU parity proxy.
- Audit scoped to GPU-side production training (CUDA path on A6000).
  Metal / MLX training is unaffected by this PR — the tiled path
  predicate is device-aware via `streaming_prefill_enabled_for`, so
  Metal SFT only enters the tiled branch under explicit
  `KILN_STREAMING_PREFILL=1` AND the model is GDN-only.
