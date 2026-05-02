# Phase 10 — Fully-detached tail SFT memory path

Date: 2026-05-02
Status: **GREEN for a bounded implementation PR, with a strict exact-gradient gate** — compute the LM-head/final-RMSNorm loss gradient analytically, then propagate it through later transformer blocks with the existing layer-pair gradient-injection pattern instead of retaining the whole tail forward graph. Do not implement a hand-written backward for every Qwen3.5-4B op unless the blockwise VJP path fails parity.
Scope: Doc-only audit. No GPU pod, no Rust/CUDA changes.

## 0. TL;DR

- The current hybrid tiled path still OOMs at `T=8192` on A6000 because every segment iteration first builds a grad-tracked tail graph through later segments and the head. The prior layer-pair audit measured the peak at **48 648 MiB / 49 140 MiB** and pinned the dominant cost to the tail forward's full-`T` saved tensors.
- The next viable path is not another GDN tile size tweak. It is to remove the tail graph root: compute `d loss / d final_hidden` analytically from final RMSNorm + LM head + cross-entropy, then walk later segments backward one block at a time with gradient injection.
- This keeps exact gradients while making the tail peak look like the existing per-block path: one FA block or one GDN tile graph live at a time, plus detached boundary/gradient tensors. It plausibly clears the A6000 OOM headroom for `T=8192`.
- Implementation complexity is medium-high but bounded: mostly trainer-side refactor around `layer_pair_tiled_segment_recompute_and_backward`; no new CUDA kernel is required for the first attempt.
- Verdict: **go**, behind `KILN_DETACHED_TAIL_SFT=1` initially, with CPU parity before any A6000 probe.

## 1. Current path and binding constraint

`crates/kiln-train/src/trainer.rs` routes SFT through `checkpointed_forward_backward` whenever gradient-checkpoint segments are configured (`crates/kiln-train/src/trainer.rs:421`). The checkpointed path builds detached segment boundaries first (`crates/kiln-train/src/trainer.rs:1994`), decides whether the tiled route applies (`crates/kiln-train/src/trainer.rs:2038`), and sends Qwen3.5-4B hybrid models to `layer_pair_tiled_segment_recompute_and_backward` (`crates/kiln-train/src/trainer.rs:2058`).

Inside the layer-pair path, Step 1 wraps the segment output in a `Var` (`crates/kiln-train/src/trainer.rs:1681`), forwards through all later segments with detached LoRA (`crates/kiln-train/src/trainer.rs:1691`), then runs FLCE or the unfused head loss (`crates/kiln-train/src/trainer.rs:1719`). The only useful result is `grad_at_seg_output`, extracted from `tail_loss.backward()` (`crates/kiln-train/src/trainer.rs:1736`). The graph is explicitly dropped afterwards (`crates/kiln-train/src/trainer.rs:1755`), but that is too late for peak VRAM.

The previous audit states the failure directly: at `seg_idx = 0`, the tail spans 24 later layers plus the head; LoRA Vars are detached, but per-layer activations still trace back to `seg_output_var`, so full-`T` saved tensors exceed the A6000 ceiling at `T=8192` with `KILN_USE_FLCE=1` (`docs/audits/PHASE10_GDN_TRAINING_LAYER_PAIR_TILED.md:219`). It also names the exact lever: switch the tail to a fully-detached forward path and replace `grad_at_seg_output` extraction with an analytic gradient at the LM-head input (`docs/audits/PHASE10_GDN_TRAINING_LAYER_PAIR_TILED.md:248`). The follow-up per-tile audit rejected deeper tail tiling because its 2.9× best-case memory reduction carried too much FA-bridge bookkeeping, and recommended this detached-tail audit instead (`docs/audits/PHASE10_GDN_TRAINING_PER_TILE_FWDBWD.md:1188`).

## 2. Analytic LM-head/final-norm gradient

The required analytic seed is exact and small compared with the transformer tail. Current FLCE routes call `model_forward_final_norm` before `fused_linear_cross_entropy_dispatch` (`crates/kiln-train/src/trainer.rs:2137`), while the unfused path calls `model_forward_head` and then `cross_entropy_loss` (`crates/kiln-train/src/trainer.rs:2149`). Model code confirms `model_forward_head` is exactly final RMSNorm plus `lm_head_forward` (`crates/kiln-model/src/forward.rs:6017`), and `model_forward_final_norm` is just the RMSNorm half (`crates/kiln-model/src/forward.rs:6033`).

For SFT next-token loss over active positions, compute this without retaining a candle graph:

1. Run the full boundary forward with detached LoRA and no graph. Keep `h_final = boundary_states[num_segments]`, shape `[1, T, H]`.
2. Compute final RMSNorm values `y = rms_norm(h_final, final_norm_weight, eps)` and a chunked LM-head softmax against `weights.embed_tokens_t`, matching FLCE chunking so `[T, V]` logits are never materialized.
3. For each active shifted label position `t`, form `g_z[t, v] = (softmax(z[t])[v] - 1[v == label[t]]) / N_active`. Inactive rows are zero.
4. Accumulate `g_y += g_z @ W^T` chunk-by-chunk, where `W` is `embed_tokens_t`. This is `[1, T, H]` and can be accumulated in F32 then cast to the hidden dtype if parity allows; keep F32 for the first implementation.
5. Backprop final RMSNorm analytically. For each row, with `inv = rsqrt(mean(h^2) + eps)`, `u = g_y * final_norm_weight`, and hidden size `H`: `g_h = inv * u - h * (inv^3 / H) * sum(u * h)`. The final-norm weight is frozen for LoRA SFT, so only `g_h = d loss / d h_final` is needed.

This seed replaces the current `tail_loss.backward()` root. It also keeps FLCE's memory contract: FLCE already exists to avoid the `[1, T, V]` logits tensor, and model docs call out that the logits materialization dominates long-context head memory (`crates/kiln-model/src/forward.rs:6042`). The LoRA precision closure is a useful warning here: the earlier ~42% FP32 SGEMM bucket may include LM-head/embedding work, not just LoRA matmuls (`docs/audits/PHASE10_LORA_PRECISION_STUDY.md:311`), so the first implementation should validate memory and correctness, not promise a step-time win.

## 3. Propagating through a fully-detached tail

A pure analytic backward for every later transformer op would be too large. The bounded path is a semi-analytic VJP:

- Use the analytic head gradient only to seed `grad_at_final_hidden`.
- For each segment iteration, start from `grad_at_current_output = grad_at_final_hidden` and walk later segments from last to `seg_idx + 1`.
- Within each later segment, reuse the existing block partitioning invariant: FA blocks are one layer wide and monolithic; GDN blocks are time-tiled. The current code already does this for the current segment in `layer_pair_tiled_segment_recompute_and_backward` (`crates/kiln-train/src/trainer.rs:1799`).
- Recompute each later block from detached block-boundary values with detached LoRA, form `scalar = (block_output * grad_at_current_output).sum_all()`, call `backward()`, extract only the input gradient, detach it, and immediately drop the block graph. This is the same gradient-injection pattern used for FA blocks (`crates/kiln-train/src/trainer.rs:1804`) and GDN tiles (`crates/kiln-train/src/trainer.rs:1849`), except `accumulate_grads` is deliberately skipped for tail blocks.
- Once the walk reaches `boundary_states[seg_idx + 1]`, the resulting tensor is the exact `grad_at_seg_output` seed needed by the existing current-segment block backward.

This is still exact if the recomputed detached block inputs match the original forward values. It does not ignore the Jacobian of later layers; it merely computes each VJP in isolation rather than keeping one tail-wide graph alive.

The important distinction from the rejected per-tile-across-tail audit is lifetime. That audit tried to tile the tail while preserving a graph across FA bridges and found the qualitative complexity too high (`docs/audits/PHASE10_GDN_TRAINING_PER_TILE_FWDBWD.md:1190`). Here, each block VJP has one scalar root, one block graph, one input gradient, and then drops. FA bridges remain monolithic, but only one FA layer's saved tensors are live at a time.

## 4. Live tensors at `T=8192` on A6000

Assume Qwen3.5-4B dimensions from current training profiles: hidden `H=2560`, rank-8 LoRA, 4 checkpoint segments, 8 layers per segment, and `T=8192`. One BF16 hidden/gradient tensor `[1, 8192, 2560]` is about **40 MiB**.

With the detached-tail path, the persistent training-step live set should be:

| Tensor group | Approx live VRAM | Notes |
| --- | ---: | --- |
| Base model + LoRA + optimizer state | unchanged | Same as current SFT; not addressed by this audit. |
| `boundary_states` for 5 segment boundaries | ~200 MiB | Stored detached by `checkpointed_forward_backward` today (`crates/kiln-train/src/trainer.rs:1998`). |
| Current segment `block_boundaries` | ~200 MiB worst-case | Five detached block boundaries for the 4-block Qwen segment layout described in the per-tile audit (`docs/audits/PHASE10_GDN_TRAINING_PER_TILE_FWDBWD.md:327`). Can be built per segment and dropped. |
| Tail block input/output/gradient buffers | ~80-200 MiB plus op scratch | One block/tile at a time; FA one-layer block is full `T`, GDN block is `tile_size`. |
| Analytic head gradient buffers | ~80-160 MiB plus LM-head chunk scratch | `g_y` and `g_h`; chunked vocab softmax avoids `[T, V]`. |
| Autograd saved tensors | one block/tile only | Replaces the current 24-layer tail graph at `seg_idx=0`. |

The path should eliminate the dominant saved-tensor group identified by PR #637: saved activations from three later 8-layer segments plus the head. What remains is comparable to the current Step 2/3 per-block work, which the prior audit called sub-dominant (`docs/audits/PHASE10_GDN_TRAINING_LAYER_PAIR_TILED.md:232`).

## 5. Does this plausibly clear the OOM headroom?

**Yes, plausibly.** The current measured failure is near the A6000 ceiling: 48 648 MiB / 49 140 MiB (`docs/audits/PHASE10_GDN_TRAINING_LAYER_PAIR_TILED.md:31`). But the culprit is not a few hundred MiB; the same doc says the tail saved-tensor peak is already **≥30 GiB** before per-block work begins (`docs/audits/PHASE10_GDN_TRAINING_LAYER_PAIR_TILED.md:35`). Removing that tail-wide graph and replacing it with a one-block graph should create multi-GiB headroom even after adding analytic head-gradient buffers.

The caveat is Step 1. Current boundary construction uses `params.as_lora_weights()` (`crates/kiln-train/src/trainer.rs:1996`) and detaches only each segment output (`crates/kiln-train/src/trainer.rs:2016`). The per-tile audit reasoned that the graph becomes unreachable after the detach, but PR #637 still called Step 1 a possible peak contributor (`docs/audits/PHASE10_GDN_TRAINING_LAYER_PAIR_TILED.md:238`). The implementation must switch Step 1 to `lora_weights_detached(params)` or an explicit no-grad forward helper too. Otherwise the tail graph may be fixed while Step 1 still pays avoidable graph construction/transient costs.

Decision gate for memory: `T=8192`, A6000, `KILN_USE_FLCE=1`, `KILN_STREAMING_PREFILL=1`, rank-8 LoRA must complete one SFT step below **45 GiB peak**. This gives at least ~4 GiB slack under the 49 140 MiB device ceiling and avoids declaring victory on a run that survives only by allocator luck.

## 6. Implementation complexity

Estimated shape: one focused implementation PR, roughly **300-600 LOC** if it reuses existing trainer primitives.

Required pieces:

1. `analytic_lm_head_final_norm_grad(...) -> (loss_value, grad_final_hidden)`, sharing FLCE chunk size and shifted-label semantics with `cross_entropy_loss` (`crates/kiln-train/src/trainer.rs:995`) and `tile_loss_explicit` (`crates/kiln-train/src/trainer.rs:1359`).
2. `tail_vjp_detached(...) -> grad_at_seg_output`, walking later segments backwards with detached LoRA and block/tile VJPs.
3. A small refactor that lets the current segment and tail segment walkers share block-boundary construction and GDN tile VJP code.
4. Feature flag: `KILN_DETACHED_TAIL_SFT=1`, default off until CPU parity and A6000 memory pass.
5. Step 1 no-grad cleanup: use detached LoRA for boundary-value forwards.

Do not start with custom CUDA or a hand-written backward for GDN/FlashAttention. The current `model_forward_segment` already dispatches training-time streaming for GDN and keeps FA monolithic because training has no KV cache (`crates/kiln-model/src/forward.rs:5895`). The first implementation should exploit that existing behavior.

## 7. Correctness risks

- **Shifted-label mismatch.** The analytic loss must exactly match next-token indexing in `cross_entropy_loss`, where logits `0..seq_len-1` predict labels `1..seq_len` (`crates/kiln-train/src/trainer.rs:1006`). A one-position slip will pass memory and fail training.
- **FLCE parity mismatch.** The chunked analytic softmax must match `fused_linear_cross_entropy_dispatch` numerically closely enough. Use F32 accumulation first; optimize later only after parity.
- **Final RMSNorm backward.** The rowwise `1/H` factor and final-norm weight multiplication are easy to get wrong. Include a tiny CPU finite-difference/unit comparison against candle autograd before model-scale tests.
- **Tail VJP lifetime bugs.** Accidentally retaining `tile_grads`, `block_grads`, or block outputs across loop iterations would recreate the saved-tensor peak. Explicit `drop(...)` calls are justified in this path.
- **Double-counted LoRA gradients.** Tail blocks must use detached LoRA and must not call `accumulate_grads`; current-segment blocks remain responsible for trainable LoRA gradients, matching the existing comment at `crates/kiln-train/src/trainer.rs:1683`.
- **FA full-sequence floor.** Each FA block still saves one layer's full-`T` FlashAttention tensors because training has no KV cache (`crates/kiln-model/src/forward.rs:5923`). The memory math says this is acceptable, but it is the residual floor to measure.

## 8. Validation plan

### 8.1 Doc-only validation for this PR

Run locally only:

```bash
rg -n 'fully-detached tail|analytic LM-head gradient|T=8192|A6000|go/no-go|Validation plan' docs/audits/PHASE10_GDN_TRAINING_DETACHED_TAIL.md
rg -n 'PHASE10_GDN_TRAINING_DETACHED_TAIL' docs/audits/README.md docs/audits/PHASE10_GDN_TRAINING_DETACHED_TAIL.md || true
git diff --check
```

### 8.2 Implementation validation before enabling by default

1. **Analytic head unit tests, CPU:** compare `analytic_lm_head_final_norm_grad` against candle autograd for small `(T, H, V)` tensors. Loss tolerance `1e-6`, gradient tolerance `1e-5` F32.
2. **Trainer parity, CPU mini-model:** compare `standard_forward_backward` (`crates/kiln-train/src/trainer.rs:2170`) with detached-tail checkpointing for GDN-only, FA-only, and Qwen-pattern hybrid attention. Loss tolerance `1e-5`; LoRA gradient tolerance `1e-4` absolute.
3. **GPU non-OOM probe:** Run A6000 `T ∈ {4096, 8192, 12288}`, `tile_size ∈ {1024, 2048}`, rank-8 LoRA, `KILN_USE_FLCE=1`, `KILN_STREAMING_PREFILL=1`, `KILN_DETACHED_TAIL_SFT=1`. Acceptance: `T=8192` completes under 45 GiB peak.
4. **Gradient parity at feasible length:** At `T=4096`, compare 10 fixed-seed SFT steps old path vs detached-tail path. Final loss relative drift ≤0.5%; per-step max drift ≤1.5%, matching the LoRA precision study's successful parity gates (`docs/audits/PHASE10_LORA_PRECISION_STUDY.md:295`).
5. **Lifetime regression check:** Add memory trace markers around analytic head, tail VJP, current-segment VJP. Confirm no tail-wide saved-tensor plateau remains before per-block work.
6. **Default-off soak:** Land behind env flag; flip default only after an A6000 run proves `T=8192` and no correctness drift.

GPU validation must use the project-standard RunPod kiln image and hard wall-clock budget, but this audit PR intentionally does not launch a pod.

## 9. Go/no-go verdict

**Go for implementation, bounded to the semi-analytic detached-tail design.**

This is the first post-#637 path whose memory lifetime attacks the actual binding constraint while reusing the existing layer-pair tiled infrastructure. It is not a speculative kernel port, and it does not depend on LoRA precision speedup assumptions that the A6000 null-result already rejected (`docs/audits/PHASE10_LORA_PRECISION_STUDY.md:316`).

Strict stop conditions for the implementation PR:

- If CPU parity against candle autograd fails for the analytic head/final-norm gradient, stop before GPU.
- If CPU mini-model trainer parity fails beyond the tolerances in §8.2, stop before GPU.
- If the only way to pass parity is hand-writing full backward formulas for GDN recurrent, FlashAttention, MLP, and RMSNorm, reclassify as **no-go** and prefer activation offload or fused-kernel saved-tensor reduction instead.
- If A6000 `T=8192` still peaks above 45 GiB after Step 1 no-grad cleanup and detached-tail VJP, stop and write a null-result closure.

## 10. References

- PR #634 — `docs/audits/PHASE10_GDN_TRAINING_STREAMING.md`, original training-time streaming audit.
- PR #635 — `docs/audits/PHASE10_GDN_TRAINING_STREAMING_IMPL.md`, model-forward streaming dispatch.
- PR #636 — `docs/audits/PHASE10_GDN_TRAINING_TILED_BACKWARD.md`, GDN-only tiled backward.
- PR #637 — `docs/audits/PHASE10_GDN_TRAINING_LAYER_PAIR_TILED.md`, hybrid layer-pair tiled path and detached-tail recommendation.
- PR #681/#682 — `docs/audits/PHASE10_LORA_PRECISION_STUDY.md`, LoRA precision null-result and A6000 parity gates.
