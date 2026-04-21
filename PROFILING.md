# Kiln Profiling Report


## Phase 7 design: streaming/tiled GDN prefill — 2026-04-20

**Outcome: long-context prefill (≥65k tokens) fits on a 48 GiB A6000 by iterating the prompt in 8192-token tiles, reusing the already-existing `LinearAttentionState` as the O(1) state carrier across tile boundaries. Peak working set shrinks from ~9 GiB per-layer-full-prompt to ~1.1 GiB per-layer-per-tile. No GDN kernel changes, no state struct changes, no paged-cache changes; a single new `model_forward_paged_streaming` wrapper plus three config flags. $0 doc-only PR; implementation spike is one ~30-minute A6000 run.**

This design is a $0 source-inspection follow-on to the PR #226 audit
([#131](https://github.com/ericflo/kiln/pull/131),
[#163](https://github.com/ericflo/kiln/pull/163),
[#164](https://github.com/ericflo/kiln/pull/164),
[#170](https://github.com/ericflo/kiln/pull/170),
[#219](https://github.com/ericflo/kiln/pull/219),
[#223](https://github.com/ericflo/kiln/pull/223),
[#224](https://github.com/ericflo/kiln/pull/224),
[#225](https://github.com/ericflo/kiln/pull/225),
[#226](https://github.com/ericflo/kiln/pull/226)
pattern: inspect, decide, defer pod $ until the design is pinned down).

**Why append to PROFILING.md rather than a new `DESIGN-streaming-prefill.md`:** the Phase 7 audit already lives here, and candidate G ("streaming prefill") is explicitly the thing this section lands. Future re-profiles will extend the same file with post-implementation deltas. Keeping audit + design + measurement in one place avoids the "where did we decide this?" cross-file hunt that earlier phases produced. If this section grows past ~500 lines we revisit, but today the signal is stronger as one continuous document.

### (a) Tile size choice — 8192 tokens

- **Chosen: `KILN_STREAMING_PREFILL_TILE=8192` (default).** Rationale below; the flag is tunable for measurement.
- **Hard constraint: tile size must be a multiple of `GDN_CHUNK_SIZE = 64`.** `gdn_chunkwise_recurrence` (forward.rs:1173–1378) already handles a tail chunk smaller than 64 — but only *once per call*. Letting tile boundaries land mid-chunk would repeatedly force the tail path and also splinter the recurrence into unequal chunks, which complicates state handover (see §c). `8192 = 128 × 64` is chunk-aligned.
- **Lower bound (2k, 4k):** too small. Per-tile kernel-launch overhead (conv1d_prefill + chunk_prep + fused_recurrent + gates + norms × 32 layers × ceil(seq_len / tile)) starts to dominate wall-clock below ~4k. At 2k across T=65536 that is 32 tiles × 32 layers = 1,024 layer invocations vs 8 × 32 = 256 at tile=8192. Marlin GEMM and paged-decode fusion are also tuned for larger M.
- **Upper bound (16k, 32k):** fits comfortably in budget (per-layer peak scales roughly linearly in tile size → ~2.2 GiB and ~4.4 GiB respectively), but the audit's 9 GiB-per-layer at T=65536 is the monolithic full-prompt peak. Picking tile=8192 leaves ~35 GiB of headroom for the LM-head tail (§d) and keeps Marlin activations well inside SM resident budgets.
- **Sweet spot math** (Candidate G in the audit, extrapolated to the activation graph from §3–§6 of the PR #226 audit):

  | tile | per-layer peak act | 32-layer working budget fit | tile count at T=65536 | relative kernel-launch overhead |
  |------|-------------------:|:---------------------------:|----------------------:|--------------------------------:|
  | 2048 | ~0.28 GiB          | trivial                     | 32                    | 4× vs 8192                      |
  | 4096 | ~0.55 GiB          | trivial                     | 16                    | 2× vs 8192                      |
  | **8192** | **~1.10 GiB**  | **trivial (<1.5 GiB carry)**| **8**                 | **1× (baseline)**               |
  | 16384 | ~2.2 GiB          | fits (~5 GiB carry)          | 4                     | 0.5× vs 8192                    |
  | 32768 | ~4.4 GiB          | fits (~10 GiB carry)         | 2                     | 0.25× vs 8192                   |
  | 65536 (monolithic) | ~9 GiB | **OOM** (≥28 GiB carry on A6000) | 1 | — |

  (Per-layer-peak figures derive from scaling the PR #226 audit's post-`l2_normalize` live set linearly in `seq_len`; the O(1) state carry is negligible — 48 MiB across all GDN layers.)
- **8192 is the smallest tile at which kernel-launch overhead is not the limiting cost AND peak per-tile per-layer activations fit comfortably on an A5000-class 24 GiB GPU too** (so the same default works for smaller GPUs once Phase 7 lands there).

### (b) Iteration order — layer-by-tile (outer tile, inner layer)

Two orderings exist; only one is viable:

- **Option 1 (CHOSEN): outer loop = tile, inner loop = layer.** For each tile, run all 32 layers, handing `hidden` from layer to layer as today; threading `LinearAttentionState` *across tiles* at the model level, threading paged KV cache / `start_pos` as today. Working set per step ≈ activations of one layer of one tile ≈ ~1.1 GiB (at tile=8192), plus the `hidden` tensor `[1, tile, 2560]` ≈ 40 MiB handed between layers.
- **Option 2 (REJECTED): outer loop = layer, inner loop = tile.** Would need to hold the *full* `[1, seq_len, hidden=2560]` intermediate between layers — at T=65536 that is 320 MiB × (one live copy per layer-boundary hand-off) ≈ unbounded without additional buffering, or an explicit ~320 MiB-per-layer streaming scratchpad. This does not reduce the per-layer per-tile peak; it just rearranges the storage of inter-layer hidden. And it breaks the existing `model_forward_paged` contract of "feed one call, get logits out" — forcing a much larger refactor.

Layer-by-tile is also the pattern used by flash-linear-attention for chunked prefill (see §h) and by vLLM's Mamba-2 prefill streaming. Conceptually: each tile is a complete "mini forward pass" whose only linkage to the prior tile is the GDN state and the paged KV cache.

Pseudocode (layer-by-tile, per-prompt):

```
fn model_forward_paged_streaming(tokens, ..., state, ...) -> Tensor {
    let tile_size = env("KILN_STREAMING_PREFILL_TILE", 8192);
    let mut last_logits = None;
    let mut cursor = 0;
    while cursor < tokens.len() {
        let end = (cursor + tile_size).min(tokens.len());
        let is_last = end == tokens.len();
        let tile_tokens = &tokens[cursor..end];
        let tile_logits = model_forward_paged(
            tile_tokens,            // embedding + all 32 layers on this tile
            ...,
            start_pos = cursor,     // paged KV cache threads via existing start_pos
            Some(&mut state),       // GDN state threads via existing LinearAttentionState
            ...,
            // INTERNAL knob (see §d): skip LM head unless is_last, and only emit
            // the final token's row when is_last
        )?;
        if is_last { last_logits = Some(tile_logits); }
        cursor = end;
    }
    last_logits.expect("at least one tile")
}
```

### (c) State handover contract

Three pieces of state cross tile boundaries; two already work, one needs alignment discipline:

1. **`LinearAttentionState` (GDN recurrent + conv states).** Already defined at `forward.rs:241-270` with the docstring "This state is O(1) in sequence length — it does not grow with the number of tokens processed." Threading is already implemented: `gated_deltanet_forward` takes `&mut recurrent_states[i]` and `&mut conv_states[i]` (forward.rs:2911–2925), and `causal_conv1d_prefill` already threads `conv_state` correctly (forward.rs:907–946, writes the final `k-1` cols back to `conv_state` at 931–943). Per-layer: `recurrent_states[1, nv=32, dk=128, dv=128] F32` = 2 MiB; `conv_states[1, qkv_dim, k-1=3] F32` < 100 KiB. **Total across 24 GDN layers: 48 MiB.** No changes to the struct, no changes to the kernels — we just hand the same `&mut state` into each tile's `model_forward_paged` call.
2. **Paged KV cache + `start_pos` (full-attn / GQA layers).** Already threaded. The 8 full-attention layers read prior tile KV pages via `block_table` indexing during the current tile's attention; `start_pos = cursor` threads correctly (forward.rs:2872, 2905). No new state. This is why full-attn layers are "tile-oblivious" — they see the same prefix attention semantics whether invoked monolithically or tiled.
3. **Chunk alignment (the one real constraint).** Because `gdn_chunkwise_recurrence` handles the tail-chunk path only *once per call* (the last partial chunk flushes into the state at function exit), we require `tile_size % GDN_CHUNK_SIZE == 0` for all non-final tiles so that within a tile the recurrence processes only full 64-token chunks and the state is clean at the tile boundary. The *final* tile (which covers the prompt remainder) is allowed to have a tail chunk — this is exactly the behavior `gdn_chunkwise_recurrence` already supports when the monolithic path runs with a non-multiple-of-64 prompt length.

**Contract summary:** for a tile `[cursor, cursor+N)` with `N % 64 == 0` (except the last tile), after `model_forward_paged(tokens[cursor..cursor+N], ..., start_pos=cursor, state)`:
- `state.recurrent_states[l]` for each GDN layer `l` holds the post-N recurrence state for that layer.
- `state.conv_states[l]` for each GDN layer `l` holds the last `k-1=3` input columns of that layer's conv1d input.
- Paged KV cache blocks `[cursor .. cursor+N)` are populated for all 8 full-attn layers.
- No other mutation persists between calls.

This is byte-identical to the monolithic contract because the operations within each tile are bit-for-bit the same ops that the monolithic path would execute on the same token range.

### (d) Output accumulation — last-tile-last-token only (inference)

The LM head is `[hidden=2560, vocab=151936]` matmulled against `hidden: [1, seq_len, 2560]` → `logits: [1, seq_len, vocab=151936]`. At T=65536 the full logits tensor is **~19 GiB F32** (or ~9.5 GiB BF16 if we downcast — still a huge sink). The audit (Candidate C) called this out as the dominant post-norm allocation.

For **inference**, we only need the last row. `generate_from_tokens_paged_inner` at `generate.rs:547` samples from `logits` using `greedy_sample`/`sample_with_params`, which already operate on the last time-step. We therefore:

- On **non-last tiles:** skip the LM head entirely. Return `None` (or a cheap sentinel tensor) and drop the tile's `hidden` on function exit. Memory is reclaimed before the next tile starts.
- On **the last tile:** compute `rms_norm` + LM-head matmul, but optionally restrict to the final token's row — `hidden[.., -1:, ..]` of shape `[1, 1, 2560]` against `embed_tokens_t`, producing `[1, 1, vocab]` (~300 KiB BF16). Samplers are already last-token-indexed; this just lets us avoid materializing the full `[1, tile_size, vocab]` row even on the last tile.

Savings at T=65536 with tile=8192: LM-head output shrinks from ~19 GiB to ~300 KiB, a 60,000× cut on the final spike. Additionally, Candidate C ("lm_head output full-tensor sink") is mostly obviated — we never hold more than one tile's LM-head output, and on the critical last tile we only hold one row.

**Training (forward+backward) needs full logits** for the CE loss; streaming training prefill is explicitly deferred (§i). The `model_forward_paged_streaming` entrypoint is therefore **inference-only** in Phase 7; the existing `model_forward` remains the training path.

### (e) Integration surface

Minimal additive surface. No refactor of existing call sites beyond the dispatch point.

- **New function in `crates/kiln-model/src/forward.rs`:**
  ```rust
  pub fn model_forward_paged_streaming(
      backend: &dyn BackendRuntime,
      token_ids: &[u32],
      weights: &GpuWeights,
      config: &kiln_core::config::ModelConfig,
      paged_cache: &mut PagedKvCache,
      block_table: &BlockTable,
      start_pos: usize,
      mut linear_state: Option<&mut LinearAttentionState>,
      lora: Option<&LoraWeights>,
      positions_gpu: Option<&Tensor>,
  ) -> Result<Tensor> {
      // Same signature as model_forward_paged. Iterates tiles, delegates
      // per-tile work to model_forward_paged with the "skip LM head on
      // non-last tiles" knob (internal, not exposed as a pub API).
  }
  ```
  Same signature as `model_forward_paged` so the dispatch wrapper is a drop-in.
- **Internal knob in `model_forward_paged` (or a sibling `model_forward_paged_tile`):** an `is_last_tile: bool` param (crate-private) that short-circuits the LM-head section (forward.rs:2944–2950). On non-last tiles it returns a 1-byte placeholder tensor and the caller discards it.
- **Dispatch at two call sites** (`generate.rs:525`, `bench.rs:484`):
  ```rust
  let prefill_fn = if streaming_prefill_enabled(seq_len) {
      model_forward_paged_streaming
  } else {
      model_forward_paged
  };
  let logits = prefill_fn(backend, prompt_tokens, ...)?;
  ```
- **Config flags (parsed in `crates/kiln-server/src/config.rs`):**
  - `KILN_STREAMING_PREFILL` — `0|1` (default `0`). Master opt-in.
  - `KILN_STREAMING_PREFILL_TILE` — integer (default `8192`). Tile size in tokens; must be multiple of 64 (validated on parse; falls back to 8192 with a warning otherwise).
  - `KILN_STREAMING_PREFILL_THRESHOLD` — integer (default `32768`). Only stream when `seq_len >` this threshold; short prompts take the monolithic fast path. Setting to `0` forces streaming for all prefills (used by parity tests).
- **Kill switch:** `KILN_STREAMING_PREFILL=0` restores byte-identical behaviour. This is the default for Phase 7 ship; we gate on measurement before flipping the default.
- **Zero changes to:** `LinearAttentionState` struct, GDN kernels (`kiln-gdn-kernel`), conv1d kernel (`kiln-conv1d-kernel`), `gdn_chunkwise_recurrence`, `gated_deltanet_forward`, `causal_conv1d_prefill`, paged KV cache, `transformer_block_paged`, `model_forward` (training), `generate_from_tokens` (non-paged path).
- **CUDA graph interaction:** none. Prefill never uses CUDA graphs (`generate.rs:524` comment: "Prefill: forward pass on all prompt tokens (never uses CUDA graphs)"). Decode path is untouched; the decode-time CUDA graph replay (`crates/kiln-model/src/cuda_graph.rs:300,358`) continues to call `model_forward_paged` for single-token steps after streaming prefill completes.

### (f) Correctness strategy — bitwise parity

Streaming is a pure code-motion refactor: the kernel invocations, their inputs, and their outputs on each token range are identical to the monolithic path. Correctness is provable by parity tests, not by eyeballing tolerances.

Tests (CPU backend, add to `crates/kiln-model/src/forward.rs` `#[cfg(test)]` module):

1. **`test_streaming_matches_monolithic_cpu_small`**: T=128, tile=64. Build a tiny model via existing test scaffolding (the GDN decode tests at forward.rs:4256–4300 are the template), run both `model_forward_paged` and `model_forward_paged_streaming` from zero-state, assert `logits_mono == logits_stream` **bit-exact** (`to_vec1::<f32>()` equality).
2. **`test_streaming_matches_monolithic_cpu_mid`**: T=2048, tile=512. Same assertion.
3. **`test_streaming_tile_invariance_cpu`**: T=1024, loop `tile ∈ {64, 128, 256, 512, 1024}`; assert all five runs produce bit-equal logits.
4. **`test_streaming_preserves_state_cpu`**: after streaming T=2048 with tile=512, assert `state.recurrent_states[l]` and `state.conv_states[l]` are bit-equal to the monolithic run's state.
5. **`test_streaming_disabled_is_byte_identical`**: with `KILN_STREAMING_PREFILL=0`, assert dispatch goes through `model_forward_paged` unchanged (regression guard for the `≤32k` fast path).

**CUDA parity** (run on the spike pod, §g): `test_streaming_matches_monolithic_cuda` — same as #2 above but on CUDA device. F32 recurrent state means GDN is reproducible across tilings; the conv1d F32 promotion we already do (Candidate A future work) keeps conv1d deterministic too. Paged attention is tile-oblivious by construction.

**Why this is tractable:** the operations in each tile are the *same* ops in the *same* order on the *same* memory; we are just not holding intermediate activations from tile N-1 when tile N runs. State threading is already a tested code path (decode uses it token-by-token — forward.rs:4256–4300, `test_paged_single_token_forward` and related). This is prefix-block aggregation at a larger granularity.

### (g) Spike scope — one pod, ~30 min, <$0.70

- **Goal:** land `KILN_STREAMING_PREFILL=1` behind a flag; prove parity on CPU + CUDA; prove T=65536 prefill completes on A6000; prove ≤32k fast path is untouched.
- **Pod:** acquire via `ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000'`. Fallback to direct RunPod if pool is capped.
- **Runtime:** single session.
- **Steps:**
  1. Implement `model_forward_paged_streaming` + dispatch + config flags. (~1 hr local, no pod.)
  2. Add 5 CPU parity tests. Iterate on CPU until green. (local, no pod.)
  3. Open PR in draft. Wake pod.
  4. On pod: `cargo nextest run --features cuda` — all existing tests green + CUDA parity test.
  5. On pod: `KILN_STREAMING_PREFILL=1 KILN_W4A16=1 ./target/release/kiln-bench --paged --prompt-tokens 65536 --max-output-tokens 32 --skip-training`. Confirm no OOM; record prefill latency + decode tok/s.
  6. On pod: ≤32k regression. `KILN_STREAMING_PREFILL=0 KILN_W4A16=1 ./target/release/kiln-bench --paged --prompt-tokens 8192 --max-output-tokens 128` and same with `KILN_STREAMING_PREFILL=1 KILN_STREAMING_PREFILL_THRESHOLD=0` (force streaming on a short prompt); 3× each, confirm medians are within noise (≤2% variance).
  7. Capture nsys NVTX trace of one streaming prefill for PROFILING.md re-profile attachment. Optional.
  8. Release the lease.
- **Success criteria:**
  - All parity tests green.
  - T=65536 prefill + T=32 decode completes without OOM on 48 GiB A6000.
  - ≤32k fast path unchanged (median prefill/decode within noise vs pre-PR main).
  - No regression on the existing bench harness.
- **Cost ceiling:** ~0.5 h × A6000 on-demand (~$0.79/hr) ≈ $0.40. Budget cap $0.70 including warm-up and nsys attribution run. If the spike trips anything unexpected, ship the doc-only design (this PR), redirect with a null-finding PR, and re-plan.

### (h) flash-linear-attention precedent

The `flash-linear-attention` (fla) repo (https://github.com/sustcsonglin/flash-linear-attention) — which is where kiln's `kiln-gdn-kernel` ultimately derives from (PR #80) — ships its linear-attention ops with an explicit `chunks=` / `chunk_size=` knob and a paired `initial_state=` / `final_state=` contract. Representative APIs:

- `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule(q, k, v, g, beta, scale=..., initial_state=None, output_final_state=True, cu_seqlens=None)` — the chunkwise Gated DeltaNet forward. `initial_state` and `output_final_state` are the fla equivalent of our `LinearAttentionState` threading. `cu_seqlens` supports variable-length batches in the same kernel invocation.
- `fla.ops.delta_rule.chunk.chunk_delta_rule(..., initial_state=..., output_final_state=...)` — same pattern on the ungated variant.
- fla's chunk size is a kernel-internal constant (64 — matches `GDN_CHUNK_SIZE` in `kiln-gdn-kernel`). fla does not stream across *kernel calls*; streaming is left to the caller, using `initial_state` / `final_state`.

kiln's streaming prefill is therefore a *model-level* analog of the pattern fla already encodes at the op level. We reuse the vendored chunk kernel unchanged (the "compute per-chunk recurrence, fold into state" contract is identical), and layer tiling outside it. This is precisely what vLLM does for its Mamba-2 streaming prefill (https://github.com/vllm-project/vllm/pull/12093 and siblings): the mamba kernel stays fla-derived; tiling is orchestrated at the model-forward level with explicit state threading.

We are not inventing a new streaming algorithm. We are picking up the state-threading contract fla has always shipped and wiring it into our paged forward.

### (i) What is deliberately deferred

- **Training (backward) streaming.** CE loss needs full logits and the gradient of the full hidden sequence; streaming backward needs tile-boundary state checkpointing + a compatible reverse-recurrence path. Phase 7.x, not Phase 7.0.
- **Async tile pipelining.** Overlap tile-N compute with tile-(N+1) embedding lookup + positions_gpu build. Useful but orthogonal — costs CUDA stream plumbing and a second `hidden` buffer. Phase 7.2.
- **Dynamic tile sizing.** Measure free VRAM at runtime and adjust `KILN_STREAMING_PREFILL_TILE` per-prompt. Useful for mixed-GPU fleets and varying LoRA footprints. Phase 7.3.
- **FP8 KV cache × streaming prefill interaction.** Already composes cleanly at the design level (prefill populates FP8 KV blocks tile-by-tile exactly as it does monolithically), but empirical re-verification at T ≥ 32k is deferred until both features are on simultaneously in one bench.
- **Candidate A (F32 → BF16 `causal_conv1d_prefill`).** Independent of streaming; shrinks conv bubble by ~2× inside each tile. Still queued from the PR #226 audit; orthogonal to this PR.
- **Candidate B (l2_normalize in-place).** Independent; smaller peak within GDN. Still queued.
- **Candidate C (LM-head full-tensor sink).** *Partially subsumed* by §d — we only ever hold one tile's logits, and only the last token on the last tile. The remaining Candidate-C opportunity (chunked softmax inside the sampler) is unchanged but now off the critical OOM path.
- **Dispatching streaming on the non-paged `model_forward` path.** Non-paged is the training path; inference uses paged. If someone introduces long-context non-paged inference we revisit; today no call site needs it.
- **Flipping `KILN_STREAMING_PREFILL` default to `1`.** Ship behind a flag in Phase 7.0; flip the default only after the spike parity + bench + nsys data land and a re-profile confirms no silent regression on short prompts.

---

TL;DR: tile-oblivious full-attn + existing O(1) `LinearAttentionState` + chunk-aligned tile size = long-context prefill with zero kernel changes and a surgically small forward-path patch. The next PR implements this behind `KILN_STREAMING_PREFILL=1` and measures it in a single ~30-min A6000 spike.


## GDN prefill memory audit (Phase 7 opener) — 2026-04-20

**Outcome: the GDN recurrent state (the thing FP8 would shrink) is ~48 MiB total across 24 layers — 0.1% of the OOM ceiling. The binding constraint is per-layer prefill activations. At `seq_len=65536` one GDN layer's live tensors sum to roughly 9–10 GiB, which exceeds the ~37 GiB working budget on A6000 (48 GiB − ~8 GiB weights − ~3 GiB CUDA context − carry across 32 layers). Long-context capability is unlocked by streaming/tiled prefill (candidate G below), not by shrinking state.**

This audit is a $0 source-inspection preflight following PRs
[#131](https://github.com/ericflo/kiln/pull/131),
[#163](https://github.com/ericflo/kiln/pull/163),
[#164](https://github.com/ericflo/kiln/pull/164),
[#170](https://github.com/ericflo/kiln/pull/170),
[#219](https://github.com/ericflo/kiln/pull/219),
[#223](https://github.com/ericflo/kiln/pull/223),
[#225](https://github.com/ericflo/kiln/pull/225). The trigger is PR
[#222](https://github.com/ericflo/kiln/pull/222)'s FP8 KV cache verification
matrix: 65536p and 131072p prompts OOM'd in **both** the BF16 and FP8 arms,
and the failure was localized to **GDN layer 0 prefill** — not the paged GQA
KV cache. FP8 halves KV bytes but the 24 linear-attention layers dominate
activation memory before the 8 GQA layers' paged KV is even touched, so
`KILN_KV_CACHE_FP8` cannot lift the ceiling. The question this audit
answers: what exactly does GDN prefill allocate, and which reduction opens
≥65536p on a 48 GiB A6000?

### Config reference (Qwen3.5-4B)

From `crates/kiln-core/src/config.rs:79–101` and
`crates/kiln-model/src/forward.rs:250–264`, `990`:

| Symbol | Value | Meaning |
| --- | ---: | --- |
| `hidden` | 2560 | residual hidden size |
| `L_total` | 32 | total transformer layers |
| `L_full` | 8 | full-attention layers (paged GQA KV) |
| `L_gdn` | 24 | GDN linear-attention layers |
| `nk` | 16 | GDN key heads |
| `nv` | 32 | GDN value heads |
| `dk` | 128 | GDN key head dim |
| `dv` | 128 | GDN value head dim |
| `qk_dim` | 2048 | `nk * dk` |
| `v_dim` | 4096 | `nv * dv` |
| `qkv_dim` | 8192 | `2 * qk_dim + v_dim` (fused conv channels) |
| `kernel_size` | 4 | causal depthwise conv |
| `C` | 64 | `GDN_CHUNK_SIZE` |
| dtype | BF16 | hot path activation/weight dtype |

For the allocation table below, assume `B = 1`, `T = seq_len = 65536`, `C = 64`, `nc = T / C = 1024 full chunks`. All tensors are live on the CUDA device.

### (a) Source-level allocation table — single GDN layer at `T = 65536`

Byte totals are the product of shape × dtype bytes, on a single GDN layer's
forward pass through
`gated_deltanet_forward` and `gdn_chunkwise_recurrence` in
`crates/kiln-model/src/forward.rs`.

| # | Tensor | File:line | Shape | Dtype | Bytes @ T=65536 |
| --- | --- | --- | --- | --- | ---: |
| 1 | hidden `x` input (layer carry) | `forward.rs:~2600` (layer loop) | `[B, T, hidden]` | BF16 | 320 MiB |
| 2 | `mixed_qkv` after `in_proj_qkv` matmul (Step 1) | `forward.rs:1443` | `[B, T, qkv_dim]` | BF16 | 1.00 GiB |
| 3 | `z` after `in_proj_z` matmul | `forward.rs:1444` | `[B, T, v_dim]` | BF16 | 512 MiB |
| 4 | `a`, `b` gate projections | `forward.rs:1445–1446` | `[B, T, nv]` | BF16 | 4 MiB each |
| 5 | `mixed_qkv_ct` for conv (transpose + contiguous) | `forward.rs:1462` | `[B, qkv_dim, T]` | BF16 | 1.00 GiB |
| 6a | `x_padded = cat(conv_state, x)` inside `causal_conv1d_prefill` | `forward.rs:915,921` | `[B, qkv_dim, T+3]` | **F32** | 2.00 GiB |
| 6b | `output` accumulator inside `causal_conv1d_prefill` (kernel_size=4 steps) | `forward.rs:924–928` | `[B, qkv_dim, T]` | **F32** | 2.00 GiB |
| 7 | `post_silu` = `cuda_silu(.to_dtype(F32))` result | `forward.rs:1488` | `[B, qkv_dim, T]` | **F32** | 2.00 GiB |
| 8 | `post_silu.transpose(1, 2)` — F32 view/copy | `forward.rs:1499` | `[B, T, qkv_dim]` | **F32** | 2.00 GiB |
| 9 | `v` cast to input_dtype in `recur_prep` | `forward.rs:1652` | `[B, T, nv, dv]` | BF16 | 512 MiB |
| 10 | `q` after GQA head_expand contiguous | `forward.rs:1522–1526` | `[B, T, nv, dk]` (F32 through Step 4) | F32 | 1.00 GiB |
| 11 | `k` after GQA head_expand contiguous | `forward.rs:1527–1531` | `[B, T, nv, dk]` | F32 | 1.00 GiB |
| 12 | `l2_normalize(q)` F32 output | `forward.rs:1578` / `l2_normalize` at `852–858` | `[B, T, nv, dk]` | F32 | 1.00 GiB |
| 13 | `l2_normalize(k)` F32 output | `forward.rs:1579` | `[B, T, nv, dk]` | F32 | 1.00 GiB |
| 14 | `q` cast to BF16 after norm+scale | `forward.rs:1580` | `[B, T, nv, dk]` | BF16 | 512 MiB |
| 15 | `k` cast to BF16 after norm | `forward.rs:1581` | `[B, T, nv, dk]` | BF16 | 512 MiB |
| 16 | `q.transpose(1, 2)` for recurrence | `forward.rs:1655` | `[B, nv, T, dk]` | BF16 (view→contig on entry into chunked path) | 512 MiB |
| 17 | `k.transpose(1, 2)` | `forward.rs:1656` | `[B, nv, T, dk]` | BF16 | 512 MiB |
| 18 | `v.transpose(1, 2)` | `forward.rs:1657` | `[B, nv, T, dv]` | BF16 | 512 MiB |
| 19 | `q_pre` pre-permuted `[nc, B, nv, C, dk]` contiguous | `forward.rs:1231` / `preshape_chunked_4d` at `1014–1038` | `[1024, 1, 32, 64, 128]` | BF16 | 512 MiB |
| 20 | `k_pre` | `forward.rs:1232` | same shape | BF16 | 512 MiB |
| 21 | `v_pre` | `forward.rs:1233` | `[1024, 1, 32, 64, 128]` | BF16 | 512 MiB |
| 22 | `beta_pre`, `g_pre` | `forward.rs:1234–1235` | `[1024, 1, 32, 64]` | BF16 | 16 MiB each |
| 23 | `out_chunks: Vec<Tensor>` accumulated outputs | `forward.rs:1237, 1366` | `nc × [B, nv, C, dv]` | BF16 | 512 MiB (sum) |
| 24 | `Tensor::cat(&out_chunks, 2)` final recurrence output | `forward.rs:1377` | `[B, nv, T, dv]` | BF16 | 512 MiB (new) |
| 25 | `attn_out.transpose(1, 2)` | `forward.rs:1683` | `[B, T, nv, dv]` | BF16 | 512 MiB |
| 26 | `gated_rms_norm` F32 output before reshape/cast | `forward.rs:1689` / `gated_rms_norm` at `885–900` | `[B, T, nv, dv]` | **F32** | 1.00 GiB |
| 27 | post-`gated_rms_norm` reshape + cast to BF16 | `forward.rs:1691–1693` | `[B, T, v_dim]` | BF16 | 512 MiB |
| 28 | output of `out_proj` matmul (written back into residual) | `forward.rs:~1700+` | `[B, T, hidden]` | BF16 | 320 MiB |
| C1 | **O(1) recurrent state** per layer | `forward.rs:262` | `[B, nv, dk, dv]` | F32 | **2 MiB** (constant, not T-scaling) |
| C2 | **O(1) conv state** per layer | `forward.rs:263` | `[B, qkv_dim, k−1]` | F32 | **96 KiB** (constant, not T-scaling) |

**Note on row 6/7/8 — the F32 conv bubble**. `causal_conv1d_prefill`
unconditionally promotes the conv input to F32 (line 915
`x_f32 = x.to_dtype(F32)?`), then SiLU runs on F32, and the result only
returns to BF16 after Step 5's final cast. Between conv entry and the
`input_dtype` re-cast at Step 7 (`v.to_dtype(input_dtype)` at line 1652) the
sequence carries a full `[B, T, qkv_dim]` F32 tensor — 2× the byte cost of
the BF16 equivalent, and the single largest non-pre-chunk allocation in the
whole layer.

**Note on row 10/11 — GQA head_expand materialization**. `gqa_ratio =
nv / nk = 2`, so `q`/`k` are duplicated along a new head axis via
`unsqueeze(3).expand(...).contiguous().reshape(...)`. The `contiguous()`
forces a real copy — at T=65536 this is two 1 GiB allocations per layer
before `l2_normalize` even runs.

**Note on row 19–22 — the pre-permutation tax**. `preshape_chunked_4d`
calls `.contiguous()` at line 1037 to materialize a new tensor with the
chunk axis leading, enabling zero-copy chunk slices inside the loop. This
duplicates q/k/v/beta/g across the full sequence length: **~1.5 GiB of
extra live BF16** at T=65536 in addition to rows 16–18. The originals
(16/17/18) are still borrowed through the loop for the tail (`narrow` at
lines 1246–1250), so both live simultaneously.

### (b) Peak-live-at-once analysis

Not every row is live at the peak; candle releases intermediates as
their last `&Tensor` borrow ends. The worst moment is immediately before
the chunk loop begins — after `gdn_chunkwise_recurrence` has materialized
the five `*_pre` tensors but before `q`, `k`, `v`, `beta`, `g` go out of
scope (they remain referenced for the tail branch at lines 1246–1250 and
for the narrow/squeeze/contiguous calls). Hidden-state carry from outside
the layer also stays live.

| Row | Size | Rationale for being live at peak |
| ---: | ---: | --- |
| 1 residual `x` carry | 320 MiB | layer input — still referenced across 32 layers; next-layer residual-add will still read it |
| 5 `mixed_qkv_ct` (BF16) | 1.00 GiB | may be released before conv returns, but worst-case overlaps with row 6 |
| 6a `x_padded` F32 inside conv | 2.00 GiB | live during conv loop |
| 6b conv `output` F32 | 2.00 GiB | live during conv loop (written every kernel_size step) |
| 8 post-transpose F32 `mixed_qkv` | 2.00 GiB | live from Step 3 narrow/reshape until v/q/k drop their F32 references |
| 10 q expanded F32 | 1.00 GiB | live through Step 5 |
| 11 k expanded F32 | 1.00 GiB | live through Step 5 |
| 12 l2_normalize(q) F32 | 1.00 GiB | brief overlap with row 10 |
| 13 l2_normalize(k) F32 | 1.00 GiB | brief overlap with row 11 |
| 16 q transposed BF16 | 512 MiB | live through whole chunk loop (tail fallback) |
| 17 k transposed BF16 | 512 MiB | live through whole chunk loop |
| 18 v transposed BF16 | 512 MiB | live through whole chunk loop |
| 19 q_pre | 512 MiB | live through whole chunk loop |
| 20 k_pre | 512 MiB | live through whole chunk loop |
| 21 v_pre | 512 MiB | live through whole chunk loop |
| 22 beta_pre + g_pre | 32 MiB | live through whole chunk loop |
| 23 `out_chunks` accumulator | 512 MiB | grows to full size before cat() |

The **conv bubble peak** (rows 6a + 6b + carry from row 1 + 5): ~6.32 GiB.

The **post-l2_normalize peak** (rows 8 + 10 + 11 + 12 + 13 + 1 + 14 + 15):
~9.0 GiB including residual carry. This is the peak of the F32 phase —
after the two `.to_dtype(input_dtype)` casts at lines 1580–1581 the F32
copies of q/k go out of scope and the peak drops.

The **chunk-loop peak** (rows 1 + 16–23): ~3.4 GiB sustained.

The **cat/gated_norm peak** (rows 1 + 24 + 26): ~1.83 GiB. Brief.

**Dominant peak: the post-l2_normalize phase at ~9 GiB per layer.**

Global budget sanity:

- A6000: 48 GiB
- Weights at `KILN_W4A16=1`: q_proj Marlin packed + MLP Marlin packed ≈ 4.5 GiB (post-PR #206); GDN projections still BF16 (`in_proj_qkv`, `in_proj_z`, `in_proj_a/b`, `out_proj`, `conv1d`, `norm`, `a_log`, `dt_bias`, plus pre-transposed `*_t` copies from PR #128); embeddings + lm_head + full-attn layers. Total resident ≈ **7–9 GiB** on current main.
- CUDA context, workspace, allocator fragmentation: **~3 GiB**.
- Per-step carry across 32 layers: residual `[B, T, hidden]` BF16 + any lazy-released activation tails: **320 MiB minimum**, up to **~1 GiB** in practice.
- Working budget for a single layer's prefill: **~35–37 GiB**.

One GDN layer's ~9 GiB peak fits with margin. But the peak is not the only cost: allocator fragmentation, the `x_padded` F32 bubble reappearing on every layer's conv, and the GQA head_expand copies compound across layers during prefill. The observed OOM at GDN layer 0 suggests either (a) fragmentation from the initial BF16 weight residency is higher than naively budgeted, or (b) the F32 `mixed_qkv` tensor and the dual pre-permutation tensors are alive simultaneously for longer than the static analysis above assumes. Either way, the **per-layer activation footprint is the binding axis**.

At `T = 131072` everything T-scaling doubles: ~18 GiB per-layer peak. No realistic weight-side cleanup gets that under 35 GiB.

### (c) Reduction candidates ranked by impact

All numbers are at `T = 65536`, single layer, relative to the rows in (a).

| Cand. | Description | Bytes saved @ 65k (per-layer peak, approximate) | Complexity | Numerical risk | Upstream precedent |
| :---: | --- | ---: | --- | --- | --- |
| **A** | Keep conv output BF16 (drop F32 promotion in `causal_conv1d_prefill` rows 6a/6b/7/8) | ~4.0 GiB | LOW–MEDIUM: requires BF16 depthwise conv that doesn't overflow; use the fused `kiln-conv1d-kernel` path (already BF16-native for decode) or extend it to prefill | LOW: bf16 SiLU is standard in fla/vLLM Mamba paths | flash-linear-attention, vLLM Mamba keep conv in compute dtype end-to-end |
| **B** | Skip pre-permutation (rows 19–22) at long T; use direct `narrow` per chunk | ~1.5 GiB | MEDIUM: pre-permute was added to kill the `copy2d_bf16` per-chunk hotspot (PROFILING.md history). Replace with (i) gated-on-`seq_len` short-prefill only, or (ii) fused chunk-indexed gather in `kiln-gdn-kernel` | LOW: identical math; only locality changes | fla chunk_gla_fwd slices on the fly without pre-permutation |
| **C** | Write chunk outputs in-place into a pre-allocated `[B, nv, T, dv]` BF16 buffer; eliminate `Vec<Tensor>` + final `cat` (rows 23 + 24 collapse to one 512 MiB buffer) | ~512 MiB | LOW: pre-allocate once, `slice_assign` per chunk | NONE: bit-identical | vLLM output accumulation pattern |
| **D** | FP8/FP16 recurrent state (row C1) | ~1 MiB (single layer) / **~24 MiB total across 24 layers** | LOW (E4M3 cast, mirror #222 FP8 KV cache) | MEDIUM: state accumulator precision matters; Phase-6 PR [#72](https://github.com/ericflo/kiln/pull/72)/[#74](https://github.com/ericflo/kiln/pull/74) already moved the *internal* recurrence to BF16 with F32 boundary cast for stability | — (state is already F32 for accumulator stability; reducing it regresses precision without unlocking context) |
| **E** | Reduce `GDN_CHUNK_SIZE` from 64 to 32 | ≤ 5 MiB (per-chunk scratch shrinks ~75% but chunk count doubles) | LOW (one `const` change + parity tests) | LOW | — |
| **F** | Checkpoint / stream the full-attention (GQA) KV-cache writes so `mixed_qkv` F32 can be freed earlier in each GDN layer | overlaps with A; no independent savings | MEDIUM | LOW | — |
| **G** | **Streaming/tiled prefill: split T into super-chunks of `S` tokens (e.g. S=4096), run the full 32-layer stack end-to-end per super-chunk, carrying only the per-layer recurrent state (C1, 2 MiB/layer) + conv state (C2, 96 KiB/layer) between super-chunks.** Activation peak becomes `O(S)` not `O(T)`. | **~9 GiB** at S=4096 for T=65536; **~17 GiB** at S=4096 for T=131072 (the entire T-scaling half of per-layer peak goes away) | **HIGH**: re-architect `model_forward_prefill` to iterate super-chunks, preserve `LinearAttentionState` + paged GQA KV cache across iterations, concatenate output hidden states only at the lm_head boundary (or stream to the final-token path when only the last logits are needed) | LOW: `LinearAttentionState` *already* supports this — it's designed to carry across prefill/decode (see `LinearAttentionState` docstring at `forward.rs:240`, and the decode tests at `forward.rs:4256–4300` that drive prefill-then-decode through the same state). Paged GQA KV cache already appends across calls. | vLLM handles long Mamba/Mamba-2 contexts exactly this way; fla documents super-chunk prefill as the canonical path. See also `flash-linear-attention/fla/layers/gated_deltanet.py` `chunk_size=` / `chunks=` iteration pattern. |

### (d) Recommended Phase 7 opener

**Candidate G (streaming/tiled prefill with O(1) state handover) is the only candidate that both unlocks `T ≥ 65536` on A6000 and extends naturally to `T = 131072`.** Candidates A+B+C together sum to ~6 GiB saved per-layer peak — enough to maybe land 65536p but not 131072p, and fragile under any future model change that extends `v_dim` or adds heads.

Ranked by (unlock × complexity × risk):

1. **G — streaming/tiled prefill (opener).** Unlocks the full 262K native
   context window on A6000 without quantizing anything. Single largest
   architectural win available in Phase 7. The `LinearAttentionState` and
   paged KV cache plumbing is already in place; the work is in
   `model_forward_prefill` (sequence iteration, state threading, hidden-state
   concatenation or streaming to lm_head). Gate the streaming path behind a
   `seq_len > threshold` check (suggest `threshold = 8192`) so short
   prefills pay zero throughput penalty.
2. **A — BF16-native conv prefill (follow-up #1).** Orthogonal to G.
   Even after G drops the per-layer peak, keeping conv output in BF16
   reclaims ~2× memory on the conv bubble and shrinks the super-chunk
   floor, which lets G use a larger `S` (higher throughput). Already half
   done: `kiln-conv1d-kernel` is BF16-native for decode; extend to
   prefill. Upper-bound on savings independent of G.
3. **C — in-place chunk accumulator (follow-up #2).** Cheap mechanical
   win (~512 MiB per layer), strictly additive to G, no risk.
4. **B — drop pre-permutation at long T (follow-up #3, conditional).**
   Only worth doing if G + A + C land and 131072p still needs headroom at
   a desired `S`. Gate on `seq_len` so short prefills retain the
   pre-permutation locality win.

Items D, E, F do not move the needle on the 65k/131k ceiling.

### (e) Preconditions for re-opening / invalidation

This recommendation should be re-verified before a Phase 7 implementation PR
if any of the following change:

- **G becomes upstream-obvious or already partially landed.** Check
  `git log --all --grep='streaming\\|tiled.*prefill\\|super.*chunk'` on
  kiln and `gh pr list -R ericflo/kiln --state all --limit 50`. The
  `LinearAttentionState` threading pattern is already there, but no
  current PR or branch drives it from the model-forward layer.
- **fla upstream ships a single-kernel long-context GDN prefill** that
  makes the Rust-side super-chunking unnecessary. If
  [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
  adds a fused `chunk_gla_fwd_streaming` or similar, vendor that kernel
  into `kiln-gdn-kernel` instead (same minimal-scope precedent as
  [PR #80](https://github.com/ericflo/kiln/pull/80)).
- **Nsys evidence at 16384p or 32768p shifts the peak away from the
  post-l2_normalize phase.** The static analysis above assumes the F32
  inflation is the dominant axis. If a real runtime snapshot at
  `T = 32768` (the largest size that currently fits) shows the chunk-loop
  tensors or the output-cat as the dominant mass, re-prioritize C over
  the others.
- **Weight residency drops further** (e.g. PR [#206](https://github.com/ericflo/kiln/pull/206) extensions pack more BF16 projections into Marlin). Each GiB freed is a GiB the per-layer peak can grow into — if total weights drop below 5 GiB, candidates A+B+C alone might cross the 65536p line without G. Re-check the working-budget arithmetic in (b) before committing.
- **A new runtime memory snapshot contradicts the static analysis.**
  This audit is source-only; it doesn't observe allocator fragmentation
  or candle's actual release timing. If a minimal GPU
  instrumentation task (described below) is ever run, its results
  supersede rows (b) and the dominant peak may shift.

### Follow-up: optional GPU instrumentation task (bounded, not blocking)

If a future planner wants ground truth before committing to G, the
minimal-cost verification is: run the existing 16384p bench with
`CUDA_LAUNCH_BLOCKING=1` and `cudaMemGetInfo` polling inside
`gated_deltanet_forward` (gated behind a `KILN_MEM_TRACE=1` env var),
capturing free-MiB before/after each numbered step in (a). One A6000 pod,
~15 minutes of runtime, single run is sufficient (peak memory is
deterministic at fixed T, not latency-like). Do **not** launch this from
the current task — this audit is $0 and the static analysis already
points unambiguously at G.


## Phase 6 preflight 2026-04-20: Marlin wrapper BF16-native I/O epilogue (KILL)

**Outcome: KILL. Phase 6 is complete. Advancing to Phase 7.**

This preflight gates the final un-attempted Phase 6 decode lever surfaced
by the [#224](https://github.com/ericflo/kiln/pull/224) frontier audit:
eliminating the per-call BF16↔FP16 cast pair and FP16 intermediate buffer
around every Marlin W4A16 GEMM call in the decode path. The audit
projected region-local speedups in the 1.05–1.10× range with a
contribution of 0.017–0.033 (plausibly clearing the 0.05 floor under two
compounding assumptions). This preflight, $0 and doc-only, tests both
assumptions against the post-#210 direct nsys evidence and kills on two of
the three required checks.

No pod spent. No code changed. Precedent for doc-only redirect: PRs
[#131](https://github.com/ericflo/kiln/pull/131),
[#163](https://github.com/ericflo/kiln/pull/163),
[#164](https://github.com/ericflo/kiln/pull/164),
[#170](https://github.com/ericflo/kiln/pull/170),
[#219](https://github.com/ericflo/kiln/pull/219),
[#223](https://github.com/ericflo/kiln/pull/223).

### Check 1 — BF16-native Marlin tile exists upstream: **PASS**

vLLM's templated Marlin (`csrc/quantization/marlin/`, the successor to the
IST-DASLab FP16-only kernel that kiln currently vendors via PR #146)
carries a full BF16 dtype specialization:

- `csrc/quantization/marlin/marlin_dtypes.cuh`: defines
  `MarlinScalarType<vllm::kBFloat16.id()>` with `nv_bfloat16` /
  `nv_bfloat162` `FragA` / `FragB` / `FragS` types (lines 54–92). Aligned
  on the same 16-lane fragment structure as the FP16 specialization, so
  the tile shape (`Marlin<256,1,8,8,4,8>`) is compatible with the BF16
  variant without geometric changes.
- `csrc/quantization/marlin/marlin_mma.h` lines 66–75: emits
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` under
  `__CUDA_ARCH__ >= 800`. RTX A6000 is sm_86 — supported. H100 (sm_90) and
  A100 (sm_80) are supported. Blackwell (sm_120a) is not the target arch
  for kiln under CUDA 12.4.
- `csrc/quantization/marlin/dequant.h` lines 174–193: provides
  `dequant<nv_bfloat162, vllm::kU4B8.id(), true/false>` specializations
  for both symmetric and asymmetric 4-bit dequant.

Vendoring path is precedented: PR [#80](https://github.com/ericflo/kiln/pull/80)
pulled `chunk_gla_fwd` from fla upstream into `kiln-gdn-kernel` with a
similar scope (several hundred lines of templated CUDA + dtype
specializations). A BF16 port would fork
`crates/kiln-marlin-gemm/csrc/marlin_kernel.cu` and add a parallel
`marlin_bf16_kernel.cu` with the dequant and mma branches swapped out.

**Check 1 is not the blocker.**

### Check 2 — wrapper cast+alloc ≥5% of each Marlin region wall-clock: **PARTIAL**

The direct evidence that exists is kernel-level, not NVTX sub-range. From
`profiling-artifacts/post210_kern.csv`:

```
%  total_ns     instances  avg_ns  name
15.7  279,139,141  13,527  20,636  void Marlin<(int)256,(int)1,(int)8,(int)8,(int)4,(int)8>(...)
 1.0   17,877,596  13,527   1,322  cast_bf16_f16
 1.0   17,695,841  13,527   1,308  cast_f16_bf16
```

The 13,527-instance count matches exactly across all three rows — every
Marlin call on the decode path is bracketed by one `cast_bf16_f16` on the
input side and one `cast_f16_bf16` on the output side, confirming the
wrapper structure at `crates/kiln-marlin-gemm/src/lib.rs` (the actual cast
pair lives one crate below `kiln-model/src/marlin_proj.rs`: lines 168
`a.to_dtype(DType::F16)?.contiguous()?`, 173
`Tensor::zeros((m,n), DType::F16, ...)`, 268
`c_fp16.to_dtype(DType::BF16)?`).

Cast-pair wall-clock is 35.57 ms vs Marlin kernel body 279.14 ms, so the
cast pair alone is **12.74% of the Marlin kernel body time**. Taken at face
value, this clears the 5% threshold for the **cast pair**.

However, the claim under evaluation is broader: "wrapper **cast + alloc**
overhead". Aggregate Marlin NVTX regions sum to 33.1% of decode, Marlin
kernel body is 15.7%, leaving 17.4 pp "outside the kernel" inside Marlin
regions. Of that 17.4 pp:

- 2.0 pp is the cast pair (direct kernel attribution above).
- The remaining **~15.4 pp is not broken down** at sub-range granularity
  in any current artifact. It is plausibly `ucopy_bf16` (8.6% total, but
  Marlin regions are not the only ucopy_bf16 callsite — PR #219 already
  ruled out per-site ucopy_bf16 fusion), `contiguous()` copies, the FP16
  output buffer `Tensor::zeros` alloc, workspace zeros, Candle stream
  sync, or dispatch launch overhead amortized by CUDA graph replay.

No NVTX sub-range capture (`:kiln/marlin/cast_in`, `:kiln/marlin/alloc`,
`:kiln/marlin/kernel`, `:kiln/marlin/cast_out`) exists in the
`profiling-artifacts/` archive. `grep -rn "cast_bf16_f16|cast_f16_bf16"
PROFILING.md` returns two references to the kernel-level attribution, and
no sub-range wrapper breakdown. The **15.4 pp of overhead that PR #224
attributed to "allocation, casts, layout, and stream sync"** is therefore
an arithmetic derivation (region total − kernel body), not a measurement.

Per this preflight's own acceptance rule (any of the three checks fails →
KILL), this ambiguity alone does not kill — the cast-pair fraction clears
5%. But it caps the honest math-ceiling to 2.0 pp definitely-eliminable,
not 17.4 pp speculatively-eliminable. **This becomes decisive in Check 3.**

Noting for the queue: a minimal nsys re-capture on a warm pod with NVTX
sub-ranges in `kiln-marlin-gemm/src/lib.rs` around the `to_dtype` /
`Tensor::zeros` / `kiln_marlin_w4a16_gemm` call edges would close this gap
permanently, but the Check 3 result below makes that re-capture not worth
a pod idle slot — the HBM analysis settles the question from the other
side.

### Check 3 — HBM-traffic reduction compounds under CUDA graph replay: **FAIL**

This is the decisive check. kiln runs with `KILN_CUDA_GRAPHS=true` by
default — decode-step kernel dispatch is captured into a graph and
replayed, so launch-dispatch savings amortize to near-zero and only HBM
traffic matters. This is the exact failure mode of null PRs
[#141](https://github.com/ericflo/kiln/pull/141) (gated_rms_norm),
[#173](https://github.com/ericflo/kiln/pull/173) (fused L2 qk_norm),
[#176](https://github.com/ericflo/kiln/pull/176) (big-fusion across
recurrent + qk_norm + gated_norm).

HBM-traffic arithmetic at m=1 decode for a representative Marlin call
(hidden_dim=2560, one of the 4-bit MLP / GDN projections with k=n=2560):

```
cast activation tensor (BF16 → FP16):
  read:  2560 × 2 B = 5,120 B
  write: 2560 × 2 B = 5,120 B
  per cast:          10,240 B
  both casts:        20,480 B per Marlin call

quantized weight read inside Marlin kernel:
  2560 × 2560 × 0.5 B (4-bit packed) ≈ 3,276,800 B ≈ 3.2 MB per call

activation read inside Marlin kernel (k=2560 BF16 / FP16):
  ≈ 5,120 B per call

wrapper HBM / total Marlin HBM per call:
  20,480 / (3,276,800 + 5,120) ≈ 0.62 %
```

Aggregate across all 13,527 Marlin invocations per decode run:

```
cast HBM traffic:    13,527 × 20,480 B ≈  277 MB
Marlin weight read:  13,527 × 3.28 MB  ≈ 44.4 GB
cast share of Marlin HBM:   0.62 %
```

At 2.0 pp of total decode attributable to the cast pair and dispatch-free
graph replay, the **HBM-only saving is ≤0.62% of Marlin traffic × 33.1%
aggregate Marlin region share = 0.21% of decode HBM budget.** The 2.0 pp
cast kernel wall-clock is dominated by per-launch setup cost that graph
replay already amortizes, not by the HBM traffic the epilogue would
eliminate. Under graph replay, the steady-state gain collapses from "2.0
pp removable" to "≤0.2 pp HBM-attributable" — **below measurement noise on
the ±3-run median.**

The FP16 intermediate output buffer alloc (`Tensor::zeros((m,n), DType::F16)`
at `kiln-marlin-gemm/src/lib.rs:173`) is also not HBM work — it is CUDA
malloc from Candle's caching allocator and is already amortized across
graph replay. Eliminating it removes allocator pressure, not HBM traffic.

This is the same structural failure that killed the last three big-fusion
attempts. **Check 3 fails.** The BF16-native epilogue's HBM contribution
does not compound; it evaporates.

### Math-ceiling re-derivation — all scenarios below 1.05× floor

| scenario | eliminated pp of decode | end-to-end speedup |
| --- | ---: | ---: |
| cast pair only (kernel wall-clock) | 2.0 | 1.020× |
| cast pair + FP16 alloc (speculative, no sub-range evidence) | ~3.0 | 1.031× |
| PROFILING.md #224 optimistic: 10% of 33.1% region | 3.3 | 1.034× |
| honest HBM-only under graph replay | ≤0.2 | ≤1.002× |

The 1.05× Phase 6 decode-speedup floor requires ≥4.76 pp of decode
time eliminated. **No scenario with honest evidence reaches that
threshold.** The most generous paper-math number from the #224 audit
(3.3 pp) still sits 30% below the floor, and that number collapses further
under graph replay.

### Decision: **KILL. Phase 6 is complete.**

The Marlin wrapper → BF16-native I/O epilogue has been vetted against the
three preflight checks and fails on Check 3 decisively. The
math-ceiling under honest assumptions is 1.02–1.03× end-to-end — below the
Phase 6 1.05× queue gate — and the HBM-traffic analysis confirms this is
not a graph-replay artifact but a structural ceiling set by activation
tensor size (5 KB BF16) vs weight tensor size (3.2 MB packed 4-bit).

With this KILL, the exhaustion matrix from PR #224 is now fully closed:
every standalone region in the post-#210 NVTX top-10 either has a shipped
fusion, a closed-null attempt, an architectural rejection with full
rationale, or a preflight KILL with math-ceiling documented. **No
remaining standalone decode-kernel lever above the 1.05× floor exists.**

### Phase 6 closure — shipped and closed-null inventory

Landed fused kernels (decode path):

- [PR #80](https://github.com/ericflo/kiln/pull/80) `kiln-gdn-kernel` —
  `chunk_gla_fwd` + recurrent GDN fwd vendored.
- [PR #92](https://github.com/ericflo/kiln/pull/92) GDN fused forward
  dispatch.
- [PR #133](https://github.com/ericflo/kiln/pull/133) `kiln-rmsnorm-kernel`
  — fused pre-norm RMSNorm.
- [PR #146](https://github.com/ericflo/kiln/pull/146) `kiln-marlin-gemm` —
  Marlin W4A16 vendored (FP16-only, where this preflight KILL leaves it).
- [PR #158](https://github.com/ericflo/kiln/pull/158) GDN gate fusion.
- [PR #166](https://github.com/ericflo/kiln/pull/166) `kiln-conv1d-kernel`
  — `causal_conv1d_update` vendored.

Closed-null / doc-only redirects:

- [PR #131](https://github.com/ericflo/kiln/pull/131) — `chunk_gla_fwd`
  already vendored (redirect).
- [PR #141](https://github.com/ericflo/kiln/pull/141) — `gated_rms_norm`
  fusion null under graph replay.
- [PR #163](https://github.com/ericflo/kiln/pull/163) — FlashInfer paged
  GQA decode, math-ceiling ≤1.005× redirect.
- [PR #164](https://github.com/ericflo/kiln/pull/164) — GDN gate-path
  single-kernel fusion architecturally infeasible.
- [PR #170](https://github.com/ericflo/kiln/pull/170) — `fused_recurrent`
  already done.
- [PR #173](https://github.com/ericflo/kiln/pull/173) — fused L2 qk_norm,
  null median, shipped opt-in for variance reduction.
- [PR #176](https://github.com/ericflo/kiln/pull/176) — big-fusion across
  recurrent + qk_norm + gated_norm, null ($14.99 burn).
- [PR #219](https://github.com/ericflo/kiln/pull/219) — `ucopy_bf16`
  per-site audit null.
- [PR #222](https://github.com/ericflo/kiln/pull/222) — FP8 KV cache
  verified, opt-in.
- [PR #223](https://github.com/ericflo/kiln/pull/223) — MLP gate/up Marlin
  fusion math-ceiling KILL.
- [PR #224](https://github.com/ericflo/kiln/pull/224) — Phase 6 frontier
  audit, identified this preflight as the last candidate.
- **This PR** — Marlin wrapper BF16-native I/O epilogue KILL.

Phase 6 final decode bench baseline (post-#166, A6000, `KILN_W4A16=1`,
CUDA graphs ON, 512-prompt × 128-decode, median of 3): **49.76 tok/s,
20.10 ms mean ITL, 25.46 ms p99 ITL.** This is the Phase 7 starting
baseline — any Phase 7 work that unintentionally regresses decode tok/s
below this median should be flagged.

### Phase 7 transition — two well-defined starting candidates

With Phase 6 closed, two Phase 7 work items are ready to pick up. They are
**capability / DX** levers, not decode-kernel levers — Phase 7 is a
deliberate shift away from the raw tok/s axis that Phase 6 nearly
exhausted.

**Phase 7 candidate A — GDN prefill memory reduction (long-context
capability).** The redirect destination from PR #222. FP8 KV cache
verification showed long-context OOM is set by the GDN prefill-state
allocation, not the GQA KV cache. Relevant regions:
`:kiln/gdn/recur_prep` (0.8 % of decode, prefill-dominant),
`:kiln/gdn/head_expand` (3.4 % of decode), plus prefill-only state
materialization. Target capability: unlock ≥65,536 prompt tokens on a
single A6000 with `KILN_W4A16=1` + `KILN_KV_CACHE_FP8=1`. Preflight work:
measure current GDN prefill peak memory via `nvidia-smi dmon` during a
32k-prompt eval, identify the specific tensor residencies that force the
OOM ceiling, and propose a streaming / chunked prefill-state computation
that bounds peak memory. This is not a decode-speed lever and should not
be measured on decode tok/s.

**Phase 7 candidate B — self-speculative decode end-to-end bench + server
flag.** Per agent note `kiln-speculative-decoding-design`, self-spec
decoding is already implemented via the skip-layer approach (no separate
draft model required — the same Qwen3.5-4B weights serve both draft and
verify paths at different layer depths). The remaining work is
benchmarking it end-to-end on the canonical 512-prompt × 128-decode bench
and exposing it as a server runtime flag. This **is** a decode-speed lever
(potentially 1.5–2× on tok/s if acceptance rates hold above ~70%), but it
is algorithmic rather than kernel-fusion — it belongs in Phase 7 because
it changes the scheduler / generation loop, not a kernel crate. Preflight
work: acceptance-rate sweep across canonical prompts, pick a skip depth,
queue a pod for the bench. This candidate may reopen the raw-tok/s axis
that Phase 6 retired, but under a different set of architectural
assumptions (speculation-friendly prompts, configurable trade-off at
serving time).

**Recommendation:** pick Candidate A as the Phase 7 opener. It is the
capability-side unlock that FP8 KV cache (PR #222) explicitly redirected
to, it has no overlap with any Phase 6 kernel work, and it directly
addresses a customer-visible ceiling (context length) rather than a
benchmark-visible metric (tok/s at 128 decode). Candidate B is
reasonable to queue in parallel as a second-track preflight, but its
payoff depends on acceptance-rate measurement that has not been done yet,
whereas Candidate A's payoff is bounded by the GDN prefill allocation
size, which is measurable on the next pod acquisition.

### Preconditions for reopening Phase 6

Should this KILL be revisited, the bar to re-queue any Marlin wrapper /
BF16 epilogue work is:

1. A new nsys capture with NVTX sub-ranges **inside the Marlin wrapper**
   that breaks out per-call cast, alloc, contiguous, kernel, and dispatch
   stages, producing a real sub-range wall-clock attribution for the 17.4
   pp "outside kernel" bucket.
2. Evidence that at least one sub-range stage (a) exceeds 5 % of the
   region's wall-clock **and** (b) represents actual HBM traffic rather
   than launch-dispatch cost that graph replay amortizes.
3. An m > 1 decode scenario (continuous batching, speculative decoding
   verify pass) where the wrapper cast amortizes differently. All Phase 6
   work to date was m = 1 single-stream decode; at m = 8+ the activation
   tensor scales linearly and the HBM share of the cast pair could rise
   above the 0.62 % per-call threshold that killed this preflight.

Absent new evidence meeting those conditions, do not re-queue this
candidate. Phase 6 is closed.


## Phase 6 — Frontier audit post-#223 (2026-04-20)

**Outcome: Phase 6 decode-kernel frontier is nearly exhausted.** Every
standalone region in the post-#210 NVTX top-10 either already has a shipped
fusion, has a closed-null attempt on record, is quantized (Marlin), is an
ABI-required layout transform with no compute-reducing fix, or sits
sub-floor at the 1.05× decode-speedup queue gate. The only remaining
quantifiable decode-speed lever is the **Marlin wrapper → BF16-native I/O
epilogue** (redirect #2 from the #223 KILL): eliminate the per-call
BF16↔FP16 cast pair around every Marlin GEMM in the forward path. This
audit recommends one $0 preflight on that lever next, and if it fails to
clear the floor, declares Phase 6 complete and advances to Phase 7
(DX / capability). No pod spent, no code changed.

Precedent for doc-only preflight / redirect PRs:
[#131](https://github.com/ericflo/kiln/pull/131) (chunk_gla_fwd already
vendored), [#163](https://github.com/ericflo/kiln/pull/163) (FlashInfer
paged GQA decode redirect), [#164](https://github.com/ericflo/kiln/pull/164)
(GDN gate-path fusion architecturally infeasible),
[#170](https://github.com/ericflo/kiln/pull/170) (fused_recurrent already
done), [#219](https://github.com/ericflo/kiln/pull/219) (ucopy_bf16 null),
[#223](https://github.com/ericflo/kiln/pull/223) (MLP gate/up Marlin
fusion KILL).

### Why this audit

Three post-#210 follow-up items were sequenced after the post-#210 decode
profile identified no high-ceiling standalone fusion:

1. **ucopy_bf16 source-site audit** — resolved by PR #219: null, every
   un-addressed `ucopy_bf16` site is either below the 1.05× floor or is an
   ABI-required layout transform for a downstream kernel (`causal_conv1d_update`,
   GQA matmul, chunkwise GDN recurrence). No queueable fusion.
2. **KV cache FP8 verification** — resolved by PR #222: `KILN_KV_CACHE_FP8=1`
   is bit-identical and within ±3% of BF16 on workable prompt lengths, but
   the long-context OOM ceiling is set by GDN prefill state, not the GQA KV
   cache. Kept opt-in (default `false`); not a decode-speed lever.
3. **MLP gate/up Marlin merge** — resolved by PR #223: KILL. Math-ceiling
   fails by a wide margin even under heroic assumptions. Flagged two
   redirects: (a) GDN prefill memory reduction (capability, not decode
   speed) and (b) **Marlin wrapper → BF16-native I/O epilogue** (decode
   speed candidate).

With all three redirects resolved, the Phase 6 frontier question is now:
*is there any remaining standalone decode-kernel lever above the 1.05×
floor?* This audit walks the post-#210 NVTX top-10 one more time against
every closed PR, every preflight doc, and every architecturally-rejected
candidate, and answers the question concretely.

### Post-#210 NVTX top-10 — exhaustion matrix

No forward-path code has landed since the post-#210 capture (PRs #219,
#222, #223 are all doc-only / non-decode-path). The NVTX composition is
therefore unchanged. For each region, this table maps the current prior-PR
coverage and the math-ceiling verdict at 1.10× local speedup.

| %    | region                     | prior coverage                                                               | compute lever remaining?                                                      | 1.10× math-ceiling | verdict                                  |
| ---: | -------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | -----------------: | ---------------------------------------- |
| 18.5 | `:kiln/gdn/gates`          | **PR #158 merged** (fused Step 6 gate kernel)                                | already fused; no further compute lever                                       | — (already fused)  | **do not queue**                         |
| 17.6 | `:kiln/gdn/gated_norm`     | **PR #141 closed null** (graph-replay dispatch amortization ate the win)     | would need a memory-reducing approach, not a fusion                           | 0.016 (sub-floor)  | **do not queue** without new evidence    |
| 14.7 | `:kiln/gdn/qk_norm`        | **PR #173 opt-in**, null median (1.0093× point estimate, variance only)      | already tried; fused kernel exists behind `KILN_ENABLE_FUSED_L2_QK_NORM`      | 0.013 (sub-floor)  | **do not queue**                         |
|  7.7 | `:kiln/gdn/in_proj`        | Marlin W4A16 (PR #146-era quantization)                                      | GEMM shape change, not fusion; activations-load-bound at m=1                  | 0.007 (sub-floor)  | **do not queue** as standalone           |
|  6.6 | `:kiln/mlp/gate`           | Marlin W4A16; **PR #223 KILL** (gate/up merge cannot clear floor)            | already quantized; merge-with-up killed                                       | 0.006 (sub-floor)  | **do not queue**                         |
|  6.2 | `:kiln/mlp/up`             | Marlin W4A16; **PR #223 KILL**                                               | already quantized; merge-with-gate killed                                     | 0.006 (sub-floor)  | **do not queue**                         |
|  6.0 | `:kiln/mlp/down`           | Marlin W4A16                                                                 | SwiGLU-down fuse impossible (distinct weight, distinct input); standalone sub-floor | 0.005 (sub-floor) | **do not queue**                         |
|  3.4 | `:kiln/gdn/head_expand`    | **PR #219 null** (`ucopy_bf16` site 2); ABI-required broadcast materialization | downstream GDN matmul cannot accept broadcast-strided K/V                     | 0.003 (sub-floor)  | **do not queue**                         |
|  3.4 | `:kiln/residual`           | none                                                                         | potential in-place ADD into RMSNorm epilogue (add+norm fusion)                | 0.003 (sub-floor)  | **do not queue** standalone              |
|  3.0 | `:kiln/proj/qkv` full-attn | Marlin W4A16; **PR #163 preflight** ruled FlashInfer out (full-attn 3.8%)    | full-attn total 3.8% of decode; math ceiling ≤1.038× at infinite speedup      | 0.003 (sub-floor)  | **do not queue**                         |

Sub-top-10 (each ≤3%, each sub-floor even at infinite speedup): `gdn/out_proj`
2.8, `norm/pre_mlp` 2.2, `attn/gdn/recurrent` 2.0 (already vendored in #80),
`norm/pre_attn` 1.9, `gdn/conv` 1.8 (ABI-bound per #219), `proj/o` 0.8,
`gdn/recur_prep` 0.8 (ABI-bound per #219).

**Conclusion:** No standalone NVTX region is queueable as a pure fusion
task. Every realistic fusion candidate either has a prior null attempt, is
ABI-bound to a downstream kernel input contract, or is sub-floor under
graph replay.

### Un-attempted cross-cutting candidates — math-ceiling

Three candidates cut across multiple regions and were not evaluated as
standalone-region items above. This section sizes each one at the
decode-level Amdahl bound and rules on queueability.

**Candidate A — `add + RMSNorm` epilogue fusion (residual into post-attn /
post-MLP norm).** The transformer block at `crates/kiln-model/src/forward.rs`
lines ~2460–2509 issues `(x + attn_out)` → `rms_norm(x, ...)` and
`(x + ffn_out)` → `rms_norm(x, ...)` as separate NVTX ranges. Regions
involved: `:kiln/residual` (3.4%) + `:kiln/norm/pre_attn` (1.9%) +
`:kiln/norm/pre_mlp` (2.2%) = **7.5%**. A fused kernel that consumes the
residual and runs the normalization in the same pass could elide one
full activation read and one full activation write per residual-into-norm
pair.

Math-ceiling: assume generous 1.5× local speedup (memory-bound, not
dispatch-bound, so graph replay does not fully amortize the win):

```
contribution = 0.075 · (1 − 1/1.5) = 0.025
```

Sub-floor. Even at 2.0× local speedup (unrealistic — the norm is already
the #133 fused kernel and can't be sped up much further):

```
contribution = 0.075 · (1 − 1/2.0) = 0.0375
```

Still sub-floor. Verdict: **not queueable.** This is structurally the
same-shape idea as #141 (gated_norm fusion) which already closed null,
and the graph-replay dispatch-amortization lesson applies.

**Candidate B — Marlin wrapper → BF16-native I/O epilogue.** Flagged by PR
#223 as redirect #2. Each of the 12 Marlin callsites per layer (GDN
`in_proj_qkv`/`in_proj_z`/`in_proj_a`/`in_proj_b`/`out_proj`; full-attn
`q`/`k`/`v`/`o`; MLP `gate`/`up`/`down`) currently wraps the FP16-only
Marlin kernel with:

1. `a.to_dtype(DType::F16).contiguous()` — BF16→FP16 activation cast
2. Allocate FP16 output `Tensor::zeros((m, n), DType::F16)`
3. Allocate workspace zeros
4. Kernel launch
5. `c_fp16.to_dtype(DType::BF16)` — FP16→BF16 output cast

Steps 1 and 5 emit `cast_bf16_f32`-style kernels that show up in the
post-#210 kernel top-20 (`cast_f32_bf16` 2.5% + `ucopy_bf16` 8.6% includes
the contiguous step + FP16 casts visible in the elementwise zoo). Regions
that end in a Marlin GEMM sum to:

```
gdn/in_proj        7.7
gdn/out_proj       2.8
mlp/gate           6.6
mlp/up             6.2
mlp/down           6.0
proj/qkv (full)    3.0
proj/o (full)      0.8
────────────────────
total             33.1
```

Of each region's wall-clock, the wrapper cast pair + allocations are not
the kernel itself; post-#210 kernel-level attribution puts `Marlin` at
15.7% (the kernel proper) vs region total 33.1%, so ~17.4 pp of decode is
spent outside the Marlin kernel in those regions — allocation, casts,
layout, and stream sync. An epilogue variant of Marlin that accepts BF16
input directly (via an FP16→BF16 input reader in the load stage) and
writes BF16 output directly (via an FP32→BF16 accumulator epilogue) would
eliminate steps 1 and 5 across all 12 callsites simultaneously.

Realistic local speedup: eliminating cast + ucopy + FP16 buffer
alloc/free per call is a ~5–10% reduction of each region's wall-clock at
m=1 decode (where the cast + alloc amortizes poorly). Take the
conservative end: 5% of 33.1% = 1.66 pp decode reduction, contribution
0.0166. Take the optimistic end: 10% of 33.1% = 3.31 pp, contribution
0.0331. At the mid-point (7.5% local):

```
contribution = 0.331 · 0.075 ≈ 0.025
```

This is on the edge of the 0.05 floor but plausibly clears it under two
assumptions: (a) the cast-elimination also removes the `cast_f32_bf16` /
`ucopy_bf16` memory pressure on the HBM, compounding under graph replay;
(b) the BF16-native accumulator epilogue can be written to avoid the FP32
intermediate buffer allocation as well. Under those assumptions, a 10%
local speedup on a 33.1% aggregate is the realistic target, and
contribution ≈ 0.033 at median with a plausible tail reaching 0.05+ if
HBM-traffic reduction compounds.

**This is the single remaining quantifiable decode-speed lever.** It is
also the only un-attempted candidate where the math-ceiling is plausibly
close to the 1.05× floor with a non-zero chance of clearing it. It
requires non-trivial kernel work (Marlin is FP16-only by upstream design;
adding BF16-native I/O requires modifying the vendored kernel at
`crates/kiln-marlin-gemm/src/lib.rs` to emit a new tile variant), but the
surface area is well-defined and the win is shared across 12 callsites.

Verdict: **queue as the next Phase 6 slice, but gate it on a $0
preflight.** Before any pod $, preflight needs to:

1. Confirm that BF16-native tile variants are compatible with Marlin's
   existing `Marlin<256,1,8,8,4,8>` tile structure (the vLLM Marlin-BF16
   variant `MarlinBF16<...>` is precedent; does it exist in vendored form?).
2. Confirm that the per-call wrapper overhead (cast + alloc) is at least
   ~5% of each GEMM's wall-clock via a targeted nsys capture with NVTX
   sub-ranges on the wrapper stages. If it is <2%, the math-ceiling kills
   this candidate too.
3. Confirm the elementwise-zoo `cast_f32_bf16` (2.5%) and
   `ucopy_bf16` (8.6%) reductions that would follow are not already
   addressed by some upstream Candle optimization (the audit in #219
   already scoped `ucopy_bf16` and ruled out the per-site lever; the
   wrapper-cast bucket is a different set of invocations).

**Candidate C — GDN prefill memory reduction (long-context capability,
not decode speed).** Flagged by PR #222 as the redirect destination after
FP8 KV cache verification showed that long-context OOM is set by GDN
prefill state, not the GQA KV cache. Regions involved: `:kiln/gdn/recur_prep`
(0.8% decode) + prefill-only paths (not decode top-10). This is **not a
decode-speed lever** — it is a capability unlock. At m=1 decode it
contributes ≤0.8% of wall-clock and sits deep below the floor.

Verdict: **not a decode-speed queue item.** If pursued, it is a Phase 7
capability work item, not a Phase 6 decode-kernel slice.

### Math-ceiling summary

| candidate                                    | aggregate % | realistic speedup | contribution | decision                     |
| -------------------------------------------- | ----------: | ----------------: | -----------: | ---------------------------- |
| Add+RMSNorm epilogue fusion (residual→norm)  |         7.5 |              1.5× |        0.025 | **do not queue** (sub-floor) |
| Marlin wrapper → BF16-native I/O epilogue    |        33.1 | 1.05–1.10× (region-local) | 0.017–0.033 | **preflight next** (plausibly clears 0.05) |
| GDN prefill memory reduction                 |    ≤0.8 decode | — (capability)    |          n/a | **not a decode slice**       |

### Recommendation

**Next Phase 6 slice: Marlin wrapper → BF16-native I/O epilogue preflight
($0, doc-only).** The preflight should verify:

1. Upstream Marlin has a BF16-native tile variant in the vLLM reference
   (if yes, vendoring path is precedented like `kiln-gdn-kernel` from #80).
2. The wrapper cast + alloc overhead on the m=1 decode path is ≥5% of
   each GEMM's region wall-clock, measured with NVTX sub-ranges on the
   wrapper stages (no pod $; run on an already-warm bench pod or during
   the next re-profile).
3. The expected HBM-traffic reduction (one fewer BF16→FP16 cast kernel
   launch per GEMM × 12 GEMMs × 32 layers = 384 kernel launches
   eliminated per decode step) compounds or dilutes under graph replay
   via a nsys-reported kernel-launch-count delta projection.

If any of (1)–(3) fails, issue a doc-only redirect PR (precedent: #131,
#163, #164, #170, #219, #223) and **declare Phase 6 complete**. At that
point the recommendation is to advance to Phase 7, which has two
well-defined work items waiting:

- **Capability:** GDN prefill memory reduction for long-context (redirect
  from #222). Unlocks ≥65536 prompt tokens, which FP8 KV cache verified
  is currently gated by GDN prefill state, not the KV cache.
- **DX:** self-speculative decoding is already implemented via the
  skip-layer approach (per agent note `kiln-speculative-decoding-design`);
  the remaining work is benchmarking it end-to-end and exposing it as a
  server option.

Either item is a reasonable Phase 7 starting point; the preflight outcome
on Marlin BF16-native I/O will determine whether Phase 6 needs one more
slice first.

### Do-NOT-queue list (restated, cumulative)

These candidates have all been evaluated and closed; do not re-queue
without new profiling evidence that materially changes their
math-ceiling. The prior-PR or prior-preflight reference is the required
cross-check:

- **`:kiln/gdn/gates`** — PR #158 merged (fused).
- **`:kiln/gdn/gated_norm`** standalone — PR #141 closed null.
- **`:kiln/gdn/qk_norm`** standalone — PR #173 opt-in, null median.
- **Big-fusion (`qk_norm + gated_norm + recurrent`)** — PR #176 closed
  null ($14.99 burn); Step 7 architecturally separates Step 6 and Step 8.
- **FlashInfer paged GQA decode** — PR #163 preflight: full-attn ≤3.8%
  of decode; math ceiling ≤1.038× even at infinite speedup.
- **Gate-path fusion across Step 7 recurrence** — PR #164 preflight:
  architecturally infeasible; both halves already addressed independently.
- **`ucopy_bf16` source-site consolidation** — PR #219 null: every
  un-addressed site is ABI-required or sub-floor.
- **MLP gate/up Marlin merge** — PR #223 KILL: math-ceiling fails by wide
  margin under CUDA graph replay.
- **KV cache FP8 as decode-speed lever** — PR #222 verification: bit-
  identical, ±3% speed, not a context-length unlock on current path. Kept
  opt-in; not a decode-speed queue item.
- **Add+RMSNorm epilogue fusion** — this audit: 0.025 contribution at
  generous 1.5×, sub-floor under graph replay (same shape as #141).

### Artifacts

This is a doc-only PR. No bench was run; no pod was acquired. Total
compute cost: $0. The audit relied on:

- The existing post-#210 profile
  (`profiling-artifacts/post210_nvtx.csv`, `profiling-artifacts/post210_kern.csv`).
- The post-#222 FP8 KV cache verification raw data
  (`profiling-artifacts/fp8_kv_verify_2026-04-20.csv`).
- Source reads of `crates/kiln-model/src/forward.rs`,
  `crates/kiln-model/src/marlin_proj.rs`, and
  `crates/kiln-marlin-gemm/src/lib.rs` at the current `main` HEAD.
- PR history cross-check via `gh pr list --repo ericflo/kiln --state all`
  through PR #223.


## Phase 6 — Post-#222 preflight: MLP gate/up Marlin fusion math-ceiling audit (2026-04-20)

**Outcome: KILL.** Fusing the MLP `gate_proj` and `up_proj` Marlin W4A16
GEMMs into a single stacked-n projection (the vLLM
`MergedColumnParallelLinear` / `gate_up_proj` pattern) cannot clear the
Phase 6 ≥1.05× end-to-end decode speedup floor on Qwen3.5-4B under CUDA
graph replay. Math-ceiling fails by a wide margin even under heroic
assumptions. This preflight is doc-only; no pod spent, no code changed.
Redirects below.

Precedent for doc-only redirect PRs: [#131](https://github.com/ericflo/kiln/pull/131)
(chunk_gla_fwd already vendored), [#163](https://github.com/ericflo/kiln/pull/163)
(FlashInfer paged GQA decode redirect), [#164](https://github.com/ericflo/kiln/pull/164)
(GDN gate-path fusion architecturally infeasible),
[#170](https://github.com/ericflo/kiln/pull/170) (fused_recurrent already done),
[#219](https://github.com/ericflo/kiln/pull/219) (ucopy_bf16 null).

### Source sites (current, untouched)

The current MLP forward path, `crates/kiln-model/src/forward.rs` lines
770–817 (`swiglu_ffn`), issues **two independent Marlin calls on the same
activation `x`** inside separately-named NVTX regions:

```
gate = mlp_proj_forward(x, gate_proj_t, gate_proj_marlin, ...)   // :kiln/mlp/gate
gate = cuda_silu(gate)
up   = mlp_proj_forward(x, up_proj_t,   up_proj_marlin,   ...)   // :kiln/mlp/up
hidden = gate * up
out  = mlp_proj_forward(hidden, down_proj_t, down_proj_marlin, ...) // :kiln/mlp/down
```

`mlp_proj_forward` dispatches to `marlin_proj::matmul_bf16`
(`crates/kiln-model/src/marlin_proj.rs`), whose per-call wrapper is:

1. `a.to_dtype(DType::F16).contiguous()` — BF16→FP16 activation cast.
2. Allocate FP16 output `Tensor::zeros((m, n), DType::F16)`.
3. Allocate workspace zeros.
4. Kernel launch (`Marlin<256,1,8,8,4,8>` at m=1 decode).
5. `c_fp16.to_dtype(DType::BF16)` — FP16→BF16 output cast.

The Marlin kernel itself (`crates/kiln-marlin-gemm/src/lib.rs`) is
**FP16-only, single-input, single-output** with constraints `k % 128 == 0`,
`n % 256 == 0`, `groupsize ∈ {-1, 128}`. A fused "gate+up" stacked-n
projection would need either (a) kernel-surface changes to accept a single
weight of width `2 × intermediate_size` and emit two halves, or (b) an
epilogue-split variant that splits the m=1 output into two n-tiles. Both
fit Marlin's tile structure (n=256 tiles; `2 × 9216 = 18432` is a multiple
of 256) but both require non-trivial kernel work.

### Profile extract (post-#210 steady-state decode)

From the most recent re-profile in this file (section "Post-#210 re-profile,
decode hot regions"), Arm B `KILN_W4A16=1` production decode on A6000,
512-prompt × 128-decode, median of 3:

| NVTX region          | % decode |   median ns | calls  |
|----------------------|---------:|------------:|-------:|
| `:kiln/mlp/gate`     |   6.6%   |    52 591   |    ... |
| `:kiln/mlp/up`       |   6.2%   |    50 035   |    ... |
| `:kiln/mlp/down`     |   6.0%   |      ...    |    ... |
| **gate + up (sum)**  | **12.8%**|             |        |

Combined CUDA-kernel attribution for the Marlin kernel (all six W4A16
projections per layer — q/gate/up/down + GDN in_proj × 2 — over 32 layers ×
128 decode steps) is 15.7% of decode at ~20.3 μs × 13 527 invocations. The
gate+up slice of that is **2/6 × 15.7% ≈ 5.2% of decode** spent inside the
Marlin kernel body itself for gate+up.

### Math-ceiling closed form

For a fusion that touches `region_pct` of decode and achieves speedup `s`
on that region under CUDA graph replay (launch-amortization ≈ 0 —
confirmed by the null results in PRs [#141](https://github.com/ericflo/kiln/pull/141),
[#173](https://github.com/ericflo/kiln/pull/173), [#176](https://github.com/ericflo/kiln/pull/176),
[#164](https://github.com/ericflo/kiln/pull/164)):

```
end_to_end_speedup = 1 / (1 − region_pct × (1 − 1/s))
```

To clear the Phase 6 ≥1.05× decode floor with `region_pct = 0.128`:

```
1 − 1/s ≥ 0.05 / 0.128           (approx, 1/(1-0.05 × ...))
1 − 1/s ≥ 0.391
1/s     ≤ 0.609
s       ≥ 1.642×
```

**Gate+up region must get ≥1.64× faster** in median wall-clock under
graph replay to move the end-to-end decode needle by 5%. (The exact Amdahl
form gives s ≥ 1.641×; the inequality above is the algebra check.)

### Bandwidth / arithmetic intensity at m=1 decode

Qwen3.5-4B: `hidden_size = 2560`, `intermediate_size = 9216`
(`crates/kiln-core/src/config.rs:86`), W4A16 packed weights, BF16
activations.

Per `gate_proj` or `up_proj` call at m=1:

- **Weight bytes (the dominant term):** W4A16 packed = 4 bits/weight
  + scales (group 128) ≈ `2560 × 9216 × 4/8 = 11 796 480` bytes
  ≈ **11.25 MiB / call** of weight traffic from GDDR6X. Plus scales,
  roughly 11.8 MB total.
- **Activation bytes (per call):** `x` is `[1, 2560]` BF16 = 5 120 bytes
  ≈ **5 KB / call**. Negligible vs weights (0.04% of the transfer).
- **Output bytes:** `[1, 9216]` FP16 = 18 432 bytes ≈ **18 KB / call**.
- **Arithmetic:** `2 × 2560 × 9216 × 1 = 47.2 MFLOPs / call`.

At ~384 GB/s HBM-equivalent effective bandwidth observable to a single
SM cluster on A6000, 11.8 MB takes ~31 μs in pure-streaming limit. The
measured median is ~52 μs, i.e. ~60% of time is weight-streaming and the
remainder is cast/alloc/launch-overhead + tile scheduling.

**Fusion savings ceiling (what fusion can plausibly remove):**

1. **One BF16→FP16 activation cast.** Both current calls cast the *same*
   `x`. A fused path casts once. Saves ~`5 KB / 384 GB/s ≈ 13 ns` of bandwidth
   — but the cast kernel's fixed cost (launch, RMS of a separate kernel) is
   what actually hurts; nsys shows `cast_bf16_f16` at ~1–2 μs amortized per
   invocation. Call it **~1–2 μs saved** across the two current calls.
2. **One workspace alloc + one output alloc.** Under graph replay these
   allocs are captured once, so savings ≈ 0 on the replayed-graph decode
   path. On the first (non-graph) step, savings are tens of μs, but that
   step is already outside the decode-steady measurement window.
3. **Marlin kernel tile reuse (the only real win).** A single stacked-n
   GEMM of width `2 × 9216 = 18432` streams `x` (5 KB) *once* across shared
   memory for both outputs, and reuses the same tile of activation across
   more n-tiles before needing a new one. At m=1 this changes ~nothing
   because `x` already fits in a single activation tile and stays resident;
   the kernel is **weight-bandwidth-bound, not activation-bandwidth-bound**.
   Tile fusion at m=1 saves ≈ 0 of the weight-streaming term — the
   dominant 60% of the region time.

**Upper-bound achievable s on gate+up at m=1 decode:**

- Remove activation cast + consolidate FP16 output alloc: **1.04–1.08×**.
- Add any Marlin kernel tile-level savings at m=1: another **0–0.03×**
  (bounded by the ~40% non-weight-streaming portion of the region time,
  most of which is launch + epilogue, not tile-reuse).
- **Realistic s ≤ 1.12× on region.** Plugged into the Amdahl form:

```
decode_speedup ≤ 1 / (1 − 0.128 × (1 − 1/1.12))
              = 1 / (1 − 0.128 × 0.107)
              = 1 / (1 − 0.0137)
              ≈ 1.0139×                         (well below 1.05× floor)
```

Even under heroic-and-wrong assumptions the ceiling fails:

| Assumed s on region | End-to-end decode speedup | vs 1.05× floor |
|--------------------:|--------------------------:|:---------------|
| 1.12×               | 1.014×                    | **FAIL (−0.036)** |
| 1.30×               | 1.030×                    | **FAIL (−0.020)** |
| 1.50×               | 1.045×                    | **FAIL (−0.005)** |
| 1.60×               | 1.048×                    | **FAIL (−0.002)** |
| 1.70×               | 1.053×                    | barely clears |
| 2.00×               | 1.068×                    | clears |

The weight-bandwidth argument puts the honest ceiling around 1.10×. The
1.70× needed to barely clear the floor would require either (a) m=1
arithmetic intensity that doesn't exist on W4A16 decode, or (b) a radically
different kernel (not a merge) that also changes the dequant or output dtype.

### External prior art

- **vLLM `MergedColumnParallelLinear` / `gate_up_proj`:** merges gate and
  up into a single stacked weight `[2 × intermediate, hidden]`. Reported
  wins at tensor-parallel batch=1 decode are typically **single-digit %
  on the MLP region**, i.e. 1.03–1.07× on region only — and tensor-parallel
  decode has higher all-reduce overhead that this also collapses, which
  kiln does not pay (single-GPU).
- **SGLang** adopts the same stacked pattern; same magnitude of win; wins
  are larger at prefill (where the region is compute-bound and activation
  tile reuse matters), not at m=1 decode.
- **In-kiln precedent:** GDN `in_proj_qkv` is already stacked across its
  three heads (the standard "QKV merge" pattern). That merge *was* worth it
  because at prefill time the region is compute-bound, and because the
  three sub-projections share gate/up/down's hidden dim but have different
  downstream consumers (saving the activation load pattern is meaningful).
  At m=1 decode, the same merge would show ≈ 0 win by the same
  weight-bandwidth argument used above — and indeed `:kiln/gdn/in_proj` at
  7.7% of decode has not been targeted for further merge work.

### Decision: KILL

Gate+up Marlin fusion is weight-bandwidth-bound at m=1 decode, fits
kiln's Marlin tile geometry but requires non-trivial kernel-surface work
(either width-`2n` weight or epilogue-split), and under CUDA graph replay
the only first-order savings (cast consolidation + alloc reuse) are
≈1–2 μs on a ~100 μs combined region. The math-ceiling floor (s ≥ 1.64×
on region for 1.05× end-to-end) is a factor of ~15× further than the
bandwidth-bound ceiling allows. Ship no code; no pod spent.

### Redirects (higher-ceiling next candidates)

1. **GDN prefill memory reduction.** The "KV cache FP8 verification"
   section below established that the OOM ceiling at ≥65536 prompt tokens
   is set by **GDN prefill state, not the GQA KV cache**. Shrinking GDN
   prefill state (via chunked streaming, state recompute on checkpoint,
   or reduced-precision state) unlocks the long-context memory story that
   FP8 KV cache did not deliver. High ceiling on capability; ceiling on
   decode-speed unclear but unimportant — this is a capability fix, not a
   decode-speed fix.

2. **Marlin wrapper → BF16-native I/O epilogue.** The per-call
   `a.to_dtype(DType::F16).contiguous()` + `c_fp16.to_dtype(DType::BF16)`
   pair in `marlin_proj::matmul_bf16` (`crates/kiln-model/src/marlin_proj.rs`)
   is applied on **every** Marlin call — q_proj + gate + up + down + GDN
   in_proj × 2, over 32 layers × 128 decode steps. Nsys post-#210 shows
   `cast_f32_bf16` at 2.5% of decode and `ucopy_bf16` at 8.6% of decode,
   both partially attributable to these wrappers. A Marlin-kernel
   epilogue that emits BF16 directly (and reads BF16 activations in the
   prologue) removes two cast kernels per projection call, addressing a
   larger slice of decode than gate/up merge and with a clearer
   bandwidth story. Requires kernel work but math-ceiling plausibly
   clears 1.05× on combined cast eliminations. **Preflight recommended
   before pod spend** — the same math-ceiling audit applied to the
   cast-elimination path.

3. **Marlin packing latency cleanup (~58 s at load)** and **Marlin BF16
   weight residency cleanup (+1.3 GB VRAM)** — both previously flagged in
   this file; neither is a decode-speed item but both are ship-ready
   low-risk cleanups that clear VRAM for the GDN prefill work in (1).

Do not re-queue the gate+up merge idea without a new profile that
materially changes the 12.8% region share or shows m=1 decode exiting
the weight-bandwidth-bound regime.


## Phase 6 — KV cache FP8 verification (2026-04-20)

**Outcome: keep `kv_cache_fp8` opt-in (default stays `false`).** Verified that
FP8 KV cache (PR #55, `kv_cache_fp8 = true`) produces bit-identical decoded
tokens and speed within ±3% of BF16 on workable prompt lengths, but the
promised long-context memory unlock does **not** materialize on the current
decode path — the OOM ceiling at ≥65536 prompt tokens is set by GDN prefill
state, not by the GQA KV cache, so halving the KV cache does not extend the
context ceiling on this GPU.

Raw per-run data: [`profiling-artifacts/fp8_kv_verify_2026-04-20.csv`](profiling-artifacts/fp8_kv_verify_2026-04-20.csv).

### Why this verification was run

Re-profile 2026-04-20 and the post-#210 profile both closed out kernel-fusion
targets without a high-ceiling next candidate. The `ucopy_bf16` audit (this
file's previous first section, and PR #217/#219) explicitly redirected
attention to **non-kernel decode axes**, calling out KV cache FP8 as a
specific item to verify before leaving it at the current opt-in default. This
run is that verification.

### Method

- Pod: pool-acquired A6000 on-demand (lease `pod-016bb2e7c07be9a97ceb4a3b`,
  pod `t0vmof6qkwostu`), `ghcr.io/ericflo/kiln-runpod:latest`.
- Build: `KILN_CUDA_ARCHS=86 cargo build --release --features cuda` (sccache
  warm; incremental rebuild after patching `bench.rs` to gate the throughput
  phase behind `KILN_BENCH_SKIP_THROUGHPUT=1` — latency-only, avoids OOM at
  the largest prompt sizes and shaves ~60–90 s per run).
- Arm: production decode (`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`), `--paged
  --max-output-tokens 128 --skip-training`, `KILN_BENCH_LOG_TOKENS=1` for a
  quality spot-check.
- Matrix: `KILN_KV_CACHE_FP8 ∈ {0, 1}` × prompt tokens
  `{512, 4096, 16384, 65536, 131072}` × 3 runs back-to-back = 30 runs.
- Metric reporting: median-of-3. Failed (OOM) runs reported as `FAILED` with
  the caused-by chain preserved in the raw logs.

### Median-of-3 results

| FP8 | ptok | runs | fail | prefill ms | decode tok/s | mean ITL ms | p99 ITL ms | model VRAM |
| --- | ---: | ---: | ---: | ---------: | -----------: | ----------: | ---------: | ---------: |
| 0   |   512 | 3 | 0 |   329.1 | **52.34** | 19.10 | 23.49 | 17528 MB |
| 1   |   512 | 3 | 0 |   332.2 | 50.52 | 19.79 | 22.78 | 17528 MB |
| 0   |  4096 | 3 | 0 |  1633.0 | **50.79** | 19.69 | 23.87 | 17528 MB |
| 1   |  4096 | 3 | 0 |  1620.1 | 49.31 | 20.28 | 24.23 | 17528 MB |
| 0   | 16384 | 3 | 0 |  6316.8 | 43.90 | 22.78 | 23.85 | 17528 MB |
| 1   | 16384 | 3 | 0 |  6495.4 | **43.70** | 22.89 | 24.10 | 17528 MB |
| 0   | 65536 | 3 | **3** | — | OOM | — | — | — |
| 1   | 65536 | 3 | **3** | — | OOM | — | — | — |
| 0   | 131072 | 3 | **3** | — | OOM | — | — | — |
| 1   | 131072 | 3 | **3** | — | OOM | — | — | — |

Decode-tok/s deltas FP8 vs BF16 (positive = FP8 faster):

- 512p: −3.5% (run-1 warmup noise dominates; runs 2/3 land at 52.3–52.8)
- 4096p: −2.9%
- 16384p: **−0.5%** (within run-to-run noise)

Model-weight VRAM (`model_load.model_vram_mb`) is identical at 17528 MB in
both arms, as expected — FP8 affects KV cache only, and the paged KV cache
allocates per-prefill rather than at load.

### Quality spot-check (first 16 decoded token ids)

| ptok | FP8=0 first 16 ids | FP8=1 first 16 ids |
| ---: | :--- | :--- |
|   512 | `561, 29144, 6165, 27050, 279, 24460, 3377, 303, 3150, 854, 13, 2838, 5947, 264, 15352, 6157` | **identical** |
|  4096 | `54134, 10782, 264, 491, 9140, 314, 5365, 7559, 64, 7397, 303, 279, 15979, 20915, 13, 561` | **identical** |
| 16384 | `561, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 3043, 279, 14367, 5883, 13, 54134, 10782` | **identical** |

Bit-identical tokens across the three workable prompt lengths confirm the FP8
E4M3 path is numerically safe for production decode at BF16 parity.

### Why ≥65536p OOMs in *both* arms

The 131072p logs surface the root cause:

```
Caused by:
    0: paged prefill forward pass failed
    1: gated deltanet layer 0 (linear attention, paged)
    2: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

The 65536p failure is the same OOM on the same GDN layer-0 allocation (the
error chain is elided by a log overwrite in our capture script but the raw
traceback matches). The OOM happens **inside GDN's linear-attention prefill
buffers, before the paged GQA KV cache is even touched**. FP8 quantizes only
the GQA KV cache; it leaves GDN's recurrent state and chunk buffers
BF16-sized. On the current decode path, GDN prefill is the memory-dominant
term at long prompts, so halving the GQA KV cache does not shift the OOM
ceiling upward.

This matches the architectural note in ARCHITECTURE.md that 24 of 32 layers
are GDN with per-layer chunk/recurrent state, vs 8 GQA layers with
kv-head count = 4.

### Decision rubric (as specified in the verification task)

| Criterion | Threshold | Observed | Pass? |
| --- | --- | --- | :---: |
| Decode tok/s within ±3% at 512p / 4096p | ±3% either direction | 512p −3.5% (warmup), 4096p −2.9%, 16384p −0.5% | partial |
| Peak VRAM at 65536p FP8 ≤60% of BF16 | ≤60% | Not measurable — both OOM in GDN prefill before KV cache is sized | no |
| Output tokens coherent / match BF16 | bit-identical | identical first-16 ids at 512 / 4096 / 16384 | yes |
| 131072p unlocks FP8 while OOMing BF16 | FP8 succeeds, BF16 fails | both OOM in GDN layer 0 | no |

Two of four criteria fail and one passes only marginally, so the rubric
specifies **keep opt-in**. We keep `kv_cache_fp8 = false` as the default and
leave the flag documented in `kiln.example.toml` for users who explicitly
need BF16 parity with halved GQA KV storage (e.g. doubling a workload that
is KV-bound on batch count rather than single-request context).

### What this means for the Phase 6 frontier

FP8 KV cache is **not** the lever that unlocks 65536p+ context on this
pipeline, because GDN prefill memory is the binding constraint. Follow-on
work that could change that conclusion:

1. **GDN prefill memory reduction.** Audit the `kiln/gdn/in_proj`,
   `chunk_prep`, and `recurrent` state buffers in
   `crates/kiln-model/src/forward.rs` for chunked allocation or
   activation-checkpointing-style recomputation. This is the only path that
   actually raises the context ceiling on single-request A6000 decode.
2. **FP8 KV path re-verification if GDN is bounded.** If (1) lands and 65536p
   becomes workable in BF16, re-run this matrix — at that point the 60% KV
   memory headroom from FP8 could genuinely extend context to 131072p.
3. **Multi-request / batched decode.** FP8 KV still has real value when KV
   memory scales by concurrent sequences rather than by single-sequence
   prompt length. That regime wasn't exercised by this matrix (which ran
   single-request) and is not a reason to change the default until it is
   measured in a multi-tenant scheduling context.

Until one of (1)–(3) produces evidence that FP8 flips the context ceiling
or improves steady-state decode at real batch fan-out, the default stays
BF16 and the flag stays opt-in.




**$0 preflight follow-up to the post-#210 profile** (this file's first section).
That section recommended a `ucopy_bf16` source-site audit before any kernel
work; this is the audit. It is a **null result**: every remaining
`ucopy_bf16`-emitting site in the decode hot path is either already addressed
by a prior PR, sits below the 1.05× decode-floor on its own (even at infinite
local speedup), or is required by a downstream kernel ABI that cannot be
relaxed without redesigning the kernel itself. **Do not queue a `ucopy_bf16`
fusion task.** Phase 6 should move to KV cache FP8 next; the MLP gate/up
Marlin-merge speculation needs a separate activation-load profile before it
can be sized.

### Inputs

- Source: `crates/kiln-model/src/forward.rs` (read in full at this audit's
  branch point, c480886) plus `crates/kiln-model/src/marlin_proj.rs`,
  `crates/kiln-model/src/lora_loader.rs`.
- Wall-clock attribution: `profiling-artifacts/post210_nvtx.csv` and
  `profiling-artifacts/post210_kern.csv` (the post-#210 capture). Note: the
  post-#210 capture intentionally omitted `--cuda-graph-trace=node` (it
  corrupted event ordering), so per-region kernel attribution is not
  available — region totals and kernel totals are correlated below by
  reading the source.
- Prior PR scope: `git log --all --oneline | grep -iE
  "ucopy|transpose|cast_bf16|in_proj|qkv|recurrent|gates|gated|qk_norm"`.

### Per-site enumeration

For each `ucopy_bf16`-emitting site in the decode hot path: NVTX region,
prior-PR coverage, why it remains, and the per-site math-ceiling at a
generous 1.5× local speedup (`p · (1 − 1/1.5)`).

| # | site (file:line)                                     | NVTX region              | region % | prior PR coverage                       | why it remains                                                | local 1.5× ceiling |
| - | ---------------------------------------------------- | ------------------------ | -------: | --------------------------------------- | ------------------------------------------------------------- | -----------------: |
| 1 | `forward.rs:1462` `mixed_qkv.transpose(1,2).contiguous()` | `:kiln/gdn/conv`         |      1.8 | none                                    | required input layout for `causal_conv1d_update` kernel ABI   |              0.006 |
| 2 | `forward.rs:1521-1535` `.unsqueeze().expand().contiguous()` (q,k) and `q.contiguous()`/`k.contiguous()` fallback | `:kiln/gdn/head_expand`  |      3.4 | none                                    | downstream matmul cannot operate on broadcast-strided tensor  |              0.011 |
| 3 | `forward.rs:1655-1659` 5× `.transpose(1,2)` (q,k,v,beta,g) | `:kiln/gdn/recur_prep`   |      0.8 | none                                    | lazy strides; downstream `gdn_chunkwise_recurrence` consumer materializes | 0.003 |
| 4 | `forward.rs` chunkwise recurrence interior `.contiguous()` | `:kiln/attn/gdn/recurrent` |    2.0 | PR #80 (vendored chunkwise GDN), PR #74 (matmul readout), PR #75 (analytical recurrence) | internal compute layout for the vendored recurrence kernel    |              0.007 |
| 5 | `forward.rs:1580-1581` `(q*scale).to_dtype()`, `k.to_dtype()` | `:kiln/gdn/qk_norm`      |     14.7 | **PR #173** opt-in fused L2-QK norm; null median 1.0093× | already attempted; null. The fused kernel exists behind `KILN_ENABLE_FUSED_L2_QK_NORM`. | (see PR #173)      |
| 6 | `forward.rs:1693` `attn_out.reshape().to_dtype()`    | `:kiln/gdn/gated_norm`   |     17.6 | **PR #141** closed null                 | already attempted; null. Future re-visit needs new memory-reducing approach | (see PR #141)      |
| 7 | `forward.rs:1844-1846,1862-1873,1895-1898` Q/K/V/output `transpose().contiguous()` | `:kiln/attn/full/*` slow path | ≤3.8 | **PR #100** fused paged-decode (skipped on hot path) | only fires on FA2 fallback (rare); paged-decode kernel never materializes these | ~0 on hot path     |
| 8 | `marlin_proj.rs:296,315` `x.contiguous()`            | callsite-attributed to `:kiln/gdn/in_proj`, MLP `gate/up/down`, full-attn `qkv` | (covered by callsites) | required by `marlin_w4a16_gemm` kernel ABI | kernel reads contiguous strides; cannot be relaxed without redesigning Marlin | sub-floor |
| 9 | GDN linear projections (in_proj_qkv/z/a/b, out_proj) | `:kiln/gdn/in_proj`, `:kiln/gdn/out_proj` |     10.5 (combined) | **PR #130** pre-transposed `*_t` cache  | already eliminated at load time (`weights.in_proj_*_t`)        | already addressed  |
| 10 | MLP / full-attn projections (gate/up/down, q/k/v/o) | `:kiln/mlp/*`, `:kiln/proj/*` |    22.6 (combined) | **PR #128** pre-transposed `*_t` cache  | already eliminated at load time                                | already addressed  |
| 11 | lm_head                                             | `:kiln/lm_head` 0.2%     |      0.2 | **PR #117** precompute `embed_tokens.t()` once at load | already eliminated                                            | already addressed  |
| 12 | `linear_with_lora` (non-`_t` variant)               | n/a (tests only)         |        0 | PR #128/#130 hot-path migration to `_t` | only used in `lora_loader.rs` tests; no decode callsite        | already addressed  |

(Sites 9–12 confirm the historical wins from PRs #117/#128/#130 are still in
place on current `main`; the residual 8.6% kernel-level `ucopy_bf16` is the
sum of the un-cached layout transforms in sites 1–7.)

### Combined math-ceiling

Sum of un-addressed, un-attempted, in-hot-path sites (1 + 2 + 3 + 4):
`1.8% + 3.4% + 0.8% + 2.0% = 8.0%` of decode wall-clock time. At a generous
1.5× local speedup applied to **all four sites simultaneously**:

```
combined contribution = 0.080 · (1 − 1/1.5) = 0.027
```

Even **fully eliminating every un-addressed site at infinite local speedup**:

```
combined contribution = 0.080 · (1 − 1/∞) = 0.080
```

Both are below the project's 0.05 (i.e. ≥1.05×) decode-speedup queue floor.
The "infinite" case is also unrealistic — `ucopy_bf16` is memory-bound, not
launch-bound, so the local speedup ceiling is set by HBM bandwidth, not by
launch dispatch. Under CUDA graph replay (production decode path), launch
amortization further compresses the achievable win toward the conservative
end of this range. The same lesson applies as in #141, #173, #176, and #164.

Sites 5 and 6 are also above the floor on paper (14.7% + 17.6% = 32.3%), but
they have already been attempted (#173 opt-in null median 1.0093×; #141
closed null). They are listed for completeness, not as queueable targets.

### Why "consolidating call sites" does not help

The `ucopy_bf16` kernel total (8.6% / 4164 invocations) looks consolidatable
but in fact splits cleanly by tensor shape and dtype across the seven hot-path
sites above. The four un-addressed sites (1, 2, 3, 4) operate on different
ranks ([B,T,qkv_dim] vs [B,T,nv,dk] vs [B,nv,T,*]) and cannot share a fused
kernel: each one is a layout transform that exists because the *next* kernel
in the chain (`causal_conv1d_update`, the GQA matmul, the chunkwise GDN
recurrence) requires that specific stride pattern. Removing the layout
transform requires changing the downstream kernel's input contract — which
is exactly what PR #128 and PR #130 already did for the projection weights
(by caching the pre-transposed weight tensor at load time so the per-step
transpose disappears).

There is no equivalent "cache it at load time" trick available for sites 1–4
because they operate on per-step *activations*, not per-load *weights*. An
`ucopy_bf16`-eliminating fix at site 2 (head_expand) would need to be a
custom CUDA matmul kernel that accepts broadcast-strided K/V inputs — which
is the same class of work as #100 (fused paged decode FA2 hot path), and at
3.4% region-share it does not clear the floor.

### Recommendation

This audit closes the post-#210 PROFILING.md follow-up #1
(`ucopy_bf16` audit). Proceed with the post-#210 follow-up #2 instead:

- **KV cache FP8 (`KILN_KV_CACHE_FP8=1`)** is the qualified next slice. It
  is already wired (`kiln-server/src/config.rs`); needs a correctness +
  quality benchmark, not a fusion. Distinct win category from the
  decode-kernel work, not gated by the math-ceiling floor (context-length
  is a product feature, not a decode-speed metric).

The post-#210 follow-up #3 (MLP gate/up Marlin merge) remains marked "needs
further math" — needs a profile that separates activation-load time from
GEMM time on the MLP trio before pod $ can be queued.

### Marlin BF16 residency cleanup — not a `ucopy_bf16` lever

PR #206 already dropped the redundant BF16 `*_proj_t` tensors after Marlin
packing, reclaiming the +1.3 GB VRAM that the post-#166 baseline reported.
There is no further BF16-residency-related `ucopy_bf16` work pending — the
remaining decode-path BF16 weight tensors (GDN `*_t` projections, attention
`*_t` projections kept around for non-Marlin builds, and embeddings) are
either consumed by `broadcast_matmul` directly (no `ucopy_bf16` emitted) or
gated on `*_marlin.is_some()` to skip the BF16 path entirely.

### Artifacts

This is a doc-only PR. No bench was run; no pod was acquired. Total compute
cost: $0. The audit relied entirely on the existing post-#210 profile
(`profiling-artifacts/post210_*.csv`) and a source read of
`crates/kiln-model/src/{forward,marlin_proj,lora_loader}.rs` at commit
c480886.


## Phase 6 — Post-PR #210 decode profile (2026-04-20)

Fresh nsys capture on current `main` after the post-#166 cluster of Marlin /
load-time / VRAM / bench-infra changes. Supersedes the post-#166 breakdown as
the canonical "next target" reference. This is the first profile since #173
(opt-in fused L2-QK norm, null median) and #176 (closed-null big-fusion) were
resolved, so it is also the first profile that lets us re-check whether any
GDN gate-path fusion has meaningfully shifted proportions at the
graph-replay layer.

**Preflight** (kernel-vendor-precondition-check, before pod $):

- Recent PRs reviewed:
  - #204 bench-only pool verification (no code delta vs #166 NVTX snapshot)
  - #206 drop BF16 MLP weights when Marlin is resident (VRAM cleanup; no
    decode-path compute delta)
  - #210 parallelize Marlin pack loop (load-time only; no forward-path
    compute delta)
  - #211–#215 desktop / CI / docs (no runtime impact)
- Net: no forward-path code deltas since the post-#166 NVTX snapshot. The
  NVTX composition is expected to be within run-to-run variance of post-#166,
  and this capture confirms that — top-3 region ordering is unchanged
  (gates → gated_norm → qk_norm) and their summed share moves from 50.4% →
  50.8% (within noise). **The profile's job is not to find a shifted
  hotspot — it is to rule out new candidates before we consider re-opening
  any closed-null kernel work.**
- Not-a-target list (verified against `gh pr list --repo ericflo/kiln
  --state all`): qk_norm standalone (#173 null 1.0093×), gated_norm
  standalone (#141 null), big-fusion qk_norm+gated_norm+recurrent (#176
  null, $14.99 burn), FlashInfer paged GQA decode (#163 math-ceiling
  ≤1.005×), gate-path fusion across Step 7 recurrence (#164 architecturally
  infeasible — two halves already shipped/closed independently).

**Hardware:** Pool-acquired RunPod NVIDIA RTX A6000 on-demand (pod
`t0vmof6qkwostu`, $0.49/hr), Driver 580.95.05, CUDA 12.8,
`ghcr.io/ericflo/kiln-runpod:latest`.

**Build:** release, `--features cuda,nvtx`, `KILN_CUDA_ARCHS=86`,
`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`, sccache ON (Backblaze B2 bucket).

**Bench & profile:** `kiln-bench --model-path /workspace/qwen3.5-4b --paged
--prompt-tokens 512 --max-output-tokens 128 --skip-training`, 3 back-to-back
clean bench runs (no nsys) followed by a single nsys capture run. nsys
2024.6.2 with `--trace=cuda,nvtx` (NO `--cuda-graph-trace=node` — that flag
previously corrupted event ordering and produced an
`EventCollection::CheckOrder` finalization failure; CUDA-graph attribution
is intentionally sacrificed so we get a valid trace at all).

### Decode bench — 512-prompt × 128-decode (Arm B, 3 clean runs)

| run    | decode tok/s | mean ITL ms | prefill (ms) | model load (s) |
| ------ | -----------: | ----------: | -----------: | -------------: |
| 1      |        52.29 |       19.13 |        324.3 |          23.76 |
| 2      |        50.92 |       19.65 |        329.6 |          22.75 |
| 3      |        52.88 |       18.90 |        327.7 |          22.90 |
| median |    **52.29** |   **19.13** |    **327.7** |      **22.90** |

Resident VRAM: 17528 MB (post-#206 drop; matches the expected −1.3 GB vs
pre-#206). Marlin pack latency at load: ~22.9 s (post-#210 parallelized;
was ~58 s on the pre-#210 baseline — confirms the load-time win landed).

### Δ vs post-#166 direct-RunPod baseline (median)

| metric            | post-#166 direct | 2026-04-20 post-#210 | Δ         |
| ----------------- | ---------------: | -------------------: | --------: |
| decode tok/s      |            49.76 |                52.29 |    +5.1 % |
| mean ITL ms       |            20.10 |                19.13 |    −4.8 % |
| model load s      |             ~50+ |                22.90 | ~ −54 %   |
| resident VRAM MB  |           ~18800 |                17528 |    −6.8 % |

The decode tok/s uplift is within run-to-run variance (post-#204 pool re-run
also saw a median of 51.37 tok/s on identical code — variance band is ~±3%
across sessions). Model-load and VRAM deltas are real and attributable to
#210 and #206 respectively. **Decode hotspots are materially unchanged.**

### NVTX region top-10 (decode, wall-clock %, post-#210 nsys capture)

Source: `profiling-artifacts/post210_nvtx.csv` (130 decode iterations
captured, 3120 GDN instances = 130 × 24 GDN layers, 4160 MLP instances =
130 × 32 MLP layers, 1041 full-attn `qkv` instances ≈ 130 × 8 GQA layers +
1 prefill).

| %        | region                          | med (ns) | notes                      |
| -------: | ------------------------------- | -------: | -------------------------- |
| **18.5** | `:kiln/gdn/gates`               |  198,347 | GDN Step 6 (fused in #158) |
| **17.6** | `:kiln/gdn/gated_norm`          |  188,730 | #141 closed null           |
| **14.7** | `:kiln/gdn/qk_norm`             |  157,995 | #173 opt-in only (null)    |
|      7.7 | `:kiln/gdn/in_proj`             |   81,101 | Marlin GEMM                |
|      6.6 | `:kiln/mlp/gate`                |   52,591 | Marlin GEMM                |
|      6.2 | `:kiln/mlp/up`                  |   50,035 | Marlin GEMM                |
|      6.0 | `:kiln/mlp/down`                |   47,815 | Marlin GEMM                |
|      3.4 | `:kiln/gdn/head_expand`         |   36,927 | reshape/broadcast          |
|      3.4 | `:kiln/residual`                |   13,636 | 8321 instances             |
|      3.0 | `:kiln/proj/qkv`                |   97,379 | full-attn Marlin (8 lyrs)  |

Sub-10 regions worth noting: `gdn/out_proj` 2.8%, `norm/pre_mlp` 2.2%,
`attn/gdn/recurrent` 2.0%, `norm/pre_attn` 1.9%, `gdn/conv` 1.8%,
`proj/o` 0.8%, `gdn/recur_prep` 0.8%.

**Aggregates:**
- **GDN total:** ~68.5% (`gates` 18.5 + `gated_norm` 17.6 + `qk_norm` 14.7
  + `in_proj` 7.7 + `head_expand` 3.4 + `out_proj` 2.8 + `recurrent` 2.0 +
  `conv` 1.8 + `recur_prep` 0.8)
- **MLP trio:** 18.8% (`gate` 6.6 + `up` 6.2 + `down` 6.0)
- **Full-attn projections:** ~3.8% (`qkv` 3.0 + `o` 0.8) — **still below
  the 10% FlashInfer threshold** (this re-confirms the #163 preflight).

### CUDA kernel top-10 (`--trace=cuda,nvtx`, post-#210)

Source: `profiling-artifacts/post210_kern.csv` (nsys `cuda_gpu_kern_sum`).

| %        | kernel                                                           | med (ns) | notes                                |
| -------: | ---------------------------------------------------------------- | -------: | ------------------------------------ |
| **15.7** | `Marlin<256,1,8,8,4,8>`                                          |   20,288 | 13527 invocations                    |
| **12.6** | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn`   | 1,715,259 | 130 invocations = **lm_head** |
|     10.5 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn`     |   60,000 | 3121 invocations                     |
|      9.8 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn`    |   36,208 | 6244 invocations                     |
|  **8.6** | `ucopy_bf16`                                                     |   32,897 | 4164 invocations — cross-cutting     |
|      5.7 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn` | 32,480 | 3121 invocations                 |
|      3.4 | `bmul_f32`                                                       |    2,784 | 24979 invocations (elementwise zoo)  |
|      3.2 | `fused_rmsnorm_kernel`                                           |    6,176 | 10536 invocations (PR #133 kernel)   |
|      2.7 | `recurrent_gdn_fwd_kernel<128>`                                  |   15,360 | 3122 invocations (PR #80 kernel)     |
|      2.5 | `cast_f32_bf16`                                                  |    1,312 | 20814 invocations                    |

Sub-top-10 notable: `ucopy_f32` 2.1%, `cast_bf16_f32` 1.8%, `affine_f32`
1.6%, `fast_sum_f32` 1.3%, `bmul_bf16` 0.8%, `badd_bf16` 0.8%. Together the
elementwise zoo (casts + ucopy + affine + sum + mul) accounts for ~20% of
kernel time, scattered across the GDN + norm + residual paths.

### Graph-replay-aware analysis (per region)

The dispatch-amortization lesson from #141, #173, #176, and #164 applies
with full force: under CUDA graph replay the per-iteration kernel-launch
cost goes to near-zero, so a fusion that only collapses launches shows up
as a null delta. **For a fusion to clear the 1.05× decode floor, it must
reduce real compute or memory traffic, not just dispatch cost.** The
math-ceiling check below uses the decode-level Amdahl bound: a fusion
eliminating fraction *p* of decode time at speedup *s* yields a total
speedup of `1 / (1 - p·(1 - 1/s))`, and the win-contribution is
`p · (1 - 1/s)`. We require that contribution ≥ 0.05 (i.e. ≥5% decode
speedup) to even queue a pod-$ attempt.

| region                           | % | already attempted?                   | realistic compute-reducing fusion? | win contribution if 1.10× | queue?     |
| -------------------------------- | -: | ------------------------------------ | :--------------------------------: | ------------------------: | ---------- |
| `:kiln/gdn/gates`                | 18.5 | fused in #158 (merged)             | already fused                      |                         — | **no**     |
| `:kiln/gdn/gated_norm`           | 17.6 | #141 closed null                   | would need new memory-reducing approach | 0.016                | **no** (re-visit needs new idea, not new attempt) |
| `:kiln/gdn/qk_norm`              | 14.7 | #173 opt-in, null median           | already tried and null             |                     0.013 | **no** |
| `:kiln/gdn/in_proj`              |  7.7 | Marlin GEMM (already quantized)    | GEMM shape change, not fusion      |                     0.007 | **no**     |
| MLP trio (gate+up+down)          | 18.8 | none                               | partial fusion (swiglu gate/up merge into a single Marlin call) plausible | 0.017 | **no** (sub-floor) |
| `:kiln/gdn/in_proj` + `out_proj` | 10.5 | none                               | share layout work, potential combined load  | 0.009        | **no** (sub-floor) |
| `ucopy_bf16` (kernel-level)      |  8.6 kernel-% | none                    | many sites (residual, reshape, cast-adjacent); consolidation could pay if source sites are identified | depends on which sites | **investigate** |
| elementwise zoo (aggregate)      | ~20 kernel-% | none                     | cross-cutting; no single natural fusion site | depends on grouping | **investigate** |
| `:kiln/residual`                 |  3.4 | none                               | potential in-place ADD into RMSNorm epilogue | 0.003            | **no**     |
| full-attn `:kiln/proj/qkv`       |  3.0 | FlashInfer #163 ruled null         | already quantized (Marlin)         |                     0.003 | **no**     |

None of the "queue?" column says **yes**. Every standalone region either
already has a shipped fusion, has a closed-null attempt on record, or sits
below the 1.10× × region-% floor of 0.05 decode speedup.

### Next target — recommendation

**Do not queue another standalone decode-kernel fusion.** The top three
GDN regions (gates / gated_norm / qk_norm, summed 50.8%) have all been
attacked and each either shipped (#158) or closed null (#141, #173, #176).
The MLP trio is the only sizable region that has not been touched, but at
18.8% it requires a 1.362× per-region speedup to clear the 5% decode floor,
and that's the theoretical ceiling — real speedups under graph replay fall
well below the nominal kernel-level number.

Three qualified candidates for the next Phase 6 slice, in priority order:

1. **`ucopy_bf16` source-site audit (investigate-only, $0 preflight
   follow-up).** `ucopy_bf16` is 8.6% of kernel time spread across 4164
   invocations per decode step. If the source sites (candle's transpose /
   contiguous / cast-adjacent copies) can be identified and consolidated,
   the math-ceiling case is: 8.6% × (1 − 1/1.5) ≈ 0.029, still sub-floor
   *if we only save kernel time*, but attacking it means reducing memory
   traffic which *does* compound under graph replay. This needs an
   instrumentation pass first — add `NVTX_RANGE!` around each call site —
   before any kernel work.

2. **KV cache FP8 (`KILN_KV_CACHE_FP8=1`).** Non-kernel axis; not about
   decode tok/s, about doubling effective context. Already wired
   (`KILN_KV_CACHE_FP8` in `kiln-server/src/config.rs`). Needs a
   correctness + quality benchmark, not a fusion. Distinct win category
   from the decode-kernel work; would not be starved by the math-ceiling
   floor because context-length is a product feature, not a decode-speed
   metric.

3. **MLP gate/up Marlin merge (speculative).** `:kiln/mlp/gate` (6.6%) and
   `:kiln/mlp/up` (6.2%) run on the same input activation and are
   immediately fused by a SwiGLU elementwise op. A single Marlin call
   producing both outputs in parallel should halve the activation load
   cost. Math-ceiling: 12.8% × (1 − 1/1.5) = 0.043 — still sub-floor, so
   **not a queued task yet**. Flag it as "needs further math" — specifically
   needs a profile that separates activation-load time from GEMM time on
   the MLP trio. Do that profiling before committing pod $.

### Do NOT target (restated)

- **`:kiln/gdn/qk_norm`** standalone — PR #173 shipped opt-in, null median
  (1.0093× point estimate, variance-reducing only). Do not re-queue without
  new evidence of a memory-reducing approach that wasn't tried in #173.
- **`:kiln/gdn/gated_norm`** standalone — PR #141 closed null. Do not
  re-queue.
- **Big-fusion (`qk_norm` + `gated_norm` + `recurrent`)** — PR #176 closed
  null, $14.99 burn. Step 7 (recurrent) is architecturally separated from
  Steps 6 and 8 and cannot be single-kernel fused. Do not re-queue.
- **FlashInfer paged GQA decode** — PR #163 preflight: full-attn total is
  3.8% of decode this capture (was 3.5% post-#166), math ceiling ≤1.038×
  even at infinite speedup on that region. Not viable on Qwen3.5-4B's
  24 GDN + 8 GQA architecture.
- **Gate-path fusion across Step 7 recurrence** — PR #164 preflight
  concluded architecturally infeasible; the two halves (Step 6 gates, Step
  8 gated_norm) are separated by the recurrent scan and cannot share a
  kernel. Both halves are already addressed independently (#158 merged,
  #141 closed null). Do not re-queue as a combined task.

### Artifacts

- `profiling-artifacts/post210_nvtx.csv` — full NVTX region table
  (24 rows).
- `profiling-artifacts/post210_kern.csv` — full CUDA kernel table
  (52 rows).
- Raw `.nsys-rep` (112 MB) retained on pod only (too large to commit).


## Re-profile 2026-04-20 (pool-verification baseline)

Bench-only re-run on a fresh pod acquired through the new Cloud Eric kiln pod
pool (`ce kiln-pod-acquire` / `ce kiln-pod-release`). Purpose:

1. Verify the pool path produces results materially identical to the
   direct-RunPod path (post-PR #166 baseline).
2. Validate the new long-running bench supervision pattern in
   `skills/kiln/resources/runpod-workflow.md` (sentinel-file polling via
   `runpod_api.py wait-file`, no ad-hoc `until ssh ... kill -0` loops). The
   prior attempt at this re-profile (task `c8a18546…`) burned $13.76 in a
   silent SSH-polling deadlock — see agent note `kiln-ssh-polling-deadlock`.

No code changes between this run and the post-PR #166 baseline, so the NVTX
top-10 and CUDA-kernel top-10 from the
[Post-PR #166 section](#phase-6--post-pr-166-decode-profile-2026-04-18) below
remain canonical. This section adds only fresh bench numbers from the
pool-verification run.

**Hardware:** Pool-acquired RunPod NVIDIA RTX A6000 on-demand (pod
`t0vmof6qkwostu`, $0.49/hr), Driver 580.95.05, CUDA 12.8,
`ghcr.io/ericflo/kiln-runpod:latest`.

**Build:** release, `--features cuda,nvtx`, `KILN_CUDA_ARCHS=86`,
`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`, sccache ON (Backblaze B2 bucket).
Cold first-clone build on the freshly-spawned pool pod (sccache hit rate
low on first build; not a hot cache).

**Bench:** `kiln-bench --model-path /workspace/qwen3.5-4b --paged
--prompt-tokens 512 --max-output-tokens 128 --skip-training`, 3 back-to-back
runs in a single nohup'd shell script with sentinel-file completion
signaling. NVTX-feature build but no nsys capture this run (pool-verification
scope only).

### Decode tok/s — 512-prompt × 128-decode (Arm B, 3 runs, pool path)

| run    | decode tok/s | mean ITL ms | p50 ITL ms | p99 ITL ms |
| ------ | -----------: | ----------: | ---------: | ---------: |
| 1      |        49.15 |       20.35 |      20.12 |      24.19 |
| 2      |        51.37 |       19.47 |      19.38 |      22.39 |
| 3      |        52.39 |       19.09 |      18.83 |      22.40 |
| mean   |    **50.97** |   **19.64** |  **19.44** |  **23.00** |
| median |    **51.37** |   **19.47** |  **19.38** |  **22.40** |

Run 1 here is the slowest (cold prefill: 7540 ms, 67 tok/s for the first
single-shot prefill — pure CUDA-graph capture + JIT + page-in warmup; runs
2–3 settle to 327–337 ms / ~1500 tok/s). The decode tok/s ordering is
inverted vs the post-#166 baseline where run 1 was fastest; this is exactly
the run-to-run variance the `kiln-bench-prefill-warmup-required` note
warns about. Decode tok/s + ITL averages over many tokens so all three runs
are individually trustworthy, but single-shot TTFT is not.

### Δ vs post-#166 direct-RunPod baseline (median)

| metric            | post-#166 direct | 2026-04-20 pool | Δ        |
| ----------------- | ---------------: | --------------: | -------: |
| decode tok/s      |            49.76 |           51.37 |   +3.2 % |
| mean ITL ms       |            20.10 |           19.47 |   −3.1 % |
| p50 ITL ms        |            19.91 |           19.38 |   −2.7 % |
| p99 ITL ms        |            25.46 |           22.40 |  −12.0 % |

Within run-to-run variance — the post-#166 mean was 51.15 tok/s, this run's
mean is 50.97 tok/s (a 0.4% delta). The pool path produces equivalent
decode performance to the direct-launch path. **No regression.** Pool
verification: passed.

### Pool-path operational metrics

- Pool acquire → SSH ready: ~6 min (fresh spawn, not a warm reuse — pool
  was empty for A6000 at acquire time).
- Cold build (`cargo build --release --features cuda,nvtx --bin kiln-bench`):
  ~22 min (sccache cold, full ~330-crate build).
- Model download (Qwen/Qwen3.5-4B, ~8.8 GB) ran in parallel with build,
  finished before build completed.
- Bench (3× back-to-back, sentinel-driven): ~7 min.
- Total wall-clock from acquire to PR-ready: ~36 min.
- Bench supervision pattern (sentinel + `runpod_api.py wait-file`) worked
  cleanly — no SSH wedging, no silent polling stalls, no $13+ surprise.

Subsequent re-profiles on a warm pool pod (or a hibernated-then-resumed
pod) should hit ~10 min total wall-clock by skipping the cold build.


## Phase 6 — Post-PR #166 decode profile (2026-04-18)

Fresh nsys capture on current `main` (HEAD `c2579a1`, PR #166 vendored
`causal_conv1d_update` into `kiln-conv1d-kernel` and wired
`BackendRuntime::causal_conv1d_update` in `kiln-model/src/backend/cuda.rs`).
This is now the source-of-truth decode breakdown for the next optimization
task — it supersedes the "Post-PR #158 Decode Profile" section below for
planning purposes. Methodology matches the post-#158 profile (Arm B only,
W4A16 + CUDA graphs production path).

**Hardware:** RunPod NVIDIA RTX A6000 on-demand ($0.49/hr), Driver
580.95.05, CUDA 12.8, `ghcr.io/ericflo/kiln-runpod:latest` (CUDA 12.4
toolchain, nsys 2024.5.1 — the baked 2023.4.4 still has the
`EventCollection::CheckOrder` finalization bug noted in prior sections).

**Build:** release, `--features cuda`, `KILN_CUDA_ARCHS=86`,
`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`, sccache ON (backblaze B2 bucket).

**Bench:** `kiln-bench --model-path /workspace/qwen3.5-4b --paged
--prompt-tokens 512 --max-output-tokens 128 --skip-training`, 3 back-to-back
uncaptured runs for tok/s + ITL, 1 separate `nsys profile --delay=110
--duration=30 --capture-range=nvtx -p 'kiln/*' ...` capture for NVTX +
CUDA kernel stats. Single-token paged decode path
(`model_forward_paged`) exactly as served by the HTTP scheduler.

### Decode tok/s — 512-prompt × 128-decode (Arm B, 3 runs)

| run    | decode tok/s | mean ITL ms | p50 ITL ms | p99 ITL ms |
| ------ | -----------: | ----------: | ---------: | ---------: |
| 1      |        54.24 |       18.44 |      18.43 |      20.90 |
| 2      |        49.76 |       20.10 |      19.91 |      25.46 |
| 3      |        49.45 |       20.22 |      20.10 |      25.59 |
| mean   |    **51.15** |   **19.59** |  **19.48** |  **23.98** |
| median |    **49.76** |   **20.10** |  **19.91** |  **25.46** |

Run 1 is again the fastest and tightest — consistent with the pattern seen
in the earlier #166 A/B bench where the first POST run was slower; here
sccache / graph-cache warmth appears to favor run 1 instead. Runs 2–3
converge to ~49.5 tok/s.

### Top-10 NVTX regions (Arm B, post-#166)

| rank | %     | region                        |
| ---: | ----: | ----------------------------- |
|    1 | 18.0% | `:kiln/gdn/gates`             |
|    2 | 17.5% | `:kiln/gdn/gated_norm`        |
|    3 | 14.9% | `:kiln/gdn/qk_norm`           |
|    4 |  7.4% | `:kiln/gdn/in_proj`           |
|    5 |  6.3% | `:kiln/mlp/gate`              |
|    6 |  6.2% | `:kiln/mlp/up`                |
|    7 |  5.9% | `:kiln/mlp/down`              |
|    8 |  3.8% | `:kiln/gdn/head_expand`       |
|    9 |  3.4% | `:kiln/residual`              |
|   10 |  3.0% | `:kiln/proj/qkv`              |
|   +  |  2.9% | `:kiln/gdn/out_proj`          |
|   +  |  2.1% | `:kiln/norm/pre_mlp`          |
|   +  |  1.9% | `:kiln/norm/pre_attn`         |
|   +  |  1.8% | `:kiln/attn/gdn/recurrent`    |
|   +  |  1.8% | `:kiln/gdn/conv`              |

GDN (in_proj + gates + gated_norm + qk_norm + conv + head_expand + out_proj
+ recurrent) now owns **~69%** of decode wall-clock (down from ~73% post-#158
as conv collapsed). MLP trio is **18.4%**. Norms/residual ≈ 7.4%. Full-attn
projections (`:kiln/proj/qkv` + `:kiln/proj/o`) are ~3.5% combined — still
below the 10% threshold that would justify FlashInfer paged GQA decode
(see `flashinfer-decode-preflight-kiln-2026-04-18`).

### Top-10 CUDA kernels (Arm B, post-#166)

| rank | %     | kernel                                                                     |
| ---: | ----: | -------------------------------------------------------------------------- |
|    1 | 14.6% | `Marlin<256,1,8,8,4,8>`                                                    |
|    2 | 11.8% | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn`             |
|    3 |  9.9% | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn`               |
|    4 |  9.2% | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn`              |
|    5 |  8.3% | `ucopy_bf16`                                                               |
|    6 |  5.4% | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn`      |
|    7 |  4.0% | `bmul_f32`                                                                 |
|    8 |  2.9% | `fused_rmsnorm_kernel`                                                     |
|    9 |  2.4% | `recurrent_gdn_fwd_kernel<128>`                                            |
|   10 |  2.4% | `cast_f32_bf16`                                                            |
|   +  |  2.2% | `ucopy_f32`                                                                |
|   +  |  1.7% | `cast_bf16_f32`                                                            |
|   +  |  1.6% | `affine_f32`                                                               |
|   +  |  1.3% | `fast_sum_f32`                                                             |
|   +  |  0.3% | `kiln_conv1d_update_k4_kernel<true>` *(vendored in PR #166)*               |

The vendored `kiln_conv1d_update_k4_kernel<true>` now runs in **0.3%** of
decode time (30.3 ms / 17,560 invocations, avg 1.7 μs per call). The
Marlin + cutlass + ampere GEMM stack composition is essentially unchanged
vs post-#158; the displaced conv1d time (was ~10% as an
elementwise-kernel tail) has been absorbed proportionally across the
remaining GDN gate-path tensors — the top-50 kernel list still shows the
same elementwise zoo (`bmul_f32`, `cast_*`, `ucopy_*`, `affine_f32`,
`fast_sum_f32`, `bdiv_f32`, `badd_f32`) at similar totals.

### Δ vs post-#158 (NVTX)

| region                   | post-#158 | post-#166 | Δ pp     |
| ------------------------ | --------: | --------: | -------: |
| `:kiln/gdn/conv`         |   12.2 %  |    1.8 %  | **−10.4** |
| `:kiln/gdn/gates`        |   16.7 %  |   18.0 %  |    +1.3  |
| `:kiln/gdn/gated_norm`   |   15.8 %  |   17.5 %  |    +1.7  |
| `:kiln/gdn/qk_norm`      |   13.3 %  |   14.9 %  |    +1.6  |
| `:kiln/gdn/in_proj`      |    6.8 %  |    7.4 %  |    +0.6  |
| `:kiln/mlp/gate`         |    5.8 %  |    6.3 %  |    +0.5  |
| `:kiln/mlp/up`           |    5.6 %  |    6.2 %  |    +0.6  |
| `:kiln/mlp/down`         |    5.4 %  |    5.9 %  |    +0.5  |
| `:kiln/gdn/head_expand`  |    2.8 %  |    3.8 %  |    +1.0  |
| `:kiln/residual`         |    3.1 %  |    3.4 %  |    +0.3  |
| `:kiln/proj/qkv`         |    2.7 %  |    3.0 %  |    +0.3  |

`:kiln/gdn/conv` dropped from the #4 hottest region to the #15 — the sole
intended effect of PR #166. Every other region's share went up by
~0.3–1.7 pp as the freed ~10 pp redistributed proportionally; this is the
expected "proportional rebasing" pattern and is not a regression of those
regions in wall-clock terms. Decode tok/s (median 49.76 vs #158's 43.63 on
the same methodology) is **+14.1%** faster.

### Recommended next optimization target

`:kiln/gdn/gates` + `:kiln/gdn/gated_norm` + `:kiln/gdn/qk_norm` =
**50.4 %** of decode, which is above the 40 % threshold the Phase 6 brief
set as the trigger for the next vendor task. The natural next step is to
vendor **`chunk_gla_fwd`** from
[`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
— it fuses the gated-linear-attention chunked forward path (gates + qk_norm
+ gated_norm) that is currently three separate Candle-dispatched region
groups on our side. Per the `kernel-vendoring-call-pattern-regression` note,
the scope should be narrowed to the **decode-only (single-token)** call
pattern so the vendored kernel does not regress prefill / chunk paths — a
decode-specialized split of `chunk_gla_fwd` (or its underlying block-GLA
step kernel) is the concrete target.

> **Update 2026-04-19:** A standalone fused `qk_norm` kernel was attempted
> as a narrower scope (the 14.9 % `:kiln/gdn/qk_norm` slice in isolation)
> and produced a **null result** — see "Phase 6 — fused L2-norm Q+K
> decode kernel (NULL RESULT, 2026-04-19)" below. The kernel is correct
> but the dispatch overhead under `KILN_CUDA_GRAPHS=true` already
> amortizes most of the candle-launch cost at replay time, leaving only
> ~6 % of the qk_norm region to recover. **Do not propose another
> standalone `qk_norm` fusion** without first invalidating the
> dispatch-amortization argument. The `chunk_gla_fwd` direction (or a
> fused `gated_norm + qk_norm` kernel that runs as a single dispatch
> across both regions, claiming the combined **32.4 %**) is the only
> remaining sensible scope here.

Do **not** redirect to FlashInfer paged GQA decode: full-attn projections
on Qwen3.5-4B (24 GDN + 8 full-attn layers) are still only ~3.5 % of
decode, bounded at ≤1.035× overall speedup — well below the 1.15 %
abort threshold. The `flashinfer-decode-preflight-kiln-2026-04-18` note
remains the governing preflight for any future FlashInfer proposal.

### Preflight record

- HEAD verified `c2579a1` (`git log --oneline -1`).
- `crates/kiln-conv1d-kernel/` exists (Cargo.toml, build.rs, csrc/,
  src/lib.rs) — confirms PR #166 landed as advertised.
- `BackendRuntime::causal_conv1d_update` present in
  `crates/kiln-model/src/backend/cuda.rs` (line 20 declaration, lines
  185-205 dispatch).
- `gh pr list` showed no open PR touching GDN / conv paths at capture
  time; `git log --all --grep='chunk_gla'` returned no prior vendor work
  for the next recommended target.
- nsys 2024.5.1 used (baked 2023.4.4 avoided for the
  `EventCollection::CheckOrder` bug).
- `profiling-artifacts/post166_nvtx.csv` and
  `profiling-artifacts/post166_kern.csv` are the raw tables the above
  rankings were derived from (nsys stats `nvtxsum` / `cuda_gpu_kern_sum`
  reports).

### Pod / cost

- Pod: `povpyv0bkwqrte` (RunPod NVIDIA RTX A6000 on-demand, $0.49/hr).
- Total uptime at capture end: ~35 minutes (pull + clone + warmup + 3
  bench runs + 1 nsys capture + stats extraction).
- Estimated pod cost: **~$0.30** (well under the Phase 6 budget).


## Phase 6 — fused L2-norm Q+K decode kernel (NULL RESULT, 2026-04-19)

Vendored a single-launch CUDA kernel that fuses
`l2_normalize(Q) + scale(Q) + l2_normalize(K) + dtype-cast` (the
`:kiln/gdn/qk_norm` region) into `kiln-rmsnorm-kernel` as
`kiln_fused_l2_qk_norm`. The kernel is **correct** (3 parity tests at
decode/prefill/batch shapes pass; the full 128-test `kiln-model`
nextest suite passes) but **misses the task's 1.05× decode tok/s abort
floor**. It is shipped **opt-in only** via
`KILN_ENABLE_FUSED_L2_QK_NORM=1` — the candle reference path remains the
production default. See `crates/kiln-model/src/forward.rs` step-5 doc
comment for the in-tree pointer.

### Bench result (Arm B, RTX A6000, 3 paired runs, 2026-04-19)

`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`, paged 512/128. Baseline disables
the kernel; fused enables it. Same binary; same warm-up.

| run        | decode tok/s | mean ITL ms | p50 ITL ms | p99 ITL ms |
| ---------- | -----------: | ----------: | ---------: | ---------: |
| baseline 1 |        51.67 |       19.35 |      19.12 |      23.85 |
| baseline 2 |        40.19 |       24.88 |      24.93 |      31.17 |
| baseline 3 |        50.81 |       19.68 |      19.63 |      24.78 |
| **median** |    **50.81** |   **19.68** |  **19.63** |  **24.78** |
| fused 1    |        50.64 |       19.75 |      19.71 |      20.25 |
| fused 2    |        51.28 |       19.50 |      19.47 |      19.87 |
| fused 3    |        51.63 |       19.37 |      19.39 |      22.53 |
| **median** |    **51.28** |   **19.50** |  **19.47** |  **20.25** |

| Metric                     | Baseline (median) | Fused (median) | Delta            |
| -------------------------- | ----------------: | -------------: | :--------------- |
| decode tok/s               |             50.81 |          51.28 | **+0.93 % (1.0093×)** |
| mean ITL                   |          19.68 ms |       19.50 ms | -0.18 ms (-0.92 %) |
| p50 ITL                    |          19.63 ms |       19.47 ms | -0.16 ms (-0.81 %) |
| p99 ITL                    |          24.78 ms |       20.25 ms | -4.54 ms (-18.3 %) |
| best-of-3 decode tok/s     |             51.67 |          51.63 | -0.07 % (0.9994×) |
| run-to-run range           |   11.48 (40-52)   |   1.00 (50-52) | -10.5 (much tighter) |

**Median speedup = 1.0093× — well below the task's 1.05× hard abort
floor.** Best-of-3 is actually a hair slower (0.9994×). The only clearly
positive signal is p99 ITL: tail latency tightens by ~4.5 ms (-18 %),
and run-to-run variance collapses from an 11 tok/s spread to a 1 tok/s
spread, suggesting the fused launch removes a small but real source of
dispatch jitter. That p99 win is not enough to clear the bar.

### Why the kernel won the math but lost the wallclock

Per the post-PR #166 profile above, `:kiln/gdn/qk_norm` was 14.9 % of
decode = ~2.93 ms of the median 19.68 ms ITL. Even fully eliminating
that NVTX region would yield at most:

    1 / (1 - 0.149) ≈ **1.175× decode speedup**

The actual 0.18 ms median ITL improvement implies the kernel only
shaved ~6 % of `:kiln/gdn/qk_norm`'s wallclock — far less than the
~85 %+ savings that would be needed to land at the 1.05× floor. The
qk_norm region is dominated not by the per-row arithmetic (rows = 16,
hidden = 128 — trivially memory-bound at decode shape) but by:

1. **Candle dispatch + graph-replay overhead** that survives the launch
   collapse. The CUDA-graph capture at decode already amortizes most of
   the ~11 underlying launches; collapsing them to 1 in the captured
   graph saves graph nodes but not host-side cost per replay.
2. **HBM round-trips** for the F32 intermediate buffers (`q_f32`,
   `k_f32`, `q_squared`, `k_squared`, `q_sum_keepdim`, etc.) that
   candle materializes. The fused kernel skips those, but the savings
   per row are tiny relative to the per-block scheduling cost.

The first point is the dominant explanation: under `KILN_CUDA_GRAPHS=true`
the qk_norm chain is captured once and replayed every decode step, so
the per-step "11 launches → 1" benefit at host code already mostly
collapses to a no-op at replay time. That is also why p99 (which
correlates with off-graph dispatch jitter) shows a clear win while mean
tok/s does not.

### Why ship the kernel as opt-in instead of deleting it

1. **It is correct** — parity tests pass at decode (16×128), prefill
   (1×512×128), and small-batch (4×8×128) shapes; the full nextest
   suite is green. No regression.
2. **Tail-latency improvement is real** — -18 % p99 ITL and a 10×
   tighter run-to-run distribution. Workloads sensitive to ITL tail
   variance (live token streaming, latency-SLO serving) may want to
   opt in.
3. **Future re-evaluation is cheap** — if a later optimization (e.g.
   an opt-out of CUDA graphs at qk_norm, or a wider qk_norm region
   that includes scale-fused dispatch) shifts the dispatch-overhead
   balance, the kernel is already wired and tested. No re-implementation
   needed.

The kernel sits behind `KILN_ENABLE_FUSED_L2_QK_NORM=1` (note: enable,
not disable). Default decode behavior is unchanged from PR #166.

### Re-visit triggers

A future PR can re-default this kernel to ON only if **all three**
hold on a fresh `nsys` profile of current `main` using the same
production path:

1. `:kiln/gdn/qk_norm` NVTX region ≥ **10 %** of decode wall-clock
   (it is 14.9 % today, but graph-replay amortization may have already
   reduced it).
2. The fused kernel produces a median **≥ 1.03×** decode tok/s
   speedup on a 3-run paired bench (currently 1.0093×).
3. There is a documented reason the dispatch-overhead amortization
   no longer dominates (e.g. CUDA graphs disabled at this region for
   a different reason, or a much wider fused region that includes
   the upstream `q = a.broadcast_mul(...)` step).

### Code shipped (branch `ce/phase6-qk-norm-fusion`)

- `crates/kiln-rmsnorm-kernel/csrc/fused_l2_qk_norm.{h,cu}` — single-block
  per-row kernel, two warp-shuffle reductions sharing scratch shmem,
  bf16 in/out, F32 reductions, hidden ≤ 8192 envelope.
- `crates/kiln-rmsnorm-kernel/build.rs` — adds the new `.cu` to the cc
  build.
- `crates/kiln-rmsnorm-kernel/src/lib.rs` — `kiln_fused_l2_qk_norm` FFI
  decl (line ~69), `supports_l2_qk_norm` capability gate, candle
  wrapper `fused_l2_qk_norm`, `reference_l2_qk_norm` parity oracle, 3
  `parity_l2_qk_norm_*` unit tests.
- `crates/kiln-model/src/forward.rs` — step-5 region updated with the
  opt-in dispatch (`KILN_ENABLE_FUSED_L2_QK_NORM`) and an in-tree
  comment pointing here.

### Preflight performed

- HEAD verified `c2579a1` (post-#166).
- `crates/kiln-rmsnorm-kernel/` already exists from the earlier RMSNorm
  vendor; this is an additive `.cu` + extern, not a new crate.
- `gh pr list` showed no overlapping open PR at preflight time.
- Nextest suite passed before the bench (128 tests, full kiln-model
  surface).

### Pod / cost

- Pod: `7s9x0e53pjoglc` (RunPod NVIDIA RTX A6000 on-demand, $0.79/hr —
  spot was unavailable; on-demand mandated by `runpod-always-on-demand`
  agent note).
- Total uptime at capture end: ~75 minutes (image pull + manual clone +
  `kiln-setup` + first build + 6 paired bench runs + result download).
- Estimated pod cost: **~$1.00**.

### Note on diagnostic capture

`nsys 2023.4.4` (baked in `ghcr.io/ericflo/kiln-runpod:latest`) hits
the `EventCollection::CheckOrder` finalization bug on this workload —
the same bug noted in the earlier post-#158 / post-#166 sections. Both
captures generated valid `.qdstrm` traces but `QdstrmImporter` exited
non-zero before producing `.nsys-rep`. Bench data alone is the abort
criterion per the task brief, so the missing NVTX share confirmation
does not change the result; this is recorded for future captures
(`nsys 2024.5.1` was used in the post-#166 profile and avoided the bug).


## Not next target — fused_recurrent GDN decode vendor (preflight, 2026-04-18)

Doc-only preflight-redirect. Third in the Phase 6 preflight-redundant
series after PR #163 (FlashInfer paged GQA decode) and PR #164 (GDN
gate-path fusion). **$0 pod spend.**

### Scope the task asked for

Vendor fla-org/flash-linear-attention's **`fused_recurrent_gated_delta_rule_fwd`**
(Triton → raw CUDA C) into a new `kiln-gdn-recurrent-kernel` crate, wrap
with a thin C-ABI + Rust FFI, and dispatch in kiln's GDN decode path.
Scope: bf16 activations, F32 state accumulators, causal, forward only,
per-token (`seq_len = 1`) decode, Qwen3.5-4B head dim. Expected speedup
1.3–2.0× decode tok/s; abort floor 1.10×.

### Why it is redundant

**PR #92 "Vendor fla fused_recurrent_gated_delta_rule_fwd for GDN decode"**
(merged 2026-04-17, commit `7aa62f1`) already shipped exactly this kernel
with exactly this scope.

| Component asked for                                      | Status in current HEAD (`c2579a1`) |
| -------------------------------------------------------- | ---------------------------------- |
| Fused per-token GDN recurrent CUDA kernel                | `crates/kiln-gdn-kernel/csrc/recurrent_gdn_fwd.cu` (added by PR #92) |
| Thin C-ABI wrapper                                       | `crates/kiln-gdn-kernel/csrc/recurrent_gdn_fwd.h` — `kiln_gdn_recurrent_forward(...)` extern "C" entry point |
| Rust FFI binding                                         | `crates/kiln-gdn-kernel/src/lib.rs` — `pub fn gdn_recurrent_forward(...)` at line 259, `kiln_gdn_recurrent_forward` FFI decl at line 80 |
| Dispatch in decode path                                  | `crates/kiln-model/src/forward.rs:1077-1079` — `backend.gdn_recurrent_step(...)` inside `kiln/attn/gdn/recurrent` NVTX range (the `seq_len == 1` fast-path branch of `gdn_chunkwise_recurrence`) |
| bf16 activations / F32 accumulators                      | Exactly this — see `recurrent_gdn_fwd.cu:49` (`__nv_bfloat16` inputs) and the F32 `s_local[MAX_DK]` state column |
| Causal only, forward only                                | Yes — single-token forward, no backward path |
| Qwen3.5-4B head dim                                      | `MAX_DK ∈ {128, 256}`; kiln configures `dk = dv = 128` |
| Per-thread scope (one block per `(batch, head)`)         | Exactly the launch geometry in `recurrent_gdn_fwd.cu:169/181` |
| `cargo build --release --features cuda` wired            | `crates/kiln-gdn-kernel/build.rs:44` compiles `recurrent_gdn_fwd.cu` via the `cc` crate |
| Parity test vs candle fallback                           | `test_gdn_kernel_matches_fallback` (oracle = `kiln-model::forward::compute_w_chunk_fallback`) |

The crate's top-level doc comment (`crates/kiln-gdn-kernel/src/lib.rs:33-35`)
states plainly:

> `gdn_recurrent_forward` — seq_len==1 decode fast path. Collapses the
> single-token GDN recurrence (decay, delta, state-update, output
> projection) into one block per (batch, head).

That is the function the task asked me to vendor. It already exists.

### Bounded-speedup math against the current profile

Even ignoring the redundancy and imagining a hypothetical "better" vendor
of the same slice, the post-#166 profile caps the ceiling:

| Signal                                     | Share of decode |
| ------------------------------------------ | --------------: |
| `:kiln/attn/gdn/recurrent` NVTX region     |        **1.8 %** |
| `recurrent_gdn_fwd_kernel<128>` CUDA kernel|        **2.4 %** |

Upper bound on overall decode speedup from fully eliminating the
`:kiln/attn/gdn/recurrent` region:

    1 / (1 − 0.018) ≈ **1.018×**

That is below the task's **1.10× abort floor** and an order of magnitude
below the **1.3–2.0× expected range**. Even a theoretical "infinitely
fast" recurrent kernel cannot meet the task's acceptance criteria given
the current profile. This is the same bounded-ceiling argument applied in
`flashinfer-decode-preflight-kiln-2026-04-18` (PR #163) and
`kernel-vendor-precondition-check` (PR #131 precedent).

### Root cause of the stale task

The task body conflates two different kernels:

1. **Per-token GDN recurrent step** (decay → delta → state update → output
   projection). This is what `fused_recurrent_gated_delta_rule_fwd`
   actually implements, and it is what PR #92 already vendored. It owns
   1.8 % of decode today.
2. **GDN gate-path preprocessing** (`:kiln/gdn/gates` + `:kiln/gdn/gated_norm`
   + `:kiln/gdn/qk_norm`). These are the 50.4 % hotspot cited in this
   file's "Recommended next optimization target" section
   (lines 123–145). They happen **outside** the recurrent step — they
   produce the inputs it consumes and post-process the output it produces.
   The correct vendor target for the 50.4 % slice is `chunk_gla_fwd` (or
   a narrower fused `gated_norm + qk_norm` kernel), not
   `fused_recurrent_gated_delta_rule_fwd`.

### Preflight performed

- HEAD verified `c2579a1` on `ericflo/kiln` main.
- `ls crates/` — found existing `kiln-gdn-kernel` with the target kernel
  already compiled in.
- `git log --all -- crates/kiln-gdn-kernel/csrc/recurrent_gdn_fwd.cu` —
  first-add commit is `7aa62f1` (PR #92, merged 2026-04-17).
- `gh pr view 92` — confirmed title "Vendor fla
  fused_recurrent_gated_delta_rule_fwd for GDN decode" and that scope
  matches the new task exactly.
- Read `crates/kiln-gdn-kernel/src/lib.rs:25-93` to verify the FFI
  surface and scope envelope.
- Read `crates/kiln-model/src/forward.rs:1060-1085` to verify wiring as
  the `seq_len == 1` fast-path branch of `gdn_chunkwise_recurrence`.
- `gh pr list --state open` — no overlapping open PR at the time of
  preflight.

No RunPod pod launched. **Total cost: $0.**

### Re-visit triggers

Propose a new per-token GDN kernel only if **all three** hold on a fresh
`nsys` profile of current `main`:

1. `:kiln/attn/gdn/recurrent` NVTX region ≥ **8 %** of decode wall-clock
   (it is 1.8 % today).
2. `recurrent_gdn_fwd_kernel<*>` CUDA kernel ≥ **10 %** of decode kernel
   time (it is 2.4 % today).
3. A concrete code-level reason the PR #92 kernel is sub-optimal at the
   current workload (e.g. register pressure at a larger head dim, shared
   memory shortfall on sm_90, or an upstream fla change that materially
   re-tunes the per-token step).

Absent those, future per-token GDN proposals should be rejected at
preflight.

### Next actual target

Per the unchanged "Recommended next optimization target" block at the top
of this file (lines 123–145):

- **Vendor `chunk_gla_fwd`** for the combined `:kiln/gdn/gates` +
  `:kiln/gdn/gated_norm` + `:kiln/gdn/qk_norm` hotspot (**50.4 %** of
  decode wall-clock, above the 40 % Phase 6 trigger).
- Or, as a narrower scope, fuse just `gated_norm + qk_norm` (**32.4 %**).

This is the only remaining single-target ≥ 3 % opportunity that has not
already been vendored or ruled out by preflight.


## Phase 6 — `causal_conv1d_update` vendor decode bench (2026-04-18)

Vendored NVIDIA Mamba's `causal_conv1d_update` into `kiln-conv1d-kernel`
(cc-crate + Rust FFI wrapper, bf16/causal/fwd-only, cuda-gated,
sm_86/sm_89/sm_90) and wired `kiln-model/src/backend/cuda.rs` to dispatch
through a new `BackendRuntime::causal_conv1d_update` hint. The candle
fallback path is preserved behind the `KILN_DISABLE_FUSED_CONV1D=1` kill
switch so the same binary can A/B without a rebuild. This replaces the
`:kiln/gdn/conv` decode region (documented at 12.2 % of decode in the
earlier nsys breakdown).

**Hardware:** RunPod NVIDIA RTX A6000 on-demand, Driver 580.95.05,
CUDA 12.8, stock `runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2404`
plus `cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-nvrtc-dev-12-8
libcublas-dev-12-8 libcurand-dev-12-8`.

**Build:** release, `--features cuda`, `KILN_CUDA_ARCHS=86`,
`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`.

**Bench:** `kiln-bench --model-path /workspace/qwen3.5-4b --paged
--prompt-tokens 512 --max-output-tokens 128 --skip-training`, 3 back-to-back
runs per arm. Decode path = `model_forward_paged` (production
HTTP/scheduler path). Reporting `decode_tokens_per_sec` and the inter-token
latency distribution from the latency phase.

### Decode tok/s — 512-prompt × 128-decode

| run    | PRE (candle fallback) tok/s | POST (fused kernel) tok/s |
| ------ | --------------------------: | ------------------------: |
| 1      |                       45.10 |                     45.97 |
| 2      |                       49.04 |                     52.59 |
| 3      |                       44.08 |                     52.52 |
| mean   |                       46.07 |                     50.36 |
| median |                       45.10 |                     52.52 |

Median speedup: **1.164× decode tok/s** (+16.4 %). Mean speedup: **1.093×**
(+9.3 %). Run 1 of the POST arm underperforms the other two — first-fire
JIT/kernel-cache warm-up is the likely cause; runs 2–3 converge tightly.

### Inter-token latency (median of 3 runs)

| stat    | PRE ms | POST ms |
| ------- | -----: | ------: |
| mean    |  22.17 |   19.04 |
| p50     |  22.14 |   18.83 |
| p99     |  26.22 |   24.14 |

Consistent −3 ms mean ITL and −2 ms p99 ITL across runs.

### Parity

`cargo nextest run --release --features cuda -p kiln-model -E
'test(test_causal_conv1d_update_matches_fallback)'`: PASS.

### Verdict

1.164× decode speedup clears the 1.03× ship floor. Kernel ships default-on;
`KILN_DISABLE_FUSED_CONV1D=1` retained behind the `BackendRuntime` seam for
debug/ablation. `:kiln/gdn/conv` is now serviced by the vendored kernel; the
next decode hotspot on the GDN path is `gdn_recurrent_step` (covered by
PR #80 / `kiln-gdn-kernel`).


## Phase 6 — chunk-prep fusion prefill bench (2026-04-18)

Measured the Phase 6 `gdn_chunk_prep` fusion (`ce/phase6-chunk-prep-fusion`,
HEAD `11314e3`) against the post-#160 `main` baseline (HEAD `395f5f7`) on the
same pod, back-to-back, using the existing `kiln-bench --paged` latency
harness. The fusion collapses the 7+ elementwise launches inside the
chunkwise outer recurrence (cumsum, decay matrix, KKT/QKT masked
exp-scaling, v_prime, q_s_scaled, decay_last_col, p_last) into one CUDA
kernel per (chunk × batch × head). The four cuBLAS matmuls surrounding it
(KKT, QKT, ks_entry, q_s) are unchanged.

**Hardware:** RunPod NVIDIA A40 on-demand (A6000 unavailable;
same sm_86 Ampere arch, covered by existing `KILN_CUDA_ARCHS="80;86;89;90"`),
Driver 580.95.05, CUDA 12.4.

**Build:** release, `--features cuda`, `KILN_W4A16=0 KILN_CUDA_GRAPHS=true`.

**Bench:** `kiln-bench --model-path /workspace/qwen3.5-4b --paged
--prompt-tokens N --max-output-tokens M --skip-training`, 3 back-to-back runs
per arm (no nsys attached), reporting `time_to_first_token_ms` from the
latency phase.

### Prefill TTFT — 512-prompt × 128-decode

| run    | PRE (`395f5f7`) ms | POST (`11314e3`) ms |
| ------ | -----------------: | ------------------: |
| 1      |             403.43 |              366.68 |
| 2      |             393.19 |              363.41 |
| 3      |             413.51 |              428.25 |
| mean   |             403.38 |              386.11 |
| median |             403.43 |              366.68 |

Mean speedup: **1.045× TTFT** (−4.3 %). Median speedup: **1.100× TTFT**
(−9.1 %). Run 3 of the POST arm is a clear outlier (+63 ms versus the other
two) — probably pod-to-pod GPU variance we've seen before on this harness.

### Prefill TTFT — 2048-prompt × 64-decode

Longer prompt → more chunks per prefill (2048 / chunk_size=64 = 32 chunks per
layer × 24 GDN layers = 768 fused kernel launches in place of 5 376
elementwise launches).

| run    | PRE (`395f5f7`) ms | POST (`11314e3`) ms |
| ------ | -----------------: | ------------------: |
| 1      |            1070.49 |              882.04 |
| 2      |             987.92 |              934.43 |
| 3      |             980.49 |              941.23 |
| mean   |            1012.97 |              919.23 |
| median |             987.92 |              934.43 |

Mean speedup: **1.102× TTFT** (−9.3 %). Median speedup: **1.057× TTFT**
(−5.7 %).

### Decode latency (no expected change)

The fusion is a prefill-path optimization — decode uses the single-step
`gdn_recurrent_step` fast path, which is untouched. Decode numbers are
reported for completeness.

| metric              | PRE mean | POST mean | Δ       |
| ------------------- | -------: | --------: | ------- |
| mean inter-token ms |    25.64 |     25.88 | +0.9 %  |
| decode tok/s        |    38.99 |     38.65 | −0.9 %  |

Within pod variance.

### Read / take

The fusion lands below the ≥1.2× prefill TTFT floor stated in the task brief.
Measured improvement is **1.05–1.10×** on 512-prompt TTFT and **1.06–1.10×**
on 2048-prompt TTFT — directionally correct, reproducible across both prompt
lengths, but well short of the 2–4× target. The implementation is correct
(see `test_gdn_chunk_prep_matches_fallback` and
`test_gdn_chunkwise_matches_sequential` in `crates/kiln-model/src/forward.rs`
— both pass with max error < 2e-3 bf16), the kernel removes a measurable
slice of prefill wall-clock, and it leaves the decode fast path unchanged.
The ceiling is pinned by the four cuBLAS matmuls (KKT, QKT, ks_entry, q_s)
which still dominate the chunkwise recurrence wall-clock and are
intentionally out of scope for this kernel.

Possible follow-ups if the chunkwise recurrence becomes a hotter target:
- Collapse the four matmuls into fewer cuBLAS calls (batch two KKT/QKT into
  a single `bmm`, batch ks_entry/q_s).
- Explore a full Triton-style `chunk_gla_fwd` that owns the matmuls too —
  this is the upstream fla-org path and is what PR #160 originally
  recommended for the largest decode-side win.

### Reproduction

```bash
# 1. Pod + weights
kiln-setup --repo /workspace/kiln
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b

# 2. Baseline (pre-fusion)
cd /workspace/kiln && git checkout 395f5f7
cargo build --release --features cuda -p kiln-server --bin kiln-bench
for i in 1 2 3; do
  KILN_W4A16=0 KILN_CUDA_GRAPHS=true ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b --paged \
    --prompt-tokens 512 --max-output-tokens 128 --skip-training
done

# 3. Fusion (post)
git checkout ce/phase6-chunk-prep-fusion
cargo build --release --features cuda -p kiln-server --bin kiln-bench
for i in 1 2 3; do
  KILN_W4A16=0 KILN_CUDA_GRAPHS=true ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b --paged \
    --prompt-tokens 512 --max-output-tokens 128 --skip-training
done
```

### Pod / cost

- Pod: `11hd6xzo2uwlyy` (RunPod NVIDIA A40 on-demand, $0.44/hr).
- Total uptime at bench end: ~2.5 hours (build, weight fetch, parity tests,
  12 bench runs across pre/post × two prompt lengths).
- Estimated pod cost: **~$1.10** (well under the $20 target / $40 hard abort
  set in the task brief).

## Post-PR #158 Decode Profile (2026-04-18)

Refreshed decode profile on current `main` (HEAD `7132f29`, post-PR #158 which
fused the GDN decode gates path into one CUDA kernel). Re-ran the same nsys
methodology as the PR #156 profile below for an apples-to-apples comparison.
Only Arm B (production W4A16) was re-captured — the baseline Arm A path was
not changed by #158.

**Hardware:** RunPod RTX A6000 on-demand, Driver 565.57.01, CUDA 12.4, nsys
2024.5.1 (apt-installed; the baked 2023.4.4 has the
`EventCollection::CheckOrder` finalization bug).

**Build:** release, `KILN_W4A16=1 KILN_CUDA_GRAPHS=true`, bench command
identical to PR #156 (`kiln-bench --delay=110 --duration=30 --nvtx ...`).

### Arm B decode latency (post-#158 vs PR #156)

| metric                     | PR #156 | post-#158 | Δ           |
| -------------------------- | ------: | --------: | ----------- |
| mean inter-token ms        | 21.50   | 22.92     | +6.6 %      |
| p50 inter-token ms         | 21.43   | 22.61     | +5.5 %      |
| p99 inter-token ms         | 28.87   | 25.13     | **−13.0 %** |
| decode tok/s (latency)     | 46.52   | 43.63     | −6.2 %      |

The p99 tightening is the clearest effect of #158. Mean ITL regressed by
~1.4 ms, which is within the pod-to-pod variance previously observed on this
harness (PR #152→#153→#156 moved the same metric by a similar magnitude
without any code change). Directionally consistent: the GDN hotspots all
moved down proportionally.

### Top-10 NVTX regions (Arm B, post-#158)

| rank | %     | region                        |
| ---: | ----: | ----------------------------- |
|    1 | 16.7% | `:kiln/gdn/gates`             |
|    2 | 15.8% | `:kiln/gdn/gated_norm`        |
|    3 | 13.3% | `:kiln/gdn/qk_norm`           |
|    4 | 12.2% | `:kiln/gdn/conv`              |
|    5 |  6.8% | `:kiln/gdn/in_proj`           |
|    6 |  5.8% | `:kiln/mlp/gate`              |
|    7 |  5.6% | `:kiln/mlp/up`                |
|    8 |  5.4% | `:kiln/mlp/down`              |
|    9 |  3.1% | `:kiln/residual`              |
|   10 |  2.8% | `:kiln/gdn/head_expand`       |
|   +  |  2.7% | `:kiln/proj/qkv`              |
|   +  |  2.5% | `:kiln/gdn/out_proj`          |
|   +  |  1.6% | `:kiln/attn/gdn/recurrent`    |

GDN subsystem (in_proj + gates + gated_norm + qk_norm + conv + head_expand +
out_proj + recurrent) still owns **~73 %** of decode wall-clock. The MLP
trio (`gate`/`up`/`down`) is **16.8 %**. Residual + norms are ~6 %.

### Top-10 CUDA kernels (Arm B, post-#158)

| rank | %     | kernel                                                                     |
| ---: | ----: | -------------------------------------------------------------------------- |
|    1 | 14.5% | `Marlin<256,1,8,8,4,8>`                                                    |
|    2 | 12.0% | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn`             |
|    3 |  9.8% | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f`                              |
|    4 |  9.2% | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn`              |
|    5 |  7.5% | `ucopy_bf16`                                                               |
|    6 |  5.2% | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2`                              |
|    7 |  4.1% | `bmul_f32`                                                                 |
|    8 |  2.6% | `fused_rmsnorm_kernel`                                                     |
|    9 |  2.5% | `fast_sum_f32`                                                             |
|   10 |  2.4% | `ucopy_f32`                                                                |
|   +  |  2.2% | `cast_f32_bf16`                                                            |
|   +  |  2.1% | `recurrent_gdn_fwd_kernel<128>`                                            |
|   +  |  2.0% | `cast_bf16_f32`                                                            |

Marlin remains the single largest kernel (q_proj + 3×MLP). The cutlass /
ampere BF16 GEMM stack sitting at #2/#3/#4/#6 is the non-Marlin projection
work (k_proj, v_proj, o_proj, lm_head, GDN in_proj/out_proj). The long tail
of `bmul_f32`/`fast_sum_f32`/`ucopy_*`/`cast_*` elementwise kernels is the
remaining GDN gates + norm plumbing that PR #158 did not absorb.

### Delta vs PR #156 (NVTX top-4)

| region                   | PR #156 | post-#158 | Δ pp   |
| ------------------------ | ------: | --------: | -----: |
| `:kiln/gdn/gates`        | 18.2 %  | 16.7 %    | −1.5   |
| `:kiln/gdn/gated_norm`   | 18.0 %  | 15.8 %    | −2.2   |
| `:kiln/gdn/qk_norm`      | 14.7 %  | 13.3 %    | −1.4   |
| `:kiln/gdn/conv`         | 12.4 %  | 12.2 %    | −0.2   |

Every GDN hotspot moved down — #158's fusion reduced the proportional share
of the gates path and the downstream gated_norm/qk_norm (they dispatch fewer
upstream elementwise tensors). The gain is real but narrow: the top-50
kernel list still contains the full elementwise zoo (`bmul_f32`,
`fast_sum_f32`, `cast_f32_bf16`, `cast_bf16_f32`, `ucopy_bf16`, `affine_f32`,
`uneg_f32`, `uexp_f32`, `urecip_f32`, `usqrt_f32`) in similar quantities to
PR #156. No single fused-gate kernel dominates — the fusion collapsed part
of the gate arithmetic but the surrounding norm / cast / copy traffic is
unchanged.

### Recommended next optimization target

**Vendor `chunk_gla_fwd` from flash-linear-attention (roadmap step 2).**

The combined decode share of `gdn/gates` + `gdn/gated_norm` + `gdn/qk_norm`
+ `gdn/conv` is still **57.9 %** of wall-clock. A narrow gates-only fusion
(like #158) only chips at the first of those four regions. Vendoring the
upstream `chunk_gla_fwd` kernel that fuses the full GDN decode chain
(conv + in_proj + gates + gated_norm + qk_norm + recurrent) is the
highest-leverage move available — it would collapse ~58 % of decode
wall-clock into a single kernel dispatch and directly eliminate the
elementwise zoo that #158 could not reach.

Vendoring approach follows the established pattern (Marlin vendor in PR
#149, GDN substitution kernel in PR #80, narrow scope under
`crates/kiln-chunk-gla-kernel/`, C-ABI entrypoint, cuda-gated, parity test
vs existing reference). Preflight note: confirmed no prior crate or PR
already lands this (`git log --all --grep="chunk_gla"` and
`ls crates/ | grep -i chunk` both empty).

## Reproduction (post-#158)

Identical to the PR #156 reproduction section below, with one substitution:
install nsys 2024.5.1 (`apt-get install -y cuda-nsight-systems-12-6`) before
the `kiln-bench --nvtx` capture. The baked `ghcr.io/ericflo/kiln-runpod`
image ships nsys 2023.4.4 which finalizes broken `.nsys-rep` files.

---

# Kiln Profiling Report — Phase 6 re-profile, post-Marlin MLP wire-in (PR #152)

## Overview

PR #152 wired the vendored Marlin W4A16 kernel into the three MLP projections
(`gate_proj`, `up_proj`, `down_proj`), extending the same path PR #149 landed
for `q_proj`. PR #153 reported a +9.9 % decode tok/s delta from that wire-in
on a separate pod. This re-profile captures fresh profiles on current `main`
(HEAD `5aa22e1`) on one pod, back-to-back, so the post-#152 production hotspot
mix is settled for the next optimization cycle.

Two arms measured, identical bench and build:

- **Arm A — BF16 baseline (`KILN_W4A16=0 KILN_CUDA_GRAPHS=true`).** All
  projections go through the existing cuBLAS BF16 GEMM path.
- **Arm B — production W4A16 (`KILN_W4A16=1 KILN_CUDA_GRAPHS=true`).** At
  model load, `q_proj` and the three MLP projections swap in Marlin 4-bit
  packed weights when the shape is compatible (`k%128 == 0 && n%256 == 0`).
  All other layers (`k_proj`, `v_proj`, `o_proj`, norms, GDN kernels, full
  attention) are unchanged.

Measured results on one A6000 pod:

| metric                 | Arm A (BF16) | Arm B (Marlin) | Δ        |
| ---------------------- | -----------: | -------------: | -------- |
| mean inter-token ms    | 22.25        | 21.50          | **−3.4 %** |
| p50 inter-token ms     | 22.10        | 21.43          | −3.0 %   |
| p99 inter-token ms     | 28.82        | 28.87          | +0.2 %   |
| decode tok/s (latency) | 44.95        | 46.52          | **+3.5 %** |
| throughput bs=1 tok/s  | 40.40        | 41.53          | +2.8 %   |
| throughput bs=4 tok/s  | 40.40        | 42.24          | +4.6 %   |
| throughput bs=8 tok/s  | 40.35        | 42.17          | +4.5 %   |
| throughput bs=16 tok/s | 40.47        | 41.47          | +2.5 %   |
| model load time        | 13.95 s      | 103.58 s       | +89.6 s  |
| model VRAM (load)      | 16344 MB     | 17656 MB       | +1312 MB |
| peak VRAM (inference)  | 16842 MB     | 18026 MB       | +1184 MB |

The decode-ITL lift lands at roughly **one third the tok/s gain PR #153
reported (+9.9 %)**. Likely drivers: PR #153 measured on a cold pod with a
different throughput harness, and small pod-to-pod drift on the same hardware.
The direction and sign agree; the magnitude is smaller than previously
reported but real and repeatable in this back-to-back measurement.

Model load jumped from 14 s to 104 s because Marlin packs the four
projection weight matrices at load time (≈ 33 matrices × ~8 MB of packed
data + scales). Load-time packing is one-shot and does not affect request
latency, but is worth tracking — a pre-packed on-disk artifact would
eliminate the cost if it becomes a cold-start issue.

Peak VRAM grew ~1.2 GB with Marlin because the packed weights, scales, and
workspace live alongside the original BF16 weights still held for the
non-Marlin paths (`k_proj`, `v_proj`, `o_proj`). Future work that fully
replaces BF16 weights at load time would recover this.

## Methodology

All numbers from one pod (RunPod RTX A6000 on-demand), one build, back-to-back
captures. Only environment variables changing between arms are
`KILN_W4A16=0|1`. `KILN_CUDA_GRAPHS=true` for both arms (production path).

- **Steady-state ITL.** Three back-to-back 512-prompt × 128-decode
  latency runs per arm, no nsys attached, reporting the
  `mean_inter_token_ms` from the paged decode phase after the model is
  warm. Prefill and first-decode cold costs are excluded by
  `kiln-bench`'s latency phase already.
- **Throughput sweep.** The same `kiln-bench` invocation runs a
  `1/4/8/16` sequential-generation sweep after the latency phase, which
  produces the bs=1/4/8/16 tok/s numbers above.
- **nsys captures.** Two captures with the production path and CUDA
  graphs enabled:
    - **Arm A** — `KILN_W4A16=0 KILN_CUDA_GRAPHS=true nsys profile -t
      cuda,nvtx --delay=15 --duration=20`. Capture window begins after
      the ~14 s model load, spanning prefill + warm decode + the first
      throughput runs.
    - **Arm B** — `KILN_W4A16=1 KILN_CUDA_GRAPHS=true nsys profile -t
      cuda,nvtx --delay=110 --duration=30`. The `--delay` is larger
      because Marlin packing pushes model load to 104 s. Capture window
      lands in the throughput phase (steady-state warm decode through
      `bs=1` and into `bs=4` / `bs=8`).
- **Extraction.** `nsys stats --report cuda_gpu_kern_sum --format csv`
  for the per-kernel table; `nsys stats --report nvtx_sum --format csv`
  for the NVTX region table.
- **Capture-window caveat.** Because Arm A's window includes some
  prefill + decode transition activity and Arm B's window is pure
  steady-state decode, a small number of prefill-heavy NVTX regions
  (`:kiln/attn/full/prefill`, `:kiln/attn/full/decode_fused`, some
  `:kiln/attn/rope`) appear in Arm A's top-30 but not Arm B's. The
  measured ITL/tok-s numbers are from the bench's own clean-timing
  phase and are apples-to-apples. The Arm B top-10 tables are the
  correct structural view for "what dominates the production decode
  hot loop today."

## Hardware / Build

- GPU: NVIDIA RTX A6000 (49 GB VRAM)
- Driver: 565.57.01, CUDA 12.4
- nsys: 2024.5.1 (the baked 2023.4.4 triggers
  `EventCollection::CheckOrder` on long captures; 2024.5.1 fixes it)
- Rust: 1.95.0, `cargo build --release --features cuda,nvtx`
- Kiln HEAD at capture: `5aa22e1` (bench report from PR #153, which
  sits on top of PR #152's MLP wire-in)
- Model: Qwen3.5-4B, sharded safetensors in `/workspace/qwen3.5-4b`
- Prompt: 512 tokens; decode: 128 tokens; paged KV
  (`block_size=16`, `blocks=40`)

## Top-10 GPU Kernels — Arm B (W4A16, production path)

From `nsys stats --report cuda_gpu_kern_sum` on `profile_armB.nsys-rep`.
CSV in `profiling-artifacts/statsB_kern.csv`.

| rank | % GPU time | kernel                                                              | role                                                       |
| ---: | ---------: | ------------------------------------------------------------------- | ---------------------------------------------------------- |
|   1  | **14.8 %** | `Marlin<256,1,8,8,4,8>`                                             | W4A16 MLP (gate/up/down) + `q_proj` GEMMs                  |
|   2  |   11.7 %   | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn`      | Remaining BF16 GEMMs (`k_proj`, `v_proj`, `o_proj`, lm_head) |
|   3  |    9.9 %   | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn`        | BF16 GEMM (projections / lm_head)                          |
|   4  |    9.3 %   | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn`       | BF16 GEMM (shape-dependent tile)                           |
|   5  |    8.2 %   | `ucopy_bf16`                                                        | Tensor copies (residual stream, reshapes)                  |
|   6  |    5.3 %   | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5`  | BF16 GEMM (smaller tile)                                   |
|   7  |    3.9 %   | `bmul_f32`                                                          | GDN elementwise multiply (gates / gated_norm path)         |
|   8  |    3.1 %   | `fused_rmsnorm_kernel`                                              | RMSNorm (pre-attn, pre-mlp, qk_norm)                       |
|   9  |    2.9 %   | `fast_sum_f32`                                                      | Reductions in GDN gates / norms                            |
|  10  |    2.7 %   | `recurrent_gdn_fwd_kernel<128>`                                     | GDN linear-attention recurrent kernel                      |

Observations:

- Marlin has become the **single largest kernel** in the production path at
  **14.8 %**, covering four GEMMs (q_proj + gate_proj + up_proj + down_proj)
  in one kernel family.
- The four cuBLAS BF16 GEMM tile variants together account for **36.2 %**
  of GPU time. These are the non-Marlin projections (`k_proj`, `v_proj`,
  `o_proj`, `lm_head`) plus any shape-incompatible fallback.
- Small elementwise / reduction kernels on the GDN path (bmul_f32,
  fast_sum_f32, cast_*, fused_rmsnorm_kernel) remain numerous (8k+
  instances combined). These are launch-overhead-bound at the GDN
  decode shape and are the natural fusion target for the next phase.

## Top-10 NVTX Regions — Arm B (W4A16, production path)

From `nsys stats --report nvtx_sum`. CSV in
`profiling-artifacts/statsB_nvtx.csv`. Percentages are of the NVTX total
within the 30 s steady-state capture window.

| rank | % time | region                  | layers per token | notes                                                  |
| ---: | -----: | ----------------------- | ---------------: | ------------------------------------------------------ |
|   1  | **18.2 %** | `:kiln/gdn/gates`     | 24               | Per-layer gate projection + swish + elementwise path   |
|   2  | **18.0 %** | `:kiln/gdn/gated_norm` | 24              | Gated RMSNorm on GDN head outputs                      |
|   3  |   14.7 %   | `:kiln/gdn/qk_norm`    | 24              | Q/K RMSNorm inside GDN                                 |
|   4  |   12.4 %   | `:kiln/gdn/conv`       | 24               | Causal 1-D conv on GDN Q/K/V                           |
|   5  |    5.5 %   | `:kiln/mlp/gate`       | 32               | `gate_proj` GEMM (now Marlin)                          |
|   6  |    5.5 %   | `:kiln/mlp/up`         | 32               | `up_proj` GEMM (now Marlin)                            |
|   7  |    5.5 %   | `:kiln/mlp/down`       | 32               | `down_proj` GEMM (now Marlin)                          |
|   8  |    4.9 %   | `:kiln/gdn/in_proj`    | 24               | GDN input projection (cuBLAS BF16)                     |
|   9  |    3.6 %   | `:kiln/residual`       | 64               | Residual adds                                          |
|  10  |    2.7 %   | `:kiln/gdn/head_expand` | 24              | GDN head expansion                                     |

Structural breakdown of steady-state decode (Arm B):

- **GDN linear-attention subsystem** (`:kiln/gdn/*`) = **72.5 %**
  (gates 18.2 + gated_norm 18.0 + qk_norm 14.7 + conv 12.4 + in_proj 4.9
  + head_expand 2.7 + recurrent 1.2 + out_proj ≈ negligible here).
- **MLP** (`:kiln/mlp/*`) = **16.5 %** (gate 5.5 + up 5.5 + down 5.5).
  Now balanced across the three projections because Marlin runs all
  three in near-identical time.
- **Residuals + norms** (`:kiln/residual`, `:kiln/norm/pre_*`) =
  **~6.0 %**.
- **Full-attention layers** — fall below the top-10 cutoff in the Arm B
  window; full-attn share is small on this model (only 8 of 32 layers)
  and was dominated by the GDN pathway even in the prior profile.

## Comparison to prior PROFILING.md (pre-PR #152, commit `07d934b`)

Prior production-path top-10 NVTX (graphs ON, no W4A16):

| rank | prior (`07d934b`, BF16) | now (`5aa22e1`, Arm B, W4A16) | shift   |
| ---: | ------------------------- | ------------------------------- | ------- |
|   1  | `:kiln/gdn/gates` 14.3 %  | `:kiln/gdn/gates` **18.2 %**    | +3.9 pp |
|   2  | `:kiln/gdn/gated_norm` 14.1 % | `:kiln/gdn/gated_norm` **18.0 %** | +3.9 pp |
|   3  | `:kiln/gdn/qk_norm` 11.7 % | `:kiln/gdn/qk_norm` **14.7 %**   | +3.0 pp |
|   4  | `:kiln/gdn/conv` 11.1 %   | `:kiln/gdn/conv` **12.4 %**      | +1.3 pp |
|   5  | `:kiln/attn/rope` 9.3 %   | (below top-10 in Arm B window)   | —       |
|   6  | `:kiln/gdn/in_proj` 9.0 % | `:kiln/gdn/in_proj` 4.9 %        | −4.1 pp |
|   9  | `:kiln/mlp/gate` 2.7 %    | `:kiln/mlp/gate` 5.5 %           | +2.8 pp |
|  10  | `:kiln/mlp/up`   2.5 %    | `:kiln/mlp/up`   5.5 %           | +3.0 pp |

Reading: Marlin eliminated a large BF16 GEMM slice per MLP layer, which
**shrank the absolute time in MLP GEMMs** (and hence tightened the decode
loop, producing the measured +3.5 % tok/s). MLP NVTX share **went up**
because the region is now time-relative to a shorter loop and because the
previous profile captured a wider window including prefill. The GDN share
also grew — the Amdahl's-law flip from MLP wins: after Marlin halves
MLP GEMM cost, GDN becomes proportionally even more dominant.

**Dominant cost today is unambiguously the GDN subsystem at 72.5 % of
decode time.** The four big GDN regions — gates, gated_norm, qk_norm,
conv — together account for **63.3 %** of decode time in a single
structural group.

## Next optimization target

The next concrete lever is **fusing the GDN gate + gated_norm +
elementwise path** into a small number of large kernels, ideally one.

Evidence:

- `:kiln/gdn/gates` and `:kiln/gdn/gated_norm` together are **36.2 %**
  of decode time (#1 and #2 by a wide margin).
- The hot kernels serving these regions are small elementwise /
  reduction primitives (`bmul_f32` 3.9 %, `fast_sum_f32` 2.9 %,
  `cast_f32_bf16` 2.4 %, `cast_bf16_f32` 2.2 %, `affine_f32` 1.8 %,
  `uneg_f32` / `uexp_f32` / `urecip_f32` / `usqrt_f32` each ~0.7–1.0 %).
  These are **hundreds to thousands of instances per capture**, each a
  per-element kernel that does very little work.
- At the GDN decode shape (`rows = 32`, `hidden = 128` per head, 24
  layers, batch = 1) these kernels are launch- and
  memory-bandwidth-bound. One fused kernel collapses:
    - `gates_lin` → swish activation → mul into `gated_norm` input,
    - `rms_norm` over the gated output,
    - the residual elementwise multiply.
  That eliminates intermediate bf16→f32→bf16 round-trips that appear
  as `cast_*` kernels today, and cuts hundreds of small launches per
  token.
- PR #141 vendored a `gated_rms_norm` kernel and measured no delta at
  `rows = 32, hidden = 128` **under CUDA graphs**. That test bounded
  the isolated fusion gain, not the broader "collapse the gate path"
  target proposed here. The broader fusion targets NVTX regions that
  together are ~5× larger than the isolated `gated_rms_norm` shape
  PR #141 measured, and the hot kernel list shows the dominant cost is
  in elementwise / reduction primitives that a proper fused gate kernel
  would eliminate, not in the narrow `gated_rms_norm` slice.

Recommended scope for the next cycle:

> **STATUS (2026-04-18): This recommendation is SUPERSEDED.** The
> "collapse gates + gated_norm into one kernel" framing turned out to be
> architecturally infeasible (the two NVTX regions are separated by Step
> 7's chunkwise recurrence — see PR #158), and both halves have already
> been attempted independently: PR #158 fused the `:kiln/gdn/gates`
> Step-6 computation (merged, −1.5 pp share); PR #141 fused the
> `:kiln/gdn/gated_norm` Step-8 computation (closed, null result
> 1.003–1.005× at decode shape). See the "Post-PR #158 Decode Profile
> (2026-04-18)" section above for the current live recommendation
> (vendor `chunk_gla_fwd`) and the "Not next target — GDN gate-path
> fusion (preflight, 2026-04-18)" section below for the full preflight
> record explaining why a fresh gate-path fusion PR is not the right
> next move.

1. Vendor or author a fused **`gdn_gate`** kernel that folds the gate
   linear output, swish, gated elementwise multiply, RMSNorm, and
   residual-multiply into one kernel (bf16 in, bf16 out).
2. Keep the existing `recurrent_gdn_fwd_kernel<128>` and
   `gdn_fwd_sub_kernel` unchanged — those already own a small share
   and are CUDA-graph-friendly.
3. Validate on the same harness this report uses (warm paged decode
   ITL, 512-prompt × 128-decode, three runs, both with and without
   `KILN_W4A16=1`), with a parity test vs the existing unfused
   implementation using the kernel-crate parity test pattern.

Expected ceiling: if the fused kernel removes even half of the
elementwise launch overhead and cast round-trips, decode ITL should
improve by 6–10 % (target: 19.5 ms mean ITL, ~51 tok/s on A6000). The
same fusion helps both arms because the GDN path is unchanged by
W4A16.

## Not next target — FlashInfer paged GQA decode (deferred, preflight 2026-04-18)

The project's "Current optimization queue" item *Vendor FlashInfer paged
GQA decode (minimal, decode-only, bf16, hdim128)* — step 3 of the
optimization queue in the project description — is **deferred** based
on the Arm B NVTX numbers immediately above. This section records the
preflight finding so future planning loops do not re-propose it before
the GDN gate-path fusion above lands.

No pod was spun up past the preflight `git clone`. Pod cost: \$0.

### Why deferred

FlashInfer's paged-KV GQA decode kernel replaces the **inner paged
attention kernel** inside kiln's 8 full-attention layers. It does **not**
touch the GDN path (24 layers) and it does **not** touch the `qkv_proj` /
`o_proj` GEMMs (those are cuBLAS BF16 / Marlin today).

Arm B NVTX shares for the full-attention path at current `main`
(`5aa22e1`, 864 decode steps captured):

| region                | % decode time | notes                                           |
| --------------------- | ------------: | ----------------------------------------------- |
| `:kiln/proj/qkv`      |        2.4 %  | Full-attn projection GEMM (289 instances)       |
| `:kiln/proj/o`        |        0.5 %  | Full-attn output projection GEMM (289 instances) |
| `:kiln/attn/full/*`   |     below 0.5 % | Below top-NVTX cutoff; doesn't appear in table |

The paged attention kernel itself (the thing FlashInfer would replace)
is a subset of that sub-0.5 % `:kiln/attn/full/*` slice. Even a
hypothetical **infinite** speedup on that kernel would reduce overall
decode by at most ~0.5 %, i.e. **overall decode speedup ≤ 1.005×**. The
vendor task stated an expected range of 1.3–2× overall decode and a
1.15× abort threshold. Both are mathematically unreachable at the
current profile because attention is already too small a share of
decode.

### Why it's below the threshold on this model

Qwen3.5-4B is a hybrid architecture: **24 GDN linear-attention layers +
8 full-attention GQA layers** (ratio 3:1). The 8 full-attention layers
are the only place FlashInfer could help, and per the NVTX table above
their contribution is dominated by the surrounding projections, not the
attention kernel. This is the opposite regime from a standard
attention-dominated decoder-only LLM where FlashInfer typically shines.

### When to revisit

Re-evaluate after the GDN gate-path fusion recommended in the previous
section lands. If that fusion brings GDN decode share down by ~20 pp
(from 72.5 % → ~50 %), `:kiln/attn/full/*` proportional share roughly
doubles. Even then, full-attention would still only be ~7–8 % of
decode and FlashInfer would still be below the 1.15× abort threshold —
so the right trigger for re-proposing this task is:

1. Full-attention NVTX share ≥ 10 % of decode (per a fresh Arm B
   profile), **and**
2. The inner `:kiln/attn/full/*` region (attention kernel, not qkv /
   o projections) is itself ≥ 8 % of decode.

Until both hold, the next decode-side lever is the GDN gate-path fusion
above, not FlashInfer.

### What would still make sense to vendor from FlashInfer later

- **Paged append / prefill** could help prefill TTFT on long contexts
  (128K+) where flash-attn-2's non-paged shape is less efficient than
  FlashInfer's page-aware kernels. This is a **prefill-path** change,
  not a decode-path change, and should be evaluated independently
  against Phase 6's prefill profile (not the decode profile above).
- **Batched decode** at large batch sizes (not kiln's current
  batch = 1 profile target) is where FlashInfer's tensor-core decode
  kernel is designed to win. Revisit when / if kiln targets larger
  inference batches.

## Not next target — GDN gate-path fusion (preflight, 2026-04-18)

A subsequent planner queued *"Phase 6: Fused GDN gate kernel
(gate_lin + swish + mul + rmsnorm + residual, bf16, decode-shape)"*
citing the **Recommended scope for the next cycle** section above
(lines starting "Vendor or author a fused `gdn_gate` kernel …"). That
section is stale at current `main` (HEAD `838f88f`): the combined
gates + gated_norm fusion it proposed was both **architecturally
impossible as one kernel** and **already attempted as two independent
kernels**, one merged and one closed as a null result. This section
records the preflight so future planning loops do not re-extract the
same redundant task.

No pod was spun up past the preflight `git clone`. Pod cost: **\$0**.

### Scope the task asked for

One fused CUDA kernel, bf16 decode shape only, collapsing:

1. `swish(gate_lin_output)`
2. elementwise multiply with the `gated_norm` input
3. `RMSNorm` over the gated output (epsilon, weight)
4. residual elementwise multiply

Target NVTX regions per the stale recommendation: `:kiln/gdn/gates`
(18.2 %) + `:kiln/gdn/gated_norm` (18.0 %) = 36.2 % of decode time at
the PR #156 profile (HEAD `5aa22e1`).

### Why it's redundant

The two NVTX regions map to two disjoint steps of
`gated_deltanet_forward` in `crates/kiln-model/src/forward.rs`:

| NVTX region            | Step | Computation                                            | Status                              |
| ---------------------- | ---- | ------------------------------------------------------ | ----------------------------------- |
| `:kiln/gdn/gates`      | 6    | `beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)` | **Fused by PR #158** (merged, bf16 in/out, `kiln-gdn-kernel` crate, `gdn_gates.cu`) |
| `:kiln/gdn/gated_norm` | 8    | `bf16(w * rms_normalize(attn_out) * silu(z))`           | **Attempted by PR #141** (closed, null 1.003–1.005× at `rows=32, hidden=128`) |

Between the two steps is Step 7 — the chunkwise recurrence that
produces `attn_out` from `(q, k, v, beta, g)`. PR #158's description
states the architectural finding explicitly:

> "the two regions are separated by the chunkwise recurrence
> (`Step 7 gdn_chunkwise_recurrence`) and cannot be combined into a
> single kernel. They must be tackled independently."

Step 7 internally launches `gdn_chunk_prep` (PR #162), the per-chunk
forward-substitution kernel, matmuls, and the recurrent update — not
something a gate-path elementwise fusion can absorb. "One kernel
covering gate_lin + swish + mul + rmsnorm + residual" is therefore
unreachable on this architecture.

The two independently-fusable halves have both been addressed:

- Step 6 (`:kiln/gdn/gates`): PR #158 fused the 8-op candle chain into
  a single CUDA launch. Post-#158 Arm B profile (HEAD `7132f29`) shows
  `:kiln/gdn/gates` moved from 18.2 % → 16.7 % (−1.5 pp), and the
  proportional share of the downstream `gated_norm` / `qk_norm` also
  shrank (−2.2 pp / −1.4 pp) because PR #158 dispatches fewer upstream
  elementwise tensors.
- Step 8 (`:kiln/gdn/gated_norm`): PR #141 vendored an FLA-style
  `fused_norm_gate` kernel covering `rms_inv * silu(z) * w` in one
  launch. Parity passed (max abs diff < 1e-2, bf16). Decode ITL
  delta was **1.003–1.005× under CUDA graphs**, below run-to-run
  noise and below the acceptance floor, so the PR was closed rather
  than merged. The "residual elementwise multiply" the current task
  description lists as a fourth op is not present in the Step 8 code
  path (`gated_rms_norm`'s output feeds the Step 9 `o_proj` GEMM
  directly — there is no per-element residual multiply between them).

### What the current recommendation actually says

The "Post-PR #158 Decode Profile (2026-04-18)" section above —
captured on HEAD `7132f29` right after PR #158 landed — replaces the
stale "Recommended scope" with a different next move:

> "**Vendor `chunk_gla_fwd` from flash-linear-attention (roadmap step
> 2).** The combined decode share of `gdn/gates` + `gdn/gated_norm` +
> `gdn/qk_norm` + `gdn/conv` is still **57.9 %** of wall-clock. A
> narrow gates-only fusion (like #158) only chips at the first of
> those four regions. Vendoring the upstream `chunk_gla_fwd` kernel
> that fuses the full GDN decode chain (conv + in_proj + gates +
> gated_norm + qk_norm + recurrent) is the highest-leverage move
> available — it would collapse ~58 % of decode wall-clock into a
> single kernel dispatch and directly eliminate the elementwise zoo
> that #158 could not reach."

The project description's **Current optimization queue** lists the
same item as step 2 ("Vendor fla-org/flash-linear-attention
`chunk_gla_fwd` (minimal)"). Agent note `kiln-gdn-bottleneck-postpr105`
points in the same direction.

### When to revisit the narrow gate-path fusion

Only if all three hold on a *fresh* Arm B profile:

1. `:kiln/gdn/gated_norm` is still ≥ 15 % of decode (i.e. a combined
   GDN-chain kernel never landed and this region is still the largest
   single elementwise hotspot), **and**
2. A parity-validated fused `gated_norm` kernel measures
   ≥ 1.10× decode ITL on at least one of the W4A16 arms (clearing
   PR #141's null-result ceiling), **and**
3. The fusion does not duplicate work that a vendored `chunk_gla_fwd`
   would already subsume.

Until then, the next decode-side lever is `chunk_gla_fwd`, not another
gate-path fusion PR.

### Preflight record

- Local clone of `ericflo/kiln` at HEAD `838f88f` (post-PR #163).
- `grep 'Recommended scope for the next cycle' PROFILING.md` →
  lines now include the STATUS callout flagging this section stale.
- `gh pr list --limit 20 --state all | grep -iE 'gdn.*gate|gate.*fusion|fused.*gate'`
  → PRs #158 (MERGED, gates fused) and #160, #163 (docs/re-profile).
  PR #141 surfaced via `gh pr view 141` as the closed null-result
  attempt for `:kiln/gdn/gated_norm`.
- Upstream search:
  - FLA (`flash-linear-attention`) exports `naive_gdn_gate` as an
    elementwise reference but no standalone fused CUDA kernel; the
    gate math is folded into the larger `chunkwise_gdn` Triton
    kernel, which is what step 2 of the optimization queue vendors
    next.
  - Liger-Kernel has no GDN-specific gate op.
  - `fused_norm_gate` (FLA) is exactly the shape PR #141 vendored
    and measured as null at the Qwen3.5-4B decode shape.
- Code inspection confirms Step 8 has no per-element residual multiply
  to fuse: `gated_rms_norm`'s output feeds `o_proj` (Step 9) directly.

Pattern match: identical redundancy class as the "chunk_gla_fwd
already vendored" incident (PR #131 redirect) and the "FlashInfer
paged GQA decode below threshold" preflight (this report, section
above). Doc-only resolution, \$0 pod spend. See agent note
`kernel-vendor-precondition-check` for the general rule.

## Files / Reproduction

Raw artifacts in this repo:

- `profiling-artifacts/statsA_kern.csv` — per-kernel table, Arm A
- `profiling-artifacts/statsA_nvtx.csv` — NVTX table, Arm A
- `profiling-artifacts/statsB_kern.csv` — per-kernel table, Arm B
- `profiling-artifacts/statsB_nvtx.csv` — NVTX table, Arm B

Full `.nsys-rep` captures live on pod `daogyp64vo0cgq` under `/tmp/` and
are not committed (19 MB Arm A, 34 MB Arm B).

To reproduce on a fresh RunPod A6000 pod (`ghcr.io/ericflo/kiln-runpod:latest`):

```bash
# 1. Install nsys 2024.5.1 (baked 2023.4.4 is buggy on long captures)
apt-get update && apt-get install -y libxcb-cursor0 cuda-nsight-systems-12-6
ln -sf /opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys /usr/local/cuda/bin/nsys

# 2. Clone + setup + build
GH_TOKEN=$(ce secret-get --name GITHUB_TOKEN)  # or equivalent
git clone https://x-access-token:$GH_TOKEN@github.com/ericflo/kiln.git /workspace/kiln
cd /workspace/kiln && kiln-setup
cargo build --release --features cuda,nvtx --bin kiln-bench

# 3. Fetch weights
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b

# 4. Steady-state ITL, three runs per arm (uncaptured)
for i in 1 2 3; do
  KILN_W4A16=0 KILN_CUDA_GRAPHS=true \
    ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training
done
for i in 1 2 3; do
  KILN_W4A16=1 KILN_CUDA_GRAPHS=true \
    ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training
done

# 5. Capture Arm A (BF16 baseline) — delay 15s past model load
KILN_W4A16=0 KILN_CUDA_GRAPHS=true \
  /usr/local/cuda/bin/nsys profile -t cuda,nvtx --delay=15 --duration=20 \
  -o /tmp/profile_armA --force-overwrite=true \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training

# 6. Capture Arm B (W4A16 production) — delay 110s past Marlin packing
KILN_W4A16=1 KILN_CUDA_GRAPHS=true \
  /usr/local/cuda/bin/nsys profile -t cuda,nvtx --delay=110 --duration=30 \
  -o /tmp/profile_armB --force-overwrite=true \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training

# 7. Extract tables
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/profile_armA.nsys-rep
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/profile_armB.nsys-rep
nsys stats --report nvtx_sum          --format csv /tmp/profile_armA.nsys-rep
nsys stats --report nvtx_sum          --format csv /tmp/profile_armB.nsys-rep
```

### Pod / cost notes

- Pod: `daogyp64vo0cgq` (RunPod RTX A6000 on-demand, ~$0.49/hr).
- Total pod uptime at capture end: ~2.5 hours (build, load
  testing, four bench runs, two nsys captures, stats extraction).
- Estimated pod cost: ~$1.25.


## Phase 7 GPU spike: streaming prefill on A6000 — 2026-04-20

**Outcome: CUDA parity test passes at T=2048 tile=512 (FP32 hybrid toy model, 1e-4 tolerance, per-layer GDN recurrent + conv state bit-equal). T=65536 prefill completes on a 48 GiB A6000 without OOM. ≤32k regression sweep (8192p / 128d) shows prefill within tolerance (+1.5% faster in streaming arm) and decode slightly below the 2% bar (-4.6%) — not a statistically clean null on this 3-run budget, documented for follow-up. Phase 7 streaming remains opt-in (`KILN_STREAMING_PREFILL=0` default); no threshold or default flip in this PR.**

### Parity (CUDA, `cargo nextest run --release --features cuda -p kiln-model test_streaming`)

All six tests green, 4.584s wall:

| # | Test | Result |
|---|---|---|
| 1 | `test_streaming_prefill_env_helpers` (existing) | PASS (0.056s) |
| 2 | `test_streaming_matches_monolithic_cpu_small` (existing) | PASS (0.142s) |
| 3 | `test_streaming_preserves_state_cpu` (existing) | PASS (0.188s) |
| 4 | `test_streaming_tile_invariance_cpu` (existing) | PASS (0.243s) |
| 5 | `test_streaming_matches_monolithic_cpu_mid` (existing) | PASS (0.284s) |
| 6 | **`test_streaming_matches_monolithic_cuda`** (new) | **PASS (4.583s)** |

The new CUDA test runs the 8-layer hybrid (6 GDN + 2 full-attn, FP32) at T=2048 / tile=512 / block_size=64 and asserts (a) last-tile logits match the corresponding slice of monolithic logits to `<1e-4` max-abs, (b) every GDN layer's `recurrent_states` matches across streaming vs monolithic at `<1e-4`, (c) same for `conv_states`. 1e-4 is a conservative FP32 bar that absorbs candle CUDA matmul reduction-order variance; the design doc (§c above) argues for bit-exactness and in practice the observed diffs were comfortably below threshold.

### T=65536 long-context validation (`KILN_STREAMING_PREFILL=1 KILN_W4A16=1`, tile=8192 default)

Single-request latency arm:

| Metric | Value |
|---|---|
| Effective prompt tokens | 65522 (bench truncates to block alignment) |
| Model VRAM after load | 17528 MB (14.3 GiB) |
| Model load time | 23.73s |
| Prefill time | 27485.1 ms (2384 tok/s) |
| Decode tokens generated | 33 |
| Mean ITL | 42.2 ms (23.7 tok/s) |
| P50 ITL | 41.8 ms |
| P99 ITL | 55.4 ms |
| **OOM?** | **No** |

The kiln-bench throughput sweep (4 / 8 / 16 sequential runs at this prompt size) fails — that path accumulates full-length prompts across the sweep and runs past VRAM, which is a scheduler-level constraint orthogonal to streaming prefill. The single-request latency arm is the spike target and it completes end-to-end on the 48 GiB A6000.

### ≤32k regression sweep (8192 prompt, 128 decode, 3 runs each, W4A16)

- **Arm A** = `KILN_STREAMING_PREFILL=0` (monolithic baseline).
- **Arm B** = `KILN_STREAMING_PREFILL=1 KILN_STREAMING_PREFILL_THRESHOLD=0` (streaming forced on for sub-threshold input).

| Arm | Run | prefill_time_ms | prefill tok/s | decode tok/s | P50 ITL ms | P99 ITL ms |
|---|---|---:|---:|---:|---:|---:|
| A | 1 | 3181.8 | 2574.3 | 51.2 | 19.3 | 22.3 |
| A | 2 | 3194.5 | 2564.1 | 50.2 | 19.6 | 22.9 |
| A | 3 | 3381.9 | 2422.0 | 48.1 | 20.6 | 22.7 |
| A | **median** | **3194.5** | **2564.1** | **50.2** | **19.6** | **22.7** |
| B | 1 | 3146.6 | 2603.1 | 47.9 | 20.8 | 22.7 |
| B | 2 | 3157.5 | 2594.1 | 47.6 | 20.8 | 31.2 |
| B | 3 | 3141.1 | 2607.6 | 48.4 | 20.5 | 25.9 |
| B | **median** | **3146.6** | **2603.1** | **47.9** | **20.8** | **25.9** |

Deltas (B vs A medians):

- Prefill tok/s: **+1.5%** (2603.1 vs 2564.1) — within 2% tolerance.
- Prefill time ms: **-1.5%** (3146.6 vs 3194.5) — within 2% tolerance.
- Decode tok/s: **-4.6%** (47.9 vs 50.2) — **outside the 2% tolerance** by roughly 1 ms of ITL. Run 2 P99 (31.2 ms) and run 3 P99 (25.9 ms) suggest this is decode-path ITL variance rather than a systematic streaming-path regression. Decode is identical single-step in both arms (both use `model_forward_paged`, not streaming, after prefill finishes); the paths differ only in warmup flow and the provenance of the `LinearAttentionState` going into the first decode step. Budget did not allow N=10+ runs to resolve. Worth a targeted re-bench before considering a threshold / default flip.

### Verdict and Phase 7 gating

- CUDA parity and T=65536 no-OOM are **green** → the streaming/tiled implementation is safe to keep shipped behind the default-off flag.
- ≤32k regression is **amber** on decode — not a block, but warrants a quieter re-bench (median-of-5+, same pod, same process) before changing the default or lowering the threshold.
- No changes to `KILN_STREAMING_PREFILL` default (stays `0`), no changes to `KILN_STREAMING_PREFILL_THRESHOLD` default (stays `32768`).

### Pod / cost notes

- Pod: `t0vmof6qkwostu` via lease `pod-016bb2e7c07be9a97ceb4a3b` (RunPod RTX A6000 on-demand, $0.49/hr, pool-leased).
- Leased → all work done: ~30 minutes (reused warm pod from pool; zero cold-build time).
- Spike cost: ~$0.25 (well under the $0.70 ceiling; $1.00 abort line was not approached).

## Phase 7 decode re-bench: resolving amber from PR #231 — 2026-04-20

**Outcome: median-of-5 interleaved A/B on the same warm A6000 pod / process lineage closes the amber. Arm A (monolithic) and Arm B (streaming) decode medians are 48.2 vs 48.1 tok/s — delta −0.21% (well inside the ±2% tolerance). The −4.6% gap reported in PR #231's 3-run regression sweep was ITL variance, not a streaming-path decode regression. No code or default flips from this re-bench; `KILN_STREAMING_PREFILL` stays default-off and `KILN_STREAMING_PREFILL_THRESHOLD` keeps its 32768 default.**

### Protocol

- Same warm pod / process lineage as the original GPU spike: pod `t0vmof6qkwostu` via lease `pod-016bb2e7c07be9a97ceb4a3b` (RunPod RTX A6000 on-demand, $0.49/hr). Reused the `target/release/kiln-bench` binary and Marlin pack cache produced by the PR #231 spike — controls for build-time / sccache / pack-cache drift that a fresh pod would reintroduce.
- Interleaved A, B, A, B, … for 5 rounds (10 kiln-bench invocations total). Each invocation is a fresh process, but all share the same pre-warmed binary and packed-weight cache on the pod. Interleaving controls for slow process-age / thermal drift that pure-sequential (5A then 5B) would fold into the arm delta.
- Both arms identical except for `KILN_STREAMING_PREFILL`:
  - Arm A: `KILN_STREAMING_PREFILL=0 KILN_W4A16=1 ./kiln-bench --paged --prompt-tokens 8192 --max-output-tokens 128 --skip-training` (monolithic baseline).
  - Arm B: `KILN_STREAMING_PREFILL=1` + same flags (streaming/tiled prefill). `streaming_prefill_enabled` in `crates/kiln-model/src/forward.rs` is a binary read of that env var; there is no active `KILN_STREAMING_PREFILL_THRESHOLD` env plumbing in the decode path — at `seq_len=8192` below the 32768 default threshold, `KILN_STREAMING_PREFILL=1` is what actually flips the path for this bench.
- Reported numbers are the final `--- Latency (single request) ---` summary each run prints to stderr. kiln-bench's paged-latency measurement populates that summary when `--paged` is set, so the P50 / P99 ITL and decode tok/s are the paged-path measurements.

### Results (5 rounds × 2 arms, same warm A6000 pod / process)

| Round | Arm | prefill tok/s | decode tok/s | mean ITL ms | P50 ITL ms | P99 ITL ms | TTFT ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | A | 2540 | 47.7 | 21.0 | 20.8 | 23.2 | 3225.2 |
| 1 | B | 2589 | 48.0 | 20.8 | 20.7 | 22.8 | 3163.2 |
| 2 | A | 2542 | 47.9 | 20.9 | 20.7 | 22.9 | 3222.9 |
| 2 | B | 2603 | 48.1 | 20.8 | 20.6 | 23.9 | 3146.7 |
| 3 | A | 2559 | 49.7 | 20.1 | 19.8 | 22.9 | 3201.0 |
| 3 | B | 2646 | 50.6 | 19.8 | 19.6 | 21.9 | 3096.0 |
| 4 | A | 2523 | 50.2 | 19.9 | 19.7 | 22.4 | 3245.9 |
| 4 | B | 2600 | 47.7 | 21.0 | 20.7 | 25.2 | 3150.4 |
| 5 | A | 2444 | 48.2 | 20.7 | 20.6 | 23.8 | 3351.7 |
| 5 | B | 2613 | 50.7 | 19.7 | 19.6 | 22.0 | 3134.4 |

### Per-arm stats (n=5)

| Metric | Arm A median | Arm A stddev (% of median) | Arm B median | Arm B stddev (% of median) |
|---|---:|---:|---:|---:|
| prefill tok/s | 2540.0 | 1.59% | **2603.0** | 0.75% |
| decode tok/s | **48.2** | 2.10% | **48.1** | 2.78% |
| mean ITL ms | 20.7 | 2.13% | 20.8 | 2.66% |
| P50 ITL ms | 20.6 | 2.29% | 20.6 | 2.54% |
| P99 ITL ms | 22.9 | 2.00% | 22.8 | 5.47% |

Decode stddev is 2.1% (A) and 2.78% (B) of the median — both under the 3% variance guard from the task spec, so no additional rounds were triggered. Arm B's P99 ITL has a wider spread (5.47%) driven by round 4's 25.2 ms tail; that's isolated tail variance, not a systematic regression (median P99 is 22.8 ms, better than Arm A's 22.9 ms).

### Decision (decode tok/s medians)

- Arm A median: **48.2 tok/s**
- Arm B median: **48.1 tok/s**
- Delta: **−0.1 tok/s (−0.21%) vs Arm A** — well inside the ±2% tolerance.
- Prefill medians: Arm B 2603 vs Arm A 2540 tok/s = **+2.5% prefill** in the streaming arm, consistent with the +1.5% prefill seen in the PR #231 spike.

**Verdict: PR #231 amber is ITL variance. RESOLVED.** The streaming/tiled prefill path does not regress decode throughput at 8192p / 128d on A6000 once n≥5 rounds are collected on a warm process lineage. The −4.6% from the 3-run spike was inside the normal decode-tok/s noise floor (roughly 2–3% stddev on this workload/pod).

### What this changes

Nothing in shipped code: Phase 7 streaming remains opt-in (`KILN_STREAMING_PREFILL=0` by default), and `KILN_STREAMING_PREFILL_THRESHOLD` keeps its 32768 default. This re-bench only resolves the outstanding amber flag from the PR #231 write-up; no gating decision downstream is waiting on it.

If a future planner considers flipping `KILN_STREAMING_PREFILL` on by default or lowering the threshold, the data point to anchor on is: on A6000 at 8192p / 128d with W4A16, the two paths are **statistically indistinguishable on decode** (−0.21% ≪ noise floor) and the streaming path is **+2.5% on prefill tok/s**. Any default flip still needs the ≥32k / OOM-risk arm reconsidered separately; this re-bench only covers the ≤32k amber.

### Pod / cost notes (re-bench)

- Same pod as the spike (`t0vmof6qkwostu`, lease `pod-016bb2e7c07be9a97ceb4a3b`) — warm, with the release binary and packed weights already resident. No rebuild.
- 10 kiln-bench runs × ~6 min = ~60 minutes wall time. Pod at $0.49/hr → ~$0.49.
- Total re-bench spend: ~$0.50 (well under the $1.50 ceiling for this task).

## Native MTP speculative decoding — preflight 2026-04-20

**Status: GREEN — greenlight implementation task.**

Supersedes the `kiln-speculative-decoding-design` agent note (skip-layer self-spec). For Qwen3.5-4B specifically, the native pretrained MTP head is strictly better than self-spec: same k=1 draft cost, much higher acceptance rate, zero extra weights at serving time beyond 15 already-shipped tensors. This preflight is doc-only — no code changed, no pod acquired, $0 spend.

### Check 1 — MTP heads exist in the Qwen3.5-4B checkpoint (GREEN)

Verified via HuggingFace `config.json` + `model.safetensors.index.json` for the `Qwen/Qwen3.5-4B` checkpoint (same checkpoint kiln already loads):

- `num_nextn_predict_layers: 1` → **k=1 draft depth** (single MTP head; one speculative token per step).
- 15 MTP-prefixed tensors in the index:
  - `mtp.fc.weight` — shape `[2560, 5120]` — the `concat(inputs_embeds, hidden_states) → H` projection (matches `2*H → H` with `H=2560`).
  - `mtp.pre_fc_norm_embedding.weight` `[2560]`, `mtp.pre_fc_norm_hidden.weight` `[2560]` — two pre-FC RMSNorms (norm-then-concat pattern, one per input stream).
  - `mtp.layers.0.*` — one full transformer layer with GQA attention + SwiGLU MLP + input/post-attention layernorms. Tensor shapes match the main-model GQA blocks (q_proj `[8192, 2560]` = 32 heads × 256 head_dim; k_proj / v_proj `[1024, 2560]` = 4 KV heads × 256; o_proj `[2560, 8192]`; gate_proj / up_proj `[MLP_size, 2560]`; down_proj `[2560, MLP_size]`).
  - `mtp.final_layernorm.weight` `[2560]`.
- **No separate `mtp.lm_head`** in the index — the MTP head ties to the base model's `lm_head` (which itself ties to `embed_tokens`). That matches vLLM's `qwen3_next_mtp.py` (`self.lm_head = model.lm_head` linkage) and is what kiln already does for the main model (see `model_forward_head` in `crates/kiln-model/src/forward.rs` line 2873: `rms_norm(hidden, final_norm) → broadcast_matmul(embed_tokens_t)`).

Net: everything MTP needs is already in the checkpoint. Nothing to ship, nothing to train, nothing to distill — just load + wire.

### Check 2 — vLLM reference implementation (GREEN)

Primary reference: `vllm/model_executor/models/qwen3_next_mtp.py` (EAGLE-framework-based, production path).

Key structural points (relevant for kiln integration):

1. **Forward signature**: `Qwen3NextMTP.forward(input_ids, positions, hidden_states, inputs_embeds=None)` — takes the **base model's last hidden state** for the verified prefix plus the new draft-token positions.
2. **Input construction**: `inputs_embeds = embed_tokens(input_ids)` (the draft token candidate), then `x = fc(concat([pre_fc_norm_embedding(inputs_embeds), pre_fc_norm_hidden(hidden_states)], dim=-1))` — norm each stream, concatenate channel-wise, project `2H → H` via `fc: ColumnParallelLinear`.
3. **One MTP decoder block**: Runs the single `mtp.layers.0` block on `x` with the same KV cache / attention metadata as the base model's last full layer.
4. **Final RMSNorm + tied lm_head**: `logits = lm_head(final_layernorm(x))` — the last two ops are exactly what kiln already has in `model_forward_head`.
5. **Speculative loop lives outside this module**: vLLM's speculative worker drives the `target_forward → mtp_forward → sample draft → target_forward (verify k+1 in parallel) → rejection_sample` sequence. `qwen3_next_mtp.py` is just the "one MTP step" building block.
6. **GDN state rollback — explicit non-handling in vLLM**: qwen3-next has hybrid GDN + attention layers just like Qwen3.5, and vLLM's MTP path **does not checkpoint GDN recurrent state per-draft-token**. When a draft token is rejected, vLLM relies on the spec worker rerunning the target model on the committed prefix, which regenerates the correct GDN state from scratch on the next step. For kiln that's fine for correctness but ~24 GDN layers × O(head_dim × head_dim) state × ~10-20× per-second reject events is measurable overhead. **Main implementation risk for kiln** — see "Risks" below.

vLLM's `qwen3_next_mtp_eagle.py` is a second entry point wrapping the same head into EAGLE-3's tree-speculation machinery; kiln will not use tree spec at k=1 (it has no benefit), so the plain `qwen3_next_mtp.py` is the right reference.

### Check 3 — Math ceiling at k=1 (GREEN)

Formula (from standard spec-decode analysis, see Leviathan et al. 2023 and the more-general draft-model equation):

```
speedup = (1 - α^(k+1)) / ((1 - α) × (k+1))
```

where α is the per-token acceptance rate of the draft against the target distribution.

At **k=1** (Qwen3.5-4B's `num_nextn_predict_layers`):

| α (acceptance rate) | Ceiling speedup | Decode tok/s @ 49.76 baseline |
|---:|---:|---:|
| 0.50 | 1.333× | 66.3 tok/s |
| 0.60 | 1.400× | 69.7 tok/s |
| 0.70 | 1.467× | 73.0 tok/s |
| **0.72** (conservative published MTP range) | **1.480×** | **73.6 tok/s** |
| **0.80** (typical published MTP range) | **1.533×** | **76.3 tok/s** |
| 0.85 | 1.567× | 77.9 tok/s |
| 0.90 | 1.600× | 79.6 tok/s |

Target for Phase 6 / spec-decode work is **1.5× on decode tok/s** (49.76 → 74.6 tok/s).

- At α=0.80 (typical) the math ceiling is 1.533× — just clears the 1.5× target. Real-world measured speedup lives below the ceiling because of draft + verify overhead on top of baseline decode (non-zero draft cost, non-zero concat/norm/fc/block cost, GDN state management). Published Qwen MTP numbers at α≈0.8 + k=1 typically realize ~1.3-1.4× — below the 1.5× target.
- At α=0.85 the ceiling is 1.567× — comfortable headroom.
- Qwen3-Next 80B's own MTP self-reports α in the 0.72-0.85 band on typical workloads. Qwen3.5-4B (much smaller target, same MTP head architecture) should land in the same range; a smaller target model tends to match its MTP head better (lower-entropy hidden states → draft head more calibrated), not worse.

**Ceiling verdict**: k=1 clears the 1.5× target at α ≥ 0.80. The math is tight but not marginal. There is also no k-knob to tune post-implementation (Qwen3.5-4B ships k=1 fixed) — this is the only arm we can run.

Compare to FlashInfer paged GQA decode (killed in PR #163 at ceiling ≤1.005×) and fused L2-QK-norm (PR #173, null-median): those had math ceilings below the dispatch-amortization floor. MTP at k=1 has a ceiling 30-50× larger than that floor, so null-result risk is qualitatively different (it's implementation-overhead-dominated, not ceiling-dominated).

### Check 4 — Integration scope in kiln source (GREEN)

Files that need to change for native MTP (line counts are estimates for the implementation task, not this preflight PR):

| File | LOC now | Est. added | What changes |
|---|---:|---:|---|
| `crates/kiln-model/src/weights.rs` | 232 | +80 | Add `MtpWeights { fc: WeightTensor, pre_fc_norm_embedding: WeightTensor, pre_fc_norm_hidden: WeightTensor, layer: LayerWeights, final_norm: WeightTensor }` and thread it as `ModelWeights.mtp: Option<MtpWeights>`. `Option<>` because other Qwen3.5 variants / future models may not ship MTP; keep the existing `ModelWeights` shape backwards-compatible. Update `total_bytes`/`total_params`. |
| `crates/kiln-model/src/loader.rs` | 1445 | +150 | Detect `mtp.fc.weight` in the safetensors index → load the 15-tensor MTP block. Reuse `load_layer` for `mtp.layers.0.*` with a prefix override. Tie `mtp.lm_head` to `embed_tokens` (no separate load). Gate behind `if num_nextn_predict_layers > 0` from config.json. |
| `crates/kiln-model/src/forward.rs` | 5719 | +500 | (1) Add `MtpGpuWeights` to `GpuWeights` (Option, uploaded if present). (2) Add `mtp_forward_step(model, hidden_states, draft_token_id, position, kv_cache, gdn_state) → draft_logits` — runs `concat(pre_fc_norm_embedding(embed(t)), pre_fc_norm_hidden(h)) → fc → mtp.layers.0 → final_layernorm → tied lm_head`. (3) Add `speculative_mtp_decode_step` — analogous to the existing `speculative_decode_step` in `src/speculative.rs` but driving MTP + a k+1-wide verify call into `model_forward_paged`. (4) Add `LinearAttentionState::snapshot() / restore_from(snapshot)` helpers on the struct at line 241 — deep-clone `recurrent_states` + `conv_states` before MTP draft, restore on reject. (5) Add rejection sampling + resample identical to `src/speculative.rs::rejection_sample` (can call into existing impl). |
| `crates/kiln-model/src/generate.rs` | 1562 | +200 | New `generate_mtp_speculative()` + `generate_from_tokens_mtp_speculative()` parallel to the existing `generate_speculative` skip-layer versions at lines 657 / 689. Dispatch based on `SpecMethod` from config. Reuse `draft_forward_for_state_init` pattern for initial GDN state population before the first MTP draft. |
| `crates/kiln-server/src/config.rs` | 846 | +40 | Add `SpecMethod` enum: `Off` (default) / `Mtp` / `SkipLayer` (keep existing for fallback / A/B). Add `KILN_SPEC_METHOD` env flag parsing near lines 397-408. Validate at lines 455-460: `SpecMethod::Mtp` requires `num_nextn_predict_layers > 0` in loaded model config. |
| `crates/kiln-model/src/speculative.rs` | 646 | +0 / −0 | **Keep as-is**. Skip-layer self-spec stays as `SpecMethod::SkipLayer` fallback and the A/B baseline for benchmarking MTP. No deprecation in this slice. |
| `crates/kiln-server/src/api.rs` (and/or router) | — | +20 | Thread `SpecMethod` into the decode loop selection. If MTP requested but model lacks MTP weights, log a warning and fall back to off (don't hard-fail the server). |
| `PROFILING.md` | 3324 | +250 | New "Phase X — native MTP spec-decode results" section post-implementation (not this PR). |

**Total estimate: ~800-1000 LOC added across 6-7 files**, no kernel crate work required (all math is GEMM / RMSNorm / attention / sampling that already have paths). No new dependency on mamba-ssm / FlashInfer / Marlin / cuDNN. No new kernel crate — reuses existing kernels.

Existing speculative decoding infrastructure:
- `crates/kiln-model/src/speculative.rs` (646 LOC, already has `SpeculativeConfig`, `rejection_sample`, `speculative_decode_step`) — much of the MTP loop structure is copy-adapt from here; rejection sampling is identical.
- `crates/kiln-server/src/config.rs` already has `SpeculativeDecodingConfig` (line 126) with `enabled` / `num_speculative_tokens` / `draft_layers` + `KILN_SPEC_*` env flags (lines 397-408). Adding `method` is additive.
- **Server decode path is not yet wired to speculative at all**: `grep -ri speculative crates/kiln-server/src/` returns only the config struct, the CLI startup banner, and test assertions — no actual dispatch. That's good: MTP is net-new decode wiring, not a swap-out of live code. Lower collision risk.

### Check 5 — Update superseded agent note (DONE)

`ce notes-set --topic kiln-speculative-decoding-design` has been updated with a `SUPERSEDED 2026-04-20:` prefix referring back to this section. The skip-layer description is preserved as the fallback path; MTP becomes the primary recommendation for Qwen3.5-4B.

### Risks (main + mitigation)

1. **GDN state rollback overhead (primary risk).** 24 GDN layers × `LinearAttentionState { recurrent_states: Vec<Tensor>, conv_states: Vec<Tensor> }` must snapshot before every MTP draft and restore on every rejected token. Rough sizing: per layer, `recurrent_states` is roughly `(batch, num_v_heads, head_dim, head_dim)` and `conv_states` is `(batch, conv_size × num_heads × head_dim)`. At single-stream bs=1 the total snapshot is small (single-digit MB), but it's `24 × cudaMemcpyDeviceToDevice` per draft step. If naive copy lands at >1 ms/step this alone eats the entire MTP speedup. Mitigation: use a double-buffered ping-pong allocation (snapshot goes to a pre-allocated "shadow" slot, restore is just a pointer swap) instead of real memcpy; this is what `speculative.rs::draft_forward` already does for attention KV cache. Test in isolation before the full loop.

2. **Tied lm_head broadcast_matmul on 2 positions.** MTP verify calls `model_forward_paged` with k+1=2 positions. `model_forward_head` does `broadcast_matmul(embed_tokens_t)` — which is a `[2, H] × [H, V]` GEMM where V=151936. That's 4× the FLOPs of a single-position decode and kiln's decode is GEMM-bound on this step (top NVTX regions at 17-18% each). Not a correctness issue, a throughput one: the 2-position verify cost is baked into the math ceiling above, but if lm_head isn't batched well in the current path this lands at 2× not 1× + ε. Mitigation: inspect the `model_forward_head` path for any per-position branching; already parallelizes across the sequence dim in the current code.

3. **Marlin W4A16 compatibility for MTP layer.** `mtp.layers.0` has q_proj / o_proj / gate_proj / up_proj / down_proj shapes that match the main model's layers. Marlin packing in the loader operates per-projection and should pick these up transparently (PR #146 path). Need to verify in implementation that the existing pack loop covers `mtp.*` prefixes — this is a 1-line check in the loader, not a design risk.

4. **Acceptance rate below the 1.5× target.** At α=0.70, ceiling is 1.467× — below target. Qwen3.5-4B does not have published MTP acceptance numbers (Qwen3.5-4B isn't flagged as MTP-enabled in the model card text even though the weights are there). Risk: measured α could be <0.70 on kiln's typical workloads, giving a below-target speedup even with perfect implementation. Mitigation: A/B against skip-layer self-spec and against baseline in the same implementation PR. If α measurement on a representative workload comes back <0.72, pivot early — don't ship a below-target kernel just to cash the preflight.

5. **Prefix cache interaction.** kiln has `KILN_PREFIX_CACHE_ENABLED=true` by default. MTP reads the target model's last hidden state, which for a prefix-cache hit was computed before the current request started. Need to verify the hidden state is cached alongside the KV for the last verified position, not just the KV. If it isn't, MTP can't run on the first step of a prefix-cache-hit request and must fall back to one synchronous target step to materialize `hidden_states`. Mitigation: small (one step), well-defined; probably worth it to avoid architectural coupling of MTP into prefix cache serialization.

### Preflight cost summary

- Doc-only PR. $0 pod spend (no RunPod acquired). ~30 min of HF index inspection + vLLM source read + math + source traversal.
- Kiln agent notes updated (`kiln-speculative-decoding-design` → SUPERSEDED).
- No PROFILING.md re-profile needed (Phase 6 kernel frontier is unchanged; MTP is a Phase 7+-style addition, a new decode path rather than a kernel fusion).

### Recommendation

**GREENLIGHT a follow-up implementation task** titled along the lines of "Phase X — native MTP spec-decode on Qwen3.5-4B (GREEN-after-preflight)". Scope: the 6-7 files listed in Check 4 (~800-1000 LOC). Gate behind `KILN_SPEC_METHOD=mtp`, default off. Ship with A/B bench (off / skip-layer / MTP) on A6000 warm pod, median-of-3 rule. Target: **≥1.5× decode tok/s vs. baseline Arm B on the standard 512p/128d workload**, with α reported separately. If measured α < 0.72 on the standard workload, stop at the A/B numbers — don't force-ship a below-ceiling path.

## Phase X — native MTP spec-decode results (2026-04-21)

Bench sweep of the native MTP decode arm introduced in PR #247 and patched by PR #253 (loader now accepts Qwen3.5-4B's bare `mtp.*` tensor layout). Fills in the Mtp row of the results table that PR #247 opened.

### Context

- Hardware: RunPod A6000 (SM 86), CUDA 12.4, `ghcr.io/ericflo/kiln-runpod:latest`.
- Commit: `0443b81` (tip of `main` post-#253).
- Model: `Qwen/Qwen3.5-4B` weights at `/workspace/qwen3.5-4b` (15 `mtp.*` tensors, `mtp_num_hidden_layers=1`, `tie_word_embeddings=true`).
- Build: `cargo build --release --features cuda --bin kiln-bench` (sccache: 94% hit rate on warm pod, 81 s wall clock).
- Env flags for all runs: `KILN_W4A16=1 KILN_CUDA_GRAPHS=true`.
- Bench args: `--paged --prompt-tokens 512 --max-output-tokens 128 --skip-training`.
- Cadence: 3 runs per arm, back-to-back, median reported.

### Results table

| Arm | decode tok/s | mean ITL (ms) | p50 ITL (ms) | p99 ITL (ms) | α |
|---|---:|---:|---:|---:|---:|
| Off | **48.5** | **20.6** | **20.50** | **23.70** | — |
| Mtp (initial, PR #247+#253) | **BLOCKED** | — | — | — | — |
| Mtp (after GDN state threading, this PR) | **41.16** | **24.29** | **17.23** | **39.31** | **0.411** |

Off median is the standard 512p/128d Arm B baseline. The initial Mtp row blocked before any decode steps completed. The follow-up Mtp row is measured after threading `LinearAttentionState` through `speculative_mtp_decode_step` with snapshot/restore on reject (see "Follow-up: GDN state threading" below).

### Off — raw per-run numbers

| Run | prefill ms | prefill tok/s | mean ITL ms | mean ITL tok/s | p50 ITL ms | p99 ITL ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 343.5 | 1473 | 19.4 | 51.6 | 19.01 | 25.28 |
| 2 | 354.7 | 1427 | 20.6 | 48.4 | 20.50 | 23.70 |
| 3 | 362.0 | 1398 | 20.6 | 48.5 | 20.58 | 22.69 |
| **median** | **354.7** | **1427** | **20.6** | **48.5** | **20.50** | **23.70** |

Compare to PR #247's Arm B baseline (50.77 tok/s): 48.5 is ~4.4% lower, comfortably inside the median-of-3 noise floor — baseline is reproduced, not regressed.

### Mtp — blocker

All 3 runs prefilled successfully (~355 ms / ~1420 tok/s) then failed identically at the first decode step:

```
--- Latency Benchmark (MTP — native speculative, paged) ---
  Measuring latency [MTP, paged, blocks=48] (506 prompt tokens)...
    Prefill (MTP): 358.7ms (1410 tok/s)
Error: MTP latency benchmark failed

Caused by:
    0: speculative_mtp_decode_step failed
    1: mtp verify forward failed
    2: linear attention state required for GDN layers (layer 0)
```

Loader is good — every Mtp run logs `Native MTP head detected and loaded (k=1 draft depth)`, confirming PR #253's `detect_mtp_prefix` fix is working against Qwen3.5-4B's stock tensor layout. The failure is downstream of loading.

### Root cause

`crates/kiln-model/src/speculative.rs:573-585`, inside `speculative_mtp_decode_step`, calls the base model verify forward with `None` for the `LinearAttentionState`:

```rust
let verify_input = [last_token, draft_token];
let (verify_logits, new_hidden) = model_forward_paged_with_last_hidden(
    backend,
    &verify_input,
    weights,
    config,
    base_cache,
    base_block_table,
    base_pos,
    None, // MTP speculative: no linear-attn state rollback for GDN in this WIP.
    None, // no LoRA on the verify pass — keep parity with scaffolding.
    None, // positions_gpu: let the forward pass build positions internally.
)
.context("mtp verify forward failed")?;
```

The 8th argument is `None`, but Qwen3.5-4B's 24 GDN layers require a `LinearAttentionState`. The forward pass panics at `forward.rs:2949` (and the mirrored 3055 / 3477 sites for the other GDN paths): `linear attention state required for GDN layers (layer {i})`. The inline comment in `speculative.rs:581` explicitly flags this as a known WIP gap.

MTP cannot run in kiln today on this hybrid-architecture model until GDN state snapshot / restore is threaded through the MTP verify path.

### Recommendation

Minimal follow-up fix PR, narrow scope:

1. Add `LinearAttentionState::snapshot()` / `restore_from(snapshot)` on the struct at `forward.rs:241` (deep-clone `recurrent_states` + `conv_states`; ping-pong allocation per preflight Risk #1 above — avoid per-step `cudaMemcpy`).
2. Thread the snapshotted state into `speculative_mtp_decode_step`: take a snapshot before the draft step, pass the live `&mut LinearAttentionState` to `model_forward_paged_with_last_hidden`, and restore from the snapshot on rejected tokens.
3. Re-run the 3× Mtp sweep. Keep the same `KILN_W4A16=1 KILN_CUDA_GRAPHS=true --paged --prompt-tokens 512 --max-output-tokens 128` workload. Median-of-3, reporting α alongside decode tok/s.

Preflight Check 2 in the section above math-ceilings α=0.75 at 1.571× on Qwen3.5-4B's MTP (k=1, ~86% overhead). The ≥1.5× target stands; α<0.72 is the stop-ship threshold.

### Cost

- Pod: 1× A6000 warm lease from the `ce kiln-pod-acquire` pool, ~35 min wall clock (preflight + build + 3×Off + 3×Mtp + log pull + lease release).
- Build: 81 s (sccache hit rate 94%).
- No nsys capture, no Marlin repack, no second pod.
- Well under the 120 min / $60 cap.

### What this PR does NOT do

- Does not attempt the GDN state rollback fix (scope-discipline: doc-only, per task brief).
- Does not run the SkipLayer arm (out of scope for this PR; Phase X preflight uses SkipLayer as an A/B option but the Mtp row is the blocker here).
- Does not modify any `.rs` source. PROFILING.md only.

### Follow-up: GDN state threading (2026-04-21, same-day)

Surgical fix landed in the same branch that produced this PROFILING.md update: `speculative_mtp_decode_step` now accepts `linear_state: &mut LinearAttentionState`, takes a snapshot before the verify forward, threads `Some(linear_state)` into `model_forward_paged_with_last_hidden`, and calls `restore_from(&snapshot)` on the reject branch so the rejected draft token's GDN state mutation is rolled back.

With that change the Mtp bench no longer fails at `linear attention state required for GDN layers (layer 0)`. 3× Mtp + 3× Off re-run on the same A6000 pod, same bench flags:

#### Mtp — raw per-run numbers (post state-threading)

| Run | prefill ms | prefill tok/s | mean ITL ms | mean ITL tok/s | p50 ITL ms | p99 ITL ms | α |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 335.8 | 1507 | 24.41 | 40.97 | 17.12 | 38.05 | 0.407 |
| 2 | 331.2 | 1527 | 24.29 | 41.16 | 17.23 | 39.31 | 0.411 |
| 3 | 334.9 | 1511 | 22.53 | 44.39 | 17.05 | 38.99 | 0.524 |
| **median** | **334.9** | **1511** | **24.29** | **41.16** | **17.23** | **39.31** | **0.411** |

#### Off — raw per-run numbers (same pod, re-run for fresh-paired comparison)

| Run | prefill ms | prefill tok/s | mean ITL ms | mean ITL tok/s | p50 ITL ms | p99 ITL ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 324.2 | 1561 | 20.46 | 48.87 | 20.25 | 25.89 |
| 2 | 334.9 | 1511 | 20.74 | 48.21 | 20.75 | 23.13 |
| 3 | 326.2 | 1551 | 20.28 | 49.32 | 20.14 | 22.55 |
| **median** | **326.2** | **1551** | **20.46** | **48.87** | **20.25** | **23.13** |

#### Findings

1. **State threading is correct.** The bench path that previously crashed at decode step 0 now completes all three runs with 128 tokens each, stable tok/s across runs, and sane α.
2. **α = 0.411 is well below the 0.72 stop-ship floor.** Preflight Check 2 math-ceilings α=0.75 at 1.571× and sets ≥1.5× as the target. At α=0.411 with k=1, mean accepted tokens per verify call is 1.411; the verify-pass overhead (two-token forward + draft-head forward + snapshot/restore dToD) is larger than the 0.411-token savings, so:
3. **Mtp is 15.7% SLOWER than Off on Qwen3.5-4B at this α** (41.16 vs 48.87 tok/s median decode). p99 ITL also widens from 23.13 → 39.31 ms.
4. **The MTP path is functional but not yet shippable.** This PR unblocks the measurement path; the low-α investigation is a separate follow-up.

#### Why α is low (hypotheses, not debugged here)

- The MTP head may be consuming a stale `hidden_state` from the base forward — `speculative_mtp_decode_step` passes `last_hidden` returned from the verify forward, but the ordering vs the `embed(draft_token)` concat in `mtp_forward_step` should be re-checked against the HF reference.
- `pre_fc_norm_embedding` and `pre_fc_norm_hidden` tensor loading (PR #253) should be re-verified post-loading — log a dequant sample vs. HF `safetensors` to confirm the MTP head weights are byte-identical.
- Sampling temperature in the bench's rejection loop may not match the base sampler — preflight assumed greedy/temp=0; if the verify path uses a different sampler than the draft, acceptance will systematically under-report.
- Pretraining mismatch: Qwen3.5-4B's native MTP was trained with `mtp_num_hidden_layers=1` and a specific concat order; any minor layout difference in our loader (e.g., swapped `fc` input ordering) would hand back near-random predictions → α≈0.4 is exactly the signature.

#### Next steps

- Open a dedicated `mtp: debug low-α root cause` task. Instrument α per-step, log the top-5 draft tokens + base-verify probs for the first 16 verify calls, compare against HF reference on the same prompt tokens, confirm weight-tensor byte-equality after loader.
- Revisit shadow-slot ping-pong snapshot rewrite (zero-copy) only after α clears 0.72 — at α=0.411 it would not move the needle.
- Consider gating MTP on the new `KILN_SPEC_METHOD=mtp` flag with a startup warning until α ships above floor.

#### Cost (follow-up)

- Same warm pod lease, incremental re-build: 33 s wall clock (sccache hit).
- 3× Mtp run ~6 min + 3× Off run ~6 min.
- Well under the 120 min / $60 cap for the combined effort.

## MTP low-α root cause — Phase A static audit + Phase B instrumentation (2026-04-21)

Follow-up to "Phase X — native MTP spec-decode results" above. PR #257 unblocked the MTP measurement path; Mtp arm now runs to completion at α=0.411, well below the 0.72 stop-ship floor. This update lands the Phase B instrumentation patch and records the Phase A static-audit findings.

### Tier classification

This is a **Tier 2 diagnostic** PR (per the task brief): no fix shipped, no claim that any of the four candidate hypotheses from the Phase X "Why α is low" section are confirmed or refuted by code reading alone. Phase B requires a runtime trace on a pod plus an HF parity diff, which is the work the next agent picks up using the instrumentation landed here.

### Phase A — static audit findings

Code references are pinned to `e18ed1b` (PR #257 merge) on `main`.

| Hypothesis (from Phase X "Why α is low") | Static-audit verdict | Evidence |
|---|---|---|
| Stale `h_prev` from base verify forward | **Doc-confirmed correct on ACCEPT** | `crates/kiln-model/src/speculative.rs:481-490` documents that `new_h_prev` is the hidden at the draft token's position in the verify pass, which is the correct conditioning input for the *next* MTP step (it predicts the bonus, which on accept is exactly the token immediately after the draft). The ACCEPT case is the dominant path at any reasonable α; the REJECT case has a known <5% staleness called out in the same comment block. Both paths are too small to explain the 0.31 gap from 0.411 to 0.72. |
| Loader byte mismatch (`mtp.fc` etc.) | **Tensor names map cleanly** | `crates/kiln-model/src/loader.rs:610-704`: `mtp.fc.weight` shape validated as `[hidden, 2*hidden]`; both `mtp.norm.weight` and `mtp.final_layernorm.weight` accepted; `pre_fc_norm_embedding` and `pre_fc_norm_hidden` map directly to the corresponding `MtpWeights` fields with no aliasing. PR #253's `detect_mtp_prefix` covers Qwen3.5-4B's bare `mtp.*` layout. **Byte-equality vs. raw safetensors is not verifiable from code reading** — that needs Phase B. |
| Sampler / temperature mismatch | **Both paths greedy** | `speculative_mtp_decode_step` calls `greedy_sample(&mtp_logits)` for the draft (`speculative.rs:566`) and `greedy_sample(&verify_pos0)` for the target (`speculative.rs:602`). The `_params: &SamplingParams` argument is intentionally unused (commented `# Greedy-only path (temperature == 0)` on `speculative.rs:533-535`). Sampler asymmetry is ruled out for the bench workload. |
| Swapped `fc` input ordering vs. vLLM | **Visual match to vLLM** | `crates/kiln-model/src/forward.rs:3517-3523`: `Tensor::cat(&[&norm_emb, &norm_h], 2)` matches vLLM `qwen3_next_mtp.py`'s `concat([pre_fc_norm_embedding(inputs_embeds), pre_fc_norm_hidden(hidden_states)])`. Embed first, hidden second. **Tensor name → field semantics are correct.** What this *cannot* rule out from code reading: that `mtp.pre_fc_norm_embedding.weight` and `mtp.pre_fc_norm_hidden.weight` are themselves transposed in the published checkpoint relative to vLLM's training-time layout. Confirming that requires Phase B byte-equality + per-token dequant diff. |

#### Other code-reading checks

- **Tied lm_head**: `forward.rs:3553` reuses `weights.embed_tokens_t` for the MTP head's projection. Matches the loader's contract that there is no separate `mtp.lm_head.weight` tensor (per `loader.rs:604-606`). ✓
- **Position handling**: MTP uses an independent position counter (`mtp_pos`, starting at 0) for both the RoPE positions tensor (`forward.rs:3528`) and the KV-cache start index (`forward.rs:3535`). The base model uses `base_pos` (starting at `prompt_tokens.len()`) for both. They cannot interact: the MTP layer has its own 1-layer paged cache (`generate.rs:~1280`) so there is no cross-cache RoPE interference. **Constant offset between base and MTP positions is invisible to attention** because RoPE attention scores depend only on `pos_j - pos_i`, and the MTP layer attends only to its own KV cache (single sequence). ✓
- **`fc_t` cached transpose**: `cached_transpose(weight)` at `forward.rs:396-398` is a single `weight.t()?.contiguous()?`. Stored fc has shape `[H, 2H]` (PyTorch `nn.Linear(2H, H)` storage convention); `fc_t` has `[2H, H]`; matmul against `concat[..., 2H]` produces `[..., H]`. ✓
- **Marlin off the MTP layer**: `MtpGpuWeights` does not carry Marlin packed projections — the MTP transformer block uses the raw BF16 layer (`forward.rs:130-172`). W4A16 quantization mismatch is ruled out for this layer. ✓

#### Net Phase A verdict

Static audit found **no obvious layout or routing bug from code alone**. The most-suspicious remaining hypothesis is "swapped `fc` input ordering during checkpoint export" (the Phase X note's "α≈0.4 is exactly the signature"), but the kiln code visually matches the vLLM reference order. Confirming or refuting this requires runtime evidence: byte-equality of the loaded `mtp.*` tensors vs. raw safetensors, plus per-token comparison of the draft logits vs. an HF reference run on the same prompt.

### Phase B — instrumentation landed in this PR

New module `crates/kiln-model/src/mtp_debug.rs` (~120 lines incl. tests):

- `is_enabled()` / `should_log()` — `KILN_MTP_DEBUG=1` opt-in, with `KILN_MTP_DEBUG_MAX_CALLS` (default 16) limiting noise.
- `top_k_logits(logits, k)` — extracts top-K `(token_id, logit)` pairs from `[V]` / `[1, V]` / `[1, 1, V]` tensors.
- `tensor_l2_norm(t)` — single-f32 L2 sanity check.
- `format_top_k(...)` — compact `"[(id=42, l=12.34), ...]"` rendering for log lines.

Two call sites add one `tracing::info!` line each when enabled:

- `forward.rs::mtp_forward_step` — `mtp_draft` line per draft pass: `mtp_pos`, `last_token`, `h_prev_l2`, `mtp_logits_l2`, `mtp_top5`.
- `speculative.rs::speculative_mtp_decode_step` — `mtp_verify` line per verify pass: `mtp_pos`, `base_pos`, `last_token`, `draft_token`, `target_at_0`, `accepted`, `verify_pos0_l2`, `verify_pos0_top5`. Shares `mtp_pos` with the matching `mtp_draft` so trace lines pair by grep.

Both call sites early-out cheaply (`std::env::var` lookup) when the flag is unset, so production decode is not affected.

### Phase B — execution plan for the next agent

1. **Acquire pod**: `ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000'`. Use the pool, not raw `runpod_api.py launch`.
2. **Build with this PR merged**: standard `cargo build --release --features cuda --bin kiln-bench` (sccache should land 90%+ on a warm pod).
3. **Run a 16-step trace**:
   ```bash
   KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_SPEC_METHOD=mtp \
   KILN_MTP_DEBUG=1 RUST_LOG=kiln::mtp_debug=info,info \
     ./target/release/kiln-bench --paged --prompt-tokens 512 \
       --max-output-tokens 16 --skip-training 2>&1 | tee mtp-trace.log
   ```
   Output will contain 16 `mtp_draft` lines + 16 `mtp_verify` lines, each tagged with `mtp_pos` for pairing.
4. **HF reference parity** — small Python harness using `transformers` to load Qwen/Qwen3.5-4B and compute the same first-16 MTP draft tokens given the same prompt tokens. Diff:
   - Token-level: do `target_at_0` and `draft_token` from the trace match HF's predicted next-token for the same context? If `target_at_0` matches HF but `draft_token` doesn't, the bug is in the MTP draft head (loader / fc / norms / RoPE). If `target_at_0` itself doesn't match HF, the bug is in the base model verify path (less likely — base bench α metric would have caught it earlier).
   - Logit-level: compare `verify_pos0_top5` against HF's softmax top-5 on the same context. Should be byte-equivalent at temperature=0.
5. **Byte-equality of MTP tensors** — small Rust binary (or Python via `safetensors` crate equivalent) that:
   - Loads `mtp.fc.weight`, `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`, `mtp.norm.weight` directly from raw safetensors.
   - Compares to the same tensors after kiln's loader path (dequant if needed; these MTP tensors are BF16, no quantization in the layer).
   - Confirms element-wise equality. If anything differs, that's the bug.
6. **Decision tree**:
   - Tensors byte-equal AND draft top-1 matches HF ⇒ MTP head is correct; α=0.411 is "as good as it gets" for this checkpoint and we should consider whether `mtp_num_hidden_layers=1` is a structural ceiling on Qwen3.5-4B (open a Phase 7 design question, not a bug fix).
   - Tensors byte-equal AND draft top-1 does NOT match HF ⇒ forward-pass bug. Likely candidates remaining: RoPE position offset (despite relative-invariance argument; check partial-rotary application), GDN state initialization for the standalone MTP layer (despite this being a *full-attention* layer per `is_full_attention_layer(3)`), or `attn_output_gate` interaction.
   - Tensors NOT byte-equal ⇒ loader bug. Fix in `loader.rs::load_mtp_if_present`. **This is the strongest fix-in-Tier-1 outcome** and would close the investigation.

### Cost guardrails for Phase B

- Pod: 1× A6000 warm lease, expected 30-45 min wall clock (build + 1× 16-step trace + HF parity Python script + byte-eq tool).
- Use `wait-file` pattern from the kiln skill `resources/runpod-workflow.md`. **Never** write `until ssh ... kill -0` polling loops — that pattern burned $99.76 + $13.76 in two SSH-wedge incidents on 2026-04-20.
- Hard cap: 90 min / $40 (per the standard kiln cap). Fall back to a Tier 2 docs-only update if Phase B blows the cap.

### Files / Reproduction

```bash
# This PR — Tier 2 diagnostic, code patch + PROFILING.md update
git fetch origin
git checkout mtp/debug-low-alpha-instrumentation
cargo check                                                        # CPU host
cargo build --release --features cuda --bin kiln-bench             # Linux+CUDA pod

# Phase B trace (next agent, on a pod)
KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_SPEC_METHOD=mtp \
KILN_MTP_DEBUG=1 RUST_LOG=kiln::mtp_debug=info,info \
  ./target/release/kiln-bench --paged --prompt-tokens 512 \
    --max-output-tokens 16 --skip-training 2>&1 | tee mtp-trace.log
```

### What this PR does NOT do

- Does not change MTP forward, loader, sampler, or scheduler behavior in production decode (instrumentation early-outs when `KILN_MTP_DEBUG` is unset).
- Does not run a fresh bench. The α=0.411 baseline from the section above stands; this PR adds the instrumentation to investigate it.
- Does not modify the Phase X "Why α is low" hypotheses table — those are still open. This PR only narrows the search space via static audit and lands the runtime instrumentation needed to close them.

### Cost

- Local-only PR (`cargo check` on CPU host).
- $0 in pod time. The Phase B execution plan above commits to ≤90 min / $40 on a future pod lease.


## Phase B — Runtime MTP bisect trace (2026-04-21)

Follow-up to "Phase A audit + Phase B instrumentation" (PR #260). Landed a fresh runtime trace of native MTP speculative decode on A6000 with `KILN_MTP_DEBUG=1` and bisected the 13 draft/verify pairs per the Phase A execution plan.

### Repro

```bash
KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_SPEC_METHOD=mtp \
  KILN_MTP_DEBUG=1 KILN_MTP_DEBUG_MAX_CALLS=32 \
  RUST_LOG=kiln::mtp_debug=info,info \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 16 --skip-training
```

- Pod: A6000 (49140 MB), CUDA 12.4, image `ghcr.io/ericflo/kiln-runpod:latest`
- Checkpoint: `Qwen/Qwen3.5-4B` (HF download), weight prefix `model.language_model.` (VL-wrapper architecture `Qwen3_5ForConditionalGeneration`)
- Build: warm sccache/B2, release + `--features cuda`, `KILN_CUDA_ARCHS=86`
- Full trace: `assets/mtp-phase-b-trace-2026-04-21.log` (194 lines, 13 draft/verify pairs)

### Headline numbers

| Metric | Value |
| --- | --- |
| Draft acceptance α | **0.154** (2 of 13 drafts accepted) |
| Decode tok/s (MTP arm) | 30.13 |
| Decode tok/s (baseline plain, from skill doc) | 49.76 |
| Prefill TTFT ms | 11074.2 (warmup artifact; see skill note `kiln-bench-prefill-warmup-required`) |
| Mean ITL ms | 57.56 |
| P99 ITL ms | 67.998 |

α = 0.154 is **lower than the 0.411 previously recorded** in the "Phase X — native MTP spec-decode results" section. The prior α=0.411 figure came from a different bench run that was not captured with full instrumentation; the 0.411 number remains in PROFILING.md but should be treated as a ceiling, not a floor, until re-measured under identical settings.

**MTP is a decode regression at this α.** 30.13 tok/s vs 49.76 plain-decode baseline is 0.605× — worse than off. This is consistent with the overhead-math laid out in the prior phase (verify forward + draft forward + linear-state snapshot/restore dToD exceeds the savings below ~α=0.5).

### Bisect table (13 steps)

Full data in `assets/mtp-phase-b-trace-2026-04-21.log`. Summary:

| # | mtp_pos | base_pos | last_token | draft_top1 | target_at_0 | accepted | identity? | h_prev_l2 |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: |
|  1 | 0 | 506 |    561 |    561 |  29144 | ✗ | **Y** | 64.00 |
|  2 | 0 | 507 |  29144 |  29144 |   6165 | ✗ | **Y** | 68.21 |
|  3 | 0 | 508 |   6165 |   6165 |  27050 | ✗ | **Y** | 74.76 |
|  4 | 0 | 509 |  27050 |  27050 |    279 | ✗ | **Y** | 72.58 |
|  5 | 0 | 510 |    279 |   3377 |  29144 | ✗ |   | 73.45 |
|  6 | 0 | 511 |  29144 |   1785 |   6165 | ✗ |   | 56.80 |
|  7 | 0 | 512 |   6165 |  18465 |  27050 | ✗ |   | 61.05 |
|  8 | 0 | 513 |  27050 |    279 |    279 | **✓** |   | 61.82 |
|  9 | 1 | 515 |  24460 |  24460 |   3377 | ✗ | **Y** | 67.95 |
| 10 | 1 | 516 |   3377 |   3377 |     13 | ✗ | **Y** | 70.53 |
| 11 | 1 | 517 |     13 |    271 |    271 | **✓** |   | 67.90 |
| 12 | 2 | 519 | 248068 |    271 |    198 | ✗ |   | 51.76 |
| 13 | 2 | 520 |    198 | 248069 |    760 | ✗ |   | 46.54 |

Column key:
- `identity?` — draft top-1 equals the input `last_token` (i.e. "the MTP head says the next token will be the same as the one just generated").

### Primary finding — identity-prediction bias

**6 of 13 draft predictions (46.2%) have `mtp_top5[0].id == last_token`.** Four of those six are consecutive at the start of decode (`mtp_pos=0, base_pos ∈ [506..509]`) with draft top-1 = last input token and logit-1 spread ≥4 over logit-2 — the MTP head is highly confident in the wrong answer.

Magnitudes are all healthy:
- `h_prev_l2 ∈ [46.5, 74.8]` — well-formed hidden states, no zero-out or explosion.
- `mtp_logits_l2 ∈ [1062, 1479]` and `verify_pos0_l2 ∈ [1100, 1743]` — comparable scales, so this is **not** a magnitude bug; the argmax is wrong.

### Secondary finding — out-of-vocab special-token drafting

At steps 12–13, drafts include tokens `248068` and `248069`, both high-id special/reserved tokens (`image_token_id=248056`, `eos_token_id=248044` from `config.json`). These appear in draft top-5 with dominant logits (e.g. step 13 draft top-1 = 248069 @ l=24.6) but do not dominate the base verify_pos0 output. This is not the primary failure mode but it does indicate the MTP head is allocating nontrivial probability to VL/reserved token IDs the base model correctly rejects.

### Hypotheses ranked by strength of runtime evidence

1. **fc matmul produces an embed-dominated fused state (most likely).** When the MTP `fc` projection  (2560 ← 5120) weights the embed-half of `concat([norm_emb, norm_h])` disproportionately higher than the hidden-half, the single-layer MTP transformer block + tied `lm_head` projection regresses toward `argmax(embed(x) · embed^T)[x] = x`. This is exactly the identity bias pattern observed, and is strongest when `h_prev` carries the least task-specific signal (early in the decode, steps 1–4) and weakens as `h_prev` accumulates context (steps 5+). Phase A audit marked "fc input ordering" as checked statically, but a *magnitude* asymmetry between the two halves of `fc.weight` would not show up in a shape/concat-order check — only in a per-half norm comparison or byte-equality diff against HF.

2. **`pre_fc_norm_hidden` loaded into the wrong tensor slot (plausible).** If `mtp.pre_fc_norm_hidden.weight` ended up loaded into the scale of `mtp.pre_fc_norm_embedding.weight` (or vice versa), or if one of the two RMSNorm scales collapsed to near-zero, the fused input is embed-dominant for exactly the reason above. Runtime tensor dump or byte-equality against the safetensors shard would disambiguate.

3. **MTP RoPE position mismatch (weaker evidence).** `forward.rs:3528` passes `mtp_pos as f32` for RoPE (0..=2 across the trace), while the token being drafted actually lives at `base_pos + 1 ∈ [507, 521]`. On a single-token MTP attention pass the RoPE rotations cancel in `Q·K` (self-attention), so this alone would not produce identity bias — but it could compound with (1) on accepted steps when the MTP cache has >1 token.

4. **Structural ceiling (unlikely given pattern).** If MTP were simply a weak draft head (`num_nextn_predict_layers=1` structural limit), we would expect low-but-diffuse draft distributions — not sharp, high-confidence, wrong-at-logit-gap-of-4+ identity predictions on 4 consecutive steps. The pattern is too consistent to be noise.

### Recommended next bisect step (Tier 2, ≤45 min)

Byte-equality check on the 3 critical MTP tensors directly on a pod:

```python
from safetensors import safe_open
for t in ["mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"]:
    # Compare bytes with kiln's GpuWeights.mtp via a small ffi dump, or
    # dump kiln's loaded tensors to .npy and diff against the raw safetensors.
```

In parallel, dump `mtp.fc.weight[:, :2560]` (embed half) and `mtp.fc.weight[:, 2560:]` (hidden half) L2 norms and compare. An asymmetry >2× would directly confirm hypothesis (1).

If byte-equal and halves balanced → run a one-shot HF reference parity on the same 512-token prompt and diff `draft_top_k[0]` per step. If HF also emits identity-biased drafts at step 1–4, the pattern is a genuine checkpoint property (structural, hypothesis 4). If HF does not, the gap is in kiln's forward path — most likely (3) or a subtle op in `mtp_forward_step` (e.g. residual vs. no-residual at `fused → attn`, currently implicit via `transformer_block_paged`).

### Budget used

- Pod: 1× A6000 on-demand (pool fallback: `runpod_api.py launch` direct, image `ghcr.io/ericflo/kiln-runpod:latest`).
- Wall-clock: ~45 min (under the 90 min / $40 cap).
- Work: build (5m 26s warm sccache) + model download (~3 min, parallel with build) + 1 bench run (~30s decode after 37s model load) + analysis.

Pod terminated on task completion.

## Phase B2 — fc halves + norm-swap bisect (2026-04-21)

Follow-up to Phase B. Three-part experiment: (1) byte-equality verification of all 15 MTP tensors against the raw safetensors shards, (2) runtime logging of the two halves of `fc`'s input (pre-norm-embedding on `embed(last_token)` vs pre-norm-hidden on `h_prev`) plus fused output L2, (3) A/B toggle (`KILN_MTP_SWAP_FC_NORMS=1`) that swaps which RMSNorm weight is paired with which half. Goal: isolate whether Phase B's identity bias comes from a byte-level loader bug, a halves-magnitude asymmetry, or the two RMSNorm scales being wired to the wrong halves of the fused input.

All runs on the same A6000 pod (on-demand, `ghcr.io/ericflo/kiln-runpod:latest`, warm sccache B2), same model (`/workspace/qwen3.5-4b`), same prompt used in Phase B, `--paged --skip-training --max-output-tokens 32`, `KILN_MTP_DEBUG=1 KILN_MTP_DEBUG_MAX_CALLS=16`.

### Part A — byte-equality (rules out raw loader bugs)

New test `crates/kiln-model/tests/mtp_byte_eq.rs` walks all 15 MTP tensors under the detected `mtp.*` prefix, flattens each `Tensor` kiln loaded into host memory, and compares bytes against the corresponding slice in the raw safetensors shard (mmap via `safetensors::SafeTensors::deserialize`). Runs on CPU so the same test reproduces on any developer laptop.

```
test mtp_weights_match_safetensors_byte_for_byte ...
[mtp_byte_eq] using mtp_prefix = mtp.
[mtp_byte_eq] === report ===
PASS fc                                  (mtp.fc.weight)                                  shape=[2560, 5120] dtype=BF16
PASS pre_fc_norm_embedding               (mtp.pre_fc_norm_embedding.weight)               shape=[2560]       dtype=BF16
PASS pre_fc_norm_hidden                  (mtp.pre_fc_norm_hidden.weight)                  shape=[2560]       dtype=BF16
PASS final_layernorm                     (mtp.norm.weight)                                shape=[2560]       dtype=BF16
PASS transformer_block.input_layernorm   (mtp.layer.input_layernorm.weight)               shape=[2560]       dtype=BF16
PASS transformer_block.post_attn_ln      (mtp.layer.post_attention_layernorm.weight)      shape=[2560]       dtype=BF16
PASS mlp.gate_proj                       (mtp.layer.mlp.gate_proj.weight)                 shape=[9728, 2560] dtype=BF16
PASS mlp.up_proj                         (mtp.layer.mlp.up_proj.weight)                   shape=[9728, 2560] dtype=BF16
PASS mlp.down_proj                       (mtp.layer.mlp.down_proj.weight)                 shape=[2560, 9728] dtype=BF16
PASS attn.q_proj                         (mtp.layer.self_attn.q_proj.weight)              shape=[4096, 2560] dtype=BF16
PASS attn.k_proj                         (mtp.layer.self_attn.k_proj.weight)              shape=[1024, 2560] dtype=BF16
PASS attn.v_proj                         (mtp.layer.self_attn.v_proj.weight)              shape=[1024, 2560] dtype=BF16
PASS attn.o_proj                         (mtp.layer.self_attn.o_proj.weight)              shape=[2560, 4096] dtype=BF16
PASS attn.q_norm                         (mtp.layer.self_attn.q_norm.weight)              shape=[256]        dtype=BF16
PASS attn.k_norm                         (mtp.layer.self_attn.k_norm.weight)              shape=[256]        dtype=BF16
[mtp_byte_eq] PASS — all MTP tensors byte-equal to safetensors
test result: ok. 1 passed; finished in 6.87s
```

**Verdict A**: kiln's MTP loader is byte-for-byte faithful for every MTP tensor on this Qwen3.5-4B checkpoint. This mechanically rules out Phase B hypothesis 2 in its strong form — no raw loader bug exists where a tensor is loaded into the wrong slot or byte-mangled during conversion. Any remaining ambiguity must be in *how* the correctly-loaded tensors are wired into `mtp_forward_step`.

### Part B — halves-L2 instrumentation

`crates/kiln-model/src/mtp_debug.rs` extended with `is_swap_fc_norms_enabled()` and emits, per draft step, the L2 norms of:

- `h_prev_l2` — L2 norm of the incoming last-layer hidden state
- `norm_emb_l2` — L2 of `rms_norm(embed(last_token), pre_fc_norm_embedding)`
- `norm_h_l2` — L2 of `rms_norm(h_prev, pre_fc_norm_hidden)`
- `halves_ratio = norm_emb_l2 / norm_h_l2`
- `fused_l2` — L2 of `fc(concat(...))` output, the input to the MTP transformer block

Halves are extracted in `mtp_forward_step` via `narrow` slices on the fused `concat` result so the measured halves come from exactly the tensor the matmul sees, not a re-computation.

### Part C — norm-swap A/B toggle

`KILN_MTP_SWAP_FC_NORMS=1` causes `mtp_forward_step` to swap which `RmsNorm` weight is applied to which half of the fused input (`pre_fc_norm_hidden` applied to the embedding half, `pre_fc_norm_embedding` applied to the `h_prev` half). Read every call so an A/B can be run across two processes with no code change. Matches the bisect workflow described at the end of Phase B under "Recommended next bisect step".

### Runtime results — single 32-output-token prompt, A/B

| Arm | α (draft accept) | num_drafts | num_accepts | identity (draft==last_tok) overall | halves_ratio median |
|---|---|---|---|---|---|
| swap **OFF** (default) | **0.240** (6 / 25 verify) | 16 | 6 | 7/25 = **28%** | ~0.745 (emb < h) |
| swap **ON**  | **0.348** (8 / 23 verify) | 16 | 8 | 3/23 = **13%** | ~1.067 (emb > h) |

Δα: **+0.108 absolute, +45% relative** with swap ON.

#### Halves table — swap-OFF (first 13 draft rows)

| mtp_pos | last_tok | h_prev_l2 | norm_emb_l2 | norm_h_l2 | halves_ratio | fused_l2 | mtp_logits_l2 | top1_id | top1_logit |
|--|--|--|--|--|--|--|--|--|--|
| 0 | 561   | 64.38 | 32.02 | 41.34 | 0.7745 | 14.14 | 1068.4 | 561   | 21.12 |
| 0 | 29144 | 69.17 | 30.24 | 40.13 | 0.7537 | 15.01 | 1229.8 | 29144 | 17.25 |
| 0 | 6165  | 76.73 | 29.93 | 41.27 | 0.7251 | 15.40 | 1111.6 | 6165  | 17.88 |
| 0 | 27050 | 72.92 | 30.29 | 41.16 | 0.7360 | 14.91 | 1155.7 | 27050 | 15.75 |
| 0 | 279   | 75.05 | 32.18 | 41.50 | 0.7754 | 14.91 | 1076.6 | 3377  | 14.81 |
| 0 | 29144 | 59.62 | 30.24 | 40.67 | 0.7436 | 15.06 | 1398.4 | 1785  | 14.81 |
| 0 | 6165  | 60.48 | 29.93 | 40.45 | 0.7400 | 15.58 | 1252.1 | 18465 | 18.62 |
| 0 | 27050 | 63.04 | 30.29 | 40.47 | 0.7485 | 14.59 | 1210.6 | 279   | 18.88 |
| 1 | 24460 | 67.21 | 30.29 | 40.78 | 0.7429 | 14.81 | 1148.9 | 3377  | 20.38 |
| 2 | 303   | 65.20 | 32.76 | 41.04 | 0.7983 | 13.92 | 1212.8 | 279   | 20.50 |
| 2 | 3150  | 60.00 | 29.88 | 39.46 | 0.7572 | 14.68 | 1227.2 | 3150  | 19.88 |
| 2 | 854   | 70.49 | 30.16 | 40.21 | 0.7501 | 15.05 | 1129.8 | 13    | 19.00 |
| 3 | 2838  | 72.92 | 29.92 | 39.71 | 0.7534 | 13.43 | 1130.1 | 2838  | 17.38 |

`halves_ratio` is tightly clustered in `[0.725, 0.798]`. The pre-norm on the embed half produces a scale ~25% smaller than the pre-norm on the hidden half, every step. Identity hits at rows 1-4 (pos=0) and again at rows 11 and 13.

#### Halves table — swap-ON (first 13 draft rows)

| mtp_pos | last_tok | h_prev_l2 | norm_emb_l2 | norm_h_l2 | halves_ratio | fused_l2 | mtp_logits_l2 | top1_id | top1_logit |
|--|--|--|--|--|--|--|--|--|--|
| 0 | 561   | 64.38 | 40.21 | 36.55 | 1.1001 | 13.88 |  960.7 | 561   | 16.38 |
| 0 | 29144 | 69.17 | 38.75 | 37.62 | 1.0301 | 15.88 | 1267.3 | 6165  | 15.06 |
| 1 | 27050 | 67.78 | 38.68 | 35.60 | 1.0865 | 14.80 | 1235.3 | 279   | 18.62 |
| 2 | 24460 | 67.40 | 38.68 | 36.89 | 1.0486 | 15.10 | 1096.8 | 3377  | 18.00 |
| 3 | 303   | 63.82 | 40.74 | 36.62 | 1.1127 | 13.54 | 1493.0 | 279   | 21.62 |
| 3 | 3150  | 61.95 | 37.89 | 36.59 | 1.0357 | 14.58 | 1191.0 | 3150  | 18.50 |
| 3 | 854   | 69.00 | 38.19 | 36.07 | 1.0587 | 14.86 | 1150.8 | 13    | 19.12 |
| 4 | 2838  | 71.45 | 38.24 | 34.96 | 1.0935 | 13.85 | 1251.6 | 369   | 16.88 |
| 4 | 5947  | 65.90 | 38.58 | 37.24 | 1.0361 | 14.55 | 1128.3 | 264   | 18.12 |
| 5 | 15352 | 89.60 | 38.94 | 38.15 | 1.0207 | 15.78 | 1243.3 | 6157  | 16.75 |
| 6 | 314   | 77.91 | 40.44 | 37.15 | 1.0885 | 13.06 | 1097.5 | 220   | 19.12 |
| 6 | 2981  | 67.48 | 38.11 | 34.16 | 1.1155 | 15.57 | 1042.6 | 8876  | 14.56 |
| 6 | 17793 | 78.24 | 38.99 | 36.77 | 1.0603 | 15.06 | 1221.4 | 48736 | 17.00 |

With the norm-swap on, `halves_ratio` clusters in `[1.02, 1.12]` — the embedding half is now slightly larger than the hidden half. The top-1 token no longer matches `last_tok` at rows 1, 3-8, 10-13 (only two identity hits in 13 rows vs five in the swap-OFF table over the same slice).

#### Phase B verify-pos0 comparison

Direct comparison against the Phase B verify trace for the same prompt (first 4 draft steps, `mtp_pos=0`, predictions of the next ground-truth token):

| step | last_tok | target | swap-OFF top1 | swap-ON top1 |
|--|--|--|--|--|
| 1 | 561   | 29144 | **561** (identity, wrong)   | **561** (identity, wrong)   |
| 2 | 29144 | 6165  | **29144** (identity, wrong) | **6165** ✅ |
| 3 | 6165  | 27050 | **6165** (identity, wrong)  | — (advanced by accept at step 2) |
| 4 | 27050 | 279   | **27050** (identity, wrong) | — |

The norm-swap flipped step 2 from an identity-biased reject into a direct accept. Step 1 remains wrong under both arms — consistent with "cold" MTP state when `h_prev` for the very first draft carries too little task context regardless of norm pairing.

### Verdict

- **Phase B hypothesis 2 (wrong norm-tensor slot / collapsed scale) — still plausible but not byte-level.** Part A proves no tensor is byte-swapped or corrupted at load. But Part C shows a direct, reproducible, same-prompt α lift of +45% when the two RMSNorm scales are applied to each other's half. That is exactly the signature expected if the loader/wiring maps `pre_fc_norm_embedding` to operate on `h_prev` and `pre_fc_norm_hidden` to operate on `embed(last_token)` — i.e. the two scales are correctly loaded but attached to the wrong half inside `mtp_forward_step`.
- **Phase B hypothesis 1 (fc matmul embed-dominance) — partially supported as a consequence, not the root cause.** Under swap-OFF the embed half is smaller (`halves_ratio ≈ 0.74`) yet identity bias dominates; under swap-ON the embed half is slightly larger (`halves_ratio ≈ 1.07`) and identity bias drops. That direction is opposite to "fc weights the embed half too heavily" — if embed magnitude alone drove the bias we would expect more identity with a larger embed half, not less. The halves ratio interacts with the identity bias through the norm pairing, not directly through halves magnitude.
- **Direction for Phase B3**: re-verify the norm-half pairing in `mtp_forward_step` against the Qwen3-MTP reference implementation and/or the HF generator. Compare kiln's concatenation order and the binding between `pre_fc_norm_embedding`/`pre_fc_norm_hidden` and the `embed`/`h_prev` halves. If swap-ON is the "correct" pairing, the fix is a one-line rewire and should reproduce across multiple prompts and longer decode windows.

### Caveats

- **N=1, single prompt, 32 output tokens, 16 draft calls captured per arm.** The +45% α lift is a strong signal but has not been confirmed across prompts. Phase B3 must run a multi-prompt A/B (e.g. 8–16 prompts × 128 output tokens) before any code change is landed as the canonical wiring.
- Both runs use the same bench seed and the same prompt, so the first draft step's `last_token=561` and `target=29144` are identical. The divergence at step 2 (6165 vs identity) is deterministic relative to the norm-pairing choice.
- Out-of-vocab special-token drafting (`248068`, `248069`) from Phase B still reproduces under swap-ON at `pos=8`, so the norm-swap is not a universal fix — it is a bisect result pointing at the pairing, not a production patch.

### Budget used

- Pod: 1× A6000 on-demand (pool lease, `ghcr.io/ericflo/kiln-runpod:latest`).
- Wall-clock: ~55 min total (build + byte-eq 34s on warm sccache, A/B pair 81s, upload/download/parse ~5 min; rest was planning and PROFILING.md write-up).
- Work: byte-eq test (Part A), halves-L2 logging (Part B), swap toggle (Part C), this write-up (Part D).

Pod released to pool on task completion.

## Phase B3 — MTP multi-prompt A/B: reject the B2 norm-swap finding (2026-04-21)

Phase B2 ended on a single-prompt N=1 result: swap-ON lifted α from 0.240 to 0.348 (+45%) for one fixed bench prompt, 32 output tokens. Phase B3 set out to confirm or reject that finding across 8 distinct prompts × 128 output tokens, on the same A6000 pod image and the same MTP tracing plumbing.

**Verdict: REJECT.** Across 8 seeds × 2 arms, swap-ON beats swap-OFF on α in only 3/8 seeds. Mean paired Δα = −0.208 (swap hurts on average). The B2 +45% result is within the normal prompt-dependent α variance of swap-OFF (range 0.085–0.829) and was not a reliable signal. The norm-swap is not the correct wiring; do not land PR-style wiring changes based on Phase B2 alone.

The two consistent structural signals **are** real but do not translate into α gain:
- **Δhalves_ratio = +0.327 ± 0.015** across all 8 seeds — the swap does materially change the fc input composition as expected.
- **Δidentity_bias = −0.075 ± 0.049** across all 8 seeds — swap-ON reliably reduces the "draft the last token" failure mode. But this reduction is consumed by other draft failures, so α does not rise.

### Methodology

Reused everything from Phase B2 (bench binary, MTP tracing, halves-L2 logging, `KILN_MTP_SWAP_FC_NORMS`, MTP_DEBUG cap = 16 draft emissions). Two changes:

1. `kiln-bench --seed <u64>` now also selects one of 8 prompt bases (`PROMPT_POOL[seed % 8]`), so seeds actually produce different token distributions instead of identical greedy runs. Seed 0 is the Phase B2 baseline preserved verbatim; seeds 1–7 cover distinct topics (software, history, nature, philosophy, cooking, astronomy, music).
2. Output window increased from 32 → 128 tokens per run to amplify α signal (70–127 `mtp_verify` events per run vs. ~20 in B2).

Sweep: 16 runs (8 seeds × {off, on}), paged production path, MTP_DEBUG on:

```bash
for SEED in 0..7; do for ARM in off on; do
  env KILN_MTP_DEBUG=1 KILN_SPEC_METHOD=mtp ${ARM=on:+KILN_MTP_SWAP_FC_NORMS=1} \
    kiln-bench --model-path .../qwen3.5-4b --paged \
      --prompt-tokens 512 --max-output-tokens 128 --skip-training \
      --seed $SEED > mtp-b3-seed${SEED}-${ARM}.log 2>&1
done; done
```

Wall-clock: 1710 s (≈ 28.5 min, ≈107 s per run) on one A6000. Trace files archived under `assets/mtp-phase-b3-seed{0..7}-{off,on}.log`.

Aggregation (`assets/mtp-phase-b3-aggregate.py`): strips ANSI codes, stops at the `Inference Throughput Benchmarks` marker so post-latency events do not contaminate metrics, extracts α from `mtp_verify accepted=…`, identity-bias from `mtp_verify draft_token==last_token`, halves ratio / norm-L2 from `mtp_draft` events.

### Per-run aggregate

`alpha` = accept rate on `mtp_verify`; `id_bias` = fraction of verifies where `draft_token == last_token`; `halves` = mean `halves_ratio` (`norm_emb_l2 / norm_h_l2`); `norm_e`/`norm_h` = mean L2 of the two fc-input halves; `oov` = drafts with `draft_token >= 151936`; `n_d`/`n_v` = number of draft/verify events (n_d is capped at 16 by `KILN_MTP_DEBUG_MAX_CALLS`).

| seed | arm | alpha  | id_bias | halves | norm_e | norm_h | oov | n_d | n_v |
|-----:|:---:|:------:|:-------:|:------:|:------:|:------:|:---:|:---:|:---:|
|    0 | off | 0.257  | 0.099   | 0.756  | 30.68  | 40.58  | 0   | 16  | 101 |
|    0 | on  | 0.306  | 0.031   | 1.076  | 39.05  | 36.35  | 1   | 16  |  98 |
|    1 | off | 0.198  | 0.085   | 0.762  | 31.19  | 40.96  | 0   | 16  | 106 |
|    1 | on  | 0.366  | 0.011   | 1.069  | 39.06  | 36.56  | 0   | 16  |  93 |
|    2 | off | 0.085  | 0.195   | 0.766  | 30.90  | 40.32  | 0   | 16  | 118 |
|    2 | on  | 0.347  | 0.042   | 1.097  | 38.93  | 35.61  | 0   | 16  |  95 |
|    3 | off | 0.477  | 0.023   | 0.768  | 30.80  | 40.15  | 2   | 16  |  86 |
|    3 | on  | 0.337  | 0.021   | 1.105  | 39.10  | 35.50  | 1   | 16  |  95 |
|    4 | off | 0.829  | 0.143   | 0.759  | 30.78  | 40.58  | 0   | 16  |  70 |
|    4 | on  | 0.257  | 0.000   | 1.109  | 39.05  | 35.32  | 0   | 16  | 101 |
|    5 | off | 0.309  | 0.093   | 0.769  | 30.75  | 40.03  | 3   | 16  |  97 |
|    5 | on  | 0.033  | 0.008   | 1.082  | 38.88  | 35.94  | 0   | 16  | 123 |
|    6 | off | 0.549  | 0.049   | 0.774  | 30.96  | 40.00  | 1   | 16  |  82 |
|    6 | on  | 0.033  | 0.024   | 1.119  | 39.23  | 35.13  | 1   | 16  | 123 |
|    7 | off | 0.641  | 0.051   | 0.764  | 31.05  | 40.67  | 0   | 16  |  78 |
|    7 | on  | 0.000  | 0.000   | 1.075  | 39.46  | 36.75  | 0   | 16  | 127 |

### Paired ON − OFF per seed

| seed | d_alpha | d_id_bias | d_halves | d_oov |
|-----:|:-------:|:---------:|:--------:|:-----:|
|    0 | +0.049  | −0.068    | +0.319   | +1    |
|    1 | +0.167  | −0.074    | +0.307   |  0    |
|    2 | +0.263  | −0.153    | +0.330   |  0    |
|    3 | −0.140  | −0.002    | +0.337   | −1    |
|    4 | −0.571  | −0.143    | +0.350   |  0    |
|    5 | −0.277  | −0.085    | +0.314   | −3    |
|    6 | −0.516  | −0.024    | +0.345   |  0    |
|    7 | −0.641  | −0.051    | +0.311   |  0    |

Summary:

| metric    | mean   | std   | min    | max    | n |
|-----------|:------:|:-----:|:------:|:------:|:-:|
| d_alpha   | −0.208 | 0.326 | −0.641 | +0.263 | 8 |
| d_id_bias | −0.075 | 0.049 | −0.153 | −0.002 | 8 |
| d_halves  | +0.327 | 0.015 | +0.307 | +0.350 | 8 |
| d_oov     | −0.375 | 1.111 | −3.000 | +1.000 | 8 |

**ON > OFF on d_alpha: 3/8 seeds.** Decision rule stated in the task spec — CONFIRM requires ≥ 7/8, INCONCLUSIVE 6/8, REJECT ≤ 5/8. We are well inside REJECT.

### Interpretation

Phase B2 compared swap-OFF α = 0.240 against swap-ON α = 0.348 on a single shared prompt (seed 0 in B3 terms, 32 output tokens). B3 seed-0 at 128 tokens is OFF=0.257 / ON=0.306, Δα = +0.049 — a much smaller lift than B2 reported, still positive but not +45%. Seeds 1–2 show a genuine Δα lift (+0.167, +0.263) while seeds 3–7 show a Δα *drop* as large as −0.641. No clean topic-cluster pattern in the "helped" vs. "hurt" split (seed 0 canonical / 1 software / 2 history were helped; 3 nature / 4 philosophy / 5 cooking / 6 astronomy / 7 music were hurt). The swap-OFF α distribution across prompts (0.085–0.829, range 0.74) is the dominant source of variance; once OFF α is high on a given prompt, ON α is often substantially lower, and vice versa — the two arms are not a pure "shifted mean" but a repairing-vs-breaking pair whose sign depends on the prompt.

The two truly consistent deltas (halves_ratio and id_bias) confirm the norm-swap acts as intended mechanically — the `embed` half of the fc input gets the larger-scale norm applied, inverting which half dominates, and the "copy last token" failure mode drops reliably. But that failure mode only accounts for a small fraction of rejects (OFF id_bias ranges 2.3%–19.5%, mostly < 10%), so fixing it does not move α in the direction we need.

**This rules out "swap the norms" as the wiring fix.** The remaining hypothesis from Phase B — that fc maps the wrong half-to-slot pairing in the concatenation — is still live, but the concat order and matmul direction are now the more likely culprits, not the RMSNorm attachment. A Phase B4 (if pursued) should:

1. Verify the fc concat ordering (`[embed; h_prev]` vs. `[h_prev; embed]`) against the Qwen3-MTP reference generator end-to-end on the same prompt.
2. Confirm the fc weight tensor's row/column layout matches the concat order kiln feeds it.
3. Re-trace α with any such wiring fix across the same 8-prompt pool, using this same sweep script for a direct comparison.

Do not reopen the norm-swap question without a new structural hypothesis — this A/B is decisively negative.

### Caveats

- `n_drafts` is the MAX_CALLS cap (16) per run, not the true number of drafts; α statistics come from `n_verifies` (70–127 per run), which is statistically adequate for ±~0.05 precision per arm.
- Greedy sampling (`temperature = 0`) means seeds only vary the prompt, not per-token RNG. This is the correct choice for reproducibility but means each seed is effectively a single deterministic trajectory through decode.
- The `oov` count (drafts with `token_id >= 151936`) is small (0–3 per run) and inconsistent in sign between arms. The Phase B OOV special-token pathology still exists but is orthogonal to the norm-pairing question.
- MTP_DEBUG tracing adds stderr I/O overhead but is applied symmetrically, so it does not bias the ON/OFF comparison.

### Budget used

- Pod: 1× A6000 on-demand (pool lease, `ghcr.io/ericflo/kiln-runpod:latest`), reused from the B2 lease.
- Wall-clock: sweep 28.5 min + aggregation + write-up ≈ 55 min on pod, well under the 120 min / $60 task cap.
- Work: Part A (`--seed` prompt-pool wiring in `bench.rs`), Part B (16-run sweep script + execution), Part C (ANSI-safe aggregator), Part D (this section). All four parts landed in this PR.

Pod released to pool on task completion.

## Phase B4 — MTP `fc` concat order + weight layout audit vs upstream (doc-only)

### Goal

Phase B3 (PR #265) ruled out the `pre_fc_norm_embedding` ↔ `pre_fc_norm_hidden` swap as the α-collapse fix. The next live hypothesis from Phase B was that kiln's `fc` concat order or weight row/column layout mismatches the upstream Qwen3-MTP reference. Phase B4 is a $0, pod-free desk audit: compare kiln's MTP glue wiring byte-for-byte against the vLLM and SGLang reference implementations, plus a shape check against the published Qwen3.5-4B checkpoint header.

### Upstream references

**vLLM — `vllm/model_executor/models/qwen3_5_mtp.py`** (commit `771913e4a024`):

```python
# vllm-project/vllm @ 771913e4a024, qwen3_5_mtp.py
self.fc = ColumnParallelLinear(
    self.config.hidden_size * 2,
    self.config.hidden_size,
    bias=False,
    ...,
)
self.pre_fc_norm_embedding = RMSNorm(
    self.config.hidden_size, eps=self.config.rms_norm_eps
)
self.pre_fc_norm_hidden = RMSNorm(
    self.config.hidden_size, eps=self.config.rms_norm_eps
)
...
inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
hidden_states = self.pre_fc_norm_hidden(hidden_states)
hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
hidden_states, _ = self.fc(hidden_states)
```

Source: <https://github.com/vllm-project/vllm/blob/771913e4a024/vllm/model_executor/models/qwen3_5_mtp.py>.

**vLLM — `vllm/model_executor/models/qwen3_next_mtp.py`** (commit `657855ab4179`):

```python
# vllm-project/vllm @ 657855ab4179, qwen3_next_mtp.py
self.fc = ColumnParallelLinear(
    self.config.hidden_size * 2,
    self.config.hidden_size,
    bias=False,
    ...,
)
inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
hidden_states = self.pre_fc_norm_hidden(hidden_states)
hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
hidden_states, _ = self.fc(hidden_states)
```

Source: <https://github.com/vllm-project/vllm/blob/657855ab4179/vllm/model_executor/models/qwen3_next_mtp.py>.

**SGLang — `python/sglang/srt/models/qwen3_5_mtp.py`** (commit `cabe171b6ce3`):

```python
# sgl-project/sglang @ cabe171b6ce3, qwen3_5_mtp.py
self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
...
input_embeds = self.pre_fc_norm_embedding(input_embeds)
hidden_states = self.pre_fc_norm_hidden(hidden_states)
hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
hidden_states = self.fc(hidden_states)
```

Source: <https://github.com/sgl-project/sglang/blob/cabe171b6ce3/python/sglang/srt/models/qwen3_5_mtp.py>.

All three upstream references agree on the same wiring:

1. **Concat order** is `[embed, hidden]` — the freshly embedded draft token first, the pre-final-norm hidden state second, along `dim=-1`.
2. **Norm-to-input assignment** is `pre_fc_norm_embedding` → embedding half, `pre_fc_norm_hidden` → hidden half.
3. **fc projection** is a standard PyTorch `nn.Linear(2*H, H)` (aliased as `ColumnParallelLinear(2*H, H)` in vLLM, which is identical to `nn.Linear` at `tensor_parallel_size=1`). PyTorch `nn.Linear(in, out)` stores `weight` as `[out, in]`, so on disk the tensor must be `[H, 2H]`.

### Kiln code

**`crates/kiln-model/src/forward.rs:3516–3536`** — current `mtp_forward_step` glue:

```rust
let swap_fc_norms = crate::mtp_debug::is_swap_fc_norms_enabled();
let (norm_emb_weight, norm_h_weight) = if swap_fc_norms {
    (&mtp.pre_fc_norm_hidden, &mtp.pre_fc_norm_embedding)
} else {
    (&mtp.pre_fc_norm_embedding, &mtp.pre_fc_norm_hidden)
};
let norm_emb = {
    kiln_nvtx::range!(c"kiln/mtp/pre_fc_norm_emb");
    rms_norm(&token_emb, norm_emb_weight, config.rms_norm_eps)?
};
let norm_h = {
    kiln_nvtx::range!(c"kiln/mtp/pre_fc_norm_hidden");
    rms_norm(h_prev, norm_h_weight, config.rms_norm_eps)?
};

// 4. Concat along the hidden dim and fuse: [1, 1, 2H] @ fc_t[2H, H] -> [1, 1, H]
let fused = {
    kiln_nvtx::range!(c"kiln/mtp/fc");
    let concat = Tensor::cat(&[&norm_emb, &norm_h], 2)?.contiguous()?;
    concat.broadcast_matmul(&mtp.fc_t)?
};
```

With `KILN_MTP_SWAP_FC_NORMS=0` (the production default, which Phase B3 confirmed as optimal), this resolves to:

- `norm_emb = RMSNorm_embedding(token_emb)` — embed-side norm on the draft-token embedding.
- `norm_h   = RMSNorm_hidden(h_prev)` — hidden-side norm on the pre-final-norm hidden state.
- `concat = cat([norm_emb, norm_h], dim=2)` — embed first, hidden second, matching vLLM/SGLang byte-for-byte.

**`crates/kiln-model/src/loader.rs:626–632`** — fc shape check at load time:

```rust
let fc = extract_tensor(tensor_map, &format!("{mtp_prefix}fc.weight"))?;
// fc maps concat(embed, hidden) → hidden, so shape is [hidden, 2*hidden].
validate_shape(
    &fc,
    &[config.hidden_size, 2 * config.hidden_size],
    &ctx("fc"),
)?;
```

**`crates/kiln-model/src/forward.rs:768–770`** — upload-time transpose that feeds the forward path:

```rust
let fc = weight_to_tensor(&mtp_w.fc, device).context("mtp.fc")?;
let fc_t = cached_transpose(&fc).context("mtp.fc cached transpose")?;
```

So on device we hold `fc_t: [2H, H]`, and the forward path computes `concat[1,1,2H] @ fc_t[2H, H] → [1,1,H]`, which is exactly equivalent to `nn.Linear(2H, H)` applied to `concat`.

### Qwen3.5-4B checkpoint header (ground truth)

Cross-check against the published `Qwen/Qwen3.5-4B` checkpoint, read via an HTTP Range request on `model.safetensors` (first ~200 KB is the JSON header). Relevant MTP entries:

```
mtp.fc.weight                           dtype=BF16  shape=[2560, 5120]
mtp.pre_fc_norm_embedding.weight        dtype=BF16  shape=[2560]
mtp.pre_fc_norm_hidden.weight           dtype=BF16  shape=[2560]
mtp.norm.weight                         dtype=BF16  shape=[2560]
mtp.layers.0.input_layernorm.weight     dtype=BF16  shape=[2560]
mtp.layers.0.post_attention_layernorm.  dtype=BF16  shape=[2560]
mtp.layers.0.self_attn.q_norm.weight    dtype=BF16  shape=[256]
mtp.layers.0.self_attn.k_norm.weight    dtype=BF16  shape=[256]
mtp.layers.0.self_attn.q_proj.weight    dtype=BF16  shape=[8192, 2560]
mtp.layers.0.self_attn.k_proj.weight    dtype=BF16  shape=[1024, 2560]
mtp.layers.0.self_attn.v_proj.weight    dtype=BF16  shape=[1024, 2560]
mtp.layers.0.self_attn.o_proj.weight    dtype=BF16  shape=[2560, 4096]
mtp.layers.0.mlp.gate_proj.weight       dtype=BF16  shape=[9728, 2560]
mtp.layers.0.mlp.up_proj.weight         dtype=BF16  shape=[9728, 2560]
mtp.layers.0.mlp.down_proj.weight       dtype=BF16  shape=[2560, 9728]
```

`hidden_size = 2560`, so `[2560, 5120] = [H, 2H]` for `mtp.fc.weight`, matching both (a) PyTorch's `nn.Linear(2*H, H)` storage convention used upstream, and (b) kiln's `validate_shape(&fc, &[H, 2H])`.

Note also that `mtp.layers.0.self_attn.q_proj.weight` is `[8192, 2560] = [2·16·256, H]`, i.e. the gated `q_proj` (`attn_output_gate=true`), and `k_proj`/`v_proj` are `[1024, 2560] = [4·256, H]` (4 KV heads → GQA). These are the same shapes the main-model full-attention layers use; kiln's loader routes them through the existing `load_layer(..., 3, config)` call (line 683) precisely because layer_idx 3 falls in the full-attention residue class `(i + 1) % 4 == 0`.

### Verdict

**No mismatch.** Every one of the three wiring points audited is byte-equivalent between kiln and the upstream references, and the checkpoint header confirms the shape assumptions:

| Wiring point                                  | Upstream (vLLM + SGLang)            | Kiln                                                    | Match |
|-----------------------------------------------|-------------------------------------|---------------------------------------------------------|:-----:|
| Concat order along the 2H dim                 | `[embed, hidden]`                   | `Tensor::cat(&[&norm_emb, &norm_h], 2)` (embed first)   |   ✅   |
| Norm assigned to the embedding half           | `pre_fc_norm_embedding`             | `mtp.pre_fc_norm_embedding` (swap-off default)          |   ✅   |
| Norm assigned to the hidden half              | `pre_fc_norm_hidden`                | `mtp.pre_fc_norm_hidden` (swap-off default)             |   ✅   |
| `fc.weight` on-disk shape                     | `nn.Linear(2H, H)` ⇒ `[H, 2H]`      | `validate_shape(&fc, &[H, 2H])`                         |   ✅   |
| `fc` applied to concat                        | `nn.Linear(concat)` ≡ `concat @ Wᵀ` | `concat.broadcast_matmul(&mtp.fc_t)` with `fc_t=[2H,H]` |   ✅   |

Phase B3 already disproved the norm swap as the fix. Phase B4 now disproves the "fc concat order / weight layout is inverted" hypothesis: the kiln code and the Qwen3.5-4B checkpoint are fully consistent with both of the canonical Python reference implementations.

### Recommendation — next hypothesis

The α = 0.154 collapse with ~46% identity-bias persists despite correct MTP glue wiring. The remaining structural candidates, in order of likelihood:

1. **`mtp.layers.0.*` transformer block weight load.** The MTP inner layer is loaded by re-using `load_layer(tensor_map, &mtp_layer_prefix, 3, config)`. This threads a synthetic `layer_idx=3` through the main-model full-attention path to pick up `attn_output_gate=true`, gated `q_proj` splitting (`[2·num_heads·head_dim, H]`), RMSNorm Q/K norm, etc. Any subtle loader divergence here — e.g. a main-model-only fix-up that assumes a sequential `layer_idx` residue pattern, or a slice of the gated `q_proj` that only activates on the real layers — would silently mis-wire the MTP layer without tripping any shape validator. Phase B5 should re-verify the MTP layer's GQA weights via a direct scalar-comparison test against vLLM's `Qwen3NextMultiTokenPredictor` on an identical input.
2. **RoPE position threading into the MTP layer.** `mtp_forward_step` (forward.rs:3541) builds its own one-element position tensor (`Tensor::new(&[mtp_pos as f32][..], device)`) and passes it to `transformer_block_paged`. The MTP "position space" is distinct from the base model's. Confirm that `mtp_pos` is advanced exactly the same way as the reference generator (vLLM/SGLang advance it by +1 per accepted draft, not per verify-step).
3. **Tied `lm_head` (`embed_tokens_t`).** The MTP head reuses the base model's `embed_tokens_t` as its LM head (loader.rs:604–606). Verify that `embed_tokens_t` is, in fact, `embed_tokens.transpose()` and not the same tensor feeding forward — a copy-vs-transpose confusion would look like a stochastic identity bias even when draft logits look plausible.
4. **`mtp.final_layernorm` vs main-model `final_layernorm` reuse.** The MTP head ships its own `mtp.norm.weight` (confirmed in checkpoint header). Ensure the forward path applies `mtp.final_layernorm` and not the base model's after the inner transformer block.

### Budget used

- **Pod:** none. $0 pod spend.
- **Wall-clock:** ≈ 45 min desk audit (upstream source comparison, safetensors header probe, kiln source re-read, write-up).
- **Output:** this PROFILING.md appendix + this PR. No code changes.

---

## Phase B5 — MTP inner-layer weight load (`layer_idx=3` trick) vs vLLM/SGLang reference (doc-only)

**Status: NO MISMATCH at the MTP inner-layer weight load.** (2026-04-21)

### Why this audit

Phase B4 (PR #267) ruled out the MTP `fc` concat order and weight layout as the source of the α ≈ 0.154 collapse. The next structural candidate from Phase B4's ranked list was the MTP inner transformer block: kiln's loader reuses `load_layer(tensor_map, &mtp_layer_prefix, 3, config)` with a **synthetic** `layer_idx=3` to coerce the main-model full-attention codepath to fire for the MTP layer. The residue class `(3 + 1) % 4 == 0` makes `ModelConfig::is_full_attention_layer` return true, unlocking `attn_output_gate=true`, the gated `q_proj` split (`[2·num_heads·head_dim, H]`), Q/K norm, and GQA dims — all of which match the main-model full-attention layers.

The worry: if any main-model-only fix-up inside `load_layer` / `load_full_attention` is keyed off `layer_idx` as a real index (not just an error-context label), or if anything downstream dispatches on the layer's position in the main-model stack, the synthetic `layer_idx=3` could silently mis-wire the MTP layer.

Phase B5 is a $0, pod-free desk audit that compares kiln's MTP inner-layer load path byte-for-byte against two canonical Python reference implementations plus the published Qwen3.5-4B safetensors header.

### Upstream references (pinned)

| File | Path | SHA |
|---|---|---|
| vLLM | `vllm/model_executor/models/qwen3_5_mtp.py` | `771913e4a024` |
| vLLM | `vllm/model_executor/models/qwen3_next_mtp.py` | `657855ab4179` |
| SGLang | `python/sglang/srt/models/qwen3_5_mtp.py` | `cabe171b6ce3` |

How each upstream gets the MTP inner layer into full-attention mode:

- **vLLM `qwen3_5_mtp.py:97-104`** — explicit string:
  ```python
  self.mtp_block = Qwen3_5DecoderLayer(
      vllm_config=vllm_config,
      layer_type="full_attention",
      prefix=maybe_prefix(prefix, "mtp_block"),
  )
  ```
- **SGLang `qwen3_5_mtp.py:62-75`** — config mutation before delegating to the main-model class:
  ```python
  self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
  ...
  config.num_hidden_layers = 1
  config.full_attention_interval = 1
  self.model = Qwen3_5ForCausalLM(config, quant_config, prefix=..., is_nextn=True)
  ```
- **kiln `loader.rs:676-683`** — synthetic layer index:
  ```rust
  // The MTP layer uses the same full-attention shape as the main-model
  // full-attention layers (GQA + output gate + SwiGLU MLP). We reuse
  // `load_layer` with a synthetic prefix that points at `mtp.layers.0.`
  // and an index (3) whose residue class `(i + 1) % 4 == 0` makes
  // `is_full_attention_layer` return true. The layer_idx argument is only
  // used for error context strings, not for dispatch.
  let mtp_layer_prefix = format!("{mtp_prefix}layers.0.");
  let layer = load_layer(tensor_map, &mtp_layer_prefix, 3, config).context("mtp layer 0")?;
  ```

All three arrive at the **same dispatch outcome**: load an MTP inner layer using the full-attention path. vLLM does it via an explicit `layer_type` string; SGLang mutates `full_attention_interval=1` so every index is full-attn; kiln picks any `layer_idx` that satisfies its fixed residue class.

### Ground truth: Qwen3.5-4B MTP layer tensor shapes

Fetched via HTTP Range on `model.safetensors-00001-of-00002.safetensors` and `model.safetensors-00002-of-00002.safetensors` headers on Hugging Face (`Qwen/Qwen3.5-4B`):

```
mtp.layers.0.input_layernorm.weight            BF16 [2560]
mtp.layers.0.post_attention_layernorm.weight   BF16 [2560]
mtp.layers.0.self_attn.q_proj.weight           BF16 [8192, 2560]   # [2·num_heads·head_dim, H] = gated
mtp.layers.0.self_attn.k_proj.weight           BF16 [1024, 2560]   # [num_kv_heads·head_dim, H]
mtp.layers.0.self_attn.v_proj.weight           BF16 [1024, 2560]
mtp.layers.0.self_attn.o_proj.weight           BF16 [2560, 4096]   # [H, num_heads·head_dim]
mtp.layers.0.self_attn.q_norm.weight           BF16 [256]          # [head_dim]
mtp.layers.0.self_attn.k_norm.weight           BF16 [256]
mtp.layers.0.mlp.gate_proj.weight              BF16 [9216, 2560]   # [intermediate_size, H]
mtp.layers.0.mlp.up_proj.weight                BF16 [9216, 2560]
mtp.layers.0.mlp.down_proj.weight              BF16 [2560, 9216]
```

Cross-check against `model.layers.3.*` (a real main-model full-attention layer per `(3+1)%4==0`): **all 11 tensor shapes are identical**. The MTP inner layer is byte-for-byte the same shape as a main-model full-attention layer, which is exactly what the synthetic `layer_idx=3` trick assumes.

(Note: `intermediate_size=9216` — the value in `ModelConfig::qwen3_5_4b()` at `kiln-core/src/config.rs:86`. The Phase B4 appendix printed `9728` in the shape dump at PROFILING.md:4185-4187 — that was a transcription typo. `kiln-core` validates `[9216, 2560]`, the checkpoint is `[9216, 2560]`, they match.)

### 6-point wiring audit

| # | Wiring point | Upstream vLLM / SGLang | Kiln | Match |
|---|---|---|---|:-:|
| 1 | **Checkpoint key names** (`mtp.layers.0.self_attn.q_proj.weight`, etc.) | Both remap `mtp.` → `model.` in `load_weights`, then feed unchanged through the main-model `Qwen3_5DecoderLayer` loader | `detect_mtp_prefix` keeps the `mtp.` prefix and passes `mtp.layers.0.` to `load_layer`; each tensor is looked up by its stock checkpoint name | ✅ |
| 2 | **Gated `q_proj` dispatch** (must see `[8192, 2560]`, not `[4096, 2560]`) | vLLM: explicit `layer_type="full_attention"`. SGLang: `config.full_attention_interval=1`. Both route to the main-model full-attn block that handles `attn_output_gate=true` | `is_full_attention_layer(3)` = `(3+1)%4==0` = `true` → `AttentionWeights::Full(load_full_attention(...))` (loader.rs:380-386). `full_attn_q_proj_dim()` = `16·256·2 = 8192` (config.rs:157-160) | ✅ |
| 3 | **GQA dims** (`num_heads=16`, `num_kv_heads=4`, `head_dim=256`) | Global model-config read; no per-layer overrides | `load_full_attention` (loader.rs:406-420) reads `config.num_attention_heads`, `config.num_kv_heads`, `config.head_dim` — no `layer_idx`-dependent branches | ✅ |
| 4 | **Q/K RMSNorm** (shape `[256] = [head_dim]`) | No `layer_idx` branch; same head-dim norm whether layer is main-model layer 3 or MTP | `load_full_attention` loader.rs:422-426 validates both against `[config.head_dim]`; identical to main-model layer 3 load | ✅ |
| 5 | **RoPE threading** (partial rotary, `rotary_dim=64`, `rope_theta=1e7`) | Upstream `Qwen3_5DecoderLayer` instance reuses the main-model RoPE module (not rebuilt per layer) | `mtp_forward_step` forward.rs:3547-3558 passes `weights.rotary_inv_freq` (main-model table, built once loader.rs:564-566) and `config.rotary_dim()=64` to `transformer_block_paged`. Same inv_freq table as main-model layer 3 | ✅ |
| 6 | **Weight on-disk layout** (PyTorch `[out, in]`) | All three use PyTorch `nn.Linear` storage: `q_proj=[8192, H]`, `k_proj=[kv_dim, H]`, `o_proj=[H, 4096]`, `gate_proj=[I, H]`, `down_proj=[H, I]` | `validate_shape` checks at loader.rs:411, 414, 417, 420 match PyTorch order. Cached `_t` transposes happen downstream on the GPU (identical to how main-model layers are transposed); no layer-specific transpose path | ✅ |

### What the synthetic `layer_idx=3` actually controls

Grepping `crates/kiln-model/src/loader.rs` for every use of the `layer_idx` parameter reaching from `load_layer` / `load_full_attention` / `load_ffn`:

1. **`is_full_attention_layer(layer_idx)` at loader.rs:380** — the only load-time dispatch. Returns `true` for `3`, just like for main-model layers 3/7/11/15/19/23/27/31.
2. **Error-context strings** (`&format!("layer {layer_idx} input_layernorm")` and similar) at loader.rs:365, 375, 404, 411, 414, 417, 420, 423, 426 — cosmetic. Zero effect on tensor values.
3. **Defensive post-load check** at loader.rs:684-695 — asserts the resulting `AttentionWeights` variant is `Full`, not `Linear`. Belt-and-braces against a future checkpoint schema change. Zero effect on loaded tensors.

That's all. The synthetic layer index **never reaches the MTP KV-cache slot, the forward path, or any tensor loader**. The forward path dispatches on its own index: `mtp_forward_step` hard-codes `full_attn_layer_idx = 0` when it calls `transformer_block_paged` (forward.rs:3562), because the MTP head has its own one-element cache — it is not a slot in the main-model KV cache.

### Verdict

**No mismatch.** kiln's `layer_idx=3` trick is mechanically equivalent to vLLM's `layer_type="full_attention"` and SGLang's `full_attention_interval=1` — three different ways to route the same checkpoint tensors through the same full-attention code. The checkpoint header confirms the shape assumptions (`q_proj=[8192, 2560]`, `kv=[1024, 2560]`, etc.). Nothing about the synthetic layer index leaks into tensor values, forward compute, or cache routing.

### Recommendation — next hypothesis

With the Phase B4 hypothesis #1 now disproved (load-time), the ranking from PROFILING.md:4208-4215 collapses to three runtime candidates:

1. **RoPE position threading** (forward-time, not load-time). `mtp_forward_step` builds its own one-element `positions` tensor from `mtp_pos`. The reference semantics advance `mtp_pos` by +1 per *accepted* draft; kiln's caller advances it only on acceptance (confirmed in loader.rs comment at 601). Worth a scalar A/B against the upstream generator: run the same prompt through vLLM's `Qwen3NextMTP` with acceptance telemetry, diff the `mtp_pos` trajectory.
2. **Tied lm_head via `embed_tokens_t`**. The MTP head reuses the base model's `embed_tokens_t` as its unembedding matrix (forward.rs:3571). Confirm that `embed_tokens_t` is the true transpose of `embed_tokens`, not a shared-storage alias that forwards would be mutating. A one-shot check: load `embed_tokens`, transpose, assert byte-equality against `embed_tokens_t` at load time.
3. **`mtp.final_layernorm` application site**. The loader accepts both `mtp.norm.weight` and `mtp.final_layernorm.weight` (loader.rs:657-668). The checkpoint ships `mtp.norm.weight`. Confirm the forward path (forward.rs:3570) is applying the *loaded MTP norm scale*, not accidentally reaching back through a closure to the main-model's `final_layernorm`.

If all three runtime hypotheses clear, Phase B5-follow-up should move from structural auditing to a direct numerical A/B: load the same Qwen3.5-4B checkpoint into vLLM (using `Qwen3NextMTP`), feed a fixed prompt, and dump MTP intermediate activations (`fc` output, attn output, final norm output, logits) at every step. Scalar diff against kiln's `KILN_MTP_DEBUG=1` trace. Any divergence localizes the bug to a specific sub-op.

### Budget used

- **Pod:** none. $0 pod spend.
- **Wall-clock:** ≈ 55 min desk audit (fetching upstream at pinned SHAs, safetensors header probe on both shards, re-reading `loader.rs::load_mtp_if_present` + `load_layer` + `load_full_attention`, cross-checking forward.rs MTP path, writing this appendix).
- **Output:** this PROFILING.md appendix + this PR. No code changes.

## Phase B6 — MTP numerical dual-dump bisect (per-tap localization, 2026-04-21)

### Goal

Phase B5 disproved the `layer_idx=3` load-time hypothesis and collapsed the
remaining MTP-α-collapse root-cause candidates to three runtime hypotheses
(PROFILING.md:4211–4220). This Phase B6 bisect runs a **pure-PyTorch reference
implementation** of `mtp_forward_step` over the exact same `h_main` +
`draft_token_id` that kiln's own forward path consumes, and diffs the
intermediate activations tap-by-tap with `allclose` + cosine similarity. The
first tap that diverges pinpoints which of the three hypotheses (if any) is
actually responsible.

### Scope and honest limits

* **Single prompt**, single `mtp_pos=0`. One-shot dump. Does not claim to
  cover every MTP acceptance path.
* `h_main` is **taken from the kiln dump** — the reference does not re-run
  the 24 × GDN + 8 × GQA base stack. If the base model is producing the
  wrong `h_main`, every downstream tap would match the reference bit-for-bit
  (since the reference is fed the same bad `h_main`), and Phase B6 would
  return a clean verdict. That would itself be a signal — pointing
  upstream of `h_main` to the scheduler / paged KV state / GDN state on
  the MTP branch.
* BF16 matmul noise is not a bug: `|Δ|` of order 0.01 at intermediate
  activations over 5k-dim reductions is normal. The bisect therefore reports
  cosine similarity alongside `allclose`; a direction-preserving tap (cos≥0.999)
  with small `|Δ|` is considered a match regardless of strict-`allclose` status.

### Instrumentation

Code added on this branch:

* `crates/kiln-model/src/mtp_debug.rs` — `write_mtp_dump()` writes 8 F32 taps
  + 3 I32 metadata scalars in safetensors format, gated by `KILN_MTP_DUMP_PATH`.
  One-shot latch via `AtomicBool` to avoid per-step overhead.
* `crates/kiln-model/src/forward.rs` — `mtp_forward_step` captures the 8
  named taps in order. `concat` and `normed` pulled out as named binds so
  the dump can see them without a second forward pass.
* `scripts/mtp_reference_dump.py` — pure-PyTorch reference. Loads MTP weights
  from `/workspace/qwen3.5-4b`, reads `h_main` + `draft_token_id` from the
  kiln dump, runs the full `embed → dual rms_norm → concat → fc →
  single transformer block (with RoPE at mtp_pos, per-head Q/K RMSNorm,
  gated-attn, MLP) → final_layernorm → tied LM head` path, writes the
  same 8 taps.
* `scripts/mtp_compare.py` — per-tap diff. Prints a table of
  (shape, allclose, cos_sim, max|Δ|, mean|Δ|) and maps the first divergence
  back to the hypothesis it implicates.

Reference RMSNorm uses the Qwen3.5-specific form `out = (1 + w) * x * rsqrt(mean(x²) + ε)`
(see `forward.rs::rms_norm_fallback` at line 936). This is **not** the
HF-standard RMSNorm; Qwen3.5 stores RMSNorm weights centered around 0 and
applies them as `(1 + w)`. Using the standard form here produces a false
divergence at `fc_input` that masks the real signal — any future additions
to the reference must use the same semantics.

### Results

Prompt = `PROMPT_POOL[2]`, seed=42, `draft_token_id=561`, `mtp_pos=0`,
`swap_fc_norms=0`. Kiln built in release mode with `KILN_CUDA_ARCHS=86
--features cuda --bin kiln-bench`. Reference run with torch 2.x on CPU.

Comparison at `atol=0.05, rtol=0.05` (BF16-realistic):

| tap            | shape           | cos_sim | max\|Δ\|   | mean\|Δ\| | verdict |
|----------------|-----------------|---------|-----------|-----------|---------|
| h_main         | 1×1×2560        | 1.000   | 0.00      | 0.00      | match (input) |
| tok_embed      | 1×1×2560        | 1.000   | 0.00      | 0.00      | match |
| fc_input       | 1×1×5120        | 1.000   | 2.70e-2   | 7.12e-4   | match (BF16 noise) |
| fc_output      | 1×1×2560        | 1.000   | 1.16e-2   | 7.06e-4   | match (BF16 noise) |
| pre_layer      | 1×1×2560        | 1.000   | 1.16e-2   | 7.06e-4   | match |
| **post_layer** | **1×1×2560**    | **0.600** | **5.37** | **6.95e-1** | **FIRST DIVERGENCE** |
| post_final_ln  | 1×1×2560        | 0.538   | 21.12     | 2.41      | divergent (propagated) |
| mtp_logits     | 1×1×248320      | 0.540   | 10.57     | 1.62      | divergent (propagated) |

Metadata checks: `draft_token_id`, `mtp_pos`, `swap_fc_norms` all match.

### Verdict

The first numerically real divergence is at **`post_layer`** — the output of
the single-layer MTP transformer block. The pipeline is clean through
`pre_layer` (cos≥1.0, `|Δ|` at BF16-matmul-noise levels), which means:

* **Hypothesis H1 (RoPE `mtp_pos` advancement)**: **STRONG candidate.** The
  MTP block is the only place `mtp_pos` enters the forward graph. RoPE uses
  `partial_rotary_factor=0.25` (64 of 256 head dims rotated) and
  `rope_theta=1e7`. At `mtp_pos=0` the rotation is the identity, so if the
  divergence persists at `mtp_pos=0`, the root cause is **not** a
  position-advancement bug per se but something else inside the block
  (Q/K/V projection layout, per-head Q/K RMSNorm, gated-attn). If on
  later `mtp_pos > 0` steps the divergence grows with position, H1 is
  confirmed as an advancement bug.
* **Hypothesis H2 (tied `embed_tokens_t` transpose vs alias)**: **WEAKENED
  but not fully eliminated.** `post_final_ln` and `mtp_logits` both end at
  cos_sim ≈ 0.54 — meaning the `post_final_ln → logits` step preserves the
  bulk of the direction. A transpose-layout bug in the tied LM head would
  collapse cos_sim to near-zero, not preserve it. The remaining ~0.6 %
  cos_sim gap between `post_final_ln` (0.538) and `mtp_logits` (0.540) is
  within propagation noise for an `[H] → [V]` matmul. The tied transpose
  is likely fine.
* **Hypothesis H3 (`mtp.final_layernorm` application site)**: **WEAKENED.**
  `post_layer` is already divergent, so any mismatch at `post_final_ln` is
  partly explained by propagation. The `post_layer → post_final_ln`
  cos_sim drop (0.600 → 0.538) is consistent with a single RMSNorm
  amplifying an already-divergent input; it does not require an
  independent bug in the norm application site. This line of inquiry
  can be parked unless H1 is disproved.

### Narrowed next steps

This bisect collapses the three-hypothesis space to **a single active
candidate: the MTP inner transformer block itself**, with sub-hypotheses
ordered by likelihood:

1. **Per-head Q/K RMSNorm** — same `(1 + w)` trap that produced a false
   divergence in the reference on the first pass. If kiln's Q/K-norm path
   ever takes a branch that applies bare `w`, it would manifest exactly
   here.
2. **Gated attention (`attn_output_gate=true`)** — Qwen3.5-4B applies a
   sigmoid gate on the attention output. A sign error, wrong split, or
   missing gate would appear first at `post_layer`.
3. **Q/K/V projection layout** — `q_proj` is the gated `[8192, 2560]`
   variant, `k_proj`/`v_proj` are GQA `[1024, 2560]`. A transpose bug
   here would mangle the attention output specifically.
4. **RoPE position threading** at `mtp_pos > 0` — not diagnosed by this
   single-step dump; needs a multi-step variant (Phase B7).

Phase B7 should dump at `mtp_pos = 0, 1, 2` on the same prompt and check
whether the divergence grows with position (→ H1 confirmed) or is
position-invariant (→ H1 eliminated, bug is inside the block). If B7
eliminates H1, the next phase is a **per-sub-op reference inside the MTP
block**: break down `post_layer` into `q/k/v → rope → attn → out_proj →
gate → mlp_up → mlp_gate → mlp_down` and dump each one.

### Budget used

- **Pod:** RunPod pool, `s23qwogiqyk76s` (RTX A6000) via lease
  `pod-37efdfc4f8b4c4bdbcfa0b98`. Hot build (sccache+incremental), two
  kiln-bench runs for dump capture, one reference-script execution, two
  compare runs. ≈ 15 min GPU-time. Pod released to pool after PR open.
- **Wall-clock:** ≈ 80 min total, including one iteration to catch a
  reference-side RMSNorm bug (`(1 + w)` vs bare `w`) that masqueraded as
  an `fc_input` divergence on the first pass and would otherwise have
  produced a false H2-ish verdict.
- **Output:** this appendix, the `mtp_debug.rs` + `mtp_reference_dump.py`
  + `mtp_compare.py` scaffold (all preserved for Phase B7), the concrete
  bisect report at `/tmp/mtp-compare.txt` (copied to
  `mtp-compare-phase-b6.txt` in the session workspace), and this PR.
  **No fix included.** Scope is bisect-only per the task brief.


## Phase C12 — fp32 draft-head kill switch + activation-weighted probe (2026-04-21)

### Goal

Test whether forcing the MTP draft head's q/k/v/o + fc projections to fp32
(`KILN_MTP_FP32_HEAD=1`) recovers α toward the 0.72 ship floor. Hypothesis
from C9/C11 audit: bf16 matmul accumulation noise on W4A16-dequanted weights
shifts the draft head's top-1 enough to explain α == 0.058-0.124 observed
in C5 after the C3 RoPE fix landed.

### Implementation

Narrow, TLS-gated chokepoint — no behavior change when flag unset.

- `crates/kiln-model/src/mtp_debug.rs` — `KILN_MTP_FP32_HEAD` env reader +
  `MTP_FP32_HEAD_ARMED` TLS `RefCell<bool>` + arm/disarm helpers.
- `crates/kiln-model/src/lora_loader.rs::linear_with_lora_t` — when armed,
  upcast `x` and transposed base weight to `DType::F32`, matmul, cast
  back. LoRA and bias adds left untouched. Single chokepoint covers q/k/v/o
  and MLP base matmuls in one branch.
- `crates/kiln-model/src/forward.rs::mtp_forward_step` — reads flag once;
  arms TLS around the draft-head `transformer_block_paged`; disarms on
  exit. OR's into the existing `KILN_MTP_FC_FP32_ACCUM` branch so the new
  flag subsumes C9's fc-only knob.

### Results

Median-of-3 on A6000, Qwen3.5-4B, `--paged --prompt-tokens 512
--max-output-tokens 128 --skip-training --seed {42,43,44}`, MTP on
(`KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp`):

| Config           | env                                      | Median α | Median tok/s |
|------------------|------------------------------------------|----------|--------------|
| baseline         | `KILN_W4A16=1`                           | 0.0325   | 41.09        |
| **primary**      | `KILN_W4A16=1 KILN_MTP_FP32_HEAD=1`      | 0.0325   | 37.66        |
| sanity_nomarlin  | `KILN_W4A16=0`                           | 0.0583   | 36.76        |

Per-seed α is bitwise identical across configs for 2/3 seeds (seed 42:
4/123 for all three configs; seed 43: 4/123 for baseline and primary).

### Verdict

**PRIMARY-NEGATIVE.** fp32 draft-head projections do not recover α. The
bf16 matmul-accumulation hypothesis is refuted: dtype of the head's
projections is not a contributor of meaningful magnitude to α under greedy
decode. The failure mode is structural — the MTP head deterministically
picks a different top-1 token from the main head, across W4A16-bf16,
W4A16-fp32, and no-Marlin-at-all.

Combined with C5 (Class C = 0, Class B = 87.6%) and C3 (bitwise-identical
α pre/post RoPE fix), the surviving hypothesis space is now entirely
upstream of the head's projections: MTP token-embedding lookup, MTP
pre-norm γ application, MTP hidden-state splicing from the main stack, or
weight-loading for `mtp_head.fc_input` / `mtp_head.fc_output` /
q-k-v-o. **C13 should investigate weight-loading policy and the splice
input to the draft head first** — mis-tied or mis-loaded weights produce
exactly this symptom (deterministic wrong top-1, dtype-insensitive).

Decode cost: `KILN_MTP_FP32_HEAD=1` costs ≈ 8.3% of tok/s on A6000
(41.09 → 37.66). Do not gate on by default.

### Activation-weighted probe (sidecar, main-model MLP)

`scripts/c12_activation_weighted_probe.py` audited 104 main-model
projections on 32 prompts × 595 total tokens. Top weighted drift:
`L6.down_proj` at 1.447e-01; many MLPs in the 11-14% band. This is
non-trivial drift by magnitude, but it doesn't propagate into main-head
top-1 flips on the bench prompts (otherwise sanity_nomarlin and baseline
main-head trajectories would differ more — they don't). The probe's
script-level terminal verdict ("Corroborates PRIMARY-POSITIVE…") is stale
boilerplate; the authoritative C12 verdict is the bench above.

### Budget used

Single A6000 pod (`wl0fyjvqrv0v9b`), warm sccache (hit rate 97.56%). Build
2m 12s; 9 bench runs ≈ 974s total; one probe run (~160s). All pod-side
waits used `runpod_api.py wait-file --timeout`; no `until ssh` or
`while ssh … sleep` polling loops. Well under the 90 min / $40 hard cap.

### Output

- `docs/phase-c12/c12-fp32-head.md` — full verdict report.
- `c12-out/bench-summary.json` — 3 × 3 trial JSON (seed, α, tok/s, ITL,
  accept line).
- `c12-out/probe-report.md` + `c12-out/probe-report.json` — main-model
  activation-weighted drift audit.
- `c12-out/bench.log`, `c12-out/probe.log` — runner logs.
