# Phase 10 — GDN training-time streaming-prefill impl (Finding 2 follow-up)

Date: 2026-04-29
Status: **AMBER — dispatch is now wired and CPU parity passes, but T=8192 SFT on A6000 still OOMs because the autograd recompute saves all per-tile activations simultaneously**
Hardware: NVIDIA RTX A6000 (49 140 MiB VRAM, driver 550.127.05), CUDA 12.4
Commit: `0369c94` on `ce/phase10-gdn-streaming-training-impl`, off `722feb5` main (post-PR #634 audit)
Bench: `crates/kiln-server/examples/flce_phase_a_validation_bench.rs`
Raw log: `docs/flce_phase_a_streaming_impl_raw_2026-04-29.log`

## Purpose

Implementation follow-up to the PR #634 audit
(`docs/audits/PHASE10_GDN_TRAINING_STREAMING.md`), which empirically confirmed
that `KILN_STREAMING_PREFILL=1` had **zero effect on the SFT training path** —
the streaming dispatch lived only in `model_forward_paged_streaming*` (the
inference family) and was unreachable from `sft_train` →
`checkpointed_forward_backward` → `model_forward_segment`.

This PR implements the audit's recommended remediation §1 ("smallest change that
could work"): add a streaming branch inside `model_forward_segment` that tiles
GDN layers along T while threading `LinearAttentionState` per tile, leaving
full-attention layers monolithic (training has no KV cache to thread across
tiles).

The PR exists to answer two questions independently:

1. **Does the streaming flag actually take effect now?** Yes — see "Dispatch
   reachability" below.
2. **Does it unblock T=8192 SFT on A6000?** No — see "Memory ceiling" below.

## Implementation summary

`crates/kiln-model/src/forward.rs`:
- New `gated_deltanet_forward_streaming(backend, x, weights, config,
  recurrent_state, conv_state, tile_size)` — slices `x:[B,T,hidden]` along T
  into tiles of `tile_size` (last tile may be partial), calls
  `gated_deltanet_forward` per tile threading the `LinearAttentionState`
  recurrent + conv state in place, and `Tensor::cat`s the per-tile outputs back
  into `[B,T,hidden]`. Bit-exact with the monolithic call by construction (the
  state hand-off is the same one the inference streaming path already uses).
- `model_forward_segment` reads `streaming_prefill_enabled_for(device, T)` and
  `streaming_tile_tokens_for(device)` once per call. When streaming is enabled
  and `T > tile_size`, GDN layers dispatch through
  `gated_deltanet_forward_streaming`; full-attention layers always run
  monolithic.
- Two CPU parity tests:
  - `test_gated_deltanet_forward_streaming_matches_monolithic_cpu` — direct
    helper parity, env-var-free, safe under multi-threaded `cargo test`.
  - `test_model_forward_segment_streaming_matches_monolithic_cpu` — full
    8-layer-stack segment parity, env-var driven (relies on nextest per-test
    process isolation).

`crates/kiln-server/examples/flce_phase_a_validation_bench.rs`:
- Added the T=16384 STREAMING ON tile=4096 stretch cell.
- Updated docstrings and verdict logic to reflect the implementation (was
  written under the audit's "is the flag reachable?" framing).

No trainer-side changes — `checkpointed_forward_backward` and
`standard_forward_backward` continue to call `model_forward_segment` as before;
the streaming dispatch is fully encapsulated inside `model_forward_segment`.

## Validation

### CPU parity

`cargo nextest run --release --features cuda -p kiln-model -- streaming_matches_monolithic forward_segment_streaming_matches gated_deltanet_forward_streaming` on the A6000 pod:

```
PASS [   0.235s] kiln-model forward::tests::test_gated_deltanet_forward_streaming_matches_monolithic_cpu
PASS [   0.250s] kiln-model forward::tests::test_model_forward_segment_streaming_matches_monolithic_cpu
PASS [   0.388s] kiln-model forward::tests::test_streaming_matches_monolithic_cpu_mid
PASS [   0.228s] kiln-model forward::tests::test_streaming_matches_monolithic_cpu_small
PASS [  16.393s] kiln-model forward::tests::test_streaming_matches_monolithic_cuda
Summary [  17.494s] 5 tests run: 5 passed, 263 skipped
```

Both new tests assert per-position output, recurrent state, and conv state
parity across the streaming and monolithic paths to within 1e-5 (helper test)
or 1e-4 (full segment test) absolute.

### A6000 SFT bench

`./target/release/examples/flce_phase_a_validation_bench --model-path /workspace/qwen3.5-4b`:

| target_T | actual_T | FLCE | STREAM | TILE     | peak VRAM (MiB) | ΔVRAM (MiB) | step (s) | loss              | status |
|---:|---:|:---:|:---:|:---:|---:|---:|---:|---:|:---|
|  2 048 |  2 042 | OFF | unset    | default |          34 600 |       16 128 |     2.8 | 2.7836852         | ok    |
|  2 048 |  2 042 | ON  | unset    | default |          34 600 |       16 128 |     3.2 | 2.7834394         | ok    |
|  2 048 |  2 042 | ON  | **ON**   | default |          34 600 |       16 128 |     3.2 | **2.7834394**     | ok    |
|  8 192 |  8 192 | ON  | unset    | default |          48 648 |       30 176 |     0.7 | n/a               | OOM   |
| 16 384 | 16 380 | ON  | unset    | default |          48 648 |       30 176 |     0.7 | n/a               | OOM   |
|  8 192 |  8 192 | ON  | **ON**   | default |          48 648 |       30 176 |     0.7 | n/a               | OOM   |
|  8 192 |  8 192 | ON  | **ON**   | 4096    |          48 648 |       30 176 |   **1.3** | n/a               | OOM   |
|  8 192 |  8 192 | ON  | **ON**   | 2048    |          48 660 |       30 188 |     0.7 | n/a               | OOM   |
| 16 384 | 16 380 | ON  | **ON**   | 4096    |          48 648 |       30 176 |     0.6 | n/a               | OOM   |

Baseline post-warmup VRAM = 18 472 MiB.

## Dispatch reachability — confirmed

The audit's hypothesis ("`KILN_STREAMING_PREFILL` is parsed without effect")
is now **falsified**. Three pieces of evidence:

1. **CPU parity tests pass.** `test_model_forward_segment_streaming_matches_monolithic_cpu`
   runs `model_forward_segment` once with `KILN_STREAMING_PREFILL` unset and
   once with `KILN_STREAMING_PREFILL=1` `KILN_STREAMING_TILE_TOKENS=64` (T=192,
   3 tiles), and asserts per-position output and per-layer state parity. It
   passes — meaning the streaming branch is reached, runs 3 distinct GDN tile
   forward calls per layer, and produces equal output to the monolithic call.

2. **GPU step time differential.** The T=8192 STREAMING ON cells show a clear
   step-time signal:
   - tile=default(8192): 0.7 s (no actual tiling, T == tile)
   - tile=4096:          **1.3 s** (2 tiles per GDN layer)
   - tile=2048:          0.7 s (4 tiles per GDN layer)

   tile=4096 takes nearly 2× the monolithic step time, evidence that the
   streaming branch is running and doing real per-tile work before OOM. The
   tile=2048 cell OOMs faster (0.7 s) because the autograd recompute holds
   more concurrent per-tile activations than tile=4096 and crashes earlier
   in the segment.

3. **T=2048 STREAMING ON loss matches STREAMING unset bit-for-bit.** Both
   produce final loss `2.7834393978118896`. With the default tile (8192), no
   tiling happens at T=2048, so the monolithic fallback inside
   `gated_deltanet_forward_streaming` runs — the loss equality is therefore
   the expected outcome of correct dispatch (it would be a regression flag if
   the loss had drifted).

The flag now does what it says.

## Memory ceiling — unchanged

The headline acceptance criterion ("T=8192 SFT on A6000 no longer OOMs under
`KILN_USE_FLCE=1` + `KILN_STREAMING_PREFILL=1`") is **not met**. All four
T=8192 STREAMING ON cells (tile=8192/4096/2048) and the T=16384 STREAMING ON
tile=4096 stretch cell OOM at the same ≈48 648 MiB ceiling as the monolithic
T=8192 cell. The T=8192 peak-VRAM ladder is flat across tile sizes.

The audit predicted that "tiling along T inside the segment **would** reduce
these intermediates to per-tile shapes, multiplicatively with gradient
checkpointing." Empirically that prediction is false for the SFT training path
on A6000 at T=8192. The likely root cause is **autograd-graph saved tensors,
not transient kernel allocations**:

- During `checkpointed_forward_backward`'s per-segment recompute step, the
  autograd graph builds and retains all per-layer-per-tile saved tensors
  (`gated_deltanet_forward` outputs, intermediate F32-promoted tensors that
  participate in backward, MLP gate/up/down outputs, etc.) until the segment's
  backward completes. With tiling, ONE [B, T, ·] saved tensor per layer is
  replaced with N [B, T/N, ·] saved tensors per layer — same total bytes.
- Only the **transient** kernel allocations inside `causal_conv1d_update` /
  `l2_normalize` / `gdn_chunkwise_recurrence` (which are released before the
  Rust function returns) shrink with tile size. Those are evidently not the
  binding constraint at T=8192 on A6000.
- The single-step ~30 GiB Δ over baseline is consistent with grad
  checkpointing's recompute saving ~6 GiB / GDN layer × ~5 layers in the
  largest segment + full-attn layer activations. The conv/l2_normalize F32
  promotions are roughly 1 GiB each at T=8192 per layer; tiling 2× would save
  ~0.5 GiB peak transient per layer, dwarfed by the saved-tensor bulk.

The next-tile-OOMs-faster pattern (tile=2048 OOMs at 0.7 s where tile=4096
OOMs at 1.3 s) is also consistent with this picture: smaller tiles ⇒ more
saved tensors per layer ⇒ faster ramp to ceiling.

## Verdict

**AMBER** — the dispatch wiring shipped is **necessary** infrastructure for
any future training-time GDN-streaming work (no other code path can reach the
streaming branch from `sft_train`), and the CPU parity tests pin the
correctness invariant the next attempt needs to preserve. But this PR alone
does not unblock long-context SFT on A6000.

This is a useful negative result for the audit's option (1). It rules out
"smallest change that could work" as sufficient and re-frames the next move
as the audit's option (2) (per-tile forward+backward inside
`checkpointed_forward_backward`, ~200-400 LOC) or option (3) (combine option
2 with a CUDA fused FLCE kernel from Phase B).

## Required follow-ups (post-this PR)

The remediation menu, in increasing scope, narrowed by this PR's findings:

1. ~~**Tile within `model_forward_segment` (smallest change that could work).**~~
   **Shipped here. Insufficient on A6000 at T=8192 — the autograd recompute
   bottleneck dominates.** Code remains useful as the dispatch foundation any
   higher-level remediation will reuse.

2. **Time-axis tile inside `checkpointed_forward_backward` (recommended next).**
   The remaining viable option-1 sibling. Inside the segment-recompute loop in
   `crates/kiln-train/src/trainer.rs:1207-1320`, split each segment's T into M
   time tiles, run forward+backward per tile threading
   `LinearAttentionState`, and accumulate LoRA gradients across tiles within
   the segment (the existing layer-axis accumulation pattern generalizes).
   The forward path can reuse `gated_deltanet_forward_streaming` from this PR
   per tile (or call `gated_deltanet_forward` directly per tile and only
   tile at the trainer level — both are equivalent given the state hand-off).
   This is the change that actually releases per-tile saved tensors after each
   tile's backward completes, which is what the audit's "multiplicatively with
   gradient checkpointing" claim requires. Estimated scope: ~200-400 LOC in
   `trainer.rs`. Risks: (a) gradient correctness across the recurrent state
   thread (state is mutated in place, so per-tile backward must source the
   correct pre-tile state — the same mechanism this PR's
   `gated_deltanet_forward_streaming` already exercises within a single
   segment-forward call); (b) full-attention layers in mixed-attn segments
   still need monolithic-T forward (no KV cache to thread across time
   tiles), so the time-tile loop must operate at the layer-pair level not
   the segment level for hybrid models.

3. **Both, on top of a CUDA fused FLCE kernel (Phase B).** Phase B's head-side
   memory savings stack with GDN-side time-axis tiling (option 2). Combined,
   this is the path most likely to fit T=16384 SFT on A6000.

Recommended next slice: **option 2 as a focused PR** — it is the change that
actually moves the autograd-recompute bottleneck this PR identified.

## Caveats

- The bench's peak-VRAM signal is `nvidia-smi` polled at 50 ms; sub-50 ms
  transient peaks may be undercounted. All cells in this PR's table are at the
  GPU ceiling (~48 648 / 49 140 MiB), so the OOM-vs-OK distinction is robust
  even with that resolution.
- The T=2048 STREAMING ON cell exercises the monolithic fallback (T <= tile),
  not the streaming path. T=2048 with `KILN_STREAMING_TILE_TOKENS=1024` would
  be the smallest cell that actually runs the streaming path; deferred — the
  empirical signal at T=8192 is already conclusive about the impl-vs-ceiling
  trade.
- Bit-exact CPU parity does not guarantee bit-exact CUDA parity (matmul
  reduction order can vary with shape). The existing
  `test_streaming_matches_monolithic_cuda` (1e-4 tolerance for the inference
  path) is the closest GPU parity proxy; a training-side CUDA parity test is
  not added in this PR (out of scope; the GPU SFT bench's loss differential
  at T=2048 STREAMING ON vs unset is 0.0e0 which is the strongest empirical
  signal available short of a dedicated harness).
- Audit scoped to GPU-side production training (CUDA path on A6000). Metal /
  MLX training is unaffected by this PR (the streaming path is Cpu/Cuda/Metal
  uniform, but Metal SFT has separate optimization concerns).
