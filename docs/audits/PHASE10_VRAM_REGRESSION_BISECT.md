# Phase 10 §1 — VRAM regression bisect (post-#637 → main)

Date: 2026-04-29
Status: **Bisect complete — PR #638 owns the full regression. PR #639 is inert.**
Hardware: NVIDIA A40 (46 068 MiB VRAM, driver 570.195.03), CUDA 12.4
Pod: RunPod direct-launch fallback (A6000 hosts at "not enough free GPUs"; A40 has the same Ampere `sm_86` arch and the regression is large enough to be observable on the smaller ceiling)
Bench: `crates/kiln-server/examples/phase10_rmsnorm_bench.rs` from main (`cc1fb7e`), modified to run only the T=2048 / `custom_op=true` cell so the same script can be applied at each commit (see "Bench harness" below)
Model: Qwen3.5-4B (V=151 936, hidden=2560, 32 layers, 24 GDN + 8 GQA), bf16, no W4A16
Companion to: [`PHASE10_RMSNORM_E2E.md`](PHASE10_RMSNORM_E2E.md), which documented the regression but did not bisect it.

## TL;DR

The **+14–16 GiB delta-VRAM regression at T=2048 SFT** between the PR #637 baseline (run on the morning of 2026-04-29) and current main (run that afternoon, post-#638 + #639) is **fully introduced by PR #638**. PR #639 changes only test code (`tiny_weights` RNG seeding + an `ENV_LOCK` mutex) and adds **zero** runtime regression on top of #638 — the bench numbers are bit-equal between #638 and #639.

| Commit | PR | Peak VRAM (MiB) | Δ VRAM (MiB) | Step (s) | Final loss | Status |
|--------|----|----------------:|-------------:|---------:|-----------:|--------|
| `c983ca7` | post-#637 baseline | **23 367** | **6 761** | 15.36 | 2.7834 | ok |
| `1edecd7` | #638 (RMSNorm CustomOp) | **42 023** | **25 680** | 10.32 | 2.7834 | ok |
| `1ad54c2` | #639 (NaN parity test fix) | **42 023** | **25 680** | 10.33 | 2.7834 | ok |

`peak(#638) − peak(#637) = +18 656 MiB` (≈ +18.2 GiB). On A6000's 49 140 MiB ceiling that delta is what pushes T=2048 from "fits with 26 GiB headroom" into "OOM at the rail" reported in [`PHASE10_RMSNORM_E2E.md`](PHASE10_RMSNORM_E2E.md). The numerical regression sign is the same on A40; the absolute peak differs because the trainer auto-detects 48 GiB and configures 8 segments × 4 layers (vs the audit doc's A6000 4 × 8 segmentation), which lowers the absolute peak floor while preserving the per-PR delta.

Final loss is bit-exact (2.78343939…) at all three commits. The math is unchanged. **The cost is purely peak VRAM.**

## Procedure

1. Acquired RunPod A40 (capacity fallback path; both pool A6000+A40 hosts and direct-launch A6000 returned `SUPPLY_CONSTRAINT` at acquire time).
2. `kiln-setup --clone` (sccache + B2 + clone + Qwen3.5-4B weights at `/workspace/qwen3.5-4b`).
3. For each `(SHA, label)` in:
   * `(c983ca7, c983ca7)` — post-#637, baseline
   * `(1edecd7, 1edecd7)` — PR #638, RMSNorm CustomOp
   * `(1ad54c2, 1ad54c2)` — PR #639, NaN parity test fix
4. `git checkout <SHA>` (detached HEAD), `cp /tmp/phase10_rmsnorm_bench_t2048only.rs crates/kiln-server/examples/phase10_rmsnorm_bench.rs`, `KILN_CUDA_ARCHS=86 cargo build --release --features cuda -p kiln-server --example phase10_rmsnorm_bench` (incremental; sccache cache hit ≥ 98.8% on the cold first build).
5. `KILN_DISABLE_RMSNORM_KERNEL=1 ./target/release/examples/phase10_rmsnorm_bench --model-path /workspace/qwen3.5-4b`. The kill switch forces the candle-op-chain `rms_norm_fallback` at all three commits, so the dispatch wrapper change in #638 is **not** in the call path. Same SFT path runs in all three runs.

## Bench harness

The bench file used at each commit is identical: the main-branch `phase10_rmsnorm_bench.rs` (which only exists from `1edecd7` onwards — at `c983ca7` it's a new file copied in for the bisect) with cells 2–6 of the original 6-cell sequence stripped so only Cell 1 (T=2048, `custom_op=true`) runs. This:

* Holds the bench harness identical across the three commits (same example builder, same VRAM poller, same SftConfig, same FLCE setting).
* Avoids spending time on the T=4096/8192 OOM cells that aren't necessary for bisecting a regression already visible at T=2048.
* Uses `custom_op=true` (the bench's "ON" cell) — this means the inner `run_one()` does **not** set `KILN_DISABLE_RMSNORM_BACKWARD`, so only the outer `KILN_DISABLE_RMSNORM_KERNEL=1` kill switch is engaged, forcing the candle-op-chain fallback path on all three commits.

The `phase10_rmsnorm_bench` API surface (`SftConfig`, `SftExample`, `ChatMessage`, `sft_train`, `ProgressCallback`, `TrainingProgress`) is byte-equal between `c983ca7` and `1edecd7`, so the same source compiles unmodified at `c983ca7` against the `kiln-train` API there.

The bench panics with `index out of bounds: the len is 1 but the index is 1` after printing the T=2048 row (the original bench's per-T parity calculator references both ON and OFF cells, which the bisect harness skips). This panic is harmless — the row prints first and is the only datum needed.

## Mechanism — what changed in #638 vs #637

PR #638 modifies eight files. Comparing the diff against the production call path with `KILN_DISABLE_RMSNORM_KERNEL=1` set:

| File | Lines | In hot path? |
|------|------:|--------------|
| `crates/kiln-train/src/trainer.rs` | +87 | **No.** Diff is entirely under `mod tests {}` (line 3365+). `git diff c983ca7..1edecd7 -- trainer.rs` above line 3360 produces zero output — the production training loop is byte-equal. |
| `crates/kiln-model/src/forward.rs` | ±35 | **No** under the kill switch. Only `rms_norm()` dispatch wrapper changed (added a second env-var read and switched the `if`-branch target from `fused_rmsnorm` to `fused_rmsnorm_with_autograd`). With `kernel_disabled = true`, both pre- and post-#638 versions short-circuit to `rms_norm_fallback` at the same line. `rms_norm_fallback` itself is byte-equal between commits (verified via `diff` of the function bodies). |
| `crates/kiln-rmsnorm-kernel/src/lib.rs` | +884 | **No.** New code is `RmsNormCustomOp` impl, `fused_rmsnorm_backward`, `fused_rmsnorm_with_autograd`, `supports_autograd`, and accompanying CPU/CUDA tests. None is reachable when `kernel_disabled = true`. The pre-existing `fused_rmsnorm()` body is byte-equal between commits, and `supports()` is byte-equal between commits. |
| `crates/kiln-rmsnorm-kernel/csrc/fused_rmsnorm_bwd.{cu,h}` | +289 | **No.** New CUDA kernels, no static `cudaMalloc`, no `__device__` globals, no `extern "C"` ctors. They're purely runtime-callable kernel symbols. |
| `crates/kiln-rmsnorm-kernel/build.rs` | +1 | Build-only; compiles the new `.cu` file into the static library. |
| `crates/kiln-rmsnorm-kernel/examples/phase10_microbench.rs` | +109 | New example, not in this build. |
| `crates/kiln-server/examples/phase10_rmsnorm_bench.rs` | +456 | This is the bench file. Replaced by the bisect harness at all three commits. |
| `Cargo.lock`, `Cargo.toml`, `crates/*/Cargo.toml` | 0 | `git diff c983ca7..1edecd7 -- Cargo.lock Cargo.toml crates/*/Cargo.toml` produces zero output — no dependency drift. |

**Every code path that runs in the bisect bench is byte-equal between c983ca7 and 1edecd7.** Yet peak VRAM grows by +18.2 GiB and step time *falls* by 33% (15.36s → 10.32s). The new symbols added to the binary by #638 are reachable only behind a kill switch that is engaged for every cell.

## Hypotheses (in order of plausibility)

1. **Candle CUDA allocator caching changed by binary-layout difference** — the +884-line `kiln-rmsnorm-kernel/src/lib.rs` and the new `.cu`-compiled symbols expand the binary's `.text` and `.rodata` sections enough that one of candle's internal `OnceCell` / `lazy_static` site-of-first-use ordering changes. If a candle CUDA allocator pool is initialized earlier or with different sizing on the new binary, `peak − baseline` from `nvidia-smi` can shift dramatically without any user-visible code change. This is the **only** hypothesis consistent with "every executed line is byte-equal but VRAM grows by 18 GiB." It also predicts the observed step-time *speedup* — a fatter pool that grabs memory earlier reduces alloc-on-demand stalls during the SFT step.

2. **Autograd graph structure altered by an inlining/codegen difference** — the new `RmsNormCustomOp` implementations and trait impls in `kiln-rmsnorm-kernel/src/lib.rs` may shift which `BackpropOp` variants candle's autograd derives. Even if the call path uses `rms_norm_fallback`, the *same* function at `c983ca7` and `1edecd7` may have its candle-op chain emit a different `BackpropOp` graph because the linker has more `Tensor`-typed methods to consider. This would change saved-tensor count without changing computed loss. Plausibility: lower than (1) — Rust generic monomorphization shouldn't change autograd behavior across binary changes in unrelated modules — but not zero.

3. **A subtle correctness bug in #638 dispatch** that I'm not seeing — e.g. `KILN_DISABLE_RMSNORM_KERNEL=1` is parsed differently, or the new branch is taken even when it shouldn't be. This is contradicted by the bit-equal final loss (math is identical) but worth re-verifying via a `tracing::debug` printf inside `rms_norm()`.

What this audit explicitly rules out:
* The new CustomOp2 path itself (kill switch is engaged; `RmsNormCustomOp` is never invoked).
* Test-code changes in trainer.rs and #639 (production loop is byte-equal pre- and post-test-edits).
* FLCE config drift (`KILN_USE_FLCE=1` set per cell at all three commits; FLCE call sites in trainer.rs are byte-equal line-for-line).
* Cargo dependency drift (no Cargo.lock changes).
* Static initialization in the new CUDA kernel files (no `cudaMalloc`/ctor/`__device__` globals).

## Recommended next slice

**Tier 1 (cheap diagnostic, ~15 min pod time):** add `tracing::info!` at the top of `rms_norm()` dispatch with the values of `kernel_disabled`, `bwd_disabled`, `supports(x, weight)`, and which branch is taken. Re-run the bench at `1edecd7` and confirm every call site logs `kernel_disabled=true → rms_norm_fallback`. This rules out (3) and confirms (1)/(2) is the real space.

**Tier 2 (allocator instrumentation, ~30 min pod time):** patch `crates/kiln-train/src/trainer.rs` to call `nvidia-smi --query-gpu=memory.used` (or candle's `device.metric("memory.used")` if exposed) immediately before and after each segment's forward and backward pass. Walk the per-segment delta-VRAM trace at `c983ca7` and `1edecd7` to identify which segment grows and by how much. If the delta is uniform across all 8 segments, it's hypothesis (1) — a fatter pool. If it's concentrated in specific segments (e.g. those with more RMSNorm sites), it's hypothesis (2) — a graph-structure shift.

**Tier 3 (revert-and-verify):** if Tier 1+2 don't pin a mechanism, revert *only* `crates/kiln-rmsnorm-kernel/src/lib.rs` and `csrc/fused_rmsnorm_bwd.{cu,h}` and `build.rs` from `1edecd7` back to `c983ca7` state (keeping `forward.rs` post-#638 — but it's a no-op when kill switch set). Re-run the bench. If peak VRAM drops back to ~23 GiB, the regression is in the kernel crate's binary footprint (hypothesis 1). If not, the regression is in the `forward.rs` dispatch wrapper itself.

This audit is **bisect-only** by design — pinning the offending PR. The mechanism diagnosis and fix are the next slice's scope. The Phase 10 §1 RMSNorm work is gated on understanding why a PR that documents itself as "saves 21 GB of saved F32 intermediates per training segment" *also* somehow adds 18.2 GiB of unaccounted peak VRAM when the kill switch is set — i.e., before the per-layer reduction it was supposed to deliver is even on.

## Cost

* Pod: A40 on-demand at $0.44/hr.
* Time on pod: ~15 min wall-clock (3 builds with sccache hit ≥ 98.8% + 3 SFT runs at ~10–15s each + setup).
* Cost: well within the 90 min / $40 cap. No SSH-wedge incidents.

## References

* PR #637 (`c983ca7`): layer-pair time-axis tiled forward+backward — baseline for this bisect.
* PR #638 (`1edecd7`): Liger-style RMSNorm with custom CUDA backward — **regression-introducing PR**.
* PR #639 (`1ad54c2`): fix(trainer) — green main CI, NaN parity tests — bisect confirms zero VRAM impact.
* PR #640 (`cc1fb7e`): Phase 10 §1 RMSNorm E2E SFT validation — the audit doc this bisect closes the loop on.
* [`PHASE10_RMSNORM_E2E.md`](PHASE10_RMSNORM_E2E.md): the audit that discovered and documented the regression but did not pin the offending PR.
