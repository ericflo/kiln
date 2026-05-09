# Vulkan Strix Halo GPU Decode Log, 2026-05-09

## Scope

This log resumes the durable Vulkan optimization audit trail for Qwen3.5-4B on
AMD Strix Halo. Raw command output may still be captured under `/tmp` while a
run is active, but durable decisions, measurements, rollback notes, and failed
experiments belong here and in the matching shortlog.

For future Vulkan optimization work, update this file and the shortlog before
or with every accepted change and every rejected experiment that produces a
useful measurement. Scratch logs are only temporary evidence; this document is
the audit trail.

Do not treat the older May 3 Vulkan anchors as correctness baselines unless the
run also proves non-garbage token output. Several very fast historical anchors
were taken while the Vulkan path was producing unusable token IDs.

## Host And Harness

- Host GPU: AMD Radeon 8060S Graphics (RADV_STRIX_HALO), 98304 MB VRAM from
  Linux DRM sysfs.
- Model: local `Qwen3.5-4B`.
- Branch: `codex/vulkan-gpu-decode-correctness`.
- PR: `https://github.com/ericflo/kiln/pull/1001`.
- Primary serial harness:
  `KILN_BENCH_LOG_ITL=1 KILN_BENCH_LOG_TOKENS=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
- Short profiling harness:
  `KILN_PROFILE_PAGED_LAYERS=1 KILN_PROFILE_GDN_STAGES=1 KILN_PROFILE_FULL_ATTN_STAGES=1 KILN_PROFILE_MLP_STAGES=1 KILN_BENCH_LOG_ITL=1 KILN_BENCH_LOG_TOKENS=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 2 --skip-training`
- Batch smoke shape:
  `/v1/completions/batch`, four distinct prompts, `temperature=0`, `top_p=1`,
  `max_tokens=4`, measured only after background inference prewarm completes.

## Current Validation Gates

Run on this Linux/Strix Halo host:

- `rustfmt --edition 2024 --check crates/kiln-model/src/forward.rs`
- `cargo check -p kiln-model`
- `cargo check -p kiln-model --features vulkan`
- `cargo test -p kiln-model test_backend_linear_decode_adds_lora_delta --lib -- --nocapture`
- `cargo test -p kiln-model test_gdn_chunkwise_masks_decay_before_exp --lib -- --nocapture`
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`

Platform checks attempted but blocked before project typecheck on this host:

- CUDA: `cargo check -p kiln-model --features cuda` stops in `cudarc` because
  `nvcc --version` is not available.
- Metal: `cargo check -p kiln-model --features metal` stops in `objc2` because
  this is not an Apple target.

## Decisions And Experiments

### 2026-05-09 A001: Establish Vulkan Correctness Before Further Perf Work

Problem:
- Recent memory was that Vulkan output may have been garbage.
- Initial release Vulkan bench also failed before decode with
  `unsupported dtype BF16 for op matmul`.

Change:
- Promoted CPU-bound fallback tensors to F32 where Candle CPU matmul needed it.
- Routed projection fallbacks through `backend.linear_decode` on Vulkan where
  supported.
- Fixed GDN chunkwise recurrence masking so masked decay values are not
  exponentiated into `inf * 0 -> NaN`.

Evidence:
- Server selected Vulkan device and returned coherent Qwen reasoning text for
  a short chat smoke instead of repeated punctuation or all-garbage output.
- Short paged bench completed on backend `vulkan` with token IDs
  `[2838,6587,310,5227,1024,75119,220]`.
- Post-fix serial baseline before later improvements:
  prefill around `648.1ms`, mean ITL around `277.2ms`, `3.6 tok/s`.
- `cargo test -p kiln-model test_gdn_chunkwise_masks_decay_before_exp --lib -- --nocapture`
  passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed,
  19 tests.

Verdict:
- Keep. Correct GPU-routed Vulkan output is table stakes and the starting point
  for all further throughput claims.
- Commit: `6bcc114804c0153d7d64d3bd1810feac2cd4ba7d`
  (`fix vulkan gpu decode correctness`).

### 2026-05-09 A002: Route LoRA Base Projections Through Backend Decode

Problem:
- With LoRA present, projection helpers fell back to CPU/Candle for both base
  projection and LoRA delta. That lost Vulkan's cached base projection path.

Change:
- `linear_with_lora_t_backend_decode_if` now first asks
  `backend.linear_decode(x, weight_t)` for the base projection, then adds the
  LoRA delta if present.
- Full-attention projections, MLP projections, GDN projections, and LM-head
  paths can use the same backend hook.

Evidence:
- Focused unit test `test_backend_linear_decode_adds_lora_delta` passed.
- Synthetic rank-8 Qwen-shaped LoRA projection benchmark:
  `backend_ms=4.084`, `fallback_ms=58.747`, `speedup=14.385x`, parity max
  diff below `1e-4`.
- No-LoRA smoke retained the same coherent token IDs and mean ITL around
  `277.1ms`, so the helper did not regress the normal path.

Verdict:
- Keep. This is a measurable LoRA-path gain without changing CUDA/Metal because
  those backends still decline `linear_decode` through the default `Ok(None)`.
- Commit: `bd585d3` (`route lora projections through backend decode`).

### 2026-05-09 A003: Reject Simple Shader/Transfer Retiles That Do Not Win

Experiments rejected before this log was created:

- Direct host-visible input/output for single-submit `linear_decode`.
  Parity passed, but mean ITL regressed to about `278.9ms`.
- Single-token `linear_decode.comp` retile to 32x8.
  Parity passed, bench around `276.5ms`, effectively noise against baseline.
- Single-token `linear_decode.comp` retile to 64x4.
  Parity passed, bench around `285.0ms`, regression.
- `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED=1`.
  Same tokens but around `274.7ms`, not a real win.
- `KILN_ENABLE_VULKAN_FUSED_CONV1D=1`.
  Around `285.2ms`, regression.
- Both fused GDN and fused conv1d enabled.
  Around `291.9ms`, regression.
- `KILN_DISABLE_VULKAN_LINEAR_DECODE=1`.
  Correct but much slower, mean ITL around `646.1ms`.

Verdict:
- Do not repeat these without a new hypothesis. The immediate bottleneck was
  missed backend routing, not another small workgroup retile.

### 2026-05-09 A004: Route GDN Out Projection Through Backend Decode

Problem:
- Profiling after correctness showed GDN `out_proj` was still using the generic
  helper instead of the backend linear decode hook.
- Previous profile over prefill plus two decode steps:
  `gdn out_proj total=337.330ms`, `count=72`, `mean=4.685ms`.

Change:
- In `gated_deltanet_forward_decode_if`, changed GDN `out_proj` to call
  `linear_with_lora_t_backend_decode_if(Some(backend), ...)`.

Evidence:
- Short serial bench stayed on the same token IDs
  `[2838,6587,310,5227,1024,75119,220]`.
- Serial latency improved from the corrected baseline around `277ms` mean ITL
  to `179.2ms` mean ITL (`5.6 tok/s`), with prefill `525.1ms` in that run.
- New profile:
  `gdn out_proj total=51.707ms`, `count=72`, `mean=0.718ms`.
- Warmed four-prompt greedy batch smoke improved from the earlier `~4.77s`
  baseline to `~4.03s` wall for 16 completion tokens.
- Validation gates passed:
  `rustfmt --edition 2024 --check crates/kiln-model/src/forward.rs`,
  `cargo check -p kiln-model`,
  `cargo check -p kiln-model --features vulkan`,
  focused LoRA and GDN tests, Vulkan GDN parity suite, and release Vulkan build.

Verdict:
- Keep. This is the largest clean serial gain in this work so far.
- Commit: `f8820c1` (`route gdn out projection through backend decode`).
- Raw scratch profile was captured at
  `/tmp/kiln-vulkan-profile-gdn-out-1778292150.log`; the durable summary is
  this entry.

### 2026-05-09 A005: Generic Continuous Batch Path Audit

Observation:
- The normal `/v1/completions/batch` route with greedy sampling uses the live
  greedy batcher and did not clearly exercise the generic sampled continuous
  batching path.
- A sampled four-prompt batch without explicit actor routing completed in
  `~3.93s` before generic-path edits and `~3.91s` after, which is below noise
  and not evidence of a real performance win.

Experiment:
- Enabled the explicit batching actor with `KILN_BATCHING_ENGINE=1` and sent a
  sampled four-prompt batch, `temperature=0.7`, `top_p=0.95`, `top_k=40`,
  `seed=1234`, `max_tokens=12`.

Result:
- Current actor path returned HTTP 500 before completing generation:
  `batched decode transformer block 3 (full attention, paged): backend declined batched contiguous paged attention`.

Interpretation:
- Vulkan does not currently provide the batched contiguous paged full-attention
  backend used by the generic continuous batching path.
- Treat this as a correctness/availability issue for continuously batched
  sampled or non-uniform traffic, not as optional performance work.

Status:
- Fixed by A006. The next performance target is still a true Vulkan batched
  paged full-attention backend; the fallback below is an availability fix, not
  the final fast path.

### 2026-05-09 A006: Rowwise Full-Attention Fallback For Generic Continuous Batch

Problem:
- With `KILN_BATCHING_ENGINE=1`, sampled/non-uniform continuous batch traffic
  entered `model_forward_paged_batched_decode_hidden`.
- That loop called the contiguous batched full-attention primitive and treated
  backend decline as fatal.
- Vulkan currently declines that batched paged full-attention hook, so sampled
  continuous batch returned HTTP 500.

Change:
- In the generic batched model loop, if
  `transformer_block_paged_decode_contiguous_batch` declines for a
  full-attention layer, run that full-attention block rowwise and concatenate
  the row outputs back into the batch.
- The GDN and MLP portions of the generic batch loop remain batch-shaped.
- The generic sampled path can also use the backend-aware MLP helper and a
  backend-aware LM-head projection path where supported.

Evidence:
- Before fallback:
  `KILN_BATCHING_ENGINE=1`, sampled four-prompt batch, `temperature=0.7`,
  `top_p=0.95`, `top_k=40`, `seed=1234`, `max_tokens=12` returned HTTP 500:
  `backend declined batched contiguous paged attention`.
- After fallback:
  same request returned HTTP 200, wall `9159.9ms`, usage
  `prompt_tokens=68`, `completion_tokens=48`, `total_tokens=116`, all four
  finish reasons `length`.
- Rebuilt release server rerun after validation:
  equivalent sampled four-prompt actor batch returned HTTP 200,
  `time_total=9.525668s`, usage `prompt_tokens=72`, `completion_tokens=48`,
  `total_tokens=120`, all four finish reasons `length`.
- Serial paged Vulkan bench after this change stayed coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]`, prefill `553.0ms`, mean ITL
  `174.7ms`, `5.7 tok/s`.
- Validation gates passed:
  `rustfmt --edition 2024 --check crates/kiln-model/src/forward.rs`,
  `git diff --check`, `cargo check -p kiln-model`,
  `cargo check -p kiln-model --features vulkan`, focused LoRA and GDN tests,
  Vulkan GDN parity suite, and release Vulkan build.
- CUDA and Metal feature checks were attempted again. CUDA is blocked by
  missing `nvcc`; Metal is blocked by `objc2` requiring an Apple target.

Verdict:
- Keep. This restores correctness/availability for sampled continuous batching
  on Vulkan without changing the already-good serial token output.
- Do not count as a throughput win. The fallback is slower than a real batched
  paged-attention Vulkan implementation should be.

## Current Open Work

- Implement a real Vulkan batched paged full-attention backend if sampled or
  non-uniform continuous batching needs throughput, because A006 is only a
  rowwise availability fallback.
- Keep updating this log and
  `docs/audits/vulkan-strix-halo-2026-05-09-gpu-decode-shortlog.md` for every
  accepted or rejected optimization experiment.
