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
- Base as of A011: `origin/main` at
  `459162855b5cb00265e53a44ed1c8bce642bcde2`
  (`fix(metal): drop &mut from paged_kv_write_token_major helpers (#1002)`).
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
- Commit after rebase onto PR #1002:
  `717a7de305b83cdefb412d0604a288f3b7d5748b`
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
- Commit after rebase onto PR #1002:
  `3066ed45a71a6a88ea1176ef22ebf52a90e0c1b8`
  (`route lora projections through backend decode`).

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
- Commit after rebase onto PR #1002:
  `6ae731c09ae1291dbe982ffe051cc2d83f99d143`
  (`route gdn out projection through backend decode`).
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

### 2026-05-09 A007: Recheck Existing Vulkan Feature Gates Before New Shader Work

Reason:
- Before writing new kernels, re-ran the existing Vulkan env-gated paths to see
  whether any recently landed CUDA or Metal-style work had made an older Vulkan
  switch worth enabling.

Experiment:
- Compared the default paged serial path with these toggles:
  `KILN_DISABLE_VULKAN_FULL_ATTN_QKV=1`,
  `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED=1`,
  `KILN_ENABLE_VULKAN_FUSED_CONV1D=1`, both fused GDN and fused conv1d
  together, and `KILN_DISABLE_VULKAN_MLP_DECODE=1`.

Evidence:
- All runs preserved the corrected token IDs
  `[2838,6587,310,5227,1024,75119,220]`.
- First pass:
  default `529.6ms` prefill, `184.5ms` mean ITL;
  disable full-attn QKV `561.9ms`, `175.5ms`;
  GDN fused `576.1ms`, `178.8ms`;
  fused conv1d `544.4ms`, `197.3ms`;
  GDN fused plus conv1d `564.0ms`, `197.1ms`;
  disable MLP decode `839.7ms`, `209.8ms`.
- Full-attn QKV reruns were noisy and did not prove a win:
  default reruns `169.3ms` and `176.5ms` mean ITL; disabled-QKV reruns
  `171.5ms` and `178.0ms` mean ITL.

Verdict:
- Keep the default gates.
- Reject fused conv1d and disabling MLP decode as clear regressions.
- Do not enable the experimental fused GDN decode path from this evidence.
- Keep Vulkan full-attn QKV enabled; disabling it is noise at best.

### 2026-05-09 A008: Route Full-Attention Out Projection Through Backend Decode

Problem:
- After A006, profiling showed full-attention decode layer time was still much
  higher than the measured score/softmax/value work explained.
- Added temporary stage profiling around the rowwise full-attention fallback to
  split KV read, grouped layout, scores, softmax, weighted sum, attention gate,
  and `o_proj`.

Finding:
- The full-attention decode fallback bottleneck was `o_proj`, not the grouped
  attention math.
- Before the change, short profile totals over decode-only `seq_len=1` stages:
  `o_proj count=16 total=68.313ms mean=4.270ms`; `decode_scores mean=0.037ms`;
  `decode_softmax mean=0.007ms`; `decode_weighted_sum mean=0.006ms`;
  `kv_read mean=0.009ms`.

Change:
- Routed full-attention output projections through
  `linear_with_lora_t_backend_decode_if(Some(backend), ...)` so Vulkan can use
  cached `linear_decode` for `o_proj`.
- Kept the existing Metal decode GEMV gate argument at the call sites that
  already used it, and passed `false` in non-decode/prefill call sites.
- Left the new full-attention stage profiling behind the existing
  `KILN_PROFILE_FULL_ATTN_STAGES` gate for future audits.

Evidence:
- After the change, the same short profile shape reported
  `o_proj count=16 total=5.413ms mean=0.338ms`.
- Full-attention decode layer profile dropped to
  `count=24 total=133.027ms mean=5.543ms`.
- Short profiled run stayed coherent with token IDs `[2838,6587,310]`,
  prefill `451.9ms`, mean ITL `137.7ms`, `7.3 tok/s`.
- Unprofiled serial harness stayed coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]`, prefill `480.8ms`, mean ITL
  `135.1ms`, `7.4 tok/s`.
- Validation gates passed: `rustfmt --edition 2024 --check
  crates/kiln-model/src/forward.rs`, `git diff --check`,
  `cargo check -p kiln-model`, `cargo check -p kiln-model --features vulkan`,
  focused LoRA and GDN tests, Vulkan GDN parity suite, and release Vulkan
  build.
- CUDA and Metal feature checks were attempted again. CUDA is still blocked by
  missing `nvcc`; Metal is still blocked by `objc2` requiring an Apple target.

Verdict:
- Keep. This is a real serial decode gain: A006's accepted serial anchor was
  `174.7ms` mean ITL, and A008 measures `135.1ms` with the same tokens.
- The next serial hotspots are the remaining GDN input projection, MLP fused
  decode, and full-attention QKV projection work; do not start there until this
  change is committed and pushed.

### 2026-05-09 A009: Reject MLP Decode Single-Submit Recording

Hypothesis:
- Recent CUDA work moved toward graph replay and reduced per-step synchronization.
  Vulkan MLP decode still records and submits separate operations around upload,
  gate/up, down projection, and readback.
- A single command buffer containing upload, gate/up, down projection, and
  readback might reduce queue-submit overhead for the MLP hotspot.

Experiment:
- Implemented an uncommitted `dispatch_mlp_decode_cached_single_submit` path in
  `crates/kiln-vulkan-kernel/src/kernels.rs`, default-enabled behind
  `KILN_DISABLE_VULKAN_MLP_DECODE_SINGLE_SUBMIT`.
- The focused parity test `cargo test -p kiln-vulkan-kernel
  mlp_decode_matches_cpu_reference --test gdn_parity -- --nocapture` passed.
- The release Vulkan bench build also passed.

Evidence:
- Default single-submit trial preserved token IDs
  `[2838,6587,310,5227,1024,75119,220]` but regressed the serial harness to
  prefill `529.5ms`, mean ITL `159.6ms`, `6.3 tok/s`.
- Rerunning the same binary with
  `KILN_DISABLE_VULKAN_MLP_DECODE_SINGLE_SUBMIT=1` preserved the same token IDs
  and returned to prefill `446.7ms`, mean ITL `135.8ms`, `7.4 tok/s`, matching
  the A008 anchor.

Verdict:
- Rejected and removed before commit. The single-submit recording changed
  synchronization shape but did not improve wall-clock decode on Strix Halo.
- Do not retry this exact MLP single-submit path without profiling why the
  combined command buffer loses; the next MLP attempt should target shader
  arithmetic or residency, not merely submit count.

### 2026-05-09 A010: Full-Attention QKV Single-Submit Dispatch

Hypothesis:
- The full-attention QKV fused shader still used separate helper calls for x
  upload, compute dispatch, and output readback. Unlike the rejected MLP
  single-submit trial, this path is a single compute kernel, so combining the
  transfer/compute/readback sequence into one command buffer should reduce
  queue-submit overhead without changing shader arithmetic.

Change:
- Added a default-enabled single-submit path in
  `dispatch_full_attn_qkv_decode_cached`.
- Added rollback guard
  `KILN_DISABLE_VULKAN_FULL_ATTN_QKV_SINGLE_SUBMIT=1`.
- Kept the existing multi-submit path as the rollback implementation.

Evidence:
- Focused QKV parity passed with the default path and with the rollback env.
- Full Vulkan kernel parity suite passed, 19 tests.
- Default no-profile serial runs preserved the known token IDs
  `[2838,6587,310,5227,1024,75119,220]` and measured `132.3ms` and `133.1ms`
  mean ITL.
- Rollback first measured `134.2ms` mean ITL with the same token IDs; a second
  rollback run measured `139.0ms` but diverged to token IDs
  `[2838,29772,220,16,17,18237,791]`, so treat rollback timing as noisy rather
  than a strict same-token comparison.
- Short profiled run with default single-submit stayed coherent with token IDs
  `[2838,6587,310]`, prefill `458.2ms`, mean ITL `131.5ms`.
- The profiled full-attention QKV bucket moved only slightly:
  A008 `qkv_proj count=24 total=59.768ms mean=2.490ms`; A010
  `qkv_proj count=24 total=59.019ms mean=2.459ms`.

Validation:
- `rustfmt --edition 2024 --check crates/kiln-vulkan-kernel/src/kernels.rs`
- `git diff --check`
- `cargo check -p kiln-model`
- `cargo check -p kiln-model --features vulkan`
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`

Verdict:
- Keep, but characterize as a small Vulkan-only source win. The benefit is in
  whole-token no-profile latency, while stage profiling shows only a marginal
  QKV bucket reduction.
- No CUDA or Metal source is touched by this change. CUDA/Metal feature checks
  remain environment-blocked on this Linux host as recorded above.

### 2026-05-09 A011: Vulkan Dyn-Seqlen Paged Decode Batch Attention

Problem:
- A006 restored sampled/non-uniform continuous batching by falling back to
  rowwise full-attention blocks when Vulkan declined the batched paged decode
  attention hook.
- That made the actor path correct and available but slow: the accepted A006
  rebuilt sampled four-prompt actor batch took `time_total=9.525668s` for
  `48` completion tokens.

Change:
- Added `paged_attn_decode_batch.comp`, a decode-only Vulkan F32 attention
  shader for compacted K/V windows with per-row sequence lengths.
- Added `dispatch_paged_attn_decode_batch_f32` and pipeline/prewarm entries.
- Implemented
  `VulkanBackend::flash_attn_paged_decode_contiguous_batch_dyn_seqlen` for
  CPU F32 tensors, `q_len=1`, block-table/seqused-k addressing, and integer
  GQA.
- Added dedicated rollback guard
  `KILN_DISABLE_VULKAN_PAGED_ATTN_DECODE_BATCH=1`.
- During K/V compaction, unused positions beyond each row's live sequence
  length are filled with slot 0 rather than validating unused block-table
  entries; the shader reads only `t < seq_len`.

Evidence:
- New direct parity test
  `paged_attn_decode_batch_matches_cpu_reference` passed.
- Full Vulkan kernel parity suite passed with the new test included:
  `20 passed`.
- Release server with `KILN_BATCHING_ENGINE=1` selected Vulkan on
  `AMD Radeon 8060S Graphics (RADV STRIX_HALO)` and completed background
  prewarm.
- Default sampled four-prompt actor batch, `temperature=0.7`, `top_p=0.95`,
  `top_k=40`, `seed=1234`, `max_tokens=12`, returned HTTP 200 with
  `time_total=7.754065s`, usage `prompt_tokens=82`,
  `completion_tokens=48`, `total_tokens=130`, all finish reasons `length`.
- A second sampled `/no_think` four-prompt actor batch returned HTTP 200 with
  `time_total=7.709984s`, usage `prompt_tokens=84`,
  `completion_tokens=48`, `total_tokens=132`, all finish reasons `length`.
  Visible `text` fields were empty in both default and rollback runs because
  the non-streaming chat reasoning split stores pre-`</think>` text in hidden
  `reasoning_content`, and the batch item serializer does not expose that
  field.
- Rollback server with
  `KILN_DISABLE_VULKAN_PAGED_ATTN_DECODE_BATCH=1` on the same `/no_think`
  request returned the same visible response shape and usage, but took
  `time_total=8.571721s`.
- Serial paged bench after A011 stayed coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]`, prefill `460.6ms`, mean ITL
  `130.0ms`, `7.7 tok/s`.

Validation:
- `rustfmt --edition 2024 --check crates/kiln-vulkan-kernel/src/kernels.rs crates/kiln-vulkan-kernel/tests/gdn_parity.rs crates/kiln-model/src/backend/vulkan.rs`
- `git diff --check`
- `cargo check -p kiln-model`
- `cargo check -p kiln-model --features vulkan`
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
- `cargo test -p kiln-model test_backend_linear_decode_adds_lora_delta --lib -- --nocapture`
- `cargo test -p kiln-model test_gdn_chunkwise_masks_decay_before_exp --lib -- --nocapture`
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`
- Server sampled batch smoke with default path and rollback env.
- Serial paged Vulkan bench with token logging.

Verdict:
- Keep. This replaces A006's rowwise full-attention fallback for supported
  sampled/non-uniform Vulkan continuous batches and improves the measured actor
  batch from `9.525668s` to `7.754065s` on the primary sampled smoke.
- This is still not a final fast path: the backend compacts K/V windows on the
  CPU and uploads compact K/V each call. The next batch-attention target should
  avoid that compaction/upload by reading the paged KV pool and block table
  directly in Vulkan or by making K/V residency explicit.

### 2026-05-09 A012: Reject MLP Gate/Up Retiles

Hypothesis:
- Current serial decode profiling after A011 showed the largest remaining
  decode-stage bucket was `mlp:fused`: over a short profiled run with three
  generated tokens, `mlp:fused total=200.336ms`, `count=96`,
  `mean=2.087ms`.
- The single-token `mlp_gate_up_decode.comp` shader used `64x4` local geometry:
  64 output columns per workgroup and 4 reduction lanes per column.
- Testing one higher-reduction variant (`32x8`) and one lower-overhead variant
  (`128x2`) would show whether this simple geometry is a real serial hotspot
  lever.

Experiment:
- Trial 1 changed single-token `mlp_gate_up_decode.comp` to `32x8` and changed
  the dispatch width to `intermediate.div_ceil(32)`.
- Trial 2 changed the same shader to `128x2` and changed dispatch width to
  `intermediate.div_ceil(128)`.
- Both trials were kept uncommitted and restored before this entry.

Evidence:
- Trial 1 focused parity:
  `cargo test -p kiln-vulkan-kernel mlp_decode_matches_cpu_reference --test gdn_parity -- --nocapture`
  passed.
- Trial 1 serial bench preserved token IDs
  `[2838,6587,310,5227,1024,75119,220]` but regressed to prefill `464.4ms`,
  mean ITL `134.1ms`, `7.5 tok/s`.
- Trial 2 focused parity passed with the same command.
- Trial 2 serial bench preserved the same token IDs but regressed to prefill
  `451.9ms`, mean ITL `134.6ms`, `7.4 tok/s`.
- Current accepted anchors before the trials are A010/A011: serial mean ITL
  roughly `130.0ms` to `132.3ms` with the same token IDs.

Verdict:
- Reject both retiles and keep the original `64x4` single-token MLP gate/up
  geometry.
- The next MLP attempt should target arithmetic intensity, reduced memory
  traffic, data residency, or a model-specific fused gate/up/down design; simple
  workgroup geometry changes were not enough.

### 2026-05-09 A013: Reject Generic Vulkan Fused GDN Batch Hook

Hypothesis:
- Vulkan already had an implementation of
  `gdn_decode_gates_recurrent_rmsnorm`, but the generic GDN forward path did
  not call that backend trait hook.
- Wiring the hook for `seq_len == 1` could let sampled continuous batches use
  Vulkan's fused gates + recurrent update + gated RMSNorm path, instead of the
  split gates/recurrent/gated-norm stages shown in profiling.

Experiment:
- Added an uncommitted generic call to
  `backend.gdn_decode_gates_recurrent_rmsnorm(...)` after the existing Metal
  direct fast path.
- Added an uncommitted rollback env,
  `KILN_DISABLE_VULKAN_GDN_DECODE_FUSED_BATCH=1`, around Vulkan batch use.
- Kept Vulkan serial `batch == 1` behavior behind the existing opt-in
  `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED` gate.

Evidence:
- `cargo check -p kiln-model --features vulkan` passed.
- Release build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Serial paged bench stayed coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]`, prefill `427.4ms`, mean ITL
  `131.3ms`.
- Primary sampled actor batch with the fused GDN batch hook returned HTTP 200
  but regressed badly: `time_total=14.470374s` for `48` completion tokens.
- The same binary with
  `KILN_DISABLE_VULKAN_GDN_DECODE_FUSED_BATCH=1` returned HTTP 200 with
  `time_total=7.668599s`, matching the A011 performance envelope.

Verdict:
- Rejected and removed before commit. The existing fused GDN batch kernel is
  correct enough to return HTTP 200 on this smoke, but it is much slower in the
  current pathway.
- Do not simply wire this trait hook by default. A future GDN batch fusion
  should first address state/input residency and avoid extra upload/readback
  churn around the fused kernel.

### 2026-05-09 A014: Reject Sampled Contiguous-Batch Logits Route

Hypothesis:
- The continuous sampled batch path falls through
  `model_forward_paged_batched_decode_hidden`, while the accepted A011 work
  added a faster dyn-seqlen contiguous-batch paged attention backend.
- Routing sampled `row_count > 1` decode steps through
  `model_forward_paged_decode_contiguous_batch` and then sampling per row from
  full logits might reuse the A011 backend path for non-greedy continuous
  batches.

Experiment:
- Added an uncommitted `decode_logits_paged_contiguous_batch` helper on
  `ModelRunner`.
- The helper assembled batched `LinearAttentionState`, called
  `model_forward_paged_decode_contiguous_batch`, scattered state rows back, and
  returned batch logits for per-row sampling.
- Added an uncommitted rollback env,
  `KILN_DISABLE_PAGED_CONTIG_BATCH_LOGITS=1`, to compare against the existing
  generic sampled batch route in the same binary.
- Removed the code after measurement.

Evidence:
- `cargo check -p kiln-model --features vulkan` passed.
- Release build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Serial paged bench stayed coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]`, prefill `424.5ms`, mean ITL
  `133.6ms`.
- Primary sampled actor batch with the logits route returned HTTP 200 for
  `82` prompt tokens and `48` completion tokens, but measured
  `time_total=8.430564s` before the rollback env and `time_total=8.311119s`
  after rebuilding with the env gate.
- The same binary with `KILN_DISABLE_PAGED_CONTIG_BATCH_LOGITS=1` returned HTTP
  200 and measured `time_total=7.757198s`, matching the accepted A011/A013
  performance envelope.

Verdict:
- Rejected and removed before commit. The route is correctness-safe on the
  sampled smoke, but it is slower than the existing generic path for the primary
  4-row continuous sampled batch.
- Do not pursue a broad sampled logits reroute without first reducing the
  state packing/scatter and logits materialization overhead. A narrower future
  attempt should attach the A011 dyn-seqlen attention win inside the existing
  generic sampled path or add decode-stage profiling there before changing the
  route.

### 2026-05-09 A015: Online Softmax for Vulkan Dyn-Seqlen Paged Decode Batch

Hypothesis:
- The A011 Vulkan dyn-seqlen paged decode attention shader recomputed each QK
  dot product three times: max pass, denominator pass, and value pass.
- Replacing those passes with the standard online softmax recurrence should
  preserve numerical stability while computing each QK dot product once.
- This mirrors the row-max/row-sum correction pattern used by modern CUDA
  attention implementations, including the vLLM/CUTLASS MLA decode path
  inspected during this experiment.

Change:
- Updated `paged_attn_decode_batch.comp` to maintain `row_max`, `row_sum`, and
  a rescaled per-lane value accumulator in one token loop.
- The change is Vulkan-only and does not touch CUDA, Metal, routing, cache
  ownership, or host/device transfer behavior.

Evidence:
- Focused Vulkan parity passed:
  `cargo test -p kiln-vulkan-kernel paged_attn_decode_batch_matches_cpu_reference --test gdn_parity -- --nocapture`.
- Full Vulkan kernel parity passed:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
  (`20` tests passed).
- Vulkan model check passed:
  `cargo check -p kiln-model --features vulkan`.
- Final release build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Final serial paged bench stayed coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]`, prefill `454.0ms`, mean ITL
  `131.8ms`.
- Final primary sampled actor batch returned HTTP 200 for `82` prompt tokens
  and `48` completion tokens at `time_total=7.743591s`, matching the accepted
  A011/A013/A014 envelope.
- Proper old-vs-online rebuild comparison on a longer sampled batch:
  - Old three-pass shader rebuild: HTTP 200, `411` prompt tokens, `64`
    completion tokens, `time_total=27.612193s`.
  - Online-softmax shader rebuild: HTTP 200, same request shape,
    `time_total=27.327221s`.

Verdict:
- Keep. This is a small but measurable win for longer-context sampled
  continuous batches, and it removes redundant per-token QK dot work without
  expanding backend-specific surface area.
- The remaining larger bottleneck is still data movement around the dyn-seqlen
  path: CPU gather/compaction of K/V, upload of compact K/V, and readback of
  attention output.

### 2026-05-09 A016: Reject Direct Rust K/V Compaction for Dyn-Seqlen Batch

Hypothesis:
- The A011 dyn-seqlen path currently builds a gather tensor and uses Candle
  `index_select`/`reshape`/`contiguous` to compact paged K/V before the Vulkan
  dispatch.
- For CPU F32 contiguous K/V pools, directly copying selected slots from Candle
  CPU storage into compact `Vec<f32>` buffers could avoid that tensor churn and
  reduce overhead before the Vulkan upload.

Experiment:
- Added an uncommitted compact-slice Vulkan dispatcher and a backend path that
  gathered K/V slots directly from CPU F32 pool storage.
- Added an uncommitted rollback env,
  `KILN_DISABLE_VULKAN_PAGED_ATTN_DIRECT_COMPACT=1`.
- After short-context regression appeared, tried a thresholded variant that
  only used direct compaction for `max_seqlen_k >= 64`, with
  `KILN_VULKAN_PAGED_ATTN_DIRECT_COMPACT_MIN_SEQLEN` as an uncommitted tuning
  env.
- Removed the code after measurement.

Evidence:
- Focused paged-attention parity passed with the compact-slice dispatcher added.
- `cargo check -p kiln-model --features vulkan` passed.
- Release build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Unthresholded direct compaction returned HTTP 200 but regressed the primary
  short sampled batch to `time_total=7.795046s` and `time_total=7.807411s`.
- The same binary with
  `KILN_DISABLE_VULKAN_PAGED_ATTN_DIRECT_COMPACT=1` returned HTTP 200 at
  `time_total=7.670023s` on the primary short sampled batch.
- Unthresholded direct compaction improved the longer sampled batch to
  `time_total=26.873467s`, versus rollback `time_total=27.058533s`.
- The thresholded default avoided most of the short-context regression
  (`time_total=7.727216s`) but no longer showed a meaningful longer-context
  win (`time_total=27.040179s` versus rollback `27.058533s`).

Verdict:
- Rejected and removed before commit. Direct Rust K/V compaction can help a
  longer request, but the short-context regression and weak thresholded win do
  not justify adding another compaction path.
- The next K/V data-movement attempt should avoid host compaction more
  fundamentally, likely by changing residency/upload strategy rather than
  replacing Candle gather with Rust-side gather.

### 2026-05-09 A017: Reject Exp2 Online Softmax Variant

Hypothesis:
- vLLM/CUTLASS attention code often uses base-2 exponentials with an
  `M_LOG2E` correction factor for softmax scaling.
- Replacing the A015 shader's `exp(...)` calls with
  `exp2(... * 1.4426950408889634)` might map to a faster instruction sequence
  on RADV while preserving output.

Experiment:
- Added an uncommitted `LOG2_E` constant to `paged_attn_decode_batch.comp`.
- Changed the online softmax recurrence scales from `exp(delta)` to
  `exp2(delta * LOG2_E)`.
- Removed the change after measurement.

Evidence:
- Focused paged-attention parity passed.
- Full Vulkan kernel parity passed (`20` tests).
- `cargo check -p kiln-model --features vulkan` passed.
- Release build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Exp2 sampled batch smoke returned HTTP 200 at `time_total=7.689383s` for the
  primary `82` prompt / `48` completion token shape.
- Exp2 longer sampled batch returned HTTP 200 at `time_total=27.282147s` for
  the `411` prompt / `64` completion token shape.
- Fresh non-exp2 rollback evidence from A016 was stronger:
  `time_total=7.670023s` on the primary short shape and
  `time_total=27.058533s` on the longer shape.

Verdict:
- Rejected and removed before commit. The exp2 variant is correct but did not
  beat the current `exp` online softmax shader on the measured sampled batch
  shapes.

### 2026-05-09 A018: Reject Batch-4 MLP Row-Pair Threshold

Hypothesis:
- The Vulkan MLP gate/up and down-projection kernels already include row-pair
  variants that share weight loads across two rows.
- Those row-pair kernels are gated to `batch >= 8`, so the primary 4-row
  continuous sampled decode batch still uses the ordinary batched kernels.
- Lowering the row-pair gate to `batch >= 4` might improve the continuous batch
  decode step without touching single-request serial decode.

Experiment:
- Changed the uncommitted `use_prefill_row_pair_matmul` threshold from
  `batch >= 8` to `batch >= 4`.
- Left the existing `KILN_DISABLE_VULKAN_PREFILL_ROW_PAIR_MATMUL=1` env in
  place for debugging.
- Removed the change after measurement.

Evidence:
- Focused parity passed:
  `cargo test -p kiln-vulkan-kernel mlp_decode_batched_matches_cpu_reference --test gdn_parity -- --nocapture`.
- Focused gate/up parity passed:
  `cargo test -p kiln-vulkan-kernel mlp_gate_up_decode_matches_cpu_reference --test gdn_parity -- --nocapture`.
- `cargo check -p kiln-model --features vulkan` passed.
- Release build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Two fresh default runs of the primary sampled batch with the `batch >= 4`
  threshold returned HTTP 200 but measured `time_total=7.884895s` and
  `time_total=7.811283s`.
- That is slower than the current accepted standard sampled-batch envelope from
  A015/A016/A017 (`~7.67s` to `7.74s`).
- A same-binary env-disabled run measured `8.035747s`, but that is not a clean
  old-threshold rollback because it also disables the existing prefill row-pair
  path for larger row counts.

Verdict:
- Rejected and removed before commit. Batch-4 row-pair MLP kernels are correct
  but did not beat the accepted sampled-batch envelope.
- Keep the row-pair path gated at `batch >= 8`.

### 2026-05-09 A019: Reject Vulkan Two-Stage LoRA Delta Decode

Hypothesis:
- A002 already routes the base projection through `backend.linear_decode` when
  LoRA is active, then computes the low-rank delta with the generic Candle path.
- For a rank-8 adapter, replacing `compute_lora_delta(x, A, B)` with a
  Vulkan two-stage path (`x @ A^T` followed by `hidden @ B^T`) might avoid CPU
  matmul overhead in LoRA serial and continuously batched decode.

Experiment:
- Built a temporary generator for a full-shape synthetic rank-8 PEFT adapter
  under `target/kiln-audit-adapters/synthetic-rank8`; it was not committed.
- The adapter used tiny deterministic non-zero F32 values for all trained
  modules: full-attention `q_proj`, `k_proj`, `v_proj`, `o_proj` on the eight
  full-attention layers, plus MLP `gate_proj`, `up_proj`, and `down_proj` on
  all 32 layers.
- Added uncommitted Vulkan shaders:
  - `lora_a_decode.comp`: compute `[row, rank] = x @ A^T`.
  - `lora_b_decode.comp`: compute `[row, out_dim] = hidden @ B^T * scale`.
- Added an uncommitted backend hook,
  `BackendRuntime::lora_delta_decode`, and a Vulkan impl with rollback env
  `KILN_DISABLE_VULKAN_LORA_DELTA_DECODE=1`.
- Removed all code after measurement.

Evidence:
- The temporary full-shape synthetic adapter loaded through the normal server
  adapter path with `KILN_ADAPTER_DIR=target/kiln-audit-adapters`.
- Baseline active-adapter sampled batch before the shader experiment returned
  HTTP 200 at `time_total=13.581623s` for `98` prompt tokens and `48`
  completion tokens. The first load-including request was `11.836511s` for
  `82` prompt and `48` completion tokens, so subsequent measurements avoided
  cache hits and varied prompt suffixes.
- Focused parity passed:
  `cargo test -p kiln-vulkan-kernel lora_delta_decode_matches_cpu_reference --test gdn_parity -- --nocapture`.
- `cargo check -p kiln-model --features vulkan` passed.
- Release build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- With the Vulkan LoRA delta path enabled, active-adapter sampled batches
  returned HTTP 200 but measured `time_total=14.988255s` and
  `time_total=15.065539s` for `94` prompt tokens and `48` completion tokens.
- Same-binary rollback with
  `KILN_DISABLE_VULKAN_LORA_DELTA_DECODE=1` returned HTTP 200 at
  `time_total=12.825559s` and `time_total=12.724448s` for `98` prompt tokens
  and `48` completion tokens.

Verdict:
- Rejected and removed before commit. The shader math is correct, but the
  implementation adds two Vulkan dispatches per LoRA projection. With full
  rank-8 LoRA coverage, that launch/readback shape loses to the current CPU
  low-rank delta path.
- Future LoRA work should target fused projection+delta kernels, adapter
  residency, or a way to fuse LoRA into the existing MLP/full-attention
  projection dispatches. A standalone two-stage delta backend is not the right
  next step.

### 2026-05-09 A020: Reject Vulkan Batched GDN In-Projection 32x8 Tile

Hypothesis:
- The current batched GDN input-projection shader uses an unusual
  `local_size_x = 80, local_size_y = 3` tile. Recent profiles show GDN
  `in_proj` remains a top decode hotspot, and CUDA/Metal work elsewhere points
  toward regular tiles and better occupancy. A `32x8` tile might improve memory
  access and reduction balance for batched decode.

Experiment:
- Temporarily changed
  `crates/kiln-vulkan-kernel/csrc/shaders/gdn_in_proj_decode_batched.comp`
  from `80x3` to `32x8`.
- Updated the two matching Vulkan dispatch group counts in
  `crates/kiln-vulkan-kernel/src/kernels.rs` from `div_ceil(80)` to
  `div_ceil(32)`.
- Restored both files after measurement.

Evidence:
- Focused parity passed:
  `cargo test -p kiln-vulkan-kernel gdn_in_proj_decode_batched_matches_cpu_reference --test gdn_parity -- --nocapture`.
- `cargo check -p kiln-model --features vulkan` passed.
- Release build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- The 32x8 candidate server path used Vulkan on
  `AMD Radeon 8060S Graphics (RADV STRIX_HALO)` and returned HTTP 200 for the
  sampled four-prompt batch, but measured `time_total=9.170221s` and
  `time_total=9.119555s` for `106` prompt tokens and `48` completion tokens.
- Same-branch rollback to the original 80x3 tile returned HTTP 200 at
  `time_total=8.824926s` for the same `106` prompt-token /
  `48` completion-token shape. Additional rollback samples were
  `time_total=8.742473s` and `time_total=8.506257s` for `102` prompt tokens and
  `48` completion tokens.

Verdict:
- Rejected and removed before commit. The `32x8` tile is correct but slower in
  the sampled continuous-batch serving path.
- Keep the existing `80x3` batched GDN input-projection geometry until a
  broader GDN residency/fusion change can reduce launch and data-movement cost.

### 2026-05-09 A021: Reject Vulkan MLP Gate/Up Two-Column Decode Shader

Hypothesis:
- The Metal audit accepted a two-adjacent-column MLP gate/up decode kernel and
  rejected wider cooperative variants. Vulkan serial profiling still ranks
  `mlp:fused` as the largest decode-stage bucket, so the same two-column idea
  might reduce repeated hidden-row loads in the single-token Vulkan F32 shader.

Experiment:
- Temporarily changed
  `crates/kiln-vulkan-kernel/csrc/shaders/mlp_gate_up_decode.comp` so each
  `local_size_x = 64, local_size_y = 4` workgroup computed two adjacent output
  columns per `col_lane` and covered `128` intermediate columns per workgroup.
- Updated the two matching single-row MLP gate/up launch counts in
  `crates/kiln-vulkan-kernel/src/kernels.rs` from `div_ceil(64)` to
  `div_ceil(128)`.
- Restored both files after measurement.

Evidence:
- Focused MLP parity passed:
  `cargo test -p kiln-vulkan-kernel mlp_ --test gdn_parity -- --nocapture`
  (`3` tests passed).
- `cargo check -p kiln-model --features vulkan` passed.
- Candidate release build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Candidate serial paged bench stayed coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]`, but regressed to
  `141.6ms` mean ITL.
- Candidate short profile showed the targeted bucket got worse:
  `mlp:fused total=218.030ms`, `count=96`, `mean=2.271ms`, versus the
  pre-trial profile at about `202.545ms` total / `2.110ms` mean.
- Same-branch rollback to the original one-column shader rebuilt cleanly and
  returned to coherent token IDs
  `[2838,6587,310,5227,1024,75119,220]`, `450.9ms` prefill, and
  `129.1ms` mean ITL.

Verdict:
- Rejected and removed before commit. On Strix Halo Vulkan F32, the extra
  accumulators/shared memory and lower workgroup count lose more than the
  adjacent-column input-load reuse saves.
- Do not port Metal's two-column gate/up decode shape directly to Vulkan
  without a different shader structure or new hardware/profile evidence.

### 2026-05-09 A022: Reject Direct Decode-Loop GDN Resident-State Scope

Hypothesis:
- The Vulkan backend already has a resident recurrent-state path for GDN
  decode, and the serial bench uses a resident-state scope around its decode
  loop. The real direct generation loops did not. Wrapping those direct loops
  could avoid recurrent-state readback/upload across generated tokens in
  non-streaming and non-batched streaming paths.

Experiment:
- Temporarily added a small RAII guard in `crates/kiln-model/src/generate.rs`
  that called `BackendRuntime::enter_gdn_recurrent_resident_state_scope` before
  direct decode loops and exited it on drop.
- Applied it to `generate_from_tokens_paged_interleaved`.
- Applied it to `run_stream_decode_loop_with_first` only when the greedy live
  decode batcher would not handle the decode step. This avoided handing a stale
  CPU `LinearAttentionState` to the batcher worker after a caller-thread
  resident-state direct fallback.
- Removed the code after measurement.

Evidence:
- `rustfmt --edition 2024 --check crates/kiln-model/src/generate.rs` passed
  while the temporary source was applied.
- `cargo check -p kiln-model --features vulkan` passed.
- Release build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Candidate direct non-streaming `/v1/chat/completions` requests used Vulkan
  on `AMD Radeon 8060S Graphics (RADV_STRIX_HALO)` and returned HTTP 200.
- Candidate 24-token sampled direct requests measured `time_total=4.975489s`
  and `4.920384s` for `30` prompt tokens and `24` completion tokens.
- Same-binary rollback with
  `KILN_DISABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE=1` measured
  `time_total=4.796903s` for the same `30` prompt-token /
  `24` completion-token shape; nearby rollback runs were `4.969028s` and
  `4.949892s` with `31` prompt tokens.
- Longer 48-token check also regressed: rollback measured
  `time_total=8.307010s` for `28` prompt tokens and `48` completion tokens,
  while the candidate measured `time_total=8.477510s` for the same token
  shape.

Verdict:
- Rejected and removed before commit. The resident-state scope is useful for
  the benchmark's long-lived decode loop, but in the real direct API path the
  added residency bookkeeping did not beat the current state readback/upload
  behavior.
- A production resident-state win likely needs recurrent state to become an
  owned request/batcher resource instead of a thread-local temporary scope.

### 2026-05-09 A023: Accept Packed BF16 Weights for Vulkan Linear Decode

Hypothesis:
- The model's projection weights are BF16 on disk, but the Vulkan decode weight
  cache expands them to F32 device buffers. The transposed GEMV and LM-head
  argmax kernels are bandwidth-heavy, so storing immutable linear weights as
  two packed BF16 values per `u32` and expanding in shader with
  `uintBitsToFloat(bits << 16)` should reduce memory traffic without requiring
  native shader BF16 arithmetic.

Implementation:
- Added packed-BF16 Vulkan shader variants for linear decode, batched linear
  decode, single-row argmax blocks, and batched argmax blocks.
- Added `upload_tensor_bf16_packed_buffer` and BF16-weight dispatch entry points
  in `crates/kiln-vulkan-kernel/src/kernels.rs`.
- Added a separate packed-BF16 weight cache in
  `crates/kiln-model/src/backend/vulkan.rs`; it is Vulkan-only and gated by
  `KILN_DISABLE_VULKAN_BF16_PACKED_LINEAR_WEIGHTS=1`.
- Prewarmed packed BF16 weights for no-LoRA linear decode surfaces that use the
  new path during ordinary serving: `embed_tokens_t`, full-attention `o_proj_t`,
  and GDN `out_proj_t`.

Evidence:
- New focused parity tests passed for packed BF16 single linear decode, batched
  linear decode, single argmax, and batched argmax.
- Full Vulkan kernel parity passed:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
  reported `24 passed`.
- `cargo check -p kiln-model --features vulkan` passed.
- Release build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- `git diff --check` passed.
- Serial paged decode stayed coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]`.
- Serial candidate runs measured `121.4ms` and `114.0ms` mean ITL. Same-binary
  rollback with `KILN_DISABLE_VULKAN_BF16_PACKED_LINEAR_WEIGHTS=1` measured
  `132.5ms` mean ITL with the same token IDs.
- Sampled four-request continuous batch with `KILN_BATCHING_ENGINE=1`,
  `temperature=0.7`, `top_p=0.95`, `top_k=40`, `seed=1234`, and `max_tokens=12`
  returned HTTP 200 for all four requests. Candidate wall time was `8.394559s`
  for `111` prompt tokens and `48` completion tokens; rollback wall time on the
  same prompt/request shape was `9.049283s`.
- Greedy four-request continuous batch with `temperature=0`, `top_p=1`,
  `seed=1234`, and `max_tokens=12` returned HTTP 200 for all four requests.
  Candidate wall time was `8.284110s`; rollback wall time was `8.940242s`.
- Synthetic rank-8 LoRA adapter smoke using
  `target/kiln-audit-adapters/synthetic-rank8` returned HTTP 200 for a sampled
  direct request, `28` prompt tokens and `8` completion tokens.

Verdict:
- Keep. This is a measurable Vulkan-only serial and continuous-batch win, with
  coherent output and a same-binary rollback switch.
- CUDA and Metal source paths are not changed by this experiment. Environment
  validation remains Linux-limited: CUDA checks are blocked here by missing
  `nvcc`, and Metal checks require an Apple target.

### 2026-05-09 A024: Accept Packed BF16 Weights for Vulkan GDN In-Projection

Hypothesis:
- GDN `in_proj` is one of the largest remaining serial decode buckets and uses
  four transposed BF16 source matrices expanded to F32 Vulkan buffers. Applying
  the packed-weight strategy from A023 to the fused single-token and batched
  GDN `in_proj` shaders should cut weight bandwidth on both serial and
  continuous-batch paths.

Implementation:
- Added packed-BF16 shader variants for `gdn_in_proj_decode` and
  `gdn_in_proj_decode_batched`.
- Added `dispatch_gdn_in_proj_decode_cached_bf16_weights` in the Vulkan kernel
  crate.
- Routed Vulkan GDN `in_proj` through the packed-BF16 path when all four
  weights are BF16. Rollback:
  `KILN_DISABLE_VULKAN_BF16_PACKED_GDN_IN_PROJ_WEIGHTS=1`.
- Prewarmed packed-BF16 buffers for GDN `in_proj_qkv_t`, `in_proj_z_t`,
  `in_proj_a_t`, and `in_proj_b_t`.

Evidence:
- Focused GDN in-proj parity passed for F32 single, F32 batched, packed-BF16
  single, and packed-BF16 batched variants.
- Full Vulkan kernel parity passed:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
  reported `26 passed`.
- `cargo check -p kiln-model --features vulkan` and `cargo check -p kiln-model`
  passed.
- Release build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- `git diff --check` passed.
- Serial paged decode stayed coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]`.
- Serial candidate runs measured `115.5ms` and `118.2ms` mean ITL. Same-binary
  rollback with `KILN_DISABLE_VULKAN_BF16_PACKED_GDN_IN_PROJ_WEIGHTS=1`
  measured `121.4ms` mean ITL with the same token IDs.
- Sampled four-request continuous batch with `KILN_BATCHING_ENGINE=1`,
  `temperature=0.7`, `top_p=0.95`, `top_k=40`, `seed=1234`, and `max_tokens=12`
  returned HTTP 200 for all four requests. Candidate wall time was `7.663571s`
  for `111` prompt tokens and `48` completion tokens; rollback on the same
  prompt/request shape was `8.525863s`.

Verdict:
- Keep. This is another Vulkan-only measurable win and improves the current
  sampled continuous-batch shape below the A023-only result.
- CUDA and Metal source paths remain unchanged.

### 2026-05-09 A025: Accept Explicit Chat-Template Kwargs for Visible Output Correctness

Issue:
- Follow-up correctness check found that short Qwen3.5 chat requests could
  return `choices[0].message.content == ""`. Full JSON inspection showed the
  generated tokens were coherent and were being placed in `reasoning_content`
  because the Qwen3.5 template prefills an open `<think>\n` block by default.
- A prompt-text `/no_think` convention is not a safe API control: a user can
  legitimately talk about that literal string. The fix should expose the real
  template variable path instead of sniffing message content.

Implementation:
- Added `ChatTemplateOptions` to `kiln-core` and forward arbitrary
  `template_kwargs` as top-level Jinja context variables, while reserving
  Kiln-owned variables (`messages`, `tools`, `tool_choice`,
  `add_generation_prompt`, `bos_token`, `eos_token`).
- Added `chat_template_kwargs` to `/v1/chat/completions` and
  `/v1/completions/batch`. Example:
  `{"chat_template_kwargs":{"enable_thinking":false}}`.
- Included `chat_template_kwargs` in rendered-prompt, deterministic chat, chat
  choices, and batch cache keys so default-thinking and no-thinking prompts
  cannot share stale cached responses.
- Threaded the field through synthesized per-choice and per-batch chat
  requests.

Evidence:
- `cargo check -p kiln-server` passed.
- `cargo test -p kiln-core qwen35_4b_chat_template_can_disable_thinking`
  passed.
- `cargo test -p kiln-server chat_template_kwargs` passed, including chat and
  batch cache-key coverage.
- `cargo test -p kiln-server qwen35_no_think_text_is_not_a_control_flag`
  passed.
- Release Vulkan build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- `git diff --check` passed.
- Rebuilt Vulkan server (`KILN_BATCHING_ENGINE=1`, Qwen3.5-4B) confirmed:
  default direct chat at `max_tokens=12` still returns empty visible content
  with coherent `reasoning_content` starting `Thinking Process...`; explicit
  `chat_template_kwargs: {"enable_thinking": false}` returns
  `content == "blue"`, `reasoning_content == null`, and `finish_reason == stop`
  in `3` completion tokens; a prompt merely mentioning `/no_think` remains on
  the default reasoning path.
- Batch endpoint with a unique uncached prompt and
  `chat_template_kwargs: {"enable_thinking": false}` returned two completions
  with `text == "blue"`, `finish_reason == stop`, and no reasoning content.
- CUDA and Metal feature checks were attempted on the shared server crate but
  remain environment-blocked on this Linux host before reaching project code:
  CUDA failed at `cudarc` because `nvcc` is absent; Metal failed because
  `objc2` requires an Apple target.

Verdict:
- Keep. The observed empty `content` was a chat-template/output-channel issue,
  not Vulkan compute garbage. The explicit template kwargs API fixes visible
  output correctness for callers that want no-thinking Qwen behavior without
  treating user prompt text as control metadata.
- This changes shared tokenizer/server request wiring only; CUDA, Metal, and
  Vulkan kernel code are untouched.

### 2026-05-09 A026: Pack Full-Attention QKV BF16 Weights for Vulkan Decode

Issue:
- A023 and A024 showed that keeping BF16 weights packed on device and expanding
  in-shader is faster than pre-expanding BF16 weights into cached F32 buffers
  for high-traffic Vulkan decode projections.
- The fused full-attention QKV decode path still used cached F32 weight buffers,
  so every BF16 Q/K/V projection paid roughly 2x the weight bandwidth versus a
  packed path.

Implementation:
- Added `full_attn_qkv_decode_bf16w.comp`, mirroring the existing fused
  full-attention QKV decode kernel while reading packed BF16 weight buffers and
  converting with `uintBitsToFloat(bf16_bits << 16)` in the shader.
- Registered and prewarmed the new Vulkan shader/pipeline.
- Added `dispatch_full_attn_qkv_decode_cached_bf16_weights` beside the existing
  cached F32 dispatch.
- Added a Vulkan backend gate that uses packed BF16 Q/K/V weights only when all
  three projection weights are BF16; mixed or F32 groups stay on the existing
  cached F32 path.
- Reused the existing packed-BF16 weight cache and added rollback env
  `KILN_DISABLE_VULKAN_BF16_PACKED_FULL_ATTN_QKV_WEIGHTS=1`.
- Added CPU-reference parity coverage for the packed-BF16 QKV shader.

Evidence:
- `rustfmt --edition 2024` on the touched Vulkan files passed.
- `cargo check -p kiln-vulkan-kernel` passed.
- `cargo check -p kiln-model --features vulkan` passed; warnings were existing
  unused-code warnings.
- Full Vulkan parity passed:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
  returned `27 passed`, including the new packed-BF16 full-attention QKV test.
- Release Vulkan build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- `git diff --check` passed.
- Serial paged latency on `main` with Qwen3.5-4B stayed coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]` and measured `398.8ms` prefill,
  `111.5ms` mean ITL.
- Same-binary rollback with
  `KILN_DISABLE_VULKAN_BF16_PACKED_FULL_ATTN_QKV_WEIGHTS=1` produced the same
  token IDs and measured `621.6ms` prefill, `118.4ms` mean ITL.
- Earlier same-binary A/B before the merge-to-main transition also favored the
  candidate: candidate `111.8ms` / `113.3ms` mean ITL versus rollback
  `115.8ms` / `120.9ms`, all with the same token IDs.
- Main-branch Vulkan server smoke (`KILN_BATCHING_ENGINE=1`, Qwen3.5-4B) with
  `chat_template_kwargs: {"enable_thinking": false}` returned non-empty visible
  text:
  - direct chat: `content == "Blue"`, `3` completion tokens, `1.218s` wall;
  - explicit `/v1/completions/batch`: four `text == "Blue"` completions,
    `12` total completion tokens, `5.796s` wall;
  - four concurrent `/v1/chat/completions` requests through the live batcher:
    all returned `content == "1 2 3 4 5 6 7 8 "`, no reasoning content,
    `16` completion tokens each, `9.611s` total wall.

Verdict:
- Keep. This is a Vulkan-only serial decode win with matching token IDs,
  focused shader parity coverage, and successful direct/batch/concurrent
  visible-output smokes.
- CUDA and Metal code paths are untouched.

### 2026-05-09 A027: Pack Single-Row MLP Decode BF16 Weights for Vulkan

Issue:
- After A023/A024/A026, the fused MLP decode remained the largest remaining
  BF16 projection path still reading cached F32 weights. The serial decode path
  performs gate, up, and down projections for every MLP layer, so packed BF16
  weight bandwidth should matter if the shader conversion cost stays low.

Implementation:
- Added `mlp_gate_up_decode_bf16w.comp` for the single-row SwiGLU gate/up
  kernel, reading packed BF16 gate/up weights and expanding with
  `uintBitsToFloat(bf16_bits << 16)`.
- Reused `linear_decode_bf16w.comp` for the down projection inside the existing
  fused MLP two-kernel dispatch, keeping the hidden activation resident on the
  Vulkan device.
- Added `dispatch_mlp_decode_cached_bf16_weights` and a backend gate that uses
  the packed path only when the flattened row count is one and gate/up/down
  weights are all BF16.
- Added rollback env
  `KILN_DISABLE_VULKAN_BF16_PACKED_MLP_DECODE_WEIGHTS=1`.
- Kept F32 MLP weights prewarmed as well as packed weights. This is intentional:
  prefill and multi-row batch MLP still use the existing F32 kernels, and an
  earlier version that prewarmed only packed MLP weights regressed cold prefill
  to `3298.7ms` despite improving decode to `96.7ms` mean ITL.
- Added CPU-reference parity coverage for the packed-BF16 fused MLP dispatch.

Evidence:
- `cargo check -p kiln-vulkan-kernel` passed.
- `cargo check -p kiln-model --features vulkan` passed; warnings were existing
  unused-code warnings.
- Focused packed MLP parity passed:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode_bf16_packed_weights_match_cpu_reference -- --nocapture`.
- Full Vulkan parity passed:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
  returned `28 passed`.
- Release Vulkan build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- `git diff --check` passed.
- Serial paged latency on `main` with Qwen3.5-4B stayed coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]`.
  - Candidate after F32+packed MLP prewarm: `390.6ms` prefill / `102.1ms`
    mean ITL, then `357.7ms` prefill / `95.0ms` mean ITL.
  - Same-binary rollback with
    `KILN_DISABLE_VULKAN_BF16_PACKED_MLP_DECODE_WEIGHTS=1`: `415.4ms`
    prefill / `112.4ms` mean ITL, then `358.0ms` prefill / `113.6ms`
    mean ITL.
- Main-branch Vulkan server smoke (`KILN_BATCHING_ENGINE=1`, Qwen3.5-4B) with
  `chat_template_kwargs: {"enable_thinking": false}` returned non-empty visible
  text:
  - direct chat: `content == "Blue"`, `3` completion tokens, `1.105s` wall;
  - explicit `/v1/completions/batch`: four `text == "Blue"` completions,
    `12` total completion tokens, `6.041s` wall;
  - four concurrent `/v1/chat/completions` requests through the live batcher:
    all returned `content == "1 2 3 4 5 6 7 8 "`, no reasoning content,
    `16` completion tokens each, `9.970s` total wall.

Verdict:
- Keep. This is a Vulkan-only serial decode win with matching token IDs, no
  cold-prefill regression after keeping F32 MLP prewarm, and successful
  direct/batch/concurrent visible-output smokes.
- CUDA and Metal code paths are untouched.

### 2026-05-09 A028: Extend Packed-BF16 MLP Decode to Small Multi-Row Batches

Issue:
- A027 kept packed-BF16 MLP decode to flattened row count `1` to avoid
  disturbing prefill and continuous-batch paths. Four-request live batching
  still ran MLP through cached F32 weights even though the row count stays below
  the existing row-pair threshold.

Implementation:
- Added `mlp_gate_up_decode_batched_bf16w.comp`, mirroring the current
  batched gate/up shader while reading packed BF16 gate/up weights.
- Extended `dispatch_mlp_decode_cached_bf16_weights` to use:
  - `mlp_gate_up_decode_bf16w.comp` + `linear_decode_bf16w.comp` for row count
    `1`;
  - `mlp_gate_up_decode_batched_bf16w.comp` +
    `linear_decode_batched_bf16w.comp` for row counts `2..7`.
- Left row-pair MLP batches (`row_count >= 8`) on the current F32 row-pair
  shaders. This intentionally avoids changing larger prefill behavior before a
  dedicated packed-BF16 row-pair shader has evidence.
- Reused A027's rollback env:
  `KILN_DISABLE_VULKAN_BF16_PACKED_MLP_DECODE_WEIGHTS=1`.
- Expanded packed-BF16 MLP parity coverage to batch `1` and batch `3`.

Evidence:
- `rustfmt --edition 2024` on the touched Vulkan files passed.
- `cargo check -p kiln-vulkan-kernel` passed.
- `cargo check -p kiln-model --features vulkan` passed; warnings were existing
  unused-code warnings.
- Focused packed MLP parity passed for batch `1` and batch `3`.
- Release Vulkan build passed:
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Full Vulkan parity passed:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
  returned `28 passed`.
- `git diff --check` passed.
- Serial paged latency remained coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]`; the extended candidate measured
  `419.3ms` prefill / `98.3ms` mean ITL.
- Four-prompt explicit batch and four-concurrent-chat A/B with
  `chat_template_kwargs: {"enable_thinking": false}` returned non-empty visible
  text on both candidate and rollback:
  - First pair: candidate explicit batch `4.390s` vs rollback `4.330s`
    (slightly worse); candidate concurrent chat `9.493s` vs rollback `9.818s`
    (better).
  - Second pair with fresh prompts: candidate explicit batch `4.320s` vs
    rollback `4.544s`; candidate concurrent chat `10.092s` vs rollback
    `10.378s`.

Verdict:
- Keep. The first explicit-batch sample was mixed, but the repeated pair and
  both concurrent-chat samples favored the packed small-batch path, correctness
  stayed clean, and row-pair prefill/batch behavior is still left untouched.
- CUDA and Metal code paths are untouched.

### 2026-05-09 A029: Reject Packed-BF16 MLP Row-Pair Prefill Path

Issue:
- A028 deliberately left row-pair MLP batches (`row_count >= 8`) on the
  existing F32 shaders. A row-pair packed-BF16 variant could reduce weight
  bandwidth for prompt prefill, but it also changes a path where previous
  row-pair geometry was already tuned.

Implementation Tried:
- Temporarily added packed-BF16 row-pair variants for both pieces of fused MLP:
  `mlp_gate_up_decode_batched_rows2_bf16w.comp` and
  `linear_decode_batched_rows2_bf16w.comp`.
- Temporarily extended the packed MLP dispatcher to use those shaders when the
  existing row-pair threshold selected row-pair geometry.
- Added a row-pair-specific rollback env during the trial:
  `KILN_DISABLE_VULKAN_BF16_PACKED_MLP_ROWPAIR_WEIGHTS=1`.
- Extended focused parity to batch `9`, which exercises the row-pair path.

Evidence:
- The temporary row-pair code passed `cargo check -p kiln-vulkan-kernel`,
  `cargo check -p kiln-model --features vulkan`, focused packed MLP parity
  including batch `9`, release Vulkan build, and `git diff --check`.
- Candidate serial paged smoke stayed coherent with token IDs
  `[2838,6587,310,5227,1024,75119,220]` and measured `384.1ms` prefill /
  `96.5ms` mean ITL.
- Same-binary candidate after adding the row-pair-specific rollback env measured
  `390.2ms` prefill / `98.0ms` mean ITL.
- Same-binary row-pair rollback with
  `KILN_DISABLE_VULKAN_BF16_PACKED_MLP_ROWPAIR_WEIGHTS=1` produced the same
  token IDs and measured `383.4ms` prefill / `96.5ms` mean ITL.

Verdict:
- Rejected and removed. The packed-BF16 row-pair path did not beat the current
  F32 row-pair shaders on the prompt-prefill shape, so A028 remains scoped to
  row counts below the row-pair threshold.
- Do not retry row-pair BF16 by direct shader mirroring; it needs a different
  geometry or broader prefill fusion to justify another pass.

### 2026-05-09 A030: Enable Mixed-Sequence Live Decode Batching on Vulkan

Issue:
- PR #1002 enabled mixed-sequence live decode batching by default for Metal, but
  Vulkan still defaulted to same-position-only grouping because Candle exposes
  Vulkan runs through `Device::Cpu`.
- The Vulkan backend already has a dyn-seqlen paged-attention batch path, and
  A011 proved that path correct for non-uniform sequence lengths. Leaving the
  live greedy batcher in same-position mode fragmented streaming workloads into
  one-row batches.

Implementation:
- Added `DecodeBatcherConfig::from_env_for_backend(device, backend_name)` and a
  small default-policy helper.
- Exposed `ModelRunner::backend_name()` and passed the runtime backend name from
  server state when spawning the live decode batcher.
- Default mixed-sequence grouping is now enabled for Metal devices and for
  runtime backend `"vulkan"`. CPU and CUDA remain off by default unless the
  caller explicitly sets `KILN_DECODE_BATCH_MIXED_SEQ=1`.

Evidence:
- Pre-change environment A/B, no batching actor, streaming four concurrent
  prompts with different prompt lengths and `chat_template_kwargs:
  {"enable_thinking": false}`:
  - `KILN_DECODE_BATCH_MIXED_SEQ=1`,
    `KILN_DECODE_BATCH_WAIT_US=50000`: `12.444s`, `69` submitted rows,
    `20` batches, max observed batch `4`.
  - `KILN_DECODE_BATCH_MIXED_SEQ=0`,
    `KILN_DECODE_BATCH_WAIT_US=50000`: `17.034s`, `69` submitted rows,
    `69` batches, max observed batch `1`.
  - With default wait (`0us`), enabled measured `12.921s`, `69` submitted
    rows, `38` batches, max observed batch `3`; disabled measured `13.470s`,
    `69` submitted rows, `69` batches, max observed batch `1`.
- Post-change no-env smoke (`KILN_BATCHING_ENGINE` unset,
  `KILN_DECODE_BATCH_MIXED_SEQ` unset) logged
  `backend="vulkan" mixed_seq_lens=true` at startup.
- The same no-env streaming smoke returned HTTP 200 for all four streams,
  visible non-empty content, empty `reasoning_content`, and finished with
  `13.365s` wall time, `68` submitted rows, `38` batches, and max observed
  batch `3`.
- The non-streaming route and `KILN_BATCHING_ENGINE=1` actor route bypass this
  live decode batcher, so their zero decode-batcher counters were not used as
  evidence for this change.

Validation:
- `rustfmt --edition 2024 crates/kiln-model/src/generate.rs
  crates/kiln-server/src/state.rs`
- `cargo test -p kiln-model
  test_decode_batcher_default_mixed_seq_lens_backend_policy --lib`
- `cargo check -p kiln-model --features vulkan`
- `cargo check -p kiln-server --features vulkan`
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`

Verdict:
- Keep. This is a backend-aware default change, not a new kernel path. It turns
  on the already-tested Vulkan dyn-seqlen batch behavior for live streaming
  requests, improves the measured default no-wait streaming batch shape, and
  preserves CPU/CUDA defaults plus Metal's existing default.
- Rollback/override: set `KILN_DECODE_BATCH_MIXED_SEQ=0`.

## Current Open Work

- Improve the new Vulkan dyn-seqlen paged attention backend by eliminating CPU
  K/V compaction and per-call compact K/V uploads.
- Do not extend packed-BF16 MLP to row-pair (`row_count >= 8`) by direct shader
  mirroring; A029 measured that as slower than the current F32 row-pair path.
- Improve LoRA throughput without adding per-projection dispatch fanout. A019
  shows standalone low-rank delta kernels are correct but slower.
- Continue profiling the remaining serial decode hotspots: GDN recurrent/gated
  norm, paged-attention K/V movement, and any residual fused MLP/down-projection
  time after A027.
- For GDN batch fusion, do not reuse the generic
  `gdn_decode_gates_recurrent_rmsnorm` hook without first changing its
  residency/data-movement behavior; A013 measured it as a large regression.
- For sampled continuous batch routing, do not broadly reroute through
  `model_forward_paged_decode_contiguous_batch` logits; A014 measured it as a
  regression against the current generic path.
- Keep updating this log and
  `docs/audits/vulkan-strix-halo-2026-05-09-gpu-decode-shortlog.md` for every
  accepted or rejected optimization experiment.
