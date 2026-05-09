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

## Current Open Work

- Improve the new Vulkan dyn-seqlen paged attention backend by eliminating CPU
  K/V compaction and per-call compact K/V uploads.
- Continue profiling the remaining serial decode hotspots: fused MLP decode,
  GDN `in_proj`, GDN recurrent/gated norm, and full-attention QKV projection.
- For GDN batch fusion, do not reuse the generic
  `gdn_decode_gates_recurrent_rmsnorm` hook without first changing its
  residency/data-movement behavior; A013 measured it as a large regression.
- For sampled continuous batch routing, do not broadly reroute through
  `model_forward_paged_decode_contiguous_batch` logits; A014 measured it as a
  regression against the current generic path.
- Keep updating this log and
  `docs/audits/vulkan-strix-halo-2026-05-09-gpu-decode-shortlog.md` for every
  accepted or rejected optimization experiment.
