# Vulkan Strix Halo Optimization Log

## Objective

Make Vulkan a first-class Kiln inference backend and push Qwen3.5-4B performance on the local AMD Strix Halo machine as far as current architecture allows. Track both latency and throughput experiments. Treat passing tests as correctness evidence only; speed claims require direct benchmark output.

## Test Host

- Date: 2026-05-03
- GPU: Radeon 8060S Graphics (RADV STRIX_HALO)
- Detected VRAM: 98304 MiB via Linux DRM sysfs
- Model: local `Qwen3.5-4B/`
- Benchmark binary: `./target/release/kiln-bench --features vulkan`
- Main latency harness: `--latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`

## Running Verdict

Vulkan is functional and integrated, and the best observed no-env default single-user decode result remains E084's 129.1ms mean inter-token latency after scoped resident split GDN recurrent state, GDN input-projection single-submit, and 64x4 MLP gate/up decode workgroups. E087 keeps a retiled full-attention QKV fusion as the current default because it wins same-binary rollback checks at 130.4-130.7ms versus 132.3-133.7ms. The old 318-367ms bs=1 source anchors are superseded. Native batch is default again for eligible greedy batch endpoint traffic, but broader bs>1 throughput still needs true multi-sequence paged attention and scheduling. The largest structural blocker remains that candle-core has no native Vulkan tensor storage here: many tensors still cross CPU memory between Vulkan kernels, so a large share of time is transfer, tensor materialization, and unfused host-side work rather than GPU arithmetic.

## Experiments

### E001: First-Class Build And Runtime Integration

Change:
- Added Vulkan feature paths alongside CUDA and Metal in build, CI, release, docs, desktop, and benchmark selection.
- Added Linux DRM sysfs VRAM detection for AMD/Intel Vulkan systems.

Evidence:
- Vulkan builds with `cargo build --release --features vulkan --bin kiln-bench`.
- Benchmark reports backend `"vulkan"` and GPU `"Radeon 8060S Graphics (RADV STRIX_HALO)"`.

Verdict: keep.

### E002: Vulkan Device Lifetime

Change:
- Made `VulkanDevice` own the `ash::Entry` lifetime.

Reasoning:
- Avoids dangling Vulkan loader/entry lifetime during teardown.

Verdict: keep.

### E003: F32-Compatible Vulkan GDN Paths

Change:
- Relaxed GDN gates, gated RMSNorm, and recurrent step to support F32 where the Vulkan-selected CPU tensor path produces F32.

Evidence:
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed after F32 parity cases were added.

Verdict: keep.

### E004: Cached Vulkan Weight Buffers

Change:
- Added a `TensorId` keyed Vulkan weight cache in `VulkanBackend`.
- Uploads immutable projection weights once as F32 device-local Vulkan buffers.

Reasoning:
- Single-token decode repeatedly uses the same transposed weights; repeated host-to-device weight upload is pure waste.

Verdict: keep.

### E005: GDN Input Projection Fusion

Change:
- Added `gdn_in_proj_decode.comp` and backend dispatch for single-token GDN input projections.
- Collapses qkv, z, a, and b projections into one Vulkan dispatch.

Evidence:
- `gdn_in_proj_decode_matches_cpu_reference` passed in the Vulkan parity test suite.

Verdict: keep.

### E006: Generic Vulkan Single-Token Linear Decode

Change:
- Added `linear_decode.comp`, cached dispatch, and forward routing for q/k/v/o, MLP gate/up/down, GDN out projection, and later LM head.

Evidence:
- `linear_decode_matches_cpu_reference` passed.
- Earlier baseline was roughly 2.3s/token; generic cached linear reduced warmed decode to roughly 1.2s/token before weight prewarm.

Verdict: keep.

### E007: Reusable Transient Command Pool

Change:
- Reused a transient command pool for hot upload/readback helpers instead of creating and destroying a command pool per dispatch.

Evidence:
- Vulkan parity suite stayed green.
- Best warmed latency before later prewarm was roughly 1.2s/token.

Verdict: keep.

### E008: Fused MLP Decode

Change:
- Added `mlp_gate_up_decode.comp` and full fused MLP decode dispatch.

Evidence:
- `mlp_gate_up_decode_matches_cpu_reference` and `mlp_decode_matches_cpu_reference` passed.
- Benchmarks were neutral or worse than generic cached GEMV.

Verdict:
- Keep implementation opt-in via `KILN_ENABLE_VULKAN_MLP_GATE_UP=1` and `KILN_ENABLE_VULKAN_MLP_DECODE=1`.
- Do not promote until tiled/tuned or tensor residency changes.

### E009: Fused GDN Decode Gates + Recurrent + RMSNorm

Change:
- Added `gdn_decode_gates_recurrent_rmsnorm.comp`.

Evidence:
- `gdn_decode_gates_recurrent_rmsnorm_matches_f32_cpu_reference` passed.
- Benchmark worsened or failed to beat generic split path.

Verdict:
- Keep opt-in via `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED=1`.

### E010: Fused Full-Attention QKV

Change:
- Added `full_attn_qkv_decode.comp`.

Evidence:
- `full_attn_qkv_decode_matches_cpu_reference` passed.
- Combined opt-in benchmark after weight prewarm:
  - Command: `KILN_BENCH_LOG_ITL=1 KILN_ENABLE_VULKAN_MLP_DECODE=1 KILN_ENABLE_VULKAN_GDN_DECODE_FUSED=1 KILN_ENABLE_VULKAN_FULL_ATTN_QKV=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prefill: 1919.1ms
  - Decode steps: 898.5, 802.8, 925.3, 806.0, 903.0, 810.0ms
  - Mean ITL: 857.6ms

Verdict:
- Keep opt-in via `KILN_ENABLE_VULKAN_FULL_ATTN_QKV=1`.
- Do not promote; default mean 825.5ms is better.

### E011: Decode Weight Prewarm And LM Head Linear Routing

Change:
- Added `BackendRuntime::prewarm_decode_weights`.
- Vulkan prewarms all base decode transposed projection weights and tied LM-head table into its F32 weight cache.
- Bench and `ModelRunner` call prewarm once after backend creation.
- Decoupled generic `linear_decode` from the experimental full-attn QKV env gate.
- Routed one-token LM head through backend `linear_decode` where supported.

Evidence:
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 10 tests.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- Default benchmark:
  - Command: `KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prewarm: 249 weights, 16040 MiB F32 cache, 30430ms
  - Prefill: 2371.0ms (4 tok/s)
  - Decode steps: 833.8, 819.7, 827.2, 839.3, 814.0, 818.8ms
  - Mean ITL: 825.5ms
  - P50 ITL: 827.2ms
  - Decode throughput: 1.2 tok/s

Verdict:
- Keep. This removes the prior first-token lazy upload spike.
- Not sufficient; warmed decode remains slow.

### E012: Correct Vulkan Causal Conv1d State Handling

Change:
- Updated `causal_conv1d.comp` to read `conv_state` for the left side of the causal window instead of treating missing taps as zero.
- Updated `causal_conv1d_state_advance.comp` to preserve chronological state order, matching the portable fallback.
- Fixed the Rust dispatch binding count for the state-aware output shader.
- Embedded `causal_conv1d_state_advance.comp` at build time and added both conv pipelines to Vulkan prewarm.
- Added low-level Vulkan parity tests for single-token update and prefill, including the `seq_len < K-1` state-shift case.

Evidence:
- Initial parity failed and showed the output shader was still using only three bindings.
- After binding fix: `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 12 tests.
- `cargo check -p kiln-server --features vulkan` passed.

Verdict:
- Keep the correctness fix.
- Do not enable by default until latency beats the host fallback.

### E013: A/B Corrected Vulkan Conv1d Opt-In

Change:
- Enabled corrected conv1d via `KILN_ENABLE_VULKAN_FUSED_CONV1D=1`.

Evidence:
- Command: `KILN_BENCH_LOG_ITL=1 KILN_ENABLE_VULKAN_FUSED_CONV1D=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
- Before command-pool cleanup:
  - Prefill: 1931.9ms
  - Decode steps: 809.9, 832.3, 845.0, 842.6, 846.5, 835.9ms
  - Mean ITL: 835.4ms
- Default rerun with same binary family:
  - Prefill: 2357.3ms
  - Decode steps: 815.6, 829.0, 825.8, 834.4, 812.0, 809.5ms
  - Mean ITL: 821.1ms

Verdict:
- Corrected conv1d is not a decode latency win.
- Keep opt-in.

### E014: Remove Conv1d Command-Pool Churn

Change:
- Converted conv1d input/state uploads and output/state readbacks from per-call command-pool helpers to the reusable transient command pool.

Evidence:
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 12 tests.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- Opt-in benchmark:
  - Command: `KILN_BENCH_LOG_ITL=1 KILN_ENABLE_VULKAN_FUSED_CONV1D=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prefill: 2019.2ms
  - Decode steps: 803.4, 850.1, 838.6, 847.2, 836.5, 847.0ms
  - Mean ITL: 837.1ms

Verdict:
- Not a real decode win. Command-pool cleanup is still correct and may help this opt-in path, but the two extra Vulkan dispatches/readbacks do not beat the CPU fallback in the measured decode loop.
- Keep conv1d opt-in.

### E015: Vulkan Greedy LM-Head Argmax

Change:
- Added a two-dispatch Vulkan argmax path for single-token transposed linear projection:
  - `linear_decode_argmax_blocks.comp` computes 16 logits per workgroup and writes per-block max pairs.
  - `linear_decode_argmax_reduce.comp` reduces block max pairs to one token id.
- Added `BackendRuntime::linear_decode_argmax` and `supports_linear_decode_argmax`.
- Implemented Vulkan `linear_decode_argmax` using cached F32 weights.
- Routed paged greedy prefill/decode through `model_forward_paged_*_greedy` when the backend supports argmax, not only on Metal.

Reasoning:
- Greedy decode does not need a host `[1, 1, vocab]` logits tensor. Keeping logits on Vulkan removes a readback/materialization step and CPU argmax.

Evidence:
- `linear_decode_argmax_matches_cpu_reference` passed with a non-multiple-of-16 output size.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 13 tests.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- Default benchmark:
  - Command: `KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prefill: 1883.7ms
  - Decode steps: 774.8, 790.8, 800.2, 769.3, 791.3, 796.2ms
  - Mean ITL: 787.1ms
  - P50 ITL: 791.3ms
  - Decode throughput: 1.3 tok/s

Verdict:
- Keep and use by default for greedy paged Vulkan.
- This is the current best default latency result.

### E016: Conv1d Opt-In Retest After Vulkan Argmax

Change:
- Retested corrected conv1d with Vulkan argmax now active.

Evidence:
- Command: `KILN_BENCH_LOG_ITL=1 KILN_ENABLE_VULKAN_FUSED_CONV1D=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
- Prefill: 1858.3ms
- Decode steps: 782.1, 801.8, 810.7, 809.5, 822.0, 794.7ms
- Mean ITL: 803.4ms

Verdict:
- Conv1d may slightly improve prefill in this short harness, but it remains worse for decode latency than the default 787.1ms result.
- Keep opt-in.

### E017: Argmax Workgroup Shape 8x32

Change:
- Changed `linear_decode_argmax_blocks.comp` from 16 output columns x 16 reduction lanes to 8 output columns x 32 reduction lanes.

Reasoning:
- More reduction lanes per output might reduce hidden-dimension loop time on RDNA, at the cost of doubling output-block workgroups.

Evidence:
- Focused parity: `cargo test -p kiln-vulkan-kernel --test gdn_parity linear_decode_argmax_matches_cpu_reference -- --nocapture` passed.
- Benchmark:
  - Command: `KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prefill: 2141.9ms
  - Decode steps: 786.6, 785.2, 810.1, 808.2, 784.8, 806.2ms
  - Mean ITL: 796.8ms

Verdict:
- Worse than the 16x16 argmax baseline of 787.1ms mean ITL.
- Reverted to 16x16.

### E018: Production Paged Generation Vulkan Argmax Wiring

Change:
- Added `ModelRunner::supports_paged_greedy_argmax`, true for Metal and any backend with `supports_linear_decode_argmax`.
- Replaced Metal-only greedy gates in production paged prefill/decode paths with that helper:
  - prefix-cache paged prefill
  - prefix-cache paged decode
  - non-shared paged prefill
  - non-shared paged decode
  - streaming paged locked decode loop

Reasoning:
- The benchmark path already used Vulkan greedy LM-head argmax, but real server generation still had Metal-only gates. That meant greedy Vulkan server requests could fall back to full logits materialization even though the backend had a faster argmax path.

Evidence:
- `cargo fmt --all` passed.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 13 tests.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- Post-wiring default benchmark:
  - Command: `KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prewarm: 249 weights, 16040 MiB F32 cache, 23140ms
  - Prefill: 2755.9ms
  - Decode steps: 785.1, 783.1, 766.1, 782.5, 793.3, 796.9ms
  - Mean ITL: 784.5ms
  - P50 ITL: 785.1ms
  - P99 ITL: 796.9ms
  - Decode throughput: 1.3 tok/s

Verdict:
- Keep. This closes the production/bench routing mismatch.
- The prefill number moved worse on this run, but decode ITL is slightly better than the prior 787.1ms anchor and within short-run variance.

### E019: HTTP Server bs=1 And bs=4 Throughput

Change:
- No code change. Measured the release server after E018.

Server command:
- `KILN_MODEL_PATH=Qwen3.5-4B KILN_PORT=18420 KILN_REQUEST_TIMEOUT_SECS=600 KILN_PREFIX_CACHE_ENABLED=0 ./target/release/kiln serve`

Workload:
- OpenAI-compatible chat completions with `temperature=0.0`, `max_tokens=4`, short unique prompts, served model `qwen3.5-4b-kiln`.
- Prefix cache disabled to avoid shared-prefix cache effects in raw bs>1 numbers.

Evidence:
- Warmup:
  - 4 completion tokens in 4.498s.
- bs=1, three measured runs:
  - 4 tokens in 4.504s, 0.888 completion tok/s end-to-end
  - 4 tokens in 4.501s, 0.889 completion tok/s end-to-end
  - 4 tokens in 4.482s, 0.893 completion tok/s end-to-end
  - Mean latency: 4.496s for 4 completion tokens.
- bs=4 as four concurrent `/v1/chat/completions` requests:
  - Wall time: 18.309s
  - Completion tokens: 16
  - Aggregate completion throughput: 0.874 tok/s
  - Per-request latencies: 10.980s, 13.418s, 15.882s, 18.300s
- bs=4 through `/v1/completions/batch`:
  - Wall time: 18.248s
  - Completion tokens: 16
  - Aggregate completion throughput: 0.877 tok/s

Verdict:
- The HTTP server is using Vulkan successfully, but bs>1 does not improve throughput. This is expected from the current architecture: the batch endpoint fans out per-prompt tasks and the shared paged-cache path interleaves them, but decode work still runs as independent single-sequence forward passes with serialized access to shared GPU/cache state.
- Do not market this as batched Vulkan inference. The next performance step for bs>1 is native multi-sequence Vulkan decode, not another single-token scalar optimization.

### E020: Opt-In Decode Profiler

Change:
- Added `KILN_VULKAN_DECODE_PROFILE=1` / `KILN_PROFILE_DECODE=1` instrumentation around paged decode operation families.
- The profiler is active only for single-token decode (`seq_len == 1 && start_pos > 0`) and reports aggregate bucket totals plus the largest per-layer events.

Evidence:
- `cargo check -p kiln-server --features vulkan` passed.
- Initial profile command:
  - `KILN_VULKAN_DECODE_PROFILE=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 2 --skip-training`
- Initial profile result:
  - Prefill: 2168.9ms
  - Mean ITL: 807.6ms
  - Step 1 major buckets: `gdn.in_proj=195.7ms/24`, `gdn.recurrent=199.4ms/72`, `mlp.down=102.5ms/32`, `mlp.gate=69.0ms/32`, `mlp.up=73.3ms/32`, `layer.full=142.0ms/8`, `gdn.gated_norm=34.5ms/24`, `gdn.out_proj=25.0ms/24`, `lm_head.argmax=16.2ms/1`
  - Step 2 major buckets: `gdn.in_proj=198.2ms/24`, `gdn.recurrent=200.1ms/72`, `mlp.down=101.2ms/32`, `mlp.gate=70.4ms/32`, `mlp.up=74.1ms/32`, `layer.full=133.4ms/8`, `lm_head.argmax=13.2ms/1`

Verdict:
- Keep instrumentation.
- The first profile exposed two useful facts:
  - The largest remaining decode time is split across GDN input projection/recurrent state work and MLP/full-attention projections.
  - The `gdn.recurrent` count was polluted by prefill-only probe calls in single-token decode, so the profiler itself pointed to a small cleanup.

### E021: GDN Recurrent Transfer Command-Pool Cleanup

Change:
- Updated `dispatch_gdn_recurrent_step` to use `upload_data_with_command_pool` and `read_back_with_command_pool` with `vk_device.transient_command_pool()` instead of creating/destroying command pools through the older upload/readback helpers.

Evidence:
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 13 tests.
- `cargo check -p kiln-server --features vulkan` passed.
- Release anchors after the cleanup:
  - First six-token run: prefill 2027.6ms, mean ITL 801.8ms.
  - Profile run: prefill 1867.3ms, mean ITL 774.4ms; `gdn.recurrent` dropped to `176.6ms/72` and `190.3ms/72`.
  - Second six-token run: prefill 1869.2ms, mean ITL 791.6ms.

Verdict:
- Keep as cleanup: it removes obvious command-pool churn in a hot recurrent-step path.
- Do not update the best anchor from this change alone; the six-token runs did not beat E018's 784.5ms anchor.

### E022: Rejected Batched Transfer Helper

Change:
- Tried adding `upload_many_with_command_pool` and `read_back_many_with_command_pool` to batch the recurrent-step q/k/v/beta/g/state transfers into fewer command submissions.

Evidence:
- During the experiment, parity and integration stayed green:
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 13 tests.
  - `cargo check -p kiln-server --features vulkan` passed.
- Performance got worse:
  - Six-token anchor: prefill 1899.1ms, mean ITL 792.0ms.
  - Profile run: mean ITL 799.7ms; `gdn.recurrent=187.0ms/72` and `198.2ms/72`.

Verdict:
- Reverted the helper usage and removed the unused helper functions.
- The batched helper reduced submission count but did not improve the measured decode loop on this stack.

### E023: Skip Single-Token GDN Prefill-Only Recurrent Probes

Change:
- In `gated_deltanet_forward_decode_if`, when `seq_len == 1`, call `gdn_chunkwise_recurrence` directly after `gdn.recur_prep`.
- This skips `gdn_recurrent_prefill_head_last` and `gdn_chunkwise_recurrence_head_last_full_chunks`, which are prefill/full-chunk paths that immediately return `None` for single-token decode.

Evidence:
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 13 tests.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- Focused profile command:
  - `KILN_VULKAN_DECODE_PROFILE=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 2 --skip-training`
- Focused profile result:
  - Prefill: 2063.5ms
  - Mean ITL: 777.1ms
  - `gdn.recurrent` is now counted as 24 real recurrent calls instead of 72 total probe+real calls.
  - Step 1 major buckets: `gdn.in_proj=177.5ms/24`, `gdn.recurrent=194.7ms/24`, `mlp.down=98.7ms/32`, `mlp.gate=69.1ms/32`, `mlp.up=73.2ms/32`, `layer.full=126.4ms/8`, `lm_head.argmax=12.8ms/1`
  - Step 2 major buckets: `gdn.in_proj=190.5ms/24`, `gdn.recurrent=189.2ms/24`, `mlp.down=99.4ms/32`, `mlp.gate=69.5ms/32`, `mlp.up=74.6ms/32`, `layer.full=132.1ms/8`, `lm_head.argmax=15.7ms/1`
- Standard anchor:
  - Command: `KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prewarm: 249 weights, 16040 MiB F32 cache, 9865ms
  - Prefill: 1992.9ms
  - Decode steps: 762.3, 784.0, 773.1, 776.7, 784.0, 781.4ms
  - Mean ITL: 776.9ms
  - P50 ITL: 781.4ms
  - P99 ITL: 784.0ms
  - Decode throughput: 1.3 tok/s

HTTP rerun:
- Server command:
  - `KILN_MODEL_PATH=Qwen3.5-4B KILN_PORT=18421 KILN_REQUEST_TIMEOUT_SECS=600 KILN_PREFIX_CACHE_ENABLED=0 ./target/release/kiln serve`
- bs=1, three measured 4-token chat runs:
  - 4 tokens in 4.585s, 0.872 completion tok/s end-to-end
  - 4 tokens in 4.549s, 0.879 completion tok/s end-to-end
  - 4 tokens in 4.538s, 0.881 completion tok/s end-to-end
  - Mean latency: 4.557s for 4 completion tokens.
- bs=4 as four concurrent `/v1/chat/completions` requests:
  - Wall time: 18.272s
  - Completion tokens: 16
  - Aggregate completion throughput: 0.876 tok/s
  - Per-request latencies: 18.264s, 13.544s, 6.766s, 15.885s
- bs=4 through `/v1/completions/batch`:
  - Wall time: 18.252s
  - Completion tokens: 16
  - Aggregate completion throughput: 0.877 tok/s

Verdict:
- Keep. This is the new best default single-user bench anchor.
- The HTTP server result did not materially improve and still has no bs>1 throughput scaling. The server bottleneck remains native batched decode, not this single-token control-flow cleanup.

### E024: MLP Gate/Up Fusion Retest

Change:
- Retested the existing `KILN_ENABLE_VULKAN_MLP_GATE_UP=1` opt-in after the latest single-token cleanup.

Evidence:
- Command: `KILN_BENCH_LOG_ITL=1 KILN_ENABLE_VULKAN_MLP_GATE_UP=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
- Prefill: 2013.8ms
- Decode steps: 879.9, 897.8, 890.1, 894.5, 894.8, 886.1ms
- Mean ITL: 890.5ms

Verdict:
- Still do not promote. The fused gate/up path is much slower than the default split projection path on this Strix Halo stack.

### E025: Tile Generic Linear Decode

Change:
- Changed `linear_decode.comp` from one output column per 256-lane workgroup to 16 output columns x 16 reduction lanes per workgroup.
- Updated `dispatch_linear_decode_cached` to dispatch `ceil(out_dim / 16)` workgroups.

Reasoning:
- Weights are stored as row-major `[hidden, out_dim]`. The old one-column kernel read `weight[h * out_dim + col]` with a stride of `out_dim` across neighboring lanes. The tiled shape reads contiguous output columns across `local_size_x`, matching the already successful LM-head argmax layout and removing 15/16ths of the projection workgroup launches.

Evidence:
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 13 tests.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- Standard anchor:
  - Prefill: 2044.3ms
  - Decode steps: 518.9, 530.1, 527.5, 533.2, 529.7, 528.4ms
  - Mean ITL: 528.0ms
- Focused profile after this change:
  - Mean ITL: 507.0ms
  - `mlp.down` fell to ~21.6ms/32.
  - `mlp.gate` fell to ~24-25ms/32.
  - `mlp.up` fell to ~25-26ms/32.
  - `layer.full` fell to ~31-32ms/8.

Verdict:
- Keep. This is the largest single shader-geometry win so far.

### E026: Tile GDN Input Projection

Change:
- Changed `gdn_in_proj_decode.comp` to the same 16 output columns x 16 reduction lanes shape.
- Updated its dispatch count to `ceil(total_out / 16)`.

Evidence:
- First attempt failed parity because `total_out` was accidentally changed in the push constants instead of the dispatch count; fixed before benchmarking.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 13 tests.
- `cargo check -p kiln-server --features vulkan` passed.
- Standard anchor after the fixed tiled shader:
  - Prefill: 1936.5ms
  - Decode steps: 356.1, 367.1, 358.1, 380.9, 370.5, 387.8ms
  - Mean ITL: 370.1ms
- Focused profile:
  - Mean ITL: 384.9ms
  - `gdn.in_proj` fell from ~165-178ms/24 after E025 to ~24-25ms/24.
  - Remaining largest bucket became GDN recurrent state work.

Verdict:
- Keep. This removes most of the GDN input projection cost without changing model behavior.

### E027: Fused GDN Decode Retest And Revert Promotion

Change:
- Retested `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED=1` after E025/E026.
- Briefly promoted it to default, then reverted the promotion after current-source no-env reruns showed the split path was more stable.

Evidence:
- Fused opt-in run 1:
  - Prefill: 2862.6ms
  - Decode steps: 359.7, 234.7, 362.4, 250.9, 370.7, 269.0ms
  - Mean ITL: 307.9ms
- Fused opt-in run 2:
  - Prefill: 2158.7ms
  - Decode steps: 356.7, 260.6, 381.9, 273.6, 386.5, 283.2ms
  - Mean ITL: 323.8ms
- Profile with fused opt-in captured only high-latency steps:
  - Mean ITL: 386.9ms
  - `gdn.fused_gates_recur_norm` was ~216-221ms/24.
- After brief default promotion, no-env default reruns regressed:
  - Mean ITL: 394.2ms, then 384.2ms.
- Split-path A/B through `KILN_DISABLE_VULKAN_GDN_DECODE_FUSED=1` on that promoted binary:
  - Prefill: 1868.2ms
  - Decode steps: 367.1, 358.0, 350.9, 359.8, 361.6, 372.6ms
  - Mean ITL: 361.7ms.
- Final source state returns fused GDN decode to opt-in only.

Verdict:
- Keep fused GDN decode opt-in only. It can produce faster transient means, but it is not stable enough to be the default on this host.

### E028: Full-Attn QKV And MLP Fusion Retests After Tiling

Change:
- Retested old projection-related opt-ins after the tiled generic projection work:
  - `KILN_ENABLE_VULKAN_FULL_ATTN_QKV=1`
  - `KILN_ENABLE_VULKAN_MLP_DECODE=1`
- Also tiled `mlp_gate_up_decode.comp` and fixed the fused MLP down-projection dispatch geometry to use `ceil(out_dim / 16)`.

Evidence:
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 13 tests, including `mlp_gate_up_decode_matches_cpu_reference` and `mlp_decode_matches_cpu_reference`.
- Full-attn QKV opt-in:
  - Prefill: 2804.5ms
  - Mean ITL: 354.7ms
  - Verdict: worse than the best split-path tiled projection result and worse prefill.
- Full MLP decode opt-in:
  - Prefill: 2642.0ms
  - Decode steps: 365.1, 366.5, 361.6, 365.1, 379.3, 370.7ms
  - Mean ITL: 368.0ms
  - Verdict: not better than split-path default; do not promote.

Verdict:
- Keep the tiled MLP shader correctness fix, but keep the MLP/full-attn fusion paths opt-in.

### E029: Host-Visible Fused GDN State Experiment

Change:
- Added opt-in `KILN_ENABLE_VULKAN_GDN_HOST_VISIBLE_STATE=1` for the fused GDN path.
- The state buffer is allocated host-visible/coherent, written directly before dispatch, and read directly after dispatch, avoiding explicit transfer commands for that state buffer.

Evidence:
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 13 tests.
- `cargo check -p kiln-server --features vulkan` passed.
- Opt-in benchmark:
  - Prefill: 2099.4ms
  - Decode steps: 375.8, 379.9, 369.3, 365.6, 391.2, 379.2ms
  - Mean ITL: 376.8ms

Verdict:
- Keep disabled. On this stack, avoiding transfer commands did not compensate for shader access to host-visible state memory.

### E030: Final Default And HTTP Server Rerun After Tiling

Change:
- Final default keeps:
  - tiled generic `linear_decode`
  - tiled `gdn_in_proj_decode`
  - Vulkan greedy LM-head argmax
  - fused GDN/MLP/full-attn/host-visible-state paths opt-in only

Evidence:
- Final no-env default bench:
  - Command: `KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prefill: 2540.7ms
  - Decode steps: 362.6, 365.3, 365.0, 372.6, 379.8, 361.4ms
  - Mean ITL: 367.8ms
  - P50 ITL: 365.3ms
  - P99 ITL: 379.8ms
  - Decode throughput: 2.7 tok/s
- Final server command:
  - `KILN_MODEL_PATH=Qwen3.5-4B KILN_PORT=18423 KILN_REQUEST_TIMEOUT_SECS=600 KILN_PREFIX_CACHE_ENABLED=0 ./target/release/kiln serve`
- bs=1, three measured 4-token chat runs:
  - 4 tokens in 3.182s, 1.257 tok/s
  - 4 tokens in 3.058s, 1.308 tok/s
  - 4 tokens in 3.217s, 1.243 tok/s
  - Mean latency: 3.153s for 4 completion tokens.
- bs=4 as four concurrent `/v1/chat/completions` requests:
  - Wall time: 12.848s
  - Completion tokens: 16
  - Aggregate completion throughput: 1.245 tok/s
  - Per-request latencies: 12.840s, 10.638s, 5.301s, 11.717s
- bs=4 through `/v1/completions/batch`:
  - Wall time: 13.070s
  - Completion tokens: 16
  - Aggregate completion throughput: 1.224 tok/s

Verdict:
- Keep. This is a real default latency/server win versus E023:
  - Bench mean ITL improved from 776.9ms to 367.8ms.
  - HTTP bs=1 improved from 4.557s to 3.153s for 4 completion tokens.
- bs>1 still does not scale; aggregate throughput remains close to one active decode stream.

### E031: GDN Recurrent Workgroup Shape 128-Wide

Change:
- Tried changing `gdn_recurrent_prefill.comp` from `local_size_x=256` to `local_size_x=128`.
- Updated `dispatch_gdn_recurrent_step` workgroup count from `ceil(total / 256)` to `ceil(total / 128)`.

Reasoning:
- For Qwen3.5-4B, `dv=128`, so a 128-wide group maps one GDN head per workgroup. The existing 256-wide group maps two heads per workgroup. Since the shader does no inter-lane sharing, a narrower group might improve scheduling or reduce idle work.

Evidence:
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 13 tests.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- First anchor:
  - Prefill: 1873.3ms
  - Decode steps: 345.8, 356.3, 349.3, 373.0, 378.9, 373.2ms
  - Mean ITL: 362.7ms
- Confirmation anchor:
  - Prefill: 2283.1ms
  - Decode steps: 375.0, 397.5, 380.4, 385.2, 389.5, 384.1ms
  - Mean ITL: 385.3ms

Verdict:
- Reverted to the stable 256-wide workgroup. The first run was slightly better than the 367.8ms default anchor, but the confirmation run was worse; this is not a reliable promotion.

### E032: Reusable Transient Descriptor Pool

Change:
- Added a reusable transient descriptor pool to `VulkanDevice`.
- Changed `run_compute_pipeline` to allocate each dispatch descriptor set from that pool, wait for the dispatch, reset the pool, and keep the existing transient command-pool synchronization.
- This removes one `vkCreateDescriptorPool` / `vkDestroyDescriptorPool` pair from every cached Vulkan compute dispatch.

Reasoning:
- After projection tiling, decode still launches many small kernels per token. Descriptor pool create/destroy is pure CPU/Vulkan-driver churn and does not do model work.
- The existing helper already waits the queue idle after each dispatch, so a single reset-after-dispatch transient descriptor pool preserves the current synchronization semantics.

Evidence:
- `cargo check -p kiln-vulkan-kernel` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 13 tests.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- Standard anchor run 1:
  - Prefill: 1878.3ms
  - Decode steps: 374.6, 350.1, 355.1, 367.8, 364.3, 369.3ms
  - Mean ITL: 363.5ms
- Standard anchor run 2:
  - Prefill: 1903.8ms
  - Decode steps: 353.8, 346.5, 346.9, 360.0, 361.1, 367.7ms
  - Mean ITL: 356.0ms
- Two-run mean ITL: 359.8ms, versus the prior stable no-env default anchor of 367.8ms.
- Production HTTP rerun, prefix cache disabled, current prompt fixture:
  - Server command: `KILN_MODEL_PATH=Qwen3.5-4B KILN_PORT=18424 KILN_REQUEST_TIMEOUT_SECS=600 KILN_PREFIX_CACHE_ENABLED=0 ./target/release/kiln serve`
  - bs=1: 4-token chat runs 3.200s, 3.176s, 3.192s; mean 3.189s.
  - bs=4 concurrent `/v1/chat/completions`: 16 tokens in 14.290s, 1.120 tok/s aggregate; per-request latencies 10.808s, 14.289s, 13.075s, 11.955s.
  - bs=4 `/v1/completions/batch`: 16 tokens in 15.544s, 1.029 tok/s aggregate.

Verdict:
- Keep. This is a small but clean bs=1 source-level win with no correctness fallout.
- It does not fix bs>1. The server path remains effectively one active decode forward at a time, so higher batch sizes still need native multi-sequence Vulkan decode instead of endpoint fanout.

### E033: Greedy Deterministic Batch Dedupe

Change:
- Changed `/v1/completions/batch` to synthesize the minimal set of generation jobs for deterministic greedy requests.
- When `temperature == 0.0`, duplicate prompts and `n > 1` completions now share one `generate_one_response` call and duplicate that deterministic result in the batch response.
- Non-greedy sampling keeps the previous one-job-per-output behavior because derived seeds and sampling randomness are part of the output contract.

Reasoning:
- The previous batch endpoint always spawned one task per `(prompt, completion)` pair. That is correct but wasteful for greedy duplicate work.
- For greedy decode, the seed is irrelevant and identical rendered prompts produce identical outputs. Computing those duplicates repeatedly cannot improve quality or diversity; it only spends decode time.

Evidence:
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo test -p kiln-server batch_ --features vulkan` passed, including 11 batch endpoint unit tests.
- `cargo build --release --features vulkan --bin kiln` passed.
- Production server command:
  - `KILN_MODEL_PATH=Qwen3.5-4B KILN_PORT=18425 KILN_REQUEST_TIMEOUT_SECS=600 KILN_PREFIX_CACHE_ENABLED=0 ./target/release/kiln serve`
- Focused HTTP batch benchmark, `temperature: 0`, `max_tokens: 4`:
  - One prompt with `n=4`: 16 completion tokens in 3.265s, 4.900 aggregate tok/s.
  - Four duplicate prompts with `n=1`: 16 completion tokens in 3.201s, 4.999 aggregate tok/s.
  - Four distinct prompts with `n=1`: 16 completion tokens in 12.727s, 1.257 aggregate tok/s.

Verdict:
- Keep. This is a real bs>1 endpoint win for deterministic duplicate work and follows the "remove the need to do it" rule.
- It does not solve distinct-prompt batching. The distinct-prompt case still behaves like one serialized decode stream.

### E034: Batch-Aware Vulkan Linear Decode Primitive

Change:
- Made the cached Vulkan linear-decode path capable of handling `x` shaped `[batch, 1, hidden]` for future native multi-sequence decode.
- First tried extending the existing `linear_decode.comp` shader with a `batch` push constant and a flattened `batch * ceil(out_dim / 16)` dispatch.
- After batch=1 anchors regressed, restored the original single-stream shader and split the batched variant into `linear_decode_batched.comp`, used only when `batch > 1`.

Reasoning:
- Distinct-prompt bs>1 will eventually need backend kernels that operate over multiple active sequences in one forward call. Generic projection is one of the common primitives.
- The production server does not yet construct `[batch, 1, hidden]` decode tensors, so this is a building block rather than an endpoint speedup.
- The batch=1 hot path must stay on the fastest known shader.

Evidence:
- Direct parity:
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 14 tests.
  - New test: `linear_decode_batched_matches_cpu_reference`.
- Full server check:
  - `cargo check -p kiln-server --features vulkan` passed.
- Rejected hot-path attempt:
  - Shared batch-aware `linear_decode.comp` anchor 1: mean ITL 368.5ms.
  - Shared batch-aware `linear_decode.comp` anchor 2: mean ITL 371.9ms.
  - This was worse than the descriptor-pool source-state reruns at 363.5ms / 356.0ms.
- Final split-shader guard:
  - `cargo build --release --features vulkan --bin kiln-bench` passed.
  - Standard anchor after restoring the single-stream shader:
    - Prefill: 1865.7ms
    - Decode steps: 360.2, 365.6, 369.1, 363.1, 377.6, 370.1ms
    - Mean ITL: 367.6ms

Verdict:
- Keep the split primitive. It advances the native batching substrate without imposing the batch-aware shader on single-stream decode.
- No default bs=1 win; do not count this as a latency improvement.

### E035: Promote Full Vulkan MLP Decode

Change:
- Retested the full fused Vulkan MLP decode path after E032 descriptor-pool reuse and E034 shader split.
- Promoted it from opt-in (`KILN_ENABLE_VULKAN_MLP_DECODE=1`) to default.
- Added `KILN_DISABLE_VULKAN_MLP_DECODE=1` as the escape hatch.

Reasoning:
- Earlier full MLP decode was not better than split generic projections, but the cost model changed after descriptor-pool reuse and the projection shader work.
- Full MLP decode removes host-visible round trips for the gate/up hidden buffer and combines the MLP's gate/up/down work into two Vulkan dispatches.
- The path is already shape-gated to single-token, CPU-F32, no-LoRA decode and has direct parity coverage.

Rejected Retests:
- `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED=1`:
  - Prefill: 1923.3ms
  - Decode steps: 376.4, 365.5, 369.4, 377.5, 389.1, 374.2ms
  - Mean ITL: 375.3ms
  - Verdict: still worse than default; keep opt-in.
- `KILN_ENABLE_VULKAN_FULL_ATTN_QKV=1`:
  - Prefill: 1888.5ms
  - Decode steps: 399.7, 403.1, 404.1, 411.3, 416.9, 423.5ms
  - Mean ITL: 409.8ms
  - Verdict: much worse; keep opt-in.

Evidence:
- MLP decode opt-in run 1:
  - Prefill: 1889.5ms
  - Decode steps: 318.2, 322.0, 330.9, 341.8, 346.2, 350.6ms
  - Mean ITL: 334.9ms
- MLP decode opt-in run 2:
  - Prefill: 1972.9ms
  - Decode steps: 352.4, 337.9, 345.4, 354.5, 356.5, 368.3ms
  - Mean ITL: 352.5ms
- `cargo fmt --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 14 tests including `mlp_decode_matches_cpu_reference`.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- Promoted no-env default run 1:
  - Prefill: 2112.9ms
  - Decode steps: 336.5, 339.0, 338.1, 346.4, 359.3, 353.1ms
  - Mean ITL: 345.4ms
- Promoted no-env default run 2:
  - Prefill: 1923.8ms
  - Decode steps: 342.5, 340.0, 335.8, 340.8, 365.5, 370.8ms
  - Mean ITL: 349.2ms
- Promoted no-env two-run mean ITL: 347.3ms.
- Production HTTP rerun, prefix cache disabled:
  - Server command: `KILN_MODEL_PATH=Qwen3.5-4B KILN_PORT=18426 KILN_REQUEST_TIMEOUT_SECS=600 KILN_PREFIX_CACHE_ENABLED=0 ./target/release/kiln serve`
  - bs=1: 4-token chat runs 3.234s, 3.126s, 3.087s; mean 3.149s.
  - bs=4 concurrent `/v1/chat/completions`, distinct prompts: 16 tokens in 12.993s, 1.231 tok/s aggregate; per-request latencies 12.992s, 9.670s, 10.805s, 11.909s.
  - bs=4 `/v1/completions/batch`, distinct prompts: 16 tokens in 13.031s, 1.228 tok/s aggregate.
  - bs=4 `/v1/completions/batch`, one duplicate prompt with `n=4`: 16 tokens in 3.292s, 4.861 tok/s aggregate.

Verdict:
- Keep and default-enable full Vulkan MLP decode.
- This is a real source-level bs=1 decode win: latest default mean ITL improves from the E034 guard of 367.6ms to a two-run mean of 347.3ms.
- End-to-end HTTP bs=1 is only slightly better than the prior best because fixed prompt/template/prefill/server costs dominate the 4-token fixture.
- Distinct-prompt bs>1 still does not scale; native multi-sequence paged decode remains the missing production path.

### E036: Promote Host-Visible Split GDN Recurrent State

Change:
- Added a host-visible recurrent state path to the split `dispatch_gdn_recurrent_step`.
- The default now maps/writes/reads the mutable recurrent state buffer directly as host-visible coherent memory, avoiding explicit state upload/readback copy commands for each GDN linear layer and token.
- Added `KILN_DISABLE_VULKAN_GDN_RECURRENT_HOST_VISIBLE_STATE=1` as the escape hatch.

Reasoning:
- After E035, the decode profiler showed `gdn.recurrent` as the dominant bucket:
  - Around 176-190ms per token across 24 linear-attention layers.
  - Each layer was paying for mutable recurrent-state transfer around the recurrent kernel.
- The earlier host-visible fused-GDN experiment was slower, but that path also fused gates/recurrent/norm and changed more of the memory access pattern. The split recurrent kernel deserved an isolated test.

Evidence:
- Profile before this change, after E035:
  - `gdn.recurrent`: 176.4-190.3ms per token across 24 calls.
  - `mlp.fused`: ~53-54ms across 32 calls.
  - `gdn.gated_norm`: ~31.9-33.5ms across 24 calls.
- `KILN_ENABLE_VULKAN_GDN_RECURRENT_HOST_VISIBLE_STATE=1 cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 14 tests.
- Opt-in run 1:
  - Prefill: 1869.0ms
  - Decode steps: 324.7, 327.0, 320.6, 342.9, 345.3, 351.2ms
  - Mean ITL: 335.3ms
- Opt-in run 2:
  - Prefill: 1896.4ms
  - Decode steps: 323.7, 321.1, 346.7, 352.4, 343.9, 355.8ms
  - Mean ITL: 340.6ms
- After promotion:
  - `cargo fmt --check` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 14 tests.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln-bench` passed.
- Promoted no-env default run 1:
  - Prefill: 1922.1ms
  - Decode steps: 319.3, 319.7, 325.2, 342.2, 355.3, 339.0ms
  - Mean ITL: 333.4ms
- Promoted no-env default run 2:
  - Prefill: 2009.0ms
  - Decode steps: 349.4, 337.7, 330.1, 342.9, 336.2, 350.1ms
  - Mean ITL: 341.1ms
- Promoted no-env two-run mean ITL: 337.3ms.
- Production HTTP rerun, prefix cache disabled:
  - Server command: `KILN_MODEL_PATH=Qwen3.5-4B KILN_PORT=18427 KILN_REQUEST_TIMEOUT_SECS=600 KILN_PREFIX_CACHE_ENABLED=0 ./target/release/kiln serve`
  - bs=1: 4-token chat runs 3.228s, 3.104s, 3.102s; mean 3.145s.
  - bs=4 concurrent `/v1/chat/completions`, distinct prompts: 16 tokens in 12.541s, 1.276 tok/s aggregate; per-request latencies 3.125s, 10.471s, 12.540s, 11.493s.
  - bs=4 `/v1/completions/batch`, distinct prompts: 16 tokens in 12.552s, 1.275 tok/s aggregate.
  - bs=4 `/v1/completions/batch`, one duplicate prompt with `n=4`: 16 tokens in 3.183s, 5.027 tok/s aggregate.
- Promoted profile check:
  - Profiled token times: 333.0ms, 332.1ms, 341.5ms.
  - `gdn.recurrent`: 156.4ms, 158.9ms, 168.1ms across 24 calls.
  - Other large buckets: `mlp.fused` ~53ms/32, `gdn.gated_norm` ~28-29ms/24, `layer.full` ~26-28ms/8, `gdn.in_proj` ~20-21ms/24, `lm_head.argmax` ~13-18ms.

Verdict:
- Keep and default-enable host-visible split GDN recurrent state.
- This is the best default source-level decode result so far: current two-run mean ITL is 337.3ms, down from E035's 347.3ms.
- It improves distinct bs=4 aggregate slightly, but not by true batching; the server still runs one active forward stream at a time for distinct prompts.

### E037: Reject Host-Visible Split GDN Recurrent I/O

Change:
- Tested an opt-in path that made the split recurrent step's Q/K/V/beta/g inputs and output buffer host-visible coherent memory.
- Kept E036's accepted host-visible recurrent state path in place during the test, so this isolated the transient input/output buffers.
- Removed the opt-in path after measurement instead of leaving another slow hot-path switch.

Reasoning:
- Hypothesis: remove five tiny device-local upload submissions and the output readback copy from every recurrent layer.
- On Strix Halo, shader access to host-visible input/output memory was not better than keeping those transient tensors device-local.

Evidence:
- `KILN_ENABLE_VULKAN_GDN_RECURRENT_HOST_VISIBLE_IO=1 cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 14 tests.
- Opt-in run 1:
  - Prefill: 1885.6ms
  - Decode steps: 337.4, 321.1, 323.4, 343.3, 344.7, 341.1ms
  - Mean ITL: 335.2ms
- Opt-in run 2:
  - Prefill: 2089.2ms
  - Decode steps: 344.5, 349.7, 337.6, 332.0, 360.8, 355.0ms
  - Mean ITL: 346.6ms
- Opt-in two-run mean ITL: 340.9ms.
- Current no-env default after E036 remains 333.4ms and 341.1ms, two-run mean 337.3ms.

Verdict:
- Reject and remove the host-visible recurrent I/O path.
- Keep Q/K/V/beta/g inputs and recurrent output device-local.
- E036 remains the current default: host-visible recurrent state only.

### E038: GDN Gates/Gated RMSNorm Transfer Cleanup

Change:
- Replaced the old `VulkanBuffer::upload_data` / `read_back` helpers in `dispatch_gdn_gates` and `dispatch_gdn_gated_rms_norm` with the reusable transient command-pool helpers.
- This removes per-transfer command pool creation/destruction from two default GDN decode kernels that run in every linear-attention layer.

Reasoning:
- After E036/E037, `gdn.gated_norm` was still a visible 24-call bucket.
- The kernels were already using the cached compute-pipeline and descriptor-pool path, but their surrounding tensor transfers still used the older transfer helpers.

Evidence:
- `cargo fmt --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 14 tests.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- No-env default run 1:
  - Prefill: 1835.3ms
  - Decode steps: 313.7, 316.9, 320.2, 333.3, 343.5, 337.4ms
  - Mean ITL: 327.5ms
- No-env default run 2:
  - Prefill: 1889.0ms
  - Decode steps: 325.9, 326.8, 325.1, 347.0, 338.3, 343.1ms
  - Mean ITL: 334.4ms
- Two-run mean ITL: 330.9ms.
- Profile after the transfer cleanup:
  - Profiled token times: 346.2ms, 346.6ms, 350.1ms.
  - `gdn.gated_norm`: 22.9ms, 20.1ms, 24.2ms across 24 calls.
  - `gdn.gates`: 7.6ms, 7.3ms, 7.1ms across 24 calls.

Verdict:
- Keep the transfer cleanup.
- This moved the default source-level bs=1 anchor from E036's 337.3ms two-run mean to 330.9ms.

### E039: Cache Immutable GDN Gates/Norm Weights

Change:
- Added cached-buffer dispatch variants for:
  - `dispatch_gdn_gates_cached`, reusing device-local `A_log` and `dt_bias` buffers.
  - `dispatch_gdn_gated_rms_norm_cached`, reusing the device-local GDN norm weight buffer.
- Routed the Vulkan backend through the existing f32 weight cache for those immutable per-layer tensors.
- Extended the Vulkan parity test to cover the cached gates and cached gated-RMSNorm paths.

Reasoning:
- Even after E038, the default path still uploaded tiny immutable GDN weights every token and every layer.
- These weights are stable model parameters and fit the existing Vulkan weight-cache design already used by projections.

Evidence:
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 14 tests including the cached gates/norm assertions.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- No-env default run 1:
  - Prefill: 1899.2ms
  - Decode steps: 337.7, 327.8, 320.0, 330.0, 326.8, 339.5ms
  - Mean ITL: 330.3ms
- No-env default run 2:
  - Prefill: 1924.5ms
  - Decode steps: 321.3, 319.0, 323.0, 345.6, 332.8, 340.1ms
  - Mean ITL: 330.3ms
- Two-run mean ITL: 330.3ms.
- Profile after cached weights:
  - Profiled token times: 337.4ms, 343.2ms, 339.6ms.
  - `gdn.gates`: 5.1ms, 4.8ms, 5.0ms across 24 calls.
  - `gdn.gated_norm`: 19.0ms, 15.8ms, 19.2ms across 24 calls.
  - Dominant remaining bucket: `gdn.recurrent` at 176.7-184.4ms across 24 calls.

Verdict:
- Keep cached immutable GDN gates/norm weights.
- Current default source-level bs=1 anchor is 330.3ms mean ITL across two no-env runs.
- This improves single-stream decode but does not change distinct-prompt bs>1 behavior; production still needs native multi-sequence paged decode to scale distinct prompts.

### E040: HTTP Server Rerun After GDN Transfer/Cache Cleanup

Change:
- No additional code change. Measured the release server after E038/E039.

Server command:
- `KILN_MODEL_PATH=Qwen3.5-4B KILN_PORT=18428 KILN_REQUEST_TIMEOUT_SECS=600 KILN_PREFIX_CACHE_ENABLED=0 ./target/release/kiln serve`

Workload:
- OpenAI-compatible chat completions and `/v1/completions/batch`.
- `temperature=0.0`, `top_p=1.0`, `max_tokens=4`, served model `qwen3.5-4b-kiln`.
- Prefix cache disabled to avoid shared-prefix effects.

Evidence:
- `cargo build --release --features vulkan --bin kiln` passed.
- Warmup:
  - 4 completion tokens in 3.131s.
- bs=1, three measured chat runs:
  - 4 tokens in 3.051s, 1.311 completion tok/s end-to-end.
  - 4 tokens in 3.116s, 1.284 completion tok/s end-to-end.
  - 4 tokens in 2.984s, 1.341 completion tok/s end-to-end.
  - Mean latency: 3.050s for 4 completion tokens.
- bs=4 as four concurrent `/v1/chat/completions` requests with distinct prompts:
  - Wall time: 12.601s.
  - Completion tokens: 16.
  - Aggregate completion throughput: 1.270 tok/s.
  - Per-request latencies: 9.624s, 11.628s, 12.594s, 10.605s.
- bs=4 through `/v1/completions/batch` with distinct prompts:
  - Wall time: 12.281s.
  - Completion tokens: 16.
  - Aggregate completion throughput: 1.303 tok/s.
- bs=4 through `/v1/completions/batch`, one duplicate prompt with `n=4`:
  - Wall time: 2.991s.
  - Completion tokens: 16.
  - Aggregate completion throughput: 5.349 tok/s.

Verdict:
- Keep. The current release server is measurably faster for bs=1 and deterministic duplicate work.
- Distinct-prompt bs=4 is slightly better than E036 but still not true batching; the server continues to serialize distinct prompt forward work.

### E041: Promote Single-Submit Split GDN Recurrent Step

Change:
- Added a single-submit path for the split `dispatch_gdn_recurrent_step`.
- For each recurrent layer, the path records Q/K/V/beta/g staging copies, optional state upload, compute dispatch, output copy, and optional state readback copy into one command buffer.
- Promoted it to default after A/B; `KILN_DISABLE_VULKAN_GDN_RECURRENT_SINGLE_SUBMIT=1` is the escape hatch.

Reasoning:
- E037 showed shader reads from host-visible Q/K/V/beta/g were slower, so the winning direction is still device-local recurrent inputs/output.
- The previous split recurrent path paid multiple queue submissions around each layer: five small input uploads, one compute dispatch, and one output readback.
- Combining those operations keeps fast device-local shader buffers while removing repeated transfer-submit overhead.

Evidence:
- `KILN_ENABLE_VULKAN_GDN_RECURRENT_SINGLE_SUBMIT=1 cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_recurrent_step -- --nocapture` passed, 2 recurrent tests.
- After promotion:
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 14 tests.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln-bench` passed.
- Opt-in run 1:
  - Prefill: 1835.1ms
  - Decode steps: 325.0, 317.3, 324.8, 339.5, 346.4, 340.9ms
  - Mean ITL: 332.3ms
- Opt-in run 2:
  - Prefill: 2399.7ms
  - Decode steps: 325.1, 308.2, 311.0, 325.5, 324.6, 329.2ms
  - Mean ITL: 320.6ms
- Opt-in run 3:
  - Prefill: 2291.9ms
  - Decode steps: 305.2, 316.5, 321.8, 336.2, 327.4, 319.7ms
  - Mean ITL: 321.1ms
- Promoted no-env default run 1:
  - Prefill: 1882.4ms
  - Decode steps: 321.7, 321.7, 316.6, 334.8, 339.6, 323.2ms
  - Mean ITL: 326.3ms
- Promoted no-env default run 2:
  - Prefill: 1847.5ms
  - Decode steps: 321.7, 321.7, 325.0, 325.0, 337.3, 330.1ms
  - Mean ITL: 326.8ms
- Promoted no-env two-run mean ITL: 326.5ms.
- Profile after promotion:
  - Profiled token times: 332.1ms, 335.5ms, 338.5ms.
  - `gdn.recurrent`: 169.8ms, 171.3ms, 172.8ms across 24 calls.
  - Other large buckets: `mlp.fused` ~53-54ms/32, `layer.full` ~27-28ms/8, `gdn.gated_norm` ~23ms/24, `gdn.in_proj` ~20-21ms/24, `lm_head.argmax` ~14-19ms.

Verdict:
- Keep and default-enable single-submit split GDN recurrent.
- This is the best source-level bs=1 default so far: current two-run mean ITL is 326.5ms, down from E039's 330.3ms.
- The remaining dominant bucket is still GDN recurrent state work; further substantial wins likely require keeping recurrent state and GDN intermediates resident across layers/tokens or implementing true multi-sequence decode for bs>1.

### E042: HTTP Server Rerun After Single-Submit Recurrent

Change:
- No additional code change. Measured the release server after E041.

Server command:
- `KILN_MODEL_PATH=Qwen3.5-4B KILN_PORT=18429 KILN_REQUEST_TIMEOUT_SECS=600 KILN_PREFIX_CACHE_ENABLED=0 ./target/release/kiln serve`

Workload:
- OpenAI-compatible chat completions and `/v1/completions/batch`.
- `temperature=0.0`, `top_p=1.0`, `max_tokens=4`, served model `qwen3.5-4b-kiln`.
- Prefix cache disabled to avoid shared-prefix effects.

Evidence:
- `cargo build --release --features vulkan --bin kiln` passed.
- Warmup:
  - 4 completion tokens in 3.136s.
- bs=1, three measured chat runs:
  - 4 tokens in 2.980s, 1.342 completion tok/s end-to-end.
  - 4 tokens in 2.994s, 1.336 completion tok/s end-to-end.
  - 4 tokens in 3.000s, 1.334 completion tok/s end-to-end.
  - Mean latency: 2.991s for 4 completion tokens.
- bs=4 as four concurrent `/v1/chat/completions` requests with distinct prompts:
  - Wall time: 12.100s.
  - Completion tokens: 16.
  - Aggregate completion throughput: 1.322 tok/s.
  - Per-request latencies: 5.031s, 9.976s, 12.096s, 11.085s.
- bs=4 through `/v1/completions/batch` with distinct prompts:
  - Wall time: 12.003s.
  - Completion tokens: 16.
  - Aggregate completion throughput: 1.333 tok/s.
- bs=4 through `/v1/completions/batch`, one duplicate prompt with `n=4`:
  - Wall time: 2.954s.
  - Completion tokens: 16.
  - Aggregate completion throughput: 5.417 tok/s.

Verdict:
- Keep. This is the best measured production fixture so far for bs=1 and duplicate deterministic batch work.
- Distinct-prompt bs=4 remains serialized in practice; native multi-sequence paged Vulkan decode is still the required bs>1 throughput work.

### E043: Reject Single-Submit GDN Gated RMSNorm

Change:
- Tested an opt-in single-submit path for `dispatch_gdn_gated_rms_norm_cached`.
- The path staged x/z, dispatched the cached-weight gated RMSNorm kernel, and copied output back in one command buffer/queue submit.
- Removed the opt-in path after measurement.

Reasoning:
- After E041, `gdn.gated_norm` was still a visible 24-call bucket.
- The same single-submit idea that helped recurrent state might have removed transfer-submit overhead around gated RMSNorm.

Evidence:
- `KILN_ENABLE_VULKAN_GDN_GATED_NORM_SINGLE_SUBMIT=1 cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_gates_and_gated_rms_norm -- --nocapture` passed, 1 focused test.
- Opt-in run 1:
  - Prefill: 2203.3ms
  - Decode steps: 326.7, 323.5, 318.3, 338.1, 326.5, 339.6ms
  - Mean ITL: 328.8ms
- Opt-in run 2:
  - Prefill: 2093.4ms
  - Decode steps: 333.4, 318.0, 331.6, 339.7, 346.5, 332.5ms
  - Mean ITL: 333.6ms
- Current no-env default after E041 remains 326.3ms and 326.8ms, two-run mean 326.5ms.

Verdict:
- Reject and remove.
- Unlike the recurrent kernel, gated RMSNorm did not benefit from manually combining the transfer and compute work into one command buffer. Keep the existing split helper path.

### E044: Promote Single-Submit Generic Linear Decode

Change:
- Added a single-submit path for cached generic `linear_decode`.
- The path records x staging copy, cached-weight projection dispatch, and output copy into one command buffer/queue submit.
- It supports both the existing batch=1 shader and the split `linear_decode_batched.comp` shader for `batch > 1`.
- Promoted it to default after A/B; `KILN_DISABLE_VULKAN_LINEAR_DECODE_SINGLE_SUBMIT=1` is the escape hatch.

Reasoning:
- Several remaining decode buckets route through generic cached `linear_decode`: GDN output projection and parts of full-attention decode.
- Each call was still paying separate transfer, compute, and readback submissions around a cached weight buffer.
- This is a higher-leverage boundary reduction than the rejected gated-RMSNorm experiment because it applies across multiple projection families.

Evidence:
- `KILN_ENABLE_VULKAN_LINEAR_DECODE_SINGLE_SUBMIT=1 cargo test -p kiln-vulkan-kernel --test gdn_parity linear_decode -- --nocapture` passed, 3 tests including batch=1, batched, and argmax coverage.
- After promotion:
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity linear_decode -- --nocapture` passed, 3 tests.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln-bench` passed.
- Opt-in run 1:
  - Prefill: 1955.7ms
  - Decode steps: 312.6, 314.0, 310.4, 330.5, 331.2, 329.3ms
  - Mean ITL: 321.3ms
- Opt-in run 2:
  - Prefill: 2277.6ms
  - Decode steps: 302.9, 313.0, 314.0, 322.6, 315.9, 319.6ms
  - Mean ITL: 314.7ms
- Promoted no-env default run 1:
  - Prefill: 1868.8ms
  - Decode steps: 335.1, 328.4, 318.1, 327.2, 331.9, 328.5ms
  - Mean ITL: 328.2ms
- Promoted no-env default run 2:
  - Prefill: 1946.0ms
  - Decode steps: 317.5, 303.2, 306.8, 315.6, 319.7, 322.8ms
  - Mean ITL: 314.3ms
- Promoted no-env default run 3:
  - Prefill: 1878.4ms
  - Decode steps: 310.1, 306.4, 314.9, 329.0, 340.1, 329.1ms
  - Mean ITL: 321.6ms
- Promoted no-env three-run mean ITL: 321.4ms.
- Profile after promotion:
  - Profiled token times: 333.6ms, 323.5ms, 318.4ms.
  - `gdn.out_proj`: 13.2ms, 12.4ms, 11.2ms across 24 calls.
  - `layer.full`: 27.3ms, 25.9ms, 25.3ms across 8 calls.
  - `gdn.recurrent`: 166.7ms, 160.0ms, 161.1ms across 24 calls.
  - Other large buckets: `mlp.fused` ~53-55ms/32, `gdn.in_proj` ~24ms/24, `gdn.gated_norm` ~21-23ms/24, `lm_head.argmax` ~13-18ms.

Verdict:
- Keep and default-enable generic linear-decode single-submit.
- This is the best source-level bs=1 default so far: current three-run mean ITL is 321.4ms, down from E041's 326.5ms.
- It also keeps the batch-aware linear primitive covered for future native bs>1 decode, though production distinct-prompt batching still needs a multi-sequence forward path.

### E045: HTTP Server Rerun After Generic Linear Single-Submit

Change:
- No additional code change. Measured the release server after E044.

Server command:
- `KILN_MODEL_PATH=Qwen3.5-4B KILN_PORT=18430 KILN_REQUEST_TIMEOUT_SECS=600 KILN_PREFIX_CACHE_ENABLED=0 ./target/release/kiln serve`

Workload:
- Same short OpenAI-compatible chat and `/v1/completions/batch` fixture as E040/E042.
- `temperature=0.0`, `top_p=1.0`, `max_tokens=4`, served model `qwen3.5-4b-kiln`.
- Prefix cache disabled to avoid shared-prefix effects.

Evidence:
- `cargo build --release --features vulkan --bin kiln` passed.
- HTTP pass 1:
  - Warmup: 4 tokens in 3.039s.
  - bs=1 measured chat runs: 3.014s, 3.060s, 3.139s; mean 3.071s.
  - bs=4 concurrent distinct chat: 16 tokens in 12.346s, 1.296 tok/s aggregate; per-request latencies 12.338s, 10.265s, 9.301s, 11.339s.
  - bs=4 distinct batch: 16 tokens in 12.641s, 1.266 tok/s aggregate.
  - duplicate `n=4` batch: 16 tokens in 2.990s, 5.351 tok/s aggregate.
- HTTP pass 2:
  - bs=1 measured chat runs: 3.066s, 3.077s, 3.019s; mean 3.054s.
  - bs=4 concurrent distinct chat: 16 tokens in 11.767s, 1.360 tok/s aggregate; per-request latencies 2.948s, 10.762s, 11.764s, 5.834s.
  - bs=4 distinct batch: 16 tokens in 12.107s, 1.322 tok/s aggregate.
  - duplicate `n=4` batch: 16 tokens in 2.994s, 5.345 tok/s aggregate.
- Same-binary source-level disable guard:
  - `KILN_DISABLE_VULKAN_LINEAR_DECODE_SINGLE_SUBMIT=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prefill: 1885.2ms.
  - Decode steps: 334.7, 327.0, 327.1, 328.9, 328.0, 336.3ms.
  - Mean ITL: 330.3ms.

Verdict:
- Keep E044 as a source-level default because the same-binary controlled benchmark shows a clear win over disabling it.
- Do not count this as a production HTTP fixture win. E042 remains the best short HTTP fixture for bs=1 and duplicate deterministic batch work.
- Distinct-prompt bs>1 is still serialized; this experiment does not change that architecture.

### E046: Reject Single-Submit GDN Input Projection

Change:
- Tested an opt-in single-submit path for `dispatch_gdn_in_proj_decode_cached`.
- The path staged x, dispatched the fused QKV/Z/A/B projection against cached weights, and copied the fused output back in one command buffer/queue submit.
- Removed the opt-in path after measurement.

Reasoning:
- After E044, `gdn.in_proj` remained around 24ms across 24 GDN calls.
- It looked structurally similar to generic `linear_decode`, where single-submit helped.

Evidence:
- `KILN_ENABLE_VULKAN_GDN_IN_PROJ_SINGLE_SUBMIT=1 cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_in_proj -- --nocapture` passed, 1 focused test.
- Opt-in run 1:
  - Prefill: 1841.5ms
  - Decode steps: 326.6, 322.6, 327.4, 323.9, 343.3, 328.7ms
  - Mean ITL: 328.7ms
- Opt-in run 2:
  - Prefill: 1799.6ms
  - Decode steps: 326.6, 323.1, 323.5, 320.4, 339.7, 329.5ms
  - Mean ITL: 327.1ms
- Current no-env default after E044 remains 328.2ms, 314.3ms, and 321.6ms, three-run mean 321.4ms.

Verdict:
- Reject and remove.
- The fused GDN input projection does not benefit from the manual single-submit wrapper on this driver; keep the existing upload/compute/readback helper path.

### E047: Promote Native Greedy Batch Decode Bridge

Change:
- Added `gdn_in_proj_decode_batched.comp` and `mlp_gate_up_decode_batched.comp` for batch>1 single-token decode.
- Extended the Vulkan decode wrappers so `gdn_in_proj_decode`, `mlp_gate_up_decode`, and `mlp_decode` accept `[batch, 1, hidden]` while keeping the established batch=1 shaders on the single-stream path.
- Added `model_forward_paged_next_tokens_greedy_batch`, which runs GDN/MLP layers with a real batch dimension and still processes full-attention layers one row at a time through the existing `BlockTable`/paged-KV API.
- Added `ModelRunner::generate_paged_shared_tokens_batch_greedy` and promoted it in `/v1/completions/batch` for eligible Vulkan greedy batches.
- Rollback: `KILN_DISABLE_VULKAN_BATCH_GREEDY_DECODE=1`.

Eligibility gates:
- Real Vulkan backend only.
- `temperature == 0.0`.
- No stop sequences.
- No active LoRA adapter.
- Prefix cache disabled. Prefix-cache-enabled traffic keeps the existing per-request path so cached-prefix lookup/registration semantics are not bypassed.
- More than one unique deterministic generation job after duplicate-output dedupe.

Reasoning:
- Distinct-prompt batch work was still fanout over independent single-request jobs.
- A full multi-sequence paged-KV rewrite is larger because `BlockTable` and `PagedKvCache::{write,read}` are single-sequence contracts today.
- Qwen3.5-4B has 24 linear-attention/GDN-heavy layers and only 8 full-attention layers, so a partial lockstep bridge can batch the dominant GDN/MLP decode work without changing the paged-KV table shape.

Evidence:
- `cargo test -p kiln-vulkan-kernel --test gdn_parity batched -- --nocapture` passed, 3 focused batched tests.
- Final validation `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 16 tests.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Short HTTP fixture with opt-in prototype, prefix cache disabled, `temperature=0`, `top_p=1`, `max_tokens=4`:
  - Warmup chat: 4 tokens in 3.502s.
  - bs=1 chat runs: 3.719s, 3.756s, 3.472s; mean 3.649s.
  - bs=4 concurrent distinct chat: 16 tokens in 12.438s, 1.286 tok/s aggregate.
  - bs=4 distinct batch: 16 tokens in 11.857s, 1.349 tok/s aggregate.
  - duplicate `n=4` batch: 16 tokens in 3.063s, 5.223 tok/s aggregate.
- Longer same-prompt distinct batch fixture, prefix cache disabled, `temperature=0`, `top_p=1`, `max_tokens=16`:
  - Opt-in prototype: 64 tokens in 26.312s, 2.432 tok/s.
  - Same-source unflagged pre-promotion baseline: 64 tokens in 29.596s, 2.162 tok/s.
  - Promoted no-env default after backend batch-MLP gate fix: 64 tokens in 26.615s, 2.405 tok/s.
  - Promoted same-binary disable guard after backend batch-MLP gate fix (`KILN_DISABLE_VULKAN_BATCH_GREEDY_DECODE=1`): 64 tokens in 28.697s, 2.230 tok/s.
- Final bs=1 source bench after promotion:
  - `KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prefill: 2496.8ms.
  - Decode steps: 317.5, 321.3, 312.0, 314.5, 329.6, 336.9ms.
  - Mean ITL: 321.9ms, 3.1 tok/s.

Verdict:
- Keep and default-enable for the eligible batch endpoint path.
- This is the first accepted distinct-prompt bs>1 server improvement: the same-binary promoted disable guard shows a 7.8% aggregate throughput win on the 64-token batch fixture, while bs=1 source latency stays in the existing 321ms band.
- This is still not full continuous batching. Prefill remains per prompt, full-attention layers still run row-by-row, completed rows stay in the fixed batch shape as dummy work, and prefix-cache-enabled traffic falls back to the old path.

### E048: Prefix-Cache-Aware Native Batch On Default Server Config

Change:
- Removed the blanket "prefix cache must be disabled" gate from the native greedy batch endpoint path.
- Added `RealPrefixCache::would_hit`, a non-mutating prefix-cache probe.
- If any prompt would hit the prefix cache, the batch endpoint keeps the established per-request path so cached KV blocks and linear-attention state are reused.
- If every prompt would miss, the endpoint now runs the native Vulkan batch path even when prefix cache is enabled, then registers any block-aligned prompt snapshots returned by batch generation.
- Batch generation now returns prompt-prefix registrations and allocated block ownership so the API layer can retain registered blocks and free the rest, matching the single-request prefix-cache ownership pattern.

Reasoning:
- E047 was measured with `KILN_PREFIX_CACHE_ENABLED=0`, but the production default has prefix caching enabled.
- Prefix cache is a performance feature, not an output semantics feature. The safe default-server policy is to preserve cache-hit behavior and use native batch only for miss-only batches where the old path would not reuse any cached prefix anyway.
- Registering block-aligned prompt snapshots keeps future prefix-cache reuse available instead of blindly bypassing the cache.

Evidence:
- `cargo test -p kiln-server real_prefix_cache --features vulkan -- --nocapture` passed, including `would_hit` coverage that confirms the probe does not increment hit/miss counters.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed after the runtime change.
- Default prefix-cache-enabled server command:
  - `KILN_MODEL_PATH=Qwen3.5-4B KILN_PORT=18453 KILN_REQUEST_TIMEOUT_SECS=600 ./target/release/kiln serve`
  - Startup confirmed prefix cache enabled: `max_blocks=31350`, `max_entries=20`.
- Prefix-enabled 64-token distinct batch fixture, `temperature=0`, `top_p=1`, `max_tokens=16`:
  - No-env default: 64 tokens in 26.977s, 2.372 tok/s.
  - Same-binary disable guard (`KILN_DISABLE_VULKAN_BATCH_GREEDY_DECODE=1`): 64 tokens in 28.427s, 2.251 tok/s.

Verdict:
- Keep.
- The default server configuration now gets the distinct-prompt native batch win for prefix-cache miss batches: 5.4% aggregate throughput improvement on the 64-token fixture versus the disable guard.
- Prefix-cache hits still fall back to the prior path; the remaining work is native batching for suffix prefill/decode after shared cached prefixes.

### E049: Mixed Prefix-Hit/Prefix-Miss Batch Split

Change:
- Extended the E048 gate from "all prompts must miss the prefix cache" to a mixed execution plan.
- Prefix-cache miss jobs are grouped into the native Vulkan greedy batch path when at least two misses are present.
- Prefix-cache hit jobs are still executed through the existing single-request path so they perform the real `lookup`, reuse cached blocks/GDN state, and update prefix-cache metrics/refcounts normally.
- The final batch response merges native miss-job outputs with hit-job responses before expanding duplicate deterministic outputs.

Reasoning:
- A batch with one cached long-prefix request and several unrelated misses was still falling back entirely to the old per-job fanout.
- There is no need to choose between prefix reuse and native batching: the miss subset can be batched, while hit requests keep the more efficient cached-prefix path.

Evidence:
- Added `Clone` derives for `ChatCompletionRequest`, `Message`, and `AdapterRef` so hit jobs can be spawned through the existing response generator while native miss jobs run separately.
- `cargo fmt --check` passed.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo test -p kiln-server real_prefix_cache --features vulkan -- --nocapture` passed.
- Mixed hit/miss fixture:
  - Registered a 32-token base prompt with `max_tokens=0`; prefix metrics showed one cached entry, 2 cached blocks.
  - Used a delimiter-shaped suffix prompt so its rendered prompt starts with the cached 32-token base.
  - No-env split path metrics after batch: `kiln_prefix_cache_lookups_total{result="hit"} 1`, `kiln_prefix_cache_hit_tokens_total 32`, `kiln_prefix_cache_hit_blocks_total 2`.
  - No-env split path: 64 tokens in 27.345s, 2.340 tok/s.
  - Same fixture with `KILN_DISABLE_VULKAN_BATCH_GREEDY_DECODE=1`: 64 tokens in 28.458s, 2.249 tok/s, also with one prefix hit.

Verdict:
- Keep.
- Mixed prefix-hit/prefix-miss traffic gets a 4.0% aggregate throughput improvement while preserving real prefix-cache reuse for hit jobs.
- The remaining structural limitation is that hit-job suffixes are not themselves part of the native batch; this still requires a batch generator that accepts per-row cached prefixes and linear-attention states.

### E050: Native Batch With Per-Row Cached Prefixes

Change:
- Extended `ModelRunner::generate_paged_shared_tokens_batch_greedy` to accept an optional per-row `PagedPrefixReuse`.
- For prefix-hit rows, batch generation now appends newly allocated blocks after cached KV blocks, starts prefill at the cached token count, resumes from the cached linear-attention state, and registers the completed block-aligned prompt snapshot after success.
- Updated `/v1/completions/batch` so eligible Vulkan greedy batches do real prefix-cache lookups for every row, pass hit snapshots into one native batch, release hit refcounts on every error/timeout/success path, and register returned prompt snapshots before freeing unretained blocks.
- Removed the E049 mixed split path; prefix-hit and prefix-miss rows now share one native batch whenever the batch endpoint is otherwise eligible.

Reasoning:
- E049 preserved cache reuse but ran hit rows outside the native batch, so mixed traffic still paid a separate per-request path for cached suffixes.
- The cache hit already provides exactly the reusable state needed by lockstep decode: cached block IDs plus a single-row linear-attention snapshot. The missing piece was row-local suffix prefill before stacking the linear states for batched decode.
- Keeping lookup/release/register ownership in the API layer preserves the single-request cache semantics while letting the model batch generator stay focused on block tables, suffix prefill, and decode.

Evidence:
- `cargo fmt --check` passed.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo test -p kiln-server real_prefix_cache --features vulkan -- --nocapture` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 16 tests.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Final bs=1 source bench after the server change:
  - `KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prefill: 1869.9ms.
  - Decode steps: 303.9, 313.6, 316.1, 329.7, 340.4, 325.7ms.
  - Mean ITL: 321.6ms, 3.1 tok/s.
- Mixed hit/miss fixture, default prefix-cache-enabled server:
  - Registered the same 32-token base prompt with `max_tokens=0`; prefix metrics showed one cached entry and 2 cached blocks.
  - Batch had one delimiter-shaped prefix-hit row plus three unrelated miss rows, `temperature=0`, `top_p=1`, `max_tokens=16`.
  - No-env all-native path metrics after batch: `kiln_prefix_cache_lookups_total{result="hit"} 1`, `kiln_prefix_cache_hit_tokens_total 32`, `kiln_prefix_cache_hit_blocks_total 2`.
  - No-env all-native path: 64 tokens in 26.984s, 2.372 tok/s.
  - Same fixture with `KILN_DISABLE_VULKAN_BATCH_GREEDY_DECODE=1`: 64 tokens in 29.245s, 2.188 tok/s, also with one prefix hit.

Verdict:
- Keep.
- This replaces the E049 split path with a simpler all-native mixed path.
- The current rerun shows an 8.4% aggregate throughput improvement versus the disable guard, and it is also slightly faster than the E049 split result (27.345s / 2.340 tok/s).
- Remaining bs>1 work is now below this bridge: batched full-attention paged-KV reads/writes, variable-row compaction after EOS, and broader continuous batching outside `/v1/completions/batch`.

### E051: Retest Rejected Full-Attn QKV And Fused GDN Decode Paths

Change:
- No accepted runtime change.
- Retested `KILN_ENABLE_VULKAN_FULL_ATTN_QKV=1` on the current E050 source because full-attention layers remain row-by-row in native batch decode.
- Retested `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED=1` because the latest decode profile still shows `gdn.recurrent` as the dominant bs=1 bucket.
- Briefly tried a 128-lane `gdn_decode_gates_recurrent_rmsnorm.comp` workgroup for Qwen3.5-4B's `dv=128`; reverted it after measurement.

Reasoning:
- Full-attn QKV was previously rejected for bs=1, but not against the current native batch bridge.
- Fused GDN decode can remove separate gates/recurrent/gated-norm round trips in principle, but earlier measurements were unstable. Current defaults changed enough to justify one retest.
- The fused GDN shader used 256 lanes per head while Qwen3.5-4B has `dv=128`, so a 128-lane variant was a small contained tuning attempt.

Evidence:
- Current decode profile (`KILN_PROFILE_DECODE=1`, 3 decode steps) still identifies `gdn.recurrent` as the largest bs=1 bucket:
  - `gdn.recurrent`: 177.0ms, 176.7ms, 173.9ms across 24 calls.
  - `mlp.fused`: 55.1ms, 56.8ms, 54.1ms across 32 calls.
  - `layer.full`: 26.1ms, 26.8ms, 25.7ms across 8 calls.
- `KILN_ENABLE_VULKAN_FULL_ATTN_QKV=1`:
  - bs=1 source bench: prefill 1873.9ms, mean ITL 379.8ms, decode steps 381.7, 368.7, 381.7, 375.3, 390.7, 380.5ms.
  - Mixed 64-token cache-hit/miss batch: 30.826s, 2.076 tok/s, with one prefix hit, 32 hit tokens, 2 hit blocks.
- `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED=1` before shader tuning:
  - First run: prefill 1890.2ms, mean ITL 290.0ms, decode steps 226.9, 346.6, 223.4, 348.9, 242.7, 351.3ms.
  - Confirmation run: prefill 1926.3ms, mean ITL 362.2ms, decode steps 358.2, 353.8, 350.9, 367.7, 362.4, 380.2ms.
- 128-lane fused-shader tuning:
  - Focused parity `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_decode_gates_recurrent_rmsnorm -- --nocapture` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - Opt-in bench was prefill 1932.2ms, mean ITL 363.7ms, decode steps 366.0, 353.1, 357.3, 364.2, 379.9, 361.9ms.

Verdict:
- Reject.
- Full-attn QKV is worse for both current bs=1 and the mixed batch fixture.
- Fused GDN decode remains too unstable to promote; the 128-lane tuning did not help.
- The 128-lane shader and wider-shape guard changes were reverted. Keep the fused GDN decode path opt-in only.

### E052: Batch Full-Attention Layer Post-MLP Work

Change:
- Added `transformer_block_paged_batch_decode_rows` for native batch decode.
- Full-attention layers still run paged attention row-by-row through the existing single-sequence `BlockTable` API, but no longer run the whole transformer block row-by-row.
- After row-local attention outputs are concatenated, the batch path now performs the residual add, post-attention RMSNorm, and SwiGLU MLP once over `[batch, 1, hidden]`.

Reasoning:
- E050 batched GDN and MLP layers, but the 8 full-attention layers still called the full transformer block once per row.
- That row loop included each full-attention layer's MLP, so batch>1 traffic still paid separate MLP dispatches for those 8 layers even though MLP decode already supports `batch > 1`.
- This keeps the hard part, paged-KV attention, unchanged while removing unnecessary per-row MLP work.

Evidence:
- `cargo fmt --check` passed.
- `cargo check -p kiln-server --features vulkan` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 16 tests.
- `cargo test -p kiln-server real_prefix_cache --features vulkan -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Mixed hit/miss fixture, default prefix-cache-enabled server:
  - Registered the same 32-token base prompt with `max_tokens=0`; prefix metrics showed one cached entry and 2 cached blocks.
  - Batch had one delimiter-shaped prefix-hit row plus three unrelated miss rows, `temperature=0`, `top_p=1`, `max_tokens=16`.
  - No-env native path metrics after batch: `kiln_prefix_cache_lookups_total{result="hit"} 1`, `kiln_prefix_cache_hit_tokens_total 32`, `kiln_prefix_cache_hit_blocks_total 2`.
  - No-env native path: 64 tokens in 26.037s, 2.458 tok/s.
  - Same-source disable guard (`KILN_DISABLE_VULKAN_BATCH_GREEDY_DECODE=1`): 64 tokens in 30.892s, 2.072 tok/s.
  - Compared with E050's same fixture result, no-env improved from 26.984s / 2.372 tok/s to 26.037s / 2.458 tok/s.
- Short miss-only distinct prompt check, default prefix-cache-enabled server:
  - 64 tokens in 26.401s, 2.424 tok/s, with 77 prompt tokens.
- bs=1 source checks after the patch:
  - Post-E052 reruns: mean ITL 329.5ms and 334.1ms.
  - This edit only changes `model_forward_paged_next_tokens_greedy_batch`, so it is not on the bs=1 source path. The best E050 no-env bs=1 anchor remains 321.6ms.

Verdict:
- Keep.
- Mixed cache-hit/miss native batch throughput improves another 3.6% over E050 and 18.6% versus the current disable guard.
- This is still not full multi-sequence paged attention: QKV projection, KV write/read, SDPA, and o_proj for full-attention layers remain row-local.

### E053: Rejected Batch Full-Attention QKV/O-Projection Factoring

Change:
- Added a batch-only projected-QKV paged decode core for `transformer_block_paged_batch_decode_rows`.
- The experiment batched full-attention QKV projection, QK norm, and o_proj over `[batch, 1, hidden]`, then kept RoPE, paged KV write/read, and SDPA row-local.
- LoRA and MTP/debug capture paths still fell back to the existing row-local `gqa_attention_paged_with_rope_tables`.
- Reverted after measurement.

Reasoning:
- E052 removed per-row MLP work from full-attention layers, but QKV and o_proj still ran once per row.
- The existing Vulkan `linear_decode_batched` primitive can handle `batch > 1`, so factoring projection and o_proj out of the row-local attention loop looked like the next low-risk step before building true batched paged attention.

Evidence:
- With the experiment applied:
  - `cargo fmt --check` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 16 tests.
  - `cargo test -p kiln-server real_prefix_cache --features vulkan -- --nocapture` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Mixed hit/miss fixture, default prefix-cache-enabled server:
  - First valid run: 64 tokens in 27.042s, 2.367 tok/s, with one prefix hit, 32 hit tokens, 2 hit blocks.
  - Confirmation run: 64 tokens in 26.540s, 2.411 tok/s, again with 32 hit tokens and 2 hit blocks.
  - Both runs were slower than E052's 26.037s / 2.458 tok/s.
- Short miss-only distinct prompt check:
  - 64 tokens in 26.241s, 2.439 tok/s, with 77 prompt tokens.
  - This slightly beat E052's short miss-only check (26.401s / 2.424 tok/s), but the gain was small and did not justify regressing the mixed prefix-cache fixture.
- After reverting:
  - `cargo fmt --check` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 16 tests.
  - `cargo test -p kiln-server real_prefix_cache --features vulkan -- --nocapture` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
  - Mixed hit/miss fixture returned to 64 tokens in 26.009s, 2.461 tok/s, with one prefix hit, 32 hit tokens, 2 hit blocks.

Verdict:
- Reject.
- The projection/o_proj factoring is not enough on its own. It slightly helped the short all-miss case but regressed the mixed cache-hit/cache-miss fixture that exercises the default server path.
- Keep the E052 structure: batched post-attention MLP for full-attention layers, row-local attention including QKV and o_proj.

### E054: Rejected GDN Recurrent Host-Visible State Buffer Cache

Change:
- Added an opt-in `KILN_ENABLE_VULKAN_GDN_RECURRENT_STATE_CACHE=1` experiment.
- The experiment remembered the host-visible Vulkan buffer for the CPU tensor returned by the previous `gdn_recurrent_step`, then reused that buffer when the next step consumed the same tensor id.
- The goal was to skip re-extracting and rewriting the 2 MiB recurrent state into a fresh host-visible buffer on every GDN layer/token.
- Reverted after measurement.

Reasoning:
- The current split recurrent path already uses a host-visible mutable state buffer, but the model state is still represented as a CPU `Tensor` between calls for rollback/prefix-cache semantics.
- Since the recurrent step reads the updated state back to CPU, the next step usually writes those same bytes back to Vulkan. A tensor-id keyed cache looked like a contained way to avoid one direction of that copy without changing public state semantics.

Evidence:
- With the opt-in implementation applied:
  - `cargo fmt --check` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 16 tests.
  - `cargo test -p kiln-server real_prefix_cache --features vulkan -- --nocapture` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Same-source no-env bs=1 anchor:
  - Prefill 1867.3ms.
  - Decode steps: 320.4, 316.6, 315.2, 336.2, 322.5, 333.1ms.
  - Mean ITL 324.0ms.
- `KILN_ENABLE_VULKAN_GDN_RECURRENT_STATE_CACHE=1`:
  - First run: prefill 2484.1ms, mean ITL 333.3ms, decode steps 366.9, 326.0, 313.0, 332.4, 328.8, 332.9ms.
  - Confirmation run: prefill 1858.2ms, mean ITL 329.1ms, decode steps 364.0, 320.1, 314.6, 323.9, 327.3, 324.5ms.
- After reverting:
  - `cargo fmt --check` passed.
  - `cargo check -p kiln-server --features vulkan` passed.

Verdict:
- Reject.
- The cache removes one apparent state write, but the extra buffer lifetime/map path did not beat the simple current single-submit recurrent implementation.
- A real win likely needs a stronger representation change: keep recurrent state resident through the full decode loop and only materialize CPU snapshots for rollback/prefix-cache boundaries.

### E055: Rejected Native Batch Profiling And Batch Device-Local GDN State

Change:
- Added temporary native-batch decode profiling around the lockstep batch path.
- Also tested making split GDN recurrent state device-local by default for `batch > 1`, while keeping host-visible recurrent state for bs=1.
- Removed the batch profiler hooks and reverted the GDN state policy after measurement.

Reasoning:
- The mixed batch path needed a current breakdown before attempting deeper bs>1 work.
- Short profiling showed native batch decode was dominated by GDN recurrent state, MLP, and LM-head work, not the row-local full-attention attention calls.
- Device-local recurrent state looked attractive for batch because the short profiled fixture improved, but the full 64-token fixture is the acceptance gate.

Evidence:
- `KILN_PROFILE_DECODE=1`, mixed fixture with `max_tokens=4`, host-visible recurrent state:
  - `gdn.recurrent`: 529.0ms, 734.1ms, 672.6ms across 24 calls on three batch decode steps.
  - `mlp.fused`: ~195-202ms across 32 calls.
  - `lm_head.batch`: ~86-98ms per step.
  - `full.row.attn`: ~51-56ms across 32 calls.
- `KILN_DISABLE_VULKAN_GDN_RECURRENT_HOST_VISIBLE_STATE=1` improved the short 4-token fixture from 12.348s to 11.598s, but promoting a batch-specific device-local default regressed the full mixed fixture:
  - Default with batch device-local policy: 26.894s / 2.380 tok/s, then 27.322s / 2.342 tok/s.
- After reverting the GDN policy but keeping the batch profiler code inactive, the mixed fixture still measured 26.888s / 2.380 tok/s, slower than the E052/E053-revert 26.009s / 2.461 tok/s confirmation.
- Removed the batch-only profiler hooks from the default path.

Verdict:
- Reject.
- The profile data is useful, but the temporary instrumentation and device-local batch policy are not default-speed improvements.
- The next bs>1 work should target persistent state/intermediate residency or a better batched LM-head/GDN strategy without adding inactive-path overhead.

### E056: Rejected Batched Vulkan LM-Head Argmax

Change:
- Added a batched Vulkan LM-head argmax experiment to avoid reading back `[batch, vocab]` logits for greedy native batch decode.
- First version used separate upload, block-argmax dispatch, reduce dispatch, and readback.
- Second version combined upload, both dispatches, and readback into one command buffer.
- Removed the model/backend/shader/test surfaces after measurement.

Reasoning:
- Batch profiles showed `lm_head.batch` around 86-98ms per decode step because greedy sampling materializes full vocab logits on CPU.
- A batched argmax should theoretically keep logits device-local and read back only one token id per row.

Evidence:
- Focused parity passed for both single and batched argmax while the experiment was applied.
- Multi-submit batched argmax on the 64-token mixed fixture regressed to 54.129s / 1.182 tok/s.
- Single-submit batched argmax was worse:
  - 73.428s / 0.872 tok/s on the mixed fixture.
  - 73.292s / 0.873 tok/s confirmation.
- After removing the argmax route and doing a clean rebuild, bs=1 source latency remained healthy:
  - Prefill 1982.8ms, mean ITL 328.5ms.

Verdict:
- Reject.
- On Strix Halo/RADV, the argmax shader shape is much slower than materializing logits through the existing batched linear path.
- LM-head optimization still matters, but this reduction kernel is not the right implementation.

### E057: Native Batch Bridge Rolled Back To Opt-In

Change:
- Changed `KILN_ENABLE_VULKAN_BATCH_GREEDY_DECODE=1` to opt into the native greedy batch bridge.
- `KILN_DISABLE_VULKAN_BATCH_GREEDY_DECODE=1` remains an explicit override if both env vars are present.

Reasoning:
- After the profiler and argmax experiments were removed, the native batch bridge in this source state repeatedly measured as pathological on the mixed fixture.
- The existing non-native batch path is slower than E052's best native result, but it is much faster than the current native path and preserves correctness/prefix-cache behavior.

Evidence:
- Clean no-debug default before rollback, mixed prefix-hit/prefix-miss fixture:
  - 75.371s / 0.849 tok/s, with one hit, 32 hit tokens, 2 hit blocks.
- Existing rollback guard on the same binary:
  - `KILN_DISABLE_VULKAN_BATCH_GREEDY_DECODE=1`: 28.546s / 2.242 tok/s, with the same prefix-cache metrics.
- New no-env default after making native batch opt-in:
  - 28.863s / 2.217 tok/s, with one hit, 32 hit tokens, 2 hit blocks.
- Validation after rollback:
  - `cargo fmt --all --check` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 16 tests.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.

Verdict:
- Keep the rollback.
- This gives up the earlier E052 native-batch default win until the pathology is found, but it avoids shipping a 3x default regression.
- Current safest default mixed-batch result is 28.863s / 2.217 tok/s; native batch remains available only for targeted debugging with `KILN_ENABLE_VULKAN_BATCH_GREEDY_DECODE=1`.

### E058: Fixed Opt-In Native Batch GDN Input Projection

Change:
- Added env-gated native-batch timing behind `KILN_PROFILE_VULKAN_BATCH_GREEDY=1`.
- Reused the existing decode sub-op profiler inside `model_forward_paged_next_tokens_greedy_batch` so native batch can attribute GDN/MLP sub-ops without affecting the no-profile path.
- Removed the stale `batch != 1` guard from the Vulkan backend's `gdn_in_proj_decode` route. The kernel layer already had `gdn_in_proj_decode_batched.comp`, but the model backend never called it for native batch rows.
- Kept native batch opt-in only.

Reasoning:
- E057 proved the current native batch bridge was pathological, but not why.
- Coarse profiling showed the post-prefill native decode steps were taking about 4.0-4.4s each on the 4-row mixed fixture.
- Sub-op profiling showed the dominant cost was not LM head: `gdn.in_proj` alone was around 2.9-3.1s per decode step across 24 GDN layers because `batch > 1` fell back to four CPU `broadcast_matmul`s per layer.

Evidence:
- Before the backend gate fix, profiled opt-in native batch on the mixed `max_tokens=4` fixture:
  - End-to-end batch: 20.466s for 16 generated tokens.
  - Decode step profiles: 4171.7ms, 4023.2ms, 3990.9ms.
  - `linear.gdn`: 3833.4ms, 3677.0ms, 3641.5ms across 24 layers.
  - `gdn.in_proj`: 3123.5ms, 2912.4ms, 2947.6ms across 24 layers.
- After enabling the existing batched Vulkan GDN in-proj shader:
  - Profiled mixed `max_tokens=4`: 15.141s for 16 generated tokens.
  - Decode step profiles: 970.4ms, 1168.4ms, 1192.1ms.
  - `gdn.in_proj`: 86.7ms, 86.8ms, 85.3ms across 24 layers.
  - Remaining decode bottleneck became `gdn.recurrent`: 491.6ms, 657.5ms, 690.5ms across 24 layers.
- Full no-profile mixed `max_tokens=16` fixture after the fix:
  - Opt-in native: 27.750s / 2.306 tok/s.
  - Opt-in native confirmation: 29.851s / 2.144 tok/s.
  - Same-binary no-env fallback: 29.137s / 2.197 tok/s.
- Longer no-profile mixed `max_tokens=32` fixture:
  - No-env fallback: 50.908s / 2.514 tok/s.
  - Opt-in native: 54.790s / 2.336 tok/s.
- Validation after the fix and profiler cleanup:
  - `cargo fmt --all --check` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 16 tests.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.

Verdict:
- Keep the batched GDN in-proj backend fix and the env-gated profiler.
- Keep native batch opt-in only.
- This removes the worst native-batch pathology and makes targeted debugging viable again, but it is not a reliable default throughput win. The longer fixture shows the remaining native batch path is still limited by row prefill and batched GDN recurrent state traffic.

### E059: Promoted Native Batch With Batch-Default Fused GDN Decode

Change:
- Made `gdn_decode_gates_recurrent_rmsnorm.comp` batch-aware for `[batch, 1, nv, *]` inputs and `[batch, nv, dk, dv]` recurrent state.
- Updated the Vulkan fused GDN decode dispatcher to pass `batch` as a push constant and dispatch `batch * nv` workgroups.
- Extended the existing GDN fused decode parity test to a `batch=3` shape.
- Changed the Vulkan backend so fused GDN decode remains opt-in for bs=1, but is enabled by shape for `batch > 1`.
- Re-promoted native greedy batch decode as the default. `KILN_DISABLE_VULKAN_BATCH_GREEDY_DECODE=1` remains the rollback guard.

Reasoning:
- E058 made native batch use the batched Vulkan input-projection shader, exposing `gdn.recurrent` as the remaining dominant native-batch decode bucket.
- The existing fused GDN decode path was single-row only and therefore invisible to native batch.
- Fusing gates + recurrent update + gated RMSNorm is a good batch-specific fit because it removes the separate gates/recurrent/gated-norm CPU/Vulkan boundaries from every GDN layer and token.
- The old bs=1 fused GDN path remains too unstable to default, so the promotion is deliberately shape-scoped to `batch > 1`.

Evidence:
- Validation for the batch-aware fused shader:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 16 tests, including the batch-shaped fused GDN decode test.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Profiled native batch with explicit fused opt-ins, mixed `max_tokens=4`:
  - End-to-end: 12.390s for 16 generated tokens.
  - Decode step profiles: 823.0ms, 1397.4ms, 1395.6ms.
  - `gdn.fused_gates_recur_norm`: 331.8ms, 898.8ms, 895.4ms across 24 layers.
  - This was not uniformly better per short decode step than E058, but it showed the batch-fused path was active and correct.
- No-profile explicit fused native measurements:
  - `max_tokens=16`: 28.586s / 2.239 tok/s.
  - `max_tokens=32`: 36.617s / 3.496 tok/s, confirmation 37.817s / 3.385 tok/s.
- After making fused GDN automatic for `batch > 1`, but keeping native batch opt-in:
  - `max_tokens=16`: 23.358s / 2.740 tok/s.
  - `max_tokens=32`: 37.429s / 3.420 tok/s.
- After re-promoting native batch to no-env default:
  - `max_tokens=16`: 23.640s / 2.707 tok/s.
  - `max_tokens=32`: 41.595s / 3.077 tok/s.
- Same-binary rollback guard:
  - `KILN_DISABLE_VULKAN_BATCH_GREEDY_DECODE=1`, `max_tokens=16`: 28.761s / 2.225 tok/s.
  - `KILN_DISABLE_VULKAN_BATCH_GREEDY_DECODE=1`, `max_tokens=32`: 49.893s / 2.565 tok/s.
- bs=1 source anchor after the promotion:
  - `KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prefill 2585.9ms.
  - Mean ITL 318.1ms, p50 319.0ms, p99 334.4ms.

Verdict:
- Keep.
- This establishes a new default mixed-batch best on the current source state: 23.640s / 2.707 tok/s for the 64-token fixture, a 21.6% throughput gain over the same-binary disable guard.
- Longer generation benefits more: 41.595s / 3.077 tok/s for 128 tokens, a 20.0% throughput gain over the same-binary disable guard.
- The shape-scoped fused GDN policy preserves bs=1 behavior while making native batch default-safe again.

### E060: Rejected Cached Immutable Fused-GDN Tensors

Change:
- Tried caching the immutable fused-GDN decode side tensors (`A_log`, `dt_bias`, and gated RMSNorm weight) in Vulkan device-local buffers instead of uploading them inside every fused GDN decode dispatch.
- Added those tensors to decode weight prewarm for linear-attention layers.
- Reverted the experiment after measurement; the fused GDN decode dispatcher now uses the E059 direct-upload path again.

Reasoning:
- These tensors are small, but the fused GDN batch path calls them at every GDN layer and decode step, so avoiding repeated uploads looked like a low-risk throughput candidate.
- In practice, the added cache/prewarm path increased startup work and made the mixed batch fixture slower, especially on longer generation.
- The E059 direct path remains simpler and faster for the current batch-fused GDN decode implementation.

Evidence:
- Cached experiment startup:
  - Prewarm count increased from the E059 249 weights to 321 weights.
  - Log line: `weights=321`, `f32_cache_mb=16040`, `elapsed_ms=19232`.
- Cached experiment, no-env mixed fixture:
  - `max_tokens=16`: 24.808s / 2.580 tok/s, worse than E059 default 23.640s / 2.707 tok/s.
  - `max_tokens=32`: 54.755s / 2.338 tok/s, much worse than E059 default 41.595s / 3.077 tok/s.
- Validation after reverting to the E059 direct path:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 16 tests.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.

Verdict:
- Reject and revert.
- Small immutable-buffer caching is not useful at this layer boundary on Strix Halo; it adds prewarm/cache pressure without improving the batch-fused decode loop.
- Current default remains E059: native batch is enabled by default, fused GDN decode is automatic for `batch > 1`, and bs=1 fused GDN stays opt-in.

### E061: Native Batch Active-Row Compaction After EOS

Change:
- Changed `ModelRunner::generate_paged_shared_tokens_batch_greedy` so finished rows are removed from later native batch decode calls.
- The loop now tracks active sequence indices. Rows that hit EOS or `max_tokens` stop contributing `input_tokens`, `start_positions`, `BlockTable`s, and sequence-length updates.
- Added `select_linear_attention_state_rows` to compact the stacked GDN linear-attention recurrent/conv state to the remaining active rows before the next decode pass.
- Added focused tests for compacting linear-attention state rows and for native batch greedy stopping empty EOS rows.

Reasoning:
- Before this change, if one row hit EOS while other rows kept generating, the lockstep native batch loop still decoded the finished row's stale token and advanced its paged position until every row finished or the global `max_tokens` cap was reached.
- Removing a row after EOS is the cleanest kind of speedup: for early-finish traffic, the server no longer performs attention, GDN, MLP, LM-head, KV writes, or recurrent-state traffic for work that cannot affect the response.
- This does not change the bs=1 path, and it is a no-op for the current fixed-`max_tokens` benchmark fixture when every row finishes by length.

Evidence:
- Validation:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-model generate::tests::compact_linear_attention_state_selects_active_batch_rows -- --nocapture` passed.
  - `cargo test -p kiln-model generate::tests::native_batch_greedy_stops_empty_rows_on_eos -- --nocapture` passed.
  - `cargo test -p kiln-model generate::tests::test_paged_eos_detection -- --nocapture` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Server sanity run after the change:
  - Prefix-cache-enabled default Vulkan server, `/v1/completions/batch`, 4 distinct prompts, `temperature=0`, `top_p=1`, `max_tokens=16`.
  - All rows finished by length, so active-row compaction did not trigger.
  - 64 tokens in 24.231s, 2.641 tok/s.
- bs=1 source anchor after the change, with no server resident:
  - `KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`
  - Prefill 2195.1ms.
  - Mean ITL 320.9ms, p50 324.8ms, p99 332.7ms.

Verdict:
- Keep.
- This is primarily a correctness and wasted-work removal for variable-length batch traffic; it will not improve the fixed-length benchmark unless some rows naturally hit EOS early.
- The single-request path remains in band with the E059/E060 source state, and native batch default behavior for max-length rows is unchanged apart from negligible active-row bookkeeping.

### E062: Native Batch Single-Row Tail Route

Change:
- Extended the E061 active-row compaction path so when only one sequence remains active, the native batch loop calls `model_forward_paged_next_token_greedy` instead of `model_forward_paged_next_tokens_greedy_batch`.
- The remaining row still uses its original block table and the compacted single-row GDN linear-attention state.
- Added a deterministic zero-layer tied-weight test where one row hits EOS immediately and the other row continues, forcing the single-active-row tail route.

Reasoning:
- E061 removes finished rows, but after compaction a variable-length batch can spend many remaining steps with `active_rows.len() == 1`.
- The native batch bridge has batch-specific overhead for block-table slicing, batched hidden tensors, batched full-attention row plumbing, and batched LM-head output handling. Once only one row remains, none of that buys throughput.
- Routing the tail to the established bs=1 greedy paged path removes that overhead and keeps using the most tuned single-stream decode implementation.

Evidence:
- Validation:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-model generate::tests::native_batch_greedy_routes_single_active_tail_after_eos -- --nocapture` passed.
  - `cargo test -p kiln-model generate::tests::native_batch_greedy_stops_empty_rows_on_eos -- --nocapture` passed.
  - `cargo test -p kiln-model generate::tests::compact_linear_attention_state_selects_active_batch_rows -- --nocapture` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- No fixed-length throughput measurement was repeated for this sub-change:
  - The standard `max_tokens=16` fixture from E061/E059 has all rows finishing by length, so it does not exercise the single-active-tail route.
  - The expected effect is only on variable-length batches where all but one row hit EOS before the global `max_tokens` cap.

Verdict:
- Keep.
- This is a narrow removal of tail work for variable-length native batches and does not affect the bs=1 request path or full-batch fixed-length decode.

### E063: Rejected Default Single-Submit Batch Fused-GDN Decode

Change:
- Added an experimental single-submit implementation for `dispatch_gdn_decode_gates_recurrent_rmsnorm`.
- The experimental path records all input staging copies, fused GDN compute, output copy, and recurrent-state copy into one command buffer and one queue submit.
- After measurement, kept it opt-in only behind `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED_SINGLE_SUBMIT=1`; the no-env default uses the E059 direct fused-GDN path.

Reasoning:
- The fresh native-batch profile after E062 showed the batch-fused GDN decode path was still the dominant decode bucket:
  - Step 0 `gdn.fused_gates_recur_norm`: 381.0ms across 24 layers.
  - Step 1: 674.4ms.
  - Step 2: 865.9ms.
- The direct dispatcher still paid separate upload/readback command-pool helpers around many small inputs and the large mutable recurrent state. A one-command-buffer implementation was a plausible way to reduce CPU/Vulkan boundary cost without changing shader math.

Evidence:
- Validation:
  - `cargo fmt --all --check` passed.
  - Focused fused-GDN parity passed: `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_decode_gates_recurrent_rmsnorm -- --nocapture`.
  - Full Vulkan GDN parity passed: `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`, 16 tests.
  - Focused native-batch generation tests passed:
    - `cargo test -p kiln-model generate::tests::native_batch_greedy_routes_single_active_tail_after_eos -- --nocapture`
    - `cargo test -p kiln-model generate::tests::native_batch_greedy_stops_empty_rows_on_eos -- --nocapture`
    - `cargo test -p kiln-model generate::tests::compact_linear_attention_state_selects_active_batch_rows -- --nocapture`
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Profile fixture, 4 distinct prompts, `temperature=0`, `top_p=1`, `max_tokens=4`:
  - Before single-submit experiment: 12.156s / 1.316 tok/s.
  - Single-submit default: 12.107s / 1.322 tok/s.
  - The profile did not show a clear bucket reduction; `gdn.fused_gates_recur_norm` shifted from 381.0/674.4/865.9ms to 525.0/706.9/697.9ms across the three decode steps.
- No-profile same-binary `max_tokens=16` fixture, same four prompts:
  - Single-submit default: 26.743s / 2.393 tok/s.
  - Disable guard for the same binary (`KILN_DISABLE_VULKAN_GDN_DECODE_FUSED_SINGLE_SUBMIT=1` before the final opt-in flip): 26.603s / 2.406 tok/s.

Verdict:
- Reject as a default.
- The experiment is left opt-in only because it preserves parity and may be useful for driver/runtime A/B work, but the default path is restored to the direct fused-GDN dispatcher.
- This result suggests the dominant batch-fused GDN cost is not just command submission count; deeper state residency or a wider fused region is still the right direction.

### E064: Embedded Batched SPIR-V Lookup Fix

Change:
- Added the missing `SHADER_SPIRVS` lookup entries for `gdn_in_proj_decode_batched` and `mlp_gate_up_decode_batched`.
- These shaders were already compiled and embedded by `build.rs`, but the runtime lookup table did not name them, so `compile_shader` could fall back to temp-file/runtime SPIR-V lookup for the batched variants.

Reasoning:
- The default native batch path uses the batched GDN input projection shader for `batch > 1`.
- Wiring the already-embedded modules into the lookup table removes an avoidable runtime fallback and makes deployment more robust if `glslc` or temp cached `.spv` files are absent.
- A companion attempt to return borrowed embedded SPIR-V instead of allocating a `Vec<u8>` was rejected immediately: `include_bytes!` is not guaranteed 4-byte aligned, and shader module creation currently casts SPIR-V bytes to `u32`.

Evidence:
- Validation:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 16 tests.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Server startup still prewarmed 249 decode weights; Vulkan pipeline prewarm completed in 2ms in the measured run.
- Same four-prompt no-profile batch fixture used during E063, `max_tokens=16`:
  - 64 completion tokens in 26.911s, 2.378 tok/s.
  - This did not beat the recent same-fixture E063 direct-path result of 26.603s / 2.406 tok/s, and it remains behind the E059 standard mixed fixture.
- bs=1 source anchor after the change:
  - Prefill 1924.0ms.
  - Mean ITL 323.1ms, p50 323.5ms, p99 336.6ms.

Verdict:
- Keep as a correctness/robustness fix, not a throughput win.
- The performance result confirms that the remaining batch cost is dominated by GDN state traffic and wider decode work, not shader lookup overhead.

### E065: Rejected Named Pipeline Key For Batch-Fused GDN

Change:
- Tried a named pipeline-cache key for `gdn_decode_gates_recurrent_rmsnorm` and routed the default direct fused-GDN decode dispatcher through it.
- The intent was to skip cloning embedded SPIR-V bytes and hashing the SPIR-V payload on every batch-fused GDN dispatch once prewarm had populated the pipeline cache.
- Reverted after measurement.

Reasoning:
- E064 showed shader lookup work is not the main bottleneck, but the batch-fused GDN path still calls the dispatcher for every GDN layer and generated token.
- A named key was a narrower alternative to borrowed SPIR-V: it preserved the existing aligned `Vec<u8>` creation on pipeline-cache miss, while cache hits could avoid both allocation and byte hashing.

Evidence:
- Validation before measurement:
  - `cargo fmt --all --check` passed after formatting.
  - Focused fused-GDN parity passed: `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_decode_gates_recurrent_rmsnorm -- --nocapture`.
  - Full Vulkan GDN parity passed: `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`, 16 tests.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Same four-prompt no-profile batch fixture used during E063/E064, `max_tokens=16`:
  - Named-key experiment: 64 completion tokens in 27.238s, 2.350 tok/s.
  - E064 source state: 26.911s, 2.378 tok/s.
  - E063 direct-path comparison: 26.603s, 2.406 tok/s.
- Validation after reverting to E064 source state:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed, 16 tests.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.

Verdict:
- Reject and revert.
- The current default should keep the simple byte-hash pipeline cache path; removing shader lookup overhead does not address the dominant GDN state-transfer/compute cost.

### E066: Final-Step Batch GDN State Readback Skip

Change:
- Added a scoped internal `with_vulkan_gdn_state_readback_skip` guard for decode calls known to be terminal because of `max_tokens`.
- In native greedy batch generation, the decode call that computes the final token now runs under this guard.
- The Vulkan fused GDN decode path still computes the updated recurrent state because it is needed for the current token's output, but skips reading that state back to a CPU tensor when the caller will not run another token.
- Added parity coverage: the skip-readback fused-GDN path must produce the same output as the normal path, while returning the original state tensor unchanged.

Reasoning:
- In the batch greedy loop, after token `N-1` is pushed, the decode call that computes token `N` is known to be the last call when `N == max_tokens`.
- The updated recurrent state from that final decode can never affect the response: the next loop iteration only pushes the sampled final token and exits.
- This directly removes work instead of trying to make it faster: one full fused-GDN recurrent-state readback across 24 linear layers is skipped for every max-token-capped native batch response.

Evidence:
- Validation:
  - `cargo fmt --all --check` passed.
  - Focused fused-GDN parity passed, including the skip-readback assertion: `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_decode_gates_recurrent_rmsnorm -- --nocapture`.
  - Full Vulkan GDN parity passed: `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`, 16 tests.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Same four-prompt no-profile batch fixture used during E063-E065, `max_tokens=16`:
  - E066: 64 completion tokens in 26.496s, 2.415 tok/s.
  - E064 source state: 26.911s, 2.378 tok/s.
  - E063 direct-path comparison: 26.603s, 2.406 tok/s.
- bs=1 source anchor after the change:
  - Prefill 1975.7ms.
  - Mean ITL 319.7ms, p50 321.8ms, p99 334.1ms.

Verdict:
- Keep.
- This is a narrow, safe batch win for max-token-capped generation and does not harm the bs=1 source anchor.
- The same idea could be extended later to final-step KV cache writes or to non-batch paged decode paths, but those need separate correctness review.

### E067: Rejected Final-Step Split GDN Recurrent State Readback Skip

Change:
- Temporarily extended the E066 final-decode readback-skip guard to the normal paged single-request decode path.
- The attempted route marked `decode_next_token_paged_interleaved` calls known to be terminal by `max_tokens`, threaded a `skip_state_readback` flag through the split `dispatch_gdn_recurrent_step` kernel, and avoided assigning the returned recurrent state in the Vulkan backend when the guard was active.
- Added temporary recurrent-step parity coverage: the skip-readback path had to produce the same output as the normal path while returning the original state tensor unchanged.
- Reverted after measurement.

Reasoning:
- The default bs=1 path does not use the fused GDN decode kernel unless explicitly enabled; it uses the split `gdn_recurrent_step` path.
- If a final-by-`max_tokens` decode call is known before sampling, the updated GDN recurrent state from that call should be dead just like in E066's batch-fused case.
- This was a narrow "remove work" test for bs=1 without changing GDN math or persistent state semantics outside the final token.

Evidence:
- Validation before measurement:
  - `cargo fmt --all --check` passed.
  - Focused recurrent parity passed with skip assertions: `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_recurrent_step -- --nocapture`.
  - Full Vulkan GDN parity passed: `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`, 16 tests.
  - The non-single-submit recurrent path also passed the focused parity test with `KILN_DISABLE_VULKAN_GDN_RECURRENT_SINGLE_SUBMIT=1`.
  - Model-level max-token and batch routing tests passed:
    - `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture`
    - `cargo test -p kiln-model generate::tests::test_generate_paged_shared_max_tokens -- --nocapture`
    - `cargo test -p kiln-model native_batch_greedy -- --nocapture`
    - `cargo test -p kiln-model compact_linear_attention_state_selects_active_batch_rows -- --nocapture`
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- bs=1 source anchors with the experiment:
  - Run 1: prefill 2161.5ms, mean ITL 331.9ms, p50 337.0ms, p99 341.5ms.
  - Run 2: prefill 2127.3ms, mean ITL 330.0ms, p50 330.1ms, p99 347.2ms.
  - The final decode step did not show a benefit: 337.3ms and 347.2ms in the two runs.
- Validation after reverting:
  - `cargo fmt --all --check` passed.
  - Full Vulkan GDN parity passed: `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`, 16 tests.
  - `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Post-revert bs=1 source anchor on the kept code path:
  - Prefill 2097.1ms.
  - Mean ITL 330.9ms, p50 331.5ms, p99 338.7ms.

Verdict:
- Reject and revert.
- The post-revert anchor shows the 330ms measurements were environmental/run noise rather than a clear E067 regression, but the experiment still failed to create a measurable win.
- The useful direction for bs=1 remains deeper GDN recurrent state residency or a wider fused region, not skipping only the final split-state readback.

### E068: Current Recurrent-Path Profile And Escape-Hatch Retest

Change:
- No code change.
- Re-profiled the final kept source state and retested the existing recurrent-path escape hatches under the same noisy runtime conditions seen around E067.

Reasoning:
- E067 did not produce a win, but the post-revert run showed the machine was currently measuring around 330ms mean ITL instead of the earlier 319-321ms band.
- Before trying a larger recurrent-state rewrite, the current profile and existing env guards should confirm the dominant bucket and whether any already-implemented alternative path deserves promotion.

Evidence:
- Short profiled bs=1 run:
  - Command: `KILN_PROFILE_DECODE=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 3 --skip-training`.
  - Prefill 2124.9ms.
  - Mean ITL 329.6ms, p50 329.7ms, p99 331.6ms.
  - `gdn.recurrent`: 183.7ms, 180.5ms, 178.3ms across 24 calls on the three decode steps.
  - Next buckets: `mlp.fused` 55.4-55.7-54.6ms/32, `layer.full` 27.3-26.6-26.3ms/8, `gdn.in_proj` 25.0-24.7-24.0ms/24, `gdn.gated_norm` 18.4-18.1-20.4ms/24, `lm_head.argmax` 14.5-16.3-16.9ms/1.
- Existing recurrent state-buffer guard:
  - `KILN_DISABLE_VULKAN_GDN_RECURRENT_HOST_VISIBLE_STATE=1`: prefill 3992.6ms, mean ITL 329.7ms, p50 332.4ms, p99 340.0ms.
  - Result: neutral for decode and worse for prefill; no default flip.
- Existing recurrent submit guard:
  - `KILN_DISABLE_VULKAN_GDN_RECURRENT_SINGLE_SUBMIT=1`: prefill 3881.6ms, mean ITL 332.4ms, p50 336.8ms, p99 340.3ms.
  - Result: slower; keep single-submit default.
- Existing bs=1 fused-GDN opt-in:
  - `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED=1`: prefill 4136.4ms, mean ITL 330.4ms, p50 330.0ms, p99 338.8ms.
  - Result: neutral in this run and historically unstable; do not promote.

Verdict:
- Keep defaults unchanged.
- The current profile again points at split GDN recurrent state traffic/compute as the main bs=1 bottleneck.
- The next source-level candidate should be a real recurrent-state residency or shader-algorithm change; the existing guards do not expose another small default win.

### E069: Batch-Local Common Prefix Reuse

Change:
- Added a native-batch-only local prefix reuse path for greedy Vulkan batches where all rows miss the external prefix cache but share an identical block-aligned prompt prefix.
- The shared prefix is prefetched once into shared KV blocks and a linear-attention state snapshot; each row then prefills only its suffix before the rows decode together.
- Added an 8-block minimum shared-prefix threshold after A/B testing showed small shared prefixes regress.
- Added rollback guard `KILN_DISABLE_VULKAN_BATCH_LOCAL_PREFIX_REUSE=1`.
- Added targeted tests for the common-prefix threshold and shared-prefix block allocation in native batch greedy generation.

Reasoning:
- This follows the "remove the need to do it at all" principle for a common server traffic shape: multiple requests in one batch can share a long system/developer/user preamble even when the global prefix cache has no reusable entry yet.
- Reusing one in-flight batch prefix should avoid duplicate prefill work without mutating external prefix-cache stats/refcounts and without changing bs=1 generation.
- The state snapshot cost and extra prefix/suffix split overhead are only worthwhile for long prefixes, so the final implementation requires at least eight full shared blocks.

Evidence:
- Validation:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-model batch_local_prefix_reuse_requires_minimum_shared_blocks -- --nocapture` passed.
  - `cargo test -p kiln-model native_batch_greedy_reuses_common_block_prefix -- --nocapture` passed.
  - `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed, 3 tests.
  - `cargo test -p kiln-model compact_linear_attention_state_selects_active_batch_rows -- --nocapture` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Short/common-prefix guard finding:
  - Initial unthresholded 4-row fixture with 209 prompt tokens total and only 32 shared tokens triggered local reuse and regressed: 17.128s / 0.934 tok/s versus rollback guard 13.913s / 1.150 tok/s.
  - After adding the 8-block threshold, the same fixture no longer triggered local reuse (`cached_tokens=0` on all rows) and measured 14.213s / 1.126 tok/s, in the same noisy band as the guarded run.
- Long common-prefix win:
  - Fixture: 4 rows, 1757 prompt tokens total, 8 completion tokens, 416 shared prompt tokens / 26 shared blocks.
  - Rollback guard `KILN_DISABLE_VULKAN_BATCH_LOCAL_PREFIX_REUSE=1`: 39.222s / 0.204 tok/s; row prefills were 9486.6ms, 9688.8ms, 9675.5ms, and 9791.5ms.
  - Thresholded no-env path: 17.869s / 0.448 tok/s; common prefix prefill was 9030.0ms, suffix prefills were 2060.7ms, 2036.9ms, 2059.6ms, and 2093.5ms.
  - This is a 2.20x throughput improvement for long shared-prefix all-miss native batch traffic.
- The standard short all-miss batch fixture did not share a full block-aligned prefix and therefore did not trigger this path (`cached_tokens=0` rows); the observed 11.792s / 1.357 tok/s was not attributable to local prefix reuse.

Verdict:
- Keep.
- The final threshold preserves the long shared-prefix win while avoiding the measured small-prefix regression.
- This does not change bs=1 behavior and composes with the external prefix cache: it only runs when every row misses the external cache.

### E070: Rejected Prefix-Only No-Logits Prefill

Change:
- Temporarily exposed `LmHeadMode::Skip` as a paged no-logits forward helper.
- Used it only for E069's batch-local common-prefix prefill, where the prefix's next-token logits are immediately discarded because every row has a non-empty suffix.
- Reverted after measurement.

Reasoning:
- E069 still spent about 9.0s on the common-prefix prefill in the 416-token shared-prefix fixture.
- The prefix forward must still run all model layers to fill KV blocks and advance GDN linear state, but it should not need final RMSNorm, LM-head projection, or argmax for the prefix-only segment.
- This was a narrow "remove unused output" test with the same long-prefix fixture as E069.

Evidence:
- Validation before measurement:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-model native_batch_greedy_reuses_common_block_prefix -- --nocapture` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Long common-prefix fixture:
  - E069 thresholded baseline: 17.869s / 0.448 tok/s; common prefix prefill 9030.0ms.
  - E070 no-logits experiment: 18.223s / 0.439 tok/s; common prefix prefill 9173.1ms.

Verdict:
- Reject and revert.
- Skipping the prefix LM-head/argmax is below the measurement noise here and did not reduce the dominant layer-loop cost.
- The kept path remains E069's thresholded local prefix reuse.

### E071: Final Kept-Branch bs=1 Anchor After Batch-Prefix Work

Change:
- No code change.
- Re-ran the standard short bs=1 source benchmark after E069 was kept and E070 was reverted, to verify the batch-only prefix work did not disturb single-stream latency.

Evidence:
- Command: `KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`.
- Prefill: 1900.5ms.
- Decode steps: 323.6ms, 316.7ms, 319.4ms, 350.4ms, 327.0ms, 338.2ms.
- Mean ITL: 329.2ms, p50 327.0ms, p99 350.4ms.

Verdict:
- No bs=1 regression signal from the kept E069 branch state.
- This remains in the same noisy ~329-331ms band observed in E067-E068, while the best recent source anchor is still E059's 318.1ms.

### E072: Register Batch-Local Common Prefixes In The External Prefix Cache

Change:
- Extended native batch output with an optional shared-prefix registration produced by E069's batch-local common-prefix prefill.
- The server now registers that shared prefix in the normal prefix cache before row-level prompt registrations after a successful native batch response.
- The registration owns the shared prefix KV blocks and the matching linear-attention state snapshot, so later batches with the same long prefix but different suffixes can hit the external cache instead of recomputing the common prefix.

Reasoning:
- E069 removes duplicate prefill inside one all-miss batch, but row prompt lengths are often not block-aligned and therefore do not necessarily register anything reusable for later batches.
- The common prefix itself is already block-aligned by construction and has a complete KV/linear-state snapshot.
- Registering it turns the first long shared-prefix all-miss batch into a warmup for later traffic with the same system/developer/user preamble.

Evidence:
- Validation:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-model native_batch_greedy_reuses_common_block_prefix -- --nocapture` passed and now asserts a shared-prefix registration with 32 tokens / 8 blocks in the unit fixture.
  - `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed, 3 tests.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Fresh-server two-batch long shared-prefix fixture:
  - Batch 1, cache empty: 18.124s / 0.441 tok/s. Logs confirmed E069 local prefix prefill: `cached_tokens=416`, 26 blocks, 9090.4ms common-prefix prefill, then 23-24 token suffix prefills.
  - Batch 2, same 416-token shared prefix with different suffixes: 9.045s / 0.884 tok/s. Logs showed no local-prefix prefill; each row arrived as an external prefix-cache hit with `cached_tokens=416` and only 23-25 suffix tokens.
  - Relative to the E069 first-batch baseline, repeated long-prefix traffic is about 2.0x faster; relative to the E069 rollback guard from E069 (39.222s / 0.204 tok/s), the warmed second batch is about 4.34x faster.

Verdict:
- Keep.
- This removes repeated long-prefix prefill across batches while preserving E069's first-batch all-miss win.
- Rollback for a clean server remains `KILN_DISABLE_VULKAN_BATCH_LOCAL_PREFIX_REUSE=1`, which prevents the batch-local shared-prefix prefill and therefore prevents this registration path from producing new shared-prefix entries.

### E073: Rejected Vulkan GDN Recurrent 4D Input Path

Change:
- Temporarily let the Vulkan split GDN recurrent step accept the existing `[B,H,1,D]` decode tensors directly.
- Added an opt-in `KILN_ENABLE_VULKAN_GDN_RECURRENT_4D_INPUTS=1` branch in the decode recurrence path to skip the five `squeeze(2)?.contiguous()?` calls before dispatch.
- Added temporary parity coverage that compared 4D recurrent dispatch against the existing 3D recurrent dispatch and CPU reference.
- Reverted after measurement.

Reasoning:
- The bs=1 profile still points at split GDN recurrent work as the dominant decode bucket.
- The recurrent shader reads flattened Q/K/V/beta/g buffers; for a singleton sequence dimension, `[B,H,1,D]` has the same flattened layout as `[B,H,D]`.
- If the squeeze/contiguous preamble was material enough, passing the tensors directly should remove avoidable per-layer copy work.

Evidence:
- Validation before measurement:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_recurrent_step_matches_cpu_reference -- --nocapture` passed with the temporary 4D parity assertions.
  - `cargo build --release --features vulkan --bin kiln-bench` passed.
- bs=1 opt-in run:
  - Command: `KILN_ENABLE_VULKAN_GDN_RECURRENT_4D_INPUTS=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`.
  - Prefill 1811.1ms.
  - Mean ITL 330.3ms, p50 328.7ms, p99 338.3ms.
  - Current kept-code anchor from E071 was 329.2ms mean ITL under the same noisy conditions.

Verdict:
- Reject and revert.
- Avoiding the explicit squeeze/contiguous calls did not move the decode anchor; the dominant cost remains the recurrent dispatch/upload/readback itself rather than that shape preamble.

### E074: Rejected Host-Visible Split GDN Recurrent Output Buffer

Change:
- Temporarily added `KILN_ENABLE_VULKAN_GDN_RECURRENT_HOST_VISIBLE_OUT=1`.
- In the single-submit split GDN recurrent path, the experiment made the tiny recurrent output buffer host-visible and skipped the explicit output copy into a staging buffer.
- Reverted after A/B measurement.

Reasoning:
- The split GDN recurrent output is small, but it is read back every linear layer and decode token.
- The state buffer is already host-visible by default on Strix Halo; making the output host-visible might remove one copy command per recurrent dispatch.

Evidence:
- Validation before measurement:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_recurrent_step_matches_cpu_reference -- --nocapture` passed.
  - `cargo build --release --features vulkan --bin kiln-bench` passed.
- bs=1 A/B on the same binary:
  - Opt-in run 1: prefill 1840.8ms, mean ITL 322.5ms, p50 325.5ms, p99 339.4ms.
  - No-env comparison: prefill 1860.4ms, mean ITL 322.8ms, p50 322.5ms, p99 334.0ms.
  - Opt-in confirmation: prefill 1874.9ms, mean ITL 327.6ms, p50 328.1ms, p99 336.1ms.

Verdict:
- Reject and revert.
- The apparent first-run improvement was environmental; same-binary no-env matched it, and confirmation was worse.
- The output copy is not the limiting piece of the recurrent path.

### E075: Rejected Row-Aligned Prefix Registration Split

Change:
- Temporarily added `KILN_ENABLE_VULKAN_BATCH_ROW_ALIGNED_PREFIX_REGISTRATION=1`.
- For native batch rows that had a cached/shared prefix but whose full prompt ended off a block boundary, the experiment split the suffix prefill at the next block boundary, snapshotted linear-attention state there, and registered that longer row-specific prefix.
- Reverted after measurement.

Reasoning:
- E072 registers the long shared prefix across batches, but repeated exact prompts still only hit that shared prefix when the full prompt length is not block-aligned.
- In the long-prefix fixture, rows with 439-441 prompt tokens hit the 416-token shared prefix, then still prefill 23-25 suffix tokens on every repeat.
- Registering a 432-token per-row prefix could reduce repeated exact-prompt suffix prefill to 7-9 tokens.

Evidence:
- Validation before measurement:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-model row_aligned_prefix_registration_len_leaves_tail_after_cache -- --nocapture` passed.
  - `cargo test -p kiln-model native_batch_greedy_reuses_common_block_prefix -- --nocapture` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Fresh-server three-batch long shared-prefix fixture with the opt-in:
  - Batch 1, cache empty: 24.985s / 0.320 tok/s. It did the E069 416-token common prefix, then split each row suffix; final tail prefills were only 7-8 tokens, but per-row elapsed was still ~3.82-3.88s because each row now had two prefill calls.
  - Batch 2, same shared prefix with new suffixes: 15.427s / 0.519 tok/s. It hit the 416-token shared prefix but paid split-registration overhead to seed 432-token row prefixes.
  - Batch 3, identical to batch 2: 7.893s / 1.014 tok/s. It hit 432 cached tokens per row and only prefetched 7-9 tail tokens.
- Comparison:
  - E072's repeated-prefix second-batch baseline was 9.045s / 0.884 tok/s with 416 cached tokens per row.
  - The third batch is faster than E072 by about 1.15s, but the second batch is about 6.38s slower than E072; this only breaks even after many repeated exact/near-exact batches with the same first 432 tokens.

Verdict:
- Reject and revert.
- The repeated exact-prompt speedup is real, but the registration cost is too high and the traffic shape too narrow for a default path.
- If needed later, a safer variant would need a much stronger repetition heuristic or an asynchronous/off-critical-path registration strategy.

### E076: Batch-Local Prefix Threshold Sanity Check

Change:
- No code change.
- Re-tested E069's default `BATCH_LOCAL_PREFIX_MIN_BLOCKS = 8` guard with a medium shared-prefix fixture that lands exactly at the threshold.

Reasoning:
- E069 showed that a very small shared prefix regressed before thresholding, while a 416-token shared prefix was a large win.
- The remaining risk was that the 8-block cutoff might still be too low and should be raised.
- A prompt fixture with 3 repeats of the shared-context phrase produced a 128-token shared prefix, exactly 8 blocks with the current 16-token block size.

Evidence:
- Same release server binary as the kept branch, first with default batch-local prefix reuse, then with `KILN_DISABLE_VULKAN_BATCH_LOCAL_PREFIX_REUSE=1`.
- Default no-env run:
  - Request: four distinct prompts, `max_tokens=2`, 609 total prompt tokens, 8 completion tokens.
  - Server log: `batch greedy local prefix prefill: rows=4 cached_tokens=128 blocks=8 elapsed_ms=3929.0`.
  - Per-row suffix prefills after the shared prefix: 24, 24, 25, and 24 tokens; elapsed 2106.3ms, 2123.3ms, 2330.0ms, and 2108.9ms.
  - Total: 13.153s / 0.608 tok/s; server batch total 13137.8ms.
- Rollback guard run:
  - Env: `KILN_DISABLE_VULKAN_BATCH_LOCAL_PREFIX_REUSE=1`.
  - Each row prefetched the full 152-153-token prompt; elapsed 4185.6ms, 4263.1ms, 4325.5ms, and 4355.4ms.
  - Total: 17.690s / 0.452 tok/s; server batch total 17670.5ms.

Verdict:
- Keep the 8-block threshold unchanged.
- The exact-threshold fixture was about 4.54s faster than the rollback guard, so raising the guard would discard a useful win.

### E077: Opt-In Resident Split GDN Recurrent State

Change:
- Added experimental `KILN_ENABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE=1`.
- The split GDN recurrent step can now keep each mutable recurrent state in a device-local Vulkan buffer keyed by the CPU tensor id.
- The first decode step uploads the CPU state into that resident buffer; later decode steps reuse it and only read back the small recurrent output.
- The resident state cache is now thread-local and scoped to normal decode loops, with cache clear on scope exit. Speculative paths do not enter the scope yet.
- Added `gdn_recurrent_resident_state_matches_two_step_reference` to verify that a second resident-state recurrent step matches the normal CPU-visible state path.

Reasoning:
- The fresh E077 pre-change decode profile still showed `gdn.recurrent` dominating bs=1 decode:
  - Current no-env profile, `max_output_tokens=4`: prefill 1863.6ms, mean ITL 327.7ms.
  - `gdn.recurrent` totals were 176.9ms, 174.3ms, 176.4ms, and 175.1ms across 24 linear layers.
- The recurrent output is tiny, but the mutable recurrent state is large and was uploaded/read back at every linear layer and decode token.
- Avoiding that state roundtrip is the highest-leverage way to remove work rather than tuning around it.

Evidence:
- Validation:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_recurrent_step_matches_cpu_reference -- --nocapture` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_recurrent_resident_state_matches_two_step_reference -- --nocapture` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Same rebuilt binary, default no-env bs=1 short source bench before scoping:
  - Command: `KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`.
  - Prefill 1831.0ms.
  - Mean ITL 321.0ms, p50 322.7ms, p99 328.9ms.
- Scoped-cache confirmation on the same fixture:
  - Default no-env: prefill 1841.3ms; mean ITL 318.8ms, p50 316.7ms, p99 331.7ms.
  - Opt-in resident-state: prefill 1892.4ms; mean ITL 134.5ms, p50 133.5ms, p99 142.7ms.
- Initial opt-in resident-state bs=1 short source bench:
  - Command: `KILN_ENABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`.
  - Prefill 1875.5ms.
  - Mean ITL 137.4ms, p50 135.5ms, p99 147.3ms.
- Opt-in profile, `max_output_tokens=4`:
  - Prefill 1864.2ms, mean ITL 135.5ms.
  - `gdn.recurrent` dropped to 13.6ms on the first decode step, then 6.8ms, 6.5ms, and 6.2ms once resident buffers were hot.
  - The new dominant buckets are `mlp.fused` at about 54ms/32 layers, full-attention layers at about 24-25ms/8 layers, GDN input projection at about 20-22ms/24 layers, and LM-head argmax at about 14-17ms.

Verdict:
- Superseded by E078 promotion.
- The latency win is real and large. Scoping removes the obvious backend-global leak/stale-cache problem for normal decode loops.
- Before enabling the path in speculative decode, resident state ownership should move into `LinearAttentionState` or gain explicit materialization hooks so snapshots/rollback inside a resident scope cannot observe stale CPU state.

### E078: Promote Scoped Resident Split GDN Recurrent State

Change:
- Promoted the scoped resident split GDN recurrent state path to default on Vulkan.
- Added rollback env `KILN_DISABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE=1`.
- The old opt-in env is no longer required; normal decode loops enter a thread-local resident-state scope by default and clear cached state buffers on scope exit.

Reasoning:
- E077 reduced the dominant bs=1 bucket by removing the full recurrent-state upload/readback loop.
- The scope guard means normal decode no longer leaves resident buffers in a backend-global cache after the request.
- The remaining stale CPU-state risk is isolated to code that would snapshot/rollback within a resident scope; speculative paths do not enter the scope yet, and prefix-cache registrations happen outside the decode scope.

Evidence:
- Validation:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_recurrent_resident_state_matches_two_step_reference -- --nocapture` passed.
  - `cargo check -p kiln-server --features vulkan` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
  - `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
- Same rebuilt binary A/B:
  - Default no-env: prefill 1901.8ms; mean ITL 137.6ms, p50 135.2ms, p99 150.1ms.
  - Rollback guard `KILN_DISABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE=1`: prefill 1912.7ms; mean ITL 319.3ms, p50 323.2ms, p99 325.4ms.
- Token-id parity check on the same short fixture:
  - Default generated first token ids `[0,0,0,0,0,0,0]`, mean ITL 140.4ms.
  - Rollback guard generated first token ids `[0,0,0,0,0,0,0]`, mean ITL 333.2ms.

Verdict:
- Keep as default.
- This is the new default bs=1 source anchor: 137.6ms mean ITL in the main A/B, with 140.4ms in the token-parity run.
- Roll back with `KILN_DISABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE=1`.

### E079: Reject MLP Gate/Up Tuning and Single-Submit MLP Decode

Change:
- Temporarily changed the bs=1 `mlp_gate_up_decode` shader from a 16-column x 16-reduction-lane workgroup to 8 columns x 32 reduction lanes, with matching workgroup dispatch math.
- Temporarily added an opt-in `KILN_ENABLE_VULKAN_MLP_DECODE_SINGLE_SUBMIT=1` path that recorded x upload, gate/up dispatch, hidden-activation barrier, down dispatch, and final output copy in one command buffer.
- Reverted both after measurement; no code from this experiment is kept.

Reasoning:
- The fresh post-E078 profile moved the largest bs=1 bucket from GDN recurrent state to MLP:
  - Default profile, `max_output_tokens=4`: prefill 1918.1ms, mean ITL 136.5ms.
  - `mlp.fused`: 54.0-55.3ms across 32 layers.
  - Next buckets: full-attention layers 24.8-25.4ms/8, `gdn.in_proj` 20.3-21.2ms/24, `lm_head.argmax` 12.6-15.2ms/1.
- Splitting the MLP path with existing env gates showed the gate/up half was the larger sub-bucket, though that split path is slower overall because it reads back the hidden activation:
  - `KILN_DISABLE_VULKAN_MLP_DECODE=1 KILN_ENABLE_VULKAN_MLP_GATE_UP=1`, `max_output_tokens=3`: mean ITL 149.1ms.
  - `mlp.gate_up`: 43.0-43.1ms/32.
  - `mlp.down`: 19.8-20.3ms/32.

Evidence:
- 8x32 gate/up shader:
  - `cargo fmt --all --check` passed before the trial.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode -- --nocapture` passed.
  - `cargo build --release --features vulkan --bin kiln-bench` passed.
  - Profiled run regressed to prefill 1819.7ms, mean ITL 153.0ms.
  - `mlp.fused` worsened to 64.9-67.6ms/32.
- Opt-in MLP single-submit:
  - `KILN_ENABLE_VULKAN_MLP_DECODE_SINGLE_SUBMIT=1 cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode -- --nocapture` passed.
  - `cargo build --release --features vulkan --bin kiln-bench` passed.
  - Profiled opt-in run: prefill 1842.7ms, mean ITL 139.8ms, `mlp.fused` 51.9-52.1ms/32.
  - Same-binary no-profile A/B:
    - Opt-in single-submit: prefill 1872.5ms; mean ITL 142.6ms, p50 141.0ms, p99 159.0ms.
    - No-env default: prefill 1861.5ms; mean ITL 142.7ms, p50 142.6ms, p99 153.0ms.

Verdict:
- Reject and remove.
- The shader workgroup shape is a clear regression.
- MLP single-submit slightly lowers the profiled MLP bucket, but not enough to move whole-token latency; it should not be promoted or carried as a runtime branch.
- Future MLP work likely needs a real shader-level algorithm improvement or broader layer residency, not submit-count reduction around the existing two kernels.

### E080: Retest Existing Full-Attn QKV and Fused GDN Decode Opt-Ins After Resident State

Change:
- No code change.
- Retested the existing `KILN_ENABLE_VULKAN_FULL_ATTN_QKV=1` and `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED=1` paths against the new resident-state default.

Reasoning:
- E078 changed the bottleneck order enough that old rejected opt-ins deserved a quick same-source check.
- Full-attention layers were the second largest post-E078 bucket after MLP.
- The fused GDN decode path previously had transient bs=1 wins, but the current default now depends on scoped resident split recurrent state.

Evidence:
- Full-attn QKV opt-in:
  - Command: `KILN_ENABLE_VULKAN_FULL_ATTN_QKV=1 KILN_PROFILE_DECODE=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 4 --skip-training`.
  - Prefill 1902.9ms; mean ITL 194.6ms, p50 195.1ms, p99 197.2ms.
  - `layer.full` worsened to 74.1-85.1ms/8 versus the post-E078 default profile's 24.8-25.4ms/8.
- Fused GDN decode opt-in:
  - Command: `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED=1 KILN_PROFILE_DECODE=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 4 --skip-training`.
  - Prefill 1900.2ms; mean ITL 328.6ms, p50 328.2ms, p99 348.5ms.
  - `gdn.fused_gates_recur_norm` was 171.8-205.7ms/24; this path bypasses the resident split recurrent-state win and falls back near the old recurrent bottleneck.

Verdict:
- Reject both for default promotion.
- Keep the current resident split GDN path as the bs=1 default.
- Full-attention work needs something deeper than the existing projection-only QKV fusion.
- Fused GDN decode would need resident-state semantics before it can compete with split resident recurrence.

### E081: Promote Single-Submit Vulkan LM-Head Argmax

Change:
- Added a single-submit path for `dispatch_linear_decode_argmax_cached`.
- The new path records x upload, block-argmax dispatch, block-result barrier, reduce dispatch, final-index copy, and scalar readback behind one queue submit.
- Promoted it to default with rollback guard `KILN_DISABLE_VULKAN_LINEAR_ARGMAX_SINGLE_SUBMIT=1`.
- Increased the reusable transient descriptor pool to allow multiple descriptor sets in a fused command buffer.

Reasoning:
- After E078, `lm_head.argmax` is a smaller but still visible bucket at roughly 13-17ms per decode token.
- The existing argmax path used separate upload, block dispatch, reduce dispatch, and readback submits.
- Unlike the MLP single-submit attempt, this path has little compute between submit boundaries and only reads back a 4-byte token id, so submit consolidation has a clearer chance to help.

Evidence:
- Opt-in trial:
  - `KILN_ENABLE_VULKAN_LINEAR_ARGMAX_SINGLE_SUBMIT=1 cargo test -p kiln-vulkan-kernel --test gdn_parity linear_decode_argmax_matches_cpu_reference -- --nocapture` passed.
  - `cargo build --release --features vulkan --bin kiln-bench` passed.
  - Profiled opt-in run: prefill 2028.5ms; mean ITL 142.3ms, p50 140.8ms, p99 155.8ms. `lm_head.argmax` remained in the noisy 12.8-17.1ms range.
  - Same-binary no-profile A/B before promotion:
    - Opt-in: prefill 1811.3ms; mean ITL 140.7ms, p50 139.1ms, p99 149.2ms.
    - No-env default: prefill 1818.6ms; mean ITL 142.4ms, p50 142.0ms, p99 149.1ms.
  - Opt-in confirmation: prefill 1841.4ms; mean ITL 140.4ms, p50 138.9ms, p99 151.6ms.
- Promoted default:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity linear_decode_argmax_matches_cpu_reference -- --nocapture` passed.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
  - Default no-env: prefill 1863.0ms; mean ITL 140.7ms, p50 139.8ms, p99 146.7ms.
  - Rollback guard `KILN_DISABLE_VULKAN_LINEAR_ARGMAX_SINGLE_SUBMIT=1`: prefill 1822.9ms; mean ITL 143.1ms, p50 138.6ms, p99 165.4ms.

Verdict:
- Keep default with rollback guard.
- This is a small same-binary win, not a new best source anchor; E078's 137.6ms remains the best verified bs=1 anchor.

### E082: Promote Single-Submit Vulkan GDN Input Projection

Change:
- Added a single-submit path for `dispatch_gdn_in_proj_decode_cached`.
- The path records x upload, fused input-projection dispatch, output copy, and readback through one command buffer.
- Promoted it to default with rollback guard `KILN_DISABLE_VULKAN_GDN_IN_PROJ_SINGLE_SUBMIT=1`.
- Shared the output slicing helper between the old direct path and the single-submit path.

Reasoning:
- After E081, the fresh bs=1 profile still showed `gdn.in_proj` at about 24ms across 24 GDN layers.
- E044 had rejected GDN in-proj single-submit before resident recurrent state, but the latency budget and bottleneck order changed enough to retest it.

Evidence:
- Fresh post-E081 default profile:
  - Command: `KILN_PROFILE_DECODE=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 4 --skip-training`.
  - Prefill 1854.5ms; mean ITL 140.6ms, p50 139.6ms, p99 146.6ms.
  - `gdn.in_proj` was 23.7-24.0ms/24.
- Opt-in trial:
  - `KILN_ENABLE_VULKAN_GDN_IN_PROJ_SINGLE_SUBMIT=1 cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_in_proj -- --nocapture` passed, including batch=1 and batched parity.
  - `cargo build --release --features vulkan --bin kiln-bench` passed.
  - Profiled opt-in run: prefill 1967.0ms; mean ITL 143.8ms, p50 142.2ms, p99 154.8ms. `gdn.in_proj` fell slightly to 22.5-23.2ms/24, but whole-token profiled latency was noisy and worse.
  - No-profile same-binary A/B:
    - Opt-in: prefill 1911.2ms; mean ITL 136.7ms, p50 135.5ms, p99 149.2ms.
    - No-env default: prefill 1917.6ms; mean ITL 141.8ms, p50 142.3ms, p99 151.4ms.
  - Opt-in confirmation: prefill 2005.4ms; mean ITL 138.3ms, p50 137.4ms, p99 146.4ms.
- Promoted default:
  - `cargo fmt --all --check` passed.
  - `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_in_proj -- --nocapture` passed, including batch=1 and batched parity.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
  - `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
  - Default no-env: prefill 1910.5ms; mean ITL 137.2ms, p50 136.8ms, p99 141.8ms.
  - Rollback guard `KILN_DISABLE_VULKAN_GDN_IN_PROJ_SINGLE_SUBMIT=1`: prefill 1886.6ms; mean ITL 141.4ms, p50 137.2ms, p99 168.4ms.

Verdict:
- Keep default with rollback guard.
- This is the new best verified no-env default bs=1 source anchor: 137.2ms mean ITL.

### E083: Tune Vulkan MLP Gate/Up Decode Workgroup Shape

Change:
- Changed the single-token MLP gate/up decode shader from 16 output columns x 16 reduction lanes to 32 output columns x 8 reduction lanes.
- Updated batch=1 dispatch sizing for `dispatch_mlp_gate_up_decode_cached` and `dispatch_mlp_decode_cached` to cover 32 intermediate columns per workgroup.
- Left the batched MLP gate/up path unchanged at the existing 16-wide layout.

Reasoning:
- E079's rejected 8x32 shape gave each column too many reduction lanes and regressed `mlp.fused`.
- After resident recurrent state and GDN in-proj single-submit, MLP is again the largest visible decode bucket.
- The 32x8 shape keeps 256 invocations per workgroup but increases output-column parallelism while leaving enough reduction lanes for hidden-size accumulation.

Evidence:
- `cargo fmt --all --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode -- --nocapture` passed, including batch=1 and batched MLP parity.
- `cargo build --release --features vulkan --bin kiln-bench` passed before latency measurement.
- Final validation passed:
  - `git diff --check`.
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
  - `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture`.
- Profiled run:
  - Command: `KILN_PROFILE_DECODE=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 4 --skip-training`.
  - Prefill 1922.5ms; mean ITL 132.0ms, p50 129.5ms, p99 143.1ms.
  - `mlp.fused` fell to 49.1-49.4ms/32 layers, down from roughly 54ms in the post-E082 profile.
- No-profile default runs:
  - First: prefill 1923.7ms; mean ITL 133.7ms, p50 134.7ms, p99 141.0ms.
  - Confirmation: prefill 1959.2ms; mean ITL 136.4ms, p50 134.8ms, p99 146.4ms.

Verdict:
- Keep.
- This is the new best verified no-env default bs=1 source anchor: 133.7ms mean ITL, with a confirmation run at 136.4ms.

### E084: Tune Vulkan MLP Gate/Up Decode Workgroups To 64x4

Change:
- Changed the single-token MLP gate/up decode shader from 32 output columns x 8 reduction lanes to 64 output columns x 4 reduction lanes.
- Updated batch=1 dispatch sizing for `dispatch_mlp_gate_up_decode_cached` and `dispatch_mlp_decode_cached` to cover 64 intermediate columns per workgroup.
- Left the batched MLP gate/up path unchanged.

Reasoning:
- E083 confirmed that increasing output-column parallelism was the right direction for this shader.
- The 64x4 shape still uses 256 invocations per workgroup and keeps the same shared-memory footprint, but halves the number of MLP gate/up workgroups again.
- This trades fewer reductions and launches across intermediate columns for longer per-lane hidden loops.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode -- --nocapture` passed, including batch=1 and batched MLP parity.
- `cargo build --release --features vulkan --bin kiln-bench` passed before latency measurement.
- Final validation passed:
  - `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
  - `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture`.
- Profiled run:
  - Command: `KILN_PROFILE_DECODE=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 4 --skip-training`.
  - Prefill 1875.5ms; mean ITL 131.5ms, p50 130.4ms, p99 141.1ms.
  - `mlp.fused` fell to 46.8-47.6ms/32 layers, down from E083's 49.1-49.4ms.
- No-profile default runs:
  - First: prefill 1926.7ms; mean ITL 130.2ms, p50 129.5ms, p99 139.0ms.
  - Confirmation: prefill 1882.1ms; mean ITL 129.1ms, p50 126.9ms, p99 137.8ms.

Verdict:
- Keep.
- This supersedes E083 as the best verified no-env default bs=1 source anchor: 129.1ms mean ITL, with a second no-profile run at 130.2ms.

### E085: Reject 128x2 Vulkan MLP Gate/Up Decode Workgroups

Change:
- Temporarily changed the single-token MLP gate/up decode shader from 64 output columns x 4 reduction lanes to 128 output columns x 2 reduction lanes.
- Temporarily updated batch=1 dispatch sizing to cover 128 intermediate columns per workgroup.
- Reverted both changes after measurement.

Reasoning:
- E083 and E084 showed that increasing output-column parallelism helped, but 128x2 tests whether reducing the reduction-lane count further starves hidden-dimension accumulation.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode -- --nocapture` passed, including batch=1 and batched MLP parity.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- Profiled run:
  - Command: `KILN_PROFILE_DECODE=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 4 --skip-training`.
  - Prefill 1865.2ms; mean ITL 133.7ms, p50 133.2ms, p99 141.4ms.
  - `mlp.fused` was 47.6-48.8ms/32 layers, worse than E084's 46.8-47.6ms.

Verdict:
- Reject and revert.
- 64x4 remains the best measured single-token MLP gate/up workgroup shape.

### E086: Reject 32x8 Generic Vulkan Linear Decode Workgroups

Change:
- Temporarily changed the single-token generic `linear_decode` shader from 16 output columns x 16 reduction lanes to 32 output columns x 8 reduction lanes.
- Temporarily updated the batch=1 dispatch counts for direct, single-submit, and MLP-down uses of `linear_decode`.
- Reverted all changes after measurement.

Reasoning:
- After E084, `mlp.fused`, `layer.full`, and GDN output projection remain visible profile buckets.
- These paths share the generic single-token `linear_decode` shader, so the MLP gate/up output-column-parallel tuning direction was worth testing on generic projections.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed all 17 Vulkan parity tests.
- `cargo build --release --features vulkan --bin kiln-bench` passed.
- Profiled run:
  - Command: `KILN_PROFILE_DECODE=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 4 --skip-training`.
  - Prefill 1864.4ms; mean ITL 141.5ms, p50 140.7ms, p99 150.9ms.
  - `mlp.fused` regressed to 51.3-51.8ms/32 layers.
  - `layer.full` regressed to 27.5-28.4ms/8 layers.
  - `gdn.out_proj` regressed to 10.3-10.6ms/24 layers.

Verdict:
- Reject and revert.
- The generic projection shader keeps the 16x16 shape; the 64x4 win is specific to the fused MLP gate/up shader.

### E087: Promote Retiled Vulkan Full-Attention QKV Fusion

Change:
- Retiled the fused single-token full-attention Q/K/V projection shader from one output column per 256-lane workgroup to 16 output columns x 16 reduction lanes.
- Reduced the dispatch count from `total_out` workgroups to `ceil(total_out / 16)`.
- Promoted full-attention QKV fusion to default with rollback guard `KILN_DISABLE_VULKAN_FULL_ATTN_QKV=1`.

Reasoning:
- E080 rejected the original full-attention QKV opt-in because `layer.full` ballooned to 74.1-85.1ms/8 layers and whole-token latency regressed to 194.6ms.
- The rejected shader used one full workgroup per output column, unlike the tuned generic projection shader, so it did far too much workgroup scheduling for Qwen-shaped Q/K/V projections.
- The retiled shader keeps the fusion benefit of one x upload, one dispatch, and one readback for Q/K/V, while using the same 16x16 workgroup geometry that generic `linear_decode` kept after E086.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity full_attn_qkv_decode_matches_cpu_reference -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
- Opt-in profile before promotion:
  - Command: `KILN_ENABLE_VULKAN_FULL_ATTN_QKV=1 KILN_PROFILE_DECODE=1 KILN_BENCH_LOG_ITL=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 4 --skip-training`.
  - Prefill 1855.0ms; mean ITL 133.0ms, p50 131.0ms, p99 141.9ms.
  - `layer.full` was 22.6-23.1ms/8 layers, down from E084's roughly 23.9-25.0ms profile band and far below E080's 74.1-85.1ms rejected opt-in.
- Discarded no-profile parallel A/B:
  - An accidental simultaneous opt-in/default bench contended on the GPU and produced invalid 209-236ms means; ignored.
- Sequential no-profile checks before promotion:
  - Opt-in first run: prefill 1899.4ms; mean ITL 128.1ms, p50 127.5ms, p99 137.6ms.
  - Opt-in confirmation: prefill 2888.7ms; mean ITL 128.7ms, p50 128.9ms, p99 137.3ms.
  - Same-binary no-env before promotion: prefill 1861.3ms; mean ITL 132.4ms, p50 131.7ms, p99 144.1ms.
- Promoted default:
  - Default no-env first run: prefill 3863.6ms; mean ITL 130.4ms, p50 129.9ms, p99 137.0ms.
  - Default no-env confirmation: prefill 1879.8ms; mean ITL 130.7ms, p50 129.5ms, p99 140.6ms.
  - Rollback guard first run: prefill 1814.1ms; mean ITL 132.3ms, p50 132.4ms, p99 139.4ms.
  - Rollback guard confirmation: prefill 1832.0ms; mean ITL 133.7ms, p50 132.9ms, p99 146.6ms.

Verdict:
- Keep default with rollback guard.
- This is a same-binary micro-win, not a new best observed source anchor; E084's 129.1ms run remains the best observed no-env bs=1 result.
- The full-attention QKV fusion is now no longer the catastrophic E080 path because the shader geometry matches the proven generic projection tiling.

### E088: Guard Retiled Full-Attention QKV Fusion To Single-Row Decode

Change:
- Intended to add an explicit `batch == 1` backend gate to `VulkanBackend::full_attn_qkv_decode`.
- The edit actually landed on `VulkanBackend::gdn_in_proj_decode`, changing its gate from `seq_len == 1` to `batch == 1 && seq_len == 1`.

Reasoning:
- E087 promoted the full-attention QKV path from opt-in to default.
- The kernel and dispatch helper already assumed one row, but the backend gate only checked `seq_len == 1`.
- Native batch work must not accidentally route multi-row tensors into a single-row helper.
- The targeted tests did not include a fresh no-profile batch throughput check, so they missed that batched GDN input projection had fallen back to CPU.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed.
- `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
- Fresh no-profile four-prompt `/v1/completions/batch` check before E089:
  - `max_tokens=16`, 64 completion tokens.
  - 46.581s, 1.374 tok/s.

Verdict:
- Superseded by E089.
- The intended full-attention QKV safety guard is still needed, but E088's landed code regressed native batch by disabling the existing batched Vulkan GDN input projection path.

### E089: Restore Batched GDN Input Projection And Apply Single-Row Guards

Change:
- Restored `VulkanBackend::gdn_in_proj_decode` to accept any batch with `seq_len == 1`, so `batch > 1` reaches the existing `gdn_in_proj_decode_batched.comp` Vulkan path again.
- Added the intended `batch == 1 && seq_len == 1` guard to `VulkanBackend::full_attn_qkv_decode`.
- Added the same single-row guard to `VulkanBackend::linear_decode_argmax`, whose kernel returns one scalar token id and already requires a single hidden row.

Reasoning:
- E058/E063 already proved that native batch must keep GDN input projection on Vulkan; falling back to CPU made `gdn.in_proj` a multi-second per-step bucket.
- The retiled full-attention QKV shader is a single-row helper, so it must not consume native batch tensors.
- The LM-head argmax helper reads back one token id and validates one hidden row in the kernel dispatcher, so the backend should reject multi-row tensors before dispatch.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed all 17 Vulkan parity tests, including batched GDN input projection, full-attention QKV, and LM-head argmax.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
- Fresh release-server no-profile four-prompt `/v1/completions/batch` check after the fix:
  - Same fixture as the E088 regression check: `temperature=0`, `top_p=1`, `max_tokens=16`, 64 completion tokens.
  - 25.457s, 2.514 tok/s.
  - This restores the result to the earlier native-batch performance band and is 1.83x faster than the broken E088 build on the same fixture.

Verdict:
- Keep.
- This is a correctness and bs>1 regression fix; it does not change the best observed bs=1 source anchor.

### E090: Retest Existing Batch Fused-GDN State Options

Change:
- No code change.
- Retested existing fused-GDN host-visible state and fused-GDN single-submit options on top of the E089 fixed build.

Reasoning:
- The fixed batch profile showed non-final decode steps dominated by `gdn.fused_gates_recur_norm`:
  - Short profiled four-prompt `max_tokens=4` run: decode steps 0 and 1 spent 585.8ms and 662.5ms across 24 GDN fused recurrent/norm calls.
  - The final max-token-capped decode step dropped to 53.5ms because E066 skips recurrent-state readback there.
- Existing opt-ins already targeted state transfer or command submission, so they were cheap to falsify before adding a new resident-state path.

Evidence:
- Fixed E089 default fresh-server four-prompt `max_tokens=16` fixture: 25.457s / 2.514 tok/s.
- `KILN_ENABLE_VULKAN_GDN_HOST_VISIBLE_STATE=1`: 25.927s / 2.468 tok/s.
- `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED_SINGLE_SUBMIT=1`: 25.313s / 2.528 tok/s.

Verdict:
- Reject both as promotions.
- Host-visible state is slower; single-submit is only a tiny/noisy difference and does not address the dominant recurrent-state residency problem.

### E091: Promote Batch Fused-GDN Resident State

Change:
- Added `dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state`, which keeps the fused GDN recurrent state in a device-local Vulkan buffer across lockstep batch decode steps.
- Added `BackendRuntime::materialize_gdn_recurrent_resident_state` so native batch can safely read resident state back before active-row compaction.
- Native batch now enters the GDN recurrent resident-state scope around the lockstep decode loop and materializes recurrent state before selecting active rows after EOS/finish compaction.
- Promoted the resident fused-GDN path for `batch > 1` by default. Rollback guard: `KILN_DISABLE_VULKAN_GDN_DECODE_FUSED_RESIDENT_STATE=1`.

Reasoning:
- E090's profile showed that non-final batch fused-GDN calls are dominated by recurrent-state readback and CPU tensor materialization.
- E066 already proved the state readback is dead on the final max-token-capped step. E091 extends that principle across non-final lockstep decode steps while preserving correctness by keeping the state buffer resident for the next step.
- Active-row compaction can happen after EOS. Materializing resident recurrent states immediately before compaction keeps the existing row-selection logic correct, then the next compacted decode can upload/cache the new compacted tensors under their new tensor IDs.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed all 18 Vulkan parity tests, including the new two-step resident fused-GDN parity test.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
- Opt-in before promotion, fresh release server, four-prompt `max_tokens=16`: 15.545s / 4.117 tok/s.
- Promoted no-env default, same fixture: 15.498s / 4.130 tok/s.
- Same-binary rollback guard `KILN_DISABLE_VULKAN_GDN_DECODE_FUSED_RESIDENT_STATE=1`: 25.917s / 2.469 tok/s.
- Relative to the E089 fixed default 25.457s / 2.514 tok/s, the promoted resident path is 1.64x faster on this short all-miss batch fixture.

Verdict:
- Keep default with rollback guard.
- This is the largest current bs>1 decode win after native batch was re-promoted; the remaining large batch targets are true multi-sequence paged attention and broader scheduling.

### E092: Retile Batched Vulkan MLP Gate/Up To 64x4

Change:
- Retiled `mlp_gate_up_decode_batched.comp` from 16 output columns x 16 reduction lanes to 64 output columns x 4 reduction lanes.
- Updated batched MLP gate/up dispatch counts in both standalone gate/up and full MLP decode paths from `ceil(intermediate / 16)` to `ceil(intermediate / 64)` per batch row.
- The bs=1 shader already used this 64x4 shape from E084.

Reasoning:
- The E091 profile moved the bottleneck away from fused-GDN state readback. The largest remaining batch subop was `mlp.fused`, about 192-196ms across 32 MLP layers on the short profiled fixture.
- The 64x4 shape is already the best observed single-row MLP gate/up shape on Strix Halo. Applying it to the batched shader reduces workgroup count by 4x for the gate/up phase while preserving the same 256 invocations per workgroup.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode -- --nocapture` passed, including batched MLP parity.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed all 18 Vulkan parity tests.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed.
- `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
- Fresh release-server no-profile four-prompt `max_tokens=16` fixture:
  - E091 default before this retile: 15.498s / 4.130 tok/s.
  - E092 retiled batched MLP: 14.550s / 4.399 tok/s.

Verdict:
- Keep.
- This is a batch-only decode win and does not affect the bs=1 MLP shader, which was already on 64x4.

### E093: Retile Batched Vulkan GDN Input Projection To 64x4

Change:
- Retiled `gdn_in_proj_decode_batched.comp` from 16 output columns x 16 reduction lanes to 64 output columns x 4 reduction lanes.
- Updated batched GDN input-projection dispatch counts in both direct and single-submit paths from `ceil(total_out / 16)` to `ceil(total_out / 64)` per batch row.
- The bs=1 GDN input-projection shader remains on its existing 16x16 shape.

Reasoning:
- The E092 profile showed `gdn.in_proj` had become one of the largest remaining subop buckets, about 90-94ms across 24 GDN layers on the short profiled fixture.
- Like E092, this reduces batched projection workgroup count by 4x while preserving 256 invocations per workgroup.
- Earlier E086 rejected a generic bs=1 `linear_decode` retile, so this experiment is deliberately batch-only and scoped to the GDN input-projection batched shader.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_in_proj -- --nocapture` passed, including batched GDN input-projection parity.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed all 18 Vulkan parity tests.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed.
- `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
- Fresh release-server no-profile four-prompt `max_tokens=16` fixture:
  - E092 before this retile: 14.550s / 4.399 tok/s.
  - E093 retiled batched GDN input projection: 13.981s / 4.578 tok/s.

Verdict:
- Keep.
- This is a batch-only GDN projection win and does not change the bs=1 GDN input-projection shader.

### E094: Retile Batched Generic Vulkan Linear Decode To 32x8

Change:
- Retiled `linear_decode_batched.comp` from 16 output columns x 16 reduction lanes to 32 output columns x 8 reduction lanes.
- Updated batched generic linear dispatch counts in direct, single-submit, and MLP-down paths from `ceil(out_dim / 16)` to `ceil(out_dim / 32)` per batch row.
- Left the bs=1 generic `linear_decode.comp` unchanged at 16x16 because E086 already showed that retile direction regresses single-stream decode.

Reasoning:
- A fresh post-E093 batch profile showed remaining non-final decode buckets clustered around full-vocab LM-head materialization, batched generic GDN/MLP projection work, and row-local full-attention:
  - Same four-prompt profiled fixture, `max_tokens=4`, 72 prompt tokens / 16 completion tokens: 9.062s / 1.766 tok/s.
  - Decode step totals were about 397ms, 361ms, and 362ms.
  - Non-final buckets were roughly `lm_head` 93-95ms, `full.block` 70ms/8 layers, `linear.gdn` 105-106ms/24 layers, and `linear.mlp` 89-90ms/24 layers.
  - Subops showed E093 had moved `gdn.in_proj` down to about 62ms/24 layers, while `gdn.out_proj` was about 19-20ms/24 layers and `mlp.fused` about 120ms/32 layers.
- The generic batched linear shader feeds GDN out projection, MLP down projection, and the batched LM-head logits materialization path, so reducing output-column workgroups is a plausible batch-only win.
- A 64x4 variant was also tested but was too small/noisy as a generic promotion: same-fixture 64-token run measured 13.082s / 4.892 tok/s versus a 16x16 rollback at 13.204s / 4.847.
- The 32x8 shape keeps more reduction parallelism than 64x4 while still halving output-column workgroups relative to 16x16.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity linear_decode_batched -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed all 18 Vulkan parity tests.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed.
- `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
- Same-fixture fresh release-server no-profile four-prompt `max_tokens=16` checks, 72 prompt tokens / 64 completion tokens:
  - 16x16 rollback: 13.204s / 4.847 tok/s.
  - 64x4 trial: 13.082s / 4.892 tok/s.
  - 32x8 kept shape: 12.942s / 4.945 tok/s.

Verdict:
- Keep the 32x8 batch-only generic linear shape.
- Reject the 64x4 generic linear shape as too small/noisy for a broad shader path.

### E095: Retile Batched Vulkan MLP Gate/Up To 128x2

Change:
- Retiled `mlp_gate_up_decode_batched.comp` from 64 output columns x 4 reduction lanes to 128 output columns x 2 reduction lanes.
- Updated batched MLP gate/up dispatch counts in both standalone gate/up and full MLP decode paths from `ceil(intermediate / 64)` to `ceil(intermediate / 128)` per batch row.
- Left the bs=1 MLP gate/up shader unchanged at 64x4; E085 already showed that 128x2 regresses the single-token path.

Reasoning:
- The post-E094 profile moved generic linear work down, but `mlp.fused` still dominated the subop profile at roughly 110-113ms/32 layers on the short four-prompt fixture.
- E092's 64x4 batched MLP retile was a clear win over 16x16. The batch path has more output columns active than bs=1, so the next output-column-parallel shape deserved a direct batch-only test despite E085's bs=1 rejection.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode -- --nocapture` passed, including batched MLP parity.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed all 18 Vulkan parity tests.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed.
- `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
- Same-fixture fresh release-server no-profile four-prompt `max_tokens=16` checks, 72 prompt tokens / 64 completion tokens:
  - E094 32x8 generic linear with 64x4 batched MLP gate/up: 12.942s / 4.945 tok/s.
  - E095 128x2 batched MLP gate/up: 11.880s / 5.387 tok/s.

Verdict:
- Keep the 128x2 batch-only MLP gate/up shape.
- This does not change the bs=1 MLP shader.

### E096: Test Batched Vulkan GDN Input Projection At 128x2

Change:
- Temporarily retiled `gdn_in_proj_decode_batched.comp` from 64 output columns x 4 reduction lanes to 128 output columns x 2 reduction lanes.
- Temporarily updated batched GDN input-projection dispatch counts in both direct and single-submit paths from `ceil(total_out / 64)` to `ceil(total_out / 128)` per batch row.

Reasoning:
- A fresh post-E095 profile showed `gdn.in_proj` as the largest remaining GDN subop bucket:
  - Same four-prompt profiled fixture, `max_tokens=4`, 72 prompt tokens / 16 completion tokens: 8.760s / 1.826 tok/s.
  - Decode step totals were about 294ms, 260ms, and 260ms.
  - Non-final buckets were roughly `linear.gdn` 101-102ms/24 layers, `lm_head` 60-61ms, `full.block` 54-55ms/8 layers, and `linear.mlp` 41-42ms/24 layers.
  - Subops showed `gdn.in_proj` about 60-61ms/24 layers, `mlp.fused` about 55ms/32 layers, `gdn.out_proj` about 16ms/24 layers, and fused resident GDN recurrent/norm about 15ms/24 non-final layers.
- E095 showed the batched MLP path still benefited from pushing to 128x2 despite bs=1 rejecting that shape, so the analogous batch-only GDN projection shape was worth falsifying.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_in_proj -- --nocapture` passed, including batched GDN input-projection parity.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Same-fixture fresh release-server no-profile four-prompt `max_tokens=16` checks, 72 prompt tokens / 64 completion tokens:
  - E095 64x4 batched GDN input projection: 11.880s / 5.387 tok/s.
  - E096 128x2 batched GDN input projection: 12.542s / 5.103 tok/s.

Verdict:
- Reject and revert.
- Keep the E093 64x4 batched GDN input-projection shape.

### E097: Test Skipping Final-Step Paged KV Write

Change:
- Temporarily marked native greedy batch decode calls that are known final by `max_tokens`.
- Temporarily skipped the paged K/V cache write for that final decode token, read only the prefix K/V from cache, and concatenated the current-token K/V in memory for the attention read path.

Reasoning:
- The final decode-by-`max_tokens` step already skips fused-GDN recurrent-state readback because generation stops immediately afterward.
- A paged K/V write for the final generated token is similarly dead for future decode steps, but attention still needs the current token's K/V for the current step.
- The experiment tested whether replacing final-step cache write/read of `total_seq_len` with prefix read plus current-token concatenation removes useful work or merely shifts it into extra tensor traffic.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed.
- `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Same-fixture fresh release-server no-profile four-prompt `max_tokens=16` checks, 72 prompt tokens / 64 completion tokens:
  - E095 normal final-step paged K/V write: 11.880s / 5.387 tok/s.
  - E097 skip final-step paged K/V write: 11.902s / 5.377 tok/s.

Verdict:
- Reject and revert.
- The result is neutral/slightly slower, and the code adds a thread-local generation guard plus a special attention read path. Keep the simpler final-step cache behavior until a broader paged-attention restructuring can remove more than one terminal write.

### E098: Test Batched Vulkan MLP Gate/Up At 256x1

Change:
- Temporarily retiled `mlp_gate_up_decode_batched.comp` from 128 output columns x 2 reduction lanes to 256 output columns x 1 reduction lane.
- Temporarily updated the two batched MLP gate/up dispatch counts from `ceil(intermediate / 128)` to `ceil(intermediate / 256)` per batch row.

Reasoning:
- E095 showed that the batch-only MLP gate/up shader still benefits from pushing beyond the bs=1 limit of 64x4 to 128x2.
- E098 tested whether halving workgroup count again is still worth it, or whether single-lane hidden-dimension reduction becomes too serial on RADV Strix Halo.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode -- --nocapture` passed, including batched MLP decode parity.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Same-fixture fresh release-server no-profile four-prompt `max_tokens=16` checks, 72 prompt tokens / 64 completion tokens:
  - E095 128x2 batched MLP gate/up: 11.880s / 5.387 tok/s.
  - E098 256x1 batched MLP gate/up: 12.081s / 5.298 tok/s.

Verdict:
- Reject and revert.
- Keep E095's 128x2 batch-only MLP gate/up shape.

### E099: Retile Batched Vulkan GDN Input Projection To 80x3

Change:
- Retiled `gdn_in_proj_decode_batched.comp` from 64 output columns x 4 reduction lanes to 80 output columns x 3 reduction lanes.
- Updated batched GDN input-projection dispatch counts in both direct and single-submit paths from `ceil(total_out / 64)` to `ceil(total_out / 80)` per batch row.

Reasoning:
- E096 showed 128x2 was too wide for batched GDN input projection, regressing the standard 72-prompt-token batch fixture to 12.542s / 5.103 tok/s.
- The current profile still shows `gdn.in_proj` as the largest GDN subop bucket at roughly 61ms/24 layers on non-final four-prompt decode steps.
- 80x3 is an intermediate shape: it cuts workgroups by 20% versus 64x4 while keeping more reduction parallelism than the rejected 128x2 shape.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_in_proj -- --nocapture` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture` passed.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed.
- `cargo test -p kiln-model generate::tests::test_generate_paged_max_tokens -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Current kept-source profile before the change, four-prompt `max_tokens=4`, 72 prompt tokens / 16 completion tokens:
  - 8.691s / 1.841 tok/s.
  - Decode step totals: about 295ms, 261ms, and 261ms.
  - Non-final buckets: `linear.gdn` about 102ms/24 layers, `lm_head` about 62ms, `full.block` about 55ms/8 layers, `linear.mlp` about 41ms/24 layers.
  - Subops: `gdn.in_proj` about 61ms/24 layers, `mlp.fused` about 54ms/32 layers, `gdn.out_proj` about 16ms/24 layers, and fused resident GDN recurrent/norm about 14ms/24 non-final layers.
- Same-fixture fresh release-server no-profile four-prompt `max_tokens=16` checks, 72 prompt tokens / 64 completion tokens:
  - E095/E098 kept 64x4 batched GDN input projection with 128x2 batched MLP gate/up baseline: 11.880s / 5.387 tok/s.
  - E099 first 80x3 run: 11.715s / 5.463 tok/s.
  - E099 fresh-server confirmation: 11.789s / 5.429 tok/s.

Verdict:
- Keep the 80x3 batch-only GDN input-projection shape.
- This does not change the bs=1 GDN input-projection shader.

### E100: Test Batched Vulkan GDN Input Projection At 96x3

Change:
- Temporarily retiled `gdn_in_proj_decode_batched.comp` from 80 output columns x 3 reduction lanes to 96 output columns x 3 reduction lanes.
- Temporarily updated batched GDN input-projection dispatch counts from `ceil(total_out / 80)` to `ceil(total_out / 96)` per batch row.

Reasoning:
- E099 showed that 80x3 improves over 64x4 while 128x2 was already rejected.
- 96x3 keeps the same hidden-dimension reduction parallelism as E099 and reduces workgroup count further, so it was the next narrow shape to falsify.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_in_proj -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Same-fixture fresh release-server no-profile four-prompt `max_tokens=16` checks, 72 prompt tokens / 64 completion tokens:
  - E099 80x3 confirmation: 11.789s / 5.429 tok/s.
  - E100 96x3 trial: 11.984s / 5.341 tok/s.

Verdict:
- Reject and revert.
- Keep E099's 80x3 batch-only GDN input-projection shape.

### E101: Test Batched Generic Vulkan Linear Decode At 48x6

Change:
- Temporarily retiled `linear_decode_batched.comp` from 32 output columns x 8 reduction lanes to 48 output columns x 6 reduction lanes.
- Temporarily updated the three batched generic linear dispatch counts from `ceil(out_dim / 32)` to `ceil(out_dim / 48)` per batch row.

Reasoning:
- The current E099 profile has `lm_head`, `mlp.fused`, `full.block`, and `linear.gdn` close together.
- E094 showed that moving generic batched linear from 16x16 to 32x8 helped, while a 64x4 trial was too small/noisy. A 48x6 middle point could reduce workgroups for LM-head, MLP down, GDN out, and full-attention projections without giving up as much reduction parallelism as 64x4.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity linear_decode_batched -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Same-fixture fresh release-server no-profile four-prompt `max_tokens=16` checks, 72 prompt tokens / 64 completion tokens:
  - E099 current source: 11.789s / 5.429 tok/s confirmation.
  - E101 48x6 generic batched linear: 11.964s / 5.349 tok/s.

Verdict:
- Reject and revert.
- Keep E094's 32x8 generic batched linear shape.

### E102: Test Single-Token Vulkan MLP Gate/Up At 80x3

Change:
- Temporarily retiled `mlp_gate_up_decode.comp` from 64 output columns x 4 reduction lanes to 80 output columns x 3 reduction lanes.
- Temporarily updated the two batch=1 MLP gate/up dispatch counts from `ceil(intermediate / 64)` to `ceil(intermediate / 80)`.

Reasoning:
- A fresh bs=1 profile showed `mlp.fused` as the largest remaining decode bucket at about 48ms/32 layers, ahead of `gdn.in_proj`, `layer.full`, and `lm_head.argmax`.
- E084 kept 64x4 and E085 rejected 128x2 for bs=1 MLP gate/up, so 80x3 was the next middle shape to falsify.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Fresh current-source bs=1 profile before the change:
  - `KILN_PROFILE_DECODE=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 4 --skip-training`
  - Mean ITL: 133.2ms.
  - Non-final profile buckets: `mlp.fused` about 48ms/32 layers, `gdn.in_proj` about 23ms/24 layers, `layer.full` about 21ms/8 layers, `lm_head.argmax` about 14-17ms.
- Standard bs=1 no-profile latency checks:
  - E102 first run: 128.9ms mean ITL, p50 128.7ms, p99 136.6ms.
  - E102 confirmation: 134.8ms mean ITL, p50 131.0ms, p99 153.8ms.

Verdict:
- Reject and revert.
- The first run was slightly better than the best prior 129.1ms observation, but the confirmation regressed badly. Keep E084's 64x4 bs=1 MLP gate/up shape until a change produces stable improvement.

### E103: Attempt To Refresh Mixed Prefix-Hit/Miss Anchor

Change:
- No source change.
- Rebuilt the kept source after the rejected E102 trial so the release binary no longer contained experimental shader code.
- Tried to reconstruct the older E059 mixed cache-hit/cache-miss fixture on the current source.

Reasoning:
- The shortlog still carried the older E059 mixed-cache anchor and explicitly marked it stale after the later E096-E102 batch shader work.
- A valid mixed refresh must show real prefix-cache hit metrics, not just complete a four-row batch.

Evidence:
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed on the kept source.
- First attempted base prompt:
  - Registration `max_tokens=0`: 27 prompt tokens.
  - Batch: 12.743s / 5.022 tok/s for 64 completion tokens.
  - Metrics delta: 0 hits, 4 misses, 0 hit tokens, 0 hit blocks.
  - Verdict: invalid mixed fixture because the registered prompt was shorter than two 16-token blocks.
- Second attempted base prompt:
  - Registration `max_tokens=0`: 42 prompt tokens.
  - Batch: 12.847s / 4.982 tok/s for 64 completion tokens.
  - Metrics delta: 0 hits, 4 misses, 0 hit tokens, 0 hit blocks.
  - Verdict: invalid mixed fixture because the prompt length was not block-aligned for registration.
- Third attempted base prompt:
  - Registration `max_tokens=0`: 32 prompt tokens.
  - Server metrics showed one prefix-cache entry retaining 2 cached blocks.
  - Batch with same user text plus suffix: 12.638s / 5.064 tok/s for 64 completion tokens.
  - Metrics delta: 0 hits, 4 misses, 0 hit tokens, 0 hit blocks.
  - Verdict: invalid mixed fixture because the rendered hit prompt did not start with the cached token sequence.
- Fourth attempted hit row using an assistant-prefix message:
  - Existing cache metrics before batch: 1 cached entry, 2 cached blocks.
  - Batch: 12.840s / 4.984 tok/s for 64 completion tokens.
  - Metrics delta: 0 hits, 4 misses, 0 hit tokens, 0 hit blocks.
  - Verdict: invalid mixed fixture; still not a prefix-cache hit.

Verdict:
- Do not refresh the E059 mixed anchor from these timings.
- The no-hit 12.6-12.8s runs are effectively current short all-miss or miss-only shaped batches, not the older mixed cache-hit/cache-miss scenario.
- A valid refresh needs the exact delimiter/message construction that makes the rendered batch prompt start with a block-aligned cached registration, or a dedicated benchmark helper that builds token-prefix-compatible prompts directly.

### E104: Add Native Batch Mixed Cached-Prefix Regression Test

Change:
- Added `native_batch_greedy_mixes_cached_prefix_hits_and_misses`.
- The test builds raw token prompts directly against `generate_paged_shared_tokens_batch_greedy`, with one block-aligned `PagedPrefixReuse` row and three uncached miss rows.
- It asserts deterministic greedy outputs for all rows, confirms batch-local common-prefix registration stays disabled for mixed external cache input, verifies the hit row remains registrable, and checks that the cached prefix block is not returned in `allocated_blocks`.

Reasoning:
- E103 showed that reconstructing the historical mixed cache-hit/cache-miss fixture through server chat formatting is brittle: every attempted "hit" row missed the prefix cache despite one 32-token registration retaining two blocks.
- The model-level native batch API already takes exact token vectors plus `Vec<Option<PagedPrefixReuse>>`, so this is the stable contract that future mixed-cache performance fixtures should build from.
- This is a regression/harness step, not a speed change.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed.
  - 4 native batch greedy tests passed, including the new mixed cached-prefix hit/miss case.

Verdict:
- Keep.
- The stale E059 mixed cache-hit/cache-miss performance anchor is still the latest valid mixed anchor.
- A current mixed performance refresh should use an exact token-prefix-compatible benchmark/helper rather than relying on reconstructed chat-message delimiters.

### E105: Test Batched Vulkan MLP Gate/Up At 96x3

Change:
- Temporarily retiled `mlp_gate_up_decode_batched.comp` from 128 output columns x 2 reduction lanes to 96 output columns x 3 reduction lanes.
- Temporarily updated the two batched MLP gate/up dispatch counts from `ceil(intermediate / 128)` to `ceil(intermediate / 96)` per batch row.
- Reverted after measurement.

Reasoning:
- E095 kept 128x2 as the current batch-only MLP gate/up shape, while E098 rejected 256x1 as too wide/serial.
- 96x3 is a conservative middle shape that increases hidden-dimension reduction parallelism relative to 128x2 while still reducing workgroups versus the older 64x4 shape.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode -- --nocapture` passed for the 96x3 trial.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed for the 96x3 trial and again after reverting to the kept 128x2 source.
- Fresh release-server no-profile same-request A/B on a four-prompt all-miss `/v1/completions/batch` fixture, `temperature=0`, `top_p=1`, `max_tokens=16`, 88 prompt tokens / 64 completion tokens:
  - E105 96x3 trial: 12.490s / 5.124 tok/s.
  - Kept 128x2 baseline: 12.301s / 5.203 tok/s.

Verdict:
- Reject and revert.
- Keep E095's 128x2 batch-only MLP gate/up shape.

### E106: Test Per-Row Vulkan LM-Head Argmax In Native Batch Decode

Change:
- Temporarily changed `model_forward_paged_next_tokens_greedy_batch` to avoid materializing `[batch, 1, vocab]` LM-head logits when the backend supports `linear_decode_argmax`.
- The trial narrowed the final-normalized hidden state to one row at a time and called the existing single-row Vulkan LM-head argmax path for each active batch row.
- Reverted after measurement.

Reasoning:
- The E099 profile still showed `lm_head` around 62-64ms on non-final four-prompt decode steps.
- Greedy batch decode only needs token IDs, so materializing and reading back full vocab logits is theoretically removable work.
- The risk is that replacing one batched LM-head projection with multiple per-row argmax dispatches adds enough submit/reduction overhead to erase the readback savings.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed for the trial.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed for the trial.
- Fresh release-server no-profile same-request A/B on the same four-prompt all-miss `/v1/completions/batch` fixture as E105, `temperature=0`, `top_p=1`, `max_tokens=16`, 88 prompt tokens / 64 completion tokens:
  - Kept batched LM-head logits baseline from E105: 12.301s / 5.203 tok/s.
  - E106 per-row Vulkan argmax trial: 12.270s / 5.216 tok/s.

Verdict:
- Reject and revert.
- The measured delta is only 31ms on a 12.3s request, about 0.25%, which is below the noise floor and not enough to justify replacing the simpler batched logits path with multiple per-row argmax dispatches.
- A future LM-head win should be a true batched argmax/reduction path or a broader resident/output-buffer redesign, not per-row dispatch fan-out.

### E107: Reuse Batch Decode Row Position Tensors Across Full-Attention Layers

Change:
- Precompute one one-token position tensor per active batch row in `model_forward_paged_next_tokens_greedy_batch`.
- Pass those tensors into `transformer_block_paged_batch_decode_rows` so the eight full-attention layers reuse them instead of rebuilding the same per-row position tensor inside every layer.

Reasoning:
- Full-attention batch decode still runs attention row-by-row, so any per-row setup inside the full-attention layer loop is multiplied by `active_rows * full_attention_layers * decode_steps`.
- The absolute row positions are fixed for a decode step and already passed as `start_positions`; creating the same one-element tensors repeatedly is unnecessary host-side work.
- This does not change paged-KV behavior, RoPE positions, or model math.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Fresh release-server no-profile same-request A/B on the same four-prompt all-miss `/v1/completions/batch` fixture as E105/E106, `temperature=0`, `top_p=1`, `max_tokens=16`, 88 prompt tokens / 64 completion tokens:
  - Kept source baseline from E105: 12.301s / 5.203 tok/s.
  - E107 row position tensor reuse: 12.243s / 5.227 tok/s.

Verdict:
- Keep.
- The measured win is small, about 58ms / 0.47% on this short fixture, but the change removes repeated setup work from the full-attention row path and is simpler than recomputing those tensors in every full-attention layer.

### E108: Reuse Batch Decode Row RoPE Tables Across Full-Attention Layers

Change:
- Precompute one `(cos, sin)` RoPE table pair per active batch row in `model_forward_paged_next_tokens_greedy_batch`.
- Pass those row table pairs into `transformer_block_paged_batch_decode_rows`, which now routes full-attention row decode through the existing `gqa_attention_paged_with_rope_tables` fast path.
- This builds on E107's row position tensor reuse.

Reasoning:
- E107 reused the one-token position tensors, but each full-attention layer still rebuilt the same row-specific RoPE frequency, cos, and sin tensors.
- Absolute row positions are fixed for the decode step, and the same one-token RoPE tables are valid for all eight full-attention layers.
- This removes repeated host-side tensor math from the row-local full-attention path without changing Q/K rotation semantics.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-model native_batch_greedy -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Fresh release-server no-profile same-request A/B on the same four-prompt all-miss `/v1/completions/batch` fixture as E105-E107, `temperature=0`, `top_p=1`, `max_tokens=16`, 88 prompt tokens / 64 completion tokens:
  - Pre-E107 kept baseline from E105: 12.301s / 5.203 tok/s.
  - E107 row position tensor reuse: 12.243s / 5.227 tok/s.
  - E108 row RoPE table reuse: 11.988s / 5.339 tok/s.

Verdict:
- Keep.
- This is a small but clear full-attention row-path win, about 2.1% throughput over E107 on this short all-miss fixture, by removing repeated per-layer RoPE table construction.

### E109: Test Batched Vulkan GDN Input Projection At 80x4

Change:
- Temporarily retiled `gdn_in_proj_decode_batched.comp` from 80 output columns x 3 reduction lanes to 80 output columns x 4 reduction lanes.
- Kept the batch dispatch count at `ceil(total_out / 80)` per batch row.
- Reverted after measurement.

Reasoning:
- E099's 80x3 shape is the best current batched GDN input-projection tile, while E096's 128x2 and E100's 96x3 were both too wide.
- This checked whether the useful 80-column group from E099 was limited by hidden-dimension reduction parallelism rather than output-column dispatch count.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity gdn_in_proj -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Fresh release-server no-profile same-request A/B on the same four-prompt all-miss `/v1/completions/batch` fixture as E105-E108, `temperature=0`, `top_p=1`, `max_tokens=16`, 88 prompt tokens / 64 completion tokens:
  - E108 row RoPE table reuse baseline with kept 80x3 GDN input projection: 11.988s / 5.339 tok/s.
  - E109 80x4 GDN input-projection trial: 12.366s / 5.175 tok/s.

Verdict:
- Reject and revert.
- Keep E099's 80x3 batch-only GDN input-projection shape; adding a fourth reduction lane at the same 80-column width regressed the current short all-miss fixture.

### E110: Test Batched Vulkan MLP Gate/Up At 160x2

Change:
- Temporarily retiled `mlp_gate_up_decode_batched.comp` from 128 output columns x 2 reduction lanes to 160 output columns x 2 reduction lanes.
- Temporarily updated the two batched MLP gate/up dispatch counts from `ceil(intermediate / 128)` to `ceil(intermediate / 160)` per batch row.
- Reverted after measurement.

Reasoning:
- E095's 128x2 shape is the best current batch-only MLP gate/up tile, while E098's 256x1 and E105's 96x3 both regressed.
- 160x2 keeps the same hidden-dimension reduction parallelism as 128x2 while reducing MLP workgroup count, checking whether dispatch count was still limiting this bucket.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Fresh release-server no-profile same-request A/B on the same four-prompt all-miss `/v1/completions/batch` fixture as E105-E109, `temperature=0`, `top_p=1`, `max_tokens=16`, 88 prompt tokens / 64 completion tokens:
  - E108 current kept baseline with 128x2 MLP gate/up: 11.988s / 5.339 tok/s.
  - E110 160x2 MLP gate/up trial: 12.669s / 5.052 tok/s.

Verdict:
- Reject and revert.
- Keep E095's 128x2 batch-only MLP gate/up shape; widening to 160 output columns without increasing reduction parallelism was a clear regression.

### E111: Test Batched Generic Vulkan Linear Decode At 40x8

Change:
- Temporarily retiled `linear_decode_batched.comp` from 32 output columns x 8 reduction lanes to 40 output columns x 8 reduction lanes.
- Temporarily updated the two batched generic linear dispatch counts from `ceil(out_dim / 32)` to `ceil(out_dim / 40)` per batch row.
- Reverted after measurement.

Reasoning:
- E094's 32x8 shape is the best current batch-only generic linear tile, while E101's 48x6 and the earlier 64x4 trial both regressed.
- 40x8 keeps the same hidden-dimension reduction parallelism as 32x8 while reducing output workgroup count, checking whether the generic path can keep eight reduction lanes and still benefit from slightly wider output groups.

Evidence:
- `cargo fmt --all --check` passed.
- `git diff --check` passed.
- `cargo test -p kiln-vulkan-kernel --test gdn_parity linear_decode_batched -- --nocapture` passed.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench` passed.
- Fresh release-server no-profile same-request A/B on the same four-prompt all-miss `/v1/completions/batch` fixture as E105-E110, `temperature=0`, `top_p=1`, `max_tokens=16`, 88 prompt tokens / 64 completion tokens:
  - E108 current kept baseline with 32x8 generic batched linear: 11.988s / 5.339 tok/s.
  - E111 40x8 generic batched linear trial: 13.135s / 4.872 tok/s.

Verdict:
- Reject and revert.
- Keep E094's 32x8 batch-only generic linear shape; the wider 40-column group at the same eight reduction lanes was a clear whole-request regression.

## Next Candidate

With E108 removing repeated row RoPE table construction and E107 removing repeated row-position tensor construction in the full-attention batch path, the best observed bs=1 short-source anchor remains E084's 129.1ms mean ITL, with a second E084 no-profile run at 130.2ms. The current E087 default measured 130.4ms and 130.7ms while its rollback guard measured 132.3ms and 133.7ms, so the retiled full-attention QKV path is kept as a current-source micro-win and is now correctly gated to batch=1. E089 restored native-batch GDN input projection after the E088 misplaced guard, moving the four-prompt `max_tokens=16` fixture from the broken 46.581s / 1.374 tok/s back to 25.457s / 2.514 tok/s. E091 promoted batch fused-GDN resident state and moved the same fixture to 15.498s / 4.130 tok/s, with rollback `KILN_DISABLE_VULKAN_GDN_DECODE_FUSED_RESIDENT_STATE=1` at 25.917s / 2.469 tok/s. E092 retiled batched MLP gate/up to 64x4 and moved the fixture to 14.550s / 4.399 tok/s. E093 retiled batched GDN input projection to 64x4 and moved that fixture to 13.981s / 4.578 tok/s. E094 retiled the generic batched linear shader to 32x8 and improved a same-fixture 72-prompt-token four-prompt run from 13.204s / 4.847 tok/s with 16x16 rollback to 12.942s / 4.945 tok/s. E095 retiled batched MLP gate/up to 128x2 and moved that same 72-prompt-token fixture to 11.880s / 5.387 tok/s. E096 tried batched GDN input projection at 128x2 and regressed to 12.542s / 5.103. E097 tried skipping final-step paged K/V writes and measured 11.902s / 5.377 tok/s, effectively neutral versus E095's 11.880s / 5.387, so that special cache path was rejected. E098 tried batched MLP gate/up at 256x1 and regressed to 12.081s / 5.298, E105 tried 96x3 and regressed to 12.490s / 5.124 versus a same-request kept 128x2 baseline at 12.301s / 5.203, and E110 tried 160x2 and regressed to 12.669s / 5.052 versus E108's 11.988s / 5.339 same-request baseline, so E095's 128x2 MLP shape remains best. E099 retiled batched GDN input projection to 80x3 and confirmed 11.715s / 5.463 then 11.789s / 5.429, so 80x3 is now the best batched GDN input-projection shape. E100 tried 96x3 and regressed to 11.984s / 5.341, and E109 tried 80x4 and regressed to 12.366s / 5.175 versus E108's 11.988s / 5.339 same-request baseline, confirming the E099 shape is the useful middle point between 64x4, wider groups, and extra reduction lanes. E101 tried generic batched linear at 48x6 and regressed to 11.964s / 5.349, and E111 tried 40x8 and regressed to 13.135s / 4.872 versus E108's 11.988s / 5.339 same-request baseline, confirming E094's 32x8 shape remains best. E102 tried bs=1 MLP gate/up at 80x3 and produced one 128.9ms run but regressed to 134.8ms on confirmation, so E084's 64x4 shape remains best. E103 attempted to refresh the stale mixed cache-hit/cache-miss anchor but all reconstructed hit rows missed the prefix cache, so those timings are not valid mixed anchors; E104 now protects the exact model-level mixed hit/miss path but does not replace the stale performance anchor. E106 showed that simply fanning LM-head argmax out to per-row dispatches is effectively neutral, so the next useful LM-head attempt needs a true batched reduction or broader resident/output-buffer redesign. E079 showed that the opposite 8x32 MLP shape and two-kernel single-submit recording do not improve whole-token latency; E083/E084 show that increasing output-column parallelism is the useful direction for the single-token MLP gate/up shader, with 64x4 currently best. E085 confirmed 128x2 goes too far and regresses the bs=1 MLP bucket, so E095 is deliberately batch-only. E086 confirmed that applying 32x8 to bs=1 generic `linear_decode` regresses MLP down, full-attention, and GDN output projections, so E094 is also batch-only. The next useful single-user experiments should revisit MLP/full-attention with deeper shader-level or layer-residency changes, or look for removable work in the generation/server path. Speculative decode can consider resident-state scopes later, now that explicit materialization exists for resident recurrent state.

For bs>1, duplicate deterministic work is eliminated, batched GDN input projection is on Vulkan, E059 promoted native batch with batch-default fused GDN decode, E061 stops decoding rows after EOS, E062 routes single-row tails back to the tuned bs=1 decode path, E069 removes duplicate prefill for long in-batch common prefixes, E072 reuses those common prefixes across later batches, E091 keeps fused-GDN recurrent state resident through lockstep batch decode, E094 reduces generic batched projection workgroup count, E095 reduces batched MLP gate/up workgroup count further, and E099 lands the best current batched GDN input-projection workgroup balance. The remaining throughput work is now true multi-sequence paged attention and scheduling: batched KV write/read and SDPA for full-attention layers, continuous batching beyond the batch endpoint, and reducing remaining MLP/lm-head/full-attention transfer boundaries.
