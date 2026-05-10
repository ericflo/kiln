# WSL CUDA Qwen3.5-4B Throughput Log

Date: 2026-05-09

Host: WSL2, CUDA 12.4, NVIDIA GeForce RTX 4090 Laptop GPU, 16376 MiB VRAM.

Model: `./Qwen3.5-4B`

Server environment used for accepted measurements:

```text
KILN_MODEL_PATH=./Qwen3.5-4B
KILN_DROP_PROJECTION_ORIGINALS=1
KILN_NUM_BLOCKS=512
KILN_PREFIX_CACHE_MAX_ENTRIES=1
KILN_GRAD_CHECKPOINT_SEGMENTS=32
KILN_USE_FLCE=1
```

## Recovery Notes

Recent CUDA graph experiments were rejected after the machine showed WSL/CUDA
memory pressure and a driver crash:

- `dmesg` showed `dxgkio_make_resident: Ioctl failed: -12` around 14:57.
- `KILN_CUDA_GRAPH_STABLE_PAGED_METADATA=1` crashed in `libcuda.so.1.1` at
  15:21 with a `tokio-rt-worker` SIGSEGV.
- Raising `KILN_CUDA_GRAPH_CACHE_MAX` to 128 improved warmed latency after the
  cache filled, but its cold-fill behavior and memory risk are not acceptable
  on 16 GiB GPUs.

Those graph paths were not kept. The accepted change is limited to RMSNorm
dispatch.

## Accepted Change

`crates/kiln-model/src/forward.rs` now routes CUDA BF16 RMSNorm as follows:

- If both tensors are outside Candle autograd (`!track_op()`), inference uses
  `kiln_rmsnorm_kernel::fused_rmsnorm`, the forward-only fused kernel.
- If either tensor is tracked by autograd, training keeps the existing
  `should_use_fused_rmsnorm()` 47 GiB VRAM gate before using
  `fused_rmsnorm_with_autograd`.
- `KILN_DISABLE_RMSNORM_KERNEL` and `KILN_DISABLE_RMSNORM_BACKWARD` still force
  the fallback path.

This keeps the training OOM protection intact while removing the RMSNorm
fallback penalty from normal inference on the 4090 Laptop GPU.

## Endpoint Probe

Probe shape:

- endpoint: `/v1/chat/completions`
- prompt: integer-list prompt with unique nonce
- `max_tokens`: 64
- `temperature`: 0

Recovered fallback baseline, RMSNorm gate OFF:

```text
3.806, 3.296, 3.296, 3.284, 3.273, 3.371 seconds
```

Forced fused RMSNorm proxy (`KILN_FORCE_RMSNORM_KERNEL=1`):

```text
3.3585, 2.8251, 2.8154, 2.8015, 2.7930, 2.7735 seconds
```

Patched default server, no force override:

```text
3.2055, 2.7625, 2.7347, 2.7066, 2.7377, 2.7352 seconds
```

Warmed average, runs 1-5:

| path | seconds | completion tok/s |
| --- | ---: | ---: |
| fallback baseline | 3.3040 | 19.37 |
| forced fused proxy | 2.8017 | 22.84 |
| patched default | 2.7353 | 23.40 |

Patched default vs fallback baseline:

- 17.2% lower warmed request latency.
- 20.8% higher warmed completion throughput.

## Memory

Patched default server health after load:

```json
{
  "blocks_total": 512,
  "blocks_free": 512,
  "gpu_memory": {
    "total_vram_gb": 17.171480576,
    "model_gb": 10.036969472,
    "kv_cache_gb": 0.268435456,
    "training_budget_gb": 6.866075648
  }
}
```

After the six-request patched probe, `nvidia-smi` reported:

```text
16376 MiB total, 10046 MiB used, 6005 MiB free
```

## Training Verification

Training smoke request:

- endpoint: `/v1/train/sft`
- examples: 1
- epochs: 1
- LoRA rank: 4
- `auto_load`: false
- output adapter: `smoke-rmsnorm-inference-fastpath`

Result:

```text
job_id=3e7ed8d6-f66b-4d4f-ad0e-cc49e58021c4
state=completed
final_loss=4.8016462326049805
```

Server log confirmed training stayed on the protected path:

```text
kiln rmsnorm gate total_vram_mib=16376 threshold_mib=48128 detection_source=nvidia-smi force_override=false fused_path="OFF"
```

The smoke adapter was then unloaded and deleted through the adapter API.

## Validation

Commands run:

```bash
cargo fmt --all --check
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" cargo build --release --features cuda --bin kiln
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" cargo test --release -p kiln-model --features cuda rms_norm --lib
```

Focused test result:

```text
3 passed; 0 failed; 293 filtered out
```

## Follow-Up: CUDA Graph Cache-Full Warning Spam

The default graph cache remains intentionally small on 16 GiB GPUs after the
128-entry cache experiment was rejected. With that default cache, decode falls
back to eager for uncached metadata shapes after the cache fills. Before this
follow-up, every fallback step emitted:

```text
CUDA graph capture skipped: paged metadata shape cache is full
```

That produced 369 warning lines during the same six-request, 64-token endpoint
probe. A quiet-log A/B showed this was a small but measurable default-path
throughput tax:

| path | cache-full warning lines | warmed seconds | completion tok/s |
| --- | ---: | ---: | ---: |
| default logging, repeated warn | 369 | 2.7687 | 23.12 |
| `KILN_LOG_LEVEL=error` | 0 | 2.7251 | 23.49 |
| patched default, warn once | 1 | 2.7435 | 23.33 |

The accepted follow-up keeps the first cache-full message at WARN so operators
still see the graph-cache state, then emits subsequent repeats at DEBUG. LoRA
adapter invalidation resets the one-shot warning flag because the graph cache is
cleared at the same time.

Validation:

```bash
cargo fmt --all --check
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" cargo build --release --features cuda --bin kiln
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" cargo test --release -p kiln-model --features cuda cuda_graph --lib
```

Focused test result:

```text
4 passed; 0 failed; 291 filtered out
```

## Follow-Up: Rejected CUDA Continuous-Batch and Native-MTP Defaults

After the RMSNorm and log-spam fixes, the next gap was concurrent/continuous
CUDA serving. The relevant routing facts:

- default non-streaming `/v1/chat/completions` does not use the live
  `DecodeBatcher`;
- `KILN_BATCHING_ENGINE=1` routes streaming and non-streaming completions
  through the heavier server-side batching actor;
- default streaming uses the live greedy `DecodeBatcher`, but CUDA defaults to
  `wait_us=0`, so it schedules single rows unless an explicit admission delay
  is configured.

Measured on the same WSL RTX 4090 Laptop setup with `KILN_NUM_BLOCKS=512`,
64-token greedy completions:

| path | concurrency | wall seconds | aggregate tok/s | notes |
| --- | ---: | ---: | ---: | --- |
| default non-streaming | 2 | 5.5668 | 22.99 | decode batcher counters stayed at 0 |
| `KILN_BATCHING_ENGINE=1` non-streaming | 2 | 19.2046 | 6.67 | rejected; residency rose to about 13.0 GiB |
| default streaming | 2 | 5.5358 | 23.12 | batcher ran, but max observed batch was 1 |
| streaming with `KILN_DECODE_BATCH_WAIT_US=1000` | 2 | 19.3178 | 6.63 | coalesced to batch 2, but was much slower |
| streaming with `KILN_DECODE_BATCHER=0` | 2 | 7.4672 | 17.14 | unique prompts; worse than default scheduler |

Conclusion: the lightweight default streaming batcher is useful as a single-row
scheduler, but the current CUDA multi-row GDN decode path is not shippable for
continuous batching. It repeatedly assembles/scatters batched linear-attention
state and loses the CUDA graph row fast path, causing a large latency and memory
regression. Do not enable a CUDA admission wait or the server batching actor by
default until the multi-row GDN state path is redesigned.

Native MTP was also tested as a short-greedy CUDA default candidate:

| path | warmed seconds | completion tok/s | correctness |
| --- | ---: | ---: | --- |
| explicit opt-out control, `KILN_SPEC_ENABLED=0` | 2.8463 | 22.49 | baseline |
| auto/default MTP prototype | 2.5971 | 24.64 | rejected; output parity failed |
| auto/default MTP prototype with `KILN_MTP_ARGMAX_FP32=1` | 2.6259 | 24.37 | rejected; still matched MTP output, not baseline |

The parity prompt was:

```text
Write exactly one sentence about a blue cube on a desk. parity-auto-2026-05-09
```

For `max_tokens=32`, `temperature=0`, non-speculative output began:

```text
Thinking Process!!!!!!!!
```

Native-MTP output began:

```text
Thinking Process:
```

Because the continuation changed, native MTP remains opt-in only. The temporary
auto-default router patch was reverted and no native-MTP default was shipped.

## Follow-Up: CUDA Graph Correctness Correction

The parity failure above was later narrowed to CUDA graph replay on the
server's graph-capturable CUDA stream, not native MTP. A direct diagnostic with
the server-style stream showed:

- no-graph eager decode matched the expected reasoning prefix:
  `Thinking Process:\n\n1.  **Analyze the Request:**`;
- graph-enabled decode produced the bad `Thinking Process!!!!!!!!` prefix;
- native MTP matched the eager/no-graph tokens for the same prompt.

The memory failure observed during this investigation was WSL/CUDA residency
pressure (`dxgkio_make_resident: Ioctl failed: -12`) and a CUDA allocation OOM
while uploading Qwen3.5-4B when projection originals were retained. It was not
a Linux OOM-killer event in `journalctl`/`dmesg`. The working 16 GiB server
profile still requires `KILN_DROP_PROJECTION_ORIGINALS=1`.

Accepted safety fix: make CUDA graphs opt-in by default
(`MemoryConfig::default().cuda_graphs = false`). `KILN_CUDA_GRAPHS=true` still
exists for experiments, but the default server path now uses eager/interleaved
decode until graph-capturable stream replay has a token-parity fix. This leaves
the training path and its FLCE/checkpointing memory guards unchanged.

Validation after the default change:

```bash
cargo fmt
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_TERM_COLOR=never cargo test -p kiln-server config::tests::test_defaults --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_TERM_COLOR=never cargo build --quiet --release --features cuda --bin kiln
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_TERM_COLOR=never cargo test -p kiln-train test_flce_parity_vs_naive_loss --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_TERM_COLOR=never cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet
```

The default server was started without `KILN_CUDA_GRAPHS` and with the accepted
16 GiB WSL CUDA memory profile. `/health` reported `blocks_free=512` and
`training_budget_gb=6.870269952`. The 32-token greedy parity prompt returned
the expected eager prefix:

```text
Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Topic: A blue cube on a desk.\n    *   Constraint:
```

Post-push live training smoke on the same pushed HEAD:

- temporary adapter dir: `/tmp/kiln-adapters-smoke-a4f0decb`;
- request: one SFT example, `epochs=1`, `lora_rank=1`, `lora_alpha=2.0`,
  `seed=1234`, `auto_load=false`;
- job id: `421d2e58-6604-42c0-bca7-4120a43ccd83`;
- result: `state=completed`, `progress=1.0`,
  `final_loss=0.9267818927764893`;
- training logs confirmed `gradient checkpointing enabled` with 32 one-layer
  segments and `fused_path="OFF"` for the RMSNorm training gate on the 16 GiB
  GPU.

## Follow-Up: CUDA BF16 Inference GDN State

Default CUDA inference now creates `LinearAttentionState` recurrent tensors in
the model dtype for BF16/FP16 inference. The training/test constructor remains
unchanged. This removes the per-token GDN decode cast cycle where each of the
24 GDN layers converted recurrent state F32 -> BF16 for the fused decode kernel
and then BF16 -> F32 after the layer. Since the BF16 kernel already quantizes
the state update each step, keeping inference state BF16 removes overhead
without changing the effective decode precision.

Rollback:

```text
KILN_DISABLE_CUDA_BF16_INFERENCE_STATE=1
```

Same-binary WSL RTX 4090 Laptop A/B, default CUDA graphs still off, 64-token
greedy chat completions:

| path | warmed seconds | completion tok/s |
| --- | ---: | ---: |
| rollback, `KILN_DISABLE_CUDA_BF16_INFERENCE_STATE=1` | 2.7920 | 22.92 |
| default BF16 inference state | 2.6879 | 23.81 |

Raw warmed timings:

```text
rollback: 2.730052, 2.746741, 2.719610, 2.836108, 2.927611
default:  2.694294, 2.683983, 2.681245, 2.699614, 2.680242
```

The 32-token greedy parity prompt still returned the expected eager prefix:

```text
Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Topic: A blue cube on a desk.\n    *   Constraint:
```

Validation:

```bash
cargo fmt
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_TERM_COLOR=never cargo test -p kiln-model linear_attention_state --lib --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_TERM_COLOR=never cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_TERM_COLOR=never cargo build --quiet --release --features cuda --bin kiln
```

Live training + LoRA inference smoke on the candidate binary:

- temporary adapter dir: `/tmp/kiln-adapters-bf16state-smoke`;
- request: one SFT example, `epochs=1`, `lora_rank=1`, `lora_alpha=2.0`,
  `seed=5678`, `auto_load=true`;
- job id: `37e08d7f-daff-4682-bd21-050b1d476f31`;
- result: `state=completed`, `progress=1.0`,
  `final_loss=1.0715702772140503`;
- `/health` reported `active_adapter="bf16-state-smoke"`;
- a LoRA-backed `/v1/chat/completions` request completed with the active
  adapter.

## Follow-Up: Guard CUDA Live Decode Batching

Current CUDA live streaming batching was remeasured after the BF16 inference
state change. With the default wait-zero policy, two concurrent 64-token
streaming requests still drain as one-row worker passes:

```text
KILN_DECODE_BATCH_WAIT_US=0
round walls: 5.4340s, 5.4869s, 5.3235s
metrics: submitted=393 batches=393 rows=393 max_observed_batch=1
```

Forcing a small admission delay still coalesces rows onto the slow CUDA
batch-2 GDN decode path. Two completed `KILN_DECODE_BATCH_WAIT_US=100`
64-token rounds took `18.1668s` and `17.7507s` before the sweep was stopped.
A short profiled wait-100 run confirmed `max_observed_batch=2`.

Accepted policy fix: CUDA now defaults the live decode batcher's backend
`max_batch` to 1 unless `KILN_DECODE_BATCH_MAX` is explicitly set. This keeps
the current fast rowwise scheduling even when an operator sets
`KILN_DECODE_BATCH_WAIT_US`, while still allowing forced A/B testing via
`KILN_DECODE_BATCH_MAX=2` or higher.

Candidate live WSL CUDA recheck with `KILN_DECODE_BATCH_WAIT_US=100` and no
explicit max override:

```text
round walls: 5.3862s, 5.3810s, 5.3344s
metrics: submitted=393 batches=393 rows=393 max_observed_batch=1
```

That is a 3.35x wall-time recovery versus the completed pre-guard wait-100
rounds (`17.9588s` average -> `5.3672s` average) and matches the default
wait-zero throughput envelope.

Validation:

```bash
cargo fmt
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_TERM_COLOR=never cargo test -p kiln-model decode_batcher_default_ --lib --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_TERM_COLOR=never cargo build --quiet --release --features cuda --bin kiln
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_TERM_COLOR=never cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet
```

The 32-token greedy parity prompt on the candidate still returned the expected
eager prefix:

```text
Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Topic: A blue cube on a desk.\n    *   Co
```

Live training + auto-loaded LoRA smoke on the candidate binary:

- temporary adapter dir: `/tmp/kiln-adapters-cuda-batcher-guard-smoke`;
- request: one SFT example, `epochs=1`, `lora_rank=1`, `lora_alpha=2.0`,
  `seed=6789`, `auto_load=true`;
- job id: `b76ac6a9-c65a-48fd-9856-af92355467be`;
- result: `state=completed`, `progress=1.0`,
  `final_loss=0.9538126587867737`;
- `/health` reported `active_adapter="cuda-batcher-guard-smoke"`;
- an adapter-backed `/v1/chat/completions` request completed.

## Follow-Up: CUDA GDN Decode Unexpanded Q/K

CUDA's fused GDN decode gates+recurrent kernel already accepts native GQA
Q/K heads (`q_heads <= value_heads`) and maps each value head back to its Q/K
group inside the kernel. The model forward path now keeps Q/K unexpanded for
the single-token CUDA fused decode path and only expands them if the backend
declines and the split recurrent fallback is needed. This removes the
GDN-decode Q/K GQA expansion from the hot path without changing prefill,
fallback, debug-tap, or training sequence paths.

Rollback:

```text
KILN_DISABLE_GDN_DECODE_UNEXPANDED_QK=1
```

Same-binary WSL RTX 4090 Laptop A/B, CUDA graphs unset, paged latency bench,
64 prompt tokens, 65 measured decode tokens:

| path | mean ITL runs (ms) | avg mean ITL | avg decode tok/s |
| --- | --- | ---: | ---: |
| rollback, `KILN_DISABLE_GDN_DECODE_UNEXPANDED_QK=1` | 41.064, 40.990, 41.042 | 41.032 ms | 24.371 |
| default unexpanded Q/K | 40.236, 40.418, 40.495 | 40.383 ms | 24.763 |

Average decode ITL improved by 1.6% on this WSL 16 GiB CUDA host. Prefill
stayed flat, as expected, because the new path is gated to `seq_len == 1`.

Validation:

```bash
cargo fmt --check
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo test -p kiln-gdn-kernel test_cuda_decode_gates_recurrent --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo test -p kiln-model test_causal_conv1d_update_matches_fallback --lib --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench
```

The `kiln-model` filter compiled the crate but matched zero tests in this
checkout. The CUDA kernel test now includes an explicit unexpanded-Q/K parity
case against the expanded split recurrent path.

Live training + auto-loaded LoRA smoke on the candidate binary:

- temporary adapter dir: `/tmp/kiln-adapters-unexpanded-qk-smoke`;
- request: one SFT example, `epochs=1`, `lora_rank=1`, `lora_alpha=2.0`,
  `seed=7890`, `auto_load=true`;
- job id: `fb0ed096-45cf-4617-b969-7b3d3423e804`;
- result: `state=completed`, `progress=1.0`,
  `final_loss=1.338575839996338`;
- `/health` reported `active_adapter="cuda-unexpanded-qk-smoke"`;
- adapter-backed `/v1/chat/completions` requests completed with and without
  thinking enabled.

## Follow-Up: CUDA Fused RoPE Q/K

The full-attention decode profile after unexpanded GDN Q/K showed RoPE as the
largest remaining non-GDN stage. CUDA now has a fused RoPE(Q,K) kernel for
contiguous bf16 Q/K tensors with precomputed f32 cos/sin tables. The model
forward path uses it for CUDA when Q/K are `[batch, seq, heads, head_dim]` and
tables are `[seq, rotary_dim / 2]`; unsupported shapes, dtypes, and backends
fall back to the existing Candle path. This covers the eager table-backed paged
forward path and the tensor-backed CUDA graph-compatible path without changing
CPU, Metal, or training fallback semantics.

Rollback:

```text
KILN_DISABLE_FUSED_CUDA_ROTARY_QK=1
```

Same-binary WSL RTX 4090 Laptop A/B, CUDA graphs unset, paged latency bench,
64 prompt tokens, 65 measured decode tokens:

| path | mean ITL runs (ms) | avg mean ITL | avg decode tok/s |
| --- | --- | ---: | ---: |
| rollback, `KILN_DISABLE_FUSED_CUDA_ROTARY_QK=1` | 40.443, 40.420, 40.621 | 40.495 ms | 24.695 |
| default fused RoPE(Q,K) | 37.838, 38.006, 38.111 | 37.985 ms | 26.326 |

Average decode ITL improved by 6.2% on this WSL 16 GiB CUDA host.

Validation:

```bash
cargo fmt --check
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo test -p kiln-rmsnorm-kernel rotary_qk_parity_qwen_shape --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench
```

The new CUDA kernel parity test uses Qwen-shaped Q/K head counts and exact BF16
equality against the existing contiguous-half RoPE reference.

Live training + auto-loaded LoRA smoke on the candidate binary:

- temporary adapter dir: `/tmp/kiln-adapters-rotary-qk-smoke-20260509-1903`;
- request: one SFT example, `epochs=1`, `lora_rank=1`, `lora_alpha=2.0`,
  `seed=8912`, `auto_load=true`;
- job id: `2fadfa06-78cd-4faa-9b60-66a4f0d08594`;
- result: `state=completed`, `progress=1.0`,
  `final_loss=1.5020499229431152`;
- `/health` reported `active_adapter="cuda-rotary-qk-smoke"`;
- adapter-backed `/v1/chat/completions` requests completed with and without
  thinking enabled.

## Follow-Up: CUDA Fused MLP SiLU Multiply

After fused RoPE(Q,K), the decode profile still showed the MLP middle as two
separate Candle operations: `gate_silu` and `hidden_mul`. CUDA now has a
forward-only bf16 kernel for `silu(gate) * up`, selected only for matching
contiguous CUDA bf16 tensors and only when the tensors are not tracking
autograd. Training and other unsupported paths stay on the existing Candle
ops, preserving gradient semantics.

Rollback:

```text
KILN_DISABLE_FUSED_CUDA_MLP_SILU_MUL=1
```

Same-binary WSL RTX 4090 Laptop A/B, CUDA graphs unset, paged latency bench,
64 prompt tokens, 65 measured decode tokens:

| path | mean ITL runs (ms) | avg mean ITL | avg decode tok/s |
| --- | --- | ---: | ---: |
| rollback, `KILN_DISABLE_FUSED_CUDA_MLP_SILU_MUL=1` | 37.928, 37.956, 38.354 | 38.080 ms | 26.261 |
| default fused MLP SiLU multiply | 37.030, 36.920, 36.982 | 36.977 ms | 27.044 |

Average decode ITL improved by 2.9% on this WSL 16 GiB CUDA host.

Validation:

```bash
cargo fmt --check
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo test -p kiln-rmsnorm-kernel mlp_silu_mul_parity_qwen_shape --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench
```

The new CUDA kernel parity test uses Qwen-shaped MLP intermediate width and
checks the fused bf16 result against the existing f32-reference SiLU/multiply
path cast back to bf16.

Live training + auto-loaded LoRA smoke on the candidate binary:

- temporary adapter dir: `/tmp/kiln-adapters-mlp-silu-mul-smoke-20260509`;
- request: one SFT example, `epochs=1`, `lora_rank=1`, `lora_alpha=2.0`,
  `seed=9123`, `auto_load=true`;
- job id: `992b70ec-0491-42ee-94fa-e3dc7afa8f72`;
- result: `state=completed`, `progress=1.0`,
  `final_loss=1.603104591369629`;
- `/health` reported `active_adapter="cuda-mlp-silu-mul-smoke"`;
- adapter-backed `/v1/chat/completions` requests completed with and without
  thinking enabled, including a no-thinking response of `kiln blue`.

## Follow-Up: CUDA Fold GDN QK Norm Into Decode Recurrent

After fused MLP SiLU multiply, the largest exposed GDN decode stage was
`qk_norm`. On CUDA decode, the causal-conv update produces raw F32 Q/K, the
split path L2-normalizes them, casts normalized Q/K to bf16, and then feeds
the fused gates+recurrent kernel. CUDA now has a fused single-token GDN path
that accepts raw F32 or bf16 unexpanded Q/K, performs the same L2
normalization and bf16 epilogue inside the gates+recurrent kernel, and skips
the separate tiny qk_norm launch and intermediate Q/K tensors. Gated RMSNorm
remains separate; this deliberately avoids repeating the previously rejected
recurrent+gated-RMSNorm fusion.

Rollback:

```text
KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT=1
```

Same-binary WSL RTX 4090 Laptop A/B, CUDA graphs unset, paged latency bench,
64 prompt tokens, 65 measured decode tokens:

| path | mean ITL runs (ms) | avg mean ITL | avg decode tok/s |
| --- | --- | ---: | ---: |
| rollback, `KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT=1` | 37.062, 37.194, 36.972 | 37.076 ms | 26.972 |
| default folded GDN QK norm + recurrent | 33.430, 33.378, 33.468 | 33.425 ms | 29.918 |

Average decode ITL improved by 9.8% on this WSL 16 GiB CUDA host.

A short profile confirmed the mechanism: `gdn_stage qk_norm` at `seq_len=1`
dropped from the previous post-MLP profile's `52.523ms` aggregate to
`0.139ms`, with the new `qk_norm_gates_recur` stage accounting for
`10.633ms` aggregate under profiling overhead.

Validation:

```bash
cargo fmt --check
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo test -p kiln-gdn-kernel test_cuda_decode --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench
```

The GDN kernel tests cover the existing bf16 decode gates+recurrent path, the
unexpanded-Q/K path, and the new F32-Q/K folded qk_norm+gates+recurrent path
against the split normalization/recurrent reference.

Live training + auto-loaded LoRA smoke on the candidate binary:

- temporary adapter dir: `/tmp/kiln-adapters-qk-norm-recurrent-smoke-20260509`;
- request: one SFT example, `epochs=1`, `lora_rank=1`, `lora_alpha=2.0`,
  `seed=9234`, `auto_load=true`;
- job id: `722a161d-35e6-4ba1-85ea-46240af21dd0`;
- result: `state=completed`, `progress=1.0`,
  `final_loss=1.5285838842391968`;
- `/health` reported `active_adapter="cuda-qk-norm-recurrent-smoke"`;
- adapter-backed `/v1/chat/completions` requests completed with and without
  thinking enabled, including a no-thinking response of `kiln green`.

## Follow-Up: CUDA Fused Attention Output Gate

After folded GDN QK norm + recurrent, the attention output gate remained as
two Candle ops on CUDA decode: sigmoid over the gate projection followed by an
elementwise multiply with the attention output. CUDA now has a forward-only
bf16 kernel for `x * sigmoid(gate)`, selected only for matching contiguous CUDA
bf16 tensors and only when neither tensor is tracked by Candle autograd.
Training and unsupported paths stay on the existing differentiable Candle ops.

Rollback:

```text
KILN_DISABLE_FUSED_CUDA_ATTN_SIGMOID_MUL=1
```

Same-binary WSL RTX 4090 Laptop A/B, CUDA graphs unset, paged latency bench,
64 prompt tokens, 65 measured decode tokens:

| path | mean ITL runs (ms) | avg mean ITL | avg decode tok/s |
| --- | --- | ---: | ---: |
| rollback, `KILN_DISABLE_FUSED_CUDA_ATTN_SIGMOID_MUL=1` | 33.144, 34.127, 33.457 | 33.576 ms | 29.783 |
| default fused attention gate | 33.064, 33.057, 32.802 | 32.974 ms | 30.327 |

Average decode ITL improved by 1.8% on this WSL 16 GiB CUDA host.

Validation:

```bash
cargo fmt --check
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo test -p kiln-rmsnorm-kernel sigmoid_mul_parity_qwen_attn_gate_shape --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet
PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" CARGO_BUILD_JOBS=1 cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench
```

The new CUDA kernel parity test uses Qwen attention-gate shape `[1, 2, 4096]`
and checks the fused bf16 result against the existing f32-reference
sigmoid/multiply path cast back to bf16.

Live training + auto-loaded LoRA smoke on the candidate binary:

- temporary adapter dir:
  `/tmp/kiln-adapters-attn-sigmoid-mul-smoke-20260510`;
- request: one SFT example, `epochs=1`, `lora_rank=1`, `lora_alpha=2.0`,
  `seed=9234`, `auto_load=true`;
- job id: `ed56d6c3-efe1-4102-babf-6f811c911a59`;
- result: `state=completed`, `progress=1.0`,
  `final_loss=1.5107368230819702`;
- `/health` reported `active_adapter="attn-sigmoid-mul-smoke-r1"`;
- adapter-backed `/v1/chat/completions` requests completed with and without
  thinking enabled, including a no-thinking response of `kiln amber`.
