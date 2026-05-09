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
