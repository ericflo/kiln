# Phase 5 — bs>1 batched CUDA graph fault: compute-sanitizer memcheck trace (#1082)

Snapshot taken on `main` at `81b963cb`. Reproduces and pins the
`CUDA_ERROR_ILLEGAL_ADDRESS` documented in `cuda-graph-status.md`
section "0. Re-validate `KILN_CUDA_GRAPHS_BATCHED=1`" as the highest-
priority remaining Phase 5 task.

## TL;DR

The bs>1 batched CUDA graph fault is **dangling device pointers from
intra-graph `Tensor::from_slice` allocations**, not stream mismatch or
GDN-state drift. `compute-sanitizer --tool memcheck` traces the first
illegal read to a candle `ucopy_f32` kernel **inside `cuGraphLaunch`**
on the very first replay (zero graph re-runs needed to trigger), with
addresses up to **906 MiB past** the nearest live allocation. The
buffers being read are the per-step `block_table_tensor` and
`seqused_k_tensor` that `CachedPagedDecodeMeta::build` allocates fresh
inside the captured forward via `Tensor::from_slice`; when the local
`CachedPagedDecodeMeta` drops at the end of capture, candle's
`cudaFree` releases the storage that the captured kernels still
reference.

The fix is **not a one-liner** — it requires threading the stable
`graph_inputs.block_table` and `graph_inputs.seqused_k` device pointers
from `BatchedPagedDecodeGraphInputs` (already pre-allocated and
populated by the runner) into `model_forward_paged_decode_contiguous_batch_hidden_inner`
so the inner forward bypasses `CachedPagedDecodeMeta::build` on the
capture path. Documenting the trace + analysis here so the follow-up
PR is mechanical.

## Reproduction

A6000, kiln `main` @ `7edde0f0` (kiln-runpod image
`ghcr.io/ericflo/kiln-runpod:latest`):

```bash
# build
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln
# heal sccache for kiln-gdn-kernel (issue #1066) if /tmp/server.log
# surfaces `kiln_gdn_gates_bf16 failed with status 500`
SCCACHE_RECACHE=1 cargo build --release --features cuda --bin kiln \
  # (or scope per-package — there is no `kiln-gdn-kernel.cuda` feature)

# server, both bs=1 and bs>1 graphs ON, plus a non-default live-batch
# admission window so 4 concurrent requests collapse into one bs=4 step
KILN_MODEL_PATH=/workspace/Qwen3.5-4B \
KILN_CUDA_GRAPHS=true KILN_CUDA_GRAPHS_BATCHED=1 \
KILN_DECODE_BATCH_MAX=4 KILN_DECODE_BATCH_WAIT_US=5000 KILN_DECODE_BATCH_MIXED_SEQ=1 \
RUST_LOG=kiln=info,kiln_model::cuda_graph=trace \
compute-sanitizer --tool memcheck --print-limit 5 --launch-timeout 600 \
  --error-exitcode 42 ./target/release/kiln serve

# concurrent driver — bs=1 warmup then 4 chat requests
curl -s -X POST http://127.0.0.1:8420/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-4B","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"temperature":0}'

for i in 1 2 3 4; do
  curl -s -X POST http://127.0.0.1:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"Qwen3.5-4B\",\"messages\":[{\"role\":\"user\",\"content\":\"Say $i\"}],\"max_tokens\":8,\"temperature\":0}" &
done
wait
```

`KILN_DECODE_BATCH_MAX=4` is required: on CUDA the live greedy decode
batcher defaults `max_batch=1` (see `default_decode_batcher_max_batch`
in `crates/kiln-model/src/generate.rs:577`) so concurrent requests
never collapse into a bs>1 decode step otherwise.

`KILN_DECODE_BATCH_WAIT_US=5000` plus `KILN_DECODE_BATCH_MIXED_SEQ=1`
gives the admission window enough room to absorb the 4 same-position
requests into one bs=4 batched decode step and tolerates the per-row
position skew that `chat_template` injects.

## Timeline (kiln_model::cuda_graph trace)

```
04:08:21.124  INFO  CUDA graph captured for decode (32 layers)             # bs=1 capture (single-shape warmup)
04:08:21.785  INFO  CUDA graph captured for batched decode batch_size=4 max_seqlen_k=128
                    # ← argmax over captured output_logits at line 1616 SUCCEEDS (capture phase)
04:08:21.785+ ===== Invalid __global__ read of size 4 bytes ============   # FIRST replay (cuGraphLaunch) FAULTS
                    at ucopy_f32+0x700
                    by thread (32,0,0) in block (0,0,0)
                    Address 0x2fa8022600 is out of bounds
                    and is 805364225 bytes after the nearest allocation
                    at 0x2f7800f400 of size 20480 bytes
```

Five reports follow with identical kernel + base address but different
thread IDs (32, 33, 34, 35, 36) and per-thread addresses spaced by
`0x180_0000` ≈ 25 MiB — `ucopy_f32` strides through device memory
using indices that were valid at capture time and are wildly out of
range at replay time. `--print-limit 5` truncates at 5 reports; the
final `ERROR SUMMARY` line says **5504 errors** total over the single
replay attempt (one per faulting thread within the launched grid).

## Faulting host backtrace (verbatim)

```
Host Frame:cuGraphLaunch                                                  [libcuda.so.1]
Host Frame:cudarc::driver::safe::graph::CudaGraph::launch                 [/target/release/kiln]
Host Frame:kiln_model::cuda_graph::CudaGraphRunner::decode_step_paged_batched
Host Frame:kiln_model::generate::ModelRunner::paged_batched_decode_step
Host Frame:<kiln_server::batching_engine::RealDecodeForward as DecodeForward>::forward_decode
Host Frame:std::sys::backtrace::__rust_begin_short_backtrace
Host Frame:core::ops::function::FnOnce::call_once{{vtable.shim}}
Host Frame:_RNvNvMs0_NtNtNtCsjrHSEGnQ3l9_3std3sys6thread4unixNtB7_6Thread3new12thread_start
Host Frame: [libc.so.6 +0x94ac2]
Host Frame:clone                                                          [libc.so.6]
```

The kernel-launch host backtrace points at **`cuGraphLaunch`**, which
means the faulting kernel was a node baked into the captured graph at
capture time — not a kernel queued after replay. The post-replay
argmax that the audit suggested is therefore *not* the proximate
cause; it merely surfaces the fault when its DtoH copy hits the
poisoned device context.

## Surfaced error chain (server response)

```
Text generation failed: batched decode forward pass failed:
DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered")
   0: candle_core::error::Error::bt
   1: <core::result::Result<O,E> as candle_core::cuda_backend::error::WrapErr<O>>::w
   2: <candle_core::cuda_backend::device::CudaDevice as BackendDevice>::storage_from_cpu_storage_owned
   3: candle_core::device::Device::storage
   4: candle_core::tensor::Tensor::new
   5: kiln_model::forward::embedding_lookup
   6: kiln_model::forward::model_forward_paged_batched_decode_hidden
   7: kiln_model::generate::ModelRunner::paged_batched_decode_step
   8: <kiln_server::batching_engine::RealDecodeForward as DecodeForward>::forward_decode
```

The chain points at `embedding_lookup → Tensor::new → storage_from_cpu_storage_owned`
in the **eager fallback** path that runs after `decode_step_paged_batched`
returns `Ok(None)` from the post-replay argmax error handler — the
CUDA context is already poisoned by the time the eager retry tries to
HtoD-copy a fresh token-id tensor, so the synchronous `storage_from_cpu_storage_owned`
inherits the sticky error. **This chain is downstream of the real
fault**; compute-sanitizer's `cuGraphLaunch` frame is the proximate
site.

## Root-cause analysis

### Layout of the bs>1 capture wrapper

`CudaGraphRunner::try_capture_batched` (`crates/kiln-model/src/cuda_graph.rs:1457`)
pre-allocates every "stable" device buffer the captured graph will
read from or write to **before** `begin_capture`:

```rust
let token_buffer = Self::new_batched_token_buffer(device, token_ids)?;
let position_buffer = Self::new_batched_position_buffer(...)?;
let rotary_cos_buffer = Self::new_batched_rotary_cos_buffer(...)?;
let rotary_sin_buffer = Self::new_batched_rotary_sin_buffer(...)?;
let block_table_buffer = Self::new_batched_block_table_buffer(...)?;
let seqused_k_buffer = Self::new_batched_seqused_k_buffer(...)?;
let kv_slot_buffer = Self::new_batched_kv_slot_buffer(...)?;
let output_logits = Self::new_batched_output_logits(...)?;
let (paged_decode_outputs, paged_decode_lse) = Self::new_batched_paged_decode_outputs(...)?;
let gdn_decode_outputs = Self::new_batched_gdn_decode_outputs(...)?;
```

These are bundled into `BatchedPagedDecodeGraphInputs` and passed
into `model_forward_paged_batched_with_graph_inputs` (`forward.rs:19547`),
which in turn calls the inner contiguous-batch forward:

```rust
let hidden = model_forward_paged_decode_contiguous_batch_hidden_inner(
    backend, input_tokens, weights, config, paged_cache,
    block_tables, sequence_lengths,
    Some(graph_inputs.linear_state),
    lora,
    Some(graph_inputs.positions),   // ← stable
    Some(graph_inputs.token_ids),   // ← stable
)?;
```

**The wrapper only threads `positions` and `token_ids` through; the
other stable buffers in `graph_inputs` (`block_table`, `seqused_k`,
`kv_slot`, `rotary_cos`, `rotary_sin`, `attn_out`, `softmax_lse`,
`output_logits`) are never reached by the inner forward call.**

### The fresh-allocation site

Inside `model_forward_paged_decode_contiguous_batch_hidden_inner`
(`forward.rs:17959`), once per step:

```rust
let cached_paged_meta: Option<CachedPagedDecodeMeta> = if has_full_attention_layer {
    Some(
        CachedPagedDecodeMeta::build(device, paged_cache, block_tables, start_positions)
            .context("build cached paged decode metadata for batched step")?,
    )
} else {
    None
};
```

`CachedPagedDecodeMeta::build` (`forward.rs:15512`) allocates a fresh
`[batch, max_blocks_per_seq] u32` `block_table_tensor` and a
`[batch] i32` `seqused_k_tensor` via `Tensor::from_slice` on every
call. During capture this allocation happens *inside* the captured
stream window, but the underlying storage is candle's regular
`cudaMalloc`-backed `CudaStorage` — **not** `cuMemAllocAsync`. The
captured kernels record the device pointer as an immediate kernel
argument.

When `try_capture_batched` returns, the function-local
`CachedPagedDecodeMeta` (held inside the inner forward) goes out of
scope and is dropped. Candle's `Drop` for `CudaStorage` calls
`cudaFree`, releasing the storage. The captured graph still holds
those pointers in its node arguments.

Result: every subsequent `cuGraphLaunch` reads from freed memory at
the `ucopy_f32` site, where some downstream op (likely a `seqused_k`
gather or a `block_table` slice into the paged-attention pipeline)
strides off the end of whatever else now occupies that address space.

### Why bs=1 doesn't have this bug

The bs=1 wrapper `model_forward_paged_with_graph_inputs`
(`forward.rs:19491`) passes `graph_inputs: Option<&PagedDecodeGraphInputs<'_>>`
all the way down to `model_forward_paged_inner`, which threads
`block_table`, `seqused_k`, `kv_slot`, `rotary_cos`, `rotary_sin`,
`attn_out`, and `softmax_lse` into every paged-attention call site.
The bs=1 path *never* calls `CachedPagedDecodeMeta::build` from inside
the captured forward — the captured kernels read directly from the
runner-owned stable buffers.

The bs>1 capture wrapper was written to plug into the existing eager
batched contiguous-decode hot path (`model_forward_paged_decode_contiguous_batch_hidden_inner`),
which was designed to compute its own per-step meta. The graph-input
plumbing for the meta tensors never landed.

### Why argmax-at-capture-time worked

Capture-time argmax at `cuda_graph.rs:1616` succeeded because at that
moment the `CachedPagedDecodeMeta` is still live (it's owned by the
in-flight closure scope) and the post-capture argmax kernel reads
`output_logits` (a stable, pre-allocated tensor) — not the meta
tensors. Replay-time argmax fails not because of stream mismatch but
because the `cuGraphLaunch` itself segfaults first, poisoning the
context and surfacing the error when the next op (the argmax DtoH)
synchronizes.

### Allocation-size sanity check

The nearest allocation in the report is **20 480 bytes**. The bs=4
faulting context candidates:

- `block_table_tensor`: `[4, max_blocks_per_seq]` u32. At
  `max_seqlen_k=128` and the default paged block size 16 →
  `max_blocks_per_seq = 8`, so `4 × 8 × 4 = 128 bytes`. Too small.
- `seqused_k_tensor`: `[4] i32` = 16 bytes. Too small.
- `output_logits`: `[4, 1, vocab_size=151_936]` BF16
  = `4 × 151_936 × 2 = 1 215 488 bytes`. Too big.
- `paged_decode_outputs[i]`: `[4, 1, n_heads=16, head_dim=256]` BF16
  = `4 × 16 × 256 × 2 = 32 768 bytes`. Closest live candidate.
- 20 480 = `4 × 16 × 320` or `4 × 5120` — likely a candle internal
  scratch tensor (RoPE working buffer or similar), so the "nearest"
  base is a still-live tensor that *isn't* the offending one. The
  faulting kernel's expected base lives at much higher addresses
  (well after the 20 KB scratch), which is consistent with a freed
  allocation whose virtual address has been recycled.

## Suspect ruled-in / ruled-out

| Suspect (per `cuda-graph-status.md`) | Ruled-in | Notes |
|---|---|---|
| Intra-graph allocations the runner does not pin | **YES** | `CachedPagedDecodeMeta::build` allocates `block_table_tensor` + `seqused_k_tensor` via candle `Tensor::from_slice` (regular `cudaMalloc`) inside the captured forward; storage is freed when the per-call local drops. |
| Stream mismatch on post-replay argmax | NO | `cuGraphLaunch` itself faults — the argmax error is downstream. Both capture and replay run on `cuda_dev.cuda_stream()`. |
| GDN-state pointer drift | NO | The `batched_state_pool` slot is owned by the runner across replays; `Self::prepare_gdn_recurrent_state_for_capture` + `with_decode_gates_recurrent_outputs` install stable pointers before capture begins. No GDN-state kernel appears in the host backtrace at the fault. |

## Proposed fix (sketch — out of scope for this commit)

1. Extend `model_forward_paged_decode_contiguous_batch_hidden_inner`
   with two new optional parameters
   `stable_block_table_gpu: Option<&Tensor>` and
   `stable_seqused_k_gpu: Option<&Tensor>`. When `Some`, the function
   builds a hybrid `CachedPagedDecodeMeta`-like value that reuses the
   caller-supplied stable tensors and only computes the host-side
   bookkeeping (`max_seqlen_k`, `max_blocks_per_seq`,
   `uniform_start_pos`, `strict_start_slots`).
2. Update `model_forward_paged_batched_with_graph_inputs` to pass
   `graph_inputs.block_table` and `graph_inputs.seqused_k` into the
   new parameters.
3. Audit the rest of the inner forward (RoPE table builds, paged
   attention per-layer scratch, embedding-lookup intermediate, MLP
   working buffers, residual scratch) for any other
   `Tensor::from_slice` / `Tensor::zeros` calls that produce capture-
   transient pointers, and thread `graph_inputs.rotary_cos`,
   `graph_inputs.rotary_sin`, `graph_inputs.attn_out`,
   `graph_inputs.softmax_lse`, and `graph_inputs.kv_slot` through to
   the per-layer sites accordingly. The bs=1 forward
   (`model_forward_paged_inner`) is the existing pattern.
4. Re-run the same memcheck repro after the fix and confirm the
   `Invalid __global__ read` count drops to zero across a full bs=4
   chat-completion exchange.

Step 1 + Step 2 alone should clear the *currently observed* fault
(the meta tensors are the ones that match the reported pointer
geometry). Step 3 is needed before flipping `KILN_CUDA_GRAPHS_BATCHED=1`
to default-on because any remaining transient allocation will reproduce
the same class of fault on a different decode shape.

## What this rules out for future planners

- It is **not** worth re-investigating "stream mismatch on the
  post-replay argmax" (PR #176-class hypothesis) — the captured
  kernels themselves are reading freed memory; the argmax surface is
  a symptom.
- It is **not** worth re-investigating "GDN-state pointer drift"
  (`d564067c`-era hypothesis) — the state pool is correctly pinned
  via `with_decode_gates_recurrent_outputs` + the persistent batched
  state slot, and no GDN kernel is on the faulting backtrace.
- The single right answer is "audit every per-step host→device build
  in the inner contiguous-batch forward and replace it with the
  graph-input stable pointer." This is the same discipline the bs=1
  forward already follows.

## Artifacts

- compute-sanitizer log: `bench-results/cuda-graph-bs2-memcheck.log`
  (5504 errors, 195 lines of host backtraces, captured in this run).
- Plain server log (no sanitizer wrapper) reproducing the same
  surface error against `RUST_LOG=kiln_model::cuda_graph=trace`:
  `bench-results/cuda-graph-bs2-server.log`.

Both pulled from a fresh A6000 pod (`kiln-1082-sub-bs2fault`,
RunPod `76vdxmxymztzs3`, kiln-runpod image
`ghcr.io/ericflo/kiln-runpod:latest`). Pod terminated at end of
investigation.
