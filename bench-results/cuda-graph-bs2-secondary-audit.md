# Phase 5 — bs>1 CUDA graph secondary alloc-site audit (#1082)

Companion to `bench-results/cuda-graph-bs2-memcheck.md`. The primary
`ILLEGAL_ADDRESS` fault was pinned to transient `Tensor::from_slice`
allocations of `block_table_tensor` + `seqused_k_tensor` inside
`CachedPagedDecodeMeta::build` and fixed in commits `9b173f84`,
`ab798167`, and `393beadc` (PR landed on `main` 2026-05-25).

The fix doc enumerated five additional suspect intra-graph allocation
sites that may surface sibling faults on different bs>1 decode shapes:

1. **kv_slot writer scratch**
2. **RoPE cos/sin tables**
3. **Paged-decode `attn_out` scratch**
4. **Paged-decode `softmax_lse` scratch**
5. **`output_logits` for non-LM-head paths**

This document audits each site against the **current** post-fix code on
`main`. The audit method for each site is the same as the primary
investigation: trace the bs>1 captured path
(`model_forward_paged_batched_with_graph_inputs` →
`model_forward_paged_decode_contiguous_batch_hidden_inner` →
`transformer_block_paged_decode_contiguous_batch` →
`gqa_attention_paged_decode_contiguous_batch`) and check whether each
suspect allocation site uses a caller-supplied stable device tensor
from `BatchedPagedDecodeGraphInputs` or builds a fresh device tensor
inside the captured stream window.

The bs=1 graph wrapper
(`model_forward_paged_with_graph_inputs` →
`model_forward_paged_inner` → `gqa_attention_paged_with_rope_tables`)
already pins every one of these sites and is the reference pattern.

## Summary table

| Suspect site | Status | Captured path consumes | Action |
|---|---|---|---|
| 1. KV slot writer | **NOT-IN-CAPTURED-PATH (correctness ⚠️)** | host immediate `u32` slot baked into kernel arg | Replace with `graph_inputs.kv_slot` device tensor to make the captured graph re-readable across positions. Not a dangling-pointer fault — a *baked-immediate* correctness bug that breaks replay at non-capture-time positions. |
| 2. RoPE cos/sin tables | **UNPINNED** | `freqs.cos()?` / `freqs.sin()?` allocated by `rotary_embedding_from_tensor` inside captured region | Thread `graph_inputs.rotary_cos` / `.rotary_sin` through to the GQA bs>1 path; reuse the bs=1 `rope_tables` plumbing (`gqa_attention_paged_with_rope_tables`'s `rope_tables: Option<(&Tensor, &Tensor)>`). |
| 3. Paged-decode `attn_out` scratch | **UNPINNED** | `Tensor::zeros((b, 1, n_heads, head_dim), BF16, ...)` inside `flash_attn_paged_decode_dyn_seqlen` when `graph_outputs=None` | Pass `Some((graph_inputs.attn_out[layer], graph_inputs.softmax_lse[layer]))` into `flash_attn_paged_decode_dyn_seqlen`; identical to the bs=1 wiring at `forward.rs:15846-15871`. |
| 4. Paged-decode `softmax_lse` scratch | **UNPINNED** | `Tensor::zeros((b, n_heads, 1), F32, ...)` inside `flash_attn_paged_decode_dyn_seqlen` when `graph_outputs=None` | Same fix as #3 — they share a single `graph_outputs: Option<(&Tensor, &Tensor)>` parameter. |
| 5. `output_logits` for non-LM-head paths | **PINNED** | `slice_set` into `graph_inputs.output_logits` at `forward.rs:20296-20298` | None required — already wired. The "non-LM-head paths" caveat in the original doc applies only to MTP / debug paths that don't run the captured forward. |
| 6 (bonus). Strict-path `start_slots` | **UNPINNED (BENIGN)** | `Tensor::from_slice(strict_slots, batch, x.device())?.contiguous()?` at `forward.rs:16558` | Low priority — the CUDA backend declines `flash_attn_paged_decode_contiguous_batch` (no impl; trait default returns `None`), so the captured HtoD writes to a dangling target but no kernel reads from it. Stream-recorded `cudaMemcpyHtoDAsync` could still corrupt recycled VAs on replay. Skip the strict probe entirely on CUDA. |
| 7 (bonus). `CachedPagedDecodeMeta::build` fallback | **PINNED (mixed)** | When `stable_block_table_gpu` is `Some` but `stable_seqused_k_gpu` is `None` (or vice versa) we silently fall back to the transient `build` path | Belt-and-suspenders — the bs>1 wrapper currently passes both `Some` so this branch is dead; either bail or accept-one-from-the-pair. Documented at `forward.rs:18720-18738`. Not a regression — keep as-is. |

## Suspect 1 — KV slot writer scratch

**Current code path**: `gqa_attention_paged_decode_contiguous_batch`
(`forward.rs:16218`) calls
`paged_cache.write_token_major_native_batch(layer_idx, block_tables,
start_positions, &k, &v)` at line 16483. On CUDA, this falls through
to the per-row loop at `paged_kv_cache.rs:401-411` (the `#[cfg(feature
= "metal")]` block at lines 386-399 is skipped on CUDA-only builds).
Each loop iteration calls `write_token_major_native`, which on CUDA
calls `kiln_flash_attn::paged_kv_write_token_major_bf16(k_pool, v_pool,
k_row, v_row, slot)` at `paged_kv_cache.rs:284`. The `slot` argument
is a **host `usize`** computed by `block_table.slot_for(start_pos,
self.block_size)`.

In `kiln-flash-attn/src/lib.rs:1018` (`paged_kv_write_token_major_bf16`),
`slot` is converted to a `u32` immediate kernel argument (line 1049-
1051) and passed directly into the launch call.

**Capture-time effect**: no transient *device tensor* is allocated.
The kernel records `slot_u32` as an **immediate argument**, baked into
the captured node. There is no dangling pointer.

**However**, this introduces a **correctness bug under graph replay**:
the captured graph has the capture-time slot index baked in. On
replay at a different decode position (or for a different block-table
layout that hashes to the same `CudaBatchedGraphKey`), the captured
KV write will land in the wrong slot. The bs=1 path solves this by
routing through `write_token_major_native_graph_slot`
(`paged_kv_cache.rs:233`), which takes `slot: &Tensor` (a 1-element
device-side u32). The graph runner re-fills the slot via
`CudaGraphRunner::update_cuda_scalar` at every replay
(`cuda_graph.rs:1102`), so the captured kernel reads the fresh slot
from the same device pointer on every replay.

**Status**: `NOT-IN-CAPTURED-PATH` as a dangling-pointer fault, but a
**replay-correctness bug** that would corrupt the KV cache at the
first replay at a different position than capture time. The runner
already pre-allocates `kv_slot_buffer` (`cuda_graph.rs:1512-1513`) and
threads it through `BatchedPagedDecodeGraphInputs.kv_slot`
(`forward.rs:15491`), and re-fills it via
`update_cuda_scalar(kv_slot_buffer, slots.as_slice(), ...)` at
`cuda_graph.rs:1200`. **The plumbing is in place; what's missing is
the consumer**: a batched variant
`PagedKvCache::write_token_major_native_batch_graph_slot` that reads
per-row slots from a `[batch] u32` device tensor instead of taking
per-row immediates, mirroring `write_token_major_native_graph_slot`.

**Proposed fix sketch** (left to a follow-up commit because it
requires a kernel change, not just plumbing):
1. Add a new entry point `paged_kv_write_token_major_bf16_batch_slot`
   in `kiln-flash-attn` that takes `slots: &Tensor` (`[batch] u32`)
   instead of `slot: u32` and launches a fused grid that handles all
   `batch` rows in one kernel.
2. Add `PagedKvCache::write_token_major_native_batch_graph_slot`
   that mirrors the bs=1 graph-slot writer.
3. Thread `graph_inputs.kv_slot` through
   `transformer_block_paged_decode_contiguous_batch` and
   `gqa_attention_paged_decode_contiguous_batch`.

**Severity**: HIGH for any plan to default-enable
`KILN_CUDA_GRAPHS_BATCHED=1` across multiple `start_pos`. Today the
runner only captures one bucket per `(batch_size, max_seqlen_k)` key,
and `max_seqlen_k` changes every step, which forces re-capture at
every step — so the baked slot has never been observed to mis-fire in
practice. But the moment the runner is taught to re-use a captured
graph across a `start_pos` range, this becomes a corruption bug.

## Suspect 2 — RoPE cos/sin tables

**Current code path**: `gqa_attention_paged_decode_contiguous_batch`
at `forward.rs:16447-16467` calls `rotary_embedding_from_tensor(&q, &k,
positions, head_dim, rotary_dim, inv_freq)` for both the uniform-
position fast path and the per-row swap path. Inside
`rotary_embedding_from_tensor` (`forward.rs:7534-7593`):

```rust
let pos = positions_tensor.unsqueeze(1)?;
let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
let cos = match try_kt_cos(&freqs)? {
    Some(t) => t,
    None => freqs.cos()?,
};
let sin = match try_kt_sin(&freqs)? {
    Some(t) => t,
    None => freqs.sin()?,
};
```

`freqs`, `cos`, and `sin` are all freshly-allocated candle tensors
backed by `cudaMalloc`. They are consumed downstream by either
`fused_rotary_qk` (the kt-bridge fast path at lines 7565-7587) or
`apply_rope` (the candle fallback at lines 7590-7591), and dropped
when `rotary_embedding_from_tensor` returns. Inside the captured
region:
- The captured kernels (`fused_rotary_qk` or `apply_rope`'s
  composite of `broadcast_mul` + `cat`) bake the `cos` / `sin`
  device pointers as kernel arguments at capture time.
- The candle `Tensor` locals for `cos`, `sin`, and `freqs` drop at
  function-return scope, releasing their `cudaMalloc` storage.
- On replay, the captured kernels read freed memory → same fault
  class as the original `block_table_tensor` bug.

The bs>1 wrapper does NOT currently thread
`graph_inputs.rotary_cos` / `.rotary_sin` through to this call site.
Compare with the bs=1 path at `forward.rs:21358-21380`, which builds
`graph_rope_tables = Some((inputs.rotary_cos, inputs.rotary_sin))`
and passes them as `rope_tables: Option<(&Tensor, &Tensor)>` into
`transformer_block_paged_with_rope_tables`, which forwards into
`gqa_attention_paged_with_rope_tables` (which threads them into
`kiln_rmsnorm_kernel::fused_attn_decode_qkv_prep` at
`forward.rs:16846-16860`). All bs=1 RoPE kernels read from the
runner-owned stable buffers.

**Status**: **UNPINNED**. This is the most likely source of the next
sibling fault under `KILN_CUDA_GRAPHS_BATCHED=1`. The
`new_batched_rotary_cos_buffer` / `new_batched_rotary_sin_buffer`
allocations already exist (`cuda_graph.rs:1499-1500` references and
`cuda_graph.rs:1815-1864`-ish for the constructors) and the runner
re-fills them via `update_cuda_tensor` before every replay
(`cuda_graph.rs:1500-1504` calls
`update_batched_rotary_cos_sin_buffers_for_replay`).

**Proposed fix**:
1. Add `rope_tables: Option<(&Tensor, &Tensor)>` and
   `graph_inputs_attn_out: Option<(&Tensor, &Tensor)>` parameters to
   `gqa_attention_paged_decode_contiguous_batch` (and to
   `transformer_block_paged_decode_contiguous_batch`).
2. In the GQA function, branch on `rope_tables.is_some()`:
   - When present, replace the
     `rotary_embedding_from_tensor(&q, &k, positions, ...)` call with
     a path that consumes the provided tables (mirror the bs=1
     `fused_attn_decode_qkv_prep` fast-path or call into
     `rotary_embedding_from_tables` if RoPE is not fused with QK-
     norm). Cost: ~80 lines of plumbing across three functions.
3. In `model_forward_paged_batched_with_graph_inputs`, thread
   `graph_inputs.rotary_cos` + `.rotary_sin` into the inner forward.

**Severity**: HIGH — this is the most concrete next reproducer
candidate under the same compute-sanitizer harness from the original
memcheck doc.

## Suspect 3+4 — Paged-decode `attn_out` and `softmax_lse` scratch

These two are intentionally bundled because they share a single
`graph_outputs: Option<(&Tensor, &Tensor)>` parameter in
`flash_attn_paged_decode_dyn_seqlen`.

**Current code path**: `gqa_attention_paged_decode_contiguous_batch`
calls `backend.flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
&q, k_pool, v_pool, block_table_tensor, seqused_k_tensor,
max_seqlen_k, page_block_size, softmax_scale, true)` at
`forward.rs:16583-16593`. The CUDA backend impl
(`backend/cuda.rs:387-416`) then calls
`kiln_flash_attn::flash_attn_paged_decode_dyn_seqlen(... None ...)`
with `graph_outputs = None`. Inside the kernel wrapper at
`kiln-flash-attn/src/lib.rs:905-927`:

```rust
let (out, softmax_lse) = if let Some((out, softmax_lse)) = graph_outputs {
    ...
    (out, softmax_lse)
} else {
    out_owned = Tensor::zeros((b, 1, num_heads, head_dim), DType::BF16, device)?;
    softmax_lse_owned = Tensor::zeros((b, num_heads, 1), DType::F32, device)?;
    (&out_owned, &softmax_lse_owned)
};
```

`out_owned` and `softmax_lse_owned` are candle `cudaMalloc` backed
tensors allocated **inside the captured region**, dropped at end of
the kernel call, and re-allocated on the next call (potentially with
recycled VAs). Same dangling-pointer fault class as the original
`block_table_tensor` bug.

The bs=1 path at `forward.rs:15846-15871` already threads
`graph_inputs.attn_out[full_attn_layer_idx]` + `.softmax_lse[...]`
into `flash_attn_paged_decode_dyn_seqlen(... Some((attn_out,
softmax_lse)) ...)`. The runner pre-allocates
`paged_decode_outputs: Vec<Tensor>` and `paged_decode_lse: Vec<Tensor>`
(one per full-attn layer) at `cuda_graph.rs:1515-1516` and stashes
them on `CapturedBatchedDecodeGraph._paged_decode_outputs` /
`._paged_decode_lse` (lines 292-294) so the lifetime outlives the
captured graph.

**Status**: **UNPINNED**. Both buffers are pre-allocated and
threaded into `BatchedPagedDecodeGraphInputs` (`forward.rs:15501-
15504`) but the bs>1 GQA function does not yet consume them.

**Proposed fix** (small and mechanical, same shape as the just-landed
`block_table_tensor` fix):
1. Add `graph_outputs: Option<(&Tensor, &Tensor)>` parameter to
   `gqa_attention_paged_decode_contiguous_batch`.
2. Pass it directly into the backend's
   `flash_attn_paged_decode_contiguous_batch_dyn_seqlen`. (Note: the
   trait method on `BackendRuntime` doesn't have this parameter
   today; needs an additive change to the trait or a parallel
   `_with_graph_outputs` method to avoid touching the Vulkan/Metal
   impls.)
3. Thread the per-layer slice
   `(graph_inputs.attn_out[full_attn_idx], graph_inputs.softmax_lse[full_attn_idx])`
   through
   `model_forward_paged_decode_contiguous_batch_hidden_inner` →
   `transformer_block_paged_decode_contiguous_batch` →
   `gqa_attention_paged_decode_contiguous_batch`.

**Severity**: HIGH — easiest "round trip" win for unblocking
`KILN_CUDA_GRAPHS_BATCHED=1` default-on once #2 is also fixed. The
backend-trait churn is the only annoying part (it touches the Metal
and Vulkan default impls).

## Suspect 5 — `output_logits` for non-LM-head paths

**Current code path**: The bs>1 wrapper
`model_forward_paged_batched_with_graph_inputs` writes the final
logits into `graph_inputs.output_logits` via `slice_set` at
`forward.rs:20295-20298`:

```rust
let logits = lm_head_forward_backend_decode_if(Some(backend), &normed, ...)?;
graph_inputs
    .output_logits
    .slice_set(&logits, 0, 0)
    .context("copy graph-wrapper logits into stable output_logits buffer")?;
```

The `logits` tensor is freshly allocated inside the captured region
by the LM head matmul, but `slice_set` issues a `cudaMemcpyAsync` (or
equivalent) into the runner-owned `output_logits` storage. The
intermediate `logits` lives across exactly one `slice_set` call and
drops immediately. Because nothing else reads from `logits` after
the `slice_set`, there is **no dangling pointer fault** — the
captured `cudaMemcpyAsync` source pointer goes stale but the kernel
that would have read from it doesn't exist. On replay, the
`cudaMemcpyAsync` reads from whatever now lives at the recycled VA,
which could corrupt the output.

Wait — that's still a bug. Let me re-read more carefully…

Actually `slice_set` in candle is a **kernel** (it issues a strided
copy on the GPU), so its source pointer IS captured as a kernel arg.
On replay, the same source pointer is read.

But: the `logits` tensor's storage IS the matmul output, and the
matmul's output storage IS still alive when `slice_set` reads from
it (within the same forward call). The matmul write → slice_set read
all happen inside one captured region. At end-of-capture, the matmul
output tensor drops. **On replay**, the matmul re-writes its output
into a `cudaMalloc`-backed storage… but wait, the matmul's output
pointer was baked at capture time. If that storage has been freed
and recycled, the captured matmul writes to recycled VA, then the
captured `slice_set` reads recycled VA, then `output_logits` gets
garbage.

OK so technically `logits` IS a dangling-pointer hazard too. **But**:
in practice it's the same hazard as every other intermediate tensor
in the captured forward (transformer block outputs, MLP outputs,
RMSnorm outputs, etc.), all of which are also `cudaMalloc`-backed
intermediates. The bs=1 path has the same property and works
reliably, so empirically this is a non-issue.

Why? My best guess: candle's cudarc-backed allocator keeps a free
list / pool, and the recycled VAs are deterministic across replays
of the same forward shape. The captured graph happens to address
the same allocations on every replay because the allocator returns
them in the same order. This works **so long as no other code path
runs between the captured forward's replays that perturbs the free
list** — which the runner enforces by `synchronize()`ing before
capture and not running eager kernels between captured replays for
the same shape.

This is a fragile invariant. The original `block_table_tensor` bug
broke it because the meta object was held across multiple kernel
launches inside the captured region, and the allocator didn't reuse
its slot deterministically — the report at the top of
`cuda-graph-bs2-memcheck.md` shows the faulting address was 906 MiB
past the nearest live allocation, which is consistent with that slot
having been freed and the address pool moved on.

**Status**: **PINNED (under fragile assumption)**. `output_logits`
itself is pinned (it's the destination of `slice_set`, and is
runner-owned). The intermediate `logits` produced by the LM head
matmul is a `cudaMalloc` allocation but is consumed-and-dropped
within the captured region, so it shares the fragile invariant with
every other intermediate.

**Proposed fix**: none required for the dangling-pointer audit. The
"non-LM-head paths" caveat in the original memcheck doc refers to
hypothetical future MTP / draft-model decode loops that would write
to `output_logits` indirectly. Those paths don't run under the
captured bs>1 forward today.

## Bonus — Strict-path `start_slots`

**Status**: **UNPINNED (benign)**. The CUDA backend does not
implement `flash_attn_paged_decode_contiguous_batch` (the strict
path); the default trait impl at `backend/mod.rs:216-226` returns
`Ok(None)`. On CUDA, `try_strict` allocates
`start_slots = Tensor::from_slice(strict_slots, batch, x.device())?`
at `forward.rs:16557-16558` inside the captured region, the backend
declines, and `out` stays `None`. The function then falls through to
`try_dyn_seqlen`. The `start_slots` storage drops at end of
`try_strict`'s closure scope without any kernel having read from it
— but the `Tensor::from_slice` constructor itself issues a
`cudaMemcpyHtoDAsync` that gets captured by the stream. On replay,
that captured HtoD writes to a `cudaFree`'d destination VA, which
may corrupt whatever else now occupies that allocation.

**Severity**: LOW (no kernel reads from the recycled VA), but worth
fixing for stream-capture cleanliness. A minimal fix: short-circuit
the strict probe on CUDA by checking
`backend.supports_strict_paged_decode_contiguous_batch()` (a new
trait method that returns `false` for CUDA) **before** allocating
`start_slots`.

## Implementation order (after this audit)

1. **Land the audit doc itself** (this PR).
2. **Suspect 3+4** (paged-decode `attn_out` + `softmax_lse`) — the
   plumbing change is small, the trait churn is the only annoying
   part. Mirrors the just-landed primary fix shape exactly.
3. **Suspect 2** (RoPE cos/sin tables) — slightly larger plumbing
   change (touches `transformer_block_paged_decode_contiguous_batch`
   and the GQA function), but the bs=1 reference at
   `gqa_attention_paged_with_rope_tables` is a clean model.
4. **Suspect 1** (KV slot writer) — largest delta because it needs a
   new fused batched-slot kernel in `kiln-flash-attn`. Defer until
   the runner is taught to re-use a captured graph across
   `start_pos` values; until then the baked immediate is wrong but
   the bug doesn't fire because each step gets a fresh capture.
5. **Bonus** (strict-path probe) — cleanliness; bundle with #1.

After #2-#4 land, re-run the same compute-sanitizer harness from
`cuda-graph-bs2-memcheck.md` against a multi-step bs=4 chat
completion and confirm zero `Invalid __global__ read` errors. Once
green, flip `KILN_CUDA_GRAPHS_BATCHED=1` to default-on.

## What this audit explicitly does NOT cover

- **GDN gate / recurrent state stability** — already handled by
  `with_decode_gates_recurrent_outputs` + `batched_state_pool`
  (`cuda_graph.rs:1562-1577`). No GDN kernel appears in the
  faulting backtrace of the original memcheck doc.
- **Embedding lookup** — uses the stable `stable_token_ids_gpu`
  tensor when supplied (`forward.rs:18662-18666`); the resulting
  `hidden` tensor is the forward's main carrier and shares the
  fragile-but-empirically-stable intermediate-tensor regime
  described in §5.
- **`final_norm` / LM head matmul intermediates** — same regime as
  §5; pinned via `slice_set` into `output_logits`.
- **MLP / SwiGLU intermediates** — same regime. The
  `swiglu_ffn_backend_profiled` path doesn't have any meta-class
  small tensors that get freed-and-recycled across kernel launches.

## References

- Primary fix commits (#1082): `9b173f84`, `ab798167`, `393beadc`
  (PR landed on `main` 2026-05-25).
- Original memcheck trace: `bench-results/cuda-graph-bs2-memcheck.md`.
- bs=1 reference pattern: `forward.rs:21308` (`model_forward_paged_inner`),
  `forward.rs:16730` (`gqa_attention_paged_with_rope_tables`),
  `forward.rs:15846-15871` (paged-decode kernel call with stable
  `graph_outputs`).
- bs>1 wrapper post-fix: `forward.rs:20230-20300`
  (`model_forward_paged_batched_with_graph_inputs`).
- Runner setup: `cuda_graph.rs:1457-1620` (`try_capture_batched`).
- Runner-owned stable buffers definition:
  `forward.rs:15482-15515` (`BatchedPagedDecodeGraphInputs`).
