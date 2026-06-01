# CUDA-graph decode: status & the real blocker (2026-06-01)

> TL;DR — On `main`, decode CUDA-graph **capture is a silent no-op**: capture is
> attempted on the kt NULL/default stream, the driver returns
> `CU_STREAM_CAPTURE_UNSUPPORTED`, and the path falls back to eager. Decode is
> therefore **correct but gets zero graph speedup today.** Making capture
> actually engage (capture on a non-default stream) exposes the *next* layer of
> the problem: the captured graph references per-layer forward **intermediates
> that are freed when the capture closure's kt tensors drop**, so replay reads
> unmapped memory → `CUDA_ERROR_ILLEGAL_ADDRESS`. The headline decode-graph win
> is blocked on **graph-safe (persistent / stream-ordered) allocation of decode
> intermediates**, not on stream plumbing.

## What was tried (branch `wip/1082-p5-rebased`, 3 commits on top of current main)

1. **Capture on a non-default stream, not the NULL stream**
   (`ctx.new_stream()` instead of `ctx.default_stream()`). This is *necessary*:
   `begin_capture` on the NULL stream returns `CU_STREAM_CAPTURE_UNSUPPORTED`, so
   on `main` capture never engages and decode silently runs eager. With a fresh
   stream, `begin_capture` succeeds and `CUDA graph captured for decode (32
   layers)` is logged.
2. **Sync the kt default stream before replay** — defensive; the graph-stable
   I/O buffers are filled by an H2D on the kt default stream, so it must be
   synced before the captured forward / before replay.
3. **Execute the captured graph on first capture** — stream capture *records*
   ops without executing them, so the pre-fix code returned the *uninitialized*
   `output_logits` from the capture call (the "prefix-correct-then-garbage"
   symptom) and never advanced recurrent/KV state. The fix launches the
   instantiated graph + synchronizes right after capture so step N actually
   executes.

## Observed result (W4A16 serve, A6000, greedy + eager oracle)

```
CUDA graph captured for decode (32 layers)
CUDA graph capture failed: sync after first captured-graph launch:
  DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was
  encountered"), using eager decode
batched real generation failed: ... eager decode forward pass failed:
  host_to_cuda_copy: clone_htod failed:
  DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, ...)  -> HTTP 500
```

The launch surfaces a genuine fault: **replaying the captured graph touches
illegal memory.** Because an `ILLEGAL_ADDRESS` poisons the whole CUDA context,
the subsequent eager fallback also fails (every later CUDA call errors) and the
request 500s. The eager-only path (`KILN_CUDA_GRAPHS=false`) is unaffected and
produces correct output — confirming the fault is exclusively in graph
capture/replay, not the forward math.

## Root cause (per the code's own comment, cuda_graph.rs ~line 1436)

> "Defense-in-depth for the bs=1 path (which has historically worked by
> allocator-determinism luck at small `[1, 1, vocab]` shapes)."

The capture window pins only the **I/O** buffers as struct fields on
`CapturedDecodeGraph` (`output_logits`, `token_buffer`, `position_buffer`,
`paged_decode_outputs`, `gdn_decode_outputs`, `lm_head_output_buffer`, rotary
buffers, block-table buffers). The **hundreds of per-layer intermediates** the
forward allocates (matmul outputs, norm scratch, GDN/attention temporaries) are
*not* pinned — they drop when the `with_active_cuda_stream(... forward ...)`
closure returns and are returned to the kt pool. The captured kernels still
encode those pool addresses; once the pool reuses/frees them, replay dereferences
unmapped memory.

This "worked" historically only because at bs=1, `[1,1,*]` shapes, the pool
happened to hand back identical addresses on the next step (determinism luck).
That luck no longer holds (the kt-substrate allocator behaves differently than
the pre-migration candle path), so the latent bug is now load-bearing.

## The fix that's actually needed (separable perf project, not candle-removal)

Decode-graph capture needs **graph-safe allocation** for *all* intermediates the
captured region touches. Two standard approaches:

- **Stream-ordered allocation** (`cudaMallocAsync` + a per-graph mempool) so the
  intermediates become graph-owned alloc/free nodes and replay re-allocates them
  correctly. Pairs with the existing
  `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH` instantiate flag.
- **A capture arena** — a bump allocator whose lifetime is tied to the
  `CapturedDecodeGraph`; allocations made during the capture window are retained
  (not returned to the general pool) until the graph is dropped, so every encoded
  address stays mapped.

Either is a real allocator change touching the kt `cuda_allocator` + the decode
forward path. It is **not** a prerequisite for the candle→kiln-tensor migration
(CUDA decode is already candle-free and correct via the eager path); it is the
decode **throughput** win and should be scoped as its own effort. Next concrete
diagnostic step when it is picked up: run the W4A16 graphs-on serve under
`compute-sanitizer --tool memcheck` to name the exact first kernel + buffer that
faults, then pin/arena that allocation and iterate.

## Decision

- The 3 commits live on `wip/1082-p5-rebased` (pushed) as documented WIP. They are
  **not merged to main**: merging would convert today's silent-correct-eager
  decode into a hard 500 whenever capture engages.
- `main` keeps the safe behavior (capture fails on NULL stream → eager).
- This file supersedes the earlier "ILLEGAL_ADDRESS is obsolete/stale" note — the
  `ILLEGAL_ADDRESS` is real and is the load-bearing blocker.
