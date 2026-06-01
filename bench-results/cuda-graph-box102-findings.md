# box-102 cuda-graph keystone — definitive diagnosis (2026-06-01)

Branch: `wip/1082-p5-freeze-pointers` @ `40fee844`. Hardware: A6000, Qwen3.5-4B.
Issue: #1082 boxes 98–101 (cuda-graph freeze-pointers / capture-lifetime).

## TL;DR

The keystone is **two distinct bugs**, not one. BUG 1 (OOB) is root-caused with
a confirmed fix; BUG 2 (replay correctness) is isolated but open. **Do not merge
until BUG 2 is fixed** — the path produces wrong output.

## How to reproduce (critical: bench does NOT exercise this)

`kiln-bench --paged` does **not** drive the CUDA-graph runner — `CudaGraphRunner::new`
in `forward.rs` is inside `mod tests`. The production runner is `generate.rs:193`
(`cuda_graph: Mutex<CudaGraphRunner>`), reached only via the **server**. Any
bench-based "graph validation" is actually eager and meaningless.

```
KILN_W4A16=1 KILN_CUDA_GRAPHS=true KILN_MODEL_PATH=/workspace/Qwen3.5-4B \
KILN_BATCHING_ENGINE=0 RUST_LOG=info,kiln_model::cuda_graph=debug \
  ./target/release/kiln serve
# curl /v1/chat/completions  {"model":"Qwen3.5-4B","messages":[...],"max_tokens":48}
```
`KILN_BATCHING_ENGINE=0` forces the true bs=1 `decode_step_paged`. (With the batching
engine on, single requests route through `decode_step_paged_batched` = boxes 100/101.)

## BUG 1 — OOB on captured-graph launch (ROOT-CAUSED, FIX CONFIRMED)

compute-sanitizer (server under `compute-sanitizer --tool memcheck`):
```
Invalid __global__ read of size 16 bytes
  at flash_fwd_splitkv_kernel<256,64,64,4,...>(Flash_fwd_params)+0x1de0
  Address 0xffffcdb1c19e3c00 is out of bounds   (wild/wrapped pointer)
  CudaGraph::launch → decode_step_paged
```
Root cause: `cuda_graph.rs:1425` builds the graph-stable `block_table_buffer` /
`seqused_k_buffer` / `kv_slot_buffer` only `if key.stable_metadata`, which is
`KILN_CUDA_GRAPH_STABLE_PAGED_METADATA` — **default false** (`:156`). With it off,
`graph_inputs = None` and the captured forward builds a transient block_table that is
freed when capture returns; the captured flash splitkv kernel then dereferences a
dangling pointer → wild offset. The freeze-pointers arena retains forward activations
(Q/K/V proj) but NOT this internally-built metadata.

**Fix confirmed:** `KILN_CUDA_GRAPH_STABLE_PAGED_METADATA=1` → "CUDA graph replay
succeeded" every step, HTTP 200, zero OOB. Proper fix = default the stable-metadata
path on (or remove the gate) — but only together with BUG 2, since on its own it
trades a crash for wrong output.

## BUG 2 — replay produces token-doubling garbage (OPEN)

With stable metadata on, no crash, but output doubles tokens:
- eager (graphs off): `Here's a thinking process that could lead to the story above: 1. Analyze the Request...`
- graph replay:        `Here's a a a a a a a thinking thinking process process that that leads leads to to the the story story...`

Same tokens, each repeated. NOT a stream-ordering race — persists (cleaner 2×) under
`CUDA_LAUNCH_BLOCKING=1`. The generate loop is correct (`seq_len += 1`; `sample_step`
feeds back once). The per-step KV write reads `inputs.kv_slot` (the in-place-refreshed
device buffer), not a host immediate (`forward.rs:19964`). So the defect is in the
captured-graph computation/replay semantics — consistent with the in-code suspect
"per-step KV-slot writer baked-immediate correctness under graph replay across
start_pos."

**Update (instrumented `update_paged_metadata_buffers`):** the per-replay metadata is
**correct** — across replay steps the dump showed `slot` advancing 19→31, `attn_len`
20→32, `block_table=[0,1,2,2,2,2]` consistent. So BUG 2 is **NOT** the metadata refresh
or the KV-slot value; it is in the **captured-kernel computation on replay**. Since
24/32 layers are GDN (Gated DeltaNet, recurrent linear-attn), the prime suspect is the
**GDN recurrent-state handling under capture/replay** — the non-idempotent `linear_state`
the capture code snapshots/restores around the Pass1/Pass2 double-forward
(`cuda_graph.rs:~1494,1506`). Either (a) a GDN per-layer recurrent/conv state buffer is
not frozen by the capture arena (so the captured graph reads/writes a stale buffer and
the state stops evolving across replays), or (b) the snapshot/restore leaves the first
replay starting from an off-by-one state that compounds. Next step: instrument GDN
recurrent-state norms per replay step (does the state evolve, or is it frozen/stale?),
and verify every GDN state buffer enters the capture arena's retained set.

**Hypotheses tested + RULED OUT for BUG 2 (2026-06-01):**
1. Metadata refresh wrong — RULED OUT (instrumented dump shows slot/attn_len/block_table all correct per replay).
2. GDN recurrent-state double-advance during capture — RULED OUT on reading: CUDA stream capture only RECORDS (kernels don't execute), so Pass2 does not advance state; the single first-launch advances it once (`cuda_graph.rs:1648-1655` comment is correct). Snapshot/restore at `:1543-1545` is sound.
3. `AUTO_FREE_ON_LAUNCH` freeing an arena-missed `cudaMemAllocNode` → dangling on replay — RULED OUT: switching the bs=1 `end_capture` to the batched path's `no_flags` (0) produced identical doubling.

**Remaining approach for BUG 2:** differential per-layer instrumentation — capture the per-layer hidden-state norm (or logits) at decode step N via BOTH eager and graph-replay for the same inputs, and find the first layer where they diverge. That localizes which captured kernel/buffer produces wrong values on replay (candidate: a GDN conv/recurrent intermediate, an RMSNorm/residual scratch, or an attention-softmax accumulator that isn't reset/frozen between replays). This needs a small KILN_-gated dump hook in the forward, not black-box probing.

## Status of #1082 boxes 98–101

- 98 (freeze-pointers allocator mode): arena exists + retains forward activations.
  Partial — does not yet cover paged metadata; gated path.
- 99 (capture-lifetime / dangling-pointer rule): BUG 1 is a concrete violation found
  and root-caused. Mechanism exists but incomplete.
- 100/101 (bs>1 batched): separate path, still gated off.

None are merge-complete. The bs=1 keystone needs BUG 1 fix + BUG 2 fix together.
