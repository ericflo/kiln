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
4. Kernel-disable discriminator (2026-06-01, on pod): set `KILN_DISABLE_{RMSNORM,FUSED_CONV1D,GDN,FUSED_GDN_GATES,FUSED_PAGED_DECODE}` to run every op eager under capture, to test "is it a specific fused kernel?" — INCONCLUSIVE: the eager paths call `alloc_uninit_ctx` *during* capture, which the Frozen arena rejects (`gated deltanet layer 0 ... alloc_uninit_ctx: active_cuda_stream` error). So only the FULLY-FUSED path is capture-compatible and the doubling lives there; kernel-disable can't isolate it.
5. KV-slot baked as an immediate (the in-code "suspect 1") — RULED OUT for the W4A16/BF16 path: `paged_kv_write_token_major_bf16_slot_kt` passes the slot as a LIVE device pointer (`sl_ptr = cuda_input_device_ptr(slot,...)`, kt_api.rs:662), read fresh each launch — not baked. (NOTE: the FP8 path at `paged_kv_cache_kt.rs` DOES `slot.to_scalar::<u32>()` host-side → that one IS baked and would break under capture with `KILN_KV_CACHE_FP8=1` — a separate latent bug, untested.)

**Precise symptom (pod, W4A16, graphs+stable_metadata):** graph output is the SAME token sequence as eager but each token repeated a decreasing count: `Here's a a a a a a a thinking thinking process process that that leads leads to to the the the story story story story` vs eager `Here's a thinking process that could lead to the story`. Reads like the effective context/state advances slower than the token count — the model re-emits a token until the state catches up.

## BUG 2 ROOT CAUSE — CONFIRMED (2026-06-01, on pod, `KILN_DEBUG_GDN_STATE` probe)

**The GDN recurrent + conv state does not advance across graph replays.** A `KILN_DEBUG_GDN_STATE`-gated dump of `linear_state.recurrent_states[0]`/`conv_states[0]` sum-of-squares after each decode step (added to `eager_forward` + the replay-success path in cuda_graph.rs, branch `wip/1082-box2-gdn-probe`) showed, on the W4A16 graphs+stable_metadata path:
```
GDNSTATE [replay] step=15 rs0_sumsq=773.337871 conv0_sumsq=54433.580599
GDNSTATE [replay] step=16 rs0_sumsq=773.337871 conv0_sumsq=54433.580599   <- identical
GDNSTATE [replay] step=17 rs0_sumsq=773.337871 conv0_sumsq=54433.580599   <- identical
GDNSTATE [replay] step=18 rs0_sumsq=773.337871 conv0_sumsq=54433.580599   <- identical
GDNSTATE [replay] step=19 rs0_sumsq=886.110784 conv0_sumsq=72051.618362   <- JUMP (re-capture's eager warm pass)
GDNSTATE [replay] step=20 rs0_sumsq=886.110784 conv0_sumsq=72051.618362   <- frozen again
GDNSTATE [replay] step=21 rs0_sumsq=886.110784 conv0_sumsq=72051.618362
```
A recurrent state MUST change every decode step (`S_t = f(S_{t-1}, x_t)`). Here it is FROZEN between re-capture boundaries and only jumps when `try_capture`'s eager warm pass advances it once. So the captured graph's GDN state update **does not persist into the `linear_state` buffer the next replay reads** — the new state lands in a capture-time / arena buffer and the Rust-side linkage that would point `linear_state` at it does not run on replay. The model decodes against a stuck state → re-emits each token until the next re-capture nudges it → the observed token-doubling.

**THE FIX (next session):** the GDN decode recurrent + conv state update must be TRULY IN-PLACE into the persistent `linear_state.recurrent_states[i]` / `conv_states[i]` buffers the next replay reads — captured kernel reads+writes the SAME persistent device pointer, NO functional new-tensor + Rust-swap. Audit the GDN decode in `forward.rs` (`model_forward_paged*`): find where `recurrent_states[i]`/`conv_states[i]` are produced and ensure the write targets the persistent (resident) buffer in-place. (The KV cache already survives replay because its writer takes the pool + a live slot ptr and writes in-place — the GDN state needs the same treatment; cf. `LinearAttentionState::materialize_gdn_recurrent_resident_states` / the Vulkan resident pattern.) Re-run the probe after the fix: the state must change every step.

(Earlier note, now superseded: "replay runs captured kernels only, so a Rust per-layer dump can't observe replay" — true, but the persistent `linear_state` IS Rust-readable BETWEEN replay steps, which is exactly how this probe localized the bug without kernel-level instrumentation.)

## Status of #1082 boxes 98–101

- 98 (freeze-pointers allocator mode): arena exists + retains forward activations.
  Partial — does not yet cover paged metadata; gated path.
- 99 (capture-lifetime / dangling-pointer rule): BUG 1 is a concrete violation found
  and root-caused. Mechanism exists but incomplete.
- 100/101 (bs>1 batched): separate path, still gated off.

None are merge-complete. The bs=1 keystone needs BUG 1 fix + BUG 2 fix together.

## BUG 2 FIX — partial, validated (2026-06-01, branch `wip/1082-box2-fix-validate`)

Applied an in-place-restore wrapper around `gated_deltanet_forward_decode_if` (forward.rs): snapshot the persistent recurrent+conv state buffers, run the (functional) decode, then `slice_set` the new state back into the persistent buffers (captured copy, survives replay) and restore the slots. Re-ran the `KILN_DEBUG_GDN_STATE` probe under graphs+replay:

- **RECURRENT state now ADVANCES every step** (rs0_sumsq 1483→1421→1508→1602→1503→1487… all distinct) — the fix works for `recurrent_states`. ✓
- **CONV state still partially frozen** (conv0_sumsq has consecutive repeats: 98572 at steps 49/50 & 56/57; 79703 at 58/59) — `conv_states` is not fully advancing under replay. ✗
- **Output still doubled** ("Here's a a a a thinking thinking thinking thinking…") — so fixing the recurrent state alone does NOT fix the doubling; the conv state (and possibly other per-decode buffers) freezing under replay is still in play.

**Conclusion:** the "functional state update doesn't survive replay" bug is a CLASS, not a single site. The recurrent-state half is fixed + validated; the conv-state half is not (the id-guard wrapper either skips it because the conv update is already "in-place" to a NON-persistent buffer, or the conv update path differs from the recurrent one). **Next:** trace the conv-state (`causal_conv1d_update` / `conv_states[i]`) decode update — confirm whether it writes the persistent `conv_states[i]` buffer in-place under capture, or a transient one; apply the same in-place-into-persistent treatment. Then re-run the probe (conv0_sumsq must change every step) AND confirm coherent output before merging. The fix is NOT mergeable until the output is coherent.

## BUG 2 — conv state RULED OUT; doubling is in per-layer captured compute (2026-06-01)

`causal_conv1d_update` (backend/cuda.rs:1640) calls `kiln_conv1d_kernel::causal_conv1d_update_kt(x, weight, conv_state, ...)` which **mutates `conv_state` IN-PLACE** (per the impl comment "the kernel's in-place mutation of conv_state") — no functional reassignment, so the captured kernel writes the persistent `conv_states[i]` buffer directly and survives replay. The probe's occasional consecutive conv0_sumsq repeats were coincidental sum-of-squares matches, not a freeze. **Conv state is NOT the doubling cause.**

State of the doubling hunt: recurrent state FIXED (in-place wrapper, advances every step); conv state in-place (fine); full-attn KV uses the live-slot writer (in-place). All per-step STATE now advances correctly under replay — **yet the output still doubles.** So the doubling is in the per-layer CAPTURED COMPUTE itself, not a frozen state buffer: a captured kernel produces wrong values on replay even with correct inputs+state. Suspects: the GDN q/k/v in_proj path, the fused gates/gated-norm, or a full-attn flash output — some captured intermediate scratch that isn't reset/frozen correctly between replays, OR a captured op that reads a stale per-step input the metadata probe didn't cover.

**NEXT (needs fresh focus — substantial forward.rs instrumentation):** add a `KILN_DEBUG_LAYER_NORMS`-gated per-layer hidden-state norm dump baked INTO the captured graph (a persistent `[num_layers]` f32 buffer + a captured sqr→sum→slice_set after each transformer block), then diff eager-vs-replay per-layer norms to find the FIRST layer that diverges → localizes the offending captured op. The GDN-state probe technique (read a persistent buffer between replay steps) does NOT work for per-layer hidden states (not persistent), so the dump must be a captured op writing a persistent buffer. Do this in a focused session, not at marathon-tail fatigue.

## BUG 2 — per-layer dump is FEASIBLE; exact plumbing specified (2026-06-01)

Confirmed the localization instrumentation is implementable NOW (no new kernels): kt has device-side `Tensor::sqr()` (method_api.rs:416) + `sum_all()` (:633) + `slice_set(src,0,offset)` (:830) + `to_vec` — so a captured per-layer norm dump works: `hidden.sqr()?.sum_all()?` → reshape `[1]` → `buf.slice_set(&ss, 0, i)` writes layer-i norm into a persistent `[num_layers]` f32 buffer, all on-device/captured (NO `to_vec` during capture — that D2H breaks capture). Read `buf.to_vec()` AFTER each replay step (Rust runs between steps).

EXACT PLUMBING (fresh-focus session):
1. Allocate the `[num_layers]` f32 buffer with `Tensor::zeros_on(device,...)` in `try_capture` (cuda_graph.rs, next to `lm_head_output_buffer` ~:1478) so it is PERSISTENT (outside the capture window — do NOT lazy-alloc inside the captured forward or the arena may free it). Thread it via a `with_layer_norm_debug_buffer(buf, || {...})` thread-local scope mirroring `with_lm_head_output_buffer`.
2. In `model_forward_paged_inner` block loop `for (i, layer) in weights.layers.iter().enumerate()` (forward.rs:24989), at the END of each iteration call `record_layer_norm_debug(&hidden, i)` — gated by `KILN_DEBUG_LAYER_NORMS`; it reads the thread-local buf + does the captured sqr→sum_all→slice_set into `buf[i]`.
3. In cuda_graph.rs after the replay-success launch (and in `eager_forward`), read `buf.to_vec::<f32>()` + eprintln the per-layer norms (like the GDN-state probe).
4. Run graphs-ON; the eager (capture-pass) norms vs replay norms diverge at the FIRST broken layer → that layer's captured op is the doubling bug. Fix it (likely an in-place/scratch issue analogous to the GDN recurrent-state fix), re-validate (output coherent), then merge the keystone + check boxes 98-101.

Partial-fix branch (recurrent-state in-place wrapper, validated): `wip/1082-box2-fix-validate`. Do NOT merge until output is coherent.
