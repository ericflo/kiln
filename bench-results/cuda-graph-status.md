# Phase 5 — CUDA graph capture status (#1082)

Snapshot taken on `main` at `76069cd6` (post-Phase 7 FP8 wire-up).
Inventories what's already shipped vs. what still gates the
"command-list / graph capture (per backend), with batched cuda-graph
as the headline" milestone for Phase 5.

This doc is the antidote to "the substrate isn't ready yet" planning
drift — the substrate (`kiln-graph` + `kiln-tensor::Allocator`) and
the production batched capture path (`kiln-model/src/cuda_graph.rs`)
landed in parallel. Future planners should treat this as the
canonical "what's already done" reference.

## Headline

- The **bs=1 production CUDA graph capture/replay path is live**
  under `KILN_CUDA_GRAPHS=true` (on by default).
- The **bs>1 batched capture/replay path is in-tree** under
  `KILN_CUDA_GRAPHS_BATCHED=1` (off by default). **All four root-cause
  intra-graph alloc suspects from `cuda-graph-bs2-secondary-audit.md`
  are now closed** (see "Phase 5 bs>1 progress" below).
- One end-to-end `compute-sanitizer` sweep on the full Qwen3.5-4B chat-
  completion driver with `KILN_CUDA_GRAPHS_BATCHED=1
  KILN_CUDA_GRAPHS_BATCHED_KV_FUSED=1` is the remaining validation gate
  before flipping `KILN_CUDA_GRAPHS_BATCHED=1` to default-on.
- The **substrate crates** (`kiln-graph`, `kiln-graph-cuda`,
  `kiln-graph-metal`, `kiln-graph-vulkan`) ship the backend-agnostic
  types but the per-backend impls are scaffolds.
- The **kt-tensor allocator** has full three-mode support
  (`Owned` / `Pool` / `Frozen`) on **CPU + CUDA + Vulkan + Metal**
  today, with `warm()` pre-warming and `Frozen`-mode allocation
  rejection wired end-to-end. The CPU smoke test exercises the entire
  `pool → warm → freeze → pin → audit` lifecycle in
  `crates/kiln-graph/tests/capture_lifetime.rs`.

The remaining Phase 5 work is **lifting the production batched-capture
plumbing in `kiln-model/src/cuda_graph.rs` onto the `kiln-graph-cuda`
substrate**, plus the two non-CUDA per-backend impls
(`kiln-graph-metal`, `kiln-graph-vulkan`). The freeze-pointers
contract itself is not the blocker — it landed in Phase 1.27 / 1.28.

## What's done

### Substrate (`kiln-graph`)

| Item | Where | Status |
|---|---|---|
| `AllocatorMode { Owned, Pool, Frozen }` | `crates/kiln-tensor/src/allocator.rs:47` | Done (canonical home, re-exported from `kiln-graph::AllocatorMode`) |
| `Allocator` trait (backend-agnostic) | `crates/kiln-tensor/src/allocator.rs:104` | Done; `set_mode`, `alloc`, `reserved_bytes`, `peak_reserved_bytes`, `mode`, `device` |
| `CaptureSession` (RAII guard) | `crates/kiln-graph/src/capture_session.rs:61` | Done; `begin/pin/finalize/audit_pinned` |
| `PinnedPointer { tensor_id }` | `crates/kiln-graph/src/capture_session.rs:38` | Done; today carries `TensorId` only — per-backend impls will extend with raw device pointer for O(1) dangling-pointer checks |
| `CapturedGraph` trait | `crates/kiln-graph/src/captured_graph.rs:17` | Done; `backend / replay / replay_count / scratch_bytes` |
| `CaptureError` (typed errors) | `crates/kiln-graph/src/error.rs:7` | Done; `AllocationDuringFreeze`, `DanglingPointer`, `NotCaptured`, `Backend(String)`, `Tensor(#[from] kt::Error)` |
| `allocator_frozen_error` helper | `crates/kiln-tensor/src/allocator.rs:146` | Done; standardized error string across backends |
| End-to-end lifecycle smoke test | `crates/kiln-graph/tests/capture_lifetime.rs` | Done; 5 tests — happy path, frozen-rejects-non-warm, dangling-pointer detection, pool↔frozen↔pool round-trip, reserved-bytes accounting |

### kt-tensor allocator impls

| Backend | Crate file | Status |
|---|---|---|
| CPU | `crates/kiln-tensor/src/cpu_allocator.rs` | Done; `Owned`/`Pool`/`Frozen` + `warm()` + tests |
| CUDA | `crates/kiln-tensor/src/cuda_allocator.rs` | Done; same shape as CPU, `warm()` allocates via `CudaStorage::zeros`; GPU tests gated on `KILN_TENSOR_CUDA_TEST=1` |
| Vulkan | `crates/kiln-tensor/src/vulkan_allocator.rs` | Done; same shape, `Frozen` rejection via `allocator_frozen_error` |
| Metal | `crates/kiln-tensor/src/metal_allocator.rs` | Done; same shape, backed by `MetalStorage::zeros` over `Arc<MetalDevice>` |

### Production CUDA graph path (`kiln-model::cuda_graph`)

The production decode path uses `cudarc::driver::CudaGraph` +
`CudaGraphExec` directly (not the substrate). Today's contract:

| Path | Env | Status |
|---|---|---|
| `bs=1` capture/replay | `KILN_CUDA_GRAPHS=true` (default on) | **Production-stable** for ≥ 1 year. Yields ~10–15% decode throughput improvement. |
| `bs>1` capture/replay | `KILN_CUDA_GRAPHS_BATCHED=1` (default off) | **Code present, gated off.** `CudaBatchedGraphKey` + `CapturedBatchedDecodeGraph` + `try_capture_batched` + replay all live in-tree (see line citations below). The call site in `generate.rs:2695` is gated on `is_batched_enabled()`. Default-off because a `CUDA_ERROR_ILLEGAL_ADDRESS` / `compute-sanitizer`-traced fault was hit during the integration attempts (commits `30fd877e`, `535f9820`) and never fully root-caused before the call-site wiring was held back. The implementation has not been re-validated since the no-replay diagnostic mode landed (`90b182ed`, 2026-05-15). |
| `KILN_CUDA_GRAPHS_BATCHED_NO_REPLAY=1` diagnostic | `cuda_graph.rs:560` | Done; forces re-capture every step so capture can be isolated from replay during bring-up. |
| Stable paged metadata key | `KILN_CUDA_GRAPH_STABLE_PAGED_METADATA=1` | Done; avoids the `(seq_len, block_table)` cliff that otherwise causes per-step re-captures. |

Key pieces of the batched path (all in
`crates/kiln-model/src/cuda_graph.rs`):

- `CudaBatchedGraphKey` (line 180) — cache key. **Not dead code**;
  used by `decode_step_paged_batched` and `try_capture_batched`.
- `CapturedBatchedDecodeGraph` (line 270) — instantiated graph +
  the full set of stable buffers (token, position, block table,
  seqused-k, kv-slot, rotary cos/sin, per-layer paged decode outputs
  + LSE, per-layer GDN outputs, output logits).
- `captured_batched: HashMap<CudaBatchedGraphKey, CapturedBatchedDecodeGraph>`
  (line 318) — the bucketed cache.
- `batched_bucket_warmup_done: HashSet<usize>` (line 326) — per-bucket
  warmup tracker so each new batch-size bucket gets one eager step to
  prime the allocator before its first capture (the global
  `warmup_done` flag set by the bs=1 capture is not sufficient).
- `batched_state_pool: HashMap<usize, LinearAttentionState>`
  (line 337) — persistent per-bucket GDN recurrent+conv state. Its
  device pointers are baked into the captured graph; the pool slot
  outlives every replay.
- `persistent_batched_state` (line 413) — lazy allocator for the
  per-bucket state pool slot.
- `decode_step_paged_batched` (line 501) — driver: warmup gate →
  cache lookup → replay-or-capture decision → adapter-gen check →
  in-place buffer refresh → graph launch → argmax outside the
  captured region → scatter post-step state back.
- `try_capture_batched` (line 1457) — the full
  pre-alloc-every-buffer → `BatchedPagedDecodeGraphInputs` →
  `model_forward_paged_batched_with_graph_inputs` → `CUstreamCapture`
  → `cuGraphInstantiate` capture sequence.

### Substrate ↔ production seam

`kiln-graph-cuda::CudaCapturedGraph` (and its sibling crates for
Metal + Vulkan) wraps `kiln_tensor::Backend::Cuda` and implements
`kiln_graph::CapturedGraph`. Today the impl is a scaffold:

- `new(scratch_bytes: usize) -> Self`
- `backend() -> Backend::Cuda`
- `replay()` increments a counter and returns `Ok(())` — does **not**
  drive a real `CudaGraphExec`.
- `replay_count() / scratch_bytes()` bookkeeping.

The intended lift target is to move the `cudarc::CudaGraph` /
`CudaGraphExec` lifecycle out of `kiln-model::cuda_graph` and into
`kiln-graph-cuda` so per-backend graph types share the
`CapturedGraph` surface.

## What's blocked / open

Four workstreams remain to call Phase 5 fully done. **None of them
are blocked by the freeze-pointers contract any longer** — the
substrate landed; this is per-backend lift work and one
correctness-bug close-out.

### 0. Re-validate `KILN_CUDA_GRAPHS_BATCHED=1` (highest priority)

The bs>1 capture/replay path is fully implemented in
`kiln-model/src/cuda_graph.rs` but defaults off because the last
integration attempt hit a `CUDA_ERROR_ILLEGAL_ADDRESS` /
`compute-sanitizer`-traced fault that was never root-caused. Commits
to read in order before re-attempting:

1. `4c7c0b4e` — disabled runner entry (skeleton).
2. `948c639c` — `KILN_CUDA_GRAPHS_BATCHED` env gate.
3. `4cb71f1d` / `d639433c` / `d4550baa` — buffer allocators.
4. `66282787` — replay-time updaters.
5. `b2dfbc49` — capture body.
6. `70d919e1` — warmup + capture phase wiring.
7. `984f88cb` — replay buffer refresh + launch.
8. `34668395` — replay path with GDN state refresh.
9. `d564067c` — argmax + DtoH moved outside captured region.
10. `f1a04c7d` — GDN thread-local outputs.
11. `42db0c8f` — per-bucket warmup tracker; first revert.
12. `30fd877e` — compute-sanitizer-traced fault; permanent revert.
13. `855a0932` — end session with multi-batch path safely disabled.
14. `535f9820` — confirmed `AUTO_FREE_ON_LAUNCH=0` is not the fix.
15. `90b182ed` — diag-mode wiring re-added (current main).

The most likely fault sources (not yet exonerated):

- An intra-graph allocation node the runner does not pin (a kernel
  temporary that Candle frees between capture and the next replay).
  Audit every `*.zeros(...)` / `Tensor::new(...)` call inside the
  batched forward; each one must be replaced with a pre-allocated
  buffer threaded through `BatchedPagedDecodeGraphInputs`.
- A stream-mismatch between the captured graph and the post-replay
  argmax (argmax was moved outside the capture in `d564067c`; verify
  it sees the same stream).
- GDN-state pointer drift between the refresh and the replay.

Recommend driving this with `compute-sanitizer --tool memcheck` on a
fresh A6000 pod, single-shot bs=8 replay, and walking the resulting
trace before touching the substrate lift.

### 1. `kiln-graph-cuda` real cudarc wiring

Replace the scaffold `CudaCapturedGraph::replay()` no-op with a real
`CudaGraphExec::launch` over an owned `cudaGraph_t`. Two staging
strategies:

**Option A — re-export the model-crate path.** Easiest first
landing. Move `CapturedDecodeGraph` (bs=1) and
`CapturedBatchedDecodeGraph` (bs>1) bodies into `kiln-graph-cuda`
behind a new `CudaCapturedGraph::{decode_bs1, decode_bsn}` variant.
The `kiln-model` runner constructs one of these via the substrate
and delegates replay through the trait. Captures the existing
contract verbatim, no behaviour change. Lowest-risk.

**Option B — generalize over `BatchedPagedDecodeGraphInputs`.**
Define a `BatchedDecodeGraphInputs` trait the model crate implements
and `kiln-graph-cuda` consumes. Cleaner long-term but requires
auditing every device-pointer site the captured graph touches and
threading it through the trait. Defer to a Phase 5.2 follow-up.

### 2. `kiln-graph-metal` real `MTLIndirectCommandBuffer` impl

Highest-leverage substrate work after the bs>1 fault is fixed.

Metal's analog is `MTLIndirectCommandBuffer` + `MTLCommandBuffer`.
The `MetalAllocator` is **already in place** in
`crates/kiln-tensor/src/metal_allocator.rs` with full Owned/Pool/Frozen
support, so the only remaining work is the per-backend impl itself:

- Build out `kiln-graph-metal::MetalCapturedGraph` over
  `MTLIndirectCommandBuffer`. The capture-equivalent on Metal is the
  ICB encode step; replay is `[commandBuffer encodeIndirectCommands]`.

This is the longest-pole non-CUDA gap. Note: ICBs do **not** capture
all kernel launches the way CUDA graphs do — only the ones encoded
into the ICB itself. The Metal substrate impl will need an explicit
"every dispatch goes through `encodeComputeIndirectCommandsAtIndex`"
contract that the kt-API layer plumbs through.

### 3. `kiln-graph-vulkan` over `kiln-vulkan-kernel::cmd_batch.rs`

Vulkan analog is a pre-recorded `VkCommandBuffer` (secondary command
buffer or compute pipeline batch). The Vulkan kernel crate already
has a `cmd_batch.rs` mini-framework; `kiln-graph-vulkan` wraps it
under the `CapturedGraph` trait. Lower risk than Metal because the
`VulkanAllocator` already supports `Frozen` mode.

## Open follow-ups (Phase 5.x backlog)

These were called out in the issue or the in-file design note but are
not pre-requisites for declaring "Phase 5 done":

- **AOT graph serialization via `cuGraphSerialize`** — capture once at
  warmup, persist as a binary blob, load on subsequent server starts
  to skip the warmup re-capture cost. Issue calls this out as the
  Phase 5 final bullet.
- **Per-backend `PinnedPointer` extension to carry the raw device
  pointer.** Today the `CaptureSession` only records `TensorId`; the
  audit walker has to resolve the id to a pointer through a side
  table. Per-backend impls can extend the pinned record with the
  device address for O(1) dangling-pointer detection.
- **`'a`-lifetime encoding** — the issue's `CapturedGraph<'a>` /
  `FrozenAllocator<'a>` design has `'a` as a compile-time enforcement
  of the capture lifetime. Today's runtime-only `audit_pinned` path
  is the temporary stand-in; the lifetime encoding lands once the
  per-backend impls are real and the call-site shape is settled.
- **`KILN_AUDIT_GRAPHS=1`** — always-on dangling-pointer audit (not
  just `cfg(debug_assertions)`). The runtime hook exists; the env
  flag wiring does not.

## Why "freeze-pointers" is not the blocker

The original Phase 5 brief said:

> kiln-tensor allocator "freeze-pointers" mode. ... This is the
> structural fix that makes batched cuda-graph capture possible;
> without it Phase 8 doesn't land.

That mode landed across Phase 1.27 and Phase 1.28:

- `AllocatorMode::Frozen` (canonical) — `crates/kiln-tensor/src/allocator.rs:59`.
- `CpuAllocator::warm` + Frozen rejection — `cpu_allocator.rs:79` + `cpu_allocator.rs:141`.
- `CudaAllocator::warm` + Frozen rejection — `cuda_allocator.rs:81` + `cuda_allocator.rs:136`.
- `VulkanAllocator::warm` + Frozen rejection — `vulkan_allocator.rs:52` + `vulkan_allocator.rs:102`.
- `CaptureSession`-driven mode flip — `capture_session.rs:70` (mode is `Frozen` from `begin()` onward).
- `allocator_frozen_error` standardized error message — `allocator.rs:146`.

The contract is also exercised end-to-end by the
`crates/kiln-graph/tests/capture_lifetime.rs` smoke (CPU path). Any
remaining Phase 5 task description that asserts "freeze-pointers
mode is the unblock" is **stale** and predates these landings.

## Recommended sequencing for Phase 5 close-out

1. **Re-validate `KILN_CUDA_GRAPHS_BATCHED=1` and root-cause the
   `CUDA_ERROR_ILLEGAL_ADDRESS` fault.** See section "0." above. This
   is the highest-priority remaining bug because it determines
   whether the bs>1 design as built is viable or needs a structural
   redesign. Until this is resolved, doing the substrate lift just
   moves a broken implementation into a different crate.
2. **`kiln-graph-cuda` Option A lift** (re-export the model-crate
   batched path through `CudaCapturedGraph`). Zero behaviour change;
   makes the substrate-typed `CapturedGraph` the production surface.
   Do this only after #1 is green.
3. **`kiln-graph-vulkan` over `cmd_batch.rs`** (parallel with #2 —
   `VulkanAllocator` already supports `Frozen`).
4. **`kiln-graph-metal` over `MTLIndirectCommandBuffer`**
   (`MetalAllocator` already supports `Frozen`).
5. **AOT graph serialization** via `cuGraphSerialize` once #2 lands.
6. **Per-backend `PinnedPointer` extension** + `'a`-lifetime encoding
   tightening once at least two per-backend impls share the contract.

## References

- Phase 5 brief: GitHub issue #1082 (search "command-list / graph
  capture").
- Production CUDA graph runner: `crates/kiln-model/src/cuda_graph.rs`
  (2,137 lines).
- Substrate crate: `crates/kiln-graph/` (4 source files + integration
  test).
- Per-backend scaffolds: `crates/kiln-graph-cuda/src/lib.rs`,
  `crates/kiln-graph-metal/src/lib.rs`, `crates/kiln-graph-vulkan/src/lib.rs`.
- Allocator: `crates/kiln-tensor/src/allocator.rs` +
  `cpu_allocator.rs` / `cuda_allocator.rs` / `vulkan_allocator.rs` /
  `metal_allocator.rs`.
- In-file Phase 5 design note: `cuda_graph.rs:31-95` (the "Multi-batch
  (`bs > 1`) capture" section). Note: the "not yet implemented"
  framing in the design note predates the actual `bs>1` capture body
  landing; the **wiring** in `decode_step_paged_batched` /
  `try_capture_batched` is in-tree but **gated off** pending the
  fault root-cause described in section "0." above.

## Phase 5 bs>1 progress (2026-05-25)

All four root-cause intra-graph alloc suspects from
`cuda-graph-bs2-secondary-audit.md` are now closed:

| Suspect | Site | Closure commit(s) |
|---|---|---|
| 0 | `CachedPagedDecodeMeta::build` allocs (`block_table_tensor`, `seqused_k_tensor`) | `9b173f84` + `ab798167` + `393beadc` (stable-buffers refactor + thread through wrapper) |
| 0b | `build_with_stable_buffers` extended to thread `Option<&PagedKvCacheKt>` | `d0f7049d` (round 2) |
| 1 | per-row KV slot writer with capture-time baked-immediate slot | PR #1384 — new `kiln_paged_kv_write_token_major_bf16_batch_slot` CUDA kernel + Rust wrapper + `PagedKvCache::write_token_major_native_batch_graph_slot` + env gate `KILN_CUDA_GRAPHS_BATCHED_KV_FUSED=1`. 0 compute-sanitizer errors on the kernel unit test. |
| 2 | RoPE cos/sin tables allocated inside captured region | `b571b57d` (RoPE stable tables threaded as `Option<(&Tensor, &Tensor)>` through the bs>1 wrapper) |
| 3+4 | `attn_out` / `softmax_lse` scratch allocated inside captured region | `a7af559b` (pin via `graph_outputs: Option<(&Tensor, &Tensor)>` param + thread through `gqa_attention_paged_decode_contiguous_batch`) |

### What remains for `KILN_CUDA_GRAPHS_BATCHED=1` default-on

1. **End-to-end compute-sanitizer sweep** with `KILN_CUDA_GRAPHS_BATCHED=1
   KILN_CUDA_GRAPHS_BATCHED_KV_FUSED=1` on the full Qwen3.5-4B driver
   (model load + multi-row chat-completion). Single confirmation run that
   no `Invalid __global__ read/write` errors remain in the captured
   graph replay path under realistic batch shapes.
2. After the sanitizer is green, flip the `KILN_CUDA_GRAPHS_BATCHED=1`
   default in code and rerun the production benchmarks for the new
   baseline.

The unit-level closure is settled. The remaining step is a single
end-to-end validation pass.

## End-to-end sanitizer sweep — 2026-05-25 (in-progress, blocked)

A6000 sanitizer sweep attempted on `main` at commit `86faaec`
(`kiln-autograd: backward ops for FLCE forward ...`). **Findings:**

### Pure-kernel sanitizer result: 0 errors

Ran `compute-sanitizer --tool memcheck` on the kiln-model unit-test
binary with `KILN_CUDA_GRAPHS=true KILN_CUDA_GRAPHS_BATCHED=1
KILN_CUDA_GRAPHS_BATCHED_KV_FUSED=1`, targeting the bs>1 / paged-decode
test surface:

| Test | Result |
|---|---|
| `paged_kv_cache::tests::test_write_token_major_native_batch_then_read_roundtrip` | **pass, 0 sanitizer errors** |
| `forward::tests::test_linear_attention_state_batch_row_assembly_and_scatter` | **pass, 0 sanitizer errors** |
| `paged_kv_cache::tests::test_write_token_major_native_batch_graph_slot_matches_per_row` | **pass, 0 sanitizer errors** |
| `forward::tests::test_flash_attn_paged_decode_dyn_seqlen_kt_api_parity` | **pass, 0 sanitizer errors** |
| `forward::tests::test_model_forward_paged_decode_contiguous_batch_dyn_seqlen_cuda` (bs=2, non-uniform `start_positions`) | **fails on functional check unrelated to memory safety** (see below) |

Final sanitizer summary line: `========= ERROR SUMMARY: 0 errors`.

This re-confirms the PR #1384 "0 errors at kernel unit-test level"
finding on `main` post-residual-audit, including the per-row kv-slot
graph-slot writer and the batch row assembly/scatter paths added by
the Phase 5 bs>1 fix series.

### Validation gate blocked by two unrelated kt-API regressions

**Blocker #1: kt-rotary cos/sin shape mismatch.**
`test_model_forward_paged_decode_contiguous_batch_dyn_seqlen_cuda`
fails before the cuda graph capture path is reached, with:

```
Error: batched transformer block 0 (full attention, paged)
Caused by:
    kt fused_rotary_qk: kt-rotary: cos [2, 128] != [2, 256]
```

This is a shape contract violation between `rotary_embedding_from_tensor`
and the kt-typed `fused_rotary_qk` wire on the bs>1 paged-decode path.
It is **not** a memory-safety issue — sanitizer is silent — and it is
**not** introduced by the Phase 5 bs>1 work; it predates the
`KILN_CUDA_GRAPHS_BATCHED=1` enable code path entirely.

**Blocker #2: kt-bridge gdn_gates contiguous failure on the live
driver.** `kiln-bench --paged` and `kiln serve` both fail at first
forward pass with:

```
gdn decode gates fused backend
kt-adapter: gdn_gates a → kt failed
kt-bridge: kt_tensor_from_candle_cuda_borrow: tensor must be contiguous
```

This regression is fresh on `main` HEAD — the kt-typed gdn_gates wire
landed in commit `64e0b5d8` and was flipped on by default in commit
`efcba50b` (2026-05-25 19:22:04 UTC). With `KILN_DISABLE_KT_API_GDN=1`,
the failure mode shifts to `kiln_gdn_gates_bf16 failed with status 500`
(known issue #1066 sccache class), which persisted even after
`cargo clean -p kiln-gdn-kernel` + `SCCACHE_RECACHE=1` rebuild with an
invalidated source.

`kiln-smoke-check` (the official sccache-corruption healer) also
**fails on the same path**. The skill notes say: "No known
sccache-corruption signature in output." Both blockers reproduce
across fresh sccache namespaces and identical commits, so this is a
real `main` HEAD bug, not a per-pod cache artifact.

### Default-flip held back

Per the task contract ("if sanitizer surfaces NEW errors not captured
by the existing audit, document them clearly and DO NOT flip the
default"), the `KILN_CUDA_GRAPHS_BATCHED=1` default flip is held back.

Strictly, sanitizer surfaced **zero new errors** — the bs>1 path
itself appears memory-safe. But the validation gate cannot be
declared green until the production driver (`kiln-bench --paged`,
`kiln serve` with concurrent requests) can actually reach the bs>1
captured graph replay under sanitizer. Both blockers above prevent
that, and both are upstream of the bs>1 capture path.

### Recommended next steps

1. **Fix blocker #1 (kt-rotary shape contract)** — audit the call to
   `fused_rotary_qk` from `gqa_attention_paged_decode_contiguous_batch`
   and ensure `cos` / `sin` are shaped `[batch, head_dim]` to match the
   kernel's expectation. The error message `cos [2, 128] != [2, 256]`
   suggests `head_dim=256` on the kernel side but `[2, 128]` from the
   inv_freq broadcast — likely a missing `partial_rotary_factor`
   adjustment or a halved `rotary_dim` somewhere.
2. **Fix blocker #2 (kt-bridge contiguous on gdn_gates `a`)** — the
   `a` tensor passed into `gdn_gates_bf16_kt` must be made contiguous
   at the call site, or the kt-bridge must accept non-contiguous
   inputs with an explicit copy. The candle fallback (set
   `KILN_DISABLE_KT_API_GDN=1`) currently triggers a different sccache
   class of failure that needs a separate diagnosis.
3. **Re-run the end-to-end sanitizer sweep** once the two blockers
   land. With those fixed, the same compute-sanitizer command on
   `kiln serve` + 4 concurrent chat completions should run to
   completion and produce `ERROR SUMMARY: 0 errors` against the full
   captured-graph replay surface.
4. **Then flip the default** by changing `batched_graph_enabled()` in
   `crates/kiln-model/src/cuda_graph.rs` to default-on
   (`KILN_DISABLE_CUDA_GRAPHS_BATCHED` opt-out) and update this doc's
   "Headline" section.

Pod artifacts (sanitizer log on the kiln-model test binary):
saved to `/tmp/test-san.log` on pod `vrz9h2elkn71fa` (preserved as
this commit's reference data).

