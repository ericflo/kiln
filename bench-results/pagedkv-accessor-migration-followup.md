# PagedKvCacheKt accessor migration — follow-up progress (#1082)

Snapshot after commits `1a4acebb` and `fdd29049` (this branch) on top of
the merged `40666082` `ce/1082-pagedkv-accessors-migrate` batch.

## State as of this doc

The five PagedKvCacheKt parity helpers are now wired at all production
read sites inside `gqa_attention_paged_decode_contiguous_batch` and
`try_flash_attn_paged_decode`:

| Helper | Wired at lines (forward.rs) |
| --- | --- |
| `try_kt_paged_kv_block_size` | 16999, 17820 *(this branch)*, 18097 *(this branch)*, 18681, 18718, 18981, 19028 |
| `try_kt_paged_kv_is_fp8` | 17042, 17756, 18607, 18672, 18712, 18897, 18967, 19017 |
| `try_kt_paged_kv_num_layers` | 17761 |
| `try_kt_paged_kv_pool_tensors_present` | 16744 |
| `try_kt_paged_kv_num_blocks` | (helper exists, no caller threaded yet) |

All wires use the same CUDA-gated shape:

```rust
#[cfg(feature = "cuda")]
let x = try_kt_paged_kv_<accessor>(paged_cache.<accessor>(), kt_paged_cache);
#[cfg(not(feature = "cuda"))]
let x = paged_cache.<accessor>();
```

with `kt_paged_cache: Option<&PagedKvCacheKt>` threaded through the
enclosing fn signatures.

## Remaining `paged_cache.<accessor>()` reads in `forward.rs`

These are bare reads at sites where `kt_paged_cache` is not yet
threaded into the enclosing fn. Each represents a small follow-up
threading PR; none are required for parity correctness today because
the helper short-circuits to the candle path when `kt_paged_cache` is
`None`.

### Inside `CachedPagedDecodeMeta`

- `CachedPagedDecodeMeta::build` (line 17184: `let page_block_size =
  paged_cache.block_size();`) — used by the batched-decode
  metadata-build path on the non-stable-buffer branch.
- `CachedPagedDecodeMeta::build_with_stable_buffers` (line 17317: same
  read) — the captured-graph stable-buffer variant.

Wiring requires adding `kt_paged_cache: Option<&PagedKvCacheKt>` to
both build fn signatures and threading it from
`model_forward_paged_decode_contiguous_batch_hidden_inner` (line
20291) through the call site at 20431 / 20443.

### Inside Vulkan paths

- `model_forward_paged_last_token_resident_native_vk` (line 21541:
  `paged_cache.num_blocks(), paged_cache.block_size()`).
- `try_resident_block_full_attn_b1` (line 21809: same two reads).

These are Vulkan-only paths; `PagedKvCacheKt` is CUDA-only (`#[cfg(feature = "cuda")]`),
so threading `kt_paged_cache` here is a no-op on Vulkan builds.
The cleanest path is to add the `cfg(feature = "cuda")`-gated parameter
and pass `None` from the Vulkan callers — which both gives the eventual
CUDA-graph dispatcher a place to thread real kt cache through, and
keeps Vulkan callers explicit about not having a kt twin.

### Inside `mtp_forward_step`

- `mtp_forward_step` (line 22656) uses `mtp_cache: &PagedKvCache` —
  there is currently no PagedKvCacheKt twin for the MTP cache. If/when
  MTP migrates, the same accessor-helper pattern applies.

## Why this is the right ordering

The accessor helpers were introduced (db2055c1 through 40666082) so
each batched-decode call site can flip to the kt path one wire at a
time, with parity-checked CUDA-graph-stable reads, instead of one
mega-PR replacing every `paged_cache.*()` at once. The two new wires
in this branch follow that grain: each is 4 lines of CUDA-gated code
inside an already-threaded fn, defaults off, and asserts kt parity
when the env gate is on.

The remaining `CachedPagedDecodeMeta` / Vulkan / MTP sites need
threading work first — not just one-line wires — and they all sit
outside the gqa contiguous batch fn that consumes the cache during
the actual decode kernel dispatch. So those land in follow-up PRs.

## Compute-sanitizer re-run status

Re-running compute-sanitizer memcheck on the merged Phase 5 +
PagedKvCacheKt accessor state is the next deferred Phase 5 step
(see `bench-results/cuda-graph-bs2-memcheck.md`). Not run in this
branch. To run:

```bash
# A6000 pod, kiln main + this branch
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln
compute-sanitizer --tool memcheck --report-api-errors no \
  ./target/release/kiln serve <bs2-repro-args>
```

The expected result, given that 14872d65 fused the kv_slot writer and
the bs>1 strict-paged probe was short-circuited (b5c7156c), plus
b571b57d wired the bs>1 graph wrapper to stable RoPE tables, is **zero
ILLEGAL_ADDRESS faults**. If that holds, the `KILN_CUDA_GRAPHS_BATCHED`
default flip is a separate landing.
