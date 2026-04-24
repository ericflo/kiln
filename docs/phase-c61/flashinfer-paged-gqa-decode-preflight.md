# Phase 6 FlashInfer-Style Paged GQA Decode Preflight

Date: 2026-04-24

## Scope

Check whether current `main` still has remaining work for a new minimal
FlashInfer-style paged GQA decode crate, or whether the remaining-work
precondition fails before spending RunPod time.

The requested implementation envelope was intentionally narrow:

- bf16 Q/K/V
- single query token
- Qwen3.5-4B GQA head ratio
- existing Kiln paged-KV block table layout
- causal decode only
- forward path only
- no Triton or Python runtime
- thin C ABI and Rust FFI only

## Remaining-Work Preflight

Result: stop with a documentation-only PR.

- `gh pr list -R ericflo/kiln --limit 10` returned no open overlapping PRs.
- `PROFILING.md` includes the post-PR-#500 / PR #502 current-main profile and
  the post-PR-#502 GDN full-chunk audit. Those artifacts keep decode led by GDN
  ranges, not full-attention paged decode, and explicitly warn not to retry GDN
  full-chunk micro-ports.
- `crates/kiln-model/src/forward.rs` still uses the paged KV path for
  full-attention decode, but the single-token path already routes through a
  native CUDA paged GQA decode backend when eligible.
- `crates/kiln-model/src/paged_kv_cache.rs` already exposes
  `pool_tensors(...)` for fused attention kernels to borrow the raw K/V pools.
- `crates/kiln-flash-attn/src/lib.rs` already exposes
  `flash_attn_paged_decode(...)`, and
  `crates/kiln-flash-attn/csrc/flash_attn/flash_api_c.h` declares the C ABI
  `kiln_flash_attn_fwd_paged_decode(...)` for single-query GQA paged decode.

## Current Decode Path

The full-attention paged path already has the essential pieces this task asked
for:

1. `gqa_attention_paged(...)` writes the current token's K/V into
   `PagedKvCache` and, for `seq_len == 1`, enters the fused-decode fast path
   when the backend advertises `supports_flash_attn_paged_decode()`.
2. `try_flash_attn_paged_decode(...)` borrows the raw per-layer K/V pools via
   `PagedKvCache::pool_tensors(...)`, verifies the single-token GQA envelope,
   builds the paged block-table tensor, and calls the backend fused decode API.
3. The CUDA backend's fused decode implementation is backed by the vendored
   `kiln-flash-attn` C ABI entrypoint
   `kiln_flash_attn_fwd_paged_decode(...)`.
4. If any eligibility check fails, the existing Candle materialized read path
   remains the fallback.

That means current `main` does not lack a native CUDA paged GQA decode kernel
for single-token full-attention decode. Adding a second
`kiln-paged-gqa-kernel` crate would duplicate an existing hot-path backend,
not satisfy remaining implementation work.

## Decision

No CUDA or Rust source change is appropriate for this task. The remaining-work
precondition fails because the native CUDA paged GQA decode path already exists
and is wired into `crates/kiln-model/src/forward.rs` for the single-token paged
full-attention decode case.

This is separate from earlier FlashInfer-specific redirects: Kiln does not have
a FlashInfer-vendored crate, but it already has the native CUDA single-query
paged GQA decode capability that the task's precondition required to be absent.
The current Phase 6 source of truth also leaves full-attention decode below the
optimization keep bar compared with GDN decode/prefill work.

## Validation

No RunPod was launched and no CUDA build was run because the precondition
failed before source edits. Local validation was limited to:

```bash
git diff --check
```

## Next Step

Do not queue another minimal FlashInfer-style paged GQA decode port unless a
fresh profile and code inspection show that the existing
`kiln_flash_attn_fwd_paged_decode(...)` path was removed or no longer serves
single-token paged full-attention decode. The next Phase 6 task should pick a
different current hotspot from fresh profiling rather than duplicate this
backend.
