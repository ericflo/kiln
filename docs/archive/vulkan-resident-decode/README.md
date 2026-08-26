# Vulkan-resident decode record (archived)

[`vk_resident_decode_plan.md`](vk_resident_decode_plan.md) was the load-bearing
acceptance spec for closing the Vulkan-vs-CUDA decode-throughput gap on
Qwen3.5-4B: eliminate the per-kernel CPU↔Vulkan round trip so one decode step
pays at most a single host→device upload (input) and a single device→host
readback (sampled token).

**Status: fully landed.** Verified against the live tree at archival time
(2026-08-26):

- The `dispatch_*_resident` kernel surface (`resident.rs`), the decode buffer
  pool (`decode_resident_pool.rs`), the single-submit `CommandBatch`
  (`cmd_batch.rs`), and the resident paged KV cache (`vk_paged_kv_cache.rs`)
  all remain live in `crates/kiln-vulkan-kernel/`.
- The round-trip elimination itself — the native single-submit orchestrator
  `model_forward_paged_last_token_resident_native_vk` plus the per-block
  resident transformer blocks in `crates/kiln-model/src/vk_decode_resident.rs`
  — is the production decode fast-path, selected through the `ReplayBackend`
  contract in `forward/model_dispatch.rs`.
- The plan's final 10-run bench records gate (e.2) reached on sustained p50
  (54.6 tok/s, 99.3% of the 55 tok/s target); end-to-end parity against the
  legacy path stayed bit-identical
  (`crates/kiln-model/tests/vk_resident_decode_parity.rs`).
- The only remaining lever the plan names — cooperative-matrix BF16 GEMMs —
  was explicitly "out of scope for this goal" even while the plan was active;
  it is a new shader-level optimization effort, not unfinished work this record
  tracks.

Kept as the historical acceptance spec and performance narrative (the commit
ladder from 1.04 → ~45.5 tok/s mean / 54.6 tok/s sustained). Its present-tense
"remaining work" sections describe the state as of 2026-05-16, not current.
