# Phase 6 C53 GDN conv decode hotspot audit

Date: 2026-04-24
Base: current `main` after C52 (`3ea3747`, with C52 artifact commit `55a5d9f`).
Scope: audit `:kiln/gdn/conv` in the native-MTP decode workload before attempting any new causal-conv implementation.

## Precondition

The C52 recommendation is still the live profiling recommendation in `PROFILING.md`: `:kiln/gdn/conv` is the rank-1 decode NVTX range at 15.4% wall-clock, 562.361 ms over 2328 instances. No open PR was found targeting `:kiln/gdn/conv`, `kiln/gdn/conv`, `causal_conv1d`, or C53. No newer profiling artifact supersedes `docs/archive/phase-c/phase-c52/summary.json`.

## Static call-path audit

Current CUDA decode uses the vendored causal-conv update path for the C52 native-MTP decode shape:

- `crates/kiln-model/src/forward.rs` enters `:kiln/gdn/conv`, transposes `mixed_qkv` to `[B, C, T]`, and for `seq_len == 1` calls `backend.causal_conv1d_update(...)` when `backend.supports_causal_conv1d_update()` is true.
- `crates/kiln-model/src/backend/cuda.rs` returns true unless `KILN_DISABLE_FUSED_CONV1D` is set, then declines only if `kiln_conv1d_kernel::supports(...)` rejects the exact envelope.
- `crates/kiln-conv1d-kernel/src/lib.rs` supports the Qwen3.5 decode envelope: BF16 `x`, BF16 weights, F32 state, `[B, C, 1]`, kernel size 4, CUDA storage, and contiguous inputs.
- `crates/kiln-conv1d-kernel/csrc/causal_conv1d_update.cu` launches `kiln_conv1d_update_k4_kernel` with one thread per `(batch, channel)`, updates F32 state in place, and fuses SiLU.

For the C52 workload, native-MTP decode forwards are single-token decode calls (`seq_len == 1`) on CUDA with the normal kill switch unset. The fallback `causal_conv1d_decode` path is therefore not expected for supported CUDA decode; remaining `:kiln/gdn/conv` wall-clock is expected to be layout/contiguity, wrapper/output allocation, and the vendored update kernel itself.

## Instrumentation shipped

C53 adds child NVTX ranges under `:kiln/gdn/conv`:

- `:kiln/gdn/conv/layout` for `transpose(1, 2)?.contiguous()` into `[B, C, T]`.
- `:kiln/gdn/conv/update` for the backend fused update call, including wrapper checks, output allocation, and CUDA launch.
- `:kiln/gdn/conv/prefill_update` for the analogous prefill fused call.
- `:kiln/gdn/conv/fallback_decode` and `:kiln/gdn/conv/fallback_prefill` for explicit fallback attribution if support checks ever decline.

This is intentionally measurement-only instrumentation. It does not change tensor values, backend selection, or kernel dispatch.

## GPU attempt record

Required RunPod validation/profiling could not complete because both A6000 pods developed SSH failures during the build/validation phase. Both pods were terminated immediately to stop spend.

| Pod | GPU | Outcome | Termination |
| --- | --- | --- | --- |
| `nszu2wno80dvef` | RTX A6000 | `cargo test`/build started with sccache hits, then `wait-file` timed out and SSH began resetting connections. | Terminated |
| `efeldsa2tpx69s` | RTX A6000 | Fresh retry reached the same `wait-file` SSH timeout during validation; subsequent bounded SSH state check produced no output. | Terminated |

Commands attempted on each retry followed the required pattern:

```bash
cargo test -p kiln-conv1d-kernel
cargo test -p kiln-model test_causal_conv1d_update_matches_fallback --features cuda -- --nocapture
cargo build --release --features cuda --bin kiln-bench
```

No C53 before/after benchmark is claimed.

## Conclusion

Audit-only, no implementation. The existing `kiln-conv1d-kernel` vendored causal-conv update already covers the C52 native-MTP decode shape. Without a successful child-range profile, there is no evidence for a minimal wrapper/layout fix, and forcing another causal-conv implementation would violate the vendor-first/minimal-scope rule.

Next target recommendation: profile with the new C53 child NVTX ranges when RunPod SSH is healthy. If `:kiln/gdn/conv/update` dominates, treat causal-conv as kernel/runtime-bound and move to the adjacent `:kiln/gdn/gates` + `:kiln/gdn/gated_norm` cluster. If `:kiln/gdn/conv/layout` dominates, scope a separate minimal layout/contiguity task.
