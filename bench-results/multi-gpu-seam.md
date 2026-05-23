# Phase 0.6 — Multi-GPU seam audit

Source of truth: `bench-results/multi-gpu-seam.csv` (one row per hardcoded
device-0 site).
Regenerate: `scripts/audit-multi-gpu-seam.sh`.

Multi-GPU stays **out of scope** for #1082 per anti-pattern 12 — but the
hardcoded `Device::Cuda(0)` pattern stays a bug regardless because:

1. Two kiln processes on one box need to land on different GPUs (replica
   serving, eval-while-training, multi-tenant dev pods).
2. A future TP rewrite must not have to revisit ~100+ call sites.

The audit greps for `Device::new_cuda(0)`, `Device::new_cuda_with_stream(0)`,
and `Device::Cuda(0)`. A `#[test]` / `#[cfg(test)]` walk-back categorizes
each site as `test` / `example` / `production`.

## Headline numbers

| Crate | Total | production | test | example |
|---|---:|---:|---:|---:|
| kiln-model | 76 | 0 | 76 | 0 |
| kiln-train | 23 | 0 | 18 | 5 |
| kiln-gdn-kernel | 9 | 0 | 8 | 1 |
| kiln-opd-loss-kernel | 8 | 0 | 7 | 1 |
| kiln-flash-attn | 3 | 0 | 3 | 0 |
| kiln-rmsnorm-kernel | 2 | 0 | 1 | 1 |
| kiln-marlin-gemm | 2 | 0 | 2 | 0 |
| kiln-conv1d-kernel | 1 | 0 | 1 | 0 |
| **kiln-server** | **2** | **2** | 0 | 0 |
| **TOTAL** | **126** | **2** | **116** | **8** |

The issue's "~25 sites confirmed" was a partial enumeration of specific
files in the spec body. The complete picture is **126 sites**, but only
**2 are production** — and both already live in the file we want them in
(`crates/kiln-server/src/device.rs:24,31`), which is the canonical
device-selection accessor.

The work is therefore concentrated in test code: introduce a
`kiln_core::test_support::cuda_device()` helper that test runners can
override (env var → device index → fall through to `Device::new_cuda(0)`),
and mechanically substitute the 116 test call sites.

## Causal links forward

- **Phase 1 dependency**: when `kiln_tensor::Device` lands, the test helper
  is the shim that lets `_test.rs` files reach it without re-rebuilding
  test fixtures.
- **Anti-pattern 12 enforcement**: this audit re-runs as part of Phase 9's
  bench-gate. A merge that introduces a new hardcoded `Device::Cuda(0)`
  outside the two whitelisted sites is rejected.
- **Multi-process serve safety**: `crates/kiln-server/src/device.rs:24,31`
  should already be reading `CUDA_VISIBLE_DEVICES` / a `KILN_DEVICE_INDEX`
  env var rather than hardcoding `0`; verify before Phase 9 lands.

## Replacement map

| Bucket | Replacement |
|---|---|
| `production` | `kiln_core::device::primary_cuda()` (already implemented in `kiln-server/src/device.rs`; expose at crate boundary) |
| `test` | `kiln_core::test_support::cuda_device()` (new helper, returns `None` to skip when no GPU is visible) |
| `example` | `kiln_core::device::cuda_from_args_or_primary()` (accept `--gpu N` CLI flag, fall through to primary) |
