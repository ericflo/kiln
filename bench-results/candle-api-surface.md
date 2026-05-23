# Phase 0.1 — Candle API surface

Source of truth: `bench-results/candle-api-surface.csv` (one row per distinct
`candle_*::Path::Symbol`, sorted by call-site count desc).
Per-call-site detail: `bench-results/candle-api-surface.raw.tsv`.
Regenerate: `scripts/audit-candle-usage.sh`.

Excludes the vendored `vendor/candle-core/` tree — that is what we are
removing in Phase 7, not surface we have to migrate.

## Headline numbers

- 55 distinct candle API paths used across `crates/`
- 1,799 total call sites
- Top buckets by count concentrate in error machinery (`bail!`, `Error::Msg`,
  `Result`), backend storage variants (`Storage::Metal`, `Storage::Cuda`),
  and a handful of identity types (`Tensor`, `Device`, `DType`, `TensorId`,
  `Var`).

## Migration buckets (the shape of the work, not the order)

The 55 APIs partition into seven kinds. Each kind has a different shape of
replacement work in subsequent phases.

### A. Error / Result plumbing — replaced wholesale (~640 sites)

- `candle_core::bail!` — 491
- `candle_core::Error::Msg` — 106
- `candle_core::Result` — 43

Action: introduce `kiln_tensor::Error` + `kiln_tensor::Result` in Phase 1.
These are macro / type aliases the codebase pulls in via `use` — mechanical
search-and-replace once Phase 1 lands. Not on any critical perf path.

### B. Backend storage variants — disappear with the backend port (~427 sites)

- `candle_core::Storage::Metal` — 232
- `candle_core::metal_backend::buffer_o` — 232 (raw-buffer extraction helper
  for our custom MSL kernels — all in `crates/kiln-model/src/backend/metal.rs`)
- `candle_core::Storage::Cuda` — 195
- `candle_core::metal_backend::MetalDevice` — 46
- `candle_core::metal_backend::DeviceId` — 46
- `candle_core::CpuStorage` — 6
- `candle_core::cuda_backend::cudarc::driver::DevicePtr` — 6

Action: these vanish when `BackendRuntime` implementations route through
`kiln_tensor::Storage` (Phase 1). The Metal `buffer_o` site count alone
gates the Metal Phase-2/3 ports — kiln-tensor's Metal storage variant must
expose an equivalent `as_metal_buffer()` zero-copy accessor before the
`backend/metal.rs` rewrite can land.

### C. Identity / dtype / device types — Phase 1 deliverables (~150 sites)

- `candle_core::Tensor` — 20
- `candle_core::Device` / `::Cpu` / `::Metal` / `::Cuda` / `::new_cuda` — 30+24+18+14+5 = 91
- `candle_core::DType` / `::F32` / `::BF16` / `::F16` — 7+39+10+6 = 62
- `candle_core::TensorId` — 30
  (Note: `vk_autograd.rs:21` already keys `VkGradStore` on this — Phase 2.5's
  stable `kiln_tensor::TensorId` contract has a direct mapping target.)
- `candle_core::D::Minus1` — 85 (axis sentinel; replace with `Axis::Last` or
  `-1isize`-style negative-axis helper in `kiln_tensor`)

Action: Phase 1 ships drop-in replacements; per-call-site migration is the
gradual rollout under per-backend feature flags `KILN_USE_KILN_TENSOR_*`.

### D. Autograd surface — Phase 6a target (~25 sites)

- `candle_core::Var` — 14
- `candle_core::Var::from_tensor` — 14
- `candle_core::backprop::GradStore` — 6
- `candle_core::op::BackpropOp` — 5

Action: lifted into `kiln-autograd` (Phase 6a) using `vk_autograd.rs` (173 LOC)
as the template. `Var` becomes a marker on `Parameter::backward_storage`
(Phase 2.5); `GradStore` becomes `kiln_autograd::GradStore` keyed on the
stable `kiln_tensor::TensorId`.

### E. I/O — port-and-keep (~14 sites)

- `candle_core::safetensors::load` — 14

Action: port the safetensors loader into a shared `kiln-tensor` helper.
Independent of any backend. Already noted as "Safetensors loading shared
across backends (port from candle)" in Phase 1.

### F. SDPA — Metal-specific kernel binding (~9 sites)

- `candle_nn::ops::sdpa` — 9 (all in `crates/kiln-model/src/backend/metal.rs`)

Action: keep the MPS-backed SDPA logic; route it through `kiln-mps` in Phase 3
so it composes with the stream-planner. The kernel itself is fine; the
candle-typed signature is what changes.

### G. Device-probe utility — trivial (~7 sites)

- `candle_core::utils::cuda_is_available` — 7

Action: replace with `kiln_core::Device::probe_cuda()` or similar; existing
references at `crates/kiln-server/src/device.rs:21` already centralize this.

## What this audit does NOT prove

This audit answers the "what must move" question, not the "how hard is each one"
question. In particular:

- It cannot detect `use candle_core::Tensor as T;` aliasing.
- It does not see `Tensor::*` method calls (`x.matmul(y)`, `x.contiguous()`,
  etc.) — those don't carry the `candle_core::` prefix at the call site.
  Those are the bulk of the actual surface and are buckets onto the
  `BackendRuntime` trait + `kiln_tensor::Tensor` methods in Phases 1/3/4/6.
- It does not measure perf cost. The 491 `bail!` sites are the cheapest to
  retire; the 232 `buffer_o` sites are the heaviest (every metal kernel uses
  it).

A complementary "Tensor:: method audit" against `crates/kiln-vulkan-kernel/`
(where `VkTensor` mirrors the API we'll lift to `kiln_tensor::Tensor`) is the
natural follow-up — Phase 1's parity-test suite needs that map to know which
methods to ship in the first PR.

## Causal links forward

- **Phase 1 dependency**: `kiln_tensor::{Tensor, Device, DType, TensorId,
  Storage, Result, Error}` and the `bail!` analogue must land before any
  bulk migration of bucket A/C above is possible.
- **Phase 2.5 dependency**: the 30 `TensorId` sites must continue to round-
  trip stably across storage-variant transitions (anti-pattern 11).
- **Phase 3 dependency**: 232 Metal `buffer_o` sites concentrate in one file
  and one backend — the Metal port is the largest single-file rewrite in
  this migration. Plan accordingly: that file is `backend/metal.rs` and a
  staged rewrite under `KILN_USE_KILN_TENSOR_METAL=1` is the only safe
  approach.
- **Phase 6a dependency**: 25 autograd-surface sites all live in
  `kiln-vulkan-kernel`, `kiln-train`, and the loss-kernel crates. The lift
  is mechanical once `kiln-autograd::{Var, GradStore, BackpropOp}` ships.
- **Phase 7 gate**: the Definition of Done item "No candle in any
  `[dependencies]` block" is verifiable by re-running this script and
  confirming a count of zero.
