# Archived Metal integration records

Historical records from the #1082 candle-removal campaign. The work they
describe is **fully landed**: `candle-core` / `candle-nn` are completely
removed from the workspace, and the Metal substrate is kt-native objc2-metal
throughout (see `crates/kiln-tensor/src/metal_types.rs` and
`crates/kiln-tensor/src/metal_storage.rs`). These docs are kept as migration-
pattern references only — their present-tense descriptions of candle APIs no
longer reflect the codebase.

- `METAL_INTEGRATION.md` — the Phase 4 pattern for wiring a real Metal
  implementation into a `DeviceOp::metal_fwd` scaffold, written when the
  substrate was shared with candle's Metal kernels.
- `metal-types-objc2-swap-plan-2026-05-28.md` — the multi-PR sequence that
  retired the `candle_metal_kernels::*` / `candle_core::*` re-exports in
  `metal_types.rs` and flipped every caller onto the kt-native objc2-metal
  substrate.

Companion STOP/status docs from the same campaign remain under `docs/`
(e.g. `docs/archive/candle-removal/metal-cargo-toml-candle-drop-stop-2026-05-28.md`,
`docs/archive/candle-removal/candle-removal-status-2026-05-28-pm.md`).
