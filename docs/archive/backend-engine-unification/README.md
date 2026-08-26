# Backend & engine unification records (archived)

The three-document record of the backend/engine unification effort across
CUDA, ROCm, Metal, and Vulkan — original plan (2026-06-05), adversarial
outcome review (2026-06-07), and the completion plan that drove the
behavioral migration to "type A" genuine unification.

**Status: fully landed.** Verified against the live tree at archival time
(2026-08-26):

- **Phase 1 — focused traits authoritative.** `BackendRuntime`
  (`crates/kiln-model/src/backend/mod.rs`) now declares only identity +
  composition glue (name/device/as_any over the focused-trait supertraits);
  every concrete backend (`CpuBackend`, `CudaBackend`, `MetalBackend`,
  `RocmBackend`, `VulkanBackend`) implements the focused traits directly, and
  no `impl<T: BackendRuntime> XBackend for T` blanket-forward shim remains.
- **Phase 3 — residency routed through `ResidentRegistry`.** It is now a
  required supertrait of `ResidencyBackend` (routing direction fixed),
  implemented per backend, and the registry no longer lives only in a test
  mock.
- **Phase 4 — matmul/linear via the request surface.** `ops/matmul.rs`'s
  per-backend `Device::` arms are gone (the file no longer exists in
  kiln-model); `supports_matmul_request` answers through
  `LinearBackend::runtime_supports_matmul_request`, not a name table.
- **Phase 5 — replay is a production path.** The Vulkan-resident
  single-submit decode orchestrator (`vk_decode_resident.rs`,
  `CommandBatch`) runs through the `ReplayBackend` contract in
  `forward/model_dispatch.rs`.
- **Phase 6 — training policy.** `TrainingPrecisionPolicy::for_device_family`
  is test-only; production selects via the runtime capability route.
- **Scoreboard derived, gate local.** `scripts/generate_backend_capability_report.py`
  computes every phase's `genuine` status from machine signals (zero
  hardcoded `"status": "covered"` literals), and
  `scripts/check_unification_gates.sh` is the runner-agnostic pre-push gate.
  The quick-win defects from the review (stale workflow dropdown, stale
  `tensor.rs` doc) are also fixed in the live tree.

These docs are kept as historical design and review records; their
present-tense "status: active" / "do not treat covered as migrated" markers
describe the 2026-06-07 point in time, not current state. The live capability
report surface — `docs/backend-capability-report.{md,json}` +
`scripts/generate_backend_capability_report.py` +
`crates/kiln-model/tests/backend_capability_contract.rs` — remains at `docs/`
root and is NOT part of this archive.
