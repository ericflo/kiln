# Backend And Engine Unification — Completion Plan (Drive Everything to A)

Date: 2026-06-07
Status: active
Predecessor: [`backend-engine-unification-plan.md`](backend-engine-unification-plan.md) (original plan, Phases 0–8)
Evidence base: [`backend-engine-unification-review-2026-06-07.md`](backend-engine-unification-review-2026-06-07.md) (what actually landed)

## Purpose

The original plan's contracts, module decomposition, and observability landed. The
**behavioral migration** did not, for Phases 1, 3, 4, and 5: the new shared surfaces
are additive scaffolds (type **B**) that coexist with the original `Device::`-keyed
dispatch. This plan closes that gap. The goal is **A — genuine unification — for
every dispatch family**, and a capability report that proves it instead of asserting
it.

Read the review doc first. Then work the workstreams below. **Do W0 first** — it
makes progress measurable and un-fakeable; without it, every later workstream can be
"completed" by adding another scaffold.

## Tooling stance: runner-agnostic, not GitHub-bound

Coordinate through the **GitHub issue and the git branch/PR** — those are fine. But
**do not add or depend on GitHub Actions, GitHub-hosted runners, or
`.github/workflows/*.yml`** for any gate in this plan (their uptime has been
unreliable). Concretely:

- All gates run from **one runner-agnostic command** that works on any host —
  developer laptop, the bench hosts (runpod A6000, 4090-via-WSL), or an agent. Add a
  `scripts/check_unification_gates.sh` (or a `make gates` target) that runs, in
  order: `python3 scripts/generate_backend_capability_report.py --check`, the
  `backend_capability_contract` test, the grep-guards (DoD #6), the CPU conformance
  tests, and `python3 scripts/check_backend_latency_fixtures.py … --require-covered`.
  Make it the documented pre-push step; optionally ship a repo `core.hooksPath`
  pre-push hook that calls it. This is the gate the existing nightly GitHub workflow
  *would* have run — owned by the repo, not by GitHub.
- Existing workflows (`perf-regression-nightly.yml`) are **not** the source of truth.
  Don't extend them; don't make new gates require them. Treat them as optional
  convenience that may be flaky or removed.
- Latency-artifact provenance is **host-captured at run time**, not bound to a CI
  identity (see W0.4).
- Where this plan earlier said "CI", read "the local gate command run by a human or
  agent on a real host."

## Definition of A (the target for every dispatch family)

A dispatch family X (matmul, linear, attention, GDN, conv, sampling, optimizer,
residency, replay) is **A — genuinely unified** when ALL of the following hold and
are machine-checked:

1. **Call sites select via contract, not identity.** In the designated dispatch
   files for X, behavior is chosen through a capability / route / request object or a
   focused-trait method — never `match device { Cuda => …, Rocm => … }` or
   `matches!(t.device(), Device::Cuda(_))` to pick an execution path.
   *(Device matching is still allowed for the sanctioned exceptions below.)*
2. **Backends own the implementation through the focused trait.** Each concrete
   backend `impl`s the focused trait for X directly; the blanket `impl<T:
   BackendRuntime> XBackend for T` forwarding shim is removed for X.
3. **The monolith shrinks.** `BackendRuntime` no longer declares X's methods (they
   live on the focused trait). Method-count is gated (see W1).
4. **A behavioral test proves the seam.** A test using a CPU backend or an
   instrumented mock proves the call site invokes the contract/route path (not the
   legacy branch). Hardware parity tests may remain `#[ignore]`/availability-gated,
   but at least one CPU test (runnable in the local gate, no GPU) must exercise the
   dispatch seam for X.
5. **The report status for X is derived, not declared.** The generator computes X's
   status from machine signals (grep-negative on the designated files + focused-trait
   impl count + the named behavioral test passing), not a hardcoded literal.
6. **A regression guard exists.** A test asserts the legacy `Device::` dispatch
   branch for X does not reappear in the designated files, and the forwarding shim
   does not reappear.

### Sanctioned `Device::` matching (these stay; they are not B)

- **Backend construction:** `for_device_kt` (`mod.rs:3668`) and
  `TrainingAccelerationProfile::for_device` (`mod.rs:493`) — constructing the right
  concrete backend is inherently device-keyed. Document these as the *only* allowed
  dispatch-selection `Device::` matches.
- **Physical resource plumbing:** stream synchronization
  (`cuda_synchronize_default_stream` vs `rocm_…`, `forward.rs:2511-2514`), device
  index threading (`forward.rs:1777`), and device-handle extraction. These move
  bytes/handles; they do not choose an algorithm.
- **CPU-vs-not gates** are a code smell but acceptable *only* when expressed as a
  capability query (`supports_*` / a `Support` predicate), not a raw
  `matches!(.., Cuda | Metal | Vulkan | Rocm)`.

## Un-gameable Definition-of-Done template (apply per work item)

Every B→A work item lands with: (a) the migration edit; (b) removal of the
superseded branch/shim; (c) a behavioral seam test (DoD #4); (d) a regression-guard
test (DoD #6); (e) the generator updated so X's status derives from real signals
(DoD #5); (f) `python3 scripts/generate_backend_capability_report.py --check`
passes; (g) `cargo check` on the portable path passes. A PR that adds types/tests
but leaves the legacy branch in place is **not done** — it is another scaffold.

---

## W0 — Honest, derived, un-gameable scoreboard (DO FIRST)

Without this, "done" is unmeasurable. Make the report tell the truth so every later
workstream has an objective finish line.

- **W0.1 — Two-axis phase status.** In
  `scripts/generate_backend_capability_report.py`, replace the 21 hardcoded
  `"status": "covered"` literals (phase table ~`1236-1472`; conformance gates
  ~`944-1200`) with a computed object per phase: `{contract: landed|absent,
  migration: none|partial|complete, genuine: bool}`, where `genuine = (migration ==
  complete)`. Initialize honestly: Phases 0/2/6/7 → `migration: complete` for the
  parts the review confirmed; Phases 1/3/4/5 → `migration: none`.
- **W0.2 — Derive `migration` from machine signals.** For each migration-bearing
  family, the generator runs the DoD #5 signals (a grep over the designated dispatch
  files returning zero legacy branches; focused-trait impl count == number of
  concrete backends; the named behavioral test present). `migration: complete` is
  emitted only when all signals pass. No literal may set `genuine: true`.
- **W0.3 — Make `--check` part of the local gate, not GitHub CI.** Add
  `python3 scripts/generate_backend_capability_report.py --check` and the
  `backend_capability_contract` test to `scripts/check_unification_gates.sh` (the
  runner-agnostic command from the Tooling stance), and document it as the required
  pre-push step (optionally a repo pre-push hook). Today `--check` is enforced only by
  a Rust unit test and is wired into nothing that runs by default, so a hand-edited
  green report goes uncaught — the local gate fixes that without depending on
  GitHub-hosted runners.
- **W0.4 — Close the latency-gate trust holes with host-captured provenance**
  (review defects #1, #3), without relying on GitHub Actions/runner identity:
  - Capture provenance **at run time on the actual hardware host**, embedded in the
    artifact by `run_backend_latency_fixture.py` / `write_backend_latency_result_artifact.py`:
    hostname, GPU model + UUID (`nvidia-smi --query-gpu=uuid` / `rocminfo` / Metal
    device id), driver/toolkit version, monotonic+wall timestamps, the exact command,
    and source/raw-log checksums. `check_backend_latency_fixtures.py` verifies these
    are present, internally consistent, and that the claimed GPU matches the fixture's
    `hardware`/`runner_labels`. The trust boundary ("self-attested but
    hardware-fingerprinted and reproducible from the committed command") is documented
    honestly in `backend-latency-result-schema.md` — fingerprinting raises the cost of
    a fake far above editing a JSON field, which is the realistic threat here.
  - Make `tracked_git_dirty` use `--untracked-files=all` so fabricated new files are
    not treated as clean (`write_backend_latency_result_artifact.py:180`).
  - Keep verification entirely in the **local gate command** (run by the bench-host
    agents that already produce these artifacts), not a `pull_request.paths` trigger.
  - The stale GitHub workflow dropdown (`perf-regression-nightly.yml:75`,
    `vulkan_rtx6000_decode_microbench` → `vulkan_strix_halo_decode_microbench`) is now
    a *low-priority* cleanup only if that workflow is retained; since we're
    de-emphasizing GitHub Actions, prefer dispatching fixtures via the bench-host
    agents directly. (See the quick-win list.)
- **DoD:** no `"status": "covered"` literal remains for a migration-bearing phase;
  every `genuine: true` is computed; `scripts/check_unification_gates.sh` runs
  `--check` and is documented as the pre-push gate; the latency gate rejects a
  fabricated/host-mismatched artifact (add a self-test case proving it).

---

## W1 — Phase 1: invert the forwarding; shrink `BackendRuntime` (B→A)

This is the structural prerequisite. After W1, W3/W4/W5 become "implement the focused
trait authoritatively" instead of "add a parallel trait."

Current: 12 focused traits are blanket forwards `impl<T: BackendRuntime> XBackend
for T` (`mod.rs:2604-3643`); concrete backends impl only `BackendRuntime` (95
methods, `mod.rs:655-1902`, intact).

- **W1.1 — Make focused traits authoritative, family by family.** For each family,
  move the method bodies out of `impl BackendRuntime for {Cuda,Rocm,Metal,Vulkan}
  Backend` into `impl XBackend for …Backend`. Replace the blanket `impl<T:
  BackendRuntime> XBackend for T` shim with the inverse: `BackendRuntime`'s X methods
  (if kept transitionally) become default methods that call `XBackend`, or are
  deleted outright once call sites move.
- **W1.2 — Migrate call sites to the focused trait.** Every production caller of a
  `BackendRuntime::<X method>` calls the `XBackend` method instead. (`generate.rs`,
  `forward.rs`, `trainer.rs` are the heavy callers.)
- **W1.3 — Shrink `BackendRuntime`.** Target: `BackendRuntime` declares only identity
  + composition glue. Add a test gating its method count at `<= K` (recommend K ≤ 8:
  identity, device, as_any, and the focused-trait supertrait bounds). Track the
  count down per family.
- **W1.4 — Keep construction device-keyed.** `for_device_kt` stays; it now returns an
  object implementing the focused traits. Annotate it as a sanctioned `Device::`
  match (DoD exception list).
- **DoD:** for each family, no blanket-forward shim remains; concrete backends impl
  `XBackend` directly; `BackendRuntime` method count below gate; a behavioral test
  resolves a call to the focused-trait method; a regression guard rejects
  re-introduction of `impl<T: BackendRuntime> XBackend for T`.

Suggested family order (low-risk → high-risk): `BackendIdentity`, `StartupBackend`,
`ConvBackend`, `SamplingBackend`, `OptimizerBackend`, `PagedKvBackend`,
`AttentionBackend`, `GdnBackend`, `LinearBackend`, `ResidencyBackend`,
`TrainingLossBackend`, `ReplayBackend`. (The last three are co-owned by W3/W4/W5/W6.)

---

## W3 — Phase 3: route residency through `ResidentRegistry` (B→A)

Current: `ResidentRegistry` blanket adapter (`mod.rs:3458-3516`) is consumed only by
a `cfg(test)` mock; lifecycle states hardcoded `RegisteredClean` (`residency.rs:140`);
per-backend registries are process-global statics.

- **W3.1 — Fix routing direction.** Production resident-activation APIs
  (`register/update/evict/resolve` on each backend) must delegate **to**
  `ResidentRegistry` (the plan: "route existing APIs through the registry"), not the
  reverse. Make `ResidentRegistry` the authoritative trait backends implement.
- **W3.2 — Drive the lifecycle.** Backends transition
  `RegisteredClean → DirtyDevice/DirtyHost` on update and `→ Unregistered` on evict.
  Set `replay_stability` per-resource at registration based on the backend's actual
  guarantee, not a constructor default.
- **W3.3 — Fix ownership/leak.** Move the Vulkan/CUDA/ROCm/Metal registries off
  process-global `static`s to per-backend-instance state, drained on `Drop`
  (review defect #7; `vulkan_residency.rs:10`, `cuda_rocm_common.rs:12,95`). Add a
  test that dropping a backend drains its registry.
- **W3.4 — Make `ResidentResource` describe the resident resource**, not the query
  tensor (`residency.rs:118-143`): shape/dtype/byte_len/layout should reflect the
  actually-resident buffer.
- **DoD:** a CPU/mock behavioral test exercises `register → resolve → update → evict`
  through `ResidentRegistry` and observes state transitions; production residency
  call sites use the registry (grep-guard); drop-drains-registry test passes.

---

## W4 — Phase 4: route matmul/linear through the request surface (B→A)

Current: `ops/matmul.rs:294-338` and `396-438` branch four per-backend arms;
`forward.rs:5941` selects `cuda_matmul`/`rocm_matmul` by device; the request types
never reach a backend; `supports_matmul_request` is a `match self.name()` table
(`capability.rs:2220`); the descriptor models only RowMajor rank-2-bias.

- **W4.1 — Extend the descriptor to be lossless first** (review defect #4). Teach
  `MatmulRequest` / `MatmulBlasRequest` to model ColMajor/transposed operands,
  mixed-dtype (lhs≠rhs≠out), and batched rank-3+ (`capability.rs:238-291`). Without
  this, migration silently drops capability. Add projection tests for each shape.
- **W4.2 — Add an authoritative dispatch method.** Give `LinearBackend` (or a new
  `MatmulBackend`) a `matmul(&self, req: &MatmulRequest, lhs, rhs) -> Result<Tensor>`
  that each concrete backend implements through its native primitive (cublasLt /
  hipBLASLt / MSL/MPS / SPIR-V), preserving the existing algo caches and tile
  heuristics.
- **W4.3 — Migrate the call sites.** `ops/matmul.rs` rank-2 / rank-3 paths and the
  `forward.rs` linear/lm-head selection call the dispatch method; delete the
  per-backend `matches!(device, …)` arms.
- **W4.4 — Replace the identity table.** `supports_matmul_request` answers per-backend
  via the trait, not `match self.name()` (`capability.rs:2220`).
- **W4.5 — Model real linear / surface cache stats.** Rename/extend `LinearRequest`
  beyond sampling kinds (`capability.rs:424`); expose `AlgoCacheStats` through a
  reporting hook so Phase-8 "matmul cache reporting" is live (review defect #5).
- **DoD:** `ops/matmul.rs` has zero `Device::` dispatch branches (grep-guard); a
  local-gate (CPU) parity test compares request-routed matmul against the CPU oracle
  for rank-2, rank-3 batched, transposed, and mixed-dtype; `supports_matmul_request`
  resolves per-backend.

---

## W5 — Phase 5: make replay authoritative (B→A)

Current: `cuda_graph.rs`, `rocm_graph.rs`, `vk_decode_resident.rs`, `cmd_batch.rs` =
zero diff vs `main`; `ReplayPlan` used only in tests; parity tests wrap a no-op
scaffold.

- **W5.0 — Fix the contract bugs first** (they become live on wiring): `byte_len = 0`
  for packed dtypes (`replay_plan.rs:82` — compute packed byte length, do not use
  `size_in_bytes()` blindly); make `StableWithinStep` actually participate in
  validation (`replay_plan.rs:50-89`); carry stride/offset/contiguity in
  `ResidentResourceRef`; make `validate_inputs` able to detect a key change
  (`KeyChanged`, `replay_plan.rs:287`); construct the currently-dead
  `InvalidateReason::Backend` / `CaptureError::DanglingPointer` where they apply.
- **W5.1 — Choose option (1) from the original plan: move runners behind one
  `ReplayBackend`.** Make `CudaGraphRunner` / `RocmGraphRunner` / `MetalGraphRunner` /
  the Vulkan `CommandBatch` decode plan implement (or be wrapped by) a backend-native
  `ReplayPlan`, so the shared layer owns bucketing keys, invalidation, input/output
  stability, and counters, while the backend owns capture/record/bind/submit.
- **W5.2 — Wire the production decode path.** The model decode hot path constructs the
  backend's `ReplayPlan` and calls `replay(...)`, instead of branching on
  `self.cuda_graph` / `self.rocm_graph` / resident-decode directly.
- **W5.3 — Real parity tests.** Replace the no-op scaffold tests with eager-vs-replay
  output comparisons (hardware-gated where needed) plus a CPU/mock `ReplayPlan`
  contract test that runs in the local gate (CPU, no GPU).
- **DoD:** a production decode path executes `ReplayPlan::replay` (behavioral test
  proves it); real eager==replay parity passes on ≥1 backend with hardware; the
  contract bugs in W5.0 are fixed with tests; grep-guard that decode no longer
  branches per-backend to choose the replay primitive.

---

## W2 — Phase 2 residuals (A polish)

- **W2.1** Assign `FallbackPolicy::ErrorInHotPath` where it belongs, or delete the
  dead variant + `Support::RequiresFeature` (review defect #8).
- **W2.2** Audit Metal/ROCm per-op attention decode paths that declare
  `decode_hot_path = NativeRequired` but may host-fallback at a different site
  (`forward.rs:22035-22043, 24509-24526`); close the gap so the guarantee holds.
- **W2.3** Consume the DeviceOp host-fallback counters in an enforcement/sentinel
  gate (currently telemetry-only, `device_op.rs:198-385`); wire into the Phase-8
  `no_unexpected_host_fallback` gate so it means something.
- **W2.4** Dedup the two diverging `decode_batch_generic_fallback_enabled` helpers
  (`generate.rs:744`, `forward.rs:244`) and the duplicate env parsers for
  `KILN_DECODE_HOT_PATH_DEBUG_FALLBACK`.

## W6 — Phase 6 residuals (finish the A)

- **W6.1** Remove `TrainingPrecisionPolicy::for_device_family(device)` from
  production (7 sites: `tape_forward.rs:445,1414`; `trainer.rs:384`;
  `forward.rs:6178,6192,6198,7344`). The original plan already says this helper is
  "only a compatibility wrapper for tests" — make that true by routing production
  through `runtime_training_precision_policy`.
- **W6.2** Convert `tape_forward.rs` not-CPU guards (`matches!(x.device(), Cuda |
  Metal | Vulkan | Rocm)`) to a capability query.
- **W6.3** Step proofs: add the plan's loss-decrease assertion; route the CUDA proof
  through the `OptimizerBackend` seam (not the direct kernel,
  `cuda_sft_step_proof.rs:117-151`); make proofs exercise the trainer's real route
  selection. Fix the stale error text at `trainer.rs:7843`.

## W7 — Phase 7 residuals (C completeness)

- **W7.1** Confirm BLASLt request conversion is fully shared: the dedup lives in
  `kiln-tensor/src/blaslt_request.rs` (not `cuda_rocm_common.rs`); either document
  that as the chosen home or relocate per the original plan line 901. No behavior
  change.
- **W7.2** (Falls out of W1) The Metal/Vulkan `BackendRuntime` impls in
  `metal_runtime.rs` (888 lines) / `vulkan.rs` split naturally into focused-trait
  impls as W1 proceeds.

## W8 — Phase 8 residuals (real conformance)

- **W8.1** (Folds into W0) Conformance gate statuses derive from real signals.
- **W8.2** Add CPU-backend conformance coverage so the local gate exercises *behavior*
  (storage roundtrip, DeviceOp parity, optimizer parity, the dispatch seams from
  W1/W4/W5) rather than only source-string presence. Real device parity tests
  (`rocm_matmul_parity`, `vk_matmul_parity`, the SFT proofs) stay hardware-gated.
- **W8.3** (Folds into W0.4) Latency-gate spoofing closed via host-captured
  provenance, verified in the local gate (no GitHub-Actions dependency).

## Quick-win defects (independent, do anytime)

- Stale dropdown (`perf-regression-nightly.yml:75`) — low priority; fix only if that
  GitHub workflow is kept. We are de-emphasizing GitHub Actions, so prefer dispatching
  fixtures from the bench-host agents instead.
- `tensor.rs:423` stale "Vulkan not yet implemented" doc (Phase-0 deliverable).
- Vulkan double-upload of MLP weights (`vulkan_weights.rs:255-274`).
- `scatter_gdn_recurrent_resident_batch_rows` partial mutation on error
  (`vulkan.rs:781-804`).

## Sequencing

```
W0 (honest scoreboard)  ──►  everything is measured against it
   │
   ├─ Quick wins (anytime)
   │
W1 (focused traits authoritative; shrink BackendRuntime)
   │   └─ prerequisite for clean W3/W4/W5/W6
   ├─ W4.1 (extend MatmulRequest)  ─┐
   ├─ W5.0 (fix replay contract)   ─┤ contract fixes precede wiring
   ▼                                │
W3 ── W4 ── W5  (route the three remaining families through their contracts)
   │
W6, W2 (residuals, parallel)
   │
W7, W8 (completeness; W8 largely folds into W0/W4/W5)
   ▼
Exit: regenerate report — every `genuine` flag computes true.
```

Critical path: `W0 → W1 → {W4, W5} → W8 → exit`. W3 and W6 run parallel to W4/W5
after W1.

## Exit criteria — "everything is A"

The effort is complete when:

1. `scripts/generate_backend_capability_report.py` emits `genuine: true` for Phases
   1–6 (computed, not declared), and all per-family `migration` signals pass.
2. `BackendRuntime` is below the method-count gate; no `impl<T: BackendRuntime>
   XBackend for T` forwarding shim remains.
3. `ops/matmul.rs`, the residency call sites, and the decode replay path contain no
   `Device::`-keyed dispatch selection (only the sanctioned construction/plumbing
   matches), enforced by regression-guard tests.
4. `for_device_family` is gone from production; training selects via
   `runtime_training_precision_policy`.
5. A CPU behavioral test (runnable in the local gate, no GPU) exists for each
   migrated dispatch seam; real-hardware parity tests pass on available hardware.
6. The latency gate rejects a fabricated/host-mismatched artifact;
   `scripts/check_unification_gates.sh` (the runner-agnostic gate, not GitHub
   Actions) runs `--check` and is the documented pre-push step; the report
   has no hardcoded `covered` literals for migration-bearing phases.
7. `cargo check` passes on the portable path and on each `--features <backend> --lib`.

At that point a model/training call site asks only the questions the original plan's
"Expected End State" listed — *"can the active backend execute this request
natively? is host fallback allowed? is it replay-safe? what resident resources does
it touch? what precision policy applies?"* — and never *"is this CUDA?"*. One engine
contract, four native execution models.
