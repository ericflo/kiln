# Backend And Engine Unification — Outcome & Code Review

Date: 2026-06-07
Branch reviewed: `unify-engines` (GitHub PR #1465, base `main`)
Scope: +53,791 / −24,454 across 116 files, 224 commits.
Companion plan: [`backend-engine-unification-plan.md`](backend-engine-unification-plan.md) (the original plan)
Follow-up plan: [`backend-engine-unification-completion-plan.md`](backend-engine-unification-completion-plan.md) (how to close everything below)

This document records the state of the unification work as built, measured against
the original plan. It is the evidence base for the completion plan. Findings were
produced by a 42-agent adversarial review (one mapper per plan phase + per backlog
item + a self-certification-integrity pass; one skeptical code reviewer per
abstraction; every critical/high finding independently verified against the code —
14 confirmed, 3 refuted, 2 uncertain). Refuted claims were dropped.

## How to read this: the A / B / C rubric

Every piece of the work is classified by *what kind of done* it is:

- **A — Genuine unification.** Call sites actually select behavior through the shared
  contract (capability / route / request object). Backend-identity branching and
  duplication are removed. The monolith shrinks. This is the plan's intended end
  state: *"the training loop should not know whether the backend is CUDA, ROCm,
  Metal, or Vulkan except through capabilities and policy objects."*
- **B — Additive scaffold.** A typed contract / capability-query / forwarding facade
  exists, is well-typed and tested, but **coexists with** the original dispatch,
  which still branches on `Device::` underneath. No call site uses the new surface
  for a real decision. This is what "scaffold" / "blanket-forwarding … for zero
  behavior change" / "without changing dispatch behavior" mean in the PR body.
- **C — Complete mechanical work.** Module decomposition or report generation that is
  fully and correctly done, behavior-preserving. There is no "A" beyond it.

## Bottom line

The branch completed the **contract + decomposition + observability** half of the
plan to a high standard, and left the **behavioral-migration** half largely undone
for four of the seven structural phases. The generated capability report marks all
eight phases `covered`, but `covered` is computed mostly from **hardcoded Python
string literals gated on file existence** (21 such literals in the generator) — so
it certifies *"a contract and a test exist"*, not *"call sites were migrated"*. The
two are conflated.

The single clearest illustration: Phase 5 ("make replay a first-class engine
concept") is reported `covered`, yet the four production replay runners
(`cuda_graph.rs`, `rocm_graph.rs`, `vk_decode_resident.rs`,
`kiln-vulkan-kernel/src/cmd_batch.rs`) have **zero diff** versus `main` and never
reference the new `kiln_graph::ReplayPlan`. It is a well-built vocabulary with no
speakers.

The portable (no-GPU-features) path compiles clean at HEAD
(`cargo check -p kiln-model -p kiln-train -p kiln-tensor -p kiln-graph` → exit 0).
GPU paths were validated per-crate (`cargo check --features cuda/rocm --lib`, etc.)
as recorded in the PR; full link/run requires hardware. As of commit `26d897b6`
(2026-06-07) the hardware-latency gate flipped to `covered` when RTX 4090 CUDA
results landed, so all five fixtures are now `locked_threshold`.

## Scorecard

| Phase | Report | Review verdict | Type |
|---|---|---|---|
| 0 — Capability reporting | covered | **Substantially landed** — real source introspection; mismatch test parses method bodies and is non-vacuous | C / A |
| 1 — Focused traits | covered | **Scaffold only** — 12 blanket forwards over the intact 95-method `BackendRuntime`; no call site migrated; trait *grew* | B |
| 2 — Fallback policy | covered | **Substantially landed (the hard part)** — decode + training `NativeRequired` guards genuinely error before fallback | A + B |
| 3 — Resident resources | covered | **Partially landed** — `ResidentRegistry` consumed only by a `cfg(test)` mock; lifecycle states never transitioned | B |
| 4 — Matmul/linear | covered | **Scaffold only** — unified request reaches no backend; `ops/matmul.rs` still branches `Device::`; predicate re-adds `match name()` | B |
| 5 — Replay layer | covered | **Scaffold only** — production runners unchanged; `ReplayPlan` used only in tests | B |
| 6 — Training | covered | **Substantially landed (strongest)** — loss/precision routed on enums, not `Device::`; residual `for_device_family` remains | A |
| 7 — Decompose modules | covered | **Substantially landed / complete** — `metal.rs` 16,015→98-line facade; `cuda_rocm_common` dedup verified identical | C |
| 8 — Conformance/perf | covered | **Substantially landed (mixed)** — tooling complete; conformance is mostly source-string assertions; latency gate spoofable | C + B |

## What genuinely landed (A / C)

- **Phase 7 — Module decomposition (C, exemplary).** `backend/metal.rs` went from a
  16,015-line monolith (`main`) to a **98-line facade** plus 17 `metal_*.rs`
  modules; bodies moved verbatim (byte-identical spot checks), public surface
  preserved via re-exports, visibility tightened correctly. Vulkan similarly
  (6,123 → 1,295-line facade + 13 modules). `cuda_rocm_common.rs` factors only
  *genuinely identical* code (verified bit-identical against the pre-refactor bodies
  across the five extraction commits). This is pure downside-risk work, done well.
- **Phase 2 — Hot-path guards (A, the most valuable behavior change).** Decode
  (`generate.rs`) and training-optimizer (`trainer.rs:6743-6958`) now hard-error on
  a missing native kernel instead of silently staging through CPU, routed through
  the typed `FallbackPolicy` + capability query, with env opt-ins. This directly
  attacks the "implicit performance cliffs" the plan was written to remove.
- **Phase 6 — Training loss/precision routing (A, strongest unification).**
  `trainer.rs` selects SFT/GRPO/OPD loss roots and optimizer dispatch via route
  enums each backend advertises (`runtime_sft_flce_loss_route`,
  `runtime_grpo_loss_route`, `runtime_opd_loss_route`, etc.) — never a `Device::`
  match for the loss/optimizer decision. This is the proof the pattern works and can
  be carried to Phases 1/3/4/5.
- **Phase 0 — Capability report generator (C/A).** Feature fanout, override counts,
  support-method states, and env gates are introspected from source (verified
  byte-identical on regen; `--check` exits 0). The supports_*/decline mismatch test
  genuinely parses brace-matched method bodies and is non-vacuous (an injected
  `supports→true / body→Ok(None)` trips it).

## What is scaffold (B): "covered" ≠ "unified"

### Phase 1 — Focused traits forward to an intact monolith
- All 12 focused traits are blanket impls `impl<T: BackendRuntime + ?Sized> XBackend
  for T` that forward verbatim to the same `BackendRuntime` vtable methods
  (`mod.rs:2604-3643`). Forwarding is mechanically correct (every `runtime_*`
  targets the right method, args in order — verified).
- `BackendRuntime` did **not** shrink: ~95 trait methods, intact (`mod.rs:655-1902`);
  it grew vs `main`. Migration steps 4–5 of the plan (move call sites, then
  delete/shrink the all-purpose trait) were not started.
- `for_device_kt` (`mod.rs:3668-3695`) still matches `Device::Cuda/Rocm/Metal` (+ a
  Vulkan runtime probe) and returns `Arc<dyn BackendRuntime>`. The focused traits
  sit above this unchanged dispatch.

### Phase 3 — ResidentRegistry is a parallel contract nobody calls
- The blanket `impl<T> ResidentRegistry for T` (`mod.rs:3458-3516`) is real but its
  only call sites are in `#[cfg(test)]` (`mod.rs:3700+`, a fake probe backend).
  Production register/update/evict/resolve do **not** route through it (confirmed).
- Routing is inverted vs the plan ("route existing APIs *through* the registry"):
  the adapter forwards *downward* into `ResidencyBackend::runtime_*` →
  `BackendRuntime`.
- The unified lifecycle (`Unregistered/RegisteredClean/DirtyHost/DirtyDevice`) is
  inert: `from_tensor_for_backend` hardcodes `state = RegisteredClean`
  (`residency.rs:140`) and `replay_stability` at construction; no backend ever
  transitions state.
- **Confirmed defect:** the Vulkan resident-activation registry is a process-global
  `static OnceLock<Mutex<HashMap<TensorId, Arc<VulkanBuffer>>>>`
  (`vulkan_residency.rs:10-20`) never drained on backend `Drop` — VRAM-leak / cross-
  instance-bleed risk. CUDA/ROCm/Metal membership registries are also process-global
  statics.

### Phase 4 — Unified matmul/linear request reaches no backend
- Real dense matmul still branches on device: `ops/matmul.rs:294-338` and `396-438`
  each have four per-backend arms (`matches!(a.device(), Device::Cuda|Rocm|Metal|
  Vulkan)`); `forward.rs:5941-5947` selects `cuda_matmul`/`rocm_matmul` by device.
- The engine `MatmulRequest` / `MatmulBlasRequest` / `LinearRequest` types and
  `supports_*_request` predicates never reach any backend dispatch crate (grep
  across kiln-tensor/kiln-blas/kiln-rocblas/kiln-vulkan-kernel finds only tests).
- `supports_matmul_request` *re-introduces* backend-identity branching:
  `capability.rs:2220-2240` is a hardcoded `match self.name() { "cuda"|"rocm" => …,
  "metal" => …, "vulkan" => … }` lookup table, not a per-backend trait answer.
- The descriptor is also too narrow to migrate onto without capability loss: it
  models only RowMajor rank-2-bias (`capability.rs:238-291`), returns `Unsupported`
  for any non-RowMajor (`:2207`), and cannot represent transposed (ColMajor) or
  mixed-dtype matmul. `LinearRequest` models only sampling kinds, not linear.

### Phase 5 — Replay contract has no production speakers
- The four production runners are untouched (`cuda_graph.rs`, `rocm_graph.rs`,
  `vk_decode_resident.rs`, `cmd_batch.rs` = **zero diff** vs `main`;
  `metal_graph.rs` only +31/−19). None reference `kiln_graph::ReplayPlan`.
- `CapturedGraphReplayPlan` / `ReplayPlan::replay` are exercised only in
  `#[cfg(test)]`. `ReplayBackend` is a query-only blanket-forward facade.
- The new "replay parity" tests wrap a counted no-op scaffold graph
  (`CudaCapturedGraph::replay` does `fetch_add` then `Ok(())`), not a real
  eager-vs-replay comparison.
- **Latent contract bugs that become live the moment this is wired** (must fix
  first): `ResidentResourceRef::from_tensor` computes `byte_len` via
  `dtype().size_in_bytes()`, which returns **0** for `Int4Packed`/`Fp4Packed`
  (`replay_plan.rs:82`, `dtype.rs:108`); `StableWithinStep` is a dead variant in
  validation (`replay_plan.rs:50-89`); `ResidentResourceRef` drops all
  stride/offset/contiguity layout; `validate_inputs` cannot detect a key change
  (`KeyChanged` unreachable, `replay_plan.rs:287`).

## The self-certification integrity problem

`docs/backend-capability-report.{md,json}` is generated by
`scripts/generate_backend_capability_report.py` and validated by
`crates/kiln-model/tests/backend_capability_contract.rs`. The mechanism is partly
circular and partly hand-authored:

- The **Migration Phase Status** table and 12 of 13 conformance gates are hardcoded
  `"status": "covered"` Python literals (21 occurrences), gated only on a listed
  evidence *file existing* (`generate_backend_capability_report.py:944-1200,
  1236-1472`). Only `hardware_latency_thresholds` is computed.
- The contract test then reads the generated JSON and asserts it matches the source
  it was generated from. So a phase reads `covered` by adding a scaffold + a test
  that checks the scaffold is present — with zero dispatch change.
- The freshness `--check` is enforced **only** by a Rust unit test and is wired into
  **no** `.github/workflows/*.yml`. If that test is skipped/gated behind GPU
  features, report drift ships silently.
- The fallback/decode/training-policy and conformance-command sections are static
  Python dicts that can drift from code without failing `--check` (both sides come
  from the same literals).

Net: the report's feature/override/support/env-gate/mismatch sections are
source-truth; its phase-status/conformance/policy sections are author-asserted.
"8/8 covered" overstates genuine unification.

## Confirmed defects worth fixing (ranked)

> Remediation note: per the project's tooling stance (de-emphasize GitHub Actions /
> hosted runners; keep issue + branch coordination), the GitHub-flavored items below
> (#1 stale dropdown, the `pull_request.paths` gap in #3) are fixed by moving
> enforcement into a runner-agnostic local gate rather than by hardening the GitHub
> workflow. See the completion plan's Tooling stance and W0.3/W0.4.

1. **(High, operational) Stale CI dispatch dropdown.**
   `.github/workflows/perf-regression-nightly.yml:75` still offers
   `vulkan_rtx6000_decode_microbench`, renamed to
   `vulkan_strix_halo_decode_microbench` in commit `2ca5579d`. The Vulkan fixture is
   currently **undispatchable** via the workflow.
2. **(High, governance) `covered` is a hardcoded literal + file-existence check.**
   `generate_backend_capability_report.py:944` (and the phase table). The scoreboard
   cannot distinguish scaffold from migration.
3. **(Medium, security/trust) Latency gate is spoofable.** `check_backend_latency_
   fixtures.py --require-covered` validates only self-declared provenance fields; a
   hand-fabricated raw log + manifest passes (reproduced end-to-end). `tracked_git_
   dirty` uses `--untracked-files=no`, so brand-new fabricated evidence reads clean.
   `pull_request.paths` omits `bench-results/backend-latency/**`, so a PR
   editing/deleting a committed result won't trip the gate.
4. **(Medium) Unified `MatmulRequest` cannot model transposed/mixed-dtype** matmul
   (`capability.rs:238-291`) — blocks a lossless Phase-4 migration until extended.
5. **(Medium) `AlgoCacheStats` counted but never surfaced** anywhere
   (`cublaslt_handle.rs:372-538`) — Phase 8 "matmul cache reporting" is inert.
6. **(Low) `byte_len = 0` for packed dtypes** (`replay_plan.rs:82`) — inert today,
   live trap on Phase-5 wiring.
7. **(Low) Vulkan resident registry not drained on `Drop`** (`vulkan_residency.rs:10`).
8. **(Low) Dead contract surface never constructed for real backends:**
   `Support::RequiresFeature` (`capability.rs:25`), `FallbackPolicy::ErrorInHotPath`
   (only a `_ =>` default arm), `InvalidateReason::Backend`,
   `CaptureError::DanglingPointer`.
9. **(Low) Phase-0 deliverable missed:** `tensor.rs:423` still documents "Vulkan is
   not yet implemented" though the body routes `host_to_vulkan_copy`.
10. **(Low) Vulkan double-uploads MLP weights** into both bf16-packed and f32 caches
    (`vulkan_weights.rs:255-274`); `scatter_gdn_recurrent_resident_batch_rows` leaves
    partial mutation on error (`vulkan.rs:781-804`).

## Refuted claims (recorded so they are not re-raised)

- *"Phase 1 facade prevents backends from implementing focused traits directly."*
  Refuted — backends *can*; the blanket impl coexists, it does not block per-backend
  impls.
- *"Two `ResidentResourceRef` builders disagree on `byte_len`, causing cross-builder
  mismatch."* Refuted — both builders are currently inert (no production caller).
- *"The decline-mismatch test went vacuous for decomposed Metal/Vulkan modules."*
  Refuted — a parallel enforced guard still covers the supports_*/decline contract.

## Appendix: full finding inventory

The complete per-dimension finding list (all severities, with verification verdicts
CONFIRMED/REFUTED/UNCERTAIN and file:line) is preserved in the review run artifact.
The confirmed high/critical items are all reflected in the ranked defects and the
per-phase sections above; the medium/low items feed the residual workstreams in the
completion plan (W2/W6/W7/W8 and the quick-win list).
