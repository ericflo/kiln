# Verification Policy

Kiln tests behavior, public contracts, and typed metadata. Tests must not infer
correctness from implementation source text.

## Prohibited evidence

The following do not establish a correctness claim:

- Searching Rust, C++, CUDA, HIP, shader, Python, JavaScript, or documentation
  source for a function name, call order, spelling, or absent string.
- Counting source occurrences of a synchronization, allocation, fallback, or
  backend dispatch call.
- Treating a compile-only check, mock, skipped accelerator test, or generated
  prose fragment as runtime backend evidence.
- Treating a source-path listing in a report as proof that the cited test ran.

`python3 scripts/check_source_parsing_tests.py` scans Rust and qualification
tests and rejects any reintroduction of source-text verification. Its checked-in
contract has zero allowed tests, reads, and text assertions. The scanner follows
module `include_str!` bindings, helper-call chains, workspace-relative reader
helpers, dynamic implementation paths, and recursive Rust-source walkers.

## Evidence ownership

| Invariant family | Authoritative evidence |
| --- | --- |
| Backend capability and dispatch shape | Rust trait/type checking; typed `Backend*` descriptors; structured `docs/backend-capability-report.json`; `scripts/generate_backend_capability_report.py --self-test --check`; backend request and parity tests. |
| Tape scope and exact gradients | `crates/kiln-model/tests/tape_forward_parity.rs`; `crates/kiln-model/tests/vk_tape_record_proof.rs`; backend SFT step proofs; exact gradient-set validation in trainer tests. |
| Frozen GDN/RMSNorm weights | Compile-time Rust/FFI signatures; `crates/kiln-gdn-kernel/tests/rocm_gdn_parity.rs`; the activation-only GDN tape tests. |
| Replay and graph lifetime | `kiln_graph::ReplayPlan` state tests; `crates/kiln-graph/tests/capture_lifetime.rs`; backend eager/replay parity; source-bound graph correctness, resilience, and failure-containment qualification. |
| ROCm stream and pointer lifetime | `kiln-hip` execution-gate, submission, quarantine, and settlement tests; ROCm capture-arena tests; source-bound synchronization A/B and failure-containment qualification. |
| Serving ownership, cancellation, and mutation | Injected lifecycle/unit tests at the owning API; public adapter mutation qualification; mixed-load, pressure, graph, and soak qualification receipts. |
| Runtime configuration and environment | Typed configuration parse/validation tests; exact runtime-default and environment inventories; startup/API behavior tests. |
| Generated reports and documentation | Canonical JSON/schema validation; generator `--check` mode; documentation builder and rendered-site smoke tests. Rendered text is tested only at the renderer boundary. |

Accelerator behavior is accepted only from a non-skipped, source-bound local
qualification receipt on the named device. Portable tests can reject a change;
they cannot promote an unexecuted backend claim.

## Production file budget

`contracts/production-file-budget-v1.json` sets a 5,000-physical-line ceiling
for Rust, JavaScript, and CSS production files under `crates/**/src`. Extracted
`tests` child trees are excluded because they do not ship in the product.
`python3 scripts/check_production_file_budget.py` enforces the contract in the
cheap repository-hygiene workflow.

An oversized file is allowed only as a sorted, path-specific exception with an
exact non-growth ceiling and a concrete ownership rationale. New unlisted
oversized files fail. Existing exceptions fail when they grow past their cap,
disappear, leave production scope, or shrink under the default without having
their stale exception removed. A split therefore reduces the policy surface;
it does not silently convert the old size into a permanent allowance.

## Migration record

The zero baseline replaced or deleted 112 source-text tests. Most were
concentrated in `backend_capability_contract.rs`,
`resource_concurrency_invariants.rs`, and `tape_scope_routing_contract.rs`.
Behavioral, type-level, structured-metadata, and hardware-qualification suites
remain. The deleted tests were not retained as a second layer because their
source spellings drifted frequently while proving neither call execution nor
failure behavior.

Run the portable gate with:

```bash
python3 scripts/check_source_parsing_tests.py
python3 -m unittest scripts.qualification.tests.test_source_parsing_tests
python3 scripts/generate_backend_capability_report.py --self-test
python3 scripts/generate_backend_capability_report.py --check
```

Run accelerator suites through `scripts/qualification/run.py` and their
checked-in workload manifests. Do not invoke an accelerator test directly and
interpret a skip as a pass.
