# Verification policy

Kiln tests behavior, public contracts, and typed metadata. Tests must not infer
correctness from implementation source text.

## Start here

Choose evidence by the claim you are making:

| Claim | Minimum acceptable evidence |
| --- | --- |
| A value parses, validates, or serializes correctly | A behavioral test through the owning typed API |
| A type or trait boundary prevents an invalid state | A compile-time assertion or compile-fail test |
| A generator and its committed artifact agree | Validation of the canonical structured artifact plus generator `--check` |
| A request succeeds, fails, cancels, or cleans up correctly | An injected lifecycle test at the owning public boundary |
| Concurrent or stateful behavior preserves an invariant | A property, state-machine, stress, or failure-containment test |
| An accelerator behaves correctly | A non-skipped, source-bound local qualification receipt from the requested real device |
| Performance or memory changed | Comparable before/after receipts with identical claim-defining inputs |
| Documentation renders correctly | Source/contract validation, the documentation build, browser smoke, and visual review |

Use the narrowest evidence that can actually falsify the claim. Add broader
evidence when the behavior crosses ownership, process, device, or persistence
boundaries.

## Prohibited evidence

The following do not establish a correctness claim:

- Searching Rust, C++, CUDA, HIP, shader, Python, JavaScript, or documentation
  source for a function name, call order, spelling, or missing string.
- Counting source occurrences of a synchronization, allocation, fallback, or
  backend-dispatch call.
- Treating a compile-only check, mock, skipped accelerator test, or generated
  prose fragment as runtime backend evidence.
- Treating a source-path list in a report as proof that the cited test ran.
- Editing a receipt, generated report, or snapshot until its text matches the
  expected result.

Source text describes one spelling of an implementation. It does not show that
the code compiled, executed, took the intended branch, handled failure, or
preserved an invariant.

`python3 scripts/check_source_parsing_tests.py` rejects in-scope Rust and
qualification tests that reintroduce source-text verification. The checked-in
contract has zero allowed tests, reads, and text assertions. The scanner
follows module `include_str!` bindings, helper-call chains, workspace-relative
reader helpers, dynamic implementation paths, and recursive Rust-source
walkers.

A zero scanner result proves only that the scanner found no in-scope pattern.
It does not prove that every remaining test is useful or that out-of-scope
languages contain no similar test.

## Evidence ownership

| Invariant family | Authoritative evidence |
| --- | --- |
| Backend capability and dispatch shape | Rust trait and type checking; typed `Backend*` descriptors; `docs/backend-capability-report.json`; the report generator’s self-test and check mode; backend request and parity tests |
| Tape scope and exact gradients | `crates/kiln-model/tests/tape_forward_parity.rs`; `crates/kiln-model/tests/vk_tape_record_proof.rs`; backend SFT-step proofs; exact gradient-set validation in trainer tests |
| Frozen GDN and RMSNorm weights | Compile-time Rust/FFI signatures; `crates/kiln-gdn-kernel/tests/rocm_gdn_parity.rs`; activation-only GDN tape tests |
| Replay and graph lifetime | `kiln_graph::ReplayPlan` state tests; `crates/kiln-graph/tests/capture_lifetime.rs`; backend eager/replay parity; source-bound graph correctness, resilience, and failure-containment qualification |
| ROCm stream and pointer lifetime | `kiln-hip` execution-gate, submission, quarantine, and settlement tests; ROCm capture-arena tests; source-bound synchronization and failure-containment qualification |
| Serving ownership, cancellation, and mutation | Injected lifecycle tests at the owning API; adapter-mutation, mixed-load, pressure, graph, and soak qualification |
| Runtime configuration and environment | Typed parse and validation tests; exact runtime-default and environment inventories; startup and API behavior tests |
| Generated reports and documentation | Canonical JSON or schema validation; generator `--check`; documentation build and rendered-site smoke; visual review for meaning and layout |

The table identifies owners, not a fixed list of filenames for every future
change. When ownership moves, update the implementation, tests, structured
report, and this policy together.

## Hardware evidence boundary

Accelerator behavior is accepted only from a non-skipped, source-bound local
qualification receipt on the named device. Portable tests can reject a
change; they cannot promote an unexecuted backend claim.

A valid hardware result must preserve:

- the exact source-tree identity;
- model, tokenizer, workload, and comparison-policy identity;
- the requested backend and captured device identity;
- all required cases, including failures and first-nonfit boundaries;
- raw-artifact hashes when strict local validation is requested; and
- the runner’s verdict without hand-edited overrides.

The device identity is evidence provenance, not dispatch policy. Workloads and
runtime defaults must remain capability-driven unless a contract explicitly
requires a capability or resource bound.

## Production file budget

`contracts/production-file-budget-v1.json` sets a 5,000-physical-line ceiling
for Rust, JavaScript, and CSS production files under `crates/**/src`. Extracted
`tests` child trees are excluded because they do not ship in the product.
`python3 scripts/check_production_file_budget.py` enforces the contract in the
repository-hygiene workflow.

An oversized file is allowed only through a sorted, path-specific exception
with an exact non-growth ceiling and a concrete ownership rationale. The
checker rejects:

- a new unlisted oversized file;
- growth beyond an exception’s exact ceiling;
- a stale exception for a missing or out-of-scope file; and
- headroom after an excepted file shrinks.

When an excepted file shrinks, lower its ceiling in the same change. When it
falls below the default limit, remove the exception. A split must reduce the
policy surface instead of turning an old size into a permanent allowance.

## Exceptions and test replacement

There is no waiver that turns source-text parsing into correctness evidence. If
a scanner false positive appears, narrow the scanner with a regression test;
do not add a debt allowance.

Replace a prohibited test according to the behavior it was trying to protect:

| Intended protection | Replacement |
| --- | --- |
| An API calls a dependency with the right values | Inject the dependency and assert the observed call |
| A capability is present or absent | Query the typed capability descriptor |
| A state transition occurs in order | Assert emitted events or state-machine transitions |
| A generated field exists | Parse the canonical artifact and assert the typed field |
| A failure cleans up resources | Inject the failure and assert cleanup and recovery |
| An accelerator path executes | Run a required hardware case and retain its receipt |

Tests for a text renderer may assert rendered text at that renderer’s public
boundary. They must not read unrelated implementation source and present its
spelling as behavior.

## Migration record

The zero baseline replaced or deleted 112 source-text tests. Most were
concentrated in `backend_capability_contract.rs`,
`resource_concurrency_invariants.rs`, and `tape_scope_routing_contract.rs`.
Behavioral, type-level, structured-metadata, and hardware-qualification suites
remain. The deleted tests were not retained as a second layer because their
source spellings drifted while proving neither execution nor failure behavior.

## Run the gates

Run the portable source-parsing and capability-report gates with:

```bash
python3 scripts/check_source_parsing_tests.py
python3 -m unittest scripts.qualification.tests.test_source_parsing_tests
python3 scripts/generate_backend_capability_report.py --self-test
python3 scripts/generate_backend_capability_report.py --check
python3 scripts/check_production_file_budget.py
```

Run accelerator suites through `scripts/qualification/run.py` and their
checked-in workload manifests. Do not invoke an accelerator test directly and
interpret a skip as a pass.

When a gate fails, fix the owning behavior or canonical artifact. Do not weaken
the claim, delete a required case, bless generated drift, or raise a numeric
ceiling merely to make the check green.
