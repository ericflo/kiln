# Contributing to Kiln

Kiln welcomes bug fixes, performance work, backend and kernel improvements,
tests, documentation, examples, and developer-experience changes.

Kiln is deliberately focused: it is a single-process Rust serving and training
runtime for Qwen3.5-4B. Its accelerator paths cover CUDA, ROCm, Vulkan, and
Metal. A contribution should deepen that implementation instead of turning it
into a general model framework or adding a sidecar runtime.

## Start here

| Your change | Read first | Evidence expected |
| --- | --- | --- |
| Runtime, scheduler, memory, or backend behavior | [`ARCHITECTURE.md`](ARCHITECTURE.md) | Focused behavioral tests and affected contract checks |
| Public API or configuration | [HTTP API contract](contracts/kiln-http-api-v1.openapi.json) or [configuration](docs/public/CONFIGURATION.md) | Schema, implementation, examples, and docs updated together |
| Training or evaluation | Relevant guide under `docs/` | Request, lifecycle, receipt, and failure-path tests |
| Performance | [`BENCHMARKS.md`](BENCHMARKS.md), [`PROFILING.md`](docs/archive/profiling/PROFILING.md), and [local qualification](docs/policies/qualification.md) | Comparable before/after evidence on the same declared workload |
| Documentation site | [`docs/plans/public-site-audit-and-copyediting-plan.md`](docs/plans/public-site-audit-and-copyediting-plan.md) | Docs build, smoke test, links, and desktop/mobile review |

Open an issue before a non-trivial change, new dependency, new kernel, new
public surface, or model-scope proposal. Describe the problem, the intended
boundary, and how you will prove the result. Search open and closed issues and
pull requests first; prior experiments may already contain measurements or a
rejected design.

## Non-negotiable design boundaries

- **Capability-driven accelerators.** Runtime selection must use available
  backend capabilities and validated contracts. Marketing names, laptop
  models, device IDs, and one machine’s memory size belong in evidence—not
  product dispatch or defaults.
- **One model family.** Support for another architecture requires prior design
  agreement because scheduling, memory, kernels, tokenization, and training are
  specialized for Qwen3.5-4B.
- **One process and one model allocation.** Do not add a Python sidecar or a
  second model copy as an implementation shortcut.
- **Fail closed.** Missing capabilities, inconsistent identity, unsafe memory
  bounds, and malformed evidence must produce explicit errors rather than
  silently selecting an unproved path.
- **Claims follow evidence.** A compile-only backend job is not hardware
  execution, a microbenchmark is not end-to-end throughput, and one device’s
  receipt is not support policy for every device.

## Development setup

Rustup automatically reads the pinned toolchain in `rust-toolchain.toml`.

```bash
rustc --version
rustup show active-toolchain
cargo fmt --version
```

Do not override the repository toolchain with a floating toolchain. A toolchain
upgrade is an explicit change to `rust-toolchain.toml`.

On Linux, build the default no-accelerator configuration with the bounded
wrapper:

```bash
scripts/cargo-bounded.sh build --locked
```

The wrapper serializes compilation, applies a no-swap memory boundary, and
keeps the compiler and linker process tree together. Direct `cargo` remains
appropriate when the operator has already isolated resources.

Build one accelerator feature only on a host with its required toolchain:

```bash
# NVIDIA CUDA
cargo build --locked --features cuda

# Vulkan loader, driver, and shader toolchain
cargo build --locked --features vulkan

# Apple Metal
cargo build --locked --features metal

# ROCm/HIP
ROCM_PATH=/opt/rocm \
  cargo build --locked --no-default-features --features rocm
```

`KILN_CUDA_ARCHS` and `KILN_ROCM_ARCHS` are optional build controls for
explicit toolchain targets. If you use them, record the exact values in build
or qualification evidence. Never turn a local architecture value into a
runtime device allowlist.

## Run the right tests

Start with the narrowest test that exercises your change, then widen the
verification surface.

```bash
# One package
scripts/cargo-bounded.sh test --locked -p kiln-core

# Default workspace members
scripts/cargo-bounded.sh test --locked
```

`test_health_with_real_backend` is ignored because it requires a real model and
backend. Do not convert an unavailable accelerator test into a passing skip
when the claim requires hardware; run the matching local qualification
workload and retain its receipt.

On Apple Silicon, serialize Metal feature tests because concurrent
`MetalDevice::new` calls still hit an upstream device-construction race:

```bash
cargo test --locked --features metal -- --test-threads=1
```

`cargo nextest run --locked` is an optional throughput-oriented local runner.
Do not run it concurrently with another build or accelerator workload, and do
not treat it as the repository’s bounded verification path.

Before pushing Rust changes:

```bash
cargo fmt --all --check
scripts/cargo-bounded.sh build --locked
scripts/cargo-bounded.sh test --locked
```

Run dependency policy when `Cargo.toml` or `Cargo.lock` changes:

```bash
cargo deny check --all-features
```

Install the pinned-compatible `cargo-deny` tool if it is not already available.
The policy in `deny.toml` checks licenses, bans, sources, and advisories.

## Contract and qualification checks

Run the checker owned by every contract you edit. Common examples:

```bash
python3 scripts/check_runtime_env_contract.py
python3 scripts/check_source_parsing_tests.py
python3 scripts/check_repository_artifacts.py
python3 scripts/qualification/workload.py qualification/workloads/*.json
find qualification/receipts -type f -name '*.json' -print0 \
  | sort -z \
  | xargs -0 python3 scripts/qualification/receipt.py
```

Use the explicit checker named by a guide or test suite when its contract is
not in this representative list.

Tests that read implementation source and assert substrings are migration debt,
not correctness evidence. Replace them with compile-time constraints, runtime
behavior, property/state-machine tests, or typed contract assertions. The
[source-parsing test debt](docs/policies/VERIFICATION_TEST_INVENTORY.md) page explains
the zero-debt gate.

## Documentation changes

Update the source document, schema, example, and navigation label together.
Generated pages must be changed through their generator or machine-readable
contract.

Install the pinned docs dependencies once:

```bash
npm ci --prefix scripts/docs-site
```

Then run:

```bash
npm test --prefix scripts/docs-site
node scripts/docs-site/build.mjs --out /tmp/kiln-docs-site
KILN_DOCS_SITE_ROOT=/tmp/kiln-docs-site \
KILN_DOCS_REQUIRE_GENERATED=true \
  node scripts/check_docs_site_smoke.mjs
```

Inspect every changed route at desktop and mobile widths. A successful build
does not catch unreadable tables, misleading hierarchy, stale screenshots, or
copy that is technically true but impossible to follow.

## Performance changes

Begin with a hypothesis and profile the affected path. Name the kernel, route,
or scheduling region you changed and cite the relevant profile evidence.

Before and after measurements must keep model, tokenizer, prompts, sampling,
output length, concurrency, repetitions, configuration, backend, and device
identity comparable. Use the same committed workload and comparison policy.
Report correctness, throughput, latency, memory, failures, and the raw-evidence
hash—not only the metric that improved.

Use whatever supported device is available for development. The receipt should
identify that machine precisely, while the implementation remains
capability-driven. If a change is intended to help Vulkan generally, explain
the required Vulkan capabilities and test more than one implementation when
available; do not branch on the device that happened to produce the first
measurement.

An isolated kernel microbenchmark can localize a change, but it does not replace
an end-to-end serving receipt when the pull request makes a serving claim.
Failed or first-nonfit rows are counterevidence and must remain visible.

## Adding an eval scorer

`kiln_eval::scorers::Scorer` is a serde-tagged enum, not a trait. To add a
scorer:

1. Add the snake-case serde variant and its `kind_label()` arm.
2. Implement scoring in `crates/kiln-eval/src/scorers/<name>.rs`.
3. Dispatch the variant from `score_completion`.
4. Extend `auto_detect_scorer` only when the target shape is unambiguous.
5. Add focused unit and end-to-end eval lifecycle tests.
6. Document the request and result behavior in `docs/guides/EVAL_GUIDE.md` and the
   public eval page.

Keep schema, CLI, UI, synthesis, and server behavior aligned with the enum’s
single source of truth.

## Code and error style

- Run `cargo fmt --all`; do not hand-format around the pinned formatter.
- Justify every new dependency in the pull request.
- Do not add `unwrap()` or `expect()` to request, persistence, or cleanup paths.
- Errors should say what failed, why it matters, and what the operator can do.
- Keep comments load-bearing: explain the invariant or reason, not the syntax.
- Preserve unrelated work in a dirty tree and avoid broad mechanical rewrites
  unless they are part of the reviewed change.

## Prepare the pull request

Before opening a pull request:

1. Rebase or merge the current `main` according to the repository workflow.
2. Keep one logical change per pull request.
3. Review the diff for generated files, accidental artifacts, secrets, and
   unrelated formatting.
4. Run formatting, focused tests, the default build/test path, and every
   contract checker affected by the diff.
5. Build and visually inspect changed documentation.
6. Include the problem, design boundary, behavior change, failure behavior,
   tests, and remaining limitations in the description.
7. Attach comparable receipts for performance or hardware claims.

Use a plain descriptive title without a project prefix. Automatic CI reruns
formatting, Linux default-feature checks, policy checks, and documentation
checks based on changed paths. Backend compile jobs are deliberate manual
checks; real backend behavior comes from local, source-bound qualification.
See [CI and local qualification policy](docs/policies/ci-policy.md).

## Changes that need prior agreement

Expect design discussion before work that:

- adds another model family;
- adds a sidecar process or duplicate model allocation;
- creates public configuration outside the typed startup registry;
- dispatches by a marketing device name or machine-specific identity;
- bypasses memory, identity, provenance, cleanup, or validation checks;
- introduces a broad refactor without a behavioral reason; or
- claims a performance improvement without comparable evidence.

The goal is not to block ambitious work. It is to agree on the boundary and
proof before a large implementation makes review expensive.

## License and discussion

Kiln is MIT-licensed; see [`LICENSE`](LICENSE). By submitting a pull request,
you agree to release your contribution under that license.

Use [GitHub issues and discussions](https://github.com/ericflo/kiln) for design
questions, bug reports, and contributor coordination. There is no project
Discord or Slack.
