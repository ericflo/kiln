# CI and local hardware qualification policy

Kiln separates inexpensive repository checks from accelerator evidence.
GitHub-hosted CI can prove that selected code builds, tests pass, contracts
validate, and repository policy holds. It cannot, by itself, prove that a real
CUDA, ROCm, Vulkan, or Metal device is correct, stable, or fast.

## Start here

| If you need to know… | Use this evidence |
| --- | --- |
| Did the changed code pass portable checks? | The path-matched GitHub Actions jobs below |
| Does an accelerator feature still compile? | The corresponding manually dispatched backend job |
| Did the change run correctly on a real accelerator? | A validated local qualification receipt |
| Did latency, throughput, or memory improve? | Comparable before/after receipts from the same workload and policy |
| Is a release ready to publish? | Green portable checks, relevant local receipts, and a manual packaging run from the intended tag |

A skipped job usually means its path filter or manual-dispatch condition did
not match. It is not a pass. Open the workflow file and confirm the trigger
before drawing a conclusion from the check list.

## What a green check proves

Every check has a bounded claim:

- A CPU test proves only the behavior exercised by that test.
- A feature build proves that the selected source and toolchain compiled and
  linked. It does not prove on-device execution.
- A software Vulkan run can catch API and shader regressions, but it does not
  qualify a physical Vulkan implementation.
- A portable receipt check proves that checked-in evidence matches its schema
  and retained-artifact policy. It does not rerun the workload.
- A local receipt proves only the declared source, model, workload, backend,
  device, and pass criteria. One machine never becomes a device allowlist or a
  universal performance claim.

Mocks, source-text assertions, and tests that return success after skipping
required hardware do not count as accelerator evidence.

## Automatic checks

The repository currently uses these automatic workflows:

| Workflow | Trigger | Primary claim |
| --- | --- | --- |
| `ci.yml` | Matching Rust, manifest, toolchain, or policy changes | Formatting, the Linux default-feature build and tests, focused substrate checks, qualification-tool tests, portable receipts, and dependency policy |
| `repository-hygiene.yml` | Every push and pull request | Tracked artifacts and production-file budgets satisfy repository policy |
| `qualification-contract.yml` | Matching contract, qualification, benchmark-receipt, or related source changes | Environment and source-parsing ratchets, qualification tooling, retained evidence, and portable receipts validate |
| `ui-smoke.yml` | Matching server-dashboard, desktop-UI, or thinking-budget changes | Browser and static UI contracts pass |
| `release-version-drift.yml` | Matching version-owning or version-consuming changes | User-facing release examples and runtime defaults agree with their canonical owners |
| `opd-bench-gate.yml` | Matching pull-request paths | The inexpensive OPD gate parser detects known pass and regression fixtures |
| `pages.yml` | Matching changes merged to `main`, or manual dispatch | The documentation contracts, build, browser smoke, and Pages artifact pass before deployment |

Path filters intentionally avoid unrelated work. When a change crosses a
contract boundary that the filters do not express, run the affected checker
locally and update the workflow filter in the same change.

The ordinary `ci.yml` path launches three automatic jobs: Rust formatting, one
Linux default-feature job, and dependency policy. Its accelerator jobs require
manual dispatch.

## Manual backend checks

Dispatch a backend check against the exact ref under review:

```bash
gh workflow run ci.yml --ref <commit-or-branch> -f backend_build=metal
gh workflow run ci.yml --ref <commit-or-branch> -f backend_build=vulkan
gh workflow run ci.yml --ref <commit-or-branch> -f backend_build=cuda
gh workflow run ci.yml --ref <commit-or-branch> -f backend_build=rocm
gh workflow run ci.yml --ref <commit-or-branch> -f backend_build=all
```

The lanes do not all provide the same evidence:

| Lane | What it currently does | Remaining proof |
| --- | --- | --- |
| Metal | Builds and tests on a hosted Apple Silicon runner | Run the intended production workload and retain a source-bound receipt |
| Vulkan | Checks the server feature and runs the Vulkan kernel suite with a software implementation | Run on each physical implementation relevant to the claim |
| CUDA | Compiles and links with the CUDA toolkit, without a device | Run on a physical CUDA device |
| ROCm | Compiles and links with the ROCm toolkit, without a device | Run on a physical ROCm device |

The CUDA and ROCm architecture values in those workflow jobs are compile-time
targets chosen to bound hosted cost. They are not runtime support policy.
Backend dispatch and user-facing defaults must remain capability-driven.

## Local qualification

Use [Local hardware qualification](qualification.md) for the complete
new-machine, workload, validation, comparison, and publication procedure. The
short form is:

1. Start from the exact source revision you intend to claim.
2. Select an existing backend-generic workload, or add one with explicit pass
   criteria.
3. Build before starting the device workload; do not overlap another Cargo or
   accelerator job.
4. Capture the environment and run the workload on the requested real device.
5. Validate the receipt, required cases, source identity, and local artifact
   hashes.
6. Compare only receipts with compatible workload and comparison policy.
7. Commit compact receipts and manifests; keep raw logs, traces, and profiles
   in ignored `.qualification/` storage.

Capture an environment receipt with an operator-chosen stable host label:

```bash
python3 scripts/qualification/environment.py \
  --backend vulkan \
  --host-id <stable-host-id>
```

Replace the backend with `cuda`, `rocm`, or `metal` as appropriate. The
captured device identity belongs in the receipt; do not copy it into runtime
dispatch, workload admission, or product defaults.

Validate every portable qualification and serving-benchmark receipt:

```bash
mapfile -d '' receipts < <(
  find qualification/receipts -type f -name '*.json' -print0 | sort -z
)
if ((${#receipts[@]})); then
  python3 scripts/qualification/receipt.py "${receipts[@]}"
fi

mapfile -d '' benchmark_receipts < <(
  find benchmarks/receipts -type f -name '*.json' -print0 | sort -z
)
if ((${#benchmark_receipts[@]})); then
  python3 scripts/bench-concurrent-batch.py \
    --validate-receipt "${benchmark_receipts[@]}"
fi
```

On the qualification machine, add the strict local checks to the relevant
receipt paths:

```bash
python3 scripts/qualification/receipt.py \
  --require-current-source \
  --require-local-artifacts \
  qualification/receipts/<backend>/<host-id>/*.json
```

A passing hardware claim requires a clean source tree, the requested real
device, no skipped required case, exact source/model/workload identity, and a
validated receipt.

## Bounded local Cargo work

On Linux qualification hosts, run ad hoc Cargo work through
`scripts/cargo-bounded.sh`. It serializes builds, checks available memory,
reserves host memory, and contains the compiler/linker process tree within a
memory-capped, no-swap cgroup.

Source-bound qualification cases run inside a PID and network boundary.
Declared builds use the wrapper’s transient-service mode so the build retains
an aggregate memory limit, private network, runtime limit, and control-group
cleanup. Its versioned environment policy admits only the paths, locale, home,
and user-service variables required for the build; ambient compiler flags,
target directories, credentials, and API tokens are excluded.

Examples:

```bash
scripts/cargo-bounded.sh check --locked -p kiln-server --lib
scripts/cargo-bounded.sh test --locked \
  -p kiln-server --lib config::tests -- --test-threads=1
```

Always name `--lib`, `--bin`, or `--test` for a filtered package test. A test
name filters execution, but Cargo otherwise still builds every integration
test target in that package.

## Manual packaging

Packaging is deliberate and is not triggered by tag creation alone. Inspect
the intended tag and relevant local receipts before dispatch:

```bash
gh workflow run server-release.yml --ref kiln-vX.Y.Z
gh workflow run docker-server-release.yml --ref kiln-vX.Y.Z
gh workflow run desktop-build.yml --ref desktop-vX.Y.Z
gh workflow run runpod-image.yml --ref <commit-or-branch>
```

A run from the wrong ref is not release evidence. Confirm the checked-out ref,
artifact version, digests or attestations, and publication result in the
workflow summary.

## Failure triage

| Symptom | First response |
| --- | --- |
| Expected job is absent | Check the workflow path filter and event; dispatch it manually if the claim needs it |
| Portable contract check fails | Run the named checker locally; change the canonical schema, generator, or implementation rather than editing generated output |
| Backend build fails | Reproduce with the same feature, locked dependencies, and toolchain; do not weaken an on-device claim |
| Hardware case skips | Treat it as missing evidence; fix device admission or run on a qualifying device |
| Receipt rejects source identity | Return to the claimed clean revision and rerun; do not edit the identity by hand |
| Before/after receipts are incompatible | Rerun one side with the same workload and comparison policy |
| Packaging succeeds but qualification is absent | Do not publish; obtain and validate the relevant local receipt first |

## Historical cost record

The automatic/manual split was introduced after measurements from completed
GitHub runs on 2026-07-09. The previous Rust workflow started nine jobs for an
ordinary matching change. In one representative run, accelerator compile jobs
used 40.5 of 46.75 aggregate runner-minutes; CUDA and ROCm used 31.3 minutes.
The first successful run after the split used 3m52s wall time and about 4m36s
of aggregate hosted time.

Those figures explain the policy; they are not a current performance promise.
Re-measure representative runs before using them for capacity or cost
planning.

The legacy performance-fixture workflow remains manual while structured local
qualification replaces it. Its machine-bound thresholds are compatibility
fixtures, not backend support policy or portable performance baselines.

## Ownership

- GitHub Actions owns inexpensive syntax, CPU behavior, dependency and
  repository policy, and portable evidence validation.
- The operator of each declared local machine owns backend correctness,
  latency, memory, soak, and performance evidence from that machine.
- Release operators own checking the relevant receipt set and dispatching
  packaging from the intended tag.
- Contributors own updating path filters when a new source begins to affect an
  existing contract.
