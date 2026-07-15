# CI And Local Hardware Qualification Policy

Kiln uses GitHub Actions as a small platform-independent safety net. It does
not use hosted CI as proof that a GPU backend is correct, stable, or fast.
Backend qualification runs locally on named hardware and produces compact,
validated receipts under `qualification/receipts/`. Detailed cross-engine
serving measurements live separately under `benchmarks/receipts/` and are
validated by the shared serving driver.

## Automatic Work

An ordinary Rust push or pull request may run:

- Rust formatting.
- One Linux default-feature build/test job, including the focused substrate,
  dependency-tree, qualification-tool, and portable-receipt checks.
- Dependency advisory, license, source, and ban policy.
- UI smoke, Pages, or release-version checks only when their narrowly scoped
  paths change.
- The small OPD gate-parser self-test when its own paths change.

Automatic jobs must be cheap and useful without a GPU. A passing automatic run
does not qualify CUDA, ROCm, Vulkan, Metal, packaging, or performance.

## Manual Work

Compile-only backend jobs remain available for deliberate compatibility checks:

```bash
gh workflow run ci.yml --ref main -f backend_build=metal
gh workflow run ci.yml --ref main -f backend_build=vulkan
gh workflow run ci.yml --ref main -f backend_build=cuda
gh workflow run ci.yml --ref main -f backend_build=rocm
gh workflow run ci.yml --ref main -f backend_build=all
```

These jobs still do not count as hardware evidence. The real device must run
the qualification workload and produce a receipt.

Release packaging is also deliberate. Create and inspect the tag first, verify
the relevant local receipts, and dispatch from that tag:

```bash
gh workflow run server-release.yml --ref kiln-vX.Y.Z
gh workflow run docker-server-release.yml --ref kiln-vX.Y.Z
gh workflow run desktop-build.yml --ref desktop-vX.Y.Z
gh workflow run runpod-image.yml --ref main
```

The old performance-fixture workflow has no schedule or pull-request trigger.
It is retained temporarily as a manual compatibility path while the structured
local qualification runner replaces it.

## Measured Cost Before This Policy

The measurements below came from recent completed GitHub runs on 2026-07-09.
They are wall times unless aggregate runner-minutes are stated.

| Workflow | Previous automatic trigger | Hosted platform | Average wall time | Distinct signal |
| --- | --- | --- | ---: | --- |
| Rust CI | Rust push/PR | Ubuntu and macOS | 17m53s | Formatting, CPU tests, dependency policy, and compile-only backend compatibility |
| Desktop build | Desktop push/PR/tag | Ubuntu, Windows, and macOS | 10m40s | Cross-platform package construction |
| Server release | `kiln-v*` tag | Ubuntu, Windows, and macOS | 94m50s | Cross-platform release archives |
| CUDA Docker release | `kiln-v*` tag | Ubuntu | 48m25s | CUDA image construction and publication |
| RunPod image | Relevant push and weekly schedule | Ubuntu | 19m39s | Container heartbeat and manifest smoke |
| OPD bench gate | Selected PR paths | Ubuntu | 13s | Gate-parser self-test; its CUDA work was already manual |
| Pages | Site changes on `main` | Ubuntu | 1m27s | Site smoke and deployment |
| Performance nightly | Nightly and selected PR paths | Ubuntu, optional self-hosted | 17s | No useful signal: 10 of 10 recent runs failed before hardware work |
| Release-version drift | Broad docs and UI paths | Ubuntu | 1m43s | Release example consistency; browser checks duplicated Pages/UI smoke |
| UI smoke | UI changes | Ubuntu | 1m23s | Headless server-dashboard behavior |

The old Rust workflow started nine jobs on an ordinary matching change. Four
were Metal, Vulkan, CUDA, and ROCm compile jobs. A representative run spent
40.5 of 46.75 aggregate runner-minutes on those jobs; CUDA and ROCm alone used
31.3 minutes. A desktop run used 23.8 aggregate runner-minutes. One server
release used 173.4, with another 51 for its Docker release. RunPod runs ranged
from 5m53s to 26m40s.

Platform billing multipliers can make macOS and Windows more expensive than
the raw minute totals imply.

After this policy, an ordinary Rust change launches three jobs: formatting,
one Linux CPU job, and dependency policy. Backend jobs require an explicit
dispatch, and packaging and performance workflows have no automatic trigger.
The first successful post-change run, GitHub Actions run `29049575526`, measured
3m52s wall time and approximately 4m36s of aggregate hosted time. This is one
sample, not a stable range; retain measurements from later representative Rust
changes before treating 4-6 aggregate minutes as the established baseline.

## Local Receipt Flow

On Linux qualification hosts, run ad hoc Cargo verification through
`scripts/cargo-bounded.sh`. The wrapper serializes build work, preflights
available memory, preserves a host-memory reserve, and places the compiler and
linker process tree in one memory-capped, no-swap systemd scope. Source-bound
qualification cases run inside a bubblewrap PID namespace, which cannot attach
its namespaced Cargo PID to the host user manager as a scope. Those declared
builds use the wrapper's transient-service mode instead: the service has the
same aggregate memory/no-swap cgroup, a private network, a hard runtime limit,
and control-group teardown. Its versioned closed-source environment policy
admits only the pinned Cargo/ROCm paths, locale, user-home, and user-systemd
connection variables required to build; ambient compiler flags, target paths,
credentials, and API tokens do not enter the wrapper or service. The device
runtime remains in the original network/PID-isolated case. Qualification
workloads must not overlap a separate Cargo build; build first, then run the
device workload.

The Strix Halo Vulkan development soak also has a committed thermal policy.
After the source-bound build launches the server, the driver resolves Linux
hwmon by `k10temp` plus `Tctl`, samples every 250 ms, and terminates the complete
server process group at 97 C. Missing, duplicated, or malformed telemetry fails
closed. The workload manifest and receipt bind the selector, interval, limit,
start/peak/end readings, and trip count; no ambient environment variable can
change them.

```bash
scripts/cargo-bounded.sh check --locked -p kiln-server --lib
scripts/cargo-bounded.sh test --locked -p kiln-server --lib config::tests -- --test-threads=1
```

Always name `--lib`, `--bin`, or `--test` for a filtered Cargo test. A filter
changes which tests execute, but without a target selector Cargo still builds
every integration-test executable in the package before applying that filter.

Raw logs, traces, and profiles belong under ignored `.qualification/` paths.
Only compact receipts, workload manifests, schemas, and useful summaries are
checked in. The complete new-machine, workload, validation, comparison, and
check-in procedure is in [Local Hardware Qualification](qualification.md).

Capture this machine's backend environment:

```bash
python3 scripts/qualification/environment.py \
  --backend rocm --host-id strix-halo

python3 scripts/qualification/environment.py \
  --backend vulkan --host-id strix-halo
```

Validate portable receipts on any clone:

```bash
mapfile -d '' receipts < <(
  find qualification/receipts -type f -name '*.json' -print0 | sort -z
)
python3 scripts/qualification/receipt.py "${receipts[@]}"

mapfile -d '' benchmark_receipts < <(
  find benchmarks/receipts -type f -name '*.json' -print0 | sort -z
)
python3 scripts/bench-concurrent-batch.py \
  --validate-receipt "${benchmark_receipts[@]}"
```

On the machine that still has ignored raw artifacts, also require local hashes
and the current source identity:

```bash
python3 scripts/qualification/receipt.py \
  --require-current-source \
  --require-local-artifacts \
  qualification/receipts/rocm/strix-halo/*.json
```

Passing hardware evidence requires a clean source tree, a real requested
device, no skipped required test, exact source/model/workload identity, and a
validated receipt. Compile-only jobs, mocks, source-string assertions, and
tests that return success after skipping the GPU do not satisfy that contract.

## Ownership

- GitHub Actions owns cheap syntax, CPU behavior, dependency policy, and
  portable receipt validation.
- The named local machine owns backend correctness, latency, memory, soak, and
  performance evidence.
- Release operators own verifying the local receipt set before manually
  dispatching packaging.
- `docs/plans/confidence-hardening-goal.md` is the temporary execution source of
  truth until the final common-source-tree qualification is complete.
