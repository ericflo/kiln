# Local Hardware Qualification

Kiln qualifies GPU backends on named physical machines. GitHub Actions and
compile-only backend jobs are portability checks, not hardware evidence. A
qualification run starts from a clean commit, executes a checked-in workload,
keeps bounded raw output under `.qualification/`, and writes one compact JSON
receipt under `qualification/receipts/`.

Detailed Kiln/vLLM serving sweeps use the separate protocol in
[`BENCHMARKS.md`](../BENCHMARKS.md) and belong under `benchmarks/receipts/`.
Do not place that schema in `qualification/receipts/`; both trees have their
own strict validator and CI checks both.

## Prepare A Machine

Fetch and fast-forward `main`, then confirm that the checkout is clean. Do this
again before every receipt so its commit and source-tree identity are exact.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git status --short
```

Install the backend runtime and every command-line probe named by the workload:

| Backend | Required environment probes |
| --- | --- |
| ROCm | `rocminfo`, `hipcc`, AMD DRM sysfs, and the Rust toolchain. The standard installation is `/opt/rocm`; source-building workloads use `ROCM_PATH=/opt/rocm`. |
| Vulkan | `vulkaninfo`, either `glslc` or `glslangValidator`, DRM sysfs when available, and the Rust toolchain. |
| CUDA | `nvidia-smi`, `nvcc`, and the Rust toolchain. Put `nvcc` on `PATH` or install the toolkit at `/usr/local/cuda`; the closed case does not inherit an ambient `CUDA_HOME`. |
| Metal | `system_profiler`, `sysctl`, `sw_vers`, `xcrun --find metal`, the macOS SDK, Apple Clang, and the Rust toolchain. A full Xcode toolchain is required when Command Line Tools alone do not provide `metal`. |

Fetch Rust and backend dependencies before entering the offline or
network-isolated qualification environment.

Validate the workload contract before spending device time:

```bash
python3 scripts/qualification/workload.py \
  qualification/workloads/environment-v1.json \
  qualification/workloads/correctness-core-v1.json
```

The runner rejects a dirty worktree, an uncommitted workload, missing required
variables, a missing required device, silent skips, and an existing receipt or
raw-run directory. Do not bypass those checks.
The runner also owns interruption containment. It starts each case in a new
process group and, on timeout or `Ctrl-C`, signals case descendants before the
outer sandbox leader so a Python serving driver can stop its independently
grouped server, delete private model snapshots, and publish no misleading
receipt. The default cleanup allowance is 65 seconds and may be set explicitly
with `--term-grace-seconds` up to the hard 75-second bound. If execution is
interrupted before receipt publication, the runner removes the exact
`.qualification/runs/<receipt-id>` directory transactionally and exits 130
without a traceback. A cleanup failure is itself a runner failure rather than
ignored residue. `SIGKILL`, kernel failure, or machine power loss cannot run
userspace cleanup; after such an event, verify process ownership before
removing the exact ignored directory.

### Capture Environment And Memory Capability

Run the committed `environment-v1` workload first on every physical machine.
It emits the ordinary outer source-bound qualification receipt and a second
ignored collector receipt named by `output_path`. The checked-in outer receipt
owns the verdict; the ignored receipt and raw probe output exist for local
diagnosis. Use a unique ignored path and never point it into the source tree.
`${host_id}` is runner-owned and resolves directly from `--host-id`; do not
repeat it as a workload variable.

Strix Halo ROCm and Vulkan:

```bash
PATH="$HOME/.cargo/bin:$PATH" python3 scripts/qualification/run.py \
  --variant rocm \
  --host-id strix-halo \
  --var output_path=.qualification/environment/strix-halo-rocm.json \
  qualification/workloads/environment-v1.json

PATH="$HOME/.cargo/bin:$PATH" python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  --var output_path=.qualification/environment/strix-halo-vulkan.json \
  qualification/workloads/environment-v1.json
```

16 GB RTX 4090 Laptop GPU:

```bash
PATH="$HOME/.cargo/bin:$PATH" python3 scripts/qualification/run.py \
  --variant cuda-rtx4090-laptop-16gb \
  --host-id rtx4090-laptop \
  --wsl2-thermal-policy qualification/host-policies/rtx4090-laptop-wsl2-cgroup-pacing-v2.json \
  --var output_path=.qualification/environment/rtx4090-laptop-cuda.json \
  qualification/workloads/environment-v1.json
```

24 GB desktop RTX 4090:

```bash
PATH="$HOME/.cargo/bin:$PATH" python3 scripts/qualification/run.py \
  --variant cuda-rtx4090-desktop-24gb \
  --host-id rtx4090-desktop \
  --var output_path=.qualification/environment/rtx4090-desktop-cuda.json \
  qualification/workloads/environment-v1.json
```

M1 MacBook Air:

```bash
PATH="$HOME/.cargo/bin:$PATH" python3 scripts/qualification/run.py \
  --variant metal-m1-macbook-air \
  --host-id m1-macbook-air \
  --var output_path=.qualification/environment/m1-macbook-air-metal.json \
  qualification/workloads/environment-v1.json
```

The CUDA variants select logical device zero and fail unless the complete
reported device name and memory class match the committed machine. The laptop
range is 15,000 through 17,000 MiB; the desktop range is 23,000 through 25,000
MiB. This accommodates driver-reported capacity while preventing one 4090 from
standing in for the other. Edit and review the committed workload if a machine
uses a different logical index; do not substitute `CUDA_VISIBLE_DEVICES` in the
shell. The Metal variant requires the complete `Apple M1` device name, exactly
eight reported GPU cores, positive physical-memory total, explicit Metal-family
support, and `unified_memory=true`.

On WSL2, the CUDA environment contains an additional closed `platform` object.
The outer runner capture binds all of the following to the receipt:

- WSL application/kernel and Windows OS versions;
- Windows CIM GPU name, PNP identity, and display-driver version, cross-checked
  against both `nvidia-smi` and NVML;
- hashes and resolved paths for the WSL `libcuda`, NVML, and `nvidia-smi`
  bridge files;
- `/usr/local/cuda` manifest, `nvcc`, concrete `libcudart`, and independent
  CUDA driver/runtime API calls;
- native-ext4 mount identity plus fsync, atomic replace, case, hardlink, and
  symlink semantics;
- `/proc/meminfo`, cgroup v2 controller identity, and a create/write/read/remove
  probe beneath the delegated user `app.slice`;
- system and user-systemd state, Linux or Windows formatted thermal-zone
  telemetry, and direct NVML GPU temperature telemetry; and
- the runner's accepted network/PID containment mechanism.

The WSL2 runner always uses the util-linux namespace path even if bubblewrap is
installed. Its tracked launcher requires Landlock, permits native toolchain
execution, and denies WSL's root-level `/init` interpreter. Preflight must
prove private loopback, an unreachable external route, and a real permission
denial for `cmd.exe`; otherwise no run artifacts are created. Landlock ABI 2 or
newer is required so the launcher can allow the `REFER` right from `/` while
granting `EXECUTE` only below native executable roots. The same preflight moves
a file across two directories on the native workspace filesystem. This keeps
Rust's temporary-archive rename usable without allowing `/init` execution and
fails before evidence if the namespace would instead return `EXDEV`. The
contained environment case repeats the interface, route, user-map, and
Windows-interop checks. A caller-provided environment variable alone cannot
attest containment.

Capability values are exactly `available` or `unavailable`; an unavailable
safeguard is also named in `unsupported` and is never converted into a passing
probe. The current WSL2 boundary creates exact runtime `dbus.socket` and
`dbus.service` units when the distribution omitted the user bus, then proves an
owned user transient unit. A fresh WSL user manager can also omit
`/run/user/<uid>/systemd/user`; the launcher creates that directory with mode
0700 only after the runtime and `systemd` parents prove they are real
directories owned by the current user and not group/world-writable. A symlink,
wrong owner, writable parent, publication conflict, missing bus socket, or
failed transient-unit proof aborts the case. Continuous supervision opens one
persistent Windows process. Its startup handshake reads the exact CPU name from
`HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0`,
resolves exactly one `\_TZ.THRM` instance and all four expected
`Thermal Zone Information` performance counters, and exchanges bounded,
strictly sequenced request/response records over pipes. No CIM/WMI query is
used during admission or continuous sampling. Each sample
cross-checks whole- and tenth-Kelvin fields and converts the high-precision
field to millicelsius. The GPU side holds one initialized NVML handle selected
by exact UUID and reads temperature in process. It does not launch PowerShell,
WMI, or `nvidia-smi` for every sample. Before launching a v2-policy child, the
outer supervisor requires the configured three consecutive readings at or
below the current 75.05/70 C host/GPU pacing-resume thresholds. The host value
is the exact nearest reading in the Windows counter's tenth-Kelvin
representation. Pre-admission
samples remain included in the outer sample count and peaks but are not sent
to a scope that does not yet exist. Timeout or a hard-limit reading fails
without launching the child. The checked
`rtx4090-laptop-wsl2-boundary-v1` policy binds that source to the exact
`Intel(R) Core(TM) Ultra 9 185H`, uses a 95 C hard limit below Intel's 110 C
Tjunction, and independently binds the exact GPU UUID/name to an 85 C limit.
The outer supervisor treats either telemetry read failure or either inclusive
limit as a trip, terminates the complete inner scope, and requires three
post-exit samples at or below 85 C host and 75 C GPU. It injects only the
content hash into the child. Missing counter identity, malformed or reordered
records, a ten-second sensor response timeout, NVML failure, nonzero sensor
exit, stderr, trailing output, or cleanup failure is fatal. Closing the
supervisor closes the counter's sole stdin writer, and the Windows process must
exit cleanly before evidence can complete.

Every WSL2 CUDA case is also placed in a user scope before `unshare`. The scope
round-trips a 10 GiB aggregate `memory.max`, zero `memory.swap.max`, group OOM
kill, and 512-PID limit. This WSL systemd delegation exposes no CPU controller,
so the trusted outer controller reads aggregate `cpu.stat`, freezes/unfreezes
the complete scope against a 50% budget, and retains a sentinel until final CPU
use is no greater than the elapsed allowance. Missing bus units, an inert
limit, malformed telemetry, absent containment, accounting failure, timeout,
or scope residue fails the case. The outer process inventory also rejects an
already-running host Cargo/rustc process before the private PID namespace can
hide it. User-manager launch is deliberately not attempted inside Landlock
because the manager would execute the requested process outside the private
namespaces.

Toolkit provenance is descriptive: recording CUDA 12.4 does not by itself
establish the workload manifests' cudarc CUDA 12.8 API contract or any
accelerator correctness.

#### WSL2 RTX 4090 Laptop environment results

The first source-bound laptop environment run passed from clean pushed commit
`c55bff4a76ced998bd51ebc3098822383d5f28d1` and tree hash
`sha256:74fcaa95635e7a003a2c91c9b8308fa7a58d1778620761b81f7831e48e1905a2`.
Its retained outer receipt is
`qualification/receipts/cuda/rtx4090-laptop/20260725t042515089130z-cuda-rtx4090-laptop-local-environment-v1-df3e8fee15-v1.json`.
The outer receipt and ignored inner collector receipt both pass current-source,
local-artifact, and known-commit validation. The case exited zero with empty
stderr, no output-assertion or infrastructure failure, the accepted
`util-linux-unshare-user-net-pid-landlock-v1` mechanism, and no process residue.

The selected device is logical index zero, NVIDIA GeForce RTX 4090 Laptop GPU,
UUID `GPU-fff83066-80fa-ac5f-edbe-4ebd3ac9bbfd`, PCI
`00000000:01:00.0`, SM 8.9, with 17,171,480,576 total bytes and
15,680,405,504 free bytes at capture. Windows driver 32.0.15.9636 maps to
NVIDIA 596.36 and agrees with both `nvidia-smi` and NVML. The receipt records
NVML 13.595.71.01, libcuda driver API 13.2, CUDA SDK/runtime 12.4, nvcc
12.4.99, rustc/Cargo 1.96.1, 16,459,141,120 WSL VM bytes, and 4 GiB swap.

This closes the laptop environment and capture-time memory-capability gate
only. The receipt explicitly leaves `systemd_user_transient` and
`host_thermal_guard` unavailable. It does not establish CUDA kernel
correctness, the cudarc 12.8 workload contract, model loading, low-memory
recovery, serving performance, or endurance.

Those two unavailable values remain truthful for the historical `c55bff4a7`
receipt. Later source repairs do not rewrite it: current-source runs require
the content-hashed Windows/NVML thermal supervisor and the repaired user-scope
boundary described above. A new run must retain its own raw preflight,
continuous samples, final CPU/memory/PID counters, cooldown, and scope-removal
evidence.

The post-repair environment run passed from clean pushed commit
`b77f10c9aa07befd6c7a5c47b3dbec1468f1aec3` and tree hash
`sha256:3a8c73bc786a40136d05fd6cad159c503914fc65031d13041a8cae4cc44334ea`.
Its retained outer receipt is
`qualification/receipts/cuda/rtx4090-laptop/20260725t051116746290z-cuda-rtx4090-laptop-local-environment-v1-df3e8fee15-v1.json`.
All 16 platform capabilities are `available` and `unsupported` is empty. The
case ran inside the required Landlock boundary and a removed user scope with a
10 GiB memory ceiling, zero swap, 512-PID ceiling, and settled 50-percent
aggregate CPU allowance. Its outer thermal supervisor continuously observed
the exact Windows thermal zone and GPU UUID, reached 86.05 C host and 63 C GPU,
stayed below the 95/85 C hard limits, and completed the required safe handoff.
This second receipt proves that the repaired safeguards compose around the
environment workload. It does not establish any CUDA correctness, model,
pressure, serving, performance, or endurance claim.

New environment receipts retain the common device fields and may additionally
carry these closed optional fields:

| Field | Meaning |
| --- | --- |
| `logical_index` | Backend-reported zero-based device selection. CUDA binds this to the requested `nvidia-smi` index; Vulkan and Metal retain their enumerated index. |
| `device_uuid`, `pci_bus_id` | Stable NVIDIA hardware identity where the runtime reports it. They are `null` on backends without an equivalent probe. |
| `compute_capability` | NVIDIA major/minor capability, with `architecture` derived as `sm_<major><minor>`. |
| `compute_units` | ROCm compute-unit or Apple GPU-core count when the backend exposes it reliably. |
| `memory_available_bytes` | Capture-time device-available memory from `nvidia-smi` or AMD DRM. `null` means the backend has no honest equivalent; it never means zero. |

`memory_bytes` remains total device-visible capacity. On Apple silicon it is
the unified physical-memory total, not a promise that Metal may allocate all of
it. On an APU, DRM free memory is a capture-time observation rather than a
serving admission budget. Every CUDA/ROCm/Vulkan memory value is binary bytes
even when the source tool reports MiB. The raw artifact records the exact probe
arguments and output hash, selected class constraints, and redacted relevant
backend environment. Sensitive variable names are hashed; ambient controls do
not enter the closed case.

Native Linux cases prefer bubblewrap network and PID namespaces. When
bubblewrap is not installed, the runner uses util-linux
`unshare` with a mapped private user namespace, a private mount/proc view, a
private network namespace, and a private PID namespace whose supervisor uses
`--kill-child=SIGKILL`. A tracked helper enables `lo` before executing the case.
WSL2 always takes this helper path because it additionally requires Landlock
to deny Windows-interoperability execution.
Both routes first prove that loopback can carry traffic, that the namespace has
no interface other than `lo`, and that an external test-net address is
unreachable. WSL2 also proves a real Windows executable is denied. The live
containment regression launches a signal-ignoring
`setsid` child and proves it cannot survive either normal parent exit or a
runner timeout. On macOS the runner first proves that its `sandbox-exec`
profile preserves loopback while an external test-net connection is denied,
then applies that profile to the case. A missing executable, denied namespace,
failed probe, or cleanup escape stops before a run directory or receipt is
created. The environment receipt proves machine identity, runtime/toolchain
presence, and capture-time memory capability only. It does not prove Kiln
compiles, executes correctly, survives pressure, or meets a throughput target;
the later correctness, pressure, performance, and soak workloads own those
claims.

Each case runs under `closed-qualification-case-v1`: a fixed base containing
only path, toolchain-home, locale, temporary-directory, user, and user-session
plumbing, followed by the committed case's exact `environment` object and the
runner-owned result-path and variant identifiers. Ambient backend controls,
device selectors, compiler/linker flags, product `KILN_*` values, credentials,
and library paths do not enter a case. Declare a required value in the workload
instead. The ignored effective-run artifact records the policy, redacted base,
base hash, exact overrides, and final per-case environment hash; the compact
receipt carries the same base hash as its execution-environment identity.
The ROCm mixed-load driver also rejects ambient `KILN_*` server controls before
building. Configuration changes must be declared in a committed workload
variant; inherited shell overrides are never silently ignored or accepted as
source-bound evidence.

No checked-in qualification case may invoke Cargo directly. Workload validation
rejects `cargo` by basename, including an absolute path. Standalone Rust test
cases use `scripts/qualification/cargo-test-bounded.sh`, which pins one job, a
50% aggregate CPU quota (at most half of one logical CPU on average), the
platform floor (14 GiB on the 15.3 GiB WSL VM, 15 GiB on the existing native
Linux hosts), offline Cargo, zero scope/service swap, private networking, and
a 1,740-second cap. WSL2 re-verifies its current scope, exact memory/swap/PID/OOM
files, CPU-policy marker, thermal-policy hash, live loopback-only namespace,
private user map, and denied Windows execution before Cargo starts. Native
Linux continues to use the transient-service path.
Its
`closed-qualification-test-v1` environment is the source-build allowlist plus
the non-secret `KILN_QUALIFICATION` required-device gate, model/oracle paths,
and committed `CUDARC_CUDA_VERSION` and `KILN_CUDA_ARCHS` build controls;
credentials, ambient compiler flags, target directories, and unrelated
`KILN_*` values do not enter the process. The wrapper path and arguments remain
part of the committed workload and effective run artifact.

Source-building ROCm and Vulkan serving drivers likewise select an immutable
backend build specification, resolve the
requested toolchain, then execute the exact package, binary, feature, locked,
and offline build through `scripts/cargo-bounded.sh` with one job and a 15 GiB
`MemAvailable` floor. A systemd `CPUQuota=50%` bounds aggregate compiler,
linker, and helper CPU consumption even when a single Cargo job fans out inside
LLVM. ROCm alone receives `ROCM_PATH` and
`KILN_ROCM_ARCHS`; the Vulkan build strips ambient ROCm toolchain variables and
uses only the `vulkan` feature. The wrapper refuses overlapping
Cargo/rustc processes. Because the case retains bubblewrap PID isolation, the
offline build runs as a transient systemd user service rather than attempting
to attach the namespaced Cargo PID to a host scope. The service has an
aggregate `MemoryMax`, host reserve, zero swap, `PrivateNetwork=yes`,
control-group kill, a hard runtime cap, and a fail-closed package-temperature
watchdog. The typed build spec selects exactly one Linux hwmon input by
`name=k10temp` and `label=Tctl`, polls it every 250 ms, refuses to start at or
above 90,000 millicelsius, and stops the complete transient service if a later
reading reaches that limit. Missing, ambiguous, unreadable, non-integer, or
implausible telemetry also prevents or terminates the build. Ordinary ROCm and
Vulkan build services are capped at 840 seconds with a 900-second caller
timeout; the real-ROCm fault corpus uses 1140 and 1200 seconds respectively. This
60-second ordering ensures systemd can stop and collect the complete cgroup
before an outer qualification timeout can kill the wrapper. The measured server
still runs inside the case's separate network and PID namespaces. The committed
effective build config records the wrapper, job count, CPU quota, floor,
execution mode, private-network requirement, versioned environment policy, both
deadlines, memory policy, and all four thermal-selector fields. The driver mechanically
derives both `KILN_CARGO_CPU_QUOTA_PERCENT` and the wrapper's
`KILN_CARGO_HOST_THERMAL_*` controls from those typed fields; ambient values
cannot alter the quota, selector, or limit. `closed-source-build-v1` retains
only Cargo/Rustup homes, the pinned PATH and ROCm architecture/path, locale,
user/home, temporary-directory, and user-systemd connection variables. It
excludes ambient compiler flags, target directories, credentials, and API tokens before invoking the wrapper;
the wrapper independently applies the same policy when constructing the
service environment. Bounded stderr records the machine-specific
available/reserve/limit values. Do not lower the floor or bypass the wrapper to
obtain a receipt. Let the machine recover memory and rerun from the same clean
commit.

Outside a source-bound workload, `scripts/cargo-bounded.sh` also names every
ordinary systemd scope and stops that complete unit from its `EXIT` trap, so a
cancelled terminal or tool client cannot leave Cargo, rustc, or the linker
running in an orphaned scope. When exactly one `k10temp/Tctl` input exists, the
wrapper automatically applies the same 90,000-millicelsius, 250 ms guard to
ordinary commands and reports `thermal=automatic:...` in its preamble. A host
without that exact sensor runs with `thermal=disabled`; qualification source
builds do not accept that fallback because their four explicit typed fields make
missing or ambiguous telemetry a preflight failure. Operators may explicitly
configure a different stable selector only by setting all four documented
`KILN_CARGO_HOST_THERMAL_*` wrapper controls together.

For non-qualification use, `KILN_CARGO_CPU_QUOTA_PERCENT` optionally applies
the same aggregate systemd CPU limit in either execution mode; `100` represents
one logical CPU and values through `10000` are accepted. It is independent of
`KILN_CARGO_JOBS`: the latter limits Cargo's build graph concurrency, while the
quota contains all descendant threads. The wrapper reports the effective quota
in its preamble and rejects malformed or out-of-range values before launch.

Every serving driver creates a collision-resistant mode-0700 workspace below
`.qualification/serving` (the pressure driver uses
`.qualification/serving-pressure`). Names never depend on the sandbox PID,
because PID namespaces can assign the same PID on every run. Normal teardown
removes the workspace and private model snapshot. The runner sends `SIGINT` to
leaf case commands while keeping sandbox supervisors alive, allowing Python
`finally` owners to execute before hard containment. The ROCm mixed-load driver
also converts direct `SIGTERM` into a catchable interruption. A normal timeout
or `Ctrl-C` in that driver must therefore leave neither workspace nor copied
model payload. Treat residue after an uncatchable process or machine failure as
an explicit recovery condition: confirm no qualification or Kiln process still
references it before removing that exact directory.

Batching qualification must bind the complete typed startup policy, not only a
legacy actor environment switch. A serving workload that exercises the actor
declares `[batching]` values in its source-bound config, restarts the server,
and attests these exact runtime targets before measurement:

```text
GET /v1/config -> .batching.configuration
GET /v1/config -> .batching.actor_active
GET /health -> .decode_runtime.batching_configuration
GET /health -> .decode_runtime.batching_engine
```

The immutable objects at `/v1/config`
`.batching.configuration` and `/health`
`.decode_runtime.batching_configuration` must be equal. For a real backend,
`actor_active` must be true and the live health `batching_engine` snapshot must
be present and accepting before measurement. The attestation records rowwise
and prefix-aware values and sources; admission quantum intent, backend default,
effective clamp and source; actor-cycle idle; backend-owned burst admission;
and the numerical tile-alignment requirement. A malformed value, an unexpected
source, or a missing actor fails before device work.

Use only canonical mechanically derived names in new workload manifests:
`KILN_BATCHING_ROWWISE_DECODE`,
`KILN_BATCHING_PREFIX_AWARE_ADMISSION`, and
`KILN_BATCHING_PREFILL_ADMISSION_QUANTUM`, plus
`KILN_BATCHING_ACTOR_CYCLE_IDLE_MS` when pacing is part of the arm. Production
actor activation and the former direct-rendezvous worker are not configurable.
The removed names are ignored and must not appear in new launch environments.

Streaming-prefill qualification has the same source-bound rule. Declare the
complete `[streaming_prefill]` table in the committed variant, restart between
arms, and require these three serialized objects to be equal before device
work:

```text
GET /v1/config -> .streaming_prefill
GET /health -> .prefill_runtime.streaming_prefill
GET /v1/debug/model-state -> .streaming_prefill   # trusted-debug variant only
```

The attestation must retain configured source, backend dispatch, effective
dispatch and authority, threshold applicability, base/tape/detached effective
tiles and inheritance sources, derived detached boundary/replay tiles,
last-token LM-head policy, and both restart flags. New manifests use only
`KILN_STREAMING_PREFILL_MODE`,
`KILN_STREAMING_PREFILL_THRESHOLD_TOKENS`,
`KILN_STREAMING_PREFILL_TILE_TOKENS`,
`KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS`,
`KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS`, and
`KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD`. The six shorter historical names
and legacy TOML `enabled` exist for deployment compatibility, not new evidence.

A correctness arm should cover the prompt-length boundary immediately below
and at the effective threshold, compare forced `disabled` monolithic output
with forced `enabled` tiled output, exercise a prompt longer than one base tile,
and cover training work longer than the effective tape and detached tiles when
that backend supports the route. Record exact tokens/losses, cancellation and
settlement, TTFT/prefill time, per-stage timing, peak primary and host-backed
memory, allocator reclaim/resize events, device synchronizations, and any
non-finite or ownership failures. Do not interpret `mode="enabled"` on CPU or
Vulkan as support by itself; it is an explicit test route and still needs real
backend evidence.

For pause or OOM diagnosis, run one variable per arm: backend `auto`, forced
`disabled`, then a changed threshold or one tile. An explicit base tile feeds
`auto` tape and detached routes, so either set those specialized values
explicitly or record their inherited effective values. Attribute a pause only
when the trace shows the corresponding scheduler wait, tile computation,
external-yield synchronization, allocator reclaim, physical KV resize, or
memory-pressure event. A temporal gap alone is not evidence of VRAM
rebalancing. These ROCm, Vulkan, CUDA, and Metal runs belong on the named local
machines; hosted GPU CI is neither required nor accepted as qualification.

## Refresh The GRPO Reference Oracle

The compact fixture at
`crates/kiln-train/tests/fixtures/grpo_trl_oracle_v1.json` pins scalar GRPO
semantics independently of Kiln. Its generator hash-checks TRL 1.8.0's
`grpo_trainer.py`, calls the real `GRPOTrainer._compute_loss` with precomputed
policy/behavior/reference log-probabilities, differentiates with PyTorch
2.13.0, and takes one `torch.optim.AdamW` step. It runs entirely on CPU.

Use PyTorch's CPU wheel index so refreshing a scalar fixture does not download
CUDA libraries:

```bash
uv run \
  --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  --with 'torch==2.13.0+cpu' \
  --with 'trl==1.8.0' \
  python scripts/qualification/grpo_trl_oracle.py --check
```

Omit `--check` only when intentionally regenerating the fixture after changing
the pinned oracle or its input cases. Review the entire JSON diff. Automatic CI
validates the pins, canonical encoding, input hash, coverage, finiteness, and
shapes without installing TRL or PyTorch; Rust tests consume the numeric outputs
directly on each supported backend.

## Run A Workload

Choose a stable, non-secret host ID that identifies the physical machine. Run
one backend variant at a time. The runner prints the final receipt path and
stores bounded stdout/stderr plus their hashes under `.qualification/runs/`.

### CUDA And Metal Core Handoff

Run the environment variant for the machine first, then commit its receipt.
From that exact clean pushed source, run the matching core-correctness variant.
The CUDA variants require logical device zero to have the exact laptop or
desktop RTX 4090 product name. The Metal variant requires an Apple M1 with eight
GPU cores. Missing hardware is a test failure under `KILN_QUALIFICATION=1`;
the output contract also rejects skip diagnostics.

RTX 4090 Laptop GPU, 16 GB:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant cuda-rtx4090-laptop-16gb \
  --host-id rtx4090-laptop \
  --wsl2-thermal-policy qualification/host-policies/rtx4090-laptop-wsl2-cgroup-pacing-v2.json \
  qualification/workloads/cuda-metal-core-correctness-v1.json
```

Desktop RTX 4090, 24 GB:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant cuda-rtx4090-desktop-24gb \
  --host-id rtx4090-desktop \
  qualification/workloads/cuda-metal-core-correctness-v1.json
```

M1 MacBook Air, eight-core GPU:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant metal-m1-macbook-air \
  --host-id m1-macbook-air \
  qualification/workloads/cuda-metal-core-correctness-v1.json
```

Each run produces five required case results. `device-probe` binds the intended
machine class. `tensor-parity` covers device allocation, transfer, and layout
behavior. `matmul-parity` compares accelerator results with deterministic CPU
oracles. `graph-parity` requires an actual captured replay rather than an eager
fallback: CUDA compares single-row replay with independent eager token/logit
output, while Metal checks single-row and two-row ICB replay across short and
long sequence buckets, including KV-cache parity. `training-oracles` runs a
complete tiny LoRA SFT forward/backward/AdamW step and compares twenty native
BF16 AdamW updates with the source-pinned PyTorch trajectory. CUDA compilation
is pinned to SM 8.9 and the cudarc CUDA 12.8 API contract by the manifest; the
environment receipt records the actual installed driver and toolkit. These
bounded tests do not claim full-model correctness, multi-row CUDA graph safety,
resize behavior, memory-pressure safety, serving performance, or soak
completion.

#### Current WSL2 RTX 4090 Laptop correctness evidence

The first post-boundary core run from clean pushed source
`547d15754e5491abf135ddca1b172821ef604b0e` and source tree
`sha256:1979c8f4b40358e529c97221af5ee44cfa714f4c60d05e81fe08e1316ab2a8d7`
is retained at
`qualification/receipts/cuda/rtx4090-laptop/20260725t052543938882z-cuda-rtx4090-laptop-cuda-metal-core-correctn-9f21d75c94-v1.json`
with receipt hash
`sha256:16deee09107bf4a577270bb04404cff1ffa90ff9d201ebded3a03148798e0303`.
It is a strict-valid failed receipt, not partial acceptance.

Device selection, CUDA tensor resize/copy parity, all four F32 and three BF16
matmul shapes, both 20-step native BF16 AdamW trajectories, and the complete
tiny LoRA SFT step passed. The required graph case performed a real capture and
replay but selected token zero while the independent eager path selected token
seven. The test failed explicitly as graph-replay corruption. Its scope reached
7,978,455,040 bytes and 28 PIDs without a cgroup limit, swap, or OOM event;
settled 851,538,135 CPU microseconds inside an 851,549,685-microsecond
allowance; and removed itself. Continuous supervision peaked at 94.05 C host
and 66 C GPU below the 95/85 C hard limits, completed the safe handoff, and
left no process or scope residue. The graph defect must be fixed and the full
five-case workload rerun from a new clean pushed source before CUDA core
correctness passes.

The source repair that follows this receipt closes two independent defects
before a clean rerun. First, `Tensor::from_vec_on` inside the forward path used
`host_to_cuda_copy`, which bypassed the capture arena. CUDA captured device
allocation, pageable host-to-device copy, and free operations whose source
addresses belonged to temporary CPU tensors. Those host allocations were gone
before graph launch, so the current WSL2 driver read invalid source storage and
the valid warm-pass hidden state became all zero. The arena now uploads
host-initialized tensors once before capture, retains their device allocations,
requires byte-identical initialization during capture, and emits no captured
pageable-host copy for them. It also distinguishes zero, uninitialized, and
host-initialized allocation sequences and rejects underruns after the capture
forward.

Second, the old graph test repeated one sequence position three times. The
runner's owner-timeline guard deliberately evicts a graph unless the next
position advances by one, so the supposed third-call replay was actually a
second capture. The permanent test now uses three monotonically advancing
positions, computes independent eager references for all three, requires a
captured graph before the third call, and checks warmup, first-launch, and real
replay token parity. Under the full WSL2 boundary it returned eager and graph
tokens `7/7/7`, with zero replay logit difference. The new arena invariant test
proved stable pointer reuse and fail-closed host-byte, initialization-kind, and
sequence checks. Complete guarded CUDA library runs passed `1,027/1,027`
`kiln-tensor` tests and `412/412` `kiln-model` tests. These source-repair checks
do not replace the five-case clean-source receipt.

The unchanged workload subsequently passed all five required cases from exact
clean pushed source `1a6941330dee21b814caaa2f40d9652daffc0ef3` and source tree
`sha256:d44de18f440643c2ffd7b1620f9203f67c6fd1458fa49aa63f01381cc2fb4c20`.
The retained receipt is
`qualification/receipts/cuda/rtx4090-laptop/20260725t064445179639z-cuda-rtx4090-laptop-cuda-metal-core-correctn-9f21d75c94-v1.json`
with receipt hash
`sha256:500a0cf760cac67995a8452c266896790445f638026f1ca42317ed2c0dbafad3`.
Strict current-source, local-artifact, and known-commit validation passed.
Device probe, tensor parity, F32/BF16 matmul parity, real single-row graph
capture/replay parity, and the native BF16 AdamW plus tiny LoRA SFT training
oracles all passed in 148.181 seconds with no unsupported item.

The graph case returned eager and graph tokens `7/7/7` for warmup, first
captured launch, and actual replay, with zero maximum and mean replay-logit
difference. All five cases ran through the Windows/NVML thermal supervisor,
named user scope, 10 GiB memory and zero-swap bounds, 512-PID limit, 50-percent
feedback CPU controller, user/network/PID/mount namespaces, and Landlock.
Every scope was removed; peak scoped memory was 1,681,219,584 bytes, peak PIDs
were 29, and every memory-limit and OOM counter stayed zero. Each thermal
supervisor completed its stable handoff; the worst host/GPU peaks were
87.05/67 C below the 95/85 C hard limits. No matching process or scope residue
remained. This receipt closes only the bounded CUDA core-correctness subset on
the named 16 GB Laptop GPU under WSL2. It does not claim public-model loading,
multi-row CUDA graph safety, low-memory pressure and recovery, serving
performance, soak completion, native Linux, or a desktop RTX 4090.

### CUDA Memory Lifecycle Handoff

After the environment and core-correctness receipts pass, run the memory
lifecycle variant for the same 4090 class and exact clean pushed source.

RTX 4090 Laptop GPU, 16 GB:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant cuda-rtx4090-laptop-16gb \
  --host-id rtx4090-laptop \
  --wsl2-thermal-policy qualification/host-policies/rtx4090-laptop-wsl2-cgroup-pacing-v2.json \
  qualification/workloads/cuda-memory-lifecycle-v1.json
```

Desktop RTX 4090, 24 GB:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant cuda-rtx4090-desktop-24gb \
  --host-id rtx4090-desktop \
  qualification/workloads/cuda-memory-lifecycle-v1.json
```

The allocator case requires at least 4 GiB free before it starts. It first
allocates and trims 1 GiB, then configures the stream-ordered pool to retain a
freed 2 GiB allocation and requires an explicit trim to return at least 1 GiB
to device-visible free memory. The server-admission case performs no large
allocation: it reads the real `cuMemGetInfo` ceiling, retains a 1 GiB safety
floor, asks the exact server paged-KV admission gate for one 1 MiB block beyond
that ceiling, and requires the typed remediation error before cache
construction. The paged-KV case requires at least 3 GiB free, allocates a
four-layer BF16 cache, and preserves an exact marker through a 4,000-to-500
shrink and 500-to-4,000 grow. The largest replacement interval is about 1.1
GiB. All cases are serial, bounded, and fail on a skip diagnostic.

This proves the allocator-aware server gate rejects an oversized KV request
without inducing an OOM, plus healthy-headroom allocator and replacement
lifecycle. It is not the complete laptop low-memory gate: it does not load the
public model, drive serving near the 16 GB limit, or prove recovery under
model-resident pressure. Keep that later hardware run bounded by a declared
free-memory floor; do not fill the card merely to manufacture an OOM.

#### Current WSL2 RTX 4090 Laptop memory-lifecycle result

The four-case lifecycle workload passed from exact clean pushed source
`9fadc2592f1814d7dd68b3c96ab008e8b886d665` and source tree
`sha256:d44de18f440643c2ffd7b1620f9203f67c6fd1458fa49aa63f01381cc2fb4c20`.
The retained receipt is
`qualification/receipts/cuda/rtx4090-laptop/20260725t065312369587z-cuda-rtx4090-laptop-cuda-memory-lifecycle-v1-61a2e68c95-v1.json`
with receipt hash
`sha256:377583f15bc6365c4baf8a12f02c8f38e4f7b6a863ebd5958bdb321204956aeb`.
Strict current-source, local-artifact, and known-commit validation passed. All
four required cases passed in 495.495 seconds with no unsupported item.

The allocator began with 15,044 MiB free, showed a freed 2 GiB allocation
remained held by the stream-ordered pool, and required explicit trim to return
all 2,048 MiB. The production server gate retained its 1,024 MiB floor and
rejected block 14,021 above the live 14,020-block ceiling before allocation.
The four-layer BF16 paged-KV cache preserved its marker through physical
`4,000 -> 500 -> 4,000` replacement. Every named zero-swap scope was removed,
peak scoped memory was 5,336,924,160 bytes during the server test build, peak
PIDs were 28, and every memory-limit and OOM counter remained zero. Stable
thermal handoff completed for each case after worst host/GPU peaks of
92.05/66 C below the 95/85 C hard limits.

This is allocator, admission, and replacement substrate evidence under healthy
headroom. It does not close the laptop low-memory item and does not claim
public-model residency, model-resident pressure, request recovery, serving
performance, soak completion, native Linux, or a desktop RTX 4090.

#### Declared WSL2 RTX 4090 Laptop low-memory gate

`serving-cuda-low-memory-v1.json` is the bounded public-model continuation of
the retained lifecycle receipt. Run it only from a clean pushed source tree
with the local two-shard Qwen3.5-4B directory:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant cuda-rtx4090-laptop-16gb \
  --host-id rtx4090-laptop \
  --model "$(pwd)/Qwen3.5-4B" \
  --model-id Qwen/Qwen3.5-4B \
  --wsl2-thermal-policy qualification/host-policies/rtx4090-laptop-wsl2-cgroup-pacing-v2.json \
  qualification/workloads/serving-cuda-low-memory-v1.json
```

The single case first verifies by hash that the earlier lifecycle receipt
passed the production non-allocating admission rejection. It then builds the
current CUDA release server through the delegated 10 GiB, zero-swap, 50-percent
CPU WSL scope and starts the immutable stable-profile laptop launch. The build
requires 13 GiB live host availability and preserves 3 GiB outside its 10 GiB
aggregate ceiling. Readiness must expose the exact public model, two weight
shards, positive post-load CUDA residency, healthy `nvidia-smi` sampling,
clean-source execution provenance, and the exact source-built executable hash.
The laptop input leaves `memory.gpu_memory_gb` unset so the selected
`nvidia-smi` device is the single capacity authority, and pins the previously
observed 62-block/130,023,424-byte BF16 KV pool so removing the stale capacity
cap cannot enlarge that physical allocation.

After warmup, the driver records a deterministic 32-token baseline. A separate
CUDA-driver process allocates in at most 256 MiB chunks until free memory is at
or below 1,024 MiB. It refuses to start when the resident server is already at
that target unless it can still create a qualifying 64 MiB global drop,
allocates at most 1,280 MiB, requires at least 64 MiB of real external
allocation, samples global free memory through the reviewed WSL2 `nvidia-smi`
binary every 100 ms, and immediately fails and releases if free memory crosses
the 768 MiB floor. Peer-local `cuMemGetInfo` remains separate allocator
evidence. While the peer remains alive, the server must complete another
32-token request and its independent `nvidia-smi` sampler must corroborate
bounded pressure.

The peer must then exit on `SIGTERM` without a forced kill, free every
allocation, report no release error, and recover within 512 MiB of its
pre-pressure CUDA-driver reading. The server sampler must recover within 768
MiB of baseline in 60 seconds. A final request with the exact baseline prompt
and seed must reproduce all 32 token IDs and the finish reason. Any request
error, OOM/device-fault/quarantine log, stale memory sampler, backend
quarantine, nonzero or forced exit, process-group residue, or model-snapshot
payload fails the case.

This is a committed source contract only until its exact clean pushed-source
receipt passes. It deliberately combines safe bounded pressure with the
already retained pre-allocation rejection instead of manufacturing an OOM. It
does not qualify serving throughput, concurrency expansion, an eight-hour
soak, native Linux, or the desktop RTX 4090.

The first exact pushed-source attempt, receipt
`20260725t073758504795z-cuda-rtx4090-laptop-serving-cuda-low-memory--647fb2a614-v1`,
failed closed at the declared 1,800-second build limit while compiling the
BF16 FlashAttention CUDA objects from a cold cache. It never launched the
server, loaded the model, or started the pressure peer. The bounded scope
reported zero memory-limit or OOM events, peaked at 7,255,072,768 bytes and 29
PIDs, was removed cleanly, and completed thermal handoff after 92.05 C host
and 66 C GPU peaks. This is retained counterevidence for cold-cache build
completion within the current limit; the low-memory gate remains open.

A clean cache-warm restart, receipt
`20260725t081034583702z-cuda-rtx4090-laptop-serving-cuda-low-memory--647fb2a614-v1`,
then failed before compilation because the nested Cargo preflight observed 13
whole GiB available against the unchanged 14 GiB admission floor. Its scope
ran for 0.356 seconds, peaked at 42,921,984 bytes and seven PIDs, reported zero
memory-limit or OOM events, and was removed cleanly. This second receipt also
contains no model or pressure evidence.

A second clean pushed-source retry, receipt
`20260725t081310957001z-cuda-rtx4090-laptop-serving-cuda-low-memory--647fb2a614-v1`,
reproduced that 13-versus-14 GiB refusal after environment capture saw
15,046,885,376 bytes available. The WSL VM exposes only 15.33 GiB total, and
the runner plus its scope cross the old whole-GiB threshold. The revised typed
build boundary therefore requires 13 GiB available, reserves 3 GiB for the
host, and retains the same 10 GiB aggregate compiler scope with swap disabled.
It does not raise compiler capacity or change any GPU pressure, thermal,
network, process, filesystem, or cleanup limit. A new clean pushed-source
receipt must still pass before this gate closes.

The first run with the revised boundary, receipt
`20260725t081814165066z-cuda-rtx4090-laptop-serving-cuda-low-memory--647fb2a614-v1`,
admitted Cargo and completed all six BF16 causal FlashAttention CUDA objects
plus the FlashAttention crate. It then reached the unchanged 1,800-second
build limit while compiling later Rust dependencies, before server launch.
The scope peaked at 7,146,455,040 bytes and 30 PIDs, reported zero
memory-limit or OOM events, removed cleanly, and completed thermal handoff
after 90.05 C host and 66 C GPU peaks. This timeout is retained separately;
it still contains no model or pressure evidence.

The next exact pushed-source restart, receipt
`20260725t085019331711z-cuda-rtx4090-laptop-serving-cuda-low-memory--647fb2a614-v1`,
completed the CUDA release build and the full two-shard model upload. All 32
layers and 8,411,510,272 source bytes uploaded, and the post-upload pass
verified all 9,319,828,096 snapshot bytes before the CPU weights dropped.
Startup then rejected KV allocation without attempting one. NVIDIA reported
14,675,869,696 of 16,106,127,360 bytes used, leaving 1,430,257,664 bytes free
below the configured 1.5 GiB floor. Independently, the verification pass had
repopulated the clean snapshot page cache inside the 10 GiB host scope, so the
host governor reported no available budget.

The retained failed receipt has file
`sha256:66cc0097820efe4819f2d23f00b9e639c21ddd20c9fb35267baf03baeee0b159`.
The scope peaked at its exact 10,737,418,240-byte limit and recorded 5,737
limit events, but all OOM counters remained zero, the 66-PID peak drained, and
the scope and private snapshot were removed. Thermal supervision completed its
stable handoff after 93.05 C host and 68 C GPU peaks below the 95/85 C limits.
No KV cache, pressure peer, baseline request, pressure request, or recovery
request ran, so the low-memory gate remains open.

The next source contract closes both observed startup constraints without
raising the host cap. After the exact post-upload content check, Linux startup
must release every retained read-only mapping with `MADV_DONTNEED`, require a
successful `POSIX_FADV_DONTNEED` for every verified shard file, log the exact
released-shard count, and only then drop CPU weights. A cache-release failure
aborts startup. The laptop server floor is now 1.0 GiB, below the observed
1.430 GiB post-load free memory, and its inference-memory fraction is 0.1 so
auto-KV leaves bounded external-pressure room. The peer target, safety floor,
and maximum allocation are 1,024 MiB, 768 MiB, and 1,024 MiB respectively.
The revised workload is `sha256:f42fad082d86aa82480acb3a2a7e0f8d5f5f9ce189c972952015a944c31f9b2b`;
it still requires a real allocation, a complete request under pressure,
deterministic recovery, clean shutdown, and zero fault or residue evidence.

The first clean pushed-source attempt at that contract, receipt
`20260725t143235401396z-cuda-rtx4090-laptop-serving-cuda-low-memory--647fb2a614-v1`,
failed in preflight before Cargo. The exact TOML, hashed manifest, and driver
effective configuration all declared inference fraction 0.1, but a duplicated
validation table still demanded the historical 0.7 value. The strict current
source/local-artifact/known-commit receipt has file
`sha256:a474bd87130e92631b08c3227cd937e7098a67078ef8522e66d29b394efddcca`.
Its 0.319-second scope peaked at 24,743,936 bytes and four PIDs with zero
limit/OOM events, removed cleanly, and completed thermal handoff after
77.05/62 C host/GPU peaks.

The validator now reads the floor and inference fraction from the same
committed effective contract that must exactly equal the workload manifest.
A direct executable regression passes the checked-in TOML through that
validation function and proves that restoring 0.7 is rejected. This preflight
failure contains no build, model, KV, peer, request, or recovery evidence.

The next exact pushed-source run, receipt
`20260725t143631678533z-cuda-rtx4090-laptop-serving-cuda-low-memory--647fb2a614-v1`,
proved the cache and KV repair and then failed closed at the HTTP listener
preflight. The source cache released both verified shards, CUDA reported
2,231,185,408 allocator bytes free, and the server allocated 58 BF16 KV blocks,
121,634,816 bytes, on its first attempt. Before readiness, Linux returned a
425,984-byte raw `SO_SNDBUF`, or 212,992 effective bytes, for the configured
1,048,576-byte request. The server rejected the clamp and removed its private
snapshot before exit. The strict current-source/local-artifact/known-commit
receipt has file
`sha256:99b910665cbfede4a955c5fd2da82beb99986d5c93df867595e27217db94b724`.
Its scope peaked at the exact 10,737,418,240-byte host limit, recorded 7,715
limit events and zero OOM events, removed cleanly, and completed thermal
handoff after 91.05/67 C host/GPU peaks. No readiness, baseline request,
pressure peer, pressure request, or recovery request ran.

The laptop config now requests the host's attested 212,992-byte effective send
buffer without disabling the listener preflight. That value and config hash
`sha256:a55467d566cde26021b9fcd49ba960f522dad1d01c26f53c9e8d2e342fa3e915`
are part of the driver's effective contract and must exactly match the workload
manifest. A direct regression rejects restoration of the incompatible 1 MiB
request, while the desktop bootstrap retains its separate 1 MiB contract. The
revised low-memory workload is
`sha256:a0cec58ef62ac01a7210cc2001e9302581fbb4ada0bff7e91f8e2e8b4be057ac`.
Only a clean pushed-source rerun may close the low-memory gate.

That exact-source rerun, receipt
`20260725t150047861492z-cuda-rtx4090-laptop-serving-cuda-low-memory--647fb2a614-v1`,
passed listener preflight, readiness, model identity, warmup, and the 32-token
deterministic baseline, then failed before any allocation because the dedicated
peer still required `--minimum-free-mib` to be at least 1,024. The driver and
manifest passed 768, matching the reviewed hard floor. The strict
current-source/local-artifact/known-commit receipt has file
`sha256:63cab4eccfcc71be75284bad8a0c5c67627271e69596614197728546deafcf45`.
The listener returned exactly 212,992 effective bytes, both source-cache shards
released, and 58 BF16 KV blocks allocated. Both pre-pressure requests returned
HTTP 200 with 32 tokens and `finish_reason=length`; no pressure peer allocation
or pressure/recovery request ran. The scope peaked at 10,737,418,240 bytes and
66 PIDs, recorded 1,974 limit events and zero OOM events, removed cleanly, and
completed thermal handoff after 93.05/67 C host/GPU peaks.

The peer's fail-closed minimum is now the same reviewed 768 MiB floor, while its
target-minus-floor margin remains at least 256 MiB. Its standalone defaults are
also the declared 1,024 MiB target, 768 MiB floor, 256 MiB chunk, and 1,024 MiB
allocation cap. A direct argument regression accepts that exact envelope and
rejects 767 MiB. No allocation, polling, release, or cleanup check was removed.
A new clean pushed-source run is still required.

That rerun, receipt
`20260725t150911705759z-cuda-rtx4090-laptop-serving-cuda-low-memory--647fb2a614-v1`,
proved the peer can allocate and release, but exposed a WSL2 accounting
boundary. It allocated four 256 MiB chunks and released the complete 1 GiB
without error. Its process-local `cuMemGetInfo` still reported 13,005,041,664
bytes free at the low point even though the resident server's OS-level
`nvidia-smi` accounting reported 14,873,001,984 bytes used before KV
allocation. The peer therefore exhausted its bounded cap without reaching a
1 GiB process-local free target and failed before publishing readiness. The
strict current-source/local-artifact/known-commit receipt has file
`sha256:dc07aa26895c4eb65718a0a570c5d6d317863f926a06799448d0e055f64e38b0`.
The scope peaked at 10,737,418,240 bytes and 66 PIDs, recorded 2,130 limit
events and zero OOM events, removed cleanly, and completed thermal handoff
after 91.05/68 C host/GPU peaks. No pressure request or recovery request ran.

The peer now uses the same global `nvidia-smi` source required by server health
for its target, floor, hold samples, and release snapshot. It retains
`cuMemGetInfo` as explicitly labeled allocator-local evidence instead of
treating it as global residency. Readiness schema v2 requires both source
labels, a measured global free-memory drop of at least 64 MiB, a global value at
or below the effective 1,024 MiB target, and a value at or above the 768 MiB
floor. The bounded cap is 1,280 MiB, sufficient for the measured physical
capacity while leaving the independent 256 MiB target-to-floor margin. The
revised workload is
`sha256:419d37b415f4dae158f259e90c91bb12c697f80605abbd86aff14c1931526cba`.
If WSL2 does not expose the peer allocation through global accounting, the run
still fails without pressure acceptance.

The exact-source rerun from
`8d645066cb1016503658301e0fd2362a96e36345` reached readiness after one
67,108,864-byte allocation. Global peer free memory was 1,042,284,544 bytes at
readiness, never fell below 975,175,680 bytes while held, and recovered to
1,290,797,056 bytes after a clean release. The pressure request completed all
32 tokens with HTTP 200. The server sampler independently reported only
252,706,816 bytes free, so the driver correctly rejected corroboration before
the recovery request.

The retained failed receipt is
`20260725t152359537452z-cuda-rtx4090-laptop-serving-cuda-low-memory--647fb2a614-v1`
with file
`sha256:afcb43b9b11834b0f1920df5dc0164cf943f6e4a56473e93b417868266545c08`.
It is strict current-source, local-artifact, and known-commit valid. The exact
10,737,418,240-byte scope peak produced 2,995 limit events but no OOM event;
server, peer, snapshot, and scope cleanup completed, and thermal supervision
handed off after 92.05/69 C host/GPU peaks.

This is capacity-contract counterevidence, not a pressure failure hidden by a
tolerance. The device reports a 16,376 MiB physical total, while the bootstrap
caps the governor at 15,360 MiB. Kiln projects the global used count onto that
smaller total, so server health has 1,016 MiB less free capacity than the raw
peer observation despite retaining `nvidia-smi` provenance. The documented
raw 1,024/768 MiB pressure envelope and cap-adjusted health sampler therefore
cannot represent the same contract. The low-memory gate remains open until a
source repair removes that contradiction and an exact clean pushed-source run
completes sampler corroboration and deterministic recovery.

The repaired source contract removes the laptop-only 15 GiB
`memory.gpu_memory_gb` override while retaining the exact device-name,
architecture, and 15,000-17,000 MiB hardware checks. The configured 1 GiB
server floor, 768 MiB peer floor, pressure target, allocation cap, and sampler
tolerances do not change. Rather than letting the additional detected capacity
increase auto-sized KV, the config fixes the pool at the last run's measured 62
BF16 blocks and 130,023,424 bytes. The driver rejects any reintroduced capacity
override or block-count drift. Config hash
`sha256:455dee1f50c87e7d7eb2674fcf30823073500141e1f05a311b068633deb6f494`
and workload
`sha256:2c741a773ccc4f02e714b78fcb30c32c578cb83c6019e56432706f69191f2d45`
bind the correction. Runtime model attestation also requires the server's
`total_vram_bytes` to exactly equal the preflight physical-device total and
requires `kv_cache_bytes=130023424`. This is still a source contract until a
clean pushed run passes the complete gate.

The exact rerun from
`6725896cf3147cec121e825b9e2a93cec8bd1498` proved that the server now used
the full physical capacity and retained the exact 62-block/130,023,424-byte KV
allocation. The peer allocated 195,035,136 bytes, reached exactly
1,073,741,824 bytes free, observed a 1,006,632,960-byte minimum while held, and
released cleanly to 1,451,229,184 bytes. The pressure request again completed
all 32 tokens with HTTP 200.

The server sampler reported 1,349,517,312 bytes free, which is 7 MiB above the
existing target-plus-256 MiB corroboration allowance. The driver rejected it;
the recovery request did not run. The strict current-source, local-artifact,
and known-commit valid receipt is
`20260725t154633387643z-cuda-rtx4090-laptop-serving-cuda-low-memory--647fb2a614-v1`
with file
`sha256:d7a65609398c748c72bf0b1edc15751b9402e94dd02a54e1af0549f8bd2f826c`.
The 10,737,418,240-byte scope peak produced 2,857 limit events and zero OOM
events; server, peer, scope, and snapshot cleanup completed, and thermal
handoff passed after 93.05/68 C host/GPU peaks.

This second discrepancy is inside the `nvidia-smi` probe semantics. Kiln
currently queries `memory.total,memory.used` and synthesizes free as their
difference, while the peer queries `memory.free` directly. WSL2 reports a
reserved gap, so those values are not additive. The remedy is not a wider
tolerance: server sampling must query all three counters, retain their raw
diagnostics, and use the conservative minimum of reported free and
`total - used` as effective free. The low-memory gate remains open.

The repaired NVIDIA probe now requests total, used, and free in one bounded
`nvidia-smi` process. It rejects missing, extra, multi-row, zero-capacity,
non-integer, and out-of-range results. Effective free is the lower of reported
free and `total - reported_used`; effective used is recomputed from that value
so the safe snapshot remains internally consistent. Raw reported used and free
remain separately exposed in memory observations, including the WSL2 reserved
gap. Focused tests cover both directions of disagreement. No pressure target,
floor, tolerance, or lifecycle bound changed.

The unchanged workload passed from exact clean pushed source
`385bcf20874ff93ebbd91b4099935c554d03ad15`. Receipt
`20260725t155941791666z-cuda-rtx4090-laptop-serving-cuda-low-memory--647fb2a614-v1`
is strict current-source, local-artifact, and known-commit valid; its file hash
is
`sha256:c6fb85c2ffa0d437e7b0bbacdb2e8811b5c7ceb8e8a42a5d895e71924dccb4b2`.
The server attested the 17,171,480,576-byte physical capacity,
14,975,762,432-byte resident model state, and exact
62-block/130,023,424-byte BF16 KV pool. The peer allocated 260,046,848 bytes
in two bounded allocations, reached 1,007,681,536 bytes free, observed
940,572,672 bytes at its minimum across 17 held-pressure samples, and released
cleanly to 1,449,132,032 bytes. The unchanged server sampler corroborated the
pressure envelope and recovered in 1.445 ms. Baseline, pressure, and recovery
requests each returned HTTP 200 with 32 completion tokens, and the recovery
token IDs plus finish reason exactly matched the deterministic baseline.

The 10 GiB zero-swap scope reached its exact 10,737,418,240-byte bound and a
66-PID peak. Its 5,994 `memory.max` events did not produce an OOM, OOM kill, or
group kill. Server and peer exits were graceful; process-group, snapshot, and
scope cleanup completed without residue. Continuous supervision completed its
safe handoff after 92.05 C host and 69 C GPU peaks, below the 95/85 C hard
limits. This accepted receipt and the retained lifecycle receipt close the
bounded WSL2 Laptop GPU low-memory admission, explicit rejection, held-pressure
request, release recovery, deterministic replay, and cleanup gate. They do not
qualify the performance matrix, eight-hour soak, native Linux, or desktop RTX
4090.

### CUDA Serving Bootstrap Handoff

After the environment, core-correctness, and memory-lifecycle receipts pass and
are pushed, materialize that machine's host thermal policy and build the CUDA
server. Do not reuse Strix Halo sensor names or thresholds. The checked-in Kiln
bootstrap inputs are:

- `kiln-cuda-rtx4090-laptop-serving-bootstrap-v1` for the 16 GiB Laptop GPU;
- `kiln-cuda-rtx4090-desktop-serving-bootstrap-v1` for the 24 GiB desktop GPU;
  and
- `vllm-cuda-rtx4090-serving-bootstrap-v1` for the common bounded vLLM arm.

Run the labeled-sensor inventory first. Create a content-hashed
`hard_limit_only` policy from the unique host-package selector and reviewed
machine limits, then commit it. Pass that same policy to
`scripts/cargo-bounded.sh --host-thermal-policy` so source compilation also
waits for the policy's stable handoff target and remains thermally contained.
The wrapper rejects a conflicting set of legacy thermal
environment fields. The complete policy-materialization, bounded CUDA build,
typed-config preview, immutable vLLM runtime-manifest, NVML UUID selection,
single-concurrency bootstrap, comparison, failure, and expansion procedure is
in [Serving Benchmark Protocol](SERVING_BENCHMARK_PROTOCOL.md#rtx-4090-serving-bootstrap).

The WSL2 laptop is the explicit exception to the Linux-hwmon preparation step:
use the already reviewed
`qualification/host-policies/rtx4090-laptop-wsl2-cgroup-pacing-v2.json` with the
qualification runner. Its outer Windows/NVML supervisor also contains the
source build. Do not pass this WSL policy to
`cargo-bounded.sh --host-thermal-policy`, whose schema intentionally accepts
only a Linux hwmon selector.
Serving benchmark driver v21 and campaign schema v9 extend that same rule to
owned benchmark launches. A committed WSL2 workload passes the policy as
`--external-wsl2-thermal-policy`; the child driver revalidates the live private
network, Landlock, exact UID-qualified systemd scope, memory/swap/PID/OOM
controls, and usage-feedback CPU controller. The nested benchmark receipt
requires its enclosing qualification receipt, which remains the sole authority
for final Windows/NVML thermal evidence, cgroup accounting and removal, process
cleanup, and safe handoff. Running the external mode directly or attaching to a
pre-existing server is rejected.
Use `scripts/qualification/capture_vllm_runtime_manifest.py` with the tracked
vLLM launch JSON; do not retype its inference arguments. The tool requires a
clean commit, two byte-identical strict-valid captures, bounded child output,
and a new destination before it publishes the exact runtime manifest. The
launch JSON must be a tracked regular file with bytes identical to `HEAD`.
On this WSL2 laptop, pass the committed boundary as
`--wsl2-thermal-policy`. Each capture then runs in a separate
Windows-thermal-zone/NVML wrapper lifecycle and must complete stable handoff
before the next starts. The command reports both strict-valid thermal records;
a trip, omitted policy, uncommitted policy or supervisor bytes, missing event,
timeout, or incomplete cooldown prevents publication.
Each pass also uses the same reviewed WSL2 private network/PID/mount namespace,
Landlock interop denial, 10 GiB memory/zero-swap/512-PID user scope, group OOM
handling, and 50-percent usage-feedback CPU controller as the qualification
runner. The v2 policy makes that controller freeze the complete scope at
80 C host or 75 C GPU and resume only after three consecutive samples at or
below 75.05/70 C. Each pause has a 300-second deadline. The independent outer
supervisor retains the unchanged 95/85 C hard limits and post-exit handoff,
and streams its accepted samples over a one-way inherited pipe. The scope
controller does not duplicate the expensive Windows/NVML probes, and the
contained payload does not inherit the pipe descriptor. Every transition must
round-trip both the requested `cgroup.freeze` value and the kernel-reported
`cgroup.events` frozen state.
The capture result retains both scope records, including pacing samples,
pause counts and durations, peaks, and inactive completion. Missing or
reordered scope events, a policy-binding or control mismatch, an incomplete
or timed-out pause, excess CPU accounting, any memory-limit/OOM event,
nonzero child, or scope residue prevents publication. The v1 hard-limit-only
document remains parseable for historical receipts but is rejected for new
WSL2 CUDA qualification and manifest capture.
The laptop performance launch pins
`--max-provenance-read-mib-per-second=32`, cumulatively pacing all
launcher-owned model/snapshot/adapter/runtime hashing and applying the same
ceiling to the fresh child runtime recheck. This is startup-only policy and
does not enter timed request throughput.

The first retained laptop performance runtime manifest is
`qualification/runtime/vllm/cuda/rtx4090-laptop/performance-v1.json`, captured
twice from exact clean pushed source
`2eb37adb9cd7279cc1472f5763ed939fbbe55add`. Its file hash is
`sha256:50d46bd54df16f1ea9095dace7656708b7347db3591ad8ebc74d1238d284d125`
and its installed-runtime content hash is
`8b3b7273f3e031c427591a4c4447e7541e85023edb92bdf5da51a1882e5e5abb`.
Both strict-valid captures produced the same 2,608 bytes and bind vLLM 0.23.0,
the closed Qwen3.5-4B content, 32,768-token context, top-K 20, BF16, and the
reviewed performance launch.

The two scopes completed in 1,279.766 and 1,075.023 seconds. They used
42,063,074/639,883,214 and 43,962,310/537,511,551 CPU microseconds, peaked at
10,106,621,952 and 10,551,422,976 bytes with 27 PIDs each, recorded zero
memory-limit/OOM events, and were both removed. Thermal pacing completed all
39 and 41 pauses, totaling 958.839 and 466.370 seconds; the longest pauses were
149.116 and 42.091 seconds. Host/GPU lifecycle peaks were 94.05/66 C and
93.05/64 C, below the 95/85 C hard limits, and both outer supervisors completed
stable handoff. This closes the runtime-manifest prerequisite only. It is not a
server startup, request, performance-matrix, soak, native-Linux, or desktop-4090
claim.

Every model-bearing WSL2 CUDA run now brackets its case with two additional
model-fingerprint lifecycles. The runner reads at a fixed 32 MiB/s and places
each initial and final fingerprint inside its own private namespace, 10 GiB
scope, v2 cgroup-pacing controller, outer Windows/NVML hard-limit supervisor,
and stable handoff. Their bounded JSON and supervision streams are retained
with the parent receipt. The case starts only after the initial scope is
removed, and final source/commit validation occurs only after the final scope
is removed. The case itself remains a separate lifecycle, so neither long
preflight nor post-run hashing can execute outside the WSL2 safety boundary.
Both fingerprint launches receive the same closed runner base environment plus
the exact private-containment mechanism binding required by the scope
controller; an omitted or different binding fails before model I/O.

The first source-bound performance checkpoint is the complete five-profile c1
pair:

```bash
python3 scripts/qualification/run.py \
  --variant cuda-rtx4090-laptop-c1 \
  --host-id rtx4090-laptop \
  --model .qualification/cuda-rtx4090-laptop/performance-model-v1 \
  --model-id Qwen/Qwen3.5-4B \
  --wsl2-thermal-policy qualification/host-policies/rtx4090-laptop-wsl2-cgroup-pacing-v2.json \
  qualification/workloads/serving-cuda-performance-c1-v1.json
```

It builds the exact committed CUDA server once, then runs `greedy-short`,
`api-default-sampled`, `long-prefill`, `prefix-hit`, and `mixed` through five
owned Kiln lifecycles. Only after all five strict receipts pass does it run the
matching five owned vLLM lifecycles against those exact references. Generated
campaign artifacts live under the commit-qualified ignored
`.qualification/serving/cuda-rtx4090-laptop-performance-c1-v1/` root and are
never reused. Retain the parent qualification receipt and all ten validated
nested receipts before making a c1 request or performance claim. This
preparatory workload does not itself close the open matrix or soak gates.

The first corrected exact invocation from clean pushed source
`c903f7dd97c7250c862db7a393a774e7ca48261e` is retained as failed receipt
`qualification/receipts/cuda/rtx4090-laptop/20260725t203426391915z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`
with file
`sha256:06618553060c9405de939d7f518b1810b3067a282e27cd698148d224f17894ad`.
The initial fingerprint passed, but the current-source CUDA build accumulated
1,551.390 thermally paused seconds inside its 1,616.727-second scope. Its
fifty-first pause reached 301.470 seconds, exceeding the unchanged 300-second
policy limit and failing closed before server startup or campaign output. The
case stayed below the independent hard limits at 93.05/66 C host/GPU, peaked
at 1,011,679,232 scoped bytes and 14 PIDs, and recorded zero memory-limit/OOM
events. The independently supervised final fingerprint completed and was byte
identical to the initial fingerprint. All three scopes were removed, all outer
stable handoffs completed, and no process or campaign artifact remained. This
valid receipt is thermal-boundary counterevidence, not performance evidence.
The failure path also revealed a diagnostic-only second close of the thermal
telemetry descriptor after the causal failed event. The controller now closes
that descriptor exactly once and preserves the primary failure. Commit and
push this checkpoint before an unchanged exact c1 retry.

The unchanged retry from exact clean pushed source
`1715532c56bc7241797691ba2ee8df38f3e28be6` was rejected before substantive
fingerprint reads. Its initial scope remained in a verified thermal freeze for
300.520 seconds and failed at the same unchanged per-pause limit. Pacing peaked
at 82.05/65 C host/GPU; the independent outer supervisor completed stable
handoff at 75.05/63 C after 228 samples. The Linux scope was removed and no
child, campaign root, or source change remained. The corrected failure output
preserved the pacing cause without the former descriptor-close diagnostic.
The runner does not create a receipt until its initial fingerprint succeeds,
so this is explicitly non-receipt console counterevidence and makes no model,
server, request, or performance claim. Do not rerun while the idle host cannot
reach the reviewed resume threshold. Allow an independent cooldown, then retry
the unchanged c1 workload from another clean pushed checkpoint without raising
the thermal thresholds or timeout.

After a five-minute independent cooldown, an unchanged invocation from exact
clean pushed source `8444359ccb5cd8d70bb912c20bef68ada61f91fb` advanced to an
observed 807,174,144-byte fingerprint scope peak before freezing. Its complete
three-PID scope stayed frozen for 300.851 seconds and failed at the unchanged
per-pause limit. Host/GPU peaks were 85.05/64 C, and the outer lifecycle
completed stable handoff at 76.05/63 C after 246 samples. No case, receipt,
campaign root, process, scope, or source residue remained. This is a second
consecutive idle-host rejection and does not justify an immediate rerun.

Initial fingerprint failures now retain a strict failed parent receipt rather
than deleting the run directory. The receipt records `model: null` because no
complete fingerprint exists, marks environment identity unavailable because
the collector was deliberately not started, fails every declared required
case without executing it, leaves effective config empty, and retains the
exact source/workload binding, policy snapshot, bounded fingerprint stdout,
complete supervision stderr, and run config. Passed model-bearing receipts
still require the full weight/config/tokenizer/template identity. A failed
receipt with no model is pre-model boundary evidence only; it cannot be cited
for model integrity, device identity, case execution, or performance.

After a ten-minute independent cooldown, the unchanged invocation from exact
clean pushed source `98f7c2db72523b35c042818a457ea4fdfa637a11`
exercised that retention path. It published failed parent receipt
`qualification/receipts/cuda/rtx4090-laptop/20260725t221058934964z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`
with file
`sha256:c8d8e3364d46e84f7b89eac810dc1ccbd036280bce49261c4f69017128802bee`.
The incomplete initial fingerprint scope ran for 336.353 seconds, peaked at
1,253,879,808 bytes and four PIDs, used 1,314,771 of 168,176,288 allowed CPU
microseconds, and recorded zero memory-limit or OOM events. Two verified
freezes totaled 334.129 seconds; the second reached 300.510 seconds and failed
at the unchanged per-pause bound after an 86.05/64 C host/GPU peak. The scope
was removed, the outer supervisor completed stable handoff at 76.05/63 C, and
no case, final fingerprint, campaign root, or process remained. The strict
known-commit/local-artifact validator accepts the receipt and all five hashed
artifacts. Its `model: null`, unavailable environment, empty effective config,
and failed unexecuted c1 result make it retained thermal-boundary
counterevidence only, not a model or performance result.

After the portable endurance handoff was pushed, a policy probe read 74.05/63 C
host/GPU and the exact c1 invocation from clean pushed source
`04d2c229bab5c5d668b77166b7900fbac5717342` was allowed to proceed. Its formal
preflight read 78.05/63 C, and it retained failed parent receipt
`qualification/receipts/cuda/rtx4090-laptop/20260725t231031150626z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`
with file
`sha256:d77cfb5a1a9838c83b8ac5addb184aa87989b53530e4a88c4a95bcf417caabfd`.
The 714.265-second initial fingerprint scope advanced to a
4,264,914,944-byte/four-PID peak and used 4,184,272 of 357,132,684 allowed CPU
microseconds. It completed three thermal pauses; four pauses totaled 709.341
seconds, and the fourth failed at 300.648 seconds under the unchanged
per-pause bound. Host/GPU peaks were 94.05/66 C, below the independent 95/85 C
hard limits. Memory-limit, OOM, OOM-kill, and swap-limit events stayed zero.
The scope was removed and outer supervision completed its three-sample stable
handoff at 78.05/64 C after 534 samples.

The receipt and all five local artifacts pass strict known-commit and
local-artifact validation. The fingerprint output is empty, `model` remains
null, environment collection did not start, effective config is empty, and
the required c1 result failed without execution. No final fingerprint,
campaign root, process, or scope survived. This is deeper pre-model thermal
counterevidence, not a model, device, request, or c1 performance result. The
host approached its hard limit and then could not reach the unchanged resume
boundary within 300 seconds; do not immediately retry the same laptop path or
weaken the policy to force progress.

Reviewing that rejection exposed an observer effect outside the measured
scope. The fingerprint itself used only 4.184 CPU-seconds over 714.265 seconds,
but the outer guard launched a new PowerShell process, formatted-data WMI
query, and `nvidia-smi` process for every one-second sample. A Windows process
snapshot immediately after a probe showed `WmiPrvSE` at 21 percent CPU, and an
idle eight-sample trend oscillated from 73.05 to 83.05 C. The source now keeps
one exact-instance Windows performance-counter process and one exact-UUID NVML
handle alive for the complete preflight/runtime/handoff lifecycle. It preserves
the policy content hash, one-second cadence, 80/75 C pacing, 75/70 C resume,
95/85 C hard limits, telemetry schema, and three-sample safe handoff.

A 60-sample live sampler probe completed in 60.182 seconds. The persistent
PowerShell bridge accumulated 0.000 WSL-visible CPU-seconds, the complete host
range was 73.05-81.05 C, displayed samples 15/30/45/60 read
73.05/74.05/74.05/73.05 C, GPU temperature remained 63 C, and the sensor
process exited without residue. A separate live guarded no-op exercised the
exact outer supervisor, inherited telemetry pipe,
10 GiB/zero-swap/512-PID user scope, 50-percent feedback controller, private
namespace, pacing state, child handoff, and scope removal. It started at
75.05/63 C, recorded an 84.05/63 C outer peak, completed at 75.05/63 C after
five samples, and left no process, PowerShell, or user-unit residue. These are
live dirty-source boundary probes, not qualification receipts or c1 evidence.
Commit and push the repair before another exact c1 attempt.

The exact retry from clean pushed observer-repair source
`1e22141cfd2a60efe4e00d674dcd0190ec95a348` retained failed receipt
`qualification/receipts/cuda/rtx4090-laptop/20260725t235849138625z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`,
file
`sha256:bad5e9d085176e035617f27ad687b60bb40cdf29485dd4a95eb886b45f3b10bd`.
The admission probe immediately before launch read 74.05/64 C, but formal
preflight after the independent CPU-identity query read 86.05/64 C. The
751.564-second fingerprint scope reached 9,387,716,608 bytes and four PIDs and
used 13,122,123 of 375,782,151 allowed CPU microseconds. Three pacing pauses
totaled 459.892 seconds; two completed, while the third remained active and
reached 300.867 seconds. Scope and outer peaks were 90.05/64 C. All
memory-limit, OOM, OOM-kill, and swap-limit events remained zero. The scope was
removed and the outer supervisor completed stable handoff at 74.05/64 C after
752 total samples.

The strict known-commit/local-artifact validator accepts the receipt and all
five bounded files. The fingerprint output remains empty, `model` is null,
environment collection and the case did not start, effective config is empty,
and the required c1 result failed without execution. No final fingerprint,
campaign root, server, Cargo/rustc process, PowerShell process, or scope
survived. Persistent sampling removed most of the former outer observer work
and allowed the bounded reader to reach a 9.388 GB scoped peak, but the child
still started above the pacing threshold and could not hold three samples at
or below 75 C within the unchanged pause limit. This is pre-model
counterevidence only. The next repair must remove the pre-sampler WMI
CPU-identity query and require a stable pacing-resume boundary before child
launch; do not weaken the policy.

The follow-on source repair removes `_powershell_json` and its fresh
`Get-CimInstance Win32_Processor` process entirely. The already-persistent
Windows sensor process reads `ProcessorNameString` directly from the local
machine registry, binds it in the exact ready record, and still resolves the
same thermal instance/counters. With a v2 policy, outer supervision now waits
for `pacing.resume_stable_samples` consecutive host/GPU readings at or below
the existing resume thresholds before emitting `preflight`, creating the
scope, or launching any child. Those wait samples contribute to the outer
sample count and peak. Sensor failure, a hard-limit sample, or the unchanged
300-second pacing timeout rejects admission without child work.

A live dirty-source handshake read 75.05, 77.05, 74.05, 74.05, and 74.05 C
host with a fixed 64 C GPU and exited without sensor residue. The exact guarded
no-op then admitted at 74.05/64 C only after the stable boundary, completed
with zero scope pauses, 51,108 of 59,376 allowed CPU microseconds, a
13,934,592-byte/four-PID peak, zero memory/OOM events, scope removal, and a
74.05/64 C handoff. Its outer record included all nine samples and the earlier
77.05 C pre-admission peak. This proves only the repaired dirty-source
admission and lifecycle. Commit and push before another exact c1 attempt.

The exact clean pushed-source retry at
`7e3406917ec369a2bf10f83918d637944932d805` retained failed receipt
`qualification/receipts/cuda/rtx4090-laptop/20260726t002130041819z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`,
file
`sha256:bac0abc5fc9a8469738c13758c4a4d57619f1aa3633ad6744189773c4c94b26e`.
Formal preflight correctly began at the stable 74.05/64 C boundary. The
475.132-second initial fingerprint scope reached 5,503,045,632 bytes/four PIDs
and used 5,939,727 of 237,565,837 allowed CPU microseconds. Four pauses totaled
442.090 seconds; three completed and the fourth remained active for 300.360
seconds. Host/GPU peaks were 90.05/64 C. Every memory-limit, OOM, OOM-kill, and
swap-limit event stayed zero. The scope was removed and outer supervision
completed its 74.05/64 C stable handoff after 483 samples.

The strict validator accepts the receipt and all five local artifacts. Model
fingerprint output is empty, `model` is null, environment and case collection
did not start, effective config is empty, and no build, server, request,
campaign root, process, sensor, or scope survived. This establishes that stable
admission works and isolates the remaining heat to the fingerprint lifecycle.
Source audit found that `_ReadRateLimiter` uses an absolute cumulative
wall-clock deadline. A cgroup freeze advances that clock while the reader
cannot run, so resume gives it accumulated read credit and permits an
unthrottled catch-up burst. The next repair must rebase a stale deadline on
resume so idle or frozen time never creates credit, while retaining the exact
32 MiB/s maximum, two integrity passes, thermal limits, and workload semantics.

The limiter now keeps an explicit next deadline rather than deriving every
deadline from process start and total bytes. It still counts every byte through
one limiter across discovery, all files, and both integrity passes. Before
charging each chunk, however, it rebases a deadline that is older than the
current monotonic time. The current chunk must then wait its complete
`bytes / configured_rate` interval. Cgroup freezes, scheduler delay, parsing,
and other idle work can make the effective rate lower but can never grant
catch-up credit or make it exceed the configured maximum. The read chunk
remains 8 MiB and the laptop cap remains 32 MiB/s.

The deterministic limiter regression paces two consecutive half-MiB chunks at
one MiB/s from time 10 to 11, jumps the clock to 100 as a synthetic external
pause, then requires the next half-MiB to consume the full interval through
100.5 while retaining 1.5 MiB total accounting. This portable test does not
establish laptop model identity or c1. Commit and push the repair before the
exact retry.

The exact retry from clean pushed limiter-repair source
`dba9b9062b9cc7f53e29bbc65f444f33c98fae12` retained failed receipt
`qualification/receipts/cuda/rtx4090-laptop/20260726t003539048501z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`,
file
`sha256:fff67591c5fce479fbda102b104b02e1d9c6164e62d0a8ca551af0cc1c08ff0f`.
For the first time on this c1 path, the initial model fingerprint, all 16
required environment capabilities, the case boundary, and the final model
fingerprint completed. The two 824-byte fingerprint artifacts are byte- and
hash-identical at
`sha256:31b656e8b4992a7dd56d31f0c4cf18fcb360601e69af4f538b780c40de585612`.
They bind both exact weight shards totaling 9,319,828,096 bytes plus the
configuration, tokenizer, and chat-template digests for
`Qwen/Qwen3.5-4B`.

The 632.493-second initial fingerprint scope used 17,454,711 of 316,246,590
allowed CPU microseconds, peaked at 9,388,126,208 bytes/four PIDs, and
completed two pauses totaling 57.216 seconds. The 929.768-second final scope
used 19,342,218 of 464,884,048 allowed CPU microseconds, peaked at
9,387,827,200 bytes/four PIDs, and completed three pauses totaling 351.605
seconds. Both recorded zero memory-limit, OOM, OOM-kill, or swap-limit events
and removed their scopes. The retained environment fixes the exact Laptop GPU
UUID/name/17,171,480,576-byte capacity, NVIDIA 596.36 driver, CUDA 12.4
toolkit, ext4 filesystem probes, WSL2 identity, system and user systemd, memory
accounting, private network/process containment, and thermal authorities.

The required c1 case still failed before service startup. Its source build
reached `kiln-server` compilation, then the driver rejected
`CUDA performance build exceeded 1800.000 seconds`. The enclosing
1,802.090-second scope used 185,237,153 of 901,045,144 allowed CPU
microseconds, peaked at 2,123,210,752 bytes/14 PIDs, recorded zero memory/OOM
events, and completed all 140 pauses totaling 1,615.144 seconds. Host/GPU
peaks were 94.05/66 C, below the unchanged hard limits. Scope and outer
supervision both removed cleanly and completed a stable handoff.

This is model-integrity, environment, containment, thermal-pacing, cleanup,
and source-build counterevidence only. No build completion, server startup,
request, nested campaign receipt, throughput, c1 acceptance, or endurance
claim follows. The strict known-commit/local-artifact validator accepts the
receipt and all 11 hashed files, and no Cargo/rustc, Kiln, driver, PowerShell,
sensor, campaign, or scope residue remained. The failure isolates a timeout
accounting defect: verified cgroup-freeze intervals consumed the inner
1,800-second source-build wall deadline, leaving only about 185 CPU-seconds for
the compile. Retain and push this evidence before making the build bound
pause-aware; do not weaken the thermal policy or resource boundary.

The corrected c1 workload has file
`sha256:d54e339986d6962d55eea7a893b1d9e9bb5038535129fd4d6d0715950d45fb53`.
The source-build active allowance remains 1,800 seconds. A separate 14,400
seconds of verified WSL2 thermal-pacing overlap may extend it, while an
independent 16,200-second wall deadline remains absolute and inside the
unchanged 43,200-second case scope. `build_duration_ms` now reports wall time
minus only verified thermal overlap. CPU-feedback freezes, scheduler delay,
and ordinary idle time continue to count against that active duration.

The contained parent owns the runtime deadline and launches
`cargo-bounded.sh` in a new process session. Every poll opens the private
mode-0400 transition stream without following the final path, checks its
normalized absolute path, owner, parent permissions, regular-file type, 8 MiB
bound, ASCII/strict-JSON encoding, exact field set, policy hash, sequence,
pause identity, monotonic interval, temperatures, duration, and complete
start/completion pairing. Only completed interval overlap after build start is
credited. Missing, unsafe, partial, unpaired, reordered, duplicate-field,
non-finite, or drifted evidence fails closed and sends TERM then KILL to the
complete Cargo/rustc process group if necessary. The former transient-service
runtime setting is no longer presented as an authority in delegated mode.

A live dirty-source guarded probe exercised the exact persistent Windows/NVML
outer supervisor, private network/PID/mount namespace, Landlock boundary,
10 GiB/zero-swap/512-PID/50-percent-CPU scope, inherited event path, strict
reader, status handoff, scope removal, and final thermal handoff. The reader
accepted a valid empty stream while no pause occurred. The 0.170-second scope
used 74,448 of 85,028 allowed CPU microseconds, peaked at 15,302,656 bytes and
four PIDs, and recorded zero memory/OOM events. Outer supervision started and
ended at 73.05/63 C, retained a 78.05/63 C peak across nine samples, and left
no process, sensor, or unit residue. This proves the dirty-source reader
composition only; commit and push before another exact c1 attempt.

The exact clean pushed retry from
`57d297e0f7835d98ba12f3b4b728603a0f344caa` retained failed parent receipt
`qualification/receipts/cuda/rtx4090-laptop/20260726t015018730815z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`,
file
`sha256:2cce798022fce948a78e855ff35a175f667b0427119fab1cf24fe8527104f2cd`.
The pause-aware source build completed successfully. It consumed 2,161.060
wall seconds, including 1,987.708 seconds of exact completed pacing overlap
and 173.352 active seconds, and produced `target/release/kiln` at
`sha256:38da3668a40b3df1b92c89c66cefbe7063665ab08357c1222445ba3fad36ac3e`.
This establishes that the revised build authority clears the former timeout
without crediting ordinary delay or weakening any resource or thermal limit.
It is not c1 acceptance.

The first two Kiln profiles each wrote a strict-valid failed benchmark receipt,
now retained as
`benchmarks/receipts/cuda/rtx4090-laptop/20260726t030335-cuda-rtx4090-laptop-c1-greedy-short-startup-race-57d297e0-v1.kiln.json`
and
`benchmarks/receipts/cuda/rtx4090-laptop/20260726t032626-cuda-rtx4090-laptop-c1-api-default-startup-race-57d297e0-v1.kiln.json`.
Their file hashes are
`sha256:4e4fc71f882aba58edb75c88ca0cc02bf6d881bcae1936a56f9b53d0ab5c3eda`
and
`sha256:e7fe629fb50d1e3094c98114b12af73b12105a415e08015f8dcdbbfbfc507640`.
Both passed final model, repository, and runtime-artifact checks, but completed
zero requests. External-WSL owned-server mode launches the server and then
uses a one-shot `/v1/models` probe rather than the existing owned readiness
loop; both probes raced startup and recorded connection refusal. Their
shutdown waits also used undifferentiated wall time, crossed authenticated
cgroup freezes, and escalated to SIGKILL. These are benchmark lifecycle
defects, not server output or OOM evidence.

Before the third profile completed, the outer hard authority rejected its
persistent Windows input because `Temperature` and
`HighPrecisionTemperature` disagreed. It sent SIGTERM to the complete case
boundary. The 5,023.139-second scope used 250,572,502 of 2,511,569,367 allowed
CPU microseconds, peaked at 10,737,639,424 bytes and 33 PIDs, and recorded
137,891 `memory.max` reclaim events with zero OOM, OOM-kill, or group-kill
events. All 155 pacing episodes completed, totaling 3,613.488 seconds; the
outer peak was 94.05/66 C and stable handoff completed at 74.05/64 C after
5,009 samples. Initial and final model fingerprints remained byte-identical,
all 16 environment capabilities passed, every scope was removed, and no
Cargo, rustc, Kiln, benchmark, runner, sensor, or unit process remained.

The parent and nested receipts are retained counterevidence only. They prove
build completion, strict containment, fail-closed sensor handling, integrity,
and cleanup. They do not prove server readiness, a request, a profile pass,
vLLM comparison, c1 performance, or endurance. Repair external-WSL readiness
and shutdown accounting against the same authenticated pacing stream. Diagnose
the Windows field mismatch as malformed input, provider drift, or a true
invalid boundary before changing its interpretation; do not bypass the
independent hard-limit authority.

Driver v21 closes the owned-lifecycle defect without changing the launch
documents, workload, thermal policy, or resource limits. Every owned launch now
uses the retrying readiness loop even when the thermal guard is external.
External-WSL startup, TERM grace, post-leader process-group drain, and KILL
grace count wall time minus only exact completed intervals from the existing
controller-authenticated pacing stream. Evidence is reread at projected active
deadlines rather than on every poll. Invalid evidence rejects the run, and a
shutdown-accounting error still performs an independent wall-bounded SIGKILL
and process-group drain before it is reported. This prevents a legitimate
controller freeze from causing an immediate readiness failure or forced
shutdown while preserving bounded cleanup.

The Windows mismatch was an observation-tearing defect in the persistent
sensor process, not grounds to relax the cross-field check. Four independent
`PerformanceCounter.NextValue()` calls could sample different provider updates
and combine them as one row. The v2 host-counter protocol now performs one
`PerformanceCounterCategory.ReadCategory()` per requested sequence and indexes
the exact `\_TZ.THRM` instance for all four required counters from that single
category snapshot. It requires `NumberOfItems32` raw counters, one identical
positive `TimeStamp100nSec` across all four values, and a strictly advancing
timestamp between rows. The existing whole-Kelvin versus tenths-Kelvin
agreement tolerance, exact CPU/zone identity, hard limits, pacing thresholds,
sample cadence, and fail-closed behavior are unchanged.

Live dirty-source diagnostics support that normalization. A direct atomic
snapshot returned raw `Temperature=354` and
`HighPrecisionTemperature=3542` with the same timestamp and counter type.
Ten 50 ms diagnostic reads advanced coherently. The production persistent
reader then completed 60 samples in 59.034 seconds, from 78.05 C to 72.05 C
with a 71.05-78.05 C range and no PowerShell residue. A subsequent exact
guarded no-op, after supplying the runner-owned network binding, admitted at
71.05/62 C and completed a 0.159-second scope using 71,685 of 79,612 allowed
CPU microseconds, with a 15,298,560-byte/four-PID peak, zero memory/OOM events,
scope removal, 72.05/62 C handoff, and an 80.05/62 C outer peak across nine
samples. These are observer, containment, and cleanup probes only; they do not
establish server readiness, a request, c1 performance, or endurance.

The repaired benchmark driver is
`sha256:c02748e6e5134f17f186c9599251d6243be14f9b3673f0361fa33b9d06a286b3`;
the atomic WSL2 supervisor is
`sha256:32995b7d53b3cad5c2b1ba432db9f369557ceab14f21c2ff355f02720f45a03d`.
All 94 focused lifecycle/pacing/sensor tests and all 836 qualification-tooling
tests pass. All 173 compact qualification receipts, 53 detailed serving
receipts, and 17 specialized oracle results remain strict-valid, including the
two retained v20 startup failures. Nine documentation-builder tests pass with
only the Chromium-dependent case unavailable; the 55-document/five-asset build
and assembled static smoke pass. Commit and push this repair before the next
exact c1 invocation; the performance and endurance gates remain open.

The exact retry from clean pushed source
`853374d4b65734612461164e47bebe90ce742c47` retained failed parent receipt
`qualification/receipts/cuda/rtx4090-laptop/20260726t041101771428z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`
with file
`sha256:0fac3dbbe3d582373d6374cf3175569cbe70c9c26389b96bffccb52f969c3b66`.
It passes structural, current-source, and local-artifact validation. Initial
and final model fingerprints are identical, all 16 environment capabilities
are available, and the cached CUDA build reproduced binary
`sha256:38da3668a40b3df1b92c89c66cefbe7063665ab08357c1222445ba3fad36ac3e`.

Driver v21 then proved the repaired owned lifecycle with a real request. The
strict-valid passed nested receipt is
`benchmarks/receipts/cuda/rtx4090-laptop/20260726t045839-cuda-rtx4090-laptop-c1-greedy-short-853374d4-v1.kiln.json`
with file
`sha256:af34bfb0490d548415aabbdb23e6cca5eb8e2a9e567a5bdb19b6bb716b082bdf`.
The owned server waited through the bounded load, completed one warmup and one
64-token measured request, peaked at 15,577,939,968 device-memory bytes below
the 16.5 GB limit, accepted SIGTERM, exited zero without force in 0.354
seconds, removed its private snapshot, drained its process group, and passed
all final identity checks. Thermal freezes remain deliberately visible in
client latency: measured TTFT was 526.8 ms, but p99 ITL and E2E were
88,268.2 and 195,313.1 ms, leaving raw SLO goodput at zero. This profile-level
receipt is retained evidence, not an accepted paired c1 result.

The second Kiln profile reached full snapshot, 32-layer upload, and
post-upload verification before outer scope authority rejected its 79th
pacing interval. The scope completed 78 pauses totaling 2,474.186 seconds,
but the final pause lasted 300.239 seconds and ended at 75.05 C, above the
exact 75.00 C host-resume target. The scope used 302,311,575 of
1,724,275,706 allowed CPU microseconds, reached the exact 10 GiB memory cap
with 43,807 reclaim events, peaked at 68 PIDs, and recorded zero OOM,
OOM-kill, or group-kill events. Host/GPU peaks were 92.05/67 C below the
95/85 C hard limits; outer supervision handed off at 74.05/64 C and removed
the scope. The timeout correctly prevented the campaign from producing its
required case result or starting vLLM. It also interrupted the second owned
server before graceful snapshot cleanup, so the temporary private snapshot
required explicit post-run removal. Do not rerun this unchanged gate. Preserve
the exact hard limits and first determine a fail-closed response to the
host-counter's 0.05 C representable boundary and abrupt timeout cleanup.
After retention, all 174 compact receipts, 54 detailed serving receipts, and
17 oracle results validate. Nine of ten documentation-builder tests pass with
only local Chromium unavailable; the 55-document/five-asset build and
assembled static smoke pass.

The follow-on policy revision preserves the 80/75 C pacing starts, 300-second
pause deadline, three-sample hysteresis, 85/75 C safe handoff, and independent
95/85 C hard limits. It changes only the host resume value from 75.00 to
75.05 C, the exact nearest value representable after converting the Windows
tenth-Kelvin counter to Celsius. The content hash is now
`sha256:d389a7f632baab0448bd41efc205349dee4ff3944152b48cf17e52866322e3e9`.
All current performance and endurance bindings use that exact hash.

A pacing timeout now unfreezes the scope and interrupts only current leaf
processes through membership-rechecked pidfds. As each leaf exits, the next
supervisor layer is interrupted, giving Python `finally` blocks and owned
server signal handlers a bounded opportunity to delete snapshots and drain
their process groups. The cleanup is capped at 75 wall seconds; any survivor
still reaches unconditional `cgroup.kill`, and the original pacing timeout
remains the reported failure. Other hard-limit, telemetry, runtime, and
accounting failures retain their immediate kill behavior.

The repaired scope controller is
`sha256:df75f1ae712f289491d744ca9da5a22b89b4b43b28ce384583067676b740524c`.
All 841 qualification-tooling tests and all 27 workload manifests pass. A live
two-process scope probe restored normal Python interrupt handling, then proved
leaf cleanup before parent cleanup, successful scope removal during the final
cgroup-read race, and no process, scope, or marker residue. A separate guarded
no-op exercised the revised policy with a 0.124-second scope, 13,643,776-byte
and four-PID peaks, zero memory-limit/OOM events, 73.05/63 C start,
82.05/63 C outer peak, 74.05/63 C handoff, and clean removal. These are
dirty-source boundary checks, not c1 or endurance evidence.

The exact clean-source retry at
`1a5a67cee37fa22736dd38184e6da7e43b0feb3e` retained failed parent receipt
`qualification/receipts/cuda/rtx4090-laptop/20260726t060631471251z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`
with file
`sha256:b9c860152ac91020ce5cf19243fd6b224586c0dd93c7cadde9acc15daed8b31d`.
It passes structural, current-source, local-artifact, and known-commit
validation. Initial and final model fingerprints are identical, all 16
environment capabilities are available, and the cached build reproduced the
exact CUDA binary at
`sha256:38da3668a40b3df1b92c89c66cefbe7063665ab08357c1222445ba3fad36ac3e`.

This retry clears the preceding pacing and timeout-cleanup blockers. The case
completed 207 of 207 pauses totaling 3,350.647 seconds; the longest was
97.301 seconds, below the unchanged 300-second limit. It used 798,798,821 of
3,272,703,111 allowed CPU microseconds, peaked at the exact 10 GiB memory cap
and 68 PIDs, recorded 170,996 memory-limit events but zero OOM, OOM-kill, or
group-kill events, and stayed below hard thermal limits at 94.05/68 C. The
scope was removed after 6,545.406 seconds and the outer supervisor handed off
at 74.05/64 C with no failure reason. Post-run inspection found no campaign
process, scope, snapshot, or temporary-directory residue.

All five Kiln profiles reached terminal detailed receipts. `greedy-short` and
`api-default-sampled` passed warmup, one measured 64-token request, all
request/accounting/identity gates, private-snapshot removal, and unforced
return-code-zero shutdown:

- `benchmarks/receipts/cuda/rtx4090-laptop/20260726t064133-cuda-rtx4090-laptop-c1-greedy-short-1a5a67ce-v1.kiln.json`,
  file
  `sha256:8d309d0540ba6337d46cdf0c5451cc8c7caac39f4501bb3faacd48c9d6ac60d3`,
  canonical
  `sha256:ffa5168f4c487d9c6fa874073e58036f43cc74bc83b761ebc6b5bc1c0e2267ae`.
  It measured 1.517 tokens/second, 499.5 ms TTFT, 18,358.9 ms p99 ITL,
  42,193.7 ms E2E, and 15,590,719,488 peak device-memory bytes.
- `benchmarks/receipts/cuda/rtx4090-laptop/20260726t070742-cuda-rtx4090-laptop-c1-api-default-sampled-1a5a67ce-v1.kiln.json`,
  file
  `sha256:2bc9f81f5eaa18f90d99e3fb54f10b598c624c04b82d81d096e9559eb99ac2fd`,
  canonical
  `sha256:340bcb47d0748a7c7b12b09e0efe0c4851765a3484eb2b8575fdf588c1ba3472`.
  It measured 0.713 tokens/second, 509.3 ms TTFT, 30,240.2 ms p99 ITL,
  89,758.2 ms E2E, and 15,590,031,360 peak device-memory bytes. Its
  graceful-shutdown wall time was 56.628 seconds because authenticated thermal
  freezes remained visible outside the server's active-time accounting.

The remaining three receipts are strict-valid counterevidence. `long-prefill`
and `prefix-hit` loaded the model and initialized the 62-block KV cache, but
their warmups were correctly rejected with HTTP 400 because the fixed prompts
produced 4,056 and 4,063 message tokens plus 16 requested output tokens against
the 3,968-token server context. Both servers then removed their snapshots,
drained their process groups, and exited zero without force. `mixed` completed
model upload and post-upload verification, then exited one before readiness:
its live allocator reported 2,257,895,424 free bytes, but the stricter combined
accelerator/host residency gate admitted zero of the explicitly configured 62
KV blocks. Its owned process group died naturally and its private snapshot was
removed. The retained receipts are:

- `benchmarks/receipts/cuda/rtx4090-laptop/20260726t072925-cuda-rtx4090-laptop-c1-long-prefill-1a5a67ce-v1.kiln.json`,
  file
  `sha256:d0f46230f6fa569cd978e89ff36653a5f539bc13d16b0898fdccd7efbd92d6c9`,
  canonical
  `sha256:5cea593ac5157ec52a256766d8ac7bf5485df31b775c948e7321680e1643199c`.
- `benchmarks/receipts/cuda/rtx4090-laptop/20260726t074644-cuda-rtx4090-laptop-c1-prefix-hit-1a5a67ce-v1.kiln.json`,
  file
  `sha256:a1b59d2a5c318b1c792f680c527a8fce49c4d15d839e84160353b53ed879dfcf`,
  canonical
  `sha256:5476c7dbf87ecc894348920e1eb519bdb97b2a0ab8c533276e7c57bf88860852`.
- `benchmarks/receipts/cuda/rtx4090-laptop/20260726t080744-cuda-rtx4090-laptop-c1-mixed-1a5a67ce-v1.kiln.json`,
  file
  `sha256:2edd5ca9130cccc6802ab3a66893ec2037dd8ce1325fa347de944dba53fbe45c`,
  canonical
  `sha256:7e69565a624812ad2945e6b4c7a8be5460f2823c10842e99873dc01d40b938ac`.

The Kiln campaign therefore returned 2 and vLLM was correctly not started.
These receipts establish pacing progress, two real request profiles, terminal
owned lifecycles, and cleanup only. They do not establish paired c1
performance. Do not repeat the unchanged run: first make both deterministic
long prompts fit the declared context including output reserve, and explain or
repair the mixed-profile live KV admission failure without weakening the
memory floor.

Driver v22 closes both blockers without changing the 62-block KV allocation,
one-GiB memory floor, allocator fraction, request output length, thermal
policy, or c1 acceptance gates. Prompt template
`fixed-serving-profiles-v2` reduces the shared long-prompt block count from 64
to 61. Before build or server startup, the c1 driver now uses the pinned local
Transformers runtime and exact model tokenizer/chat template to render every
warmup and measured prompt with thinking disabled. It fails closed unless all
five profiles fit the declared 3,968-token server context and the exact
observed maximum remains 3,883 input tokens. The retained local check covered
10 prompts; the maximum total was 3,947 tokens including the 64-token output
reserve, leaving 21 tokens of headroom. It binds tokenizer
`sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
and chat template
`sha256:a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`.
Historical v1 receipts remain validated against their original 64-block
prompts rather than being reinterpreted with v2.

The mixed-profile failure was stale fail-closed governor evidence, not
allocator exhaustion. Model loading and the blocking all-process residency
probe took longer than the governor's five-second cached-sample limit while
the outer controller could freeze the complete cgroup, including its
background sampler. The subsequent admission correctly projected that stale
cached snapshot to zero even though a fresh CUDA allocator probe reported
2,257,895,424 free bytes. Initial KV admission now performs one synchronous
selected-device governor refresh immediately before combining governor,
allocator, and host-backed budgets. A failed probe or unhealthy refreshed
sample still refuses allocation. This corrective source checkpoint is not c1
evidence: commit and push it, then rerun the exact command above and retain all
parent and nested receipts before making a paired performance claim.

The exact retry from clean pushed source
`bf1b385fd1c6da1df266e296014b73660ffa7c1f` proved both repairs on the
Laptop GPU but did not pass aggregate c1. `long-prefill` completed with 3,876
input plus 64 output tokens, and `prefix-hit` completed with the exact
3,883-input/64-output ceiling. `mixed` refreshed the initial KV governor to
2,260,729,856 free and 1,186,988,032 available bytes with a sub-millisecond
sample age, then allocated all 62 explicit blocks, 130,023,424 bytes, on
attempt one. Greedy, long-prefill, prefix-hit, and mixed each emitted a
strict-valid passed driver-v22 receipt below the 16.5 GB device-memory limit
and shut down without force:

- `20260726t093501-cuda-rtx4090-laptop-c1-greedy-short-bf1b385f-v1.kiln.json`
- `20260726t102019-cuda-rtx4090-laptop-c1-long-prefill-bf1b385f-v1.kiln.json`
- `20260726t104656-cuda-rtx4090-laptop-c1-prefix-hit-bf1b385f-v1.kiln.json`
- `20260726t110614-cuda-rtx4090-laptop-c1-mixed-bf1b385f-v1.kiln.json`

The API-default measured request also completed 64 tokens and its server
logged an unforced return-code-zero drained shutdown, but a recoverable
finalizer exception was swallowed before log evidence assignment. Receipt
validation then reported only `receipt.server_lifecycle.log must be an
object`, so no API-default receipt exists. The failed compact parent is
`20260726t085636979438z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`.
It binds identical initial and final model fingerprints, all 16 environment
capabilities, zero OOM events, and complete scope removal. The Kiln campaign
failed and vLLM correctly did not start.

Driver v23 makes owned finalization evidence-preserving. Shutdown and log
closure are attempted independently. A recoverable shutdown-accounting
exception carries conservative forced-shutdown evidence after the bounded
emergency drain; a log durability exception carries the readable stable log
identity. Only `EINTR`, `EAGAIN`, `EBUSY`, and `ETIMEDOUT` log `fsync`
failures receive up to three bounded attempts. Non-transient durability errors
still fail the receipt. If catastrophic cleanup prevents lifecycle evidence,
the driver reports the original structured finalization failure before receipt
validation instead of masking it with a null-field schema error. Driver v22
receipts remain validation-compatible. Commit and push v23 before the next
exact c1 retry; none of the evidence above is paired c1, wider-concurrency, or
endurance acceptance.

The source-bound laptop endurance gate is now declared separately as
`qualification/workloads/serving-cuda-endurance-v1.json` (file
`sha256:2e81344e95637821046fd0dee2ff61495af6b91a5e728214975eeb7785507d63`).
Run it only from clean pushed source after the final accepted performance
checkpoint:

```bash
python3 scripts/qualification/run.py \
  --variant cuda-rtx4090-laptop-endurance \
  --host-id rtx4090-laptop \
  --model .qualification/cuda-rtx4090-laptop/performance-model-v1 \
  --model-id Qwen/Qwen3.5-4B \
  --wsl2-thermal-policy qualification/host-policies/rtx4090-laptop-wsl2-cgroup-pacing-v2.json \
  qualification/workloads/serving-cuda-endurance-v1.json
```

The case builds current CUDA source inside the delegated scope, selects the
exact retained Laptop GPU UUID/name/17,171,480,576-byte capacity through one
persistent NVML counter, and starts one stable eager server with 62 fixed KV
blocks, reclaim and graphs disabled, and a one-request active ceiling. It runs
16/64/256/1,024-word prompt cohorts for at least 28,800 active seconds,
excluding verified thermal-pacing overlap, completes 32-token deterministic
responses, and exercises cancellation every four waves. Cumulative pacing is
limited to 14,400 seconds, so the 44,100-second measurement deadline and
47,880-second case bound cannot silently turn a mostly frozen process into an
eight-hour result. Post-stabilization device-memory and RSS growth are each
bounded at 512 MiB; unexplained ITL gaps, faults, graph activity, invalid
responses, worker/snapshot/process residue, forced shutdown, or capacity drift
fail the case.

WSL2 thermal freezes occur outside the contained driver, so the scope
controller now creates a private mode-0400 transition stream while retaining
the sole append descriptor. It freezes the scope before publishing a start
record and publishes the complete matching record before resuming, preventing
the reader from racing a partial append. Every transition binds the exact v2
policy hash, monotonic sequence, contiguous pause identity, interval, and
host/GPU sample. The contained driver reads it with `O_NOFOLLOW`, exact
owner/type/mode/schema checks and an 8 MiB bound, uses completed transitions to
attribute otherwise unexplained ITL gaps, and reports pause count, total
measurement-overlap seconds, longest interval, and derived active duration. A
missing, writable, partial, reordered,
policy-drifted, unpaired, or still-active stream fails closed. The stream does
not replace the parent receipt's authoritative outer hard-limit, final scope
accounting/removal, cleanup, and safe-handoff evidence.

This is a checked-in portable gate, not an endurance result. No eight-hour
laptop run has been performed for this workload, and it does not close the
performance, endurance, native-Linux, or desktop-4090 requirements.

The initial Kiln inputs are stable-profile, eager-only baselines. They are not
an optimality claim and must not be edited in place after a receipt binds them.
Land a new source-bound candidate for graph capture, scheduler widening,
quantization, or another vLLM memory fraction, and change one causal family at
a time. Commit and push the environment receipt, thermal policy, guarded build
and config checkpoint, first c1 receipt, each defect/fix, each wider accepted
matrix, and final soak separately.

The laptop performance handoff follows that rule with separate
`kiln-cuda-rtx4090-laptop-serving-performance-v1` and
`vllm-cuda-rtx4090-laptop-serving-performance-v1` launch inputs. They share the
closed `.qualification/cuda-rtx4090-laptop/performance-model-v1` base-model
view produced by `scripts/qualification/prepare_cuda_serving_model.py`. The
tool excludes exactly the operational `.cache` and `adapters` directories,
rejects every other non-regular root entry, publishes no duplicate weight
bytes, and requires exact source hardlinks plus complete content and strict
model-fingerprint agreement on every reuse. Each server still owns a separate
immutable runtime snapshot. The complete materialization, pinned vLLM 0.23.0
CUDA 12.9 environment, copied-interpreter runtime library, and manifest-capture
commands are in the serving benchmark protocol.

ROCm core correctness:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant rocm \
  --host-id strix-halo \
  qualification/workloads/correctness-core-v1.json
```

ROCm token-budgeted prefill correctness (Strix Halo/gfx1151):

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant rocm \
  --host-id strix-halo \
  qualification/workloads/prefill-scheduling-v1.json
```

This workload pairs ROCm with a Vulkan variant for later cross-backend receipt
comparison. Each arm combines the literal short-decode/1K-prefill/16K-prefill
actor test with a real-device deterministic hybrid-model parity test. The
latter compares monolithic prefill against six bounded quanta, including
recurrent state, the block-aligned prefix snapshot, the first following decode
token, and KV-block release. Qualification mode turns a missing device into
failure.

After the ROCm receipt is checked in, run the paired Vulkan arm from the same
source tree:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  qualification/workloads/prefill-scheduling-v1.json
```

#### Current Vulkan prefill result

The source-bound 2026-07-15 run on RADV Strix Halo passed from clean commit
`d6d14bfaf` and tree hash
`sha256:0a135f0ad1eca6ccc1bfa6df503b9d1f7c9a0f684a993600417135f439fc0f5b`.
Its retained receipt is
`qualification/receipts/vulkan/strix-halo/20260715t000104133598z-vulkan-strix-halo-prefill-scheduling-v1-6899c96516-v1.json`.
The headless probe enumerated `GPU0`; the literal short-decode/1K/16K actor
case passed its token-budget fairness and cancellation-cleanup contract; and
the production Vulkan hybrid-model case matched monolithic prefill across
quanta `[17, 17, 17, 17, 12, 1]`, six layer yields, block-aligned recurrent
state and prefix cache, first token 13, following decode token 13, and final
KV-block release. Both Rust cases ran through the bounded private-network
service with no ignored tests or output-assertion failures.

The receipt's execution identity is `closed-qualification-case-v1` hash
`sha256:4e2f59070351a66c62498c8952245ba078b66548fd1f7b7b4083b32b2ab41a93`.
Its effective-run artifact proves that neither `ROCM_PATH` nor
`KILN_ROCM_ARCHS` entered the Vulkan base environment. This fixture uses a
small deterministic hybrid model; it proves the named scheduling and state
transitions, not public-model tokenizer/logit/sampling parity or serving
performance.

Vulkan core correctness:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  qualification/workloads/correctness-core-v1.json
```

Both lower-level Vulkan workloads run `vulkaninfo --summary` without inheriting
`DISPLAY`. Presentation and surface discovery may therefore report that their
headless-only information was skipped. That is not a device skip: the probe
must still exit zero and emit a concrete `GPU<n>:` entry. Missing entries, “no
Vulkan device,” or an explicitly skipped Vulkan or physical device fail the
case.

### Current Vulkan core result

The source-bound 2026-07-14 run on RADV Strix Halo passed from clean commit
`ea4c0775f` and tree hash
`sha256:9f25964ce481ac7eb09c4e23d86c17ee631b6abd2a00bcd53efc61305d7b4e2f`.
Its retained receipt is
`qualification/receipts/vulkan/strix-halo/20260714t235017349724z-vulkan-strix-halo-core-correctness-v1-0a09f3bcee-v1.json`.
The required device probe selected the AMD Radeon 8060S Graphics through RADV
Mesa 26.1.3. The bounded, private-network Cargo service then ran 16 Vulkan
tensor, cast, transfer, reduction, reshape, and autograd parity tests and nine
dense, transposed, batched-BF16, and LoRA-composition matmul parity tests. All
25 passed, none failed or were ignored, both output-skip guards remained clear,
and the complete receipt plus every local artifact hash validated against the
current source before documentation mutation.

This lower-level receipt does not load a model and therefore makes no claim
about tokenizer behavior, model logits, sampling, paged-cache lifecycle,
cancellation, eval, public serving, or throughput. Those remain separate
source-bound Vulkan gates.

### Vulkan inference oracles

Run the model-level deterministic oracle workload from a clean pushed tree:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen/Qwen3.5-4B \
  qualification/workloads/vulkan-inference-oracles-v1.json
```

The six required cases are sequential and network-isolated:

1. A headless `vulkaninfo` probe must enumerate a physical device.
2. Rows4 and rows8 BF16 fused argmax, including tail rows, must match CPU
   projection and tie-breaking expectations.
3. Greedy and non-greedy fused sampling must match a separate CPU contract
   across temperature, top-k, top-p, min-p, repetition/presence/frequency
   penalties, six fixed seeds, F32, BF16, and batched paths.
4. F32/BF16 selected log-probs, FLCE/GRPO losses, and backward gradients must
   match analytical CPU, finite-difference, and pinned TRL/PyTorch references.
5. Six source-pinned HF-derived Qwen chat-rendering, token-ID,
   assistant-mask, label, and tokenizer/config/template-hash cases must match
   exactly.
6. After identical Qwen prefill and decode state, all full-vocabulary logits
   from the production resident and nonresident Vulkan paths must satisfy the
   declared `1e-4` relative tolerance. Qualification fails if the model,
   Vulkan runtime, or resident pool is unavailable.

Every Rust case enters `scripts/qualification/cargo-test-bounded.sh`: offline,
one job, a 50% aggregate CPU quota, private network, zero service swap, 17 GiB
aggregate ceiling, 1,740 second deadline, and the unchanged 15 GiB host-admission
floor. The runner
derives `KILN_QUALIFICATION_MODEL_PATH` from the manifest's `${model_path}` and
the inner closed Cargo policy forwards only that test binding plus
`KILN_QUALIFICATION`; it is not a product runtime setting. Ordinary source
builds, credentials, compiler flags, backend controls, and unrelated
`KILN_*` values remain excluded.

#### Current Vulkan inference-oracle result

The source-bound 2026-07-15 run passed from clean commit `8a1edd250` and tree
hash `sha256:1ac91d8cea7f50eeaa53875fc7fe2559a99a1e1b7703e3110d2e94693e2d1c1a`.
Its retained receipt is
`qualification/receipts/vulkan/strix-halo/20260715t002226653824z-vulkan-strix-halo-vulkan-inference-oracles-bfc6c14dd8-v1.json`.
All six cases passed with zero ignored tests or output-assertion failures. The
real Qwen case took 368.881 seconds and returned bit-identical logits across
all 248,320 entries (`max_abs=0`, `max_rel=0`); total workload time was 382.453
seconds. Systemd recorded a 17 GiB service-memory peak. Live cgroup monitoring
observed ceiling events but zero OOM/OOM-kill and zero service swap, and host
availability returned to 25 GiB after teardown.

This receipt closes deterministic tokenizer, lower-level CPU/TRL sampling and
selected-logprob math, and resident/nonresident Qwen equivalence. The Qwen
full-vocabulary comparison uses two Kiln Vulkan paths, not an independent HF
forward. It therefore does not close the independent public-model HF-logit,
broader prefix-cache/cancellation, eval-execution, soak, or throughput gates.

### Vulkan cache, cancellation, and live eval

Run the model-route workload from a clean pushed tree. It uses a deterministic
small BF16 hybrid model and therefore does not need `--model` or network
access:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  qualification/workloads/vulkan-model-routes-v1.json
```

The manifest runs a required headless physical-device probe followed by one
contained, sequential Rust case. Qualification mode makes a missing Vulkan
device, a skipped hardware test, a missing pass marker, a test failure, or any
output-assertion mismatch fail the workload. The Rust case proves four routes:

1. A cancelled retained prefill and an explicitly discarded retained prefill
   release every KV block, prefix-cache lease, recurrent-state entry, and
   pending-release record.
2. The production batching forward retains only a safe block-aligned strict
   prefix. An identical request and a longer second turn both hit that entry,
   reuse its blocks, and retire all leases without mutating a shared partial
   block.
3. The nonbatched generation path restores both paged KV state and the hybrid
   model's GDN linear state from a split entry. Its generated token IDs must be
   identical to an uncached full prefill; a plumbing-only cache hit is not
   sufficient.
4. `LiveEvalGenerator` and the real eval executor send a one-example exact-match
   suite through the production batching actor, `RealDecodeForward`, and the
   native Vulkan model runner. The deterministic fixture must consume six
   prompt tokens, emit exactly two completion tokens decoding to `t1 t1`, and
   report one pass with zero failed, invalid, or errored examples.

The focused eval fixture starts with `AppState`'s dependency-complete mock
construction and replaces only its inference backend with the real bounded
Vulkan runner, cache objects, forward implementation, and batching actor. This
avoids attempting a second process-global production memory admission inside
one test process; token generation, scoring, cache ownership, and backend
execution are not mocked. The test also requires `ModelRunner::backend_name()`
to be `vulkan` for each exercised model path and explicitly stops the batching
actor before returning.

This qualification route exposed a serving-profile bug in the eval adapter
transition. Base-model eval previously marked a base-to-base selection as
changed content, so stable serving correctly rejected it as a live weight
mutation. Base eval now performs a non-mutating selection. A nonempty named
adapter still always requests content reload because an adapter may have been
retrained under the same serving name and name equality cannot prove that its
weights are current.

#### Current Vulkan model-route result

The source-bound 2026-07-15 run passed from clean commit `bcb245ac7` and tree
hash `sha256:7cd165f542fba1e855a4a915516952b02038e2001fc1ef028d149e04fac54d16`.
Its retained receipt is
`qualification/receipts/vulkan/strix-halo/20260715t004650723123z-vulkan-strix-halo-vulkan-model-routes-v1-e2287dab6c-v1.json`.
Both required cases passed with zero output-assertion failures on AMD Radeon
8060S Graphics, RADV Mesa 26.1.3. The model-route case took 2.543 seconds and
the complete workload took 2.760 seconds. The receipt and all eight local
artifact hashes validated against the known current commit before this
documentation mutation; teardown left no qualification service or build
process and host availability returned to 24 GiB.

This receipt closes the named deterministic cache lifecycle, cancellation, and
live eval execution subsets. It does not compare full-model logits against an
independent Hugging Face forward, measure public-model eval quality, exercise a
public HTTP eval job, establish long-duration stability, or make a throughput
claim. Those remain separate qualification gates.

### Vulkan independent Hugging Face full-model oracle

Run the independent full-model oracle only from a clean pushed tree. Prepare a
Python executable with the exact ROCm PyTorch and package versions in
`scripts/hf_trl/requirements-sft.lock` as described in
[HF/TRL Interoperability](HF_TRL_INTEROP.md). The oracle additionally pins the
PyTorch commit and hashes the installed Qwen3.5 Transformers modeling and
configuration modules. Optional fused linear-attention packages must be absent:
the reference deliberately uses Transformers' independent torch fallback.

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen/Qwen3.5-4B \
  --var trainer_python=/absolute/path/to/pinned-venv/bin/python \
  qualification/workloads/vulkan-hf-full-model-oracle-v1.json
```

The single required case executes two accelerator stages sequentially. It
never holds the model in ROCm and Vulkan at the same time:

1. The HF stage requires at least 23 GiB `MemAvailable`, then creates a
   private-network systemd service with `MemoryMax=16G`, zero service swap, a
   600-second cap, and control-group teardown. Inside that service, the thermal
   supervisor validates the content-hashed
   `qualification/host-policies/strix-halo-hf-oracle-v1.json` policy, requires
   20 consecutive 50 ms package samples at or below 45 C, and only then creates
   the accelerator worker in a new process group. The worker blocks on a private
   start gate until the supervisor has attached the continuous guard.
2. The guard samples the exact `k10temp/Tctl` input every 50 ms, stops the
   complete worker process group at 58 C, resumes it after 20 consecutive
   samples at or below 50 C, and terminates the group at 93 C. It remains armed
   through eager BF16 model load and forward, then requires the dead worker's
   host package to produce 20 consecutive samples at or below 45 C before the
   service can report success. Missing or ambiguous sensors, malformed samples,
   unreconciled stops, a trip, or a 300-second cooldown timeout fails closed.
3. The pinned worker uses deterministic algorithms, disables TF32, evaluates
   fixed input IDs `[1,2,3,4,5,6,7,8,100]`, and writes one safetensors artifact
   containing the input IDs and all 248,320 final-position F32 logits. It reads
   its own cgroup-v2 memory peak, swap, and
   OOM/limit events before it exits. This self-report is intentional: the
   qualification runner's bubblewrap PID/network namespace can launch and wait
   for a user service, but a post-exit `systemctl --user show` cannot reconnect
   to the host user bus. `systemd-run --wait` remains the exit verdict, and a
   missing, malformed, swapped, or OOM-bearing telemetry record fails closed.
4. After HF exits and its cooldown completes, the driver requires 24 GiB
   `MemAvailable` before Vulkan can
   start. The Rust test runs through `cargo-test-bounded.sh` with offline Cargo,
   private networking, `CPUQuota=50%`, `MemoryMax=17G`, zero service swap, a
   1,740-second cap, and a seven-GiB host reserve. The closed qualification
   environment forwards only the runner-owned model and HF-reference paths plus the hardware gate.
5. Kiln loads the same weights and input IDs through both its production
   resident and nonresident Vulkan paths. Those two paths must remain
   bit-identical. The resident result is then compared with every HF logit;
   argmax must match exactly, top-10 overlap must be at least 9/10, maximum
   absolute error at most `0.5`, mean absolute error at most `0.05`, and cosine
   similarity at least `0.9999`. Non-finite values, a missing device, an ignored
   test, an incomplete result, or any threshold failure rejects the workload.

The compact result records the comparison metrics, HF cgroup peak and service
swap, the deterministic raw-logit tensor hash, the independently computed
reference-artifact hash, memory ceilings, attention routes, exact input IDs,
content-hashed thermal policy, prelaunch samples, runtime package peak, pacing
count and duration, and post-exit cooldown evidence. Raw model output remains
below the ignored `.qualification/` run tree.
Validate a new receipt before changing documentation:

```bash
python3 scripts/qualification/receipt.py \
  --require-current-source \
  --require-local-artifacts \
  --require-known-commit \
  qualification/receipts/vulkan/<host>/<receipt>.json
```

#### Current Vulkan/HF full-model result

The source-bound 2026-07-15 run passed from clean commit `4d6697c52` and tree
hash `sha256:b21d95b47650ee831d27a678e85b3842d369b6bc78e0f21ec84c9a9da65bcfa4`.
Its retained receipt is
`qualification/receipts/vulkan/strix-halo/20260715t013710403012z-vulkan-strix-halo-vulkan-hf-full-model-ora-39c1bc8042-v1.json`.
The 390.443-second workload compared all 248,320 logits and passed with exact
argmax 1, 10/10 top-token overlap, maximum absolute error `0.12433958`, mean
absolute error `0.020353988`, and cosine similarity `0.999941539769`. The HF
stage reported a 9,254,346,752-byte cgroup peak, zero service swap, zero high,
limit, OOM, and OOM-kill events, and deterministic raw-logit hash
`sha256:0b902c0d74a8ed54aefefcdab50adeb6fedd7adb3e45a2338c27276e90abeeaf`.
Kiln's resident and nonresident Vulkan results were bit-identical.

Live host monitoring observed the Vulkan service contact its 17 GiB cgroup
ceiling without OOM or service swap; system-wide swap rose by roughly 0.4 GiB
while host available memory remained above the reserved floor, then 24 GiB was
available after teardown. This is retained as a host-pressure signal for the
development soak rather than hidden behind the passing numerical verdict.

This historical receipt closes the numerical Phase 6 independent CPU/HF
full-model comparison when combined with the tokenizer, sampling,
selected-logprob, cache, cancellation, and live-eval receipts above. It predates
the now-required thermal supervisor evidence, so it does not satisfy the current
workload manifest or the final common-tree gate without a rerun. It covers one
deterministic next-token full-vocabulary forward. It does not establish
multi-token public-model output parity, public HTTP eval quality, long-duration
stability, large-batch throughput, or competitive performance against vLLM.

### ROCm serving first-divergence oracle

Use the focused ROCm oracle only after a source-paired exact greedy comparison
has retained both engine outputs and localized their first different token. It
is a diagnostic correctness gate, not a serving throughput measurement. The
tracked request
`qualification/oracles/rocm-strix-halo-greedy-c1-first-divergence-v1.json`
binds the exact Kiln and vLLM receipt file/content hashes, their common source
commit, the model weights/config/tokenizer/template identity, the original
user message, chat-template invocation, 163 prompt token IDs, and the first
three common continuation tokens `[1206,5517,264]`. Its complete 166-token
input has canonical hash
`sha256:709d0a314cde9072ac79b0752e795f1b76bfaea5b553ccdf26f7fbd5ac44b1a0`.
The two declared candidates are Kiln token `25045` (` baseline`) and vLLM token
`15787` (` foundation`).
The machine-readable field references are published as
[HF Next-Token Request Schema](https://ericflo.github.io/kiln/docs/hf-next-token-request-schema/)
and
[ROCm HF Next-Token Result Schema](https://ericflo.github.io/kiln/docs/rocm-hf-next-token-result-schema/).
Both close every object to unknown fields; the executable validator additionally
enforces cross-field hashes, token concatenation, source-receipt contents,
candidate ranks, thermal reconciliation, and the canonical result self-hash.

Run it only from a clean `HEAD` already pushed to `origin/main`; every path
below is intentionally absolute:

```bash
python3 scripts/qualification/rocm_hf_next_token_oracle.py run \
  --model "$(pwd)/Qwen3.5-4B" \
  --trainer-python "$(pwd)/target/qualification/hf-trl-roundtrip/.venv/bin/python" \
  --request "$(pwd)/qualification/oracles/rocm-strix-halo-greedy-c1-first-divergence-v1.json" \
  --host-thermal-policy "$(pwd)/qualification/host-policies/strix-halo-hf-oracle-v1.json" \
  --out "$(pwd)/.qualification/rocm-hf-next-token-result.json"
```

Before an accelerator service exists, the driver strictly validates both source
receipts, requires `HEAD == origin/main`, resolves the declared thermal policy
and interpreter, and fingerprints the complete model in a separate process.
That process receives only `HOME`, locale, `PATH`, `PYTHONHASHSEED`, and its
private `TMPDIR`; no ambient token or `KILN_*` configuration crosses the
boundary. The supervisor owns the worker's private start gate, requires the
stable 45 C prelaunch boundary, attaches the continuous package guard, and only
then releases model hashing. Hashing is stopped at 58 C, resumes after 20
consecutive samples at or below 50 C, terminates at the 93 C hard limit, and
must complete the same 45 C post-exit cooldown. An exception or timeout tears
down the complete fingerprint process group and prevents result creation.

The fingerprint worker holds non-symlink regular-file descriptors, hashes all
declared weight shards and configuration/tokenizer/template inputs, and
rechecks their metadata and contents before accepting the identity. The driver
then verifies at least 23 GiB `MemAvailable`. The accelerator systemd service
supplies the same 16 GiB memory, zero-swap, private-network, 600-second
control-group boundary used by the full Vulkan oracle. Its gated worker
independently reloads the pinned tokenizer and must reproduce every prompt ID
and every retained token's decoded bytes before loading the model. It pins ROCm
PyTorch, Transformers sources and versions, eager attention, the Transformers
torch linear-attention fallback, BF16, deterministic algorithms, and TF32-off
execution.

The compact result retains the full-vocabulary F32-logit hash, argmax and text,
top ten token IDs/text/logits, both candidate logits and vocabulary ranks,
argmax margin, package versions, cgroup peak/limit/OOM/swap counters, thermal
policy and lifecycle, clean pushed source identity, implementation hashes, and
an explicit `kiln`, `vllm`, or `neither` attribution. New results also contain
`model_fingerprint`, binding the fingerprint script and interpreter hashes plus
its complete prelaunch, pacing, peak, and cooldown evidence. The checker accepts
that field's absence only for the exact canonical hashes of the four retained
pre-correction oracle/path/layer results; an arbitrary new result cannot omit
it. The roughly one-MiB raw logit artifact remains ignored. The result has a
canonical `result_sha256` and can be checked without running the model:

```bash
python3 scripts/qualification/rocm_hf_next_token_oracle.py check \
  .qualification/rocm-hf-next-token-result.json \
  --require-current-source
```

Specialized oracle results live under `qualification/oracle-results`, separate
from generic qualification receipts and serving-benchmark receipts because
each family has a different closed schema and validator. Both portable
workflows discover and check every JSON result in that tree without running an
accelerator. The current retained Strix Halo result is
`qualification/oracle-results/rocm/strix-halo/20260719t003452-rocm-strix-halo-hf-next-token-first-divergence-v1.json`,
executed from clean pushed source `c5640c090f295cd50a73aae63ef48d403fd13d98`.
The eager HF reference selects vLLM's token `15787` (` foundation`) at rank one
with logit `18.5`; Kiln's token `25045` (` baseline`) is rank two with logit
`18.25`, a `0.25` top-logit margin. The full F32 logit vector has canonical
hash `sha256:d4bc2aeb7d6bef608dfe500f6b9759b1fd4ca5eeb924b55b9e63b2e49a0e96d0`.
The guarded forward completed without memory, swap, OOM, thermal-trip, or
cooldown residue: the cgroup peaked at 9,529,192,448 bytes, runtime temperature
peaked at 60 C, and four completed pacing events occupied 9.381 seconds. This
closes the candidate attribution at that exact first divergence: vLLM matches
the independent reference and Kiln does not. It does not yet locate Kiln's
numerical error or establish broader sequence parity.

The current guarded-provenance verification is retained at
`qualification/oracle-results/rocm/strix-halo/20260719t030411-rocm-strix-halo-hf-next-token-guarded-fingerprint-v1.json`
from clean pushed source `34ed8241e5e8d9a2772c0baca8fb108ce6328565`.
The fingerprint phase began at 39.5 C, peaked at 59 C, completed all six pacing
events over 13.755 seconds, and cooled to 42.875 C with zero trip or timeout.
The independently guarded HF phase peaked at 60.125 C, completed all five
pacing events over 12.112 seconds, cooled to 42.75 C, and reported a
9,457,963,008-byte cgroup peak with zero high, limit, OOM, OOM-kill, or swap
events. The complete lifecycle took 52.355 seconds. It reproduced argmax
`15787`, both candidate logits/ranks, and full-logit hash
`sha256:d4bc2aeb7d6bef608dfe500f6b9759b1fd4ca5eeb924b55b9e63b2e49a0e96d0`
exactly. This closes the observed hashing-safety counterexample for the focused
oracle path; it does not qualify broader serving workloads or the remaining
accelerated-route bisection.

#### ROCm numerical-path attribution

Use the path-attribution runner after the independent HF result above has been
retained and its ignored raw safetensors artifact is still available. This is a
Kiln defect-localization workload, not a new oracle and not a performance
benchmark. It loads the model once and evaluates four distinct routes against
the same 248,320-element F32 HF logit vector:

1. `eager_full` uses the ordinary paged prefill/decode path and reads back the
   complete final logit vector.
2. `eager_greedy` uses the production eager greedy-selection API, isolating
   selection from full-logit readback.
3. `graph_full` primes one retained HIP-graph slot, releases its logical row,
   and requires all three exact continuation steps to replay before comparing
   the complete final logit vector.
4. `graph_greedy` uses a separate recurrent state and logical row and requires
   the same three retained replays through the production graph greedy API.

Each route starts from an independent KV cache and linear-attention state. The
worker forces the request's three known-common continuation tokens rather than
feeding one route's prediction into the next step. Therefore, all four final
comparisons describe the same 166-token state immediately before the disputed
fourth generated token. The graph prime reproduces the retained-slot condition
of the source serving process; a warmup-only call or a capture without replay
does not pass.

Run only from a clean commit already present at `origin/main`. The raw HF
reference from the accepted run currently lives at
`.qualification/.rocm-hf-next-token-result.artifacts/hf-reference.safetensors`
and is deliberately ignored; the runner accepts it only when its size and hash
match the retained compact oracle result. All paths are intentionally absolute:

```bash
python3 scripts/qualification/rocm_hf_path_attribution.py run \
  --model "$(pwd)/Qwen3.5-4B" \
  --request "$(pwd)/qualification/oracles/rocm-strix-halo-greedy-c1-first-divergence-v1.json" \
  --oracle-result "$(pwd)/qualification/oracle-results/rocm/strix-halo/20260719t003452-rocm-strix-halo-hf-next-token-first-divergence-v1.json" \
  --hf-reference "$(pwd)/.qualification/.rocm-hf-next-token-result.artifacts/hf-reference.safetensors" \
  --host-thermal-policy "$(pwd)/qualification/host-policies/strix-halo-hf-oracle-v1.json" \
  --python "$(pwd)/target/qualification/hf-trl-roundtrip/.venv/bin/python" \
  --out "$(pwd)/.qualification/rocm-hf-path-attribution-result.json"
```

Before device execution, the runner revalidates the request, both source
receipts, retained HF result, full model content, raw reference, interpreter,
clean source identity, and at least 23 GiB of host-available memory. Full model
content uses the same separately gated, continuously guarded fingerprint
process described above, and the path result binds its implementation,
interpreter, and thermal lifecycle. It performs
an offline locked `gfx1151` release build through `scripts/cargo-bounded.sh`
with a 15 GiB build floor, 50 percent CPU quota, zero swap, private networking,
and the versioned `closed-source-build-v1` environment policy. Ambient
`KILN_*`, compiler flags, credentials, and target paths cannot enter that
service; only the runner-owned build controls and `KILN_ROCM_ARCHS=gfx1151`
do. The result records both the architecture and build-policy identity. The
worker itself receives a closed six-variable environment with no product
configuration variables and installs the qualified model/tensor policy,
`auto` kiln-tensor API mode, and legacy host barriers explicitly before the
primary ROCm context exists.

Execution is a private-network systemd user service with `MemoryMax=48G`, zero
swap, control-group kill semantics, and a 900-second outer lifetime. The
existing thermal supervisor completes the stable prelaunch boundary, starts a
new worker process group blocked on a private file gate, attaches the thermal
guard, and only then releases execution. The generic guarded-exec shim validates
the exact argv, working directory, complete environment, executable path, and
SHA-256. It holds an open descriptor across the gate and executes
`/proc/self/fd/<n>`, so replacing the build pathname after validation cannot
change the admitted inode. A pass additionally requires worker-observed cgroup
v2 limit/peak/current counters, zero high/limit/OOM/OOM-kill events, zero swap,
complete thermal settlement, and no graph fallback or replay failure.

The compact result schema is published as
[ROCm/HF Path Attribution Result Schema](https://ericflo.github.io/kiln/docs/rocm-hf-path-attribution-result-schema/).
It retains source/tree identity, binary and implementation hashes, request and
oracle references, model identity, full containment evidence, both full-logit
hashes and error summaries, both candidate logits/ranks, all four observed
token sequences, graph capture/replay counters, and one mechanically derived
attribution:

- `eager_full_logits`: the ordinary eager numerical path already disagrees.
- `hip_graph_full_logits`: eager full logits agree, but retained graph logits do
  not.
- `eager_greedy_selection`: full logits agree, but eager selection does not.
- `hip_graph_greedy_selection`: graph full logits agree, but graph selection
  does not.
- `serving_only_or_not_reproduced`: all isolated routes agree; the serving-only
  composition or a non-reproduced condition remains.

These labels identify the first tested boundary that disagrees; they do not by
themselves identify a layer or kernel. The executable checker recomputes the
label and canonical self-hash and cross-binds the observed prefix, candidates,
input hash/count, model, original receipts, retained HF result, and raw-artifact
identity. Check one result directly with:

```bash
python3 scripts/qualification/rocm_hf_path_attribution.py check \
  .qualification/rocm-hf-path-attribution-result.json \
  --require-current-source
```

The current retained Strix Halo result is
`qualification/oracle-results/rocm/strix-halo/20260719t014807-rocm-strix-halo-hf-path-attribution-v1.json`,
executed from clean pushed source `034e548e86910c08080d43c7c4196103fce2f9f3`.
It reports `eager_full_logits`: eager and retained-graph full logits are
bit-identical with canonical vector hash
`sha256:b5acbda785044ca46d6cddb9aea03258dd4f99fb1904dcb5983be49ef68fd603`,
and the eager and graph greedy routes select that vector's argmax. All four
routes emit the common prefix `1206, 5517, 264` followed by Kiln token `25045`.
At the disputed position, Kiln ranks token `25045` first at `18.625` and HF's
token `15787` second at `18.5`; the independent HF vector instead ranks token
`15787` first at `18.5`. Relative to HF, Kiln has `0.998147822` cosine
similarity, `0.84375` maximum absolute error, `0.140939553` mean absolute
error, and nine of ten top tokens in common.

The retained graph evidence is one successful capture and seven successful
replays with zero capture failure, replay failure, or fallback. This rules out
HIP-graph replay and greedy selection as the first tested divergence boundary;
the defect is already present in the ordinary eager full-logit forward. The
112.012-second guarded lifecycle peaked at 14,657,327,104 cgroup bytes and
59.75 C, recorded zero memory-limit, OOM, swap, or thermal-trip events,
completed 28 thermal pacing events totaling 64.109 seconds, and handed the
host back at 42.75 C. The result does not identify the first divergent model
layer or kernel. The next bounded probe must compare layer-boundary outputs for
this exact 166-token state before changing serving configuration or repeating
the concurrency matrix.

#### ROCm layer-boundary attribution

Use the layer-attribution runner only after a retained path result reports
`eager_full_logits`. It executes two sequential, independently contained model
loads for the same 166-token state. The pinned eager Transformers worker first
captures the F32 final hidden row after the token embedding, each of the 32
decoder layers, and final RMSNorm. After that process exits and host memory
recovers, the Kiln worker prefills the 163-token prompt, advances the first two
known continuation tokens normally, and captures the same 34 boundaries while
processing the third continuation token. This preserves the production paged
KV and recurrent-state route instead of substituting a monolithic Kiln
forward.

```bash
python3 scripts/qualification/rocm_hf_layer_attribution.py run \
  --model "$(pwd)/Qwen3.5-4B" \
  --request "$(pwd)/qualification/oracles/rocm-strix-halo-greedy-c1-first-divergence-v1.json" \
  --host-thermal-policy "$(pwd)/qualification/host-policies/strix-halo-hf-oracle-v1.json" \
  --python "$(pwd)/target/qualification/hf-trl-roundtrip/.venv/bin/python" \
  --kernel-profile qualified \
  --out "$(pwd)/.qualification/rocm-hf-layer-attribution-result.json"
```

`--kernel-profile` is a closed typed choice. `qualified` is the production
model/tensor policy and remains the default. Twelve layer-only diagnostic
profiles preserve the same ROCm device, BF16 weights, paged state, prompt, and
layer capture:

Current source keeps fused RMSNorm disabled in `qualified` and
`experimental_multiblock`; those profiles install 43 and 44 of the 45
accelerated model/tensor routes, respectively. Retained results from earlier
source revisions bind their historical profile composition by commit and tree
and must not be interpreted as the current `qualified` composition.

- `portable_fallback` declines every accelerated model route and every optional
  accelerated tensor route.
- `model_fallback` declines accelerated model routes while retaining the
  qualified low-level tensor policy.
- `tensor_fallback` retains qualified model dispatch while declining optional
  accelerated tensor routes.
- `gdn_fallback` declines the 12 model-policy leaves that own GDN projections,
  gates, recurrent mixing, fused decode, fused convolution, and the default-off
  multiblock route while retaining the other 18 model leaves and qualified
  tensor policy.
- `non_gdn_fallback` is the exact model-policy inverse: it retains the qualified
  GDN/recurrent family, including the default-off multiblock value, declines
  the other 18 model leaves, and retains qualified tensor policy.
- `fused_norm_mlp_fallback` declines fused RMSNorm and both fused MLP routes
  while preserving every other qualified model route and qualified tensor
  policy.
- `fused_norm_mlp_only` is its exact inverse: it enables only those three model
  routes on portable model policy and retains qualified tensor policy.
- `fused_rmsnorm_fallback` is retained as a historical diagnostic name and is
  identical to the repaired current `qualified` policy.
- `fused_mlp_silu_mul_fallback` and `fused_mlp_gate_up_prefill_fallback` each
  decline the named MLP route while inheriting the current qualified policy,
  including its disabled fused RMSNorm route, and retain qualified tensor
  policy.
- `split_q_gate_fallback` changes only the split q/gate F32-output projection
  leaf from the qualified model policy and retains qualified tensor policy.
- `split_q_gate_only` is its exact inverse: it enables only that model leaf on
  the portable model policy and retains qualified tensor policy.

Use the source-paired portable profile first to determine whether acceleration
as a group changes the error curve. If it does, run both mixed profiles: a
correct `model_fallback` with an incorrect `tensor_fallback` localizes the
defect to accelerated model dispatch, while the inverse localizes it to
low-level tensor dispatch. If both are wrong or both are correct, the result is
an interaction or a non-exclusive cause and must not be reported as a single
group. The worker marker, compact result, JSON Schema, and checker bind the
chosen profile; the closed worker environment never translates it through an
ambient environment variable. The path-attribution mode remains qualified-only
because its graph route is itself the object under test.

After the mixed model/tensor pair localizes a defect to model dispatch, run
both GDN subgroup profiles. A correct `gdn_fallback` paired with an incorrect
`non_gdn_fallback` localizes the defect to the GDN/recurrent family; the inverse
localizes it to the other model routes. As with the broader pair, two matching
verdicts establish an interaction or non-exclusive cause and must not be forced
into a single subgroup. These profile names are selectable only by the guarded
layer-attribution binary and are not server configuration values.

After the GDN subgroup pair localizes the defect to the remaining model leaves,
run both split q/gate profiles. Correct fallback plus incorrect split-only
evidence makes that leaf causal. The inverse excludes it. Matching verdicts
again require reporting an interaction instead of claiming single-route
causality.

For pre-repair source revisions, if the split q/gate pair excludes that route,
run both fused norm/MLP profiles.
Correct fallback plus incorrect fused-only evidence localizes the defect to the
three-route group. The inverse excludes the group. Matching verdicts require an
interaction claim before any member is disabled in the qualified policy.
If both group arms are correct, run all three single-route fallbacks. A correct
single fallback establishes that the named route is necessary under the
qualified composition; an incorrect result excludes that route as the sole
necessary member. Multiple correct results require another interaction claim,
not an arbitrary production-policy change. On current source, run `qualified`
directly for requalification; the fused RMSNorm fallback is intentionally a
no-op retained for source-bound evidence compatibility.

The HF arm uses the pinned PyTorch/Transformers fallback implementation,
BF16 weights, eager full attention, deterministic algorithms, TF32 disabled,
a private-network 20 GiB zero-swap service, and the same closed thermal
supervisor as the next-token oracle. The 23 GiB host-availability preflight is
unchanged. The layer arm has a larger ceiling than the independent 16 GiB
next-token arm because a cold unified-memory load can charge both file-backed
safetensors pages and GPU allocations to its cgroup. Any high, max, OOM,
OOM-kill, or swap event still rejects the evidence. Forward hooks retain only
one cloned last row per boundary; the ignored safetensors artifact contains the
34-by-2,560 F32 matrix, the exact input IDs, and final logits. The marker and
compact result bind its aggregate row hash, ordered names, model/request
identity, installed source hashes, cgroup events, and complete thermal
lifecycle. Before this arm, the separately gated fingerprint process uses the
same declared policy and its own complete lifecycle is retained alongside the
model identity. Retained historical results record the earlier 16 GiB ceiling;
the checker accepts that value only as historical evidence, while
`--require-current-source` requires the current 20 GiB contract.

The Kiln arm rebuilds the shared attribution example through the offline
`gfx1151` closed-source build service. Its exact-argv open-inode gate then runs
the qualified model/tensor policy in a private-network 48 GiB zero-swap
service. Snapshot tensors remain on the device until the ordinary forward is
complete; each captured BF16 row is explicitly converted to F32 before host
readback. The worker must reproduce the three common predictions, retain the
final Kiln logit hash, bind the HF matrix hash, observe clean cgroup counters,
and complete cooldown without process or service residue.

For every boundary the result records the HF and Kiln row hashes, cosine
similarity, maximum and mean absolute error, RMSE, HF RMS magnitude, and
relative RMSE. It also mechanically selects the boundary with the largest
increase in relative RMSE over its predecessor, with earliest-index tie
breaking. This is an error-growth locator, not an assertion that normal BF16
rounding must be bit-identical or that the selected layer's entire block is the
root cause. A result narrows the next probe to that block's pre-norm, mixer,
residual, post-norm, and MLP boundaries; it does not itself authorize a repair.
The published schema is
[ROCm/HF Layer Attribution Result Schema](https://ericflo.github.io/kiln/docs/rocm-hf-layer-attribution-result-schema/).

Validate a retained compact result and its current-source binding with:

```bash
python3 scripts/qualification/rocm_hf_layer_attribution.py check \
  .qualification/rocm-hf-layer-attribution-result.json \
  --require-current-source
```

Portable repository gates discover heterogeneous files under
`qualification/oracle-results` and dispatch each declared schema through
`scripts/qualification/check_oracle_results.py`. Unknown schemas fail instead
of being interpreted as a different result family. After a passing local run,
retain only the compact JSON result in that tree; keep the raw logits and
execution workspace ignored. A route attribution is sufficient to choose the
next narrower numerical probe or repair, but it does not satisfy multi-token
parity, throughput, soak, or final common-tree acceptance.

The current retained Strix Halo result is
`qualification/oracle-results/rocm/strix-halo/20260719t022838-rocm-strix-halo-hf-layer-attribution-v1.json`,
executed from clean pushed source `3df1365ea3fd67955e36e640cdaed2f72703cc2d`.
The embedding rows are bit-identical. Relative RMSE then rises from `0` to
`0.014871503`, `0.036848969`, and `0.065551177` across linear-attention layers
0, 1, and 2. Layer 2 is the mechanically selected largest sequential increase,
with delta `0.028702207`; the following full-attention layer increases relative
RMSE by only `0.000543160` to `0.066094336`. This localizes the next comparison
to the pre-norm, recurrent mixer, residual, post-norm, and MLP boundaries inside
the early linear-attention blocks. It does not yet distinguish a kernel defect
from expected BF16 implementation drift that later changes the close argmax.
The final norm has `0.996377129` cosine similarity, `1.08203125` maximum
absolute error, `0.213972354` mean absolute error, and `0.085119899` relative
RMSE. The 192.563-second guarded lifecycle included a 68.109-second build; HF
and Kiln peaked at 9,459,863,552 and 14,658,502,656 cgroup bytes, respectively,
with zero high, limit, OOM, OOM-kill, or swap events. Both runtime arms peaked
at 59.875 C, all 4 HF and 28 Kiln pacing events completed, both cooldowns
settled, and no owned process or service remained.

The source-bound portable-fallback comparison is retained at
`qualification/oracle-results/rocm/strix-halo/20260719t024003-rocm-strix-halo-hf-layer-attribution-portable-fallback-v1.json`,
from clean pushed source `e2316e7bb4e58e08777e66a45bec1f306352964a`.
It restores the independent HF/vLLM argmax: the observed sequence is `1206,
5517, 264, 15787` instead of the qualified result's final token `25045`.
Relative RMSE falls from `0.014871503` to `0.002575195` at layer 0, from
`0.065551177` to `0.007065248` at layer 2, and from `0.085119899` to
`0.015876121` after final RMSNorm. The qualified accelerated ROCm policy is
therefore the causal policy group for this first-token correctness defect; the
remaining work is to bisect its model routes and repair or disable the failing
route. Portable fallback is diagnostic evidence, not a performance-qualified
replacement policy.

The portable lifecycle completed in 80.812 seconds, including a 6.577-second
build. The Kiln cgroup peaked at 12,110,475,264 bytes with zero high, limit,
OOM, OOM-kill, or swap events; runtime peaked at 59.75 C, seven pacing events
completed in 17.018 seconds, and cooldown returned the host at 43.25 C with no
owned residue. Model provenance hashing immediately before HF supervision
heated the package to 93.5 C, above the runtime policy's 93 C limit but below
the outer source-build guard's 97 C ceiling. The prelaunch gate then waited
5.721 seconds and required 20 stable samples at or below 45 C before creating
the process, so inference was contained, but hashing itself was not. That
counterexample prompted the guarded fingerprint process described above. The
portable result remains valid numerical evidence but does not claim guarded
hashing; current-source receipts must retain that separate lifecycle, and a
small guarded hardware run must verify it before any wider workload resumes.

The guarded mixed-profile pair is retained at
`qualification/oracle-results/rocm/strix-halo/20260719t031503-rocm-strix-halo-hf-layer-attribution-model-fallback-v1.json`
and
`qualification/oracle-results/rocm/strix-halo/20260719t032012-rocm-strix-halo-hf-layer-attribution-tensor-fallback-v1.json`,
both from clean pushed source `2b5b33665bb6af835ee7034b5bd43903843cc5a3`.
`model_fallback` restores token `15787` while preserving the qualified tensor
policy. Its relative RMSE is `0.002575195`, `0.004210653`, and `0.007065248`
after linear-attention layers 0 through 2, then `0.016856438` after final
RMSNorm. Conversely, `tensor_fallback` preserves the qualified model policy
and reproduces token `25045`, final logit hash
`sha256:b5acbda785044ca46d6cddb9aea03258dd4f99fb1904dcb5983be49ef68fd603`,
and the qualified relative-RMSE curve exactly: `0.014871503`, `0.036848969`,
`0.065551177`, and `0.085119899` at the same boundaries. This pair localizes
the defect to accelerated model dispatch rather than the optional low-level
tensor routes. The first observed error remains layer 0 linear attention; the
next discriminator must divide model-level recurrent-mixer dispatch from the
other model routes before any repair is selected.

The model- and tensor-fallback lifecycles completed in 89.496 and 147.114
seconds. Across their six independently supervised fingerprint, HF, and Kiln
arms, peak package temperature was 59.875 C; every pacing interval and cooldown
completed, all cgroup high, limit, OOM, OOM-kill, and swap counters remained
zero, and no owned process or service remained. The two Kiln cgroups peaked at
12,109,606,912 and 14,658,228,224 bytes. These results authorize a narrower
correctness discriminator only. They do not qualify fallback performance,
multi-token parity, serving throughput, a soak, or final ROCm acceptance.

The guarded GDN subgroup pair is retained at
`qualification/oracle-results/rocm/strix-halo/20260719t033606-rocm-strix-halo-hf-layer-attribution-gdn-fallback-v1.json`
and
`qualification/oracle-results/rocm/strix-halo/20260719t033810-rocm-strix-halo-hf-layer-attribution-non-gdn-fallback-v1.json`,
both from clean pushed source `d2d3656556b0fc73858499fbc63650af9505caf2`.
`gdn_fallback` keeps the other 18 model leaves qualified and remains wrong at
token `25045`; its layer-0/1/2 and final-norm relative RMSE values are
`0.014459307`, `0.037805305`, `0.066248279`, and `0.088236589`.
`non_gdn_fallback` keeps only the qualified GDN/recurrent family and restores
token `15787`; its corresponding values fall to `0.002066817`, `0.003870959`,
`0.006575082`, and `0.016082606`. Under these disjoint compositions, the
GDN/recurrent policy family is excluded and the causal group is the remaining
18 model leaves.

The first divergence still occurs inside layer 0, before a full-attention
layer can affect that token's hidden state. Within the remaining group, the
live early-layer candidates are the split q/gate F32-output projection, fused
RMSNorm, and fused MLP routes; full-attention, LoRA, sampled-head, and weight-8
routes cannot be selected for this BF16, adapter-free layer-0 boundary. The
next source-bound discriminator should therefore test the split q/gate route
directly before widening the internal boundary inventory. The GDN and
non-GDN lifecycles completed in 234.968 and 89.977 seconds, respectively; the
first includes a 67-second cold release build. All six guarded arms remained
at or below 59.875 C, every pacing and cooldown interval completed, cgroup
event and swap counters remained zero, and no owned residue remained. Kiln
cgroups peaked at 14,658,535,424 and 12,110,376,960 bytes.

The guarded split q/gate pair is retained at
`qualification/oracle-results/rocm/strix-halo/20260719t035301-rocm-strix-halo-hf-layer-attribution-split-q-gate-fallback-v1.json`
and
`qualification/oracle-results/rocm/strix-halo/20260719t035556-rocm-strix-halo-hf-layer-attribution-split-q-gate-only-v1.json`,
both from clean pushed source `ae6504f3b445f8d45302c030f221071f94582491`.
Disabling only `split_q_gate_f32_output` in the otherwise qualified policy is
bit-for-bit identical to the qualified result: it emits token `25045`, retains
final logit hash
`sha256:b5acbda785044ca46d6cddb9aea03258dd4f99fb1904dcb5983be49ef68fd603`,
and has relative RMSE `0.014871503`, `0.036848969`, `0.065551177`, and
`0.085119899` after layers 0 through 2 and final RMSNorm. Enabling only that
leaf on the portable model policy restores token `15787`; its corresponding
values are `0.002575195`, `0.004210653`, `0.007065248`, and `0.016856438`.
The direct and inverse arms therefore exclude this route as the cause for this
state; it neither repairs the qualified composition when disabled nor induces
the defect when enabled alone.

The two complete lifecycles lasted 219.112 and 90.214 seconds. Their guarded
arms stayed at or below 61.25 C, all pacing and cooldown intervals completed,
all cgroup event and swap counters remained zero, and no owned residue
remained. Kiln cgroups peaked at 14,657,499,136 and 12,110,065,664 bytes. This
evidence authorizes a grouped discriminator over the remaining live
layer-common routes, starting with fused RMSNorm and fused MLP dispatch. It
does not repair the qualified profile or establish parity, performance, or
endurance acceptance.

The guarded fused norm/MLP pair is retained at
`qualification/oracle-results/rocm/strix-halo/20260719t041300-rocm-strix-halo-hf-layer-attribution-fused-norm-mlp-fallback-v1.json`
and
`qualification/oracle-results/rocm/strix-halo/20260719t041456-rocm-strix-halo-hf-layer-attribution-fused-norm-mlp-only-v1.json`,
both from clean pushed source `d3ddbdb775b5dfbaa2f7c29976baf6858ca6b5d5`.
Both arms restore token `15787`, so the group produces an interaction verdict,
not a single-group cause. Declining the three routes in the otherwise qualified
policy yields relative RMSE `0.014672443`, `0.036750649`, `0.072686307`, and
`0.085907010` after layers 0 through 2 and final RMSNorm. The error curve is
not materially repaired even though the disputed argmax changes. Enabling only
the three routes on portable policy yields `0.002511601`, `0.004042406`,
`0.006887352`, and `0.015432001` at the same boundaries. The group is therefore
not sufficient to induce the qualified failure, while disabling the group is
sufficient to change that failure's argmax.

The complete lifecycles lasted 226.798 and 89.987 seconds; the first includes a
67-second cold release build. All six arms stayed at or below 60.125 C, every
pacing and cooldown interval completed, cgroup event and swap counters remained
zero, and no owned residue remained. Kiln cgroups peaked at 14,658,469,888 and
12,110,426,112 bytes. The next discriminator must test each fused norm/MLP leaf
individually as a qualified-policy fallback. This evidence does not authorize a
qualified-policy change by itself.

The three individual fused-route fallbacks are retained at
`qualification/oracle-results/rocm/strix-halo/20260719t042441-rocm-strix-halo-hf-layer-attribution-fused-rmsnorm-fallback-v1.json`,
`qualification/oracle-results/rocm/strix-halo/20260719t042746-rocm-strix-halo-hf-layer-attribution-fused-mlp-silu-mul-fallback-v1.json`,
and
`qualification/oracle-results/rocm/strix-halo/20260719t043056-rocm-strix-halo-hf-layer-attribution-fused-mlp-gate-up-prefill-fallback-v1.json`,
all from clean pushed source `d956487224dcef59b4a78c4459a2cbc020cadc7b`.
Declining fused RMSNorm alone is bit-for-bit identical to the three-route
fallback: it restores token `15787`, final logit hash
`sha256:f2dee0a6c8ef4514ed5759ad5d92c9d6e6ae012db8c87a98cd6ad49ceb13ba62`,
and relative RMSE `0.014672443`, `0.036750649`, `0.072686307`, and
`0.085907010` at layers 0 through 2 and final RMSNorm. Declining either fused
MLP route alone is bit-for-bit identical to qualified: both emit token `25045`,
retain final logit hash
`sha256:b5acbda785044ca46d6cddb9aea03258dd4f99fb1904dcb5983be49ef68fd603`,
and reproduce qualified relative RMSE `0.014871503`, `0.036848969`,
`0.065551177`, and `0.085119899`. Within this complete three-route set, fused
RMSNorm is the only necessary route for the observed qualified-composition
failure; the two fused MLP routes are excluded.

The complete lifecycles lasted 221.965, 165.417, and 169.198 seconds; only the
first includes a 67-second cold release build. All nine arms stayed at or below
60.25 C, every pacing and cooldown interval completed, cgroup event and swap
counters remained zero, and no owned residue remained. Kiln cgroups peaked at
14,658,228,224, 14,657,458,176, and 14,658,273,280 bytes. This evidence permits
disabling fused RMSNorm in the qualified profile for a source-bound repair
candidate. The changed profile must still pass a fresh production next-token
boundary plus multi-token serving, throughput, and endurance qualification
before release acceptance.

The repaired production `qualified` profile is requalified at
`qualification/oracle-results/rocm/strix-halo/20260719t050322-rocm-strix-halo-hf-layer-attribution-qualified-rmsnorm-repair-v1.json`
from exact clean pushed source `62a9a0c243567768c5f583cd9009a431f2054b09`.
It restores the HF/vLLM token `15787` and reproduces the isolated RMSNorm
fallback's final logit hash
`sha256:f2dee0a6c8ef4514ed5759ad5d92c9d6e6ae012db8c87a98cd6ad49ceb13ba62`
plus relative RMSE `0.014672443`, `0.036750649`, `0.072686307`, and
`0.085907010` at layers 0 through 2 and final RMSNorm. This proves that the
typed production profile installs the intended repaired composition at the
known 166-token boundary. It does not show a general reduction in numerical
error and does not replace multi-token or serving parity.

The complete fingerprint, HF, and Kiln lifecycle lasted 226.804 seconds,
including a 68.615-second cold release build. The HF and Kiln arms peaked at
9,487,720,448 and 14,658,351,104 cgroup bytes, respectively; every high, max,
OOM, OOM-kill, swap, and thermal-guard counter remained zero. HF and Kiln
peaked at 59.375 C and 60.375 C, completed all pacing and cooldown intervals,
and left no process or service residue. The result uses the current 20 GiB
layer-reference containment contract and closes the production next-token
boundary. The independent 16 GiB next-token runner is an HF-only reference arm:
it never starts Kiln or installs a ROCm kernel profile, so rerunning it cannot
requalify the repaired production composition. Exact multi-token public serving
parity is the next correctness gate.

That next gate failed on exact clean pushed source
`b392a74dced1b0969c0b2e12cd50bcd2348963a3`. The retained driver-v7 receipt is
`benchmarks/receipts/rocm/strix-halo/20260719t052911-rocm-strix-halo-greedy-c1-rmsnorm-repair-failed-v1.kiln.json`.
Its complete 64-token request passed the request, fixed-length, memory, and
server-error gates, but an offline invocation of the driver's source-paired
exact-output comparator against the retained vLLM arm reports a mismatch at
UTF-8 byte 15. Kiln begins `To establish a baseline` while vLLM begins
`To establish a foundation`; disabling fused RMSNorm therefore repairs the
pinned 166-token boundary but does not establish multi-token parity.

The same attempt failed closed at the independent host-safety boundary. The
measured window reported 4.259 output tokens/second and 4.258 thermally
sustainable output tokens/second, but the process-group guard recorded one
hard trip at 93.125 C and a 93.25 C peak from post-measurement thermal inertia.
It terminated the server, completed cooldown at 43.375 C, removed the private
model snapshot, and left no process or listener residue. Model fingerprinting before
server creation also peaked at 92.875 C, outside continuous guard ownership.
These values are counterevidence, not performance results. Further ROCm serving
execution is gated on containing provenance hashing and thermal overshoot,
preserving reference comparison on otherwise complete failed runs, and
localizing the new first-token divergence with the guarded layer oracle.
Reference comparison is independent of lifecycle acceptance: a complete
output comparison must remain visible even when the thermal verdict fails.

The next bounded discriminator changes one typed field rather than repeating
the 64-token attempt. Launch
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-graph-disabled-v1.json`
with the same pinned prompt and stop after four generated tokens. Its TOML is
the ordinary Strix Halo comparison profile with only
`accelerator.rocm_graph_mode = "disabled"`; the repaired `qualified` kernel
profile, batching policy, memory policy, model, and server settings remain
unchanged. The expected fourth token is the HF/vLLM token `15787`
(` foundation`); token `25045` (` baseline`) reproduces the public-serving
defect outside graph replay. This run is source-bound diagnostic evidence and
cannot satisfy output parity, performance, graph, soak, or ROCm acceptance.

That graph-disabled discriminator is retained at
`benchmarks/receipts/rocm/strix-halo/20260719t060940-rocm-strix-halo-greedy-c1-graph-disabled-counterevidence-v1.kiln.json`
from exact clean pushed source `357403724843ae6c0f07349747d8b37d64e9c527`.
The measured request reproduces the exact 163-token prompt-set hash
`sha256:faf24ebb93fc7e75a2e78111b921a32aece716d56d9b67d2907ae251217a8d9e`
from both prior engine arms and still emits `To establish a baseline`.
Therefore HIP graph execution is not the cause of the pinned first divergence.
The warmup emitted `foundation`, but its prompt had 160 tokens and a distinct
hash; do not interpret it as repeated-request evidence.

Driver v8 kept provenance hashing and server execution inside the typed thermal
contract. The initial fingerprint, server, and final fingerprint peaked at
59.625 C, 60 C, and 60 C, respectively, with zero hard trips and complete
cooldowns. The measured request peaked at 51.625 C without pacing, the server
shut down normally, and no listener, process, or snapshot remained. Its 6.608
output tokens/second and 6.562 thermally sustainable output tokens/second are
diagnostic-only.

The next historical one-field discriminator launched
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-graph-disabled-no-prefix-cache-v1.json`.
Relative to the graph-disabled arm it changes only
`prefix_cache.enabled = false`, retaining the same batching actor, repaired
kernel profile, memory policy, model, prompt, and four-token bound. A
`foundation` result attributes the defect to prefix-cache KV/recurrent-state
reuse; `baseline` excludes that subsystem and advances the comparison to the
then-distinct actor/direct serving boundary.

The no-prefix-cache result is retained at
`benchmarks/receipts/rocm/strix-halo/20260719t061953-rocm-strix-halo-greedy-c1-no-prefix-cache-counterevidence-v1.kiln.json`
from exact clean pushed source `e1c2b91a1f25dc5d4769501c89b2388ad5792306`.
Its exact 163-token measured prompt still emits `To establish a baseline`, so
cross-request prefix-cache KV and recurrent-state restoration are excluded from
the pinned divergence. Initial fingerprint, server lifecycle, and final
fingerprint peaked at 59.375 C, 59.875 C, and 59.75 C with zero hard trips and
complete cooldown. The measured request peaked at 53 C without pacing, memory
peaked at 35,214,360,576 bytes, shutdown was normal, and no listener, process,
or snapshot remained. Its 6.738 output tokens/second and 6.706 thermally
sustainable output tokens/second are diagnostic-only.

The following historical one-field discriminator launched
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-graph-disabled-no-prefix-cache-no-batching-v1.json`.
Relative to the no-prefix-cache arm it changes only
`batching.mode = "disabled"`. The direct streaming rendezvous remains `auto`
and therefore uses the ROCm backend's typed policy. A `foundation` result
attributes the defect to actor-owned prefill/decode scheduling; `baseline`
excludes the actor and advances the comparison to the direct rendezvous/model
forward boundary. The no-batching launch/config pair is now retired and deleted;
current source always constructs the actor for a real backend. Its receipt is
preserved solely as source-bound localization evidence.

The first direct-path attempt retained
`benchmarks/receipts/rocm/strix-halo/20260719t062650-rocm-strix-halo-greedy-c1-direct-driver-v8-diagnostics-failed-v1.kiln.json`.
Its four-token warmup completed and emitted `To establish a foundation`, but
driver v8 failed the warmup verdict before the pinned 163-token measurement
because it unconditionally required `decode_runtime.batching_engine`. The
historical typed arm had disabled that actor, so the health response correctly omitted it
and exposed the live direct-rendezvous worker instead. This is failed harness
evidence only: the distinct 160-token warmup cannot answer the measured prompt's
correctness discriminator. Initial fingerprint, server, and final fingerprint
lifecycles stayed below 60 C, all 35 server pacing intervals completed, and
shutdown/cooldown left no listener, process, or snapshot residue.

Historical serving benchmark driver v9 retained the v8 provenance containment and made
server diagnostics route-aware before retrying that arm. Driver v8 runs both
the initial and final model fingerprints
as start-gated child process groups under the same typed host thermal policy,
requires complete pacing and post-exit cooldown for each, and retains both
closed lifecycles plus implementation and Python hashes in
`host_thermal.model_fingerprint`. Owned mode shuts down and cools the accelerator
server before the final full rehash. A missing initial lifecycle is invalid; a
missing final lifecycle requires a failed model-identity finalization check.
Driver v9 additionally binds universal request-status deltas, effective
actor/direct ownership, batching-engine deltas when that actor exists, and
direct-rendezvous deltas when that worker exists. Exact server request
accounting and route-local error/budget gates now apply to either route without
inventing zero actor counters. Strict-valid driver-v7 and driver-v8 references
remain comparison-compatible, so the retained vLLM arm does not need an
expensive rerun merely to exercise either safety repair.

The driver-v9 retry is retained at
`benchmarks/receipts/rocm/strix-halo/20260719t065243-rocm-strix-halo-greedy-c1-direct-rendezvous-actor-exclusion-v1.kiln.json`
from exact clean pushed source `b9af5eecd35b7c9a64f6a977eaa71bec9b55831b`.
The measured prompt has the exact 163-token count and
`sha256:faf24ebb93fc7e75a2e78111b921a32aece716d56d9b67d2907ae251217a8d9e`
prompt-set hash used by the actor-enabled counterexamples, and it emits
`To establish a foundation`, matching HF/vLLM at generated token index three.
Both warmup and measurement route through `direct_streaming`; each reports
three submitted/executed direct-rendezvous rows, three runner calls, width one,
zero busy/failure jobs, no runner-call-budget violation, one server `ok`, and
zero request errors, timeouts, rejections, or active requests at the boundary.
The batching-engine record is correctly null.

Graphs and prefix caching were disabled in both sides of this historical
one-field comparison. The actor-enabled arm emits `baseline`; changing only
`batching.mode` to `disabled` emits `foundation`. This causally localizes the
pinned serving divergence to the batching actor's prefill/decode path and
excludes the direct streaming/rendezvous/model path under this request. It does
not yet distinguish actor prefill from actor decode/state assembly, establish
64-token parity, or qualify concurrency, performance, or endurance. The
four-token measurement reported 9.396 output tokens/second and 9.292 thermally
sustainable output tokens/second without request-window pacing; both are
diagnostic-only. Initial fingerprint, server, and final fingerprint peaks were
59.5, 60.0, and 59.875 C, every lifecycle and cooldown completed without a
trip, and no listener, process, or snapshot remained.

The actor-enabled parent records three prompt-token chunks, 24 prefill
forwards, 21 layer yields, and 96 completed transformer layers for the exact
163-token prompt. The retired direct route has no actor prefill counter. To separate
actor token chunking from layer resumption, launch
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-graph-disabled-no-prefix-cache-prefill-256-v1.json`.
Relative to the actor-enabled no-prefix parent, it changes only
`server.max_prefill_tokens_per_cycle = 256`; graphs stay disabled, prefix
caching stays disabled, the actor stays enabled, and the four-layer quantum is
unchanged. The prompt must therefore complete as one token chunk across eight
layer-group forwards and seven yields. `foundation` makes the 64-token chunk
boundary causal. `baseline` excludes token chunking and moves the next repair
to layer resumption or the remaining actor handoff. This bounded four-token arm
cannot qualify performance, concurrency, or endurance.

The exact-prompt result is retained at
`benchmarks/receipts/rocm/strix-halo/20260719t071900-rocm-strix-halo-greedy-c1-actor-prefill-single-chunk-v1.kiln.json`
from exact clean pushed source `3375fb714e6782ec722746fa30c39b85d05445a8`.
The run deliberately reuses the prior run ID because fixed serving profiles
include it in the prompt. This restores the exact 163-token input and
`sha256:faf24ebb93fc7e75a2e78111b921a32aece716d56d9b67d2907ae251217a8d9e`
prompt-set hash. An earlier invocation with a longer run ID produced a
167-token prompt and is rejected as causal evidence rather than silently
compared.

The exact arm emits `To establish a foundation`, matching the direct and
HF/vLLM arms where the 64-token actor parent emits `baseline`. The route-aware
record proves the batching actor remained active, processed one 163-token
chunk across eight prefill forwards, seven layer yields, and 32 layers, then
ran three one-row decode forwards with no error. Since layer resumption remains
active, the one changed prefill-token boundary is causal and the layer-yield
implementation is excluded for this discriminator. The request reported 9.493
output tokens/second, 9.401 thermally sustainable output tokens/second, 235.7
ms TTFT, and 62.1 ms p99 ITL without request-window pacing; these are
diagnostic-only. Initial fingerprint, server, and final fingerprint lifecycles
peaked at 59.875, 60.125, and 59.75 C with zero trips, complete cooldowns,
normal shutdown, and no listener, process, or snapshot residue.

The repaired production input is now
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-v2.json`.
Its TOML differs from the preserved v1 production profile only by changing the
actor prefill ceiling from 64 to 256. The backend policy simultaneously makes
256 the ROCm tiled streaming-prefill threshold and base/tape tile. A typed
`actor_prefill_tile_alignment_required=true` diagnostic explains the
fail-closed startup checks: the actor ceiling must equal the effective tile,
tiled streaming prefill must cover the first split, and `max_batch_tokens` must fit
the tile plus the effective decode width. This prevents configuration drift
from reintroducing route-dependent deterministic output. Historical 64-token
actor profiles remain receipt-bound evidence and are not rewritten; a repaired
binary rejects them when the actor is effective. The v2 profile is not accepted
until exact rebuilt-source c1 parity, longer output parity, and concurrent
mixed-prompt qualification pass on this Strix Halo.

An attributed argmax identifies which engine selected the eager HF reference's
top token at the first divergence. It does not prove multi-token parity, explain
the losing engine's numerical defect, accept either thermal pacing policy for
performance, or replace the complete concurrency/profile matrix.

Model-serving workloads additionally require `--model` with the exact local
model directory and `--model-id` with its public identity. Select each declared
A/B arm explicitly; the manifest, not an ambient environment variable, owns
the effective configuration recorded in the receipt.
Every passed serving, performance, training, eval, or soak receipt contains the
complete validated model identity. If the initial fingerprint fails after a
run identity has been created, the failed receipt instead records
`model: null`, fails its declared cases without executing them, and retains the
bounded fingerprint evidence. That pre-model receipt proves only the failed
boundary and cleanup; it is never model or workload-execution evidence.

The source-bound serving drivers materialize that declared policy as a private TOML
file inside the ignored run directory and start `kiln serve --config <file>`.
Server profile, bind address, model/snapshot/adapter paths, thinking default,
transport bounds, scheduling ceilings, logging, memory reclaim, synchronization,
graph mode, graph entry capacity, graph byte capacity, and Vulkan decode-weight
prewarm enable/rate policy therefore travel
through the same typed parser and source diagnostics as an operator config.
The process environment is scrubbed of ambient `KILN_*` controls before build
and launch. `memory.kv_autoscale` now carries both enabled and disabled requests.
Ordinary serving arms write `memory.kv_force_blocks = 0`; the dedicated
maintenance arm writes its declared positive target. Health and debug must
report `config_file` provenance for both fields. The same TOML sets
`server.debug_model_state = true` to grant trusted readback without enabling
eval-mode request semantics. No `KILN_*` launch exception remains; `RUST_LOG`
is the ordinary tracing filter. Qualification rejects every ambient runtime
control, including the deprecated `KILN_KV_AUTOSCALE` and
`KILN_KV_FORCE_BLOCKS` aliases.

The mixed-load and development-soak drivers bind the response policy
`ascending_zero_padded_integers_prefix_v1` in their effective configurations.
For every ordinary, warmup, stabilization, measured, and intentionally
cancelled request, the client concatenates streamed `delta.content` fragments
in order and requires the result to be a nonempty prefix of `000000 000001
000002 ...`. Tokenizer-dependent fragment boundaries and a final partial
integer are allowed; repeated numbers, punctuation, newlines, commentary,
reasoning content, tool calls, empty special-token output, multiple choices,
and malformed semantic events are not. Protocol success, positive usage, and
exact length termination are therefore necessary but not sufficient for a
passing request. A row that fails this oracle is excluded from successful
latency and throughput aggregates and increments the request-failure count.
Failure details retain the oracle reason, exact accepted token IDs when
performance metadata is enabled, and an escaped output excerpt capped at 256
characters. This bound preserves actionable corruption evidence without
allowing model output to exhaust the result-detail envelope.

The measured mixed-load prompt identity is
`variant_invariant_fixed_output_v5`. It requests exactly 64 ascending six-digit
integers and tells the model that server truncation before the target is
expected. At roughly seven model tokens per formatted integer, the target
remains beyond the 32-, 128-, and 256-token measured response caps without
inviting the output-limit refusals observed at one million and 1,024 requested
integers. The instruction requires immediate sequence output and forbids early
stopping, end markers, explanation, refusal, summaries, or discussion of output
limits. Prompt-length padding uses nonnumeric `itemNN` tokens and explicitly has
no relationship to response length. The separate unvalidated stalled-socket
request retains a 1,024-integer fill target so its 4,096-token response envelope
does not weaken. Every arm records the prompt identity,
`response_oracle_target_integer_count = 64`,
`slow_response_target_integer_count = 1024`, and
`long_prefill_marker_role = "long-prefill"`. The diagnostic and acceptance
paths therefore tokenize the same named long request; a wording, denominator,
fill target, or marker-role change invalidates direct A/B configuration identity.

Mixed-load failure results are cumulative evidence, not a replacement set of
zeros. The driver records the deterministic measurement boundary before the
first dispatch, every attempted measured request name, every returned worker
result regardless of terminal success, and the latest successfully read `/health`
snapshot obtained while the measured window remains active. Worker cleanup
harvests completed futures even when an earlier future, health poll, server
exit, timeout, or thermal trip raised the primary error. Runtime counter deltas
for batching, graphs, prefix-cache activity, external-yield synchronization,
and the ROCm W8 LM head are then computed against the post-warmup baseline and
the last valid snapshot. On failure those deltas are explicit observed lower
bounds, not an inferred terminal snapshot. Regressing or malformed counters make partial-evidence
serialization fail explicitly; they are never coerced into plausible values.

`completion_token_count` and `output_token_throughput_per_second` remain the
strict numerator and rate for fully qualified measured responses only.
`observed_completion_token_count` instead counts validated per-token timing
records from every returned measured stream, including an interrupted stream
that never emitted terminal usage. `observed_output_token_throughput_per_second`
divides that diagnostic count by `measurement_duration_seconds`, spanning the
measured start through the failure boundary or last returned result. It is
useful for a rejected-run bottleneck comparison but cannot satisfy a
performance or correctness gate. `request_count` records attempted measured
requests; an attempted request with no qualified result contributes to
`request_failure_count`. A failure before measurement still uses the explicit
zero-measurement sentinel and one case-level request failure. Thermal,
shutdown, cooldown, and residue evidence is merged after teardown in both
paths, so the causal error and containment result survive together.

After the ordinary mixed-load measurement window, every ROCm arm also runs a
separate fixed-seed sampled profile at concurrency eight: 32 tokens per request,
temperature 0.7, top-p 0.9, top-k 40, and min-p 0.0. Those requests disable
thinking, ignore EOS, and retain terminal performance metadata, but use a
sampling-appropriate semantic oracle: exactly one nonempty plain-text choice,
with no reasoning content or tool calls. Random output is not compared with the
greedy ascending-integer sequence. The driver drains the deterministic work and
captures its terminal health snapshot before dispatching the sampled wave, then
waits for all eight sampled streams and the batching engine to drain and
reattests runtime policy. It finally replays `normal-00` with the exact original
prompt and seed and requires the same token vector. Sampled counters and
latencies therefore remain dedicated evidence without contaminating the
long-standing deterministic denominator, while the replay canary detects any
process-state corruption caused by randomized sampling.

The mixed-load client and server each use the same 180-second per-request
containment bound. Cooling remains inside that wall clock. The alignment lets
the deliberately long prefill reach the server's terminal result under required
host pacing instead of having the client abandon a still-valid request at an
older, shorter local deadline; it does not remove or normalize the measured
TTFT, E2E latency, workload duration, or sustainable throughput cost.

Measured mixed-load, development-soak, and endurance receipts also retain the
complete validated request-phase population instead of discarding terminal
performance metadata. Every fixed phase `P` has a
`latency_phase_P_ms_total` and `latency_phase_P_request_count`; nullable phases
contribute neither time nor count, while a measured zero contributes one count
and zero time. `latency_phase_metadata_missing_count` must remain zero. Broad
actor phases contain narrower backend candidates and delivery can overlap the
next forward, so these totals rank candidates within their documented layer;
they must not be summed into a synthetic critical path. This makes pre/post
optimization receipts sufficient to decide whether sampling, readback,
synchronization, queueing, or model execution actually moved.

The sampled wave emits `sampled_profile_*` request, fixed-output, phase,
throughput, and batching metrics. It requires a measured `sampling_ms` value on
all eight ROCm requests, positive total sampled-tail and actor-decode duration,
zero separate `readback_ms` observations, no batching errors, at least one
batched decode forward, more decode rows than forwards, and an observed batch
width of at least two. `sampled_profile_output_token_throughput_per_second` is
the aggregate 256-token completion count divided by the concurrent wave wall
window. `sampled_profile_per_request_output_token_throughput_per_second_p50` is
the median of each request's completion count divided by its own end-to-end
wall time. Neither is a transformer-only kernel rate, and the narrower sampling
phase remains contained by the broad actor `decode_ms` interval.

The ROCm token-only LM-head candidate requires a second, independent route
proof identified in effective workload configuration as
`rocm_w8_lm_head_route_evidence = "health_counter_delta_v1"`. The mixed-load
driver strictly parses all ten fields in
`/health.decode_runtime.rocm_w8_lm_head` and snapshots them at both measurement
boundaries. Missing, additional, negative, non-integral, regressing, or
internally inconsistent fields fail the case.

For the ordinary deterministic window,
`rocm_w8_lm_head_argmax_dispatch_count` must be positive and
`rocm_w8_lm_head_argmax_row_count` must cover every accepted completion token.
Both ordinary sample counters and the dispatch-failure counter must remain
zero. Each sampled request's first completion token is emitted by prefill
before the request enters decode-continuation scheduling. The eight-request,
32-token sampled window must therefore report exactly 248 fused decode rows in
`sampled_profile_rocm_w8_lm_head_sample_row_count`, while the response oracle
and usage accounting independently require all 256 completion tokens. Sample
dispatches must be positive and fewer than rows, proving multi-row execution.
Every sample dispatch must be W8A8, with zero W8A16, argmax, sparse-history,
or failed dispatches. The process-lifetime maximum top-k must advance from exactly zero
to exactly 40, and maximum token-only batch width must be at least two. These
counters prove route selection; request usage, semantic validation, phase
accounting, the post-sampled replay canary, and the exact output/reference gates
remain independently authoritative for results.

The focused gfx1151 kernel test is an independent mathematical discriminator,
not a serving substitute. It projects an identity-shaped 64-token fixture
through both W8A16 and W8A8, covers batch widths 1, 2, 4, and 6 and `top_k`
through 64, and compares penalties, stable tie ordering, softmax, min-p, top-p,
and the SplitMix64 categorical draw against a CPU implementation. Repeating the
same six-row seeded batch must return the exact same vector. Serving promotion
still requires the source-bound sampled wave, ordinary correctness, the
30-minute development soak, and the comparable performance row.

Every ROCm mixed-load arm applies the same independent host thermal policy from
server launch through readiness, warmup, the isolated sampled wave, ordinary
measurement, drain, server exit, and a bounded post-exit cooldown. The typed
`host_safety` object selects exactly one Linux hwmon input by `name=k10temp`
and `label=Tctl` and polls every 250 ms. Runtime protection is
hard-limit-only: a 97,000-millicelsius reading, missing or ambiguous selector,
malformed input, controller error, or termination error fails closed and
terminates the server group. It never sends `SIGSTOP` while accelerator work
may be outstanding.

The checked Strix Halo mixed-load profile uses one source-bound cooperative
operating point across all six policy arms. `server.max_decode_batch = 4`,
`server.max_prefill_staging_slots = 4`, and
`server.max_active_requests = 8` separate ordinary decode width from admitted
prefill capacity. `server.max_prefill_tokens_per_cycle = 256` and
`server.max_prefill_layers_per_cycle = 8` bound each admitted prefill to one
quarter of the Qwen layer stack before another scheduling opportunity.
`batching.actor_cycle_idle_ms = 50` inserts one
cooperative wait after an actor cycle advances model work. The server receives
that policy through the private TOML launch file; both health and trusted debug
state must report the exact value with `config_file` provenance. The 97 C hard
limit, request set, token counts, model-startup pacing, memory bounds, graph and
cache policies, and correctness gates are unchanged.

This is a qualification-profile candidate, not a new product-wide default or
an accepted performance result. The former width-eight, four-layer, zero-idle
profile reached the independent 97 C limit during real mixed inference. The
width-four, 32-layer, 50 ms profile stayed contained at 89.75 C and completed
all deterministic and sampled requests, but fourteen nonthermal ITL outliers
coincided with long prefill, decode, and graph work; p99/p99.9 ITL reached
972.672/1,849.384 ms. That source-bound receipt rejects a full-stack prefill
quantum rather than accepting explained pauses. The eight-layer successor is
the single declared correction: it retains the contained decode duty cycle but
adds three inter-layer scheduling opportunities per Qwen prefill. It must be
committed and pushed before hardware execution, and only its own source-bound
receipt may establish thermal containment, fused-LM-head route coverage,
throughput, or latency.

The first source-bound eight-layer `autoscale-off` discriminator passed. Its
strict receipt is
`qualification/receipts/rocm/strix-halo/20260720t071927133129z-rocm-strix-halo-serving-mixed-rocm-v1-184c082f9e-v1.json`.
All ten deterministic requests produced 1,312 tokens at 12.748 tokens/second;
all eight sampled requests produced 256 tokens at 8.055 tokens/second. The
prefill path executed 260 forwards, 2,080 layers, and 195 inter-layer yields,
with a 203.229 ms maximum prefill forward. No thermal-attributed,
nonthermal-attributed, or unexplained ITL outlier remained; p99/p99.9 ITL fell
to 821.918/961.994 ms. The cost is visible rather than hidden: aggregate output
fell 8.2 percent relative to the rejected 32-layer row, p99 TTFT rose to 88.748
seconds, and p99 E2E rose to 102.676 seconds. The package peaked at 88.75 C
without a trip or pacing event. This passes the declared discriminator but does
not promote the operating point: the remaining policy arms and the 30-minute
development soak still have to pass from committed source.

The source-bound `default` arm also passed from the receipt checkpoint commit:
`qualification/receipts/rocm/strix-halo/20260720t073015406855z-rocm-strix-halo-serving-mixed-rocm-v1-f2e983b84c-v1.json`.
It attested requested and effective KV autoscaling with graphs enabled. The
bounded workload did not trigger a physical resize or reclaim: KV capacity
remained exactly 4,096 blocks and resize/reclaim latency stayed zero. All three
ITL outlier populations remained zero, aggregate output was 12.935
tokens/second, and the package peaked at 89.75 C without pacing or a trip. This
closes the autoscaling-enabled arm for this workload; it is not evidence that a
separate forced-resize workload is unnecessary.

The paired source-bound `graphs-off` arm passed at
`qualification/receipts/rocm/strix-halo/20260720t073935606283z-rocm-strix-halo-serving-mixed-rocm-v1-c5256e3c8c-v1.json`.
All graph counters were exactly zero. Relative to `default`, deterministic
output rose from 12.935 to 13.710 tokens/second, sampled output rose from 8.319
to 14.247 tokens/second, and p99/p99.9 ITL fell from 797.113/1,037.370 to
474.283/508.025 ms. The maximum decode forward fell from 681.762 to 151.756 ms.
The graph-on arm had spent 3.698 seconds warming 21 candidates and 1.532
seconds in native capture; successful replay did not repay those costs in this
short dynamic-shape workload. Peak device memory fell by 455,360,020 bytes and
the package peak fell from 89.75 to 89.0 C. This is evidence against assuming
graphs help this workload, not a cross-device or universal graph-disable claim;
the graph-specific correctness and resilience gates remain independently
required.

The source-bound `both-off` arm then passed at
`qualification/receipts/rocm/strix-halo/20260720t074836933014z-rocm-strix-halo-serving-mixed-rocm-v1-b68874d2e9-v1.json`.
It differs from `graphs-off` by disabling requested and effective KV
autoscaling. Neither arm resized or reclaimed. Their results were close:
13.813 versus 13.710 deterministic tokens/second, 14.400 versus 14.247 sampled
tokens/second, p99 ITL 481.584 versus 474.283 ms, and maximum decode forward
151.908 versus 151.756 ms. That pair provides no evidence that an armed but
inactive autoscaling monitor caused the graph-on pauses. It does not replace
the separate forced-resize workload, where the autoscaler actually mutates KV
capacity.

The paired `both-off-prefix-cache-off` arm failed from exact clean source
`69b687485247cc580a4ec501d564c0e0635ec462`; retain its strict receipt at
`qualification/receipts/rocm/strix-halo/20260720t075729232030z-rocm-strix-halo-serving-mixed-rocm-v1-66daf29769-v1.json`.
Nine deterministic requests and all eight sampled requests passed, but the
2,432-token `normal-07` prompt produced a coherent refusal instead of the
required ascending sequence. The response completed all 128 requested tokens,
so this is a semantic route failure rather than a timeout or zero-token result.
Every prefix-cache state, lookup, hit, lease, entry, block, and byte metric was
exactly zero, all graph counters were zero, all ITL-outlier populations were
zero, the package stayed below the hard guard at 90.25 C, and shutdown was
unforced and residue-free. An older source-bound run of the same variant at
`qualification/receipts/rocm/strix-halo/20260718t092608589093z-rocm-strix-halo-serving-mixed-rocm-v1-66daf29769-v1.json`
also failed deterministic correctness. Consequently, disabling the prefix
cache on the ROCm batching-actor path is not qualified for this workload. Do
not proceed to the stable arm or a soak, and do not rerun this arm unchanged,
until the fresh-prefill enabled/disabled route difference has been explained
and covered by a focused equivalence test. This evidence does not by itself
attribute the divergence to cached reuse: the failed arm proves the cache was
inactive, while the enabled arms may still differ in initialization,
registration, snapshot, or numerical routing before any cache hit.

The immediate source audit found that numerical-routing difference. The model
API's `capture_prefix_split` boolean did more than its name promised: it also
created the last block-aligned prompt chunk. For `normal-07`, the enabled miss
split the 2,432-token prompt at token 2,416, while disabled mode retained the
ordinary 128-token final quantum. Each completed GDN prefill chunk applies a
recurrent-state precision boundary, so a cache storage toggle changed model
numerics without serving a hit. Corrected source gives the prefill state an
explicit registration capability, preserves the canonical block-aligned split
regardless of that capability, and gates snapshotting, completed-prompt
registration, rolling snapshots, and publication on the capability alone. A
portable tiny-GDN integration runs both arms through 17-token/one-layer quanta
and requires identical chunk schedules, first tokens, recurrent states, and
following decode tokens while also requiring no disabled-arm registration or
snapshot. This closes the source-level coupling only. The prefix-cache-off arm
must pass again on ROCm from the clean pushed corrective source before stable
or soak execution resumes.

The first clean-source ROCm hardware discriminator for this correction passed
from pushed commit `e1ea3dc57e2fae8e3584f52c890aee3e247a52e4`. The tiny GDN
model executed registration-enabled and registration-disabled resumable
prefill on the Strix Halo ROCm backend. Both routes scheduled and completed
token quanta `[17,17,17,17,12,1]` across six layer-yield cycles, produced first
token 13 and following decode token 13, and satisfied recurrent-state parity.
This is direct device evidence for the corrected route contract, but it is
deliberately not a substitute for the production Qwen workload: the full
source-bound `both-off-prefix-cache-off` arm must still reproduce the formerly
failing 2,432-token request and every cache-zero, thermal, latency, and teardown
gate.

The full production-model replacement arm then passed from exact clean pushed
source `f818f4f25e0a8ef5e695ee75413f0183d2dd5f74`; retain its strict receipt at
`qualification/receipts/rocm/strix-halo/20260720t083646672237z-rocm-strix-halo-serving-mixed-rocm-v1-66daf29769-v1.json`.
The formerly failing 2,432-token `normal-07` request emitted the required
ascending sequence for all 128 tokens. All ten deterministic requests produced
1,312 tokens at 13.867 tokens/second, all eight sampled requests produced 256
tokens at 14.585 tokens/second, and the post-determinism canary matched. Every
prefix-cache lookup, hit, miss, token, block, entry, state-byte, lease, pending
release, and capacity observation was exactly zero, as were all graph counters
and all three ITL-outlier populations. p99/p99.9 ITL was 470.415/486.035 ms,
p99 TTFT was 80.715 seconds, and p99 E2E was 94.315 seconds. The package peaked
at 89.75 C without pacing or a guard trip; cooldown completed, the server exited
zero without force in 213.800 ms, and no process or snapshot residue remained.
The 848.187-second owned lifecycle does not describe inference latency:
562.251 seconds were the clean release rebuild, while the deterministic
measurement window was 94.616 seconds. This closes the prefix-cache-disabled
ROCm arm and unblocks the stable arm; it does not qualify a soak or promote the
operating point by itself.

The source-bound `stable` arm passed next at
`qualification/receipts/rocm/strix-halo/20260720t085444578521z-rocm-strix-halo-serving-mixed-rocm-v1-8dd0211c5b-v1.json`.
The typed profile retained operator requests for KV autoscaling, automatic
memory reclaim, and ROCm graphs while resolving their effective states to
false, `off`, and false respectively. KV capacity remained exactly 4,096
blocks, and resize, reclaim, graph warmup, capture, replay, live graph, and
transient graph bytes all stayed zero. All deterministic and sampled requests,
the post-determinism canary, pressure overlap, cancellation, fused routes,
external synchronization, and terminal actor-idle checks passed. Aggregate
deterministic output was 13.632 tokens/second, sampled output was 14.240
tokens/second, p99/p99.9 ITL was 479.783/488.150 ms, p99 TTFT was 82.002
seconds, and p99 E2E was 95.691 seconds. The cache was active and recorded one
128-token/two-block hit, 39 final entries, no active lease or pending release,
and internally consistent 1,073,479,680 bytes of recurrent state. The package
peaked at 88.875 C without pacing or a trip; cooldown completed, shutdown was
unforced and zero in 263.957 ms, and no residue remained. This qualifies the
stable mixed-load arm and authorizes the 30-minute development soak from the
same operating policy; it is not itself soak or endurance evidence.

The first exact-source development-soak attempt after that arm is retained at
`qualification/receipts/rocm/strix-halo/20260720t090457062502z-rocm-strix-halo-serving-rocm-development-053e89eca9-v1.json`.
It failed closed before measurement after 29 valid warmup requests across four
waves filled all 39 prefix-cache entries. A health snapshot reported the
batching actor drained while one ROCm graph slot or continuity timeline was
still active. The actor did remove a terminal row and publish its zero-active
snapshot before the model finish/discard boundary released graph ownership,
creating one real transient false-drained window. The later corrected-source
rerun described below proves that race was not the only cause of this rejection.
Measurement and accelerator-telemetry counters remain zero sentinels because
the measurement sampler never started. Host memory stayed
above 22,550,761,472 bytes, swap did not grow, package temperature peaked at
89.875 C below the 97 C guard, cooldown completed in 2.506 seconds, shutdown was
unforced and zero, and no process or snapshot remained. The rejected receipt
does not qualify the soak.

The correction keeps the last public batching snapshot conservative across both
normal `finish_request` and error/cancellation `discard_request` cleanup. The
actor still removes the row from its private scheduling vector before cleanup,
so it cannot be scheduled twice, but it does not publish that removal until the
model boundary has released graph, recurrent-state, prefix, and private-KV
ownership. Deterministic blocked-cleanup tests require `active_decode=1` with an
otherwise empty actor during each boundary and zero only afterward. A complete
1,091-test server library run, all 652 qualification-tooling tests, and bounded
ROCm/gfx1151 and Vulkan all-target checks pass. This is portable causal evidence;
the failed receipt remains rejected.

The exact pushed correction was rerun at
`qualification/receipts/rocm/strix-halo/20260720t092808480302z-rocm-strix-halo-serving-rocm-development-053e89eca9-v1.json`.
It again completed 29 valid warmup requests, four waves, and full 39-entry prefix
residency, then rejected the first stabilization drain with the same graph-slot
message before any measurement sample. The actor-ordering correction was active
and its conservative terminal boundary completed, isolating a separate telemetry
contract error introduced with native batched graphs: a persistent slot reserved
for a batch width has `batch_size=Some(width)` even while it owns no live request,
but `active_graph_slot_count` counts that reservation as active. The field's
documented contract is live logical decode-row ownership; reusable retained slots
belong in the idle count. This was a fail-closed observability rejection, not a
device fault, request error, leak, or throughput result. Host memory remained
above 21,735,788,544 bytes, swap growth was zero, package temperature peaked at
89.5 C below the guard, cooldown completed in 2.757 seconds, shutdown was
unforced and zero, and no process or snapshot remained. Correct the active/idle
slot classification and cover retained batched slots before another device run.

The portable correction derives graph-slot liveness solely from
`assigned_row`: a retained slot reserved by `batch_size` is idle between
cohorts, while its graph, recurrent state, width mapping, and exact retained
bytes remain intact. The total remains `active + idle`, and the independent
tracked-owner counter continues to expose live continuity timelines. The
width-two retained-slot regression now requires one total, zero active, and one
idle slot after two unrelated cohorts refresh the same tensor handles. All 405
enabled ROCm-feature model tests, 1,091 server tests, and 652 qualification tests
pass; the ROCm/gfx1151 and Vulkan all-target server feature graphs compile.
Dashboard, architecture, API, qualification, and troubleshooting references use
the same live-ownership definition. This removes a fail-closed telemetry false
positive.

The prior exact clean pushed-source hardware rerun passed from
`a130e975f42e7c98a689bc309cea5f1d6eaa9a28`; retain its strict receipt at
`qualification/receipts/rocm/strix-halo/20260720t095657771585z-rocm-strix-halo-serving-rocm-development-053e89eca9-v1.json`.
After 29 warmup requests and six stabilization cycles, it completed 804 valid
requests, 25,728 output tokens, 111 waves, and 22 confirmed cancellations over
1,803.201 measured seconds at 14.268 aggregate output tokens/second. All 135
ITL outliers were attributed and none were unexplained; p50/p99/p99.9 ITL was
189.397/660.015/884.196 ms and p50/p99 TTFT was 5.060/21.534 seconds. The run
captured 1,164 graphs, replayed 7,458, reused retained slots 197 times, and ended
drained with zero active and three idle slots. It recorded zero request,
batching, graph, device, synchronization, non-finite, KV-ownership, cache-lease,
or residue failures. GPU allocation grew 1,372,160 bytes with a 61,463,376-byte
peak delta; RSS grew 59,527,168 bytes; host swap did not grow; and available
host memory stayed above 21,756,338,176 bytes. Active accelerator samples had
57-percent median busy and 2.389 GHz median SCLK, with no sample below half the
advertised maximum. Package temperature peaked at 90.75 C below the 97 C hard
guard, cooldown completed, and shutdown was unforced and zero. This remains a
retained high-throughput historical row, not the current performance headline.

The current exact clean pushed-source rerun at `3887b640c` also passed. Retain
`qualification/receipts/rocm/strix-halo/20260720t144932814398z-rocm-strix-halo-serving-rocm-development-053e89eca9-v1.json`.
It completed 688 requests, 22,016 output tokens, 95 waves, and 19 cancellations
over 1,808.128 measured seconds at 12.176 aggregate output tokens/second. P50/p99
ITL was 194.493/907.601 ms and p50/p99 TTFT was 6.773/25.496 seconds. All 521
long gaps had runtime-event attribution and none was unexplained. Request,
batching, graph, device, synchronization, finite-value, ownership, memory,
thermal, shutdown, and residue gates passed. KV capacity stayed fixed at 4,096
blocks; GPU allocation grew 13,334,052 bytes with 66,288,624-byte peak growth;
RSS grew 51,048,448 bytes and swap did not grow. Package temperature peaked at
89.625 C with zero thermal pacing.

That throughput is 14.66% below the prior row, so the current result was paired
with the same `autoscale-off` mixed workload on old source `a130e975f` and
current source. Old/current deterministic throughput was 12.662/12.432 and
sampled throughput 8.144/7.945 tokens/second, only 1.82%/2.45% apart. This is
below the 3% material source-regression threshold and does not implicate the
compact prompt-logprob route, which is inactive for these requests. The public
number remains the slower 12.176 result, and the larger actor decode/prefill plus
graph-sync outlier population remains explicit. This closes the current
material-change correctness, 30-minute development, and benchmark gate. It does
not replace the final 24-hour soak, characterize the variance distribution, or
establish competitive large-batch performance.

The later exact clean sampler-readback source `91f22bcc8` retained both sides of
the same variance instead of selecting the faster row. Its development receipt
at `qualification/receipts/rocm/strix-halo/20260721t121713716043z-rocm-strix-halo-serving-rocm-development-053e89eca9-v1.json`
passed correctness, graph, fixed-capacity memory, thermal, shutdown, and residue
gates over 1,816.003 measured seconds, but produced 12.775 aggregate output
tokens/second, 13.05% below the immediately preceding 14.692 row. All 725
requests published positive exact sampler readback totaling 376,397.090 ms, all
371 ITL outliers had runtime-event attribution, KV stayed at 4,096 blocks,
autoscaling and reclaim stayed disabled, and GPU growth remained zero. The
receipt is therefore valid readback/safety evidence and a performance
counterexample, not proof of a code regression or VRAM rebalance.

The canonical `autoscale-off` exact-prompt discriminator at `610f8dc27` is
retained at `qualification/receipts/rocm/strix-halo/20260721t131545614136z-rocm-strix-halo-serving-mixed-rocm-v1-184c082f9e-v1.json`.
It produced 12.822 deterministic and 8.440 fused-sampled output tokens/second.
Against the most recent same-variant pre-readback control at `14e3f3e4d`,
12.568/8.576, those changes are +2.02%/-1.59%, both inside the existing 3%
material source-effect threshold; p99 ITL improved from 779.484 to 653.534 ms.
Every 10 deterministic and eight sampled measured requests reported positive
readback, graph capture/replay succeeded 20/475 times without failure, and the
run drained cleanly. The strict comparator does not issue a formal verdict
because the expanded metric contract changed the committed workload hash. This
is a manual same-variant source discriminator, not a cross-hash policy pass, and
it does not erase the slower development receipt or characterize long-run
variance.

Five measured-window metrics close the cooperative-idle evidence. The configured
gauge is `batching_actor_cycle_idle_ms_configured`. The monotonic count and
elapsed-time deltas are `batching_actor_cycle_idle_count` and
`batching_actor_cycle_idle_ms_total`.
`batching_actor_cycle_idle_ms_max_end` is the process-lifetime maximum observed
at the terminal snapshot, while `batching_actor_cycle_idle_active_end` records
whether a wait remained active after final drain. A passing ROCm mixed-load
case requires the configured value to equal 50, positive count, total, and
maximum evidence, and an inactive terminal state. The start and end gauges must
also agree exactly. Counter regression, missing fields, non-finite values,
wrong provenance, a silently unexercised wait, or an active wait after drain
fails the case. Drain polling treats the actor's cooperative-idle gauge as
active work, so a final response cannot race the terminal snapshot while the
last 50 ms wait is still in progress. On an interrupted run the latest valid
deltas remain explicit lower-bound diagnostics, but they cannot turn the failed
run into an accepted thermal or performance result.

The ROCm 30-minute development-soak contract inherits the same width, prefill,
and 50 ms actor-idle candidate, so a later soak cannot silently test a different
operating point. Its protected graph-cache entry count correspondingly moves
from 12 to 8, preserving exactly one protected geometry per declared active
request rather than retaining four unreachable entries. Vulkan remains a separate phase and is pinned explicitly to
decode width two, four active requests, four prefill layers per cycle, and
`batching.actor_cycle_idle_ms = 0`; adding the typed zero does not change its
runtime behavior. Vulkan development and endurance manifests record that zero
so future shared-default changes cannot contaminate their source-bound identity.

This replaces the earlier 88/86 C process-group stop policy. Two exact Strix
Halo ROCm serving rows showed that stopping the host process can leave queued
GPU work reporting 100-percent busy while package temperature remains above
the resume gate. The independent 180/300-second watchdog successfully
contained those runs, but a timeout does not make active-work suspension a
valid pacing mechanism. Source-bound ROCm and Vulkan serving qualification
therefore prohibits process-stop pacing. A future cooperative mechanism must
first prove that the backend has reached an idle ownership boundary.

The first standalone c8 replay under the replacement policy started its
measurement immediately after an 88.25 C warmup and reached the 93 C ceiling
before all streams finished. The guard observed 94 C at its next 250 ms sample,
terminated the server, completed graceful shutdown and stable cooldown, and
recorded zero pacing events. This validates fail-closed containment but rejects
the warmup-to-measurement handoff. Serving benchmark driver v11 now requires
two content-hashed idle-boundary cooldown records for every hard-limit-only
row: one before requests and one after all request workers drain. Each wait
requires the policy's consecutive stable samples, is bounded by its phase
settlement timeout, leaves the hard limit active, and cannot arm `SIGSTOP`.
Both waits remain inside thermally sustainable phase time. Raising the hard
limit is not an accepted correction.

The first exact driver-v11 replay from clean pushed source did not reach a row
boundary. Although the initial fingerprint and owned server both started after
eight samples below 45 C, ROCm model startup reached 93.25 C while uploading
accelerator weights and the hard-limit-only guard terminated it before
readiness. The server exited zero without force, removed its private snapshot,
cooled stably, and left no process or listener residue. This is a distinct
startup-load counterexample: idle-boundary cooling fixes hot handoff between
request waves, but cannot bound one uninterrupted startup stage. Do not retry
unchanged or raise the ceiling. Pace or partition accelerator-weight startup at
cooperative product boundaries, with typed configuration and explicit progress
evidence, before repeating the c8 control.

The source-bound Strix Halo ROCm and Vulkan serving profiles set both
`model.checkpoint_read_mib_per_second = 256` and
`model.accelerator_weight_upload_mib_per_second = 256`; these are named-host
qualification policies, not product-wide defaults. Generated server TOML and
every checked workload `effective_config.model` must agree exactly. Checkpoint
read pacing applies a fresh cumulative schedule to the snapshot copy, initial
full content verification, and post-upload full verification; reflinked bytes
are logical progress but are not charged as reads. Each phase checks shutdown
between bounded chunks and publishes logical/read bytes plus elapsed/paced time.

Accelerator upload reserves the cumulative base-model source-byte budget before
the base group and every layer, then publishes completed progress after the
unit. The base group checks shutdown after embedding upload, transpose, W8
packing, final norm, and rotary initialization. Backend operations remain
individually non-interruptible and conversions can perform more work than the
source byte count implies. Both policies poll shutdown every 25 ms while
waiting and end before readiness. Once ready, `GET /v1/config.model_startup`
must show complete checkpoint-read observations, matching reserved/completed
upload bytes and layers, and `active_during_inference=false` for both. The
exclusive server log remains content-hashed evidence for intermediate progress.
A replay is not accepted merely because startup survives: the unchanged host
thermal, memory, swap, cleanup, output, and throughput gates still apply.

The first paced replay proved that startup contract: all 8,411,510,272 source
bytes and 32 layers completed before readiness. Its following greedy c8 row
still reached the unchanged 93 C hard limit after 5.069 measured seconds, so
the blocker is now sustained concurrent inference rather than accelerator
upload. Do not run the mixed profile or a longer matrix unchanged. The next
tracked launch is
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-decode-batch-4-v1.json`.
Its parsed TOML differs from production v2 only in
`server.max_decode_batch = 4`; concurrency remains eight and the exact run ID,
prompts, output length, thermal policy, graph/cache policy, prefill policy, and
memory limit remain fixed. This arm may identify whether eight-row decode
packing is causal. It is not an accepted default, does not weaken the thermal
ceiling, and must stop after any guard, lifecycle, route, graph, output, or
receipt failure.

The first attempt from clean pushed source `f3189b703` never reached that arm.
Its initial provenance worker started after eight stable samples at 41.875 C,
performed the required double content read at the historical unlimited cached
I/O rate, and reached 93.125 C. The independent guard terminated it, completed
7.509 seconds of post-exit cooldown to 43 C, and the driver rejected the run
before server creation. No Kiln process, listener, receipt, or workspace residue
remained. Because guarded model fingerprinting is classified as pre-process
validation, no formal benchmark receipt exists; the exact console evidence is
recorded in the hardening ledger. Do not retry that unbounded path.

Serving driver v12 adds the typed
`--model-fingerprint-read-mib-per-second` bound, accepts `64..=16384`, and
defaults to 256. Campaign v5 and later record and forward the same value. One monotonic
cumulative schedule covers both integrity passes and every input, including the
second full content read that detects same-length concurrent mutation; an 8 MiB
read can be followed by sleeps of at most 25 ms. The v2 fingerprint thermal
record binds `read_mib_per_second`, the exact worker implementation and Python,
and both guarded lifecycles. This pacing is outside request timing and does not
alter server upload or inference. The exact source-bound replay from
`75ae28f54` proved that isolation: the initial and final double-pass workers
peaked at 39.25 C and 40.875 C, respectively, with zero trip and complete stable
cooldown. The owned server then failed independently before readiness. Starting
at 39.375 C, it copied the 9,319,828,096-byte private snapshot in 2.662 seconds,
loaded 8,022 MB of CPU weights in another 4.708 seconds, and reached 93.5 C
2.156 seconds after entering the first accelerator upload group. The guard
terminated it cleanly, removed the snapshot, cooled to 42 C, and left no process
or listener. No warmup, request, or width-four decode occurred. This rejects the
startup lifecycle, not provenance pacing or decode width. Do not repeat it
unchanged: first introduce source-bound cooperative boundaries and phase
evidence across snapshot materialization, CPU checkpoint loading, and the first
base-weight upload quantum without weakening model immutability or the 93 C
ceiling.

The corrected replay from clean pushed source `2058849e2` closed that startup
blocker and finally exercised width four. Snapshot copy, initial verification,
accelerator upload, and post-upload verification completed in 35.311, 34.719,
31.718, and 34.719 seconds. Their paced waits were 32.107, 27.338, 16.919, and
28.777 seconds, all 9,319,828,096 checkpoint bytes were accounted in every read
phase, and all 8,411,510,272 upload source bytes plus 32 layers completed. The
server reached readiness with a 59.125 C startup peak; the c1 warmup passed and
peaked at 60.375 C. This is the required hardware proof that bounded checkpoint
materialization and leading upload reservations solve the pre-readiness trip.

Width four is not sufficient for c8. The measured row began from 41.75 C, the
actor admitted all eight requests with effective decode width four, and the
guard sampled 93.25 C during the 8.288-second request window. Four requests
finished all 32 tokens; the other four correctly rejected streams without a
terminal usage record after `SIGTERM`. The 15.44 partial output tokens/second
and 12.74 thermally sustainable partial tokens/second are diagnostic only.
Device memory peaked at 35,346,485,248 bytes below the 50 GB gate. Shutdown was
unforced, returned zero in 214 ms, removed the snapshot, cooled stably, and left
no process or listener; both bounded fingerprint lifecycles remained below
42.625 C. The strict failed receipt is
`benchmarks/receipts/rocm/strix-halo/20260719t175109-rocm-strix-halo-greedy-c8-decode-batch-4-checkpoint-paced-v12.kiln.json`.
This rejects width four as the thermal correction. Do not retry it unchanged or
run the mixed/longer matrix; the next source-bound arm must further reduce or
cooperatively bound concurrent decode work without weakening the 93 C ceiling.

The completed one-field discriminator was
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-decode-batch-4-actor-cycle-idle-100ms-v1.json`.
It retains the rejected width-four profile and changes only
`batching.actor_cycle_idle_ms = 100`. This fixed safe-boundary delay starts only
after an actor cycle advanced model work and synchronous accelerator execution
returned. It is not sensor feedback and does not weaken or replace the unchanged
93 C hard-limit guard. All eight responses, lifecycle, memory, graph, route, and
driver-v13 duty-cycle accounting gates passed. The measured row peaked at
74.875 C, and the complete owned server lifecycle peaked at 89.375 C without a
trip. The actor accounted 159 waits totaling 15.910 seconds, however, so output
fell to 10.299 tokens/second, p99 TTFT reached 18.514 seconds, p99 ITL reached
276.688 ms, and no request met the declared latency SLO. The strict receipt is
`benchmarks/receipts/rocm/strix-halo/20260719t185455-rocm-strix-halo-greedy-c8-decode-batch-4-actor-cycle-idle-100ms-v13.kiln.json`.
This proves the mechanism and thermal effect but rejects the 100 ms/four-layer
combination for production latency; do not expand it to the mixed matrix.

The completed prefill-amortization discriminator was
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-decode-batch-4-actor-cycle-idle-100ms-prefill-layers-32-v1.json`.
It retains the thermally contained profile and changes only
`server.max_prefill_layers_per_cycle` from four to 32. For this eight-request
Qwen3.5-4B row, that can reduce the measured 128 prefill forwards to 16 while
leaving the 100 ms decode-cycle boundary intact. The purpose is to amortize
prefill waits and recover TTFT in one larger step, not to weaken decode cooling.
The exact run reduced prefill forwards to 17 and actor waits from 159 to 66,
recovering p99 TTFT from 18.514 to 2.799 seconds and aggregate output from
10.299 to 15.174 tokens/second. It stayed thermally contained at 78.375 C for
the measured row and 90 C for the complete lifecycle, and all eight requests
completed exactly 32 tokens. The actor wait itself remained bounded to
100.244 ms.

The strict receipt is
`benchmarks/receipts/rocm/strix-halo/20260719t190930-rocm-strix-halo-greedy-c8-decode-batch-4-actor-cycle-idle-100ms-prefill-layers-32-v13.kiln.json`.
Its verdict is failed. Two single-row graphs were already resident from warmup,
but the measured width-four path recorded zero graph capture attempts, replays,
fallbacks, failures, or an unavailable reason. The required
`rocm_graph_execution_accounted` gate therefore failed. Separately, keeping all
eight prompts decode-ready exposed the cost of rotating two eager width-four
cohorts: each forward took roughly 122--141 ms before the explicit 100 ms idle,
p50/p99 ITL rose to 461.309/1,058.994 ms, and SLO goodput remained zero. This is
not actor-wait oversleep; the process maximum proves that wait stayed near its
configured bound. Do not retry this profile or run the longer matrix. Before
another performance arm, the multi-row eager route must report why the requested
single-row ROCm graph runner was inapplicable, and the evidence gate must
distinguish explicit inapplicability from silent zero activity.

Driver v14 makes that distinction explicit. A successful eager contiguous
decode with more than one row now increments
`multi_row_batch_unsupported` once per completed forward and records its
duration. Diagnostics v5 requires that counter to agree with requested graph
capture and an observed multi-row batching route. The receipt remains
structurally valid, but `rocm_graph_execution_accounted` stays failed because
the measured row did not execute a graph. This is route evidence, not a waiver
for the graph-performance gate.

The exact source-bound confirmation is
`benchmarks/receipts/rocm/strix-halo/20260719t200129-rocm-strix-halo-multi-row-graph-accounting-v14-c8.kiln.json`.
All 62 measured decode forwards were width four, and diagnostics v5 reported
exactly 62 `multi_row_batch_unsupported` fallbacks, zero captures, zero replays,
zero graph failures, and zero other fallback reasons. All eight requests still
completed exactly 32 tokens without a server error, the measured row peaked at
76.75 C, and the owned lifecycle peaked at 89.375 C with no guard trip and a
clean unforced exit. This closes the silent-accounting defect.

It does not rehabilitate the performance arm. Aggregate output was 14.780
tokens/second, thermally sustainable output was 9.684 tokens/second, p99 TTFT
was 2.800 seconds, p50/p99 ITL was 479.098/1,049.803 ms, and SLO goodput was
zero. The strict receipt therefore failed the graph-performance gate as
designed. Do not retry or expand this configuration. A subsequent batched
performance arm must either disable ROCm graphs explicitly and qualify itself
as eager, or first add and qualify real multi-row graph capture; resident
single-row graphs must not imply graph acceleration for a multi-row route.

Current source supersedes that historical single-row limitation for contiguous
BF16 batches. The runner now keys HIP graphs by width and attention bucket,
refreshes paged metadata and GDN state through stable device storage, and keeps
LM head and sampling eager. A bounded gfx1151 width-four regression executed one
native capture plus 23 changing-input replays; every hidden value and the full
K/V pools matched an independent eager cache exactly, with zero fallback or
graph failure. That focused fixture contains full attention only. Production
Qwen GDN parity, a source-bound c8 serving receipt, and the longer thermal gates
remain mandatory before this becomes accepted performance evidence. A second
bounded hybrid fixture now covers one production ROCm GDN layer followed by one
full-attention layer: one capture plus seven replays matched eager hidden,
recurrent, convolution, and full K/V state exactly, with 286,720 retained GDN
slot bytes fully accounted.

The first source-bound production-model c8 arm is retained at
`benchmarks/receipts/rocm/strix-halo/20260719t230545-rocm-strix-halo-native-batch-graphs-v16-exact-prompt-c8.kiln.json`.
Its measured row passed every route and lifecycle gate: all eight requests
completed 32 tokens, effective decode width was exactly four, one multi-row
graph capture led to 61 replays, and no fallback or graph failure occurred.
Decode service wall fell from the exact-prompt eager row's 8.020 seconds to
6.598 seconds and aggregate output reached 16.616 tokens/second. The row peaked
at 80.375 C, the complete owned lifecycle peaked at 89.375 C without tripping
the 93 C guard, and device memory stayed below 50 GB at 35,433,390,080 bytes.
These are candidate measurements only. The strict receipt failed exact-output
comparison at request indices 1, 2, 4, and 5, so none of the performance result
is accepted for promotion.

A same-binary, same-prompt diagnostic with only typed
`accelerator.rocm_graph_mode` changed to `disabled` also differed from the
historical eager receipt, at indices 4 and 7. That shows the prior rotating-c8
receipt is not a sufficient bitwise cross-process oracle by itself. The graph
failure nevertheless exposed a concrete actor-lifecycle defect: extracting a
one-row GDN state with `narrow(...).contiguous()` could return a view because
the narrow was already contiguous. Per-request recurrent and convolution state
therefore aliased the reusable batch slot, and loading the second width-four
cohort could overwrite the first cohort's saved state. Row extraction now uses
an explicit device copy in both shared state helpers and the legacy eager
batched GDN route. Portable tests assert independent storage, and an
actor-faithful gfx1151 regression alternates two disjoint width-four cohorts for
16 turns through one persistent graph slot. It records one capture and 15
replays while matching eager hidden, recurrent, convolution, and complete K/V
state exactly at every turn, with zero fallback or failure. A rebuilt
fixed-source c8 graph/eager pair remains required before accepting output
parity or throughput.

That pair is source-controlled rather than assembled with environment
overrides. The graph arm uses
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-decode-batch-4-actor-cycle-idle-100ms-prefill-layers-32-v1.json`;
the eager oracle uses the adjacent `-graph-disabled-v1.json` launch record.
Their parsed TOML documents differ only at typed
`accelerator.rocm_graph_mode`, from `profile` to `disabled`. Run the eager arm
first, then bind the graph arm's exact-output comparison to that new receipt.

The fixed-source eager oracle is
`benchmarks/receipts/rocm/strix-halo/20260719t235800-rocm-strix-halo-fixed-source-state-copy-eager-v16-c8.kiln.json`.
It passed strict validation on clean pushed source `94ab3e0d6`: 8/8 requests
completed exactly 32 tokens, all 62 decode forwards had width four, and graph
capture, replay, fallback, and failure counts were all zero under the typed
disabled mode. Decode service wall was 7.556 seconds, aggregate output was
15.674 tokens/second, and thermally sustainable output was 9.964
tokens/second. P50/p99 ITL was 443.893/938.433 ms and p99 TTFT was 2.778
seconds. The measured row peaked at 77.125 C, the owned lifecycle peaked at
89.875 C without a guard trip, memory peaked at 35,233,689,600 bytes, shutdown
was unforced, and no process or listener residue remained. This receipt is the
bitwise output oracle for the fixed-source graph arm; it is not a throughput or
latency promotion by itself.

The paired graph arm is retained as the strict-valid failed receipt
`benchmarks/receipts/rocm/strix-halo/20260720t000300-rocm-strix-halo-fixed-source-state-copy-graphs-v16-c8.kiln.json`.
All route and lifecycle gates passed: 8/8 requests completed 32 tokens at width
four, one capture led to 61 replays with no fallback or graph failure, decode
service wall was 6.658 seconds, aggregate output was 16.559 tokens/second, and
thermally sustainable output was 10.007 tokens/second. P50/p99 ITL was
412.821/1,020.318 ms, p99 TTFT was 2.855 seconds, measured/lifecycle temperature
peaked at 77.625/90.5 C without a trip, memory peaked at 35,434,250,240 bytes,
and shutdown was unforced with no residue. Exact output still differed at seven
request indices, so none of those candidate performance values is promoted.
Every graph-arm output hash recurs in earlier retained receipts, including
eager profiles, and its vector closely follows earlier process trajectories.
That recurrence proves the cross-process hash mismatch is not by itself a novel
graph-corruption signature; it does not waive the failed comparison.

Current source therefore gates graph admission with an in-process oracle. It
retains the candidate's eager warm hidden output, every recurrent/conv state,
and only the current K/V rows, restores the input state, launches the graph once,
and compares all values exactly before admission. Temporary copies, one live
K/V gather, and one U8 comparison mask are reserved before allocation and are
included with the exact candidate bytes in transient high-water telemetry. A
comparison error or mismatch settles and rolls state back, discards the graph,
disables graph execution for that runner, and enters the existing
`capture_failure` eager fallback only after containment succeeds. The structured
`rocm_graph_capture_parity_check` event attributes bytes, duration, hidden
equality, and the first recurrent, convolution, K, or V mismatch layer.

The corresponding explicit gfx1151 oracle now uses the real 32-layer Qwen
topology (24 GDN plus eight full-attention layers), distinct synthetic
allocations for every layer, and production W8 packed projection routes. Two
disjoint width-four cohorts are seeded eagerly and alternate through one slot.
One capture plus 15 replays matched eager hidden, all recurrent/conv state, and
the complete K/V pools exactly, with zero fallback or failure. The run retained
11,854,304 graph/slot bytes, reserved a measured 18,885,089-byte peak transient
working set, and completed native capture plus first-launch parity in 74.899 ms.
A new committed-source paired production run is still required; this guard and
focused oracle do not retroactively promote the failed receipt.

That committed-source pair now exists. Eager reference
`20260720t012200-rocm-strix-halo-guarded-parity-eager-v16-c8.kiln.json`
passed from source `e4e45e276bf74c2181bf279ac3ca724be7d23673` and binary
`sha256:98fd03c6ac9535cdf464ad6b604e17f816868fbeb1cc69897d7a721b7eba916a`.
Its same-binary graph arm
`20260720t013300-rocm-strix-halo-guarded-parity-graphs-v16-c8.kiln.json`
passed every route, request, graph, thermal, memory, shutdown, and finalization
gate. The exclusive log contains exactly one multi-row
`rocm_graph_capture_parity_check`: width four, `outcome=passed`,
`comparison_complete=true`, 220,504,064 exact bytes, 11.558 ms, and equal
hidden output with no state/K/V mismatch field. The measured row then recorded
one capture and 61 replays with no fallback or graph failure, 16.502 aggregate
and 10.290 thermally sustainable output tokens/second, a 77.0 C measured peak,
and a 90.0 C lifecycle peak without a trip.

The driver-v16 receipt remains correctly failed because the separate-process
output trajectory differed from its eager reference at indices 4 and 6. Both
actual hashes recur in older retained trajectories, and one is present in an
earlier eager receipt. This does not overturn the strict comparison result. It
does establish that two different claims are currently coupled: the graph
candidate passed exact same-process admission, while whole-process deterministic
replay did not. Do not repeat cross-process exact-output pairs as if they
isolated graph correctness.

Serving benchmark driver v17 now enforces that separation in a typed receipt
rather than through log interpretation. Diagnostics schema v6 retains
measured-window and lifetime batched-capture/parity counters, reconciles every
closed outcome, and adds the required
`rocm_graph_capture_parity_accounted` gate. Its default
`qualification_gate` role retains all prior exact-output verdict behavior. The
narrow `same_artifact_graph_eager_discriminator` role requires a v17 Kiln eager
reference with observed graph-disabled execution, the identical runtime binary
and identity, and a graph candidate whose every measured row passes graph-route
and first-launch-parity gates. Cross-process mismatches remain structured
`evidence_only` reproducibility evidence only in that case. Any other role,
artifact, reference execution, missing comparison, or parity edit fails strict
validation. Campaign v7 forwards and records the role and reference directory
for all five profiles.

The required source-bound v17 pair now exists. Ordinary eager reference
`20260720t030727-rocm-strix-halo-parity-eager-v17-c8.kiln.json` and discriminator
candidate `20260720t031650-rocm-strix-halo-parity-graphs-v17-c8.kiln.json` use
the same source-tree hash, workload fingerprint, prompt-set hash, runtime
identity, and exact release binary
`sha256:ddf2b627965710156e1df28f85a88b5ec2ce47a2427acfe8025611b2ace35d2a`.
The eager warmup and measured row observed graph-disabled execution with every
graph and parity counter zero. The candidate measured row recorded one batched
capture, one passed first-launch parity check over 220,504,064 bytes in
11.569 ms, and 61 successful replays with zero fallback, capture/replay failure,
parity mismatch, or comparison error. Both receipts pass strict validation.

The comparison remains `matched=false`: candidate outputs differ from the eager
reference at request indices 0 and 3. Driver v17 retains those structured rows
with `verdict_effect=evidence_only`; they are not waived or presented as equal.
The candidate passes because same-process graph correctness is independently
proved. At this exact c8 point, graph execution reduced route decode wall from
7.590 to 6.704 seconds, raised aggregate output from 15.619 to 16.491
tokens/second, and raised thermally sustainable output from 9.940 to 10.177
tokens/second. This closes the graph-attribution blocker for this pair, not the
remaining repeated-profile, mixed-load, 30-minute, or endurance gates.

Serving benchmark driver v15 closes the next attribution gap before another
ROCm arm. Each successful Kiln stream now retains its exact terminal
`metadata.performance` object, including the request-local latency phases and
stall reasons, and the receipt derives per-phase request populations plus p50,
p99, and maximum values. Usage, finish reason, emitted-token counts, retained
gaps, stall totals, and the complete closed phase/reason sets are reconciled by
strict validation. The common Kiln/vLLM request body is unchanged: the tracked
Kiln profiles already enable terminal summaries through typed server config,
while vLLM rows use explicit nulls. Request phase durations are not summed
across concurrent clients because shared actor work overlaps; route-level
health deltas remain the service-wall evidence. In particular, the configured
cooperative actor wait now reaches each request that remains active through the
wait as `actor_cycle_idle_ms` and can classify its next token stall instead of
appearing as unexplained ITL.

The observability schema generator, serving benchmark, and source-bound
mixed-load driver consume one canonical fixed-cardinality request-latency field
registry. The mixed-load receipt includes
`latency_phase_actor_cycle_idle_ms_total` and
`latency_phase_actor_cycle_idle_request_count` with the other request-local
phase metrics. A missing or additional phase or stall-reason field fails before
measurement and reports the exact missing and additional names. This keeps a
server schema addition from being silently ignored by one qualification path.

The retained source-bound ROCm receipt
`qualification/receipts/rocm/strix-halo/20260720t044950539805z-rocm-strix-halo-serving-mixed-rocm-v1-184c082f9e-v1.json`
is the counterexample that closed this gap. The release server loaded,
prewarmed, returned the correct 32-token warmup prefix, and shut down normally;
the old mixed-load parser then rejected the already-published
`actor_cycle_idle` stall-reason field. Its zero measured tokens and throughput
are sentinels, not performance evidence. The corrected shared registry must be
present in the exact pushed source used by the next run.

The source-bound confirmation is
`benchmarks/receipts/rocm/strix-halo/20260719t204756-rocm-strix-halo-request-phase-attribution-v15-c8.kiln.json`.
All eight requests contributed complete performance and latency objects and
finished with exactly 32 tokens. Median request-local phase time was 7.955
seconds for decode, 6.404 seconds for configured actor-cycle idle, 2.128
seconds for prefill, and 6.131 ms unexplained. The route-level deltas independently
reported 8.020 seconds across 62 eager width-four decode forwards, 6.604 seconds
across 66 actor waits, and 2.283 seconds across 17 prefill forwards. There were
no request stalls, unexplained stalls, server errors, thermal trips, or cleanup
failures. Aggregate output was 15.215 tokens/second, thermally sustainable
output was 9.775 tokens/second, p99 TTFT was 2.801 seconds, and p50/p99 ITL was
458.133/1,064.860 ms. The receipt remains intentionally failed because all 62
multi-row forwards used the explicitly accounted eager fallback instead of a
ROCm graph. This ranks eager decode first and the fixed cooling wait second;
prefill is third and residual unaccounted wall time is immaterial. The next
optimization must address the measured multi-row decode route rather than
another blind scheduler-delay trial.

The first implementation checkpoint removes one concrete source of ROCm
multi-row submission and synchronization overhead before attempting graph
capture. The production BF16 KV writer now resolves the batch's physical slots
once per decode step, uploads one `[batch]` U32 tensor, and reuses it across all
full-attention layers. Each layer writes the complete K and V batch with two
on-device indexed scatters. The former eager path issued two device copies per
row per layer; the nominal device-slot helper additionally copied the slot
tensor back to the host before issuing those copies. The new primitive consumes
the slot tensor on-device, does not synchronize it through the host, and is
also wired into the stable-buffer path required by a future multi-row HIP
graph. An explicit real-gfx1151 test writes both production host-resolved slots
and caller-owned device slots into noncontiguous KV pool rows and verifies every
BF16 row exactly. This is correctness and launch-count evidence, not a
throughput claim. Host-resolved batches also reject duplicate physical decode
slots before device work, preserving exclusive mutable-KV ownership instead of
introducing a racy indexed write. The exact pushed-source c8 attribution
workload must determine whether the optimization materially changes decode time
or thermal load.

The first source-bound c8 run of that checkpoint is retained at
`benchmarks/receipts/rocm/strix-halo/20260719t212213-rocm-strix-halo-kv-device-slot-scatter-v15-c8.kiln.json`,
but it is not an exact A/B result. All eight requests completed exactly 32
tokens, request-local evidence was complete, the measured row peaked at 79 C,
and the owned lifecycle peaked at 89.625 C without a trip. The route reported
8.421 seconds of decode, 6.604 seconds of fixed actor idle, and 2.269 seconds of
prefill; aggregate and thermally sustainable output were 14.873 and 9.455
tokens/second. Those values do not show an optimization win. More importantly,
driver v15 derives prompt text from the operational `run_id`. Giving the
candidate a distinct receipt/log identity therefore changed every prompt from
150 to 152 tokens and changed both prompt/output set hashes. The result is valid
as a candidate correctness and thermal receipt but incomparable to the prior
8.020-second decode row. Do not retry with a reused `run_id`, because that would
overwrite the earlier external server log. The benchmark protocol must first
separate stable prompt-set identity from unique run/artifact identity and bind
both into strict validation.

Serving benchmark driver v16 implements that repair without changing the
prompt template. Every measured invocation must now provide both a unique
`--run-id` and an explicit stable `--prompt-set-id`. Only the prompt-set ID is
used for marker ordering and the model-visible `Benchmark run:` value, so using
`rocm-strix-halo-request-phase-attribution-v15-c8-greedy-short` as the new
prompt-set ID reconstructs the v15 baseline prompts byte for byte while a new
run ID selects a collision-free receipt and server log. The v16 validator
reconstructs the expected prompt-set hash for warmup and every measured row;
editing the identity and recomputing both receipt hashes still fails as stale
evidence.

For v16, `workload_fingerprint` covers the entire comparable contract except
the operational run ID. It still covers prompt-set identity, profile, prompt
shape, sampling, seeds, concurrency, repeats, output length, template mode,
arrival policy, SLOs, and memory ceiling. The canonical receipt hash covers
both identities. This permits separate A/B and Kiln/vLLM lifecycle artifacts
without weakening workload equality. Historical v2-v15 receipts continue to
validate with their original exact-workload fingerprint rule and schemas. The
KV-scatter candidate must be rerun from clean pushed source with the baseline
prompt-set ID before its performance disposition can change.

That exact-prompt rerun is retained at
`benchmarks/receipts/rocm/strix-halo/20260719t214802-rocm-strix-halo-kv-device-slot-scatter-v16-exact-prompt-c8.kiln.json`.
Driver v16 source `efb78d5a5` controlled the evidence lifecycle while the
explicitly bound runtime remained the unchanged dfea candidate binary
`sha256:d5fc30c68b3928bb6531937ddf240c183f08456b3b67264515d999cff2900c3f`.
Both the one-row warmup and eight-row measurement exactly reproduce the v15
baseline prompt-set hashes and ordered token counts: 147 warmup tokens and
eight 150-token measured prompts. The warmup output also matches exactly.

The candidate is not a throughput win. Route decode changed from 8,020.113 to
8,020.094 ms across the same 62 width-four eager forwards, a 0.00024 percent
reduction. Request-window aggregate throughput changed from 15.2146 to 15.2255
tokens/second, only 0.0717 percent, while thermally sustainable throughput fell
from 9.7750 to 9.5032 tokens/second because the post-row cooldown was longer.
Prefill improved 0.55 percent, p99 TTFT improved 0.54 percent, p50 ITL worsened
0.36 percent, and p99 ITL improved 1.78 percent; none establishes a material
KV-scatter effect. Peak device memory differed by only 4 KiB. The measured row
peaked at 78.25 C and the complete guarded lifecycle at 90.5 C without a trip.

Six of eight measured outputs match the baseline exactly; request indices 3
and 4 have different content hashes despite identical prompts, seeds, lengths,
and a common width-four route. The baseline admitted all eight requests
concurrently while the candidate's process active peak was six, so this
throughput configuration did not preserve cohort formation and cannot claim
cross-run exact output parity. The existing real-device KV regression remains
the positive bit-exact evidence for the writer itself. Keep the device-slot
scatter because it removes the host round trip and per-row copy structure
needed for future graph capture, but reject it as a performance optimization
and make no output-parity claim. Multi-row graph work still requires an exact
eager-versus-replay parity gate before any serving promotion.

After the server exits, the controller keeps sampling until eight consecutive
250 ms observations are at or below 75,000 millicelsius. The first cool reading
does not suffice: the consecutive-sample condition protects against the package
temperature rebound observed after an earlier run returned control at 88.75 C.
The cooldown is bounded to 180 seconds. A sensor failure, hard-limit reading, or
timeout remains a failed qualification; the dead server cannot resume work, and
the controller continues the bounded cooldown attempt rather than silently
calling the host ready.

The receipt retains start/peak/end package temperature, guard error and trip
counts, zero-valued compatibility pacing counters, cooldown
active/completed/timeout counts, duration, sample count, consecutive stable
count, and post-exit peak. Adaptive ITL evidence retains the total attributed
count and still partitions it into `host_thermal_pacing_itl_outlier_count` and
`non_thermal_attributed_itl_outlier_count`; the thermal-pacing partition must
now be zero because the policy cannot create an active-work pause. A pass
requires zero guard error, trip, cooldown timeout, non-thermal attributed
outlier, unexplained outlier, or active final controller, exactly one completed
cooldown, and zero started pacing intervals. These checks apply to the short
mixed-load gate, the resident-prefill oracle, and the longer
development/endurance soaks; the source-build watchdog is a separate guard
around the compiler/linker service.

### Vulkan serving baseline

Run this only after the required ROCm receipts have passed on the same clean,
pushed source tree. It is the first full-server Vulkan gate; the earlier core
and prefill workloads exercise lower-level routes but do not prove the public
SSE server, source-built executable identity, batching actor, or teardown.

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan-serving-baseline \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-vulkan-baseline-v1.json
```

The driver builds `kiln-server` once with `--no-default-features --features
vulkan`, through the bounded wrapper, offline, with one compile job and the
unchanged 15 GiB host-memory admission floor. The compile window is 900 seconds
and is separate from the 240-second server-readiness window. The generated
private TOML selects the experimental profile only to keep the batching actor
available; it explicitly disables KV autoscaling, allocator reclaim, and ROCm
graphs. Runtime attestation requires all three policies and their
`config_file` provenance before measurement and again after the final wave.
Readiness requires both the passing health check and causal log evidence. The
Vulkan-native path satisfies the latter with `Vulkan decode weight prewarm
complete`; its typed 256 MiB/s rate shapes startup pressure, and shutdown must
join the task before reporting a clean stop. It does not run or claim the
synthetic inference prewarm used by other serving paths.

After one fixed warmup, four thread-barrier waves dispatch concurrency 1, 4, 8,
and 12 with mixed prompt lengths from 16 through 1,024 deterministic words.
All 25 measured requests disable thinking and sampling, ignore EOS, stream
usage and performance metadata, and must finish by the exact 32-token limit.
The 600-second per-request limit is a correctness-containment deadline for the
longest synchronized prefill, not a latency SLO or a passing performance
threshold. Actual TTFT, end-to-end duration, output throughput, active width,
and decode batch width remain unchanged receipt evidence; the cross-engine
serving matrix owns the competitive performance verdict. Raising containment
therefore cannot turn a slow run into a fast one or suppress a timeout.
The final two waves cover the configured eight decode slots and the four
derived short-prefill staging slots. Batching counters must prove at least one
multi-row decode, more decode rows than forwards, and a prefill forward for
every request. The receipt publishes per-wave p99 TTFT, ITL, and end-to-end
latency, duration and token throughput, plus aggregate batching width, row,
prefill, synchronization, memory, and capacity evidence. These values are a
Vulkan scaling baseline, not a claim of parity with vLLM.

`/health.execution_identity` must agree exactly with the complete trusted-debug
execution provenance. The backend must be `vulkan`, the device must be
`vulkan:0`, the compiled kernel feature set must include `vulkan` and exclude
`rocm`, the source must be clean, and the executable digest must equal the
binary built by the driver. Compact result details bind that executable, the
generated TOML, the effective server/environment configuration identities, the
kernel contract, the execution-provenance envelope, and one ordered canonical
hash of all 25 streamed semantic outputs. Dynamic response IDs and timestamps
are excluded from the semantic hash; request names, usage counts, and semantic
deltas are included. The public response headers use the paired value `base`
for both loaded-adapter fields when no adapter is resident. Qualification
normalizes only that exact pair to no identity; a missing header, a mixed
base/named pair, a malformed revision, or any named identity in this baseline
fails closed.

A failed result preserves every boundary it actually crossed instead of
replacing the run with a generic zero record. Its compact `details.milestones`
array advances through `build`, `config`, `startup`, `warmup`, each completed
`wave:<name>`, `measurement`, and `teardown`. The build and generated-config
digests survive startup failure; a returned startup identity survives a later
warmup or request failure; and every completed wave immediately contributes its
real request, token, latency, termination, pause, and ordered semantic-output
hash evidence. Typed device, graph, resize, and reclaim events are collected
from server start through shutdown, while memory samples span measured load.
The final pass/fail gate therefore sees post-measurement teardown faults as
well as events observed during load.

Compact details remain valid JSON even when a failure list exceeds the receipt
limit. Long strings are bounded inside the JSON value, while `_truncation`
records the SHA-256 and original character count plus the number of omitted
top-level fields. Runner-level failures such as a nonzero process exit are
added under `runner_failures`; the runner never appends text after a JSON
object. The ignored full stdout/stderr artifacts remain hash-bound to the
receipt for causal inspection.

Metrics for a boundary absent from `milestones` mean "not observed," not a
successful zero-count measurement. For example, zero requests with no
`wave:*` milestone says that the workload never issued a complete measured
wave; zero shutdown failures with no `teardown` milestone does not attest clean
shutdown. Keep and validate these failed receipts as causal counterexamples,
but do not compare their partial performance metrics with passing baselines.

An HTTP-200 SSE response can still terminate with the closed structured error
envelope `{message, type: "server_error", code: "generation_error"}` followed
by `[DONE]`. Every serving driver treats that envelope as the primary request
failure, validates its exact shape, and retains the bounded server message in
the case result. It must never be reclassified as a successful empty stream or
reduced to a secondary missing-finish-reason error.

The gate fails on any request error, non-length or short output, missing actor
timing, unexpected named adapter identity, device fault, resize/reclaim/graph
event, batching error, failed or at-least-100-ms external-yield synchronization,
changed KV capacity, missing memory sample, unexplained adaptive ITL outlier, or ITL gap
above two seconds. Gaps above 250 ms are always counted as stall evidence even
when they remain below the hard pause gate. The server must drain, exit zero
without force inside the shared 60-second grace period, and leave no private
snapshot payload.

#### Current Strix Halo result

The source-bound 2026-07-14 run on RADV Strix Halo passed from clean commit
`e2efd5dff` and tree hash
`sha256:c5e7d485a9afb8319435e87c4eb91808652a2cc3d0a88fbac602b794a10cad66`.
Its retained receipt is
`qualification/receipts/vulkan/strix-halo/20260714t232612510178z-vulkan-strix-halo-serving-vulkan-baseline--8da450e169-v1.json`.
The run completed 25/25 measured requests and 800/800 completion tokens with
exact length termination, zero request/batching/device/policy/sampler errors,
zero graph/resize/reclaim activity, stable 5,365-block KV capacity, zero
unexplained or two-second ITL pauses, clean zero-exit shutdown, and no snapshot
residue. Four adaptive ITL outliers were causally attributed to concurrent
prefill; 395 client-visible gaps exceeded the 250 ms evidence threshold but
none crossed the two-second failure gate.

This accepted correctness result is also negative performance evidence. The
single, four-way, eight-way, and twelve-way waves took 6.473, 93.384, 324.697,
and 464.650 seconds. Overall output throughput was 0.900 tokens/second; the
eight- and twelve-way waves achieved 0.788 and 0.826 tokens/second. Although
eight active requests were observed, only two decode rows were ever combined,
with 2 batched forwards out of 773 decode forwards. Peak sampled unified-device
usage was 54,288,658,432 bytes. This receipt accepts bounded mixed/long-prompt
correctness and the absence of unexplained pauses or runtime memory-policy
mutation. It does not establish competitive Vulkan batching, and it must not
be presented as a vLLM parity result.

A passing receipt closes only the Phase 6 short/mixed/long serving-baseline
item and the no-silent-skip/no-unexplained-outlier condition for this bounded
run. It does not close the CPU/HF oracle comparison, 30-minute development
soak, eight-hour final soak, cross-engine benchmark, or final common-source-tree
release gate. Keep those items open until their separate receipts exist.

A positive `memory.kv_force_blocks` value is intentionally narrower than the
normal control loop. It is accepted only with `memory.kv_autoscale = true` and
`server.serving_profile = "maintenance"`, where inference admission is already
disabled. The one-shot resize still reserves the complete replacement pool,
drains the actor, invalidates graphs, publishes capacity transactionally, and
emits a `gpu_memory_operation` record with reason `forced_configuration`.
`/health`, `/v1/config`, and the trusted debug state expose the requested value,
effective autoscaler state, bounded reason, and `config_file` source.

For the supported Strix Halo ROCm serving contract, run the `stable` arm. It
deliberately requests autoscaling, automatic allocator reclaim, and ROCm graphs,
then requires the stable profile to suppress all three while mixed SSE load,
long prefill, cancellation, and socket backpressure are active:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant stable \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-mixed-rocm-v1.json
```

The receipt also records every backend external-yield synchronization boundary,
call/failure/slow count, total time, and maximum duration. A failed sync, a sync
lasting at least 100 ms, any physical resize/reclaim/graph event, or any
unexplained ITL outlier fails the stable arm.

Graph-on mixed-load arms also conserve request phase attribution against the
same measured-window lifetime graph telemetry. A shared batched capture or
replay envelope is expected to appear on every ready request row that
participated, so the request aggregate may exceed device work; it must not
exceed the matching lifetime phase total multiplied by maximum observed decode
width. Exceeding that bound fails the arm because it proves phase charging
escaped the participating cohort. Use the request total to diagnose causal
overlap and the lifetime total to quantify process/device work.

Experimental ROCm graph runs expose a closed fallback contract at
`/health.decode_runtime.rocm_graphs.fallbacks`. It reports the total and the
fourteen reason counts (`multi_row_batch_unsupported`,
`cold_cache_host_round_trip`,
`persistent_host_round_trip`, `shape_dependent_attention`, `graph_cache_capacity`,
`graph_cache_byte_budget`, `graph_accounting_incomplete`,
`moderate_memory_pressure`, `tight_memory_pressure`, `critical_memory_pressure`,
`memory_reservation_denied`, `memory_governor_selector_mismatch`,
`capture_failure`, and `replay_failure`) plus slow, total-duration, and
maximum-duration counters. The first occurrence of each
reason, every fallback lasting at least 100 ms, and every failed eager fallback
also emits `event=rocm_graph_fallback` with attempt, eager, and total duration.
Qualification validates the health invariants and attributes these events to
the exact ITL window; unknown reason strings do not receive graph attribution.
When a warm candidate observes a host upload, it also emits one
`event=rocm_graph_capture_host_transfer` record per unique source site. The
record carries the same closed fallback `reason`, build-source file, line and
column, dtype, elements and bytes per copy, copy count, and total bytes. The
dynamic observer is thread- and device-scoped, aggregates at most 32 unique
sites per candidate, and emits one `source_file=bounded_site_overflow` aggregate
if more sites occur. These records contain no request text, token ID, tensor
value, allocation address, or unbounded shape label. Use them to distinguish a
one-time shape-cache fill from a persistent per-forward host path; do not infer
capture or replay success from their absence without checking graph counters.
ROCm materialized broadcast is capture-safe by construction for Kiln's
documented maximum rank of eight: the host wrapper copies shape and stride
values into a 272-byte descriptor that the runtime records as a kernel-node
value. It creates no device metadata buffers and performs no metadata upload,
so replay has no cache-eviction or external-buffer lifetime dependency.
The mixed-load receipt also records call, slow-call, cumulative duration, and
maximum duration for pre-candidate headroom, candidate warm, pre-native
reservation, native capture, and rejected-candidate cleanup, plus peak exact
transient-candidate bytes. These values remain distinct from retained graph
bytes and make one-time or repeated graph setup pauses comparable across runs.
Native capture timing includes the settled first launch, defensive cache
admission/publication, and blocking committed governor debit; rejected cleanup
starts only when an unretained candidate enters destruction and settlement.
The driver treats the full cache snapshot and phase telemetry independent of
the model and graph-runner locks as separate authorities. Config and trusted
debug must report
`rocm_graphs`/`rocm_graphs_unavailable_reason` independently from
`rocm_graph_telemetry`/`rocm_graph_telemetry_unavailable_reason`; health
flattens them under `decode_runtime.rocm_graphs` with separate `state`,
`unavailable_reason`, `phase_telemetry_available`, and
`phase_telemetry_unavailable_reason` fields. Missing data must use exactly one
of `backend_without_graph_runner`, `model_runner_busy`,
`model_runner_lock_poisoned`, `graph_runner_busy`, or
`graph_runner_lock_poisoned`, never fabricated zeroes. Prometheus exposes the
same distinction with `kiln_rocm_graph_telemetry_available`,
`kiln_rocm_graph_snapshot_unavailable{reason}`,
`kiln_rocm_graph_phase_telemetry_available`, and
`kiln_rocm_graph_phase_telemetry_unavailable{reason}`. The phase handle lives
outside both runner locks, so it remains available for every real backend while
model-runner or graph-runner contention/poison blocks the full snapshot;
currently only a backend without a graph runner makes the phase channel null.
The same health object exposes retained graph and reusable-slot gauges plus
lifetime slot-create/reuse counters. A logical decode row borrows one slot;
request drain removes its continuity timeline and returns the slot to an idle
pool without destroying native graphs or their graph-stable recurrent buffers.
Adapter invalidation destroys native graphs before their buffers. Graceful
server shutdown closes and joins the decode worker before accelerator teardown.

Run the graph-memory and concurrency gate after the exact source has compiled
successfully on the Strix Halo host:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant headroom-vs-tight-budget \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-rocm-graph-resilience-v1.json
```

This is one source-bound server binary and two sequential server processes. The
headroom arm permits 64 entries and 1 GiB of retained graph allocations; the
tight arm keeps the same entry limit but uses the minimum supported 64 MiB byte
budget. Each arm warms a real graph outcome, then runs mixed prompt buckets at
client concurrency 1, 8, 16, 32, and 64, for 121 measured requests per arm.
Every request is fixed to eight output tokens and the driver compares canonical
streamed semantic deltas exactly across arms, excluding only dynamic response
envelope fields such as request ID and creation time.

The gate fails on an HTTP/request error, output mismatch, non-length finish,
missing token timing, device fault, graph capture/replay failure, retained bytes
above the configured ceiling, entries above capacity, active owner/slot residue,
dirty model snapshot, forced/nonzero shutdown, or any ITL gap above
`max(250 ms, 5 * rolling p50 ITL)`. Both attributed gaps, including graph setup
and synchronization, and unexplained gaps are failures; attribution is retained
to diagnose the source, not to waive a pause. The headroom arm must capture and
replay. The tight arm must make at least one typed byte-budget decision through
pre-capture skip, post-capture rejection, deterministic budget eviction, or the
closed `graph_cache_byte_budget` eager fallback. The receipt publishes p99 TTFT,
ITL, and E2E latency at every concurrency plus graph and peak-memory counters.
A passing receipt therefore has zero attributed or unexplained ITL outliers.

The 300-second client deadline and 360-second server deadline are containment
ceilings for this correctness matrix, not performance targets. The fixed
cross-engine serving campaign owns comparative throughput and latency verdicts;
increasing these ceilings cannot erase the measured TTFT or E2E values. A failed
matrix receipt retains attempted, completed, and failed request counts for every
headroom wave, the highest fully completed concurrency, per-arm
start/attempt/completion coverage, the latest graph counters, peak memory, and
observed pause counts. Unstarted arms therefore remain explicitly distinguishable
from started-but-incomplete arms and measured zero events.

Run the destructive-identity and fallback-containment corpus separately. It
uses the bounded Cargo wrapper, one compile job, the unchanged 15 GiB host-memory
floor, offline dependencies, one test thread, and a required real ROCm device:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant real-rocm-graph-fault-corpus \
  --host-id strix-halo \
  qualification/workloads/serving-rocm-graph-failure-containment-v1.json
```

All three ignored hardware tests must actually execute. The first proves that a
shape-dependent attention geometry is cached as a typed eager fallback rather
than captured. The second crosses sequence buckets and block tables, reuses a
released slot after cancellation/prefix activity, invalidates at an adapter
boundary, and requires exact eager parity. The third poisons only a live graph's
retained pool generation while leaving the physical allocation valid; the
guard must prevent native launch, count one replay failure, clear and disable
graphs, preserve the cache identity, and return exact same-cache eager output.
Missing devices and skipped tests are failures, not successful no-ops.

A native capture failure is eligible for the `capture_failure` eager fallback
only when `capture_rollback` first settles physical work with execution
admission open and logical recurrent-state rollback also succeeds. Failure of
either proof, or any armed/unclassified capture guard leaving scope, must
publish sticky STOP, use `error_recovery` only for a post-STOP diagnostic drain,
reject the eager continuation, and require process restart. Qualification must
never classify that quarantine as an expected fallback.

### ROCm Public Mutation Lifecycle

Run the public adapter and maintenance-mutation gate with a real adapter whose
base model matches the selected Qwen3.5 model:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant public-mutation-lifecycle \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  --var adapter_path=/absolute/path/to/Qwen3.5-4B/adapters/rocm-sft-test \
  qualification/workloads/serving-rocm-public-mutation-lifecycle-v1.json
```

The adapter input must be an absolute directory containing regular,
non-symlink `adapter_config.json` and `adapter_model.safetensors` files. The
driver copies only those two files into a private per-run adapter registry,
re-hashes the copy, and records both source SHA-256 values in the command result
details. It also records the one exact source-built `kiln` binary hash and the
two generated typed-config hashes. The qualification runner separately binds
the clean source tree, model weights, tokenizer, template, manifest, variable,
stdout, and command-result artifacts in the receipt.

The first arm starts that binary under the experimental profile with KV
autoscaling and reclaim off and lazy ROCm graph execution on. It requires real
capture and replay before measuring the lifecycle. A deterministic base request
must return `x-kiln-loaded-adapter: base` and revision `base`. Public
`POST /v1/adapters/load` must return one 64-hex content revision; health,
trusted debug state, and `GET /v1/adapters` must all publish that same
name/revision. The adapter inference request explicitly sends
`"adapter": "qualification-adapter"`; its response headers must bind that exact
name/revision. The driver then starts a 64-token named-adapter request and waits
for its first token before issuing `POST /v1/adapters/load` with
`"reload": true`. That operation must re-read the already-live exact revision,
and the cached health snapshot must report
`actor_barrier_adapter_active=true` and
`actor_barrier_resize_active=false` before a second named-adapter request is
submitted. Both responses must retain the same content-revision headers. The
request active before the barrier must retain `adapter_ms: null`; the request
queued after the positive gauge must report positive `adapter_ms` inside its
positive actor-queue envelope. This distinguishes causal request ownership
from a process-global mutation timestamp without changing adapter bytes.

Base probes explicitly send `"adapter": null`. The driver then calls the public unload
endpoint, requires every surface and response header to return to base, and
compares canonical streamed semantic deltas from identical pre-load and
post-unload base requests exactly. Dynamic IDs and creation timestamps are the
only excluded envelope fields. Load, same-revision reload, and unload
barrier-swap reasons must appear in order,
at least one captured graph must be invalidated, and the actor, graph slots,
owners, process, and snapshot directory must drain cleanly.

The second arm reuses the same binary under a separate maintenance-profile TOML
file with graphs disabled, `memory.kv_autoscale = true`, and
`memory.kv_force_blocks = 1`. It requires exactly one structured
`gpu_memory_operation` resize with reason `forced_configuration`, an initial
capacity above the target, exact `requested_blocks` and `actual_blocks` of one,
and finite nonnegative barrier/GPU/model wait and total mutation durations.
Because this profile intentionally disables inference, `/health` must return a
structured HTTP 503 with status `maintenance`, a failed
`inference_admission` check, and every other readiness check passing. That is
maintenance readiness, not a transport or startup failure.
`/v1/config` must observe the one-block physical capacity while health and debug
report the requested target and `config_file` provenance. A public chat request
must return HTTP 503 with `inference_disabled_by_profile` without changing
batching admission, prefill, decode, or used-block counters.

Either arm fails on an HTTP or stream error, malformed identity, stale adapter
header, changed base output, missing graph invalidation, wrong resize target,
inference work in maintenance, device-fault signature, forced/nonzero shutdown,
or private snapshot residue. The receipt publishes transition and rejection
counts, adapter bytes, load/reload/unload latency, active and queued overlap
populations, the queued request's exact adapter and actor-queue milliseconds,
revision-header matches, the positive barrier observation, graph invalidations, resize source
and target blocks, released bytes, coordination wait, mutation duration, and
all failure counters. This gate proves one controlled lifecycle at one source
revision; it does not replace concurrent mixed-load stress, the graduated
concurrency gate, or the long soak.

The current retained Strix Halo ROCm evidence is
`20260721t141928678273z-rocm-strix-halo-serving-rocm-public-muta-1437dc4bd3-v1`.
It is bound to clean pushed source `fe15f555f`, passed strict current-source,
local-artifact, and known-commit validation, and completed both arms. The active
request retained `adapter_ms: null`; the queued request reported 3,530.155 ms
of adapter time inside 3,531.217 ms of actor queue, and both headers matched the
same revision. The run retained three ordered adapter transitions, five graph
invalidations, exact base restoration, and a 4,096-to-1 block resize releasing
8,587,837,440 bytes. Request, revision-header, base-output, device, shutdown,
and snapshot-residue failure counts were all zero. This is positive mutation
and maintenance-lifecycle evidence, not a throughput, soak, parity, or release
claim.

Command-result evidence is accumulated at each completed boundary rather than
synthesized only after both arms pass. A failed result therefore retains any
completed binary build, copied adapter identity, public load or unload,
semantic-output hash, graph invalidation, physical resize, rejection, and
transition already observed. `arms_started` and `arms_completed` distinguish an
unstarted arm from a partial one. Request failures count actual HTTP/stream
failures only; dirty shutdown and snapshot residue derive from teardown rather
than being invented for every failed case.

The mixed serving run also attests the 256-token prompt-work ceiling
(`server.max_prefill_tokens_per_cycle`), the 512-token combined scheduling
budget (`server.max_batch_tokens`), the candidate eight-layer yield ceiling
(`server.max_prefill_layers_per_cycle`), and both startup provenances. Admission
and resumable prefill share the token ceiling after ready decode rows reserve
their tokens. A retained token chunk then yields between transformer-layer
groups without replaying completed layers. The receipt records both effective
values, processed-layer and layer-yield counts, plus cumulative/max actor-phase
times; a run that exercises no inter-layer yield fails. A chunk is charged to
the new-token ceiling exactly once when selected, not again when its retained
final layer completes. After at most four staged-priority prefill dispatches,
the next prefill turn is global round-robin; priority may accelerate only the
shortest tail of at most four token chunks.
The receipt records this bounded-priority count and fails when the mixed
workload does not exercise it. Any ITL outlier remains a failure even when its
phase is explained unless it overlaps the required named-host thermal pacing
controller. Thermal-attributed counts must reconcile exactly; non-thermal
attributed and unexplained counts remain disqualifying.
The same run attests an effective decode width of four, four bounded
short-prefill staging slots, and a total active-request ceiling of eight in
both health and debug state. It also requires a maximum staged-priority burst
of four before the mandatory global prefill turn. Measurement must record at
least one staging admission, at least one rotating staged-priority forward, and
an observed active width above four without ever exceeding eight.
Staged-priority forwards must remain a subset of the bounded short-priority
count. The final cancellation drain requires ordinary decode, prefill, staged
occupancy, the waiting queue, and the actor-cycle-idle gauge all to reach zero.
This proves that the latency path ran without treating the staging capacity as
a wider backend decode batch, accepting an active prefill as drained, or racing
the final cooperative wait.

### ROCm decode-width selection

Do not raise `server.max_decode_batch` from a single throughput sample. The
committed `serving-rocm-decode-width-campaign-v1` workload is the bounded,
offline autotuner for the current Strix Halo ROCm profile. It reruns the
accepted width-4 control, then tests primitive-supported widths 6 and 8 in
ascending order against one source-built release binary. Width 2 is excluded:
its retained rejection was roughly 30% slower than width 4 and failed the
deterministic argmax-route coverage and non-thermal ITL gates. Each arm
gets a fresh server process, private adapter/snapshot directories, the same
typed configuration apart from the derived decode/staging/active widths, a
97 C package-temperature guard, and a completed cooldown before the next arm.
The source build itself uses the bounded Cargo wrapper with one job, a 50%
CPU quota, a 15 GiB available-memory floor, no swap-backed build service, and
`gfx1151` as the only ROCm architecture.

Run it only from a clean commit already present at `origin/main`:

```bash
PATH="$HOME/.cargo/bin:$PATH" python3 scripts/qualification/run.py \
  --variant rocm \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-rocm-decode-width-campaign-v1.json
```

Every arm runs the established mixed-load contract: 1,312 fixed deterministic
output tokens, the long-prefill/slow-consumer/cancellation overlap, eight
seeded 32-token sampled requests at temperature 0.7/top-p 0.9/top-k 40, and a
post-sampling deterministic replay canary. An arm is correctness-qualified
only when both deterministic and sampled traffic actually reach its declared
width; graph capture and replay succeed without fallback or failure; the
fused W8A8 sampled LM-head route reaches the same width; all request, device,
external-yield, finite-value, output-oracle, outlier, thermal, shutdown, and
residue gates pass. A failure stops wider arms and makes the campaign fail,
rather than interpreting a crash or partial measurement as evidence for the
last smaller width.

Among correctness-qualified arms, the selector scores the minimum of the
deterministic and sampled throughput ratios relative to the freshly rerun
width-4 control. A wider arm
is ineligible when p99 ITL, TTFT, or end-to-end latency regresses more than 25%,
or when peak GPU allocation grows by more than 1 GiB. The narrowest width
within 2% of the best score wins, and promotion above width 4 additionally
requires at least a 3% minimum-throughput gain. The receipt exposes every
candidate's throughput, tail latency, exact observed widths, graph activity,
deterministic argmax dispatch/row coverage, fused-sampling activity, peak
memory, temperature/guard/pacing counts, and attributed/unexplained outlier
counts. Its
bounded `details` value records one short selected/rejected summary per arm;
it does not treat raw console logs as the experiment verdict.

This is qualification-time autotuning, not online self-tuning. It never changes
a live scheduler, retries a failed width in the same process, edits a user's
configuration, or claims portability to a different GPU/model/source tree.
After reviewing a passed receipt, pin the selected integer in
`server.max_decode_batch`, rerun the focused correctness gates, then rerun the
30-minute development soak and serving benchmark from that exact pushed source.
CUDA, Vulkan, and Metal require separate machine-local campaigns before their
backend defaults or checked profiles can be promoted.

The pressure peer also requires terminal request-scoped performance metadata.
The 256-token peer is twice the ordinary response length. The driver dispatches
it before the slow consumer and waits for its first
producer-ready token before opening the stalled socket. The acceptance window
then requires further producer-ready tokens during and after the slow
consumer's request-attributed backpressure interval. This ordering proves
continuity of an already-decoding peer instead of falsely requiring a queued
request to have emitted before a pressure window that already started.
The effective workload records a 90-second delivery-pressure observation
window. It remains bounded by the case deadline but is long enough for the
declared 2-second stall grace to complete after a pressure onset delayed by the
contained scheduler; a hidden fixed 45-second observer timeout is not part of
the contract.
Its actor queue, slot-admission, and admission-to-first-ready wall durations are
recorded separately and must fit inside TTFT; accumulated model prefill must fit
inside admission plus admitted-prefill wall time. Missing, duplicate,
nonnumeric, or internally impossible phase evidence fails the run. These fields
distinguish active-set saturation from slow admitted prefill before any
scheduler policy is changed.

For the historical dynamic-runtime A/B, run each of `default`,
`autoscale-off`, `graphs-off`, and `both-off` separately. These four arms now
pin `server.serving_profile = "experimental"` in the generated typed launch
file so their requested graph/autoscale differences retain the semantics they
had before stable became the default:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant default \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-mixed-rocm-v1.json
```

The variant named `default` preserves the graph-on/autoscale-on A/B baseline,
not the production serving default. The manifest intentionally applies one
shared qualification transport envelope to every arm.

When deterministic output differs even with graphs and KV autoscaling disabled,
run `both-off-prefix-cache-off` against `both-off`. The diagnostic arm also
writes `prefix_cache.enabled = false`; no environment override is accepted. Its
effective configuration differs only in requested/effective prefix-cache state
and reason. Startup, post-measurement, post-sampled, and final-canary health
attestation require the batching capability to remain false and every cache
capacity, lookup, hit, lease, pending release, retained block, entry, and state
byte counter to remain zero. The receipt retains the enabled bit, measurement
baseline residency, lookup/hit deltas, and final residency. This arm isolates
prefix/recurrent-state reuse; it does not silently redefine the historical
four-arm performance matrix.

### ROCm Synchronization A/B

Run the synchronization policy checkpoint before a longer mixed-load or soak
after changing ROCm stream ownership, eager operators, graph boundaries, tensor
handoffs, allocator lifetime, or host readback. The workload is self-contained:
it builds one exact binary and sequentially starts two isolated servers from
that binary, first with `legacy_host_barriers` and then with `stream_ordered`.
Do not launch the arms by exporting old `KILN_ROCM_*` switches yourself.

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant legacy-vs-stream-ordered \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-rocm-sync-ab-v1.json
```

Both arms use the experimental serving profile because `stream_ordered` is an
explicitly gated experiment. The workload disables ROCm graphs, physical KV
autoscaling, and allocator reclaim in both arms so the first checkpoint changes
only synchronization discipline. It warms one short request, then runs fixed
32-token waves at concurrency 1, 2, and 4 with prompt lengths from 16 to 384
words. The per-request deadline is 90 seconds and build/startup/work across
both arms shares a 10-minute deadline. Each server then gets the standard
separate 60-second graceful teardown bound before a forced kill is reported as
failure. Gaps at least 250 ms are retained as stall evidence, and any
inter-token gap at least two seconds fails. This is the small admission
checkpoint; it does not replace the full mixed-load, memory pressure,
30-minute development soak, or final 24-hour soak.

Before timing begins, each arm also runs the same fixed-seed, non-streaming
provenance request through the public API. The existing full-model correctness
parser validates its action-token coverage and finite selected-token
log-probabilities. The two normalized semantic traces, action token IDs, and
behavior log-probabilities must match exactly. This probe is excluded from the
timing and synchronization deltas, so semantic confidence does not contaminate
the measured A/B window.

At startup, after warmup, and after the measured wave, the driver requires the
resolved policy from both `/v1/config.accelerator_runtime` and
`/health.decode_runtime.accelerator_runtime` to match the arm exactly. It also
requires `/health.decode_runtime.rocm_synchronization` to expose all 23 fixed
reason dimensions, internally consistent aggregate counts, no telemetry error,
`cleanup_quarantined=false`, and monotonically increasing counters. Every health
reason/count/duration is
reconciled with these Prometheus families:

```text
kiln_rocm_synchronization_policy_info{mode}
kiln_rocm_cleanup_quarantined
kiln_rocm_synchronizations_total{reason,scope}
kiln_rocm_synchronization_wait_seconds_total{reason}
kiln_rocm_synchronization_skipped_total{reason}
```

The Prometheus quarantine gauge must remain `0` while telemetry availability is
`1`. A true health field or nonzero gauge is a hard arm failure: do not classify
it as an expected graph fallback or continue collecting throughput samples.

The legacy arm must execute an `external_yield` device wait and skip no
barrier. The stream-ordered arm must execute an `external_yield` stream wait
and skip at least one proven same-stream barrier. Both must produce the same
request names, prompt-token counts, exact 32-token length termination, no
request or device fault, clean unforced shutdown, and no private snapshot
residue. The receipt records TTFT, E2E, ITL p50/p99/max, throughput, peak GPU
memory, stall and pause counts, aggregate wait scope/count/time, and skipped
barriers for each arm. After each concurrency wave, its bounded raw output
records maximum ITL, stall/pause counts, and every per-reason counter/time
delta for that interval. This narrows pause attribution to a specific wave and
reason family instead of assigning a temporal gap speculatively to VRAM
rebalancing.

Passing this checkpoint means stream ordering preserved the bounded workload
and the diagnostics are trustworthy. It does not mean stream ordering is
faster: review both arms' latency, throughput, memory, and per-reason traces,
then proceed to the existing mixed-load workload with the promising policy.
New qualification drivers use the mechanically derived
`KILN_ACCELERATOR_ROCM_SYNCHRONIZATION_MODE`,
`KILN_ACCELERATOR_ROCM_GRAPH_MODE`, and
`KILN_ACCELERATOR_ROCM_GRAPH_CACHE_ENTRIES`, and
`KILN_ACCELERATOR_ROCM_GRAPH_CACHE_MAX_BYTES` names. Retired shorter graph
variables are ignored and are not evidence vocabulary.

### CUDA Synchronization Evidence

Every CUDA correctness, serving, graph, resize, reclaim, or training campaign
on a 4090 must retain synchronization snapshots before warmup, after warmup,
before the measured window, after each independently reported wave, and after
settled teardown. Read both
`/health.decode_runtime.cuda_synchronization` and, when trusted debug is
enabled, `/v1/debug/model-state.cuda_synchronization`. The two snapshots must
agree at the same settled boundary. Do not insert a synchronization merely to
make the read convenient.

An active CUDA arm requires `active=true`, `telemetry_available=true`, no
`telemetry_error`, and exactly one row for each of these twelve reasons:

```text
explicit_device_drain explicit_stream_drain tensor_handoff external_yield
in_place_mutation memory_reclaim graph_boundary full_attention_handoff
model_handoff host_readback allocation_lifetime global_state_mutation
```

Reason rows must be unique and in the declared order. Counts and nanoseconds
must be nondecreasing; each aggregate total must equal the saturating sum of
its rows. A failure delta for any reason is a hard failure even when the request
later falls back or returns an apparently valid token. A temporarily in-flight
wait may not yet appear because accounting occurs when the driver returns;
sample only after the workload boundary has settled or report the arm failed to
settle within its deadline.

Reconcile every settled JSON row with these Prometheus families:

```text
kiln_cuda_synchronization_active
kiln_cuda_synchronization_telemetry_available
kiln_cuda_synchronizations_total{reason,scope}
kiln_cuda_synchronization_failures_total{reason}
kiln_cuda_synchronization_wait_seconds_total{reason}
```

The active and availability gauges must both be `1`. Each reason must have one
device and one stream series, one failure series, and one wait-seconds series;
unknown or duplicate labels fail. The Prometheus count values must equal the
JSON counts exactly, failures must remain zero, and seconds must match
`waited_ns / 1e9` within serialization precision.

Require evidence only for routes the arm actually declares. Ordinary serving
must advance `external_yield` device waits. A graph-enabled arm must advance
`graph_boundary` stream waits and separately prove graph capture/replay. A
physical KV-resize arm must advance `global_state_mutation`; a CUDA pool-trim
arm must advance `memory_reclaim`; tensor/model/training handoff arms must
advance their matching reason. Reasons outside the workload remain valid zeroes
and must not be stimulated with artificial work.

For pause diagnosis, bracket each concurrency or mutation wave with settled
snapshots and retain per-reason count/time deltas beside request-local ITL and
phase data. A rising counter identifies a wait class that occurred in the
window; it does not by itself prove that wait caused a particular request gap.
Attribute causality only when the request-owned phase or a narrower timestamped
event overlaps the same monotonic interval. The CUDA-machine receipt must state
which expected reason deltas were exercised, which remained zero by design,
the maximum per-wave wait-time delta, and that the raw-source inventory contains
no production driver waits outside the typed wrappers.

### ROCm Development Soak

After a material ROCm serving change, run the committed 30-minute development
soak from a clean, pushed source tree:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant autoscale-off \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-rocm-development-soak-v1.json
```

The driver builds once and starts one server process with eight active-request
slots: four ordinary decode rows plus four staged prefill slots. Its
source-bound TOML pins the route-invariant 256-token ROCm prefill boundary, a
8-layer prefill quantum, a 512-token combined scheduling budget, effective
decode width four, and a 50 ms cooperative actor-cycle idle. Health and trusted
debug must report every value with `config_file` provenance. Its graph-required
operating point reserves one protected geometry for each declared active owner
and pre-reserves zero transition entries, so the checked eight-entry ceiling is
`8 * 1 + 0`. Runtime
admission consumes unused global headroom freely. Only at entry or byte
saturation does it reclaim idle owners, followed by the minimum deterministic
fair-LRU active entries; the incoming candidate counts toward its owner's share
and one graph remains protected for every active owner. This settled relief
creates transition room only when needed instead of retaining additional
native graph objects continuously. Every narrow retirement preserves recurrent-
state slots and continuity. The driver warms ROCm graphs and fills the bounded
prefix cache to its declared entry/state capacity before recording the post-
warmup memory baseline or starting the 30-minute measurement clock.

The preceding twelve-request operating point was evidence-based. With the same
Strix Halo binary, model, seed, workload, and 120-second minimum measurement, 12 entries completed
the identical first seven-wave sequence in 115.461 seconds versus 144.617
seconds at 24 entries, a 25.3 percent improvement. The 12-entry arm performed
seven settled measured capacity transactions and twelve captures with zero
fallback, graph failure, or live-slot loss. Its active SCLK was lower, not
higher, so the result isolates excessive retained native graph population on
this host rather than a graphics-clock advantage. It does not prove that eight
entries are safe, sufficient, or optimal. The current eight-entry value follows
the candidate's smaller active-request capacity and must earn its own
source-bound mixed-load and soak receipts; larger deployments remain subject to
their own qualification.
It then exercises complete fixed-prompt concurrency cycles, including periodic
cancellation, until GPU-used and server-RSS deltas remain within 64 MiB and 16
MiB respectively for two consecutive cycles. This convergence requires at
least four cycles and fails after eight instead of silently moving a growing
allocator into the baseline. The result retains completed cycle, request,
cancellation, final-delta, and maximum-delta metrics even when convergence
fails before the measured phase begins.

The measured phase keeps that process under the same fixed-output client waves
at concurrency 1, 8, and 12 with prompt lengths spanning multiple sequence
buckets. The 12-way client wave is deliberate admission pressure; at most eight
requests may be active and the remainder wait outside the active set. Every
fifth wave also cancels a longer request using a unique marker.
Slot prompts repeat across waves so prefix hits and cached-block reuse are
measured. After each wave, the driver requires the engine to drain, every used
KV block to be owned by the prefix cache, zero active cache leases or pending
releases, stable cache residency, zero active graph slots or row timelines,
at most eight retained graphs and slots, the process to remain alive, and
runtime/debug policy attestations to remain consistent. Idle slots and their
native graphs remain resident for reuse. The final drain requires every
retained slot to be idle, the measured phase must exercise slot reuse, and the
cooperative actor wait must remain source-attested rather than being treated as
thermal-sensor pacing.

Every response in graph warmup, prefix-cache warmup, stabilization, and
measurement must also satisfy the declared ascending-sequence oracle. A
cancellation is confirmed only when its first four semantic deltas form the
same valid prefix before the client disconnects and the server proves drain;
disconnect cleanup cannot mask already-corrupt output.

The result fails on any request, deterministic-response-oracle, or cancellation
error, graph capture/replay
failure, typed eager fallback, backend synchronization failure or 100 ms slow
sync, device-fault signature in either a log message or structured error,
non-finite response error, unexplained ITL outlier, capacity change,
unaccounted block, active cache lease, pending release, dirty shutdown,
snapshot residue, or GPU/RSS peak more than 512 MiB above the post-warmup
baseline. Attributed outliers remain counted for review. The receipt also
records p50/p99/p99.9 TTFT and ITL, graph activity, prefix-cache reuse,
graph/slot residency and reuse, external-yield synchronization, memory
baselines/peaks, request/token counts, aggregate measured output throughput,
and cancellation count. `output_token_throughput_per_second` is exactly the
successful measured completion-token count divided by
`soak_duration_seconds`. It excludes setup, graph/prefix warmup, and
stabilization, includes all wall time and thermal pacing after the measured
phase begins, and is zero for a partial result whose measured duration is zero.
The numerator and denominator remain separate receipt metrics so the rate can
be independently reproduced. Shutdown must return zero without force after the
decode worker is joined, and snapshot cleanup must leave no residue.

ROCm results always declare the base serving metrics plus the shared host
memory, swap, temperature, thermal-pacing, and accelerator clock/power metrics.
Vulkan declares the same accelerator telemetry schema plus its resident-prefill,
process-DRM, allocator, and mapping extensions. The same backend-specific
schema is used for both complete and partial results, so a later failure cannot
replace retained ROCm warmup evidence with a metric-set mismatch caused by
Vulkan-only fields.

The accelerator sampler resolves exactly one Linux DRM device whose `vendor`
is AMD `0x1002`, then exactly one `amdgpu` hwmon directory below that device.
It selects SCLK by `freq*_label=sclk`, average package power by
`power*_label=PPT`, and edge temperature by `temp*_label=edge`; it never embeds
boot-dependent `cardN` or `hwmonN` numbers. Every 250 ms it reads those labeled
values with `gpu_busy_percent`, while `pp_dpm_sclk` supplies the advertised
maximum SCLK. Missing, ambiguous, malformed, or implausible inputs are explicit
telemetry outcomes rather than zero readings.

Receipt metrics include telemetry availability and errors, total and active
sample counts, p50/peak busy percentage, advertised SCLK, active-sample
minimum/p50/maximum SCLK, active SCLK samples below half the advertised maximum,
active p50/peak PPT power, and active p50 plus overall peak edge temperature.
An active sample is one with at least 50 percent GPU busy, preventing idle
clocks and power from distorting the workload summaries. Aggregates are scoped
to the measured phase; partial results retain samples collected after
measurement began. ROCm requires available, error-free telemetry with at least
one measured and one active sample. Vulkan uses the same sampler when AMD sysfs
is available, but permits an unavailable device so the workload remains
portable to non-AMD Vulkan implementations; once an AMD telemetry source is
selected, any read error still fails the case. The below-half-max count is
diagnostic evidence, not by itself a performance acceptance threshold.

This `kind: soak` workload is intentionally a non-comparative pass/fail gate,
so its `comparison_policy` is null. Do not use it to claim relative throughput
or latency; its explicit rate answers what the completed soak delivered, not
what the machine can deliver at a controlled concurrency or against vLLM. Use
the serving benchmark protocol for those claims. The 30-minute receipt also
does not replace the final 24-hour ROCm phase soak.

The first clean pushed-source run after enforcing the route-invariant
256-token ROCm boundary is retained as
`20260719t113118927919z-rocm-strix-halo-serving-rocm-development-053e89eca9-v1`.
It passed the soak's correctness, graph, memory, thermal, and lifecycle gates,
but its diagnostic rate was only 7.108 output tokens/second: 12,992 tokens over
1,827.870 measured seconds. The immediately prior accepted 64-token-route run
delivered 13.194 tokens/second from 23,872 tokens over 1,809.362 seconds. This
46.1 percent drop is a release-blocking performance counterexample, not a
failed safety receipt and not a vLLM comparison. Thermal pacing does not explain
the direction: it fell from 666.690 to 98.179 seconds, while active p50 SCLK
rose from 2.507 to 2.621 GHz and p50 GPU busy fell from 43 to 30 percent.
Actor-prefill slow-phase events rose from 612 to 3,548 despite fewer requests;
per-request prefill residence rose 1.76x and per-token decode residence rose
2.28x, while per-request actor-queue and synchronization time stayed nearly
flat. Keep the receipt, investigate the correct-route layer-yield/dispatch cost
with a same-binary A/B, and do not promote the final ROCm endurance run from
this result.

Build, runtime setup, and measurement use independent absolute deadlines. The
exact source-bound build has its own 900-second limit from the checked build
specification. A successful build starts a fresh 1,200-second runtime setup
envelope for server startup, warmup, and stabilization; compilation time cannot
consume the runtime evidence budget. Only after stabilization passes does a
fresh 1,920-second measurement envelope begin: 1,800 required seconds plus one
120-second request deadline so the last admitted wave can settle. The outer
4,200-second case timeout is exactly those three limits plus a separate
180-second teardown margin. Thermal pacing remains charged to the runtime setup
or measurement phase in which it occurs. A setup failure records
`soak_duration_seconds=0`; it cannot be mistaken for measured performance.

### ROCm Final Endurance

After all source changes stop and the release-candidate commit is pushed, run
the committed 24-hour ROCm endurance gate from that exact clean checkout:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant rocm-endurance \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-rocm-endurance-v1.json
```

This is the final-duration form of the qualified ROCm development soak, not a
new performance arm. It retains the same build policy, experimental serving
profile, fixed KV capacity, disabled allocator reclaim, graph-required
execution, eight-entry graph ceiling, prefix-cache policy, batching limits,
prompt geometry, wave sequence, cancellation cadence, response oracle,
stabilization bounds, memory limits, thermal controls, and complete metric set.
Only the workload/variant identity, independent seed, generic driver path, and
duration/deadline envelope differ. The manifest-driver equality test rejects
any unreviewed difference before hardware work begins.

The driver gets an independent 900-second bounded source build, then 1,200
seconds for server startup, graph/prefix warmup, and memory stabilization. Once
stabilization passes, the measured envelope is 86,520 seconds: 86,400 required
seconds plus the existing 120-second final-request settlement allowance. The
outer case timeout is 88,800 seconds, adding the three envelopes and a separate
180-second teardown allowance. Build or setup time never counts toward the
24-hour measurement, and thermal pacing remains inside whichever runtime
envelope it overlaps.

All development-soak gates remain mandatory for the entire measured day:
ascending-sequence response and cancellation-prefix correctness; required
native graph capture/replay with no fallback or failure; zero request, device,
non-finite, synchronization, KV-accounting, cache-ownership, unexplained-ITL,
worker-residue, forced-shutdown, nonzero-exit, or snapshot-residue failure;
bounded GPU/RSS peak and end growth; host memory/swap containment; error-free
accelerator telemetry; package-temperature containment and post-exit cooldown.
Every ITL gap above the adaptive `max(250 ms, 5 * rolling p50)` boundary must
retain a bounded reason and the final unexplained count must be zero.

The compact receipt belongs under `qualification/receipts/rocm/strix-halo/`
and must validate against the pushed release-candidate source and retained raw
artifacts before the ROCm phase is checked complete. A failed or interrupted
run is evidence, not a partial pass: retain its compact failure receipt when
one was published, fix only a demonstrated defect, push the corrective commit,
and restart the complete 24-hour clock from that new source. Do not edit the
duration in the shell or promote the 30-minute development receipt as final
endurance evidence.

### Vulkan Resident-Prefill Oracle

Run the focused resident-state gate on a clean, pushed source tree before the
Vulkan development soak:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan-resident-prefill-oracle \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-vulkan-resident-prefill-v1.json
```

This is a source-bound `kind: correctness` gate for the experimental serving
profile. It builds the Vulkan-only `kiln` binary through the bounded Cargo
wrapper, verifies the executable and Vulkan execution identity, starts one
server from one typed configuration, and keeps cross-request prefix reuse,
graphs, KV resizing, and allocator reclaim disabled. Stable and maintenance
profiles are outside this workload because their typed policy must keep
resident prefill disabled with zero route activity.

After a singleton oracle-valid warmup, the driver dispatches two four-request
cohorts through thread barriers. Every row uses the same 16-word prompt length,
while completion limits are `8/12/16/20` in the first cohort and
`20/16/12/8` in the second. Equal prompt lengths make rows ready together;
different completion lengths force each cohort to shrink through changing
active-row sets and a singleton tail. Both cohorts remain in the same server
process so the second repetition detects poisoned parked state, stale row
identity, incorrect row strides, and unsafe capacity reuse. Every response
must terminate by length at its exact limit and remain a prefix of the
ascending zero-padded integer oracle. Terminal response metadata must attest
resident-prefill use on at least two requests.

Enablement is not execution evidence. The final health delta must report at
least one resident forward, more resident rows than forwards, at least one
completed row, and a maximum resident batch of at least two and at most four.
Attempts must reconcile exactly with forwards plus initial declines when no
route failure occurred. Initial declines, route failures, active resident rows
at drain, batching errors, device faults, graph activity, external-yield sync
failures or slow calls, response errors, and semantic-oracle failures must all
be zero. The parked batched-state cache may retain reusable capacity, but it
must end with zero active leases, no miss while leased, and no completion,
replacement, or explicit-invalidation eviction. The direct recurrent-state
registry, active KV blocks, unaccounted blocks, prefix-cache state bytes,
prefix leases, and pending prefix releases must all end at zero.

The same fail-closed host controls used by the Vulkan soak cover build, model
load, prewarm, both cohorts, drain, and teardown: at least 8 GiB Linux
`MemAvailable`, no more than 512 MiB new swap, and `k10temp/Tctl` below 97,000
millicelsius. Process-scoped DRM sampling records baseline, end, peak, and peak
growth and rejects more than 1 GiB of active growth. Shutdown must return zero
without force, remove the private model snapshot, and leave no payload residue
or request worker. The receipt retains the binary/config identities, a
canonical semantic-output digest, per-cohort duration and route metadata, all
resident counters, cache ownership, memory, temperature, synchronization, and
lifecycle metrics.

Passing this oracle permits the unchanged 30-minute development soak to test
longer-lived allocator convergence, cancellation, latency, and memory behavior.
It does not qualify stable-profile admission, eight-hour endurance, broad
prompt-length coverage, or throughput competitiveness with vLLM.

### Vulkan Development Soak

Run the Vulkan peer on a clean, pushed source tree after the serving baseline
and before a multi-hour Vulkan gate:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan-development-soak \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-vulkan-development-soak-v1.json
```

This arm uses the same source-built, one-process, post-warmup qualification
model as the ROCm soak, but its hardware accounting is deliberately different.
The checked profile selects a typed 128-token prompt-work ceiling while keeping
the shared four-layer yield ceiling. It also sets `server.max_decode_batch=2`
and `batching.prefill_admission_quantum=2`. The actor therefore admits an equal
pair together while retaining two decode rows plus two staged prefill rows, and
health/debug must attest the derived two-slot staging and four-request active
ceilings. On this Strix Halo, 64-token chunks made
regular progress but could not finish the declared eight-way long-prompt wave
before the unchanged 600-second request deadline. Repeated four-request A/B
runs at 128 tokens passed eight exact semantic oracles with stable process
history; 256 tokens was faster but corrupted every concurrent response while
the exact same prompt remained correct in isolation. The soak therefore binds
128 explicitly and fails if health/debug reports another value or provenance.
The candidate qualified load alternates one- and four-request waves with
16-token completions. Each one/four pair selects one of the fixed
16/32/64/96-word prompt slots; all four rows in the concurrent wave have the
same prompt length and distinct fixed row identities, and successive cycles
rotate through every slot before repeating. This forces an equal-ready resident
cohort and establishes every declared prompt/batch scratch capacity before the
measurement baseline. It is an enforced qualification operating point, not the
product-wide default or a claim that larger Vulkan quanta, more than four
simultaneously active requests, or longer-prompt throughput are competitive or
qualified.

Its timing envelopes are likewise independent and explicit in the effective
configuration. The exact source build gets the checked 900-second build limit;
after it succeeds, startup, warmup, and stabilization get a fresh 1,800
seconds. After those gates pass, the 30-minute measurement gets a fresh
2,400-second absolute deadline: 1,800 required seconds plus the unchanged
600-second per-request containment window. The outer case timeout is 5,280
seconds, exactly those limits plus 180 seconds for cancellation, server join,
private-snapshot cleanup, and result publication. Wave request threads get a
fixed 10-second cleanup window inside that teardown allowance even when the
setup or measurement work deadline has already expired. The driver sets their
shared abort signal, waits for them, records `request_worker_residue_count`, and
rejects any survivor; an
expired work deadline therefore cannot erase its own cleanup budget. Setup time
is never reported as measured soak time.
Raw stdout and stderr artifacts are flushed after every captured chunk, so a
monitor can follow structured progress while the case is still running rather
than waiting for a user-space file buffer to fill.
Steady-state warmup counts are committed after each response cohort has fully
completed and the server has drained, before the drained health snapshot is
judged. A graph, ownership, or cache-policy failure in that snapshot therefore
retains the completed request and wave counts while still failing the case;
requests from an incomplete or undrained cohort are never counted.

The clean 2026-07-16 run at commit `2587de1dd` is the pre-resident-prefill
counterexample, not a passing soak. It completed 30 oracle-valid stabilization
responses and one confirmed cancellation, then exhausted the 1,800-second
setup envelope during the second cycle's final width-four wave. Measurement
never started, so its zero measured request, wave, latency, and duration metrics
are explicit partial-evidence sentinels. The successful width-eight wave reported
256.9-562.5 second TTFT for 472-1,240 prompt tokens. Actor telemetry showed
continuous rowwise generic prefill in four-layer, 128-token slices of roughly
1.4-2.3 seconds, not an unexplained stall or mid-request VRAM rebalance. The
first complete cycle also grew process DRM by 91,250,688 bytes and therefore
failed the unchanged 64 MiB stabilization-delta gate. The retained receipt is
`qualification/receipts/vulkan/strix-halo/20260716t041351870594z-vulkan-strix-halo-serving-vulkan-developme-b5eb848d54-v1.json`.
It records clean bounded teardown, zero device faults, zero direct prompt-state
ownership after drain, at least 20,062,093,312 bytes of available host memory,
53,248 bytes of swap growth, and a 93,375 millicelsius package peak. The next
qualification attempt requires a safe generic-prefill throughput correction
or an explicitly narrower published service region; increasing the setup or
request deadline would not resolve this gate.

The clean `7f8903097` rerun after the focused resident-prefill oracle is the
current counterexample. It advanced farther, completing all 17 responses and
one confirmed cancellation in Cycle 1, then the one-, four-, and eight-way
waves in Cycle 2. The setup deadline expired in Cycle 2's final four-way wave
with three workers still awaiting completion. Thus 30 stabilization responses
were oracle-valid, but only one complete cycle was eligible for memory
attestation and measurement still never began. The retained receipt is
`qualification/receipts/vulkan/strix-halo/20260716t062158784737z-vulkan-strix-halo-serving-vulkan-developme-b5eb848d54-v1.json`.
Cycle 1 grew process DRM by 96,256,000 bytes, live Vulkan ownership by
95,944,768 bytes, and RSS by 319,447,040 bytes, so it correctly remained
nonstable. It recorded 1,249 exact recurrent-cache reuses, 571 resident-capacity
reuses, and 542 resident-prefix views with zero miss while leased, completion
eviction, explicit invalidation, or replacement eviction. The run had zero
batching or device faults and zero unexplained ITL outliers; host availability
stayed above 17,856,610,304 bytes, swap grew 12,288 bytes, and package
temperature peaked at 96,000 millicelsius. Shutdown was unforced and zero with
no snapshot residue or surviving process. Because that receipt predates the
failure-path evidence repair, its zero resident-prefill metrics are partial-
evidence sentinels rather than proof that the route was idle; the raw trace
contains multi-row, full-stack resident forwards. Current workloads separately
declare stabilization resident-prefill enablement, attempts, forwards, rows,
completed rows, maximum width, declines, route failures, and active rows at the
last drained boundary. Those deltas are published on both pass and failure,
while the existing unprefixed fields remain measurement-only. The performance
gate itself remains unchanged: either improve this heterogeneous
region or publish and enforce a narrower supported Vulkan service envelope
before qualifying that envelope.

ROCm's device-global memory counter cannot isolate another desktop process.
The Vulkan driver therefore additionally sums the server process's `drm-memory-vram`,
`drm-memory-gtt`, and `drm-memory-cpu` records from `/proc/<pid>/fdinfo`,
deduplicated by DRM client ID. It samples `VmRSS`, `RssAnon`, `RssFile`,
`RssShmem`, and `VmSwap` from `/proc/<pid>/status` separately.

Both committed Strix Halo development soaks use the same independent 250 ms
host controller. Its memory guard sends `SIGTERM` to the complete server process
group if Linux
`MemAvailable` falls below 8 GiB; missing or malformed host telemetry also
fails closed. It follows termination with `SIGCONT`, which is harmless for a
running group and lets a thermally paced group execute shutdown immediately.
The receipt records the starting, minimum, and ending available
memory, starting/peak/ending swap use, and swap growth. More than 512 MiB of
new swap is a failure even when the 8 GiB floor was not crossed.

The same 250 ms safety loop independently monitors the Strix Halo package
temperature from Linux hwmon. The committed workload selects the sensor by the
stable pair `name=k10temp` and `label=Tctl`, never by a boot-dependent
`hwmonN` path, and sets a 97,000 millicelsius limit. The driver resolves exactly
one matching `temp*_input` after launching the server. A missing, ambiguous,
non-integer, or implausible sensor reading fails closed and sends `SIGTERM` to
the server process group; a valid reading at or above the limit does the same.
The result retains starting, peak, and ending package temperature plus the
thermal-trip count, including startup or pre-measurement failures. The effective
configuration records the sensor name, label, limit, and poll interval, so a
receipt cannot silently inherit a different sensor or threshold. This guard
covers model load, native prewarm, warmup, stabilization, measurement, and final
drain. The preceding source build independently resolves the same stable sensor
selector and enforces a stricter 90,000-millicelsius threshold at the same
cadence around its complete transient compiler/linker service. A build trip
exits with status 3, while a missing or invalid sensor fails preflight with
status 2; neither can be recorded as a successful source-bound build.

Current ROCm and Vulkan serving soaks use this controller in hard-limit-only
mode. No setup, stabilization, measurement, drain, or teardown path may send
`SIGSTOP` while accelerator work can be outstanding. A 97,000-millicelsius
sample terminates the complete server group and fails the case; the lower
93,000-millicelsius ceiling in the shared standalone serving benchmark fails
that shorter experiment earlier. Sensor ambiguity, read failure,
controller-thread failure, and signal failure remain fail-closed conditions.
The receipt retains the compatibility `host_thermal_pacing_*` metrics, but a
current pass requires active, started, and completed pacing counts all to be
zero. Any ITL outlier attributed to host thermal pacing is also a failure,
because the current policy cannot legitimately create such an interval.

The guard still sends `SIGCONT` after fail-closed termination as defensive
cleanup for a process group stopped by an external actor or a historical
controller. This does not arm pacing and does not make process suspension part
of the current serving contract. Pre-launch and post-exit stable-temperature
handoffs separate experiments without changing a measured phase deadline or
hiding the thermal cost of an active workload.

Server exit is not the end of host containment. ROCm mixed-load, ROCm/Vulkan
development and endurance soaks, and the Vulkan resident-prefill oracle wait for
eight consecutive 250 ms `k10temp/Tctl` samples at or below 75,000
millicelsius, with a 180-second bound. The receipt publishes
`host_thermal_cooldown_active_end`, `completed_count`, `timeout_count`, seconds,
sample count, stable-sample count, and post-exit peak. Qualification requires
exactly one completed cooldown and no active or timed-out cooldown. The final
`host_temperature_end_millicelsius` is therefore a post-cooldown observation,
while `host_temperature_peak_millicelsius` includes any residual heat-soak
spike after the process exits.

When measurement starts but a later wave fails, the case result retains request,
latency, cancellation, memory, allocator, cache, and resident-route evidence
through the last fully completed and drained wave. An in-progress wave is never
counted as complete. `measurement_final_snapshot_complete=0` distinguishes that
partial evidence from a normal final drain; only a run that obtains and validates
the final health/debug/memory snapshots publishes `1`. This flag is diagnostic,
not a way for a failed case to satisfy an acceptance threshold.

The receipts below predate the hard-limit-only contract. They retain evidence
for the retired process-stop experiment and its chronology, but they cannot
satisfy the current source-bound serving gate even when their historical
verdict was pass.

The clean `1ea855a51` Strix Halo run disproved the former boundary-only policy
and, at the time, motivated a continuous process-stop controller. Six stabilization cycles completed 30 exact responses plus
three cancellations and converged to two consecutive cycles with zero DRM
growth, live-buffer growth, allocations, frees, pool misses, evictions, or
uncached allocations. Measurement began after 1,085.45 setup seconds. Its first
six waves then completed 15 exact responses plus one cancellation over 458.77
seconds with process DRM fixed at 50,001,174,528 bytes and every post-baseline
allocator counter still zero. The next 96-word singleton began below the pacing
threshold but drove the package to exactly 97,000 millicelsius before returning
to a harness boundary. The independent guard stopped the server; boundary
pacing correctly reported zero events. The retained failed receipt is
`qualification/receipts/vulkan/strix-halo/20260716t092911388875z-vulkan-strix-halo-serving-vulkan-developme-b5eb848d54-v1.json`.
It is strictly source/artifact/commit-valid, records at least 16,861,724,672
bytes available, 103,718,912 bytes of swap growth, clean unforced teardown, no
worker or snapshot residue, and no device or batching fault. Raising the 97 C
limit, deleting the 96-word supported prompt, or rerunning the boundary-only
policy would not close that historical gate. At that point the continuous
process-group controller still had to pass a clean pushed-source run before it
could support the then-current Vulkan qualification claim.

That historical gate passed on the clean pushed `e79d3686d` source. The retained receipt is
`qualification/receipts/vulkan/strix-halo/20260716t154944163408z-vulkan-strix-halo-serving-vulkan-developme-b5eb848d54-v1.json`.
It is strictly current-source, local-artifact, and known-commit valid. Six setup
cycles completed 30 exact responses and three cancellations, covered every
prompt cohort, and converged after 1,212.42 seconds. Cycles 5 and 6 had zero DRM
growth, live-buffer growth, allocations, frees, recycler misses, evictions, or
uncached allocations; the final setup RSS delta was 401,408 bytes. Stabilization
also proved 128 resident forwards over 256 row-forwards, eight completed rows,
maximum width two, zero active rows at drain, and no decline or route failure.

The fresh measured phase then ran for 1,804.61 seconds and completed 51 exact
responses, 816 completion tokens, five cancellations, and 21 fully drained
waves. Process DRM began and ended at exactly 50,840,551,424 bytes; its measured
peak was only 98,304 bytes higher. Live Vulkan ownership and the 3.5 GiB recycler
retention were byte-identical at baseline and final drain, with zero measured
allocations, frees, cache misses, evictions, or uncached allocations and 686,925
cache hits. RSS grew 11,317,248 bytes. The resident route made 992 forwards over
1,984 row-forwards, completed 20 rows at width two, and ended with zero active
rows, initial declines, or route failures. There were zero request, batching,
device, non-finite-response, worker-residue, synchronization, shutdown, or
snapshot failures, and every KV/cache ownership gauge drained to its declared
state.

The continuous controller started and completed exactly 123 pauses, totaling
72.13 seconds; its longest pause was 1.001 seconds and its hottest starting
sample was 89,625 millicelsius. No pause remained active and the unchanged 97 C
guard did not trip. All 240 ITL outliers had bounded attribution and none was
unexplained. Host availability stayed at or above 17,548,128,256 bytes and swap
grew 341,397,504 bytes, inside their unchanged limits. Shutdown was unforced and
zero, the private snapshot was removed, and no process survived. The case result
hash is `sha256:bfc5defbb8889d2ebe73c0f5890fc6d0a0f378ed65487c0bd60245541f9bddbe`;
the receipt file hash is
`sha256:d7e4459d774e86dc6e560c834c8f4d847a1eb33ff7965dcd6b810d8274ba82ea`.

This passed the retired 30-minute Vulkan development-soak contract for the exact
named machine and declared four-active profile. Its 123 active-work pauses mean
it does not pass the current hard-limit-only contract, which requires a fresh
clean run with zero pacing events. It does not establish eight-hour
endurance, stable-profile resident admission, broader prompt/concurrency
coverage, CUDA or Metal parity, or throughput competitiveness with vLLM. In
particular, measured p99 TTFT was 150,147.60 ms; this is acceptable for the
current containment/correctness gate but remains an explicit performance
backlog item rather than a production latency claim.

The current hard-limit-only contract passed again from exact clean pushed source
`fb5cb029d9fefc13796b9ecdf928062d62445f78`; retain its strict receipt at
`qualification/receipts/vulkan/strix-halo/20260720t105341024462z-vulkan-strix-halo-serving-vulkan-developme-b5eb848d54-v1.json`.
Six stabilization cycles completed 30 exact responses and ended with two flat
allocator/DRM boundaries. Measurement then completed 55 exact responses, 880
tokens, 22 waves, and five cancellations over 1,935.219 seconds at 0.455
aggregate output tokens/second. P50/p99/p99.9 ITL was
76.721/1,768.377/1,798.398 ms; all 228 outliers were attributed and none were
unexplained. P50/p99 TTFT was 83.194/150.588 seconds, which remains an explicit
performance failure relative to the product's competitive ambitions even
though the development safety gate passed.

Measured process DRM began and ended at exactly 53,687,603,200 bytes with only
a 98,304-byte peak delta. Live Vulkan ownership and 3,527,540,736 retained pool
bytes were byte-identical at the boundaries. There were zero measured
allocations, frees, cache misses, evictions, or uncached allocations and
917,184 cache hits. RSS grew 1,224,704 bytes, host availability stayed above
20,761,858,048 bytes, and swap did not grow. The resident path completed 1,606
forwards over 3,212 row-forwards at width two with zero decline or failure and
drained all active rows. The batched-state cache recorded 2,668 exact reuses and
ended with its reusable entry resident but no active lease. Request, batching,
device, non-finite, synchronization, KV/cache ownership, worker, shutdown, and
snapshot failures were all zero. Hard-limit-only containment produced no pacing
event or guard trip, package temperature peaked at 89.75 C, cooldown completed,
and shutdown was unforced and zero. This flat allocator trace does not support
mid-inference VRAM rebalancing as the cause of the long request phases. It
qualifies the current 30-minute development gate, not final eight-hour endurance
or vLLM parity.

At the drained warmup baseline and after each of the at most eight Vulkan
stabilization cycles, the driver also reads `/proc/<pid>/smaps`. This is bounded
diagnostic work, not a hot-loop sampler. Every mapping must contain Linux's
`Size`, `Rss`, `Pss`, `Anonymous`, `AnonHugePages`, `Private_Dirty`, and `Swap`
fields or the run fails closed. `AnonHugePages` must not exceed `Anonymous`, and
the remaining page-accounting fields are checked against mapping size and RSS.
Mappings are assigned to a fixed, low-cardinality set:

- `anonymous`: unnamed and `[anon:*]` mappings;
- `heap`: the process `[heap]` mapping;
- `stack`: main and named thread stacks;
- `shared_memory`: `/dev/shm`, `memfd`, and System V shared-memory mappings;
- `device`: other `/dev/*` mappings, including DRM render nodes;
- `file`: ordinary file-backed mappings, including model shards and libraries;
- `kernel`: kernel pseudo-mappings such as `[vdso]` and `[vvar]`.

Each cycle trace records signed RSS deltas for every category, total PSS,
anonymous-page, anonymous-huge-page, private-dirty, and swap deltas, plus the
eight largest positive mapping-level RSS changes. Each retained mapping row
includes its current anonymous-huge-page bytes. A mapping identity includes its
path (or `[anonymous]`) and virtual address range, so a fixed mapping becoming
resident can be distinguished from a newly mapped object. The bounded case-result
metrics retain start/end `smaps` RSS, positive growth by category, and positive
anonymous, anonymous-huge-page, private-dirty, and swap growth across the
complete observed stabilization-through-measurement window. The exact metric
for huge-page growth is
`vulkan_process_smaps_anonymous_huge_pages_growth_bytes`. These diagnostics
attribute a safety failure; they do not weaken, replace, or exempt it from the
RSS gate.

Vulkan stabilization alternates concurrency one and four with 16-token outputs
and a cancellation every fourth wave. Each pair uses one shared prompt-length
slot, rotating through 16/32/64/96 words; the source-bound configuration
enforces decode width two, two staged prefill rows, and four total active
requests. Stabilization must observe a multirow resident prefill before the
baseline, which also forces batch-dependent scratch growth into the setup
phase. The candidate explicitly raises the idle Vulkan scratch pool from the
3.0 GiB product default to 3.5 GiB; health must attest the exact byte cap. The
additional 512 MiB is bounded by the same host-memory, swap, process-DRM, and
thermal guards. Client concurrency above four may wait outside the active set, but this
workload makes no latency or throughput claim for that queue or for prompts
beyond the declared slots. The retained width-eight/384-word counterexamples
remain performance evidence and the separate vLLM comparison campaign remains
open. Cross-request prefix
reuse is correctness-quarantined on this backend, so warmup does not try to
fill the inert cache. Instead, the driver requires
`prefix_cache_enabled=false` and zero lookups, hits, misses, retained blocks,
retained entries, state bytes, leases, and pending releases at initial warmup,
every stabilization and measured drain, and final drain. It then requires two
consecutive cycles with no live `VulkanBuffer` ownership growth, allocation,
free, buffer-pool cache miss, eviction, or uncached allocation, plus at most
64 MiB of process-scoped DRM growth. Net-zero bytes cannot hide allocator
churn. It cannot accept before four
cycles and fails after eight. RSS is a
separate cumulative host-safety signal on unified memory: stabilization fails
if process RSS grows by more than 512 MiB from its baseline, while the host
guard independently enforces the 8 GiB `MemAvailable` floor and 512 MiB swap
growth ceiling. This distinction is intentional because an already-live fixed
Vulkan mapping can become resident without a new buffer or DRM allocation.
The measured phase runs the same fixed-slot prompt set for at least 30 minutes.
At every drained wave boundary, process DRM and RSS may grow by at most 512 MiB
from the post-stabilization baseline. The 250 ms DRM sampler separately permits
at most 1 GiB of active-workload growth, bounding concurrent scratch without
misclassifying buffers that are allocated during a wave and fully freed at
drain as retained growth. The receipt records both peak bytes and peak growth,
plus the exact `VulkanBuffer` live high-water mark. Pool retention and total live
buffer ownership must return to their measurement baselines. The same response,
drain, cache-ownership, pause, device-fault, shutdown, and residue gates apply
as in the ROCm arm. Vulkan graphs must remain disabled: any graph capture,
replay, slot, or fallback activity fails.

The allocation counter must also remain unchanged after stabilization. Prefix
cache activity must remain zero, and even one new `VulkanBuffer` allocation
fails the next measured wave. This catches allocator churn that has balanced
ownership at drain yet still creates driver-side RSS growth or inference
pauses.

#### Vulkan buffer ownership telemetry

A Vulkan build adds `vulkan_buffers` to `GET /health`. The object is omitted on
other builds and contains these process-lifetime counters:

| Field group | Meaning |
|---|---|
| `live_device_local_buffers`, `live_host_visible_buffers` | Number of live `VulkanBuffer` allocations by constructor intent. |
| `live_device_local_bytes`, `live_host_visible_bytes` | Vulkan memory-requirement bytes still bound to those live buffers. |
| `peak_live_bytes` | Process-lifetime high-water mark across both memory kinds. |
| `*_allocations`, `*_allocated_bytes` | Successful allocation count and bytes since process start. |
| `*_frees`, `*_freed_bytes` | Destructions and bytes freed since process start. |

The byte values use Vulkan's allocation requirement, which can exceed the
requested logical buffer size because of alignment. `device_local` and
`host_visible` describe the allocation route; on a unified-memory device they
do not imply physically separate VRAM and system-RAM chips. These counters
cover memory owned through `VulkanBuffer`. Driver-private allocations and
memory not created by that wrapper remain visible only through DRM, RSS, swap,
and host-availability evidence.

The same response adds `vulkan_buffer_pool`, which separates recycler
retention from all other live buffers:

| Field group | Meaning |
|---|---|
| `max_retained_bytes` | Immutable typed cap derived from `memory.vulkan_buffer_pool_gb`. |
| `bucket_count`, `buffer_count`, `retained_bytes` | Current exact cache inventory. |
| `free_*`, `borrowed_*` | Idle versus caller-owned portions; each pair must reconcile to the inventory total. |
| `cache_hits`, `cache_misses` | Process-lifetime recycler lookup outcomes. |
| `device_local_cache_misses`, `host_visible_cache_misses` | Process-lifetime miss attribution by allocation route. Their sum must equal `cache_misses`. |
| `last_cache_miss` | One bounded diagnostic record, or `null` before the first miss. A record contains the miss `sequence`, allocation `route`, requested and bucket-rounded bytes, and the Rust caller file and line. Its sequence equals `cache_misses`. |
| `eviction_count`, `evicted_bytes` | Idle entries released to admit a newer working set or satisfy pressure reclaim. |
| `uncached_allocation_count`, `uncached_allocated_bytes` | Overflow scratch allocations returned without a cache owner and freed on normal drop. |

Retained bytes may never exceed the cap. Eviction is oldest-idle-first and
never removes a borrowed buffer. Pressure reclaim runs only while the batching
actor is idle, after exclusive GPU coordination and a second health/activity
check. `GET /v1/config` exposes the pool limit, inventory, free and borrowed
bytes, both lookup outcomes, both route-specific miss counters, eviction totals,
and uncached overflow totals. It deliberately omits the source-level last-miss
record; use `GET /health` for that live diagnostic.

Device-local lookup is exact-bucket because legacy device-buffer consumers can
still use physical buffer size as a copy extent. Host-visible staging carries
an explicit logical extent, so its lookup first checks the exact bucket, then
selects the smallest sufficient idle larger bucket for the same Vulkan device
and host-memory type, with oldest-use order breaking ties. An undersized or
currently borrowed slot can never satisfy a request. Consequently, a
host-visible `cache_hit` may reuse a larger bucket; that is expected and does
not increase retained ownership or allocation count.

Prometheus exports the same state as:

```text
kiln_vulkan_buffer_live_buffers{memory="device_local|host_visible"}
kiln_vulkan_buffer_live_bytes{memory="device_local|host_visible"}
kiln_vulkan_buffer_peak_live_bytes
kiln_vulkan_buffer_allocations_total{memory="device_local|host_visible"}
kiln_vulkan_buffer_allocated_bytes_total{memory="device_local|host_visible"}
kiln_vulkan_buffer_frees_total{memory="device_local|host_visible"}
kiln_vulkan_buffer_freed_bytes_total{memory="device_local|host_visible"}
kiln_vulkan_buffer_pool_limit_bytes
kiln_vulkan_buffer_pool_bytes{state="retained|free|borrowed"}
kiln_vulkan_buffer_pool_buffers{state="retained|free|borrowed"}
kiln_vulkan_buffer_pool_buckets
kiln_vulkan_buffer_pool_requests_total{result="hit|miss"}
kiln_vulkan_buffer_pool_misses_total{route="device_local|host_visible"}
kiln_vulkan_buffer_pool_evictions_total
kiln_vulkan_buffer_pool_evicted_bytes_total
kiln_vulkan_buffer_pool_uncached_allocations_total
kiln_vulkan_buffer_pool_uncached_allocated_bytes_total
```

#### Resumable GDN prefill residency telemetry

Vulkan prompt-prefill residency is currently correctness-quarantined. Typed
runtime policy prevents production requests from entering its request/layer
scope, so ordinary prompt chunks materialize recurrent state before yielding.
The direct-registry telemetry remains a closed contract for detecting accidental
activation, retaining failed-experiment evidence, and qualifying a future
repair. It is distinct from the persistent batched decode-state cache described
below. Trusted `GET /v1/debug/model-state` exposes the complete current direct
registry at `caches.resident_recurrent_state`:

The clean `7dcad0d95` no-override qualification probe is the current quarantine
acceptance anchor. With the checked profile's historical decode opt-in intact,
the 138-token q128 request emitted the exact required 32-token ascending prefix,
reported no semantic failure, and left no direct prompt-registry ownership or
process residue. Its 31.34-second prefill does not satisfy a throughput or soak
gate; those remain separate.

| Field | Meaning |
|---|---|
| `entry_count` | Current logical recurrent-state slots mapped to backend-private buffers. Resumable prefill owns a slot by request ID plus linear-layer index; the aggregate deliberately does not expose either value. |
| `buffer_bytes` | Sum of addressable Vulkan buffer extents for those entries. |
| `allocation_bytes` | Sum of Vulkan allocation requirements, including alignment beyond the addressable extent. This must be at least `buffer_bytes`. |

The object is deliberately aggregate and bounded: it contains no request ID,
prompt data, tensor ID, or per-layer label. Internally, a secondary tensor alias
supports decode and handle-local lookup, but it is not the resumable-prefill
owner. All three values must remain zero throughout current production prompt
execution and after successful decode handoff, cancellation, generation error,
actor discard, and final server drain. A future explicitly test-enabled arm may
observe nonzero values only while a prefill quantum is live. Its cleanup must be
scoped to the stable request owner; clearing the whole registry would be an
invalid implementation because it could corrupt concurrent rows.

No environment variable can enable the quarantined prompt or decode scope;
`kiln.vulkan-kernel-policy.v3` fixes both off. A fully materialized control
requires a reviewed source-policy change and separately attested binary. A
control that merely reports `resident_prefill_used=false` is
insufficient: that field describes the native multi-row prefill route, not this
direct GDN registry. A valid control additionally observes zero direct-registry
ownership and records no direct resident GDN use. Each source-policy A/B arm
requires a fresh server process and its own source/binary identity.

Any future device-resident repair must preserve the ordinary external-dtype boundary. For a
BF16 recurrent tensor, every completed nonfinal token chunk performs a
device-side F32 -> BF16 -> F32 round-to-nearest-even conversion before the row
can resume; an F32 tensor is unchanged. Unsupported external dtypes materialize
at the boundary. Layer-group yields within a token chunk must not add rounding,
and final materialization supplies the final chunk's handoff conversion. This
rule is independent of registry counts: all gauges can drain correctly while a
missing precision boundary still corrupts subsequent decode.

An empty registry with nonzero bytes, a nonempty registry with zero buffer
bytes, or allocation bytes below buffer bytes is internally inconsistent and
fails qualification.

Prometheus exports the same state:

```text
kiln_gdn_recurrent_state_resident_entries
kiln_gdn_recurrent_state_resident_bytes{kind="buffer|allocation"}
```

The Vulkan development-soak driver treats the debug object as a closed
contract. It validates exact field names and nonnegative integer types at
startup, after steady warmup, at the stabilization baseline, after every
stabilization wave and cancellation, at the measurement baseline, after every
measured wave and cancellation, and at final drain. Any nonzero drained value
fails immediately with the phase and all three ownership values. The checked-in
workload retains final values as
`resident_recurrent_state_entries_end`,
`resident_recurrent_state_buffer_bytes_end`, and
`resident_recurrent_state_allocation_bytes_end`; each has a required-zero
acceptance threshold.

The quarantined implementation's real-device parity regressions cover two prompt chunks,
multiple unrelated work-handle identities, reuse from an intentionally stale
zero-valued host handle on a different thread, stable-key materialization into
the caller's chosen handle, and a zero registry afterward. The BF16 arm compares
both outputs and final state with a host-materialized oracle that explicitly
quantizes between chunks; a separate F32 arm compares split execution with one
monolithic chunk. A second two-owner regression proves that evicting one
interrupted row preserves the other row's state. The serving semantic oracle
and cancellation workload remain required because kernel parity alone cannot
prove actor teardown or decode handoff. In particular, a cancellation probe
must first observe a nonzero registry during prefill, abort before a semantic
token is delivered, wait for the actor to drain, and then require both trusted
debug and Prometheus ownership to return to zero before issuing a follow-up
request in the same process. These focused tests pass bit-for-bit today, but
the clean full-model q128 prompt-resident arm failed the semantic oracle while
the fully materialized control passed. They therefore cannot authorize route
activation without the full-model gate.

#### Batched recurrent-state cache telemetry

Native batched decode also owns a persistent `LinearAttentionState` cache for
the recurrent GDN layers. Trusted `GET /v1/debug/model-state` exposes its full
snapshot at `caches.batched_recurrent_state` on every backend. The four current
ownership fields are `entry_present`, `capacity_rows`, `logical_rows`, and
`resident`. A parked resident entry may have more capacity rows than logical
rows because smaller batches use an identity-preserving prefix view of the
maximum observed allocation. An entry is temporarily absent while a forward
call owns its lease.

The remaining fields are process-lifetime monotonic counters:

| Field group | Meaning |
|---|---|
| `active_leases`, `max_active_leases` | Current and peak simultaneous checked-out or newly assembled states. More than one proves overlapping batched-state forwards. |
| `take_hit_count`, `take_miss_count` | Cache checkout outcomes. The first eligible forward normally misses. |
| `take_miss_while_leased_count` | Misses observed while another state lease is active. This is the direct signal for concurrent checkout of a single-slot cache. |
| `exact_reuse_count` | Reuse with the same ordered request-ID fingerprint; no state-row refresh is needed. |
| `resident_capacity_reuse_count` | Reuse of backend-resident allocation capacity. This includes same-width and smaller-prefix reuse. |
| `resident_prefix_view_count` | Capacity reuse through a smaller logical axis-0 prefix. |
| `resident_refresh_count` | In-place row refresh because the ordered request IDs changed. |
| `fresh_assembly_count` | New batched state assembled after a miss or rejected entry. |
| `rejected_*_count` | A checked-out entry could not be reused because row IDs were absent, input rows were nonresident, the cache was nonresident, or capacity was insufficient. Exactly one rejection reason is recorded for each rejected entry. |
| `park_count` | A forward returned its state to the persistent cache. |
| `park_replacement_eviction_count` | A returning lease found another parked entry and evicted it. This should remain zero when one cache owner cannot overlap another. |
| `explicit_invalidation_count`, `explicit_invalidation_eviction_count` | Adapter/model lifecycle invalidations requested, and those that actually removed a parked entry. |
| `completed_row_preservation_count`, `completed_row_eviction_count` | Completed request rows that appeared in the cache fingerprint and caused resident preservation or nonresident eviction. |
| `lease_drop_eviction_count` | Checked-out capacity released instead of parked, including rejected entries and forward/error exits. Use the rejection counters and request failures to distinguish expected capacity replacement from errors. |
| `resident_prefix_snapshot_suppression_count` | Whole-prompt, strict-prefix split, or rolling prefix snapshots rejected because backend-resident recurrent/conv state, rather than the logical tensors, was authoritative. This is a correctness guard, not a cache or request failure. |

Prometheus exports the same bounded-cardinality state as:

```text
kiln_batched_recurrent_state_cache_entry
kiln_batched_recurrent_state_cache_rows{kind="capacity|logical"}
kiln_batched_recurrent_state_cache_resident
kiln_batched_recurrent_state_cache_leases{kind="active|max"}
kiln_batched_recurrent_state_cache_takes_total{result="hit|miss"}
kiln_batched_recurrent_state_cache_misses_while_leased_total
kiln_batched_recurrent_state_cache_reuses_total{kind="exact|resident_capacity|prefix_view|refresh"}
kiln_batched_recurrent_state_cache_assemblies_total
kiln_batched_recurrent_state_cache_rejections_total{reason="missing_row_ids|nonresident_rows|nonresident_cache|insufficient_capacity"}
kiln_batched_recurrent_state_cache_parks_total
kiln_batched_recurrent_state_cache_invalidations_total
kiln_batched_recurrent_state_cache_completed_rows_total{action="preserve|evict"}
kiln_batched_recurrent_state_cache_evictions_total{reason="park_replacement|explicit_invalidation|completed_row|lease_drop"}
kiln_prefix_cache_snapshot_suppressions_total
```

The reuse counters deliberately describe overlapping properties: an exact
fingerprint can also reuse resident capacity and a prefix view. Rejection
reasons, by contrast, are mutually exclusive. In a warmed, serialized,
fixed-maximum-capacity workload, fresh assemblies and insufficient-capacity
rejections stop increasing after the largest batch is seen;
`take_miss_while_leased_count`, `park_replacement_eviction_count`, and
`max_active_leases - 1` remain zero. Growth in those three concurrency signals
alongside flat semantic/device error counters identifies ownership overlap,
not allocator pressure. Increasing `rejected_insufficient_capacity_count`
without overlap identifies legitimate high-water growth. Increasing
`lease_drop_eviction_count` without a rejection requires an accompanying
forward/error investigation.

Vulkan full-attention KV seed flags and recurrent GDN state have different
lifetimes. The former are indexed by layer and reset when an unidentified
single-request decode crosses a start-position session boundary, so a new
request cannot read the prior request's KV contents. GDN buffers are indexed
by tensor ID and remain owned by that request row or the batched-state cache;
a session boundary must not clear their initialization markers. Only explicit
eviction of the same tensor ID may remove its recurrent buffer, convolution
buffer, and initialization marker. A rising
`rejected_nonresident_cache_count` with live buffers still present indicates a
violation of this lifetime split, not buffer-pool eviction.

Prefix-cache snapshots have a related authority boundary. Generic execution
does not by itself prove that logical GDN tensors remain authoritative: an
accelerated prefill kernel may advance backend-private recurrent and
convolution buffers without updating those logical tensors. Native resident
decode can additionally advance backend-private KV without writing generic
decoded-token KV positions. Copying the logical tensors in either state would
publish an internally inconsistent cache entry. Kiln therefore puts every
whole-prompt, strict-prefix split, and rolling snapshot through the same
authority gate. It captures only while no layer reports backend-resident GDN
authority; otherwise it omits the cache registration. This may deliberately
forgo a prefix-cache hit, but never changes the live request state and never
introduces a hidden device readback on the prefill or decode hot path.

`resident_prefix_snapshot_suppression_count` and
`kiln_prefix_cache_snapshot_suppressions_total` prove that guard fired. The
counter is expected to rise only when an admitted cache encounters an otherwise
eligible capture under native resident decode authority. Vulkan still
quarantines the cross-request prefix cache. Stable and maintenance profiles
also quarantine resident token-prefill and require its advertised capability
and every activity counter to remain zero. The experimental profile admits
resident token-prefill after backend and request eligibility checks; its soak
must instead prove at least one multi-row forward, positive row and completion
counts, zero active rows at drain, and zero route failures or initial declines.
Captures remain subject to the same authority gate. The suppression counter
must be monotonic; a zero delta is meaningful only when the workload also proves
that an eligible cache capture and resident execution occurred. Qualification
retains its delta but does not treat a positive value as an error on an admitted
backend.

The Vulkan development soak treats this snapshot as a closed qualification
contract. It validates the exact field set and types at startup, after warmup,
after every drained stabilization cycle, after every measured wave, and at
final drain. Per-cycle traces include the current ownership gauges and the
delta of every lifecycle counter. The case result retains the complete
observed stabilization-through-measurement delta, including runs that fail
before measurement. A drained wave fails if it leaves an active lease; after
stabilization, any new miss while leased or park-replacement eviction fails
immediately. The final verdict also requires one resident parked entry, no
active lease, a peak of at most one active lease, and zero completed-row,
explicit-invalidation, replacement, or lease-overlap eviction in the observed
window. These gates distinguish cache ownership overlap from buffer-pool
pressure without inferring either from RSS alone.

The soak validates that allocation/free deltas reconcile exactly with the
change in live count and bytes, that pool free plus borrowed ownership equals
retention, that the cap never changes or overflows, and that every cache
counter remains monotonic. Initial stabilization cycles may fill the
fixed-shape recycler working set. Two later cycles must show no positive live
ownership growth before measurement begins; any measurement-phase live or pool
retention growth fails the run. Allocation and free deltas that match while
live bytes stay flat indicate temporary buffer churn; if RSS continues to grow
within the cumulative host limit, investigate driver or unified-memory page
residency rather than mislabeling it as retained Rust ownership. Flat
allocation/free counters and flat live bytes with growing RSS instead point to
pages becoming resident in already-live allocations. The stabilization and
measured-wave traces record total and route-specific pool miss deltas. They
include the single `vulkan_pool_last_cache_miss` record only for an interval
that made at least one miss, so an unexpected warmed-path allocation identifies
its final request size, pool bucket, route, and source callsite without
unbounded event capture. The stabilization trace also records buffer, pool,
DRM, RSS, swap, fixed-category `smaps`, anonymous huge-page, and bounded
per-mapping deltas per cycle. A run that fails
before measurement retains the observed stabilization window in the Vulkan
allocation, pool, cache-lifecycle, and mapping metrics instead of replacing it
with zeroes.
The cumulative stabilization RSS gate runs only after the completed cycle's
health, debug snapshot, lifecycle deltas, and memory deltas have been traced
and stored, so the cycle that crosses the safety limit is not lost from the
diagnostic evidence.

### Vulkan Final Endurance Soak

Run the final Strix Halo endurance gate from a clean, pushed source tree after
the 30-minute Vulkan development soak passes:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan-endurance \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-vulkan-endurance-v1.json
```

This is a distinct source-bound qualification identity, not a duration override
on a development receipt. It deliberately retains the development soak's exact
four-active-request service envelope, one/four-way prompt cohorts, periodic
cancellation, semantic oracle, fixed KV and Vulkan recycler policy, resident
route gates, 8 GiB host-memory floor, 512 MiB swap-growth cap, continuous
hard-limit-only 97 C host protection, and zero permitted pacing events. The different
deterministic seed and eight-hour duration exercise a longer request sequence
without silently widening the already-qualified operating point.

The timing contract gives the exact source build its checked 900-second limit,
then gives startup, warmup, and stabilization a fresh 1,800 seconds. Successful
stabilization starts a fresh 29,400-second measurement deadline: 28,800 required
seconds plus the unchanged 600-second request containment window. The outer
case timeout is 32,280 seconds, exactly those limits plus a separate 180 seconds
for cancellation, process-group shutdown, private-snapshot cleanup, and result
publication. Thermal protection never suspends active work or extends runtime
setup and measurement deadlines. A
setup failure still reports zero measured duration, and a measurement failure
retains evidence only through the last fully completed and drained wave with
`measurement_final_snapshot_complete=0`.

A pass requires the same exact response, device, ownership, memory, thermal,
worker, and lifecycle verdicts as the development soak across the full measured
window. In particular, hardware absence or a Vulkan skip fails the required
case; every ITL outlier must have a bounded known attribution; thermal pacing
must remain identically zero; final health, debug, allocator, cache, process-memory,
and DRM snapshots must complete; and teardown must leave no server, request
worker, or private snapshot. This endurance result qualifies only this named
host and declared experimental operating point. It does not establish a
high-concurrency performance claim, stable-profile resident admission, or
portability to CUDA, Metal, or another Vulkan machine. Historical endurance
receipts that contain process-stop pacing remain diagnostic records and do not
satisfy this current contract.

The clean pushed `3897239fe` source passed the retired process-stop version of
this contract on the named Strix Halo. The retained receipt is
`qualification/receipts/vulkan/strix-halo/20260716t165320275412z-vulkan-strix-halo-serving-vulkan-endurance-7db5d986fd-v1.json`
with file hash
`sha256:118274f07578024cd1a65af2342a388f8be66dee636853d6e0d99698575ce604`.
Before any evidence documentation changed the tree, strict current-source,
local-artifact, and known-commit validation passed. The required case selected
`AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, reported no unsupported arm, and
left no server, qualification worker, compiler, or transient build unit.

Setup completed six cycles, 30 exact responses, and three cancellations in
1,099.02 seconds. The first four cycles established the fixed working set;
cycles five and six had zero DRM or live-buffer growth, allocation, free,
recycler miss, eviction, or uncached allocation. The final stabilization RSS
delta was 282,624 bytes and the second flat cycle admitted measurement. Setup
also exercised 520 resident forwards over 1,040 rows, completed ten rows at
width two, and ended with no active row, initial decline, or route failure.

Measurement ran for 28,867.72 seconds and completed 820 exact responses,
13,120 completion tokens, 82 confirmed cancellations, and 328 fully drained
waves. Process DRM began and ended at exactly 51,510,386,688 bytes, with only a
98,304-byte sampled active peak. Live Vulkan ownership began and ended at
51,492,696,448 bytes; the recycler retained exactly 3,574,202,368 bytes at both
boundaries. Across the measured window there were zero Vulkan allocations,
frees, cache misses, evictions, or uncached allocations and 11,366,760 recycler
hits. RSS grew 21,610,496 bytes from a 398,393,344-byte baseline and peaked at
468,049,920 bytes. The resident route made 18,512 forwards over 37,024 rows,
completed 320 rows at width two, and drained with no decline, failure, or active
row.

The external controller completed all 1,063 pacing events and left none active.
Cooling consumed 758.36 seconds of the unchanged deadlines; the longest pause
was 1.502 seconds, the highest pacing start and sampled package peak were
90,125 millicelsius, and the 97 C guard never tripped. All 3,907 ITL outliers
were attributed and none was unexplained. Host availability stayed at or above
17,928,245,248 bytes, host swap growth was 24,604,672 bytes inside the 512 MiB
cap, and the server's `smaps` swap growth was zero. Request, batching, device,
non-finite-response, synchronization, graph, ownership, worker, shutdown, and
snapshot failures were all zero; final snapshots completed, shutdown was
unforced and zero, and the private snapshot was removed.

This historical result established eight-hour endurance for the then-declared
four-active-request operating point on this named host. Its 1,063 pacing events
mean it does not satisfy the current hard-limit-only contract; that gate
requires a fresh zero-pacing run. The historical result also does not make the
point fast: aggregate measured completion goodput was 0.454 tokens/second,
p99 TTFT was 152,290.00 ms, and p99 ITL was 2,477.96 ms, including required
cooling time. Those values remain explicit performance limitations and do not
support a vLLM competitiveness, production-latency, higher-concurrency, broader
prompt, stable-profile, or cross-machine claim.

Never edit a receipt to make it pass. A failed receipt is useful evidence: keep
it when it identifies a reproducible product defect, fix the defect in a new
commit, and run a new receipt with a new ID. When a structurally valid command
result reports an effective configuration that differs from the selected
variant, the runner fails the case and clears the receipt-level
`effective_config` attestation, but retains the command's metrics, tolerances,
and details as counterexample evidence. Those measurements are diagnostic only;
they cannot support an accepted performance comparison because their effective
configuration was not verified.

## Validate The Result

First validate the portable schema and internal hashes:

```bash
python3 scripts/qualification/receipt.py \
  qualification/receipts/rocm/strix-halo/<receipt>.json
```

On the originating machine, require both the current committed source and the
ignored raw artifacts to match:

```bash
python3 scripts/qualification/receipt.py \
  --require-current-source \
  --require-local-artifacts \
  qualification/receipts/rocm/strix-halo/<receipt>.json
```

Review the compact verdict, skips, failures, exact effective configuration,
model/workload/source hashes, device identity, and unexplained-outlier count.
Inspect `.qualification/runs/<receipt-id>/` only for diagnosis; do not add raw
logs, traces, profiles, model output, or model weights to Git.

Compare only receipts accepted by the declared workload comparison policy:

```bash
python3 scripts/qualification/compare_receipts.py \
  qualification/receipts/rocm/strix-halo/<baseline>.json \
  qualification/receipts/rocm/strix-halo/<candidate>.json
```

The comparison command deliberately rejects mismatched source trees, models,
workloads, or undeclared configuration differences.

## Check In Evidence

Add only the compact receipt and the documentation or plan entry that explains
what it proves. Re-run portable validation on the staged receipt, inspect the
staged diff, then commit and push immediately so another machine can continue
from the same evidence chain.

```bash
git add qualification/receipts/<backend>/<host-id>/<receipt>.json \
  docs/plans/confidence-hardening-goal.md
python3 scripts/qualification/receipt.py \
  qualification/receipts/<backend>/<host-id>/<receipt>.json
git diff --cached --check
git diff --cached
git commit -m "Record <backend> <workload> qualification"
git push origin main
```

Before moving to another machine, verify `git status --short` is empty and
`git rev-parse HEAD` equals `git rev-parse origin/main`. Final cross-platform
claims require every relevant receipt to name one common source-tree hash.
