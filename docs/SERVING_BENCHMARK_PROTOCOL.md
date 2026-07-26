# Serving Benchmark Protocol

Kiln and vLLM serving comparisons use one fail-closed OpenAI-compatible driver:
`scripts/bench-concurrent-batch.py`. The driver sends only streaming
`POST /v1/chat/completions` requests on the measured path. Kiln health requests
are untimed side evidence. A passing receipt binds the source tree, runtime,
model content, workload, memory source, process group, host thermal policy, and
complete lifecycle.

## Server Ownership

Every potentially passing measured run requires one thermal boundary and
exactly one server ownership mode. Native Linux uses `--host-thermal-policy`.
WSL2 may instead use `--external-wsl2-thermal-policy` only from inside the
qualification runner's already active Windows/NVML supervisor, private
namespace, and exact user scope:

- `--server-launch-config PATH` is the default for local qualification. The
  driver fingerprints the model in its own guarded child lifecycle, fingerprints
  source and runtime, then requires stable host cooldown before server process
  creation. It launches a new process group, arms thermal containment before the
  first readiness request, requires the loopback port to be unbound before
  launch, waits for `/v1/models`, proves the listening socket belongs to that
  process group, runs the workload, verifies the live runtime, sends `SIGTERM`,
  requires the whole process group to disappear, monitors the hard limit through
  exit, completes host cooldown, repeats the model fingerprint in a second
  guarded lifecycle, and hashes the server log.
- `--server-pid PID` attaches to an already running process-group leader. The
  guard is armed before `/v1/models` or `/health`. The driver leaves the process
  alive only after the configured safe-handoff temperature is stable.
- `--unsafe-no-host-thermal-guard` exists only to retain diagnostic
  counterevidence. It can never produce a passing receipt.

Driver v21 requires an owned `--server-launch-config` with the external WSL2
mode. It revalidates the WSL2 kernel, policy hash, network/Landlock boundary,
host UID, exact `.scope` cgroup path, 10 GiB memory maximum, zero swap, 512 PID
maximum, group OOM handling, absence of delegated `cpu.max`, and the outer
50-percent usage-feedback controller before model hashing or launch. Each
measured row records wall time and sustainable throughput under that boundary.
The child receipt deliberately marks the enclosing qualification receipt as
required: only the parent owns final thermal samples, scope accounting, cleanup,
and stable handoff, so a child v20 or v21 WSL2 receipt is never sufficient by
itself.

External WSL2 owned launch uses the same retrying `/v1/models` readiness loop
as native owned launch. Its startup and shutdown grace periods count wall time
minus only complete, exact-policy intervals from the controller-authenticated
thermal-pacing stream. The driver rereads that stream only when the projected
active deadline is reached. Missing, unsafe, partial, unpaired, policy-drifted,
or arithmetically invalid evidence rejects the run. A pacing-accounting failure
during shutdown still sends `SIGKILL` and drains the process group under an
independent emergency wall bound before surfacing the failure. No controller
freeze can by itself force shutdown escalation or consume the readiness
allowance; ordinary delay and CPU-feedback freezes still consume it.

Linux `killpg(pgid, 0)` also succeeds when every remaining member is a zombie.
After the owned leader has been reaped, the driver therefore enumerates
`/proc/*/stat` and treats the group as execution-quiescent only when a complete
scan finds at least one exact-PGID member and every such member is in state
`Z`. Zombies cannot execute or retain descriptors. A live member, an empty
scan while `killpg` still succeeds, unreadable or malformed process evidence,
or a permission failure continues to mean that cleanup is incomplete. This
avoids escalating a clean shutdown solely because PID-namespace init has not
reaped an orphan while preserving fail-closed handling for executable
descendants, uninterruptible processes, and uncertain membership.

Owned launch eliminates the manual gap in which model loading or inference
prewarm could run before the guard attached. Driver v8 also removes the
model-provenance gap: both full model fingerprints run as start-gated child
process groups under the selected thermal policy and complete their own stable
post-exit cooldown. The server pre-launch cooldown then clears heat from source
and runtime hashing plus launch validation before server creation. PID attachment
remains useful for remote orchestration that already owns process creation, but
that orchestrator is responsible for containment and its starting host state
before the benchmark starts. Its server is returned after safe handoff before
the final guarded fingerprint; it must remain quiescent during that recheck.

### Model-fingerprint read pacing

Driver v12 keeps the v8 double-read integrity contract but makes its I/O rate
explicit and bounded. Each model input is opened without following symlinks,
hashed once for identity, rechecked by file and directory stat identity, and
hashed a second time to detect a concurrent same-length rewrite even where
filesystem timestamps are too coarse. One monotonic cumulative limiter covers
every byte in both passes. It accounts after each read and sleeps in at most
25 ms increments; it does not reset at a shard or pass boundary.

`--model-fingerprint-read-mib-per-second` accepts `64..=16384` and defaults to
256 for measured direct and campaign runs. The worker receives the value as an
argv field in its closed environment. Receipt schema
`kiln.serving-model-fingerprint-thermal.v2` records it as
`host_thermal.model_fingerprint.read_mib_per_second` beside the exact worker
implementation and Python hashes plus initial/final thermal lifecycles.
Campaign v7 records and forwards the same value to every profile. The setting
applies only to provenance reads; it cannot pace server startup or inference
and is outside every request-timing window. Standalone `model_fingerprint.py`
remains unlimited when its optional `--max-read-mib-per-second` is omitted.

### Server checkpoint-read and upload pacing

The driver's fingerprint policy does not govern the owned server. Named-host
ROCm and Vulkan profiles separately set
`model.checkpoint_read_mib_per_second = 256`. The server applies an independent
cumulative schedule to each private snapshot copy, initial full loader-owned
content verification, and post-upload full verification. Bounded chunks check
shutdown; reflinked bytes count as logical snapshot progress without consuming
the read budget. `GET /v1/config.model_startup.checkpoint_read` exposes all
three completed phase records, including logical and rate-limited bytes,
elapsed time, pacing time, and the startup-only invariant.

Those profiles also set `model.accelerator_weight_upload_mib_per_second = 256`.
The upload pacer reserves the next cumulative source-byte target before the base
group and each layer rather than letting a hot unit run before its wait. It
checks shutdown again after embedding upload, transpose, W8 packing, final norm,
and rotary initialization, and after every layer. The current backend operation
remains non-interruptible. API evidence binds reserved and completed bytes and
layers; the exclusive content-hashed log binds intermediate stages. Neither
policy is active after readiness or permits weakening the external thermal
guard.

Owned mode accepts only an origin-only loopback HTTP base URL. On Linux it
matches the listening TCP/TCP6 socket inode with descriptors held by the leader
or another member of the launched process group. An old listener on the port,
a readiness response from an unrelated process, or a listener that escaped the
group fails before traffic.

## Launch Configuration

The closed `kiln.serving-benchmark-server-launch.v1` document has these fields:

| Field | Contract |
|---|---|
| `schema` | Exact schema identifier. |
| `id` | Stable portable identifier, 3 to 128 characters. |
| `command` | Non-empty argv array. No shell parsing is performed. The executable must be absolute or explicitly relative such as `./target/release/kiln`. |
| `working_directory` | Child working directory. Relative paths resolve from the launch document's directory. |
| `log_directory` | Directory for exclusive `<run-id>.server.log` files. Relative paths resolve from the launch document's directory. |
| `readiness_poll_interval_ms` | Delay between failed `/v1/models` attempts, from 1 through 60,000 ms. |
| `startup_timeout_seconds` | Positive readiness deadline. Native execution counts wall time; external WSL2 execution subtracts only authenticated thermal-pacing overlap. Individual probes are bounded to two seconds and failed probes are retried. |
| `shutdown_timeout_seconds` | Positive grace period after process-group `SIGTERM` before containment uses `SIGKILL` and fails the run. Native execution counts wall time; external WSL2 execution subtracts only authenticated thermal-pacing overlap. |
| `acceptable_exit_codes` | Sorted, unique, non-empty accepted status list. A forced shutdown fails regardless of status. |

Kiln's selected NVIDIA startup identity and capacity probes use strict
`nvidia-smi` output parsing and a two-second child lifetime. Startup now allows
at most three attempts separated by 100 ms. A failed or malformed third sample
still establishes zero capacity and rejects accelerator startup; a configured
capacity remains a cap and cannot replace hardware evidence. The process-wide
memory governor independently applies the same fixed three-attempt, 100 ms
startup budget only at allocation-transition boundaries: pre-model startup,
post-upload/pre-KV sizing, initial KV admission, and the final pre-ready check.
The probe lock serializes those attempts, and a failed third sample publishes
zero free memory and an unhealthy cache. Ordinary governor refreshes, the
persistent sampler, and live runtime admission remain single-sample and
fail-closed, preserving the distinction between bounded startup recovery and
runtime counter loss.

The launch contract intentionally has no environment map and accepts no shell
command string. Runtime behavior belongs in the server's typed configuration;
secrets belong in the server's normal credential mechanism. For an owned Kiln
run, `command[0]` must resolve to the exact `--runtime-artifact` file. For vLLM,
the runtime artifact remains the immutable teacher/runtime manifest and the
command must launch `scripts/vllm_teacher.py` with
`--process-group-mode=inherited`. The teacher launcher fails unless the driver
has already made it the leader of an isolated Linux process group. Detached
vLLM children cannot satisfy listener ownership and are never thermally guarded
by implication from their parent.

The driver enforces that statement before model hashing. The second argv entry
must resolve directly to the tracked non-symlink teacher script. The launcher
options for model, served ID, process-group mode, snapshot root, cache root,
top-K bound, and model-length bound must each occur exactly once before one
explicit `--` boundary. The optional cumulative provenance-read ceiling may
occur at most once. Manifest, dry-run, and precomputed-identity modes are
forbidden. The parsed served ID, top-K bound, and model-length bound must match
the immutable runtime manifest. An arbitrary server command cannot satisfy an
owned vLLM run merely by returning a syntactically valid system fingerprint.

The vLLM wrapper owns its runtime cache through the typed `--cache-root` option.
Each real launch derives a unique empty `VLLM_CACHE_ROOT`, and removes it after
the supervised process group exits. Tracked launch documents should name an
explicit ignored local parent so the filesystem and capacity assumption is
auditable. The generic launch schema intentionally still has no environment
map: ambient cache variables are rejected, and randomized cache paths never
become shell or JSON environment overrides. Every campaign profile therefore
starts without compiled-code, model-info, or autotuning cache state from a
different arm; warmup remains inside that profile's live server lifecycle.

The tracked Strix Halo ROCm comparison inputs are
`qualification/server-config/kiln-rocm-strix-halo-serving-comparison-v1.toml`
and
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-v1.json`.
They pin the qualified ROCm kernel policy, graph bounds, fixed KV behavior,
batching limits, model path/served ID, debug evidence surface, and private
snapshot roots used by the retained comparison receipts. Do not treat its
absolute model path or host policy as portable.

Equivalent local Kiln launch document structure:

```json
{
  "schema": "kiln.serving-benchmark-server-launch.v1",
  "id": "kiln-rocm-strix-halo-serving-comparison-v1",
  "command": [
    "./target/release/kiln",
    "serve",
    "--config",
    "qualification/server-config/kiln-rocm-strix-halo-serving-comparison-v1.toml"
  ],
  "working_directory": "../..",
  "log_directory": "../../.qualification/serving/logs",
  "readiness_poll_interval_ms": 250,
  "startup_timeout_seconds": 600.0,
  "shutdown_timeout_seconds": 65.0,
  "acceptable_exit_codes": [0]
}
```

### RTX 4090 serving bootstrap

The repository also carries bounded source inputs for the first serving run on
each NVIDIA host:

| Machine/runtime | Tracked input | Initial contract |
|---|---|---|
| 16 GiB RTX 4090 Laptop GPU, Kiln | `qualification/server-config/kiln-cuda-rtx4090-laptop-serving-bootstrap-v1.toml` plus its same-name launch JSON | Stable eager profile, selected-device capacity, 1 GiB live floor, fixed 62-block BF16 KV pool, decode width 16, no reclaim, physical KV resize, FP8 KV, Marlin, or CUDA graph capture. |
| 24 GiB desktop RTX 4090, Kiln | `qualification/server-config/kiln-cuda-rtx4090-desktop-serving-bootstrap-v1.toml` plus its same-name launch JSON | Stable eager profile, 23 GiB capacity cap, 2 GiB live floor, decode width 32, and the same disabled mutation/experimental routes. |
| Either RTX 4090, vLLM | `qualification/server-launch/vllm-cuda-rtx4090-serving-bootstrap-v1.json` | BF16, 32K context/batch-token bound, 64 sequence slots, 75 percent device-memory utilization, isolated per-launch caches, prefix caching, FCFS, seed zero, and text-only serving. |

Both Kiln configs use repository-relative `Qwen3.5-4B`, adapter, and snapshot
paths. The launch documents own the process group, loopback port, readiness
deadline, shutdown deadline, and exclusive log root. The vLLM document uses no
ROCm-only attention override; vLLM must resolve a CUDA-supported backend from
the exact recorded runtime. It launches through a regular copied interpreter at
`.qualification/vllm-cuda-venv/bin/python-kiln`, not a venv symlink which would
canonicalize to the base interpreter and break runtime identity.

The laptop performance campaign does not edit either receipt-bound bootstrap
input. It uses
`kiln-cuda-rtx4090-laptop-serving-performance-v1.toml` and its matching launch
JSON, plus `vllm-cuda-rtx4090-laptop-serving-performance-v1.json`. The Kiln
runtime fields are byte-for-byte equivalent to the laptop bootstrap after
normalizing only `model.path`, `model.adapter_dir`, and `model.snapshot_dir`.
Both performance arms use
`.qualification/cuda-rtx4090-laptop/performance-model-v1`.

That closed model view is necessary because a development model directory may
also contain operational `.cache` and `adapters` trees, including adapter
symlinks that are not model inputs. Prepare it with:

```bash
mkdir -p .qualification/cuda-rtx4090-laptop
python3 scripts/qualification/prepare_cuda_serving_model.py \
  --source Qwen3.5-4B \
  --target .qualification/cuda-rtx4090-laptop/performance-model-v1 \
  --model-id Qwen/Qwen3.5-4B
```

The materializer accepts exactly root regular files plus the two named excluded
directories. It rejects a root symlink, special file, or any other directory.
The target is a read-only directory of same-filesystem hardlinks; reuse
requires every entry to remain the exact source inode and all source/target
bytes and strict model fingerprints to agree. This avoids a persistent 9.3 GB
duplicate on the laptop while preserving per-launch immutable snapshot copies,
initial/final benchmark fingerprints, and source-change detection.

The vLLM performance launch also pins
`--max-provenance-read-mib-per-second=32`. One cumulative launcher schedule
covers model/snapshot/adapter/runtime hashing, and the fresh child runtime
recheck receives the same ceiling. This pacing is outside request timing. It
does not rate-limit vLLM's later accelerator upload, which remains under the
outer WSL2 thermal supervisor and must fail closed if unsafe.
The readiness deadline is 3,600 seconds because a copy-fallback real launch
performs nine complete model reads across staging, verification, identity, and
pre-spawn revalidation before vLLM loads weights. At 32 MiB/s, those reads
consume about 42 minutes of the deadline before accelerator upload. The
deadline is containment, not evidence that startup completed or was thermally
safe.

These are bootstrap baselines, not performance recommendations. First retain a
passing environment receipt, core-correctness receipt, memory-lifecycle receipt,
and eager serving receipt. A tuned scheduler, graph-enabled Kiln input, Marlin
input, different vLLM memory fraction, or different concurrency ceiling is a
new source-bound candidate. Change one causal family at a time and compare it
against the unchanged bootstrap artifact; never edit the bootstrap file in
place after it has a retained receipt.

On the NVIDIA machine, begin from a clean fast-forward of `origin/main` and run
the matching environment, correctness, and lifecycle variants in [Local
Hardware Qualification](qualification.md#cuda-and-metal-core-handoff).
Commit and push each passing receipt before proceeding. Then inventory the
actual host-package sensors:

```bash
python3 scripts/qualification/prepare_host_thermal_policy.py inventory
```

That inventory/policy path applies to native Linux. The current WSL2 laptop has
no readable Linux package sensor and must not convert that absence into a
disabled guard. Its committed
`qualification/host-policies/rtx4090-laptop-wsl2-cgroup-pacing-v2.json` instead
binds the exact Windows `\_TZ.THRM` formatted thermal zone and exact NVML GPU
UUID. Pass it as `--wsl2-thermal-policy` to every CUDA qualification command.
The runner supervises Windows/NVML outside Landlock and places the full private
namespace in the repaired user scope. A WSL run without that policy is rejected
before artifacts are created. The v2 policy adds cgroup pacing at 80/75 C
host/GPU with 75.05/70 C three-sample resume targets and a 300-second pause
deadline; its outer 95/85 C hard-trip boundary remains independent. The host
value is the exact nearest reading exposed by the Windows counter's
tenth-Kelvin representation; it does not alter either hard limit.

After a hard trip, `wsl_thermal_exec.py` terminates the complete supervised
child, continues sampling with no workload alive until the configured stable
handoff target is reached, emits the final safe temperatures, and only then
returns failure. A trip can never be reclassified as success; telemetry loss or
handoff timeout remains a failure and no receipt may claim completed handoff.

Choose a uniquely resolving CPU/package sensor, a hard limit supported by the
machine or CPU vendor, and a conservative idle handoff target observed to be
reachable on that host. A GPU-edge sensor alone is insufficient because source
builds, model hashing, tokenization, and server supervision also heat the host.
Do not copy the Strix Halo thresholds or guess an `hwmonN` number. Materialize
the selected policy without overwrite; the example values are placeholders and
must be replaced before execution:

```bash
python3 scripts/qualification/prepare_host_thermal_policy.py create \
  --id rtx4090-laptop-serving-hard-limit-v1 \
  --hwmon-name '<inventory hwmon_name>' \
  --label '<inventory label>' \
  --limit-millicelsius '<vendor-supported hard limit>' \
  --safe-handoff-target-millicelsius '<measured idle target>' \
  --output qualification/host-policies/rtx4090-laptop-serving-hard-limit-v1.json
```

The writer accepts only `hard_limit_only`, validates all relationships through
the production policy parser, resolves exactly one current input, requires a
valid reading below the hard limit, adds the canonical content hash, fsyncs the
new file, and refuses replacement. Duplicate names/labels are visibly marked
non-unique by the inventory. Review and commit this machine input before a long
build or accelerator run. The desktop uses its own ID and output file because
its CPU, chassis, cooling, and safe threshold may differ.

Build the exact CUDA server through the same policy. The wrapper derives its
selector, hard limit, cadence, and stable prelaunch handoff from the
content-hashed document; the
remaining bounds are build-resource policy rather than product configuration:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
CUDARC_CUDA_VERSION=12080 \
KILN_CARGO_CPU_QUOTA_PERCENT=50 \
KILN_CARGO_MIN_AVAILABLE_GIB=12 \
scripts/cargo-bounded.sh \
  --host-thermal-policy qualification/host-policies/rtx4090-laptop-serving-hard-limit-v1.json \
  build --locked --release -p kiln-server --bin kiln --features cuda
```

Set the available-memory floor for the actual host; do not lower it merely to
force a build through. `cargo-bounded.sh` still supplies one build job, an
aggregate memory cgroup, zero swap allowance, optional CPU quota, overlap
refusal, stable cool-start gate, process-group cleanup, and a thermal trip exit.
The cool-start gate rejects a reading at or above the hard limit immediately
instead of waiting beneath an unsafe ceiling. An explicit policy
conflicts with the four legacy `KILN_CARGO_HOST_THERMAL_*` fields instead of
silently choosing one authority.

Do not use the Linux `--host-thermal-policy` build example on WSL2.
`cargo-test-bounded.sh` accepts the WSL boundary only after the qualification
runner has established and bound the outer thermal supervisor and user scope.
A standalone WSL `cargo-bounded.sh` invocation therefore fails closed. Use a
committed qualification workload for any source build whose output will support
WSL evidence.

Validate the typed server input before loading the model:

```bash
target/release/kiln config \
  --file qualification/server-config/kiln-cuda-rtx4090-laptop-serving-bootstrap-v1.toml \
  --backend cuda --json
```

For vLLM, create `.qualification/vllm-cuda-venv` and install the reviewed
`vllm==0.23.0` CUDA 12.9 runtime rather than a floating package. The current
managed Python is dynamically linked, so copy both the resolved interpreter
and its matching `libpython3.12.so.1.0`; the launch executable must be a regular
file:

```bash
uv venv --python 3.12 --seed --managed-python \
  .qualification/vllm-cuda-venv
uv pip install \
  --python .qualification/vllm-cuda-venv/bin/python \
  'vllm==0.23.0' --torch-backend=cu129
resolved="$(readlink -f .qualification/vllm-cuda-venv/bin/python)"
base="$(dirname "$(dirname "$resolved")")"
cp --reflink=auto "$resolved" \
  .qualification/vllm-cuda-venv/bin/python-kiln
cp --reflink=auto "$base/lib/libpython3.12.so.1.0" \
  .qualification/vllm-cuda-venv/lib/libpython3.12.so.1.0
chmod 0755 .qualification/vllm-cuda-venv/bin/python-kiln
```

From the resulting clean source tree, prepare the closed model view above,
create the empty machine directory, and capture through the performance launch
JSON:

```bash
mkdir -p qualification/runtime/vllm/cuda/rtx4090-laptop
python3 scripts/qualification/capture_vllm_runtime_manifest.py \
  --server-launch-config qualification/server-launch/vllm-cuda-rtx4090-laptop-serving-performance-v1.json \
  --wsl2-thermal-policy qualification/host-policies/rtx4090-laptop-wsl2-cgroup-pacing-v2.json \
  --output qualification/runtime/vllm/cuda/rtx4090-laptop/performance-v1.json
```

The capture tool validates the owned-launch contract, inserts only
`--manifest-only` before the existing vLLM argument boundary, and executes it
twice with bounded output and a per-capture deadline. It requires two
byte-identical strict-valid results under the benchmark driver's manifest plus
launch-binding validators. It publishes the exact repeated bytes
with fsync and no overwrite, and reports the source commit, manifest hash,
runtime-content hash, system fingerprint, both stderr hashes, both thermal
lifecycles, and both scope lifecycles. The launch JSON must be a tracked regular
file whose bytes match `HEAD`. On WSL2, omitting the policy is rejected
automatically. The explicit
content-hashed repository policy and supervisor have the same file and commit
binding. Each pass is nested inside the existing root-owned util-linux private
network/PID/mount/Landlock boundary and a distinct verified systemd user scope
with a 10 GiB memory maximum, zero swap, 512-PID maximum, group OOM handling,
and the 50-percent usage-feedback CPU controller. The same scope controller
adds policy-bound thermal pacing: it freezes the complete cgroup at 80 C host
or 75 C GPU, and resumes only after three consecutive samples at or below
75.05/70 C. A pause that reaches 300 seconds, telemetry loss, a hard-limit sample,
or failure to leave the scope unfrozen fails closed. The independent outer
supervisor keeps the unchanged 95/85 C hard limits and safe handoff. It is the
sole Windows/NVML probe owner and sends strict sequenced samples over an
inherited pipe that the scope controller consumes and does not pass to the
contained payload. The supervisor holds one exact-instance Windows
`Thermal Zone Information` performance-counter process and one exact-UUID NVML
handle for the complete lifecycle. The Windows startup handshake reads the
exact CPU name from the local-machine processor registry key; no CIM/WMI query
is required. It does not launch a new PowerShell or `nvidia-smi` process at the
one-second cadence. Before a child exists, a v2 policy requires three
consecutive readings at or below its 75.05/70 C resume boundary. The Windows
request/response channel binds a monotonically increasing sequence and exact
four-counter schema; missing identity, malformed or reordered data, timeout,
unexpected output, nonzero exit, or cleanup failure is fatal. Freeze and
resume each require both the requested
`cgroup.freeze` value and the kernel's `cgroup.events` state to agree. The tool
requires exactly one successful
`preflight`/`complete` pair per pass, peaks below both hard limits, and the
configured stable handoff before starting the next pass. It also requires
ordered scope start/completion evidence, the exact policy and cgroup controls,
CPU usage within allowance, a matching v2 pacing record with complete pause
counts/durations and sub-limit peaks, no memory-limit/OOM events, successful
child exit, and scope removal. The earlier v1 policy remains valid historical
input but cannot start a new WSL2 qualification or manifest capture. A dirty
tree, existing output, source or commit change during capture, nonzero/timeout
child, missing or malformed thermal/scope evidence, scope residue, oversized
output, invalid manifest, or repeat mismatch fails without publication. A
timeout or interruption terminates the complete capture session, kills the
cgroup through its controller, and waits for the wrapper's handoff before
escalation.

A pacing-timeout failure first unfreezes the scope and sends `SIGINT` through
membership-rechecked pidfds to current leaf processes. It walks upward only as
each leaf exits, giving owned signal handlers and `finally` blocks at most 75
wall seconds to drain process groups and remove private snapshots. Scope
disappearance during that walk is successful cleanup only when the cgroup no
longer exists. Any surviving process still reaches unconditional
`cgroup.kill`. Telemetry, hard-limit, runtime, and accounting failures retain
immediate kill behavior.

Commit the manifest before startup.
The tracked laptop launch makes both captures use the explicit 32 MiB/s
cumulative provenance-read ceiling; do not remove or override it to accelerate
capture.

The retained Laptop GPU manifest at
`qualification/runtime/vllm/cuda/rtx4090-laptop/performance-v1.json` was
captured twice from exact clean pushed source
`2eb37adb9cd7279cc1472f5763ed939fbbe55add`. Both captures produced the same
2,608-byte file with
`sha256:50d46bd54df16f1ea9095dace7656708b7347db3591ad8ebc74d1238d284d125`;
the bound installed-runtime content hash is
`8b3b7273f3e031c427591a4c4447e7541e85023edb92bdf5da51a1882e5e5abb`.
The separate scopes completed in 1,279.766 and 1,075.023 seconds with
10,106,621,952- and 10,551,422,976-byte memory peaks, 27-PID peaks, zero
memory-limit/OOM events, and clean removal. They completed 39 and 41 thermal
pauses totaling 958.839 and 466.370 seconds. Outer host/GPU peaks were
94.05/66 C and 93.05/64 C, both below the unchanged hard limits, and both
lifecycles completed stable handoff. This artifact is the immutable vLLM input
for the next exact-source performance run; it does not itself establish server
startup, request correctness, throughput, or endurance.

Before a model-bearing WSL2 case starts, the qualification runner now performs
the initial model fingerprint in an independent private namespace, 10 GiB
scope, v2 thermal-pacing controller, and outer Windows/NVML lifecycle at a
fixed 32 MiB/s. It repeats the same independently supervised fingerprint after
the case. Each scope must be removed and complete stable handoff; the parent
receipt retains both bounded JSON and supervision streams. The campaign case is
a third lifecycle. This prevents the long before/after provenance reads from
escaping the boundary that protects server work.
The runner passes the same closed base environment and exact private-network
containment marker to these scopes as it passes to ordinary cases. The scope
controller revalidates that marker before any fingerprint read.

The manifest must identify the expected RTX 4090 class, `sm_89`, model and
tokenizer content, interpreter, Python/native packages, CUDA runtime, and every
inference option. Any package, driver, accelerator, model, environment, or
option change invalidates it and requires a new source-bound capture.

Run the first Laptop GPU performance checkpoint through its committed c1
workload, not by transcribing the lower-level campaign command:

```bash
python3 scripts/qualification/run.py \
  --variant cuda-rtx4090-laptop-c1 \
  --host-id rtx4090-laptop \
  --model .qualification/cuda-rtx4090-laptop/performance-model-v1 \
  --model-id Qwen/Qwen3.5-4B \
  --wsl2-thermal-policy qualification/host-policies/rtx4090-laptop-wsl2-cgroup-pacing-v2.json \
  qualification/workloads/serving-cuda-performance-c1-v1.json
```

The workload builds current source once, fixes concurrency at one, and runs
all five campaign profiles in order for Kiln and then vLLM. It validates all
five Kiln receipts before starting vLLM, passes that exact directory as the
reference, and requires all ten nested receipts plus zero exact-output
mismatches. Its deterministic ignored artifact root includes the full source
commit and refuses reuse. Retain the enclosing qualification receipt, both
campaign summaries, and all ten strict-valid nested receipts before citing the
result. Widening concurrency is a later committed workload and must preserve
the same model, prompts, sampling, fixed output length, memory ceiling, launch
inputs, and reference role.

Under WSL2, the c1 source build has three independent bounds. It may use at
most 1,800 active seconds after subtracting only controller-authenticated
thermal-pause overlap, at most 14,400 seconds of that verified overlap, and at
most 16,200 wall seconds. The contained parent reads the outer controller's
private mode-0400 transition stream and validates its exact policy hash,
schema, sequence, pause pairing, monotonic interval, ownership, and size before
crediting any pause. CPU-quota freezes, scheduling, and ordinary idle time are
not thermal credit. Missing, unsafe, partial, active-ended, reordered, or
policy-drifted evidence fails the build and terminates its new-session Cargo
process group. `build_duration_ms` records wall time minus verified thermal
overlap; the trace also retains wall and pause seconds.

After the accepted performance matrix is retained, run the independent laptop
endurance workload:

```bash
python3 scripts/qualification/run.py \
  --variant cuda-rtx4090-laptop-endurance \
  --host-id rtx4090-laptop \
  --model .qualification/cuda-rtx4090-laptop/performance-model-v1 \
  --model-id Qwen/Qwen3.5-4B \
  --wsl2-thermal-policy qualification/host-policies/rtx4090-laptop-wsl2-cgroup-pacing-v2.json \
  qualification/workloads/serving-cuda-endurance-v1.json
```

`serving-cuda-endurance-v1` is source-bound and has file
`sha256:2f34ab2dc62641d247306c9ce29d62c68e5c12b16cd326e2860ce00061a345ac`.
It keeps one stable eager CUDA server alive for at least eight active measured
hours, subtracting only verified thermal-pacing overlap, with fixed 62-block KV
capacity, graphs/reclaim/autoscaling disabled, one active request, varied
16/64/256/1,024-word cohorts, 32 completion tokens, and periodic cancellation.
Cumulative pacing is limited to four hours inside a 44,100-second measurement
deadline and 47,880-second case bound. A persistent NVML counter must
continuously resolve the exact Laptop GPU UUID, product name, and
17,171,480,576-byte capacity.
Stabilization requires two consecutive bounded device-memory/RSS cycles;
post-stabilization end and active-peak growth, request/oracle integrity,
latency attribution, KV/prefix ownership, graph inactivity, shutdown, worker,
snapshot, and process cleanup remain closed receipt gates.

The outer WSL2 controller writes thermal-pacing transitions to a private
mode-0400 JSONL stream. It freezes the complete scope before publishing each
start and publishes each complete matching transition before resuming. The
contained driver opens that stream without following symlinks and requires its
exact owner, type, mode, size, v1 event schema, v2 policy hash, monotonic
sequence, contiguous pause identity, paired state, and interval arithmetic.
Completed events may attribute an ITL gap that spans a verified external
freeze; all other unexplained gaps still fail. The result reports external
pause count, measurement-overlap seconds, longest interval, and active measured
duration. Stream failure or excess cumulative pacing is case failure, and the
enclosing qualification receipt remains authoritative for outer 95/85 C hard
limits, cgroup accounting/removal, process cleanup, and stable thermal handoff.

The workload is a portable handoff only until an exact clean pushed-source run
publishes a retained passing receipt. Do not cite the declaration or its unit
tests as eight-hour laptop evidence.

The first corrected exact invocation from clean pushed source
`c903f7dd97c7250c862db7a393a774e7ca48261e` retained failed parent receipt
`qualification/receipts/cuda/rtx4090-laptop/20260725t203426391915z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`
with file
`sha256:06618553060c9405de939d7f518b1810b3067a282e27cd698148d224f17894ad`.
The initial fingerprint completed, then the current-source CUDA build spent
1,551.390 of 1,616.727 scope seconds thermally paused. Pause 51 crossed the
unchanged 300-second bound at 301.470 seconds, so the controller removed the
scope before server launch or campaign output. Case host/GPU peaks were
93.05/66 C, scoped memory/PID peaks were 1,011,679,232 bytes/14, and all
memory-limit/OOM counters were zero. The independent final fingerprint
completed and matched the initial fingerprint byte for byte. All three scopes
were removed and every outer supervisor completed stable handoff. The receipt
is valid failure evidence only; it provides no request or performance row.
That path also exposed a post-event double close of the controller's telemetry
descriptor. The controller now closes it exactly once and preserves the causal
pacing error. Commit and push the receipt and repair before retrying the
unchanged c1 workload.

The unchanged retry from exact clean pushed source
`1715532c56bc7241797691ba2ee8df38f3e28be6` failed even earlier because the
idle host could not satisfy the pacing resume threshold. The initial
fingerprint scope remained verifiably frozen for 300.520 seconds, peaked at
82.05/65 C host/GPU, and failed at the unchanged per-pause bound before
substantive model reads. The outer supervisor completed its stable handoff at
75.05/63 C after 228 samples. The scope was removed and no child or campaign
root remained. Because initial fingerprint rejection precedes run creation,
this is non-receipt console counterevidence, not a performance artifact. It
also verifies that the corrected failure path reports the causal pacing error
without the former descriptor-close error. Allow the host to cool independently
before another unchanged clean-source attempt; do not weaken the resume or
per-pause limits to obtain a row.

After a five-minute independent cooldown, the next exact invocation from clean
pushed source `8444359ccb5cd8d70bb912c20bef68ada61f91fb` reached an observed
807,174,144-byte fingerprint scope peak before its first freeze. The complete
three-PID scope then stayed frozen for 300.851 seconds and failed at the
unchanged bound. Host/GPU peaks were 85.05/64 C, and the outer supervisor
completed stable handoff at 76.05/63 C after 246 samples. No case, receipt,
campaign root, process, scope, or source residue remained. This second
consecutive idle-host rejection makes another immediate retry unjustified.

The runner now retains failures that occur after the run directory and receipt
identity exist. An initial fingerprint failure publishes a failed parent
receipt with `model: null`, an explicit unavailable environment record, empty
effective config, every declared case failed without execution, and the exact
source, workload, policy snapshot, bounded fingerprint stdout, complete
supervision stderr, and run config. It does not run the environment collector,
case, or final fingerprint after that boundary failure. Receipt v1 permits the
missing model identity only for failed receipts; every passed serving,
performance, training, eval, or soak receipt still requires the complete
fingerprint. This closes evidence retention, not host thermal availability or
performance acceptance.

After a ten-minute independent cooldown, the unchanged invocation from exact
clean pushed source `98f7c2db72523b35c042818a457ea4fdfa637a11`
published the first such retained failure:
`qualification/receipts/cuda/rtx4090-laptop/20260725t221058934964z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`,
file
`sha256:c8d8e3364d46e84f7b89eac810dc1ccbd036280bce49261c4f69017128802bee`.
The incomplete initial fingerprint scope ran for 336.353 seconds, peaked at
1,253,879,808 bytes and four PIDs, used 1,314,771 of 168,176,288 allowed CPU
microseconds, and recorded zero memory-limit or OOM events. Two verified
freezes totaled 334.129 seconds; the second reached 300.510 seconds and failed
at the unchanged per-pause bound after an 86.05/64 C host/GPU peak. The scope
was removed and the outer supervisor completed stable handoff at 76.05/63 C.
No environment collection, case, final fingerprint, campaign output, or
process survived. The receipt and all five local artifacts pass strict
known-commit/local-artifact validation, but `model: null` and the failed
unexecuted c1 result make it boundary counterevidence only.

The next exact c1 invocation from clean pushed source
`04d2c229bab5c5d668b77166b7900fbac5717342` retained failed receipt
`qualification/receipts/cuda/rtx4090-laptop/20260725t231031150626z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`,
file
`sha256:d77cfb5a1a9838c83b8ac5addb184aa87989b53530e4a88c4a95bcf417caabfd`.
The formal preflight read 78.05/63 C host/GPU. Its initial fingerprint scope
ran for 714.265 seconds, reached 4,264,914,944 bytes and four PIDs, used
4,184,272 of 357,132,684 allowed CPU microseconds, and recorded zero memory
limit or OOM events. Four pauses totaled 709.341 seconds; three completed, and
the fourth reached 300.648 seconds at a 94.05/66 C peak before the unchanged
per-pause bound rejected it. Scope removal and the 78.05/64 C outer stable
handoff completed, with no process or campaign residue. The strict-valid
receipt still has `model: null`, unavailable environment identity, empty
effective config, and an unexecuted failed c1 result. It is pre-model thermal
counterevidence only. Do not cite it as performance evidence or weaken the
thermal policy for another retry.

That failure also exposed observer work outside the measured scope. The
fingerprint used only 4.184 CPU-seconds while the former outer guard launched a
new PowerShell/WMI query and `nvidia-smi` process for each of its 534 samples.
A post-failure Windows counter snapshot showed `WmiPrvSE` at 21 percent CPU,
and a read-only idle trend oscillated from 73.05 to 83.05 C. The repaired
supervisor keeps both sensor sources open. A 60-sample live dirty-source probe
completed in 60.182 seconds with 0.000 WSL-visible CPU-seconds in the
persistent PowerShell bridge, a 73.05-81.05 C host range, displayed samples
15/30/45/60 at 73.05/74.05/74.05/73.05 C, fixed 63 C GPU readings, and no
sensor-process residue. An exact guarded no-op then passed the telemetry, paced scope,
namespace, accounting, cleanup, and stable-handoff lifecycle with a
75.05/63 C start, 84.05/63 C outer peak, and 75.05/63 C end. Neither probe is a
serving receipt. Push the repair before rerunning fixed c1; every thermal
threshold and workload semantic remains unchanged.

The clean pushed-source retry at
`1e22141cfd2a60efe4e00d674dcd0190ec95a348` retained strict-valid failed
receipt
`qualification/receipts/cuda/rtx4090-laptop/20260725t235849138625z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`,
file
`sha256:bad5e9d085176e035617f27ad687b60bb40cdf29485dd4a95eb886b45f3b10bd`.
The external admission probe read 74.05/64 C, but formal preflight after the
separate CPU-identity WMI query read 86.05/64 C. The initial fingerprint scope
ran 751.564 seconds, peaked at 9,387,716,608 bytes/four PIDs, used 13,122,123
of 375,782,151 allowed CPU microseconds, and recorded zero memory-limit or OOM
events. Two of three pauses completed; all pauses totaled 459.892 seconds, and
the active third pause reached 300.867 seconds at a 90.05/64 C peak. Scope
removal and 74.05/64 C outer handoff completed after 752 samples with no
process or campaign residue. Model identity, environment collection, build,
server, and request work did not complete. This is pre-model counterevidence,
not c1. Replace the remaining WMI identity read with an exact non-WMI source
and admit a child only after the pacing-resume boundary is stable; do not
change the thermal limits.

The next source repair removes the fresh CPU-identity WMI query. The persistent
Windows process reads the exact `ProcessorNameString` from
`HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0` and binds
it into the same ready handshake as the exact thermal-zone/counter identity.
Before any v2-policy child exists, the outer supervisor now requires the
policy's three consecutive readings at or below the unchanged 75/70 C
pacing-resume boundary. Pre-admission samples remain in outer count/peak
evidence. Timeout, sensor failure, or a hard-limit sample rejects the launch.
A dirty-source five-read probe observed host
75.05/77.05/74.05/74.05/74.05 C and fixed 64 C GPU with clean sensor exit. An
exact guarded no-op then admitted at 74.05/64 C, ran with zero scope pauses,
removed the scope, and handed off at 74.05/64 C; its nine-sample outer peak
retained the earlier 77.05 C reading. This is a boundary probe, not serving
evidence. Push the repair before fixed c1.

The clean pushed-source retry at
`7e3406917ec369a2bf10f83918d637944932d805` retained strict failed receipt
`qualification/receipts/cuda/rtx4090-laptop/20260726t002130041819z-cuda-rtx4090-laptop-serving-cuda-performance-54e7111044-v1.json`,
file
`sha256:bac0abc5fc9a8469738c13758c4a4d57619f1aa3633ad6744189773c4c94b26e`.
Formal preflight was the intended 74.05/64 C. The fingerprint scope ran
475.132 seconds, reached 5,503,045,632 bytes/four PIDs, used 5,939,727 of
237,565,837 allowed CPU microseconds, and recorded zero memory-limit/OOM
events. Three of four pauses completed; total pacing was 442.090 seconds and
the active fourth pause expired at 300.360 seconds after a 90.05/64 C peak.
Scope removal and 74.05/64 C outer handoff completed after 483 samples with no
residue. Model identity, environment, build, server, and request work did not
complete, so this remains pre-model counterevidence.

The fingerprint limiter's cumulative deadline explains the repeated reheating.
SIGSTOP time advances `time.monotonic()` while the scoped reader cannot run;
on resume the deadline is far behind current time, allowing full-speed reads
until cumulative bytes catch up. Rebase a stale deadline at the first
post-pause chunk so frozen/idle time grants no credit. Keep the same 32 MiB/s
maximum, shared limiter across both integrity passes, and unchanged thermal
policy before retrying fixed c1.

`_ReadRateLimiter` now retains one next deadline. Each charged chunk advances
that deadline by `bytes / configured_rate`; if external stop or ordinary idle
time has moved the clock past it, the limiter first rebases to current time.
The next 8 MiB read therefore pays its complete 250 ms interval at the
unchanged laptop 32 MiB/s cap instead of spending pause credit. All bytes from
all files and both integrity passes remain in one total. A synthetic
one-MiB/s regression paces two half-MiB chunks from time 10 to 11, jumps to
time 100, and proves the next half-MiB still paces through 100.5. This closes
the portable catch-up-burst defect only; push the source before fixed c1.

For the first Kiln campaign, use the exact UUID from
`.environment.device.device_uuid` in the environment receipt, the matching
bootstrap launch JSON, and `target/release/kiln` as `--runtime-artifact`. Use a
machine-reviewed memory ceiling below the selected device's reported capacity:

```bash
python3 scripts/run-serving-benchmark-campaign.py \
  --engine kiln \
  --base-url http://127.0.0.1:8420 \
  --model Qwen3.5-4B \
  --model-path .qualification/cuda-rtx4090-laptop/performance-model-v1 \
  --runtime-identity "kiln-git:$(git rev-parse HEAD)" \
  --runtime-artifact target/release/kiln \
  --campaign-id cuda-rtx4090-laptop-bootstrap-attempt-001 \
  --prompt-set-id cuda-rtx4090-bootstrap-v1 \
  --out-dir .qualification/serving/cuda-rtx4090-laptop-bootstrap-v1 \
  --sizes 1 \
  --memory-source nvml \
  --memory-device-uuid '<environment receipt GPU UUID>' \
  --memory-limit-bytes '<reviewed whole-device ceiling>' \
  --model-fingerprint-read-mib-per-second 256 \
  --external-wsl2-thermal-policy qualification/host-policies/rtx4090-laptop-wsl2-cgroup-pacing-v2.json \
  --server-launch-config qualification/server-launch/kiln-cuda-rtx4090-laptop-serving-performance-v1.json \
  --output-evidence hashes
```

Run that WSL2 command only through a committed qualification workload. Direct
shell execution has no parent receipt and fails the external-boundary
revalidation. On native Linux, use the machine-specific
`--host-thermal-policy` from the preparation procedure instead.

Run vLLM only after its manifest is committed. Change `--engine`, loopback port,
runtime artifact, campaign/output IDs, and launch JSON; point `--reference-dir`
at the matching accepted Kiln profile receipts. Keep model, prompt-set identity,
sampling contract, output length, UUID, memory ceiling, thermal policy, and
sample cadence identical. A startup failure, model mismatch, nonzero request
error, output mismatch, thermal trip, forced shutdown, NVML error, process
residue, or memory-ceiling violation rejects the arm. Start at concurrency one
and expand only through rows that preserve these gates. Commit failed compact
counterevidence when it reveals a product or harness defect; fix that defect
before widening the workload.

## Thermal Contract

`kiln.host-thermal-policy.v1` selects exactly one hwmon sensor and defines a
hard termination limit, polling cadence, an explicit protection mode,
phase-settlement deadline, and a stable cooldown target. `hard_limit_only`
continuously monitors and terminates at the limit but never suspends active
work. `process_group_stop` additionally defines stop/resume hysteresis and
sends `SIGSTOP`/`SIGCONT` to the complete process group; use it only where that
mechanism has been qualified for the runtime and device. Crossing the hard
limit always sends `SIGTERM`, records the trip, and fails the run.

For `process_group_stop`, the phase-settlement deadline is also installed in
the independent monitor as the maximum duration of any single stopped
interval. It is not only checked by the request-driving thread: if that thread
is blocked while already-submitted accelerator work prevents the package from
reaching the resume gate, the monitor sends `SIGTERM`, releases the stopped
group with `SIGCONT`, and fails closed. `hard_limit_only` has no stopped
interval, so the field remains a receipt-compatible phase bound but never arms
`SIGSTOP`. Driver v11 instead uses it as the maximum duration of each verified
idle-boundary cooldown before and after a warmup or measured row.

The closed policy fields are:

| Field | Contract |
|---|---|
| `schema` | Exact `kiln.host-thermal-policy.v1` identifier. |
| `id` | Stable portable policy identifier, 3 to 128 characters. |
| `sensor.hwmon_name` | Exact Linux hwmon device name. The selector must resolve once. |
| `sensor.label` | Exact temperature-channel label beneath that hwmon device. |
| `limit_millicelsius` | Hard termination boundary. It must be strictly above the pacing start when process-stop pacing is selected. |
| `poll_interval_ms` | Positive cadence shared by pre-launch cooling, runtime protection, pacing, and final cooldown. |
| `pacing.mode` | Required tagged choice: `hard_limit_only` or `process_group_stop`. |
| `pacing.start_millicelsius` | Required only for `process_group_stop`; temperature at which the complete process group is stopped. |
| `pacing.resume_millicelsius` | Required only for `process_group_stop`; lower temperature at which resume becomes eligible. |
| `pacing.resume_stable_samples` | Required only for `process_group_stop`; consecutive eligible samples required before `SIGCONT`. |
| `safe_handoff.target_millicelsius` | Maximum temperature accepted before owned process creation, at each driver-v11 hard-limit-only idle row boundary, and after shutdown or before returning an attached live process. Under process-stop pacing it cannot exceed the pacing-resume temperature. |
| `safe_handoff.stable_samples` | Consecutive samples required at or below the target at every applicable boundary. |
| `safe_handoff.timeout_seconds` | Positive deadline for each boundary cooldown. |
| `phase_settlement_timeout_seconds` | Positive phase bound. Under `process_group_stop`, it is also the independently enforced maximum active pacing interval. Under `hard_limit_only`, it bounds each live-server idle cooldown; no stop interval exists. |

Driver v8 resolves the selected sensor before model hashing. Each initial and
final fingerprint worker starts blocked on a private gate; the supervisor first
requires a stable prelaunch boundary, attaches the continuous selected-mode
guard, releases the worker, reconciles every pacing interval, requires a clean
exit, and completes a stable post-exit cooldown. The receipt retains both closed
`kiln.hf-thermal-containment.v1` records under
`host_thermal.model_fingerprint`, plus the fingerprint implementation and Python
hashes. Missing initial evidence is invalid. Missing final evidence is accepted
only with a failed `model_identity_unchanged` check. Both evidence policies must
equal the server guard's content-hashed policy.

After the initial fingerprint and remaining provenance work, owned mode does not
call the server `Popen` until `safe_handoff.stable_samples` consecutive readings
are at or below `safe_handoff.target_millicelsius`. A timeout fails before the
server child exists. Driver v5 and later receipts retain the sensor path, policy
values, start, peak, and end temperatures, sample count, stable count, elapsed
time, scope, and completion state for this boundary. The validator requires
those values to match the runtime guard's sensor.

Cooling remains part of wall-clock service cost. Each row records both request
window output throughput and thermally sustainable throughput including any
pacing settlement. The top-level receipt records the selected content-hashed
policy, start/peak/final temperature, every pacing duration (zero under
`hard_limit_only`), guard errors and trips, cooldown duration and samples,
sensor path, process identity, and final liveness.

Owned mode disables new pacing immediately before shutdown, releases any active
stop, and keeps hard-limit monitoring active until the child exits. Attached
mode instead cools the live process to the safe-handoff target before releasing
the operator back to an unguarded server.

Driver v23 attempts owned shutdown and log finalization independently, so one
recoverable exception cannot suppress the other artifact. A shutdown-accounting
error that still reaches the bounded emergency drain carries conservative
`forced=true` lifecycle evidence and fails the receipt. A durability error that
leaves a stable readable log carries that log's exact path, byte count, and
hash while also failing the receipt. Log `fsync` retries are restricted to
`EINTR`, `EAGAIN`, `EBUSY`, and `ETIMEDOUT`, with three bounded attempts;
non-transient errors are never converted into passing evidence. When lifecycle
evidence cannot be serialized after catastrophic cleanup, the driver surfaces
the original finalization detail before schema validation. Historical v22
receipts keep their original lifecycle contract.

The WSL2 outer supervisor owns a nonblocking thermal-sample pipe into the
resource-scope controller. Closing that pipe before the scope exits is still a
hard loss of pacing authority. One narrower teardown case is not a thermal
trip: if a sample write fails specifically with `EPIPE`, the outer supervisor
waits no longer than the configured poll cadence and never more than one second
for the owned scope process to exit, then preserves its exact status. A live
scope after that interval fails closed. `EAGAIN`, other write errors, partial
writes, hard-limit samples, and invalid telemetry never enter this settlement
path. This keeps thermal protection continuous through work while avoiding a
false trip after the scope has already completed and closed its reader.

## Workload Matrix

The fixed profiles are:

| Profile | Purpose | Comparison |
|---|---|---|
| `greedy-short` | Short deterministic decode and concurrency scaling. | Exact output. |
| `api-default-sampled` | Default sampling behavior at fixed seed/input. | Inputs only. |
| `long-prefill` | Large prompt admission and prefill. | Exact output. |
| `prefix-hit` | Shared-prefix cache behavior. | Exact output. |
| `mixed` | Non-uniform prompt lengths and scheduling. | Exact output. |

Every request uses a launch barrier, deterministic prompt identity, explicit
sampling and template fields, streaming usage, a fixed completion length, and a
bounded timeout. The receipt retains TTFT, client-visible ITL, end-to-end
latency, request throughput, output-token throughput, SLO goodput, dispatch
spread, prompt/output hashes, whole-device memory, failures, and Kiln server-route
diagnostics.

Prompt template `fixed-serving-profiles-v2` uses 61 copies of the shared
long-prompt block. The source-bound RTX 4090 Laptop c1 handoff renders every
warmup and measured prompt through the exact pinned tokenizer and chat template
before build or server startup. It requires a maximum of 3,883 input tokens and
3,947 total tokens after the 64-token output reserve, leaving 21 tokens below
the declared 3,968-token KV context. Tokenizer, template, driver, and prompt
template identities are part of the effective workload. Historical
`fixed-serving-profiles-v1` receipts retain their original 64-block prompt
identity and validation.

## Device-Memory Telemetry

Drivers v19 and later sample whole-device used memory through one typed source. This is
the same scope for both supported mechanisms: it is not per-process memory and
therefore includes other users of the selected accelerator. Qualification
machines must be quiescent enough that the recorded baseline and peak delta are
meaningful.

| `--memory-source` | Selector | Receipt identity |
|---|---|---|
| `auto` | Succeeds only when exactly one DRM or NVML device is measurable. | Resolved source and identity below. |
| `drm` | `--memory-path auto` requires one Linux `mem_info_vram_used` file; an explicit path selects one of several DRM devices. | Canonical sysfs path, `source=drm_vram_used`. |
| `nvml` | `--memory-device-uuid GPU-...` selects stable identity; `--memory-device-index N` selects a physical NVML index. Both may be omitted only when NVML enumerates exactly one GPU. | Resolved index, enumeration count, UUID, product name, total bytes, loaded library, and NVML version, `source=nvml_used`. |

An explicit `--memory-path` with the default source is normalized to `drm`; an
explicit NVML index or UUID is normalized to `nvml`. Supplying an index and UUID
together, or combining a DRM path with either, is an error. On a mixed
AMD/NVIDIA host, or any host with several candidate devices,
`auto` fails rather than guessing which device serves the model. NVML indices
are physical NVML enumeration indices, not remapped CUDA logical indices.
Prefer the UUID captured in the environment receipt when `CUDA_VISIBLE_DEVICES`
or container remapping is involved. UUID selection enumerates NVML and requires
exactly one match.

The sampler calls NVML in-process instead of polling `nvidia-smi`, avoiding a
subprocess on every 50 ms sample. It reads once at each run boundary and on the
background cadence. Counter reads and baseline/peak state changes share one
boundary lock, so an in-flight sample from the prior row cannot be committed
after the next row resets its baseline. A negative or inconsistent value, any
background read failure, a sampler thread that does not stop, or failed NVML
shutdown makes the receipt fail. A memory limit larger than the selected NVML
device's recorded capacity is rejected before measurement. Passed v19-v21
Kiln/vLLM comparisons must
use the same cadence and source and must bind the same DRM path or NVML
UUID/capacity. Thus peak-memory comparisons cannot silently come from different
GPUs.

Linux AMD/ROCm and Vulkan example:

```bash
--memory-source drm \
--memory-path /sys/class/drm/card1/device/mem_info_vram_used \
--memory-limit-bytes 50000000000
```

Linux NVIDIA/CUDA example bound to the environment receipt's GPU UUID:

```bash
--memory-source nvml \
--memory-device-uuid GPU-01234567-89ab-cdef-0123-456789abcdef \
--memory-limit-bytes 15000000000
```

Choose the limit below total capacity to preserve the machine-specific safety
margin. The 16 GB laptop and 24 GB desktop must use their own limits; changing
the limit between Kiln and vLLM changes the workload fingerprint and makes the
receipts incomparable.

### Run and prompt identity

Driver v16 separates two identities that earlier receipts coupled:

- `workload.run_id` is unique to one execution. It selects the owned server-log
  path and distinguishes the receipt, candidate, engine, and retry. It is never
  included in a request body or prompt hash.
- `workload.prompt_set_id` is stable across runs intended for comparison. It
  seeds marker ordering and replaces the old run-ID value in the unchanged
  model-visible `Benchmark run:` line. Use the old run ID as this value when a
  v16 arm must reproduce a v15 prompt byte for byte.

Both are required 3-to-128-character portable identifiers for measured runs,
and their values must differ; there is no fallback from one to the other.
Distinct run IDs with one prompt-set ID produce identical prompt text and
prompt-set hashes. Changing only the
prompt-set ID changes the prompts. Strict validation reconstructs every warmup
and measured prompt hash from the recorded prompt-set ID, phase, row width, and
profile, rejecting missing, malformed, or stale identity even when the receipt
self-hash has been recomputed.

The v16 `workload_fingerprint` hashes the complete model-visible and scheduling
contract except `run_id`; `prompt_set_id` remains included. The receipt's
canonical self-hash still covers `run_id`. Consequently reference comparison
accepts unique lifecycle identities while rejecting any prompt, sampling,
shape, seed, model alias, SLO, or memory-limit drift. V2-v15 validators retain
their original exact-workload fingerprint semantics.

Driver v7 through v20 retain one ordered `output_evidence` row for every successful
request. Hash-only evidence is the default and includes the combined semantic
output hash, separate reasoning/content hashes, UTF-8 byte counts, completion
tokens, and finish reason. The validator requires these rows to cover exactly
the successful request indices, reproduce the aggregate output-set hash, and
reproduce the run's completion-token total. An aggregate mismatch can therefore
no longer hide whether one request or the entire row diverged.

### Request-local performance evidence

Driver v15 retains Kiln's terminal `metadata.performance` object for every
successful measured request. The comparison profiles enable that response field
through the typed `server.chat_performance_metadata` setting. The driver does
not add `include_performance` or another Kiln-only field to the measured request
body, so Kiln and vLLM continue to receive identical inputs. vLLM rows record
both `request_performance` and `request_phase_summary` as `null`.

Each Kiln `request_performance` row is bound to its request index. Strict
validation requires its prompt/completion usage and finish reason to agree with
the independent usage and output-evidence records. It also requires the closed
latency object to reconcile emitted tokens, retained gaps, total stall count,
all 19 stall-reason counts, and all 20 nullable phase fields. Missing terminal
metadata or a null latency object fails the `request_performance_accounted`
gate for an otherwise successful Kiln request; malformed metadata fails that
request and remains visible as a structured counterexample.

`request_phase_summary` is derived entirely from those retained request rows.
For each phase and terminal request metric it records the non-null request
population plus p50, p99, and maximum. It also totals emitted tokens and closed
stall-reason counts. Phase durations are deliberately not summed: concurrent
requests share and overlap actor/device work, so summing request-local phase
time would not describe service-wall time. Use route-level before/after server
counters to rank service-wall cost and request distributions to identify which
clients experienced it. The validator recomputes the complete summary and
rejects edited or stale aggregates.

### Idle-boundary cooldown evidence

Driver v11 adds `idle_boundary_cooldowns` to every guarded row. Under
`hard_limit_only`, it contains exactly two ordered records: `pre_run` and
`post_run`. The caller enters these waits only after it owns a boundary with no
live request wave. The thermal guard samples with pacing disabled, so the wait
cannot send `SIGSTOP`; the independent hard limit remains active and still
terminates the complete process group on a trip, sensor error, process exit, or
timeout.

Each record binds its sensor path, poll interval, target, stable-sample
requirement, timeout, sample count, elapsed time, start/peak/end temperature,
scope, position, and completion status to the content-hashed policy. The
validator requires both records for a non-tripped row, permits only the ordered
prefix when a hard-limit trip interrupts a row, and rejects a combined cooldown
duration greater than the phase wall time. Because the phase clock starts
before `pre_run` and ends after `post_run`,
`thermally_sustainable_output_token_throughput_per_s` charges both cooling
waits while request-window throughput remains separately observable.

For a retained `process_group_stop` policy, the v11 array is empty and the
historical pacing fields retain their original meaning. New serving
qualification on Strix Halo uses `hard_limit_only`; this compatibility path
does not re-authorize active-work suspension there.

### Server route diagnostics

Drivers v18 through v20 use `kiln.serving-benchmark-server-diagnostics.v7`. Its
`request_route` is exactly `batching_engine`; the batching-engine record is
mandatory; and the record contains no direct-worker, rendezvous, or alternate
route object. The before/after actor ownership must be stable, all cumulative
counters must be monotonic, and the end-of-window active count must be zero.
This is the current qualification contract for every real backend.

Drivers v9 through v17 retain the following historical route-aware contracts
so committed receipts remain independently valid. Driver v9 replaced the
then-batching-only `server` record with
`kiln.serving-benchmark-server-diagnostics.v2`. This is a receipt-contract
change, not a request-workload change; strict-valid v7 and v8 receipts remain
accepted and comparison-compatible.

The v9 record always retains the effective request route as either
`batching_engine` or `direct_streaming`. It also retains:

- deltas for the process-wide `ok`, `error`, `timeout`, and `rejected` request
  counters, plus the end-of-window active count and process peak;
- the effective batching-actor bit and the complete direct-rendezvous ownership
  state: backend availability and reason, actor/worker liveness, scope, and
  whether that worker is routable;
- batching-engine decode, prefill, admission, error, width, and timing deltas
  when the actor exists; and
- direct-rendezvous submitted/executed row, batch, runner-call, busy, failure,
  width, and runner-call-budget evidence when that worker exists.

Presence is strict. An effective batching actor requires batching-engine
diagnostics; an absent actor requires them to be null. Direct-worker liveness
must agree with the decode-batcher record, and route availability must equal
`backend_available && !actor_active && worker_active`. Before/after ownership
must be identical. All cumulative counters must be monotonic.

Two run gates consume this evidence. `server_reported_no_errors` requires zero
request errors/timeouts/rejections, zero actor errors, zero direct-worker
failures, and no direct runner-call-budget violation. The independent
`server_request_accounting` gate requires exactly one server `ok` result per
declared client request and zero active requests after the window. A direct
stream without a rendezvous worker remains observable through the universal
request counters rather than being rejected merely because actor counters are
absent. Console rows name the effective route and show the corresponding actor
width/mean or rendezvous width/runner-call mean.

Driver v10 extends this route record to
`kiln.serving-benchmark-server-diagnostics.v3` with one required
`rocm_graphs` object. Driver-v9 `v2` records remain strict-valid and
comparison-compatible; the validator dispatches by receipt driver version and
does not reinterpret an old record as `v3`.

The graph record is a before/after measurement-window contract. It retains:

- the end-of-window `enabled`/`disabled` state, requested and effective capture
  bits, and an explicit unavailable reason when the backend has no ROCm graph
  runner;
- monotonic-window deltas for `capture_attempts`, `capture_successes`,
  `capture_deferrals`, and `capture_failures`; `replay_attempts`,
  `replay_successes`, and `replay_failures`; aggregate `failures`;
  `graph_slot_create_count`, `graph_slot_reuse_count`, and
  `cache_admission_successes`;
- start and end gauges for captured graphs and total, active, and idle graph
  slots; and
- deltas for every closed eager-fallback reason, fallback count, slow-fallback
  count, accumulated fallback duration, and the process-wide maximum fallback
  duration observed by the end boundary.

Driver v14 advances this graph record to
`kiln.serving-benchmark-server-diagnostics.v5`. Its only graph-shape addition
is the required `multi_row_batch_unsupported` fallback counter. A positive
value is valid only when graph capture was requested and the measured batching
route observed more than one row. Driver v13 retains diagnostics v4 without
this key and remains strict-valid under its original contract.

The closed fallback reason keys are `multi_row_batch_unsupported`,
`cold_cache_host_round_trip`,
`persistent_host_round_trip`, `shape_dependent_attention`,
`graph_cache_capacity`, `graph_cache_byte_budget`,
`graph_accounting_incomplete`, `moderate_memory_pressure`,
`tight_memory_pressure`, `critical_memory_pressure`,
`memory_reservation_denied`, `memory_governor_selector_mismatch`,
`capture_failure`, and `replay_failure`. The record also carries `total`,
`slow`, `total_duration_micros`, and `process_max_duration_micros`.

The validator requires replay attempts to equal successful plus failed replay
launches, aggregate graph failures to equal capture plus replay failures, and
active plus idle slots to equal total slots at both boundaries. Every monotonic
counter and the process maximum must not regress. The fallback total must equal
the sum of its closed reason counters. Requested policy cannot change during a
run; a circuit breaker may change effective execution from enabled to disabled,
but cannot silently rearm. Diagnostic availability cannot appear or disappear
between boundaries.

A backend without a ROCm graph runner emits `state=unavailable`,
`unavailable_reason=backend_without_graph_runner`, and null graph
counters/gauges. Busy or poisoned ownership is not treated as backend absence.
It fails the graph-accounting gate because the benchmark cannot prove what ran.

Driver v10 adds `rocm_graph_execution_accounted`. When native capture is not
requested, or the backend explicitly has no ROCm graph runner, the gate passes
without inventing activity. When capture is requested, the run passes only if
capture remains enabled, at least one capture or replay succeeds, graph failures
are zero, and the complete eager-fallback delta is zero. This prevents a
nominally graph-enabled performance row from passing after executing only the
eager path. Graph failures also participate in `server_reported_no_errors`.
Current ROCm capture includes the contiguous BF16 multi-row route. A qualified
batched graph row must show successful capture/replay, zero graph failures, and
zero fallback delta; `multi_row_batch_unsupported` remains a closed historical
reason for receipt compatibility. Older binaries incremented that counter for
every successful multi-row eager forward, so those receipts remain
evidence-valid but fail `rocm_graph_execution_accounted`. The current runner
keys retained batch graphs by width and attention bucket, keeps LM head and
sampling eager, and reports the shared width slot as active.
Capture success in current source occurs only after the automatic first-launch
parity oracle compares hidden output, all GDN recurrent/conv state, and the
current K/V rows with that candidate's own eager warm pass. The structured
`rocm_graph_capture_parity_check` event is the detailed attribution record; a
mismatch or comparison error increments capture failure, disables graphs for the
runner, and cannot satisfy this gate through its later contained eager retry.

Driver v17 advances the closed server-diagnostics record to
`kiln.serving-benchmark-server-diagnostics.v6` and makes that admission proof
part of the receipt. `rocm_graphs.capture_parity` retains measured-window
batched capture attempts/successes/deferrals/failures, parity
checks/passes/failures/errors, successfully compared bytes, and comparison
microseconds. It also retains start/end lifetime values for batched successes
and every parity outcome. Validation requires batched outcomes and parity
outcomes to reconcile,
lifetime successful batched captures never to exceed passed parity checks,
monotonic boundaries, and positive compared bytes for a nonzero check delta. A
parity pass can precede a later cache-admission error, so a failed lifetime may
contain more passes than admissions. A clean measured promotion row permits no
such failure and requires the counts to be equal.

The `rocm_graph_capture_parity_accounted` gate is not inferred from a log. For a
measured multi-row route with capture requested, it requires an armed runner,
zero graph failure, zero lifetime parity failure/error, equal lifetime batched
success/pass counts, at least one admitted batch graph, and exact measured
deltas. A single-row or graph-disabled row must report zero measured batched
parity activity. Backends without a ROCm graph runner report null parity
evidence and remain not applicable. The structured log event remains the
detailed mismatch attribution record; health counters and the receipt are the
durable promotion gate.

### Reference comparison roles

Driver v17 records one top-level `reference_role` selected by
`--reference-role`:

- `qualification_gate` is the default. Any prompt, token-count, or exact-output
  mismatch keeps the entire receipt failed, exactly as in earlier drivers.
- `same_artifact_graph_eager_discriminator` keeps cross-process exact-output
  comparison as explicit reproducibility evidence while graph correctness is
  gated by the same-process parity record. It is available only for Kiln and
  requires `--reference-receipt`.

The discriminator is deliberately narrow. The reference must itself be a
passed, ordinary current-driver receipt with observed graph-disabled warmup and
measured rows, zero graph activity/failure/fallback, the identical runtime
binary hash and runtime identity, and the same model, thermal policy, and
workload fingerprint. The candidate must request graph capture and pass
graph-execution plus parity gates
on every measured row. Its comparison object records
`verdict_effect=evidence_only`, retains `matched` and every structured mismatch,
and embeds the eager-reference execution summary. Changing the role, binary,
reference graph state, or candidate parity data and recomputing the receipt hash
still fails strict validation. No other comparison can become evidence-only.

### Cooperative actor-cycle idle evidence

Driver v13 advances the route record to
`kiln.serving-benchmark-server-diagnostics.v4`. It adds six closed fields to a
present batching-engine record: immutable `actor_cycle_idle_ms` and
`actor_cycle_idle_source`, measured-window `actor_cycle_idle_count` and
`actor_cycle_idle_seconds`, end-boundary `actor_cycle_idle_active_end`, and the
process-lifetime `process_max_actor_cycle_idle_ms`. Driver-v10 through v12 `v3`
records remain strict-valid under their original receipt versions.

Before and after each request window, v13 polls the health snapshot to a
cooperative-idle boundary. This polling is outside request timing and is bounded
by the configured delay plus one second and the request timeout. The measured
delta therefore covers waits caused by the declared request wave without
mistaking an in-progress final wait for missing accounting.

The `actor_cycle_idle_accounted` gate passes an absent actor as not applicable.
For a present actor, it requires the end boundary to be inactive. A zero policy
must have zero wait count, elapsed time, and process maximum. A nonzero policy
must have explicit `config_file` or `environment` provenance plus positive
count, elapsed time, and maximum. This makes a fixed duty-cycle comparison
auditable and prevents either a silently inactive pacing knob or unexplained
actor sleep from being accepted as measured throughput. The field is not a
thermal controller; the independent host guard and all thermal gates remain
unchanged.

### Output divergence diagnostics

Use `--output-evidence full` on both engine arms when investigating exact-output
parity. In addition to the mandatory hashes, each request retains its reasoning
and visible content as canonical padded base64. This mode is explicit because
generated output becomes part of the receipt and may be unsuitable for public
retention when prompts can contain private data. The benchmark's fixed prompts
contain no operator input, but repositories that adapt the driver must make
their own disclosure decision.

Full evidence is bounded to 1 MiB of combined UTF-8 reasoning and content per
request. Exceeding the bound fails closed instead of silently truncating the
correctness record. Receipt validation decodes the text, enforces canonical
base64 and UTF-8, and recomputes byte counts plus the separate and combined
hashes. When both paired receipts contain full evidence, an `output_mismatch`
reports:

- the mismatch count and ordered request indices;
- the exact fields that differ for each request;
- the expected and actual combined output hashes; and
- the first divergent UTF-8 byte in reasoning and visible content.

The offsets are bytes, not tokenizer IDs. They remain meaningful across engines
without importing either engine's tokenizer into the measurement driver. If
either arm is hash-only, request cardinality and changed fields still remain
available, while the byte offsets are null and `exact_output_compared` is
false. Full text is never copied into the comparison summary itself.

Reference comparison is independent of lifecycle acceptance. When every
declared request row completed, the driver retains the comparison even if a
repository, identity, shutdown, or thermal finalization check failed. The
receipt verdict remains failed; preserving the comparison prevents a safety
failure from hiding whether the completed model outputs matched.

### Current Strix Halo ROCm comparison

The first exact-source pair uses commit `8f1e026b3`, Qwen3.5-4B, the fast
58/50/93 C host guard, greedy decoding, 64 output tokens, one repeat, and the
tracked launch inputs above. Both engines completed every request and every
ownership, memory, identity, cooldown, and teardown gate. The compact receipts
are:

- `benchmarks/receipts/rocm/strix-halo/20260718t223203-rocm-strix-halo-greedy-short-c1-32-sourcepair-v1.kiln.json`
- `benchmarks/receipts/rocm/strix-halo/20260718t223203-rocm-strix-halo-greedy-short-c1-32-sourcepair-v1.vllm.json`

| Concurrency | Kiln output tok/s | vLLM output tok/s | vLLM / Kiln |
|---:|---:|---:|---:|
| 1 | 5.91 | 3.72 | 0.63x |
| 8 | 8.39 | 18.04 | 2.15x |
| 16 | 7.49 | 34.36 | 4.59x |
| 32 | 7.12 | 51.59 | 7.25x |

This pair is a retained counterexample, not an accepted competitiveness claim.
The prompt bytes and 166-token input counts matched exactly, but every
concurrency had a different greedy output-set hash, so the vLLM receipt fails
the required exact-output comparison. In addition, the guard intentionally
introduced multi-second process stops and both engines recorded zero
SLO-goodput under the declared 5 s TTFT, 250 ms ITL, and 60 s end-to-end gates.
The throughput values still show that vLLM is the recommended high-concurrency
backend for this currently measured ROCm workload; they do not establish an
unpaced latency region. Kiln's lifecycle peaked at 92.875 C, only 0.125 C below
the hard limit, while vLLM peaked at 86.625 C. Diagnose the output divergence
and establish a continuous thermally sustainable policy before promotion or a
broader performance claim.

The source-paired driver-v7 c1 diagnostic receipts are:

- `benchmarks/receipts/rocm/strix-halo/20260718t232632-rocm-strix-halo-greedy-c1-divergence-v1.kiln.json`
- `benchmarks/receipts/rocm/strix-halo/20260718t232632-rocm-strix-halo-greedy-c1-divergence-v1.vllm.json`

Both arms use exact clean source `1434e0d11`, retain bounded full output
evidence, complete 64 tokens, and have empty reasoning output. They agree on
the first three generated tokenizer tokens: `1206` (`To`), `5517`
(` establish`), and `264` (` a`). The visible content first differs at UTF-8
byte 15 and generated token index 3: Kiln selects token `25045` (` baseline`),
while vLLM selects token `15787` (` foundation`). This rules out an aggregate
hash artifact and localizes the remaining question to the fourth greedy decode
choice; it does not establish which engine is numerically correct. An
independent eager Transformers/PyTorch next-token oracle is still required.

The c1 request windows reported 6.21 output tokens/second for Kiln and 3.83 for
vLLM, but both had zero SLO-goodput and roughly 2.8 to 3.0 second p99 ITL due to
host pacing. The vLLM lifecycle spent 162.63 active seconds in
`torch.compile`, 30 active seconds in graph capture, and 748.90 wall seconds
thermally paused. These diagnostic throughput values must not replace the
source-paired c1-32 table above.

A repaired-profile rerun is retained at
`benchmarks/receipts/rocm/strix-halo/20260719t052911-rocm-strix-halo-greedy-c1-rmsnorm-repair-failed-v1.kiln.json`.
It binds clean source `b392a74dc`, the current ROCm release binary, the same
model and workload fingerprints, and the same vLLM reference arm. The request
completed 64 tokens, but an offline exact comparison still differs at UTF-8
byte 15: Kiln emits `baseline` where vLLM emits `foundation`. The receipt is
also failed because post-measurement thermal inertia crossed the 93 C hard
limit and the guard terminated the owned server. Its 4.26 output tokens/second
is diagnostic-only. A complete request does not override either an output
mismatch or a failed thermal lifecycle.

Before repeating a long request, isolate the only known execution-policy
difference between the correct eager layer oracle and the failed public server.
The tracked launch contract
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-graph-disabled-v1.json`
is equivalent in operator intent to the ordinary Kiln comparison arm except
that its source-bound TOML sets
`accelerator.rocm_graph_mode = "disabled"`. Run only the first four greedy
tokens for the pinned request. Token index three must be `15787`
(` foundation`); `25045` (` baseline`) reproduces the serving defect. This is
a correctness discriminator, not throughput evidence. Do not compare its
four-token workload to the retained 64-token vLLM receipt, and do not use the
result to qualify graph execution, concurrency, or endurance.

The graph-disabled result is retained at
`benchmarks/receipts/rocm/strix-halo/20260719t060940-rocm-strix-halo-greedy-c1-graph-disabled-counterevidence-v1.kiln.json`.
Its measured prompt has the same 163-token count and
`sha256:faf24ebb93fc7e75a2e78111b921a32aece716d56d9b67d2907ae251217a8d9e`
prompt-set hash as both 64-token arms. It still emits `To establish a baseline`
instead of vLLM's `To establish a foundation`, excluding ROCm graph execution
as the cause of this first divergence. The warmup emitted `foundation`, but it
was a distinct 160-token prompt with a different prompt hash; it is not
same-request statefulness evidence.

Driver v8 contained the complete lifecycle: the server guard and both model
fingerprints recorded zero trips, the server lifecycle peaked at 60 C, every
cooldown completed, and no process, listener, or snapshot remained. The 6.61
output tokens/second request window remains diagnostic-only. The next tracked
arm keeps graphs disabled and changes only `prefix_cache.enabled` to `false`:
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-graph-disabled-no-prefix-cache-v1.json`.
It tests whether cross-request KV/recurrent-state reuse explains the remaining
public-serving divergence.

That result is retained at
`benchmarks/receipts/rocm/strix-halo/20260719t061953-rocm-strix-halo-greedy-c1-no-prefix-cache-counterevidence-v1.kiln.json`.
With graph execution and prefix caching both disabled, the same 163-token
prompt still emits `To establish a baseline`. Prefix-cache KV/recurrent-state
reuse is therefore also excluded from this first divergence. Driver v8 again
recorded zero thermal trips, a sub-60 C complete lifecycle, clean shutdown and
cooldown, and no owned residue. Its 6.74 output tokens/second is
diagnostic-only.

The historical next arm changed only the now-removed `batching.mode` from
`enabled` to `disabled` relative to that result:
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-graph-disabled-no-prefix-cache-no-batching-v1.json`.
That launch artifact has been retired from the runnable qualification matrix;
the immutable receipt below remains localization evidence from the source that
still supported the alternate route.

The strict-valid driver-v9 result is retained at
`benchmarks/receipts/rocm/strix-halo/20260719t065243-rocm-strix-halo-greedy-c1-direct-rendezvous-actor-exclusion-v1.kiln.json`.
Its exact 163-token prompt emits `To establish a foundation`, matching HF/vLLM
where the actor-enabled arm emits `To establish a baseline`. The v9 route
record proves the batching actor absent and the direct rendezvous available:
one server `ok`, no request or worker error, three submitted and executed
decode rows, three runner calls, width one, and no runner-call-budget violation.
Graphs and prefix caching remain disabled, so the single changed field makes
the batching actor's prefill/decode path causal for the pinned fourth-token
divergence and excludes the corresponding direct path.

This four-token c1 result is historical localization evidence only. It does not separate
actor prefill from actor decode/state assembly, prove 64-token parity, qualify
the retired actor-disabled configuration for production, or support concurrency,
throughput, latency, or endurance claims. Its 9.40 output tokens/second is
diagnostic-only.

The actor-enabled parent reports three prompt-token chunks, 24 prefill
forwards, 21 layer yields, and 96 completed transformer layers for the measured
163-token prompt. The direct arm does not use this actor prefill route. The
next source-bound discriminator is
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-graph-disabled-no-prefix-cache-prefill-256-v1.json`.
It changes only `server.max_prefill_tokens_per_cycle` from 64 to 256. The
measured prompt therefore fits in one actor token chunk while the unchanged
four-layer quantum still requires eight forwards and seven layer yields.
`foundation` localizes the defect to token-chunk boundaries; `baseline` rules
out that boundary and leaves layer resumption or a later actor transition.
Neither outcome is performance or acceptance evidence. Repair the resulting
subpath and repeat source-paired multi-token parity before returning to the
serving matrix.

The exact-prompt result is retained at
`benchmarks/receipts/rocm/strix-halo/20260719t071900-rocm-strix-halo-greedy-c1-actor-prefill-single-chunk-v1.kiln.json`.
It reuses the parent run ID because that ID participates in fixed-profile
prompt construction: the measured input is again 163 tokens with prompt-set
hash `sha256:faf24ebb93fc7e75a2e78111b921a32aece716d56d9b67d2907ae251217a8d9e`.
The first invocation used a longer run ID, changed the prompt to 167 tokens,
and is deliberately not retained as causal evidence.

With the exact prompt, the actor emits `To establish a foundation`, matching
the direct and HF/vLLM arms. Its server record proves one 163-token prefill
chunk across eight forwards, seven layer yields, and 32 transformer layers,
then three one-row decode forwards. The 64-token actor parent used three chunks
and emitted `baseline`. Layer resumption remains active in both actor arms, so
the changed prompt-token chunk boundary is causal and layer yielding is
excluded for this discriminator. The 9.49 output tokens/second window is
diagnostic-only. The complete guarded lifecycle had zero trips or errors,
peaked at 60.125 C, shut down without force, cooled completely, and left no
process, listener, or snapshot residue.

The product repair is the source-bound
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-v2.json`
candidate. Relative to the historical v1 production input, its only TOML
change is `server.max_prefill_tokens_per_cycle = 256`. The implementation also
makes 256 the ROCm direct-streaming crossover and base/tape tile, then exposes
and enforces the backend-owned
`actor_prefill_tile_alignment_required` contract. With the actor effective,
startup rejects a mismatched actor ceiling, a direct-streaming crossover later
than the first tile, or a combined budget smaller than the tile plus effective
decode width. The historical 64-token actor profiles and receipts remain
immutable counterevidence; they are expected to fail startup with a repaired
binary unless their actor is disabled. V2 is a static candidate until an exact
rebuilt-binary hardware arm reproduces `foundation`, followed by multi-token
and concurrent parity.

### ROCm actor-prefill layer-yield screening

The repaired v2 route still yields actor-owned prefill after every four of the
model's 32 transformer layers. The current development-soak counterexample
shows lower GPU occupancy, 46.1 percent lower aggregate output throughput, and
far more slow actor-prefill phases than the immediately preceding accepted
run. A guarded four-arm discriminator tests whether repeated actor dispatch and
resumption is causal:

| Arm | Launch record | `max_prefill_layers_per_cycle` |
| --- | --- | ---: |
| control | `qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-prefill-layers-4-v1.json` | 4 |
| candidate | `qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-prefill-layers-8-v1.json` | 8 |
| candidate | `qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-prefill-layers-16-v1.json` | 16 |
| candidate | `qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-prefill-layers-32-v1.json` | 32 |

The control launches the production-v2 TOML. The other three TOMLs differ
from it in exactly that one typed field, enforced by a qualification-tooling
test. The historical driver-v10 arms reused one run ID because that version
coupled prompt and log identity. New v16 arms use a distinct `--run-id` per arm
and one shared `--prompt-set-id`, making request bodies and sampling seeds
identical without risking a log collision. Run them serially from one clean pushed
source and one immutable runtime artifact with the same model fingerprint,
`mixed` profile, concurrency rows, output length, thermal policy, and memory
limit.

This is a screening experiment, not production qualification. Any arm with an
output mismatch, request or route error, graph failure or fallback, thermal
trip, forced shutdown, source drift, or residue is rejected regardless of
speed. Compare request-window and thermally sustainable aggregate output
throughput, TTFT/ITL tails, prefill forwards and layer yields, decode batching,
GPU memory, graph counters, and pacing. Promote only a Pareto improvement that
reduces dispatch work without hiding latency behind thermal pauses; confirm the
chosen arm against the four-layer control before changing the production
default or resuming the longer soak.

The first exact-source control attempt on `11057981f` completed its eight-way
mixed row but did not produce a receipt and is not performance evidence. All
eight requests produced 256 tokens at a printed 1.98 aggregate tokens/second;
p99 TTFT/ITL were 113.3/6.55 seconds, and native decode occupancy reached only
two rows with a 1.09-row mean. During the following twelve-way row, the c32
host policy stopped the server at 63.25 C and could not reach its 45 C resume
gate. The process remained stopped for 453.167 seconds; Tctl stayed near 52 C
and the DRM busy counter reported 100 percent. An operator interrupt exercised
the owned cleanup path: the server resumed, handled `SIGTERM`, removed its
private snapshot, drained graph state, and the controller cooled the host to
44.875 C with no process residue. The bound 96,380-byte server log has SHA-256
`d2bdd83287d082e506a95174cc4dcb099923ac6c14acd69dfbbb889bd4e14cc0`.

This attempt exposed a controller defect rather than a layer-quantum result.
The synchronous settlement timeout was unreachable while the workload thread
waited for the paused requests. The shared thermal monitor now owns the same
typed timeout and can terminate plus release a stopped process group without
waiting for request completion. Screening restarts with a bounded c8 row from
the corrected clean pushed source; c12 is not retried until the c8 arms identify
a viable candidate and the watchdog behavior is retained in a failed receipt
if another resume gate becomes unreachable.

The corrected `587fb31ee` c8 screen and its control/candidate repeats are
retained as driver-v10 receipts:

| Layers/run | Receipt verdict | Requests/tokens | Request-window / sustainable tok/s | p99 TTFT / ITL | Prefill forwards / yields | Decode mean / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 A | passed | 8/8, 256 | 2.075 / 1.971 | 107.978 / 6.549 s | 448 / 392 | 1.09 / 2 |
| 4 B | passed | 8/8, 256 | 2.001 / 2.001 | 105.805 / 6.574 s | 448 / 392 | 1.11 / 2 |
| 8 | passed locally | 8/8, 256 | 2.056 / 2.056 | 103.396 / 6.309 s | 224 / 168 | 1.36 / 3 |
| 16 | failed | 7/8, 224 | 2.037 / 2.037 | 88.568 / 6.317 s | 111 / 55 | 1.55 / 3 |
| 32 A | passed locally | 8/8, 256 | 2.460 / 2.460 | 88.441 / 6.690 s | 60 / 0 | 2.38 / 6 |
| 32 B | passed locally | 8/8, 256 | 2.321 / 2.321 | 89.025 / 6.560 s | 60 / 0 | 2.43 / 6 |

Every run binds source tree
`sha256:055ea6a94998d43a7aec55c98bbfb163bccfb7fd2d8111c57c872929fdf364f1`,
runtime
`sha256:99c3bb5cf565ec4ebf60ab7d5489d990b29290d41d70c51d979b8a5e0ba7fbe6`,
model content
`sha256:b1f2cce5baf9e662fc4c2c421b79fccc8787521e6e85810eebffbb4aa43b309f`,
and prompt set
`sha256:d5217da422b553e6488a46f917af8ee82773e69a67a19adf6a7ecc525bdf4223`.
All six validate strictly. The 16-layer arm lost one usage record, reported one
batching error, and recorded one `critical_memory_pressure` graph fallback, so
it is unconditionally rejected.

The two 32-layer runs average 17.3 percent more request-window throughput and
20.3 percent more thermally sustainable throughput than the two controls.
Their mean p99 TTFT is 17.0 percent lower, mean p99 ITL is 1.0 percent higher,
and both have zero request, graph, thermal, or lifecycle failure. This is a
repeatable performance signal, but not a production promotion. The two
unchanged four-layer controls reproduce only two of eight greedy output hashes;
the two 32-layer repeats reproduce only three of eight. Cross-arm output drift
therefore cannot be attributed to the layer quantum, and the locally passed
candidate receipts do not constitute a correctness comparison.

The experiment-level verdict keeps four layers as the production input and
promotes 32 layers only to a diagnostic candidate. Before another throughput
matrix, retain full synthetic outputs for repeated controls, locate the first
divergent token and its prompt/batch route, and compare it with an independent
HF/vLLM reference. This must distinguish harmless near-tie numerical variation
from request-state or batching corruption. Only a correctness-stable 32-layer
candidate may advance to a longer mixed-load confirmation.

The first full-output control replay from clean pushed source `a4bb04cc0` is a
thermal-controller counterexample, not output evidence. Its strict failed
receipt is
`benchmarks/receipts/rocm/strix-halo/20260719t140336-rocm-strix-halo-prefill-layer-control-full-thermal-trip-v1.kiln.json`.
During the c8 row, process-group `SIGSTOP` left the ROCm device reporting 100
percent busy while Tctl held near 49--51 C instead of reaching the 45 C resume
gate. The independent watchdog fired after 300.24 seconds, sent `SIGTERM` and
`SIGCONT`, and completed owned shutdown and cooldown with no process residue.
Only two of eight requests completed, so the run cannot compare outputs or
throughput.

This reproduces the same active-GPU stop failure as the preliminary c12 arm and
invalidates `continuous_process_group_stop` as a serving-qualification pacing
mechanism on this ROCm device. The timeout is retained as necessary
containment, but successful containment does not make the pacing policy valid.
Do not run another serving row under the c32 policy. Replace active-work
`SIGSTOP` pacing with a typed hard-limit-only policy plus pre-launch and
post-exit stable handoff before resuming the full-output comparison. A future
cooperative idle-boundary pacing mechanism may be qualified separately; it
must never suspend a host process while ROCm work is outstanding.

The first hard-limit-only replay from clean pushed source `61c61f7d3` removed
the pause but exposed a second contract defect. The warmup ended at 88.25 C and
the c8 measurement started immediately, leaving only 4.75 C below the 93 C
ceiling. Six requests completed 192 tokens before the 250 ms sampler observed
94 C and terminated the server; the other two streams ended without usage.
There were zero pacing events, shutdown was graceful in 264 ms, the package
cooled to 43.375 C, and no process remained. The strict failed receipt is
`benchmarks/receipts/rocm/strix-halo/20260719t143536-rocm-strix-halo-prefill-layer-control-hard-limit-trip-v1.kiln.json`.

Do not raise the ceiling or interpret the partial 7.94 output tokens/second as
a completed performance result. The driver must reuse the typed stable
temperature target at a verified idle boundary after warmup and around each
measured row, while keeping that cooling time inside thermally sustainable
throughput. Only then may the same c8 full-output control be repeated.

The paced-startup replay completed all 8,411,510,272 source bytes and 32 layers
before readiness, but the following cold-boundary greedy c8 row reached the
unchanged 93 C hard limit after 5.069 measured seconds. The guard terminated the
server cleanly and all eight clients rejected incomplete streams without
terminal usage. This moves the active blocker from startup to concurrent
inference; the mixed profile and longer matrix remain prohibited unchanged.

The next source-bound discriminator is
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-decode-batch-4-v1.json`.
Its TOML differs from production v2 only by changing
`server.max_decode_batch` from 8 to 4. Client concurrency remains eight, as do
the prompt set, 32-token exact-output requirement, four-layer prefill quantum,
256-token prefill tile, graph/cache policy, upload pacing, memory limit, and
93 C guard. Reuse the exact
`rocm-strix-halo-prefill-layer-pacing-c8-v1-greedy-short` run ID so request
bodies and seeds stay fixed. This is a bounded thermal and decode-packing
discriminator, not a production default or a throughput acceptance run. A trip,
incomplete response, output mismatch, graph/route error, forced shutdown, or
residue rejects it and prohibits an unchanged retry.

The first invocation from `f3189b703` did not test decode width: the historical
unlimited initial model-fingerprint worker rose from 41.875 C to 93.125 C and
tripped before server creation. Its guard terminated the worker, cooled for
7.509 seconds to 43 C, and left no process, listener, receipt, or workspace
residue. The width-4 arm remains pending. Its next invocation must use driver
v12's 256 MiB/s model-fingerprint bound and prove both guarded provenance
lifecycles remain below the unchanged hard limit before the inference result is
interpreted.

The exact replay from clean pushed source `75ae28f54` proved both provenance
lifecycles, but still did not test decode width. The initial and final
double-pass workers peaked at 39.25 C and 40.875 C with zero trip. The server
started at 39.375 C, copied its 9,319,828,096-byte private snapshot, loaded
8,022 MB across 32 CPU layers, and reached 93.5 C during the first base-weight
upload group before readiness. The guard sent `SIGTERM`; shutdown was unforced,
the snapshot was removed, stable cooldown ended at 42 C, and no process or
listener remained. The strict failed v12 receipt is
`benchmarks/receipts/rocm/strix-halo/20260719t165036-rocm-strix-halo-greedy-c8-decode-batch-4-v12.kiln.json`.
Its empty warmup and run list make the boundary explicit. Do not retry this
launch unchanged or interpret it as evidence about width four. The next source
change must bound and identify snapshot materialization, CPU checkpoint load,
and the first accelerator upload quantum while preserving the immutable private
snapshot and full checkpoint integrity contract.

The exact corrected-source replay from `2058849e2` proves the startup change
and rejects the width-four inference arm. The three checkpoint-read phases
completed all 9,319,828,096 logical/read bytes, accelerator upload completed all
8,411,510,272 source bytes and 32 layers, and the server reached readiness with
a 59.125 C peak. The c1 warmup passed with a 60.375 C peak. The c8 row then
started from 41.75 C with effective decode width four, reached 93.25 C, and
tripped the unchanged guard. Four requests completed 128 tokens; four rejected
missing terminal usage after the guard's `SIGTERM`. The partial 15.44 output
tokens/second is not performance evidence. Memory remained below the 50 GB
gate, shutdown was unforced with exit zero, cooldown and both fingerprints
completed, and no process, listener, or snapshot remained. The strict failed
receipt is
`benchmarks/receipts/rocm/strix-halo/20260719t175109-rocm-strix-halo-greedy-c8-decode-batch-4-checkpoint-paced-v12.kiln.json`.
Do not repeat width four unchanged. A new source-bound discriminator must lower
or cooperatively regulate concurrent decode work before any mixed or longer
campaign is allowed.

## Running One Profile

```bash
python3 scripts/bench-concurrent-batch.py \
  --engine kiln \
  --base-url http://127.0.0.1:8420 \
  --model Qwen3.5-4B \
  --model-path /absolute/path/to/Qwen3.5-4B \
  --runtime-artifact target/release/kiln \
  --run-id rocm-greedy-short-kiln-attempt-001 \
  --prompt-set-id rocm-greedy-short-comparison-v1 \
  --workload-profile greedy-short \
  --sizes 1,8,16,32,64,128 \
  --repeats 3 \
  --max-tokens 64 \
  --memory-source drm \
  --memory-path /sys/class/drm/card1/device/mem_info_vram_used \
  --memory-limit-bytes 50000000000 \
  --model-fingerprint-read-mib-per-second 256 \
  --host-thermal-policy qualification/host-policies/strix-halo-serving-benchmark-hard-limit-v1.json \
  --server-launch-config qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-v1.json \
  --output-evidence hashes \
  --out .qualification/serving/greedy-short.kiln.json
```

The driver refuses a dirty source tree, an existing receipt or log, a changed
runtime/model/source identity, missing memory evidence, incomplete rows,
unsettled pacing, guard errors, forced shutdown, process residue, or failed
cooldown. Failed rows completed before a later measurement or finalization
failure remain in a self-hashing failed receipt.

Once an owned child has been created, driver v6 also retains failures before the
first request. Readiness exit or timeout, listener ownership mismatch, absent
model ID, malformed `/v1/models`, and Kiln health/identity failure become a
structured `server_startup` completion failure. The failed receipt records an
empty available-model list when discovery never completed, no warmup or rows,
the exact launch and prelaunch-cooldown evidence, server-log hash, observed exit
status, process-group residue check, thermal trip/cooldown evidence, and all
independent source/model/runtime finalization checks. A Kiln startup that never
returned health records a null execution identity only together with an
explicitly failed `execution_identity_unchanged` check. Pre-process validation
failures still return without a receipt because no measured lifecycle began.

## Running the Five-Profile Campaign

`scripts/run-serving-benchmark-campaign.py` defines all five profiles. In owned
mode each runnable profile gets a fresh process group and a run-specific log.
This avoids cross-profile allocator/cache state and keeps startup and teardown
independently auditable. Profile receipts are staged outside the output tree, so
an output directory inside the repository cannot make later profiles reject the
source as dirty. After execution, every staged receipt is published through an
atomic rename and the self-hashing campaign summary is published last.

Campaign summary v8 records the campaign's stable prompt-set base, selected
output-evidence mode, `reference_role`, and resolved `reference_dir`. Each child
receives an engine-qualified unique run ID and a profile-qualified prompt-set
ID; Kiln and vLLM children for the same profile therefore share prompts but
never a log identity. The role is forwarded explicitly to every child instead
of relying on a driver default. It also records `execution_policy`, and every
expected profile has a closed `status`: `completed` or
`not_run_after_failure`. A completed row records its exit code and receipt hash;
a skipped row records the earlier `blocked_by_profile` and has null exit and
receipt-hash fields. Missing receipts count as failures even when the child exits
zero. `model_fingerprint_read_mib_per_second` records the bounded provenance
read rate forwarded to every child. The summary also records the typed memory
source, path or NVML index/UUID, cadence, and absolute limit forwarded unchanged to
all five profiles.

The default execution policy is fail-fast. The first nonzero child exit or
missing receipt prevents every later profile from starting, while preserving
the failed counterexample and a complete summary of what was not run. This is
the required policy for unattended campaigns and for any accelerator run where
a failure may represent a thermal, memory, process-lifecycle, or device-safety
event. `--continue-after-failure` is an explicit diagnostic override for a
known non-safety failure; do not use it after a thermal trip, guard failure,
forced shutdown, process residue, OOM, or device error.

```bash
python3 scripts/run-serving-benchmark-campaign.py \
  --engine kiln \
  --base-url http://127.0.0.1:8420 \
  --model-path /absolute/path/to/Qwen3.5-4B \
  --runtime-identity kiln-git:$(git rev-parse HEAD) \
  --runtime-artifact target/release/kiln \
  --campaign-id rocm-qualified-kiln-attempt-001 \
  --prompt-set-id rocm-qualified-comparison-v1 \
  --out-dir .qualification/serving/rocm-qualified-v1 \
  --memory-source drm \
  --memory-path /sys/class/drm/card1/device/mem_info_vram_used \
  --memory-limit-bytes 50000000000 \
  --model-fingerprint-read-mib-per-second 256 \
  --host-thermal-policy qualification/host-policies/strix-halo-serving-benchmark-hard-limit-v1.json \
  --server-launch-config qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-v1.json \
  --output-evidence hashes
```

For a vLLM campaign, add `--reference-dir` pointing at the matching Kiln
receipts. That campaign always uses the default `qualification_gate` role; the
wrapper rejects the graph/eager discriminator for vLLM. Profile, prompt,
sampling, model identity, fixed output length, thermal policy content, and
comparison mode must agree.

On either 4090, replace the DRM argument in the examples with the UUID from its
environment receipt:

```bash
--memory-source nvml \
--memory-device-uuid GPU-01234567-89ab-cdef-0123-456789abcdef
```

Use the same UUID, sample cadence, and memory limit for the Kiln and vLLM
campaign. The child v19-v21 receipts bind the resolved index and UUID; the campaign
records the requested selector. An explicit physical index remains available
for a stable, non-remapped single-tenant host.

For a Kiln graph campaign paired to eager Kiln receipts from the exact same
runtime artifact, add both:

```bash
--reference-role same_artifact_graph_eager_discriminator \
--reference-dir .qualification/serving/rocm-eager-reference
```

The wrapper rejects a Kiln reference directory without that role and rejects
the role without a reference directory. The driver then independently proves
the same binary, eager reference execution, candidate graph execution, and
first-launch parity for every measured row. Cross-process output differences
remain in each receipt's comparison object but do not impersonate an in-process
graph-corruption result. Performance claims should use median and tail
distributions across repeats, include thermally sustainable throughput, and
keep failed or unsafe diagnostic runs visibly separate.

## Receipt Validation

Committed receipts can be checked without accelerator access:

```bash
mapfile -d '' receipts < <(
  find benchmarks/receipts -type f -name '*.json' -print0 | sort -z
)
python3 scripts/bench-concurrent-batch.py --validate-receipt "${receipts[@]}"
```

Driver v21 is the current contract. It retains v20's parent-bound external
WSL2 thermal/scope mode and adds retrying owned readiness plus authenticated
active-time startup and shutdown bounds. Driver v20 retained v19's typed DRM/NVML
whole-device memory contract and added the parent-bound external WSL2
thermal/scope mode described above. Driver v19 retained v18's actor-only
diagnostics and added typed memory telemetry, stable NVML identity, fail-closed
sampling, and same-device comparison. Driver v18 retained v17's
closed ROCm batch-capture parity evidence and fail-closed reference roles, then
replaced route-aware diagnostics with the actor-only v7 schema described above.
Driver v17 retained
v16 unique run/log identity and stable prompt-set identity. Driver v16 verifies every retained prompt-set
hash against `prompt_set_id` and excludes only the operational run ID from the
workload comparison fingerprint. Driver v15 added strict per-request Kiln terminal performance
evidence and derived phase distributions without changing the common request
body, workload, output contract, or thermal limits. Driver v14
added explicit multi-row ROCm graph fallback accounting and diagnostics v5.
Driver v13 added closed cooperative
actor-cycle idle policy and measured-window accounting. Driver v12 added a bounded,
receipt-recorded model-fingerprint read rate without changing the double-read
integrity contract. Driver v11 added typed idle-boundary
cooldown evidence to v10. Driver v10 added closed ROCm graph execution evidence;
v9 added historical route-aware batching-actor and direct-rendezvous
diagnostics; and v8 added mandatory initial and final guarded model-fingerprint
lifecycles. A v17 through v21 graph/eager discriminator requires a compatible
eager reference with the same prompt-set ID, model-visible
workload, runtime identity, and exact binary hash. Ordinary v7-v21 references
remain comparison-compatible when their version-appropriate workload
fingerprints match. The current arm must still satisfy v16 prompt identity, v15
request-performance accounting, v14 multi-row
graph-route accounting, v13 actor-cycle idle accounting, v12 fingerprint pacing, v11
idle cooling, v10 graph accounting, v9 routing, and v8 containment. Driver v7 added mandatory ordered
per-request output evidence and
structured mismatch localization to the v6 lifecycle contract.

Driver v6 retains the v5 mandatory post-provenance, pre-process cooldown and
adds the structured startup-failure evidence described above. It also validates
embedded vLLM identity objects by JSON value while continuing to bind the
launcher's exact canonical JSON bytes; sorting the outer receipt cannot change
identity semantics. Owned evidence contains the content-hashed launch document,
absolute server-log fingerprint, shutdown signal/status/timing,
forced-shutdown flag, and process-group liveness. Attached and explicitly
unsafe runs serialize null lifecycle artifacts so ownership cannot be inferred
from missing fields. Historical driver v2 through v19 receipts remain valid
under their original contracts, but do not satisfy current v20 performance
acceptance.
