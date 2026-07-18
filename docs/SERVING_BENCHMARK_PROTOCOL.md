# Serving Benchmark Protocol

Kiln and vLLM serving comparisons use one fail-closed OpenAI-compatible driver:
`scripts/bench-concurrent-batch.py`. The driver sends only streaming
`POST /v1/chat/completions` requests on the measured path. Kiln health requests
are untimed side evidence. A passing receipt binds the source tree, runtime,
model content, workload, memory source, process group, host thermal policy, and
complete lifecycle.

## Server Ownership

Every measured run requires `--host-thermal-policy` and exactly one server
ownership mode:

- `--server-launch-config PATH` is the default for local qualification. The
  driver fingerprints the source, model, and runtime, then requires stable host
  cooldown before process creation. It launches a new process group, arms
  thermal containment before the first readiness request, requires the loopback
  port to be unbound before launch, waits for `/v1/models`, proves the listening
  socket belongs to that process group, runs the workload, verifies the
  live runtime, sends `SIGTERM`, requires the whole process group to disappear,
  monitors the hard limit through exit, completes host cooldown, and hashes the
  server log.
- `--server-pid PID` attaches to an already running process-group leader. The
  guard is armed before `/v1/models` or `/health`. The driver leaves the process
  alive only after the configured safe-handoff temperature is stable.
- `--unsafe-no-host-thermal-guard` exists only to retain diagnostic
  counterevidence. It can never produce a passing receipt.

Owned launch eliminates the manual gap in which model loading or inference
prewarm could run before the guard attached. The pre-launch cooldown also
eliminates a subtler gap: model-content and runtime hashing can heat a shared
CPU/GPU package before the child exists. PID attachment remains useful for
remote orchestration that already owns process creation, but that orchestrator
is responsible for containment and its starting host state before the benchmark
starts.

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
| `startup_timeout_seconds` | Positive total readiness deadline. Individual probes are bounded to two seconds. |
| `shutdown_timeout_seconds` | Positive grace period after process-group `SIGTERM` before containment uses `SIGKILL` and fails the run. |
| `acceptable_exit_codes` | Sorted, unique, non-empty accepted status list. A forced shutdown fails regardless of status. |

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
snapshot roots used by the retained comparison receipts. Other machines must
create their own source-bound inputs rather than treating these absolute paths
or host policy as portable.

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

## Thermal Contract

`kiln.host-thermal-policy.v1` selects exactly one hwmon sensor and defines a
hard termination limit, polling cadence, stop/resume pacing hysteresis,
consecutive stable samples before resume, phase-settlement deadline, and a
stable cooldown target. The controller sends `SIGSTOP` and `SIGCONT` to the
complete server process group. Crossing the hard limit sends `SIGTERM`, records
the trip, and fails the run.

The closed policy fields are:

| Field | Contract |
|---|---|
| `schema` | Exact `kiln.host-thermal-policy.v1` identifier. |
| `id` | Stable portable policy identifier, 3 to 128 characters. |
| `sensor.hwmon_name` | Exact Linux hwmon device name. The selector must resolve once. |
| `sensor.label` | Exact temperature-channel label beneath that hwmon device. |
| `limit_millicelsius` | Hard termination boundary, strictly above the pacing start. |
| `poll_interval_ms` | Positive cadence shared by pre-launch cooling, runtime protection, pacing, and final cooldown. |
| `pacing.start_millicelsius` | Temperature at which the complete process group is stopped. |
| `pacing.resume_millicelsius` | Lower temperature at which resume becomes eligible. |
| `pacing.resume_stable_samples` | Consecutive eligible samples required before `SIGCONT`. |
| `safe_handoff.target_millicelsius` | Maximum temperature accepted before owned process creation and after shutdown, or before returning an attached live process. It cannot exceed the pacing-resume temperature. |
| `safe_handoff.stable_samples` | Consecutive samples required at or below the target. |
| `safe_handoff.timeout_seconds` | Positive deadline for each boundary cooldown. |
| `phase_settlement_timeout_seconds` | Positive deadline for an active pacing interval to settle before or after each workload row. |

For owned runs, the driver resolves and reads the selected sensor only after all
source/model/runtime fingerprinting and launch validation have completed. It
does not call `Popen` until `safe_handoff.stable_samples` consecutive readings
are at or below `safe_handoff.target_millicelsius`. A timeout fails before any
child exists. Driver v5 and later receipts retain the sensor path, policy
values, start, peak, and end temperatures, sample count, stable count, elapsed
time, scope, and completion state for this boundary. The receipt validator
requires those values to match the content-hashed thermal policy and the runtime
guard's sensor.

Cooling remains part of wall-clock service cost. Each row records both request
window output throughput and thermally sustainable throughput including pacing
settlement. The top-level receipt records start, peak, and final temperature;
all pacing durations; guard errors and trips; cooldown duration and samples;
sensor path; process identity; and final liveness.

Owned mode disables new pacing immediately before shutdown, releases any active
stop, and keeps hard-limit monitoring active until the child exits. Attached
mode instead cools the live process to the safe-handoff target before releasing
the operator back to an unguarded server.

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
spread, prompt/output hashes, DRM memory, failures, and Kiln batching deltas.

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

## Running One Profile

```bash
python3 scripts/bench-concurrent-batch.py \
  --engine kiln \
  --base-url http://127.0.0.1:8420 \
  --model Qwen3.5-4B \
  --model-path /absolute/path/to/Qwen3.5-4B \
  --runtime-artifact target/release/kiln \
  --run-id rocm-greedy-short-v1 \
  --workload-profile greedy-short \
  --sizes 1,8,16,32,64,128 \
  --repeats 3 \
  --max-tokens 64 \
  --memory-path /sys/class/drm/card1/device/mem_info_vram_used \
  --memory-limit-bytes 50000000000 \
  --host-thermal-policy qualification/host-policies/strix-halo-serving-benchmark-c32-v1.json \
  --server-launch-config qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-v1.json \
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

`scripts/run-serving-benchmark-campaign.py` runs all five profiles. In owned
mode each profile gets a fresh process group and a run-specific log. This avoids
cross-profile allocator/cache state and keeps startup and teardown independently
auditable.

```bash
python3 scripts/run-serving-benchmark-campaign.py \
  --engine kiln \
  --base-url http://127.0.0.1:8420 \
  --model-path /absolute/path/to/Qwen3.5-4B \
  --runtime-identity kiln-git:$(git rev-parse HEAD) \
  --runtime-artifact target/release/kiln \
  --campaign-id rocm-qualified-v1 \
  --out-dir .qualification/serving/rocm-qualified-v1 \
  --memory-path /sys/class/drm/card1/device/mem_info_vram_used \
  --memory-limit-bytes 50000000000 \
  --host-thermal-policy qualification/host-policies/strix-halo-serving-benchmark-c32-v1.json \
  --server-launch-config qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-v1.json
```

For a vLLM campaign, add `--reference-dir` pointing at the matching Kiln
receipts. Profile, prompt, sampling, model identity, fixed output length, thermal
policy content, and comparison mode must agree. Performance claims should use
median and tail distributions across repeats, include thermally sustainable
throughput, and keep failed or unsafe diagnostic runs visibly separate.

## Receipt Validation

Committed receipts can be checked without accelerator access:

```bash
mapfile -d '' receipts < <(
  find benchmarks/receipts -type f -name '*.json' -print0 | sort -z
)
python3 scripts/bench-concurrent-batch.py --validate-receipt "${receipts[@]}"
```

Driver v6 is the current contract. It retains the v5 mandatory
post-provenance, pre-process cooldown and adds the structured startup-failure
evidence described above. It also validates embedded vLLM identity objects by
JSON value while continuing to bind the launcher's exact canonical JSON bytes;
sorting the outer receipt cannot change identity semantics. Owned evidence
contains the content-hashed launch document, absolute server-log fingerprint,
shutdown signal/status/timing, forced-shutdown flag, and process-group liveness.
Attached and explicitly unsafe runs serialize null lifecycle artifacts so
ownership cannot be inferred from missing fields. Historical driver v2 through
v5 receipts remain valid under their original contracts, but do not satisfy
current performance acceptance.
