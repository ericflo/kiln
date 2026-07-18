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
  driver launches a new process group, arms thermal containment before the first
  readiness request, requires the loopback port to be unbound before launch,
  waits for `/v1/models`, proves the listening socket belongs to that process
  group, runs the workload, verifies the
  live runtime, sends `SIGTERM`, requires the whole process group to disappear,
  monitors the hard limit through exit, completes host cooldown, and hashes the
  server log.
- `--server-pid PID` attaches to an already running process-group leader. The
  guard is armed before `/v1/models` or `/health`. The driver leaves the process
  alive only after the configured safe-handoff temperature is stable.
- `--unsafe-no-host-thermal-guard` exists only to retain diagnostic
  counterevidence. It can never produce a passing receipt.

Owned launch eliminates the manual gap in which model loading or inference
prewarm could run before the guard attached. PID attachment remains useful for
remote orchestration that already owns process creation, but that orchestrator
is responsible for containment before the benchmark starts.

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
the runtime artifact remains the immutable teacher/runtime manifest.

Example local Kiln launch document:

```json
{
  "schema": "kiln.serving-benchmark-server-launch.v1",
  "id": "strix-halo-rocm-qualified-v1",
  "command": [
    "./target/release/kiln",
    "serve",
    "--config",
    ".qualification/serving/rocm-qualified.toml"
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
  --host-thermal-policy qualification/host-policies/strix-halo-serving-benchmark-v1.json \
  --server-launch-config .qualification/serving/rocm-launch.json \
  --out .qualification/serving/greedy-short.kiln.json
```

The driver refuses a dirty source tree, an existing receipt or log, a changed
runtime/model/source identity, missing memory evidence, incomplete rows,
unsettled pacing, guard errors, forced shutdown, process residue, or failed
cooldown. Failed rows completed before a later measurement or finalization
failure remain in a self-hashing failed receipt.

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
  --host-thermal-policy qualification/host-policies/strix-halo-serving-benchmark-v1.json \
  --server-launch-config .qualification/serving/rocm-launch.json
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

Driver v4 receipts add `server_lifecycle`. Owned evidence contains the
content-hashed launch document, absolute server-log fingerprint, shutdown
signal/status/timing, forced-shutdown flag, and process-group liveness. Attached
and explicitly unsafe runs serialize null lifecycle artifacts so ownership
cannot be inferred from missing fields. Historical driver v2 and v3 receipts
remain valid under their original contracts.
