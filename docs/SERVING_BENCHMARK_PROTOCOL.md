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

Driver v8 resolves the selected sensor before model hashing. Each initial and
final fingerprint worker starts blocked on a private gate; the supervisor first
requires a stable prelaunch boundary, attaches the continuous process-group
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
spread, prompt/output hashes, DRM memory, failures, and Kiln server-route
diagnostics.

Driver v7 through v9 retain one ordered `output_evidence` row for every successful
request. Hash-only evidence is the default and includes the combined semantic
output hash, separate reasoning/content hashes, UTF-8 byte counts, completion
tokens, and finish reason. The validator requires these rows to cover exactly
the successful request indices, reproduce the aggregate output-set hash, and
reproduce the run's completion-token total. An aggregate mismatch can therefore
no longer hide whether one request or the entire row diverged.

### Server route diagnostics

Driver v9 replaces the batching-only `server` record with
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

The next tracked arm changes only `batching.mode` from `enabled` to `disabled`
relative to that result:
`qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-graph-disabled-no-prefix-cache-no-batching-v1.json`.
The direct streaming rendezvous retains its typed `auto` backend policy. This
isolates actor-owned prefill/decode scheduling before any lower-level direct
route is disabled.

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

This four-token c1 result is localization evidence only. It does not separate
actor prefill from actor decode/state assembly, prove 64-token parity, qualify
the actor-disabled configuration for production, or support concurrency,
throughput, latency, or endurance claims. Its 9.40 output tokens/second is
diagnostic-only. The next correction must repair or disable the divergent actor
subpath, then repeat source-paired multi-token parity before returning to the
serving matrix.

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

`scripts/run-serving-benchmark-campaign.py` runs all five profiles. In owned
mode each profile gets a fresh process group and a run-specific log. This avoids
cross-profile allocator/cache state and keeps startup and teardown independently
auditable. Campaign summary v3 records the selected output-evidence mode and
forwards it unchanged to every profile.

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
  --server-launch-config qualification/server-launch/kiln-rocm-strix-halo-serving-comparison-v1.json \
  --output-evidence hashes
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

Driver v8 is the current contract. It adds mandatory initial and final guarded
model-fingerprint lifecycles to the v7 output-evidence contract. A v8 exact-output
run may use a strict-valid v7 or v8 reference because the workload, model,
thermal-policy, prompt, and output contracts are unchanged; the current arm must
still satisfy v8 containment. Driver v7 added mandatory ordered per-request
output evidence and structured mismatch localization to the v6 lifecycle contract.
Driver v6 retains the v5 mandatory post-provenance, pre-process cooldown and
adds the structured startup-failure evidence described above. It also validates
embedded vLLM identity objects by JSON value while continuing to bind the
launcher's exact canonical JSON bytes; sorting the outer receipt cannot change
identity semantics. Owned evidence contains the content-hashed launch document,
absolute server-log fingerprint, shutdown signal/status/timing,
forced-shutdown flag, and process-group liveness. Attached and explicitly
unsafe runs serialize null lifecycle artifacts so ownership cannot be inferred
from missing fields. Historical driver v2 through v7 receipts remain valid
under their original contracts, but do not satisfy the current v8 provenance
containment or current performance acceptance.
