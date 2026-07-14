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

Install the backend runtime and the command-line probe named by the workload.
For example, ROCm workloads expect `rocminfo` and `ROCM_PATH=/opt/rocm`; Vulkan
workloads expect `vulkaninfo`. Install the Rust toolchain and fetch dependencies
before entering an offline or network-isolated qualification environment.

Validate the workload contract before spending device time:

```bash
python3 scripts/qualification/workload.py \
  qualification/workloads/environment-v1.json \
  qualification/workloads/correctness-core-v1.json
```

The runner rejects a dirty worktree, an uncommitted workload, missing required
variables, a missing required device, silent skips, and an existing receipt or
raw-run directory. Do not bypass those checks.
The ROCm mixed-load driver also rejects ambient `KILN_*` server controls before
building. Configuration changes must be declared in a committed workload
variant; inherited shell overrides are never silently ignored or accepted as
source-bound evidence.

Source-building ROCm and Vulkan serving workloads never invoke Cargo directly.
Their drivers select an immutable backend build specification, resolve the
requested toolchain, then execute the exact package, binary, feature, locked,
and offline build through `scripts/cargo-bounded.sh` with one job and a 15 GiB
`MemAvailable` floor. ROCm alone receives `ROCM_PATH` and
`KILN_ROCM_ARCHS`; the Vulkan build strips ambient ROCm toolchain variables and
uses only the `vulkan` feature. The wrapper refuses overlapping
Cargo/rustc processes. Because the case retains bubblewrap PID isolation, the
offline build runs as a transient systemd user service rather than attempting
to attach the namespaced Cargo PID to a host scope. The service has an
aggregate `MemoryMax`, host reserve, zero swap, `PrivateNetwork=yes`,
control-group kill, and a hard runtime cap. Ordinary ROCm and Vulkan build
services are capped at 840 seconds with a 900-second caller timeout; the
real-ROCm fault corpus uses 1140 and 1200 seconds respectively. This
60-second ordering ensures systemd can stop and collect the complete cgroup
before an outer qualification timeout can kill the wrapper. The measured server
still runs inside the case's separate network and PID namespaces. The committed
effective build config records the wrapper, job count, floor, execution mode,
private-network requirement, versioned environment policy, both deadlines, and
memory policy. `closed-source-build-v1` retains only Cargo/Rustup homes, the
pinned PATH and ROCm architecture/path, locale, user/home, temporary-directory,
and user-systemd connection variables. It excludes ambient compiler flags,
target directories, credentials, and API tokens before invoking the wrapper;
the wrapper independently applies the same policy when constructing the
service environment. Bounded stderr records the machine-specific
available/reserve/limit values. Do not lower the floor or bypass the wrapper to
obtain a receipt. Let the machine recover memory and rerun from the same clean
commit.

Every serving driver creates a collision-resistant mode-0700 workspace below
`.qualification/serving` (the pressure driver uses
`.qualification/serving-pressure`). Names never depend on the sandbox PID,
because PID namespaces can assign the same PID on every run. Normal teardown
removes the workspace and private model snapshot. An externally interrupted
run can leave ignored payloads behind; confirm that no qualification or Kiln
process references a stale directory before removing that exact directory.

Batching qualification must bind the complete typed startup policy, not only a
legacy actor environment switch. A serving workload that exercises the actor
declares `[batching]` values in its source-bound config, restarts the server,
and attests these exact runtime targets before measurement:

```text
GET /v1/config -> .batching.configuration
GET /v1/config -> .batching.actor_active
GET /v1/config -> .batching.direct_decode_rendezvous
GET /health -> .decode_runtime.batching_configuration
GET /health -> .decode_runtime.batching_engine
GET /health -> .decode_runtime.direct_decode_rendezvous
```

The immutable objects at `/v1/config`
`.batching.configuration` and `/health`
`.decode_runtime.batching_configuration` must be equal. The two actual
direct-rendezvous objects at `/v1/config`
`.batching.direct_decode_rendezvous` and `/health`
`.decode_runtime.direct_decode_rendezvous` must also be equal. `actor_active`
must agree with whether the optional live health `batching_engine` snapshot is
present and enabled; the snapshot is not itself equal to that boolean. The
attestation records mode
intent, backend default, effective selection and source; rowwise and
prefix-aware values and sources; admission quantum intent, backend default,
effective clamp and source; and backend-owned burst admission. A malformed
value, a canonical/deprecated-alias conflict, an unexpected source, or a
missing actor in an actor-required variant fails before device work. The direct
rendezvous policy within `batching.configuration` records configured,
backend-policy, effective, and source values for mode, max batch, wait
microseconds, and mixed sequence lengths. Its sibling status object records the
exact scope plus backend, actor, worker, and route availability. A worker may be
active while the route is unavailable because the actor is active.

Use only canonical mechanically derived names in new workload manifests:
`KILN_BATCHING_MODE`, `KILN_BATCHING_ROWWISE_DECODE`,
`KILN_BATCHING_PREFIX_AWARE_ADMISSION`, and
`KILN_BATCHING_PREFILL_ADMISSION_QUANTUM`, plus
`KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MODE`,
`KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MAX_BATCH`,
`KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_WAIT_US`, and
`KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MIXED_SEQ_LENS`. The eight historical
spellings are compatibility inputs for existing deployments, not qualification
vocabulary. A direct-rendezvous variant must disable the actor, restart, prove
`scope="direct_streaming_greedy_only"` and `route_available=true`, and send only
streaming effectively-greedy requests. Actor, sampled, and non-streaming runs
cannot qualify this fallback path.

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

Vulkan core correctness:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  qualification/workloads/correctness-core-v1.json
```

Model-serving workloads additionally require `--model` with the exact local
model directory and `--model-id` with its public identity. Select each declared
A/B arm explicitly; the manifest, not an ambient environment variable, owns
the effective configuration recorded in the receipt.

The source-bound serving drivers materialize that declared policy as a private TOML
file inside the ignored run directory and start `kiln serve --config <file>`.
Server profile, bind address, model/snapshot/adapter paths, thinking default,
transport bounds, scheduling ceilings, logging, memory reclaim, synchronization,
graph mode, graph entry capacity, and graph byte capacity therefore travel
through the same typed parser and source diagnostics as an operator config.
The process environment is scrubbed of ambient `KILN_*` controls before build
and launch. `memory.kv_autoscale` now carries both enabled and disabled requests.
Ordinary serving arms write `memory.kv_force_blocks = 0`; the dedicated
maintenance arm writes its declared positive target. Health and debug must
report `config_file` provenance for both fields. `KILN_DEBUG_ENDPOINTS=1` is the
only `KILN_*` launch exception: it is the internal qualification capability that
grants the trusted debug readback, not public server policy. `RUST_LOG` remains
the ordinary tracing filter. Qualification rejects every other ambient runtime
control, including the deprecated `KILN_KV_AUTOSCALE` and
`KILN_KV_FORCE_BLOCKS` aliases.

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
complete`; it does not run or claim the synthetic inference prewarm used by
other serving paths.

After one fixed warmup, four thread-barrier waves dispatch concurrency 1, 4, 8,
and 12 with mixed prompt lengths from 16 through 1,024 deterministic words.
All 25 measured requests disable thinking and sampling, ignore EOS, stream
usage and performance metadata, and must finish by the exact 32-token limit.
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
deltas are included.

The gate fails on any request error, non-length or short output, missing actor
timing, adapter identity, device fault, resize/reclaim/graph event, batching
error, failed or at-least-100-ms external-yield synchronization, changed KV
capacity, missing memory sample, unexplained adaptive ITL outlier, or ITL gap
above two seconds. Gaps above 250 ms are always counted as stall evidence even
when they remain below the hard pause gate. The server must drain, exit zero
without force inside the shared 60-second grace period, and leave no private
snapshot payload.

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

Experimental ROCm graph runs expose a closed fallback contract at
`/health.decode_runtime.rocm_graphs.fallbacks`. It reports the total and the
thirteen reason counts (`cold_cache_host_round_trip`,
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
name/revision. Base probes explicitly send `"adapter": null`. The driver then
calls the public unload
endpoint, requires every surface and response header to return to base, and
compares canonical streamed semantic deltas from identical pre-load and
post-unload base requests exactly. Dynamic IDs and creation timestamps are the
only excluded envelope fields. Both barrier-swap reasons must appear in order,
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
counts, adapter bytes, load/unload latency, graph invalidations, resize source
and target blocks, released bytes, coordination wait, mutation duration, and
all failure counters. This gate proves one controlled lifecycle at one source
revision; it does not replace concurrent mixed-load stress, the graduated
concurrency gate, or the long soak.

Command-result evidence is accumulated at each completed boundary rather than
synthesized only after both arms pass. A failed result therefore retains any
completed binary build, copied adapter identity, public load or unload,
semantic-output hash, graph invalidation, physical resize, rejection, and
transition already observed. `arms_started` and `arms_completed` distinguish an
unstarted arm from a partial one. Request failures count actual HTTP/stream
failures only; dirty shutdown and snapshot residue derive from teardown rather
than being invented for every failed case.

The stable serving run also attests the default 64-token prompt-work ceiling
(`server.max_prefill_tokens_per_cycle`), the default four-layer yield ceiling
(`server.max_prefill_layers_per_cycle`), and both startup provenances. Admission
and resumable prefill share the token ceiling after ready decode rows reserve
their tokens. A retained token chunk then yields between transformer-layer
groups without replaying completed layers. The receipt records both effective
values, processed-layer and layer-yield counts, plus cumulative/max actor-phase
times; a run that exercises no inter-layer yield fails. A chunk is charged to
the new-token ceiling exactly once when selected, not again when its retained
final layer completes. Every third prefill dispatch remains round-robin; the
other two may accelerate the shortest tail of at most four token chunks.
The receipt records this bounded-priority count and fails when the mixed
workload does not exercise it. Any ITL outlier remains a failure even when its
phase is explained.
The same run attests an effective decode width of eight, four bounded
short-prefill staging slots, and a total active-request ceiling of twelve in
both health and debug state. It also requires a maximum staged-priority burst
of four before the mandatory global prefill turn. Measurement must record at
least one staging admission, at least one rotating staged-priority forward, and
an observed active width above eight without ever exceeding twelve.
Staged-priority forwards must remain a subset of the bounded short-priority
count. The final cancellation drain requires ordinary decode, prefill, staged
occupancy, and the waiting queue all to reach zero. This proves that the latency
path ran without treating the staging capacity as a wider backend decode batch
or accepting an active prefill as drained.
The pressure peer also requires terminal request-scoped performance metadata.
Its actor queue, slot-admission, and admission-to-first-ready wall durations are
recorded separately and must fit inside TTFT; accumulated model prefill must fit
inside admission plus admitted-prefill wall time. Missing, duplicate,
nonnumeric, or internally impossible phase evidence fails the run. These fields
distinguish active-set saturation from slow admitted prefill before any
scheduler policy is changed.

For the historical dynamic-runtime A/B, run each of `default`,
`autoscale-off`, `graphs-off`, and `both-off` separately. These four arms now
pin `KILN_SERVER_SERVING_PROFILE=experimental` so their requested graph/autoscale
differences retain the semantics they had before stable became the default:

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
`KILN_ACCELERATOR_ROCM_GRAPH_CACHE_MAX_BYTES` names. The shorter graph variables
remain deployment compatibility aliases and are not new evidence vocabulary.

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

The driver builds once, starts one server process with a 12-graph residency
limit, warms ROCm graphs, and fills the bounded prefix cache to its declared
entry/state capacity before recording the post-warmup memory baseline or
starting the 30-minute measurement clock.
It then exercises complete fixed-prompt concurrency cycles, including periodic
cancellation, until GPU-used and server-RSS deltas remain within 64 MiB and 16
MiB respectively for two consecutive cycles. This convergence requires at
least four cycles and fails after eight instead of silently moving a growing
allocator into the baseline. The result retains completed cycle, request,
cancellation, final-delta, and maximum-delta metrics even when convergence
fails before the measured phase begins.

The measured phase keeps that process under the same fixed-output waves at
concurrency 1, 8, and 12 with prompt lengths spanning multiple sequence
buckets. Every fifth wave also cancels a longer request using a unique marker.
Slot prompts repeat across waves so prefix hits and cached-block reuse are
measured. After each wave, the driver requires the engine to drain, every used
KV block to be owned by the prefix cache, zero active cache leases or pending
releases, stable cache residency, zero active graph slots or row timelines,
at most 12 retained graphs and slots, the process to remain alive, and
runtime/debug policy attestations to remain consistent. Idle slots and their
native graphs remain resident for reuse. The final drain requires every
retained slot to be idle, and the measured phase must exercise slot reuse.

The result fails on any request or cancellation error, graph capture/replay
failure, typed eager fallback, backend synchronization failure or 100 ms slow
sync, device-fault signature in either a log message or structured error,
non-finite response error, unexplained ITL outlier, capacity change,
unaccounted block, active cache lease, pending release, dirty shutdown,
snapshot residue, or GPU/RSS peak more than 512 MiB above the post-warmup
baseline. Attributed outliers remain counted for review. The receipt also
records p50/p99/p99.9 TTFT and ITL, graph activity, prefix-cache reuse,
graph/slot residency and reuse, external-yield synchronization, memory
baselines/peaks, request/token counts, and cancellation count. Shutdown must
return zero without force after the decode worker is joined, and snapshot
cleanup must leave no residue.

This `kind: soak` workload is intentionally a non-comparative pass/fail gate,
so its `comparison_policy` is null. Do not use it to claim relative throughput
or latency; use the serving benchmark protocol for those claims. The 30-minute
receipt also does not replace the final 24-hour ROCm phase soak.

Never edit a receipt to make it pass. A failed receipt is useful evidence: keep
it when it identifies a reproducible product defect, fix the defect in a new
commit, and run a new receipt with a new ID.

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
