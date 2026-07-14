# Configuration Reference

This document is the canonical operator reference for Kiln's typed server
configuration. It describes the configuration accepted by the current
`KilnConfig` and `RequestLogConfig` implementations, including current defaults,
validation, working environment-variable aliases, and runtime provenance.

The machine-readable contract is
[`contracts/kiln-config-v1.schema.json`](../contracts/kiln-config-v1.schema.json).
The documentation website renders its complete field reference directly from
that schema. `python3 scripts/check_config_schema.py --self-test` verifies that
the schema, these tables, and `kiln.example.toml` agree.

Kiln's intended public environment-variable naming rule is mechanical:

```text
KILN_<SECTION>_<FIELD>
```

Both the section and field are converted to uppercase snake case. For example,
`server.host` maps to the canonical target `KILN_SERVER_HOST`, and
`request_log.max_file_bytes` maps to
`KILN_REQUEST_LOG_MAX_FILE_BYTES`.

Every fixed field with an environment override implements its canonical name.
The "Working spelling(s) today" column records either that same canonical name
or a temporary compatibility spelling:

- **implemented** means the mechanically derived canonical name works;
- **deprecated compatibility** means a non-canonical alias still works but
  emits a structured startup warning naming the field, alias, and replacement;
- **none** means there is no supported environment override for that field;
- **target only; not implemented** means setting the mechanically derived name
  currently has no effect because the field is intentionally config-file-only.

When a canonical name and deprecated compatibility alias are both present,
Kiln parses both strictly. They are accepted only when they resolve to the same
typed value; a conflict stops startup and names both variables. Values are
never included in compatibility warnings.

Do not assume that an arbitrary `KILN_*` name is public configuration. The
repository contains internal diagnostics, qualification switches, kernel
experiments, and temporary compatibility flags. Only the canonical names and
deprecated compatibility spellings listed here are supported inputs.

The generated [Runtime Environment Inventory](RUNTIME_ENVIRONMENT_INVENTORY.md)
catalogs every direct environment read and mutation in crate-owned source,
classifies its owner boundary, and exposes the remaining typed-config migration
queue. It is an engineering inventory, not a second list of supported settings.

## Resolution and startup

### Config file selection

Kiln selects at most one TOML file, in this order:

1. The explicit path supplied by the command, such as
   `kiln --config /path/kiln.toml serve`.
2. The path in `KILN_CONFIG`.
3. `./kiln.toml`, if it exists in the process working directory.
4. No file; built-in defaults are used.

An explicitly selected file that cannot be read or parsed is a startup error.
`KILN_CONFIG` selects the file; it is not an override for a TOML field and is
not included in the field counts below.

### Value precedence

For fields with a working environment alias, effective startup values resolve
in this order, highest priority first:

1. Typed CLI overrides applied after environment resolution
   (`serve --served-model-id` and `serve --eval-mode`).
2. A canonical environment name or deprecated compatibility alias listed here.
3. The selected TOML file.
4. The built-in default.

The `-v`/`-vv` and `--quiet` logging flags override `logging.level` for the
running command. `RUST_LOG`, when present, takes precedence over that resolved
level when the tracing filter is built.

Environment resolution happens once at startup. Restart the server after
changing a file or environment value. Model, training, and request execution
receive resolved policy rather than re-reading public configuration from the
process environment.

### Strict failure behavior

The top-level document and every typed section reject unknown fields. This
includes `[request_log]`, `[eval]`, `[agent]`, `[teachers]`, and each dynamic
teacher credential. A misspelled key does not fall back to a default. The one
intentional open subtree is `agent.self_improve`, whose structured request body
is validated later by the self-improvement subsystem.

TOML type errors, invalid enum values, malformed URLs, out-of-range values, and
semantically invalid combinations stop startup. Present centralized environment
overrides must be valid UTF-8 and must parse successfully; malformed values stop
startup and the diagnostic names the variable and raw value. The centralized
loader's boolean aliases accept, case-insensitively and with surrounding
whitespace ignored:

```text
true:  true, 1, yes, on
false: false, 0, no, off
```

Optional TOML values are represented by omitting the field; TOML has no `null`
literal. Environment-specific clearing syntax is documented per field.

Validate without starting the server:

```bash
kiln config --file kiln.toml
```

This command uses the same typed loader and environment precedence as serving,
but its pretty output is a summary rather than a complete effective-config
dump.

## Coverage summary

The accepted TOML surface contains 15 top-level sections and 98 fixed leaf
fields. Dynamic `teachers.credentials.<id>` entries add two leaf fields per
credential. Of the 98 fixed fields:

- 93 implement the canonical mechanical environment name;
- 66 also retain one or more deprecated compatibility spellings (69 aliases
  total);
- 5 are config-file-only and have no environment override;
- the 69 aliases include `KILN_DEFAULT_NO_THINK`, the second deprecated
  compatibility spelling for `server.default_thinking_enabled`.

The tables below cover all 98 fixed fields and both dynamic credential fields.
The schema additionally records the accepted deprecated TOML-only
`streaming_prefill.enabled` compatibility field so validators match the loader.

## `[server]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `server.serving_profile` | string enum; `"stable"` | `KILN_SERVER_SERVING_PROFILE` (implemented) | `KILN_SERVING_PROFILE` (deprecated compatibility) | `stable`, `experimental`, or `maintenance`, case-insensitive. Process-lifetime policy; restart required. |
| `server.deterministic` | boolean; `false` | `KILN_SERVER_DETERMINISTIC` (implemented) | `KILN_DETERMINISTIC` (deprecated compatibility) | Enables deterministic tensor behavior and forces the effective concurrent decode width to one. |
| `server.host` | string; `"127.0.0.1"` | `KILN_SERVER_HOST` (implemented) | `KILN_HOST` (deprecated compatibility) | Must be non-empty. Binding beyond loopback exposes an unauthenticated inference and training API; use a trusted network or authenticated reverse proxy. |
| `server.port` | unsigned 16-bit integer; `8420` | `KILN_SERVER_PORT` (implemented) | `KILN_PORT` (deprecated compatibility) | `1..=65535`. |
| `server.request_timeout_secs` | unsigned integer; `600` | `KILN_SERVER_REQUEST_TIMEOUT_SECS` (implemented) | `KILN_REQUEST_TIMEOUT_SECS` (deprecated compatibility) | Must be greater than zero. Bounds a request, including model work and cleanup settlement. |
| `server.terminal_access` | string enum; `"loopback_only"` | `KILN_SERVER_TERMINAL_ACCESS` (implemented) | `KILN_TERMINAL` (deprecated compatibility; boolean spellings map to enabled/disabled) | `loopback_only`, `enabled`, or `disabled`. Compatibility boolean spellings are accepted from the environment. This capability can execute arbitrary code; changing it requires restart. |
| `server.http_send_buffer_bytes` | optional unsigned integer; omitted (`None`) | `KILN_SERVER_HTTP_SEND_BUFFER_BYTES` (implemented) | `KILN_HTTP_SEND_BUFFER_BYTES` (deprecated compatibility) | When set, `1024..=16777216`. Applied to accepted sockets. Startup preflights the listener and rejects an OS read-back smaller than requested. |
| `server.stream_stall_grace_ms` | unsigned integer; `2000` | `KILN_SERVER_STREAM_STALL_GRACE_MS` (implemented) | `KILN_STREAM_STALL_GRACE_MS` (deprecated compatibility) | `10..=2000`. A request retaining KV state with no streaming delivery progress is selected for cancellation after this grace. |
| `server.max_batch_tokens` | unsigned integer; `512` | `KILN_SERVER_MAX_BATCH_TOKENS` (implemented) | `KILN_MAX_BATCH_TOKENS` (deprecated compatibility) | `2..=65536`. Combined decode-plus-prefill token budget for one batching-actor cycle. |
| `server.max_prefill_tokens_per_cycle` | unsigned integer; `64` | `KILN_SERVER_MAX_PREFILL_TOKENS_PER_CYCLE` (implemented) | `KILN_MAX_PREFILL_TOKENS_PER_CYCLE` (deprecated compatibility) | `1..=65536`. Independent new-prompt-token ceiling within the combined actor budget. |
| `server.max_prefill_layers_per_cycle` | unsigned integer; `4` | `KILN_SERVER_MAX_PREFILL_LAYERS_PER_CYCLE` (implemented) | `KILN_MAX_PREFILL_LAYERS_PER_CYCLE` (deprecated compatibility) | `1..=1024`. Number of transformer layers an in-flight prefill chunk may execute before yielding to decode. |
| `server.max_decode_batch` | `"auto"` or unsigned integer; `"auto"` | `KILN_SERVER_MAX_DECODE_BATCH` (implemented) | `KILN_MAX_DECODE_BATCH` (deprecated compatibility) | `auto`, `backend`, and `backend_policy` all delegate to backend policy; an integer must be `1..=65536`. Deterministic mode and `max_batch_tokens` may lower the final width. |
| `server.eval_mode` | boolean; `false` | `KILN_SERVER_EVAL_MODE` (implemented) | `KILN_EVAL_MODE` (deprecated compatibility) | Enables deterministic eval-serving defaults, headers, adapter warnings, and transient-cache cleanup behavior. `serve --eval-mode` applies a typed override after environment resolution and wins without mutating process environment. |
| `server.default_thinking_enabled` | optional boolean; omitted (`None`) | `KILN_SERVER_DEFAULT_THINKING_ENABLED` (implemented) | `KILN_DEFAULT_THINKING_ENABLED` (deprecated compatibility); `KILN_DEFAULT_NO_THINK` is a deprecated presence-only alias | `None` preserves the model template default. Requests may override with `chat_template_kwargs.enable_thinking`. Without the canonical spelling, any presence of `KILN_DEFAULT_NO_THINK`, even a value such as `0`, first selects `false`; the explicit `KILN_DEFAULT_THINKING_ENABLED` value is applied afterward and wins. When the canonical spelling is present, every present compatibility alias must resolve to the same boolean. There is no environment spelling that restores `None`. |
| `server.default_thinking_budget_tokens` | optional unsigned integer; omitted (`None`) | `KILN_SERVER_DEFAULT_THINKING_BUDGET_TOKENS` (implemented) | `KILN_DEFAULT_THINKING_BUDGET_TOKENS` (deprecated compatibility) | Integer values include `0`, which closes thinking immediately. Case-insensitive `unlimited` clears a TOML limit back to `None`. Requests may inherit, replace, or explicitly disable the limit. |
| `server.default_thinking_budget_ms` | optional unsigned integer; omitted (`None`) | `KILN_SERVER_DEFAULT_THINKING_BUDGET_MS` (implemented) | `KILN_DEFAULT_THINKING_BUDGET_MS` (deprecated compatibility) | Integer values include `0`. `unlimited` clears a TOML limit. The clock starts at the first decode candidate, after queueing and prefill. The first token or time limit reached forces the model's closing `</think>` sequence. |
| `server.fold_reasoning_into_content` | boolean; `false` | `KILN_SERVER_FOLD_REASONING_INTO_CONTENT` (implemented) | `KILN_FOLD_REASONING_INTO_CONTENT` (deprecated compatibility) | Copies separated reasoning into `content` for compatibility clients. A request can override it. |
| `server.chat_performance_metadata` | boolean; `false` | `KILN_SERVER_CHAT_PERFORMANCE_METADATA` (implemented) | `KILN_CHAT_PERFORMANCE_METADATA` (deprecated compatibility) | Default for chat response performance metadata; requests can override with `include_performance`. |
| `server.chat_config_hash_metadata` | boolean; `false` | `KILN_SERVER_CHAT_CONFIG_HASH_METADATA` (implemented) | `KILN_CHAT_CONFIG_HASH_METADATA` (deprecated compatibility) | Default for chat response config hashes; requests can override with `include_config_hashes`. |
| `server.slow_request_warn_secs` | unsigned integer; `30` | `KILN_SERVER_SLOW_REQUEST_WARN_SECS` (implemented) | `KILN_SLOW_REQUEST_WARN_SECS` (deprecated compatibility) | `0` disables slow-request warnings; otherwise a request at least this old emits a structured warning. |
| `server.shutdown_timeout_secs` | unsigned integer; `5` | `KILN_SERVER_SHUTDOWN_TIMEOUT_SECS` (implemented) | `KILN_SHUTDOWN_TIMEOUT_SECS` (deprecated compatibility) | Must be greater than zero. Hard ceiling for graceful drain before forced exit. |

### Serving profiles

The profile owns GPU mutation and admission policy; individual requests cannot
override it.

| Profile | Inference | Training GPU ownership | Adapter weight transitions | Dynamic KV resize / allocator reclaim | Live graph capture | Exclusive behavior |
|---|---|---|---|---|---|---|
| `stable` | admitted | rejected | rejected | disabled | disabled | reject |
| `experimental` | admitted | allowed | allowed | enabled | enabled | writer priority |
| `maintenance` | not admitted | allowed | allowed | enabled | disabled | drain, then exclusive |

`memory.cuda_graphs = true` does not override the profile: the default `stable`
profile selects eager-only model-runner options because live graph capture is
disabled.

## `[accelerator]`

This section owns process-lifetime accelerator execution behavior that must be
fixed before the primary device context or model runner is created. The
resolved object uses schema `kiln.accelerator-runtime-policy.v2`. Startup,
`kiln config`, `GET /v1/config`, `/health`, trusted debug state, and the
dashboard all report the same configured/effective/source values; lower model,
tensor, and kernel paths do not re-read these public environment names.

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `accelerator.rocm_synchronization_mode` | string enum; `"legacy_host_barriers"` | `KILN_ACCELERATOR_ROCM_SYNCHRONIZATION_MODE` (implemented) | none | `legacy_host_barriers` or `stream_ordered`, case-insensitive. `stream_ordered` requires `server.serving_profile = "experimental"`; other profiles fail startup rather than silently weakening the request. Restart required. |
| `accelerator.rocm_graph_mode` | string enum; `"profile"` | `KILN_ACCELERATOR_ROCM_GRAPH_MODE` (implemented) | `KILN_ROCM_GRAPHS` and `KILN_ROCM_GRAPH_CAPTURE` (deprecated compatibility) | `profile`, `disabled`, `warmup_then_eager`, or `lazy_capture_replay`, case-insensitive. `profile` resolves to `disabled` under stable/maintenance and `lazy_capture_replay` under experimental. The two explicit non-disabled modes require the experimental profile. Restart required. |
| `accelerator.rocm_graph_cache_entries` | unsigned integer; `8` | `KILN_ACCELERATOR_ROCM_GRAPH_CACHE_ENTRIES` (implemented) | `KILN_ROCM_GRAPH_CACHE_MAX` (deprecated compatibility) | `1..=64`. Bounds retained native graph entries in every product and embedding constructor; zero or unbounded capacities are rejected. Restart required. |
| `accelerator.rocm_graph_cache_max_bytes` | unsigned integer bytes; `1073741824` (1 GiB) | `KILN_ACCELERATOR_ROCM_GRAPH_CACHE_MAX_BYTES` (implemented) | none | `67108864..=17179869184` (64 MiB through 16 GiB). Independently bounds requested physical bytes retained by graph-owned stable tensors, capture arenas, private-stream hipBLASLt workspaces, and owner slot state. Opaque HIP graph/exec/stream/event overhead is counted as objects and remains subject to live driver-pressure policy. Restart required. |

### ROCm synchronization semantics

`legacy_host_barriers` is the qualified default. It preserves the historical
device-wide barriers around large hipBLASLt results and BF16/F32 cast
boundaries, the historical active-stream waits after eager tensor operations,
and a device-wide external-yield drain before generated progress is published.
Choosing it is intended to produce the same ordering as the pre-policy runtime,
with the new reasoned counters added.

`stream_ordered` is an explicit local-qualification mode. It omits only
barriers whose call site asserts that producer and consumer are FIFO-ordered on
the same HIP stream. It does not omit host readback, host-buffer lifetime,
in-place mutation, allocator reclaim, graph transition, error recovery, or
global-state drains. External yield waits for the primary producer stream
rather than the whole device. A replayed graph records completion on its
capture stream and makes the primary stream wait on that event before the
external-yield stream wait, so replay work is included.

The primary ROCm context stores the immutable policy. A second caller asking
for a different policy on the same device receives a startup error naming the
already-installed and requested policies. This prevents two model/runtime
owners from silently mixing synchronization disciplines.

Synchronization telemetry has 23 fixed reason values. `/health` reports device
wait count, stream wait count, waited nanoseconds, and skipped count for every
reason under `decode_runtime.rocm_synchronization`. The same object exposes
`cleanup_quarantined`: when true, a fatal execution or synchronization failure
made device state unsafe. The flag is shared by every context for that device
ordinal and is process-lifetime sticky: a later recovery drain may collect
diagnostics, but cannot re-enable execution or cleanup. New ROCm execution is
rejected, possibly in-flight HIP resources are retained instead of being
destroyed unsafely, backend admission fails closed, and the process must be
restarted. Active ROCm telemetry becoming unavailable is treated as the same
fail-closed health condition rather than as a false quarantine value. The
central server health gate observes and latches this state before adapter,
training, import, or other backend mutation admission, so those paths cannot
slip through before the model runner notices the fault. This is live state, so it is
reported by health and trusted debug rather than `/v1/config`; the latter remains
the immutable policy authority. `/metrics` exports the same
fixed dimensions as `kiln_rocm_synchronizations_total`,
`kiln_rocm_synchronization_wait_seconds_total`, and
`kiln_rocm_synchronization_skipped_total`, plus policy, availability, and
`kiln_rocm_cleanup_quarantined` gauges. Treat the quarantine gauge as unknown
unless `kiln_rocm_synchronization_telemetry_available` is `1`.
Reading these surfaces loads atomics only; it never synchronizes the device or
probes the driver.

`capture_rollback` and `error_recovery` are deliberately different reasons.
After a native capture error, `capture_rollback` may settle the device while
execution admission remains open; eager continuation is allowed only after the
runner also proves that its logical recurrent state was restored. A settlement
failure, a logical-rollback failure, or an armed/unclassified capture guard
leaving scope publishes the process-lifetime STOP state first, then uses
`error_recovery` only as a diagnostic drain. That fatal path remains
cleanup-quarantined even if the later drain succeeds and always requires a
process restart.

Stream-ordered mode is not promoted by configuration alone. Qualify legacy and
stream-ordered runs on the same source tree using token/logit parity,
throughput at concurrency 1/8/16/32/64, p50/p95/p99/max inter-token latency,
reasoned synchronization time, and stable memory plateaus before changing the
default.

### ROCm graph lifecycle

- `disabled` bypasses graph-shaped warmup, capture, and replay.
- `warmup_then_eager` runs one graph-shaped eager warmup, never captures, and
  remains eager. It exists for controlled graph-state-machine comparisons.
- `lazy_capture_replay` warms eagerly and may capture/replay eligible decode
  shapes while serving. Capture/fallback/replay state remains visible in
  `/health`; use this only in the experimental profile.
- `profile` is the recommended default because the serving profile remains the
  authority: stable and maintenance stay eager, while experimental selects
  lazy capture/replay.

Entry count and retained bytes are independent hard limits. Byte accounting
deduplicates physical ROCm allocations by device pointer, counts graph-stable
I/O, freeze-pointer capture arenas, each private stream's actual hipBLASLt
workspace, and persistent recurrent/conv slot state once per owner. Health,
trusted debug, and Prometheus split those categories and report peak bytes,
evictions, admission rejections, and the number of opaque HIP graph, exec,
stream, and event objects. The byte budget deliberately does not pretend that
HIP exposes the allocator overhead of those opaque objects; live driver memory
pressure remains a separate complementary signal.

Before native capture, the runner reclaims eligible idle owners at entry/byte
saturation, performs one settled warm Record pass, measures the candidate's
exact queryable allocation identities, rechecks matching-device pressure, and
atomically reserves global governor headroom. Native capture begins only after
that sequence admits it. A governor device-selector mismatch fails closed rather
than consuming another accelerator's headroom. The Record pass necessarily
allocates the candidate before its exact size is knowable. Phase telemetry
independent of the model and graph-runner locks plus last/peak
transient-candidate bytes make that residual exposure observable. A candidate
that alone exceeds the byte budget or cannot
be accounted exactly makes its geometry non-capture-safe for this runner.
Aggregate byte-budget suppression clears after ownership or budget relief;
global-reservation denial retries after the matching device reports enough
cached headroom. A selector mismatch remains fail-closed rather than consuming
another accelerator's budget. None repeats a full warm pass on every token.

Ordinary budget eviction is deterministic least-recently-used eviction of
idle owners as a unit, including every graph for the owner and its retained
slot state. It never evicts an active owner merely to admit another graph.
Moderate pressure permits retained replay but blocks cache growth. Tight
pressure additionally reclaims eligible idle owners while preserving an active
cache hit. Critical or unavailable pressure disables replay,
reclaims all graphs only after device settlement, and uses eager execution only
if both pre-drop and post-drop settlement prove cleanup safe. A destructor,
async-free, or settlement failure activates the process-lifetime device
quarantine; no reclaimed-byte success is reported and no eager retry follows.

### ROCm graph observability contract

`GET /v1/config` and trusted `GET /v1/debug/model-state` expose two independent
live values and a reason beside each one:

- `rocm_graphs` is the nonblocking full cache/counter snapshot, with
  `rocm_graphs_unavailable_reason`.
- `rocm_graph_telemetry` is the current/completed phase and transient-candidate
  snapshot, independent of the model and graph-runner locks, with
  `rocm_graph_telemetry_unavailable_reason`.

Each value is either an object with a null reason or `null` with exactly one of
`backend_without_graph_runner`, `model_runner_busy`,
`model_runner_lock_poisoned`, `graph_runner_busy`, or
`graph_runner_lock_poisoned`. The full snapshot can use any of those five. The
phase handle is stored outside both runner locks, so a real backend retains live
phase telemetry through model-runner and graph-runner contention or poison;
currently only `backend_without_graph_runner` makes that value null. The phase
reason family nevertheless uses the same closed schema. Do not interpret any
null value as an empty cache or zero activity.

Health flattens both authorities into `decode_runtime.rocm_graphs` for an
operational view. Its `state` is `enabled` or `disabled` when full statistics
are present, `busy` only for `model_runner_busy` or `graph_runner_busy`, and
`unavailable` for the no-runner and poisoned-lock reasons. `unavailable_reason`
belongs to the full statistics. `phase_telemetry_available` and
`phase_telemetry_unavailable_reason` independently govern `current_phase`,
`current_phase_elapsed_micros`, five phase summary objects, and last/peak
transient candidate bytes. Every unavailable field is serialized as `null`;
the server never fabricates a zero snapshot.

For a coherent point-in-time full snapshot, `rocm_graphs` also serializes the
five completed phase summaries and last/peak transient bytes. The separate
`rocm_graph_telemetry` object is the authoritative current-phase source and
continues advancing when full statistics cannot acquire either runner lock.
Health sources its phase and transient fields from that separate channel.

The phase names are exactly `pre_candidate_headroom`, `candidate_warm`,
`pre_native_reservation`, `native_capture`, and
`rejected_candidate_cleanup`. Each completed phase object contains `calls`,
`slow`, `total_duration_micros`, and `max_duration_micros`; `slow` means at
least 100 ms. `current_phase_elapsed_micros` uses a monotonic clock while a
phase is active. `native_capture` remains active through capture, the settled
first launch, defensive cache admission/publication, and the blocking committed
governor debit. `rejected_candidate_cleanup` begins only when an unretained
candidate enters destruction and settlement. `last_transient_candidate_bytes` and
`peak_transient_candidate_bytes` measure the exact deduplicated queryable bytes
allocated by the measured pre-admission candidate, excluding already-owned
recurrent slot state and opaque native objects. They are not retained-cache
bytes and are not governed as if they were already published.

The eager-fallback reasons are exactly `cold_cache_host_round_trip`,
`persistent_host_round_trip`, `shape_dependent_attention`,
`graph_cache_capacity`, `graph_cache_byte_budget`,
`graph_accounting_incomplete`, `moderate_memory_pressure`,
`tight_memory_pressure`, `critical_memory_pressure`,
`memory_reservation_denied`, `memory_governor_selector_mismatch`,
`capture_failure`, and `replay_failure`. The full snapshot reports one counter
per reason plus total, slow, total-duration, and maximum-duration values.

Prometheus always emits two availability gauges and two closed one-hot reason
families. A zero `kiln_rocm_graph_telemetry_available` is explained by
`kiln_rocm_graph_snapshot_unavailable{reason}` and omits only the full
cache/counter families. A zero
`kiln_rocm_graph_phase_telemetry_available` is explained by
`kiln_rocm_graph_phase_telemetry_unavailable{reason}` and omits only current
phase, phase latency, and transient-candidate families. Both reason labels use
the same five-value set above. Phase availability remains one for a real backend
while full snapshot availability is zero during either runner-lock contention
or poison. Full
families cover state; cache, slot, owner, retained-byte and opaque-object
gauges; admissions, evictions and their four causes; three post-capture
rejection reasons; five pre-capture skip reasons; capture/replay outcomes; and
all 13 fallback reasons and latency. Live families cover the one-hot current
phase, active elapsed seconds, calls/slow/total/max duration for all five
phases, and last/peak transient bytes. Every label set is closed; request,
shape, allocation, and configured-byte values never become labels.

Graph-stable paged metadata is now a correctness invariant rather than a knob.
The runtime no longer reads `KILN_ROCM_GRAPH_STABLE_PAGED_METADATA`, graph
cache/capture flags, matmul synchronization thresholds, or full-attention
handoff flags inside decode. The three historical graph variables listed in
the table are parsed once by the typed startup loader. Setting both historical
graph-mode aliases is rejected even when their booleans happen to agree; use
the canonical enum to state one unambiguous lifecycle.

## `[batching]`

These values are resolved once, after backend selection and after
`server.deterministic`, `server.max_decode_batch`, and
`server.max_batch_tokens` determine the effective decode width. Every change
requires a process restart; none is a live tuning control.

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `batching.mode` | string enum; `"auto"` | `KILN_BATCHING_MODE` (implemented) | `KILN_BATCHING_ENGINE` (deprecated compatibility) | TOML accepts `auto`, `enabled`, or `disabled`, case-insensitively. Environment input also accepts the strict boolean spellings, mapping true to `enabled` and false to `disabled`. `auto` delegates actor selection to the immutable backend policy; an explicit value wins. Restart required. |
| `batching.rowwise_decode` | boolean; `false` | `KILN_BATCHING_ROWWISE_DECODE` (implemented) | `KILN_BATCH_DECODE_ROWWISE` (deprecated compatibility) | `false` sends each ready cohort through one true batched forward. `true` issues one forward per row while retaining actor ownership; it is an emergency correctness comparison, normally reduces throughput, and does not increase the effective decode width. Restart required. |
| `batching.prefix_aware_admission` | boolean; `true` | `KILN_BATCHING_PREFIX_AWARE_ADMISSION` (implemented) | `KILN_BATCH_PREFIX_AWARE_ADMISSION` (deprecated compatibility) | When true, a queued same-adapter strict descendant waits while its active shorter prefix can become reusable; independent rows may still be admitted. Disable only for a controlled admission A/B. Restart required. |
| `batching.prefill_admission_quantum` | `"auto"` or unsigned integer; `"auto"` | `KILN_BATCHING_PREFILL_ADMISSION_QUANTUM` (implemented) | `KILN_BATCH_PREFILL_ADMISSION_QUANTUM` (deprecated compatibility) | An integer must be `1..=65536` and caps how many queued prompts the actor admits in one cycle before returning to decode. `auto` is case-insensitive and uses the backend policy below. The selected value is then clamped to `1..=effective max_decode_batch`; the diagnostics name `effective_decode_width` as final authority when it performs that clamp. Restart required. |
| `batching.direct_decode_rendezvous_mode` | string enum; `"auto"` | `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MODE` (implemented) | `KILN_DECODE_BATCHER` (deprecated compatibility) | Selects only the fallback worker used by actor-absent direct streaming effectively-greedy requests. TOML accepts `auto`, `enabled`, or `disabled`, case-insensitively; environment input also accepts the strict boolean spellings. `auto` enables the worker on every real backend. Sampled requests, non-streaming requests, and every route using the batching actor bypass this worker. Restart required. |
| `batching.direct_decode_rendezvous_max_batch` | `"auto"` or unsigned integer; `"auto"` | `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MAX_BATCH` (implemented) | `KILN_DECODE_BATCH_MAX` (deprecated compatibility) | An explicit integer must be `1..=65536`. `auto` uses the backend policy below. Either selection is clamped to the already-resolved effective decode width; diagnostics use `effective_decode_width` when that ceiling wins. Restart required. |
| `batching.direct_decode_rendezvous_wait_us` | `"auto"` or unsigned integer; `"auto"` | `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_WAIT_US` (implemented) | `KILN_DECODE_BATCH_WAIT_US` (deprecated compatibility) | An explicit value is any non-negative `u64` number of microseconds, including `0`; `auto` uses backend policy. A negative, overflowing, malformed, or non-UTF-8 environment value stops startup. In particular, malformed legacy `KILN_DECODE_BATCH_WAIT_US` now fails startup instead of silently becoming zero. Restart required. |
| `batching.direct_decode_rendezvous_mixed_seq_lens` | `"auto"` or boolean; `"auto"` | `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MIXED_SEQ_LENS` (implemented) | `KILN_DECODE_BATCH_MIXED_SEQ` (deprecated compatibility) | TOML accepts only the string `auto` or a native boolean; quoted boolean strings are rejected. Environment input accepts `auto` or the strict boolean spellings. The value controls whether one compatible fallback cohort may contain different decode positions. Restart required. |

### Backend-owned `auto` policy

`auto` preserves backend qualification rather than imposing one cross-device
default. The matrix is evaluated against the final effective decode width, not
the unconstrained backend maximum.

| Backend | `batching.mode = "auto"` | Backend decode width before server constraints | `prefill_admission_quantum = "auto"` | `burst_prefill_admission` diagnostic |
|---|---:|---:|---:|---:|
| CUDA | enabled | 8 | effective decode width | true |
| ROCm | enabled | 8 | 4, clamped to effective width | false |
| Vulkan | enabled | 64 | effective decode width | false |
| Metal | disabled | 8 | 4, clamped to effective width | false |
| CPU/other | enabled | 8 | 4, clamped to effective width | false |

`burst_prefill_admission` is backend-owned and intentionally has no TOML or
public environment field. On CUDA it lets admission refill the decode width in
one actor turn. It is reported with the four primary-actor settings so an observed
CUDA/Vulkan/ROCm difference is not mistaken for a hidden runtime override.

The fallback direct-stream rendezvous has a separate backend-owned `auto`
matrix. Its mode is enabled on every real backend, but a live worker is not the
same as an available request route:

| Backend | `direct_decode_rendezvous_mode = "auto"` | `max_batch = "auto"` | `wait_us = "auto"` | `mixed_seq_lens = "auto"` |
|---|---:|---:|---:|---:|
| CPU/other | enabled | 8 | 0 | false |
| CUDA | enabled | 1 | 0 | false |
| ROCm | enabled | 8 | 0 | false |
| Metal | enabled | 8 | 100 | true |
| Vulkan | enabled | 64 | 5000 | true |

The selected `max_batch` is always clamped to the effective decode width. The
worker is constructed independently of the primary actor so startup and worker
failures remain observable, but its route is available only for a real backend
when the actor is absent and the worker is active. An actor-active process can
therefore legitimately report `worker_active=true` and
`route_available=false`. Under the complete default policy, this makes the
route available only on Metal: every real backend auto-enables the worker, but
Metal is the only backend whose primary actor is auto-disabled. Explicitly
disabling the actor can make the route available on another real backend.

### Effective policy and provenance

`GET /v1/config` reports the exact startup result at
`batching.configuration` and whether an actor was actually constructed at
`batching.actor_active`. A typical ROCm default is:

```json
{
  "batching": {
    "configuration": {
      "mode": {
        "configured": "auto",
        "configured_source": "default",
        "backend_policy_enabled": true,
        "effective_enabled": true,
        "effective_source": "backend_policy"
      },
      "rowwise_decode": { "enabled": false, "source": "default" },
      "prefix_aware_admission": { "enabled": true, "source": "default" },
      "prefill_admission_quantum": {
        "configured": null,
        "configured_source": "default",
        "backend_policy": 4,
        "effective": 4,
        "effective_source": "backend_policy"
      },
      "direct_decode_rendezvous": {
        "mode": {
          "configured": "auto",
          "configured_source": "default",
          "backend_policy_enabled": true,
          "effective_enabled": true,
          "effective_source": "backend_policy"
        },
        "max_batch": {
          "configured": null,
          "configured_source": "default",
          "backend_policy": 8,
          "effective": 8,
          "effective_source": "backend_policy"
        },
        "wait_us": {
          "configured": null,
          "configured_source": "default",
          "backend_policy": 0,
          "effective": 0,
          "effective_source": "backend_policy"
        },
        "mixed_seq_lens": {
          "configured": null,
          "configured_source": "default",
          "backend_policy": false,
          "effective": false,
          "effective_source": "backend_policy"
        }
      },
      "burst_prefill_admission": false
    },
    "actor_active": true,
    "direct_decode_rendezvous": {
      "scope": "direct_streaming_greedy_only",
      "backend_available": true,
      "actor_active": true,
      "worker_active": true,
      "route_available": false
    }
  }
}
```

For explicit values, the configured source is `config_file` or `environment`.
The complete effective-source enum is `default`, `backend_policy`,
`config_file`, `environment`, or `effective_decode_width`. The mode normally
uses the middle three; `default` is reachable only for an explicit
programmatic value carrying default provenance, not for ordinary built-in
`auto`, which resolves through `backend_policy`. The admission quantum uses
`effective_decode_width` when the final decode width lowers the selected value.
A configured quantum or direct-rendezvous width is retained even when clamped,
so intent and execution cannot be confused. The direct policy is nested at
`batching.configuration.direct_decode_rendezvous`; the sibling
`batching.direct_decode_rendezvous` is actual process state, with exact fields
`scope`, `backend_available`, optional `backend_unavailable_reason`,
`actor_active`, `worker_active`, and `route_available`. Mock mode reports
`backend_available=false` and `backend_unavailable_reason="mock_backend"`.

`GET /health` repeats the same immutable object at
`decode_runtime.batching_configuration` beside the optional live
`decode_runtime.batching_engine` snapshot and reports actual fallback state at
`decode_runtime.direct_decode_rendezvous`. The trusted debug response keeps the
existing `batching_engine.enabled` actor-activity flag, adds the immutable
object at `batching_engine.configuration`, and reports actual fallback state at
`batching_engine.direct_decode_rendezvous`. Actor activity can be false even
when configured intent is visible, and worker activity does not prove routing;
clients must use `route_available` for this deliberately narrow direct route.

`kiln config --file <path>` validates and prints all eight `[batching]` startup
values: actor mode, rowwise decode, prefix-aware admission, prefill quantum,
direct rendezvous mode, maximum batch, wait microseconds, and mixed-sequence
policy. It cannot report backend-effective or live route state because it does
not construct a model; inspect `/v1/config` after restart for those facts.

## `[model]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `model.path` | optional string; omitted (`None`) | `KILN_MODEL_PATH` (implemented) | `KILN_MODEL_PATH` | Must be non-empty when set. Omitted starts the server in mock mode; a real model path enables real inference. |
| `model.model_id` | string; `"Qwen/Qwen3.5-4B"` | `KILN_MODEL_MODEL_ID` (implemented) | `KILN_MODEL_ID` (deprecated compatibility) | Must be non-empty. Used for model/tokenizer identity and as the source of the default served id. Kiln still applies its built-in Qwen3.5-4B runtime profile. |
| `model.tokenizer_path` | optional string; omitted (`None`) | `KILN_MODEL_TOKENIZER_PATH` (implemented) | `KILN_TOKENIZER_PATH` (deprecated compatibility) | Must be non-empty when set. |
| `model.adapter_dir` | optional string; omitted (`None`) | `KILN_MODEL_ADAPTER_DIR` (implemented) | `KILN_ADAPTER_DIR` (deprecated compatibility) | Must be non-empty when set. For the Qwen3.5-4B profile, omission resolves to `<model.path>/adapters`. |
| `model.snapshot_dir` | optional string; omitted (`None`) | `KILN_MODEL_SNAPSHOT_DIR` (implemented) | `KILN_MODEL_SNAPSHOT_DIR` | Must be non-empty when set. The environment alias uniquely treats an empty or whitespace-only value as a request to clear the TOML value. Without a value, Kiln tries a location beside the model and then the system temporary directory. |
| `model.served_model_id` | optional string; omitted (`None`) | `KILN_MODEL_SERVED_MODEL_ID` (implemented) | `KILN_SERVED_MODEL_ID` (deprecated compatibility) | Must be non-empty when set. Otherwise the effective id is the final slash-separated component of `model.model_id` (`Qwen3.5-4B` by default). `serve --served-model-id` applies a typed, validated override after environment resolution. |

## `[memory]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `memory.num_blocks` | optional unsigned integer; omitted (`None`) | `KILN_MEMORY_NUM_BLOCKS` (implemented) | `KILN_NUM_BLOCKS` (deprecated compatibility) | Must be greater than zero when set. Omission invokes backend-aware automatic KV-block sizing. |
| `memory.gpu_memory_gb` | optional finite number; omitted (`None`) | `KILN_MEMORY_GPU_MEMORY_GB` (implemented) | `KILN_GPU_MEMORY_GB` (deprecated compatibility) | Must be finite, greater than zero, and representable as bytes. Units are GiB. This is a capacity cap, not a hardware override: it may reduce the detected safe capacity but never expands physical VRAM, host-backed unified memory, or a cgroup-bounded capacity. A request above the safe detected capacity is clamped down. |
| `memory.inference_memory_fraction` | finite number; `0.7` | `KILN_MEMORY_INFERENCE_MEMORY_FRACTION` (implemented) | `KILN_INFERENCE_MEMORY_FRACTION` (deprecated compatibility) | Loader validation accepts `0.0..=1.0`; real-state construction clamps the configured value to `0.1..=1.0` before KV sizing. The remainder is available to the training budget. |
| `memory.training_memory_gb` | optional finite number; omitted (`None`) | `KILN_MEMORY_TRAINING_MEMORY_GB` (implemented) | `KILN_TRAINING_MEMORY_GB` (deprecated compatibility) | Must be finite, greater than zero, and representable as bytes when set. Optional training-budget cap in GiB; it can reduce but never expand the capacity remaining after resident model and KV allocations. |
| `memory.floor_gb` | finite number; `1.0` | `KILN_MEMORY_FLOOR_GB` (implemented) | `KILN_MEMORY_FLOOR_GB` | Must be finite, non-negative, representable as bytes, and strictly smaller than the selected accelerator's effective capacity after `memory.gpu_memory_gb` is applied. Units are GiB. Accelerator startup rejects an equal or larger floor before model upload and reports both configured and effective byte values. The process-wide governor subtracts this additional floor, then outstanding soft reservations, from live free memory when computing allocation headroom. On unified-memory devices it is separate from the physical-memory reserve applied during safe-capacity detection. |
| `memory.probe_ms` | unsigned integer; `500` | `KILN_MEMORY_PROBE_MS` (implemented) | `KILN_MEMORY_PROBE_MS` | Must be greater than zero. Sets the background memory-sampler cadence. Request, inference, health, and metrics paths read only the published sample and never run a driver/OS probe synchronously. Cached admission fails closed when the sample is older than `max(5000 ms, 4 * probe_ms)`, the latest probe failed, or a required sampler is not running. An explicit refresh after a material allocation or release bypasses the cadence. |
| `memory.reclaim_mode` | string enum; `"off"` | `KILN_MEMORY_RECLAIM_MODE` (implemented) | `KILN_MEMORY_RECLAIM_MODE` | Exactly `off`, `on-demand`, or `automatic`, case-insensitive with surrounding whitespace ignored. `off` prevents execution of registered allocator reclaim hooks; `on-demand` permits explicit pressure and allocation-retry reclaim calls; `automatic` also permits the background pressure monitor. The immutable serving profile remains authoritative: a profile with allocator reclaim disabled keeps the effective mode off and does not start the monitor. |
| `memory.kv_autoscale` | boolean; `true` | `KILN_MEMORY_KV_AUTOSCALE` (implemented) | `KILN_KV_AUTOSCALE` (deprecated compatibility) | Requests the pressure-driven physical KV-cache control loop. The serving profile and backend remain authoritative: stable mode and backends without device-resident KV pressure report the request as unavailable rather than silently enabling mutation. `/health`, `/v1/config`, and the trusted debug state expose the request, effective state, bounded reason, and source. |
| `memory.kv_force_blocks` | unsigned integer; `0` (disabled) | `KILN_MEMORY_KV_FORCE_BLOCKS` (implemented) | `KILN_KV_FORCE_BLOCKS` (deprecated compatibility) | A positive value requests one exact startup resize before the normal autoscaler loop. It requires `memory.kv_autoscale=true` and `server.serving_profile="maintenance"`; every other combination fails configuration validation. Zero disables the one-shot operation. The resize still uses full replacement-pool reservation, exclusive GPU ownership, graph invalidation, transactional publication, and typed `forced_configuration` attribution. This is an offline maintenance/qualification control, not a per-request tuning knob. |
| `memory.kv_cache_fp8` | boolean; `false` | `KILN_MEMORY_KV_CACHE_FP8` (implemented) | `KILN_KV_CACHE_FP8` (deprecated compatibility) | Requests E4M3FN KV storage. Backend storage policy may reject or disable the request when unsupported. |
| `memory.cuda_graphs` | boolean; `true` | `KILN_MEMORY_CUDA_GRAPHS` (implemented) | `KILN_CUDA_GRAPHS` (deprecated compatibility) | CUDA-only request. Non-CUDA backends ignore it, and a serving profile with live graph capture disabled selects eager-only execution regardless of this value. |

Capacity detection is device-scoped. Discrete accelerators use the selected
device's driver-reported VRAM and never count GTT as device-local capacity.
Linux DRM topology keeps the primary VRAM heap and a separately admissible
host-backed GTT tier distinct. Any nonzero VRAM heap remains the primary pool,
even when GTT is larger, because that is also a common discrete-AMD shape. This
preserves large carved pools such as Strix Halo's 96 GiB VRAM rather than
incorrectly capping it to Linux's smaller CPU-online pool.

The host-backed tier is independently bounded by GTT free bytes,
`MemAvailable`, the most restrictive finite `memory.max` and `memory.high`
headroom across the visible cgroup hierarchy, and the unified-memory reserve.
Kiln's current Vulkan paged KV cache and recurrent prefix state are
host-resident, so startup requires them to fit both the primary accelerator
budget and this host-backed budget. Prefix state is reserved first; automatic
KV sizing uses the configured inference fraction of the remaining host-backed
headroom. Missing or exhausted host-tier evidence yields a startup error before
the zero-filled Rust allocation is attempted. CUDA and ROCm KV pools remain
device-resident and are not capped by this separate tier.

Only a driver reporting no local VRAM and a nonzero GTT heap is automatically
treated as fully host-shared. A host-shared primary pool retains
`max(6 GiB, 25% of backing capacity)` for the OS and CPU workloads. Apple
Silicon applies the same reserve to its unified pool and samples live host
pressure. The optional `memory.gpu_memory_gb` cap is applied only after safe
physical capacity is established.

Physical-device identity is currently fail-closed while backend selection and
memory probes still expose unrelated logical ordinals. CUDA, ROCm, and Vulkan
startup therefore accepts only logical device `0` when the relevant NVIDIA or
DRM candidate set is provably singular. Multi-device hosts, nonzero ordinals,
failed candidate enumeration, and visibility/remapping controls such as
`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`,
`KILN_VULKAN_DEVICE`, or `GGML_VK_VISIBLE_DEVICES` are rejected before model
upload. `Auto` probing remains diagnostic-only; CPU performs no accelerator
probe, and Apple Silicon uses its single unified physical memory pool. This is
an interim safety restriction until backend selection and probing share a
typed PCI-address or UUID identity.

## `[training]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `training.grad_checkpoint_segments` | optional unsigned integer; omitted (`None`) | `KILN_TRAINING_GRAD_CHECKPOINT_SEGMENTS` (implemented) | `KILN_GRAD_CHECKPOINT_SEGMENTS` (deprecated compatibility) | Must be greater than zero when set. When present, selects an explicit process-lifetime gradient-checkpoint segment count for native training; omission leaves workload- and capacity-aware automatic planning enabled. |
| `training.no_grad_checkpoint` | boolean; `false` | `KILN_TRAINING_NO_GRAD_CHECKPOINT` (implemented) | `KILN_NO_GRAD_CHECKPOINT` (deprecated compatibility) | Disables gradient checkpoint execution for native training. The disabled state and any explicit segment count are retained together in the immutable training policy and exact-resume identity. Disabling checkpointing can materially increase training memory. |
| `training.recompute_checkpoint_boundaries` | string enum; `"auto"` | `KILN_TRAINING_RECOMPUTE_CHECKPOINT_BOUNDARIES` (implemented) | `KILN_RECOMPUTE_CHECKPOINT_BOUNDARIES` (deprecated compatibility) | `auto`, `enabled`, or `disabled`, case-insensitive with surrounding whitespace ignored. `auto` replays sparse SFT boundaries when sequence length is at least `recompute_boundary_threshold_tokens`; `enabled` always replays them and `disabled` always retains them. The canonical environment name accepts only those three words. The deprecated alias additionally accepts historical `1`/`true`/`yes` and `0`/`false`/`no`; `on` and `off` are not accepted for this alias. |
| `training.recompute_boundary_threshold_tokens` | positive unsigned integer; `8192` | `KILN_TRAINING_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS` (implemented) | `KILN_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS` (deprecated compatibility) | Inclusive sequence-length threshold used only by automatic SFT boundary replay. Zero, negative, overflowing, malformed, and non-UTF-8 environment values stop startup. |
| `training.checkpoint_boundary_anchor_stride` | `"auto"` or positive unsigned integer; `"auto"` | `KILN_TRAINING_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE` (implemented) | `KILN_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE` (deprecated compatibility) | When sparse SFT replay is active, a concrete value retains every Nth segment boundary as an anchor. `auto` derives a shape-specific positive stride from sequence length, segment count, hidden width, boundary dtype, and the cache target. Zero and strings other than `auto` fail startup. |
| `training.checkpoint_boundary_cache_gb` | positive floating-point GiB value; `6.0` | `KILN_TRAINING_CHECKPOINT_BOUNDARY_CACHE_GB` (implemented) | `KILN_CHECKPOINT_BOUNDARY_CACHE_GB` (deprecated compatibility) | Automatic anchor-stride memory target. Despite the historical `_gb` spelling, the unit is GiB (`2^30` bytes). The value must be finite, positive, convert to at least one byte, and remain below the `u64` byte limit. Startup converts it once to integral bytes using the historical truncating conversion. |
| `training.checkpoint_interval` | optional unsigned integer; omitted (`None`) | `KILN_TRAINING_CHECKPOINT_INTERVAL` (implemented) | `KILN_CHECKPOINT_INTERVAL` (deprecated compatibility) | Must be greater than zero when set. Number of committed optimizer steps between checkpoints; per-job configuration overrides it. Omission disables periodic checkpoints. |
| `training.webhook_url` | optional string; omitted (`None`) | `KILN_TRAINING_WEBHOOK_URL` (implemented) | `KILN_TRAINING_WEBHOOK_URL` | Must be a non-empty valid HTTP(S) URL. An exactly empty environment value clears a TOML URL; whitespace is not a clearing value and fails validation. Delivery is fire-and-forget with a five-second timeout after terminal state is recorded. |
| `training.logit_cache_dir` | optional path string; omitted (beside the effective adapter directory) | `KILN_TRAINING_LOGIT_CACHE_DIR` (implemented) | `KILN_LOGIT_CACHE_DIR` (deprecated compatibility) | Must be a non-empty path when set. Resolved once at startup; request handlers never reread the environment. |
| `training.max_queued_jobs` | unsigned integer; `32` | `KILN_TRAINING_MAX_QUEUED_JOBS` (implemented) | `KILN_TRAINING_MAX_QUEUED_JOBS` | Must be greater than zero. At capacity, submissions return HTTP 503 with `Retry-After: 30`. |
| `training.max_tracked_jobs` | unsigned integer; `1024` | `KILN_TRAINING_MAX_TRACKED_JOBS` (implemented) | `KILN_TRAINING_MAX_TRACKED_JOBS` | Must be greater than zero and at least `max_queued_jobs`. Counts queued, running, completed, and failed entries. |
| `training.tracked_job_ttl_secs` | unsigned integer; `604800` | `KILN_TRAINING_TRACKED_JOB_TTL_SECS` (implemented) | `KILN_TRAINING_TRACKED_JOB_TTL_SECS` | Must be greater than zero. Terminal entries older than the TTL are removed; active jobs are never age-evicted. |

Training GPU work is also governed by `server.serving_profile`; the default
`stable` profile does not grant training GPU ownership.

### Optimizer support is a resident capability, not configuration

There is intentionally no startup or request switch for optimizer fallback or
rounding. Product SFT, GRPO, OPD, the distinct DistillRefresh workload, and
OPD-backed distillation use immutable round-to-nearest. Stochastic rounding
remains available only to explicit Rust optimizer-library callers; it is not a
server product mode. Consequently, the
mechanical `KILN_<SECTION>_<FIELD>` rule produces no optimizer-policy
environment variable.

The following old process-global inputs were removed without aliases:

- `KILN_BF16_STOCHASTIC_ROUND`
- `KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK`
- `KILN_CUDA_TRAINING_OPTIMIZER_FALLBACK`
- `KILN_ROCM_TRAINING_OPTIMIZER_FALLBACK`
- `KILN_METAL_TRAINING_OPTIMIZER_FALLBACK`
- `KILN_VULKAN_TRAINING_OPTIMIZER_FALLBACK`

Delete them from service definitions. There is no replacement field: silent
host fallback during product training would change performance, precision,
residency, and checkpoint identity. Unknown environment variables are not a
supported way to select behavior.

`GET /v1/config` exposes the resident contract at
`training.optimizer_support`. A real runner returns schema ID
`kiln.training-optimizer-support` and version `1`; a mock runner or a runner
whose snapshot cannot be read returns `null`. `immutable_after_startup=true`
means this is a process-lifetime capability snapshot, not mutable
configuration. The object has this complete shape (the arrays always enumerate
all four workloads and all three optimizer kinds):

```json
{
  "schema": {"id": "kiln.training-optimizer-support", "version": 1},
  "backend": "rocm",
  "device": "rocm:0",
  "base_weight_dtype": "bf16",
  "resolved_lora_parameter_dtype": "bf16",
  "immutable_after_startup": true,
  "rounding_modes": ["round_to_nearest"],
  "backend_implementation_rounding_modes": ["round_to_nearest"],
  "optimizer_tuple_kinds": ["muon", "adam_w", "sgd"],
  "workloads": [
    {
      "workload": "sft",
      "supported": true,
      "unavailable_reason": null,
      "allowed_optimizer_kinds": ["muon", "adam_w", "sgd"]
    },
    {
      "workload": "grpo",
      "supported": true,
      "unavailable_reason": null,
      "allowed_optimizer_kinds": ["muon", "adam_w", "sgd"]
    },
    {
      "workload": "opd",
      "supported": true,
      "unavailable_reason": null,
      "allowed_optimizer_kinds": ["muon", "adam_w", "sgd"]
    },
    {
      "workload": "distill_refresh",
      "supported": false,
      "unavailable_reason": "distill_refresh is unavailable until admission pins separate exact SFT and OPD phase plans, prepares the exact SFT rows, and reserves the maximum sequential working set",
      "allowed_optimizer_kinds": []
    }
  ],
  "optimizers": [
    {
      "kind": "muon",
      "backend_implementation": {
        "supported": true,
        "route": "native_device_hook",
        "native_device_hook": true,
        "parameter_dtypes": ["f32", "bf16"]
      },
      "optimizer_tuple": {
        "supported": true,
        "unavailable_reason": null,
        "lora_rank": {
          "minimum": 2,
          "maximum": 48,
          "backend_maximum": 48,
          "model_maximum": 1024,
          "live_memory_admission_required": true
        }
      }
    },
    {
      "kind": "adam_w",
      "backend_implementation": {
        "supported": true,
        "route": "native_device_hook",
        "native_device_hook": true,
        "parameter_dtypes": ["f32", "bf16"]
      },
      "optimizer_tuple": {
        "supported": true,
        "unavailable_reason": null,
        "lora_rank": {
          "minimum": 1,
          "maximum": 1024,
          "backend_maximum": null,
          "model_maximum": 1024,
          "live_memory_admission_required": true
        }
      }
    },
    {
      "kind": "sgd",
      "backend_implementation": {
        "supported": true,
        "route": "native_device_hook",
        "native_device_hook": true,
        "parameter_dtypes": ["f32", "bf16"]
      },
      "optimizer_tuple": {
        "supported": true,
        "unavailable_reason": null,
        "lora_rank": {
          "minimum": 1,
          "maximum": 1024,
          "backend_maximum": null,
          "model_maximum": 1024,
          "live_memory_admission_required": true
        }
      }
    }
  ]
}
```

Read the response as four distinct gates:

1. `backend_implementation` reports whether the optimizer library contains a
   raw executable update implementation, its `portable_reference` or
   `native_device_hook` route, whether that route is an accelerator-native
   hook, and its parameter dtypes. It does not claim that resident server
   weights can execute a training workload. CPU therefore reports
   `portable_reference` with `native_device_hook=false`. An unavailable raw
   implementation reports `supported=false`, route `unavailable`, and
   `native_device_hook=false`; `parameter_dtypes` remains the backend's
   advertised kind-specific dtype set.
2. `optimizer_tuple` combines that implementation with the exact resident
   backend/device identity, base-weight dtype, resolved LoRA dtype, immutable
   product rounding rule, optimizer kind, and static LoRA-rank range. Within
   `lora_rank`, `backend_maximum` is the optimizer backend's optional ceiling,
   `model_maximum` is derived from the smallest dimension among every trained
   projection, and `maximum` is their effective minimum. A null
   `backend_maximum` means the backend adds no ceiling; it never makes the
   effective `maximum` null because the resident model still supplies one.
   `optimizer_tuple_kinds` is the summary of kinds whose resident tuple is
   supported. A supported tuple has `unavailable_reason: null`; an unsupported
   tuple retains its concrete reason next to the kind.
3. Each member of `workloads` independently reports whether the complete
   server substrate for `sft`, `grpo`, `opd`, or `distill_refresh` is
   executable. Its
   `allowed_optimizer_kinds` is the intersection of that workload gate and
   `optimizer_tuple_kinds`; it is empty whenever the workload is unavailable,
   and that entry carries a concrete `unavailable_reason`.
   This per-workload array, not a raw hook or tuple, is the static HTTP and
   dashboard admission authority.
4. `live_memory_admission_required=true` is deliberately outside the static
   promise. A supported tuple and workload can still be rejected when the
   exact request shape does not fit the current memory budget. Admission
   rejects the request; it never lowers rank, changes optimizer, or switches
   execution route.

`rounding_modes` is the server-product policy and currently contains only
`round_to_nearest`. `backend_implementation_rounding_modes` describes what the
resident backend optimizer implementation reports. Those fields are separate
so a future implementation capability cannot silently become a product mode.
`resolved_lora_parameter_dtype: null` means the resident base dtype cannot be
resolved under the backend precision policy; the per-kind tuple reasons carry
the failure rather than treating the dtype as unknown.

The sibling `training.native_training_supported` and
`native_training_unavailable_reason` fields are a coarse compatibility summary.
They do not replace `workloads`: a true summary does not promise that all four
workloads are available, and clients selecting SFT, GRPO, OPD, or
DistillRefresh must inspect the matching workload member.

For canonical Qwen3.5-4B, whose trained-projection model ceiling is 1024, the
resident optimizer-tuple matrix before workload and live-memory admission is:

| Backend | Base -> LoRA dtype | SGD | AdamW | Muon |
|---|---|---|---|---|
| CPU portable reference | F32 -> F32 | rank 1..=1024 (backend unbounded) | rank 1..=1024 (backend unbounded) | rank 2..=1024 (backend unbounded) |
| CUDA | F32 -> F32; BF16 -> BF16 | rank 1..=1024 (backend unbounded) | rank 1..=1024 (backend unbounded) | rank 2..=48 |
| ROCm | F32 -> F32; BF16 -> BF16 | rank 1..=1024 (backend unbounded) | rank 1..=1024 (backend unbounded) | rank 2..=48 |
| Metal | BF16 -> BF16 | unsupported | rank 1..=1024 (backend unbounded) | rank 2..=32 |
| Vulkan tuple | F32/BF16 -> F32 | rank 1..=1024 (backend unbounded) | rank 1..=1024 (backend unbounded) | rank 2..=32 |

This table is not a server-execution matrix. Current CPU processes expose the
portable F32 optimizer tuples for diagnostics and direct library testing, while
their `sft`, `grpo`, `opd`, and `distill_refresh` workload entries remain
unsupported. A Vulkan
process can likewise expose native hooks and resident optimizer tuples while a
workload fails closed. In the usual hybrid Vulkan server, model weights are
CPU-hosted, so exact native backend/device identity and resident-weight checks
reject every workload even though raw Vulkan hooks exist.

Static workload admission requires all of the following: a real readable
runner; a serving profile that grants training GPU ownership; agreement between
configured and resident weight devices; a training runtime that resolves those
weights; exact native backend and device identity; no Marlin-packed model
projection; and the authoritative `kt_tape_authoritative` forward/backward
route. SFT additionally rejects effective multi-segment checkpointing with a
`full_logits` loss route. OPD additionally requires both its loss route and
phase-B backward route. GRPO uses the common authoritative-tape gate. The
specific failed condition appears in `workloads[].unavailable_reason`.

DistillRefresh is not an OPD alias for admission. It is a sequential composite
with an SFT knowledge phase and an OPD behavior-recovery phase. The
`distill_refresh` row is unconditionally fail-closed with this stable reason:
`distill_refresh is unavailable until admission pins separate exact SFT and OPD
phase plans, prepares the exact SFT rows, and reserves the maximum sequential
working set`. Enabling it requires one admission artifact that binds both exact
phase plans, the precise SFT rows that phase one will consume, and a reservation
for the larger of the two sequential working sets. Adding the phases' estimates
or admitting only the OPD-shaped request is not equivalent.

F16 is inference-only on CUDA/ROCm. `maximum` is the effective static ceiling,
computed as `min(backend_maximum, model_maximum)` while treating a null backend
maximum as unbounded. Live memory may impose a lower request-time ceiling but
does not rewrite these static fields. The server validates
the cheap workload gate, optimizer kind, hyperparameters, base/LoRA dtype,
rounding, and rank before checkpoint loading or corpus scanning across dedicated
training endpoints, the intent-tagged front door, recipes, judge/self-improve,
DistillRefresh, and OPD-backed distillation. Checkpoint loading and corpus
preparation happen only after those checks. The worker
revalidates the checkpoint and the workload/tuple before memory reservation and
residency.
Invalid kind, rank, or hyperparameters use structured
`training_invalid_request`; an unsupported base dtype, backend/device identity,
or workload substrate uses `training_backend_unsupported`.

Cheap teacher-alias validation and metadata pinning retain their established
request-error ordering and may precede the workload guard. Remote/local teacher
materialization, checkpoint loading, corpus scanning, memory preflight, and GPU
reservation do not.

For a queued resume, admission fully validates the manifest and every declared
artifact hash, then retains a compact checkpoint-ID/manifest-digest identity
plus the effective seed. The manifest digest covers its artifact hash entries.
Before memory reservation at dequeue, the worker fully reloads the checkpoint
and requires both the recomputed identity and effective seed to match. This is
not an external-file snapshot: Kiln does not copy or pin checkpoint files in
the queue, and a mutation after the reload remains a filesystem race. Keep the
checkpoint directory immutable for its entire use.

`GET /v1/recipes` applies the same static checks to every built-in recipe step
and returns `admission: {supported, unavailable_reason}` on each descriptor. A
false descriptor identifies the first unsupported workload or optimizer/rank
tuple. A true descriptor is not a memory reservation: recipe submission repeats
the full preflight before any step is prepared, and live-memory admission still
occurs for each queued job.

### Tape scope is internal execution authority, not configuration

The kt autograd tape is a required workload substrate. Kiln opens its
thread-local scope internally around the forward/backward region that must
record gradients and closes it when that region completes. SFT, GRPO, and OPD
admission first require an authoritative tape route; a request cannot open a
scope, suppress one recorder, or choose a partially tape-backed execution.
There is intentionally no TOML field, API member, CLI flag, or mechanically
derived environment name for tape activation.

Scope presence is also the complete inference boundary. Ordinary inference
does not open a training tape scope and therefore retains forward-only fast
paths. In particular, GDN chunkwise recurrence and CUDA's weight-aware
embedding lookup defer to their tape-aware alternatives only while a scope is
actually active. They do not consult a cached process-global tape setting. This
prevents a default-on global flag from changing inference routing when no
gradient graph exists.

The following former correctness switches were removed without compatibility
aliases or replacement fields:

- `KILN_USE_TAPE_FORWARD`
- `KILN_USE_TAPE_FLASH_ATTN`
- `KILN_USE_TAPE_SDPA`
- `KILN_USE_TAPE_LORA_ADD`
- `KILN_USE_TAPE_GDN`
- `KILN_USE_TAPE_GDN_CONV`
- `KILN_USE_TAPE_GDN_QK_NORM`
- `KILN_USE_TAPE_GDN_GATED_NORM`

Delete these names from service definitions instead of translating them. A
debug or performance control may select an implementation only when the active
scope still records the required analytic backward and preserves the graph. It
must not disable the tape, bypass an individual recorder, or fall through to a
forward-only kernel. If no graph-preserving route exists, the workload fails
closed through `training.optimizer_support.workloads[]` rather than accepting a
correctness opt-out.

This contract proves static ownership and route isolation; it does not by
itself qualify numerical correctness, latency, memory use, or stability on a
particular accelerator. Those claims require the platform-specific
qualification workload and committed evidence described in
[`NATIVE_SFT_PROFILE.md`](NATIVE_SFT_PROFILE.md).

#### Frozen parameters are constants, not optimizer leaves

Native SFT, GRPO, and OPD train only LoRA A/B tensors. Embedding tables, base
projection matrices, RMSNorm and GDN gated-RMSNorm weights, GDN gate parameters,
MTP projection weights, and loss-head transposes are saved constants. Their
tape nodes record only differentiable activations and LoRA leaves. Backward
therefore returns only required activation and LoRA gradients; it does not
allocate, retain, or deposit frozen-weight gradients, and frozen projection
paths do not execute a dWeight matrix multiplication.

The split Q/gate projection keeps the original full LoRA A/B tensor IDs on both
recorded chunks. Each chunk pads its B contribution to the full shape before
the tape combines them, and tape-aware reshapes preserve the activation chain.
Sliced base matrices and temporary B slices are never trainable leaves. This is
an execution invariant with no TOML, request, CLI, debug, or environment
override. It reduces frozen-gradient memory and compute by construction, but
does not by itself qualify backend throughput.

The optimizer boundary independently requires an exact LoRA gradient set. The
observed tensor IDs must equal the configured trainable leaves; shape,
backward-compute dtype, master device, and finite values must match before any
parameter, optimizer state, or step counter is changed. Missing or unknown
entries reject the step, while a valid all-zero gradient is accepted.
Checkpointed execution applies the same identity and metadata contract to the
exact leaves in each layer range. The tape bridge reduces distinct recorded
input contributions to one accumulated entry per leaf, but rejects the result
if any registered differentiable input is absent. A connected clone therefore
cannot hide a disconnected use of the same leaf. Split output-projection chunks
assemble zero-padded contributions under the original full leaf ID rather than
temporary slice IDs. A checkpoint range with no configured LoRA leaves accepts
only an empty gradient set, and the merge rejects duplicate leaf IDs across
disjoint segments. Token-level GRPO
accumulation validates identity and metadata for each source but defers its
finite-value reduction to the single final optimizer boundary, avoiding a
per-completion device synchronization multiplier. None of these checks has a
configuration or environment bypass. CUDA, ROCm, and Vulkan
use backend finite reducers. Metal performs a synchronized full-gradient host
scan at the same boundary until a native Metal reducer is qualified; this is a
correctness fallback, not an advertised throughput path.

### Backend-owned SFT loss route is not configuration

There is intentionally no `[training]` field for the native SFT loss route.
The resident backend reports one typed capability: CUDA and ROCm report
`kt_tape_flce`, Vulkan reports `vulkan_active_rows`, and Metal reports
`full_logits`. These values are capability and artifact names, not accepted
TOML enums, request fields, CLI flags, or `/v1/config` controls. Because no
configuration field exists, the mechanical naming rule does not produce an
environment variable for this choice.

The old `KILN_USE_FLCE` input has been removed. It is not a deprecated alias,
the typed loader does not consume it, and setting it has no effect on current
SFT loss selection. Operators should remove it from service definitions rather
than translating it to a new name. Backend loss routing is not an A/B switch.

SFT admission resolves the route from the resident model runner and applies a
route-specific, saturating memory upper bound. CUDA/ROCm account for kt-tape
FLCE buffers and any F32 head promotion; Vulkan accounts for its maximum legal
active-row chunk and F32 workspace; `full_logits` accounts for dense `[T, V]`
logits and cross-entropy forward/backward residency. Overflow saturates toward
rejection. An over-budget request returns HTTP 413 with estimated and available
capacity and a breakdown containing `loss workspace ... (route=<route>)`.

The admitted enum is stored with the queued SFT job. The worker revalidates it
against the resident runner before memory reservation or reclamation, and the
trainer revalidates it against its execution backend before resident or
trainable allocations. The pinned value drives each forward/backward step,
appears in new SFT receipts as `runtime.sft_loss_route`, and participates in
the SFT exact-resume planning identity. `full_logits` is not compatible with a
multi-segment checkpoint plan; that combination returns
`training_invalid_request` before queue publication and is checked again by the
trainer. These are immutable execution contracts, not omitted configuration
knobs.

### SFT checkpoint-boundary policy

These four fields control activation boundaries inside gradient-checkpointed
native SFT. They do not enable or disable gradient checkpointing; that remains
the job of `no_grad_checkpoint` and the resolved segment plan. When replay is
inactive, SFT retains every segment input. When replay is active, it retains a
sparse set of anchors and recomputes intervening segment inputs during the
backward pass.

For `checkpoint_boundary_anchor_stride = "auto"`, Kiln computes the bytes in
one boundary as `sequence length * hidden size * bytes per element`, determines
how many anchors fit in `checkpoint_boundary_cache_gb`, reserves one slot for
replay, and spreads the remaining anchors across the planned segments. The
automatically selected result is always at least one, and auto mode selects
stride one for a one-segment plan. An explicit stride wins even for a
one-segment plan and bypasses only this shape calculation. It does not override
the recompute mode.

Startup resolves all four values into one immutable
`CheckpointBoundaryPolicy`. The typed loader retains `default`, `config_file`,
or `environment` source attribution for each configured value. Admission and
execution receive the same copyable policy and call the same pure threshold and
shape functions, so a request cannot be admitted under one boundary-memory
estimate and execute another. Trainer and preflight code do not re-read any of
the four environment names. Changing any value requires restarting the server;
queued and running jobs retain the policy installed at startup.

Sparse checkpoint-boundary replay is currently SFT-only. Native GRPO and OPD
retain all `num_segments + 1` boundaries and do not use these fields to choose
their live boundary layout. The common training runtime still records the
resolved policy in exact-resume planning identity. GRPO and OPD use
`kiln.training-checkpoint-planning.v3`. SFT extends the same object to v4 with
its pinned backend loss route. A changed boundary value, including one that is
execution-inert for GRPO or OPD today, is therefore exact-resume drift. An SFT
v3 identity also cannot authorize continuation under v4 because it cannot
prove the admitted loss implementation. See [Native Training
Checkpoints](training-checkpoints.md#checkpoint-planning-identity).

The resolved object has this stable JSON shape:

```json
{
  "recompute_mode": "auto",
  "recompute_threshold_tokens": 8192,
  "anchor_stride": null,
  "cache_target_bytes": 6442450944
}
```

`GET /v1/config` exposes it at `training.checkpoint_boundary_policy` alongside
the gradient-checkpoint plan. `GET /health` and its `/v1/health` alias expose it
at `training.checkpoint_boundary_policy`; trusted
`GET /v1/debug/model-state` exposes the same object at
`training.checkpoint_boundary_policy`. The dashboard's Runtime Config training
group renders mode, threshold, explicit-or-auto stride, cache target, and the
restart requirement. These are observations of immutable application state,
not live controls or fresh environment reads.

## `[logging]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `logging.level` | string; `"info"` | `KILN_LOGGING_LEVEL` (implemented) | `KILN_LOG_LEVEL` (deprecated compatibility) | One of `trace`, `debug`, `info`, `warn`, `error`, or a valid `tracing_subscriber::EnvFilter` directive such as `kiln=debug,tower_http=warn`. `RUST_LOG` overrides the effective filter but does not make an invalid typed value valid. |
| `logging.format` | string; `"auto"` | `KILN_LOGGING_FORMAT` (implemented) | `KILN_LOG_FORMAT` (deprecated compatibility) | Exactly `auto`, `json`, `pretty`, `text`, or `human`. `auto` is pretty on a stderr TTY and JSON otherwise; `text` and `human` select pretty output. |

Logging is bootstrapped before the full file is validated so parse failures can
be reported through the requested renderer. The authoritative loader then
validates the complete file and stops startup if it differs or is invalid.
Malformed or non-UTF-8 `RUST_LOG` is fatal.

## `[prefix_cache]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `prefix_cache.enabled` | boolean; `true` | `KILN_PREFIX_CACHE_ENABLED` (implemented) | `KILN_PREFIX_CACHE_ENABLED` | Enables reuse of KV blocks and recurrent-state snapshots for shared prompt prefixes. |
| `prefix_cache.max_blocks` | optional unsigned integer; omitted (`None`) | `KILN_PREFIX_CACHE_MAX_BLOCKS` (implemented) | `KILN_PREFIX_CACHE_MAX_BLOCKS` | Must be greater than zero when set. `None` resolves to half of the allocated KV block pool. |
| `prefix_cache.max_entries` | optional unsigned integer; omitted (`None`) | `KILN_PREFIX_CACHE_MAX_ENTRIES` (implemented) | `KILN_PREFIX_CACHE_MAX_ENTRIES` | Must be greater than zero when set. `None` resolves from the relevant safe allocation tier and per-entry recurrent-state bytes, with at least one entry. Vulkan reserves this state from the separately bounded host-backed tier before sizing its host-resident KV pool; an explicit count that cannot fit stops startup. |

## `[speculative]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `speculative.enabled` | boolean; `false` | `KILN_SPECULATIVE_ENABLED` (implemented) | `KILN_SPEC_ENABLED` (deprecated compatibility) | `false` forces the effective method to `off`. For compatibility, `true` with `method = "off"` selects `skip_layer`; that effective method is then subject to the fail-closed startup gate below. |
| `speculative.method` | string enum; `"off"` | `KILN_SPECULATIVE_METHOD` (implemented) | `KILN_SPEC_METHOD` (deprecated compatibility) | TOML accepts `off`, `skip_layer`, or `mtp`. The environment parser also accepts `none`, `0`, `false`; `skiplayer`, `skip-layer`, `self`; and `native_mtp`, `native-mtp`. The deprecated method alias also controls `enabled` when neither enabled spelling is present: non-off enables and off disables. The canonical method name has literal field semantics and does not implicitly enable. |
| `speculative.num_speculative_tokens` | unsigned integer; `4` | `KILN_SPECULATIVE_NUM_SPECULATIVE_TOKENS` (implemented) | `KILN_SPEC_NUM_TOKENS` (deprecated compatibility) | Draft proposal bound K. Must be in `1..=4`; out-of-range values stop startup before accelerator allocation. This conservative ceiling matches the planned local K=1/2/4 qualification matrix and cannot be raised without new accelerator evidence. |
| `speculative.draft_layers` | unsigned integer; `8` | `KILN_SPECULATIVE_DRAFT_LAYERS` (implemented) | `KILN_SPEC_DRAFT_LAYERS` (deprecated compatibility) | Must be greater than zero. When speculative decoding is enabled, model-dependent startup validation also requires this value to be less than the selected model's transformer-layer count. Invalid geometry stops startup; it does not fall back at request time. |

The loaded `SpeculativeDecodingConfig` is the sole serving authority. TOML,
canonical environment overrides, and deprecated aliases are resolved and
validated once during typed startup, then the immutable result is passed to
serving state. Request dispatch does not re-read speculative environment
variables. Restart the server to apply a change.

**Current fail-closed status:** `off` is the only serving method available.
Every enabled method, including `skip_layer` and `mtp`, stops startup until its
cancellation, owner-settlement, EOS, context-capacity, and burst-admission
contracts pass local accelerator qualification. The fields remain part of the
typed configuration so their intended contract is explicit and can be qualified
without introducing a second configuration path. `kiln config --file ...`
applies the same availability gate as `serve`, native MTP weights remain
deferred (`load_mtp=false`), request dispatch contains no speculative branch,
and Desktop always launches with speculative serving off. Benchmark-only
speculative paths require both `KILN_QUALIFICATION=1` and the qualification
harness result-path contract before model loading.

## `[streaming_prefill]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `streaming_prefill.mode` | string enum; `"auto"` | `KILN_STREAMING_PREFILL_MODE` (implemented) | `KILN_STREAMING_PREFILL` and `KILN_STREAMING_PREFILL_ENABLED` (deprecated compatibility) | `auto`, `enabled`, or `disabled`, case-insensitive with surrounding whitespace ignored. `auto` delegates dispatch to backend policy, `enabled` selects every non-empty prompt, and `disabled` selects none. The canonical environment name accepts only those three words. Deprecated aliases also accept the strict boolean forms listed above, mapping true to `enabled` and false to `disabled`. Legacy TOML `enabled = true` or `enabled = false` remains accepted; if `mode` is also present, both must express the same non-auto intent or startup fails. |
| `streaming_prefill.threshold_tokens` | `"auto"` or positive unsigned integer; `"auto"` | `KILN_STREAMING_PREFILL_THRESHOLD_TOKENS` (implemented) | `KILN_STREAMING_PREFILL_THRESHOLD_TOKENS` | In `auto` mode, an integer replaces the backend crossover only when that backend already has a threshold-based automatic policy. It does not make CPU or Vulkan auto-dispatch streaming. `0`, negative values, and strings other than `auto` fail startup. `enabled` and `disabled` modes ignore this crossover for dispatch while retaining it in diagnostics. |
| `streaming_prefill.tile_tokens` | `"auto"` or positive unsigned integer; `"auto"` | `KILN_STREAMING_PREFILL_TILE_TOKENS` (implemented) | `KILN_STREAMING_TILE_TOKENS` (deprecated compatibility) | Base tile for ordinary tiled prefill and non-tape GDN segment execution. Concrete values must be positive multiples of 64. When this field is concrete and either specialized tile below is `auto`, that specialized route inherits this base value rather than its backend default. |
| `streaming_prefill.tape_tile_tokens` | `"auto"` or positive unsigned integer; `"auto"` | `KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS` (implemented) | `KILN_TAPE_STREAMING_TILE_TOKENS` (deprecated compatibility) | Tile used by tape-authoritative training forward paths. Concrete values must be positive multiples of 64. `auto` inherits an explicit `tile_tokens`; when both are `auto`, backend policy owns the value. |
| `streaming_prefill.detached_full_attn_tile_tokens` | `"auto"` or positive unsigned integer; `"auto"` | `KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS` (implemented) | `KILN_DETACHED_FULL_ATTN_TILE_TOKENS` (deprecated compatibility) | Tile for detached materialized full-attention training work. A concrete value also controls its derived boundary-forward and tape-replay variants. `auto` inherits an explicit `tile_tokens`; when both are `auto`, each variant keeps its backend default. Every concrete value must be a positive multiple of 64. |
| `streaming_prefill.last_token_lm_head` | boolean; `true` | `KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD` (implemented) | `KILN_STREAMING_LAST_TOKEN_LM_HEAD` (deprecated compatibility) | When true, the final inference streaming tile computes the LM head only for its last row. All centralized strict boolean spellings, including `on` and `off`, work identically. |

Every deprecated alias is parsed strictly and emits a value-free startup
warning naming its canonical replacement. When a canonical name is present,
each present alias must normalize to the same typed value or startup fails and
names both variables. For `mode`, if both deprecated aliases are present
without `KILN_STREAMING_PREFILL_MODE`, both are validated and the historical
higher-precedence `KILN_STREAMING_PREFILL_ENABLED` value wins over
`KILN_STREAMING_PREFILL`. A malformed lower-precedence alias still fails; it is
never ignored because another spelling is present.

The selected backend supplies the following `auto` policy. Dispatch thresholds
are inclusive. Tile counts are tokens:

| Backend | Automatic dispatch | Base | Tape | Detached full attention | Detached boundary | Detached tape replay |
|---|---:|---:|---:|---:|---:|---:|
| CPU | never | 8192 | 8192 | 8192 | 8192 | 8192 |
| CUDA | prompt tokens >= 2048 | 1024 | 1024 | 8192 | 65536 | 65536 |
| ROCm | prompt tokens >= 2048 | 1024 | 1024 | 8192 | 8192 | 8192 |
| Metal | prompt tokens >= 2048 | 2048 | 2048 | 8192 | 8192 | 8192 |
| Vulkan | never | 2048 | 2048 | 8192 | 8192 | 8192 |

`mode = "enabled"` can deliberately force non-empty CPU or Vulkan inputs onto
the streaming path; this is an operator override, not a claim that the route is
qualified or faster on that backend. Conversely, `mode = "disabled"` is the
clean isolation control for a monolithic-prefill A/B. A threshold override
only adjusts an existing CUDA, ROCm, or Metal automatic crossover. Dispatch
eligibility and physical splitting are distinct: an eligible route creates
multiple tiles only when its sequence is longer than the effective tile for
that route.

Startup resolves this section once after backend selection and injects the same
immutable execution policy into ordinary generation, prompt-logprob scoring,
native SFT/GRPO/OPD admission and training forwards, local OPD teachers, MTP
alignment, checkpoint planning, and the benchmark harness. Server-owned
training and teacher construction fail closed if that startup policy is
missing. No production model or trainer path re-reads these public environment
names. Every change requires a restart; an existing job or request cannot
observe a mid-process change.

Kiln prompt-logprob teacher identity uses
`kiln.prompt-logprobs.inference-config.v2` and hashes the complete resolved
execution policy: mode, threshold, every base/tape/detached tile variant, and
last-token LM-head behavior. Changing any of these fields therefore changes
`inference_config_sha256`, the canonical teacher content revision, and every
identity-bound logit-cache key. A deployment must re-register the teacher and
must not resume an OPD checkpoint pinned to the previous revision.

`GET /v1/config` exposes the complete resolved object at
`streaming_prefill`. Health repeats it at
`prefill_runtime.streaming_prefill`, and trusted debug exposes it at top-level
`streaming_prefill`. The object separates:

- configured mode/source, backend dispatch rule, effective dispatch rule, and
  effective authority;
- configured/backend/effective threshold, including whether an override
  actually changed a threshold-based backend policy;
- configured/backend/effective values and sources for base, tape, detached,
  detached-boundary, and detached-replay tiles;
- last-token LM-head configured/effective state and source; and
- `immutable_after_startup` plus `restart_required_to_change`.

An inherited specialized tile reports a closed source such as
`inherited_from_tile_tokens_config_file`; an untouched automatic value reports
`backend_policy`. There is intentionally no single `effective_enabled` boolean:
in automatic mode dispatch depends on the current prompt length.

Examples:

```toml
# Keep backend dispatch and tile policy (recommended default).
[streaming_prefill]
mode = "auto"
threshold_tokens = "auto"
tile_tokens = "auto"
tape_tile_tokens = "auto"
detached_full_attn_tile_tokens = "auto"
last_token_lm_head = true
```

```toml
# Isolate a suspected tiled-prefill pause.
[streaming_prefill]
mode = "disabled"
```

```toml
# ROCm tuning experiment: stream from 4096 tokens, use one base tile for all
# ordinary, tape, detached, boundary, and replay routes through inheritance.
[streaming_prefill]
mode = "auto"
threshold_tokens = 4096
tile_tokens = 2048
tape_tile_tokens = "auto"
detached_full_attn_tile_tokens = "auto"
```

## `[adapters]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `adapters.library_url` | string URL; `"https://library.kiln.run"` | `KILN_ADAPTERS_LIBRARY_URL` (implemented) | `KILN_ADAPTER_LIBRARY_URL` (deprecated compatibility) | Must be a non-empty valid HTTP(S) URL. Resolved once at startup. |
| `adapters.max_disk_bytes` | optional unsigned integer; `107374182400` (100 GiB) | `KILN_ADAPTERS_MAX_DISK_BYTES` (implemented) | `KILN_ADAPTERS_MAX_DISK_BYTES` | Caps finalized adapter bytes under `adapter_dir`, excluding upload staging and the composed cache. For the environment alias only, empty or `0` means `None` and disables the cap. TOML `0` is accepted as a literal zero cap, not as `None`. |
| `adapters.composed_cache_max_bytes` | optional unsigned integer; `10737418240` (10 GiB) | `KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES` (implemented) | `KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES` | LRU byte cap for `.composed`. Environment empty or `0` disables this dimension; TOML `0` remains a zero cap. |
| `adapters.composed_cache_max_entries` | optional unsigned integer; `64` | `KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES` (implemented) | `KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES` | LRU entry-count cap for `.composed`. Environment empty or `0` disables this dimension; TOML `0` remains a zero cap. |

Because all three typed defaults are `Some(...)` and TOML has no null value,
the supported way to opt out is the working environment alias with `0` or an
empty value.

## `[teachers]`

`teachers.credentials` is a dynamic map of server-owned credential handles.
There is no field-level environment override for the map. Each credential
points to a separately named secret environment variable; that secret is not
stored in TOML, `AppState`, API responses, or receipts.

```toml
[teachers.credentials.primary-vllm]
origin = "https://vllm.example.com:8443"
api_key_env = "VLLM_TEACHER_API_KEY"
```

| TOML field template | Type and default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `teachers.credentials.<id>.origin` | required string for each entry; no entries by default | none; structured trust policy is TOML-only | none | Must be the exact canonical `scheme://host[:port]` origin, with no path, query, fragment, or embedded credentials. HTTPS is required unless the host is loopback. |
| `teachers.credentials.<id>.api_key_env` | required string for each entry | none; this value names the secret variable | the named secret variable itself | Must match `[A-Za-z_][A-Za-z0-9_]*`, length `1..=128`. The named variable must exist and contain a non-whitespace value at startup. |

Credential ids must be `1..=64` ASCII letters, digits, `_`, or `-`. A handle is
accepted only for its exact configured origin. Non-loopback teachers require a
configured credential handle; credential-free teachers are restricted to
loopback. Secret availability is checked again when resolving a teacher, but
operators should still treat secret changes as restart-scoped configuration.

## `[eval]`

The section is optional. When absent, the runtime uses the same queue defaults
shown below and roots eval data at `<adapter_dir>/.eval`. It creates `suites`,
`datasets`, and `judgments` below that root.

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `eval.eval_dir` | optional path string; omitted (`None`) | `KILN_EVAL_EVAL_DIR` (target only; not implemented) | none | Must be non-empty when set. Despite the historical field name, runtime treats it as the shared eval root and creates the three registry subdirectories below it. |
| `eval.max_queued_jobs` | unsigned integer; `32` | `KILN_EVAL_MAX_QUEUED_JOBS` (target only; not implemented) | none | Must be greater than zero. |
| `eval.max_tracked_jobs` | unsigned integer; `1024` | `KILN_EVAL_MAX_TRACKED_JOBS` (target only; not implemented) | none | Must be greater than zero and at least `max_queued_jobs`. |
| `eval.webhook_url` | optional string; omitted (`None`) | `KILN_EVAL_WEBHOOK_URL` (target only; not implemented) | none | Must be a non-empty valid HTTP(S) URL. Terminal eval notifications are fire-and-forget. |

Names such as `eval.root`, `eval.max_tracked_eval_jobs`, and an
`[eval.generation]` table are not accepted by the current typed schema.

## `[request_log]`

Request logging is enabled by default. It records inference request and response
JSON on a dedicated bounded writer thread; overload drops log rows instead of
blocking inference. Rotation and retention bound disk use.

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `request_log.enabled` | boolean; `true` | `KILN_REQUEST_LOG_ENABLED` (implemented) | `KILN_REQUEST_LOG_ENABLED` | Master switch. Initialization failure logs a warning and disables logging rather than aborting an otherwise valid server startup. |
| `request_log.dir` | optional path string; omitted (`None`) | `KILN_REQUEST_LOG_DIR` (implemented) | `KILN_REQUEST_LOG_DIR` | Must be non-empty when set. `None` resolves to `<adapter_dir>/.requests`. The environment alias cannot clear a TOML directory; an empty value is fatal. |
| `request_log.max_file_bytes` | unsigned integer; `67108864` (64 MiB) | `KILN_REQUEST_LOG_MAX_FILE_BYTES` (implemented) | `KILN_REQUEST_LOG_MAX_FILE_BYTES` | Must be at least `4096`. The active JSONL file rotates after reaching the threshold. |
| `request_log.max_total_bytes` | unsigned integer; `2147483648` (2 GiB) | `KILN_REQUEST_LOG_MAX_TOTAL_BYTES` (implemented) | `KILN_REQUEST_LOG_MAX_TOTAL_BYTES` | Must be greater than zero. Oldest rotated files are removed until retained bytes fit. There is no validation requiring this value to exceed `max_file_bytes`. |
| `request_log.compress` | boolean; `true` | `KILN_REQUEST_LOG_COMPRESS` (implemented) | `KILN_REQUEST_LOG_COMPRESS` | Gzip rotated files when true. |
| `request_log.max_capture_bytes` | unsigned integer; `4194304` (4 MiB) | `KILN_REQUEST_LOG_MAX_CAPTURE_BYTES` (implemented) | `KILN_REQUEST_LOG_MAX_CAPTURE_BYTES` | Must be greater than zero. Per-request and per-response storage cap; truncation affects the log only, never the wire response. |

The default is intentionally data-bearing. Review log-directory permissions,
retention, and the captured request content before exposing the service to
untrusted traffic.

## `[agent]`

The section is optional. Its absence disables the scheduled self-improvement
loop, while the embedded run engine still receives the default concurrency and
timeout.

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `agent.self_improve_interval_hours` | optional unsigned integer; omitted (`None`) | `KILN_AGENT_SELF_IMPROVE_INTERVAL_HOURS` (implemented) | `KILN_AGENT_SELF_IMPROVE_INTERVAL_HOURS` | Omission disables scheduling. `0` is accepted and also results in no scheduler. The first run occurs one full interval after startup; cadence is persisted under the adapter directory. |
| `agent.self_improve` | optional structured value; omitted (`None`) | `KILN_AGENT_SELF_IMPROVE` (target only; not implemented) | none | Request template submitted to the same self-improvement path as the API. This value is intentionally open structured data; its inner request contract is validated by that subsystem rather than by `KilnConfig`. |
| `agent.max_concurrent_runs` | unsigned integer; `2` | `KILN_AGENT_MAX_CONCURRENT_RUNS` (implemented) | `KILN_AGENT_MAX_CONCURRENT_RUNS` | Must be greater than zero. Embedded pi runs above the limit queue FIFO. |
| `agent.run_timeout_secs` | unsigned integer; `900` | `KILN_AGENT_RUN_TIMEOUT_SECS` (implemented) | `KILN_AGENT_RUN_TIMEOUT_SECS` | Must be at least `10`. A per-run timeout can override the server default. |
| `agent.runs_access` | string enum; `"loopback_only"` | `KILN_AGENT_RUNS_ACCESS` (implemented) | `KILN_AGENT_RUNS` (deprecated compatibility; boolean spellings map to enabled/disabled) | `loopback_only`, `enabled`, or `disabled`. Compatibility boolean spellings are accepted from the environment. Embedded runs can execute arbitrary code; changing this requires restart. |
| `agent.pi_bin` | optional path string; omitted (search startup PATH) | `KILN_AGENT_PI_BIN` (implemented) | `KILN_PI_BIN` (deprecated compatibility) | Must name a non-empty existing file when explicitly configured. The resolved executable is immutable for the process lifetime. |
| `agent.pi_sessions_dir` | optional path string; omitted (`$HOME/.pi/agent/sessions`) | `KILN_AGENT_PI_SESSIONS_DIR` (implemented) | `KILN_PI_SESSIONS_DIR` (deprecated compatibility) | Must be a non-empty path when set. Relative paths and the HOME fallback are resolved once at startup. |

Terminal and embedded-run access default to `loopback_only`. The host decision,
the pi executable, session root, adapter-library URL, and teacher-logit cache
root are resolved once during startup and published under `operational` in
`GET /v1/config`. Request handlers use that immutable snapshot; changing TOML,
`PATH`, `HOME`, or a compatibility environment alias requires a restart.

## Effective values and provenance

### Library runtime boundaries

Applications embedding `kiln-model` directly must initialize accelerator
memory governance as an explicit startup step. Call
`InferenceMemoryRuntime::initialize(device, governor_config)` before building
the runner, then pass the returned binding to
`ModelRunner::new_with_initialized_runtime`. Initialization validates the
backend/probe physical-device identity, detects safe capacity, treats
`GovernorConfig::capacity_limit_bytes` as a cap, publishes a live sample, and
starts the background sampler. The fallible runner constructor verifies the
binding against the model-weight device and installed process policy without
probing again. `ModelRunner::new`, `new_with_options`, and
`new_with_runtime_options` remain non-probing compatibility constructors for
owners such as `kiln-server` that install the same process-wide policy before
model construction.

Accelerator training has the same explicit identity rule. Construct
`TrainingRuntimeContext::new_for_device(device, effective_vram, policy)` and
use the `*_with_runtime` SFT, GRPO, streamed-GRPO, or OPD entry point. A context
without a device binding cannot authorize accelerator-backed weights. Weight
storage and native training execution must currently name the same device.
Vulkan serving uses CPU-host weight handles, but its full-model resident upload
path is not production-qualified and is therefore rejected before queue or
dataset admission with `training_backend_unsupported`; there is no environment
escape hatch. Vulkan-device training fixtures remain valid when their weights
are genuinely Vulkan-resident. Use the HF/TRL export/import workflow for
production training while serving through the hybrid Vulkan backend.

Kiln currently exposes several complementary, partial views:

- `kiln config --file <path>` performs authoritative parsing and validation,
  then prints a human-oriented subset of resolved fields, including the ROCm
  synchronization mode and configured/effective graph policy.
- `GET /v1/config` reports runtime diagnostics for serving profile, the
  versioned accelerator runtime policy, effective
  decode width, speculative configuration and availability, VRAM/KV state,
  native-training runtime/weight devices and
  support reason, the versioned implementation/resident-tuple/per-workload
  optimizer contract, gradient-checkpoint segmentation, the immutable SFT
  checkpoint-boundary policy, memory budgets, and generation defaults. Its
  `training.checkpoint_boundary_policy` object reports resolved recompute mode,
  inclusive automatic threshold, optional explicit stride, and integral cache
  target bytes. Its `speculative` object distinguishes `configured_method` and
  `configured_effective_method` from the actual `serving_effective_method`
  (`off`), reports `serving_routable=false`, the stable
  `serving_unavailable_reason`, the K ceiling, and immutable backend MTP
  support facts. It does not expose live routing, admission, or weight-readiness
  fields because those are not serving authorities. A running process cannot
  have a non-off configured method through the supported startup path.
  `training.native_training_supported=false` is accompanied by
  `training.native_training_unavailable_reason` and matches the admission
  error without scanning a corpus. `training.optimizer_support` separately
  exposes backend implementations, exact resident optimizer tuples, and
  per-workload executable kinds. Consumers must use `workloads[].supported` and
  `workloads[].allowed_optimizer_kinds`, not infer product support from
  `backend_implementation` or `optimizer_tuple_kinds`. When
  checkpoint execution is disabled,
  `training.checkpoint_segments` is `0`; an optional segment count retained in
  `training.checkpoint_policy` is provenance, not an active execution plan. The
  endpoint is not a serialization of all accepted TOML.
- `GET /health` and `/v1/health` repeat the accelerator policy and reasoned
  ROCm synchronization counters under `decode_runtime`, and repeat the resolved boundary policy under
  `training.checkpoint_boundary_policy`; trusted `GET /v1/debug/model-state`
  reports the identical object at the same path. The dashboard reads
  `/v1/config` and renders the same process-lifetime values. None of these
  surfaces parses configuration or rereads process environment.
- `GET /v1/config` also reports cached-sample age, maximum age, staleness,
  sampler requirement/liveness/health, cgroup `memory.high`, and the separately
  bounded `vram.live.raw_observations.host_backed` tier. `/health` exposes the
  same liveness decision, while `/metrics` exports fixed-cardinality gauges for
  alerting. A stale or failed sample remains visible diagnostically but
  contributes zero allocation headroom.
- `/health` and `/v1/debug/model-state` expose config hashes and other runtime
  identity data.
- Startup logs record serving-profile provenance and configured/backend/final
  decode-width sources.

Explicit source tracking (`default`, `config_file`, or `environment`) currently
exists for `server.serving_profile`, `server.deterministic`,
`server.stream_stall_grace_ms`, the three server batching/prefill budgets,
`server.max_decode_batch`, all eight `[batching]` fields, all six
`[streaming_prefill]` fields, all four `[accelerator]` fields, and
`memory.reclaim_mode`. Other fields have
resolved values but do not yet carry per-field source metadata.

The `kiln_env_config_hash` binds the serialized effective typed configuration
and the process's complete `KILN_*` environment map. It is an identity digest,
not a human-readable effective-config dump.

There is not yet one endpoint or CLI mode that dumps every effective field,
its source, backend-derived adjustment, and restart requirement. Until that
lands, use the typed validation command plus startup/runtime diagnostics.

## Known configuration migration limitations

These are current implementation facts, not recommended architecture:

1. **Effective dump:** neither `kiln config` nor `/v1/config` covers the whole
   typed object with provenance and backend-derived values.
2. **Deprecated aliases:** 61 non-canonical spellings across 58 fields remain
   temporarily for compatibility, including `KILN_DEFAULT_NO_THINK`. Each use
   warns at startup; canonical and compatibility names cannot silently
   disagree.

The intended migration is to resolve every public runtime field exactly once,
pass immutable typed configuration into its owner, generate the environment
contract mechanically, and reject direct production environment reads outside
that boundary. Internal experiment and qualification flags should remain
separate from this public API rather than being promoted by documentation.
