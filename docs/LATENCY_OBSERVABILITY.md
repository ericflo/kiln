# Latency Observability

Kiln records request-local token timing so concurrent streams are never joined
into a false inter-token interval. The same bounded observations feed chat
performance metadata, per-token SSE diagnostics, recent-request drill-down,
the rolling decode endpoint, and Prometheus.

This contract is diagnostic evidence, not a claim that every backend exposes
every internal phase. Unsupported or unavailable phases are `null`. A zero
means a supported phase was measured and took zero at the reporting
resolution; it must not be used as a substitute for missing instrumentation.

## Enable Request Timing

Set `include_performance` on a chat request:

```json
{
  "model": "Qwen3.5-4B",
  "messages": [{"role": "user", "content": "Explain the pause."}],
  "stream": true,
  "include_performance": true
}
```

For a non-streaming real-model request, the response carries
`metadata.performance.latency`. For a real-model streaming request, the
terminal chat chunk carries the request summary and each emitted model token
is followed by a `kiln.token_timing` SSE object. Custom timing objects are emitted
only after an explicit request opt-in; enabling the server-wide performance
metadata default does not silently add non-OpenAI SSE objects.

Completed instrumented requests also retain the summary in the bounded
`GET /v1/stats/recent-requests` ring. The dashboard renders it under **Latency
diagnosis**. Every real-model request uses the batching actor, so actor queue,
admission, prefill, decode, delivery, and backend phase attribution share one
request-local timing model.

## Timing Boundaries

All boundaries use one process-local monotonic clock. Values ending in `_ms`
are floating-point milliseconds relative to request receipt or elapsed between
the named boundaries.

| Boundary | Meaning |
| --- | --- |
| request receipt | Entry to chat request handling, before prompt rendering and tokenization |
| ready | The batching actor made the accepted token ready for delivery |
| producer delivered | The batching delivery worker enqueued the token into the actor-to-handler channel |
| handler received | The async chat producer received the bridged token event |
| body enqueued | The producer successfully enqueued the rendered content into the HTTP response body channel |

`body_enqueued` does not mean the remote client acknowledged network bytes.
TCP buffering, proxies, and client rendering are outside the server's clock.
The public field remains `client_delivery_ms` for compatibility, but its exact
boundary is handler receipt to response-body enqueue.

The compatibility `queue_delay_ms` field is
`handler_received_ms - ready_ms`. It equals
`response_delivery_ms + handler_queue_ms`. New consumers should use those two
components directly.

## Request Summary

`metadata.performance.latency` and a recent request's optional `latency` field
use the same `RequestLatencyDiagnostics` object:

```json
{
  "emitted_tokens": 3,
  "gap_samples": 2,
  "retained_gap_samples": 2,
  "gap_samples_truncated": false,
  "ttft_ms": 40.0,
  "itl_ms_p50": 8.0,
  "itl_ms_p99": 8.0,
  "itl_ms_p999": 8.0,
  "max_itl_ms": 8.0,
  "stall_threshold_ms": 250.0,
  "stall_count": 0,
  "unexplained_stall_count": 0,
  "stall_reasons": {
    "actor_queue": 0,
    "actor_admission": 0,
    "actor_prefill": 0,
    "actor_decode": 0,
    "actor_cycle_idle": 0,
    "response_delivery": 0,
    "handler_queue": 0,
    "client_delivery": 0,
    "sampling": 0,
    "readback": 0,
    "gpu_lock_wait": 0,
    "graph_capture": 0,
    "graph_replay": 0,
    "synchronization": 0,
    "resize": 0,
    "trim": 0,
    "adapter": 0,
    "training": 0,
    "unexplained": 0
  },
  "phases": {
    "actor_queue_ms": 12.0,
    "actor_admission_ms": 3.0,
    "tokenization_ms": 1.0,
    "prefill_ms": 20.0,
    "decode_ms": 8.0,
    "actor_cycle_idle_ms": 0.0,
    "sampling_ms": null,
    "readback_ms": null,
    "response_delivery_ms": 1.0,
    "handler_queue_ms": 1.0,
    "client_delivery_ms": 1.0,
    "gpu_lock_wait_ms": 0.2,
    "graph_capture_ms": null,
    "graph_replay_ms": null,
    "synchronization_ms": 0.5,
    "resize_ms": null,
    "trim_ms": null,
    "adapter_ms": null,
    "training_ms": null,
    "unexplained_ms": 0.0
  }
}
```

`emitted_tokens` includes the first token. `gap_samples` counts only intervals
between successive tokens from the same request, so it is normally
`emitted_tokens - 1`. A request retains at most 8,192 gaps. Once that cap is
exceeded, Kiln evicts the oldest samples, keeps the lifetime `gap_samples`
count, and sets `gap_samples_truncated=true`; percentiles and stall counts then
describe the retained suffix.

Percentiles use R-7 linear interpolation over request-local gaps. TTFT is
request receipt to first producer-ready token. A request with no token has `null`
TTFT and ITL values; a one-token request has TTFT but no ITL sample.

## Phase Attribution

The currently measured batching path exposes:

| Phase | Current boundary |
| --- | --- |
| `actor_queue_ms` | API enqueue until the actor begins admission |
| `actor_admission_ms` | Actor slot and request preparation that blocks active actor work |
| `tokenization_ms` | Prompt token encoding in the request handler |
| `prefill_ms` | Actor prefill wall time accumulated while the request is active |
| `decode_ms` | Actor decode wall time accumulated while the request is active |
| `actor_cycle_idle_ms` | Configured cooperative post-work idle elapsed while the request remained active |
| `sampling_ms` | Post-transformer final norm, LM head, penalties/filters, and selection when that tail has a distinct boundary; excludes separately measured readback |
| `readback_ms` | Existing device-to-host token transfer when the backend exposes that exact boundary without another synchronization |
| `gpu_lock_wait_ms` | Time waiting to acquire the shared inference GPU-coordination guard for actor-owned admission, prefill, or decode work |
| `graph_capture_ms` | Request-owned graph capture attempt work exposed by the selected backend |
| `graph_replay_ms` | Request-owned graph replay work exposed by the selected backend |
| `synchronization_ms` | Time spent settling actor-owned admission, prefill, resident-prefill, or decode work at the backend's external-yield boundary |
| `resize_ms` | Ordered KV-resize barrier time that delayed this queued request, or an overlapping GPU-writer resize interval |
| `trim_ms` | Exclusive allocator-pool trim time that overlapped this request's inference-lock wait |
| `adapter_ms` | Ordered adapter mutation barrier time that delayed this queued request |
| `training_ms` | Exclusive server-training GPU ownership that overlapped this request's inference-lock wait |
| `response_delivery_ms` | Producer-ready to bounded-channel enqueue or bridge receipt |
| `handler_queue_ms` | Producer delivery to handler receipt |
| `client_delivery_ms` | Handler receipt to response-body enqueue for streams |
| `unexplained_ms` | Wall time between tokens not covered by a measured phase candidate |

The terminal performance object also reports `resident_prefill_used`. It is
`true` only when that request completed at least one prompt token through a
successful native multi-row resident-prefill forward and `false` when the
request never entered that route. This is request-scoped route evidence, not an inference from process-global
counters; a native decline that performs no mutation leaves it `false`.
Production currently publishes `resident_prefill_enabled=false` and therefore
must emit `false` for every batching request: guarded Vulkan model runs found
semantic corruption on this optimization. Health, trusted debug state, and the
Prometheus capability gauge expose that quarantine explicitly.

Vulkan also publishes `prefix_cache_enabled=false`. Guarded runs with resident
token-prefill already disabled produced correct first-wave responses and
semantically corrupt exact repeats after cross-request KV/GDN restoration.
`GET /v1/config` distinguishes configured intent from this effective
`vulkan_correctness_quarantine`; health and trusted debug expose the live
capability, and Prometheus exports
`kiln_batching_engine_prefix_cache_enabled`. While false, every prefix-cache
lookup, hit, miss, residency, state-byte, lease, and pending-release value must
remain zero. Exact repeats therefore pay fresh generic-prefill latency. This is
an intentional correctness cost and must not be diagnosed as an undersized
cache from the absence of hits.

The batching engine measures `gpu_lock_wait_ms` and `synchronization_ms` on
owned admission, ordinary or resident prefill, and decode invocations. An
admission or prefill invocation blocks the single actor, so its backend
envelope is retained by the selected request and the already-active requests
whose next progress it delayed. A decode envelope remains attached only to the
ready requests that supplied rows to that step. The actor likewise adds the actual elapsed cooperative
`batching.actor_cycle_idle_ms` wait only to rows that remain active when the
wait completes. The phase is measured as zero when that policy is disabled,
and can become the dominant `actor_cycle_idle` stall reason; it is not folded
into actor decode or left as unexplained wall time. The actor path also measures
`tokenization_ms`, producer-to-delivery-worker `response_delivery_ms`,
delivery-worker-to-handler `handler_queue_ms`, response-body enqueue
`client_delivery_ms`, request-local `unexplained_ms`, and the invocation-owned
backend phases described below. The model event envelope carries its distinct
sampler tail and external-yield synchronization wait with the token they produced. A
decode that selects EOS instead of emitting another token carries those phases
on the terminal event, so successful request totals do not discard the final
backend invocation. Measured zero remains distinct from an unsupported `null`
phase.

`sampling_ms` is a caller-wall-time candidate around the existing
sampler operation. Because accelerator work may be asynchronous, it can include
completion of work submitted immediately before that sampler; it is not claimed
as isolated GPU-kernel time and is not added to another broad phase. Kiln does
not insert a synchronization to manufacture a cleaner number. Fused token
routes leave sampling `null` when they do not expose a separate sampler
boundary, and readback remains `null` until that route's backend owner
can split the existing transfer exactly.

Qualified ROCm W8 sampled paged decode reports `sampling_ms` for the distinct
post-transformer tail and `readback_ms` for the existing token-index
`to_vec1` device-to-host transfer. The readback timer surrounds that transfer
inside the backend owner; it adds no operation or synchronization. The decode
envelope subtracts the exact readback duration from sampling before both are
published, so the two candidates do not double-count. The bounded top-k batch
and single-row full-distribution Gumbel routes both carry this envelope.

Every profiled ordinary-batching, behavior-logprob, and direct decode invocation
also owns a nested-safe readback scope. Exact existing accelerator host reads
inside that scope contribute only to its result. This covers generic greedy
scalar and row argmax, penalty gathers, Gumbel token reads, full-distribution
and host-top-k reads, CUDA softmax and scalar argmax reads, CUDA/ROCm device
top-k pair reads, ROCm W8 greedy reads, and Metal LM-head greedy, row, and
sampled-token reads. Value-only W8 convenience APIs forward their backend-owned
duration into the scope; APIs that return an explicit profiled duration do not,
so the caller can merge it once. Nested invocations write only to the innermost
scope, failed invocations remove their scope, and reads outside an owned decode
invocation are not assigned to a request.

The timer surrounds only an existing host-transfer call. CUDA and ROCm device
top-k sum the two existing value/index reads while excluding kernel launch and
host conversion. Because those calls establish an existing completion boundary,
their wall time can include waiting for earlier asynchronous work on the same
stream; `readback_ms` is therefore an owned host-read boundary, not pure DMA
engine time. No timer inserts a copy, wait, or synchronization. When a sampler
wall candidate exists, the invocation merge subtracts these exact readback
durations before adding them to `readback_ms`; explicit ROCm W8 sampled
durations and scope observations remain single-counted.

Vulkan's native fused token samplers remain deliberately `null`: their one
submission combines compute, device-to-host copy, completion wait, and mapped
host access, so host wall timing cannot isolate the copy without device
timestamps or another synchronization. A generic Vulkan sampler fallback with
an independently exposed `to_vec1` read can report that exact boundary. Greedy
and native fused-forward routes still leave `sampling_ms` `null` when the
transformer/sampler boundary itself is not independently observable.

The exact clean Strix Halo ROCm development receipt at
`qualification/receipts/rocm/strix-halo/20260721t121713716043z-rocm-strix-halo-serving-rocm-development-053e89eca9-v1.json`
proves the production transport: all 725 measured requests carried positive
readback for 376,397.090 ms total with zero missing phase metadata. The paired
short exact-prompt receipt at
`qualification/receipts/rocm/strix-halo/20260721t131545614136z-rocm-strix-halo-serving-mixed-rocm-v1-184c082f9e-v1.json`
carried the boundary on all 10 deterministic and all eight fused-sampled
requests. These are request-ownership results, not pure-transfer benchmarks;
the qualification guide separately preserves the slow long-soak performance
counterexample and its narrower source-effect discriminator.

Qualified ROCm HIP-graph decode reports request-owned `graph_capture_ms` and
`graph_replay_ms` on batching-engine streams. The graph runner
holds its mutex across one decode invocation and snapshots the fixed phase
counters before and after that call, so another request cannot contaminate the
delta. Capture is the sum of candidate headroom, warm, reservation, native
capture, and rejected-candidate cleanup phases that completed in the call; it
is retained even when the request continues through eager fallback. Replay
covers device input updates, cross-stream dependency setup, and native launch
submission. The later external-yield device settlement remains
`synchronization_ms` and is not double-counted as replay.

CUDA and Metal use the same request envelope without deriving deltas from
process-global counters. Each ordinary batching, behavior-logprob batching,
or direct decode invocation opens an invocation-local graph phase scope on the
calling thread. Native graph owners add their capture and replay wall time only
to the innermost active scope, and the completed scope is merged into the token
or terminal event produced by that invocation. A failed capture attempt is
therefore retained when execution continues eagerly, while graph work outside
an owned model invocation is not assigned to an unrelated request. Nested
scopes are isolated. ROCm and Vulkan builds compile the CUDA/Metal graph scope
machinery out; their separate readback scope remains available for exact sampler
host reads.

For CUDA, capture surrounds the complete existing single-row or dormant
batched capture attempt, including failure settlement. Replay surrounds the
native `ReplayPlan` call, or the dormant batched graph launch plus its existing
capture-stream completion wait. Input refresh ordering that completes before
that replay boundary, eager LM-head work, sampling, and the later model
external-yield settlement are excluded. For Metal, capture surrounds each
participating attention layer's ICB record/build operation. Replay is the sum
of the participating layer ICB operations, including their scalar argument
updates, native replay, and existing completion wait. These timers add no
device operation or synchronization. CUDA and Metal source boundaries are
portable contract coverage; their runtime values remain unqualified until the
required NVIDIA and Apple hardware campaigns pass.

A batching-engine graph invocation is shared work. Its phase envelope is
attached once to every ready request that supplied a row to that invocation so
each request can explain the wall interval it overlapped. Prefilling,
backpressured, first-token-pending, and otherwise non-ready active requests are
not charged. Consequently, request-level totals are not process-work totals:
they may duplicate one shared invocation by at most the observed decode width.
The ROCm mixed-load gate enforces that capture and replay request totals do not
exceed their matching lifetime phase totals multiplied by maximum observed
decode width. Use lifetime graph telemetry for device-work totals and request
phases for causal overlap.

Resize and adapter attribution use an actor-barrier timeline that is separate
from GPU-lock ownership. The first queued exclusive mutation starts its typed
interval as soon as the actor accepts the command, including any time spent
draining the active batch; ordered mutations switch to the next typed interval
only after the prior mutation returns. A request snapshots that timeline before
enqueue and receives only intervals that overlap its queue-to-admission
boundary. Requests already active when a mutation is requested continue to
completion and are not charged for the later barrier. This makes a non-null
`resize_ms` or `adapter_ms` a causal explanation for queued request latency,
not a process-global event correlation.

The cached health and trusted-debug batching snapshots expose the live bounded
gauges `actor_barrier_resize_active` and `actor_barrier_adapter_active`.
Prometheus publishes the same state as
`kiln_batching_engine_actor_barrier_resize_active` and
`kiln_batching_engine_actor_barrier_adapter_active`. A gauge is true from actor
command acceptance through active-request drain and mutation completion. It is
an operator-visible coordination signal, not request attribution: a request
that was already active remains uncharged, while only a later request whose
enqueue-to-admission boundary overlaps the interval receives non-null phase
time.

Resize, trim, and server-training writers also use a distinct GPU-ownership
timeline. An inference acquisition first tries the shared lock without
touching telemetry. Only after contention does it snapshot the writer timeline,
wait for its read guard, and retain the exact typed intervals that overlapped
that wait. `gpu_lock_wait_ms` covers the complete acquisition wait while
`resize_ms`, `trim_ms`, and `training_ms` are narrower causal candidates inside
it; they must not be added together. Training observation starts only after a
step, checkpoint, or other coordinated training phase owns the write guard and
ends before that guard is released. Allocator trim observation follows the
same ordering. CUDA, ROCm, and Vulkan reclaimers all require an idle actor, a
healthy backend, and a nonblocking exclusive acquisition before trimming; a
racing request waits safely and receives the trim overlap instead of allowing a
pool mutation underneath live inference.

Stable serving disables resize, trim, adapter mutation, and server training,
so those fields normally remain `null` there. They are also `null` when a
permitted operation did not overlap the request's relevant ownership boundary.
Backends without graph execution and device routes without an independently
owned readback boundary remain explicit nullable fields. Kiln never infers request
time from unlocked process-global counters. A diagnostic consumer must preserve
the difference between `null` (not observed or not measurable on this path) and
`0` (measured below the reporting resolution).

The mixed-load, development-soak, and endurance qualification clients preserve
that distinction in compact receipts. For each fixed phase `P`, they emit
`latency_phase_P_ms_total` and `latency_phase_P_request_count`; the total sums
only non-null request observations, while the count identifies the population
that contributed to it. `latency_phase_metadata_missing_count` counts
successful measured requests that omitted the entire terminal phase object and
is a hard qualification failure. Shared batch-phase aggregates are additionally
guarded by the lifetime-times-width conservation bound above. These are
source-bound measured-window
aggregates, not values reconstructed from threshold-filtered logs. Older
receipts created before this contract are not backfilled.

Serving comparison driver v15 preserves the full terminal performance object
for every successful Kiln request as ordered `request_performance` evidence and
derives `request_phase_summary` from it. Each phase and terminal request metric
has an `observed_request_count`, p50, p99, and maximum; missing phases therefore
remain distinguishable from measured zero. The strict validator reconciles the
objects with independent usage/output records and recomputes the summary. The
comparison request body remains common with vLLM: typed server configuration
enables Kiln's terminal summary, while per-token timing objects remain disabled.
Because actor/device work can be shared across concurrent requests, these phase
distributions are request-impact evidence, not additive service-wall totals.

The ROCm mixed-load workload runs a separate fixed-seed sampled profile before
its ordinary measurement window: eight concurrent requests, 32 output tokens
each, temperature 0.7, top-p 0.9, top-k 40, and min-p 0.0. It waits for the
batching engine to drain and takes a fresh health baseline before starting the
existing greedy workload, so sampled-profile counters cannot contaminate the
deterministic mixed-load metrics. Dedicated `sampled_profile_*` metrics retain
request/completion counts, aggregate completion-window throughput, median
per-request end-to-end throughput, TTFT/E2E percentiles, decode and sampling
phase totals/populations/percentiles, the readback population, and decode
forward/row/batch-width evidence. The sampled semantic oracle requires one
nonempty plain-text choice with thinking and tools absent; it does not require
random output to match the greedy ascending-integer oracle. A pass requires all
eight exact-length requests, terminal phase metadata, a positive sampled-tail
duration for every request, zero separate readback observations, and proven
multi-row decode. Aggregate throughput describes the eight-request service
window; the per-request p50 describes individual request wall time. They answer
different questions and must not be substituted for one another.

The same mixed-load window retains host thermal pacing as an external causal
interval. The Strix Halo controller samples `k10temp/Tctl` every 250 ms, stops
the complete server process group at 88 C, resumes it at 86 C, and independently
fails closed at 97 C. A token gap overlapping a stop/resume interval is assigned
the bounded `host_thermal_pacing` category. Start/peak/end temperature, guard
errors and trips, pacing counts, duration, maximum interval, and active-at-end
state are receipt metrics. Cooling time remains inside request and workload wall
clocks, so throughput and latency describe sustainable service on the named
host rather than an unpaced burst. This external attribution does not change or
populate any request-local backend phase. Mixed-load receipts partition the
total attributed gap count into thermal-paced and non-thermal runtime counts;
only a fully reconciled thermal-only attributed population may pass, and the
unexplained count must remain zero. The client and server share a 180-second
request containment bound, with pacing time still included. At teardown, pacing is disabled and
any active stop is released before `SIGTERM`, while hard-limit monitoring stays
active through server exit. Post-exit cooldown is not an ITL category because no
request remains: the runner separately requires eight consecutive 250 ms samples
at or below 75 C within 180 seconds and retains its duration, sample count,
stable count, peak, completion, timeout, and active-at-end evidence.

Phase values are blocking candidates, not an additive critical-path
decomposition. Work can overlap, especially response delivery with the next
actor forward, and the broad `decode_ms` interval contains any measured backend
subphases. Do not sum every phase and compare it with total request time.

For ROCm graph candidates, `event=rocm_graph_capture_host_transfer` attributes
each unique host-upload site observed by the thread- and device-local warm-pass
scope. Its closed fallback reason and build-source file/line/column are paired
with dtype, elements/bytes per copy, copy count, and total bytes. A candidate is
bounded to 32 unique sites; additional sites collapse into one
`source_file=bounded_site_overflow` row. This is causal capture-eligibility
evidence, not a request payload or a Prometheus label, and it must be read next
to `rocm_graph_fallback` attempt/eager/total duration and graph capture/replay
counters.
Production ROCm materialized broadcast passes its at-most-eight-rank
shape/stride descriptor as a captured kernel value. It does not allocate or
upload device metadata per broadcast. A transfer event naming
`rocm_ops/index_select.rs` metadata construction therefore identifies a stale
binary or a regression, not expected cold-cache behavior.
Each new batched graph also emits one
`event=rocm_graph_capture_parity_check` after its settled first launch. Read its
`outcome`, `comparison_complete`, `duration_ms`, and `compared_bytes` with
`hidden_match` for a completed comparison, plus the first recurrent,
convolution, K-cache, or V-cache mismatch layer when one exists, or `error` when
the comparison kernel itself fails. The check compares against the candidate's own eager warm
execution, uses device-side exact equality with scalar readback, and is part of
`native_capture` time. A failed or errored check is a capture failure followed
by settled rollback and contained eager retry, never a successful admission.
Its snapshot, current-row gather, and equality-mask reservation is included in
the transient-candidate high-water mark rather than hidden as unattributed
allocator activity.
The detailed event is complemented by durable process-lifetime fields under
`GET /health.decode_runtime.rocm_graphs`: `batched_capture_attempts`,
`batched_capture_successes`, `batched_capture_deferrals`,
`batched_capture_failures`, `capture_parity_checks`, `capture_parity_passes`,
`capture_parity_failures`, `capture_parity_errors`,
`capture_parity_compared_bytes`, and `capture_parity_duration_micros`. Batched
attempts equal their three outcomes, parity checks equal their three outcomes,
and successful batched captures cannot exceed passed checks. A clean
qualification window additionally requires zero failure/error and equal
success/pass deltas; a later post-parity admission error may otherwise leave a
pass without a successful admission.

Prometheus publishes the same cumulative evidence as
`kiln_rocm_graph_batched_capture_attempts_total`,
`kiln_rocm_graph_batched_capture_outcomes_total{outcome}`,
`kiln_rocm_graph_capture_parity_checks_total`,
`kiln_rocm_graph_capture_parity_outcomes_total{outcome}`,
`kiln_rocm_graph_capture_parity_compared_bytes_total`, and
`kiln_rocm_graph_capture_parity_duration_seconds_total`. These fixed-cardinality
counters survive log rotation and are the source for serving-receipt deltas;
the structured event remains the source for mismatch-layer attribution.
When computing `unexplained_ms`, Kiln conservatively subtracts the larger of
serial actor work, response delivery, and the largest backend candidate rather
than their sum, so overlap cannot falsely erase missing wall time.

## CUDA Synchronization Counters

CUDA host waits are attributed at the driver call rather than inferred from a
request pause. `GET /health.decode_runtime.cuda_synchronization` and
`GET /v1/debug/model-state.cuda_synchronization` expose the same process-lifetime
snapshot:

| Field | Meaning |
| --- | --- |
| `active` | The selected model-weight device is CUDA |
| `telemetry_available` | The binary contains CUDA telemetry for the selected CUDA device |
| `telemetry_error` | Why an active CUDA device cannot provide telemetry; omitted when there is no error |
| `total_device_wait_count` | Sum of context-wide wait attempts across all reasons |
| `total_stream_wait_count` | Sum of stream wait attempts across all reasons |
| `total_failure_count` | Sum of failed waits across all reasons |
| `total_waited_ns` | Host monotonic wall time spent inside all completed wait calls |
| `reasons` | Exactly twelve fixed reason rows for active CUDA telemetry; empty for an inactive backend |

Each reason row has `device_wait_count`, `stream_wait_count`, `failure_count`,
and `waited_ns`. An attempt is recorded after the driver call returns, including
failed calls. Reading the snapshot touches only relaxed process atomics: it does
not create a context, query the driver, allocate device memory, or synchronize.
Totals are saturating and must equal the sum of the reason rows in a settled
snapshot.

| Reason | Boundary |
| --- | --- |
| `explicit_device_drain` | An operator or diagnostic explicitly requests a context-wide drain |
| `explicit_stream_drain` | An operator, diagnostic, or compatibility path explicitly drains one stream |
| `tensor_handoff` | A tensor must be ready before another subsystem consumes it |
| `external_yield` | Accelerator work must settle before the scheduler publishes progress or releases ownership |
| `in_place_mutation` | An asynchronous in-place write must finish before the caller proceeds |
| `memory_reclaim` | In-flight work must settle before allocator pages can be released |
| `graph_boundary` | CUDA graph capture, first launch, replay input, or replay completion ordering requires a host wait |
| `full_attention_handoff` | Full-attention output crosses an ownership boundary |
| `model_handoff` | Model output crosses from model execution to its caller |
| `host_readback` | Device work must finish before an existing host readback is consumed |
| `allocation_lifetime` | An asynchronous operation must finish before temporary storage can be reused or dropped |
| `global_state_mutation` | Device work must settle before process-visible state such as KV capacity is changed |

Not every workload exercises every reason; an unexercised reason remains a
real zero, not missing telemetry. Production CUDA context and stream waits go
through the typed wrapper, so adding a raw driver wait is a source-contract
regression. The compatibility `cuda_synchronize_default_stream` helper records
`explicit_stream_drain`; owners with a narrower boundary use the reasoned form.

Prometheus exports the same cumulative state:

```text
kiln_cuda_synchronization_active
kiln_cuda_synchronization_telemetry_available
kiln_cuda_synchronizations_total{reason,scope="device|stream"}
kiln_cuda_synchronization_failures_total{reason}
kiln_cuda_synchronization_wait_seconds_total{reason}
```

Reason and scope are closed labels. Device index, request identity, model,
adapter, and error text never become labels. The wait-seconds value is
`waited_ns / 1e9`; a qualification snapshot must reconcile it with the JSON
counter at normal floating-point rendering precision.

These counters are operational device-work evidence. Request-local
`synchronization_ms` remains the invocation-owned external-yield wall time and
must not be replaced by unlocked before/after deltas from these global
counters. A request can overlap graph, model-handoff, or allocation-lifetime
waits owned by shared work, and process counters can include maintenance or
training outside that request.

For each inter-token gap, Kiln selects the largest measured candidate since
the preceding token. For classification, the broad actor-decode candidate is
reduced by only the largest measured backend subphase; detailed phases may
overlap, so subtracting their sum would manufacture coverage. A candidate must
explain at least the smaller of 250 ms or half the gap; otherwise the reason is
`unexplained`. The bounded reason set is `actor_queue`, `actor_admission`,
`actor_prefill`, `actor_decode`, `actor_cycle_idle`, `response_delivery`,
`handler_queue`,
`client_delivery`, `sampling`, `readback`, `gpu_lock_wait`, `graph_capture`,
`graph_replay`, `synchronization`, `resize`, `trim`, `adapter`, `training`, and
`unexplained`. Reasons whose phase is unsupported remain zero.

## Per-Token SSE Object

An opted-in real-model stream emits this non-chat object after the corresponding
content has entered the response-body channel:

```json
{
  "object": "kiln.token_timing",
  "source": "batching_engine",
  "token_index": 7,
  "token_id": 4242,
  "ready_ms": 12.0,
  "producer_delivered_ms": 14.0,
  "handler_received_ms": 17.0,
  "body_enqueued_ms": 20.0,
  "response_delivery_ms": 2.0,
  "handler_queue_ms": 3.0,
  "queue_delay_ms": 5.0,
  "client_delivery_ms": 3.0,
  "blocking_phase": "actor_prefill",
  "blocking_phase_ms": 280.0
}
```

`source` is the closed value `batching_engine`. The field remains explicit so
captured diagnostics identify the scheduling authority without inference from
other payload fields.

`token_id` is the exact accepted model token represented by this timing row.
It remains present when the tokenizer decodes a special token to an empty text
fragment, allowing an opted-in client to distinguish hidden-token generation
from a missing delivery event. This is additional diagnostic disclosure: it can
identify a special token that produced no visible response text. It does not
include logprobs, candidate tokens, or behavior-policy capture.

`blocking_phase` and `blocking_phase_ms` are `null` for the first token because
there is no preceding token gap. They can also be `null` when no gap
observation exists. Clients that require a strict OpenAI-only SSE stream
should leave `include_performance` false and use the terminal response,
recent-request endpoint, or Prometheus instead.

## Rolling Decode Endpoint

`GET /v1/stats/decode` describes request-local gaps observed during the last
60 seconds:

- `sample_count`: retained gaps in the active rolling window
- `tok_per_sec`: gap count divided by the sum of those gaps
- `p50_itl_ms`, `p99_itl_ms`, `p999_itl_ms`, `mean_itl_ms`, `max_itl_ms`
- `stall_threshold_ms`: `max(250 ms, 5 * p50_itl_ms)`
- `stall_count`, `unexplained_stall_count`, and fixed `stall_reasons`

This throughput is a decode cadence diagnostic, not end-to-end request
throughput. The rolling ring is bounded, and concurrent requests are never
cross-paired. When the window is empty, counts and numeric summaries are zero.

## Prometheus

`GET /metrics` exports only fixed-cardinality labels:

| Metric | Type | Meaning |
| --- | --- | --- |
| `kiln_request_ttft_seconds` | histogram | Request receipt to first producer-ready token |
| `kiln_token_itl_seconds` | histogram | Every request-local inter-token gap |
| `kiln_request_latency_phase_seconds{phase}` | histogram | Measured request phase; unsupported `null` phases emit no observation |
| `kiln_token_stalls_total{reason}` | counter | Gaps at or above the fixed 250 ms absolute floor by dominant reason |

Histogram buckets are 5 ms, 10 ms, 25 ms, 50 ms, 100 ms, 250 ms, 500 ms,
1 s, 2.5 s, 5 s, 10 s, 30 s, 60 s, and `+Inf`. The `phase` and `reason`
labels use only the closed sets in the schemas; request IDs, model names,
adapter names, prompts, and clients are never labels.

Prometheus stall counters deliberately use the fixed 250 ms floor so a
counter's definition never changes with workload mix. The rolling endpoint and
per-request summary use the adaptive `max(250 ms, 5 * p50)` threshold. Account
for that difference when comparing the two surfaces.

Example queries:

```promql
histogram_quantile(0.999, sum by (le) (rate(kiln_token_itl_seconds_bucket[5m])))
sum by (reason) (rate(kiln_token_stalls_total[5m]))
histogram_quantile(0.99, sum by (phase, le) (rate(kiln_request_latency_phase_seconds_bucket[5m])))
```

## Measurement Cost

The per-token path uses monotonic timestamps, bounded `VecDeque` insertion,
fixed-index relaxed atomics, and one short rolling-ring mutex hold. It performs
no serialization, percentile sort, logging, or unbounded label lookup while a
token is being recorded. Percentiles are sorted once when a request summary or
rolling snapshot is requested.

`latency_measurement_hot_path_is_bounded` measures 20,000 token observations,
including request tracking, Prometheus histograms, rolling-ring insertion, and
both final percentile snapshots. The portable debug-build gate allows at most
100 microseconds per token and reports the measured nanoseconds per token on
failure or with test output enabled. The per-request retained sample storage is
also statically limited to 256 KiB; the current 8,192-sample representation
must fit below that limit. These are regression ceilings, not performance
claims for a particular accelerator.

## Diagnose A Pause

1. Confirm the gap is request-local in the recent-request drill or
   `kiln.token_timing`; do not infer it from adjacent global token timestamps.
2. Check the dominant blocking reason and its duration. Treat `unexplained` as
   missing evidence, not as proof of a device or allocator fault.
3. Correlate the same monotonic interval with typed actor, synchronization,
   graph, allocator, KV-resize, trim, adapter, and training telemetry.
4. On Vulkan, inspect `caches.resident_recurrent_state` in trusted model-state
   diagnostics. Prompt residency is correctness-quarantined, so any nonzero
   entry or byte count during current production prompt execution is unexpected
   route activation or an ownership defect, not VRAM rebalancing. Compare
   `buffer` with `allocation` bytes before attributing an aligned recycler
   allocation to live tensor growth. In an explicitly test-enabled repair, if
   entry count grows by roughly the number of GDN layers after cancellation,
   verify that prefill entered an explicit request-owner scope, each recurrence
   entered its linear layer scope, successful handoff materialized by
   `(request, layer)`, and discard evicted that request owner. Tensor-handle
   rekeying is only a secondary alias and must not be the sole cleanup authority.
5. If ownership drains to zero but a same-process or standalone semantic oracle
   fails in a test-enabled resident arm after apparently correct prompt logits,
   compare the recurrent state's external precision at every completed
   token-chunk boundary. An F32 accumulator for a BF16 state must still apply
   the device-side BF16 round-to-nearest-even boundary before the next chunk.
   The existing small-state test passes this check bit-for-bit but the full
   model still fails, so precision parity alone is not evidence to lift the
   quarantine.
6. Change one source-bound configuration field per benchmark arm and preserve
   failed receipts. A temporal gap without a matching typed operation is not
   evidence of VRAM rebalancing.

The authoritative machine-readable field contracts are
`contracts/kiln-inference-v1.schema.json` and
`contracts/kiln-observability-v1.schema.json`.
