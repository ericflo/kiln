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

For a non-streaming batching-engine request, the response carries
`metadata.performance.latency`. For either real-model streaming path, the
terminal chat chunk carries the request summary and each emitted model token
is followed by a `kiln.token_timing` SSE object. Custom timing objects are emitted
only after an explicit request opt-in; enabling the server-wide performance
metadata default does not silently add non-OpenAI SSE objects.

Completed instrumented requests also retain the summary in the bounded
`GET /v1/stats/recent-requests` ring. The dashboard renders it under **Latency
diagnosis**. Direct streaming records model-producer and bridge boundaries,
but leaves actor-only phases `null` and attributes the otherwise unpartitioned
interval between model-ready tokens to `unexplained`.

## Timing Boundaries

All boundaries use one process-local monotonic clock. Values ending in `_ms`
are floating-point milliseconds relative to request receipt or elapsed between
the named boundaries.

| Boundary | Meaning |
| --- | --- |
| request receipt | Entry to chat request handling, before prompt rendering and tokenization |
| ready | The batching actor or direct model producer made the accepted token ready for delivery |
| producer delivered | The batching delivery worker enqueued the token into the actor-to-handler channel, or the direct bridge received it from the model channel |
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
| `sampling_ms` | Post-transformer final norm, LM head, penalties/filters, selection, and token return when that tail has a distinct boundary |
| `gpu_lock_wait_ms` | Time waiting to acquire the shared inference GPU-coordination guard for a decode step |
| `synchronization_ms` | Time spent settling the decode step at the backend's external-yield boundary |
| `response_delivery_ms` | Producer-ready to bounded-channel enqueue or bridge receipt |
| `handler_queue_ms` | Producer delivery to handler receipt |
| `client_delivery_ms` | Handler receipt to response-body enqueue for streams |
| `unexplained_ms` | Wall time between tokens not covered by a measured phase candidate |

The terminal performance object also reports `resident_prefill_used`. It is
`true` only when that request completed at least one prompt token through a
successful native multi-row resident-prefill forward, `false` when a batching
request never entered that route, and `null` on direct paths that cannot use
it. This is request-scoped route evidence, not an inference from process-global
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

The batching engine measures `gpu_lock_wait_ms` and `synchronization_ms` on the
owned decode invocation and propagates those observations only to requests
active for that step. Direct streaming measures `tokenization_ms`, model-ready-to-bridge
`response_delivery_ms`, bridge-to-handler `handler_queue_ms`, response-body
enqueue `client_delivery_ms`, and request-local `unexplained_ms`. Its
`actor_queue_ms`, `actor_admission_ms`, `prefill_ms`, and `decode_ms` fields are
`null`, because that path does not expose actor phase boundaries; its backend
subphase fields also remain `null` until the direct model path returns the same
invocation-owned timing envelope.

ROCm sampled paged decode reports `sampling_ms` for the distinct
post-transformer tail, including the final token transfer that completes the
call. Behavior-logprob capture reports the same boundary on every backend.
`readback_ms` remains `null` because the current backend interface cannot split
transfer time from the sampling call without adding a second synchronization;
the broad sampling candidate must not be added to a future nested readback
candidate. Greedy and native fused-forward routes leave sampling `null` when
the transformer/sampler boundary is not independently observable.

Device readback, graph capture/replay, resize, trim, adapter, and training
remain explicit nullable fields. Their aggregate subsystems have
separate operational telemetry, but they are not yet joined to each
request-token timeline. In particular, Kiln does not infer a request's graph
time from process-global before/after counters because concurrent inference
could make that attribution false. A diagnostic consumer must preserve the
difference between `null` (not measured on this path) and `0` (measured with no
material elapsed time).

The mixed-load, development-soak, and endurance qualification clients preserve
that distinction in compact receipts. For each fixed phase `P`, they emit
`latency_phase_P_ms_total` and `latency_phase_P_request_count`; the total sums
only non-null request observations, while the count identifies the population
that contributed to it. `latency_phase_metadata_missing_count` counts
successful measured requests that omitted the entire terminal phase object and
is a hard qualification failure. These are source-bound measured-window
aggregates, not values reconstructed from threshold-filtered logs. Older
receipts created before this contract are not backfilled.

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

Phase values are blocking candidates, not an additive critical-path
decomposition. Work can overlap, especially response delivery with the next
actor forward, and the broad `decode_ms` interval contains any measured backend
subphases. Do not sum every phase and compare it with total request time.
When computing `unexplained_ms`, Kiln conservatively subtracts the larger of
serial actor work, response delivery, and the largest backend candidate rather
than their sum, so overlap cannot falsely erase missing wall time.

For each inter-token gap, Kiln selects the largest measured candidate since
the preceding token. For classification, the broad actor-decode candidate is
reduced by only the largest measured backend subphase; detailed phases may
overlap, so subtracting their sum would manufacture coverage. A candidate must
explain at least the smaller of 250 ms or half the gap; otherwise the reason is
`unexplained`. The bounded reason set is `actor_queue`, `actor_admission`,
`actor_prefill`, `actor_decode`, `response_delivery`, `handler_queue`,
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

`source` is the closed value `batching_engine` or `direct`. The
`producer_delivered_ms` boundary is therefore meaningful without pretending a
direct stream uses the batching actor.

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
