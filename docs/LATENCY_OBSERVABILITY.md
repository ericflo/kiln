# Latency observability

Kiln measures token timing per request. Tokens from concurrent streams are
never paired with one another, so an inter-token latency (ITL) sample always
describes one request.

Use this page to answer three different questions:

| Question | Start here |
|---|---|
| How long did one request wait for its first token? | `ttft_ms` in response metadata or the recent-request view |
| Where did a pause between two tokens occur? | `kiln.token_timing`, request phase totals, and stall reasons |
| Is tail latency changing across the server? | `GET /v1/stats/decode` and `GET /metrics` |

These surfaces provide diagnostic wall-clock evidence. They are not
device-kernel profiles, and they do not all measure the same population.

## Choose an observation surface

| Surface | Scope | Important limitation |
|---|---|---|
| Response `metadata.performance.latency` | One completed request | Available only when performance metadata is enabled |
| `kiln.token_timing` SSE events | One token at a time in one stream | Kiln-specific events; require explicit request opt-in |
| `GET /v1/stats/recent-requests` | Bounded recent request history | Can include prompt and completion text; protect this endpoint |
| Dashboard **Latency diagnosis** | Human-readable recent-request drill-down | Shows the same bounded server-side observations |
| `GET /v1/stats/decode` | Request-local gaps retained from the last 60 seconds | Decode cadence, not end-to-end throughput |
| `GET /metrics` | Process aggregates | Fixed-cardinality histograms and counters, not request traces |

Kiln does not authenticate the observability endpoints. Keep the server on
loopback or place it behind an authenticated reverse proxy, especially when
recent-request data is enabled.

## Enable request timing

Set `include_performance` on a chat request:

```json
{
  "model": "Qwen3.5-4B",
  "messages": [
    {
      "role": "user",
      "content": "Explain the pause."
    }
  ],
  "stream": true,
  "include_performance": true
}
```

For a non-streaming real-model request, Kiln adds the summary at
`metadata.performance.latency`. For a streaming real-model request, the
terminal chat chunk carries the same summary.

An explicit request value of `include_performance: true` also adds a
`kiln.token_timing` event after each emitted model token. The server-wide
performance-metadata default can enable terminal metadata, but it does not
silently add Kiln-specific objects to an otherwise OpenAI-compatible stream.
Leave the request field false or omitted when a client accepts only standard
chat chunks.

## Understand the clocks

All request boundaries use one process-local monotonic clock. Fields ending in
`_ms` are floating-point milliseconds.

| Boundary | Exact meaning |
|---|---|
| Request receipt | Chat request handling begins, before prompt rendering and tokenization |
| Ready | The batching actor has produced an accepted token |
| Producer delivered | The delivery worker enqueues the token for the HTTP handler |
| Handler received | The asynchronous chat producer receives the token event |
| Body enqueued | The producer enqueues rendered content into the HTTP response body |

`body_enqueued` does not mean that the remote client received or rendered the
bytes. TCP buffers, proxies, network transit, and client work are outside
Kiln's clock.

The compatibility field `queue_delay_ms` is:

```text
response_delivery_ms + handler_queue_ms
```

The public name `client_delivery_ms` is also retained for compatibility. Its
actual boundary is handler receipt to response-body enqueue—not delivery to
the remote client.

## Read a request summary

The summary reports:

| Field | Meaning |
|---|---|
| `emitted_tokens` | Generated tokens emitted by this request, including the first token |
| `gap_samples` | Lifetime count of gaps between consecutive tokens from this request |
| `retained_gap_samples` | Gaps still retained for percentile and stall calculation |
| `gap_samples_truncated` | Whether older samples were evicted from the bounded request buffer |
| `ttft_ms` | Request receipt to the first ready token |
| `itl_ms_p50`, `itl_ms_p99`, `itl_ms_p999` | R-7 percentiles over retained request-local gaps |
| `max_itl_ms` | Largest retained request-local gap |
| `stall_threshold_ms` | Adaptive request threshold: `max(250 ms, 5 × p50 ITL)` |
| `stall_count` | Retained gaps at or above that threshold |
| `unexplained_stall_count` | Stalls for which no measured phase explains enough of the gap |
| `stall_reasons` | Stall counts by dominant measured reason |
| `phases` | Request-impact time by bounded phase |

`gap_samples` is normally `emitted_tokens - 1`. Kiln retains at most 8,192
gaps per request. After that limit, percentiles and stall counts describe the
retained suffix while `gap_samples` continues to count the full request.

A request that emits no token has null TTFT and ITL values. A one-token request
has TTFT but no ITL sample.

## Interpret phase attribution

| Phase | Current boundary |
|---|---|
| `actor_queue_ms` | API enqueue until the batching actor begins admission |
| `actor_admission_ms` | Actor work that prepares and admits the request |
| `tokenization_ms` | Prompt encoding in the request handler |
| `prefill_ms` | Actor prefill wall time while the request is active |
| `decode_ms` | Actor decode wall time while the request is active |
| `actor_cycle_idle_ms` | Configured cooperative actor idle time while the request remains active |
| `sampling_ms` | Separately measurable final normalization, LM-head, filtering, and token selection |
| `readback_ms` | An existing device-to-host token read when the backend exposes that boundary |
| `response_delivery_ms` | Ready token to delivery-worker handoff |
| `handler_queue_ms` | Delivery-worker handoff to handler receipt |
| `client_delivery_ms` | Handler receipt to response-body enqueue |
| `gpu_lock_wait_ms` | Time waiting for shared inference GPU ownership |
| `graph_capture_ms` | Backend graph-capture work attributable to this request |
| `graph_replay_ms` | Backend graph-replay work attributable to this request |
| `synchronization_ms` | Backend work settlement at an external-yield boundary |
| `resize_ms` | KV-resize work that delayed request admission or overlapped GPU-lock waiting |
| `trim_ms` | Allocator trim that overlapped GPU-lock waiting |
| `adapter_ms` | Adapter mutation that delayed request admission |
| `training_ms` | Training GPU ownership that overlapped inference-lock waiting |
| `unexplained_ms` | Inter-token wall time not covered by a measured candidate |

Preserve the distinction between these values:

- `null` means the path did not observe or could not isolate that phase;
- `0` means the path measured the phase at zero at the reporting resolution;
- a positive value is request-impact wall time, not necessarily exclusive
  device work.

Phase values can overlap. `decode_ms` can contain narrower backend phases, and
delivery can overlap later actor work. Never sum every phase and compare the
result with request duration.

A shared batched invocation can be attributed to every ready request that
participated in it. Request phase totals therefore explain causal overlap; they
are not process-wide work totals. Use backend lifetime telemetry when you need
device-work totals.

Stable serving normally leaves resize, trim, adapter, and training attribution
null because those operations are prohibited. A permitted operation also
remains null when it did not overlap the request's relevant boundary.

## Understand stall reasons

For each inter-token gap, Kiln chooses the largest measured phase candidate.
The candidate must explain at least the smaller of 250 ms or half of the gap;
otherwise the reason is `unexplained`.

The closed reason set is:

```text
actor_queue, actor_admission, actor_prefill, actor_decode,
actor_cycle_idle, response_delivery, handler_queue, client_delivery,
sampling, readback, gpu_lock_wait, graph_capture, graph_replay,
synchronization, resize, trim, adapter, training, unexplained
```

`unexplained` means the current instrumentation did not cover enough of the
gap. It is not evidence by itself of allocator activity, VRAM rebalancing, a
driver pause, or a particular backend defect.

## Read per-token SSE events

With explicit request opt-in, a real-model stream emits a Kiln-specific object
after the corresponding content enters the response-body channel:

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

`token_index` is zero-based. `token_id` is the accepted model token represented
by the row, including a special token that decodes to no visible text. The
event does not include log probabilities or alternative candidates.

`blocking_phase` and `blocking_phase_ms` are null for the first token because
there is no preceding gap. They can also be null when no gap observation
exists.

## Use the rolling decode endpoint

`GET /v1/stats/decode` summarizes request-local gaps retained during the last
60 seconds:

- `sample_count`: gaps in the active rolling window;
- `tok_per_sec`: gap count divided by the sum of those gap durations;
- `p50_itl_ms`, `p99_itl_ms`, `p999_itl_ms`, `mean_itl_ms`, and
  `max_itl_ms`;
- `stall_threshold_ms`: `max(250 ms, 5 × p50 ITL)`; and
- `stall_count`, `unexplained_stall_count`, and `stall_reasons`.

This `tok_per_sec` value is a decode-cadence diagnostic. It excludes TTFT,
prompt prefill before the first token, queueing outside the sampled gaps, and
request-window concurrency. Do not compare it with end-to-end output
throughput from a serving benchmark. When the rolling window is empty, counts
and numeric summaries are zero.

## Query Prometheus

`GET /metrics` exports the request timing families:

| Metric | Type | Population |
|---|---|---|
| `kiln_request_ttft_seconds` | histogram | Request receipt to first ready token |
| `kiln_token_itl_seconds` | histogram | Every retained request-local token gap |
| `kiln_request_latency_phase_seconds{phase}` | histogram | Non-null request phase observations |
| `kiln_token_stalls_total{reason}` | counter | Request-local gaps at or above the fixed 250 ms floor |

The histogram buckets are 5 ms, 10 ms, 25 ms, 50 ms, 100 ms, 250 ms, 500 ms,
1 s, 2.5 s, 5 s, 10 s, 30 s, 60 s, and `+Inf`. Labels use closed phase and
reason sets. Request IDs, prompts, model names, adapter names, client
identities, and error text are never labels.

Prometheus uses a fixed 250 ms stall definition so the counter remains
comparable as workload mix changes. Request summaries and the rolling endpoint
use the adaptive threshold. Account for that difference before comparing
their counts.

```promql
histogram_quantile(
  0.999,
  sum by (le) (rate(kiln_token_itl_seconds_bucket[5m]))
)

sum by (reason) (rate(kiln_token_stalls_total[5m]))

histogram_quantile(
  0.99,
  sum by (phase, le) (
    rate(kiln_request_latency_phase_seconds_bucket[5m])
  )
)
```

## Corroborate backend attribution

Request phases tell you which work overlapped a request. Health, trusted debug
state, and backend metric families tell you what the process and accelerator
were doing.

- Compare `gpu_lock_wait_ms` with typed resize, trim, training, and adapter
  overlap. The narrower values sit inside the broader lock wait; do not add
  them together.
- Treat `readback_ms` as an owned host-read boundary. Because accelerator work
  can be asynchronous, it can include waiting for earlier work on the same
  stream; it is not pure DMA time.
- Use CUDA and ROCm synchronization counters to identify typed host waits.
  Never infer request time by subtracting unlocked process-global snapshots.
- Use graph lifetime telemetry for capture, replay, cache, and retained-memory
  totals. Request graph phases explain overlap, not total device work.
- On Vulkan, inspect the effective capability gauges. Cross-request prefix
  reuse is currently correctness-quarantined on Vulkan, so exact repeats pay
  fresh prefill. Resident prefill is disabled by `stable` and admitted under
  `experimental` only when backend and request checks pass.

Hardware receipts and host names are evidence about specific benchmark runs.
They do not alter these metric definitions, enable a route, or become
machine-specific dispatch policy.

## Diagnose a pause

1. Classify the symptom as slow TTFT, slow ITL, or slow end-to-end completion.
   Do not substitute one metric for another.
2. Open the affected request in **Latency diagnosis** or capture its terminal
   metadata. Confirm the pause is request-local.
3. Inspect the dominant stall reason and its duration. Preserve null, zero,
   and positive phase values.
4. Corroborate that reason with the matching actor, synchronization, graph,
   allocator, mutation, or training telemetry over the same interval.
5. Treat `unexplained` as an instrumentation gap. Gather more evidence before
   assigning a backend cause.
6. Reproduce with a source-bound workload. Change one configuration field per
   comparison arm, compare like-for-like metrics, and retain failed evidence.

The machine-readable field authorities are the
[inference schema](../contracts/kiln-inference-v1.schema.json) and
[observability schema](../contracts/kiln-observability-v1.schema.json).
