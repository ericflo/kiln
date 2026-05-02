# Round-10 workers=2 throughput investigation

Date: 2026-05-02

## Summary

Round-10 did not expose a kiln workers=2 scheduling regression. It exposed a workload/acceptance-rate throughput limit: the server kept two production-shaped requests in flight almost continuously, but each finalized trajectory required many long requests.

The actionable kiln gap was observability. Before this investigation, `/metrics` exposed only whole-request latency and aggregate counters, so the next throughput regression could not be split into prefill, decode, queueing, and concurrency from Prometheus alone. This audit adds the missing server-side metrics.

## Evidence

Round-10 artifacts:

- `b2://clouderic/corrections-experiment/phase2-round10/REPORT.md`
- `b2://clouderic/corrections-experiment/phase2-round10/preflight-stats.json`
- `b2://clouderic/corrections-experiment/phase2-round10/logs/round10-kiln-server.log`
- `b2://clouderic/corrections-experiment/phase2-round10/logs/round10-metrics.prom`

From `REPORT.md` and `preflight-stats.json`:

- Wall clock: 10,087.38 s
- Finalized trajectories: 20
- Inference requests: 277 OK, 0 error, 0 timeout
- Attempt latency: p50 54.86 s, p90 154.66 s, p99 377.87 s
- Per-trajectory throughput: 8.41 min/trajectory

Parsing `round10-kiln-server.log` response spans gives:

- Response count including smoke: 284
- Total request service time: 20,194.24 s
- Mean request latency: 71.11 s
- Median request latency: 53.04 s
- Peak overlapping requests: 3 including smoke/metrics timing overlap
- Mean overlapping requests during collection: 1.96
- Requests >=120 s: 44
- Requests >=240 s: 5

The observed throughput is explained by Little's Law:

```text
13.85 requests/trajectory × 71.11 s/request ÷ 1.96 concurrent requests = 502 s/trajectory
502 s/trajectory = 8.36 min/trajectory
```

That matches the report's 8.41 min/trajectory. In other words, the server was already keeping roughly two long requests active; the trajectory-level slowdown came from request count and request length, not idle workers.

## Round-5 comparison caveat

The quoted round-5 `~5.76 min/trajectory` is not an apples-to-apples kiln throughput baseline:

- Round-5 ran workers=1 only after workers=2 had already hit the pre-#672/#694 reliability cascade.
- Round-5 report records `max_prompt_chars: 32000 (default)` while round-10 explicitly used unbounded production prompts.
- Round-5 did not preserve the same per-request latency histogram and prefill/decode split needed for direct attribution.
- Round-10 finalized only 20 trajectories but issued 277 successful requests, i.e. 13.85 requests per trajectory.

## Hypothesis results

1. Streaming-prefill threshold: not supported by round-10 logs. Production-shaped requests are long, but PR #694-class streaming preserved reliability and request latencies are decode-heavy enough that reverting or raising the threshold would risk OOM/timeout regressions without explaining trajectory count.
2. Prefix-cache hit-rate cap: not supported as a throughput root cause. Round-10 prefix cache remained comfortably below the #697 cap: peak 5/20 cached entries and 263 MB/1.05 GB state bytes.
3. workers=2 scheduler underlap: not supported. Server logs show mean overlap 1.96 across the run.
4. #701 KV headroom: not supported. `kiln_blocks_used` final was 9,441/45,914 and no OOM/eviction signal appeared.
5. metrics overhead: not supported. Round-10 metrics collection was sparse and the hot path used atomic counters.
6. workload mix / retries: supported. The trajectory-level throughput is explained by 13.85 long successful requests per trajectory.

## Added observability

This change adds:

- `kiln_request_duration_seconds` as a histogram while preserving count/sum names.
- `kiln_request_prefill_duration_seconds` histogram for prefix-cache production requests.
- `kiln_request_decode_duration_seconds` histogram for prefix-cache production requests.
- `kiln_active_requests_peak` gauge for peak in-flight concurrency.

These metrics make future workers=2 regressions immediately attributable from `/metrics`: if workers are idle, `kiln_active_requests_peak` and the live gauge show it; if long-prefill regresses, prefill buckets move; if decode dominates, decode buckets move.

## Validation receipts

On an A6000 RunPod pod (`ghcr.io/ericflo/kiln-runpod:latest`):

```bash
cargo test -p kiln-server metrics::tests:: -- --nocapture
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln
```

Both passed.

A live A6000 request run against a 53,307-token prompt with `max_tokens=256` produced populated split metrics:

```json
{
  "request_sum": 213.538474,
  "prefill_sum": 78.333799,
  "prefill_count": 4,
  "decode_sum": 134.731954,
  "decode_count": 4,
  "active_peak": 1,
  "tokens": 1024
}
```

## Recommendation

Do not ship a speculative throughput code change for round-10. Run round-12 with this observability build first. If trajectory throughput is still low, compare:

- `kiln_active_requests` / `kiln_active_requests_peak`
- `kiln_request_prefill_duration_seconds_*`
- `kiln_request_decode_duration_seconds_*`
- prefix-cache hit tokens / entries / state bytes
- requests per finalized trajectory

The likely next non-kiln lever is prompt/retry shaping in corrections-experiment, not kiln GPU scheduling.
