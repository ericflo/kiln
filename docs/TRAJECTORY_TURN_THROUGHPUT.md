# Trajectory Turn Throughput

This benchmark drives Kiln against trajectory-trainer turns materialized by
`/data/apps/trajectory-trainer/scripts/materialize_turn.py`. It is intended for
long-context adapter inference sweeps, especially adapters stored under:

```text
b2://clouderic/trajectory-trainer/turn-analysis/adapters/
```

Run it on a RunPod Kiln pod, not on a shared local worker.

## Server Setup

Start Kiln with a typed, receipt-bound batching and prefix-cache policy. The
explicit values make benchmark artifacts readable without relying on ambient
process environment:

```toml
# trajectory-benchmark.toml
[server]
max_decode_batch = 16

[batching]
mode = "enabled"
rowwise_decode = false
prefix_aware_admission = true
prefill_admission_quantum = "auto"
direct_decode_rendezvous_mode = "auto"
direct_decode_rendezvous_max_batch = "auto"
direct_decode_rendezvous_wait_us = "auto"
direct_decode_rendezvous_mixed_seq_lens = "auto"

[prefix_cache]
enabled = true

[model]
path = "/workspace/Qwen3.5-4B"
adapter_dir = "/workspace/kiln-adapters"

[memory]
cuda_graphs = true
```

```bash
./target/release/kiln serve --config trajectory-benchmark.toml
curl -fsS http://127.0.0.1:8420/v1/config \
  | jq '{batching, decode_runtime}' \
  > /workspace/bench/trajectory-batching-config.json
```

`batching.prefix_aware_admission = true` makes the batching actor hold a queued
request when an active same-adapter request is a strict token prefix that can be
reused by the real prefix cache. This avoids duplicate long-prefill work for
consecutive turns from the same trajectory while still admitting unrelated rows
that can fill the decode batch.

Verify `batching.actor_active=true`, effective mode enabled, rowwise decode
false, and prefix-aware admission true in the captured JSON. The automatic
admission quantum is backend-owned: it is the effective decode width on CUDA
and Vulkan, and 4 on ROCm, Metal, and CPU, always clamped to that effective
width. The response records the exact backend, configured, effective, clamp,
and source values so two runs cannot silently compare different policies.
The captured direct-rendezvous worker may be active, but this actor-backed
trajectory run requires its sibling `route_available=false`. That fallback is
only for actor-absent direct streaming effectively-greedy work; it is not part
of this benchmark path.

## Benchmark

The benchmark script can sync an adapter directory from B2, or download and
unpack a `.tar.gz` adapter archive from B2, load it once through Kiln, then
materialize turns as `prompt-chosen-jsonl` and issue either
`/v1/completions/batch`, concurrent `/v1/chat/completions`, or both:

```bash
python3 scripts/bench-trajectory-turns.py \
  --host http://127.0.0.1:8420 \
  --db /workspace/trajectory-trainer/trajectories.db \
  --adapter-uri b2://clouderic/trajectory-trainer/turn-analysis/adapters/<adapter-name>.tar.gz \
  --adapter-dir /workspace/kiln-adapters \
  --mode batch \
  --order session \
  --batch-size 16 \
  --http-workers 1 \
  --max-tokens 64 \
  --out /workspace/bench/trajectory-turn-throughput.json
```

Use `--limit 256` for smoke runs, then remove the limit for the full sweep.
`--order session` keeps consecutive turns together so the prefix-aware admission
path has the best chance to reuse long shared trajectory prefixes.

The JSON report includes aggregate completion tok/s plus metric deltas from
`/metrics`, including:

- `prefix_hit_tokens_delta`
- `batching_prefill_tokens_delta`
- `batching_decode_tokens_delta`
- `batching_prefix_deferrals_delta`
- `batching_errors_delta`

If `batching_prefix_deferrals_delta` is zero, the selected materialized turns
did not produce queued prompts whose complete rendered prompt was a strict token
prefix of a later queued prompt. In that case this admission policy is correctly
idle, and the run is still useful as a long-context batching smoke.

For an A/B run, create a second TOML file whose only change is
`batching.prefix_aware_admission = false`, restart Kiln, and execute the same
command using the same turn selection, adapter, `max_tokens`, and batch
settings. Capture `/v1/config.batching` after each restart and keep it with the
corresponding report. Do not toggle the deprecated
`KILN_BATCH_PREFIX_AWARE_ADMISSION` alias between live requests; all eight
batching values are immutable for the process lifetime.
