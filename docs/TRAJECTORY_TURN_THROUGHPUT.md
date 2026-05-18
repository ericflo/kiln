# Trajectory Turn Throughput

This benchmark drives Kiln against trajectory-trainer turns materialized by
`/data/apps/trajectory-trainer/scripts/materialize_turn.py`. It is intended for
long-context adapter inference sweeps, especially adapters stored under:

```text
b2://clouderic/trajectory-trainer/turn-analysis/adapters/
```

Run it on a RunPod Kiln pod, not on a shared local worker.

## Server Setup

Start Kiln with the batching engine and prefix cache enabled. They are on by
default, but the explicit environment makes benchmark artifacts easier to read:

```bash
export KILN_BATCHING_ENGINE=1
export KILN_BATCH_PREFIX_AWARE_ADMISSION=1
export KILN_PREFIX_CACHE_ENABLED=1
export KILN_MAX_DECODE_BATCH=16
export KILN_ADAPTER_DIR=/workspace/kiln-adapters
export KILN_W4A16=1
export KILN_CUDA_GRAPHS=true

./target/release/kiln serve --host 0.0.0.0 --port 8420
```

`KILN_BATCH_PREFIX_AWARE_ADMISSION=1` makes the batching actor hold a queued
request when an active same-adapter request is a strict token prefix that can be
reused by the real prefix cache. This avoids duplicate long-prefill work for
consecutive turns from the same trajectory while still admitting unrelated rows
that can fill the decode batch.

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

For an A/B run, execute the same command once with
`KILN_BATCH_PREFIX_AWARE_ADMISSION=0` and once with it enabled, using the same
turn selection, adapter, `max_tokens`, and batch settings.
