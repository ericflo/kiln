# Benchmarks

Kiln publishes comparable measurements, the exact command behind them, and
the limits that keep each number honest. Start here for the answer; the
regression analysis and raw receipts follow below.

## Current measured position

**On the tracked short diagnostic, Vulkan decode is back in its historical
range. Prefill is better than the regressed build, but still behind the May
baseline.**

> **Latest verified source result:** 13.46 decode tok/s, 74.29 ms mean
> inter-token latency, and 2.805 s prefill time at
> [`f3ae29e4a`](https://github.com/ericflo/kiln/commit/f3ae29e4a). This
> correction is verified locally; it is not yet a published release.

| Build | Prefill time | Mean ITL | Decode rate |
|---|---:|---:|---:|
| Capability correction, `f3ae29e4a` | 2,805 ms | 74.29 ms | **13.46 tok/s** |
| [Kiln v0.5.1](https://github.com/ericflo/kiln/releases/tag/kiln-v0.5.1) | 2,588 ms | 74.48 ms | **13.43 tok/s** |
| Regression, [`48fb3f7b`](https://github.com/ericflo/kiln/commit/48fb3f7bd) | 11,343 ms | 7,059.9 ms | **0.142 tok/s** |
| May 9 checkpoint A113 | 996 ms | 81.1 ms | **about 12.3 tok/s** |

As of July 30, 2026, v0.5.1 is the latest published release. It restored decode
speed, but its global route table is superseded on `main` by the
capability-derived correction above.

These rows use the same Qwen3.5-4B short serial workload on the same AMD Radeon
8060S test system. They answer one narrow question: did single-stream Vulkan
decode regress? **Yes—by roughly 87×—and it has recovered.** They do not
establish performance for every Vulkan device, long prompts, or concurrent
serving.

The latest correction changes *how routes are selected*, not which machine gets
a special path. Kiln derives Vulkan route legality from reported workgroup,
shared-memory, descriptor, push-constant, subgroup, API, and memory-topology
capabilities. Device name, vendor ID, device ID, PCI identity, and driver name
are never policy inputs.

## What happened

| Date | Change | Result |
|---|---|---|
| May 9 | Qualified Vulkan optimization work | About 12.3 decode tok/s |
| July 27 | [`28d8c6028`](https://github.com/ericflo/kiln/commit/28d8c6028d1b187107688c9adea0b079d203f687) replaced native routes with a broad portable fallback | 0.142 decode tok/s |
| July 30 | v0.5.1 restored standards-based compute routes | 13.43 decode tok/s |
| July 30 | Policy v6 replaced the global table with per-device capability derivation | 13.46 decode tok/s |

The regression happened because the July 27 fallback disabled resident decode,
packed-weight kernels, fused projections, GPU gather, and most fused submission
routes together. Removing machine-derived policy was correct; disabling generic
Vulkan fast paths was not.

v0.5.1 restored performance, but its one global “native default” still made an
unsupported assumption: that the complete fast route set was legal everywhere.
Policy v6 corrects that architecture. The common shader set targets Vulkan 1.0.
Three subgroup-tiled attention shaders target Vulkan 1.1 and are selected only
when the device reports the required compute-stage subgroup operations;
otherwise Kiln keeps the Vulkan 1.0 untiled route.

## Reproduce the latest result

```bash
./target/release/kiln-bench \
  --model-path Qwen3.5-4B \
  --paged \
  --latency-only \
  --latency-warmup-runs 1 \
  --prompt-tokens 64 \
  --max-output-tokens 8 \
  --seed 117 \
  --quiet
```

Measured environment:

| Input | Value |
|---|---|
| Model | Qwen3.5-4B |
| Backend | Vulkan |
| GPU | AMD Radeon 8060S Graphics |
| Driver | RADV Strix Halo, Mesa 26.1.5 |
| Realized workload | 56 prompt tokens, 9 generated tokens |
| Source | [`f3ae29e4a`](https://github.com/ericflo/kiln/commit/f3ae29e4a) |

This is intentionally a short latency diagnostic. A release claim needs wider
correctness, workload, device, and soak coverage.

## Read the metrics correctly

| Metric | Definition | Use it to answer |
|---|---|---|
| Decode rate | `1000 / mean inter-token latency in ms` for a defined request path | How quickly are tokens emitted after the first one? |
| TTFT | Request arrival to first token | How long does the user wait before output begins? |
| Prefill time | Time spent processing the prompt before decode | How long did prompt processing take for this request? |
| Prefill throughput | Prompt tokens divided by prefill time | How quickly is the prompt processed? |
| Request-window throughput | All output tokens divided by the full measured request window | What did this exact concurrent workload deliver end to end? |
| SLO goodput | Output from requests that met declared latency and correctness gates | How much useful work met the service objective? |

Numbers are comparable only when hardware, model, source, driver, prompt
distribution, output length, sampling, concurrency, and metric definition
match.

## Why the old 0.455 number did not describe decode speed

The previous page placed **0.455 aggregate output tok/s** beside decode rates
without making the different denominators obvious.

That July 20 receipt was a 30-minute mixed-load soak:

- 55 requests and 880 output tokens;
- prompt sizes from 178 to 418 tokens;
- concurrency waves of one and four;
- the full 1,935-second request window in the denominator;
- p50 time to first token of 83.2 seconds;
- p50 inter-token latency of 76.7 ms, equivalent to about 13.0 decode tok/s at
  the median.

The `0.455` value described aggregate output divided by the entire request
window, dominated by prefill and queueing. It did **not** mean the decode loop
ran at 0.455 tok/s, and it predates the July 27 regression.

[Inspect the July 20 Vulkan soak receipt](https://github.com/ericflo/kiln/blob/main/qualification/receipts/vulkan/strix-halo/20260720t105341024462z-vulkan-strix-halo-serving-vulkan-developme-b5eb848d54-v1.json).

## Product boundary

Kiln does not claim high-concurrency parity with vLLM. Choose Kiln for the
integrated local serve, train, evaluate, and adapter-promotion loop. Choose a
serving-focused engine when maximum multi-user throughput is the only
objective.

That boundary does not excuse a backend regression. Vulkan should use fast,
standards-based compute routes on every device that exposes the capabilities
each route requires, with narrow fallbacks for capabilities it lacks.

## Evidence and protocol

- [Raw benchmark evidence ledger](https://github.com/ericflo/kiln/blob/main/BENCHMARKS.md)
- [May 9 Vulkan decode qualification log](https://github.com/ericflo/kiln/blob/main/docs/audits/vulkan-strix-halo-2026-05-09-gpu-decode-shortlog.md)
- [Serving benchmark protocol](https://github.com/ericflo/kiln/blob/main/docs/serving/SERVING_BENCHMARK_PROTOCOL.md)
- [Qualification receipts](https://github.com/ericflo/kiln/tree/main/qualification/receipts)
