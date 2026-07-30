# Benchmarks

## Bottom line

Vulkan performance **did regress catastrophically** after the previously
qualified work, and the old public page made that hard to see.

On 2026-07-30, current `main` at commit `48fb3f7b` measured **0.142 decode
tokens/s** on an AMD Radeon 8060S using the same short serial workload that
measured about **12.3 decode tokens/s** in May. Prefill rose from about
**1.0 s** to **11.3 s**. That is roughly an **87× decode regression** and an
**11× prefill regression**.

The regression boundary is the 2026-07-27 change
[`28d8c6028`](https://github.com/ericflo/kiln/commit/28d8c6028d1b187107688c9adea0b079d203f687),
which replaced the native Vulkan policy with a fallback that disables resident
decode, packed-weight kernels, fused projections, GPU gather, and most
single-submit or chained routes. The change was intended to remove
machine-derived policy, but disabling the generic Vulkan fast paths was not a
portable performance strategy.

The 2026-07-30 v5 release candidate restores those standards-based routes
without a device-name, vendor-ID, PCI-ID, or machine allowlist. The clean
same-workload result is **13.43 decode tokens/s** at **74.48 ms mean
inter-token latency**: about **95× faster than regressed main** and back in the
historical range. Prefill is **2.588 s**, a **4.4× recovery** from main but
still about **2.6× slower** than the May checkpoint. The candidate is
parity-tested but is not labeled shipped until the release is published.

## The comparable Vulkan result

These two rows use the same model family, hardware family, benchmark command,
prompt target, output target, and seed. This is the comparison that answers
whether serial decode regressed.

| Source | Prefill | Mean inter-token latency | Decode rate | Status |
|---|---:|---:|---:|---|
| 2026-05-09 Vulkan optimization checkpoint A113 | 995.9 ms | 81.1 ms | about 12.3 tok/s | Historical qualified implementation |
| 2026-07-30 `main` at `48fb3f7b` | 11,343.1 ms | 7,059.9 ms | 0.142 tok/s | Reproduced regression |
| 2026-07-30 v5 release candidate | 2,587.6 ms | 74.48 ms | 13.43 tok/s | Clean local run; parity-tested, not yet shipped |

Hardware for the current run: AMD Radeon 8060S Graphics, RADV Strix Halo,
Mesa 26.1.5, Qwen3.5-4B, Vulkan backend. The current run generated nine tokens;
the benchmark reports 56 realized prompt tokens from the 64-token target.

Reproduction command:

```bash
KILN_BENCH_LOG_TOKENS=1 ./target/release/kiln-bench \
  --model-path Qwen3.5-4B \
  --paged \
  --latency-only \
  --latency-warmup-runs 1 \
  --prompt-tokens 64 \
  --max-output-tokens 8 \
  --seed 117 \
  --quiet
```

Historical evidence:
[Vulkan Strix Halo decode shortlog](https://github.com/ericflo/kiln/blob/main/docs/audits/vulkan-strix-halo-2026-05-09-gpu-decode-shortlog.md).

## What the correction actually selects

The release candidate does not recognize the benchmark machine. It selects
routes from:

- whether a Vulkan logical device was created;
- tensor shape, dtype, layout, and storage;
- workload size, including host/device crossover thresholds;
- queried dispatch-grid and shared-memory limits where a route needs them.

The multi-token Gated DeltaNet recurrent shader assigns independent
state columns to Vulkan invocations and loops over sequence positions inside
each invocation. Its result and final recurrent state pass a sequential CPU
oracle, including GQA head expansion and multiple batches. Small
host-resident convolution and elementwise prefill work stays on the reference
path when a Vulkan bridge would cost more than the arithmetic; its threaded
state returns to its entry storage so decode remains resident.

Every optimized route must continue to decline unsupported shapes or limits.
Completing and qualifying the per-route capability matrix across vendors
remains work; substituting a device identity for that matrix is not acceptable.

## Why the old 0.455 number was misleading

The previous page put **0.455 aggregate output tok/s** beside other
tokens-per-second results without making the denominators visually
incompatible.

That July 20 Vulkan receipt was a 30-minute mixed-load soak:

- 55 requests and 880 output tokens;
- prompt sizes from 178 to 418 tokens;
- concurrency waves of one and four;
- the whole 1,935-second request window in the denominator;
- p50 time to first token of 83.2 seconds;
- p50 inter-token latency of 76.7 ms, equivalent to about 13.0 decode tok/s at
  the median.

So `0.455` described **aggregate output tokens divided by the entire request
window**, dominated by terrible prefill and queueing. It did not mean the
single-stream decode loop itself ran at 0.455 tok/s. It also predates the July
27 policy regression and therefore says nothing about current `main`.

The receipt remains valid evidence for its exact old source and workload:
[July 20 Vulkan soak receipt](https://github.com/ericflo/kiln/blob/main/qualification/receipts/vulkan/strix-halo/20260720t105341024462z-vulkan-strix-halo-serving-vulkan-developme-b5eb848d54-v1.json).

## Read each metric correctly

| Metric | Definition | Answers |
|---|---|---|
| Decode rate | `1000 / inter-token latency in ms` for a defined request path | How quickly does the decode loop emit tokens after the first token? |
| TTFT | Request arrival to first token | How long does the user wait before output begins? |
| Prefill throughput | Prompt tokens divided by prefill time | How quickly is the prompt processed? |
| Request-window output throughput | All output tokens divided by wall time from the first dispatch to the last completion | What did this exact concurrent workload deliver end to end? |
| SLO goodput | Output from requests that met declared latency and correctness gates | How much useful work met the service objective? |

Numbers from different rows are comparable only when hardware, model, source,
driver, prompt distribution, output length, sampling, concurrency, and metric
definition match.

## Product position

Kiln currently makes no high-concurrency parity claim against vLLM. Choose Kiln
for the integrated local serve/train/eval/adapter loop. Choose a serving-focused
engine when maximum multi-user serving throughput is the only objective.

That product boundary does not excuse a Vulkan regression. Vulkan should use
fast, standards-based compute routes on every device that exposes the required
capabilities, with narrow fallbacks for capabilities it lacks.

## Full evidence and protocol

This page is the reader-facing interpretation. The detailed source ledger,
historical experiments, rejected candidates, and acceptance machinery remain
available for audit:

- [Raw benchmark evidence ledger](https://github.com/ericflo/kiln/blob/main/BENCHMARKS.md)
- [Serving benchmark protocol](../SERVING_BENCHMARK_PROTOCOL.md)
- [Qualification receipts](https://github.com/ericflo/kiln/tree/main/qualification/receipts)
