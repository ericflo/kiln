# Marlin W4A16 MLP Wire-in Benchmark (PR #152 follow-up)

Benchmarks the Marlin W4A16 MLP wire-in from
[PR #152](https://github.com/ericflo/kiln/pull/152). Marlin is gated via
the `KILN_W4A16` env flag (`0` = baseline BF16 matmul, `1` = Marlin W4A16
GEMM). This run expands Marlin coverage from `q_proj` only (PR #149, 8/32
layers, 1/7 projections — see [MARLIN_QPROJ_BENCH.md](MARLIN_QPROJ_BENCH.md))
to `gate_proj + up_proj + down_proj` on all 32 layers
(PR #152, 32/32 layers, 3/7 projections).

Per PROFILING.md, MLP projections are the largest single share of decode
GPU time, so this is the first wire-in where a steady-state decode speedup
should be visible.

## Headline (warm-run steady state, runs 2–3 averaged)

| Metric                         | Baseline (`KILN_W4A16=0`) | Marlin MLP (`KILN_W4A16=1`) | Δ         |
|--------------------------------|--------------------------:|----------------------------:|----------:|
| Decode throughput (tok/s)      | 43.30                     | 47.58                       | **+9.9%** |
| Mean ITL (ms)                  | 23.10                     | 21.02                       | **−9.0%** |
| P50 ITL (ms)                   | 22.89                     | 20.79                       | **−9.2%** |
| P99 ITL (ms)                   | 27.46                     | 25.47                       | −7.2%     |
| Throughput bs=1/512→256 (tok/s)| 39.90                     | 43.82                       | **+9.8%** |
| Throughput bs=4 (tok/s)        | 39.97                     | 44.07                       | **+10.3%**|
| Throughput bs=8 (tok/s)        | 39.70                     | 43.47                       | **+9.5%** |
| Throughput bs=16 (tok/s)       | 39.78                     | 43.48                       | **+9.3%** |
| Prefill, warm (ms)             | 405.4                     | 473.1                       | +16.7% (see §Prefill) |
| Prefill, warm (tok/s)          | 1248.2                    | 1082.4                      | −13.3%    |

## VRAM

| Metric                    | Baseline    | Marlin MLP  | Δ          |
|---------------------------|------------:|------------:|-----------:|
| Model VRAM after load     | 16,343 MB   | 17,655 MB   | **+1,312 MB** (+8.0%) |
| Peak runtime VRAM (bs=16) | 16,840 MB   | 18,056 MB   | +1,216 MB (+7.2%) |
| Model load time (mean)    | 8.81 s      | 67.48 s     | **+58.67 s** (Marlin weight packing) |

## Verdict: **real +9–10% decode speedup**, first non-null result for Marlin

PR #149's q_proj wire-in landed a null decode result (+0.12%, below the
±5% floor) because q_proj is only 1 of 7 projections on 8 of 32 layers —
kernel cost was hidden by the rest of the step. PR #152 wires the three
MLP projections (12× the call sites — 96 projection calls per layer-pass
vs q_proj's 8), and the headline decode number moves in line with what
PROFILING.md projected:

- **Decode**: mean ITL 23.10 ms → 21.02 ms (−9.0%); P50 22.89 → 20.79
  (−9.2%); P99 27.46 → 25.47 (−7.2%). Consistent across 2 warm runs
  (mean ITL 20.93 ms and 21.10 ms on Marlin, vs 23.06 ms and 23.14 ms
  on baseline). Clear steady-state win.
- **Throughput** (batched sequential): +9.3% to +10.3% across bs=1/4/8/16,
  tracking the ITL improvement. No batch-size-dependent shape to the
  speedup — consistent across the sweep.
- **Memory cost**: +1.3 GB model VRAM and +1.2 GB peak — the packed
  Marlin weights sit alongside the original BF16 MLP weights rather
  than replacing them in this wire-in, so the base weight pages still
  occupy space. A future cleanup can drop the BF16 MLP weights when
  `KILN_W4A16=1` and recover that delta.
- **Load-time cost**: +58.7 s. Marlin packing runs once at load for each
  of the 3 × 32 = 96 MLP projection tensors, and the current pack path
  is not parallelized. For a server that lives for hours this is a
  one-time hit; for CI / cold-start scenarios it's a meaningful
  regression that a future task should address (precompute packed
  artifacts to disk, or parallelize the pack loop).
- **Gotcha**: run 1 on *each* mode showed a clear cold-start penalty
  (see §Cold-run). For baseline that manifests as an 8,368 ms prefill
  (mirroring the "20× prefill illusion" flagged in
  MARLIN_QPROJ_BENCH.md); for Marlin it manifests as decode ITL 23.66 ms
  instead of the steady-state 21.0 ms (first-call CUDA graph capture on
  the new Marlin decode path). This is exactly why the task description
  called for prefill warmup. Warm runs 2–3 are the honest A/B.

### Prefill

Warm-run prefill is mixed and noisy:

- Baseline warm prefill: 402.9 ms, 407.9 ms → 405.4 ms mean, very tight
- Marlin warm prefill: 524.5 ms, 421.7 ms → 473.1 ms mean, high variance

The Marlin run-2 value (524.5 ms) is an outlier; run-3 (421.7 ms) is
within noise of baseline warm (405 ms). Prefill in the paged bench
does only one forward pass per process — there's no in-process warmup
before the measured prefill, so even the "warm" runs are single-sample
and noisy at the ~10% level. Call prefill parity within noise; do not
credit or charge Marlin with a prefill delta on this evidence.

### Cold-run (reference, do NOT use for A/B)

|                        | Baseline run 1 | Marlin run 1 |
|------------------------|---------------:|-------------:|
| Model load             | 9.38 s         | 74.81 s      |
| Prefill (cold)         | 8,368.6 ms     | 423.0 ms     |
| Decode mean ITL (cold) | 22.94 ms       | 23.66 ms     |
| Decode tok/s (cold)    | 43.59          | 42.27        |

Baseline's cold-start prefill blow-up (8.4 s) dwarfs Marlin's (0.4 s)
because the two modes exercise different first-call init paths.
Marlin run 1 decode ITL regresses to baseline-parity (23.66 vs 23.10)
because the Marlin decode graph is captured on the first decode step.
Both anomalies disappear by run 2, so the warm-run table above is the
honest A/B.

## Next steps

1. **Cache packed Marlin weights on disk** (or parallelize the pack loop)
   to reclaim the +58 s cold-start cost. Current pack implementation
   runs serially across 96 projection tensors at load.
2. **Drop the BF16 MLP weights from VRAM when `KILN_W4A16=1`** to recover
   the +1.3 GB overhead. The Marlin forward path is already gated on
   `*_marlin.is_some()`, so the BF16 tensors are unused in that mode.
3. **Wire Marlin into the remaining GQA projections (`k_proj` / `v_proj`
   / `o_proj`) and the GDN `in_proj_*` matmuls** to extend the decode
   win. q_proj+k_proj+v_proj+o_proj on the 8 full-attn layers is
   secondary (per PROFILING.md) but the 24 GDN layers' in_proj_ is a
   larger secondary lever.
4. **Add in-process prefill warmup to `bench_latency_paged`** so a
   single-run prefill number is steady-state, not cold. Right now the
   only honest way to read prefill from this bench is to run the
   binary ≥2 times and discard the first.

## Hardware / build

- GPU: NVIDIA RTX A6000 (48 GB), driver 570.195.03
- CUDA toolkit: 12.4.1
- Image: `ghcr.io/ericflo/kiln-runpod:latest`
- rustc 1.95.0, cargo 1.95.0, sccache 0.9.1
- Build: `cargo build --release --features cuda --bin kiln-bench`
- Kiln commit: `e99d5ae` (PR #152 merge,
  `CE: Wire Marlin W4A16 into MLP projections (gate/up/down_proj)`)
- Marlin coverage: q_proj (from PR #149, 8/32 layers) +
  gate_proj / up_proj / down_proj (from PR #152, 32/32 layers each).
  Total packed projections: 8 q_proj + 96 MLP = 104 of 224 projection
  call sites per layer-pass (7 projections × 32 layers = 224).
- Bench command (ran 3× per mode back-to-back to isolate cold-start):
  ```
  for i in 1 2 3; do
    KILN_W4A16={0,1} ./target/release/kiln-bench \
        --model-path /workspace/Qwen3.5-4B \
        --prompt-tokens 512 \
        --max-output-tokens 256 \
        --skip-training \
        --paged
  done
  ```
- Pod: single A6000 on-demand (RunPod), ~40 minutes wall clock including
  build + model download + 6 bench iterations.
- Per-run JSON output captured in the logs; this table cites the JSON
  (`mean_inter_token_ms`, `p50_inter_token_ms`, `p99_inter_token_ms`,
  `decode_tokens_per_sec`, `prefill_ms`, `model_vram_mb`,
  `inference[].peak_vram_mb`, `inference[].tokens_per_sec`) not the
  stderr summary.

## Appendix — Warm run JSON (runs 2 & 3)

### Baseline (`KILN_W4A16=0`)

Run 2:
```
prefill_ms: 402.9          prefill_tok/s: 1255.86
decode_tokens_per_sec: 43.37
mean_ITL: 23.06 ms   p50: 22.97 ms   p99: 27.56 ms
throughput bs=1: 39.87 tok/s   bs=4: 40.05   bs=8: 39.62   bs=16: 39.62
model_vram: 16343 MB   peak_vram: 16840 MB
```
Run 3:
```
prefill_ms: 407.9          prefill_tok/s: 1240.46
decode_tokens_per_sec: 43.22
mean_ITL: 23.14 ms   p50: 22.81 ms   p99: 27.36 ms
throughput bs=1: 39.94   bs=4: 39.90   bs=8: 39.79   bs=16: 39.94
model_vram: 16343 MB   peak_vram: 16840 MB
```

### Marlin (`KILN_W4A16=1`)

Run 2:
```
prefill_ms: 524.5          prefill_tok/s: 964.67
decode_tokens_per_sec: 47.77
mean_ITL: 20.93 ms   p50: 20.60 ms   p99: 25.08 ms
throughput bs=1: 44.79   bs=4: 44.18   bs=8: 43.84   bs=16: 43.49
model_vram: 17655 MB   peak_vram: 18056 MB
```
Run 3:
```
prefill_ms: 421.7          prefill_tok/s: 1200.04
decode_tokens_per_sec: 47.40
mean_ITL: 21.10 ms   p50: 20.97 ms   p99: 25.85 ms
throughput bs=1: 42.86   bs=4: 43.97   bs=8: 43.10   bs=16: 43.46
model_vram: 17655 MB   peak_vram: 18056 MB
```
