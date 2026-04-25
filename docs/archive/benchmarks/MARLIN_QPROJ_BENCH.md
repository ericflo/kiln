# Marlin W4A16 `q_proj` Benchmark (PR #149 follow-up)

Benchmarks the Marlin W4A16 q_proj wire-in from
[PR #149](https://github.com/ericflo/kiln/pull/149). Marlin is gated
via the `KILN_W4A16` env flag (`0` = baseline f16 matmul, `1` = Marlin
W4A16 GEMM).

## Headline

| Metric                       | Baseline (`KILN_W4A16=0`) | Marlin (`KILN_W4A16=1`) | Δ        |
|------------------------------|--------------------------:|------------------------:|---------:|
| Decode throughput (tok/s)    | 44.066                    | 44.118                  | **+0.12%** |
| Mean ITL (ms)                | 22.693                    | 22.667                  | -0.12%   |
| P50 ITL (ms)                 | 22.608                    | 22.536                  | -0.32%   |
| P99 ITL (ms)                 | 23.895                    | 25.612                  | +7.19%   |
| Throughput bs=16/4096 (tok/s)| 40.124                    | 40.295                  | +0.42%   |
| Throughput bs=8/2048 (tok/s) | 40.320                    | 40.446                  | +0.31%   |
| Prefill (tok/s) — **cold**   | 60.593                    | 1256.297                | *see note* |
| TTFT (ms) — **cold**         | 8350.8                    | 402.8                   | *see note* |

## VRAM

| Metric                    | Baseline    | Marlin      | Δ          |
|---------------------------|------------:|------------:|-----------:|
| Model VRAM after load     | 16343 MB    | 16471 MB    | +128 MB    |
| Peak runtime VRAM (bs=16) | 16840 MB    | 16936 MB    | +96 MB     |
| Model load time           | 9.34 s      | 11.85 s     | +2.51 s    |

## Verdict: **NULL** for decode, **prefill result flagged** for follow-up

- **Decode**: +0.12% throughput, -0.12% mean ITL. Below the ±5% floor
  — treat as parity. `q_proj` is a single projection inside full
  attention, which itself runs on only 8 of 32 layers (GQA), so the
  kernel cost is hidden by `kv_copy`, RoPE, RMSNorm, GDN layers, and
  the attention kernel itself. This matches the expectation: wiring
  Marlin into a single q_proj on only a quarter of the layers does
  not move end-to-end decode in a measurable way. Shipping Marlin
  still has value (smaller weights once we quantize more projections;
  paves the path for full W4A16 on q/k/v/o/gate/up/down).
- **Throughput** (batched sequential): parity across batch sizes
  (+0.3-0.4%), well within run-to-run noise.
- **P99 ITL**: +7.19% (23.9 ms → 25.6 ms). Single-sample measurement,
  likely noise but worth a re-run. Mean and P50 improved, so typical
  behavior is unaffected.
- **Prefill / TTFT**: the cold-start prefill shows an apparent 20.7x
  speedup (8350 ms → 402 ms). This is a **single-shot measurement
  taken as the very first GPU op after model load**, with no prefill
  warmup. Both runs use CUDA graphs for decode and paged attention
  for prefill. Plausible explanations for the delta: (1) CUDA graph
  instantiation / first-call kernel JIT is absorbed by the baseline
  run but not the Marlin run because the two paths exercise different
  kernels on first call; (2) lazy allocator / page-fault-in of a
  memory-mapped `q_proj` weight tensor on baseline that the Marlin
  path skips because packed Marlin weights are eagerly loaded (this
  also explains the +2.51 s model load for Marlin). **Do NOT claim a
  20x prefill win from this run.** Needs a follow-up bench with
  explicit prefill warmup (≥3 throwaway prefills before measurement)
  before the prefill number can be trusted.

### Recommended follow-up

1. Add a prefill-warmup phase to `kiln-bench --paged` so the latency
   metric doesn't absorb first-call init cost.
2. Re-run baseline vs Marlin with the warmup and report the steady-state
   TTFT delta.
3. Once q_proj is ratified, extend Marlin to k_proj/v_proj/o_proj and
   the FFN projections to target a meaningful decode and memory win.

## Hardware / build

- GPU: NVIDIA RTX A6000 (48 GB), driver 570.195.03
- CUDA toolkit: 12.4.1
- Image: `ghcr.io/ericflo/kiln-runpod:latest`
- rustc 1.95.0, cargo 1.95.0, sccache 0.9.1
- Build: `cargo build --release --features cuda --bin kiln-bench`
- Kiln commit: `820663c` (PR #149 merge, `CE: Wire Marlin W4A16 into q_proj forward path`)
- Bench command:
  ```
  KILN_W4A16={0,1} ./target/release/kiln-bench \
      --model-path /workspace/Qwen3.5-4B \
      --prompt-tokens 512 \
      --max-output-tokens 256 \
      --skip-training \
      --paged
  ```
- Pod: single A6000 on-demand (RunPod), ~25 minutes wall clock.

## Appendix A — Baseline log (`KILN_W4A16=0`)

```
=== kiln-runpod image ===
  GPU: NVIDIA RTX A6000, 570.195.03
  CUDA toolkit: release 12.4
  rustc: 1.95.0 | cargo: 1.95.0
  sccache: 0.9.1 | nextest: (65e806bd5
  torch: 2.4.1+cu124 (cuda=12.4)

=== Kiln Benchmark Suite ===

GPU: NVIDIA RTX A6000 (49140 MB)
Loading model from /workspace/Qwen3.5-4B...
Loaded 4206M parameters (8022 MB) across 32 layers
CUDA available — using GPU device 0
Model loaded in 9.34s (backend: cuda, VRAM: 16343 MB)

--- Latency Benchmark (PAGED — production path) ---
  Measuring latency [PAGED, block_size=16, blocks=48] (506 prompt tokens)...
    Prefill (paged): 8350.8ms (61 tok/s)
    Decode (paged): 257 tokens, mean ITL 22.7ms (44.1 tok/s)

--- Inference Throughput Benchmarks ---

1 sequential runs:
    Run 1/1: 256 tokens in 6312.2ms (40.6 tok/s)
  => 40.6 tok/s aggregate

4 sequential runs:
  => 40.3 tok/s aggregate

8 sequential runs:
  => 40.3 tok/s aggregate

16 sequential runs:
  => 40.2 tok/s aggregate

latency (JSON):
  prefill_time_ms:         8350.808712
  prefill_tokens_per_sec:  60.593
  time_to_first_token_ms:  8350.808712
  mean_inter_token_ms:     22.693
  p50_inter_token_ms:      22.608
  p99_inter_token_ms:      23.895
  num_tokens_generated:    257
  decode_tokens_per_sec:   44.066

throughput (JSON):
  bs=1,  out=256,   50.6 tok/s
  bs=4,  out=1024,  40.3 tok/s
  bs=8,  out=2048,  40.3 tok/s
  bs=16, out=4096,  40.1 tok/s
  peak_vram_mb:     16840
```

## Appendix B — Marlin log (`KILN_W4A16=1`)

```
=== kiln-runpod image ===
  GPU: NVIDIA RTX A6000, 570.195.03
  CUDA toolkit: release 12.4
  rustc: 1.95.0 | cargo: 1.95.0

=== Kiln Benchmark Suite ===

GPU: NVIDIA RTX A6000 (49140 MB)
Loading model from /workspace/Qwen3.5-4B...
Loaded 4206M parameters (8022 MB) across 32 layers
CUDA available — using GPU device 0
Model loaded in 11.85s (backend: cuda, VRAM: 16471 MB)

--- Latency Benchmark (PAGED — production path) ---
  Measuring latency [PAGED, block_size=16, blocks=48] (506 prompt tokens)...
    Prefill (paged): 402.8ms (1256 tok/s)
    Decode (paged): 257 tokens, mean ITL 22.7ms (44.1 tok/s)

--- Inference Throughput Benchmarks ---

1 sequential runs:
    Run 1/1: 256 tokens in 6294.7ms (40.7 tok/s)
  => 40.7 tok/s aggregate

4 sequential runs:
  => 40.4 tok/s aggregate

8 sequential runs:
  => 40.4 tok/s aggregate

16 sequential runs:
  => 40.3 tok/s aggregate

latency (JSON):
  prefill_time_ms:         402.771
  prefill_tokens_per_sec:  1256.297
  time_to_first_token_ms:  402.771
  mean_inter_token_ms:     22.667
  p50_inter_token_ms:      22.536
  p99_inter_token_ms:      25.612
  num_tokens_generated:    257
  decode_tokens_per_sec:   44.118

throughput (JSON):
  bs=1,  out=256,   40.7 tok/s
  bs=4,  out=1024,  40.4 tok/s
  bs=8,  out=2048,  40.4 tok/s
  bs=16, out=4096,  40.3 tok/s
  peak_vram_mb:     16936
```
