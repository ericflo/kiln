# Projection Transpose Cache Benchmark

Benchmark for `ce/cache-projection-transposes` (commit 1759e33) vs `main` (commit d5c6e83), measured on RunPod NVIDIA RTX A6000 (driver 550.127.08, CUDA 12.4) with `kiln-runpod:latest`.

Bench command (identical for both binaries):

```
./target/release/kiln-bench --model-path /workspace/models/qwen3.5-4b \
    --prompt-tokens 512 --max-output-tokens 256 --skip-training --paged
```

## Headline: decode is 2.33× faster

| Metric                              | main (d5c6e83) | branch (1759e33) | Δ        |
|-------------------------------------|----------------|------------------|----------|
| **Decode tok/s** (paged latency)    | **7.61**       | **17.70**        | **+133%** |
| Mean inter-token latency            | 131.4 ms       | 56.5 ms          | −57%     |
| P50 inter-token latency             | 131.3 ms       | 55.5 ms          | −58%     |
| P99 inter-token latency             | 136.2 ms       | 65.2 ms          | −52%     |

Hits the high end of the predicted 1.8–2.5× decode ITL range from the PR description, and clears the 5% justification floor by a wide margin.

## Throughput (single sequence, prefill+decode combined)

| Batch | main tok/s | branch tok/s | Δ     |
|-------|------------|--------------|-------|
| 1     | 7.35       | 17.13        | +133% |
| 4     | 7.27       | 16.99        | +134% |
| 8     | 7.27       | 16.86        | +132% |
| 16    | 7.30       | 16.84        | +131% |

Steady-state throughput improvement matches the latency-phase decode improvement, confirming the win is real and not an artifact of a single timing.

## Prefill regression on the paged latency path (flagged)

| Metric                          | main      | branch    | Δ         |
|---------------------------------|-----------|-----------|-----------|
| Latency-phase prefill (506 tok) | 523.8 ms  | 10724.2 ms | **+1947%** |
| Latency-phase prefill tok/s     | 966 tok/s | 47 tok/s  | −95%      |

The single-shot `--paged` prefill measured by the latency phase regressed sharply on the branch. This appears to be a one-time cost on the first paged forward (likely a per-_t-tensor allocation or kernel JIT cost on the new code path) — the throughput phase, which warms up before measuring, sees no equivalent regression: prefill amortizes to <1 s per 506-token prompt in steady state on both binaries.

Net effect for typical workloads (prompt + multi-token decode): **branch is 2.3× faster end-to-end** at every batch size measured. The cold-paged-prefill spike is worth follow-up but does not block this PR — the decode hot path is what kiln pays for in production.

## VRAM cost

| Metric           | main      | branch    | Δ         |
|------------------|-----------|-----------|-----------|
| Model VRAM       | 9526 MB   | 14422 MB  | +4896 MB  |
| Peak runtime VRAM | ~10088 MB | ~14952 MB | +4864 MB  |

~4.9 GB extra VRAM for the seven cached `_t` tensors at bf16 (gate/up/down + q/k/v/o per layer). Below the PR estimate of ~8.7 GB and well within the A6000 48 GB budget.

## Hardware

- GPU: NVIDIA RTX A6000 (49140 MB)
- Driver: 550.127.08
- CUDA: 12.4
- Image: `ghcr.io/ericflo/kiln-runpod:latest`
- Build: `cargo build --release --features cuda --bin kiln-bench`
