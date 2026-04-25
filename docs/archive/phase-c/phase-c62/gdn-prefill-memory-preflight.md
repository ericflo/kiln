# Phase C62: GDN Prefill Memory Preflight

Date: 2026-04-24
Repository: `ericflo/kiln` at `4abc2ca` (`origin/main`)
Task: Phase 7 opener — measure current GDN prefill memory ceiling

## Summary

The required A6000 probes show that current `origin/main` no longer OOMs at 64k prompt length with `KILN_W4A16=1 KILN_KV_CACHE_FP8=1`, but the default CUDA monolithic prefill path is still activation-bound. It reaches 33.3 GiB peak at 64k and 48.6 GiB at 128k, leaving only ~0.6 GiB of headroom on a 49.1 GiB A6000.

The already-shipped opt-in streaming path (`KILN_STREAMING_PREFILL=1`) is the useful Phase 7 capability lever: the same 128k latency probe peaked at 25.4 GiB with essentially unchanged single-run prefill throughput.

## Precondition Check

- `gh pr list -R ericflo/kiln --limit 10` returned no open PRs.
- `PROFILING.md` had Phase 7 streaming design and earlier streaming validation sections, but no 2026-04-24-or-newer A6000 peak-memory table for 32k/64k with both `KILN_W4A16=1` and `KILN_KV_CACHE_FP8=1`.
- `docs/archive/phase-c/phase-c62/gdn-prefill-memory-preflight.md` did not exist before this task.
- Source has changed since the older Phase 7 design: `model_forward_paged_streaming` is already implemented and opt-in on CUDA, so this task measures current default monolithic behavior plus the existing streaming escape hatch rather than designing streaming from scratch.

## RunPod Environment

- Pod pool acquire was attempted first and failed with A6000 supply exhaustion while resuming pool pod `mfk88l8i8tab02`.
- Direct fallback pod: `jc3jwpaps4e8ro`, name `kiln-gdn-prefill-memory-c62`.
- Image: `ghcr.io/ericflo/kiln-runpod:latest`.
- GPU: NVIDIA RTX A6000, 49140 MiB, driver 550.127.08.
- CUDA runtime: 12.4.
- Model path: `/workspace/qwen3.5-4b`.
- Cleanup: pod `jc3jwpaps4e8ro` was explicitly terminated after measurement.

## Commands

`kiln-bench` currently requires `--model-path`, so every required command below adds `--model-path /workspace/qwen3.5-4b`. Peak memory was sampled with `nvidia-smi --query-gpu=timestamp,memory.used --format=csv -lms 250` while each run executed.

```bash
source /root/.kiln-build-env
cd /workspace/kiln
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench
```

Required 32k probe:

```bash
KILN_CUDA_ARCHS=86 \
KILN_W4A16=1 \
KILN_KV_CACHE_FP8=1 \
KILN_CUDA_GRAPHS=true \
KILN_STREAMING_PREFILL=0 \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 32768 --max-output-tokens 1 --skip-training
```

Required 64k probe:

```bash
KILN_CUDA_ARCHS=86 \
KILN_W4A16=1 \
KILN_KV_CACHE_FP8=1 \
KILN_CUDA_GRAPHS=true \
KILN_STREAMING_PREFILL=0 \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 65536 --max-output-tokens 1 --skip-training
```

Exploratory 128k ceiling probe:

```bash
KILN_CUDA_ARCHS=86 \
KILN_W4A16=1 \
KILN_KV_CACHE_FP8=1 \
KILN_CUDA_GRAPHS=true \
KILN_STREAMING_PREFILL=0 \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 131072 --max-output-tokens 1 --skip-training
```

Exploratory 128k streaming probe:

```bash
KILN_CUDA_ARCHS=86 \
KILN_W4A16=1 \
KILN_KV_CACHE_FP8=1 \
KILN_CUDA_GRAPHS=true \
KILN_STREAMING_PREFILL=1 \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 131072 --max-output-tokens 1 --skip-training --latency-only
```

Note: the source-controlled env knob for tile size is `KILN_STREAMING_TILE_TOKENS`; this run used the default 8192-token tile. The measurement wrapper also set `KILN_STREAMING_PREFILL_TILE=8192`, but current code ignores that older spelling.

## Measurements

| Prompt request | Actual prompt | Path | Peak memory | Model-load memory | Delta over load | Prefill time | Prefill tok/s | Exit/status |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 32768 | 32768 | monolithic (`KILN_STREAMING_PREFILL=0`) | 25320 MiB | 17526 MiB | 7794 MiB | 20340.1 ms | 1611 | `0` |
| 65536 | 65533 | monolithic (`KILN_STREAMING_PREFILL=0`) | 33327 MiB | 17526 MiB | 15801 MiB | 25868.8 ms | 2533 | `0` |
| 131072 | 131065 | monolithic (`KILN_STREAMING_PREFILL=0`) | 48562 MiB | 17526 MiB | 31036 MiB | 58397.6 ms | 2244 | latency pass; `143` after manual SIGTERM during throughput sweep |
| 131072 | 131065 | streaming (`KILN_STREAMING_PREFILL=1`) | 25384 MiB | 17526 MiB | 7858 MiB | 58013.7 ms | 2259 | `0` with `--latency-only` |

The 32k and 64k required validation commands completed and exited `0`. `kiln-bench` printed `generation failed` in the later throughput section, but still emitted latency data and returned success; this memory preflight uses the latency prefill/decode segment. The optional 128k monolithic run also completed latency prefill/decode, then was stopped during the unrelated throughput sweep to stay within the 90-minute budget.

## Observations

- 64k is no longer the A6000 failure point on current main with FP8 KV; it completed at 33.3 GiB peak.
- Monolithic prefill memory scales almost linearly with prompt length after model load: +7.8 GiB at 32k, +15.8 GiB at 64k, +31.0 GiB at 128k.
- The practical monolithic ceiling on a 48 GiB A6000 is therefore around 128k. The 128k probe peaked at 48562 MiB against a 49140 MiB device total.
- The streaming path capped 128k peak memory at 25384 MiB, nearly identical to the 32k monolithic peak, which matches the expected “one 8192-token tile plus model” residency pattern.
- Single-run 128k prefill throughput was not worse in the streaming arm: 2259 tok/s streaming vs 2244 tok/s monolithic.
- `KILN_KV_CACHE_FP8=1` did not change the shape of the limiting curve enough to make KV cache the bottleneck; GDN/full-forward activation residency remains the relevant long-context capability target.

## Source-Path Evidence

- `crates/kiln-model/src/forward.rs` defines CUDA streaming prefill as opt-in through `KILN_STREAMING_PREFILL`; without an env override, CUDA defaults to the monolithic path while long Metal prompts opt into streaming.
- `crates/kiln-model/src/forward.rs` implements `model_forward_paged_streaming`, which slices token IDs into tiles and calls `model_forward_paged_inner` per tile while carrying `LinearAttentionState` and paged KV state across tile boundaries.
- `crates/kiln-model/src/forward.rs` runs each GDN layer through `gated_deltanet_forward` with mutable recurrent and convolution state from `LinearAttentionState`, so the state handoff already exists and is O(1) relative to prompt length.
- `crates/kiln-model/src/forward.rs` allocates each GDN recurrent state as `[1, nv, dk, dv]` and conv state as `[1, conv_dim, kernel-1]`; this is small compared with the measured multi-GiB activation slope.
- `crates/kiln-gdn-kernel/src/lib.rs` allocates chunk/full-chunk temporaries such as `[b, h, c, dv]`, `[b, h, c, c]`, and output chunks. Those allocations are chunk-local inside the kernel wrapper, not the persistent O(T) state.
- `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu` and `crates/kiln-gdn-kernel/csrc/gdn_chunk_prep.cu` are chunk kernels; they do not introduce a new prompt-length persistent cache that would explain the 32k→64k→128k scaling.
- The task-listed `crates/kiln-model/src/gdn.rs` path no longer exists on current main; GDN forward/state code lives in `crates/kiln-model/src/forward.rs` and the CUDA wrapper crate.

## Top Tensor Residency Suspects

1. Full-prompt hidden/residual and per-layer GDN intermediates on the monolithic path, because peak delta over model load doubles from 32k to 64k and doubles again from 64k to 128k.
2. GDN projection and q/k normalization intermediates, because they materialize sequence-length-shaped tensors before chunk kernels run.
3. Causal conv1d prefill intermediates, especially F32 promoted buffers, because the conv prefill path is per-GDN-layer and prompt-length-shaped.
4. Full-attention prefill activations in the 8 GQA layers, but FP8 KV and the much smaller layer count make this secondary to the 24 GDN layers for this measurement.
5. LM-head output is no longer the dominant inference prefill spike for this benchmark shape because current paged prefill returns last-token logits rather than a full `[T, vocab]` tensor on the measured path.

## Bounded Implementation Plan

Because streaming prefill already exists and the 128k stream probe confirms the memory reduction, the next implementation slice should be a defaulting and guardrails PR, not a new CUDA kernel port.

1. Add an interleaved A/B bench script for CUDA prefill memory and latency at 32768, 65536, and 131072 tokens with `KILN_STREAMING_PREFILL=0` vs `1` and `KILN_KV_CACHE_FP8=1`.
2. Record peak memory, TTFT, prefill tok/s, and first decode ITL; use `--latency-only` to avoid the unrelated throughput sweep for long prompts.
3. If 32k remains neutral and 64k/128k match this probe, enable CUDA streaming by default above a conservative threshold, recommended first value `65536` tokens.
4. Keep `KILN_STREAMING_PREFILL=0` as a kill switch and document `KILN_STREAMING_TILE_TOKENS` as the actual tile-size override.
5. Add a memory-budget guard that forces streaming when projected monolithic peak would exceed a fixed A6000 safety margin, even if the threshold is not crossed.
6. Defer kernel-level GDN memory work until after default streaming is guarded; this measurement shows tiling removes ~23 GiB at 128k without changing CUDA kernels.

## Validation

- Pod-side `cargo build --release --features cuda --bin kiln-bench`: passed.
- Pod-side 32768 prompt validation with W4A16 + FP8 KV + CUDA graphs: passed, peak 25320 MiB.
- Pod-side 65536 prompt validation with W4A16 + FP8 KV + CUDA graphs: passed, peak 33327 MiB.
- Local `git diff --check`: run before PR creation.
