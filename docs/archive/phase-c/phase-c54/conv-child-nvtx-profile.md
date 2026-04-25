# Phase 6 C54 Conv Child NVTX Profile

Date: 2026-04-24

Commit: `13a2a3d437680d299a1f4a17029cbca8b700701f`

GPU: NVIDIA RTX A6000 (`sl53yvx5seviyx`, lease `pod-c11fe496c432600caf0baa6a`)

## Goal

C54 reruns the C52 C40f-style native-MTP decode profile after C53 added child
NVTX ranges under `:kiln/gdn/conv`. The goal is to split the C52 rank-1 GDN
conv hotspot into layout, fused update, prefill update, and fallback ranges
before choosing the next Phase 6 implementation target.

Preconditions were satisfied on current `origin/main`: `docs/archive/phase-c/phase-c53/summary.json`
is present, no `docs/archive/phase-c/phase-c54` or later profile existed, and no open PR already
reran the C53 child attribution.

## Validation

RunPod pool lease `pod-c11fe496c432600caf0baa6a` used the required
`ghcr.io/ericflo/kiln-runpod:latest` image on an on-demand RTX A6000. The pod
was released after copying artifacts; `runpod_api.py status` reported no active
pods afterward.

Required commands:

```bash
cd /workspace/kiln
git fetch origin main && git reset --hard origin/main
source /root/.kiln-build-env
export KILN_CUDA_ARCHS=86
cargo test -p kiln-conv1d-kernel --release -- --nocapture
cargo test -p kiln-model --release --features cuda test_causal_conv1d_update_matches_fallback -- --nocapture
cargo build --release --features cuda,nvtx --bin kiln-bench
```

Results:

- `kiln-conv1d-kernel`: 3 tests passed in release mode.
- `kiln-model test_causal_conv1d_update_matches_fallback`: 1 test passed with CUDA.
- `kiln-bench`: built successfully with `cuda,nvtx`.

The baked image had Nsight Systems `2023.4.4`; C54 installed only
`nsight-systems-2024.5.1` from the already-configured NVIDIA CUDA apt repo and
used `/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys` for capture and
stats.

## Profile Command

The first literal command from the task brief omitted the currently required
`--model-path` argument and produced an empty no-CUDA/no-NVTX report. The
retained profile reran the same C52 workload with the C52 model path restored:

```bash
mkdir -p docs/archive/phase-c/phase-c54 /tmp/kiln-c54
NSYS=/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys
$NSYS profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-memory-usage=false --output /tmp/kiln-c54/c54-conv-child-ranges \
  env KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_ARGMAX_FP32=1 KILN_W4A16=1 KILN_CUDA_GRAPHS=true \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --prompt-subset humaneval --chat-template --latency-only --temperature 0.0 --seed 1 \
  2>&1 | tee docs/archive/phase-c/phase-c54/nsys-profile.log
$NSYS stats --report nvtx_sum,nvtx_pushpop_trace,cuda_gpu_kern_sum,cuda_api_sum,cuda_kern_exec_sum --format csv --output docs/archive/phase-c/phase-c54/c54 /tmp/kiln-c54/c54-conv-child-ranges.nsys-rep \
  2>&1 | tee docs/archive/phase-c/phase-c54/nsys-stats.log
jq . docs/archive/phase-c/phase-c53/summary.json >/dev/null
```

Nsight generated the retained `.nsys-rep`. A stale SQLite export warning during
stats processing required a stats-only rerun with `--force-export=true`; the
final reports used the 2024.5.1 report names and imported successfully.

The profiled run produced 128 tokens with α `0.693`, decode `22.16 tok/s`, mean
ITL `45.13 ms`, P50 ITL `31.74 ms`, and P99 ITL `105.52 ms`. Treat those as
profiled attribution context, not unprofiled throughput anchors.

## Decode Window Method

The reduction matches C52: use the first `:kiln/mtp/step` start through the
final decode NVTX end, then exclude the `:kiln/mtp/step` parent wrapper from the
NVTX-duration denominator. For C54 that window is `5880.202 ms`; the summed
non-parent NVTX denominator is `4848.439 ms` across 75 MTP steps.

The full `nvtx_pushpop_trace` CSV was used locally for reduction but is not
committed because it is a multi-MB raw trace-like artifact. The committed
summary is `docs/archive/phase-c/phase-c54/conv-child-ranges.csv`.

## Top Decode NVTX Ranges

| Rank | Decode range | Wall-clock share | Total time | Instances |
| ---: | --- | ---: | ---: | ---: |
| 1 | `:kiln/gdn/conv` | 12.45% | 603.799 ms | 2352 |
| 2 | `:kiln/gdn/gates` | 11.47% | 555.891 ms | 2352 |
| 3 | `:kiln/gdn/conv/fallback_prefill` | 11.11% | 538.451 ms | 1800 |
| 4 | `:kiln/gdn/gated_norm` | 10.79% | 523.019 ms | 2352 |
| 5 | `:kiln/gdn/qk_norm` | 8.76% | 424.748 ms | 2352 |

## Conv Child Attribution

| Child range | Decode-window share | Total time | Instances | Share of conv parent |
| --- | ---: | ---: | ---: | ---: |
| `:kiln/gdn/conv/fallback_prefill` | 11.11% | 538.451 ms | 1800 | 89.18% |
| `:kiln/gdn/conv/layout` | 0.90% | 43.672 ms | 2352 | 7.23% |
| `:kiln/gdn/conv/update` | 0.23% | 11.114 ms | 552 | 1.84% |
| `:kiln/gdn/conv/prefill_update` | 0.00% | 0.000 ms | 0 | 0.00% |
| `:kiln/gdn/conv/fallback_decode` | 0.00% | 0.000 ms | 0 | 0.00% |

The dominant child is not the `seq_len == 1` fused update wrapper and not the
layout transpose. It is `fallback_prefill`, which accounts for nearly all of the
remaining conv parent time.

## Fallback Reason

The C53 static audit was correct for single-token decode: CUDA implements
`supports_causal_conv1d_update` and `causal_conv1d_update`, and C54 saw no
material `:kiln/gdn/conv/fallback_decode` time.

The material fallback is a different support-envelope gap. Native MTP decode
invokes a `seq_len > 1` GDN forward for the draft path during each MTP step. The
CUDA backend does not override `supports_causal_conv1d_prefill` or
`causal_conv1d_prefill`, so `forward.rs` routes those draft-path conv calls
through the portable prefill fallback. The 1800 `fallback_prefill` instances are
exactly 24 GDN layers × 75 MTP steps.

## Recommendation

Next target: add a minimal CUDA `causal_conv1d_prefill` path for the native-MTP
draft GDN conv shape before moving to `:kiln/gdn/gates` +
`:kiln/gdn/gated_norm`.

Suggested files:

- `crates/kiln-conv1d-kernel/src/lib.rs`
- `crates/kiln-conv1d-kernel/csrc/causal_conv1d_update.cu`
- `crates/kiln-conv1d-kernel/csrc/causal_conv1d_update.h`
- `crates/kiln-model/src/backend/cuda.rs`
- `crates/kiln-model/src/forward.rs`

Scope guard: this is not a retry of the existing `seq_len == 1` update kernel.
If preflight shows the minimal CUDA prefill support envelope cannot cover bf16
`[B,C,T]` with F32 state and `kernel_size == 4` for the MTP draft shape, stop
and retarget to the adjacent gates/gated-norm cluster.
