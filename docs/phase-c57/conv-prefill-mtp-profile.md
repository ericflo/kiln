# Phase 6 C57 Conv1d Prefill MTP Profile Recovery

Date: 2026-04-24

Commit: `b63e241` plus this PR's test-only regression change

GPU: NVIDIA RTX A6000 (`6lv4pu241ofanf`)

## Goal

C57 verifies that the CUDA `causal_conv1d_prefill` fast path no longer blocks
the C56 native-MTP profile workload, and refreshes the native-MTP decode source
of truth after the status-3 prefill failure documented in `docs/phase-c56/`.

## Precondition

Current `main` had no newer artifact than C56 that both used the CUDA
causal-conv1d prefill fast path and reached native-MTP decode
(`:kiln/mtp/step`). `PROFILING.md` had a newer post-#486 plain decode/prefill
profile, but that was not a native-MTP decode-window profile.

Current `main` already includes PR #481 (`7227b4c`), which fixed the C56 root
cause by matching the prefill kernel `__launch_bounds__` to the largest
256-thread launch path. This PR therefore adds a focused model-level regression
coverage increase for the C56 prefill envelope and records the recovered C57
profile.

## Validation

RunPod pod `6lv4pu241ofanf` used the required `ghcr.io/ericflo/kiln-runpod:latest`
image on an on-demand RTX A6000. The pod reported driver `550.127.08`, CUDA
12.4, and 49140 MiB VRAM.

Commands run from `/workspace/kiln` with `source /root/.kiln-build-env` and
`KILN_CUDA_ARCHS=86`:

```bash
cargo test -p kiln-conv1d-kernel --release -- --nocapture
cargo test -p kiln-model --release --features cuda causal_conv1d_prefill -- --nocapture
cargo test -p kiln-model --release --features cuda test_causal_conv1d_update_matches_fallback -- --nocapture
cargo build --release --features cuda,nvtx --bin kiln-bench
```

Results:

- `kiln-conv1d-kernel`: 4 tests passed.
- `causal_conv1d_prefill`: 1 CUDA model test passed at the C56 512-token prefill envelope, max abs diff `1.4901161e-8`, state parity `0`.
- `test_causal_conv1d_update_matches_fallback`: 1 CUDA model test passed, max abs diff `7.450581e-9`, state parity `0`.
- `kiln-bench`: built successfully with `cuda,nvtx`.

## Profile Command

The task brief's `--latency-paged` / `--max-new-tokens` spelling is not present
in the current bench CLI, so C57 used the existing C52/C54/C56 native-MTP
workload spelling:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none \
  --cpuctxsw=none --cuda-memory-usage=false \
  --output /tmp/kiln-c57/c57-mtp-conv-prefill-v2 \
  env KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_ARGMAX_FP32=1 KILN_W4A16=1 KILN_CUDA_GRAPHS=true \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged \
    --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --prompt-subset humaneval --chat-template --latency-only --temperature 0.0 --seed 1
nsys stats --force-export=true \
  --report nvtx_sum,nvtx_pushpop_trace,cuda_gpu_kern_sum,cuda_api_sum,cuda_kern_exec_sum \
  --format csv --output docs/phase-c57/c57-mtp-conv-prefill-v2 \
  /tmp/kiln-c57/c57-mtp-conv-prefill-v2.nsys-rep
```

The first attempt with `--trace=cuda,nvtx,osrt` hit a Nsight 2023.4.4 importer
wrong-event-order error after the benchmark had completed successfully. The
retained run removed `osrt`; it generated a valid `.nsys-rep` and CSV exports.

## Result

C57 reached native-MTP decode. The profile contains `:kiln/mtp/step` ranges and
no `kiln_causal_conv1d_prefill_bf16_f32` status-3 failure.

Latency metrics from `nsys-profile-v2.log`:

- Prompt tokens: 515
- Prefill: 417.7 ms (1233 tok/s)
- Generated tokens: 128
- Decode: 26.5 tok/s
- Mean ITL: 37.8 ms
- MTP alpha: 0.764

## Decode Window Method

The reduction follows C52/C54: first `:kiln/mtp/step` start through final decode
NVTX end, excluding the `:kiln/mtp/step` parent wrapper from the denominator.
C57's decode window is 4737.861 ms and the summed non-parent NVTX denominator is
3383.625 ms.

## Top Decode NVTX Ranges

Source: `top_decode_nvtx.csv`.

| Rank | Decode range | Wall-clock share | Total time | Instances |
| ---: | --- | ---: | ---: | ---: |
| 1 | `:kiln/gdn/gates` | 14.45% | 489.003 ms | 2088 |
| 2 | `:kiln/gdn/gated_norm` | 13.55% | 458.578 ms | 2088 |
| 3 | `:kiln/gdn/qk_norm` | 11.02% | 372.938 ms | 2088 |
| 4 | `:kiln/attn/rope` | 8.56% | 289.762 ms | 768 |
| 5 | `:kiln/mlp/gate` | 4.70% | 159.151 ms | 2856 |

## Top Prefill NVTX Ranges

Source: `top_prefill_nvtx.csv`.

| Rank | Prefill range | Wall-clock share | Total time | Instances |
| ---: | --- | ---: | ---: | ---: |
| 1 | `:kiln/gdn/in_proj` | 59.49% | 172.210 ms | 24 |
| 2 | `:kiln/attn/full/prefill_initial` | 12.67% | 36.662 ms | 8 |
| 3 | `:kiln/gdn/qk_norm` | 4.47% | 12.941 ms | 24 |
| 4 | `:kiln/gdn/gates` | 3.37% | 9.751 ms | 24 |
| 5 | `:kiln/mlp/down` | 2.52% | 7.286 ms | 32 |

## Artifacts

Committed reduced artifacts:

- `summary.json`
- `top_decode_nvtx.csv`
- `top_prefill_nvtx.csv`
- `c57-mtp-conv-prefill-v2_cuda_gpu_kern_sum.csv`
- `nsys-profile-v2.log`
- `nsys-stats-v2.log`

The raw `.nsys-rep`, `.sqlite`, and full `nvtx_pushpop_trace` export were not
committed.

## Recommendation

The C56 blocker is cleared on current `main`; Phase 6 can again use a native-MTP
decode profile as source of truth. The next material native-MTP decode targets
are `:kiln/gdn/gates`, `:kiln/gdn/gated_norm`, and `:kiln/gdn/qk_norm`; do not
select a new optimization from the failed C56 artifact.
