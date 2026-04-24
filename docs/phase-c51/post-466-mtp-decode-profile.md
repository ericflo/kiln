# Phase C51 Post-#466 C40f-Style Native MTP Decode Profile

Date: 2026-04-24

Commit: `831d879d2ddb2500fd49f76c9b5df89aedd923b1`

GPU: RunPod on-demand NVIDIA RTX A6000, `ghcr.io/ericflo/kiln-runpod:latest`.

## Purpose

PR #466 landed the fused CUDA GDN gated RMSNorm kernel. This C51 artifact re-runs the C40f-style native-MTP decode workload on current `origin/main` so the next Phase 6 optimization decision does not rely on the stale post-#442/C50 decode hotspot table.

## Environment

Captured in `docs/phase-c51/environment.txt`:

| Field | Value |
| --- | --- |
| GPU | NVIDIA RTX A6000 |
| Driver | 550.127.08 |
| CUDA | Build cuda_12.4.r12.4/compiler.34097967_0 |
| Rust | rustc 1.95.0 (59807616e 2026-04-14) |
| Nsight Systems | NVIDIA Nsight Systems version 2023.4.4.54-234433681190v0 |
| Build arch | `KILN_CUDA_ARCHS=86` |

## Setup Commands

```bash
export RUNPOD_API_KEY=$(ce secret-get --name RUNPOD_API_KEY)
RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py
python3 $RP launch --kiln-image --gpu "NVIDIA RTX A6000" \
  --name kiln-post466-mtp-profile --disk-gb 80
python3 $RP wait eyu0zh4ilfa2kh

B2_KEY_ID=$(ce secret-get --name B2_APPLICATION_KEY_ID)
B2_KEY=$(ce secret-get --name B2_APPLICATION_KEY)
python3 $RP ssh eyu0zh4ilfa2kh \
  "export B2_APPLICATION_KEY_ID='$B2_KEY_ID' B2_APPLICATION_KEY='$B2_KEY'; kiln-setup --clone"

python3 $RP bg eyu0zh4ilfa2kh /tmp/build.log \
  'source /root/.kiln-build-env && cd /workspace/kiln && git fetch origin main && git checkout -B ce/post466-mtp-profile origin/main && KILN_CUDA_ARCHS=86 cargo build --release --features cuda,nvtx --bin kiln-bench'
python3 $RP wait-file eyu0zh4ilfa2kh /workspace/kiln/target/release/kiln-bench --timeout 3600
```

## Benchmark Command

```bash
env KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_ARGMAX_FP32=1 \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
  --seed <seed> --chat-template --latency-only --prompt-subset humaneval \
  --temperature 0.0
```

## Raw Benchmark Table

| Seed | Prompt tokens | α | Decode tok/s | Mean ITL ms | P50 ITL ms | P99 ITL ms | Prefill ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 515 | 0.707 | 41.41 | 24.15 | 17.44 | 55.94 | 10162.14 |
| 2 | 510 | 0.588 | 37.07 | 26.97 | 18.13 | 59.26 | 369.99 |
| 3 | 494 | 0.568 | 32.06 | 31.20 | 20.00 | 67.65 | 375.73 |

| Metric | Median | Mean |
| --- | ---: | ---: |
| Acceptance α | 0.588 | 0.621 |
| Decode tok/s | 37.07 | 36.85 |
| Mean ITL ms | 26.97 | 27.44 |
| P50 ITL ms | 18.13 | 18.53 |
| P99 ITL ms | 59.26 | 60.95 |
| Prefill ms | 375.73 | 3635.96 |

Seed 1 includes a cold prefill outlier from the first benchmark process after build/model setup; decode metrics are the artifact of interest for this task.

## Profiler Command

```bash
env KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_ARGMAX_FP32=1 \
  nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
  --output /tmp/kiln-post466-mtp \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
  --seed 1 --chat-template --latency-only --prompt-subset humaneval \
  --temperature 0.0

nsys stats --report nvtxsum /tmp/kiln-post466-mtp.nsys-rep | tee docs/phase-c51/nsys-nvtxsum.txt || true
nsys stats --report gpukernsum /tmp/kiln-post466-mtp.nsys-rep | tee docs/phase-c51/nsys-gpu-kernsum.txt || true
nsys stats --report cudaapisum /tmp/kiln-post466-mtp.nsys-rep | tee docs/phase-c51/nsys-cuda-api.txt || true
```

## Profiler Summary

The `nsys profile` process exited `0` and generated `/tmp/kiln-post466-mtp.qdstrm`, but the Nsight Systems 2023.4.4 importer failed before producing `/tmp/kiln-post466-mtp.nsys-rep`. The committed stats files therefore contain the expected missing-input error rather than decode attribution.

Failure excerpt from `docs/phase-c51/nsys-profile.log`:

```text
Importer error status: Importation failed.
Import Failed with unexpected exception: ... QuadDCommon::RuntimeException
ErrorText = "Wrong event order has been detected when adding events to the collection"
Generated:
    /tmp/kiln-post466-mtp.qdstrm
```

Raw profiler breadcrumb from `docs/phase-c51/nsys-raw-files.txt`:

```text
-rw-rw-r-- 1 root root 94M Apr 24 03:20 /tmp/kiln-post466-mtp.qdstrm
```

The `.qdstrm` file is intentionally not committed because it is a multi-MB raw profiler artifact and the importer failure makes it unusable by `nsys stats` in this environment.

## Top Decode Hotspots

No fresh post-#466 wall-clock percentages are available from this run. The top-three decode range fields in `docs/phase-c51/summary.json` are intentionally `null` rather than carrying stale percentages forward as if they were current.

For context only, C50 carried forward the latest successful current-main decode NVTX attribution from the post-#442 profile: `:kiln/gdn/gates` `17.9%`, `:kiln/gdn/gated_norm` `17.3%`, and `:kiln/gdn/qk_norm` `15.0%`. Those percentages predate the PR #466 fused gated RMSNorm change and should be treated as stale until a profiler capture imports successfully.

## Recommendation

Do not start a new kernel implementation from invented post-#466 percentages. The single next implementation target is conditional: continue with the GDN gate/gated-norm cluster only if a successful post-#466 profiler capture confirms it remains dominant. Otherwise, first repair or bypass the Nsight importer failure by using a compatible newer `nsys`, a smaller decode-window capture, or another profiler path that can produce current NVTX/kernel attribution.

This artifact replaces C50 as the benchmark anchor for post-#466 C40f-style native MTP decode medians, but it does not replace C50 with fresh hotspot percentages because the importer failed again.
