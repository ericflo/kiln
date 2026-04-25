# Phase C63: CUDA streaming prefill default

Date: 2026-04-24
Branch: `c63-cuda-streaming-prefill-default`
Current-main base: `76c7614`
Hardware: RunPod on-demand `NVIDIA A40` fallback, 48 GiB VRAM, sm_86
Image: `ghcr.io/ericflo/kiln-runpod:latest`

## Decision

Enable CUDA streaming prefill by default at `>= 65533` actual prompt tokens while preserving `KILN_STREAMING_PREFILL=0` as a monolithic-path kill switch and `KILN_STREAMING_PREFILL=1` as a force-on override below the threshold.

The A6000 requested by the task was supply-constrained. The run used the documented A40 fallback: same 48 GiB class and sm_86 target, with lower absolute throughput than A6000 but enough to validate memory and regression direction. The decision threshold follows the task rule: 65k streaming was within noise/slightly faster, 128k monolithic failed with CUDA OOM, and 128k streaming completed.

## Commands

```bash
export RUNPOD_API_KEY=$(ce secret-get --name RUNPOD_API_KEY)
RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py
python3 $RP status
python3 $RP launch --kiln-image --gpu "NVIDIA RTX A6000" \
  --name kiln-cuda-streaming-prefill-c63 --disk-gb 80
# A6000 was supply-constrained, so fallback used:
python3 $RP launch --kiln-image --gpu "NVIDIA A40" \
  --name kiln-cuda-streaming-prefill-c63-a40 --disk-gb 80
python3 $RP wait 4jnrcnwyn4btpl
B2_KEY_ID=$(ce secret-get --name B2_APPLICATION_KEY_ID)
B2_KEY=$(ce secret-get --name B2_APPLICATION_KEY)
python3 $RP ssh 4jnrcnwyn4btpl \
  "export B2_APPLICATION_KEY_ID='$B2_KEY_ID' B2_APPLICATION_KEY='$B2_KEY'; kiln-setup --clone"
python3 $RP ssh 4jnrcnwyn4btpl \
  "export B2_APPLICATION_KEY_ID='$B2_KEY_ID' B2_APPLICATION_KEY='$B2_KEY'; kiln-setup --repo /workspace/kiln"
python3 $RP bg 4jnrcnwyn4btpl /tmp/c63_build.log \
  'source /root/.kiln-build-env && cd /workspace/kiln && KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench'
python3 $RP wait-file 4jnrcnwyn4btpl /workspace/kiln/target/release/kiln-bench --timeout 3600
```

The interleaved matrix used `--latency-only` and sampled peak memory with `nvidia-smi` every 200 ms:

```bash
source /root/.kiln-build-env
cd /workspace/kiln
for tokens in 32768 65536 131072; do
  for stream in 0 1; do
    KILN_W4A16=1 KILN_KV_CACHE_FP8=1 KILN_CUDA_GRAPHS=true KILN_STREAMING_PREFILL=$stream \
      ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged \
      --prompt-tokens $tokens --max-output-tokens 1 --skip-training --latency-only \
      2>&1 | tee /tmp/c63_${tokens}_stream${stream}.log
  done
done
```

## Results

| Prompt tokens | Actual tokens | Mode | Exit | Peak VRAM | Prefill / TTFT | Prefill tok/s | First decode ITL |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32768 | 32768 | monolithic (`KILN_STREAMING_PREFILL=0`) | 0 | 25319 MiB | 20075.7 ms | 1632 | 102.3 ms |
| 32768 | 32768 | streaming (`KILN_STREAMING_PREFILL=1`) | 0 | 20071 MiB | 12881.7 ms | 2544 | 101.1 ms |
| 65536 | 65533 | monolithic (`KILN_STREAMING_PREFILL=0`) | 0 | 33063 MiB | 28482.0 ms | 2301 | 169.3 ms |
| 65536 | 65533 | streaming (`KILN_STREAMING_PREFILL=1`) | 0 | 21863 MiB | 27693.8 ms | 2366 | 170.7 ms |
| 131072 | 131065 | monolithic (`KILN_STREAMING_PREFILL=0`) | 1 | 44455 MiB sampled before OOM | OOM in GDN layer 0 | n/a | n/a |
| 131072 | 131065 | streaming (`KILN_STREAMING_PREFILL=1`) | 0 | 25383 MiB | 62930.8 ms | 2083 | 310.6 ms |

Notes:

- The raw summary script accidentally included timestamp rows in its first peak calculation; the table above uses the corrected `nvidia-smi` memory rows only.
- 65k streaming reduced sampled peak memory by 11200 MiB and improved prefill by 2.8% on this single interleaved run.
- 128k monolithic OOMed on the A40 fallback, while streaming completed with a peak near the prior PR #507 A6000 128k streaming result.
- Decode ITL was unchanged within measurement noise at 32k and 65k; 128k has no monolithic decode comparator because the arm OOMed before decode.

## Threshold rationale

The default threshold is `65533` actual prompt tokens, which is the bench harness's realized prompt length for the requested `--prompt-tokens 65536` case. It is conservative enough to leave shorter prompts on their historical path unless explicitly forced, while covering the first prompt size where PR #507 showed monolithic peak growth becoming material. The C63 A/B did not show a material 65k latency or first-decode regression; it showed a small prefill win and a large memory win. At 128k, streaming is the only successful mode on the fallback 48 GiB GPU.

## Cleanup

Pod `4jnrcnwyn4btpl` (`kiln-cuda-streaming-prefill-c63-a40`) was explicitly terminated after validation.
