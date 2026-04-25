# Phase C36 — post-#417 current-main identity-bias attribution

> **Note on artifact links below:** Raw nvtx CSVs and bench logs were
> removed from the working tree on 2026-04-25 to clean up the top-level
> directory. The exact files cited in this report are preserved in git
> history at commit `07993f0272b0ef229246e4f3cc979abb7b05e145` (origin/main immediately before the
> cleanup PR). To inspect any referenced artifact, run:
> `git show 07993f0272b0ef229246e4f3cc979abb7b05e145:<path>` (e.g. `profile-out/...`, `c12-out/...`,
> `profiling-artifacts/...`).


**Date:** 2026-04-23  
**Hardware:** RunPod RTX A6000, image `ghcr.io/ericflo/kiln-runpod:latest`  
**Branch:** `ce/phase-c36-identity-bias`  
**Current main baseline:** PR #417 (`d5c57ed`) reported native MTP at `27.67 tok/s`,
`α=0.245` on the standard `--paged --prompt-tokens 512 --max-output-tokens 128 --skip-training`
workload.

## Preflight

Proceed conditions from the task brief:

1. **No newer PR already landed this exact artifact:** satisfied. Fresh repo +
   `gh pr list --repo ericflo/kiln --state all` showed PR #417 as the latest
   native-MTP current-main benchmark PR, and neither repo search nor recent PR
   history contained a post-#417 `c1_identity` artifact or a Phase C36
   identity-bias write-up for the same workload.
2. **Native MTP still below the ship bar on current main:** satisfied. The
   fresh rerun below remains well under `α>=0.72`, with per-seed decode tok/s
   far below the old `off` median from PR #417 (`53.87 tok/s`).

## Commands run

Build and setup:

```bash
export RUNPOD_API_KEY=$(ce secret-get --name RUNPOD_API_KEY)
RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py
POD_ID=$(python3 $RP launch --kiln-image --gpu "NVIDIA RTX A6000" --name kiln-c36-identity-bias --disk-gb 80 | jq -r .id)
python3 $RP wait $POD_ID
B2_KEY_ID=$(ce secret-get --name B2_APPLICATION_KEY_ID)
B2_KEY=$(ce secret-get --name B2_APPLICATION_KEY)
python3 $RP ssh $POD_ID "export B2_APPLICATION_KEY_ID=\"$B2_KEY_ID\" B2_APPLICATION_KEY=\"$B2_KEY\"; kiln-setup --clone"
python3 $RP ssh $POD_ID "cd /workspace/kiln && git fetch origin ce/phase-c36-identity-bias && git checkout ce/phase-c36-identity-bias"
python3 $RP bg $POD_ID /tmp/build.log 'source /root/.kiln-build-env && export KILN_CUDA_ARCHS=86 && cd /workspace/kiln && cargo build --release --features cuda --bin kiln-bench'
python3 $RP wait-file $POD_ID /workspace/kiln/target/release/kiln-bench --timeout 3600
```

Attribution rerun:

```bash
python3 $RP ssh $POD_ID '
  source /root/.kiln-build-env
  cd /workspace/kiln
  hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b
  mkdir -p docs/archive/phase-c/phase-c36 profiling-artifacts
  for seed in 0 1 2; do
    KILN_SPEC_METHOD=mtp \
    KILN_BENCH_FORCE_MTP=1 \
    KILN_C1_ATTR_PATH=/workspace/kiln/docs/archive/phase-c/phase-c36/c1_seed${seed}.csv \
    ./target/release/kiln-bench \
      --model-path /workspace/qwen3.5-4b \
      --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
      --seed $seed \
      > /workspace/kiln/profiling-artifacts/post417_20260423_c1_seed${seed}.json
  done
  python3 scripts/mtp_c1_summarize.py \
    --json profiling-artifacts/post417_20260423_c1_identity.json \
    docs/archive/phase-c/phase-c36/c1_seed0.csv docs/archive/phase-c/phase-c36/c1_seed1.csv docs/archive/phase-c/phase-c36/c1_seed2.csv \
    > docs/archive/phase-c/phase-c36/c36-identity-bias.md
'
```

## Fresh current-main results

### Per-seed native MTP bench

| Seed | prompt tokens | decode tok/s | α | C1 rows | identity rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 494 | 36.61 | 0.7397 | 73 | 10.96% |
| 1 | 508 | 20.00 | 0.3093 | 97 | 59.79% |
| 2 | 501 | 21.95 | 0.3956 | 91 | 52.75% |

Median decode tok/s across the three fresh runs is **21.95 tok/s**. Median
α is **0.3956**. That keeps native MTP well below the ship bar on the
standard workload.

### Aggregate C1 identity-bias summary

Source: [`profiling-artifacts/post417_20260423_c1_identity.json`](../../profiling-artifacts/post417_20260423_c1_identity.json)

- total attribution rows: **261**
- aggregate α / top-k match rate: **45.98%** (`120 / 261`)
- Class A rows: **0**
- Class B rows: **141**
- overall `draft_equals_last_token_rate`: **43.68%** (`114 / 261`)
- reject-conditioned identity-bias rate: **80.85%** (`114 / 141`)
- accept-conditioned identity-bias rate: **0.00%** (`0 / 120`)

### Per-`mtp_pos` shape

The new summary script also emits identity counts/rates per `mtp_pos` bucket.
The strongest concentration remains early in decode:

- `seed0`: bucket `mtp_pos 0..6` carries all 8 identity-biased rows
  (`42.11%` of that bucket); later buckets are clean.
- `seed1`: buckets `0..11` sit at `64%` to `72%` identity bias, with another
  hot bucket at `16..19` (`70%`).
- `seed2`: buckets `0..19` mostly sit between `50%` and `65%`, then cool off
  in the middle before spiking again at `30..39`.

## Verdict

**Yes, identity bias still dominates the low-α failure mode on the standard
current-main workload, but it no longer dominates every seed uniformly.**
Across the fresh rerun, **80.85% of all reject rows are identity-biased**
(`draft_top1 == last_token`), while **0% of accepted rows** are. That keeps
identity bias as the main reject-mode signature and justifies making it a
first-class committed metric. At the same time, the seed split matters: one
seed (`α=0.7397`) nearly clears the paper floor with only `10.96%` identity
bias, while the other two seeds stay in the low-α regime with `~53%` to `60%`
overall identity bias. So the right next step is not another decode-kernel
task; it is a tighter MTP-forward/root-cause bisect that explains why the same
standard workload sometimes exits the identity-biased regime and sometimes
stays trapped in it.

## Summarizer stdout

This file was originally generated from the summarizer stdout; the full
human-readable summary remains reproducible with:

```bash
python3 scripts/mtp_c1_summarize.py \
  --json profiling-artifacts/post417_20260423_c1_identity.json \
  docs/archive/phase-c/phase-c36/c1_seed0.csv docs/archive/phase-c/phase-c36/c1_seed1.csv docs/archive/phase-c/phase-c36/c1_seed2.csv
```
